"""Offline comparison of synchronized G1 logs. No robot or policy imports."""

from __future__ import annotations

import base64
import html
import io
import json
from pathlib import Path

import msgpack
import numpy as np
from scipy.spatial.transform import Rotation


def _read(path, warnings):
    events, consumed = [], 0
    with path.open("rb") as stream:
        unpacker = msgpack.Unpacker(stream, raw=False)
        for event in unpacker:
            if not isinstance(event, dict) or not {"kind", "payload", "monotonic_time_ns"} <= event.keys():
                raise ValueError(f"{path.name}: invalid event")
            events.append(event)
            consumed = unpacker.tell()
    if consumed != path.stat().st_size:
        warnings.append(f"{path.name}: truncated final record")
    if not events or events[0]["kind"] != "header" or events[0]["payload"].get("schema_version") != 1:
        raise ValueError(f"{path.name}: expected schema version 1 header")
    if any(a["monotonic_time_ns"] > b["monotonic_time_ns"] for a, b in zip(events, events[1:], strict=False)):
        raise ValueError(f"{path.name}: monotonic clock moved backwards")
    summaries = [e["payload"] for e in events if e["kind"] == "summary"]
    if not summaries:
        warnings.append(f"{path.name}: missing completion summary")
    elif summaries[-1].get("dropped", 0) or summaries[-1].get("error") or summaries[-1].get("complete") is False:
        warnings.append(f"{path.name}: incomplete recording: {summaries[-1]}")
    return events


def load_run(directory):
    directory = Path(directory)
    warnings = []
    state = _read(directory / "state.msgpack", warnings)
    motion = _read(directory / "motion.msgpack", warnings)
    sm, mm = (events[0]["payload"].get("metadata", {}) for events in (state, motion))
    if not sm.get("run_id") or sm.get("run_id") != mm.get("run_id"):
        raise ValueError("state/motion run IDs must exist and match")
    names = mm.get("environment_joint_names", [])
    mapping = mm.get("joint_to_hardware_motor", [])
    if len(names) != len(set(names)) or len(mapping) != len(names) or not names:
        raise ValueError("missing/ambiguous environment joint mapping")
    if any(not isinstance(i, int) or i < 0 for i in mapping):
        raise ValueError("hardware motor mapping contains unknown indices")
    return dict(
        directory=directory,
        state=state,
        motion=motion,
        metadata=mm,
        warnings=warnings,
        segments={e["payload"]["segment"]: e["payload"] for e in motion if e["kind"] == "segment"},
    )


def select_segment(run, segment, reference_fps, start, end, context=2.0):
    if (
        not np.isfinite([reference_fps, start, end, context]).all()
        or reference_fps <= 0
        or start < 0
        or end <= start
        or context < 0
    ):
        raise ValueError("require fps>0, 0<=start<end and context>=0")
    cycles = [
        e
        for e in run["motion"]
        if e["kind"] == "cycle"
        and e["payload"].get("phase") in ("playing", "paused", "done")
        and isinstance(e["payload"].get("inference"), dict)
        and e["payload"]["inference"].get("query_frame") is not None
    ]
    available = sorted({e["payload"]["segment"] for e in cycles})
    if segment is None:
        if len(available) != 1:
            raise ValueError(f"select --segment from formal playback segments: {available}")
        segment = available[0]
    cycles = [e for e in cycles if e["payload"]["segment"] == segment]
    if not cycles or segment not in run["segments"]:
        raise ValueError(f"no formal playback in segment {segment}; available={available}")
    times = np.array([e["payload"]["inference"]["query_frame"] / reference_fps for e in cycles])
    if not np.isfinite(times).all() or np.any(np.diff(times) < 0):
        raise ValueError("reference frame moves backwards within a segment")
    if not np.any((times >= start) & (times <= end)):
        raise ValueError("requested core interval has no recorded cycles")
    selected = [e for e, t in zip(cycles, times, strict=True) if max(0.0, start - context) <= t <= end + context]
    return dict(
        segment=segment,
        info=run["segments"][segment],
        cycles=selected,
        origin_ns=cycles[0]["monotonic_time_ns"],
        fps=reference_fps,
        start=start,
        end=end,
        context=context,
    )


def _vector(value, width):
    try:
        array = np.asarray(value, dtype=float).reshape(-1)
        return array if len(array) == width else np.full(width, np.nan)
    except (ValueError, TypeError):
        return np.full(width, np.nan)


def _sample(events, grid, getter, width, max_gap_ns, *, hold=False, quaternion=False):
    """Interpolate bounded brackets; commands use causal zero-order hold."""
    out = np.full((len(grid), width), np.nan)
    ages = np.full(len(grid), np.nan)
    gaps = np.full(len(grid), np.nan)
    if not events:
        return out, ages, gaps
    times = np.array([e["monotonic_time_ns"] for e in events], dtype=np.int64)
    # Keep the last event at an identical receive timestamp.
    keep = np.r_[np.diff(times) != 0, True]
    events = [e for e, k in zip(events, keep, strict=True) if k]
    times = times[keep]
    values = np.array([_vector(getter(e["payload"]), width) for e in events])
    right = np.searchsorted(times, grid, side="left")
    for i, t in enumerate(grid):
        j = right[i]
        if j < len(times) and times[j] == t:
            out[i] = values[j]
            ages[i] = gaps[i] = 0
        elif hold:
            k = j - 1
            if k >= 0:
                ages[i] = (t - times[k]) / 1e6
                if t - times[k] <= max_gap_ns:
                    out[i] = values[k]
        elif 0 < j < len(times):
            gap = times[j] - times[j - 1]
            ages[i] = min(t - times[j - 1], times[j] - t) / 1e6
            gaps[i] = gap / 1e6
            if gap <= max_gap_ns:
                alpha = (t - times[j - 1]) / gap
                if quaternion:
                    pair = values[j - 1 : j + 1]
                    if np.isfinite(pair).all() and np.all(np.linalg.norm(pair, axis=1) > 1e-8):
                        rotations = Rotation.from_quat(pair[:, [1, 2, 3, 0]])
                        rotation = rotations[0] * Rotation.from_rotvec(
                            (rotations[0].inv() * rotations[1]).as_rotvec() * alpha
                        )
                        out[i] = rotation.as_quat()[[3, 0, 1, 2]]
                else:
                    out[i] = (1 - alpha) * values[j - 1] + alpha * values[j]
    if quaternion:
        norms = np.linalg.norm(out, axis=1)
        valid = np.isfinite(out).all(axis=1) & (norms > 1e-8)
        out[~valid] = np.nan
        out[valid] /= norms[valid, None]
    return out, ages, gaps


def _reference_bodies(selection, model_file, warnings):
    info = selection["info"]
    names = info["policy_joint_names"]
    path = Path(model_file or info.get("policy_file", ""))
    if not path.is_file():
        warnings.append(
            "Reference body names unavailable: supply --model-file with the deployed ONNX; no body index was guessed."
        )
        return {}
    try:
        import onnx

        metadata = {v.key: v.value for v in onnx.load(str(path), load_external_data=False).metadata_props}
        if metadata.get("joint_names", "").split(",") != names:
            raise ValueError("ONNX joint order differs from recorded policy joint order")
        body_names = metadata["body_names"].split(",")
    except (ImportError, KeyError, ValueError, OSError) as exc:
        warnings.append(f"Reference body metadata unavailable: {exc}")
        return {}
    output = {}
    for name, label in (("pelvis", "base"), ("torso_link", "torso")):
        if name not in body_names:
            warnings.append(f"Reference metadata has no {name}")
            continue
        quats = []
        for event in selection["cycles"]:
            value = event["payload"]["inference"].get("output_reference", {}).get("body_quat_w")
            array = np.asarray(value, dtype=float).squeeze()
            quats.append(array[body_names.index(name)] if array.shape == (len(body_names), 4) else np.full(4, np.nan))
        output[f"reference_{label}_quat"] = np.array(quats)
    return output


def align_signals(run, selection, max_gap_ms=100.0, model_file=None):
    if not np.isfinite(max_gap_ms) or max_gap_ms <= 0:
        raise ValueError("max-gap-ms must be positive")
    cycles = selection["cycles"]
    info = selection["info"]
    names = list(info.get("policy_joint_names", []))
    env_names = run["metadata"]["environment_joint_names"]
    if not names or len(set(names)) != len(names) or any(name not in env_names for name in names):
        raise ValueError("policy joint names cannot be mapped to environment joints")
    env_order = [env_names.index(name) for name in names]
    motors = np.array(run["metadata"]["joint_to_hardware_motor"])[env_order]
    motor_width = int(max(motors)) + 1
    states = [e for e in run["state"] if e["kind"] == "lowstate"]
    commands = [e for e in run["state"] if e["kind"] == "lowcmd"]
    torso = [e for e in run["state"] if e["kind"] == "torso_imu"]

    def motor(payload, field):
        raw = np.asarray(payload.get("motor", {}).get(field, []), dtype=float).reshape(-1)
        return raw[motors] if len(raw) >= motor_width else None

    grid = np.array([e["monotonic_time_ns"] for e in cycles], dtype=np.int64)
    n = len(names)
    gap = max_gap_ms * 1e6
    data = dict(
        monotonic_time_ns=grid,
        time_s=(grid - selection["origin_ns"]) / 1e9,
        motion_time_s=np.array([e["payload"]["inference"]["query_frame"] / selection["fps"] for e in cycles]),
        joint_names=np.array(names),
        cycle=np.array([e["payload"]["cycle"] for e in cycles]),
        phase=np.array([e["payload"]["phase"] for e in cycles]),
        query_frame=np.array([e["payload"]["inference"]["query_frame"] for e in cycles]),
    )
    for key, field in (("actual_q", "q"), ("actual_dq", "dq"), ("tau_est", "tau_est")):
        data[key], data[f"{key}_age_ms"], data[f"{key}_bracket_ms"] = _sample(
            states, grid, lambda p, f=field: motor(p, f), n, gap
        )
    data["dds_q"], data["dds_q_age_ms"], data["dds_q_bracket_ms"] = _sample(
        commands, grid, lambda p: motor(p, "q"), n, gap, hold=True
    )
    for key in ("reference_q", "reference_dq", "observation_reference_q", "policy_q", "final_q"):
        values = []
        for e in cycles:
            payload = e["payload"]
            inference = payload["inference"]
            if key == "final_q":
                value = _vector(payload.get("final_target"), len(env_names))[env_order]
            elif key == "policy_q":
                value = _vector(inference.get("policy_target"), n)
            elif key == "observation_reference_q":
                value = _vector((inference.get("observation_reference") or {}).get("joint_pos"), n)
            else:
                value = _vector(
                    inference.get("output_reference", {}).get("joint_pos" if key == "reference_q" else "joint_vel"), n
                )
            values.append(value)
        data[key] = np.array(values)
    results = {e["payload"].get("cycle"): e["payload"] for e in run["motion"] if e["kind"] == "send_result"}
    data["send_status"] = np.array([results.get(e["payload"]["cycle"], {}).get("status", "missing") for e in cycles])
    for label, events, get_imu in (("base", states, lambda p: p.get("base_imu", {})), ("torso", torso, lambda p: p)):
        for field, width in (("quaternion", 4), ("gyroscope", 3), ("accelerometer", 3)):
            key = f"{label}_{field}"
            data[key], data[f"{key}_age_ms"], data[f"{key}_bracket_ms"] = _sample(
                events,
                grid,
                lambda p, f=field, getter=get_imu: getter(p).get(f),
                width,
                gap,
                quaternion=field == "quaternion",
            )
    data.update(_reference_bodies(selection, model_file, run["warnings"]))
    data["core_mask"] = (data["motion_time_s"] >= selection["start"]) & (data["motion_time_s"] <= selection["end"])
    data["cycle_gap_before_ms"] = np.r_[0, np.diff(grid) / 1e6]
    for key, value in list(data.items()):
        if value.ndim == 2 and np.issubdtype(value.dtype, np.number):
            data[f"{key}_valid"] = np.isfinite(value)
    for e in run["motion"]:
        if grid[0] <= e["monotonic_time_ns"] <= grid[-1] and e["kind"] in (
            "phase",
            "commands",
            "native_state",
            "lifecycle",
        ):
            run.setdefault("report_events", []).append(
                dict(
                    time_s=(e["monotonic_time_ns"] - selection["origin_ns"]) / 1e9, kind=e["kind"], payload=e["payload"]
                )
            )
    if not states:
        run["warnings"].append("No lowstate samples")
    if not commands:
        run["warnings"].append("No lowcmd samples")
    if not torso:
        run["warnings"].append("No secondary IMU samples")
    run["warnings"].append(
        "IMU mounting calibration is not applied; IMU and reference attitudes are shown separately, "
        "without direct attitude-error metrics."
    )
    return data


def compute_comparisons(data):
    results = []
    core = data["core_mask"]
    for a, b, label in (
        ("actual_q", "reference_q", "actual_minus_reference"),
        ("actual_q", "final_q", "actual_minus_final_target"),
        ("actual_q", "dds_q", "actual_minus_dds"),
        ("actual_dq", "reference_dq", "velocity_minus_reference"),
    ):
        error = data[a] - data[b]
        data[label] = error
        data[f"{label}_valid"] = np.isfinite(error)
        for j, name in enumerate(data["joint_names"]):
            ids = np.flatnonzero(core & np.isfinite(error[:, j]))
            if not len(ids):
                results.append(dict(signal=label, joint=str(name), samples=0, rms=None, peak_abs=None))
                continue
            values = error[ids, j]
            i = ids[np.argmax(abs(values))]
            results.append(
                dict(
                    signal=label,
                    joint=str(name),
                    samples=len(ids),
                    rms=float(np.sqrt(np.mean(values**2))),
                    peak_abs=float(abs(error[i, j])),
                    peak_time_s=float(data["time_s"][i]),
                    peak_motion_time_s=float(data["motion_time_s"][i]),
                    peak_query_frame=int(data["query_frame"][i]),
                )
            )
    return results


def write_report(run, selection, data, comparisons, output_dir, max_gap_ms):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_dir = Path(output_dir)
    # Check output policy before rendering or writing anything.
    if output_dir.exists():
        raise FileExistsError(f"output directory already exists: {output_dir}")
    images = []
    x = data["time_s"]
    breaks = np.flatnonzero(data["cycle_gap_before_ms"] > max_gap_ms)

    def plot(ax, y, label, color=None):
        ax.plot(
            np.insert(x, breaks, np.nan), np.insert(np.asarray(y), breaks, np.nan), label=label, color=color, lw=0.9
        )

    def decorate(ax):
        ids = np.flatnonzero(data["core_mask"])
        ax.axvspan(x[ids[0]], x[ids[-1]], color="#f6d365", alpha=0.15)
        ax.grid(alpha=0.2)
        ax.set_xlabel("Elapsed playback time (s)")

    def embed(fig, title):
        stream = io.BytesIO()
        fig.savefig(stream, format="png", dpi=115, bbox_inches="tight")
        plt.close(fig)
        images.append(
            f'<h2>{html.escape(title)}</h2><img alt="{html.escape(title)}" '
            f'src="data:image/png;base64,{base64.b64encode(stream.getvalue()).decode()}">'
        )

    n = len(data["joint_names"])
    cols = min(3, n)
    rows = int(np.ceil(n / cols))
    for velocity in (False, True):
        fig, axes = plt.subplots(rows, cols, figsize=(5.2 * cols, 2.7 * rows), squeeze=False)
        for j, name in enumerate(data["joint_names"]):
            ax = axes.flat[j]
            signals = (
                (("reference_dq", "Reference", "#2563eb"), ("actual_dq", "Measured", "#16a34a"))
                if velocity
                else (
                    ("reference_q", "Reference output", "#2563eb"),
                    ("policy_q", "Policy target", "#9333ea"),
                    ("final_q", "Final attempted target", "#64748b"),
                    ("dds_q", "DDS command", "#ea580c"),
                    ("actual_q", "Measured", "#16a34a"),
                )
            )
            for key, label, color in signals:
                plot(ax, np.rad2deg(data[key][:, j]), label, color)
            ax.set_title(str(name), fontsize=9)
            ax.set_ylabel("deg/s" if velocity else "deg")
            decorate(ax)
        for ax in list(axes.flat)[n:]:
            ax.set_visible(False)
        handles, labels = axes.flat[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper center", ncol=len(labels), fontsize=8)
        fig.tight_layout(rect=(0, 0, 1, 0.97))
        embed(fig, "Joint velocities" if velocity else "Joint positions")
    fig, axes = plt.subplots(2, 3, figsize=(16, 7))
    for row, label in enumerate(("base", "torso")):
        quats = data[f"{label}_quaternion"]
        valid = np.isfinite(quats).all(axis=1)
        rpy = np.full((len(x), 3), np.nan)
        if valid.any():
            rpy[valid] = Rotation.from_quat(quats[valid][:, [1, 2, 3, 0]]).as_euler("xyz", degrees=True)
        for j, name in enumerate(("roll", "pitch")):
            plot(axes[row, 0], rpy[:, j], name)
        for column, field, unit in ((1, "gyroscope", "rad/s"), (2, "accelerometer", "m/s2")):
            for j, name in enumerate("xyz"):
                plot(axes[row, column], data[f"{label}_{field}"][:, j], name)
            axes[row, column].set_ylabel(unit)
        axes[row, 0].set_ylabel("IMU deg (uncalibrated to model)")
        for column in range(3):
            ax = axes[row, column]
            ax.set_title(f"{label}: " + ("attitude", "gyro", "acceleration")[column])
            decorate(ax)
            ax.legend(fontsize=8)
    fig.tight_layout()
    embed(fig, "Measured IMUs (sensor axes)")
    ref_keys = [key for key in ("reference_base_quat", "reference_torso_quat") if key in data]
    if ref_keys:
        fig, axes = plt.subplots(len(ref_keys), 1, figsize=(12, 3 * len(ref_keys)), squeeze=False)
        for ax, key in zip(axes.flat, ref_keys, strict=True):
            quat = data[key]
            valid = np.isfinite(quat).all(axis=1) & (np.linalg.norm(quat, axis=1) > 1e-8)
            angles = np.full((len(x), 3), np.nan)
            if valid.any():
                angles[valid] = Rotation.from_quat(quat[valid][:, [1, 2, 3, 0]]).as_euler("xyz", degrees=True)
            for j, name in enumerate(("roll", "pitch")):
                plot(ax, angles[:, j], name)
            ax.set_title(key + " (reference world)")
            decorate(ax)
            ax.legend()
        fig.tight_layout()
        embed(fig, "Reference body attitudes")
    metadata = dict(
        run_id=run["metadata"]["run_id"],
        segment=selection["segment"],
        policy=selection["info"],
        reference_fps=selection["fps"],
        interval=[selection["start"], selection["end"]],
        context=selection["context"],
        max_gap_ms=max_gap_ms,
        warnings=run["warnings"],
        events=run.get("report_events", []),
        alignment="host monotonic receive/call times; no fitted time shift; linear state brackets, causal command hold",
    )
    quality = {
        key: int((~np.isfinite(data[key])).any(axis=1).sum())
        for key in ("actual_q", "dds_q", "base_quaternion", "torso_quaternion")
    }

    def text(value):
        return html.escape(str(value))

    table = "".join(
        "<tr>"
        + "".join(
            f"<td>{text(row.get(k, ''))}</td>"
            for k in (
                "signal",
                "joint",
                "samples",
                "rms",
                "peak_abs",
                "peak_time_s",
                "peak_motion_time_s",
                "peak_query_frame",
            )
        )
        + "</tr>"
        for row in comparisons
    )
    document = (
        '<!doctype html><html lang="en"><meta charset="utf-8"><title>G1 motion comparison</title><style>'
        "body{font:15px system-ui;max-width:1650px;margin:30px auto;padding:0 20px;color:#18212b}"
        "img{width:100%}pre{white-space:pre-wrap;background:#f1f5f9;padding:16px}"
        "table{border-collapse:collapse;font-size:12px}td,th{border:1px solid #ddd;padding:5px}</style>"
    )
    document += (
        "<h1>G1 motion comparison</h1><p>Read-only analysis. Yellow shading: requested core interval. "
        "X axis: host elapsed time; motion time is query frame / supplied reference FPS. "
        "Repeated paused frames remain separate. No automatic anomaly labels or compensation.</p>"
    )
    document += (
        "<h2>Recording and alignment</h2><pre>" + text(json.dumps(metadata, ensure_ascii=False, indent=2)) + "</pre>"
    )
    document += (
        "<p>Rows with unavailable samples: "
        + text(quality)
        + ". Send results: "
        + text(dict(zip(*np.unique(data["send_status"], return_counts=True), strict=True)))
        + ". Returned means the send call returned, not proof of physical execution.</p>"
    )
    document += (
        "".join(images)
        + "<h2>Core interval differences (rad or rad/s)</h2><table><tr>"
        + "".join(
            f"<th>{v}</th>"
            for v in (
                "signal",
                "joint",
                "samples",
                "RMS",
                "absolute peak",
                "host time s",
                "motion time s",
                "query frame",
            )
        )
        + "</tr>"
        + table
        + "</table></html>"
    )
    output_dir.mkdir(parents=True)
    np.savez_compressed(
        output_dir / "aligned.npz", **data, metadata_json=np.array(json.dumps(metadata, ensure_ascii=False))
    )
    (output_dir / "report.html").write_text(document, encoding="utf-8")
    return output_dir / "report.html"
