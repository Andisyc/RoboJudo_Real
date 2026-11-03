"""Offline check of a BeyondMimic ONNX before putting it in a native-loco mimic config.

Reads only the model (onnxruntime + numpy); no robot, SDK or network. Prints:
  - DoF count and whether the joint set is the G1 23DoF or 29DoF set,
  - the frame where the reference motion stops changing (a lower bound for max_timestep),
  - how long A/Select would wait for a WALKRUN stance, using the pipeline thresholds.

Usage:
    python scripts/check_mimic_model.py assets/models/g1/beyondmimic/<name>.onnx [--max-timestep N]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from robojudo.tools.mimic_safety import JOINT_LIMITS, JOINT_LIMITS_29  # noqa: E402

# Keep in sync with G1NativeLocoMimicV3Pipeline.STANCE_*.
LEG_KEYS = ("hip", "knee", "ankle", "waist")
LEG_TOL, HEIGHT_TOL, REF_VEL, STOP_WAIT_S = 0.35, 0.03, 0.6, 20.0
FREQ = 50


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("onnx")
    parser.add_argument("--max-timestep", type=int, default=None,
                        help="value planned for the config; default scans until the motion stops changing")
    parser.add_argument("--scan-limit", type=int, default=30000)
    args = parser.parse_args()

    session = ort.InferenceSession(args.onnx, providers=["CPUExecutionProvider"])
    names = session.get_modelmeta().custom_metadata_map["joint_names"].split(",")
    joint_set = {23: JOINT_LIMITS, 29: JOINT_LIMITS_29}.get(len(names))
    print(f"model: {args.onnx}")
    print(f"DoF: {len(names)}; joint set matches G1 {len(names)}DoF: {joint_set is not None and set(names) == set(joint_set)}")

    obs = np.zeros((1, session.get_inputs()[0].shape[1]), dtype=np.float32)
    end = args.max_timestep if args.max_timestep else args.scan_limit
    J, V, Z = [], [], []
    for t in range(end + 1):
        q, dq, body = session.run(["joint_pos", "joint_vel", "body_pos_w"],
                                  {"obs": obs, "time_step": np.array([[t]], dtype=np.float32)})
        J.append(np.asarray(q).squeeze()); V.append(np.asarray(dq).squeeze()); Z.append(np.asarray(body).squeeze()[0, 2])
        # Without --max-timestep, stop 5 s after the reference stopped changing.
        if args.max_timestep is None and t > 5 * FREQ and np.abs(np.array(J[-5 * FREQ:]) - J[-1]).max() < 1e-6:
            break
    J, V, Z = np.array(J), np.array(V), np.array(Z)
    changing = np.where(np.abs(J - J[-1]).max(1) > 1e-6)[0]
    motion_end = int(changing.max()) + 1 if len(changing) else 0
    print(f"reference stops changing at frame {motion_end} ({motion_end / FREQ:.1f}s); scanned {len(J) - 1} frames")
    if args.max_timestep is None:
        print(f"-> max_timestep must be >= {motion_end}; use the frame count from training if it is larger")

    legs = [i for i, n in enumerate(names) if any(k in n for k in LEG_KEYS)]
    stances = [(J[0, legs], Z[0]), (J[-1, legs], Z[-1])]
    ok = np.array([
        np.abs(V[t, legs]).max() <= REF_VEL and any(
            np.abs(J[t, legs] - s_q).max() <= LEG_TOL and abs(Z[t] - s_z) <= HEIGHT_TOL for s_q, s_z in stances)
        for t in range(len(J))
    ])
    waits = [np.argmax(ok[t:]) / FREQ if ok[t:].any() else np.inf for t in range(0, max(motion_end, 1), 10)]
    over = sum(w > STOP_WAIT_S for w in waits)
    print(f"pelvis height: start {Z[0]:.3f} m, end {Z[-1]:.3f} m, lowest {Z.min():.3f} m")
    print(f"A/Select wait for a stance (perfect tracking): mean {np.mean(waits):.1f}s, "
          f"p90 {np.percentile(waits, 90):.1f}s, max {max(waits):.1f}s; presses that would hit the "
          f"{STOP_WAIT_S:.0f}s hold: {over}/{len(waits)}")


if __name__ == "__main__":
    main()
