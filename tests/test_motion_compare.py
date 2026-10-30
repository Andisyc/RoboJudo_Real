"""One focused end-to-end fixture with known mapping, lag, pause, reset and gaps."""

import hashlib
import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import msgpack
import numpy as np
import onnx

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("motion_compare", ROOT / "robojudo/tools/motion_compare.py")
compare = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(compare)


def fixture(directory):
    origin = 10_000_000_000

    def event(kind, t, payload):
        return dict(kind=kind, monotonic_time_ns=origin + round(t * 1e9), payload=payload)

    def imu(t):
        return dict(quaternion=[1, 0, 0, 0], gyroscope=[t, 0, 0], accelerometer=[0, 0, 9.81])

    # Policy order: right,left. Environment order: left,right. Hardware order: spare,right,left.
    meta = dict(
        run_id="fixture",
        environment_joint_names=["left", "right"],
        joint_to_hardware_motor=[2, 1],
        hardware_joint_names=["unused", "right", "left"],
    )
    motion = [
        event("header", 0, dict(schema_version=1, metadata=meta)),
        event(
            "segment",
            0,
            dict(
                segment=1,
                policy_name="fixture",
                policy_file="missing.onnx",
                policy_joint_names=["right", "left"],
                configured_policy_hz=50,
            ),
        ),
    ]
    # 25 FPS reference. Pause holds frame 10 at two distinct receive times.
    samples = [(0.1, 0), (0.2, 5), (0.3, 10), (0.4, 10), (0.5, 15), (0.6, 20), (0.7, 25), (0.8, 30), (0.9, 35)]
    for cycle, (t, frame) in enumerate(samples):
        phase = "paused" if t == 0.4 else "playing"
        # Actual channels are delayed 0.05s: actual right=t-.05; final right=t+.1.
        q = [t + 0.1, 2 * t + 0.2]
        motion.append(
            event(
                "cycle",
                t,
                dict(
                    cycle=cycle,
                    segment=1,
                    phase=phase,
                    final_target=q[::-1],
                    inference=dict(
                        query_frame=frame,
                        policy_target=q,
                        observation_reference=dict(joint_pos=[t - 0.01, 2 * t - 0.02]),
                        output_reference=dict(
                            joint_pos=[t, 2 * t],
                            joint_vel=[1, 2],
                            body_quat_w=[[np.cos(0.1), 0.0, np.sin(0.1), 0.0], [1.0, 0.0, 0.0, 0.0]],
                        ),
                    ),
                ),
            )
        )
        motion.append(event("send_result", t + 0.001, dict(cycle=cycle, status="returned", result=None)))
    motion.extend(
        [
            event("segment", 1.01, dict(segment=2, policy_joint_names=["right", "left"])),
            event(
                "cycle",
                1.1,
                dict(
                    cycle=9,
                    segment=2,
                    phase="playing",
                    final_target=[999, 999],
                    inference=dict(query_frame=0, output_reference=dict(joint_pos=[999, 999])),
                ),
            ),
            event("summary", 1.2, dict(complete=True, dropped=0, error=None)),
        ]
    )
    state = [event("header", 0, dict(schema_version=1, metadata=dict(run_id="fixture")))]
    for i in range(121):
        t = i / 100
        if 0.35 < t < 0.65:
            continue  # A known 300ms gap must stay missing.
        state.append(
            event(
                "lowstate",
                t,
                dict(
                    motor=dict(q=[999, t - 0.05, 2 * (t - 0.05)], dq=[999, 1, 2], tau_est=[999, 3, 4]), base_imu=imu(t)
                ),
            )
        )
        state.append(event("lowcmd", t, dict(motor=dict(q=[999, t + 0.1, 2 * t + 0.2]))))
        state.append(event("torso_imu", t, imu(t)))
    state.append(event("summary", 1.3, dict(dropped=0)))
    for name, records in (("motion", motion), ("state", state)):
        (directory / f"{name}.msgpack").write_bytes(b"".join(msgpack.packb(e, use_bin_type=True) for e in records))


class TestMotionCompare(unittest.TestCase):
    def test_end_to_end(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture(root)
            inputs = [root / "state.msgpack", root / "motion.msgpack"]
            hashes = [hashlib.sha256(p.read_bytes()).digest() for p in inputs]
            run = compare.load_run(root)
            with self.assertRaisesRegex(ValueError, "select --segment"):
                compare.select_segment(run, None, 25, 0, 1.4)
            selected = compare.select_segment(run, 1, 25, 0.2, 1.2, context=0.2)
            data = compare.align_signals(run, selected, max_gap_ms=100)
            np.testing.assert_array_equal(data["query_frame"], [0, 5, 10, 10, 15, 20, 25, 30, 35])
            self.assertEqual(data["motion_time_s"][2], data["motion_time_s"][3])
            self.assertNotEqual(data["time_s"][2], data["time_s"][3])
            self.assertEqual(data["phase"][3], "paused")
            self.assertTrue(np.isnan(data["actual_q"][3:6]).all())
            self.assertTrue(np.isnan(data["dds_q"][5]).all())
            np.testing.assert_allclose(data["actual_q"][1], [0.15, 0.3])
            np.testing.assert_allclose(data["final_q"][1], [0.3, 0.6])
            np.testing.assert_allclose(data["dds_q"][1], [0.3, 0.6])
            self.assertFalse(np.any(data["reference_q"] == 999))
            metrics = compare.compute_comparisons(data)
            metric = next(m for m in metrics if m["signal"] == "actual_minus_final_target" and m["joint"] == "right")
            self.assertAlmostEqual(metric["rms"], 0.15)
            np.testing.assert_allclose(data["actual_minus_reference"][1], [-0.05, -0.1])
            # Interpolation never extrapolates, and command sampling never looks ahead.
            e = [dict(monotonic_time_ns=t, payload={"q": [v]}) for t, v in ((100, 1), (200, 2))]
            sampled, _, _ = compare._sample(e, np.array([90, 150, 210]), lambda p: p["q"], 1, 100)
            self.assertTrue(np.isnan(sampled[[0, 2]]).all())
            self.assertEqual(sampled[1, 0], 1.5)
            held, _, _ = compare._sample(e, np.array([90, 150]), lambda p: p["q"], 1, 100, hold=True)
            self.assertTrue(np.isnan(held[0, 0]))
            self.assertEqual(held[1, 0], 1)
            model = onnx.helper.make_model(onnx.helper.make_graph([], "metadata", [], []))
            onnx.helper.set_model_props(model, {"joint_names": "right,left", "body_names": "torso_link,pelvis"})
            model_file = root / "deployed.onnx"
            onnx.save(model, model_file)
            with_bodies = compare.align_signals(run, selected, model_file=model_file)
            np.testing.assert_allclose(with_bodies["reference_base_quat"], np.tile([1.0, 0, 0, 0], (9, 1)))
            np.testing.assert_allclose(with_bodies["reference_torso_quat"][:, 2], np.sin(0.1))
            output = root / "comparison"
            result = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts/compare_g1_motion.py"),
                    "--run-dir",
                    str(root),
                    "--segment",
                    "1",
                    "--reference-fps",
                    "25",
                    "--start",
                    ".2",
                    "--end",
                    "1.2",
                    "--context",
                    ".2",
                    "--output-dir",
                    str(output),
                    "--model-file",
                    str(model_file),
                ],
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            report = (output / "report.html").read_text()
            self.assertIn("data:image/png;base64,", report)
            self.assertIn("fixture", report)
            self.assertIn("Reference body attitudes", report)
            with np.load(output / "aligned.npz", allow_pickle=False) as saved:
                np.testing.assert_array_equal(saved["joint_names"], ["right", "left"])
                np.testing.assert_allclose(saved["actual_q"], data["actual_q"], equal_nan=True)
                self.assertEqual(json.loads(str(saved["metadata_json"]))["reference_fps"], 25)
                self.assertFalse(saved["actual_q_valid"][4].any())
            self.assertEqual(hashes, [hashlib.sha256(p.read_bytes()).digest() for p in inputs])
            # Wrong-run and truncated-tail behavior belongs to this same input boundary.
            damaged = (root / "state.msgpack").read_bytes().replace(b"fixture", b"WRONGID")
            (root / "state.msgpack").write_bytes(damaged)
            with self.assertRaisesRegex(ValueError, "run IDs"):
                compare.load_run(root)
            fixture(root)
            with (root / "state.msgpack").open("ab") as stream:
                stream.write(b"\x81\xa7partial")
            self.assertTrue(any("truncated" in s for s in compare.load_run(root)["warnings"]))


if __name__ == "__main__":
    unittest.main()
