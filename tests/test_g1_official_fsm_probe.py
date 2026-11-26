from pathlib import Path
import sys
import unittest
from types import ModuleType
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from g1_official_fsm_probe import (
    ProbeError, StorageWatch, confirm_host, run_phase, stand_in_debug, wait_for_operator_if_active,
)


class Clock:
    seconds = 0.0

    def now(self):
        return self.seconds

    def sleep(self, seconds):
        self.seconds += seconds


class FakeEnv:
    def __init__(self):
        self.fsm = 500
        self.commands = []
        self.last_walkrun_fsm_trace = []
        self.last_regular_fsm_trace = []
        self.walkrun_target = 801
        self.regular_result = 0
        self.published = 0

    def sync_operator_fsm(self):
        return self.fsm

    def get_control_status(self):
        return {"authority_state": "INTERNAL", "publish_enabled": False, "fsm_id": self.fsm,
                "active_publish_count": self.published}

    def switch_regular_to_walkrun(self):
        self.commands.append(self.walkrun_target)
        self.fsm = self.walkrun_target
        return 0

    def switch_walkrun_to_regular(self):
        self.commands.append(500)
        if self.regular_result == 0:
            self.fsm = 500
        return self.regular_result


def ignore_storage(*, force=False):
    pass


class SwitchProbeTest(unittest.TestCase):
    def setUp(self):
        self.clock = Clock()
        self.env = FakeEnv()
        self.events = []

    def run_probe(self, phase, *, stationary=lambda: (True, "still"),
                  check_storage=ignore_storage):
        run_phase(self.env, phase, stationary=stationary, check_storage=check_storage,
                  still_seconds=1.0, wait_seconds=8.0, period=0.02,
                  now=self.clock.now, sleep=self.clock.sleep,
                  output=lambda name, **fields: self.events.append((name, fields)))

    def test_five_fast_cycles_then_observe(self):
        self.run_probe("fast")
        self.assertEqual(self.env.commands, [801, 500] * 5)
        requests = [x["source_dwell_s"] for name, x in self.events if name == "switch_request"]
        self.assertEqual(requests[::2], [0] * 5)
        self.assertTrue(all(1 <= seconds <= 1.02 for seconds in requests[1::2]))
        self.assertGreaterEqual(self.clock.now(), 155)
        self.assertEqual(self.env.published, 0)

    def test_five_paced_cycles_reuse_still_time_inside_dwell(self):
        self.run_probe("paced")
        self.assertEqual(self.env.commands, [801, 500] * 5)
        requests = [x["source_dwell_s"] for name, x in self.events if name == "switch_request"]
        self.assertTrue(all(5 <= seconds <= 5.02 for seconds in requests))
        self.assertGreaterEqual(self.clock.now(), 200)
        self.assertEqual(self.env.published, 0)

    def test_requires_exact_source_fsm(self):
        self.env.fsm = 501
        with self.assertRaisesRegex(ProbeError, "read back 501"):
            self.run_probe("fast")
        self.assertEqual(self.env.commands, [])

    def test_detects_publishing_before_first_request(self):
        self.env.published = 1
        with self.assertRaisesRegex(ProbeError, "active_publish_count"):
            self.run_probe("fast")
        self.assertEqual(self.env.commands, [])

    def test_damp_readback_prevents_further_switches(self):
        self.env.walkrun_target = 1
        with self.assertRaisesRegex(ProbeError, "read back 1"):
            self.run_probe("fast")
        self.assertEqual(self.env.commands, [1])

    def test_standing_gate_times_out_without_forced_return(self):
        with self.assertRaisesRegex(ProbeError, "FSM 801 did not meet"):
            self.run_probe("fast", stationary=lambda: (self.env.fsm == 500, "moving"))
        self.assertEqual(self.env.commands, [801])

    def test_aer_change_between_switches_stops_before_return_request(self):
        watch = StorageWatch(sample=lambda: {
            "nvme": "live", "root_rw": True, "aer": {"fatal": int(self.env.fsm == 801)}
        })
        with patch("g1_official_fsm_probe.emit"):
            with self.assertRaisesRegex(ProbeError, "AER counters changed"):
                self.run_probe("fast", check_storage=watch.check)
        self.assertEqual(self.env.commands, [801])

    def test_bridge_failure_ends_group(self):
        self.env.regular_result = -20002
        with self.assertRaisesRegex(ProbeError, "failed: -20002"):
            self.run_probe("fast")
        self.assertEqual(self.env.commands, [801, 500])

    def test_storage_failure_during_observation_prevents_pass(self):
        def check_storage(*, force=False):
            if self.clock.now() >= 130:
                raise ProbeError("NVMe lost")

        with self.assertRaisesRegex(ProbeError, "NVMe lost"):
            self.run_probe("fast", check_storage=check_storage)
        self.assertEqual(self.env.commands, [801, 500] * 5)
        self.assertEqual(self.events[-1][0], "observation_started")

    @patch("g1_official_fsm_probe.subprocess.run")
    @patch("g1_official_fsm_probe.read", return_value="different-machine")
    def test_wrong_target_exits_before_power_probe(self, read, subprocess_run):
        with self.assertRaisesRegex(ProbeError, "fingerprint does not match"):
            confirm_host("0" * 64)
        read.assert_called_once_with("/etc/machine-id")
        subprocess_run.assert_not_called()



class FakeDebugEnv(FakeEnv):
    def __init__(self, np, names):
        super().__init__()
        self.np = np
        self.joint_names = names
        self.default_pos = np.zeros(len(names))
        self.current = np.full(len(names), 0.1)
        self.stiffness = np.full(len(names), 100.0)
        self.position_limits = np.tile([-1.0, 1.0], (len(names), 1))
        self.targets = []
        self.release_result = 0
        self.sensors_age = 0.02

    def get_control_status(self):
        status = super().get_control_status()
        if self.fsm == 1000:
            status.update(authority_state="USER_ACTIVE", publish_enabled=True)
        return status

    def update(self):
        pass

    def get_data(self):
        return dict(dof_pos=self.current.copy(), dof_vel=self.np.zeros(len(self.joint_names)),
                    base_quat=[0, 0, 0, 1], base_ang_vel=[0, 0, 0], base_lin_vel=[0, 0, 0])

    def measured_joint_torque(self):
        return self.np.full(len(self.joint_names), 20.0)

    def safety_state_age(self):
        return self.sensors_age

    def acquire_user_control(self, target):
        self.commands.append(1000)
        self.fsm = 1000
        self.step(target)
        return 0

    def step(self, target):
        self.targets.append(target.copy())
        self.current = target.copy()
        self.published += 1
        return self.published

    def release_to_walkrun(self):
        self.commands.append(801)
        if self.release_result == 0:
            self.fsm = 801
        return self.release_result


class DebugStandingTest(unittest.TestCase):
    def setUp(self):
        try:
            import numpy as np
            import scipy.spatial.transform  # noqa: F401
        except ImportError:
            self.skipTest("debug standing test requires NumPy/SciPy")
        # Use the real guard and entry blend without importing live SDK bindings.
        repo_root = Path(__file__).resolve().parents[1]
        robojudo = ModuleType("robojudo")
        robojudo.__path__ = [str(repo_root / "robojudo")]
        isolated_import = patch.dict(sys.modules, {"robojudo": robojudo})
        isolated_import.start()
        self.addCleanup(isolated_import.stop)
        from robojudo.tools.mimic_safety import JOINT_LIMITS, MimicSafetyGuard
        self.env = FakeDebugEnv(np, list(JOINT_LIMITS))
        self.guard = MimicSafetyGuard(self.env.joint_names)
        self.clock = Clock()
        self.events = []
        self.output = lambda event, **fields: self.events.append((event, fields))

    def debug_stand(self, check_storage=ignore_storage):
        return stand_in_debug(
            self.env, self.guard, list(range(13)), check_storage, 0.02, 0.2, 0.2,
            now=self.clock.now, sleep=self.clock.sleep, output=self.output,
        )

    def test_full_sequence_blends_then_holds_five_seconds_and_stops_publishing(self):
        left_elbow = self.env.joint_names.index("left_elbow_joint")
        right_elbow = self.env.joint_names.index("right_elbow_joint")
        self.env.current[:13] = 0.061
        self.env.current[left_elbow] = 0.824
        self.env.current[right_elbow] = 0.784
        starting_pose = self.env.current.copy()
        final_count = run_phase(
            self.env, "debug", stationary=lambda: (True, ""),
            check_storage=ignore_storage, still_seconds=1, wait_seconds=8,
            period=0.02, debug_hold=self.debug_stand, now=self.clock.now,
            sleep=self.clock.sleep, output=self.output,
        )
        self.assertEqual(self.env.commands, [801, 1000, 801, 500])
        self.assertGreater(final_count, 250)
        self.assertEqual(self.env.fsm, 500)
        self.assertFalse(self.env.get_control_status()["publish_enabled"])
        events = dict(self.events)
        self.assertGreaterEqual(events["debug_stand_hold_complete"]["hold_s"], 5)
        self.assertEqual(events["debug_handback_confirmed"]["publish_count"], final_count)
        self.assertTrue((self.env.targets[-1][:13] == self.env.default_pos[:13]).all())
        self.assertTrue(all((target[13:] == starting_pose[13:]).all()
                            for target in self.env.targets))
        self.assertEqual(events["debug_stance_target"]["max_default_delta_joint"], "left_elbow_joint")
        self.assertLess(abs(self.env.targets[0][0] - starting_pose[0]), 0.06)

    def test_large_stance_error_never_enters_user_control(self):
        self.env.fsm = 801
        self.env.current[0] = 0.3
        with self.assertRaisesRegex(ProbeError, "too far"):
            self.debug_stand()
        self.assertEqual(self.env.commands, [])

    def test_stale_feedback_never_enters_user_control(self):
        from robojudo.tools.mimic_safety import SafetyViolation
        self.env.fsm = 801
        self.env.sensors_age = 0.2
        with self.assertRaisesRegex(SafetyViolation, "stale entry LowState"):
            self.debug_stand()
        self.assertEqual(self.env.commands, [])

    def test_measured_arm_outside_limits_never_enters_user_control(self):
        self.env.fsm = 801
        elbow = self.env.joint_names.index("left_elbow_joint")
        self.env.current[elbow] = 1.1
        with self.assertRaisesRegex(ProbeError, "outside joint limits: left_elbow_joint"):
            self.debug_stand()
        self.assertEqual(self.env.commands, [])

    def test_storage_fault_while_in_debug_does_not_request_more_modes(self):
        self.env.fsm = 801

        def check_storage(*, force=False):
            if self.env.fsm == 1000:
                raise ProbeError("AER changed")

        with self.assertRaisesRegex(ProbeError, "AER changed"):
            self.debug_stand(check_storage)
        self.assertEqual(self.env.commands, [1000])
        self.assertTrue(self.env.get_control_status()["publish_enabled"])

    def test_failed_handback_waits_for_operator_without_requesting_again(self):
        self.env.fsm = 801
        self.env.release_result = -1
        with self.assertRaisesRegex(ProbeError, "handback"):
            self.debug_stand()
        self.assertEqual(self.env.fsm, 1000)
        calls = []

        def operator_pause(seconds):
            calls.append(seconds)
            if len(calls) == 1:
                raise KeyboardInterrupt
            self.env.fsm = 801

        wait_for_operator_if_active(self.env, sleep=operator_pause, output=self.output)
        self.assertEqual(len(calls), 2)
        self.assertEqual(self.env.commands, [1000, 801])
        self.assertEqual(self.events[-1][0], "debug_manual_takeover_confirmed")



if __name__ == "__main__":
    unittest.main()
