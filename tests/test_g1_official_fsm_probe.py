from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from g1_official_fsm_probe import ProbeError, StorageWatch, confirm_host, run_phase


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


if __name__ == "__main__":
    unittest.main()
