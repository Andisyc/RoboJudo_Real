"""Legacy 29DoF completion flow, without SDK imports or robot connections."""
import ast
import logging
import shutil
import subprocess
import tempfile
import unittest
from enum import Enum, auto
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

ROOT = Path(__file__).resolve().parents[1]
LOGGER = logging.getLogger(__name__)


def load_nodes(relative, names, namespace):
    tree = ast.parse((ROOT / relative).read_text())
    nodes = [node for node in tree.body if getattr(node, "name", None) in names]
    for node in nodes:
        node.decorator_list = []
    future = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
    module = ast.fix_missing_locations(ast.Module(body=[future, *nodes], type_ignores=[]))
    exec(compile(module, relative, "exec"), namespace)
    return namespace


class TestNormalMode(unittest.TestCase):
    def setUp(self):
        self.clock = SimpleNamespace(value=100.0)
        self.time = SimpleNamespace(perf_counter=lambda: self.clock.value, sleep=Mock())
        namespace = load_nodes(
            "robojudo/pipeline/g1_native_loco_mimic_pipeline.py",
            {"NativeLocoMimicState", "G1NativeLocoMimicPipeline"},
            {"Pipeline": object, "Enum": Enum, "auto": auto, "time": self.time,
             "logger": LOGGER, "G1NativeLocoMimicPipelineCfg": object},
        )
        self.states = namespace["NativeLocoMimicState"]
        cls = namespace["G1NativeLocoMimicPipeline"]
        self.pipe = cls.__new__(cls)
        self.calls = []
        self.owned = True
        self.authority_fault = False
        self.release_code = 0
        self.release_wait_s = 0.0
        self.release_loses_control = False
        self.switch_error = None
        self.result = dict(code=0, target_confirmed=True, observed_fsm=500,
                           request_sent=True, sdk_request_result=0, stage="confirm", reason="confirmed")
        self.status = lambda: dict(authority_state="USER_ACTIVE" if self.owned else "INTERNAL",
                                   publish_enabled=self.owned, fsm_id=1000 if self.owned else 801)
        owner = self

        class FakeEnv:
            update = Mock()
            get_data = Mock(return_value={})
            reset = Mock()
            get_control_status = staticmethod(owner.status)

            @property
            def has_user_control(self):
                return owner.owned

            @property
            def has_authority_fault(self):
                return owner.authority_fault

            def release_to_walkrun(self):
                owner.calls.append("release")
                owner.clock.value += owner.release_wait_s
                if owner.release_code == 0 or owner.release_loses_control:
                    owner.owned = False
                return owner.release_code

            def switch_walkrun_to_normal(self):
                owner.calls.append("switch")
                owner.assertFalse(owner.owned)
                owner.clock.value += 1.2
                if owner.switch_error:
                    raise owner.switch_error
                return owner.result

            def shutdown(self, *, preserve_internal_mode=False):
                owner.calls.append("detach" if preserve_internal_mode else "passive")
                return 0

        self.pipe.env = FakeEnv()
        self.pipe.cfg = SimpleNamespace(normal_on_motion_done=True, debug=SimpleNamespace(log_obs=False),
                                        env=SimpleNamespace(is_sim=False), device="cpu",
                                        pipeline_type="G1NativeLocoMimicPipeline", run_fullspeed=False)
        self.pipe.ctrl_manager = SimpleNamespace(get_ctrl_data=Mock(return_value={"COMMANDS": []}),
                                                 post_step_callback=Mock())
        self.pipe.policy_manager = SimpleNamespace(policy=SimpleNamespace(reset=Mock(), post_step_callback=Mock()))
        self.pipe.state = self.states.MIMIC_ACTIVE
        self.pipe.should_stop = False
        self.pipe.timestep = 0
        self.pipe.dt = 0.02
        self.pipe.last_normal_mode_switch = None
        self.pipe._non_realtime_step_s = 0.0
        self.pipe._preserve_internal_mode_on_exit = False
        self.pipe.trace_event = Mock()
        self.pipe._step_mimic = Mock(return_value=({"CALLBACK": ["[MOTION_DONE]"]}, [0.0] * 29))
        self.pipe._safety_check = Mock()

    def test_natural_completion_switches_once_after_release(self):
        self.pipe.step()
        self.pipe.step()
        self.assertEqual(self.calls, ["release", "switch"])
        self.assertIs(self.pipe.last_normal_mode_switch, self.result)
        self.assertEqual(self.pipe.state, self.states.NATIVE_LOCO)

    def test_manual_select_only_returns_walkrun(self):
        self.pipe._handle_commands(["[POLICY_LOCO]"], {}, {})
        self.assertEqual(self.calls, ["release"])

    def test_disabled_config_only_returns_walkrun(self):
        self.pipe.cfg.normal_on_motion_done = False
        self.pipe.step()
        self.assertEqual(self.calls, ["release"])

    def test_failed_release_does_not_switch(self):
        self.release_code = 42
        self.pipe.step()
        self.assertEqual(self.calls, ["release"])
        self.assertEqual(self.pipe.state, self.states.MIMIC_ACTIVE)
        self.assertFalse(self.pipe._preserve_internal_mode_on_exit)

    def test_failed_release_without_ownership_does_not_switch(self):
        self.release_code = 42
        self.release_loses_control = True
        with self.assertRaisesRegex(RuntimeError, "did not reach WALKRUN"):
            self.pipe.step()
        self.assertEqual(self.calls, ["release"])

    def test_unconfirmed_result_never_retries_or_damps(self):
        self.result.update(code=-20108, target_confirmed=False, observed_fsm=801)
        self.pipe.step()
        self.pipe.step()
        self.pipe.close()
        self.assertEqual(self.calls, ["release", "switch", "detach"])
        self.assertFalse(self.pipe.last_normal_mode_switch["target_confirmed"])

    def test_unknown_fsm_fault_detaches_without_damp(self):
        self.result.update(code=-20106, target_confirmed=False, observed_fsm=-1)
        self.authority_fault = True
        self.pipe.step()
        self.assertTrue(self.pipe.should_stop)
        self.assertEqual(self.pipe.state, self.states.FAULT)
        self.pipe.close()
        self.assertEqual(self.calls, ["release", "switch", "detach"])

    def test_switch_exception_preserves_internal_mode_on_cleanup(self):
        self.switch_error = RuntimeError("RPC transport failed")
        with self.assertRaisesRegex(RuntimeError, "RPC transport"):
            self.pipe.step()
        self.pipe.close()
        self.assertEqual(self.calls, ["release", "switch", "detach"])

    def test_a_button_still_requests_passive_after_normal(self):
        self.pipe.step()
        self.pipe._handle_commands(["[SHUTDOWN]"], {}, {})
        self.assertEqual(self.calls, ["release", "switch", "passive"])
        self.assertEqual(self.pipe.state, self.states.CLOSED)

    def test_cleanup_after_normal_preserves_mode(self):
        self.pipe.step()
        self.pipe.close()
        self.pipe.close()
        self.assertEqual(self.calls, ["release", "switch", "detach"])

    def test_transition_time_is_consumed_once_and_never_when_publishing(self):
        self.pipe.step()
        self.assertAlmostEqual(self.pipe.consume_non_realtime_step_time(), 1.2)
        self.assertEqual(self.pipe.consume_non_realtime_step_time(), 0.0)
        self.owned = True
        self.pipe._non_realtime_step_s = 1.2
        self.assertEqual(self.pipe.consume_non_realtime_step_time(), 0.0)

    def run_entrypoint(self):
        original_step = self.pipe.step

        def single_step():
            original_step()
            self.pipe.should_stop = True

        self.pipe.step = single_step
        self.pipe.prepare = Mock()
        self.pipe.close_motion_trace = Mock()
        self.entry_logger = Mock()
        config_manager = lambda **_kwargs: SimpleNamespace(get_cfg=lambda: self.pipe.cfg)
        scope = load_nodes("scripts/run_pipeline.py", {"main"}, {
            "parse_args": lambda: SimpleNamespace(config="g1_native_loco_mimic"),
            "ConfigManager": config_manager, "logger": self.entry_logger,
            "robojudo": SimpleNamespace(pipeline=SimpleNamespace(G1NativeLocoMimicPipeline=lambda **_: self.pipe)),
            "time": self.time, "os": SimpleNamespace(environ={}),
        })
        scope["main"]()

    def test_official_entrypoint_does_not_damp_during_expected_wait(self):
        self.run_entrypoint()
        self.assertEqual(self.calls, ["release", "switch", "detach"])
        self.entry_logger.critical.assert_not_called()
        self.time.sleep.assert_not_called()

    def test_slow_confirmed_walkrun_release_is_also_an_expected_wait(self):
        self.release_wait_s = 0.5
        self.run_entrypoint()
        self.assertEqual(self.calls, ["release", "switch", "detach"])
        self.entry_logger.critical.assert_not_called()

    def test_official_entrypoint_keeps_active_control_watchdog(self):
        def slow_policy(*_args):
            self.clock.value += 0.3
            return {}, [0.0] * 29

        self.pipe._step_mimic = slow_policy
        self.run_entrypoint()
        self.assertIn("passive", self.calls)
        self.assertNotIn("switch", self.calls)
        self.entry_logger.critical.assert_called_once()

    def test_slow_inference_is_not_hidden_by_later_normal_wait(self):
        def slow_policy(*_args):
            self.clock.value += 0.3
            return {"CALLBACK": ["[MOTION_DONE]"]}, [0.0] * 29

        self.pipe._step_mimic = slow_policy
        self.run_entrypoint()
        self.assertIn("passive", self.calls)
        self.entry_logger.critical.assert_called_once()

    def test_runner_respects_pipeline_frame_drop_opt_out(self):
        def slow_policy(*_args):
            self.clock.value += 0.3
            return {}, [0.0] * 29

        self.pipe.exit_on_frame_drop = False
        self.pipe._step_mimic = slow_policy
        self.run_entrypoint()
        # Only the ordinary close may release; the frame watchdog must not
        # call env.shutdown() while this pipeline keeps its command stream.
        self.assertEqual(self.calls, ["passive"])
        self.entry_logger.critical.assert_not_called()
        self.time.sleep.assert_not_called()


class TestEnvWrapper(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        namespace = load_nodes("robojudo/environment/g1_loco_env.py", {"G1LocoEnv"}, {
            "Environment": object, "UnitreeEnvCfg": object, "logger": LOGGER,
        })
        cls.env_class = namespace["G1LocoEnv"]

    def test_wrapper_blocks_disabled_act_and_old_binary(self):
        env = self.env_class.__new__(self.env_class)
        env.enabled = False
        with self.assertRaisesRegex(RuntimeError, "act=False"):
            env.switch_walkrun_to_normal()
        env.enabled = True
        env.unitree = object()
        with self.assertRaisesRegex(RuntimeError, "rebuild"):
            env.switch_walkrun_to_normal()

    def test_wrapper_returns_sdk_evidence_and_uses_explicit_cleanup(self):
        env = self.env_class.__new__(self.env_class)
        env.enabled = True
        result = dict(code=0, target_confirmed=True, observed_fsm=500, sdk_request_result=42)
        env.unitree = SimpleNamespace(switch_walkrun_to_normal=Mock(
            return_value=SimpleNamespace(to_dict=lambda: result)),
            close=Mock(return_value=0), close_preserving_internal_mode=Mock(return_value=0))
        env.get_control_status = lambda: {}
        self.assertEqual(env.switch_walkrun_to_normal(), result)
        env.shutdown(preserve_internal_mode=True)
        env.unitree.close_preserving_internal_mode.assert_called_once()
        env.unitree.close.assert_not_called()
        env.shutdown()
        env.unitree.close.assert_called_once()


class TestCppTransition(unittest.TestCase):
    def test_offline_transition_fault_scenarios(self):
        compiler = shutil.which("c++")
        self.assertIsNotNone(compiler, "A C++17 compiler is required")
        with tempfile.TemporaryDirectory() as directory:
            executable = str(Path(directory) / "normal_switch")
            subprocess.run([compiler, "-std=c++17", "-Wall", "-Wextra", "-Werror",
                            "-I", str(ROOT / "packages/g1_loco_bridge/src"),
                            str(ROOT / "tests/cpp/test_normal_mode_switch.cpp"),
                            "-o", executable], check=True)
            subprocess.run([executable], check=True, timeout=10)


if __name__ == "__main__":
    unittest.main()
