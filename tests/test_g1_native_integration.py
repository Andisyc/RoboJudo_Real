"""Shared native route checks; never import an SDK or construct a controller."""
import importlib.abc
from pathlib import Path
import runpy
import sys
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


class NoHardware(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.startswith(("g1_loco_bridge", "unitree_sdk2py", "unitree_cpp", "zed_proxy")):
            raise AssertionError(f"Test attempted to import hardware: {fullname}")
        return None


guard = NoHardware()
sys.meta_path.insert(0, guard)
try:
    from robojudo.config import cfg_registry
    from robojudo.config.config_manager import ConfigManager
    from robojudo.environment.g1_loco_env import _load_g1_loco_bridge
    from robojudo.pipeline import pipeline_registry
finally:
    sys.meta_path.remove(guard)


class TestNativeIntegration(unittest.TestCase):
    def setUp(self):
        guard = NoHardware()
        sys.meta_path.insert(0, guard)
        self.addCleanup(sys.meta_path.remove, guard)

    def config(self, name):
        return ConfigManager(config_name=name).get_cfg()

    def test_all_native_configs_use_one_pipeline(self):
        expected = pipeline_registry.get("G1NativeLocoMimicPipeline")
        for name in ("g1_native_loco_mimic", "g1_native_loco_mimic_v2",
                     "g1_native_loco_mimic_23dof", "g1_native_loco_mimic_23dof_backpack"):
            with self.subTest(name=name):
                cfg = self.config(name)
                self.assertIs(pipeline_registry.get(cfg.pipeline_type), expected)
                self.assertNotIn("bridge_variant", type(cfg.env).model_fields)
        self.assertFalse(expected.exit_on_frame_drop)
        self.assertNotIn("G1NativeLocoMimicV3Pipeline", pipeline_registry.types)
        self.assertNotIn("g1_native_loco_mimic_29dof", cfg_registry.types)
        self.assertNotIn("g1_native_loco_mimic_legacy", cfg_registry.types)

    def test_original_launch_preserves_robot_and_motion_settings(self):
        cfg = self.config("g1_native_loco_mimic")
        self.assertEqual(len(cfg.env.dof.joint_names), 29)
        self.assertEqual(cfg.env.unitree.net_if, "enP8p1s0")
        self.assertEqual(cfg.env.unitree.regular_fsm_id, 501)
        self.assertEqual(cfg.mimic_policies[0].policy_name, "g1_dance2")
        self.assertEqual(cfg.mimic_policies[0].max_timestep, 1000)
        self.assertTrue(Path(cfg.mimic_policies[0].policy_file).is_file())
        self.assertTrue(cfg.env.update_with_fk)
        self.assertEqual(cfg.env.torso_name, "torso_link")
        self.assertTrue(Path(cfg.env.forward_kinematic.xml_path).is_file())
        self.assertTrue(cfg.return_to_regular_loco)
        self.assertTrue(cfg.walkrun_on_start)

    def test_23dof_preserves_checkpoint_and_fk_settings(self):
        for name, motion, steps in [("g1_native_loco_mimic_23dof", "eight_cut_1", -1),
                                    ("g1_native_loco_mimic_23dof_backpack", "eight_cut_1_backpack", 11252)]:
            with self.subTest(name=name):
                cfg = self.config(name)
                self.assertEqual(len(cfg.env.dof.joint_names), 23)
                self.assertEqual(cfg.mimic_policies[0].policy_name, motion)
                self.assertEqual(cfg.mimic_policies[0].max_timestep, steps)
                self.assertFalse(cfg.env.update_with_fk)
                self.assertIsNone(cfg.env.forward_kinematic)

    def test_fixed_hip_bias_at_send_boundary(self):
        import numpy as np
        from pydantic import ValidationError
        from robojudo.pipeline.pipeline_cfgs import FixedHipBiasCfg

        cfg = self.config("g1_native_loco_mimic_23dof_backpack")
        self.assertFalse(cfg.fixed_hip_bias.enabled)
        self.assertEqual(cfg.mimic_policies[0].policy_name, "eight_cut_1_backpack")
        cfg.fixed_hip_bias = FixedHipBiasCfg(enabled=True)
        pipeline_type = pipeline_registry.get("G1NativeLocoMimicPipeline")
        pipeline = pipeline_type.__new__(pipeline_type)  # Never construct SDK/environment.
        pipeline.cfg = cfg
        names = list(reversed(cfg.env.dof.joint_names))  # Deliberately different order.
        pipeline._hip_bias_indices = [names.index(n) for n in
                                     ("left_hip_pitch_joint", "right_hip_pitch_joint")]
        original = np.linspace(-0.2, 0.2, len(names), dtype=np.float32)
        baseline = original.copy()
        action_history = np.ones(len(names), dtype=np.float32)
        policy = SimpleNamespace(
            cfg_policy=SimpleNamespace(policy_name=cfg.fixed_hip_bias.policy_name),
            timestep=0, play_speed=1., last_action=action_history.copy(),
            get_observation=Mock(return_value=("observation", {"hand_pose": "hands"})),
            get_pd_target=Mock(return_value=original),
        )
        pipeline.policy_manager = SimpleNamespace(policy=policy)
        pipeline.send_traced = Mock()

        def sent(frame):
            policy.timestep = frame
            _, target = pipeline._step_mimic({}, {"COMMANDS": ["test"]})
            np.testing.assert_array_equal(pipeline.send_traced.call_args.args[0], target)
            self.assertEqual(pipeline.send_traced.call_args.args[1], "hands")
            self.assertEqual(pipeline.send_traced.call_args.kwargs["commands"], ["test"])
            return target

        # Independently specified boundary/midpoint values, not the implementation formula.
        for frame, fraction in [(1099, 0), (1100, 0), (1120.9, .5), (1140, 1),
                                (1200, 1), (1310, 1), (1330, .5), (1350, 0), (2550, 0)]:
            with self.subTest(frame=frame):
                expected = np.zeros_like(original)
                expected[pipeline._hip_bias_indices] = np.deg2rad(.5) * fraction
                np.testing.assert_allclose(sent(frame) - baseline, expected, atol=2e-8, rtol=0)

        # Frozen query frames retain the same bias; repeated calls never accumulate it.
        policy.play_speed = 0
        paused = sent(1120).copy()
        np.testing.assert_array_equal(sent(1120), paused)
        np.testing.assert_array_equal(sent(0), baseline)  # Replay/reset starts outside the window.
        policy.cfg_policy.policy_name = "different_motion"
        np.testing.assert_array_equal(sent(1200), baseline)
        policy.cfg_policy.policy_name = cfg.fixed_hip_bias.policy_name
        cfg.fixed_hip_bias.enabled = False
        self.assertIs(sent(1200), original)
        cfg.fixed_hip_bias.enabled = True
        cfg.fixed_hip_bias.hip_pitch_deg = 0
        self.assertIs(sent(1200), original)
        np.testing.assert_array_equal(original, baseline)
        np.testing.assert_array_equal(policy.last_action, action_history)

        for invalid in ({"hip_pitch_deg": float("nan")}, {"hip_pitch_deg": float("inf")},
                        {"hip_pitch_deg": .51}, {"hip_pitch_deg": -.1},
                        {"ramp_in_end": 1100}, {"ramp_in_start": -1},
                        {"ramp_out_start": 1350}, {"policy_name": " "}):
            with self.subTest(invalid=invalid), self.assertRaises(ValidationError):
                FixedHipBiasCfg(**invalid)

    def test_both_dofs_import_same_bridge_and_reject_stale_binary(self):
        bridge = ModuleType("g1_loco_bridge")
        bridge.MIMIC_SAFETY_REVISION = 2026110310
        bridge.AuthorityState, bridge.G1LocoController = object(), object()
        bridge.RobotState, bridge.SportState = object(), object()
        with patch.dict(sys.modules, {"g1_loco_bridge": bridge}):
            for dofs in (23, 29):
                self.assertIs(_load_g1_loco_bridge(dofs)[1], bridge.G1LocoController)
            bridge.MIMIC_SAFETY_REVISION = 2026110305
            with self.assertRaisesRegex(RuntimeError, "Rebuild packages/g1_loco_bridge"):
                _load_g1_loco_bridge(29)
            del bridge.MIMIC_SAFETY_REVISION
            with self.assertRaisesRegex(RuntimeError, "Rebuild packages/g1_loco_bridge"):
                _load_g1_loco_bridge(23)
        with self.assertRaisesRegex(ValueError, "23 or 29"):
            _load_g1_loco_bridge(12)

    def test_cut3_candidates_are_isolated_from_original_routes(self):
        original=self.config("g1_native_loco_mimic_23dof_backpack_cut3_recovery")
        switch=self.config("g1_native_loco_mimic_23dof_backpack_cut3_fsm500")
        pitch=self.config("g1_native_loco_mimic_23dof_backpack_cut3_pitch")
        ramp=self.config("g1_native_loco_mimic_23dof_backpack_cut3_transition")
        entry=self.config("g1_native_loco_mimic_23dof_backpack_cut3_entry_transition")
        self.assertEqual((entry.entry_pose_transition_s,entry.exit_pose_transition_s),(3.,0.))
        self.assertEqual(original.env.unitree.regular_fsm_id,501)
        self.assertEqual(original.mimic_policies[0].policy_name,"cut3_recovery_r1_i80")
        self.assertEqual(switch.mimic_policies[0].policy_name,original.mimic_policies[0].policy_name)
        for cfg in (switch,pitch,ramp):
            self.assertEqual(cfg.env.unitree.regular_fsm_id,500)
            self.assertEqual(len(cfg.env.dof.joint_names),23)
            self.assertFalse(cfg.env.update_with_fk)
            self.assertEqual(cfg.env.odometry_type,original.env.odometry_type)
            self.assertEqual(cfg.mimic_policies[0].max_timestep,5299)
        self.assertEqual(pitch.mimic_policies[0].policy_name,"cut3_pitch_ref4_20261112")
        self.assertEqual(pitch.entry_pose_transition_s,0)
        self.assertEqual(pitch.exit_pose_transition_s,0)
        self.assertEqual((ramp.entry_pose_transition_s,ramp.exit_pose_transition_s),(3.,3.))
        self.assertEqual(self.config("g1_native_loco_mimic").env.unitree.regular_fsm_id,501)
        switch.env.unitree.regular_fsm_id=501
        self.assertEqual(pitch.env.unitree.regular_fsm_id,500)

    def test_original_cli_resolves_shared_pipeline_without_hardware(self):
        # Exercise the official parser/ConfigManager/registry path, replacing
        # construction before it can subscribe to DDS or request any mode.
        import robojudo.pipeline
        namespace = runpy.run_path(str(ROOT / "scripts/run_pipeline.py"), run_name="offline_entry_check")
        fake = Mock(should_stop=True)
        fake.close.return_value = 0
        constructor = Mock(return_value=fake)
        with patch.object(robojudo.pipeline, "G1NativeLocoMimicPipeline", constructor), \
                patch.dict("os.environ", {}, clear=True), \
                patch.object(sys, "argv", ["scripts/run_pipeline.py", "-c", "g1_native_loco_mimic"]):
            namespace["main"]()
        cfg = constructor.call_args.kwargs["cfg"]
        self.assertEqual(cfg.pipeline_type, "G1NativeLocoMimicPipeline")
        self.assertEqual(cfg.mimic_policies[0].policy_name, "g1_dance2")
        fake.prepare.assert_called_once()
        fake.step.assert_not_called()
        fake.close.assert_called_once()


if __name__ == "__main__":
    unittest.main()
