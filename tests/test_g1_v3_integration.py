"""V3 registry/config/Bridge contracts; hardware imports are blocked throughout."""
import importlib.abc
from pathlib import Path
import sys
from types import ModuleType
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


class NoHardware(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.startswith(("g1_loco_bridge", "unitree_sdk2py", "unitree_cpp", "zed_proxy")):
            raise AssertionError(f"Test attempted to import hardware: {fullname}")
        return None


_import_guard = NoHardware()
sys.meta_path.insert(0, _import_guard)
try:
    from robojudo.config.config_manager import ConfigManager
    from robojudo.environment.g1_loco_env import _load_g1_loco_bridge
    from robojudo.pipeline import pipeline_registry
finally:
    sys.meta_path.remove(_import_guard)


class TestV3Integration(unittest.TestCase):
    def setUp(self):
        guard = NoHardware()
        sys.meta_path.insert(0, guard)
        self.addCleanup(sys.meta_path.remove, guard)

    def config(self, name):
        return ConfigManager(config_name=name).get_cfg()

    def test_backpack_preserves_local_motion_and_selects_v3(self):
        cfg = self.config("g1_native_loco_mimic_23dof_backpack")
        self.assertEqual(cfg.pipeline_type, "G1NativeLocoMimicV3Pipeline")
        self.assertEqual(cfg.env.bridge_variant, "v3")
        self.assertEqual(len(cfg.env.dof.joint_names), 23)
        self.assertEqual(cfg.env.unitree.regular_fsm_id, 501)
        self.assertTrue(cfg.walkrun_on_start)
        self.assertTrue(cfg.return_to_regular_loco)
        self.assertEqual(cfg.mimic_policies[0].policy_name, "eight_cut_1_backpack")
        self.assertEqual(cfg.mimic_policies[0].max_timestep, 11252)
        self.assertTrue(Path(cfg.mimic_policies[0].policy_file).is_file())
        self.assertFalse(pipeline_registry.get(cfg.pipeline_type).exit_on_frame_drop)

    def test_new_29dof_uses_local_model_and_independent_pipeline(self):
        cfg = self.config("g1_native_loco_mimic_29dof")
        self.assertEqual(cfg.env.bridge_variant, "v3")
        self.assertEqual(len(cfg.env.dof.joint_names), 29)
        self.assertEqual(cfg.env.unitree.regular_fsm_id, 501)
        self.assertEqual(cfg.mimic_policies[0].policy_name, "g1_dance2")
        self.assertEqual(cfg.mimic_policies[0].max_timestep, 1000)
        self.assertTrue(Path(cfg.mimic_policies[0].policy_file).is_file())

    def test_legacy_29dof_routes_keep_original_pipeline(self):
        for name, variant, normal in [("g1_native_loco_mimic_legacy", "legacy", True),
                                      ("g1_native_loco_mimic_v2", "v2", False)]:
            with self.subTest(name=name):
                cfg = self.config(name)
                self.assertEqual(cfg.pipeline_type, "G1NativeLocoMimicPipeline")
                self.assertEqual(cfg.env.bridge_variant, variant)
                self.assertEqual(cfg.normal_on_motion_done, normal)
                self.assertEqual(cfg.mimic_policies[0].policy_name, "g1_dance2")
                self.assertTrue(getattr(pipeline_registry.get(cfg.pipeline_type), "exit_on_frame_drop", True))

    def test_existing_launch_name_enables_v3_and_preserves_robot_settings(self):
        cfg = self.config("g1_native_loco_mimic")
        old = self.config("g1_native_loco_mimic_legacy")
        self.assertEqual(cfg.pipeline_type, "G1NativeLocoMimicV3Pipeline")
        self.assertIs(pipeline_registry.get(cfg.pipeline_type),
                      pipeline_registry.get("G1NativeLocoMimicV3Pipeline"))
        self.assertEqual(cfg.env.bridge_variant, "v3")
        self.assertEqual(len(cfg.env.dof.joint_names), 29)
        self.assertEqual(cfg.env.unitree.regular_fsm_id, 501)
        self.assertTrue(cfg.walkrun_on_start)
        self.assertTrue(cfg.return_to_regular_loco)
        self.assertFalse(cfg.normal_on_motion_done)
        self.assertFalse(cfg.env.unitree.enable_normal_mode_switch)
        self.assertFalse(pipeline_registry.get(cfg.pipeline_type).exit_on_frame_drop)
        self.assertEqual(cfg.ctrl, old.ctrl)
        self.assertEqual(cfg.mimic_policies, old.mimic_policies)
        self.assertEqual(cfg.env.dof, old.env.dof)
        self.assertEqual(cfg.env.unitree.net_if, old.env.unitree.net_if)
        self.assertEqual(cfg.env.forward_kinematic, old.env.forward_kinematic)
        self.assertEqual(cfg.env.torso_name, old.env.torso_name)
        self.assertTrue(cfg.env.update_with_fk)
        self.assertTrue(Path(cfg.env.forward_kinematic.xml_path).is_file())

    def test_v3_rejects_legacy_bridge_route(self):
        cfg = self.config("g1_native_loco_mimic_23dof_backpack")
        values = cfg.model_dump()
        values["env"] = cfg.env.model_copy(update={"bridge_variant": "legacy"})
        with self.assertRaisesRegex(ValueError, "requires bridge_variant"):
            type(cfg).model_validate(values)

    def test_bridge_selection_and_version_check_without_controllers(self):
        modules = {}
        for name in ["g1_loco_bridge", "g1_loco_bridge_v2", "g1_loco_bridge_23dof", "g1_loco_bridge_29dof"]:
            module = ModuleType(name)
            module.MIMIC_SAFETY_REVISION = 2026110310
            module.AuthorityState = object()
            module.G1LocoController = object()
            module.RobotState = object()
            module.SportState = object()
            modules[name] = module
        with patch.dict(sys.modules, modules):
            for dofs, variant, package in [(23, "v3", "g1_loco_bridge_23dof"),
                                           (29, "v3", "g1_loco_bridge_29dof"),
                                           (29, "legacy", "g1_loco_bridge"),
                                           (29, "v2", "g1_loco_bridge_v2")]:
                self.assertIs(_load_g1_loco_bridge(dofs, variant)[1], modules[package].G1LocoController)
            modules["g1_loco_bridge_29dof"].MIMIC_SAFETY_REVISION = 2026110305
            with self.assertRaisesRegex(RuntimeError, "MIMIC_SAFETY_REVISION mismatch"):
                _load_g1_loco_bridge(29, "v3")


if __name__ == "__main__":
    unittest.main()
