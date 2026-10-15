from pathlib import Path
import xml.etree.ElementTree as ET

try:
    import mujoco
except ModuleNotFoundError:  # Structural fallback for lightweight local environments.
    mujoco = None


ROOT = Path(__file__).resolve().parents[1]


def test_g1_23dof_backpack_asset_and_pipeline_contract() -> None:
    base_path = ROOT / "assets/robots/g1/g1_23dof_rev_1_0.xml"
    backpack_path = ROOT / "assets/robots/g1/g1_23dof_backpack.xml"
    if mujoco is not None:
        base_model = mujoco.MjModel.from_xml_path(str(base_path))
        backpack_model = mujoco.MjModel.from_xml_path(str(backpack_path))
        backpack_body_id = mujoco.mj_name2id(
            backpack_model, mujoco.mjtObj.mjOBJ_BODY, "thor_backpack_link"
        )
        actuator_names = [
            mujoco.mj_id2name(backpack_model, mujoco.mjtObj.mjOBJ_ACTUATOR, index)
            for index in range(backpack_model.nu)
        ]
        assert backpack_body_id >= 0
        assert abs(backpack_model.body_mass[backpack_body_id] - 0.8) < 1e-12
        assert abs(
            (backpack_model.body_mass.sum() - base_model.body_mass.sum()) - 0.8
        ) < 1e-12
    else:
        base_root = ET.parse(base_path).getroot()
        backpack_root = ET.parse(backpack_path).getroot()
        backpack_body = backpack_root.find(".//body[@name='thor_backpack_link']")
        assert backpack_body is not None
        assert float(backpack_body.find("inertial").get("mass")) == 0.8
        base_mass = sum(float(node.get("mass")) for node in base_root.findall(".//inertial"))
        backpack_mass = sum(float(node.get("mass")) for node in backpack_root.findall(".//inertial"))
        assert abs((backpack_mass - base_mass) - 0.8) < 1e-12
        actuator_names = [node.get("name") for node in backpack_root.findall("./actuator/motor")]

    base_actuator_names = [node.get("name") for node in ET.parse(base_path).findall("./actuator/motor")]
    assert len(actuator_names) == 23
    assert actuator_names == base_actuator_names

    sim_cfg_source = (ROOT / "robojudo/config/g1/g1_cfg.py").read_text()
    real_cfg_source = (ROOT / "robojudo/config/g1/g1_native_loco_mimic_cfg.py").read_text()
    env_cfg_source = (ROOT / "robojudo/config/g1/env/g1_env_cfg.py").read_text()
    assert "class G1_23BackpackEnvCfg" in env_cfg_source
    assert "class g1_beyondmimic_23dof_backpack" in sim_cfg_source
    assert "class g1_native_loco_mimic_23dof_backpack" in real_cfg_source
    assert 'policy_name="eight_cut_1_backpack"' in sim_cfg_source
    assert 'policy_name="eight_cut_1_backpack"' in real_cfg_source


if __name__ == "__main__":
    test_g1_23dof_backpack_asset_and_pipeline_contract()
    print("G1 backpack contract: OK")
