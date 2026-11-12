"""Check the fixed suite release without constructing an environment or SDK."""
import hashlib
import json
import sys
from pathlib import Path

from check_g1_native_config import NoHardware

sys.meta_path.insert(0, NoHardware())
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from robojudo.config.config_manager import ConfigManager
from robojudo.pipeline.rl_pipeline import PolicyWrapper
from robojudo.pipeline.g1_baduanjin_suite_pipeline import validate_blend_policies

MODELS = [
    ("cut1_scratch_v2_i15000_zero_translation", 8627, "2afc0f0cb58605b5abc7eb2c127148f9e5e0993e6dc4b54c8ff0540af155e53d"),
    ("eight_cut_2_v010_1.2x", 5960, "bbca0a34feab8f9f73f150003910190cfe9412ef91b4a00f0b65197a2a0312bc"),
    ("cut3_pitch_ref6_20261112", 5299, "47b75e8c34d8735d3bd788bfed90935e5e99d93f18f063dadcc7ac0d4966398c"),
]


def require(ok, message):
    if not ok:
        raise SystemExit(message)


def main():
    cfg = ConfigManager("g1_baduanjin_23dof_suite").get_cfg()
    manual = ConfigManager("g1_baduanjin_23dof_select").get_cfg()
    require(cfg.auto_sequence and not manual.auto_sequence, "Automatic/manual entry mismatch")
    require(cfg.env.dof.num_dofs == 23 and cfg.env.odometry_type == "UNITREE", "23DoF/UNITREE contract mismatch")
    require(cfg.env.unitree.regular_fsm_id == 500, "This 23DoF release requires FSM request 500")
    require(not cfg.fixed_hip_bias.enabled, "Old hip correction must remain disabled")
    require((cfg.entry_pose_transition_s, cfg.segment_transition_s, cfg.exit_pose_transition_s) == (3., 3., 0.), "Transition config mismatch")
    require(len(cfg.mimic_policies) == len(MODELS), "Expected three models")
    policies, results = [], []
    for index, (entry, expected) in enumerate(zip(cfg.mimic_policies, MODELS)):
        name, length, digest = expected
        require(entry.policy_name == name and entry.max_timestep == length, f"Wrong model/index at {index}")
        require(not entry.without_state_estimator, f"Changed input layout at {index}")
        actual = hashlib.sha256(Path(entry.policy_file).read_bytes()).hexdigest()
        require(actual == digest, f"SHA-256 mismatch: {name}; found {actual}")
        policy = PolicyWrapper(entry, cfg.env.dof, "cpu")
        policies.append(policy)
        require(policy.session.get_inputs()[0].shape == [1, 130], f"Wrong observation shape: {name}")
        require(policy.session.get_outputs()[0].shape == [1, 23], f"Wrong action shape: {name}")
        results.append(dict(index=index, model=name, sha256=actual, max_timestep=length))
    validate_blend_policies(policies, cfg.env.dof.joint_names)
    print(json.dumps(dict(models=results, regular_fsm_id=500, odometry_type="UNITREE", hardware_initialized=False), indent=2))
    print("Suite contract passed; no hardware SDK, no environment, no control commands.")


if __name__ == "__main__":
    main()
