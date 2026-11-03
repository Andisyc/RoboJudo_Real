"""Offline replay of the A/Select stop logic with the real PolicyWrapper, ONNX and pipeline class.

No robot, SDK or network: the environment is replaced by perfect tracking of the
reference (measured joints = reference joints, mapped to env/motor order by name).
Checks:
  1. G1NativeLocoMimicV3Pipeline._at_stance agrees, frame by frame, with a stance check
     done directly on joint names;
  2. freezing the reference reaches the wrapped policy (timestep stops advancing).

Usage:
    python scripts/replay_stop_stance_offline.py [-c CONFIG] [--frames N]
"""
from __future__ import annotations

import argparse
import sys
import types
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from robojudo.config.config_manager import ConfigManager  # noqa: E402
from robojudo.pipeline.g1_native_loco_mimic_v3_pipeline import G1NativeLocoMimicV3Pipeline as P  # noqa: E402
from robojudo.pipeline.rl_pipeline import PolicyWrapper  # noqa: E402
from robojudo.tools.dof import DoFAdapter  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", default="g1_native_loco_mimic_23dof_backpack")
    parser.add_argument("--frames", type=int, default=None, help="default: the policy max_timestep")
    args = parser.parse_args()

    cfg = ConfigManager(config_name=args.config).get_cfg()
    cfg_policy = cfg.mimic_policies[0]
    wrapper = PolicyWrapper(cfg_policy, cfg.env.dof, "cpu")
    model_names = list(wrapper.policy.cfg_action_dof.joint_names)
    env_names = list(cfg.env.dof.joint_names)
    to_env = DoFAdapter(model_names, env_names)
    print(f"config {args.config}: model order == env order: {model_names == env_names}")

    pipeline = P.__new__(P)
    pipeline.policy_manager = types.SimpleNamespace(policy=wrapper)
    pipeline.env = types.SimpleNamespace(joint_names=env_names)
    pipeline._stance_cache = {}

    legs = [i for i, n in enumerate(model_names) if any(k in n for k in P.STANCE_JOINT_KEYS)]
    stances = [wrapper.reference_frame(cfg_policy.start_timestep)]
    if cfg_policy.max_timestep > 0:
        stances.append(wrapper.reference_frame(cfg_policy.max_timestep))
    frames = args.frames if args.frames is not None else cfg_policy.max_timestep
    hits = truth = missed = wrong = 0
    for t in range(frames + 1):
        q, dq, z = wrapper.reference_frame(t)
        wrapper.policy.command = {"joint_pos": q, "joint_vel": dq, "body_pos_w": np.array([[0.0, 0.0, z]])}
        env_data = {"dof_pos": to_env.fit(q), "dof_vel": to_env.fit(dq)}
        hit = bool(P._at_stance(pipeline, env_data))
        ok = bool(np.abs(dq[legs]).max() <= P.STANCE_REF_VEL and any(
            np.abs(q[legs] - s[0][legs]).max() <= P.STANCE_LEG_TOL and abs(z - s[2]) <= P.STANCE_HEIGHT_TOL
            for s in stances))
        hits += hit; truth += ok; missed += ok and not hit; wrong += hit and not ok
    print(f"frames 0-{frames}: stance by joint name {truth}, pipeline _at_stance {hits}, "
          f"missed {missed}, wrongly triggered {wrong}")

    wrapper.policy.timestep = 200
    wrapper.policy.play_speed = 1.0
    P._freeze_reference(pipeline)
    wrapper.policy.post_step_callback([])
    frozen = wrapper.policy.play_speed == 0.0 and wrapper.policy.timestep == 200
    print(f"freeze: wrapped play_speed={wrapper.policy.play_speed}, timestep 200 -> {wrapper.policy.timestep}")
    ok = missed == 0 and wrong == 0 and frozen
    print("RESULT:", "OK" if ok else "MISMATCH")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
