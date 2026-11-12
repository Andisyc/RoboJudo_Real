"""Read a native deployment config without loading a bridge or creating an env."""
from __future__ import annotations

import argparse
import importlib.abc
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]


class NoHardware(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.startswith(("g1_loco_bridge", "unitree_sdk2py", "unitree_cpp", "zed_proxy")):
            raise RuntimeError(f"Offline configuration check blocked hardware import: {fullname}")
        return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-c", "--config", default="g1_native_loco_mimic")
    parser.add_argument("--net-if", help="Require the recorder interface to match the control config")
    args = parser.parse_args()
    sys.meta_path.insert(0, NoHardware())
    sys.path.insert(0, str(ROOT))

    from robojudo.config.config_manager import ConfigManager

    cfg = ConfigManager(config_name=args.config).get_cfg()
    if cfg.pipeline_type not in ("G1NativeLocoMimicPipeline", "G1BaduanjinSuitePipeline"):
        raise SystemExit("错误：当前配置不是 G1 native-loco 配置。")
    if cfg.env.is_sim or cfg.env.env_type != "G1LocoEnv":
        raise SystemExit("错误：当前配置不是 G1 native 真机配置。")
    if args.net_if and args.net_if != cfg.env.unitree.net_if:
        raise SystemExit(f"错误：记录网卡 {args.net_if} 与控制网卡 {cfg.env.unitree.net_if} 不一致。")

    dofs = len(cfg.env.dof.joint_names)
    bridge = "g1_loco_bridge"
    print(f"配置: {args.config}; pipeline: {cfg.pipeline_type}")
    print(f"网卡: {cfg.env.unitree.net_if}; DoF: {dofs}; bridge: {bridge}; 要求版本: 2026110310")
    print(f"regular_fsm_id: {cfg.env.unitree.regular_fsm_id}; walkrun_on_start: {cfg.walkrun_on_start}")
    for policy in cfg.mimic_policies:
        model = Path(policy.policy_file)
        if not model.is_file():
            raise SystemExit(f"错误：模型不存在：{model}")
        print(f"模型: {model}; max_timestep: {policy.max_timestep}")
        if policy.max_timestep <= 0:
            raise SystemExit("错误：请填写 max_timestep，才能在动作结束后自动交还。")
    print("配置和模型路径检查通过；未加载硬件 SDK、未创建环境。")


if __name__ == "__main__":
    main()
