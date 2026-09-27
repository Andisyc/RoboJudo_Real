from __future__ import annotations

from typing import Literal

from robojudo.environment.env_cfgs import UnitreeEnvCfg

from .g1_env_cfg import G1EnvCfg
from .g1_real_env_cfg import G1UnitreeCfg


class G1LocoUnitreeCfg(G1UnitreeCfg):
    """Configuration consumed by the RoboJuDo-owned g1_loco_bridge."""

    user_lowcmd_topic: str = "rt/user_lowcmd"
    sdk_timeout: float = 5.0
    fsm_confirm_timeout: float = 1.0


class G1LocoEnvCfg(G1EnvCfg, UnitreeEnvCfg):
    """Real G1 environment that can hand control to and from Unitree loco."""

    env_type: str = "G1LocoEnv"
    unitree: G1LocoUnitreeCfg = G1LocoUnitreeCfg(
        net_if="eth0",
        robot="g1",
        msg_type="hg",
        enable_odometry=True,
    )
    odometry_type: Literal["NONE", "DUMMY", "UNITREE", "ZED"] = "UNITREE"
    joint2motor_idx: list[int] | None = None
