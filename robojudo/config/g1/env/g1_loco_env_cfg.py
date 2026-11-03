from __future__ import annotations

from typing import Literal

from robojudo.environment.env_cfgs import UnitreeEnvCfg

from .g1_env_cfg import G1EnvCfg
from .g1_real_env_cfg import G1UnitreeCfg


class G1LocoUnitreeCfg(G1UnitreeCfg):
    """Configuration consumed by the RoboJuDo-owned g1_loco_bridge."""

    user_lowcmd_topic: str = "rt/user_lowcmd"
    enable_normal_mode_switch: bool = False
    sdk_timeout: float = 5.0
    fsm_confirm_timeout: float = 1.0
    # Wait for FSM 500/501 after each WALKRUN -> regular loco request; the
    # request is repeated up to regular_attempts times while still in WALKRUN.
    regular_confirm_timeout: float = 2.0
    regular_attempts: int = 3
    # FSM requested for regular loco: 500 (official Start()) or 501. unitree-001
    # (29DoF) ignores 500 from WALKRUN and switches on 501 (site test 2026-11-03).
    regular_fsm_id: Literal[500, 501] = 500
    # Start pressed in regular loco: request WALKRUN (801), wait this long per
    # request, re-request up to walkrun_attempts times while still in regular
    # loco. Any other FSM (e.g. damp) stops the switch without another request.
    walkrun_confirm_timeout: float = 2.0
    walkrun_attempts: int = 3


class G1LocoEnvCfg(G1EnvCfg, UnitreeEnvCfg):
    """Real G1 environment that can hand control to and from Unitree loco."""

    env_type: str = "G1LocoEnv"
    # "v3": the no-damp handoff bridge (23DoF: g1_loco_bridge_23dof, 29DoF: g1_loco_bridge_29dof).
    # "legacy" keeps the old routing (23DoF -> g1_loco_bridge_23dof, 29DoF -> g1_loco_bridge).
    bridge_variant: Literal["legacy", "v2", "v3"] = "legacy"
    unitree: G1LocoUnitreeCfg = G1LocoUnitreeCfg(
        net_if="eth0",
        robot="g1",
        msg_type="hg",
        enable_odometry=True,
    )
    odometry_type: Literal["NONE", "DUMMY", "UNITREE", "ZED"] = "UNITREE"
    joint2motor_idx: list[int] | None = None
