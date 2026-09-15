from __future__ import annotations

from robojudo.config import cfg_registry
from robojudo.controller.ctrl_cfgs import UnitreeCtrlCfg
from robojudo.pipeline.pipeline_cfgs import G1NativeLocoMimicPipelineCfg

from .env.g1_loco_env_cfg import G1LocoEnvCfg, G1LocoUnitreeCfg
from .policy.g1_beyondmimic_policy_cfg import G1BeyondMimicPolicyCfg


@cfg_registry.register
class g1_native_loco_mimic(G1NativeLocoMimicPipelineCfg):
    """Unitree native WALKRUN with explicit handoff to RoboJuDo dances."""

    env: G1LocoEnvCfg = G1LocoEnvCfg(
        unitree=G1LocoUnitreeCfg(net_if="eth0"),
    )
    ctrl: list[UnitreeCtrlCfg] = [
        UnitreeCtrlCfg(
            combination_init_buttons=[],
            triggers={
                "A": "[SHUTDOWN]",
                "Select": "[POLICY_LOCO]",
                "Start": "[POLICY_MIMIC]",
                "R1": "[POLICY_SWITCH],NEXT",
                "L1": "[POLICY_SWITCH],LAST",
            },
        )
    ]
    mimic_policies: list[G1BeyondMimicPolicyCfg] = [
        G1BeyondMimicPolicyCfg(
            policy_name="Dance_wose", without_state_estimator=True
        ),
        G1BeyondMimicPolicyCfg(
            policy_name="Violin", without_state_estimator=False, max_timestep=500
        ),
        G1BeyondMimicPolicyCfg(
            policy_name="Waltz", without_state_estimator=False, max_timestep=850
        ),
    ]
    entry_transition_steps: int = 100
    do_safety_check: bool = True
