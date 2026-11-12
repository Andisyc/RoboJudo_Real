from __future__ import annotations

from robojudo.config import cfg_registry
from robojudo.controller.ctrl_cfgs import UnitreeCtrlCfg
from robojudo.pipeline.pipeline_cfgs import FixedHipBiasCfg, G1NativeLocoMimicPipelineCfg

from .env.g1_env_cfg import G1_23DoF
from .env.g1_loco_env_cfg import G1LocoEnvCfg, G1LocoUnitreeCfg
from .policy.g1_beyondmimic_policy_cfg import (
    G1BeyondMimic23DoFPolicyCfg,
    G1BeyondMimicPolicyCfg,
)


@cfg_registry.register
class g1_native_loco_mimic(G1NativeLocoMimicPipelineCfg):
    """Unitree native WALKRUN with explicit handoff to RoboJuDo dances."""

    env: G1LocoEnvCfg = G1LocoEnvCfg(
        unitree=G1LocoUnitreeCfg(net_if="enP8p1s0", regular_fsm_id=501),
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
            policy_name="g1_dance2", 
            without_state_estimator=False, 
            max_timestep=1000
        ),
        # G1BeyondMimicPolicyCfg(
        #     policy_name="Dance_wose", without_state_estimator=True
        # ),
        # G1BeyondMimicPolicyCfg(
        #     policy_name="Violin", without_state_estimator=False, max_timestep=500
        # ),
        # G1BeyondMimicPolicyCfg(
        #     policy_name="Waltz", without_state_estimator=False, max_timestep=850
        # ),
    ]
    do_safety_check: bool = True


@cfg_registry.register
class g1_native_loco_mimic_v2(g1_native_loco_mimic):
    """Compatibility config; uses the same native controller without auto-switching."""

    return_to_regular_loco: bool = True
    walkrun_on_start: bool = True
    env: G1LocoEnvCfg = G1LocoEnvCfg(
        unitree=G1LocoUnitreeCfg(net_if="enP8p1s0"),
    )


@cfg_registry.register
class g1_native_loco_mimic_23dof(G1NativeLocoMimicPipelineCfg):
    """Unitree native WALKRUN with explicit handoff to RoboJuDo 23DoF dances."""

    env: G1LocoEnvCfg = G1LocoEnvCfg(
        unitree=G1LocoUnitreeCfg(net_if="enP8p1s0", regular_fsm_id=501),
        dof=G1_23DoF(),
        forward_kinematic=None,
        update_with_fk=False,
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
    mimic_policies: list[G1BeyondMimic23DoFPolicyCfg] = [
        G1BeyondMimic23DoFPolicyCfg(
            policy_name="eight_cut_1", without_state_estimator=False
        ),
    ]
    do_safety_check: bool = True


@cfg_registry.register
class g1_native_loco_mimic_23dof_backpack(G1NativeLocoMimicPipelineCfg):
    """Unitree native WALKRUN handoff to the backpack-trained 23DoF policy."""

    # Candidate first-raise window. Align video to query frames before enabling.
    # Deliberately preserve mimic_policies below; a different model must not match.
    fixed_hip_bias: FixedHipBiasCfg = FixedHipBiasCfg(
        enabled=False,
        policy_name="eight_cut_1_v034_1.2x",
        hip_pitch_deg=0.5,
        ramp_in_start=1100,
        ramp_in_end=1140,
        ramp_out_start=1310,
        ramp_out_end=1350,
    )

    env: G1LocoEnvCfg = G1LocoEnvCfg(
        unitree=G1LocoUnitreeCfg(net_if="enP8p1s0", regular_fsm_id=501),
        dof=G1_23DoF(),
        forward_kinematic=None,
        update_with_fk=False,
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
    mimic_policies: list[G1BeyondMimic23DoFPolicyCfg] = [
        G1BeyondMimic23DoFPolicyCfg(
            policy_name="eight_cut_1_backpack",
            without_state_estimator=False,
            max_timestep=11252,
        ),
    ]
    do_safety_check: bool = True


@cfg_registry.register
class g1_native_loco_mimic_23dof_backpack_cut3_recovery(g1_native_loco_mimic_23dof_backpack):
    """Offline-tested Cut3 recovery candidate; preserves the original model list."""

    mimic_policies: list[G1BeyondMimic23DoFPolicyCfg] = [
        G1BeyondMimic23DoFPolicyCfg(
            policy_name="cut3_recovery_r1_i80",
            without_state_estimator=False,
            max_timestep=5299,
        ),
    ]


@cfg_registry.register
class g1_native_loco_mimic_23dof_backpack_cut3_fsm500(g1_native_loco_mimic_23dof_backpack_cut3_recovery):
    """Isolate the 23DoF FSM request change; original recovery ONNX and no ramps."""

    env: G1LocoEnvCfg = g1_native_loco_mimic_23dof_backpack.model_fields["env"].default.model_copy(deep=True)
    env.unitree.regular_fsm_id = 500


@cfg_registry.register
class g1_native_loco_mimic_23dof_backpack_cut3_pitch(g1_native_loco_mimic_23dof_backpack_cut3_fsm500):
    """Reference-only pitch candidate; preserves entry/exit control behavior."""
    mimic_policies: list[G1BeyondMimic23DoFPolicyCfg] = [
        G1BeyondMimic23DoFPolicyCfg(
            policy_name="cut3_pitch_ref4_20261112",
            without_state_estimator=False,
            max_timestep=5299,
        ),
    ]


@cfg_registry.register
class g1_native_loco_mimic_23dof_backpack_cut3_entry_transition(g1_native_loco_mimic_23dof_backpack_cut3_pitch):
    """Separate startup test: 3s reference transition, original ending handback."""

    entry_pose_transition_s: float = 3.0


@cfg_registry.register
class g1_native_loco_mimic_23dof_backpack_cut3_transition(g1_native_loco_mimic_23dof_backpack_cut3_pitch):
    """EXPERIMENTAL / OFFLINE ONLY: final standing tracking has not passed acceptance.

    Do not select this entry for the first hardware test. Use cut3_pitch instead.
    """

    entry_pose_transition_s: float = 3.0
    exit_pose_transition_s: float = 3.0
