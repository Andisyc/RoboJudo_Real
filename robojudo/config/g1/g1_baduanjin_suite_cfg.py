from copy import deepcopy
from robojudo.config import cfg_registry
from robojudo.pipeline.pipeline_cfgs import G1BaduanjinSuitePipelineCfg
from .g1_native_loco_mimic_cfg import g1_native_loco_mimic_23dof_backpack_cut3_fsm500
from .env.g1_loco_env_cfg import G1LocoEnvCfg
from .policy.g1_beyondmimic_policy_cfg import G1BeyondMimic23DoFPolicyCfg
from robojudo.controller.ctrl_cfgs import UnitreeCtrlCfg


@cfg_registry.register
class g1_baduanjin_23dof_suite(G1BaduanjinSuitePipelineCfg):
    """One process; automatic cut1 -> cut2 -> cut3 with live policy transitions."""
    env: G1LocoEnvCfg = g1_native_loco_mimic_23dof_backpack_cut3_fsm500.model_fields['env'].default.model_copy(deep=True)
    ctrl: list[UnitreeCtrlCfg] = deepcopy(g1_native_loco_mimic_23dof_backpack_cut3_fsm500.model_fields['ctrl'].default)
    entry_pose_transition_s: float = 3.
    exit_pose_transition_s: float = 0.
    segment_transition_s: float = 3.
    mimic_policies: list[G1BeyondMimic23DoFPolicyCfg] = [
        G1BeyondMimic23DoFPolicyCfg(policy_name='cut1_leftsmall_i80_20261112',without_state_estimator=False,max_timestep=8627),
        G1BeyondMimic23DoFPolicyCfg(policy_name='eight_cut_2_v010_1.2x',without_state_estimator=False,max_timestep=5960),
        G1BeyondMimic23DoFPolicyCfg(policy_name='cut3_jointv1_pitch8_rightless_20261112',without_state_estimator=False,max_timestep=5299),
    ]


@cfg_registry.register
class g1_baduanjin_23dof_select(g1_baduanjin_23dof_suite):
    """Same three models; return to native after each segment, process stays alive."""
    auto_sequence: bool = False
