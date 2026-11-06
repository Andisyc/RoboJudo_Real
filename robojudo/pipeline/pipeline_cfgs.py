from __future__ import annotations
import math
from typing import Union, List, Optional
from typing import Any

from pydantic import model_validator

from robojudo.config import Config
from robojudo.controller import CtrlCfg
from robojudo.environment import EnvCfg
from robojudo.policy import PolicyCfg
from robojudo.tools.debug_log import DebugCfg


class PipelineCfg(Config):
    pipeline_type: str  # name of the pipeline class
    # ===== Pipeline Config =====
    device: str = "cpu"

    debug: DebugCfg = DebugCfg()

    run_fullspeed: bool = False
    """If True, run the pipeline at full speed, ignoring the desired frequency"""

    do_safety_check: bool = False
    """
    If True, perform safety check after each step.
    We recommend enabling this, however if motion is very aggressive, you may disable it.
    """

class RlPipelineCfg(PipelineCfg):
    pipeline_type: str = "RlPipeline"

    # ===== Pipeline Config =====
    robot: str  # robot name, e.g. "g1"

    # py3.10 -> py3.8
    # env: EnvCfg | Any # py3.10
    env: Union[EnvCfg, Any] # py3.8
    
    ctrl: list[CtrlCfg | Any] = []
    policy: PolicyCfg | Any


class RlMultiPolicyPipelineCfg(PipelineCfg):
    pipeline_type: str = "RlMultiPolicyPipeline"

    # ===== Pipeline Config =====
    robot: str  # robot name, e.g. "g1"

    env: EnvCfg | Any
    ctrl: list[CtrlCfg | Any] = []

    policies: list[PolicyCfg | Any] = []
    """First policy as init, rest as extra policies, can be switched to"""


class RlLocoMimicPipelineCfg(PipelineCfg):
    pipeline_type: str = "RlLocoMimicPipeline"

    # ===== Pipeline Config =====
    robot: str  # robot name, e.g. "g1"

    env: EnvCfg | Any
    ctrl: list[CtrlCfg | Any] = []

    loco_policy: PolicyCfg | Any
    """LocoMotion policy, as init"""
    mimic_policies: list[PolicyCfg | Any] = []
    """MotionMimic policies, can be switched to"""

    # ===== Upper body override Config =====
    upper_dof_num: int = 0
    upper_dof_pos_default: list[float] | None = []
    """Default positions of the upper body DOFs"""
    upper_dof_override_indices: list[int] | None = []
    """Indices of the upper body DOFs to be overridden"""

    @model_validator(mode="after")
    def check_upper_dof(self):
        if self.upper_dof_pos_default is not None:
            if len(self.upper_dof_pos_default) != self.upper_dof_num:
                raise ValueError(
                    f"Length of upper_dof_pos_default ({len(self.upper_dof_pos_default)}) "
                    f"must be equal to upper_dof_num ({self.upper_dof_num})")
        
        if self.upper_dof_override_indices is not None:
            for idx in self.upper_dof_override_indices:
                if idx < -self.upper_dof_num or idx >= 0:
                    raise ValueError(
                        f"upper_dof_override_indices contains invalid index {idx}, "
                        f"must be in [-{self.upper_dof_num}, 0)")
        
        return self


class FixedHipBiasCfg(Config):
    """Human-selected, query-frame-based target offset; disabled by default."""

    enabled: bool = False
    policy_name: str = "eight_cut_1_v034_1.2x"
    hip_pitch_deg: float = 0.5
    ramp_in_start: int = 1100
    ramp_in_end: int = 1140
    ramp_out_start: int = 1310
    ramp_out_end: int = 1350

    @model_validator(mode="after")
    def check_fixed_hip_bias(self):
        if not self.policy_name.strip():
            raise ValueError("fixed_hip_bias requires a policy_name")
        if not math.isfinite(self.hip_pitch_deg) or not 0 <= self.hip_pitch_deg <= 0.5:
            raise ValueError("fixed_hip_bias.hip_pitch_deg must be finite and in [0, 0.5]")
        if not 0 <= self.ramp_in_start < self.ramp_in_end < self.ramp_out_start < self.ramp_out_end:
            raise ValueError("fixed_hip_bias requires 0 <= start < fade-in end < fade-out start < end")
        return self


class G1NativeLocoMimicPipelineCfg(PipelineCfg):
    """One native-loco/Mimic controller for both 23DoF and 29DoF G1."""

    pipeline_type: str = "G1NativeLocoMimicPipeline"
    robot: str = "g1"
    env: EnvCfg | Any
    ctrl: list[CtrlCfg | Any] = []
    mimic_policies: list[PolicyCfg | Any] = []
    return_to_regular_loco: bool = True
    walkrun_on_start: bool = True
    fixed_hip_bias: FixedHipBiasCfg = FixedHipBiasCfg()

    @model_validator(mode="after")
    def check_native_loco_mimic(self):
        if not self.mimic_policies:
            raise ValueError("At least one mimic policy is required")
        if self.return_to_regular_loco and not self.env.act:
            raise ValueError("return_to_regular_loco requires env.act=True")
        return self
