from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator

from robojudo.config import ASSETS_DIR
from robojudo.config.config_class import Config
from robojudo.tools.tool_cfgs import DoFConfig

from .g1_unilab_policy_cfg import G1UniLabDistillDoF, G1UniLabPolicyCfg


class FADAOnlineAdaptationCfg(Config):
    method: Literal["none", "fada", "expert_difference"] = "none"
    execution_mode: Literal["async", "sync"] = "async"
    lora_rank: int = Field(default=8, gt=0, strict=True)
    lora_alpha: float = Field(default=16.0, gt=0.0, allow_inf_nan=False)
    lora_dropout: float = Field(default=0.05, ge=0.0, lt=1.0, allow_inf_nan=False)
    update_interval: int = Field(default=10, gt=0, strict=True)
    window_size: int = Field(default=10, gt=0, strict=True)
    warmup_steps: int = Field(default=40, ge=0, strict=True)
    learning_rate: float = Field(default=1e-4, gt=0.0, allow_inf_nan=False)
    max_grad_norm: float | None = Field(default=1.0, gt=0.0, allow_inf_nan=False)
    lambda_forget: float = Field(default=0.99, gt=0.0, le=1.0, allow_inf_nan=False)
    gamma: float = Field(default=0.05, ge=0.0, allow_inf_nan=False)
    sigma: float = Field(default=1e-4, ge=0.0, allow_inf_nan=False)
    max_sensor_age_s: float | None = Field(default=None, gt=0.0, allow_inf_nan=False)
    learner_device: str = Field(default="cpu", min_length=1)
    learner_threads: int = Field(default=1, gt=0, strict=True)
    pending_capacity: int = Field(default=40, gt=0, strict=True)
    startup_timeout_s: float = Field(default=30.0, gt=0.0, allow_inf_nan=False)
    update_timeout_s: float = Field(default=5.0, gt=0.0, allow_inf_nan=False)
    close_timeout_s: float = Field(default=1.0, gt=0.0, allow_inf_nan=False)

    @model_validator(mode="after")
    def check_legacy_capacity(self):
        if self.execution_mode == "sync" and self.method != "fada":
            raise ValueError("synchronous updates require the legacy FADA method")
        if self.method == "fada" and self.pending_capacity < self.update_interval:
            raise ValueError("legacy pending_capacity must cover at least one update_interval")
        return self


class G1FADAPlannerIDMPolicyCfg(G1UniLabPolicyCfg):
    """Native FADA Planner-IDM deployment contract."""

    policy_type: str = "FADAPlannerIDMPolicyAdapter"
    policy_name: str = "planner_idm_v022"
    checkpoint_filename: str = "g1_hang_book_real.onnx" # "planner_idm_close_some_dr_v001.onnx"
    checkpoint_path: str | None = Field(default=None, min_length=1)
    online_adaptation: FADAOnlineAdaptationCfg = FADAOnlineAdaptationCfg()
    expected_obs_dim: int = 98
    expected_action_dim: int = 29
    history_length: int = 30
    prediction_horizon: int = 6
    observation_contract: str = "g1_fada_state_v2"
    gait_phase_enabled: bool = True
    fixed_gait_phase: list[float] = [0.0, 3.141592653589793]
    freeze_phase_during_dry_run: bool = False
    preserve_state_during_dry_run: bool = False
    # Keep keyboard commands inside the close_some_dr_v001 training ranges.
    # Forward uses the curriculum's walking command; lateral and yaw use the
    # source task limits.
    keyboard_command_magnitudes: list[float] = [0.6, 0.4, 0.8]
    # v022 inherits the FADA distill MuJoCo control contract, which maps raw
    # action chunks directly to joint-position offsets.
    execution_action_scale: float = 1.0
    action_scale: float = 1.0
    action_beta: float = 1.0
    action_clip: float | None = None
    obs_dof: DoFConfig = G1UniLabDistillDoF()
    action_dof: DoFConfig = obs_dof

    @property
    def policy_file(self) -> str:
        if self.checkpoint_path is not None:
            return Path(self.checkpoint_path).expanduser().as_posix()
        return (
            ASSETS_DIR
            / f"models/{self.robot}/fada/{self.policy_name}/{self.checkpoint_filename}"
        ).as_posix()
