"""Trainable FADA modules, independent of robot control and optimizer scheduling."""
from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import torch

from .checkpoint import load_fada_policy_checkpoint
from .lora import FADALoRAConfig, inject_fada_idm_lora
from .model import FADAInverseDynamicsModel, FADAPlannerIDMPolicy
from .observation import (
    FADA_G1_ACTION_DIM, FADA_G1_ACTOR_OBS_DIM, FADA_G1_COMMAND_DIM,
    FADA_G1_STATE_DIM, FADA_G1_STATE_OBSERVATION_CONTRACT,
)
from .online_data import FADAInferenceSnapshot


@dataclass(frozen=True)
class FADATrainableRuntime:
    method: Literal["fada", "expert_difference"]
    policy: FADAPlannerIDMPolicy
    lora_config: FADALoRAConfig
    base_idm: FADAInverseDynamicsModel | None

    @torch.no_grad()
    def infer(self, observation_history, action_history, command) -> torch.Tensor:
        return self.policy(observation_history, action_history, command).action.detach()

    @torch.no_grad()
    def infer_with_snapshot(self, observation_history, action_history, command) -> FADAInferenceSnapshot:
        output = self.policy(observation_history, action_history, command)
        return FADAInferenceSnapshot(
            observation_history, action_history, command, output.predicted_future, output.action,
        )


def prepare_fada_trainable_runtime(cfg_policy, *, device: str | torch.device = "cpu") -> FADATrainableRuntime:
    """Load one explicit paired PT checkpoint; do not create hardware or an optimizer.

    Forward execution is graph-free. A learner may call runtime.policy.idm with
    autograd enabled, choosing train/eval mode according to its objective.
    """
    online = cfg_policy.online_adaptation
    if online.method not in ("fada", "expert_difference"):
        raise ValueError("trainable FADA runtime requires method=fada or expert_difference")
    path = Path(cfg_policy.policy_file)
    if path.suffix.lower() not in (".pt", ".pth"):
        raise ValueError("online FADA requires a paired PyTorch .pt/.pth checkpoint, not ONNX/TorchScript")
    if not path.is_file():
        raise FileNotFoundError(f"FADA checkpoint not found: {path}")
    lora_config = FADALoRAConfig(
        rank=online.lora_rank, alpha=online.lora_alpha, dropout=online.lora_dropout,
    )
    loaded = load_fada_policy_checkpoint(path, device=device)
    policy = loaded.policy
    config = policy.config
    expected = (
        FADA_G1_STATE_DIM, FADA_G1_ACTION_DIM, FADA_G1_COMMAND_DIM,
        int(cfg_policy.history_length), int(cfg_policy.prediction_horizon),
        FADA_G1_STATE_OBSERVATION_CONTRACT,
    )
    observed = (
        config.obs_dim, config.action_dim, config.command_dim,
        config.history_length, config.prediction_horizon, config.observation_contract,
    )
    if observed != expected:
        raise ValueError(f"FADA runtime contract mismatch: expected={expected}, got={observed}")
    if (cfg_policy.expected_obs_dim != FADA_G1_ACTOR_OBS_DIM
            or cfg_policy.expected_action_dim != FADA_G1_ACTION_DIM
            or cfg_policy.observation_contract != FADA_G1_STATE_OBSERVATION_CONTRACT):
        raise ValueError("online FADA requires the unchanged 98->66 state and 29-action deployment contract")
    if cfg_policy.action_scale != 1.0 or cfg_policy.action_beta != 1.0 or cfg_policy.action_clip is not None:
        raise ValueError("FADA playback forbids action scaling, smoothing, and clipping")
    base_idm = None
    if online.method == "expert_difference":
        base_idm = copy.deepcopy(policy.idm).eval().requires_grad_(False)
    adapted = inject_fada_idm_lora(policy, lora_config)
    return FADATrainableRuntime(online.method, adapted.policy.eval(), adapted.lora_config, base_idm)
