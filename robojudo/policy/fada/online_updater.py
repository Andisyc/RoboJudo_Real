# Adapted from TTT online_updater.py (revisions c4ff3bf6 and 0f8a410b).
# Licensed under Apache-2.0; see LICENSE-TTT.
"""Standalone LoRA objectives; scheduling and collection acknowledgements are separate."""
from __future__ import annotations

import math
from collections.abc import Sequence

import torch
from torch.nn import functional as F

from .lora import assert_fada_adaptation_parameter_ownership, fada_adapter_named_parameters
from .model import FADAInverseDynamicsModel
from .online_data import FADACausalWindow, FADAOnlineSample, FADATrajectoryPair
from .online_runtime import FADATrainableRuntime


def _prepare_adapters(runtime, method, lr, max_grad_norm):
    if runtime.method != method:
        raise ValueError(f"{method} updater requires a matching trainable runtime")
    if not math.isfinite(lr) or lr <= 0.0:
        raise ValueError("lr must be finite and positive")
    if max_grad_norm is not None and (
        not math.isfinite(max_grad_norm) or max_grad_norm <= 0.0
    ):
        raise ValueError("max_grad_norm must be finite and positive or None")
    assert_fada_adaptation_parameter_ownership(runtime.policy, runtime.lora_config)
    return dict(fada_adapter_named_parameters(runtime.policy))


def _require_frozen_base(base_idm: FADAInverseDynamicsModel | None) -> None:
    if base_idm is None or any(module.training for module in base_idm.modules()) or any(
        parameter.requires_grad for parameter in base_idm.parameters()
    ):
        raise ValueError("base IDM must be independently prepared, frozen and in eval mode")


@torch.no_grad()
def build_expert_samples(
    base_idm: FADAInverseDynamicsModel,
    trajectories: Sequence[FADATrajectoryPair],
    *,
    gamma: float = 0.05,
) -> tuple[FADAOnlineSample, ...]:
    """Freeze snapshot + gamma * (base(nominal) - base(realized)) once per batch."""
    if not trajectories:
        raise ValueError("expert labels require a non-empty trajectory window")
    if not math.isfinite(gamma) or gamma < 0.0:
        raise ValueError("gamma must be finite and non-negative")
    _require_frozen_base(base_idm)
    reference = next(base_idm.parameters())
    snapshots = tuple(pair.snapshot for pair in trajectories)
    observations = torch.cat([snapshot.observation_history for snapshot in snapshots]).to(reference)
    actions = torch.cat([snapshot.action_history for snapshot in snapshots]).to(reference)
    nominal = torch.cat([snapshot.nominal_future for snapshot in snapshots]).to(reference)
    realized = torch.cat([pair.realized_future for pair in trajectories]).to(reference)
    executed = torch.cat([snapshot.action_snapshot for snapshot in snapshots]).to(reference)
    base_nominal = base_idm(observations, actions, nominal)[:, 0]
    base_realized = base_idm(observations, actions, realized)[:, 0]
    if base_nominal.shape != executed.shape or base_realized.shape != executed.shape:
        raise ValueError("expert predictions must match first-action snapshots")
    targets = executed + gamma * (base_nominal - base_realized)
    return tuple(
        FADAOnlineSample(pair, target[None].to(pair.snapshot.action_snapshot))
        for pair, target in zip(trajectories, targets)
    )


class FADALegacyUpdater:
    """Legacy weighted first-action MSE, including train-mode LoRA dropout.

    Mature windows belong here, not in a second collector cache. An ordinary
    episode reset must retain this object, its Adam state and completed windows.
    """

    def __init__(
        self,
        runtime: FADATrainableRuntime,
        *,
        lambda_forget: float = 0.99,
        lr: float = 1e-4,
        max_grad_norm: float | None = 1.0,
    ) -> None:
        if not 0.0 < lambda_forget <= 1.0:
            raise ValueError("lambda_forget must be in (0, 1]")
        adapters = _prepare_adapters(runtime, "fada", lr, max_grad_norm)
        self.policy = runtime.policy
        self.lambda_forget = float(lambda_forget)
        self.max_grad_norm = max_grad_norm
        self._parameters = tuple(adapters.values())
        self.optimizer = torch.optim.Adam(self._parameters, lr=lr, weight_decay=0.0)
        self._window_buffer: list[FADACausalWindow] = []

    def append_window(self, window: FADACausalWindow) -> None:
        self._window_buffer.append(window)

    @torch.enable_grad()
    def compute_loss(self) -> torch.Tensor:
        if not self._window_buffer:
            raise ValueError("legacy regression requires at least one mature window")
        reference = self._parameters[0]
        count = len(self._window_buffer)
        weights = [self.lambda_forget ** (count - 1 - i) for i in range(count)]
        modes = tuple((module, module.training) for module in self.policy.idm.modules())
        self.policy.idm.train()
        try:
            total = reference.new_zeros(())
            # Preserve donor forward order and dropout draws, rather than batch-vectorizing.
            for window, weight in zip(self._window_buffer, weights):
                predicted = self.policy.idm(
                    window.observation_history.to(reference),
                    window.action_history.to(reference),
                    window.realized_future.to(reference),
                )[:, 0]
                target = window.executed_action_chunk[:, 0].to(reference)
                total = total + weight * F.mse_loss(predicted, target)
            return total / sum(weights)
        finally:
            for module, training in modes:
                module.training = training

    @torch.enable_grad()
    def update(self) -> dict[str, float | int] | None:
        """Take one step for any non-empty batch; clear only after success."""
        if not self._window_buffer:
            return None
        self.optimizer.zero_grad(set_to_none=True)
        total = self.compute_loss()
        if not bool(torch.isfinite(total)):
            raise ValueError("legacy adaptation loss must be finite")
        total.backward()
        if self.max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(self._parameters, self.max_grad_norm)
        self.optimizer.step()
        stats = {"loss": float(total.detach()), "num_windows": len(self._window_buffer)}
        self._window_buffer.clear()
        return stats


class FADAExpertUpdater:
    """Regress fixed expert labels against nominal conditions with a deployment anchor.

    Create once before learning. Do not recreate theta_base/Adam on episode reset.
    This updater neither retains samples nor acknowledges the collector's window.
    """

    def __init__(
        self,
        runtime: FADATrainableRuntime,
        *,
        sigma: float = 1e-4,
        lr: float = 1e-4,
        max_grad_norm: float | None = 1.0,
    ) -> None:
        if not math.isfinite(sigma) or sigma < 0.0:
            raise ValueError("sigma must be finite and non-negative")
        self._adapters = _prepare_adapters(runtime, "expert_difference", lr, max_grad_norm)
        _require_frozen_base(runtime.base_idm)
        self.policy = runtime.policy
        self.sigma = float(sigma)
        self.max_grad_norm = max_grad_norm
        self._parameters = tuple(self._adapters.values())
        self.theta_base = {
            name: parameter.detach().clone() for name, parameter in self._adapters.items()
        }
        self.optimizer = torch.optim.Adam(self._parameters, lr=lr, weight_decay=0.0)

    @torch.enable_grad()
    def compute_loss(
        self, samples: Sequence[FADAOnlineSample],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if not samples:
            raise ValueError("expert regression requires a non-empty sample window")
        reference = self._parameters[0]
        snapshots = tuple(sample.trajectory.snapshot for sample in samples)
        observations = torch.cat([snapshot.observation_history for snapshot in snapshots]).to(reference)
        actions = torch.cat([snapshot.action_history for snapshot in snapshots]).to(reference)
        nominal = torch.cat([snapshot.nominal_future for snapshot in snapshots]).to(reference)
        targets = torch.cat([sample.target_action for sample in samples]).to(reference)
        modes = tuple((module, module.training) for module in self.policy.idm.modules())
        self.policy.idm.eval()
        try:
            predicted = self.policy.idm(observations, actions, nominal)[:, 0]
            if predicted.shape != targets.shape:
                raise ValueError("expert targets must match first-action predictions")
            regression = 0.5 * (predicted - targets).square().sum(dim=-1).mean()
            anchor = reference.new_zeros(())
            for name, parameter in self._adapters.items():
                anchor = anchor + (parameter - self.theta_base[name]).square().sum()
            anchor = (0.5 * self.sigma) * anchor
            return regression + anchor, regression, anchor
        finally:
            for module, training in modes:
                module.training = training

    @torch.enable_grad()
    def update(self, samples: Sequence[FADAOnlineSample]) -> dict[str, float | int]:
        self.optimizer.zero_grad(set_to_none=True)
        total, regression, anchor = self.compute_loss(samples)
        if not bool(torch.isfinite(total)):
            raise ValueError("expert adaptation loss must be finite")
        total.backward()
        if self.max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(self._parameters, self.max_grad_norm)
        self.optimizer.step()
        return {
            "loss": float(total.detach()),
            "regression_loss": float(regression.detach()),
            "anchor_loss": float(anchor.detach()),
            "num_samples": len(samples),
        }
