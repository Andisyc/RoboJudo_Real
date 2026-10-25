# Adapted from TTT online_data.py and windows.py (revision 0f8a410b).
# Licensed under Apache-2.0; see LICENSE-TTT.
"""Owned learning data; collecting and acknowledging real frames is separate."""
from __future__ import annotations

import math
from dataclasses import dataclass
from numbers import Integral
from typing import Literal

import torch


def _index(name: str, value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")


def _tensor(name: str, value: torch.Tensor, shape: tuple[int, ...], reference: torch.Tensor) -> None:
    if not isinstance(value, torch.Tensor) or tuple(value.shape) != shape:
        raise ValueError(f"{name} must have shape {shape}")
    if not value.is_floating_point() or value.dtype != reference.dtype or value.device != reference.device:
        raise ValueError(f"{name} must share floating dtype/device with observation_history")
    if not bool(torch.isfinite(value).all()):
        raise ValueError(f"{name} must be finite")


def _history(observations: torch.Tensor, actions: torch.Tensor) -> int:
    if observations.ndim != 3 or observations.shape[0] != 1 or observations.shape[1] < 1:
        raise ValueError("observation_history must have shape (1, H, 66), H > 0")
    history_length = observations.shape[1]
    _tensor("observation_history", observations, (1, history_length, 66), observations)
    _tensor("action_history", actions, (1, history_length, 29), observations)
    return history_length


def _future(value: torch.Tensor, reference: torch.Tensor) -> int:
    if value.ndim != 3 or value.shape[0] != 1 or value.shape[1] < 1:
        raise ValueError("future must have shape (1, K, 66), K > 0")
    horizon = value.shape[1]
    _tensor("future", value, (1, horizon, 66), reference)
    return horizon


@dataclass(frozen=True)
class FADAControlEvidence:
    """Source-tagged control metadata; missing hardware data never implies simulation."""

    episode_id: int
    control_step: int
    policy_version: int
    sensor_tick: int | None
    torso_imu_sequence: int | None
    policy_control_active: bool
    action_dispatched: bool
    lowstate_age_s: float | None = None
    torso_imu_age_s: float | None = None
    sensor_read_started_ns: int | None = None
    action_submitted_ns: int | None = None
    source: Literal["hardware", "simulation"] = "hardware"
    simulation_frame_id: int | None = None
    simulation_state_ready_ns: int | None = None

    def __post_init__(self) -> None:
        for name in ("episode_id", "control_step", "policy_version", "sensor_tick", "torso_imu_sequence",
                     "sensor_read_started_ns", "action_submitted_ns",
                     "simulation_frame_id", "simulation_state_ready_ns"):
            value = getattr(self, name)
            if value is not None or name in ("episode_id", "control_step", "policy_version"):
                _index(name, value)
        if type(self.policy_control_active) is not bool or type(self.action_dispatched) is not bool:
            raise ValueError("control activation and dispatch evidence must be booleans")
        for name in ("lowstate_age_s", "torso_imu_age_s"):
            value = getattr(self, name)
            if value is not None and (isinstance(value, bool) or not math.isfinite(value) or value < 0.0):
                raise ValueError(f"{name} must be a finite non-negative age or None")
        if self.source not in ("hardware", "simulation"):
            raise ValueError("control evidence source must be hardware or simulation")
        hardware = (self.sensor_tick, self.torso_imu_sequence, self.lowstate_age_s, self.torso_imu_age_s)
        simulation = (self.simulation_frame_id, self.simulation_state_ready_ns)
        if ((self.source == "simulation" and any(value is not None for value in hardware))
                or (self.source == "hardware" and any(value is not None for value in simulation))):
            raise ValueError("hardware and simulation evidence must not be mixed")


@dataclass(frozen=True)
class FADAInferenceSnapshot:
    """One model query. Consumers must not mutate the owned tensors."""

    observation_history: torch.Tensor
    action_history: torch.Tensor
    command: torch.Tensor
    nominal_future: torch.Tensor
    action_snapshot: torch.Tensor

    def __post_init__(self) -> None:
        _history(self.observation_history, self.action_history)
        reference = self.observation_history
        _tensor("command", self.command, (1, 3), reference)
        _future(self.nominal_future, reference)
        _tensor("action_snapshot", self.action_snapshot, (1, 29), reference)
        for name in ("observation_history", "action_history", "command", "nominal_future", "action_snapshot"):
            object.__setattr__(self, name, getattr(self, name).detach().clone())


@dataclass(frozen=True)
class FADACausalWindow:
    """Legacy regression inputs; executed_action_chunk contains issued raw commands."""

    observation_history: torch.Tensor
    action_history: torch.Tensor
    command: torch.Tensor
    realized_future: torch.Tensor
    executed_action_chunk: torch.Tensor
    evidence: FADAControlEvidence

    def __post_init__(self) -> None:
        _history(self.observation_history, self.action_history)
        reference = self.observation_history
        horizon = _future(self.realized_future, reference)
        _tensor("command", self.command, (1, 3), reference)
        _tensor("executed_action_chunk", self.executed_action_chunk, (1, horizon, 29), reference)
        for name in ("observation_history", "action_history", "command", "realized_future", "executed_action_chunk"):
            object.__setattr__(self, name, getattr(self, name).detach().clone())


@dataclass(frozen=True)
class FADATrajectoryPair:
    snapshot: FADAInferenceSnapshot
    realized_future: torch.Tensor
    evidence: FADAControlEvidence

    def __post_init__(self) -> None:
        _tensor("realized_future", self.realized_future,
                tuple(self.snapshot.nominal_future.shape), self.snapshot.nominal_future)
        object.__setattr__(self, "realized_future", self.realized_future.detach().clone())


@dataclass(frozen=True)
class FADAOnlineSample:
    trajectory: FADATrajectoryPair
    target_action: torch.Tensor

    def __post_init__(self) -> None:
        _tensor("target_action", self.target_action, (1, 29), self.trajectory.snapshot.action_snapshot)
        object.__setattr__(self, "target_action", self.target_action.detach().clone())
