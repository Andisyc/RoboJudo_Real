# Adapted from TTT online_collector.py/windows.py, revisions c4ff3bf6/0f8a410b.
# Licensed under Apache-2.0; see LICENSE-TTT.
"""Causal control-event collection; no optimizer, control loop or SDK dependency."""
from __future__ import annotations

import math
from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass
from numbers import Integral
from typing import Iterable

import torch

from .model import FADAArchitectureConfig
from .online_data import (
    FADACausalWindow, FADAControlEvidence, FADAInferenceSnapshot, FADATrajectoryPair,
    _tensor,
)


@dataclass(frozen=True)
class FADACollectionResult:
    accepted: bool
    sample: FADACausalWindow | FADATrajectoryPair | None = None
    rejection_reason: str | None = None


@dataclass(frozen=True)
class _Pending:
    snapshot: FADAInferenceSnapshot
    evidence: FADAControlEvidence
    previous_action: torch.Tensor | None


@dataclass(frozen=True)
class _Transition:
    observation: torch.Tensor
    previous_action: torch.Tensor
    command: torch.Tensor
    action: torch.Tensor
    next_observation: torch.Tensor
    evidence: FADAControlEvidence


def _frame_key(evidence: FADAControlEvidence) -> tuple:
    return (
        evidence.episode_id, evidence.control_step, evidence.sensor_tick,
        evidence.torso_imu_sequence, evidence.sensor_read_started_ns,
        evidence.lowstate_age_s, evidence.torso_imu_age_s,
        evidence.source, evidence.simulation_frame_id, evidence.simulation_state_ready_ns,
    )


class _CausalCollector(ABC):
    def __init__(self, config: FADAArchitectureConfig, *, max_sensor_age_s: float) -> None:
        if isinstance(max_sensor_age_s, bool) or not math.isfinite(max_sensor_age_s) or max_sensor_age_s <= 0:
            raise ValueError("max_sensor_age_s must be explicitly finite and positive")
        self.config = config
        self.max_sensor_age_s = float(max_sensor_age_s)
        self._episode_id: int | None = None
        self._observed: tuple[torch.Tensor, torch.Tensor, FADAControlEvidence] | None = None
        self._last_sensor: FADAControlEvidence | None = None
        self._frame_accepted = False
        self._recorded = False
        self._pending: _Pending | None = None
        self._last_action: torch.Tensor | None = None

    def _reject(self, reason: str) -> FADACollectionResult:
        self._pending = None
        self._frame_accepted = False
        self._drop_partial()
        return FADACollectionResult(False, rejection_reason=reason)

    @abstractmethod
    def _drop_partial(self) -> None:
        """Discard only uncompleted raw/in-flight data."""

    @abstractmethod
    def _complete(self, pending: _Pending, state: torch.Tensor) -> FADACausalWindow | FADATrajectoryPair | None:
        """Close a genuinely executed transition and possibly emit a sample."""

    @abstractmethod
    def _record_pending(self, pending: _Pending) -> None:
        """Admit a confirmed action using this method's boundary rules."""

    def reset(self, *, episode_id: int | None = None) -> None:
        if episode_id is None:
            episode_id = 0 if self._episode_id is None else self._episode_id + 1
        if isinstance(episode_id, bool) or not isinstance(episode_id, Integral) or episode_id < 0:
            raise ValueError("episode_id must be a non-negative integer")
        self._episode_id = int(episode_id)
        self._observed = None
        self._last_sensor = None
        self._pending = None
        self._frame_accepted = False
        self._recorded = False
        self._last_action = None
        self._drop_partial()

    def _frame_error(self, evidence: FADAControlEvidence,
                     previous: FADAControlEvidence | None) -> str | None:
        if not evidence.policy_control_active:
            return "policy_control_inactive"
        last = self._last_sensor
        if last is not None and evidence.source != last.source:
            return "evidence_source_changed"
        if evidence.source == "simulation":
            if (evidence.simulation_frame_id is None or evidence.simulation_state_ready_ns is None
                    or evidence.simulation_state_ready_ns <= 0 or evidence.sensor_read_started_ns is None):
                return "missing_simulation_evidence"
            if evidence.simulation_state_ready_ns > evidence.sensor_read_started_ns:
                return "simulation_state_after_read"
            if last is not None:
                assert last.simulation_frame_id is not None and last.simulation_state_ready_ns is not None
                if evidence.simulation_frame_id <= last.simulation_frame_id:
                    return "duplicate_or_reordered_simulation_frame"
                self._last_sensor = evidence
                if evidence.simulation_frame_id != last.simulation_frame_id + 1:
                    return "simulation_frame_gap"
                if evidence.simulation_state_ready_ns <= last.simulation_state_ready_ns:
                    return "non_monotonic_simulation_state"
        else:
            if (evidence.sensor_tick is None or not 0 <= evidence.sensor_tick < 2**32
                    or evidence.torso_imu_sequence is None or evidence.torso_imu_sequence <= 0
                    or evidence.sensor_read_started_ns is None
                    or evidence.lowstate_age_s is None or evidence.torso_imu_age_s is None):
                return "missing_sensor_evidence"
            if max(evidence.lowstate_age_s, evidence.torso_imu_age_s) > self.max_sensor_age_s:
                return "stale_sensor_frame"
            if last is not None:
                previous_tick = last.sensor_tick
                previous_imu = last.torso_imu_sequence
                assert previous_tick is not None and previous_imu is not None
                delta = (evidence.sensor_tick - previous_tick) & 0xFFFFFFFF
                if not 0 < delta < 2**31:
                    return "duplicate_or_reordered_tick"
                if evidence.torso_imu_sequence <= previous_imu:
                    return "duplicate_or_reordered_torso_imu"
        # A received frame must not be reused even if its transition is rejected later.
        self._last_sensor = evidence
        if previous is not None:
            if evidence.control_step != previous.control_step + 1:
                return "control_step_gap"
            if (previous.sensor_read_started_ns is not None
                    and evidence.sensor_read_started_ns <= previous.sensor_read_started_ns):
                return "non_monotonic_read"
        return None

    @torch.no_grad()
    def observe(self, projected_state: torch.Tensor, command: torch.Tensor,
                evidence: FADAControlEvidence) -> FADACollectionResult:
        _tensor("projected_state", projected_state, (1, 66), projected_state)
        _tensor("command", command, (1, 3), projected_state)
        if self._episode_id is None:
            self._episode_id = evidence.episode_id
        if evidence.episode_id != self._episode_id:
            return self._reject("episode_mismatch_requires_reset")
        previous = self._observed[2] if self._observed is not None else None
        state = projected_state.detach().clone()
        self._observed = (state, command.detach().clone(), evidence)
        self._frame_accepted = False
        self._recorded = False
        error = self._frame_error(evidence, previous)
        if error is not None:
            return self._reject(error)
        assert evidence.sensor_read_started_ns is not None
        pending = self._pending
        sample = None
        if pending is not None:
            submitted = pending.evidence.action_submitted_ns
            assert submitted is not None
            if evidence.source == "simulation":
                assert evidence.simulation_state_ready_ns is not None
                if evidence.simulation_state_ready_ns <= submitted:
                    return self._reject("frame_not_proven_after_submission")
            else:
                assert evidence.lowstate_age_s is not None and evidence.torso_imu_age_s is not None
                elapsed = (evidence.sensor_read_started_ns - submitted) / 1e9
                if elapsed <= max(evidence.lowstate_age_s, evidence.torso_imu_age_s):
                    return self._reject("frame_not_proven_after_submission")
            if state.dtype != pending.snapshot.action_snapshot.dtype or state.device != pending.snapshot.action_snapshot.device:
                return self._reject("transition_dtype_or_device_changed")
            sample = self._complete(pending, state)
        else:
            # No issued action for the preceding interval: do not compress time.
            self._drop_partial()
        self._pending = None
        self._frame_accepted = True
        return FADACollectionResult(True, sample=sample)

    @torch.no_grad()
    def record_action(self, snapshot: FADAInferenceSnapshot,
                      evidence: FADAControlEvidence) -> FADACollectionResult:
        if self._observed is None or _frame_key(evidence) != _frame_key(self._observed[2]):
            return self._reject("action_does_not_match_observed_frame")
        if self._recorded:
            raise RuntimeError("an action was already recorded for this control step")
        if not evidence.policy_control_active or not evidence.action_dispatched:
            return self._reject("action_not_dispatched_by_policy")
        if (evidence.action_submitted_ns is None or evidence.sensor_read_started_ns is None
                or evidence.action_submitted_ns < evidence.sensor_read_started_ns):
            return self._reject("missing_or_invalid_submission_time")
        state, command, _ = self._observed
        if (snapshot.observation_history.shape != (1, self.config.history_length, 66)
                or snapshot.action_history.shape != (1, self.config.history_length, 29)
                or snapshot.nominal_future.shape != (1, self.config.prediction_horizon, 66)):
            raise ValueError("snapshot history/future must match the checkpoint H/K")
        if (snapshot.action_snapshot.dtype != state.dtype or snapshot.action_snapshot.device != state.device
                or not torch.equal(snapshot.observation_history[:, -1], state)
                or not torch.equal(snapshot.command, command)):
            return self._reject("snapshot_does_not_match_observed_inputs")
        previous_action = self._last_action
        if previous_action is None and evidence.control_step == 0:
            previous_action = torch.zeros_like(snapshot.action_snapshot)
        # The physical command ledger remains valid even when a sensor frame is rejected.
        self._last_action = snapshot.action_snapshot.detach().clone()
        self._recorded = True
        if not self._frame_accepted:
            return self._reject("origin_frame_rejected")
        self._record_pending(_Pending(snapshot, evidence, previous_action))
        return FADACollectionResult(True)


class FADALegacyTrajectoryCollector(_CausalCollector):
    """Emit donor-compatible causal windows; the updater owns mature accumulation."""

    def __init__(self, config: FADAArchitectureConfig, *, max_sensor_age_s: float) -> None:
        super().__init__(config, max_sensor_age_s=max_sensor_age_s)
        self._record_count = config.history_length + config.prediction_horizon - 1
        self._records: deque[_Transition] = deque(maxlen=self._record_count)

    def _drop_partial(self) -> None:
        self._records.clear()

    def _record_pending(self, pending: _Pending) -> None:
        self._pending = pending

    def _complete(self, pending: _Pending, state: torch.Tensor) -> FADACausalWindow | None:
        if pending.previous_action is None:
            self._records.clear()
            return None
        snapshot = pending.snapshot
        self._records.append(_Transition(
            snapshot.observation_history[:, -1].clone(), pending.previous_action,
            snapshot.command, snapshot.action_snapshot, state, pending.evidence,
        ))
        if len(self._records) < self._record_count:
            return None
        records = tuple(self._records)
        history = records[:self.config.history_length]
        future = records[self.config.history_length - 1:]
        if any(not torch.equal(record.command, future[0].command) for record in future[1:]):
            return None
        return FADACausalWindow(
            observation_history=torch.stack([record.observation for record in history], dim=1),
            action_history=torch.stack([record.previous_action for record in history], dim=1),
            command=future[0].command,
            realized_future=torch.stack([record.next_observation for record in future], dim=1),
            executed_action_chunk=torch.stack([record.action for record in future], dim=1),
            evidence=future[0].evidence,
        )


class FADAExpertTrajectoryCollector(_CausalCollector):
    """Match saved queries with K real successor states, without constructing labels."""

    def __init__(self, config: FADAArchitectureConfig, *, max_sensor_age_s: float,
                 window_size: int = 10) -> None:
        super().__init__(config, max_sensor_age_s=max_sensor_age_s)
        if isinstance(window_size, bool) or not isinstance(window_size, Integral) or window_size <= 0:
            raise ValueError("window_size must be a positive integer")
        self.window_size = int(window_size)
        self._samples: deque[FADATrajectoryPair] = deque(maxlen=self.window_size)
        self._inflight: deque[_Pending] = deque()
        self._future: deque[torch.Tensor] = deque(maxlen=config.prediction_horizon)
        self._command: torch.Tensor | None = None

    def _drop_partial(self) -> None:
        self._inflight.clear()
        self._future.clear()
        self._command = None

    def _record_pending(self, pending: _Pending) -> None:
        command = pending.snapshot.command
        if self._command is not None and not torch.equal(self._command, command):
            self._drop_partial()
        self._command = command
        self._pending = pending

    def _complete(self, pending: _Pending, state: torch.Tensor) -> FADATrajectoryPair | None:
        self._inflight.append(pending)
        self._future.append(state)
        if len(self._future) < self.config.prediction_horizon:
            return None
        first = self._inflight.popleft()
        pair = FADATrajectoryPair(first.snapshot, torch.stack(tuple(self._future), dim=1), first.evidence)
        self._samples.append(pair)
        return pair

    @property
    def ready(self) -> bool:
        return len(self._samples) == self.window_size

    def get_window(self) -> tuple[FADATrajectoryPair, ...]:
        return tuple(self._samples)

    def acknowledge(self, sample_ids: Iterable[tuple[int, int]]) -> int:
        """Consume only confirmed origins, never newly arrived or in-flight samples."""
        identities = set(sample_ids)
        before = len(self._samples)
        self._samples = deque(
            (sample for sample in self._samples
             if (sample.evidence.episode_id, sample.evidence.control_step) not in identities),
            maxlen=self.window_size,
        )
        return before - len(self._samples)

    def reset(self, *, episode_id: int | None = None) -> None:
        super().reset(episode_id=episode_id)
        self._samples.clear()
