"""Standalone control-thread coordinator; official robot integration is separate."""
from __future__ import annotations

import threading
from collections import deque
from numbers import Integral
from typing import NoReturn

import torch

from .lora import fada_adapter_named_parameters
from .online_collector import FADAExpertTrajectoryCollector, FADALegacyTrajectoryCollector
from .online_data import FADACausalWindow, FADAControlEvidence, FADAInferenceSnapshot
from .online_runtime import FADATrainableRuntime
from .online_worker import FADALearnerError, FADALearnerWorker


class FADAOnlineCoordinator:
    """One control writer, one learner, one pending full publication.

    Order on each active control step: observe -> poll/apply -> try_submit ->
    runtime inference -> real submission -> record_action. Application and
    inference must run on this object's owning thread, never concurrent readers.
    start/close are cold/lifecycle operations, not control-step operations.
    """

    def __init__(self, runtime: FADATrainableRuntime, config) -> None:
        if config.max_sensor_age_s is None:
            raise ValueError("online collection requires an explicit max_sensor_age_s")
        self.runtime = runtime
        self.config = config.model_copy(deep=True)
        if runtime.method == "fada":
            self.collector = FADALegacyTrajectoryCollector(
                runtime.policy.config, max_sensor_age_s=config.max_sensor_age_s,
            )
        else:
            self.collector = FADAExpertTrajectoryCollector(
                runtime.policy.config, max_sensor_age_s=config.max_sensor_age_s,
                window_size=config.window_size,
            )
        self.worker = FADALearnerWorker(runtime, self.config)
        self._owner = threading.get_ident()
        self.episode_id = 0
        self.generation = 0
        self.policy_version = 0
        self.collector.reset(episode_id=self.episode_id)
        self._outbox: deque[FADACausalWindow] = deque()
        self._inflight = ()
        self._publication = None
        self._step = -1
        self._active = False
        self._accepted_frame = False
        self._last_submitted = None
        self._stopping = False
        self._closed = False
        self.failure: str | None = None
        self.last_update: dict | None = None

    def _check_owner(self) -> None:
        if threading.get_ident() != self._owner:
            raise FADALearnerError("coordinator requires a single owning control thread")
        if self.failure is not None:
            raise FADALearnerError(self.failure)

    def _fail(self, message: str) -> NoReturn:
        self.failure = message
        self._publication = None
        self.worker.request_stop()
        raise FADALearnerError(message)

    @property
    def status(self) -> dict:
        pending = (len(self._outbox) if isinstance(self.collector, FADALegacyTrajectoryCollector)
                   else len(self.collector.get_window()))
        return {
            **self.worker.status,
            "state": "STOPPED" if self._closed else "FAILED" if self.failure else self.worker.state,
            "failure": self.failure or self.worker.failure,
            "episode_id": self.episode_id, "generation": self.generation,
            "control_step": self._step, "policy_version": self.policy_version,
            "pending_samples": pending, "inflight_samples": len(self._inflight),
            "publication_pending": self._publication is not None,
            "last_update": self.last_update,
        }

    def start(self) -> None:
        self._check_owner()
        if self._stopping:
            raise FADALearnerError("stopped coordinator cannot be started")
        self.worker.start()

    def observe(self, projected_state: torch.Tensor, command: torch.Tensor,
                evidence: FADAControlEvidence):
        self._check_owner()
        if self._stopping:
            raise FADALearnerError("coordinator is stopping")
        if evidence.episode_id != self.episode_id:
            raise ValueError("episode changes require coordinator.reset")
        self.poll()
        result = self.collector.observe(projected_state, command, evidence)
        self._step = max(self._step, evidence.control_step)
        self._active = evidence.policy_control_active
        self._accepted_frame = result.accepted
        if isinstance(result.sample, FADACausalWindow):
            if len(self._outbox) >= self.config.pending_capacity:
                origin = (result.sample.evidence.episode_id, result.sample.evidence.control_step)
                self._fail(f"legacy mature outbox full at sample {origin}; pending batches remain unacknowledged")
            self._outbox.append(result.sample)
        return result

    def record_action(self, snapshot: FADAInferenceSnapshot, evidence: FADAControlEvidence):
        self._check_owner()
        if self._stopping:
            raise FADALearnerError("coordinator is stopping")
        if evidence.policy_version != self.policy_version:
            self._fail("issued snapshot policy_version does not match applied weights")
        return self.collector.record_action(snapshot, evidence)

    def try_submit(self):
        self._check_owner()
        if (self._stopping or not self._active or not self._accepted_frame
                or self._step <= 0 or self._step % self.config.update_interval != 0
                or self._last_submitted == (self.episode_id, self._step)):
            return None
        self.poll()
        if self._publication is not None or self.worker.state != "READY":
            return None
        if isinstance(self.collector, FADALegacyTrajectoryCollector):
            if not self._outbox:
                return None
            samples = tuple(self._outbox)
        else:
            if self._step < self.config.warmup_steps or not self.collector.ready:
                return None
            samples = self.collector.get_window()
        try:
            job = self.worker.submit(
                samples, generation=self.generation, episode_id=self.episode_id,
                control_step=self._step, policy_version=self.policy_version,
            )
        except FADALearnerError as error:
            self._fail(str(error))
        if job is not None:
            self._inflight = samples
            if self.runtime.method == "fada":
                self._outbox.clear()  # Ownership moved to this one in-flight batch, not consumed yet.
            self._last_submitted = (self.episode_id, self._step)
        return job

    def poll(self):
        self._check_owner()
        if self._stopping:
            return None
        try:
            completed = self.worker.poll()
        except FADALearnerError as error:
            self._fail(str(error))
        if completed is None:
            return None
        if self._publication is not None:
            self._fail("received another completion before the prior version was applied")
        result, values = completed
        identities = tuple((s.evidence.episode_id, s.evidence.control_step) for s in self._inflight)
        if (result.identity.sample_ids != identities or result.identity.generation > self.generation
                or result.identity.policy_version != self.policy_version):
            self._fail("completed batch does not match coordinator ownership")
        acknowledged = len(identities)
        if isinstance(self.collector, FADAExpertTrajectoryCollector):
            acknowledged = self.collector.acknowledge(identities)
        self._inflight = ()
        self._publication = (result, values)
        self.last_update = {
            "job_id": result.identity.job_id, "generation": result.identity.generation,
            "episode_id": result.identity.episode_id, "sample_ids": identities,
            "submitted_step": result.identity.control_step, "completed_step": self._step,
            "acknowledged": acknowledged, "stats": dict(result.stats),
            "duration_s": result.duration_s, "latency_s": self.worker.last_latency_s,
        }
        return result

    @torch.no_grad()
    def apply_pending_at_step_boundary(self) -> bool:
        self._check_owner()
        if self._stopping or not self._active or not self._accepted_frame:
            return False
        self.poll()
        if self._publication is None:
            return False
        result, values = self._publication
        if result.learner_version != self.policy_version + 1:
            self._fail("publication version is not the next full learner version")
        parameters = dict(fada_adapter_named_parameters(self.runtime.policy))
        try:
            if set(parameters) != set(values) or any(
                parameters[name].shape != value.shape or parameters[name].dtype != value.dtype
                for name, value in values.items()
            ):
                raise ValueError("actor LoRA schema changed after learner initialization")
            # Prepare every transfer before writing any actor parameter. No forward may interleave.
            staged = {name: value.to(parameters[name]) for name, value in values.items()}
            for name, parameter in parameters.items():
                parameter.copy_(staged[name])
        except Exception as error:
            self._fail(f"full LoRA application failed; actor must not infer again: {error}")
        self.policy_version = result.learner_version
        self._publication = None
        return True

    def reset(self, *, episode_id: int | None = None) -> None:
        self._check_owner()
        if self._stopping:
            raise FADALearnerError("coordinator is stopping")
        new_episode = self.episode_id + 1 if episode_id is None else episode_id
        if isinstance(new_episode, bool) or not isinstance(new_episode, Integral) or new_episode <= self.episode_id:
            raise ValueError("reset requires a strictly increasing episode_id")
        self.episode_id = int(new_episode)
        self.generation += 1
        self.collector.reset(episode_id=self.episode_id)
        self._step = -1
        self._active = False
        self._accepted_frame = False
        # Keep outbox, accepted job, staged result and the worker's optimizer/anchor.

    def request_stop(self) -> None:
        self._stopping = True
        self._active = False
        self._publication = None
        self.worker.request_stop()

    def close(self) -> None:
        self.request_stop()
        self.worker.close()
        self._closed = True
