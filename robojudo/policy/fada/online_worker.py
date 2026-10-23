"""One spawn learner, owned byte messages, and no robot/control dependencies."""
from __future__ import annotations

import io
import math
import multiprocessing as mp
import queue
import threading
import time
import traceback
from collections.abc import Sequence
from dataclasses import asdict, dataclass, fields
from typing import NoReturn

import torch

from .lora import FADALoRAConfig, assert_fada_adaptation_parameter_ownership, fada_adapter_named_parameters, inject_fada_idm_lora
from .model import FADAArchitectureConfig, FADAInverseDynamicsModel, FADAPlannerIDMPolicy
from .online_data import FADACausalWindow, FADAControlEvidence, FADAInferenceSnapshot, FADATrajectoryPair
from .online_runtime import FADATrainableRuntime
from .online_updater import FADAExpertUpdater, FADALegacyUpdater, build_expert_samples


class FADALearnerError(RuntimeError):
    """A terminal learning/protocol failure; never an inference-only fallback."""


class _StopFlag:
    """Parent-written monotone byte: abrupt child death cannot abandon a lock."""

    def __init__(self, context):
        self._value = context.RawValue("b", 0)

    def set(self):
        self._value.value = 1

    def is_set(self):
        return bool(self._value.value)


def _dump(value) -> bytes:
    buffer = io.BytesIO()
    torch.save(value, buffer)
    return buffer.getvalue()


def _load(data: bytes, device="cpu"):
    return torch.load(io.BytesIO(data), map_location=device, weights_only=True)


def _cpu_state(module) -> dict[str, torch.Tensor]:
    # Serialization is synchronous; the byte result owns the copy before it is queued.
    return {name: value.detach().cpu() for name, value in module.state_dict().items()}


def _pack_record(record):
    result = {}
    for field in fields(record):
        value = getattr(record, field.name)
        if isinstance(value, torch.Tensor):
            value = value.detach().cpu()
        elif isinstance(value, FADAControlEvidence):
            value = asdict(value)
        elif isinstance(value, FADAInferenceSnapshot):
            value = _pack_record(value)
        result[field.name] = value
    return result


def _unpack_samples(method, data, device):
    samples = []
    for record in _load(data, device):
        evidence = FADAControlEvidence(**record.pop("evidence"))
        if method == "fada":
            sample = FADACausalWindow(evidence=evidence, **record)
        else:
            snapshot = FADAInferenceSnapshot(**record.pop("snapshot"))
            sample = FADATrajectoryPair(snapshot=snapshot, evidence=evidence, **record)
        samples.append(sample)
    return tuple(samples)


def _initial_snapshot(runtime: FADATrainableRuntime) -> bytes:
    return _dump({
        "method": runtime.method,
        "architecture": asdict(runtime.policy.config),
        "lora": asdict(runtime.lora_config),
        "policy": _cpu_state(runtime.policy),
        "base": None if runtime.base_idm is None else _cpu_state(runtime.base_idm),
    })


def _restore_runtime(data: bytes, device: str) -> FADATrainableRuntime:
    initial = _load(data)
    config = FADAArchitectureConfig(**initial["architecture"])
    adapted = inject_fada_idm_lora(FADAPlannerIDMPolicy(config), FADALoRAConfig(**initial["lora"]))
    adapted.policy.to(dtype=next(iter(initial["policy"].values())).dtype)
    adapted.policy.load_state_dict(initial["policy"], strict=True)
    adapted.policy.to(device).eval()
    base = None
    if initial["method"] == "expert_difference":
        if initial["base"] is None:
            raise ValueError("expert learner requires the deployment base snapshot")
        base = FADAInverseDynamicsModel(config)
        base.to(dtype=next(iter(initial["base"].values())).dtype)
        base.load_state_dict(initial["base"], strict=True)
        base.to(device).eval().requires_grad_(False)
    return FADATrainableRuntime(initial["method"], adapted.policy, adapted.lora_config, base)


@dataclass(frozen=True)
class FADAJobIdentity:
    job_id: int
    generation: int
    episode_id: int
    control_step: int
    policy_version: int
    sample_ids: tuple[tuple[int, int], ...]


@dataclass(frozen=True)
class FADAUpdateJob:
    identity: FADAJobIdentity
    data: bytes


@dataclass(frozen=True)
class FADAUpdateResult:
    identity: FADAJobIdentity
    learner_version: int
    adapters: bytes
    stats: dict[str, float | int]
    duration_s: float


def _learner_main(initial, config, jobs, results, stopping):
    """Importable spawn target. No SDK, environment, or Planner forward is created."""
    try:
        torch.set_num_threads(config["learner_threads"])
        runtime = _restore_runtime(initial, config["learner_device"])
        common = {"lr": config["learning_rate"], "max_grad_norm": config["max_grad_norm"]}
        if runtime.method == "fada":
            updater = FADALegacyUpdater(runtime, lambda_forget=config["lambda_forget"], **common)
        else:
            updater = FADAExpertUpdater(runtime, sigma=config["sigma"], **common)
        results.send({"kind": "ready"})
        version = 0
        while not stopping.is_set():
            try:
                job = jobs.get(timeout=0.1)
            except queue.Empty:
                continue
            if stopping.is_set():
                break
            started = time.monotonic()
            samples = _unpack_samples(runtime.method, job.data, config["learner_device"])
            identities = tuple((s.evidence.episode_id, s.evidence.control_step) for s in samples)
            if identities != job.identity.sample_ids or not samples:
                raise ValueError("learner batch identities do not match the submitted job")
            if isinstance(updater, FADALegacyUpdater):
                for sample in samples:
                    updater.append_window(sample)
                stats = updater.update()
            else:
                assert runtime.base_idm is not None
                fixed = build_expert_samples(runtime.base_idm, samples, gamma=config["gamma"])
                stats = updater.update(fixed)
            adapters = {name: parameter.detach().cpu()
                        for name, parameter in fada_adapter_named_parameters(runtime.policy)}
            if not all(bool(torch.isfinite(value).all()) for value in adapters.values()):
                raise ValueError("learner produced non-finite LoRA weights")
            if stats is None or not all(math.isfinite(float(value)) for value in stats.values()):
                raise ValueError("learner produced invalid update statistics")
            packet = _dump(adapters)
            duration = time.monotonic() - started
            version += 1
            if not stopping.is_set():
                results.send(FADAUpdateResult(job.identity, version, packet, stats, duration))
    except Exception:
        if not stopping.is_set():
            try:
                results.send({"kind": "error", "error": traceback.format_exc()})
            except (BrokenPipeError, EOFError, OSError):
                pass


class FADALearnerWorker:
    """One outstanding job. Only cold start and explicit close may wait.

    The coordinator additionally holds admission until the completed version is
    applied. All IPC tensors are serialized bytes, never shared torch storages.
    """

    def __init__(self, runtime: FADATrainableRuntime, config) -> None:
        if config.method == "none" or config.method != runtime.method:
            raise ValueError("learner requires a matching online method")
        actual = runtime.lora_config
        if (config.lora_rank, config.lora_alpha, config.lora_dropout) != (
            actual.rank, actual.alpha, actual.dropout,
        ):
            raise ValueError("learner config does not match the prepared LoRA runtime")
        assert_fada_adaptation_parameter_ownership(runtime.policy, actual)
        if runtime.method == "expert_difference":
            base = runtime.base_idm
            if (base is None or any(module.training for module in base.modules())
                    or any(parameter.requires_grad for parameter in base.parameters())):
                raise ValueError("expert learner requires an already frozen eval base")
        self.config = config.model_copy(deep=True)
        self._initial = _initial_snapshot(runtime)
        self._spec = {name: (tuple(value.shape), value.dtype)
                      for name, value in fada_adapter_named_parameters(runtime.policy)}
        self._context = mp.get_context("spawn")
        self._jobs = self._context.Queue(maxsize=1)
        self._results, self._result_sender = self._context.Pipe(duplex=False)
        self._inbox = queue.Queue(maxsize=1)
        self._received = None
        self._receiver = None
        self._stopping = _StopFlag(self._context)
        self._process = None
        self._job: FADAUpdateJob | None = None
        self._submitted_at = None
        self._next_job_id = 1
        self.learner_version = 0
        self.state = "NEW"
        self.failure: str | None = None
        self.last_latency_s: float | None = None

    @property
    def process(self):
        """Process identity for liveness inspection; never a control/SDK process."""
        return self._process

    @property
    def pending_job(self) -> FADAUpdateJob | None:
        return self._job

    @property
    def status(self) -> dict:
        return {
            "state": self.state, "failure": self.failure,
            "job_id": None if self._job is None else self._job.identity.job_id,
            "learner_version": self.learner_version,
            "elapsed_s": None if self._submitted_at is None else time.monotonic() - self._submitted_at,
            "last_latency_s": self.last_latency_s,
        }

    def _fail(self, message: str) -> NoReturn:
        self.failure = message
        self.state = "FAILED"
        self._stopping.set()
        raise FADALearnerError(message)

    def _receive_results(self) -> None:
        while not self._stopping.is_set():
            try:
                message = self._results.recv()
            except (EOFError, OSError):
                message = {"kind": "error", "error": "learner result channel closed"}
            while not self._stopping.is_set():
                try:
                    self._inbox.put(message, timeout=0.1)
                    break
                except queue.Full:
                    continue
            if isinstance(message, dict) and message.get("kind") == "error":
                return

    def wait_for_message(self, timeout: float) -> bool:
        """Offline/lifecycle wait only. Control paths must use poll instead."""
        if self._received is not None:
            return True
        try:
            self._received = self._inbox.get(timeout=timeout)
        except queue.Empty:
            return False
        return True

    def start(self) -> None:
        if self.state != "NEW":
            raise FADALearnerError("learner can only be started once")
        self.state = "STARTING"
        self._process = self._context.Process(
            target=_learner_main,
            args=(self._initial, self.config.model_dump(), self._jobs, self._result_sender, self._stopping),
            name="fada-learner",
        )
        try:
            self._process.start()
            # Close unused endpoints so child death yields EOF/EPIPE, not a stranded reader/feeder.
            self._result_sender.close()
            self._jobs._reader.close()
            self._receiver = threading.Thread(target=self._receive_results, name="fada-result-reader", daemon=True)
            self._receiver.start()
            message = self._inbox.get(timeout=self.config.startup_timeout_s)
            if not isinstance(message, dict) or message.get("kind") != "ready":
                raise FADALearnerError(str(message))
            if not self._process.is_alive():
                raise FADALearnerError("learner exited during startup")
            self._initial = b""
            self.state = "READY"
        except Exception as error:
            self.failure = str(error) or "learner startup timed out"
            self.state = "FAILED"
            self.close()
            self.state = "FAILED"
            raise FADALearnerError(self.failure) from error

    def submit(self, samples: Sequence[FADACausalWindow | FADATrajectoryPair], *,
               generation: int, episode_id: int, control_step: int,
               policy_version: int) -> FADAUpdateJob | None:
        if self.state == "FAILED":
            raise FADALearnerError(self.failure)
        if self.state == "BUSY":
            return None
        if self.state != "READY":
            raise FADALearnerError("learner is not accepting jobs")
        if self._process is None or not self._process.is_alive():
            self._fail("learner exited unexpectedly before submission")
        expected_type = FADACausalWindow if self.config.method == "fada" else FADATrajectoryPair
        if not samples or any(not isinstance(sample, expected_type) for sample in samples):
            raise ValueError("batch must contain matching non-empty causal samples")
        identities = tuple((s.evidence.episode_id, s.evidence.control_step) for s in samples)
        if len(set(identities)) != len(identities):
            raise ValueError("batch sample IDs must be unique")
        identity = FADAJobIdentity(self._next_job_id, generation, episode_id, control_step,
                                   policy_version, identities)
        job = FADAUpdateJob(identity, _dump([_pack_record(s) for s in samples]))
        try:
            self._jobs.put_nowait(job)
        except queue.Full:
            self._fail("single-job queue unexpectedly full")
        self._next_job_id += 1
        self._job = job
        self._submitted_at = time.monotonic()
        self.state = "BUSY"
        return job

    def validate_adapters(self, data: bytes) -> dict[str, torch.Tensor]:
        try:
            values = _load(data)
            if not isinstance(values, dict) or set(values) != set(self._spec):
                raise ValueError("LoRA packet must contain the complete exact parameter set")
            for name, value in values.items():
                shape, dtype = self._spec[name]
                if (not isinstance(value, torch.Tensor) or tuple(value.shape) != shape
                        or value.dtype != dtype or not bool(torch.isfinite(value).all())):
                    raise ValueError(f"invalid LoRA tensor: {name}")
            return values
        except Exception as error:
            self._fail(f"invalid learner publication: {error}")

    def poll(self) -> tuple[FADAUpdateResult, dict[str, torch.Tensor]] | None:
        if self.state == "FAILED":
            raise FADALearnerError(self.failure)
        if self.state in ("NEW", "STOPPING", "STOPPED"):
            return None
        if self._received is not None:
            result, self._received = self._received, None
        else:
            try:
                result = self._inbox.get_nowait()
            except queue.Empty:
                result = None
        if isinstance(result, dict) and result.get("kind") == "error":
            self._fail(result["error"])
        if self._process is None or not self._process.is_alive():
            self._fail("learner exited unexpectedly")
        if result is None:
            if self._job is not None:
                assert self._submitted_at is not None
                if time.monotonic() - self._submitted_at > self.config.update_timeout_s:
                    self._fail(f"learner update timed out for job {self._job.identity.job_id}")
            return None
        if (not isinstance(result, FADAUpdateResult) or self._job is None
                or result.identity != self._job.identity or result.learner_version != self.learner_version + 1):
            self._fail("learner result identity/version mismatch")
        if not math.isfinite(result.duration_s) or not 0 <= result.duration_s <= self.config.update_timeout_s:
            self._fail("learner update exceeded its duration budget")
        values = self.validate_adapters(result.adapters)
        expected_count = len(self._job.identity.sample_ids)
        count_key = "num_windows" if self.config.method == "fada" else "num_samples"
        if (result.stats.get(count_key) != expected_count
                or not all(math.isfinite(float(v)) for v in result.stats.values())):
            self._fail("learner result statistics do not match the batch")
        assert self._submitted_at is not None
        self.last_latency_s = time.monotonic() - self._submitted_at
        self.learner_version = result.learner_version
        self._job = None
        self._submitted_at = None
        self.state = "READY"
        return result, values

    def request_stop(self) -> None:
        self._stopping.set()
        if self.state != "STOPPED":
            self.state = "STOPPING"

    def close(self) -> None:
        if self.state == "STOPPED":
            return
        self.request_stop()
        process = self._process
        if process is not None and process.pid is not None:
            budget = self.config.close_timeout_s / 4
            process.join(timeout=budget)
            if process.is_alive():
                process.terminate()
                process.join(timeout=budget)
            if process.is_alive():
                process.kill()
                process.join(timeout=budget)
            if process.is_alive():
                self._fail("learner could not be reaped within close budget")
        self._result_sender.close()
        if self._receiver is not None:
            self._receiver.join(timeout=self.config.close_timeout_s / 4)
            if self._receiver.is_alive():
                self._fail("learner result reader could not be reaped within close budget")
        self._results.close()
        self._jobs.cancel_join_thread()
        self._jobs.close()
        self.state = "STOPPED"
