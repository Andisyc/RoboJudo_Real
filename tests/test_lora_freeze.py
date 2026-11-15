"""Real tiny learners and adapter methods, without RoboJuDo/SDK/environment startup."""
from __future__ import annotations

import ast
import copy
import functools
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Literal
from unittest.mock import patch

import numpy as np
import pytest
import torch
from pydantic import BaseModel, Field, model_validator

ROOT = Path(__file__).resolve().parents[1]
# FADA's standalone package uses only relative imports; spawn inherits this path.
sys.path.insert(0, str(ROOT / "robojudo/policy"))
from fada.lora import FADALoRAConfig, fada_adapter_named_parameters, inject_fada_idm_lora
from fada.model import FADAArchitectureConfig, FADAPlannerIDMPolicy
from fada.observation import FADA_G1_ACTOR_OBS_DIM, FADA_G1_STATE_DIM
from fada.online_controller import FADAOnlineCoordinator
from fada.online_data import FADAControlEvidence
from fada.online_runtime import FADATrainableRuntime
from fada.online_updater import FADAExpertUpdater, FADALegacyUpdater, build_expert_samples
from fada.online_worker import _learner_main, _restore_runtime
from fada.playback import FADAPlaybackController


def config_type():
    path = ROOT / "robojudo/config/g1/policy/g1_fada_policy_cfg.py"
    node = next(n for n in ast.parse(path.read_text()).body if isinstance(n, ast.ClassDef) and n.name == "FADAOnlineAdaptationCfg")
    namespace = {"Config": BaseModel, "Literal": Literal, "Field": Field, "model_validator": model_validator}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), namespace)
    namespace["FADAOnlineAdaptationCfg"].model_rebuild(_types_namespace=namespace)
    return namespace["FADAOnlineAdaptationCfg"]


def adapter_type():
    path = ROOT / "robojudo/policy/fada_policy.py"
    tree = ast.parse(path.read_text())
    # Isolate project service imports only. No model or native environment is constructed.
    tree.body = [n for n in tree.body if not (
        isinstance(n, ast.ImportFrom) and (n.level or (n.module or "").startswith("robojudo"))
    )]
    namespace = {
        "policy_registry": SimpleNamespace(register=lambda cls: cls), "UniLabPolicy": object,
        "FADAControlEvidence": FADAControlEvidence, "FADA_G1_ACTOR_OBS_DIM": FADA_G1_ACTOR_OBS_DIM,
        "FADA_G1_STATE_DIM": FADA_G1_STATE_DIM,
    }
    exec(compile(tree, str(path), "exec"), namespace)
    return namespace["FADAPlannerIDMPolicyAdapter"]


CONFIG = config_type()
ADAPTER = adapter_type()


def seeded_learner(initial, config, jobs, results, stopping, *, entered, release):
    torch.manual_seed(731)

    class Requests:
        def get(self, **kwargs):
            job = jobs.get(**kwargs)
            if job.identity.job_id == 2:
                entered.set()
                if not release.wait(15):
                    raise TimeoutError("offline learner gate not released")
            return job

    _learner_main(initial, config, Requests(), results, stopping)


class Frame(SimpleNamespace):
    def get(self, name, default=None):
        return getattr(self, name, default)


class Driver:
    def __init__(self, method, mode, start_frozen=False):
        torch.manual_seed(611)
        arch = FADAArchitectureConfig(
            obs_dim=66, action_dim=29, command_dim=3, observation_contract="g1_fada_state_v2",
            history_length=2, prediction_horizon=2, hidden_dim=16, num_heads=4,
            planner_layers=1, idm_encoder_layers=1, idm_decoder_layers=1, feedforward_dim=32, dropout=0.0,
        )
        policy = FADAPlannerIDMPolicy(arch).eval()
        base = copy.deepcopy(policy.idm).eval().requires_grad_(False) if method == "expert_difference" else None
        lora = FADALoRAConfig(rank=2, alpha=4.0, dropout=0.25)
        adapted = inject_fada_idm_lora(policy, lora)
        with torch.no_grad():
            for name, parameter in fada_adapter_named_parameters(adapted.policy):
                if "_B." in name:
                    parameter.fill_(0.03)
        self.runtime = FADATrainableRuntime(method, adapted.policy.eval(), adapted.lora_config, base)
        self.cfg = CONFIG(method=method, execution_mode=mode, freeze_lora=start_frozen,
                          lora_rank=2, lora_alpha=4.0, lora_dropout=0.25,
                          max_sensor_age_s=0.04, learning_rate=0.01, warmup_steps=0,
                          update_interval=1, window_size=1, close_timeout_s=2.0)
        self.coordinator = FADAOnlineCoordinator(self.runtime, self.cfg)
        torch.manual_seed(731)
        self.reference = _restore_runtime(self.coordinator.worker._initial, "cpu")
        common = {"lr": self.cfg.learning_rate, "max_grad_norm": self.cfg.max_grad_norm}
        self.reference_updater = (FADALegacyUpdater(self.reference, lambda_forget=self.cfg.lambda_forget, **common)
                                  if method == "fada" else FADAExpertUpdater(self.reference, sigma=self.cfg.sigma, **common))
        self.entered, self.release = self.coordinator.worker._context.Event(), self.coordinator.worker._context.Event()
        target = functools.partial(seeded_learner, entered=self.entered, release=self.release)
        with patch("fada.online_worker._learner_main", target):
            self.coordinator.start()
        self.batches = []
        submit = self.coordinator.worker.submit

        def record(samples, **identity):
            job = submit(samples, **identity)
            if job is not None:
                self.batches.append(tuple(samples))
            return job

        self.coordinator.worker.submit = record
        self.actor = object.__new__(ADAPTER)
        self.actor.__dict__.update(
            online_coordinator=self.coordinator, _lora_frozen=start_frozen, _control_active=True,
            _control_step=0, _adaptation_started=False, _pending_collection=False,
            _pending_snapshot=None, _pending_evidence=None, _pending_sensor=None,
            _pending_observation=None, _pending_command=None, _stopping=False,
            _held_motion_keys=set(), _keyboard_command=np.zeros(3, dtype=np.float32),
            keyboard_command_magnitudes=np.array([0.4, 0.4, 0.8], dtype=np.float32),
            keyboard_command_mode="latch", _keyboard_hold_blocked=False,
            default_dof_pos=np.zeros(29), last_action=np.zeros(29), gait_phase=np.array([0.0, np.pi]),
            execution_action_scale=1.0, _runtime={"runner": self.runtime},
        )
        self.actor.playback_controller = FADAPlaybackController(
            self.runtime, device="cpu", history_length=2, observation_contract="g1_fada_state_v2",
            action_dim=29, command_dim=3,
        )

    def step(self, key=None, command=None):
        i = self.actor._control_step
        now = 1_000_000_000 + i * 20_000_000
        frame = Frame(policy_gyro=np.zeros(3), policy_gravity=np.array([0.0, 0.0, -1.0]),
                      dof_pos=0.01 * np.sin(i * 0.1 + np.arange(29)), dof_vel=np.zeros(29),
                      source="simulation", simulation_frame_id=i, sensor_read_started_ns=now,
                      simulation_state_ready_ns=now - 1)
        events = [] if key is None else [{"type": "keyboard", "name": key, "pressed": True}]
        controls = {"KeyboardCtrl": {"keyboard_event": events}, "COMMANDS": [command] if command else []}
        observation, info = self.actor.get_observation(frame, controls)
        self.actor.get_action(observation)
        self.actor.confirm_action_submission(now + 1_000_000)
        return info

    def until(self, predicate):
        deadline = time.monotonic() + 10
        while not predicate():
            assert time.monotonic() < deadline, self.coordinator.status
            self.step()
            time.sleep(0.005)

    def weights(self):
        return {name: parameter.detach().clone() for name, parameter in fada_adapter_named_parameters(self.runtime.policy)}

    def assert_weights(self, before):
        for name, parameter in fada_adapter_named_parameters(self.runtime.policy):
            torch.testing.assert_close(parameter, before[name], rtol=0, atol=0)

    def reference_update(self, samples):
        if self.runtime.method == "fada":
            for sample in samples:
                self.reference_updater.append_window(sample)
            self.reference_updater.update()
        else:
            fixed = build_expert_samples(self.reference.base_idm, samples, gamma=self.cfg.gamma)
            self.reference_updater.update(fixed)

    def close(self):
        self.release.set()
        self.coordinator.close()


@pytest.fixture(autouse=True)
def restore_torch_state():
    threads = torch.get_num_threads()
    with torch.random.fork_rng():
        torch.set_num_threads(1)
        yield
    torch.set_num_threads(threads)


@pytest.mark.parametrize("method,mode,delivery", [
    ("fada", "sync", "sync"),
    ("fada", "async", "busy"),
    ("expert_difference", "async", "busy"),
    ("expert_difference", "async", "staged"),
])
def test_standing_manual_freeze_and_discarded_optimizer_step(method, mode, delivery):
    driver = Driver(method, mode)
    coordinator = driver.coordinator
    try:
        before = driver.weights()
        for _ in range(5):
            assert not driver.step()["lora_updates_enabled"]
        assert not driver.batches and coordinator.policy_version == 0
        driver.assert_weights(before)
        driver.step("w")
        driver.until(lambda: coordinator.policy_version == 1)
        driver.reference_update(driver.batches[0])
        committed = driver.weights()
        assert any(not torch.equal(committed[name], before[name]) for name in before)
        if mode == "async":
            driver.until(driver.entered.is_set)
            if delivery == "staged":
                driver.release.set()
                assert coordinator.worker.wait_for_message(10)
                coordinator.poll()
                assert coordinator.status["publication_pending"]
        else:
            driver.release.set()
        driver.step(command="[LORA_FREEZE]")
        count = len(driver.batches)
        for _ in range(5):
            assert not driver.step()["lora_updates_enabled"]
        assert len(driver.batches) == count
        driver.assert_weights(committed)
        driver.step("x")
        assert not driver.step(command="[LORA_UNFREEZE]")["lora_updates_enabled"]
        assert not driver.actor._lora_frozen
        driver.assert_weights(committed)
        resume_step = driver.actor._control_step
        driver.step("w")
        driver.release.set()
        driver.until(lambda: coordinator.policy_version == 2)
        driver.actor.set_lora_frozen(True)
        accepted = driver.batches[2] if mode == "async" else driver.batches[1]
        assert min(sample.evidence.control_step for sample in accepted) >= resume_step
        driver.reference_update(accepted)
        for name, parameter in fada_adapter_named_parameters(driver.runtime.policy):
            expected = dict(fada_adapter_named_parameters(driver.reference.policy))[name]
            torch.testing.assert_close(parameter, expected, rtol=0, atol=1e-6)
        if mode == "async":
            assert coordinator.discarded_updates >= 1
    finally:
        driver.close()


def test_start_frozen_retains_loaded_nonzero_lora_and_resumes_without_restart():
    driver = Driver("fada", "sync", start_frozen=True)
    try:
        before = driver.weights()
        driver.step("w")
        for _ in range(5):
            assert not driver.step()["lora_updates_enabled"]
        assert not driver.batches
        driver.assert_weights(before)
        driver.release.set()
        assert driver.step(command="[LORA_UNFREEZE]")["lora_updates_enabled"]
        driver.until(lambda: driver.coordinator.policy_version == 1)
        assert any(not torch.equal(driver.weights()[name], before[name]) for name in before)
    finally:
        driver.close()
