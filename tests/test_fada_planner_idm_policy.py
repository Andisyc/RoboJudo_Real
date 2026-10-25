from __future__ import annotations

import importlib
import importlib.util
import sys
from contextlib import contextmanager
import functools
import time
import tempfile
import unittest
from dataclasses import asdict, replace
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

from robojudo.config.g1.g1_cfg import g1_real_fada_planner_idm
from robojudo.config.g1.policy.g1_fada_policy_cfg import (
    FADAOnlineAdaptationCfg, G1FADAPlannerIDMPolicyCfg,
)
from robojudo.policy.fada.checkpoint import (
    _canonical_state_dict_sha256,
    load_fada_policy_checkpoint,
)
from robojudo.policy.fada.model import FADAArchitectureConfig, FADAPlannerIDMPolicy
from robojudo.policy.fada.lora import fada_adapter_named_parameters
from robojudo.policy.fada.online_runtime import prepare_fada_trainable_runtime
from robojudo.policy.fada.onnx_runtime import FADATorchRuntime
from robojudo.policy.fada.observation import FADATorsoImuProjector, project_fada_g1_state
from robojudo.policy.fada.online_collector import (
    FADAExpertTrajectoryCollector, FADALegacyTrajectoryCollector,
)
from robojudo.policy.fada.online_data import (
    FADACausalWindow, FADAControlEvidence, FADAInferenceSnapshot,
    FADAOnlineSample, FADATrajectoryPair,
)
from robojudo.policy.fada.online_updater import (
    FADAExpertUpdater, FADALegacyUpdater, build_expert_samples,
)
from robojudo.policy.fada.online_controller import FADAOnlineCoordinator
from robojudo.policy.fada.online_worker import (
    FADALearnerError, FADALearnerWorker, FADAUpdateResult,
    _dump, _load, _restore_runtime, _unpack_samples,
    _learner_main as _real_fada_learner_main,
)
from robojudo.policy.fada_policy import FADAPlannerIDMPolicyAdapter




def _write_checkpoint(path: Path) -> None:
    config = FADAArchitectureConfig(
        obs_dim=66,
        action_dim=29,
        command_dim=3,
        observation_contract="g1_fada_state_v2",
        history_length=30,
        prediction_horizon=6,
        hidden_dim=16,
        num_heads=4,
        planner_layers=1,
        idm_encoder_layers=1,
        idm_decoder_layers=1,
        feedforward_dim=32,
    )
    policy = FADAPlannerIDMPolicy(config)
    payload = {
        "schema_version": 5,
        "architecture": asdict(config),
        "planner_state_dict": policy.planner.state_dict(),
        "idm_state_dict": policy.idm.state_dict(),
        "training_schedule": "alternating_idm_then_planner",
        "planner_optimizer_state_dict": {},
        "idm_optimizer_state_dict": {},
    }
    payload["idm_sha256"] = _canonical_state_dict_sha256(payload["idm_state_dict"])
    torch.save(payload, path)


class TestFADAPlannerIDMMigration(unittest.TestCase):

    def test_real_fada_torso_imu_matches_sim_tilt_and_aligns_starting_yaw(self):
        def quat_mul(lhs, rhs):
            lw, lx, ly, lz = lhs
            rw, rx, ry, rz = rhs
            return np.asarray(
                [
                    lw * rw - lx * rx - ly * ry - lz * rz,
                    lw * rx + lx * rw + ly * rz - lz * ry,
                    lw * ry - lx * rz + ly * rw + lz * rx,
                    lw * rz + lx * ry - ly * rx + lz * rw,
                ]
            )

        yaw = np.deg2rad(70.0)
        q_yaw = np.asarray([np.cos(yaw / 2.0), 0.0, 0.0, np.sin(yaw / 2.0)])
        for tilt_deg in (30.0, -30.0):
            tilt = np.deg2rad(tilt_deg)
            sin_tilt = np.sin(tilt)
            cos_tilt = np.cos(tilt)
            cases = [
                (
                    np.asarray([np.cos(tilt / 2.0), np.sin(tilt / 2.0), 0.0, 0.0]),
                    [0.0, sin_tilt, -cos_tilt],
                ),
                (
                    np.asarray([np.cos(tilt / 2.0), 0.0, np.sin(tilt / 2.0), 0.0]),
                    [-sin_tilt, 0.0, -cos_tilt],
                ),
            ]
            for q_tilt, expected_gravity in cases:
                projector = FADATorsoImuProjector()
                projector.project(
                    SimpleNamespace(
                        quaternion=q_yaw,
                        gyroscope=[0.0, 0.0, 0.0],
                        accelerometer=[0.0, 0.0, 9.81],
                    )
                )
                gyro, gravity = projector.project(
                    SimpleNamespace(
                        quaternion=quat_mul(q_yaw, q_tilt),
                        gyroscope=[0.1, -0.2, 0.3],
                        accelerometer=[0.0, 0.0, 9.81],
                    )
                )
                np.testing.assert_allclose(gyro, [0.1, -0.2, 0.3])
                np.testing.assert_allclose(gravity, expected_gravity, atol=1e-6)

        with self.assertRaisesRegex(RuntimeError, "quaternion norm"):
            projector.project(
                SimpleNamespace(
                    quaternion=[0.0, 0.0, 0.0, 0.0],
                    gyroscope=[0.0, 0.0, 0.0],
                    accelerometer=[0.0, 0.0, 9.81],
                )
            )

        with self.assertRaisesRegex(RuntimeError, "has not received"):
            projector.project(
                SimpleNamespace(
                    quaternion=[1.0, 0.0, 0.0, 0.0],
                    gyroscope=[0.0, 0.0, 0.0],
                    accelerometer=[0.0, 0.0, 0.0],
                )
            )

    def test_projection_uses_exact_non_leaking_indices(self):
        raw = np.arange(98, dtype=np.float32)[None, :]
        projected = project_fada_g1_state(raw)
        np.testing.assert_array_equal(
            projected, np.concatenate((raw[:, :64], raw[:, 96:98]), axis=1)
        )
        with self.assertRaises(ValueError):
            project_fada_g1_state(np.zeros((1, 97), dtype=np.float32))

    def test_keyboard_commands_are_idempotent_and_report_changes(self):
        adapter = object.__new__(FADAPlannerIDMPolicyAdapter)
        adapter._held_motion_keys = set()
        adapter._keyboard_command = np.zeros(3, dtype=np.float32)
        adapter.keyboard_command_magnitudes = np.asarray([0.6, 0.4, 0.8], dtype=np.float32)

        def key_events(name):
            return {
                "KeyboardCtrl": {
                    "keyboard_event": [
                        {"type": "keyboard", "name": name, "pressed": True},
                        {"type": "keyboard", "name": name, "pressed": False},
                    ]
                }
            }

        with self.assertLogs("robojudo.policy.fada_policy", level="INFO") as captured:
            np.testing.assert_allclose(adapter._get_commands(key_events("w")), [0.6, 0.0, 0.0])
            np.testing.assert_allclose(adapter._get_commands(key_events("w")), [0.6, 0.0, 0.0])
            np.testing.assert_allclose(adapter._get_commands(key_events("a")), [0.6, 0.4, 0.0])
            np.testing.assert_allclose(adapter._get_commands(key_events("x")), [0.0, 0.0, 0.0])

        self.assertEqual(len(captured.records), 3)
        self.assertIn("key=w vx=0.600 vy=0.000 yaw=0.000", captured.output[0])
        self.assertIn("key=x vx=0.000 vy=0.000 yaw=0.000", captured.output[-1])

    def test_checkpoint_identity_and_stateful_adapter(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "planner_idm.pt"
            _write_checkpoint(checkpoint)
            loaded = load_fada_policy_checkpoint(checkpoint)
            self.assertFalse(loaded.policy.training)

            cfg = G1FADAPlannerIDMPolicyCfg(checkpoint_path=str(checkpoint))
            adapter = FADAPlannerIDMPolicyAdapter(cfg, "cpu")
            env_data = SimpleNamespace(
                base_quat=np.asarray([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
                base_ang_vel=np.zeros(3, dtype=np.float32),
                policy_gyro=np.zeros(3, dtype=np.float32),
                policy_gravity=np.asarray([0.0, 0.0, -1.0], dtype=np.float32),
                dof_pos=np.asarray(adapter.default_dof_pos, dtype=np.float32),
                dof_vel=np.zeros(29, dtype=np.float32),
            )
            obs, extras = adapter.get_observation(env_data, {})
            action = adapter.get_action(obs)
            self.assertEqual(obs.shape, (98,))
            self.assertEqual(action.shape, (29,))
            np.testing.assert_allclose(obs[-2:], [0.0, np.pi])
            self.assertEqual(extras["fada_projected_obs_dim"], 66)
            self.assertEqual(
                tuple(adapter.playback_controller._observation_history.shape), (1, 30, 66)
            )
            self.assertEqual(
                tuple(adapter.playback_controller._action_history.shape), (1, 30, 29)
            )
            np.testing.assert_allclose(action, adapter.last_action)
            np.testing.assert_allclose(
                adapter.playback_controller._action_history[0, -1].cpu().numpy(),
                adapter.last_action,
            )
            with self.assertRaises(RuntimeError):
                adapter.get_action(obs)
            adapter.post_step_callback([])
            phase_delta = 2.0 * np.pi * adapter.gait_frequency * adapter.dt
            np.testing.assert_allclose(
                adapter.gait_phase,
                np.asarray([phase_delta, np.pi + phase_delta]) % (2.0 * np.pi),
            )
            adapter.reset()
            self.assertIsNone(adapter.playback_controller._observation_history)

            payload = torch.load(checkpoint, weights_only=True)
            first = next(iter(payload["idm_state_dict"].values()))
            first.view(-1)[0] += 1.0
            torch.save(payload, checkpoint)
            with self.assertRaisesRegex(ValueError, "IDM identity mismatch"):
                load_fada_policy_checkpoint(checkpoint)

    def test_real_observation_uses_unitree_torso_imu(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "planner_idm.pt"
            _write_checkpoint(checkpoint)
            cfg = G1FADAPlannerIDMPolicyCfg(checkpoint_path=str(checkpoint))
            adapter = FADAPlannerIDMPolicyAdapter(cfg, "cpu")
            env_data = SimpleNamespace(
                base_quat=np.asarray([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
                base_ang_vel=np.asarray([9.0, 9.0, 9.0], dtype=np.float32),
                torso_imu_state=SimpleNamespace(
                    quaternion=[1.0, 0.0, 0.0, 0.0],
                    gyroscope=[0.1, -0.2, 0.3],
                    accelerometer=[0.0, 0.0, 9.81],
                ),
                dof_pos=np.asarray(adapter.default_dof_pos, dtype=np.float32),
                dof_vel=np.zeros(29, dtype=np.float32),
            )
            obs, _ = adapter.get_observation(env_data, {})
            action = adapter.get_action(obs)

            snapshot = adapter.snapshot_state()
            obs_next, _ = adapter.get_observation(env_data, {})
            adapter.get_action(obs_next)
            adapter.restore_state(snapshot)
            restored = adapter.snapshot_state()
            torch.testing.assert_close(
                restored["playback"]["observation_history"],  # type: ignore[index]
                snapshot["playback"]["observation_history"],  # type: ignore[index]
            )
            torch.testing.assert_close(
                restored["playback"]["action_history"],  # type: ignore[index]
                snapshot["playback"]["action_history"],  # type: ignore[index]
            )
            np.testing.assert_array_equal(restored["last_action"], snapshot["last_action"])

            np.testing.assert_allclose(obs[:3], [0.025, -0.05, 0.075])
            np.testing.assert_allclose(obs[3:6], [0.0, 0.0, -1.0])
            self.assertEqual(obs.shape, (98,))
            self.assertEqual(action.shape, (29,))
            self.assertTrue(np.isfinite(action).all())


class TestMujocoShoulderPayload(unittest.TestCase):
    def test_rigid_mass_gravity_and_reset_preserve_live_state(self):
        import mujoco
        from robojudo.environment.utils.mujoco_payload import MujocoShoulderPayload

        model = mujoco.MjModel.from_xml_string("""
            <mujoco><worldbody><body name="torso_link" pos="0 0 0.8">
            <freejoint/><inertial pos="0.01 -0.01 0.03" mass="10"
            diaginertia="0.2 0.3 0.4"/>
            </body></worldbody></mujoco>
        """)
        data = mujoco.MjData(model)
        data.time = 0.75
        mujoco.mj_forward(model, data)
        body = model.body("torso_link").id
        baseline = tuple(array[body].copy() for array in (
            model.body_mass, model.body_ipos, model.body_inertia, model.body_iquat,
        ))
        bias = data.qfrc_bias.copy()
        qpos, qvel, clock = data.qpos.copy(), data.qvel.copy(), data.time
        disabled = MujocoShoulderPayload(model, data, enabled=False)
        disabled.before_step([0.6, 0.0, 0.0])
        self.assertEqual(float(model.body_mass[body]), 10.0)
        payload = MujocoShoulderPayload(model, data, enabled=True)
        payload.before_step([0.0, 0.0, 0.0])
        self.assertFalse(payload.active)
        payload.before_step([0.6, 0.0, 0.0])
        self.assertEqual(float(model.body_mass[body]), 13.0)
        np.testing.assert_allclose(
            data.qfrc_bias[3:6] - bias[3:6],
            -np.cross(payload.position, payload.mass_kg * model.opt.gravity),
            atol=1e-9,
        )
        np.testing.assert_array_equal(data.qpos, qpos)
        np.testing.assert_array_equal(data.qvel, qvel)
        self.assertEqual(data.time, clock)
        payload.before_step([0.6, 0.0, 0.0])
        self.assertEqual(float(model.body_mass[body]), 13.0)
        self.assertTrue(np.all(model.body_inertia[body] > 0))
        payload.reset()
        self.assertFalse(payload.active)
        for actual, expected in zip((
            model.body_mass, model.body_ipos, model.body_inertia, model.body_iquat,
        ), baseline):
            np.testing.assert_array_equal(actual[body], expected)
        np.testing.assert_allclose(data.qfrc_bias, bias, atol=1e-9)


class TestFADATrainableRuntime(unittest.TestCase):
    def test_zero_delta_and_real_lora_update_preserve_frozen_modules(self):
        with tempfile.TemporaryDirectory() as directory, torch.random.fork_rng():
            torch.manual_seed(23)
            checkpoint = Path(directory) / "planner_idm.pt"
            _write_checkpoint(checkpoint)
            original = load_fada_policy_checkpoint(checkpoint).policy
            observations = torch.linspace(-1.0, 1.0, 30 * 66).reshape(1, 30, 66)
            actions = torch.linspace(-0.2, 0.2, 30 * 29).reshape(1, 30, 29)
            command = torch.tensor([[0.4, 0.0, 0.0]])
            reference = FADATorchRuntime(original).infer(observations, actions, command)

            for method in ("fada", "expert_difference"):
                with self.subTest(method=method):
                    cfg = G1FADAPlannerIDMPolicyCfg(
                        checkpoint_path=str(checkpoint),
                        online_adaptation=FADAOnlineAdaptationCfg(method=method),
                    )
                    runtime = prepare_fada_trainable_runtime(cfg)
                    snapshot = runtime.infer_with_snapshot(observations, actions, command)
                    torch.testing.assert_close(snapshot.action_snapshot, reference)
                    self.assertFalse(snapshot.action_snapshot.requires_grad)
                    before = {name: value.detach().clone() for name, value in runtime.policy.named_parameters()}
                    adapters = dict(fada_adapter_named_parameters(runtime.policy))
                    trainable = {name for name, value in runtime.policy.named_parameters() if value.requires_grad}
                    self.assertEqual(trainable, {name for name in before if "lora_" in name})
                    base_before = None
                    if method == "expert_difference":
                        base = runtime.base_idm
                        self.assertIsNotNone(base)
                        self.assertFalse(base.training)
                        self.assertTrue(all(not value.requires_grad for value in base.parameters()))
                        base_before = {name: value.clone() for name, value in base.state_dict().items()}
                        execution_storage = {value.data_ptr() for value in runtime.policy.parameters()}
                        self.assertTrue(all(value.data_ptr() not in execution_storage for value in base.parameters()))
                    else:
                        self.assertIsNone(runtime.base_idm)

                    # This local SGD step exercises the module, not either future updater.
                    optimizer = torch.optim.SGD(adapters.values(), lr=0.01)
                    target = snapshot.action_snapshot + 0.25
                    predicted = runtime.policy.idm(
                        snapshot.observation_history, snapshot.action_history, snapshot.nominal_future,
                    )[:, 0]
                    loss = (predicted - target).square().mean()
                    loss.backward()
                    self.assertTrue(all(value.grad is None for name, value in runtime.policy.named_parameters()
                                        if name not in adapters))
                    optimizer.step()
                    changed = {name for name, value in runtime.policy.named_parameters()
                               if not torch.equal(value, before[name])}
                    self.assertTrue(changed)
                    self.assertLessEqual(changed, set(adapters))
                    with torch.no_grad():
                        after = runtime.policy.idm(
                            snapshot.observation_history, snapshot.action_history, snapshot.nominal_future,
                        )[:, 0]
                    self.assertLess(float((after - target).square().mean()), float(loss.detach()))
                    if base_before is not None:
                        for name, value in runtime.base_idm.state_dict().items():
                            torch.testing.assert_close(value, base_before[name], rtol=0, atol=0)
                    preserved = snapshot.observation_history.clone()
                    observations.add_(1.0)
                    torch.testing.assert_close(snapshot.observation_history, preserved, rtol=0, atol=0)
                    observations.sub_(1.0)

    def test_invalid_requests_do_not_initialize_hardware(self):
        import robojudo.environment
        from robojudo.pipeline.rl_pipeline import RlPipeline

        class ForbiddenHardware:
            def __init__(self, *args, **kwargs):
                raise AssertionError("hardware constructor must not be reached")

        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "planner_idm.pt"
            _write_checkpoint(checkpoint)
            policy = G1FADAPlannerIDMPolicyCfg(
                checkpoint_path=str(checkpoint),
                online_adaptation=FADAOnlineAdaptationCfg(method="fada"),
            )
            cfg = g1_real_fada_planner_idm(device="cpu", policy=policy)
            # Replacing the dictionary entry avoids importing the actual SDK module.
            with patch.dict(robojudo.environment.__dict__, {"UnitreeCppEnv": ForbiddenHardware}):
                with self.assertRaises(ValueError):
                    RlPipeline(cfg)
                cfg.policy.online_adaptation.method = "none"
                cfg.policy.history_length = 29
                with self.assertRaises(ValueError):
                    RlPipeline(cfg)
                cfg.policy.history_length = 30
                cfg.policy.checkpoint_path = str(Path(directory) / "missing.pt")
                with self.assertRaises(FileNotFoundError):
                    RlPipeline(cfg)
            policy.online_adaptation.method = "expert_difference"
            policy.checkpoint_path = str(Path(directory) / "policy.onnx")
            with self.assertRaises(ValueError):
                prepare_fada_trainable_runtime(policy)


class TestFADACausalCollectors(unittest.TestCase):
    @staticmethod
    def architecture(history=30, horizon=6):
        return FADAArchitectureConfig(
            obs_dim=66, action_dim=29, command_dim=3,
            observation_contract="g1_fada_state_v2",
            history_length=history, prediction_horizon=horizon,
            hidden_dim=16, num_heads=4, planner_layers=1,
            idm_encoder_layers=1, idm_decoder_layers=1, feedforward_dim=32,
        )

    @staticmethod
    def evidence(step, episode=0, **changes):
        evidence = FADAControlEvidence(
            episode_id=episode, control_step=step, policy_version=step // 10,
            sensor_tick=(0xFFFFFFD0 + 1000 * episode + 10 * step) & 0xFFFFFFFF,
            torso_imu_sequence=1000 * episode + step + 1,
            policy_control_active=True, action_dispatched=False,
            lowstate_age_s=0.001, torso_imu_age_s=0.001,
            sensor_read_started_ns=1_000_000_000 + episode * 10_000_000_000 + step * 20_000_000,
        )
        return replace(evidence, **changes)

    @staticmethod
    def snapshot(config, step, command):
        history = torch.arange(step - config.history_length + 1, step + 1, dtype=torch.float32)
        observations = history.clamp_min(0).reshape(1, -1, 1).expand(1, -1, 66).clone()
        previous = torch.where(history > 0, history + 999, torch.zeros_like(history))
        actions = previous.reshape(1, -1, 1).expand(1, -1, 29).clone()
        nominal = torch.arange(step + 1, step + config.prediction_horizon + 1, dtype=torch.float32) + 0.5
        return FADAInferenceSnapshot(
            observations, actions, command,
            nominal.reshape(1, -1, 1).expand(1, -1, 66).clone(),
            torch.full((1, 29), 1000.0 + step),
        )

    def event(self, collector, step, *, episode=0, command_value=0.4,
              dispatched=True, **evidence_changes):
        command = torch.tensor([[command_value, 0.0, 0.0]])
        evidence = self.evidence(step, episode, **evidence_changes)
        result = collector.observe(torch.full((1, 66), float(step)), command, evidence)
        submission = replace(
            evidence, action_dispatched=dispatched,
            action_submitted_ns=evidence.sensor_read_started_ns + 1_000_000,
        )
        recorded = collector.record_action(self.snapshot(collector.config, step, command), submission)
        return result, recorded

    def test_simulation_frames_require_execution_and_cannot_cross_reset(self):
        config = self.architecture(history=3, horizon=2)

        def simulation(step, episode=0, **changes):
            read = self.evidence(step, episode).sensor_read_started_ns
            values = dict(
                source="simulation", sensor_tick=None, torso_imu_sequence=None,
                lowstate_age_s=None, torso_imu_age_s=None,
                simulation_frame_id=step, simulation_state_ready_ns=read - 1_000_000,
            )
            values.update(changes)
            return values

        read0 = self.evidence(0).sensor_read_started_ns
        cases = (
            ({"simulation_frame_id": 0}, "duplicate_or_reordered_simulation_frame"),
            ({"simulation_frame_id": 2}, "simulation_frame_gap"),
            ({"simulation_state_ready_ns": read0 + 500_000}, "frame_not_proven_after_submission"),
        )
        for changes, reason in cases:
            with self.subTest(reason=reason):
                collector = FADALegacyTrajectoryCollector(config, max_sensor_age_s=0.02)
                self.event(collector, 0, **simulation(0))
                result, recorded = self.event(collector, 1, **simulation(1, **changes))
                self.assertFalse(result.accepted)
                self.assertEqual(result.rejection_reason, reason)
                self.assertFalse(recorded.accepted)
                collector.reset(episode_id=1)
                samples = []
                for step in range(5):
                    result, recorded = self.event(collector, step, episode=1, **simulation(step, 1))
                    self.assertTrue(result.accepted, result.rejection_reason)
                    self.assertTrue(recorded.accepted, recorded.rejection_reason)
                    if result.sample is not None:
                        samples.append(result.sample)
                self.assertEqual(samples[0].evidence.episode_id, 1)
                torch.testing.assert_close(
                    samples[0].observation_history[0, :, 0], torch.tensor([0., 1., 2.]),
                )
                torch.testing.assert_close(
                    samples[0].realized_future[0, :, 0], torch.tensor([3., 4.]),
                )
                torch.testing.assert_close(
                    samples[0].executed_action_chunk[0, :, 0], torch.tensor([1002., 1003.]),
                )
        hardware = FADALegacyTrajectoryCollector(config, max_sensor_age_s=0.02)
        result, _ = self.event(hardware, 0, sensor_tick=None)
        self.assertEqual(result.rejection_reason, "missing_sensor_evidence")
        with self.assertRaises(ValueError):
            self.evidence(0, simulation_frame_id=0)

    def test_exact_causal_slices_tick_wrap_and_late_acknowledgement(self):
        config = self.architecture()
        legacy = FADALegacyTrajectoryCollector(config, max_sensor_age_s=0.04)
        expert = FADAExpertTrajectoryCollector(config, max_sensor_age_s=0.04)
        old_samples, new_samples = [], []
        for step in range(41):
            for collector, samples in ((legacy, old_samples), (expert, new_samples)):
                result, recorded = self.event(collector, step)
                self.assertTrue(result.accepted, result.rejection_reason)
                self.assertTrue(recorded.accepted, recorded.rejection_reason)
                if result.sample is not None:
                    samples.append(result.sample)
        self.assertEqual([sample.evidence.control_step for sample in old_samples], list(range(29, 35)))
        first = old_samples[0]
        torch.testing.assert_close(first.observation_history[0, :, 0], torch.arange(30, dtype=torch.float32))
        torch.testing.assert_close(
            first.action_history[0, :, 0],
            torch.cat((torch.zeros(1), torch.arange(1000, 1029, dtype=torch.float32))),
        )
        torch.testing.assert_close(first.realized_future[0, :, 0], torch.arange(30, 36, dtype=torch.float32))
        torch.testing.assert_close(first.executed_action_chunk[0, :, 0], torch.arange(1029, 1035, dtype=torch.float32))
        self.assertEqual(new_samples[0].evidence.control_step, 0)
        torch.testing.assert_close(new_samples[0].realized_future[0, :, 0], torch.arange(1, 7, dtype=torch.float32))
        torch.testing.assert_close(
            new_samples[0].snapshot.nominal_future[0, :, 0], torch.arange(1, 7, dtype=torch.float32) + 0.5,
        )
        submitted = {(sample.evidence.episode_id, sample.evidence.control_step) for sample in expert.get_window()}
        self.assertEqual(submitted, {(0, origin) for origin in range(25, 35)})
        for step in (41, 42):
            self.event(expert, step)
        expert.acknowledge(submitted)
        self.assertEqual([sample.evidence.control_step for sample in expert.get_window()], [35, 36])
        result, _ = self.event(expert, 43)
        self.assertEqual(result.sample.evidence.control_step, 37)
        self.assertEqual([sample.evidence.control_step for sample in expert.get_window()], [35, 36, 37])

    def test_command_future_boundary_and_reset(self):
        config = self.architecture()
        legacy = FADALegacyTrajectoryCollector(config, max_sensor_age_s=0.04)
        expert = FADAExpertTrajectoryCollector(config, max_sensor_age_s=0.04)
        old_samples = []
        for step in range(41):
            command = 0.4 if step < 32 else 0.0
            result, _ = self.event(legacy, step, command_value=command)
            self.event(expert, step, command_value=command)
            if result.sample is not None:
                old_samples.append(result.sample)
        self.assertEqual([sample.evidence.control_step for sample in old_samples], [32, 33, 34])
        self.assertEqual(float(old_samples[0].observation_history[0, 0, 0]), 3.0)
        self.assertEqual(
            [sample.evidence.control_step for sample in expert.get_window()],
            list(range(20, 27)) + [32, 33, 34],
        )
        legacy.reset(episode_id=1)
        expert.reset(episode_id=1)
        self.assertEqual(expert.get_window(), ())
        reset_old, reset_new = [], []
        for step in range(36):
            for collector, samples in ((legacy, reset_old), (expert, reset_new)):
                result, _ = self.event(collector, step, episode=1)
                if result.sample is not None:
                    samples.append(result.sample)
        self.assertEqual((reset_old[0].evidence.episode_id, reset_old[0].evidence.control_step), (1, 29))
        self.assertEqual(float(reset_old[0].action_history[0, 0, 0]), 0.0)
        self.assertEqual((reset_new[0].evidence.episode_id, reset_new[0].evidence.control_step), (1, 0))
        torch.testing.assert_close(reset_new[0].realized_future[0, :, 0], torch.arange(1, 7, dtype=torch.float32))

    def test_rejected_frames_cannot_be_compressed_into_futures(self):
        config = self.architecture(history=3, horizon=2)
        cases = (
            (4, {"lowstate_age_s": None}, "missing_sensor_evidence"),
            (4, {"sensor_tick": self.evidence(3).sensor_tick}, "duplicate_or_reordered_tick"),
            (4, {"torso_imu_sequence": self.evidence(3).torso_imu_sequence}, "duplicate_or_reordered_torso_imu"),
            (4, {"lowstate_age_s": 0.05}, "stale_sensor_frame"),
            (4, {"lowstate_age_s": 0.0195}, "frame_not_proven_after_submission"),
            (5, {}, "control_step_gap"),
            (4, {"policy_control_active": False}, "policy_control_inactive"),
        )
        for bad_step, changes, reason in cases:
            with self.subTest(reason=reason):
                collector = FADAExpertTrajectoryCollector(config, max_sensor_age_s=0.04)
                for step in range(4):
                    self.event(collector, step)
                result, recorded = self.event(collector, bad_step, **changes)
                self.assertFalse(result.accepted)
                self.assertEqual(result.rejection_reason, reason)
                self.assertFalse(recorded.accepted)
                for step in range(bad_step + 1, bad_step + 6):
                    self.event(collector, step)
                self.assertEqual(
                    [sample.evidence.control_step for sample in collector.get_window()],
                    [0, 1] + list(range(bad_step + 1, bad_step + 4)),
                )
                recovered = collector.get_window()[2]
                torch.testing.assert_close(
                    recovered.realized_future[0, :, 0],
                    torch.tensor([float(bad_step + 2), float(bad_step + 3)]),
                )
        for dispatched in (True, False):
            with self.subTest(ledger_dispatched=dispatched):
                collector = FADALegacyTrajectoryCollector(config, max_sensor_age_s=0.04)
                for step in range(4):
                    self.event(collector, step)
                changes = {"lowstate_age_s": 0.05} if dispatched else {}
                _, recorded = self.event(collector, 4, dispatched=dispatched, **changes)
                self.assertFalse(recorded.accepted)
                recovered = []
                for step in range(5, 10):
                    result, _ = self.event(collector, step)
                    if result.sample is not None:
                        recovered.append(result.sample)
                self.assertEqual(recovered[0].evidence.control_step, 7)
                self.assertEqual(float(recovered[0].action_history[0, 0, 0]), 1004.0 if dispatched else 1003.0)
        collector = FADALegacyTrajectoryCollector(config, max_sensor_age_s=0.04)
        late_start = []
        for step in range(5, 11):
            result, _ = self.event(collector, step)
            if result.sample is not None:
                late_start.append(result.sample)
        self.assertEqual(late_start[0].evidence.control_step, 8)
        self.assertEqual(float(late_start[0].action_history[0, 0, 0]), 1005.0)


class TestFADAOnlineUpdaters(unittest.TestCase):
    @staticmethod
    def runtime(method):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "planner_idm.pt"
            _write_checkpoint(checkpoint)
            return prepare_fada_trainable_runtime(G1FADAPlannerIDMPolicyCfg(
                checkpoint_path=str(checkpoint),
                online_adaptation=FADAOnlineAdaptationCfg(
                    method=method, lora_rank=2, lora_alpha=4.0, lora_dropout=0.4,
                ),
            ))

    @staticmethod
    def pairs(runtime, count=2):
        config = runtime.policy.config
        pairs = []
        for step in range(count):
            snapshot = runtime.infer_with_snapshot(
                torch.randn(1, config.history_length, 66) * 0.2,
                torch.randn(1, config.history_length, 29) * 0.1,
                torch.tensor([[0.4, 0.1 * step, 0.0]]),
            )
            realized = snapshot.nominal_future + torch.linspace(
                -0.6, 0.9, config.prediction_horizon * 66,
            ).reshape(1, config.prediction_horizon, 66)
            pairs.append(FADATrajectoryPair(
                snapshot, realized, TestFADACausalCollectors.evidence(step),
            ))
        return tuple(pairs)

    def assert_only_adapters_changed(self, runtime, before):
        adapters = dict(fada_adapter_named_parameters(runtime.policy))
        changed = {name for name, parameter in runtime.policy.named_parameters()
                   if not torch.equal(parameter, before[name])}
        self.assertTrue(changed)
        self.assertLessEqual(changed, set(adapters))
        self.assertTrue(all(parameter.grad is None for name, parameter in runtime.policy.named_parameters()
                            if name not in adapters))

    def test_legacy_weighted_gradients_dropout_and_partial_batch(self):
        with torch.random.fork_rng():
            torch.manual_seed(51)
            runtime = self.runtime("fada")
            adapters = dict(fada_adapter_named_parameters(runtime.policy))
            with torch.no_grad():
                for name, parameter in adapters.items():
                    if "_B." in name:
                        parameter.normal_(std=0.03)
            updater = FADALegacyUpdater(runtime)
            pairs = self.pairs(runtime)
            windows = tuple(FADACausalWindow(
                pair.snapshot.observation_history, pair.snapshot.action_history,
                pair.snapshot.command, pair.realized_future,
                pair.snapshot.action_snapshot[:, None].expand(-1, 6, -1) + 0.2 * (i + 1),
                pair.evidence,
            ) for i, pair in enumerate(pairs))
            before = {name: parameter.detach().clone()
                      for name, parameter in runtime.policy.named_parameters()}
            self.assertIsNone(updater.update())
            for window in windows:
                updater.append_window(window)

            runtime.policy.idm.train()
            torch.manual_seed(103)
            squared_errors = []
            for window in windows:
                chunk = runtime.policy.idm(
                    window.observation_history, window.action_history, window.realized_future,
                )
                squared_errors.append((chunk[:, 0] - window.executed_action_chunk[:, 0]).square().sum())
            expected = (0.99 * squared_errors[0] + squared_errors[1]) / (1.99 * 29)
            expected_gradients = torch.autograd.grad(expected, tuple(adapters.values()))
            runtime.policy.idm.eval()
            torch.manual_seed(103)
            with torch.no_grad():
                actual = updater.compute_loss()
            self.assertFalse(runtime.policy.idm.training)
            torch.testing.assert_close(actual, expected)
            for actual_grad, expected_grad in zip(
                torch.autograd.grad(actual, tuple(adapters.values())), expected_gradients,
            ):
                torch.testing.assert_close(actual_grad, expected_grad, atol=1e-6, rtol=1e-5)
            with torch.no_grad():
                stats = updater.update()
            self.assertEqual(stats["num_windows"], 2)
            self.assert_only_adapters_changed(runtime, before)
            self.assertIsNone(updater.update())

            # A shape-valid sample for a different H must fail before Adam and stay pending.
            invalid = replace(windows[0],
                              observation_history=windows[0].observation_history[:, 1:],
                              action_history=windows[0].action_history[:, 1:])
            updater.append_window(invalid)
            before_failure = {name: parameter.detach().clone() for name, parameter in adapters.items()}
            for _ in range(2):
                with self.assertRaises(ValueError):
                    updater.update()
            for name, parameter in adapters.items():
                torch.testing.assert_close(parameter, before_failure[name], rtol=0, atol=0)
            self.assertFalse(runtime.policy.idm.training)
            with self.assertRaises(ValueError):
                FADAExpertUpdater(runtime)

    def test_expert_fixed_targets_nominal_gradients_and_anchor(self):
        with torch.random.fork_rng():
            torch.manual_seed(61)
            runtime = self.runtime("expert_difference")
            adapters = dict(fada_adapter_named_parameters(runtime.policy))
            with torch.no_grad():
                for name, parameter in adapters.items():
                    if "_B." in name:
                        parameter.normal_(std=0.03)
            initial = {name: parameter.detach().clone() for name, parameter in adapters.items()}
            updater = FADAExpertUpdater(runtime, sigma=0.3)
            pairs = self.pairs(runtime)
            samples = build_expert_samples(runtime.base_idm, pairs, gamma=0.2)
            with torch.no_grad():
                for pair, sample in zip(pairs, samples):
                    snapshot = pair.snapshot
                    nominal = runtime.base_idm(
                        snapshot.observation_history, snapshot.action_history, snapshot.nominal_future,
                    )[:, 0]
                    realized = runtime.base_idm(
                        snapshot.observation_history, snapshot.action_history, pair.realized_future,
                    )[:, 0]
                    torch.testing.assert_close(
                        sample.target_action, snapshot.action_snapshot + 0.2 * (nominal - realized),
                    )
                    self.assertFalse(sample.target_action.requires_grad)
                next(iter(adapters.values())).add_(0.04)
            targets = torch.cat([sample.target_action for sample in samples])
            observations = torch.cat([pair.snapshot.observation_history for pair in pairs])
            actions = torch.cat([pair.snapshot.action_history for pair in pairs])
            nominal = torch.cat([pair.snapshot.nominal_future for pair in pairs])
            predicted = runtime.policy.idm(observations, actions, nominal)[:, 0]
            error = (predicted - targets).flatten()
            expected_regression = torch.dot(error, error) / (2 * len(samples))
            displacement = torch.cat([(parameter - initial[name]).flatten()
                                      for name, parameter in adapters.items()])
            expected_anchor = 0.3 * torch.dot(displacement, displacement) / 2
            expected = expected_regression + expected_anchor
            expected_gradients = torch.autograd.grad(expected, tuple(adapters.values()))
            runtime.policy.idm.train()
            with torch.no_grad():
                total, regression, anchor = updater.compute_loss(samples)
            self.assertTrue(runtime.policy.idm.training)
            torch.testing.assert_close(regression, expected_regression)
            torch.testing.assert_close(anchor, expected_anchor)
            torch.testing.assert_close(total, expected)
            for actual_grad, expected_grad in zip(
                torch.autograd.grad(total, tuple(adapters.values())), expected_gradients,
            ):
                torch.testing.assert_close(actual_grad, expected_grad, atol=1e-6, rtol=1e-5)

            before = {name: parameter.detach().clone()
                      for name, parameter in runtime.policy.named_parameters()}
            base_before = {name: value.clone() for name, value in runtime.base_idm.state_dict().items()}
            with torch.no_grad():
                stats = updater.update(samples)
            self.assertEqual(stats["num_samples"], 2)
            self.assert_only_adapters_changed(runtime, before)
            for name, value in runtime.base_idm.state_dict().items():
                torch.testing.assert_close(value, base_before[name], rtol=0, atol=0)
            for name, value in updater.theta_base.items():
                torch.testing.assert_close(value, initial[name], rtol=0, atol=0)
            torch.testing.assert_close(torch.cat([sample.target_action for sample in samples]),
                                       targets, rtol=0, atol=0)
            runtime.policy.idm.eval()
            error_after = (runtime.policy.idm(observations, actions, nominal)[:, 0] - targets).flatten()
            torch.testing.assert_close(updater.compute_loss(samples)[1],
                                       torch.dot(error_after, error_after) / (2 * len(samples)))
            with self.assertRaises(ValueError):
                FADALegacyUpdater(runtime)

    def test_zero_residual_does_not_decay_nonzero_deployment_parameters(self):
        with torch.random.fork_rng():
            torch.manual_seed(71)
            runtime = self.runtime("expert_difference")
            adapters = dict(fada_adapter_named_parameters(runtime.policy))
            with torch.no_grad():
                for parameter in adapters.values():
                    parameter.fill_(0.1)
            updater = FADAExpertUpdater(runtime, sigma=0.3)
            pair = self.pairs(runtime, count=1)[0]
            snapshot = pair.snapshot
            exact_target = runtime.policy.idm(
                snapshot.observation_history, snapshot.action_history, snapshot.nominal_future,
            )[:, 0].detach()
            sample = FADAOnlineSample(pair, exact_target)
            before = {name: parameter.detach().clone() for name, parameter in adapters.items()}
            with torch.no_grad():
                stats = updater.update((sample,))
            self.assertEqual(stats["loss"], 0.0)
            for name, parameter in adapters.items():
                torch.testing.assert_close(parameter, before[name], rtol=0, atol=0)


def _controlled_fada_learner(initial, config, jobs, results, stopping, *,
                             entered=None, release=None, corrupt=False, partial_reply=False):
    """Test-only IPC gates/fault injection around the real spawn learner."""
    class Requests:
        def get(self, **kwargs):
            job = jobs.get(**kwargs)
            if entered is not None and not partial_reply:
                entered.set()
                if not release.wait(timeout=30):
                    raise TimeoutError("test job gate was not released")
            return job

    class Results:
        def send(self, message):
            if corrupt and isinstance(message, FADAUpdateResult):
                values = _load(message.adapters)
                values.pop(next(iter(values)))
                message = replace(message, adapters=_dump(values))
            if partial_reply and isinstance(message, FADAUpdateResult):
                import os
                import struct
                from multiprocessing.reduction import ForkingPickler
                payload = ForkingPickler.dumps(message)
                os.write(results.fileno(), struct.pack("!i", len(payload)))
                os.write(results.fileno(), payload[:8])
                entered.set()
                release.wait(timeout=30)
                raise RuntimeError("injected truncated reply")
            results.send(message)

    _real_fada_learner_main(initial, config, Requests(), Results(), stopping)


class TestFADAOnlineWorker(unittest.TestCase):
    @staticmethod
    def prepared(method, **overrides):
        cfg = FADAOnlineAdaptationCfg(
            method=method, lora_rank=2, lora_alpha=4.0, lora_dropout=0.0,
            max_sensor_age_s=0.01, learning_rate=0.01, sigma=0.5,
            **overrides,
        )
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "planner_idm.pt"
            _write_checkpoint(checkpoint)
            runtime = prepare_fada_trainable_runtime(G1FADAPlannerIDMPolicyCfg(
                checkpoint_path=str(checkpoint), online_adaptation=cfg,
            ))
        return runtime, cfg

    def event(self, coordinator, step):
        episode = coordinator.episode_id
        config = coordinator.runtime.policy.config
        command = torch.tensor([[0.4, 0.0, 0.0]])
        indices = torch.arange(step - config.history_length + 1, step + 1).clamp_min(0)
        observations = torch.sin(indices[None, :, None] * 0.02 + torch.arange(66) * 0.1)
        state = observations[:, -1]
        actions = torch.stack([
            self.issued.get((episode, int(t) - 1), torch.zeros(1, 29))
            if int(t) > 0 else torch.zeros(1, 29) for t in indices
        ], dim=1)
        evidence = TestFADACausalCollectors.evidence(
            step, episode=episode, policy_version=coordinator.policy_version,
        )
        collected = coordinator.observe(state, command, evidence)
        self.assertTrue(collected.accepted, collected.rejection_reason)
        job = coordinator.try_submit()
        snapshot = coordinator.runtime.infer_with_snapshot(observations, actions, command)
        sent = replace(evidence, action_dispatched=True,
                       action_submitted_ns=evidence.sensor_read_started_ns + 1_000_000)
        recorded = coordinator.record_action(snapshot, sent)
        self.assertTrue(recorded.accepted, recorded.rejection_reason)
        self.issued[(episode, step)] = snapshot.action_snapshot
        self.probe = snapshot
        return job

    def completion(self, coordinator):
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            self.assertTrue(coordinator.worker.wait_for_message(deadline - time.monotonic()))
            result = coordinator.poll()
            if result is not None:
                return result
        self.fail("learner did not produce a completion")

    def reference_step(self, runtime, updater, job, cfg, result):
        samples = _unpack_samples(runtime.method, job.data, "cpu")
        if isinstance(updater, FADALegacyUpdater):
            for sample in samples:
                updater.append_window(sample)
            updater.update()
        else:
            updater.update(build_expert_samples(runtime.base_idm, samples, gamma=cfg.gamma))
        published = _load(result.adapters)
        for name, parameter in fada_adapter_named_parameters(runtime.policy):
            torch.testing.assert_close(published[name], parameter, rtol=1e-6, atol=1e-7)

    def exercise_real_method(self, method):
        self.issued = {}
        with torch.random.fork_rng():
            torch.manual_seed(91)
            runtime, cfg = self.prepared(method)
            coordinator = FADAOnlineCoordinator(runtime, cfg)
            reference = _restore_runtime(coordinator.worker._initial, "cpu")
            reference_updater = (FADALegacyUpdater(reference, lr=cfg.learning_rate)
                                 if method == "fada" else
                                 FADAExpertUpdater(reference, lr=cfg.learning_rate, sigma=cfg.sigma))
            frozen_before = {n: p.detach().clone() for n, p in runtime.policy.named_parameters()
                             if not p.requires_grad}
            base_before = None if runtime.base_idm is None else {
                n: p.clone() for n, p in runtime.base_idm.state_dict().items()
            }
            context = coordinator.worker._context
            entered, release = context.Event(), context.Event()
            target = functools.partial(_controlled_fada_learner, entered=entered, release=release)
            try:
                with patch("robojudo.policy.fada.online_worker._learner_main", target):
                    coordinator.start()
                pid = coordinator.worker.process.pid
                first = None
                for step in range(41):
                    job = self.event(coordinator, step)
                    if step < 40:
                        self.assertIsNone(job)
                    else:
                        first = job
                self.assertIsNotNone(first)
                self.assertEqual(len(first.identity.sample_ids), 6 if method == "fada" else 10)
                self.assertTrue(entered.wait(timeout=10))
                actor_before = {n: p.detach().clone() for n, p in fada_adapter_named_parameters(runtime.policy)}
                for step in range(41, 47):
                    self.assertIsNone(self.event(coordinator, step))
                self.assertEqual(coordinator.status["state"], "BUSY")
                probe_before = runtime.infer(self.probe.observation_history, self.probe.action_history,
                                             self.probe.command)
                current_ids = (None if method == "fada" else {
                    (s.evidence.episode_id, s.evidence.control_step)
                    for s in coordinator.collector.get_window()
                })
                release.set()
                result1 = self.completion(coordinator)
                self.reference_step(reference, reference_updater, first, cfg, result1)
                for name, parameter in fada_adapter_named_parameters(runtime.policy):
                    torch.testing.assert_close(parameter, actor_before[name], rtol=0, atol=0)
                if method == "expert_difference":
                    remaining = {(s.evidence.episode_id, s.evidence.control_step)
                                 for s in coordinator.collector.get_window()}
                    self.assertEqual(remaining, current_ids - set(first.identity.sample_ids))
                else:
                    self.assertEqual(coordinator.status["pending_samples"], 6)
                self.assertEqual(coordinator.policy_version, 0)
                self.assertTrue(coordinator.apply_pending_at_step_boundary())
                published = _load(result1.adapters)
                for name, parameter in fada_adapter_named_parameters(runtime.policy):
                    torch.testing.assert_close(parameter, published[name], rtol=0, atol=0)
                probe_after = runtime.infer(self.probe.observation_history, self.probe.action_history,
                                            self.probe.command)
                self.assertGreater(float(torch.linalg.vector_norm(probe_after - probe_before)), 0)

                entered.clear()
                release.clear()
                second = None
                for step in range(47, 51):
                    second = self.event(coordinator, step)
                self.assertIsNotNone(second)
                self.assertTrue(entered.wait(timeout=10))
                for step in range(51, 57):
                    self.event(coordinator, step)
                coordinator.reset()
                self.assertEqual(coordinator.status["pending_samples"], 6 if method == "fada" else 0)
                self.assertFalse(coordinator.apply_pending_at_step_boundary())
                for step in range(7):
                    self.event(coordinator, step)
                new_ids = (None if method == "fada" else {
                    (s.evidence.episode_id, s.evidence.control_step)
                    for s in coordinator.collector.get_window()
                })
                release.set()
                result2 = self.completion(coordinator)
                self.reference_step(reference, reference_updater, second, cfg, result2)
                self.assertEqual(result2.identity.episode_id, 0)
                self.assertEqual(coordinator.episode_id, 1)
                self.assertEqual(coordinator.worker.process.pid, pid)
                if method == "expert_difference":
                    self.assertEqual({(s.evidence.episode_id, s.evidence.control_step)
                                      for s in coordinator.collector.get_window()}, new_ids)
                self.assertTrue(coordinator.apply_pending_at_step_boundary())
                self.assertEqual(coordinator.policy_version, 2)
                for name, parameter in runtime.policy.named_parameters():
                    if name in frozen_before:
                        torch.testing.assert_close(parameter, frozen_before[name], rtol=0, atol=0)
                if base_before is not None:
                    for name, value in runtime.base_idm.state_dict().items():
                        torch.testing.assert_close(value, base_before[name], rtol=0, atol=0)

                entered.clear()
                release.clear()
                third = None
                for step in range(7, 41):
                    job = self.event(coordinator, step)
                    if job is not None:
                        third = job
                        break
                self.assertIsNotNone(third)
                self.assertTrue(entered.wait(timeout=10))
                coordinator.request_stop()
                self.assertIsNone(coordinator.try_submit())
                self.assertFalse(coordinator.apply_pending_at_step_boundary())
                coordinator.close()  # Deliberately blocked learner exercises bounded termination.
                self.assertFalse(coordinator.worker.process.is_alive())
                self.assertEqual(coordinator.policy_version, 2)
                print("PUBLIC_WORKER_SMOKE", method, "first_samples=", len(first.identity.sample_ids),
                      "completed/applied_versions=2", "reset_retains_adam=True", "child_reaped=True")
            finally:
                if coordinator.worker.process is not None and coordinator.worker.process.is_alive():
                    release.set()
                coordinator.close()

    def test_real_methods_publication_and_reset(self):
        for method in ("fada", "expert_difference"):
            with self.subTest(method=method):
                self.exercise_real_method(method)

    def test_bad_publication_never_mutates_or_acknowledges(self):
        self.issued = {}
        runtime, cfg = self.prepared("expert_difference", window_size=1, warmup_steps=0, update_interval=1)
        coordinator = FADAOnlineCoordinator(runtime, cfg)
        before = {n: p.detach().clone() for n, p in runtime.policy.named_parameters()}
        try:
            with patch("robojudo.policy.fada.online_worker._learner_main",
                       functools.partial(_controlled_fada_learner, corrupt=True)):
                coordinator.start()
            for step in range(7):
                self.event(coordinator, step)
            self.assertTrue(coordinator.worker.wait_for_message(10))
            with self.assertRaises(FADALearnerError):
                coordinator.poll()
            self.assertEqual(coordinator.status["state"], "FAILED")
            self.assertEqual(coordinator.policy_version, 0)
            self.assertEqual(len(coordinator.collector.get_window()), 1)
            self.assertEqual(coordinator.status["inflight_samples"], 1)
            for name, parameter in runtime.policy.named_parameters():
                torch.testing.assert_close(parameter, before[name], rtol=0, atol=0)
        finally:
            coordinator.close()
        self.assertFalse(coordinator.worker.process.is_alive())

    def test_timeout_and_capacity_are_terminal(self):
        self.issued = {}
        runtime, cfg = self.prepared("fada")
        coordinator = FADAOnlineCoordinator(runtime, cfg)
        entered, release = coordinator.worker._context.Event(), coordinator.worker._context.Event()
        try:
            with patch("robojudo.policy.fada.online_worker._learner_main",
                       functools.partial(_controlled_fada_learner, entered=entered, release=release)):
                coordinator.start()
            for step in range(41):
                self.event(coordinator, step)
            self.assertTrue(entered.wait(timeout=10))
            self.assertFalse(coordinator.worker.wait_for_message(0))
            coordinator.worker._submitted_at -= cfg.update_timeout_s + 1
            with self.assertRaises(FADALearnerError):
                coordinator.poll()
            self.assertTrue(coordinator.worker.process.is_alive())
            self.assertEqual(coordinator.status["state"], "FAILED")
            self.assertEqual(coordinator.status["inflight_samples"], 6)
            self.assertEqual(coordinator.policy_version, 0)
        finally:
            if coordinator.worker.process is not None and coordinator.worker.process.is_alive():
                release.set()
            coordinator.close()
        self.assertFalse(coordinator.worker.process.is_alive())

        runtime, cfg = self.prepared("fada", pending_capacity=10)
        coordinator = FADAOnlineCoordinator(runtime, cfg)
        self.issued = {}
        try:
            # No worker startup: admitted raw data must still have bounded mature storage.
            for step in range(45):
                self.event(coordinator, step)
            with self.assertRaises(FADALearnerError):
                self.event(coordinator, 45)
            self.assertEqual(coordinator.status["pending_samples"], 10)
            self.assertEqual(coordinator.status["state"], "FAILED")
        finally:
            coordinator.close()

    def test_real_worker_model_failure_retains_job(self):
        runtime, cfg = self.prepared("fada")
        worker = FADALearnerWorker(runtime, cfg)
        pair = TestFADAOnlineUpdaters.pairs(runtime, count=1)[0]
        s = pair.snapshot
        invalid = FADACausalWindow(
            s.observation_history[:, 1:], s.action_history[:, 1:], s.command,
            pair.realized_future, s.action_snapshot[:, None].expand(-1, 6, -1), pair.evidence,
        )
        before = {n: p.detach().clone() for n, p in runtime.policy.named_parameters()}
        try:
            worker.start()
            job = worker.submit((invalid,), generation=0, episode_id=0, control_step=40, policy_version=0)
            self.assertTrue(worker.wait_for_message(10))
            with self.assertRaises(FADALearnerError):
                worker.poll()
            self.assertEqual(worker.status["state"], "FAILED")
            self.assertEqual(worker.pending_job, job)
            for name, parameter in runtime.policy.named_parameters():
                torch.testing.assert_close(parameter, before[name], rtol=0, atol=0)
        finally:
            worker.close()
        self.assertFalse(worker.process.is_alive())


    def test_partial_reply_and_abrupt_exit_do_not_block_control(self):
        self.issued = {}
        runtime, cfg = self.prepared("expert_difference", window_size=1, warmup_steps=0, update_interval=1)
        coordinator = FADAOnlineCoordinator(runtime, cfg)
        entered, release = coordinator.worker._context.Event(), coordinator.worker._context.Event()
        before = {n: p.detach().clone() for n, p in runtime.policy.named_parameters()}
        try:
            with patch("robojudo.policy.fada.online_worker._learner_main", functools.partial(
                _controlled_fada_learner, entered=entered, release=release, partial_reply=True,
            )):
                coordinator.start()
            for step in range(7):
                self.event(coordinator, step)
            self.assertTrue(entered.wait(timeout=10))
            self.assertIsNone(coordinator.poll())
            output = runtime.infer(self.probe.observation_history, self.probe.action_history, self.probe.command)
            torch.testing.assert_close(output, self.probe.action_snapshot, rtol=0, atol=0)
            coordinator.worker.process.kill()
            coordinator.worker.process.join(timeout=1)
            with self.assertRaises(FADALearnerError):
                coordinator.poll()
            self.assertEqual(coordinator.status["state"], "FAILED")
            self.assertEqual(coordinator.policy_version, 0)
            self.assertEqual(coordinator.status["inflight_samples"], 1)
            for name, parameter in runtime.policy.named_parameters():
                torch.testing.assert_close(parameter, before[name], rtol=0, atol=0)
        finally:
            if coordinator.worker.process is not None and coordinator.worker.process.is_alive():
                release.set()
            coordinator.close()
        self.assertFalse(coordinator.worker.process.is_alive())
        self.assertFalse(coordinator.worker._receiver.is_alive())


@contextmanager
def _fada_sdk_modules(sdk):
    missing = object()
    names = ("unitree_cpp", "robojudo.environment.unitree_cpp_env")
    previous = {name: sys.modules.get(name, missing) for name in names}
    sys.modules["unitree_cpp"] = sdk
    sys.modules.pop(names[1], None)
    try:
        yield
    finally:
        for name, module in previous.items():
            if module is missing:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module


@contextmanager
def _fada_deployment_harness():
    """Real Python deployment owners with a test-only SDK/sensor/actuator boundary."""
    import robojudo.environment as environments
    import robojudo.policy as policies
    import robojudo.pipeline.rl_pipeline as pipeline_module
    from box import Box
    from robojudo.tools.dof import merge_dof_cfgs

    sdk = ModuleType("unitree_cpp")
    sdk.RobotState = type("RobotState", (), {"sensor_evidence_version": 1})
    sdk.SportState = type("SportState", (), {})

    def forbidden_sdk(*args, **kwargs):
        raise AssertionError("an actual SDK controller must never be constructed")
    sdk.UnitreeController = forbidden_sdk
    with _fada_sdk_modules(sdk), \
            patch.dict(environments.env_registry.registered_modules), patch.dict(environments.__dict__):
        sys.modules.pop("robojudo.environment.unitree_cpp_env", None)
        native_module = importlib.import_module("robojudo.environment.unitree_cpp_env")
        events = []

        class NativeBoundary:
            def __init__(self, owner=None):
                self.owner = owner
                self.targets = []
                self.safe = False
                self.fail = False

            def step(self, positions):
                if self.fail:
                    raise OSError("injected command submission failure")
                if self.owner is not None and self.owner.before_send is not None:
                    self.owner.before_send(np.asarray(positions))
                self.targets.append(np.asarray(positions).copy())

            def shutdown(self):
                self.safe = True
                events.append("actuator_safe")

        class SensorsAndActuator(native_module.UnitreeCppEnv):
            last = None

            def __init__(self, cfg_env, device="cpu"):
                type(self).last = self
                self.cfg_env, self.device = cfg_env, device
                self.enabled = cfg_env.act
                self._shutdown_attempted = False
                self._shutdown_error = None
                self.dof_cfg = cfg_env.dof
                self.num_dofs = self.dof_cfg.num_dofs
                self.default_pos = np.asarray(self.dof_cfg.default_pos)
                self._dof_pos = self.default_pos.copy()
                self._base_quat = np.array([0., 0., 0., 1.])
                self.visualizer = None
                self.hand_retarget = None
                self.clock_ns = 1_000_000_000
                self.tick = 0
                self.before_send = None
                self.unitree = NativeBoundary(self)

            def update_dof_cfg(self, override_cfg=None):
                self.dof_cfg = merge_dof_cfgs(self.cfg_env.dof, override_cfg)
                self.default_pos = np.asarray(self.dof_cfg.default_pos)
                self._dof_pos = self.default_pos.copy()

            def self_check(self):
                pass

            def reset(self):
                self._dof_pos = self.default_pos.copy()

            def update(self):
                self.tick += 1
                self.clock_ns += 20_000_000
                self._dof_pos = self.default_pos + 0.02 * np.sin(np.arange(29) * 0.1 + self.tick * 0.02)

            def get_data(self):
                return Box(
                    dof_pos=self.dof_pos.copy(), dof_vel=np.zeros(29),
                    policy_gyro=np.zeros(3), policy_gravity=np.array([0., 0., -1.]),
                    sensor_tick=self.tick, torso_imu_sequence=self.tick,
                    lowstate_age_s=0.001, torso_imu_age_s=0.001,
                    sensor_read_started_ns=self.clock_ns,
                )

        class Commands:
            fail_init = False

            def __init__(self, **kwargs):
                if self.fail_init:
                    raise RuntimeError("injected controller construction failure")
                self.commands = []
                self.interrupt = False
                self.keyboard_events = []

            def get_ctrl_data(self, env_data):
                if self.interrupt:
                    raise KeyboardInterrupt
                events, self.keyboard_events = self.keyboard_events, []
                return Box(COMMANDS=list(self.commands), KeyboardCtrl={"keyboard_event": events})

            def post_step_callback(self, ctrl_data):
                pass

            def press(self, name):
                self.keyboard_events.append({"type": "keyboard", "name": name, "pressed": True})

            def reset(self):
                self.commands.clear()
                self.keyboard_events.clear()

        class ObservedAdapter(FADAPlannerIDMPolicyAdapter):
            instances = []

            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.instances.append(self)

        clocks = SimpleNamespace(
            monotonic_ns=lambda: SensorsAndActuator.last.clock_ns + 1_000_000,
            sleep=lambda _: None,
        )
        fast_prepare = SimpleNamespace(time=time.time, perf_counter=time.perf_counter, sleep=lambda _: None)
        with patch.dict(environments.__dict__, {"UnitreeCppEnv": SensorsAndActuator}), \
                patch.dict(policies.__dict__, {"FADAPlannerIDMPolicyAdapter": ObservedAdapter}), \
                patch.object(pipeline_module, "CtrlManager", Commands), \
                patch.object(pipeline_module, "time", fast_prepare), \
                patch.object(native_module, "time", clocks):
            yield SimpleNamespace(
                Pipeline=pipeline_module.RlPipeline, Env=SensorsAndActuator,
                Commands=Commands, Adapter=ObservedAdapter, native=native_module,
                NativeBoundary=NativeBoundary, events=events,
            )


class TestFADADeploymentIntegration(unittest.TestCase):
    @staticmethod
    def config(checkpoint, method):
        policy = G1FADAPlannerIDMPolicyCfg(
            checkpoint_path=str(checkpoint), execution_action_scale=1.7,
            preserve_state_during_dry_run=(method != "none"),
            online_adaptation=FADAOnlineAdaptationCfg(
                method=method, lora_rank=2, lora_alpha=4.0, lora_dropout=0.0,
                max_sensor_age_s=0.01,
            ),
        )
        cfg = g1_real_fada_planner_idm(device="cpu", policy=policy, do_safety_check=False)
        cfg.debug.log_obs = False
        cfg.env.forward_kinematic = None
        dof = cfg.env.dof.model_copy(deep=True)
        for name in dof.prop_keys:
            setattr(dof, name, list(reversed(getattr(dof, name))))
        cfg.env.dof = dof
        return cfg

    def exercise_mode(self, method):
        with tempfile.TemporaryDirectory() as directory, torch.random.fork_rng(), _fada_deployment_harness() as h:
            torch.manual_seed(101)
            checkpoint = Path(directory) / "paired.pt"
            _write_checkpoint(checkpoint)
            cfg = self.config(checkpoint, method)
            pipeline = h.Pipeline(cfg)
            adapter = pipeline.policy.policy
            coordinator = adapter.online_coordinator
            try:
                pipeline.prepare()
                self.assertEqual(len(pipeline.env.unitree.targets), 1000)
                if coordinator is not None:
                    self.assertEqual(coordinator.status["pending_samples"], 0)
                    self.assertIsNone(coordinator.worker.pending_job)
                    self.assertEqual(coordinator.worker.learner_version, 0)
                    self.assertEqual(adapter._control_step, 0)
                    self.assertEqual(torch.count_nonzero(adapter.playback_controller._action_history), 0)
                else:
                    self.assertIsNone(adapter.online_coordinator)

                sent_queries = []

                def before_send(target):
                    if coordinator is not None:
                        snapshot = adapter._pending_snapshot
                        self.assertIsNotNone(snapshot)
                        torch.testing.assert_close(adapter.playback_controller._action_history,
                                                   snapshot.action_history, rtol=0, atol=0)
                        np.testing.assert_array_equal(adapter.last_action,
                                                      snapshot.action_history[0, -1].numpy())
                        raw = snapshot.action_snapshot[0].numpy()
                        sent_queries.append(snapshot)
                    else:
                        raw = adapter.last_action
                    by_joint = dict(zip(adapter.cfg_action_dof.joint_names,
                                        adapter.default_pos + 1.7 * raw))
                    expected = [by_joint[name] for name in pipeline.env.dof_cfg.joint_names]
                    np.testing.assert_allclose(target, expected, rtol=1e-6, atol=1e-6)

                pipeline.env.before_send = before_send
                if coordinator is None:
                    before = adapter.playback_controller.snapshot_state()
                    frozen = load_fada_policy_checkpoint(checkpoint).policy
                    pipeline.step()
                    histories = adapter.playback_controller._observation_history
                    previous = before["action_history"]
                    if previous is None:
                        previous = torch.zeros(1, 30, 29)
                    with torch.no_grad():
                        expected = frozen(histories, previous, torch.zeros(1, 3)).action[0].numpy()
                    np.testing.assert_allclose(adapter.last_action, expected, rtol=1e-5, atol=1e-6)
                else:
                    pipeline.ctrl_manager.press("w")
                    for _ in range(41):
                        pipeline.step()
                    self.assertEqual(len(sent_queries), 41)
                    self.assertEqual(torch.count_nonzero(sent_queries[0].action_history), 0)
                    self.assertTrue(coordinator.worker.wait_for_message(10))
                    completed = coordinator.poll()
                    self.assertEqual(completed.stats["num_windows" if method == "fada" else "num_samples"],
                                     6 if method == "fada" else 10)
                    self.assertEqual(coordinator.policy_version, 0)
                    pipeline.step()
                    self.assertEqual(coordinator.policy_version, 1)
                    self.assertEqual(coordinator.collector._pending.evidence.policy_version, 1)
                    torch.testing.assert_close(adapter.playback_controller._action_history[:, -1],
                                               sent_queries[-1].action_snapshot)
                    weights = {n: p.detach().clone() for n, p in fada_adapter_named_parameters(coordinator.runtime.policy)}
                    pid, generation = coordinator.worker.process.pid, coordinator.generation
                    pipeline.reset()
                    self.assertEqual(coordinator.generation, generation + 1)
                    self.assertEqual(coordinator.worker.process.pid, pid)
                    pipeline.step()
                    self.assertEqual(torch.count_nonzero(sent_queries[-1].action_history), 0)
                    for name, parameter in fada_adapter_named_parameters(coordinator.runtime.policy):
                        torch.testing.assert_close(parameter, weights[name], rtol=0, atol=0)
                    pipeline.step(dry_run=True)
                    self.assertEqual(adapter._control_step, 0)
                    self.assertIsNone(adapter._pending_snapshot)
                    self.assertEqual(coordinator.worker.learner_version, 1)
                    pipeline.env.before_send = None

                count = len(pipeline.env.unitree.targets)
                if method == "none":
                    pipeline.do_safety_check = True
                    pipeline.env._base_quat = np.array([1., 0., 0., 0.])
                    pipeline.safety_check()
                    pipeline.step()
                elif method == "fada":
                    coordinator.worker.process.terminate()
                    coordinator.worker.process.join(timeout=1)
                    with self.assertRaises(FADALearnerError):
                        pipeline.step()
                else:
                    pipeline.ctrl_manager.commands = ["[SHUTDOWN]"]
                    pipeline.step()
                self.assertTrue(pipeline.env.unitree.safe)
                self.assertTrue(pipeline.should_stop())
                self.assertEqual(len(pipeline.env.unitree.targets), count)
            finally:
                pipeline.close()
            if coordinator is not None:
                self.assertFalse(coordinator.worker.process.is_alive())
                self.assertFalse(coordinator.worker._receiver.is_alive())
            print("DEPLOYMENT_API_SMOKE", method, "prepare_samples=0", "receipt_committed=True",
                  "actuator_safe_before_close=True")

    def test_modes_preserve_control_and_causal_submission(self):
        for method in ("none", "fada", "expert_difference"):
            with self.subTest(method=method):
                self.exercise_mode(method)

    def test_submission_and_constructor_failures_request_safety(self):
        with tempfile.TemporaryDirectory() as directory, _fada_deployment_harness() as h:
            checkpoint = Path(directory) / "paired.pt"
            _write_checkpoint(checkpoint)
            cfg = self.config(checkpoint, "expert_difference")
            pipeline = h.Pipeline(cfg)
            adapter = pipeline.policy.policy
            try:
                pipeline.step()
                before = adapter.last_action.copy()
                confirmed = adapter._control_step
                pipeline.env.unitree.fail = True
                with self.assertRaises(OSError):
                    pipeline.step()
                self.assertEqual(adapter._control_step, confirmed)
                np.testing.assert_array_equal(adapter.last_action, before)
                self.assertIsNone(adapter._pending_snapshot)
                self.assertTrue(pipeline.env.unitree.safe)
            finally:
                pipeline.close()
            self.assertFalse(adapter.online_coordinator.worker.process.is_alive())

            h.Commands.fail_init = True
            with self.assertRaises(RuntimeError):
                h.Pipeline(cfg)
            failed = h.Adapter.instances[-1]
            self.assertTrue(h.Env.last.unitree.safe)
            self.assertFalse(failed.online_coordinator.worker.process.is_alive())
            self.assertFalse(failed.online_coordinator.worker._receiver.is_alive())
            h.Commands.fail_init = False

            cfg.env.act = False
            previous = h.Env.last
            with self.assertRaises(ValueError):
                h.Pipeline(cfg)
            self.assertIs(h.Env.last, previous)
            self.assertIsNone(h.Adapter.instances[-1].online_coordinator.worker.process)

    def test_native_partial_initialization_and_disabled_receipt(self):
        with tempfile.TemporaryDirectory() as directory, _fada_deployment_harness() as h:
            checkpoint = Path(directory) / "paired.pt"
            _write_checkpoint(checkpoint)
            cfg = self.config(checkpoint, "none")
            native = h.NativeBoundary()
            with patch.object(h.native, "UnitreeController", return_value=native), \
                    patch.object(h.native.UnitreeCppEnv, "self_check", side_effect=RuntimeError("self-check failed")):
                with self.assertRaises(RuntimeError):
                    h.native.UnitreeCppEnv(cfg.env)
            self.assertTrue(native.safe)

            env = h.Env(cfg.env)
            env.update()
            stamp = env.step(np.zeros(29))
            self.assertGreater(stamp, env.clock_ns)
            env.enabled = False
            count = len(env.unitree.targets)
            self.assertIsNone(env.step(np.ones(29)))
            self.assertEqual(len(env.unitree.targets), count)
            env.shutdown()
            env.shutdown()
            self.assertTrue(env.unitree.safe)

    def test_main_interrupt_closes_real_online_pipeline(self):
        import robojudo.pipeline as pipelines
        with tempfile.TemporaryDirectory() as directory, _fada_deployment_harness() as h:
            checkpoint = Path(directory) / "paired.pt"
            _write_checkpoint(checkpoint)
            cfg = self.config(checkpoint, "expert_difference")
            cfg.pipeline_type = "FADASmokePipeline"
            captured = []

            class InterruptPipeline(h.Pipeline):
                def __init__(self, **kwargs):
                    super().__init__(**kwargs)
                    captured.append(self)

                def prepare(self):
                    super().prepare()
                    self.ctrl_manager.interrupt = True

            path = Path(__file__).parents[1] / "scripts" / "run_pipeline.py"
            spec = importlib.util.spec_from_file_location("_fada_entry_smoke", path)
            entry = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(entry)
            with patch.dict(pipelines.__dict__, {"FADASmokePipeline": InterruptPipeline}), \
                    patch.object(entry, "parse_args", return_value=SimpleNamespace(config="offline", left_shoulder_load=False)), \
                    patch.object(entry, "ConfigManager", return_value=SimpleNamespace(get_cfg=lambda: cfg)):
                entry.main()
            pipeline = captured[0]
            self.assertTrue(pipeline.env.unitree.safe)
            self.assertTrue(pipeline.should_stop())
            self.assertFalse(pipeline.policy.online_coordinator.worker.process.is_alive())
            self.assertFalse(pipeline.policy.online_coordinator.worker._receiver.is_alive())

    def test_registered_comparison_profiles_run_through_main(self):
        import robojudo.pipeline as pipelines
        from robojudo.config.config_manager import ConfigManager

        profiles = (
            ("g1_real_fada_baseline", "none"),
            ("g1_real_fada_online", "fada"),
            ("g1_real_fada_expert_online", "expert_difference"),
        )
        with tempfile.TemporaryDirectory() as directory, torch.random.fork_rng():
            torch.manual_seed(102)
            checkpoint = Path(directory) / "paired.pt"
            _write_checkpoint(checkpoint)
            original_get_cfg = ConfigManager.get_cfg

            def offline_cfg(manager):
                cfg = original_get_cfg(manager)
                cfg.device = "cpu"
                cfg.run_fullspeed = True
                cfg.debug.log_obs = False
                cfg.env.forward_kinematic = None
                cfg.policy.checkpoint_path = str(checkpoint)
                cfg.policy.online_adaptation.max_sensor_age_s = 0.01
                return cfg

            path = Path(__file__).parents[1] / "scripts" / "run_pipeline.py"
            spec = importlib.util.spec_from_file_location("_fada_registered_entry_smoke", path)
            entry = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(entry)
            for profile, method in profiles:
                with self.subTest(profile=profile), _fada_deployment_harness() as h:
                    captured = []
                    case = self
                    waiting_steps = 53
                    completion_step = waiting_steps + 41
                    reset_waiting_steps = 12
                    retrigger_step = completion_step + reset_waiting_steps + 1
                    stop_step = retrigger_step + 2

                    class SmokePipeline(h.Pipeline):
                        def __init__(self, **kwargs):
                            self.smoke_active = False
                            self.smoke_steps = 0
                            super().__init__(**kwargs)
                            captured.append(self)
                            adapter = self.policy.policy
                            self.probe = (torch.zeros(1, 30, 66), torch.zeros(1, 30, 29),
                                          torch.zeros(1, 3))
                            self.before_output = adapter._runtime["runner"].infer(*self.probe).clone()
                            model = adapter._runtime["model"]
                            parameters = (model.named_parameters() if method == "none"
                                          else fada_adapter_named_parameters(model))
                            self.before_weights = {n: p.detach().clone() for n, p in parameters}

                        def prepare(self):
                            super().prepare()
                            self.smoke_active = True

                        def step(self, dry_run=False):
                            next_step = self.smoke_steps + 1
                            if self.smoke_active and not dry_run and method != "none":
                                if next_step in (waiting_steps + 1, retrigger_step):
                                    self.ctrl_manager.press("w" if next_step == waiting_steps + 1 else "s")
                                elif next_step == retrigger_step + 1:
                                    self.ctrl_manager.press("x")
                            super().step(dry_run=dry_run)
                            if not self.smoke_active or dry_run:
                                return
                            self.smoke_steps += 1
                            adapter = self.policy.policy
                            coordinator = adapter.online_coordinator
                            model = adapter._runtime["model"]
                            if method == "none":
                                case.assertIsNone(coordinator)
                                if self.smoke_steps == 42:
                                    for name, parameter in model.named_parameters():
                                        torch.testing.assert_close(parameter, self.before_weights[name],
                                                                   rtol=0, atol=0)
                                    self.request_stop()
                                return

                            if self.smoke_steps <= waiting_steps:
                                case.assertFalse(adapter._adaptation_started)
                                case.assertIsNone(coordinator.status["collection_step"])
                                case.assertIsNone(coordinator.worker.pending_job)
                                case.assertIsNone(coordinator.collector._pending)
                                case.assertEqual(coordinator.status["pending_samples"], 0)
                                case.assertEqual(coordinator.worker.learner_version, 0)
                                case.assertEqual(adapter._control_step, self.smoke_steps)
                                torch.testing.assert_close(adapter.playback_controller._action_history[0, -1],
                                                           torch.from_numpy(adapter.last_action))
                                if self.smoke_steps == waiting_steps:
                                    self.waiting_history = adapter.playback_controller._action_history.clone()
                                    for name, parameter in fada_adapter_named_parameters(model):
                                        torch.testing.assert_close(parameter, self.before_weights[name],
                                                                   rtol=0, atol=0)
                            elif self.smoke_steps == waiting_steps + 1:
                                case.assertTrue(adapter._adaptation_started)
                                case.assertEqual(coordinator.status["collection_start_control_step"], waiting_steps)
                                case.assertEqual(coordinator.status["collection_step"], 0)
                                torch.testing.assert_close(coordinator.collector._pending.snapshot.action_history,
                                                           self.waiting_history, rtol=0, atol=0)
                            elif self.smoke_steps < completion_step:
                                case.assertIsNone(coordinator.worker.pending_job)
                                case.assertEqual(coordinator.worker.learner_version, 0)
                            elif self.smoke_steps == completion_step:
                                case.assertEqual(coordinator.runtime.method, method)
                                case.assertEqual(coordinator.status["collection_step"], 40)
                                case.assertEqual(coordinator.worker.pending_job.identity.control_step,
                                                 waiting_steps + 40)
                                case.assertTrue(coordinator.worker.wait_for_message(10))
                                case.assertIsNotNone(coordinator.poll())
                                case.assertEqual(coordinator.policy_version, 0)
                                case.assertTrue(coordinator.status["publication_pending"])
                                self.learner_pid = coordinator.worker.process.pid
                                generation = coordinator.generation
                                self.reset()
                                case.assertEqual(coordinator.generation, generation + 1)
                                case.assertFalse(adapter._adaptation_started)
                                case.assertIsNone(coordinator.status["collection_step"])
                            elif self.smoke_steps < retrigger_step:
                                case.assertFalse(adapter._adaptation_started)
                                case.assertIsNone(coordinator.status["collection_step"])
                                case.assertEqual(coordinator.policy_version, 0)
                                case.assertTrue(coordinator.status["publication_pending"])
                                case.assertEqual(coordinator.worker.process.pid, self.learner_pid)
                                for name, parameter in fada_adapter_named_parameters(model):
                                    torch.testing.assert_close(parameter, self.before_weights[name],
                                                               rtol=0, atol=0)
                            else:
                                case.assertTrue(adapter._adaptation_started)
                                case.assertEqual(coordinator.status["collection_start_control_step"],
                                                 reset_waiting_steps)
                                case.assertEqual(coordinator.status["collection_step"],
                                                 self.smoke_steps - retrigger_step)
                                case.assertEqual(coordinator.worker.learner_version, 1)
                                case.assertEqual(coordinator.policy_version, 1)
                                case.assertEqual(coordinator.worker.process.pid, self.learner_pid)
                                case.assertIsNone(coordinator.worker.pending_job)
                                case.assertTrue(any(not torch.equal(p, self.before_weights[n])
                                                    for n, p in fada_adapter_named_parameters(model)))
                                after = adapter._runtime["runner"].infer(*self.probe)
                                case.assertFalse(torch.equal(after, self.before_output))
                                if self.smoke_steps > retrigger_step:
                                    np.testing.assert_array_equal(adapter._keyboard_command, np.zeros(3))
                                if self.smoke_steps == stop_step:
                                    self.request_stop()

                    with patch.object(ConfigManager, "get_cfg", offline_cfg), \
                            patch.dict(pipelines.__dict__, {"RlPipeline": SmokePipeline}), \
                            patch.object(sys, "argv", [str(path), "-c", profile]):
                        entry.main()
                    pipeline = captured[0]
                    self.assertEqual(pipeline.smoke_steps, 42 if method == "none" else stop_step)
                    self.assertTrue(pipeline.env.unitree.safe)
                    coordinator = pipeline.policy.online_coordinator
                    if coordinator is not None:
                        self.assertFalse(coordinator.worker.process.is_alive())
                        self.assertFalse(coordinator.worker._receiver.is_alive())
                    print("REGISTERED_ENTRY_SMOKE", profile, "method=" + method,
                          "frozen_reference=passed" if coordinator is None else "velocity_and_reset_gate=passed",
                          "applied_version=" + str(0 if coordinator is None else coordinator.policy_version),
                          "safe_requested=True")


if __name__ == "__main__":
    unittest.main()
