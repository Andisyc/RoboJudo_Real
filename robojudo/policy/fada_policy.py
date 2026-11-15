from __future__ import annotations

import logging
from dataclasses import replace
from numbers import Integral
from pathlib import Path

import numpy as np
import torch

from robojudo.policy import policy_registry

from .fada.checkpoint import load_fada_policy_checkpoint
from .fada.onnx_runtime import FADAOnnxRuntime, FADATorchRuntime
from .fada.observation import (
    FADATorsoImuProjector,
    FADA_G1_ACTION_DIM,
    FADA_G1_ACTOR_OBS_DIM,
    FADA_G1_COMMAND_DIM,
    FADA_G1_STATE_DIM,
    FADA_G1_STATE_OBSERVATION_CONTRACT,
)
from .fada.playback import FADAPlaybackController
from .fada.online_controller import FADAOnlineCoordinator
from .fada.online_data import FADAControlEvidence
from .fada.online_runtime import prepare_fada_trainable_runtime
from .unilab_policy import UniLabPolicy

logger = logging.getLogger(__name__)

_KEY_PAIRS = (("w", "s"), ("a", "d"), ("q", "e"))
KEY_AXES = {key: (axis, sign) for axis, pair in enumerate(_KEY_PAIRS) for key, sign in zip(pair, (1, -1))}


@policy_registry.register
class FADAPlannerIDMPolicyAdapter(UniLabPolicy):
    """RoboJuDo runtime adapter for a native FADA Planner-IDM checkpoint."""

    _preset_command: np.ndarray | None = None
    _manual_key_speeds: dict[str, float] | None = None

    def __init__(self, cfg_policy, device):
        self.online_coordinator = None
        self._lora_frozen = bool(cfg_policy.online_adaptation.freeze_lora)
        self._control_active = None
        self._control_step = 0
        self._adaptation_started = False
        self._pending_collection = False
        self._last_submitted_moving: bool | None = None
        self._issued_moving = False
        self._restart_motion_blocked = False
        self._pending_snapshot = None
        self._pending_evidence = None
        self._pending_sensor = None
        self._stopping = False
        super().__init__(cfg_policy=cfg_policy, device=device)
        self.gait_phase_enabled = bool(cfg_policy.gait_phase_enabled)
        self.fixed_gait_phase = np.asarray(cfg_policy.fixed_gait_phase, dtype=np.float32)
        if self.fixed_gait_phase.shape != (2,):
            raise ValueError("FADA fixed_gait_phase must contain two values")
        self.execution_action_scale = float(cfg_policy.execution_action_scale)
        if not np.isfinite(self.execution_action_scale) or self.execution_action_scale <= 0.0:
            raise ValueError("FADA execution_action_scale must be finite and positive")
        self.keyboard_command_magnitudes = np.asarray(
            cfg_policy.keyboard_command_magnitudes, dtype=np.float32
        )
        if (
            self.keyboard_command_magnitudes.shape != (3,)
            or not bool(np.isfinite(self.keyboard_command_magnitudes).all())
            or not bool((self.keyboard_command_magnitudes > 0.0).all())
        ):
            raise ValueError(
                "FADA keyboard_command_magnitudes must contain three positive values"
            )
        if cfg_policy.keyboard_speed_overrides:
            self._manual_key_speeds = {
                key: cfg_policy.keyboard_speed_overrides.get(key, float(self.keyboard_command_magnitudes[axis]))
                for key, (axis, _) in KEY_AXES.items()
            }
        self.keyboard_command_mode = cfg_policy.keyboard_command_mode
        self._keyboard_hold_blocked = False
        self._held_motion_keys: set[str] = set()
        self._keyboard_command = np.zeros(3, dtype=np.float32)
        if cfg_policy.preset_velocity_command is not None:
            self._preset_command = np.asarray(cfg_policy.preset_velocity_command, dtype=np.float32)
        self._preset_active = False
        self._preset_key_down = False
        self._preset_wait_release = False
        self.playback_controller = FADAPlaybackController(
            self._runtime["runner"],
            device=self.device,
            history_length=int(cfg_policy.history_length),
            observation_contract=FADA_G1_STATE_OBSERVATION_CONTRACT,
            action_dim=FADA_G1_ACTION_DIM,
            command_dim=FADA_G1_COMMAND_DIM,
        )
        self._torso_imu_projector = FADATorsoImuProjector()
        self._pending_command: np.ndarray | None = None
        self._pending_observation: np.ndarray | None = None
        self.reset()
        if cfg_policy.online_adaptation.method != "none":
            self.online_coordinator = FADAOnlineCoordinator(
                self._runtime["runner"], cfg_policy.online_adaptation,
            )

    def _build_runtime(self, policy_file: str):
        suffix = Path(policy_file).suffix.lower()
        if suffix in {".pt", ".pth"} and torch.device(self.device).type == "cpu":
            # Bound CPU inference jitter before constructing either PT runtime.
            torch.set_num_threads(1)
        expected_contract = (
            FADA_G1_STATE_DIM,
            FADA_G1_ACTION_DIM,
            FADA_G1_COMMAND_DIM,
            int(self.cfg_policy.history_length),
            int(self.cfg_policy.prediction_horizon),
            FADA_G1_STATE_OBSERVATION_CONTRACT,
        )
        if self.cfg_policy.online_adaptation.method != "none":
            runtime = prepare_fada_trainable_runtime(self.cfg_policy, device=self.device)
            return {
                "kind": "fada_planner_idm_online", "model": runtime.policy,
                "runner": runtime, "contract": expected_contract,
            }
        if suffix == ".onnx":
            runner = FADAOnnxRuntime(
                policy_file,
                device=self.device,
                history_length=int(self.cfg_policy.history_length),
                observation_dim=FADA_G1_STATE_DIM,
                action_dim=FADA_G1_ACTION_DIM,
                command_dim=FADA_G1_COMMAND_DIM,
            )
            return {
                "kind": "fada_planner_idm_onnx",
                "runner": runner,
                "contract": expected_contract,
            }
        if suffix not in {".pt", ".pth"}:
            raise ValueError(f"Unsupported FADA Planner-IDM policy format: {policy_file}")
        loaded = load_fada_policy_checkpoint(policy_file, device=self.device)
        config = loaded.policy.config
        return {
            "kind": "fada_planner_idm_torch",
            "model": loaded.policy,
            "runner": FADATorchRuntime(loaded.policy),
            "checkpoint": loaded.checkpoint,
            "contract": (
                config.obs_dim,
                config.action_dim,
                config.command_dim,
                config.history_length,
                config.prediction_horizon,
                config.observation_contract,
            ),
        }

    def _check_runtime_contract(self):
        observed = self._runtime["contract"]
        expected = (
            FADA_G1_STATE_DIM,
            FADA_G1_ACTION_DIM,
            FADA_G1_COMMAND_DIM,
            int(self.cfg_policy.history_length),
            int(self.cfg_policy.prediction_horizon),
            FADA_G1_STATE_OBSERVATION_CONTRACT,
        )
        if observed != expected:
            raise ValueError(f"FADA runtime contract mismatch: expected={expected}, got={observed}")
        if self.expected_obs_dim != FADA_G1_ACTOR_OBS_DIM:
            raise ValueError("FADA adapter requires the 98-D G1 actor observation")
        if self.action_scale != 1.0 or self.action_beta != 1.0 or self.action_clip is not None:
            raise ValueError("FADA playback forbids action scaling, smoothing, and clipping")

    def reset(self):
        super().reset()
        if hasattr(self, "_held_motion_keys"):
            if self._preset_command is not None or self.keyboard_command_mode == "hardware_hold":
                self._cancel_preset()
            else:
                self._held_motion_keys.clear()
                self._keyboard_command.fill(0.0)
                self._keyboard_hold_blocked = False
        if hasattr(self, "gait_phase_enabled") and not self.gait_phase_enabled:
            self.gait_phase = self.fixed_gait_phase.copy()
        if hasattr(self, "playback_controller"):
            self.playback_controller.reset()
        if hasattr(self, "_torso_imu_projector"):
            self._torso_imu_projector.reset()
        self._pending_command = None
        self._pending_observation = None
        self._pending_snapshot = None
        self._pending_evidence = None
        self._pending_sensor = None
        self._control_step = 0
        self._adaptation_started = False
        self._pending_collection = False
        if self.online_coordinator is not None:
            self.online_coordinator.reset()
            self._control_active = False

    def start_online(self):
        if self.online_coordinator is not None:
            self.online_coordinator.start()

    def set_lora_frozen(self, frozen: bool) -> None:
        """Freeze the current actor version; unfreezing waits for a moving command."""
        if type(frozen) is not bool:
            raise TypeError("frozen must be boolean")
        self._lora_frozen = frozen
        if frozen and self.online_coordinator is not None:
            self.online_coordinator.set_updates_enabled(False)
            self._adaptation_started = False
            self._pending_collection = False
        logger.info("[LORA] manual_frozen=%s", frozen)


    def set_control_active(self, active: bool):
        if not active and (self._preset_command is not None or getattr(self, "keyboard_command_mode", None) == "hardware_hold"):
            self._cancel_preset()
        if self.online_coordinator is None:
            self._control_active = bool(active)
            return
        if self._stopping:
            raise RuntimeError("FADA control has stopped")
        if active and self.online_coordinator.worker.state not in ("READY", "BUSY"):
            raise RuntimeError("online FADA learner is not ready for active control")
        if bool(active) != self._control_active:
            if self._control_active is True:
                self.online_coordinator.reset()
            self.playback_controller.reset()
            self.last_action = np.zeros(self.num_actions, dtype=np.float32)
            self._pending_snapshot = None
            self._pending_evidence = None
            self._control_step = 0
            self._adaptation_started = False
            self._pending_collection = False
        self._control_active = bool(active)

    def confirm_action_submission(self, submitted_ns: int | None):
        if self._stopping or self._pending_snapshot is None or self._pending_evidence is None:
            raise RuntimeError("FADA submission requires one active unconfirmed query")
        if isinstance(submitted_ns, bool) or not isinstance(submitted_ns, Integral) or submitted_ns <= 0:
            raise RuntimeError("FADA command was not confirmed by the environment")
        read_started = self._pending_evidence.sensor_read_started_ns
        if read_started is not None and submitted_ns < read_started:
            raise RuntimeError("FADA submission precedes its sensor read")
        snapshot = self._pending_snapshot
        evidence = replace(
            self._pending_evidence, action_dispatched=True,
            action_submitted_ns=int(submitted_ns),
        )
        self.playback_controller.commit_action(snapshot.action_snapshot)
        self.last_action = snapshot.action_snapshot.squeeze(0).cpu().numpy().copy()
        self._last_submitted_moving = self._issued_moving
        if self._pending_collection:
            self.online_coordinator.record_action(snapshot, evidence)
        self._pending_snapshot = None
        self._pending_evidence = None
        self._pending_collection = False
        self._control_step += 1

    def request_stop(self):
        self._stopping = True
        self._pending_snapshot = None
        self._pending_evidence = None
        self._pending_collection = False
        if self.online_coordinator is not None:
            self.online_coordinator.request_stop()

    def close(self):
        self.request_stop()
        if self.online_coordinator is not None:
            self.online_coordinator.close()

    def snapshot_state(self) -> dict[str, object]:
        state: dict[str, object] = dict(super().snapshot_state())
        state.update(
            {
                "held_motion_keys": set(self._held_motion_keys),
                "keyboard_command": self._keyboard_command.copy(),
                "keyboard_hold_blocked": self._keyboard_hold_blocked,
                "pending_command": (
                    None if self._pending_command is None else self._pending_command.copy()
                ),
                "pending_observation": (
                    None
                    if self._pending_observation is None
                    else self._pending_observation.copy()
                ),
                "playback": self.playback_controller.snapshot_state(),
            }
        )
        return state

    def restore_state(self, state: dict[str, object]):
        if self.online_coordinator is not None and self._control_active:
            raise RuntimeError("cannot rewind an active online physical history")
        super().restore_state(state)  # type: ignore[arg-type]
        if self._preset_command is not None:
            self._cancel_preset()
        else:
            self._held_motion_keys = set(state["held_motion_keys"])  # type: ignore[arg-type]
            self._keyboard_hold_blocked = bool(state["keyboard_hold_blocked"])
            self._keyboard_command = np.asarray(state["keyboard_command"], dtype=np.float32).copy()
        pending_command = state.get("pending_command")
        self._pending_command = (
            None
            if pending_command is None
            else np.asarray(pending_command, dtype=np.float32).copy()
        )
        pending_observation = state.get("pending_observation")
        self._pending_observation = (
            None
            if pending_observation is None
            else np.asarray(pending_observation, dtype=np.float32).copy()
        )
        if self._preset_command is not None:
            self._pending_command = self._pending_observation = None
        self.playback_controller.restore_state(state["playback"])  # type: ignore[arg-type]

    def post_step_callback(self, commands: list[str] | None = None):
        if self.gait_phase_enabled:
            super().post_step_callback(commands)

    def _manual_key_speed(self, key: str, axis: int) -> float:
        if self._manual_key_speeds is None:
            return float(self.keyboard_command_magnitudes[axis])
        return self._manual_key_speeds[key]

    def _command_from_held_keys(self, keys, *, remote: bool = False) -> np.ndarray:
        self._keyboard_command.fill(0.0)
        for axis, (positive, negative) in enumerate(_KEY_PAIRS):
            forward = positive in keys
            if forward == (negative in keys):
                continue
            key = positive if forward else negative
            speed = self.keyboard_command_magnitudes[axis] if remote else self._manual_key_speed(key, axis)
            self._keyboard_command[axis] = speed if forward else -speed
        return self._keyboard_command.copy()

    def _cancel_preset(self) -> None:
        self._preset_active = False
        self._preset_wait_release = bool(self._held_motion_keys or self._preset_key_down)
        self._keyboard_command.fill(0.0)

    def _commands_with_preset(self, events, *, inhibit: bool = False) -> np.ndarray:
        cancel = any(
            event.get("type") == "keyboard" and event.get("name") == "x" and event.get("pressed", False)
            for event in events
        )
        inactive = self._control_active is False or (
            self.keyboard_command_mode == "hardware_hold" and self._control_active is not True
        )
        blocked = inhibit or cancel or self._preset_wait_release or inactive
        for event in events:
            if event.get("type") != "keyboard":
                continue
            key, pressed = event.get("name"), bool(event.get("pressed", False))
            if key == "p":
                if self._preset_command is not None and pressed and not self._preset_key_down and not blocked:
                    self._preset_active = True
                self._preset_key_down = pressed
            elif key in KEY_AXES:
                if pressed:
                    self._held_motion_keys.add(key)
                    if not blocked and not self._preset_active and self.keyboard_command_mode == "latch":
                        axis, sign = KEY_AXES[key]
                        self._keyboard_command[axis] = sign * self._manual_key_speed(key, axis)
                else:
                    self._held_motion_keys.discard(key)

        if blocked:
            self._cancel_preset()
        elif self._preset_active:
            assert self._preset_command is not None
            return self._preset_command.copy()
        elif self.keyboard_command_mode in ("hold", "hardware_hold"):
            return self._command_from_held_keys(self._held_motion_keys)
        return self._keyboard_command.copy()

    def _get_commands(self, ctrl_data) -> np.ndarray:
        """Build remote, local held, or legacy latched velocity commands.

        Legacy keyboard input is delivered as press/release events. A key
        press sets its axis to the corresponding trained command magnitude;
        release events do not stop motion because an SSH terminal cannot track
        key holds.  The x key clears all three command axes.
        """
        remote = ctrl_data.get("RemoteKeyboardCtrl")
        if getattr(self, "keyboard_command_mode", None) == "hardware_hold":
            keyboard_data = ctrl_data.get("KeyboardCtrl")
            if keyboard_data is None or remote is not None:
                self._cancel_preset()
                return self._keyboard_command.copy()
            return self._commands_with_preset(keyboard_data.get("keyboard_event", []))
        if remote is not None:
            return self._command_from_held_keys(
                remote["keys"] if remote["fresh"] and remote["armed"] else (), remote=True,
            )
        keyboard_data = ctrl_data.get("KeyboardCtrl")
        if keyboard_data is None:
            return super()._get_commands(ctrl_data)
        if self._preset_command is not None:
            return self._commands_with_preset(keyboard_data.get("keyboard_event", []))
        if getattr(self, "keyboard_command_mode", "latch") == "hold":
            for event in keyboard_data.get("keyboard_event", []):
                if event.get("type") != "keyboard":
                    continue
                key = event.get("name")
                if key in KEY_AXES:
                    if event.get("pressed", False):
                        self._held_motion_keys.add(key)
                    else:
                        self._held_motion_keys.discard(key)
                elif key == "x" and event.get("pressed", False):
                    self._keyboard_hold_blocked = True
            if self._keyboard_hold_blocked:
                self._keyboard_command.fill(0.0)
                if not self._held_motion_keys:
                    self._keyboard_hold_blocked = False
            else:
                return self._command_from_held_keys(self._held_motion_keys)
            return self._keyboard_command.copy()

        key_axes = KEY_AXES
        for event in keyboard_data.get("keyboard_event", []):
            if event.get("type") != "keyboard":
                continue
            name = event.get("name")
            if name not in key_axes and name != "x":
                continue
            if not event.get("pressed", False):
                self._held_motion_keys.discard(name)
                continue

            self._held_motion_keys.add(name)
            next_command = self._keyboard_command.copy()
            if name == "x":
                next_command.fill(0.0)
            else:
                axis, sign = key_axes[name]
                next_command[axis] = sign * self._manual_key_speed(name, axis)

            if np.array_equal(next_command, self._keyboard_command):
                continue
            self._keyboard_command = next_command
            logger.info(
                "[KEYBOARD] key=%s vx=%.3f vy=%.3f yaw=%.3f",
                name,
                self._keyboard_command[0],
                self._keyboard_command[1],
                self._keyboard_command[2],
            )
        return self._keyboard_command.copy()

    def _commands_with_adaptation_restart(self, ctrl_data) -> np.ndarray:
        coordinator = self.online_coordinator
        if coordinator is None:
            return self._get_commands(ctrl_data)
        keyboard = ctrl_data.get("KeyboardCtrl")
        if coordinator.restart_pending or self._restart_motion_blocked:
            events = keyboard.get("keyboard_event", []) if keyboard is not None else ()
            commands = self._commands_with_preset(events, inhibit=True)
            self._restart_motion_blocked = bool(self._held_motion_keys or self._preset_key_down)
            if self._control_active:
                coordinator.advance_restart_at_step_boundary()
            return commands
        commands = self._get_commands(ctrl_data)
        if "[LORA_RESTART]" in ctrl_data.get("COMMANDS", []):
            if (keyboard is None or ctrl_data.get("RemoteKeyboardCtrl") is not None
                    or not self._control_active or self._last_submitted_moving is not False
                    or np.any(commands != 0.0)):
                logger.warning("[LORA RESTART] rejected: requires local active control and zero current and previous velocity commands")
            elif coordinator.request_adaptation_restart():
                self._adaptation_started = False
                self._cancel_preset()
                coordinator.advance_restart_at_step_boundary()
        return commands

    def get_observation(self, env_data, ctrl_data):
        if self._stopping:
            raise RuntimeError("FADA control has stopped")
        if self._pending_snapshot is not None:
            raise RuntimeError("previous FADA action has not been confirmed")
        if self.online_coordinator is not None and self._control_active is None:
            raise RuntimeError("online FADA requires a lifecycle-aware single-policy pipeline")
        commands_control = ctrl_data.get("COMMANDS", [])
        if "[LORA_FREEZE]" in commands_control:
            self.set_lora_frozen(True)
        elif "[LORA_UNFREEZE]" in commands_control:
            self.set_lora_frozen(False)
        if self.online_coordinator is not None and self._control_active:
            self._pending_sensor = {
                name: env_data.get(name) for name in (
                    "sensor_tick", "torso_imu_sequence", "lowstate_age_s",
                    "torso_imu_age_s", "sensor_read_started_ns",
                    "simulation_frame_id", "simulation_state_ready_ns",
                )
            }
            self._pending_sensor["source"] = env_data.get("source", "hardware")
        commands = self._commands_with_adaptation_restart(ctrl_data)
        if self.online_coordinator is not None:
            self._issued_moving = bool(np.any(commands != 0.0))
            enabled = self._issued_moving and not self._lora_frozen
            if enabled != self.online_coordinator.updates_enabled:
                self.online_coordinator.set_updates_enabled(enabled)
                self._adaptation_started = False
                logger.info("[LORA] updates_enabled=%s standing=%s manual_frozen=%s",
                            enabled, not bool(np.any(commands != 0.0)), self._lora_frozen)
        policy_gyro = getattr(env_data, "policy_gyro", None)
        policy_gravity = getattr(env_data, "policy_gravity", None)
        if policy_gyro is None or policy_gravity is None:
            torso_imu_state = getattr(env_data, "torso_imu_state", None)
            if torso_imu_state is None:
                raise RuntimeError(
                    "FADA Planner-IDM requires MuJoCo policy IMU observations or "
                    "Unitree torso_imu_state"
                )
            policy_gyro, policy_gravity = self._torso_imu_projector.project(
                torso_imu_state
            )
        base_ang_vel = np.asarray(policy_gyro, dtype=np.float32)
        gravity = np.asarray(policy_gravity, dtype=np.float32)
        if (
            base_ang_vel.shape != (3,)
            or gravity.shape != (3,)
            or not bool(np.isfinite(base_ang_vel).all() and np.isfinite(gravity).all())
        ):
            raise RuntimeError("FADA Planner-IDM policy IMU observation must be finite 3-D vectors")
        gravity_norm = float(np.linalg.norm(gravity))
        if not 0.999 <= gravity_norm <= 1.001:
            raise RuntimeError(
                f"FADA Planner-IDM policy gravity norm is invalid: {gravity_norm:.6f}"
            )
        obs = np.concatenate(
            [
                base_ang_vel * 0.25,
                gravity,
                np.asarray(env_data.dof_pos - self.default_dof_pos, dtype=np.float32),
                np.asarray(env_data.dof_vel, dtype=np.float32) * 0.05,
                np.asarray(self.last_action, dtype=np.float32),
                commands,
                self.gait_phase.astype(np.float32),
            ]
        ).astype(np.float32)
        if obs.shape != (FADA_G1_ACTOR_OBS_DIM,) or not bool(np.isfinite(obs).all()):
            raise ValueError(f"FADA raw observation must be finite 98-D, got {obs.shape}")
        self._last_obs = obs.copy()
        self._pending_observation = obs.copy()
        self._pending_command = commands.copy()
        return obs, {
            "commands": commands,
            "lora_frozen": self._lora_frozen,
            "lora_updates_enabled": self.online_coordinator is not None and self.online_coordinator.updates_enabled,
            "gait_phase": self.gait_phase.copy(),
            "fada_raw_obs_dim": FADA_G1_ACTOR_OBS_DIM,
            "fada_projected_obs_dim": FADA_G1_STATE_DIM,
        }

    def get_action(self, obs: np.ndarray) -> np.ndarray:
        candidate = np.asarray(obs, dtype=np.float32)
        if self._pending_observation is None or self._pending_command is None:
            raise RuntimeError("FADA get_action requires one fresh get_observation call")
        if candidate.shape != self._pending_observation.shape or not np.array_equal(
            candidate, self._pending_observation
        ):
            raise ValueError("FADA action observation does not match the pending control step")
        if self.online_coordinator is None and self._control_active is not False:
            action_tensor = self.playback_controller.act(candidate, self._pending_command)
        else:
            histories = self.playback_controller.prepare_query(candidate, self._pending_command)
            if self.online_coordinator is not None and self._control_active:
                coordinator = self.online_coordinator
                legacy = coordinator.runtime.method == "fada"
                if not self._adaptation_started and coordinator.updates_enabled and np.any(self._pending_command != 0.0):
                    self._adaptation_started = True
                    if legacy:
                        coordinator.enable_updates(self._control_step)
                    logger.info(
                        "FADA online updates triggered at control_step=%d command=%s",
                        self._control_step, self._pending_command.tolist(),
                    )
                evidence = FADAControlEvidence(
                    episode_id=coordinator.episode_id, control_step=self._control_step,
                    policy_version=coordinator.policy_version,
                    policy_control_active=True, action_dispatched=False,
                    **self._pending_sensor,
                )
                collect = coordinator.updates_enabled
                if collect:
                    coordinator.observe(histories[0][:, -1], histories[2], evidence)
                    coordinator.apply_pending_at_step_boundary()
                    if self._adaptation_started:
                        coordinator.try_submit()
                else:
                    coordinator.poll()
                snapshot = coordinator.runtime.infer_with_snapshot(*histories)
                self._pending_snapshot = snapshot
                self._pending_evidence = replace(evidence, policy_version=coordinator.policy_version)
                self._pending_collection = collect
                action_tensor = snapshot.action_snapshot
            else:
                action_tensor = self._runtime["runner"].infer(*histories)
        action = action_tensor.squeeze(0).detach().cpu().numpy().astype(np.float32)
        if self.online_coordinator is None and self._control_active is not False:
            self.last_action = action.copy()
        self._pending_observation = None
        self._pending_command = None
        return action * self.execution_action_scale
