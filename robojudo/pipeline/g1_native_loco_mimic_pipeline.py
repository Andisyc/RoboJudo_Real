from __future__ import annotations

import logging
import time
from enum import Enum, auto

import numpy as np

import robojudo.environment
from robojudo.controller import CtrlManager
from robojudo.environment import Environment
from robojudo.pipeline import Pipeline, pipeline_registry
from robojudo.pipeline.pipeline_cfgs import G1NativeLocoMimicPipelineCfg
from robojudo.pipeline.rl_multi_policy_pipeline import PolicyManager
from robojudo.utils.util_func import get_gravity_orientation

from robojudo.tools.mimic_safety import (
    HOLD_TEXT,
    NOT_WALKRUN_TEXT,
    REGULAR_FAILED_TEXT,
    REGULAR_SKIPPED_TEXT,
    STANCE_TIMEOUT_TEXT,
    STOPPED_TEXT,
    STOPPING_TEXT,
    WALKRUN_FAILED_TEXT,
    WALKRUN_SWITCH_TEXT,
    WALKRUN_UNSTEADY_TEXT,
    WARNING_TEXT,
    MimicSafetyGuard,
    SafetySpeaker,
    SafetyViolation,
)

logger = logging.getLogger(__name__)


class NativeLocoMimicState(Enum):
    NATIVE_LOCO = auto()
    MIMIC_ACTIVE = auto()
    # The policy cannot run (errors, process closing): the native writer
    # streams the last target until WALKRUN is confirmed or the operator takes
    # over. LowCmd is never stopped while FSM 1000 may be active.
    HANDBACK = auto()
    FAULT = auto()
    CLOSED = auto()


@pipeline_registry.register
class G1NativeLocoMimicPipeline(Pipeline):
    """Run Unitree native loco until explicit handoff to a mimic policy."""

    cfg: G1NativeLocoMimicPipelineCfg

    # Entry/handback steps block by design; a frame-drop shutdown would stop
    # LowCmd mid-dance, so scripts/run_pipeline.py must not exit on frame drop.
    exit_on_frame_drop = False
    HANDBACK_RETRY_S = 1.0
    # After this many failed WALKRUN requests stop retrying and hold, so the
    # robot can be hung on the gantry; A/Select asks again.
    HANDBACK_MAX_FAILURES = 3
    HOLD_POLL_S = 1.0
    HOLD_ANNOUNCE_S = 10.0
    MAX_CONSECUTIVE_STEP_ERRORS = 3
    # A/Select during the dance: keep dancing until the reference is back near
    # a stance WALKRUN accepted (motion start = taken from WALKRUN, motion end =
    # handed to WALKRUN), freeze the reference there and hand back.
    STOP_WAIT_S = 20.0
    STANCE_LEG_TOL = 0.35
    STANCE_HEIGHT_TOL = 0.03
    STANCE_REF_VEL = 0.6
    STANCE_MEASURED_TOL = 0.45
    STANCE_MEASURED_VEL = 1.0
    STANCE_JOINT_KEYS = ("hip", "knee", "ankle", "waist")
    # Request regular loco only once the legs have been still for REGULAR_STILL_S:
    # a request sent while WALKRUN is still settling its steps is accepted but
    # ignored by the firmware (field log 2026-11-03, FSM stayed 801).
    REGULAR_STILL_S = 1.0
    REGULAR_STILL_VEL = 0.2
    REGULAR_WAIT_S = 8.0
    # Start in regular loco: switch to WALKRUN, then enter mimic once the legs
    # have been still for REGULAR_STILL_S (same rule), within WALKRUN_ENTRY_WAIT_S.
    WALKRUN_ENTRY_WAIT_S = 8.0
    WALKRUN_FSM_IDS = (801, 802)  # 802: walk-run on some (29DoF) firmware
    REGULAR_FSM_IDS = (500, 501)

    def __init__(self, cfg: G1NativeLocoMimicPipelineCfg):
        super().__init__(cfg=cfg)

        env_class: type[Environment] = getattr(
            robojudo.environment, self.cfg.env.env_type
        )
        self.env = env_class(cfg_env=self.cfg.env, device=self.device)
        required_env_methods = (
            "acquire_user_control",
            "release_to_walkrun",
        )
        if not all(hasattr(self.env, name) for name in required_env_methods):
            raise TypeError("G1NativeLocoMimicPipeline requires G1LocoEnv")

        self.ctrl_manager = CtrlManager(
            cfg_ctrls=self.cfg.ctrl, env=self.env, device=self.device
        )
        self.policy_manager = PolicyManager(
            cfg_policies=self.cfg.mimic_policies,
            env=self.env,
            device=self.device,
        )
        self.env.update_dof_cfg(override_cfg=self.policy.cfg_action_dof)

        # Targets returned by PolicyWrapper are already in environment order.
        self._hip_bias_indices = [
            list(self.env.joint_names).index(name)
            for name in ("left_hip_pitch_joint", "right_hip_pitch_joint")
        ]
        self.freq = self.policy.cfg_policy.freq
        self.dt = 1.0 / self.freq
        self.state = NativeLocoMimicState.NATIVE_LOCO
        self.should_stop = False
        self.mimic_guard = MimicSafetyGuard(self.env.joint_names) if self.env.num_dofs in (23, 29) else None
        self.safety_speaker = SafetySpeaker(self.env.speak_safety_warning) if self.mimic_guard else None
        self._entry_blocked = False
        self._step_errors = 0
        self._handback_next_try = 0.0
        self._regular_deadline = None
        self._regular_still_since = None
        self._walkrun_entry_deadline = None
        self._walkrun_still_since = None
        self._stance_cache = {}
        self._clear_stop_flags()

        self.self_check()
        self.reset()

    @property
    def policy(self):
        return self.policy_manager.policy

    def self_check(self):
        self.env.self_check()
        self.env.update()

    def reset(self):
        self.timestep = 0
        self.ctrl_manager.reset()
        for policy in self.policy_manager.policies:
            policy.reset()
        self.env.reset()

    def prepare(self):
        """Warm policies without requesting user control or sending LowCmd."""
        self.env.update()
        env_data = self.env.get_data()
        ctrl_data = self.ctrl_manager.get_ctrl_data(env_data)
        ctrl_data["COMMANDS"] = []
        for policy in self.policy_manager.policies:
            policy.reset()
            observation, _ = policy.get_observation(env_data, ctrl_data)
            policy.get_pd_target(observation)
            policy.reset()
        logger.warning(
            "Ready in Unitree native loco; press Start to enter mimic: %s",
            self.env.get_control_status(),
        )

    def _select_mimic(self, delta: int):
        if self.state != NativeLocoMimicState.NATIVE_LOCO:
            logger.warning("Mimic selection is only available in native loco")
            return
        policy_id = (
            self.policy_manager.current_policy_id + delta
        ) % self.policy_manager.num_policies
        self.policy_manager.set_policy(policy_id)
        self.freq = self.policy.cfg_policy.freq
        self.dt = 1.0 / self.freq

    def _start_pressed(self, env_data, ctrl_data):
        if self.state != NativeLocoMimicState.NATIVE_LOCO or self._entry_blocked:
            return self._enter_mimic(env_data, ctrl_data)
        if self._walkrun_entry_deadline is not None:
            return None  # already switching to WALKRUN; entry follows once still
        if getattr(self.cfg, "walkrun_on_start", True) and hasattr(self.env, "switch_regular_to_walkrun"):
            try:
                fsm_id = self.env.sync_operator_fsm()  # query only
            except Exception as exc:
                logger.error("FSM query on Start failed: %r", exc)
                fsm_id = None
            if fsm_id in self.REGULAR_FSM_IDS:
                self._switch_to_walkrun(fsm_id)
                return None
        return self._enter_mimic(env_data, ctrl_data)

    def _switch_to_walkrun(self, fsm_id):
        """Start in regular loco: request WALKRUN; mimic entry follows once still."""
        self._regular_deadline = None
        logger.warning("WALKRUN_AUTO: Start in regular loco (FSM %s); requesting WALKRUN", fsm_id)
        if self.safety_speaker:
            self.safety_speaker.announce(WALKRUN_SWITCH_TEXT)
        try:
            result = self.env.switch_regular_to_walkrun()
        except Exception as exc:
            result = repr(exc)
        status = self.env.get_control_status()
        fsm_trace = getattr(self.env, "last_walkrun_fsm_trace", None)
        self.trace_event("walkrun_auto", result=result, fsm_id=status["fsm_id"], fsm_trace=fsm_trace)
        if result != 0:
            # No retry from here: damp or any other mode stays the operator's.
            self._entry_blocked = True
            logger.error("WALKRUN_AUTO_FAILED: rc=%s; no further mode request: %s fsm_trace=%s",
                         result, status, fsm_trace)
            if self.safety_speaker:
                self.safety_speaker.announce(
                    NOT_WALKRUN_TEXT if status["fsm_id"] in self.REGULAR_FSM_IDS else WALKRUN_FAILED_TEXT)
            return
        logger.warning("WALKRUN_AUTO: regular loco -> WALKRUN confirmed: %s fsm_trace=%s", status, fsm_trace)
        self._walkrun_entry_deadline = time.monotonic() + self.WALKRUN_ENTRY_WAIT_S
        self._walkrun_still_since = None

    def _advance_walkrun_entry(self, env_data, ctrl_data):
        """After an automatic WALKRUN switch: enter mimic once the legs are still."""
        if self._walkrun_entry_deadline is None:
            return None
        if time.monotonic() > self._walkrun_entry_deadline:
            self._walkrun_entry_deadline = None
            logger.warning("WALKRUN_AUTO_ENTRY_SKIPPED: not stationary within %.1fs; staying in WALKRUN",
                           self.WALKRUN_ENTRY_WAIT_S)
            self.trace_event("walkrun_auto", result="not_stationary")
            if self.safety_speaker:
                self.safety_speaker.announce(WALKRUN_UNSTEADY_TEXT)
            return None
        self._walkrun_still_since = self._held_still(env_data, self._walkrun_still_since)
        if self._walkrun_still_since is None or time.monotonic() - self._walkrun_still_since < self.REGULAR_STILL_S:
            return None
        self._walkrun_entry_deadline = None
        logger.warning("WALKRUN_AUTO: still in WALKRUN; entering mimic")
        return self._enter_mimic(env_data, ctrl_data)

    def _held_still(self, env_data, since):
        """Start of the current still interval (None while moving)."""
        if self.mimic_guard:
            try:
                self.mimic_guard.state(env_data, self.env.safety_state_age())
            except SafetyViolation:
                return None
        legs = [i for i, n in enumerate(self.env.joint_names) if any(k in n for k in self.STANCE_JOINT_KEYS)]
        if np.abs(np.asarray(env_data["dof_vel"])[legs]).max() > self.REGULAR_STILL_VEL:
            return None
        return time.monotonic() if since is None else since

    def _enter_mimic(self, env_data, ctrl_data):
        if self.state != NativeLocoMimicState.NATIVE_LOCO:
            logger.warning("Mimic control is already active or transitioning")
            return None

        if self._entry_blocked:
            return None
        self._regular_deadline = None
        try:
            if self.mimic_guard:
                if not self.env.born_place_align:
                    raise SafetyViolation("born_place_align must be enabled")
                # Observe a short stationary interval under native control.
                for _ in range(11):
                    self.env.update()
                    fresh = self.env.get_data()
                    self.mimic_guard.state(fresh, self.env.safety_state_age())
                    time.sleep(.02)
            before_reset_quat = np.asarray(self.env.get_data()["base_quat"]).tolist()
            self.env.reset()
            env_data = self.env.get_data()  # Never reuse the caller's pre-reset observation.
            self.policy.reset()
            if self.mimic_guard:
                self.mimic_guard.reset()
                self.mimic_guard.state(env_data, self.env.safety_state_age())
                yaw = self.mimic_guard.heading(env_data)
                logger.warning("MIMIC_REANCHOR: before=%s after=%s yaw_deg=%.3f",
                               before_reset_quat, env_data["base_quat"], np.rad2deg(yaw))
                self.trace_event("entry_reanchor", before_quat_xyzw=before_reset_quat,
                                 after_quat_xyzw=np.asarray(env_data["base_quat"]).tolist(),
                                 yaw_rad=yaw, state_age_s=self.env.safety_state_age())
            self.env.update_dof_cfg(override_cfg=self.policy.cfg_action_dof)
            observation, extras = self.policy.get_observation(env_data, ctrl_data)
            if self.mimic_guard:
                self.mimic_guard.observation(observation)
            pd_target = self.policy.get_pd_target(observation)
            if self.mimic_guard:
                pd_target = self.mimic_guard.target(pd_target, env_data["dof_pos"])
            result = self.send_traced(pd_target, phase="acquiring", sender=self.env.acquire_user_control)
        except Exception as exc:
            self._entry_failed(exc)
            return None
        if result != 0:
            logger.error("Failed to acquire G1 user control: %s", result)
            status = self.env.get_control_status()
            text = WARNING_TEXT if status["fsm_id"] in self.WALKRUN_FSM_IDS else NOT_WALKRUN_TEXT
            self._entry_failed(RuntimeError(f"acquire failed: {result}"), text)
            return None

        status = self.env.get_control_status()
        if (
            status["authority_state"] != "USER_ACTIVE"
            or status["fsm_id"] != 1000
            or not status["publish_enabled"]
            or status["active_publish_count"] <= 0
        ):
            self._entry_failed(RuntimeError(f"user control not armed: {status}"))
            return None

        self.state = NativeLocoMimicState.MIMIC_ACTIVE
        self._step_errors = 0
        self._clear_stop_flags()
        self.trace_event("native_state", state=self.state.name)
        self.trace_event(
            "fixed_hip_bias", **self.cfg.fixed_hip_bias.model_dump(),
            policy_matches=self.policy.cfg_policy.policy_name == self.cfg.fixed_hip_bias.policy_name,
            joint_names=["left_hip_pitch_joint", "right_hip_pitch_joint"],
        )
        logger.warning("G1 user control acquired; mimic policy active: %s", status)
        return extras, pd_target

    def _entry_failed(self, exc, text=WARNING_TEXT):
        status = self.env.get_control_status()
        if status["authority_state"] == "USER_ACTIVE" or status["publish_enabled"]:
            # The USER request may be in effect and the bridge keeps streaming
            # the first target: hand back to WALKRUN instead of stopping LowCmd.
            self.state = NativeLocoMimicState.MIMIC_ACTIVE
            self._request_handback(f"entry not confirmed: {exc}")
            return
        if status["authority_state"] == "INTERNAL":
            # Unitree loco still owns the robot: reject only this Start.
            self._entry_blocked = True
            logger.error("MIMIC_ENTRY_REJECTED: %s status=%s", exc, status)
            self.trace_event("safety_rejected", reason=str(exc), phase="entry")
            if self.safety_speaker:
                self.safety_speaker.announce(text)
            return
        self._fault(exc)

    def _clear_stop_flags(self):
        self._stop_reason = None
        self._stop_deadline = None
        self._handback_reason = None
        self._handback_failures = 0
        self._hold = False
        self._hold_text = HOLD_TEXT
        self._hold_next_poll = 0.0
        self._hold_next_announce = 0.0

    def _return_to_native_loco(self, reason="requested"):
        if self.state == NativeLocoMimicState.NATIVE_LOCO:
            return
        if self.state not in (NativeLocoMimicState.MIMIC_ACTIVE, NativeLocoMimicState.HANDBACK):
            logger.error("Cannot return to native loco from state %s", self.state.name)
            return
        if self._hold:
            return  # no automatic requests while holding; A/Select asks again
        now = time.monotonic()
        if now < self._handback_next_try:
            return
        self._handback_next_try = now + self.HANDBACK_RETRY_S

        result = self.env.release_to_walkrun()
        if result == 0:
            self.state = NativeLocoMimicState.NATIVE_LOCO
            self._step_errors = 0
            self._clear_stop_flags()
            self.trace_event("native_state", state=self.state.name, reason=str(reason))
            self.policy.reset()
            self.env.reset()
            if getattr(self.cfg, "return_to_regular_loco", True):
                self._regular_deadline = time.monotonic() + self.REGULAR_WAIT_S
                self._regular_still_since = None
            logger.warning(
                "Control returned to Unitree WALKRUN (%s): %s",
                reason,
                self.env.get_control_status(),
            )
            return

        status = self.env.get_control_status()
        if self.env.has_user_control:
            # Bridge restored the hold stream; keep the current state and retry.
            self._handback_failures += 1
            logger.error("WALKRUN handoff failed %d/%d (%s): rc=%s; holding: %s",
                         self._handback_failures, self.HANDBACK_MAX_FAILURES, reason, result, status)
            if self._handback_failures >= self.HANDBACK_MAX_FAILURES:
                self._enter_hold(f"WALKRUN handoff failed {self._handback_failures} times: {reason}")
            return
        if status["authority_state"] == "INTERNAL":
            # Firmware already left FSM 1000 for another mode (e.g. operator
            # action); the bridge requested nothing and stopped streaming.
            self.state = NativeLocoMimicState.NATIVE_LOCO
            self._clear_stop_flags()
            self.trace_event("native_state", state=self.state.name, reason=str(reason))
            self.policy.reset()
            logger.error("Robot is in Unitree FSM %s, not WALKRUN; no further mode request: rc=%s",
                         status["fsm_id"], result)
            return
        self._fault(RuntimeError(f"G1 did not reach WALKRUN after release: {result} status={status}"))

    def _request_handback(self, reason, text=None):
        if self.state not in (NativeLocoMimicState.MIMIC_ACTIVE, NativeLocoMimicState.HANDBACK):
            return
        if self.state != NativeLocoMimicState.HANDBACK:
            logger.error("MIMIC_HANDBACK: %s; holding last target until WALKRUN is confirmed", reason)
            self.trace_event("handback", reason=str(reason), automatic_mode_change="WALKRUN")
            self.state = NativeLocoMimicState.HANDBACK
            self._handback_next_try = 0.0
            if text and self.safety_speaker:
                self.safety_speaker.announce(text)
        self._return_to_native_loco(reason)

    def _enter_hold(self, reason, text=HOLD_TEXT):
        if self._hold:
            return
        self._hold = True
        self._hold_text = text
        self._hold_next_poll = time.monotonic() + self.HOLD_POLL_S
        self._hold_next_announce = time.monotonic() + self.HOLD_ANNOUNCE_S
        logger.critical("MIMIC_HOLD: %s; holding, no further automatic mode request; "
                        "A/Select asks for WALKRUN again, remote damp is detected: %s",
                        reason, self.env.get_control_status())
        self.trace_event("hold", reason=str(reason), state=self.state.name)
        if self.safety_speaker:
            self.safety_speaker.announce(text)

    def _hold_tick(self):
        """While holding: detect an operator mode change (query only), repeat the prompt."""
        now = time.monotonic()
        if now >= self._hold_next_poll:
            self._hold_next_poll = now + self.HOLD_POLL_S
            try:
                fsm_id = self.env.sync_operator_fsm()
            except Exception as exc:
                logger.error("FSM query while holding failed: %r", exc)
                fsm_id = None
            if not self.env.has_user_control:
                logger.warning("MIMIC_HOLD_RELEASED: operator moved the robot to FSM %s", fsm_id)
                self.trace_event("native_state", state="NATIVE_LOCO", reason=f"operator FSM {fsm_id}")
                self.state = NativeLocoMimicState.NATIVE_LOCO
                self._clear_stop_flags()
                self.policy.reset()
                return
        if now >= self._hold_next_announce:
            self._hold_next_announce = now + self.HOLD_ANNOUNCE_S
            if self.safety_speaker:
                self.safety_speaker.announce(self._hold_text)

    def _operator_stop(self, command):
        """A/Select during the dance or a handback."""
        if self._hold:
            logger.warning("MIMIC_HOLD_RETRY: operator %s asks for WALKRUN again", command)
            self._hold = False
            self._handback_failures = self.HANDBACK_MAX_FAILURES - 1  # one attempt, then hold again
            self._handback_next_try = 0.0
            if self.state == NativeLocoMimicState.MIMIC_ACTIVE and self._handback_reason is None:
                self._handback_reason = f"operator {command} after hold"
            self._return_to_native_loco(self._handback_reason or f"operator {command} after hold")
            return
        if self.state != NativeLocoMimicState.MIMIC_ACTIVE or self._handback_reason is not None:
            logger.warning("Handback already in progress; %s ignored", command)
            return
        if self._stop_reason is not None:
            # Repeated presses never skip the stance wait (accidental presses).
            logger.warning("Stop already requested; waiting for a WALKRUN stance, %s ignored", command)
            return
        if self._stance_targets() is None:
            if self.safety_speaker:
                self.safety_speaker.announce(STOPPED_TEXT)
            self._begin_handback(f"operator {command} (no stance reference)")
            return
        self._stop_reason = f"operator {command} during dance"
        self._stop_deadline = time.monotonic() + self.STOP_WAIT_S
        logger.warning("MIMIC_STOP_REQUESTED: %s; dancing on until a WALKRUN stance (max %.0fs)",
                       self._stop_reason, self.STOP_WAIT_S)
        self.trace_event("stop_requested", reason=self._stop_reason, timestep=self.timestep)
        if self.safety_speaker:
            self.safety_speaker.announce(STOPPING_TEXT)

    def _begin_handback(self, reason):
        """Policy still healthy: freeze the reference so it actively holds this
        pose, and hand back while it keeps balancing."""
        self._stop_reason = None
        self._stop_deadline = None
        self._handback_reason = reason
        self._freeze_reference()
        logger.warning("MIMIC_HANDBACK: %s; reference frozen, policy holds until WALKRUN is confirmed", reason)
        self.trace_event("handback", reason=str(reason), automatic_mode_change="WALKRUN", policy_holding=True)
        self._handback_next_try = 0.0
        self._return_to_native_loco(reason)

    def _freeze_reference(self):
        # self.policy is a PolicyWrapper that only delegates attribute reads;
        # play_speed must be set on the wrapped policy that advances timestep.
        policy = getattr(self.policy, "__dict__", {}).get("policy", self.policy)
        if hasattr(policy, "play_speed"):
            policy.play_speed = 0.0

    def _stance_targets(self):
        """Leg joints (model order and env order) and (joint_pos, pelvis z) of the
        motion's start/end stances, or None."""
        key = id(self.policy)
        if key in self._stance_cache:
            return self._stance_cache[key]
        targets = None
        try:
            frame = getattr(self.policy, "reference_frame", None)
            names = list(getattr(self.policy.cfg_action_dof, "joint_names", []))
            legs = [i for i, n in enumerate(names) if any(k in n for k in self.STANCE_JOINT_KEYS)]
            if callable(frame) and legs:
                # The reference is in model joint order; env_data is in env
                # (motor) order. Match measured joints by name.
                env_names = list(self.env.joint_names)
                env_legs = [env_names.index(names[i]) for i in legs]
                cfg = self.policy.cfg_policy
                steps = [cfg.start_timestep] + ([cfg.max_timestep] if cfg.max_timestep > 0 else [])
                stances = [(np.asarray(f[0])[legs], f[2]) for f in (frame(t) for t in steps) if f is not None]
                targets = (legs, env_legs, stances) if stances else None
        except Exception as exc:
            logger.error("Stance reference unavailable: %r", exc)
        self._stance_cache[key] = targets
        return targets

    def _at_stance(self, env_data):
        targets = self._stance_targets()
        current = getattr(self.policy, "current_reference", lambda: None)()
        if targets is None or current is None:
            return False
        legs, env_legs, stances = targets
        ref_q, ref_dq, ref_z = np.asarray(current[0])[legs], np.asarray(current[1])[legs], current[2]
        q = np.asarray(env_data["dof_pos"])[env_legs]
        dq = np.asarray(env_data["dof_vel"])[env_legs]
        if np.abs(ref_dq).max() > self.STANCE_REF_VEL or np.abs(dq).max() > self.STANCE_MEASURED_VEL:
            return False
        return any(
            np.abs(ref_q - s_q).max() <= self.STANCE_LEG_TOL
            and abs(ref_z - s_z) <= self.STANCE_HEIGHT_TOL
            and np.abs(q - s_q).max() <= self.STANCE_MEASURED_TOL
            for s_q, s_z in stances
        )

    def _advance_stop(self, env_data, extras):
        """Per dance frame, after the policy target was sent."""
        if self._hold:
            self._hold_tick()
        elif self._handback_reason is not None:
            self._return_to_native_loco(self._handback_reason)
        elif "[MOTION_DONE]" in extras.get("CALLBACK", []):
            self._begin_handback("motion done")
        elif self._stop_reason is not None:
            if self._at_stance(env_data):
                self._begin_handback(f"{self._stop_reason}: stance reached")
            elif time.monotonic() > self._stop_deadline:
                # Never force WALKRUN from an unknown pose: freeze and hold.
                reason = f"{self._stop_reason}: no stance within {self.STOP_WAIT_S:.0f}s"
                self._stop_reason = None
                self._freeze_reference()
                self._enter_hold(reason, text=STANCE_TIMEOUT_TEXT)

    def _advance_regular_switch(self, env_data):
        """After a confirmed WALKRUN return: wait until the legs are still, then request regular loco."""
        if self._regular_deadline is None:
            return
        if time.monotonic() > self._regular_deadline:
            self._regular_deadline = None
            logger.warning("REGULAR_LOCO_SKIPPED: not stationary within %.1fs; staying in WALKRUN",
                           self.REGULAR_WAIT_S)
            self.trace_event("regular_loco", result="not_stationary")
            if self.safety_speaker:
                self.safety_speaker.announce(REGULAR_SKIPPED_TEXT)
            return
        self._regular_still_since = self._held_still(env_data, self._regular_still_since)
        if self._regular_still_since is None or time.monotonic() - self._regular_still_since < self.REGULAR_STILL_S:
            return
        self._regular_deadline = None
        result = self.env.switch_walkrun_to_regular()
        status = self.env.get_control_status()
        fsm_trace = getattr(self.env, "last_regular_fsm_trace", None)
        if result == 0:
            logger.warning("REGULAR_LOCO: WALKRUN -> regular loco (FSM 500/501) confirmed: %s fsm_trace=%s",
                           status, fsm_trace)
        else:
            logger.error("REGULAR_LOCO_FAILED: rc=%s; robot stays in Unitree loco: %s fsm_trace=%s",
                         result, status, fsm_trace)
            if self.safety_speaker:
                self.safety_speaker.announce(
                    REGULAR_SKIPPED_TEXT if status["fsm_id"] in self.WALKRUN_FSM_IDS else REGULAR_FAILED_TEXT)
        self.trace_event("regular_loco", result=result, fsm_id=status["fsm_id"], fsm_trace=fsm_trace)

    def _shutdown(self):
        result = self.close()
        if result != 0:
            raise RuntimeError(
                f"G1 local shutdown failed: {result}"
            )

    def close(self) -> int:
        if self.state == NativeLocoMimicState.CLOSED:
            return 0
        if self._hold:
            # Holding for the operator: only watch for a remote mode change.
            self._hold_tick()
            if self.env.has_user_control:
                logger.critical("Close deferred: holding the last target; take over with the remote")
                return -1
        elif (self.state in (NativeLocoMimicState.MIMIC_ACTIVE, NativeLocoMimicState.HANDBACK)
                or self.env.has_user_control):
            # Never stop LowCmd while FSM 1000 may be active: hand back first.
            if self.state not in (NativeLocoMimicState.MIMIC_ACTIVE, NativeLocoMimicState.HANDBACK):
                self.state = NativeLocoMimicState.HANDBACK
            self._request_handback("process closing")
            if self.env.has_user_control:
                logger.critical("Close deferred: WALKRUN not confirmed; bridge keeps holding the last target")
                return -1
        result = self.env.shutdown()
        if self.safety_speaker:
            self.safety_speaker.finish()
        if result == 0:
            self.state = NativeLocoMimicState.CLOSED
            self.should_stop = True
            return 0
        self.state = NativeLocoMimicState.FAULT
        logger.error("Failed to close G1 controller safely: %s", result)
        return result

    def _handle_commands(self, commands, env_data, ctrl_data):
        initial_mimic_step = None
        if self.state in (NativeLocoMimicState.FAULT, NativeLocoMimicState.CLOSED):
            return None
        if "[POLICY_MIMIC]" not in commands:
            self._entry_blocked = False
        if commands:
            self.trace_event("commands", commands=commands, state=self.state.name)
        for command in commands:
            if command in ("[SHUTDOWN]", "[POLICY_LOCO]") and self.state in (
                    NativeLocoMimicState.MIMIC_ACTIVE, NativeLocoMimicState.HANDBACK):
                # Operator stop during the dance: back to Unitree loco at a
                # stance, process keeps running.
                self._operator_stop(command)
                return None
            if command == "[SHUTDOWN]":
                self._shutdown()
                return None
            if command == "[POLICY_LOCO]" and self._walkrun_entry_deadline is not None:
                self._walkrun_entry_deadline = None
                logger.warning("WALKRUN_AUTO: entry cancelled by operator; staying in WALKRUN")
                self.trace_event("walkrun_auto", result="cancelled")
                continue
            if command == "[POLICY_MIMIC]":
                initial_mimic_step = self._start_pressed(env_data, ctrl_data)
            elif command.startswith("[POLICY_SWITCH]"):
                _, _, target = command.partition(",")
                if target == "NEXT":
                    self._select_mimic(1)
                elif target == "LAST":
                    self._select_mimic(-1)
        return initial_mimic_step

    def _fault(self, reason):
        # Only reached when the bridge is not streaming (FAULT/CLOSED); there is
        # no LowCmd left to stop, so exiting cannot drop an active dance.
        self.state = NativeLocoMimicState.FAULT
        self.should_stop = True
        logger.error("MIMIC_RUNTIME_ERROR: %s; status=%s; no automatic mode change",
                     reason, self.env.get_control_status())
        self.trace_event("runtime_error", reason=str(reason), automatic_mode_change=False)
        if self.safety_speaker:
            self.safety_speaker.announce()

    def _step_error(self, exc):
        # The native writer keeps streaming the previous target meanwhile.
        self._step_errors += 1
        logger.error("MIMIC_STEP_ERROR %d/%d: %r", self._step_errors,
                     self.MAX_CONSECUTIVE_STEP_ERRORS, exc)
        self.trace_event("runtime_error", reason=repr(exc), consecutive=self._step_errors,
                         automatic_mode_change=False)
        if self._step_errors >= self.MAX_CONSECUTIVE_STEP_ERRORS:
            self._request_handback(f"{self._step_errors} consecutive step errors: {exc!r}")

    def _apply_fixed_hip_bias(self, target, query_frame):
        """Copy-on-write offset, with no feedback adaptation or accumulated state."""
        cfg = self.cfg.fixed_hip_bias
        if (not cfg.enabled or cfg.hip_pitch_deg == 0
                or self.policy.cfg_policy.policy_name != cfg.policy_name):
            return target

        a, b = cfg.ramp_in_start, cfg.ramp_in_end
        c, d = cfg.ramp_out_start, cfg.ramp_out_end
        if query_frame <= a or query_frame >= d:
            return target

        if query_frame < b:
            x = (query_frame - a) / (b - a)
        elif query_frame > c:
            x = (d - query_frame) / (d - c)
        else:
            x = 1.0
        # Quintic fade: zero first and second derivatives at both ends.
        weight = x**3 * (10 - 15*x + 6*x*x)
        corrected = target.copy()
        corrected[self._hip_bias_indices] += np.deg2rad(cfg.hip_pitch_deg) * weight
        return corrected

    def _step_mimic(self, env_data, ctrl_data):
        # Runtime observations and outputs are assumed valid per deployment scope.
        observation, extras = self.policy.get_observation(env_data, ctrl_data)
        # Match the exact integer time_step sent to ONNX, before the callback advances it.
        query_frame = int(self.policy.timestep)
        pd_target = self.policy.get_pd_target(observation)
        pd_target = self._apply_fixed_hip_bias(pd_target, query_frame)
        self.send_traced(pd_target, extras.get("hand_pose", None), commands=ctrl_data.get("COMMANDS", []))
        return extras, pd_target

    def step(self, dry_run: bool = False):
        if dry_run or self.state in (NativeLocoMimicState.CLOSED, NativeLocoMimicState.FAULT):
            return
        if self.state == NativeLocoMimicState.HANDBACK:
            self._step_handback()
            return
        try:
            self._step_once()
        except Exception as exc:
            if self.state == NativeLocoMimicState.MIMIC_ACTIVE:
                self._step_error(exc)
            else:
                # Unitree loco (or the pending handback) owns the robot; keep running.
                logger.error("Step error in %s: %r", self.state.name, exc)

    def _step_handback(self):
        """Policy not running; the bridge streams the last target."""
        ctrl_data = None
        try:
            self.env.update()
            env_data = self.env.get_data()
            ctrl_data = self.ctrl_manager.get_ctrl_data(env_data)
            for command in ctrl_data.get("COMMANDS", []):
                if command in ("[SHUTDOWN]", "[POLICY_LOCO]"):
                    self._operator_stop(command)
        except Exception as exc:
            logger.error("Step error in HANDBACK: %r", exc)
        if self.state == NativeLocoMimicState.HANDBACK:
            if self._hold:
                self._hold_tick()
            else:
                self._return_to_native_loco("handback retry")
        if ctrl_data is not None:
            self.ctrl_manager.post_step_callback(ctrl_data)

    def _step_once(self):
        self.env.update()
        env_data = self.env.get_data()
        ctrl_data = self.ctrl_manager.get_ctrl_data(env_data)
        commands = ctrl_data.get("COMMANDS", [])
        initial_mimic_step = self._handle_commands(commands, env_data, ctrl_data)
        if self.state == NativeLocoMimicState.NATIVE_LOCO and initial_mimic_step is None:
            initial_mimic_step = self._advance_walkrun_entry(env_data, ctrl_data)

        if self.state != NativeLocoMimicState.MIMIC_ACTIVE:
            if self.state == NativeLocoMimicState.NATIVE_LOCO:
                self._advance_regular_switch(env_data)
            self.ctrl_manager.post_step_callback(ctrl_data)
            return

        if initial_mimic_step is not None:
            extras, pd_target = initial_mimic_step
        else:
            extras, pd_target = self._step_mimic(env_data, ctrl_data)
        self._step_errors = 0

        self.policy.post_step_callback(commands)
        self.ctrl_manager.post_step_callback(ctrl_data)
        self._advance_stop(env_data, extras)

        if self.cfg.debug.log_obs:
            self.debug_logger.log(
                env_data=env_data,
                ctrl_data=ctrl_data,
                extras=extras,
                pd_target=pd_target,
                timestep=self.timestep,
            )
        self.timestep += 1
