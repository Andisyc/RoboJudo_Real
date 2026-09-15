from __future__ import annotations

import logging
from enum import Enum, auto

import numpy as np

import robojudo.environment
from robojudo.controller import CtrlManager
from robojudo.environment import Environment
from robojudo.pipeline import Pipeline, pipeline_registry
from robojudo.pipeline.pipeline_cfgs import G1NativeLocoMimicPipelineCfg
from robojudo.pipeline.rl_multi_policy_pipeline import PolicyManager
from robojudo.utils.util_func import get_gravity_orientation

logger = logging.getLogger(__name__)


class NativeLocoMimicState(Enum):
    NATIVE_LOCO = auto()
    ENTERING_MIMIC = auto()
    MIMIC_ACTIVE = auto()
    FAULT = auto()
    CLOSED = auto()


@pipeline_registry.register
class G1NativeLocoMimicPipeline(Pipeline):
    """Run Unitree native loco until explicit handoff to a mimic policy."""

    cfg: G1NativeLocoMimicPipelineCfg

    def __init__(self, cfg: G1NativeLocoMimicPipelineCfg):
        super().__init__(cfg=cfg)

        env_class: type[Environment] = getattr(
            robojudo.environment, self.cfg.env.env_type
        )
        self.env = env_class(cfg_env=self.cfg.env, device=self.device)
        required_env_methods = (
            "acquire_user_control",
            "release_to_walkrun",
            "release_to_passive",
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

        self.freq = self.policy.cfg_policy.freq
        self.dt = 1.0 / self.freq
        self.state = NativeLocoMimicState.NATIVE_LOCO
        self.should_stop = False
        self._entry_step = 0
        self._entry_start = self.env.default_pos.copy()
        self._entry_target = self.env.default_pos.copy()

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

    def _enter_mimic(self):
        if self.state != NativeLocoMimicState.NATIVE_LOCO:
            logger.warning("Mimic control is already active or transitioning")
            return

        self.policy.reset()
        self.env.update_dof_cfg(override_cfg=self.policy.cfg_action_dof)
        self._entry_start = self.env.dof_pos
        self._entry_target = self.policy.get_init_dof_pos()
        result = self.env.acquire_user_control()
        if result != 0:
            logger.error("Failed to acquire G1 user control: %s", result)
            self.state = NativeLocoMimicState.FAULT
            self.should_stop = True
            raise RuntimeError(
                f"G1 user control acquire failed: {result} "
                f"status={self.env.get_control_status()}"
            )

        status = self.env.get_control_status()
        if (
            status["authority_state"] != "USER_ACTIVE"
            or status["fsm_id"] != 1000
            or not status["publish_enabled"]
            or status["active_publish_count"] <= 0
        ):
            self.env.release_to_passive()
            self.state = NativeLocoMimicState.FAULT
            self.should_stop = True
            raise RuntimeError(f"G1 user control was not armed: {status}")

        self._entry_step = 0
        self.state = NativeLocoMimicState.ENTERING_MIMIC
        logger.warning("G1 user control acquired; entering mimic: %s", status)

    def _return_to_native_loco(self):
        if self.state == NativeLocoMimicState.NATIVE_LOCO:
            return
        if self.state not in (
            NativeLocoMimicState.ENTERING_MIMIC,
            NativeLocoMimicState.MIMIC_ACTIVE,
        ):
            logger.error("Cannot return to native loco from state %s", self.state.name)
            return

        result = self.env.release_to_walkrun()
        if result == 0:
            self.state = NativeLocoMimicState.NATIVE_LOCO
            self.policy.reset()
            self.env.reset()
            logger.warning(
                "Control returned to Unitree WALKRUN: %s",
                self.env.get_control_status(),
            )
            return

        if self.env.has_user_control:
            self.state = NativeLocoMimicState.MIMIC_ACTIVE
            logger.error("Unitree WALKRUN handoff failed; retaining user control: %s", result)
            return

        self.state = NativeLocoMimicState.FAULT
        self.should_stop = True
        raise RuntimeError(
            f"G1 did not reach WALKRUN after release: {result} "
            f"status={self.env.get_control_status()}"
        )

    def _shutdown(self):
        result = self.close()
        if result != 0:
            raise RuntimeError(
                f"G1 shutdown could not confirm release to PASSIVE: {result}"
            )

    def close(self) -> int:
        if self.state == NativeLocoMimicState.CLOSED:
            return 0
        result = self.env.shutdown()
        if result == 0:
            self.state = NativeLocoMimicState.CLOSED
            self.should_stop = True
            return 0
        self.state = NativeLocoMimicState.FAULT
        logger.error("Failed to close G1 controller safely: %s", result)
        return result

    def _handle_commands(self, commands):
        for command in commands:
            if command == "[SHUTDOWN]":
                self._shutdown()
                return
            if command == "[POLICY_MIMIC]":
                self._enter_mimic()
            elif command == "[POLICY_LOCO]":
                self._return_to_native_loco()
            elif command.startswith("[POLICY_SWITCH]"):
                _, _, target = command.partition(",")
                if target == "NEXT":
                    self._select_mimic(1)
                elif target == "LAST":
                    self._select_mimic(-1)

    def _step_mimic_entry(self, env_data, ctrl_data):
        observation, extras = self.policy.get_observation(env_data, ctrl_data)
        progress = min(
            (self._entry_step + 1) / self.cfg.entry_transition_steps, 1.0
        )
        pd_target = (
            (1.0 - progress) * self._entry_start
            + progress * self._entry_target
        )
        self.env.step(pd_target)
        self._entry_step += 1
        if self._entry_step >= self.cfg.entry_transition_steps:
            self.state = NativeLocoMimicState.MIMIC_ACTIVE
            logger.warning("Mimic policy active: %s", self.policy.name)
        return extras, pd_target

    def _step_mimic(self, env_data, ctrl_data):
        observation, extras = self.policy.get_observation(env_data, ctrl_data)
        pd_target = self.policy.get_pd_target(observation)
        self.env.step(pd_target, extras.get("hand_pose", None))
        return extras, pd_target

    def _safety_check(self):
        if not self.do_safety_check or not self.env.has_user_control:
            return
        gravity_orientation = get_gravity_orientation(self.env.base_quat)
        angle = np.arccos(np.clip(-gravity_orientation[2], -1.0, 1.0))
        if abs(angle) <= 1.0:
            return
        logger.error("Robot fallen during mimic; returning to PASSIVE")
        result = self.env.release_to_passive()
        if result != 0:
            logger.error("Failed to return G1 to PASSIVE: %s", result)
            self.state = NativeLocoMimicState.FAULT
            raise RuntimeError(
                f"G1 safety shutdown could not confirm release to PASSIVE: {result}"
            )
        self.state = NativeLocoMimicState.FAULT
        self.should_stop = True

    def step(self, dry_run: bool = False):
        if self.state == NativeLocoMimicState.CLOSED:
            return

        self.env.update()
        env_data = self.env.get_data()
        ctrl_data = self.ctrl_manager.get_ctrl_data(env_data)
        commands = ctrl_data.get("COMMANDS", [])
        self._handle_commands(commands)

        if dry_run or self.state in (
            NativeLocoMimicState.NATIVE_LOCO,
            NativeLocoMimicState.FAULT,
            NativeLocoMimicState.CLOSED,
        ):
            self.ctrl_manager.post_step_callback(ctrl_data)
            return

        try:
            step_state = self.state
            if self.state == NativeLocoMimicState.ENTERING_MIMIC:
                extras, pd_target = self._step_mimic_entry(env_data, ctrl_data)
            else:
                extras, pd_target = self._step_mimic(env_data, ctrl_data)
        except Exception:
            if self.env.has_user_control:
                self.env.release_to_passive()
            self.state = NativeLocoMimicState.FAULT
            raise

        if step_state == NativeLocoMimicState.MIMIC_ACTIVE:
            self.policy.post_step_callback(commands)
        self.ctrl_manager.post_step_callback(ctrl_data)
        if (
            step_state == NativeLocoMimicState.MIMIC_ACTIVE
            and "[MOTION_DONE]" in extras.get("CALLBACK", [])
        ):
            self._return_to_native_loco()

        self._safety_check()
        if self.cfg.debug.log_obs:
            self.debug_logger.log(
                env_data=env_data,
                ctrl_data=ctrl_data,
                extras=extras,
                pd_target=pd_target,
                timestep=self.timestep,
            )
        self.timestep += 1
