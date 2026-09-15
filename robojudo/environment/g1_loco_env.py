from __future__ import annotations

import logging
import time

import numpy as np
from g1_loco_bridge import (  # type: ignore
    AuthorityState,
    G1LocoController,
    RobotState,
    SportState,
)

from robojudo.environment import Environment, env_registry
from robojudo.environment.env_cfgs import UnitreeEnvCfg
from robojudo.utils.util_func import quat_rotate_inverse_np

logger = logging.getLogger(__name__)


@env_registry.register
class G1LocoEnv(Environment):
    """G1 state adapter and explicit Unitree/user control authority boundary."""

    cfg_env: UnitreeEnvCfg

    def __init__(self, cfg_env: UnitreeEnvCfg, device: str = "cpu"):
        self.enabled = bool(cfg_env.act)
        self.RemoteControllerHandler = None
        super().__init__(cfg_env=cfg_env, device=device)

        cfg_unitree = cfg_env.unitree
        if cfg_unitree.robot != "g1":
            raise ValueError("G1LocoEnv only supports Unitree G1")

        self._enable_torso_imu = bool(cfg_unitree.enable_torso_imu)
        self._dof_idx = cfg_env.joint2motor_idx
        self._odometry_type = cfg_env.odometry_type
        self.robot_state: RobotState | None = None
        self.sport_state: SportState | None = None

        cfg_unitree_dict = cfg_unitree.to_dict()
        cfg_unitree_dict["num_dofs"] = self.num_dofs
        cfg_unitree_dict["stiffness"] = self.stiffness.tolist()
        cfg_unitree_dict["damping"] = self.damping.tolist()
        self.unitree = G1LocoController(cfg_unitree_dict)

        if self._odometry_type == "ZED":
            if self.cfg_env.zed_cfg is None:
                raise ValueError("zed_cfg must be set if odometry_type is 'ZED'")
            from robojudo.tools.zed_odometry import ZedOdometry

            self.zed_odometry = ZedOdometry(self.cfg_env.zed_cfg)

        self.self_check()

    def self_check(self):
        for _ in range(30):
            time.sleep(0.1)
            if self.unitree.self_check():
                logger.info("G1LocoEnv self check passed; control remains with Unitree loco")
                return
        raise RuntimeError(
            "G1LocoEnv self check failed or FSM 1000 is already owned by another process"
        )

    def reset(self):
        if not self.born_place_align:
            return
        self.born_place_align = False
        self.update()
        self.born_place_align = True
        self.set_born_place()
        self.update()

    def set_born_place(
        self, quat: np.ndarray | None = None, pos: np.ndarray | None = None
    ):
        quat_value = self.base_quat if quat is None else quat
        pos_value = self.base_pos if pos is None else pos
        super().set_born_place(quat_value, pos_value)
        if self._odometry_type == "ZED":
            self.zed_odometry.set_zreo()

    def update(self):
        self.robot_state = self.unitree.get_robot_state()
        motor_state = self.robot_state.motor_state
        if self._dof_idx is None:
            self._dof_pos = np.asarray(motor_state.q, dtype=np.float32)
            self._dof_vel = np.asarray(motor_state.dq, dtype=np.float32)
        else:
            self._dof_pos = np.asarray(
                [motor_state.q[index] for index in self._dof_idx], dtype=np.float32
            )
            self._dof_vel = np.asarray(
                [motor_state.dq[index] for index in self._dof_idx], dtype=np.float32
            )

        imu_state = self.robot_state.imu_state
        base_quat = np.asarray(imu_state.quaternion, dtype=np.float32)[[1, 2, 3, 0]]
        if self.born_place_align:
            base_quat = self.base_align.align_quat(base_quat)
        self._base_quat = base_quat
        self._base_ang_vel = np.asarray(imu_state.gyroscope, dtype=np.float32)
        self._base_rpy = np.asarray(imu_state.rpy, dtype=np.float32)

        if self._odometry_type == "ZED":
            self.zed_odometry.update()
            if self.zed_odometry.is_valid:
                self._base_pos = self.zed_odometry.pos
                self._base_lin_vel = self.zed_odometry.lin_vel
        elif self._odometry_type == "DUMMY":
            self._base_pos = np.asarray([0.0, 0.0, 0.8], dtype=np.float32)
            self._base_lin_vel = np.zeros(3, dtype=np.float32)
        elif self._odometry_type == "UNITREE":
            self.sport_state = self.unitree.get_sport_state()
            base_pos = np.asarray(self.sport_state.position, dtype=np.float32)
            lin_vel = np.asarray(self.sport_state.velocity, dtype=np.float32)
            self._base_lin_vel = quat_rotate_inverse_np(self.base_quat, lin_vel)
            self._base_pos = (
                self.base_align.align_pos(base_pos) if self.born_place_align else base_pos
            )

        if self.update_with_fk:
            fk_info = self.fk()
            self._torso_pos = fk_info[self._torso_name]["pos"]
            self._torso_quat = fk_info[self._torso_name]["quat"]
            self._torso_ang_vel = fk_info[self._torso_name]["ang_vel"]

        if self.RemoteControllerHandler is not None:
            self.RemoteControllerHandler(self.robot_state.wireless_remote)

    def get_data(self):
        env_data = super().get_data()
        if self._enable_torso_imu and self.robot_state is not None:
            env_data["torso_imu_state"] = self.robot_state.torso_imu_state
        return env_data

    @property
    def authority_state(self):
        return self.unitree.get_authority_state()

    @property
    def has_user_control(self) -> bool:
        return self.authority_state == AuthorityState.USER_ACTIVE

    @property
    def has_internal_control(self) -> bool:
        return self.authority_state == AuthorityState.INTERNAL

    @property
    def has_authority_fault(self) -> bool:
        return self.authority_state == AuthorityState.FAULT

    def get_control_status(self) -> dict:
        authority_state = self.authority_state
        return {
            "authority_state": getattr(authority_state, "name", str(authority_state)),
            "fsm_id": int(self.unitree.get_cached_fsm_id()),
            "last_loco_api_result": int(self.unitree.get_last_loco_api_result()),
            "publish_enabled": bool(self.unitree.is_publish_enabled()),
        }

    def _handoff(self, operation: str, callback) -> int:
        before = self.get_control_status()
        result = int(callback())
        after = self.get_control_status()
        logger.warning(
            "G1 control handoff %s: result=%s before=%s after=%s",
            operation,
            result,
            before,
            after,
        )
        return result

    def acquire_user_control(self) -> int:
        return self._handoff("USER", self.unitree.acquire_user_control)

    def release_to_walkrun(self) -> int:
        return self._handoff("WALKRUN", self.unitree.release_to_walkrun)

    def release_to_passive(self) -> int:
        return self._handoff("PASSIVE", self.unitree.release_to_passive)

    def step(self, pd_target, hand_pose=None):
        if hand_pose is not None:
            raise NotImplementedError("G1LocoEnv does not control dexterous hands")
        if not self.enabled:
            return
        if not self.has_user_control:
            raise RuntimeError("G1LocoEnv cannot send PD targets without user control")
        if len(pd_target) != self.num_dofs:
            raise ValueError("pd_target length must match environment num_dofs")
        self.unitree.step(np.asarray(pd_target, dtype=np.float64).tolist())

    def shutdown(self) -> int:
        if not hasattr(self, "unitree"):
            return 0
        result = int(self.unitree.close())
        if result == 0:
            self.enabled = False
        else:
            logger.error(
                "G1 controller close failed; retaining control resources: %s status=%s",
                result,
                self.get_control_status(),
            )
        return result

    def set_gains(self, stiffness, damping):
        if not hasattr(self, "unitree"):
            return
        self.unitree.set_gains(
            np.asarray(stiffness, dtype=np.float64).tolist(),
            np.asarray(damping, dtype=np.float64).tolist(),
        )
