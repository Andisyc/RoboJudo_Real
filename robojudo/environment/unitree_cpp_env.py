from __future__ import annotations
import logging
import time

import numpy as np
from unitree_cpp import RobotState, SportState, UnitreeController  # type: ignore

from robojudo.environment import Environment, env_registry
from robojudo.environment.env_cfgs import UnitreeEnvCfg
from robojudo.tools.retarget import HandRetarget
from robojudo.utils.rotation import TransformAlignment
from robojudo.utils.util_func import quat_rotate_inverse_np

logger = logging.getLogger(__name__)


@env_registry.register
class UnitreeCppEnv(Environment):
    cfg_env: UnitreeEnvCfg

    @staticmethod
    def validate_online_config(cfg_env):
        if (cfg_env.is_sim or not cfg_env.act or cfg_env.unitree.robot != "g1"
                or cfg_env.unitree.control_mode != "position"
                or not cfg_env.unitree.enable_torso_imu):
            raise ValueError("online FADA requires active G1 position control and torso IMU")
        if int(getattr(RobotState, "sensor_evidence_version", 0)) != 1:
            raise ValueError("online FADA requires rebuilt sensor-evidence v1 native bindings")

    def __init__(self, cfg_env: UnitreeEnvCfg, device="cpu"):
        self._sensor_evidence_version = int(getattr(RobotState, "sensor_evidence_version", 0))
        self._sensor_read_started_ns: int | None = None
        self.enabled: bool = cfg_env.act
        self._shutdown_attempted = False
        self._shutdown_error = None
        super().__init__(cfg_env=cfg_env, device=device)
        self.RemoteControllerHandler = None

        cfg_unitree: UnitreeEnvCfg.UnitreeCfg = cfg_env.unitree
        self._enable_torso_imu = bool(cfg_unitree.enable_torso_imu)

        cfg_unitree_dict: dict = cfg_unitree.to_dict()
        cfg_unitree_dict["num_dofs"] = self.num_dofs
        cfg_unitree_dict["stiffness"] = self.stiffness
        cfg_unitree_dict["damping"] = self.damping

        self.robot = cfg_unitree.robot
        self._dof_idx = cfg_env.joint2motor_idx
        self._odometry_type = cfg_env.odometry_type
        if self._odometry_type == "ZED":
            assert self.cfg_env.zed_cfg is not None, "zed_cfg must be set if odometry_type is 'ZED'"
            from robojudo.tools.zed_odometry import ZedOdometry

            self.zed_odometry = ZedOdometry(self.cfg_env.zed_cfg)
        elif self._odometry_type == "DUMMY":
            pass
        elif self._odometry_type == "UNITREE":
            pass

        self.hand_type = cfg_unitree.hand_type
        if self.hand_type == "Inspire":
            self.hand_retarget = HandRetarget(cfg_env.hand_retarget)
        elif self.hand_type == "Dex-3":
            self.hand_retarget = None  # TODO
        else:
            self.hand_retarget = None

        self.sport_state: SportState = None  # pyright: ignore[reportAttributeAccessIssue]
        self.robot_state: RobotState = None  # pyright: ignore[reportAttributeAccessIssue]

        self.unitree = UnitreeController(cfg_unitree_dict)

        try:
            if self.robot == "h1":
                self.torso_align = TransformAlignment()
            self.self_check()
        except BaseException:
            try:
                self.shutdown()
            except BaseException:
                logger.exception("Native environment initialization safety request failed")
            raise

    def self_check(self):
        for _ in range(30):
            time.sleep(0.1)
            if self.unitree.self_check():
                logger.info("UnitreeCppEnv self check passed!")
                break
        if not self.unitree.self_check():
            logger.critical("UnitreeCppEnv self check failed!")
            exit()

    def reset(self):
        if self.born_place_align:  # TODO: merge
            self.born_place_align = False  # disable during reset
            self.update()
            self.born_place_align = True  # enable after reset
            self.set_born_place()
            self.update()

    def set_born_place(self, quat: np.ndarray | None = None, pos: np.ndarray | None = None):
        quat_ = self.base_quat if quat is None else quat
        pos_ = self.base_pos if pos is None else pos
        super().set_born_place(quat_, pos_)

        if self.robot == "h1":
            self.torso_align.set_base(quat=self.torso_quat)

        if self._odometry_type == "ZED":
            self.zed_odometry.set_zreo()

    def update(self):
        # robot state
        self._sensor_read_started_ns = time.monotonic_ns()
        self.robot_state = self.unitree.get_robot_state()
        if self._dof_idx is None:
            self._dof_pos = np.array(self.robot_state.motor_state.q, dtype=np.float32)
            self._dof_vel = np.array(self.robot_state.motor_state.dq, dtype=np.float32)
        else:
            self._dof_pos = np.array(
                [self.robot_state.motor_state.q[self._dof_idx[i]] for i in range(len(self._dof_idx))],
                dtype=np.float32,
            )
            self._dof_vel = np.array(
                [self.robot_state.motor_state.dq[self._dof_idx[i]] for i in range(len(self._dof_idx))],
                dtype=np.float32,
            )

        if self.robot == "g1":
            quat = np.array(self.robot_state.imu_state.quaternion, dtype=np.float32)[[1, 2, 3, 0]]
            ang_vel = np.array(self.robot_state.imu_state.gyroscope, dtype=np.float32)
            rpy = np.array(self.robot_state.imu_state.rpy, dtype=np.float32)

            if self.born_place_align:
                quat = self.base_align.align_quat(quat)

            self._base_quat = quat
            self._base_ang_vel = ang_vel
            self._base_rpy = rpy

        elif self.robot == "h1":
            raise NotImplementedError("H1 robot with unitree_cpp not supported yet.")

        # odometry
        if self._odometry_type == "ZED":
            self.zed_odometry.update()
            if self.zed_odometry.is_valid:
                # born place aligned in zed_odometry
                self._base_pos = self.zed_odometry.pos
                self._lin_vel = self.zed_odometry.lin_vel
        elif self._odometry_type == "DUMMY":
            self._base_pos = np.array([0.0, 0.0, 0.8])
            self._base_lin_vel = np.array([0.0, 0.0, 0.0])
        elif self._odometry_type == "UNITREE":
            self.sport_state = self.unitree.get_sport_state()
            base_pos = np.asarray(self.sport_state.position, dtype=np.float32)
            lin_vel = np.asarray(self.sport_state.velocity, dtype=np.float32)
            self._base_lin_vel = quat_rotate_inverse_np(self.base_quat, lin_vel)
            if self.born_place_align:
                self._base_pos = self.base_align.align_pos(base_pos)

        # FK
        if self.update_with_fk:
            fk_info = self.fk()
            self._torso_pos = fk_info[self._torso_name]["pos"]
            if self.robot != "h1":
                self._torso_quat = fk_info[self._torso_name]["quat"]
                self._torso_ang_vel = fk_info[self._torso_name]["ang_vel"]

        # controller
        if self.RemoteControllerHandler:
            self.RemoteControllerHandler(self.robot_state.wireless_remote)

    def get_data(self):
        env_data = super().get_data()
        env_data["sensor_tick"] = int(self.robot_state.tick)
        env_data["sensor_read_started_ns"] = self._sensor_read_started_ns
        if self._sensor_evidence_version == 1:
            env_data["torso_imu_sequence"] = self.robot_state.torso_imu_sequence
            env_data["lowstate_age_s"] = self.robot_state.lowstate_age_s
            env_data["torso_imu_age_s"] = self.robot_state.torso_imu_age_s
        else:
            env_data["torso_imu_sequence"] = None
            env_data["lowstate_age_s"] = None
            env_data["torso_imu_age_s"] = None
        if self._enable_torso_imu:
            torso_imu_state = getattr(self.robot_state, "torso_imu_state", None)
            if torso_imu_state is not None:
                env_data["torso_imu_state"] = torso_imu_state
        return env_data

    def step(self, pd_target, hand_pose=None) -> int | None:
        assert len(pd_target) == self.num_dofs, "pd_target len should be num_dofs of env"

        # limits = self.position_limits
        # pd_target_clipped = np.clip(pd_target, limits[:, 0], limits[:, 1])

        # delta = pd_target - pd_target_clipped
        # if np.any(delta != 0):
        #     logger.warning(f"JOINT out of LIMIT-> {delta}")

        # positions = pd_target_clipped
        positions = pd_target
        submitted_ns = None
        if self.enabled:
            self.unitree.step(positions.tolist())
            submitted_ns = time.monotonic_ns()

        if hand_pose is not None:
            assert type(hand_pose) is np.ndarray, "hand_pose should be a numpy array"
            assert hand_pose.shape[0] == 2, "hand_pose should be of shape (2, -1)"
            if self.hand_retarget is not None:
                hand_pose = self.hand_retarget.from_pose_to_cmd(hand_pose)
                logger.debug(f"Hand pose retargeted: {hand_pose}")
            hand_pose = hand_pose.tolist()

            if self.enabled:
                self.unitree.step_hands(hand_pose[0], hand_pose[1])
        return submitted_ns

    def shutdown(self):
        self.enabled = False
        if self._shutdown_attempted:
            if self._shutdown_error is not None:
                raise self._shutdown_error
            return
        self._shutdown_attempted = True
        try:
            self.unitree.shutdown()
        except BaseException as error:
            self._shutdown_error = error
            raise

    def set_gains(self, stiffness, damping):
        if not hasattr(self, "unitree"):  # TODO
            return
        if not self.enabled:
            return
        self.unitree.set_gains(stiffness, damping)


if __name__ == "__main__":
    from robojudo.config.g1.env.g1_real_env_cfg import G1RealEnvCfg

    env = UnitreeCppEnv(cfg_env=G1RealEnvCfg())
    env.set_gains(
        stiffness=[kp * 0.0 for kp in env.stiffness],
        damping=[kd * 0.1 for kd in env.damping],
    )
    while 1:
        # env.step(np.zeros(29), np.ones((2, 7)) * -0)
        env.step(np.zeros(29), None)
        # if controller.remote_controller("A"):
        #     controller.shutdown()
        print(env.base_rpy)
        print(env.dof_pos)
        print(env.base_pos)
        env.update()
        # print(env.base_pos)
        time.sleep(0.1)
    # print("Exit")
    # from robojudo.controller import UnitreeCtrl
    # ctrl = UnitreeCtrl(env=env)

    # while True:
    #     env.update()
    #     state = ctrl.get_state()
    #     events = ctrl.get_events()
    #     print("State:", state)
    #     print("Events:", events)
    #     time.sleep(0.1)  # Simulate a control loop
