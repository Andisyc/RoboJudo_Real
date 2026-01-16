from typing import Tuple, List, Optional, Union
import logging
import time

import numpy as np
from scipy.spatial.transform import Rotation as sRot

# Unitree SDK
from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelPublisher, ChannelSubscriber  # type: ignore
from unitree_sdk2py.idl.default import (  # type: ignore
    unitree_go_msg_dds__LowCmd_,
    unitree_go_msg_dds__LowState_,
    unitree_go_msg_dds__SportModeState_,
    unitree_hg_msg_dds__HandCmd_,
    unitree_hg_msg_dds__LowCmd_,
    unitree_hg_msg_dds__LowState_,
)
from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowCmd_ as LowCmdGo  # type: ignore
from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowState_ as LowStateGo  # type: ignore
from unitree_sdk2py.idl.unitree_go.msg.dds_ import SportModeState_ as SportModeState  # type: ignore
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import HandCmd_  # type: ignore
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_ as LowCmdHG  # type: ignore
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_ as LowStateHG  # type: ignore
from unitree_sdk2py.utils.crc import CRC  # type: ignore
from unitree_sdk2py.utils.thread import RecurrentThread  # type: ignore

from robojudo.environment import Environment, env_registry
from robojudo.environment.env_cfgs import UnitreeEnvCfg
from robojudo.environment.utils.unitree_command import (
    MotorMode,
    create_damping_cmd,
    create_zero_cmd,
    init_cmd_go,
    init_cmd_hg,
)
from robojudo.environment.utils.unitree_rotation import transform_imu_data
from robojudo.tools.retarget import HandRetarget
from robojudo.utils.rotation import TransformAlignment
from robojudo.utils.util_func import calc_heading_quat_np, quat_rotate_inverse_np

logger = logging.getLogger(__name__)


@env_registry.register
class UnitreeEnv(Environment):
    cfg_env: UnitreeEnvCfg

    def __init__(self, cfg_env: UnitreeEnvCfg, device="cpu"):
        self.enabled: bool = cfg_env.act
        super().__init__(cfg_env=cfg_env, device=device)

        ChannelFactoryInitialize(0, self.cfg_env.unitree.net_if)

        self.RemoteControllerHandler = None
        self.robot = self.cfg_env.unitree.robot
        self._control_dt = self.cfg_env.unitree.control_dt
        self._msg_type = self.cfg_env.unitree.msg_type
        self._dof_idx = self.cfg_env.joint2motor_idx
        self._control_mode = self.cfg_env.unitree.control_mode
        self._control_joint_idx = list(range(self.num_dofs))  # control all joints

        self.set_gains(self.stiffness, self.damping)

        self.sport_state: SportModeState = None
        self.low_state: LowStateHG | LowStateGo = None

        if self._msg_type == "hg":
            # g1 and h1_2 use the hg msg type
            self.low_cmd = unitree_hg_msg_dds__LowCmd_()
            self.low_state = unitree_hg_msg_dds__LowState_()
            self.mode_pr_ = MotorMode.PR
            self.mode_machine_ = 0

            self.lowcmd_publisher_ = ChannelPublisher(self.cfg_env.unitree.lowcmd_topic, LowCmdHG)
            self.lowcmd_publisher_.Init()
            self.lowstate_subscriber = ChannelSubscriber(self.cfg_env.unitree.lowstate_topic, LowStateHG)
            self.lowstate_subscriber.Init(self.LowStateHgHandler, 10)

            init_cmd_hg(self.low_cmd, self.mode_machine_, self.mode_pr_)
        elif self._msg_type == "go":
            # h1 uses the go msg type
            self.low_cmd = unitree_go_msg_dds__LowCmd_()
            self.low_state = unitree_go_msg_dds__LowState_()

            self.lowcmd_publisher_ = ChannelPublisher(self.cfg_env.unitree.lowcmd_topic, LowCmdGo)
            self.lowcmd_publisher_.Init()
            self.lowstate_subscriber = ChannelSubscriber(self.cfg_env.unitree.lowstate_topic, LowStateGo)
            self.lowstate_subscriber.Init(self.LowStateGoHandler, 10)

            init_cmd_go(self.low_cmd, weak_motor=self.cfg_env.weak_motor)
        else:
            raise ValueError("Invalid msg_type")

        self._init_done = False
        self.wait_for_low_state()

        # Odometry setup
        self._odometry_type = self.cfg_env.odometry_type
        if self._odometry_type == "ZED":
            assert self.cfg_env.zed_cfg is not None, "zed_cfg must be set if odometry_type is 'ZED'"
            from robojudo.tools.zed_odometry import ZedOdometry

            self.zed_odometry = ZedOdometry(self.cfg_env.zed_cfg)
        elif self._odometry_type == "UNITREE":
            self.sport_state = unitree_go_msg_dds__SportModeState_()
            self.sport_state_subscriber = ChannelSubscriber(self.cfg_env.unitree.sport_state_topic, SportModeState)
            self.sport_state_subscriber.Init(self.SportStateHandler, 10)

        # Hand setup
        self.hand_type = self.cfg_env.unitree.hand_type
        if self.hand_type == "Inspire":
            self.hand_retarget = HandRetarget(self.cfg_env.hand_retarget)
        else:
            self.hand_retarget = None

        if self.hand_type == "Dex-3":
            self.left_hand_cmd = unitree_hg_msg_dds__HandCmd_()
            self.left_hand_cmd_publisher = ChannelPublisher("rt/dex3/left/cmd", HandCmd_)
            self.left_hand_cmd_publisher.Init()

            self.right_hand_cmd = unitree_hg_msg_dds__HandCmd_()
            self.right_hand_cmd_publisher = ChannelPublisher("rt/dex3/right/cmd", HandCmd_)
            self.right_hand_cmd_publisher.Init()
        elif self.hand_type == "Inspire":
            from inspire_sdkpy import inspire_dds, inspire_hand_defaut  # type: ignore

            self.left_hand_cmd = inspire_hand_defaut.get_inspire_hand_ctrl()
            self.left_hand_cmd_publisher = ChannelPublisher("rt/inspire_hand/ctrl/l", inspire_dds.inspire_hand_ctrl)
            self.left_hand_cmd_publisher.Init()

            self.right_hand_cmd = inspire_hand_defaut.get_inspire_hand_ctrl()
            self.right_hand_cmd_publisher = ChannelPublisher("rt/inspire_hand/ctrl/r", inspire_dds.inspire_hand_ctrl)
            self.right_hand_cmd_publisher.Init()
        elif self.hand_type == "NONE":
            pass
        else:
            raise ValueError(f"Invalid hand type: {self.hand_type}")

        self.lowcmd_send_thread = RecurrentThread(interval=self._control_dt, target=self.send_cmd, name="control")
        self.lowcmd_send_thread.Start()

        # born place alignment extra for h1 torso
        if self.robot == "h1":
            self.torso_align = TransformAlignment()
        
        # === [新增] 性能优化：预分配内存 ===
        # 根据实际关节数分配固定内存
        self._dof_pos = np.zeros(self.num_dofs, dtype=np.float32)
        self._dof_vel = np.zeros(self.num_dofs, dtype=np.float32)

        # 预计算关节索引映射，避免在 update 里做 if else 判断
        # 如果没有指定映射，就用 0 ~ num_dofs
        if self._dof_idx is None:
            self._map_dof_idx = list(range(self.num_dofs))
        else:
            self._map_dof_idx = self._dof_idx

        # 预分配 IMU 缓存
        self._base_quat = np.zeros(4, dtype=np.float32)
        self._base_ang_vel = np.zeros(3, dtype=np.float32)
        self._base_rpy = np.zeros(3, dtype=np.float32)
        # =================================

        # update()在self_check()被调用
        # 性能初始化需在self_check()之前
        self.self_check()

    def wait_for_low_state(self):
        while self.low_state.tick == 0:
            time.sleep(self.cfg_env.unitree.control_dt)
        logger.info("Successfully connect to the robot")
        self._init_done = True

        if self._msg_type == "hg":
            mode_machine = self.low_state.mode_machine
            if mode_machine != self.mode_machine_:
                logger.info(f"[UnitreeEnv] {self.robot} mode_machine set to {mode_machine}")
                self.mode_machine_ = mode_machine
                init_cmd_hg(self.low_cmd, self.mode_machine_, self.mode_pr_)

    def self_check(self):
        assert self._init_done, "[UnitreeEnv] not inited as expected, no data received."
        # logger.debug("Testing observation ...")
        for _ in range(10):
            self.update()
            time.sleep(0.02)
        # logger.debug("Observation test done!")

    def reset(self):
        if self.born_place_align:  # TODO: merge
            self.born_place_align = False  # disable during reset
            self.update()
            self.born_place_align = True  # enable after reset
            self.set_born_place()
            self.update()

    # def set_born_place(self, quat: np.ndarray | None = None, pos: np.ndarray | None = None):
    def set_born_place(self, quat: Optional[np.ndarray] = None, pos: Optional[np.ndarray] = None):
        quat_ = self.base_quat if quat is None else quat
        pos_ = self.base_pos if pos is None else pos
        super().set_born_place(quat_, pos_)
        logger.info(f"[UnitreeEnv] born place set to pos: {pos_}, quat: {quat_}")

        if self.robot == "h1":
            torso_quat = self.torso_quat
            torso_quat = calc_heading_quat_np(torso_quat)  # keep yaw only
            self.torso_align.set_base(quat=torso_quat)

        if self._odometry_type == "ZED":
            self.zed_odometry.set_zreo()

    def SportStateHandler(self, msg: SportModeState):
        self.sport_state = msg

    def LowStateHgHandler(self, msg: LowStateHG):
        self.low_state = msg
        # self.mode_machine_ = self.low_state.mode_machine

    def LowStateGoHandler(self, msg: LowStateGo):
        self.low_state = msg

    
    def update(self):
        # robot state
        dof_pos = []
        dof_vel = []
        for motor_state in self.low_state.motor_state:
            dof_pos.append(motor_state.q)
            dof_vel.append(motor_state.dq)
        dof_pos = np.asarray(dof_pos, dtype=np.float32)
        dof_vel = np.asarray(dof_vel, dtype=np.float32)
        if self._dof_idx is None:
            dof_pos = dof_pos[: self.num_dofs]
            dof_vel = dof_vel[: self.num_dofs]
        else:
            dof_pos = dof_pos[self._dof_idx]
            dof_vel = dof_vel[self._dof_idx]
        self._dof_pos = dof_pos
        self._dof_vel = dof_vel

        # read in imu info
        if self.robot == "g1":
            quat = np.array(self.low_state.imu_state.quaternion, dtype=np.float32)[[1, 2, 3, 0]]
            ang_vel = np.array(self.low_state.imu_state.gyroscope, dtype=np.float32)
            rpy = np.array(self.low_state.imu_state.rpy, dtype=np.float32)

            if self.born_place_align:
                quat = self.base_align.align_quat(quat)

            self._base_quat = quat
            self._base_ang_vel = ang_vel
            self._base_rpy = rpy

        elif self.robot == "h1":
            # h1 imu is on the torso
            # imu data needs to be transformed to the pelvis frame
            torso_quat = np.array(self.low_state.imu_state.quaternion, dtype=np.float32)[[1, 2, 3, 0]]
            torso_ang_vel = np.array(self.low_state.imu_state.gyroscope, dtype=np.float32)

            if self.born_place_align:
                torso_quat = self.torso_align.align_quat(torso_quat)

            self._torso_quat = torso_quat
            self._torso_ang_vel = torso_ang_vel

            # Warn: torso index fixed
            waist_yaw = self.low_state.motor_state.q[self._dof_idx[10] if self._dof_idx is not None else 10]
            waist_yaw_omega = self.low_state.motor_state.dq[self._dof_idx[10] if self._dof_idx is not None else 10]
            base_quat, base_ang_vel = transform_imu_data(
                waist_yaw=waist_yaw,
                waist_yaw_omega=waist_yaw_omega,
                imu_quat=torso_quat[[3, 0, 1, 2]],
                imu_omega=torso_ang_vel,)

            self._base_quat = base_quat[[1, 2, 3, 0]]
            self._base_ang_vel = base_ang_vel
            self._base_rpy = sRot.from_quat(base_quat, scalar_first=True).as_euler("xyz")

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
            base_pos = np.array(self.sport_state.position, dtype=np.float32)
            lin_vel = np.array(self.sport_state.velocity, dtype=np.float32)
            self._base_lin_vel = quat_rotate_inverse_np(self.base_quat, lin_vel)
            if self.born_place_align:
                self._base_pos = self.base_align.align_pos(base_pos)

        # FK (Forward Kinematics)
        if self.update_with_fk:
            fk_info = self.fk()
            self._torso_pos = fk_info[self._torso_name]["pos"]
            if self.robot != "h1":
                self._torso_quat = fk_info[self._torso_name]["quat"]
                self._torso_ang_vel = fk_info[self._torso_name]["ang_vel"]

        # controller
        if self.RemoteControllerHandler:
            self.RemoteControllerHandler(self.low_state.wireless_remote)
    
    """
    def update(self): # performance optimize
        # === [优化版] 零内存分配读取 ===
        
        # 1. 读取电机状态 (避免创建 list 和 numpy array)
        ms = self.low_state.motor_state
        # 直接通过索引映射填入预分配的数组
        for i, motor_id in enumerate(self._map_dof_idx):
            # 注意：这里假设 ms 是可索引的，或者 ms 足够长
            state = ms[motor_id]
            self._dof_pos[i] = state.q
            self._dof_vel[i] = state.dq
        
        # 2. 读取 IMU (避免切片产生的拷贝)
        if self.robot == "g1":
            imu = self.low_state.imu_state
            
            # 手动重排 quaternion: [w, x, y, z] -> [x, y, z, w]
            # 原始代码: quat = np.array(...)[[1, 2, 3, 0]]
            q = imu.quaternion
            self._base_quat[0] = q[1]
            self._base_quat[1] = q[2]
            self._base_quat[2] = q[3]
            self._base_quat[3] = q[0]

            # 直接填入角速度
            gyro = imu.gyroscope
            self._base_ang_vel[0] = gyro[0]
            self._base_ang_vel[1] = gyro[1]
            self._base_ang_vel[2] = gyro[2]
            
            # 直接填入 RPY
            rpy = imu.rpy
            self._base_rpy[0] = rpy[0]
            self._base_rpy[1] = rpy[1]
            self._base_rpy[2] = rpy[2]

            if self.born_place_align:
                # 注意：align_quat 通常会返回新数组，这里尽量优化，或者暂时接受
                # 如果 align_quat 可以原地修改更好，如果不行，这里会有一次拷贝
                self._base_quat[:] = self.base_align.align_quat(self._base_quat)

        elif self.robot == "h1":
            # H1 的逻辑稍微复杂，但也可以用类似方法优化
            # 暂时保留原逻辑，但建议用类似 G1 的方式手动提取 imu 数据
            # ... (如果你的机器人是 G1，H1 分支不会执行，可以暂时不动)
            pass

        # 3. Odometry (保持原样，通常开销不大)
        if self._odometry_type == "ZED":
            self.zed_odometry.update()
            if self.zed_odometry.is_valid:
                self._base_pos = self.zed_odometry.pos
                self._lin_vel = self.zed_odometry.lin_vel
        elif self._odometry_type == "DUMMY":
            self._base_pos[:] = [0.0, 0.0, 0.8]
            self._base_lin_vel[:] = 0.0
        elif self._odometry_type == "UNITREE":
            # 优化：避免反复创建 array
            # 这里假设 sport_state 已经准备好
            pos = self.sport_state.position
            vel = self.sport_state.velocity
            
            # 这是一个计算函数，会有计算开销，但比内存分配快
            self._base_lin_vel = quat_rotate_inverse_np(self._base_quat, np.array(vel))
            
            if self.born_place_align:
                self._base_pos = self.base_align.align_pos(np.array(pos))
            else:
                self._base_pos[:] = pos

        # 4. FK (Forward Kinematics)
        if self.update_with_fk:
            fk_info = self.fk()
            self._torso_pos = fk_info[self._torso_name]["pos"]
            if self.robot != "h1":
                self._torso_quat = fk_info[self._torso_name]["quat"]
                self._torso_ang_vel = fk_info[self._torso_name]["ang_vel"]

        # 5. Controller Handler
        if self.RemoteControllerHandler:
            self.RemoteControllerHandler(self.low_state.wireless_remote)
    """

    def step(self, pd_target, hand_pose=None):
        assert len(pd_target) == self.num_dofs, "pd_target len should be num_dofs of env"

        # limits = self.position_limits
        # pd_target_clipped = np.clip(pd_target, limits[:, 0], limits[:, 1])

        # delta = pd_target - pd_target_clipped
        # if np.any(delta != 0):
        #     logger.warning(f"JOINT out of LIMIT-> {delta}")

        # positions = pd_target_clipped
        positions = pd_target

        # hand retarget
        if hand_pose is not None and self.hand_retarget is not None:
            hand_pose = self.hand_retarget.from_pose_to_cmd(hand_pose)
            logger.debug(f"Hand pose retargeted: {hand_pose}")

        self.control_joints(positions, hand_pose)

    def shutdown(self):
        self.set_damping_mode()
        self.enabled = False

    def set_zero_torque_mode(self):
        create_zero_cmd(self.low_cmd)
        self.send_cmd(self.low_cmd)

    def set_damping_mode(self):
        create_damping_cmd(self.low_cmd)
        self.send_cmd(self.low_cmd)

    # def send_cmd(self, cmd: LowCmdGo | LowCmdHG = None):
    def send_cmd(self, cmd: Union[LowCmdGo, LowCmdHG] = None):
        
        if cmd is None:
            cmd = self.low_cmd
        cmd.crc = CRC().Crc(cmd)
        self.lowcmd_publisher_.Write(cmd)

    def set_cmd_i(self, i, command, kp=0, kd=0, control_type="position"):
        # if i > 19:
        #     raise IndexError(f"{i} is bigger than 18!")
        if control_type == 0 or control_type == "position":
            self.low_cmd.motor_cmd[i].q = command
            self.low_cmd.motor_cmd[i].qd = 0
            self.low_cmd.motor_cmd[i].kp = kp
            self.low_cmd.motor_cmd[i].kd = kd
            self.low_cmd.motor_cmd[i].tau = 0
        elif control_type == 1 or control_type == "velocity":
            self.low_cmd.motor_cmd[i].q = 0
            self.low_cmd.motor_cmd[i].qd = command
            self.low_cmd.motor_cmd[i].kp = kp
            self.low_cmd.motor_cmd[i].kd = kd
            self.low_cmd.motor_cmd[i].tau = 0
        elif control_type == 2 or control_type == "torque":
            self.low_cmd.motor_cmd[i].q = 0
            self.low_cmd.motor_cmd[i].qd = 0
            self.low_cmd.motor_cmd[i].kp = 0
            self.low_cmd.motor_cmd[i].kd = 0
            self.low_cmd.motor_cmd[i].tau = command
        else:
            raise ValueError(f"No such control mode: {command}")

    def set_gains(self, stiffness, damping):
        if not self.enabled:
            return
        assert len(stiffness) == self.num_dofs and len(damping) == self.num_dofs, f"list shape must be {self.num_dofs}!"

        self.kps = stiffness
        self.kds = damping

    def send_dex_hand_cmd(self, hand_pose):
        assert len(hand_pose) == 14

        for i in range(7):
            self.left_hand_cmd.motor_cmd[i].q = hand_pose[i]
            self.left_hand_cmd.motor_cmd[i].qd = 0
            self.left_hand_cmd.motor_cmd[i].kp = 1
            self.left_hand_cmd.motor_cmd[i].kd = 0.1
            self.left_hand_cmd.motor_cmd[i].tau = 0

            self.right_hand_cmd.motor_cmd[i].q = hand_pose[i + 7]
            self.right_hand_cmd.motor_cmd[i].qd = 0
            self.right_hand_cmd.motor_cmd[i].kp = 1
            self.right_hand_cmd.motor_cmd[i].kd = 0.1
            self.right_hand_cmd.motor_cmd[i].tau = 0

        self.left_hand_cmd_publisher.Write(self.left_hand_cmd)
        self.right_hand_cmd_publisher.Write(self.right_hand_cmd)

    def send_inspire_hand_cmd(self, hand_pose: np.ndarray):
        assert hand_pose.shape == (2, 6), "inspire hand_pose should be a (2, 6) array"

        if hand_pose.dtype != np.int32:
            hand_pose = hand_pose.astype(np.int32)

        self.left_hand_cmd.angle_set = hand_pose[0].tolist()
        self.left_hand_cmd.mode = 0b0001  # angle control
        self.right_hand_cmd.angle_set = hand_pose[1].tolist()
        self.right_hand_cmd.mode = 0b0001  # angle control

        self.left_hand_cmd_publisher.Write(self.left_hand_cmd)
        self.right_hand_cmd_publisher.Write(self.right_hand_cmd)

    def control_joints(self, commands, hand_pose=None):
        if not self.enabled:
            return

        if hand_pose is not None:
            # logger.debug(f"Control hand pose: {hand_pose}")
            """
            match self.hand_type:
                case "Dex-3":
                    self.send_dex_hand_cmd(hand_pose)
                case "Inspire":
                    self.send_inspire_hand_cmd(hand_pose)
            """
            if self.hand_type == "Dex-3":
                self.send_dex_hand_cmd(hand_pose)
            elif self.hand_type == "Inspire":
                self.send_inspire_hand_cmd(hand_pose)

        for j in range(self.num_dofs):
            if self._dof_idx is None:
                motor_idx = j
            else:
                motor_idx = self._dof_idx[j]
            command = commands[j]
            if j not in self._control_joint_idx:
                self.set_cmd_i(i=motor_idx, command=0, control_type=self._control_mode)
            else:
                self.set_cmd_i(
                    i=motor_idx, command=command, kp=self.kps[j], kd=self.kds[j], control_type=self._control_mode
                )

        self.send_cmd(self.low_cmd)


if __name__ == "__main__":
    from robojudo.config.g1.env.g1_real_env_cfg import G1RealEnvCfg

    env = UnitreeEnv(cfg_env=G1RealEnvCfg())
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
    print("Exit")
