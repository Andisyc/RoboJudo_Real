from __future__ import annotations
import logging

import torch
import numpy as np
import onnxruntime as ort

from robojudo.environment.utils.mujoco_viz import MujocoVisualizer
from robojudo.policy import Policy, policy_registry
from robojudo.policy.policy_cfgs import BeyondMimicPolicyCfg
from robojudo.tools.dof import DoFConfig
from robojudo.utils.progress import ProgressBar
from robojudo.utils.rotation import TransformAlignment
from robojudo.utils.util_func import matrix_from_quat, subtract_frame_transforms

logger = logging.getLogger(__name__)


@policy_registry.register
class BeyondMimicPolicy(Policy):
    cfg_policy: BeyondMimicPolicyCfg

    def __init__(self, cfg_policy: BeyondMimicPolicyCfg, device):
        # init onnx, override dof cfg if needed
        sess_options = ort.SessionOptions()

        # device select
        device = "cpu"
        if device == "cpu":
            providers = ["CPUExecutionProvider"]
        elif device == "cuda":
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        elif device == "tensorrt":
            # Jetson
            providers = [
                "TensorrtExecutionProvider",
                "CUDAExecutionProvider",
                "CPUExecutionProvider",]
        else:
            raise ValueError(f"Unknown device: {device}")

        # create inference instance (load policy to device)
        self.session = ort.InferenceSession(cfg_policy.policy_file, sess_options, providers=providers)
        self.input_names = [i.name for i in self.session.get_inputs()]
        self.output_names = [o.name for o in self.session.get_outputs()]
        self.motion_anchor_body_index = -1

        # read dof config from .onnx
        cfg_policy_new = cfg_policy.model_copy()
        if cfg_policy_new.use_modelmeta_config:
            logger.info("[BeyondMimicPolicy] Using modelmeta as config ...")
            modelmeta = self.session.get_modelmeta()  # all str,
            modelmeta_dict = modelmeta.custom_metadata_map

            # dict_keys(['joint_names', 'run_path', 'command_names', 'joint_stiffness', 'joint_damping',
            # 'default_joint_pos', 'action_scale', 'observation_names', 'anchor_body_name', 'body_names'])
            def parse_floats(s):
                return [float(item) for item in s.split(",")]

            def parse_strings(s):
                return [item for item in s.split(",")]

            # resolve dof name, default pos, Kp/Kd
            dof_config = DoFConfig(
                joint_names=parse_strings(modelmeta_dict["joint_names"]),
                default_pos=parse_floats(modelmeta_dict["default_joint_pos"]),
                stiffness=parse_floats(modelmeta_dict["joint_stiffness"]),
                damping=parse_floats(modelmeta_dict["joint_damping"]),)
            
            action_scales = parse_floats(modelmeta_dict["action_scale"])

            # resolve anchor body point
            anchor_body_name = modelmeta_dict["anchor_body_name"]
            body_names = parse_strings(modelmeta_dict["body_names"])
            self.motion_anchor_body_index = body_names.index(anchor_body_name)

            # command_names = parse_strings(modelmeta_dict["command_names"])
            # observation_names = parse_strings(modelmeta_dict["observation_names"])

            # update config for retargeting
            cfg_policy_new.action_dof = dof_config
            cfg_policy_new.obs_dof = dof_config
            cfg_policy_new.action_scales = action_scales

        # init body keypoint
        super().__init__(cfg_policy=cfg_policy_new, device=device)
        self.action_scales = np.asarray(self.cfg_policy.action_scales)

        # running with or without state estimator
        self.without_state_estimator = self.cfg_policy.without_state_estimator

        # running with or without init anchor joint pos
        self.override_robot_anchor_pos = self.cfg_policy.override_robot_anchor_pos

        # regressive (self generate next action) or teleoperate (follow outside cmd)
        self.use_motion_from_model = self.cfg_policy.use_motion_from_model

        self.max_timestep = self.cfg_policy.max_timestep
        self.command = None
        self.reset()

        if self.use_motion_from_model:
            assert self.motion_anchor_body_index >= 0, "motion_anchor_body_index not set"
            assert self.command is not None, "command not initialized"
            command_init = self.command.copy()

            # motion init2anchor alignment
            anchor_pos_w_init = command_init["body_pos_w"][self.motion_anchor_body_index, :]
            anchor_quat_w_init = command_init["body_quat_w"][self.motion_anchor_body_index, :][[1, 2, 3, 0]]

            self.command_init_align = TransformAlignment(
                quat=anchor_quat_w_init, pos=anchor_pos_w_init, yaw_only=True, xy_only=True)

    def _prepare_policy(self):
        obs_shape = self.session.get_inputs()[0].shape  # e.g. [1, 154]
        obs = np.zeros(obs_shape[1], dtype=np.float32)
        self.get_action(obs)

    def reset(self):
        self.timestep: float = self.cfg_policy.start_timestep
        if self.use_motion_from_model:
            self.pbar = ProgressBar(f"Beyondmimic {self.cfg_policy.policy_name}", self.max_timestep)
        else:
            self.pbar = None
        self.play_speed: float = 1.0
        self.flag_motion_done = False
        self._prepare_policy()

    def post_step_callback(self, commands: list[str] | None = None):
        # add time step
        self.timestep += 1 * self.play_speed

        # add progress bar
        if self.pbar:
            self.pbar.set(self.timestep)

        # examine action completion
        if 0 < self.max_timestep <= self.timestep:
            self.play_speed = 0.0
            self.flag_motion_done = True

        # process outside cmds
        for command in commands or []:
            """
            match command:
                case "[MOTION_RESET]":
                    self.reset()
                case "[MOTION_FADE_IN]":
                    self.play_speed = 1.0
                case "[MOTION_FADE_OUT]":
                    self.play_speed = 0.0
            """
            if command == "[MOTION_RESET]":
                self.reset()
            elif command ==  "[MOTION_FADE_IN]":
                self.play_speed = 1.0
            elif command ==  "[MOTION_FADE_OUT]":
                self.play_speed = 0.0

    def _get_command(self, env_data, ctrl_data):
        if not self.use_motion_from_model:
            assert "BeyondMimicCtrl" in ctrl_data, "BeyondMimicCtrl not found in ctrl_data"
            command = ctrl_data.get("BeyondMimicCtrl")
            self.command = command
            # print(command.time_steps[0])
            return (
                command.command,
                command.robot_anchor_pos_w,
                command.robot_anchor_quat_w,
                command.anchor_pos_w,
                command.anchor_quat_w,
                command.get("hand_pose", None),)
            
        else:
            assert self.command is not None, "command not initialized"
            # print(self.command["time_step"])
            command = np.concatenate([self.command["joint_pos"], self.command["joint_vel"]], axis=-1)

            anchor_pos_w = self.command["body_pos_w"][self.motion_anchor_body_index, :]
            anchor_quat_w = self.command["body_quat_w"][self.motion_anchor_body_index, :][[1, 2, 3, 0]]

            if self.command_init_align is not None:
                anchor_quat_w, anchor_pos_w = self.command_init_align.align_transform(anchor_quat_w, anchor_pos_w)

            if self.override_robot_anchor_pos:  # OVERRIDE
                robot_anchor_pos_w = anchor_pos_w.copy()
            else:
                base_pos = env_data.torso_pos
                robot_anchor_pos_w = base_pos

            robot_anchor_quat_w = env_data.torso_quat

            return command, robot_anchor_pos_w, robot_anchor_quat_w, anchor_pos_w, anchor_quat_w, None

    def get_observation(self, env_data, ctrl_data):
        # get proprioception
        dof_pos = env_data.dof_pos
        dof_vel = env_data.dof_vel
        ang_vel = env_data.base_ang_vel # root ang vel
        lin_vel = env_data.base_lin_vel # root lin vel

        # get command from ctrl
        command, robot_anchor_pos_w, robot_anchor_quat_w, anchor_pos_w, anchor_quat_w, hand_pose = self._get_command(
            env_data, ctrl_data)

        # subtract_frame_transforms compute 
        # relative transform between frames
        pos, ori = subtract_frame_transforms(
            robot_anchor_pos_w,
            robot_anchor_quat_w,
            anchor_pos_w,
            anchor_quat_w,)
        
        # matrix_from_quat transform pose 
        # to matrix for kinematic computation
        mat = matrix_from_quat(ori)

        obs_command = command
        obs_motion_anchor_pos_b = pos
        obs_motion_anchor_ori_b = mat[:, :2].flatten()

        # prepare obs (proprioception)
        obs_base_lin_vel = lin_vel
        obs_base_ang_vel = ang_vel
        obs_joint_pos_rel = dof_pos - self.default_dof_pos
        obs_joint_vel_rel = dof_vel
        obs_last_action = self.last_action

        # concat obs (proprioception)
        obs_prop = np.concatenate(
            [
                obs_command,
                obs_motion_anchor_pos_b if not self.without_state_estimator else [],
                obs_motion_anchor_ori_b,
                obs_base_lin_vel if not self.without_state_estimator else [],
                obs_base_ang_vel,
                obs_joint_pos_rel,
                obs_joint_vel_rel,
                obs_last_action,
            ])

        # ready to return
        obs = obs_prop # local frame (for policy)
        extras = { # world frame (for mujoco)
            "pos": pos, # displace vector
            "ori": ori, # quaternion
            "robot_anchor_pos_w": robot_anchor_pos_w, # mujoco arrow
            "robot_anchor_quat_w": robot_anchor_quat_w, # mujoco arrow
            "anchor_pos_w": anchor_pos_w, # ref motion world frame
            "anchor_quat_w": anchor_quat_w, # ref motion world frame
            "command": command, # original command
            "hand_pose": hand_pose,
            "CALLBACK": ["[MOTION_DONE]"] if self.flag_motion_done else [],}
        
        return obs, extras

    def get_action(self, obs: np.ndarray) -> np.ndarray:
        # init onnx input
        ort_inputs = {
            "obs": np.expand_dims(obs, axis=0).astype(np.float32),
            "time_step": np.expand_dims(np.array([int(self.timestep)]), axis=0).astype(np.float32),}
        
        # forward propagation
        ort_outputs = self.session.run(
            [
                "actions",
                "joint_pos",
                "joint_vel",
                "body_pos_w",
                "body_quat_w",
            ],
            ort_inputs,)
        
        actions: np.ndarray = np.asarray(ort_outputs[0]).squeeze()

        # low pass filter (soomth action)
        actions = (1 - self.action_beta) * self.last_action + self.action_beta * actions
        self.last_action = actions.copy()

        # resize action (regularization)
        scaled_actions = actions * self.action_scales

        # update regressive cmd
        if self.use_motion_from_model:
            self.command = {
                "time_step": self.timestep,
                "joint_pos": np.asarray(ort_outputs[1]).squeeze(),
                "joint_vel": np.asarray(ort_outputs[2]).squeeze(),
                "body_pos_w": np.asarray(ort_outputs[3]).squeeze(),
                "body_quat_w": np.asarray(ort_outputs[4]).squeeze(),}  # as [w, x, y, z]
            
        return scaled_actions

    def get_init_dof_pos(self) -> np.ndarray:
        """
        Return first frame of the reference motion.
        """
        if self.command is not None:
            joint_pos = self.command["joint_pos"]
            return joint_pos.copy()
        else:
            return self.default_dof_pos.copy()

    def debug_viz(self, visualizer: MujocoVisualizer, env_data, ctrl_data, extras):
        robot_anchor_pos_w = extras["robot_anchor_pos_w"]
        robot_anchor_quat_w = extras["robot_anchor_quat_w"]
        anchor_pos_w = extras["anchor_pos_w"]
        anchor_quat_w = extras["anchor_quat_w"]

        pos = extras["pos"]
        # ori = extras["ori"]

        visualizer.draw_arrow(anchor_pos_w, anchor_quat_w, [0.2, 0, 0], color=[1, 0, 0, 1], scale=2, id=0)
        visualizer.draw_arrow(
            robot_anchor_pos_w,
            robot_anchor_quat_w,
            [0.2, 0, 0],
            color=[0, 1, 0, 1],
            scale=2,
            id=1,)
        
        visualizer.draw_arrow(robot_anchor_pos_w, robot_anchor_quat_w, pos, color=[0, 1, 1, 1], scale=2, id=2)

        torso_pos = env_data["torso_pos"]
        torso_quat = env_data["torso_quat"]

        visualizer.draw_arrow(torso_pos, torso_quat, [0.2, 0, 0], color=[1, 1, 0, 1], scale=2, id=3)

"""
@policy_registry.register
class BeyondMimicPolicy(Policy):
    cfg_policy: BeyondMimicPolicyCfg

    def __init__(self, cfg_policy: BeyondMimicPolicyCfg, device):
        # 1. 保持原有的 ONNX 初始化逻辑
        sess_options = ort.SessionOptions()
        
        # 即使是 GPU 模式，ONNX Runtime 的 python api 输出通常还是 numpy
        # 所以我们暂时保持原样，或者根据你的环境调整 providers
        device_providers = "cpu" # 强制内部逻辑先用 CPU 跑 ONNX，避免 CUDA 冲突
        if device == "cpu":
            providers = ["CPUExecutionProvider"]
        elif device == "cuda":
            providers = [
                "CUDAExecutionProvider", 
                "CPUExecutionProvider"]
        elif device == "tensorrt":
            providers = [
                "TensorrtExecutionProvider", 
                "CUDAExecutionProvider", 
                "CPUExecutionProvider"]
        else:
            providers = ["CPUExecutionProvider"]

        # 创建推理实例, 将策略导入设备
        self.session = ort.InferenceSession(cfg_policy.policy_file, sess_options, providers=providers)
        self.input_names = [i.name for i in self.session.get_inputs()]
        self.output_names = [o.name for o in self.session.get_outputs()]
        self.motion_anchor_body_index = -1

        # ... (保留原有的 modelmeta 解析逻辑) ...
        # 从onnx文件中读取dof config
        cfg_policy_new = cfg_policy.model_copy()
        if cfg_policy_new.use_modelmeta_config:
            logger.info("[BeyondMimicPolicy] Using modelmeta as config ...")
            modelmeta = self.session.get_modelmeta()
            modelmeta_dict = modelmeta.custom_metadata_map

            def parse_floats(s):
                return [float(item) for item in s.split(",")]

            def parse_strings(s):
                return [item for item in s.split(",")]

            # 解析dof名称, 默认姿态, Kp Kd
            dof_config = DoFConfig(
                joint_names=parse_strings(modelmeta_dict["joint_names"]),
                default_pos=parse_floats(modelmeta_dict["default_joint_pos"]),
                stiffness=parse_floats(modelmeta_dict["joint_stiffness"]),
                damping=parse_floats(modelmeta_dict["joint_damping"]),)
            
            action_scales = parse_floats(modelmeta_dict["action_scale"])

            anchor_body_name = modelmeta_dict["anchor_body_name"]
            body_names = parse_strings(modelmeta_dict["body_names"])
            self.motion_anchor_body_index = body_names.index(anchor_body_name)

            cfg_policy_new.action_dof = dof_config
            cfg_policy_new.obs_dof = dof_config
            cfg_policy_new.action_scales = action_scales

        # 2. 初始化父类 (此时 self.last_action 变为 GPU Tensor)
        super().__init__(cfg_policy=cfg_policy_new, device=device)
        # ... (保留原有的 modelmeta 解析逻辑) ...

        self.device = torch.device(device)

        # 3. [关键修改] 将 action_scales 转为 Tensor，用于 get_action 中的 GPU 计算
        self.action_scales = torch.tensor(
            self.cfg_policy.action_scales, device=self.device, dtype=torch.float32)
        
        # 兼容性：为了 get_action 里的 numpy 转换，保留一个 cpu 版本（可选，如果逻辑需要）
        # self.action_scales_np = np.asarray(self.cfg_policy.action_scales)

        # ... (保留原有的 modelmeta 解析逻辑) ...
        # running with or without state estimator
        self.without_state_estimator = self.cfg_policy.without_state_estimator

        # running with or without init anchor joint pos
        self.override_robot_anchor_pos = self.cfg_policy.override_robot_anchor_pos

        # regressive (self generate next action) or teleoperate (follow outside cmd)
        self.use_motion_from_model = self.cfg_policy.use_motion_from_model

        self.max_timestep = self.cfg_policy.max_timestep
        self.command = None
        self.reset()

        if self.use_motion_from_model:
            assert self.motion_anchor_body_index >= 0, "motion_anchor_body_index not set"
            assert self.command is not None, "command not initialized"
            command_init = self.command.copy()

            # motion init2anchor alignment
            anchor_pos_w_init = command_init["body_pos_w"][self.motion_anchor_body_index, :]
            anchor_quat_w_init = command_init["body_quat_w"][self.motion_anchor_body_index, :][[1, 2, 3, 0]]
            
            self.command_init_align = TransformAlignment(
                quat=anchor_quat_w_init, pos=anchor_pos_w_init, yaw_only=True, xy_only=True)
        # ... (保留原有的 modelmeta 解析逻辑) ...

    def _prepare_policy(self):
        obs_shape = self.session.get_inputs()[0].shape
        # 构造 dummy obs (numpy)
        obs = np.zeros(obs_shape[1], dtype=np.float32)
        # 预热推理
        self.get_action(obs)

    def reset(self):
        self.timestep: float = self.cfg_policy.start_timestep
        if self.use_motion_from_model:
            self.pbar = ProgressBar(f"Beyondmimic {self.cfg_policy.policy_name}", self.max_timestep)
        else:
            self.pbar = None
        self.play_speed: float = 1.0
        self.flag_motion_done = False
        self._prepare_policy()

    def post_step_callback(self, commands: list[str] | None = None):
        # add time step
        self.timestep += 1 * self.play_speed

        # add progress bar
        if self.pbar:
            self.pbar.set(self.timestep)
        
        # examine action completion
        if 0 < self.max_timestep <= self.timestep:
            self.play_speed = 0.0
            self.flag_motion_done = True

        # process outside cmds
        for command in commands or []:
            if command == "[MOTION_RESET]":
                self.reset()
            elif command == "[MOTION_FADE_IN]":
                self.play_speed = 1.0
            elif command == "[MOTION_FADE_OUT]":
                self.play_speed = 0.0

    def _to_numpy(self, data):
        # 辅助函数: 将 Tensor 或 Numpy 转为 Numpy
        if isinstance(data, torch.Tensor):
            return data.detach().cpu().numpy()
        return data

    def _get_command(self, env_data, ctrl_data):
        # 这里的 env_data 可能包含 Tensor，需要转换
        if not self.use_motion_from_model:
            assert "BeyondMimicCtrl" in ctrl_data, "BeyondMimicCtrl not found in ctrl_data"
            command = ctrl_data.get("BeyondMimicCtrl")
            self.command = command
            return (
                command.command,
                command.robot_anchor_pos_w,
                command.robot_anchor_quat_w,
                command.anchor_pos_w,
                command.anchor_quat_w,
                command.get("hand_pose", None),)
            
        else:
            assert self.command is not None, "command not initialized"
            
            command = np.concatenate([self.command["joint_pos"], self.command["joint_vel"]], axis=-1)
            
            anchor_pos_w = self.command["body_pos_w"][self.motion_anchor_body_index, :]
            anchor_quat_w = self.command["body_quat_w"][self.motion_anchor_body_index, :][[1, 2, 3, 0]]

            if self.command_init_align is not None:
                anchor_quat_w, anchor_pos_w = self.command_init_align.align_transform(anchor_quat_w, anchor_pos_w)

            if self.override_robot_anchor_pos:
                robot_anchor_pos_w = anchor_pos_w.copy()
            else:
                # 兼容 Tensor/Numpy 且使用 key 访问
                if "torso_pos" in env_data:
                    base_pos = self._to_numpy(env_data["torso_pos"])
                else:
                    base_pos = self._to_numpy(env_data["base_pos"])
                robot_anchor_pos_w = base_pos

            # [修改点 2] 使用字典方式获取 torso_quat
            if "torso_quat" in env_data:
                robot_anchor_quat_w = self._to_numpy(env_data["torso_quat"])
            else:
                robot_anchor_quat_w = self._to_numpy(env_data["base_quat"])

            return command, robot_anchor_pos_w, robot_anchor_quat_w, anchor_pos_w, anchor_quat_w, None

    def get_observation(self, env_data, ctrl_data):
        # [修改点 3] 全部改为字典 ["key"] 访问
        dof_pos = env_data["dof_pos"]
        dof_vel = env_data["dof_vel"]
        ang_vel = env_data["base_ang_vel"]
        lin_vel = env_data["base_lin_vel"]

        # 处理可能的 Tensor (防止 PolicyWrapper 漏网之鱼)
        def ensure_numpy(val):
            if isinstance(val, torch.Tensor):
                return val.detach().cpu().numpy()
            return val
        
        dof_pos = ensure_numpy(dof_pos)
        dof_vel = ensure_numpy(dof_vel)
        ang_vel = ensure_numpy(ang_vel)

        # 2. 获取命令
        command, robot_anchor_pos_w, robot_anchor_quat_w, anchor_pos_w, anchor_quat_w, hand_pose = self._get_command(
            env_data, ctrl_data)

        # 3. 计算相对变换 (NumPy 计算)
        # subtract_frame_transforms compute 
        # relative transform between frames
        pos, ori = subtract_frame_transforms(
            robot_anchor_pos_w,
            robot_anchor_quat_w,
            anchor_pos_w,
            anchor_quat_w,)
        
        # matrix_from_quat transform pose 
        # to matrix for kinematic computation
        mat = matrix_from_quat(ori)

        obs_command = command
        obs_motion_anchor_pos_b = pos
        obs_motion_anchor_ori_b = mat[:, :2].flatten()

        # prepare obs (proprioception)
        obs_base_lin_vel = lin_vel
        obs_base_ang_vel = ang_vel
        obs_joint_pos_rel = dof_pos - self.default_dof_pos
        obs_joint_vel_rel = dof_vel
        
        # 注意：self.last_action 已经是 Tensor，这里为了拼接 obs，需要转 numpy
        obs_last_action = self._to_numpy(self.last_action)

        obs_prop = np.concatenate(
            [
                obs_command,
                obs_motion_anchor_pos_b if not self.without_state_estimator else [],
                obs_motion_anchor_ori_b,
                obs_base_lin_vel if not self.without_state_estimator else [],
                obs_base_ang_vel,
                obs_joint_pos_rel,
                obs_joint_vel_rel,
                obs_last_action,
            ])
        
        # ready to return
        obs = obs_prop
        extras = {
            "pos": pos,
            "ori": ori,
            "robot_anchor_pos_w": robot_anchor_pos_w,
            "robot_anchor_quat_w": robot_anchor_quat_w,
            "anchor_pos_w": anchor_pos_w,
            "anchor_quat_w": anchor_quat_w,
            "command": command,
            "hand_pose": hand_pose,
            "CALLBACK": ["[MOTION_DONE]"] if self.flag_motion_done else [],}
        
        # 返回 NumPy obs，因为 get_action 的 ONNX 需要 NumPy
        return obs, extras

    def get_action(self, obs: np.ndarray) -> torch.Tensor:
        # 1. ONNX 推理 (CPU/NumPy)
        ort_inputs = {
            "obs": np.expand_dims(obs, axis=0).astype(np.float32),
            "time_step": np.expand_dims(np.array([int(self.timestep)]), axis=0).astype(np.float32),}
        
        ort_outputs = self.session.run(
            [
                "actions",
                "joint_pos",
                "joint_vel",
                "body_pos_w",
                "body_quat_w",
            ],
            ort_inputs,)
        
        # actions 是 NumPy 数组
        actions_np: np.ndarray = np.asarray(ort_outputs[0]).squeeze()

        # 2. [关键修复] 立即转为 GPU Tensor
        actions_tensor = torch.from_numpy(actions_np).float().to(self.device)

        # 3. 滤波 (在 GPU 上进行)
        # self.last_action 是 Tensor，actions_tensor 是 Tensor
        filtered_actions = (1 - self.action_beta) * self.last_action + self.action_beta * actions_tensor
        
        # 4. 更新 last_action (使用 clone 替代 copy)
        self.last_action = filtered_actions.clone()

        # 5. Scale (Tensor * Tensor)
        scaled_actions = filtered_actions * self.action_scales

        # 6. 更新 regressive cmd (NumPy 逻辑保持不变)
        if self.use_motion_from_model:
            self.command = {
                "time_step": self.timestep,
                "joint_pos": np.asarray(ort_outputs[1]).squeeze(),
                "joint_vel": np.asarray(ort_outputs[2]).squeeze(),
                "body_pos_w": np.asarray(ort_outputs[3]).squeeze(),
                "body_quat_w": np.asarray(ort_outputs[4]).squeeze(),}

        # 7. 返回 Tensor，满足 PolicyWrapper 的要求
        return scaled_actions

    def get_init_dof_pos(self) -> np.ndarray:
        # 这个函数只需返回 numpy，PolicyWrapper 会处理
        if self.command is not None:
            joint_pos = self.command["joint_pos"]
            return joint_pos.copy()
        else:
            return self.default_dof_pos.copy()

    def debug_viz(self, visualizer: MujocoVisualizer, env_data, ctrl_data, extras):
        # 保持原样...
        pass
"""