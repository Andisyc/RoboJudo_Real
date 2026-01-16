import numpy as np
import torch
from robojudo.environment.utils.mujoco_viz import MujocoVisualizer
from robojudo.policy import Policy, policy_registry
from robojudo.policy.policy_cfgs import UnitreePolicyCfg, UnitreeWoGaitPolicyCfg
from robojudo.utils.util_func import command_remap, get_gravity_orientation


@policy_registry.register
class UnitreePolicy(Policy):
    cfg_policy: UnitreePolicyCfg

    def __init__(self, cfg_policy, device):
        super().__init__(cfg_policy=cfg_policy, device=device)

        # load in obs scale coef
        self.obs_scales = self.cfg_policy.obs_scales

        # load in maximum cmd range
        self.max_cmd = self.cfg_policy.max_cmd

        # load in cmd mapping (joystick / keyboard)
        self.commands_map = self.cfg_policy.commands_map

        self.reset()

    def reset(self):
        self.timestep: int = 0

        self._init_history(np.zeros(self.history_obs_size))

    def post_step_callback(self, commands=None):
        self.timestep += 1

    def _get_phase(self):
        cycle_time = 0.8

        # compute current phase
        phase = self.timestep * self.dt / cycle_time
        return phase

    def _get_commands(self, ctrl_data): # process cmd input
        commands = np.zeros(3)
        for key in ctrl_data.keys():
            # process joystick input
            if key in ["JoystickCtrl", "UnitreeCtrl"]:
                axes = ctrl_data[key]["axes"]
                lx, ly, rx, ry = axes["LeftX"], axes["LeftY"], axes["RightX"], axes["RightY"]

                # mapping joystick ctrl to vel cmd
                commands[0] = command_remap(ly, self.commands_map[0])
                commands[1] = command_remap(lx, self.commands_map[1])
                commands[2] = command_remap(rx, self.commands_map[2])
                break
            
            # process keyboard input
            if key in ["KeyboardCtrl"]:
                keys = ctrl_data[key]["keyboard_event"]
                for event in keys:
                    if event["type"] == "keyboard":
                        value = event["pressed"] * 1.5
                        """
                        match event["name"]:
                            case "w":
                                commands[0] = command_remap(value, self.commands_map[0])
                            case "s":
                                commands[0] = command_remap(-value, self.commands_map[0])
                            case "a":
                                commands[1] = command_remap(-value, self.commands_map[1])
                            case "d":
                                commands[1] = command_remap(value, self.commands_map[1])
                            case "e":
                                commands[2] = command_remap(value, self.commands_map[2])
                            case "q":
                                commands[2] = command_remap(-value, self.commands_map[2])
                        """
                        if event["name"] == "w":
                            commands[0] = command_remap(value, self.commands_map[0])
                        elif event["name"] == "s":
                            commands[0] = command_remap(-value, self.commands_map[0])
                        elif event["name"] == "a":
                            commands[1] = command_remap(-value, self.commands_map[1])
                        elif event["name"] == "d":
                            commands[1] = command_remap(value, self.commands_map[1])
                        elif event["name"] == "e":
                            commands[2] = command_remap(value, self.commands_map[2])
                        elif event["name"] == "q":
                            commands[2] = command_remap(-value, self.commands_map[2])
                break
        return commands

    def get_observation(self, env_data, ctrl_data): # process obs input
        phase = self._get_phase()
        commands = self._get_commands(ctrl_data)

        # compute clock signal (raise / touch)
        sin_pos = [np.sin(2 * np.pi * phase)]
        cos_pos = [np.cos(2 * np.pi * phase)]

        # compute gravity vector projection
        gravity_orientation = get_gravity_orientation(env_data.base_quat)

        # concat all vector as obs
        obs = np.concatenate(
            [
                env_data.base_ang_vel * self.obs_scales.ang_vel,
                gravity_orientation,
                commands * self.obs_scales.command * self.max_cmd,
                env_data.dof_pos - self.default_dof_pos,
                env_data.dof_vel * self.obs_scales.dof_vel,
                self.last_action,
                sin_pos,
                cos_pos,
            ])
        
        # extras for Mujoco visualization
        extras = {"phase": phase, "commands": commands,}
        
        return obs, extras

    def debug_viz(self, visualizer: MujocoVisualizer, env_data, ctrl_data, extras):
        base_pos = env_data["base_pos"]
        base_quat = env_data["base_quat"]
        command_x = extras["commands"][0]
        command_y = extras["commands"][1]
        command_yaw = extras["commands"][2]

        visualizer.draw_arrow(
            base_pos,
            base_quat,
            [command_x, 0, 0],
            color=[1, 0, 0, 1],
            scale=2,
            horizontal_only=True,
            id=0,)
        
        visualizer.draw_arrow(
            base_pos,
            base_quat,
            [0, command_y, 0],
            color=[0, 1, 0, 1],
            scale=2,
            horizontal_only=True,
            id=1,)
        
        visualizer.draw_arrow(
            base_pos + np.array([0.0, 0, 0.6]),
            base_quat,
            [0, command_yaw, 0],
            color=[1, 1, 1, 1],
            scale=2,
            horizontal_only=True,
            id=2,)


@policy_registry.register
class UnitreeWoGaitPolicy(UnitreePolicy):
    cfg_policy: UnitreeWoGaitPolicyCfg

    def __init__(self, cfg_policy, device):
        super().__init__(cfg_policy=cfg_policy, device=device)

    def reset(self):
        self.timestep: int = 0

        # init history obs buffer
        history_obs_dims = self.cfg_policy.history_obs_dims
        default_history = [np.zeros(dim, dtype=np.float32) for dim in history_obs_dims.values()]
        self._init_history(default_history)

    def get_observation(self, env_data, ctrl_data):
        commands = self._get_commands(ctrl_data)

        # compute gravity vector projection
        gravity_orientation = get_gravity_orientation(env_data.base_quat)
        obs_current = [
            env_data.base_ang_vel * self.obs_scales.ang_vel,
            gravity_orientation * self.obs_scales.gravity,
            commands * self.obs_scales.command * self.max_cmd,
            (env_data.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos,
            env_data.dof_vel * self.obs_scales.dof_vel,
            self.last_action,]
        
        # put obs into history buffer
        self.history_buf.append(obs_current)

        # concat history obs as one vector
        # history_list = [np.concatenate(items, axis=0) for items in zip(*self.history_buf, strict=True)]
        history_list = [np.concatenate(items, axis=0) for items in zip(*self.history_buf)]
        obs = np.concatenate(history_list, axis=0)

        extras = {"commands": commands,}
        
        return obs, extras

"""
@policy_registry.register
class UnitreeWoGaitPolicy(UnitreePolicy):
    def __init__(self, cfg_policy, device):
        super().__init__(cfg_policy=cfg_policy, device=device)
        self.device = torch.device(device)
        
        # === [核心修复] 必须将 List 转为 GPU Tensor ===
        # 否则 Tensor * List 会导致 "only integer tensors..." 报错
        self.scale_ang_vel = torch.tensor(self.cfg_policy.obs_scales.ang_vel, device=self.device, dtype=torch.float32)
        self.scale_dof_pos = torch.tensor(self.cfg_policy.obs_scales.dof_pos, device=self.device, dtype=torch.float32)
        self.scale_dof_vel = torch.tensor(self.cfg_policy.obs_scales.dof_vel, device=self.device, dtype=torch.float32)
        self.scale_cmd = torch.tensor(self.cfg_policy.obs_scales.command, device=self.device, dtype=torch.float32)
        
        # max_cmd 也要转，防止它是 scalar 或 list 导致的广播错误
        self.max_cmd = torch.tensor(self.cfg_policy.max_cmd, device=self.device, dtype=torch.float32)
        
        # 预加载默认姿态
        self.default_dof_pos_tensor = torch.tensor(
            self.default_dof_pos, device=self.device, dtype=torch.float32)

    def reset(self):
        self.timestep = 0
        
        # === [核心优化] GPU Ring Buffer ===
        # 计算单帧 Observation 的总维度
        # 假设: ang_vel(3) + gravity(3) + cmd(3) + dof_pos(N) + dof_vel(N) + last_act(N)
        # 你需要根据 get_observation 的拼接逻辑精确计算这个维度
        # 这里用一种动态方式获取一次，然后固定
        self.history_len = 5 # 假设长度
        
        # 我们不再存储 list of arrays，而是存储 (History, Features) 
        # 但为了配合 zip(*buf) 那种 weird 的拼接逻辑 (Feature-wise History)，
        # 我们最好存储一个平铺的 Buffer
        
        # 初始化 buffer (全零)
        # 注意：你需要手动计算 raw_obs_dim
        # self.raw_obs_dim = 3 + 3 + 3 + num_dofs + num_dofs + num_dofs
        self.obs_history_buffer = None # 第一次运行时懒加载，或手动计算
        self.last_action = None

    def _get_commands_tensor(self, ctrl_data):
        # 简化版命令获取，不再遍历 keys
        # 假设 ctrl_data 是 dict
        cmd = torch.zeros(3, device=self.device)
        
        if "JoystickCtrl" in ctrl_data:
            axes = ctrl_data["JoystickCtrl"]["axes"]
            # 这里的 axes 可能是 float，直接构建
            # 建议优化 CtrlManager 让它直接返回 Tensor，或者在这里转
            # 暂时保持 CPU 读取，因为只有 4 个浮点数，开销可忽略
            lx = axes.get("LeftX", 0.0)
            ly = axes.get("LeftY", 0.0)
            rx = axes.get("RightX", 0.0)
            
            # 简单的映射逻辑，建议硬编码避免查表
            cmd[0] = -ly * 1.0 # 示例 map
            cmd[1] = -lx * 1.0
            cmd[2] = -rx * 1.0
            
        elif "KeyboardCtrl" in ctrl_data:
            # ... 保持原逻辑，但这部分主要用于调试，不用太优化 ...
            pass
            
        return cmd

    def get_observation(self, env_data, ctrl_data):
        # env_data 现在是包含 GPU Tensor 的字典

        def to_tensor(val):
            if isinstance(val, np.ndarray):
                return torch.from_numpy(val).float().to(self.device, non_blocking=True)
            return val
        
        # 从 env_data 提取并转换
        # 注意：这里需要配合之前我们在 __init__ 里生成的 indices
        # 如果 env_data 是 dict:
        dof_pos = to_tensor(env_data["dof_pos"])
        dof_vel = to_tensor(env_data["dof_vel"])
        base_ang_vel = to_tensor(env_data["base_ang_vel"])
        base_quat = to_tensor(env_data["base_quat"])

        # Indexing (如果需要)
        if hasattr(self, 'obs_dof_indices'):
            dof_pos = dof_pos[self.obs_dof_indices]
            dof_vel = dof_vel[self.obs_dof_indices]
        
        # 1. 准备数据 (全部在 GPU)
        if self.last_action is None:
            self.last_action = torch.zeros_like(dof_pos)

        base_ang_vel_scaled = base_ang_vel * self.scale_ang_vel
        
        # 重力投影 (手写 tensor 计算或用 helper)
        gravity_orientation = torch_get_gravity_orientation(base_quat)
        
        commands = self._get_commands_tensor(ctrl_data) * self.scale_cmd * self.max_cmd
        
        dof_pos_scaled = (dof_pos - self.default_dof_pos_tensor) * self.scale_dof_pos
        dof_vel_scaled = dof_vel * self.scale_dof_vel
        
        # 2. 拼接当前帧 (Raw Observation)
        current_obs_list = [
            base_ang_vel_scaled,
            gravity_orientation,
            commands,
            dof_pos_scaled,
            dof_vel_scaled,
            self.last_action]
        
        current_obs = torch.cat(current_obs_list, dim=0) # Shape: (Raw_Dim,)
        
        # 3. 初始化 Buffer (仅第一次)
        if self.obs_history_buffer is None:
            raw_dim = current_obs.shape[0]
            # Buffer Shape: (History_Len, Raw_Dim)
            self.obs_history_buffer = torch.zeros(
                (self.history_len, raw_dim), device=self.device, dtype=torch.float32)
            
            # 预计算切片索引 (用于模拟 zip(*buf) 的拼接效果)
            # 原逻辑是：[Feat1_T0...T5, Feat2_T0...T5, ...]
            # 我们记录每个特征的长度
            self.feature_lengths = [x.shape[0] for x in current_obs_list]
            
        # 4. 更新 Ring Buffer (GPU shift)
        # 向前滚动一行，丢弃最旧的 (Row 0)
        self.obs_history_buffer = torch.roll(self.obs_history_buffer, shifts=-1, dims=0)
        # 将当前帧填入最后一行
        self.obs_history_buffer[-1] = current_obs
        
        # 5. 重构输出 (Flatten 且按特征分组)
        # Buffer: [ [A0, B0], [A1, B1], ... ]
        # Target: [ A0, A1, ..., B0, B1, ... ]
        # 我们按列切片，然后 flatten
        
        final_parts = []
        start_col = 0
        for length in self.feature_lengths:
            end_col = start_col + length
            # 取出该特征的所有历史帧: (History, Length)
            feat_slice = self.obs_history_buffer[:, start_col:end_col]
            # Flatten -> (History * Length)
            final_parts.append(feat_slice.flatten()) 
            start_col = end_col
            
        obs_tensor = torch.cat(final_parts, dim=0)
        
        # 更新 last_action (来自外部输入还是这里？通常是上一帧网络输出)
        # 这里仅占位，实际上需要在 get_action 后更新
        
        return obs_tensor, {"commands": commands}


@torch.jit.script
def torch_get_gravity_orientation(quat):
    # quat: [x, y, z, w] or [w, x, y, z] -> 取决于你的数据源
    # Unitree SDK 通常是 [w, x, y, z] (scalar first)
    # 或者是 [x, y, z, w]。这里假设代码里对齐到了 [x, y, z, w]
    # 重力向量 g = [0, 0, -1]
    # 我们需要 R_inv * g，即 R 的第三行 (Row 2) 的负数
    
    x, y, z, w = quat[0], quat[1], quat[2], quat[3]
    
    # 旋转矩阵第三行 R[2, :]
    # r20 = 2 * (x * z + w * y)
    # r21 = 2 * (y * z - w * x)
    # r22 = 1 - 2 * (x * x + y * y)
    
    # 投影向量 = [r20, r21, r22] * -1 (因为是 [0,0,-1])
    # 注意：具体符号取决于坐标系定义，请参照原 np 版本核对
    # 这里给出一个标准实现
    
    gx = 2 * (w * y - z * x) # 注意正负号可能需要根据原版调整
    gy = -2 * (z * y + w * x)
    gz = 1 - 2 * (w * w + z * z)
    
    # 假设重力向下是 -z，且我们需要在机身系下的重力向量
    # return torch.stack([-gx, -gy, -gz]) 
    # 为了保险，直接用四元数旋转公式: q_inv * g * q
    
    return torch.stack([gx, gy, gz]) # 占位，请尽量复用原逻辑的 tensor 版
"""