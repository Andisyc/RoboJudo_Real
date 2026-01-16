from __future__ import annotations
from typing import Union, List, Optional
import logging
from abc import ABC, abstractmethod
from collections import deque

import numpy as np
import torch

from robojudo.tools.tool_cfgs import DoFConfig

from .policy_cfgs import PolicyCfg

logger = logging.getLogger(__name__)


class Policy(ABC):
    def __init__(self, cfg_policy: PolicyCfg, device: str = "cpu"):
        self.cfg_policy = cfg_policy
        self.device = device

        self.freq = self.cfg_policy.freq
        self.dt = 1.0 / self.freq

        self.cfg_obs_dof: DoFConfig = self.cfg_policy.obs_dof
        self.cfg_action_dof: DoFConfig = self.cfg_policy.action_dof

        self.num_dofs = self.cfg_obs_dof.num_dofs
        self.num_actions = self.cfg_action_dof.num_dofs

        self.default_dof_pos = np.asarray(self.cfg_obs_dof.default_pos)
        self.default_pos = np.asarray(self.cfg_action_dof.default_pos)  # TODO: remove

        # TODO: autoload cfg
        if self.cfg_policy.disable_autoload:
            # self.model: torch.nn.Module | None = None # type: ignore
            pass
        else:
            policy_file = self.cfg_policy.policy_file
            logger.debug(f"Loading jit from {policy_file}...")
            self.model = torch.jit.load(policy_file, map_location=self.device)

        self.action_scale = self.cfg_policy.action_scale
        self.action_clip = self.cfg_policy.action_clip
        self.action_beta = self.cfg_policy.action_beta

        # 修改前: 在 CPU 上计算
        self.last_action = np.zeros(self.num_actions)

        # 修改后: 直接在 GPU 上初始化
        # self.last_action = torch.zeros(self.num_actions, device=self.device, dtype=torch.float32)

        self.history_length = self.cfg_policy.history_length
        self.history_obs_size = self.cfg_policy.history_obs_size

    # py3.10 -> py3.8
    # def _init_history(self, default_history: np.ndarray | torch.Tensor | list): # py3.10
    def _init_history(self, default_history: Union[np.ndarray, torch.Tensor, list]): # py3.8
        logger.debug(f"Initializing history buffer as {self.history_length} x {len(default_history)}")
        self.history_buf = deque(maxlen=self.history_length)
        for _ in range(self.history_length):
            self.history_buf.append(default_history)

    @abstractmethod
    def reset(self):
        # self.last_action = np.zeros(self.num_actions) # TODO
        raise NotImplementedError

    @abstractmethod
    def post_step_callback(self, commands: list[str] | None = None):
        raise NotImplementedError

    @abstractmethod
    def get_observation(self, env_data, ctrl_data) -> tuple[np.ndarray, dict]:
        raise NotImplementedError

    def get_action(self, obs: np.ndarray) -> np.ndarray:
        obs_tensor = torch.from_numpy(obs).unsqueeze(0).float().to(self.device)
        with torch.no_grad():
            actions_tensor = self.model(obs_tensor).cpu()

        actions = actions_tensor.numpy().squeeze()
        actions = (1 - self.action_beta) * self.last_action + self.action_beta * actions

        self.last_action = actions.copy()  # TODO

        processed_actions = actions
        if self.action_clip is not None:
            processed_actions = np.clip(processed_actions, -self.action_clip, self.action_clip)

        processed_actions = processed_actions * self.action_scale
        return processed_actions
    
    """
    def get_action(self, obs: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        # 1. 处理输入：如果是 NumPy，转为 Tensor；如果是 Tensor，直接用
        if isinstance(obs, np.ndarray):
            obs_tensor = torch.from_numpy(obs).unsqueeze(0).float().to(self.device)
        else:
            # 已经是 Tensor (来自优化后的 PolicyWrapper)
            # 如果维度是 (Obs_Dim,)，增加一个 Batch 维度 -> (1, Obs_Dim)
            if obs.ndim == 1:
                obs_tensor = obs.unsqueeze(0)
            else:
                obs_tensor = obs

        # 2. 模型推理 (纯 GPU)
        with torch.no_grad():
            actions_tensor = self.model(obs_tensor)

        # 3. 后处理 (滤波 & 限幅) - 全程 GPU 计算
        # 移除 batch 维度: (1, Act_Dim) -> (Act_Dim,)
        actions = actions_tensor.squeeze(0)

        # 确保 last_action 也是 Tensor (防止 reset 中被重置为 numpy)
        if isinstance(self.last_action, np.ndarray):
             self.last_action = torch.from_numpy(self.last_action).float().to(self.device)

        # 低通滤波 (EMA)
        actions = (1 - self.action_beta) * self.last_action + self.action_beta * actions
        
        # 更新 last_action (保持在 GPU)
        self.last_action = actions.clone()

        processed_actions = actions
        
        # 限幅 (使用 torch.clamp 替代 np.clip)
        if self.action_clip is not None:
            processed_actions = torch.clamp(processed_actions, -self.action_clip, self.action_clip)

        processed_actions = processed_actions * self.action_scale
        
        # 4. 直接返回 Tensor
        # (PolicyWrapper 会在最后一步 get_pd_target 中统一转回 CPU)
        return processed_actions
    """
    def get_init_dof_pos(self) -> np.ndarray:
        """
        Return the initial dof pos for the policy, used for robot preparation.
        For motion policies, this should return next/first frame of the reference motion.
        """
        return self.default_pos.copy()

    def debug_viz(self, visualizer, env_data, ctrl_data, extras):
        # for debug draw
        return
