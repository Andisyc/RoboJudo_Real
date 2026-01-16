from __future__ import annotations
import logging

from box import Box

import robojudo.controller
from robojudo.controller import Controller, ControllerHook, CtrlCfg

logger = logging.getLogger(__name__)


class CtrlManager:
    # A manager that handles multiple controllers and their interactions.

    def __init__(
        self,
        cfg_ctrls: list[CtrlCfg] | None = None,
        env=None,
        device="cpu",
    ):
        self.cfg_ctrls = cfg_ctrls
        self.env = env
        self.device = device

        controllers = {}
        for cfg_ctrl in self.cfg_ctrls or []:
            ctrl_type = cfg_ctrl.ctrl_type
            if ctrl_type in controllers.keys():
                logger.warning(f"Controller type {ctrl_type} already exists, skipping.")
                continue

            ctrl_class: type[Controller] = getattr(robojudo.controller, ctrl_type)
            controller: Controller = ctrl_class(cfg_ctrl=cfg_ctrl, env=self.env, device=self.device)
            controllers[ctrl_type] = {
                "inst": controller,
                "cfg": cfg_ctrl,
                # "triggers": []
            }

        self.controllers = Box(controllers)

    def reset(self): # Reset all controllers.
        for controller in self.controllers.values():
            controller.inst.reset()

    def post_step_callback(self, ctrl_data: Box): # Call post step callback for all controllers.
        # self.process_triggers()
        commands = ctrl_data.get("COMMANDS", [])
        for controller in self.controllers.values():
            controller.inst.post_step_callback(commands)

    def get_ctrl_data(self, env_data):
        ctrl_data_all = {}
        ctrl_commands_all = set()
        for ctrl_type, controller in self.controllers.items():
            if isinstance(controller.inst, ControllerHook):
                ctrl_data = controller.inst.get_data_with_hook(prior_ctrl_data=ctrl_data_all, env_data=env_data)
            else:
                ctrl_data = controller.inst.get_data()
            ctrl_data_triggered, ctrl_commands = controller.inst.process_triggers(ctrl_data)

            ctrl_data_all[ctrl_type] = ctrl_data_triggered
            ctrl_commands_all.update(ctrl_commands)

        ctrl_data_all["COMMANDS"] = list(ctrl_commands_all)
        return Box(ctrl_data_all)

"""
class CtrlManager:
    # Optimized for Zero-Allocation loop.

    def __init__(
        self,
        cfg_ctrls: list[CtrlCfg] | None = None,
        env=None,
        device="cpu",
    ):
        # [原样]
        self.cfg_ctrls = cfg_ctrls
        self.env = env
        self.device = device

        # 使用原生字典，而非 Box
        self.controllers = {}
        
        # [优化] 预先将控制器展平为列表，避免在循环中反复调用 .items()
        self._active_controllers = []

        # [原样]
        for cfg_ctrl in self.cfg_ctrls or []:
            ctrl_type = cfg_ctrl.ctrl_type
            if ctrl_type in self.controllers:
                logger.warning(f"Controller type {ctrl_type} already exists, skipping.")
                continue

            ctrl_class: type[Controller] = getattr(robojudo.controller, ctrl_type)
            controller: Controller = ctrl_class(cfg_ctrl=cfg_ctrl, env=self.env, device=self.device)
            
            # [优化]
            ctrl_entry = {
                "inst": controller,
                "cfg": cfg_ctrl,}
            self.controllers[ctrl_type] = ctrl_entry
            
            # [优化] 缓存起来供 get_ctrl_data 快速遍历
            # 存储结构: (ctrl_type, controller_instance)
            self._active_controllers.append((ctrl_type, controller))

        # [优化] 预分配数据容器，避免在每一帧中创建
        self._cached_ctrl_data = {} 
        self._cached_command_set = set()

    def reset(self): # Reset all controllers.
        for entry in self.controllers.values():
            entry["inst"].reset()

    def post_step_callback(self, ctrl_data: dict): # Call post step callback for all controllers.
        commands = ctrl_data.get("COMMANDS", [])
        for entry in self.controllers.values():
            entry["inst"].post_step_callback(commands)

    def get_ctrl_data(self, env_data):
        # 1. 清空缓存 (比创建新字典快得多)
        self._cached_ctrl_data.clear()
        self._cached_command_set.clear()

        # 2. 遍历预生成的列表
        for ctrl_type, controller in self._active_controllers:
            # 获取数据
            if isinstance(controller, ControllerHook):
                # 传入当前的引用，允许 Hook 修改它
                ctrl_data = controller.get_data_with_hook(prior_ctrl_data=self._cached_ctrl_data, env_data=env_data)
            else:
                ctrl_data = controller.get_data()
            
            # 处理触发器
            ctrl_data_triggered, ctrl_commands = controller.process_triggers(ctrl_data)

            # 存入缓存
            self._cached_ctrl_data[ctrl_type] = ctrl_data_triggered
            
            # 更新指令集 (避免创建新 set)
            if ctrl_commands:
                self._cached_command_set.update(ctrl_commands)

        # 3. 将指令转为列表 (这是唯一不可避免的微小分配)
        # 如果下游只读，其实不需要转 list，但为了兼容性保留
        self._cached_ctrl_data["COMMANDS"] = list(self._cached_command_set)

        # 4. 直接返回原生字典 (移除 Box 包装)
        return self._cached_ctrl_data


if __name__ == "__main__":
    # Example usage
    from robojudo.config.g1.env.g1_dummy_env_cfg import G1DummyEnvCfg
    from robojudo.controller.ctrl_cfgs import JoystickCtrlCfg, KeyboardCtrlCfg
    from robojudo.environment.dummy_env import DummyEnv

    env = DummyEnv(cfg_env=G1DummyEnvCfg(forward_kinematic=None))
    cfg_ctrls = [KeyboardCtrlCfg(), JoystickCtrlCfg()]

    ctrl_manager = CtrlManager(cfg_ctrls=cfg_ctrls, env=env)  # pyright: ignore[reportArgumentType]
    ctrl_manager.reset()
    ctrl_data = ctrl_manager.get_ctrl_data(None)
    print(ctrl_data)
    # ctrl_manager.post_step_callback()
    print("Controller manager initialized and ready.")
"""