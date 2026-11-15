"""Remote held-key input; this controller never creates a robot environment."""
from __future__ import annotations

from robojudo_keyboard import KeyboardReceiver

from robojudo.controller import Controller, ctrl_registry
from robojudo.controller.ctrl_cfgs import RemoteKeyboardCtrlCfg


@ctrl_registry.register
class RemoteKeyboardCtrl(Controller):
    def __init__(self, cfg_ctrl: RemoteKeyboardCtrlCfg, env=None, device="cpu"):
        super().__init__(cfg_ctrl=cfg_ctrl, env=env, device=device)
        self.receiver = KeyboardReceiver(cfg_ctrl.port, cfg_ctrl.timeout_s)

    def get_data(self):
        return self.receiver.snapshot()

    def reset(self):
        self.receiver.reset()

    def close(self):
        self.receiver.close()

    def process_triggers(self, ctrl_data):
        return ctrl_data, ["[SHUTDOWN]"] if ctrl_data["shutdown"] else []

    def post_step_callback(self, commands=None):
        if commands and "[SHUTDOWN]" in commands:
            self.close()
