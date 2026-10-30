import logging
from abc import ABC, abstractmethod

from robojudo.tools.debug_log import DebugLogger

from .pipeline_cfgs import PipelineCfg

logger = logging.getLogger(__name__)


class Pipeline(ABC):
    """
    Base Controller Module
    """

    def __init__(self, cfg: PipelineCfg):
        self.cfg = cfg

        if cfg.device == "auto":
            import torch

            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = cfg.device
        logger.info(f"Using device: {self.device}")

        if self.cfg.debug.log_obs:
            self.debug_logger = DebugLogger(run_cfg=cfg)

        self.dt = 1.0 / 50  # default
        self.timestep = 0
        self.do_safety_check = self.cfg.do_safety_check
        self._motion_trace = None
        self._motion_trace_phase = "initializing"

    def configure_motion_trace(self, directory, run_id):
        from pathlib import Path
        from robojudo.tools.motion_trace import MotionTrace
        from robojudo.config.g1.env.g1_env_cfg import G1_29DoF

        names = list(self.env.dof_cfg.joint_names)
        hardware_names = list(G1_29DoF().joint_names)
        motor_map = getattr(self.cfg.env, "joint2motor_idx", None)
        if motor_map is None:
            motor_map = [hardware_names.index(name) if name in hardware_names else None for name in names]
        self._motion_trace = MotionTrace(Path(directory)/"motion.msgpack", {
            "run_id": run_id, "config_name": type(self.cfg).__name__,
            "pipeline_type": type(self).__name__, "environment_joint_names": names,
            "joint_to_hardware_motor": motor_map,
            "hardware_joint_names": hardware_names,
        })
        manager = getattr(self, "policy_manager", None)
        policies = manager.policies if manager is not None else [self.policy]
        for wrapper in policies:
            policy = getattr(wrapper, "policy", wrapper)
            policy._motion_trace_enabled = True
        self.trace_event("lifecycle", phase="initialized")

    def trace_event(self, kind, **payload):
        trace = getattr(self, "_motion_trace", None)
        if trace is not None:
            trace.record_event(kind, **payload)

    def send_traced(self, target, *args, phase=None, commands=(), sender=None):
        """Observe the original send call; preserve its arguments, return and exception."""
        trace = getattr(self, "_motion_trace", None)
        callback = self.env.step if sender is None else sender
        if trace is None:
            return callback(target, *args)
        policy = self.policy
        if self._motion_trace_phase == "preparing":
            phase = "preparing"
        elif phase is None:
            phase = "playing"
            if getattr(policy, "flag_motion_done", False):
                phase = "done"
            elif getattr(policy, "play_speed", 1.) == 0:
                phase = "paused"
        cycle = trace.record_cycle(policy, target, phase, commands, self.timestep)
        try:
            result = callback(target, *args)
        except BaseException as exc:
            trace.record_event("send_result", cycle=cycle, status="raised", error=str(exc))
            raise
        trace.record_event("send_result", cycle=cycle, status="returned", result=result)
        return result

    def close_motion_trace(self):
        trace = getattr(self, "_motion_trace", None)
        if trace is not None:
            trace.record_event("lifecycle", phase="controller_exit")
            trace.close()

    @abstractmethod
    def step(self):
        raise NotImplementedError

    @abstractmethod
    def prepare(self):
        raise NotImplementedError

    def safety_check(self):
        return
