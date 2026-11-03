"""G1 23/29DoF mimic safety checks. No SDK import, network, or actuator side effects."""
from __future__ import annotations
import logging
import threading
import numpy as np

logger = logging.getLogger(__name__)
SAFETY_REVISION = 2026110310
WARNING_TEXT = "朝向异常，请重新启动八段锦"
NOT_WALKRUN_TEXT = "请先切换到走跑模式，再启动八段锦"
STOPPED_TEXT = "八段锦已停止，交还走跑"
STOPPING_TEXT = "收到停止，回到站姿后交还走跑"
HOLD_TEXT = "交还走跑失败，正在保持姿态，请挂好吊机，用遥控器处理"
STANCE_TIMEOUT_TEXT = "未能回到站姿，正在保持姿态，请挂好吊机，用遥控器处理"
REGULAR_SKIPPED_TEXT = "进入常规运控不稳定，已停留在走跑，请用遥控器切换"
REGULAR_FAILED_TEXT = "未能进入常规运控，请用遥控器确认当前模式"
WALKRUN_SWITCH_TEXT = "正在切换到走跑，站稳后开始八段锦"
WALKRUN_FAILED_TEXT = "未能切换到走跑，请用遥控器确认当前模式"
WALKRUN_UNSTEADY_TEXT = "走跑没有站稳，请站稳后重新启动八段锦"
# Physical joint ranges from assets/robots/g1/g1_23dof_backpack.xml, radians.
JOINT_LIMITS = {'left_hip_pitch_joint': [-2.5307, 2.8798], 'left_hip_roll_joint': [-0.5236, 2.9671], 'left_hip_yaw_joint': [-2.7576, 2.7576], 'left_knee_joint': [-0.087267, 2.8798], 'left_ankle_pitch_joint': [-0.87267, 0.5236], 'left_ankle_roll_joint': [-0.2618, 0.2618], 'right_hip_pitch_joint': [-2.5307, 2.8798], 'right_hip_roll_joint': [-2.9671, 0.5236], 'right_hip_yaw_joint': [-2.7576, 2.7576], 'right_knee_joint': [-0.087267, 2.8798], 'right_ankle_pitch_joint': [-0.87267, 0.5236], 'right_ankle_roll_joint': [-0.2618, 0.2618], 'waist_yaw_joint': [-2.618, 2.618], 'left_shoulder_pitch_joint': [-3.0892, 2.6704], 'left_shoulder_roll_joint': [-1.5882, 2.2515], 'left_shoulder_yaw_joint': [-2.618, 2.618], 'left_elbow_joint': [-1.0472, 2.0944], 'left_wrist_roll_joint': [-1.97222, 1.97222], 'right_shoulder_pitch_joint': [-3.0892, 2.6704], 'right_shoulder_roll_joint': [-2.2515, 1.5882], 'right_shoulder_yaw_joint': [-2.618, 2.618], 'right_elbow_joint': [-1.0472, 2.0944], 'right_wrist_roll_joint': [-1.97222, 1.97222]}
# Extra 29DoF joints, from assets/robots/g1/g1_29dof_rev_1_0.xml, radians.
JOINT_LIMITS_29 = {**JOINT_LIMITS, 'waist_roll_joint': [-0.52, 0.52], 'waist_pitch_joint': [-0.52, 0.52],
                   'left_wrist_pitch_joint': [-1.61443, 1.61443], 'left_wrist_yaw_joint': [-1.61443, 1.61443],
                   'right_wrist_pitch_joint': [-1.61443, 1.61443], 'right_wrist_yaw_joint': [-1.61443, 1.61443]}
MAX_FIRST_DELTA = 0.8
MAX_ENTRY_STATE_AGE = 0.1

class SafetyViolation(RuntimeError):
    pass

class MimicSafetyGuard:
    """Pre-entry checks only. Never used to monitor or modify a running dance."""
    def __init__(self, joint_names):
        self.names = list(joint_names)
        if not any(len(self.names) == len(limits) and set(self.names) == set(limits)
                   for limits in (JOINT_LIMITS, JOINT_LIMITS_29)):
            raise ValueError("Entry check requires the exact G1 23DoF or 29DoF joint set")
        self.n = len(self.names)

    def reset(self):
        pass

    def state(self, data, state_age):
        if not np.isfinite(state_age) or not 0 <= state_age <= MAX_ENTRY_STATE_AGE:
            raise SafetyViolation(f"stale entry LowState: age={state_age}")
        for key, shape in (("dof_pos", (self.n,)), ("dof_vel", (self.n,)),
                           ("base_quat", (4,)), ("base_ang_vel", (3,)),
                           ("base_lin_vel", (3,))):
            value = np.asarray(data[key])
            if value.shape != shape or not np.isfinite(value).all():
                raise SafetyViolation(f"invalid entry state: {key}")
        q = np.asarray(data["base_quat"])
        if abs(np.linalg.norm(q) - 1.0) > .02:
            raise SafetyViolation("invalid entry orientation quaternion norm")
        q = q / np.linalg.norm(q)
        tilt = np.arccos(np.clip(1 - 2 * (q[0]**2 + q[1]**2), -1, 1))
        if tilt > np.deg2rad(10):
            raise SafetyViolation(f"entry tilt: {np.rad2deg(tilt):.2f} deg")
        if (np.max(np.abs(data["dof_vel"])) > .3
                or np.linalg.norm(data["base_ang_vel"]) > .15
                or np.linalg.norm(data["base_lin_vel"]) > .05):
            raise SafetyViolation("robot is not stationary before entry")

    def heading(self, data):
        x, y, z, w = np.asarray(data["base_quat"])
        yaw = np.arctan2(2*(w*z+x*y), 1-2*(y*y+z*z))
        if not np.isfinite(yaw) or abs(yaw) > np.deg2rad(5):
            raise SafetyViolation(f"heading reanchor failed: {np.rad2deg(yaw):.2f} deg")
        return float(yaw)

    def observation(self, observation):
        if not np.isfinite(np.asarray(observation)).all():
            raise SafetyViolation("nonfinite entry observation")

    def target(self, target, measured):
        q = np.asarray(target, dtype=float)
        reference = np.asarray(measured, dtype=float)
        if q.shape != (self.n,) or not np.isfinite(q).all():
            raise SafetyViolation("invalid entry target shape or NaN/Inf")
        if reference.shape != (self.n,) or not np.isfinite(reference).all():
            raise SafetyViolation("invalid entry measured position")
        i = int(np.argmax(np.abs(q-reference)))
        if abs(q[i]-reference[i]) > MAX_FIRST_DELTA:
            raise SafetyViolation(f"entry target jump: {self.names[i]} delta={q[i]-reference[i]:.6f}")
        return q.copy()

class SafetySpeaker:
    """One bounded asynchronous audio request; never wait on TTS in control code."""
    def __init__(self, speak):
        self.speak = speak
        self._lock = threading.Lock()
        self._thread = None

    def finish(self, timeout=1.2):
        if self._thread is not None:
            self._thread.join(timeout=timeout)

    def announce(self, text=WARNING_TEXT):
        if not self._lock.acquire(blocking=False):
            return False
        def run():
            try:
                result = self.speak(text)
                if result != 0:
                    logger.error("Safety speech failed: rc=%s; text=%s", result, text)
            except Exception:
                logger.exception("Safety speech failed; text=%s", text)
            finally:
                self._lock.release()
        self._thread = threading.Thread(target=run, name="g1-safety-speech", daemon=True)
        try:
            self._thread.start()
        except Exception:
            self._lock.release()
            logger.exception("Cannot start safety speech")
        return True
