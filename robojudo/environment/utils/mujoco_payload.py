"""Optional rigid left-shoulder payload for G1 MuJoCo experiments.

Rigid-body combination follows TTT's G1RuntimePayload (Apache-2.0;
see LICENSE-TTT), with all derived mass properties cached at initialization.
"""
from __future__ import annotations

import logging

import mujoco
import numpy as np

logger = logging.getLogger(__name__)


class MujocoShoulderPayload:
    mass_kg = 3.0

    def __init__(self, model, data, *, enabled: bool):
        self.model = model
        self.data = data
        self.enabled = enabled
        self.active = False
        if not enabled:
            return
        self.body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "torso_link")
        if self.body_id <= 0:
            raise ValueError("left-shoulder payload requires the G1 torso_link body")
        self.position = np.array([0.004, 0.22, 0.15])
        self.half_size = np.array([0.08, 0.06, 0.12])
        self._baseline = (
            float(model.body_mass[self.body_id]), model.body_ipos[self.body_id].copy(),
            model.body_inertia[self.body_id].copy(), model.body_iquat[self.body_id].copy(),
        )
        mass, com, moments, quaternion = self._baseline
        axes = np.empty(9)
        mujoco.mju_quat2Mat(axes, quaternion)
        axes = axes.reshape(3, 3)
        inertia = (axes * moments) @ axes.T
        total_mass = mass + self.mass_kg
        displacement = self.position - com
        target_com = com + self.mass_kg / total_mass * displacement
        square = self.half_size ** 2
        box_inertia = np.diag(self.mass_kg / 3.0 * (square.sum() - square))
        target_inertia = inertia + box_inertia + mass * self.mass_kg / total_mass * (
            np.dot(displacement, displacement) * np.eye(3) - np.outer(displacement, displacement)
        )
        target_moments, target_axes = np.linalg.eigh(target_inertia)
        if np.linalg.det(target_axes) < 0:
            target_axes[:, 0] *= -1
        target_quaternion = np.empty(4)
        mujoco.mju_mat2Quat(target_quaternion, np.ascontiguousarray(target_axes).reshape(9))
        self._loaded = (total_mass, target_com, target_moments, target_quaternion)
        # mj_setConst may modify its MjData: never use the live rollout state.
        self._scratch = mujoco.MjData(model)

    def _set(self, active: bool):
        mass, com, moments, quaternion = self._loaded if active else self._baseline
        self.model.body_mass[self.body_id] = mass
        self.model.body_ipos[self.body_id] = com
        self.model.body_inertia[self.body_id] = moments
        self.model.body_iquat[self.body_id] = quaternion
        mujoco.mj_setConst(self.model, self._scratch)
        mujoco.mj_forward(self.model, self.data)
        self.active = active

    def before_step(self, velocity_command):
        if (self.enabled and not self.active and velocity_command is not None
                and velocity_command[0] > 0):
            self._set(True)
            logger.warning("[PAYLOAD] Left shoulder +3 kg attached on forward command")

    def reset(self):
        if self.active:
            self._set(False)

    def draw(self, viewer):
        if self.active:
            axes = self.data.xmat[self.body_id].reshape(3, 3)
            viewer.add_marker(
                pos=self.data.xpos[self.body_id] + axes @ self.position,
                mat=axes.copy(), size=self.half_size, rgba=[1.0, 0.35, 0.05, 1.0],
                type=mujoco.mjtGeom.mjGEOM_BOX, label="3 kg",
            )
