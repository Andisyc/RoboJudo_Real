import mujoco
import numpy as np


class AnkleStance:
    """Horizontal ankle separation along and across the mean foot heading (metres)."""

    def __init__(self, xml_path, joint_names):
        self.model = mujoco.MjModel.from_xml_path(str(xml_path))
        self.data = mujoco.MjData(self.model)
        self.indices = [self.model.jnt_qposadr[self.model.joint(n).id] for n in joint_names]
        self.feet = [self.model.body(s + '_ankle_roll_link').id for s in ('left', 'right')]

    def measure(self, joint_pos, base_xyzw):
        self.data.qpos[self.indices] = joint_pos
        self.data.qpos[3:7] = np.asarray(base_xyzw)[[3, 0, 1, 2]]
        mujoco.mj_kinematics(self.model, self.data)
        axes = self.data.xmat[self.feet].reshape(2, 3, 3)[:, :2, 0]
        headings = axes / np.maximum(np.linalg.norm(axes, axis=1, keepdims=True), 1e-8)
        forward = headings.sum(axis=0)
        norm = np.linalg.norm(forward)
        if not np.isfinite(norm) or norm < 1e-6:
            return np.full(2, np.nan)
        forward /= norm
        delta = self.data.xpos[self.feet[0], :2] - self.data.xpos[self.feet[1], :2]
        return np.array([delta @ forward, delta @ [-forward[1], forward[0]]])
