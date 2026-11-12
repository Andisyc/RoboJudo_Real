"""Timed reference interpolation while the tracking actor remains closed-loop.

This is a tracking command, not a kinematically reconstructed full-body motion.
Only joint position/velocity and the consumed anchor orientation are overridden.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation, Slerp


class PoseReferenceTransition:
    def __init__(self, command, start_q, goal_q, start_anchor_wxyz,
                 goal_anchor_wxyz, anchor_index, duration_s, started_at):
        self.command = {k: np.asarray(v).copy() for k, v in command.items()}
        self.start_q = np.asarray(start_q, dtype=float).copy()
        self.goal_q = np.asarray(goal_q, dtype=float).copy()
        self.duration_s = float(duration_s)
        self.started_at = float(started_at)
        self.anchor_index = int(anchor_index)
        if (self.start_q.shape != self.goal_q.shape or self.start_q.ndim != 1
                or not np.isfinite(self.start_q).all() or not np.isfinite(self.goal_q).all()
                or not np.isfinite(self.duration_s) or self.duration_s <= 0):
            raise ValueError("Invalid pose transition endpoints or duration")
        quats = np.asarray([start_anchor_wxyz, goal_anchor_wxyz], dtype=float)
        self.rotations = Slerp([0., 1.], Rotation.from_quat(quats[:, [1, 2, 3, 0]]))

    def sample(self, now):
        u = float(np.clip((now - self.started_at) / self.duration_s, 0., 1.))
        # Roundoff near u=1 can otherwise put Slerp's argument just above 1.
        w = float(np.clip(u**3 * (10 - 15*u + 6*u*u), 0., 1.))
        rate = 30*u*u*(1-u)**2 / self.duration_s
        command = {k: v.copy() for k, v in self.command.items()}
        command["joint_pos"] = self.start_q + w * (self.goal_q - self.start_q)
        command["joint_vel"] = rate * (self.goal_q - self.start_q)
        command["body_quat_w"][self.anchor_index] = self.rotations(w).as_quat()[[3, 0, 1, 2]]
        return command, u >= 1.
