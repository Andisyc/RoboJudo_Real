"""Blend the final joint targets during the transfer from loaded native stance."""
import numpy as np
from scipy.spatial.transform import Rotation


class EntryTargetBlend:
    def __init__(self, names, policy_names, state, stiffness, torque_samples, duration_s):
        self.duration_s = duration_s
        self.policy_order = [names.index(name) for name in policy_names]
        self.ankles = [[names.index(f"{side}_ankle_{axis}_joint") for axis in ("roll", "pitch")]
                       for side in ("left", "right")]
        leg = np.array([any(joint in name for joint in ("hip", "knee", "ankle")) for name in names])
        self.anchor = np.array(state["dof_pos"], dtype=float, copy=True)
        kp = np.asarray(stiffness)[leg]
        samples = np.asarray(torque_samples)
        if (samples.ndim != 2 or samples.shape[1] != len(names) or len(samples) < 2
                or not np.isfinite(samples).all() or not np.isfinite(kp).all() or np.any(kp <= 0)):
            raise ValueError("Invalid native torque samples or leg stiffness for Cut3 entry")
        tau = np.median(samples[:, leg], axis=0)
        self.anchor[leg] += .25 * np.clip(tau / kp, -np.deg2rad(20), np.deg2rad(20))
        self.initial_rp = Rotation.from_quat(state["base_quat"]).as_euler("xyz")[:2]

    def target(self, raw_target, state, elapsed):
        u = np.clip(elapsed / self.duration_s, 0., 1.)
        weight = u**3 * (10 - 15*u + 6*u*u)
        target = self.anchor + weight * (raw_target - self.anchor)
        rp = Rotation.from_quat(state["base_quat"]).as_euler("xyz")[:2]
        correction = .6 * (rp - self.initial_rp + .15 * np.asarray(state["base_ang_vel"])[:2])
        target[self.ankles] += (1 - weight) * np.clip(correction, -.25, .25)
        return target
