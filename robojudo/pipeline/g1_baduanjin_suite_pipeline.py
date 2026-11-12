"""A persistent three-policy session with an opt-in closed-loop segment blend."""
import logging
import time
import numpy as np
from scipy.spatial.transform import Rotation

from robojudo.pipeline import pipeline_registry
from robojudo.pipeline.g1_native_loco_mimic_pipeline import G1NativeLocoMimicPipeline, NativeLocoMimicState
from robojudo.tools.pose_transition import PoseReferenceTransition
from robojudo.utils.rotation import TransformAlignment

logger = logging.getLogger(__name__)


def raw_policy(policy):
    return getattr(policy, '__dict__', {}).get('policy', policy)


def validate_blend_policies(policies, names):
    """Reject incompatible gains/ordering/frequency before taking control."""
    baseline = None
    for policy in policies:
        cfg = policy.cfg_action_dof
        order = [list(cfg.joint_names).index(n) for n in names]
        if len(cfg.joint_names) != len(names) or set(cfg.joint_names) != set(names):
            raise ValueError('Suite joint contract differs between policies')
        values = np.r_[np.asarray(cfg.stiffness)[order], np.asarray(cfg.damping)[order]]
        if baseline is not None and not np.allclose(values, baseline, atol=1e-4, rtol=1e-5):
            raise ValueError('Suite requires matching PD gains; no live gain interpolation')
        baseline = values
        if policy.cfg_policy.freq != policies[0].cfg_policy.freq:
            raise ValueError('Suite requires identical policy frequencies')
        if policy.cfg_policy.max_timestep <= 0:
            raise ValueError('Suite requires explicit finite motion lengths')


@pipeline_registry.register
class G1BaduanjinSuitePipeline(G1NativeLocoMimicPipeline):
    def __init__(self, cfg):
        self._segment_blend = None
        self._last_sent_target = None
        self._suite_cancelled = False
        super().__init__(cfg)
        validate_blend_policies(self.policy_manager.policies, self.env.joint_names)

    def send_traced(self, target, *args, **kwargs):
        result = super().send_traced(target, *args, **kwargs)
        self._last_sent_target = np.asarray(target).copy()
        return result

    def _enter_mimic(self, env_data, ctrl_data):
        if self.state == NativeLocoMimicState.NATIVE_LOCO:
            self._suite_cancelled = False
            # A previous continuous transition changed only this model's
            # alignment. A fresh Start must restore its original anchor first.
            try:
                self.policy.reset()
                raw = raw_policy(self.policy)
                anchor = raw.motion_anchor_body_index
                raw.command_init_align = TransformAlignment(
                    quat=raw.command['body_quat_w'][anchor][[1, 2, 3, 0]],
                    pos=raw.command['body_pos_w'][anchor].copy(), yaw_only=True, xy_only=True)
            except Exception as exc:
                self._entry_failed(exc)
                return None
        return super()._enter_mimic(env_data, ctrl_data)

    def _operator_stop(self, command):
        self._suite_cancelled = True
        if self._segment_blend is not None:
            # Complete this smooth transition, then use the ordinary stance-aware
            # stop path. Never force native mode halfway between two poses.
            self.trace_event('suite_stop_pending', command=command)
            return
        return super()._operator_stop(command)

    def _advance_stop(self, env_data, extras):
        if (self.cfg.auto_sequence and not self._suite_cancelled and not self._hold
                and self._handback_reason is None and self._stop_reason is None
                and '[MOTION_DONE]' in extras.get('CALLBACK', [])
                and self.policy_manager.current_policy_id + 1 < self.policy_manager.num_policies):
            try:
                self._begin_segment_blend(env_data)
            except Exception as exc:
                # The outgoing actor has completed normally and is still usable.
                # Cancel sequencing and use its original end-of-motion handback.
                # Do not repeatedly retry setup at the terminal frame: the base
                # step resets its inference-error count before _advance_stop().
                self._suite_cancelled = True
                self._segment_blend = None
                logger.exception("SUITE_TRANSITION_FAILED: cancelling remaining segments")
                self.trace_event('segment_transition', event='failed', reason=repr(exc))
                self._begin_handback('suite transition preparation failed')
            return
        return super()._advance_stop(env_data, extras)

    def _begin_segment_blend(self, env_data):
        manager = self.policy_manager
        incoming_id = manager.current_policy_id + 1
        incoming = manager.policy_by_id(incoming_id)
        raw = raw_policy(incoming)
        incoming.reset()
        anchor = raw.motion_anchor_body_index
        ref_q = raw.command['joint_pos'].copy()
        ref_anchor = raw.command['body_quat_w'][anchor].copy()
        ref_yaw = Rotation.from_quat(ref_anchor[[1, 2, 3, 0]]).as_euler('xyz')[2]
        actual_yaw = Rotation.from_quat(env_data['base_quat']).as_euler('xyz')[2]
        rot = Rotation.from_euler('z', ref_yaw-actual_yaw)
        origin = raw.command['body_pos_w'][anchor] - rot.apply(np.asarray(env_data['torso_pos']))
        # Keep the environment heading/odometry unchanged for the outgoing actor.
        # Align only the incoming reference to the current robot pose.
        raw.command_init_align = TransformAlignment(quat=rot.as_quat(), pos=origin, yaw_only=True, xy_only=True)
        order = [list(self.env.joint_names).index(n) for n in incoming.cfg_action_dof.joint_names]
        measured_q = np.asarray(env_data['dof_pos'])[order]
        start_anchor = (rot * Rotation.from_quat(env_data['torso_quat'])).as_quat()[[3, 0, 1, 2]]
        transition = PoseReferenceTransition(raw.command, measured_q, ref_q, start_anchor,
                    ref_anchor, anchor, self.cfg.segment_transition_s, time.monotonic())
        self._seed_history(incoming, self._last_sent_target)
        self._freeze_reference()
        self._segment_blend = dict(incoming=incoming, incoming_id=incoming_id, reference=transition)
        logger.warning("SUITE_TRANSITION: index %d -> %d (%s), %.1fs", manager.current_policy_id,
                       incoming_id, incoming.cfg_policy.policy_name, self.cfg.segment_transition_s)
        self.trace_event('segment_transition', event='begin', from_index=manager.current_policy_id,
                         to_index=incoming_id, duration_s=self.cfg.segment_transition_s,
                         model=incoming.cfg_policy.policy_name, environment_reset=False)

    def _seed_history(self, policy, sent):
        if sent is None:
            return
        raw = raw_policy(policy)
        order = [list(self.env.joint_names).index(n) for n in policy.cfg_action_dof.joint_names]
        raw.last_action = ((np.asarray(sent)[order]-raw.default_pos)/raw.action_scales).astype(np.float32)

    def _step_once(self):
        if self._segment_blend is None or self.state != NativeLocoMimicState.MIMIC_ACTIVE:
            return super()._step_once()
        self.env.update()
        data = self.env.get_data()
        ctrl = self.ctrl_manager.get_ctrl_data(data)
        commands = ctrl.get('COMMANDS', [])
        self._handle_commands(commands, data, ctrl)
        blend = self._segment_blend
        incoming = blend['incoming']
        transition = blend['reference']
        reference, done = transition.sample(time.monotonic())
        raw_policy(incoming).command = reference
        old_obs, _ = self.policy.get_observation(data, ctrl)
        old_target = self.policy.get_pd_target(old_obs)
        new_obs, _ = incoming.get_observation(data, ctrl)
        new_target = incoming.get_pd_target(new_obs)
        u = float(np.clip((time.monotonic()-transition.started_at)/transition.duration_s, 0., 1.))
        weight = float(np.clip(u**3*(10-15*u+6*u*u), 0., 1.))
        target = (1-weight)*old_target + weight*new_target
        self.send_traced(target, phase='segment_transition', commands=commands)
        self.trace_event('segment_blend_target', from_index=self.policy_manager.current_policy_id,
                         to_index=blend['incoming_id'], weight=weight,
                         outgoing_target=np.asarray(old_target).tolist(), incoming_target=np.asarray(new_target).tolist())
        self._seed_history(self.policy, target)
        self._seed_history(incoming, target)
        self.ctrl_manager.post_step_callback(ctrl)
        self._step_errors = 0
        self.timestep += 1
        if done:
            # Do not call set_policy(): it resets the shared environment heading.
            self.policy_manager._current_policy_id = blend['incoming_id']
            raw_policy(incoming).play_speed = 1.
            self._segment_blend = None
            logger.warning("SUITE_TRANSITION_COMPLETE: index %d, model %s", blend['incoming_id'],
                           incoming.cfg_policy.policy_name)
            self.trace_event('segment_transition', event='complete', to_index=blend['incoming_id'],
                             incoming_frame=int(incoming.timestep), environment_reset=False)
            if self._suite_cancelled:
                super()._operator_stop('[POLICY_LOCO]')

    def _request_handback(self, reason, text=None):
        self._suite_cancelled = True
        self._segment_blend = None
        return super()._request_handback(reason, text)
