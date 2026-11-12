"""Offline only: no SDK construction, DDS, or mode requests to any robot."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace as NS
import unittest
from unittest.mock import Mock, patch
import numpy as np
from scipy.spatial.transform import Rotation

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('legacy',ROOT/'tests/test_mimic_safety_offline.py')
legacy=importlib.util.module_from_spec(spec);spec.loader.exec_module(legacy)
spec=importlib.util.spec_from_file_location('pose',ROOT/'robojudo/tools/pose_transition.py')
pose=importlib.util.module_from_spec(spec);spec.loader.exec_module(pose)
legacy.ns['Rotation']=Rotation;legacy.ns['PoseReferenceTransition']=pose.PoseReferenceTransition

class TestPoseTransition(unittest.TestCase):
    def make(self):
        p=legacy.make(policy_cls=legacy.StancePolicy)
        raw=p.policy.policy
        raw.command={'joint_pos':np.zeros(len(p.env.names)), 'joint_vel':np.ones(len(p.env.names)),
                     'body_quat_w':np.tile([1.,0.,0.,0.],(14,1)), 'body_pos_w':np.zeros((14,3)), 'time_step':0}
        raw.motion_anchor_body_index=7;raw.command_init_align=NS(R_base=Rotation.identity())
        raw.timestep=0;raw.play_speed=1.
        raw.cfg_policy=NS(start_timestep=0,max_timestep=100)
        p._standing_pose={n:float(i)*.005 for i,n in enumerate(p.env.names)}
        p.cfg.entry_pose_transition_s=p.cfg.exit_pose_transition_s=3.
        p.cfg.fixed_hip_bias=NS(enabled=False)
        raw.get_observation=Mock(side_effect=lambda d,c:(raw.command['joint_pos'].copy(),{'CALLBACK':[]}))
        raw.get_pd_target=Mock(side_effect=lambda obs:obs.copy())
        p.env.get_data=lambda:dict(dof_pos=p.env.q.copy(),dof_vel=p.env.velocity.copy(),
                                  base_quat=np.array([0.,0.,0.,1.]),torso_quat=np.array([0.,0.,0.,1.]))
        return p,raw

    def test_quintic_endpoint_velocity_and_midpoint(self):
        command={'joint_pos':np.zeros(2),'joint_vel':np.zeros(2),'body_quat_w':np.array([[1.,0,0,0]])}
        t=pose.PoseReferenceTransition(command,[0,0],[1,-2],[1,0,0,0],[1,0,0,0],0,3,10)
        a,done=t.sample(10);np.testing.assert_array_equal(a['joint_pos'],[0,0]);np.testing.assert_array_equal(a['joint_vel'],[0,0]);self.assertFalse(done)
        a,done=t.sample(11.5);np.testing.assert_allclose(a['joint_pos'],[.5,-1]);np.testing.assert_allclose(a['joint_vel'],[.625,-1.25])
        a,done=t.sample(13);np.testing.assert_array_equal(a['joint_pos'],[1,-2]);np.testing.assert_array_equal(a['joint_vel'],[0,0]);self.assertTrue(done)
        np.testing.assert_array_equal(command['joint_pos'],[0,0])
        for now in [13-1e-12,13+1e-12,20]:
            a,_=t.sample(now);self.assertTrue(np.isfinite(a['body_quat_w']).all())

    def test_entry_freezes_frame_and_maps_joints_by_name(self):
        p,raw=self.make();p.env.q=np.arange(len(p.env.names))*.01
        p._begin_pose_transition('entry',p.env.get_data())
        np.testing.assert_array_equal(raw.command['joint_pos'],p.env.q[::-1])
        p.state=legacy.State.MIMIC_ACTIVE
        raw.post_step_callback=Mock()
        p._step_once();raw.post_step_callback.assert_not_called()
        legacy.CLOCK[0]+=3.;p._step_once();self.assertIsNone(p._pose_transition);raw.post_step_callback.assert_not_called()
        p._step_once();raw.post_step_callback.assert_called_once()

    def test_exit_waits_three_seconds_before_handback_and_holds_goal(self):
        p,raw=self.make();p.state=legacy.State.MIMIC_ACTIVE;p.env.has_user_control=True;p.env.authority='USER_ACTIVE';p.env.fsm=1000
        p._begin_pose_transition('exit',p.env.get_data())
        p.env.release_rc=-20003 # failed handback retains active actor + final reference
        p._step_pose_transition(p.env.get_data(),{});self.assertEqual(p.env.releases,0)
        legacy.CLOCK[0]+=2.99;p._step_pose_transition(p.env.get_data(),{});self.assertEqual(p.env.releases,0)
        legacy.CLOCK[0]+=.02;p._step_pose_transition(p.env.get_data(),{});self.assertEqual(p.env.releases,0)
        p.env.q=np.array([p._standing_pose[n] for n in p.env.names])
        p._step_pose_settle(p.env.get_data(),{});self.assertEqual(p.env.releases,0)
        legacy.CLOCK[0]+=.31;p._step_pose_settle(p.env.get_data(),{});self.assertEqual(p.env.releases,1)
        goal=np.array([p._standing_pose[n] for n in raw.cfg_action_dof.joint_names])
        np.testing.assert_allclose(p._pose_hold_command['joint_pos'],goal)
        p.cfg.fixed_hip_bias=NS(enabled=False)
        raw.command['joint_pos'][:]=77
        p._step_mimic(p.env.get_data(),{});np.testing.assert_allclose(raw.get_observation.call_args[0][0]['dof_pos'],p.env.q[::-1])
        np.testing.assert_allclose(raw.get_pd_target.call_args.args[0],goal)

    def test_operator_stop_during_transition_freezes_current_reference(self):
        p,raw=self.make();p.state=legacy.State.MIMIC_ACTIVE;p.env.has_user_control=True;p.env.authority='USER_ACTIVE';p.env.fsm=1000;p.env.release_rc=-20003
        p._begin_pose_transition('entry',p.env.get_data());legacy.CLOCK[0]+=1.5
        expected,_=p._pose_transition.sample(legacy.CLOCK[0]);p._begin_handback('operator')
        self.assertIsNone(p._pose_transition)
        np.testing.assert_array_equal(p._pose_hold_command['joint_pos'],expected['joint_pos'])

    def test_invalid_duration_rejected(self):
        for duration in (0,-1,float('nan'),float('inf')):
            with self.assertRaises(ValueError):pose.PoseReferenceTransition({},[0],[1],[1,0,0,0],[1,0,0,0],0,duration,0)

    def test_exit_preserves_current_base_heading_not_old_torso_twist(self):
        p,raw=self.make()
        raw.command_init_align.R_base=Rotation.from_euler('z',-.7)
        raw.command['body_quat_w'][7]=Rotation.from_euler('z',1.).as_quat()[[3,0,1,2]]
        data=p.env.get_data();data['base_quat']=Rotation.from_euler('xyz',[.02,.04,.2]).as_quat()
        p._begin_pose_transition('exit',data)
        command,_=p._pose_transition.sample(legacy.CLOCK[0]+3.)
        actual=Rotation.from_quat(command['body_quat_w'][7][[1,2,3,0]])
        self.assertAlmostEqual((actual.inv()*Rotation.from_euler('z',-.5)).magnitude(),0.,places=7)

    def test_unreached_standing_pose_holds_without_mode_request(self):
        p,raw=self.make();p.state=legacy.State.MIMIC_ACTIVE
        p._begin_pose_transition('exit',p.env.get_data());legacy.CLOCK[0]+=3.01
        p._step_pose_transition(p.env.get_data(),{})
        p.env.q[:]=1.
        legacy.CLOCK[0]+=3.01;p._step_pose_settle(p.env.get_data(),{})
        self.assertTrue(p._hold);self.assertEqual(p.env.releases,0)
        self.assertIn('站姿恢复未完成',p.announcements[-1])

if __name__=='__main__':unittest.main()
