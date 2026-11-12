"""No SDK construction: exercise segment selection, alignment and stop semantics."""
import sys, unittest
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import Mock, patch
import numpy as np
from scipy.spatial.transform import Rotation as R
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from robojudo.pipeline import g1_baduanjin_suite_pipeline as mod
from robojudo.pipeline.g1_native_loco_mimic_pipeline import G1NativeLocoMimicPipeline, NativeLocoMimicState

NAMES=['left_hip_pitch_joint','right_hip_pitch_joint']
class Policy:
    def __init__(self,name,target):
        self.cfg_policy=NS(policy_name=name,freq=50,max_timestep=20)
        self.cfg_action_dof=NS(joint_names=NAMES,stiffness=[40,40],damping=[2,2])
        self.target=np.array(target);self.default_pos=np.zeros(2);self.action_scales=np.ones(2)
        self.reset()
    def reset(self):
        self.timestep=0;self.play_speed=1.;self.last_action=np.zeros(2)
        self.motion_anchor_body_index=0
        self.command={'joint_pos':np.array([.2,.4]),'joint_vel':np.zeros(2),'body_pos_w':np.array([[1.,2.,.8]]),
                      'body_quat_w':R.from_euler('z',.7).as_quat()[[3,0,1,2]][None,:]}
    def get_observation(self,data,ctrl):return data,{}
    def get_pd_target(self,obs):return self.target.copy()
    def post_step_callback(self,commands):self.timestep+=self.play_speed

class Manager:
    def __init__(self):self.policies=[Policy(str(i),[i,i+1]) for i in range(3)];self._current_policy_id=0;self.num_policies=3
    @property
    def current_policy_id(self):return self._current_policy_id
    @property
    def policy(self):return self.policies[self._current_policy_id]
    def policy_by_id(self,i):return self.policies[i]
    def set_policy(self,i):self._current_policy_id=i

class TestSuite(unittest.TestCase):
    def make(self):
        p=mod.G1BaduanjinSuitePipeline.__new__(mod.G1BaduanjinSuitePipeline)
        p.policy_manager=Manager();p._segment_blend=None;p._suite_cancelled=False;p._last_sent_target=np.array([0.,1.])
        p.cfg=NS(auto_sequence=True,segment_transition_s=3.)
        p.state=NativeLocoMimicState.MIMIC_ACTIVE;p._hold=False;p._handback_reason=None;p._stop_reason=None;p.timestep=0
        p._motion_trace=None;p._step_errors=0
        p._entry_blocked=False
        p.data={'base_quat':R.from_euler('z',.4).as_quat(),'torso_quat':R.from_euler('z',.4).as_quat(),
                'dof_pos':np.array([.1,.3]),'torso_pos':np.array([3.,4.,.8])}
        p.env=NS(joint_names=NAMES,update=Mock(),get_data=lambda:p.data,step=Mock(return_value=0),reset=Mock())
        p.ctrl_manager=NS(get_ctrl_data=lambda _: {'COMMANDS':[]},post_step_callback=Mock())
        p.trace_event=Mock();p._handle_commands=Mock()
        return p

    def test_compatible_gains_and_reject_mismatch_before_acquisition(self):
        p=self.make();mod.validate_blend_policies(p.policy_manager.policies,NAMES)
        p.policy_manager.policies[1].cfg_action_dof.stiffness[0]=60
        with self.assertRaises(ValueError):mod.validate_blend_policies(p.policy_manager.policies,NAMES)

    def test_transition_freezes_both_frames_and_never_resets_environment(self):
        p=self.make();clock=[0.]
        with patch.object(mod,'time',NS(monotonic=lambda:clock[0])):
            p.policy.timestep=20;p._begin_segment_blend(p.data);incoming=p.policy_manager.policy_by_id(1)
            p._step_once();np.testing.assert_array_equal(p.env.step.call_args.args[0],[0,1])
            clock[0]=1.5;p._step_once();np.testing.assert_allclose(p.env.step.call_args.args[0],[.5,1.5])
            self.assertEqual(p.policy.timestep,20);self.assertEqual(incoming.timestep,0)
            clock[0]=3;p._step_once();np.testing.assert_array_equal(p.env.step.call_args.args[0],[1,2])
            self.assertEqual(p.policy_manager.current_policy_id,1);self.assertEqual(incoming.timestep,0)
            self.assertIsNone(p._segment_blend);p.env.reset.assert_not_called()

    def test_incoming_alignment_matches_actual_heading(self):
        p=self.make();p._begin_segment_blend(p.data);inc=p.policy_manager.policy_by_id(1)
        q=inc.command_init_align.align_quat(inc.command['body_quat_w'][0][[1,2,3,0]])
        self.assertAlmostEqual(R.from_quat(q).as_euler('xyz')[2],.4)
        np.testing.assert_allclose(inc.command_init_align.align_pos(inc.command['body_pos_w'][0])[:2],[3,4])

    def test_stop_during_blend_cancels_remaining_sequence(self):
        p=self.make();clock=[0.]
        with patch.object(mod,'time',NS(monotonic=lambda:clock[0])):
            p._begin_segment_blend(p.data);p._operator_stop('[POLICY_LOCO]')
            self.assertTrue(p._suite_cancelled);self.assertIsNotNone(p._segment_blend)
            with patch.object(G1NativeLocoMimicPipeline,'_operator_stop') as stop:
                clock[0]=3.;p._step_once();stop.assert_called_once_with('[POLICY_LOCO]')

    def test_only_motion_done_advances_and_last_segment_uses_normal_handback(self):
        p=self.make();p._begin_segment_blend=Mock()
        with patch.object(G1NativeLocoMimicPipeline,'_advance_stop') as parent:
            p._advance_stop(p.data,{});parent.assert_called_once()
            p._advance_stop(p.data,{'CALLBACK':['[MOTION_DONE]']});p._begin_segment_blend.assert_called_once()
            p.policy_manager._current_policy_id=2;p._advance_stop(p.data,{'CALLBACK':['[MOTION_DONE]']})
            self.assertEqual(parent.call_count,2)
            p.policy_manager._current_policy_id=0;p.cfg.auto_sequence=False
            p._advance_stop(p.data,{'CALLBACK':['[MOTION_DONE]']});self.assertEqual(parent.call_count,3)

    def test_manual_stop_does_not_start_next_segment(self):
        p=self.make();p._stop_reason='operator';p._begin_segment_blend=Mock()
        with patch.object(G1NativeLocoMimicPipeline,'_advance_stop') as parent:
            p._advance_stop(p.data,{'CALLBACK':['[MOTION_DONE]']});parent.assert_called_once()
        p._begin_segment_blend.assert_not_called()

    def test_fault_clears_blend_and_never_starts_next_model(self):
        p=self.make();p._segment_blend={'incoming_id':1}
        with patch.object(G1NativeLocoMimicPipeline,'_request_handback') as parent:
            p._request_handback('error');parent.assert_called_once_with('error',None)
        self.assertIsNone(p._segment_blend);self.assertTrue(p._suite_cancelled)

    def test_new_start_restores_reference_alignment(self):
        p=self.make();p.state=NativeLocoMimicState.NATIVE_LOCO
        with patch.object(G1NativeLocoMimicPipeline,'_enter_mimic'):
            p._enter_mimic(p.data,{})
        aligned=p.policy.command_init_align.align_quat(p.policy.command['body_quat_w'][0][[1,2,3,0]])
        self.assertAlmostEqual(R.from_quat(aligned).as_euler('xyz')[2],0.)

    def test_every_suite_entry_starts_cut1_from_frame_zero(self):
        p=self.make();p.state=NativeLocoMimicState.NATIVE_LOCO
        for previous in (0,1,2):
            with self.subTest(previous=previous):
                p.policy_manager._current_policy_id=previous
                p.policy_manager.policies[0].timestep=20
                with patch.object(G1NativeLocoMimicPipeline,'_enter_mimic') as enter:
                    p._enter_mimic(p.data,{})
                    enter.assert_called_once()
                self.assertEqual(p.policy_manager.current_policy_id,0)
                self.assertEqual(p.policy.timestep,0)

    def test_single_segment_entry_keeps_selected_cut(self):
        p=self.make();p.state=NativeLocoMimicState.NATIVE_LOCO;p.cfg.auto_sequence=False
        for selected in (1,2):
            with self.subTest(selected=selected):
                p.policy_manager._current_policy_id=selected;p.policy.timestep=20
                with patch.object(G1NativeLocoMimicPipeline,'_enter_mimic'):
                    p._enter_mimic(p.data,{})
                self.assertEqual(p.policy_manager.current_policy_id,selected)
                self.assertEqual(p.policy.timestep,0)

    def test_active_or_blocked_entry_never_resets_cut(self):
        p=self.make();p.policy_manager._current_policy_id=2;p.policy.timestep=12
        for state,blocked in ((NativeLocoMimicState.MIMIC_ACTIVE,False),
                              (NativeLocoMimicState.HANDBACK,False),
                              (NativeLocoMimicState.NATIVE_LOCO,True)):
            with self.subTest(state=state,blocked=blocked):
                p.state=state;p._entry_blocked=blocked
                with patch.object(G1NativeLocoMimicPipeline,'_enter_mimic'):
                    p._enter_mimic(p.data,{})
                self.assertEqual(p.policy_manager.current_policy_id,2)
                self.assertEqual(p.policy.timestep,12)

    def test_entry_reset_failure_uses_existing_rejection_without_acquiring(self):
        p=self.make();p.state=NativeLocoMimicState.NATIVE_LOCO
        p.policy.reset=Mock(side_effect=ValueError('bad reference'));p._entry_failed=Mock()
        with patch.object(G1NativeLocoMimicPipeline,'_enter_mimic') as enter:
            p._enter_mimic(p.data,{})
            enter.assert_not_called();p._entry_failed.assert_called_once()

    def test_transition_preparation_failure_cancels_sequence_and_uses_outgoing_handback(self):
        p=self.make();p.policy_manager.policy_by_id(1).reset=Mock(side_effect=ValueError('bad reference'))
        p._begin_handback=Mock()
        p._advance_stop(p.data,{'CALLBACK':['[MOTION_DONE]']})
        p._begin_handback.assert_called_once_with('suite transition preparation failed')
        self.assertIsNone(p._segment_blend);self.assertTrue(p._suite_cancelled)
        self.assertEqual(p.policy_manager.current_policy_id,0)
        with patch.object(G1NativeLocoMimicPipeline,'_advance_stop') as parent:
            p._advance_stop(p.data,{'CALLBACK':['[MOTION_DONE]']});parent.assert_called_once()

if __name__=='__main__':unittest.main()
