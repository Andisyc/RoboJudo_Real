"""Run directly with Python + numpy. AST loads production classes without SDK imports."""
from pathlib import Path
import ast, importlib.util, logging, threading, time, unittest
from enum import Enum, auto
from types import SimpleNamespace as NS
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('mimic_safety', ROOT/'robojudo/tools/mimic_safety.py')
safety = importlib.util.module_from_spec(spec); spec.loader.exec_module(safety)
Guard, Violation = safety.MimicSafetyGuard, safety.SafetyViolation

def load_classes(path, wanted, namespace):
    tree = ast.parse(path.read_text())
    nodes = [n for n in tree.body if isinstance(n, ast.ClassDef) and n.name in wanted]
    for n in nodes: n.decorator_list = []
    exec(compile(ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[])), str(path), 'exec'), namespace)
    return namespace

class FakeBase:
    def trace_event(self, kind, **payload): self.events.append((kind, payload))
    def send_traced(self, target, *args, sender=None, **kwargs):
        return (sender or self.env.step)(target)

def gravity(q):
    x,y,z,w=q
    return np.array([0,0,-(1-2*(x*x+y*y))])

ns = dict(np=np, Enum=Enum, auto=auto, Pipeline=FakeBase,
          G1NativeLocoMimicPipelineCfg=object, logger=logging.getLogger('test'),
          MimicSafetyGuard=Guard, SafetySpeaker=safety.SafetySpeaker, SafetyViolation=Violation,
          WARNING_TEXT=safety.WARNING_TEXT, NOT_WALKRUN_TEXT=safety.NOT_WALKRUN_TEXT,
          STOPPED_TEXT=safety.STOPPED_TEXT, STOPPING_TEXT=safety.STOPPING_TEXT, HOLD_TEXT=safety.HOLD_TEXT,
          STANCE_TIMEOUT_TEXT=safety.STANCE_TIMEOUT_TEXT, REGULAR_SKIPPED_TEXT=safety.REGULAR_SKIPPED_TEXT,
          REGULAR_FAILED_TEXT=safety.REGULAR_FAILED_TEXT, WALKRUN_SWITCH_TEXT=safety.WALKRUN_SWITCH_TEXT,
          WALKRUN_FAILED_TEXT=safety.WALKRUN_FAILED_TEXT, WALKRUN_UNSTEADY_TEXT=safety.WALKRUN_UNSTEADY_TEXT,
          get_gravity_orientation=gravity, time=NS(sleep=lambda _:None, monotonic=lambda:CLOCK[0]))
CLOCK=[1000.]
load_classes(ROOT/'robojudo/pipeline/g1_native_loco_mimic_pipeline.py',
             ['NativeLocoMimicState','G1NativeLocoMimicPipeline'], ns)
P, State = ns['G1NativeLocoMimicPipeline'], ns['NativeLocoMimicState']

class Env:
    def __init__(self, yaw, names=None):
        self.names=list(names or safety.JOINT_LIMITS);n=len(self.names)
        self.yaw=yaw; self.reset_count=0; self.sent=[]; self.age=0; self.reset_works=True
        self.born_place_align=True; self.has_user_control=False; self.authority='INTERNAL'; self.fsm=801
        self.publish=False; self.trip_rc=0; self.trip_throws=False; self.acquire_rc=0; self.trips=0
        self.acquire_unconfirmed=False; self.release_rc=0; self.releases=0; self.regular_calls=0; self.closed=False
        self.walkrun_calls=0; self.walkrun_to=801
        self.q=np.zeros(n); self.q[[3,9]]=.3; self.velocity=np.zeros(n)
        self.base_quat=np.array([0,0,np.sin(yaw/2),np.cos(yaw/2)])
    def update(self): pass
    def get_data(self):
        return dict(dof_pos=self.q.copy(), dof_vel=self.velocity.copy(),base_quat=self.base_quat.copy(),
                    base_ang_vel=np.zeros(3), base_lin_vel=np.zeros(3))
    def safety_state_age(self): return self.age
    def reset(self):
        self.reset_count+=1
        if self.reset_works:self.base_quat=np.array([0,0,0,1.])
    def update_dof_cfg(self, **kwargs): pass
    def acquire_user_control(self,q):
        if self.acquire_unconfirmed:  # bridge keeps streaming the first target, FSM unknown
            self.sent.append(q.copy());self.has_user_control=True;self.authority='USER_ACTIVE';self.publish=True
            return -20007
        if self.acquire_rc:return self.acquire_rc
        if self.fsm not in (801,802):return -20006  # bridge acquires from WALKRUN only
        self.sent.append(q.copy()); self.has_user_control=True;self.authority='USER_ACTIVE';self.fsm=1000;self.publish=True
        return 0
    def get_control_status(self):
        return dict(authority_state=self.authority,fsm_id=self.fsm,publish_enabled=self.publish,active_publish_count=len(self.sent))
    def step(self,q):self.sent.append(q.copy()); return len(self.sent)
    def release_to_walkrun(self):
        self.releases+=1
        if self.release_rc:return self.release_rc  # bridge restored the hold stream
        self.has_user_control=False;self.authority='INTERNAL';self.fsm=801;self.publish=False;return 0
    def release_to_passive(self):raise AssertionError('PASSIVE must never be requested')
    def switch_walkrun_to_regular(self):
        self.regular_calls+=1
        if self.fsm!=801 or self.has_user_control:return -20006
        self.fsm=getattr(self,'regular_fsm',500);self.last_regular_fsm_trace=[(0.,0,801,0),(.62,1,self.fsm,0)];return 0
    def switch_regular_to_walkrun(self):  # walkrun_to: FSM the firmware ends in (1 = damp)
        self.walkrun_calls+=1
        if self.fsm not in (500,501) or self.has_user_control:return -20006
        self.last_walkrun_fsm_trace=[(0.,0,self.fsm,0),(.4,1,self.walkrun_to,0)];self.fsm=self.walkrun_to
        return 0 if self.fsm in (801,802) else -20003 if self.fsm in (500,501) else -20006
    def shutdown(self):self.publish=False;self.closed=True;return 0  # no mode request
    def sync_operator_fsm(self):  # query only; bridge drops the hold once the operator left FSM 1000
        self.syncs=getattr(self,'syncs',0)+1
        if self.has_user_control and self.fsm!=1000:
            self.has_user_control=False;self.authority='INTERNAL';self.publish=False
        return self.fsm

class Policy:
    cfg_action_dof=object()
    def __init__(self, env):self.env=env;self.observed=[];self.target=env.q.copy();self.inferences=0
    def reset(self):pass
    def get_observation(self,d,c):self.observed.append(d);return np.r_[d['base_quat'],d['dof_pos']],{}
    def get_pd_target(self,o):self.inferences+=1;return self.target.copy()
    def post_step_callback(self,c):pass

NAMES=list(safety.JOINT_LIMITS)
LEGS=[i for i,n in enumerate(NAMES) if any(k in n for k in ("hip","knee","ankle","waist"))]

class StancePolicy(Policy):
    """Inner BeyondMimic-like policy in MODEL joint order (reversed env order, like the
    real ONNX whose order differs from the motors). Reference starts at the measured
    stance; tests move it with self.ref_offset."""
    def __init__(self, env):
        super().__init__(env);self.cfg_policy=NS(start_timestep=0,max_timestep=100)
        self.model_names=env.names[::-1];self.cfg_action_dof=NS(joint_names=self.model_names)
        self.to_model=[env.names.index(n) for n in self.model_names]
        self.legs=[i for i,n in enumerate(self.model_names) if any(k in n for k in ("hip","knee","ankle","waist"))]
        self.target=env.q[self.to_model].copy()
        self.stance=env.q[self.to_model].copy();self.z=.78;self.ref_offset=0.;self.ref_vel=0.;self.play_speed=1.;self.frames=0
    def reset(self):self.play_speed=1.
    def reference_frame(self,t):return self.stance.copy(),np.zeros(len(self.stance)),self.z
    def current_reference(self):
        q=self.stance.copy();q[self.legs]+=self.ref_offset;return q,np.full(len(q),self.ref_vel),self.z
    def post_step_callback(self,c):self.frames+=self.play_speed

class Wrapper:
    """Same contract as robojudo.pipeline.rl_pipeline.PolicyWrapper: reorders dof
    env<->model and delegates attribute READS only (writes stay on the wrapper)."""
    def __init__(self, policy, env_names):
        self.policy=policy;self.to_env=[policy.model_names.index(n) for n in env_names]
    def get_observation(self,d,c):
        d=dict(d);d['dof_pos']=np.asarray(d['dof_pos'])[self.policy.to_model];d['dof_vel']=np.asarray(d['dof_vel'])[self.policy.to_model]
        return self.policy.get_observation(d,c)
    def get_pd_target(self,o):return np.asarray(self.policy.get_pd_target(o))[self.to_env]
    def __getattr__(self,name):return getattr(self.policy,name)

# G1_29DoF joint order (== motor order), as used by the 29DoF configuration.
NAMES_29=[*NAMES[:12],'waist_yaw_joint','waist_roll_joint','waist_pitch_joint',
          *[f'{s}_{j}_joint' for s in ('left','right') for j in ('shoulder_pitch','shoulder_roll','shoulder_yaw','elbow','wrist_roll','wrist_pitch','wrist_yaw')]]

def make(yaw=0, policy_cls=Policy, names=None):
    p=P.__new__(P);p.env=Env(yaw,names);p.env.joint_names=p.env.names;policy=policy_cls(p.env)
    if isinstance(policy,StancePolicy):policy=Wrapper(policy,p.env.names)
    p.policy_manager=NS(policy=policy)
    if not hasattr(policy, 'cfg_policy'):policy.cfg_policy=NS(start_timestep=0,max_timestep=100)
    policy.cfg_policy.policy_name='offline_test'
    if not hasattr(policy, 'timestep'):policy.timestep=0
    p.state=State.NATIVE_LOCO;p.should_stop=False;p.events=[];p.mimic_guard=Guard(p.env.names)
    p.announcements=[];p.safety_speaker=NS(announce=lambda:p.announcements.append(safety.WARNING_TEXT),finish=lambda:None)
    p._entry_blocked=False;p.do_safety_check=True;p.timestep=0;p.cfg=NS(debug=NS(log_obs=False),return_to_regular_loco=True,walkrun_on_start=True)
    p.cfg.fixed_hip_bias=NS(enabled=False,policy_name='disabled',model_dump=lambda:dict(enabled=False))
    p.cfg.entry_pose_transition_s=0.;p.cfg.exit_pose_transition_s=0.;p.cfg.env=NS(unitree=NS(regular_fsm_id=500))
    p._pose_transition=None;p._pose_hold_command=None
    p._step_errors=0;p._handback_next_try=0.;p._regular_deadline=None;p._regular_still_since=None
    p._walkrun_entry_deadline=None;p._walkrun_still_since=None
    p._stance_cache={};p._clear_stop_flags()
    p.safety_speaker=NS(announce=lambda text=safety.WARNING_TEXT:p.announcements.append(text),finish=lambda:None)
    p.ctrl_manager=NS(get_ctrl_data=lambda d:{'COMMANDS':[]},post_step_callback=lambda d:None)
    return p

def settle(p):
    """Legs still for REGULAR_STILL_S after the WALKRUN return."""
    p.step();CLOCK[0]+=p.REGULAR_STILL_S+.01;p.step()

class TestSafety(unittest.TestCase):
    def test_all_entry_headings_use_new_data(self):
        for deg in [0,90,-90,175,180,-180,38.7]:
            with self.subTest(deg=deg):
                p=make(np.deg2rad(deg));old=p.env.get_data();p._enter_mimic(old,{})
                self.assertEqual(p.state,State.MIMIC_ACTIVE);self.assertEqual(len(p.env.sent),1)
                np.testing.assert_allclose(p.policy.observed[0]['base_quat'],[0,0,0,1]);self.assertEqual(p.env.reset_count,1)
    def test_reentry_after_native_turn(self):
        p=make();p._enter_mimic(p.env.get_data(),{});p._return_to_native_loco()
        p.env.base_quat=np.array([0,0,1.,0]);p._enter_mimic(p.env.get_data(),{})
        self.assertEqual(len(p.env.sent),2);np.testing.assert_allclose(p.policy.observed[-1]['base_quat'],[0,0,0,1])
    def test_failed_reanchor_rejects_before_acquire_and_speaks(self):
        p=make(np.pi);p.env.reset_works=False;p._enter_mimic(p.env.get_data(),{})
        self.assertEqual(p.env.sent,[]);self.assertEqual(p.state,State.NATIVE_LOCO);self.assertEqual(p.announcements,[safety.WARNING_TEXT])
        p._enter_mimic(p.env.get_data(),{});self.assertEqual(len(p.announcements),1)
        p._handle_commands([],p.env.get_data(),{});p.env.reset_works=True;p._enter_mimic(p.env.get_data(),{});self.assertEqual(len(p.env.sent),1)
    def test_preentry_stale_moving_nan_disabled(self):
        for mode in ['stale','moving','nan','disabled']:
            with self.subTest(mode=mode):
                p=make()
                if mode=='stale':p.env.age=.11
                if mode=='moving':p.env.velocity[0]=1
                if mode=='nan':p.env.q[0]=np.nan
                if mode=='disabled':p.env.born_place_align=False
                p._enter_mimic(p.env.get_data(),{});self.assertEqual(p.env.sent,[]);self.assertEqual(p.env.trips,0)
    def test_first_target_faults(self):
        for kind in ['nan','inf','range','jump','shape']:
            with self.subTest(kind=kind):
                p=make()
                if kind=='nan':p.policy.target[0]=np.nan
                if kind=='inf':p.policy.target[0]=np.inf
                if kind=='range':p.policy.target[3]=12.57
                if kind=='jump':p.policy.target[0]=1
                if kind=='shape':p.policy.target=np.zeros(22)
                p._enter_mimic(p.env.get_data(),{});self.assertEqual(p.env.sent,[]);self.assertEqual(len(p.announcements),1)
    def test_active_path_does_not_call_any_entry_guard(self):
        p=make();p._enter_mimic(p.env.get_data(),{})
        def forbidden(*args,**kwargs):raise AssertionError('entry guard called during dance')
        for name in ['state','observation','target','heading']:setattr(p.mimic_guard,name,forbidden)
        p.env.age=10.;p.env.base_quat=np.array([np.sin(.7),0,0,np.cos(.7)])
        for value in [.9,0,.9,0,.9,0,1.]:
            p.policy.target[0]=value;p.step()
            self.assertEqual(p.state,State.MIMIC_ACTIVE)
        self.assertEqual(len(p.env.sent),8);self.assertEqual(p.announcements,[]);self.assertEqual(p.env.fsm,1000)
    def test_close_in_native_requests_nothing_and_during_dance_hands_back_first(self):
        p=make();p.close();self.assertEqual(p.env.fsm,801);self.assertEqual(p.env.releases,0);self.assertTrue(p.env.closed)
        p=make();p._enter_mimic(p.env.get_data(),{});p.close()
        self.assertEqual(p.env.fsm,801);self.assertEqual(p.env.releases,1);self.assertTrue(p.env.closed)
    def test_close_deferred_while_walkrun_unconfirmed(self):
        p=make();p._enter_mimic(p.env.get_data(),{});p.env.release_rc=-20003
        self.assertNotEqual(p.close(),0);self.assertFalse(p.env.closed);self.assertTrue(p.env.publish)
        p.env.release_rc=0;CLOCK[0]+=p.HANDBACK_RETRY_S+.01
        self.assertEqual(p.close(),0);self.assertEqual(p.env.fsm,801);self.assertTrue(p.env.closed)
    def test_single_step_error_keeps_dancing(self):
        p=make();p._enter_mimic(p.env.get_data(),{})
        real=p.policy.get_pd_target;fails=[1]
        def flaky(obs):
            if fails:fails.pop();raise RuntimeError('transient inference failure')
            return real(obs)
        p.policy.get_pd_target=flaky;p.step();p.step()
        self.assertEqual(p.state,State.MIMIC_ACTIVE);self.assertFalse(p.should_stop)
        self.assertEqual(p._step_errors,0);self.assertEqual(p.env.fsm,1000);self.assertEqual(p.env.releases,0)
    def test_persistent_step_errors_hand_back_to_walkrun_not_stop(self):
        p=make();p._enter_mimic(p.env.get_data(),{})
        def failed_inference(obs):raise RuntimeError('ordinary process failure')
        p.policy.get_pd_target=failed_inference
        for _ in range(p.MAX_CONSECUTIVE_STEP_ERRORS-1):
            p.step();self.assertEqual(p.state,State.MIMIC_ACTIVE);self.assertTrue(p.env.publish)
        p.step()
        self.assertEqual(p.state,State.NATIVE_LOCO);self.assertFalse(p.should_stop)
        self.assertEqual(p.env.fsm,801);self.assertFalse(p.env.closed)
    def test_shutdown_button_during_dance_hands_back_and_keeps_process(self):
        p=make();p._enter_mimic(p.env.get_data(),{})
        p.ctrl_manager.get_ctrl_data=lambda d:{'COMMANDS':['[SHUTDOWN]']};p.step()
        self.assertEqual(p.state,State.NATIVE_LOCO);self.assertFalse(p.should_stop);self.assertFalse(p.env.closed)
        self.assertEqual(p.env.fsm,801);self.assertIn(safety.STOPPED_TEXT,p.announcements)
        p.step();self.assertEqual(p.state,State.CLOSED);self.assertTrue(p.should_stop);self.assertEqual(p.env.fsm,801)
    def test_walkrun_failure_holds_and_retries(self):
        p=make();p._enter_mimic(p.env.get_data(),{});p.env.release_rc=-20003
        p._request_handback('test');self.assertEqual(p.state,State.HANDBACK);self.assertTrue(p.env.publish)
        n=len(p.env.sent);p.step();self.assertEqual(p.env.releases,1);self.assertEqual(len(p.env.sent),n)
        p.env.release_rc=0;CLOCK[0]+=p.HANDBACK_RETRY_S+.01;p.step()
        self.assertEqual(p.state,State.NATIVE_LOCO);self.assertEqual(p.env.fsm,801);self.assertFalse(p.should_stop)
    def test_unconfirmed_entry_hands_back_instead_of_stopping(self):
        p=make();p.env.acquire_unconfirmed=True;p._enter_mimic(p.env.get_data(),{})
        self.assertEqual(p.state,State.NATIVE_LOCO);self.assertEqual(p.env.fsm,801);self.assertFalse(p.should_stop)
    def test_regular_loco_after_stationary_walkrun(self):
        p=make();p._enter_mimic(p.env.get_data(),{});p._return_to_native_loco('motion done')
        p.step();CLOCK[0]+=p.REGULAR_STILL_S-.01;p.step();self.assertEqual(p.env.fsm,801)
        CLOCK[0]+=.02;p.step();self.assertEqual(p.env.fsm,500);self.assertEqual(p.env.regular_calls,1)
        p.step();self.assertEqual(p.env.regular_calls,1)
    def test_regular_loco_waits_until_legs_still(self):
        # Field log 2026-11-03: requested while WALKRUN was still settling -> ignored.
        p=make();p._enter_mimic(p.env.get_data(),{});p._return_to_native_loco('motion done')
        knee=NAMES.index('left_knee_joint');arm=NAMES.index('left_shoulder_pitch_joint')
        # .25: above REGULAR_STILL_VEL, below the 0.3 rad/s guard limit (arms: guard only)
        self.assertLess(p.REGULAR_STILL_VEL,.25);p.env.velocity[knee]=.25;p.env.velocity[arm]=.25
        for _ in range(3):p.step();CLOCK[0]+=1.
        self.assertEqual(p.env.regular_calls,0)
        p.env.velocity[knee]=0;p.step();CLOCK[0]+=.5;p.step();self.assertEqual(p.env.regular_calls,0)
        p.env.velocity[knee]=.25;p.step();p.env.velocity[knee]=0  # timer restarts
        p.step();CLOCK[0]+=.6;p.step();self.assertEqual(p.env.regular_calls,0)
        CLOCK[0]+=.45;p.step();self.assertEqual(p.env.regular_calls,1);self.assertEqual(p.env.fsm,500)
    def test_regular_loco_501_counts_and_trace_is_logged(self):
        p=make();p.env.regular_fsm=501;p._enter_mimic(p.env.get_data(),{});p._return_to_native_loco('motion done')
        settle(p)
        self.assertEqual(p.env.fsm,501);self.assertEqual(p.announcements,[])
        ev=[e for k,e in p.events if k=='regular_loco'][-1]
        self.assertEqual(ev['result'],0);self.assertEqual(ev['fsm_trace'],[(0.,0,801,0),(.62,1,501,0)])
    def test_regular_loco_waits_for_stationary_then_gives_up(self):
        p=make();p._enter_mimic(p.env.get_data(),{});p._return_to_native_loco('motion done')
        p.env.velocity[0]=1
        for _ in range(30):p.step()
        CLOCK[0]+=p.REGULAR_WAIT_S+.01;p.step();p.env.velocity[0]=0
        for _ in range(30):p.step()
        self.assertEqual(p.env.regular_calls,0);self.assertEqual(p.env.fsm,801)
    def test_regular_loco_disabled_by_config(self):
        p=make();p.cfg.return_to_regular_loco=False;p._enter_mimic(p.env.get_data(),{});p._return_to_native_loco('done')
        for _ in range(30):p.step()
        self.assertEqual(p.env.regular_calls,0)
    def test_start_from_regular_loco_is_rejected_with_walkrun_prompt(self):
        p=make();p.env.fsm=500;p.env.acquire_rc=-20006;p._enter_mimic(p.env.get_data(),{})
        self.assertEqual(p.state,State.NATIVE_LOCO);self.assertFalse(p.should_stop)
        self.assertEqual(p.announcements,[safety.NOT_WALKRUN_TEXT])
    def hold_start(self,p,frames=1):
        p.ctrl_manager.get_ctrl_data=lambda d:{'COMMANDS':['[POLICY_MIMIC]']}
        for _ in range(frames):p.step()
        p.ctrl_manager.get_ctrl_data=lambda d:{'COMMANDS':[]}
    def test_start_in_regular_loco_switches_to_walkrun_then_enters_when_still(self):
        for fsm in (500,501):
            p=make();p.env.fsm=fsm;self.hold_start(p,frames=5)  # held: one request only
            self.assertEqual(p.env.walkrun_calls,1);self.assertEqual(p.env.fsm,801)
            self.assertEqual(p.state,State.NATIVE_LOCO);self.assertEqual(p.env.sent,[])
            self.assertEqual(p.announcements,[safety.WALKRUN_SWITCH_TEXT])
            p.step();CLOCK[0]+=p.REGULAR_STILL_S-.01;p.step();self.assertEqual(p.state,State.NATIVE_LOCO)
            self.hold_start(p);self.assertEqual(p.env.walkrun_calls,1);self.assertEqual(p.state,State.NATIVE_LOCO)
            CLOCK[0]+=.02;p.step()
            self.assertEqual(p.state,State.MIMIC_ACTIVE);self.assertEqual(p.env.fsm,1000)
            np.testing.assert_allclose(p.policy.observed[0]['base_quat'],[0,0,0,1])  # re-anchored
            ev=[e for k,e in p.events if k=='walkrun_auto'][0]
            self.assertEqual(ev['result'],0);self.assertEqual(ev['fsm_trace'],[(0.,0,fsm,0),(.4,1,801,0)])
    def test_start_in_walkrun_enters_directly_without_switch(self):
        p=make();self.hold_start(p);self.assertEqual(p.state,State.MIMIC_ACTIVE);self.assertEqual(p.env.walkrun_calls,0)
    def test_auto_walkrun_waits_for_still_legs_then_gives_up(self):
        p=make();p.env.fsm=501;self.hold_start(p);p.env.velocity[NAMES.index('left_knee_joint')]=.25
        for _ in range(3):p.step();CLOCK[0]+=1.
        CLOCK[0]+=p.WALKRUN_ENTRY_WAIT_S;p.step();p.env.velocity[:]=0
        for _ in range(3):p.step();CLOCK[0]+=1.
        self.assertEqual(p.state,State.NATIVE_LOCO);self.assertEqual(p.env.sent,[]);self.assertEqual(p.env.fsm,801)
        self.assertEqual(p.announcements,[safety.WALKRUN_SWITCH_TEXT,safety.WALKRUN_UNSTEADY_TEXT])
        self.hold_start(p);self.assertEqual(p.state,State.MIMIC_ACTIVE)  # Start again in WALKRUN
    def test_auto_walkrun_ending_in_damp_requests_nothing_more(self):
        p=make();p.env.fsm=501;p.env.walkrun_to=1;self.hold_start(p,frames=5)
        self.assertEqual(p.env.walkrun_calls,1);self.assertEqual(p.env.fsm,1);self.assertEqual(p.env.sent,[])
        self.assertEqual(p.announcements,[safety.WALKRUN_SWITCH_TEXT,safety.WALKRUN_FAILED_TEXT])
        for _ in range(3):p.step();CLOCK[0]+=1.
        self.hold_start(p)  # new press in damp: acquire refuses, no mode request
        self.assertEqual(p.env.walkrun_calls,1);self.assertEqual(p.env.sent,[]);self.assertEqual(p.state,State.NATIVE_LOCO)
        self.assertEqual(p.announcements[-1],safety.NOT_WALKRUN_TEXT)
        self.assertFalse(p.should_stop)
    def test_auto_walkrun_ignored_or_raising_stays_in_regular(self):
        p=make();p.env.fsm=501;p.env.walkrun_to=501;self.hold_start(p)
        self.assertEqual(p.announcements,[safety.WALKRUN_SWITCH_TEXT,safety.NOT_WALKRUN_TEXT])
        self.assertEqual(p.env.fsm,501);self.assertIsNone(p._walkrun_entry_deadline)
        p=make();p.env.fsm=500
        def boom():raise RuntimeError('rpc')
        p.env.switch_regular_to_walkrun=boom;self.hold_start(p)
        self.assertEqual(p.announcements,[safety.WALKRUN_SWITCH_TEXT,safety.NOT_WALKRUN_TEXT])
        self.assertEqual(p.state,State.NATIVE_LOCO);self.assertFalse(p.should_stop)
    def test_select_cancels_auto_walkrun_entry(self):
        p=make();p.env.fsm=501;self.hold_start(p);self.press(p,'[POLICY_LOCO]')
        CLOCK[0]+=p.REGULAR_STILL_S+.01;p.step();p.step()
        self.assertEqual(p.state,State.NATIVE_LOCO);self.assertEqual(p.env.sent,[]);self.assertEqual(p.env.fsm,801)
    def test_auto_walkrun_disabled_by_config(self):
        p=make();p.cfg.walkrun_on_start=False;p.env.fsm=501;p.env.acquire_rc=-20006;self.hold_start(p)
        self.assertEqual(p.env.walkrun_calls,0);self.assertEqual(p.announcements,[safety.NOT_WALKRUN_TEXT])
    def test_dance_then_regular_then_start_dances_again(self):
        p=make(names=NAMES_29);p.env.regular_fsm=501;p._enter_mimic(p.env.get_data(),{})
        p._return_to_native_loco('motion done');settle(p);self.assertEqual(p.env.fsm,501)
        self.hold_start(p);p.step();CLOCK[0]+=p.REGULAR_STILL_S+.01;p.step()
        self.assertEqual(p.state,State.MIMIC_ACTIVE);self.assertEqual(p.env.walkrun_calls,1)
    def press(self,p,command):
        p.ctrl_manager.get_ctrl_data=lambda d:{'COMMANDS':[command]};p.step()
        p.ctrl_manager.get_ctrl_data=lambda d:{'COMMANDS':[]}
    def dancing(self,command='[SHUTDOWN]'):
        p=make(policy_cls=StancePolicy);p._enter_mimic(p.env.get_data(),{});p.policy.policy.ref_offset=.6
        self.press(p,command);return p
    def test_stop_during_dance_waits_for_stance_then_hands_back(self):
        for command in ['[SHUTDOWN]','[POLICY_LOCO]']:
            with self.subTest(command=command):
                p=self.dancing(command)
                self.assertEqual(p.state,State.MIMIC_ACTIVE);self.assertEqual(p.env.releases,0)
                self.assertEqual(p.announcements,[safety.STOPPING_TEXT])
                n=len(p.env.sent)
                for _ in range(20):p.step()
                self.assertEqual(p.env.releases,0);self.assertEqual(len(p.env.sent),n+20);self.assertEqual(p.policy.frames,21)
                p.policy.policy.ref_vel=1.;p.policy.policy.ref_offset=.1;p.step();self.assertEqual(p.env.releases,0)  # still moving
                p.policy.policy.ref_vel=0.;p.env.q[LEGS[0]]+=.6;p.step();self.assertEqual(p.env.releases,0)  # robot not there
                p.env.q[LEGS[0]]-=.6;p.step()
                self.assertEqual(p.env.releases,1);self.assertEqual(p.state,State.NATIVE_LOCO);self.assertEqual(p.env.fsm,801)
                self.assertFalse(p.should_stop);self.assertFalse(p.env.closed)
    def test_stop_pressed_again_still_waits_for_stance(self):
        p=self.dancing();p.step()
        for command in ['[SHUTDOWN]','[POLICY_LOCO]','[SHUTDOWN]']:self.press(p,command)
        self.assertEqual(p.state,State.MIMIC_ACTIVE);self.assertEqual(p.env.releases,0)
        self.assertEqual(p.policy.play_speed,1.);self.assertEqual(p.announcements,[safety.STOPPING_TEXT])
        p.policy.policy.ref_offset=0.;p.step()
        self.assertEqual(p.state,State.NATIVE_LOCO);self.assertEqual(p.env.releases,1)
    def test_no_stance_within_wait_freezes_and_holds_without_mode_request(self):
        p=self.dancing();CLOCK[0]+=p.STOP_WAIT_S+.01;p.step()
        self.assertTrue(p._hold);self.assertEqual(p.policy.policy.play_speed,0.);self.assertEqual(p.env.releases,0)
        self.assertIn(safety.STANCE_TIMEOUT_TEXT,p.announcements);n=len(p.env.sent)
        for _ in range(5):CLOCK[0]+=2;p.step()
        self.assertEqual(p.env.releases,0);self.assertEqual(len(p.env.sent),n+5);self.assertEqual(p.state,State.MIMIC_ACTIVE)
        self.press(p,'[SHUTDOWN]');self.assertEqual(p.env.releases,1);self.assertEqual(p.state,State.NATIVE_LOCO)
    def test_failed_handback_policy_keeps_holding_then_stops_retrying(self):
        p=self.dancing();p.env.release_rc=-20003;p.policy.policy.ref_offset=0.;p.step()
        self.assertEqual(p.env.releases,1);self.assertEqual(p.policy.policy.play_speed,0.);self.assertNotIn('play_speed',vars(p.policy))
        frames=p.policy.frames;p.step();p.step();self.assertEqual(p.policy.frames,frames)  # reference really frozen
        n=len(p.env.sent);p.step();self.assertEqual(len(p.env.sent),n+1);self.assertEqual(p.env.releases,1)  # policy holds, rate limit
        for _ in range(p.HANDBACK_MAX_FAILURES-1):CLOCK[0]+=p.HANDBACK_RETRY_S+.01;p.step()
        self.assertEqual(p.env.releases,p.HANDBACK_MAX_FAILURES);self.assertTrue(p._hold)
        self.assertEqual(p.announcements.count(safety.HOLD_TEXT),1)
        for _ in range(12):CLOCK[0]+=1.01;p.step()
        self.assertEqual(p.env.releases,p.HANDBACK_MAX_FAILURES);self.assertTrue(p.env.publish)
        self.assertGreaterEqual(p.announcements.count(safety.HOLD_TEXT),2)  # repeated prompt
        self.press(p,'[POLICY_LOCO]');self.assertEqual(p.env.releases,p.HANDBACK_MAX_FAILURES+1);self.assertTrue(p._hold)
        p.env.release_rc=0;self.press(p,'[SHUTDOWN]');self.assertEqual(p.state,State.NATIVE_LOCO);self.assertFalse(p._hold)
    def test_hold_detects_operator_damp_and_requests_nothing(self):
        p=self.dancing();p.env.release_rc=-20003;p.policy.policy.ref_offset=0.;p.step()
        for _ in range(p.HANDBACK_MAX_FAILURES):CLOCK[0]+=p.HANDBACK_RETRY_S+.01;p.step()
        self.assertTrue(p._hold);releases=p.env.releases
        self.assertNotEqual(p.close(),0);self.assertFalse(p.env.closed);self.assertEqual(p.env.releases,releases)
        p.env.fsm=1;CLOCK[0]+=p.HOLD_POLL_S+.01;p.step()  # operator damped with the remote
        self.assertEqual(p.state,State.NATIVE_LOCO);self.assertFalse(p.env.publish);self.assertEqual(p.env.releases,releases)
        for _ in range(30):p.step()
        self.assertEqual(p.env.regular_calls,0);self.assertEqual(p.close(),0);self.assertEqual(p.env.fsm,1)
    def test_static_handback_failures_hold_and_close_waits_for_operator(self):
        p=make();p._enter_mimic(p.env.get_data(),{});p.env.release_rc=-20003
        def failed_inference(obs):raise RuntimeError('broken')
        p.policy.get_pd_target=failed_inference
        for _ in range(p.MAX_CONSECUTIVE_STEP_ERRORS):p.step()
        self.assertEqual(p.state,State.HANDBACK)
        for _ in range(p.HANDBACK_MAX_FAILURES+3):CLOCK[0]+=p.HANDBACK_RETRY_S+.01;p.step()
        self.assertTrue(p._hold);self.assertEqual(p.env.releases,p.HANDBACK_MAX_FAILURES)
        for _ in range(3):CLOCK[0]+=1;self.assertNotEqual(p.close(),0)
        self.assertEqual(p.env.releases,p.HANDBACK_MAX_FAILURES);self.assertTrue(p.env.publish)
        p.env.fsm=1;CLOCK[0]+=p.HOLD_POLL_S+.01;self.assertEqual(p.close(),0);self.assertTrue(p.env.closed)
    def test_motion_done_freezes_reference_and_hands_back(self):
        p=make(policy_cls=StancePolicy);p._enter_mimic(p.env.get_data(),{})
        p.policy.get_observation=lambda d,c:(np.zeros(3),{'CALLBACK':['[MOTION_DONE]']});p.step()
        self.assertEqual(p.state,State.NATIVE_LOCO);self.assertEqual(p.env.releases,1)
    def test_regular_loco_skip_and_failure_are_announced(self):
        p=make();p._enter_mimic(p.env.get_data(),{});p._return_to_native_loco('motion done')
        p.env.velocity[0]=1;p.step();CLOCK[0]+=p.REGULAR_WAIT_S+.01;p.step()
        self.assertEqual(p.announcements,[safety.REGULAR_SKIPPED_TEXT]);self.assertEqual(p.env.fsm,801)
        p=make();p._enter_mimic(p.env.get_data(),{});p._return_to_native_loco('motion done')
        p.env.switch_walkrun_to_regular=lambda:-20003
        settle(p)
        self.assertEqual(p.announcements,[safety.REGULAR_SKIPPED_TEXT])
    def test_frame_drop_exit_disabled_for_this_pipeline(self):
        self.assertIs(P.exit_on_frame_drop,False)
        src=(ROOT/'scripts/run_pipeline.py').read_text()
        self.assertIn('getattr(pipeline, "exit_on_frame_drop", True)',src)

    def test_dry_run_cannot_handle_start(self):
        p=make();p.ctrl_manager.get_ctrl_data=lambda d:{'COMMANDS':['[POLICY_MIMIC]']}
        p.step(dry_run=True);self.assertEqual(p.env.sent,[]);self.assertEqual(p.env.reset_count,0)
    def test_acquire_failed_with_loco_in_control_rejects_only(self):
        p=make();p.env.acquire_rc=99;p._enter_mimic(p.env.get_data(),{})
        self.assertEqual(p.state,State.NATIVE_LOCO);self.assertFalse(p.should_stop)
        self.assertEqual(p.env.sent,[]);self.assertEqual(p.announcements,[safety.WARNING_TEXT])
    def test_speech_async_duplicate_failure(self):
        entered=threading.Event();release=threading.Event();calls=[]
        def speak(text):calls.append(text);entered.set();release.wait(.5);raise RuntimeError('no audio')
        s=safety.SafetySpeaker(speak);s.announce();self.assertTrue(entered.wait(.5));self.assertFalse(s.announce())
        release.set();s.finish();self.assertEqual(calls,[safety.WARNING_TEXT]);self.assertFalse(s._lock.locked())
    def test_29dof_same_flow(self):
        self.assertEqual(sorted(NAMES_29),sorted(safety.JOINT_LIMITS_29))
        for deg in [0,90,180]:
            p=make(np.deg2rad(deg),names=NAMES_29);p._enter_mimic(p.env.get_data(),{})
            self.assertEqual(p.state,State.MIMIC_ACTIVE);self.assertEqual(p.env.sent[0].shape,(29,))
            np.testing.assert_allclose(p.policy.observed[0]['base_quat'],[0,0,0,1])
        p=make(names=NAMES_29);p.policy.target[14]=1.;p._enter_mimic(p.env.get_data(),{})  # waist pitch jump
        self.assertEqual(p.env.sent,[]);self.assertEqual(p.announcements,[safety.WARNING_TEXT])
        p=make(policy_cls=StancePolicy,names=NAMES_29);p._enter_mimic(p.env.get_data(),{});p.policy.policy.ref_offset=.6
        self.press(p,'[SHUTDOWN]');p.step();self.assertEqual(p.env.releases,0)
        p.policy.policy.ref_offset=0.;p.step();self.assertEqual(p.state,State.NATIVE_LOCO);self.assertEqual(p.env.releases,1)
        settle(p)
        self.assertEqual(p.env.fsm,500)
        self.assertRaises(ValueError,Guard,NAMES_29[:27])
    def test_walkrun_802_counts_as_walkrun_for_prompts(self):
        p=make(names=NAMES_29);p.env.fsm=802;p.env.acquire_rc=99;p._enter_mimic(p.env.get_data(),{})
        self.assertEqual(p.announcements,[safety.WARNING_TEXT])
        p=make();p._enter_mimic(p.env.get_data(),{});p._return_to_native_loco('motion done')
        p.env.fsm=802;p.env.switch_walkrun_to_regular=lambda:-20003
        settle(p)
        self.assertEqual(p.announcements,[safety.REGULAR_SKIPPED_TEXT])
    def test_29dof_extra_joint_limits_match_asset(self):
        import xml.etree.ElementTree as ET
        xml=ET.parse(ROOT/'assets/robots/g1/g1_29dof_rev_1_0.xml')
        for n in set(safety.JOINT_LIMITS_29)-set(safety.JOINT_LIMITS):
            np.testing.assert_allclose(safety.JOINT_LIMITS_29[n],np.fromstring(xml.find(f'.//joint[@name="{n}"]').get('range'),sep=' '))
    def test_joint_limits_match_asset_by_name(self):
        import xml.etree.ElementTree as ET
        xml=ET.parse(ROOT/'assets/robots/g1/g1_23dof_backpack.xml')
        for n,lim in safety.JOINT_LIMITS.items():
            np.testing.assert_allclose(lim,np.fromstring(xml.find(f'.//joint[@name="{n}"]').get('range'),sep=' '))


class TestEnvironment(unittest.TestCase):
    def env(self, yaw):
        from scipy.spatial.transform import Rotation as R
        class Base:
            base_quat=property(lambda s:s._base_quat.copy())
            base_pos=property(lambda s:s._base_pos.copy())
            base_lin_vel=property(lambda s:s._base_lin_vel.copy())
            base_ang_vel=property(lambda s:s._base_ang_vel.copy())
            def set_born_place(s,q,p):s.base_align=NS(align_quat=lambda x:(R.from_euler('z',R.from_quat(q).as_euler('xyz')[2]).inv()*R.from_quat(x)).as_quat(),align_pos=lambda x:x-np.asarray(p))
        envns=dict(Environment=Base,UnitreeEnvCfg=object,np=np,time=time,
                   quat_rotate_inverse_np=lambda q,v:R.from_quat(q).inv().apply(v))
        load_classes(ROOT/'robojudo/environment/g1_loco_env.py',['G1LocoEnv'],envns)
        e=envns['G1LocoEnv'].__new__(envns['G1LocoEnv']);e.born_place_align=True;e._dof_idx=None
        e._odometry_type='UNITREE';e.update_with_fk=False;e.RemoteControllerHandler=None
        q=R.from_euler('z',yaw).as_quat();e.base_align=NS(align_quat=lambda x:(R.from_quat(q).inv()*R.from_quat(x)).as_quat(),align_pos=lambda x:x)
        e.sample=NS(motor_state=NS(q=np.zeros(23),dq=np.zeros(23)),
                    imu_state=NS(quaternion=q[[3,0,1,2]],gyroscope=np.zeros(3),rpy=np.zeros(3)),state_age_s=.01)
        e.unitree=NS(get_robot_state=lambda:e.sample,get_sport_state=lambda:NS(position=[0,0,0],velocity=[1,0,0]))
        return e
    def test_world_velocity_uses_raw_orientation(self):
        e=self.env(np.pi/2);e.update()
        np.testing.assert_allclose(e.base_quat,[0,0,0,1],atol=1e-6)
        np.testing.assert_allclose(e.base_lin_vel,[0,-1,0],atol=1e-6)
    def test_real_reset_reanchors_changed_pose(self):
        from scipy.spatial.transform import Rotation as R
        for deg in [0,90,175,180]:
            e=self.env(0);q=R.from_euler('z',deg,degrees=True).as_quat();e.sample.imu_state.quaternion=q[[3,0,1,2]]
            e.reset();np.testing.assert_allclose(e.base_quat,[0,0,0,1],atol=1e-6)
    def test_failed_update_restores_alignment_flag(self):
        e=self.env(0)
        def fail():raise RuntimeError('missing state')
        e.unitree.get_robot_state=fail
        with self.assertRaises(RuntimeError):e.reset()
        self.assertTrue(e.born_place_align)
    def test_snapshot_age_includes_time_since_update(self):
        e=self.env(0);e.update();e._state_read_time-=.2
        self.assertGreater(e.safety_state_age(),.2)

if __name__=='__main__':unittest.main(verbosity=2)
