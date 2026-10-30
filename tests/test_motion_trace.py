"""Offline proof that tracing observes sends without changing control behavior."""
import abc
import ast
import importlib.util
import logging
import sys
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import msgpack
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("robojudo.tools.motion_trace", ROOT/"robojudo/tools/motion_trace.py")
trace_module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(trace_module)


def load_class(path, name, namespace):
    """Use production method bodies without importing robot/GPU dependencies."""
    tree = ast.parse((ROOT/path).read_text())
    node = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == name)
    node.decorator_list = []
    module = ast.Module(body=[node], type_ignores=[])
    exec(compile(module, str(path), "exec"), namespace)
    return namespace[name]


Pipeline = load_class(Path('robojudo/pipeline/base_pipeline.py'), 'Pipeline', {
    'ABC': abc.ABC, 'abstractmethod': abc.abstractmethod, 'PipelineCfg': object,
    'logger': logging.getLogger('test'),
})


class Harness(Pipeline):
    def __init__(self, trace):
        self._motion_trace = trace
        self._motion_trace_phase = 'playing'
        self.timestep = 0
        self.sent = []
        self.env = SimpleNamespace(step=self.send)
        self.policy = SimpleNamespace(
            cfg_policy=SimpleNamespace(policy_name='cut1', policy_file='cut1.onnx', freq=50),
            cfg_action_dof=SimpleNamespace(joint_names=['left_knee_joint', 'right_knee_joint']),
            _motion_trace_enabled=trace is not None, _motion_trace_generation=1,
            _motion_trace_snapshot=None, play_speed=1., flag_motion_done=False,
        )

    def send(self, target, hand=None):
        self.sent.append((target.copy(), hand))
        return 7

    def step(self):
        pass

    def prepare(self):
        pass


def read(path):
    with path.open('rb') as handle:
        return list(msgpack.Unpacker(handle, raw=False))


class TestMotionTrace(unittest.TestCase):
    def test_actual_pipeline_send_sites(self):
        with tempfile.TemporaryDirectory() as directory:
            trace = trace_module.MotionTrace(Path(directory)/'motion.msgpack', {'run_id': 'integration'})
            observed, baseline = Harness(trace), Harness(None)
            target = np.array([.3, .6])
            for harness in (observed, baseline):
                harness.env.update = lambda: None
                harness.env.get_data = lambda: {}
                harness.ctrl_manager = SimpleNamespace(get_ctrl_data=lambda _data: {'COMMANDS': []})
                harness.policy.cfg_policy.preserve_state_during_dry_run = True
                harness.policy.get_observation = lambda *_args: (np.zeros(2), {'hand_pose': 'hand'})
                harness.policy.get_pd_target = lambda _obs: target.copy()
                harness.policy.obs_adapter = SimpleNamespace(fit=lambda value: value)
                harness.policy_manager = SimpleNamespace(
                    current_policy_id=0, policy_loco_id=0,
                    override_dof_pos=np.array([.9, .8]),
                    interp_state=SimpleNamespace(name='IDLE'))
                harness.override_dof_indices = [0]
                harness.post_step_callback = lambda *_args: None
            for path, cls, method in (
                ('robojudo/pipeline/rl_pipeline.py', 'RlPipeline', 'step'),
                ('robojudo/pipeline/g1_native_loco_mimic_pipeline.py', 'G1NativeLocoMimicPipeline', '_step_mimic'),
                ('robojudo/pipeline/rl_loco_mimic_pipeline.py', 'RlLocoMimicPipeline', 'step'),
            ):
                tree = ast.parse((ROOT/path).read_text())
                owner = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name==cls)
                fn = next(n for n in owner.body if isinstance(n, ast.FunctionDef) and n.name==method)
                scope = {}
                exec(compile(ast.Module(body=[fn], type_ignores=[]), path, 'exec'), scope)
                for harness in (observed, baseline):
                    if method == 'step':
                        scope[method](harness)
                    else:
                        scope[method](harness, {}, {'COMMANDS': []})
            observed.close_motion_trace()
            self.assertEqual(len(observed.sent), 3)
            for a,b in zip(observed.sent, baseline.sent):
                np.testing.assert_array_equal(a[0], b[0])
                self.assertEqual(a[1], b[1])
            cycles = [e['payload'] for e in read(trace.path) if e['kind']=='cycle']
            self.assertEqual(len(cycles), 3)
            np.testing.assert_array_equal(cycles[-1]['final_target'], [.9, .6])

    def test_control_identity_alignment_and_lifecycle(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'motion.msgpack'
            trace = trace_module.MotionTrace(path, {'run_id': 'run01'})
            observed, baseline = Harness(trace), Harness(None)
            target = np.array([.2, .4])
            for phase, frame, generation, speed in [
                ('preparing', 0, 1, 1.), ('playing', 0, 1, 1.),
                ('playing', 1, 1, 1.), ('playing', 1, 1, 0.),
                ('playing', 0, 2, 1.), ('playing', 1, 2, 1.),
            ]:
                observed._motion_trace_phase = baseline._motion_trace_phase = phase
                policy = observed.policy
                policy._motion_trace_generation = generation
                policy.play_speed = speed
                outputs = [target.copy(), target.copy(), target*0,
                           np.zeros((14, 3)), np.tile([1., 0, 0, 0], (14, 1))]
                trace_module.capture_inference(policy, frame, {'time_step': max(0, frame-1)},
                                               outputs, target, target)
                trace_module.capture_target(policy, target)
                self.assertEqual(observed.send_traced(target, 'hand'), baseline.send_traced(target, 'hand'))
                target += .1  # Queued records must retain the old values.
                observed.timestep += 1
            observed.close_motion_trace()
            events = read(path)
            cycles = [e['payload'] for e in events if e['kind'] == 'cycle']
            self.assertEqual(len(observed.sent), len(baseline.sent))
            for a, b, cycle in zip(observed.sent, baseline.sent, cycles):
                np.testing.assert_array_equal(a[0], b[0])
                np.testing.assert_array_equal(cycle['final_target'], a[0])
                np.testing.assert_array_equal(cycle['inference']['policy_target'], a[0])
            self.assertEqual([c['inference']['query_frame'] for c in cycles], [0, 0, 1, 1, 0, 1])
            self.assertEqual([c['phase'] for c in cycles], ['preparing', 'playing', 'playing', 'paused', 'playing', 'playing'])
            self.assertNotEqual(cycles[3]['segment'], cycles[4]['segment'])
            self.assertEqual(cycles[2]['inference']['observation_reference']['time_step'], 0)
            self.assertEqual(events[0]['payload']['metadata']['run_id'], 'run01')
            self.assertEqual(events[-1]['kind'], 'summary')
            self.assertTrue(events[-1]['payload']['complete'])
            self.assertEqual(len([e for e in events if e['kind']=='send_result']), 6)
            self.assertTrue(all(a['monotonic_time_ns'] <= b['monotonic_time_ns'] for a,b in zip(events,events[1:])))

    def test_actual_onnx_query_snapshot_without_extra_inference(self):
        policy_class = load_class(Path('robojudo/policy/beyondmimic_policy.py'), 'BeyondMimicPolicy', {
            'Policy': object, 'BeyondMimicPolicyCfg': object, 'np': np,
            'MujocoVisualizer': object,
        })
        policy = policy_class.__new__(policy_class)
        policy._motion_trace_enabled = True
        policy._motion_trace_generation = 2
        policy.timestep = 12.5
        policy.last_action = np.array([.1, .2])
        policy.action_beta = .5
        policy.action_scales = np.array([2., 3.])
        policy.use_motion_from_model = True
        policy.command = {'time_step': 11, 'joint_pos': np.zeros(2)}
        outputs = [np.array([[.3, .4]]), np.array([[1., 2.]]), np.zeros((1, 2)),
                   np.zeros((1, 14, 3)), np.tile([1., 0, 0, 0], (1, 14, 1))]
        policy.session = SimpleNamespace(run=mock.Mock(return_value=outputs))
        with mock.patch.dict(sys.modules, {'robojudo.tools.motion_trace': trace_module}):
            result = policy.get_action(np.zeros(5))
        np.testing.assert_allclose(result, [.4, .9])
        self.assertEqual(policy.session.run.call_count, 1)
        self.assertEqual(policy._motion_trace_snapshot['query_frame'], 12)
        self.assertEqual(policy._motion_trace_snapshot['observation_reference']['time_step'], 11)
        np.testing.assert_array_equal(policy._motion_trace_snapshot['output_reference']['joint_pos'], [[1., 2.]])

    def test_write_failure_and_original_send_exception(self):
        with tempfile.TemporaryDirectory() as directory:
            trace = trace_module.MotionTrace(Path(directory)/'motion.msgpack', {})
            observed = Harness(trace)
            with mock.patch.object(trace, '_write', side_effect=OSError('disk full')):
                trace.record_event('test')
                deadline = time.monotonic()+1
                while not trace.error and time.monotonic()<deadline:
                    time.sleep(.005)
                self.assertIn('disk full', trace.error)
                target = np.array([.2, .4])
                self.assertEqual(observed.send_traced(target), 7)
                np.testing.assert_array_equal(observed.sent[-1][0], target)
                failure = RuntimeError('original send failure')
                def fail(*_args):
                    raise failure
                with self.assertRaises(RuntimeError) as caught:
                    observed.send_traced(target, sender=fail)
                self.assertIs(caught.exception, failure)
            observed.close_motion_trace()


if __name__ == '__main__':
    unittest.main()
