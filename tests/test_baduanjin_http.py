"""In-process HTTP compatibility checks; block SDK imports and replace only I/O."""
import importlib.abc
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace as NS
from unittest.mock import Mock, patch
import unittest
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

class NoHardware(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.startswith(('g1_loco_bridge', 'unitree_sdk2py', 'unitree_cpp', 'zed_proxy')):
            raise AssertionError(f'HTTP test attempted hardware import: {fullname}')

_guard = NoHardware()
sys.meta_path.insert(0, _guard)
try:
    from fastapi.testclient import TestClient
    from scripts import run_service as api
    from robojudo.pipeline.g1_baduanjin_suite_pipeline import G1BaduanjinSuitePipeline
    from robojudo.pipeline.g1_native_loco_mimic_pipeline import G1NativeLocoMimicPipeline, NativeLocoMimicState
finally:
    sys.meta_path.remove(_guard)


class Manager:
    def __init__(self, cfg):
        self.policies = [NS(cfg_policy=p, name='BeyondMimicPolicy@'+p.policy_name) for p in cfg.mimic_policies]
        self._current_policy_id = 0
        self.num_policies = len(self.policies)
        self.selections = []
    @property
    def policy(self): return self.policies[self._current_policy_id]
    @property
    def current_policy_id(self): return self._current_policy_id
    def set_policy(self, index):
        self._current_policy_id = index
        self.selections.append(index)


class TestHTTPCompatibility(unittest.TestCase):
    def setUp(self):
        sys.meta_path.insert(0, _guard)
        self.addCleanup(sys.meta_path.remove, _guard)
        api.pipeline = None
        api.pending_commands.clear()
        api.stop_event.clear()
        self.client = TestClient(api.app)
        self.addCleanup(self.client.close)
        self.env_patch = patch.dict(os.environ, {}, clear=False)
        self.env_patch.start(); self.addCleanup(self.env_patch.stop)
        os.environ.pop('ROBOJUDO_RECORD_DIR', None)
        self.thread_patch = patch.object(api.threading, 'Thread')
        self.thread_patch.start(); self.addCleanup(self.thread_patch.stop)
        self.factory_patch = patch.object(api.robojudo.pipeline, 'G1BaduanjinSuitePipeline', side_effect=self.fake_pipeline)
        self.factory_patch.start(); self.addCleanup(self.factory_patch.stop)
        self.addCleanup(lambda: setattr(api, 'pipeline', None))

    def fake_pipeline(self, cfg):
        p = G1BaduanjinSuitePipeline.__new__(G1BaduanjinSuitePipeline)
        p.cfg = cfg; p.policy_manager = Manager(cfg)
        p.state = NativeLocoMimicState.NATIVE_LOCO
        p._walkrun_entry_deadline = None; p._entry_blocked = False
        p._segment_blend = None; p._suite_cancelled = False
        p.should_stop = False; p.timestep = 0; p.dt = .02
        p.env = NS(get_data=lambda: {}, get_control_status=lambda: dict(fsm_id=500, publish_enabled=False))
        p.ctrl_manager = NS(get_ctrl_data=lambda _: {'COMMANDS': []})
        p.trace_event = Mock(); p.prepare = Mock(); p.configure_motion_trace = Mock()
        p._start_pressed = Mock(); p._operator_stop = Mock()
        # Run the REAL inherited command dispatcher through the REAL service loop.
        def step():
            data = p.env.get_data(); ctrl = p.ctrl_manager.get_ctrl_data(data)
            p._handle_commands(ctrl.get('COMMANDS', []), data, ctrl)
            api.stop_event.set()
        p.step = step
        return p

    def start(self, config='g1_baduanjin_23dof_suite'):
        response = self.client.post('/start', json={'config_name': config})
        self.assertEqual(response.status_code, 200, response.text)
        return api.pipeline

    def drain(self):
        api.stop_event.clear()
        api.control_loop()

    def real_entry_dispatch(self, p):
        """Use real Start/entry selection, replacing only reference and acquisition I/O."""
        p._start_pressed = G1NativeLocoMimicPipeline._start_pressed.__get__(p)
        for policy in p.policy_manager.policies:
            policy.motion_anchor_body_index = 0
            policy.command = {'body_quat_w': np.array([[1., 0., 0., 0.]]),
                              'body_pos_w': np.array([[0., 0., .8]])}
            policy.timestep = policy.cfg_policy.max_timestep
            policy.reset = Mock(side_effect=lambda actor=policy: setattr(actor, 'timestep', 0))
        entered = []
        def acquire(actor, data, ctrl):
            if actor.state == NativeLocoMimicState.NATIVE_LOCO and not actor._entry_blocked:
                entered.append((actor.policy_manager.current_policy_id, actor.policy.timestep))
                actor.state = NativeLocoMimicState.MIMIC_ACTIVE
        patcher = patch.object(G1NativeLocoMimicPipeline, '_enter_mimic', autospec=True, side_effect=acquire)
        patcher.start(); self.addCleanup(patcher.stop)
        return entered

    def test_http_enter_alone_starts_cut1_again_after_suite_or_early_stop(self):
        p = self.start(); entered = self.real_entry_dispatch(p)
        for previous in (0, 2, 1):
            with self.subTest(previous=previous):
                p.state = NativeLocoMimicState.NATIVE_LOCO
                p.policy_manager._current_policy_id = previous
                p.policy_manager.policies[0].timestep = 8627
                self.assertEqual(self.client.post('/enter_mimic').status_code, 200)
                self.drain()
                self.assertEqual(entered[-1], (0, 0))
                self.assertEqual(self.client.get('/status').json()['sequence_index'], 0)
        self.assertEqual(len(entered), 3)

    def test_http_single_segment_still_starts_selected_cut(self):
        p = self.start('g1_baduanjin_23dof_select'); entered = self.real_entry_dispatch(p)
        self.assertEqual(self.client.post('/select_policy', json={'sequence_index': 2}).status_code, 200)
        self.client.post('/enter_mimic'); self.drain()
        self.assertEqual(entered, [(2, 0)])

    def test_repeated_enter_during_dance_or_walkrun_wait_does_not_reset(self):
        p = self.start(); entered = self.real_entry_dispatch(p)
        p.policy_manager._current_policy_id = 2
        p.state = NativeLocoMimicState.MIMIC_ACTIVE
        self.client.post('/enter_mimic'); self.drain()
        self.assertEqual(entered, []); self.assertEqual(p.policy_manager.current_policy_id, 2)
        p.state = NativeLocoMimicState.NATIVE_LOCO
        p._walkrun_entry_deadline = time.monotonic() + 30
        self.client.post('/enter_mimic'); self.drain()
        self.assertEqual(entered, []); self.assertEqual(p.policy_manager.current_policy_id, 2)
        # The original delayed WALKRUN entry must eventually select cut1 too.
        p._walkrun_still_since = time.monotonic() - 2
        p._held_still = Mock(return_value=p._walkrun_still_since)
        p._advance_walkrun_entry({}, {})
        self.assertEqual(entered, [(0, 0)])

    def test_original_start_enter_routes_select_suite_and_deliver_start_command(self):
        api.pending_commands.append('[POLICY_MIMIC]')  # Must not survive an old session.
        p = self.start()
        self.assertEqual(api.pending_commands, [])
        p._start_pressed.assert_not_called()
        self.assertTrue(p.cfg.auto_sequence)
        self.assertEqual(p.cfg.env.unitree.regular_fsm_id, 500)
        self.assertEqual(self.client.post('/enter_mimic').json()['status'], 'enter_mimic_queued')
        p._start_pressed.assert_not_called()
        self.drain()
        p._start_pressed.assert_called_once()

    def test_exact_selection_by_index_and_name_and_restart_from_zero(self):
        p = self.start()
        names = self.client.get('/policies').json()['policies']
        self.assertEqual(names, [v.policy_name for v in p.cfg.mimic_policies])
        response = self.client.post('/select_policy', json={'policy_name': names[2]})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()['sequence_index'], 2)
        response = self.client.post('/select_policy', json={'sequence_index': 0})
        self.assertEqual(response.json()['policy_name'], names[0])
        self.client.post('/enter_mimic'); self.drain()
        self.assertEqual(p.policy_manager.current_policy_id, 0)
        p._start_pressed.assert_called_once()

    def test_select_rejected_during_dance_or_pending_entry(self):
        p = self.start()
        for state, pending in [(NativeLocoMimicState.MIMIC_ACTIVE, None), (NativeLocoMimicState.NATIVE_LOCO, 1.)]:
            p.state = state; p._walkrun_entry_deadline = pending
            self.assertEqual(self.client.post('/select_policy', json={'sequence_index': 1}).status_code, 400)
        self.assertEqual(p.policy_manager.selections, [])

    def test_invalid_selection_never_changes_index(self):
        p = self.start()
        for payload, code in [({}, 422), ({'policy_name': 'missing'}, 400), ({'sequence_index': -1}, 400),
                              ({'sequence_index': 3}, 400), ({'sequence_index': True}, 422),
                              ({'sequence_index': 1.2}, 422), ({'sequence_index': 0, 'policy_name': 'anything'}, 422)]:
            with self.subTest(payload=payload):
                self.assertEqual(self.client.post('/select_policy', json=payload).status_code, code)
        self.assertEqual(p.policy_manager.selections, [])

    def test_status_preserves_old_keys_and_exposes_selection(self):
        self.start(); status = self.client.get('/status').json()
        self.assertEqual(status['config_name'], 'g1_baduanjin_23dof_suite')
        self.assertEqual(status['sequence_index'], 0)
        self.assertEqual(status['policy_name'], 'cut1_scratch_v2_i15000_zero_translation')
        for key in ('state', 'env', 'policy', 'timestep'): self.assertIn(key, status)

    def test_switch_and_return_routes_keep_existing_command_semantics(self):
        p = self.start()
        self.assertEqual(self.client.post('/switch_policy?direction=next').status_code, 200)
        self.drain(); self.assertEqual(p.policy_manager.current_policy_id, 1)
        p.state = NativeLocoMimicState.MIMIC_ACTIVE
        self.assertEqual(self.client.post('/return_loco').status_code, 200)
        self.drain(); p._operator_stop.assert_called_once_with('[POLICY_LOCO]')

    def test_trace_enabled_for_service_without_starting_second_controller(self):
        os.environ['ROBOJUDO_RECORD_DIR'] = '/offline-fixture/no-real-writer'
        os.environ['ROBOJUDO_RUN_ID'] = 'http-test'
        p = self.start()
        p.configure_motion_trace.assert_called_once_with('/offline-fixture/no-real-writer', 'http-test')
        p.prepare.assert_called_once(); self.assertEqual(p._motion_trace_phase, 'playing')

    def test_old_defaults_and_unimplemented_alias_not_silently_remapped(self):
        self.assertEqual(api.StartRequest().config_name, 'g1_native_loco_mimic')
        self.assertEqual(self.client.post('/select_sequence', json={'sequence_index': 0}).status_code, 404)
        self.assertEqual(self.client.post('/enter_mimic').status_code, 400)

if __name__ == '__main__': unittest.main()
