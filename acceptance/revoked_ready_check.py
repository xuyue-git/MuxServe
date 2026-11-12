"""CPU-only deterministic PREPARE/cancel/late-READY revocation regression."""
import ast
from copy import deepcopy
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'acceptance'))
from controller_check import Fixture
from muxserve import trace
from muxserve.config import DeploymentConfig, ModelDeployment
from muxserve.contracts import BatchPlan, IntegrationError, RequestRef, RequestSpec
from muxserve.muxsched.controller import MuxController
from muxserve.flexstore.storage import HeadwiseStore


class RevokedReadyChecks(unittest.TestCase):
    def cancelled(self, *, timeout=False, protected=False, decode=False):
        fixture = Fixture()
        fixture.submit(max_new=3, timeout=1000 if timeout else 120000)
        plan, batch = fixture.prepare()
        if decode:
            fixture.ready_start(plan, batch)
            fixture.stage_prefill(plan)
            fixture.gpu(plan, batch, 4, 2)
            fixture.handoff(plan)
            fixture.batch_quiesce(plan, batch)
            fixture.ack_all_commands()
            plan, batch = fixture.prepare(stage='D', batch_id='decode')
        if timeout:
            fixture.clock.now += 1.1
            fixture.controller.tick()
        else:
            fixture.controller.cancel_request('m', 'r', 'http_disconnect')
        if protected:
            fixture.controller.protect_failure('separate-failure')
        return fixture, plan, batch

    def test_cancelled_prefill_and_decode_late_ready_keep_barrier_and_sequence(self):
        for decode in (False, True):
            for timeout in (False, True):
                with self.subTest(decode=decode, timeout=timeout):
                    fixture, plan, batch = self.cancelled(decode=decode, timeout=timeout)
                    reservation = {'reservation_id': batch['reservation_id']}
                    for stage in ('P', 'D'):
                        reply = fixture.report(stage, plan.batch_id, 'READY', reservation)
                        self.assertTrue(reply.get('obsolete'), 'Revoked READY must be acknowledged as obsolete')
                        again = fixture.controller.rank_report(fixture.group, stage, 0,
                            1001 if stage == 'P' else 1002, str(fixture.seq[stage]),
                            plan.batch_id, 'READY', reservation)
                        self.assertTrue(again['duplicate'])
                    active = fixture.controller.groups['m'].batches[plan.batch_id]
                    self.assertEqual(active.status, 'REVOKING')
                    self.assertFalse(active.ready)
                    self.assertTrue(fixture.store.snapshot()['allocations'])
                    with self.assertRaises(IntegrationError):
                        fixture.controller.start_batch(fixture.group, plan.batch_id)
                    proof = dict(reservation, request_ids=('r',))
                    fixture.report('P', plan.batch_id, 'QUIESCED', proof)
                    self.assertTrue(fixture.store.snapshot()['allocations'])
                    fixture.report('D', plan.batch_id, 'QUIESCED', proof)
                    fixture.cleanup('r')
                    self.assertFalse(fixture.store.snapshot()['allocations'])
                    self.assertEqual(fixture.controller.request_output('m', 'r')['state'],
                                     'TIMED_OUT' if timeout else 'CANCELLED')
                    self.assertTrue(fixture.controller.groups['m'].ready)

    def test_protected_revoke_accepts_matching_ready_without_unfencing(self):
        fixture, plan, batch = self.cancelled(protected=True)
        for stage in ('P', 'D'):
            self.assertTrue(fixture.report(stage, plan.batch_id, 'READY',
                {'reservation_id': batch['reservation_id']}).get('obsolete'))
        self.assertTrue(fixture.controller.groups['m'].fenced)
        self.assertTrue(fixture.store.snapshot()['allocations'])
        with self.assertRaises(IntegrationError):
            fixture.submit('new')

    def test_wrong_identity_sequence_and_nonrevoked_states_stay_rejected(self):
        fixture, plan, batch = self.cancelled()
        for pid, sequence, reservation in ((9999, '1', batch['reservation_id']),
                (1001, '2', batch['reservation_id']), (1001, '1', 'wrong')):
            with self.subTest(pid=pid, sequence=sequence, reservation=reservation), self.assertRaises(IntegrationError):
                fixture.controller.rank_report(fixture.group, 'P', 0, pid, sequence,
                    plan.batch_id, 'READY', {'reservation_id': reservation})
        self.assertEqual(fixture.controller.groups['m'].registered[('P', 0)].last_report_seq, 0)
        running = Fixture()
        running.submit(max_new=2)
        plan, batch = running.prepare()
        running.ready_start(plan, batch)
        with self.assertRaises(IntegrationError) as caught:
            running.report('P', plan.batch_id, 'READY', {'reservation_id': batch['reservation_id']})
        self.assertEqual(caught.exception.code, 'STALE_REPORT')

    def test_prepare_timeout_late_ready_is_obsolete_and_changed_replay_still_fences(self):
        fixture = Fixture()
        fixture.submit(max_new=3)
        plan, batch = fixture.prepare()
        fixture.clock.now += fixture.controller.config.prepare_timeout_ms / 1000 + 0.01
        fixture.controller.tick()
        payload = {'reservation_id': batch['reservation_id']}
        self.assertTrue(fixture.report('P', plan.batch_id, 'READY', payload)['obsolete'])
        with self.assertRaises(IntegrationError) as caught:
            fixture.controller.rank_report(fixture.group, 'P', 0, 1001, str(fixture.seq['P']),
                plan.batch_id, 'READY', {'reservation_id': 'changed'})
        self.assertEqual(caught.exception.code, 'IDENTITY_CONFLICT')
        self.assertTrue(fixture.controller.groups['m'].fenced)
        self.assertTrue(fixture.store.snapshot()['allocations'])

    def test_new_ready_after_that_rank_quiesced_is_still_rejected(self):
        fixture, plan, batch = self.cancelled()
        payload = {'reservation_id': batch['reservation_id']}
        fixture.report('P', plan.batch_id, 'QUIESCED', dict(payload, request_ids=('r',)))
        with self.assertRaises(IntegrationError) as caught:
            fixture.report('P', plan.batch_id, 'READY', payload)
        self.assertEqual(caught.exception.code, 'STALE_REPORT')
        self.assertTrue(fixture.store.snapshot()['allocations'])

    def test_tp4_retains_storage_until_all_eight_revoke_proofs(self):
        model = ModelDeployment('m', 'unused', '0'*40, 'unused', '0'*40,
                                1, 4, 512, 30, 18100, 18101, 18102)
        store = HeadwiseStore(dict.fromkeys(range(4), 64))
        controller = MuxController(DeploymentConfig((model,), tp_size=4,
            kv_bytes_per_rank=64*8192), store, expected_lengths={'m': 16})
        group = controller.group_id('m')
        keys = [(stage, rank) for stage in ('P', 'D') for rank in range(4)]
        seq = dict.fromkeys(keys, 0)
        cap = dict(vocab_size=32, eos_token_ids=(9,), native_pd=True, headwise_kv=True)
        pid = lambda key: 2000 + (key[0] == 'D')*10 + key[1]
        for key in keys:
            controller.register_rank(group, *key, pid(key), cap)
        controller.submit_request(RequestSpec('m', 'r', (1, 2), 2, 120000,
            {'temperature': 0, 'ignore_eos': False}))
        plan = BatchPlan(group, 'p', 'm', 'P', (RequestRef('r', 0, 0),))
        batch = controller.prepare_batch(plan)
        controller.cancel_request('m', 'r')
        def report(key, kind, payload):
            seq[key] += 1
            return controller.rank_report(group, *key, pid(key), str(seq[key]), 'p', kind, payload)
        for key in keys:
            self.assertTrue(report(key, 'READY', {'reservation_id': batch['reservation_id']}).get('obsolete'))
        self.assertFalse(controller.groups['m'].batches['p'].ready)
        for key in keys[:-1]:
            report(key, 'QUIESCED', {'reservation_id': batch['reservation_id'], 'request_ids': ('r',)})
            self.assertTrue(store.snapshot()['allocations'])
        report(keys[-1], 'QUIESCED', {'reservation_id': batch['reservation_id'], 'request_ids': ('r',)})
        self.assertEqual(controller.groups['m'].batches['p'].status, 'ABORTED')
        self.assertFalse(store.snapshot()['allocations'])
        for key in keys:
            self.assertFalse(any(item['kind'] == 'START' for item in controller.poll_commands(group, *key, pid(key))))

    def test_actual_native_report_retries_lost_ack_then_advances_to_quiesced(self):
        # Execute the unchanged native method without importing Torch or SGLang.
        tree = ast.parse((ROOT / 'muxserve/flexserver/sglang_runtime.py').read_text())
        owner = next(node for node in tree.body if isinstance(node, ast.ClassDef)
                     and node.name == 'ControlledScheduler')
        method = next(node for node in owner.body if isinstance(node, ast.FunctionDef)
                      and node.name == '_report')
        require = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'require')
        namespace = dict(deepcopy=deepcopy, IntegrationError=IntegrationError, trace=trace)
        exec(compile(ast.Module(body=[require, method], type_ignores=[]), 'native-report-fixture', 'exec'), namespace)
        fixture, plan, batch = self.cancelled()
        lost = [True]
        def rpc(name, *args):
            self.assertEqual(name, 'rank_report')
            reply = fixture.controller.rank_report(fixture.group, *args)
            if lost[0]:
                lost[0] = False
                raise TimeoutError('Synthetic lost RPC response after accepted obsolete READY')
            return reply
        worker = SimpleNamespace(report_seq=0, pending_report=None, rpc=rpc,
                                 ctx=SimpleNamespace(stage='P'), rank=0, pid=1001)
        report = namespace['_report']
        payload = {'reservation_id': batch['reservation_id']}
        with self.assertRaises(TimeoutError):
            report(worker, plan.batch_id, 'READY', payload)
        self.assertIsNotNone(worker.pending_report)
        report(worker, plan.batch_id, 'READY', payload)
        self.assertEqual(worker.report_seq, 1)
        self.assertIsNone(worker.pending_report)
        report(worker, plan.batch_id, 'QUIESCED', dict(payload, request_ids=('r',)))
        self.assertEqual(worker.report_seq, 2)
        self.assertTrue(fixture.store.snapshot()['allocations'])
        fixture.report('D', plan.batch_id, 'QUIESCED', dict(payload, request_ids=('r',)))
        self.assertFalse(fixture.store.snapshot()['allocations'])


if __name__ == '__main__':
    unittest.main(verbosity=2)
