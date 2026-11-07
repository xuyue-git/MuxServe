"""CPU regressions for command polling and protected revocation acknowledgments."""
from pathlib import Path
from dataclasses import asdict, replace
import sys
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'acceptance'))
from controller_check import Fixture
from muxserve_sglang.contracts import IntegrationError, KVTarget, RequestIdentity, pack_tables
from muxserve_sglang.storage import HeadwiseStore
from muxserve_sglang.startup import _InitializationController, warmup_shapes


class CommandPollingChecks(unittest.TestCase):
    def test_idle_poll_does_not_materialize_block_tables(self):
        fixture = Fixture()
        fixture.store.reserve_kv('live-mapping', [KVTarget(RequestIdentity('m', 'held'), 32)], 'holder')
        with patch.object(fixture.store, 'snapshot', side_effect=AssertionError('Full KV table copied during polling')):
            self.assertEqual(fixture.controller.poll_commands(fixture.group, 'P', 0, 1001), ())
            self.assertEqual(fixture.controller.poll_commands(fixture.group, 'D', 0, 1002), ())

    def test_poll_after_completed_batch_does_not_copy_live_mapping(self):
        fixture = Fixture()
        fixture.submit(max_new=2)
        plan, batch = fixture.prepare()
        fixture.ready_start(plan, batch)
        fixture.stage_prefill(plan)
        fixture.gpu(plan, batch, 4, 2)
        fixture.handoff(plan)
        fixture.batch_quiesce(plan, batch)
        with patch.object(fixture.store, 'snapshot', side_effect=AssertionError('Full KV table copied during polling')):
            fixture.controller.poll_commands(fixture.group, 'P', 0, 1001)
        self.assertIn(plan.batch_id, fixture.controller.groups['m'].batches)
        self.assertTrue(fixture.store.snapshot()['allocations'], 'Live decode allocation retired prematurely')

    def test_allocation_ids_are_independent_immutable_snapshots(self):
        store = HeadwiseStore({0: 4})
        from muxserve_sglang.contracts import ModelLayout
        store.register_model(ModelLayout('m', 1, 1, (0,)))
        initial = store.allocation_ids()
        self.assertIsInstance(initial, frozenset)
        first = store.reserve_kv('first', [KVTarget(RequestIdentity('m', 'first'), 1)], 'holder')
        first_id = first.views[0].allocation_id
        one = store.allocation_ids()
        second = store.reserve_kv('second', [KVTarget(RequestIdentity('m', 'second'), 1)], 'holder')
        self.assertEqual(initial, frozenset())
        self.assertEqual(one, frozenset((first_id,)))
        self.assertEqual(store.allocation_ids(), frozenset((first_id, second.views[0].allocation_id)))


class ProtectedRevocationChecks(unittest.TestCase):
    def test_requested_revoke_proofs_survive_supervisor_protection(self):
        fixture = Fixture()
        fixture.submit()
        plan, batch = fixture.prepare()
        fixture.controller.cancel_request('m', 'r')
        self.assertEqual(fixture.controller.groups['m'].batches[plan.batch_id].status, 'REVOKING')
        fixture.controller.protect_failure('STARTUP_WARMUP_FAILED')
        payload = {'reservation_id': batch['reservation_id'], 'request_ids': ('r',)}
        fixture.report('P', plan.batch_id, 'QUIESCED', payload)
        self.assertTrue(fixture.store.snapshot()['allocations'], 'One rank proof released storage early')
        with self.assertRaises(IntegrationError) as caught:
            fixture.report('D', plan.batch_id, 'QUIESCED', {**payload, 'reservation_id': 'wrong'})
        self.assertEqual(caught.exception.code, 'INVALID_REPORT')
        fixture.report('D', plan.batch_id, 'QUIESCED', payload)
        self.assertEqual(fixture.controller.groups['m'].batches[plan.batch_id].status, 'ABORTED')
        self.assertFalse(fixture.store.snapshot()['allocations'])
        self.assertTrue(fixture.controller.groups['m'].fenced)
        with self.assertRaises(IntegrationError) as caught:
            fixture.submit('new')
        self.assertEqual(caught.exception.code, 'PROTECTED_FAILURE')

    def test_unrequested_proof_cannot_release_running_fenced_batch(self):
        fixture = Fixture()
        fixture.submit()
        plan, batch = fixture.prepare()
        fixture.ready_start(plan, batch)
        fixture.controller.protect_failure('uncertain-execution')
        with self.assertRaises(IntegrationError) as caught:
            fixture.report('P', plan.batch_id, 'QUIESCED', {
                'reservation_id': batch['reservation_id'], 'request_ids': ('r',)})
        self.assertEqual(caught.exception.code, 'STALE_REPORT')
        self.assertTrue(fixture.store.snapshot()['allocations'])
        self.assertEqual(fixture.controller.groups['m'].batches[plan.batch_id].status, 'FENCED')


class PreparationMetadataChecks(unittest.TestCase):
    def test_policy_snapshot_needs_only_rank_counts(self):
        fixture = Fixture()
        with patch.object(fixture.store, 'snapshot', side_effect=AssertionError('Policy copied full block tables')):
            snapshot = fixture.controller._snapshot_policy()
        self.assertEqual(snapshot.rank_free_blocks, {0: 64})

    def test_payload_packs_tables_once_per_batch(self):
        fixture = Fixture()
        fixture.submit()
        plan, _ = fixture.prepare()
        controller = fixture.controller
        batch = controller.groups['m'].batches[plan.batch_id]
        original = batch.views['r']
        expected = {**asdict(original), 'block_tables': pack_tables(original.block_tables)}

        class GuardedTables(dict):
            def items(self):
                raise AssertionError('Payload re-traversed full block tables after packing them once')

        batch.views['r'] = replace(original, block_tables=GuardedTables(original.block_tables))
        for key in (('P', 0), ('D', 0)):
            payload = controller._command_payload(controller.groups['m'], batch, key)
            view = payload['requests'][0]['reservation_view']
            self.assertEqual(view, expected)
            self.assertIs(view['block_tables'], batch.packed['r'])
            self.assertEqual(payload['requests'][0]['rank_block_table'], view['block_tables'][key[1]])
            self.assertEqual(view['block_tables'][0], original.block_tables[0])

    def test_command_owns_one_copy_and_poll_returns_independent_data(self):
        fixture = Fixture()
        copies = []

        class Counted:
            def __deepcopy__(self, memo):
                copies.append(True)
                return self

        group = fixture.controller.groups['m']
        source = {'nested': [[1]], 'copy_probe': Counted()}
        fixture.controller._command(group, ('P', 0), 'TEST', source)
        self.assertEqual(len(copies), 1, 'An unused return value copied the command again')
        source['nested'][0][0] = 9
        first = fixture.controller.poll_commands(fixture.group, 'P', 0, 1001)
        self.assertEqual(first[0]['nested'], [[1]])
        first[0]['nested'][0][0] = 8
        second = fixture.controller.poll_commands(fixture.group, 'P', 0, 1001)
        self.assertEqual(second[0]['nested'], [[1]])


class InitializationBudgetChecks(unittest.TestCase):
    def test_private_warmup_uses_bounded_startup_budget(self):
        fixture = Fixture(initializing=True)
        from muxserve_sglang.contracts import RequestSpec
        spec = RequestSpec('m', 'init', (1, 2), 1, 120000)
        wrapper = _InitializationController(fixture.controller, 700.0)
        with patch('muxserve_sglang.startup.time.monotonic', return_value=100.25):
            result = wrapper.submit_request(spec)
        self.assertEqual(result['deadline'], 699.75)
        self.assertEqual(spec.timeout_ms, 120000, 'Original serving request was mutated')
        self.assertEqual(fixture.controller.config.request_timeout_ms, 120000)
        self.assertEqual(fixture.controller.config.prepare_timeout_ms, 30000)
        self.assertEqual(warmup_shapes(fixture.controller.config, 2048),
                         (('maximum_prefill', 1792, 1), ('maximum_decode', 1792, 256)))

    def test_expired_startup_budget_rejects_admission(self):
        fixture = Fixture(initializing=True)
        from muxserve_sglang.contracts import RequestSpec
        with patch('muxserve_sglang.startup.time.monotonic', return_value=700.0):
            with self.assertRaises(IntegrationError) as caught:
                _InitializationController(fixture.controller, 700.0).submit_request(
                    RequestSpec('m', 'expired', (1, 2), 1))
        self.assertEqual(caught.exception.code, 'STARTUP_TIMEOUT')
        self.assertFalse(fixture.controller.requests)

    def test_private_budget_does_not_open_initialization_admission(self):
        fixture = Fixture(initializing=True)
        with self.assertRaises(IntegrationError) as caught:
            fixture.submit('ordinary')
        self.assertEqual(caught.exception.code, 'NOT_READY')


if __name__ == '__main__':
    unittest.main(verbosity=2)
