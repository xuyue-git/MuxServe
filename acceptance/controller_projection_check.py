"""CPU regression: ordinary P/D feedback reads counters, not physical KV tables."""
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'acceptance'))
from controller_check import Fixture
from muxserve.flexstore.storage import HeadwiseStore


class ObservedStore(HeadwiseStore):
    snapshots = 0

    def snapshot(self):
        self.snapshots += 1
        return super().snapshot()


class CounterProjectionChecks(unittest.TestCase):
    def test_prefill_decode_feedback_preserves_counts_without_copying_kv_tables(self):
        for mode in ('shared', 'static'):
            with self.subTest(pool_mode=mode):
                store = ObservedStore({0: 64}, pool_mode=mode, model_ids=('m',))
                f = Fixture(store=store)
                store.snapshots = 0  # Initialization still validates the complete pool.
                f.submit(max_new=2)
                p, batch = f.prepare()
                f.ready_start(p, batch)
                f.stage_prefill(p)
                f.gpu(p, batch, 4, 2)
                f.handoff(p)
                f.batch_quiesce(p, batch)
                d, batch = f.prepare(stage='D', batch_id='decode')
                f.ready_start(d, batch)
                f.gpu(d, batch, 5, 3)
                f.batch_quiesce(d, batch)
                f.cleanup('r')

                self.assertEqual(f.controller.request_output('m', 'r')['output_token_ids'], (4, 5))
                self.assertEqual(f.controller.request_output('m', 'r')['state'], 'FINISHED')
                self.assertEqual(store.rank_usage()[0]['free_blocks'], 64)
                feedback = list(f.controller.policy.feedback_history)
                self.assertEqual([item.stage for item in feedback], ['P', 'D'])
                self.assertEqual([item.reserved_blocks_by_rank for item in feedback], [{0: 1}, {0: 0}])
                self.assertTrue(all(item.succeeded for item in feedback))
                self.assertEqual([item.gpu_forward_ms for item in feedback], [1.0, 1.0])
                self.assertEqual(store.snapshots, 0,
                    'Ordinary forward feedback unnecessarily materialized full KV block tables')
                self.assertTrue(store.assert_integrity()['ok'])


    def test_forget_and_metrics_do_not_copy_kv_tables(self):
        store = ObservedStore({0: 64}, model_ids=('m',))
        f = Fixture(store=store)
        f.submit(max_new=1)
        p, batch = f.prepare()
        f.ready_start(p, batch)
        f.stage_prefill(p)
        f.gpu(p, batch, 4, 2)
        f.handoff(p)
        f.batch_quiesce(p, batch)
        f.cleanup('r')
        events = f.controller.poll_events(f.group)
        f.controller.ack_events(f.group, events[-1].event_seq)
        f.ack_all_commands()
        store.snapshots = 0
        f.controller.metrics()
        f.controller.forget_request('m', 'r', 0)
        self.assertEqual(store.snapshots, 0, 'forget/metrics materialized every live KV block table')
        self.assertEqual(f.controller.metrics()['active_requests'], 0)


if __name__ == '__main__':
    unittest.main(verbosity=2)
