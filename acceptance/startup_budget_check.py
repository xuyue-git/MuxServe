"""CPU checks for the 1 GiB startup profile and early physical-memory refusal."""
from dataclasses import replace
import json
from pathlib import Path
import sys
from threading import Event
import time
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'acceptance'))
import deployment_check as fixtures
from muxserve.config import DeploymentConfig, ModelDeployment
from muxserve.contracts import IntegrationError, KVTarget, RequestIdentity
from muxserve.flexstore.storage import HeadwiseStore
from muxserve import startup
from muxserve.muxsched import driver as controller_driver


class StartupBudgetChecks(unittest.TestCase):
    def test_one_gib_keeps_full_shapes_and_real_single_request_capacity(self):
        raw = json.loads((ROOT / 'configs/sglang_a100.json').read_text())
        models = tuple(ModelDeployment(**item) for item in raw.pop('models'))
        config = replace(DeploymentConfig(models=models, **raw), kv_bytes_per_rank=1024**3)
        store = HeadwiseStore(dict.fromkeys(range(4), config.pool_blocks_per_rank))
        needed_bytes = {}
        for model in config.models:
            layout = model.layout(4)
            store.register_model(layout)
            capacity = config.pool_blocks_per_rank // layout.blocks_per_token_group * 16
            self.assertGreaterEqual(capacity, 2048)
            self.assertEqual(startup.warmup_shapes(config, min(2048, capacity)),
                             (('maximum_prefill', 1792, 1), ('maximum_decode', 1792, 256)))
            reservation = store.reserve_kv(model.model_id,
                [KVTarget(RequestIdentity(model.model_id, 'full-profile'), 2047)], 'cpu-check')
            required_blocks = layout.blocks_per_token_group * 128
            self.assertEqual(reservation.views[0].capacity_tokens, 2048)
            self.assertTrue(all(row['used_blocks'] == required_blocks for row in store.rank_usage().values()))
            needed_bytes[model.model_id] = required_blocks * 8192
            store.abort_kv_batch(reservation.reservation_id)
            self.assertFalse(store.snapshot()['allocations'])
            store.assert_integrity()
        self.assertEqual(needed_bytes['llama30b'], 780 * 1024**2)
        self.assertLessEqual(max(needed_bytes.values()), config.kv_bytes_per_rank)
        # This private smoke profile does not claim all three maximum requests
        # can simultaneously occupy the pool without capacity recovery.
        self.assertGreater(sum(needed_bytes.values()), config.kv_bytes_per_rank)
        self.assertEqual(config.extra_headroom_bytes_per_rank, 2 * 1024**3)
        self.assertEqual((config.request_timeout_ms, config.prepare_timeout_ms), (120000, 30000))

    def test_low_memory_stops_an_unfinished_profile_before_waiting_for_timeout(self):
        owner, driver_type, drivers, sampler, tokenizer = fixtures.DeploymentChecks().initialization_fixture()
        evidence = []

        class BusyDriver(driver_type):
            def step(self):
                # The real request remains queued. The sampled physical reserve
                # has already failed, so waiting out its deadline is pointless.
                sampler.minimum_free['GPU-a'] = 1024**3

        with patch.object(controller_driver, 'ControllerDriver', BusyDriver), \
                patch.object(startup, 'MemorySampler', return_value=sampler):
            with self.assertRaises(IntegrationError) as caught:
                startup.run_initialization(owner, {'m': tokenizer}, [],
                    time.monotonic() + 0.2, lambda: None, Event(), evidence.append)
        self.assertEqual(caught.exception.code, 'STARTUP_HEADROOM_INSUFFICIENT')
        self.assertEqual(evidence[-1]['failure']['code'], 'STARTUP_HEADROOM_INSUFFICIENT')
        self.assertFalse(evidence[-1]['passed'])
        self.assertTrue(owner.requests, 'Unfinished ownership was discarded')
        self.assertFalse(any(profile.get('finished_and_forgotten') for profile in evidence[-1]['profiles']))
        sampler.stop.assert_called()


if __name__ == '__main__':
    unittest.main(verbosity=2)
