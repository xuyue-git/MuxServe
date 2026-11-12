"""CPU checks: startup does short real-lifecycle warmup; capacity is opt-in."""
import os
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
from muxserve import startup
from muxserve.muxsched import driver as controller_driver
from muxserve.contracts import IntegrationError


class StartupSmokeChecks(unittest.TestCase):
    def test_short_shapes_and_explicit_capacity_shapes_are_distinct(self):
        config = fixtures.SimpleNamespace(max_input_tokens=1792, max_new_tokens=256, max_context_tokens=2048)
        self.assertEqual(startup.warmup_shapes(config, 2048, mode='smoke'),
                         (('smoke_prefill', 32, 1), ('smoke_decode', 32, 8)))
        self.assertEqual(startup.warmup_shapes(config, 2048, mode='smoke', batch_size=2),
                         (('smoke_prefill', 32, 1), ('smoke_decode', 32, 8)))
        self.assertEqual(startup.warmup_shapes(config, 2048, mode='capacity'),
                         (('maximum_prefill', 1792, 1), ('maximum_decode', 1792, 256)))
        with self.assertRaises(IntegrationError):
            startup.warmup_shapes(config, 2048, mode='unknown')

    def run_fixture(self, mode=None):
        owner, driver_type, drivers, sampler, tokenizer = fixtures.DeploymentChecks().initialization_fixture()
        owner.config.max_input_tokens = 1792
        owner.config.max_new_tokens = 256
        owner.config.max_context_tokens = 2048
        evidence = []
        options = {} if mode is None else {'warmup_profile': mode}
        with patch.object(controller_driver, 'ControllerDriver', driver_type), \
                patch.object(startup, 'MemorySampler', return_value=sampler), patch.dict(os.environ):
            os.environ.pop('MUXSERVE_STARTUP_WARMUP', None)
            state, driver = startup.run_initialization(owner, {'m': tokenizer}, [],
                time.monotonic() + 10, lambda: None, Event(), evidence.append, **options)
        return owner, evidence[-1]

    def test_default_initialization_is_short_drained_and_does_not_claim_capacity(self):
        owner, evidence = self.run_fixture()
        specs = [call.args[0] for call in owner.submit_initialization_request.call_args_list]
        self.assertEqual([(len(spec.input_ids), spec.max_new_tokens) for spec in specs], [(32, 1), (32, 8)])
        self.assertEqual(evidence['warmup_profile'], 'smoke')
        self.assertFalse(evidence['maximum_load_validated'])
        self.assertTrue(evidence['passed'])
        self.assertTrue(all(profile['finished_and_forgotten'] for profile in evidence['profiles']))
        self.assertEqual(evidence['profiles'][-1]['started_batch_size'], {'m:P': 1, 'm:D': 1})
        self.assertFalse(owner.requests)
        owner.submit_request.assert_not_called()

    def test_explicit_capacity_mode_keeps_existing_capacity_admission(self):
        owner, evidence = self.run_fixture('capacity')
        specs = [call.args[0] for call in owner.submit_initialization_request.call_args_list]
        # This fixture owns only 64 physical tokens. The independent shape test
        # above verifies the production 2048-token capacity profile unchanged.
        self.assertEqual([(len(spec.input_ids), spec.max_new_tokens) for spec in specs], [(64, 1), (1, 64)])
        self.assertEqual(evidence['warmup_profile'], 'capacity')
        self.assertTrue(evidence['maximum_load_requested'])
        self.assertFalse(evidence['maximum_load_validated'], '64-token fixture cannot validate 1792-token maximum load')
        self.assertTrue(evidence['passed'])


if __name__ == '__main__':
    unittest.main(verbosity=2)
