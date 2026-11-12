"""KV pool sized like original MuxServe (gpu_memory_utilization x total memory)."""
from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from muxserve.config import kv_bytes_from_fraction, load_config
from muxserve.contracts import IntegrationError


def original_flexstore_bytes(utilization, total_memory):
    # Verbatim arithmetic of muxserve/flexstore/manager.py get_num_gpu_blocks()
    # with head_size=128, block_size=32 (MuxServeArgs default) and FP16.
    available = round(utilization * total_memory) - 1
    key_shape = (128 // (16 // 2), 32, 16 // 2)
    value_shape = (128, 32)
    block_mem = 2 * (key_shape[0] * key_shape[1] * key_shape[2] + value_shape[0] * value_shape[1])
    blocks = (available // block_mem // 128) * 128
    return blocks * block_mem, blocks


class KVFractionChecks(unittest.TestCase):
    def test_matches_original_formula_and_maps_to_16_token_blocks(self):
        for total in (42_505_273_344, 42_949_672_960, 34_089_730_048, 85_899_345_920):
            for fraction in (0.01, 0.025, 0.05, 0.08, 0.1, 0.4):
                expected, blocks = original_flexstore_bytes(fraction, total)
                got = kv_bytes_from_fraction(fraction, total)
                self.assertEqual(got, expected)
                self.assertEqual(got % 8192, 0)
                self.assertEqual(got // 8192, 2 * blocks)

    def _write(self, directory, **fields):
        source = json.loads((ROOT / "configs/sglang_a100.json").read_text())
        source.pop("kv_bytes_per_rank")
        source.update(fields)
        path = Path(directory) / "config.json"
        path.write_text(json.dumps(source))
        return path

    def test_config_accepts_fraction_and_rejects_ambiguity(self):
        with tempfile.TemporaryDirectory() as directory:
            config = load_config(self._write(directory, kv_memory_fraction=0.05), require_assets=False)
            self.assertEqual(config.kv_memory_fraction, 0.05)
            for bad in (0, 1, 1.5, -0.1, True, "0.1"):
                with self.assertRaises((IntegrationError, TypeError)):
                    load_config(self._write(directory, kv_memory_fraction=bad), require_assets=False)
            with self.assertRaises(IntegrationError):
                load_config(self._write(directory, kv_memory_fraction=0.05,
                                        kv_bytes_per_rank=2 ** 31), require_assets=False)
            # The launcher writes the resolved runtime config this way.
            resolved = load_config(self._write(directory, kv_memory_fraction=None,
                                               kv_bytes_per_rank=kv_bytes_from_fraction(0.05, 42_505_273_344)),
                                   require_assets=False)
            self.assertIsNone(resolved.kv_memory_fraction)


if __name__ == "__main__":
    unittest.main(verbosity=2)
