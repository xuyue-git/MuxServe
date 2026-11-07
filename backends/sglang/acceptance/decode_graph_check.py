"""CPU checks for D-stage decode CUDA graphs: config/launcher wiring and the
replay metadata (identical slots/tables to the eager per-layer metadata)."""
import ast
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest

import numpy

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'acceptance')]
from metadata_upload_check import Uploads, fixture  # noqa: E402
from muxserve_sglang.config import load_config  # noqa: E402
from muxserve_sglang.contracts import IntegrationError  # noqa: E402


def launcher_functions():
    tree = ast.parse((ROOT / 'muxserve_sglang/launcher.py').read_text())
    body = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in ('_native_fields', '_graph_fields')]
    namespace = {}
    exec(compile(ast.Module(body=body, type_ignores=[]), 'launcher-fields', 'exec'), namespace)
    return namespace


class DecodeGraphConfigChecks(unittest.TestCase):
    def config(self, **extra):
        with open(ROOT / 'configs/sglang_a100.json') as handle:
            raw = json.load(handle)
        raw.update(extra)
        directory = tempfile.mkdtemp()
        raw['asset_lock_dir'] = raw['runtime_dir'] = directory
        path = Path(directory) / 'c.json'
        path.write_text(json.dumps(raw))
        return load_config(path, require_assets=False)

    def test_flag_defaults_off_and_must_be_boolean(self):
        self.assertFalse(self.config().decode_cuda_graph)
        self.assertTrue(self.config(decode_cuda_graph=True).decode_cuda_graph)
        with self.assertRaises(IntegrationError):
            self.config(decode_cuda_graph=1)

    def test_only_d_stage_enables_full_decode_graphs(self):
        ns = launcher_functions()
        model = SimpleNamespace(model_path='p', tokenizer_path='t', revision='r', prefill_port=1,
                                decode_port=2, bootstrap_port=3)
        for enabled in (False, True):
            cfg = SimpleNamespace(tp_size=4, request_timeout_ms=20000, max_model_inflight=2, queue_limit=64,
                                  max_context_tokens=2048, decode_cuda_graph=enabled)
            p = ns['_native_fields'](cfg, model, 'P')
            d = ns['_native_fields'](cfg, model, 'D')
            self.assertTrue(p['disable_cuda_graph'] and p['disable_decode_cuda_graph'])
            self.assertTrue(d['disable_prefill_cuda_graph'])
            if enabled:
                self.assertFalse(d['disable_cuda_graph'] or d['disable_decode_cuda_graph'])
                self.assertEqual((d['cuda_graph_backend_decode'], d['cuda_graph_bs_decode']), ('full', [1, 2]))
                self.assertEqual(d['cuda_graph_backend_prefill'], 'disabled')
            else:
                self.assertTrue(d['disable_cuda_graph'] and d['disable_decode_cuda_graph'])
                self.assertNotIn('cuda_graph_bs_decode', d)


class GraphReplayMetadataChecks(unittest.TestCase):
    def test_replay_slots_and_tables_match_eager_layer_metadata(self):
        for lengths in ((17, 33), (16,), (1792, 17), (2048,)):
            with self.subTest(lengths=lengths):
                pool, _, _ = fixture(Uploads(), 'DECODE', lengths, layers=5, start_layer=1, end_layer=3)
                bound = pool.bound_forward
                graph_replay_arrays = pool.bind_forward.__func__.__globals__['graph_replay_arrays']
                pages = pool.bind_forward.__func__.__globals__['GRAPH_PAGES']
                tables, slots, lens = graph_replay_arrays(bound, 1, 3, pool.head_num, True)
                self.assertEqual(lens.tolist(), list(lengths))
                self.assertEqual(tables.shape, (3, len(lengths), pool.head_num, pages))
                for layer in (1, 2, 3):
                    eager_slots, eager_tables = pool.layer_metadata(layer)
                    self.assertEqual(slots[layer - 1].tolist(), eager_slots.tolist())
                    width = len(eager_tables.tolist()[0][0])
                    self.assertEqual(tables[layer - 1][:, :, :width].tolist(), eager_tables.tolist())
                    self.assertFalse(tables[layer - 1][:, :, width:].any())
                none_tables, again, _ = graph_replay_arrays(bound, 1, 3, pool.head_num, False)
                self.assertIsNone(none_tables)
                self.assertTrue(numpy.array_equal(again, slots))

    def test_replay_rejects_non_decode_or_oversized_context(self):
        pool, _, _ = fixture(Uploads(), 'PREFILL', (17,), layers=2)
        graph_replay_arrays = pool.bind_forward.__func__.__globals__['graph_replay_arrays']
        with self.assertRaises(IntegrationError):
            graph_replay_arrays(pool.bound_forward, 0, 1, pool.head_num, True)
        pool, _, _ = fixture(Uploads(), 'DECODE', (17,), layers=2)
        bound = pool.bound_forward
        bound.target_lengths = (2049,)
        with self.assertRaises(IntegrationError):
            graph_replay_arrays(bound, 0, 1, pool.head_num, True)


if __name__ == '__main__':
    unittest.main()
