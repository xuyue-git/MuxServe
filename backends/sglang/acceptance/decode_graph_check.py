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
    body = [n for n in tree.body if (isinstance(n, ast.FunctionDef) and n.name in ('_native_fields', '_graph_fields'))
            or (isinstance(n, ast.Assign) and any(getattr(t, 'id', '') == 'PREFILL_GRAPH_TOKENS' for t in n.targets))]
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

    def test_pair_hold_needs_its_prerequisites(self):
        pair = dict(max_model_inflight=2, decode_run_merge=True, inline_start=True, inline_prepare=True)
        self.assertEqual(self.config().prefill_pair_hold_ms, 0)
        self.assertEqual(self.config(prefill_pair_hold_ms=5, **pair).prefill_pair_hold_ms, 5)
        for bad in (dict(pair, prefill_pair_hold_ms=51), dict(pair, prefill_pair_hold_ms=True),
                    dict(pair, prefill_pair_hold_ms=5, max_model_inflight=1),
                    dict(pair, prefill_pair_hold_ms=5, decode_run_merge=False),
                    dict(pair, prefill_pair_hold_ms=5, inline_prepare=False)):
            with self.assertRaises(IntegrationError):
                self.config(**bad)

    def test_unparked_start_and_local_handoff_need_their_base_switches(self):
        base = self.config()
        self.assertFalse(base.prefill_start_unparked or base.handoff_local_success)
        wait = dict(decode_step_join=True, handoff_ack_in_poll=True, broker_status_in_poll=True,
                    decode_wait_commands=True)
        good = self.config(decode_run_park=True, prefill_start_unparked=True, handoff_local_success=True, **wait)
        self.assertTrue(good.prefill_start_unparked and good.handoff_local_success)
        for bad in (dict(prefill_start_unparked=True), dict(handoff_local_success=True),
                    dict(decode_run_park=True, prefill_start_unparked=1)):
            with self.assertRaises(IntegrationError):
                self.config(**bad)

    def test_prefill_graph_tokens_need_the_graph_and_increase(self):
        self.assertEqual(tuple(self.config().prefill_graph_tokens), ())
        good = self.config(prefill_cuda_graph=True, prefill_graph_tokens=[64, 128, 512, 2048])
        self.assertEqual(list(good.prefill_graph_tokens), [64, 128, 512, 2048])
        for bad in (dict(prefill_graph_tokens=[64]), dict(prefill_cuda_graph=True, prefill_graph_tokens=[128, 64]),
                    dict(prefill_cuda_graph=True, prefill_graph_tokens=[8]),
                    dict(prefill_cuda_graph=True, prefill_graph_tokens=[64, 64]),
                    dict(prefill_cuda_graph=True, prefill_graph_tokens="512")):
            with self.assertRaises(IntegrationError):
                self.config(**bad)

    def test_step_join_flags_default_off_and_per_page_needs_join(self):
        config = self.config()
        self.assertFalse(config.decode_step_join or config.decode_kv_per_page)
        both = self.config(decode_step_join=True, decode_kv_per_page=True)
        self.assertTrue(both.decode_step_join and both.decode_kv_per_page)
        leave = self.config(decode_step_join=True, decode_leader_reports=True, decode_step_leave=True,
                            decode_kv_grow_pages=4)
        self.assertTrue(leave.decode_step_leave and leave.decode_kv_grow_pages == 4)
        for bad in (dict(decode_kv_per_page=True), dict(decode_step_join=1),
                    dict(decode_step_join=True, decode_step_leave=True),
                    dict(decode_kv_grow_pages=9), dict(decode_kv_grow_pages=True),
                    dict(decode_step_pipeline=True)):
            with self.assertRaises(IntegrationError):
                self.config(**bad)

    def test_park_flag_defaults_off_and_must_be_boolean(self):
        self.assertFalse(self.config().decode_run_park)
        self.assertTrue(self.config(decode_run_park=True).decode_run_park)
        with self.assertRaises(IntegrationError):
            self.config(decode_run_park=1)

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


class PrefillGraphChecks(unittest.TestCase):
    def config(self, **extra):
        return DecodeGraphConfigChecks.config(self, **extra)

    def test_prefill_graph_flag_and_launcher_fields(self):
        self.assertFalse(self.config().prefill_cuda_graph)
        self.assertTrue(self.config(prefill_cuda_graph=True).prefill_cuda_graph)
        with self.assertRaises(IntegrationError):
            self.config(prefill_cuda_graph=1)
        ns = launcher_functions()
        model = SimpleNamespace(model_path='p', tokenizer_path='t', revision='r', prefill_port=1,
                                decode_port=2, bootstrap_port=3)
        cfg = SimpleNamespace(tp_size=4, request_timeout_ms=20000, max_model_inflight=1, queue_limit=64,
                              max_context_tokens=2048, decode_cuda_graph=True, prefill_cuda_graph=True)
        p = ns['_native_fields'](cfg, model, 'P')
        d = ns['_native_fields'](cfg, model, 'D')
        self.assertFalse(p['disable_cuda_graph'] or p['disable_prefill_cuda_graph'])
        self.assertEqual((p['cuda_graph_backend_prefill'], p['cuda_graph_bs_prefill']), ('breakable', [512]))
        cfg.prefill_pair_hold_ms = 5
        self.assertEqual(ns['_native_fields'](cfg, model, 'P')['cuda_graph_bs_prefill'], [512, 1024])
        cfg.prefill_pair_hold_ms = 0
        cfg.prefill_graph_tokens = (64, 128, 256, 512, 1024, 2048)
        self.assertEqual(ns['_native_fields'](cfg, model, 'P')['cuda_graph_bs_prefill'],
                         [64, 128, 256, 512, 1024, 2048])
        cfg.prefill_graph_tokens = ()
        self.assertTrue(p['disable_decode_cuda_graph'])
        self.assertEqual(p['cuda_graph_backend_decode'], 'disabled')
        self.assertEqual((d['cuda_graph_backend_decode'], d['cuda_graph_backend_prefill']), ('full', 'disabled'))
        cfg.prefill_cuda_graph = False
        p = ns['_native_fields'](cfg, model, 'P')
        self.assertTrue(p['disable_cuda_graph'] and p['disable_prefill_cuda_graph'])


class FakeTensor:
    def __init__(self, shape, name='t'):
        self.shape, self.name, self.dtype, self.device = tuple(shape), name, 'fp16', 'cuda'

    def reshape(self, *shape):
        total = 1
        for size in self.shape:
            total *= size
        known = 1
        for size in shape:
            known *= size if size != -1 else 1
        return FakeTensor(tuple(total // known if size == -1 else size for size in shape), self.name)

    def contiguous(self):
        return self

    def transpose(self, a, b):
        shape = list(self.shape)
        shape[a], shape[b] = shape[b], shape[a]
        return FakeTensor(shape, self.name)

    def unsqueeze(self, dim):
        shape = list(self.shape)
        shape.insert(dim, 1)
        return FakeTensor(shape, self.name)

    def squeeze(self, dim):
        shape = list(self.shape)
        shape.pop(dim)
        return FakeTensor(shape, self.name)


def attention_backend(stores, breakable):
    tree = ast.parse((ROOT / 'muxserve_sglang/sglang_attention.py').read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'HeadwiseAttentionBackend')
    cls.body = [n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name in (
        'forward', '_forward_extend_graph_break', '_extend_attention', '_validate_layer', 'init_forward_metadata')]
    cls.bases = []
    torch = SimpleNamespace(Tensor=FakeTensor, float16='fp16',
                            cuda=SimpleNamespace(is_current_stream_capturing=lambda: False),
                            cat=lambda parts, dim=0: FakeTensor((sum(p.shape[0] for p in parts),) + parts[0].shape[1:]))
    namespace = dict(torch=torch, math=__import__('math'),
                     is_in_breakable_cuda_graph=lambda: breakable[0],
                     scaled_dot_product_attention=lambda q, k, v, **kw: FakeTensor(q.shape, 'out'),
                     ops=SimpleNamespace(store=lambda k, v, key, value, slots: stores.append(slots)),
                     AttentionType=SimpleNamespace(DECODER='decoder'), IntegrationError=IntegrationError)
    def require(condition, message, code='INTEGRATION_ERROR'):
        if not condition:
            raise IntegrationError(code, message)
    namespace['_require'] = require
    exec(compile(ast.Module(body=[cls], type_ignores=[]), 'attention-graph-break', 'exec'), namespace)
    return namespace['HeadwiseAttentionBackend']


class PrefillGraphBreakAttentionChecks(unittest.TestCase):
    def setup(self, bound=True):
        stores, breakable = [], [True]
        backend = attention_backend(stores, breakable).__new__(attention_backend(stores, breakable))
        pool, _, live = fixture(Uploads(), 'PREFILL', (33,), layers=3)
        if not bound:
            pool._bound = None
        pool.pool_key = pool.pool_value = 'pool'
        pool.device = 'cuda'
        backend.token_to_kv_pool, backend._validated_layers, backend._static = pool, set(), False
        layer = SimpleNamespace(layer_id=1, xai_temperature_len=-1, tp_q_head_num=3, tp_k_head_num=3,
                                tp_v_head_num=3, head_dim=128, qk_head_dim=128, v_head_dim=128,
                                attn_type='decoder', is_cross_attention=False, pos_encoding_mode='NONE',
                                use_irope=False, logit_cap=0, sliding_window_size=-1, scaling=0.1)
        static = SimpleNamespace(forward_mode=SimpleNamespace(is_decode=lambda: False), batch_size=1,
                                 global_num_token_non_padded_cpu=33)
        return backend, pool, live, static, layer, stores

    def qkv(self, rows=33):
        return [FakeTensor((rows, 3 * 128), n) for n in 'qkv']

    def test_capture_runs_attention_without_storing(self):
        backend, pool, _, static, layer, stores = self.setup(bound=False)
        backend.init_forward_metadata(static)
        self.assertIsNone(backend.forward_metadata)
        out = backend.forward(*self.qkv(512), layer, static, True)
        self.assertEqual((out.shape, stores), ((512, 384), []))

    def test_replay_stores_the_granted_slots_and_checks_geometry(self):
        backend, pool, live, static, layer, stores = self.setup()
        backend.init_forward_metadata(live)  # replay prep validates the live batch
        out = backend.forward(*self.qkv(), layer, static, True)
        self.assertEqual(out.shape, (33, 384))
        self.assertEqual(len(stores), 1)
        self.assertIs(stores[0], pool.layer_slots(1))
        for change in (dict(batch_size=2), dict(global_num_token_non_padded_cpu=32)):
            with self.subTest(change=change):
                with self.assertRaises(IntegrationError):
                    backend.forward(*self.qkv(), layer, SimpleNamespace(**{**vars(static), **change}), True)
        with self.assertRaises(IntegrationError):
            backend.forward(*self.qkv(32), layer, static, True)
        with self.assertRaises(IntegrationError):
            backend.forward(*self.qkv(), layer, static, True, sinks='x')

    def test_replay_requires_a_validated_prefill_grant(self):
        backend, pool, _, static, layer, _ = self.setup()
        pool.bound_forward.extend_checked = False
        with self.assertRaises(IntegrationError):
            backend.forward(*self.qkv(), layer, static, True)
        decode, _, _ = fixture(Uploads(), 'DECODE', (33,), layers=3)
        decode.pool_key = decode.pool_value = 'pool'
        backend.token_to_kv_pool = decode
        with self.assertRaises(IntegrationError):
            backend.forward(*self.qkv(1), layer, static, True)


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
