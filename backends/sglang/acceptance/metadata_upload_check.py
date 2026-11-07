"""CPU-only index/lifecycle checks of the actual Head-wise adapter methods.

No Torch/SGLang import, GPU context, model or installed dependency is needed.
The tiny tensor double stores nested scalar values and records upload calls;
it does not establish GPU numerical correctness or predict GPU speedup.
"""
from __future__ import annotations

import ast
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
from typing import Sequence
import unittest

import numpy

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from muxserve_sglang.contracts import IntegrationError, KVView, ModelLayout, RequestIdentity, PackedBlocks


class Tensor:
    def __init__(self, values, *, dtype, device='cpu'):
        if hasattr(values, 'tolist') and not isinstance(values, (list, tuple)):
            values = values.tolist()  # numpy arrays upload like nested lists
        self.values = deepcopy(values)
        self.dtype = dtype
        self.device = SimpleNamespace(type=str(device))
        def shape(value):
            if not isinstance(value, (list, tuple)):
                return ()
            child = shape(value[0]) if value else ()
            if any(shape(item) != child for item in value):
                raise ValueError('ragged tensor fixture')
            return (len(value),) + child
        self.shape = shape(values)
        self.ndim = len(self.shape)

    def tolist(self):
        def lists(value):
            return [lists(item) for item in value] if isinstance(value, (list, tuple)) else value
        return lists(self.values)

    def numel(self):
        value = 1
        for count in self.shape:
            value *= count
        return value

    def __getitem__(self, index):
        return Tensor(self.values[index], dtype=self.dtype, device=self.device.type)

    def is_contiguous(self):
        return True


class Uploads:
    Tensor = Tensor
    int32, int64 = 'int32', 'int64'

    def __init__(self):
        self.calls = []

    def tensor(self, values, *, dtype, device):
        result = Tensor(values, dtype=dtype, device=device)
        self.calls.append(result)
        return result


def adapter(upload):
    """Compile actual source nodes, excluding CUDA-dependent imports/init."""
    source = (ROOT / 'muxserve_sglang/sglang_memory.py').read_text(encoding='utf-8')
    tree = ast.parse(source)
    native = next(node for node in tree.body
                  if isinstance(node, ast.ClassDef) and node.name == 'HeadwiseTokenToKVPool')
    methods = [node for node in native.body if isinstance(node, ast.FunctionDef)
               and node.name != '__init__']
    selected = [node for node in tree.body if getattr(node, 'name', '')
                in ('_require', '_cpu_ints', 'BoundForward', 'graph_replay_arrays')
                or (isinstance(node, ast.Assign) and all(
                    isinstance(t, ast.Name) and t.id.startswith('GRAPH_') for t in node.targets))]
    cls = ast.ClassDef(name='Pool', bases=[], keywords=[], body=methods, decorator_list=[])
    selected.append(cls)
    module = ModuleType('_metadata_adapter_check')
    sys.modules[module.__name__] = module
    namespace = module.__dict__
    namespace.update(dataclass=dataclass, field=field, Sequence=Sequence, torch=upload, np=numpy,
                     IntegrationError=IntegrationError, KVView=KVView, ModelLayout=ModelLayout,
                     PackedBlocks=PackedBlocks)
    text = 'from __future__ import annotations\n' + '\n'.join(ast.unparse(node) for node in selected)
    exec(compile(text, 'actual-sglang-memory-methods', 'exec'), namespace)
    return namespace['Pool']


def fixture(upload, mode='DECODE', lengths=(17, 33), *, layers=5, start_layer=0, end_layer=None, packed=False):
    heads, ranks = 3, (0, 1, 2, 3)
    layout = ModelLayout('m', layers, heads, ranks)
    cursor, views = 0, []
    for index, target in enumerate(lengths):
        pages = (target + 15) // 16
        # Reverse physical pages and offset each layer/head/request. Distinct
        # rank mappings must remain disjoint within each rank, not across GPUs.
        table = []
        for _ in range(layers):
            selected = []
            for _ in range(heads):
                selected.append(list(range(cursor, cursor + pages))[::-1])
                cursor += pages
            table.append(selected)
        rank_tables = {rank: [[[block + rank * 10000 for block in blocks]
                              for blocks in local_heads] for local_heads in table]
                       for rank in ranks}
        valid = target - 1 if mode == 'DECODE' else (7 if mode == 'REBUILD' else 0)
        if packed:
            rank_tables = {rank: PackedBlocks.pack(t) for rank, t in rank_tables.items()}
        views.append(KVView('a-' + str(index), RequestIdentity('m', 'r-' + str(index)),
                            target, valid, 16, rank_tables))
    pool = adapter(upload)()
    pool._bound = None
    pool.layout, pool.rank = layout, 2
    pool.layer_num, pool.head_num = layers, heads
    pool.start_layer = start_layer
    pool.end_layer = layers - 1 if end_layer is None else end_layer
    pool.physical_block_count = 40000
    pool.device = 'cuda'
    pool.bind_forward(views, list(range(len(views))), lengths, mode)
    bound = pool.bound_forward
    forward = SimpleNamespace(forward_mode=SimpleNamespace(name='DECODE' if mode == 'DECODE' else 'EXTEND'),
        batch_size=len(views), req_pool_indices_cpu=list(range(len(views))),
        seq_lens_cpu=list(lengths), input_ids=Tensor([0] * bound.total_written, dtype='int64'),
        out_cache_loc=Tensor([0] * bound.total_written, dtype='int64'),
        extend_seq_lens_cpu=list(bound.write_lengths), extend_prefix_lens_cpu=[0] * len(views),
        extend_num_tokens=bound.total_written)
    pool.validate_forward(forward)
    return pool, views, forward


def legacy_metadata(bound, layer):
    """Frozen pre-optimization scalar indexing, independent of tensor uploads."""
    slots, tables = [], []
    pages = (max(bound.target_lengths) + 15) // 16
    for row, (start, count, target) in enumerate(zip(
            bound.write_starts, bound.write_lengths, bound.target_lengths)):
        heads = bound.tables[row][layer]
        for position in range(start, start + count):
            slots.append([blocks[position // 16] * 16 + position % 16 for blocks in heads])
        tables.append([list(blocks[:(target + 15) // 16])
                       + [0] * (pages - (target + 15) // 16) for blocks in heads])
    return slots, tables


class MetadataUploadChecks(unittest.TestCase):
    def test_one_forward_uploads_metadata_twice_for_all_layers(self):
        uploads = Uploads()
        pool, _, _ = fixture(uploads, layers=60)
        before = len(uploads.calls)
        for layer in range(60):
            first = pool.layer_metadata(layer)
            self.assertIs(first, pool.layer_metadata(layer))
        self.assertEqual(len(uploads.calls) - before, 2,
                         'Per-layer H2D uploads must be batched once per forward')

    def test_decode_extend_rebuild_indices_match_legacy_with_padding_and_layer_offsets(self):
        for mode in ('DECODE', 'PREFILL', 'REBUILD'):
            for lengths in ((15, 16, 17, 33), (128, 143), (1792, 17)):
                with self.subTest(mode=mode, lengths=lengths):
                    uploads = Uploads()
                    pool, _, _ = fixture(uploads, mode, lengths, layers=5, start_layer=1, end_layer=3)
                    bound = pool.bound_forward
                    # Request layers out of order; physical mapping must still
                    # match absolute layer indices and every original scalar.
                    for layer in (3, 1, 2):
                        slots, tables = pool.layer_metadata(layer)
                        expected_slots, expected_tables = legacy_metadata(bound, layer)
                        self.assertEqual(slots.tolist(), expected_slots)
                        self.assertEqual(tables.tolist(), expected_tables)
                        self.assertEqual(slots.dtype, 'int64')
                        self.assertEqual(tables.dtype, 'int32')
                        self.assertTrue(slots.is_contiguous())
                        self.assertTrue(tables.is_contiguous())
                    for layer in (0, 4):
                        with self.assertRaises(IntegrationError):
                            pool.layer_metadata(layer)

    def test_packed_tables_produce_identical_layer_metadata(self):
        for mode in ('DECODE', 'PREFILL', 'REBUILD'):
            for lengths in ((15, 16, 17, 33), (1792, 17)):
                with self.subTest(mode=mode, lengths=lengths):
                    plain, _, _ = fixture(Uploads(), mode, lengths, layers=5, start_layer=1, end_layer=3)
                    packed, _, _ = fixture(Uploads(), mode, lengths, layers=5, start_layer=1, end_layer=3,
                                           packed=True)
                    for layer in (3, 1, 2):
                        a_slots, a_tables = plain.layer_metadata(layer)
                        b_slots, b_tables = packed.layer_metadata(layer)
                        self.assertEqual(a_slots.tolist(), b_slots.tolist())
                        self.assertEqual(a_tables.tolist(), b_tables.tolist())
                        self.assertEqual(b_slots.dtype, 'int64')
                        self.assertEqual(b_tables.dtype, 'int32')
                        expected_slots, expected_tables = legacy_metadata(packed.bound_forward, layer)
                        self.assertEqual(b_slots.tolist(), expected_slots)
                        self.assertEqual(b_tables.tolist(), expected_tables)

    def test_forward_identity_and_clear_rebind_do_not_reuse_old_metadata(self):
        uploads = Uploads()
        pool, views, forward = fixture(uploads)
        first_bound = pool.bound_forward
        first_slots = pool.layer_metadata(0)[0]
        context_lens = first_bound.context_lens
        self.assertIs(pool.validate_forward(forward).context_lens, context_lens)
        self.assertEqual(context_lens.tolist(), [17, 33])
        forward.seq_lens_cpu[0] += 1
        with self.assertRaises(IntegrationError):
            pool.validate_forward(forward)
        forward.seq_lens_cpu[0] -= 1
        wrong_forward = deepcopy(forward)
        with self.assertRaises(IntegrationError):
            pool.validate_forward(wrong_forward)
        pool.clear_forward()
        with self.assertRaises(IntegrationError):
            pool.layer_metadata(0)
        changed = deepcopy(views)
        for view in changed:
            for table in view.block_tables.values():
                for heads in table:
                    for blocks in heads:
                        blocks[:] = [block + 1000 for block in blocks]
        pool.bind_forward(changed, (0, 1), (17, 33), 'DECODE')
        with self.assertRaises(IntegrationError):
            pool.layer_metadata(0)
        pool.validate_forward(forward)
        self.assertIsNot(pool.bound_forward.context_lens, context_lens)
        second = pool.layer_metadata(0)[0]
        self.assertIsNot(first_bound, pool.bound_forward)
        self.assertIsNot(first_slots, second)
        self.assertNotEqual(first_slots.tolist(), second.tolist())
        self.assertEqual(second.tolist(), legacy_metadata(pool.bound_forward, 0)[0])

    def test_nonlocal_rank_alias_and_out_of_pool_indices_are_still_rejected(self):
        for corrupt in ('alias', 'outside'):
            uploads = Uploads()
            pool, views, _ = fixture(uploads)
            pool.clear_forward()
            changed = deepcopy(views)
            mapping = changed[0].block_tables[0]
            mapping[0][0][0] = mapping[0][1][0] if corrupt == 'alias' else pool.physical_block_count
            with self.subTest(corrupt=corrupt), self.assertRaises(IntegrationError):
                pool.bind_forward(changed, (0, 1), (17, 33), 'DECODE')
            self.assertIsNone(pool._bound)


def run_fixture(upload, start_lengths=(17, 30), steps=20, layers=4):
    """DECODE bind whose views reserve ``steps`` more tokens than the first step."""
    heads, ranks = 3, (0, 1, 2, 3)
    layout = ModelLayout('m', layers, heads, ranks)
    cursor, views = 0, []
    for index, target in enumerate(start_lengths):
        capacity = ((target + steps - 1 + 15) // 16) * 16
        pages = capacity // 16
        table = []
        for _ in range(layers):
            selected = []
            for _ in range(heads):
                selected.append(list(range(cursor, cursor + pages))[::-1])
                cursor += pages
            table.append(selected)
        rank_tables = {rank: [[[block + rank * 10000 for block in blocks] for blocks in local]
                              for local in table] for rank in ranks}
        views.append(KVView('a-' + str(index), RequestIdentity('m', 'r-' + str(index)),
                            capacity, target - 1, 16, rank_tables))
    pool = adapter(upload)()
    pool._bound = None
    pool.layout, pool.rank = layout, 1
    pool.layer_num, pool.head_num = layers, heads
    pool.start_layer, pool.end_layer = 0, layers - 1
    pool.physical_block_count = 40000
    pool.device = 'cuda'
    pool.bind_forward(views, list(range(len(views))), start_lengths, 'DECODE')
    return pool, views


def forward_for(bound):
    return SimpleNamespace(forward_mode=SimpleNamespace(name='DECODE'),
        batch_size=len(bound.target_lengths), req_pool_indices_cpu=list(bound.req_pool_indices),
        seq_lens_cpu=list(bound.target_lengths),
        input_ids=Tensor([0] * bound.total_written, dtype='int64'),
        out_cache_loc=Tensor([0] * bound.total_written, dtype='int64'))


class DecodeRunMetadataChecks(unittest.TestCase):
    def test_advanced_steps_match_legacy_and_upload_the_table_once(self):
        uploads = Uploads()
        pool, _ = run_fixture(uploads, steps=20)
        lengths = [17, 30]
        table_uploads = []
        for step in range(20):
            bound = pool.bound_forward
            pool.validate_forward(forward_for(bound))
            for layer in range(4):
                slots, tables = pool.layer_metadata(layer)
                expected_slots, expected_tables = legacy_metadata(bound, layer)
                self.assertEqual(slots.tolist(), expected_slots)
                # Pages the kernel reads (ceil(context/16)) match the legacy table.
                for row, target in enumerate(bound.target_lengths):
                    pages = (target + 15) // 16
                    for head in range(3):
                        self.assertEqual(tables.tolist()[row][head][:pages],
                                         expected_tables[row][head][:pages])
                self.assertGreaterEqual(len(tables.tolist()[0][0]) * 16, max(bound.target_lengths))
            table_uploads.append(bound.table_buffer)
            if step < 19:
                lengths = [length + 1 for length in lengths]
                pool.advance_decode([0, 1], lengths)
        self.assertTrue(all(buffer is table_uploads[0] for buffer in table_uploads),
                        'A decode run must reuse its uploaded block table')

    def test_advance_requires_executed_step_and_stays_inside_capacity(self):
        uploads = Uploads()
        pool, views = run_fixture(uploads, start_lengths=(17,), steps=3)
        with self.assertRaises(IntegrationError):
            pool.advance_decode([0], [18])  # Current step never executed.
        pool.validate_forward(forward_for(pool.bound_forward))
        with self.assertRaises(IntegrationError):
            pool.advance_decode([0], [19])  # Must append exactly one token.
        with self.assertRaises(IntegrationError):
            pool.advance_decode([1], [18])  # Rows cannot change.
        pool.advance_decode([0], [18])
        pool.validate_forward(forward_for(pool.bound_forward))
        pool.advance_decode([0], [19])
        pool.validate_forward(forward_for(pool.bound_forward))
        capacity = views[0].capacity_tokens
        while pool.bound_forward.target_lengths[0] < capacity:
            next_length = pool.bound_forward.target_lengths[0] + 1
            pool.advance_decode([0], [next_length])
            pool.validate_forward(forward_for(pool.bound_forward))
        with self.assertRaises(IntegrationError):
            pool.advance_decode([0], [capacity + 1])


if __name__ == '__main__':
    unittest.main(verbosity=2)
