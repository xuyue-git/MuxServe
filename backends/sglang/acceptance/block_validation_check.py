"""CPU checks for complete KV validation without per-block guard calls."""
from copy import deepcopy
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'acceptance')]
from metadata_upload_check import adapter, Uploads
from dataclasses import replace

import numpy as np

from muxserve_sglang.contracts import IntegrationError, KVView, ModelLayout, PackedBlocks, RequestIdentity, pack_tables


def case(*, layers=2, heads=2, pages=2, requests=1):
    pool = adapter(Uploads())()
    pool.layout = ModelLayout('m', layers, heads, (0, 1, 2, 3))
    pool.rank, pool.layer_num, pool.head_num = 0, layers, heads
    pool.physical_block_count = layers * heads * pages * requests + 1
    pool._bound = None
    views = []
    for request in range(requests):
        offset = request * layers * heads * pages
        tables = {rank: [[[offset + (layer * heads + head) * pages + page
                           for page in range(pages)] for head in range(heads)]
                         for layer in range(layers)] for rank in pool.layout.ranks}
        views.append(KVView('a-' + str(request), RequestIdentity('m', str(request)),
            pages * 16, pages * 16 - 1, 16, tables))
    return pool, views


def bind(pool, views):
    pool.bind_forward(views, tuple(range(len(views))), [view.valid_tokens + 1 for view in views], 'DECODE')


class BlockValidationChecks(unittest.TestCase):
    def test_tp4_30b_validation_does_not_dispatch_a_guard_for_every_block(self):
        pool, views = case(layers=60, heads=13, pages=40)
        namespace = pool.bind_forward.__func__.__globals__
        original = namespace['_require']
        calls = 0
        def counted(*args):
            nonlocal calls
            calls += 1
            return original(*args)
        namespace['_require'] = counted
        try:
            bind(pool, views)
        finally:
            namespace['_require'] = original
        self.assertLess(calls, 10000, 'Full TP4 validation dispatches Python guards for every physical block')
        self.assertEqual(pool.bound_forward.target_lengths, (640,))

    def test_invalid_values_at_every_rank_and_tail_are_rejected_before_binding(self):
        class IntSubclass(int):
            def __hash__(self):
                raise AssertionError('Invalid type reached hashing before validation')
        for rank in range(4):
            for invalid in (-1, 1000000, True, 3.0, '3', None, IntSubclass(3)):
                with self.subTest(rank=rank, invalid=repr(invalid)):
                    pool, views = case()
                    views[0].block_tables[rank][-1][-1][-1] = invalid
                    with self.assertRaises(IntegrationError) as caught:
                        bind(pool, views)
                    self.assertEqual(caught.exception.code, 'LAYOUT_MISMATCH')
                    self.assertIsNone(pool._bound)

    def test_aliases_within_across_heads_layers_and_requests_are_rejected(self):
        for rank in range(4):
            for destination in ('same_head', 'head', 'layer', 'request'):
                with self.subTest(rank=rank, destination=destination):
                    pool, views = case(requests=2)
                    first = views[0].block_tables[rank][0][0][0]
                    if destination == 'same_head':
                        views[0].block_tables[rank][0][0][1] = first
                    elif destination == 'head':
                        views[0].block_tables[rank][0][1][0] = first
                    elif destination == 'layer':
                        views[0].block_tables[rank][1][0][0] = first
                    else:
                        views[1].block_tables[rank][0][0][0] = first
                    with self.assertRaisesRegex(IntegrationError, 'Aliased physical blocks'):
                        bind(pool, views)
                    self.assertIsNone(pool._bound)

    def test_shape_checks_and_frozen_local_tables_are_preserved(self):
        for broken in ('rank', 'layer', 'head', 'page'):
            pool, views = case()
            table = views[0].block_tables
            if broken == 'rank':
                del table[3]
            elif broken == 'layer':
                table[3].pop()
            elif broken == 'head':
                table[3][-1].pop()
            else:
                table[3][-1][-1].pop()
            with self.subTest(broken=broken), self.assertRaises(IntegrationError):
                bind(pool, views)
            self.assertIsNone(pool._bound)
        pool, views = case(requests=2)
        original = deepcopy(views[0].block_tables[0])
        bind(pool, views)
        views[0].block_tables[0][0][0][0] = 999
        expected = tuple(tuple(tuple(blocks) for blocks in heads) for heads in original)
        self.assertEqual(pool.bound_forward.tables[0], expected)
        # Identical physical IDs on different GPUs remain valid.
        self.assertEqual(pool.bound_forward.req_pool_indices, (0, 1))


def packed(views):
    return [replace(view, block_tables=pack_tables(view.block_tables)) for view in views]


class PackedBlockChecks(unittest.TestCase):
    """Owner->rank messages carry PackedBlocks; validation must stay complete."""

    def test_packed_binding_matches_nested_lists(self):
        for requests in (1, 2):
            pool, views = case(layers=3, heads=2, pages=3, requests=requests)
            bind(pool, views)
            expected = pool.bound_forward
            other, _ = case(layers=3, heads=2, pages=3, requests=requests)
            bind(other, packed(views))
            got = other.bound_forward
            for name in ('mode', 'req_pool_indices', 'target_lengths', 'write_lengths', 'write_starts', 'capacities'):
                self.assertEqual(getattr(got, name), getattr(expected, name), name)
            for mine, theirs in zip(got.tables, expected.tables):
                self.assertTrue(np.array_equal(np.asarray(mine), np.asarray(theirs)))

    def test_packed_out_of_pool_and_aliases_rejected_on_every_rank(self):
        for rank in range(4):
            for change in ('negative', 'beyond', 'same_head', 'layer', 'request'):
                with self.subTest(rank=rank, change=change):
                    pool, views = case(requests=2)
                    tables = views[0].block_tables[rank]
                    if change == 'negative':
                        tables[-1][-1][-1] = -1
                    elif change == 'beyond':
                        tables[-1][-1][-1] = pool.physical_block_count
                    elif change == 'same_head':
                        tables[0][0][1] = tables[0][0][0]
                    elif change == 'layer':
                        tables[1][0][0] = tables[0][0][0]
                    else:
                        views[1].block_tables[rank][0][0][0] = tables[0][0][0]
                    with self.assertRaises(IntegrationError) as caught:
                        bind(pool, packed(views))
                    self.assertEqual(caught.exception.code, 'LAYOUT_MISMATCH')
                    self.assertIsNone(pool._bound)

    def test_packed_shape_mismatch_rejected(self):
        for broken in ('layer', 'head', 'page'):
            with self.subTest(broken=broken):
                pool, views = case()
                table = views[0].block_tables[2]
                if broken == 'layer':
                    table.pop()
                elif broken == 'head':
                    for heads in table:
                        heads.pop()
                else:
                    for heads in table:
                        for blocks in heads:
                            blocks.pop()
                with self.assertRaises(IntegrationError):
                    bind(pool, packed(views))
                self.assertIsNone(pool._bound)

    def test_packing_rejects_ragged_and_non_int(self):
        for bad in ([[[1, 2], [3]]], [[[1.0]]], [[[True]]], [[[2 ** 40]]], [[['1']]]):
            with self.subTest(bad=repr(bad)), self.assertRaises(IntegrationError):
                PackedBlocks.pack(bad)

    def test_packed_30b_validation_uses_few_guards(self):
        pool, views = case(layers=60, heads=13, pages=40)
        namespace = pool.bind_forward.__func__.__globals__
        original = namespace['_require']
        calls = 0
        def counted(*args):
            nonlocal calls
            calls += 1
            return original(*args)
        namespace['_require'] = counted
        try:
            bind(pool, packed(views))
        finally:
            namespace['_require'] = original
        self.assertLess(calls, 100)
        self.assertEqual(pool.bound_forward.tables[0].shape, (60, 13, 40))

    def own_rank(self, views, rank=0):
        return [replace(view, block_tables={rank: pack_tables(view.block_tables)[rank]}) for view in views]

    def test_own_rank_table_binds_the_same_forward(self):
        pool, views = case(layers=4, heads=3, pages=2, requests=2)
        bind(pool, views)
        full = pool.bound_forward.tables
        pool._bound = None
        pool.bind_forward(self.own_rank(views), (0, 1), [v.valid_tokens + 1 for v in views], 'DECODE',
                          own_rank_only=True)
        for a, b in zip(full, pool.bound_forward.tables):
            self.assertTrue(np.array_equal(a, b))

    def test_own_rank_mode_still_rejects_bad_own_tables_and_foreign_rank_sets(self):
        pool, views = case(layers=2, heads=2, pages=2)
        own = self.own_rank(views)
        targets = [v.valid_tokens + 1 for v in views]
        with self.assertRaises(IntegrationError):  # all ranks sent, own-rank mode expects only rank 0
            pool.bind_forward(views, (0,), targets, 'DECODE', own_rank_only=True)
        with self.assertRaises(IntegrationError):  # only rank 0 sent, full mode expects all ranks
            pool.bind_forward(own, (0,), targets, 'DECODE')
        tables = deepcopy(views[0].block_tables)
        tables[0][1][1][1] = tables[0][0][0][0]  # alias inside rank 0's own table
        aliased = [replace(views[0], block_tables={0: tables[0]})]
        with self.assertRaises(IntegrationError):
            pool.bind_forward(aliased, (0,), targets, 'DECODE', own_rank_only=True)
        outside = deepcopy(views[0].block_tables)
        outside[0][0][0][0] = pool.physical_block_count
        with self.assertRaises(IntegrationError):
            pool.bind_forward([replace(views[0], block_tables={0: outside[0]})], (0,), targets, 'DECODE',
                              own_rank_only=True)
        self.assertIsNone(pool._bound)


if __name__ == '__main__':
    unittest.main(verbosity=2)
