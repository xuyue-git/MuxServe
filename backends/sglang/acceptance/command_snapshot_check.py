"""CPU checks for detached command snapshots and one global mapping validation."""
import ast
from collections import OrderedDict
from copy import deepcopy
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'acceptance'))
from controller_check import Fixture
from muxserve_sglang import ipc
from muxserve_sglang.contracts import IntegrationError, KVTarget, KVView, ModelLayout, RequestIdentity
from muxserve_sglang import storage


class Mapping(dict):
    def __deepcopy__(self, memo):
        raise AssertionError('Whole KV mapping traversed by Python deepcopy')


def command():
    tables = Mapping({rank: [[[rank * 10, rank * 10 + 1]]] for rank in range(4)})
    return dict(kind='PREPARE', group_id='g', model_id='m', stage='P', rank=0,
        command_seq=1, batch_id='batch', batch_stage='D', ordered_request_ids=('r',),
        requests=[dict(request_id='r', reservation_view=dict(block_tables=tables),
                       rank_block_table=tables[0])])


def runtime_method(name, namespace):
    tree = ast.parse((ROOT / 'muxserve_sglang/sglang_runtime.py').read_text(encoding='utf-8'))
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == 'ControlledScheduler')
    method = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == name)
    exec(compile(ast.Module(body=[method], type_ignores=[]), 'actual-controlled-method', 'exec'), namespace)
    return namespace[name]


class SnapshotChecks(unittest.TestCase):
    def test_decode_within_a_page_does_not_sort_free_blocks_and_still_requires_all_ranks(self):
        store = storage.HeadwiseStore({0: 64, 1: 64})
        store.register_model(ModelLayout('m', 1, 1, (0, 1)))
        identity = RequestIdentity('m', 'r')
        first = store.reserve_kv('first', [KVTarget(identity, 1)], 'holder')
        allocation = first.views[0].allocation_id
        store.start_kv_batch(first.reservation_id)
        store.finish_kv_batch(first.reservation_id, 'SUCCESS', {allocation: 1}, (0, 1))
        counts = store.rank_usage()
        with patch.object(storage, 'sorted', side_effect=AssertionError('Decode sorted the entire free pool without allocating'), create=True):
            second = store.reserve_kv('second', [KVTarget(identity, 2, allocation)], 'holder')
        self.assertEqual(store.rank_usage(), counts)
        self.assertEqual(second.views[0].capacity_tokens, 16)
        self.assertEqual(store._batches[second.reservation_id].ranks, {0, 1})
        store.start_kv_batch(second.reservation_id)
        with self.assertRaises(IntegrationError):
            store.finish_kv_batch(second.reservation_id, 'SUCCESS', {allocation: 2}, (0,))
        store.finish_kv_batch(second.reservation_id, 'SUCCESS', {allocation: 2}, (0, 1))
        self.assertTrue(store.assert_integrity()['ok'])

    def test_store_views_and_reservation_replays_keep_mutable_mapping_isolated(self):
        store = storage.HeadwiseStore({0: 64, 1: 64})
        store.register_model(ModelLayout('m', 1, 1, (0, 1)))
        identity = RequestIdentity('m', 'r')
        target = KVTarget(identity, 1)
        first = store.reserve_kv('first', [target], 'holder')
        allocation = store._allocations[first.views[0].allocation_id]
        original = allocation.tables[0][0][0][0]
        detached = store.bind_kv(allocation.allocation_id, 'other')
        # Views are immutable packed snapshots; neither binds nor replays alias
        # the store's mutable lists.
        with self.assertRaises(Exception):
            detached.block_tables[0].data = b''
        self.assertIsNot(detached.block_tables, first.views[0].block_tables)
        self.assertEqual(detached.block_tables[0].tolist(), allocation.tables[0])
        replay = store.reserve_kv('first', [target], 'holder')
        replay.views[0].block_tables.clear()
        again = store.reserve_kv('first', [target], 'holder')
        self.assertEqual(again.views[0].block_tables[0].tolist()[0][0][0], original)
        self.assertEqual(allocation.tables[0][0][0][0], original)

    def test_ready_validates_global_mapping_once_and_keeps_passive_rank_unbound(self):
        bindings = []
        class Pool:
            def __init__(self, *args):
                self.layout, self.rank, self.pool_key, self.pool_value, self.size = None, 0, None, None, 32
                self._bound = None
            def bind_forward(self, views, rows, targets, mode):
                bindings.append(self)
                for view in views:
                    for rank in view.block_tables.values():
                        for heads in rank:
                            for blocks in heads:
                                if any(block < 0 for block in blocks):
                                    raise IntegrationError('INVALID_BLOCK', 'Nonlocal block rejected')
                self._bound = True
        class Values:
            def tolist(self): return [17]
            def cpu(self): return self
        def require(ok, message, code=None):
            if not ok: raise IntegrationError(code or 'INVALID', message)
        namespace = dict(__package__='muxserve_sglang', require=require, RequestIdentity=RequestIdentity,
            view_from_dict=lambda value: KVView(**{**value, 'identity': RequestIdentity(**value['identity'])}))
        validate = runtime_method('_validate_views', namespace)
        ready = runtime_method('_ready', namespace)
        value = command()
        value.update(purpose='GENERATE', reservation_id='reservation')
        item = value['requests'][0]
        item.update(attempt=0, target_valid_tokens=17)
        item['reservation_view'].update(allocation_id='a', identity=dict(model_id='m', request_id='r', attempt=0),
            capacity_tokens=32, valid_tokens=16, block_size=16, version=1)
        batch = SimpleNamespace(req_pool_indices_cpu=SimpleNamespace(tolist=lambda: [0]),
            req_pool_indices=SimpleNamespace(cpu=lambda: SimpleNamespace(tolist=lambda: [0])),
            seq_lens_cpu=Values(), seq_lens=Values())
        pool = Pool()
        worker = SimpleNamespace(pool=pool, rank=0, ctx=SimpleNamespace(model=SimpleNamespace(model_id='m')),
            sync=lambda: None, _report=lambda *args: None)
        worker._validate_views = lambda *args, **kwargs: validate(worker, *args, **kwargs)
        def state():
            return SimpleNamespace(command=value, reqs=[SimpleNamespace(kv=SimpleNamespace(req_pool_idx=0))], batch=batch)
        with patch.dict(sys.modules, {'muxserve_sglang.sglang_memory': SimpleNamespace(HeadwiseTokenToKVPool=Pool)}):
            ready(worker, state())
            self.assertEqual(len(bindings), 1, 'Compute readiness validated the same complete mapping twice')
            self.assertIs(bindings[0], pool)
            pool._bound = None
            bindings.clear()
            ready(worker, state(), logical_only=True)
            self.assertEqual(len(bindings), 1)
            self.assertIsNone(pool._bound, 'Passive rank received a compute binding')
            item['reservation_view']['block_tables'][3][0][0][0] = -1
            with self.assertRaisesRegex(IntegrationError, 'Nonlocal block rejected'):
                ready(worker, state())

    def test_light_decode_ready_checks_cpu_metadata_without_gpu_reads(self):
        def require(ok, message, code=None):
            if not ok:
                raise IntegrationError(code or 'INVALID', message)
        def forbidden(*args):
            raise AssertionError('Light decode READY touched the GPU')
        reported = []
        namespace = dict(__package__='muxserve_sglang', require=require)
        ready = runtime_method('_ready', namespace)
        cpu = lambda values: SimpleNamespace(tolist=lambda: values)
        gpu = SimpleNamespace(cpu=forbidden)
        def worker(batch):
            return SimpleNamespace(pool=SimpleNamespace(_bound=None), sync=forbidden,
                ctx=SimpleNamespace(stage='D', config=SimpleNamespace(decode_prepare_light=True)),
                _validate_views=lambda *args: (None, [17], 'DECODE'),
                _report=lambda *args: reported.append(args))
        def state(batch):
            return SimpleNamespace(batch=batch, ready=False, reqs=[SimpleNamespace(kv=SimpleNamespace(req_pool_idx=0))],
                                   command=dict(batch_id='b', batch_stage='D', reservation_id='r', requests=[{}]))
        good = SimpleNamespace(req_pool_indices_cpu=cpu([0]), seq_lens_cpu=cpu([17]),
                               req_pool_indices=gpu, seq_lens=gpu)
        ready(worker(good), state(good))
        self.assertEqual(len(reported), 1)
        wrong = SimpleNamespace(req_pool_indices_cpu=cpu([0]), seq_lens_cpu=cpu([18]),
                                req_pool_indices=gpu, seq_lens=gpu)
        with self.assertRaisesRegex(IntegrationError, 'metadata differs'):
            ready(worker(wrong), state(wrong))

    def test_prepare_publication_and_replay_are_detached_without_python_table_copy(self):
        fixture = Fixture()
        group = fixture.controller.groups['m']
        value = command()
        original = value['requests'][0]['reservation_view']['block_tables']
        fixture.controller._command(group, ('P', 0), 'PREPARE', {'requests': value['requests']})
        original[0][0][0][0] = 99
        first = fixture.controller.poll_commands(fixture.group, 'P', 0, 1001)
        table = first[0]['requests'][0]['reservation_view']['block_tables']
        self.assertEqual(table[0][0][0][0], 0)
        self.assertEqual(set(table), {0, 1, 2, 3}, 'Global mapping was projected away')
        self.assertIs(first[0]['requests'][0]['rank_block_table'], table[0])
        table[0][0][0][0] = 88
        replay = fixture.controller.poll_commands(fixture.group, 'P', 0, 1001)
        self.assertEqual(replay[0]['requests'][0]['reservation_view']['block_tables'][0][0][0][0], 0)

    def test_native_replay_and_prepared_state_own_independent_snapshots(self):
        def require(ok, message, code=None):
            if not ok:
                raise IntegrationError(code or 'INVALID', message)
        namespace = dict(require=require, IntegrationError=IntegrationError, deepcopy=deepcopy,
            copy_message=getattr(ipc, 'copy_message', None), CommandSnapshot=ipc.CommandSnapshot,
            Prepared=lambda value: SimpleNamespace(command=value))
        accept = runtime_method('_accept', namespace)
        prepare = runtime_method('_prepare', namespace)
        selected = []
        worker = SimpleNamespace(ctx=SimpleNamespace(group_id='g', stage='P', model=SimpleNamespace(model_id='m')),
            rank=0, cursor=0, commands=OrderedDict(), prepared={},
            _ready=lambda state, **kwargs: selected.append(state))
        worker._prepare = lambda value: prepare(worker, value)
        value = command()
        accept(worker, value)
        self.assertEqual(worker.cursor, 1)
        self.assertEqual(len(selected), 1)
        value['requests'][0]['reservation_view']['block_tables'][0][0][0][0] = 99
        replay = worker.commands[1]['requests'][0]['reservation_view']['block_tables']
        prepared = selected[0].command['requests'][0]['reservation_view']['block_tables']
        self.assertEqual(replay[0][0][0][0], 0)
        self.assertEqual(prepared[0][0][0][0], 0)
        replay[0][0][0][0] = 88
        self.assertEqual(prepared[0][0][0][0], 0)

    def test_copy_message_retains_transport_size_limit(self):
        self.assertTrue(callable(getattr(ipc, 'copy_message', None)), 'Detached transport snapshot helper missing')
        with patch.object(ipc, '_MAX_MESSAGE_BYTES', 32):
            with self.assertRaises(IntegrationError) as caught:
                ipc.copy_message({'too_large': b'x' * 64})
        self.assertEqual(caught.exception.code, 'MESSAGE_TOO_LARGE')


if __name__ == '__main__':
    unittest.main(verbosity=2)
