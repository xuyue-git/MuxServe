"""CPU checks for idle-only allocator trimming; no Torch or GPU is imported."""
import ast
from collections import OrderedDict
from contextlib import nullcontext, redirect_stdout
from copy import deepcopy
import io
import json
import os
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from muxserve_sglang.contracts import IntegrationError


def checked_runtime(cuda):
    source = ROOT / 'muxserve_sglang/sglang_runtime.py'
    tree = ast.parse(source.read_text())
    controlled = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'ControlledScheduler')
    controlled.body = [n for n in controlled.body if isinstance(n, ast.FunctionDef)
                       and n.name in ('_accept', '_trim_idle_cuda_cache')]
    functions = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'require']
    namespace = dict(__name__='muxserve_sglang.sglang_runtime', __package__='muxserve_sglang', torch=SimpleNamespace(cuda=cuda),
                     IntegrationError=IntegrationError, deepcopy=deepcopy, json=json, os=os)
    exec(compile(ast.Module(body=[*functions, controlled], type_ignores=[]), str(source), 'exec'), namespace)
    return namespace['ControlledScheduler']


class Tensor:
    device = SimpleNamespace(type='cuda', index=0)
    shape = (1,)
    dtype = 'torch.float16'

    def __init__(self, pointer):
        self.pointer = pointer

    def data_ptr(self): return self.pointer
    def storage_offset(self): return 0
    def stride(self): return (1,)


class IdleCacheChecks(unittest.TestCase):
    def fixture(self):
        self.trace = []
        self.reserved = 256 * 1024**2
        self.allocated = 8 * 1024**2

        def empty():
            self.trace.append('empty_cache')
            self.reserved = self.allocated

        cuda = SimpleNamespace(device=lambda device: nullcontext(),
            memory_allocated=lambda device: self.allocated,
            memory_reserved=lambda device: self.reserved, empty_cache=empty)
        runtime_type = checked_runtime(cuda)
        runtime = runtime_type.__new__(runtime_type)
        runtime.ctx = SimpleNamespace(group_id='group', model=SimpleNamespace(model_id='llama13b'), stage='D')
        runtime.rank, runtime.pid, runtime.cursor = 0, 1234, 0
        runtime.prepared, runtime.requests, runtime.commands = {}, {}, OrderedDict()
        runtime.pool = SimpleNamespace(_bound=None, pool_key=Tensor(100), pool_value=Tensor(200))
        runtime.manager = SimpleNamespace(descriptors={})
        weight = Tensor(300)
        model = SimpleNamespace(named_parameters=lambda **kwargs: [('weight', weight)], named_buffers=lambda: [])
        runtime.runner = SimpleNamespace(gpu_id=0, model=model, _muxserve_permit=None)
        module = ModuleType('muxserve_sglang.weight_loader')
        tree = ast.parse((ROOT / 'muxserve_sglang/weight_loader.py').read_text())
        body = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in ('_local_alias', 'audit_shared_weights')]
        namespace = dict(IntegrationError=IntegrationError)
        exec(compile(ast.Module(body=body, type_ignores=[]), 'shared-weight-audit', 'exec'), namespace)
        model._muxserve_weight_imports = {'weight': dict(alias=namespace['_local_alias'](weight),
            allocation_digest='owner-allocation', refcounter_digest='owner-import', is_param=True)}
        module._local_alias, module.audit_shared_weights = namespace['_local_alias'], namespace['audit_shared_weights']
        self.addCleanup(patch.stopall)
        patch.dict(sys.modules, {'muxserve_sglang.weight_loader': module}).start()
        runtime.sync = lambda: self.trace.append('sync')
        runtime._release_request = lambda identity: self.trace.append(('release_request', identity))
        runtime._report = lambda *args: self.trace.append(('report', args))
        return runtime, cuda, weight

    def command(self):
        return dict(group_id='group', model_id='llama13b', stage='D', rank=0,
            kind='CLEANUP', command_seq=1, cleanup_id='cleanup:test', request_id='request', attempt=1)

    def test_cleanup_releases_unused_cache_before_quiesced_without_changing_live_aliases(self):
        runtime, _, weight = self.fixture()
        self.assertTrue(hasattr(runtime, '_trim_idle_cuda_cache'), 'Idle cache release is missing')
        output = io.StringIO()
        with redirect_stdout(output):
            runtime._accept(self.command())
        evidence = json.loads(output.getvalue().split(' ', 1)[1])
        self.assertEqual(evidence['released_reserved_bytes'], 248 * 1024**2)
        self.assertEqual(evidence['allocated_before_bytes'], evidence['allocated_after_bytes'])
        self.assertEqual((runtime.pool.pool_key.pointer, runtime.pool.pool_value.pointer, weight.pointer), (100, 200, 300))
        self.assertLess(self.trace.index('empty_cache'), next(i for i, item in enumerate(self.trace) if isinstance(item, tuple) and item[0] == 'report'))
        self.assertEqual(self.trace[:3], ['sync', ('release_request', ('request', 1)), 'sync'])
        self.assertEqual(runtime.cursor, 1)

    def test_trim_walks_the_model_once_after_release(self):
        runtime, _, _ = self.fixture()
        module = sys.modules['muxserve_sglang.weight_loader']
        audit, calls = module.audit_shared_weights, []
        def counted(model):
            calls.append('empty_cache' in self.trace)
            return audit(model)
        module.audit_shared_weights = counted
        with redirect_stdout(io.StringIO()):
            runtime._accept(self.command())
        self.assertEqual(calls, [True], 'One weight audit, after the cache release')

    def test_threshold_keeps_a_small_cache_and_still_reports(self):
        runtime, _, _ = self.fixture()  # 248 MiB reserved but unused
        runtime.ctx.config = SimpleNamespace(idle_cache_trim_mib=512)
        output = io.StringIO()
        with redirect_stdout(output):
            runtime._accept(self.command())
        evidence = json.loads(output.getvalue().split(' ', 1)[1])
        self.assertNotIn('empty_cache', self.trace)
        self.assertEqual((evidence['trimmed'], evidence['released_reserved_bytes']), (False, 0))
        self.assertTrue(any(isinstance(item, tuple) and item[0] == 'report' for item in self.trace))
        self.assertEqual(runtime.cursor, 1)

    def test_threshold_trims_a_large_cache(self):
        runtime, _, _ = self.fixture()
        runtime.ctx.config = SimpleNamespace(idle_cache_trim_mib=128)
        output = io.StringIO()
        with redirect_stdout(output):
            runtime._accept(self.command())
        evidence = json.loads(output.getvalue().split(' ', 1)[1])
        self.assertIn('empty_cache', self.trace)
        self.assertEqual((evidence['trimmed'], evidence['released_reserved_bytes']), (True, 248 * 1024**2))

    def test_live_request_preparation_mapping_or_room_prevents_trimming(self):
        for field, value in (('requests', {'live': object()}), ('prepared', {'batch': object()}),
                             ('binding', object()), ('descriptors', {'room': object()})):
            with self.subTest(field=field):
                runtime, _, _ = self.fixture()
                self.assertTrue(hasattr(runtime, '_trim_idle_cuda_cache'), 'Idle cache release is missing')
                if field == 'binding': runtime.pool._bound = value
                elif field == 'descriptors': runtime.manager.descriptors = value
                else: setattr(runtime, field, value)
                runtime._trim_idle_cuda_cache('cleanup:test')
                self.assertNotIn('empty_cache', self.trace)

    def test_cuda_failure_does_not_report_quiescence(self):
        runtime, cuda, _ = self.fixture()
        self.assertTrue(hasattr(runtime, '_trim_idle_cuda_cache'), 'Idle cache release is missing')
        def fail(): raise RuntimeError('CUDA cache release failed')
        cuda.empty_cache = fail
        with self.assertRaisesRegex(RuntimeError, 'CUDA cache release failed'):
            runtime._accept(self.command())
        self.assertFalse(any(isinstance(item, tuple) and item[0] == 'report' for item in self.trace))
        self.assertEqual(runtime.cursor, 0)

    def test_changed_weight_or_kv_alias_cannot_report_quiescence(self):
        for target in ('weight', 'kv'):
            with self.subTest(target=target):
                runtime, cuda, weight = self.fixture()
                self.assertTrue(hasattr(runtime, '_trim_idle_cuda_cache'), 'Idle cache release is missing')
                def change():
                    tensor = weight if target == 'weight' else runtime.pool.pool_key
                    tensor.pointer += 1
                cuda.empty_cache = change
                with self.assertRaises(IntegrationError):
                    runtime._accept(self.command())
                self.assertFalse(any(isinstance(item, tuple) and item[0] == 'report' for item in self.trace))


if __name__ == '__main__':
    unittest.main(verbosity=2)
