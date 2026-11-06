"""CPU lifecycle checks for loading-only owner communicators; no GPU imports."""
import ast
from contextlib import nullcontext, redirect_stdout
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


class Tensor:
    device = SimpleNamespace(type='cuda', index=0)
    shape = (2,)
    dtype = 'torch.float16'

    def __init__(self, pointer): self.pointer = pointer
    def data_ptr(self): return self.pointer
    def storage_offset(self): return 0
    def stride(self): return (1,)


class OwnerCollectivesChecks(unittest.TestCase):
    def fixture(self):
        self.trace = []
        self.initialized = True
        self.weight = Tensor(12345)
        self.buffer = Tensor(67890)
        trace = self.trace

        class Base:
            def load(owner): trace.append('load')

        path = ROOT / 'muxserve_sglang/weight_owner.py'
        tree = ast.parse(path.read_text(encoding='utf-8'))
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'SharedWeightOwner')
        cls.body = [n for n in cls.body if isinstance(n, ast.FunctionDef)
                    and n.name in ('load', '_release_loading_collectives', 'serve')]
        namespace = dict(WeightCacheDaemon=Base, IntegrationError=IntegrationError,
                         json=json, os=os, __package__='muxserve_sglang')
        exec(compile(ast.Module(body=[cls], type_ignores=[]), str(path), 'exec'), namespace)
        owner = namespace['SharedWeightOwner'].__new__(namespace['SharedWeightOwner'])
        owner.model_path, owner.gpu_id, owner.tp_rank = '/storage/models/llama-7b', 0, 0
        owner.tp_size, owner.pp_size, owner.dp_size = 4, 1, 1
        owner.ep_size, owner.attn_cp_size, owner.moe_dp_size = 1, 1, 1
        owner.enable_dp_attention = False
        owner.authorized, owner.exports = {}, {}
        owner._loading_collectives_released = False
        owner.model = SimpleNamespace(
            named_parameters=lambda **kw: [('weight', self.weight)],
            named_buffers=lambda **kw: [('cache', self.buffer)])
        owner.state_tensors = {'weight': (self.weight, True), 'cache': (self.buffer, False)}

        def close(comm):
            self.trace.append(('nccl_destroy', comm))

        raw = SimpleNamespace(available=True, disabled=True, comm='raw-TP-comm',
                              nccl=SimpleNamespace(ncclCommDestroy=close))
        self.group = SimpleNamespace(cpu_group='owner-TP-gloo', pynccl_comm=raw)
        self.raw = raw
        cuda = SimpleNamespace(device=lambda device: nullcontext(),
            synchronize=lambda device: trace.append('cuda_sync'),
            empty_cache=lambda: trace.append('empty_cache'),
            memory_allocated=lambda device: 4096,
            memory_reserved=lambda device: 8192,
            mem_get_info=lambda device: (1024**3, 40 * 1024**3))
        dist = SimpleNamespace(is_initialized=lambda: self.initialized,
            barrier=lambda **kw: trace.append(('barrier', kw['group'])))

        def destroy_model(): self.trace.append('destroy_model_parallel')
        def destroy_environment():
            self.trace.append('destroy_distributed_environment')
            self.initialized = False

        parallel = ModuleType('sglang.srt.distributed.parallel_state')
        parallel.get_tp_group = lambda: self.group
        parallel.destroy_model_parallel = destroy_model
        parallel.destroy_distributed_environment = destroy_environment
        torch = ModuleType('torch')
        torch.cuda, torch.distributed = cuda, dist
        weights = ModuleType('muxserve_sglang.weight_loader')
        source = ast.parse((ROOT / 'muxserve_sglang/weight_loader.py').read_text(encoding='utf-8'))
        exec(compile(ast.Module(body=[n for n in source.body if isinstance(n, ast.FunctionDef)
            and n.name == '_local_alias'], type_ignores=[]), 'real-local-alias', 'exec'), weights.__dict__)
        self.addCleanup(patch.stopall)
        patch.dict(sys.modules, {'torch': torch, 'torch.distributed': dist,
            'sglang.srt.distributed.parallel_state': parallel,
            'muxserve_sglang.weight_loader': weights}).start()
        return owner

    def test_load_closes_only_loading_groups_before_ready_and_keeps_all_aliases(self):
        owner = self.fixture()
        self.assertTrue(hasattr(owner, '_release_loading_collectives'), 'Owner communicator release is missing')
        model, tensors = owner.model, owner.state_tensors
        output = io.StringIO()
        with redirect_stdout(output): owner.load()
        report = json.loads(output.getvalue().split(' ', 1)[1])
        self.assertEqual(self.trace, ['load', ('barrier', 'owner-TP-gloo'), 'cuda_sync',
            ('nccl_destroy', 'raw-TP-comm'), 'destroy_model_parallel',
            'destroy_distributed_environment', 'empty_cache', 'cuda_sync'])
        self.assertTrue(owner._loading_collectives_released)
        self.assertIs(owner.model, model)
        self.assertIs(owner.state_tensors, tensors)
        self.assertEqual((self.weight.pointer, self.buffer.pointer), (12345, 67890))
        self.assertIsNone(self.group.pynccl_comm)
        self.assertIsNone(self.raw.comm)
        self.assertFalse(self.raw.available)
        self.assertEqual(report['tensors_verified'], 2)
        self.assertTrue(report['pynccl_closed'])
        self.assertTrue(report['torch_distributed_closed'])

    def test_authorized_or_exported_consumer_prevents_any_teardown(self):
        for field in ('authorized', 'exports'):
            with self.subTest(field=field):
                owner = self.fixture()
                self.assertTrue(hasattr(owner, '_release_loading_collectives'), 'Owner communicator release is missing')
                setattr(owner, field, {99: 'P'})
                with self.assertRaises(IntegrationError): owner._release_loading_collectives()
                self.assertEqual(self.trace, [])

    def test_unreviewed_topology_is_rejected_before_collective(self):
        for field, value in (('tp_size', 2), ('pp_size', 2), ('dp_size', 2),
                             ('ep_size', 2), ('attn_cp_size', 2), ('moe_dp_size', 2),
                             ('enable_dp_attention', True)):
            with self.subTest(field=field):
                owner = self.fixture()
                self.assertTrue(hasattr(owner, '_release_loading_collectives'), 'Owner communicator release is missing')
                setattr(owner, field, value)
                with self.assertRaises(IntegrationError): owner._release_loading_collectives()
                self.assertEqual(self.trace, [])

    def test_destroy_failure_keeps_weights_and_cannot_publish_success(self):
        owner = self.fixture()
        self.assertTrue(hasattr(owner, '_release_loading_collectives'), 'Owner communicator release is missing')
        model, tensors = owner.model, owner.state_tensors
        def fail(comm): raise RuntimeError('NCCL close failed')
        self.raw.nccl.ncclCommDestroy = fail
        output = io.StringIO()
        with redirect_stdout(output), self.assertRaisesRegex(RuntimeError, 'NCCL close failed'):
            owner.load()
        self.assertFalse(owner._loading_collectives_released)
        self.assertEqual(output.getvalue(), '')
        self.assertIs(owner.model, model)
        self.assertIs(owner.state_tensors, tensors)
        self.assertNotIn('destroy_model_parallel', self.trace)

    def test_mutated_model_or_export_tensor_alias_cannot_publish_success(self):
        for target in ('model', 'state'):
            with self.subTest(target=target):
                owner = self.fixture()
                self.assertTrue(hasattr(owner, '_release_loading_collectives'), 'Owner communicator release is missing')
                import torch
                def change():
                    if target == 'model': owner.model.named_parameters = lambda **kw: [('weight', Tensor(999))]
                    else: owner.state_tensors['weight'] = (Tensor(999), True)
                torch.cuda.empty_cache = change
                output = io.StringIO()
                with redirect_stdout(output), self.assertRaises(IntegrationError): owner.load()
                self.assertFalse(owner._loading_collectives_released)
                self.assertEqual(output.getvalue(), '')

    def test_serve_cannot_publish_ready_before_collective_release(self):
        owner = self.fixture()
        with self.assertRaises(IntegrationError): owner.serve()

    def test_missing_cpu_group_cannot_fall_back_to_gpu_barrier(self):
        owner = self.fixture()
        self.group.cpu_group = None
        with self.assertRaises(IntegrationError): owner.load()
        self.assertEqual(self.trace, ['load'])

    def test_repeated_release_cannot_destroy_the_same_communicator_twice(self):
        owner = self.fixture()
        with redirect_stdout(io.StringIO()): owner.load()
        before = list(self.trace)
        with self.assertRaises(IntegrationError): owner._release_loading_collectives()
        self.assertEqual(self.trace, before)


if __name__ == '__main__':
    unittest.main(verbosity=2)
