"""CPU reference-lifetime checks; no Torch/SGLang/GPU imports."""
import ast
from contextlib import redirect_stdout
import gc
import io
import json
import os
from pathlib import Path
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import Mock, patch
import weakref

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from muxserve.contracts import IntegrationError


class Holder:
    pass


class ConsumerTeardownChecks(unittest.TestCase):
    def fixture(self):
        source = ROOT / 'muxserve/flexserver/sglang_hooks.py'
        functions = [node for node in ast.parse(source.read_text()).body if isinstance(node, ast.FunctionDef)
                     and node.name == 'release_process_imports']
        self.assertEqual(len(functions), 1, 'Explicit consumer IPC cleanup is missing')
        self.context = SimpleNamespace(_drain_complete=True, model=SimpleNamespace(model_id='llama7b'), stage='P')
        model, pool = Holder(), Holder()
        self.model_ref, self.pool_ref = weakref.ref(model), weakref.ref(pool)
        self.cache = {'native-RoPE-module': model}
        self.trace = []
        self.imported = SimpleNamespace(metadata=SimpleNamespace(rank=0), closed=False)
        def close():
            self.assertIsNone(self.model_ref(), 'Native model still aliases imported weights')
            self.assertIsNone(self.pool_ref(), 'Native KV adapter still aliases imported buffers')
            self.imported.closed = True
            self.trace.append('close_import')
        self.imported.close = close
        self.context._consumer_cleanup = dict(model=weakref.ref(model), pool=weakref.ref(pool),
                                             imported=self.imported)
        factory = ModuleType('sglang.srt.layers.rotary_embedding.factory')
        factory._ROPE_DICT = self.cache
        torch = ModuleType('torch')
        torch.cuda = SimpleNamespace(synchronize=lambda device: self.trace.append('sync'),
            ipc_collect=lambda: self.trace.append('ipc_collect'), empty_cache=lambda: self.trace.append('empty_cache'))
        self.addCleanup(patch.stopall)
        patch.dict(sys.modules, {'torch': torch, 'sglang.srt.layers.rotary_embedding.factory': factory}).start()
        namespace = dict(context=lambda: self.context, IntegrationError=IntegrationError, os=os, json=json)
        exec(compile(ast.Module(body=functions, type_ignores=[]), str(source), 'exec'), namespace)
        return namespace['release_process_imports']

    def test_global_rope_reference_and_imported_pool_are_released_before_process_exit(self):
        release = self.fixture()
        output = io.StringIO()
        with tempfile.TemporaryDirectory() as temporary, redirect_stdout(output):
            self.context.runtime_dir = Path(temporary)
            release(0)
            proof = json.loads((Path(temporary) / 'llama7b-P-r0-closed.json').read_text())
        self.assertFalse(self.cache)
        self.assertTrue(self.imported.closed)
        self.assertIsNone(self.context._consumer_cleanup)
        self.assertEqual(self.trace, ['sync', 'close_import', 'ipc_collect', 'empty_cache', 'sync'])
        self.assertTrue(proof['model_collected'] and proof['kv_adapter_collected'] and proof['pool_closed'])

    def test_missing_drain_or_surviving_model_prevents_pool_close(self):
        for kind in ('not_drained', 'live_model'):
            with self.subTest(kind=kind):
                release = self.fixture()
                if kind == 'not_drained': self.context._drain_complete = False
                else: external_model = self.model_ref()
                with tempfile.TemporaryDirectory() as temporary:
                    self.context.runtime_dir = Path(temporary)
                    with self.assertRaises(IntegrationError): release(0)
                    self.assertFalse(list(Path(temporary).iterdir()))
                self.assertFalse(self.imported.closed)

    def test_native_failure_cannot_reach_consumer_cleanup(self):
        tree = ast.parse((ROOT / 'muxserve/launch.py').read_text())
        function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == '_rank_process')
        calls = [node for node in function.body if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call)]
        names = [getattr(node.value.func, 'id', None) for node in calls]
        self.assertGreater(names.index('release_process_imports'), names.index('run_scheduler_process'))
        self.assertFalse(any(isinstance(node, ast.Try) for node in function.body), 'No cleanup on uncertain native exceptions')


class ProducerReferenceChecks(unittest.TestCase):
    def test_global_rope_cache_is_cleared_only_after_all_consumers_exit(self):
        tree = ast.parse((ROOT / 'muxserve/flexstore/weight_owner.py').read_text())
        cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == 'SharedWeightOwner')
        cls.bases = []
        cls.body = [node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == 'close_after_safe_drain']
        namespace = dict(IntegrationError=IntegrationError, os=os, json=json)
        exec(compile(ast.Module(body=[cls], type_ignores=[]), 'owner-close', 'exec'), namespace)
        owner = namespace['SharedWeightOwner']()
        value = Holder()
        reference = weakref.ref(value)
        factory = ModuleType('sglang.srt.layers.rotary_embedding.factory')
        factory._ROPE_DICT = {'cached-rope': value}
        owner.state_tensors, owner.model = {'cos_sin': value}, value
        del value
        owner._safe_close = True
        owner.authorized, owner.exited, owner.exports = {10: 'P', 20: 'D'}, {10, 20}, {10: 'SENT', 20: 'SENT'}
        owner.model_path, owner.tp_rank, owner.gpu_id = '/models/llama7b', 0, 0
        owner.shutdown = lambda: setattr(owner, 'model', None)
        def collect():
            gc.collect()
            self.assertIsNone(reference(), 'Global RoPE cache still pins an exported CUDA storage')
        torch = ModuleType('torch')
        torch.cuda = SimpleNamespace(synchronize=lambda device: None, ipc_collect=collect, empty_cache=lambda: None)
        with patch.dict(sys.modules, {'torch': torch, 'sglang.srt.layers.rotary_embedding.factory': factory}), redirect_stdout(io.StringIO()):
            owner._safe_close = False
            with self.assertRaises(IntegrationError): owner.close_after_safe_drain()
            self.assertIsNotNone(reference())
            self.assertTrue(factory._ROPE_DICT)
            owner._safe_close = True
            owner.close_after_safe_drain()
        self.assertIsNone(reference())


class SupervisorProofChecks(unittest.TestCase):
    def test_normal_waitpid_alone_cannot_release_owner_leases(self):
        source = ROOT / 'muxserve/launch.py'
        cls = next(node for node in ast.parse(source.read_text()).body
                   if isinstance(node, ast.ClassDef) and node.name == 'Supervisor')
        cls.body = [node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == '_observe']
        namespace = dict(IntegrationError=IntegrationError, json=json, safe_path=lambda value: value)
        exec(compile(ast.Module(body=[cls], type_ignores=[]), str(source), 'exec'), namespace)
        owner = namespace['Supervisor']()
        owner.native_failure = SimpleNamespace(is_set=lambda: False)
        owner.rpc = owner.driver = None
        owner.weights, owner.reaped = {}, set()
        owner.controller = Mock()
        owner.controller.group_id.return_value = 'group'
        owner.controller.close_ready.return_value = {'safe_to_exit': True}
        owner.controller.snapshot.return_value = {'groups': {}}
        owner.gpu, owner._owner_command = Mock(), Mock()
        owner.ranks = {('llama7b', 'P', 0): {'process': SimpleNamespace(pid=123, exitcode=0, join=lambda: None)}}
        with tempfile.TemporaryDirectory() as temporary:
            owner.directory = Path(temporary)
            with self.assertRaises(IntegrationError): owner._observe()
            owner.gpu.release_consumer.assert_not_called()
            owner._owner_command.assert_not_called()
            proof = dict(model_id='llama7b', stage='P', rank=0, pid=123,
                         model_collected=True, kv_adapter_collected=True, pool_closed=True)
            (owner.directory / 'llama7b-P-r0-closed.json').write_text(json.dumps(proof))
            owner._observe()
            owner.gpu.release_consumer.assert_called_once()
            owner._owner_command.assert_called_once()
            self.assertEqual(owner.reaped, {('llama7b', 'P', 0)})


if __name__ == '__main__': unittest.main(verbosity=2)
