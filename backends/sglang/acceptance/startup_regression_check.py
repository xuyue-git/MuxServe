"""CPU regressions for native output capture, RoPE reservation and Linux RPC.

RoPE uses the pinned reservation function and actual owner export/audit methods
with lightweight buffer stand-ins; this does not test Torch math or CUDA IPC.
RPC tests use real AF_UNIX sockets, authentication and request frames on Linux.
"""
from __future__ import annotations

import ast
from collections import OrderedDict
from contextlib import nullcontext
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field, fields, FrozenInstanceError, make_dataclass, replace
import errno
import logging
from multiprocessing.connection import Connection, answer_challenge, deliver_challenge
import os
from pathlib import Path
import pickle
import socket
import sys
import tempfile
from threading import Thread
import time
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from muxserve_sglang.contracts import IntegrationError


def functions(path, names, namespace):
    tree = ast.parse((ROOT / path).read_text(encoding='utf-8'))
    body = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    exec(compile(ast.Module(body=body, type_ignores=[]), path, 'exec'), namespace)


class EagerForwardChecks(unittest.TestCase):
    def setUp(self):
        class Tensor:
            def __init__(self, count):
                self.shape, self.count = (count,), count

            def numel(self):
                return self.count

        class Event:
            def __init__(self, **kwargs):
                pass

            def record(self):
                pass

        self.tensor = Tensor
        self.no_copy = False
        self.torch = SimpleNamespace(Tensor=Tensor, int32='int32',
            tensor=lambda values, **kwargs: Tensor(len(values)),
            cuda=SimpleNamespace(Event=Event, is_current_stream_capturing=lambda: False))
        self.namespace = dict(__name__=__name__, dataclass=dataclass, field=field, replace=replace,
            KVCache=object, AttentionBackend=object, BaseRunner=object, torch=self.torch,
            IntegrationError=IntegrationError,
            envs=SimpleNamespace(SGLANG_EAGER_INPUT_NO_COPY=SimpleNamespace(get=lambda: self.no_copy)))
        for path, class_name, methods in (
            ('muxserve_sglang/sglang_memory.py', None, None),
            ('muxserve_sglang/sglang_attention.py', 'HeadwiseAttentionBackend', {'init_forward_metadata'}),
            ('third_party/sglang/python/sglang/srt/model_executor/runner/eager_runner.py', 'EagerRunner', {'load_batch'}),
            ('third_party/sglang/python/sglang/srt/model_executor/cuda_graph_buffer_registry.py',
             'CudaGraphBufferRegistry', {'extract_buffer'})):
            tree = ast.parse((ROOT / path).read_text(encoding='utf-8'))
            if class_name is None:
                body = [node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.ClassDef))]
            else:
                selected = next(node for node in tree.body
                                if isinstance(node, ast.ClassDef) and node.name == class_name)
                selected.body = [node for node in selected.body
                                 if isinstance(node, ast.FunctionDef) and node.name in methods]
                body = [selected]
            exec(compile(ast.Module(body=body, type_ignores=[]), path, 'exec'), self.namespace)
        self.batch_class = make_dataclass('ForwardInput', [
            'forward_mode', 'batch_size', 'req_pool_indices_cpu', 'seq_lens_cpu',
            'input_ids', 'out_cache_loc', 'extend_seq_lens_cpu',
            'extend_prefix_lens_cpu', 'extend_num_tokens'])
        self.registry = self.namespace['CudaGraphBufferRegistry'].__new__(self.namespace['CudaGraphBufferRegistry'])
        self.registry_calls = []
        self.registry.fill_from = lambda *args, **kwargs: self.registry_calls.append(kwargs)
        self.registry._slots = dict(out_cache_loc=SimpleNamespace(
            enabled=True, buffer=Tensor(8), name='out_cache_loc', post_fill=None,
            copy_from_fb=True, source_fn=None, slice_for=lambda *args: Tensor(args[1])))
        self.eager = self.namespace['EagerRunner'].__new__(self.namespace['EagerRunner'])
        self.eager._eager_registry = self.registry

    def batch(self, mode, size=1):
        target, written = (5, 1) if mode == 'DECODE' else (4, 4)
        return self.batch_class(SimpleNamespace(name='DECODE' if mode == 'DECODE' else 'EXTEND'),
            size, tuple(range(size)), (target,) * size,
            self.tensor(written * size), self.tensor(written * size),
            (written,) * size, (0,) * size, written * size)

    def test_granted_prefill_decode_and_rebuild_keep_exact_batch(self):
        from muxserve_sglang.sglang_hooks import assert_forward_permitted
        for mode in ('PREFILL', 'DECODE', 'REBUILD'):
            for size in (1, 2):
                for self.no_copy in (False, True):
                    with self.subTest(mode=mode, size=size, native_no_copy=self.no_copy):
                        batch = self.batch(mode, size)
                        pool = self.namespace['HeadwiseTokenToKVPool'].__new__(self.namespace['HeadwiseTokenToKVPool'])
                        pool.device = 'cpu'
                        pool._bound = self.namespace['BoundForward'](
                            mode, batch.req_pool_indices_cpu, batch.seq_lens_cpu,
                            (1 if mode == 'DECODE' else 4,) * size,
                            (4 if mode == 'DECODE' else 0,) * size, ())
                        runner = SimpleNamespace(token_to_kv_pool=pool,
                            _muxserve_permit={'grant_id': 'fresh', 'consumed': False})
                        backend = self.namespace['HeadwiseAttentionBackend'].__new__(self.namespace['HeadwiseAttentionBackend'])
                        backend.token_to_kv_pool = pool
                        with patch.dict(os.environ, MUXSERVE_EXECUTION_GROUP='test-granted-eager'), \
                                patch.dict(sys.modules, torch=self.torch):
                            assert_forward_permitted(runner, batch)
                            loaded = self.eager.load_batch(batch)
                            # This is the real post-load metadata validation that failed on GPU.
                            backend.init_forward_metadata(loaded)
                            self.assertIs(loaded, batch)
                            self.assertIs(loaded.out_cache_loc, batch.out_cache_loc)
                            self.assertIs(backend.forward_metadata.batch, batch)
                            self.assertIs(pool.validate_forward(loaded), pool._bound)
                            with self.assertRaises(IntegrationError) as replay:
                                assert_forward_permitted(runner, loaded)
                            self.assertEqual(replay.exception.code, 'FORWARD_NOT_GRANTED')
                            with self.assertRaises(IntegrationError) as different:
                                pool.validate_forward(replace(batch))
                            self.assertEqual(different.exception.code, 'FORWARD_MISMATCH')
        self.assertEqual(self.registry_calls, [])

    def test_native_eager_copy_paths_remain_unchanged(self):
        with patch.dict(os.environ, MUXSERVE_EXECUTION_GROUP=''):
            batch = self.batch('PREFILL')
            copied = self.eager.load_batch(batch)
            self.assertIsNot(copied, batch)
            self.assertIsNot(copied.out_cache_loc, batch.out_cache_loc)
            self.assertEqual(len(self.registry_calls), 1)
            self.no_copy = True
            shallow = self.eager.load_batch(batch)
            self.assertIsNot(shallow, batch)
            self.assertIs(shallow.out_cache_loc, batch.out_cache_loc)
            self.assertEqual(len(self.registry_calls), 1)


class OutputCaptureChecks(unittest.TestCase):
    def test_native_frozen_processor_rebound_for_prefill_and_decode(self):
        # Execute the pinned class and real controlled constructor without
        # importing their Torch/CUDA dependency graph. Only context inputs vary.
        native_path = 'third_party/sglang/python/sglang/srt/managers/scheduler_components/batch_result_processor.py'
        native_tree = ast.parse((ROOT / native_path).read_text(encoding='utf-8'))
        native = next(node for node in native_tree.body
                      if isinstance(node, ast.ClassDef) and node.name == 'SchedulerBatchResultProcessor')
        runtime_path = 'muxserve_sglang/sglang_runtime.py'
        runtime_tree = ast.parse((ROOT / runtime_path).read_text(encoding='utf-8'))
        capture = next(node for node in runtime_tree.body
                       if isinstance(node, ast.ClassDef) and node.name == 'CoordinatorOutputCapture')
        controlled = next(node for node in runtime_tree.body
                          if isinstance(node, ast.ClassDef) and node.name == 'ControlledScheduler')
        controlled.body = [node for node in controlled.body
                           if isinstance(node, ast.FunctionDef) and node.name == '__init__']
        namespace = dict(__name__=__name__, dataclass=dataclass,
                         OrderedDict=OrderedDict, os=os, IntegrationError=IntegrationError,
                         DisaggregationMode=SimpleNamespace(DECODE=object()),
                         get_disagg=lambda: SimpleNamespace(disaggregation_decode_enable_radix_cache=False))
        functions(runtime_path, {'require'}, namespace)
        imports = [node for node in runtime_tree.body
                   if isinstance(node, ast.ImportFrom) and node.module == 'dataclasses']
        exec(compile(ast.Module(body=[*imports, native, capture, controlled], type_ignores=[]),
                     'native-output-capture-test', 'exec'), namespace)
        processor_class = namespace['SchedulerBatchResultProcessor']
        for stage in ('P', 'D'):
            with self.subTest(stage=stage):
                values = {item.name: object() for item in fields(processor_class)}
                values['disaggregation_mode'] = namespace['DisaggregationMode'].DECODE
                values['tree_cache'] = SimpleNamespace(is_chunk_cache=lambda: True)
                original = processor_class(**values)
                with self.assertRaises(FrozenInstanceError):
                    original.output_streamer = object()
                runner = SimpleNamespace(token_to_kv_pool=SimpleNamespace(headwise_shared=True),
                                         ps=SimpleNamespace(tp_rank=0, pp_size=1))
                prefill_manager, decode_manager = object(), object()
                scheduler = SimpleNamespace(
                    tp_worker=SimpleNamespace(model_runner=runner), enable_overlap=False,
                    spec_algorithm=SimpleNamespace(is_none=lambda: True), tree_cache=values['tree_cache'],
                    enable_hisparse=False, enable_decode_hicache=False, chunked_req=None,
                    transfer_backend=SimpleNamespace(value='shared_headwise'),
                    server_args=SimpleNamespace(dist_timeout=10), output_streamer=values['output_streamer'],
                    batch_result_processor=original,
                    disagg_prefill_bootstrap_queue=SimpleNamespace(kv_manager=prefill_manager),
                    disagg_decode_prealloc_queue=SimpleNamespace(kv_manager=decode_manager))
                context = SimpleNamespace(stage=stage, config=SimpleNamespace(request_timeout_ms=20000))
                runtime = namespace['ControlledScheduler'](scheduler, context)
                rebound = scheduler.batch_result_processor
                self.assertIsNot(rebound, original)
                self.assertIs(type(rebound), processor_class)
                self.assertIs(rebound.output_streamer, runtime.capture)
                self.assertIs(scheduler.output_streamer, runtime.capture)
                self.assertIs(original.output_streamer, values['output_streamer'])
                for item in fields(original):
                    if item.name != 'output_streamer':
                        self.assertIs(getattr(rebound, item.name), getattr(original, item.name))
                self.assertIs(runtime.manager, prefill_manager if stage == 'P' else decode_manager)
                if stage == 'D':
                    rebound.process_batch_result_prebuilt(SimpleNamespace(reqs=[], return_logprob=False))
                    self.assertEqual(runtime.capture.calls, 1)


class Buffer:
    def __init__(self, rows):
        self.shape, self.dtype = (rows, 128), 'float32'
        self.device = SimpleNamespace(type='cuda', index=0)
        self.data = self

    def data_ptr(self):
        return id(self)

    def storage_offset(self):
        return 0

    def stride(self):
        return (128, 1)


class Rotary:
    def __init__(self):
        self.cos_sin_cache = Buffer(2048)
        self._parameters, self._buffers = {}, {'cos_sin_cache': self.cos_sin_cache}

    def register_buffer(self, name, tensor):
        setattr(self, name, tensor)
        self._buffers[name] = tensor

    def _ensure_cos_sin_cache_length(self, needed):
        if needed >= self.cos_sin_cache.shape[0]:
            self.cos_sin_cache = Buffer(needed + 1)


class Model:
    name = 'model.layers.0.self_attn.rotary_emb.cos_sin_cache'

    def __init__(self):
        self.rope = Rotary()
        self.model = SimpleNamespace(layers=SimpleNamespace())
        setattr(self.model.layers, '0', SimpleNamespace(self_attn=SimpleNamespace(rotary_emb=self.rope)))

    def children(self):
        return [self.rope]

    def named_parameters(self, **kwargs):
        return []

    def state_dict(self):
        return {}

    def named_buffers(self):
        return [(self.name, self.rope.cos_sin_cache)]


class RopeExportChecks(unittest.TestCase):
    def test_owner_exports_reserved_cache_and_consumer_keeps_alias(self):
        model = Model()
        context = SimpleNamespace(context_length=2048)
        spec = SimpleNamespace(speculative_num_steps=0, speculative_num_draft_tokens=0)
        environment = ModuleType('sglang.srt.environ')
        environment.envs = SimpleNamespace(**{key: SimpleNamespace(get=lambda value=value: value)
            for key, value in [('SGLANG_SPEC_EXPANSION_SAFETY_FACTOR', 1),
                               ('SGLANG_ROPE_CACHE_SAFETY_MARGIN', 256), ('SGLANG_ROPE_CACHE_ALIGN', 128)]})
        reserve_namespace = dict(get_model=lambda: context, get_spec=lambda: spec)
        functions('third_party/sglang/python/sglang/srt/utils/common.py',
                  {'reserve_rope_cache_for_long_sequences'}, reserve_namespace)
        reserve = reserve_namespace['reserve_rope_cache_for_long_sequences']
        utils = ModuleType('sglang.srt.utils')
        utils.reserve_rope_cache_for_long_sequences = reserve
        platforms = ModuleType('sglang.srt.platforms')
        platforms.current_platform = SimpleNamespace(synchronize=lambda: None, empty_cache=lambda: None)
        torch = ModuleType('torch')
        torch.cuda = SimpleNamespace(memory_reserved=lambda gpu: model.rope.cos_sin_cache.shape[0] * 128 * 4)
        class Base:
            def _init_distributed(self, server_args, model_config):
                pass
        tree = ast.parse((ROOT / 'muxserve_sglang/weight_owner.py').read_text())
        owner_class = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == 'SharedWeightOwner')
        methods = [node for node in owner_class.body if isinstance(node, ast.FunctionDef)
                   and node.name in ('_init_distributed', '_export_state')]
        selected = ast.ClassDef(name='Owner', bases=[ast.Name(id='Base', ctx=ast.Load())],
                                keywords=[], body=methods, decorator_list=[])
        namespace = dict(Base=Base, logger=logging.getLogger('rope-test'), TorchIpcTransportBackend=lambda: None)
        exec(compile(ast.fix_missing_locations(ast.Module(body=[selected], type_ignores=[])),
                     'owner-export-test', 'exec'), namespace)
        owner = namespace['Owner']()
        owner.model, owner.gpu_id = model, 0
        owner.preloaded_weights_bytes = 2048 * 128 * 4
        config = SimpleNamespace(context_len=2048, hf_text_config=SimpleNamespace(max_position_embeddings=2048))
        modules = {'sglang.srt.environ': environment, 'sglang.srt.utils': utils,
                   'sglang.srt.platforms': platforms, 'torch': torch}
        with patch.dict(sys.modules, modules):
            owner._init_distributed(None, config)
            owner._export_state()
            exported = owner.state_tensors[model.name][0]
            # Execute the real native IPC shape validation and assignment with
            # a fresh consumer skeleton. CUDA tensor creation is the stand-in.
            native_namespace = dict(torch=SimpleNamespace(device=lambda _: nullcontext()), time=time,
                nn=SimpleNamespace(Module=object), logger=logging.getLogger('ipc-test'),
                _initialize_model=lambda *args: Model())
            native_tree = ast.parse((ROOT / 'third_party/sglang/python/sglang/srt/weight_cache/ipc_loader.py').read_text())
            native_class = next(node for node in native_tree.body if isinstance(node, ast.ClassDef)
                                and node.name == 'IpcModelLoader')
            native_methods = [node for node in native_class.body if isinstance(node, ast.FunctionDef)
                              and node.name in ('_load_zero_copy_mode', '_initialize_model', '_set_module_tensor')]
            base_class = ast.ClassDef(name='IpcModelLoader', bases=[], keywords=[], body=native_methods, decorator_list=[])
            exec(compile(ast.fix_missing_locations(ast.Module(body=[base_class], type_ignores=[])),
                         'native-loader-test', 'exec'), native_namespace)
            custom_tree = ast.parse((ROOT / 'muxserve_sglang/weight_loader.py').read_text())
            custom_class = next(node for node in custom_tree.body if isinstance(node, ast.ClassDef)
                                and node.name == 'SharedWeightLoader')
            custom_methods = [node for node in custom_class.body if isinstance(node, ast.FunctionDef)
                              and node.name == '_initialize_model']
            subclass = ast.ClassDef(name='SharedWeightLoader',
                bases=[ast.Name(id='IpcModelLoader', ctx=ast.Load())], keywords=[],
                body=custom_methods or [ast.Pass()], decorator_list=[])
            exec(compile(ast.fix_missing_locations(ast.Module(body=[subclass], type_ignores=[])),
                         'shared-loader-test', 'exec'), native_namespace)
            loader_utils = ModuleType('sglang.srt.model_loader.utils')
            loader_utils.set_default_torch_dtype = lambda _: nullcontext()
            config.dtype = 'float16'
            with patch.dict(sys.modules, {'sglang.srt.model_loader.utils': loader_utils}):
                loader = native_namespace['SharedWeightLoader']()
                loader.load_config = None
                loader._transport_backend = SimpleNamespace(import_tensor=lambda entry: entry['tensor'])
                entries = {model.name: dict(tensor=exported, is_param=False)}
                consumer = loader._load_zero_copy_mode(config, None, entries, None)
                reserve(consumer, config)
                default_loader = native_namespace['IpcModelLoader']()
                default_loader.load_config = None
                default_loader._transport_backend = loader._transport_backend
                ordinary = default_loader._load_zero_copy_mode(config, None,
                    {model.name: dict(tensor=Buffer(2048), is_param=False)}, None)
                self.assertEqual(ordinary.rope.cos_sin_cache.shape[0], 2048)
        self.assertIs(exported, consumer.rope.cos_sin_cache,
                      'Consumer RoPE reservation replaced the exported shared buffer')
        self.assertEqual(exported.shape[0], 2304)
        self.assertEqual(owner.preloaded_weights_bytes, 2304 * 128 * 4)
        audit_namespace = dict(IntegrationError=IntegrationError)
        functions('muxserve_sglang/weight_loader.py', {'_local_alias', 'audit_shared_weights'}, audit_namespace)
        consumer._muxserve_weight_imports = {model.name: dict(alias=audit_namespace['_local_alias'](exported),
            allocation_digest='allocation', refcounter_digest='counter', is_param=False)}
        audit_namespace['audit_shared_weights'](consumer)
        consumer.rope.cos_sin_cache = Buffer(2304)
        with self.assertRaises(IntegrationError) as caught:
            audit_namespace['audit_shared_weights'](consumer)
        self.assertEqual(caught.exception.code, 'WEIGHT_ALIAS_CHANGED')


@unittest.skipUnless(sys.platform.startswith('linux'), 'Real AF_UNIX congestion test requires Linux')
class RPCConnectChecks(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(prefix='mx-startup-')
        os.chmod(self.directory.name, 0o700)
        self.address = str(Path(self.directory.name) / 'rpc.sock')
        self.listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.listener.bind(self.address)
        os.chmod(self.address, 0o600)
        self.listener.listen(1)
        self.listener.settimeout(3)
        self.fillers = []
        self.key = os.urandom(32)

    def tearDown(self):
        for filler in self.fillers:
            filler.close()
        self.listener.close()
        self.directory.cleanup()

    def fill_queue(self):
        for _ in range(16):
            candidate = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            candidate.settimeout(0.05)
            try:
                candidate.connect(self.address)
            except BlockingIOError as exc:
                candidate.close()
                self.assertIn(exc.errno, (errno.EAGAIN, errno.EWOULDBLOCK))
                return
            self.fillers.append(candidate)
        self.fail('Could not reproduce full AF_UNIX accept queue')

    def serve_one(self, *, drop_reply=False):
        for _ in self.fillers:
            accepted, _ = self.listener.accept()
            accepted.close()
        accepted, _ = self.listener.accept()
        with Connection(accepted.detach()) as connection:
            deliver_challenge(connection, self.key)
            answer_challenge(connection, self.key)
            request = pickle.loads(connection.recv_bytes())
            self.received.append(request)
            if not drop_reply:
                connection.send_bytes(pickle.dumps(dict(ok=True, result='once')))

    def test_queue_congestion_recovers_before_one_request_is_sent(self):
        from muxserve_sglang.ipc import StorageRPCClient
        self.fill_queue()
        self.received = []
        with ThreadPoolExecutor(max_workers=2) as workers:
            client = workers.submit(StorageRPCClient(self.address, self.key, 2).call, 'snapshot')
            time.sleep(0.08)
            serving = workers.submit(self.serve_one)
            try:
                self.assertEqual(client.result(timeout=3), 'once')
                serving.result(timeout=4)
            finally:
                self.listener.close()
        self.assertEqual([item['method'] for item in self.received], ['snapshot'])

    def test_persistent_congestion_respects_original_deadline(self):
        from muxserve_sglang.ipc import StorageRPCClient
        self.fill_queue()
        started = time.monotonic()
        with self.assertRaises(IntegrationError) as caught:
            StorageRPCClient(self.address, self.key, 0.12).call('snapshot')
        self.assertEqual(caught.exception.code, 'RPC_TIMEOUT')
        self.assertGreaterEqual(time.monotonic() - started, 0.10)
        self.assertLess(time.monotonic() - started, 0.7)

    def test_lost_reply_is_not_replayed(self):
        from muxserve_sglang.ipc import StorageRPCClient
        self.received = []
        with ThreadPoolExecutor(max_workers=1) as workers:
            serving = workers.submit(self.serve_one, drop_reply=True)
            with self.assertRaises(IntegrationError) as caught:
                StorageRPCClient(self.address, self.key, 1).call('mutate_once')
            serving.result(timeout=3)
        self.assertEqual(caught.exception.code, 'RPC_UNAVAILABLE')
        self.assertEqual(len(self.received), 1)

    def test_server_accept_queue_handles_eight_rank_startups(self):
        from muxserve_sglang.ipc import StorageRPCServer
        from muxserve_sglang.storage import HeadwiseStore
        owner = SimpleNamespace(export_pool=lambda *args: None, release_consumer=lambda *args: None,
                                owner_snapshot=lambda: {}, close=lambda: None)
        address = str(Path(self.directory.name) / 'server.sock')
        server = StorageRPCServer(HeadwiseStore({0: 8}), owner, address, self.key)
        clients = []
        try:
            for _ in range(8):
                client = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
                client.settimeout(0.05)
                clients.append(client)
                client.connect(address)
        finally:
            for client in clients:
                client.close()
            server._listener.close()
            server._failure_hold.set()  # CPU stand-in owns no GPU resources or consumers.
            server._failure_thread.join(3)
        self.assertFalse(server._failure_thread.is_alive())


if __name__ == '__main__':
    unittest.main(verbosity=2)
