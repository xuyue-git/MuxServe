"""The SGLang engine for the original MuxServe runtime.

This file replaces what MuxServe-vLLM did inside one P/D rank and nothing else:

  * PipeWorker + MuxServe's vLLM LLaMA (flexserver/models/llama.py)
      -> SGLang's LlamaForCausalLM, built by SGLang's own ModelRunner;
  * vLLM's head-wise PagedAttention (layers/attention.py, MuxServe branch)
      -> HeadwiseAttention below. The K/V store and the decode attention are
         MuxServe-vLLM's own head-wise ops (sglang.kernels.ops.attention.headwise
         in the SGLang fork), called as PagedAttention.forward calls them, on the
         KV pool the original FlexStore hands out (only those kernels read that
         layout). The prompt attention is SGLang's
         default for this GPU: FlashInfer's ragged causal prefill, as
         SGLang's FlashInferAttnBackend runs a prompt without a cached prefix;
  * GreedySampler (flexserver/sampler.py) -> greedy_sample below, same steps;
  * model_loader.load_from_server -> FlexStoreModelLoader below: the same
    FlexStore "weight" request and the same rebuild_cuda_tensor of each
    parameter, mapped into SGLang's model.

Everything SGLang decides is left to SGLang: its ServerArgs get only what the
original's command line fixes (the model, the TP size, the per-batch token
limit as SGLang's chunked prefill size, ports) and are resolved
by SGLang for this machine. The CUDA-graph configuration (decode FULL graphs at
SGLang's batch sizes, prefill BREAKABLE graphs at SGLang's token buckets), the
attention backend and the TP all-reduce are read from that resolution. A prefill
rank uses only the prefill graphs and a decode rank only the decode graphs,
SGLang's own rule for disaggregated prefill/decode servers.

Everything around it -- the C++ batch scheduler, FlexStore KV allocation, the
shared-memory step protocol, the TP broadcast of each step, ADBS -- is the
original code, unchanged (sglang_runtime.py ports LLMRuntime line by line).
"""
from __future__ import annotations

import contextlib
import itertools
import os
import time
from types import SimpleNamespace
from typing import List

import numpy as np
import torch
from torch.multiprocessing.reductions import rebuild_cuda_tensor

from muxserve.logger import get_logger

logger = get_logger()

EXECUTION_GROUP = "original-sglang"  # read by sglang_hooks.enabled()


class InputMetadata:
    """The fields of vLLM 0.2.0's InputMetadata that MuxServe's head-wise path
    reads (slot_mapping [tokens, layers, heads], block_tables [layers, tokens,
    heads, blocks]); the counts are computed exactly as vLLM does."""

    def __init__(self, prompt_lens: List[int], slot_mapping: torch.Tensor,
                 context_lens: torch.Tensor, max_context_len: int,
                 block_tables: torch.Tensor) -> None:
        if slot_mapping.ndim != 3:
            raise RuntimeError("MuxServe's head-wise slot mapping is "
                               "[tokens, layers, heads]")
        self.prompt_lens = prompt_lens
        self.slot_mapping = slot_mapping
        self.context_lens = context_lens
        self.max_context_len = max_context_len
        self.block_tables = block_tables
        self.num_prompts = len(prompt_lens)
        self.num_prompt_tokens = sum(prompt_lens)
        self.num_generation_tokens = context_lens.shape[0]
        self.num_valid_tokens = slot_mapping.shape[0]
        assert self.block_tables.shape[1] == self.num_generation_tokens
        assert context_lens.shape[0] == self.num_generation_tokens


class RaggedPrefill:
    """SGLang's default prompt attention on this GPU (FlashInferAttnBackend,
    extend without a cached prefix): one FlashInfer ragged prefill wrapper on
    SGLang's workspace size, planned once per step from the prompt lengths
    (qo_indptr = kv_indptr, as SGLang plans it) and run causally per layer."""

    def __init__(self, num_qo_heads: int, num_kv_heads: int, head_dim: int,
                 dtype: torch.dtype, device: torch.device) -> None:
        from flashinfer import BatchPrefillWithRaggedKVCacheWrapper
        from sglang.srt.environ import envs
        self.workspace = torch.empty(envs.SGLANG_FLASHINFER_WORKSPACE_SIZE.get(),
                                     dtype=torch.uint8, device=device)
        self.wrapper = BatchPrefillWithRaggedKVCacheWrapper(self.workspace, "NHD",
                                                            backend="auto")
        self.num_qo_heads = num_qo_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.dtype = dtype
        self.device = device

    def plan(self, prompt_lens: List[int]) -> None:
        lens = torch.tensor(prompt_lens, dtype=torch.int32, device=self.device)
        qo_indptr = torch.zeros(len(prompt_lens) + 1, dtype=torch.int32,
                                device=self.device)
        qo_indptr[1:] = torch.cumsum(lens, dim=0)
        self.wrapper.begin_forward(qo_indptr, qo_indptr, self.num_qo_heads,
                                   self.num_kv_heads, self.head_dim,
                                   q_data_type=self.dtype)

    def run(self, q, k, v, scale: float, logits_soft_cap: float) -> torch.Tensor:
        return self.wrapper.forward(q, k, v, causal=True, sm_scale=scale,
                                    logits_soft_cap=logits_soft_cap)


class HeadwiseAttention:
    """MuxServe-vLLM's PagedAttention.forward, head-wise branch, as an SGLang
    attention backend (RadixAttention calls forward with SGLang's q/k/v)."""

    def __init__(self, key_cache: torch.Tensor, value_cache: torch.Tensor,
                 num_heads: int, num_kv_heads: int,
                 prefill: RaggedPrefill = None) -> None:
        from sglang.kernels.ops.attention import headwise
        self.ops = headwise
        self.key_cache = key_cache
        self.value_cache = value_cache
        self.block_size = value_cache.shape[-1]
        self.prefill = prefill
        # PagedAttention.__init__: head_mapping, built once.
        self.head_mapping = torch.repeat_interleave(
            torch.arange(num_kv_heads, dtype=torch.int32, device=key_cache.device),
            num_heads // num_kv_heads)

    def forward(self, q, k, v, layer, forward_batch, save_kv_cache=True,
                **kwargs):
        if kwargs or not save_kv_cache:
            raise RuntimeError(f"Unsupported attention options {sorted(kwargs)}")
        meta: InputMetadata = forward_batch.input_metadata
        num_heads, num_kv_heads = layer.tp_q_head_num, layer.tp_k_head_num
        head_size = layer.head_dim
        query = q.view(-1, num_heads, head_size)
        key = k.view(-1, num_kv_heads, head_size)
        value = v.view(-1, num_kv_heads, head_size)
        output = torch.empty_like(query)

        # Prompt run: SGLang's default prompt attention (FlashInfer ragged).
        num_prompt_tokens = meta.num_prompt_tokens
        if num_prompt_tokens > 0:
            assert meta.num_generation_tokens == 0
            output[:num_prompt_tokens] = self.prefill.run(
                query[:num_prompt_tokens].contiguous(), key[:num_prompt_tokens],
                value[:num_prompt_tokens], layer.scaling, layer.logit_cap)

        # Store this step's K/V in the head-wise blocks FlexStore allocated.
        num_valid_tokens = meta.num_valid_tokens
        if num_valid_tokens > 0:
            self.ops.headwise_reshape_and_cache(
                key[:num_valid_tokens], value[:num_valid_tokens], self.key_cache,
                self.value_cache, meta.slot_mapping[:, layer.layer_id, :])

        # Decoding run: the head-wise single-query kernel over the block table.
        if meta.num_generation_tokens > 0:
            assert num_prompt_tokens == 0
            self.ops.single_query_cached_kv_headwise_attention(
                output[num_prompt_tokens:num_valid_tokens],
                query[num_prompt_tokens:num_valid_tokens], self.key_cache,
                self.value_cache, self.head_mapping, layer.scaling,
                meta.block_tables[layer.layer_id], meta.context_lens,
                self.block_size, meta.max_context_len, None)

        return output.view(-1, num_heads * head_size)


class _ForwardBatch:
    """What SGLang's LLaMA layers read from a forward batch on this path."""

    def __init__(self, forward_mode, input_metadata: InputMetadata):
        self.forward_mode = forward_mode
        self.input_metadata = input_metadata
        self.mha_return_lse = False


def _scratch_attention(attention: HeadwiseAttention, num_heads: int,
                       num_kv_heads: int) -> HeadwiseAttention:
    """A one-block pool of the real layout: capture warm-ups run for real and
    write K/V, so they write here instead of into FlexStore's pool."""
    key = attention.key_cache
    value = attention.value_cache
    return HeadwiseAttention(key.new_zeros((1, *key.shape[1:])),
                             value.new_zeros((1, *value.shape[1:])),
                             num_heads, num_kv_heads, attention.prefill)


# SGLang's decode CUDA graph (DecodeCudaGraphRunner, FULL backend, its default):
# the model's whole decode forward captured per batch size, at the batch sizes
# SGLang resolves for this machine (cuda_graph_config.decode.bs), largest first
# on SGLang's global graph pool, inside graph_capture() (graph-mode TP
# all-reduce), model_capture_mode() and freeze_gc, two warm-ups per size. A
# batch runs on the smallest captured size that holds it; larger ones eagerly.


class _GraphMetadata:
    """InputMetadata of a captured decode batch: views of static buffers."""

    def __init__(self, graphs, bs: int):
        self.prompt_lens = []
        self.num_prompts = 0
        self.num_prompt_tokens = 0
        self.num_generation_tokens = bs
        self.num_valid_tokens = bs
        self.slot_mapping = graphs.slots[:bs]
        self.block_tables = graphs.tables[:, :bs]
        self.context_lens = graphs.context_lens[:bs]
        self.max_context_len = graphs.max_context


class DecodeGraphs:

    def __init__(self, worker, sizes: List[int], max_context: int):
        from sglang.srt.distributed.parallel_state import graph_capture
        from sglang.srt.model_executor.forward_batch_info import ForwardMode
        from sglang.srt.model_executor.forward_context import (ForwardContext,
                                                               forward_context)
        from sglang.srt.model_executor.runner.base_cuda_graph_runner import freeze_gc
        from sglang.srt.model_executor.runner_utils.capture_mode import model_capture_mode
        from sglang.srt.model_executor.runner_utils.pool import (
            get_or_create_global_graph_capture_stream,
            get_or_create_global_graph_memory_pool)

        self.allocate(sizes, worker.num_hidden_layers, worker.num_kv_heads,
                      worker.attention.block_size, max_context,
                      worker.attention.key_cache.device)
        scratch = ForwardContext(attn_backend=_scratch_attention(
            worker.attention, worker.num_heads, worker.num_kv_heads))
        real = ForwardContext(attn_backend=worker.attention)
        pool = get_or_create_global_graph_memory_pool(torch.cuda)
        tic = time.perf_counter()
        with torch.inference_mode(), model_capture_mode(), freeze_gc(False), \
                graph_capture(stream=get_or_create_global_graph_capture_stream()) as context:
            for bs in reversed(self.sizes):
                batch = _ForwardBatch(ForwardMode.DECODE, _GraphMetadata(self, bs))
                with forward_context(scratch):
                    for _ in range(2):
                        torch.cuda.synchronize()
                        worker.tp_group.barrier()
                        worker.model.model(self.input_ids[:bs], self.positions[:bs], batch)
                graph = torch.cuda.CUDAGraph()
                with forward_context(real):
                    with torch.cuda.graph(graph, pool=pool, stream=context.stream):
                        out = worker.model.model(self.input_ids[:bs], self.positions[:bs], batch)
                self.graphs[bs], self.outputs[bs] = graph, out
        torch.cuda.synchronize()
        logger.info(f"Captured SGLang decode CUDA graphs for batch sizes {self.sizes} "
                    f"in {time.perf_counter() - tic:.1f} s")

    def allocate(self, sizes, num_layers, num_heads, block_size, max_context, device):
        self.sizes = sorted(sizes)
        self.max_bs = self.sizes[-1]
        self.max_context = max_context
        self.input_ids = torch.zeros(self.max_bs, dtype=torch.int64, device=device)
        self.positions = torch.zeros(self.max_bs, dtype=torch.int64, device=device)
        self.slots = torch.zeros((self.max_bs, num_layers, num_heads), dtype=torch.int64,
                                 device=device)
        self.tables = torch.zeros((num_layers, self.max_bs, num_heads,
                                   (max_context + block_size - 1) // block_size),
                                  dtype=torch.int32, device=device)
        self.context_lens = torch.ones(self.max_bs, dtype=torch.int32, device=device)
        self.graphs, self.outputs = {}, {}

    def run(self, input_tensor, positions, meta: InputMetadata) -> torch.Tensor:
        bs = meta.num_generation_tokens
        size = next(s for s in self.sizes if s >= bs)
        self.input_ids[:bs].copy_(input_tensor[:bs])
        self.positions[:bs].copy_(positions[:bs])
        self.slots[:bs].copy_(meta.slot_mapping)
        self.tables[:, :bs, :, :meta.block_tables.shape[-1]].copy_(meta.block_tables)
        self.context_lens[:bs].copy_(meta.context_lens)
        if size > bs:
            # Padding to the captured size, as SGLang pads a decode batch. SGLang
            # points padded rows at a reserved dummy slot of its own KV pool; the
            # original FlexStore pool has none, so padded rows repeat the last
            # request: they store exactly its K/V again and their outputs are
            # dropped.
            self.input_ids[bs:size] = self.input_ids[bs - 1]
            self.positions[bs:size] = self.positions[bs - 1]
            self.slots[bs:size] = self.slots[bs - 1]
            self.tables[:, bs:size] = self.tables[:, bs - 1:bs]
            self.context_lens[bs:size] = self.context_lens[bs - 1]
        self.graphs[size].replay()
        return self.outputs[size][:bs]


# SGLang's prefill CUDA graph (its BREAKABLE backend, the default on CUDA): per
# token bucket (cuda_graph_config.prefill.bs, as SGLang resolves it for this
# machine) everything but attention is captured as graph segments; at each
# attention the replay breaks out and runs it eagerly on the live batch, which
# SGLang's attention op reads from its piecewise forward context. The same
# machinery is used here (BreakableCUDAGraphCapture, enable_breakable_cuda_graph,
# set_tc_piecewise_forward_context, RadixAttention's breakable op), with one
# output buffer for all buckets as in SGLang's BreakableCudaGraphBackend.
# Attention runs on the real tokens only (global_num_token_non_padded_cpu), so
# the padded tail of a bucket never writes KV; larger prefills run eagerly.


class _PrefillForwardBatch(_ForwardBatch):
    """The ForwardBatch fields SGLang's breakable attention op reads."""

    def __init__(self, extend_mode, input_metadata, num_tokens: int, out_cache_loc):
        super().__init__(extend_mode, input_metadata)
        self.global_num_token_non_padded_cpu = num_tokens
        self.out_cache_loc = out_cache_loc  # only sliced by the op, never used here
        self.positions = None


def _scratch_prompt_metadata(tokens: int, layers: int, heads: int, device) -> InputMetadata:
    """One prompt of `tokens` tokens whose K/V all go to slot 0 (scratch pool)."""
    return InputMetadata([tokens],
                         torch.zeros((tokens, layers, heads), dtype=torch.int64, device=device),
                         torch.zeros((0,), dtype=torch.int32, device=device), 0,
                         torch.zeros((layers, 0, heads, 0), dtype=torch.int32, device=device))


class PrefillGraphs:

    def __init__(self, forward_fn, attention_layers, attention: HeadwiseAttention,
                 scratch: HeadwiseAttention, sizes, num_layers: int, num_heads: int,
                 capture_scope, barrier):
        from sglang.srt.model_executor.forward_batch_info import ForwardMode
        from sglang.srt.model_executor.forward_context import ForwardContext, forward_context
        from sglang.srt.model_executor.runner_backend_utils.breakable_cuda_graph import (
            BreakableCUDAGraph, BreakableCUDAGraphCapture, enable_breakable_cuda_graph)
        from sglang.srt.model_executor.runner_backend_utils.tc_piecewise_cuda_graph.context_manager import (
            set_tc_piecewise_forward_context)

        self._forward_context = forward_context
        self._enable = enable_breakable_cuda_graph
        self._set_context = set_tc_piecewise_forward_context
        self._extend = ForwardMode.EXTEND
        device = attention.key_cache.device
        self.sizes = sorted(sizes)
        self.max_tokens = self.sizes[-1]
        self.layers = attention_layers
        self.real = ForwardContext(attn_backend=attention)
        self.input_ids = torch.zeros(self.max_tokens, dtype=torch.int64, device=device)
        self.positions = torch.zeros(self.max_tokens, dtype=torch.int64, device=device)
        self.out_cache_loc = torch.zeros(self.max_tokens, dtype=torch.int64, device=device)
        self.graphs, self.outputs = {}, {}
        self.out_buffer = None
        scratch_context = ForwardContext(attn_backend=scratch)
        tic = time.perf_counter()
        with torch.inference_mode(), capture_scope() as scope, enable_breakable_cuda_graph():
            for tokens in reversed(self.sizes):
                meta = _scratch_prompt_metadata(tokens, num_layers, num_heads, device)
                batch = _PrefillForwardBatch(self._extend, meta, tokens, self.out_cache_loc)
                scratch.prefill.plan(meta.prompt_lens)
                with forward_context(scratch_context), set_tc_piecewise_forward_context(
                        batch, self.layers, None, [], [], num_tokens=tokens, raw_num_tokens=tokens):
                    # Two warm-ups per bucket, as SGLang's breakable backend does.
                    for _ in range(2):
                        torch.cuda.synchronize()
                        barrier()
                        warm = forward_fn(self.input_ids[:tokens], self.positions[:tokens], batch)
                    if self.out_buffer is None:  # largest bucket first
                        self.out_buffer = warm.new_empty((self.max_tokens, *warm.shape[1:]))
                    del warm
                    graph = BreakableCUDAGraph()
                    with BreakableCUDAGraphCapture(cuda_graph=graph, pool=scope.pool,
                                                   stream=scope.stream, barrier_fn=barrier):
                        out = forward_fn(self.input_ids[:tokens], self.positions[:tokens], batch)
                        self.out_buffer[:tokens].copy_(out)
                    del out
                self.graphs[tokens] = graph
                self.outputs[tokens] = self.out_buffer[:tokens]
        torch.cuda.synchronize()
        logger.info(f"Captured SGLang breakable prefill CUDA graphs for {len(self.sizes)} buckets "
                    f"{self.sizes[0]}..{self.max_tokens} tokens in {time.perf_counter() - tic:.1f} s")

    def run(self, input_tensor, positions, meta: InputMetadata) -> torch.Tensor:
        """Replays the bucket of this step; the prompt attention must already
        be planned for it (SGLang plans before the replay too)."""
        tokens = meta.num_prompt_tokens
        bucket = next(size for size in self.sizes if size >= tokens)
        self.input_ids[:tokens].copy_(input_tensor[:tokens])
        self.input_ids[tokens:bucket].zero_()
        self.positions[:tokens].copy_(positions[:tokens])
        self.positions[tokens:bucket].zero_()
        batch = _PrefillForwardBatch(self._extend, meta, tokens, self.out_cache_loc)
        with self._forward_context(self.real), self._set_context(
                batch, self.layers, None, [], [], num_tokens=bucket, raw_num_tokens=tokens), \
                self._enable():
            self.graphs[bucket].replay()
        return self.outputs[bucket][:tokens]


@contextlib.contextmanager
def _sglang_prefill_capture_scope():
    """SGLang's prefill capture environment (PrefillCudaGraphRunner.capture):
    frozen GC, graph-mode TP all-reduce on the global capture stream, its
    global graph memory pool (capture mode comes from the breakable flag)."""
    from sglang.srt.distributed.parallel_state import graph_capture
    from sglang.srt.model_executor.runner.base_cuda_graph_runner import freeze_gc
    from sglang.srt.model_executor.runner_utils.pool import (
        get_or_create_global_graph_capture_stream, get_or_create_global_graph_memory_pool)
    with freeze_gc(False), \
            graph_capture(stream=get_or_create_global_graph_capture_stream()) as context:
        yield SimpleNamespace(stream=context.stream,
                              pool=get_or_create_global_graph_memory_pool(torch.cuda))


def greedy_sample(lm_head_weight: torch.Tensor, hidden_states: torch.Tensor,
                  input_metadata: InputMetadata, vocab_size: int,
                  tp_group) -> List[int]:
    """flexserver/sampler.py GreedySampler.forward with vLLM's _get_logits
    (matmul, TP all-gather, cut to vocab_size)."""
    if input_metadata.num_prompts > 0:
        assert input_metadata.num_generation_tokens == 0
        indices = np.cumsum(input_metadata.prompt_lens) - 1
        indices = torch.tensor(indices, dtype=torch.int,
                               device=hidden_states.device)
        hidden_states = hidden_states.index_select(0, indices)

    logits = torch.matmul(hidden_states, lm_head_weight.t())
    logits = tp_group.all_gather(logits, dim=-1)
    logits = logits[:, :vocab_size]

    if input_metadata.num_prompts > 0:
        num_tokens = input_metadata.num_prompts
    if input_metadata.num_generation_tokens > 0:
        num_tokens = input_metadata.num_generation_tokens
    next_tokens = torch.argmax(logits, dim=-1)
    next_tokens = next_tokens.cpu().numpy().tolist()
    return next_tokens[:num_tokens]


def _local_rope_caches(model) -> int:
    """FlexStore lends parameters only. Each RoPE cos/sin cache is computed
    here on this rank's GPU, exactly as SGLang's RotaryEmbedding computes it
    when the model is built on the GPU."""
    device = torch.device("cuda", torch.cuda.current_device())
    count = 0
    with torch.device(device):
        for module in model.modules():
            cache = getattr(module, "cos_sin_cache", None)
            if (not isinstance(cache, torch.Tensor) or cache.device.type != "meta"
                    or not hasattr(module, "_compute_cos_sin_cache")):
                continue
            computed = module._compute_cos_sin_cache()
            if computed.shape != cache.shape or computed.dtype != cache.dtype:
                raise RuntimeError("Local RoPE cache differs from the model's")
            module.register_buffer("cos_sin_cache", computed, persistent=False)
            count += 1
    return count


def flexstore_model_loader(load_config, tcp_client, model_name):
    from sglang.srt.model_loader.loader import BaseModelLoader

    class FlexStoreModelLoader(BaseModelLoader):
        """model_loader.load_from_server for an SGLang model: build the model
        on the meta device (no memory), ask FlexStore for this rank's weights
        and map every parameter onto FlexStore's CUDA memory."""

        def download_model(self, model_config) -> None:
            pass

        def load_model(self, *, model_config, device_config):
            from sglang.srt.model_loader.loader import (
                _get_quantization_config, _initialize_model)
            from sglang.srt.model_loader.utils import set_default_torch_dtype
            from sglang.srt.weight_cache.ipc_loader import IpcModelLoader

            quant_config = _get_quantization_config(model_config,
                                                    self.load_config)
            if quant_config is not None:
                raise RuntimeError("FlexStore serves FP16 weights only")
            with set_default_torch_dtype(model_config.dtype):
                with torch.device("meta"):
                    model = _initialize_model(model_config, self.load_config,
                                              quant_config)

            rank = torch.distributed.get_rank()
            logger.info(f"Asking FlexStore for {model_name} rank {rank} weights")
            tcp_client.send_pyobj(["weight", [rank, model_name]])
            data = tcp_client.recv_pyobj()

            params = dict(model.named_parameters())
            if set(params) != set(data):
                raise RuntimeError(
                    "FlexStore weights differ from SGLang's parameters: only "
                    f"FlexStore {sorted(set(data) - set(params))[:8]}, only "
                    f"SGLang {sorted(set(params) - set(data))[:8]}")
            for name, param in params.items():
                tensor = rebuild_cuda_tensor(torch.Tensor, **data[name])
                if (not tensor.is_cuda or tensor.shape != param.shape
                        or tensor.dtype != param.dtype):
                    raise RuntimeError(
                        f"{name}: FlexStore {tuple(tensor.shape)} "
                        f"{tensor.dtype} vs SGLang {tuple(param.shape)} "
                        f"{param.dtype}")
                IpcModelLoader._set_module_tensor(model, name, tensor,
                                                  is_param=True)
            ropes = _local_rope_caches(model)
            left = [name for name, tensor in itertools.chain(
                model.named_parameters(), model.named_buffers())
                    if tensor.device.type == "meta"]
            if left:
                raise RuntimeError(f"Tensors left on meta: {left[:8]}")
            logger.info(f"Mapped {len(params)} FlexStore tensors, computed "
                        f"{ropes} RoPE caches")
            return model.eval()

    return FlexStoreModelLoader(load_config)


class SGLangWorker:
    """Builds SGLang's model on this rank (SGLang's own ModelRunner does the
    distributed setup and the model build) and runs one forward per step."""

    def __init__(self, model_path: str, model_name: str,
                 tensor_parallel_size: int, local_rank: int, tcp_client,
                 max_num_batched_tokens: int, decode_graphs: bool = False,
                 prefill_graphs: bool = False) -> None:
        torch.cuda.set_device(local_rank)
        os.environ["MUXSERVE_EXECUTION_GROUP"] = EXECUTION_GROUP
        # torch.distributed.launch already serves the rendezvous store on
        # MASTER_ADDR:MASTER_PORT; SGLang joins it like the original (env://).
        os.environ["SGLANG_DISTRIBUTED_INIT_METHOD_OVERRIDE"] = "env://"
        # SGLang's FlashInfer workspace (default 384 MB per process) at 128 MB,
        # FlashInfer's own recommended size: six rank processes share the
        # ~12.6 GB FlexStore leaves on each 40 GB GPU, and 6 x 384 MB does not
        # fit next to the CUDA graphs. Only the temporary space shrinks; the
        # kernels and the computation are SGLang's default ones.
        os.environ.setdefault("SGLANG_FLASHINFER_WORKSPACE_SIZE", str(128 * 1024 * 1024))

        from sglang.srt.arg_groups.overrides import (attention_backends_of,
                                                     resolved_view, resolving_view)
        from sglang.srt.configs.model_config import ModelConfig
        from sglang.srt.distributed import get_tp_group
        from sglang.srt.distributed.parallel_state_wrapper import ParallelState
        from sglang.srt.layers.dp_attention import compute_dp_attention_world_info
        from sglang.srt.layers.moe import initialize_moe_config
        from sglang.srt.layers.quantization.fp4_utils import initialize_fp4_gemm_config
        from sglang.srt.layers.quantization.fp8_utils import initialize_fp8_gemm_config
        from sglang.srt.model_executor.model_runner import ModelRunner
        from sglang.srt.runtime_context import get_exec, publish
        from sglang.srt.server_args import ServerArgs
        from sglang.srt.utils import configure_logger
        from muxserve.flexserver import sglang_hooks

        master_port = int(os.environ["MASTER_PORT"])
        tp_rank = int(os.environ["RANK"])
        # Only what the original's command line fixes; SGLang resolves the rest
        # for this machine. The original's per-batch token limit
        # (--max-num-batched-tokens, 2048: the scheduler never sends a larger
        # prefill) is SGLang's chunked prefill size, from which SGLang derives
        # its prefill CUDA-graph buckets. Each model group has its own
        # MASTER_PORT, so the ports derived from it never collide between groups.
        server_args = ServerArgs(model_path=model_path, tp_size=tensor_parallel_size,
                                 chunked_prefill_size=max_num_batched_tokens,
                                 port=master_port + 1000, nccl_port=master_port)
        server_args.resolve_once()
        server_args.check_server_args()
        publish(server_args, role="scheduler")
        initialize_moe_config()
        initialize_fp8_gemm_config()
        initialize_fp4_gemm_config()
        configure_logger(server_args, prefix=f" {model_name} TP{tp_rank}")

        cfg = resolving_view(server_args)
        model_config = ModelConfig.from_server_args(server_args)
        attn_tp_rank, attn_tp_size, attn_dp_rank, attn_dp_size = (
            compute_dp_attention_world_info(cfg.enable_dp_attention, tp_rank,
                                            cfg.tp_size, cfg.dp_size,
                                            cfg.attn_cp_size))
        ps = ParallelState(
            tp_rank=tp_rank, tp_size=cfg.tp_size, pp_rank=0, pp_size=1,
            dp_rank=None, dp_size=cfg.dp_size, attn_tp_rank=attn_tp_rank,
            attn_tp_size=attn_tp_size, attn_cp_rank=0,
            attn_cp_size=cfg.attn_cp_size, attn_dcp_rank=tp_rank % cfg.dcp_size,
            attn_dcp_size=cfg.dcp_size, attn_dp_rank=attn_dp_rank,
            attn_dp_size=attn_dp_size,
            moe_ep_rank=tp_rank // (cfg.tp_size // cfg.ep_size),
            moe_ep_size=cfg.ep_size, moe_dp_rank=None,
            moe_dp_size=cfg.moe_dp_size, gpu_id=local_rank)

        sglang_hooks.set_weight_source(
            lambda load_config: flexstore_model_loader(load_config, tcp_client,
                                                       model_name))
        # Only the model build: SGLang's KV pool, attention backends and CUDA
        # graphs are never created by ModelRunner here (alloc_memory_pool etc.
        # are not called); KV lives in FlexStore and attention is
        # HeadwiseAttention.
        self.model_runner = ModelRunner(
            model_config=model_config,
            mem_fraction_static=cfg.mem_fraction_static, gpu_id=local_rank,
            ps=ps, nccl_port=master_port, server_args=server_args)
        self.model = self.model_runner.model
        self.hf_config = model_config.hf_config
        self.vocab_size = self.hf_config.vocab_size
        self.num_hidden_layers = self.hf_config.num_hidden_layers
        attn = self.model.model.layers[0].self_attn.attn
        self.num_heads = attn.tp_q_head_num
        self.num_kv_heads = attn.tp_k_head_num
        self.head_dim = attn.head_dim
        self.max_context = model_config.context_len
        self.tp_group = get_tp_group()
        self.attention = None

        # SGLang's resolution for this machine.
        prefill_backend, decode_backend = attention_backends_of(resolved_view(server_args))
        graph_config = get_exec().graph.cuda_graph_config
        self.decode_graph_config = graph_config.decode
        self.prefill_graph_config = graph_config.prefill
        logger.info(
            f"SGLang configuration resolved for this machine: attention prefill "
            f"{prefill_backend} / decode {decode_backend}; decode CUDA graph "
            f"{graph_config.decode.backend} batch sizes {graph_config.decode.bs}; "
            f"prefill CUDA graph {graph_config.prefill.backend} token buckets "
            f"{graph_config.prefill.bs}; custom all-reduce "
            f"{'off' if cfg.disable_custom_all_reduce else 'on'}")
        if prefill_backend != "flashinfer":
            raise RuntimeError(f"SGLang resolved prefill attention {prefill_backend!r}; "
                               "this engine implements its FlashInfer path only")
        # A prefill rank uses the prefill graphs, a decode rank the decode
        # graphs (SGLang's rule for disaggregated prefill / decode servers).
        self.capture_decode_graphs = (decode_graphs and
                                      self._graph_enabled(graph_config.decode, "full"))
        self.capture_prefill_graphs = (prefill_graphs and
                                       self._graph_enabled(graph_config.prefill, "breakable"))
        self.decode_graphs = None
        self.prefill_graphs = None

    @staticmethod
    def _graph_enabled(phase_config, implemented: str) -> bool:
        backend = str(phase_config.backend)
        if backend == "disabled":
            return False
        if backend != implemented:
            raise RuntimeError(f"SGLang resolved CUDA graph backend {backend!r}; "
                               f"this engine implements {implemented!r} for this phase")
        return True

    def set_cache(self, key_cache: torch.Tensor, value_cache: torch.Tensor):
        from sglang.srt.model_executor.forward_context import ForwardContext
        prefill = RaggedPrefill(self.num_heads, self.num_kv_heads, self.head_dim,
                                self.model_runner.dtype, key_cache.device)
        self.attention = HeadwiseAttention(key_cache, value_cache, self.num_heads,
                                           self.num_kv_heads, prefill)
        self.forward_context = ForwardContext(attn_backend=self.attention)
        if self.capture_decode_graphs:
            self.decode_graphs = DecodeGraphs(self, self.decode_graph_config.bs,
                                              self.max_context)
        if self.capture_prefill_graphs:
            self.prefill_graphs = PrefillGraphs(
                self.model.model, [layer.self_attn.attn for layer in self.model.model.layers],
                self.attention,
                _scratch_attention(self.attention, self.num_heads, self.num_kv_heads),
                self.prefill_graph_config.bs, self.num_hidden_layers, self.num_kv_heads,
                _sglang_prefill_capture_scope, self.tp_group.barrier)

    def _prepare_inputs_metadata(self, prompt_lens: List[int],
                                 slot_mapping: torch.Tensor,
                                 context_lens: torch.Tensor,
                                 max_context_len: int,
                                 block_tables: torch.Tensor) -> InputMetadata:
        return InputMetadata(prompt_lens, slot_mapping, context_lens,
                             max_context_len, block_tables)

    @torch.inference_mode()
    def pure_forward(self, input_tensor: torch.Tensor, positions: torch.Tensor,
                     input_metadata: InputMetadata) -> List[int]:
        from sglang.srt.model_executor.forward_batch_info import ForwardMode
        from sglang.srt.model_executor.forward_context import forward_context

        prompt_tokens = input_metadata.num_prompt_tokens
        if prompt_tokens > 0:
            # SGLang plans the prompt attention per batch, before the forward
            # or the graph replay.
            self.attention.prefill.plan(input_metadata.prompt_lens)
            graphs = self.prefill_graphs
            if graphs is not None and prompt_tokens <= graphs.max_tokens:
                hidden_states = graphs.run(input_tensor, positions, input_metadata)
            else:
                with forward_context(self.forward_context):
                    hidden_states = self.model.model(
                        input_tensor, positions,
                        _ForwardBatch(ForwardMode.EXTEND, input_metadata))
        else:
            graphs = self.decode_graphs
            if graphs is not None and input_metadata.num_generation_tokens <= graphs.max_bs:
                hidden_states = graphs.run(input_tensor, positions, input_metadata)
            else:
                with forward_context(self.forward_context):
                    hidden_states = self.model.model(
                        input_tensor, positions,
                        _ForwardBatch(ForwardMode.DECODE, input_metadata))
        return greedy_sample(self.model.lm_head.weight, hidden_states,
                             input_metadata, self.vocab_size, self.tp_group)
