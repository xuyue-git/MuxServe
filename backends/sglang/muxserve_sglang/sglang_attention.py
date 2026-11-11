"""Eager FP16 MHA attention using original Head-wise store/decode kernels."""

from __future__ import annotations

import math

import torch
from torch.nn.functional import scaled_dot_product_attention
from sglang.srt.layers.attention.base_attn_backend import AttentionBackend
from sglang.srt.layers.radix_attention import AttentionType

from . import ops
from .sglang_memory import GRAPH_MAX_CONTEXT, HeadwiseTokenToKVPool, _require

try:  # SGLang breakable (segmented) prefill CUDA graphs run attention eagerly.
    from sglang.srt.model_executor.runner_backend_utils.breakable_cuda_graph import (
        is_in_breakable_cuda_graph)
except ImportError:  # pragma: no cover - older native trees
    def is_in_breakable_cuda_graph():
        return False


class HeadwiseAttentionBackend(AttentionBackend):
    def __init__(self, runner):
        super().__init__()
        _require(isinstance(runner.token_to_kv_pool, HeadwiseTokenToKVPool),
                 "Head-wise backend requires the imported Head-wise KV pool")
        self.token_to_kv_pool = runner.token_to_kv_pool
        self.data_type = self.kv_cache_dtype = torch.float16
        self.forward_metadata = None
        # Static per-layer configuration is validated once per layer object;
        # tensors, shapes and the bound grant are still checked on every call.
        self._validated_layers = set()
        self._static = False

    def init_forward_metadata(self, forward_batch):
        # Eager entry point (eager runner): always the bound, validated grant.
        _require(not torch.cuda.is_current_stream_capturing(), "Eager metadata cannot be captured")
        self._static = False
        if self.token_to_kv_pool._bound is None and is_in_breakable_cuda_graph():
            # P-stage prefill graph capture/warmup at model-runner init, before
            # any grant: attention runs eagerly at graph breaks and writes no KV.
            self.forward_metadata = None
            return
        self.forward_metadata = self.token_to_kv_pool.validate_forward(forward_batch)

    # ---- decode CUDA graph (native DecodeCudaGraphRunner, D stage only) ----
    # Capture/warmup forwards run once at model-runner init, before any grant,
    # on static per-layer slot/page-table/length buffers. Every replay is
    # preceded by ModelRunner.forward's grant check and validate_forward of the
    # live batch; init_forward_metadata_out_graph then copies that validated
    # grant's slots/tables/lengths into the same static buffers.
    def init_cuda_graph_state(self, max_bs, max_num_tokens):
        _require(max_num_tokens == max_bs, "Head-wise decode graphs carry one token per request")
        self.token_to_kv_pool.init_graph_buffers(max_bs)

    def get_cuda_graph_seq_len_fill_value(self):
        return 1

    def init_forward_metadata_out_graph(self, forward_batch, in_capture=False):
        _require(forward_batch.forward_mode.is_decode(), "Only decode forwards are graph-captured")
        if in_capture:
            self.token_to_kv_pool.graph_capture_metadata(forward_batch.batch_size)
            self._static = True
        else:
            # Replay prep (runs eagerly outside the graph).
            self.token_to_kv_pool.prepare_graph_replay(forward_batch.batch_size)
            self._static = False
        self.forward_metadata = None

    def init_forward_metadata_in_graph(self, forward_batch):
        """Nothing to record: replay prep wrote the static buffers eagerly."""

    def forward(self, q, k, v, layer, forward_batch, save_kv_cache=True, **kwargs):
        if getattr(self, "_static", False):
            return self._forward_decode_static(q, k, v, layer, forward_batch, save_kv_cache, kwargs)
        if is_in_breakable_cuda_graph() and not forward_batch.forward_mode.is_decode():
            return self._forward_extend_graph_break(q, k, v, layer, forward_batch, save_kv_cache, kwargs)
        # Override the native idle fast path: an unbound or idle call must
        # fail. Both methods validate the grant first thing (via _inputs).
        method = self.forward_decode if forward_batch.forward_mode.is_decode() else self.forward_extend
        return method(q, k, v, layer, forward_batch, save_kv_cache=save_kv_cache, **kwargs)

    def _inputs(self, q, k, v, layer, forward_batch, save_kv_cache, kwargs):
        _require(not torch.cuda.is_current_stream_capturing(), "CUDA graphs are unsupported")
        bound = self.token_to_kv_pool.validate_forward(forward_batch)
        _require(save_kv_cache and not kwargs, "Each granted layer must write its K/V without extra options")
        _require(not getattr(forward_batch, "mha_return_lse", False),
                 "MHA log-sum-exp output is unsupported")
        heads = self.token_to_kv_pool.head_num
        if id(layer) not in self._validated_layers:
            self._validate_layer(layer, heads)
            self._validated_layers.add(id(layer))
        expected = (bound.total_written, heads, 128)
        shaped = []
        for tensor in (q, k, v):
            _require(isinstance(tensor, torch.Tensor), "Q/K/V must all be present")
            _require(tensor.dtype == torch.float16 and tensor.device == self.token_to_kv_pool.device,
                     "Q/K/V must use the imported pool CUDA device and FP16")
            _require(tuple(tensor.shape) in (expected, (expected[0], heads * 128)),
                     "Q/K/V geometry mismatch")
            shaped.append(tensor.reshape(expected).contiguous())
        return bound, shaped

    def _validate_layer(self, layer, heads):
        xai_temperature_len = getattr(layer, "xai_temperature_len", -1)
        _require(xai_temperature_len is None or xai_temperature_len <= 0,
                 "xAI temperature attention is unsupported")
        _require(layer.tp_q_head_num == layer.tp_k_head_num == layer.tp_v_head_num == heads
                 and layer.head_dim == layer.qk_head_dim == layer.v_head_dim == 128,
                 "Only local MHA with H128 is supported")
        _require(layer.attn_type == AttentionType.DECODER and not layer.is_cross_attention
                 and layer.pos_encoding_mode == "NONE" and not layer.use_irope
                 and layer.logit_cap == 0 and layer.sliding_window_size == -1,
                 "Only causal decoder MHA without ALiBi/capping/sliding-window is supported")
        _require(getattr(layer, "k_scale", None) is None and getattr(layer, "v_scale", None) is None,
                 "KV quantization/scales are unsupported")
        _require(isinstance(layer.scaling, (float, int)) and math.isfinite(layer.scaling)
                 and layer.scaling > 0, "Invalid attention scale")

    def forward_extend(self, q, k, v, layer, forward_batch, save_kv_cache=True, **kwargs):
        bound, (q, k, v) = self._inputs(q, k, v, layer, forward_batch, save_kv_cache, kwargs)
        _require(bound.mode in ("PREFILL", "REBUILD"), "Extend requires full prefill or rebuild")
        self.token_to_kv_pool.set_kv_buffer(layer, forward_batch.out_cache_loc, k, v)
        return self._extend_attention(q, k, v, bound.write_lengths, layer.scaling)

    @staticmethod
    def _extend_attention(q, k, v, write_lengths, scaling):
        total = sum(write_lengths)
        if len(write_lengths) == 1:
            # One request: same attention without slicing or a concatenation.
            output = scaled_dot_product_attention(
                q.transpose(0, 1).unsqueeze(0), k.transpose(0, 1).unsqueeze(0),
                v.transpose(0, 1).unsqueeze(0), dropout_p=0.0, is_causal=True, scale=scaling)
            return output.squeeze(0).transpose(0, 1).reshape(total, -1)
        outputs, start = [], 0
        for length in write_lengths:
            end = start + length
            # Independent segments are essential: concatenated requests must
            # never share attention even though their tokens share a forward.
            output = scaled_dot_product_attention(
                q[start:end].transpose(0, 1).unsqueeze(0),
                k[start:end].transpose(0, 1).unsqueeze(0),
                v[start:end].transpose(0, 1).unsqueeze(0),
                dropout_p=0.0, is_causal=True, scale=scaling,
            )
            outputs.append(output.squeeze(0).transpose(0, 1))
            start = end
        return torch.cat(outputs, dim=0).reshape(total, -1)

    # ---- P-stage breakable prefill CUDA graph (prefill_cuda_graph) ----------
    # Native BCG captures the transformer body in segments and runs attention
    # eagerly at the graph breaks, with the runner's static batch (Q/K/V already
    # narrowed to the real tokens). ModelRunner.forward validated the live
    # grant first (validate_forward sets extend_checked); here the static batch
    # must carry exactly that grant's geometry, and K/V go to its write slots.
    def _forward_extend_graph_break(self, q, k, v, layer, forward_batch, save_kv_cache, kwargs):
        _require(not torch.cuda.is_current_stream_capturing(),
                 "Graph-break attention runs eagerly, never inside a captured segment")
        _require(save_kv_cache and not kwargs and not getattr(forward_batch, "mha_return_lse", False),
                 "Each granted layer must write its K/V without extra options")
        pool = self.token_to_kv_pool
        heads = pool.head_num
        if id(layer) not in self._validated_layers:
            self._validate_layer(layer, heads)
            self._validated_layers.add(id(layer))
        rows = q.shape[0]
        if pool._bound is None:
            # Capture/warmup at init: no grant exists, so nothing is stored.
            shaped = [tensor.reshape(rows, heads, 128) for tensor in (q, k, v)]
            return self._extend_attention(*shaped, (rows,), layer.scaling)
        bound = pool.bound_forward
        if self._break_slim():
            return self._graph_break_slim(q, k, v, layer, forward_batch, bound, heads, rows)
        _require(bound.mode in ("PREFILL", "REBUILD") and bound.extend_checked and bound.batch is not None,
                 "Prefill graph replay requires the validated live grant", "FORWARD_MISMATCH")
        _require(forward_batch.batch_size == len(bound.req_pool_indices)
                 and getattr(forward_batch, "global_num_token_non_padded_cpu", None) == bound.total_written
                 and rows == bound.total_written,
                 "Prefill graph replay geometry differs from the grant", "FORWARD_MISMATCH")
        shaped = []
        for tensor in (q, k, v):
            _require(isinstance(tensor, torch.Tensor) and tensor.dtype == torch.float16
                     and tensor.device == pool.device
                     and tuple(tensor.shape) in ((rows, heads, 128), (rows, heads * 128)),
                     "Q/K/V geometry mismatch")
            shaped.append(tensor.reshape(rows, heads, 128).contiguous())
        q, k, v = shaped
        ops.store(k, v, pool.pool_key, pool.pool_value, pool.layer_slots(layer.layer_id))
        return self._extend_attention(q, k, v, bound.write_lengths, layer.scaling)

    # ---- prefill_break_slim: the same graph break with less per-layer work ----
    # The grant's mode and Q/K/V dtype/device are checked once per forward (its first
    # layer); every layer still checks the batch geometry, its row count and shapes. Q/K/V stay views
    # of the projection output (the Head-wise store takes a token stride, attention any
    # strides), and attention writes straight into the output buffer SGLang's graph
    # break provides, so SGLang's own copy of the result is skipped.
    def _break_slim(self):
        slim = getattr(self, "_slim", None)
        if slim is None:
            try:
                from .sglang_hooks import context
                slim = bool(getattr(context().config, "prefill_break_slim", False))
            except Exception:
                slim = False
            self._slim = slim
        return slim

    def _graph_break_slim(self, q, k, v, layer, forward_batch, bound, heads, rows):
        pool = self.token_to_kv_pool
        if getattr(self, "_slim_checked", None) is not bound:
            _require(bound.mode in ("PREFILL", "REBUILD") and bound.extend_checked and bound.batch is not None,
                     "Prefill graph replay requires the validated live grant", "FORWARD_MISMATCH")
            for tensor in (q, k, v):
                _require(isinstance(tensor, torch.Tensor) and tensor.dtype == torch.float16
                         and tensor.device == pool.device, "Q/K/V must use the pool CUDA device and FP16")
            self._slim_checked = bound
        _require(forward_batch.batch_size == len(bound.req_pool_indices)
                 and getattr(forward_batch, "global_num_token_non_padded_cpu", None) == bound.total_written
                 and rows == bound.total_written,
                 "Prefill graph replay geometry differs from the grant", "FORWARD_MISMATCH")
        shapes = ((rows, heads, 128), (rows, heads * 128))
        _require(tuple(q.shape) in shapes and tuple(k.shape) in shapes and tuple(v.shape) in shapes,
                 "Q/K/V geometry mismatch")
        q, k, v = (self._head_view(tensor, rows, heads) for tensor in (q, k, v))
        ops.store(k, v, pool.pool_key, pool.pool_value, pool.layer_slots(layer.layer_id))
        out = getattr(forward_batch, "_attn_output", None)
        if out is None or tuple(out.shape) != (rows, heads * 128) or not out.is_contiguous():
            out = q.new_empty((rows, heads * 128))
        self._attention_into(q, k, v, bound.write_lengths, layer.scaling, out.view(rows, heads, 128))
        return out

    @staticmethod
    def _head_view(tensor, rows, heads):
        """[rows, heads, 128] without a copy when the last dimension is dense."""
        if tensor.dim() == 3 and tensor.stride(2) == 1 and tensor.stride(1) == 128:
            return tensor
        if tensor.stride(-1) == 1:
            try:
                return tensor.view(rows, heads, 128)
            except RuntimeError:
                pass
        return tensor.reshape(rows, heads, 128).contiguous()

    @staticmethod
    def _attention_into(q, k, v, write_lengths, scaling, target):
        """Causal attention of each request's own tokens, written into ``target``
        ([rows, heads, 128]); requests never attend to each other."""
        start = 0
        for length in write_lengths:
            end = start + length
            output = scaled_dot_product_attention(
                q[start:end].transpose(0, 1).unsqueeze(0), k[start:end].transpose(0, 1).unsqueeze(0),
                v[start:end].transpose(0, 1).unsqueeze(0), dropout_p=0.0, is_causal=True, scale=scaling)
            target[start:end].copy_(output.squeeze(0).transpose(0, 1))
            start = end

    def _forward_decode_static(self, q, k, v, layer, forward_batch, save_kv_cache, kwargs):
        """Graph warmup/capture body; replay re-executes exactly these kernels."""
        pool = self.token_to_kv_pool
        _require(forward_batch.forward_mode.is_decode() and pool._bound is None
                 and pool.graph is not None and pool.graph.static,
                 "Static graph forwards exist only for capture before any grant", "FORWARD_NOT_GRANTED")
        _require(save_kv_cache and not kwargs and not getattr(forward_batch, "mha_return_lse", False),
                 "Each graph layer must write its K/V without extra options")
        heads = pool.head_num
        if id(layer) not in self._validated_layers:
            self._validate_layer(layer, heads)
            self._validated_layers.add(id(layer))
        bs = forward_batch.batch_size
        shaped = []
        for tensor in (q, k, v):
            _require(isinstance(tensor, torch.Tensor) and tensor.dtype == torch.float16
                     and tensor.device == pool.device
                     and tuple(tensor.shape) in ((bs, heads, 128), (bs, heads * 128)),
                     "Q/K/V graph geometry mismatch")
            shaped.append(tensor.reshape(bs, heads, 128).contiguous())
        q, k, v = shaped
        slots, table, lengths = pool.graph_layer_metadata(layer.layer_id, bs)
        ops.store(k, v, pool.pool_key, pool.pool_value, slots)
        output = ops.decode(q, pool.pool_key, pool.pool_value, table, lengths, layer.scaling,
                            GRAPH_MAX_CONTEXT)
        return output.reshape(bs, -1)

    def forward_decode(self, q, k, v, layer, forward_batch, save_kv_cache=True, **kwargs):
        bound, (q, k, v) = self._inputs(q, k, v, layer, forward_batch, save_kv_cache, kwargs)
        _require(bound.mode == "DECODE", "Decode requires a one-token append grant")
        self.token_to_kv_pool.set_kv_buffer(layer, forward_batch.out_cache_loc, k, v)
        _, table = self.token_to_kv_pool.layer_metadata(layer.layer_id)
        output = ops.decode(q, self.token_to_kv_pool.pool_key, self.token_to_kv_pool.pool_value,
                            table, bound.context_lens, layer.scaling, max(bound.target_lengths))
        return output.reshape(bound.total_written, -1)
