"""Eager FP16 MHA attention using original Head-wise store/decode kernels."""

from __future__ import annotations

import math

import torch
from torch.nn.functional import scaled_dot_product_attention
from sglang.srt.layers.attention.base_attn_backend import AttentionBackend
from sglang.srt.layers.radix_attention import AttentionType

from . import ops
from .contracts import IntegrationError
from .sglang_memory import HeadwiseTokenToKVPool, _require


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

    def init_forward_metadata(self, forward_batch):
        _require(not torch.cuda.is_current_stream_capturing(), "CUDA graphs are unsupported")
        self.forward_metadata = self.token_to_kv_pool.validate_forward(forward_batch)

    def init_forward_metadata_out_graph(self, forward_batch, in_capture=False):
        _require(not in_capture, "CUDA graph capture is unsupported")
        self.init_forward_metadata(forward_batch)

    def init_forward_metadata_in_graph(self, forward_batch):
        _require(not torch.cuda.is_current_stream_capturing(), "CUDA graphs are unsupported")

    def init_cuda_graph_state(self, *args, **kwargs):
        raise IntegrationError("UNSUPPORTED", "Head-wise execution requires eager forwards")

    def forward(self, q, k, v, layer, forward_batch, save_kv_cache=True, **kwargs):
        # Override the native idle fast path: an unbound or idle call must fail.
        self.token_to_kv_pool.validate_forward(forward_batch)
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
        outputs, start = [], 0
        for length in bound.write_lengths:
            end = start + length
            # Independent segments are essential: concatenated requests must
            # never share attention even though their tokens share a forward.
            output = scaled_dot_product_attention(
                q[start:end].transpose(0, 1).unsqueeze(0),
                k[start:end].transpose(0, 1).unsqueeze(0),
                v[start:end].transpose(0, 1).unsqueeze(0),
                dropout_p=0.0, is_causal=True, scale=layer.scaling,
            )
            outputs.append(output.squeeze(0).transpose(0, 1))
            start = end
        return torch.cat(outputs, dim=0).reshape(bound.total_written, -1)

    def forward_decode(self, q, k, v, layer, forward_batch, save_kv_cache=True, **kwargs):
        bound, (q, k, v) = self._inputs(q, k, v, layer, forward_batch, save_kv_cache, kwargs)
        _require(bound.mode == "DECODE", "Decode requires a one-token append grant")
        self.token_to_kv_pool.set_kv_buffer(layer, forward_batch.out_cache_loc, k, v)
        _, table = self.token_to_kv_pool.layer_metadata(layer.layer_id)
        output = ops.decode(q, self.token_to_kv_pool.pool_key, self.token_to_kv_pool.pool_value,
                            table, bound.context_lens, layer.scaling, max(bound.target_lengths))
        return output.reshape(bound.total_written, -1)
