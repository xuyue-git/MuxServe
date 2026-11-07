"""Fixed SGLang adapter over controller-owned original Head-wise buffers.

Logical SGLang token IDs never index these buffers. Physical addresses come only
from an explicitly bound, externally granted KVView snapshot.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
import torch
from sglang.srt.mem_cache.memory_pool import KVCache

from . import ops
from .contracts import IntegrationError, KVView, ModelLayout, PackedBlocks


def _require(condition: bool, message: str, code: str = "LAYOUT_MISMATCH") -> None:
    if not condition:
        raise IntegrationError(code, message)


def _cpu_ints(value, name: str) -> tuple[int, ...]:
    _require(value is not None, f"Missing {name}", "FORWARD_MISMATCH")
    if isinstance(value, torch.Tensor):
        _require(value.device.type == "cpu" and value.ndim == 1,
                 f"{name} must be a one-dimensional CPU mirror", "FORWARD_MISMATCH")
        value = value.tolist()
    result = tuple(value)
    _require(all(type(v) is int for v in result), f"{name} must contain integers")
    return result


@dataclass
class BoundForward:
    mode: str
    req_pool_indices: tuple[int, ...]
    target_lengths: tuple[int, ...]
    write_lengths: tuple[int, ...]
    write_starts: tuple[int, ...]
    # Frozen CPU copies: request -> layer -> head -> physical block IDs.
    tables: tuple[tuple[tuple[tuple[int, ...], ...], ...], ...]
    batch: object | None = None
    layer_metadata: dict[int, tuple[torch.Tensor, torch.Tensor]] = field(default_factory=dict)
    metadata_buffers: tuple[torch.Tensor, torch.Tensor] | None = None
    context_lens: torch.Tensor | None = None
    # Reserved token capacity per request; a decode run advances within it.
    capacities: tuple[int, ...] = ()
    # Per-request numpy copies of ``tables`` and the uploaded padded block
    # table. Both depend only on the frozen tables, so a decode run that
    # advances inside its reserved capacity reuses them for every step.
    np_tables: list | None = None
    table_buffer: torch.Tensor | None = None

    @property
    def total_written(self) -> int:
        return sum(self.write_lengths)


class HeadwiseTokenToKVPool(KVCache):
    """One imported physical K/V pair shared by all layers and local heads."""

    headwise_shared = True

    def __init__(self, layout: ModelLayout, rank: int, pool_key: torch.Tensor,
                 pool_value: torch.Tensor, logical_token_capacity: int):
        _require(rank in layout.ranks, "Rank absent from model layout")
        _require(type(logical_token_capacity) is int and logical_token_capacity > 0,
                 "Logical token capacity must be positive")
        _require(pool_key.dtype == pool_value.dtype == torch.float16,
                 "Shared Head-wise buffers must be FP16")
        _require(pool_key.is_cuda and pool_key.device == pool_value.device,
                 "Shared Head-wise buffers must be on the same CUDA device")
        _require(pool_key.is_contiguous() and pool_value.is_contiguous(),
                 "Shared Head-wise buffers must be contiguous")
        _require(pool_key.ndim == 4 and tuple(pool_key.shape[1:]) == (16, 16, 8),
                 "Original K layout must be [N,16,16,8]")
        _require(tuple(pool_value.shape) == (pool_key.shape[0], 128, 16)
                 and pool_key.shape[0] > 0, "Original V layout must be [N,128,16]")
        self.layout, self.rank = layout, rank
        self.pool_key, self.pool_value = pool_key, pool_value
        self.physical_block_count = pool_key.shape[0]
        # Do not call KVCache.__init__: it initializes optional native custom
        # memory pools. Imported storage has exactly one authoritative owner.
        self.size, self.page_size, self.kernel_page_blocks = logical_token_capacity, 1, 1
        self.dtype = self.store_dtype = torch.float16
        self.device = pool_key.device
        self.head_num, self.head_dim = layout.kv_heads_per_rank, 128
        self.v_head_dim, self.layer_num = 128, layout.num_layers
        self.start_layer, self.end_layer = 0, layout.num_layers - 1
        self.kv_buffer = (pool_key, pool_value)
        self.k_buffer, self.v_buffer = (pool_key,), (pool_value,)
        self.use_mla = self.use_hnd = self.is_quantized_kv_cache = False
        self.quant_method = None
        self.allocation_label = "controller-owned Head-wise shared pool"
        self.enable_custom_mem_pool, self.custom_mem_pool = False, None
        self.memory_saver_adapter = None
        self.layer_transfer_counter = None
        self.cpu_offloading_chunk_size = 8192
        self.host_capacity_tokens = logical_token_capacity
        self.host_capacity_bytes = sum(self.get_kv_size_bytes())
        self.mem_usage = self.host_capacity_bytes / (1 << 30)
        self._bound: BoundForward | None = None

    @property
    def bound_req_pool_indices(self) -> tuple[int, ...]:
        return self.bound_forward.req_pool_indices

    @property
    def bound_forward(self) -> BoundForward:
        _require(self._bound is not None, "No externally granted forward is bound", "UNBOUND_FORWARD")
        return self._bound

    def bind_forward(self, views: Sequence[KVView], req_pool_indices: Sequence[int],
                     target_lengths: Sequence[int], mode: str) -> None:
        _require(self._bound is None, "Previous forward has not been cleared", "FORWARD_BUSY")
        _require(mode in ("PREFILL", "DECODE", "REBUILD"), "Unsupported bound mode")
        rows = _cpu_ints(req_pool_indices, "bound request rows")
        targets = _cpu_ints(target_lengths, "bound target lengths")
        views = tuple(views)
        _require(len(views) == len(rows) == len(targets) and len(views) > 0,
                 "Bound request metadata lengths differ")
        _require(len(set(rows)) == len(rows) and all(r >= 0 for r in rows),
                 "Duplicate or negative request row")
        _require(len({v.identity for v in views}) == len(views)
                 and len({v.allocation_id for v in views}) == len(views),
                 "Duplicate request identity or allocation")
        tables, starts, writes = [], [], []
        used_blocks: dict[int, list] = {r: [] for r in self.layout.ranks}
        for view, target in zip(views, targets):
            _require(view.identity.model_id == self.layout.model_id and bool(view.allocation_id),
                     "Wrong model or absent allocation identity")
            _require(type(view.capacity_tokens) is int and type(view.valid_tokens) is int
                     and 0 <= view.valid_tokens <= view.capacity_tokens
                     and 0 < target <= min(view.capacity_tokens, 2048)
                     and view.block_size == 16, "Invalid view capacity, length or block size")
            _require(set(view.block_tables) == set(self.layout.ranks), "Incomplete rank mapping")
            if mode == "DECODE":
                _require(target == view.valid_tokens + 1, "Decode must append exactly one token")
                starts.append(view.valid_tokens)
                writes.append(1)
            else:
                _require(mode != "PREFILL" or view.valid_tokens == 0,
                         "Initial prefill requires empty committed KV")
                starts.append(0)
                writes.append(target)
            pages = (view.capacity_tokens + 15) // 16
            for rank, rank_table in view.block_tables.items():
                if isinstance(rank_table, PackedBlocks):
                    _require(rank_table.layers == self.layer_num, "Incomplete layer mapping")
                    _require(rank_table.heads == self.head_num, "Incomplete head mapping")
                    _require(rank_table.pages == pages, "Incomplete physical page mapping")
                    rank_blocks = rank_table.array().reshape(-1)
                else:
                    _require(len(rank_table) == self.layer_num, "Incomplete layer mapping")
                    flat = []
                    for heads in rank_table:
                        _require(len(heads) == self.head_num, "Incomplete head mapping")
                        for blocks in heads:
                            _require(len(blocks) == pages, "Incomplete physical page mapping")
                            flat.extend(blocks)
                    _require(set(map(type, flat)) == {int}, "Physical block index outside imported pool")
                    rank_blocks = np.asarray(flat, dtype=np.int64)
                # Every block on every rank is still validated, with numpy
                # reductions instead of per-block Python dispatch.
                _require(rank_blocks.size > 0 and int(rank_blocks.min()) >= 0
                         and int(rank_blocks.max()) < self.physical_block_count,
                         "Physical block index outside imported pool")
                used_blocks[rank].append(rank_blocks)
            own = view.block_tables[self.rank]
            tables.append(own.array().astype(np.int64) if isinstance(own, PackedBlocks) else
                          tuple(tuple(tuple(blocks) for blocks in heads) for heads in own))
        for rank, arrays in used_blocks.items():
            combined = np.concatenate(arrays) if len(arrays) > 1 else arrays[0]
            _require(np.unique(combined).size == combined.size,
                     "Aliased physical blocks across requests/layers/heads")
        self._bound = BoundForward(mode, rows, targets, tuple(writes), tuple(starts), tuple(tables),
                                   capacities=tuple(view.capacity_tokens for view in views))

    def advance_decode(self, req_pool_indices: Sequence[int], target_lengths: Sequence[int]) -> None:
        """Re-bind the next step of a granted decode run without re-reading views.

        Only an executed DECODE binding may advance, by exactly one token per
        request, inside the capacity reserved and fully validated at bind time.
        """
        bound = self.bound_forward
        _require(bound.mode == "DECODE" and bound.batch is not None,
                 "Only an executed decode binding can advance", "FORWARD_MISMATCH")
        rows = _cpu_ints(req_pool_indices, "bound request rows")
        targets = _cpu_ints(target_lengths, "bound target lengths")
        _require(rows == bound.req_pool_indices, "Decode run rows changed", "FORWARD_MISMATCH")
        _require(len(targets) == len(bound.target_lengths) and all(
                     new == old + 1 and new <= min(capacity, 2048)
                     for new, old, capacity in zip(targets, bound.target_lengths, bound.capacities)),
                 "Decode run must append one token inside its reserved capacity", "FORWARD_MISMATCH")
        self._bound = BoundForward("DECODE", rows, targets, (1,) * len(rows), bound.target_lengths,
                                   bound.tables, capacities=bound.capacities,
                                   np_tables=bound.np_tables, table_buffer=bound.table_buffer)

    def clear_forward(self) -> None:
        """Caller must hold the grant until all-rank GPU completion before clearing."""
        self._bound = None

    def validate_forward(self, forward_batch) -> BoundForward:
        bound = self.bound_forward
        _require(getattr(forward_batch.forward_mode, "name", None)
                 == ("DECODE" if bound.mode == "DECODE" else "EXTEND"),
                 "Forward mode does not match granted mode", "FORWARD_MISMATCH")
        _require(forward_batch.batch_size == len(bound.req_pool_indices),
                 "Forward batch size mismatch", "FORWARD_MISMATCH")
        _require(_cpu_ints(forward_batch.req_pool_indices_cpu, "req_pool_indices_cpu")
                 == bound.req_pool_indices, "Request row order mismatch", "FORWARD_MISMATCH")
        _require(_cpu_ints(forward_batch.seq_lens_cpu, "seq_lens_cpu") == bound.target_lengths,
                 "Target lengths mismatch", "FORWARD_MISMATCH")
        _require(getattr(forward_batch, "spec_info", None) is None,
                 "Speculative attention is unsupported")
        spec = getattr(forward_batch, "spec_algorithm", None)
        _require(spec is None or (hasattr(spec, "is_none") and spec.is_none()),
                 "Speculative decoding is unsupported")
        for name in ("encoder_lens", "encoder_lens_cpu", "encoder_out_cache_loc"):
            _require(getattr(forward_batch, name, None) is None, "Encoder attention is unsupported")
        if bound.mode != "DECODE":
            _require(_cpu_ints(forward_batch.extend_seq_lens_cpu, "extend_seq_lens_cpu")
                     == bound.write_lengths, "Extend lengths mismatch", "FORWARD_MISMATCH")
            _require(_cpu_ints(forward_batch.extend_prefix_lens_cpu, "extend_prefix_lens_cpu")
                     == (0,) * len(bound.req_pool_indices),
                     "Prefix/chunked prefill is unsupported", "FORWARD_MISMATCH")
            _require(forward_batch.extend_num_tokens == bound.total_written,
                     "Extend token count mismatch", "FORWARD_MISMATCH")
        _require(forward_batch.input_ids.numel() == bound.total_written
                 and forward_batch.out_cache_loc.numel() == bound.total_written,
                 "Input/write geometry mismatch", "FORWARD_MISMATCH")
        _require(bound.batch is None or bound.batch is forward_batch,
                 "A grant cannot be reused for a different ForwardBatch", "FORWARD_MISMATCH")
        bound.batch = forward_batch
        if bound.context_lens is None:
            bound.context_lens = torch.tensor(bound.target_lengths, dtype=torch.int32, device=self.device)
        return bound

    def layer_metadata(self, layer_id: int) -> tuple[torch.Tensor, torch.Tensor]:
        self._check_layer(layer_id)
        bound = self.bound_forward
        _require(bound.batch is not None, "ForwardBatch must be validated before layer use", "UNBOUND_FORWARD")
        if bound.metadata_buffers is None:
            # All layers retained this metadata until clear_forward already.
            # At most two uploads per forward; a leading layer dimension gives
            # contiguous kernel views. Index math runs in numpy, not per block.
            selected = slice(self.start_layer, self.end_layer + 1)
            layers = self.end_layer - self.start_layer + 1
            if bound.np_tables is None:
                bound.np_tables = [np.asarray(table, dtype=np.int64) for table in bound.tables]
            if bound.table_buffer is None:
                # Pad every request to the widest reserved mapping. The kernel
                # reads only ceil(context/16) pages; unused entries are zero or
                # blocks this request already owns.
                max_pages = max(table.shape[2] for table in bound.np_tables)
                padded = np.zeros((layers, len(bound.np_tables), self.head_num, max_pages), dtype=np.int32)
                for row, table in enumerate(bound.np_tables):
                    padded[:, row, :, :table.shape[2]] = table[selected]
                bound.table_buffer = torch.tensor(padded, dtype=torch.int32, device=self.device)
            slots = np.empty((layers, bound.total_written, self.head_num), dtype=np.int64)
            offset = 0
            for row, (start, count) in enumerate(zip(bound.write_starts, bound.write_lengths)):
                positions = np.arange(start, start + count, dtype=np.int64)
                pages = bound.np_tables[row][selected][:, :, positions // 16]
                slots[:, offset:offset + count, :] = (pages * 16 + positions % 16).transpose(0, 2, 1)
                offset += count
            bound.metadata_buffers = (
                torch.tensor(slots, dtype=torch.int64, device=self.device),
                bound.table_buffer,
            )
        if layer_id not in bound.layer_metadata:
            offset = layer_id - self.start_layer
            slots, tables = (buffer[offset] for buffer in bound.metadata_buffers)
            _require(slots.is_contiguous() and tables.is_contiguous(),
                     "Layer metadata must be contiguous")
            bound.layer_metadata[layer_id] = slots, tables
        return bound.layer_metadata[layer_id]

    def _check_layer(self, layer_id: int) -> None:
        _require(type(layer_id) is int and self.start_layer <= layer_id <= self.end_layer,
                 "Layer outside model layout")

    def get_key_buffer(self, layer_id: int) -> torch.Tensor:
        self._check_layer(layer_id)
        return self.pool_key

    def get_value_buffer(self, layer_id: int) -> torch.Tensor:
        self._check_layer(layer_id)
        return self.pool_value

    def get_kv_buffer(self, layer_id: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.get_key_buffer(layer_id), self.get_value_buffer(layer_id)

    def get_v_head_dim(self) -> int:
        return self.v_head_dim

    def get_kv_size_bytes(self) -> tuple[int, int]:
        return self.pool_key.nbytes, self.pool_value.nbytes

    def get_contiguous_buf_infos(self):
        tensors = self.kv_buffer
        return ([t.data_ptr() for t in tensors], [t.nbytes for t in tensors],
                [t[0].nbytes for t in tensors])

    def set_kv_buffer(self, layer, loc, cache_k: torch.Tensor, cache_v: torch.Tensor,
                      k_scale=None, v_scale=None, layer_id_override=None, dcp_kv_mask=None) -> None:
        _require(k_scale is None and v_scale is None and dcp_kv_mask is None,
                 "Quantized/scaled/masked writes are unsupported")
        _require(layer_id_override is None, "Layer remapping is unsupported")
        bound = self.bound_forward
        _require(bound.batch is not None and loc is bound.batch.out_cache_loc,
                 "Write locations must be the validated logical forward locations", "FORWARD_MISMATCH")
        expected = (bound.total_written, self.head_num, self.head_dim)
        _require(tuple(cache_k.shape) == tuple(cache_v.shape) == expected,
                 "K/V writes must be [written tokens,local heads,128]")
        _require(cache_k.dtype == cache_v.dtype == self.dtype
                 and cache_k.device == cache_v.device == self.device,
                 "K/V write dtype or device mismatch")
        slots, _ = self.layer_metadata(layer.layer_id)
        ops.store(cache_k.contiguous(), cache_v.contiguous(), self.pool_key, self.pool_value, slots)

    def get_cpu_copy(self, *args, **kwargs):
        raise IntegrationError("UNSUPPORTED", "Controller PD transport owns physical KV movement")

    def load_cpu_copy(self, *args, **kwargs):
        raise IntegrationError("UNSUPPORTED", "Native CPU offload is unsupported for shared Head-wise storage")

    def move_kv_cache(self, *args, **kwargs):
        raise IntegrationError("UNSUPPORTED", "Native cache mutation is unsupported for shared Head-wise storage")

    def delete_buffers(self):
        raise IntegrationError("UNSUPPORTED", "Only the authoritative storage owner may release shared buffers")
