"""Thread-safe logical ownership of per-rank Head-wise KV blocks.

Completion reports are trusted statements from the rank controller: callers must
only report a rank after its CUDA work has completed. A fence never frees memory.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
import pickle
from threading import RLock
from typing import Any, Sequence
from uuid import uuid4

import numpy as np

from muxserve.contracts import (IntegrationError, KVTarget, KVView, ModelLayout, PackedBlocks, RequestIdentity,
                        Reservation)


def _copy_metadata(value: Any) -> Any:
    """Detach owner-generated CPU descriptors with the existing wire value types."""
    return pickle.loads(pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL))


class _FreeBlocks:
    """One pool's free block IDs as a stack: take/put move whole arrays.

    Blocks are handed out lowest ID first from a fresh pool, then most
    recently freed first. Only the store holds it, under the store lock.
    """
    def __init__(self, blocks):
        self.data = np.array(sorted(blocks), dtype=np.int32)[::-1].copy()
        self.top = len(self.data)

    def __len__(self) -> int:
        return self.top

    def take(self, count: int) -> np.ndarray:
        if not 0 <= count <= self.top:
            raise IntegrationError("INVARIANT_VIOLATION", "Free block stack underflow")
        chosen = self.data[self.top - count:self.top][::-1].copy()
        self.top -= count
        return chosen

    def put(self, blocks: np.ndarray) -> None:
        count = len(blocks)
        if self.top + count > len(self.data):
            raise IntegrationError("INVARIANT_VIOLATION", "Free block stack overflow (double free)")
        self.data[self.top:self.top + count] = blocks[::-1]
        self.top += count

    def as_set(self) -> set[int]:
        return set(self.data[:self.top].tolist())


@dataclass
class _Allocation:
    allocation_id: str
    identity: RequestIdentity
    layout: ModelLayout
    # Per rank int32[layer, head, page]; pages [0, capacity // block_size)
    # are mapped, the rest is unused room up to the context limit.
    arrays: dict[int, np.ndarray]
    capacity: int = 0
    valid: int = 0
    version: int = 0
    holders: set[str] = field(default_factory=set)
    active: str | None = None
    # Bumped on every physical mapping change; keys the packed view cache.
    epoch: int = 0
    packed: tuple | None = None

    @property
    def pages(self) -> int:
        return self.capacity // self.layout.block_size

    def mapped(self, rank: int) -> np.ndarray:
        return self.arrays[rank][:, :, :self.pages]

    @property
    def tables(self) -> dict[int, list[list[list[int]]]]:
        """Nested-list copy of the mapping (snapshots, checks, diagnostics)."""
        return {rank: self.mapped(rank).tolist() for rank in self.arrays}


@dataclass
class _Batch:
    reservation: Reservation
    signature: tuple
    holder: str
    old_capacity: dict[str, int]
    new_ids: set[str]
    ranks: set[int]
    introduced_holders: set[str]
    state: str = "PREPARING"
    completed: set[int] = field(default_factory=set)
    success_ranks: set[int] = field(default_factory=set)
    success_lengths: dict[str, int] | None = None
    failure_seen: bool = False
    accepted_reports: set[tuple] = field(default_factory=set)
    finish_signature: tuple | None = None
    result: tuple[KVView, ...] = ()


class HeadwiseStore:
    HISTORY_LIMIT = 100_000

    def __init__(self, pool_blocks_by_rank: dict[int, int], max_context_tokens: int = 2048,
                 *, pool_mode: str = "shared", model_ids: Sequence[str] = (),
                 batch_history_limit: int = 100_000):
        if (not pool_blocks_by_rank or any(type(r) is not int or r < 0
                or type(n) is not int or n <= 0 for r, n in pool_blocks_by_rank.items())
                or type(max_context_tokens) is not int or max_context_tokens <= 0
                or not isinstance(pool_mode, str) or pool_mode not in ("shared", "static")
                or type(batch_history_limit) is not int or not 1 <= batch_history_limit <= 4_000_000):
            raise IntegrationError("INVALID_ARGUMENT", "Positive rank pools and context limit required")
        try:
            model_ids = tuple(model_ids)
        except TypeError as exc:
            raise IntegrationError("INVALID_ARGUMENT", "Model IDs must be a sequence") from exc
        if pool_mode == "static":
            if (not model_ids or any(not isinstance(mid, str) or not mid for mid in model_ids)
                    or len(set(model_ids)) != len(model_ids)
                    or any(n < len(model_ids) for n in pool_blocks_by_rank.values())):
                raise IntegrationError("INVALID_ARGUMENT", "Static pool requires unique model IDs and at least one block per model")
        self._lock = RLock()
        self._pool = dict(pool_blocks_by_rank)
        # Shared mode: one free stack per rank. Static mode: one per model and
        # rank (self._free stays empty); a rank's free count is their sum.
        self._free = ({r: _FreeBlocks(range(n)) for r, n in self._pool.items()}
                      if pool_mode == "shared" else {})
        self._pool_mode = pool_mode
        self._batch_history_limit = batch_history_limit
        self._model_ids = model_ids if pool_mode == "static" else ()
        self._model_ranges: dict[str, dict[int, tuple[int, ...]]] = {}
        self._model_free: dict[str, dict[int, _FreeBlocks]] = {}
        if pool_mode == "static":
            for model_index, model_id in enumerate(model_ids):
                self._model_ranges[model_id] = {}
                self._model_free[model_id] = {}
                for rank, total in self._pool.items():
                    base, remainder = divmod(total, len(model_ids))
                    count = base + (1 if model_index < remainder else 0)
                    start = model_index * base + min(model_index, remainder)
                    blocks = tuple(range(start, start + count))
                    self._model_ranges[model_id][rank] = blocks
                    self._model_free[model_id][rank] = _FreeBlocks(blocks)
        self._models: dict[str, ModelLayout] = {}
        self._allocations: dict[str, _Allocation] = {}
        self._identities: dict[RequestIdentity, str] = {}
        self._latest_attempt: dict[tuple[str, str], int] = {}
        self._retired_ids: set[str] = set()
        self._batches: dict[str, _Batch] = {}
        self._batch_ids: dict[str, str] = {}
        self._retired_batch_ids: set[str] = set()
        self._max_context = max_context_tokens
        self._closing = False

    def register_model(self, layout: ModelLayout) -> None:
        with self._lock:
            self._ensure_open()
            if not isinstance(layout, ModelLayout):
                raise IntegrationError("INVALID_ARGUMENT", "ModelLayout required")
            if any(r not in self._pool for r in layout.ranks):
                raise IntegrationError("UNKNOWN_RANK", "Model ranks must have assigned pools")
            if self._pool_mode == "static" and layout.model_id not in self._model_ranges:
                raise IntegrationError("UNKNOWN_MODEL", "Model has no static KV partition")
            old = self._models.get(layout.model_id)
            if old is not None and old != layout:
                raise IntegrationError("IDENTITY_CONFLICT", "Model layout already registered differently")
            self._models[layout.model_id] = layout

    def reserve_kv(self, batch_id: str, targets: Sequence[KVTarget], holder: str) -> Reservation:
        targets = tuple(targets)
        if not batch_id or not isinstance(batch_id, str) or not targets:
            raise IntegrationError("INVALID_ARGUMENT", "Nonempty batch and targets required")
        self._validate_holder(holder)
        if any(not isinstance(t, KVTarget) for t in targets):
            raise IntegrationError("INVALID_ARGUMENT", "KVTarget entries required")
        signature = (holder, tuple((t.identity, t.target_tokens, t.allocation_id) for t in targets))
        with self._lock:
            existing = self._batch_ids.get(batch_id)
            if existing is not None:
                batch = self._batches[existing]
                if batch.signature != signature:
                    raise IntegrationError("IDENTITY_CONFLICT", "Batch replay has different content")
                # A terminal replay returns its original descriptor, never new blocks.
                return _copy_metadata(batch.reservation)
            if batch_id in self._retired_batch_ids:
                raise IntegrationError("RETIRED_BATCH", "Batch replay is outside retained history")
            if len(self._batch_ids) + len(self._retired_batch_ids) >= self._batch_history_limit:
                raise IntegrationError("HISTORY_FULL", "Store batch history requires safe owner restart")
            self._ensure_open()
            seen: set[RequestIdentity] = set()
            seen_requests: set[tuple[str, str]] = set()
            proposed: list[tuple[_Allocation, int, bool]] = []
            needs = {r: 0 for r in self._pool}
            needs_by_model = {mid: {r: 0 for r in self._pool} for mid in self._model_ids}
            new_request_keys: set[tuple[str, str]] = set()
            new_allocation_count = 0
            for target in targets:
                identity = target.identity
                if not isinstance(identity, RequestIdentity):
                    raise IntegrationError("INVALID_ARGUMENT", "RequestIdentity required")
                key = (identity.model_id, identity.request_id)
                if identity in seen or key in seen_requests:
                    raise IntegrationError("IDENTITY_CONFLICT", "Duplicate request in batch")
                seen.add(identity)
                seen_requests.add(key)
                layout = self._models.get(identity.model_id)
                if layout is None:
                    raise IntegrationError("UNKNOWN_MODEL", "Model is not registered")
                tokens = target.target_tokens
                if type(tokens) is not int or tokens <= 0 or tokens > self._max_context:
                    raise IntegrationError("INVALID_ARGUMENT", "Target exceeds context or is not positive")
                if target.allocation_id is not None and (not isinstance(target.allocation_id, str)
                                                         or not target.allocation_id):
                    raise IntegrationError("INVALID_ARGUMENT", "Allocation ID must be a nonempty string")
                latest = self._latest_attempt.get(key, -1)
                if identity.attempt < latest:
                    raise IntegrationError("STALE_ATTEMPT", "Request attempt has been superseded")
                aid = target.allocation_id or self._identities.get(identity)
                allocation = self._allocations.get(aid) if aid else None
                if target.allocation_id is not None and allocation is None:
                    raise IntegrationError("UNKNOWN_ALLOCATION", "Allocation is missing or retired")
                if allocation is not None:
                    if allocation.identity != identity:
                        raise IntegrationError("IDENTITY_CONFLICT", "Allocation belongs to another identity")
                    if allocation.active is not None:
                        raise IntegrationError("ALLOCATION_BUSY", "Allocation has an active or fenced batch")
                    if tokens < allocation.valid:
                        raise IntegrationError("INVALID_ARGUMENT", "Target is below committed valid length")
                else:
                    # Retrying the same retired attempt must use a new attempt number.
                    if latest == identity.attempt and identity not in self._identities:
                        raise IntegrationError("STALE_ATTEMPT", "Retired request attempt cannot be reused")
                    max_pages = (self._max_context + layout.block_size - 1) // layout.block_size
                    allocation = _Allocation(uuid4().hex, identity, layout,
                        {r: np.empty((layout.num_layers, layout.kv_heads_per_rank, max_pages), np.int32)
                         for r in layout.ranks})
                    new_allocation_count += 1
                if key not in self._latest_attempt:
                    new_request_keys.add(key)
                capacity = max(allocation.capacity, ((tokens + layout.block_size - 1)
                    // layout.block_size) * layout.block_size)
                groups = (capacity - allocation.capacity) // layout.block_size
                for rank in layout.ranks:
                    requested = groups * layout.blocks_per_token_group
                    needs[rank] += requested
                    if self._pool_mode == "static":
                        needs_by_model[identity.model_id][rank] += requested
                proposed.append((allocation, capacity, allocation.allocation_id not in self._allocations))
            if (len(self._latest_attempt) + len(new_request_keys) > self.HISTORY_LIMIT
                    or len(self._allocations) + len(self._retired_ids) + new_allocation_count >
                    self.HISTORY_LIMIT * 2):
                raise IntegrationError("HISTORY_FULL", "Store identity history requires safe owner restart")
            if any(needs[r] > self._free_count(r) for r in needs):
                raise IntegrationError("OUT_OF_CAPACITY", "Atomic all-rank KV reservation failed")
            if self._pool_mode == "static" and any(
                    needs_by_model[mid][rank] > len(self._model_free[mid][rank])
                    for mid in self._model_ids for rank in self._pool):
                raise IntegrationError("OUT_OF_CAPACITY", "Static model KV partition is exhausted")
            rid = uuid4().hex
            old_capacity: dict[str, int] = {}
            new_ids: set[str] = set()
            ranks: set[int] = set()
            introduced_holders: set[str] = set()
            for allocation, capacity, is_new in proposed:
                aid = allocation.allocation_id
                old_capacity[aid] = allocation.capacity
                if is_new:
                    new_ids.add(aid)
                    self._allocations[aid] = allocation
                    self._identities[allocation.identity] = aid
                groups = (capacity - allocation.capacity) // allocation.layout.block_size
                first = allocation.pages
                layers, heads = allocation.layout.num_layers, allocation.layout.kv_heads_per_rank
                for rank, table in allocation.arrays.items():
                    ranks.add(rank)
                    if not groups:
                        continue
                    # Each (layer, head) gets the next ``groups`` blocks, in
                    # layer-major order, appended after its mapped pages.
                    chosen = self._free_source(allocation, rank).take(groups * layers * heads)
                    table[:, :, first:first + groups] = chosen.reshape(layers, heads, groups)
                if groups:
                    allocation.epoch += 1
                allocation.capacity = capacity
                allocation.active = rid
                if holder not in allocation.holders:
                    introduced_holders.add(aid)
                allocation.holders.add(holder)
                allocation.version += 1
                key = (allocation.identity.model_id, allocation.identity.request_id)
                self._latest_attempt[key] = allocation.identity.attempt
            reservation = Reservation(rid, batch_id, tuple(self._view(a) for a, _, _ in proposed))
            self._batches[rid] = _Batch(reservation, signature, holder, old_capacity, new_ids, ranks,
                                       introduced_holders)
            self._batch_ids[batch_id] = rid
            return _copy_metadata(reservation)

    def bind_kv(self, allocation_id: str, holder: str) -> KVView:
        self._validate_holder(holder)
        with self._lock:
            self._ensure_open()
            allocation = self._allocation(allocation_id)
            key = (allocation.identity.model_id, allocation.identity.request_id)
            if allocation.identity.attempt < self._latest_attempt.get(key, -1):
                raise IntegrationError("STALE_ATTEMPT", "Request attempt has been superseded")
            if (allocation.active is not None
                    and self._batch(allocation.active).state != "PREPARING"):
                raise IntegrationError("ALLOCATION_BUSY", "Bind requires an idle or preparing allocation")
            # A prepared bind only pins ownership; it grants no read access.
            # The view still reports the committed length (possibly zero).
            allocation.holders.add(holder)
            return self._view(allocation)

    def start_kv_batch(self, reservation_id: str) -> None:
        with self._lock:
            batch = self._batch(reservation_id)
            if batch.state == "RUNNING":
                return
            if batch.state != "PREPARING":
                raise IntegrationError("INVALID_STATE", "Only a prepared reservation can start")
            batch.state = "RUNNING"

    def extend_kv_batch(self, reservation_id: str, targets: Sequence[KVTarget]) -> tuple[KVView, ...]:
        """Grow a running reservation, all or nothing: a target whose allocation is
        already in it gets its mapping grown to cover the target; an idle allocation
        (another request joining a running decode run) is added to it first, exactly as
        reserve_kv would have added it. Finish, rollback and fencing then treat every
        added allocation like one reserved at the start. Returns the targets' views."""
        targets = tuple(targets)
        if not targets or any(not isinstance(t, KVTarget) or not isinstance(t.allocation_id, str)
                              for t in targets):
            raise IntegrationError("INVALID_ARGUMENT", "KVTargets of existing allocations required")
        with self._lock:
            self._ensure_open()
            batch = self._batch(reservation_id)
            if batch.state != "RUNNING":
                raise IntegrationError("INVALID_STATE", "Only a running reservation can grow")
            proposed: list[tuple[_Allocation, int, bool]] = []
            needs = {r: 0 for r in self._pool}
            needs_by_model = {mid: {r: 0 for r in self._pool} for mid in self._model_ids}
            for target in targets:
                allocation = self._allocation(target.allocation_id)
                if allocation.identity != target.identity or any(
                        a is allocation for a, _, _ in proposed):
                    raise IntegrationError("IDENTITY_CONFLICT", "Allocation identity differs or repeats")
                joining = allocation.allocation_id not in batch.old_capacity
                if joining:
                    key = (allocation.identity.model_id, allocation.identity.request_id)
                    if allocation.identity.attempt < self._latest_attempt.get(key, -1):
                        raise IntegrationError("STALE_ATTEMPT", "Request attempt has been superseded")
                    if allocation.active is not None:
                        raise IntegrationError("ALLOCATION_BUSY", "Allocation has an active or fenced batch")
                elif allocation.active != reservation_id:
                    raise IntegrationError("INVARIANT_VIOLATION", "Reservation allocation mismatch")
                tokens, layout = target.target_tokens, allocation.layout
                if (type(tokens) is not int or tokens <= 0 or tokens > self._max_context
                        or tokens < allocation.valid):
                    raise IntegrationError("INVALID_ARGUMENT", "Target exceeds context or committed length")
                capacity = max(allocation.capacity, ((tokens + layout.block_size - 1)
                    // layout.block_size) * layout.block_size)
                groups = (capacity - allocation.capacity) // layout.block_size
                for rank in layout.ranks:
                    needs[rank] += groups * layout.blocks_per_token_group
                    if self._pool_mode == "static":
                        needs_by_model[allocation.identity.model_id][rank] += groups * layout.blocks_per_token_group
                proposed.append((allocation, capacity, joining))
            if any(needs[r] > self._free_count(r) for r in needs):
                raise IntegrationError("OUT_OF_CAPACITY", "Atomic all-rank KV growth failed")
            if self._pool_mode == "static" and any(
                    needs_by_model[mid][rank] > len(self._model_free[mid][rank])
                    for mid in self._model_ids for rank in self._pool):
                raise IntegrationError("OUT_OF_CAPACITY", "Static model KV partition is exhausted")
            for allocation, capacity, joining in proposed:
                aid = allocation.allocation_id
                if joining:
                    batch.old_capacity[aid] = allocation.capacity
                    batch.ranks.update(allocation.arrays)
                    allocation.active = reservation_id
                    if batch.holder not in allocation.holders:
                        batch.introduced_holders.add(aid)
                    allocation.holders.add(batch.holder)
                groups = (capacity - allocation.capacity) // allocation.layout.block_size
                if groups:
                    first = allocation.pages
                    layers, heads = allocation.layout.num_layers, allocation.layout.kv_heads_per_rank
                    for rank, table in allocation.arrays.items():
                        chosen = self._free_source(allocation, rank).take(groups * layers * heads)
                        table[:, :, first:first + groups] = chosen.reshape(layers, heads, groups)
                    allocation.epoch += 1
                allocation.capacity = capacity
                allocation.version += 1
            return _copy_metadata(tuple(self._view(a) for a, _, _ in proposed))

    def detach_kv_batch(self, reservation_id: str, allocation_id: str, valid_tokens: int,
                        trim_to_valid: bool = False) -> KVView:
        """Commit one allocation of a running reservation on its own and take it out of the
        reservation (a request leaving a running decode run). The caller states that every
        rank's GPU work writing its first ``valid_tokens`` positions has completed and none
        writes it again in this reservation. The batch holder leaves it; its other holders
        stay. Finish, rollback and fencing of the rest of the reservation no longer see it."""
        with self._lock:
            batch = self._batch(reservation_id)
            allocation = self._allocation(allocation_id)
            if (batch.state != "RUNNING" or allocation_id not in batch.old_capacity
                    or allocation.active != reservation_id):
                raise IntegrationError("INVALID_STATE", "Only a member of a running reservation can leave")
            if (type(valid_tokens) is not int or valid_tokens < allocation.valid
                    or valid_tokens > allocation.capacity or valid_tokens > self._max_context):
                raise IntegrationError("INVALID_ARGUMENT", "Invalid committed KV length")
            if len(batch.old_capacity) == 1:
                raise IntegrationError("INVALID_STATE", "The last member finishes with the reservation")
            allocation.valid = valid_tokens
            if trim_to_valid:
                self._trim(allocation)
            allocation.version += 1
            allocation.active = None
            del batch.old_capacity[allocation_id]
            batch.introduced_holders.discard(allocation_id)
            allocation.holders.discard(batch.holder)
            view = self._view(allocation)
            self._collect(allocation)
            return _copy_metadata(view)

    def finish_kv_batch(self, reservation_id: str, outcome: str,
                        valid_lengths: dict[str, int], completed_ranks: Sequence[int],
                        trim_to_valid: bool = False) -> tuple[KVView, ...]:
        """trim_to_valid: on success, return every page past the committed length to the
        free pool (a decode run reserves all of its steps up front and may stop early).
        The next reservation appends pages again; no rank keeps a mapping between batches."""
        if not isinstance(outcome, str):
            raise IntegrationError("INVALID_ARGUMENT", "Outcome must be a string")
        outcome = outcome.upper()
        if outcome not in {"SUCCESS", "FAILURE", "UNKNOWN"}:
            raise IntegrationError("INVALID_ARGUMENT", "Outcome must be SUCCESS, FAILURE or UNKNOWN")
        if not isinstance(valid_lengths, dict):
            raise IntegrationError("INVALID_ARGUMENT", "Valid lengths must be an allocation mapping")
        ranks = set(completed_ranks)
        signature = (outcome, tuple(sorted(valid_lengths.items())), tuple(sorted(ranks)))
        with self._lock:
            batch = self._batch(reservation_id)
            if batch.finish_signature is not None:
                if signature not in batch.accepted_reports:
                    raise IntegrationError("IDENTITY_CONFLICT", "Finish replay has different content")
                return _copy_metadata(batch.result)
            if batch.state not in {"RUNNING", "FENCED"}:
                raise IntegrationError("INVALID_STATE", "Only running or fenced batches can finish")
            if any(type(r) is not int for r in ranks) or not ranks <= batch.ranks:
                raise IntegrationError("INVALID_ARGUMENT", "Completion includes unassigned ranks")
            if outcome == "SUCCESS" and not ranks:
                raise IntegrationError("INVALID_ARGUMENT", "Success needs explicit rank evidence")
            ids = set(batch.old_capacity)
            if outcome == "SUCCESS":
                if set(valid_lengths) != ids:
                    raise IntegrationError("INVALID_ARGUMENT", "Success requires every allocation valid length")
                for aid, valid in valid_lengths.items():
                    allocation = self._allocation(aid)
                    if (type(valid) is not int or valid < allocation.valid
                            or valid > allocation.capacity or valid > self._max_context):
                        raise IntegrationError("INVALID_ARGUMENT", "Invalid committed KV length")
            elif valid_lengths:
                raise IntegrationError("INVALID_ARGUMENT", "Failure/unknown must not commit valid lengths")
            # Validate against retained evidence before changing any completion
            # or outcome state. A conflicting success report cannot be reconciled
            # by whichever rank happens to report last.
            if (outcome == "SUCCESS" and batch.success_lengths is not None
                    and batch.success_lengths != valid_lengths):
                raise IntegrationError("IDENTITY_CONFLICT", "Successful valid lengths disagree")
            batch.completed.update(ranks)
            if outcome == "FAILURE":
                batch.failure_seen = True
            elif outcome == "SUCCESS":
                if batch.success_lengths is None:
                    batch.success_lengths = dict(valid_lengths)
                batch.success_ranks.update(ranks)
            batch.accepted_reports.add(signature)
            # UNKNOWN may establish GPU completion, but only explicit compatible
            # reports for every rank can establish a successful outcome. An
            # explicit failure is sticky and wins once all GPU work has drained.
            if (batch.completed != batch.ranks or
                    not batch.failure_seen and batch.success_ranks != batch.ranks):
                batch.state = "FENCED"
                raise IntegrationError("BATCH_FENCED", "GPU completion or outcome remains uncertain")
            if batch.failure_seen:
                self._rollback(batch)
                batch.state = "FAILED"
            else:
                result = []
                for aid in batch.old_capacity:
                    allocation = self._allocation(aid)
                    allocation.valid = batch.success_lengths[aid]
                    if trim_to_valid:
                        self._trim(allocation)
                    allocation.version += 1
                    allocation.active = None
                    result.append(self._view(allocation))
                    self._collect(allocation)
                batch.result = tuple(result)
                batch.state = "FINISHED"
            batch.finish_signature = signature
            return _copy_metadata(batch.result)

    def abort_kv_batch(self, reservation_id: str) -> None:
        with self._lock:
            batch = self._batch(reservation_id)
            if batch.state in {"ABORTED", "FAILED"}:
                return
            if batch.state == "PREPARING":
                self._rollback(batch)
                batch.state = "ABORTED"
            elif batch.state in {"RUNNING", "FENCED"}:
                batch.state = "FENCED"
                raise IntegrationError("BATCH_FENCED", "Running batch requires all-rank completion proof")
            else:
                raise IntegrationError("INVALID_STATE", "Completed reservation cannot be aborted")

    def release_kv(self, allocation_id: str, holder: str) -> None:
        self._validate_holder(holder)
        with self._lock:
            allocation = self._allocations.get(allocation_id)
            if allocation is None:
                if allocation_id in self._retired_ids:
                    return
                raise IntegrationError("UNKNOWN_ALLOCATION", "Allocation is unknown")
            allocation.holders.discard(holder)
            self._collect(allocation)

    def fence_batch(self, reservation_id: str) -> None:
        with self._lock:
            batch = self._batch(reservation_id)
            if batch.state in {"PREPARING", "RUNNING", "FENCED"}:
                batch.state = "FENCED"

    def retire_history(self, batch_ids: Sequence[str]) -> dict:
        """Drop heavy terminal reservation views after all allocations are gone.

        Exact batch ID tombstones remain until the owner is safely restarted.
        Replaying a retired batch is rejected, never reconstructed.
        """
        ids = tuple(batch_ids)
        if not ids or any(not isinstance(bid, str) or not bid for bid in ids) or len(set(ids)) != len(ids):
            raise IntegrationError("INVALID_ARGUMENT", "Unique nonempty batch IDs required")
        with self._lock:
            selected = []
            for batch_id in ids:
                if batch_id in self._retired_batch_ids:
                    continue
                reservation_id = self._batch_ids.get(batch_id)
                if reservation_id is None:
                    raise IntegrationError("UNKNOWN_BATCH", "Cannot retire unknown batch")
                batch = self._batches[reservation_id]
                if (batch.state not in {"FINISHED", "FAILED", "ABORTED"}
                        or any(aid in self._allocations for aid in batch.old_capacity)):
                    raise IntegrationError("NOT_READY", "Batch allocation or completion still retained")
                selected.append((batch_id, reservation_id))
            for batch_id, reservation_id in selected:
                del self._batch_ids[batch_id]
                del self._batches[reservation_id]
                self._retired_batch_ids.add(batch_id)
            return {"retired": len(selected), "retired_total": len(self._retired_batch_ids)}

    def allocation_ids(self) -> frozenset[str]:
        """Copy only live ownership IDs; command polling does not need KV tables."""
        with self._lock:
            return frozenset(self._allocations)

    def allocation_identities(self) -> frozenset:
        """Live allocation identities only; never copies block tables."""
        with self._lock:
            return frozenset(a.identity for a in self._allocations.values())

    def history(self) -> dict:
        """The snapshot's bounded history counters without copying any tables."""
        with self._lock:
            return {"retained_batches": len(self._batches),
                    "retired_batch_ids": len(self._retired_batch_ids),
                    "request_keys": len(self._latest_attempt),
                    "retired_allocation_ids": len(self._retired_ids),
                    "lifetime_batch_limit": self._batch_history_limit}

    def rank_usage(self) -> dict:
        """Return capacity counters without materializing allocation mappings."""
        with self._lock:
            return {rank: {"total_blocks": total, "free_blocks": self._free_count(rank),
                           "used_blocks": total - self._free_count(rank)}
                    for rank, total in self._pool.items()}

    def snapshot(self) -> dict:
        with self._lock:
            return {
                "closing": self._closing,
                "pool_mode": self._pool_mode,
                "ranks": {r: {"total_blocks": n, "free_blocks": self._free_count(r),
                              "used_blocks": n - self._free_count(r)} for r, n in self._pool.items()},
                "model_partitions": {mid: {
                    "block_ranges_by_rank": {r: (blocks[0], blocks[-1] + 1)
                                              for r, blocks in ranks.items()},
                    "total_blocks_by_rank": {r: len(blocks) for r, blocks in ranks.items()},
                    "free_blocks_by_rank": {r: len(self._model_free[mid][r])
                                             for r in self._pool}}
                    for mid, ranks in self._model_ranges.items()},
                "allocations": {aid: {"identity": a.identity, "capacity_tokens": a.capacity,
                    "valid_tokens": a.valid, "holders": sorted(a.holders), "active_batch": a.active,
                    "version": a.version, "block_tables": a.tables,
                    "headwise_blocks_by_rank": {r: int(a.mapped(r).size) for r in a.arrays}}
                    for aid, a in self._allocations.items()},
                "reservations": {rid: {"batch_id": b.reservation.batch_id, "state": b.state,
                    "assigned_ranks": sorted(b.ranks), "completed_ranks": sorted(b.completed),
                    "success_ranks": sorted(b.success_ranks), "failure_seen": b.failure_seen,
                    "success_lengths": deepcopy(b.success_lengths)}
                    for rid, b in self._batches.items()},
                "fences": sorted(rid for rid, b in self._batches.items() if b.state == "FENCED"),
                "history": {"retained_batches": len(self._batches),
                    "retired_batch_ids": len(self._retired_batch_ids),
                    "request_keys": len(self._latest_attempt),
                    "retired_allocation_ids": len(self._retired_ids),
                    "lifetime_batch_limit": self._batch_history_limit},
            }

    def assert_integrity(self) -> dict:
        """Raise INVARIANT_VIOLATION on ownership/mapping corruption; no CUDA use."""
        with self._lock:
            owned: dict[int, set[int]] = {r: set() for r in self._pool}
            model_owned = {mid: {r: set() for r in self._pool} for mid in self._model_ids}
            def check(condition: bool, message: str) -> None:
                if not condition:
                    raise IntegrationError("INVARIANT_VIOLATION", message)
            for aid, a in self._allocations.items():
                check(self._identities.get(a.identity) == aid, "Identity index mismatch")
                check(0 <= a.valid <= min(a.capacity, self._max_context), "Invalid valid length")
                check(bool(a.holders) or a.active is not None, "Unprotected allocation")
                if a.active is not None:
                    batch = self._batches.get(a.active)
                    check(batch is not None and batch.state in {"PREPARING", "RUNNING", "FENCED"}
                          and aid in batch.old_capacity, "Batch access mismatch")
                check(set(a.arrays) == set(a.layout.ranks), "Rank mapping mismatch")
                groups = a.capacity // a.layout.block_size
                for r, table in a.arrays.items():
                    check(table.dtype == np.int32 and table.ndim == 3, "Block table type mismatch")
                    check(table.shape[0] == a.layout.num_layers, "Layer count mismatch")
                    check(table.shape[1] == a.layout.kv_heads_per_rank, "Head count mismatch")
                    check(table.shape[2] >= groups, "Ceil token mapping mismatch")
                    for block in a.mapped(r).reshape(-1).tolist():
                        check(0 <= block < self._pool[r] and block not in owned[r], "Duplicate/invalid block")
                        owned[r].add(block)
                        if self._pool_mode == "static":
                            check(block in self._model_ranges[a.identity.model_id][r],
                                  "Allocation escaped its static model partition")
                            model_owned[a.identity.model_id][r].add(block)
            for r, blocks in owned.items():
                free = self._free_set(r)
                check(len(free) == self._free_count(r), "Duplicate free block")
                check(not (blocks & free), "Live block marked free")
                check(blocks | free == set(range(self._pool[r])), "Pool accounting mismatch")
            if self._pool_mode == "static":
                for mid in self._model_ids:
                    for rank in self._pool:
                        partition = set(self._model_ranges[mid][rank])
                        free = self._model_free[mid][rank].as_set()
                        check(len(free) == len(self._model_free[mid][rank]), "Duplicate free block")
                        check(not (model_owned[mid][rank] & free), "Live model block marked free")
                        check(model_owned[mid][rank] | free == partition,
                              "Static model partition accounting mismatch")
            for rid, b in self._batches.items():
                if b.state in {"PREPARING", "RUNNING", "FENCED"}:
                    check(all(aid in self._allocations and self._allocations[aid].active == rid
                              for aid in b.old_capacity), "Reservation allocation mismatch")
            return {"ok": True, "allocations": len(self._allocations),
                    "used_blocks_by_rank": {r: len(bs) for r, bs in owned.items()}}

    def begin_teardown(self) -> dict:
        """Freeze new ownership before GPU owner teardown, atomically with store mutations."""
        with self._lock:
            self._closing = True
            return self.snapshot()

    def _ensure_open(self) -> None:
        if self._closing:
            raise IntegrationError("OWNER_CLOSING", "Storage owner is draining")

    @staticmethod
    def _validate_holder(holder: str) -> None:
        if not isinstance(holder, str) or not holder:
            raise IntegrationError("INVALID_ARGUMENT", "Nonempty holder required")

    def _allocation(self, allocation_id: str) -> _Allocation:
        if allocation_id not in self._allocations:
            raise IntegrationError("UNKNOWN_ALLOCATION", "Allocation is missing or retired")
        return self._allocations[allocation_id]

    def _batch(self, reservation_id: str) -> _Batch:
        if reservation_id not in self._batches:
            raise IntegrationError("UNKNOWN_RESERVATION", "Reservation is unknown")
        return self._batches[reservation_id]

    def _free_count(self, rank: int) -> int:
        if self._pool_mode == "static":
            return sum(len(self._model_free[mid][rank]) for mid in self._model_ids)
        return len(self._free[rank])

    def _free_set(self, rank: int) -> set[int]:
        if self._pool_mode == "static":
            return set().union(*(self._model_free[mid][rank].as_set() for mid in self._model_ids))
        return self._free[rank].as_set()

    def _free_source(self, allocation: _Allocation, rank: int) -> _FreeBlocks:
        if self._pool_mode == "static":
            return self._model_free[allocation.identity.model_id][rank]
        return self._free[rank]

    @staticmethod
    def _view(a: _Allocation) -> KVView:
        # Views carry immutable PackedBlocks: callers cannot alias or mutate
        # the store's mapping. The store's int32[layer, head, page] array is
        # PackedBlocks' byte layout, so packing is one copy of the mapped pages.
        if a.packed is None or a.packed[0] != a.epoch:
            pages = a.pages
            a.packed = (a.epoch, {rank: PackedBlocks(a.layout.num_layers, a.layout.kv_heads_per_rank,
                                                     pages, a.mapped(rank).tobytes())
                                  for rank in a.arrays})
        return KVView(a.allocation_id, a.identity, a.capacity, a.valid,
                      a.layout.block_size, dict(a.packed[1]), a.version)

    def _trim(self, allocation: _Allocation) -> None:
        size = allocation.layout.block_size
        keep = (allocation.valid + size - 1) // size
        if keep >= allocation.pages:
            return
        for rank, table in allocation.arrays.items():
            self._free_source(allocation, rank).put(table[:, :, keep:allocation.pages].reshape(-1))
        allocation.capacity = keep * size
        allocation.epoch += 1

    def _rollback(self, batch: _Batch) -> None:
        for aid, capacity in batch.old_capacity.items():
            allocation = self._allocation(aid)
            if allocation.active != batch.reservation.reservation_id:
                raise IntegrationError("INVARIANT_VIOLATION", "Stale reservation rollback")
            groups = capacity // allocation.layout.block_size
            for rank, table in allocation.arrays.items():
                released = table[:, :, groups:allocation.pages].reshape(-1)
                self._free_source(allocation, rank).put(released)
            allocation.epoch += 1
            allocation.capacity = capacity
            allocation.version += 1
            allocation.active = None
            if aid in batch.introduced_holders:
                allocation.holders.discard(batch.holder)
            self._collect(allocation)

    def _collect(self, allocation: _Allocation) -> None:
        if allocation.holders or allocation.active is not None:
            return
        for rank in allocation.arrays:
            self._free_source(allocation, rank).put(allocation.mapped(rank).reshape(-1))
        self._allocations.pop(allocation.allocation_id)
        self._identities.pop(allocation.identity)
        self._retired_ids.add(allocation.allocation_id)
