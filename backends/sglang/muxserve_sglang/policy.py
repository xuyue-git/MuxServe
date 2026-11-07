"""Source ADBS decisions; estimates never own KV. See controller-report.md."""
from __future__ import annotations
from collections import deque
from dataclasses import dataclass, field
import time
from muxserve.muxsched.resource import SMResource  # original MuxServe SM accounting
from . import original_muxserve_path
from .contracts import IntegrationError

original_muxserve_path()  # this checkout's muxserve, not another installed copy


@dataclass(frozen=True)
class ModelDemand:
    model_id: str
    prefill: tuple[str, ...]
    decode: tuple[str, ...]
    rebuild: tuple[str, ...] = ()
    oldest_wait_s: float = 0.0
    ordinals: dict[str, int] = field(default_factory=dict)
    prompt_lengths: dict[str, int] = field(default_factory=dict)


@dataclass(frozen=True)
class PolicySnapshot:
    demands: tuple[ModelDemand, ...]
    available_credits: int
    busy: frozenset[tuple[str, str]]
    blocked: frozenset[str]
    inflight_counts: dict[str, int]
    max_model_inflight: int
    rank_free_blocks: dict[int, int]
    credit_admissible: frozenset[tuple[str, str]] | None = None
    pause_prefill: bool | frozenset[str] = False


@dataclass(frozen=True)
class BatchSuggestion:
    model_id: str
    stage: str
    request_ids: tuple[str, ...]
    purpose: str = "GENERATE"


@dataclass(frozen=True)
class BatchFeedback:
    model_id: str
    stage: str
    admitted: int
    requested_limit: int
    duration_s: float
    backlog: int
    succeeded: bool
    input_tokens: int = 0
    output_tokens: int = 0
    purpose: str = "GENERATE"
    pd_handoff_ms: float | None = None
    prepare_handshake_ms: float | None = None
    gpu_forward_ms: float | None = None
    reserved_blocks_by_rank: dict[int, int] = field(default_factory=dict)
    released_blocks_by_rank: dict[int, int] = field(default_factory=dict)
    failure_reason: str | None = None

    @property
    def duration_ms(self) -> float:
        """Legacy policy duration: synchronized execution wall time, not GPU active time."""
        return self.duration_s * 1000


def initial_limits(models, blocks_per_group, total_blocks, rates=None,
                   expected_lengths=None, block_size=16):
    """Source init_max_num_seqs: explicit hints, equal-rate/default416 online."""
    rates = rates or {m: 1.0 for m in models}
    lengths = {m: (expected_lengths or {}).get(m, 416) for m in models}
    if (set(rates) != set(models) or any(not 0 <= r < float('inf') for r in rates.values())
            or any(type(n) is not int or n <= 0 for n in lengths.values())
            or any(blocks_per_group[m] <= 0 for m in models)):
        raise ValueError('Invalid source-policy initialization hints')
    base = sum(rates[m] * blocks_per_group[m] for m in models)
    if base <= 0:
        raise ValueError('At least one positive arrival-rate hint required')
    limits = {}
    for m in models:
        local = int(rates[m] * blocks_per_group[m] / base * total_blocks)
        limits[m] = local // blocks_per_group[m] // max(1, lengths[m] // block_size)
    return limits, lengths


def redistribute(limits, histories, lengths, blocks_per_group, waiting, running,
                 fragment, block_size=16):
    """scheduler.py:348-484, including strict > and final fractional carry."""
    costs = {m: max(1, lengths[m] // block_size) * blocks_per_group[m] for m in limits}
    averages = {m: sum(histories[m][-500:]) / len(histories[m][-500:])
                if histories.get(m) else 30.0 for m in limits}
    blocks = {m: costs[m] * limits[m] for m in limits}
    high, low = [], []
    for m, limit in limits.items():
        utilization = averages[m] / limit if limit else float('inf')
        if utilization >= 0.9:
            high.append((m, utilization))
        else:
            low.append((m, int(max(1, (0.95 - utilization) * limit)) * costs[m]))
    high.sort(key=lambda item: item[1], reverse=True)
    total_empty = sum(n for _, n in low)
    stolen = {m: 0 for m, _ in low}
    for m, _ in high:
        backlog, avg = waiting[m] + running[m], averages[m]
        if backlog < avg * 1.1:
            continue
        desired = min(max(1, int(0.1 * avg)), int(backlog - avg)) * costs[m]
        for src, empty in low:
            steal = min(int(desired * empty / total_empty), empty - stolen[src])
            stolen[src] += steal
            blocks[m] += steal
    result = {}
    for m in [m for m, _ in low] + [m for m, _ in high]:
        remain = blocks[m] - stolen.get(m, 0)
        count, remainder = divmod(remain, costs[m])
        if remainder + fragment > costs[m]:
            count += 1
            fragment -= costs[m] - remainder
        else:
            fragment += remainder
        result[m] = count
    for m in sorted(limits, key=lambda m: waiting[m], reverse=True):
        extra = fragment // costs[m]
        result[m] += extra
        fragment -= extra * costs[m]
    return result, fragment


class ADBS:
    def __init__(self, decode_credits, *, blocks_per_group, total_blocks, rates=None,
                 expected_lengths=None, maximum_batch=2, max_batched_tokens=2048,
                 clock=time.monotonic):
        self.decode_credits = dict(decode_credits)
        self.models = tuple(decode_credits)
        self.blocks_per_group = dict(blocks_per_group)
        self.limits, self.lengths = initial_limits(self.models, self.blocks_per_group,
                                                   total_blocks, rates, expected_lengths)
        self.maximum_batch, self.max_batched_tokens = maximum_batch, max_batched_tokens
        self.histories = {m: deque(maxlen=500) for m in self.models}
        self.feedback_history, self.adjustments = deque(maxlen=500), deque(maxlen=500)
        self.feedback_totals = {m: {stage: {purpose: {
            "batches": 0, "successes": 0, "failures": 0, "execution_wall_ms": 0.0,
            "gpu_forward_ms": 0.0, "gpu_forward_samples": 0,
            "prepare_handshake_ms": 0.0,
            "pd_handoff_ms": 0.0, "admitted": 0, "input_tokens": 0, "output_tokens": 0}
            for purpose in ("GENERATE", "REBUILD")}
            for stage in ("P", "D")} for m in self.models}
        self.fragment, self.clock, self.last_adjustment = 0, clock, clock()
        self.cursors = {'P': 0, 'D': 0}

    def feedback(self, item: BatchFeedback) -> None:
        self.feedback_history.append(item)
        totals = self.feedback_totals[item.model_id][item.stage][item.purpose]
        totals["batches"] += 1
        totals["successes" if item.succeeded else "failures"] += 1
        totals["execution_wall_ms"] += item.duration_ms
        if item.gpu_forward_ms is not None:
            totals["gpu_forward_ms"] += item.gpu_forward_ms
            totals["gpu_forward_samples"] += 1
        if item.prepare_handshake_ms is not None:
            totals["prepare_handshake_ms"] += item.prepare_handshake_ms
        if item.pd_handoff_ms is not None:
            totals["pd_handoff_ms"] += item.pd_handoff_ms
        totals["admitted"] += item.admitted
        totals["input_tokens"] += item.input_tokens
        totals["output_tokens"] += item.output_tokens
        # Source updates hist_num_seqs only on completed D batches (line1088).
        if item.stage == 'D' and item.succeeded:
            self.histories[item.model_id].append(item.admitted)

    def adjust(self, snapshot: PolicySnapshot, now=None) -> None:
        now = self.clock() if now is None else now
        if now - self.last_adjustment < 8.0:
            return
        waiting = {d.model_id: len(d.prefill) + len(d.rebuild) for d in snapshot.demands}
        running = {m: snapshot.inflight_counts.get(m, 0) for m in self.models}
        before = dict(self.limits)
        self.limits, self.fragment = redistribute(self.limits,
            {m: tuple(h) for m, h in self.histories.items()}, self.lengths,
            self.blocks_per_group, waiting, running, self.fragment)
        self.adjustments.append({'at': now, 'before': before, 'after': dict(self.limits),
            'reason': 'source_8s_block_redistribution', 'fragment': self.fragment})
        self.last_adjustment = now

    def suggestions(self, snapshot: PolicySnapshot) -> tuple[BatchSuggestion, ...]:
        """Source rotated P sweep then D sweep; update cursor only on admission."""
        self.adjust(snapshot)
        def prefill_paused(model_id):
            pause = snapshot.pause_prefill
            return pause if isinstance(pause, bool) else model_id in pause
        demands, result = {d.model_id: d for d in snapshot.demands}, []
        for stage in ('P', 'D'):
            for offset in range(1, len(self.models) + 1):
                m = self.models[(self.cursors[stage] + offset) % len(self.models)]
                d = demands[m]
                if m in snapshot.blocked or (m, stage) in snapshot.busy:
                    continue
                if snapshot.credit_admissible is not None:
                    if (m, stage) not in snapshot.credit_admissible:
                        continue
                elif (10 if stage == 'P' else self.decode_credits[m]) > snapshot.available_credits:
                    continue
                if stage == 'P' and prefill_paused(m):
                    continue
                ids = sorted(d.prefill + d.rebuild if stage == 'P' else d.decode,
                             key=lambda rid: d.ordinals.get(rid, 0))
                cap = min(self.maximum_batch, max(0, self.limits[m]))
                if stage == 'P':
                    inflight = snapshot.inflight_counts.get(m, 0)
                    cap = min(cap, snapshot.max_model_inflight - inflight,
                              self.limits[m] - inflight)
                chosen, tokens = [], 0
                for rid in ids:
                    if len(chosen) >= cap:
                        break
                    if stage == 'P':
                        length = d.prompt_lengths[rid]
                        if tokens + length > self.max_batched_tokens:
                            break
                        if chosen and (rid in d.rebuild) != (chosen[0] in d.rebuild):
                            break
                        tokens += length
                    chosen.append(rid)
                if chosen:
                    result.append(BatchSuggestion(m, stage, tuple(chosen),
                        'REBUILD' if stage == 'P' and chosen[0] in d.rebuild else 'GENERATE'))
        pending_p = any((d.prefill or d.rebuild) and d.model_id not in snapshot.blocked
            and not prefill_paused(d.model_id)
            and snapshot.inflight_counts.get(d.model_id, 0) <
            min(snapshot.max_model_inflight, self.limits[d.model_id]) for d in snapshot.demands)
        if (pending_p and not any(s == 'P' for _, s in snapshot.busy)
                and not any(s.stage == 'P' for s in result)):
            return ()
        return tuple(result)

    def suggest(self, snapshot: PolicySnapshot) -> BatchSuggestion | None:
        suggestions = self.suggestions(snapshot)
        return suggestions[0] if suggestions else None

    def admitted(self, suggestion: BatchSuggestion) -> None:
        self.cursors[suggestion.stage] = self.models.index(suggestion.model_id)


class CreditLedger:
    """Ten SM parts PER TP RANK, each rank one original MuxServe SMResource
    (muxserve/muxsched/resource.py), which allocates and frees the parts.

    This ledger adds only what the external grant protocol needs: one lease
    per batch id (a repeated acquire with the same contents is a no-op, a
    changed one a conflict), at most one lease per (model, stage), and the
    all-or-nothing allocation across ranks. ``multiplier`` (1 or 3) is passed
    to SMResource as its overload_threshold explicitly; SMResource's own
    default (4) is never used. Like the original scheduler, every allocation
    uses overload=True and an overloaded part is picked with random.choice.
    """
    def __init__(self, multiplier=3, ranks=(0,)):
        if multiplier not in (1, 3) or not ranks or len(set(ranks)) != len(ranks):
            raise ValueError('Invalid software-credit profile')
        self.capacity, self.threshold = 10, multiplier
        self.resources = {r: SMResource(overload_threshold=multiplier) for r in ranks}
        if any(len(res._sm_parts) != self.capacity for res in self.resources.values()):
            raise ValueError('Original SMResource no longer has ten parts per GPU')
        self.leases = {}
        self._parts = {}  # batch id -> {rank: [SMPart, ...]} handed back to SMResource.free

    @property
    def refs(self):
        """Per-rank reference count of each part (diagnostics and checks)."""
        return {r: [res._sm_parts[i].ref_count for i in range(self.capacity)]
                for r, res in self.resources.items()}

    @property
    def available(self):
        return min(res.num_free_sms + min(res.overload_threshold, len(res.unoverloaded_sms))
                   for res in self.resources.values())

    def can_allocate(self, credits):
        return (type(credits) is int and 0 < credits <= self.capacity
                and all(res.can_allocate(credits, overload=True) for res in self.resources.values()))

    def acquire(self, batch_id, model_id, stage, credits):
        signature = (model_id, stage, credits)
        if batch_id in self.leases:
            if self.leases[batch_id][0] != signature:
                raise IntegrationError('IDENTITY_CONFLICT', 'Credit lease changed contents')
            return True
        if any(item[0][:2] == signature[:2] for item in self.leases.values()) or not self.can_allocate(credits):
            return False
        parts = {rank: res.allocate(credits, overload=True) for rank, res in self.resources.items()}
        self._parts[batch_id] = parts
        self.leases[batch_id] = (signature, {rank: tuple(part.id for part in items)
                                             for rank, items in parts.items()})
        return True

    def release(self, batch_id):
        lease = self.leases.pop(batch_id, None)
        if lease is None:
            return
        for rank, items in self._parts.pop(batch_id).items():
            self.resources[rank].free(items)
