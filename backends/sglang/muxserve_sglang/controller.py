"""Authenticated, CPU-only coordinator for one P/D execution group per model.

The owner registers ``rpc_methods()`` on its existing authenticated local RPC
server.  Rank processes never mutate the HeadwiseStore directly.
"""
from __future__ import annotations

from collections import OrderedDict
from copy import deepcopy
from dataclasses import dataclass, field
from threading import Condition, Event, RLock
import math
import time
from typing import Any
from uuid import uuid4

from .config import DeploymentConfig
from .contracts import (BatchPlan, IntegrationError, KVTarget, PackedBlocks, RequestIdentity,
                        RequestRef, RequestSpec, ResumeRecord, pack_tables)
from .events import RuntimeOutbox
from .ipc import CommandSnapshot
from . import trace
from .policy import (ADBS, BatchFeedback, BatchSuggestion, CreditLedger, ModelDemand,
                     PolicySnapshot)


_STAGES = frozenset(("P", "D"))
_REPORTS = frozenset(("READY", "GPU_DONE", "HANDOFF_READY", "QUIESCED", "ERROR"))
_COMMAND_WINDOW = 1024
_REPORT_WINDOW = 4096
_INT64_MAX = (1 << 63) - 1
# A parked decode run ends normally if it could not resume within this.
_PARK_LIMIT_S = 2.0


class _RetiredIds:
    """Exact tombstones with a fixed lifetime ceiling; exhaustion fails closed."""

    LIMIT = 100_000

    def __init__(self, batch_limit: int = 100_000) -> None:
        self.batch_limit = batch_limit
        self.ids: dict[str, set[str]] = {"request": set(), "batch": set()}

    def limit(self, category: str) -> int:
        return self.batch_limit if category == "batch" else self.LIMIT

    def add(self, category: str, identity: str) -> None:
        selected = self.ids[category]
        if identity not in selected and len(selected) >= self.limit(category):
            raise IntegrationError("HISTORY_FULL", "Group ID history requires a fresh epoch")
        selected.add(identity)

    def contains(self, category: str, identity: str) -> bool:
        return identity in self.ids[category]

    def full(self, category: str) -> bool:
        return len(self.ids[category]) >= self.limit(category)


@dataclass
class _Rank:
    stage: str
    rank: int
    pid: int
    capabilities: dict[str, Any]
    next_command: int = 1
    acked_command: int = 0
    commands: OrderedDict[int, dict | CommandSnapshot] = field(default_factory=OrderedDict)
    reports: OrderedDict[str, dict] = field(default_factory=OrderedDict)
    last_report_seq: int = 0
    alive: bool = True
    exit_code: int | None = None


@dataclass
class _Request:
    spec: RequestSpec
    submitted_at: float
    deadline: float
    ordinal: int
    attempt: int = 0
    output: list[int] = field(default_factory=list)
    state: str = "QUEUED"
    finish_reason: str | None = None
    allocation_id: str | None = None
    valid_tokens: int = 0
    version: int = 0
    first_token_at: float | None = None
    finished_at: float | None = None
    pending_reason: str | None = None
    resume_record_id: str | None = None
    resumed_attempt: int | None = None
    # Every P rank released this attempt's native request: its last prefill
    # batch quiesced normally on all P ranks, and none was prepared since.
    p_released: bool = False

    @property
    def identity(self) -> RequestIdentity:
        return RequestIdentity(self.spec.model_id, self.spec.request_id, self.attempt)


@dataclass
class _Batch:
    plan: BatchPlan
    reservation: Any
    holder: str
    views: dict[str, Any]
    started_at: float | None = None
    gpu_completed_at: float | None = None
    ready_at: float | None = None
    quiesce_deadline: float | None = None
    prepared_at: float = 0.0
    status: str = "PREPARING"
    ready: set[tuple[str, int]] = field(default_factory=set)
    gpu: dict[int, dict] = field(default_factory=dict)
    handoff: set[int] = field(default_factory=set)
    pd_handoff_ms: float | None = None
    quiesced: set[tuple[str, int]] = field(default_factory=set)
    output_metadata: dict[str, dict] = field(default_factory=dict)
    committed_views: dict[str, Any] = field(default_factory=dict)
    failure_reason: str | None = None
    feedback_sent: bool = False
    quiesce_required: set[tuple[str, int]] = field(default_factory=set)
    reserved_blocks: dict[int, int] = field(default_factory=dict)
    rolled_back_blocks: dict[int, int] = field(default_factory=dict)
    free_before_feedback: dict | None = None
    selected_input_tokens: int = 0
    # Multi-step D grant (plan.steps > 1): per-step all-rank agreed results,
    # memoized continue decisions and the run's starting request positions.
    progress_at: float | None = None
    run_start_valid: dict[str, int] = field(default_factory=dict)
    run_start_output: dict[str, int] = field(default_factory=dict)
    run_published: dict[str, int] = field(default_factory=dict)
    run_reports: dict[int, dict[int, tuple]] = field(default_factory=dict)
    run_decisions: dict[int, bool | str] = field(default_factory=dict)
    # Set while a parked run (decision PARK) has released its credits.
    parked_at: float | None = None
    run_applied: int = 0
    run_steps_done: int | None = None
    # PackedBlocks per request for this batch's reserved / committed views.
    # Views are immutable for the batch's lifetime, so each is packed once.
    packed: dict[str, dict[int, PackedBlocks]] = field(default_factory=dict)
    packed_committed: dict[str, dict[int, PackedBlocks]] = field(default_factory=dict)
    # (stage, rank) keys that prepare, READY and quiesce this batch. Every
    # rank, except decode batches with decode_skip_prefill_ranks (D only).
    participants: frozenset = frozenset()
    # prefill_early_token: the prefill's tokens were published at commit.
    tokens_published: bool = False


def _packed_view(batch: _Batch, request_id: str, committed: bool = False) -> dict[int, PackedBlocks]:
    cache = batch.packed_committed if committed else batch.packed
    packed = cache.get(request_id)
    if packed is None:
        views = batch.committed_views if committed else batch.views
        packed = cache[request_id] = pack_tables(views[request_id].block_tables)
    return packed


def _same_table(packed: PackedBlocks, supplied: Any) -> bool:
    try:
        return packed == PackedBlocks.pack(supplied)
    except IntegrationError:
        return False


@dataclass
class _Cleanup:
    cleanup_id: str
    identity: RequestIdentity
    reason: str
    deadline: float
    quiesced: set[tuple[str, int]] = field(default_factory=set)
    participants: frozenset = frozenset()


@dataclass
class _Group:
    model_id: str
    group_id: str
    ranks: tuple[int, ...]
    outbox: RuntimeOutbox
    registered: dict[tuple[str, int], _Rank] = field(default_factory=dict)
    capability: dict[str, Any] | None = None
    fenced: str | None = None
    draining: bool = False
    drain_issued: bool = False
    batches: dict[str, _Batch] = field(default_factory=dict)
    broker: dict[int, dict] = field(default_factory=dict)
    cleanups: dict[str, _Cleanup] = field(default_factory=dict)
    next_room: int = 1
    retired: _RetiredIds = field(default_factory=_RetiredIds)
    command_reserved: dict[tuple[str, int], dict[str, int]] = field(default_factory=dict)
    uncertain_batches: set[str] = field(default_factory=set)
    # Bumped whenever broker state that a waiting rank may poll changes.
    epoch: int = 0

    @property
    def required(self) -> set[tuple[str, int]]:
        return {(stage, rank) for stage in _STAGES for rank in self.ranks}

    @property
    def ready(self) -> bool:
        return not self.fenced and not self.draining and self.required == set(self.registered)


class MuxController:
    """Synchronous mutation API; external callers schedule their own polling."""

    def __init__(self, config: DeploymentConfig, store: Any, *, clock=time.monotonic,
                 arrival_rates: dict[str, float] | None = None,
                 expected_lengths: dict[str, int] | None = None,
                 event_window: int = 1024, initializing: bool = False):
        if not isinstance(config, DeploymentConfig):
            raise IntegrationError("INVALID_ARGUMENT", "DeploymentConfig required")
        self.config, self.store, self.clock = config, store, clock
        self.lock = RLock()
        # Rank long-polls wait here (releasing the owner lock); the driver
        # thread waits on ``wake`` instead of a fixed sleep.
        self._cv = Condition(self.lock)
        # Parked decode runs wait here for credits to come back.
        self._park_cv = Condition(self.lock)
        self.wake = Event()
        # Set by every runtime event; the driver's event pump thread waits here.
        self.events_wake = Event()
        if type(initializing) is not bool:
            raise IntegrationError("INVALID_ARGUMENT", "Initialization gate must be boolean")
        self.initializing = initializing
        self._initialization_token = uuid4().hex if initializing else None
        self.protected_failure: str | None = None
        self.protected_fence_errors: dict[str, str] = {}
        self.groups: dict[str, _Group] = {}
        self.by_group: dict[str, _Group] = {}
        self.requests: dict[tuple[str, str], _Request] = {}
        self.records: dict[str, ResumeRecord] = {}
        self.capacity_blocked: set[str] = set()
        self.recovery_protected: dict[str, tuple[str, str]] = {}
        self.recovery_cycles = self.recovery_suspensions = 0
        self._ordinal = 0
        self.credits = CreditLedger(config.credit_multiplier, tuple(range(config.tp_size)))
        blocks_per_group = {}
        self._capacities = {}
        capacity_snapshot = store.snapshot()
        for model in config.models:
            layout = model.layout(config.tp_size)
            store.register_model(layout)
            group_id = uuid4().hex
            group = _Group(model.model_id, group_id, layout.ranks,
                           RuntimeOutbox(group_id, model.model_id, event_window),
                           retired=_RetiredIds(config.batch_history_limit))
            group.outbox.reserve("group_error", 1)
            group.outbox.on_emit = self.events_wake.set
            self.groups[model.model_id] = group
            self.by_group[group.group_id] = group
            blocks_per_group[model.model_id] = layout.blocks_per_token_group
            capacities = (capacity_snapshot["model_partitions"][model.model_id]["total_blocks_by_rank"]
                          if capacity_snapshot["pool_mode"] == "static" else
                          {rank: capacity_snapshot["ranks"][rank]["total_blocks"]
                           for rank in layout.ranks})
            self._capacities[model.model_id] = {
                "model_id": model.model_id, "pool_mode": capacity_snapshot["pool_mode"],
                "blocks_by_rank": dict(capacities),
                "blocks_per_token_group": layout.blocks_per_token_group,
                "capacity_tokens": min(config.max_context_tokens,
                    min(capacities.values()) // layout.blocks_per_token_group * layout.block_size)}
        self._policy_parameters = dict(
            decode_credits={m.model_id: m.decode_mps // 10 for m in config.models},
            blocks_per_group=blocks_per_group, total_blocks=config.pool_blocks_per_rank,
            rates=deepcopy(arrival_rates), expected_lengths=deepcopy(expected_lengths),
            maximum_batch=config.max_model_inflight, clock=clock)
        self.policy = ADBS(**self._policy_parameters)
        self._tick_count = 0
        self._tick_duration_ms = 0.0
        # inline_start: allowed only while a scheduling driver owns grants
        # (ControllerDriver enables it and clears it on pause/failure). An
        # unexpected inline START failure is kept for the driver to raise.
        self._inline_start_allowed = False
        self._inline_starting = False
        self.inline_start_error: BaseException | None = None
        self.inline_start_count = 0

    def set_inline_start(self, allowed: bool) -> None:
        with self.lock:
            self._inline_start_allowed = bool(allowed) and self.config.inline_start

    def _participants(self, group: _Group, stage: str) -> frozenset:
        if stage == "D" and self.config.decode_skip_prefill_ranks:
            return frozenset(("D", rank) for rank in group.ranks)
        return frozenset(group.required)

    def _start_ready_inline(self) -> None:
        """Caller holds the lock. With inline_start, START every fully READY
        preparation now rather than on the driver's next pass. start_batch
        makes every check; a batch it refuses (NOT_READY: credits, expiry;
        EVENT_BACKPRESSURE) stays PREPARING for the driver's ordinary retry.
        Any other failure is kept and re-raised by the driver, exactly as if
        its own start_batch call had raised it."""
        if (not self._inline_start_allowed or self._inline_starting
                or self.inline_start_error is not None or self.protected_failure is not None):
            return
        self._inline_starting = True
        try:
            for group in self.groups.values():
                if not group.ready:
                    continue
                # Prefills first: a READY prefill is what parked decodes wait for.
                for batch in sorted(tuple(group.batches.values()), key=lambda b: b.plan.stage != "P"):
                    if batch.status != "PREPARING" or batch.ready != batch.participants:
                        continue
                    try:
                        self.start_batch(group.group_id, batch.plan.batch_id)
                        if batch.status == "RUNNING":
                            self.inline_start_count += 1
                    except IntegrationError as exc:
                        if exc.code not in ("NOT_READY", "EVENT_BACKPRESSURE"):
                            self.inline_start_error = exc
                            self._inline_start_allowed = False
                            self.wake.set()
                            return
                    except Exception as exc:  # pragma: no cover - surfaced by the driver
                        self.inline_start_error = exc
                        self._inline_start_allowed = False
                        self.wake.set()
                        return
        finally:
            self._inline_starting = False

    def rpc_methods(self) -> dict[str, Any]:
        """Explicit allowlist for StorageRPCServer(extra_methods=...)."""
        names = ("register_rank", "poll_commands", "rank_report", "rank_failed", "rank_exited",
                 "submit_request", "cancel_request", "suspend_request", "resume_request",
                 "prepare_batch", "start_batch", "abort_batch", "tick", "snapshot",
                 "request_output", "poll_events", "ack_events", "broker_stage",
                 "broker_lookup", "broker_ack", "broker_poll", "broker_register_destination",
                 "broker_bootstrap", "begin_drain", "close_ready",
                 "forget_request", "metrics", "request_capacity",
                 "poll_commands_wait", "decode_step", "decode_resume")
        return {"controller_" + name: getattr(self, name) for name in names}

    def _release_credits(self, batch_id: str) -> None:
        """Caller holds the lock. Parked decode runs re-check on every release."""
        self.credits.release(batch_id)
        self._park_cv.notify_all()

    def _notify(self, group: _Group | None = None) -> None:
        """Caller holds the lock. Wake the driver; with a group, wake its waiting ranks."""
        with self.lock:  # Re-entrant; also safe for direct helper calls.
            if group is not None:
                group.epoch += 1
            self._cv.notify_all()
        self.wake.set()

    def group_id(self, model_id: str) -> str:
        with self.lock:
            return self._model(model_id).group_id

    def _model(self, model_id: str) -> _Group:
        try:
            return self.groups[model_id]
        except (KeyError, TypeError) as exc:
            raise IntegrationError("UNKNOWN_MODEL", "Unknown model") from exc

    def _group(self, group_id: str) -> _Group:
        try:
            return self.by_group[group_id]
        except (KeyError, TypeError) as exc:
            raise IntegrationError("STALE_GROUP", "Unknown execution group") from exc

    def _request(self, model_id: str, request_id: str) -> _Request:
        try:
            return self.requests[(model_id, request_id)]
        except (KeyError, TypeError) as exc:
            raise IntegrationError("UNKNOWN_REQUEST", "Unknown request") from exc

    def request_capacity(self, model_id: str) -> dict:
        """Immutable total model capacity, independent of current pool occupancy."""
        with self.lock:
            self._model(model_id)
            return deepcopy(self._capacities[model_id])

    def submit_initialization_request(self, spec: RequestSpec) -> dict:
        """Supervisor-local warm-up entry; deliberately absent from the RPC allowlist."""
        with self.lock:
            if not self.initializing:
                raise IntegrationError("INVALID_STATE", "Initialization has already ended")
            return self.submit_request(spec, initialization_token=self._initialization_token)

    def finish_initialization(self) -> None:
        with self.lock:
            self._require_admission()
            if not self.initializing:
                raise IntegrationError("INVALID_STATE", "Initialization has already ended")
            if (self.requests or any(not group.ready or group.cleanups or group.outbox.pending
                    or any(batch.status not in ("DONE", "ABORTED") for batch in group.batches.values())
                    or any(rank.commands for rank in group.registered.values())
                    for group in self.groups.values())):
                raise IntegrationError("NOT_READY", "Warm-up requests, output and ownership must drain")
            # Keep request/batch tombstones and command/event sequences, but
            # initialization must not train the serving policy or inflate its
            # request/compute metrics. Startup evidence records its own work.
            self.policy = ADBS(**self._policy_parameters)
            self._tick_count = 0
            self._tick_duration_ms = 0.0
            self.recovery_cycles = self.recovery_suspensions = 0
            self.initializing = False
            self._initialization_token = None

    @staticmethod
    def _sampling(spec: RequestSpec) -> None:
        sampling = spec.sampling
        if not isinstance(sampling, dict) or set(sampling) - {"temperature", "top_p", "top_k", "ignore_eos"}:
            raise IntegrationError("UNSUPPORTED_SAMPLING", "Only greedy sampling and ignore_eos are supported")
        temperature = sampling.get("temperature", 0)
        top_p = sampling.get("top_p", 1)
        top_k = sampling.get("top_k", 1)
        if (type(temperature) not in (int, float) or temperature != 0
                or type(top_p) not in (int, float) or top_p != 1
                or type(top_k) is not int or top_k not in (0, 1)
                or type(sampling.get("ignore_eos", False)) is not bool):
            raise IntegrationError("UNSUPPORTED_SAMPLING", "Greedy sampling required")

    def _validate_spec(self, spec: RequestSpec) -> None:
        if not isinstance(spec, RequestSpec):
            raise IntegrationError("INVALID_ARGUMENT", "RequestSpec required")
        self._model(spec.model_id)
        if (not isinstance(spec.request_id, str) or not 0 < len(spec.request_id) <= 256
                or type(spec.max_new_tokens) is not int or not 0 < spec.max_new_tokens <= self.config.max_new_tokens
                or type(spec.timeout_ms) is not int or not 0 < spec.timeout_ms <= 3_600_000
                or not isinstance(spec.input_ids, tuple) or not 0 < len(spec.input_ids) <= self.config.max_input_tokens
                or len(spec.input_ids) + spec.max_new_tokens > self.config.max_context_tokens
                or any(type(token) is not int or token < 0 for token in spec.input_ids)):
            raise IntegrationError("INVALID_ARGUMENT", "Invalid request limits or token IDs")
        self._sampling(spec)

    def register_rank(self, group_id: str, stage: str, rank: int, pid: int,
                      capabilities: dict[str, Any]) -> dict:
        with self.lock:
            group = self._group(group_id)
            if group.fenced or group.draining:
                raise IntegrationError("GROUP_UNAVAILABLE", "Group is fenced or draining")
            key = (stage, rank)
            if key not in group.required or type(pid) is not int or pid <= 0:
                raise IntegrationError("UNKNOWN_RANK", "Unknown rank or PID")
            if (not isinstance(capabilities, dict)
                    or set(capabilities) != {"vocab_size", "eos_token_ids", "native_pd", "headwise_kv"}
                    or type(capabilities["vocab_size"]) is not int or capabilities["vocab_size"] <= 0
                    or not isinstance(capabilities["eos_token_ids"], (tuple, list))
                    or any(type(x) is not int or not 0 <= x < capabilities["vocab_size"]
                           for x in capabilities["eos_token_ids"])
                    or capabilities["native_pd"] is not True or capabilities["headwise_kv"] is not True):
                raise IntegrationError("CAPABILITY_MISMATCH", "Native P/D and Headwise capabilities required")
            normalized = dict(capabilities, eos_token_ids=tuple(capabilities["eos_token_ids"]))
            existing = group.registered.get(key)
            if existing:
                if existing.pid != pid or existing.capabilities != normalized:
                    raise IntegrationError("IDENTITY_CONFLICT", "Rank identity or capability changed")
            else:
                if group.capability is not None and group.capability != normalized:
                    raise IntegrationError("CAPABILITY_MISMATCH", "Ranks disagree on tokenizer/runtime capabilities")
                group.capability = deepcopy(normalized)
                group.registered[key] = _Rank(stage, rank, pid, deepcopy(normalized))
            return {"group_id": group.group_id, "ready": group.ready,
                    "required": sorted(group.required), "registered": sorted(group.registered)}

    def _command(self, group: _Group, key: tuple[str, int], kind: str, payload: dict) -> None:
        target = group.registered[key]
        owner = payload.get("_command_owner")
        payload = {k: v for k, v in payload.items() if k != "_command_owner"}
        held = group.command_reserved.setdefault(key, {})
        if owner is not None and held.get(owner, 0) < 1:
            raise IntegrationError("COMMAND_BACKPRESSURE", "Missing reserved command slot")
        if len(target.commands) + sum(held.values()) + (owner is None) > _COMMAND_WINDOW:
            raise IntegrationError("COMMAND_BACKPRESSURE", "Rank command replay window full")
        command = {"command_seq": target.next_command, "command_id": uuid4().hex,
                   "group_id": group.group_id, "model_id": group.model_id,
                   "stage": key[0], "rank": key[1], "kind": kind, **payload}
        target.commands[target.next_command] = (CommandSnapshot.capture(command)
            if kind in ("PREPARE", "HANDOFF") else deepcopy(command))
        target.next_command += 1
        trace.ev("cmd", payload.get("batch_id") or payload.get("cleanup_id"), kind=kind,
                 m=group.model_id, s=key[0], r=key[1])
        if owner is not None:
            held[owner] -= 1
        self._notify()

    @staticmethod
    def _reserve_commands(group: _Group, owner: str, count: int, keys=None) -> None:
        keys = group.required if keys is None else keys
        if any(len(group.registered[key].commands) +
               sum(group.command_reserved.get(key, {}).values()) + count > _COMMAND_WINDOW
               for key in keys):
            raise IntegrationError("COMMAND_BACKPRESSURE", "Rank command replay window full")
        for key in keys:
            group.command_reserved.setdefault(key, {})[owner] = count

    @staticmethod
    def _release_commands(group: _Group, owner: str) -> None:
        for held in group.command_reserved.values():
            held.pop(owner, None)

    def poll_commands(self, group_id: str, stage: str, rank: int, pid: int,
                      after_seq: int = 0, ack_seq: int | None = None) -> tuple[dict, ...]:
        with self.lock:
            group = self._group(group_id)
            target = group.registered.get((stage, rank))
            if target is None or target.pid != pid or not target.alive:
                raise IntegrationError("UNKNOWN_RANK", "Rank registration/PID required")
            if ack_seq is not None:
                if type(ack_seq) is not int or not target.acked_command <= ack_seq < target.next_command:
                    raise IntegrationError("INVALID_ACK", "Invalid contiguous command ACK")
                for seq in list(target.commands):
                    if seq <= ack_seq:
                        del target.commands[seq]
                target.acked_command = ack_seq
            if type(after_seq) is not int or after_seq < target.acked_command or after_seq >= target.next_command:
                raise IntegrationError("COMMAND_GAP", "Command replay precedes retained history")
            self._retire_completed(group)
            return tuple(cmd.detach() if isinstance(cmd, CommandSnapshot) else deepcopy(cmd)
                         for seq, cmd in target.commands.items() if seq > after_seq)

    def poll_commands_wait(self, group_id: str, stage: str, rank: int, pid: int,
                           after_seq: int = 0, ack_seq: int | None = None,
                           wait_ms: int | float = 0, seen_epoch: int | None = None) -> dict:
        """Long-poll variant: return at once if commands exist or the group's
        broker epoch differs from ``seen_epoch``; otherwise wait up to wait_ms."""
        if type(wait_ms) not in (int, float) or not 0 <= wait_ms <= 1000:
            raise IntegrationError("INVALID_ARGUMENT", "Command wait must be 0..1000 ms")
        if seen_epoch is not None and type(seen_epoch) is not int:
            raise IntegrationError("INVALID_ARGUMENT", "Epoch must be an integer")
        deadline = time.monotonic() + wait_ms / 1000
        with self.lock:
            group = self._group(group_id)
            target = group.registered.get((stage, rank))
            if target is None or target.pid != pid or not target.alive:
                raise IntegrationError("UNKNOWN_RANK", "Rank registration/PID required")
            acked = False
            if ack_seq is not None:
                if type(ack_seq) is not int or not target.acked_command <= ack_seq < target.next_command:
                    raise IntegrationError("INVALID_ACK", "Invalid contiguous command ACK")
                if ack_seq > target.acked_command:
                    for seq in list(target.commands):
                        if seq <= ack_seq:
                            del target.commands[seq]
                    target.acked_command = ack_seq
                    acked = True
            if type(after_seq) is not int or after_seq < target.acked_command or after_seq >= target.next_command:
                raise IntegrationError("COMMAND_GAP", "Command replay precedes retained history")
            if acked:
                self._retire_completed(group)
            while (target.next_command - 1 <= after_seq and target.alive
                   and (seen_epoch is None or seen_epoch == group.epoch)):
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                self._cv.wait(remaining)
            commands = tuple(cmd.detach() if isinstance(cmd, CommandSnapshot) else deepcopy(cmd)
                             for seq, cmd in target.commands.items() if seq > after_seq)
            if trace.enabled():
                for cmd in commands:
                    trace.ev("dlv", cmd.get("batch_id") or cmd.get("cleanup_id"), kind=cmd.get("kind"),
                             m=group.model_id, s=stage, r=rank, seq=cmd.get("command_seq"))
            reply = {"commands": commands, "epoch": group.epoch}
            if self.config.bootstrap_in_poll and stage == "P":
                # Exactly the rooms broker_bootstrap would report READY (every
                # D destination registered, not aborted, group not fenced),
                # with the identity the P sender must match.
                reply["bootstrap_ready"] = () if group.fenced else tuple(
                    (room, entry["allocation_id"], entry["reserved_version"], entry["target_valid_tokens"])
                    for room, entry in group.broker.items()
                    if entry["state"] != "ABORTED" and set(entry["destinations"]) == set(group.ranks))
            return reply

    def submit_request(self, spec: RequestSpec, *, initialization_token: str | None = None) -> dict:
        trace.ev("submit", None, m=getattr(spec, "model_id", None), req=getattr(spec, "request_id", None))
        with self.lock:
            self._require_admission()
            if self.initializing and initialization_token != self._initialization_token:
                raise IntegrationError("NOT_READY", "Dedicated startup warm-up has not completed")
            self._validate_spec(spec)
            key = (spec.model_id, spec.request_id)
            old = self.requests.get(key)
            if old:
                if old.spec != spec:
                    raise IntegrationError("IDENTITY_CONFLICT", "Request replay changed contents")
                return self._output(old)
            group = self._model(spec.model_id)
            if group.retired.contains("request", spec.request_id):
                raise IntegrationError("RETIRED_REQUEST", "Request ID is outside retained replay window")
            if (len(group.retired.ids["request"]) +
                    sum(r.spec.model_id == spec.model_id for r in self.requests.values())
                    >= group.retired.LIMIT):
                raise IntegrationError("HISTORY_FULL", "Start a fresh group after safe drain")
            if not group.ready:
                raise IntegrationError("NOT_READY", "Whole P/D execution group required")
            if any(token >= group.capability["vocab_size"] for token in spec.input_ids):
                raise IntegrationError("INVALID_ARGUMENT", "Input token exceeds registered vocabulary")
            layout = self.config.model(spec.model_id).layout(self.config.tp_size)
            # The final sampled token never needs a following KV write. Check
            # total possible capacity; actual blocks still grow on each grant.
            maximum_kv = len(spec.input_ids) + spec.max_new_tokens - 1
            required = ((maximum_kv + layout.block_size - 1) // layout.block_size
                        * layout.blocks_per_token_group)
            if any(required > blocks for blocks in self._capacities[spec.model_id]["blocks_by_rank"].values()):
                raise IntegrationError("KV_CAPACITY_EXHAUSTED",
                    "Maximum request KV exceeds the total configured shared/static model capacity")
            if sum(r.spec.model_id == spec.model_id for r in self.requests.values()) >= self.config.queue_limit:
                raise IntegrationError("QUEUE_FULL", "Model retained request limit is full")
            now = self.clock()
            self._ordinal += 1
            request = _Request(deepcopy(spec), now, now + spec.timeout_ms / 1000, self._ordinal)
            self.requests[key] = request
            self._notify()
            return self._output(request)

    @staticmethod
    def _output(request: _Request) -> dict:
        return {"model_id": request.spec.model_id, "request_id": request.spec.request_id,
                "attempt": request.attempt, "output_token_ids": tuple(request.output),
                "output_count": len(request.output), "state": request.state,
                "finish_reason": request.finish_reason, "submitted_at": request.submitted_at,
                "first_token_at": request.first_token_at, "finished_at": request.finished_at,
                "deadline": request.deadline, "pending_reason": request.pending_reason,
                "resume_record_id": request.resume_record_id}

    def request_output(self, model_id: str, request_id: str) -> dict:
        with self.lock:
            return deepcopy(self._output(self._request(model_id, request_id)))

    def poll_events(self, group_id: str, after_seq: int = 0) -> tuple:
        with self.lock:
            return self._group(group_id).outbox.replay(after_seq)

    def ack_events(self, group_id: str, through_seq: int) -> None:
        with self.lock:
            group = self._group(group_id)
            group.outbox.ack(through_seq)
            self._retire_completed(group)

    def _active(self, request: _Request) -> _Batch | None:
        group = self._model(request.spec.model_id)
        for batch in group.batches.values():
            if (batch.status not in ("DONE", "ABORTED") and any(
                    ref.request_id == request.spec.request_id and ref.attempt == request.attempt
                    for ref in batch.plan.requests)):
                return batch
        return None

    def _occupies_slot(self, request: _Request) -> bool:
        """Count retained physical or native ownership, not waiting work."""
        group = self._model(request.spec.model_id)
        return (request.allocation_id is not None or self._active(request) is not None
                or any(cleanup.identity == request.identity for cleanup in group.cleanups.values()))

    def _snapshot_policy(self) -> PolicySnapshot:
        demands = []
        busy = set()
        blocked = set()
        counts = {}
        for model in self.config.models:
            group = self.groups[model.model_id]
            if not group.ready:
                blocked.add(model.model_id)
            rows = [r for r in self.requests.values() if r.spec.model_id == model.model_id]
            prefill = tuple(r.spec.request_id for r in rows if r.state == "QUEUED")
            rebuild = tuple(r.spec.request_id for r in rows if r.state == "REBUILD")
            protected = self.recovery_protected.get(self._recovery_key(model.model_id))
            decode = tuple(r.spec.request_id for r in rows if r.state == "DECODING"
                           and (protected is None or
                                (r.spec.model_id, r.spec.request_id) == protected))
            demands.append(ModelDemand(model.model_id, prefill, decode, rebuild,
                ordinals={r.spec.request_id: r.ordinal for r in rows},
                prompt_lengths={r.spec.request_id: len(r.spec.input_ids) +
                                max(0, len(r.output) - 1) for r in rows}))
            active = [b for b in group.batches.values() if b.status not in ("DONE", "ABORTED")]
            busy.update((model.model_id, b.plan.stage) for b in active)
            counts[model.model_id] = sum(self._occupies_slot(r) for r in rows)
        rank_free = {rank: data["free_blocks"] for rank, data in self.store.rank_usage().items()}
        # With decode_run_park a prefill prepares without credits (it takes
        # them at START), so it is always credit-admissible.
        admissible = frozenset((m.model_id, stage) for m in self.config.models
                               for stage in _STAGES
                               if (stage == "P" and self.config.decode_run_park)
                               or self.credits.can_allocate(10 if stage == "P" else m.decode_mps // 10))
        return PolicySnapshot(tuple(demands), self.credits.available, frozenset(busy),
                              frozenset(blocked), counts, self.config.max_model_inflight,
                              rank_free, admissible, pause_prefill=frozenset(self.capacity_blocked))

    def _single_decode_fits(self, model_id: str, request_id: str | None = None) -> bool:
        """Use physical free blocks to prove whether one idle D can advance."""
        layout = self.config.model(model_id).layout(self.config.tp_size)
        snapshot = self.store.snapshot()
        for request in self.requests.values():
            if (request.spec.model_id != model_id or request.state != "DECODING"
                    or (request_id is not None and request.spec.request_id != request_id)
                    or request.pending_reason or self._active(request) is not None
                    or self.clock() >= request.deadline):
                continue
            allocation = snapshot["allocations"].get(request.allocation_id)
            if allocation is None or allocation["active_batch"] is not None:
                continue
            target = request.valid_tokens + 1
            extra_groups = max(0, (target + layout.block_size - 1) // layout.block_size -
                               allocation["capacity_tokens"] // layout.block_size)
            free = (snapshot["model_partitions"][model_id]["free_blocks_by_rank"]
                    if snapshot["pool_mode"] == "static" else
                    {rank: data["free_blocks"] for rank, data in snapshot["ranks"].items()})
            if all(free[rank] >=
                   extra_groups * layout.blocks_per_token_group for rank in layout.ranks):
                return True
        return False

    def _recovery_key(self, model_id: str) -> str:
        return "shared" if self.config.kv_pool_mode == "shared" else model_id

    def _recovery_models(self, key: str) -> set[str]:
        return ({model.model_id for model in self.config.models}
                if self.config.kv_pool_mode == "shared" else {key})

    def _enter_capacity_recovery(self, model_id: str) -> None:
        """Protect one oldest physical owner across the affected physical pool."""
        key = self._recovery_key(model_id)
        if key in self.recovery_protected:
            return
        models = self._recovery_models(key)
        owners = sorted((r for r in self.requests.values()
                         if r.spec.model_id in models and self._occupies_slot(r)),
                        key=lambda r: r.ordinal)
        if not owners:
            return
        protected = owners[0]
        identity = (protected.spec.model_id, protected.spec.request_id)
        self.recovery_protected[key] = identity
        self.recovery_cycles += 1
        self.capacity_blocked.update(models)
        # Do not let a previously prepared younger P consume the released
        # capacity after this protection decision. Started work still completes.
        for model in models:
            group = self._model(model)
            for batch in tuple(group.batches.values()):
                identities = {(model, ref.request_id) for ref in batch.plan.requests}
                if (batch.status == "PREPARING" and identities != {identity}
                        and (batch.plan.stage == "D" or identity not in identities)):
                    self._abort_prepared(group, batch, "capacity_recovery")

    def _advance_capacity_recovery(self) -> None:
        for key, identity in tuple(self.recovery_protected.items()):
            protected = self.requests.get(identity)
            models = self._recovery_models(key)
            if protected is None or not self._occupies_slot(protected):
                # A terminal cleanup or explicit suspension ended ownership.
                # One free decode step is deliberately NOT an exit condition.
                del self.recovery_protected[key]
                self.capacity_blocked.difference_update(models)
                continue
            if (protected.pending_reason or protected.state != "DECODING"
                    or self._active(protected) is not None
                    or self._single_decode_fits(protected.spec.model_id, protected.spec.request_id)):
                continue
            victims = sorted((r for r in self.requests.values()
                if r.spec.model_id in models and (r.spec.model_id, r.spec.request_id) != identity
                and r.state == "DECODING" and not r.pending_reason
                and self._active(r) is None and self._occupies_slot(r)),
                key=lambda r: r.ordinal, reverse=True)
            if victims:
                self._stop_request(victims[0], "suspend")
                self.recovery_suspensions += 1

    def tick(self, *, include_adjustments: bool = True) -> dict:
        """Run deadline/recovery accounting and return ADBS proposals; no implicit START."""
        started = time.perf_counter()
        with self.lock:
            if self.protected_failure is not None:
                return {"suggestions": (), "adjustments": (), "groups": {
                    m: {"group_id": g.group_id, "ready": False, "fenced": g.fenced}
                    for m, g in self.groups.items()}}
            now = self.clock()
            for request in self.requests.values():
                if request.state in ("FINISHED", "CANCELLED", "TIMED_OUT", "FAILED"):
                    continue
                if now >= request.deadline and request.pending_reason is None:
                    self._stop_request(request, "timeout")
            for group in self.groups.values():
                for batch in tuple(group.batches.values()):
                    if batch.status == "PREPARING" and now - batch.prepared_at >= batch.plan.prepare_timeout_ms / 1000:
                        self._abort_prepared(group, batch, "prepare_timeout")
                    if batch.status == "DONE" and not batch.feedback_sent:
                        self._maybe_feedback(group, batch)
                    # Request expiry stops publication, not storage ownership.
                    # These independent progress deadlines detect genuine
                    # unknown completion even after a request has timed out.
                    if (not group.fenced and batch.status == "RUNNING"
                            and batch.started_at is not None
                            and now >= max(batch.started_at, batch.progress_at or 0.0)
                                + self.config.request_timeout_ms / 1000):
                        self._fence(group, batch, "execution_watchdog")
                    elif (not group.fenced and batch.status == "HANDOFF"
                            and batch.gpu_completed_at is not None
                            and now >= batch.gpu_completed_at + self.config.prepare_timeout_ms / 1000):
                        self._fence(group, batch, "handoff_watchdog")
                    elif (not group.fenced and batch.status in ("QUIESCE", "REVOKING")
                            and batch.quiesce_deadline is not None and now >= batch.quiesce_deadline):
                        self._fence(group, batch, "quiescence_watchdog")
                for cleanup in tuple(group.cleanups.values()):
                    if not group.fenced and now >= cleanup.deadline:
                        self._fence(group, None, "cleanup_watchdog")
                if (group.fenced and not group.uncertain_batches and
                        all(group.registered.get(key) and
                        group.registered[key].alive for key in group.required)
                        and all(b.status in ("DONE", "ABORTED") for b in group.batches.values())):
                    for request in self.requests.values():
                        if request.spec.model_id == group.model_id and request.state == "FENCED":
                            request.state = "CLEANUP_PENDING"
            for request in self.requests.values():
                if request.state == "CLEANUP_PENDING" and self._active(request) is None:
                    self._begin_cleanup(request)
            self._advance_capacity_recovery()
            for group in self.groups.values():
                if group.draining:
                    self._issue_drain(group)
            snap = self._snapshot_policy()
            suggestions = self.policy.suggestions(snap)
            self._tick_count += 1
            self._tick_duration_ms += (time.perf_counter() - started) * 1000
            return {"suggestions": suggestions,
                    "adjustments": deepcopy(tuple(self.policy.adjustments)) if include_adjustments else (),
                    "groups": {m: {"group_id": g.group_id, "ready": g.ready,
                                    "fenced": g.fenced} for m, g in self.groups.items()}}

    def prepare_batch(self, plan: BatchPlan, *, _after_handoff: "_Batch | None" = None) -> dict:
        """``_after_handoff`` (controller-internal, handoff_prepare_decode): the
        prefill batch whose HANDOFF was just queued; its requests may then
        prepare their first decode batch while that batch is in HANDOFF."""
        trace.ev("prep_call", getattr(plan, "batch_id", None), m=getattr(plan, "model_id", None),
                 s=getattr(plan, "stage", None))
        with self.lock:
            trace.ev("prep_lock", getattr(plan, "batch_id", None))
            self._require_admission()
            if not isinstance(plan, BatchPlan) or plan.stage not in _STAGES or not plan.requests:
                raise IntegrationError("INVALID_ARGUMENT", "Valid nonempty BatchPlan required")
            group = self._group(plan.group_id)
            if group.model_id != plan.model_id or not group.ready:
                raise IntegrationError("NOT_READY", "Model group is not ready")
            old = group.batches.get(plan.batch_id)
            if old:
                if old.plan != plan:
                    raise IntegrationError("IDENTITY_CONFLICT", "Batch replay changed contents")
                return self._batch_view(old)
            if group.retired.contains("batch", plan.batch_id):
                raise IntegrationError("RETIRED_BATCH", "Batch ID is outside retained replay window")
            if len(group.retired.ids["batch"]) + len(group.batches) >= group.retired.limit("batch"):
                raise IntegrationError("HISTORY_FULL", "Start a fresh group after safe drain")
            if (not isinstance(plan.batch_id, str) or not 0 < len(plan.batch_id) <= 256
                    or len(plan.requests) > self.config.max_model_inflight
                    or type(plan.prepare_timeout_ms) is not int or not 0 < plan.prepare_timeout_ms <= 120000
                    or plan.purpose not in ("GENERATE", "REBUILD")
                    or (plan.purpose == "REBUILD" and plan.stage != "P")
                    or type(plan.steps) is not int
                    or not 1 <= plan.steps <= max(1, self.config.decode_run_steps)
                    or (plan.steps > 1 and (plan.stage != "D" or plan.purpose != "GENERATE"))):
                raise IntegrationError("INVALID_ARGUMENT", "Invalid batch size, timeout, purpose or run length")
            if any(b.status not in ("DONE", "ABORTED") and b.plan.stage == plan.stage
                   for b in group.batches.values()):
                raise IntegrationError("STAGE_BUSY", "Stage already has active work")
            if plan.stage == "P" and plan.model_id in self.capacity_blocked:
                raise IntegrationError("OUT_OF_CAPACITY", "Fresh P admission paused during D recovery")
            protected = self.recovery_protected.get(self._recovery_key(plan.model_id))
            if protected is not None and any(
                    (plan.model_id, ref.request_id) != protected for ref in plan.requests):
                raise IntegrationError("OUT_OF_CAPACITY", "Only the protected oldest owner may advance")
            requests, targets = [], []
            for ref in plan.requests:
                if not isinstance(ref, RequestRef):
                    raise IntegrationError("INVALID_ARGUMENT", "RequestRef required")
                request = self._request(plan.model_id, ref.request_id)
                # Request state can change after a policy snapshot. Validate it
                # here as a stale selection, not as a malformed immutable plan.
                if _after_handoff is not None:
                    state_ok = (plan.stage == "D" and request.state == "RUNNING"
                                and _after_handoff.status == "HANDOFF"
                                and self._active(request) is _after_handoff)
                else:
                    state_ok = (request.state == ("REBUILD" if plan.purpose == "REBUILD" else
                                                  "QUEUED" if plan.stage == "P" else "DECODING")
                                and not self._active(request))
                if (request.attempt != ref.attempt or len(request.output) != ref.expected_output_count
                        or request.pending_reason or self.clock() >= request.deadline or not state_ok):
                    raise IntegrationError("STALE_ATTEMPT", "Request state/output changed before prepare")
                if plan.stage == "P":
                    target = len(request.spec.input_ids) + (len(request.output) - 1 if plan.purpose == "REBUILD" else 0)
                else:
                    if plan.steps > request.spec.max_new_tokens - len(request.output):
                        raise IntegrationError("STALE_ATTEMPT", "Decode run exceeds the remaining output budget")
                    # A run reserves capacity for all of its steps up front;
                    # every step then appends one token inside that mapping.
                    target = request.valid_tokens + plan.steps
                targets.append(KVTarget(request.identity, target, request.allocation_id))
                requests.append(request)
            if len({r.spec.request_id for r in requests}) != len(requests):
                raise IntegrationError("IDENTITY_CONFLICT", "Duplicate request in batch")
            if plan.stage == "P":
                occupied = sum(r.spec.model_id == plan.model_id and self._occupies_slot(r)
                               for r in self.requests.values())
                if occupied + len(requests) > min(self.config.max_model_inflight,
                                                  self.policy.limits[plan.model_id]):
                    raise IntegrationError("OUT_OF_CAPACITY", "Model P inflight budget exhausted")
            if plan.stage == "P" and group.next_room + len(requests) - 1 > _INT64_MAX:
                raise IntegrationError("ROOM_EXHAUSTED", "Native bootstrap room IDs exhausted")
            credits = 10 if plan.stage == "P" else self.config.model(plan.model_id).decode_mps // 10
            if self.config.decode_run_park and plan.stage == "D" and self._prefill_awaiting_credits():
                # A READY prefill waits for running decodes to park; a new
                # decode batch must not take the credits it is waiting for.
                raise IntegrationError("OUT_OF_CREDITS", "A ready prefill is waiting for credits")
            if self.config.decode_run_park and plan.stage == "P":
                pass  # Overlaps decoding; credits are taken at START.
            elif not self.credits.acquire(plan.batch_id, plan.model_id, plan.stage, credits):
                raise IntegrationError("OUT_OF_CREDITS", "Per-rank software credits unavailable")
            owner = "batch:" + group.group_id + ":" + plan.batch_id
            participants = self._participants(group, plan.stage)
            reservation = None
            reserve_called = False
            try:
                group.outbox.reserve(owner, (plan.steps + 1) * len(requests) + 2)
                self._reserve_commands(group, owner, 4, participants)
                free_before = self.store.rank_usage()
                reserve_called = True
                reservation = self.store.reserve_kv(plan.batch_id, targets, owner)
                views = {view.identity.request_id: view for view in reservation.views}
                trace.ev("prep_reserved", plan.batch_id)
                batch = _Batch(deepcopy(plan), reservation, owner, views, prepared_at=self.clock(),
                    participants=participants,
                    selected_input_tokens=sum((len(request.spec.input_ids) + len(request.output) - 1)
                        if plan.stage == "P" and plan.purpose == "REBUILD" else
                        len(request.spec.input_ids) if plan.stage == "P" else 1
                        for request in requests))
                group.batches[plan.batch_id] = batch
                if plan.steps > 1:
                    for request in requests:
                        batch.run_start_valid[request.spec.request_id] = request.valid_tokens
                        batch.run_start_output[request.spec.request_id] = len(request.output)
                        batch.run_published[request.spec.request_id] = 0
                if plan.stage == "P":
                    for request in requests:
                        self.store.bind_kv(views[request.spec.request_id].allocation_id,
                                           self._d_holder(request))
                free_after = self.store.rank_usage()
                batch.reserved_blocks = {r: free_before[r]["free_blocks"] - free_after[r]["free_blocks"]
                                         for r in free_before}
            except IntegrationError as exc:
                if reservation is not None or (reserve_called and exc.code in (
                        "RPC_TIMEOUT", "RPC_UNAVAILABLE")):
                    if reservation is None:
                        group.uncertain_batches.add(plan.batch_id)
                    self._fence(group, group.batches.get(plan.batch_id),
                                "prepare_uncertain:" + exc.code)
                else:
                    # A definite failed reserve owns no KV. Unwind admission
                    # before the optional store read used for recovery advice.
                    group.outbox.release(owner)
                    self._release_commands(group, owner)
                    self._release_credits(plan.batch_id)
                    if exc.code == "OUT_OF_CAPACITY":
                        try:
                            if not self._single_decode_fits(plan.model_id):
                                self._enter_capacity_recovery(plan.model_id)
                        except IntegrationError:
                            # A failed diagnostic supplies no proof for
                            # suspension. Preserve the definite capacity error.
                            pass
                raise
            for request in requests:
                request.state = "PREPARING"
                if plan.stage == "P":
                    request.p_released = False
            for key in sorted(participants):
                self._command(group, key, "PREPARE", {
                    **self._command_payload(group, batch, key), "_command_owner": owner})
            self.policy.admitted(BatchSuggestion(plan.model_id, plan.stage,
                tuple(ref.request_id for ref in plan.requests), plan.purpose))
            trace.ev("prep", plan.batch_id, m=plan.model_id, gid=plan.group_id, s=plan.stage,
                     purpose=plan.purpose, steps=plan.steps,
                     reqs=[ref.request_id for ref in plan.requests])
            return self._batch_view(batch)

    @staticmethod
    def _d_holder(request: _Request) -> str:
        return f"D:{request.spec.model_id}:{request.spec.request_id}:{request.attempt}"

    @staticmethod
    def _batch_view(batch: _Batch) -> dict:
        return {"batch_id": batch.plan.batch_id, "group_id": batch.plan.group_id,
                "status": batch.status, "ready": sorted(batch.ready),
                "participants": sorted(batch.participants),
                "reservation_id": batch.reservation.reservation_id}

    def _command_payload(self, group: _Group, batch: _Batch,
                         key: tuple[str, int]) -> dict:
        plans = []
        for ref in batch.plan.requests:
            request = self._request(batch.plan.model_id, ref.request_id)
            view = batch.views[ref.request_id]
            room = None
            if batch.plan.stage == "P":
                for number, entry in group.broker.items():
                    if entry["identity"] == request.identity and entry["batch_id"] == batch.plan.batch_id:
                        room = number
                        break
                if room is None:
                    if group.next_room > _INT64_MAX:
                        raise IntegrationError("ROOM_EXHAUSTED", "Native bootstrap room IDs exhausted")
                    room = group.next_room
                    group.next_room += 1
                    group.broker[room] = {"identity": request.identity,
                                          "batch_id": batch.plan.batch_id,
                                          "allocation_id": view.allocation_id,
                                          "reserved_version": view.version,
                                          "target_valid_tokens": (len(request.spec.input_ids) +
                                              (len(request.output) - 1 if batch.plan.purpose == "REBUILD" else 0)),
                                          "state": "PREPARING", "metadata": {},
                                          "acked": set(), "destinations": {}}
            packed = _packed_view(batch, ref.request_id)
            plans.append({"request_id": request.spec.request_id, "attempt": request.attempt,
                          "input_ids": request.spec.input_ids,
                          "output_token_ids": tuple(request.output),
                          "max_new_tokens": request.spec.max_new_tokens,
                          "sampling": deepcopy(request.spec.sampling),
                          "sampling_state": None, "deadline": request.deadline,
                          "purpose": batch.plan.purpose,
                          "selected_input_ids": request.spec.input_ids +
                              (tuple(request.output[:-1]) if batch.plan.purpose == "REBUILD" else ()),
                          "boundary_token_id": request.output[-1]
                              if batch.plan.purpose == "REBUILD" else None,
                          "target_valid_tokens": view.valid_tokens + 1
                              if batch.plan.stage == "D" else len(request.spec.input_ids) +
                              (len(request.output) - 1 if batch.plan.purpose == "REBUILD" else 0),
                          "bootstrap_room": room,
                          # _command owns the durable deep copy of this payload.
                          "reservation_view": {"allocation_id": view.allocation_id,
                              "identity": {"model_id": view.identity.model_id,
                                  "request_id": view.identity.request_id, "attempt": view.identity.attempt},
                              "capacity_tokens": view.capacity_tokens, "valid_tokens": view.valid_tokens,
                              "block_size": view.block_size, "block_tables": packed,
                              "version": view.version},
                          "rank_block_table": packed[key[1]]})
        payload = {"batch_id": batch.plan.batch_id, "reservation_id": batch.reservation.reservation_id,
                   "batch_stage": batch.plan.stage, "purpose": batch.plan.purpose,
                   "requests": plans, "ordered_request_ids": tuple(ref.request_id for ref in batch.plan.requests)}
        if batch.plan.steps > 1:
            payload["run_steps"] = batch.plan.steps
        return payload

    def start_batch(self, group_id: str, batch_id: str) -> dict:
        with self.lock:
            self._require_admission()
            group = self._group(group_id)
            batch = self._batch(group, batch_id)
            if batch.status == "RUNNING":
                return self._batch_view(batch)
            if (batch.status != "PREPARING" or not group.ready
                    or batch.ready != batch.participants):
                raise IntegrationError("NOT_READY", "Every P/D rank must prepare before fresh START")
            protected = self.recovery_protected.get(self._recovery_key(group.model_id))
            identities = {(group.model_id, ref.request_id) for ref in batch.plan.requests}
            if (protected is not None and identities != {protected}
                    and (batch.plan.stage == "D" or protected not in identities)):
                self._abort_prepared(group, batch, "capacity_recovery")
                raise IntegrationError("NOT_READY", "Capacity recovery revoked younger preparation")
            now = self.clock()
            if (now - batch.prepared_at >= batch.plan.prepare_timeout_ms / 1000 or
                    any(now >= self._request(group.model_id, ref.request_id).deadline or
                        self._request(group.model_id, ref.request_id).pending_reason
                        for ref in batch.plan.requests)):
                self._abort_prepared(group, batch, "expired_before_start")
                raise IntegrationError("NOT_READY", "Preparation or request deadline expired")
            if not group.outbox.owner_can_complete(batch.holder):
                raise IntegrationError("EVENT_BACKPRESSURE", "Start grant window unavailable")
            deferred = (self.config.decode_run_park and batch.plan.stage == "P"
                        and batch.plan.batch_id not in self.credits.leases)
            if deferred:
                # Running decode runs park at their next step boundary and
                # release credits; the driver retries this START then.
                if not self.credits.acquire(batch.plan.batch_id, batch.plan.model_id, "P", 10):
                    raise IntegrationError("NOT_READY", "Prefill waits for decode runs to park")
            try:
                self.store.start_kv_batch(batch.reservation.reservation_id)
            except IntegrationError as exc:
                if deferred:
                    self._release_credits(batch.plan.batch_id)
                if exc.code in ("RPC_TIMEOUT", "RPC_UNAVAILABLE", "BATCH_FENCED"):
                    self._fence(group, batch, "start_uncertain:" + exc.code)
                raise
            batch.status = "RUNNING"
            batch.started_at = self.clock()
            for ref in batch.plan.requests:
                self._request(batch.plan.model_id, ref.request_id).state = "RUNNING"
            for rank in group.ranks:
                self._command(group, (batch.plan.stage, rank), "START",
                              {"batch_id": batch_id, "reservation_id": batch.reservation.reservation_id,
                               "batch_stage": batch.plan.stage, "purpose": batch.plan.purpose,
                               "grant_id": batch.reservation.reservation_id,
                               "_command_owner": batch.holder})
            trace.ev("start", batch_id, m=batch.plan.model_id, s=batch.plan.stage)
            return self._batch_view(batch)

    @staticmethod
    def _batch(group: _Group, batch_id: str) -> _Batch:
        try:
            return group.batches[batch_id]
        except (KeyError, TypeError) as exc:
            raise IntegrationError("UNKNOWN_BATCH", "Unknown batch") from exc

    def abort_batch(self, group_id: str, batch_id: str, reason: str) -> dict:
        with self.lock:
            group = self._group(group_id)
            batch = self._batch(group, batch_id)
            if not isinstance(reason, str) or not reason:
                raise IntegrationError("INVALID_ARGUMENT", "Abort reason required")
            if batch.status == "PREPARING":
                self._abort_prepared(group, batch, reason)
            elif batch.status in ("RUNNING", "HANDOFF", "QUIESCE", "FENCED"):
                self._fence(group, batch, reason)
            elif batch.status not in ("REVOKING", "ABORTED", "DONE"):
                raise IntegrationError("INVALID_STATE", "Batch cannot be aborted")
            return self._batch_view(batch)

    def _abort_prepared(self, group: _Group, batch: _Batch, reason: str) -> None:
        if batch.status != "PREPARING":
            return
        batch.status, batch.failure_reason = "REVOKING", reason
        batch.quiesce_deadline = self.clock() + self.config.prepare_timeout_ms / 1000
        batch.quiesce_required = set(batch.participants)
        for key in sorted(batch.participants):
            self._command(group, key, "REVOKE", {"batch_id": batch.plan.batch_id,
                                                  "reservation_id": batch.reservation.reservation_id,
                                                  "reason": reason,
                                                  "_command_owner": batch.holder})

    def _finish_revoke(self, group: _Group, batch: _Batch) -> None:
        try:
            self.store.abort_kv_batch(batch.reservation.reservation_id)
            for ref in batch.plan.requests:
                request = self._request(group.model_id, ref.request_id)
                if batch.plan.stage == "P":
                    self.store.release_kv(batch.views[ref.request_id].allocation_id,
                                          self._d_holder(request))
        except IntegrationError as exc:
            self._fence(group, batch, "revoke_store_uncertain:" + exc.code)
            raise
        for ref in batch.plan.requests:
            request = self._request(group.model_id, ref.request_id)
            request.state = "CLEANUP_PENDING" if request.pending_reason else (
                "REBUILD" if batch.plan.purpose == "REBUILD" else
                "QUEUED" if batch.plan.stage == "P" else "DECODING")
            if batch.plan.stage == "P" and request.pending_reason is None:
                # The retired fresh allocation kept the store's attempt watermark.
                request.attempt += 1
                request.p_released = False
        for room in tuple(group.broker):
            if group.broker[room]["batch_id"] == batch.plan.batch_id:
                del group.broker[room]
        batch.status = "ABORTED"
        self._release_credits(batch.plan.batch_id)
        group.outbox.release(batch.holder)
        self._release_commands(group, batch.holder)
        for ref in batch.plan.requests:
            request = self._request(group.model_id, ref.request_id)
            if request.state == "CLEANUP_PENDING":
                self._begin_cleanup(request)

    def _fence(self, group: _Group, batch: _Batch | None, reason: str) -> None:
        if group.fenced is None:
            group.outbox.emit("GROUP_ERROR", batch_id=batch.plan.batch_id if batch else None,
                              payload={"reason": reason}, owner="group_error")
        group.fenced = reason
        for item in group.batches.values():
            if item and item.status not in ("DONE", "ABORTED"):
                former = item.status
                item.failure_reason = reason
                if former not in ("QUIESCE", "REVOKING"):
                    item.status = "FENCED"
                    try:
                        self.store.fence_batch(item.reservation.reservation_id)
                    except IntegrationError:
                        pass  # Uncertain ownership remains pinned in this fenced group.
                if former == "HANDOFF" and all(group.registered.get(key) and
                        group.registered[key].alive for key in group.required):
                    self._begin_quiesce(group, item, item.participants)
        for request in self.requests.values():
            if request.spec.model_id == group.model_id and request.state not in (
                    "FINISHED", "CANCELLED", "TIMED_OUT", "SUSPENDED", "FAILED"):
                request.state = "FENCED"
                request.pending_reason = reason

    def _require_admission(self) -> None:
        if self.protected_failure is not None:
            raise IntegrationError("PROTECTED_FAILURE", "Supervisor failure forbids new requests or START grants")

    def protect_failure(self, reason: str) -> dict:
        """Launcher-only irreversible gate; diagnostics cannot reopen admission."""
        if not isinstance(reason, str) or not reason:
            raise IntegrationError("INVALID_ARGUMENT", "Failure reason required")
        with self.lock:
            if self.protected_failure is None:
                # Set before events/store calls: their failures cannot permit work.
                self.protected_failure = reason
                for model_id, group in self.groups.items():
                    try:
                        self._fence(group, None, "supervisor:" + reason)
                    except Exception as exc:
                        group.fenced = group.fenced or "supervisor:" + reason
                        self.protected_fence_errors[model_id] = type(exc).__name__
            return {"protected_failure": self.protected_failure,
                    "fence_errors": dict(self.protected_fence_errors)}

    def rank_failed(self, group_id: str, stage: str, rank: int, pid: int,
                    reason: str) -> dict:
        with self.lock:
            group = self._group(group_id)
            target = group.registered.get((stage, rank))
            if target is None or target.pid != pid or not isinstance(reason, str) or not reason:
                raise IntegrationError("UNKNOWN_RANK", "Registered rank/PID and reason required")
            target.alive = False
            for batch in group.batches.values():
                if batch.status not in ("DONE", "ABORTED"):
                    self._fence(group, batch, "rank_failed:" + reason)
            if group.fenced is None:
                self._fence(group, None, "rank_failed:" + reason)
            return {"group_id": group_id, "fenced": group.fenced,
                    "pinned_reservations": tuple(b.reservation.reservation_id for b in
                                                  group.batches.values() if b.status == "FENCED")}

    def rank_exited(self, group_id: str, stage: str, rank: int, pid: int,
                    exit_code: int) -> dict:
        """Launcher calls this only after OS waitpid proved the registered PID exited."""
        with self.lock:
            group = self._group(group_id)
            target = group.registered.get((stage, rank))
            if target is None or target.pid != pid or type(exit_code) is not int:
                raise IntegrationError("UNKNOWN_RANK", "Registered rank/PID exit proof required")
            if target.exit_code is not None:
                if target.exit_code != exit_code:
                    raise IntegrationError("IDENTITY_CONFLICT", "Exit proof changed")
                return self.close_ready(group_id)
            if not group.drain_issued or exit_code != 0:
                self._fence(group, None, "unexpected_rank_exit")
            target.alive, target.exit_code = False, exit_code
            return self.close_ready(group_id)

    def rank_report(self, group_id: str, stage: str, rank: int, pid: int,
                    report_id: str, batch_id: str, kind: str, payload: dict) -> dict:
        trace.ev("rpt", batch_id, kind=kind, gid=group_id, s=stage, r=rank)
        with self.lock:
            group = self._group(group_id)
            key = (stage, rank)
            target = group.registered.get(key)
            if (target is None or target.pid != pid or not target.alive
                    or kind not in _REPORTS or not isinstance(report_id, str)
                    or not report_id.isdecimal() or report_id.startswith("0")
                    or len(report_id) > 19
                    or not isinstance(payload, dict)):
                raise IntegrationError("INVALID_REPORT", "Registered rank and valid report required")
            report_seq = int(report_id)
            fingerprint = {"batch_id": batch_id, "kind": kind, "payload": deepcopy(payload)}
            prior = target.reports.get(report_seq)
            if prior is not None:
                if prior != fingerprint:
                    self._fence(group, None, "report_content_conflict")
                    raise IntegrationError("IDENTITY_CONFLICT", "Report replay changed contents")
                return {"accepted": True, "duplicate": True}
            if report_seq != target.last_report_seq + 1:
                raise IntegrationError("REPORT_GAP", "Report sequence outside contiguous replay window")
            if len(target.reports) >= _REPORT_WINDOW:
                target.reports.popitem(last=False)
            if kind == "QUIESCED" and batch_id in group.cleanups:
                self._cleanup_quiesced(group, group.cleanups[batch_id], key, payload)
                target.reports[report_seq] = fingerprint
                target.last_report_seq = report_seq
                self._notify()
                return {"accepted": True, "duplicate": False, "cleanup_id": batch_id}
            batch = self._batch(group, batch_id)
            revoked_ready = (kind == "READY" and batch.status == "REVOKING"
                             and key not in batch.quiesced
                             and payload == {"reservation_id": batch.reservation.reservation_id})
            if group.fenced and batch.status != "FENCED" and not (
                    kind == "QUIESCED" and batch.status in ("QUIESCE", "REVOKING")) and not revoked_ready:
                raise IntegrationError("GROUP_FENCED", "Group is fenced")
            if stage != batch.plan.stage and kind not in ("READY", "HANDOFF_READY", "QUIESCED", "ERROR"):
                raise IntegrationError("INVALID_REPORT", "Wrong stage for GPU completion")
            if kind == "READY":
                if key not in batch.participants:
                    raise IntegrationError("STALE_REPORT", "Rank does not take part in this batch")
                if not revoked_ready and (batch.status != "PREPARING"
                        or payload != {"reservation_id": batch.reservation.reservation_id}):
                    raise IntegrationError("STALE_REPORT", "READY not for current preparation")
                # PREPARE can finish before the rank receives its queued REVOKE.
                # Consume that report's sequence; only all-rank QUIESCED may
                # release storage. Obsolete readiness never votes for START.
                if not revoked_ready:
                    batch.ready.add(key)
                    if batch.ready == batch.participants and batch.ready_at is None:
                        batch.ready_at = self.clock()
            elif kind == "GPU_DONE":
                if batch.status not in ("RUNNING", "FENCED"):
                    raise IntegrationError("STALE_REPORT", "GPU_DONE requires started batch")
                if "final_step" in payload:
                    payload = self._record_final_step(group, batch, rank, payload)
                self._gpu_done(group, batch, rank, payload)
            elif kind == "HANDOFF_READY":
                if stage != "D" or batch.plan.stage != "P" or batch.status != "HANDOFF":
                    raise IntegrationError("STALE_REPORT", "D handoff is not pending")
                self._handoff_ready(group, batch, rank, payload)
            elif kind == "QUIESCED":
                self._quiesced(group, batch, key, payload)
            else:
                if set(payload) != {"code"} or not isinstance(payload["code"], str) or not payload["code"]:
                    raise IntegrationError("INVALID_REPORT", "ERROR requires a nonempty code")
                self._fence(group, batch, "rank_error:" + str(payload.get("code", "unknown")))
            target.reports[report_seq] = fingerprint
            target.last_report_seq = report_seq
            self._notify()
            # A last READY, or credits freed by GPU_DONE/QUIESCED, can make a
            # preparation startable; START it now instead of a driver pass later.
            self._start_ready_inline()
            reply = {"accepted": True, "duplicate": False, "batch": self._batch_view(batch)}
            if revoked_ready:
                reply["obsolete"] = True
            return reply

    def broker_register_destination(self, group_id: str, room: int, rank: int, pid: int,
                                    model_id: str, request_id: str, attempt: int,
                                    allocation_id: str, reserved_version: int,
                                    target_valid_tokens: int, rank_block_table: list,
                                    metadata_slot: int, logical_token_count: int) -> dict:
        """Register native D logical preallocation before P bootstrap can finish.

        The controller already owns the complete atomic reservation. The native
        slot is a CPU ordinal, never a pointer or a transfer-completion proof.
        """
        trace.ev("brk", None, op="register_destination", gid=group_id, room=room, r=rank)
        with self.lock:
            group = self._group(group_id)
            target = group.registered.get(("D", rank))
            entry = group.broker.get(room)
            if (type(room) is not int or target is None or target.pid != pid
                    or not target.alive or entry is None or group.fenced):
                raise IntegrationError("INVALID_BROKER", "Unknown D rank/PID/bootstrap room")
            batch = self._batch(group, entry["batch_id"])
            view = next(v for v in batch.reservation.views if v.identity == entry["identity"])
            if (entry["identity"] !=
                    RequestIdentity(model_id, request_id, attempt)
                    or entry["allocation_id"] != allocation_id
                    or entry["reserved_version"] != reserved_version
                    or entry["target_valid_tokens"] != target_valid_tokens
                    or rank not in view.block_tables
                    or not _same_table(_packed_view(batch, view.identity.request_id)[rank], rank_block_table)
                    or type(metadata_slot) is not int or not 0 <= metadata_slot < 2 * self.config.max_model_inflight
                    or type(logical_token_count) is not int or logical_token_count != target_valid_tokens):
                raise IntegrationError("INVALID_BROKER", "Native destination reservation mismatch")
            destination = {"metadata_slot": metadata_slot, "logical_token_count": logical_token_count}
            previous = entry["destinations"].get(rank)
            if previous is not None and previous != destination:
                self._fence(group, batch, "bootstrap_destination_conflict")
                raise IntegrationError("IDENTITY_CONFLICT", "Native destination changed")
            if previous is not None:
                return {"room": room, "registered": True}
            if batch.status != "PREPARING":
                raise IntegrationError("INVALID_BROKER", "New native destination requires PREPARING")
            if any(other is not entry and other["state"] != "ABORTED"
                   and other["destinations"].get(rank, {}).get("metadata_slot") == metadata_slot
                   for other in group.broker.values()):
                raise IntegrationError("INVALID_BROKER", "Native metadata slot already belongs to another room")
            entry["destinations"][rank] = destination
            self._notify(group)
            return {"room": room, "registered": True}

    def broker_bootstrap(self, group_id: str, room: int, rank: int, pid: int,
                         allocation_id: str, reserved_version: int,
                         target_valid_tokens: int) -> dict:
        """Read all-D destination readiness; independent of P READY/STAGE."""
        trace.ev("brk", None, op="bootstrap", gid=group_id, room=room, r=rank)
        with self.lock:
            group = self._group(group_id)
            target = group.registered.get(("P", rank))
            entry = group.broker.get(room)
            if (type(room) is not int or target is None or target.pid != pid
                    or not target.alive or entry is None
                    or entry["allocation_id"] != allocation_id
                    or entry["reserved_version"] != reserved_version
                    or entry["target_valid_tokens"] != target_valid_tokens):
                raise IntegrationError("INVALID_BROKER", "Unknown P bootstrap identity")
            if group.fenced or entry["state"] == "ABORTED":
                return {"room": room, "status": "ABORTED"}
            return {"room": room, "status": "READY" if
                    set(entry["destinations"]) == set(group.ranks) else "PENDING"}

    def broker_stage(self, group_id: str, room: int, rank: int, pid: int,
                     metadata: dict) -> dict:
        """P sender.send stores CPU metadata privately; this never publishes it."""
        trace.ev("brk", None, op="stage", gid=group_id, room=room, r=rank)
        with self.lock:
            group = self._group(group_id)
            target = group.registered.get(("P", rank))
            entry = group.broker.get(room)
            if (type(room) is not int or not 0 < room <= _INT64_MAX
                    or target is None or target.pid != pid or not target.alive or entry is None):
                raise IntegrationError("INVALID_BROKER", "Unknown P rank, PID or room")
            batch = self._batch(group, entry["batch_id"])
            if (group.fenced or set(entry["destinations"]) != set(group.ranks)
                    or batch.status not in ("RUNNING", "HANDOFF") or entry["state"] not in ("PREPARING", "STAGED", "PUBLISHED")
                    or not isinstance(metadata, dict) or set(metadata) != {
                        "bootstrap_room", "allocation_id", "reserved_version",
                        "target_valid_tokens", "rank", "native_payload"}
                    or metadata["bootstrap_room"] != room or metadata["rank"] != rank
                    or metadata["allocation_id"] != entry["allocation_id"]
                    or metadata["reserved_version"] != entry["reserved_version"]
                    or metadata["target_valid_tokens"] != entry["target_valid_tokens"]
                    or not isinstance(metadata["native_payload"], dict)
                    or len(repr(metadata)) > 65536 or not self._plain_metadata(metadata)):
                raise IntegrationError("INVALID_BROKER", "Small plain CPU metadata required")
            previous = entry["metadata"].get(rank)
            if previous is not None and previous != metadata:
                self._fence(group, batch, "broker_content_conflict")
                raise IntegrationError("IDENTITY_CONFLICT", "Staged native metadata changed")
            entry["metadata"][rank] = deepcopy(metadata)
            if entry["state"] == "PREPARING":
                entry["state"] = "STAGED"
            self._notify(group)
            return {"room": room, "staged": True, "published": entry["state"] == "PUBLISHED"}

    @staticmethod
    def _plain_metadata(value: Any, depth: int = 0) -> bool:
        if depth > 5:
            return False
        if value is None or type(value) in (str, int, float, bool):
            return type(value) is not float or value == value and abs(value) < float("inf")
        if type(value) in (list, tuple):
            return len(value) <= 256 and all(MuxController._plain_metadata(v, depth + 1) for v in value)
        if type(value) is dict:
            return len(value) <= 256 and all(type(k) is str and
                MuxController._plain_metadata(v, depth + 1) for k, v in value.items())
        return False

    def broker_lookup(self, group_id: str, room: int, rank: int, pid: int,
                      model_id: str, request_id: str, attempt: int,
                      allocation_id: str, version: int, valid_tokens: int) -> dict:
        trace.ev("brk", None, op="lookup", gid=group_id, room=room, r=rank)
        with self.lock:
            group = self._group(group_id)
            target = group.registered.get(("D", rank))
            entry = group.broker.get(room)
            if (type(room) is not int or not 0 < room <= _INT64_MAX
                    or target is None or target.pid != pid or not target.alive or entry is None
                    or entry["state"] != "PUBLISHED" or group.fenced
                    or entry["identity"] != RequestIdentity(model_id, request_id, attempt)
                    or entry["allocation_id"] != allocation_id
                    or entry["version"] != version or entry["valid_tokens"] != valid_tokens
                    or rank not in entry["metadata"]):
                raise IntegrationError("HANDOFF_NOT_READY", "Committed room identity/version/length required")
            return {"group_id": group_id, "room": room,
                    "identity": entry["identity"], "allocation_id": allocation_id,
                    "version": version, "valid_tokens": valid_tokens,
                    "rank": rank, "rank_block_table": entry["block_tables"][rank],
                    "metadata": deepcopy(entry["metadata"][rank])}

    def broker_ack(self, group_id: str, room: int, rank: int, pid: int,
                   allocation_id: str, version: int, valid_tokens: int,
                   rank_block_table: list) -> dict:
        trace.ev("brk", None, op="ack", gid=group_id, room=room, r=rank)
        with self.lock:
            group = self._group(group_id)
            target = group.registered.get(("D", rank))
            entry = group.broker.get(room)
            if (type(room) is not int or not 0 < room <= _INT64_MAX
                    or target is None or target.pid != pid or not target.alive or entry is None
                    or entry["state"] != "PUBLISHED" or entry["allocation_id"] != allocation_id
                    or entry["version"] != version or entry["valid_tokens"] != valid_tokens
                    or rank not in entry["block_tables"]
                    or not _same_table(entry["block_tables"][rank], rank_block_table)):
                raise IntegrationError("HANDOFF_NOT_READY", "Committed broker ACK mismatch")
            entry["acked"].add(rank)
            self._notify(group)
            return {"room": room, "acked_ranks": tuple(sorted(entry["acked"]))}

    def broker_poll(self, group_id: str, room: int, rank: int, pid: int,
                    stage: str = "D") -> dict:
        """Native receiver may finish transfer once every D rank validated mapping."""
        trace.ev("brk", None, op="poll", gid=group_id, room=room, r=rank)
        with self.lock:
            group = self._group(group_id)
            target = group.registered.get((stage, rank))
            entry = group.broker.get(room)
            if stage not in _STAGES or target is None or target.pid != pid or not target.alive or entry is None:
                raise IntegrationError("INVALID_BROKER", "Unknown stage/rank/PID/room")
            if group.fenced or entry["state"] == "ABORTED":
                return {"room": room, "status": "ABORTED"}
            return {"room": room, "status": "SUCCESS" if entry["state"] == "PUBLISHED"
                    and entry["acked"] == set(group.ranks) else "PENDING"}

    def _validate_gpu_results(self, group: _Group, batch: _Batch, payload: dict,
                              rank: int | None = None) -> dict[str, dict]:
        keys = {"reservation_id", "results", "duration_ms", "gpu_forward_ms"}
        if batch.plan.steps > 1:
            keys = keys | {"steps"}
        if set(payload) != keys or payload["reservation_id"] != batch.reservation.reservation_id:
            raise IntegrationError("INVALID_REPORT", "GPU_DONE schema/reservation mismatch")
        steps_done = 1
        if batch.plan.steps > 1:
            steps_done = payload["steps"]
            reported = batch.run_reports.get(steps_done, {}) if type(steps_done) is int else {}
            # Slower ranks may still be reporting this step; full agreement is
            # rechecked once every rank's GPU_DONE has arrived.
            if (type(steps_done) is not int or not 1 <= steps_done <= batch.plan.steps
                    or (rank is not None and rank not in reported) or not reported
                    or (steps_done < batch.plan.steps and batch.run_decisions.get(steps_done) is not False)
                    or steps_done + 1 in batch.run_reports):
                raise IntegrationError("INVALID_REPORT", "Decode run end differs from its step reports")
        duration = payload["duration_ms"]
        if (type(duration) not in (int, float) or not math.isfinite(duration)
                or duration < 0 or duration > 3_600_000):
            raise IntegrationError("INVALID_REPORT", "Invalid synchronized duration")
        gpu_time = payload["gpu_forward_ms"]
        if (type(gpu_time) not in (int, float) or not math.isfinite(gpu_time)
                or not 0 <= gpu_time <= 3_600_000):
            raise IntegrationError("INVALID_REPORT", "Invalid forward CUDA event elapsed time")
        results = payload["results"]
        if not isinstance(results, (tuple, list)) or len(results) != len(batch.plan.requests):
            raise IntegrationError("INVALID_REPORT", "GPU results must preserve selected order")
        normalized = {}
        for ref, result in zip(batch.plan.requests, results):
            if not isinstance(result, dict) or set(result) != {"request_id", "attempt", "token_id", "valid_tokens"}:
                raise IntegrationError("INVALID_REPORT", "GPU result fields invalid")
            request = self._request(group.model_id, ref.request_id)
            expected = (len(request.spec.input_ids) + len(request.output) - 1
                        if batch.plan.purpose == "REBUILD" else
                        len(request.spec.input_ids) if batch.plan.stage == "P" else
                        batch.run_start_valid[ref.request_id] + steps_done if batch.plan.steps > 1
                        else request.valid_tokens + 1)
            token = result["token_id"]
            if (result["request_id"] != ref.request_id or result["attempt"] != ref.attempt
                    or type(result["valid_tokens"]) is not int or result["valid_tokens"] != expected
                    or (token is not None if batch.plan.purpose == "REBUILD" else
                        type(token) is not int or not 0 <= token < group.capability["vocab_size"])):
                raise IntegrationError("INVALID_REPORT", "GPU output/valid target mismatch")
            normalized[ref.request_id] = deepcopy(result)
        if batch.plan.steps > 1:
            agreed = next(iter(batch.run_reports[steps_done].values()))[0]
            if list(results) != list(agreed):
                raise IntegrationError("INVALID_REPORT", "Decode run final result differs from its last step")
        return normalized

    def _gpu_done(self, group: _Group, batch: _Batch, rank: int, payload: dict) -> None:
        results = self._validate_gpu_results(group, batch, payload, rank)
        old = batch.gpu.get(rank)
        if old is not None:
            if old != payload:
                self._fence(group, batch, "gpu_result_conflict")
                raise IntegrationError("IDENTITY_CONFLICT", "Rank GPU result changed")
            return
        if batch.gpu and next(iter(batch.gpu.values()))["results"] != payload["results"]:
            self._fence(group, batch, "cross_rank_token_conflict")
            raise IntegrationError("IDENTITY_CONFLICT", "Greedy ranks disagree")
        batch.gpu[rank] = deepcopy(payload)
        if len(batch.gpu) != len(group.ranks):
            return
        batch.gpu_completed_at = self.clock()
        trace.ev("gpu_all", batch.plan.batch_id, m=batch.plan.model_id, s=batch.plan.stage)
        batch.output_metadata = results
        if batch.plan.steps > 1:
            if len(batch.run_reports.get(payload["steps"], {})) != len(group.ranks):
                self._fence(group, batch, "decode_run_incomplete_step")
                raise IntegrationError("INVALID_REPORT", "Every rank must report the final run step")
            batch.run_steps_done = payload["steps"]
            batch.selected_input_tokens = batch.run_steps_done * len(batch.plan.requests)
        valid = {batch.views[ref.request_id].allocation_id: results[ref.request_id]["valid_tokens"]
                 for ref in batch.plan.requests}
        if batch.plan.stage == "P" and batch.status == "RUNNING":
            for entry in group.broker.values():
                if entry["batch_id"] == batch.plan.batch_id and set(entry["metadata"]) != set(group.ranks):
                    self._fence(group, batch, "missing_native_metadata")
                    raise IntegrationError("HANDOFF_NOT_READY", "All P ranks must stage native metadata")
        failed = bool(batch.failure_reason and batch.failure_reason.startswith("rank_error:"))
        try:
            committed = self.store.finish_kv_batch(batch.reservation.reservation_id,
                                                   "FAILURE" if failed else "SUCCESS",
                                                   {} if failed else valid, group.ranks)
        except IntegrationError as exc:
            self._fence(group, batch, "store_commit:" + exc.code)
            raise
        if failed:
            batch.rolled_back_blocks = dict(batch.reserved_blocks)
            batch.output_metadata = {}
            self._begin_quiesce(group, batch, batch.participants)
            return
        batch.committed_views = {view.identity.request_id: view for view in committed}
        for ref in batch.plan.requests:
            request = self._request(group.model_id, ref.request_id)
            view = batch.committed_views[ref.request_id]
            request.allocation_id, request.valid_tokens, request.version = (
                view.allocation_id, view.valid_tokens, view.version)
        if group.fenced or batch.status == "FENCED":
            if batch.plan.stage == "D":
                if batch.plan.steps > 1:
                    self._finish_run_tokens(group, batch)
                else:
                    self._apply_tokens(group, batch)
            self._begin_quiesce(group, batch, batch.participants)
        elif batch.plan.stage == "P":
            batch.status = "HANDOFF"
            # Credits meter SM share. Every P rank has finished its forward, so
            # the KV handoff and quiesce that follow use none: release now and
            # let yielded decode runs resume instead of waiting for the whole
            # P/D protocol (release in _finish_batch then finds no lease).
            self._release_credits(batch.plan.batch_id)
            for entry in group.broker.values():
                if entry["batch_id"] == batch.plan.batch_id:
                    request = self._request(group.model_id, entry["identity"].request_id)
                    entry["version"] = request.version
                    entry["valid_tokens"] = request.valid_tokens
                    entry["block_tables"] = _packed_view(batch, request.spec.request_id, committed=True)
                    entry["state"] = "PUBLISHED"
            self._notify(group)
            for rank_id in group.ranks:
                self._command(group, ("D", rank_id), "HANDOFF",
                              {**self._handoff_payload(group, batch, rank_id),
                               "_command_owner": batch.holder})
            if self.config.prefill_early_token and batch.plan.purpose != "REBUILD":
                # Every P rank agreed on this token and its KV is committed.
                # Publish it now; the request stays RUNNING (not decodable)
                # until every D rank has accepted the handoff.
                self._apply_tokens(group, batch, settle=False)
                batch.tokens_published = True
                if self.config.handoff_prepare_decode and batch.plan.purpose == "GENERATE":
                    self._prepare_after_handoff(group, batch)
        else:
            if batch.plan.steps > 1:
                if batch.run_applied != batch.run_steps_done:
                    self._fence(group, batch, "decode_run_unapplied_step")
                    raise IntegrationError("INVALID_REPORT", "Decode run committed an unagreed step")
                self._finish_run_tokens(group, batch)
            else:
                self._apply_tokens(group, batch)
            # Same as P above: every D rank finished its forwards, so the
            # quiesce round trip uses no SMs. Releasing now lets a waiting
            # prefill (which needs all credits) start one round trip sooner.
            self._release_credits(batch.plan.batch_id)
            self._begin_quiesce(group, batch, batch.participants)
            if self.config.cleanup_with_quiesce:
                # The request ended in this batch: queue its CLEANUP right
                # behind the QUIESCE rather than one more round trip later.
                for ref in batch.plan.requests:
                    request = self._request(group.model_id, ref.request_id)
                    if (request.attempt == ref.attempt and request.state == "FINISH_PENDING"
                            and request.finish_reason in ("eos", "length")
                            and request.pending_reason is None):
                        self._begin_cleanup(request, quiescing=batch)

    def _prepare_after_handoff(self, group: _Group, batch: _Batch) -> None:
        """Caller holds the lock; ``batch`` is a prefill whose HANDOFF commands
        were just queued. Prepare its requests' first decode batch now: its
        PREPARE follows the HANDOFF on every D rank, which executes commands in
        order (a HANDOFF holds back later commands until it completes), so
        each rank prepares right after its handoff and the last READY STARTs it
        inline. Any refusal leaves the ordinary driver path to prepare it."""
        if not self._inline_start_allowed or self.protected_failure is not None or group.fenced:
            return
        refs, remaining = [], []
        for ref in batch.plan.requests:
            request = self._request(group.model_id, ref.request_id)
            if (request.attempt != ref.attempt or request.finish_reason or request.pending_reason
                    or len(request.output) != ref.expected_output_count + 1):
                return
            refs.append(RequestRef(ref.request_id, ref.attempt, len(request.output)))
            remaining.append(request.spec.max_new_tokens - len(request.output))
        steps = max(1, min([self.config.decode_run_steps] + remaining))
        plan = BatchPlan(group.group_id, uuid4().hex, group.model_id, "D", tuple(refs),
                         purpose="GENERATE", prepare_timeout_ms=self.config.prepare_timeout_ms,
                         steps=steps)
        try:
            self.prepare_batch(plan, _after_handoff=batch)
        except IntegrationError as exc:
            trace.ev("prep_fail", None, m=group.model_id, s="D", code=exc.code, after_handoff=True)
            if exc.code not in ("OUT_OF_CAPACITY", "OUT_OF_CREDITS", "STAGE_BUSY", "NOT_READY",
                                "STALE_ATTEMPT", "EVENT_BACKPRESSURE", "COMMAND_BACKPRESSURE"):
                # Surfaced by the driver, exactly as its own prepare_batch
                # failure would have been (same contract as inline_start).
                self.inline_start_error = exc
                self._inline_start_allowed = False
                self.wake.set()

    def _record_final_step(self, group: _Group, batch: _Batch, rank: int, payload: dict) -> dict:
        """final_step_in_gpu_done: the rank ended its run locally (finished or
        last granted step), whose decision is always to stop; record that step
        exactly as its decode_step would have, then handle the GPU_DONE."""
        final = payload["final_step"]
        rest = {key: value for key, value in payload.items() if key != "final_step"}
        if (not self.config.final_step_in_gpu_done or batch.plan.stage != "D" or batch.plan.steps < 2
                or not isinstance(final, dict) or set(final) != {"step", "results", "finished"}
                or final["step"] != rest.get("steps") or final["results"] != rest.get("results")):
            raise IntegrationError("INVALID_REPORT", "GPU_DONE final step mismatch")
        trace.ev("dstep", batch.plan.batch_id, gid=group.group_id, r=rank, step=final["step"])
        decision = self._record_step(group, batch, rank, final["step"], final["results"], final["finished"])
        if decision is not False:
            self._fence(group, batch, "final_step_continues")
            raise IntegrationError("INVALID_REPORT", "A locally final decode step must stop the run")
        return rest

    def _finish_run_tokens(self, group: _Group, batch: _Batch) -> None:
        """Tokens were published per agreed step; only settle request states."""
        for ref in batch.plan.requests:
            request = self._request(group.model_id, ref.request_id)
            if request.state == "RUNNING":
                request.state = "FINISH_PENDING" if request.finish_reason else "DECODING"

    def decode_step(self, group_id: str, rank: int, pid: int, batch_id: str, step: int,
                    results: list, finished: bool) -> dict:
        """One D rank finished forward ``step`` of a multi-step grant.

        Every rank reports every step. Results must agree across ranks; a step
        is published once all ranks reported it. The continue decision is made
        once per step and replayed to every rank, so TP ranks never diverge on
        whether another forward follows.
        """
        trace.ev("dstep", batch_id, gid=group_id, r=rank, step=step)
        with self.lock:
            group = self._group(group_id)
            target = group.registered.get(("D", rank))
            if target is None or target.pid != pid or not target.alive:
                raise IntegrationError("UNKNOWN_RANK", "Registered D rank/PID required")
            batch = self._batch(group, batch_id)
            return {"continue": self._record_step(group, batch, rank, step, results, finished)}

    def _record_step(self, group: _Group, batch: _Batch, rank: int, step: int,
                     results, finished: bool) -> bool | str:
        """Caller holds the lock and verified the D rank. Returns the step's
        (once decided, replayed) continue decision."""
        batch_id = batch.plan.batch_id
        if (batch.plan.stage != "D" or batch.plan.steps < 2
                or batch.status not in ("RUNNING", "FENCED")):
            raise IntegrationError("STALE_REPORT", "No running decode run")
        if (type(step) is not int or not 1 <= step <= batch.plan.steps
                or type(finished) is not bool or not isinstance(results, (list, tuple))
                or len(results) != len(batch.plan.requests)):
            raise IntegrationError("INVALID_REPORT", "Invalid decode step report")
        entry = (tuple(deepcopy(dict(item)) if isinstance(item, dict) else item for item in results),
                 finished)
        # A rejected report must not create a step entry (GPU_DONE treats
        # any entry after the last step as a protocol error).
        reported = batch.run_reports.get(step, {})
        if rank in reported:
            if reported[rank] != entry:
                self._fence(group, batch, "decode_step_conflict")
                raise IntegrationError("IDENTITY_CONFLICT", "Decode step replay changed")
            return batch.run_decisions.get(step, False)
        if step > 1 and (rank not in batch.run_reports.get(step - 1, {})
                         or batch.run_decisions.get(step - 1) is not True):
            raise IntegrationError("REPORT_GAP", "Decode step was not granted")
        for ref, result in zip(batch.plan.requests, entry[0]):
            if (not isinstance(result, dict)
                    or set(result) != {"request_id", "attempt", "token_id", "valid_tokens"}
                    or result["request_id"] != ref.request_id or result["attempt"] != ref.attempt
                    or result["valid_tokens"] != batch.run_start_valid[ref.request_id] + step
                    or type(result["token_id"]) is not int
                    or not 0 <= result["token_id"] < group.capability["vocab_size"]):
                raise IntegrationError("INVALID_REPORT", "Decode step result mismatch")
        if reported and next(iter(reported.values())) != entry:
            self._fence(group, batch, "cross_rank_token_conflict")
            raise IntegrationError("IDENTITY_CONFLICT", "Greedy ranks disagree")
        reported[rank] = entry
        batch.run_reports[step] = reported
        batch.progress_at = self.clock()
        applied = False
        if (len(reported) == len(group.ranks) and batch.run_applied == step - 1
                and batch.status == "RUNNING"):
            self._apply_run_step(group, batch, step, entry[0])
            applied = True
        decision = batch.run_decisions.get(step)
        if decision is None:
            decision = self._run_continues(group, batch, step, finished)
            batch.run_decisions[step] = decision
        if (decision == "PARK" and len(reported) == len(group.ranks)
                and batch.parked_at is None):
            # Every rank finished (and synchronized) forward ``step``: the
            # run uses no SMs until it resumes, so its credits go to the
            # waiting prefill. KV reservation and native state stay.
            batch.parked_at = self.clock()
            trace.ev("park", batch_id, m=group.model_id, step=step)
            self._release_credits(batch_id)
            self.wake.set()
            # The last run to park frees the credits the READY prefill
            # waits for: START it now (inline_start), not a driver pass later.
            self._start_ready_inline()
        # A step report changes no command, liveness or broker epoch, so
        # the ranks' long polls need no wake-up (notify_all here woke every
        # waiting poll thread four times per token). Wake only the driver,
        # once per published step or when the run stops.
        if applied or decision is False:
            self.wake.set()
        return decision

    def decode_resume(self, group_id: str, rank: int, pid: int, batch_id: str, step: int,
                      wait_ms: int | float = 100) -> dict:
        """Long-poll of a D rank whose run parked after ``step``.

        Returns PARK (poll again), True (run step + 1) or False (end the run
        exactly as an ordinary stop: GPU_DONE with ``step`` as the last step).
        The outcome is decided once and replayed to every rank.
        """
        if type(wait_ms) not in (int, float) or not 0 <= wait_ms <= 1000:
            raise IntegrationError("INVALID_ARGUMENT", "Resume wait must be 0..1000 ms")
        deadline = time.monotonic() + wait_ms / 1000
        with self.lock:
            group = self._group(group_id)
            target = group.registered.get(("D", rank))
            if target is None or target.pid != pid or not target.alive:
                raise IntegrationError("UNKNOWN_RANK", "Registered D rank/PID required")
            batch = self._batch(group, batch_id)
            if (batch.plan.stage != "D" or batch.plan.steps < 2 or type(step) is not int
                    or rank not in batch.run_reports.get(step, {})
                    or batch.run_decisions.get(step) not in ("PARK", True, False)):
                raise IntegrationError("STALE_REPORT", "No parked decode step")
            while True:
                decision = batch.run_decisions[step]
                if decision != "PARK":
                    return {"continue": decision}
                decision = self._park_outcome(group, batch, step)
                if decision != "PARK":
                    batch.run_decisions[step] = decision
                    trace.ev("unpark", batch_id, m=group.model_id, step=step, go=decision)
                    if decision is True:
                        batch.progress_at = self.clock()
                    batch.parked_at = None
                    self._park_cv.notify_all()
                    self.wake.set()
                    return {"continue": decision}
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return {"continue": "PARK"}
                self._park_cv.wait(remaining)

    def _park_outcome(self, group: _Group, batch: _Batch, step: int) -> bool | str:
        """Caller holds the lock. Stop for any reason an ordinary run stops;
        resume once no prefill holds or awaits credits and ours are free."""
        if not self._run_continues(group, batch, step, False, park=False):
            return False
        if batch.parked_at is None:
            return "PARK"  # Slower ranks have not finished this step yet.
        if self.clock() - batch.parked_at >= _PARK_LIMIT_S:
            return False  # Fall back to an ordinary yield (end, re-prepare later).
        if self._prefill_pending():
            return "PARK"
        credits = self.config.model(batch.plan.model_id).decode_mps // 10
        if not self.credits.acquire(batch.plan.batch_id, batch.plan.model_id, "D", credits):
            return "PARK"
        return True

    def _apply_run_step(self, group: _Group, batch: _Batch, step: int, results) -> None:
        batch.run_applied = step
        for ref, result in zip(batch.plan.requests, results):
            request = self._request(group.model_id, ref.request_id)
            published = batch.run_published[ref.request_id]
            if (request.attempt != ref.attempt or len(request.output) !=
                    batch.run_start_output[ref.request_id] + published):
                self._fence(group, batch, "output_position_changed")
                raise IntegrationError("STALE_ATTEMPT", "Output position changed during GPU work")
            if self.clock() >= request.deadline and request.pending_reason in (None, "suspend"):
                request.pending_reason, request.state = "timeout", "STOP_PENDING"
            if (group.fenced or request.finish_reason
                    or request.pending_reason not in (None, "suspend")):
                continue
            token = result["token_id"]
            finish = ("eos" if token in group.capability["eos_token_ids"] and not
                      request.spec.sampling.get("ignore_eos", False) else
                      "length" if len(request.output) + 1 >= request.spec.max_new_tokens else None)
            group.outbox.emit("TOKEN", request_id=request.spec.request_id,
                              attempt=request.attempt, batch_id=batch.plan.batch_id,
                              payload={"token_ids": (token,), "output_index": len(request.output),
                                       "finish_reason": finish}, owner=batch.holder)
            request.output.append(token)
            batch.run_published[ref.request_id] = published + 1
            if request.first_token_at is None:
                request.first_token_at = self.clock()
            request.finish_reason = finish

    def _run_continues(self, group: _Group, batch: _Batch, step: int, finished: bool,
                       park: bool = True) -> bool | str:
        if (finished or step >= batch.plan.steps or batch.status != "RUNNING"
                or group.fenced or group.draining or self.protected_failure is not None):
            return False
        now = self.clock()
        for ref in batch.plan.requests:
            request = self._request(group.model_id, ref.request_id)
            if (request.attempt != ref.attempt or request.pending_reason or request.finish_reason
                    or now >= request.deadline):
                return False
        protected = self.recovery_protected.get(self._recovery_key(group.model_id))
        if protected is not None and {(group.model_id, ref.request_id)
                                      for ref in batch.plan.requests} != {protected}:
            return False
        if not park:
            return True
        # Source ADBS gives a waiting prefill priority: per-token grants stop
        # new D batches until P can take its credits. A run yields the same
        # way: it ends, or with decode_run_park it parks until P's forward ends.
        if not self._prefill_waiting_for_credits():
            return True
        return "PARK" if self.config.decode_run_park else False

    def _prefill_waiting_for_credits(self) -> bool:
        if self.config.decode_run_park:
            # Prefills prepare alongside decoding; runs yield only to one
            # that is READY and needs the credits to START.
            return self._prefill_awaiting_credits()
        if self.credits.can_allocate(10):
            return False
        return self._prefill_eligible()

    def _prefill_pending(self) -> bool:
        """A prefill holds credits (before its forward finished) or is READY
        and waiting for them: parked runs stay parked, even while credits
        are free."""
        if any(lease[0][1] == "P" for lease in self.credits.leases.values()):
            return True
        return self._prefill_awaiting_credits()

    def _prefill_awaiting_credits(self) -> bool:
        for group in self.groups.values():
            for batch in group.batches.values():
                if (batch.plan.stage == "P" and batch.status == "PREPARING"
                        and batch.ready == batch.participants
                        and batch.plan.batch_id not in self.credits.leases):
                    return True
        return False

    def _prefill_eligible(self) -> bool:
        for model in self.config.models:
            group = self.groups[model.model_id]
            if not group.ready or model.model_id in self.capacity_blocked:
                continue
            if any(b.status not in ("DONE", "ABORTED") and b.plan.stage == "P"
                   for b in group.batches.values()):
                continue
            rows = [r for r in self.requests.values() if r.spec.model_id == model.model_id]
            if not any(r.state in ("QUEUED", "REBUILD") and not r.pending_reason for r in rows):
                continue
            occupied = sum(self._occupies_slot(r) for r in rows)
            if occupied < min(self.config.max_model_inflight, self.policy.limits[model.model_id]):
                return True
        return False

    def _handoff_payload(self, group: _Group, batch: _Batch, rank: int) -> dict:
        items = []
        for ref in batch.plan.requests:
            request = self._request(group.model_id, ref.request_id)
            room = next(number for number, entry in group.broker.items()
                        if entry["batch_id"] == batch.plan.batch_id and
                        entry["identity"] == request.identity)
            extra = {}
            if self.config.handoff_inline_metadata:
                # What broker_lookup would return: this rank's staged P
                # metadata of the PUBLISHED room (checked at all-rank GPU_DONE).
                extra["native_metadata"] = deepcopy(group.broker[room]["metadata"][rank]["native_payload"])
            items.append({**extra, "request_id": ref.request_id, "attempt": ref.attempt,
                          "bootstrap_room": room, "allocation_id": request.allocation_id,
                          "version": request.version, "valid_tokens": request.valid_tokens,
                          "rank_block_table": _packed_view(batch, ref.request_id, committed=True)[rank],
                          "boundary_token_id": request.output[-1] if batch.plan.purpose == "REBUILD"
                              else batch.output_metadata[ref.request_id]["token_id"]})
        return {"batch_id": batch.plan.batch_id, "reservation_id": batch.reservation.reservation_id,
                "rank": rank, "requests": items,
                "handoff_deadline": batch.gpu_completed_at + self.config.prepare_timeout_ms / 1000}

    def _handoff_ready(self, group: _Group, batch: _Batch, rank: int, payload: dict) -> None:
        if set(payload) != {"requests"} or not isinstance(payload["requests"], (tuple, list)):
            raise IntegrationError("INVALID_REPORT", "HANDOFF_READY request list required")
        expected = self._handoff_payload(group, batch, rank)["requests"]
        if payload["requests"] != expected:
            self._fence(group, batch, "handoff_mismatch")
            raise IntegrationError("IDENTITY_CONFLICT", "D validated mapping differs")
        for item in expected:
            entry = group.broker[item["bootstrap_room"]]
            if entry["acked"] != set(group.ranks):
                raise IntegrationError("HANDOFF_NOT_READY", "All D broker ACKs precede native success")
        batch.handoff.add(rank)
        if len(batch.handoff) == len(group.ranks):
            if batch.gpu_completed_at is not None:
                batch.pd_handoff_ms = max(0.0, (self.clock() - batch.gpu_completed_at) * 1000)
            trace.ev("handoff_all", batch.plan.batch_id, m=batch.plan.model_id)
            if batch.tokens_published:
                self._settle_published_tokens(group, batch)
            elif batch.plan.purpose != "REBUILD":
                self._apply_tokens(group, batch)
            else:
                for ref in batch.plan.requests:
                    request = self._request(group.model_id, ref.request_id)
                    request.state = "DECODING"
            self._begin_quiesce(group, batch, {( "P", r) for r in group.ranks})

    def _settle_published_tokens(self, group: _Group, batch: _Batch) -> None:
        """prefill_early_token: tokens went out at commit; the handoff now
        makes exactly the requests _apply_tokens would have settled decodable."""
        for ref in batch.plan.requests:
            request = self._request(group.model_id, ref.request_id)
            if request.attempt != ref.attempt:
                self._fence(group, batch, "output_position_changed")
                raise IntegrationError("STALE_ATTEMPT", "Attempt changed during handoff")
            if (group.fenced or request.state != "RUNNING"
                    or request.pending_reason not in (None, "suspend")
                    or len(request.output) != ref.expected_output_count + 1):
                continue  # not published, or stopped since: _finish_batch decides
            request.state = "FINISH_PENDING" if request.finish_reason else "DECODING"

    def _apply_tokens(self, group: _Group, batch: _Batch, settle: bool = True) -> None:
        """Emit only indexed, all-rank agreed output; reserved slots precede mutation.

        settle=False (prefill_early_token) publishes without making the
        request decodable; _settle_published_tokens does that at handoff."""
        for ref in batch.plan.requests:
            request = self._request(group.model_id, ref.request_id)
            if request.attempt != ref.attempt or len(request.output) != ref.expected_output_count:
                self._fence(group, batch, "output_position_changed")
                raise IntegrationError("STALE_ATTEMPT", "Output position changed during GPU work")
            # All-rank completion makes KV reclaimable only after quiescence;
            # it does not authorize publishing results after a stop/deadline.
            # Suspension may commit this safe-point token before checkpointing.
            if self.clock() >= request.deadline and request.pending_reason in (None, "suspend"):
                request.pending_reason, request.state = "timeout", "STOP_PENDING"
            if group.fenced or request.pending_reason not in (None, "suspend"):
                continue
            token = batch.output_metadata[ref.request_id]["token_id"]
            if token is None:
                continue
            finish = ("eos" if token in group.capability["eos_token_ids"] and not
                      request.spec.sampling.get("ignore_eos", False) else
                      "length" if len(request.output) + 1 >= request.spec.max_new_tokens else None)
            group.outbox.emit("TOKEN", request_id=request.spec.request_id,
                              attempt=request.attempt, batch_id=batch.plan.batch_id,
                              payload={"token_ids": (token,), "output_index": len(request.output),
                                       "finish_reason": finish}, owner=batch.holder)
            request.output.append(token)
            if request.first_token_at is None:
                request.first_token_at = self.clock()
            request.finish_reason = finish
            if settle:
                request.state = "FINISH_PENDING" if finish else "DECODING"

    def _begin_quiesce(self, group: _Group, batch: _Batch,
                       keys: set[tuple[str, int]]) -> None:
        batch.status = "QUIESCE"
        batch.quiesce_deadline = self.clock() + self.config.prepare_timeout_ms / 1000
        batch.quiesced.clear()
        for key in sorted(keys):
            self._command(group, key, "QUIESCE", {"batch_id": batch.plan.batch_id,
                "reservation_id": batch.reservation.reservation_id,
                "request_ids": tuple(ref.request_id for ref in batch.plan.requests),
                "release_native": key[0] == batch.plan.stage,
                "_command_owner": batch.holder})
        batch.quiesce_required = keys

    def _quiesced(self, group: _Group, batch: _Batch,
                  key: tuple[str, int], payload: dict) -> None:
        if batch.status not in ("QUIESCE", "REVOKING") or key not in batch.quiesce_required:
            raise IntegrationError("STALE_REPORT", "No matching quiescence request")
        expected = {"reservation_id": batch.reservation.reservation_id,
                    "request_ids": tuple(ref.request_id for ref in batch.plan.requests)}
        if payload != expected:
            raise IntegrationError("INVALID_REPORT", "QUIESCED identity mismatch")
        batch.quiesced.add(key)
        if batch.quiesced == batch.quiesce_required:
            if batch.status == "REVOKING":
                self._finish_revoke(group, batch)
            else:
                self._finish_batch(group, batch)

    def _finish_batch(self, group: _Group, batch: _Batch) -> None:
        trace.ev("finish", batch.plan.batch_id, m=batch.plan.model_id, s=batch.plan.stage)
        try:
            before = {"ranks": self.store.rank_usage()}
        except IntegrationError as exc:
            self._fence(group, batch, "release_snapshot_uncertain:" + exc.code)
            raise
        batch.free_before_feedback = before
        try:
            for ref in batch.plan.requests:
                self.store.release_kv(batch.views[ref.request_id].allocation_id, batch.holder)
        except IntegrationError as exc:
            self._fence(group, batch, "release_store_uncertain:" + exc.code)
            raise
        for ref in batch.plan.requests:
            request = self._request(group.model_id, ref.request_id)
            if any(other is not batch and other.status not in ("DONE", "ABORTED")
                   and any(r.request_id == ref.request_id and r.attempt == request.attempt
                           for r in other.plan.requests)
                   for other in group.batches.values()):
                # handoff_prepare_decode: the request's first decode batch is
                # already prepared or running; that batch owns its state now.
                continue
            if request.pending_reason or request.finish_reason:
                request.state = "CLEANUP_PENDING"
            elif batch.plan.purpose == "REBUILD":
                request.state = "DECODING"
            elif batch.plan.stage == "P":
                request.state = "DECODING"
        batch.status = "DONE"
        if batch.plan.stage == "P" and all(("P", rank) in batch.quiesce_required for rank in group.ranks):
            # Every P rank executed this batch's QUIESCE, which releases each
            # of its requests' native state on that rank.
            for ref in batch.plan.requests:
                request = self._request(group.model_id, ref.request_id)
                if request.attempt == ref.attempt:
                    request.p_released = True
        if batch.plan.stage == "P":
            for room in tuple(group.broker):
                if group.broker[room]["batch_id"] == batch.plan.batch_id:
                    del group.broker[room]
        self._release_credits(batch.plan.batch_id)
        group.outbox.release(batch.holder)
        self._release_commands(group, batch.holder)
        for ref in batch.plan.requests:
            request = self._request(group.model_id, ref.request_id)
            if request.state == "CLEANUP_PENDING":
                self._begin_cleanup(request)
        self._maybe_feedback(group, batch)

    def _maybe_feedback(self, group: _Group, batch: _Batch) -> None:
        if batch.feedback_sent or batch.status != "DONE" or any(
                self._request(group.model_id, ref.request_id).state == "CLEANUP_PENDING"
                for ref in batch.plan.requests):
            return
        try:
            after = {"ranks": self.store.rank_usage()}
        except IntegrationError:
            return  # DONE remains durable; tick retries feedback after owner recovers.
        self._feedback(group, batch, not bool(batch.failure_reason), batch.failure_reason,
                       batch.free_before_feedback, after)

    def _feedback(self, group: _Group, batch: _Batch, succeeded: bool,
                  reason: str | None, before: dict | None = None,
                  after: dict | None = None) -> None:
        if batch.feedback_sent:
            return
        # Only an all-rank completed batch feeds ADBS history. Prepared aborts
        # never count as execution feedback.
        if batch.started_at is None:
            return
        duration = max((item["duration_ms"] for item in batch.gpu.values()), default=0) / 1000
        released = {r: max(0, after["ranks"][r]["free_blocks"] - before["ranks"][r]["free_blocks"])
                    for r in before["ranks"]} if before and after else {}
        for rank, count in batch.rolled_back_blocks.items():
            released[rank] = released.get(rank, 0) + count
        item = BatchFeedback(group.model_id, batch.plan.stage, len(batch.plan.requests),
                             self.policy.limits[group.model_id], duration,
                             sum(r.state == "QUEUED" for r in self.requests.values()
                                 if r.spec.model_id == group.model_id), succeeded,
                             input_tokens=batch.selected_input_tokens,
                             output_tokens=(batch.run_steps_done or 1) * sum(1 for ref in batch.plan.requests if
                                               batch.output_metadata.get(ref.request_id, {}).get("token_id") is not None),
                             purpose=batch.plan.purpose,
                             pd_handoff_ms=batch.pd_handoff_ms,
                             prepare_handshake_ms=((batch.ready_at - batch.prepared_at) * 1000
                                 if batch.ready_at is not None else None),
                             gpu_forward_ms=max((item["gpu_forward_ms"] for item in batch.gpu.values()),
                                                default=None),
                             reserved_blocks_by_rank=getattr(batch, "reserved_blocks", {}),
                             released_blocks_by_rank=released, failure_reason=reason)
        self.policy.feedback(item)
        batch.feedback_sent = True

    def _stop_request(self, request: _Request, reason: str) -> dict:
        if request.state in ("FINISHED", "CANCELLED", "TIMED_OUT", "FAILED"):
            return self._output(request)
        if request.pending_reason and request.pending_reason != reason:
            raise IntegrationError("IDENTITY_CONFLICT", "Conflicting pending stop reason")
        request.pending_reason = reason
        active = self._active(request)
        group = self._model(request.spec.model_id)
        for other in tuple(group.batches.values()):
            # handoff_prepare_decode: a decode batch prepared behind the
            # request's still-active prefill is revoked as well.
            if (other is not active and other.status == "PREPARING" and any(
                    ref.request_id == request.spec.request_id and ref.attempt == request.attempt
                    for ref in other.plan.requests)):
                self._abort_prepared(group, other, reason)
        if active is not None:
            if active.status == "PREPARING":
                self._abort_prepared(self._model(request.spec.model_id), active, reason)
                request.state = "STOP_PENDING"
                return self._output(request)
            request.state = "STOP_PENDING"
        else:
            request.state = "CLEANUP_PENDING"
            self._begin_cleanup(request)
        return self._output(request)

    def cancel_request(self, model_id: str, request_id: str, reason: str = "cancelled") -> dict:
        with self.lock:
            if not isinstance(reason, str) or not reason:
                raise IntegrationError("INVALID_ARGUMENT", "Cancellation reason required")
            result = deepcopy(self._stop_request(self._request(model_id, request_id), "cancel:" + reason))
            self._notify()
            return result

    def suspend_request(self, model_id: str, request_id: str, attempt: int) -> dict:
        with self.lock:
            request = self._request(model_id, request_id)
            if request.attempt != attempt:
                raise IntegrationError("STALE_ATTEMPT", "Suspension attempt differs")
            if request.state == "SUSPENDED":
                return {"status": "SUSPENDED", "record": deepcopy(self.records[request.resume_record_id])}
            if request.pending_reason and request.pending_reason != "suspend":
                raise IntegrationError("INVALID_STATE", "Request is already stopping")
            result = deepcopy(self._stop_request(request, "suspend"))
            self._notify()
            return result

    def _cleanup_participants(self, group: _Group, request: _Request) -> frozenset:
        if self.config.cleanup_skip_prefill_ranks and request.p_released:
            return frozenset(("D", rank) for rank in group.ranks)
        return frozenset(group.required)

    def _begin_cleanup(self, request: _Request, quiescing: _Batch | None = None) -> None:
        """``quiescing``: the request's last batch, whose QUIESCE was just
        queued; its CLEANUP may then follow on the same ranks right away."""
        group = self._model(request.spec.model_id)
        if any(item.identity == request.identity for item in group.cleanups.values()):
            return
        participants = self._cleanup_participants(group, request)
        active = self._active(request)
        if quiescing is not None and (active is not quiescing or quiescing.status != "QUIESCE"
                                      or not quiescing.quiesce_required <= participants):
            return  # _finish_batch issues the ordinary cleanup later
        if (group.uncertain_batches or (active is not None and quiescing is None) or
                any(not group.registered.get(key) or
                    not group.registered[key].alive for key in participants)):
            return
        if any(len(group.registered[key].commands) +
               sum(group.command_reserved.get(key, {}).values()) >= _COMMAND_WINDOW
               for key in participants):
            return
        cleanup_id = "cleanup:" + uuid4().hex
        owner = cleanup_id
        try:
            group.outbox.reserve(owner, 1)
        except IntegrationError as exc:
            if exc.code == "EVENT_BACKPRESSURE":
                return
            raise
        reason = request.pending_reason or request.finish_reason or "finished"
        cleanup = _Cleanup(cleanup_id, request.identity, reason,
                           self.clock() + self.config.prepare_timeout_ms / 1000,
                           participants=participants)
        group.cleanups[cleanup_id] = cleanup
        request.state = "CLEANUP_PENDING"
        for key in sorted(participants):
            self._command(group, key, "CLEANUP", {"cleanup_id": cleanup_id,
                "request_id": request.spec.request_id, "attempt": request.attempt,
                "allocation_id": request.allocation_id,
                "release_native": True, "reason": reason})

    def _cleanup_quiesced(self, group: _Group, cleanup: _Cleanup,
                          key: tuple[str, int], payload: dict) -> None:
        if key not in cleanup.participants or payload != {
                "cleanup_id": cleanup.cleanup_id,
                "request_id": cleanup.identity.request_id,
                "attempt": cleanup.identity.attempt}:
            raise IntegrationError("INVALID_REPORT", "Cleanup QUIESCED identity mismatch")
        cleanup.quiesced.add(key)
        if cleanup.quiesced != cleanup.participants:
            return
        request = self._request(cleanup.identity.model_id, cleanup.identity.request_id)
        if request.identity != cleanup.identity:
            raise IntegrationError("STALE_ATTEMPT", "Cleanup belongs to old attempt")
        if self._active(request) is not None:
            # Ranks report in command order, so a batch QUIESCE queued before
            # this CLEANUP has completed on every participant by now.
            self._fence(group, self._active(request), "cleanup_before_quiesce")
            raise IntegrationError("INVALID_REPORT", "Cleanup completed before its batch quiesced")
        if request.allocation_id:
            try:
                self.store.release_kv(request.allocation_id, self._d_holder(request))
            except IntegrationError as exc:
                self._fence(group, None, "cleanup_store_uncertain:" + exc.code)
                raise
            request.allocation_id = None
            request.valid_tokens = 0
            request.version = 0
        now = self.clock()
        reason = cleanup.reason
        if reason != "suspend" and request.resume_record_id:
            self.records.pop(request.resume_record_id, None)
            request.resume_record_id = None
        if reason == "suspend" and request.finish_reason in ("eos", "length"):
            request.state = "FINISHED"
            event_kind = "TERMINAL"
        elif reason == "suspend":
            if request.resume_record_id:
                self.records.pop(request.resume_record_id, None)
            record = ResumeRecord(uuid4().hex, request.identity, request.spec.input_ids,
                                  tuple(request.output), request.spec.max_new_tokens,
                                  deepcopy(request.spec.sampling), None, request.deadline,
                                  self.config.fingerprint)
            self.records[record.resume_record_id] = record
            request.resume_record_id = record.resume_record_id
            request.state = "SUSPENDED"
            event_kind = "SUSPENDED"
        elif reason.startswith("cancel:"):
            request.state, request.finish_reason = "CANCELLED", "cancelled"
            event_kind = "TERMINAL"
        elif reason == "timeout":
            request.state, request.finish_reason = "TIMED_OUT", "timeout"
            event_kind = "TERMINAL"
        elif request.finish_reason:
            request.state = "FINISHED"
            event_kind = "TERMINAL"
        else:
            request.state, request.finish_reason = "FAILED", reason
            event_kind = "TERMINAL"
        request.pending_reason = None
        request.finished_at = now
        group.outbox.emit(event_kind, request_id=request.spec.request_id,
            attempt=request.attempt, payload={"output_token_ids": tuple(request.output),
                "output_count": len(request.output), "finish_reason": request.finish_reason,
                "resume_record_id": request.resume_record_id if event_kind == "SUSPENDED"
                    else None}, owner=cleanup.cleanup_id)
        group.outbox.release(cleanup.cleanup_id)
        del group.cleanups[cleanup.cleanup_id]
        for batch in group.batches.values():
            if batch.status == "DONE" and any(ref.request_id == request.spec.request_id
                                              for ref in batch.plan.requests):
                self._maybe_feedback(group, batch)

    def resume_request(self, model_id: str, request_id: str,
                       resume_record_id: str) -> dict:
        with self.lock:
            self._require_admission()
            request = self._request(model_id, request_id)
            if (request.resume_record_id == resume_record_id and request.state != "SUSPENDED"
                    and request.attempt > 0):
                # The consumed record is retired; only its immediately resumed
                # attempt retains idempotent replay of this exact operation.
                if request.attempt == request.resumed_attempt:
                    return deepcopy(self._output(request))
            record = self.records.get(resume_record_id)
            if record is None or record.identity.model_id != model_id or record.identity.request_id != request_id:
                raise IntegrationError("UNKNOWN_RECORD", "Resume record identity mismatch")
            if request.attempt == record.identity.attempt + 1 and request.resume_record_id == resume_record_id:
                return deepcopy(self._output(request))
            if (request.state != "SUSPENDED" or request.resume_record_id != resume_record_id
                    or request.attempt != record.identity.attempt
                    or record.config_fingerprint != self.config.fingerprint
                    or self.clock() >= record.deadline or not self._model(model_id).ready):
                raise IntegrationError("INVALID_STATE", "Immutable record cannot resume now")
            request.attempt += 1
            request.resumed_attempt = request.attempt
            request.p_released = False
            self.records.pop(resume_record_id, None)
            request.output = list(record.output_token_ids)
            request.deadline = record.deadline
            request.allocation_id = None
            request.valid_tokens = 0
            request.version = 0
            request.finish_reason = None
            request.finished_at = None
            request.pending_reason = None
            request.state = "REBUILD" if request.output else "QUEUED"
            self._notify()
            return deepcopy(self._output(request))

    def begin_drain(self, group_id: str) -> dict:
        with self.lock:
            group = self._group(group_id)
            group.draining = True
            self._issue_drain(group)
            return self.close_ready(group_id)

    def _issue_drain(self, group: _Group) -> None:
        if (group.drain_issued or group.fenced or group.uncertain_batches or
                group.required != set(group.registered)):
            return
        if (any(b.status not in ("DONE", "ABORTED") for b in group.batches.values())
                or group.cleanups or any(r.state not in ("FINISHED", "CANCELLED", "TIMED_OUT",
                    "SUSPENDED", "FAILED") for r in self.requests.values()
                    if r.spec.model_id == group.model_id)
                or any(len(rank.commands) >= _COMMAND_WINDOW for rank in group.registered.values())):
            return
        for key in sorted(group.required):
            self._command(group, key, "DRAIN", {"reason": "graceful_close"})
        group.drain_issued = True

    def close_ready(self, group_id: str) -> dict:
        with self.lock:
            group = self._group(group_id)
            store = self.store.snapshot()
            reasons = []
            if not group.draining:
                reasons.append("drain_not_requested")
            if not group.drain_issued:
                reasons.append("drain_not_issued")
            if group.fenced or store["fences"]:
                reasons.append("fenced_resources")
            if group.uncertain_batches:
                reasons.append("uncertain_reservations")
            if any(b.status not in ("DONE", "ABORTED") for b in group.batches.values()):
                reasons.append("active_batches")
            if group.cleanups:
                reasons.append("pending_cleanup")
            if any(r.state not in ("FINISHED", "CANCELLED", "TIMED_OUT", "SUSPENDED", "FAILED")
                   for r in self.requests.values() if r.spec.model_id == group.model_id):
                reasons.append("active_requests")
            if any(a["identity"].model_id == group.model_id for a in store["allocations"].values()):
                reasons.append("held_kv")
            if any(rank.commands for rank in group.registered.values()):
                reasons.append("unacked_commands")
            if group.outbox.pending:
                reasons.append("unacked_events")
            safe_to_exit = not reasons
            if any(rank.exit_code is None for rank in group.registered.values()):
                reasons.append("rank_processes_alive")
            return {"group_id": group_id, "ready": not reasons,
                    "safe_to_exit": safe_to_exit, "reasons": tuple(reasons),
                    "rank_pids": {(s, r): value.pid for (s, r), value in group.registered.items()}}

    def forget_request(self, model_id: str, request_id: str, attempt: int) -> dict:
        """Gateway-acknowledged retention release after terminal output/event ACK."""
        with self.lock:
            group = self._model(model_id)
            request = self.requests.get((model_id, request_id))
            if request is None:
                if group.retired.contains("request", request_id):
                    return {"forgotten": True}
                raise IntegrationError("UNKNOWN_REQUEST", "Unknown request")
            if (group.uncertain_batches or request.attempt != attempt or request.state not in (
                    "FINISHED", "CANCELLED", "TIMED_OUT", "FAILED", "SUSPENDED")
                    or self._active(request) is not None or any(
                        event.request_id == request_id for event in group.outbox.pending.values())
                    or any(self._command_refs_request(group, cmd, request_id)
                           for rank in group.registered.values() for cmd in rank.commands.values())
                    or request.identity in self.store.allocation_identities()):
                raise IntegrationError("NOT_READY", "Terminal, event/command ACK and physical release required")
            group.retired.add("request", request_id)
            if request.resume_record_id:
                self.records.pop(request.resume_record_id, None)
            del self.requests[(model_id, request_id)]
            self._retire_completed(group)
            return {"forgotten": True}

    @staticmethod
    def _command_refs_request(group: _Group, command: dict | CommandSnapshot, request_id: str) -> bool:
        if isinstance(command, CommandSnapshot):
            if command.references(request_id): return True
            batch = group.batches.get(command.batch_id)
            return bool(batch and any(ref.request_id == request_id for ref in batch.plan.requests))
        if command.get("request_id") == request_id or request_id in command.get("request_ids", ()):
            return True
        if any(item.get("request_id") == request_id for item in command.get("requests", ())):
            return True
        batch = group.batches.get(command.get("batch_id"))
        return bool(batch and any(ref.request_id == request_id for ref in batch.plan.requests))

    def _retire_completed(self, group: _Group) -> None:
        allocations = self.store.allocation_ids()
        for batch_id, batch in tuple(group.batches.items()):
            if (batch.status not in ("DONE", "ABORTED")
                    or any((group.model_id, ref.request_id) in self.requests for ref in batch.plan.requests)
                    or any(command.get("batch_id") == batch_id for rank in group.registered.values()
                           for command in rank.commands.values())
                    or any(event.batch_id == batch_id for event in group.outbox.pending.values())
                    or any(view.allocation_id in allocations for view in batch.reservation.views)):
                continue
            self.store.retire_history((batch_id,))
            group.retired.add("batch", batch_id)
            del group.batches[batch_id]

    def batch_progress(self) -> dict:
        """Detached READY/lifecycle state for the event pump, without diagnostics."""
        with self.lock:
            return {model_id: {bid: self._batch_view(batch) for bid, batch in group.batches.items()}
                    for model_id, group in self.groups.items()}

    def snapshot(self) -> dict:
        with self.lock:
            return {"protected_failure": self.protected_failure,
                "initializing": self.initializing,
                "capacity_blocked": tuple(sorted(self.capacity_blocked)),
                "recovery_protected": dict(self.recovery_protected),
                "recovery_stats": {"cycles": self.recovery_cycles,
                                   "automatic_suspensions": self.recovery_suspensions},
                "protected_fence_errors": dict(self.protected_fence_errors),
                "groups": {m: {"group_id": g.group_id, "ready": g.ready,
                "draining": g.draining, "fenced": g.fenced,
                "uncertain_batches": tuple(sorted(g.uncertain_batches)),
                "ranks": {(s, r): {"pid": target.pid, "alive": target.alive,
                                  "capabilities": deepcopy(target.capabilities),
                                  "pending_commands": len(target.commands),
                                  "exit_code": target.exit_code}
                          for (s, r), target in g.registered.items()},
                "batches": {bid: self._batch_view(batch) for bid, batch in g.batches.items()},
                "pending_events": len(g.outbox.pending),
                "acked_event_seq": g.outbox.acked_seq} for m, g in self.groups.items()},
                "requests": {key: self._output(value) for key, value in self.requests.items()},
                "credits": {r: tuple(refs) for r, refs in self.credits.refs.items()},
                "policy_limits": dict(self.policy.limits),
                "policy_adjustments": deepcopy(tuple(self.policy.adjustments))}

    def metrics(self) -> dict:
        """Small bounded telemetry view for paired benchmark collection."""
        with self.lock:
            batches = [batch for group in self.groups.values() for batch in group.batches.values()]
            history = self.store.history()
            return {"active_requests": len(self.requests),
                "timing_boundaries": {
                    "execution_wall_ms": "rank START handler to synchronized result, maximum across compute ranks",
                    "gpu_forward_ms": "CUDA current-stream events around ModelRunner.forward, maximum across ranks; includes device idle/communication, not kernel active time",
                    "prepare_handshake_ms": "reservation preparation to all P/D ranks READY, including CPU mapping/control",
                    "pd_handoff_ms": "last all-rank GPU_DONE accepted to all D HANDOFF_READY; excludes prefill forward"},
                "initializing": self.initializing,
                "recovery_stats": {"cycles": self.recovery_cycles,
                                   "automatic_suspensions": self.recovery_suspensions},
                "active_batches": sum(batch.status not in ("DONE", "ABORTED") for batch in batches),
                "pending_feedback": sum(batch.status == "DONE" and not batch.feedback_sent
                                         for batch in batches),
                "feedback_totals": deepcopy(self.policy.feedback_totals),
                "history": history,
                "model_batch_history": {model_id: {
                    "used": len(group.batches) + len(group.retired.ids["batch"]),
                    "limit": group.retired.limit("batch")}
                    for model_id, group in self.groups.items()},
                "adjustments": deepcopy(tuple(self.policy.adjustments)),
                "scheduler_tick_count": self._tick_count,
                "scheduler_tick_total_ms": self._tick_duration_ms}
