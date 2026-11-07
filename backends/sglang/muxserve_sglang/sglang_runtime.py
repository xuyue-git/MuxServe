"""Permit-controlled native Scheduler for the frozen SGLang Llama path.

CPU commands select every batch. Native helpers retain request normalization,
logical allocation, sampling, finish checking, and metadata-only PD processing.
"""
from __future__ import annotations

from array import array
from collections import OrderedDict
from copy import deepcopy
from dataclasses import dataclass, field, replace
import json
import os
import time

import numpy as np
import torch
from sglang.srt.disaggregation.base.conn import KVPoll
from sglang.srt.disaggregation.prefill import maybe_release_metadata_buffer
from sglang.srt.managers.io_struct import TokenizedGenerateReqInput
from sglang.srt.managers.overlap_utils import resolve_forward_inputs
from sglang.srt.managers.schedule_batch import ScheduleBatch
from sglang.srt.managers.scheduler_components.metrics_reporter import PrefillStats
from sglang.srt.mem_cache.common import (
    kv_to_page_indices, maybe_cache_unfinished_req, release_kv_cache,
)
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.observability.metrics_collector import QueueCount
from sglang.srt.observability.req_time_stats import set_schedule_time_batch
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.srt.sampling.sampling_batch_info import SamplingBatchInfo
from sglang.srt.utils import DynamicGradMode

from . import trace
from .step_profile import StepProfiler
from .contracts import IntegrationError, KVView, RequestIdentity
from .ipc import CommandSnapshot, copy_message


def require(ok, message, code="NATIVE_STATE_MISMATCH"):
    if not ok:
        raise IntegrationError(code, message)


def view_from_dict(value):
    return KVView(**{**value, "identity": RequestIdentity(**value["identity"])})


class CoordinatorOutputCapture:
    """Native callbacks remain reachable; only controller events leave the rank."""
    def __init__(self):
        self.calls = 0

    def stream_output(self, reqs, *args, **kwargs):
        # No second output history is retained here. Native result processors
        # already update Req; the controller validates ordered results itself.
        self.calls += 1


@dataclass
class Prepared:
    command: dict
    reqs: list = field(default_factory=list)
    batch: object | None = None
    ready: bool = False
    started: bool = False
    handoff: dict | None = None
    handoff_done: bool = False
    decode_rollback: list = field(default_factory=list)


class ControlledScheduler:
    def __init__(self, scheduler, context):
        self.s, self.ctx = scheduler, context
        self.runner = scheduler.tp_worker.model_runner
        self.pool = self.runner.token_to_kv_pool
        self.rank, self.pid = self.runner.ps.tp_rank, os.getpid()
        self.cursor = self.report_seq = 0
        self.commands = OrderedDict()
        self.prepared = {}
        self.requests = {}
        self.history_offsets = {}
        self.handoff_pending = None
        self.pending_report = None
        self.drained = False
        require(self.pool.headwise_shared and not scheduler.enable_overlap
                and scheduler.spec_algorithm.is_none() and self.runner.ps.pp_size == 1
                and scheduler.tree_cache.is_chunk_cache() and not scheduler.enable_hisparse
                and not scheduler.enable_decode_hicache and scheduler.chunked_req is None,
                "Controlled runtime requires the supported non-overlap Llama/ChunkCache path")
        require(scheduler.transfer_backend.value == "shared_headwise", "Shared PD backend is required")
        timeout = scheduler.server_args.dist_timeout
        require(type(timeout) in (int, float) and 0 < timeout <= context.config.request_timeout_ms / 1000,
                "Launcher must configure a finite native distributed timeout")
        decode_graph = getattr(self.runner, "decode_cuda_graph_runner", None)
        graphs_allowed = context.stage == "D" and getattr(context.config, "decode_cuda_graph", False)
        require(decode_graph is None or (graphs_allowed and getattr(self.pool, "graph", None) is not None
                                         and getattr(decode_graph, "attn_backend", None)
                                         is getattr(self.runner, "attn_backend", None)),
                "Decode CUDA graphs require decode_cuda_graph on a D rank with the Head-wise backend")
        prefill_graph = getattr(self.runner, "prefill_cuda_graph_runner", None)
        require(prefill_graph is None or type(prefill_graph).__name__ == "EagerRunner"
                or (context.stage == "P" and getattr(context.config, "prefill_cuda_graph", False)
                    and type(getattr(prefill_graph, "backend", None)).__name__ == "BreakableCudaGraphBackend"
                    and getattr(self.runner, "attn_backend", None).__class__.__name__
                    == "HeadwiseAttentionBackend"),
                "Prefill CUDA graphs require prefill_cuda_graph on a P rank (breakable, Head-wise)")
        require(getattr(self.runner, "cuda_graph_runner", None) is None
                and not getattr(scheduler, "enable_staging", False)
                and getattr(scheduler, "decode_offload_manager", None) is None
                and getattr(scheduler, "kv_checksum_computer", None) is None,
                "Graphs, staging, native KV offload and dense KV checksums are unsupported")
        self.capture = CoordinatorOutputCapture()
        # Native result processors are frozen; retain every other field by reference.
        scheduler.batch_result_processor = replace(
            scheduler.batch_result_processor, output_streamer=self.capture)
        scheduler.output_streamer = self.capture
        if context.stage == "P":
            self.manager = scheduler.disagg_prefill_bootstrap_queue.kv_manager
        else:
            self.manager = scheduler.disagg_decode_prealloc_queue.kv_manager

    def rpc(self, name, *args):
        return self.ctx.rpc.call("controller_" + name, self.ctx.group_id, *args)

    def sync(self):
        torch.cuda.synchronize(self.runner.gpu_id)

    def _report(self, batch_id, kind, payload):
        envelope = (str(self.report_seq + 1), batch_id, kind, deepcopy(payload))
        require(self.pending_report is None or self.pending_report == envelope,
                "An uncertain report must be replayed before advancing its sequence")
        self.pending_report = envelope
        trace.ev("rep_send", batch_id, kind=kind)
        reply = self.rpc("rank_report", self.ctx.stage, self.rank, self.pid, *envelope)
        require(reply["accepted"], "Controller refused a rank report")
        self.report_seq += 1
        self.pending_report = None

    @DynamicGradMode()
    def run(self):
        trace.set_role(f"{self.ctx.model.model_id}-{self.ctx.stage}{self.rank}")
        self.step_profiler = StepProfiler(self.ctx.model.model_id, self.ctx.stage, self.rank)
        eos = self.s.model_config.hf_eos_token_id
        eos = () if eos is None else tuple(eos) if isinstance(eos, (tuple, list, set)) else (eos,)
        tokenizer = self.s.tokenizer
        if tokenizer is not None:
            eos += tuple(getattr(tokenizer, "additional_stop_token_ids", ()) or ())
            if tokenizer.eos_token_id is not None:
                eos += (tokenizer.eos_token_id,)
        eos = tuple(sorted(set(eos)))
        self.rpc("register_rank", self.ctx.stage, self.rank, self.pid,
                 {"vocab_size": self.s.model_config.vocab_size, "eos_token_ids": eos,
                  "native_pd": True, "headwise_kv": True})
        try:
            # Commands fetched but not yet accepted (held while a HANDOFF is
            # pending). Long polls ask only for newer commands, and wake early
            # when this group's broker state changes (epoch), so neither idle
            # ranks nor pending handoffs spin on the owner.
            backlog = []
            epoch = None
            wait_ms = 0
            while not self.drained:
                after = backlog[-1]["command_seq"] if backlog else self.cursor
                reply = self.rpc("poll_commands_wait", self.ctx.stage, self.rank, self.pid,
                                 after, self.cursor or None, wait_ms, epoch)
                epoch = reply["epoch"]
                backlog.extend(reply["commands"])
                accepted = 0
                while backlog:
                    # HANDOFF contains a native TP all-reduce. Every rank must
                    # complete it before consuming any later START/QUIESCE,
                    # even when command polling returns different-sized pages.
                    if self.handoff_pending is not None:
                        break
                    command = backlog.pop(0)
                    tag = command.get("batch_id") or command.get("cleanup_id")
                    trace.ev("acc", tag, kind=command.get("kind"), seq=command.get("command_seq"))
                    self._accept(command)
                    trace.ev("acc_end", tag, kind=command.get("kind"), seq=command.get("command_seq"))
                    accepted += 1
                    if self.drained or self.handoff_pending is not None:
                        break
                if self.drained:
                    # DRAIN receipt ACK is delivered before returning through
                    # native run_scheduler_process's normal finally block.
                    self.rpc("poll_commands", self.ctx.stage, self.rank, self.pid,
                             self.cursor, self.cursor)
                    self.s.gracefully_exit = True
                    trace.flush()
                    return
                self._progress()
                if accepted or (backlog and self.handoff_pending is None):
                    wait_ms = 0
                elif self.prepared or self.handoff_pending is not None:
                    wait_ms = 20
                else:
                    wait_ms = 200
        except Exception as exc:
            # Never issue a QUIESCED proof after an uncertain CUDA/API failure.
            # Preserve the original error if the owner is unreachable too.
            try:
                if self.pending_report is not None:
                    _, batch_id, kind, payload = self.pending_report
                    self._report(batch_id, kind, payload)
                active = next(iter(self.prepared), None)
                if active is not None:
                    self._report(active, "ERROR", {"code": getattr(exc, "code", type(exc).__name__)})
                else:
                    self.rpc("rank_failed", self.ctx.stage, self.rank, self.pid,
                             getattr(exc, "code", type(exc).__name__))
            except Exception:
                pass
            raise

    def _accept(self, command):
        require(command["group_id"] == self.ctx.group_id
                and command["model_id"] == self.ctx.model.model_id
                and command["stage"] == self.ctx.stage and command["rank"] == self.rank,
                "Command execution identity differs")
        seq = command["command_seq"]
        if seq <= self.cursor:
            retained = self.commands.get(seq)
            previous = retained.detach() if isinstance(retained, CommandSnapshot) else retained
            require(previous == command, "Changed or expired command replay")
            return
        require(seq == self.cursor + 1, "Command stream gap")
        self.commands[seq] = (CommandSnapshot.capture(command) if command["kind"] in ("PREPARE", "HANDOFF")
                              else deepcopy(command))
        kind = command["kind"]
        if kind == "PREPARE":
            self._prepare(command)
        elif kind == "START":
            self._start(command)
        elif kind == "HANDOFF":
            state = self.prepared[command["batch_id"]]
            require(self.ctx.stage == "D" and state.command["batch_stage"] == "P"
                    and state.command["reservation_id"] == command["reservation_id"],
                    "HANDOFF does not match native preparation")
            state.handoff = copy_message(command)
            self.handoff_pending = command["batch_id"]
            require(tuple(i["request_id"] for i in command["requests"]) ==
                    state.command["ordered_request_ids"], "Handoff order differs")
            queue = self.s.disagg_decode_transfer_queue
            for item, req in zip(command["requests"], state.reqs):
                native = next(dr for dr in queue.queue if dr.req is req)
                native.kv_receiver.accept_handoff(item, self.pool)
        elif kind in ("QUIESCE", "REVOKE"):
            self._quiesce(command)
        elif kind == "CLEANUP":
            identity = (command["request_id"], command["attempt"])
            self.sync()
            self._release_request(identity)
            self.sync()
            self._trim_idle_cuda_cache(command["cleanup_id"])
            self._report(command["cleanup_id"], "QUIESCED", {
                "cleanup_id": command["cleanup_id"], "request_id": identity[0], "attempt": identity[1]})
        elif kind == "DRAIN":
            require(not self.prepared and not self.requests and self.pool._bound is None
                    and not self.manager.descriptors, "Cannot drain live native/imported request state")
            self.sync()
            self.drained = True
        else:
            raise IntegrationError("UNKNOWN_COMMAND", f"Unsupported controlled command: {kind}")
        self.cursor = seq
        while len(self.commands) > 1024:
            self.commands.popitem(last=False)

    def _trim_idle_cuda_cache(self, cleanup_id):
        """Release unused allocator slabs after synchronized, fully idle cleanup.

        Apply the same rule to warmup and serving requests, so startup does not
        hide an allocator-cache footprint that immediately returns in service.
        Live tensors, shared imports and the owner's physical KV remain intact.
        """
        if (self.prepared or self.requests or self.pool._bound is not None
                or self.manager.descriptors):
            return
        require(self.runner._muxserve_permit is None, "Live permit prevents idle cache trimming")
        from .weight_loader import _local_alias, audit_shared_weights
        aliases = (_local_alias(self.pool.pool_key), _local_alias(self.pool.pool_value))
        # Weight aliases are audited once after the release (they were
        # audited at load and by every earlier trim); a pre-release audit
        # walked the whole model again on every request cleanup.
        device = self.runner.gpu_id
        before_allocated = torch.cuda.memory_allocated(device)
        before_reserved = torch.cuda.memory_reserved(device)
        with torch.cuda.device(device):
            torch.cuda.empty_cache()
            self.sync()
        audit_shared_weights(self.runner.model)
        require(aliases == (_local_alias(self.pool.pool_key), _local_alias(self.pool.pool_value)),
                "Shared KV alias changed during idle cache trimming", "KV_ALIAS_CHANGED")
        after_allocated = torch.cuda.memory_allocated(device)
        after_reserved = torch.cuda.memory_reserved(device)
        print("IDLE_CUDA_CACHE " + json.dumps({"model_id": self.ctx.model.model_id,
            "stage": self.ctx.stage, "rank": self.rank, "pid": self.pid, "cleanup_id": cleanup_id,
            "allocated_before_bytes": before_allocated, "allocated_after_bytes": after_allocated,
            "reserved_before_bytes": before_reserved, "reserved_after_bytes": after_reserved,
            "released_reserved_bytes": max(0, before_reserved - after_reserved),
            "shared_aliases_verified": True}), flush=True)

    def _native_request(self, item):
        identity = (item["request_id"], item["attempt"])
        require(identity not in self.requests, "Attempt already owns a native request")
        rebuild = item["purpose"] == "REBUILD"
        inputs = item["selected_input_ids"] if rebuild else item["input_ids"]
        remaining = item["max_new_tokens"] - len(item["output_token_ids"]) + 1 if rebuild else item["max_new_tokens"]
        params = SamplingParams(max_new_tokens=remaining, temperature=0, top_k=1,
                                ignore_eos=item["sampling"].get("ignore_eos", False))
        params.normalize(self.s.tokenizer)
        params.verify(self.s.model_config.vocab_size)
        # The length-prefixed ID plus attempt cannot prefix-match another
        # native request in SGLang's abort handlers.
        rid = f"mux:{self.ctx.group_id}:{len(identity[0])}:{identity[0]}:a{identity[1]}:"
        recv = TokenizedGenerateReqInput(rid=rid, input_text=None, input_ids=array("q", inputs),
            input_embeds=None, mm_inputs=None, token_type_ids=None, sampling_params=params,
            return_logprob=False, logprob_start_len=-1, top_logprobs_num=0,
            token_ids_logprob=None, stream=False, bootstrap_host="127.0.0.1",
            bootstrap_port=self.ctx.model.bootstrap_port, bootstrap_room=item["bootstrap_room"],
            disagg_prefill_dp_rank=0)
        self.manager.bind_request(item)
        self.s.handle_generate_request(recv)
        if self.ctx.stage == "P":
            req = next(r for r in self.s.disagg_prefill_bootstrap_queue.queue if r.rid == rid)
        else:
            req = next(dr.req for dr in self.s.disagg_decode_prealloc_queue.queue if dr.req.rid == rid)
        require(not req.finished() and req.to_finish is None and list(req.origin_input_ids) == list(inputs),
                "Native request normalization rejected or changed selected input")
        # Native P deliberately budgets one boundary sample; only D owns the
        # remaining output budget. Controller limits stay unchanged on both.
        expected_budget = 1 if self.ctx.stage == "P" else remaining
        require(req.sampling_params.max_new_tokens == expected_budget,
                "Native context normalization changed the controller output budget")
        self.requests[identity] = req
        self.history_offsets[identity] = len(item["output_token_ids"]) - 1 if rebuild else 0
        return req

    def _new_batch(self, reqs):
        return ScheduleBatch.init_new(reqs, self.s.req_to_token_pool,
            self.s.token_to_kv_pool_allocator, self.s.tree_cache, self.s.model_config,
            self.s.enable_overlap, self.s.spec_algorithm)

    def _validate_views(self, command, rows, logical_only=False):
        # Compute and passive ranks use the same complete global-view checks.
        # Bind the compute pool once; passive preparation gets no GPU permission.
        from .sglang_memory import HeadwiseTokenToKVPool
        validator = (HeadwiseTokenToKVPool(self.pool.layout, self.rank,
            self.pool.pool_key, self.pool.pool_value, self.pool.size) if logical_only else self.pool)
        views = [view_from_dict(item["reservation_view"]) for item in command["requests"]]
        targets = [item["target_valid_tokens"] for item in command["requests"]]
        mode = "DECODE" if command["batch_stage"] == "D" else command["purpose"] if command["purpose"] == "REBUILD" else "PREFILL"
        for view, item in zip(views, command["requests"]):
            require(view.identity == RequestIdentity(self.ctx.model.model_id, item["request_id"], item["attempt"])
                    and view.block_tables[self.rank] == item["rank_block_table"], "Reservation identity/map mismatch")
        validator.bind_forward(views, rows, targets, mode)
        return views, targets, mode

    def _prepare(self, command):
        require(command["batch_id"] not in self.prepared, "Batch preparation already exists")
        require(tuple(item["request_id"] for item in command["requests"]) == command["ordered_request_ids"],
                "Selected native request order differs")
        state = Prepared(copy_message(command))
        self.prepared[command["batch_id"]] = state
        if command["batch_stage"] == "P":
            state.reqs = [self._native_request(item) for item in command["requests"]]
            if self.ctx.stage == "D":
                queue = self.s.disagg_decode_prealloc_queue
                for req in state.reqs:
                    native = next(dr for dr in queue.queue if dr.req is req)
                    require(native.kv_receiver.poll() == KVPoll.WaitingForInput,
                            "Native D handshake is not ready for logical allocation")
                    native.waiting_for_input = True
                    req.time_stats.set_bootstrap_done_time()
                    indices = queue._pre_alloc(req)
                    self._register_destination(queue, native, indices)
                    queue.queue.remove(native)
                    self.s.disagg_decode_transfer_queue.add(native)
                    queue.note_destinations_queued(1)
                self._ready(state, logical_only=True)
        elif self.ctx.stage == "D":
            for item in command["requests"]:
                req = self.requests[(item["request_id"], item["attempt"])]
                require(not req.finished() and req.kv.holds_kv
                        and req.kv.kv_committed_len == item["target_valid_tokens"] - 1,
                        "D request does not own the selected native boundary")
                offset = self.history_offsets[(item["request_id"], item["attempt"])]
                require(list(req.output_ids) == list(item["output_token_ids"])[offset:]
                        and len(req.origin_input_ids) + len(req.output_ids) - 1 == req.kv.kv_committed_len,
                        "Native output history differs from committed controller state")
                # Preserve rollback state before native prepare_for_decode
                # appends a logical ID and advances the request counters.
                state.decode_rollback.append((req, req.kv.kv_committed_len,
                    req.kv.kv_allocated_len, req.decode_batch_idx))
                state.reqs.append(req)
            state.batch = self._new_batch(state.reqs)
            self._initialize_decode_metadata(state.batch)
            state.batch.prepare_for_decode()
            set_schedule_time_batch(state.batch)
            self._ready(state)
        else:
            # P participates in all-rank readiness without a D model forward.
            self._ready(state, logical_only=True)

    def _initialize_decode_metadata(self, batch):
        """Fresh non-spec metadata; PREBUILT belongs only to initial handoff.

        Native alloc_for_decode reads the committed row/length tensors, then
        prepare_for_decode increments them. Native run_batch resolves input_ids
        from FutureMap, populated once by PREBUILT and then by each real sample.
        No previous ScheduleBatch or SamplingBatchInfo is reused.
        """
        rows = [req.kv.req_pool_idx for req in batch.reqs]
        lengths = [req.kv.kv_committed_len for req in batch.reqs]
        batch.req_pool_indices_cpu = torch.tensor(rows, dtype=torch.int64)
        batch.req_pool_indices = batch.req_pool_indices_cpu.to(batch.device)
        batch.seq_lens_cpu = torch.tensor(lengths, dtype=torch.int64)
        batch.seq_lens = batch.seq_lens_cpu.to(batch.device)
        batch.orig_seq_lens = batch.seq_lens.to(dtype=torch.int32)
        batch.seq_lens_sum = sum(lengths)
        batch.input_ids = None
        batch.sampling_info = SamplingBatchInfo.from_schedule_batch(batch, self.s.model_config.vocab_size)
        require(not batch.sampling_info.penalizer_orchestrator.is_required,
                "Controlled greedy path does not support stateful penalties")

    def _register_destination(self, queue, native, indices):
        # v0.5.20 performs this inside pop_preallocated. Keep its supported
        # MHA/page1 metadata steps while selecting only the granted requests.
        require(self.manager.kv_args.page_size == 1
                and not self.manager.kv_args.state_types
                and not queue.transfer_queue.enable_staging,
                "Controlled destinations require page1 MHA without staging")
        native.metadata_buffer_index = queue.req_to_metadata_buffer_idx_allocator.alloc()
        require(native.metadata_buffer_index is not None, "Native metadata slots exhausted")
        page_indices = kv_to_page_indices(indices, 1).astype(np.int32)
        native.kv_receiver.send_metadata(page_indices, native.metadata_buffer_index,
                                         [], decode_prefix_len=0)
        queue._num_published_destinations += 1
        native.req.time_stats.set_decode_transfer_queue_entry_time()

    def _ready(self, state, logical_only=False):
        command = state.command
        rows = [r.kv.req_pool_idx for r in state.reqs] if state.reqs else list(range(len(command["requests"])))
        if not logical_only:
            require(self.pool._bound is None, "A selected native compute binding is still live")
        views, targets, mode = self._validate_views(command, rows, logical_only)
        if not logical_only:
            batch = state.batch
            self.sync()
            require(tuple(batch.req_pool_indices_cpu.tolist()) == tuple(rows)
                    and tuple(batch.req_pool_indices.cpu().tolist()) == tuple(rows)
                    and tuple(batch.seq_lens_cpu.tolist()) == tuple(targets)
                    and tuple(batch.seq_lens.cpu().tolist()) == tuple(targets),
                    "Native CPU/GPU row/length metadata differs before READY")
            if mode != "DECODE":
                require(batch.extend_lens == targets and batch.prefix_lens == [0] * len(rows),
                        "Native extend selection differs from bound Headwise targets")
        else:
            self.sync()  # logical preallocation writes must complete before READY
        state.ready = True
        self._report(command["batch_id"], "READY", {"reservation_id": command["reservation_id"]})

    def _progress(self):
        for state in tuple(self.prepared.values()):
            if not state.ready and self.ctx.stage == "P":
                polls = [r.disagg_kv_sender.poll() for r in state.reqs]
                require(KVPoll.Failed not in polls, "P bootstrap was aborted")
                if all(p == KVPoll.WaitingForInput for p in polls):
                    trace.ev("p_boot_ok", state.command["batch_id"])
                    queue = self.s.disagg_prefill_bootstrap_queue
                    for req in state.reqs:
                        require(queue.finalize_bootstrap(req), "Native P metadata slots exhausted")
                        queue.queue.remove(req)
                        req.init_next_round_input(self.s.tree_cache)
                        require(len(req.prefix_indices) == 0, "P prefix reuse is unsupported")
                        req.set_extend_range(0, len(req.origin_input_ids))
                        req.time_stats.set_forward_entry_time()
                    state.batch = self._new_batch(state.reqs)
                    state.batch.prepare_for_extend()
                    state.batch.prefill_stats = PrefillStats(
                        log_input_tokens=sum(len(req.origin_input_ids) for req in state.reqs),
                        log_hit_tokens=0, new_token_ratio=1.0,
                        num_running_reqs=QueueCount.from_reqs([], False),
                        num_new_seqs=len(state.reqs))
                    set_schedule_time_batch(state.batch)
                    self._ready(state)
            if state.handoff is not None and not state.handoff_done:
                require(time.monotonic() < state.handoff["handoff_deadline"],
                        "Native handoff watchdog expired; ownership remains uncertain", "HANDOFF_TIMEOUT")
                queue = self.s.disagg_decode_transfer_queue
                natives = [next(dr for dr in queue.queue if dr.req is req) for req in state.reqs]
                polls = [dr.kv_receiver.poll() for dr in natives]
                require(KVPoll.Failed not in polls, "Committed D handoff was aborted")
                if all(p == KVPoll.Success for p in polls):
                    # Exactly one native collective transfer pass, only after
                    # owner proof that every rank has ACKed every selected room.
                    trace.ev("ho_ok", state.command["batch_id"])
                    transferred = queue.pop_transferred([r.rid for r in state.reqs])
                    trace.ev("ho_pop", state.command["batch_id"])
                    require(transferred == state.reqs, "Native transfer order or success differs")
                    for req in transferred:
                        req.init_next_round_input(self.s.tree_cache)
                        req.set_extend_range(len(req.prefix_indices), req.kv.kv_committed_len)
                    batch = self._new_batch(transferred)
                    batch.prepare_for_prebuilt()
                    require(batch.inner_idle_batch is None, "PREBUILT cannot hide an ungranted idle forward")
                    batch.process_prebuilt(self.s.future_map)
                    self.s.batch_result_processor.process_batch_result_prebuilt(batch)
                    for req in transferred:
                        self._validate_finish(req)
                    trace.ev("ho_prebuilt", state.command["batch_id"])
                    self.sync()
                    trace.ev("ho_sync", state.command["batch_id"])
                    state.handoff_done = True
                    self._report(state.command["batch_id"], "HANDOFF_READY",
                                 {"requests": state.handoff["requests"]})
                    # D is not sent the P-only batch QUIESCE. This preparation
                    # owns no forward binding; retain Req, discard batch state.
                    del self.prepared[state.command["batch_id"]]
                    self.handoff_pending = None
                    for item in state.command["requests"]:
                        self.manager.forget(item["bootstrap_room"])

    def _start(self, command):
        state = self.prepared[command["batch_id"]]
        require(state.ready and not state.started and state.batch is not None
                and state.command["batch_stage"] == self.ctx.stage
                and command["grant_id"] == state.command["reservation_id"],
                "START requires the fresh selected native preparation", "FORWARD_NOT_GRANTED")
        require(self.runner._muxserve_permit is None, "Prior permit still exists")
        state.started = True
        self.runner._muxserve_permit = {"grant_id": command["grant_id"], "consumed": False}
        if state.command.get("run_steps", 1) > 1:
            return self._start_run(command, state)
        start = time.monotonic()
        previous_output_counts = [len(req.output_ids) for req in state.reqs]
        self.s.cur_batch_for_debug = state.batch
        # Opt-in diagnosis: on P ranks one profiled "step" is one prefill
        # forward (run_batch/forward .. result handling .. sync).
        profiler = getattr(self, "step_profiler", None) if self.ctx.stage == "P" else None
        if profiler is not None:
            profiler.before()
        if state.command["purpose"] == "REBUILD":
            # EXTEND executes the historical prefix but never enters sampling
            # or native prefill result handling that would append a new token.
            self.s.forward_ct += 1
            state.batch.forward_iter = self.s.forward_ct
            state.batch.launch_ts = start
            resolve_forward_inputs(state.batch, self.s.future_map)
            forward = ForwardBatch.init_new(state.batch, self.runner,
                                            return_hidden_states_before_norm=False)
            self.runner.forward(forward)
            self.sync()
            for req, item in zip(state.reqs, state.command["requests"]):
                req.time_stats.set_prefill_finished_time()
                req.output_ids.append(item["boundary_token_id"])
                maybe_cache_unfinished_req(req, self.s.tree_cache)
                self.s.disagg_prefill_inflight_queue.append(req)
                self.s.send_kv_chunk(req, last_chunk=True)
                req.time_stats.set_prefill_transfer_queue_entry_time()
        else:
            result = self.s.run_batch(state.batch)
            require(result.delay_sample_func is None, "Delayed sampling is unsupported on controlled non-overlap path")
            self.s.process_batch_result(state.batch, result)
            require(all(len(req.output_ids) == previous + 1
                        for req, previous in zip(state.reqs, previous_output_counts)),
                    "Native greedy result must append exactly one token")
            if self.ctx.stage == "D":
                for req in state.reqs:
                    self._validate_finish(req)
        self.sync()
        if profiler is not None:
            profiler.after()
        self.s.cur_batch_for_debug = None
        require(self.runner._muxserve_permit["consumed"], "Granted native forward was not consumed")
        permit = self.runner._muxserve_permit
        require(permit.get("gpu_timing_finished"), "Native forward CUDA timing boundary is missing")
        gpu_forward_ms = permit["gpu_start_event"].elapsed_time(permit["gpu_end_event"])
        results = [{"request_id": item["request_id"], "attempt": item["attempt"],
                    "token_id": None if state.command["purpose"] == "REBUILD" else req.output_ids[-1],
                    "valid_tokens": item["target_valid_tokens"]}
                   for req, item in zip(state.reqs, state.command["requests"])]
        self._report(command["batch_id"], "GPU_DONE", {
            "reservation_id": state.command["reservation_id"], "results": results,
            "gpu_forward_ms": gpu_forward_ms,
            "duration_ms": (time.monotonic() - start) * 1000})

    def _start_run(self, command, state):
        """Execute a multi-step D grant: forward, report, then continue only on
        the owner's per-step decision (identical for every TP rank)."""
        steps = state.command["run_steps"]
        require(self.ctx.stage == "D" and state.command["batch_stage"] == "D"
                and state.command["purpose"] == "GENERATE" and type(steps) is int and steps > 1,
                "Decode runs require a D GENERATE grant")
        start = time.monotonic()
        start_valid = [item["target_valid_tokens"] - 1 for item in state.command["requests"]]
        gpu_forward_ms = 0.0
        step = 1
        while True:
            previous_output_counts = [len(req.output_ids) for req in state.reqs]
            self.s.cur_batch_for_debug = state.batch
            profiler = getattr(self, "step_profiler", None)
            if profiler is not None:
                profiler.before()
            result = self.s.run_batch(state.batch)
            require(result.delay_sample_func is None, "Delayed sampling is unsupported on controlled non-overlap path")
            self.s.process_batch_result(state.batch, result)
            require(all(len(req.output_ids) == previous + 1
                        for req, previous in zip(state.reqs, previous_output_counts)),
                    "Native greedy result must append exactly one token")
            for req in state.reqs:
                self._validate_finish(req)
            self.sync()
            if profiler is not None:
                profiler.after()
            trace.ev("step_fwd", command["batch_id"], step=step)
            permit = self.runner._muxserve_permit
            require(permit["consumed"] and permit.get("gpu_timing_finished"),
                    "Granted native forward was not consumed or timed")
            gpu_forward_ms += permit["gpu_start_event"].elapsed_time(permit["gpu_end_event"])
            results = [{"request_id": item["request_id"], "attempt": item["attempt"],
                        "token_id": req.output_ids[-1], "valid_tokens": valid + step}
                       for req, item, valid in zip(state.reqs, state.command["requests"], start_valid)]
            finished = any(req.finished() for req in state.reqs)
            reply = self.rpc("decode_step", self.rank, self.pid, command["batch_id"], step,
                             results, finished)
            while reply.get("continue") == "PARK":
                # Parked for a waiting prefill: the GPU is idle (synchronized
                # above) and every native/KV state stays as it is.
                reply = self.rpc("decode_resume", self.rank, self.pid, command["batch_id"], step, 100)
            if finished or step >= steps or reply.get("continue") is not True:
                break
            step += 1
            self._advance_run(state, start_valid, step, command["grant_id"])
        self.s.cur_batch_for_debug = None
        self._report(command["batch_id"], "GPU_DONE", {
            "reservation_id": state.command["reservation_id"], "results": results,
            "gpu_forward_ms": gpu_forward_ms, "duration_ms": (time.monotonic() - start) * 1000,
            "steps": step})

    def _advance_run(self, state, start_valid, step, grant_id):
        """Prepare decode step ``step`` locally, exactly as a fresh per-token
        PREPARE would, inside the KV capacity already reserved for the run."""
        for req, valid in zip(state.reqs, start_valid):
            require(not req.finished() and req.kv.holds_kv
                    and req.kv.kv_committed_len == valid + step - 1
                    and len(req.origin_input_ids) + len(req.output_ids) - 1 == req.kv.kv_committed_len,
                    "D request does not own the next run boundary")
        state.batch = self._new_batch(state.reqs)
        self._initialize_decode_metadata(state.batch)
        state.batch.prepare_for_decode()
        set_schedule_time_batch(state.batch)
        rows = [req.kv.req_pool_idx for req in state.reqs]
        targets = [valid + step for valid in start_valid]
        self.pool.advance_decode(rows, targets)
        require(tuple(state.batch.req_pool_indices_cpu.tolist()) == tuple(rows)
                and tuple(state.batch.seq_lens_cpu.tolist()) == tuple(targets),
                "Native CPU row/length metadata differs from the run step")
        self.runner._muxserve_permit = {"grant_id": grant_id, "consumed": False}

    def _validate_finish(self, req):
        # Native precedence is EOS/stop token before length (same as owner).
        # Only the declared greedy subset is accepted, so other finish types
        # indicate an abort/corruption and must never produce GPU_DONE success.
        from sglang.srt.managers.schedule_batch import FINISH_LENGTH, FINISH_MATCHED_TOKEN
        reason = req.finished_reason
        require(reason is None or isinstance(reason, (FINISH_LENGTH, FINISH_MATCHED_TOKEN)),
                "Native request finished with an unsupported/error reason")
        require(req.to_finish is None, "Native request has an unprocessed abort")

    def _rollback_decode(self, state):
        require(not state.started and state.batch is not None, "Cannot roll back started decode")
        self.s.token_to_kv_pool_allocator.free(state.batch.out_cache_loc)
        for req, committed, allocated, decode_idx in state.decode_rollback:
            req.kv.kv_committed_len = committed
            req.kv.kv_allocated_len = allocated
            req.decode_batch_idx = decode_idx

    def _quiesce(self, command):
        state = self.prepared[command["batch_id"]]
        require(command["reservation_id"] == state.command["reservation_id"], "Quiescence reservation differs")
        self.sync()
        if command["kind"] == "REVOKE":
            require(not state.started, "Cannot revoke an already started forward")
            if state.command["batch_stage"] == "D" and self.ctx.stage == "D":
                self._rollback_decode(state)
            elif state.command["batch_stage"] == "P":
                for item in state.command["requests"]:
                    self._release_request((item["request_id"], item["attempt"]))
        elif self.ctx.stage == "P" and state.command["batch_stage"] == "P":
            # HANDOFF_READY on every D rank precedes this command, hence the
            # native P inflight poll collective executes once with Success.
            done = self.s.process_disagg_prefill_inflight_queue([r.rid for r in state.reqs])
            require(done == state.reqs, "P native transfer cleanup is incomplete")
            for item in state.command["requests"]:
                self._release_request((item["request_id"], item["attempt"]), successful_prefill=True)
        self.sync()
        if state.batch is not None:
            self.pool.clear_forward()
            self.runner._muxserve_permit = None
        del self.prepared[command["batch_id"]]
        self._report(command["batch_id"], "QUIESCED", {
            "reservation_id": command["reservation_id"],
            "request_ids": tuple(state.command["ordered_request_ids"])})

    def _release_request(self, identity, successful_prefill=False):
        req = self.requests.pop(identity, None)
        self.history_offsets.pop(identity, None)
        if req is None:
            return
        if self.ctx.stage == "P":
            queue = self.s.disagg_prefill_bootstrap_queue
            sender = getattr(req, "disagg_kv_sender", None)
            if sender is not None:
                if not successful_prefill:
                    sender.abort()
                sender.clear()
            for requests in (queue.queue, self.s.disagg_prefill_inflight_queue, self.s.waiting_queue):
                if req in requests:
                    requests.remove(req)
            self.s.clear_pending_chunk_send(req)
            maybe_release_metadata_buffer(req, self.s.req_to_metadata_buffer_idx_allocator)
            req.pending_bootstrap = False
        else:
            prealloc = self.s.disagg_decode_prealloc_queue
            transfer = self.s.disagg_decode_transfer_queue
            for queue in (prealloc.queue, prealloc.pending_reqs, transfer.queue):
                for native in tuple(queue):
                    if native.req is req:
                        if native.kv_receiver is not None:
                            native.kv_receiver.abort()
                            native.kv_receiver.clear()
                        if native.metadata_buffer_index >= 0:
                            self.manager.metadata_buffers.bootstrap_room[native.metadata_buffer_index].zero_()
                            self.s.req_to_metadata_buffer_idx_allocator.free(native.metadata_buffer_index)
                            native.metadata_buffer_index = -1
                        queue.remove(native)
            if req in self.s.waiting_queue:
                self.s.waiting_queue.remove(req)
        if req.kv.holds_kv:
            release_kv_cache(req, self.s.tree_cache, is_insert=False)
        if not successful_prefill:
            self.s._release_aborted_request(req)
        self.manager.forget(req.bootstrap_room)
