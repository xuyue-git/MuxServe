"""User-run initialization grants and sampled, physical-GPU memory gate.

No HTTP listener is created here. Sampling reports observed minima, not a
continuous allocator high-water mark; transients between samples can be missed.
The configured free-memory reserve is required on every selected physical card.
Startup defaults to short functional P/D warmup. Set
MUXSERVE_STARTUP_WARMUP=capacity to explicitly request maximum-load profiles.
"""
from __future__ import annotations

from collections import deque
from dataclasses import replace
import csv
import os
import subprocess
from threading import Event, RLock, Thread
import time

from .contracts import IntegrationError

_MIB = 1024 ** 2


def parse_free_memory(output, uuids):
    """Reject incomplete, duplicate/aliased or inconsistent inventory."""
    selected = set(uuids)
    result = {}
    try:
        for row in csv.reader(output.splitlines()):
            if len(row) != 3:
                raise ValueError("Expected UUID, free MiB, total MiB")
            identity, free, total = (cell.strip() for cell in row)
            free, total = int(free), int(total)
            if identity not in selected or identity in result or not 0 <= free <= total or total <= 0:
                raise ValueError("Inconsistent selected-device inventory")
            result[identity] = {"free_bytes": free * _MIB, "total_bytes": total * _MIB}
        if len(selected) != len(uuids) or set(result) != selected or not selected:
            raise ValueError("Missing or duplicate physical GPU")
    except (TypeError, ValueError) as exc:
        raise IntegrationError("STARTUP_MEMORY_UNAVAILABLE", str(exc)) from exc
    return result


def require_headroom(minimum_free, uuids, required_bytes):
    if (set(minimum_free) != set(uuids) or len(set(uuids)) != len(uuids)
            or not uuids or any(type(value) is not int or value < 0 for value in minimum_free.values())):
        raise IntegrationError("STARTUP_MEMORY_UNAVAILABLE", "Every selected GPU needs real free-memory samples")
    # One extra MiB is an explicit conservative allowance for nvidia-smi units.
    if any(value < required_bytes + _MIB for value in minimum_free.values()):
        raise IntegrationError("STARTUP_HEADROOM_INSUFFICIENT", "Sampled physical free memory is below the reserve")


class MemorySampler:
    interval = 0.1
    query_timeout = 3.0

    def __init__(self, cards, deadline):
        from .preflight import _executable
        self.binary = str(_executable("nvidia-smi"))
        self.uuids = tuple(card["uuid"] for card in cards)
        self.expected_totals = {card["uuid"]: card["total_mib"] * _MIB for card in cards}
        self.deadline = deadline
        self.lock = RLock()
        self.stop_event = Event()
        self.thread = Thread(target=self._run, name="startup-memory-sampler", daemon=False)
        self.failure = None
        self.count = 0
        self.minimum_free = {}
        self.minimum_observations = {}
        self.recent = deque(maxlen=256)
        self.phase = "all_models_resident"

    def sample(self):
        remaining = self.deadline - time.monotonic()
        if remaining <= 0:
            raise IntegrationError("STARTUP_TIMEOUT", "Startup memory sampling deadline expired")
        with self.lock:
            phase = self.phase
        try:
            result = subprocess.run([self.binary, "-i", ",".join(self.uuids),
                "--query-gpu=uuid,memory.free,memory.total", "--format=csv,noheader,nounits"],
                check=True, capture_output=True, text=True, timeout=min(self.query_timeout, remaining),
                env={**os.environ, "CUDA_VISIBLE_DEVICES": ""})
        except (OSError, subprocess.SubprocessError) as exc:
            raise IntegrationError("STARTUP_MEMORY_UNAVAILABLE", "Bounded nvidia-smi sample failed") from exc
        cards = parse_free_memory(result.stdout, self.uuids)
        if any(cards[identity]["total_bytes"] != total for identity, total in self.expected_totals.items()):
            raise IntegrationError("STARTUP_MEMORY_UNAVAILABLE", "Physical GPU totals changed after preflight")
        with self.lock:
            self.count += 1
            observation = {"sample": self.count, "monotonic_s": time.monotonic(), "phase": phase,
                           "cards": cards}
            self.recent.append(observation)
            for identity, card in cards.items():
                if identity not in self.minimum_free or card["free_bytes"] < self.minimum_free[identity]:
                    self.minimum_free[identity] = card["free_bytes"]
                    self.minimum_observations[identity] = observation

    def _run(self):
        try:
            while not self.stop_event.wait(self.interval):
                self.sample()
        except BaseException as exc:
            self.failure = exc

    def start(self):
        self.sample()  # Require a complete selected-card baseline before grants.
        self.thread.start()

    def stop(self):
        self.stop_event.set()
        if self.thread.ident is not None:
            self.thread.join(self.query_timeout + 1)
            if self.thread.is_alive():
                raise IntegrationError("STARTUP_MEMORY_UNAVAILABLE", "Memory sampler did not stop")

    def evidence(self):
        with self.lock:
            return {"sample_count": self.count, "interval_s": self.interval,
                    "query_timeout_s": self.query_timeout, "selected_uuids": self.uuids,
                    "minimum_free_bytes": dict(self.minimum_free),
                    "minimum_observations": dict(self.minimum_observations),
                    "observed_maximum_used_bytes": {
                        key: self.expected_totals[key] - value for key, value in self.minimum_free.items()},
                    "recent_samples": list(self.recent), "retained_sample_limit": 256,
                    "scope": "physical selected GPU memory.free, including all resident weights, shared KV and other processes",
                    "limitation": "sampled minima only; allocation transients between queries may be missed",
                    "quantization_allowance_bytes": _MIB}


class _InitializationController:
    """Gateway lifecycle reuse, with private initialization-only admission."""
    def __init__(self, controller, deadline):
        self.controller = controller
        self.deadline = deadline

    def __getattr__(self, name):
        return getattr(self.controller, name)

    def submit_request(self, spec):
        remaining_ms = min(3_600_000, int((self.deadline - time.monotonic()) * 1000))
        if remaining_ms <= 0:
            raise IntegrationError("STARTUP_TIMEOUT", "No startup budget remains for initialization")
        return self.controller.submit_initialization_request(replace(spec, timeout_ms=remaining_ms))


def warmup_shapes(config, capacity_tokens, *, batch_size=1, mode="capacity"):
    if mode not in ("smoke", "capacity"):
        raise IntegrationError("STARTUP_PROFILE_UNSUPPORTED", "Warmup mode must be smoke or capacity")
    # Capacity is already rounded to physical token blocks by the controller.
    # Divide whole blocks, rather than tokens, so two partial blocks cannot
    # silently exceed a static partition at the last decode step.
    per_request = min(config.max_context_tokens, (capacity_tokens // (16 * batch_size)) * 16)
    if per_request < 1:
        raise IntegrationError("STARTUP_PROFILE_UNSUPPORTED", "KV capacity cannot support the initialization batch")
    input_limit = min(config.max_input_tokens, 32) if mode == "smoke" else config.max_input_tokens
    prefill = min(input_limit, per_request)
    if batch_size > 1:
        # The unchanged ADBS grant path caps a prefill batch at 2048 tokens.
        prefill = min(prefill, 2048 // batch_size)
    outputs = min(config.max_new_tokens, per_request, 8) if mode == "smoke" else min(config.max_new_tokens, per_request)
    decode_input = min(input_limit, per_request - outputs + 1)
    prefix = "smoke" if mode == "smoke" else "maximum"
    return ((prefix + "_prefill", prefill, 1), (prefix + "_decode", decode_input, outputs))


def run_initialization(controller, tokenizers, cards, deadline, observe, stop_requested, write_evidence,
                       *, warmup_profile=None):
    """Finish/ACK/forget dedicated requests; return drained state and driver.

    The driver is pumped synchronously and its thread is never started here.
    Manual mode therefore has no background scheduler executing user requests.
    """
    from .driver import ControllerDriver
    from .gateway import GatewayState
    if warmup_profile is None:
        warmup_profile = os.environ.get("MUXSERVE_STARTUP_WARMUP", "smoke")
    if warmup_profile not in ("smoke", "capacity"):
        raise IntegrationError("STARTUP_PROFILE_UNSUPPORTED", "Warmup mode must be smoke or capacity")
    state = GatewayState(_InitializationController(controller, deadline), tokenizers)
    held_decode = set()
    class InitializationDriver(ControllerDriver):
        def _prepare(self, proposal):
            if proposal.stage == "D" and proposal.model_id in held_decode:
                return
            super()._prepare(proposal)
    driver = InitializationDriver(controller, state.apply_event, state.maintain)
    sampler = None
    config = controller.config
    evidence = {"passed": False, "config_fingerprint": config.fingerprint,
                "required_free_bytes_per_card": config.extra_headroom_bytes_per_rank,
                "resident_weights_and_kv_subtracted_again": False, "profiles": [],
                "request_capacity": {}, "initialization_only": True,
                "request_timeout_policy": "remaining_global_startup_budget",
                "serving_request_timeout_ms": config.request_timeout_ms,
                "startup_deadline_monotonic_s": deadline,
                "warmup_profile": warmup_profile,
                "maximum_load_requested": warmup_profile == "capacity",
                "maximum_load_validated": False,
                "scope": ("Short functional P/D warmup; maximum load and concurrency are not validated"
                          if warmup_profile == "smoke" else "Explicit capacity warmup")}
    saved_limits = dict(controller.policy.limits)
    saved_adjustment = controller.policy.last_adjustment
    try:
        sampler = MemorySampler(cards, deadline)
        sampler.start()
        require_headroom(sampler.minimum_free, sampler.uuids, config.extra_headroom_bytes_per_rank)
        tokens = {}
        for model in config.models:
            capacity = controller.request_capacity(model.model_id)
            capacity["physical_token_capacity"] = (
                min(capacity["blocks_by_rank"].values()) // capacity["blocks_per_token_group"]) * 16
            evidence["request_capacity"][model.model_id] = capacity
            with state.tokenizer_lock:
                legal = tokenizers[model.model_id].encode("Initialization warmup.", add_special_tokens=True)
            with controller.lock:
                ranks = controller.groups[model.model_id].registered.values()
                vocabulary = min(rank.capabilities["vocab_size"] for rank in ranks)
            legal = tuple(token for token in legal if type(token) is int and 0 <= token < vocabulary)
            if not legal:
                raise IntegrationError("STARTUP_PROFILE_UNSUPPORTED", "Tokenizer produced no legal initialization IDs")
            tokens[model.model_id] = legal

        def check():
            observe()
            if sampler.failure:
                raise sampler.failure
            with sampler.lock:
                minimum_free = dict(sampler.minimum_free)
            require_headroom(minimum_free, sampler.uuids, config.extra_headroom_bytes_per_rank)
            if stop_requested.is_set() or time.monotonic() >= deadline:
                raise IntegrationError("STARTUP_TIMEOUT", "Initialization requests did not finish within startup deadline")

        def phase(name, model_ids, batch_size):
            check()
            with sampler.lock:
                sampler.phase = name
            sampler.sample()
            profile = {"name": name, "batch_size": batch_size, "requests": [], "started_batch_size": {}}
            evidence["profiles"].append(profile)
            records = []
            with controller.lock:
                for model_id in model_ids:
                    shape_index = 0 if "prefill" in name else 1
                    capacity = evidence["request_capacity"][model_id]
                    budget = capacity["physical_token_capacity"] if batch_size > 1 else capacity["capacity_tokens"]
                    _, inputs, outputs = warmup_shapes(config, budget, batch_size=batch_size, mode=warmup_profile)[shape_index]
                    seed = tokens[model_id]
                    ids = [seed[index % len(seed)] for index in range(inputs)]
                    for index in range(batch_size):
                        record = state.submit(model_id, {"rid": f"initialization-{name}-{model_id}-{index}",
                            "stream": True, "input_ids": ids,
                            "sampling_params": {"temperature": 0, "top_k": 1, "top_p": 1,
                                                "max_new_tokens": outputs, "ignore_eos": True}})
                        records.append((record, outputs))
                        profile["requests"].append({"model_id": model_id, "request_id": record.internal_id,
                            "input_tokens": inputs, "max_new_tokens": outputs, "maximum_kv_tokens": inputs + outputs - 1,
                            "request_deadline_monotonic_s": record.deadline})
                if batch_size > 1 and "decode" in name:
                    # Two long prompts can exceed ADBS's prefill-token ceiling.
                    # Grant their normal P work separately, then allow a real
                    # D2 grant after both P results have been acknowledged.
                    held_decode.update(model_ids)
                    profile["decode_held_until_all_prefilled"] = True
            while True:
                check()
                # Reports cannot race past the RUNNING observation while this
                # owner lock is held; this proves actual granted batch shapes.
                with controller.lock:
                    if records and all(record.state == "DECODING" for record, _ in records):
                        held_decode.difference_update(model_ids)
                    driver.step()
                    for model_id in model_ids:
                        for batch in controller.groups[model_id].batches.values():
                            if batch.status == "RUNNING":
                                key = f"{model_id}:{batch.plan.stage}"
                                profile["started_batch_size"][key] = max(
                                    profile["started_batch_size"].get(key, 0), len(batch.plan.requests))
                    for record, outputs in records:
                        if record.error or record.state in ("FAILED", "CANCELLED", "TIMED_OUT"):
                            raise IntegrationError("STARTUP_WARMUP_FAILED", record.error or record.state)
                        if record.state == "FINISHED" and len(record.tokens) != outputs:
                            raise IntegrationError("STARTUP_WARMUP_FAILED", "Greedy ignore_eos initialization ended early")
                        if record.forgotten:
                            record.attached = False
                    state.maintain()
                    complete = all(record.forgotten for record, _ in records)
                if complete:
                    break
                stop_requested.wait(0.005)
            for model_id in model_ids:
                required_stages = ("P", "D") if "decode" in name and records[0][1] > 1 else ("P",)
                for stage in required_stages:
                    expected_size = 1 if stage == "P" and "decode" in name else batch_size
                    if profile["started_batch_size"].get(f"{model_id}:{stage}", 0) < expected_size:
                        raise IntegrationError("STARTUP_PROFILE_UNSUPPORTED", "Required initialization batch shape was not granted")
            profile["finished_and_forgotten"] = True
            sampler.sample()

        # All three models remain resident during initialization.
        # Warm up each model separately to avoid consuming request
        # deadlines while waiting for another model's GPU quota.
        phase_names = ("smoke_prefill", "smoke_decode") if warmup_profile == "smoke" else ("maximum_prefill", "maximum_decode")
        for model in config.models:
            for name in phase_names:
                phase(name, (model.model_id,), 1)
        if config.max_model_inflight >= 2:
            with controller.lock:
                controller.policy.last_adjustment = deadline
            evidence["initialization_batch2_limit_override"] = {"original_limits": saved_limits, "temporary_limit": 2}
            for model in config.models:
                with controller.lock:
                    controller.policy.limits[model.model_id] = 2
                for name in ("batch2_prefill", "batch2_decode"):
                    phase(name, (model.model_id,), 2)
        check()
        with controller.lock:
            driver.step()  # Final ACK/feedback and any completed batch retirement.
            snapshot = controller.snapshot()
            if (snapshot["requests"] or state.records or driver.pending
                    or any(group["pending_events"] or group["batches"]
                           or any(rank["pending_commands"] for rank in group["ranks"].values())
                           for group in snapshot["groups"].values())
                    or controller.store.snapshot()["allocations"]):
                raise IntegrationError("STARTUP_WARMUP_FAILED", "Initialization lifecycle is not fully drained")
        with sampler.lock:
            sampler.phase = "post_warmup_drained"
        sampler.sample()
        sampler.stop()
        if sampler.failure:
            raise sampler.failure
        check()
        require_headroom(sampler.minimum_free, sampler.uuids, config.extra_headroom_bytes_per_rank)
        maximum_shapes = {(profile["name"], request["model_id"], request["input_tokens"], request["max_new_tokens"])
            for profile in evidence["profiles"] if profile["name"] in ("maximum_prefill", "maximum_decode")
            for request in profile["requests"]}
        expected_maximum_shapes = {(name, model.model_id, config.max_input_tokens, outputs)
            for model in config.models for name, outputs in
            (("maximum_prefill", 1), ("maximum_decode", config.max_new_tokens))}
        evidence["maximum_load_validated"] = warmup_profile == "capacity" and maximum_shapes == expected_maximum_shapes
        evidence["passed"] = True
        return state, driver
    except BaseException as exc:
        evidence["failure"] = {"code": getattr(exc, "code", type(exc).__name__), "message": str(exc)}
        driver.pause()
        raise
    finally:
        with controller.lock:
            controller.policy.limits.update(saved_limits)
            controller.policy.last_adjustment = saved_adjustment
        try:
            if sampler is not None:
                sampler.stop()
        finally:
            evidence["memory"] = sampler.evidence() if sampler is not None else {"sample_count": 0}
            write_evidence(evidence)
