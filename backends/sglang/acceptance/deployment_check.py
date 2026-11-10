"""User-run CPU deployment regression source; UNEXECUTED during development.

Run from the repository root: python acceptance/deployment_check.py
Uses the real CPU controller/store and acceptance-only rank reports. Opens
temporary loopback servers; imports no Torch/SGLang and starts no GPU owners.
Software discovery uses temporary fixture files; tools remain mocked. This does
not validate a real installation, ABI, GPU behavior, or the host toolchain.
"""
from __future__ import annotations

from dataclasses import asdict
from contextlib import contextmanager
from http.client import HTTPConnection
import json
import os
from pathlib import Path
import socket
import sys
from tempfile import TemporaryDirectory
from threading import Condition, RLock, Event, Thread
import time
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from controller_check import Fixture
from muxserve_sglang.config import load_config
from muxserve_sglang import config as deployment_config
from muxserve_sglang.contracts import IntegrationError
from muxserve_sglang.gateway import GatewayServer, GatewayState, OutputRecord
from muxserve_sglang.launcher import Supervisor, _native_fields
from muxserve_sglang import preflight
from muxserve_sglang import startup
from muxserve_sglang import driver as controller_driver
from tools import lock_assets


def gateway_fixture():
    fixture = Fixture()
    tokenizer = SimpleNamespace(decode=lambda tokens, **kwargs: " ".join(map(str, tokens)))
    return fixture, GatewayState(fixture.controller, {"m": tokenizer})


def submit(state):
    return state.submit("m", {"stream": True, "input_ids": [1, 2]})


def deliver(fixture, state, *, ack=True):
    after_seq = fixture.controller.snapshot()["groups"]["m"]["acked_event_seq"]
    events = fixture.controller.poll_events(fixture.group, after_seq)
    for event in events:
        state.apply_event(event)
    if ack and events:
        fixture.controller.ack_events(fixture.group, events[-1].event_seq)


def start_server(state, *, short=False):
    server = GatewayServer(state, 0, maximum_connections=1)
    if short:
        server.admission_timeout = server.write_timeout = 0.2
    thread = Thread(target=lambda: server.serve_forever(poll_interval=0.01), daemon=True)
    thread.start()
    return server, thread


class DeploymentChecks(unittest.TestCase):
    def assert_code(self, code, callback):
        with self.assertRaises(IntegrationError) as caught:
            callback()
        self.assertEqual(caught.exception.code, code)

    @unittest.skipUnless(os.name == "posix", "Linux asset path contract")
    def test_asset_readers_resolve_alias_and_keep_personal_locks(self):
        with TemporaryDirectory() as directory:
            base = Path(directory).resolve()
            target, alias, locks = base / "stored-model", base / "public-model", base / "personal-locks"
            target.mkdir()
            (target / "config.json").write_text('{"model_type": "llama"}', encoding="utf-8")
            (target / "weights.bin").write_bytes(b"fixture weights")
            self.link_or_skip(alias, target, directory=True)
            before = {path.name: path.read_bytes() for path in target.iterdir()}
            model = SimpleNamespace(model_path=str(alias), tokenizer_path=str(alias),
                                    revision="a" * 40, tokenizer_revision="a" * 40)
            config = SimpleNamespace(models=(model,), asset_lock_dir=str(locks))
            self.assertEqual(deployment_config.read_asset_path(alias), target)
            self.assertEqual(preflight._asset_file(alias, "weights.bin"), target / "weights.bin")
            self.assertEqual(preflight._read_json(alias / "config.json"), {"model_type": "llama"})
            self.assertEqual(preflight._inventory_names(alias), set(before))
            self.assert_code("INVALID_CONFIG", lambda: deployment_config.asset_lock_path(
                alias, str(target / "nested-locks")))
            with patch.object(lock_assets, "load_config", return_value=config), patch.object(
                    sys, "argv", ["lock_assets", "unused.json"]):
                lock_assets.main()
                lock_assets.main()
            lock = deployment_config.asset_lock_path(alias, str(locks))
            self.assertEqual(json.loads(lock.read_text())["asset_root"], str(target))
            preflight._check_lock(alias, "a" * 40, set(before), str(locks))
            self.assertEqual(before, {path.name: path.read_bytes() for path in target.iterdir()})
            self.assertEqual(len(list(locks.iterdir())), 1)

    @unittest.skipUnless(os.name == "posix", "Linux asset path contract")
    def test_asset_links_resolve_and_prohibited_targets_are_rejected(self):
        with TemporaryDirectory() as directory:
            base = Path(directory).resolve()
            target, alias = base / "stored-model", base / "public-model"
            target.mkdir()
            (target / "weights.bin").write_bytes(b"fixture")
            self.link_or_skip(alias, target, directory=True)
            self.link_or_skip(target / "inside.bin", target / "weights.bin")
            self.link_or_skip(target / "blocked.bin", Path("/home/incoming/model/weights.bin"))
            self.link_or_skip(target / "loop.bin", Path("loop.bin"))
            self.assertEqual(deployment_config.read_asset_path(alias / "inside.bin"), target / "weights.bin")
            self.assert_code("FORBIDDEN_PATH", lambda: deployment_config.read_asset_path(alias / "blocked.bin"))
            self.assert_code("INVALID_PATH", lambda: deployment_config.read_asset_path(alias / "loop.bin"))

    def test_external_asset_lock_never_writes_model_directory(self):
        with TemporaryDirectory() as directory:
            root = Path(directory) / "shared-model"
            root.mkdir()
            (root / "weights.bin").write_bytes(b"fixture weights")
            locks = Path(directory) / "personal-locks"
            revision = "a" * 40
            model = SimpleNamespace(model_path=str(root), tokenizer_path=str(root),
                                    revision=revision, tokenizer_revision=revision)
            config = SimpleNamespace(models=(model,), asset_lock_dir=str(locks))
            before = {path.name: path.read_bytes() for path in root.iterdir()}
            with patch.object(lock_assets, "load_config", return_value=config), \
                    patch.object(sys, "argv", ["lock_assets", "unused.json"]):
                lock_assets.main()
                lock_assets.main()  # Validate/reuse, never overwrite an existing lock.
            self.assertEqual(before, {path.name: path.read_bytes() for path in root.iterdir()})
            self.assertEqual(len(list(locks.iterdir())), 1)
            preflight._check_lock(root, revision, {"weights.bin"}, str(locks))
            (root / "weights.bin").write_bytes(b"changed weights")
            self.assert_code("ASSET_CHANGED", lambda: preflight._check_lock(
                root, revision, {"weights.bin"}, str(locks)))
            (root / "weights.bin").write_bytes(b"fixture weights")
            lock = next(locks.iterdir())
            payload = json.loads(lock.read_text())
            payload["asset_root"] = str(root.parent / "different-model")
            lock.write_text(json.dumps(payload))
            self.assert_code("ASSET_NOT_FROZEN", lambda: preflight._check_lock(
                root, revision, {"weights.bin"}, str(locks)))

    def test_asset_lock_directory_required_and_outside_assets(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            model = SimpleNamespace(model_path=str(root), tokenizer_path=str(root),
                                    revision="a" * 40, tokenizer_revision="a" * 40)
            config = SimpleNamespace(models=(model,), asset_lock_dir="")
            with patch.object(lock_assets, "load_config", return_value=config), \
                    patch.object(sys, "argv", ["lock_assets", "unused.json"]):
                self.assert_code("INVALID_CONFIG", lock_assets.main)
                config.asset_lock_dir = str(root / "locks")
                self.assert_code("INVALID_CONFIG", lock_assets.main)
            self.assertEqual(list(root.iterdir()), [])

    def test_startup_memory_requires_complete_uuid_map_and_real_reserve(self):
        uuids = ("GPU-a", "GPU-b")
        cards = startup.parse_free_memory("GPU-b, 3000, 40536\nGPU-a, 2500, 40536", uuids)
        self.assertEqual(cards["GPU-a"]["free_bytes"], 2500 * 1024**2)
        for output in ("", "GPU-a, 3000, 40536", "GPU-a, 3000, 40536\nGPU-a, 3000, 40536",
                       "GPU-a, 50000, 40536\nGPU-b, 3000, 40536",
                       "GPU-a, N/A, 40536\nGPU-b, 3000, 40536",
                       "GPU-a, 3000, 40536\nGPU-c, 3000, 40536"):
            self.assert_code("STARTUP_MEMORY_UNAVAILABLE", lambda: startup.parse_free_memory(output, uuids))
        required = 2 * 1024**3
        self.assert_code("STARTUP_MEMORY_UNAVAILABLE", lambda: startup.require_headroom({}, uuids, required))
        self.assert_code("STARTUP_MEMORY_UNAVAILABLE", lambda: startup.require_headroom(
            {"GPU-a": required * 2}, uuids, required))
        self.assert_code("STARTUP_HEADROOM_INSUFFICIENT", lambda: startup.require_headroom(
            {"GPU-a": required * 2, "GPU-b": required - 1}, uuids, required))
        self.assert_code("STARTUP_HEADROOM_INSUFFICIENT", lambda: startup.require_headroom(
            dict.fromkeys(uuids, required), uuids, required))
        startup.require_headroom(dict.fromkeys(uuids, required + 1024**2), uuids, required)

    def test_startup_profiles_fit_static_capacity_and_batch_block_rounding(self):
        config = SimpleNamespace(max_input_tokens=1792, max_new_tokens=256, max_context_tokens=2048)
        self.assertEqual(startup.warmup_shapes(config, 1792),
                         (("maximum_prefill", 1792, 1), ("maximum_decode", 1537, 256)))
        for capacity in (1792, 1808, 2048):
            for batch_size in (1, 2):
                for _, inputs, outputs in startup.warmup_shapes(config, capacity, batch_size=batch_size):
                    physical_tokens = ((inputs + outputs - 1 + 15) // 16) * 16 * batch_size
                    self.assertLessEqual(physical_tokens, capacity)
                    self.assertLessEqual(inputs, config.max_input_tokens)
                    if batch_size == 2 and outputs == 1:
                        self.assertLessEqual(inputs * batch_size, 2048)
        self.assertEqual(startup.warmup_shapes(config, 8192, batch_size=2),
                         (("maximum_prefill", 1024, 1), ("maximum_decode", 1792, 256)))
        self.assert_code("STARTUP_PROFILE_UNSUPPORTED", lambda: startup.warmup_shapes(config, 16, batch_size=2))

    def test_startup_sampler_queries_selected_uuids_and_rejects_timeout(self):
        cards = [{"uuid": "GPU-a", "total_mib": 40536}]
        with patch.object(preflight, "_executable", return_value=Path("/usr/bin/nvidia-smi")), \
                patch.object(startup.subprocess, "run") as query:
            sampler = startup.MemorySampler(cards, time.monotonic() + 30)
            query.return_value = SimpleNamespace(stdout="GPU-a, 3000, 40536")
            sampler.sample()
            query.return_value = SimpleNamespace(stdout="GPU-a, 2100, 40536")
            sampler.sample()
            self.assertEqual(sampler.minimum_free, {"GPU-a": 2100 * 1024**2})
            self.assertEqual(sampler.evidence()["sample_count"], 2)
            self.assertEqual(query.call_args.args[0][2], "GPU-a")
            self.assertLessEqual(query.call_args.kwargs["timeout"], 3)
            query.side_effect = startup.subprocess.TimeoutExpired("nvidia-smi", 3)
            self.assert_code("STARTUP_MEMORY_UNAVAILABLE", sampler.sample)

    def initialization_fixture(self, *, fail=False):
        """Mock GPU reports only; reuse the real GatewayState lifecycle/forget barrier."""
        config = SimpleNamespace(models=(SimpleNamespace(model_id="m"),), fingerprint="fixture",
            max_model_inflight=1, max_input_tokens=8, max_new_tokens=4, max_context_tokens=64, queue_limit=128,
            request_timeout_ms=12000, extra_headroom_bytes_per_rank=2 * 1024**3, model=lambda _: None)
        owner = SimpleNamespace(config=config, lock=RLock(), clock=time.monotonic,
            policy=SimpleNamespace(limits={"m": 1}, last_adjustment=0), requests={}, acked=False,
            groups={"m": SimpleNamespace(registered={0: SimpleNamespace(capabilities={"vocab_size": 32})}, batches={})})
        owner.submit_request = Mock(side_effect=AssertionError("Public admission during initialization"))
        owner.request_capacity = lambda _: {"capacity_tokens": 64, "pool_mode": "static",
            "blocks_by_rank": {0: 4}, "blocks_per_token_group": 1}
        def admit(spec):
            owner.requests[spec.request_id] = spec
            return {"deadline": time.monotonic() + 12}
        owner.submit_initialization_request = Mock(side_effect=admit)
        def forget(model_id, identity, attempt):
            if not owner.acked:
                raise IntegrationError("NOT_READY", "Mock command/event ACK still pending")
            del owner.requests[identity]
        owner.forget_request = Mock(side_effect=forget)
        owner.store = SimpleNamespace(snapshot=lambda: {"allocations": {}})
        owner.snapshot = lambda: {"requests": dict(owner.requests), "groups": {"m": {
            "pending_events": 0, "batches": dict(owner.groups["m"].batches), "ranks": {}}}}
        drivers = []
        class FakeDriver:
            def __init__(self, controller, apply_event, maintain):
                self.state = apply_event.__self__
                self.pending = set()
                self.start = Mock(side_effect=AssertionError("Initialization driver must remain synchronous"))
                self.pause = Mock()
                drivers.append(self)
            def step(self):
                records = list(self.state.records.values())
                if not records:
                    return
                if fail:
                    raise IntegrationError("MOCK_NATIVE_FAILURE", "No release proof")
                if all(record.state == "FINISHED" for record in records):
                    owner.acked = True
                    owner.groups["m"].batches.clear()
                    self.state.maintain()
                    return
                owner.acked = False
                stage = "P" if records[0].state == "QUEUED" else "D"
                owner.groups["m"].batches["mock"] = SimpleNamespace(status="RUNNING",
                    plan=SimpleNamespace(stage=stage, requests=records))
                for record in records:
                    spec = owner.requests[record.internal_id]
                    count = 1 if stage == "P" else spec.max_new_tokens
                    record.tokens = [1] * count
                    record.state = "FINISHED" if count == spec.max_new_tokens else "DECODING"
        sampler = Mock(uuids=("GPU-a",), failure=None, minimum_free={"GPU-a": 3 * 1024**3}, lock=RLock())
        sampler.evidence.return_value = {"sample_count": 4, "minimum_free_bytes": sampler.minimum_free}
        tokenizer = SimpleNamespace(encode=lambda *a, **k: [1, 2, -1, 999])
        return owner, FakeDriver, drivers, sampler, tokenizer

    def test_startup_requests_are_private_drained_acked_and_forgotten(self):
        owner, driver_type, drivers, sampler, tokenizer = self.initialization_fixture()
        evidence = []
        with patch.object(controller_driver, "ControllerDriver", driver_type), \
                patch.object(startup, "MemorySampler", return_value=sampler):
            state, driver = startup.run_initialization(owner, {"m": tokenizer}, [],
                time.monotonic() + 10, lambda: None, Event(), evidence.append)
        self.assertIs(driver, drivers[0])
        driver.start.assert_not_called()
        owner.submit_request.assert_not_called()
        self.assertEqual(owner.submit_initialization_request.call_count, 2)
        self.assertEqual(owner.forget_request.call_count, 4)  # One NOT_READY then one confirmed ACK per request.
        self.assertFalse(owner.requests)
        self.assertFalse(state.records)
        self.assertTrue(evidence[0]["passed"])
        for call in owner.submit_initialization_request.call_args_list:
            spec = call.args[0]
            self.assertTrue(spec.sampling["ignore_eos"])
            self.assertTrue(set(spec.input_ids) <= {1, 2})
        self.assertEqual(owner.policy.limits, {"m": 1})
        sampler.stop.assert_called()

    def test_startup_failure_keeps_unproved_requests_and_records_failed_evidence(self):
        owner, driver_type, drivers, sampler, tokenizer = self.initialization_fixture(fail=True)
        evidence = []
        with patch.object(controller_driver, "ControllerDriver", driver_type), \
                patch.object(startup, "MemorySampler", return_value=sampler):
            self.assert_code("MOCK_NATIVE_FAILURE", lambda: startup.run_initialization(
                owner, {"m": tokenizer}, [], time.monotonic() + 10, lambda: None, Event(), evidence.append))
        self.assertTrue(owner.requests)
        owner.forget_request.assert_not_called()
        drivers[0].pause.assert_called_once_with()
        self.assertFalse(evidence[0]["passed"])
        self.assertEqual(evidence[0]["failure"]["code"], "MOCK_NATIVE_FAILURE")
        self.assertEqual(owner.policy.last_adjustment, 0)

    def test_startup_batch2_requires_real_grant_and_restores_initialization_override(self):
        owner, driver_type, drivers, sampler, tokenizer = self.initialization_fixture()
        owner.config.max_model_inflight = 2
        evidence = []
        with patch.object(controller_driver, "ControllerDriver", driver_type), \
                patch.object(startup, "MemorySampler", return_value=sampler):
            startup.run_initialization(owner, {"m": tokenizer}, [],
                time.monotonic() + 10, lambda: None, Event(), evidence.append)
        self.assertTrue(evidence[0]["passed"])
        self.assertEqual(len(evidence[0]["profiles"]), 4)
        prefill, decode = evidence[0]["profiles"][-2:]
        self.assertEqual(prefill["started_batch_size"]["m:P"], 2)
        self.assertEqual(decode["started_batch_size"]["m:D"], 2)
        self.assertTrue(decode["decode_held_until_all_prefilled"])
        self.assertEqual(owner.policy.limits, {"m": 1})
        self.assertEqual(owner.policy.last_adjustment, 0)
        drivers[0].start.assert_not_called()

    def test_manual_ready_requires_gate_and_never_starts_initialization_driver(self):
        supervisor = object.__new__(Supervisor)
        supervisor.manual_control = True
        supervisor.startup_gate_complete = False
        supervisor.controller = Mock()
        supervisor.config = SimpleNamespace(models=(SimpleNamespace(model_id="m", tokenizer_path="fixture"),))
        supervisor.cards, supervisor.startup_deadline = [], time.monotonic() + 10
        supervisor.stop_requested, supervisor._observe = Event(), Mock()
        supervisor.directory, supervisor.driver = Path("unused"), None
        supervisor._status = Mock()
        initialization_driver = Mock()
        tokenizer = Mock()
        transformers = SimpleNamespace(AutoTokenizer=SimpleNamespace(from_pretrained=Mock(return_value=tokenizer)))
        with patch.dict(sys.modules, {"transformers": transformers}), \
                patch.object(startup, "run_initialization", return_value=(Mock(), initialization_driver)), \
                patch("muxserve_sglang.gateway.GatewayServer") as listener:
            supervisor._initialize()
        initialization_driver.start.assert_not_called()
        initialization_driver.pause.assert_called_once_with()
        supervisor.controller.finish_initialization.assert_called_once_with()
        listener.assert_not_called()
        self.assertIsNone(supervisor.driver)
        self.assertTrue(supervisor.startup_gate_complete)
        supervisor.startup_gate_complete = False
        self.assert_code("STARTUP_NOT_READY", lambda: Supervisor._status(supervisor, "READY"))

    def test_gpu_inventory_requires_the_fixed_40gb_hardware_class(self):
        def rows(total, free=35000):
            return "\n".join(f"{rank}, GPU-test-{rank}, NVIDIA A100, 8.0, {free}, {total}"
                             for rank in range(4))
        with patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "0,1,2,3"}), \
                patch.object(preflight, "_executable", return_value=Path("/usr/bin/nvidia-smi")), \
                patch.object(preflight.subprocess, "run") as discover:
            discover.return_value = SimpleNamespace(stdout=rows(40536))
            self.assertEqual(len(preflight.inspect_gpus(4)), 4)
            for total in (81920, 20000):
                discover.return_value = SimpleNamespace(stdout=rows(total))
                self.assert_code("GPU_MAPPING_REQUIRED", lambda: preflight.inspect_gpus(4))
            discover.return_value = SimpleNamespace(stdout=rows(40536, 50000))
            self.assert_code("GPU_PREFLIGHT_FAILED", lambda: preflight.inspect_gpus(4))

    def suspend(self, fixture, state, record):
        state.control("m", record.rid, "suspend")
        fixture.cleanup(record.internal_id)
        deliver(fixture, state)
        fixture.ack_all_commands()

    def assert_retired(self, fixture, state, record):
        state.maintain()
        self.assertTrue(record.forgotten)
        self.assertNotIn(("m", record.internal_id), fixture.controller.snapshot()["requests"])
        self.assertFalse(fixture.controller.records)
        self.assertFalse(fixture.store.snapshot()["allocations"])
        state.disconnect(record)
        state.maintain()
        self.assertFalse(state.records)
        self.assertFalse(state.aliases)

    def test_suspend_disconnect_and_cancel(self):
        for operation in ("disconnect", "cancel"):
            with self.subTest(operation=operation):
                fixture, state = gateway_fixture()
                record = submit(state)
                self.suspend(fixture, state, record)
                if operation == "disconnect":
                    state.disconnect(record)
                else:
                    state.control("m", record.rid, "cancel")
                self.assertEqual(record.state, "SUSPENDED")
                self.assertTrue(record.error)
                state.maintain()
                self.assertFalse(record.forgotten)
                fixture.cleanup(record.internal_id)
                deliver(fixture, state)
                fixture.ack_all_commands()
                self.assert_retired(fixture, state, record)

    def test_disconnect_during_suspension_obeys_release_and_ack_barriers(self):
        fixture, state = gateway_fixture()
        record = submit(state)
        state.control("m", record.rid, "suspend")
        state.disconnect(record)
        state.maintain()
        self.assertFalse(record.forgotten)
        fixture.cleanup(record.internal_id)
        deliver(fixture, state, ack=False)
        self.assertFalse(record.forgotten)  # Suspension event and commands unacked.
        fixture.ack_all_commands()
        state.maintain()
        self.assertFalse(record.forgotten)  # Cancellation has its own release barrier.
        events = fixture.controller.poll_events(fixture.group, 0)
        fixture.controller.ack_events(fixture.group, events[-1].event_seq)
        state.maintain()
        self.assertFalse(record.forgotten)  # Completed suspension now needs terminal cleanup.
        fixture.cleanup(record.internal_id)
        deliver(fixture, state, ack=False)
        state.maintain()
        self.assertFalse(record.forgotten)  # Physical release alone cannot retire the request.
        fixture.ack_all_commands()
        state.maintain()
        self.assertFalse(record.forgotten)  # Terminal event ACK remains independently required.
        deliver(fixture, state)
        self.assert_retired(fixture, state, record)

    def test_expired_suspend_rejects_resume_and_retires(self):
        fixture, state = gateway_fixture()
        record = submit(state)
        original = record.deadline
        self.suspend(fixture, state, record)
        fixture.clock.now = original
        self.assert_code("INVALID_STATE", lambda: state.control("m", record.rid, "resume"))
        self.assertEqual(record.deadline, original)
        self.assertEqual(record.error, "timeout")
        state.maintain()
        self.assertFalse(record.forgotten)
        fixture.cleanup(record.internal_id)
        deliver(fixture, state)
        fixture.ack_all_commands()
        self.assert_retired(fixture, state, record)

    def test_a_token_wakes_only_its_own_stream(self):
        fixture, state = gateway_fixture()
        first, second = (OutputRecord("m", name, "id-" + name, 2, 1, wake=Condition(state._lock)) for name in "ab")
        state.records.update({first.internal_id: first, second.internal_id: second})
        for record in (first, second):
            record.wake.notify_all = Mock()
        state.condition.notify_all = Mock()
        state.apply_event(SimpleNamespace(kind="TOKEN", request_id=first.internal_id, attempt=0, model_id="m",
                                          payload={"output_index": 0, "token_ids": (7,)}))
        self.assertEqual(first.tokens, [7])
        first.wake.notify_all.assert_called_once_with()
        second.wake.notify_all.assert_not_called()
        state.condition.notify_all.assert_not_called()
        state._wake_all()  # shutdown and group errors still reach every stream
        second.wake.notify_all.assert_called_once_with()

    def test_gateway_counters_separate_admission_and_confirmed_terminal(self):
        fixture, state = gateway_fixture()
        state.count_http("generate_received")
        record = submit(state)
        self.assertEqual(state.http_metrics()["counters"]["generate_admitted"], 1)
        state.control("m", record.rid, "cancel")
        fixture.cleanup(record.internal_id)
        deliver(fixture, state, ack=False)
        counters = state.http_metrics()["counters"]
        self.assertEqual(counters["generate_received"], 1)
        self.assertEqual(counters["terminal_other"], 1)
        deliver(fixture, state)  # Terminal replay cannot increment the count twice.
        self.assertEqual(state.http_metrics()["counters"], counters)

    def test_live_resume_keeps_original_deadline(self):
        fixture, state = gateway_fixture()
        record = submit(state)
        original = record.deadline
        self.suspend(fixture, state, record)
        fixture.clock.now += 1
        result = state.control("m", record.rid, "resume")
        self.assertEqual(result["deadline"], original)
        self.assertEqual(record.deadline, original)
        self.assertEqual(result["attempt"], 1)

    def test_drain_attached_suspended_stream_emits_error_without_done(self):
        fixture, state = gateway_fixture()
        server, thread = start_server(state)
        client = HTTPConnection("127.0.0.1", server.server_port, timeout=2)
        try:
            client.request("POST", "/models/m/generate",
                           json.dumps({"stream": True, "input_ids": [1, 2]}),
                           {"Content-Type": "application/json"})
            response = client.getresponse()
            self.assertEqual(response.status, 200)
            record = next(iter(state.records.values()))
            self.suspend(fixture, state, record)
            server.begin_drain()
            body = response.read()
            self.assertIn(b"supervisor_shutdown", body)
            self.assertNotIn(b"[DONE]", body)
            server.server_close()
            thread.join(1)
            self.assertFalse(thread.is_alive())
            state.maintain()
            fixture.cleanup(record.internal_id)
            deliver(fixture, state)
            fixture.ack_all_commands()
            self.assert_retired(fixture, state, record)
        finally:
            client.close()
            if thread.is_alive():
                server.begin_drain()
            server.server_close()

    def test_idle_partial_and_trickled_admission_recovers_slot(self):
        cases = {
            "idle": b"",
            "partial_header": b"GET /status HTTP/1.1\r\nHost:",
            "trickled_header": b"GET /status HTTP/1.1\r\nX-Slow:",
            "trickled_body": b"POST /models/m/generate HTTP/1.1\r\nContent-Length: 1000\r\n\r\n{",
        }
        for name, prefix in cases.items():
            with self.subTest(case=name):
                _, state = gateway_fixture()
                server, thread = start_server(state, short=True)
                server.handle_error = Mock()
                client = socket.create_connection(("127.0.0.1", server.server_port), timeout=1)
                try:
                    if prefix:
                        client.sendall(prefix)
                    start = time.monotonic()
                    if name.startswith("trickled"):
                        while time.monotonic() - start < 0.5:
                            try:
                                client.sendall(b" ")
                            except OSError:
                                break
                            time.sleep(0.03)  # Below idle timeout; exceeds total deadline.
                    else:
                        time.sleep(0.4)
                    client.settimeout(1)
                    while client.recv(4096):
                        pass
                    self.assertLess(time.monotonic() - start, 1.5)
                    probe = HTTPConnection("127.0.0.1", server.server_port, timeout=1)
                    try:
                        probe.request("GET", "/status")
                        self.assertEqual(probe.getresponse().status, 200)
                    finally:
                        probe.close()
                finally:
                    client.close()
                    server.begin_drain()
                    server.server_close()
                    thread.join(1)
                    self.assertFalse(thread.is_alive())
                    server.handle_error.assert_not_called()

    def test_drain_interrupts_incomplete_admission(self):
        for prefix in (b"", b"GET /status HTTP/1.1\r\nHost:",
                       b"POST /models/m/generate HTTP/1.1\r\nContent-Length: 100\r\n\r\n{"):
            _, state = gateway_fixture()
            server, thread = start_server(state)
            server.handle_error = Mock()
            client = socket.create_connection(("127.0.0.1", server.server_port), timeout=1)
            try:
                if prefix:
                    client.sendall(prefix)
                time.sleep(0.05)
                start = time.monotonic()
                server.begin_drain()
                server.server_close()
                thread.join(1)
                self.assertFalse(thread.is_alive())
                self.assertLess(time.monotonic() - start, 2)
            finally:
                client.close()
                if thread.is_alive():
                    server.begin_drain()
                server.server_close()
                server.handle_error.assert_not_called()

    def test_timeout_boundaries(self):
        raw = json.loads((ROOT / "configs/sglang_a100.json").read_text())
        with TemporaryDirectory() as directory:
            file = Path(directory) / "config.json"
            for request, prepare, valid in ((999, 30000, False), (1000, 120000, True),
                                             (120000, 120001, False), (300000, 120000, True),
                                             (300001, 30000, False)):
                raw.update(request_timeout_ms=request, prepare_timeout_ms=prepare)
                file.write_text(json.dumps(raw))
                if valid:
                    config = load_config(file, require_assets=False)
                    for model in config.models:
                        timeout = _native_fields(config, model, "P")["dist_timeout"]
                        self.assertGreaterEqual(timeout, 1)
                        self.assertLessEqual(timeout * 1000, request)
                else:
                    self.assert_code("INVALID_CONFIG", lambda: load_config(file, require_assets=False))

    def test_pool_and_batch_history_boundaries(self):
        raw = json.loads((ROOT / "configs/sglang_a100.json").read_text())
        with TemporaryDirectory() as directory:
            file = Path(directory) / "config.json"
            for mode, limit, valid in (("shared", 100000, True), ("static", 4000000, True),
                                       ("other", 4000000, False), ("shared", 99999, False),
                                       ("shared", 4000001, False), ("shared", True, False)):
                raw.update(kv_pool_mode=mode, batch_history_limit=limit)
                file.write_text(json.dumps(raw))
                if valid:
                    config = load_config(file, require_assets=False)
                    self.assertEqual(config.kv_pool_mode, mode)
                    self.assertEqual(config.batch_history_limit, limit)
                else:
                    self.assert_code("INVALID_CONFIG", lambda: load_config(file, require_assets=False))

    def test_controller_request_and_batch_tombstone_limits_are_independent(self):
        from muxserve_sglang.controller import _RetiredIds
        retired = _RetiredIds(batch_limit=3)
        for index in range(3):
            retired.add("batch", f"b{index}")
        self.assertTrue(retired.full("batch"))
        self.assertFalse(retired.full("request"))
        self.assert_code("HISTORY_FULL", lambda: retired.add("batch", "b3"))
        retired.add("request", "r")
        self.assertEqual(retired.limit("request"), 100000)

    def test_software_mismatch_rejects_before_toolchain(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            self.write_distribution(root, "torch", "2.12.0")
            with patch.object(preflight.sys, "path", [str(root)]), \
                    patch.object(preflight, "_tool_output") as tool:
                self.assert_code("SOFTWARE_MISMATCH", preflight.inspect_software)
                tool.assert_not_called()

    def write_distribution(self, root, name, version):
        metadata = root / f"{name.replace('-', '_')}-{version}.dist-info"
        metadata.mkdir(parents=True)
        (metadata / "METADATA").write_text(f"Name: {name}\nVersion: {version}\n", encoding="utf-8")
        return metadata

    @contextmanager
    def simulated_boundary(self, forbidden):
        """A temporary blocked tree only; never construct the production exclusion."""
        original_safe = preflight.safe_path
        original_is_file, original_is_dir = Path.is_file, Path.is_dir
        original_read_text, original_scandir = Path.read_text, os.scandir
        validated = set()

        def guarded(value):
            path = Path(value).absolute()
            pending = list(path.parts[1:])
            current = Path(path.anchor)
            hops = 0
            while pending:
                current = current / pending.pop(0)
                if current == forbidden or forbidden in current.parents:
                    raise IntegrationError("FORBIDDEN_PATH", "Simulated boundary")
                if current.is_symlink():
                    target = current.readlink()
                    target = target if target.is_absolute() else current.parent / target
                    pending = list(target.parts[1:]) + pending
                    current = Path(target.anchor)
                    hops += 1
                    self.assertLess(hops, 40)
            result = original_safe(current)
            validated.add(result)
            return result

        def probe(original, value, *args, **kwargs):
            self.assertIn(Path(value), validated, f"Probe before validation: {value}")
            return original(value, *args, **kwargs)

        with patch.object(preflight, "safe_path", side_effect=guarded), \
                patch.object(preflight, "read_asset_path", side_effect=guarded), \
                patch.object(Path, "is_file", lambda p: probe(original_is_file, p)), \
                patch.object(Path, "is_dir", lambda p: probe(original_is_dir, p)), \
                patch.object(Path, "read_text", lambda p, *a, **k: probe(original_read_text, p, *a, **k)), \
                patch.object(preflight.os, "scandir", lambda p: probe(original_scandir, p)):
            yield

    def link_or_skip(self, link, target, *, directory=False):
        try:
            link.symlink_to(target, target_is_directory=directory)
        except OSError as exc:
            self.skipTest(f"Temporary symlink creation unavailable: {exc}")

    def test_all_search_roots_validated_before_metadata_discovery(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            blocked = root / "blocked"
            with self.simulated_boundary(blocked), \
                    patch.object(preflight.sys, "path", [str(root), str(blocked)]), \
                    patch.object(preflight, "_distribution") as discover:
                self.assert_code("FORBIDDEN_PATH", preflight.inspect_software)
                discover.assert_not_called()

    def test_metadata_child_boundaries_before_any_probe(self):
        for name, kind in (("torch", "directory"), ("torch", "METADATA"),
                           ("torch", "PKG-INFO"), ("torchaudio", "METADATA")):
            with self.subTest(name=name, kind=kind), TemporaryDirectory() as directory:
                root = Path(directory)
                site, blocked = root / "site", root / "blocked"
                site.mkdir()
                if kind == "directory":
                    self.link_or_skip(site / f"{name}-1.dist-info", blocked, directory=True)
                else:
                    metadata = site / f"{name}-1.{('egg-info' if kind == 'PKG-INFO' else 'dist-info')}"
                    metadata.mkdir()
                    self.link_or_skip(metadata / kind, blocked / kind)
                with self.simulated_boundary(blocked):
                    search = [preflight.safe_path(site)]
                    self.assert_code("FORBIDDEN_PATH", lambda: preflight._distribution(name, search))

    def test_metadata_name_and_duplicate_rejection_and_search_order(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            first, second = root / "first", root / "second"
            metadata = self.write_distribution(first, "torch", "2.13.0")
            self.write_distribution(second, "torch", "2.12.0")
            self.assertEqual(preflight._distribution("torch", [first, second])["version"], "2.13.0")
            (metadata / "METADATA").write_text("Name: unrelated\nVersion: 2.13.0\n")
            self.assert_code("SOFTWARE_MISMATCH", lambda: preflight._distribution("torch", [first]))
            (metadata / "METADATA").write_text("Name: torch\nVersion: 2.13.0\n")
            self.write_distribution(first, "torch", "2.12.0")
            self.assert_code("SOFTWARE_MISMATCH", lambda: preflight._distribution("torch", [first]))

    def test_compat_editable_source_and_metadata_binding(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            site, source = root / "site", root / "source"
            package = source / "sglang"
            package.mkdir(parents=True)
            (package / "__init__.py").write_text("raise AssertionError('must never import fixture')\n")
            metadata = self.write_distribution(site, "sglang", preflight.SGLANG_TAG.removeprefix("v"))
            direct_url = metadata / "direct_url.json"
            direct_url.write_text(json.dumps({"url": source.as_uri(), "dir_info": {"editable": True}}))
            with patch.object(preflight.sys, "meta_path", [preflight.machinery.PathFinder]), \
                    patch.dict(preflight.sys.modules, {"sglang": None}):
                distribution = preflight._distribution("sglang", [site, source])
                self.assertEqual(preflight._sglang_source([site, source], distribution), package)
                direct_url.unlink()
                self.assert_code("SOFTWARE_MISMATCH", lambda: preflight._sglang_source([site, source], distribution))

    def test_custom_editable_finder_refused_without_invocation(self):
        finder = SimpleNamespace(MAPPING={"sglang": "unused-source"}, find_spec=Mock())
        with patch.object(preflight.sys, "meta_path", [finder, preflight.machinery.PathFinder]):
            with self.assertRaisesRegex(IntegrationError, "editable_mode=compat"):
                preflight._sglang_source([], {})
        finder.find_spec.assert_not_called()

    def test_custom_path_hook_refused_without_invocation(self):
        hook = Mock()
        with patch.object(preflight.sys, "meta_path", [preflight.machinery.PathFinder]), \
                patch.object(preflight.sys, "path_hooks", [hook]):
            with self.assertRaisesRegex(IntegrationError, "Unsupported custom path hook"):
                preflight._sglang_source([], {})
        hook.assert_not_called()

    def test_cached_importer_refused_before_metadata_discovery(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            importer = SimpleNamespace(find_spec=Mock())
            with patch.object(preflight.sys, "path", [str(root)]), \
                    patch.dict(preflight.sys.path_importer_cache, {str(root): importer}), \
                    patch.object(preflight, "_distribution") as discover:
                self.assert_code("SOFTWARE_MISMATCH", preflight.inspect_software)
            discover.assert_not_called()
            importer.find_spec.assert_not_called()

    def test_source_and_direct_url_children_checked_before_probes(self):
        for child in ("sglang", "sglang/__init__.py", "sglang.py", "direct_url.json"):
            with self.subTest(child=child), TemporaryDirectory() as directory:
                root = Path(directory)
                site, blocked = root / "site", root / "blocked"
                metadata = self.write_distribution(site, "sglang", preflight.SGLANG_TAG.removeprefix("v"))
                link = metadata / child if child == "direct_url.json" else site / child
                link.parent.mkdir(parents=True, exist_ok=True)
                self.link_or_skip(link, blocked / "target", directory=child == "sglang")
                if child == "direct_url.json":
                    (site / "sglang").mkdir()
                    (site / "sglang/__init__.py").write_text("")
                with self.simulated_boundary(blocked), \
                        patch.object(preflight.sys, "meta_path", [preflight.machinery.PathFinder]):
                    search = [preflight.safe_path(site)]
                    distribution = preflight._distribution("sglang", search)
                    self.assert_code("FORBIDDEN_PATH", lambda: preflight._sglang_source(search, distribution))

    def test_compiler_lookup_validates_explicit_and_path_children(self):
        for explicit in (False, True):
            with self.subTest(explicit=explicit), TemporaryDirectory() as directory:
                root = Path(directory)
                binary, blocked = root / "bin", root / "blocked"
                binary.mkdir()
                compiler = binary / "c++"
                self.link_or_skip(compiler, blocked / "compiler")
                with self.simulated_boundary(blocked), \
                        patch.dict(os.environ, {"CXX": f'"{compiler}"' if explicit else "c++", "PATH": str(binary)}), \
                        patch.object(preflight, "_tool_output") as tool:
                    self.assert_code("FORBIDDEN_PATH", preflight._compiler_command)
                    tool.assert_not_called()

    def test_installed_patch_mismatch_rejects_before_toolchain(self):
        versions = {"torch": "2.13.0", "transformers": "5.12.1", "sglang-kernel": "0.4.7", "sglang": preflight.SGLANG_TAG.removeprefix("v")}
        def distribution(name, search):
            return {"version": versions[name]} if name in versions else None
        manifests = [{"python/sglang/srt/managers/scheduler.py": {"patched_sha256": "0" * 64}},
                     {"commit": preflight.SGLANG_COMMIT, "tag": preflight.SGLANG_TAG}]
        with patch.object(preflight, "_distribution", side_effect=distribution), \
                patch.object(preflight, "_sglang_source", return_value=ROOT / "sglang"), \
                patch.object(preflight, "_read_json", side_effect=manifests), \
                patch.object(preflight, "_asset_file", return_value=ROOT / "muxserve_sglang/preflight.py"), \
                patch.object(preflight, "_sha256", return_value="1" * 64), \
                patch.object(preflight, "_tool_output") as tool:
            self.assert_code("SOFTWARE_MISMATCH", preflight.inspect_software)
            tool.assert_not_called()

    def test_diagnostic_failure_still_enters_retention(self):
        supervisor = object.__new__(Supervisor)
        supervisor.gateway = supervisor.driver = None
        supervisor.directory = Path("unused")
        supervisor.controller = Mock()
        supervisor._status = Mock(side_effect=OSError("disk full"))
        supervisor._retain_producers = Mock()
        with patch("muxserve_sglang.launcher.logger.error", side_effect=OSError("log unavailable")), \
                patch("muxserve_sglang.launcher.os.write", side_effect=OSError("stderr unavailable")):
            supervisor._protect_failure("EVIDENCE_WRITE_FAILED")
        supervisor._retain_producers.assert_called_once_with()
        supervisor.controller.protect_failure.assert_called_once_with("EVIDENCE_WRITE_FAILED")

    def test_protected_failure_closes_manual_admission_and_pins_running_batch(self):
        fixture = Fixture()
        fixture.submit(max_new=4)
        plan, batch = fixture.prepare()
        fixture.ready_start(plan, batch)
        before = fixture.store.snapshot()
        fixture.controller.protect_failure("native_signal")
        after = fixture.store.snapshot()
        self.assertIn(batch["reservation_id"], after["fences"])
        self.assertEqual(before["ranks"], after["ranks"])
        self.assertEqual(set(before["allocations"]), set(after["allocations"]))
        self.assert_code("PROTECTED_FAILURE", lambda: fixture.submit("new"))
        self.assert_code("PROTECTED_FAILURE", lambda: fixture.controller.prepare_batch(plan))
        self.assert_code("PROTECTED_FAILURE", lambda: fixture.controller.start_batch(fixture.group, plan.batch_id))
        self.assert_code("PROTECTED_FAILURE", lambda: fixture.controller.resume_request("m", "r", "unused"))
        self.assertEqual(fixture.controller.tick()["suggestions"], ())

    def test_protected_gate_survives_fence_diagnostic_failure(self):
        fixture = Fixture()
        with patch.object(fixture.controller, "_fence", side_effect=OSError("evidence unavailable")):
            result = fixture.controller.protect_failure("native_signal")
        self.assertIn("m", result["fence_errors"])
        self.assertTrue(fixture.controller.snapshot()["groups"]["m"]["fenced"])
        self.assert_code("PROTECTED_FAILURE", lambda: fixture.submit())


if __name__ == "__main__":
    unittest.main()
