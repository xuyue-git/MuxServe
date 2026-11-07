"""User-run CPU lifecycle check. Fake rank reports live only in this file.

Run from the repository root: python acceptance/controller_check.py
Do not run during the static-only implementation phase.
"""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from muxserve_sglang.config import DeploymentConfig, ModelDeployment
from muxserve_sglang.contracts import (BatchPlan, IntegrationError, RequestRef,
                                       RequestSpec, RuntimeEvent)
from muxserve_sglang.controller import MuxController
from muxserve_sglang.events import EventReceiver, RuntimeOutbox
from muxserve_sglang.policy import ModelDemand, PolicySnapshot
from muxserve_sglang.storage import HeadwiseStore


class Clock:
    def __init__(self):
        self.now = 100.0

    def __call__(self):
        return self.now


class Fixture:
    def __init__(self, *, max_inflight=1, store_blocks=64, event_window=32,
                 store=None, initializing=False):
        model = ModelDeployment("m", "unused", "0" * 40, "unused", "0" * 40,
                                1, 1, 128, 30, 18100, 18101, 18102)
        self.clock = Clock()
        self.store = store if store is not None else HeadwiseStore({0: store_blocks})
        self.controller = MuxController(DeploymentConfig((model,), tp_size=1,
            kv_bytes_per_rank=store_blocks * 8192, max_model_inflight=max_inflight),
            self.store, clock=self.clock, event_window=event_window,
            expected_lengths={"m": 16}, initializing=initializing)
        self.group = self.controller.group_id("m")
        self.seq = {"P": 0, "D": 0}
        self.command_acked = {"P": 0, "D": 0}
        self.cap = {"vocab_size": 32, "eos_token_ids": (9,),
                    "native_pd": True, "headwise_kv": True}
        for stage, pid in (("P", 1001), ("D", 1002)):
            self.controller.register_rank(self.group, stage, 0, pid, self.cap)

    def commands(self, stage, kind):
        pid = 1001 if stage == "P" else 1002
        return [item for item in self.controller.poll_commands(self.group, stage, 0, pid,
                    after_seq=self.command_acked[stage])
                if item["kind"] == kind]

    def ack_commands(self, stage):
        pid = 1001 if stage == "P" else 1002
        pending = self.controller.poll_commands(self.group, stage, 0, pid,
            after_seq=self.command_acked[stage])
        if pending:
            last = pending[-1]["command_seq"]
            self.controller.poll_commands(self.group, stage, 0, pid,
                after_seq=last, ack_seq=last)
            self.command_acked[stage] = last
        return self.command_acked[stage]

    def report(self, stage, batch_id, kind, payload):
        reply = self.controller.rank_report(self.group, stage, 0,
            1001 if stage == "P" else 1002, str(self.seq[stage] + 1), batch_id,
            kind, payload)
        assert reply["accepted"]
        self.seq[stage] += 1
        return reply

    def submit(self, request_id="r", max_new=1, timeout=120000, input_ids=(1, 2)):
        spec = RequestSpec("m", request_id, input_ids, max_new, timeout,
                           {"temperature": 0, "ignore_eos": False})
        self.controller.submit_request(spec)
        return spec

    def prepare(self, request_id="r", stage="P", purpose="GENERATE", batch_id="b"):
        output = self.controller.request_output("m", request_id)
        plan = BatchPlan(self.group, batch_id, "m", stage,
            (RequestRef(request_id, output["attempt"], output["output_count"]),), purpose)
        batch = self.controller.prepare_batch(plan)
        return plan, batch

    def ready_start(self, plan, batch):
        if plan.stage == "P":
            command = next(cmd for cmd in self.commands("D", "PREPARE")
                           if cmd["batch_id"] == plan.batch_id)
            for slot, item in enumerate(command["requests"]):
                view, room = item["reservation_view"], item["bootstrap_room"]
                bootstrap = (self.group, room, 0, 1001, view["allocation_id"],
                             view["version"], item["target_valid_tokens"])
                assert self.controller.broker_bootstrap(*bootstrap)["status"] == "PENDING"
                destination = (self.group, room, 0, 1002, "m", item["request_id"],
                    item["attempt"], view["allocation_id"], view["version"],
                    item["target_valid_tokens"], item["rank_block_table"],
                    slot, item["target_valid_tokens"])
                assert self.controller.broker_register_destination(*destination)["registered"]
                assert self.controller.broker_register_destination(*destination)["registered"]
                assert self.controller.broker_bootstrap(*bootstrap)["status"] == "READY"
        self.report("P", plan.batch_id, "READY", {"reservation_id": batch["reservation_id"]})
        expect("NOT_READY", lambda: self.controller.start_batch(self.group, plan.batch_id))
        self.report("D", plan.batch_id, "READY", {"reservation_id": batch["reservation_id"]})
        self.controller.start_batch(self.group, plan.batch_id)
        starts = [cmd for cmd in self.commands(plan.stage, "START")
                  if cmd["batch_id"] == plan.batch_id and
                     cmd["grant_id"] == batch["reservation_id"]]
        assert len(starts) == 1
        self.controller.start_batch(self.group, plan.batch_id)
        assert [cmd for cmd in self.commands(plan.stage, "START")
                if cmd["batch_id"] == plan.batch_id and
                   cmd["grant_id"] == batch["reservation_id"]] == starts

    def stage_prefill(self, plan):
        reservation_id = self.controller.snapshot()["groups"]["m"]["batches"][plan.batch_id]["reservation_id"]
        matches = [cmd for cmd in self.commands("P", "PREPARE")
                   if cmd["batch_id"] == plan.batch_id and
                      cmd["reservation_id"] == reservation_id and
                      cmd["batch_stage"] == "P" and
                      cmd["ordered_request_ids"] == tuple(ref.request_id for ref in plan.requests)]
        assert len(matches) == 1 and len(matches[0]["requests"]) == 1
        item = matches[0]["requests"][0]
        assert item["request_id"] == plan.requests[0].request_id
        assert item["attempt"] == plan.requests[0].attempt
        room = item["bootstrap_room"]
        view = item["reservation_view"]
        self.controller.broker_stage(self.group, room, 0, 1001, {
            "bootstrap_room": room, "allocation_id": view["allocation_id"],
            "reserved_version": view["version"],
            "target_valid_tokens": item["target_valid_tokens"], "rank": 0,
            "native_payload": {"native_request_id": plan.batch_id}})
        expect("HANDOFF_NOT_READY", lambda: self.controller.broker_lookup(
            self.group, room, 0, 1002, "m", item["request_id"], item["attempt"],
            view["allocation_id"], view["version"], item["target_valid_tokens"]))
        return item

    def gpu(self, plan, batch, token, valid):
        stage = plan.stage
        item = {"request_id": plan.requests[0].request_id,
                "attempt": plan.requests[0].attempt,
                "token_id": token, "valid_tokens": valid}
        self.report(stage, plan.batch_id, "GPU_DONE", {
            "reservation_id": batch["reservation_id"],
            "results": [item], "duration_ms": 1.5, "gpu_forward_ms": 1.0})

    def handoff(self, plan):
        item = self.commands("D", "HANDOFF")[-1]["requests"][0]
        room = item["bootstrap_room"]
        found = self.controller.broker_lookup(self.group, room, 0, 1002,
            "m", item["request_id"], item["attempt"], item["allocation_id"],
            item["version"], item["valid_tokens"])
        assert found["rank_block_table"] == item["rank_block_table"]
        self.controller.broker_ack(self.group, room, 0, 1002,
            item["allocation_id"], item["version"], item["valid_tokens"],
            item["rank_block_table"])
        assert self.controller.broker_poll(self.group, room, 0, 1002)["status"] == "SUCCESS"
        assert self.controller.broker_poll(self.group, room, 0, 1001, "P")["status"] == "SUCCESS"
        self.report("D", plan.batch_id, "HANDOFF_READY", {"requests": [item]})

    def batch_quiesce(self, plan, batch):
        required = ("P",) if plan.stage == "P" else ("P", "D")
        for stage in required:
            self.report(stage, plan.batch_id, "QUIESCED", {
                "reservation_id": batch["reservation_id"],
                "request_ids": (plan.requests[0].request_id,)})

    def cleanup(self, request_id):
        command = self.commands("P", "CLEANUP")[-1]
        cleanup_id = command["cleanup_id"]
        attempt = command["attempt"]
        for stage in ("P", "D"):
            self.report(stage, cleanup_id, "QUIESCED", {
                "cleanup_id": cleanup_id, "request_id": request_id,
                "attempt": attempt})

    def ack_all_commands(self):
        for stage in ("P", "D"):
            self.ack_commands(stage)


def expect(code, fn):
    try:
        fn()
    except IntegrationError as exc:
        assert exc.code == code, (code, exc.code)
    else:
        raise AssertionError("Expected " + code)


def check_first_token_and_replay():
    f = Fixture()
    spec = f.submit(max_new=1)
    assert f.controller.submit_request(spec)["state"] == "QUEUED"
    expect("IDENTITY_CONFLICT", lambda: f.controller.submit_request(
        replace(spec, input_ids=(1, 3))))
    plan, batch = f.prepare()
    expect("IDENTITY_CONFLICT", lambda: f.controller.prepare_batch(
        replace(plan, purpose="REBUILD")))
    f.ready_start(plan, batch)
    f.stage_prefill(plan)
    f.gpu(plan, batch, 4, 2)
    assert f.controller.request_output("m", "r")["output_count"] == 0
    f.handoff(plan)
    assert f.controller.request_output("m", "r")["output_token_ids"] == (4,)
    f.batch_quiesce(plan, batch)
    f.cleanup("r")
    output = f.controller.request_output("m", "r")
    assert output["state"] == "FINISHED" and output["finish_reason"] == "length"
    assert f.store.snapshot()["ranks"][0]["free_blocks"] == 64
    assert all(not any(refs) for refs in f.controller.credits.refs.values())
    events = f.controller.poll_events(f.group)
    assert [event.kind for event in events] == ["TOKEN", "TERMINAL"]
    f.controller.ack_events(f.group, events[-1].event_seq)
    f.ack_all_commands()
    f.controller.forget_request("m", "r", 0)
    assert f.store.snapshot()["history"]["retained_batches"] == 0
    expect("RETIRED_REQUEST", lambda: f.controller.submit_request(spec))


def check_cancel_and_timeout():
    f = Fixture()
    f.submit(max_new=2)
    plan, batch = f.prepare()
    f.ready_start(plan, batch)
    pending = f.controller.cancel_request("m", "r", "client")
    assert pending["pending_reason"] == "cancel:client"
    assert f.store.snapshot()["allocations"]
    assert any(any(refs) for refs in f.controller.credits.refs.values())
    f.stage_prefill(plan)
    f.gpu(plan, batch, 4, 2)
    f.handoff(plan)
    f.batch_quiesce(plan, batch)
    f.cleanup("r")
    output = f.controller.request_output("m", "r")
    assert output["state"] == "CANCELLED" and output["output_token_ids"] == ()
    assert not any(event.kind == "TOKEN" for event in f.controller.poll_events(f.group))
    assert f.store.snapshot()["ranks"][0]["free_blocks"] == 64

    g = Fixture()
    g.submit(timeout=1000)
    plan, batch = g.prepare()
    g.ready_start(plan, batch)
    g.clock.now += 2.0
    g.controller.tick()
    assert g.controller.snapshot()["groups"]["m"]["fenced"] is None
    assert g.controller.request_output("m", "r")["state"] == "STOP_PENDING"
    assert not g.store.snapshot()["fences"]
    assert g.store.snapshot()["allocations"]
    accepted_seq = g.seq["P"]
    expect("STALE_REPORT", lambda: g.report("P", plan.batch_id, "READY",
        {"reservation_id": batch["reservation_id"]}))
    assert g.seq["P"] == accepted_seq
    g.stage_prefill(plan)
    g.gpu(plan, batch, 4, 2)
    g.handoff(plan)
    g.batch_quiesce(plan, batch)
    g.cleanup("r")
    assert g.controller.request_output("m", "r")["state"] == "TIMED_OUT"
    assert g.controller.groups["m"].ready
    assert not g.store.snapshot()["allocations"]
    assert not any(event.kind == "TOKEN" for event in g.controller.poll_events(g.group))

    hung = Fixture()
    hung.submit(timeout=1000)
    plan, batch = hung.prepare()
    hung.ready_start(plan, batch)
    hung.clock.now += hung.controller.config.request_timeout_ms / 1000 + 0.01
    hung.controller.tick()
    assert hung.controller.groups["m"].fenced == "execution_watchdog"
    assert hung.store.snapshot()["fences"] and hung.store.snapshot()["allocations"]


def check_stop_freezes_committed_output():
    for stop in ("cancel", "deadline"):
        f = Fixture()
        f.submit(max_new=3, timeout=1000)
        plan, batch = f.prepare()
        f.ready_start(plan, batch)
        f.stage_prefill(plan)
        f.gpu(plan, batch, 4, 2)
        assert f.controller.request_output("m", "r")["output_count"] == 0
        if stop == "cancel":
            f.controller.cancel_request("m", "r", "accepted_during_handoff")
        else:
            f.clock.now += 1.1
        f.handoff(plan)
        f.batch_quiesce(plan, batch)
        f.cleanup("r")
        output = f.controller.request_output("m", "r")
        assert output["output_token_ids"] == ()
        assert output["state"] == ("CANCELLED" if stop == "cancel" else "TIMED_OUT")
    for stop in ("cancel", "deadline", "timeout-fence", "suspend"):
        f = Fixture()
        f.submit(max_new=3, timeout=1000)
        prefill, prepared = f.prepare()
        f.ready_start(prefill, prepared)
        f.stage_prefill(prefill)
        f.gpu(prefill, prepared, 4, 2)
        f.handoff(prefill)
        f.batch_quiesce(prefill, prepared)
        plan, batch = f.prepare(stage="D", batch_id="stop-decode")
        f.ready_start(plan, batch)
        if stop == "cancel":
            f.controller.cancel_request("m", "r", "accepted_before_gpu_done")
        elif stop == "suspend":
            f.controller.suspend_request("m", "r", 0)
        else:
            f.clock.now += 1.1
            if stop == "timeout-fence":
                f.controller.tick()
        f.gpu(plan, batch, 5, 3)
        expected = (4, 5) if stop == "suspend" else (4,)
        assert f.controller.request_output("m", "r")["output_token_ids"] == expected
        f.batch_quiesce(plan, batch)
        f.cleanup("r")
        output = f.controller.request_output("m", "r")
        assert output["state"] == {"cancel": "CANCELLED", "deadline": "TIMED_OUT",
            "timeout-fence": "TIMED_OUT", "suspend": "SUSPENDED"}[stop]
        assert output["output_token_ids"] == expected
        assert sum(event.kind == "TOKEN" for event in f.controller.poll_events(f.group)) == len(expected)
        assert f.store.snapshot()["ranks"][0]["free_blocks"] == 64


def check_revoke_and_first_eos():
    f = Fixture()
    f.submit(max_new=2)
    plan, batch = f.prepare()
    f.report("P", plan.batch_id, "READY", {"reservation_id": batch["reservation_id"]})
    f.controller.abort_batch(f.group, plan.batch_id, "client_cancel")
    assert f.controller.snapshot()["groups"]["m"]["batches"][plan.batch_id]["status"] == "REVOKING"
    assert f.store.snapshot()["allocations"]
    expect("NOT_READY", lambda: f.controller.start_batch(f.group, plan.batch_id))
    for stage in ("P", "D"):
        f.report(stage, plan.batch_id, "QUIESCED", {
            "reservation_id": batch["reservation_id"], "request_ids": ("r",)})
    assert f.store.snapshot()["ranks"][0]["free_blocks"] == 64
    assert all(not any(refs) for refs in f.controller.credits.refs.values())
    assert f.controller.request_output("m", "r")["attempt"] == 1
    expect("STALE_ATTEMPT", lambda: f.controller.prepare_batch(BatchPlan(
        f.group, "old-attempt", "m", "P", (RequestRef("r", 0, 0),))))
    plan, batch = f.prepare(batch_id="retry")
    f.ready_start(plan, batch)
    f.stage_prefill(plan)
    f.gpu(plan, batch, 9, 2)
    f.handoff(plan)
    f.batch_quiesce(plan, batch)
    f.cleanup("r")
    assert f.controller.request_output("m", "r")["finish_reason"] == "eos"
    assert not f.commands("D", "START")


def check_suspend_rebuild():
    f = Fixture()
    f.submit(max_new=2)
    plan, batch = f.prepare()
    f.ready_start(plan, batch)
    f.stage_prefill(plan)
    f.gpu(plan, batch, 4, 2)
    f.handoff(plan)
    f.batch_quiesce(plan, batch)
    assert f.controller.request_output("m", "r")["output_token_ids"] == (4,)
    assert f.controller.suspend_request("m", "r", 0)["pending_reason"] == "suspend"
    f.cleanup("r")
    record = f.controller.suspend_request("m", "r", 0)["record"]
    assert record.output_token_ids == (4,) and record.sampling_state is None
    resumed = f.controller.resume_request("m", "r", record.resume_record_id)
    assert resumed["attempt"] == 1 and resumed["output_token_ids"] == (4,)
    assert f.controller.resume_request("m", "r", record.resume_record_id) == resumed
    expect("STALE_ATTEMPT", lambda: f.controller.prepare_batch(BatchPlan(
        f.group, "old", "m", "P", (RequestRef("r", 0, 1),), "REBUILD")))
    plan, batch = f.prepare(purpose="REBUILD", batch_id="rebuild")
    prep = f.commands("P", "PREPARE")[-1]["requests"][0]
    assert prep["selected_input_ids"] == (1, 2) and prep["boundary_token_id"] == 4
    f.ready_start(plan, batch)
    f.stage_prefill(plan)
    f.gpu(plan, batch, None, 2)
    f.handoff(plan)
    f.batch_quiesce(plan, batch)
    assert f.controller.request_output("m", "r")["output_token_ids"] == (4,)
    plan, batch = f.prepare(stage="D", batch_id="decode")
    f.ready_start(plan, batch)
    f.gpu(plan, batch, 9, 3)
    f.batch_quiesce(plan, batch)
    f.cleanup("r")
    assert f.controller.request_output("m", "r")["output_token_ids"] == (4, 9)
    feedback = tuple(f.controller.policy.feedback_history)
    assert tuple(item.input_tokens for item in feedback) == (2, 2, 1)
    assert tuple(item.output_tokens for item in feedback) == (1, 0, 1)


def check_stale_rebuild_proposal_after_stop():
    from muxserve_sglang.driver import ControllerDriver

    for stop in ("cancel", "timeout"):
        f = Fixture()
        f.submit(max_new=3)
        plan, batch = f.prepare()
        f.ready_start(plan, batch)
        f.stage_prefill(plan)
        f.gpu(plan, batch, 4, 2)
        f.handoff(plan)
        f.batch_quiesce(plan, batch)
        f.controller.suspend_request("m", "r", 0)
        f.cleanup("r")
        record = f.controller.suspend_request("m", "r", 0)["record"]
        f.controller.resume_request("m", "r", record.resume_record_id)
        proposal = next(p for p in f.controller.tick()["suggestions"] if p.purpose == "REBUILD")
        if stop == "cancel":
            f.controller.cancel_request("m", "r")
        else:
            f.clock.now = record.deadline
            f.controller.tick()
        before = set(f.store.snapshot()["allocations"])
        driver = ControllerDriver(f.controller, lambda event: None)
        driver._prepare(proposal)  # An ordinary stale proposal must not stop the driver.
        assert not driver.pending and driver.failure is None
        assert set(f.store.snapshot()["allocations"]) == before
        assert not f.controller.protected_failure and not f.controller.groups["m"].fenced


def check_event_gap_and_rank_death():
    seen = []
    receiver = EventReceiver("g", seen.append)
    second = RuntimeEvent("g", 2, "TOKEN", "m", "r", 0, "b", {"token_ids": (2,)})
    first = RuntimeEvent("g", 1, "TOKEN", "m", "r", 0, "b", {"token_ids": (1,)})
    assert receiver.receive(second) == 0
    assert receiver.receive(first) == 2 and len(seen) == 2
    expect("EVENT_CONFLICT", lambda: receiver.receive(replace(second,
        payload={"token_ids": (3,)})))
    outbox = RuntimeOutbox("g", "m", maximum_pending=8)
    for _ in range(8):
        outbox.emit("CONTROL")
    expect("EVENT_BACKPRESSURE", lambda: outbox.emit("CONTROL"))
    outbox.ack(8)
    assert not outbox.replay(8)

    f = Fixture()
    f.submit()
    plan, _ = f.prepare()
    f.controller.rank_failed(f.group, "P", 0, 1001, "process_exit")
    assert f.controller.snapshot()["groups"]["m"]["fenced"]
    assert f.store.snapshot()["fences"]


def check_deadline_and_terminal_suspend():
    f = Fixture()
    f.submit(timeout=1000)
    plan, batch = f.prepare()
    for stage in ("P", "D"):
        f.report(stage, plan.batch_id, "READY", {"reservation_id": batch["reservation_id"]})
    f.clock.now += 1.1
    expect("NOT_READY", lambda: f.controller.start_batch(f.group, plan.batch_id))
    assert f.controller.snapshot()["groups"]["m"]["batches"][plan.batch_id]["status"] == "REVOKING"
    assert not f.commands("P", "START")

    g = Fixture()
    g.submit(max_new=1)
    plan, batch = g.prepare()
    g.ready_start(plan, batch)
    assert g.controller.suspend_request("m", "r", 0)["pending_reason"] == "suspend"
    g.stage_prefill(plan)
    g.gpu(plan, batch, 9, 2)
    g.handoff(plan)
    g.batch_quiesce(plan, batch)
    g.cleanup("r")
    result = g.controller.request_output("m", "r")
    assert result["state"] == "FINISHED" and result["finish_reason"] == "eos"
    assert result["resume_record_id"] is None


def check_suspended_cancel_and_deadline():
    for timeout in (False, True):
        f = Fixture()
        f.submit(timeout=1000)
        f.controller.suspend_request("m", "r", 0)
        f.cleanup("r")
        record = f.controller.suspend_request("m", "r", 0)["record"]
        if timeout:
            f.clock.now += 1.1
            f.controller.tick()
        else:
            f.controller.cancel_request("m", "r", "suspended_acceptance")
        assert f.controller.request_output("m", "r")["state"] == "CLEANUP_PENDING"
        f.cleanup("r")
        expected = "TIMED_OUT" if timeout else "CANCELLED"
        assert f.controller.request_output("m", "r")["state"] == expected
        expect("UNKNOWN_RECORD", lambda: f.controller.resume_request("m", "r", record.resume_record_id))


def check_retention_and_admission():
    f = Fixture()
    f.submit(max_new=2)
    assert f.controller.suspend_request("m", "r", 0)["pending_reason"] == "suspend"
    f.cleanup("r")
    old = f.controller.suspend_request("m", "r", 0)["record"]
    old.sampling["ignore_eos"] = True
    assert not f.controller.suspend_request("m", "r", 0)["record"].sampling["ignore_eos"]
    resumed = f.controller.resume_request("m", "r", old.resume_record_id)
    assert f.controller.resume_request("m", "r", old.resume_record_id) == resumed
    assert not f.controller.records
    f.controller.suspend_request("m", "r", 1)
    f.cleanup("r")
    assert len(f.controller.records) == 1
    expect("UNKNOWN_RECORD", lambda: f.controller.resume_request("m", "r", old.resume_record_id))

    g = Fixture()
    g.submit(max_new=2)
    plan, batch = g.prepare()
    g.ready_start(plan, batch)
    g.stage_prefill(plan)
    g.gpu(plan, batch, 4, 2)
    g.handoff(plan)
    g.batch_quiesce(plan, batch)
    g.submit("second", max_new=2)
    expect("OUT_OF_CAPACITY", lambda: g.prepare("second", batch_id="extra"))

    policy = g.controller.policy
    policy.limits["m"] = 1
    snap = PolicySnapshot((ModelDemand("m", ("second",), ("r",),
        ordinals={"second": 2, "r": 1}, prompt_lengths={"second": 2}),),
        10, frozenset(), frozenset(), {"m": 1}, 2, {0: 64})
    assert not any(s.stage == "P" for s in policy.suggestions(snap))

    # A suspended request relinquishes its model slot. B may consume it;
    # resuming A cannot run a rebuild until B leaves that slot.
    h = Fixture()
    h.submit("a", max_new=2)
    plan, batch = h.prepare("a", batch_id="a-prefill")
    h.ready_start(plan, batch)
    h.stage_prefill(plan)
    h.gpu(plan, batch, 4, 2)
    h.handoff(plan)
    h.batch_quiesce(plan, batch)
    h.controller.suspend_request("m", "a", 0)
    h.cleanup("a")
    record = h.controller.suspend_request("m", "a", 0)["record"]
    h.submit("b", max_new=2)
    plan, batch = h.prepare("b", batch_id="b-prefill")
    h.ready_start(plan, batch)
    h.stage_prefill(plan)
    h.gpu(plan, batch, 4, 2)
    h.handoff(plan)
    h.batch_quiesce(plan, batch)
    assert h.controller.resume_request("m", "a", record.resume_record_id)["state"] == "REBUILD"
    expect("OUT_OF_CAPACITY", lambda: h.prepare("a", purpose="REBUILD", batch_id="a-blocked"))
    assert not any(s.stage == "P" for s in
                   h.controller.policy.suggestions(h.controller._snapshot_policy()))
    h.controller.cancel_request("m", "b")
    h.cleanup("b")
    assert h.prepare("a", purpose="REBUILD", batch_id="a-after-b")[1]["status"] == "PREPARING"


def check_all_waiting_rebuild_admission():
    f = Fixture()
    records = {}
    for rid in ("a", "b"):
        f.submit(rid, max_new=2)
        plan, batch = f.prepare(rid, batch_id=rid + "-prefill")
        f.ready_start(plan, batch)
        f.stage_prefill(plan)
        f.gpu(plan, batch, 4, 2)
        f.handoff(plan)
        f.batch_quiesce(plan, batch)
        f.controller.suspend_request("m", rid, 0)
        f.cleanup(rid)
        records[rid] = f.controller.suspend_request("m", rid, 0)["record"]
    for rid in ("a", "b"):
        assert f.controller.resume_request("m", rid, records[rid].resume_record_id)["state"] == "REBUILD"
    snapshot = f.controller._snapshot_policy()
    assert snapshot.inflight_counts["m"] == 0
    proposals = f.controller.policy.suggestions(snapshot)
    assert next(s for s in proposals if s.stage == "P").request_ids == ("a",)
    plan, batch = f.prepare("a", purpose="REBUILD", batch_id="a-rebuild")
    assert f.controller._snapshot_policy().inflight_counts["m"] == 1
    f.ready_start(plan, batch)
    f.stage_prefill(plan)
    f.gpu(plan, batch, None, 2)
    f.handoff(plan)
    f.batch_quiesce(plan, batch)
    expect("OUT_OF_CAPACITY", lambda: f.prepare("b", purpose="REBUILD",
        batch_id="b-while-a-owns"))
    f.controller.suspend_request("m", "a", 1)
    f.cleanup("a")
    assert f.controller._snapshot_policy().inflight_counts["m"] == 0
    assert f.prepare("b", purpose="REBUILD", batch_id="b-after-a")[1]["status"] == "PREPARING"


def check_recovery_and_reserved_capacity():
    outbox = RuntimeOutbox("g", "m", 9)
    outbox.reserve("group_error", 1)
    outbox.reserve("p", 4)
    outbox.reserve("d", 4)
    assert not outbox.grants_allowed and outbox.owner_can_complete("p")

    # Two real controller reservations fill the nine-slot window: error +
    # one P + one D. Both owners can still START and finish.
    full = Fixture(max_inflight=2, event_window=9)
    full.submit("older", max_new=2)
    first, first_view = full.prepare("older", batch_id="first-prefill")
    full.ready_start(first, first_view)
    full.stage_prefill(first)
    full.gpu(first, first_view, 4, 2)
    full.handoff(first)
    full.batch_quiesce(first, first_view)
    first_events = full.controller.poll_events(full.group)
    assert len(first_events) == 1 and first_events[0].kind == "TOKEN"
    full.controller.ack_events(full.group, first_events[-1].event_seq)
    full.submit("younger", max_new=2)
    p_plan, p_view = full.prepare("younger", batch_id="second-prefill")
    d_plan, d_view = full.prepare("older", stage="D", batch_id="first-decode")
    assert not full.controller.groups["m"].outbox.grants_allowed
    full.ready_start(p_plan, p_view)
    full.ready_start(d_plan, d_view)
    full.gpu(d_plan, d_view, 9, 3)
    full.batch_quiesce(d_plan, d_view)
    full.cleanup("older")
    full.stage_prefill(p_plan)
    full.gpu(p_plan, p_view, 4, 2)
    full.handoff(p_plan)
    full.batch_quiesce(p_plan, p_view)
    assert full.controller.request_output("m", "younger")["state"] == "DECODING"

    f = Fixture(max_inflight=2, store_blocks=3)
    for rid, bid in (("older", "p1"), ("younger", "p2")):
        f.submit(rid, max_new=2, input_ids=(1,) * 16)
        plan, batch = f.prepare(rid, batch_id=bid)
        f.ready_start(plan, batch)
        f.stage_prefill(plan)
        f.gpu(plan, batch, 4, 16)
        f.handoff(plan)
        f.batch_quiesce(plan, batch)
    plan = BatchPlan(f.group, "too-large", "m", "D",
        (RequestRef("older", 0, 1), RequestRef("younger", 0, 1)))
    expect("OUT_OF_CAPACITY", lambda: f.controller.prepare_batch(plan))
    f.controller.tick()
    assert f.controller.request_output("m", "younger")["pending_reason"] is None
    assert "m" not in f.controller.capacity_blocked
    plan, batch = f.prepare("older", stage="D", batch_id="one-fits")
    assert batch["status"] == "PREPARING"

    blocked = Fixture(max_inflight=2, store_blocks=2)
    for rid, bid in (("older", "full-p1"), ("younger", "full-p2")):
        blocked.submit(rid, max_new=2, input_ids=(1,) * 16)
        p, view = blocked.prepare(rid, batch_id=bid)
        blocked.ready_start(p, view)
        blocked.stage_prefill(p)
        blocked.gpu(p, view, 4, 16)
        blocked.handoff(p)
        blocked.batch_quiesce(p, view)
    too_large = BatchPlan(blocked.group, "no-single-fits", "m", "D",
        (RequestRef("older", 0, 1), RequestRef("younger", 0, 1)))
    expect("OUT_OF_CAPACITY", lambda: blocked.controller.prepare_batch(too_large))
    assert "m" in blocked.controller.capacity_blocked
    blocked.controller.tick()
    blocked.cleanup("younger")
    blocked.controller.tick()
    # A free step cannot end recovery; protected ownership still exists.
    assert blocked.controller._single_decode_fits("m", "older")
    assert blocked.controller.recovery_protected == {"shared": ("m", "older")}
    assert "m" in blocked.controller.capacity_blocked
    blocked.submit("cached", max_new=2)
    expect("OUT_OF_CAPACITY", lambda: blocked.prepare("cached", batch_id="cached-p"))
    assert blocked.controller.request_output("m", "cached")["state"] == "QUEUED"
    p, view = blocked.prepare("older", stage="D", batch_id="protected-completes")
    blocked.ready_start(p, view)
    blocked.gpu(p, view, 5, 17)
    blocked.batch_quiesce(p, view)
    blocked.cleanup("older")
    blocked.controller.tick()
    assert not blocked.controller.recovery_protected and not blocked.controller.capacity_blocked
    assert blocked.prepare("cached", batch_id="after-protected")[1]["status"] == "PREPARING"


def check_uncertain_reservation_is_pinned():
    class UncertainStore(HeadwiseStore):
        def reserve_kv(self, batch_id, targets, holder):
            super().reserve_kv(batch_id, targets, holder)
            raise IntegrationError("RPC_TIMEOUT", "Reservation reply was lost")

    store = UncertainStore({0: 64})
    f = Fixture(store=store)
    f.submit()
    expect("RPC_TIMEOUT", lambda: f.prepare())
    state = f.controller.snapshot()["groups"]["m"]
    assert state["fenced"] and state["uncertain_batches"] == ("b",)
    assert f.controller.poll_events(f.group)[0].kind == "GROUP_ERROR"
    assert store.snapshot()["allocations"]
    expect("IDENTITY_CONFLICT", lambda: f.controller.cancel_request("m", "r"))
    f.controller.tick()
    assert f.controller.request_output("m", "r")["state"] != "CANCELLED"

    class DefiniteCapacityThenDiagnosticTimeout(HeadwiseStore):
        def __init__(self):
            super().__init__({0: 64})
            self.reject_reserve = True
            self.fail_diagnostic = False

        def reserve_kv(self, batch_id, targets, holder):
            if self.reject_reserve:
                self.fail_diagnostic = True
                raise IntegrationError("OUT_OF_CAPACITY", "Definite atomic rejection")
            return super().reserve_kv(batch_id, targets, holder)

        def snapshot(self):
            if self.fail_diagnostic:
                raise IntegrationError("RPC_TIMEOUT", "Diagnostic read failed")
            return super().snapshot()

    definite = DefiniteCapacityThenDiagnosticTimeout()
    g = Fixture(store=definite)
    g.submit()
    expect("OUT_OF_CAPACITY", lambda: g.prepare(batch_id="definite-reject"))
    assert not any(any(refs) for refs in g.controller.credits.refs.values())
    assert g.controller.groups["m"].outbox.reserved == {"group_error": 1}
    assert all(not held for held in g.controller.groups["m"].command_reserved.values())
    assert not g.controller.groups["m"].fenced
    definite.fail_diagnostic = False
    definite.reject_reserve = False
    assert g.prepare(batch_id="after-diagnostic")[1]["status"] == "PREPARING"

    class ReadOnlyCounterTimeout(HeadwiseStore):
        def __init__(self):
            super().__init__({0: 64})
            self.fail_read = False

        def rank_usage(self):
            if self.fail_read:
                raise IntegrationError("RPC_TIMEOUT", "Read before reserve failed")
            return super().rank_usage()

    read_only = ReadOnlyCounterTimeout()
    h = Fixture(store=read_only)
    h.submit()
    read_only.fail_read = True
    expect("RPC_TIMEOUT", lambda: h.prepare(batch_id="read-failure"))
    assert not h.controller.groups["m"].fenced
    assert not any(any(refs) for refs in h.controller.credits.refs.values())
    read_only.fail_read = False
    assert h.prepare(batch_id="after-read")[1]["status"] == "PREPARING"


def check_command_completion_reservation():
    f = Fixture()
    f.submit(max_new=2)
    plan, batch = f.prepare()
    f.ready_start(plan, batch)
    group = f.controller.groups["m"]
    while True:
        try:
            f.controller._command(group, ("D", 0), "NOOP", {})
        except IntegrationError as exc:
            assert exc.code == "COMMAND_BACKPRESSURE"
            break
    while True:
        try:
            f.controller._command(group, ("P", 0), "NOOP", {})
        except IntegrationError as exc:
            assert exc.code == "COMMAND_BACKPRESSURE"
            break
    f.stage_prefill(plan)
    f.gpu(plan, batch, 4, 2)
    assert any(cmd["kind"] == "HANDOFF" for cmd in group.registered[("D", 0)].commands.values())
    f.handoff(plan)
    assert any(cmd["kind"] == "QUIESCE" and cmd["batch_id"] == plan.batch_id
               for cmd in group.registered[("P", 0)].commands.values())
    f.batch_quiesce(plan, batch)
    while True:
        try:
            f.controller._command(group, ("D", 0), "NOOP", {})
        except IntegrationError as exc:
            assert exc.code == "COMMAND_BACKPRESSURE"
            break
    f.controller.suspend_request("m", "r", 0)
    assert f.controller.request_output("m", "r")["state"] == "CLEANUP_PENDING"
    assert not f.commands("D", "CLEANUP")
    last = f.ack_commands("D")
    assert last > 0
    expect("COMMAND_GAP", lambda: f.controller.poll_commands(f.group, "D", 0,
        1002, after_seq=0))
    f.controller.tick()
    assert f.commands("D", "CLEANUP")
    f.cleanup("r")
    assert f.controller.request_output("m", "r")["state"] == "SUSPENDED"

    fenced = Fixture()
    fenced.submit(max_new=2)
    p, view = fenced.prepare()
    fenced.ready_start(p, view)
    fence_group = fenced.controller.groups["m"]
    while True:
        try:
            fenced.controller._command(fence_group, ("D", 0), "NOOP", {})
        except IntegrationError as exc:
            assert exc.code == "COMMAND_BACKPRESSURE"
            break
    fenced.stage_prefill(p)
    fenced.gpu(p, view, 4, 2)
    fenced.controller.abort_batch(fenced.group, p.batch_id, "fence_handoff")
    assert fence_group.fenced == "fence_handoff"
    assert fence_group.batches[p.batch_id].status == "QUIESCE"
    assert any(cmd["kind"] == "QUIESCE" and cmd["batch_id"] == p.batch_id
               for cmd in fence_group.registered[("D", 0)].commands.values())


def check_synchronized_feedback_duration():
    model = ModelDeployment("m", "unused", "0" * 40, "unused", "0" * 40,
                            1, 2, 256, 30, 18110, 18111, 18112)
    clock = Clock()
    store = HeadwiseStore({0: 64, 1: 64})
    controller = MuxController(DeploymentConfig((model,), tp_size=2,
        kv_bytes_per_rank=64 * 8192), store, clock=clock,
        expected_lengths={"m": 16})
    group = controller.group_id("m")
    cap = {"vocab_size": 32, "eos_token_ids": (9,),
           "native_pd": True, "headwise_kv": True}
    sequence = {(stage, rank): 0 for stage in ("P", "D") for rank in (0, 1)}
    for stage in ("P", "D"):
        for rank in (0, 1):
            controller.register_rank(group, stage, rank, 2000 + (stage == "D") * 10 + rank, cap)

    def report(stage, rank, kind, payload):
        key = (stage, rank)
        sequence[key] += 1
        return controller.rank_report(group, stage, rank,
            2000 + (stage == "D") * 10 + rank, str(sequence[key]), "p", kind, payload)

    controller.submit_request(RequestSpec("m", "r", (1, 2), 2, 120000,
        {"temperature": 0, "ignore_eos": False}))
    plan = BatchPlan(group, "p", "m", "P", (RequestRef("r", 0, 0),))
    reservation = controller.prepare_batch(plan)["reservation_id"]
    for rank in (0, 1):
        item = next(cmd for cmd in controller.poll_commands(group, "D", rank,
            2010 + rank) if cmd["kind"] == "PREPARE")["requests"][0]
        view = item["reservation_view"]
        controller.broker_register_destination(group, item["bootstrap_room"], rank,
            2010 + rank, "m", "r", 0, view["allocation_id"], view["version"],
            2, item["rank_block_table"], 1, 2)
    for stage in ("P", "D"):
        for rank in (0, 1):
            report(stage, rank, "READY", {"reservation_id": reservation})
    # Separate independently controlled CPU boundaries from fake GPU elapsed.
    controller.start_batch(group, "p")
    clock.now += .080
    for rank in (0, 1):
        prepare = next(cmd for cmd in controller.poll_commands(group, "P", rank,
            2000 + rank) if cmd["kind"] == "PREPARE")["requests"][0]
        room, view = prepare["bootstrap_room"], prepare["reservation_view"]
        controller.broker_stage(group, room, rank, 2000 + rank, {
            "bootstrap_room": room, "allocation_id": view["allocation_id"],
            "reserved_version": view["version"], "target_valid_tokens": 2,
            "rank": rank, "native_payload": {"rank": rank}})
    result = [{"request_id": "r", "attempt": 0, "token_id": 4, "valid_tokens": 2}]
    for rank, duration in ((0, 1.0), (1, 3.0)):
        report("P", rank, "GPU_DONE", {"reservation_id": reservation,
            "results": result, "duration_ms": duration, "gpu_forward_ms": duration / 2})
    clock.now += .005
    handoffs = {}
    for rank in (0, 1):
        item = next(cmd for cmd in controller.poll_commands(group, "D", rank,
            2010 + rank) if cmd["kind"] == "HANDOFF")["requests"][0]
        handoffs[rank] = item
        found = controller.broker_lookup(group, item["bootstrap_room"], rank,
            2010 + rank, "m", "r", 0, item["allocation_id"],
            item["version"], item["valid_tokens"])
        assert found["rank_block_table"] == item["rank_block_table"]
        controller.broker_ack(group, item["bootstrap_room"], rank, 2010 + rank,
            item["allocation_id"], item["version"], item["valid_tokens"],
            item["rank_block_table"])
    for rank in (0, 1):
        report("D", rank, "HANDOFF_READY", {"requests": [handoffs[rank]]})
    for rank in (0, 1):
        report("P", rank, "QUIESCED", {"reservation_id": reservation,
            "request_ids": ("r",)})
    item = controller.policy.feedback_history[-1]
    assert item.input_tokens == 2 and item.output_tokens == 1
    assert item.duration_ms == 3.0
    assert item.gpu_forward_ms == 1.5
    assert abs(item.pd_handoff_ms - 5.0) < 1e-6
    totals = controller.policy.feedback_totals["m"]["P"]["GENERATE"]
    assert totals["execution_wall_ms"] == 3.0 and totals["gpu_forward_ms"] == 1.5
    assert "duration_ms" not in totals


def check_total_model_capacity_before_admission():
    models = tuple(ModelDeployment(mid, "unused", "0" * 40, "unused", "0" * 40,
        1, 2, 256, 30, 18100 + i * 3, 18101 + i * 3, 18102 + i * 3)
        for i, mid in enumerate(("a", "b", "c")))
    store = HeadwiseStore({0: 12, 1: 9}, pool_mode="static", model_ids=("a", "b", "c"))
    controller = MuxController(DeploymentConfig(models, tp_size=2,
        kv_pool_mode="static", kv_bytes_per_rank=12 * 8192), store)
    cap = {"vocab_size": 32, "eos_token_ids": (9,), "native_pd": True, "headwise_kv": True}
    group = controller.group_id("a")
    for stage in ("P", "D"):
        for rank in (0, 1):
            controller.register_rank(group, stage, rank, 5000 + rank + (stage == "D") * 10, cap)
    assert controller.request_capacity("a")["capacity_tokens"] == 48
    assert controller.request_capacity("a")["blocks_by_rank"] == {0: 4, 1: 3}
    exact = RequestSpec("a", "exact", (1,) * 48, 1)
    assert controller.submit_request(exact)["state"] == "QUEUED"
    expect("KV_CAPACITY_EXHAUSTED", lambda: controller.submit_request(
        replace(exact, request_id="impossible", max_new_tokens=2)))
    assert ("a", "impossible") not in controller.requests
    # Temporary occupancy must not change this total-capacity admission check.
    from muxserve_sglang.contracts import KVTarget, RequestIdentity
    store.reserve_kv("occupied", (KVTarget(RequestIdentity("a", "holder"), 48),), "holder")
    assert controller.submit_request(RequestSpec("a", "wait", (1, 2), 4))["state"] == "QUEUED"
    # Real default 30B/static geometry: 87,381 blocks vs 99,840 required.
    assert (2 * 1024**3 // 8192 // 3 // (60 * 13)) * 16 == 1792
    assert ((1792 + 256 - 1 + 15) // 16) * 60 * 13 == 99840


def check_initialization_admission_and_event_cursor():
    from muxserve_sglang.driver import ControllerDriver
    f = Fixture(initializing=True)
    snapshot = f.controller.snapshot()
    assert snapshot["initializing"] and not snapshot["capacity_blocked"]
    assert not snapshot["recovery_protected"]
    assert snapshot["recovery_stats"] == {"cycles": 0, "automatic_suspensions": 0}
    expect("NOT_READY", lambda: f.submit("ordinary"))
    spec = RequestSpec("m", "init", (1, 2), 1)
    f.controller.submit_initialization_request(spec)
    plan, batch = f.prepare("init")
    f.ready_start(plan, batch)
    f.stage_prefill(plan)
    f.gpu(plan, batch, 4, 2)
    f.handoff(plan)
    f.batch_quiesce(plan, batch)
    f.cleanup("init")
    expect("NOT_READY", f.controller.finish_initialization)
    events = f.controller.poll_events(f.group)
    f.controller.ack_events(f.group, events[-1].event_seq)
    f.ack_all_commands()
    f.controller.forget_request("m", "init", 0)
    assert f.controller.policy.feedback_history
    f.controller.finish_initialization()
    assert not f.controller.initializing and not f.controller.policy.feedback_history
    assert f.controller.metrics()["scheduler_tick_count"] == 0
    assert f.controller.snapshot()["groups"]["m"]["acked_event_seq"] == events[-1].event_seq
    driver = ControllerDriver(f.controller, lambda event: None, schedule=False)
    assert driver.receivers["m"].last_seq == events[-1].event_seq
    driver.step()  # Must not replay retired initialization events from zero.
    assert f.submit("ordinary").request_id == "ordinary"


def check_pool_wide_and_static_recovery_scope():
    from muxserve_sglang.contracts import KVTarget
    for mode in ("shared", "static"):
        models = tuple(ModelDeployment(mid, "unused", "0" * 40, "unused", "0" * 40,
            1, 1, 128, 30, 18200 + i * 3, 18201 + i * 3, 18202 + i * 3)
            for i, mid in enumerate(("a", "b")))
        blocks = 2 if mode == "shared" else 4
        store = HeadwiseStore({0: blocks}, pool_mode=mode, model_ids=("a", "b"))
        clock = Clock()
        controller = MuxController(DeploymentConfig(models, tp_size=1, kv_pool_mode=mode,
            max_model_inflight=2, kv_bytes_per_rank=blocks * 8192), store, clock=clock,
            expected_lengths={"a": 16, "b": 16})
        cap = {"vocab_size": 32, "eos_token_ids": (9,), "native_pd": True, "headwise_kv": True}
        for index, model in enumerate(models):
            group = controller.group_id(model.model_id)
            for stage in ("P", "D"):
                controller.register_rank(group, stage, 0, 6000 + 10 * index + (stage == "D"), cap)

        def completed_prefill(model, rid):
            # Seed a CPU-only already-completed P boundary. No GPU is claimed.
            controller.submit_request(RequestSpec(model, rid, (1,) * 16, 2))
            request = controller.requests[(model, rid)]
            reservation = store.reserve_kv("seed-" + rid,
                (KVTarget(request.identity, 16),), controller._d_holder(request))
            store.start_kv_batch(reservation.reservation_id)
            aid = reservation.views[0].allocation_id
            view = store.finish_kv_batch(reservation.reservation_id, "SUCCESS", {aid: 16}, (0,))[0]
            request.allocation_id, request.valid_tokens, request.version = aid, 16, view.version
            request.output, request.state = [4], "DECODING"

        if mode == "static":
            completed_prefill("b", "outside-oldest")
        completed_prefill("a", "protected")
        victim_model = "b" if mode == "shared" else "a"
        completed_prefill(victim_model, "victim")
        trigger_model = "b" if mode == "shared" else "a"
        trigger_request = "victim" if mode == "shared" else "protected"
        group = controller.group_id(trigger_model)
        expect("OUT_OF_CAPACITY", lambda: controller.prepare_batch(BatchPlan(
            group, "trigger-recovery", trigger_model, "D", (RequestRef(trigger_request, 0, 1),))))
        controller.tick()
        scope = "shared" if mode == "shared" else "a"
        assert controller.recovery_protected == {scope: ("a", "protected")}
        assert controller.capacity_blocked == ({"a", "b"} if mode == "shared" else {"a"})
        assert controller.request_output(victim_model, "victim")["pending_reason"] == "suspend"
        if mode == "static":
            assert controller.request_output("b", "outside-oldest")["pending_reason"] is None
            outside = BatchPlan(controller.group_id("b"), "outside-decode", "b", "D",
                                (RequestRef("outside-oldest", 0, 1),))
            assert controller.prepare_batch(outside)["status"] == "PREPARING"
        victim_group = controller.groups[victim_model]
        cleanup = next(iter(victim_group.cleanups.values()))
        for stage in ("P", "D"):
            target = victim_group.registered[(stage, 0)]
            controller.rank_report(victim_group.group_id, stage, 0, target.pid, "1",
                cleanup.cleanup_id, "QUIESCED", {"cleanup_id": cleanup.cleanup_id,
                    "request_id": "victim", "attempt": 0})
        controller.tick()
        assert controller._single_decode_fits("a", "protected")
        assert controller.recovery_protected == {scope: ("a", "protected")}
        assert controller.snapshot()["recovery_stats"]["automatic_suspensions"] == 1
        record = controller.suspend_request(victim_model, "victim", 0)["record"]
        resumed = controller.resume_request(victim_model, "victim", record.resume_record_id)
        expect("OUT_OF_CAPACITY", lambda: controller.prepare_batch(BatchPlan(
            victim_group.group_id, "rebuild-paused", victim_model, "P",
            (RequestRef("victim", resumed["attempt"], 1),), "REBUILD")))
        for model_id in controller.capacity_blocked:
            controller.submit_request(RequestSpec(model_id, "queued", (1, 2), 2))
            expect("OUT_OF_CAPACITY", lambda model_id=model_id: controller.prepare_batch(BatchPlan(
                controller.group_id(model_id), "new-p-" + model_id, model_id, "P",
                (RequestRef("queued", 0, 0),))))


def check_independent_progress_watchdogs():
    for phase in ("handoff", "quiesce", "revoke", "cleanup"):
        f = Fixture()
        f.submit(max_new=3)
        if phase == "cleanup":
            f.controller.cancel_request("m", "r")
            deadline = next(iter(f.controller.groups["m"].cleanups.values())).deadline
            expected = "cleanup_watchdog"
        else:
            plan, batch = f.prepare()
            if phase == "revoke":
                f.controller.abort_batch(f.group, plan.batch_id, "test-revoke")
                deadline = f.controller.groups["m"].batches[plan.batch_id].quiesce_deadline
            else:
                f.ready_start(plan, batch)
                f.stage_prefill(plan)
                f.gpu(plan, batch, 4, 2)
                if phase == "handoff":
                    deadline = (f.controller.groups["m"].batches[plan.batch_id].gpu_completed_at
                                + f.controller.config.prepare_timeout_ms / 1000)
                else:
                    f.handoff(plan)
                    deadline = f.controller.groups["m"].batches[plan.batch_id].quiesce_deadline
            expected = "handoff_watchdog" if phase == "handoff" else "quiescence_watchdog"
        f.clock.now = deadline - .001
        f.controller.tick()
        assert not f.controller.groups["m"].fenced
        f.clock.now = deadline + .001
        f.controller.tick()
        assert f.controller.groups["m"].fenced == expected
        if phase == "cleanup":
            assert f.controller.groups["m"].cleanups  # No fabricated quiescence proof.
        else:
            assert f.store.snapshot()["allocations"]  # Unknown access remains pinned.


def check_prefill_credits_release_at_gpu_done():
    """P credits meter SM share: released once every P rank finished its
    forward, before the CPU-only handoff/quiesce; release stays idempotent."""
    f = Fixture()
    f.submit(max_new=2)
    plan, batch = f.prepare()
    held = lambda: any(any(refs) for refs in f.controller.credits.refs.values())
    assert held()
    f.ready_start(plan, batch)
    f.stage_prefill(plan)
    assert held()
    f.gpu(plan, batch, 4, 2)
    assert f.controller.snapshot()["groups"]["m"]["batches"][plan.batch_id]["status"] == "HANDOFF"
    assert not held() and f.controller.credits.can_allocate(10)
    f.handoff(plan)
    f.batch_quiesce(plan, batch)
    assert not held() and f.controller.credits.leases == {}


def check_decode_credits_release_at_gpu_done():
    """D credits are released once every D rank reported GPU_DONE (before quiesce)."""
    f = Fixture()
    f.submit(max_new=3)
    plan, batch = f.prepare()
    f.ready_start(plan, batch)
    f.stage_prefill(plan)
    f.gpu(plan, batch, 4, 2)
    f.handoff(plan)
    f.batch_quiesce(plan, batch)
    held = lambda: any(any(refs) for refs in f.controller.credits.refs.values())
    d_plan, d_batch = f.prepare(stage="D", batch_id="d")
    assert held()
    f.ready_start(d_plan, d_batch)
    f.gpu(d_plan, d_batch, 5, 3)
    status = f.controller.snapshot()["groups"]["m"]["batches"][d_plan.batch_id]["status"]
    assert status == "QUIESCE", status
    assert not held() and f.controller.credits.can_allocate(10)
    f.batch_quiesce(d_plan, d_batch)
    assert f.controller.credits.leases == {}


if __name__ == "__main__":
    check_first_token_and_replay()
    check_cancel_and_timeout()
    check_stop_freezes_committed_output()
    check_revoke_and_first_eos()
    check_suspend_rebuild()
    check_stale_rebuild_proposal_after_stop()
    check_event_gap_and_rank_death()
    check_deadline_and_terminal_suspend()
    check_suspended_cancel_and_deadline()
    check_retention_and_admission()
    check_all_waiting_rebuild_admission()
    check_recovery_and_reserved_capacity()
    check_uncertain_reservation_is_pinned()
    check_command_completion_reservation()
    check_synchronized_feedback_duration()
    check_total_model_capacity_before_admission()
    check_initialization_admission_and_event_cursor()
    check_pool_wide_and_static_recovery_scope()
    check_independent_progress_watchdogs()
    check_prefill_credits_release_at_gpu_done()
    check_decode_credits_release_at_gpu_done()
    print("controller CPU checks passed")
