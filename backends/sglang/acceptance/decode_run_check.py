"""CPU checks for multi-step D grants (decode runs) and command long-polls.

Simulated TP ranks live only in this file. They drive the real controller,
store, policy and driver; no Torch, SGLang, model or GPU is involved.
"""
from __future__ import annotations

from pathlib import Path
import sys
import threading
import time
import unittest
from types import SimpleNamespace
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from muxserve_sglang.config import DeploymentConfig, ModelDeployment
from muxserve_sglang.contracts import BatchPlan, IntegrationError, RequestRef, RequestSpec
from muxserve_sglang.controller import MuxController
from muxserve_sglang.driver import ControllerDriver
from muxserve_sglang.storage import HeadwiseStore


class Clock:
    def __init__(self):
        self.now = 100.0

    def __call__(self):
        return self.now


class Sim:
    """TP ranks for every model; commands are fetched exactly like real ranks."""

    def __init__(self, models=("a",), *, tp=2, run_steps=4, blocks=4096, park=False, **flags):
        deployments = tuple(ModelDeployment(m, "unused", "0" * 40, "unused", "0" * 40,
                                            2, tp, 128 * tp, 40, 18100 + 10 * i, 18101 + 10 * i,
                                            18102 + 10 * i)
                            for i, m in enumerate(models))
        self.tp = tp
        self.clock = Clock()
        self.store = HeadwiseStore({rank: blocks for rank in range(tp)})
        self.config = DeploymentConfig(deployments, tp_size=tp, kv_bytes_per_rank=blocks * 8192,
                                       decode_run_steps=run_steps, decode_run_park=park, **flags)
        self.controller = MuxController(self.config, self.store, clock=self.clock,
                                        expected_lengths={m: 16 for m in models})
        self.cap = {"vocab_size": 64, "eos_token_ids": (9,), "native_pd": True, "headwise_kv": True}
        self.acked, self.seq = {}, {}
        for model in models:
            for stage in ("P", "D"):
                for rank in range(tp):
                    self.controller.register_rank(self.group(model), stage, rank,
                                                  self.pid(model, stage, rank), self.cap)
                    self.acked[(model, stage, rank)] = 0
                    self.seq[(model, stage, rank)] = 0
        self.events = {m: [] for m in models}

    def group(self, model):
        return self.controller.group_id(model)

    def pid(self, model, stage, rank):
        return 1000 + 100 * sorted(self.controller.groups).index(model) + (10 if stage == "D" else 0) + rank

    def commands(self, model, stage, rank, kind=None):
        items = self.controller.poll_commands(self.group(model), stage, rank,
            self.pid(model, stage, rank), after_seq=self.acked[(model, stage, rank)])
        return [c for c in items if kind is None or c["kind"] == kind]

    def ack(self, model):
        for stage in ("P", "D"):
            for rank in range(self.tp):
                key = (model, stage, rank)
                pending = self.commands(model, stage, rank)
                if pending:
                    last = pending[-1]["command_seq"]
                    self.controller.poll_commands(self.group(model), stage, rank,
                        self.pid(model, stage, rank), after_seq=last, ack_seq=last)
                    self.acked[key] = last

    def report(self, model, stage, rank, batch_id, kind, payload):
        key = (model, stage, rank)
        reply = self.controller.rank_report(self.group(model), stage, rank, self.pid(model, stage, rank),
                                            str(self.seq[key] + 1), batch_id, kind, payload)
        assert reply["accepted"]
        self.seq[key] += 1
        return reply

    def submit(self, model, rid, max_new, input_ids=(1, 2, 3)):
        self.controller.submit_request(RequestSpec(model, rid, tuple(input_ids), max_new, 120000,
                                                   {"temperature": 0, "ignore_eos": True}))

    def _ready_all(self, model, batch_id, reservation):
        for stage in ("P", "D"):
            for rank in range(self.tp):
                self.report(model, stage, rank, batch_id, "READY", {"reservation_id": reservation})

    def prefill(self, model, rid, token):
        plan, out, reservation = self.prefill_ready(model, rid)
        self.controller.start_batch(self.group(model), plan.batch_id)
        return self.prefill_finish(model, rid, token, plan, out, reservation)

    def prefill_ready(self, model, rid):
        out = self.controller.request_output(model, rid)
        plan = BatchPlan(self.group(model), "p-" + rid + str(out["attempt"]), model, "P",
                         (RequestRef(rid, out["attempt"], out["output_count"]),))
        batch = self.controller.prepare_batch(plan)
        reservation = batch["reservation_id"]
        for rank in range(self.tp):
            cmd = next(c for c in self.commands(model, "D", rank, "PREPARE") if c["batch_id"] == plan.batch_id)
            item = cmd["requests"][0]
            view = item["reservation_view"]
            self.controller.broker_register_destination(self.group(model), item["bootstrap_room"], rank,
                self.pid(model, "D", rank), model, rid, item["attempt"], view["allocation_id"],
                view["version"], item["target_valid_tokens"], item["rank_block_table"], 0,
                item["target_valid_tokens"])
        self._ready_all(model, plan.batch_id, reservation)
        return plan, out, reservation

    def prefill_finish(self, model, rid, token, plan, out, reservation):
        for rank in range(self.tp):
            cmd = next(c for c in self.commands(model, "P", rank, "PREPARE") if c["batch_id"] == plan.batch_id)
            item = cmd["requests"][0]
            view = item["reservation_view"]
            room = item["bootstrap_room"]
            self.controller.broker_stage(self.group(model), room, rank, self.pid(model, "P", rank), {
                "bootstrap_room": room, "allocation_id": view["allocation_id"],
                "reserved_version": view["version"], "target_valid_tokens": item["target_valid_tokens"],
                "rank": rank, "native_payload": {"x": 1}})
        valid = len(self.controller.requests[(model, rid)].spec.input_ids)
        for rank in range(self.tp):
            self.report(model, "P", rank, plan.batch_id, "GPU_DONE", {
                "reservation_id": reservation, "duration_ms": 1.0, "gpu_forward_ms": 1.0,
                "results": [{"request_id": rid, "attempt": out["attempt"], "token_id": token,
                             "valid_tokens": valid}]})
        for rank in range(self.tp):
            item = self.commands(model, "D", rank, "HANDOFF")[-1]["requests"][0]
            self.controller.broker_ack(self.group(model), item["bootstrap_room"], rank,
                self.pid(model, "D", rank), item["allocation_id"], item["version"],
                item["valid_tokens"], item["rank_block_table"])
        for rank in range(self.tp):
            item = self.commands(model, "D", rank, "HANDOFF")[-1]["requests"][0]
            self.report(model, "D", rank, plan.batch_id, "HANDOFF_READY", {"requests": [item]})
        for rank in range(self.tp):
            self.report(model, "P", rank, plan.batch_id, "QUIESCED",
                        {"reservation_id": reservation, "request_ids": (rid,)})
        self.ack(model)
        return plan

    def prepare_decode(self, model, rid, steps):
        out = self.controller.request_output(model, rid)
        plan = BatchPlan(self.group(model), f"d-{rid}-{out['output_count']}", model, "D",
                         (RequestRef(rid, out["attempt"], out["output_count"]),), steps=steps)
        batch = self.controller.prepare_batch(plan)
        self._ready_all(model, plan.batch_id, batch["reservation_id"])
        self.controller.start_batch(self.group(model), plan.batch_id)
        return plan, batch["reservation_id"]

    def step(self, model, plan, rank, step, token, finished=False):
        rid = plan.requests[0].request_id
        start = self.controller.groups[model].batches[plan.batch_id].run_start_valid[rid]
        return self.controller.decode_step(self.group(model), rank, self.pid(model, "D", rank),
            plan.batch_id, step, [{"request_id": rid, "attempt": plan.requests[0].attempt,
                                   "token_id": token, "valid_tokens": start + step}], finished)

    def resume(self, model, plan, rank, step, wait_ms=0):
        return self.controller.decode_resume(self.group(model), rank, self.pid(model, "D", rank),
                                             plan.batch_id, step, wait_ms)["continue"]

    def finish_run(self, model, plan, reservation, steps, token, ranks=None):
        rid = plan.requests[0].request_id
        start = self.controller.groups[model].batches[plan.batch_id].run_start_valid[rid]
        for rank in (range(self.tp) if ranks is None else ranks):
            self.report(model, "D", rank, plan.batch_id, "GPU_DONE", {
                "reservation_id": reservation, "duration_ms": 2.0, "gpu_forward_ms": 1.0,
                "steps": steps, "results": [{"request_id": rid, "attempt": plan.requests[0].attempt,
                                             "token_id": token, "valid_tokens": start + steps}]})

    def quiesce(self, model, plan, reservation):
        rid = plan.requests[0].request_id
        for stage in ("P", "D"):
            for rank in range(self.tp):
                self.report(model, stage, rank, plan.batch_id, "QUIESCED",
                            {"reservation_id": reservation, "request_ids": (rid,)})

    def cleanup(self, model, rid):
        cmd = self.commands(model, "P", 0, "CLEANUP")[-1]
        for stage in ("P", "D"):
            for rank in range(self.tp):
                self.report(model, stage, rank, cmd["cleanup_id"], "QUIESCED", {
                    "cleanup_id": cmd["cleanup_id"], "request_id": rid, "attempt": cmd["attempt"]})
        self.ack(model)

    def tokens(self, model):
        events = self.controller.poll_events(self.group(model))
        return [e.payload["token_ids"][0] for e in events if e.kind == "TOKEN"]


def expect(code, fn):
    try:
        fn()
    except IntegrationError as exc:
        assert exc.code == code, (code, exc.code)
    else:
        raise AssertionError("Expected " + code)


class DecodeRunChecks(unittest.TestCase):
    def test_full_request_uses_runs_and_releases_everything(self):
        sim = Sim(run_steps=4)
        free = sim.store.rank_usage()[0]["free_blocks"]
        sim.submit("a", "r", max_new=6)
        sim.prefill("a", "r", 11)
        # The driver proposes a run bounded by config and remaining budget.
        seen = []
        driver = ControllerDriver(sim.controller, seen.append)
        driver.step()
        batch = next(b for b in sim.controller.groups["a"].batches.values()
                     if b.plan.stage == "D" and b.status == "PREPARING")
        self.assertEqual(batch.plan.steps, 4)
        prepares = sim.commands("a", "D", 0, "PREPARE")
        self.assertEqual(prepares[-1]["run_steps"], 4)
        # Capacity for the whole run was reserved once.
        view = prepares[-1]["requests"][0]["reservation_view"]
        self.assertGreaterEqual(view["capacity_tokens"], 3 + 4)
        for stage in ("P", "D"):
            for rank in range(sim.tp):
                sim.report("a", stage, rank, batch.plan.batch_id, "READY",
                           {"reservation_id": batch.reservation.reservation_id})
        sim.controller.start_batch(sim.group("a"), batch.plan.batch_id)
        plan, reservation = batch.plan, batch.reservation.reservation_id
        for step, token in enumerate((12, 13, 14, 15), start=1):
            decisions = [sim.step("a", plan, rank, step, token)["continue"] for rank in range(sim.tp)]
            self.assertEqual(len(set(decisions)), 1, "ranks must receive one decision")
            self.assertEqual(decisions[0], step < 4)
        sim.finish_run("a", plan, reservation, 4, 15)
        sim.quiesce("a", plan, reservation)
        request = sim.controller.requests[("a", "r")]
        self.assertEqual(request.output, [11, 12, 13, 14, 15])
        self.assertEqual(request.valid_tokens, 3 + 4)
        self.assertEqual(request.state, "DECODING")
        # The last token uses the original single-step protocol.
        driver.step()
        batch = next(b for b in sim.controller.groups["a"].batches.values() if b.status == "PREPARING")
        self.assertEqual(batch.plan.steps, 1)
        self.assertNotIn("run_steps", sim.commands("a", "D", 0, "PREPARE")[-1])
        for stage in ("P", "D"):
            for rank in range(sim.tp):
                sim.report("a", stage, rank, batch.plan.batch_id, "READY",
                           {"reservation_id": batch.reservation.reservation_id})
        sim.controller.start_batch(sim.group("a"), batch.plan.batch_id)
        for rank in range(sim.tp):
            sim.report("a", "D", rank, batch.plan.batch_id, "GPU_DONE", {
                "reservation_id": batch.reservation.reservation_id, "duration_ms": 1.0,
                "gpu_forward_ms": 1.0, "results": [{"request_id": "r", "attempt": 0, "token_id": 16,
                                                    "valid_tokens": 8}]})
        sim.quiesce("a", batch.plan, batch.reservation.reservation_id)
        sim.cleanup("a", "r")
        output = sim.controller.request_output("a", "r")
        self.assertEqual(output["state"], "FINISHED")
        self.assertEqual(output["output_token_ids"], (11, 12, 13, 14, 15, 16))
        driver.step()
        self.assertEqual([e.payload["token_ids"][0] for e in seen if e.kind == "TOKEN"],
                         [11, 12, 13, 14, 15, 16])
        self.assertEqual([e.payload["output_index"] for e in seen if e.kind == "TOKEN"], list(range(6)))
        self.assertEqual(seen[-1].kind, "TERMINAL")
        self.assertEqual(sim.store.rank_usage()[0]["free_blocks"], free)
        self.assertTrue(all(not any(refs) for refs in sim.controller.credits.refs.values()))
        totals = sim.controller.policy.feedback_totals["a"]["D"]["GENERATE"]
        self.assertEqual(totals["output_tokens"], 5)
        self.assertEqual(totals["batches"], 2)

    def test_step_reports_wake_only_the_driver_once_per_published_step(self):
        sim = Sim(run_steps=4)
        sim.submit("a", "r", max_new=8)
        sim.prefill("a", "r", 11)
        plan, _ = sim.prepare_decode("a", "r", 4)
        notified = []
        original = sim.controller._cv.notify_all
        sim.controller._cv.notify_all = lambda: notified.append(1) or original()
        try:
            sim.controller.wake.clear()
            self.assertTrue(sim.step("a", plan, 0, 1, 12)["continue"])
            self.assertFalse(sim.controller.wake.is_set())  # step not yet published
            self.assertTrue(sim.step("a", plan, 1, 1, 12)["continue"])
            self.assertTrue(sim.controller.wake.is_set())   # published: driver drains the token
            self.assertEqual(notified, [])                  # rank long polls were not woken
        finally:
            sim.controller._cv.notify_all = original
        self.assertEqual(sim.tokens("a")[-1:], [12])

    def test_run_yields_to_waiting_prefill_at_a_step_boundary(self):
        sim = Sim(("a", "b"), run_steps=8)
        sim.submit("a", "r", max_new=10)
        sim.prefill("a", "r", 11)
        plan, reservation = sim.prepare_decode("a", "r", 8)
        self.assertTrue(all(sim.step("a", plan, rank, 1, 12)["continue"] for rank in range(sim.tp)))
        sim.submit("b", "x", max_new=4)  # Needs 10 credits; the run holds 4.
        self.assertFalse(sim.step("a", plan, 0, 2, 13)["continue"])
        # A slower rank receives the memoized decision, not a fresh one.
        sim.controller.requests.pop(("b", "x"))
        self.assertFalse(sim.step("a", plan, 1, 2, 13)["continue"])
        sim.finish_run("a", plan, reservation, 2, 13)
        sim.quiesce("a", plan, reservation)
        request = sim.controller.requests[("a", "r")]
        self.assertEqual((request.output, request.valid_tokens, request.state), ([11, 12, 13], 5, "DECODING"))
        self.assertTrue(all(not any(refs) for refs in sim.controller.credits.refs.values()))

    def two_waiting(self, **flags):
        sim = Sim(run_steps=8, **flags)
        sim.submit("a", "r", max_new=10)
        sim.submit("a", "q", max_new=6)
        sim.prefill("a", "r", 11)
        sim.prefill("a", "q", 21)
        plan, reservation = sim.prepare_decode("a", "r", 8)
        return sim, plan, reservation

    def test_run_ends_for_a_waiting_decode_and_the_next_batch_holds_both(self):
        sim, plan, reservation = self.two_waiting(max_model_inflight=2, decode_run_merge=True)
        self.assertFalse(sim.step("a", plan, 0, 1, 12)["continue"])
        self.assertFalse(sim.step("a", plan, 1, 1, 12)["continue"])
        sim.finish_run("a", plan, reservation, 1, 12)
        sim.quiesce("a", plan, reservation)
        ControllerDriver(sim.controller, lambda event: None).step()
        batch = next(b for b in sim.controller.groups["a"].batches.values() if b.status == "PREPARING")
        self.assertEqual({ref.request_id for ref in batch.plan.requests}, {"r", "q"})
        self.assertEqual(batch.plan.steps, 5)  # q has the fewest tokens left

    def test_a_large_batch_run_fits_the_event_window(self):
        sim = Sim(run_steps=128, max_model_inflight=16, decode_run_merge=True)
        sim.controller.policy.limits["a"] = 16
        names = ["r%d" % index for index in range(10)]
        for index, name in enumerate(names):
            sim.submit("a", name, max_new=200)
            sim.prefill("a", name, 11 + index)
        controller = sim.controller
        outputs = [controller.request_output("a", name) for name in names]
        driver = ControllerDriver(controller, lambda event: None)
        steps = driver._run_steps(SimpleNamespace(model_id="a", stage="D", purpose="GENERATE"), outputs)
        window = controller.groups["a"].outbox.maximum_pending
        self.assertEqual(steps, (window - 3) // 10 - 1)  # 128 steps would need 1292 slots
        self.assertLessEqual((steps + 1) * 10 + 2, window - 1)
        events = controller.poll_events(sim.group("a"), 0)  # the prefills' events, delivered
        controller.ack_events(sim.group("a"), events[-1].event_seq)
        plan = BatchPlan(sim.group("a"), "d-ten", "a", "D", tuple(
            RequestRef(o["request_id"], o["attempt"], o["output_count"]) for o in outputs), steps=steps)
        controller.prepare_batch(plan)  # the event reservation fits
        self.assertEqual(driver._run_steps(SimpleNamespace(model_id="a", stage="D", purpose="GENERATE"),
                                           outputs[:2]), 128)

    def test_a_run_returns_its_unused_reservation_with_the_switch(self):
        for trim in (False, True):
            # q waits to decode, so r's eight-step run ends after its first step.
            sim = Sim(run_steps=8, max_model_inflight=2, decode_run_merge=True, decode_kv_trim=trim)
            sim.submit("a", "r", max_new=20, input_ids=tuple(range(1, 16)))
            sim.submit("a", "q", max_new=6)
            sim.prefill("a", "r", 11)
            sim.prefill("a", "q", 21)
            plan, reservation = sim.prepare_decode("a", "r", 8)
            store = sim.controller.store
            identity = sim.controller.requests[("a", "r")].identity
            aid = next(a for a, v in store.snapshot()["allocations"].items() if v["identity"] == identity)
            before = store.snapshot()["allocations"][aid]["capacity_tokens"]
            free_before = min(v["free_blocks"] for v in store.rank_usage().values())
            self.assertFalse(sim.step("a", plan, 0, 1, 12)["continue"])
            self.assertFalse(sim.step("a", plan, 1, 1, 12)["continue"])
            sim.finish_run("a", plan, reservation, 1, 12)
            sim.quiesce("a", plan, reservation)
            after = store.snapshot()["allocations"][aid]["capacity_tokens"]
            valid = sim.controller.requests[("a", "r")].valid_tokens
            self.assertEqual(before, 32)  # 15 prompt tokens + 8 reserved steps
            self.assertEqual(after, 16 * ((valid + 15) // 16) if trim else before)
            self.assertEqual(min(v["free_blocks"] for v in store.rank_usage().values()) > free_before, trim)
            store.assert_integrity()

    def test_a_decode_that_cannot_get_kv_preempts_a_younger_request_with_the_switch(self):
        for relaxed in (False, True):
            sim = Sim(run_steps=8, max_model_inflight=2, decode_run_merge=True,
                      capacity_recovery_all_decode=relaxed)
            sim.submit("a", "r", max_new=10)
            sim.submit("a", "q", max_new=10)
            sim.prefill("a", "r", 11)
            sim.prefill("a", "q", 21)
            controller = sim.controller
            out = controller.request_output("a", "r")
            plan = BatchPlan(sim.group("a"), "d-full", "a", "D", (RequestRef("r", out["attempt"], out["output_count"]),), steps=1)
            with mock.patch.object(controller.store, "reserve_kv",
                                   side_effect=IntegrationError("OUT_OF_CAPACITY", "full")), \
                    mock.patch.object(controller, "_single_decode_fits", return_value=True):
                with self.assertRaises(IntegrationError):
                    controller.prepare_batch(plan)
            q = controller.requests[("a", "q")]
            self.assertEqual(q.pending_reason == "suspend", relaxed)
            self.assertEqual(controller.recovery_suspensions, 1 if relaxed else 0)
            self.assertEqual("a" in controller.capacity_blocked, relaxed)

    def test_capacity_recovery_restricts_decode_only_without_the_switch(self):
        for relaxed, expected in ((False, ()), (True, ("q",))):
            sim, plan, reservation = self.two_waiting(max_model_inflight=2, decode_run_merge=True,
                                                      capacity_recovery_all_decode=relaxed)
            controller = sim.controller
            key = controller._recovery_key("a")
            controller.recovery_protected[key] = ("a", "r")
            controller.recovery_owners[key] = (("a", "r"), ("a", "q"))
            controller.capacity_blocked.add("a")
            demand = next(d for d in controller._snapshot_policy().demands if d.model_id == "a")
            self.assertEqual(demand.decode, expected)
            controller.requests[("a", "q")].state = "FINISHED"
            controller._advance_capacity_recovery()
            self.assertEqual(key in controller.recovery_protected, not relaxed)
            self.assertEqual("a" in controller.capacity_blocked, not relaxed)

    def test_a_request_stays_known_until_its_batch_is_fed_back(self):
        sim, plan, reservation = self.two_waiting(max_model_inflight=2, decode_run_merge=True)
        controller = sim.controller
        group, request = controller.groups["a"], controller.requests[("a", "r")]
        batch = group.batches[plan.batch_id]
        request.state = "TIMED_OUT"
        group.outbox.pending.clear()
        with mock.patch.object(controller, "_active", return_value=None), \
                mock.patch.object(controller, "_command_refs_request", return_value=False), \
                mock.patch.object(controller.store, "allocation_identities", return_value=set()):
            for status, fed_back in (("RUNNING", False), ("DONE", False)):
                batch.status, batch.feedback_sent = status, fed_back
                with self.assertRaises(IntegrationError) as caught:
                    controller.forget_request("a", "r", request.attempt)
                self.assertEqual(caught.exception.code, "NOT_READY")
            batch.feedback_sent = True
            self.assertTrue(controller.forget_request("a", "r", request.attempt)["forgotten"])

    def test_a_run_ends_for_a_waiting_decode_only_below_the_policy_limit(self):
        sim, plan, reservation = self.two_waiting(max_model_inflight=4, decode_run_merge=True)
        sim.controller.policy.limits["a"] = 1  # the batch holding r is already full
        self.assertTrue(sim.step("a", plan, 0, 1, 12)["continue"])
        sim, plan, reservation = self.two_waiting(max_model_inflight=4, decode_run_merge=True)
        sim.controller.policy.limits["a"] = 4
        self.assertFalse(sim.step("a", plan, 0, 1, 12)["continue"])

    def test_a_prefill_of_the_same_model_ends_the_run_and_holds_decode(self):
        sim = Sim(run_steps=8, park=True, max_model_inflight=2, decode_run_merge=True)
        sim.submit("a", "r", max_new=10)
        sim.prefill("a", "r", 11)
        plan, reservation = sim.prepare_decode("a", "r", 8)
        self.assertTrue(all(sim.step("a", plan, rank, 1, 12)["continue"] for rank in range(sim.tp)))
        sim.submit("a", "q", max_new=6)
        p_plan, p_out, p_reservation = sim.prefill_ready("a", "q")
        self.assertFalse(sim.step("a", plan, 0, 2, 13)["continue"])
        self.assertFalse(sim.step("a", plan, 1, 2, 13)["continue"])
        sim.finish_run("a", plan, reservation, 2, 13)
        sim.quiesce("a", plan, reservation)
        self.assertIn(("a", "D"), sim.controller._snapshot_policy().busy)
        sim.controller.start_batch(sim.group("a"), p_plan.batch_id)
        sim.prefill_finish("a", "q", 21, p_plan, p_out, p_reservation)
        self.assertNotIn(("a", "D"), sim.controller._snapshot_policy().busy)
        ControllerDriver(sim.controller, lambda event: None).step()
        batch = next(b for b in sim.controller.groups["a"].batches.values() if b.status == "PREPARING")
        self.assertEqual({ref.request_id for ref in batch.plan.requests}, {"r", "q"})

    PAIR = dict(run_steps=8, park=True, max_model_inflight=2, decode_run_merge=True,
                inline_start=True, inline_prepare=True, prefill_pair_hold_ms=5)

    def test_two_arrivals_at_an_idle_model_share_one_prefill(self):
        sim = Sim(**self.PAIR)
        sim.submit("a", "r", max_new=4)
        sim.clock.now += 0.002
        sim.submit("a", "q", max_new=4)
        requests = sim.controller.requests
        self.assertAlmostEqual(requests[("a", "r")].hold_until, 100.005)
        self.assertEqual(requests[("a", "q")].hold_until, requests[("a", "r")].hold_until)
        driver = ControllerDriver(sim.controller, lambda event: None)
        driver.step()
        self.assertEqual(sim.controller.groups["a"].batches, {})
        self.assertFalse(sim.controller._prefill_eligible())
        sim.clock.now = requests[("a", "r")].hold_until
        self.assertTrue(sim.controller._prefill_eligible())
        driver.step()
        batch = next(iter(sim.controller.groups["a"].batches.values()))
        self.assertEqual((batch.plan.stage, [ref.request_id for ref in batch.plan.requests]), ("P", ["r", "q"]))

    def test_a_request_goes_alone_after_the_window(self):
        sim = Sim(**self.PAIR)
        sim.submit("a", "r", max_new=4)
        sim.clock.now = 100.006
        driver = ControllerDriver(sim.controller, lambda event: None)
        driver.step()
        batch = next(iter(sim.controller.groups["a"].batches.values()))
        self.assertEqual([ref.request_id for ref in batch.plan.requests], ["r"])

    def test_no_wait_while_the_model_is_active(self):
        sim = Sim(**self.PAIR)
        sim.submit("a", "r", max_new=10)
        sim.clock.now = 100.006
        sim.prefill("a", "r", 11)
        sim.submit("a", "q", max_new=4)
        self.assertIsNone(sim.controller.requests[("a", "q")].hold_until)

    def test_expired_waiter_does_not_extend_the_window(self):
        sim = Sim(**self.PAIR)
        sim.submit("a", "r", max_new=4)
        sim.clock.now = 100.006
        sim.submit("a", "q", max_new=4)
        self.assertIsNone(sim.controller.requests[("a", "q")].hold_until)

    def test_other_models_do_not_share_a_window(self):
        sim = Sim(models=("a", "b"), **self.PAIR)
        sim.submit("a", "r", max_new=4)
        sim.clock.now += 0.002
        sim.submit("b", "q", max_new=4)
        self.assertAlmostEqual(sim.controller.requests[("b", "q")].hold_until, 100.007)

    def test_gateway_thread_waits_out_the_window_then_wakes_the_driver(self):
        sim = Sim(**self.PAIR)
        sim.submit("a", "r", max_new=4)
        sim.controller.wake.clear()
        started = time.monotonic()
        sim.controller.prepare_submitted("a", "r")
        self.assertGreaterEqual(time.monotonic() - started, 0.004)
        self.assertTrue(sim.controller.wake.is_set())
        self.assertEqual(sim.controller.groups["a"].batches, {})

    def test_without_the_switch_a_request_is_never_held(self):
        sim = Sim(run_steps=8, max_model_inflight=2, decode_run_merge=True)
        sim.submit("a", "r", max_new=4)
        self.assertIsNone(sim.controller.requests[("a", "r")].hold_until)

    def test_without_the_switch_a_run_ignores_a_waiting_decode(self):
        sim, plan, _ = self.two_waiting(max_model_inflight=2)
        self.assertTrue(all(sim.step("a", plan, rank, 1, 12)["continue"] for rank in range(sim.tp)))

    def test_a_full_batch_does_not_end_for_a_waiting_decode(self):
        sim = Sim(run_steps=8, max_model_inflight=1, decode_run_merge=True)
        sim.submit("a", "r", max_new=10)
        sim.prefill("a", "r", 11)
        plan, _ = sim.prepare_decode("a", "r", 8)
        sim.submit("a", "q", max_new=6)
        sim.controller.requests[("a", "q")].state = "DECODING"
        self.assertTrue(all(sim.step("a", plan, rank, 1, 12)["continue"] for rank in range(sim.tp)))

    def test_a_finished_or_cancelled_waiter_does_not_end_the_run(self):
        sim, plan, _ = self.two_waiting(max_model_inflight=2, decode_run_merge=True)
        sim.controller.requests[("a", "q")].pending_reason = "cancel"
        self.assertTrue(all(sim.step("a", plan, rank, 1, 12)["continue"] for rank in range(sim.tp)))

    def parked(self):
        sim = Sim(("a", "b"), run_steps=8, park=True)
        sim.submit("a", "r", max_new=10)
        sim.prefill("a", "r", 11)
        plan, reservation = sim.prepare_decode("a", "r", 8)
        self.assertTrue(all(sim.step("a", plan, rank, 1, 12)["continue"] for rank in range(sim.tp)))
        sim.submit("b", "x", max_new=4)
        credits = sim.controller.credits
        # The prefill prepares while the run keeps decoding (no credits yet).
        p_plan, p_out, p_reservation = sim.prefill_ready("b", "x")
        self.assertNotIn(p_plan.batch_id, credits.leases)
        self.assertIn(plan.batch_id, credits.leases)
        # READY, it needs all credits: START waits, the run parks at step 2.
        expect("NOT_READY", lambda: sim.controller.start_batch(sim.group("b"), p_plan.batch_id))
        self.assertEqual(sim.step("a", plan, 0, 2, 13)["continue"], "PARK")
        # Until every rank finished step 2 the run still uses its SMs.
        expect("NOT_READY", lambda: sim.controller.start_batch(sim.group("b"), p_plan.batch_id))
        self.assertEqual(sim.step("a", plan, 1, 2, 13)["continue"], "PARK")
        batch = sim.controller.groups["a"].batches[plan.batch_id]
        self.assertEqual(batch.status, "RUNNING")
        self.assertIsNotNone(batch.parked_at)
        self.assertNotIn(plan.batch_id, credits.leases)
        # Free credits alone never resume a run while the prefill awaits them.
        self.assertEqual(sim.resume("a", plan, 0, 2), "PARK")
        prefill = (p_plan, p_out, p_reservation)
        return sim, plan, reservation, prefill

    def test_prefill_prepares_while_decoding_and_parks_runs_only_when_ready(self):
        sim = Sim(("a", "b"), run_steps=8, park=True)
        sim.submit("a", "r", max_new=10)
        sim.prefill("a", "r", 11)
        plan, _ = sim.prepare_decode("a", "r", 8)
        sim.submit("b", "x", max_new=4)
        # Eligible but not prepared: decoding continues.
        self.assertIs(sim.step("a", plan, 0, 1, 12)["continue"], True)
        self.assertIs(sim.step("a", plan, 1, 1, 12)["continue"], True)
        p_plan, _, _ = sim.prefill_ready("b", "x")
        self.assertEqual(sim.step("a", plan, 0, 2, 13)["continue"], "PARK")
        # A new decode batch cannot take the credits the ready prefill awaits.
        self.assertTrue(sim.controller._prefill_awaiting_credits())
        self.assertNotIn(p_plan.batch_id, sim.controller.credits.leases)

    def test_ready_prefill_blocks_new_decode_batches_until_it_starts(self):
        sim = Sim(("a", "b", "c"), run_steps=8, park=True)
        sim.submit("c", "y", max_new=10)
        sim.prefill("c", "y", 31)
        sim.submit("b", "x", max_new=4)
        p_plan, p_out, p_reservation = sim.prefill_ready("b", "x")
        expect("OUT_OF_CREDITS", lambda: sim.prepare_decode("c", "y", 8))
        sim.controller.start_batch(sim.group("b"), p_plan.batch_id)  # credits were free
        expect("OUT_OF_CREDITS", lambda: sim.prepare_decode("c", "y", 8))  # P holds all 10
        sim.prefill_finish("b", "x", 21, p_plan, p_out, p_reservation)
        plan, _ = sim.prepare_decode("c", "y", 8)  # the prefill's forward finished
        self.assertIn(plan.batch_id, sim.controller.credits.leases)

    def test_parked_run_frees_credits_and_resumes_after_the_prefill_forward(self):
        sim, plan, reservation, (p_plan, p_out, p_reservation) = self.parked()
        sim.controller.start_batch(sim.group("b"), p_plan.batch_id)
        self.assertIn(p_plan.batch_id, sim.controller.credits.leases)
        self.assertEqual(sim.resume("a", plan, 0, 2), "PARK")  # prefill holds credits
        sim.prefill_finish("b", "x", 21, p_plan, p_out, p_reservation)
        self.assertEqual(sim.resume("a", plan, 0, 2), True)
        self.assertEqual(sim.resume("a", plan, 1, 2), True)  # replayed decision
        self.assertIn(plan.batch_id, sim.controller.credits.leases)
        self.assertIsNone(sim.controller.groups["a"].batches[plan.batch_id].parked_at)
        for step, token in enumerate((14, 15, 16, 17, 18, 19), start=3):
            decisions = {sim.step("a", plan, rank, step, token)["continue"] for rank in range(sim.tp)}
            self.assertEqual(decisions, {step < 8})
        sim.finish_run("a", plan, reservation, 8, 19)
        sim.quiesce("a", plan, reservation)
        request = sim.controller.requests[("a", "r")]
        self.assertEqual(request.output, [11, 12, 13, 14, 15, 16, 17, 18, 19])
        self.assertEqual((request.valid_tokens, request.state), (3 + 8, "DECODING"))
        self.assertEqual(sim.tokens("a"), [11, 12, 13, 14, 15, 16, 17, 18, 19])
        self.assertNotIn(plan.batch_id, sim.controller.credits.leases)
        self.assertEqual(sim.controller.requests[("b", "x")].output, [21])

    def test_parked_run_ends_like_a_yield_on_cancel_or_after_the_park_limit(self):
        for reason in ("cancel", "limit"):
            with self.subTest(reason=reason):
                sim, plan, reservation, _ = self.parked()
                if reason == "cancel":
                    sim.controller.cancel_request("a", "r")
                else:
                    sim.clock.now += 2.5
                self.assertIs(sim.resume("a", plan, 1, 2), False)
                self.assertIs(sim.resume("a", plan, 0, 2), False)
                expect("REPORT_GAP", lambda: sim.step("a", plan, 0, 3, 14))
                sim.finish_run("a", plan, reservation, 2, 13)
                sim.quiesce("a", plan, reservation)
                request = sim.controller.requests[("a", "r")]
                self.assertEqual(request.output, [11, 12, 13])
                self.assertNotIn(plan.batch_id, sim.controller.credits.leases)

    def test_resume_long_poll_wakes_when_the_prefill_releases_credits(self):
        sim, plan, _, (p_plan, p_out, p_reservation) = self.parked()
        result = {}

        def wait():
            result["value"] = sim.controller.decode_resume(
                sim.group("a"), 0, sim.pid("a", "D", 0), plan.batch_id, 2, 1000)["continue"]
            result["at"] = time.monotonic()

        waiter = threading.Thread(target=wait)
        waiter.start()
        time.sleep(0.05)
        self.assertNotIn("value", result)
        sim.controller.start_batch(sim.group("b"), p_plan.batch_id)
        time.sleep(0.05)
        self.assertNotIn("value", result)  # the prefill now holds the credits
        sim.prefill_finish("b", "x", 21, p_plan, p_out, p_reservation)
        released = time.monotonic()
        waiter.join(2)
        self.assertIs(result.get("value"), True)
        self.assertLess(result["at"] - released, 0.5)

    def test_resume_requires_a_parked_or_decided_step(self):
        sim = Sim(run_steps=4, park=True)
        sim.submit("a", "r", max_new=8)
        sim.prefill("a", "r", 11)
        plan, _ = sim.prepare_decode("a", "r", 4)
        expect("STALE_REPORT", lambda: sim.resume("a", plan, 0, 1))  # not reported yet
        self.assertIs(sim.step("a", plan, 0, 1, 12)["continue"], True)
        self.assertIs(sim.resume("a", plan, 0, 1), True)
        expect("INVALID_ARGUMENT", lambda: sim.resume("a", plan, 0, 1, 5000))

    def test_gpu_done_may_arrive_before_a_slower_rank_reports_the_last_step(self):
        sim = Sim(run_steps=2)
        sim.submit("a", "r", max_new=5)
        sim.prefill("a", "r", 11)
        plan, reservation = sim.prepare_decode("a", "r", 2)
        for rank in range(sim.tp):
            sim.step("a", plan, rank, 1, 12)
        self.assertFalse(sim.step("a", plan, 0, 2, 13)["continue"])
        sim.finish_run("a", plan, reservation, 2, 13, ranks=(0,))
        self.assertEqual(sim.controller.groups["a"].batches[plan.batch_id].status, "RUNNING")
        sim.step("a", plan, 1, 2, 13)
        sim.finish_run("a", plan, reservation, 2, 13, ranks=(1,))
        self.assertEqual(sim.controller.groups["a"].batches[plan.batch_id].status, "QUIESCE")
        self.assertEqual(sim.controller.requests[("a", "r")].output, [11, 12, 13])

    def test_rank_disagreement_or_changed_replay_fences(self):
        sim = Sim(run_steps=3)
        sim.submit("a", "r", max_new=5)
        sim.prefill("a", "r", 11)
        plan, _ = sim.prepare_decode("a", "r", 3)
        first = sim.step("a", plan, 0, 1, 12)
        self.assertEqual(sim.step("a", plan, 0, 1, 12), first, "replay returns the same decision")
        expect("IDENTITY_CONFLICT", lambda: sim.step("a", plan, 1, 1, 13))
        self.assertTrue(sim.controller.groups["a"].fenced)
        self.assertEqual(sim.controller.requests[("a", "r")].output, [11])

    def test_cancel_stops_the_run_and_withholds_later_tokens(self):
        sim = Sim(run_steps=4)
        sim.submit("a", "r", max_new=8)
        sim.prefill("a", "r", 11)
        plan, reservation = sim.prepare_decode("a", "r", 4)
        for rank in range(sim.tp):
            self.assertTrue(sim.step("a", plan, rank, 1, 12)["continue"])
        sim.controller.cancel_request("a", "r", "client")
        decisions = [sim.step("a", plan, rank, 2, 13)["continue"] for rank in range(sim.tp)]
        self.assertEqual(decisions, [False, False])
        sim.finish_run("a", plan, reservation, 2, 13)
        sim.quiesce("a", plan, reservation)
        sim.cleanup("a", "r")
        output = sim.controller.request_output("a", "r")
        self.assertEqual(output["state"], "CANCELLED")
        self.assertEqual(output["output_token_ids"], (11, 12))

    def test_ungranted_steps_and_invalid_runs_are_rejected(self):
        sim = Sim(run_steps=4)
        sim.submit("a", "r", max_new=3)
        sim.prefill("a", "r", 11)
        out = sim.controller.request_output("a", "r")
        ref = (RequestRef("r", out["attempt"], out["output_count"]),)
        expect("STALE_ATTEMPT", lambda: sim.controller.prepare_batch(
            BatchPlan(sim.group("a"), "too-long", "a", "D", ref, steps=3)))
        expect("INVALID_ARGUMENT", lambda: sim.controller.prepare_batch(
            BatchPlan(sim.group("a"), "too-many", "a", "D", ref, steps=5)))
        expect("INVALID_ARGUMENT", lambda: sim.controller.prepare_batch(
            BatchPlan(sim.group("a"), "p-run", "a", "P", ref, steps=2)))
        plan, reservation = sim.prepare_decode("a", "r", 2)
        expect("REPORT_GAP", lambda: sim.step("a", plan, 0, 2, 13))
        expect("INVALID_REPORT", lambda: sim.finish_run("a", plan, reservation, 1, 12, ranks=(0,)))

    def test_single_step_config_keeps_original_protocol(self):
        sim = Sim(run_steps=1)
        sim.submit("a", "r", max_new=4)
        sim.prefill("a", "r", 11)
        ControllerDriver(sim.controller, lambda event: None).step()
        batch = next(b for b in sim.controller.groups["a"].batches.values() if b.status == "PREPARING")
        self.assertEqual(batch.plan.steps, 1)
        expect("STALE_REPORT", lambda: sim.controller.decode_step(
            sim.group("a"), 0, sim.pid("a", "D", 0), batch.plan.batch_id, 1, [], False))


class LongPollChecks(unittest.TestCase):
    def setUp(self):
        self.sim = Sim()
        self.args = (self.sim.group("a"), "P", 0, self.sim.pid("a", "P", 0))

    def test_idle_poll_waits_then_returns_empty(self):
        started = time.monotonic()
        reply = self.sim.controller.poll_commands_wait(*self.args, 0, None, 60, None)
        self.assertEqual(reply["commands"], ())
        self.assertGreaterEqual(time.monotonic() - started, 0.05)

    def test_command_or_broker_epoch_wakes_waiter(self):
        controller = self.sim.controller
        epoch = controller.poll_commands_wait(*self.args, 0, None, 0, None)["epoch"]
        results = []
        waiter = threading.Thread(target=lambda: results.append(
            (controller.poll_commands_wait(*self.args, 0, None, 1000, epoch), time.monotonic())))
        started = time.monotonic()
        waiter.start()
        time.sleep(0.05)
        controller._notify(controller.groups["a"])
        waiter.join(2)
        reply, finished = results[0]
        self.assertLess(finished - started, 0.5)
        self.assertEqual(reply["epoch"], epoch + 1)
        results.clear()
        waiter = threading.Thread(target=lambda: results.append(
            controller.poll_commands_wait(*self.args, 0, None, 1000, epoch + 1)))
        waiter.start()
        time.sleep(0.05)
        with controller.lock:
            controller._command(controller.groups["a"], ("P", 0), "NOOP", {})
        waiter.join(2)
        self.assertEqual([c["kind"] for c in results[0]["commands"]], ["NOOP"])

    def test_invalid_wait_is_rejected(self):
        expect("INVALID_ARGUMENT", lambda: self.sim.controller.poll_commands_wait(*self.args, 0, None, 5000, None))


if __name__ == "__main__":
    unittest.main(verbosity=2)
