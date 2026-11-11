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


class StepJoinChecks(unittest.TestCase):
    """decode_step_join / decode_kv_per_page: a run takes prefill work, joined
    requests and new KV pages at its step boundaries."""

    # Prefills prepare beside a running decode only with decode_run_park.
    JOIN = dict(park=True, max_model_inflight=2, decode_run_merge=True, decode_step_join=True)

    @staticmethod
    def batch(sim, batch_id):
        return sim.controller.groups["a"].batches[batch_id]

    def step(self, sim, batch_id, rank, step, tokens, finished=False):
        batch = self.batch(sim, batch_id)
        results = [{"request_id": ref.request_id, "attempt": ref.attempt, "token_id": tokens[ref.request_id],
                    "valid_tokens": batch.run_start_valid[ref.request_id] + step} for ref in batch.members(step)]
        return sim.controller.decode_step(sim.group("a"), rank, sim.pid("a", "D", rank), batch_id, step,
                                          results, finished)

    def resume(self, sim, batch_id, rank, step, **kwargs):
        return sim.controller.decode_resume(sim.group("a"), rank, sim.pid("a", "D", rank), batch_id, step,
                                            0, **kwargs)

    def finish(self, sim, batch_id, reservation, steps, tokens):
        batch = self.batch(sim, batch_id)
        results = [{"request_id": ref.request_id, "attempt": ref.attempt, "token_id": tokens[ref.request_id],
                    "valid_tokens": batch.run_start_valid[ref.request_id] + steps} for ref in batch.members(steps)]
        extra = {"token_digest": 0} if sim.config.decode_leader_reports else {}
        for rank in range(sim.tp):
            sim.report("a", "D", rank, batch_id, "GPU_DONE", {
                "reservation_id": reservation, "duration_ms": 2.0, "gpu_forward_ms": 1.0,
                "steps": steps, "results": results, **extra})
        ids = tuple(ref.request_id for ref in batch.plan.requests)
        for stage in ("P", "D"):
            for rank in range(sim.tp):
                sim.report("a", stage, rank, batch_id, "QUIESCED", {"reservation_id": reservation, "request_ids": ids})

    def test_a_handed_off_request_joins_the_running_run(self):
        sim = Sim(run_steps=8, **self.JOIN)
        sim.submit("a", "r", max_new=10)
        sim.submit("a", "q", max_new=3)
        sim.prefill("a", "r", 11)
        sim.prefill("a", "q", 21)
        plan, reservation = sim.prepare_decode("a", "r", 8)  # q waits to decode, outside the run
        q = sim.controller.requests[("a", "q")]
        replies = [self.step(sim, plan.batch_id, rank, 1, {"r": 12}) for rank in range(sim.tp)]
        for rank, reply in enumerate(replies):
            self.assertIs(reply["continue"], True)
            (item,) = reply["join"]
            self.assertEqual((item["request_id"], item["start_valid"], item["output_token_ids"]),
                             ("q", q.valid_tokens - 1, (21,)))
            self.assertEqual(set(item["view"]["block_tables"]), {rank})  # only this rank's table
        self.assertEqual([ref.request_id for ref in self.batch(sim, plan.batch_id).plan.requests], ["r", "q"])
        self.assertEqual(q.state, "RUNNING")
        for rank in range(sim.tp):
            self.assertIs(self.step(sim, plan.batch_id, rank, 2, {"r": 13, "q": 22})["continue"], True)
        for rank in range(sim.tp):  # q reaches its length limit at step 3
            self.assertIs(self.step(sim, plan.batch_id, rank, 3, {"r": 14, "q": 23}, True)["continue"], False)
        self.finish(sim, plan.batch_id, reservation, 3, {"r": 14, "q": 23})
        r = sim.controller.requests[("a", "r")]
        self.assertEqual((r.output, r.valid_tokens, r.state), ([11, 12, 13, 14], 6, "DECODING"))
        self.assertEqual((q.output, q.finish_reason), ([21, 22, 23], "length"))
        self.assertEqual(sim.tokens("a")[-5:], [12, 13, 22, 14, 23])
        sim.store.assert_integrity()
        holders = {v["identity"].request_id: v["holders"] for v in sim.store.snapshot()["allocations"].values()}
        self.assertTrue(all(not h.startswith("batch:") for hs in holders.values() for h in hs))

    def test_a_run_waits_at_a_boundary_for_its_models_prefill_preparation(self):
        sim = Sim(run_steps=8, **self.JOIN)
        sim.submit("a", "r", max_new=10)
        sim.prefill("a", "r", 11)
        plan, reservation = sim.prepare_decode("a", "r", 8)
        sim.submit("a", "q", max_new=6)
        out = sim.controller.request_output("a", "q")
        p_plan = BatchPlan(sim.group("a"), "p-q", "a", "P", (RequestRef("q", out["attempt"], 0),))
        p_reservation = sim.controller.prepare_batch(p_plan)["reservation_id"]
        # Without the switch the run would end here (merge); now it waits.
        for rank in range(sim.tp):
            reply = self.step(sim, plan.batch_id, rank, 1, {"r": 12})
            self.assertEqual((reply["continue"], reply["handoffs"]), ("SYNC", ()))
        self.assertEqual(self.resume(sim, plan.batch_id, 0, 1)["continue"], "SYNC")
        for rank in range(sim.tp):
            sim.report("a", "D", rank, "p-q", "READY", {"reservation_id": p_reservation})
        self.assertIs(self.resume(sim, plan.batch_id, 0, 1)["continue"], True)
        self.assertIs(self.resume(sim, plan.batch_id, 1, 1)["continue"], True)  # replayed

    def test_a_parked_run_runs_the_handoff_then_takes_the_request(self):
        sim = Sim(run_steps=8, **self.JOIN)
        sim.submit("a", "r", max_new=10)
        sim.prefill("a", "r", 11)
        plan, reservation = sim.prepare_decode("a", "r", 8)
        sim.submit("a", "q", max_new=6)
        p_plan, p_out, p_reservation = sim.prefill_ready("a", "q")
        for rank in range(sim.tp):
            self.assertEqual(self.step(sim, plan.batch_id, rank, 1, {"r": 12})["continue"], "PARK")
        sim.controller.start_batch(sim.group("a"), p_plan.batch_id)
        group = sim.controller.groups["a"]
        for rank in range(sim.tp):
            item = next(c for c in sim.commands("a", "P", rank, "PREPARE")
                        if c["batch_id"] == p_plan.batch_id)["requests"][0]
            view = item["reservation_view"]
            sim.controller.broker_stage(sim.group("a"), item["bootstrap_room"], rank, sim.pid("a", "P", rank), {
                "bootstrap_room": item["bootstrap_room"], "allocation_id": view["allocation_id"],
                "reserved_version": view["version"], "target_valid_tokens": 3, "rank": rank,
                "native_payload": {"x": 1}})
        for rank in range(sim.tp):
            sim.report("a", "P", rank, p_plan.batch_id, "GPU_DONE", {
                "reservation_id": p_reservation, "duration_ms": 1.0, "gpu_forward_ms": 1.0,
                "results": [{"request_id": "q", "attempt": 0, "token_id": 21, "valid_tokens": 3}]})
        # Credits are free again, but the handoff runs at this boundary first.
        reply = self.resume(sim, plan.batch_id, 0, 1)
        self.assertEqual((reply["continue"], reply["handoffs"]), ("PARK", (p_plan.batch_id,)))
        epoch = group.epoch
        reply = sim.controller.decode_resume(sim.group("a"), 0, sim.pid("a", "D", 0), plan.batch_id, 1, 0,
                                             None, epoch, (p_plan.batch_id,))
        self.assertEqual(reply["continue"], "PARK")
        for rank in range(sim.tp):
            item = sim.commands("a", "D", rank, "HANDOFF")[-1]["requests"][0]
            sim.controller.broker_ack(sim.group("a"), item["bootstrap_room"], rank, sim.pid("a", "D", rank),
                                      item["allocation_id"], item["version"], item["valid_tokens"],
                                      item["rank_block_table"])
        self.assertNotEqual(group.epoch, epoch)  # a waiting rank wakes to read the broker state
        for rank in range(sim.tp - 1):
            item = sim.commands("a", "D", rank, "HANDOFF")[-1]["requests"][0]
            sim.report("a", "D", rank, p_plan.batch_id, "HANDOFF_READY", {"requests": [item]})
        self.assertEqual(self.resume(sim, plan.batch_id, 0, 1)["continue"], "PARK")  # one rank still handing off
        item = sim.commands("a", "D", sim.tp - 1, "HANDOFF")[-1]["requests"][0]
        sim.report("a", "D", sim.tp - 1, p_plan.batch_id, "HANDOFF_READY", {"requests": [item]})
        reply = self.resume(sim, plan.batch_id, 0, 1)  # P ranks still quiesce: q may join already
        self.assertIs(reply["continue"], True)
        self.assertEqual([item["request_id"] for item in reply["join"]], ["q"])
        self.assertEqual(self.resume(sim, plan.batch_id, 1, 1)["join"][0]["request_id"], "q")
        for rank in range(sim.tp):
            sim.report("a", "P", rank, p_plan.batch_id, "QUIESCED",
                       {"reservation_id": p_reservation, "request_ids": ("q",)})
        for rank in range(sim.tp):
            self.assertIs(self.step(sim, plan.batch_id, rank, 2, {"r": 13, "q": 22})["continue"], True)
        sim.controller.cancel_request("a", "q")
        for rank in range(sim.tp):
            self.assertIs(self.step(sim, plan.batch_id, rank, 3, {"r": 14, "q": 23})["continue"], False)
        self.finish(sim, plan.batch_id, reservation, 3, {"r": 14, "q": 23})
        self.assertEqual(sim.controller.requests[("a", "q")].output, [21, 22])  # nothing after the cancel
        sim.store.assert_integrity()
        self.assertTrue(all(not any(refs) for refs in sim.controller.credits.refs.values()))

    def test_per_page_kv_grows_one_page_at_its_boundary(self):
        sim = Sim(run_steps=40, decode_kv_per_page=True, **self.JOIN)
        sim.submit("a", "r", max_new=40)
        sim.prefill("a", "r", 11)  # 3 prompt tokens committed
        plan, reservation = sim.prepare_decode("a", "r", 30)
        batch = self.batch(sim, plan.batch_id)
        self.assertEqual(batch.views["r"].capacity_tokens, 16)  # one page, not 3 + 30
        token = 12
        for step in range(1, 13):
            for rank in range(sim.tp):
                self.assertIs(self.step(sim, plan.batch_id, rank, step, {"r": token})["continue"], True)
            token += 1
        # Step 13 needs position 16 (17 tokens): the decision adds one page.
        for rank in range(sim.tp):
            reply = self.step(sim, plan.batch_id, rank, 13, {"r": token})
            (item,) = reply["extend"]
            self.assertEqual((item["request_id"], item["capacity_tokens"], item["first_page"]), ("r", 32, 1))
            self.assertEqual(item["pages"].pages, 1)  # only the new page
        token += 1
        for step in range(14, 29):
            for rank in range(sim.tp):
                self.assertIs(self.step(sim, plan.batch_id, rank, step, {"r": token})["continue"], True)
            token += 1
        # No page left: the run ends at this boundary instead.
        with mock.patch.object(sim.store, "extend_kv_batch", side_effect=IntegrationError("OUT_OF_CAPACITY", "full")):
            for rank in range(sim.tp):
                self.assertIs(self.step(sim, plan.batch_id, rank, 29, {"r": token})["continue"], False)
        self.finish(sim, plan.batch_id, reservation, 29, {"r": token})
        r = sim.controller.requests[("a", "r")]
        self.assertEqual((len(r.output), r.valid_tokens, r.state), (30, 32, "DECODING"))
        sim.store.assert_integrity()

    def test_a_join_that_does_not_fit_the_event_window_ends_the_run(self):
        sim = Sim(run_steps=8, **self.JOIN)
        sim.submit("a", "r", max_new=10)
        sim.submit("a", "q", max_new=6)
        sim.prefill("a", "r", 11)
        sim.prefill("a", "q", 21)
        plan, reservation = sim.prepare_decode("a", "r", 8)
        group = sim.controller.groups["a"]
        with mock.patch.object(group.outbox, "grow", side_effect=IntegrationError("EVENT_BACKPRESSURE", "full")):
            self.assertIs(self.step(sim, plan.batch_id, 0, 1, {"r": 12})["continue"], False)
        self.assertEqual(sim.controller.requests[("a", "q")].state, "DECODING")
        self.assertEqual([ref.request_id for ref in self.batch(sim, plan.batch_id).plan.requests], ["r"])

    def test_store_growth_joins_and_rolls_back_like_the_first_reservation(self):
        for outcome in ("SUCCESS", "FAILURE"):
            store = HeadwiseStore({0: 4096, 1: 4096})
            from muxserve_sglang.contracts import KVTarget, ModelLayout, RequestIdentity
            store.register_model(ModelLayout("a", 2, 1, (0, 1)))
            r, q = RequestIdentity("a", "r"), RequestIdentity("a", "q")
            q_view = store.reserve_kv("p", [KVTarget(q, 5)], "P").views[0]
            store.start_kv_batch(store.reserve_kv("p", [KVTarget(q, 5)], "P").reservation_id)
            store.finish_kv_batch(store._batch_ids["p"], "SUCCESS", {q_view.allocation_id: 5}, (0, 1))
            store.bind_kv(q_view.allocation_id, "D:q")
            store.release_kv(q_view.allocation_id, "P")
            reservation = store.reserve_kv("d", [KVTarget(r, 4)], "batch")
            r_id = reservation.views[0].allocation_id
            free = store.rank_usage()[0]["free_blocks"]
            expect("INVALID_STATE", lambda: store.extend_kv_batch(reservation.reservation_id,
                                                                  [KVTarget(q, 6, q_view.allocation_id)]))
            store.start_kv_batch(reservation.reservation_id)
            views = store.extend_kv_batch(reservation.reservation_id,
                                          [KVTarget(q, 6, q_view.allocation_id), KVTarget(r, 17, r_id)])
            self.assertEqual([v.capacity_tokens for v in views], [16, 32])
            self.assertEqual(store.rank_usage()[0]["free_blocks"], free - 2)  # one page of r (2 layers)
            expect("ALLOCATION_BUSY", lambda: store.reserve_kv("x", [KVTarget(q, 7, q_view.allocation_id)], "y"))
            store.assert_integrity()
            valid = {r_id: 17, q_view.allocation_id: 6} if outcome == "SUCCESS" else {}
            store.finish_kv_batch(reservation.reservation_id, outcome, valid, (0, 1))
            for aid in (r_id, q_view.allocation_id):
                store.release_kv(aid, "batch")  # as the controller's batch finish does
            allocations = store.snapshot()["allocations"]
            self.assertEqual(allocations[q_view.allocation_id]["holders"], ["D:q"])  # the batch holder left
            self.assertEqual(allocations[q_view.allocation_id]["valid_tokens"], 6 if outcome == "SUCCESS" else 5)
            self.assertNotIn(r_id, allocations)  # only the batch held r
            self.assertEqual(allocations[q_view.allocation_id]["active_batch"], None)
            store.assert_integrity()


class StepPipelineChecks(unittest.TestCase):
    """decode_step_pipeline: the leader runs on at boundaries the owner granted ahead."""

    FLAGS = dict(park=True, max_model_inflight=2, decode_run_merge=True, decode_step_join=True,
                 decode_leader_reports=True, decode_step_pipeline=True)

    def step(self, sim, batch_id, rank, step, tokens, finished=False, pregranted=False):
        batch = sim.controller.groups["a"].batches[batch_id]
        results = [{"request_id": ref.request_id, "attempt": ref.attempt, "token_id": tokens[ref.request_id],
                    "valid_tokens": batch.run_start_valid[ref.request_id] + step} for ref in batch.members(step)]
        return sim.controller.decode_step(sim.group("a"), rank, sim.pid("a", "D", rank), batch_id, step,
                                          results, finished, pregranted)

    def running(self, **flags):
        sim = Sim(run_steps=8, **{**self.FLAGS, **flags})
        sim.submit("a", "r", max_new=12)
        sim.prefill("a", "r", 11)
        plan, reservation = sim.prepare_decode("a", "r", 8)
        return sim, plan, reservation

    def test_granted_boundaries_run_on_and_publish_every_token_in_order(self):
        sim, plan, reservation = self.running()
        reply = self.step(sim, plan.batch_id, 0, 1, {"r": 12})
        self.assertEqual((reply["continue"], reply["ahead"]), (True, True))  # boundary 2 granted
        for step, token in ((2, 13), (3, 14)):
            reply = self.step(sim, plan.batch_id, 0, step, {"r": token}, pregranted=True)
            self.assertEqual((reply["continue"], reply["ahead"]), (True, True))
        # A follower ends at step 5 before the leader's report of step 4 arrived.
        for rank in range(1, sim.tp):
            self.assertIs(self.step(sim, plan.batch_id, rank, 5, {"r": 16}, finished=True)["continue"], False)
        self.step(sim, plan.batch_id, 0, 4, {"r": 15}, pregranted=True)
        self.assertIs(self.step(sim, plan.batch_id, 0, 5, {"r": 16}, finished=True)["continue"], False)
        batch = sim.controller.groups["a"].batches[plan.batch_id]
        self.assertEqual(batch.run_decisions, {1: True, 2: True, 3: True, 4: True, 5: False})
        self.assertEqual(sim.tokens("a")[-5:], [12, 13, 14, 15, 16])

    def test_anything_new_for_the_run_withholds_the_grant(self):
        cases = {
            "prefill of its model": lambda sim: sim.controller.prepare_batch(BatchPlan(
                sim.group("a"), "p-q", "a", "P", (RequestRef("q", 0, 0),))),
            "cancelled": lambda sim: sim.controller.cancel_request("a", "r"),
        }
        for name, change in cases.items():
            with self.subTest(name=name):
                sim, plan, _ = self.running()
                sim.submit("a", "q", max_new=6)
                self.assertTrue(self.step(sim, plan.batch_id, 0, 1, {"r": 12})["ahead"])
                change(sim)
                # Step 2 was granted before the change: it still runs on ...
                reply = self.step(sim, plan.batch_id, 0, 2, {"r": 13}, pregranted=True)
                self.assertIs(reply["continue"], True)
                self.assertIs(reply["ahead"], False)  # ... and boundary 3 is decided as before
        sim = Sim(run_steps=8, **self.FLAGS)
        sim.submit("a", "r", max_new=12)
        sim.submit("a", "q", max_new=6)
        sim.prefill("a", "r", 11)
        sim.prefill("a", "q", 21)  # decodable, may join
        plan, _ = sim.prepare_decode("a", "r", 8)
        reply = self.step(sim, plan.batch_id, 0, 1, {"r": 12})
        self.assertEqual([item["request_id"] for item in reply["join"]], ["q"])
        self.assertIs(reply["ahead"], True)  # it joined: nothing else is new

    def test_a_page_boundary_is_never_granted_ahead(self):
        sim = Sim(run_steps=40, decode_kv_per_page=True, **self.FLAGS)
        sim.submit("a", "r", max_new=40)
        sim.prefill("a", "r", 11)  # 3 tokens, page of 16
        plan, _ = sim.prepare_decode("a", "r", 30)
        token, aheads = 12, []
        for step in range(1, 14):
            reply = self.step(sim, plan.batch_id, 0, step, {"r": token}, pregranted=step > 1 and aheads[-1])
            aheads.append(reply["ahead"])
            token += 1
        # Boundary 13 (step 14 writes position 16) needs a page: decided, not granted.
        self.assertEqual(aheads[:11], [True] * 11)
        self.assertIs(aheads[11], False)
        self.assertEqual(reply["extend"][0]["capacity_tokens"], 32)

    def test_another_models_prefill_in_preparation_withholds_the_grant(self):
        sim = Sim(("a", "b"), run_steps=8, **self.FLAGS)
        sim.submit("a", "r", max_new=12)
        sim.prefill("a", "r", 11)
        plan, _ = sim.prepare_decode("a", "r", 8)
        self.assertIs(self.step(sim, plan.batch_id, 0, 1, {"r": 12})["ahead"], True)
        sim.submit("b", "x", max_new=6)
        out = sim.controller.request_output("b", "x")
        sim.controller.prepare_batch(BatchPlan(sim.group("b"), "p-x", "b", "P", (RequestRef("x", out["attempt"], 0),)))
        # Step 2 was granted before: it runs on; boundary 3 waits to be decided, so the
        # run can park there as soon as the prefill is ready.
        reply = self.step(sim, plan.batch_id, 0, 2, {"r": 13}, pregranted=True)
        self.assertEqual((reply["continue"], reply["ahead"]), (True, False))

    def test_ungranted_or_follower_pregranted_reports_are_refused(self):
        sim, plan, _ = self.running()
        expect("REPORT_GAP", lambda: self.step(sim, plan.batch_id, 0, 1, {"r": 12}, pregranted=True))
        self.assertNotIn(1, sim.controller.groups["a"].batches[plan.batch_id].run_reports)
        self.step(sim, plan.batch_id, 0, 1, {"r": 12})
        expect("INVALID_ARGUMENT", lambda: self.step(sim, plan.batch_id, 1, 2, {"r": 13}, pregranted=True))
        sim2, plan2, _ = self.running(decode_step_pipeline=False)
        self.assertNotIn("ahead", self.step(sim2, plan2.batch_id, 0, 1, {"r": 12}))
        expect("INVALID_ARGUMENT", lambda: self.step(sim2, plan2.batch_id, 0, 2, {"r": 13}, pregranted=True))


class StepLeaveChecks(unittest.TestCase):
    """decode_step_leave and decode_kv_grow_pages: members leave a running run;
    pages come in chunks, to every member about to need one."""

    FLAGS = dict(park=True, max_model_inflight=2, decode_run_merge=True, decode_step_join=True,
                 decode_leader_reports=True, decode_step_leave=True)
    step = StepJoinChecks.step
    finish = StepJoinChecks.finish

    @staticmethod
    def batch(sim, batch_id):
        return sim.controller.groups["a"].batches[batch_id]

    def joined_pair(self, q_new=3, q_input=(1, 2, 3), r_new=10, **flags):
        """r runs alone for 8 steps; q (already decodable) joins at its first boundary."""
        sim = Sim(run_steps=8, **{**self.FLAGS, **flags})
        sim.submit("a", "r", max_new=r_new)
        sim.submit("a", "q", max_new=q_new, input_ids=q_input)
        sim.prefill("a", "r", 11)
        sim.prefill("a", "q", 21)
        plan, reservation = sim.prepare_decode("a", "r", 8)
        for rank in range(sim.tp):
            self.assertEqual([i["request_id"] for i in self.step(sim, plan.batch_id, rank, 1, {"r": 12})["join"]],
                             ["q"])
        return sim, plan, reservation

    def test_a_finished_member_leaves_and_the_run_goes_on(self):
        sim, plan, reservation = self.joined_pair()
        for rank in range(sim.tp):
            self.assertIs(self.step(sim, plan.batch_id, rank, 2, {"r": 13, "q": 22})["continue"], True)
        q = sim.controller.requests[("a", "q")]
        for rank in range(sim.tp):  # q reaches its length limit; r goes on
            reply = self.step(sim, plan.batch_id, rank, 3, {"r": 14, "q": 23})
            self.assertEqual((reply["continue"], reply["leave"]), (True, ("q",)))
        batch = self.batch(sim, plan.batch_id)
        self.assertEqual([ref.request_id for ref in batch.members(3)], ["r", "q"])
        self.assertEqual([ref.request_id for ref in batch.members(4)], ["r"])
        allocation = sim.store.snapshot()["allocations"][q.allocation_id]
        self.assertEqual((allocation["active_batch"], allocation["valid_tokens"]), (None, 5))
        self.assertEqual(q.state, "CLEANUP_PENDING")
        self.assertTrue(sim.commands("a", "D", 0, "CLEANUP"))
        sim.cleanup("a", "q")  # released while r's run keeps going
        self.assertEqual((q.state, q.output), ("FINISHED", [21, 22, 23]))
        self.assertNotIn(q.allocation_id, sim.store.snapshot()["allocations"])
        sim.store.assert_integrity()
        token = 15
        for step in range(4, 9):
            for rank in range(sim.tp):
                self.assertIs(self.step(sim, plan.batch_id, rank, step, {"r": token})["continue"], step < 8)
            token += 1
        self.finish(sim, plan.batch_id, reservation, 8, {"r": token - 1})
        r = sim.controller.requests[("a", "r")]
        self.assertEqual((r.output, r.valid_tokens, r.state), (list(range(11, 20)), 11, "DECODING"))
        sim.store.assert_integrity()

    def test_a_run_is_sized_by_its_longest_member_and_the_shorter_one_leaves(self):
        sim = Sim(run_steps=8, **self.FLAGS)
        sim.submit("a", "r", max_new=10)
        sim.submit("a", "q", max_new=3)
        sim.prefill("a", "r", 11)
        sim.prefill("a", "q", 21)
        driver = ControllerDriver(sim.controller, lambda event: None)
        outputs = [sim.controller.request_output("a", name) for name in ("r", "q")]
        self.assertEqual(driver._run_steps(SimpleNamespace(model_id="a", stage="D", purpose="GENERATE"), outputs), 8)
        plan = BatchPlan(sim.group("a"), "d-both", "a", "D", tuple(
            RequestRef(o["request_id"], o["attempt"], o["output_count"]) for o in outputs), steps=8)
        reservation = sim.controller.prepare_batch(plan)["reservation_id"]
        batch = self.batch(sim, "d-both")
        self.assertEqual(batch.run_event_budget, {"r": 9, "q": 3})  # q can publish 2 more tokens (+1)
        self.assertEqual(batch.views["q"].capacity_tokens, 16)  # its own budget, not 8 steps
        sim._ready_all("a", "d-both", reservation)
        sim.controller.start_batch(sim.group("a"), "d-both")
        for step, tokens in ((1, {"r": 12, "q": 22}), (2, {"r": 13, "q": 23})):
            replies = [self.step(sim, "d-both", rank, step, tokens) for rank in range(sim.tp)]
        self.assertEqual(replies[0]["leave"], ("q",))  # q reached its 3 tokens at step 2
        sim.cleanup("a", "q")
        self.assertEqual(sim.controller.requests[("a", "q")].state, "FINISHED")
        sim.store.assert_integrity()

    def test_a_cancelled_member_leaves_without_its_later_tokens(self):
        sim, plan, reservation = self.joined_pair(q_new=8)
        sim.controller.cancel_request("a", "q")
        for rank in range(sim.tp):
            reply = self.step(sim, plan.batch_id, rank, 2, {"r": 13, "q": 22})
            self.assertEqual(reply["leave"], ("q",))
        sim.cleanup("a", "q")
        q = sim.controller.requests[("a", "q")]
        self.assertEqual((q.state, q.output), ("CANCELLED", [21]))
        sim.store.assert_integrity()

    def test_a_run_whose_every_member_ended_ends_as_before(self):
        sim = Sim(run_steps=8, **self.FLAGS)
        sim.submit("a", "r", max_new=10)
        sim.prefill("a", "r", 11)
        plan, reservation = sim.prepare_decode("a", "r", 8)
        sim.controller.cancel_request("a", "r")
        for rank in range(sim.tp):
            self.assertIs(self.step(sim, plan.batch_id, rank, 1, {"r": 12})["continue"], False)
        self.finish(sim, plan.batch_id, reservation, 1, {"r": 12})
        sim.store.assert_integrity()

    def test_pages_come_in_chunks_to_every_member_about_to_need_one(self):
        sim, plan, _ = self.joined_pair(q_new=40, q_input=tuple(range(1, 15)), r_new=40,
                                        decode_kv_per_page=True, decode_kv_grow_pages=4)
        batch = self.batch(sim, plan.batch_id)
        self.assertEqual((batch.views["r"].capacity_tokens, batch.views["q"].capacity_tokens), (16, 16))
        tokens = {"r": 13, "q": 22}
        for step in (2,):
            for rank in range(sim.tp):
                self.assertIs(self.step(sim, plan.batch_id, rank, step, tokens)["continue"], True)
        # Boundary 3: q (14 prompt tokens) needs position 16; r (3 + 3) is within a page
        # of its end, so it is topped up too: 4 pages each, as far as each can use them
        # (q: 53 tokens -> 64; r: 42 -> 48).
        for rank in range(sim.tp):
            reply = self.step(sim, plan.batch_id, rank, 3, tokens)
            grown = {item["request_id"]: (item["first_page"], item["capacity_tokens"], item["pages"].pages)
                     for item in reply["extend"]}
            self.assertEqual(grown, {"q": (1, 64, 3), "r": (1, 48, 2)})
        sim.store.assert_integrity()

    def test_chunked_growth_falls_back_to_the_pages_needed_now(self):
        sim = Sim(run_steps=40, decode_kv_per_page=True, decode_kv_grow_pages=4, **self.FLAGS)
        sim.submit("a", "r", max_new=40)
        sim.prefill("a", "r", 11)
        plan, _ = sim.prepare_decode("a", "r", 30)
        token = 12
        for step in range(1, 13):
            self.assertIs(self.step(sim, plan.batch_id, 0, step, {"r": token})["continue"], True)
            token += 1
        real = sim.controller._grow_reservation
        calls = []

        def tight(group, batch, targets):
            calls.append([t.target_tokens for t in targets])
            if len(calls) == 1:
                raise IntegrationError("OUT_OF_CAPACITY", "only one page left")
            return real(group, batch, targets)

        with mock.patch.object(sim.controller, "_grow_reservation", side_effect=tight):
            reply = self.step(sim, plan.batch_id, 0, 13, {"r": token})
        self.assertEqual(calls, [[42], [17]])  # 4 pages (as far as r can use), then the one needed
        self.assertEqual(reply["extend"][0]["capacity_tokens"], 32)


class OwnerHopChecks(unittest.TestCase):
    """decode_wait_commands, prefill_stage_in_gpu_done and driver_min_interval_ms."""

    WAIT = dict(park=True, max_model_inflight=2, decode_run_merge=True, decode_step_join=True,
                decode_wait_commands=True, handoff_ack_in_poll=True, broker_status_in_poll=True)
    step = StepJoinChecks.step
    batch = staticmethod(StepJoinChecks.batch)

    def parked_with_handoff(self, **flags):
        sim = Sim(run_steps=8, **{**self.WAIT, **flags})
        sim.submit("a", "r", max_new=10)
        sim.prefill("a", "r", 11)
        plan, _ = sim.prepare_decode("a", "r", 8)
        sim.submit("a", "q", max_new=6)
        p_plan, _, p_reservation = sim.prefill_ready("a", "q")
        for rank in range(sim.tp):
            self.assertEqual(self.step(sim, plan.batch_id, rank, 1, {"r": 12})["continue"], "PARK")
        sim.controller.start_batch(sim.group("a"), p_plan.batch_id)
        return sim, plan, p_plan, p_reservation

    def stage(self, sim, p_plan, rank):
        item = next(c for c in sim.commands("a", "P", rank, "PREPARE")
                    if c["batch_id"] == p_plan.batch_id)["requests"][0]
        view = item["reservation_view"]
        return {"bootstrap_room": item["bootstrap_room"], "allocation_id": view["allocation_id"],
                "reserved_version": view["version"], "target_valid_tokens": 3, "rank": rank,
                "native_payload": {"x": 1}}

    def prefill_done(self, sim, p_plan, p_reservation, staged=False):
        for rank in range(sim.tp):
            payload = {"reservation_id": p_reservation, "duration_ms": 1.0, "gpu_forward_ms": 1.0,
                       "results": [{"request_id": "q", "attempt": 0, "token_id": 21, "valid_tokens": 3}]}
            if staged:
                payload["staged"] = (self.stage(sim, p_plan, rank),)
            else:
                metadata = self.stage(sim, p_plan, rank)
                sim.controller.broker_stage(sim.group("a"), metadata["bootstrap_room"], rank,
                                            sim.pid("a", "P", rank), metadata)
            sim.report("a", "P", rank, p_plan.batch_id, "GPU_DONE", payload)

    def resume(self, sim, plan, rank, **kwargs):
        args = dict(after_seq=sim.acked[("a", "D", rank)], epoch=None, handoffs_seen=(), ack_seq=None,
                    broker_acks=())
        args.update(kwargs)
        return sim.controller.decode_resume(sim.group("a"), rank, sim.pid("a", "D", rank), plan.batch_id, 1, 0,
                                            args["after_seq"], args["epoch"], args["handoffs_seen"],
                                            args["ack_seq"], args["broker_acks"])

    def test_a_waiting_rank_gets_the_handoff_and_acks_it_through_decode_resume(self):
        sim, plan, p_plan, p_reservation = self.parked_with_handoff()
        self.prefill_done(sim, p_plan, p_reservation)
        group = sim.controller.groups["a"]
        handoffs, acks = {}, {}
        for rank in range(sim.tp):
            reply = self.resume(sim, plan, rank)
            self.assertEqual((reply["continue"], reply["handoffs"]), ("PARK", (p_plan.batch_id,)))
            command = next(c for c in reply["commands"] if c["kind"] == "HANDOFF")
            item = command["requests"][0]
            self.assertEqual(dict(reply["broker_status"])[item["bootstrap_room"]], "PENDING")
            handoffs[rank] = item
            acks[rank] = (item["bootstrap_room"], item["allocation_id"], item["version"],
                          item["valid_tokens"], item["rank_block_table"])
        for rank in range(sim.tp):  # each rank's next call carries its ACK
            reply = self.resume(sim, plan, rank, after_seq=sim.acked[("a", "D", rank)] + 0,
                                epoch=group.epoch, handoffs_seen=(p_plan.batch_id,), broker_acks=(acks[rank],))
        room = handoffs[0]["bootstrap_room"]
        self.assertEqual(dict(reply["broker_status"])[room], "SUCCESS")  # the last ACK completes it
        self.assertEqual(group.broker[room]["acked"], set(range(sim.tp)))
        for rank in range(sim.tp):
            sim.report("a", "D", rank, p_plan.batch_id, "HANDOFF_READY", {"requests": [handoffs[rank]]})
        reply = self.resume(sim, plan, 0, handoffs_seen=(p_plan.batch_id,))
        self.assertIs(reply["continue"], True)
        self.assertEqual([item["request_id"] for item in reply["join"]], ["q"])
        expect("COMMAND_GAP", lambda: self.resume(sim, plan, 1, after_seq=10 ** 6))
        sim2, plan2, _, _ = self.parked_with_handoff(decode_wait_commands=False)
        expect("INVALID_ARGUMENT", lambda: self.resume(sim2, plan2, 0, ack_seq=0))

    def test_a_prefill_stages_its_metadata_inside_gpu_done(self):
        sim, plan, p_plan, p_reservation = self.parked_with_handoff(prefill_stage_in_gpu_done=True)
        bad = {"reservation_id": p_reservation, "duration_ms": 1.0, "gpu_forward_ms": 1.0, "staged": (),
               "results": [{"request_id": "q", "attempt": 0, "token_id": 21, "valid_tokens": 3}]}
        expect("INVALID_REPORT", lambda: sim.controller.rank_report(
            sim.group("a"), "P", 0, sim.pid("a", "P", 0), str(sim.seq[("a", "P", 0)] + 1), p_plan.batch_id,
            "GPU_DONE", bad))
        self.prefill_done(sim, p_plan, p_reservation, staged=True)
        group = sim.controller.groups["a"]
        entry = next(e for e in group.broker.values() if e["batch_id"] == p_plan.batch_id)
        self.assertEqual((entry["state"], set(entry["metadata"])), ("PUBLISHED", set(range(sim.tp))))
        self.assertEqual(sim.controller.groups["a"].batches[p_plan.batch_id].status, "HANDOFF")
        sim2, plan2, p_plan2, p_res2 = self.parked_with_handoff()
        with self.assertRaises(IntegrationError):  # without the switch a staged GPU_DONE is refused
            self.prefill_done(sim2, p_plan2, p_res2, staged=True)

    def test_the_indexed_snapshot_equals_the_scanning_one(self):
        sim = Sim(("a", "b"), run_steps=8, park=True, max_model_inflight=4, decode_run_merge=True,
                  decode_step_join=True, driver_min_interval_ms=5)
        for index, name in enumerate(("r", "q", "s")):
            sim.submit("a", name, max_new=10)
            sim.prefill("a", name, 11 + index)
        sim.submit("a", "w", max_new=6)
        sim.submit("b", "x", max_new=6)
        sim.prepare_decode("a", "r", 8)
        controller = sim.controller
        with controller.lock:
            self.assertEqual(controller._snapshot_policy(), controller._snapshot_policy_inner())
            self.assertIsNone(controller._active_lookup)

    def test_driver_passes_are_spaced(self):
        sim = Sim(driver_min_interval_ms=20)
        driver = ControllerDriver(sim.controller, lambda event: None)
        passes = []

        def step():
            passes.append(time.monotonic())
            sim.controller.wake.set()  # constant activity

        driver.step = step
        worker = threading.Thread(target=driver.run, daemon=True)
        worker.start()
        time.sleep(0.25)
        driver.stop_event.set()
        worker.join(2)
        self.assertLessEqual(len(passes), 15)
        self.assertGreaterEqual(min(b - a for a, b in zip(passes, passes[1:])), 0.018)


class ParkOverlapChecks(unittest.TestCase):
    """prefill_start_unparked and handoff_local_success."""

    def running(self, models=("a", "b"), **flags):
        sim = Sim(models, run_steps=8, park=True, prefill_start_unparked=True, **flags)
        sim.submit("a", "r", max_new=10)
        sim.prefill("a", "r", 11)
        plan, reservation = sim.prepare_decode("a", "r", 8)
        self.assertTrue(all(sim.step("a", plan, rank, 1, 12)["continue"] for rank in range(sim.tp)))
        sim.submit("b", "x", max_new=4)
        return sim, plan, sim.prefill_ready("b", "x")

    def test_a_ready_prefill_starts_while_runs_still_hold_credits(self):
        sim, plan, (p_plan, p_out, p_reservation) = self.running()
        credits = sim.controller.credits
        sim.controller.start_batch(sim.group("b"), p_plan.batch_id)
        p_batch = sim.controller.groups["b"].batches[p_plan.batch_id]
        self.assertEqual((p_batch.status, p_batch.early_start), ("RUNNING", True))
        self.assertNotIn(p_plan.batch_id, credits.leases)
        self.assertIn(plan.batch_id, credits.leases)
        # The run still parks at its next boundary, exactly as for a waiting prefill.
        self.assertTrue(sim.controller._prefill_awaiting_credits())
        self.assertEqual(sim.step("a", plan, 0, 2, 13)["continue"], "PARK")
        self.assertNotIn(p_plan.batch_id, credits.leases)
        self.assertEqual(sim.step("a", plan, 1, 2, 13)["continue"], "PARK")
        # Parked: its credits go to the running prefill, and the run stays parked.
        self.assertNotIn(plan.batch_id, credits.leases)
        self.assertIn(p_plan.batch_id, credits.leases)
        self.assertEqual(sim.resume("a", plan, 0, 2), "PARK")
        sim.prefill_finish("b", "x", 21, p_plan, p_out, p_reservation)
        self.assertEqual(sim.resume("a", plan, 0, 2), True)
        self.assertEqual(sim.resume("a", plan, 1, 2), True)
        self.assertIn(plan.batch_id, credits.leases)
        self.assertEqual(sim.controller.requests[("b", "x")].output, [21])

    def test_a_prefill_that_ends_before_the_runs_park_never_takes_credits(self):
        sim, plan, (p_plan, p_out, p_reservation) = self.running()
        sim.controller.start_batch(sim.group("b"), p_plan.batch_id)
        self.assertEqual(sim.step("a", plan, 0, 2, 13)["continue"], "PARK")
        sim.prefill_finish("b", "x", 21, p_plan, p_out, p_reservation)
        self.assertNotIn(p_plan.batch_id, sim.controller.credits.leases)
        self.assertFalse(sim.controller._prefill_pending())
        self.assertEqual(sim.step("a", plan, 1, 2, 13)["continue"], "PARK")  # the decided step
        self.assertEqual(sim.resume("a", plan, 0, 2), True)
        self.assertIn(plan.batch_id, sim.controller.credits.leases)

    def test_only_one_prefill_starts_early(self):
        sim, plan, (p_plan, _, _) = self.running(models=("a", "b", "c"))
        sim.controller.start_batch(sim.group("b"), p_plan.batch_id)
        sim.submit("c", "y", max_new=4)
        c_plan, _, _ = sim.prefill_ready("c", "y")
        expect("NOT_READY", lambda: sim.controller.start_batch(sim.group("c"), c_plan.batch_id))
        self.assertEqual(sim.controller.groups["c"].batches[c_plan.batch_id].status, "PREPARING")

    def test_without_the_switch_the_prefill_waits_for_the_park(self):
        sim = Sim(("a", "b"), run_steps=8, park=True)
        sim.submit("a", "r", max_new=10)
        sim.prefill("a", "r", 11)
        plan, _ = sim.prepare_decode("a", "r", 8)
        sim.submit("b", "x", max_new=4)
        p_plan, _, _ = sim.prefill_ready("b", "x")
        expect("NOT_READY", lambda: sim.controller.start_batch(sim.group("b"), p_plan.batch_id))

    def test_a_rank_reports_the_local_handoff_with_its_own_acks(self):
        hops = OwnerHopChecks()
        sim, plan, p_plan, p_reservation = hops.parked_with_handoff(handoff_local_success=True)
        hops.prefill_done(sim, p_plan, p_reservation)
        group = sim.controller.groups["a"]
        items, acks = {}, {}
        for rank in range(sim.tp):
            reply = hops.resume(sim, plan, rank)
            item = next(c for c in reply["commands"] if c["kind"] == "HANDOFF")["requests"][0]
            items[rank] = item
            acks[rank] = (item["bootstrap_room"], item["allocation_id"], item["version"],
                          item["valid_tokens"], item["rank_block_table"])
        room = items[0]["bootstrap_room"]
        # A report needs this rank's ACK, not every rank's.
        expect("HANDOFF_NOT_READY", lambda: sim.controller.rank_report(
            sim.group("a"), "D", 0, sim.pid("a", "D", 0), str(sim.seq[("a", "D", 0)] + 1), p_plan.batch_id,
            "HANDOFF_READY", {"requests": [items[0]]}))
        sim.report("a", "D", 0, p_plan.batch_id, "HANDOFF_READY", {"requests": [items[0]], "acks": [acks[0]]})
        self.assertEqual(group.broker[room]["acked"], {0})
        self.assertEqual(group.batches[p_plan.batch_id].handoff, {0})
        self.assertEqual(hops.resume(sim, plan, 0, handoffs_seen=(p_plan.batch_id,))["continue"], "PARK")
        sim.report("a", "D", 1, p_plan.batch_id, "HANDOFF_READY", {"requests": [items[1]], "acks": [acks[1]]})
        self.assertEqual(group.broker[room]["acked"], set(range(sim.tp)))
        reply = hops.resume(sim, plan, 0, handoffs_seen=(p_plan.batch_id,))
        self.assertIs(reply["continue"], True)
        self.assertEqual([item["request_id"] for item in reply["join"]], ["q"])

    def test_without_the_switch_acks_in_a_report_are_refused(self):
        hops = OwnerHopChecks()
        sim, plan, p_plan, p_reservation = hops.parked_with_handoff()
        hops.prefill_done(sim, p_plan, p_reservation)
        item = next(c for c in hops.resume(sim, plan, 0)["commands"] if c["kind"] == "HANDOFF")["requests"][0]
        ack = (item["bootstrap_room"], item["allocation_id"], item["version"], item["valid_tokens"],
               item["rank_block_table"])
        expect("INVALID_REPORT", lambda: sim.controller.rank_report(
            sim.group("a"), "D", 0, sim.pid("a", "D", 0), str(sim.seq[("a", "D", 0)] + 1), p_plan.batch_id,
            "HANDOFF_READY", {"requests": [item], "acks": [ack]}))


class OverlapChecks(unittest.TestCase):
    """prefill_overlap_decode: decode runs keep running through prefills."""

    def running(self, models=("a", "b"), **flags):
        sim = Sim(models, run_steps=8, park=True, prefill_overlap_decode=True, **flags)
        sim.submit("a", "r", max_new=10)
        sim.prefill("a", "r", 11)
        plan, reservation = sim.prepare_decode("a", "r", 8)
        self.assertTrue(all(sim.step("a", plan, rank, 1, 12)["continue"] for rank in range(sim.tp)))
        sim.submit("b", "x", max_new=4)
        return sim, plan, reservation, sim.prefill_ready("b", "x")

    def test_a_prefill_starts_and_the_run_goes_on(self):
        sim, plan, _, (p_plan, p_out, p_reservation) = self.running()
        self.assertFalse(sim.controller._prefill_awaiting_credits())
        self.assertIs(sim.step("a", plan, 0, 2, 13)["continue"], True)  # no park for a READY prefill
        sim.controller.start_batch(sim.group("b"), p_plan.batch_id)
        p_batch = sim.controller.groups["b"].batches[p_plan.batch_id]
        self.assertEqual(p_batch.status, "RUNNING")
        self.assertNotIn(p_plan.batch_id, sim.controller.credits.leases)
        self.assertIn(plan.batch_id, sim.controller.credits.leases)
        self.assertIs(sim.step("a", plan, 1, 2, 13)["continue"], True)
        for rank in range(sim.tp):  # and through the prefill's forward
            self.assertIs(sim.step("a", plan, rank, 3, 14)["continue"], True)
        self.assertIsNone(sim.controller.groups["a"].batches[plan.batch_id].parked_at)
        sim.prefill_finish("b", "x", 21, p_plan, p_out, p_reservation)
        self.assertEqual(sim.controller.requests[("b", "x")].output, [21])

    def test_prefills_still_run_one_at_a_time(self):
        sim, plan, _, (p_plan, _, _) = self.running(models=("a", "b", "c"))
        sim.controller.start_batch(sim.group("b"), p_plan.batch_id)
        sim.submit("c", "y", max_new=4)
        c_plan, _, _ = sim.prefill_ready("c", "y")
        expect("NOT_READY", lambda: sim.controller.start_batch(sim.group("c"), c_plan.batch_id))

    def test_new_decode_batches_are_not_held_back(self):
        sim = Sim(("a", "b", "c"), run_steps=8, park=True, prefill_overlap_decode=True)
        sim.submit("c", "y", max_new=10)
        sim.prefill("c", "y", 31)
        sim.submit("b", "x", max_new=4)
        sim.prefill_ready("b", "x")
        plan, _ = sim.prepare_decode("c", "y", 8)  # without the switch: OUT_OF_CREDITS
        self.assertIn(plan.batch_id, sim.controller.credits.leases)


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
