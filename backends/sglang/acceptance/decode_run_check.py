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

    def __init__(self, models=("a",), *, tp=2, run_steps=4, blocks=4096):
        deployments = tuple(ModelDeployment(m, "unused", "0" * 40, "unused", "0" * 40,
                                            2, tp, 128 * tp, 40, 18100 + 10 * i, 18101 + 10 * i,
                                            18102 + 10 * i)
                            for i, m in enumerate(models))
        self.tp = tp
        self.clock = Clock()
        self.store = HeadwiseStore({rank: blocks for rank in range(tp)})
        self.config = DeploymentConfig(deployments, tp_size=tp, kv_bytes_per_rank=blocks * 8192,
                                       decode_run_steps=run_steps)
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
        self.controller.start_batch(self.group(model), plan.batch_id)
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
