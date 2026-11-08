"""CPU checks for the opt-in control-plane fast paths.

inline_start, decode_skip_prefill_ranks and prefill_early_token change when
the controller acts, never what it accepts. Simulated ranks drive the real
controller, store, policy and driver (decode_run_check.Sim); no Torch,
SGLang, model or GPU is involved.
"""
from __future__ import annotations

from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from decode_run_check import Sim as _Sim, expect  # noqa: E402
from muxserve_sglang.contracts import BatchPlan, IntegrationError, RequestRef  # noqa: E402
from muxserve_sglang.driver import ControllerDriver  # noqa: E402


def Sim(*args, **kwargs):
    """decode_run_check.Sim that also logs every command it issues (ACKed
    commands leave the controller, so the log is the only full history)."""
    sim = _Sim(*args, **kwargs)
    sim.log = []
    issue = sim.controller._command

    def logged(group, key, kind, payload):
        issue(group, key, kind, payload)
        sim.log.append((group.model_id, key, kind, payload.get("batch_id")))
    sim.controller._command = logged
    return sim


def kinds(sim, model, stage, rank, batch_id):
    return [kind for m, key, kind, bid in sim.log
            if m == model and key == (stage, rank) and bid == batch_id]


def ready(sim, model, batch_id, reservation, stages=("P", "D")):
    for stage in stages:
        for rank in range(sim.tp):
            sim.report(model, stage, rank, batch_id, "READY", {"reservation_id": reservation})


def quiesce(sim, model, plan, reservation, stages=("P", "D")):
    rid = plan.requests[0].request_id
    for stage in stages:
        for rank in range(sim.tp):
            sim.report(model, stage, rank, plan.batch_id, "QUIESCED",
                       {"reservation_id": reservation, "request_ids": (rid,)})


def prepare_decode(sim, model, rid, steps):
    out = sim.controller.request_output(model, rid)
    plan = BatchPlan(sim.group(model), f"d-{rid}-{out['output_count']}", model, "D",
                     (RequestRef(rid, out["attempt"], out["output_count"]),), steps=steps)
    return plan, sim.controller.prepare_batch(plan)["reservation_id"]


class SkipPrefillRanks(unittest.TestCase):
    def test_decode_batches_involve_only_d_ranks(self):
        sim = Sim(run_steps=4, decode_skip_prefill_ranks=True)
        sim.submit("a", "r", max_new=5)
        sim.prefill("a", "r", 11)  # prefill batches still use every rank
        plan, reservation = prepare_decode(sim, "a", "r", 4)
        batch = sim.controller.groups["a"].batches[plan.batch_id]
        self.assertEqual(batch.participants, {("D", 0), ("D", 1)})
        for rank in range(sim.tp):
            self.assertEqual(kinds(sim, "a", "P", rank, plan.batch_id), [])
            self.assertEqual(kinds(sim, "a", "D", rank, plan.batch_id), ["PREPARE"])
        # A P rank cannot vote for a batch it never received.
        expect("STALE_REPORT", lambda: sim.report("a", "P", 0, plan.batch_id, "READY",
                                                  {"reservation_id": reservation}))
        expect("NOT_READY", lambda: sim.controller.start_batch(sim.group("a"), plan.batch_id))
        ready(sim, "a", plan.batch_id, reservation, stages=("D",))
        sim.controller.start_batch(sim.group("a"), plan.batch_id)
        for step, token in enumerate((12, 13, 14, 15), start=1):
            self.assertEqual({sim.step("a", plan, rank, step, token)["continue"]
                              for rank in range(sim.tp)}, {step < 4})
        sim.finish_run("a", plan, reservation, 4, 15)
        for rank in range(sim.tp):
            self.assertEqual(kinds(sim, "a", "P", rank, plan.batch_id), [])
            self.assertEqual(kinds(sim, "a", "D", rank, plan.batch_id), ["PREPARE", "START", "QUIESCE"])
        quiesce(sim, "a", plan, reservation, stages=("D",))
        self.assertEqual(batch.status, "DONE")
        request = sim.controller.requests[("a", "r")]
        self.assertEqual((request.output, request.state), ([11, 12, 13, 14, 15], "CLEANUP_PENDING"))
        sim.cleanup("a", "r")  # request cleanup still reaches every P and D rank
        output = sim.controller.request_output("a", "r")
        self.assertEqual((output["state"], output["output_token_ids"]), ("FINISHED", (11, 12, 13, 14, 15)))
        self.assertTrue(all(not any(refs) for refs in sim.controller.credits.refs.values()))

    def test_revoked_decode_preparation_quiesces_only_d_ranks(self):
        sim = Sim(run_steps=4, decode_skip_prefill_ranks=True)
        sim.submit("a", "r", max_new=5)
        sim.prefill("a", "r", 11)
        free = sim.store.rank_usage()[0]["free_blocks"]
        plan, reservation = prepare_decode(sim, "a", "r", 4)
        sim.controller.abort_batch(sim.group("a"), plan.batch_id, "test")
        for rank in range(sim.tp):
            self.assertEqual(kinds(sim, "a", "P", rank, plan.batch_id), [])
            self.assertEqual(kinds(sim, "a", "D", rank, plan.batch_id), ["PREPARE", "REVOKE"])
        quiesce(sim, "a", plan, reservation, stages=("D",))
        batch = sim.controller.groups["a"].batches[plan.batch_id]
        self.assertEqual(batch.status, "ABORTED")
        self.assertEqual(sim.controller.requests[("a", "r")].state, "DECODING")
        self.assertEqual(sim.store.rank_usage()[0]["free_blocks"], free)

    def test_driver_starts_a_d_only_batch(self):
        sim = Sim(run_steps=4, decode_skip_prefill_ranks=True)
        sim.submit("a", "r", max_new=5)
        sim.prefill("a", "r", 11)
        driver = ControllerDriver(sim.controller, lambda event: None)
        driver.step()
        batch = next(b for b in sim.controller.groups["a"].batches.values()
                     if b.plan.stage == "D" and b.status == "PREPARING")
        ready(sim, "a", batch.plan.batch_id, batch.reservation.reservation_id, stages=("D",))
        driver.step()
        self.assertEqual(batch.status, "RUNNING")

    def test_default_keeps_every_rank(self):
        sim = Sim(run_steps=4)
        sim.submit("a", "r", max_new=5)
        sim.prefill("a", "r", 11)
        plan, _ = prepare_decode(sim, "a", "r", 4)
        self.assertEqual(sim.controller.groups["a"].batches[plan.batch_id].participants,
                         {(s, r) for s in ("P", "D") for r in range(sim.tp)})
        self.assertEqual(kinds(sim, "a", "P", 0, plan.batch_id), ["PREPARE"])


class InlineStart(unittest.TestCase):
    def test_last_ready_starts_the_batch_only_while_a_driver_schedules(self):
        sim = Sim(run_steps=4, inline_start=True)
        sim.submit("a", "r", max_new=5)
        # No driver yet: inline START is off, the batch waits for start_batch.
        plan, out, reservation = sim.prefill_ready("a", "r")
        self.assertEqual(sim.controller.groups["a"].batches[plan.batch_id].status, "PREPARING")
        sim.controller.start_batch(sim.group("a"), plan.batch_id)
        sim.prefill_finish("a", "r", 11, plan, out, reservation)
        driver = ControllerDriver(sim.controller, lambda event: None)
        d_plan, d_reservation = prepare_decode(sim, "a", "r", 4)
        batch = sim.controller.groups["a"].batches[d_plan.batch_id]
        ready(sim, "a", d_plan.batch_id, d_reservation)
        self.assertEqual(batch.status, "RUNNING")  # started by the last READY
        self.assertEqual([k for k in kinds(sim, "a", "D", 0, d_plan.batch_id)], ["PREPARE", "START"])
        self.assertEqual(sim.controller.inline_start_count, 1)
        driver.step()  # the driver finds it RUNNING and issues nothing more
        self.assertEqual([k for k in kinds(sim, "a", "D", 0, d_plan.batch_id)], ["PREPARE", "START"])
        # After pause() no grant is issued inline.
        for step, token in enumerate((12, 13, 14, 15), start=1):
            for rank in range(sim.tp):
                sim.step("a", plan=d_plan, rank=rank, step=step, token=token)
        sim.finish_run("a", d_plan, d_reservation, 4, 15)
        quiesce(sim, "a", d_plan, d_reservation)
        driver.pause()
        sim.submit("a", "s", max_new=2)
        sim.cleanup("a", "r")
        p2, _, _ = sim.prefill_ready("a", "s")
        self.assertEqual(sim.controller.groups["a"].batches[p2.batch_id].status, "PREPARING")

    def test_last_parking_run_starts_the_waiting_prefill(self):
        sim = Sim(("a", "b"), run_steps=8, park=True, inline_start=True)
        ControllerDriver(sim.controller, lambda event: None)
        sim.submit("a", "r", max_new=10)
        sim.prefill("a", "r", 11)
        plan, reservation = prepare_decode(sim, "a", "r", 8)
        ready(sim, "a", plan.batch_id, reservation)  # started inline
        self.assertTrue(all(sim.step("a", plan, rank, 1, 12)["continue"] for rank in range(sim.tp)))
        sim.submit("b", "x", max_new=4)
        p_plan, p_out, p_reservation = sim.prefill_ready("b", "x")
        p_batch = sim.controller.groups["b"].batches[p_plan.batch_id]
        self.assertEqual(p_batch.status, "PREPARING")  # needs the credits the run holds
        self.assertEqual(sim.step("a", plan, 0, 2, 13)["continue"], "PARK")
        self.assertEqual(p_batch.status, "PREPARING")  # rank 1 still computes step 2
        self.assertEqual(sim.step("a", plan, 1, 2, 13)["continue"], "PARK")
        self.assertEqual(p_batch.status, "RUNNING")  # started by the last park
        self.assertIn(p_plan.batch_id, sim.controller.credits.leases)
        self.assertEqual(sim.resume("a", plan, 0, 2), "PARK")
        sim.prefill_finish("b", "x", 21, p_plan, p_out, p_reservation)
        self.assertIs(sim.resume("a", plan, 0, 2), True)

    def test_unexpected_inline_failure_is_raised_by_the_driver(self):
        sim = Sim(run_steps=4, inline_start=True)
        driver = ControllerDriver(sim.controller, lambda event: None)
        sim.submit("a", "r", max_new=5)
        original = sim.store.start_kv_batch

        def broken(reservation_id):
            raise IntegrationError("STORE_BROKEN", "test")
        sim.store.start_kv_batch = broken
        plan, out, reservation = sim.prefill_ready("a", "r")  # last READY tries START
        sim.store.start_kv_batch = original
        self.assertEqual(sim.controller.inline_start_error.code, "STORE_BROKEN")
        expect("STORE_BROKEN", driver.step)

    def test_default_never_starts_inline(self):
        sim = Sim(run_steps=4)
        ControllerDriver(sim.controller, lambda event: None)
        sim.submit("a", "r", max_new=5)
        plan, _, _ = sim.prefill_ready("a", "r")
        self.assertEqual(sim.controller.groups["a"].batches[plan.batch_id].status, "PREPARING")


class EarlyToken(unittest.TestCase):
    def run_prefill_to_gpu_done(self, sim, rid="r"):
        sim.submit("a", rid, max_new=5)
        plan, out, reservation = sim.prefill_ready("a", rid)
        sim.controller.start_batch(sim.group("a"), plan.batch_id)
        for rank in range(sim.tp):
            cmd = next(c for c in sim.commands("a", "P", rank, "PREPARE") if c["batch_id"] == plan.batch_id)
            item, view = cmd["requests"][0], cmd["requests"][0]["reservation_view"]
            sim.controller.broker_stage(sim.group("a"), item["bootstrap_room"], rank, sim.pid("a", "P", rank), {
                "bootstrap_room": item["bootstrap_room"], "allocation_id": view["allocation_id"],
                "reserved_version": view["version"], "target_valid_tokens": item["target_valid_tokens"],
                "rank": rank, "native_payload": {"x": 1}})
        for rank in range(sim.tp):
            sim.report("a", "P", rank, plan.batch_id, "GPU_DONE", {
                "reservation_id": reservation, "duration_ms": 1.0, "gpu_forward_ms": 1.0,
                "results": [{"request_id": rid, "attempt": out["attempt"], "token_id": 11,
                             "valid_tokens": 3}]})
        return plan, reservation

    def finish_handoff(self, sim, plan, reservation, rid="r"):
        for rank in range(sim.tp):
            item = sim.commands("a", "D", rank, "HANDOFF")[-1]["requests"][0]
            sim.controller.broker_ack(sim.group("a"), item["bootstrap_room"], rank, sim.pid("a", "D", rank),
                                      item["allocation_id"], item["version"], item["valid_tokens"],
                                      item["rank_block_table"])
        for rank in range(sim.tp):
            item = sim.commands("a", "D", rank, "HANDOFF")[-1]["requests"][0]
            sim.report("a", "D", rank, plan.batch_id, "HANDOFF_READY", {"requests": [item]})
        for rank in range(sim.tp):
            sim.report("a", "P", rank, plan.batch_id, "QUIESCED",
                       {"reservation_id": reservation, "request_ids": (rid,)})

    def test_token_is_published_at_commit_and_decodable_after_handoff(self):
        sim = Sim(run_steps=4, prefill_early_token=True)
        plan, reservation = self.run_prefill_to_gpu_done(sim)
        request = sim.controller.requests[("a", "r")]
        self.assertEqual(sim.tokens("a"), [11])  # before any HANDOFF_READY
        self.assertEqual((request.output, request.state), ([11], "RUNNING"))
        self.assertIsNotNone(request.first_token_at)
        expect("STALE_ATTEMPT", lambda: prepare_decode(sim, "a", "r", 4))  # not decodable yet
        self.finish_handoff(sim, plan, reservation)
        self.assertEqual(sim.tokens("a"), [11])  # published exactly once
        self.assertEqual((request.output, request.state), ([11], "DECODING"))
        d_plan, d_reservation = prepare_decode(sim, "a", "r", 4)
        self.assertEqual(sim.controller.groups["a"].batches[d_plan.batch_id].status, "PREPARING")

    def test_cancel_during_handoff_keeps_the_request_out_of_decode(self):
        sim = Sim(run_steps=4, prefill_early_token=True)
        plan, reservation = self.run_prefill_to_gpu_done(sim)
        sim.controller.cancel_request("a", "r")
        self.finish_handoff(sim, plan, reservation)
        request = sim.controller.requests[("a", "r")]
        self.assertEqual(request.output, [11])
        self.assertNotEqual(request.state, "DECODING")
        self.assertEqual(request.state, "CLEANUP_PENDING")

    def test_default_publishes_at_handoff(self):
        sim = Sim(run_steps=4)
        plan, reservation = self.run_prefill_to_gpu_done(sim)
        self.assertEqual(sim.tokens("a"), [])
        self.finish_handoff(sim, plan, reservation)
        self.assertEqual(sim.tokens("a"), [11])
        self.assertEqual(sim.controller.requests[("a", "r")].state, "DECODING")


def cleanup_ids(sim, model, stage, rank):
    return [c["cleanup_id"] for c in sim.commands(model, stage, rank, "CLEANUP")]


def cleanup_quiesced(sim, model, rid, cleanup_id, stages):
    attempt = sim.controller.requests[(model, rid)].attempt
    for stage in stages:
        for rank in range(sim.tp):
            sim.report(model, stage, rank, cleanup_id, "QUIESCED",
                       {"cleanup_id": cleanup_id, "request_id": rid, "attempt": attempt})


def decode_to_end(sim, model, rid, steps, first_token):
    """One decode run of ``steps``; returns (plan, reservation, quiescing stages)."""
    request = sim.controller.requests[(model, rid)]
    ends = len(request.output) + steps >= request.spec.max_new_tokens
    plan, reservation = prepare_decode(sim, model, rid, steps)
    stages = ("D",) if sim.config.decode_skip_prefill_ranks else ("P", "D")
    ready(sim, model, plan.batch_id, reservation, stages=stages)
    sim.controller.start_batch(sim.group(model), plan.batch_id)
    for step in range(1, steps + 1):
        for rank in range(sim.tp):
            sim.step(model, plan, rank, step, first_token + step - 1, finished=ends and step == steps)
    sim.finish_run(model, plan, reservation, steps, first_token + steps - 1)
    return plan, reservation, stages


class CleanupFastPaths(unittest.TestCase):
    def test_released_prefill_ranks_get_no_request_cleanup(self):
        sim = Sim(run_steps=4, decode_skip_prefill_ranks=True, cleanup_skip_prefill_ranks=True)
        free = sim.store.rank_usage()[0]["free_blocks"]
        sim.submit("a", "r", max_new=5)
        sim.prefill("a", "r", 11)
        self.assertTrue(sim.controller.requests[("a", "r")].p_released)
        plan, reservation, stages = decode_to_end(sim, "a", "r", 4, 12)
        quiesce(sim, "a", plan, reservation, stages=stages)
        self.assertEqual(sim.controller.requests[("a", "r")].state, "CLEANUP_PENDING")
        (cleanup_id,) = cleanup_ids(sim, "a", "D", 0)
        for rank in range(sim.tp):
            self.assertEqual(cleanup_ids(sim, "a", "P", rank), [])
        # A P rank is no participant and cannot vote for the cleanup.
        expect("INVALID_REPORT", lambda: cleanup_quiesced(sim, "a", "r", cleanup_id, ("P",)))
        cleanup_quiesced(sim, "a", "r", cleanup_id, ("D",))
        output = sim.controller.request_output("a", "r")
        self.assertEqual((output["state"], output["output_token_ids"]), ("FINISHED", (11, 12, 13, 14, 15)))
        self.assertEqual(sim.store.rank_usage()[0]["free_blocks"], free)

    def test_request_without_a_released_prefill_cleans_every_rank(self):
        sim = Sim(run_steps=4, cleanup_skip_prefill_ranks=True)
        sim.submit("a", "q", max_new=5)
        sim.controller.cancel_request("a", "q")  # never prefilled: no release proof
        self.assertEqual(len(cleanup_ids(sim, "a", "P", 0)), 1)
        (first,) = cleanup_ids(sim, "a", "D", 0)
        cleanup_quiesced(sim, "a", "q", first, ("P", "D"))
        self.assertEqual(sim.controller.request_output("a", "q")["state"], "CANCELLED")
        sim.submit("a", "r", max_new=5)
        plan, out, reservation = sim.prefill_ready("a", "r")
        sim.controller.abort_batch(sim.group("a"), plan.batch_id, "test")  # revoked, not quiesced
        request = sim.controller.requests[("a", "r")]
        self.assertFalse(request.p_released)
        sim.controller.cancel_request("a", "r")
        quiesce(sim, "a", plan, reservation)
        self.assertEqual(len(cleanup_ids(sim, "a", "P", 0)), 2)

    def test_default_cleans_every_rank(self):
        sim = Sim(run_steps=4, decode_skip_prefill_ranks=True)
        sim.submit("a", "r", max_new=5)
        sim.prefill("a", "r", 11)
        plan, reservation, stages = decode_to_end(sim, "a", "r", 4, 12)
        quiesce(sim, "a", plan, reservation, stages=stages)
        self.assertEqual(len(cleanup_ids(sim, "a", "P", 0)), 1)

    def with_quiesce(self, **flags):
        sim = Sim(run_steps=4, decode_skip_prefill_ranks=True, cleanup_skip_prefill_ranks=True,
                  cleanup_with_quiesce=True, **flags)
        sim.submit("a", "r", max_new=5)
        sim.prefill("a", "r", 11)
        plan, reservation, stages = decode_to_end(sim, "a", "r", 4, 12)
        return sim, plan, reservation, stages

    def test_cleanup_follows_the_last_quiesce_in_command_order(self):
        sim, plan, reservation, stages = self.with_quiesce()
        free = sim.store.rank_usage()[0]["free_blocks"]
        for rank in range(sim.tp):
            commands = sim.commands("a", "D", rank)
            self.assertEqual([c["kind"] for c in commands][-2:], ["QUIESCE", "CLEANUP"])
            self.assertEqual(commands[-2]["batch_id"], plan.batch_id)
            self.assertEqual(cleanup_ids(sim, "a", "P", rank), [])
        (cleanup_id,) = cleanup_ids(sim, "a", "D", 0)
        batch = sim.controller.groups["a"].batches[plan.batch_id]
        # Each rank reports its batch QUIESCED, then its cleanup QUIESCED.
        rid = plan.requests[0].request_id
        for rank in range(sim.tp):
            sim.report("a", "D", rank, plan.batch_id, "QUIESCED",
                       {"reservation_id": reservation, "request_ids": (rid,)})
            self.assertEqual(batch.status, "DONE" if rank == sim.tp - 1 else "QUIESCE")
            attempt = sim.controller.requests[("a", rid)].attempt
            sim.report("a", "D", rank, cleanup_id, "QUIESCED",
                       {"cleanup_id": cleanup_id, "request_id": rid, "attempt": attempt})
        output = sim.controller.request_output("a", "r")
        self.assertEqual((output["state"], output["output_token_ids"]), ("FINISHED", (11, 12, 13, 14, 15)))
        self.assertGreater(sim.store.rank_usage()[0]["free_blocks"], free)
        self.assertEqual(sim.controller.groups["a"].cleanups, {})

    def test_cleanup_reported_before_the_batch_quiesced_fences(self):
        sim, plan, reservation, stages = self.with_quiesce()
        (cleanup_id,) = cleanup_ids(sim, "a", "D", 0)
        expect("INVALID_REPORT", lambda: cleanup_quiesced(sim, "a", "r", cleanup_id, ("D",)))
        self.assertTrue(sim.controller.groups["a"].fenced)

    def test_no_early_cleanup_when_p_ranks_quiesce_the_decode_batch(self):
        sim = Sim(run_steps=4, cleanup_skip_prefill_ranks=True, cleanup_with_quiesce=True)
        sim.submit("a", "r", max_new=5)
        sim.prefill("a", "r", 11)
        plan, reservation, stages = decode_to_end(sim, "a", "r", 4, 12)
        self.assertEqual(stages, ("P", "D"))
        self.assertEqual(cleanup_ids(sim, "a", "D", 0), [])  # P ranks would quiesce after it
        quiesce(sim, "a", plan, reservation, stages=stages)
        self.assertEqual(len(cleanup_ids(sim, "a", "D", 0)), 1)
        self.assertEqual(cleanup_ids(sim, "a", "P", 0), [])

    def test_unfinished_run_gets_no_early_cleanup(self):
        sim = Sim(run_steps=2, decode_skip_prefill_ranks=True, cleanup_skip_prefill_ranks=True,
                  cleanup_with_quiesce=True)
        sim.submit("a", "r", max_new=5)
        sim.prefill("a", "r", 11)
        plan, reservation, stages = decode_to_end(sim, "a", "r", 2, 12)
        self.assertEqual(cleanup_ids(sim, "a", "D", 0), [])
        quiesce(sim, "a", plan, reservation, stages=stages)
        self.assertEqual(sim.controller.requests[("a", "r")].state, "DECODING")
        self.assertEqual(cleanup_ids(sim, "a", "D", 0), [])


if __name__ == "__main__":
    unittest.main()
