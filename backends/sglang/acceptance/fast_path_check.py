"""CPU checks for the opt-in control-plane fast paths.

inline_start, decode_skip_prefill_ranks and prefill_early_token change when
the controller acts, never what it accepts. Simulated ranks drive the real
controller, store, policy and driver (decode_run_check.Sim); no Torch,
SGLang, model or GPU is involved.
"""
from __future__ import annotations

from pathlib import Path
import sys
import threading
import time
import unittest
from unittest import mock

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


def stage_and_gpu_done(sim, model, rid, token, plan, out, reservation):
    """The first half of Sim.prefill_finish: P staging and all P GPU_DONE."""
    for rank in range(sim.tp):
        cmd = next(c for c in sim.commands(model, "P", rank, "PREPARE") if c["batch_id"] == plan.batch_id)
        item = cmd["requests"][0]
        view = item["reservation_view"]
        sim.controller.broker_stage(sim.group(model), item["bootstrap_room"], rank, sim.pid(model, "P", rank), {
            "bootstrap_room": item["bootstrap_room"], "allocation_id": view["allocation_id"],
            "reserved_version": view["version"], "target_valid_tokens": item["target_valid_tokens"],
            "rank": rank, "native_payload": {"x": rank}})
    valid = len(sim.controller.requests[(model, rid)].spec.input_ids)
    for rank in range(sim.tp):
        sim.report(model, "P", rank, plan.batch_id, "GPU_DONE", {
            "reservation_id": reservation, "duration_ms": 1.0, "gpu_forward_ms": 1.0,
            "results": [{"request_id": rid, "attempt": out["attempt"], "token_id": token,
                         "valid_tokens": valid}]})


def handoff_ready(sim, model, plan):
    for rank in range(sim.tp):
        item = sim.commands(model, "D", rank, "HANDOFF")[-1]["requests"][0]
        sim.controller.broker_ack(sim.group(model), item["bootstrap_room"], rank, sim.pid(model, "D", rank),
                                  item["allocation_id"], item["version"], item["valid_tokens"],
                                  item["rank_block_table"])
    for rank in range(sim.tp):
        item = sim.commands(model, "D", rank, "HANDOFF")[-1]["requests"][0]
        sim.report(model, "D", rank, plan.batch_id, "HANDOFF_READY", {"requests": [item]})


class HandoffFastPaths(unittest.TestCase):
    def chained(self, **flags):
        sim = Sim(run_steps=4, inline_start=True, prefill_early_token=True,
                  decode_skip_prefill_ranks=True, handoff_prepare_decode=True, **flags)
        sim.controller.set_inline_start(True)  # a scheduling driver would allow it
        sim.submit("a", "r", max_new=5)
        plan, out, reservation = sim.prefill_ready("a", "r")
        sim.controller.start_batch(sim.group("a"), plan.batch_id)
        stage_and_gpu_done(sim, "a", "r", 11, plan, out, reservation)
        decode = next(b for b in sim.controller.groups["a"].batches.values() if b.plan.stage == "D")
        return sim, plan, reservation, decode

    def test_first_decode_batch_is_prepared_behind_the_handoff(self):
        sim, plan, reservation, decode = self.chained()
        self.assertEqual((decode.status, decode.plan.steps), ("PREPARING", 4))
        for rank in range(sim.tp):
            kinds_d = [c["kind"] for c in sim.commands("a", "D", rank)]
            self.assertEqual(kinds_d[-2:], ["HANDOFF", "PREPARE"])
            self.assertEqual(kinds(sim, "a", "P", rank, decode.plan.batch_id), [])
        request = sim.controller.requests[("a", "r")]
        self.assertEqual((request.state, request.output), ("PREPARING", [11]))
        handoff_ready(sim, "a", plan)
        self.assertEqual(request.state, "PREPARING")  # the decode batch owns the state
        ready(sim, "a", decode.plan.batch_id, decode.reservation.reservation_id, stages=("D",))
        self.assertEqual(decode.status, "RUNNING")  # last READY STARTs it inline
        quiesce(sim, "a", plan, reservation, stages=("P",))
        self.assertEqual(sim.controller.groups["a"].batches[plan.batch_id].status, "DONE")
        self.assertEqual(request.state, "RUNNING")  # the prefill's end does not reset it
        for step, token in enumerate((12, 13, 14, 15), start=1):
            for rank in range(sim.tp):
                sim.step("a", decode.plan, rank, step, token, finished=step == 4)
        sim.finish_run("a", decode.plan, decode.reservation.reservation_id, 4, 15)
        quiesce(sim, "a", decode.plan, decode.reservation.reservation_id, stages=("D",))
        sim.cleanup("a", "r")
        output = sim.controller.request_output("a", "r")
        self.assertEqual((output["state"], output["output_token_ids"]), ("FINISHED", (11, 12, 13, 14, 15)))

    def test_driver_proposal_for_the_chained_request_is_stale(self):
        sim, plan, reservation, decode = self.chained()
        handoff_ready(sim, "a", plan)
        out = sim.controller.request_output("a", "r")
        again = BatchPlan(sim.group("a"), "d-again", "a", "D",
                          (RequestRef("r", out["attempt"], out["output_count"]),), steps=4)
        expect("STAGE_BUSY", lambda: sim.controller.prepare_batch(again))  # benign for the driver

    def test_cancel_during_handoff_revokes_the_chained_batch(self):
        sim, plan, reservation, decode = self.chained()
        sim.controller.cancel_request("a", "r")
        self.assertEqual(decode.status, "REVOKING")
        for rank in range(sim.tp):
            self.assertEqual(sim.commands("a", "D", rank)[-1]["kind"], "REVOKE")

    def test_no_chain_without_inline_permission_or_switch(self):
        sim = Sim(run_steps=4, inline_start=True, prefill_early_token=True,
                  decode_skip_prefill_ranks=True, handoff_prepare_decode=True)
        sim.submit("a", "r", max_new=5)  # inline start never allowed (no driver)
        sim.prefill("a", "r", 11)
        self.assertFalse(any(b.plan.stage == "D" for b in sim.controller.groups["a"].batches.values()))
        self.assertEqual(sim.controller.requests[("a", "r")].state, "DECODING")

    def test_handoff_carries_staged_metadata_only_when_enabled(self):
        for flag in (False, True):
            with self.subTest(handoff_inline_metadata=flag):
                sim = Sim(run_steps=4, handoff_inline_metadata=flag)
                sim.submit("a", "r", max_new=5)
                plan, out, reservation = sim.prefill_ready("a", "r")
                sim.controller.start_batch(sim.group("a"), plan.batch_id)
                stage_and_gpu_done(sim, "a", "r", 11, plan, out, reservation)
                for rank in range(sim.tp):
                    item = sim.commands("a", "D", rank, "HANDOFF")[-1]["requests"][0]
                    self.assertEqual(item.get("native_metadata"), {"x": rank} if flag else None)
                handoff_ready(sim, "a", plan)  # the echoed items still match
                self.assertEqual(sim.controller.requests[("a", "r")].state, "DECODING")

    def run_to_last_step(self, **flags):
        sim = Sim(run_steps=4, **flags)
        sim.submit("a", "r", max_new=5)
        sim.prefill("a", "r", 11)
        plan, reservation = sim.prepare_decode("a", "r", 4)
        for step, token in enumerate((12, 13, 14), start=1):
            for rank in range(sim.tp):
                self.assertIs(sim.step("a", plan, rank, step, token)["continue"], True)
        start = sim.controller.groups["a"].batches[plan.batch_id].run_start_valid["r"]
        results = [{"request_id": "r", "attempt": plan.requests[0].attempt, "token_id": 15,
                    "valid_tokens": start + 4}]
        payload = {"reservation_id": reservation, "duration_ms": 2.0, "gpu_forward_ms": 1.0,
                   "steps": 4, "results": results,
                   "final_step": {"step": 4, "results": results, "finished": True}}
        return sim, plan, reservation, payload

    def test_gpu_done_carries_the_final_step(self):
        sim, plan, reservation, payload = self.run_to_last_step(final_step_in_gpu_done=True)
        for rank in range(sim.tp):
            sim.report("a", "D", rank, plan.batch_id, "GPU_DONE", payload)
        batch = sim.controller.groups["a"].batches[plan.batch_id]
        self.assertEqual(batch.status, "QUIESCE")
        self.assertEqual(sim.controller.requests[("a", "r")].output, [11, 12, 13, 14, 15])

    def test_final_step_needs_the_switch_and_matching_results(self):
        sim, plan, reservation, payload = self.run_to_last_step()
        expect("INVALID_REPORT", lambda: sim.report("a", "D", 0, plan.batch_id, "GPU_DONE", payload))
        sim, plan, reservation, payload = self.run_to_last_step(final_step_in_gpu_done=True)
        payload["final_step"] = {**payload["final_step"], "step": 3}
        expect("INVALID_REPORT", lambda: sim.report("a", "D", 0, plan.batch_id, "GPU_DONE", payload))

    def test_long_poll_reports_bootstrap_ready_rooms_to_p_ranks(self):
        sim = Sim(run_steps=4, bootstrap_in_poll=True)
        sim.submit("a", "r", max_new=5)
        out = sim.controller.request_output("a", "r")
        plan = BatchPlan(sim.group("a"), "p-r", "a", "P", (RequestRef("r", out["attempt"], 0),))
        sim.controller.prepare_batch(plan)
        poll = lambda stage: sim.controller.poll_commands_wait(
            sim.group("a"), stage, 0, sim.pid("a", stage, 0), sim.acked[("a", stage, 0)], None, 0, None)
        self.assertEqual(tuple(poll("P")["bootstrap_ready"]), ())
        self.assertNotIn("bootstrap_ready", poll("D"))
        sim.controller.abort_batch(sim.group("a"), "p-r", "test")
        sim2 = Sim(run_steps=4, bootstrap_in_poll=True)
        sim2.submit("a", "r", max_new=5)
        plan, out, reservation = sim2.prefill_ready("a", "r")  # every D destination registered
        item = next(c for c in sim2.commands("a", "P", 0, "PREPARE") if c["batch_id"] == plan.batch_id)["requests"][0]
        view = item["reservation_view"]
        reply = sim2.controller.poll_commands_wait(sim2.group("a"), "P", 0, sim2.pid("a", "P", 0),
                                                   sim2.acked[("a", "P", 0)], None, 0, None)
        self.assertEqual([tuple(x) for x in reply["bootstrap_ready"]],
                         [(item["bootstrap_room"], view["allocation_id"], view["version"],
                           item["target_valid_tokens"])])

    def test_prepare_carries_only_the_receiving_ranks_table(self):
        for flag in (False, True):
            with self.subTest(prepare_own_rank_table=flag):
                sim = Sim(run_steps=4, prepare_own_rank_table=flag)
                sim.submit("a", "r", max_new=5)
                sim.prefill("a", "r", 11)  # the whole prefill still works
                plan, reservation = prepare_decode(sim, "a", "r", 4)
                for stage in ("P", "D"):
                    for rank in range(sim.tp):
                        item = next(c for c in sim.commands("a", stage, rank, "PREPARE")
                                    if c["batch_id"] == plan.batch_id)["requests"][0]
                        tables = item["reservation_view"]["block_tables"]
                        self.assertEqual(set(tables), {rank} if flag else set(range(sim.tp)))
                        self.assertEqual(tables[rank], item["rank_block_table"])

    def inline(self, **flags):
        sim = Sim(run_steps=4, inline_start=True, inline_prepare=True, **flags)
        sim.controller.set_inline_start(True)
        return sim

    def test_submitted_prefill_is_prepared_inline(self):
        sim = self.inline()
        sim.submit("a", "r", max_new=5)
        sim.controller.prepare_submitted("a", "r")
        (batch,) = sim.controller.groups["a"].batches.values()
        self.assertEqual((batch.plan.stage, batch.status), ("P", "PREPARING"))
        self.assertEqual(sim.controller.requests[("a", "r")].state, "PREPARING")
        sim.controller.prepare_submitted("a", "r")  # a replay prepares nothing more
        self.assertEqual(len(sim.controller.groups["a"].batches), 1)

    def test_inline_prepare_keeps_fifo_and_needs_permission(self):
        sim = self.inline()
        sim.submit("a", "old", max_new=5)  # left for the driver
        sim.submit("a", "new", max_new=5)
        sim.controller.prepare_submitted("a", "new")
        self.assertEqual(sim.controller.groups["a"].batches, {})
        sim = Sim(run_steps=4, inline_start=True, inline_prepare=True)  # no scheduling driver
        sim.submit("a", "r", max_new=5)
        sim.controller.prepare_submitted("a", "r")
        self.assertEqual(sim.controller.groups["a"].batches, {})
        sim = Sim(run_steps=4, inline_start=True)  # switch off
        sim.controller.set_inline_start(True)
        sim.submit("a", "r", max_new=5)
        sim.controller.prepare_submitted("a", "r")
        self.assertEqual(sim.controller.groups["a"].batches, {})

    def test_event_pump_thread_delivers_events(self):
        import threading
        sim = Sim(run_steps=4, event_pump_thread=True)
        seen, arrived = [], threading.Event()

        def apply(event):
            seen.append(event.kind)
            arrived.set()
        driver = ControllerDriver(sim.controller, apply, schedule=False)
        driver.start()
        try:
            self.assertTrue(driver._pumping)
            sim.submit("a", "r", max_new=5)
            sim.prefill("a", "r", 11)  # publishes the first TOKEN
            self.assertTrue(arrived.wait(5))
            self.assertIn("TOKEN", seen)
        finally:
            driver.stop()
        self.assertFalse(driver.pump_thread.is_alive())


def destinations(sim, model, plan, rank):
    """What a register_in_ready D rank puts in its prefill READY."""
    cmd = next(c for c in sim.commands(model, "D", rank, "PREPARE") if c["batch_id"] == plan.batch_id)
    item = cmd["requests"][0]
    view = item["reservation_view"]
    return ({"room": item["bootstrap_room"], "request_id": item["request_id"], "attempt": item["attempt"],
             "allocation_id": view["allocation_id"], "reserved_version": view["version"],
             "target_valid_tokens": item["target_valid_tokens"], "rank_block_table": item["rank_block_table"],
             "metadata_slot": 0, "logical_token_count": item["target_valid_tokens"]},)


class ProtocolFastPaths(unittest.TestCase):
    def prepared_prefill(self, sim, rid="r"):
        out = sim.controller.request_output("a", rid)
        plan = BatchPlan(sim.group("a"), "p-" + rid, "a", "P", (RequestRef(rid, out["attempt"], 0),))
        return plan, out, sim.controller.prepare_batch(plan)["reservation_id"]

    def test_d_destinations_ride_in_the_prefill_ready(self):
        sim = Sim(run_steps=4, register_in_ready=True, bootstrap_in_poll=True)
        sim.submit("a", "r", max_new=5)
        plan, out, reservation = self.prepared_prefill(sim)
        room = destinations(sim, "a", plan, 0)[0]["room"]
        entry = sim.controller.groups["a"].broker[room]
        # A READY without them, or with another room, registers nothing.
        expect("INVALID_REPORT", lambda: sim.report("a", "D", 0, plan.batch_id, "READY",
                                                    {"reservation_id": reservation}))
        wrong = ({**destinations(sim, "a", plan, 0)[0], "room": room + 99},)
        expect("INVALID_REPORT", lambda: sim.report("a", "D", 0, plan.batch_id, "READY",
                                                    {"reservation_id": reservation, "destinations": wrong}))
        self.assertEqual(entry["destinations"], {})
        for rank in range(sim.tp):
            sim.report("a", "D", rank, plan.batch_id, "READY",
                       {"reservation_id": reservation, "destinations": destinations(sim, "a", plan, rank)})
        self.assertEqual(set(entry["destinations"]), set(range(sim.tp)))
        reply = sim.controller.poll_commands_wait(sim.group("a"), "P", 0, sim.pid("a", "P", 0),
                                                  sim.acked[("a", "P", 0)], None, 0, None)
        self.assertEqual([x[0] for x in reply["bootstrap_ready"]], [room])
        ready(sim, "a", plan.batch_id, reservation, stages=("P",))
        sim.controller.start_batch(sim.group("a"), plan.batch_id)
        sim.prefill_finish("a", "r", 11, plan, out, reservation)  # stage, handoff: unchanged
        self.assertEqual(sim.controller.requests[("a", "r")].state, "DECODING")

    def test_default_ready_carries_no_destinations(self):
        sim = Sim(run_steps=4)
        sim.submit("a", "r", max_new=5)
        plan, out, reservation = self.prepared_prefill(sim)
        expect("STALE_REPORT", lambda: sim.report("a", "D", 0, plan.batch_id, "READY", {
            "reservation_id": reservation, "destinations": destinations(sim, "a", plan, 0)}))

    def test_long_poll_reports_broker_status_to_every_rank(self):
        for flag in (False, True):
            with self.subTest(broker_status_in_poll=flag):
                sim = Sim(run_steps=4, broker_status_in_poll=flag)
                sim.submit("a", "r", max_new=5)
                plan, out, reservation = sim.prefill_ready("a", "r")
                sim.controller.start_batch(sim.group("a"), plan.batch_id)
                stage_and_gpu_done(sim, "a", "r", 11, plan, out, reservation)
                poll = lambda stage, rank: sim.controller.poll_commands_wait(
                    sim.group("a"), stage, rank, sim.pid("a", stage, rank), sim.acked[("a", stage, rank)],
                    None, 0, None).get("broker_status")
                if not flag:
                    self.assertIsNone(poll("D", 0))
                    continue
                room = sim.commands("a", "D", 0, "HANDOFF")[-1]["requests"][0]["bootstrap_room"]
                self.assertEqual(dict(poll("D", 0)), {room: "PENDING"})
                for rank in range(sim.tp):
                    item = sim.commands("a", "D", rank, "HANDOFF")[-1]["requests"][0]
                    sim.controller.broker_ack(sim.group("a"), room, rank, sim.pid("a", "D", rank),
                                              item["allocation_id"], item["version"], item["valid_tokens"],
                                              item["rank_block_table"])
                for stage in ("P", "D"):
                    self.assertEqual(dict(poll(stage, 1)), {room: "SUCCESS"})
                    self.assertEqual(sim.controller.broker_poll(sim.group("a"), room, 1,
                                                                sim.pid("a", stage, 1), stage)["status"], "SUCCESS")

    def ended(self, **flags):
        sim = Sim(run_steps=4, decode_skip_prefill_ranks=True, cleanup_skip_prefill_ranks=True,
                  cleanup_with_quiesce=True, **flags)
        sim.submit("a", "r", max_new=5)
        sim.prefill("a", "r", 11)
        plan, reservation, stages = decode_to_end(sim, "a", "r", 4, 12)
        return sim, plan, reservation

    def finish_cleanup(self, sim, plan, reservation):
        (cleanup_id,) = cleanup_ids(sim, "a", "D", 0)
        attempt = sim.controller.requests[("a", "r")].attempt
        for rank in range(sim.tp):
            sim.report("a", "D", rank, plan.batch_id, "QUIESCED",
                       {"reservation_id": reservation, "request_ids": ("r",)})
            sim.report("a", "D", rank, cleanup_id, "QUIESCED",
                       {"cleanup_id": cleanup_id, "request_id": "r", "attempt": attempt})

    def terminals(self, sim):
        return [e for e in sim.controller.poll_events(sim.group("a")) if e.kind == "TERMINAL"]

    def test_finished_request_is_terminal_when_its_cleanup_is_queued(self):
        sim, plan, reservation = self.ended(terminal_before_cleanup=True)
        (event,) = self.terminals(sim)
        self.assertEqual((event.payload["output_token_ids"], event.payload["finish_reason"]),
                         ((11, 12, 13, 14, 15), "length"))
        tokens = [e for e in sim.controller.poll_events(sim.group("a")) if e.kind == "TOKEN"]
        self.assertLess(max(e.event_seq for e in tokens), event.event_seq)
        self.assertEqual(sim.controller.requests[("a", "r")].state, "CLEANUP_PENDING")
        expect("NOT_READY", lambda: sim.controller.forget_request("a", "r", event.attempt))
        free = sim.store.rank_usage()[0]["free_blocks"]
        (cleanup_id,) = cleanup_ids(sim, "a", "D", 0)
        self.finish_cleanup(sim, plan, reservation)
        self.assertEqual(len(self.terminals(sim)), 1)  # the cleanup emits nothing more
        self.assertEqual(sim.controller.request_output("a", "r")["state"], "FINISHED")
        self.assertGreater(sim.store.rank_usage()[0]["free_blocks"], free)
        self.assertNotIn(cleanup_id, sim.controller.groups["a"].outbox.reserved)

    def test_default_and_cancelled_requests_are_terminal_after_cleanup(self):
        sim, plan, reservation = self.ended()
        self.assertEqual(self.terminals(sim), [])
        self.finish_cleanup(sim, plan, reservation)
        self.assertEqual(len(self.terminals(sim)), 1)
        sim = Sim(run_steps=4, terminal_before_cleanup=True)
        sim.submit("a", "q", max_new=5)
        sim.controller.cancel_request("a", "q")
        self.assertEqual(self.terminals(sim), [])  # not a normal finish: after the release

    def test_d_only_cleanup_frees_the_slot_behind_its_commands(self):
        sim, plan, reservation = self.ended(slot_free_at_cleanup=True)
        sim.submit("a", "s", max_new=5)
        nxt, out, reservation_s = self.prepared_prefill(sim, "s")  # while "r" still cleans
        for rank in range(sim.tp):
            kinds_d = [c["kind"] for c in sim.commands("a", "D", rank)]
            self.assertEqual(kinds_d[-3:], ["QUIESCE", "CLEANUP", "PREPARE"])
            self.assertEqual(sim.commands("a", "D", rank)[-1]["batch_id"], nxt.batch_id)
        self.finish_cleanup(sim, plan, reservation)
        self.assertEqual(sim.controller.request_output("a", "r")["state"], "FINISHED")
        self.assertEqual(sim.controller.groups["a"].batches[nxt.batch_id].status, "PREPARING")

    def test_slot_stays_taken_without_the_switch_or_with_p_cleanup(self):
        sim, plan, reservation = self.ended()
        sim.submit("a", "s", max_new=5)
        expect("OUT_OF_CAPACITY", lambda: self.prepared_prefill(sim, "s"))
        sim = Sim(run_steps=4, cleanup_skip_prefill_ranks=True, slot_free_at_cleanup=True)
        sim.submit("a", "q", max_new=5)
        sim.controller.cancel_request("a", "q")  # never prefilled: P ranks clean too
        sim.submit("a", "s", max_new=5)
        expect("OUT_OF_CAPACITY", lambda: self.prepared_prefill(sim, "s"))

    def test_new_switches_are_boolean_and_slot_needs_d_only_cleanup(self):
        import json, tempfile
        from muxserve_sglang.config import load_config
        root = Path(__file__).resolve().parents[1]

        def load(**extra):
            raw = json.loads((root / "configs/sglang_a100.json").read_text())
            raw.update(extra)
            directory = tempfile.mkdtemp()
            raw["asset_lock_dir"] = raw["runtime_dir"] = directory
            path = Path(directory) / "c.json"
            path.write_text(json.dumps(raw))
            return load_config(path, require_assets=False)
        names = ("register_in_ready", "broker_status_in_poll", "terminal_before_cleanup")
        config = load()
        self.assertFalse(any(getattr(config, n) for n in names + ("slot_free_at_cleanup",)))
        for name in names:
            self.assertTrue(getattr(load(**{name: True}), name))
            expect("INVALID_CONFIG", lambda: load(**{name: 1}))
        expect("INVALID_CONFIG", lambda: load(slot_free_at_cleanup=True))
        self.assertTrue(load(slot_free_at_cleanup=True, cleanup_skip_prefill_ranks=True).slot_free_at_cleanup)


class HandoffRoundTrips(unittest.TestCase):
    def chained(self, **flags):
        return HandoffFastPaths.chained(self, **flags)

    def poll(self, sim, rank, wait_ms, epoch, acks=(), stage="D"):
        target = sim.controller.groups["a"].registered[(stage, rank)]
        started = time.monotonic()
        reply = sim.controller.poll_commands_wait(sim.group("a"), stage, rank, sim.pid("a", stage, rank),
                                                  target.next_command - 1, None, wait_ms, epoch, acks)
        return reply, time.monotonic() - started

    def acks(self, sim, rank):
        item = sim.commands("a", "D", rank, "HANDOFF")[-1]["requests"][0]
        return ((item["bootstrap_room"], item["allocation_id"], item["version"], item["valid_tokens"],
                 item["rank_block_table"]),)

    def published(self, **flags):
        sim = Sim(run_steps=4, broker_status_in_poll=True, **flags)
        sim.submit("a", "r", max_new=5)
        plan, out, reservation = sim.prefill_ready("a", "r")
        sim.controller.start_batch(sim.group("a"), plan.batch_id)
        stage_and_gpu_done(sim, "a", "r", 11, plan, out, reservation)
        return sim

    def test_ack_rides_on_the_poll_and_only_the_last_returns_at_once(self):
        sim = self.published(handoff_ack_in_poll=True)
        room = self.acks(sim, 0)[0][0]
        epoch = self.poll(sim, 0, 0, None)[0]["epoch"]
        reply, elapsed = self.poll(sim, 0, 40, epoch, self.acks(sim, 0))
        self.assertGreaterEqual(elapsed, 0.03)  # its own ACK does not end the wait
        self.assertEqual(dict(reply["broker_status"]), {room: "PENDING"})
        reply, elapsed = self.poll(sim, 1, 900, reply["epoch"], self.acks(sim, 1))
        self.assertLess(elapsed, 0.5)
        self.assertEqual(dict(reply["broker_status"]), {room: "SUCCESS"})
        self.assertEqual(sim.controller.groups["a"].broker[room]["acked"], {0, 1})

    def test_a_poll_ack_gets_the_same_checks_as_the_ack_rpc(self):
        sim = self.published(handoff_ack_in_poll=True)
        (ack,) = self.acks(sim, 0)
        expect("HANDOFF_NOT_READY", lambda: self.poll(sim, 0, 0, None, ((ack[0], "other", *ack[2:]),)))
        expect("INVALID_ARGUMENT", lambda: self.poll(sim, 0, 0, None, (ack[:4],)))
        expect("INVALID_ARGUMENT", lambda: self.poll(sim, 0, 0, None, (ack,), stage="P"))
        sim = self.published()
        expect("INVALID_ARGUMENT", lambda: self.poll(sim, 0, 0, None, self.acks(sim, 0)))

    def acked_all(self, sim):
        for rank in range(sim.tp):
            (ack,) = self.acks(sim, rank)
            sim.controller.broker_ack(sim.group("a"), ack[0], rank, sim.pid("a", "D", rank), *ack[1:])

    def combined(self, sim, rank, plan, decode):
        key = ("a", "D", rank)
        base = sim.seq[key]
        first = (str(base + 1), plan.batch_id, "HANDOFF_READY",
                 {"requests": sim.commands("a", "D", rank, "HANDOFF")[-1]["requests"]})
        second = (str(base + 2), decode.plan.batch_id, "READY",
                  {"reservation_id": decode.reservation.reservation_id})
        sim.seq[key] = base + 2
        return (first, second)

    def test_handoff_ready_and_ready_are_one_report_call(self):
        sim, plan, reservation, decode = self.chained(handoff_report_with_ready=True)
        self.acked_all(sim)
        call = lambda rank, pair: sim.controller.rank_reports(sim.group("a"), "D", rank, sim.pid("a", "D", rank), pair)
        pairs = [self.combined(sim, rank, plan, decode) for rank in range(sim.tp)]
        for rank, pair in enumerate(pairs):
            self.assertTrue(call(rank, pair)["accepted"])
        self.assertEqual(decode.status, "RUNNING")  # the last pair STARTed it inline
        self.assertEqual(sim.controller.groups["a"].batches[plan.batch_id].handoff, set(range(sim.tp)))
        self.assertTrue(call(0, pairs[0])["duplicate"])  # a replay is idempotent

    def test_combined_report_is_only_for_a_handoff_ready_pair(self):
        sim, plan, reservation, decode = self.chained()
        self.acked_all(sim)
        call = lambda pair: sim.controller.rank_reports(sim.group("a"), "D", 0, sim.pid("a", "D", 0), pair)
        pair = self.combined(sim, 0, plan, decode)
        expect("INVALID_REPORT", lambda: call(pair))  # switch off
        sim, plan, reservation, decode = self.chained(handoff_report_with_ready=True)
        self.acked_all(sim)
        pair = self.combined(sim, 0, plan, decode)
        expect("INVALID_REPORT", lambda: call((pair[1], pair[0])))
        expect("INVALID_REPORT", lambda: call((pair[0],)))

    def test_only_the_ready_that_starts_the_batch_returns_start(self):
        for flag in (False, True):
            with self.subTest(start_in_report_reply=flag):
                sim, plan, reservation, decode = self.chained(start_in_report_reply=flag)
                handoff_ready(sim, "a", plan)
                replies = []
                for rank in range(sim.tp):
                    key = ("a", "D", rank)
                    sim.seq[key] += 1
                    replies.append(sim.controller.rank_report(
                        sim.group("a"), "D", rank, sim.pid("a", "D", rank), str(sim.seq[key]),
                        decode.plan.batch_id, "READY", {"reservation_id": decode.reservation.reservation_id}))
                self.assertNotIn("commands", replies[0])
                if not flag:
                    self.assertNotIn("commands", replies[-1])
                    continue
                (command,) = replies[-1]["commands"]
                target = sim.controller.groups["a"].registered[("D", sim.tp - 1)]
                self.assertEqual((command["kind"], command["batch_id"], command["command_seq"]),
                                 ("START", decode.plan.batch_id, target.next_command - 1))

    def test_long_poll_wakes_only_for_its_own_command_or_group_change(self):
        sim = Sim(run_steps=4, targeted_wake=True)
        controller, group = sim.controller, sim.controller.groups["a"]
        epoch = self.poll(sim, 0, 0, None)[0]["epoch"]
        results = {}

        def waiter(rank):
            results[rank] = self.poll(sim, rank, 900, epoch)
        threads = [threading.Thread(target=waiter, args=(rank,)) for rank in (0, 1)]
        for thread in threads:
            thread.start()
        deadline = time.monotonic() + 2
        targets = [group.registered[("D", rank)] for rank in (0, 1)]
        while time.monotonic() < deadline and any(t.poll_cv is None for t in targets):
            time.sleep(0.005)
        time.sleep(0.05)
        with mock.patch.object(targets[1].poll_cv, "notify_all", wraps=targets[1].poll_cv.notify_all) as other:
            with controller.lock:
                controller._command(group, ("D", 0), "QUIESCE", {"batch_id": "x"})
            threads[0].join(1)
            self.assertFalse(threads[0].is_alive())
            self.assertEqual(other.call_count, 0)
            self.assertEqual(results[0][0]["commands"][0]["kind"], "QUIESCE")
            with controller.lock:
                controller._notify(group)  # a broker change wakes the group
            threads[1].join(1)
            self.assertFalse(threads[1].is_alive())
            self.assertEqual(other.call_count, 1)
        self.assertLess(results[1][1], 0.8)

    def test_new_round_trip_switches_validate(self):
        import json, tempfile
        from muxserve_sglang.config import load_config
        root = Path(__file__).resolve().parents[1]

        def load(**extra):
            raw = json.loads((root / "configs/sglang_a100.json").read_text())
            raw.update(extra)
            directory = tempfile.mkdtemp()
            raw["asset_lock_dir"] = raw["runtime_dir"] = directory
            path = Path(directory) / "c.json"
            path.write_text(json.dumps(raw))
            return load_config(path, require_assets=False)
        chain = dict(inline_start=True, prefill_early_token=True, decode_skip_prefill_ranks=True,
                     handoff_prepare_decode=True)
        config = load()
        for name in ("handoff_ack_in_poll", "handoff_report_with_ready",
                     "start_in_report_reply", "targeted_wake", "rpc_inline_fast_methods",
                     "decode_leader_reports", "decode_prepare_light", "handoff_prepare_after_ack"):
            self.assertFalse(getattr(config, name))
            expect("INVALID_CONFIG", lambda: load(**{name: 1}))
        for name in ("start_in_report_reply", "targeted_wake", "rpc_inline_fast_methods",
                     "decode_leader_reports", "decode_prepare_light"):
            self.assertTrue(getattr(load(**{name: True}), name))
        for name in ("handoff_report_with_ready",):
            expect("INVALID_CONFIG", lambda: load(**{name: True}))
            self.assertTrue(getattr(load(**chain, **{name: True}), name))
        expect("INVALID_CONFIG", lambda: load(handoff_ack_in_poll=True))
        self.assertTrue(load(handoff_ack_in_poll=True, broker_status_in_poll=True).handoff_ack_in_poll)
        ack = dict(chain, handoff_ack_in_poll=True, broker_status_in_poll=True)
        expect("INVALID_CONFIG", lambda: load(handoff_prepare_after_ack=True))
        expect("INVALID_CONFIG", lambda: load(**chain, handoff_prepare_after_ack=True))
        self.assertTrue(load(**ack, handoff_prepare_after_ack=True).handoff_prepare_after_ack)


class PrepareAfterAck(unittest.TestCase):
    def flow(self):
        sim = Sim(run_steps=4, inline_start=True, prefill_early_token=True, decode_skip_prefill_ranks=True,
                  handoff_prepare_decode=True, handoff_prepare_after_ack=True, handoff_ack_in_poll=True,
                  broker_status_in_poll=True)
        sim.controller.set_inline_start(True)
        sim.submit("a", "r", max_new=5)
        plan, out, reservation = sim.prefill_ready("a", "r")
        sim.controller.start_batch(sim.group("a"), plan.batch_id)
        stage_and_gpu_done(sim, "a", "r", 11, plan, out, reservation)
        return sim, plan

    def poll(self, sim, rank, acks=()):
        return HandoffRoundTrips.poll(self, sim, rank, 0, None, acks)[0]

    def decoding(self, sim):
        return [b for b in sim.controller.groups["a"].batches.values() if b.plan.stage == "D"]

    def test_decode_batch_is_reserved_only_after_every_rank_saw_the_ack(self):
        sim, plan = self.flow()
        self.assertEqual(self.decoding(sim), [])
        room = HandoffRoundTrips.acks(self, sim, 0)[0][0]
        self.assertEqual(dict(self.poll(sim, 0, HandoffRoundTrips.acks(self, sim, 0))["broker_status"]), {room: "PENDING"})
        self.assertEqual(dict(self.poll(sim, 1, HandoffRoundTrips.acks(self, sim, 1))["broker_status"]), {room: "SUCCESS"})
        self.assertIsNone(sim.controller._prepare_thread)  # rank 0 has not read SUCCESS yet
        self.assertEqual(self.decoding(sim), [])
        self.assertEqual(dict(self.poll(sim, 0)["broker_status"]), {room: "SUCCESS"})
        sim.controller._prepare_thread.join(2)
        (decode,) = self.decoding(sim)
        self.assertEqual(decode.status, "PREPARING")
        for rank in range(sim.tp):
            self.assertEqual([c["kind"] for c in sim.commands("a", "D", rank)][-2:], ["HANDOFF", "PREPARE"])
        self.assertEqual(sim.controller.groups["a"].ack_gates, {})

    def test_gate_is_dropped_once_the_prefill_left_handoff(self):
        sim, plan = self.flow()
        group = sim.controller.groups["a"]
        group.batches[plan.batch_id].status = "QUIESCE"
        self.poll(sim, 0)
        self.assertEqual(group.ack_gates, {})
        self.assertIsNone(sim.controller._prepare_thread)
        self.assertEqual(self.decoding(sim), [])


class LeaderReports(unittest.TestCase):
    def decode(self, *models, **flags):
        sim = Sim(models or ("a",), run_steps=4, decode_leader_reports=True, **flags)
        sim.submit("a", "r", max_new=5)
        sim.prefill("a", "r", 11)
        plan, reservation = sim.prepare_decode("a", "r", 4)
        return sim, plan, reservation

    def done(self, sim, plan, reservation, digests, steps=4, token=15, **extra):
        rid = plan.requests[0].request_id
        start = sim.controller.groups["a"].batches[plan.batch_id].run_start_valid[rid]
        for rank in range(sim.tp):
            payload = {"reservation_id": reservation, "duration_ms": 2.0, "gpu_forward_ms": 1.0, "steps": steps,
                       "results": [{"request_id": rid, "attempt": plan.requests[0].attempt,
                                    "token_id": token, "valid_tokens": start + steps}]}
            if digests is not None:
                payload["token_digest"] = digests[rank]
            sim.report("a", "D", rank, plan.batch_id, "GPU_DONE", payload)

    def output(self, sim):
        return sim.controller.requests[("a", "r")].output

    def test_leader_publishes_each_continuing_step_alone(self):
        sim, plan, reservation = self.decode()
        for step, token in enumerate((12, 13, 14), start=1):
            self.assertIs(sim.step("a", plan, 0, step, token)["continue"], True)
            self.assertEqual(self.output(sim)[-1], token)
        self.assertEqual(sim.controller.groups["a"].batches[plan.batch_id].run_reports[3].keys(), {0})
        for rank in range(sim.tp):  # the locally final step is reported by every rank
            self.assertIs(sim.step("a", plan, rank, 4, 15, finished=True)["continue"], False)
        self.assertEqual(self.output(sim), [11, 12, 13, 14, 15])
        self.done(sim, plan, reservation, [7, 7])
        self.assertEqual(sim.controller.groups["a"].batches[plan.batch_id].status, "QUIESCE")

    def test_non_leader_final_report_first_does_not_publish(self):
        sim, plan, reservation = self.decode()
        for step, token in enumerate((12, 13, 14), start=1):
            sim.step("a", plan, 0, step, token)
        sim.step("a", plan, 1, 4, 15, finished=True)
        self.assertEqual(self.output(sim)[-1], 14)
        sim.step("a", plan, 0, 4, 15, finished=True)
        self.assertEqual(self.output(sim)[-1], 15)

    def test_steps_must_be_granted_in_order(self):
        sim, plan, reservation = self.decode()
        expect("REPORT_GAP", lambda: sim.step("a", plan, 0, 2, 13))
        sim.step("a", plan, 0, 1, 12)
        expect("REPORT_GAP", lambda: sim.step("a", plan, 0, 3, 14))
        expect("REPORT_GAP", lambda: sim.step("a", plan, 1, 3, 14))  # step 2 has no decision yet

    def test_without_the_switch_every_rank_must_report(self):
        sim = Sim(run_steps=4)
        sim.submit("a", "r", max_new=5)
        sim.prefill("a", "r", 11)
        plan, reservation = sim.prepare_decode("a", "r", 4)
        sim.step("a", plan, 0, 1, 12)
        self.assertEqual(self.output(sim), [11])
        sim.step("a", plan, 1, 1, 12)
        self.assertEqual(self.output(sim), [11, 12])

    def test_a_park_is_still_reported_by_every_rank(self):
        sim = Sim(("a", "b"), run_steps=8, park=True, decode_leader_reports=True)
        sim.submit("a", "r", max_new=10)
        sim.prefill("a", "r", 11)
        plan, reservation = sim.prepare_decode("a", "r", 8)
        self.assertIs(sim.step("a", plan, 0, 1, 12)["continue"], True)
        sim.submit("b", "x", max_new=4)
        sim.prefill_ready("b", "x")
        credits = sim.controller.credits
        self.assertEqual(sim.step("a", plan, 0, 2, 13)["continue"], "PARK")
        self.assertIn(plan.batch_id, credits.leases)  # rank 1 has not finished step 2 yet
        self.assertEqual(sim.step("a", plan, 1, 2, 13)["continue"], "PARK")  # the rank asks: replayed
        self.assertNotIn(plan.batch_id, credits.leases)
        self.assertEqual(self.output(sim), [11, 12, 13])
        self.assertEqual(sim.resume("a", plan, 1, 2), "PARK")

    def test_token_digests_must_agree_and_are_required(self):
        sim, plan, reservation = self.decode()
        for step, token in enumerate((12, 13, 14), start=1):
            sim.step("a", plan, 0, step, token)
        for rank in range(sim.tp):
            sim.step("a", plan, rank, 4, 15, finished=True)
        expect("INVALID_REPORT", lambda: self.done(sim, plan, reservation, None))
        expect("IDENTITY_CONFLICT", lambda: self.done(sim, plan, reservation, [7, 8]))
        self.assertIsNotNone(sim.controller.groups["a"].fenced)

    def test_digest_is_refused_without_the_switch(self):
        sim = Sim(run_steps=4)
        sim.submit("a", "r", max_new=5)
        sim.prefill("a", "r", 11)
        plan, reservation = sim.prepare_decode("a", "r", 4)
        for step, token in enumerate((12, 13, 14, 15), start=1):
            for rank in range(sim.tp):
                sim.step("a", plan, rank, step, token, finished=step == 4)
        expect("INVALID_REPORT", lambda: self.done(sim, plan, reservation, [7, 7]))


class StepBoxChecks(unittest.TestCase):
    def box(self):
        import tempfile
        from muxserve_sglang.step_box import StepBox
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        path = directory.name + "/box"
        return StepBox(path), StepBox(path)

    def test_verdicts_reach_another_mapping_of_the_same_file(self):
        leader, follower = self.box()
        leader.publish(0xABCD1234, 7, True)
        self.assertIs(follower.wait(0xABCD1234, 7, 1), True)
        leader.publish(0xABCD1234, 8, False)
        self.assertIs(follower.wait(0xABCD1234, 8, 1), False)

    def test_another_batch_or_step_is_not_accepted(self):
        leader, follower = self.box()
        leader.publish(1, 7, True)
        for tag, step in ((2, 7), (1, 8)):
            expect("STEP_BOX_TIMEOUT", lambda: follower.wait(tag, step, 0.02))

    def test_waiter_sees_a_later_publish(self):
        leader, follower = self.box()
        timer = threading.Timer(0.01, leader.publish, (5, 1, True))
        timer.start()
        self.assertIs(follower.wait(5, 1, 2), True)
        timer.join()


class RpcInlineMethods(unittest.TestCase):
    def test_only_bounded_owner_lock_methods_are_listed(self):
        self.assertEqual(Sim(run_steps=4).controller.rpc_inline_methods(), ())
        controller = Sim(run_steps=4, rpc_inline_fast_methods=True).controller
        listed = set(controller.rpc_inline_methods())
        self.assertLessEqual(listed, set(controller.rpc_methods()))
        for name in ("poll_commands_wait", "rank_report", "rank_reports", "decode_step", "decode_resume"):
            self.assertIn("controller_" + name, listed)
        for name in ("register_rank", "prepare_batch", "start_batch", "snapshot", "begin_drain",
                     "close_ready", "metrics", "request_capacity"):
            self.assertNotIn("controller_" + name, listed)


class EventPumpLock(unittest.TestCase):
    def test_pump_delivers_while_the_owner_lock_is_held(self):
        import threading
        sim = Sim(run_steps=4, event_pump_thread=True)
        seen, arrived = [], threading.Event()

        def apply(event):
            seen.append(event.kind)
            arrived.set()
        driver = ControllerDriver(sim.controller, apply, schedule=False)
        driver.start()
        try:
            sim.submit("a", "r", max_new=5)
            with sim.controller.lock:  # e.g. a GPU_DONE handler still running
                sim.prefill("a", "r", 11)
                self.assertTrue(arrived.wait(5))  # TOKEN delivered before the lock is released
            self.assertIn("TOKEN", seen)
        finally:
            driver.stop()


if __name__ == "__main__":
    unittest.main()
