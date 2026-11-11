"""End-to-end CPU control-plane check with simulated rank processes.

Real HeadwiseStore, MuxController, StorageRPCServer (persistent connections),
ControllerDriver, GatewayServer and tools/bench_fixed.py. Each P/D TP rank is
a separate process that speaks the controlled command protocol over the real
authenticated RPC. GPU work is a fixed sleep, so the measured token interval
minus that sleep is control-plane overhead on this machine. No GPU claims.

  python acceptance/control_plane_e2e_check.py [--run-steps 128] [--requests 4]
"""
from __future__ import annotations

import argparse
import json
import multiprocessing
import os
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import threading
import time
import zlib
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from muxserve_sglang.config import DeploymentConfig, ModelDeployment
from muxserve_sglang.contracts import IntegrationError

PREFILL_S, DECODE_S = 0.020, 0.005
MODELS = ("a", "b", "c")
TP = 4


def token_for(output_count: int) -> int:
    return 10 + (output_count * 7) % 50


def rank_main(address, key, group, model, stage, rank, pid_tag, stop, timing=None, final_step=False,
              register_ready=False, ack_in_poll=False, combined=False, start_reply=False,
              leader_reports=False, step_join=False, pipeline=False, barrier=None, leave=False,
              wait_commands=False, stage_in_done=False, handoff_local=False):
    """final_step: config final_step_in_gpu_done (a locally final run step is
    reported inside GPU_DONE). HANDOFF native_metadata and long-poll
    bootstrap_ready are used whenever the owner provides them. step_join: config
    decode_step_join (a D run serves prefill commands at its step boundaries and
    takes joined requests and new KV pages; every step must fit its mapping)."""
    prefill_s, decode_s = timing or (PREFILL_S, DECODE_S)
    from muxserve_sglang.ipc import StorageRPCClient
    rpc = StorageRPCClient(address, key, 30)
    pid = os.getpid()
    call = lambda name, *args: rpc.call("controller_" + name, group, *args)
    call("register_rank", stage, rank, pid, {"vocab_size": 32000, "eos_token_ids": (2,),
                                             "native_pd": True, "headwise_kv": True})
    box = None
    if leader_reports:
        from muxserve_sglang.step_box import StepBox
        box = StepBox(os.path.join(os.path.dirname(address), "stepbox-" + model))
    st = SimpleNamespace(report_seq=0, deferred=None, pending_acks=[], reply_commands=(), cursor=0,
                         epoch=None, backlog=[], prepared={}, waiting_bootstrap={}, waiting_handoff=None,
                         bootstrap_ready=None, broker_status=None, drained=False, local=False)

    def report(batch_id, kind, payload):
        if st.deferred is not None:
            pair = ((str(st.report_seq + 1), *st.deferred), (str(st.report_seq + 2), batch_id, kind, payload))
            st.deferred = None
            reply = call("rank_reports", stage, rank, pid, pair)
            st.report_seq += 2
        else:
            reply = call("rank_report", stage, rank, pid, str(st.report_seq + 1), batch_id, kind, payload)
            st.report_seq += 1
        assert reply["accepted"], reply
        st.reply_commands = reply.get("commands", ())

    def absorb():
        for held in st.reply_commands:
            if held["command_seq"] == (st.backlog[-1]["command_seq"] if st.backlog else st.cursor) + 1:
                st.backlog.append(held)
        st.reply_commands = ()

    def poll(wait_ms):
        after = st.backlog[-1]["command_seq"] if st.backlog else st.cursor
        acks = tuple(st.pending_acks)
        reply = call("poll_commands_wait", stage, rank, pid, after, st.cursor or None, wait_ms, st.epoch,
                     *((acks,) if acks else ()))
        del st.pending_acks[:len(acks)]
        st.epoch = reply["epoch"]
        if "bootstrap_ready" in reply:
            st.bootstrap_ready = {tuple(room) for room in reply["bootstrap_ready"]}
        if "broker_status" in reply:
            st.broker_status = dict(reply["broker_status"])
        st.backlog.extend(reply["commands"])

    def run_decode(command, prep):
        steps = prep.get("run_steps", 1)
        reports = None
        if pipeline and box is not None and rank == 0:
            from concurrent.futures import ThreadPoolExecutor
            reports = ThreadPoolExecutor(max_workers=1)
        in_flight, ahead = None, False
        start = time.monotonic()
        members = [{"request_id": i["request_id"], "attempt": i["attempt"], "base": len(i["output_token_ids"]),
                    "start": i["target_valid_tokens"] - 1, "max_new": i["max_new_tokens"],
                    "cap": i["reservation_view"]["capacity_tokens"]} for i in prep["requests"]]
        step = 1
        final = None
        digest = 0
        tag = zlib.crc32(command["batch_id"].encode())
        while True:
            assert all(m["start"] + step <= m["cap"] for m in members), "decode step outside its KV mapping"
            if barrier is not None:
                barrier.wait(60)  # a TP forward is collective: every D rank enters it together
            time.sleep(decode_s)
            results = [{"request_id": m["request_id"], "attempt": m["attempt"],
                        "token_id": token_for(m["base"] + step - 1),
                        "valid_tokens": m["start"] + step} for m in members]
            if steps == 1:
                break
            finished = any(m["base"] + step >= m["max_new"] for m in members)
            ending = all(m["base"] + step >= m["max_new"] for m in members) if leave else finished
            if box is not None:
                digest = zlib.crc32(",".join(str(item["token_id"]) for item in results).encode(), digest)
            if in_flight is not None:
                ahead = in_flight.result().get("ahead") is True
                in_flight = None
            if final_step and (ending or step >= steps):
                final = {"step": step, "results": results, "finished": ending}
                break
            continuing = not (ending or step >= steps)
            if (reports is not None and continuing and ahead and not finished
                    and all(m["start"] + step + 1 <= m["cap"] for m in members)):
                in_flight = reports.submit(call, "decode_step", rank, pid, command["batch_id"], step,
                                           results, ending, True)
                decision, ahead = {"continue": True}, False
                box.publish(tag, step, True)
            elif box is not None and continuing and rank != 0 and box.wait(tag, step, 30):
                decision = {"continue": True}
            else:
                decision = call("decode_step", rank, pid, command["batch_id"], step, results, ending)
                ahead = decision.get("ahead") is True
                if box is not None and continuing and rank == 0:
                    box.publish(tag, step, decision["continue"] is True and not decision.get("join")
                                and not decision.get("extend") and not decision.get("leave"))
            if step_join and wait_commands:
                decision = wait_at_boundary(command["batch_id"], step, decision)
            if step_join:
                while decision["continue"] in ("PARK", "SYNC"):
                    handoffs = tuple(decision.get("handoffs", ()))
                    serve_in_run(handoffs)
                    after = st.backlog[-1]["command_seq"] if st.backlog else st.cursor
                    decision = call("decode_resume", rank, pid, command["batch_id"], step, 100,
                                    after, st.epoch, handoffs)
            while decision["continue"] == "PARK":
                decision = call("decode_resume", rank, pid, command["batch_id"], step, 100)
            if ending or step >= steps or decision["continue"] is not True:
                break
            gone = set(decision.get("leave", ()))
            if gone:
                assert gone <= {m["request_id"] for m in members}
                members = [m for m in members if m["request_id"] not in gone]
                prep["ordered_request_ids"] = tuple(m["request_id"] for m in members)
            for item in decision.get("extend", ()):
                member = next(m for m in members if m["request_id"] == item["request_id"])
                assert (item["capacity_tokens"] > member["cap"] and item["first_page"] == member["cap"] // 16
                        and item["first_page"] + item["pages"].pages == item["capacity_tokens"] // 16)
                member["cap"] = item["capacity_tokens"]
            for item in decision.get("join", ()):
                assert all(m["request_id"] != item["request_id"] for m in members)
                assert set(item["view"]["block_tables"]) == {rank}
                members.append({"request_id": item["request_id"], "attempt": item["attempt"],
                                "base": len(item["output_token_ids"]) - step, "start": item["start_valid"],
                                "max_new": item["max_new_tokens"], "cap": item["view"]["capacity_tokens"]})
                prep["ordered_request_ids"] = tuple(prep["ordered_request_ids"]) + (item["request_id"],)
            if gone:
                serve_in_run(())  # the leavers' cleanup, queued with this decision
            step += 1
        payload = {"reservation_id": prep["reservation_id"], "results": results,
                   "duration_ms": (time.monotonic() - start) * 1000,
                   "gpu_forward_ms": decode_s * 1000 * step}
        if steps > 1:
            payload["steps"] = step
        if box is not None and steps > 1:
            payload["token_digest"] = digest
        if final is not None:
            payload["final_step"] = final
        return payload

    def handle(command):
        assert command["command_seq"] == st.cursor + 1
        kind = command["kind"]
        if kind == "START" and step_join:
            st.cursor = command["command_seq"]  # a run takes later commands itself
        if kind == "PREPARE":
            st.prepared[command["batch_id"]] = command
            if command["batch_stage"] == "P" and stage == "P":
                st.waiting_bootstrap[command["batch_id"]] = command
            else:
                ready_payload = {"reservation_id": command["reservation_id"]}
                if command["batch_stage"] == "P":  # D destination registration
                    destinations = []
                    for slot, item in enumerate(command["requests"]):
                        view = item["reservation_view"]
                        if register_ready:
                            destinations.append({"room": item["bootstrap_room"],
                                "request_id": item["request_id"], "attempt": item["attempt"],
                                "allocation_id": view["allocation_id"], "reserved_version": view["version"],
                                "target_valid_tokens": item["target_valid_tokens"],
                                "rank_block_table": item["rank_block_table"], "metadata_slot": slot,
                                "logical_token_count": item["target_valid_tokens"]})
                        else:
                            call("broker_register_destination", item["bootstrap_room"], rank, pid, model,
                                 item["request_id"], item["attempt"], view["allocation_id"], view["version"],
                                 item["target_valid_tokens"], item["rank_block_table"], slot,
                                 item["target_valid_tokens"])
                    if register_ready:
                        ready_payload["destinations"] = tuple(destinations)
                report(command["batch_id"], "READY", ready_payload)
        elif kind == "START":
            prep = st.prepared[command["batch_id"]]
            start = time.monotonic()
            if prep["batch_stage"] == "P":
                staged = []
                for item in prep["requests"]:
                    view = item["reservation_view"]
                    metadata = {
                        "bootstrap_room": item["bootstrap_room"], "allocation_id": view["allocation_id"],
                        "reserved_version": view["version"], "target_valid_tokens": item["target_valid_tokens"],
                        "rank": rank, "native_payload": {}}
                    if stage_in_done:
                        staged.append(metadata)
                    else:
                        call("broker_stage", item["bootstrap_room"], rank, pid, metadata)
                time.sleep(prefill_s)
                results = [{"request_id": i["request_id"], "attempt": i["attempt"],
                            "token_id": None if prep.get("purpose") == "REBUILD" else token_for(len(i["output_token_ids"])),
                            "valid_tokens": i["target_valid_tokens"]} for i in prep["requests"]]
                payload = {"reservation_id": prep["reservation_id"], "results": results,
                           "duration_ms": (time.monotonic() - start) * 1000, "gpu_forward_ms": prefill_s * 1000}
                if stage_in_done:
                    payload["staged"] = tuple(staged)
            else:
                payload = run_decode(command, prep)
            report(command["batch_id"], "GPU_DONE", payload)
        elif kind == "HANDOFF":
            for item in command["requests"]:
                if "native_metadata" not in item:
                    call("broker_lookup", item["bootstrap_room"], rank, pid, model, item["request_id"],
                         item["attempt"], item["allocation_id"], item["version"], item["valid_tokens"])
                if ack_in_poll:
                    st.pending_acks.append((item["bootstrap_room"], item["allocation_id"], item["version"],
                                            item["valid_tokens"], item["rank_block_table"]))
                else:
                    call("broker_ack", item["bootstrap_room"], rank, pid, item["allocation_id"],
                         item["version"], item["valid_tokens"], item["rank_block_table"])
            st.waiting_handoff = command
        elif kind in ("QUIESCE", "REVOKE"):
            prep = st.prepared.pop(command["batch_id"])
            report(command["batch_id"], "QUIESCED", {"reservation_id": command["reservation_id"],
                                                     "request_ids": tuple(prep["ordered_request_ids"])})
        elif kind == "CLEANUP":
            report(command["cleanup_id"], "QUIESCED", {"cleanup_id": command["cleanup_id"],
                   "request_id": command["request_id"], "attempt": command["attempt"]})
        elif kind == "DRAIN":
            st.drained = True
        st.cursor = max(st.cursor, command["command_seq"])
        absorb()

    def progress():
        # P bootstrap readiness and D handoff completion
        for batch_id, command in tuple(st.waiting_bootstrap.items()):
            ready = True
            for item in command["requests"]:
                view = item["reservation_view"]
                if st.bootstrap_ready is not None:
                    status = "READY" if (item["bootstrap_room"], view["allocation_id"], view["version"],
                                         item["target_valid_tokens"]) in st.bootstrap_ready else "PENDING"
                else:
                    status = call("broker_bootstrap", item["bootstrap_room"], rank, pid, view["allocation_id"],
                                  view["version"], item["target_valid_tokens"])["status"]
                ready = ready and status == "READY"
            if ready:
                del st.waiting_bootstrap[batch_id]
                report(batch_id, "READY", {"reservation_id": command["reservation_id"]})
        if st.waiting_handoff is not None and st.local:
            # handoff_local_success: the native pop's TP all-reduce (here the model's
            # barrier) waits for every rank's own validation; ACKs ride in the report.
            waiting = st.waiting_handoff
            barrier.wait(60)
            rooms = {i["bootstrap_room"] for i in waiting["requests"]}
            acks = tuple(a for a in st.pending_acks if a[0] in rooms)
            st.pending_acks[:] = [a for a in st.pending_acks if a[0] not in rooms]
            report(waiting["batch_id"], "HANDOFF_READY", {"requests": waiting["requests"], "acks": acks})
            from muxserve_sglang import trace as rank_trace
            rank_trace.ev("ho_local", waiting["batch_id"])
            st.waiting_handoff = None
        if st.waiting_handoff is not None:
            waiting = st.waiting_handoff
            if all((st.broker_status or {}).get(i["bootstrap_room"]) == "SUCCESS"
                   if st.broker_status is not None and i["bootstrap_room"] in st.broker_status else
                   call("broker_poll", i["bootstrap_room"], rank, pid, "D")["status"] == "SUCCESS"
                   for i in waiting["requests"]):
                if (combined and st.backlog and st.backlog[0]["kind"] == "PREPARE"
                        and st.backlog[0]["batch_stage"] == "D"):
                    st.deferred = (waiting["batch_id"], "HANDOFF_READY", {"requests": waiting["requests"]})
                else:
                    report(waiting["batch_id"], "HANDOFF_READY", {"requests": waiting["requests"]})
                st.waiting_handoff = None
        absorb()

    def accept_in_run(handoffs):
        while st.backlog and st.waiting_handoff is None:
            command = st.backlog[0]
            kind = command["kind"]
            prep = st.prepared.get(command.get("batch_id"))
            if not ((kind == "PREPARE" and command["batch_stage"] == "P")
                    or (kind == "REVOKE" and prep is not None and prep["batch_stage"] == "P")
                    or kind == "CLEANUP"
                    or (kind == "HANDOFF" and command["batch_id"] in handoffs)):
                break
            st.backlog.pop(0)
            handle(command)

    def take(reply):
        if "commands" in reply:
            st.epoch = reply["epoch"]
            st.broker_status = dict(reply["broker_status"])
            st.backlog.extend(reply["commands"])

    def wait_at_boundary(batch_id, step, decision):
        """decode_wait_commands: commands, broker state and ACKs ride on decode_resume."""
        while decision["continue"] in ("PARK", "SYNC"):
            take(decision)
            handoffs = tuple(decision.get("handoffs", ()))
            st.local = handoff_local
            try:
                accept_in_run(handoffs)
                progress()
            finally:
                st.local = False
            acks = tuple(st.pending_acks)
            after = st.backlog[-1]["command_seq"] if st.backlog else st.cursor
            decision = call("decode_resume", rank, pid, batch_id, step, 100, after, st.epoch, handoffs,
                            st.cursor or None, acks)
            del st.pending_acks[:len(acks)]
        take(decision)
        return decision

    def serve_in_run(handoffs):
        """Exactly the commands a real D rank takes at a run's step boundary."""
        for _ in range(2):  # the second poll delivers the ACKs a HANDOFF produced
            poll(0)
            while st.backlog and st.waiting_handoff is None:
                command = st.backlog[0]
                kind = command["kind"]
                prep = st.prepared.get(command.get("batch_id"))
                if not ((kind == "PREPARE" and command["batch_stage"] == "P")
                        or (kind == "REVOKE" and prep is not None and prep["batch_stage"] == "P")
                        or kind == "CLEANUP"
                        or (kind == "HANDOFF" and command["batch_id"] in handoffs)):
                    break
                st.backlog.pop(0)
                handle(command)
            if not st.pending_acks:
                break
        progress()

    wait_ms, skip_poll = 0, False
    while not st.drained and not stop.is_set():
        if not skip_poll:
            poll(wait_ms)
        skip_poll = False
        accepted = 0
        while st.backlog and st.waiting_handoff is None:
            handle(st.backlog.pop(0))
            accepted += 1
            if st.drained or st.waiting_handoff is not None:
                break
        progress()
        skip_poll = (combined or start_reply) and bool(st.backlog) and st.waiting_handoff is None
        wait_ms = 20 if st.pending_acks else 0 if accepted or (st.backlog and st.waiting_handoff is None) else (
            20 if (st.waiting_bootstrap or st.waiting_handoff is not None or st.prepared) else 200)
    if st.drained:
        call("poll_commands", stage, rank, pid, st.cursor, st.cursor)
    rpc.close()


def free_port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


class Tokenizer:
    def decode(self, tokens, skip_special_tokens=True):
        return " ".join(map(str, tokens))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-steps", type=int, default=128)
    parser.add_argument("--requests", type=int, default=3)
    parser.add_argument("--output-len", type=int, default=32)
    parser.add_argument("--input-len", type=int, default=64)
    parser.add_argument("--layers", type=int, default=2, help="e.g. 60 for LLaMA-30B")
    parser.add_argument("--heads", type=int, default=TP, help="total KV heads, e.g. 52 for LLaMA-30B")
    parser.add_argument("--pool-blocks", type=int, default=4096, help="KV blocks per rank")
    parser.add_argument("--bench-models", default=",".join(MODELS), help="models the bench drives")
    parser.add_argument("--profile", help="write owner-side cProfile stats of RPC handlers here")
    parser.add_argument("--park", action="store_true", help="config decode_run_park=true")
    parser.add_argument("--inline-start", action="store_true", help="config inline_start=true")
    parser.add_argument("--skip-p", action="store_true", help="config decode_skip_prefill_ranks=true")
    parser.add_argument("--early-token", action="store_true", help="config prefill_early_token=true")
    parser.add_argument("--cleanup-skip-p", action="store_true", help="config cleanup_skip_prefill_ranks=true")
    parser.add_argument("--cleanup-with-quiesce", action="store_true", help="config cleanup_with_quiesce=true")
    parser.add_argument("--handoff-prepare", action="store_true", help="config handoff_prepare_decode=true")
    parser.add_argument("--inline-metadata", action="store_true", help="config handoff_inline_metadata=true")
    parser.add_argument("--final-step", action="store_true", help="config final_step_in_gpu_done=true")
    parser.add_argument("--event-pump", action="store_true", help="config event_pump_thread=true")
    parser.add_argument("--bootstrap-poll", action="store_true", help="config bootstrap_in_poll=true")
    parser.add_argument("--inline-prepare", action="store_true", help="config inline_prepare=true")
    parser.add_argument("--own-rank-table", action="store_true", help="config prepare_own_rank_table=true")
    parser.add_argument("--register-ready", action="store_true", help="config register_in_ready=true")
    parser.add_argument("--broker-poll", action="store_true", help="config broker_status_in_poll=true")
    parser.add_argument("--early-terminal", action="store_true", help="config terminal_before_cleanup=true")
    parser.add_argument("--slot-at-cleanup", action="store_true", help="config slot_free_at_cleanup=true")
    parser.add_argument("--ack-in-poll", action="store_true", help="config handoff_ack_in_poll=true")
    parser.add_argument("--combined-report", action="store_true", help="config handoff_report_with_ready=true")
    parser.add_argument("--start-in-reply", action="store_true", help="config start_in_report_reply=true")
    parser.add_argument("--targeted-wake", action="store_true", help="config targeted_wake=true")
    parser.add_argument("--rpc-inline", action="store_true", help="config rpc_inline_fast_methods=true")
    parser.add_argument("--decode-leader", action="store_true", help="config decode_leader_reports=true")
    parser.add_argument("--prepare-after-ack", action="store_true", help="config handoff_prepare_after_ack=true")
    parser.add_argument("--inflight", type=int, default=1, help="config max_model_inflight and client load")
    parser.add_argument("--clients", type=int, default=0,
                        help="closed-loop clients per model instead of --inflight (the rest queue on open streams)")
    parser.add_argument("--run-merge", action="store_true", help="config decode_run_merge=true")
    parser.add_argument("--pair-hold-ms", type=int, default=0, help="config prefill_pair_hold_ms")
    parser.add_argument("--recovery-all-decode", action="store_true", help="config capacity_recovery_all_decode=true")
    parser.add_argument("--max-new", type=int, default=256, help="config max_new_tokens")
    parser.add_argument("--kv-trim", action="store_true", help="config decode_kv_trim=true")
    parser.add_argument("--step-join", action="store_true", help="config decode_step_join=true")
    parser.add_argument("--kv-per-page", action="store_true", help="config decode_kv_per_page=true")
    parser.add_argument("--step-pipeline", action="store_true", help="config decode_step_pipeline=true")
    parser.add_argument("--step-leave", action="store_true", help="config decode_step_leave=true")
    parser.add_argument("--kv-grow-pages", type=int, default=1, help="config decode_kv_grow_pages")
    parser.add_argument("--wait-commands", action="store_true", help="config decode_wait_commands=true")
    parser.add_argument("--start-unparked", action="store_true", help="config prefill_start_unparked=true")
    parser.add_argument("--handoff-local", action="store_true", help="config handoff_local_success=true")
    parser.add_argument("--overlap", action="store_true", help="config prefill_overlap_decode=true")
    parser.add_argument("--kv-early-free", action="store_true", help="config kv_release_at_finish=true")
    parser.add_argument("--gateway-proc", action="store_true", help="config gateway_process=true")
    parser.add_argument("--lean-events", action="store_true", help="config event_lean_copies=true")
    parser.add_argument("--driver-lean", action="store_true", help="config driver_lean_locking=true")
    parser.add_argument("--switch-us", type=int, default=0, help="config owner_switch_interval_us")
    parser.add_argument("--step-history", action="store_true", help="config policy_step_history=true")
    parser.add_argument("--adbs-log", action="store_true", help="config policy_log_adjustments=true")
    parser.add_argument("--stage-in-done", action="store_true", help="config prefill_stage_in_gpu_done=true")
    parser.add_argument("--driver-interval-ms", type=int, default=0, help="config driver_min_interval_ms")
    parser.add_argument("--prefill-ms", help="per-model simulated prefill ms, e.g. 90,105,140")
    parser.add_argument("--decode-ms", help="per-model simulated decode step ms, e.g. 13,17,30")
    args = parser.parse_args()
    from muxserve_sglang.controller import MuxController
    from muxserve_sglang.driver import ControllerDriver
    from muxserve_sglang.gateway import GatewayServer, GatewayState
    from muxserve_sglang.ipc import StorageRPCServer
    from muxserve_sglang.storage import HeadwiseStore

    port = free_port()
    deployments = tuple(ModelDeployment(m, "unused", "0" * 40, "unused", "0" * 40, args.layers, args.heads,
                                        128 * args.heads, 40,
                                        20100 + 10 * i, 20101 + 10 * i, 20102 + 10 * i)
                        for i, m in enumerate(MODELS))
    config = DeploymentConfig(deployments, tp_size=TP, kv_bytes_per_rank=args.pool_blocks * 8192,
                              controller_port=port, decode_run_steps=args.run_steps,
                              decode_run_park=args.park, inline_start=args.inline_start,
                              decode_skip_prefill_ranks=args.skip_p,
                              prefill_early_token=args.early_token,
                              cleanup_skip_prefill_ranks=args.cleanup_skip_p,
                              cleanup_with_quiesce=args.cleanup_with_quiesce,
                              handoff_prepare_decode=args.handoff_prepare,
                              handoff_inline_metadata=args.inline_metadata,
                              final_step_in_gpu_done=args.final_step,
                              event_pump_thread=args.event_pump, bootstrap_in_poll=args.bootstrap_poll,
                              inline_prepare=args.inline_prepare,
                              prepare_own_rank_table=args.own_rank_table,
                              register_in_ready=args.register_ready,
                              broker_status_in_poll=args.broker_poll,
                              terminal_before_cleanup=args.early_terminal,
                              slot_free_at_cleanup=args.slot_at_cleanup,
                              handoff_ack_in_poll=args.ack_in_poll,
                              handoff_report_with_ready=args.combined_report,
                              start_in_report_reply=args.start_in_reply,
                              targeted_wake=args.targeted_wake,
                              rpc_inline_fast_methods=args.rpc_inline,
                              decode_leader_reports=args.decode_leader,
                              handoff_prepare_after_ack=args.prepare_after_ack,
                              max_model_inflight=args.inflight, decode_run_merge=args.run_merge,
                              prefill_pair_hold_ms=args.pair_hold_ms,
                              capacity_recovery_all_decode=args.recovery_all_decode, max_new_tokens=args.max_new,
                              decode_kv_trim=args.kv_trim, decode_step_join=args.step_join,
                              decode_kv_per_page=args.kv_per_page,
                              decode_step_pipeline=args.step_pipeline, decode_step_leave=args.step_leave,
                              decode_kv_grow_pages=args.kv_grow_pages,
                              decode_wait_commands=args.wait_commands,
                              prefill_stage_in_gpu_done=args.stage_in_done,
                              driver_min_interval_ms=args.driver_interval_ms,
                              prefill_start_unparked=args.start_unparked,
                              handoff_local_success=args.handoff_local,
                              prefill_overlap_decode=args.overlap,
                              kv_release_at_finish=args.kv_early_free,
                              gateway_process=args.gateway_proc, event_lean_copies=args.lean_events,
                              driver_lean_locking=args.driver_lean,
                              owner_switch_interval_us=args.switch_us,
                              policy_step_history=args.step_history,
                              policy_log_adjustments=args.adbs_log)
    if args.switch_us:
        sys.setswitchinterval(args.switch_us / 1e6)  # as the launcher does in the owner
    store = HeadwiseStore({r: args.pool_blocks for r in range(TP)}, model_ids=MODELS)
    controller = MuxController(config, store)
    directory = tempfile.mkdtemp(prefix="mx-e2e-")
    os.chmod(directory, 0o700)
    address, key = str(Path(directory) / "store.sock"), os.urandom(32)
    owner = SimpleNamespace(export_pool=lambda *a: None, release_consumer=lambda *a: None,
                            owner_snapshot=lambda: {}, close=lambda: None)
    profiles, profile_lock, handler_ms = [], threading.Lock(), {}
    local = threading.local()
    def traced(name, method):
        def call(*a, **kwargs):
            started = time.perf_counter()
            try:
                return method(*a, **kwargs)
            except IntegrationError:
                raise
            except Exception:
                import traceback
                print("OWNER_INTERNAL_ERROR", name, traceback.format_exc(), flush=True)
                raise
            finally:
                with profile_lock:
                    entry = handler_ms.setdefault(name, [0, 0.0])
                    entry[0] += 1
                    entry[1] += (time.perf_counter() - started) * 1000
        return call
    methods = {name: traced(name, method) for name, method in controller.rpc_methods().items()}
    server = StorageRPCServer(store, owner, address, key, extra_methods=methods,
                              owner_close=owner.close, inline_methods=controller.rpc_inline_methods(),
                              eager_io=config.rpc_inline_fast_methods)
    if args.profile:
        import cProfile  # Python 3.12+: one interpreter-wide profiler covers every thread
        profiles.append(cProfile.Profile())
        profiles[0].enable()
    from muxserve_sglang import trace as owner_trace
    owner_trace.start_sampler()  # MXS_TRACE_DIR only: owner CPU by thread, as in the launcher
    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()
    context = multiprocessing.get_context("spawn")  # like the launcher; fork+threads can deadlock
    stop = context.Event()
    per_model = lambda text, default: ([float(x) / 1000 for x in text.split(",")] if text
                                       else [default] * len(MODELS))
    timing = dict(zip(MODELS, zip(per_model(args.prefill_ms, PREFILL_S), per_model(args.decode_ms, DECODE_S))))
    barriers = {m: context.Barrier(TP) for m in MODELS}
    processes = [context.Process(target=rank_main, args=(address, key, controller.group_id(m), m, stage, rank,
                                                         0, stop, timing[m], args.final_step,
                                                         args.register_ready, args.ack_in_poll,
                                                         args.combined_report, args.start_in_reply,
                                                         args.decode_leader, args.step_join,
                                                         args.step_pipeline,
                                                         barriers[m] if stage == "D" else None,
                                                         args.step_leave, args.wait_commands,
                                                         args.stage_in_done, args.handoff_local),
                                   daemon=True)
                 for m in MODELS for stage in ("P", "D") for rank in range(TP)]
    for process in processes:
        process.start()
    deadline = time.monotonic() + 30
    while not all(controller.snapshot()["groups"][m]["ready"] for m in MODELS):
        if time.monotonic() > deadline:
            raise SystemExit("ranks did not register")
        time.sleep(0.05)
    connections = max(32, 6 * args.inflight, 6 * args.clients)
    if args.gateway_proc:
        from muxserve_sglang.gateway_proc import BridgedGatewayState, GatewayBridge
        state = BridgedGatewayState(controller, {m: Tokenizer() for m in MODELS})
    else:
        state = GatewayState(controller, {m: Tokenizer() for m in MODELS})
    driver = ControllerDriver(controller, state.apply_event, state.maintain)
    if args.gateway_proc:
        gateway = GatewayBridge(state, context, port, connections, {m: Tokenizer() for m in MODELS},
                                {"queue_limit": config.queue_limit, "fingerprint": config.fingerprint})
    else:
        gateway = GatewayServer(state, port, maximum_connections=connections)
    driver.start()
    http_thread = threading.Thread(target=gateway.serve_forever, daemon=True)
    http_thread.start()
    if os.environ.get("E2E_DEBUG"):
        def monitor():
            while True:
                time.sleep(5)
                with controller.lock:
                    rows = {m: [(b.plan.stage, b.plan.steps, b.status, sorted(b.ready)[:2], len(b.ready))
                                for b in g.batches.values() if b.status not in ("DONE", "ABORTED")]
                            for m, g in controller.groups.items()}
                    reqs = {k: (r.state, len(r.output), r.pending_reason) for k, r in controller.requests.items()
                            if r.state not in ("FINISHED",)}
                    print("MON", rows, reqs, controller.credits.available, flush=True)
        threading.Thread(target=monitor, daemon=True).start()
    out = Path(directory) / "bench.json"
    bench = subprocess.run([sys.executable, str(ROOT / "tools/bench_fixed.py"), "--port", str(port),
                            "--models", args.bench_models, "--requests", str(args.requests), "--warmup", "1",
                            "--per-model", str(args.clients or args.inflight),
                            "--input-len", str(args.input_len), "--output-len", str(args.output_len),
                            "--ready-timeout", "30", "--request-timeout", "120", "--out", str(out)],
                           capture_output=True, text=True, timeout=600)
    result = json.loads(out.read_text()) if out.exists() else None
    # Graceful drain of every group, exactly like the supervisor's drain path.
    gateway.begin_drain()
    driver.pause()
    for m in MODELS:
        controller.begin_drain(controller.group_id(m))
    deadline = time.monotonic() + 30
    for process in processes:
        process.join(max(0.1, deadline - time.monotonic()))
    alive = [p for p in processes if p.is_alive()]
    stop.set()
    for p in alive:
        p.terminate()
    driver.stop()
    gateway.server_close()
    server.close()
    server_thread.join(5)
    if bench.returncode != 0 or result is None or alive:
        print(bench.stdout[-3000:], bench.stderr[-3000:])
        raise SystemExit(f"E2E_FAILED bench={bench.returncode} alive_ranks={len(alive)}")
    expected = [token_for(i) for i in range(args.output_len)]
    for m in args.bench_models.split(","):
        assert result["first_output_ids"][m] == expected, (m, result["first_output_ids"][m][:8])
    if args.profile:
        import pstats
        profiles[0].disable()
        stats = None
        for profile in profiles:
            if stats is None:
                stats = pstats.Stats(profile)
            else:
                stats.add(profile)
        if stats is not None:
            stats.dump_stats(args.profile)
        busy = {k: (v[0], round(v[1], 1)) for k, v in sorted(handler_ms.items(), key=lambda kv: -kv[1][1])
                if k != "poll_commands_wait"}
        print("OWNER_HANDLER_MS", busy)
    tpot = {m: round(s["mean_tpot_s"] * 1000, 2) for m, s in result["summary"]["models"].items()}
    print("E2E_PASS run_steps=%d park=%s inline_start=%s skip_p=%s early_token=%s "
          "cleanup_skip_p=%s cleanup_with_quiesce=%s six=%s tok_s=%.2f "
          "mean_tpot_ms=%s (simulated decode ms %s) ttft_ms=%s tail_ms=%s cycle_ms=%s inline_starts=%d" % (
        args.run_steps, args.park, args.inline_start, args.skip_p, args.early_token,
        args.cleanup_skip_p, args.cleanup_with_quiesce,
        "".join("1" if flag else "0" for flag in (args.handoff_prepare, args.inline_metadata, args.final_step,
                                                   args.event_pump, args.bootstrap_poll,
                                                   args.inline_prepare, args.own_rank_table,
                                                   args.register_ready, args.broker_poll,
                                                   args.early_terminal, args.slot_at_cleanup)),
        result["summary"]["overall"]["output_tokens_per_second"], tpot,
        {m: round(t[1] * 1000, 1) for m, t in timing.items()},
        {m: round(s["mean_ttft_s"] * 1000, 1) for m, s in result["summary"]["models"].items()},
        {m: round(s["mean_tail_s"] * 1000, 1) for m, s in result["summary"]["models"].items()},
        {m: round(s["mean_cycle_s"] * 1000, 1) for m, s in result["summary"]["models"].items()},
        controller.inline_start_count))


if __name__ == "__main__":
    main()
