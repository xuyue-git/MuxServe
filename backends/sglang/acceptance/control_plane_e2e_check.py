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


def rank_main(address, key, group, model, stage, rank, pid_tag, stop, timing=None, final_step=False):
    """final_step: config final_step_in_gpu_done (a locally final run step is
    reported inside GPU_DONE). HANDOFF native_metadata and long-poll
    bootstrap_ready are used whenever the owner provides them."""
    prefill_s, decode_s = timing or (PREFILL_S, DECODE_S)
    from muxserve_sglang.ipc import StorageRPCClient
    rpc = StorageRPCClient(address, key, 30)
    pid = os.getpid()
    call = lambda name, *args: rpc.call("controller_" + name, group, *args)
    call("register_rank", stage, rank, pid, {"vocab_size": 32000, "eos_token_ids": (2,),
                                             "native_pd": True, "headwise_kv": True})
    report_seq = 0
    def report(batch_id, kind, payload):
        nonlocal report_seq
        reply = call("rank_report", stage, rank, pid, str(report_seq + 1), batch_id, kind, payload)
        assert reply["accepted"], reply
        report_seq += 1
    cursor, epoch, wait_ms, backlog = 0, None, 0, []
    prepared, waiting_bootstrap, waiting_handoff = {}, {}, None
    bootstrap_ready = None
    drained = False
    while not drained and not stop.is_set():
        after = backlog[-1]["command_seq"] if backlog else cursor
        reply = call("poll_commands_wait", stage, rank, pid, after, cursor or None, wait_ms, epoch)
        epoch = reply["epoch"]
        if "bootstrap_ready" in reply:
            bootstrap_ready = {tuple(room) for room in reply["bootstrap_ready"]}
        backlog.extend(reply["commands"])
        accepted = 0
        while backlog and waiting_handoff is None:
            command = backlog.pop(0)
            assert command["command_seq"] == cursor + 1
            kind = command["kind"]
            if kind == "PREPARE":
                prepared[command["batch_id"]] = command
                if command["batch_stage"] == "P" and stage == "P":
                    waiting_bootstrap[command["batch_id"]] = command
                else:
                    if command["batch_stage"] == "P":  # D destination registration
                        for slot, item in enumerate(command["requests"]):
                            view = item["reservation_view"]
                            call("broker_register_destination", item["bootstrap_room"], rank, pid, model,
                                 item["request_id"], item["attempt"], view["allocation_id"], view["version"],
                                 item["target_valid_tokens"], item["rank_block_table"], slot,
                                 item["target_valid_tokens"])
                    report(command["batch_id"], "READY", {"reservation_id": command["reservation_id"]})
            elif kind == "START":
                prep = prepared[command["batch_id"]]
                start = time.monotonic()
                if prep["batch_stage"] == "P":
                    for item in prep["requests"]:
                        view = item["reservation_view"]
                        call("broker_stage", item["bootstrap_room"], rank, pid, {
                            "bootstrap_room": item["bootstrap_room"], "allocation_id": view["allocation_id"],
                            "reserved_version": view["version"], "target_valid_tokens": item["target_valid_tokens"],
                            "rank": rank, "native_payload": {}})
                    time.sleep(prefill_s)
                    results = [{"request_id": i["request_id"], "attempt": i["attempt"],
                                "token_id": token_for(len(i["output_token_ids"])),
                                "valid_tokens": i["target_valid_tokens"]} for i in prep["requests"]]
                    payload = {"reservation_id": prep["reservation_id"], "results": results,
                               "duration_ms": (time.monotonic() - start) * 1000, "gpu_forward_ms": prefill_s * 1000}
                else:
                    steps = prep.get("run_steps", 1)
                    step = 1
                    final = None
                    while True:
                        time.sleep(decode_s)
                        results = [{"request_id": i["request_id"], "attempt": i["attempt"],
                                    "token_id": token_for(len(i["output_token_ids"]) + step - 1),
                                    "valid_tokens": i["target_valid_tokens"] - 1 + step} for i in prep["requests"]]
                        if steps == 1:
                            break
                        finished = any(len(i["output_token_ids"]) + step >= i["max_new_tokens"]
                                       for i in prep["requests"])
                        if final_step and (finished or step >= steps):
                            final = {"step": step, "results": results, "finished": finished}
                            break
                        decision = call("decode_step", rank, pid, command["batch_id"], step, results, finished)
                        while decision["continue"] == "PARK":
                            decision = call("decode_resume", rank, pid, command["batch_id"], step, 100)
                        if finished or step >= steps or decision["continue"] is not True:
                            break
                        step += 1
                    payload = {"reservation_id": prep["reservation_id"], "results": results,
                               "duration_ms": (time.monotonic() - start) * 1000,
                               "gpu_forward_ms": decode_s * 1000 * step}
                    if steps > 1:
                        payload["steps"] = step
                    if final is not None:
                        payload["final_step"] = final
                report(command["batch_id"], "GPU_DONE", payload)
            elif kind == "HANDOFF":
                for item in command["requests"]:
                    if "native_metadata" not in item:
                        call("broker_lookup", item["bootstrap_room"], rank, pid, model, item["request_id"],
                             item["attempt"], item["allocation_id"], item["version"], item["valid_tokens"])
                    call("broker_ack", item["bootstrap_room"], rank, pid, item["allocation_id"],
                         item["version"], item["valid_tokens"], item["rank_block_table"])
                waiting_handoff = command
            elif kind in ("QUIESCE", "REVOKE"):
                prep = prepared.pop(command["batch_id"])
                report(command["batch_id"], "QUIESCED", {"reservation_id": command["reservation_id"],
                                                         "request_ids": tuple(prep["ordered_request_ids"])})
            elif kind == "CLEANUP":
                report(command["cleanup_id"], "QUIESCED", {"cleanup_id": command["cleanup_id"],
                       "request_id": command["request_id"], "attempt": command["attempt"]})
            elif kind == "DRAIN":
                drained = True
            cursor = command["command_seq"]
            accepted += 1
            if drained or waiting_handoff is not None:
                break
        # progress: P bootstrap readiness and D handoff completion
        for batch_id, command in tuple(waiting_bootstrap.items()):
            ready = True
            for item in command["requests"]:
                view = item["reservation_view"]
                if bootstrap_ready is not None:
                    status = "READY" if (item["bootstrap_room"], view["allocation_id"], view["version"],
                                         item["target_valid_tokens"]) in bootstrap_ready else "PENDING"
                else:
                    status = call("broker_bootstrap", item["bootstrap_room"], rank, pid, view["allocation_id"],
                                  view["version"], item["target_valid_tokens"])["status"]
                ready = ready and status == "READY"
            if ready:
                del waiting_bootstrap[batch_id]
                report(batch_id, "READY", {"reservation_id": command["reservation_id"]})
        if waiting_handoff is not None:
            if all(call("broker_poll", i["bootstrap_room"], rank, pid, "D")["status"] == "SUCCESS"
                   for i in waiting_handoff["requests"]):
                report(waiting_handoff["batch_id"], "HANDOFF_READY",
                       {"requests": waiting_handoff["requests"]})
                waiting_handoff = None
        wait_ms = 0 if accepted or (backlog and waiting_handoff is None) else (
            20 if (waiting_bootstrap or waiting_handoff is not None or prepared) else 200)
    if drained:
        call("poll_commands", stage, rank, pid, cursor, cursor)
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
                              prepare_own_rank_table=args.own_rank_table)
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
                              owner_close=owner.close)
    if args.profile:
        import cProfile  # Python 3.12+: one interpreter-wide profiler covers every thread
        profiles.append(cProfile.Profile())
        profiles[0].enable()
    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()
    context = multiprocessing.get_context("spawn")  # like the launcher; fork+threads can deadlock
    stop = context.Event()
    per_model = lambda text, default: ([float(x) / 1000 for x in text.split(",")] if text
                                       else [default] * len(MODELS))
    timing = dict(zip(MODELS, zip(per_model(args.prefill_ms, PREFILL_S), per_model(args.decode_ms, DECODE_S))))
    processes = [context.Process(target=rank_main, args=(address, key, controller.group_id(m), m, stage, rank,
                                                         0, stop, timing[m], args.final_step), daemon=True)
                 for m in MODELS for stage in ("P", "D") for rank in range(TP)]
    for process in processes:
        process.start()
    deadline = time.monotonic() + 30
    while not all(controller.snapshot()["groups"][m]["ready"] for m in MODELS):
        if time.monotonic() > deadline:
            raise SystemExit("ranks did not register")
        time.sleep(0.05)
    state = GatewayState(controller, {m: Tokenizer() for m in MODELS})
    driver = ControllerDriver(controller, state.apply_event, state.maintain)
    gateway = GatewayServer(state, port)
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
                                                   args.inline_prepare, args.own_rank_table)),
        result["summary"]["overall"]["output_tokens_per_second"], tpot,
        {m: round(t[1] * 1000, 1) for m, t in timing.items()},
        {m: round(s["mean_ttft_s"] * 1000, 1) for m, s in result["summary"]["models"].items()},
        {m: round(s["mean_tail_s"] * 1000, 1) for m, s in result["summary"]["models"].items()},
        {m: round(s["mean_cycle_s"] * 1000, 1) for m, s in result["summary"]["models"].items()},
        controller.inline_start_count))


if __name__ == "__main__":
    main()
