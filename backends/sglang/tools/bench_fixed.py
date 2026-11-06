"""Fixed 512/128 closed-loop benchmark against a running launcher gateway.

Workload matches the earlier paired point: one synthetic input sequence,
greedy decoding with ignore_eos to a fixed output length, at most one request
in flight per model (closed loop, total concurrency = number of models).

Timing boundary: client side, from sending the HTTP request to receiving each
streamed token. It includes gateway/SSE overhead, so it is slightly
pessimistic relative to the in-process boundary of the earlier paired run.

Standard library only. Example:
  python tools/bench_fixed.py --port 18080 --models llama7b,llama13b,llama30b \
      --requests 64 --warmup 4 --out result.json
  python tools/bench_fixed.py ... --reference old.json   # compare output tokens
"""
from __future__ import annotations

import argparse
import http.client
import json
import random
import statistics
import sys
import threading
import time
import uuid


def synthetic_input(length: int, seed: int) -> list[int]:
    rng = random.Random(seed)
    # LLaMA-1 vocabulary is 32000; stay clear of special/byte tokens.
    return [1] + [rng.randrange(1000, 30000) for _ in range(length - 1)]


def get_json(port: int, path: str, timeout: float = 10.0):
    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=timeout)
    try:
        connection.request("GET", path)
        response = connection.getresponse()
        body = response.read()
        return response.status, (json.loads(body) if body else None)
    finally:
        connection.close()


def wait_ready(port: int, timeout_s: float) -> None:
    deadline = time.monotonic() + timeout_s
    last = None
    while time.monotonic() < deadline:
        try:
            status, body = get_json(port, "/health", 5)
            if status == 200 and body and body.get("ready"):
                return
            last = status
        except OSError as exc:
            last = repr(exc)
        time.sleep(2)
    raise SystemExit(f"Gateway not ready after {timeout_s:.0f}s (last: {last})")


def generate(port: int, model: str, input_ids: list[int], output_len: int, timeout_s: float) -> dict:
    body = json.dumps({"rid": f"bench-{uuid.uuid4().hex}", "input_ids": input_ids, "stream": True,
                       "sampling_params": {"max_new_tokens": output_len, "ignore_eos": True,
                                           "temperature": 0, "top_k": 1, "top_p": 1,
                                           "stream_interval": 1}}).encode()
    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=timeout_s)
    sent = time.monotonic()
    token_times: list[float] = []
    output_ids: list[int] = []
    error = None
    try:
        connection.request("POST", f"/models/{model}/generate", body,
                           {"Content-Type": "application/json", "Content-Length": str(len(body))})
        response = connection.getresponse()
        if response.status != 200:
            return {"sent": sent, "token_times": [], "output_ids": [],
                    "error": f"HTTP {response.status}: {response.read()[:300]!r}"}
        while True:
            line = response.readline()
            if not line:
                break
            line = line.strip()
            if not line.startswith(b"data: "):
                continue
            data = line[6:]
            if data == b"[DONE]":
                break
            payload = json.loads(data)
            if "error" in payload:
                error = payload["error"]
                break
            now = time.monotonic()
            ids = payload["output_ids"]
            for _ in range(len(ids) - len(output_ids)):
                token_times.append(now)
            output_ids = list(ids)
    except (OSError, ValueError) as exc:
        error = repr(exc)
    finally:
        connection.close()
    if error is None and len(output_ids) != output_len:
        error = f"expected {output_len} tokens, got {len(output_ids)}"
    return {"sent": sent, "token_times": token_times, "output_ids": output_ids, "error": error}


def summarize(records: list[dict]) -> dict:
    ok = [r for r in records if not r["error"] and len(r["token_times"]) > 1]
    if not ok:
        return {"requests": len(records), "completed": 0}
    start = min(r["sent"] for r in ok)
    end = max(r["token_times"][-1] for r in ok)
    tokens = sum(len(r["output_ids"]) for r in ok)
    ttft = [r["token_times"][0] - r["sent"] for r in ok]
    tpot = [(r["token_times"][-1] - r["token_times"][0]) / (len(r["token_times"]) - 1) for r in ok]
    gaps = sorted(b - a for r in ok for a, b in zip(r["token_times"], r["token_times"][1:]))
    pick = lambda q: gaps[min(len(gaps) - 1, int(q * (len(gaps) - 1)))]
    return {"requests": len(records), "completed": len(ok), "output_tokens": tokens,
            "window_s": end - start, "output_tokens_per_second": tokens / (end - start),
            "mean_ttft_s": statistics.mean(ttft), "mean_tpot_s": statistics.mean(tpot),
            "token_gap_s": {"p50": pick(0.5), "p90": pick(0.9), "p99": pick(0.99), "max": gaps[-1]}}


def stage_breakdown(before: dict | None, after: dict | None, tpot: dict) -> dict:
    """Per-model cumulative controller totals (after - before), per batch/token."""
    if not before or not after:
        return {}
    result = {}
    for model, stages in after.get("feedback_totals", {}).items():
        out = {}
        for stage in ("P", "D"):
            a = stages[stage]["GENERATE"]
            b = before["feedback_totals"][model][stage]["GENERATE"]
            delta = {k: a[k] - b[k] for k in a}
            batches = max(1, delta["batches"])
            units = max(1, delta["output_tokens"]) if stage == "D" else batches
            out[stage] = {"batches": delta["batches"], "output_tokens": delta["output_tokens"],
                          "per_unit": "token" if stage == "D" else "batch",
                          "prepare_handshake_ms": delta["prepare_handshake_ms"] / units,
                          "execution_wall_ms": delta["execution_wall_ms"] / units,
                          "gpu_forward_ms": delta["gpu_forward_ms"] / units,
                          "pd_handoff_ms": delta["pd_handoff_ms"] / units}
        result[model] = out
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--port", type=int, default=18080)
    parser.add_argument("--models", default="llama7b,llama13b,llama30b")
    parser.add_argument("--requests", type=int, default=64, help="measured requests per model")
    parser.add_argument("--warmup", type=int, default=4, help="unmeasured requests per model first")
    parser.add_argument("--input-len", type=int, default=512)
    parser.add_argument("--output-len", type=int, default=128)
    parser.add_argument("--seed", type=int, default=20261107)
    parser.add_argument("--ready-timeout", type=float, default=3600)
    parser.add_argument("--request-timeout", type=float, default=600)
    parser.add_argument("--out", required=True)
    parser.add_argument("--reference", help="earlier result JSON whose output tokens must match")
    args = parser.parse_args()
    models = [m for m in args.models.split(",") if m]
    input_ids = synthetic_input(args.input_len, args.seed)
    wait_ready(args.port, args.ready_timeout)
    print(f"BENCH_READY models={models} input={args.input_len} output={args.output_len}", flush=True)

    def run(phase: str, count: int) -> dict:
        records = {m: [] for m in models}
        def loop(model):
            for index in range(count):
                record = generate(args.port, model, input_ids, args.output_len, args.request_timeout)
                records[model].append(record)
                if record["error"]:
                    print(f"BENCH_ERROR {phase} {model} #{index}: {record['error']}", flush=True)
                    return
                if (index + 1) % 8 == 0 or index + 1 == count:
                    print(f"BENCH_PROGRESS {phase} {model} {index + 1}/{count}", flush=True)
        threads = [threading.Thread(target=loop, args=(m,)) for m in models]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        return records

    if args.warmup:
        run("warmup", args.warmup)
    before = None
    try:
        before = get_json(args.port, "/metrics")[1]
    except OSError:
        pass
    measured = run("measured", args.requests)
    after = None
    try:
        status, after = get_json(args.port, "/metrics")
        if status != 200:
            after = None
    except OSError:
        pass
    flat = [r for m in models for r in measured[m]]
    summary = {"overall": summarize(flat), "models": {m: summarize(measured[m]) for m in models}}
    errors = [r["error"] for r in flat if r["error"]]
    # Same input + greedy: every request of one model must produce identical tokens.
    outputs = {m: [r["output_ids"] for r in measured[m] if not r["error"]] for m in models}
    consistency = {m: (len({tuple(o) for o in outs}) == 1 if outs else None) for m, outs in outputs.items()}
    reference_match = None
    if args.reference:
        with open(args.reference, encoding="utf-8") as stream:
            reference = json.load(stream)
        reference_match = {m: (outputs[m][0] == reference["first_output_ids"].get(m)) if outputs[m] else None
                           for m in models}
    result = {"schema": 1, "workload": {"input_len": args.input_len, "output_len": args.output_len,
                                        "seed": args.seed, "requests_per_model": args.requests,
                                        "warmup_per_model": args.warmup,
                                        "concurrency": "one in flight per model (closed loop)",
                                        "boundary": "client HTTP send to streamed token receipt"},
              "summary": summary, "stage_breakdown_ms": stage_breakdown(before, after, summary),
              "errors": errors, "outputs_consistent": consistency, "reference_match": reference_match,
              "first_output_ids": {m: (outputs[m][0] if outputs[m] else None) for m in models},
              "controller_metrics_before": before, "controller_metrics_after": after,
              "records": {m: [{k: r[k] for k in ("sent", "token_times", "error")} for r in measured[m]]
                          for m in models}}
    with open(args.out, "w", encoding="utf-8") as stream:
        json.dump(result, stream)
    print("BENCH_SUMMARY " + json.dumps({"overall": summary["overall"],
          "models": {m: {k: v for k, v in s.items() if k in ("output_tokens_per_second", "mean_ttft_s",
                                                              "mean_tpot_s", "window_s", "completed")}
                     for m, s in summary["models"].items()},
          "outputs_consistent": consistency, "reference_match": reference_match,
          "errors": len(errors)}, indent=1), flush=True)
    print("BENCH_STAGES " + json.dumps(result["stage_breakdown_ms"], indent=1), flush=True)
    if errors or not all(v for v in consistency.values() if v is not None) or (
            reference_match and not all(reference_match.values())):
        sys.exit(2)


if __name__ == "__main__":
    main()
