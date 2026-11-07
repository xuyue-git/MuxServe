#!/usr/bin/env python3
"""Compare paired-point results: original MuxServe and/or SGLang backend runs.

    python benchmark/compare_results.py --original ORIGINAL_result.json \
        --sglang perf-run/bench.json [--sglang other-run/bench.json ...]

--original reads original MuxServe's fixed-workload result.json
(observations_raw.records: admitted_at, token_times, ttft_s, tpot_s);
--sglang reads tools/bench_fixed.py output (records: sent, token_times, done).
Both are reduced to the same per-model numbers over the measured requests:
throughput (output tokens / window), mean TTFT and TPOT (first to last token /
(n - 1)), cycle (start of one request to the next of the same model), tail
(last token to that model's next request start; for SGLang this includes
waiting for the stream to end), and the p99 token gap. Throughput of the paired point is set by the slowest model's
cycle. Clocks differ: original is in process, SGLang at the HTTP client.
"""
from __future__ import annotations

import argparse
import json
import statistics


def _pct(values, q):
    values = sorted(values)
    return values[min(len(values) - 1, int(q * (len(values) - 1)))] if values else None


def _model_stats(rows):
    """rows: dicts with start, token_times (monotonic seconds), sorted by start."""
    rows = sorted(rows, key=lambda r: r["start"])
    ttft = [r["token_times"][0] - r["start"] for r in rows]
    tpot = [(r["token_times"][-1] - r["token_times"][0]) / (len(r["token_times"]) - 1) for r in rows]
    cycles = [b["start"] - a["start"] for a, b in zip(rows, rows[1:])]
    tails = [b["start"] - a["token_times"][-1] for a, b in zip(rows, rows[1:])]
    gaps = [t2 - t1 for r in rows for t1, t2 in zip(r["token_times"], r["token_times"][1:])]
    tokens = sum(len(r["token_times"]) for r in rows)
    window = max(r["token_times"][-1] for r in rows) - min(r["start"] for r in rows)
    return {"requests": len(rows), "output_tokens": tokens, "window_s": window,
            "tok_s": tokens / window, "ttft_ms": statistics.fmean(ttft) * 1e3,
            "tpot_ms": statistics.fmean(tpot) * 1e3,
            "cycle_ms": statistics.fmean(cycles) * 1e3 if cycles else None,
            "tail_ms": statistics.fmean(tails) * 1e3 if tails else None,
            "gap_p99_ms": _pct(gaps, 0.99) * 1e3}


def _overall(per_model_rows):
    rows = [r for items in per_model_rows.values() for r in items]
    tokens = sum(len(r["token_times"]) for r in rows)
    window = max(r["token_times"][-1] for r in rows) - min(r["start"] for r in rows)
    ttft = [r["token_times"][0] - r["start"] for r in rows]
    tpot = [(r["token_times"][-1] - r["token_times"][0]) / (len(r["token_times"]) - 1) for r in rows]
    return {"requests": len(rows), "output_tokens": tokens, "window_s": window, "tok_s": tokens / window,
            "ttft_ms": statistics.fmean(ttft) * 1e3, "tpot_ms": statistics.fmean(tpot) * 1e3}


def load_original(path):
    data = json.load(open(path, encoding="utf-8"))
    by_model = {}
    for record in data["observations_raw"]["records"]:
        if record["phase"] != "measured":
            continue
        name = record["model"].replace("-", "")  # llama-7b -> llama7b, the backend's model IDs
        by_model.setdefault(name, []).append({"start": record["admitted_at"],
                                              "token_times": record["token_times"]})
    return {"label": "original", "clock": data["observations_raw"].get("observation_boundary", "in process"),
            "rows": by_model, "reference_match": None}


def load_sglang(path):
    data = json.load(open(path, encoding="utf-8"))
    by_model = {model: [{"start": r["sent"], "token_times": r["token_times"]}
                        for r in records if not r.get("error") and len(r["token_times"]) > 1]
                for model, records in data["records"].items()}
    return {"label": "sglang:" + str(path), "clock": "HTTP client send to streamed token",
            "rows": by_model, "reference_match": data.get("reference_match")}


def summarize(run):
    return {"label": run["label"], "clock": run["clock"], "reference_match": run["reference_match"],
            "overall": _overall(run["rows"]),
            "models": {model: _model_stats(rows) for model, rows in sorted(run["rows"].items())}}


def _fmt(value, digits=1):
    return "-" if value is None else f"{value:.{digits}f}"


def render(summaries):
    lines = []
    for s in summaries:
        o = s["overall"]
        lines.append(f"{s['label']}: {o['tok_s']:.2f} tok/s, window {o['window_s']:.1f} s, "
                     f"TTFT {o['ttft_ms']:.1f} ms, TPOT {o['tpot_ms']:.2f} ms ({s['clock']})"
                     + ("" if s["reference_match"] is None else f", outputs match original: {s['reference_match']}"))
        for model, m in s["models"].items():
            lines.append(f"  {model:9s} cycle {_fmt(m['cycle_ms'], 0):>6s}  TTFT {_fmt(m['ttft_ms']):>6s}  "
                         f"TPOT {_fmt(m['tpot_ms'], 2):>6s}  tail {_fmt(m['tail_ms']):>6s}  "
                         f"gap p99 {_fmt(m['gap_p99_ms']):>6s} ms")
    base = summaries[0]
    for s in summaries[1:]:
        lines.append(f"{s['label']} vs {base['label']}: throughput "
                     f"{100 * s['overall']['tok_s'] / base['overall']['tok_s']:.1f}%")
        for model in s["models"]:
            if model in base["models"] and base["models"][model]["cycle_ms"] and s["models"][model]["cycle_ms"]:
                a, b = base["models"][model], s["models"][model]
                lines.append(f"  {model:9s} cycle {b['cycle_ms'] - a['cycle_ms']:+.0f} ms = TTFT "
                             f"{b['ttft_ms'] - a['ttft_ms']:+.1f}, TPOT x(n-1) "
                             f"{(b['tpot_ms'] - a['tpot_ms']) * (s['models'][model]['output_tokens'] / s['models'][model]['requests'] - 1):+.0f}, "
                             f"tail {b['tail_ms'] - a['tail_ms']:+.1f} ms")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--original", help="original MuxServe fixed-workload result.json")
    parser.add_argument("--sglang", action="append", default=[], help="bench_fixed.py output JSON (repeatable)")
    parser.add_argument("--json", help="also write the summaries here")
    args = parser.parse_args()
    runs = ([load_original(args.original)] if args.original else []) + [load_sglang(p) for p in args.sglang]
    if not runs:
        parser.error("give --original and/or --sglang")
    summaries = [summarize(run) for run in runs]
    print(render(summaries))
    if args.json:
        with open(args.json, "w", encoding="utf-8") as stream:
            json.dump(summaries, stream, indent=1)


if __name__ == "__main__":
    main()
