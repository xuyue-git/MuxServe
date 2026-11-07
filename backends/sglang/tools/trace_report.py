#!/usr/bin/env python3
"""Summarise an MXS_TRACE_DIR timing trace (diagnosis only).

    trace_report.py TRACE_DIR [--phases PHASES_FILE] [--gap-ms 150] [--json OUT]

PHASES_FILE lines are "<name> <start_monotonic> <end_monotonic>"; batches are
assigned to a phase by their prepare time. Every number is derived from
recorded timestamps; nothing is estimated.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import glob
import json
import os
import statistics


def pct(values, q):
    values = sorted(values)
    if not values:
        return None
    k = min(len(values) - 1, max(0, round(q * (len(values) - 1))))
    return values[k]


def summary(values):
    values = [v for v in values if v is not None]
    if not values:
        return None
    return {"n": len(values), "p50": round(pct(values, 0.5), 2), "p90": round(pct(values, 0.9), 2),
            "mean": round(statistics.fmean(values), 2), "max": round(max(values), 2)}


def load(directory):
    events = []
    for path in glob.glob(os.path.join(directory, "trace-*.jsonl")):
        with open(path, encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    try:
                        events.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass  # a process killed mid-write leaves one partial line
    events.sort(key=lambda e: e["t"])
    return events


def load_phases(path):
    phases = []
    if path and os.path.exists(path):
        for line in open(path, encoding="utf-8"):
            parts = line.split()
            if len(parts) == 3:
                phases.append((parts[0], float(parts[1]), float(parts[2])))
    return phases


def first(times):
    return min(times) if times else None


def last(times):
    return max(times) if times else None


class Batch:
    def __init__(self, batch_id):
        self.id = batch_id
        self.prep = None
        self.prep_call = None
        self.info = {}
        self.cmd = defaultdict(list)       # (kind, stage) -> times
        self.dlv = defaultdict(dict)       # (kind, stage) -> {rank: first time}
        self.rpt = defaultdict(dict)       # (kind, stage) -> {rank: time}
        self.ctl = {}                      # start/gpu_all/handoff_all/finish
        self.acc = defaultdict(dict)       # (kind, role) -> start
        self.acc_end = defaultdict(dict)   # (kind, role) -> end
        self.rank = defaultdict(dict)      # ev -> {role: time}
        self.dstep = defaultdict(dict)     # step -> {rank: time}


def build(events):
    batches = {}
    gid_model = {}
    submits = {}
    rpcs = defaultdict(list)
    failures = []
    gateway = defaultdict(dict)   # (model, req) -> {ev: t}

    def get(batch_id):
        if batch_id not in batches:
            batches[batch_id] = Batch(batch_id)
        return batches[batch_id]

    for e in events:
        name, t, b = e["ev"], e["t"], e.get("batch")
        if name == "rpc":
            rpcs[(e.get("role") or "owner/driver", e.get("m"))].append(e.get("ms", 0.0))
            continue
        if name in ("gw_admit", "gw_first_token", "gw_first_sse"):
            key = (e.get("m"), e.get("req"))
            gateway[key].setdefault(name, t)
            if name == "gw_admit" and e.get("recv") is not None:
                gateway[key].setdefault("gw_recv", e["recv"])
            continue
        if name == "prep_fail":
            failures.append((t, e.get("m"), e.get("s"), e.get("code")))
            continue
        if name == "submit":
            submits.setdefault((e.get("m"), e.get("req")), t)
            continue
        if b is None or name == "brk":
            continue
        batch = get(b)
        if name == "prep_call":
            batch.prep_call = batch.prep_call or t
        elif name == "prep":
            batch.prep = t
            batch.info = {"m": e.get("m"), "s": e.get("s"), "purpose": e.get("purpose"),
                          "steps": e.get("steps"), "reqs": e.get("reqs") or []}
            gid_model[e.get("gid")] = e.get("m")
        elif name == "cmd":
            batch.cmd[(e.get("kind"), e.get("s"))].append(t)
        elif name == "dlv":
            batch.dlv[(e.get("kind"), e.get("s"))].setdefault(e.get("r"), t)
        elif name == "rpt":
            batch.rpt[(e.get("kind"), e.get("s"))].setdefault(e.get("r"), t)
        elif name in ("start", "gpu_all", "handoff_all", "finish"):
            batch.ctl.setdefault(name, t)
        elif name == "acc":
            batch.acc[(e.get("kind"), e.get("role"))].setdefault("t", t)
        elif name == "acc_end":
            batch.acc_end[(e.get("kind"), e.get("role"))].setdefault("t", t)
        elif name == "dstep":
            batch.dstep[e.get("step")].setdefault(e.get("r"), t)
        else:
            batch.rank[name].setdefault(e.get("role"), t)
    return batches, gid_model, submits, rpcs, failures, gateway


def handler_ms(batch, kind, stage_letter):
    out = []
    for (k, role), start in batch.acc.items():
        if k != kind or not role or f"-{stage_letter}" not in role:
            continue
        end = batch.acc_end.get((k, role), {}).get("t")
        if end is not None:
            out.append((end - start["t"]) * 1000)
    return max(out) if out else None


def rank_last(batch, name, stage_letter):
    times = [t for role, t in batch.rank.get(name, {}).items() if role and f"-{stage_letter}" in role]
    return last(times)


def acc_end_last(batch, kind, stage_letter):
    times = [v["t"] for (k, role), v in batch.acc_end.items()
             if k == kind and role and f"-{stage_letter}" in role]
    return last(times)


P_MILESTONES = (
    ("prepare_delivered", "owner queued PREPARE -> last of 8 ranks received it"),
    ("all_ready", "-> last READY reached owner (rank prepare work + report RPC)"),
    ("start_called", "-> driver called start_batch"),
    ("start_delivered", "-> last P rank received START"),
    ("gpu_done", "-> last P GPU_DONE reached owner (includes prefill forward)"),
    ("handoff_delivered", "-> last D rank received HANDOFF"),
    ("handoff_accepted", "-> last D rank finished HANDOFF accept (broker lookup/ack RPCs)"),
    ("handoff_polled_ok", "-> last D rank saw every rank ACKed (broker poll)"),
    ("handoff_synced", "-> last D rank finished pop/all-reduce/prebuilt/sync"),
    ("handoff_all", "-> owner accepted last HANDOFF_READY"),
    ("finished", "-> P batch quiesced and released"),
)

D_MILESTONES = (
    ("prepare_delivered", "owner queued PREPARE -> last rank received it"),
    ("all_ready", "-> last READY reached owner"),
    ("start_called", "-> driver called start_batch"),
    ("start_delivered", "-> last D rank received START"),
    ("first_step", "-> first step reported by all ranks"),
    ("last_step", "-> last step reported by all ranks"),
    ("gpu_done", "-> last GPU_DONE reached owner"),
    ("finished", "-> D batch quiesced and released"),
)


def p_timeline(batch):
    m = {"prepare": batch.prep}
    m["prepare_delivered"] = last(list(batch.dlv.get(("PREPARE", "P"), {}).values()) +
                                  list(batch.dlv.get(("PREPARE", "D"), {}).values()))
    ready = {**{("P", r): t for r, t in batch.rpt.get(("READY", "P"), {}).items()},
             **{("D", r): t for r, t in batch.rpt.get(("READY", "D"), {}).items()}}
    m["all_ready"] = last(ready.values())
    m["last_ready_stage"] = max(ready, key=ready.get)[0] if ready else None
    m["start_called"] = batch.ctl.get("start")
    m["start_delivered"] = last(batch.dlv.get(("START", "P"), {}).values())
    m["gpu_done"] = batch.ctl.get("gpu_all")
    m["handoff_delivered"] = last(batch.dlv.get(("HANDOFF", "D"), {}).values())
    m["handoff_accepted"] = acc_end_last(batch, "HANDOFF", "D")
    m["handoff_polled_ok"] = rank_last(batch, "ho_ok", "D")
    m["handoff_synced"] = rank_last(batch, "ho_sync", "D")
    m["handoff_all"] = batch.ctl.get("handoff_all")
    m["finished"] = batch.ctl.get("finish")
    return m


def step_done(batch, ranks):
    out = {}
    for step, by_rank in batch.dstep.items():
        if len(by_rank) >= ranks:
            out[step] = max(by_rank.values())
    return out


def d_timeline(batch, ranks):
    m = {"prepare": batch.prep}
    m["prepare_delivered"] = last(list(batch.dlv.get(("PREPARE", "P"), {}).values()) +
                                  list(batch.dlv.get(("PREPARE", "D"), {}).values()))
    m["all_ready"] = last(list(batch.rpt.get(("READY", "P"), {}).values()) +
                          list(batch.rpt.get(("READY", "D"), {}).values()))
    m["start_called"] = batch.ctl.get("start")
    m["start_delivered"] = last(batch.dlv.get(("START", "D"), {}).values())
    steps = step_done(batch, ranks)
    m["first_step"] = steps.get(1)
    m["last_step"] = steps[max(steps)] if steps else None
    m["gpu_done"] = batch.ctl.get("gpu_all")
    m["finished"] = batch.ctl.get("finish")
    m["steps"] = steps
    return m


TRANSITION = (
    ("prev_finished", "last token of previous batch -> previous batch quiesced/released"),
    ("driver_prepare_call", "-> driver called prepare_batch for the next D batch"),
    ("prepare_lock_wait", "-> prepare_batch acquired the owner lock"),
    ("prepare_reserve", "-> KV reserve done (store.reserve_kv and its checks)"),
    ("prepare_batch", "-> prepare_batch returned (8 PREPARE commands queued)"),
    ("all_ready", "-> last READY reached owner"),
    ("start_called", "-> driver called start_batch"),
    ("start_delivered", "-> last D rank received START"),
    ("first_forward_done", "-> last D rank finished step-1 forward (run_batch..sync)"),
    ("first_step", "-> owner had step 1 from every D rank"),
)


def transition(prev, nxt, ranks):
    """Timeline from prev's last token to nxt's first step (same model)."""
    if prev.info["s"] == "P":
        last_token = prev.ctl.get("handoff_all")
    else:
        steps = step_done(prev, ranks)
        last_token = steps[max(steps)] if steps else None
    if last_token is None or nxt.prep is None:
        return None
    first = step_done(nxt, ranks).get(1)
    fwd = [t for role, t in nxt.rank.get("step_fwd", {}).items() if role and "-D" in role]
    timeline = {"prepare": last_token,
                "prev_finished": prev.ctl.get("finish"),
                "driver_prepare_call": nxt.prep_call,
                "prepare_lock_wait": nxt.rank.get("prep_lock", {}).get(""),
                "prepare_reserve": nxt.rank.get("prep_reserved", {}).get(""),
                "prepare_batch": nxt.prep,
                "all_ready": last(list(nxt.rpt.get(("READY", "P"), {}).values()) +
                                  list(nxt.rpt.get(("READY", "D"), {}).values())),
                "start_called": nxt.ctl.get("start"),
                "start_delivered": last(nxt.dlv.get(("START", "D"), {}).values()),
                "first_forward_done": last(fwd) if len(fwd) >= ranks else None,
                "first_step": first}
    out = segments(timeline, TRANSITION)
    if first is not None:
        out["total"] = (first - last_token) * 1000
    return out


def segments(timeline, milestones):
    out = {}
    previous = timeline.get("prepare")
    for key, _ in milestones:
        value = timeline.get(key)
        if value is not None and previous is not None:
            out[key] = (value - previous) * 1000
        if value is not None:
            previous = value
    return out


def phase_of(t, phases):
    for name, start, end in phases:
        if start <= t <= end:
            return name
    return "all" if not phases else None


def overlaps(a0, a1, b0, b1):
    return max(0.0, min(a1, b1) - max(a0, b0))


def report(directory, phases_path=None, gap_ms=150.0, ranks=4):
    events = load(directory)
    phases = load_phases(phases_path)
    batches, gid_model, submits, rpcs, failures, gateway = build(events)
    result = {"events": len(events), "phases": {}}
    phase_names = [p[0] for p in phases] or ["all"]
    for phase in phase_names:
        chosen = [b for b in batches.values() if b.prep is not None and b.info.get("purpose") != "REBUILD"
                  and phase_of(b.prep, phases) == phase]
        by_model = defaultdict(lambda: {"P": [], "D": []})
        for b in chosen:
            by_model[b.info["m"]][b.info["s"]].append(b)
        p_intervals = [(b.info["m"], b.prep, b.ctl.get("finish") or b.prep) for b in chosen
                       if b.info["s"] == "P"]
        phase_out = {}
        for model in sorted(by_model):
            entry = {}
            # ---- P path
            seg = defaultdict(list)
            last_ready = defaultdict(int)
            queue = []
            handlers = defaultdict(list)
            for b in by_model[model]["P"]:
                tl = p_timeline(b)
                for k, v in segments(tl, P_MILESTONES).items():
                    seg[k].append(v)
                if tl.get("finished") and b.prep:
                    seg["total_prepare_to_finish"].append((tl["finished"] - b.prep) * 1000)
                if tl.get("handoff_all") and b.prep:
                    seg["total_prepare_to_first_token_ready"].append((tl["handoff_all"] - b.prep) * 1000)
                if tl.get("last_ready_stage"):
                    last_ready[tl["last_ready_stage"]] += 1
                for req in b.info.get("reqs", []):
                    if (model, req) in submits:
                        queue.append((b.prep_call or b.prep) - submits[(model, req)])
                for kind, stage in (("PREPARE", "P"), ("PREPARE", "D"), ("START", "P"), ("HANDOFF", "D")):
                    v = handler_ms(b, kind, stage)
                    if v is not None:
                        handlers[f"{stage}_{kind}_handler"].append(v)
                for a, z, label in (("ho_ok", "ho_pop", "D_pop_transferred"),
                                    ("ho_pop", "ho_prebuilt", "D_prebuilt"),
                                    ("ho_prebuilt", "ho_sync", "D_cuda_sync")):
                    vals = [(b.rank[z][role] - b.rank[a][role]) * 1000
                            for role in b.rank.get(z, {}) if role in b.rank.get(a, {})]
                    if vals:
                        handlers[label].append(max(vals))
            ttft = defaultdict(list)
            for b in by_model[model]["P"]:
                for req in b.info.get("reqs", []):
                    g = gateway.get((model, req), {})
                    sub = submits.get((model, req))
                    points = [("gw_recv", g.get("gw_recv")), ("submit", sub),
                              ("p_prepare_call", b.prep_call), ("p_handoff_all", b.ctl.get("handoff_all")),
                              ("gw_first_token", g.get("gw_first_token")), ("gw_first_sse", g.get("gw_first_sse"))]
                    for (_, a), (label, z) in zip(points, points[1:]):
                        if a is not None and z is not None:
                            ttft[label].append((z - a) * 1000)
                    if g.get("gw_recv") is not None and g.get("gw_first_sse") is not None:
                        ttft["total_recv_to_first_sse"].append((g["gw_first_sse"] - g["gw_recv"]) * 1000)
            entry["ttft_breakdown_ms"] = {k: summary(v) for k, v in ttft.items()}
            entry["P_batches"] = len(by_model[model]["P"])
            entry["P_segments_ms"] = {k: summary(v) for k, v in seg.items()}
            entry["P_last_ready_stage"] = dict(last_ready)
            entry["P_submit_to_prepare_ms"] = summary([q * 1000 for q in queue])
            entry["P_rank_handlers_ms"] = {k: summary(v) for k, v in handlers.items()}
            # ---- D path
            dseg = defaultdict(list)
            step_ms = []
            for b in by_model[model]["D"]:
                tl = d_timeline(b, ranks)
                for k, v in segments(tl, D_MILESTONES).items():
                    dseg[k].append(v)
                steps = tl["steps"]
                ordered = sorted(steps)
                for a, z in zip(ordered, ordered[1:]):
                    step_ms.append((steps[z] - steps[a]) * 1000)
                dseg["steps_per_batch"].append(len(steps))
            entry["D_batches"] = len(by_model[model]["D"])
            entry["D_segments_ms"] = {k: summary(v) for k, v in dseg.items()}
            entry["D_in_run_step_ms"] = summary(step_ms)
            # ---- transitions into each D batch (closed loop, same model)
            ordered = sorted((b for b in by_model[model]["P"] + by_model[model]["D"]), key=lambda b: b.prep)
            trans = {"after_prefill": defaultdict(list), "between_runs": defaultdict(list)}
            fail_counts = {"after_prefill": defaultdict(int), "between_runs": defaultdict(int)}
            for prev, nxt in zip(ordered, ordered[1:]):
                if nxt.info["s"] != "D":
                    continue
                result_t = transition(prev, nxt, ranks)
                if result_t is None:
                    continue
                kind = "after_prefill" if prev.info["s"] == "P" else "between_runs"
                for key, value in result_t.items():
                    trans[kind][key].append(value)
                window_start = prev.ctl.get("finish") or prev.prep
                window_end = nxt.prep_call or nxt.prep
                for t, fm, fs, code in failures:
                    if fm == model and fs == "D" and window_start <= t <= window_end:
                        fail_counts[kind][code] += 1
            entry["transitions_ms"] = {kind: {k: summary(v) for k, v in seg_map.items()}
                                       for kind, seg_map in trans.items()}
            entry["transition_prepare_failures"] = {k: dict(v) for k, v in fail_counts.items()}
            # ---- token gaps (closed loop: one request in flight per model)
            tokens = []
            for b in by_model[model]["P"]:
                if b.ctl.get("handoff_all"):
                    tokens.append((b.ctl["handoff_all"], "P", b.id, 0))
            for b in by_model[model]["D"]:
                for step, t in step_done(b, ranks).items():
                    tokens.append((t, "D", b.id, step))
            tokens.sort()
            buckets = defaultdict(float)
            counts = defaultdict(int)
            big = []
            for (t0, k0, b0, s0), (t1, k1, b1, s1) in zip(tokens, tokens[1:]):
                if k1 == "P":
                    continue  # next request begins; not a decode gap
                gap = (t1 - t0) * 1000
                kind = ("in_run" if k0 == "D" and b0 == b1 else
                        "after_prefill" if k0 == "P" else "between_runs")
                buckets[kind] += gap
                counts[kind] += 1
                if gap >= gap_ms:
                    other = [(m, a, z) for m, a, z in p_intervals if m != model and overlaps(t0, t1, a, z) > 0]
                    covered = sum(overlaps(t0, t1, a, z) for _, a, z in other)
                    big.append({"gap_ms": round(gap, 1), "kind": kind,
                                "other_model_prefills": sorted({m for m, _, _ in other}),
                                "other_prefill_overlap_ms": round(min(covered, t1 - t0) * 1000, 1)})
            entry["decode_time_by_gap_kind_ms"] = {k: {"total": round(buckets[k], 1), "gaps": counts[k],
                                                      "mean": round(buckets[k] / counts[k], 2)}
                                                  for k in buckets}
            entry["big_gaps"] = {"threshold_ms": gap_ms, "count": len(big),
                                 "total_ms": round(sum(g["gap_ms"] for g in big), 1),
                                 "with_other_model_prefill": sum(1 for g in big if g["other_model_prefills"]),
                                 "by_kind": {k: sum(1 for g in big if g["kind"] == k)
                                             for k in ("in_run", "after_prefill", "between_runs")},
                                 "largest": sorted(big, key=lambda g: -g["gap_ms"])[:8]}
            phase_out[model] = entry
        result["phases"][phase] = phase_out
    rpc_out = {}
    for (role, method), values in sorted(rpcs.items()):
        kind = role.split("-")[-1][0] if "-" in role else "owner"
        key = f"{kind}:{method}"
        rpc_out.setdefault(key, []).extend(values)
    result["rpc_ms"] = {k: summary(v) for k, v in sorted(rpc_out.items())}
    return result


def render(result):
    lines = [f"events={result['events']}"]
    for phase, models in result["phases"].items():
        lines.append(f"\n===== phase {phase} =====")
        for model, e in models.items():
            lines.append(f"\n--- {model}: P batches={e['P_batches']} D batches={e['D_batches']}")
            lines.append(f"  submit->prepare_batch (driver queue) ms: {e['P_submit_to_prepare_ms']}")
            if e.get("ttft_breakdown_ms"):
                lines.append("  TTFT breakdown inside the server (p50 / p90 / mean ms):")
                for key, label in (("submit", "HTTP received -> controller submit_request entered"),
                                   ("p_prepare_call", "-> driver called prepare_batch for the P batch"),
                                   ("p_handoff_all", "-> P handoff done (first token confirmed at owner)"),
                                   ("gw_first_token", "-> gateway applied first TOKEN event (driver poll)"),
                                   ("gw_first_sse", "-> first SSE chunk written to the client"),
                                   ("total_recv_to_first_sse", "TOTAL HTTP received -> first SSE")):
                    s = e["ttft_breakdown_ms"].get(key)
                    if s:
                        lines.append(f"    {key:<36}{s['p50']:>9}{s['p90']:>9}{s['mean']:>9}   {label}")
            lines.append("  P path segments (p50 / p90 / mean ms):")
            for key, label in P_MILESTONES + (("total_prepare_to_first_token_ready", "TOTAL prepare -> HANDOFF done"),
                                               ("total_prepare_to_finish", "TOTAL prepare -> P released")):
                s = e["P_segments_ms"].get(key)
                if s:
                    lines.append(f"    {key:<36}{s['p50']:>9}{s['p90']:>9}{s['mean']:>9}   {label}")
            lines.append(f"  last READY came from stage: {e['P_last_ready_stage']}")
            lines.append("  rank-side handler time (max over ranks; p50 / p90 ms):")
            for key, s in e["P_rank_handlers_ms"].items():
                if s:
                    lines.append(f"    {key:<36}{s['p50']:>9}{s['p90']:>9}")
            lines.append("  D run segments (p50 / p90 / mean ms):")
            for key, label in D_MILESTONES + (("steps_per_batch", "steps per D batch"),):
                s = e["D_segments_ms"].get(key)
                if s:
                    lines.append(f"    {key:<36}{s['p50']:>9}{s['p90']:>9}{s['mean']:>9}   {label}")
            lines.append(f"  in-run step interval ms: {e['D_in_run_step_ms']}")
            for kind, seg_map in e.get("transitions_ms", {}).items():
                if not seg_map:
                    continue
                lines.append(f"  transition {kind} (last token -> next D first step; p50 / p90 / mean ms):")
                for key, label in TRANSITION + (("total", "TOTAL"),):
                    s = seg_map.get(key)
                    if s:
                        lines.append(f"    {key:<36}{s['p50']:>9}{s['p90']:>9}{s['mean']:>9}   {label}")
                lines.append(f"    D prepare_batch refusals in these windows: "
                             f"{e.get('transition_prepare_failures', {}).get(kind, {})}")
            lines.append(f"  decode time by gap kind: {e['decode_time_by_gap_kind_ms']}")
            g = e["big_gaps"]
            lines.append(f"  gaps >= {g['threshold_ms']} ms: count={g['count']} total={g['total_ms']} ms "
                         f"by_kind={g['by_kind']} overlapping_other_model_prefill={g['with_other_model_prefill']}")
            for item in g["largest"]:
                lines.append(f"    {item}")
    lines.append("\n===== RPC latency (client side, ms) =====")
    for key, s in result["rpc_ms"].items():
        if s:
            lines.append(f"  {key:<44} n={s['n']:<7} p50={s['p50']:<8} p90={s['p90']:<8} max={s['max']}")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("trace_dir")
    parser.add_argument("--phases")
    parser.add_argument("--gap-ms", type=float, default=150.0)
    parser.add_argument("--ranks", type=int, default=4)
    parser.add_argument("--json")
    args = parser.parse_args()
    result = report(args.trace_dir, args.phases, args.gap_ms, args.ranks)
    if args.json:
        with open(args.json, "w", encoding="utf-8") as handle:
            json.dump(result, handle, indent=1)
    print(render(result))


if __name__ == "__main__":
    main()
