"""Opt-in torch.profiler capture of a few decode-run steps (diagnosis only).

Enabled with ``MXS_PROFILE_DIR``. Selected ranks (``MXS_PROFILE_RANKS``,
default ``D0``) profile ``MXS_PROFILE_STEPS`` (default 10) consecutive decode
steps starting at each count in ``MXS_PROFILE_AT`` (default ``300,1600``) of
that rank's executed decode-run steps, and write a text/JSON summary:

* forward wall per step, measured both just before the window (no profiler)
  and inside it, around ``run_batch`` + ``process_batch_result`` + sync;
* GPU busy time per step (union of CUDA kernel intervals) and idle remainder;
* kernel time per category (attention, KV store, GEMM, all-reduce, norm/rope/
  activation, sampling, other) and kernel launches per step;
* top kernels and top CPU ops by self time.

Every failure is swallowed: profiling never changes execution or results.
"""
from __future__ import annotations

import json
import os
import re
import time

_DIR = os.environ.get("MXS_PROFILE_DIR") or None

_CATEGORIES = (
    ("memcpy", re.compile(r"^memcpy|^memset", re.I)),
    ("norm_rope_act", re.compile(r"rmsnorm|layernorm|_norm|rotary|rope|silu|gelu|act_and_mul|swiglu", re.I)),
    ("attention", re.compile(r"headwise_attention|single_query|flash_?attn|paged_attention|attention_kernel", re.I)),
    ("kv_store", re.compile(r"reshape_and_cache|headwise_reshape|store_kv|set_kv", re.I)),
    ("allreduce", re.compile(r"nccl|allreduce|all_reduce|allgather|all_gather|cross_device|custom_ar", re.I)),
    ("gemm", re.compile(r"gemm|gemv|cutlass|xmma|cublas|sm80_|sm90_|ampere_|matmul|splitk", re.I)),
    ("sampling", re.compile(r"argmax|topk|top_k|softmax|sample|sort|cumsum", re.I)),
)


def _category(name: str) -> str:
    for label, pattern in _CATEGORIES:
        if pattern.search(name):
            return label
    return "other"


# GPU-side user annotations (record_function ranges mirrored on the device
# timeline, e.g. "scheduler.run_batch", "step[DECODE bs=1]", "nccl:all_reduce")
# span real kernels; counting them would double-count the whole step.
_ANNOTATION = re.compile(r"^(scheduler\.|step\[|nccl:|ProfilerStep|forward_|## )")


def _is_annotation(event) -> bool:
    if getattr(event, "is_user_annotation", False):
        return True
    return bool(_ANNOTATION.match(getattr(event, "name", "") or ""))


def _union_us(intervals):
    total, end = 0.0, None
    for start, stop in sorted(intervals):
        if end is None or start > end:
            total += stop - start
            end = stop
        elif stop > end:
            total += stop - end
            end = stop
    return total


class StepProfiler:
    def __init__(self, model_id: str, stage: str, rank: int):
        self.label = f"{model_id}-{stage}{rank}"
        self.enabled = False
        if _DIR is None:
            return
        try:
            ranks = {item.strip() for item in os.environ.get("MXS_PROFILE_RANKS", "D0").split(",") if item.strip()}
            self.at = sorted({int(x) for x in os.environ.get("MXS_PROFILE_AT", "300,1600").split(",") if x.strip()})
            self.window = max(1, int(os.environ.get("MXS_PROFILE_STEPS", "10")))
            self.enabled = f"{stage}{rank}" in ranks
        except Exception:
            self.enabled = False
        self.count = 0
        self.profiler = None
        self.active_at = None
        self.walls = []
        self.baseline = []
        self.started = None

    # Called around each decode-run forward (run_batch .. sync).
    def before(self) -> None:
        if not self.enabled:
            return
        try:
            if self.profiler is None and self.count in self.at:
                import torch
                activities = [torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]
                self.profiler = torch.profiler.profile(activities=activities, record_shapes=False,
                                                       with_stack=False, profile_memory=False)
                self.profiler.start()
                self.active_at = self.count
                self.walls = []
        except Exception:
            self.profiler = None
        self.started = time.perf_counter()

    def after(self) -> None:
        if not self.enabled or self.started is None:
            return
        wall = (time.perf_counter() - self.started) * 1000
        self.started = None
        self.count += 1
        if self.profiler is None:
            # Unprofiled steps just before the next window are the baseline.
            upcoming = [a for a in self.at if a >= self.count]
            if upcoming and upcoming[0] - self.count < self.window:
                self.baseline.append(wall)
            return
        self.walls.append(wall)
        if len(self.walls) >= self.window:
            self._finish()

    def _finish(self) -> None:
        profiler, self.profiler = self.profiler, None
        try:
            profiler.stop()
            self._write(profiler)
        except Exception as exc:  # pragma: no cover - diagnosis only
            try:
                os.makedirs(_DIR, exist_ok=True)
                with open(os.path.join(_DIR, f"profile-{self.label}-at{self.active_at}.error"), "w") as handle:
                    handle.write(repr(exc))
            except Exception:
                pass
        self.baseline = []

    def _write(self, profiler) -> None:
        steps = len(self.walls)
        kernels, cpu_ops, intervals = {}, {}, []
        for event in profiler.events():
            device_type = str(getattr(event, "device_type", ""))
            if device_type.endswith("CUDA") and _is_annotation(event):
                continue
            if device_type.endswith("CUDA"):
                start = event.time_range.start
                stop = event.time_range.end
                duration = stop - start
                intervals.append((start, stop))
                entry = kernels.setdefault(event.name, [0, 0.0])
                entry[0] += 1
                entry[1] += duration
            else:
                self_cpu = getattr(event, "self_cpu_time_total", 0.0) or 0.0
                entry = cpu_ops.setdefault(event.name, [0, 0.0])
                entry[0] += 1
                entry[1] += self_cpu
        if not kernels:
            # Some profiler builds attach kernels only to the averaged table.
            for item in profiler.key_averages():
                total = (getattr(item, "self_device_time_total", None)
                         or getattr(item, "self_cuda_time_total", 0.0) or 0.0)
                if (total > 0 and str(getattr(item, "device_type", "")).endswith("CUDA")
                        and not _ANNOTATION.match(item.key or "")):
                    kernels[item.key] = [item.count, total]
            intervals = []
        busy_us = _union_us(intervals) if intervals else sum(v[1] for v in kernels.values())
        categories = {}
        for name, (count, total) in kernels.items():
            label = _category(name)
            slot = categories.setdefault(label, [0, 0.0])
            slot[0] += count
            slot[1] += total
        per_step = lambda us: round(us / 1000 / steps, 3)
        wall = sum(self.walls) / steps
        baseline = sum(self.baseline) / len(self.baseline) if self.baseline else None
        summary = {
            "rank": self.label, "profiled_from_step": self.active_at, "steps": steps,
            "forward_wall_ms_unprofiled": round(baseline, 3) if baseline is not None else None,
            "forward_wall_ms_profiled": round(wall, 3),
            "gpu_busy_ms_per_step": per_step(busy_us),
            "gpu_idle_ms_per_step": round(wall - busy_us / 1000 / steps, 3),
            "kernel_launches_per_step": round(sum(c for c, _ in kernels.values()) / steps, 1),
            "kernel_ms_per_step_by_category": {k: {"ms": per_step(v[1]), "launches": round(v[0] / steps, 1)}
                                               for k, v in sorted(categories.items(), key=lambda kv: -kv[1][1])},
            "top_kernels": [{"name": name[:120], "ms_per_step": per_step(total), "calls_per_step": round(count / steps, 1),
                             "category": _category(name)}
                            for name, (count, total) in sorted(kernels.items(), key=lambda kv: -kv[1][1])[:25]],
            "top_cpu_ops_self": [{"name": name[:120], "ms_per_step": per_step(total), "calls_per_step": round(count / steps, 1)}
                                 for name, (count, total) in sorted(cpu_ops.items(), key=lambda kv: -kv[1][1])[:25]],
        }
        os.makedirs(_DIR, exist_ok=True)
        base = os.path.join(_DIR, f"profile-{self.label}-at{self.active_at}")
        with open(base + ".json", "w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=1)
        with open(base + ".txt", "w", encoding="utf-8") as handle:
            handle.write(render(summary))
        if os.environ.get("MXS_PROFILE_CHROME"):
            profiler.export_chrome_trace(base + ".trace.json")


def render(summary: dict) -> str:
    lines = [f"rank={summary['rank']} steps={summary['steps']} from_step={summary['profiled_from_step']}",
             f"forward wall/step: unprofiled={summary['forward_wall_ms_unprofiled']} ms  "
             f"profiled={summary['forward_wall_ms_profiled']} ms",
             f"GPU busy/step={summary['gpu_busy_ms_per_step']} ms  idle/step={summary['gpu_idle_ms_per_step']} ms  "
             f"kernel launches/step={summary['kernel_launches_per_step']}",
             "kernel time per step by category:"]
    for label, item in summary["kernel_ms_per_step_by_category"].items():
        lines.append(f"  {label:<14}{item['ms']:>9.3f} ms  launches={item['launches']}")
    lines.append("top kernels (ms/step, calls/step):")
    for item in summary["top_kernels"]:
        lines.append(f"  {item['ms_per_step']:>8.3f} {item['calls_per_step']:>7}  [{item['category']}] {item['name']}")
    lines.append("top CPU ops by self time (ms/step, calls/step; profiler-inflated):")
    for item in summary["top_cpu_ops_self"]:
        lines.append(f"  {item['ms_per_step']:>8.3f} {item['calls_per_step']:>7}  {item['name']}")
    return "\n".join(lines) + "\n"
