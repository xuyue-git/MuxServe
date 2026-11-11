"""Opt-in timing trace for diagnosis only.

Enabled when ``MXS_TRACE_DIR`` is set (the launcher strips ``MUXSERVE_*``
variables from children, so this name deliberately differs). Each process
appends JSON lines to ``trace-<pid>.jsonl``. Timestamps are
``time.monotonic()``, which is one system-wide clock on Linux, so events from
the owner and every rank can be merged. Recording never raises and never
changes control flow; with the variable unset every call is a no-op.
"""
from __future__ import annotations

import atexit
import json
import os
import threading
import time

_DIR = os.environ.get("MXS_TRACE_DIR") or None
_LOCK = threading.Lock()
_BUFFER: list[dict] = []
_STATE = {"file": None, "role": "", "flushed": 0.0, "pid": None}


def enabled() -> bool:
    return _DIR is not None


def set_role(role: str) -> None:
    _STATE["role"] = str(role)


def ev(name: str, batch=None, **fields) -> None:
    if _DIR is None:
        return
    try:
        record = {"t": time.monotonic(), "pid": os.getpid(), "role": _STATE["role"], "ev": name}
        if batch is not None:
            record["batch"] = batch
        record.update(fields)
        with _LOCK:
            _BUFFER.append(record)
            if len(_BUFFER) >= 512 or record["t"] - _STATE["flushed"] > 0.5:
                _flush_locked()
    except Exception:
        _BUFFER.clear()  # an unwritable trace drops records; never grows or raises


def _flush_locked() -> None:
    _STATE["flushed"] = time.monotonic()
    if not _BUFFER:
        return
    pid = os.getpid()
    if _STATE["file"] is None or _STATE["pid"] != pid:
        os.makedirs(_DIR, exist_ok=True)
        _STATE["file"] = open(os.path.join(_DIR, f"trace-{pid}.jsonl"), "a", encoding="utf-8")
        _STATE["pid"] = pid
    _STATE["file"].write("".join(json.dumps(r, separators=(",", ":"), default=str) + "\n"
                                 for r in _BUFFER))
    _STATE["file"].flush()
    _BUFFER.clear()


def flush() -> None:
    if _DIR is None:
        return
    try:
        with _LOCK:
            _flush_locked()
    except Exception:
        _BUFFER.clear()


atexit.register(flush)


def start_sampler(period_s: float = 1.0) -> None:
    """Diagnosis only: a daemon thread that records, once per period, this process's
    CPU time by thread (from /proc, threads grouped by name with digits removed) and
    how late its own 1 ms sleeps wake up, which measures how long a waking thread
    waits for the interpreter lock. No-op unless tracing; never raises."""
    if _DIR is None:
        return
    import re

    def stat_ticks(tid):
        with open(f"/proc/self/task/{tid}/stat", encoding="ascii") as handle:
            fields = handle.read().rsplit(")", 1)[1].split()
        return int(fields[11]) + int(fields[12])  # utime + stime

    def run():
        hertz = os.sysconf("SC_CLK_TCK")
        last = {}
        while True:
            try:
                lates = []
                end = time.monotonic() + period_s
                while time.monotonic() < end:
                    started = time.perf_counter()
                    time.sleep(0.001)
                    lates.append((time.perf_counter() - started) * 1000 - 1.0)
                names = {thread.native_id: thread.name for thread in threading.enumerate()}
                cpu = {}
                for tid in os.listdir("/proc/self/task"):
                    try:
                        ticks = stat_ticks(tid)
                    except (OSError, ValueError, IndexError):
                        continue
                    previous, last[tid] = last.get(tid), ticks
                    if previous is not None and ticks > previous:
                        name = re.sub(r"\d+", "#", names.get(int(tid), "native"))
                        cpu[name] = cpu.get(name, 0.0) + (ticks - previous) * 1000 / hertz
                lates.sort()
                ev("owner_sample", None, period_s=period_s, cpu_ms=cpu, late_n=len(lates),
                   late_mean=round(sum(lates) / len(lates), 3),
                   late_p90=round(lates[int(len(lates) * 0.9)], 3), late_max=round(lates[-1], 3))
            except Exception:
                time.sleep(period_s)

    threading.Thread(target=run, name="trace-sampler", daemon=True).start()
