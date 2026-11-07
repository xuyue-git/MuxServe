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
