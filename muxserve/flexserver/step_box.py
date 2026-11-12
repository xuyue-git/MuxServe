"""One shared word that tells a model's D ranks the leader's per-step decision.

With decode_leader_reports only D rank 0 reports a continuing decode step to
the owner. The other ranks read the outcome here instead of each making an RPC:
GO means the owner said to run the next step; ASK means anything else, and the
rank then reports to the owner as usual and follows its replayed decision.

The leader cannot publish step s + 1 before every rank has joined forward
s + 1 (the forward is collective), so a rank never misses its step's word. A
reader only accepts the exact expected value, so a torn word is simply not yet
a match. No Torch, so simulated ranks use the same code.
"""
from __future__ import annotations

import mmap
import os
import time

from muxserve.contracts import IntegrationError

_SIZE = 4096
_GO, _ASK = 1, 2
_SPIN_S = 0.002


class StepBox:
    def __init__(self, path: str):
        fd = os.open(path, os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600)
        try:
            os.ftruncate(fd, _SIZE)  # the same size for every rank: keeps the content
            self._map = mmap.mmap(fd, _SIZE)
        finally:
            os.close(fd)
        self._word = memoryview(self._map).cast("Q")

    @staticmethod
    def _value(tag: int, step: int, verdict: int) -> int:
        return (tag & 0xFFFFFFFF) << 32 | (step & 0xFFFFFF) << 8 | verdict

    def publish(self, tag: int, step: int, go: bool) -> None:
        self._word[0] = self._value(tag, step, _GO if go else _ASK)

    def wait(self, tag: int, step: int, timeout_s: float) -> bool:
        """True: run the next step. False: ask the owner."""
        go, ask = self._value(tag, step, _GO), self._value(tag, step, _ASK)
        started = time.monotonic()
        while True:
            value = self._word[0]
            if value == go:
                return True
            if value == ask:
                return False
            waited = time.monotonic() - started
            if waited >= timeout_s:
                raise IntegrationError("STEP_BOX_TIMEOUT", "Decode leader did not publish its step decision")
            if waited < _SPIN_S:
                os.sched_yield()
            else:
                time.sleep(0.0002)
