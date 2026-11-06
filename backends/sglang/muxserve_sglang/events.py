"""Bounded runtime outbox and contiguous scheduler acknowledgements."""
from __future__ import annotations

from collections import OrderedDict
from copy import deepcopy
from dataclasses import asdict
from threading import RLock
from typing import Callable

from .contracts import IntegrationError, RuntimeEvent


class RuntimeOutbox:
    def __init__(self, group_id: str, model_id: str, maximum_pending: int = 1024):
        if type(maximum_pending) is not int or not 8 <= maximum_pending <= 1024:
            raise IntegrationError("INVALID_ARGUMENT", "Event window must be 8..1024")
        self.group_id = group_id
        self.model_id = model_id
        self.maximum_pending = maximum_pending
        self.next_seq = 1
        self.acked_seq = 0
        self.pending: OrderedDict[int, RuntimeEvent] = OrderedDict()
        self.lock = RLock()
        self.reserved: dict[str, int] = {}

    def reserve(self, owner: str, count: int) -> None:
        with self.lock:
            if type(count) is not int or count < 0 or not owner:
                raise IntegrationError("INVALID_ARGUMENT", "Invalid event reservation")
            if owner in self.reserved:
                if self.reserved[owner] != count:
                    raise IntegrationError("IDENTITY_CONFLICT", "Event reservation changed")
                return
            if len(self.pending) + sum(self.reserved.values()) + count > self.maximum_pending:
                raise IntegrationError("EVENT_BACKPRESSURE", "No guaranteed completion slots")
            self.reserved[owner] = count

    def release(self, owner: str) -> None:
        with self.lock:
            self.reserved.pop(owner, None)

    @property
    def grants_allowed(self) -> bool:
        with self.lock:
            return len(self.pending) + sum(self.reserved.values()) < self.maximum_pending

    def owner_can_complete(self, owner: str) -> bool:
        """A prepared batch may start using slots it already guaranteed."""
        with self.lock:
            return self.reserved.get(owner, 0) > 0

    def emit(self, kind: str, *, request_id=None, attempt=None, batch_id=None,
             payload=None, owner: str | None = None) -> RuntimeEvent:
        payload = deepcopy(dict(payload or {}))
        if len(payload.get("token_ids", ())) > 256:
            raise IntegrationError("EVENT_TOO_LARGE", "Token payload exceeds 256 tokens")
        with self.lock:
            if owner is not None and self.reserved.get(owner, 0) < 1:
                raise IntegrationError("EVENT_BACKPRESSURE", "Missing reserved event slot")
            if owner is None and len(self.pending) + sum(self.reserved.values()) >= self.maximum_pending:
                raise IntegrationError("EVENT_BACKPRESSURE", "Runtime event outbox is full")
            event = RuntimeEvent(self.group_id, self.next_seq, kind, self.model_id,
                                 request_id, attempt, batch_id, payload)
            self.pending[event.event_seq] = event
            self.next_seq += 1
            if owner is not None:
                self.reserved[owner] -= 1
            return deepcopy(event)

    def ack(self, sequence: int) -> None:
        with self.lock:
            if type(sequence) is not int or sequence < self.acked_seq or sequence >= self.next_seq:
                raise IntegrationError("INVALID_ACK", "Invalid contiguous event acknowledgement")
            for number in list(self.pending):
                if number <= sequence:
                    del self.pending[number]
            self.acked_seq = sequence

    def replay(self, after: int = 0) -> tuple[RuntimeEvent, ...]:
        with self.lock:
            if type(after) is not int or after < self.acked_seq or after >= self.next_seq:
                raise IntegrationError("EVENT_GAP", "Requested events precede acknowledged history")
            return deepcopy(tuple(event for number, event in self.pending.items() if number > after))


class EventReceiver:
    def __init__(self, group_id: str, apply: Callable[[RuntimeEvent], None]):
        self.group_id = group_id
        self.apply = apply
        self.last_seq = 0
        self.gaps: dict[int, RuntimeEvent] = {}
        self.last_fingerprints: OrderedDict[int, dict] = OrderedDict()

    def receive(self, event: RuntimeEvent) -> int:
        if (not isinstance(event, RuntimeEvent) or event.group_id != self.group_id
                or type(event.event_seq) is not int or event.event_seq < 1
                or len(event.payload.get("token_ids", ())) > 256):
            raise IntegrationError("STALE_GROUP", "Runtime event identity rejected")
        fingerprint = asdict(event)
        if event.event_seq <= self.last_seq:
            prior = self.last_fingerprints.get(event.event_seq)
            if prior is None:
                raise IntegrationError("EVENT_GAP", "Replay is older than retained comparison window")
            if prior is not None and prior != fingerprint:
                raise IntegrationError("EVENT_CONFLICT", "Replayed sequence changed contents")
            return self.last_seq
        prior = self.gaps.get(event.event_seq)
        if prior is not None and prior != event:
            raise IntegrationError("EVENT_CONFLICT", "Buffered sequence changed contents")
        if event.event_seq - self.last_seq > 1024 or (prior is None and len(self.gaps) >= 1024):
            raise IntegrationError("EVENT_BACKPRESSURE", "Event gap exceeds replay window")
        self.gaps[event.event_seq] = deepcopy(event)
        while self.last_seq + 1 in self.gaps:
            current = self.gaps[self.last_seq + 1]
            self.apply(current)
            del self.gaps[current.event_seq]
            self.last_seq = current.event_seq
            self.last_fingerprints[current.event_seq] = asdict(current)
            if len(self.last_fingerprints) > 1024:
                self.last_fingerprints.popitem(last=False)
        return self.last_seq
