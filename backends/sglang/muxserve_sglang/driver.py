"""Independent event pump and external ADBS batch grants for the owner."""
from __future__ import annotations

import logging
from threading import Event, RLock, Thread
from uuid import uuid4

from .contracts import BatchPlan, IntegrationError, RequestRef
from .events import EventReceiver

logger = logging.getLogger(__name__)


class ControllerDriver:
    def __init__(self, controller, apply_event, maintain=lambda: None, *, schedule=True):
        self.controller = controller
        self.config = controller.config
        self.maintain = maintain
        self.schedule = schedule
        self.grant_lock = RLock()
        self.failure = None
        self.pending = set()
        self.receivers = {model.model_id: EventReceiver(controller.group_id(model.model_id), apply_event)
                          for model in self.config.models}
        groups = controller.snapshot()["groups"]
        for model_id, receiver in self.receivers.items():
            receiver.last_seq = groups[model_id]["acked_event_seq"]
        self.stop_event = Event()
        self.thread = Thread(target=self.run, name="external-adbs-driver", daemon=False)

    def start(self):
        self.thread.start()

    def run(self):
        try:
            while not self.stop_event.is_set():
                self.step()
                self.stop_event.wait(0.005)
        except Exception as exc:
            # No cleanup/rollback proof follows an unknown driver failure.
            self.failure = exc
            self.schedule = False
            logger.exception("Controller driver stopped; supervisor must retain producers")

    def step(self):
        proposals = self.controller.tick(include_adjustments=False)
        for model_id, receiver in self.receivers.items():
            group_id = receiver.group_id
            for event in self.controller.poll_events(group_id, receiver.last_seq):
                receiver.receive(event)
            if receiver.last_seq:
                self.controller.ack_events(group_id, receiver.last_seq)
        self.maintain()
        progress = self.controller.batch_progress()
        required = {(stage, rank) for stage in ("P", "D") for rank in range(self.config.tp_size)}
        for model_id, batch_id in tuple(self.pending):
            batch = progress[model_id].get(batch_id)
            if batch is None or batch["status"] in ("DONE", "ABORTED"):
                self.pending.discard((model_id, batch_id))
            elif batch["status"] == "PREPARING" and set(map(tuple, batch["ready"])) == required:
                try:
                    with self.grant_lock:
                        if self.schedule:
                            self.controller.start_batch(batch["group_id"], batch_id)
                        else:
                            self.controller.abort_batch(batch["group_id"], batch_id, "driver_draining")
                except IntegrationError as exc:
                    if exc.code not in ("NOT_READY", "EVENT_BACKPRESSURE"):
                        raise
        with self.grant_lock:
            if not self.schedule:
                return
            for proposal in proposals["suggestions"]:
                self._prepare(proposal)

    def _prepare(self, proposal):
        candidates = proposal.request_ids
        request_ids = candidates
        while request_ids:
            try:
                outputs = [self.controller.request_output(proposal.model_id, identity)
                           for identity in request_ids]
                plan = BatchPlan(
                    self.controller.group_id(proposal.model_id), uuid4().hex,
                    proposal.model_id, proposal.stage,
                    tuple(RequestRef(output["request_id"], output["attempt"], output["output_count"])
                          for output in outputs),
                    purpose=proposal.purpose, prepare_timeout_ms=self.config.prepare_timeout_ms)
                self.controller.prepare_batch(plan)
                self.pending.add((proposal.model_id, plan.batch_id))
                return
            except IntegrationError as exc:
                if exc.code == "OUT_OF_CAPACITY" and len(request_ids) > 1:
                    request_ids = request_ids[:-1]
                    continue
                if exc.code == "OUT_OF_CAPACITY" and len(candidates) > 1:
                    # The oldest row can need a new physical page while a
                    # younger row is still inside its existing page. Try that
                    # row instead of repeatedly retrying only the same prefix.
                    candidates = candidates[1:]
                    request_ids = candidates
                    continue
                if exc.code in ("OUT_OF_CAPACITY", "OUT_OF_CREDITS", "STAGE_BUSY", "NOT_READY",
                                "STALE_ATTEMPT", "UNKNOWN_REQUEST", "EVENT_BACKPRESSURE",
                                "COMMAND_BACKPRESSURE"):
                    return
                raise

    def stop(self):
        self.stop_event.set()
        self.thread.join()

    def pause(self):
        """Return after any current CPU grant call; issue no later grants."""
        with self.grant_lock:
            self.schedule = False
