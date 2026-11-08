"""Independent event pump and external ADBS batch grants for the owner."""
from __future__ import annotations

import logging
from threading import Event, RLock, Thread
from uuid import uuid4

from .contracts import BatchPlan, IntegrationError, RequestRef
from .events import EventReceiver
from . import trace

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
        # event_pump_thread: once started, a second thread alone delivers
        # runtime events (TOKEN/TERMINAL) to the gateway, so they never wait
        # for a scheduling pass (tick, START, prepare). Synchronous step()
        # calls (initialization) keep pumping inline.
        events_wake = getattr(controller, "events_wake", None)
        self.pump_thread = (Thread(target=self._pump_loop, name="event-pump", daemon=False)
                            if getattr(self.config, "event_pump_thread", False) and events_wake is not None
                            else None)
        self._pumping = False
        # A scheduling driver owns START grants; the controller may then issue
        # them inline (config inline_start) until pause() or a failure.
        self._inline(schedule)

    def _inline(self, allowed):
        setter = getattr(self.controller, "set_inline_start", None)
        if setter is not None:
            setter(allowed)

    def start(self):
        if self.pump_thread is not None:
            self._pumping = True
            self.pump_thread.start()
        self.thread.start()

    def _pump_events(self):
        for model_id, receiver in self.receivers.items():
            group_id = receiver.group_id
            for event in self.controller.poll_events(group_id, receiver.last_seq):
                receiver.receive(event)
            if receiver.last_seq:
                self.controller.ack_events(group_id, receiver.last_seq)

    def _pump_loop(self):
        wake = self.controller.events_wake
        try:
            while not self.stop_event.is_set():
                wake.clear()  # before pumping: an event emitted meanwhile keeps it set
                self._pump_events()
                wake.wait(0.02)
        except Exception as exc:
            self.failure = exc
            self.schedule = False
            self._inline(False)
            logger.exception("Event pump stopped; supervisor must retain producers")

    def run(self):
        # Wake on controller activity (reports, decode steps, submissions,
        # commands); the timeout keeps deadline/watchdog ticks periodic.
        wake = getattr(self.controller, "wake", None)
        try:
            while not self.stop_event.is_set():
                if wake is not None:
                    # Clear before stepping: activity during step() keeps the
                    # flag set, so the wait below returns at once (no lost wake).
                    wake.clear()
                self.step()
                if wake is None:
                    self.stop_event.wait(0.005)
                else:
                    wake.wait(0.02)
        except Exception as exc:
            # No cleanup/rollback proof follows an unknown driver failure.
            self.failure = exc
            self.schedule = False
            self._inline(False)
            logger.exception("Controller driver stopped; supervisor must retain producers")

    def step(self):
        inline_error = getattr(self.controller, "inline_start_error", None)
        if inline_error is not None:
            raise inline_error  # an inline START failed exactly where ours would have
        proposals = self.controller.tick(include_adjustments=False)
        if not self._pumping:
            self._pump_events()
        self.maintain()
        progress = self.controller.batch_progress()
        everyone = {(stage, rank) for stage in ("P", "D") for rank in range(self.config.tp_size)}
        for model_id, batch_id in tuple(self.pending):
            batch = progress[model_id].get(batch_id)
            if batch is None or batch["status"] in ("DONE", "ABORTED"):
                self.pending.discard((model_id, batch_id))
            elif batch["status"] == "PREPARING" and set(map(tuple, batch["ready"])) == (
                    set(map(tuple, batch.get("participants") or ())) or everyone):
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

    def _run_steps(self, proposal, outputs):
        limit = getattr(self.config, "decode_run_steps", 1)
        if proposal.stage != "D" or proposal.purpose != "GENERATE" or limit <= 1:
            return 1
        maximum = self.config.max_new_tokens
        remaining = []
        with self.controller.lock:
            for output in outputs:
                request = self.controller.requests.get((proposal.model_id, output["request_id"]))
                budget = request.spec.max_new_tokens if request is not None else maximum
                remaining.append(budget - output["output_count"])
        return max(1, min([limit] + remaining))

    def _prepare(self, proposal):
        candidates = proposal.request_ids
        request_ids = candidates
        steps = None
        while request_ids:
            try:
                outputs = [self.controller.request_output(proposal.model_id, identity)
                           for identity in request_ids]
                if steps is None:
                    steps = self._run_steps(proposal, outputs)
                plan = BatchPlan(
                    self.controller.group_id(proposal.model_id), uuid4().hex,
                    proposal.model_id, proposal.stage,
                    tuple(RequestRef(output["request_id"], output["attempt"], output["output_count"])
                          for output in outputs),
                    purpose=proposal.purpose, prepare_timeout_ms=self.config.prepare_timeout_ms,
                    steps=min(steps, self._run_steps(proposal, outputs)))
                self.controller.prepare_batch(plan)
                self.pending.add((proposal.model_id, plan.batch_id))
                return
            except IntegrationError as exc:
                trace.ev("prep_fail", None, m=proposal.model_id, s=proposal.stage, code=exc.code)
                if exc.code == "OUT_OF_CAPACITY" and steps is not None and steps > 1:
                    # A shorter run needs less reserved KV; fall back to one step
                    # before shrinking the request set.
                    steps = 1
                    continue
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
        self._inline(False)
        self.stop_event.set()
        for name in ("wake", "events_wake"):
            event = getattr(self.controller, name, None)
            if event is not None:
                event.set()
        self.thread.join()
        if self.pump_thread is not None and self.pump_thread.is_alive():
            self.pump_thread.join()

    def pause(self):
        """Return after any current CPU grant call; issue no later grants."""
        with self.grant_lock:
            self.schedule = False
            self._inline(False)
