"""CPU checks for the owner-speed switches (patch 57).

  python acceptance/owner_speed_check.py

event_lean_copies, driver_lean_locking and the gateway_process pipe logic. The
whole HTTP path through a real gateway child runs in control_plane_e2e_check.py
--gateway-proc.
"""
from __future__ import annotations

from pathlib import Path
import sys
from threading import Condition
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from muxserve.contracts import IntegrationError
from muxserve.muxsched.events import EventReceiver, RuntimeOutbox
from muxserve.gateway_proc import RemoteGatewayState

import decode_run_check as drc  # module import: its TestCases are not collected here
from decode_run_check import Sim


class LeanEventChecks(unittest.TestCase):
    def test_lean_outbox_shares_events_and_keeps_order(self):
        outbox = RuntimeOutbox("g", "m", 16, lean=True)
        first = outbox.emit("TOKEN", request_id="r", attempt=0, payload={"token_ids": (5,), "output_index": 0})
        second = outbox.emit("TOKEN", request_id="r", attempt=0, payload={"token_ids": (6,), "output_index": 1})
        self.assertEqual(outbox.replay(0), (first, second))
        self.assertIs(outbox.replay(0)[0], first)
        outbox.ack(1)
        self.assertEqual(outbox.replay(1), (second,))

    def test_lean_receiver_applies_in_order_and_rejects_changed_replays(self):
        outbox = RuntimeOutbox("g", "m", 16, lean=True)
        events = [outbox.emit("TOKEN", request_id="r", payload={"token_ids": (t,), "output_index": i})
                  for i, t in enumerate((5, 6, 7))]
        seen = []
        receiver = EventReceiver("g", seen.append, lean=True)
        receiver.receive(events[1])  # a gap is buffered
        self.assertEqual(seen, [])
        receiver.receive(events[0])
        receiver.receive(events[2])
        self.assertEqual([e.payload["token_ids"] for e in seen], [(5,), (6,), (7,)])
        self.assertEqual(receiver.receive(events[1]), 3)  # an identical replay is ignored
        changed = type(events[1])(**{**events[1].__dict__, "payload": {"token_ids": (8,), "output_index": 1}})
        with self.assertRaises(IntegrationError):
            receiver.receive(changed)

    def test_default_outbox_still_copies(self):
        outbox = RuntimeOutbox("g", "m", 16)
        event = outbox.emit("TOKEN", request_id="r", payload={"token_ids": (5,), "output_index": 0})
        self.assertIsNot(outbox.replay(0)[0], event)
        self.assertEqual(outbox.replay(0)[0], event)


class LeanDriverChecks(unittest.TestCase):
    def test_request_briefs_match_request_output(self):
        sim = Sim(run_steps=8, driver_lean_locking=True)
        sim.submit("a", "r", max_new=5)
        sim.prefill("a", "r", 11)
        brief = sim.controller.request_briefs("a", ["r"])[0]
        output = sim.controller.request_output("a", "r")
        self.assertEqual((brief["request_id"], brief["attempt"], brief["output_count"]),
                         (output["request_id"], output["attempt"], output["output_count"]))
        self.assertEqual(brief["remaining"], 5 - output["output_count"])
        with self.assertRaises(IntegrationError):
            sim.controller.request_briefs("a", ["missing"])

    def test_batch_progress_only_lists_the_asked_batches(self):
        sim = Sim(run_steps=8, driver_lean_locking=True)
        sim.submit("a", "r", max_new=20)
        sim.prefill("a", "r", 11)
        plan, _ = sim.prepare_decode("a", "r", 8)
        everything = sim.controller.batch_progress()
        only = sim.controller.batch_progress((("a", plan.batch_id), ("a", "gone")))
        self.assertEqual(only, {"a": {plan.batch_id: everything["a"][plan.batch_id]}})

    def test_tick_scans_deadlines_at_most_every_50_ms(self):
        sim = Sim(run_steps=8, driver_lean_locking=True)
        sim.controller.tick()
        sim.submit("a", "r", max_new=5)
        request = sim.controller.requests[("a", "r")]
        request.deadline = sim.clock.now - 1  # already expired
        sim.clock.now += 0.01
        sim.controller.tick()
        self.assertIsNone(request.pending_reason)  # within 50 ms of the last scan
        sim.clock.now += 0.05
        sim.controller.tick()
        self.assertTrue(request.pending_reason == "timeout" or request.state == "TIMED_OUT",
                        (request.pending_reason, request.state))


class StepHistoryChecks(unittest.TestCase):
    """policy_step_history: ADBS hears every decode-run step, as the source hears every
    decode batch; without it a run is one entry (its members at the end)."""

    step = drc.StepJoinChecks.step
    batch = staticmethod(drc.StepJoinChecks.batch)

    def run_two_steps(self, **flags):
        sim = Sim(run_steps=2, **flags)
        sim.submit("a", "r", max_new=10)
        sim.prefill("a", "r", 11)
        plan, reservation = sim.prepare_decode("a", "r", 2)
        for step, token in ((1, 12), (2, 13)):
            for rank in range(sim.tp):
                self.step(sim, plan.batch_id, rank, step, {"r": token})
        drc.StepJoinChecks.finish(self, sim, plan.batch_id, reservation, 2, {"r": 13})
        sim.controller.tick()
        return list(sim.controller.policy.histories["a"])

    @staticmethod
    def join_flags():
        return dict(drc.StepJoinChecks.JOIN, decode_leader_reports=True, decode_step_leave=True)

    def test_each_run_step_is_one_history_entry(self):
        self.assertEqual(self.run_two_steps(policy_step_history=True, **self.join_flags()), [1, 1])

    def test_without_the_switch_a_run_is_one_entry(self):
        self.assertEqual(self.run_two_steps(**self.join_flags()), [1])

    def test_adjustment_line_is_printed_only_with_the_switch(self):
        import contextlib, io
        for flag, expected in ((True, 1), (False, 0)):
            sim = Sim(run_steps=2, policy_log_adjustments=flag)
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                sim.clock.now += 9
                sim.controller.tick()
            self.assertEqual(out.getvalue().count("POLICY_ADJUST "), expected)


class _Conn:
    def __init__(self):
        self.sent = []

    def send(self, message):
        self.sent.append(message)


class RemoteStateChecks(unittest.TestCase):
    def state(self, reply):
        state = RemoteGatewayState(_Conn(), {"a": None}, {"queue_limit": 8, "fingerprint": "f"})
        state.call = lambda kind, *args, **kw: reply(kind, *args)
        return state

    def test_updates_before_the_submit_reply_are_kept(self):
        state = self.state(lambda kind, *args: ("id1", "rid", 3, 1))
        state.apply_updates([("id1", 0, [5, 6], "DECODING", None, None)])  # arrives first
        record = state.submit("a", {"input_ids": [1, 2, 3]})
        self.assertEqual(record.tokens, [5, 6])
        state.apply_updates([("id1", 1, [6, 7], "FINISHED", "length", None)])  # overlap is skipped
        self.assertEqual((record.tokens, record.state, record.finish_reason), ([5, 6, 7], "FINISHED", "length"))

    def test_disconnect_tells_the_owner_and_drops_later_updates(self):
        state = self.state(lambda kind, *args: ("id1", "rid", 3, 1))
        record = state.submit("a", {"input_ids": [1, 2, 3]})
        state.disconnect(record)
        self.assertIn(("disconnect", "id1"), state.conn.sent)
        state.apply_updates([("id1", 0, [5], "DECODING", None, None)])
        self.assertEqual(state.early, {})

    def test_errors_end_the_stream_and_drain_stops_admission(self):
        state = self.state(lambda kind, *args: ("id1", "rid", 3, 1))
        record = state.submit("a", {"input_ids": [1, 2, 3]})
        state.apply_updates([("id1", 0, [], "QUEUED", None, "timeout")])
        self.assertEqual(record.error, "timeout")
        state.begin_drain()
        self.assertFalse(state.accepting)

    def test_text_is_tokenized_in_the_child(self):
        class Tok:
            def encode(self, text, add_special_tokens=True):
                return [len(text), 7]
        seen = {}
        def reply(kind, model_id, body):
            seen.update(body)
            return ("id1", "rid", 2, 1)
        state = self.state(reply)
        state.tokenizers = {"a": Tok()}
        state.submit("a", {"text": "hello", "stream": True})
        self.assertEqual(seen, {"input_ids": [5, 7], "stream": True})


if __name__ == "__main__":
    unittest.main()
