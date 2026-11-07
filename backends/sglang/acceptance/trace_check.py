"""Opt-in timing trace: no-op when disabled, never raises, report arithmetic."""
import json
import os
import sys
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))

from muxserve_sglang import trace  # noqa: E402
import trace_report  # noqa: E402


class TraceChecks(unittest.TestCase):
    def setUp(self):
        self.saved = trace._DIR
        trace._BUFFER.clear()
        trace._STATE.update(file=None, pid=None, flushed=0.0, role="")

    def tearDown(self):
        trace._DIR = self.saved
        trace._BUFFER.clear()
        trace._STATE.update(file=None, pid=None)

    def test_disabled_is_noop(self):
        trace._DIR = None
        self.assertFalse(trace.enabled())
        trace.ev("x", "b1", k=1)
        trace.flush()
        self.assertEqual(trace._BUFFER, [])

    def test_enabled_writes_jsonl(self):
        with tempfile.TemporaryDirectory() as directory:
            trace._DIR = directory
            trace.set_role("m-P0")
            trace.ev("acc", "b1", kind="PREPARE")
            trace.flush()
            files = os.listdir(directory)
            self.assertEqual(files, [f"trace-{os.getpid()}.jsonl"])
            record = json.loads(open(os.path.join(directory, files[0])).read())
            self.assertEqual((record["ev"], record["batch"], record["role"], record["kind"]),
                             ("acc", "b1", "m-P0", "PREPARE"))

    def test_unwritable_never_raises_or_grows(self):
        with tempfile.NamedTemporaryFile() as handle:
            trace._DIR = os.path.join(handle.name, "sub")  # parent is a file
            for _ in range(2000):
                trace.ev("x")
            trace.flush()
            self.assertLess(len(trace._BUFFER), 600)

    def test_report_segments(self):
        events = []

        def add(t, name, batch=None, **fields):
            events.append({"t": t, "pid": 1, "role": fields.pop("role", ""), "ev": name,
                           **({"batch": batch} if batch else {}), **fields})
        add(0.000, "submit", m="m", req="r1")
        add(0.010, "prep_call", "p1")
        add(0.010, "prep", "p1", m="m", gid="g", s="P", purpose="GENERATE", steps=1, reqs=["r1"])
        for r in range(4):
            add(0.020 + r / 1000, "dlv", "p1", kind="PREPARE", m="m", s="P", r=r)
            add(0.021 + r / 1000, "dlv", "p1", kind="PREPARE", m="m", s="D", r=r)
            add(0.050, "rpt", "p1", kind="READY", gid="g", s="P", r=r)
            add(0.060 if r == 3 else 0.055, "rpt", "p1", kind="READY", gid="g", s="D", r=r)
        add(0.070, "start", "p1", m="m", s="P")
        add(0.080, "dlv", "p1", kind="START", m="m", s="P", r=0)
        add(0.180, "gpu_all", "p1", m="m", s="P")
        add(0.190, "dlv", "p1", kind="HANDOFF", m="m", s="D", r=0)
        add(0.191, "acc", "p1", kind="HANDOFF", role="m-D0")
        add(0.200, "acc_end", "p1", kind="HANDOFF", role="m-D0")
        add(0.300, "ho_ok", "p1", role="m-D0")
        add(0.305, "ho_pop", "p1", role="m-D0")
        add(0.306, "ho_prebuilt", "p1", role="m-D0")
        add(0.310, "ho_sync", "p1", role="m-D0")
        add(0.320, "handoff_all", "p1", m="m")
        add(0.330, "finish", "p1", m="m", s="P")
        add(0.400, "prep", "d1", m="m", gid="g", s="D", purpose="GENERATE", steps=3, reqs=["r1"])
        for step, t in ((1, 0.500), (2, 0.530), (3, 0.900)):
            for r in range(4):
                add(t, "dstep", "d1", gid="g", r=r, step=step)
        add(0.910, "gpu_all", "d1", m="m", s="D")
        add(0.920, "finish", "d1", m="m", s="D")
        events.append({"t": 0.1, "pid": 2, "role": "m-D1", "ev": "rpc", "m": "controller_broker_ack", "ms": 4.0})
        with tempfile.TemporaryDirectory() as directory:
            with open(os.path.join(directory, "trace-1.jsonl"), "w") as handle:
                handle.write("\n".join(json.dumps(e) for e in events) + "\n{partial")
            result = trace_report.report(directory, gap_ms=150)
        m = result["phases"]["all"]["m"]
        seg = m["P_segments_ms"]
        self.assertAlmostEqual(seg["prepare_delivered"]["p50"], 14.0, places=1)
        self.assertAlmostEqual(seg["all_ready"]["p50"], 36.0, places=1)
        self.assertAlmostEqual(seg["gpu_done"]["p50"], 100.0, places=1)
        self.assertAlmostEqual(seg["handoff_polled_ok"]["p50"], 100.0, places=1)
        self.assertAlmostEqual(seg["total_prepare_to_first_token_ready"]["p50"], 310.0, places=1)
        self.assertEqual(m["P_last_ready_stage"], {"D": 1})
        self.assertAlmostEqual(m["P_submit_to_prepare_ms"]["p50"], 10.0, places=1)
        self.assertAlmostEqual(m["P_rank_handlers_ms"]["D_HANDOFF_handler"]["p50"], 9.0, places=1)
        self.assertAlmostEqual(m["D_in_run_step_ms"]["max"], 370.0, places=1)
        kinds = m["decode_time_by_gap_kind_ms"]
        self.assertAlmostEqual(kinds["after_prefill"]["total"], 180.0, places=1)
        self.assertAlmostEqual(kinds["in_run"]["total"], 400.0, places=1)
        self.assertEqual(m["big_gaps"]["count"], 2)
        self.assertEqual(result["rpc_ms"]["D:controller_broker_ack"]["n"], 1)


if __name__ == "__main__":
    unittest.main()
