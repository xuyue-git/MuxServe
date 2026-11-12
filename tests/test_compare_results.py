"""benchmark/paired/compare_results.py reduces both result formats to the same numbers."""
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "benchmark" / "paired"))
import compare_results  # noqa: E402


def requests(count, ttft, tpot, gap, tokens=128):
    """Closed loop: each request starts `gap` seconds after the previous last token."""
    rows, start = [], 10.0
    for _ in range(count):
        times = [start + ttft + tpot * i for i in range(tokens)]
        rows.append((start, times))
        start = times[-1] + gap
    return rows


class CompareResults(unittest.TestCase):
    def test_both_formats_give_the_same_metrics_for_the_same_timeline(self):
        timeline = {"llama7b": requests(4, 0.15, 0.02, 0.001), "llama30b": requests(4, 0.12, 0.035, 0.07)}
        original = {"observations_raw": {"observation_boundary": "in_process", "records": [
            {"phase": "measured", "model": model.replace("llama", "llama-"), "admitted_at": start,
             "token_times": times} for model, rows in timeline.items() for start, times in rows]
            + [{"phase": "warmup", "model": "llama-7b", "admitted_at": 0.0, "token_times": [0.1, 0.2]}]}}
        sglang = {"reference_match": {"llama7b": True, "llama30b": True}, "records": {
            model: [{"sent": start, "token_times": times, "error": None} for start, times in rows]
            for model, rows in timeline.items()}}
        with tempfile.TemporaryDirectory() as directory:
            paths = [Path(directory) / name for name in ("result.json", "bench.json")]
            paths[0].write_text(json.dumps(original)), paths[1].write_text(json.dumps(sglang))
            a = compare_results.summarize(compare_results.load_original(paths[0]))
            b = compare_results.summarize(compare_results.load_sglang(paths[1]))
        self.assertEqual(set(a["models"]), {"llama7b", "llama30b"})  # warmup record ignored
        for model in a["models"]:
            for key in ("tok_s", "ttft_ms", "tpot_ms", "cycle_ms", "tail_ms", "gap_p99_ms", "window_s"):
                self.assertAlmostEqual(a["models"][model][key], b["models"][model][key], places=6)
        m = a["models"]["llama30b"]
        self.assertAlmostEqual(m["ttft_ms"], 120.0, places=6)
        self.assertAlmostEqual(m["tpot_ms"], 35.0, places=6)
        self.assertAlmostEqual(m["tail_ms"], 70.0, places=6)
        self.assertAlmostEqual(m["cycle_ms"], 120.0 + 127 * 35.0 + 70.0, places=6)
        self.assertEqual(b["reference_match"], {"llama7b": True, "llama30b": True})
        text = compare_results.render([a, b])
        self.assertIn("throughput 100.0%", text)


if __name__ == "__main__":
    unittest.main()
