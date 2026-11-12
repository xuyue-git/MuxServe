#!/usr/bin/env python3
"""Write the prompt and trace JSONL inputs that benches.toml references.

    python benchmark/xbench/make_inputs.py [--sharegpt-bundle open-a07s1.json]

Deterministic; rerunning rewrites byte-identical files. Formats are xbench's
(CrossPool 3c1f976, xbench/harness/serving/workload.py):
  prompts JSONL  {"prompt_id": str, "input_ids": [int, ...]}           one per line
  trace JSONL    {"request_id": str, "model_id": "ns/name", "arrival_seconds": float,
                  "prompt_id": str, "max_new_tokens": int}             sorted by arrival

Inputs written:
  inputs/fixed-512/prompts-<model>.jsonl
      Original MuxServe's frozen 512-token prompt for each model
      (benchmark/paired/frozen-synthetic-512-128.json).
  inputs/fixed-512/trace-poisson-0.5rps-600s.jsonl
      Arrivals from original MuxServe's own generator
      (muxserve.muxsched.workload_utils.PoissonProcess, seed = model index),
      0.5 req/s per model for 600 s, 128 output tokens per request.
  inputs/fig5-a0.7-s1.0/prompts-<model>.jsonl, trace.jsonl   (with --sharegpt-bundle)
      The paper Figure 5 point used for the ShareGPT results (alpha 0.7,
      rate scale 1.0, 60 s, 257 requests): ShareGPT token IDs, trace arrival
      times and each request's output length, from the frozen open-loop bundle.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))

from muxserve.muxsched.workload_utils import PoissonProcess  # noqa: E402  original generator

MODELS = {"llama-7b": "huggyllama/llama-7b", "llama-13b": "huggyllama/llama-13b",
          "llama-30b": "huggyllama/llama-30b"}
FROZEN = ROOT / "benchmark" / "paired" / "frozen-synthetic-512-128.json"
FIXED_RATE, FIXED_SECONDS, FIXED_OUTPUT = 0.5, 600.0, 128


def write_jsonl(path: Path, rows) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = "".join(json.dumps(row, separators=(",", ":"), allow_nan=False) + "\n" for row in rows)
    path.write_bytes(data.encode("utf-8"))
    return hashlib.sha256(data.encode("utf-8")).hexdigest()[:12]


def fixed_inputs() -> None:
    bundle = json.loads(FROZEN.read_text(encoding="utf-8"))
    out = HERE / "inputs" / "fixed-512"
    for name in MODELS:
        ids = bundle["models"][name]["input_ids"]
        assert len(ids) == 512
        digest = write_jsonl(out / f"prompts-{name}.jsonl", [{"prompt_id": "fixed-512", "input_ids": ids}])
        print(f"prompts-{name}.jsonl", digest)
    rows = []
    for index, (name, model_id) in enumerate(MODELS.items()):
        ticks = PoissonProcess(FIXED_RATE).generate_arrivals(0.0, FIXED_SECONDS, seed=index)
        rows += [{"request_id": f"{name}-{i:05d}", "model_id": model_id, "arrival_seconds": round(float(t), 6),
                  "prompt_id": "fixed-512", "max_new_tokens": FIXED_OUTPUT} for i, t in enumerate(ticks)]
    rows.sort(key=lambda row: (row["arrival_seconds"], row["request_id"]))
    digest = write_jsonl(out / "trace-poisson-0.5rps-600s.jsonl", rows)
    counts = {model: sum(row["model_id"] == model for row in rows) for model in MODELS.values()}
    print("trace-poisson-0.5rps-600s.jsonl", digest, len(rows), counts)


def sharegpt_inputs(bundle_path: Path) -> None:
    bundle = json.loads(bundle_path.read_text(encoding="utf-8"))
    assert bundle["kind"] == "SHAREGPT_OPEN_LOOP" and bundle["complete_trace"], "need the complete open-loop bundle"
    out = HERE / "inputs" / f"fig5-a{bundle['alpha']}-s{bundle['rate_scale']}"
    prompts = {name: [] for name in MODELS}
    trace = []
    for request in bundle["requests"]:
        name = request["model"]
        prompt_id = f"{request['idx']:05d}"
        prompts[name].append({"prompt_id": prompt_id, "input_ids": request["input_ids"]})
        trace.append({"request_id": f"{name}-{prompt_id}", "model_id": MODELS[name],
                      "arrival_seconds": float(request["arrival_s"]), "prompt_id": prompt_id,
                      "max_new_tokens": int(request["max_new_tokens"])})
    for name, rows in prompts.items():
        print(f"{out.name}/prompts-{name}.jsonl", write_jsonl(out / f"prompts-{name}.jsonl", rows), len(rows))
    trace.sort(key=lambda row: (row["arrival_seconds"], row["request_id"]))
    print(f"{out.name}/trace.jsonl", write_jsonl(out / "trace.jsonl", trace), len(trace),
          "bundle", bundle["bundle_id"][:12], "source trace", bundle["source_trace_sha256"][:12])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sharegpt-bundle", type=Path, help="frozen open-loop bundle (open-a07s1.json)")
    args = parser.parse_args()
    fixed_inputs()
    if args.sharegpt_bundle:
        sharegpt_inputs(args.sharegpt_bundle)


if __name__ == "__main__":
    main()
