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
      (inputs/fixed-512/frozen-synthetic-512-128.json).
  inputs/fixed-512/trace-poisson-0.5rps-600s.jsonl
      Arrivals from original MuxServe's own generator
      (muxserve.muxsched.workload_utils.PoissonProcess, seed = model index),
      0.5 req/s per model for 600 s, 128 output tokens per request.
  inputs/fig5-a0.7-s1.0/prompts-<model>.jsonl, trace.jsonl   (with --sharegpt-bundle)
      The paper Figure 5 point used for the ShareGPT results (alpha 0.7,
      rate scale 1.0, 60 s, 257 requests): ShareGPT token IDs, trace arrival
      times and each request's output length, from the frozen open-loop bundle.
  inputs/fig5-a<alpha>-s<scale>/...   and the Figure 5 grid cases in benches.toml
      (with --fig5 DIR, DIR holding the 20 alpha-*-scale-*/workload.json traces)
      Every point of the paper's Figure 5 sweep. The alpha 0.7 / scale 1.0 point
      must regenerate byte for byte (case 2bdea08f was measured on it); the
      other 19 points get one case each, with fixed IDs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import uuid

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))

from muxserve.muxsched.workload_utils import PoissonProcess  # noqa: E402  original generator

MODELS = {"llama-7b": "huggyllama/llama-7b", "llama-13b": "huggyllama/llama-13b",
          "llama-30b": "huggyllama/llama-30b"}
FROZEN = HERE / "inputs" / "fixed-512" / "frozen-synthetic-512-128.json"
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


def point_files(requests) -> dict[str, bytes]:
    """xbench prompt and trace files for one Figure 5 point, keyed by file name.

    Each request is {idx, model, arrival_s, input_ids, max_new_tokens}; the trace
    keeps every arrival time and output length of the source trace."""
    prompts = {name: [] for name in MODELS}
    trace = []
    for request in requests:
        name = request["model"]
        prompt_id = f"{request['idx']:05d}"
        prompts[name].append({"prompt_id": prompt_id, "input_ids": request["input_ids"]})
        trace.append({"request_id": f"{name}-{prompt_id}", "model_id": MODELS[name],
                      "arrival_seconds": float(request["arrival_s"]), "prompt_id": prompt_id,
                      "max_new_tokens": int(request["max_new_tokens"])})
    trace.sort(key=lambda row: (row["arrival_seconds"], row["request_id"]))
    files = {f"prompts-{name}.jsonl": rows for name, rows in prompts.items()}
    files["trace.jsonl"] = trace
    return {name: "".join(json.dumps(row, separators=(",", ":"), allow_nan=False) + "\n" for row in rows).encode()
            for name, rows in files.items()}


def point_dir(alpha, rate_scale) -> Path:
    return HERE / "inputs" / f"fig5-a{alpha}-s{rate_scale}"


def write_point(out: Path, files: dict[str, bytes], *, keep: bool = False) -> None:
    """Write a point; with keep, the files must already exist byte for byte (results were measured on them)."""
    for name, data in files.items():
        path = out / name
        digest = hashlib.sha256(data).hexdigest()[:12]
        if keep:
            if not path.is_file() or path.read_bytes() != data:
                raise SystemExit(f"FIG5_INPUT_MISMATCH {out.name}/{name}: regenerated bytes differ from the committed file")
            print(f"{out.name}/{name}", digest, "unchanged")
            continue
        out.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        print(f"{out.name}/{name}", digest, data.count(b"\n"))


def sharegpt_inputs(bundle_path: Path) -> None:
    bundle = json.loads(bundle_path.read_text(encoding="utf-8"))
    assert bundle["kind"] == "SHAREGPT_OPEN_LOOP" and bundle["complete_trace"], "need the complete open-loop bundle"
    write_point(point_dir(bundle["alpha"], bundle["rate_scale"]), point_files(bundle["requests"]))
    print("bundle", bundle["bundle_id"][:12], "source trace", bundle["source_trace_sha256"][:12])


# --- Figure 5 grid --------------------------------------------------------------
# The 20 traces of the paper's Figure 5 sweep (alpha 0.7-2.1 x rate scale 1.0-2.5),
# made once on 2026-11-05 by the paper-reproduction preparer from ShareGPT with seed
# 42, 60 s and base rate 2.0 req/s (rate = 2.0 x scale / (rank + 1) ** alpha). Each
# point folder holds the original MuxServe workload.json; it is converted exactly as
# the alpha 0.7 / scale 1.0 point behind case 2bdea08f was (build_open_bundle.py's
# checks, then the writer above).

INPUT_LIMIT, CONTEXT_LIMIT = 1792, 2048
FIG5_KEPT = {(0.7, 1.0): "2bdea08f-077a-4e4b-9b74-f7edcf0ed5c1"}  # measured already; inputs must not change
GRID_BEGIN = "# --- Figure 5 grid: written by make_inputs.py --fig5; regenerate, do not edit by hand ---"
GRID_END = "# --- end of the Figure 5 grid ---"


def fig5_point(workload_path: Path) -> dict:
    trace = json.loads(workload_path.read_bytes())
    arrivals, rows, info = trace["arrivals"], trace["requests"], trace["info"]
    if len(arrivals) != len(rows) or any(left > right for left, right in zip(arrivals, arrivals[1:])):
        raise SystemExit(f"FIG5_TRACE_INVALID {workload_path}: arrivals do not match the requests in order")
    requests = []
    for position, row in enumerate(rows):
        tokens, count, output = row["data"]
        if (row["idx"] != position or row["model_name"] not in MODELS or count != len(tokens)
                or output < 2 or not 1 <= count <= INPUT_LIMIT or count + output > CONTEXT_LIMIT):
            raise SystemExit(f"FIG5_TRACE_INVALID {workload_path}: request {position} is outside the limits")
        requests.append(dict(idx=position, model=row["model_name"], arrival_s=arrivals[position],
                             input_ids=tokens, max_new_tokens=output))
    return dict(alpha=info["alpha"], rate_scale=info["rate_scale"], duration=info["duration"],
                rates=dict(info["rates"]), requests=requests)


def fig5_case_id(alpha, rate_scale) -> str:
    """A fixed UUIDv4-shaped ID per point, so regenerating the grid keeps every case ID."""
    raw = bytearray(hashlib.sha256(f"muxserve-fig5-a{alpha}-s{rate_scale}".encode()).digest()[:16])
    raw[6] = (raw[6] & 0x0F) | 0x40
    raw[8] = (raw[8] & 0x3F) | 0x80
    return str(uuid.UUID(bytes=bytes(raw)))


def fig5_case(point: dict) -> str:
    alpha, scale = point["alpha"], point["rate_scale"]
    case, folder = fig5_case_id(alpha, scale), point_dir(alpha, scale).name
    lines = [f"[serving_cases.{case}]",
             f'description = "MuxServe-SGLang, 3 LLaMA models, paper Figure 5 point alpha {alpha} rate scale {scale}: '
             f'ShareGPT prompts, power-law Poisson trace, {point["duration"]:g} s, {len(point["requests"])} requests. '
             f'Serve with --alpha {alpha} --rate-scale {scale}."',
             'module = "serving.multi_model"', 'mode = "client"', "seed = 0", "max_inflight = 128",
             "warmup_requests_per_target = 0", "request_timeout_seconds = 330.0", "",
             f"[serving_cases.{case}.arrivals]", 'kind = "jsonl"', f'path = "inputs/{folder}/trace.jsonl"',
             f"duration_seconds = {float(point['duration'])}", ""]
    for name, model_id in MODELS.items():
        lines += [f"[[serving_cases.{case}.targets]]", f'model_id = "{model_id}"', 'api = "sglang"',
                  f'base_url = "http://127.0.0.1:18080/models/{name.replace("-", "")}"',
                  f'prompts = {{ kind = "jsonl", path = "inputs/{folder}/prompts-{name}.jsonl" }}',
                  "sampling = { temperature = 0.0, stream_interval = 1, ignore_eos = true }", ""]
    return "\n".join(lines)


def fig5_inputs(sources: list[Path]) -> None:
    paths = []
    for source in sources:
        paths += sorted(source.glob("alpha-*-scale-*/workload.json")) if source.is_dir() else [source]
    if not paths:
        raise SystemExit("FIG5_TRACE_MISSING: give point folders' workload.json or their parent directory")
    points = sorted((fig5_point(path) for path in paths), key=lambda p: (p["alpha"], p["rate_scale"]))
    blocks = []
    print("| alpha | rate scale | requests (7B/13B/30B) | case |")
    for point in points:
        key = (point["alpha"], point["rate_scale"])
        write_point(point_dir(*key), point_files(point["requests"]), keep=key in FIG5_KEPT)
        counts = "/".join(str(sum(r["model"] == name for r in point["requests"])) for name in MODELS)
        case = FIG5_KEPT.get(key) or fig5_case_id(*key)
        if key not in FIG5_KEPT:
            blocks.append(fig5_case(point))
        print(f"| {key[0]} | {key[1]} | {len(point['requests'])} ({counts}) | `{case[:8]}` |")
    catalog = HERE / "benches.toml"
    text = catalog.read_text(encoding="utf-8")
    grid = GRID_BEGIN + "\n# Serve each point with: muxserve serve CONFIG --alpha A --rate-scale S (original: the same\n" \
        "# two options on muxserve.launch). The alpha 0.7 / scale 1.0 point is case 2bdea08f above.\n\n" \
        + "\n".join(blocks) + GRID_END + "\n"
    if GRID_BEGIN in text:
        head, rest = text.split(GRID_BEGIN, 1)
        text = head + grid + rest.split(GRID_END + "\n", 1)[1]
    else:
        text = text.rstrip("\n") + "\n\n" + grid
    catalog.write_bytes(text.encode("utf-8"))  # keep LF line endings on every platform
    print(f"benches.toml: {len(blocks)} Figure 5 grid cases")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sharegpt-bundle", type=Path, help="frozen open-loop bundle (open-a07s1.json)")
    parser.add_argument("--fig5", type=Path, nargs="+", metavar="PATH",
                        help="Figure 5 traces: point folders' workload.json, or the folder holding alpha-*-scale-*/")
    args = parser.parse_args()
    fixed_inputs()
    if args.sharegpt_bundle:
        sharegpt_inputs(args.sharegpt_bundle)
    if args.fig5:
        fig5_inputs(args.fig5)


if __name__ == "__main__":
    main()
