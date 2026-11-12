# Paired fixed-workload evaluation

The point used to compare this backend with original MuxServe on one machine:
four A100-40GB GPUs, LLaMA 7B/13B/30B in FP16, each TP4 over all four GPUs,
private MPS with prefill 100% / decode 40%, 2 GiB KV per rank, 512 input and
128 output tokens (greedy, EOS ignored), one request in flight per model
(closed loop), 128 warmup (43/43/42) and 192 measured requests (64 per model).

| File | Content |
|---|---|
| `frozen-synthetic-512-128.json` | The workload: each model's 512 input token IDs and its 128 expected output token IDs, frozen from an accepted earlier run (one input sequence; not a paper dataset). Original MuxServe runs the same bundle. |
| `paired_a100.json` | Backend deployment config for this point (all D40, decode runs, CUDA graphs, parking and the opt-in fast paths). Fill in the model/tokenizer paths and `asset_lock_dir`. |
| `compare_results.py` | Reduces original MuxServe's `result.json` and `tools/bench_fixed.py` output to the same per-model numbers and prints the difference. |

## Run the backend

From the repository root, with the four GPUs reserved, either let the runner
generate the config from `configs/sglang_a100.json`:

```bash
PY=/path/to/venv/bin/python LOCKS=/abs/asset-locks \
HEADROOM_MIB=1024 DECODE_GRAPH=1 PARK=1 PREFILL_GRAPH=1 \
INLINE_START=1 SKIP_P=1 EARLY_TOKEN=1 \
CLEANUP_SKIP_P=1 CLEANUP_WITH_QUIESCE=1 TRIM_MIB=128 \
HANDOFF_PREPARE=1 INLINE_META=1 FINAL_STEP=1 EVENT_PUMP=1 BOOTSTRAP_POLL=1 \
INLINE_PREPARE=1 OWN_TABLE=1 REGISTER_READY=1 BROKER_POLL=1 \
EARLY_TERMINAL=1 SLOT_AT_CLEANUP=1 ACK_IN_POLL=1 COMBINED_REPORT=1 \
START_IN_REPLY=1 TARGETED_WAKE=1 RPC_INLINE=1 DECODE_LEADER=1 \
PREPARE_AFTER_ACK=1 PREPARE_LIGHT=1 \
FROZEN_BUNDLE=$PWD/benchmark/paired/frozen-synthetic-512-128.json WARMUP_TOTAL=128 \
tools/run_perf_point.sh /abs/out-dir 64 0 128
```

or launch `benchmark/paired/paired_a100.json` yourself and drive it with
`tools/bench_fixed.py --frozen-bundle benchmark/paired/frozen-synthetic-512-128.json
--warmup-total 128 --requests 64 --out bench.json`. The bench exits with
status 2 unless every output equals the bundle's expected tokens.
`run_perf_point.sh` documents its other switches (TRACE, PROFILE, SOLO).

## Compare

```bash
python benchmark/paired/compare_results.py --original /path/to/original/result.json \
    --sglang /abs/out-dir/bench.json
```

The paired throughput is set by the slowest model (30B): each model runs a
fixed number of requests, so the measured window ends with the 30B model. The
tool therefore reports per-model cycle = TTFT + TPOT x 127 + tail. Original
MuxServe timings are taken in process (admission to scheduler token return);
backend timings at the HTTP client. The original side of the pair was run
with the lab's own harness on the original MuxServe code
(`9fb0fa555bb4051bee612b72938e5d85796dffec`), which is not part of this
repository; `compare_results.py` reads its `result.json` format.

One run per side gives no error bars, and one fixed point says nothing about
ShareGPT-like variable lengths or higher concurrency.
