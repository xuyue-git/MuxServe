# xbench catalogue

Standard serving benchmarks for MuxServe on SGLang, written in CrossPool
xbench's catalogue format (CrossPool `3c1f976cdd1069756314ecb7142581fad2367272`,
`benches/benches.toml` schema). Each case is a client-mode case: the server is
launched with one config file and one command, then xbench drives it over HTTP.
No wrapper script sits between these commands and the measured run.

| File | Purpose |
|---|---|
| `benches.toml` | The catalogue: arrival process, rates, duration, prompts, output lengths, sampling, warmup, in-flight limit, endpoints |
| `inputs/` | Prompt and trace JSONL files referenced by the catalogue |
| `make_inputs.py` | Regenerates `inputs/` byte-identically (original MuxServe generator) |
| `suites/serving/multi_model.py` | CrossPool's serving suite module, unchanged (MIT, Copyright (c) 2026 Yip Coekjan), resolved by `module = "serving.multi_model"` |

## Cases

| Case (prefix) | Load | Arrivals | Requests | Prompt / output | Server config |
|---|---|---|---|---|---|
| `55af3ef0` | Fixed 512/128 | xbench Poisson, 0.5 req/s per model, 600 s, seed 0 | about 900 | original MuxServe's frozen 512-token prompt / exactly 128 tokens | `configs/a100_xbench_poisson.json` |
| `279fe35b` | Fixed 512/128 | trace of original MuxServe's `PoissonProcess`, 0.5 req/s per model, 600 s | 903 (298 / 290 / 315) | same | `configs/a100_xbench_poisson.json` |
| `2bdea08f` | Paper Figure 5, alpha 0.7, rate scale 1.0 | power-law Poisson trace: 2.0 / 1.2311 / 0.9269 req/s, 60 s | 257 (121 / 76 / 60) | ShareGPT V3 token IDs / each request's ShareGPT length | `configs/a100_open_loop.json` |

Common to all cases: LLaMA-7B/13B/30B (huggyllama), FP16, TP4 on four A100
40 GB GPUs; native SGLang `/generate` with SSE at
`http://127.0.0.1:18080/models/<name>`; greedy decoding, one SSE event per
token, EOS ignored so every request generates exactly its requested length; at
most 128 active HTTP requests in total (xbench's FIFO admission queues the
rest); request timeout 330 s (the server's own deadline is 300 s).

Alpha is the power-law popularity exponent: model `i` (0-based, by popularity)
receives `R * rate_scale / (i + 1)^alpha` req/s with `R = 2.0` here (the paper's
Figure 5 uses 19 models, 32 GPUs and `R = 20`).

## Run

1. Server (repository root, the four GPUs reserved). Set `asset_lock_dir` in the
   config to your personal lock directory first.

   ```bash
   export CUDA_VISIBLE_DEVICES=1,2,3,4
   python -m muxserve.launch configs/a100_xbench_poisson.json
   ```

2. Client, once `curl -s http://127.0.0.1:18080/health` reports ready, from a
   CrossPool checkout at the commit above (`uv sync --group dev` once):

   ```bash
   uv run xbench run --catalog /path/to/MuxServe/benchmark/xbench/benches.toml --case 55af3ef0
   uv run xbench report RUN_ID
   ```

   Stop the server with Ctrl-C (ordered drain and MPS shutdown).

For case `2bdea08f`, launch with `configs/a100_open_loop.json` instead. Repeat a
case with `--repetitions N`. xbench retains the prepared workload, per-request
records and its report under its run directory.

## Server configs

`configs/a100_xbench_poisson.json` and `configs/a100_open_loop.json` hold the
complete deployment, including every runtime switch of the fastest measured
configuration (see `docs/runtime.md`). They differ only in
`policy_arrival_rates`, which sets ADBS's starting per-model limits from the
case's arrival rates as original MuxServe's `init_max_num_seqs` does.

## Inputs

```bash
python benchmark/xbench/make_inputs.py --sharegpt-bundle /path/to/open-a07s1.json
```

The fixed prompts come from `benchmark/paired/frozen-synthetic-512-128.json`.
The fixed trace uses `muxserve.muxsched.workload_utils.PoissonProcess` with
seed = model index (7B 0, 13B 1, 30B 2). The Figure 5 inputs come from the frozen
open-loop bundle (bundle `959024cce558`, source trace `d1d735cc3909`) built from
ShareGPT V3 with original MuxServe's `sample_requests` (llama-7b tokenizer,
seed 42, candidate pool 4096).

## Baseline

Original MuxServe (vLLM) has no HTTP serving entry, so it cannot be driven by a
client-mode case without an added entry point. It runs from its own repository
and environment; its Figure 5 records come from its own trace replay.
