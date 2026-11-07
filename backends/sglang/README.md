# Optional SGLang Head-wise backend

This directory contains the SGLang backend of this MuxServe fork. It runs next
to the original MuxServe/vLLM implementation in the repository root and imports
its SM accounting and KV block sizing. It adds shared weights, Head-wise KV
storage and externally controlled prefill/decode execution.

Source: [MuxServe-SGLang-Head_wise_cache at a512d05](https://github.com/xuyue-git/MuxServe-SGLang-Head_wise_cache/tree/a512d05496122521fb16fa620aa44d8f4d4d3d18).
SGLang is pinned to **v0.5.20**, commit `94602c9c2b7cbdb8efd5c52802dac6a1c180089e`.
See [the original-to-backend mapping](MIGRATION.md) and [source notices](NOTICE).

## Scope

- Original MuxServe `SMResource` (one per TP rank) for SM credits and the
  original KV block formula, imported from `muxserve.muxsched.resource`.
- Shared physical Head-wise KV pools and model weights via CUDA IPC.
- Externally granted prefill/decode execution and request lifecycle control.
- Native SGLang integration through 13 pinned upstream source patches.
- Private MPS startup, admission checks and ordered shutdown.

## Build and launch

Run backend commands from this directory. Keep its environment separate from
the legacy root project. `muxserve_sglang` appends the repository root to the
import path and imports only vLLM-free original modules, so the root package
needs no installation, but the backend must stay inside this checkout. The prototype targets Linux, Python 3.10, four reserved
A100 40 GB GPUs, CUDA 13.2 and the package pins in `requirements.txt`.

1. Materialize SGLang with `python tools/bootstrap_source.py`. For offline use,
   pass `--archive /path/to/sglang-v0.5.20-source.tar.gz`; size and SHA256 are checked.
2. Install the pinned patched source from `third_party/sglang/python`, the
   backend requirements and this directory's CUDA extension in a GPU environment.
   The build uses `setup.py`; an ordinary SGLang installation lacks the hooks.
   A complete transitive dependency lock and clean GPU installation remain unvalidated.
3. Fill in explicit model/tokenizer paths and a personal `asset_lock_dir` in
   `configs/sglang_a100.json`. Keep the exact model revisions; verify the assets
   before generating their integrity locks with `python tools/lock_assets.py CONFIG`.
4. Set `CUDA_VISIBLE_DEVICES` to the four GPUs reserved for the run, then use:

```bash
python -m muxserve_sglang.preflight configs/sglang_a100.json
python -m muxserve_sglang.launcher configs/sglang_a100.json
```

The launcher creates a private MPS controller. Its gateway accepts `/health`
and per-model generation requests. Model IDs and port come from the config.

The source manifests are consumed by bootstrap and preflight checks. They
are not an embedded copy of the SGLang repository and must not be removed
without updating those checks. The materialized upstream tree is ignored by Git.

## CPU checks

With Linux Python 3.12 (WSL works), install `pyzmq` for the native-channel check,
materialize the pinned source as above, and run:

```bash
python -I -B tools/check_cpu.py
```

These checks exercise CPU storage, control, transport and source contracts.
They do not load model weights or validate CUDA execution/performance.

## Evaluation

[`benchmark/`](benchmark/README.md) holds the paired fixed-workload point used
against original MuxServe: the frozen 512/128 workload, its deployment config,
the commands to run it and `compare_results.py` to compare the results.

## Decode runs and control-plane performance

`decode_run_steps` (default 1 = original per-token grants; the sample config
uses 128) lets one D grant cover several consecutive decode forwards. KV for
the whole run is reserved once; every TP rank reports each step through
`decode_step`, a token is published once all ranks agree, and one memoized
continue decision per step keeps ranks in lockstep. A run yields at the next
step boundary when a prefill waits for credits, cancellation/timeout arrives,
or capacity recovery protects another request. Ranks keep one authenticated
RPC connection, long-poll their commands, and the driver wakes on controller
activity instead of polling. `tools/bench_fixed.py` reproduces the fixed
512/128 closed-loop point against the gateway (`/metrics` exposes the
controller's per-stage totals); `acceptance/control_plane_e2e_check.py` runs
the full control plane with simulated rank processes on CPU.

`decode_cuda_graph: true` captures D-stage decode forwards as SGLang-native
full CUDA graphs (batch sizes 1..`max_model_inflight`; P stays eager). Capture
runs at model-runner initialization, before the rank registers, on static
per-layer slot / page-table / length buffers sized for 2048-token contexts.
Every replay still passes the per-forward grant check and `validate_forward`
of the live batch; the validated grant is then copied into those buffers.

`decode_run_park: true` changes how a decode run yields to a waiting prefill.
Instead of ending (GPU_DONE, KV commit, quiesce) and being re-prepared later,
the run parks at the step boundary: once every D rank finished that step its
credits are released, while its KV reservation and native state stay. The
ranks long-poll `decode_resume`; the run resumes when no prefill holds or
awaits credits and its own credits are free again, or ends exactly like an
ordinary yield on cancel/timeout/fence or after 2 s parked. D wall-time
feedback then includes the parked time. In this mode a prefill also prepares
(KV reservation, PREPARE/READY handshake) without credits while decode runs
keep going; it takes all credits at START. Runs park only for a READY
prefill, and no new decode batch is prepared while one waits.

`prefill_cuda_graph: true` enables SGLang's breakable prefill CUDA graph on
P ranks for prompts of exactly 512 tokens (`launcher.PREFILL_GRAPH_TOKENS`;
other lengths stay eager). The transformer body is captured in segments at
model-runner init; Head-wise attention runs eagerly at every graph break.
Capture writes no KV. On replay the live batch was already validated by the
grant check, the runner's static batch must carry exactly that grant's
geometry, and K/V go to the grant's write slots.

Opt-in control-plane fast paths (all off by default): `inline_start` lets the
controller START a batch as soon as it is startable instead of on the driver's
next pass; `decode_skip_prefill_ranks` runs decode batches with the D ranks
only (P ranks no longer prepare/quiesce decode batches they never execute);
`prefill_early_token` publishes the prefill token once every P rank's result
is committed, as original MuxServe returns it at prefill end (it moves the
TTFT boundary; the request is not shorter). `tools/run_perf_point.sh` takes
`INLINE_START=1 SKIP_P=1 EARLY_TOKEN=1`, and `FROZEN_BUNDLE=... WARMUP_TOTAL=128`
to bench original MuxServe's frozen 512/128 inputs with its 43/43/42 warmup,
checking every output against the original's tokens.

## Known limitations

Prefill CUDA graphs (unless `prefill_cuda_graph`), overlap scheduling and custom all-reduce remain disabled.
Each run start still ships and validates the complete multi-rank KV view once
(as packed int32 tables). The sample
configuration uses D30 for 7B and D40 for 13B/30B; it is not the all-D40 paired
experiment configuration. Asset paths now use the common protected resolver,
with no hardcoded lab model aliases. No original-paper performance reproduction
or clean-install GPU validation is claimed for this migration.

This backend excludes the old experiment transport harness,
CrossPool client snapshot, raw results, reports, slide assets and delivery
archives. The retained regression sources cover the migrated runtime; the
full experiment matrix and CrossPool/xbench wrapper are separate work.
