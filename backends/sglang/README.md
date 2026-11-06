# Optional SGLang Head-wise backend

This directory contains the existing SGLang backend, separated from the
unchanged upstream MuxServe/vLLM implementation. It adds shared weights,
Head-wise KV storage and externally controlled prefill/decode execution.

Source: [MuxServe-SGLang-Head_wise_cache at a512d05](https://github.com/xuyue-git/MuxServe-SGLang-Head_wise_cache/tree/a512d05496122521fb16fa620aa44d8f4d4d3d18).
SGLang is pinned to **v0.5.20**, commit `94602c9c2b7cbdb8efd5c52802dac6a1c180089e`.
See [the original-to-backend mapping](MIGRATION.md) and [source notices](NOTICE).

## Scope

- Shared physical Head-wise KV pools and model weights via CUDA IPC.
- Externally granted prefill/decode execution and request lifecycle control.
- Native SGLang integration through 13 pinned upstream source patches.
- Private MPS startup, admission checks and ordered shutdown.

## Build and launch

Run backend commands from this directory. Keep its environment separate from
the legacy root project. The prototype targets Linux, Python 3.10, four reserved
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

## Known limitations

The copied runtime retains its existing per-token control overhead. KV
mapping caching and RPC optimizations have not been implemented. The sample
configuration uses D30 for 7B and D40 for 13B/30B; it is not the all-D40 paired
experiment configuration. Asset paths now use the common protected resolver,
with no hardcoded lab model aliases. No original-paper performance reproduction
or clean-install GPU validation is claimed for this migration.

This backend excludes the old experiment transport harness,
CrossPool client snapshot, raw results, reports, slide assets and delivery
archives. The retained regression sources cover the migrated runtime; the
full experiment matrix and CrossPool/xbench wrapper are separate work.
