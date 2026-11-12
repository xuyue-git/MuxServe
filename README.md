# MuxServe: Flexible Spatial-Temporal Multiplexing for LLM Serving [[paper](https://arxiv.org/abs/2404.02015)]

MuxServe is an efficient multiple-LLM serving system with flexible
spatial-temporal multiplexing. It colocates LLMs considering their popularity to
multiplex memory resources, and disaggregates and flexibly colocates prefill and
decoding phases leveraging their characteristics to multiplex computation
resources.

This repository runs MuxServe on **SGLang v0.5.20** (commit
`94602c9c2b7cbdb8efd5c52802dac6a1c180089e`). vLLM is neither used nor installed:
the vLLM-based execution, memory manager and scheduler were replaced module by
module inside the `muxserve` package (see [docs/MIGRATION.md](docs/MIGRATION.md)).

## Architecture

```text
muxserve/
  launch.py        python -m muxserve.launch CONFIG: private MPS, weight owners,
                   per-model prefill/decode ranks, owner RPC, HTTP gateway
  config.py        DeploymentConfig (what launch runs) and the placement configs
  muxsched/        scheduling: controller (request/batch lifecycle, decode runs),
                   policy (ADBS: quotas, 8 s re-sizing, SM credits), driver,
                   events, placement optimizer, SM accounting, workload generator
  flexstore/       unified memory: Head-wise KV store, CUDA storage, shared model
                   weights (one owner process per model), owner RPC
  flexserver/      execution on SGLang: rank runtime, Head-wise attention and
                   memory pool, P/D handoff, hooks used by the patched SGLang
  gateway.py       native-SGLang /generate SSE gateway (gateway_proc.py: child process)
csrc/headwise/     Head-wise CUDA kernels, built as muxserve._headwise
configs/           deployment configs
upstream-reference/ pinned SGLang source manifest and the 13 recorded patches
tools/             bootstrap/patch SGLang, preflight locks, perf runs, CPU checks
acceptance/        CPU checks (storage, control, transport, simulated 24-rank e2e)
benchmark/         xbench catalogue (standard benchmarks), paired fixed point,
                   original placement/workload tooling
```

## Installation

Linux, Python 3.10, four A100 40 GB GPUs, CUDA 13.2. Use a dedicated environment
that contains SGLang and no vLLM.

1. Materialize the pinned SGLang source: `python tools/bootstrap_source.py`
   (offline: `--archive /path/to/sglang-v0.5.20-source.tar.gz`; size and SHA-256
   are checked).
2. Apply the recorded hooks: `python tools/patch_sglang.py`, then install
   `third_party/sglang/python` and `pip install -r requirements.txt`.
3. Build this package and its CUDA extension: `pip install -e .`
4. Fill in model/tokenizer paths and a personal `asset_lock_dir` in
   `configs/sglang_a100.json`, then lock the assets:
   `python tools/lock_assets.py configs/sglang_a100.json`.

## Launch

```bash
export CUDA_VISIBLE_DEVICES=1,2,3,4
python -m muxserve.preflight configs/sglang_a100.json
python -m muxserve.launch configs/sglang_a100.json
```

The launcher starts a private MPS controller. The gateway serves `/health` and
`POST /models/<model_id>/generate` (native SGLang request and SSE stream format)
on the configured `controller_port`.

## Benchmarks

- [`benchmark/xbench/`](benchmark/xbench/README.md): the standard benchmark. A
  CrossPool xbench catalogue (`benches.toml`) with client-mode cases that state
  the arrival process, rates, duration, input and output lengths, and the one
  command that runs each case against a launched server.
- [`benchmark/paired/`](benchmark/paired/README.md): the fixed 512/128 point paired
  with original MuxServe's frozen inputs and expected outputs.
- `benchmark/end_to_end`, `benchmark/chatlmsys`, `muxserve/muxsched/workload_utils.py`
  and `muxserve/muxsched/placement.py`: the paper's workload generator (power-law
  popularity `alpha`, Poisson arrivals, ShareGPT lengths) and placement optimizer.

Runtime switches and their measured effects are documented in
[docs/runtime.md](docs/runtime.md).

## CPU checks

```bash
python -I -B tools/check_cpu.py
```

Linux Python 3.10+ (WSL works) with `numpy` and `pyzmq`. Without the materialized
SGLang source, three checks are skipped (`--require-source` makes that an error).
`acceptance/no_vllm_check.py` fails if any vLLM import or requirement appears.
These checks do not load weights or validate CUDA execution or performance.

## Citation

```
@article{duan2024muxserve,
  title={MuxServe: Flexible Multiplexing for Efficient Multiple LLM Serving},
  author={Duan, Jiangfei and Lu, Runyu and Duanmu, Haojie and Li, Xiuhong and Zhang, Xingcheng and Lin, Dahua and Stoica, Ion and Zhang, Hao},
  journal={arXiv preprint arXiv:2404.02015},
  year={2024}
}
```
