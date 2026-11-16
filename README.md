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
  cli.py           uv run muxserve serve CONFIG.toml | config dump CONFIG.toml
  launch.py        the launcher behind `serve`: private MPS, weight owners,
                   per-model prefill/decode ranks, owner RPC, HTTP gateway
  config.py        DeploymentConfig (what launch runs) and the placement configs
  muxsched/        scheduling: controller (request/batch lifecycle, decode runs),
                   policy (ADBS: quotas, 8 s re-sizing, SM credits), driver,
                   events, placement optimizer, SM accounting, workload generator
  flexstore/       unified memory: Head-wise KV store, CUDA storage, shared model
                   weights (one owner process per model and GPU), owner RPC
  flexserver/      execution on SGLang: rank runtime, Head-wise attention and
                   memory pool, P/D handoff, hooks used by the patched SGLang
  gateway.py       native-SGLang /generate SSE gateway (gateway_proc.py: child process)
csrc/headwise/     Head-wise CUDA kernels, built as muxserve._headwise
configs/           deployment configs (TOML), ablation steps, local settings template
upstream-reference/ separate-engine identity pin and the 13 recorded patch hashes
tools/             verify the separate engine, preflight locks, perf runs, CPU checks
acceptance/        CPU checks (storage, control, transport, simulated 24-rank e2e)
benchmark/         xbench catalogue (standard benchmarks), paired fixed point,
                   original placement/workload tooling
```

## Installation

Linux, four A100 40 GB GPUs, CUDA 13.2 (`nvcc`), [uv](https://docs.astral.sh/uv/).
The environment contains SGLang and no vLLM.

1. Clone the two repositories side by side; the engine directory name is fixed
   because `pyproject.toml` refers to `../MuxServe-SGLang-Engine`:

   ```bash
   git clone -b Muxserve-SGLang https://github.com/xuyue-git/MuxServe.git
   git clone https://github.com/xuyue-git/Muxserve-SGLang-Engine.git MuxServe-SGLang-Engine
   cd MuxServe
   ```

   See [the repository boundary notes](docs/two-repositories.md).
2. Verify the engine source: `python tools/bootstrap_source.py`.
3. Install both, the engine editable and this package with its CUDA extension:
   `uv sync`. (`pyproject.toml` pins the dependencies; `uv.lock` pins every
   version. Without uv: `pip install --config-settings editable_mode=compat -e
   ../MuxServe-SGLang-Engine`, `pip install -r requirements.txt`, then
   `pip install --no-build-isolation --config-settings editable_mode=compat -e .`.)
4. Machine-local settings stay out of the deployment configs, as CrossPool keeps
   them in `configs/xpool.local.toml` and `.env`:
   `cp .env.example .env && cp configs/muxserve.example.toml configs/muxserve.local.toml`
   (both git-ignored), set your personal `asset_lock_dir`, model paths and GPUs, and
   `export UV_ENV_FILE="$PWD/.env"` in every terminal. The local settings file accepts
   only those paths, so it cannot change an experimental condition;
   `uv run muxserve config dump CONFIG` prints the effective deployment.
5. Lock the assets: `uv run python tools/lock_assets.py configs/sglang_a100.toml`.

The server imports the installed `sglang` package normally. No source-path
injection or embedded engine fallback is used. The engine is maintained as
versioned source; its retained-file projection and reviewed modifications are
locked by `upstream-reference/engine-repository.json`.

## Launch

```bash
export UV_ENV_FILE="$PWD/.env"
uv run python -m muxserve.preflight configs/sglang_a100.toml
uv run muxserve serve configs/sglang_a100.toml
```

`uv run muxserve serve CONFIG` is `python -m muxserve.launch CONFIG`. KV pools are sized
as original MuxServe sizes them: `kv_memory_fraction` x the GPU memory torch reports
(0.05061 gives 2 GiB per GPU on the A100s used here; the launcher prints `KV_POOL`).

The launcher starts a private MPS controller. The gateway serves `/health` and
`POST /models/<model_id>/generate` (native SGLang request and SSE stream format)
on the configured `controller_port`.

## Benchmarks

- [`benchmark/xbench/`](benchmark/xbench/README.md): the standard benchmark. A
  CrossPool xbench catalogue (`benches.toml`) with client-mode cases that state
  the arrival process, rates, duration, input and output lengths, and the one
  command that runs each case against a launched server.
- [`benchmark/ablation/`](benchmark/ablation/README.md): the cumulative patch
  ablation, one complete server config per step (`configs/ablation/`), with each
  patch's motivation and the results table.
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
