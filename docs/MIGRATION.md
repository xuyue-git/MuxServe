# Changes relative to original MuxServe

Original MuxServe executes on a modified vLLM. In this repository every vLLM-based
module was replaced inside the `muxserve` package by an SGLang-based module with
the same role; the package keeps MuxServe's structure (`muxsched` scheduling,
`flexstore` unified memory, `flexserver` execution). No module imports vLLM and
vLLM is not a dependency (`acceptance/no_vllm_check.py`). Original MuxServe itself
is run from its own repository and environment when it is needed as a baseline.

## History and source

- Original fork baseline: `2f7172a973a89ebef4b2a3142845bebe1b7b100a`.
- Original tested MuxServe: `9fb0fa555bb4051bee612b72938e5d85796dffec`.
  These trees differ only in the README and license, not the runtime.
- Migrated project: [xuyue-git/MuxServe-SGLang-Head_wise_cache](https://github.com/xuyue-git/MuxServe-SGLang-Head_wise_cache/tree/a512d05496122521fb16fa620aa44d8f4d4d3d18),
  commit `a512d05496122521fb16fa620aa44d8f4d4d3d18`.
- SGLang: `94602c9c2b7cbdb8efd5c52802dac6a1c180089e`, v0.5.20.
- Imported CUDA code: MuxServe-vLLM `eaca02eb1a8c3654fa086594e271f4e1e09dede4`.
  Its notices, license and source manifest are retained in `native/headwise`.

The fork retains its original commits. New commits separate build, storage,
scheduling, native execution, startup, portable asset paths and validation.
Their bodies identify the source repository/commit. These are present-day
migration commits, not reconstructed or backdated development history.
The older project's independent history remains in that repository.

To inspect only this migration, run:

```bash
git log --reverse --oneline 2f7172a..HEAD
git diff --stat 2f7172a..HEAD
git diff 2f7172a..HEAD -- muxserve csrc
git diff 2f7172a..HEAD -- muxserve     # the three original files changed for reuse
```

## Functional mapping

| Original module (vLLM-based, removed) | Replacement in `muxserve` | Change |
|---|---|---|
| `flexserver/llm_runtime.py`, `pipeworker.py`, `muxserve_server.py`, `models/`, `sampler.py`, `model_loader.py`, `dist_utils.py`, `p2p_communication.py` | `flexserver/sglang_runtime.py`, `sglang_hooks.py`, `sglang_memory.py`, `sglang_attention.py`, `sglang_pd.py`, `step_box.py`, `ops.py` | Native SGLang execution with explicit forward grants; Head-wise attention and P/D handoff on SGLang |
| `flexstore/manager.py`, `weight_utils.py`, `csrc/memory_manager.*`, `zmq_utils.py` | `flexstore/storage.py`, `gpu_storage.py`, `weight_owner.py`, `weight_loader.py`, `ipc.py` | Head-wise KV store in the owner process; one weight owner per model shares weights with its P/D ranks over CUDA IPC; authenticated local RPC instead of ZMQ |
| `muxsched/scheduler.py`, `csrc/batch_scheduler.*`, `shm_utils.py`, `constants.py`, `tracer.py` | `muxsched/controller.py`, `policy.py`, `driver.py`, `events.py`; `muxserve/trace.py` | ADBS rules from the original scheduler (quotas, 8 s re-sizing, SM credits); request/batch lifecycle and decode runs |
| `launch.py`, `entrypoint.py`, `arg_utils.py`, `muxsched/launcher.py`, `scripts/*.sh` | `launch.py`, `startup.py`, `preflight.py`, `mps_shutdown.py`, `config.py` (`DeploymentConfig`) | One launch command with a JSON config; private MPS; explicit assets/devices; ordered shutdown |
| in-process workload replay | `gateway.py`, `gateway_proc.py` | Native SGLang `/generate` SSE gateway |
| `vllm.transformers_utils.tokenizer.get_tokenizer` | `sglang.srt.utils.hf_transformers.tokenizer.get_tokenizer` | Same signature; imported on use in `muxsched/workload_utils.py` |

Kept unchanged in behaviour: `muxsched/resource.py` (`SMResource`, `num_gpu_blocks`),
`muxsched/placement.py`, `muxsched/workload_utils.py` (arrival processes and ShareGPT
sampling), `config.py` `JobConfig`/`MuxServeConfig`, `logger.py`.
`policy.CreditLedger` gives each TP rank one `SMResource` (`overload_threshold` =
`credit_multiplier`, `overload=True` as in the original scheduler), and
`config.kv_bytes_from_fraction` takes its block count from `num_gpu_blocks`.

Removed with the vLLM runtime: the spatial/temporal baseline launch scripts and the
vLLM profiling benchmark (`benchmark/parallelism/benchmark_parallelism.py`); their
placement configs and plotting tools remain. The Head-wise CUDA sources in
`csrc/headwise` derive from MuxServe-vLLM and keep their notices; they build into
`muxserve._headwise` without linking any vLLM package.

## Deliberate changes during migration

One `setup.py` builds the package and `muxserve._headwise`. The source materializer pins
an archive and verifies its contents before applying 13 recorded SGLang file
patches; the downloaded upstream tree is not committed.

Historical machine-specific model alias exceptions were removed from `config.py`.
Explicit asset paths now use `safe_path`, including its forbidden-directory and
symlink-hop checks. Personal lock names use the canonical asset root. The two
alias-related deployment cases were updated and a focused regression covers
canonical resolution, external locks and rejection before prohibited access.
Other migrated runtime behavior is unchanged, apart from text line endings.

## Validation boundary

CPU checks cover allocation/ownership, scheduling, grants and replay,
transport deadlines, startup/shutdown, native-channel closure and pinned-source
bootstrap contracts. They use temporary fixtures and simulated rank reports
only within test code. No GPU correctness, capacity or performance claim follows.

Historical GPU acceptance belongs to the prior project and its harness. This
branch has not been clean-installed. Later commits reduce the per-token mapping
and control overhead. `tools/run_perf_point.sh` with `tools/bench_fixed.py`
runs the paired four-GPU point (LLaMA 7B/13B/30B, TP4, FP16, 512/128, one
request in flight per model) on original MuxServe's frozen inputs, checking
every output against the original's tokens; measured results are kept outside
this repository. SGLang timings are taken at the HTTP client.
The sample D30/D40 configuration is not the all-D40 fixed synthetic comparison.
The standard xbench catalogue is in `benchmark/xbench/`.
