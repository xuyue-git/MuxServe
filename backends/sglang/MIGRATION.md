# Changes relative to original MuxServe

The original implementation remains under the repository root. This optional
backend has its own build configuration and dependencies under `backends/sglang`.
It does not silently change the original launcher into an SGLang launcher.
It imports and calls original code where that code applies without vLLM: SM
accounting and KV block sizing (see "Reused original code").

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
git diff 2f7172a..HEAD -- backends/sglang
git diff 2f7172a..HEAD -- muxserve     # the three original files changed for reuse
```

## Functional mapping

| Original location | Backend counterpart | Change |
|---|---|---|
| `muxserve/flexserver/llm_runtime.py`, `pipeworker.py`, model execution | `sglang_runtime.py`, `sglang_hooks.py` | Use fixed native SGLang execution with explicit forward grants |
| `muxserve/muxsched/scheduler.py` | `policy.py`, `controller.py`, `driver.py`, `events.py` | Source-derived ADBS rules, request/batch state and completion protocol |
| `muxserve/muxsched/resource.py` `SMResource` / `SMPart` | `policy.py` `CreditLedger` | Reused: one original `SMResource` per TP rank allocates and frees the 10 SM parts |
| `get_num_gpu_blocks` (`flexstore/manager.py`, `muxsched/resource.py`) | `config.kv_bytes_from_fraction` | Reused: original `muxsched.resource.num_gpu_blocks`; bytes unchanged |
| `muxserve/flexstore/manager.py`, `csrc/memory_manager.*` | `storage.py`, `gpu_storage.py`, `contracts.py` | Separate logical allocation, physical CUDA storage, versions and leases |
| FlexStore weight loading | `weight_owner.py`, `weight_loader.py` | Share model weight allocations between P/D via CUDA IPC |
| External MuxServe-vLLM attention and P/D communication | `sglang_memory.py`, `sglang_attention.py`, `sglang_pd.py`, `ops.py`, `native/headwise` | Adapt the original Head-wise layout and kernels to SGLang |
| `muxserve/launch.py`, launcher/config, MPS scripts | `launcher.py`, `startup.py`, `preflight.py`, `mps_shutdown.py`, `config.py` | Private MPS, explicit assets/devices, admission and ordered shutdown |
| Offline workload and original control IPC | `gateway.py`, `ipc.py` | HTTP/SSE gateway and authenticated local control messages |

Backend Python names are relative to `muxserve_sglang`. The ADBS rules are
derived from the original scheduler, but the control implementation is new,
not the original scheduler reused unchanged; SM accounting and KV block sizing
call the original code. Original placement and workload code has not been
automatically connected to the new gateway.

## Reused original code

`muxserve_sglang/__init__.py` appends the repository root to the import path,
so the launcher and every spawned process import this checkout's `muxserve`
(another installed copy is rejected). `acceptance/original_reuse_check.py`
checks that this imports no vLLM, Torch or transformers module.

- `policy.CreditLedger` gives each TP rank one original `SMResource`, created
  with `overload_threshold` = `credit_multiplier` (1 or 3) explicitly, and
  allocates with `overload=True` as the original scheduler does. The ledger
  keeps only the grant protocol: leases keyed by batch id (idempotent replay,
  changed contents rejected), one lease per (model, stage), all-or-nothing
  allocation across ranks. Like the original, an overloaded part is picked
  with `random.choice` (the replaced ledger took the first one); free and
  overloaded part counts, and so admission, are unchanged.
- `config.kv_bytes_from_fraction` takes its block count from the original
  `num_gpu_blocks` (head_size 128, block_size 32, FP16); the bytes equal the
  previous arithmetic.

To make these imports possible without vLLM or Torch, three original files
changed without changing what they compute: `muxserve/__init__.py` imports
`MuxServeArgs`/`MuxScheduler` on first use (PEP 562; `from muxserve import
MuxScheduler` works as before); `muxserve/config.py` imports Torch only to
resolve the default `model_dtype` of `JobConfig` (still `torch.float16`);
`muxserve/muxsched/resource.py` imports Torch/transformers inside
`get_gpu_memory` and `CacheResource`, and `get_num_gpu_blocks` calls the
module-level `num_gpu_blocks` holding its formula. `SMResource` is unchanged.

## Deliberate changes during migration

The original root build/requirements and runtime files are preserved. Backend
build paths remain relative to its own directory. The source materializer pins
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
Neither original-paper reproduction nor native xbench invocation is claimed.
