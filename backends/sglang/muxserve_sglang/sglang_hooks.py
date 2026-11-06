"""Small, explicit entry points called by the pinned SGLang fork."""
from __future__ import annotations

from functools import lru_cache
import os
from pathlib import Path

from .config import load_config, safe_path, visible_devices
from .contracts import IntegrationError
from .ipc import StorageRPCClient


def enabled() -> bool:
    return bool(os.environ.get("MUXSERVE_EXECUTION_GROUP"))


@lru_cache(maxsize=1)
def context():
    if not enabled():
        raise IntegrationError("EXECUTION_CONTEXT_MISSING", "Use the project launcher")
    from types import SimpleNamespace
    config = load_config(os.environ["MUXSERVE_CONFIG"])
    model = config.model(os.environ["MUXSERVE_MODEL_ID"])
    stage = os.environ["MUXSERVE_STAGE"]
    if stage not in ("P", "D"):
        raise IntegrationError("INVALID_STAGE", "Expected Prefill or Decode stage")
    visible_devices(config.tp_size)
    runtime_dir = safe_path(os.environ["MUXSERVE_RUNTIME_DIR"])
    key_path = runtime_dir / "auth.key"
    rpc = StorageRPCClient(str(runtime_dir / "store.sock"), key_path.read_bytes(),
                           timeout_s=config.prepare_timeout_ms / 1000)
    return SimpleNamespace(config=config, model=model, stage=stage,
                           group_id=os.environ["MUXSERVE_EXECUTION_GROUP"],
                           runtime_dir=runtime_dir, rpc=rpc)


def allocate_shared_memory(runner, memory_pool_config=None):
    import torch
    from sglang.srt.disaggregation.decode import DecodeReqToTokenPool
    from sglang.srt.mem_cache.allocator.token import TokenToKVPoolAllocator
    from sglang.srt.mem_cache.memory_pool import ReqToTokenPool
    from sglang.srt.model_executor.pool_configurator import MemoryPoolConfig
    from .gpu_storage import rebuild_pool
    from .sglang_memory import HeadwiseTokenToKVPool

    ctx = context()
    if (runner.device != "cuda" or runner.is_draft_worker or runner.ps.pp_size != 1
            or runner.ps.tp_size != ctx.config.tp_size or runner.dtype != torch.float16
            or runner.page_size != 1 or not runner.spec_algorithm.is_none()):
        raise IntegrationError("UNSUPPORTED_EXECUTION", "Initial shared path requires CUDA FP16 TP/PP1/page1")
    rank = runner.ps.tp_rank
    consumer = f"{ctx.group_id}:{ctx.stage}:rank{rank}"
    descriptor = ctx.rpc.call("export_pool", rank, consumer)
    imported = rebuild_pool(descriptor, device_ordinal=runner.gpu_id)
    pool_key, pool_value = imported.key, imported.value
    if pool_key.shape[0] != ctx.config.pool_blocks_per_rank or pool_value.shape[0] != ctx.config.pool_blocks_per_rank:
        raise IntegrationError("LAYOUT_MISMATCH", "Every rank must import the configured uniform physical pool size")
    running_slots = ctx.config.max_model_inflight
    extra_slots = 2 if ctx.stage == "D" else 0
    # These slots are logical IDs. Global physical bytes are reserved separately.
    # Native Scheduler keeps one context slot and its allocator keeps one
    # additional admission slot. These are logical IDs, not physical KV bytes.
    # The controller still caps confirmed request context at 2048 tokens.
    logical_tokens = (running_slots + extra_slots) * (ctx.config.max_context_tokens + 2)
    pool = HeadwiseTokenToKVPool(ctx.model.layout(ctx.config.tp_size), rank,
                                pool_key, pool_value, logical_tokens)
    if ctx.stage == "D":
        request_pool = DecodeReqToTokenPool(running_slots, ctx.config.max_context_tokens + 8,
                                          runner.device, False, extra_slots)
    else:
        request_pool = ReqToTokenPool(running_slots, ctx.config.max_context_tokens + 8,
                                    runner.device, False)
    runner.req_to_token_pool = request_pool
    runner.token_to_kv_pool = pool
    runner.token_to_kv_pool_allocator = TokenToKVPoolAllocator(
        logical_tokens, torch.float16, runner.device, pool, need_sort=False)
    runner.max_total_num_tokens = logical_tokens
    runner.max_running_requests = running_slots
    runner.memory_pool_config = MemoryPoolConfig(logical_tokens, running_slots)
    runner._unified_memory_pool = None
    runner._muxserve_consumer_id = consumer
    runner._muxserve_imported_pool = imported
    runner._muxserve_import_pid = os.getpid()
    runner._muxserve_import_state = "MAPPED"
    runner._muxserve_permit = None
    runner._init_post_memory_pool_components()


def assert_forward_permitted(runner, forward_batch):
    if not enabled():
        return
    permit = getattr(runner, "_muxserve_permit", None)
    if permit is None or permit.get("consumed"):
        raise IntegrationError("FORWARD_NOT_GRANTED", "Every actual forward requires a fresh batch grant")
    runner.token_to_kv_pool.validate_forward(forward_batch)
    import torch
    permit["gpu_start_event"] = torch.cuda.Event(enable_timing=True)
    permit["gpu_end_event"] = torch.cuda.Event(enable_timing=True)
    permit["gpu_timing_finished"] = False
    permit["gpu_start_event"].record()
    permit["consumed"] = True


def finish_forward_timing(runner):
    """Record the granted ModelRunner.forward stream boundary, before sampling."""
    if not enabled():
        return
    permit = getattr(runner, "_muxserve_permit", None)
    if not permit or not permit.get("consumed") or permit.get("gpu_timing_finished"):
        raise IntegrationError("FORWARD_NOT_GRANTED", "Forward timing lacks a fresh consumed grant")
    permit["gpu_end_event"].record()
    permit["gpu_timing_finished"] = True


def run_controlled_scheduler(scheduler):
    import weakref
    from .sglang_runtime import ControlledScheduler
    from .weight_loader import record_runtime_evidence
    ctx = context()
    record_runtime_evidence(scheduler, ctx)
    runner = scheduler.tp_worker.model_runner
    ctx._drain_complete = False
    ctx._consumer_cleanup = dict(model=weakref.ref(runner.model),
        pool=weakref.ref(runner.token_to_kv_pool), imported=runner._muxserve_imported_pool)
    ControlledScheduler(scheduler, ctx).run()
    print("CONTROLLED_DRAIN_COMPLETE", flush=True)
    # Native watchdog callbacks strongly retain Scheduler, model and KV. Stop
    # both monitors only after controlled DRAIN, before native host teardown.
    for name in ("watchdog", "soft_watchdog"):
        watchdog = getattr(scheduler, name, None)
        if watchdog is not None:
            watchdog.stop()
    close_native_channels(scheduler)
    ctx._drain_complete = True


def close_native_channels(scheduler):
    """Close this scheduler's private channels before GC can terminate ZMQ.

    The controlled loop has drained and has no native tokenizer/RPC peer. Only
    SchedulerIpcChannels' dedicated Context(2) is terminated; publishers may
    use Context.instance(), whose other users must remain untouched.
    """
    import json
    print("NATIVE_CHANNELS_CLOSE_BEGIN", flush=True)
    channels = scheduler.ipc_channels
    sockets = [channels.recv_from_tokenizer, channels.recv_from_rpc,
               channels.send_to_tokenizer.socket, channels.send_to_detokenizer.socket,
               channels.send_metrics_from_scheduler]
    sockets = {id(socket): socket for socket in sockets if socket is not None}
    contexts = {id(socket.context): socket.context for socket in sockets.values()}
    for socket in sockets.values():
        if not socket.closed:
            socket.close(linger=0)
    for owned_context in contexts.values():
        if not getattr(owned_context, "closed", False):
            owned_context.term()
    writer = getattr(scheduler, "load_snapshot_writer", None)
    if writer is not None:
        writer.close()
        scheduler.load_snapshot_writer = None
    publisher = getattr(scheduler, "load_publisher", None)
    if publisher is not None:
        publisher.close()
    result = dict(sockets=len(sockets), contexts=len(contexts))
    print("NATIVE_CHANNELS_CLOSED " + json.dumps(result), flush=True)
    return result


def release_process_imports(gpu_id):
    """Called only after the native scheduler and its host cleanup return."""
    import gc
    import json
    import torch
    from sglang.srt.layers.rotary_embedding.factory import _ROPE_DICT

    ctx = context()
    cleanup = getattr(ctx, "_consumer_cleanup", None)
    if not getattr(ctx, "_drain_complete", False) or cleanup is None:
        raise IntegrationError("UNSAFE_WEIGHT_CLOSE", "Consumer lacks a completed controlled drain")
    imported = cleanup["imported"]
    rank = imported.metadata.rank
    if imported.closed or rank != gpu_id:
        raise IntegrationError("UNSAFE_WEIGHT_CLOSE", "Consumer import identity changed before close")
    torch.cuda.synchronize(gpu_id)
    # Each P/D process hosts one model. The native global factory otherwise
    # retains its RoPE module and imported CUDA storage past model destruction.
    _ROPE_DICT.clear()
    print("CONSUMER_TEARDOWN_PHASE gc_begin", flush=True)
    gc.collect()
    print("CONSUMER_TEARDOWN_PHASE gc_done", flush=True)
    if cleanup["model"]() is not None or cleanup["pool"]() is not None:
        raise IntegrationError("CONSUMER_ALIASES_REMAIN",
            "Native model/KV adapter still holds imported tensors "
            f"(model_alive={cleanup['model']() is not None}, kv_adapter_alive={cleanup['pool']() is not None})")
    imported.close()
    gc.collect()
    torch.cuda.ipc_collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize(gpu_id)
    proof = dict(model_id=ctx.model.model_id, stage=ctx.stage, rank=rank, pid=os.getpid(),
                 model_collected=True, kv_adapter_collected=True, pool_closed=imported.closed)
    path = ctx.runtime_dir / f"{ctx.model.model_id}-{ctx.stage}-r{rank}-closed.json"
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as output:
        json.dump(proof, output, sort_keys=True)
    ctx._consumer_cleanup = None
    print("CONSUMER_IMPORTS_CLOSED " + json.dumps(proof), flush=True)


def shared_weight_loader(load_config):
    from .weight_loader import SharedWeightLoader
    from sglang.srt.runtime_context import get_parallel
    ctx = context()
    rank = get_parallel().tp_rank
    socket_path = ctx.runtime_dir / f"{ctx.model.model_id}-w{rank}.sock"
    # Native daemon mode forbids its disk fallback. The launcher starts the
    # owner explicitly; this does not ask SGLang to spawn a second owner.
    return SharedWeightLoader(load_config, socket_path=str(socket_path),
                              weight_cache_mode="daemon")
