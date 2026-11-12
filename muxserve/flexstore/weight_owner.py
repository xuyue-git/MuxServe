"""One same-node, per-model/rank producer for native SGLang zero-copy weights.

The native daemon caches one serialized CUDA reduction and reuses it across
consumers. Torch CUDA IPC refcounters belong to one import, so this owner
reduces the original tensor once per authorized P/D process instead.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import signal
import socket
import stat
import struct
import time

from sglang.srt.weight_cache.daemon import WeightCacheDaemon
from sglang.srt.weight_cache.protocol import CacheConfig, recv_msg, send_msg
from sglang.srt.weight_cache.transport import TorchIpcTransportBackend
from sglang.srt.utils import MultiprocessingSerializer

from muxserve.config import safe_path
from muxserve.contracts import IntegrationError

logger = logging.getLogger(__name__)


class _OneReduction:
    """Pickle the already exported reduction without making a second one."""
    def __init__(self, rebuild, args):
        self.rebuild, self.args = rebuild, args

    def __reduce__(self):
        return self.rebuild, self.args


class SharedWeightOwner(WeightCacheDaemon):
    def __init__(self, server_args, gpu_id, tp_rank, socket_path, control, dist_init_method):
        super().__init__(server_args, gpu_id, tp_rank, 0, dist_init_method)
        path = safe_path(socket_path)
        parent = path.parent
        mode = stat.S_IMODE(parent.stat().st_mode)
        if not parent.is_dir() or parent.is_symlink() or parent.stat().st_uid != os.getuid() or mode & 0o077:
            raise IntegrationError("UNSAFE_ADDRESS", "Weight socket requires a private owned directory")
        if len(os.fsencode(path)) > 100 or path.exists() or path.is_symlink():
            raise IntegrationError("UNSAFE_ADDRESS", "Weight socket path is occupied or too long")
        self.socket_path = str(path)
        self.ready_path = str(path.with_suffix(".ready"))
        if safe_path(self.ready_path).exists():
            raise IntegrationError("UNSAFE_ADDRESS", "Weight owner ready path already exists")
        self.control = control
        self.authorized = {}
        self.exited = set()
        self.exports = {}
        self.allocation_evidence = {}
        self._safe_close = False
        self._interrupted = False
        self._loading_collectives_released = False

    def load(self):
        super().load()
        self._release_loading_collectives()

    def _release_loading_collectives(self):
        """Drop owner-only loading groups before any CUDA IPC consumer exists."""
        import torch
        import torch.distributed as dist
        from sglang.srt.distributed.parallel_state import (
            destroy_distributed_environment, destroy_model_parallel, get_tp_group,
        )
        from muxserve.flexstore.weight_loader import _local_alias

        if self.authorized or self.exports or self._loading_collectives_released:
            raise IntegrationError("UNSAFE_WEIGHT_CLOSE", "Loading groups must close before consumer authorization")
        # Only this frozen topology has one non-singleton native PyNccl group.
        # P/D groups belong to other processes and are never touched here.
        if (self.tp_size != 4 or self.pp_size != 1 or self.dp_size != 1
                or self.ep_size != 1 or self.attn_cp_size != 1 or self.moe_dp_size != 1
                or self.enable_dp_attention):
            raise IntegrationError("UNSUPPORTED_WEIGHT_TOPOLOGY", "Owner group release requires the reviewed TP4 topology")
        model, tensors = self.model, self.state_tensors
        if model is None or not tensors or not dist.is_initialized():
            raise IntegrationError("WEIGHT_EVIDENCE_MISSING", "Weights and loading groups must exist before release")

        def model_aliases():
            return ({name: _local_alias(tensor) for name, tensor in model.named_parameters(remove_duplicate=False)},
                    {name: _local_alias(tensor) for name, tensor in model.named_buffers()})

        aliases = {name: (_local_alias(tensor), is_param) for name, (tensor, is_param) in tensors.items()}
        live_aliases = model_aliases()
        group = get_tp_group()
        if group.cpu_group is None:
            raise IntegrationError("WEIGHT_EVIDENCE_MISSING", "Owner TP CPU barrier group is missing")
        # Every owner rank must finish disk loading and all post-load kernels.
        # Use Gloo rather than creating another GPU collective for this barrier.
        dist.barrier(group=group.cpu_group)
        with torch.cuda.device(self.gpu_id):
            torch.cuda.synchronize(self.gpu_id)
            before = dict(allocated=torch.cuda.memory_allocated(self.gpu_id),
                          reserved=torch.cuda.memory_reserved(self.gpu_id),
                          device_free=torch.cuda.mem_get_info(self.gpu_id)[0])
            raw = group.pynccl_comm
            closed = False
            if raw is not None and raw.available:
                # Frozen GroupCoordinator.destroy only drops this Python ref;
                # PyNcclCommunicator has no destructor for its native handle.
                raw.nccl.ncclCommDestroy(raw.comm)
                raw.comm, raw.disabled, raw.available = None, True, False
                group.pynccl_comm = None
                closed = True
            destroy_model_parallel()
            destroy_distributed_environment()
            torch.cuda.empty_cache()
            torch.cuda.synchronize(self.gpu_id)
            after = dict(allocated=torch.cuda.memory_allocated(self.gpu_id),
                         reserved=torch.cuda.memory_reserved(self.gpu_id),
                         device_free=torch.cuda.mem_get_info(self.gpu_id)[0])
        if (dist.is_initialized() or self.model is not model or self.state_tensors is not tensors
                or aliases != {name: (_local_alias(tensor), is_param)
                               for name, (tensor, is_param) in self.state_tensors.items()}
                or live_aliases != model_aliases()):
            raise IntegrationError("WEIGHT_ALIAS_CHANGED", "Loading group release changed weights or left groups active")
        self._loading_collectives_released = True
        print("WEIGHT_OWNER_COLLECTIVES_RELEASED " + json.dumps(dict(
            model=self.model_path, rank=self.tp_rank, pid=os.getpid(),
            tensors_verified=len(tensors), pynccl_closed=closed, torch_distributed_closed=True,
            memory_before=before, memory_after=after,
            memory_note="Device free memory includes concurrent processes; startup headroom gate remains authoritative")),
            flush=True)

    def _init_distributed(self, server_args, model_config):
        self._muxserve_model_config = model_config
        return super()._init_distributed(server_args, model_config)

    def _export_state(self):
        """Keep original tensors strongly; defer CUDA reductions until fetch."""
        import torch
        from sglang.srt.platforms import current_platform
        from sglang.srt.utils import reserve_rope_cache_for_long_sequences

        # ModelRunner reserves context plus its RoPE margin after IPC loading.
        # Do it on the owner first, so consumer reservation keeps the same buffer.
        before = torch.cuda.memory_reserved(self.gpu_id)
        reserve_rope_cache_for_long_sequences(self.model, self._muxserve_model_config, logger)
        current_platform.synchronize()
        current_platform.empty_cache()
        self.preloaded_weights_bytes = max(
            0, self.preloaded_weights_bytes + torch.cuda.memory_reserved(self.gpu_id) - before)
        param_names = {name for name, _ in self.model.named_parameters(remove_duplicate=False)}
        state = self.model.state_dict()
        self.state_tensors = {name: (tensor.data, name in param_names)
                              for name, tensor in state.items()}
        for name, tensor in self.model.named_buffers():
            if name not in self.state_tensors:
                self.state_tensors[name] = (tensor.data, False)
        self.state_entries = {}
        self.transport_backend = TorchIpcTransportBackend()

    def _peer_pid(self, conn):
        pid, uid, _ = struct.unpack("3i", conn.getsockopt(socket.SOL_SOCKET, socket.SO_PEERCRED, 12))
        if uid != os.getuid() or pid not in self.authorized or pid in self.exited:
            raise IntegrationError("UNAUTHORIZED_WEIGHT_IMPORT", "Weight importer PID/UID was not authorized")
        return pid

    def _fresh_entries(self, pid):
        from torch.multiprocessing.reductions import reduce_tensor, rebuild_cuda_tensor

        if pid in self.exports:
            raise IntegrationError("WEIGHT_IMPORT_REPLAY", "A weight importer cannot fetch twice")
        self.exports[pid] = "PREPARING"  # Any partial send remains protected.
        entries = {}
        evidence = {}
        for name, (tensor, is_param) in self.state_tensors.items():
            allocation_digest = refcounter_digest = None
            if tensor.device.type == "cuda":
                rebuild, args = reduce_tensor(tensor)
                if rebuild is not rebuild_cuda_tensor or len(args) != 15:
                    raise IntegrationError("CUDA_IPC_ABI", "Unexpected CUDA weight reduction ABI")
                handle = MultiprocessingSerializer.serialize(_OneReduction(rebuild, args), output_str=True)
                # Tensor offsets and CUDA allocation handles prove producer aliasing;
                # refcounter/event fields are per-consumer and must differ.
                allocation_digest = hashlib.sha256(repr((args[3], args[7], args[9])).encode()).hexdigest()
                refcounter_digest = hashlib.sha256(repr((args[11], args[12])).encode()).hexdigest()
                evidence[name] = allocation_digest
            else:
                # Native state_dict may include CPU metadata buffers. They do
                # not carry CUDA IPC refcounters or claim shared GPU storage.
                handle = MultiprocessingSerializer.serialize(tensor, output_str=True)
            entries[name] = {
                "handle": handle,
                "shape": list(tensor.shape), "dtype": str(tensor.dtype).replace("torch.", ""),
                "is_param": is_param, "name": name, "allocation_digest": allocation_digest,
                "refcounter_digest": refcounter_digest,
            }
        self.allocation_evidence[pid] = evidence
        return entries

    def _handle_connection(self, conn):
        pid = self._peer_pid(conn)
        request = recv_msg(conn)
        if request.get("type") == "query_config":
            send_msg(conn, {"status": "ok", "config": self.config.to_dict()})
            return
        if request.get("type") != "fetch_state":
            send_msg(conn, {"status": "error", "message": "Unsupported weight owner request"})
            return
        engine = CacheConfig.from_dict(request["config"])
        if not self.config.matches(engine):
            send_msg(conn, {"status": "mismatch", "daemon_config": self.config.to_dict()})
            return
        entries = self._fresh_entries(pid)
        self.transport_backend.send_fetch_state_response(
            conn, config=self.config.to_dict(), entries=entries, pid=os.getpid(),
            preloaded_weights_bytes=self.preloaded_weights_bytes)
        self.exports[pid] = "SENT"

    def _control(self):
        while self.control.poll():
            try:
                message = self.control.recv()
            except EOFError:
                self._interrupted = True
                return
            op = message.get("op")
            pid = message.get("pid")
            if op == "authorize":
                stage = message.get("stage")
                if stage not in ("P", "D") or type(pid) is not int or pid <= 0 or pid in self.authorized:
                    raise IntegrationError("INVALID_WEIGHT_CONSUMER", "Expected one fresh P/D rank PID")
                if stage in self.authorized.values():
                    raise IntegrationError("INVALID_WEIGHT_CONSUMER", "A weight owner permits one importer per stage")
                self.authorized[pid] = stage
                self.control.send({"ok": True, "op": op, "pid": pid})
            elif op == "exited":
                if pid not in self.authorized or message.get("exit_code") != 0:
                    raise IntegrationError("UNSAFE_WEIGHT_CLOSE", "Importer exit has no normal waitpid proof")
                self.exited.add(pid)
                self.control.send({"ok": True, "op": op, "pid": pid})
            elif op == "close":
                if (set(self.authorized.values()) != {"P", "D"}
                        or self.exited != set(self.authorized)
                        or "PREPARING" in self.exports.values()):
                    raise IntegrationError("UNSAFE_WEIGHT_CLOSE", "Weight consumers have not safely exited")
                self._safe_close = True
                self.control.send({"ok": True, "op": op})
            elif op == "snapshot":
                self.control.send({"ok": True, "op": op,
                                   "authorized": dict(self.authorized),
                                   "exited": sorted(self.exited),
                                   "exports": dict(self.exports),
                                   "allocation_evidence": self.allocation_evidence})
            else:
                raise IntegrationError("INVALID_WEIGHT_CONTROL", "Unknown weight owner control command")

    def serve(self):
        if not self._loading_collectives_released:
            raise IntegrationError("WEIGHT_EVIDENCE_MISSING", "Owner loading groups have not safely closed")
        listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        old_umask = os.umask(0o177)
        try:
            listener.bind(self.socket_path)
        finally:
            os.umask(old_umask)
        listener.listen(2)
        listener.settimeout(0.25)
        with open(self.ready_path, "x", encoding="ascii") as ready:
            ready.write(f"pid={os.getpid()}\n")
        os.chmod(self.ready_path, 0o600)

        def interrupted(_signum, _frame):
            self._interrupted = True

        signal.signal(signal.SIGTERM, interrupted)
        signal.signal(signal.SIGINT, interrupted)
        try:
            while not self._safe_close:
                try:
                    self._control()
                except Exception:
                    logger.exception("Weight owner control failed; retaining producer")
                    self._interrupted = True
                if self._interrupted:
                    # A lost supervisor/interrupt cannot prove imported CUDA
                    # storage has disappeared. Keep producer references alive.
                    time.sleep(1)
                    continue
                try:
                    conn, _ = listener.accept()
                except socket.timeout:
                    continue
                except OSError:
                    logger.exception("Weight listener failed; retaining producer")
                    self._interrupted = True
                    continue
                with conn:
                    conn.settimeout(30)
                    try:
                        self._handle_connection(conn)
                    except Exception:
                        logger.exception("Weight import failed; retaining any uncertain reduction")
        finally:
            listener.close()
            if self._safe_close:
                os.unlink(self.socket_path)
                os.unlink(self.ready_path)

    def close_after_safe_drain(self):
        if (not self._safe_close or set(self.authorized.values()) != {"P", "D"}
                or self.exited != set(self.authorized) or "PREPARING" in self.exports.values()):
            raise IntegrationError("UNSAFE_WEIGHT_CLOSE", "Missing all-consumer close proof")
        import gc
        import torch
        from sglang.srt.layers.rotary_embedding.factory import _ROPE_DICT
        torch.cuda.synchronize(self.gpu_id)
        # Drop both strong ownership paths before Torch's global IPC teardown.
        # Consumer normal-exit proofs above remain mandatory.
        _ROPE_DICT.clear()
        self.state_tensors.clear()
        self.shutdown()
        gc.collect()
        torch.cuda.ipc_collect()
        torch.cuda.empty_cache()
        print("WEIGHT_OWNER_CLOSED " + json.dumps(dict(model=self.model_path,
              rank=self.tp_rank, pid=os.getpid(), consumers_exited=sorted(self.exited))), flush=True)


def run_weight_owner(server_args, gpu_id, tp_rank, socket_path, control, dist_init_method):
    """Spawn target. Caller strips execution-group env before process import."""
    owner = SharedWeightOwner(server_args, gpu_id, tp_rank, socket_path, control, dist_init_method)
    owner.load()
    try:
        owner.serve()
    except BaseException:
        if owner.exports and not owner._safe_close:
            logger.exception("Weight producer fault after IPC export; keeping weights mapped")
            while True:
                time.sleep(60)
        raise
    owner.close_after_safe_drain()
