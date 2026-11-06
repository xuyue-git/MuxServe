"""Real CUDA pool ownership and PyTorch CUDA IPC metadata.

Importing this module does not import torch or initialize CUDA. The owner must be
a long-lived process started with spawn. Consumers must destroy imported tensors
and synchronize their devices before releasing their leases; crashes are fenced.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import os
from threading import RLock
from typing import Any, Sequence
from uuid import uuid4

from .contracts import IntegrationError
from .storage import HeadwiseStore


@dataclass(frozen=True)
class CUDATensorMetadata:
    # Keep all arguments intact, including storage offset bytes, tensor offset,
    # ref-counter and CUDA event metadata. Never replace these with data_ptr().
    rebuild_args: tuple[Any, ...]
    visible_device: str
    torch_version: str


@dataclass(frozen=True)
class PoolMetadata:
    owner_id: str
    consumer_id: str
    export_id: str
    rank: int
    device_ordinal: int
    visible_device: str
    blocks: int
    torch_version: str
    key: CUDATensorMetadata
    value: CUDATensorMetadata


# Strong references deliberately outlive local variables and accidental GC.
# Only an explicit, safe close removes an owner from this registry.
_LIVE_OWNERS: dict[str, "GPUStorage"] = {}
_OWNER_REGISTRY_LOCK = RLock()
_IMPORTED_POOLS: dict[tuple[str, str], "ImportedPool"] = {}
_CLOSED_IMPORTS: set[tuple[str, str]] = set()
_PENDING_IMPORTS: set[tuple[str, str]] = set()
_IMPORT_LOCK = RLock()


def _visible_devices() -> tuple[str, ...]:
    raw = os.environ.get("CUDA_VISIBLE_DEVICES")
    if raw is None or not raw.strip():
        raise IntegrationError("DEVICE_VISIBILITY", "Explicit CUDA_VISIBLE_DEVICES is required")
    tokens = tuple(s.strip() for s in raw.split(","))
    if (len(set(tokens)) != len(tokens)
            or any(not s or not (s.isdecimal() or s.startswith("GPU-")) for s in tokens)):
        raise IntegrationError("DEVICE_VISIBILITY", "Expected explicit GPU ordinals or GPU UUIDs")
    return tokens


def export_tensor(tensor: Any) -> CUDATensorMetadata:
    """Export one original CUDA tensor; its owner must retain it until release.

    Rebuilding the same reduction twice can consume a ref counter twice. The
    caller must cache its descriptor and each consumer's rebuilt tensor, and
    maintain its own producer lease (GPUStorage does this for KV pools).
    """
    import torch
    from torch.multiprocessing.reductions import reduce_tensor, rebuild_cuda_tensor
    visible = _visible_devices()
    if not isinstance(tensor, torch.Tensor) or tensor.device.type != "cuda":
        raise IntegrationError("INVALID_ARGUMENT", "Original CUDA tensor required")
    ordinal = tensor.device.index
    if ordinal is None or ordinal >= len(visible):
        raise IntegrationError("DEVICE_VISIBILITY", "Tensor device is outside explicit visibility")
    rebuild, args = reduce_tensor(tensor)
    if rebuild is not rebuild_cuda_tensor or len(args) != 15:
        raise IntegrationError("CUDA_IPC_ABI", "Unsupported torch CUDA reduction ABI")
    return CUDATensorMetadata(tuple(args), visible[ordinal], str(torch.__version__))


def rebuild_tensor(descriptor: CUDATensorMetadata, device_ordinal: int | None = None) -> Any:
    """Rebuild once from authenticated owner metadata on the same physical GPU."""
    if not isinstance(descriptor, CUDATensorMetadata) or len(descriptor.rebuild_args) != 15:
        raise IntegrationError("CUDA_IPC_ABI", "Unsupported CUDA tensor descriptor")
    visible = _visible_devices()
    if device_ordinal is None:
        if descriptor.visible_device not in visible:
            raise IntegrationError("DEVICE_VISIBILITY", "Exported physical GPU is not visible")
        device_ordinal = visible.index(descriptor.visible_device)
    if (type(device_ordinal) is not int or not 0 <= device_ordinal < len(visible)
            or visible[device_ordinal] != descriptor.visible_device):
        raise IntegrationError("DEVICE_VISIBILITY", "Consumer ordinal selects a different physical GPU")
    import torch
    from torch.multiprocessing.reductions import rebuild_cuda_tensor
    if str(torch.__version__) != descriptor.torch_version:
        raise IntegrationError("CUDA_IPC_ABI", "Owner and consumer torch versions differ")
    if not torch.cuda.is_available():
        raise IntegrationError("CUDA_UNAVAILABLE", "Consumer CUDA is unavailable")
    args = list(descriptor.rebuild_args)
    args[6] = device_ordinal
    with torch.cuda.device(device_ordinal):
        return rebuild_cuda_tensor(*args)


class GPUStorage:
    def __init__(self, store: HeadwiseStore, rank_devices: dict[int, int]):
        visible = _visible_devices()
        pools = store.snapshot()["ranks"]
        if (set(rank_devices) != set(pools)
                or len(set(rank_devices.values())) != len(rank_devices)
                or any(type(d) is not int or not 0 <= d < len(visible) for d in rank_devices.values())):
            raise IntegrationError("DEVICE_VISIBILITY", "Each pool rank needs a distinct visible CUDA ordinal")
        import torch
        if not torch.cuda.is_available() or torch.cuda.device_count() != len(visible):
            raise IntegrationError("CUDA_UNAVAILABLE", "Assigned CUDA devices are not available")
        self._lock = RLock()
        self._store = store
        self._torch = torch
        self._visible = visible
        self._devices = dict(rank_devices)
        self.owner_id = uuid4().hex
        self._owner_pid = os.getpid()
        self._consumers: dict[str, dict[int, PoolMetadata | None]] = {}
        self._released: dict[str, set[int]] = {}
        self._closing = False
        self._closed = False
        self._buffers: dict[int, tuple[Any, Any]] = {}
        # Shape is the original MuxServe Head-wise K/V layout, one KV head
        # per physical block. Hardware compatibility preflight belongs to root.
        for rank, ordinal in self._devices.items():
            blocks = pools[rank]["total_blocks"]
            with torch.cuda.device(ordinal):
                key = torch.empty((blocks, 128 // 8, 16, 8), dtype=torch.float16,
                                  device=f"cuda:{ordinal}")
                value = torch.empty((blocks, 128, 16), dtype=torch.float16,
                                    device=f"cuda:{ordinal}")
                self._buffers[rank] = (key, value)
        with _OWNER_REGISTRY_LOCK:
            _LIVE_OWNERS[self.owner_id] = self

    def _check_process(self) -> None:
        if os.getpid() != self._owner_pid:
            raise IntegrationError("PROCESS_MISMATCH", "CUDA owner cannot be inherited through fork")
        if self._closed:
            raise IntegrationError("OWNER_CLOSED", "CUDA storage owner is closed")

    def export_pool(self, rank: int, consumer_id: str) -> PoolMetadata:
        """Idempotent export; each descriptor must be rebuilt once per consumer."""
        if not isinstance(consumer_id, str) or not consumer_id:
            raise IntegrationError("INVALID_ARGUMENT", "Consumer identity required")
        with self._lock:
            self._check_process()
            if self._closing:
                raise IntegrationError("OWNER_CLOSING", "CUDA owner is draining")
            if rank not in self._devices:
                raise IntegrationError("UNKNOWN_RANK", "Pool rank is not assigned")
            if consumer_id in self._released:
                raise IntegrationError("STALE_CONSUMER", "Released consumer identity cannot be reused")
            leases = self._consumers.setdefault(consumer_id, {})
            if rank in leases:
                if leases[rank] is None:
                    raise IntegrationError("EXPORT_FENCED", "Partial CUDA export requires owner recovery")
                return leases[rank]
            # Pin before reducing: an exception or lost response cannot trigger
            # producer teardown while IPC ref counters may still exist.
            leases[rank] = None
            key, value = self._buffers[rank]
            ordinal = self._devices[rank]
            exported = PoolMetadata(self.owner_id, consumer_id, uuid4().hex, rank, ordinal,
                self._visible[ordinal], key.shape[0], str(self._torch.__version__),
                export_tensor(key), export_tensor(value))
            leases[rank] = exported
            return exported

    def release_consumer(self, consumer_id: str, completed_ranks: Sequence[int]) -> None:
        """Caller certifies all imports destroyed and all its GPU work complete."""
        ranks = set(completed_ranks)
        with self._lock:
            self._check_process()
            if consumer_id in self._released:
                if ranks != self._released[consumer_id]:
                    raise IntegrationError("IDENTITY_CONFLICT", "Consumer release replay differs")
                return
            leases = self._consumers.get(consumer_id)
            if leases is None:
                raise IntegrationError("UNKNOWN_CONSUMER", "Consumer lease is unknown")
            if ranks != set(leases) or any(v is None for v in leases.values()):
                raise IntegrationError("CONSUMER_FENCED", "Every exported rank needs safe release proof")
            self._released[consumer_id] = ranks
            del self._consumers[consumer_id]

    def owner_snapshot(self) -> dict:
        with self._lock:
            return {"owner_id": self.owner_id, "closing": self._closing, "closed": self._closed,
                "rank_devices": dict(self._devices),
                "consumers": {c: {"ranks": sorted(ls),
                    "fenced_exports": sorted(r for r, value in ls.items() if value is None)}
                    for c, ls in self._consumers.items()},
                "store": self._store.snapshot()}

    def sharing_evidence(self) -> dict:
        """Describe existing exports without creating another CUDA refcounter."""
        with self._lock:
            self._check_process()
            evidence = {}
            for consumer, leases in self._consumers.items():
                evidence[consumer] = {}
                for rank, metadata in leases.items():
                    if metadata is None:
                        evidence[consumer][rank] = {"state": "UNCERTAIN"}
                        continue
                    def digest(tensor):
                        args = tensor.rebuild_args
                        return hashlib.sha256(repr((args[3], args[7], args[9])).encode()).hexdigest()
                    evidence[consumer][rank] = {
                        "state": "EXPORTED", "owner_id": metadata.owner_id,
                        "consumer_id": consumer, "export_id": metadata.export_id,
                        "rank": rank, "visible_device": metadata.visible_device,
                        "blocks": metadata.blocks,
                        "key_allocation_digest": digest(metadata.key),
                        "value_allocation_digest": digest(metadata.value)}
            return evidence

    def close(self) -> None:
        """Fail closed: retain producer tensors until all allocations and leases drain."""
        with self._lock:
            if self._closed:
                return
            self._check_process()
            self._closing = True
            state = self._store.begin_teardown()
            if state["allocations"] or state["fences"] or self._consumers:
                raise IntegrationError("OWNER_BUSY", "Allocations, fences or consumers still protect CUDA pools")
            for ordinal in self._devices.values():
                self._torch.cuda.synchronize(ordinal)
            self._buffers.clear()
            self._torch.cuda.ipc_collect()
            self._closed = True
            with _OWNER_REGISTRY_LOCK:
                _LIVE_OWNERS.pop(self.owner_id, None)


@dataclass
class ImportedPool:
    metadata: PoolMetadata
    key: Any
    value: Any
    device_ordinal: int
    closed: bool = False

    def close(self) -> None:
        """Destroy this process's references before acknowledging consumer release.

        Callers must first drop all their aliases to key/value. This cannot
        detect aliases held by arbitrary application or extension code.
        """
        with _IMPORT_LOCK:
            if self.closed:
                return
            import torch
            torch.cuda.synchronize(self.device_ordinal)
            self.key = None
            self.value = None
            self.closed = True
            cache_key = (self.metadata.owner_id, self.metadata.export_id)
            _IMPORTED_POOLS.pop(cache_key, None)
            _CLOSED_IMPORTS.add(cache_key)


def rebuild_pool(metadata: PoolMetadata, device_ordinal: int | None = None) -> ImportedPool:
    """Rebuild authenticated local metadata on the same physical visible GPU."""
    if not isinstance(metadata, PoolMetadata):
        raise IntegrationError("INVALID_ARGUMENT", "PoolMetadata required")
    visible = _visible_devices()
    if device_ordinal is None:
        if metadata.visible_device not in visible:
            raise IntegrationError("DEVICE_VISIBILITY", "Exported physical GPU is not visible")
        device_ordinal = visible.index(metadata.visible_device)
    if (type(device_ordinal) is not int or not 0 <= device_ordinal < len(visible)
            or visible[device_ordinal] != metadata.visible_device):
        raise IntegrationError("DEVICE_VISIBILITY", "Consumer ordinal selects a different physical GPU")
    with _IMPORT_LOCK:
        cache_key = (metadata.owner_id, metadata.export_id)
        if cache_key in _CLOSED_IMPORTS or cache_key in _PENDING_IMPORTS:
            raise IntegrationError("STALE_IMPORT", "Closed or partial CUDA imports cannot be rebuilt again")
        if cache_key in _IMPORTED_POOLS:
            imported = _IMPORTED_POOLS[cache_key]
            if imported.metadata != metadata or imported.device_ordinal != device_ordinal:
                raise IntegrationError("IDENTITY_CONFLICT", "CUDA descriptor replay differs")
            return imported
        import torch
        if str(torch.__version__) != metadata.torch_version:
            raise IntegrationError("CUDA_IPC_ABI", "Owner and consumer torch versions differ")
        if not torch.cuda.is_available():
            raise IntegrationError("CUDA_UNAVAILABLE", "Consumer CUDA is unavailable")
        def rebuild(info: CUDATensorMetadata, shape: tuple[int, ...]) -> Any:
            if not isinstance(info, CUDATensorMetadata) or len(info.rebuild_args) != 15:
                raise IntegrationError("CUDA_IPC_ABI", "Unsupported CUDA metadata")
            args = list(info.rebuild_args)
            if (tuple(args[1]) != shape or args[5] != torch.float16
                    or args[6] != metadata.device_ordinal or args[10]):
                raise IntegrationError("LAYOUT_MISMATCH", "CUDA pool metadata layout differs")
            if info.visible_device != metadata.visible_device or info.torch_version != metadata.torch_version:
                raise IntegrationError("IDENTITY_CONFLICT", "Pool tensor metadata identifies another owner GPU")
            return rebuild_tensor(info, device_ordinal)
        with torch.cuda.device(device_ordinal):
            _PENDING_IMPORTS.add(cache_key)
            key = rebuild(metadata.key, (metadata.blocks, 16, 16, 8))
            value = rebuild(metadata.value, (metadata.blocks, 128, 16))
        imported = ImportedPool(metadata, key, value, device_ordinal)
        _IMPORTED_POOLS[cache_key] = imported
        _PENDING_IMPORTS.remove(cache_key)
        return imported
