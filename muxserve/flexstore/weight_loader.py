"""Native zero-copy loader with source-allocation and final-alias evidence."""
from __future__ import annotations

import json
import hashlib
import os

from sglang.srt.weight_cache.ipc_loader import IpcModelLoader

from muxserve.config import safe_path
from muxserve.contracts import IntegrationError


def _local_alias(tensor):
    return (tensor.device.type, tensor.device.index, tensor.data_ptr(),
            tensor.storage_offset(), tuple(tensor.shape), tuple(tensor.stride()), str(tensor.dtype))


class _RecordingTransport:
    def __init__(self, backend):
        self.backend, self.name = backend, backend.name
        self.imports = {}

    def import_tensor(self, entry):
        name = entry.get("name")
        if not isinstance(name, str) or name in self.imports:
            raise IntegrationError("WEIGHT_EVIDENCE_MISSING", "Expected one named weight import")
        tensor = self.backend.import_tensor(entry)
        allocation = entry.get("allocation_digest")
        counter = entry.get("refcounter_digest")
        if tensor.device.type == "cuda" and (not isinstance(allocation, str) or not isinstance(counter, str)):
            raise IntegrationError("WEIGHT_EVIDENCE_MISSING", "CUDA import lacks owner allocation evidence")
        self.imports[name] = {"alias": _local_alias(tensor), "is_param": entry["is_param"],
                              "allocation_digest": allocation, "refcounter_digest": counter}
        return tensor


def audit_shared_weights(model):
    imports = getattr(model, "_muxserve_weight_imports", None)
    if not imports:
        raise IntegrationError("WEIGHT_EVIDENCE_MISSING", "Model was not loaded by the shared owner")
    actual = dict(model.named_parameters(remove_duplicate=False))
    actual.update(dict(model.named_buffers()))
    if set(actual) != set(imports):
        raise IntegrationError("WEIGHT_ALIAS_CHANGED", "Final model parameter/buffer names differ from imports")
    for name, record in imports.items():
        if _local_alias(actual[name]) != record["alias"]:
            raise IntegrationError("WEIGHT_ALIAS_CHANGED", f"Final model storage changed: {name}")
    return {name: {"allocation_digest": value["allocation_digest"],
                   "refcounter_digest": value["refcounter_digest"],
                   "shape": value["alias"][4], "dtype": value["alias"][6],
                   "is_param": value["is_param"], "final_alias_matches": True}
            for name, value in imports.items()}


class SharedWeightLoader(IpcModelLoader):
    def _initialize_model(self, model_config, quant_config):
        from sglang.srt.utils import reserve_rope_cache_for_long_sequences
        model = super()._initialize_model(model_config, quant_config)
        # Still inside the native meta-device context: reserve shapes, not CUDA memory.
        reserve_rope_cache_for_long_sequences(model, model_config)
        return model

    def _fetch_from_cache(self, model_config, device_config):
        result = super()._fetch_from_cache(model_config, device_config)
        if result is not None:
            self._transport_backend = _RecordingTransport(self._transport_backend)
        return result

    def load_model(self, *, model_config, device_config):
        model = super().load_model(model_config=model_config, device_config=device_config)
        model._muxserve_weight_imports = self._transport_backend.imports
        audit_shared_weights(model)
        return model


def record_runtime_evidence(scheduler, context):
    runner = scheduler.tp_worker.model_runner
    rank = runner.ps.tp_rank
    imported = runner._muxserve_imported_pool
    metadata = imported.metadata
    pool = runner.token_to_kv_pool
    if (imported.closed or _local_alias(pool.pool_key) != _local_alias(imported.key)
            or _local_alias(pool.pool_value) != _local_alias(imported.value)
            or metadata.rank != rank or metadata.blocks != context.config.pool_blocks_per_rank
            or metadata.consumer_id != f"{context.group_id}:{context.stage}:rank{rank}"
            or runner._muxserve_import_pid != os.getpid() or runner._muxserve_import_state != "MAPPED"):
        raise IntegrationError("KV_ALIAS_CHANGED", "Runtime KV pool differs from owner imports")
    def digest(tensor):
        args = tensor.rebuild_args
        return hashlib.sha256(repr((args[3], args[7], args[9])).encode()).hexdigest()
    evidence = {"model_id": context.model.model_id, "group_id": context.group_id,
                "config_fingerprint": context.config.fingerprint, "stage": context.stage,
                "import_state": "MAPPED",
                "rank": rank, "pid": os.getpid(), "weights": audit_shared_weights(runner.model),
                "kv": {"owner_id": metadata.owner_id, "consumer_id": metadata.consumer_id,
                       "export_id": metadata.export_id, "rank": rank,
                       "visible_device": metadata.visible_device, "blocks": metadata.blocks,
                       "key_allocation_digest": digest(metadata.key),
                       "value_allocation_digest": digest(metadata.value), "aliases_match": True}}
    path = safe_path(context.runtime_dir / f"{context.model.model_id}-{context.stage}-r{rank}-weights.json")
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as output:
        json.dump(evidence, output, sort_keys=True)
    return evidence
