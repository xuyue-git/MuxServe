"""Apply auditable hooks only to the downloaded fixed upstream source."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).absolute().parents[1]
UPSTREAM = ROOT / "third_party" / "sglang"


def _is_link(path: Path) -> bool:
    junction = getattr(path, "is_junction", None)
    return path.is_symlink() or (junction is not None and junction())


def _repository_root() -> Path:
    current = Path(ROOT.anchor)
    anchor_parts = Path(ROOT.anchor).parts
    for part in ROOT.parts[len(anchor_parts):]:
        current = current / part
        prohibited = Path("/home/incoming")
        if current == prohibited or prohibited in current.parents:
            raise RuntimeError("This patch tool excludes /home/incoming")
        if _is_link(current):
            raise RuntimeError("Patch repository root contains a symlink or junction")
    if not current.is_dir():
        raise RuntimeError("Patch repository root is not a directory")
    return current


def _project_path(path: Path, *, required: bool = True) -> Path:
    """Reject path traversal and links before reading or writing project files."""
    root = _repository_root()
    try:
        relative = path.relative_to(root)
    except ValueError as exc:
        raise RuntimeError("Patch paths must remain inside this copied repository") from exc
    if relative.is_absolute() or any(part in (".", "..") for part in relative.parts):
        raise RuntimeError("Patch paths must be canonical repository-relative names")
    current = root
    for part in relative.parts:
        current = current / part
        if _is_link(current):
            raise RuntimeError(f"Patch path contains a link: {relative.as_posix()}")
    if required and not current.is_file():
        raise RuntimeError(f"Patch source is not a regular file: {relative.as_posix()}")
    return current


def apply():
    manifest_path = _project_path(ROOT / "upstream-reference/extraction.json")
    manifest = json.loads(manifest_path.read_text())
    changes = {'python/sglang/srt/managers/tp_worker.py': [('    def get_worker_info(self):\n'
                                                  '        max_req_len = min(\n'
                                                  '            self.model_config.context_len - 1,\n'
                                                  '            self.model_runner.effective_max_total_num_tokens '
                                                  '* self.ps.attn_dcp_size\n'
                                                  '            - 1,\n'
                                                  '        )\n',
                                                  '    def get_worker_info(self):\n'
                                                  '        from muxserve.flexserver.sglang_hooks import enabled as '
                                                  'muxserve_enabled\n'
                                                  '\n'
                                                  '        # Native admission reserves another slot. These are '
                                                  'logical IDs only;\n'
                                                  '        # the controller still limits every actual forward to '
                                                  'context 2048.\n'
                                                  '        context_admission_len = (\n'
                                                  '            self.model_config.context_len + 1\n'
                                                  '            if muxserve_enabled()\n'
                                                  '            else self.model_config.context_len - 1\n'
                                                  '        )\n'
                                                  '        max_req_len = min(\n'
                                                  '            context_admission_len,\n'
                                                  '            self.model_runner.effective_max_total_num_tokens '
                                                  '* self.ps.attn_dcp_size\n'
                                                  '            - 1,\n'
                                                  '        )\n'),
                                                 ('            max_req_len - 5,\n',
                                                  '            (min(self.model_config.context_len - 1, '
                                                  'max_req_len - 1)\n'
                                                  '             if muxserve_enabled() else max_req_len - 5),\n')],
     'python/sglang/srt/model_executor/model_runner.py': [('        """Allocate KV cache memory pools only (no '
                                                           'backends or cuda graphs)."""\n',
                                                           '        """Allocate KV cache memory pools only (no '
                                                           'backends or cuda graphs)."""\n'
                                                           '        from muxserve.flexserver.sglang_hooks import '
                                                           'enabled, allocate_shared_memory\n'
                                                           '        if enabled():\n'
                                                           '            return allocate_shared_memory(self, '
                                                           'memory_pool_config)\n'),
                                                          ('        # Deprecated kwarg: pre-planners mark the '
                                                           'batch themselves now.\n',
                                                           '        from muxserve.flexserver.sglang_hooks import '
                                                           'assert_forward_permitted\n'
                                                           '        assert_forward_permitted(self, '
                                                           'forward_batch)\n'
                                                           '        # Deprecated kwarg: pre-planners mark the '
                                                           'batch themselves now.\n')],
     'python/sglang/srt/model_loader/loader.py': [('    """Get a model loader based on the load format."""\n',
                                                   '    """Get a model loader based on the load format."""\n'
                                                   '    from muxserve.flexserver.sglang_hooks import enabled, '
                                                   'shared_weight_loader\n'
                                                   '    if enabled():\n'
                                                   '        return shared_weight_loader(load_config)\n')],
     'python/sglang/srt/managers/scheduler.py': [('def dispatch_event_loop(scheduler: Scheduler):\n',
                                                  'def dispatch_event_loop(scheduler: Scheduler):\n'
                                                  '    from muxserve.flexserver.sglang_hooks import enabled, '
                                                  'run_controlled_scheduler\n'
                                                  '    if enabled():\n'
                                                  '        return run_controlled_scheduler(scheduler)\n')],
     'python/sglang/srt/model_executor/forward_batch_info.py': [('    seq_lens_cpu: Optional[torch.Tensor] = '
                                                                 'None\n',
                                                                 '    seq_lens_cpu: Optional[torch.Tensor] = '
                                                                 'None\n'
                                                                 '    # Controlled Head-wise forwards verify '
                                                                 'request-row order on CPU.\n'
                                                                 '    req_pool_indices_cpu: '
                                                                 'Optional[torch.Tensor] = None\n'),
                                                                ('            seq_lens_cpu=seq_lens_cpu,\n',
                                                                 '            seq_lens_cpu=seq_lens_cpu,\n'
                                                                 '            req_pool_indices_cpu=(\n'
                                                                 '                getattr(batch, '
                                                                 '"req_pool_indices_cpu", None)\n'
                                                                 '                if '
                                                                 'getattr(model_runner.token_to_kv_pool, '
                                                                 '"headwise_shared", False)\n'
                                                                 '                else None\n'
                                                                 '            ),\n')],
     'python/sglang/srt/layers/attention/attention_registry.py': [('ATTENTION_BACKENDS = {}\n',
                                                                   'ATTENTION_BACKENDS = {}\n'
                                                                   '\n'
                                                                   'def _create_headwise_backend(runner):\n'
                                                                   '    from muxserve.flexserver.sglang_attention '
                                                                   'import HeadwiseAttentionBackend\n'
                                                                   '    return HeadwiseAttentionBackend(runner)\n'
                                                                   '\n'
                                                                   'ATTENTION_BACKENDS["headwise"] = '
                                                                   '_create_headwise_backend\n')],
     'python/sglang/srt/arg_groups/choices.py': [('\nATTENTION_BACKEND_CHOICES = [\n',
                                                  '\nATTENTION_BACKEND_CHOICES = [\n    "headwise",\n'),
                                                 ('\nDISAGG_TRANSFER_BACKEND_CHOICES = [\n',
                                                  '\n'
                                                  'DISAGG_TRANSFER_BACKEND_CHOICES = [\n'
                                                  '    "shared_headwise",\n')],
     'python/sglang/srt/disaggregation/utils.py': [('class TransferBackend(Enum):\n',
                                                    'class TransferBackend(Enum):\n'
                                                    '    SHARED_HEADWISE = "shared_headwise"\n'),
                                                   ('    if transfer_backend == TransferBackend.MOONCAKE:\n',
                                                    '    if transfer_backend == '
                                                    'TransferBackend.SHARED_HEADWISE:\n'
                                                    '        from muxserve.flexserver.sglang_pd import '
                                                    'shared_kv_class\n'
                                                    '        return shared_kv_class(class_type)\n'
                                                    '    if transfer_backend == TransferBackend.MOONCAKE:\n')]}
    # Attach native metadata objects to the custom transport without changing
    changes['python/sglang/srt/model_executor/model_runner.py'].append((
        '        return output\n\n    def _maybe_execute_deferred_mamba_cow_and_clear(',
        '        from muxserve.flexserver.sglang_hooks import finish_forward_timing\n'
        '        finish_forward_timing(self)\n'
        '        return output\n\n    def _maybe_execute_deferred_mamba_cow_and_clear('
    ))
    # Attach native metadata objects to the custom transport without changing
    # normal transfer backends. CPU metadata is handed off; physical KV is shared.
    for stage in ("prefill", "decode"):
        changes[f"python/sglang/srt/disaggregation/{stage}.py"] = [
            ('        return kv_manager\n',
             '        if self.transfer_backend.value == "shared_headwise":\n'
             '            kv_manager.attach_metadata_buffers(self.metadata_buffers)\n'
             '        return kv_manager\n')]
    changes['python/sglang/srt/utils/watchdog.py'] = [('        t = threading.Thread(target=self._watchdog_thread, daemon=True)\n        t.start()\n', '        self._stop_event = threading.Event()\n        self._thread = threading.Thread(target=self._watchdog_thread, daemon=True)\n        self._thread.start()\n\n    def stop(self, timeout: float = 5.0) -> None:\n        """Join monitoring before releasing callbacks that may own CUDA state."""\n        self._stop_event.set()\n        self._thread.join(timeout=timeout)\n        if self._thread.is_alive():\n            raise RuntimeError(f"{self.debug_name} watchdog did not stop")\n        self.get_counter = lambda: 0\n        self.is_active = lambda: False\n        self.dump_info = None\n'), ('    def _watchdog_thread(self):\n        try:\n            while True:\n', '    def _watchdog_thread(self):\n        try:\n            while not self._stop_event.is_set():\n'), ('        while True:\n            current = time.perf_counter()\n', '        while not self._stop_event.is_set():\n            current = time.perf_counter()\n'), ('            time.sleep(self.watchdog_timeout / 2)\n', '            if self._stop_event.wait(self.watchdog_timeout / 2):\n                return\n        else:\n            return\n\n        if self._stop_event.is_set():\n            return\n'), ('            time.sleep(5)\n            self.parent_process.send_signal(signal.SIGQUIT)\n', '            if not self._stop_event.wait(5):\n                self.parent_process.send_signal(signal.SIGQUIT)\n')]
    changes['python/sglang/srt/weight_cache/ipc_loader.py'] = [
        ('    def _load_zero_copy_mode(\n',
         '    def _initialize_model(self, model_config, quant_config):\n'
         '        return _initialize_model(model_config, self.load_config, quant_config)\n\n'
         '    def _load_zero_copy_mode(\n'),
        ('                model = _initialize_model(\n'
         '                    model_config,\n'
         '                    self.load_config,\n'
         '                    quant_config,\n'
         '                )\n',
         '                model = self._initialize_model(model_config, quant_config)\n')]
    changes['python/sglang/srt/model_executor/runner/eager_runner.py'] = [
        ('        if envs.SGLANG_EAGER_INPUT_NO_COPY.get():\n',
         '        from muxserve.flexserver.sglang_hooks import enabled\n\n'
         '        if enabled():\n'
         '            # A controlled grant binds this exact batch and its logical write\n'
         '            # locations. Keep both identities through eager metadata and layers.\n'
         '            return forward_batch\n'
         '        if envs.SGLANG_EAGER_INPUT_NO_COPY.get():\n')]
    pending = {}
    previous_path = _project_path(ROOT / "upstream-reference/patches.json", required=False)
    previous = json.loads(previous_path.read_text()) if previous_path.is_file() else {}
    for name, edits in changes.items():
        path = _project_path(UPSTREAM / name)
        original = path.read_bytes()
        digest = hashlib.sha256(original).hexdigest()
        record = previous.get(name, {})
        if record.get("source_sha256") != manifest["files_sha256"][name]:
            raise RuntimeError(f"Patch manifest source differs from extraction: {name}")
        if digest == record.get("patched_sha256"):
            if any(original.decode().count(new) != 1 for _, new in edits):
                raise RuntimeError(f"Manifest points to an obsolete hook revision: {name}")
            continue
        if digest == manifest["files_sha256"][name]:
            selected_edits = edits
        elif digest in record.get("upgrade_from", {}):
            selected_edits = record["upgrade_from"][digest]
        else:
            raise RuntimeError(f"Upstream source differs from frozen version: {name}")
        text = original.decode()
        for old, new in selected_edits:
            if text.count(old) != 1:
                raise RuntimeError(f"Hook must match once in fixed upstream: {name}: {old!r}")
            text = text.replace(old, new, 1)
        if hashlib.sha256(text.encode()).hexdigest() != record["patched_sha256"]:
            raise RuntimeError(f"Patched content differs from recorded target: {name}")
        pending[name] = text
    # Validate all targets before writing any file.
    for name, text in pending.items():
        data = text.encode()
        _project_path(UPSTREAM / name).write_bytes(data)
    print(f"Applied {len(pending)} fixed-source integration patches")


if __name__ == "__main__":
    apply()
