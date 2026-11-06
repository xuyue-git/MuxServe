"""Frozen initial deployment configuration; device and asset choices are explicit."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
from collections import deque
import errno

from .contracts import IntegrationError, ModelLayout


SGLANG_COMMIT = "94602c9c2b7cbdb8efd5c52802dac6a1c180089e"
SGLANG_TAG = "v0.5.20"


def safe_path(value: str | Path) -> Path:
    expanded = os.path.expanduser(os.fspath(value))
    if os.name != "posix":
        return Path(expanded).resolve()

    prohibited = Path("/home/incoming")
    pending = deque(Path(expanded).parts if os.path.isabs(expanded)
                    else Path(os.getcwd(), expanded).parts)
    resolved = Path("/")
    hops = 0
    while pending:
        part = pending.popleft()
        if part in ("", "/", "."):
            continue
        if part == "..":
            resolved = resolved.parent
            continue
        candidate = resolved / part
        # Check every textual step before lstat/readlink, including a path
        # that later escapes /home/incoming via '..'.
        if candidate == prohibited or prohibited in candidate.parents:
            raise IntegrationError("FORBIDDEN_PATH", "This deployment excludes /home/incoming")
        try:
            target = os.readlink(candidate)
        except OSError as exc:
            if exc.errno not in (errno.EINVAL, errno.ENOENT, errno.ENOTDIR):
                raise
            resolved = candidate
            continue
        hops += 1
        if hops > 40:
            raise IntegrationError("INVALID_PATH", "Too many symbolic-link hops")
        if os.path.isabs(target):
            resolved = Path("/")
        pending.extendleft(reversed(Path(target).parts))
    return resolved


_READ_ONLY_SHARED_MODELS = tuple(
    (Path("/home/LAB/yezr/models/huggyllama") / name,
     Path("/home/incoming/Models/huggyllama") / name)
    for name in ("llama-7b", "llama-13b", "llama-30b")
)


def read_asset_path(value: str | Path) -> Path:
    """Read approved model entries through their public alias; never use for outputs."""
    path = Path(os.path.expanduser(os.fspath(value)))
    if os.name != "posix" or not path.is_absolute() or ".." in path.parts:
        return safe_path(value)
    match = next(((alias, target) for alias, target in _READ_ONLY_SHARED_MODELS
                  if path == alias or alias in path.parents), None)
    if match is None:
        return safe_path(value)
    alias, expected_target = match
    if safe_path(alias.parent) != alias.parent:
        raise IntegrationError("FORBIDDEN_PATH", "Shared model parent has changed")
    try:
        target = os.readlink(alias)
    except OSError as exc:
        if exc.errno in (errno.EINVAL, errno.ENOENT, errno.ENOTDIR):
            return safe_path(value)
        raise
    if Path(target) != expected_target:
        raise IntegrationError("FORBIDDEN_PATH", "Shared model link target has changed")

    # Retain the approved alias in every filesystem operation and returned path.
    pending = deque(path.relative_to(alias).parts)
    resolved: list[str] = []
    hops = 0
    while pending:
        part = pending.popleft()
        if part in ("", "."):
            continue
        if part == "..":
            if not resolved:
                raise IntegrationError("FORBIDDEN_PATH", "Asset link leaves its model directory")
            resolved.pop()
            continue
        candidate = alias.joinpath(*resolved, part)
        try:
            target = os.readlink(candidate)
        except OSError as exc:
            if exc.errno not in (errno.EINVAL, errno.ENOENT, errno.ENOTDIR):
                raise
            resolved.append(part)
            continue
        hops += 1
        if hops > 40:
            raise IntegrationError("INVALID_PATH", "Too many symbolic-link hops")
        target_path = Path(target)
        if target_path.is_absolute():
            for base in (alias, expected_target):
                try:
                    relative = target_path.relative_to(base)
                except ValueError:
                    continue
                resolved.clear()
                pending.extendleft(reversed(relative.parts))
                break
            else:
                raise IntegrationError("FORBIDDEN_PATH", "Asset link leaves its model directory")
        else:
            pending.extendleft(reversed(target_path.parts))
    return alias.joinpath(*resolved)


def asset_lock_path(root: str | Path, lock_dir: str = "") -> Path:
    root = read_asset_path(root)
    if not lock_dir:
        # Read-only compatibility with locks made by older deliveries.
        return read_asset_path(root / ".muxserve-asset-lock.json")
    if not isinstance(lock_dir, str) or not Path(lock_dir).is_absolute():
        raise IntegrationError("INVALID_CONFIG", "asset_lock_dir must be an absolute personal directory")
    directory = safe_path(lock_dir)
    identity = next((target / root.relative_to(alias) for alias, target in _READ_ONLY_SHARED_MODELS
                     if root == alias or alias in root.parents), root)
    if any(directory == asset or asset in directory.parents for asset in (root, identity)):
        raise IntegrationError("INVALID_CONFIG", "Asset locks must be outside model/tokenizer directories")
    name = hashlib.sha256(os.fsencode(identity)).hexdigest() + ".json"
    path = directory / name
    if path.is_symlink():
        raise IntegrationError("INVALID_CONFIG", "Personal asset lock files must not be symbolic links")
    return path


@dataclass(frozen=True)
class ModelDeployment:
    model_id: str
    model_path: str
    revision: str
    tokenizer_path: str
    tokenizer_revision: str
    num_layers: int
    num_heads: int
    hidden_size: int
    decode_mps: int
    prefill_port: int
    decode_port: int
    bootstrap_port: int

    def layout(self, tp_size: int) -> ModelLayout:
        return ModelLayout(self.model_id, self.num_layers, self.num_heads // tp_size,
                           tuple(range(tp_size)))


@dataclass(frozen=True)
class DeploymentConfig:
    models: tuple[ModelDeployment, ...]
    tp_size: int = 4
    kv_pool_mode: str = "shared"
    batch_history_limit: int = 4_000_000
    kv_bytes_per_rank: int = 2 * 1024**3
    extra_headroom_bytes_per_rank: int = 2 * 1024**3
    max_model_inflight: int = 1
    credit_multiplier: int = 3
    max_input_tokens: int = 1792
    max_new_tokens: int = 256
    max_context_tokens: int = 2048
    request_timeout_ms: int = 120000
    prepare_timeout_ms: int = 30000
    controller_port: int = 18080
    queue_limit: int = 4096
    runtime_dir: str = ""
    asset_lock_dir: str = ""

    @property
    def pool_blocks_per_rank(self) -> int:
        return self.kv_bytes_per_rank // (2 * 128 * 16 * 2)

    @property
    def fingerprint(self) -> str:
        from dataclasses import asdict
        return hashlib.sha256(json.dumps(asdict(self), sort_keys=True).encode()).hexdigest()

    def model(self, model_id: str) -> ModelDeployment:
        for model in self.models:
            if model.model_id == model_id:
                return model
        raise IntegrationError("UNKNOWN_MODEL", "Model is not configured")


def load_config(path: str | Path, *, require_assets: bool = True) -> DeploymentConfig:
    path = safe_path(path)
    raw = json.loads(path.read_text(encoding="utf-8"))
    models = tuple(ModelDeployment(**item) for item in raw.pop("models"))
    config = DeploymentConfig(models=models, **raw)
    if not isinstance(config.asset_lock_dir, str):
        raise IntegrationError("INVALID_CONFIG", "asset_lock_dir must be a string")
    integer_limits = ("tp_size", "batch_history_limit", "kv_bytes_per_rank", "extra_headroom_bytes_per_rank",
                      "max_model_inflight", "credit_multiplier", "max_input_tokens",
                      "max_new_tokens", "max_context_tokens", "request_timeout_ms",
                      "prepare_timeout_ms", "controller_port", "queue_limit")
    if any(type(getattr(config, name)) is not int or getattr(config, name) <= 0
           for name in integer_limits):
        raise IntegrationError("INVALID_CONFIG", "Deployment limits must be positive integers")
    if not isinstance(config.kv_pool_mode, str) or config.kv_pool_mode not in ("shared", "static"):
        raise IntegrationError("INVALID_CONFIG", "KV pool mode must be shared or static")
    if not 100_000 <= config.batch_history_limit <= 4_000_000:
        raise IntegrationError("INVALID_CONFIG", "Batch history must be 100000..4000000 per epoch")
    if not 1000 <= config.request_timeout_ms <= 300000:
        raise IntegrationError("INVALID_CONFIG", "Request deadline must be 1000..300000 ms for native integer seconds")
    if config.prepare_timeout_ms > 120000:
        raise IntegrationError("INVALID_CONFIG", "Preparation deadline must be at most 120000 ms")
    if config.queue_limit > 100000:
        raise IntegrationError("INVALID_CONFIG", "Queue limit exceeds the bounded lifetime ceiling")
    if config.tp_size not in (1, 2, 4):
        raise IntegrationError("INVALID_CONFIG", "Initial TP is 1, 2 or 4; acceptance requires TP4")
    if config.max_model_inflight not in (1, 2) or config.credit_multiplier not in (1, 3):
        raise IntegrationError("INVALID_CONFIG", "Initial inflight and credit profiles are restricted")
    if (config.max_input_tokens, config.max_new_tokens, config.max_context_tokens) != (1792, 256, 2048):
        raise IntegrationError("INVALID_CONFIG", "Initial token limits must match the approved design")
    if config.kv_bytes_per_rank < 8192 or config.extra_headroom_bytes_per_rank < 2 * 1024**3:
        raise IntegrationError("INVALID_CONFIG", "KV budget or workspace headroom is invalid")
    if len(set(model.model_id for model in models)) != len(models) or not models:
        raise IntegrationError("INVALID_CONFIG", "Model IDs must be unique")
    ports = [config.controller_port]
    for model in models:
        if any(type(getattr(model, name)) is not int or getattr(model, name) <= 0
               for name in ("num_layers", "num_heads", "hidden_size", "decode_mps",
                            "prefill_port", "decode_port", "bootstrap_port")):
            raise IntegrationError("INVALID_CONFIG", "Model dimensions and ports must be positive integers")
        if any(type(getattr(model, name)) is not str for name in
               ("model_id", "model_path", "revision", "tokenizer_path", "tokenizer_revision")):
            raise IntegrationError("INVALID_CONFIG", "Model identity and asset fields must be strings")
        if not re.fullmatch(r"[A-Za-z0-9_-]+", model.model_id):
            raise IntegrationError("INVALID_CONFIG", "Unsafe model ID")
        if (model.hidden_size != model.num_heads * 128 or model.num_heads % config.tp_size
                or model.num_layers <= 0 or model.decode_mps not in (30, 40)):
            raise IntegrationError("INVALID_CONFIG", "Unsupported initial Llama shape or MPS limit")
        model.layout(config.tp_size)
        if config.asset_lock_dir:
            asset_lock_path(model.model_path, config.asset_lock_dir)
            asset_lock_path(model.tokenizer_path, config.asset_lock_dir)
        ports.extend((model.prefill_port, model.decode_port, model.bootstrap_port))
        if require_assets:
            for revision in (model.revision, model.tokenizer_revision):
                if not re.fullmatch(r"[0-9a-f]{40}", revision):
                    raise IntegrationError("ASSET_NOT_FROZEN", "Use exact model and tokenizer revision SHAs")
            asset = read_asset_path(model.model_path)
            tokenizer = read_asset_path(model.tokenizer_path)
            if not asset.is_dir() or not tokenizer.is_dir():
                raise IntegrationError("ASSET_MISSING", "Model/tokenizer must be staged in allowed local paths")
            hf = json.loads(read_asset_path(asset / "config.json").read_text(encoding="utf-8"))
            expected = (model.num_layers, model.num_heads, model.hidden_size, model.num_heads)
            actual = (hf["num_hidden_layers"], hf["num_attention_heads"], hf["hidden_size"],
                      hf.get("num_key_value_heads", hf["num_attention_heads"]))
            if actual != expected or hf.get("architectures") != ["LlamaForCausalLM"]:
                raise IntegrationError("ASSET_MISMATCH", "Local model configuration differs from deployment")
    if len(set(ports)) != len(ports) or any(not 1024 <= port <= 65535 for port in ports):
        raise IntegrationError("INVALID_CONFIG", "Deployment ports must be distinct and unprivileged")
    if config.runtime_dir:
        safe_path(config.runtime_dir)
    return config


def visible_devices(tp_size: int) -> tuple[str, ...]:
    raw = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    tokens = tuple(token.strip() for token in raw.split(","))
    if len(tokens) != tp_size or any(not token for token in tokens) or len(set(tokens)) != len(tokens):
        raise IntegrationError("GPU_MAPPING_REQUIRED", "Set exactly TP-size coordinated CUDA_VISIBLE_DEVICES")
    if any(not (token.isdigit() or token.startswith("GPU-")) for token in tokens):
        raise IntegrationError("GPU_MAPPING_REQUIRED", "Initial deployment uses physical GPU ordinals or UUIDs")
    return tokens
