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


def read_asset_path(value: str | Path) -> Path:
    """Resolve an explicit asset path through the common protected resolver."""
    return safe_path(value)


def asset_lock_path(root: str | Path, lock_dir: str = "") -> Path:
    root = read_asset_path(root)
    if not lock_dir:
        # Read-only compatibility with locks made by older deliveries.
        return read_asset_path(root / ".muxserve-asset-lock.json")
    if not isinstance(lock_dir, str) or not Path(lock_dir).is_absolute():
        raise IntegrationError("INVALID_CONFIG", "asset_lock_dir must be an absolute personal directory")
    directory = safe_path(lock_dir)
    if directory == root or root in directory.parents:
        raise IntegrationError("INVALID_CONFIG", "Asset locks must be outside model/tokenizer directories")
    name = hashlib.sha256(os.fsencode(root)).hexdigest() + ".json"
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
    # Maximum consecutive decode forwards under one D grant. 1 is the original
    # per-token protocol; larger runs still yield at every step boundary.
    decode_run_steps: int = 1
    # Capture D-stage decode forwards as SGLang-native CUDA graphs (one per
    # batch size 1..max_model_inflight). P stays eager. Off keeps every
    # forward eager, as before.
    decode_cuda_graph: bool = False
    # A decode run that yields to a waiting prefill parks (keeps its KV
    # reservation and native state, releases its credits) and resumes once
    # the prefill's forward finished, instead of ending and re-preparing.
    decode_run_park: bool = False
    # P-stage SGLang breakable prefill CUDA graphs (transformer body captured
    # in segments, Head-wise attention eager at the breaks) for prompts of
    # exactly PREFILL_GRAPH_TOKENS; other lengths stay eager.
    prefill_cuda_graph: bool = False
    # Original MuxServe sizing: KV = gpu_memory_utilization x total GPU memory
    # (see kv_bytes_from_fraction). When set, the launcher resolves it into
    # kv_bytes_per_rank before writing the runtime config; mutually exclusive
    # with an explicit kv_bytes_per_rank.
    kv_memory_fraction: float | None = None
    # START a batch inside the controller as soon as it becomes startable
    # (its last READY arrives, or a parking/finishing batch frees the credits
    # a READY prefill waits for), instead of on the driver thread's next pass.
    # Same checks as the driver's start_batch call.
    inline_start: bool = False
    # Decode batches involve only the D ranks. P ranks otherwise receive a
    # PREPARE/QUIESCE for every decode batch only to validate and acknowledge
    # a mapping they never use. Prefill batches still involve every rank.
    decode_skip_prefill_ranks: bool = False
    # Publish a prefill's token as soon as every P rank's result is committed
    # (same token, all ranks agreed), instead of after the D handoff. The
    # request still becomes decodable only after the handoff.
    prefill_early_token: bool = False
    # A request's CLEANUP involves only the D ranks once every P rank has
    # released its native state (its prefill batch quiesced normally, which
    # releases each request on every P rank). P ranks otherwise receive a
    # CLEANUP that finds nothing to release and only trims their allocator
    # cache; with this switch they trim after the prefill QUIESCE instead.
    cleanup_skip_prefill_ranks: bool = False
    # Queue a finished request's CLEANUP right behind its last decode batch's
    # QUIESCE on every rank, instead of after all ranks reported QUIESCED.
    # Ranks execute and report commands in order, so every rank's batch
    # QUIESCED precedes its CLEANUP QUIESCED.
    cleanup_with_quiesce: bool = False
    # Idle allocator trimming: 0 calls empty_cache at every idle cleanup (as
    # before); N > 0 only when at least N MiB sit in wholly free cached
    # segments (what empty_cache can return; split segments cannot be freed).
    idle_cache_trim_mib: int = 0
    # Prepare a prefill's first decode batch as soon as every P rank's result
    # is committed, queued behind the HANDOFF on each D rank (which executes
    # it once its handoff completed), instead of after every HANDOFF_READY.
    # Requires inline_start, prefill_early_token and decode_skip_prefill_ranks.
    handoff_prepare_decode: bool = False
    # HANDOFF carries each rank's staged P metadata, so a D rank needs no
    # broker_lookup RPC; it still ACKs the committed mapping.
    handoff_inline_metadata: bool = False
    # A D rank whose run ends locally (finished, or the last granted step)
    # reports that step inside its GPU_DONE instead of a separate decode_step.
    final_step_in_gpu_done: bool = False
    # Runtime events reach the gateway from their own driver thread rather
    # than between the driver's scheduling passes.
    event_pump_thread: bool = False
    # Command long-poll replies tell P ranks which bootstrap rooms have every
    # D destination registered, replacing their broker_bootstrap RPCs.
    bootstrap_in_poll: bool = False
    # Prepare a submitted request's prefill right away (the gateway asks once
    # its own lock is released) instead of on the driver's next pass. Only
    # when it is the model's sole queued request, so FIFO order is unchanged.
    # Requires inline_start (the driver no longer tracks such batches).
    inline_prepare: bool = False
    # Each PREPARE carries only the receiving rank's block table, and that
    # rank validates only it. Every table is still validated, by the one rank
    # that uses it; ranks no longer re-check the other ranks' tables.
    prepare_own_rank_table: bool = False
    # A D rank sends its native destination registrations for a prefill batch
    # inside that batch's READY report, instead of one broker_register_destination
    # RPC before it. The owner registers them exactly as that RPC would.
    register_in_ready: bool = False
    # Command long-poll replies carry every room's broker status (what
    # broker_poll would return), so ranks waiting for the all-rank handoff ACK
    # need no broker_poll RPC; an absent room still asks the owner.
    broker_status_in_poll: bool = False
    # A request that finished normally (eos/length) is reported TERMINAL when
    # its CLEANUP is queued, not when every rank has reported it. Its output is
    # final by then; only the release is still in flight.
    terminal_before_cleanup: bool = False
    # A request whose queued CLEANUP involves only D ranks no longer counts
    # against its model's in-flight slot: every D rank executes that CLEANUP
    # before any later PREPARE, and P ranks already released it. The store
    # still holds its KV until the cleanup completes. Requires
    # cleanup_skip_prefill_ranks.
    slot_free_at_cleanup: bool = False
    # A D rank's broker ACK rides on its next command long-poll, which also
    # returns the all-rank result: one round trip instead of an ACK RPC and a
    # poll. The ACK's checks and the all-rank proof before the native
    # collective are unchanged. Requires broker_status_in_poll.
    handoff_ack_in_poll: bool = False
    # A D rank whose next command prepares the first decode batch sends
    # HANDOFF_READY and that batch's READY in one RPC, and runs commands it
    # already holds without polling first. Requires handoff_prepare_decode.
    handoff_report_with_ready: bool = False
    # The reply to the READY that STARTs a batch carries that rank's START, so
    # the last rank to report needs no poll to receive it.
    start_in_report_reply: bool = False
    # A long-polling rank is woken only by its own command or its group's
    # broker change, not by every owner state change.
    targeted_wake: bool = False
    # The owner's short control RPCs (rank reports, step reports, command
    # polls, broker calls) run on the connection's handler thread instead of a
    # new thread per call, and its sockets read and write before polling. Their
    # waits are already bounded; every other RPC keeps its worker and deadline.
    rpc_inline_fast_methods: bool = False
    # Only D rank 0 reports a continuing decode step; the other ranks read its
    # decision from shared memory and report only when it is not "continue".
    # Each rank's token digest is compared at GPU_DONE instead of every step.
    decode_leader_reports: bool = False
    # The first decode batch is reserved once every D rank has seen that all
    # ranks ACKed the handoff, so the reservation no longer holds the owner
    # lock while they read HANDOFF and ACK. Requires handoff_prepare_decode and
    # handoff_ack_in_poll.
    handoff_prepare_after_ack: bool = False
    # READY of a decode batch checks the CPU copies of its row/length metadata
    # and skips the GPU read-back and synchronize; those tensors were just
    # built from the same CPU values on this rank.
    decode_prepare_light: bool = False

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


def kv_bytes_from_fraction(fraction: float, total_memory_bytes: int) -> int:
    """KV bytes per GPU exactly as original MuxServe sizes its cache.

    The block count comes from the original code itself:
    muxserve/muxsched/resource.py num_gpu_blocks() (the formula of
    get_num_gpu_blocks(), identical in muxserve/flexstore/manager.py):
    available = round(gpu_memory_utilization * total_memory) - 1, one block
    holds K and V for one head of block_size=32 FP16 tokens at head_size=128
    (MuxServeArgs defaults), rounded down to a multiple of 128 blocks. The
    result is a multiple of 16384 bytes, so it maps exactly onto this
    backend's 16-token (8192-byte) blocks.
    """
    from muxserve.muxsched.resource import num_gpu_blocks
    head_size, block_size, element_size = 128, 32, 2
    x = 16 // element_size  # CacheResource.get_key_block_shape / get_value_block_shape
    key_block_shape, value_block_shape = (head_size // x, block_size, x), (head_size, block_size)
    blocks = int(num_gpu_blocks(fraction, total_memory_bytes, key_block_shape, value_block_shape))
    block_mem = element_size * (head_size * block_size + head_size * block_size)
    return blocks * block_mem


def load_config(path: str | Path, *, require_assets: bool = True) -> DeploymentConfig:
    path = safe_path(path)
    raw = json.loads(path.read_text(encoding="utf-8"))
    fraction = raw.get("kv_memory_fraction")
    if fraction is not None:
        if "kv_bytes_per_rank" in raw:
            raise IntegrationError("INVALID_CONFIG", "Use kv_memory_fraction or kv_bytes_per_rank, not both")
        if type(fraction) not in (int, float) or not 0 < fraction < 1:
            raise IntegrationError("INVALID_CONFIG", "kv_memory_fraction must be in (0, 1)")
    models = tuple(ModelDeployment(**item) for item in raw.pop("models"))
    config = DeploymentConfig(models=models, **raw)
    if not isinstance(config.asset_lock_dir, str):
        raise IntegrationError("INVALID_CONFIG", "asset_lock_dir must be a string")
    integer_limits = ("tp_size", "batch_history_limit", "kv_bytes_per_rank", "extra_headroom_bytes_per_rank",
                      "max_model_inflight", "credit_multiplier", "max_input_tokens",
                      "max_new_tokens", "max_context_tokens", "request_timeout_ms",
                      "prepare_timeout_ms", "controller_port", "queue_limit", "decode_run_steps")
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
    if type(config.decode_cuda_graph) is not bool:
        raise IntegrationError("INVALID_CONFIG", "decode_cuda_graph must be a boolean")
    if type(config.decode_run_park) is not bool:
        raise IntegrationError("INVALID_CONFIG", "decode_run_park must be a boolean")
    if type(config.prefill_cuda_graph) is not bool:
        raise IntegrationError("INVALID_CONFIG", "prefill_cuda_graph must be a boolean")
    for name in ("inline_start", "decode_skip_prefill_ranks", "prefill_early_token",
                 "cleanup_skip_prefill_ranks", "cleanup_with_quiesce", "handoff_prepare_decode",
                 "handoff_inline_metadata", "final_step_in_gpu_done", "event_pump_thread",
                 "bootstrap_in_poll", "inline_prepare", "prepare_own_rank_table",
                 "register_in_ready", "broker_status_in_poll", "terminal_before_cleanup",
                 "slot_free_at_cleanup", "handoff_ack_in_poll",
                 "handoff_report_with_ready", "start_in_report_reply", "targeted_wake",
                 "rpc_inline_fast_methods", "decode_leader_reports", "handoff_prepare_after_ack",
                 "decode_prepare_light"):
        if type(getattr(config, name)) is not bool:
            raise IntegrationError("INVALID_CONFIG", name + " must be a boolean")
    if config.handoff_prepare_after_ack and not (config.handoff_prepare_decode and config.handoff_ack_in_poll):
        raise IntegrationError("INVALID_CONFIG", "handoff_prepare_after_ack requires handoff_prepare_decode "
                                                 "and handoff_ack_in_poll")
    if config.handoff_report_with_ready and not config.handoff_prepare_decode:
        raise IntegrationError("INVALID_CONFIG", "handoff_report_with_ready requires handoff_prepare_decode")
    if config.handoff_ack_in_poll and not config.broker_status_in_poll:
        raise IntegrationError("INVALID_CONFIG", "handoff_ack_in_poll requires broker_status_in_poll")
    if config.slot_free_at_cleanup and not config.cleanup_skip_prefill_ranks:
        raise IntegrationError("INVALID_CONFIG", "slot_free_at_cleanup requires cleanup_skip_prefill_ranks")
    if config.handoff_prepare_decode and not (config.inline_start and config.prefill_early_token
                                              and config.decode_skip_prefill_ranks):
        raise IntegrationError("INVALID_CONFIG", "handoff_prepare_decode requires inline_start, "
                               "prefill_early_token and decode_skip_prefill_ranks")
    if config.inline_prepare and not config.inline_start:
        raise IntegrationError("INVALID_CONFIG", "inline_prepare requires inline_start")
    if type(config.idle_cache_trim_mib) is not int or not 0 <= config.idle_cache_trim_mib <= 65536:
        raise IntegrationError("INVALID_CONFIG", "idle_cache_trim_mib must be an integer 0..65536")
    if config.decode_run_steps > config.max_new_tokens:
        raise IntegrationError("INVALID_CONFIG", "Decode run length exceeds the output budget")
    if config.queue_limit > 100000:
        raise IntegrationError("INVALID_CONFIG", "Queue limit exceeds the bounded lifetime ceiling")
    if config.tp_size not in (1, 2, 4):
        raise IntegrationError("INVALID_CONFIG", "Initial TP is 1, 2 or 4; acceptance requires TP4")
    if config.max_model_inflight not in (1, 2) or config.credit_multiplier not in (1, 3):
        raise IntegrationError("INVALID_CONFIG", "Initial inflight and credit profiles are restricted")
    if (config.max_input_tokens, config.max_new_tokens, config.max_context_tokens) != (1792, 256, 2048):
        raise IntegrationError("INVALID_CONFIG", "Initial token limits must match the approved design")
    if config.kv_bytes_per_rank < 8192 or config.extra_headroom_bytes_per_rank < 1024**3:
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
