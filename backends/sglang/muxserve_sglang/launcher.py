"""Linux A100 supervisor for one shared owner and native SGLang P/D ranks."""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import json
import logging
import multiprocessing
import os
from pathlib import Path
import secrets
import signal
import socket
import subprocess
import sys
import tempfile
from threading import Event, Thread
import time

from .config import kv_bytes_from_fraction, load_config, read_asset_path, safe_path
from .contracts import IntegrationError

logger = logging.getLogger(__name__)


def _native_fields(config, model, stage):
    """Fields are from the frozen arg_groups/fields declarations."""
    fields = dict(model_path=model.model_path, tokenizer_path=model.tokenizer_path,
                revision=model.revision, trust_remote_code=False, dtype="float16",
                context_length=2048, tp_size=config.tp_size, pp_size=1, dp_size=1,
                host="127.0.0.1", port=model.prefill_port if stage == "P" else model.decode_port,
                nccl_port=model.prefill_port if stage == "P" else model.decode_port,
                dist_timeout=min(60, config.request_timeout_ms // 1000),
                attention_backend="headwise" if stage in ("P", "D") else "triton",
                sampling_backend="pytorch", page_size=1, chunked_prefill_size=-1,
                disable_radix_cache=True, disable_overlap_schedule=True,
                disable_cuda_graph=True, disable_prefill_cuda_graph=True,
                disable_decode_cuda_graph=True, disable_custom_all_reduce=True,
                enable_torch_compile=False, enable_symm_mem=False,
                max_running_requests=config.max_model_inflight,
                max_queued_requests=config.queue_limit,
                max_prefill_tokens=config.max_model_inflight * config.max_context_tokens,
                disaggregation_mode={"P": "prefill", "D": "decode"}.get(stage, "null"),
                disaggregation_transfer_backend="shared_headwise" if stage in ("P", "D") else "mooncake",
                disaggregation_bootstrap_port=model.bootstrap_port,
                disaggregation_decode_enable_radix_cache=False,
                disaggregation_decode_enable_offload_kvcache=False,
                skip_tokenizer_init=False, enable_metrics=False,
                weight_cache_mode="client" if stage in ("P", "D") else "off",
                load_format="auto")
    fields.update(_graph_fields(config, stage))
    return fields


# Breakable prefill graphs are captured per aggregate token count; the paired
# workload's prompts are exactly 512 tokens (other lengths run eagerly).
PREFILL_GRAPH_TOKENS = (512,)


def _graph_fields(config, stage):
    """D-stage decode CUDA graphs (SGLang-native full graphs, bs 1..inflight);
    P-stage breakable prefill graphs (attention eager at graph breaks)."""
    if stage == "P" and getattr(config, "prefill_cuda_graph", False):
        sizes = list(getattr(config, "prefill_graph_tokens", ()) or ()) or (
            list(PREFILL_GRAPH_TOKENS) + ([2 * tokens for tokens in PREFILL_GRAPH_TOKENS]
                                          if getattr(config, "prefill_pair_hold_ms", 0) else []))
        return dict(disable_cuda_graph=False, disable_prefill_cuda_graph=False,
                    disable_decode_cuda_graph=True, cuda_graph_backend_prefill="breakable",
                    cuda_graph_backend_decode="disabled", cuda_graph_bs_prefill=sizes)
    if stage != "D" or not getattr(config, "decode_cuda_graph", False):
        return {}
    return dict(disable_cuda_graph=False, disable_prefill_cuda_graph=True,
                disable_decode_cuda_graph=False, cuda_graph_backend_decode="full",
                cuda_graph_backend_prefill="disabled",
                cuda_graph_bs_decode=list(range(1, config.max_model_inflight + 1)))


def _arguments(fields):
    from sglang.srt.server_args import ServerArgs
    args = ServerArgs(**fields)
    args.resolve_once()
    args.check_server_args()
    return args


def _child_environment(environment, log_path):
    os.environ.update(environment)
    descriptor = os.open(log_path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
    os.dup2(descriptor, 1)
    os.dup2(descriptor, 2)
    os.close(descriptor)


def _weight_process(fields, environment, log_path, rank, address, control, dist_method):
    _child_environment(environment, log_path)
    os.environ.pop("MUXSERVE_EXECUTION_GROUP", None)
    from .weight_owner import run_weight_owner
    run_weight_owner(_arguments(fields), rank, rank, address, control, dist_method)


def _rank_process(fields, ports, environment, log_path, rank, gate, ready, startup_timeout):
    _child_environment(environment, log_path)
    if not gate.poll(startup_timeout) or gate.recv() is not True:
        raise IntegrationError("STARTUP_GATE", "Rank importer was not authorized before startup")
    gate.close()
    from sglang.srt.server_args import PortArgs
    from sglang.srt.managers.scheduler import run_scheduler_process
    run_scheduler_process(
        server_args=_arguments(fields), port_args=PortArgs(**ports),
        gpu_id=rank, tp_rank=rank, attn_cp_rank=0, moe_dp_rank=0,
        moe_ep_rank=0, pp_rank=0, dp_rank=0, pipe_writer=ready,
    )
    from .sglang_hooks import release_process_imports
    release_process_imports(rank)


def _free_loopback_port(excluded=()):
    import errno
    import secrets
    from pathlib import Path

    low, high = map(
        int,
        Path("/proc/sys/net/ipv4/ip_local_port_range")
        .read_text().split()
    )
    unprivileged = int(
        Path("/proc/sys/net/ipv4/ip_unprivileged_port_start")
        .read_text().strip()
    )

    excluded = set(excluded)
    candidates = [
        p for p in range(max(1025, unprivileged), 65536)
        if not low <= p <= high and p not in excluded
    ]
    secrets.SystemRandom().shuffle(candidates)

    deadline = time.monotonic() + 10
    for port in candidates:
        if time.monotonic() >= deadline:
            raise IntegrationError('PORT_SELECTION_FAILED', 'TCP route checks exceeded 10 seconds')
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
            try:
                probe.bind(("0.0.0.0", port))
            except OSError as exc:
                if exc.errno in (errno.EADDRINUSE, errno.EACCES):
                    continue
                raise IntegrationError(
                    "PORT_SELECTION_FAILED",
                    f"Unable to check port {port}"
                ) from exc
            # A bound, non-listening socket must refuse a local connection.
            # A successful connection here reaches a forwarded service instead.
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as route:
                route.settimeout(0.2)
                if route.connect_ex(('127.0.0.1', port)) != errno.ECONNREFUSED:
                    continue
            return port

    raise IntegrationError(
        "PORT_SELECTION_FAILED",
        "No available non-ephemeral port"
)


def _cuda_total_memory(cards):
    """torch total_memory of the first selected GPU, as original MuxServe reads it.

    Queried in a short-lived child before the private MPS daemon starts, so the
    supervisor itself never creates a non-MPS CUDA context.
    """
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=cards[0]["uuid"])
    for key in ("CUDA_MPS_PIPE_DIRECTORY", "CUDA_MPS_LOG_DIRECTORY"):
        env.pop(key, None)
    result = subprocess.run([sys.executable, "-c",
                             "import torch; print(torch.cuda.get_device_properties(0).total_memory)"],
                            env=env, capture_output=True, text=True, timeout=120)
    try:
        total = int(result.stdout.strip().splitlines()[-1])
    except (ValueError, IndexError):
        total = 0
    if result.returncode != 0 or total <= 0:
        raise IntegrationError("STARTUP_MEMORY_UNAVAILABLE", "Could not read GPU total memory for KV sizing")
    return total


def _write_json(path, value):
    path = safe_path(path)
    descriptor, temporary = tempfile.mkstemp(prefix=".evidence-", dir=path.parent)
    with os.fdopen(descriptor, "w", encoding="utf-8") as output:
        json.dump(value, output, sort_keys=True, indent=2)
    os.replace(temporary, path)


class Supervisor:
    def __init__(self, config_path, *, startup_timeout=1800, manual_control=False):
        if os.name != "posix" or not 0 < startup_timeout <= 7200:
            raise IntegrationError("DEPLOYMENT_PLATFORM", "Use Linux and a bounded startup timeout")
        from .preflight import inspect_assets, inspect_gpus, inspect_software
        config = load_config(config_path)
        software = inspect_software()
        assets = inspect_assets(config)
        cards = inspect_gpus(config.tp_size)
        self.stop_requested = Event()
        self.native_failure = Event()
        self.startup_timeout = startup_timeout
        self.startup_gate_complete = False
        self.cards = cards
        self.manual_control = manual_control
        self.ranks, self.weights, self.reaped = {}, {}, set()
        self.driver = self.gateway = self.http_thread = None
        self.rpc = self.gpu = self.controller = None
        self.mps_started = False
        self.context = multiprocessing.get_context("spawn")
        requested = os.environ["CUDA_VISIBLE_DEVICES"]
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(card["uuid"] for card in cards)
        proxy = os.environ.get("PROXY", "http://192.168.1.248:7890")
        for key in ("http_proxy", "HTTP_PROXY", "https_proxy", "HTTPS_PROXY", "all_proxy", "ALL_PROXY"):
            os.environ[key] = proxy
        no_proxy = os.environ.get("no_proxy", "localhost,127.0.0.1,::1,10.0.0.0/8,192.168.0.0/16")
        os.environ.update(NO_PROXY=no_proxy, no_proxy=no_proxy, CUDA_HOME=software["cuda_home"],
                          HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", TOKENIZERS_PARALLELISM="false",
                          SGLANG_KILLPG_ON_SCHEDULER_EXCEPTION="0", SGLANG_RUST_SERVER="0",
                          SGLANG_ENABLE_DISAGG_SAMPLING_MASK="0")
        # Controlled workers receive all commands over our RPC event loop.
        # Native tokenizer-input broadcast queues are unused in this mode.
        os.environ["SGLANG_USE_MESSAGE_QUEUE_BROADCASTER"] = "0"
        # Replay prep must run eagerly every step (it loads the bound grant).
        os.environ["SGLANG_ENABLE_METADATA_GLUE_GRAPH"] = "0"
        os.environ["PATH"] = str(Path(software["cuda_home"]) / "bin") + ":" + os.environ.get("PATH", "")
        for key in ("PYTORCH_ALLOC_CONF", "PYTORCH_CUDA_ALLOC_CONF"):
            if any(field.strip().lower() == "expandable_segments:true"
                   for field in os.environ.get(key, "").split(",")):
                raise IntegrationError("CUDA_IPC_ABI", "Expandable CUDA segments cannot be shared")
        os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:False")
        if config.runtime_dir:
            self.directory = safe_path(config.runtime_dir)
            self.directory.mkdir(mode=0o700, parents=False, exist_ok=False)
        else:
            self.directory = Path(tempfile.mkdtemp(prefix=f"mxs-{os.getuid()}-", dir=str(safe_path("/tmp"))))
        os.chmod(self.directory, 0o700)
        kv_sizing = {"mode": "kv_bytes_per_rank"}
        if config.kv_memory_fraction is not None:
            total = _cuda_total_memory(cards)
            kv_bytes = kv_bytes_from_fraction(config.kv_memory_fraction, total)
            if kv_bytes < 8192:
                raise IntegrationError("INVALID_CONFIG", "kv_memory_fraction leaves no KV blocks")
            kv_sizing = {"mode": "original_muxserve_fraction", "kv_memory_fraction": config.kv_memory_fraction,
                         "total_memory_bytes_device0": total, "kv_bytes_per_rank": kv_bytes}
            print("KV_POOL " + json.dumps(kv_sizing), flush=True)
            config = replace(config, kv_bytes_per_rank=kv_bytes, kv_memory_fraction=None)
        config = replace(config, runtime_dir=str(self.directory), models=tuple(
            replace(model, model_path=str(read_asset_path(model.model_path)),
                    tokenizer_path=str(read_asset_path(model.tokenizer_path))) for model in config.models))
        self.config = config
        self.config_path = self.directory / "config.json"
        _write_json(self.config_path, asdict(config))
        self.authkey = secrets.token_bytes(32)
        descriptor = os.open(self.directory / "auth.key", os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(descriptor, "wb") as output:
            output.write(self.authkey)
        _write_json(self.directory / "preflight.json", {"requested_devices": requested,
                    "rank_gpus": cards, "assets": assets, "software": software,
                    "config_fingerprint": config.fingerprint,
                    "kv_bytes_per_rank": config.kv_bytes_per_rank, "kv_sizing": kv_sizing,
                    "extra_headroom_bytes_per_rank": config.extra_headroom_bytes_per_rank})
        for name in ("mps", "mps-log"):
            (self.directory / name).mkdir(mode=0o700)
        os.environ["CUDA_MPS_PIPE_DIRECTORY"] = str(self.directory / "mps")
        os.environ["CUDA_MPS_LOG_DIRECTORY"] = str(self.directory / "mps-log")
        os.environ["CUDA_MPS_ACTIVE_THREAD_PERCENTAGE"] = "100"
        signal.signal(signal.SIGINT, lambda *_: self.stop_requested.set())
        signal.signal(signal.SIGTERM, lambda *_: self.stop_requested.set())
        signal.signal(signal.SIGQUIT, lambda *_: self.native_failure.set())

    def _mps_command(self, command):
        return subprocess.run(["nvidia-cuda-mps-control"], input=command + "\n", text=True,
                              capture_output=True, check=True, timeout=15)

    def _ports(self, model, stage):
        prefix = f"{model.model_id}-{stage}"
        def endpoint(name):
            path = self.directory / f"{prefix}-{name}"
            if len(os.fsencode(path)) > 100:
                raise IntegrationError("UNSAFE_ADDRESS", "Private native IPC path is too long")
            return "ipc://" + str(path)
        return dict(tokenizer_ipc_name=endpoint("tok"), scheduler_input_ipc_name=endpoint("in"),
                    detokenizer_ipc_name=endpoint("detok"), rpc_ipc_name=endpoint("rpc"),
                    metrics_ipc_name=endpoint("metrics"), tokenizer_worker_ipc_name=None,
                    decoupled_spec_ipc_config=None, instance_id=secrets.token_hex(6),
                    nccl_port=model.prefill_port if stage == "P" else model.decode_port)

    def _owner_command(self, model, rank, message):
        from .ipc import _bounded
        connection = self.weights[(model, rank)]["control"]
        timeout = self.config.prepare_timeout_ms / 1000
        _bounded(lambda: connection.send(message), timeout)
        if not connection.poll(timeout):
            raise IntegrationError("WEIGHT_OWNER_TIMEOUT", "Weight control result is uncertain")
        result = _bounded(connection.recv, timeout)
        if result.get("ok") is not True or result.get("op") != message["op"]:
            raise IntegrationError("WEIGHT_OWNER_PROTOCOL", "Weight owner control reply differs")
        return result

    def _wait(self, condition, description, timeout=None):
        deadline = time.monotonic() + (self.startup_timeout if timeout is None else timeout)
        deadline = min(deadline, self.startup_deadline)
        while not condition():
            self._observe()
            if self.stop_requested.is_set() or time.monotonic() >= deadline:
                raise IntegrationError("STARTUP_TIMEOUT", description)
            self.stop_requested.wait(0.1)

    def start(self):
        self.startup_deadline = time.monotonic() + self.startup_timeout
        subprocess.run(["nvidia-cuda-mps-control", "-d"], check=True, timeout=15,
                       capture_output=True, text=True)
        self.mps_started = True
        self._mps_command("get_server_list")
        from .storage import HeadwiseStore
        from .controller import MuxController
        from .gpu_storage import GPUStorage
        from .ipc import StorageRPCServer
        store = HeadwiseStore(
            {rank: self.config.pool_blocks_per_rank for rank in range(self.config.tp_size)},
            pool_mode=self.config.kv_pool_mode,
            batch_history_limit=self.config.batch_history_limit,
            model_ids=tuple(model.model_id for model in self.config.models))
        rates = dict(getattr(self.config, "policy_arrival_rates", ()) or ()) or None
        self.controller = MuxController(self.config, store, initializing=True, arrival_rates=rates)
        print("POLICY_LIMITS " + json.dumps({"rates": rates, "limits": dict(self.controller.policy.limits)}),
              flush=True)
        self.gpu = GPUStorage(store, {rank: rank for rank in range(self.config.tp_size)})
        self.rpc = StorageRPCServer(store, self.gpu, str(self.directory / "store.sock"), self.authkey,
            extra_methods=self.controller.rpc_methods(), owner_close=self._close_barrier,
            inline_methods=self.controller.rpc_inline_methods(),
            eager_io=self.config.rpc_inline_fast_methods)
        self.rpc_thread = Thread(target=self.rpc.serve_forever, name="owner-rpc", daemon=False)
        self.rpc_thread.start()
        from . import trace
        trace.start_sampler()  # MXS_TRACE_DIR only: owner CPU by thread and lock-free wake delay
        environment = dict(os.environ)
        for key in tuple(environment):
            if key.startswith("MUXSERVE_"):
                del environment[key]
        reserved_ports = {self.config.controller_port}
        for configured_model in self.config.models:
            reserved_ports.update((
                configured_model.prefill_port,
                configured_model.decode_port,
                configured_model.bootstrap_port,
            ))

        for model in self.config.models:
            weight_port = _free_loopback_port(reserved_ports)
            reserved_ports.add(weight_port)
            for rank in range(self.config.tp_size):
                parent, child = self.context.Pipe()
                address = str(self.directory / f"{model.model_id}-w{rank}.sock")
                process = self.context.Process(target=_weight_process,
                    args=(_native_fields(self.config, model, "W"), environment,
                          str(self.directory / f"{model.model_id}-w{rank}.log"), rank,
                          address, child, f"tcp://127.0.0.1:{weight_port}"), daemon=False)
                process.start()
                child.close()
                self.weights[(model.model_id, rank)] = {"process": process, "control": parent,
                                                        "port": weight_port}
            def owners_ready():
                return all((self.directory / f"{model.model_id}-w{rank}.ready").is_file()
                           for rank in range(self.config.tp_size))
            self._wait(owners_ready, f"Weight owners not ready for {model.model_id}")
            for stage in ("P", "D"):
                ports = self._ports(model, stage)
                for rank in range(self.config.tp_size):
                    child_environment = {**environment, "MUXSERVE_CONFIG": str(self.config_path),
                        "MUXSERVE_RUNTIME_DIR": str(self.directory), "MUXSERVE_MODEL_ID": model.model_id,
                        "MUXSERVE_STAGE": stage, "MUXSERVE_EXECUTION_GROUP": self.controller.group_id(model.model_id),
                        "CUDA_MPS_ACTIVE_THREAD_PERCENTAGE": str(100 if stage == "P" else model.decode_mps)}
                    gate_read, gate_write = self.context.Pipe(duplex=False)
                    ready_read, ready_write = self.context.Pipe(duplex=False)
                    process = self.context.Process(target=_rank_process,
                        args=(_native_fields(self.config, model, stage), ports, child_environment,
                              str(self.directory / f"{model.model_id}-{stage}-r{rank}.log"), rank,
                              gate_read, ready_write, self.startup_timeout), daemon=False)
                    process.start()
                    gate_read.close()
                    ready_write.close()
                    self.ranks[(model.model_id, stage, rank)] = {"process": process, "ready": ready_read,
                                                               "initialized": False}
                    self._owner_command(model.model_id, rank, {"op": "authorize", "pid": process.pid, "stage": stage})
                    gate_write.send(True)
                    gate_write.close()
            self._wait(lambda: self.controller.snapshot()["groups"][model.model_id]["ready"],
                       f"Native ranks not registered for {model.model_id}")
        self._wait(self._initialized, "Native initialization pipes did not report ready")
        self._collect_sharing()
        self._initialize()
        self._status("READY")
        print(f"Resolved configuration: {self.config_path}", flush=True)
        print(f"Mode: {'manual acceptance control' if self.manual_control else 'HTTP/event/ADBS serving'}", flush=True)

    def _initialize(self):
        from transformers import AutoTokenizer
        from .gateway import GatewayState, GatewayServer
        from .driver import ControllerDriver
        from .startup import run_initialization
        self._status("INITIALIZING")
        tokenizers = {model.model_id: AutoTokenizer.from_pretrained(model.tokenizer_path,
                      local_files_only=True, trust_remote_code=False) for model in self.config.models}
        _, initialization_driver = run_initialization(
            self.controller, tokenizers, self.cards, self.startup_deadline,
            self._observe, self.stop_requested,
            lambda evidence: _write_json(self.directory / "startup-evidence.json", evidence))
        # No background thread was started during initialization. Disable its
        # synchronous grant path before opening private manual admission.
        initialization_driver.pause()
        if self.stop_requested.is_set() or time.monotonic() >= self.startup_deadline:
            raise IntegrationError("STARTUP_TIMEOUT", "Initialization gate exceeded the startup deadline")
        self.controller.finish_initialization()
        self.startup_gate_complete = True
        if self.manual_control:
            return
        self.gateway_state = GatewayState(self.controller, tokenizers)
        self.driver = ControllerDriver(self.controller, self.gateway_state.apply_event,
                                       self.gateway_state.maintain)
        # Initialization already ACKed this service epoch's event history.
        for model_id, receiver in self.driver.receivers.items():
            receiver.last_seq = initialization_driver.receivers[model_id].last_seq
        self.gateway = GatewayServer(self.gateway_state, self.config.controller_port,
                                     maximum_connections=self.config.gateway_connections)
        self.driver.start()
        self.http_thread = Thread(target=self.gateway.serve_forever, name="http-gateway", daemon=False)
        self.http_thread.start()

    def _initialized(self):
        for item in self.ranks.values():
            if not item["initialized"] and item["ready"].poll():
                result = item["ready"].recv()
                if result.get("status") != "ready":
                    raise IntegrationError("NATIVE_STARTUP_FAILED", "Native readiness response differs")
                item["initialized"] = True
        return bool(self.ranks) and all(item["initialized"] for item in self.ranks.values())

    def _collect_sharing(self):
        pool_exports = self.gpu.sharing_evidence()
        collected = {}
        for model in self.config.models:
            for rank in range(self.config.tp_size):
                owner = self._owner_command(model.model_id, rank, {"op": "snapshot"})
                pairs = []
                for stage in ("P", "D"):
                    pid = self.ranks[(model.model_id, stage, rank)]["process"].pid
                    path = self.directory / f"{model.model_id}-{stage}-r{rank}-weights.json"
                    evidence = json.loads(safe_path(path).read_text(encoding="utf-8"))
                    if (evidence["pid"] != pid or evidence["rank"] != rank or evidence["stage"] != stage
                            or evidence["model_id"] != model.model_id or evidence["import_state"] != "MAPPED"
                            or evidence["group_id"] != self.controller.group_id(model.model_id)
                            or evidence["config_fingerprint"] != self.config.fingerprint
                            or owner["exports"].get(pid) != "SENT"):
                        raise IntegrationError("SHARING_EVIDENCE_MISMATCH", "Importer identity/state differs")
                    weights = evidence["weights"]
                    allocation_map = {name: value["allocation_digest"] for name, value in weights.items()
                                      if value["allocation_digest"] is not None}
                    if (not allocation_map or allocation_map != owner["allocation_evidence"].get(pid)
                            or not all(value["final_alias_matches"] for value in weights.values())):
                        raise IntegrationError("SHARING_EVIDENCE_MISMATCH", "Final weights differ from producer exports")
                    consumer = f"{self.controller.group_id(model.model_id)}:{stage}:rank{rank}"
                    expected = pool_exports[consumer][rank]
                    if (expected["state"] != "EXPORTED" or not evidence["kv"]["aliases_match"]
                            or any(evidence["kv"].get(key) != value for key, value in expected.items() if key != "state")):
                        raise IntegrationError("SHARING_EVIDENCE_MISMATCH", "Final KV differs from producer exports")
                    pairs.append(evidence)
                    collected[f"{model.model_id}:{stage}:{rank}"] = evidence
                left, right = pairs
                if ({name: value["allocation_digest"] for name, value in left["weights"].items()} !=
                        {name: value["allocation_digest"] for name, value in right["weights"].items()}
                        or any(left["kv"][name] != right["kv"][name] for name in
                               ("owner_id", "visible_device", "blocks", "key_allocation_digest", "value_allocation_digest"))):
                    raise IntegrationError("SHARING_EVIDENCE_MISMATCH", "P and D do not share producer storage")
        _write_json(self.directory / "sharing-evidence.json", collected)

    def _observe(self):
        if self.native_failure.is_set() or (self.rpc and self.rpc.failure_event.is_set()):
            raise IntegrationError("NATIVE_GROUP_FAILED", "Native SIGQUIT or owner listener failure")
        if self.driver and self.driver.failure:
            raise IntegrationError("DRIVER_FAILED", "External driver failed; retain producers")
        for item in self.weights.values():
            if item["process"].exitcode is not None:
                raise IntegrationError("WEIGHT_OWNER_FAILED", "Weight producer exited unexpectedly")
        for key, item in self.ranks.items():
            process = item["process"]
            if key in self.reaped or process.exitcode is None:
                continue
            process.join()
            model, stage, rank = key
            group_id = self.controller.group_id(model)
            close = self.controller.close_ready(group_id)
            if (process.exitcode == 0 and not close["safe_to_exit"]
                    and set(close["reasons"]) <= {"unacked_commands", "unacked_events", "rank_processes_alive"}):
                # A normally exiting rank can precede a sibling's final DRAIN
                # ACK. Retain its lease until the complete group proof arrives.
                continue
            if process.exitcode != 0 or not close["safe_to_exit"]:
                try:
                    self.controller.rank_failed(group_id, stage, rank, process.pid, "unexpected_process_exit")
                finally:
                    raise IntegrationError("NATIVE_GROUP_FAILED", "Rank exit lacks safe drain proof")
            try:
                proof = json.loads(safe_path(self.directory / f"{model}-{stage}-r{rank}-closed.json").read_text())
            except (OSError, ValueError) as exc:
                raise IntegrationError("CONSUMER_CLOSE_UNPROVEN", "Normal rank exit lacks import cleanup evidence") from exc
            if (any(proof.get(name) != value for name, value in
                    (("model_id", model), ("stage", stage), ("rank", rank), ("pid", process.pid)))
                    or any(proof.get(name) is not True for name in
                           ("model_collected", "kv_adapter_collected", "pool_closed"))):
                raise IntegrationError("CONSUMER_CLOSE_UNPROVEN", "Import cleanup identity/proof differs")
            self.controller.rank_exited(group_id, stage, rank, process.pid, process.exitcode)
            self.gpu.release_consumer(f"{group_id}:{stage}:rank{rank}", [rank])
            self._owner_command(model, rank, {"op": "exited", "pid": process.pid, "exit_code": 0})
            self.reaped.add(key)
        if self.controller and any(group["fenced"] for group in self.controller.snapshot()["groups"].values()):
            raise IntegrationError("NATIVE_GROUP_FAILED", "Controller fenced resources")

    def _close_barrier(self):
        if not self.ranks and not self.weights:
            self.gpu.close()  # Startup failed before any importer was spawned.
            return
        if (not self.controller or any(not self.controller.close_ready(self.controller.group_id(model.model_id))["ready"]
                for model in self.config.models) or any(item["process"].exitcode != 0 for item in self.weights.values())):
            raise IntegrationError("OWNER_BUSY", "Controller or weight producers have not safely closed")
        self.gpu.close()

    def _status(self, state, reason=None):
        if state == "READY" and not self.startup_gate_complete:
            raise IntegrationError("STARTUP_NOT_READY", "Warm-up and physical memory gate must pass before READY")
        _write_json(self.directory / "status.json", {"state": state, "reason": reason,
            "manual_control": self.manual_control, "config_fingerprint": self.config.fingerprint,
            "supervisor_pid": os.getpid(), "runtime_dir": str(self.directory),
            "config": str(self.config_path), "mps_started": self.mps_started,
            "startup_gate_complete": self.startup_gate_complete,
            "startup_evidence": str(self.directory / "startup-evidence.json"),
            "rank_pids": {":".join(map(str, key)): value["process"].pid for key, value in self.ranks.items()},
            "weight_pids": {":".join(map(str, key)): value["process"].pid for key, value in self.weights.items()},
            "weight_nccl_ports": {":".join(map(str, key)): value["port"] for key, value in self.weights.items()}})

    def drain(self):
        if self.gateway:
            self.gateway.begin_drain()
        if self.driver:
            self.driver.pause()
        else:
            from .driver import ControllerDriver
            self.driver = ControllerDriver(self.controller, lambda event: None, schedule=False)
            with self.controller.lock:
                for model_id, receiver in self.driver.receivers.items():
                    receiver.last_seq = self.controller.groups[model_id].outbox.acked_seq
            self.driver.start()
        for (model_id, request_id), output in self.controller.snapshot()["requests"].items():
            if (output["state"] not in ("FINISHED", "CANCELLED", "TIMED_OUT", "FAILED", "SUSPENDED")
                    and not output["pending_reason"] and not self.gateway):
                self.controller.cancel_request(model_id, request_id, "supervisor_shutdown")
        for model in self.config.models:
            self.controller.begin_drain(self.controller.group_id(model.model_id))
        deadline = time.monotonic() + self.config.request_timeout_ms / 1000
        while len(self.reaped) != len(self.ranks):
            self._observe()
            if time.monotonic() >= deadline:
                raise IntegrationError("DRAIN_TIMEOUT", "Rank teardown remains uncertain")
            time.sleep(0.1)
        self.driver.stop()
        if self.gateway:
            self.gateway.server_close()
            self.http_thread.join()
        for (model, rank), item in self.weights.items():
            self._owner_command(model, rank, {"op": "close"})
        for item in self.weights.values():
            item["process"].join(self.config.prepare_timeout_ms / 1000)
            if item["process"].exitcode != 0:
                raise IntegrationError("WEIGHT_CLOSE_UNCERTAIN", "Weight producer close did not finish")
        self.rpc.close()
        self.rpc_thread.join(self.config.prepare_timeout_ms / 1000)
        if self.rpc_thread.is_alive():
            raise IntegrationError("OWNER_CLOSE_UNCERTAIN", "KV RPC shutdown did not finish")
        self._finish_mps_shutdown("CLOSED")

    def _finish_mps_shutdown(self, final_state, reason=None):
        if not self.mps_started:
            self._status(final_state, reason)
            return
        from .mps_shutdown import start_finalizer
        self._status("FINALIZING_MPS", reason)
        start_finalizer(self.directory, final_state=final_state, reason=reason,
            gpu_closed=self.gpu is None or self.gpu._closed,
            rank_exitcodes={":".join(map(str, key)): item["process"].exitcode for key, item in self.ranks.items()},
            weight_exitcodes={":".join(map(str, key)): item["process"].exitcode for key, item in self.weights.items()})

    def run(self):
        try:
            self.start()
            while not self.stop_requested.wait(0.1):
                self._observe()
            self.drain()
        except BaseException as exc:
            if not self.ranks and not self.weights:
                if self.rpc:
                    self.rpc.close()
                    self.rpc_thread.join(self.config.prepare_timeout_ms / 1000)
                elif self.gpu:
                    self.gpu.close()
                self._finish_mps_shutdown("STARTUP_FAILED_WITHOUT_IMPORTERS", getattr(exc, "code", type(exc).__name__))
                raise
            self._protect_failure(getattr(exc, "code", type(exc).__name__))

    def _protect_failure(self, reason):
        # Keep producers alive even when diagnostics are the original failure,
        # or when a secondary failure occurs while stopping HTTP/CPU grants.
        try:
            actions = [lambda: self._status("PROTECTED_FAILURE", reason),
                       lambda: logger.error("Deployment failed (%s); protected process evidence: %s",
                                            reason, self.directory / "status.json")]
            if self.gateway:
                actions.insert(0, lambda: self.gateway_state.fail(reason))
            if self.driver:
                actions.insert(0, self.driver.pause)
            if self.controller:
                actions.insert(0, lambda: self.controller.protect_failure(reason))
            for action in actions:
                try:
                    action()
                except BaseException:
                    try:
                        os.write(2, b"Deployment failure: diagnostics/control unavailable; retaining producers.\n")
                    except BaseException:
                        pass
        finally:
            self._retain_producers()

    def _retain_producers(self):
        # No diagnostic I/O or cleanup belongs in this lifetime barrier.
        # Timeouts/interruption never authorize freeing live CUDA imports.
        while True:
            try:
                time.sleep(1)
            except BaseException:
                continue


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config")
    parser.add_argument("--startup-timeout", type=int, default=1800)
    parser.add_argument("--manual-control", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    Supervisor(args.config, startup_timeout=args.startup_timeout,
               manual_control=args.manual_control).run()


if __name__ == "__main__":
    main()
