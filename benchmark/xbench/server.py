#!/usr/bin/env python3
# /// script
# requires-python = ">=3.8"
# dependencies = []
# ///
"""Start and stop the MuxServe server for one paper Figure 5 point, for xbench to measure.

xbench starts and stops it for the hosted cases in benches.toml (this folder). From the CrossPool
checkout, on four free GPUs:
    CUDA_VISIBLE_DEVICES=0,1,2,3 uv run xbench run --catalog <this folder>/benches.toml --case 596bb910
xbench runs `server.py requirements` (4 GPUs, the three models on one HTTP listener), then
`server.py start --model MODEL@URL ... --settings FILE` on the GPUs it leased (CUDA_VISIBLE_DEVICES)
and, after measuring, SIGTERM to the process group. By hand, to look at a point without xbench:
    uv run benchmark/xbench/server.py start --alpha 0.7 --rate-scale 2.0 --gpus 0,1,2,3
    uv run benchmark/xbench/server.py stop     # also reclaims what an interrupted xbench run left
    uv run benchmark/xbench/server.py points   # the 20 points and their xbench cases
    uv run benchmark/xbench/server.py build    # compile the extensions once

Standard library only; the server runs in the Python environment named in benchmark/xbench/.env
(copy .env.example). `start` does the manual launch's steps:
  1. GPUs -> UUIDs: under a private MPS, CUDA_VISIBLE_DEVICES must hold UUIDs;
  2. a private MPS daemon in /tmp/mxo-$USER-mps;
  3. python -m muxserve.launch fig5-a0.7-s1.0.yaml ... --alpha A --rate-scale S --http-port P
     (P is the port xbench picked, or 18080 by hand).
The server writes log/xbench/<point>-<time>/server.log (per-rank logs in proc/). This launcher
shows that log, says when the HTTP entry is open, and when stopped (xbench's SIGTERM, Ctrl-C, or
`stop` from another terminal) stops the server and quits the MPS daemon. One server at a time:
both versions use the same internal ports and MPS directory.
"""
from __future__ import annotations

import argparse
import datetime
import getpass
import json
import os
import re
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path, PurePosixPath
from urllib.parse import urlsplit

# --- What differs between the original's copy of this file and the baseline's ---------------
VERSION = "baseline"
SCRIPT = "benchmark/xbench/server.py"
YAML = "benchmark/xbench/fig5-a0.7-s1.0.yaml"
ENV_HELP = "the SGLang environment (Python 3.12, SGLang fork xuyue-git/sglang-engine muxserve-v0.5.20)"
# Extensions built by tools/build_sglang_ext.py (`build`); start checks they exist.
EXTENSIONS = ("memory_manager", "batch_scheduler")
# Kernels the SGLang fork compiles on first use; `build` compiles them ahead of time.
PREBUILD = ("sglang.kernels.ops.attention.headwise",)


def engine_env(env: dict, python_env: str, cuda: str) -> None:
    """The variables the baseline runs were started with: SGLang env first, CUDA 13.2 nvcc on
    PATH, this repository's muxserve package ahead of anything installed."""
    env["PATH"] = os.pathsep.join([f"{python_env}/bin", f"{cuda}/bin", env["PATH"]])
    env["PYTHONPATH"] = str(REPO)
# ----------------------------------------------------------------------------------------------

REPO = Path(__file__).resolve().parents[2]
ENV_FILE = Path(__file__).resolve().parent / ".env"
CATALOG = SCRIPT.rsplit("/", 1)[0] + "/benches.toml"
USER = getpass.getuser()
MPS_DIR = f"/tmp/mxo-{USER}-mps"
STATE = Path(MPS_DIR) / "server.json"
READY = b"MuxScheduler HTTP entry on"
GPUS = 4  # the yaml: num_gpus 4, tensor parallel 4
MANUAL_HTTP_PORT = 18080
# torch master, MuxScheduler, FlexStore (the HTTP port is xbench's, or MANUAL_HTTP_PORT).
INTERNAL_PORTS = (39335, 29702, 33651)
SERVER_PROCS = re.compile(r"muxserve\.launch|muxserve\.entrypoint|muxserve_server\.py|torch\.distributed\.launch")

# Paper Figure 5 grid: (alpha, rate scale, xbench case prefix, requests). Same as benches.toml;
# the request counts are xbench's Poisson draws for the cases' seed 0.
POINTS = (
    (0.7, 1.0, "596bb910", 234), (0.7, 1.5, "c02fa7c4", 372), (0.7, 2.0, "2868bf58", 497), (0.7, 2.5, "e8508945", 624),
    (0.9, 1.0, "d9dc6559", 216), (0.9, 1.5, "97cb9b08", 344), (0.9, 2.0, "3c0ec869", 453), (0.9, 2.5, "265f3d0f", 564),
    (1.3, 1.0, "5f5d389e", 183), (1.3, 1.5, "00167f40", 289), (1.3, 2.0, "0f63505d", 399), (1.3, 2.5, "fb14e827", 484),
    (1.7, 1.0, "6b5b616f", 163), (1.7, 1.5, "ccaeeca5", 259), (1.7, 2.0, "12dfe781", 353), (1.7, 2.5, "586cf3a9", 435),
    (2.1, 1.0, "95bf0bd1", 149), (2.1, 1.5, "28586f45", 233), (2.1, 2.0, "29f90517", 319), (2.1, 2.5, "fbeaf556", 392),
)

# Inherited variables the server must not see: uv's or xbench's own environment, another
# toolchain, another MPS pipe or GPU list (the same list the experiment scripts cleared).
DROP = ("VIRTUAL_ENV", "PYTHONHOME", "PYTHONPATH", "PYTHONNOUSERSITE", "UV_ENV_FILE", "CUDA_HOME",
        "LD_LIBRARY_PATH", "CUDA_MPS_PIPE_DIRECTORY", "CUDA_MPS_LOG_DIRECTORY", "CUDA_VISIBLE_DEVICES",
        "MPS", "VLLM_PROC_LOG", "SGLANG_BUILD_RUST_EXTS", "SGLANG_PLUGINS")

# Messages go to stdout (the terminal, or xbench's logs/server.log); `requirements` keeps stdout
# for its JSON answer.
MESSAGES = sys.stdout


def say(message: str) -> None:
    print(f"[server.py {datetime.datetime.now():%H:%M:%S}] {message}", file=MESSAGES, flush=True)


def die(message: str) -> None:
    say(message)
    raise SystemExit(1)


def read_settings(args: argparse.Namespace) -> dict:
    values = {}
    if ENV_FILE.exists():
        for raw in ENV_FILE.read_text().splitlines():
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("export "):
                line = line[len("export "):]
            key, sep, value = line.partition("=")
            if sep:
                values[key.strip()] = os.path.expanduser(os.path.expandvars(value.strip().strip("'\"")))
    if getattr(args, "python_env", None):
        values["MUXSERVE_ENV"] = args.python_env
    if getattr(args, "cuda_home", None):
        values["CUDA_HOME"] = args.cuda_home
    for key, what in (("MUXSERVE_ENV", ENV_HELP), ("CUDA_HOME", "the CUDA toolkit")):
        if not values.get(key):
            die(f"{key} is not set: {what}. Copy {ENV_FILE.parent.relative_to(REPO)}/.env.example "
                f"to .env there and fill it in.")
    if not Path(values["MUXSERVE_ENV"], "bin", "python").exists():
        die(f"MUXSERVE_ENV={values['MUXSERVE_ENV']} has no bin/python")
    return values


def clean_env(python_env: str) -> dict:
    env = dict(os.environ)
    skip = {os.path.abspath(f"{python_env}/bin")}
    if sys.prefix != sys.base_prefix:  # uv's (or xbench's) environment for this launcher
        skip.add(os.path.abspath(os.path.dirname(sys.executable)))
    if env.get("VIRTUAL_ENV"):
        skip.add(os.path.abspath(f"{env['VIRTUAL_ENV']}/bin"))
    for key in DROP:
        env.pop(key, None)
    keep = [p for p in env.get("PATH", "").split(os.pathsep)
            if p and os.path.abspath(p) not in skip and not re.search(r"/cuda[^/]*/bin/?$", p)]
    env["PATH"] = os.pathsep.join(keep)
    return env


def served_models() -> tuple:
    """The yaml's checkpoints as xbench names them, <organisation>/<model>: the last two parts of
    each model path (muxserve/muxsched/http_scheduler.py names them the same way)."""
    paths = re.findall(r"^\s*model:\s*(\S+)\s*$", Path(REPO, YAML).read_text(), re.M)
    return tuple("/".join(PurePosixPath(path).parts[-2:]) for path in paths)


def find_point(alpha: float, scale: float) -> tuple:
    for point in POINTS:
        if abs(point[0] - alpha) < 1e-9 and abs(point[1] - scale) < 1e-9:
            return point
    die(f"alpha {alpha:g}, rate scale {scale:g} is not one of the 20 Figure 5 points; "
        f"see: uv run {SCRIPT} points")


def settings_point(path: str) -> tuple:
    """The point in xbench's --settings file: the case's launch_settings {alpha, rate_scale}."""
    try:
        values = json.loads(Path(path).read_text())
        extra = sorted(set(values) - {"alpha", "rate_scale"})
        alpha, scale = float(values["alpha"]), float(values["rate_scale"])
    except (OSError, ValueError, KeyError, TypeError) as error:
        die(f"--settings {path}: launch_settings must be {{alpha, rate_scale}} ({error!r})")
    if extra:
        die(f"--settings {path}: unknown launch_settings {extra}; only alpha and rate_scale")
    return find_point(alpha, scale)


def hosted_port(bindings: list) -> int:
    """xbench's --model MODEL@URL, one per served model: all three on one 127.0.0.1 listener."""
    urls = {}
    for binding in bindings:
        model, sep, url = binding.partition("@")
        if not sep or not model:
            die(f"--model {binding}: expected MODEL@URL")
        urls[model] = url
    models = served_models()
    if sorted(urls) != sorted(models):
        die(f"xbench asked for {sorted(urls)}; {YAML} serves {list(models)}")
    if len(set(urls.values())) != 1:
        die(f"the three models share one HTTP listener; xbench gave {sorted(set(urls.values()))}")
    url = next(iter(urls.values()))
    address = urlsplit(url)
    if address.scheme != "http" or address.hostname != "127.0.0.1" or not address.port \
            or address.path not in ("", "/"):
        die(f"expected http://127.0.0.1:PORT from xbench, got {url}")
    return address.port


def gpu_rows(gpus: list) -> list:
    """index, UUID and memory used of each GPU, given as indices or UUIDs."""
    try:
        out = subprocess.run(["nvidia-smi", "-i", ",".join(gpus), "--query-gpu=index,uuid,memory.used",
                              "--format=csv,noheader,nounits"], capture_output=True, text=True)
    except FileNotFoundError:
        return []
    if out.returncode != 0:
        die(f"nvidia-smi -i {','.join(gpus)} failed: {out.stdout.strip()} {out.stderr.strip()}")
    return [[cell.strip() for cell in line.split(",")] for line in out.stdout.splitlines() if line.strip()]


def port_busy(port: int) -> bool:
    with socket.socket() as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind(("", port))
        except OSError:
            return True
    return False


def mps(env: dict, *args: str, stdin: str | None = None, timeout: float = 15) -> int:
    try:
        return subprocess.run(["nvidia-cuda-mps-control", *args], env=env, input=stdin, text=True,
                              timeout=timeout).returncode
    except (FileNotFoundError, subprocess.TimeoutExpired) as error:
        say(f"nvidia-cuda-mps-control {' '.join(args)}: {error}")
        return 1


def mps_env() -> dict:
    env = dict(os.environ)
    env["CUDA_MPS_PIPE_DIRECTORY"] = f"{MPS_DIR}/nvidia-mps"
    env["CUDA_MPS_LOG_DIRECTORY"] = f"{MPS_DIR}/nvidia-log"
    return env


def server_processes() -> list:
    """This user's MuxServe server processes, any repository: (pid, working directory, command)."""
    found = []
    for name in os.listdir("/proc"):
        if not name.isdigit() or int(name) == os.getpid():
            continue
        try:
            if os.stat(f"/proc/{name}").st_uid != os.getuid():
                continue
            with open(f"/proc/{name}/cmdline", "rb") as f:
                command = f.read().replace(b"\0", b" ").decode(errors="replace").strip()
            if not SERVER_PROCS.search(command):
                continue
            found.append((int(name), os.path.realpath(f"/proc/{name}/cwd"), command))
        except OSError:
            continue
    return found


def repo_processes() -> list:
    return [p for p in server_processes() if p[1] == str(REPO)]


def group_alive(pgid: int) -> bool:
    try:
        os.killpg(pgid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def catch_signals(seen: list, names=("SIGINT", "SIGTERM", "SIGHUP")) -> None:
    """Record signals (their times) instead of dying, so the server is always stopped and the MPS
    daemon quit. A signal that is ignored (nohup) stays ignored."""
    for name in names:
        sig = getattr(signal, name)
        if signal.getsignal(sig) is not signal.SIG_IGN:
            signal.signal(sig, lambda signum, frame: seen.append(time.time()))


def stop_group(pgid: int, pump, signals_seen: list) -> bool:
    """Stop the server's process group as the experiment scripts did: SIGINT (Ctrl-C), then TERM,
    then HUP. Never SIGKILL. A further Ctrl-C (2 s or more into a wait) skips the rest of it."""
    for sig, seconds in ((signal.SIGINT, 30), (signal.SIGTERM, 10), (signal.SIGHUP, 5)):
        pump()
        if not group_alive(pgid):
            return True
        say(f"sending {signal.Signals(sig).name} to the server (process group {pgid})")
        try:
            os.killpg(pgid, sig)
        except ProcessLookupError:
            return True
        begin = time.time()
        while time.time() < begin + seconds and group_alive(pgid) \
                and not any(t >= begin + 2 for t in signals_seen):
            pump()
            time.sleep(0.5)
    pump()
    return not group_alive(pgid)


def stop_processes(steps: tuple, pump=lambda: None) -> bool:
    """Stop this repository's server processes one by one: for each (signal or None to only wait,
    seconds) step, signal the ones still running and wait until they are gone. Never SIGKILL."""
    for sig, seconds in steps:
        pump()
        left = repo_processes()
        if not left:
            return True
        if sig is not None:
            for pid, _, command in left:
                say(f"{signal.Signals(sig).name} to {pid}: {command[:100]}")
                try:
                    os.kill(pid, sig)
                except ProcessLookupError:
                    pass
        end = time.time() + seconds
        while time.time() < end and repo_processes():
            pump()
            time.sleep(0.5)
    pump()
    left = repo_processes()
    for pid, _, command in left:
        say(f"still running, check by hand: {pid} {command[:100]}")
    return not left


def stop_leftovers() -> None:
    """Server processes of this repository outside the process group (none expected)."""
    stop_processes(((signal.SIGTERM, 10), (signal.SIGHUP, 5)))


def quit_mps_if_unused(timeout: float = 15) -> None:
    others = server_processes()
    if others:
        say("MPS left running: server processes still use it: "
            + "; ".join(f"{pid} in {cwd}" for pid, cwd, _ in others[:5]))
        return
    if Path(MPS_DIR, "nvidia-mps", "control").exists():
        mps(mps_env(), stdin="quit\n", timeout=timeout)
        say(f"MPS daemon in {MPS_DIR} told to quit")


def read_state() -> dict | None:
    try:
        return json.loads(STATE.read_text())
    except (OSError, ValueError):
        return None


def versions_text() -> str:
    def git(*args: str) -> str:
        try:
            return subprocess.run(["git", "-C", str(REPO), *args], capture_output=True, text=True).stdout.strip()
        except FileNotFoundError:
            return "(git not found)"
    return (f"commit {git('log', '-1', '--format=%H %ci %s')}\n"
            f"modified tracked files:\n{git('status', '--short', '--untracked-files=no') or '(none)'}\n")


def ready_banner(point: tuple, gpus: list, port: int) -> None:
    line = "=" * 78
    print(f"\n{line}\nServer ready: {VERSION}, alpha {point[0]:.1f}, rate scale {point[1]:.1f}, "
          f"GPUs {','.join(gpus)}\n"
          f"HTTP entry http://127.0.0.1:{port}: GET /v1/models, POST /v1/completions (OpenAI).\n"
          f"The benches.toml cases start their own server (xbench hosted mode): stop this one\n"
          f"before running them. The scheduler exits by itself after 240 s without requests.\n"
          f"Stop: Ctrl-C here, or uv run {SCRIPT} stop\n{line}\n", flush=True)


def write_commands_record(command: list, environment: dict) -> None:
    """launch/commands.json in xbench's working directory: the command for xbench's records."""
    try:
        Path("launch").mkdir(exist_ok=True)
        Path("launch", "commands.json").write_text(
            json.dumps({"commands": [{"argv": command, "environment": environment}]}))
    except OSError as error:
        say(f"launch/commands.json not written ({error})")


def cmd_requirements(args: argparse.Namespace) -> int:
    """xbench's first call: the GPUs and HTTP listeners the server needs, as JSON on stdout."""
    global MESSAGES
    MESSAGES = sys.stderr
    if args.settings:
        settings_point(args.settings)
    print(json.dumps({"device_count": GPUS, "listeners": [list(served_models())]}), flush=True)
    return 0


def cmd_start(args: argparse.Namespace) -> int:
    hosted = args.model is not None
    if hosted:
        if args.alpha is not None or args.rate_scale is not None or args.gpus is not None:
            die("--model and --settings come from xbench; --alpha, --rate-scale and --gpus are for "
                "starting by hand")
        if not args.settings:
            die("--settings is missing: the case's launch_settings {alpha, rate_scale}")
        port = hosted_port(args.model)
        point = settings_point(args.settings)
        gpus = [g.strip() for g in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",") if g.strip()]
        if len(gpus) != GPUS:
            die(f"xbench leased {len(gpus)} GPUs (CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '')}); "
                f"the yaml needs {GPUS}")
    else:
        if args.alpha is None or args.rate_scale is None or args.gpus is None:
            die("by hand, start needs --alpha, --rate-scale and --gpus")
        port = MANUAL_HTTP_PORT
        point = find_point(args.alpha, args.rate_scale)
        gpus = [g.strip() for g in args.gpus.split(",") if g.strip()]
        if len(gpus) != GPUS or not all(g.isdigit() for g in gpus):
            die(f"--gpus needs four GPU indices, e.g. 0,1,2,3 (the yaml uses tensor parallel 4); got {args.gpus}")
    settings = read_settings(args)
    python_env, cuda = settings["MUXSERVE_ENV"], settings["CUDA_HOME"]
    state = read_state()
    if state and state.get("pid") and not args.dry_run and group_alive(state.get("pgid", state["pid"])):
        die(f"a server is already running: {state.get('version')} in {state.get('repo')} "
            f"(pid {state.get('pid')}). Stop it first, in that repository: "
            f"uv run {state.get('script', SCRIPT)} stop")
    missing = [name for name in EXTENSIONS if not list(Path(REPO, "muxserve").glob(f"{name}*.so"))]
    if missing and not args.dry_run:
        die(f"extensions not built: {', '.join(missing)}. Run once: uv run {SCRIPT} build")

    rows = gpu_rows(gpus)
    if rows:
        uuids = [row[1] for row in rows]
        for index, uuid, used in rows:
            note = "" if not used.isdigit() or int(used) <= 100 else "  <- already in use; coordinate first"
            say(f"GPU {index} {uuid} {used} MiB used{note}")
        if len(uuids) != GPUS:
            die(f"nvidia-smi found {len(uuids)} of the GPUs {','.join(gpus)}")
    elif args.dry_run:
        uuids = [g if g.startswith("GPU-") else f"<UUID of GPU {g}>" for g in gpus]
    else:
        die("nvidia-smi not found")
    # xbench reserved the HTTP port and released it for this launch; the others are fixed.
    busy = [p for p in (INTERNAL_PORTS if hosted else (port, *INTERNAL_PORTS)) if port_busy(p)]
    if busy and not args.dry_run:
        die(f"ports {busy} are in use: another server is running. Stop it first: uv run {SCRIPT} stop")

    logdir = Path(REPO, "log", "xbench",
                  f"a{point[0]:.1f}-s{point[1]:.1f}-{datetime.datetime.now():%Y%m%d-%H%M%S}")
    env = clean_env(python_env)
    engine_env(env, python_env, cuda)
    env.update(MPS=MPS_DIR, CUDA_MPS_PIPE_DIRECTORY=f"{MPS_DIR}/nvidia-mps",
               CUDA_MPS_LOG_DIRECTORY=f"{MPS_DIR}/nvidia-log", CUDA_VISIBLE_DEVICES=",".join(uuids),
               VLLM_PROC_LOG=str(logdir / "proc"))
    python = f"{python_env}/bin/python"
    command = [python, "-m", "muxserve.launch", "--nproc_per_node", str(GPUS), "--master-port",
               str(INTERNAL_PORTS[0]), YAML, "--server-port", str(INTERNAL_PORTS[1]), "--flexstore-port",
               str(INTERNAL_PORTS[2]), "--mps-dir", MPS_DIR, "--schedule-approach", "adbs",
               "--max-num-batched-tokens", "2048", "--http-port", str(port),
               "--alpha", repr(point[0]), "--rate-scale", repr(point[1])]
    changed = {k: v for k, v in env.items() if os.environ.get(k) != v}
    removed = sorted(k for k in os.environ if k not in env)
    started_by = f"xbench hosted case {point[2]}, HTTP port {port}" if hosted else \
        f"by hand, for xbench case {point[2]}, HTTP port {port}"
    record = (f"{VERSION}: alpha {point[0]:.1f}, rate scale {point[1]:.1f}, {started_by}, "
              f"GPUs {','.join(gpus)}\ncwd {REPO}\n"
              + "".join(f"export {k}={v}\n" for k, v in sorted(changed.items()))
              + "".join(f"unset {k}\n" for k in removed)
              + " ".join(command) + "\n" + versions_text())
    if args.dry_run:
        print(record, end="")
        say(f"dry run: nothing started; ports in use: {busy or 'none'}; extensions missing: "
            f"{missing or 'none'}; logs would go to {logdir}")
        return 0

    logdir.mkdir(parents=True)
    (logdir / "command.txt").write_text(record)
    if hosted:
        write_commands_record(command, changed)
    Path(env["CUDA_MPS_PIPE_DIRECTORY"]).mkdir(parents=True, exist_ok=True)
    Path(env["CUDA_MPS_LOG_DIRECTORY"]).mkdir(parents=True, exist_ok=True)
    if mps(env, "-d") != 0:
        die(f"could not start the MPS daemon in {MPS_DIR} (still running from before?). "
            f"Run: uv run {SCRIPT} stop")
    signals_seen = []
    try:
        check = subprocess.run([python, "-c", "import torch; n = torch.cuda.device_count(); "
                                f"print('visible GPUs', n); assert n == {GPUS}"], env=env, cwd=REPO)
        if check.returncode != 0:
            raise SystemExit(f"the server environment does not see {GPUS} GPUs")
        catch_signals(signals_seen)
        log = open(logdir / "server.log", "wb")
        # By hand the server gets its own process group, which Ctrl-C and `stop` signal. Under
        # xbench it stays in this launcher's group: xbench requires the HTTP listener there and
        # stops the whole group itself.
        server = subprocess.Popen(command, cwd=REPO, env=env, stdin=subprocess.DEVNULL, stdout=log,
                                  stderr=subprocess.STDOUT, start_new_session=not hosted,
                                  # Ctrl-C must stop it even when this launcher was started with
                                  # SIGINT ignored (nohup / a background job).
                                  preexec_fn=lambda: signal.signal(signal.SIGINT, signal.SIG_DFL))
    except BaseException:
        quit_mps_if_unused()
        raise
    try:
        group = os.getpgid(server.pid)
    except ProcessLookupError:  # exited at once; supervise reports it
        group = server.pid
    STATE.write_text(json.dumps({"repo": str(REPO), "version": VERSION, "script": SCRIPT, "pid": server.pid,
                                 "pgid": group, "hosted": hosted, "log": str(logdir),
                                 "alpha": point[0], "rate_scale": point[1]}))
    say(f"{VERSION} server started (pid {server.pid}, {started_by}); log {logdir.relative_to(REPO)}/server.log")
    return supervise(server, logdir / "server.log", point, gpus, port, hosted, signals_seen)


def supervise(server: subprocess.Popen, log_path: Path, point: tuple, gpus: list, port: int,
              hosted: bool, signals_seen: list) -> int:
    tail = {"text": b"", "ready": False}
    log = open(log_path, "rb")

    def pump() -> None:
        server.poll()  # reaps the main process, so the group check sees only live processes
        chunk = log.read()
        if not chunk:
            return
        try:
            sys.stdout.buffer.write(chunk)
            sys.stdout.flush()
        except OSError:
            pass
        text = tail["text"] + chunk
        if not tail["ready"] and READY in text:
            tail["ready"] = True
            if hosted:
                say(f"HTTP entry open on port {port}; xbench measures now")
            else:
                ready_banner(point, gpus, port)
        tail["text"] = text[-200:]

    while server.poll() is None and not signals_seen:
        pump()
        time.sleep(0.5)
    pump()
    if signals_seen:
        say("stopping the server")
    elif server.returncode is not None:
        say(f"main process exited (code {server.returncode}); stopping the rest of the server")
    if hosted:
        # xbench's SIGTERM reached every process of this group at once, and xbench sends SIGKILL
        # 30 s later; the MPS daemon is outside the group and would be left. Wait for the server,
        # stop what is left of it by process id, then quit MPS, all within 25 s. If the server
        # exited by itself, the rest of it gets SIGINT first, as Ctrl-C would.
        first = None if signals_seen else signal.SIGINT
        stop_processes(((first, 12), (signal.SIGTERM, 5), (signal.SIGHUP, 3)), pump)
        quit_mps_if_unused(timeout=5)
    else:
        if not stop_group(server.pid, pump, signals_seen):
            say(f"some server processes did not stop; check: ps -o pid,cmd -g {server.pid}")
        stop_leftovers()
        quit_mps_if_unused()
    state = read_state()
    if state and state.get("pid") == server.pid:
        STATE.unlink(missing_ok=True)
    say(f"stopped. Logs: {log_path.parent}")
    code = server.returncode
    return 1 if code is None else 128 - code if code < 0 else code  # killed by signal N -> 128 + N


def cmd_stop(args: argparse.Namespace) -> int:
    signals_seen = []
    catch_signals(signals_seen, ("SIGINT",))
    state = read_state()
    if state and state.get("repo") != str(REPO):
        die(f"the running server is the {state.get('version')} one in {state.get('repo')}; "
            f"stop it there: cd {state.get('repo')} && uv run {state.get('script', SCRIPT)} stop")
    if state:
        group = state.get("pgid", state["pid"])
        started = "xbench" if state.get("hosted") else "by hand"
        say(f"stopping the {VERSION} server started {started} (pid {state['pid']}, alpha "
            f"{state['alpha']:.1f}, rate scale {state['rate_scale']:.1f})")
        if not stop_group(group, lambda: None, signals_seen):
            say(f"some server processes did not stop; check: ps -o pid,cmd -g {group}")
    else:
        say("no server started by this launcher is recorded; reclaiming leftovers only")
    stop_leftovers()
    quit_mps_if_unused()
    if state and read_state() == state:
        STATE.unlink(missing_ok=True)
    left = server_processes()
    say("done: no MuxServe server processes left" if not left else
        f"done for this repository; other server processes of {USER}: "
        + "; ".join(f"{pid} in {cwd}" for pid, cwd, _ in left[:5]))
    return 0


def cmd_points(args: argparse.Namespace) -> int:
    print("alpha  rate scale  requests  xbench case")
    for alpha, scale, case, requests in POINTS:
        print(f"{alpha:5.1f}  {scale:10.1f}  {requests:8d}  {case}")
    print(f"\nxbench starts the server itself. From the CrossPool checkout:\n"
          f"    CUDA_VISIBLE_DEVICES=0,1,2,3 uv run xbench run --catalog {REPO}/{CATALOG} --case CASE\n"
          f"By hand: uv run {SCRIPT} start --alpha A --rate-scale S --gpus 0,1,2,3")
    return 0


def cmd_build(args: argparse.Namespace) -> int:
    settings = read_settings(args)
    env = clean_env(settings["MUXSERVE_ENV"])
    engine_env(env, settings["MUXSERVE_ENV"], settings["CUDA_HOME"])
    env["CUDA_HOME"] = settings["CUDA_HOME"]
    python = f"{settings['MUXSERVE_ENV']}/bin/python"
    for command in ([python, "tools/build_sglang_ext.py"], *([python, "-m", m] for m in PREBUILD)):
        say(" ".join(command))
        code = subprocess.run(command, cwd=REPO, env=env).returncode
        if code:
            return code
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(prog=f"uv run {SCRIPT}", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    start = commands.add_parser("start", help="start the server for one Figure 5 point (foreground)")
    start.add_argument("--alpha", type=float, help="by hand: 0.7, 0.9, 1.3, 1.7 or 2.1")
    start.add_argument("--rate-scale", type=float, help="by hand: 1.0, 1.5, 2.0 or 2.5")
    start.add_argument("--gpus", help="by hand: four GPU indices as nvidia-smi shows them, e.g. 0,1,2,3")
    start.add_argument("--model", action="append", metavar="MODEL@URL",
                       help="from xbench: a served model and its HTTP listener (one per model)")
    start.add_argument("--settings", help="from xbench: the case's launch_settings (JSON)")
    start.add_argument("--dry-run", action="store_true", help="print the environment and command only")
    requirements = commands.add_parser("requirements", help="for xbench: the GPUs and listeners (JSON)")
    requirements.add_argument("--settings", help="from xbench: the case's launch_settings (JSON)")
    commands.add_parser("stop", help="stop the server and its MPS daemon, from any terminal")
    commands.add_parser("points", help="list the 20 points and their xbench cases")
    with_env = [start]
    if EXTENSIONS:
        with_env.append(commands.add_parser("build", help="compile the extensions (tools/build_sglang_ext.py)"))
    for sub in with_env:
        sub.add_argument("--python-env", help="overrides MUXSERVE_ENV from .env")
        sub.add_argument("--cuda-home", help="overrides CUDA_HOME from .env")
    args = parser.parse_args()
    needs_linux = args.command == "stop" or (args.command == "start" and not args.dry_run)
    if needs_linux and not os.path.isdir("/proc"):
        die("start and stop run on the Linux server")
    return {"start": cmd_start, "stop": cmd_stop, "points": cmd_points, "build": cmd_build,
            "requirements": cmd_requirements}[args.command](args)


if __name__ == "__main__":
    raise SystemExit(main())
