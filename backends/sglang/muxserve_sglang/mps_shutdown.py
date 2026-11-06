"""CPU-only private MPS finalizer, run after the CUDA-owning supervisor exits."""
from __future__ import annotations

import json
import os
from pathlib import Path
import select
import stat
import subprocess
import sys
import tempfile
import time


def _private(path):
    if path.resolve() != path or path.is_symlink():
        raise RuntimeError('UNSAFE_MPS_SHUTDOWN_PATH')
    info = path.stat()
    if not stat.S_ISDIR(info.st_mode) or info.st_uid != os.getuid() or info.st_mode & 0o077:
        raise RuntimeError('MPS_SHUTDOWN_REQUIRES_PRIVATE_OWNED_DIRECTORY')


def _read(path):
    if path.is_symlink() or not path.is_file() or path.stat().st_uid != os.getuid():
        raise RuntimeError('UNSAFE_SHUTDOWN_EVIDENCE')
    return json.loads(path.read_text(encoding='utf-8'))


def _write(path, value):
    descriptor, temporary = tempfile.mkstemp(prefix='.shutdown-', dir=path.parent)
    try:
        with os.fdopen(descriptor, 'w', encoding='utf-8') as output:
            json.dump(value, output, sort_keys=True, indent=2)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary): os.unlink(temporary)


def _proof_valid(proof, state):
    return (state.get('state') == 'FINALIZING_MPS'
        and proof.get('supervisor_pid') == state.get('supervisor_pid')
        and proof.get('config_fingerprint') == state.get('config_fingerprint')
        and proof.get('final_state') in ('CLOSED', 'STARTUP_FAILED_WITHOUT_IMPORTERS')
        and proof.get('gpu_closed') is True
        and isinstance(proof.get('rank_exitcodes'), dict) and isinstance(proof.get('weight_exitcodes'), dict)
        and set(proof['rank_exitcodes']) == set(state.get('rank_pids', {}))
        and set(proof['weight_exitcodes']) == set(state.get('weight_pids', {}))
        and all(type(code) is int and code == 0
                for code in (*proof['rank_exitcodes'].values(), *proof['weight_exitcodes'].values())))


def start_finalizer(directory, *, final_state, reason, gpu_closed, rank_exitcodes, weight_exitcodes):
    """Bind this live supervisor with pidfd; child never imports Torch/CUDA."""
    directory = Path(directory)
    _private(directory)
    state = _read(directory / 'status.json')
    proof = dict(supervisor_pid=os.getpid(), config_fingerprint=state.get('config_fingerprint'),
        final_state=final_state, reason=reason, gpu_closed=gpu_closed,
        rank_exitcodes=rank_exitcodes, weight_exitcodes=weight_exitcodes)
    if not _proof_valid(proof, state): raise RuntimeError('MPS_SHUTDOWN_DRAIN_PROOF_MISSING')
    path = directory / 'shutdown-proof.json'
    with path.open('x', encoding='utf-8') as output:
        json.dump(proof, output, sort_keys=True)
    os.chmod(path, 0o600)
    descriptor = os.pidfd_open(os.getpid())
    try:
        with (directory / 'mps-shutdown.log').open('xb') as output:
            process = subprocess.Popen([sys.executable, '-I', '-S', '-B', '-u',
                str(Path(__file__).resolve()), str(directory), str(descriptor)],
                env={**os.environ, 'CUDA_VISIBLE_DEVICES': ''}, stdin=subprocess.DEVNULL,
                stdout=output, stderr=subprocess.STDOUT, pass_fds=(descriptor,), start_new_session=True)
        print('MPS_FINALIZER_STARTED ' + json.dumps(dict(pid=process.pid, supervisor_pid=os.getpid())), flush=True)
        return process.pid
    finally:
        os.close(descriptor)


def _wait_exit(descriptor, timeout):
    poller = select.poll()
    poller.register(descriptor, select.POLLIN)
    events = poller.poll(max(1, int(timeout * 1000)))
    if not events:
        raise TimeoutError('CUDA_SUPERVISOR_STILL_ALIVE')
    if not any(mask & select.POLLIN for _, mask in events):
        raise RuntimeError('SUPERVISOR_PIDFD_WAIT_FAILED')


def _mps_pids(pipe, pidfds):
    result = []
    for process in Path('/proc').iterdir():
        if not process.name.isdigit(): continue
        descriptor = None
        try:
            if process.stat().st_uid != os.getuid(): continue
            if not (process / 'comm').read_text().startswith('nvidia-cuda-mps'): continue
            descriptor = os.pidfd_open(int(process.name))
            poller = select.poll()
            poller.register(descriptor, select.POLLIN)
            if poller.poll(0): continue
            # Bind metadata to this still-live process, not a reused numeric PID.
            if process.stat().st_uid != os.getuid(): continue
            if not (process / 'comm').read_text().startswith('nvidia-cuda-mps'): continue
            # Only this one fixed key is inspected; never log the environment.
            if os.fsencode('CUDA_MPS_PIPE_DIRECTORY=' + str(pipe)) in (process / 'environ').read_bytes().split(b'\0'):
                if poller.poll(0): continue
                result.append(int(process.name))
                pidfds[int(process.name)] = descriptor
                descriptor = None
        except (FileNotFoundError, ProcessLookupError): pass
        finally:
            if descriptor is not None: os.close(descriptor)
    return result


def _wait_mps_exit(pidfds, timeout):
    """After quit, wait on bound identities; exiting MPS may hide /proc files."""
    pending = set(pidfds.values())
    poller = select.poll()
    for descriptor in pending: poller.register(descriptor, select.POLLIN)
    deadline = time.monotonic() + timeout
    while pending:
        remaining = deadline - time.monotonic()
        if remaining <= 0: raise TimeoutError('PRIVATE_MPS_STILL_ALIVE')
        for descriptor, mask in poller.poll(max(1, int(remaining * 1000))):
            if not mask & select.POLLIN: raise RuntimeError('MPS_PIDFD_WAIT_FAILED')
            pending.discard(descriptor)
            poller.unregister(descriptor)


def _query(command, environment):
    result = subprocess.run(['nvidia-cuda-mps-control'], input=command + '\n', text=True,
        env=environment, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True, timeout=30)
    if command != 'quit' and any(not value.isdigit() for value in result.stdout.split()):
        raise RuntimeError('MPS_SHUTDOWN_QUERY_UNCERTAIN')
    return result.stdout


def finish(directory, descriptor):
    directory = Path(directory)
    _private(directory)
    for name in ('mps', 'mps-log'): _private(directory / name)
    proof = _read(directory / 'shutdown-proof.json')
    state = _read(directory / 'status.json')
    if not _proof_valid(proof, state): raise RuntimeError('MPS_SHUTDOWN_DRAIN_PROOF_MISSING')
    _wait_exit(descriptor, 120)  # CUDA context lifetime ends at actual process exit.
    if _read(directory / 'status.json') != state: raise RuntimeError('SHUTDOWN_STATUS_CHANGED')
    environment = {**os.environ, 'CUDA_VISIBLE_DEVICES': '',
        'CUDA_MPS_PIPE_DIRECTORY': str(directory / 'mps'),
        'CUDA_MPS_LOG_DIRECTORY': str(directory / 'mps-log')}
    pidfds = {}
    try:
        owned = _mps_pids(directory / 'mps', pidfds)
        if owned:
            servers = [int(value) for value in _query('get_server_list', environment).split()]
            for server in servers:
                if server not in owned: raise RuntimeError('UNVERIFIED_PRIVATE_MPS_SERVER')
                if _query('get_client_list ' + str(server), environment).split():
                    raise RuntimeError('MPS_CLIENTS_REMAIN; no quit sent')
            _query('quit', environment)
            _wait_mps_exit(pidfds, 20)
    finally:
        for descriptor in pidfds.values(): os.close(descriptor)
    state.update(state=proof['final_state'], reason=proof['reason'], mps_started=False,
                 mps_shutdown_confirmed=True)
    _write(directory / 'status.json', state)
    print('PRIVATE_MPS_SHUTDOWN_COMPLETE ' + json.dumps(dict(state=state['state'],
        supervisor_pid=proof['supervisor_pid'], private_mps_pids=owned)), flush=True)


if __name__ == '__main__':
    directory, descriptor = Path(sys.argv[1]), int(sys.argv[2])
    try:
        finish(directory, descriptor)
    except BaseException as exc:
        # No GPU memory is owned here; preserve a failed result instead of
        # reporting CLOSED or sending a forceful/global MPS command.
        print('MPS_SHUTDOWN_FAILED ' + json.dumps(dict(error_type=type(exc).__name__, error=str(exc))), flush=True)
        try:
            state, proof = _read(directory / 'status.json'), _read(directory / 'shutdown-proof.json')
            if _proof_valid(proof, state):
                state.update(state='PROTECTED_FAILURE', reason='MPS_SHUTDOWN_UNCERTAIN')
                _write(directory / 'status.json', state)
        finally:
            raise
    finally:
        os.close(descriptor)
