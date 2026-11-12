"""Real Linux pidfd MPS shutdown race, using CPU children and a private pipe."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from muxserve import mps_shutdown as module


@unittest.skipUnless(sys.platform == 'linux', 'Linux pidfd fixture')
class MPSExitChecks(unittest.TestCase):
    def fixture(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name).resolve()
        os.chmod(root, 0o700)
        for name in ('mps', 'mps-log'): (root / name).mkdir(mode=0o700)
        state = dict(state='FINALIZING_MPS', supervisor_pid=os.getpid(), config_fingerprint='cpu',
            rank_pids={'P:0':1}, weight_pids={'w:0':2}, mps_started=True)
        proof = dict(supervisor_pid=os.getpid(), config_fingerprint='cpu', final_state='CLOSED',
            reason=None, gpu_closed=True, rank_exitcodes={'P:0':0}, weight_exitcodes={'w:0':0})
        for name, value in (('status.json', state), ('shutdown-proof.json', proof)):
            (root / name).write_text(json.dumps(value))
        source = "import ctypes,time;ctypes.CDLL(None).prctl(15,b'nvidia-cuda-mps',0,0,0);print('READY',flush=True);time.sleep(120)"
        child = subprocess.Popen([sys.executable, '-I', '-B', '-u', '-c', source],
            env={**os.environ, 'CUDA_MPS_PIPE_DIRECTORY': str(root / 'mps')}, stdout=subprocess.PIPE)
        self.assertEqual(child.stdout.readline(), b'READY\n')
        def cleanup():
            if child.poll() is None: child.kill()
            child.wait(timeout=5)
            child.stdout.close()
        self.addCleanup(cleanup)
        return root, child

    def test_quit_waits_for_bound_exit_without_reading_tearing_down_environment(self):
        root, child = self.fixture()
        other_root, other = self.fixture()
        quitting = [False]
        calls = []
        read_bytes = Path.read_bytes
        def read(path):
            if path == Path('/proc') / str(child.pid) / 'environ' and quitting[0]:
                raise PermissionError('MPS exit makes environ unreadable')
            return read_bytes(path)
        def query(command, environment):
            calls.append(command)
            self.assertEqual(environment['CUDA_MPS_PIPE_DIRECTORY'], str(root / 'mps'))
            if command == 'get_server_list': return str(child.pid)
            if command.startswith('get_client_list '): return ''
            self.assertEqual(command, 'quit')
            quitting[0] = True
            timer = threading.Timer(0.15, child.terminate)
            timer.start()
            self.addCleanup(timer.join)
            return ''
        with patch.object(Path, 'read_bytes', read), patch.object(module, '_wait_exit'), \
             patch.object(module, '_query', side_effect=query):
            module.finish(root, 999)
        self.assertIsNotNone(child.poll(), 'CLOSED published before private MPS exit')
        self.assertEqual(json.loads((root / 'status.json').read_text())['state'], 'CLOSED')
        self.assertEqual(calls, ['get_server_list', 'get_client_list ' + str(child.pid), 'quit'])
        self.assertIsNone(other.poll(), 'Another private MPS instance was touched')

    def test_unverified_private_server_and_live_client_still_prevent_quit(self):
        for reason in ('server', 'client'):
            with self.subTest(reason=reason):
                root, child = self.fixture()
                calls = []
                def query(command, environment):
                    calls.append(command)
                    if command == 'get_server_list':
                        return str(child.pid if reason == 'client' else os.getpid())
                    return '123'
                with patch.object(module, '_wait_exit'), patch.object(module, '_query', side_effect=query), \
                     self.assertRaises(RuntimeError):
                    module.finish(root, 999)
                self.assertNotIn('quit', calls)
                self.assertIsNone(child.poll())
                self.assertEqual(json.loads((root / 'status.json').read_text())['state'], 'FINALIZING_MPS')

    def test_unreadable_identity_before_quit_still_fails_closed_without_fd_leak(self):
        root, child = self.fixture()
        before = set(os.listdir('/proc/self/fd'))
        original = Path.read_bytes
        def read(path):
            if path == Path('/proc') / str(child.pid) / 'environ':
                raise PermissionError('Identity unavailable before quit')
            return original(path)
        with patch.object(Path, 'read_bytes', read), patch.object(module, '_wait_exit'), \
             patch.object(module, '_query') as query, self.assertRaises(PermissionError):
            module.finish(root, 999)
        query.assert_not_called()
        self.assertEqual(set(os.listdir('/proc/self/fd')), before)
        self.assertIsNone(child.poll())

    def test_exit_timeout_prevents_closed_and_releases_bound_pidfds(self):
        root, child = self.fixture()
        before = set(os.listdir('/proc/self/fd'))
        wait = module._wait_mps_exit
        def query(command, environment):
            return str(child.pid) if command == 'get_server_list' else ''
        with patch.object(module, '_wait_exit'), patch.object(module, '_query', side_effect=query), \
             patch.object(module, '_wait_mps_exit', side_effect=lambda fds, timeout: wait(fds, 0.01)), \
             self.assertRaisesRegex(TimeoutError, 'PRIVATE_MPS_STILL_ALIVE'):
            module.finish(root, 999)
        self.assertEqual(set(os.listdir('/proc/self/fd')), before)
        self.assertEqual(json.loads((root / 'status.json').read_text())['state'], 'FINALIZING_MPS')
        self.assertIsNone(child.poll())


if __name__ == '__main__': unittest.main(verbosity=2)
