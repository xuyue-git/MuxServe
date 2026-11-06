"""CPU checks for normal owner retirement and CUDA-free private MPS shutdown."""
import ast
from contextlib import redirect_stdout
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from muxserve_sglang.contracts import IntegrationError


class WeightCloseChecks(unittest.TestCase):
    def fixture(self):
        path = ROOT / 'muxserve_sglang/weight_owner.py'
        tree = ast.parse(path.read_text())
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'SharedWeightOwner')
        cls.bases = []
        cls.body = [n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == 'close_after_safe_drain']
        namespace = dict(IntegrationError=IntegrationError, json=json, os=os)
        exec(compile(ast.Module(body=[cls], type_ignores=[]), str(path), 'exec'), namespace)
        owner = namespace['SharedWeightOwner']()
        owner._safe_close = True
        owner.authorized, owner.exited, owner.exports = {10: 'P', 20: 'D'}, {10, 20}, {10: 'SENT', 20: 'SENT'}
        owner.model_path, owner.tp_rank, owner.gpu_id = '/models/llama7b', 0, 0
        owner.state_tensors = {'weight': object()}
        self.trace = []
        def shutdown():
            self.assertFalse(owner.state_tensors, 'Strong export table must be cleared while Torch is alive')
            self.trace.append('shutdown')
        owner.shutdown = shutdown
        torch = ModuleType('torch')
        torch.cuda = SimpleNamespace(synchronize=lambda device: self.trace.append('sync'),
            ipc_collect=lambda: self.trace.append('ipc_collect'), empty_cache=lambda: self.trace.append('empty_cache'))
        self.addCleanup(patch.stopall)
        factory = ModuleType('sglang.srt.layers.rotary_embedding.factory')
        factory._ROPE_DICT = {}
        patch.dict(sys.modules, {'torch': torch, 'sglang.srt.layers.rotary_embedding.factory': factory}).start()
        patch('gc.collect', side_effect=lambda: self.trace.append('gc')).start()
        return owner

    def test_exports_cleared_before_ipc_collection_after_both_consumers_exit(self):
        owner = self.fixture()
        output = io.StringIO()
        with redirect_stdout(output): owner.close_after_safe_drain()
        self.assertEqual(self.trace, ['sync', 'shutdown', 'gc', 'ipc_collect', 'empty_cache'])
        self.assertFalse(owner.state_tensors)
        self.assertTrue(output.getvalue().startswith('WEIGHT_OWNER_CLOSED '))

    def test_any_missing_exit_or_partial_export_keeps_weights(self):
        for field, value in (('_safe_close', False), ('exited', {10}), ('authorized', {10: 'P'}),
                             ('exports', {10: 'PREPARING'})):
            with self.subTest(field=field):
                owner = self.fixture()
                setattr(owner, field, value)
                with self.assertRaises(IntegrationError): owner.close_after_safe_drain()
                self.assertTrue(owner.state_tensors)
                self.assertFalse(self.trace)


class MPSShutdownChecks(unittest.TestCase):
    def fixture(self):
        from muxserve_sglang import mps_shutdown
        self.module = mps_shutdown
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name).resolve()
        os.chmod(root, 0o700)
        for name in ('mps', 'mps-log'): (root / name).mkdir(mode=0o700)
        state = dict(state='FINALIZING_MPS', supervisor_pid=os.getpid(), config_fingerprint='fingerprint', mps_started=True,
                     rank_pids={'P:0': 10, 'D:0': 20}, weight_pids={'w:0': 30})
        (root / 'status.json').write_text(json.dumps(state))
        proof = dict(supervisor_pid=os.getpid(), config_fingerprint='fingerprint', final_state='CLOSED',
                     reason=None, gpu_closed=True, rank_exitcodes={'P:0': 0, 'D:0': 0}, weight_exitcodes={'w:0': 0})
        (root / 'shutdown-proof.json').write_text(json.dumps(proof))
        self.trace = []
        self.addCleanup(patch.stopall)
        patch.object(mps_shutdown, '_wait_exit', side_effect=lambda fd, timeout: self.trace.append('owner_exited')).start()
        patch.object(mps_shutdown, '_mps_pids', side_effect=[[432], []]).start()
        return root

    def test_close_is_published_only_after_parent_exit_and_private_mps_cleanup(self):
        root = self.fixture()
        def query(command, environment):
            self.trace.append(command)
            self.assertEqual(environment['CUDA_VISIBLE_DEVICES'], '')
            self.assertEqual(environment['CUDA_MPS_PIPE_DIRECTORY'], str(root / 'mps'))
            return {'get_server_list': '432', 'get_client_list 432': '', 'quit': ''}[command]
        with patch.object(self.module, '_query', side_effect=query):
            self.module.finish(root, 17)
        self.assertEqual(self.trace, ['owner_exited', 'get_server_list', 'get_client_list 432', 'quit'])
        state = json.loads((root / 'status.json').read_text())
        self.assertEqual(state['state'], 'CLOSED')
        self.assertFalse(state['mps_started'])

    def test_live_client_prevents_quit_and_closed(self):
        root = self.fixture()
        commands = []
        def query(command, environment):
            commands.append(command)
            return '432' if command == 'get_server_list' else '999'
        with patch.object(self.module, '_query', side_effect=query), self.assertRaisesRegex(RuntimeError, 'MPS_CLIENTS_REMAIN'):
            self.module.finish(root, 17)
        self.assertNotIn('quit', commands)
        self.assertEqual(json.loads((root / 'status.json').read_text())['state'], 'FINALIZING_MPS')

    def test_changed_status_or_unproven_gpu_close_prevents_commands(self):
        for mutate in ('state', 'proof'):
            with self.subTest(mutate=mutate):
                root = self.fixture()
                if mutate == 'state':
                    path = root / 'status.json'; data = json.loads(path.read_text()); data['config_fingerprint'] = 'other'
                else:
                    path = root / 'shutdown-proof.json'; data = json.loads(path.read_text()); data['gpu_closed'] = False
                path.write_text(json.dumps(data))
                with patch.object(self.module, '_query') as query, self.assertRaises(RuntimeError):
                    self.module.finish(root, 17)
                query.assert_not_called()

    def test_start_passes_bound_pidfd_and_hides_cuda_from_finalizer(self):
        root = self.fixture()
        (root / 'shutdown-proof.json').unlink()
        with patch.object(self.module.subprocess, 'Popen') as start:
            start.return_value.pid = 99
            self.module.start_finalizer(root, final_state='CLOSED', reason=None, gpu_closed=True,
                rank_exitcodes={'P:0': 0, 'D:0': 0}, weight_exitcodes={'w:0': 0})
        call = start.call_args
        self.assertEqual(call.kwargs['env']['CUDA_VISIBLE_DEVICES'], '')
        self.assertEqual(len(call.kwargs['pass_fds']), 1)
        self.assertIn('-S', call.args[0])
        self.assertTrue(call.kwargs['start_new_session'])


@unittest.skipUnless(sys.platform == 'linux', 'Linux pidfd lifecycle')
class PidfdChecks(unittest.TestCase):
    def test_invalid_pidfd_cannot_prove_exit(self):
        from muxserve_sglang.mps_shutdown import _wait_exit
        descriptor = os.open(os.devnull, os.O_RDONLY)
        os.close(descriptor)
        with self.assertRaises(RuntimeError): _wait_exit(descriptor, 0.01)

    def test_real_cpu_child_must_exit_before_wait_completes(self):
        from muxserve_sglang.mps_shutdown import _wait_exit
        child = subprocess.Popen([sys.executable, '-I', '-S', '-c', 'import time; time.sleep(60)'])
        descriptor = os.pidfd_open(child.pid)
        try:
            with self.assertRaises(TimeoutError): _wait_exit(descriptor, 0.01)
            child.terminate()
            child.wait(5)
            _wait_exit(descriptor, 0.1)
        finally:
            os.close(descriptor)
            if child.poll() is None: child.kill(); child.wait(5)

    def test_real_cpu_supervisor_exits_before_detached_finalizer_publishes_closed(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary).resolve()
            os.chmod(directory, 0o700)
            for name in ('mps', 'mps-log'): (directory / name).mkdir(mode=0o700)
            source = '''import json, os, sys
from pathlib import Path
sys.path.insert(0, sys.argv[1])
from muxserve_sglang.mps_shutdown import start_finalizer
directory = Path(sys.argv[2])
state = dict(state='FINALIZING_MPS', supervisor_pid=os.getpid(), config_fingerprint='CPU-only',
    mps_started=True, rank_pids={'P:0': 10}, weight_pids={'w:0': 20})
(directory / 'status.json').write_text(json.dumps(state))
start_finalizer(directory, final_state='CLOSED', reason=None, gpu_closed=True,
    rank_exitcodes={'P:0': 0}, weight_exitcodes={'w:0': 0})
'''
            result = subprocess.run([sys.executable, '-I', '-S', '-c', source, str(ROOT), str(directory)],
                capture_output=True, text=True, timeout=10)
            self.assertEqual(result.returncode, 0, result.stderr)
            deadline = time.monotonic() + 10
            while json.loads((directory / 'status.json').read_text())['state'] == 'FINALIZING_MPS':
                self.assertLess(time.monotonic(), deadline, (directory / 'mps-shutdown.log').read_text())
                time.sleep(0.02)
            self.assertEqual(json.loads((directory / 'status.json').read_text())['state'], 'CLOSED')
            self.assertIn('PRIVATE_MPS_SHUTDOWN_COMPLETE', (directory / 'mps-shutdown.log').read_text())


if __name__ == '__main__': unittest.main(verbosity=2)
