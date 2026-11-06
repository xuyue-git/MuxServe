"""Owned native channel cleanup; real pending-message ZMQ check when available."""
from __future__ import annotations
import importlib.util
from pathlib import Path
import subprocess
import sys
import tempfile
import time
from types import SimpleNamespace
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from muxserve_sglang import sglang_hooks


class NativeChannelCloseChecks(unittest.TestCase):
    def close(self, scheduler):
        function = getattr(sglang_hooks, 'close_native_channels', None)
        self.assertIsNotNone(function, 'Explicit owned native IPC teardown is missing')
        return function(scheduler)

    def fixture(self, *, fail=False):
        trace = []
        class Context:
            def term(self):
                self_check.assertTrue(all(socket.closed for socket in sockets))
                trace.append('context')
        class Socket:
            def __init__(self, context): self.context, self.closed = context, False
            def close(self, linger=None):
                self_check.assertEqual(linger, 0, 'Pending unconnected messages must not block teardown')
                if fail: raise RuntimeError('uncertain socket close')
                self.closed = True
                trace.append('socket')
        self_check = self
        context = Context()
        sockets = [Socket(context) for _ in range(5)]
        global_context = SimpleNamespace(term=lambda: self.fail('Global context must not be terminated'))
        writer = SimpleNamespace(_ctx=global_context, close=lambda: trace.append('writer'))
        publisher = SimpleNamespace(close=lambda: trace.append('publisher'))
        channels = SimpleNamespace(recv_from_tokenizer=sockets[0], recv_from_rpc=sockets[1],
            send_to_tokenizer=SimpleNamespace(socket=sockets[2]),
            send_to_detokenizer=SimpleNamespace(socket=sockets[3]), send_metrics_from_scheduler=sockets[4])
        return SimpleNamespace(ipc_channels=channels, load_snapshot_writer=writer,
                               load_publisher=publisher), trace

    def test_close_all_owned_sockets_before_context_and_leave_global_context_alone(self):
        scheduler, trace = self.fixture()
        result = self.close(scheduler)
        self.assertEqual(trace.count('socket'), 5)
        self.assertGreater(trace.index('context'), max(index for index, value in enumerate(trace) if value == 'socket'))
        self.assertEqual(result['sockets'], 5)
        self.assertEqual(result['contexts'], 1)
        self.assertIn('writer', trace)
        self.assertIn('publisher', trace)
        self.assertIsNone(scheduler.load_snapshot_writer)

    def test_nonleader_with_no_native_sockets_closes_without_global_context(self):
        scheduler = SimpleNamespace(ipc_channels=SimpleNamespace(recv_from_tokenizer=None, recv_from_rpc=None,
            send_to_tokenizer=SimpleNamespace(socket=None), send_to_detokenizer=SimpleNamespace(socket=None),
            send_metrics_from_scheduler=None), load_snapshot_writer=None, load_publisher=None)
        self.assertEqual(self.close(scheduler), dict(sockets=0, contexts=0))

    def test_socket_close_error_cannot_certify_context_teardown(self):
        scheduler, trace = self.fixture(fail=True)
        with self.assertRaisesRegex(RuntimeError, 'uncertain socket close'): self.close(scheduler)
        self.assertNotIn('context', trace)

    @unittest.skipUnless(importlib.util.find_spec('zmq'), 'pyzmq not installed locally; required in server CPU check')
    def test_real_zmq_pending_messages_block_default_term_but_fixed_close_exits(self):
        source = r'''
import sys
import socket as tcp
from pathlib import Path
from types import SimpleNamespace
import zmq
sys.path.insert(0, sys.argv[1])
from muxserve_sglang.sglang_hooks import close_native_channels
context = zmq.Context()
listener = tcp.socket()
listener.bind(('127.0.0.1', 0))
listener.listen(1)
socket = context.socket(zmq.PUSH)
socket.connect('tcp://127.0.0.1:%s' % listener.getsockname()[1])
socket.send(b'pending native metadata', flags=zmq.NOBLOCK)
Path(sys.argv[3]).write_text('QUEUED')
if sys.argv[2] == 'old':
    socket.close()
    context.term()
else:
    channels = SimpleNamespace(recv_from_tokenizer=None, recv_from_rpc=None,
        send_to_tokenizer=SimpleNamespace(socket=socket), send_to_detokenizer=SimpleNamespace(socket=None),
        send_metrics_from_scheduler=None)
    close_native_channels(SimpleNamespace(ipc_channels=channels, load_snapshot_writer=None, load_publisher=None))
listener.close()
'''
        with tempfile.TemporaryDirectory() as temporary:
            for mode in ('old', 'fixed'):
                marker = Path(temporary) / mode
                process = subprocess.Popen([sys.executable, '-I', '-B', '-u', '-c', source,
                    str(ROOT), mode, str(marker)], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                try:
                    deadline = time.monotonic() + 5
                    while not marker.exists():
                        if process.poll() is not None or time.monotonic() >= deadline:
                            self.fail('ZMQ fixture failed before queueing: ' + repr(process.communicate(timeout=1)))
                        time.sleep(0.01)
                    if mode == 'old':
                        with self.assertRaises(subprocess.TimeoutExpired): process.communicate(timeout=0.2)
                    else:
                        stdout, stderr = process.communicate(timeout=3)
                        self.assertEqual(process.returncode, 0, stderr)
                finally:
                    if process.poll() is None: process.kill()  # Test-created CPU fixture only.
                    process.communicate(timeout=3)


if __name__ == '__main__':
    if '--require-zmq' in sys.argv:
        sys.argv.remove('--require-zmq')
        if not importlib.util.find_spec('zmq'): raise SystemExit('Required real ZMQ CPU check unavailable')
    unittest.main(verbosity=2)
