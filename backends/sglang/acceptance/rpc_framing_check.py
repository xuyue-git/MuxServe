"""Real Linux AF_UNIX framing/authentication checks; no Torch or GPU."""
from __future__ import annotations

from contextlib import contextmanager
from multiprocessing.connection import Connection, Listener, answer_challenge, deliver_challenge
import os
from pathlib import Path
import pickle
import socket
import struct
import sys
import tempfile
from threading import Event, Thread
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from muxserve_sglang import ipc
from muxserve_sglang.contracts import IntegrationError
from muxserve_sglang.storage import HeadwiseStore


@contextmanager
def pair(timeout=1, eager=False):
    left, right = socket.socketpair()
    left.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 4096)
    connection, peer = Connection(left.detach()), Connection(right.detach())
    timed = ipc._DeadlineConnection(connection, time.monotonic()+timeout, eager=eager)
    try:
        yield timed, peer, connection
    finally:
        close = getattr(timed, 'close', None)
        if close is not None:
            close()
        peer.close()
        connection.close()


@unittest.skipUnless(sys.platform.startswith('linux'), 'Real AF_UNIX framing requires Linux')
class FramingChecks(unittest.TestCase):
    def test_normal_frame_io_does_not_create_helper_threads(self):
        with pair() as (timed, peer, connection), patch.object(ipc, 'Thread', wraps=Thread) as helper:
            peer.send_bytes(b'request')
            self.assertEqual(timed.recv_bytes(), b'request')
            timed.send_bytes(b'response')
            self.assertEqual(peer.recv_bytes(), b'response')
            self.assertEqual(helper.call_count, 0, 'Ordinary framed socket I/O must not spawn native-operation threads')

    def test_stdlib_connection_empty_and_large_frames_work_both_ways(self):
        for value in (b'', b'small', bytes(range(256))*8192):
            with self.subTest(size=len(value)), pair(timeout=3) as (timed, peer, connection):
                errors = []
                def round_trip():
                    try:
                        peer.send_bytes(value)
                        self.assertEqual(peer.recv_bytes(), value)
                    except BaseException as error:
                        errors.append(error)
                producer = Thread(target=round_trip)
                producer.start()
                self.assertEqual(timed.recv_bytes(), value)
                timed.send_bytes(value)
                producer.join(3)
                self.assertFalse(producer.is_alive())
                if errors:
                    raise errors[0]

    def test_extended_header_and_malformed_or_oversize_lengths(self):
        with pair() as (timed, peer, connection):
            os.write(peer.fileno(), struct.pack('!iQ', -1, 5)+b'hello')
            self.assertEqual(timed.recv_bytes(), b'hello')
        for header, code in ((struct.pack('!i', -2), 'RPC_PROTOCOL'),
                (struct.pack('!i', ipc._MAX_MESSAGE_BYTES+1), 'MESSAGE_TOO_LARGE'),
                (struct.pack('!iQ', -1, 2**64-1), 'MESSAGE_TOO_LARGE'),
                (struct.pack('!i', 7), 'MESSAGE_TOO_LARGE')):
            with self.subTest(header=header), pair() as (timed, peer, connection):
                os.write(peer.fileno(), header)
                with self.assertRaises(IntegrationError) as error:
                    timed.recv_bytes(6 if header == struct.pack('!i', 7) else None)
                self.assertEqual(error.exception.code, code)
        with pair() as (timed, peer, connection), patch.object(ipc, '_MAX_MESSAGE_BYTES', 6):
            with self.assertRaises(IntegrationError) as error:
                timed.send_bytes(b'1234567')
            self.assertEqual(error.exception.code, 'MESSAGE_TOO_LARGE')

    def test_partial_read_keeps_one_deadline(self):
        with pair(timeout=0.09) as (timed, peer, connection):
            stopped = Event()
            def drip():
                for value in struct.pack('!i', 5)+b'hello':
                    if stopped.is_set():
                        return
                    try:
                        os.write(peer.fileno(), bytes((value,)))
                    except OSError:
                        return
                    stopped.wait(0.035)
            producer = Thread(target=drip)
            producer.start()
            beginning = time.monotonic()
            try:
                with self.assertRaises(IntegrationError) as error:
                    timed.recv_bytes()
                self.assertEqual(error.exception.code, 'RPC_TIMEOUT')
                self.assertLess(time.monotonic()-beginning, 0.2)
            finally:
                stopped.set()
                producer.join(1)
                self.assertFalse(producer.is_alive())

    def test_eager_io_handles_partial_frames_and_stays_bounded(self):
        with pair(eager=True) as (timed, peer, connection):
            def later():
                time.sleep(0.03)
                os.write(peer.fileno(), struct.pack('!i', 5)+b'ab')
                time.sleep(0.03)
                os.write(peer.fileno(), b'cde')
            worker = Thread(target=later)
            worker.start()
            self.assertEqual(timed.recv_bytes(), b'abcde')  # nothing was readable at first
            worker.join(1)
            timed.send_bytes(b'response')
            self.assertEqual(peer.recv_bytes(), b'response')
        with pair(eager=True) as (timed, peer, connection):
            os.write(peer.fileno(), struct.pack('!i', 5)+b'ab')
            peer.close()
            with self.assertRaises((EOFError, OSError, IntegrationError)):
                timed.recv_bytes()
        with pair(timeout=0.08, eager=True) as (timed, peer, connection):  # silent peer
            beginning = time.monotonic()
            with self.assertRaises(IntegrationError) as error:
                timed.recv_bytes()
            self.assertEqual(error.exception.code, 'RPC_TIMEOUT')
            self.assertLess(time.monotonic()-beginning, 0.25)
        with pair(timeout=0.08, eager=True) as (timed, peer, connection):  # blocked write
            beginning = time.monotonic()
            with self.assertRaises(IntegrationError) as error:
                timed.send_bytes(b'x'*(2*1024**2))
            self.assertEqual(error.exception.code, 'RPC_TIMEOUT')
            self.assertLess(time.monotonic()-beginning, 0.25)
        with pair(timeout=0.01, eager=True) as (timed, peer, connection):  # readable, but expired
            peer.send_bytes(b'late')
            time.sleep(0.03)
            with self.assertRaises(IntegrationError) as error:
                timed.recv_bytes()
            self.assertEqual(error.exception.code, 'RPC_TIMEOUT')

    def test_partial_eof_and_blocked_write_are_bounded(self):
        with pair() as (timed, peer, connection):
            os.write(peer.fileno(), struct.pack('!i', 5)+b'ab')
            peer.close()
            with self.assertRaises((EOFError, OSError, IntegrationError)):
                timed.recv_bytes()
        with pair(timeout=0.08) as (timed, peer, connection):
            beginning = time.monotonic()
            with self.assertRaises(IntegrationError) as error:
                timed.send_bytes(b'x'*(2*1024**2))
            self.assertEqual(error.exception.code, 'RPC_TIMEOUT')
            self.assertLess(time.monotonic()-beginning, 0.25)

    def test_wrapped_fd_close_is_idempotent_and_preserves_original_connection(self):
        beginning = len(os.listdir('/proc/self/fd'))
        for _ in range(40):
            with pair() as (timed, peer, connection):
                timed.close()
                timed.close()
                peer.send_bytes(b'original still owns descriptor')
                self.assertEqual(connection.recv_bytes(), b'original still owns descriptor')
        self.assertEqual(len(os.listdir('/proc/self/fd')), beginning)

    def test_mutual_authentication_uses_stdlib_protocol_without_helper_threads(self):
        key = b'test-only-key-exactly-32-bytes!!!'
        with pair(timeout=2) as (timed, peer, connection):
            errors = []
            def server():
                try:
                    deliver_challenge(peer, key)
                    answer_challenge(peer, key)
                    self.assertEqual(peer.recv_bytes(), b'authenticated request')
                    peer.send_bytes(b'authenticated response')
                except BaseException as error:
                    errors.append(error)
            serving = Thread(target=server)
            serving.start()
            with patch.object(ipc, 'Thread', wraps=Thread) as helper:
                answer_challenge(timed, key)
                deliver_challenge(timed, key)
                timed.send_bytes(b'authenticated request')
                self.assertEqual(timed.recv_bytes(), b'authenticated response')
                self.assertEqual(helper.call_count, 0)
            serving.join(2)
            self.assertFalse(serving.is_alive())
            if errors:
                raise errors[0]

    def test_timed_out_native_operation_keeps_handler_pinned_until_completion(self):
        gate, entered, finished = Event(), Event(), Event()
        calls = []
        def slow():
            entered.set()
            calls.append('once')
            gate.wait(3)
            finished.set()
            return True
        def close_owner():
            if entered.is_set() and not finished.is_set():
                raise IntegrationError('OWNER_BUSY', 'Native operation still running')
        owner = SimpleNamespace(export_pool=lambda *a: None, release_consumer=lambda *a: None,
                                owner_snapshot=lambda: {}, close=close_owner)
        with tempfile.TemporaryDirectory(prefix='mx-frame-') as directory:
            os.chmod(directory, 0o700)
            address = str(Path(directory)/'owner.sock')
            key = os.urandom(32)
            server = ipc.StorageRPCServer(HeadwiseStore({0:8}), owner, address, key,
                timeout_s=0.08, extra_methods={'slow':slow}, owner_close=close_owner)
            serving = Thread(target=server.serve_forever, daemon=True)
            serving.start()
            try:
                with self.assertRaises(IntegrationError):
                    ipc.StorageRPCClient(address, key, 0.2).call('slow')
                self.assertTrue(entered.is_set())
                self.assertFalse(finished.is_set())
                self.assertFalse(server._handlers_done.is_set())
                with self.assertRaises(IntegrationError) as error:
                    server.close()
                self.assertEqual(error.exception.code, 'OWNER_BUSY')
                self.assertTrue(server._failure_thread.is_alive())
                self.assertEqual(calls, ['once'])
            finally:
                gate.set()
                self.assertTrue(finished.wait(1))
                self.assertTrue(server._handlers_done.wait(1))
                server.close()
                serving.join(2)
                server._failure_thread.join(2)
                self.assertFalse(serving.is_alive())
                self.assertFalse(server._failure_thread.is_alive())

    def test_partial_response_timeout_does_not_reconnect_or_repeat_method(self):
        gate, accepted, partial_sent = Event(), Event(), Event()
        methods, errors = [], []
        key = os.urandom(32)
        with tempfile.TemporaryDirectory(prefix='mx-frame-once-') as directory:
            os.chmod(directory, 0o700)
            address = str(Path(directory)/'owner.sock')
            listener = Listener(address, family='AF_UNIX', authkey=None)
            os.chmod(address, 0o600)
            def server():
                connection = None
                try:
                    connection = listener.accept()
                    accepted.set()
                    deliver_challenge(connection, key)
                    answer_challenge(connection, key)
                    request = pickle.loads(connection.recv_bytes())
                    methods.append(request['method'])
                    response = pickle.dumps({'ok':True,'result':'response'}, protocol=pickle.HIGHEST_PROTOCOL)
                    fragment = struct.pack('!i', len(response))+response[:len(response)//2]
                    self.assertEqual(os.write(connection.fileno(), fragment), len(fragment))
                    partial_sent.set()
                    gate.wait(1)
                except BaseException as error:
                    errors.append(error)
                finally:
                    if connection is not None:
                        connection.close()
            serving = Thread(target=server)
            serving.start()
            try:
                with patch.object(ipc, '_connect_unix', wraps=ipc._connect_unix) as connect:
                    with self.assertRaises(IntegrationError) as error:
                        ipc.StorageRPCClient(address, key, 0.08).call('exactly-once')
                    self.assertEqual(error.exception.code, 'RPC_TIMEOUT')
                    self.assertTrue(partial_sent.is_set())
                    self.assertEqual(connect.call_count, 1)
                    self.assertEqual(methods, ['exactly-once'])
            finally:
                gate.set()
                if not accepted.is_set():
                    with socket.socket(socket.AF_UNIX,socket.SOCK_STREAM) as wake:
                        wake.settimeout(0.2)
                        wake.connect(address)
                serving.join(2)
                listener.close()
                self.assertFalse(serving.is_alive())
            if errors:
                raise errors[0]


if __name__ == '__main__':
    unittest.main(verbosity=2)
