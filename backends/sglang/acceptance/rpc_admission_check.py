"""Real AF_UNIX checks for bounded owner admission; no Torch, models or GPU."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import BoundedSemaphore, Event, Thread
from types import SimpleNamespace
import os
import socket
import sys
import tempfile
import time
import threading
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from muxserve_sglang.contracts import IntegrationError
from muxserve_sglang.ipc import StorageRPCClient, StorageRPCServer
from muxserve_sglang.storage import HeadwiseStore


class ObservedCapacity(BoundedSemaphore):
    def __init__(self):
        super().__init__(32)
        self.waiter = Event()

    def acquire(self, *args, **kwargs):
        self.waiter.set()
        return super().acquire(*args, **kwargs)


@unittest.skipUnless(sys.platform.startswith('linux'), 'Real AF_UNIX owner requires Linux')
class AdmissionChecks(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(prefix='mx-admit-')
        os.chmod(self.directory.name, 0o700)
        self.address = str(Path(self.directory.name) / 'owner.sock')
        self.key = os.urandom(32)
        self.calls = []
        owner = SimpleNamespace(export_pool=lambda *a: None, release_consumer=lambda *a: None,
                                owner_snapshot=lambda: {}, close=lambda: None)
        self.server = StorageRPCServer(HeadwiseStore({0: 8}), owner, self.address,
            self.key, timeout_s=0.4, extra_methods={'once': self.once}, owner_close=owner.close)
        self.capacity = ObservedCapacity()
        self.server._workers = self.capacity
        self.held = 0
        self.serving = Thread(target=self.server.serve_forever, daemon=True)
        self.serving.start()

    def once(self, value):
        self.calls.append(value)
        return value

    def saturate(self):
        for _ in range(32):
            self.assertTrue(self.capacity.acquire(blocking=False))
            self.held += 1
        self.capacity.waiter.clear()

    def release_one(self):
        self.capacity.release()
        self.held -= 1

    def tearDown(self):
        while self.held:
            self.release_one()
        if self.serving.is_alive():
            self.server.close()
        self.serving.join(3)
        self.server._failure_thread.join(3)
        self.assertFalse(self.serving.is_alive())
        self.assertFalse(self.server._failure_thread.is_alive())
        self.directory.cleanup()

    def test_transient_full_capacity_keeps_connection_and_executes_once(self):
        self.saturate()
        with ThreadPoolExecutor(max_workers=1) as pool:
            result = pool.submit(StorageRPCClient(self.address, self.key, 1).call, 'once', 'one')
            self.assertTrue(self.capacity.waiter.wait(1))
            time.sleep(0.04)
            self.release_one()
            self.assertEqual(result.result(timeout=2), 'one')
        self.assertEqual(self.calls, ['one'])

    def test_persistent_full_capacity_stays_bounded_without_execution(self):
        self.saturate()
        started = time.monotonic()
        with self.assertRaises(IntegrationError) as caught:
            StorageRPCClient(self.address, self.key, 0.9).call('once', 'never')
        elapsed = time.monotonic() - started
        self.assertIn(caught.exception.code, ('RPC_UNAVAILABLE', 'RPC_TIMEOUT'))
        self.assertGreaterEqual(elapsed, 0.3, 'Owner discarded connection immediately under load')
        self.assertLess(elapsed, 1.3)
        self.assertEqual(self.calls, [])

    def test_shutdown_during_admission_does_not_dispatch_or_leak_slot(self):
        self.saturate()
        with ThreadPoolExecutor(max_workers=1) as pool:
            result = pool.submit(StorageRPCClient(self.address, self.key, 1).call, 'once', 'never')
            self.assertTrue(self.capacity.waiter.wait(1))
            self.server._stop.set()
            self.release_one()
            with self.assertRaises(IntegrationError):
                result.result(timeout=2)
        self.assertEqual(self.calls, [])
        self.serving.join(2)
        self.assertFalse(self.serving.is_alive())
        self.assertTrue(self.capacity.acquire(blocking=False), 'Admission token leaked on stop')
        self.capacity.release()

    def test_wait_and_authentication_share_one_server_deadline(self):
        self.saturate()
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as peer:
            peer.settimeout(1)
            peer.connect(self.address)
            self.assertTrue(self.capacity.waiter.wait(1))
            started = time.monotonic()
            time.sleep(0.24)
            self.release_one()
            # Leave authentication incomplete. Waiting for capacity must not
            # grant another full timeout after the handler finally starts.
            while peer.recv(4096):
                pass
            elapsed = time.monotonic() - started
        self.assertGreaterEqual(elapsed, 0.3)
        self.assertLess(elapsed, 0.58, 'Admission reset the authentication deadline')
        self.assertEqual(self.calls, [])

    def test_24_pollers_remain_usable_and_each_call_executes_once(self):
        def poll(rank):
            client = StorageRPCClient(self.address, self.key, 4)
            for index in range(8):
                self.assertEqual(client.call('once', (rank, index)), (rank, index))
        with ThreadPoolExecutor(max_workers=24) as pool:
            list(pool.map(poll, range(24)))
        self.assertEqual(len(self.calls), 192)
        self.assertEqual(len(set(self.calls)), 192)


    def test_persistent_connection_reaps_each_call_worker(self):
        # Finished but unjoined threads are not "alive", yet keep OS resources
        # (stacks); a persistent handler must join each call's worker.
        import muxserve_sglang.ipc as ipc_module
        started, joined = [], set()
        class Tracked(ipc_module.Thread):
            def start(self):
                started.append(self)
                return super().start()
            def join(self, *args, **kwargs):
                joined.add(id(self))
                return super().join(*args, **kwargs)
        client = StorageRPCClient(self.address, self.key, 4)
        client.call('once', 'warm')
        original = ipc_module.Thread
        ipc_module.Thread = Tracked
        try:
            for index in range(200):
                self.assertEqual(client.call('once', index), index)
        finally:
            ipc_module.Thread = original
        workers = [thread for thread in started if thread.name != 'storage-owner-lifetime-keeper']
        self.assertGreaterEqual(len(workers), 200)
        self.assertLessEqual(sum(id(thread) not in joined for thread in workers), 1)
        client.close()

@unittest.skipUnless(sys.platform.startswith('linux'), 'Real AF_UNIX owner requires Linux')
class InlineMethodChecks(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(prefix='mx-inline-')
        os.chmod(self.directory.name, 0o700)
        self.address = str(Path(self.directory.name) / 'owner.sock')
        self.key = os.urandom(32)
        self.calls = []
        self.owner = SimpleNamespace(export_pool=lambda *a: None, release_consumer=lambda *a: None,
                                     owner_snapshot=lambda: {}, close=lambda: None)
        methods = {'fast': self.fast, 'regular': self.fast, 'refused': self.refused, 'broken': self.broken}
        self.server = StorageRPCServer(HeadwiseStore({0: 8}), self.owner, self.address, self.key,
                                       timeout_s=0.4, extra_methods=methods, owner_close=self.owner.close,
                                       inline_methods=('fast', 'refused', 'broken'), eager_io=True)
        self.serving = Thread(target=self.server.serve_forever, daemon=True)
        self.serving.start()
        self.client = StorageRPCClient(self.address, self.key, 4)

    def tearDown(self):
        self.client.close()
        self.server.close()
        self.serving.join(3)
        self.server._failure_thread.join(3)
        self.assertFalse(self.serving.is_alive())
        self.directory.cleanup()

    def fast(self):
        name = threading.current_thread().name
        self.calls.append(name)
        return name

    def refused(self):
        raise IntegrationError('NOPE', 'refused')

    def broken(self):
        raise RuntimeError('secret detail')

    def test_inline_method_stays_on_the_connection_thread(self):
        inline = {self.client.call('fast') for _ in range(20)}
        regular = {self.client.call('regular') for _ in range(20)}
        self.assertEqual(len(inline), 1)
        self.assertEqual(len(regular), 20)  # any other method still gets its own worker
        self.assertFalse(inline & regular)

    def test_inline_failures_answer_like_worker_failures(self):
        for method, code in (('refused', 'NOPE'), ('broken', 'INTERNAL_ERROR')):
            with self.assertRaises(IntegrationError) as error:
                self.client.call(method)
            self.assertEqual(error.exception.code, code)
            self.assertNotIn('secret', str(error.exception))
        self.client.call('fast')  # the connection is still served

    def test_inline_method_obeys_the_operation_limit(self):
        import muxserve_sglang.ipc as ipc_module
        taken = 0
        while ipc_module._operation_slots.acquire(blocking=False):
            taken += 1
        try:
            self.assertEqual(taken, 32)
            with self.assertRaises(IntegrationError) as error:
                self.client.call('fast')
            # Like a worker call: by then the deadline has passed, so the reply
            # cannot be sent either and the client sees the connection end.
            self.assertIn(error.exception.code, ('RPC_TIMEOUT', 'RPC_UNAVAILABLE'))
            self.assertEqual(self.calls, [])
        finally:
            for _ in range(taken):
                ipc_module._operation_slots.release()
        self.client.call('fast')  # an uncertain call closed the connection; a new one works
        self.assertEqual(len(self.calls), 1)

    def test_only_explicit_extensions_may_run_inline(self):
        with self.assertRaises(IntegrationError) as error:
            StorageRPCServer(HeadwiseStore({0: 8}), self.owner, self.address + '2', self.key,
                             extra_methods={'fast': self.fast}, owner_close=self.owner.close,
                             inline_methods=('snapshot',))
        self.assertEqual(error.exception.code, 'INVALID_ARGUMENT')


if __name__ == '__main__':
    unittest.main(verbosity=2)
