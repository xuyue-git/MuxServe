"""Authenticated local AF_UNIX RPC for one long-lived storage owner.

Pickle is accepted only after multiprocessing mutual authentication and only on a
private local Unix socket. A timeout is an uncertain result: clients must fence a
running reservation and retry idempotent operations, never assume rollback.
"""
from __future__ import annotations

import errno
from dataclasses import dataclass
import os
from pathlib import Path
import pickle
from queue import Empty, Queue
import select
import socket
import stat
import struct
from threading import BoundedSemaphore, Event, Lock, Thread, current_thread
import time
from typing import Any, Callable
from multiprocessing import AuthenticationError
from multiprocessing.connection import Connection, Listener, answer_challenge, deliver_challenge

from .contracts import IntegrationError
from . import trace
from .storage import HeadwiseStore


_MAX_MESSAGE_BYTES = 64 * 1024 * 1024
# Count native owner operations whose Python callers may already have reached
# a deadline. The worker releases its slot only when it actually exits.
_operation_slots = BoundedSemaphore(32)


def _validate_endpoint(address: str, *, create: bool = False) -> str:
    if os.name != "posix":
        raise IntegrationError("IPC_PLATFORM", "Storage RPC requires Linux local AF_UNIX transport")
    if not isinstance(address, str) or not address or "\x00" in address:
        raise IntegrationError("UNSAFE_ADDRESS", "Absolute local Unix socket path required")
    path = Path(address)
    if not path.is_absolute() or ".." in path.parts or len(os.fsencode(address)) > 100:
        raise IntegrationError("UNSAFE_ADDRESS", "Unsafe or overly long local Unix socket address")
    parent = path.parent
    if create and not parent.exists():
        parent.mkdir(mode=0o700, parents=False)
    if not parent.is_dir() or parent.is_symlink():
        raise IntegrationError("UNSAFE_ADDRESS", "Private socket directory is required")
    info = parent.stat()
    if info.st_uid != os.getuid() or stat.S_IMODE(info.st_mode) & 0o077:
        raise IntegrationError("UNSAFE_ADDRESS", "Socket directory must be owned by this UID with mode 0700")
    if create and (path.exists() or path.is_symlink()):
        raise IntegrationError("ADDRESS_IN_USE", "Refusing to replace an existing socket path")
    if not create:
        info = path.lstat()
        if not stat.S_ISSOCK(info.st_mode) or info.st_uid != os.getuid() or info.st_mode & 0o077:
            raise IntegrationError("UNSAFE_ADDRESS", "Socket must be private and owned by this UID")
    return str(path)


def _validate_options(authkey: bytes, timeout_s: float) -> None:
    if not isinstance(authkey, bytes) or len(authkey) < 32:
        raise IntegrationError("INVALID_ARGUMENT", "A random authentication key of at least 32 bytes is required")
    if not isinstance(timeout_s, (int, float)) or not 0 < timeout_s <= 300:
        raise IntegrationError("INVALID_ARGUMENT", "RPC deadline must be between zero and 300 seconds")


class _DeadlineConnection:
    """Stdlib Connection frames on an AF_UNIX socket, with one fixed deadline.

    A duplicate owns only this wrapper's descriptor. MSG_DONTWAIT leaves the
    original Connection's shared descriptor flags untouched, while poll bounds
    every partial frame read/write without creating a helper thread.
    """
    def __init__(self, connection: Connection, deadline: float,
                 pending: list[Thread] | None = None):
        self.connection = connection
        self.deadline = deadline
        self.pending = pending
        self._poller = select.poll()
        duplicate = os.dup(connection.fileno())
        try:
            self._socket = socket.socket(fileno=duplicate)
        except BaseException:
            os.close(duplicate)
            raise

    def close(self) -> None:
        self._socket.close()

    def remaining(self) -> float:
        remaining = self.deadline - time.monotonic()
        if remaining <= 0:
            raise IntegrationError("RPC_TIMEOUT", "Local RPC deadline expired; result may be uncertain")
        return remaining

    def _wait(self, event: int) -> None:
        self._poller.register(self._socket, event)
        while True:
            # poll uses integer milliseconds; retain the same absolute deadline
            # after wakeups, interruptions and partial transfers.
            timeout_ms = max(1, int(self.remaining() * 1000 + 0.999))
            if self._poller.poll(timeout_ms):
                self.remaining()
                return

    def wait_readable(self, timeout_s: float) -> bool:
        """Idle wait between requests on a persistent connection (no deadline)."""
        self._poller.register(self._socket, select.POLLIN)
        return bool(self._poller.poll(max(1, int(timeout_s * 1000))))

    def _receive(self, length: int) -> bytes:
        data = bytearray(length)
        view = memoryview(data)
        offset = 0
        while offset < length:
            self._wait(select.POLLIN)
            try:
                count = self._socket.recv_into(view[offset:], length-offset, socket.MSG_DONTWAIT)
            except (BlockingIOError, InterruptedError):
                continue
            if not count:
                raise EOFError("Local RPC frame ended before completion")
            offset += count
        return bytes(data)

    def _send(self, value: bytes) -> None:
        view = memoryview(value)
        offset = 0
        while offset < len(view):
            self._wait(select.POLLOUT)
            try:
                count = self._socket.send(view[offset:], socket.MSG_DONTWAIT)
            except (BlockingIOError, InterruptedError):
                continue
            if not count:
                raise EOFError("Local RPC frame could not be sent")
            offset += count

    def recv_bytes(self, maxlength: int | None = None) -> bytes:
        if maxlength is not None and maxlength < 0:
            raise ValueError("Negative maximum frame length")
        length, = struct.unpack("!i", self._receive(4))
        if length == -1:
            length, = struct.unpack("!Q", self._receive(8))
        elif length < 0:
            raise IntegrationError("RPC_PROTOCOL", "Invalid local RPC frame length")
        maximum = _MAX_MESSAGE_BYTES if maxlength is None else min(maxlength, _MAX_MESSAGE_BYTES)
        if length > maximum:
            raise IntegrationError("MESSAGE_TOO_LARGE", "Local RPC frame exceeds maximum size")
        return self._receive(length)

    def send_bytes(self, value: bytes) -> None:
        length = len(value)
        if length > _MAX_MESSAGE_BYTES:
            raise IntegrationError("MESSAGE_TOO_LARGE", "Local RPC frame exceeds maximum size")
        header = struct.pack("!i", length)
        if length <= 16384:
            self._send(header + value)
        else:
            self._send(header)
            self._send(value)


def _bounded(operation: Callable[[], Any], timeout: float,
             connection: Connection | None = None,
             pending: list[Thread] | None = None) -> Any:
    deadline = time.monotonic() + timeout
    result: Queue = Queue(maxsize=1)
    if not _operation_slots.acquire(timeout=max(0, timeout)):
        raise IntegrationError("RPC_TIMEOUT", "Local RPC worker capacity deadline expired")
    if time.monotonic() >= deadline:
        _operation_slots.release()
        raise IntegrationError("RPC_TIMEOUT", "Local RPC deadline expired before admission")
    def work() -> None:
        try:
            result.put((True, operation()))
        except Exception as exc:
            result.put((False, exc))
        finally:
            _operation_slots.release()
    worker = Thread(target=work, daemon=True)
    try:
        worker.start()
    except BaseException:
        _operation_slots.release()
        raise
    if pending is not None:
        pending.append(worker)
    try:
        ok, value = result.get(timeout=max(0, deadline - time.monotonic()))
    except Empty as exc:
        if connection is not None:
            connection.close()
        raise IntegrationError("RPC_TIMEOUT", "Local RPC deadline expired; result may be uncertain") from exc
    if not ok:
        raise value
    return value


def _encode(value: Any) -> bytes:
    payload = pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL)
    if len(payload) > _MAX_MESSAGE_BYTES:
        raise IntegrationError("MESSAGE_TOO_LARGE", "Local RPC payload exceeds maximum size")
    return payload


def copy_message(value: Any) -> Any:
    """Detach trusted CPU metadata using the existing bounded wire codec.

    Python deepcopy walks every KV block repeatedly; the C pickle codec keeps
    the same lists, tuples and internal aliases while isolating mutable state.
    Only already-authenticated messages or owner-generated metadata belong here.
    """
    return pickle.loads(_encode(value))


@dataclass(frozen=True)
class CommandSnapshot:
    """Retain the complete command as immutable bytes, decode only for delivery.

    The small index serves ACK/retirement checks; every delivery still contains
    all ranks and owns fresh mutable tables. The wire format stays unchanged.
    """
    payload: bytes
    kind: str
    batch_id: str | None
    request_ids: frozenset[str]

    @classmethod
    def capture(cls, command: dict) -> CommandSnapshot:
        references = {item.get('request_id') for item in command.get('requests', ())}
        references.update(command.get('request_ids', ()))
        references.add(command.get('request_id'))
        references.discard(None)
        return cls(_encode(command), command['kind'], command.get('batch_id'), frozenset(references))

    def detach(self) -> dict:
        return pickle.loads(self.payload)

    def __getitem__(self, key):
        if key == 'kind': return self.kind
        if key == 'batch_id': return self.batch_id
        return self.detach()[key]

    def get(self, key, default=None):
        if key in ('kind', 'batch_id'): return self[key]
        return self.detach().get(key, default)

    def references(self, request_id: str) -> bool:
        return request_id in self.request_ids


def _connect_unix(address: str, deadline: float) -> socket.socket:
    """Retry only queue congestion before authentication or any RPC is sent."""
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise socket.timeout("Local RPC connect deadline expired")
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        try:
            sock.settimeout(remaining)
            sock.connect(address)
            sock.settimeout(None)
            return sock
        except BlockingIOError as exc:
            sock.close()
            if exc.errno not in (errno.EAGAIN, errno.EWOULDBLOCK):
                raise
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise socket.timeout("Local RPC connect deadline expired") from exc
            time.sleep(min(0.01, remaining))
        except BaseException:
            sock.close()
            raise


class StorageRPCClient:
    """Mutually authenticated client that keeps one connection per process.

    Authentication happens once per connection, not once per call. Any
    transport, framing or deadline error closes the connection: the result of
    that call stays uncertain exactly as before, and the next call reconnects
    and authenticates again. A forked child never reuses its parent's socket.
    """

    def __init__(self, address: str, authkey: bytes, timeout_s: float = 30.0):
        _validate_options(authkey, timeout_s)
        self.address = _validate_endpoint(address)
        self._authkey = authkey
        self.timeout_s = timeout_s
        self._lock = Lock()
        self._state: tuple[int, Connection, _DeadlineConnection] | None = None

    def _drop(self) -> None:
        state, self._state = self._state, None
        if state is None:
            return
        pid, conn, timed = state
        timed.close()
        conn.close()

    def _connection(self, deadline: float) -> _DeadlineConnection:
        if self._state is not None and self._state[0] != os.getpid():
            # Inherited descriptor: closing it here only affects this process.
            self._drop()
        if self._state is not None:
            timed = self._state[2]
            timed.deadline = deadline
            return timed
        _validate_endpoint(self.address)
        sock = conn = timed = None
        try:
            sock = _connect_unix(self.address, deadline)
            conn = Connection(sock.detach())
            timed = _DeadlineConnection(conn, deadline)
            answer_challenge(timed, self._authkey)
            deliver_challenge(timed, self._authkey)
        except BaseException:
            if timed is not None:
                timed.close()
            if conn is not None:
                conn.close()
            if sock is not None:
                sock.close()
            raise
        self._state = (os.getpid(), conn, timed)
        return timed

    def close(self) -> None:
        with self._lock:
            self._drop()

    def __del__(self):
        try:
            self._drop()
        except Exception:
            pass

    def call(self, method: str, *args: Any, **kwargs: Any) -> Any:
        if not trace.enabled():
            return self._call(method, *args, **kwargs)
        started = time.monotonic()
        try:
            return self._call(method, *args, **kwargs)
        finally:
            trace.ev("rpc", None, m=method, ms=(time.monotonic() - started) * 1000)

    def _call(self, method: str, *args: Any, **kwargs: Any) -> Any:
        if not isinstance(method, str) or not method or method.startswith("_"):
            raise IntegrationError("INVALID_ARGUMENT", "Explicit RPC method required")
        with self._lock:
            deadline = time.monotonic() + self.timeout_s
            try:
                request = _encode({"method": method, "args": args, "kwargs": kwargs})
                timed = self._connection(deadline)
                timed.send_bytes(request)
                response = pickle.loads(timed.recv_bytes(_MAX_MESSAGE_BYTES))
                if not isinstance(response, dict) or "ok" not in response:
                    raise IntegrationError("RPC_PROTOCOL", "Malformed storage response")
            except IntegrationError:
                self._drop()
                raise
            except AuthenticationError as exc:
                self._drop()
                raise IntegrationError("AUTHENTICATION_FAILED", "Local storage authentication failed") from exc
            except socket.timeout as exc:
                self._drop()
                raise IntegrationError("RPC_TIMEOUT", "Local RPC deadline expired; result may be uncertain") from exc
            except (OSError, EOFError, pickle.PickleError) as exc:
                self._drop()
                raise IntegrationError("RPC_UNAVAILABLE", "Local RPC failed; result may be uncertain") from exc
            except BaseException:
                self._drop()
                raise
        if not response["ok"]:
            raise IntegrationError(response["code"], response["message"])
        return response["result"]

    # Explicit methods match the logical store, convenient for controller use.
    def register_model(self, *args: Any, **kwargs: Any) -> Any:
        return self.call("register_model", *args, **kwargs)

    def reserve_kv(self, *args: Any, **kwargs: Any) -> Any:
        return self.call("reserve_kv", *args, **kwargs)

    def bind_kv(self, *args: Any, **kwargs: Any) -> Any:
        return self.call("bind_kv", *args, **kwargs)

    def start_kv_batch(self, *args: Any, **kwargs: Any) -> Any:
        return self.call("start_kv_batch", *args, **kwargs)

    def finish_kv_batch(self, *args: Any, **kwargs: Any) -> Any:
        return self.call("finish_kv_batch", *args, **kwargs)

    def abort_kv_batch(self, *args: Any, **kwargs: Any) -> Any:
        return self.call("abort_kv_batch", *args, **kwargs)

    def release_kv(self, *args: Any, **kwargs: Any) -> Any:
        return self.call("release_kv", *args, **kwargs)

    def fence_batch(self, *args: Any, **kwargs: Any) -> Any:
        return self.call("fence_batch", *args, **kwargs)

    def retire_history(self, *args: Any, **kwargs: Any) -> Any:
        return self.call("retire_history", *args, **kwargs)

    def snapshot(self) -> dict:
        return self.call("snapshot")

    def assert_integrity(self) -> dict:
        return self.call("assert_integrity")

    def export_pool(self, *args: Any, **kwargs: Any) -> Any:
        return self.call("export_pool", *args, **kwargs)

    def release_consumer(self, *args: Any, **kwargs: Any) -> Any:
        return self.call("release_consumer", *args, **kwargs)

    def close_owner(self) -> None:
        return self.call("close_owner")


class StorageRPCServer:
    def __init__(self, store: HeadwiseStore, gpu_storage: Any, address: str,
                 authkey: bytes, timeout_s: float = 30.0,
                 extra_methods: dict[str, Callable[..., Any]] | None = None,
                 allow_owner_shutdown: bool = False,
                 owner_close: Callable[[], None] | None = None,
                 failure_event: Any = None):
        # If the keeper cannot start, this constructor thread itself must be
        # capable of holding the owner until the close barrier proves safety.
        if current_thread().daemon:
            raise IntegrationError("OWNER_THREAD", "Construct storage owner server on a non-daemon thread")
        self.gpu_storage = gpu_storage
        self.failure_status: str | None = None
        self.failure_event = Event()
        self._external_failure_event = failure_event
        self._failure_hold = Event()  # Never set while ownership is uncertain.
        self._serving = Event()
        self._failure_thread: Thread | None = None
        self._owner_close: Callable[[], None] | None = None
        try:
            # Extensions may export resources outside GPUStorage. Their composite
            # owner must supply a close barrier covering every such resource.
            self._owner_close = owner_close if owner_close is not None else (
                None if extra_methods else gpu_storage.close)
            # Establish process lifetime before the listener exists or any
            # handler can be admitted. Later Thread.start failures need no
            # fresh thread allocation to retain protected CUDA resources.
            self._failure_thread = Thread(target=self._hold_failure, daemon=False,
                                          name="storage-owner-lifetime-keeper")
            self._failure_thread.start()
            self._initialize(store, address, authkey, timeout_s,
                             extra_methods, allow_owner_shutdown)
        except BaseException:
            self._mark_failure("RPC_INIT_FAILED")
            self._retain_until_safe()
            raise

    def _initialize(self, store: HeadwiseStore, address: str, authkey: bytes,
                    timeout_s: float, extra_methods: dict[str, Callable[..., Any]] | None,
                    allow_owner_shutdown: bool) -> None:
        _validate_options(authkey, timeout_s)
        self.address = _validate_endpoint(address, create=True)
        self.store = store
        self._authkey = authkey
        self.timeout_s = timeout_s
        self._methods = {name: getattr(store, name) for name in (
            "register_model", "reserve_kv", "bind_kv", "start_kv_batch", "finish_kv_batch",
            "abort_kv_batch", "release_kv", "fence_batch", "retire_history",
            "snapshot", "assert_integrity")}
        self._methods.update({"export_pool": self.gpu_storage.export_pool,
                              "release_consumer": self.gpu_storage.release_consumer,
                              "owner_snapshot": self.gpu_storage.owner_snapshot})
        # A composite owner must also protect its weight exports. Shutdown is
        # opt-in; enable it only for KV-only owners or a combined owner facade.
        if allow_owner_shutdown:
            self._methods["close_owner"] = self._request_shutdown
        # Extensions are explicitly supplied by trusted owner code, never named
        # dynamically by an RPC client. Root may add weight-export endpoints.
        for name, method in (extra_methods or {}).items():
            if not isinstance(name, str) or name.startswith("_") or name in self._methods or not callable(method):
                raise IntegrationError("INVALID_ARGUMENT", "Invalid explicit owner RPC extension")
            self._methods[name] = method
        self._stop = Event()
        self._closing_handlers = False
        # Each authenticated rank keeps one persistent connection (24 ranks for
        # three TP4 P/D groups). Operation concurrency stays bounded separately.
        self._workers = BoundedSemaphore(256)
        self._connections: set[Connection] = set()
        self._connection_lock = Lock()
        self._handlers_done = Event()
        self._handlers_done.set()
        # Authenticate in each worker, so an incomplete handshake cannot stop
        # acceptance of other local clients. No payload is unpickled beforehand.
        self._listener = Listener(self.address, family="AF_UNIX", backlog=32, authkey=None)
        os.chmod(self.address, 0o600)

    def _mark_failure(self, status: str) -> None:
        # Fixed codes are safe for a launcher to inspect; never expose exception
        # text, request payloads, or the authentication key.
        self.failure_status = status
        self.failure_event.set()
        if self._external_failure_event is not None:
            self._external_failure_event.set()

    def _hold_failure(self) -> None:
        # Bound target retains this server and its GPU owner strongly until an
        # authoritative close succeeds, whether or not serving has begun.
        self._failure_hold.wait()

    def _retain_until_safe(self) -> None:
        try:
            if hasattr(self, "_handlers_done"):
                self._handlers_done.wait()
            if self._owner_close is None:
                raise IntegrationError("OWNER_CLOSE_UNPROVEN", "Composite owner close barrier required")
            self._owner_close()
            self._failure_hold.set()
            return
        except BaseException:
            # A busy or uncertain close is a fence, including an exception from
            # synchronize/ipc_collect. The non-daemon thread keeps a composite
            # process alive even when serve_forever ran on a daemon thread.
            self._failure_hold.wait()

    def serve_forever(self) -> None:
        self._serving.set()
        try:
            try:
                while not self._stop.is_set():
                    try:
                        conn = self._listener.accept()
                    except (OSError, EOFError):
                        if self._stop.is_set():
                            break
                        raise IntegrationError("RPC_UNAVAILABLE", "Local socket accept failed")
                    if self._stop.is_set():
                        conn.close()
                        break
                    # Short-lived polls can overlap handlers finishing cleanup.
                    # Backpressure before authentication; never discard a healthy
                    # rank's connection merely because all handler slots are busy.
                    deadline = time.monotonic() + self.timeout_s
                    if not self._workers.acquire(timeout=max(0, deadline - time.monotonic())):
                        conn.close()
                        continue
                    if self._stop.is_set():
                        conn.close()
                        self._workers.release()
                        break
                    with self._connection_lock:
                        self._connections.add(conn)
                        self._handlers_done.clear()
                    try:
                        Thread(target=self._handle, args=(conn, deadline), daemon=True).start()
                    except BaseException:
                        conn.close()
                        with self._connection_lock:
                            self._connections.discard(conn)
                            if not self._connections:
                                self._handlers_done.set()
                        self._workers.release()
                        raise
            except BaseException:
                self._mark_failure("RPC_LISTENER_FAILED")
                raise
            finally:
                self._listener.close()
                # Idle persistent handlers exit at their next wakeup; admitted
                # native workers must still finish before any close barrier.
                self._closing_handlers = True
                self._handlers_done.wait()
        except BaseException:
            self._mark_failure("RPC_LISTENER_FAILED")
            self._retain_until_safe()
        else:
            # The normal stop path already passed the owner close barrier and
            # waited for admitted handlers in the inner finally block.
            self._failure_hold.set()

    def _handle(self, conn: Connection, deadline: float | None = None) -> None:
        pending: list[Thread] = []
        timed = None
        try:
            timed = _DeadlineConnection(conn, deadline if deadline is not None else
                                        time.monotonic() + self.timeout_s, pending)
            deliver_challenge(timed, self._authkey)
            answer_challenge(timed, self._authkey)
            first = True
            while True:
                if not first:
                    # Persistent connection: wait for the next request without
                    # a deadline, but leave promptly once the owner stops.
                    while not timed.wait_readable(0.2):
                        if self._stop.is_set() or self._closing_handlers:
                            return
                    if self._stop.is_set():
                        return
                    timed.deadline = time.monotonic() + self.timeout_s
                first = False
                uncertain = False
                try:
                    request = pickle.loads(timed.recv_bytes(_MAX_MESSAGE_BYTES))
                    if (not isinstance(request, dict) or set(request) != {"method", "args", "kwargs"}
                            or not isinstance(request["method"], str)
                            or not isinstance(request["args"], tuple) or not isinstance(request["kwargs"], dict)):
                        raise IntegrationError("RPC_PROTOCOL", "Malformed storage request")
                    method = self._methods.get(request["method"])
                    if method is None:
                        raise IntegrationError("METHOD_NOT_ALLOWED", "Storage RPC method is not exported")
                    # An operation that outlives a deadline may still mutate storage;
                    # reservation/consumer references therefore remain pinned.
                    try:
                        result = _bounded(lambda: method(*request["args"], **request["kwargs"]),
                                          timed.remaining(), pending=pending)
                    except IntegrationError as exc:
                        uncertain = exc.code == "RPC_TIMEOUT"
                        raise
                    response = {"ok": True, "result": result}
                except (EOFError, OSError):
                    return  # Peer closed or transport failed; no request to answer.
                except IntegrationError as exc:
                    response = {"ok": False, "code": exc.code, "message": str(exc)}
                except (TypeError, ValueError, pickle.PickleError):
                    response = {"ok": False, "code": "INVALID_ARGUMENT", "message": "Invalid storage RPC arguments"}
                except Exception:
                    response = {"ok": False, "code": "INTERNAL_ERROR", "message": "Storage owner operation failed"}
                if not uncertain:
                    # The operation has returned; reap its worker now. Unjoined
                    # threads keep OS resources, and a persistent connection
                    # serves many calls before its handler's finally runs.
                    for worker in pending:
                        worker.join()
                    pending.clear()
                timed.send_bytes(_encode(response))
                if uncertain or self._stop.is_set():
                    # A timed-out operation may still be running: end this
                    # connection and keep the handler pinned in finally.
                    return
        except Exception:
            # Authentication failures / disconnects never log secrets and never
            # remove consumer leases or roll back uncertain running batches.
            pass
        finally:
            if timed is not None:
                timed.close()
            conn.close()
            # A closed connection cannot be treated as cancellation of native
            # work. Retain the handler admission slot and shutdown barrier.
            for worker in pending:
                worker.join()
            with self._connection_lock:
                self._connections.discard(conn)
                if not self._connections:
                    self._handlers_done.set()
            self._workers.release()

    def _request_shutdown(self) -> None:
        if self._owner_close is None:
            raise IntegrationError("OWNER_CLOSE_UNPROVEN", "Composite owner close barrier required")
        self._owner_close()
        self._stop.set()
        # Wake the blocking accept on Linux; closing its descriptor from a
        # different thread alone does not reliably interrupt accept.
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as wake:
            wake.settimeout(self.timeout_s)
            try:
                wake.connect(self.address)
            except OSError:
                pass

    def close(self) -> None:
        """Drain GPU ownership first; an unsafe close leaves the service alive."""
        self._request_shutdown()
        # Closing a constructed but never served owner must release its keeper.
        # If serving has begun, serve_forever releases it after handlers drain.
        if not self._serving.is_set():
            self._failure_hold.set()


def spawn_storage_owner(pool_blocks_by_rank: dict[int, int], rank_devices: dict[int, int],
                        address: str, authkey: bytes, timeout_s: float = 30.0):
    """Spawn a non-daemon CUDA owner; caller must keep it alive and drain it.

    The process is intentionally not automatically terminated on timeout. Its
    live/fenced allocations and CUDA consumers may still depend on its buffers.
    """
    from multiprocessing import get_context
    _validate_options(authkey, timeout_s)
    _validate_endpoint(address, create=True)
    context = get_context("spawn")
    failure_event = context.Event()
    process = context.Process(target=_owner_main,
        args=(dict(pool_blocks_by_rank), dict(rank_devices), address, authkey, timeout_s, failure_event),
        daemon=False, name="muxserve-headwise-owner")
    process.storage_failure_event = failure_event
    process.start()
    return process


def _owner_main(pools: dict[int, int], devices: dict[int, int], address: str,
                authkey: bytes, timeout_s: float, failure_event: Any = None) -> None:
    from .gpu_storage import GPUStorage
    store = HeadwiseStore(pools)
    gpu = GPUStorage(store, devices)
    server = StorageRPCServer(store, gpu, address, authkey, timeout_s,
                              allow_owner_shutdown=True, failure_event=failure_event)
    server.serve_forever()
