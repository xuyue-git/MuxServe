"""gateway_process: the HTTP gateway in its own child process.

The owner keeps GatewayState unchanged: request records, confirmed events, lifecycle
(maintain, forget, cancel, resume) and the HTTP counters. The child runs the
unchanged GatewayServer/GatewayHandler (sockets, admission, SSE framing) plus the
tokenizers. One duplex pipe joins them:

  child -> owner  ("submit"|"control"|"get", call_id, ...)   answered by ("reply", call_id, ok, value)
                  ("count", name, amount)  ("disconnect", internal_id)
  owner -> child  ("updates", [(internal_id, start, new_tokens, state, finish_reason, error), ...])
                  ("drain",)  ("close",)

The owner-side cost per confirmed token is one entry in a dirty map and a share of
one pickled message; HTTP threads, JSON and detokenization no longer hold the owner
interpreter lock that every rank report needs.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import itertools
import logging
from threading import Condition, Event, Lock, RLock, Thread
from types import SimpleNamespace

from .contracts import IntegrationError
from .gateway import _TERMINAL, GatewayServer, GatewayState, OutputRecord

logger = logging.getLogger(__name__)
_READY_TIMEOUT_S = 300.0


class BridgedGatewayState(GatewayState):
    """Owner side: every wake of a record marks it for the child (callers hold the lock)."""

    def __init__(self, controller, tokenizers):
        super().__init__(controller, tokenizers)
        self.dirty = {}
        self.dirty_wake = Event()

    def _wake(self, record):
        super()._wake(record)
        self.dirty[record.internal_id] = record
        self.dirty_wake.set()

    def _wake_all(self):
        super()._wake_all()
        for record in self.records.values():
            self.dirty[record.internal_id] = record
        self.dirty_wake.set()


class GatewayBridge:
    """Owner side of the pipe; replaces GatewayServer in the launcher.
    serve_forever (the launcher's http thread) returns once the child has closed."""

    def __init__(self, state: BridgedGatewayState, context, port: int, maximum_connections: int,
                 tokenizer_specs: dict, settings: dict):
        self.state = state
        self.send_lock = Lock()
        self.sent = {}
        self.closing = False
        parent, child = context.Pipe(duplex=True)
        self.conn = parent
        self.process = context.Process(target=gateway_child, name="mxs-gateway", daemon=True,
                                       args=(child, port, maximum_connections, tokenizer_specs, settings))
        self.process.start()
        child.close()
        if not self.conn.poll(_READY_TIMEOUT_S):
            self.process.terminate()
            raise IntegrationError("GATEWAY_STARTUP", "Gateway process did not report ready")
        message = self.conn.recv()
        if message[0] != "ready":
            self.process.join(5)
            raise IntegrationError("GATEWAY_STARTUP", "Gateway process failed: %s" % (message[1:],))
        self.pool = ThreadPoolExecutor(max_workers=8, thread_name_prefix="gateway-bridge")
        self.sender = Thread(target=self._send_loop, name="gateway-bridge-send", daemon=True)
        self.sender.start()

    def _send(self, message):
        with self.send_lock:
            self.conn.send(message)

    def _send_loop(self):
        state = self.state
        while not self.closing:
            state.dirty_wake.wait(0.05)
            state.dirty_wake.clear()  # a wake after this keeps the next wait short
            with state.condition:
                if not state.dirty:
                    continue
                records = tuple(state.dirty.values())
                state.dirty.clear()
                updates = []
                for record in records:
                    start = self.sent.get(record.internal_id, 0)
                    updates.append((record.internal_id, start, record.tokens[start:], record.state,
                                    record.finish_reason, record.error))
                    self.sent[record.internal_id] = len(record.tokens)
                if len(self.sent) > 2 * len(state.records) + 1024:
                    self.sent = {key: value for key, value in self.sent.items() if key in state.records}
            try:
                self._send(("updates", updates))
            except (OSError, EOFError, ValueError):
                if not self.closing:
                    logger.exception("Gateway process pipe closed")
                return

    def _reply(self, call_id, function, *args):
        try:
            value = (True, function(*args))
        except IntegrationError as exc:
            value = (False, (exc.code, str(exc)))
        except (ValueError, TypeError, KeyError, UnicodeError) as exc:
            value = (False, ("INVALID_ARGUMENT", str(exc)))
        except Exception as exc:  # the in-process gateway would drop the connection
            logger.exception("Gateway bridge call failed")
            value = (False, ("INTERNAL", repr(exc)))
        try:
            self._send(("reply", call_id) + value)
        except (OSError, EOFError, ValueError):
            pass

    def _submit(self, model_id, body):
        record = self.state.submit(model_id, body)
        return record.internal_id, record.rid, record.prompt_tokens, record.stream_interval

    def _get(self, what):
        if what == "http_metrics":
            return self.state.http_metrics()
        if what == "metrics":
            return self.state.controller.metrics()
        if what == "groups":
            groups = self.state.controller.snapshot()["groups"]
            return {"groups": {model: {key: group[key] for key in ("ready", "group_id", "fenced", "draining")}
                               for model, group in groups.items()}}
        raise IntegrationError("INVALID_ARGUMENT", "Unknown gateway query")

    def _disconnect(self, internal_id):
        record = self.state.records.get(internal_id)
        if record is not None:
            try:
                self.state.disconnect(record)
            except IntegrationError:
                logger.exception("Disconnect cancellation remains pending")

    def serve_forever(self):
        while True:
            try:
                message = self.conn.recv()
            except (EOFError, OSError):
                break
            kind = message[0]
            try:
                if kind == "count":
                    self.state.count_http(message[1], message[2])
                elif kind == "disconnect":
                    self.pool.submit(self._disconnect, message[1])
                elif kind == "submit":
                    self.pool.submit(self._reply, message[1], self._submit, message[2], message[3])
                elif kind == "control":
                    self.pool.submit(self._reply, message[1], self.state.control, *message[2:5])
                elif kind == "get":
                    self.pool.submit(self._reply, message[1], self._get, message[2])
                elif kind == "closed":
                    break
            except RuntimeError:  # the pool already shut down (server_close)
                break
        if not self.closing:
            logger.error("Gateway process ended (exit code %s)", self.process.exitcode)

    def begin_drain(self):
        self.state.begin_drain()
        try:
            self._send(("drain",))
        except (OSError, EOFError, ValueError):
            pass

    def server_close(self):
        self.closing = True
        try:
            self._send(("close",))
        except (OSError, EOFError, ValueError):
            pass
        self.process.join(30)
        if self.process.is_alive():
            self.process.terminate()
            self.process.join(10)
        self.conn.close()
        self.pool.shutdown(wait=True)


class _ControllerView:
    def __init__(self, state):
        self.state = state

    def metrics(self):
        return self.state.call("get", "metrics")

    def snapshot(self):
        return self.state.call("get", "groups")


class RemoteGatewayState:
    """Child side: what GatewayServer/GatewayHandler use from GatewayState."""

    def __init__(self, conn, tokenizers, settings):
        self.conn = conn
        self.send_lock = Lock()
        self.tokenizers = tokenizers
        self.tokenizer_lock = RLock()
        self._lock = RLock()
        self.condition = Condition(self._lock)
        self.config = SimpleNamespace(**settings)
        self.controller = _ControllerView(self)
        self.accepting = True
        self.records = {}
        self.early = {}
        self.closed = set()
        self.calls = {}
        self.ids = itertools.count(1)

    def send(self, message):
        with self.send_lock:
            self.conn.send(message)

    def call(self, kind, *args, timeout=120.0):
        call_id = next(self.ids)
        slot = [Event(), None]
        self.calls[call_id] = slot
        try:
            self.send((kind, call_id) + args)
        except (OSError, EOFError, ValueError) as exc:
            self.calls.pop(call_id, None)
            raise IntegrationError("NOT_READY", "Owner connection closed") from exc
        if not slot[0].wait(timeout):
            self.calls.pop(call_id, None)
            raise IntegrationError("NOT_READY", "Owner did not answer the gateway")
        ok, value = slot[1]
        if ok:
            return value
        raise IntegrationError(value[0], value[1])

    def resolve(self, call_id, ok, value):
        slot = self.calls.pop(call_id, None)
        if slot is not None:
            slot[1] = (ok, value)
            slot[0].set()

    def count_http(self, name, amount=1):
        try:
            self.send(("count", name, amount))
        except (OSError, EOFError, ValueError):
            pass

    def http_metrics(self):
        return self.call("get", "http_metrics")

    def submit(self, model_id, body):
        if (isinstance(body, dict) and "input_ids" not in body and isinstance(body.get("text"), str)
                and body["text"] and model_id in self.tokenizers):
            # Tokenize here; the owner validates everything else exactly as before.
            body = dict(body)
            with self.tokenizer_lock:
                body["input_ids"] = list(self.tokenizers[model_id].encode(body.pop("text"),
                                                                          add_special_tokens=True))
        internal_id, rid, prompt_tokens, interval = self.call("submit", model_id, body)
        record = OutputRecord(model_id, rid, internal_id, prompt_tokens, interval, wake=Condition(self._lock))
        with self.condition:
            self.records[internal_id] = record
            for update in self.early.pop(internal_id, ()):
                self._apply(record, update)
        return record

    @staticmethod
    def _apply(record, update):
        _, start, tokens, state, finish_reason, error = update
        have = len(record.tokens)
        if start <= have:
            record.tokens.extend(tokens[have - start:])
        record.state, record.finish_reason = state, finish_reason
        if error:
            record.error = error
        record.wake.notify_all()

    def apply_updates(self, updates):
        with self.condition:
            for update in updates:
                record = self.records.get(update[0])
                if record is not None:
                    self._apply(record, update)
                elif update[0] not in self.closed:
                    self.early.setdefault(update[0], []).append(update)

    def control(self, model_id, rid, operation):
        return self.call("control", model_id, rid, operation)

    def disconnect(self, record):
        with self.condition:
            self.records.pop(record.internal_id, None)
            self.early.pop(record.internal_id, None)
            self.closed.add(record.internal_id)
        try:
            self.send(("disconnect", record.internal_id))
        except (OSError, EOFError, ValueError):
            pass

    def _end_all(self, reason):
        with self.condition:
            self.accepting = False
            for record in self.records.values():
                if record.state not in _TERMINAL:
                    record.error = record.error or reason
                record.wake.notify_all()
            self.condition.notify_all()

    def begin_drain(self):
        self._end_all("supervisor_shutdown")

    def payload(self, record, *, terminal=False):
        return GatewayState.payload(self, record, terminal=terminal)


def _tokenizer(spec):
    if isinstance(spec, str):
        from transformers import AutoTokenizer
        return AutoTokenizer.from_pretrained(spec, local_files_only=True, trust_remote_code=False)
    return spec


def gateway_child(conn, port, maximum_connections, tokenizer_specs, settings):
    try:
        tokenizers = {model: _tokenizer(spec) for model, spec in tokenizer_specs.items()}
        state = RemoteGatewayState(conn, tokenizers, settings)
        server = GatewayServer(state, port, maximum_connections=maximum_connections)
    except BaseException as exc:
        conn.send(("failed", repr(exc)))
        return
    http = Thread(target=server.serve_forever, name="http-gateway", daemon=False)
    http.start()
    conn.send(("ready",))
    draining = False
    try:
        while True:
            try:
                message = conn.recv()
            except (EOFError, OSError):
                break
            kind = message[0]
            if kind == "updates":
                state.apply_updates(message[1])
            elif kind == "reply":
                state.resolve(message[1], message[2], message[3])
            elif kind == "drain" and not draining:
                draining = True
                server.begin_drain()
            elif kind == "close":
                break
    finally:
        if not draining:
            server.begin_drain()
        state._end_all("owner_closed")
        http.join(30)
        server.server_close()
        try:
            conn.send(("closed",))
        except (OSError, EOFError, ValueError):
            pass
        conn.close()
