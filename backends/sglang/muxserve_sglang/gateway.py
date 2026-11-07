"""Bounded loopback native-SGLang SSE gateway backed by confirmed owner events."""
from __future__ import annotations

from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import logging
import socket
from threading import BoundedSemaphore, Condition, RLock, Timer
import time
from urllib.parse import unquote, urlsplit
from uuid import uuid4

from . import trace
from .contracts import IntegrationError, RequestSpec

logger = logging.getLogger(__name__)
_TERMINAL = frozenset(("FINISHED", "CANCELLED", "TIMED_OUT", "FAILED"))
_BODY_LIMIT = 1024 * 1024
_HTTP_COUNTERS = ("generate_received", "generate_admitted", "generate_rejected", "body_read_failed",
    "connection_admission_rejections", "confirmed_output_tokens", "terminal_normal", "terminal_other",
    "terminal_timeout", "client_disconnects")


@dataclass
class OutputRecord:
    model_id: str
    rid: str
    internal_id: str
    prompt_tokens: int
    stream_interval: int
    tokens: list[int] = field(default_factory=list)
    state: str = "QUEUED"
    attempt: int = 0
    finish_reason: str | None = None
    resume_record_id: str | None = None
    manual_suspend: bool = False
    attached: bool = True
    forgotten: bool = False
    error: str | None = None
    completed_at: float | None = None
    deadline: float = 0.0
    terminal_counted: bool = False


class GatewayState:
    def __init__(self, controller, tokenizers):
        self.controller, self.config = controller, controller.config
        self.tokenizers = tokenizers
        self.condition = Condition(RLock())
        self.tokenizer_lock = RLock()
        self.records = {}
        # Exact service-epoch RID rejection avoids returning an old controller
        # replay to a new HTTP submission. Bound the tombstones like controller.
        self.rids = set()
        self.aliases = {}
        self.accepting = True
        self.http_epoch = uuid4().hex
        self.http_counters = dict.fromkeys(_HTTP_COUNTERS, 0)

    def count_http(self, name, amount=1):
        with self.condition:
            self.http_counters[name] += amount

    def http_metrics(self):
        with self.condition:
            return {"epoch": self.http_epoch, "counters": dict(self.http_counters),
                    "receipt_boundary": "recognized generate POST headers parsed, before body/admission",
                    "terminal_boundary": "confirmed controller events, independent of client socket writes"}

    def submit(self, model_id, body):
        self.config.model(model_id)
        if not isinstance(body, dict) or set(body) - {"rid", "text", "input_ids", "stream", "sampling_params"}:
            raise IntegrationError("INVALID_ARGUMENT", "Unsupported generate fields")
        if body.get("stream") is not True or ("text" in body) == ("input_ids" in body):
            raise IntegrationError("INVALID_ARGUMENT", "Use stream=true and exactly one prompt representation")
        rid = body.get("rid", uuid4().hex)
        if not isinstance(rid, str) or not 0 < len(rid) <= 128:
            raise IntegrationError("INVALID_ARGUMENT", "RID must contain 1..128 characters")
        params = body.get("sampling_params", {})
        allowed = {"temperature", "top_k", "top_p", "max_new_tokens", "ignore_eos", "stream_interval"}
        if not isinstance(params, dict) or set(params) - allowed:
            raise IntegrationError("INVALID_ARGUMENT", "Unsupported sampling parameter")
        temperature, top_k, top_p = (params.get("temperature", 0), params.get("top_k", 1),
                                     params.get("top_p", 1))
        if (type(temperature) not in (int, float) or temperature != 0
                or type(top_k) is not int or top_k != 1
                or type(top_p) not in (int, float) or top_p != 1
                or type(params.get("ignore_eos", False)) is not bool):
            raise IntegrationError("INVALID_ARGUMENT", "Initial gateway supports greedy sampling only")
        maximum = params.get("max_new_tokens", self.config.max_new_tokens)
        interval = params.get("stream_interval", 1)
        if (type(maximum) is not int or not 1 <= maximum <= self.config.max_new_tokens
                or type(interval) is not int or not 1 <= interval <= self.config.max_new_tokens):
            raise IntegrationError("INVALID_ARGUMENT", "Invalid output budget or stream interval")
        if "text" in body:
            if not isinstance(body["text"], str) or not body["text"]:
                raise IntegrationError("INVALID_ARGUMENT", "Text prompt must be nonempty")
            with self.tokenizer_lock:
                inputs = self.tokenizers[model_id].encode(body["text"], add_special_tokens=True)
        else:
            inputs = body["input_ids"]
            if not isinstance(inputs, list) or any(type(token) is not int for token in inputs):
                raise IntegrationError("INVALID_ARGUMENT", "input_ids must be an integer list")
        if not 0 < len(inputs) <= self.config.max_input_tokens:
            raise IntegrationError("INVALID_ARGUMENT", "Prompt exceeds the fixed input limit")
        identity = uuid4().hex
        record = OutputRecord(model_id, rid, identity, len(inputs), interval)
        with self.condition:
            if not self.accepting or len(self.records) >= self.config.queue_limit:
                raise IntegrationError("NOT_READY", "Gateway is draining or its output window is full")
            key = (model_id, rid)
            if key in self.rids:
                raise IntegrationError("IDENTITY_CONFLICT", "RID was already submitted in this service epoch")
            if len(self.rids) >= 100000:
                raise IntegrationError("HISTORY_FULL", "RID history requires a safe service restart")
            self.records[identity] = record
            try:
                output = self.controller.submit_request(RequestSpec(
                    model_id, identity, tuple(inputs), maximum, self.config.request_timeout_ms,
                    {"ignore_eos": params.get("ignore_eos", False)}))
                record.deadline = output["deadline"]
            except Exception:
                del self.records[identity]
                raise
            self.rids.add(key)
            self.aliases[key] = identity
            self.http_counters["generate_admitted"] += 1
        return record

    def apply_event(self, event):
        with self.condition:
            if event.kind == "GROUP_ERROR":
                for record in self.records.values():
                    if record.model_id == event.model_id and record.state not in _TERMINAL:
                        record.error = event.payload["reason"]
                self.condition.notify_all()
                return
            record = self.records.get(event.request_id)
            if record is None:
                return
            record.attempt = event.attempt
            if event.kind == "TOKEN":
                if event.payload["output_index"] != len(record.tokens):
                    raise IntegrationError("EVENT_CONFLICT", "Confirmed token index changed")
                if not record.tokens:
                    trace.ev("gw_first_token", None, m=record.model_id, req=record.internal_id)
                record.tokens.extend(event.payload["token_ids"])
                self.http_counters["confirmed_output_tokens"] += len(event.payload["token_ids"])
                record.state = "DECODING"
            elif event.kind in ("TERMINAL", "SUSPENDED"):
                if tuple(record.tokens) != tuple(event.payload["output_token_ids"]):
                    raise IntegrationError("EVENT_CONFLICT", "Terminal cumulative history differs")
                record.resume_record_id = event.payload["resume_record_id"]
                record.finish_reason = event.payload["finish_reason"]
                record.state = "SUSPENDED" if event.kind == "SUSPENDED" else (
                    "FINISHED" if record.finish_reason in ("eos", "length") else "FAILED")
                if event.kind == "TERMINAL":
                    if not record.terminal_counted:
                        self.http_counters["terminal_normal" if record.state == "FINISHED" else "terminal_other"] += 1
                        if record.finish_reason == "timeout" or record.error == "timeout":
                            self.http_counters["terminal_timeout"] += 1
                        record.terminal_counted = True
                    record.completed_at = time.monotonic()
                    if record.state != "FINISHED":
                        record.error = record.finish_reason or "request_failed"
            self.condition.notify_all()

    def maintain(self):
        # Event ACK and lifecycle work never wait for a client's socket writes.
        with self.condition:
            records = tuple(self.records.values())
        for record in records:
            with self.condition:
                if record.state not in _TERMINAL and self.controller.clock() >= record.deadline:
                    self._abandon(record, "timeout")
                if record.error and not record.forgotten:
                    self._cancel_abandoned(record)
                if (record.state == "SUSPENDED" and not record.error
                        and not record.manual_suspend and self.accepting):
                    try:
                        output = self.controller.resume_request(record.model_id, record.internal_id,
                                                                record.resume_record_id)
                    except IntegrationError as exc:
                        if exc.code not in ("NOT_READY", "INVALID_STATE"):
                            raise
                    else:
                        record.state, record.attempt = output["state"], output["attempt"]
                        self.condition.notify_all()
            if record.state in _TERMINAL and not record.forgotten:
                try:
                    self.controller.forget_request(record.model_id, record.internal_id, record.attempt)
                except IntegrationError as exc:
                    if exc.code != "NOT_READY":
                        raise
                else:
                    record.forgotten = True
            if record.forgotten and not record.attached:
                with self.condition:
                    self.records.pop(record.internal_id, None)
                    self.aliases.pop((record.model_id, record.rid), None)

    def control(self, model_id, rid, operation):
        with self.condition:
            identity = self.aliases.get((model_id, rid))
            record = self.records.get(identity)
            if record is None:
                raise IntegrationError("UNKNOWN_REQUEST", "Unknown active RID")
            if record.forgotten:
                raise IntegrationError("INVALID_STATE", "Request retention has ended")
            if operation == "cancel":
                if record.state in _TERMINAL:
                    return self.controller.request_output(model_id, identity)
                self._abandon(record, "cancelled")
                return self._cancel_abandoned(record)
            if operation == "suspend":
                record.manual_suspend = True
                return self.controller.suspend_request(model_id, identity, record.attempt)
            if operation == "resume":
                if self.controller.clock() >= record.deadline:
                    self._abandon(record, "timeout")
                if record.error or not self.accepting:
                    raise IntegrationError("INVALID_STATE", "HTTP continuation has ended")
                result = self.controller.resume_request(model_id, identity, record.resume_record_id)
                record.manual_suspend = False
                record.state, record.attempt = result["state"], result["attempt"]
                self.condition.notify_all()
                return result
            raise IntegrationError("INVALID_ARGUMENT", "Unsupported request operation")

    def disconnect(self, record):
        with self.condition:
            record.attached = False
            if record.state not in _TERMINAL and not record.forgotten:
                self.http_counters["client_disconnects"] += 1
                self._abandon(record, "http_disconnect")
                self._cancel_abandoned(record)

    def _abandon(self, record, reason):
        """End HTTP continuation without asserting GPU completion or release."""
        record.manual_suspend = True
        record.error = record.error or reason
        self.condition.notify_all()

    def _cancel_abandoned(self, record):
        if record.error == "timeout":
            self.controller.tick()
        output = self.controller.request_output(record.model_id, record.internal_id)
        # Pending cleanup owns its barrier. A completed suspension still needs
        # cancellation to retire its checkpoint through another real barrier.
        if output["state"] in _TERMINAL or output["pending_reason"]:
            return output
        try:
            return self.controller.cancel_request(record.model_id, record.internal_id, record.error)
        except IntegrationError as exc:
            if exc.code == "IDENTITY_CONFLICT":
                current = self.controller.request_output(record.model_id, record.internal_id)
                if current["pending_reason"] or current["state"] in _TERMINAL:
                    return current
            if exc.code not in ("NOT_READY", "COMMAND_BACKPRESSURE", "EVENT_BACKPRESSURE"):
                raise
            return output

    def begin_drain(self):
        with self.condition:
            self.accepting = False
            for record in self.records.values():
                if record.state not in _TERMINAL:
                    self._abandon(record, "supervisor_shutdown")
            self.condition.notify_all()

    def fail(self, reason):
        with self.condition:
            self.accepting = False
            for record in self.records.values():
                if record.state not in _TERMINAL:
                    record.error = reason
            self.condition.notify_all()

    def payload(self, record, *, terminal=False):
        with self.condition:
            tokens = tuple(record.tokens)
            reason = record.finish_reason
        with self.tokenizer_lock:
            text = self.tokenizers[record.model_id].decode(tokens, skip_special_tokens=True)
        meta = {"id": record.rid, "prompt_tokens": record.prompt_tokens,
                "completion_tokens": len(tokens), "cached_tokens": 0,
                "finish_reason": {"type": "stop" if reason == "eos" else "length"} if terminal else None}
        return {"text": text, "output_ids": tokens, "meta_info": meta}


class GatewayServer(ThreadingHTTPServer):
    daemon_threads = False
    block_on_close = True
    allow_reuse_address = False
    admission_timeout = 10.0
    write_timeout = 10.0

    def __init__(self, state, port, maximum_connections=32):
        self.state = state
        self.maximum_connections = min(maximum_connections, state.config.queue_limit)
        self.slots = BoundedSemaphore(self.maximum_connections)
        self.admission_lock = RLock()
        self.admissions = {}
        super().__init__(("127.0.0.1", port), GatewayHandler)

    def get_request(self):
        request, address = super().get_request()
        # Installed before StreamRequestHandler.setup or any header read.
        request.settimeout(self.admission_timeout)
        return request, address

    def _expire_admission(self, request):
        with self.admission_lock:
            if self.admissions.pop(request, None) is not None:
                try:
                    request.shutdown(socket.SHUT_RDWR)
                except OSError:
                    pass

    def finish_admission(self, request):
        with self.admission_lock:
            timer = self.admissions.pop(request, None)
            if timer is None:
                raise TimeoutError("HTTP admission deadline expired")
            timer.cancel()
            request.settimeout(self.write_timeout)

    def begin_drain(self):
        self.state.begin_drain()
        self.shutdown()
        # Interrupt parsers immediately. Attached streams were notified above;
        # bounded socket writes allow their error frames to finish independently.
        with self.admission_lock:
            for request in tuple(self.admissions):
                timer = self.admissions[request]
                self._expire_admission(request)
                timer.cancel()

    def process_request(self, request, client_address):
        if not self.state.accepting or not self.slots.acquire(blocking=False):
            self.state.count_http("connection_admission_rejections")
            try:
                request.settimeout(1)
                request.sendall(b"HTTP/1.1 503 Service Unavailable\r\nContent-Length: 0\r\nConnection: close\r\n\r\n")
            finally:
                self.shutdown_request(request)
            return
        try:
            timer = Timer(self.admission_timeout, self._expire_admission, (request,))
            timer.daemon = True
            with self.admission_lock:
                self.admissions[request] = timer
                timer.start()
            super().process_request(request, client_address)
        except BaseException:
            with self.admission_lock:
                timer = self.admissions.pop(request, None)
                if timer:
                    timer.cancel()
            self.slots.release()
            raise

    def process_request_thread(self, request, client_address):
        try:
            super().process_request_thread(request, client_address)
        finally:
            with self.admission_lock:
                timer = self.admissions.pop(request, None)
                if timer:
                    timer.cancel()
            self.slots.release()


class GatewayHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, format, *args):
        logger.info("gateway %s", format % args)

    def _json(self, status, body):
        data = json.dumps(body, separators=(",", ":")).encode()
        self.close_connection = True
        try:
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Connection", "close")
            self.end_headers()
            self.wfile.write(data)
        except (ConnectionError, TimeoutError):
            # Admission expiry or a disconnected client can close the socket
            # before the rejection response is written.
            pass

    def _body(self):
        if self.headers.get("Transfer-Encoding") is not None:
            raise IntegrationError("INVALID_ARGUMENT", "Use a bounded Content-Length JSON body")
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError as exc:
            raise IntegrationError("INVALID_ARGUMENT", "Invalid Content-Length") from exc
        if not 0 < length <= _BODY_LIMIT:
            raise IntegrationError("INVALID_ARGUMENT", "JSON body must contain 1..1048576 bytes")
        data = self.rfile.read(length)
        if len(data) != length:
            raise IntegrationError("INVALID_ARGUMENT", "Incomplete JSON body")
        def reject_constant(value):
            raise ValueError("Non-finite JSON number")
        return json.loads(data, parse_constant=reject_constant)

    def do_GET(self):
        self.server.finish_admission(self.connection)
        path = urlsplit(self.path).path
        if path == "/metrics":
            # Bounded controller telemetry (cumulative per-stage timing totals).
            self._json(200, self.server.state.controller.metrics())
            return
        if path not in ("/health", "/status"):
            self._json(404, {"error": "Unknown endpoint"})
            return
        groups = self.server.state.controller.snapshot()["groups"]
        ready = self.server.state.accepting and all(group["ready"] for group in groups.values())
        self._json(200 if ready or path == "/status" else 503,
                   {"ready": ready, "config_fingerprint": self.server.state.config.fingerprint,
                    "http_metrics": self.server.state.http_metrics(),
                    "http_connection_limit": self.server.maximum_connections,
                    "groups": {model: {"ready": group["ready"], "group_id": group["group_id"],
                       "fenced": group["fenced"], "draining": group["draining"]}
                       for model, group in groups.items()}})

    def do_POST(self):
        parts = [unquote(part) for part in urlsplit(self.path).path.strip("/").split("/")]
        is_generate = len(parts) == 3 and parts[0] == "models" and parts[2] == "generate"
        if is_generate:
            self.server.state.count_http("generate_received")
            received = time.monotonic()
        try:
            if is_generate:
                body = self._body()
                self.server.finish_admission(self.connection)
                record = self.server.state.submit(parts[1], body)
                trace.ev("gw_admit", None, m=parts[1], req=record.internal_id, recv=received)
            elif len(parts) == 5 and parts[0] == "models" and parts[2] == "requests":
                self.server.finish_admission(self.connection)
                self._json(200, self.server.state.control(parts[1], parts[3], parts[4]))
                return
            else:
                self.server.finish_admission(self.connection)
                self._json(404, {"error": "Unknown endpoint"})
                return
        except IntegrationError as exc:
            if is_generate:
                self.server.state.count_http("generate_rejected")
            self._json(503 if exc.code in ("NOT_READY", "QUEUE_FULL", "HISTORY_FULL") else 400,
                       {"error": {"code": exc.code, "message": str(exc)}})
            return
        except (ValueError, UnicodeError):
            if is_generate:
                self.server.state.count_http("generate_rejected")
            self._json(400, {"error": "Invalid JSON body"})
            return
        except (TimeoutError, ConnectionError, OSError):
            if is_generate:
                self.server.state.count_http("body_read_failed")
            raise
        self._stream(record)

    def _stream(self, record):
        state = self.server.state
        try:
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "close")
            self.end_headers()
            sent = 0
            last_write = time.monotonic()
            while True:
                with state.condition:
                    state.condition.wait_for(lambda: record.error or record.state in _TERMINAL
                        or len(record.tokens) - sent >= record.stream_interval, timeout=1)
                    error, terminal = record.error, record.state in _TERMINAL
                    count = len(record.tokens)
                if error:
                    self._sse({"error": {"code": "REQUEST_FAILED", "message": error}})
                    return
                if terminal or count - sent >= record.stream_interval:
                    payload = state.payload(record, terminal=terminal)
                    self._sse(payload)
                    if sent == 0:
                        trace.ev("gw_first_sse", None, m=record.model_id, req=record.internal_id)
                    sent = len(payload["output_ids"])
                    last_write = time.monotonic()
                elif time.monotonic() - last_write >= 5:
                    self.wfile.write(b": keepalive\n\n")
                    self.wfile.flush()
                    last_write = time.monotonic()
                if terminal:
                    self.wfile.write(b"data: [DONE]\n\n")
                    self.wfile.flush()
                    return
        except (BrokenPipeError, ConnectionError, TimeoutError, OSError):
            pass
        finally:
            self.close_connection = True
            try:
                state.disconnect(record)
            except IntegrationError:
                logger.exception("Disconnect cancellation remains pending")

    def _sse(self, payload):
        self.wfile.write(b"data: " + json.dumps(payload, separators=(",", ":")).encode() + b"\n\n")
        self.wfile.flush()
