"""HTTP request entry for MuxScheduler, so an xbench client can drive MuxServe.

MuxScheduler normally replays its workload file inside the process
(``submit_workload`` moves each request into ``waiting`` at its arrival time).
``HttpMuxScheduler`` changes only where requests come from: they arrive over
HTTP and are appended to the same ``waiting`` queue under the same lock.
Placement, ADBS, FlexStore and the runtimes are unchanged.

    python -m muxserve.launch <model-config.yaml> --http-port 18080

Models are named as xbench names them: ``<organisation>/<model>``, the last two
parts of the checkpoint path in the model config (``.../huggyllama/llama-7b``
is ``huggyllama/llama-7b``).

OpenAI Completions (what xbench sends since CrossPool 9093241):
  ``GET /v1/models``        {"data": [{"id", "max_model_len"}, ...]}
  ``POST /v1/completions``  {"model", "prompt": [token ids], "max_tokens",
                             "temperature": 0, "ignore_eos": true, "stream": true}
  response  text/event-stream; one ``data: {...}`` chunk per new token with
            one choice and the cumulative ``usage`` (prompt_tokens,
            completion_tokens), the last with finish_reason "length"; then a
            usage-only chunk (no choices) and ``data: [DONE]``.

Native SGLang generate (the earlier xbench ``api = "sglang"`` client):
  ``POST /models/<model>/generate``
  request   {"rid", "input_ids": [...], "stream": true,
             "sampling_params": {"max_new_tokens", "temperature": 0,
                                 "ignore_eos": true, "stream_interval"}}
  response  text/event-stream; one ``data: {...}`` event every
            ``stream_interval`` tokens with cumulative ``output_ids`` and
            ``meta_info`` {id, prompt_tokens, completion_tokens, cached_tokens,
            finish_reason}; the last event has finish_reason "length", then
            ``data: [DONE]``.

MuxServe decodes greedily and always generates exactly the requested number of
tokens, so requests asking for sampling or for EOS stopping are refused rather
than served differently from what was asked. Text is neither tokenized nor
decoded (prompts are token ids, ``text`` is empty); xbench counts tokens from
the usage fields.

The workload file in the model config still provides the ADBS inputs it
provides for a replay (``info``: per-model rates and lengths); its requests are
not replayed. Stop with SIGTERM or SIGINT to the scheduler process: it stops
accepting requests, finishes the admitted ones and exits through its normal
clean-up. As in a replay, the scheduler also exits after 240 s without
scheduling anything.
"""
import asyncio
import json
import posixpath
import signal
import time

import numpy as np
from aiohttp import web

from muxserve.logger import get_logger
from muxserve.muxsched.scheduler import MuxScheduler
from muxserve.muxsched.workload_utils import Request, Workload

logger = get_logger()

# Request id kept in MuxScheduler.counter while the entry is open, so that
# finish_requests does not end the run when no request is in flight.
_OPEN = -1
# LLaMA context length (vLLM max_model_len for these checkpoints).
MAX_CONTEXT_TOKENS = 2048

# Request index -> event set whenever that request's output_tokens change.
_WAKES = {}
_request_setattr = Request.__setattr__


class _TokenList(list):
    """output_tokens that wakes its stream on append; pickles as a plain list."""
    __slots__ = ("_wake", )

    def append(self, token):
        super().append(token)
        self._wake.set()

    def __reduce_ex__(self, protocol):
        return (list, (list(self), ))


def _notifying_setattr(request, name, value):
    wake = _WAKES.get(request.idx) if name == "output_tokens" else None
    if wake is not None and isinstance(value, list) and not isinstance(
            value, _TokenList):
        value = _TokenList(value)
        value._wake = wake
    _request_setattr(request, name, value)
    if wake is not None:
        wake.set()


def _normalize(name):
    return "".join(ch for ch in name.lower() if ch.isalnum())


def _served_id(path):
    """xbench's Model ID for a checkpoint: the last two parts of its path."""
    parts = posixpath.normpath(path.replace("\\", "/")).split("/")
    return "/".join(parts[-2:])


def _sse(payload):
    return b"data: " + json.dumps(payload,
                                  separators=(",", ":")).encode() + b"\n\n"


_DONE = b"data: [DONE]\n\n"


class _BadRequest(ValueError):
    pass


def _check_request(input_ids, max_new_tokens, temperature, ignore_eos, stream):
    if stream is not True:
        raise _BadRequest("Only streaming requests are served")
    if (not isinstance(input_ids, list) or not input_ids or any(
            type(token) is not int or token < 0 for token in input_ids)):
        raise _BadRequest("The prompt must be a non-empty list of token ids")
    if type(max_new_tokens) is not int or max_new_tokens < 1:
        raise _BadRequest("The output length must be a positive integer")
    if len(input_ids) + max_new_tokens > MAX_CONTEXT_TOKENS:
        raise _BadRequest(f"Input plus output exceeds "
                          f"{MAX_CONTEXT_TOKENS} tokens")
    if temperature != 0 or ignore_eos is not True:
        raise _BadRequest("MuxServe serves greedy requests that ignore "
                          "EOS (temperature 0, ignore_eos true)")


class HttpMuxScheduler(MuxScheduler):

    def __init__(self, muxserve_config, http_port, http_host="127.0.0.1"):
        super().__init__(muxserve_config)
        self.http_host = http_host
        self.http_port = http_port
        self.accepting = True
        # Only this process's Request objects notify; pickled copies sent to
        # the runtimes carry plain lists.
        Request.__setattr__ = _notifying_setattr

    def get_workload(self) -> Workload:
        workload_file = self.muxserve_config.workload_config["workload_file"]
        infos = Workload.from_workload_file(workload_file).workload_infos
        self.counter = {_OPEN}
        return Workload([], [], infos)

    @property
    def _served_ids(self):
        """xbench Model ID -> MuxServe model name."""
        return {
            _served_id(self._name_to_model[name]): name
            for name in self._served_models
        }

    async def submit_workload(self, workload):
        app = web.Application()
        app.router.add_get("/health", self._health)
        app.router.add_get("/v1/models", self._models)
        app.router.add_post("/v1/completions", self._completions)
        app.router.add_post("/models/{model}/generate", self._generate)
        runner = web.AppRunner(app, access_log=None)
        await runner.setup()
        await web.TCPSite(runner, self.http_host, self.http_port).start()
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGTERM, signal.SIGINT):
            loop.add_signal_handler(sig, self._close)
        logger.info(f"MuxScheduler HTTP entry on "
                    f"http://{self.http_host}:{self.http_port}, models "
                    f"{list(self._served_ids)}")
        try:
            while not self.is_finished:
                await asyncio.sleep(0.5)
        finally:
            await runner.cleanup()

    def _close(self):
        if self.accepting:
            logger.info("MuxScheduler HTTP entry closing: finishing "
                        f"{len(self.counter) - 1} admitted requests")
        self.accepting = False
        self.counter.discard(_OPEN)

    async def _health(self, request):
        return web.json_response({"accepting": self.accepting})

    async def _models(self, request):
        data = [
            dict(id=served,
                 object="model",
                 owned_by="muxserve",
                 max_model_len=MAX_CONTEXT_TOKENS)
            for served in self._served_ids
        ]
        return web.json_response({"object": "list", "data": data})

    async def _completions(self, request):
        if not self.accepting:
            return web.json_response(
                {"error": {
                    "message": "Not accepting requests",
                    "type": "unavailable"
                }},
                status=503)
        try:
            body = await request.json()
            if not isinstance(body, dict):
                raise _BadRequest("The request body must be a JSON object")
            served = body.get("model")
            name = self._served_ids.get(served)
            if name is None:
                raise _BadRequest(f"Unknown model {served}; serving "
                                  f"{list(self._served_ids)}")
            input_ids, max_new_tokens = body.get("prompt"), body.get(
                "max_tokens")
            _check_request(input_ids, max_new_tokens,
                           body.get("temperature", 1), body.get("ignore_eos"),
                           body.get("stream"))
        except (ValueError, TypeError, AttributeError) as error:
            return web.json_response(
                {"error": {
                    "message": str(error),
                    "type": "invalid_request_error"
                }},
                status=400)

        created = int(time.time())
        prompt_tokens = len(input_ids)

        def frames(req, count, terminal):
            usage = dict(prompt_tokens=prompt_tokens,
                         completion_tokens=count,
                         total_tokens=prompt_tokens + count)
            head = dict(id=f"cmpl-{req.idx}",
                        object="text_completion",
                        created=created,
                        model=served)
            choice = dict(index=0,
                          text="",
                          logprobs=None,
                          finish_reason="length" if terminal else None)
            data = _sse(dict(head, choices=[choice], usage=usage))
            if terminal:
                usage = dict(usage, prompt_tokens_details=dict(cached_tokens=0))
                data += _sse(dict(head, choices=[], usage=usage)) + _DONE
            return data

        return await self._serve(request, name, input_ids, max_new_tokens, 1,
                                 frames)

    def _parse(self, model, body):
        names = {_normalize(name): name for name in self._served_models}
        name = names.get(_normalize(model))
        if name is None:
            raise _BadRequest(f"Unknown model {model}; serving "
                              f"{self._served_models}")
        sampling = body.get("sampling_params") or {}
        input_ids = body.get("input_ids")
        max_new_tokens = sampling.get("max_new_tokens")
        interval = sampling.get("stream_interval", 1)
        _check_request(input_ids, max_new_tokens,
                       sampling.get("temperature", 0),
                       sampling.get("ignore_eos"), body.get("stream"))
        if type(interval) is not int or interval < 1:
            raise _BadRequest("stream_interval must be a positive integer")
        return name, input_ids, max_new_tokens, interval

    async def _generate(self, request):
        if not self.accepting:
            return web.json_response({"error": "Not accepting requests"},
                                     status=503)
        try:
            body = await request.json()
            name, input_ids, max_new_tokens, interval = self._parse(
                request.match_info["model"], body)
        except (ValueError, TypeError, AttributeError) as error:
            return web.json_response({"error": str(error)}, status=400)
        rid = body.get("rid")

        def frames(req, count, terminal):
            meta = dict(id=rid,
                        prompt_tokens=len(input_ids),
                        completion_tokens=count,
                        cached_tokens=0,
                        finish_reason=dict(type="length") if terminal else None)
            data = _sse(
                dict(text="",
                     output_ids=list(req.output_tokens[:count]),
                     meta_info=meta))
            return data + _DONE if terminal else data

        return await self._serve(request, name, input_ids, max_new_tokens,
                                 interval, frames)

    async def _serve(self, request, name, input_ids, max_new_tokens, interval,
                     frames):
        """Queue one request as a replayed one is queued and stream
        frames(req, completion tokens, terminal) as its output grows."""
        response = web.StreamResponse(headers={
            "Content-Type": "text/event-stream",
            "Cache-Control": "no-cache"
        })
        await response.prepare(request)

        wake = asyncio.Event()
        async with self.lock:
            idx = len(self.workload.requests)
            _WAKES[idx] = wake
            req = Request(model_name=name,
                          slo=None,
                          idx=idx,
                          time_stamp={},
                          data=(np.array(input_ids), len(input_ids),
                                max_new_tokens))
            # Same bookkeeping as a replayed request: an arrival time, a
            # pending entry in counter, then the model's waiting queue.
            self.workload.requests.append(req)
            self.workload.arrivals = np.append(self.workload.arrivals,
                                               self.get_tick())
            self.counter.add(idx)
            self.waiting[name].append(req)

        sent = 0
        try:
            while True:
                await wake.wait()
                wake.clear()
                count = len(req.output_tokens or [])
                finished = idx not in self.counter
                if finished:
                    await response.write(frames(req, count, True))
                    break
                # A preempted request restarts from no tokens and is sent
                # again only beyond what the client already has.
                if count - sent >= interval:
                    await response.write(frames(req, count, False))
                    sent = count
        except ConnectionError:
            # The client left. MuxServe has no request cancellation; the
            # request still runs to completion inside the scheduler.
            pass
        finally:
            _WAKES.pop(idx, None)
        return response
