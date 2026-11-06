"""User-run acceptance for the frozen native controlled runtime.

No check in this file was run during static-only implementation.
--source checks AST and recorded fork hashes; --gate checks refusal without GPU.
--live drives already-launched real P/D ranks via their real owner, with no
gateway/policy driver concurrently consuming the same fresh execution group.
It never sends fabricated READY/GPU_DONE/HANDOFF_READY/QUIESCED reports.
"""
from __future__ import annotations

import argparse
import ast
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
from pathlib import PurePosixPath
import sys
import time
from types import SimpleNamespace
import uuid

ROOT = Path(__file__).absolute().parents[1]
sys.path.insert(0, str(ROOT))


def _is_link(path):
    junction = getattr(path, "is_junction", None)
    return path.is_symlink() or (junction is not None and junction())


def _repository_root():
    """Reject linked roots and the prohibited path before reading source."""
    current = Path(ROOT.anchor)
    anchor_parts = Path(ROOT.anchor).parts
    for part in ROOT.parts[len(anchor_parts):]:
        current = current / part
        prohibited = Path("/home/incoming")
        if os.name == "posix" and (current == prohibited or prohibited in current.parents):
            raise ValueError("This acceptance source excludes /home/incoming")
        if _is_link(current):
            raise ValueError("Acceptance source root contains a symlink or junction")
    if not current.is_dir():
        raise ValueError("Acceptance source root is not a directory")
    return current


def _source_file(name):
    """Read only canonical regular files inside this copied repository."""
    if (not isinstance(name, str) or not name or "\\" in name
            or name.startswith("/") or ":" in name):
        raise ValueError(f"Unsafe repository source path: {name!r}")
    relative = PurePosixPath(name)
    if relative.as_posix() != name or any(part in ("", ".", "..") for part in relative.parts):
        raise ValueError(f"Non-canonical repository source path: {name!r}")
    current = _repository_root()
    for part in relative.parts:
        current = current / part
        junction = getattr(current, "is_junction", None)
        if current.is_symlink() or (junction is not None and junction()):
            raise ValueError(f"Repository source path contains a link: {name}")
    if not current.is_file():
        raise ValueError(f"Repository source is not a regular file: {name}")
    return current


def source_check():
    extraction = json.loads(_source_file("upstream-reference/extraction.json").read_text())
    patches = json.loads(_source_file("upstream-reference/patches.json").read_text())
    for name, record in patches.items():
        data = _source_file("third_party/sglang/" + name).read_bytes()
        assert hashlib.sha256(data).hexdigest() == record["patched_sha256"], name
        assert record["source_sha256"] == extraction["files_sha256"][name], name
        ast.parse(data.decode(), filename=name)
    runtime = ast.parse(_source_file("muxserve_sglang/sglang_runtime.py").read_text())
    methods = {n.name: n for n in ast.walk(runtime) if isinstance(n, ast.FunctionDef)}
    prepare_calls = [n.func.attr for n in ast.walk(methods["_prepare"])
                     if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)]
    assert "prepare_for_decode" in prepare_calls
    assert "process_prebuilt" not in prepare_calls
    assert "prepare_for_prebuilt" not in prepare_calls
    start = methods["_start"]
    rebuild = next(n for n in ast.walk(start) if isinstance(n, ast.If)
                   and "REBUILD" in ast.unparse(n.test))
    body_calls = [ast.unparse(n.func) for s in rebuild.body for n in ast.walk(s)
                  if isinstance(n, ast.Call)]
    assert "self.runner.forward" in body_calls
    assert "self.s.run_batch" not in body_calls
    assert not any("sample" in call for call in body_calls)
    forward_ast = ast.parse(_source_file(
        "third_party/sglang/python/sglang/srt/model_executor/forward_batch_info.py").read_text())
    native_init = next(n for n in ast.walk(forward_ast)
                       if isinstance(n, ast.FunctionDef) and n.name == "init_new")
    assert "return_hidden_states_before_norm" in [a.arg for a in native_init.args.kwonlyargs]
    assert any(isinstance(n, ast.keyword) and n.arg == "return_hidden_states_before_norm"
               for n in ast.walk(rebuild))
    print("Source/hash/signature checks passed; no native or GPU execution implied.")


def gate_check():
    from muxserve_sglang.contracts import IntegrationError
    from muxserve_sglang.sglang_hooks import assert_forward_permitted
    previous = os.environ.get("MUXSERVE_EXECUTION_GROUP")
    os.environ["MUXSERVE_EXECUTION_GROUP"] = "acceptance-refusal-only"
    try:
        for permit in (None, {"grant_id": "used", "consumed": True}):
            # Refusal must occur before touching a native pool or CUDA object.
            runner = SimpleNamespace(_muxserve_permit=permit)
            try:
                assert_forward_permitted(runner, object())
            except IntegrationError as exc:
                assert exc.code == "FORWARD_NOT_GRANTED"
            else:
                raise AssertionError("Ungrant/replay forward was accepted")
    finally:
        if previous is None:
            os.environ.pop("MUXSERVE_EXECUTION_GROUP", None)
        else:
            os.environ["MUXSERVE_EXECUTION_GROUP"] = previous
    print("Gate refusal checks passed; no model forward was executed.")


class Live:
    def __init__(self, config_path, model):
        from muxserve_sglang.config import load_config, safe_path
        from muxserve_sglang.ipc import StorageRPCClient
        self.config = load_config(config_path)
        self.model = model
        self.config.model(model)
        runtime_dir = safe_path(self.config.runtime_dir)
        self.rpc = StorageRPCClient(str(safe_path(runtime_dir / "store.sock")),
                                   safe_path(runtime_dir / "auth.key").read_bytes(), timeout_s=5)
        group = self.call("snapshot")["groups"][model]
        assert group["ready"] and not group["fenced"] and not group["batches"] and not group["pending_events"]
        assert self.config.tp_size == 4, "Real acceptance requires TP4"
        self.group = group["group_id"]
        # Dedicated initialization used and fully ACKed this same group epoch.
        # Never request already retired warm-up events from sequence zero.
        self.event_seq = group["acked_event_seq"]
        self.events = []
        self.tokens = {}
        self.batch_history = []
        self.tag = "runtime-" + uuid.uuid4().hex

    def call(self, name, *args):
        return self.rpc.call("controller_" + name, *args)

    def pump(self):
        self.call("tick")
        for event in self.call("poll_events", self.group, self.event_seq):
            assert event.event_seq == self.event_seq + 1
            if event.kind == "TOKEN":
                tokens = self.tokens.setdefault(event.request_id, [])
                assert event.payload["output_index"] == len(tokens)
                tokens.extend(event.payload["token_ids"])
            self.events.append(asdict(event))
            self.event_seq = event.event_seq
        if self.event_seq:
            self.call("ack_events", self.group, self.event_seq)
        group = self.call("snapshot")["groups"][self.model]
        assert not group["fenced"], group["fenced"]
        return group

    def wait(self, predicate):
        deadline = time.monotonic() + self.config.request_timeout_ms / 1000
        while True:
            value = predicate(self.pump())
            if value:
                return value
            if time.monotonic() >= deadline:
                # A local timeout does not release resources or prove GPU idle.
                raise TimeoutError("Acceptance wait expired; inspect owner before cleanup")
            time.sleep(0.01)

    def output(self, rid):
        return self.call("request_output", self.model, rid)

    def submit(self, suffix, inputs, maximum, ignore_eos=True):
        from muxserve_sglang.contracts import RequestSpec
        rid = self.tag + "-" + suffix
        self.call("submit_request", RequestSpec(self.model, rid, tuple(inputs), maximum,
                  self.config.request_timeout_ms, {"temperature": 0, "ignore_eos": ignore_eos}))
        return rid

    def prepare(self, rid, stage, purpose="GENERATE"):
        return self.prepare_many((rid,), stage, purpose)

    def prepare_many(self, request_ids, stage, purpose="GENERATE"):
        from muxserve_sglang.contracts import BatchPlan, RequestRef
        outputs = [self.output(rid) for rid in request_ids]
        bid = self.tag + "-b" + str(len(self.batch_history))
        plan = BatchPlan(self.group, bid, self.model, stage,
            tuple(RequestRef(rid, output["attempt"], output["output_count"])
                  for rid, output in zip(request_ids, outputs)), purpose)
        self.call("prepare_batch", plan)
        self.batch_history.append({"batch_id": bid, "stage": stage, "purpose": purpose,
                                   "request_ids": tuple(request_ids)})
        self.wait(lambda g: len(g["batches"][bid]["ready"]) == 2 * self.config.tp_size)
        return bid

    def execute(self, rid, stage, purpose="GENERATE"):
        self.execute_many((rid,), stage, purpose)
        return self.output(rid)

    def execute_many(self, request_ids, stage, purpose="GENERATE"):
        bid = self.prepare_many(request_ids, stage, purpose)
        self.call("start_batch", self.group, bid)
        self.wait(lambda g: g["batches"][bid]["status"] == "DONE")
        return tuple(self.output(rid) for rid in request_ids)

    def terminal(self, rid, state="FINISHED"):
        result = self.wait(lambda _: (out if (out := self.output(rid))["state"] == state else None))
        assert tuple(self.tokens.get(rid, ())) == result["output_token_ids"]
        return result

    def cancel(self, rid):
        self.call("cancel_request", self.model, rid, "runtime_acceptance")
        return self.terminal(rid, "CANCELLED")

    def run(self, inputs, first_eos_inputs, long_rebuild=False):
        # Max-one output must complete through metadata-only D handoff.
        rid = self.submit("one", inputs, 1)
        self.execute(rid, "P")
        assert self.terminal(rid)["output_count"] == 1
        # Prepared P revocation releases logical state; retry is a new attempt.
        rid = self.submit("revoke", inputs, 2)
        bid = self.prepare(rid, "P")
        self.call("abort_batch", self.group, bid, "acceptance_prepared_abort")
        self.wait(lambda g: g["batches"][bid]["status"] == "ABORTED")
        assert self.output(rid)["output_count"] == 0
        self.execute(rid, "P")
        # Prepared D revocation restores its logical append counter and relay.
        bid = self.prepare(rid, "D")
        self.call("abort_batch", self.group, bid, "acceptance_decode_abort")
        self.wait(lambda g: g["batches"][bid]["status"] == "ABORTED")
        assert self.output(rid)["output_count"] == 1
        self.execute(rid, "D")
        assert self.terminal(rid)["output_count"] == 2
        # Real suspend/rebuild retains every indexed output and does not sample
        # an additional first token during the rebuilt prefill.
        rid = self.submit("resume", inputs, 4)
        self.execute(rid, "P")
        self.execute(rid, "D")
        before = self.output(rid)
        self.call("suspend_request", self.model, rid, before["attempt"])
        suspended = self.terminal(rid, "SUSPENDED")
        self.call("resume_request", self.model, rid, suspended["resume_record_id"])
        rebuilt = self.execute(rid, "P", "REBUILD")
        assert rebuilt["output_token_ids"] == before["output_token_ids"]
        self.execute(rid, "D")
        self.execute(rid, "D")
        assert self.terminal(rid)["output_count"] == 4
        rid = self.submit("cancel", inputs, 4)
        self.execute(rid, "P")
        assert self.cancel(rid)["output_count"] == 1
        if self.config.max_model_inflight == 2:
            # Real multi-request selection: unequal lengths, changing order,
            # and rollback of two logical append allocations before retry.
            assert len(inputs) > 0
            other_inputs = (list(inputs) + [inputs[-1]]) if len(inputs) < 1792 else list(inputs[:-1])
            first = self.submit("batch-a", inputs, 6)
            second = self.submit("batch-b", other_inputs, 5)
            self.execute_many((first, second), "P")
            self.execute_many((second, first), "D")
            before = (self.output(first)["output_token_ids"], self.output(second)["output_token_ids"])
            bid = self.prepare_many((first, second), "D")
            self.call("abort_batch", self.group, bid, "acceptance_batched_decode_abort")
            self.wait(lambda g: g["batches"][bid]["status"] == "ABORTED")
            assert before == (self.output(first)["output_token_ids"], self.output(second)["output_token_ids"])
            self.execute_many((first, second), "D")
            self.execute_many((second, first), "D")
            self.execute_many((first, second), "D")
            assert self.terminal(second)["output_count"] == 5
            self.execute(first, "D")
            assert self.terminal(first)["output_count"] == 6
            first = self.submit("overlap-d", inputs, 2)
            self.execute(first, "P")
            second = self.submit("overlap-p", inputs, 1)
            p = self.prepare(second, "P")
            d = self.prepare(first, "D")
            self.call("start_batch", self.group, p)
            self.call("start_batch", self.group, d)
            self.wait(lambda g: all(g["batches"][b]["status"] == "DONE" for b in (p, d)))
            assert self.terminal(first)["output_count"] == 2
            assert self.terminal(second)["output_count"] == 1
        if long_rebuild:
            assert inputs, "At least one real prompt ID is required"
            long_inputs = (list(inputs) * (1792 // len(inputs) + 1))[:1792]
            rid = self.submit("context-edge", long_inputs, 256)
            self.execute(rid, "P")
            for _ in range(254):
                self.execute(rid, "D")
            before = self.output(rid)
            assert before["output_count"] == 255
            self.call("suspend_request", self.model, rid, before["attempt"])
            suspended = self.terminal(rid, "SUSPENDED")
            self.call("resume_request", self.model, rid, suspended["resume_record_id"])
            rebuilt = self.execute(rid, "P", "REBUILD")
            assert rebuilt["output_token_ids"] == before["output_token_ids"]
            self.execute(rid, "D")
            assert self.terminal(rid)["output_count"] == 256
        if first_eos_inputs is not None:
            rid = self.submit("first-eos", first_eos_inputs, 4, ignore_eos=False)
            self.execute(rid, "P")
            result = self.output(rid)
            if result["finish_reason"] != "eos":
                self.cancel(rid)
                raise AssertionError("EOS fixture did not produce first-token EOS")
            assert self.terminal(rid)["output_count"] == 1
        else:
            print("First-token EOS fixture absent: that scenario remains UNEXECUTED.")
        self.wait(lambda g: all(rank["pending_commands"] == 0 for rank in g["ranks"].values()))
        scenarios = {
            "max_one_output": "PASS",
            "prepared_p_revoke_retry": "PASS",
            "prepared_d_revoke_retry": "PASS",
            "short_suspend_rebuild": "PASS",
            "cancellation": "PASS",
            "inflight2_multi_request_and_pd_overlap": (
                "PASS" if self.config.max_model_inflight == 2 else "UNEXECUTED"),
            "long_context_rebuild": "PASS" if long_rebuild else "UNEXECUTED",
            "first_token_eos": ("PASS" if first_eos_inputs is not None else "UNEXECUTED"),
        }
        return {"group_id": self.group, "batches": self.batch_history, "events": self.events,
                "scenarios": scenarios,
                "limits": "No numerical/reference, IPC pointer, fault injection or performance proof"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", action="store_true")
    parser.add_argument("--gate", action="store_true")
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--config")
    parser.add_argument("--model")
    parser.add_argument("--input-ids", help="JSON list of real prompt token IDs")
    parser.add_argument("--first-eos-input-ids", help="JSON list known to generate first-token EOS")
    parser.add_argument("--long-rebuild", action="store_true", help="Run real 1792-prompt/255-output suspend/rebuild edge")
    parser.add_argument("--output", default="runtime-acceptance-evidence.json")
    args = parser.parse_args()
    if not any((args.source, args.gate, args.live)):
        parser.error("Choose --source, --gate or --live")
    if args.source:
        source_check()
    if args.gate:
        gate_check()
    if args.live:
        if not all((args.config, args.model, args.input_ids)):
            parser.error("--live requires --config, --model and --input-ids")
        live = Live(args.config, args.model)
        evidence = live.run(json.loads(args.input_ids),
            json.loads(args.first_eos_input_ids) if args.first_eos_input_ids else None,
            args.long_rebuild)
        from muxserve_sglang.config import safe_path
        safe_path(args.output).write_text(json.dumps(evidence, indent=2) + "\n")
        print("Real native boundary scenarios completed; evidence saved to", args.output)


if __name__ == "__main__":
    main()
