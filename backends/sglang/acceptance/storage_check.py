"""User-run checks. This file has NOT been executed during implementation.

CPU: python acceptance/storage_check.py
CUDA: CUDA_VISIBLE_DEVICES=0 python acceptance/storage_check.py --cuda-ipc
Use --gpu-ordinals 0,1 for explicit multi-rank testing on visible A100 devices.
No simulated GPU backend is used. CUDA checks require Linux and installed torch.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import os
from pathlib import Path
import sys
from threading import Barrier, Event, Lock, Thread

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from muxserve_sglang.contracts import IntegrationError, KVTarget, ModelLayout, RequestIdentity
from muxserve_sglang.storage import HeadwiseStore


def expect_code(code, operation):
    try:
        operation()
    except IntegrationError as exc:
        assert exc.code == code, (exc.code, code)
    else:
        raise AssertionError(f"Expected {code}")


def static_pool_partition_checks():
    """The static-pool ablation must not borrow another model's quota."""
    model_ids = ("m1", "m2", "m3")
    store = HeadwiseStore({0: 9}, pool_mode="static", model_ids=model_ids)
    for model_id in model_ids:
        store.register_model(ModelLayout(model_id, 1, 1, (0,)))
    initial = store.snapshot()
    partitions = initial["model_partitions"]
    assert {model_id: partitions[model_id]["total_blocks_by_rank"][0]
            for model_id in model_ids} == {"m1": 3, "m2": 3, "m3": 3}
    ranges = [partitions[model_id]["block_ranges_by_rank"][0] for model_id in model_ids]
    assert ranges == [(0, 3), (3, 6), (6, 9)]

    batches = []
    def reserve(model_id, request_id):
        batch = store.reserve_kv(f"{model_id}-{request_id}",
            [KVTarget(RequestIdentity(model_id, request_id), 1)], "P")
        batches.append(batch)
        return batch

    for request_id in ("a", "b", "c"):
        reserve("m1", request_id)
    before = store.snapshot()
    expect_code("OUT_OF_CAPACITY", lambda: reserve("m1", "overflow"))
    assert store.snapshot() == before
    reserve("m2", "still-has-quota")
    after = store.snapshot()
    assert after["model_partitions"]["m1"]["free_blocks_by_rank"][0] == 0
    assert after["model_partitions"]["m2"]["free_blocks_by_rank"][0] == 2
    assert after["ranks"][0]["free_blocks"] == 5
    store.assert_integrity()
    for batch in batches:
        store.abort_kv_batch(batch.reservation_id)
    drained = store.snapshot()
    assert drained["ranks"][0]["free_blocks"] == 9
    assert all(drained["model_partitions"][model_id]["free_blocks_by_rank"][0] == 3
               for model_id in model_ids)
    store.assert_integrity()


def batch_history_budget_checks():
    """Batch history is independent of the request-key cap; no stale replay."""
    store = HeadwiseStore({0: 8}, batch_history_limit=3)
    store.register_model(ModelLayout("model", 1, 1, (0,)))
    for index in range(3):
        batch_id = f"budget-{index}"
        reservation = store.reserve_kv(batch_id,
            [KVTarget(RequestIdentity("model", f"r{index}"), 1)], "P")
        store.abort_kv_batch(reservation.reservation_id)
        store.retire_history((batch_id,))
    before = store.snapshot()
    assert before["history"]["lifetime_batch_limit"] == 3
    expect_code("RETIRED_BATCH", lambda: store.reserve_kv("budget-0",
        [KVTarget(RequestIdentity("model", "r0"), 1)], "P"))
    expect_code("HISTORY_FULL", lambda: store.reserve_kv("budget-3",
        [KVTarget(RequestIdentity("model", "r3"), 1)], "P"))
    assert store.snapshot() == before
    store.assert_integrity()


def allocator_checks():
    store = HeadwiseStore({0: 32, 1: 32})
    layout = ModelLayout("model", 2, 2, (0, 1))
    store.register_model(layout)
    identity = RequestIdentity("model", "request")
    target = KVTarget(identity, 17)
    reservation = store.reserve_kv("prefill", [target], "P")
    aid = reservation.views[0].allocation_id
    assert reservation.views[0].capacity_tokens == 32
    assert all(store.snapshot()["ranks"][r]["used_blocks"] == 8 for r in (0, 1))
    assert store.reserve_kv("prefill", [target], "P") == reservation
    expect_code("IDENTITY_CONFLICT", lambda: store.reserve_kv("prefill", [KVTarget(identity, 18)], "P"))
    expect_code("ALLOCATION_BUSY", lambda: store.reserve_kv("busy", [KVTarget(identity, 33, aid)], "P"))
    # Returned tables are immutable PackedBlocks: callers cannot corrupt
    # allocator bookkeeping, and the packed view matches the store mapping.
    returned = reservation.views[0].block_tables[0]
    try:
        returned.data = b""
    except Exception:
        pass
    else:
        raise AssertionError("Returned block table is mutable")
    assert returned.tolist() == store._allocations[aid].tables[0]
    store.assert_integrity()
    store.start_kv_batch(reservation.reservation_id)
    store.start_kv_batch(reservation.reservation_id)
    result = store.finish_kv_batch(reservation.reservation_id, "SUCCESS", {aid: 17}, (0, 1))
    assert result[0].valid_tokens == 17
    assert store.finish_kv_batch(reservation.reservation_id, "SUCCESS", {aid: 17}, (1, 0)) == result
    expect_code("IDENTITY_CONFLICT", lambda: store.finish_kv_batch(
        reservation.reservation_id, "SUCCESS", {aid: 18}, (0, 1)))
    store.bind_kv(aid, "D")
    assert store.bind_kv(aid, "D") == result[0]
    for invalid in (0, -1, 2049, 16):
        expect_code("INVALID_ARGUMENT", lambda n=invalid: store.reserve_kv(
            f"invalid-{n}", [KVTarget(identity, n, aid)], "D"))
    old = store.snapshot()["allocations"][aid]["block_tables"]
    growth = store.reserve_kv("growth", [KVTarget(identity, 33, aid)], "temporary")
    assert all(store.snapshot()["ranks"][r]["used_blocks"] == 12 for r in (0, 1))
    store.abort_kv_batch(growth.reservation_id)
    store.abort_kv_batch(growth.reservation_id)
    assert store.snapshot()["allocations"][aid]["block_tables"] == old
    assert store.snapshot()["allocations"][aid]["holders"] == ["D", "P"]
    # A stale abort cannot roll back a later reservation's growth.
    growth2 = store.reserve_kv("growth2", [KVTarget(identity, 33, aid)], "D")
    store.abort_kv_batch(growth.reservation_id)
    assert all(store.snapshot()["ranks"][r]["used_blocks"] == 12 for r in (0, 1))
    store.start_kv_batch(growth2.reservation_id)
    expect_code("BATCH_FENCED", lambda: store.finish_kv_batch(
        growth2.reservation_id, "FAILURE", {}, (0,)))
    store.release_kv(aid, "P")
    store.release_kv(aid, "D")
    assert all(store.snapshot()["ranks"][r]["used_blocks"] == 12 for r in (0, 1))
    assert growth2.reservation_id in store.snapshot()["fences"]
    store.assert_integrity()
    store.finish_kv_batch(growth2.reservation_id, "FAILURE", {}, (1,))
    assert not store.snapshot()["allocations"]
    store.release_kv(aid, "D")
    expect_code("STALE_ATTEMPT", lambda: store.reserve_kv("retired", [target], "P"))
    # New attempt has a never-reused allocation ID.
    retry = store.reserve_kv("retry", [KVTarget(RequestIdentity("model", "request", 1), 1)], "P")
    assert retry.views[0].allocation_id != aid
    store.release_kv(aid, "P")
    store.start_kv_batch(retry.reservation_id)
    expect_code("BATCH_FENCED", lambda: store.abort_kv_batch(retry.reservation_id))
    expect_code("BATCH_FENCED", lambda: store.finish_kv_batch(retry.reservation_id, "UNKNOWN", {}, (0, 1)))
    store.release_kv(retry.views[0].allocation_id, "P")
    assert all(store.snapshot()["ranks"][r]["used_blocks"] == 4 for r in (0, 1))
    store.finish_kv_batch(retry.reservation_id, "FAILURE", {}, (0, 1))
    assert not store.snapshot()["allocations"]
    store.assert_integrity()

    asymmetric = HeadwiseStore({0: 20, 1: 6})
    asymmetric.register_model(layout)
    before = asymmetric.snapshot()
    expect_code("OUT_OF_CAPACITY", lambda: asymmetric.reserve_kv("atomic", [target], "P"))
    assert asymmetric.snapshot() == before
    asymmetric.assert_integrity()

    # Concurrent ownership transitions must keep both ranks consistent.
    concurrent = HeadwiseStore({0: 128, 1: 128})
    concurrent.register_model(ModelLayout("m", 1, 1, (0, 1)))
    def transition(index):
        identity = RequestIdentity("m", str(index))
        batch = concurrent.reserve_kv(str(index), [KVTarget(identity, 17)], f"P:{index}")
        aid = batch.views[0].allocation_id
        concurrent.start_kv_batch(batch.reservation_id)
        concurrent.finish_kv_batch(batch.reservation_id, "SUCCESS", {aid: 17}, (0, 1))
        concurrent.bind_kv(aid, f"D:{index}")
        concurrent.release_kv(aid, f"P:{index}")
        concurrent.assert_integrity()
        concurrent.release_kv(aid, f"D:{index}")
        concurrent.release_kv(aid, f"D:{index}")
    with ThreadPoolExecutor(max_workers=16) as executor:
        list(executor.map(transition, range(64)))
    assert not concurrent.snapshot()["allocations"]
    concurrent.assert_integrity()

    base = concurrent.reserve_kv("base", [KVTarget(RequestIdentity("m", "shared"), 1)], "P")
    shared = base.views[0].allocation_id
    concurrent.start_kv_batch(base.reservation_id)
    concurrent.finish_kv_batch(base.reservation_id, "SUCCESS", {shared: 1}, (0, 1))
    barrier = Barrier(2)
    def compete(index):
        barrier.wait()
        try:
            return concurrent.reserve_kv(f"contend-{index}",
                [KVTarget(RequestIdentity("m", "shared"), 17, shared)], "D")
        except IntegrationError as exc:
            assert exc.code == "ALLOCATION_BUSY"
            return None
    with ThreadPoolExecutor(max_workers=2) as executor:
        contenders = list(executor.map(compete, (0, 1)))
    winners = [entry for entry in contenders if entry is not None]
    assert len(winners) == 1
    concurrent.abort_kv_batch(winners[0].reservation_id)
    concurrent.release_kv(shared, "P")
    concurrent.assert_integrity()
    assert not concurrent.snapshot()["allocations"]
    print("CPU allocator checks passed")


def reconciliation_checks():
    """User-run evidence, attempt, and prepared handoff regressions."""
    store = HeadwiseStore({0: 32, 1: 32})
    store.register_model(ModelLayout("reconcile", 1, 1, (0, 1)))

    def reserve(name, attempt=0):
        identity = RequestIdentity("reconcile", name, attempt)
        batch = store.reserve_kv(f"{name}-{attempt}", [KVTarget(identity, 5)], "P")
        return batch, batch.views[0].allocation_id

    failed, failed_aid = reserve("failed")
    store.start_kv_batch(failed.reservation_id)
    expect_code("BATCH_FENCED", lambda: store.finish_kv_batch(
        failed.reservation_id, "FAILURE", {}, (0,)))
    assert store.finish_kv_batch(failed.reservation_id, "SUCCESS", {failed_aid: 5}, (1,)) == ()
    assert store.finish_kv_batch(failed.reservation_id, "FAILURE", {}, (0,)) == ()
    assert store.snapshot()["reservations"][failed.reservation_id]["state"] == "FAILED"
    assert failed_aid not in store.snapshot()["allocations"]
    store.release_kv(failed_aid, "P")

    lengths, lengths_aid = reserve("lengths")
    store.start_kv_batch(lengths.reservation_id)
    expect_code("BATCH_FENCED", lambda: store.finish_kv_batch(
        lengths.reservation_id, "SUCCESS", {lengths_aid: 5}, (0,)))
    before = store.snapshot()["reservations"][lengths.reservation_id]
    expect_code("IDENTITY_CONFLICT", lambda: store.finish_kv_batch(
        lengths.reservation_id, "SUCCESS", {lengths_aid: 6}, (1,)))
    assert store.snapshot()["reservations"][lengths.reservation_id] == before
    assert store.finish_kv_batch(lengths.reservation_id, "SUCCESS", {lengths_aid: 5}, (1,))[0].valid_tokens == 5
    assert store.finish_kv_batch(lengths.reservation_id, "SUCCESS", {lengths_aid: 5}, (0,))[0].valid_tokens == 5
    store.release_kv(lengths_aid, "P")

    unknown, unknown_aid = reserve("unknown")
    store.start_kv_batch(unknown.reservation_id)
    expect_code("BATCH_FENCED", lambda: store.finish_kv_batch(
        unknown.reservation_id, "UNKNOWN", {}, (0, 1)))
    expect_code("BATCH_FENCED", lambda: store.finish_kv_batch(
        unknown.reservation_id, "SUCCESS", {unknown_aid: 5}, (0,)))
    assert store.finish_kv_batch(unknown.reservation_id, "SUCCESS", {unknown_aid: 5}, (1,))[0].valid_tokens == 5
    store.release_kv(unknown_aid, "P")

    prepared, prepared_aid = reserve("prepared")
    handoff = store.bind_kv(prepared_aid, "D")
    assert handoff.valid_tokens == 0 and handoff.capacity_tokens == 16
    store.start_kv_batch(prepared.reservation_id)
    expect_code("ALLOCATION_BUSY", lambda: store.bind_kv(prepared_aid, "other"))
    expect_code("BATCH_FENCED", lambda: store.abort_kv_batch(prepared.reservation_id))
    store.finish_kv_batch(prepared.reservation_id, "FAILURE", {}, (0, 1))
    store.release_kv(prepared_aid, "D")
    store.release_kv(prepared_aid, "P")

    # A prepared growth bind carries committed data metadata, but grants no
    # implicit read/forward permission and does not start the reservation.
    committed, committed_aid = reserve("committed")
    store.start_kv_batch(committed.reservation_id)
    store.finish_kv_batch(committed.reservation_id, "SUCCESS", {committed_aid: 5}, (0, 1))
    growth = store.reserve_kv("committed-growth", [KVTarget(
        RequestIdentity("reconcile", "committed"), 17, committed_aid)], "P")
    assert store.bind_kv(committed_aid, "D").valid_tokens == 5
    assert store.snapshot()["reservations"][growth.reservation_id]["state"] == "PREPARING"
    store.abort_kv_batch(growth.reservation_id)
    assert store.snapshot()["allocations"][committed_aid]["holders"] == ["D", "P"]
    store.release_kv(committed_aid, "D")
    store.release_kv(committed_aid, "P")

    aborted, aborted_aid = reserve("aborted")
    assert store.bind_kv(aborted_aid, "D").valid_tokens == 0
    store.abort_kv_batch(aborted.reservation_id)
    assert store.snapshot()["allocations"][aborted_aid]["holders"] == ["D"]
    assert store.snapshot()["allocations"][aborted_aid]["capacity_tokens"] == 0
    store.release_kv(aborted_aid, "D")

    old, old_aid = reserve("attempt")
    store.start_kv_batch(old.reservation_id)
    expect_code("BATCH_FENCED", lambda: store.finish_kv_batch(
        old.reservation_id, "UNKNOWN", {}, (0,)))
    newer, newer_aid = reserve("attempt", 1)
    assert old_aid in store.snapshot()["allocations"]
    expect_code("STALE_ATTEMPT", lambda: store.bind_kv(old_aid, "D"))
    store.finish_kv_batch(old.reservation_id, "FAILURE", {}, (1,))
    store.release_kv(old_aid, "P")
    store.abort_kv_batch(newer.reservation_id)
    store.release_kv(newer_aid, "P")
    store.assert_integrity()
    print("CPU reconciliation and prepared-bind checks passed")


def rpc_worker_bound_checks():
    """User-run check that a timed-out native call keeps its worker slot."""
    from muxserve_sglang.ipc import _bounded

    gate = Event()
    lock = Lock()
    entered = 0
    def blocked():
        nonlocal entered
        with lock:
            entered += 1
        gate.wait()
    try:
        for _ in range(32):
            expect_code("RPC_TIMEOUT", lambda: _bounded(blocked, 0.05))
        with lock:
            assert entered == 32, "Every admitted native worker must enter"
        expect_code("RPC_TIMEOUT", lambda: _bounded(blocked, 0.01))
        with lock:
            assert entered == 32, "Admission exceeded the native worker cap"
    finally:
        gate.set()
    _bounded(lambda: None, 5.0)
    print("RPC native worker bound check passed")


def rpc_transport_and_lifetime_checks():
    """User-run stdlib transport and stand-in owner fault checks (Linux only)."""
    import socket
    import tempfile
    import time
    from multiprocessing.connection import Connection
    from unittest.mock import patch
    from muxserve_sglang.ipc import _DeadlineConnection, StorageRPCClient, StorageRPCServer

    assert os.name == "posix", "AF_UNIX owner checks require Linux"
    left, right = socket.socketpair()
    pending = []
    receiver = Connection(left.detach())
    try:
        right.sendall(b"\x00\x00\x00\x05ab")  # Five-byte frame, only two bytes sent.
        timed = _DeadlineConnection(receiver, time.monotonic() + 0.1, pending)
        expect_code("RPC_TIMEOUT", lambda: timed.recv_bytes(1024))
    finally:
        right.close()
        receiver.close()
        for worker in pending:
            worker.join(timeout=5)
            assert not worker.is_alive(), "Truncated transport worker did not drain"

    class StandInOwner:
        def __init__(self):
            self.protected = True
            self.closed = False
            self.close_called = Event()

        def close(self):
            self.close_called.set()
            if self.protected:
                raise IntegrationError("OWNER_BUSY", "Stand-in resource is protected")
            self.closed = True

        def export_pool(self, *args):
            raise AssertionError("No GPU export in stand-in check")

        def release_consumer(self, *args):
            raise AssertionError("No GPU consumer in stand-in check")

        def owner_snapshot(self):
            return {"protected": self.protected}

    class FailedListener:
        def accept(self):
            raise OSError("injected accept failure")

        def close(self):
            pass

    class FailedStartThread(Thread):
        def start(self):
            raise RuntimeError("injected thread-start failure")

    class OneConnectionListener:
        def accept(self):
            return self

        def close(self):
            pass

    with tempfile.TemporaryDirectory(prefix="mx-owner-") as directory:
        os.chmod(directory, 0o700)
        store = HeadwiseStore({0: 8})
        owner = StandInOwner()
        server = StorageRPCServer(store, owner, os.path.join(directory, "failed.sock"), os.urandom(32))
        server._listener.close()
        server._listener = FailedListener()
        serving = Thread(target=server.serve_forever, daemon=True)
        serving.start()
        assert server.failure_event.wait(5)
        assert server.failure_status == "RPC_LISTENER_FAILED"
        assert owner.close_called.wait(5)
        assert serving.is_alive() and not owner.closed
        owner.protected = False
        owner.close()
        server._failure_hold.set()  # Stand-in only: allow the test thread to drain.
        serving.join(timeout=5)
        assert not serving.is_alive()
        server._failure_thread.join(timeout=5)
        assert not server._failure_thread.is_alive()

        # A handler start can fail at the process thread limit. Its keeper was
        # already started during construction and needs no new thread here.
        owner = StandInOwner()
        server = StorageRPCServer(store, owner, os.path.join(directory, "handler.sock"), os.urandom(32))
        server._listener.close()
        server._listener = OneConnectionListener()
        serving = Thread(target=server.serve_forever, daemon=True)
        with patch("muxserve_sglang.ipc.Thread", FailedStartThread):
            serving.start()
            assert server.failure_event.wait(5)
        assert owner.close_called.wait(5)
        assert serving.is_alive() and server._failure_thread.is_alive()
        owner.protected = False
        owner.close()
        server._failure_hold.set()  # Stand-in cleanup after proving retention.
        serving.join(timeout=5)
        server._failure_thread.join(timeout=5)
        assert not serving.is_alive() and not server._failure_thread.is_alive()

        owner = StandInOwner()
        owner.protected = False
        startup_failed = Event()
        expect_code("INVALID_ARGUMENT", lambda: StorageRPCServer(
            store, owner, os.path.join(directory, "init.sock"), b"short",
            failure_event=startup_failed))
        assert startup_failed.is_set() and owner.closed

        # If the keeper itself cannot start during initialization, the
        # non-daemon constructor thread holds protected ownership instead.
        owner = StandInOwner()
        failed_server = object.__new__(StorageRPCServer)
        failed_start = Event()
        keeper_failure = Event()
        def construct_with_failed_keeper():
            with patch("muxserve_sglang.ipc.Thread", FailedStartThread):
                try:
                    StorageRPCServer.__init__(failed_server, store, owner,
                        os.path.join(directory, "keeper.sock"), os.urandom(32),
                        failure_event=keeper_failure)
                except RuntimeError:
                    failed_start.set()
        constructor = Thread(target=construct_with_failed_keeper)
        constructor.start()
        assert keeper_failure.wait(5)
        assert owner.close_called.wait(5)
        assert constructor.is_alive() and not owner.closed
        owner.protected = False
        owner.close()
        failed_server._failure_hold.set()  # Stand-in cleanup only.
        constructor.join(timeout=5)
        assert not constructor.is_alive() and failed_start.is_set()

        gate, entered = Event(), Event()
        owner = StandInOwner()
        owner.protected = False
        def blocked():
            entered.set()
            gate.wait()
        address = os.path.join(directory, "drain.sock")
        authkey = os.urandom(32)
        server = StorageRPCServer(store, owner, address, authkey, timeout_s=0.2,
                                  extra_methods={"blocked": blocked}, owner_close=owner.close,
                                  allow_owner_shutdown=True)
        serving = Thread(target=server.serve_forever)
        serving.start()
        client = StorageRPCClient(address, authkey, timeout_s=0.2)
        try:
            expect_code("RPC_TIMEOUT", lambda: client.call("blocked"))
            assert entered.wait(5)
            server.close()
            assert serving.is_alive(), "Owner exited while admitted method still ran"
        finally:
            gate.set()
        serving.join(timeout=5)
        assert not serving.is_alive() and owner.closed
        server._failure_thread.join(timeout=5)
        assert not server._failure_thread.is_alive(), "Normal shutdown leaked owner keeper"
    print("RPC transport and owner lifetime checks passed")


def _cuda_consumer(address, authkey, consumer_id, ranks, write, release_event, queue):
    from muxserve_sglang.gpu_storage import rebuild_pool
    from muxserve_sglang.ipc import StorageRPCClient
    import torch
    pools = []
    try:
        client = StorageRPCClient(address, authkey)
        for rank in ranks:
            pool = rebuild_pool(client.export_pool(rank, consumer_id))
            assert rebuild_pool(pool.metadata) is pool
            pools.append(pool)
            if write:
                pool.key[0, 0, 0, 0] = 17 + rank
                pool.value[0, 0, 0] = 42 + rank
                torch.cuda.synchronize(pool.device_ordinal)
            else:
                assert pool.key[0, 0, 0, 0].item() == 17 + rank
                assert pool.value[0, 0, 0].item() == 42 + rank
        queue.put({"state": "ready"})
        if not release_event.wait(60):
            raise RuntimeError("Parent did not authorize consumer teardown")
        for pool in pools:
            pool.close()
        client.release_consumer(consumer_id, ranks)
        client.release_consumer(consumer_id, ranks)
        queue.put({"state": "done"})
    except Exception as exc:
        # Leave owner leases pinned if import or completion was uncertain.
        queue.put({"state": "error", "error": type(exc).__name__, "message": str(exc)})
        raise


def cuda_ipc_checks(ordinals):
    import multiprocessing
    import tempfile
    import time
    import torch
    from muxserve_sglang.ipc import StorageRPCClient, spawn_storage_owner
    assert os.name == "posix", "CUDA IPC checker requires Linux"
    assert os.environ.get("CUDA_VISIBLE_DEVICES"), "Set explicit CUDA_VISIBLE_DEVICES"
    assert torch.cuda.is_available(), "Real CUDA is required"
    assert len(set(ordinals)) == len(ordinals) and ordinals
    for ordinal in ordinals:
        assert 0 <= ordinal < torch.cuda.device_count()
        assert "A100" in torch.cuda.get_device_name(ordinal), "Initial acceptance target is A100"
    context = multiprocessing.get_context("spawn")
    directory = tempfile.mkdtemp(prefix="mx-kv-", dir="/tmp")
    os.chmod(directory, 0o700)
    address = os.path.join(directory, "owner.sock")
    authkey = os.urandom(32)
    ranks = tuple(range(len(ordinals)))
    owner = spawn_storage_owner({r: 8 for r in ranks}, dict(zip(ranks, ordinals)), address, authkey)
    print(f"CUDA owner PID={owner.pid}; socket={address}")
    deadline = time.monotonic() + 60
    while not os.path.exists(address):
        if not owner.is_alive():
            raise RuntimeError("CUDA owner exited during startup")
        if time.monotonic() >= deadline:
            raise RuntimeError("Owner startup deadline expired; inspect the printed owner PID")
        time.sleep(0.05)
    client = StorageRPCClient(address, authkey)
    client.register_model(ModelLayout("ipc-check", 1, 1, ranks))
    release = context.Event()
    writer_queue, reader_queue = context.Queue(), context.Queue()
    writer = context.Process(target=_cuda_consumer,
        args=(address, authkey, "acceptance-writer", ranks, True, release, writer_queue))
    writer.start()
    message = writer_queue.get(timeout=60)
    assert message["state"] == "ready", message
    reader = context.Process(target=_cuda_consumer,
        args=(address, authkey, "acceptance-reader", ranks, False, release, reader_queue))
    reader.start()
    message = reader_queue.get(timeout=60)
    assert message["state"] == "ready", message
    # Both consumers retain imports. Shutdown must fail closed and drain.
    expect_code("OWNER_BUSY", client.close_owner)
    release.set()
    for child, queue in ((writer, writer_queue), (reader, reader_queue)):
        message = queue.get(timeout=60)
        assert message["state"] == "done", message
        child.join(timeout=60)
        assert child.exitcode == 0, "Consumer did not exit cleanly; keep owner alive"
    client.assert_integrity()
    client.close_owner()
    owner.join(timeout=60)
    assert owner.exitcode == 0, "Owner did not drain and exit"
    os.rmdir(directory)
    print("Real CUDA IPC sharing and producer lifetime checks passed")


def allocation_cost_checks():
    """Reserving new blocks must not sort every free block per rank."""
    import muxserve_sglang.storage as storage_module
    store = HeadwiseStore({r: 129408 for r in range(4)}, model_ids=("m",))
    store.register_model(ModelLayout("m", 60, 13, (0, 1, 2, 3)))
    identity = RequestIdentity("m", "r")

    def forbidden(*_args, **_kwargs):
        raise AssertionError("reserve_kv sorted the free pool")

    storage_module.sorted = forbidden
    try:
        first = store.reserve_kv("p", [KVTarget(identity, 512)], "h")
    finally:
        del storage_module.sorted
    aid = first.views[0].allocation_id
    store.start_kv_batch(first.reservation_id)
    store.finish_kv_batch(first.reservation_id, "SUCCESS", {aid: 512}, (0, 1, 2, 3))
    storage_module.sorted = forbidden
    try:
        store.reserve_kv("d", [KVTarget(identity, 640, aid)], "h2")
    finally:
        del storage_module.sorted
    tables = store._allocations[aid].tables
    for rank in range(4):
        flat = [b for heads in tables[rank] for blocks in heads for b in blocks]
        assert len(flat) == 60 * 13 * 40 and len(set(flat)) == len(flat)
        assert not set(flat) & store._free[rank]
    store.assert_integrity()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cuda-ipc", action="store_true")
    parser.add_argument("--gpu-ordinals", default="0")
    args = parser.parse_args()
    allocator_checks()
    allocation_cost_checks()
    batch_history_budget_checks()
    static_pool_partition_checks()
    reconciliation_checks()
    rpc_worker_bound_checks()
    if os.name == "posix":
        rpc_transport_and_lifetime_checks()
    if args.cuda_ipc:
        cuda_ipc_checks(tuple(int(item) for item in args.gpu_ordinals.split(",")))


if __name__ == "__main__":
    main()
