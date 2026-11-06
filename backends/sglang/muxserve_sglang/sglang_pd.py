"""Native SGLang PD adapters over owner-committed shared Headwise KV.

This transport registers logical destinations and moves only greedy CPU
metadata. FlexStore alone owns physical blocks and CUDA IPC import leases.
"""
from __future__ import annotations

from copy import deepcopy
import os
from types import SimpleNamespace

import torch
from sglang.srt.disaggregation.base.conn import (
    BaseKVBootstrapServer, BaseKVManager, BaseKVReceiver, BaseKVSender,
    KVArgs, KVPoll, KVTransferMetric,
)
from sglang.srt.disaggregation.utils import KVClassType

from .contracts import IntegrationError
from .sglang_hooks import context


def require(ok, message):
    if not ok:
        raise IntegrationError("SHARED_PD_MISMATCH", message)


class SharedHeadwiseKVManager(BaseKVManager):
    def __init__(self, args, disaggregation_mode, server_args, is_mla_backend=False):
        self.ctx = context()
        require(not is_mla_backend and args.pp_rank == 0 and args.system_dp_rank == 0
                and args.page_size == 1 and len(args.kv_data_ptrs) == 2
                and not args.kv_layer_ids and not args.state_types and not args.num_draft_entries,
                "Shared PD requires global K/V spans, MHA, TP/PP1/DP1 and no auxiliary state")
        self.kv_args = args
        self.rank, self.pid = args.engine_rank, os.getpid()
        self.prefill_info_table = {}
        self.transfer_infos = {}
        self.descriptors = {}
        self.metadata_buffers = None
        self.is_dummy_cp_rank = False
        self.try_ensure_parallel_info(f"127.0.0.1:{self.ctx.model.bootstrap_port}")

    def attach_metadata_buffers(self, buffers):
        require(not buffers.enable_sampling_mask and buffers.output_ids.device.type == "cpu"
                and buffers.output_dsa_topk_indices is None,
                "Greedy shared PD uses native CPU metadata buffers")
        self.metadata_buffers = buffers

    def register_to_bootstrap(self):
        # Topology is frozen by the execution group; readiness is room-specific.
        return None

    def try_ensure_parallel_info(self, bootstrap_addr):
        require(bootstrap_addr == f"127.0.0.1:{self.ctx.model.bootstrap_port}",
                "Only the configured same-node bootstrap address is supported")
        self.prefill_info_table[bootstrap_addr] = SimpleNamespace(
            dp_size=1, tp_size=self.ctx.config.tp_size, pp_size=1, follow_bootstrap_room=False)
        return True

    def bind_request(self, item):
        room = item["bootstrap_room"]
        require(type(room) is int and room > 0, "A coordinator room is required")
        previous = self.descriptors.get(room)
        require(previous is None or previous == item, "Room reservation replay changed")
        require(previous is not None or len(self.descriptors) < self.ctx.config.max_model_inflight,
                "Native room state limit reached")
        self.descriptors[room] = deepcopy(item)

    def descriptor(self, room):
        require(room in self.descriptors, "Room has no selected coordinator reservation")
        return self.descriptors[room]

    def rpc(self, name, *args):
        return self.ctx.rpc.call("controller_" + name, self.ctx.group_id, *args)

    def sync(self):
        torch.cuda.synchronize(self.kv_args.gpu_id)

    def forget(self, room):
        self.descriptors.pop(room, None)
        self.transfer_infos.pop(room, None)


class _Room:
    def _room_init(self, mgr, room):
        self.kv_mgr, self.bootstrap_room = mgr, room
        self.item = mgr.descriptor(room)
        self.aborted = self.cleared = False

    def failure_exception(self):
        raise IntegrationError("SHARED_PD_ABORTED", "Native shared transfer was aborted or fenced")

    def abort(self):
        # This only aborts local native state. Owner commands retain the physical
        # holders until every rank proves genuine synchronized quiescence.
        self.aborted = True

    def clear(self):
        self.cleared = True

    def _status(self, stage):
        if self.aborted:
            return KVPoll.Failed
        reply = self.kv_mgr.rpc("broker_poll", self.bootstrap_room,
                                self.kv_mgr.rank, self.kv_mgr.pid, stage)
        return {"ABORTED": KVPoll.Failed, "PENDING": KVPoll.Transferring,
                "SUCCESS": KVPoll.Success}[reply["status"]]


class SharedHeadwiseKVSender(_Room, BaseKVSender):
    def __init__(self, mgr, bootstrap_addr, bootstrap_room, dest_tp_ranks, pp_rank,
                 req_has_disagg_prefill_dp_rank=False):
        require(mgr.ctx.stage == "P" and pp_rank == 0 and dest_tp_ranks == [mgr.rank],
                "Shared sender must match the local P rank")
        mgr.try_ensure_parallel_info(bootstrap_addr)
        self._room_init(mgr, bootstrap_room)
        self.aux_index = None
        self.sent = False

    def poll(self):
        if self.aborted:
            return KVPoll.Failed
        if self.sent:
            return self._status("P")
        view = self.item["reservation_view"]
        reply = self.kv_mgr.rpc("broker_bootstrap", self.bootstrap_room,
                                self.kv_mgr.rank, self.kv_mgr.pid,
                                view["allocation_id"], view["version"],
                                self.item["target_valid_tokens"])
        return {"READY": KVPoll.WaitingForInput, "PENDING": KVPoll.Bootstrapping,
                "ABORTED": KVPoll.Failed}[reply["status"]]

    def init(self, num_kv_indices, aux_index=None):
        require(num_kv_indices == self.item["target_valid_tokens"]
                and type(aux_index) is int and aux_index >= 0,
                "Native sender length/metadata slot mismatch")
        require(self.aux_index is None or self.aux_index == aux_index, "Sender slot replay changed")
        self.aux_index = aux_index

    def send(self, kv_indices, state_indices=None, num_kv_tokens=None):
        require(not self.aborted and self.aux_index is not None and not state_indices,
                "Shared sender is not initialized or has unsupported auxiliary state")
        require(len(kv_indices) == self.item["target_valid_tokens"],
                "Only a complete selected prefill/rebuild transfer is supported")
        require(num_kv_tokens in (None, self.item["target_valid_tokens"]), "KV token count mismatch")
        self.kv_mgr.sync()
        buffers = self.kv_mgr.metadata_buffers
        require(buffers is not None, "Native metadata buffers are missing")
        require(int(buffers.bootstrap_room[self.aux_index, 0]) == self.bootstrap_room,
                "Native metadata room mismatch")
        # The supported greedy subset contains no tensor addresses, logits,
        # hidden states, physical K/V bytes, or large full-buffer copies.
        payload = {"output_id": int(buffers.output_ids[self.aux_index, 0]),
                   "cached_tokens": buffers.cached_tokens[self.aux_index].tolist(),
                   "bootstrap_room": self.bootstrap_room}
        view = self.item["reservation_view"]
        self.kv_mgr.rpc("broker_stage", self.bootstrap_room, self.kv_mgr.rank,
                       self.kv_mgr.pid, {"bootstrap_room": self.bootstrap_room,
                       "allocation_id": view["allocation_id"], "reserved_version": view["version"],
                       "target_valid_tokens": self.item["target_valid_tokens"],
                       "rank": self.kv_mgr.rank, "native_payload": payload})
        self.sent = True

    def get_transfer_metric(self):
        return KVTransferMetric(transfer_total_bytes=0)


class SharedHeadwiseKVReceiver(_Room, BaseKVReceiver):
    def __init__(self, mgr, bootstrap_addr, bootstrap_room=None):
        require(mgr.ctx.stage == "D", "Shared receiver must be a D rank")
        mgr.try_ensure_parallel_info(bootstrap_addr)
        self._room_init(mgr, bootstrap_room)
        self.inited = self.registered = self.validated = False
        self.aux_index = None
        self.logical_indices = None

    def init(self, prefill_dp_rank):
        require(prefill_dp_rank == 0, "Shared transport supports prefill DP rank zero")
        self.inited = True

    def send_metadata(self, kv_indices, aux_index=None, state_indices=None,
                      decode_prefix_len=None):
        require(self.inited and not self.aborted and not state_indices
                and decode_prefix_len in (None, 0),
                "Unsupported native shared destination")
        indices = tuple(int(i) for i in kv_indices)
        require(len(indices) == self.item["target_valid_tokens"] and len(set(indices)) == len(indices)
                and all(i > 0 for i in indices) and type(aux_index) is int and aux_index >= 0,
                "Native logical preallocation/metadata slot mismatch")
        require(not self.registered or (self.aux_index == aux_index and self.logical_indices == indices),
                "Receiver destination replay changed")
        view = self.item["reservation_view"]
        self.kv_mgr.rpc("broker_register_destination", self.bootstrap_room, self.kv_mgr.rank,
                       self.kv_mgr.pid, self.kv_mgr.ctx.model.model_id,
                       self.item["request_id"], self.item["attempt"], view["allocation_id"],
                       view["version"], self.item["target_valid_tokens"],
                       self.item["rank_block_table"], aux_index, len(indices))
        self.aux_index, self.logical_indices, self.registered = aux_index, indices, True

    def accept_handoff(self, handoff, pool):
        require(self.registered and not self.aborted, "D destination is not registered")
        view = self.item["reservation_view"]
        require(handoff["request_id"] == self.item["request_id"]
                and handoff["attempt"] == self.item["attempt"]
                and handoff["bootstrap_room"] == self.bootstrap_room
                and handoff["allocation_id"] == view["allocation_id"]
                and handoff["version"] == view["version"] + 1
                and handoff["valid_tokens"] == self.item["target_valid_tokens"]
                and handoff["rank_block_table"] == self.item["rank_block_table"]
                and pool.rank == self.kv_mgr.rank, "Committed handoff differs from imported reservation")
        reply = self.kv_mgr.rpc("broker_lookup", self.bootstrap_room, self.kv_mgr.rank,
                               self.kv_mgr.pid, self.kv_mgr.ctx.model.model_id,
                               handoff["request_id"], handoff["attempt"], handoff["allocation_id"],
                               handoff["version"], handoff["valid_tokens"])
        require(reply["rank_block_table"] == handoff["rank_block_table"], "Broker mapping mismatch")
        payload = reply["metadata"]["native_payload"]
        require(set(payload) == {"output_id", "cached_tokens", "bootstrap_room"}
                and payload["bootstrap_room"] == self.bootstrap_room
                and payload["output_id"] == handoff["boundary_token_id"]
                and len(payload["cached_tokens"]) == 16, "Native greedy boundary metadata mismatch")
        buffers = self.kv_mgr.metadata_buffers
        buffers.output_ids[self.aux_index].zero_()
        buffers.output_ids[self.aux_index, 0] = payload["output_id"]
        buffers.cached_tokens[self.aux_index].copy_(torch.tensor(payload["cached_tokens"], dtype=torch.int32))
        buffers.bootstrap_room[self.aux_index].zero_()
        buffers.bootstrap_room[self.aux_index, 0] = self.bootstrap_room
        self.kv_mgr.rpc("broker_ack", self.bootstrap_room, self.kv_mgr.rank, self.kv_mgr.pid,
                       handoff["allocation_id"], handoff["version"], handoff["valid_tokens"],
                       handoff["rank_block_table"])
        self.validated = True

    def poll(self):
        if self.aborted:
            return KVPoll.Failed
        if not self.inited:
            return KVPoll.Bootstrapping
        if not self.registered:
            # Logical D admission must happen before P bootstrap readiness.
            return KVPoll.WaitingForInput
        if not self.validated:
            return KVPoll.Transferring
        return self._status("D")


class SharedHeadwiseKVBootstrapServer(BaseKVBootstrapServer):
    def __init__(self, host, port):
        ctx = context()
        require(host in ("127.0.0.1", "localhost") and port == ctx.model.bootstrap_port,
                "Shared bootstrap uses only the configured local coordinator")
        self.host, self.port = host, port
        # Native Python disagg_service retains this object. No network listener
        # or RDMA registration is needed: authenticated owner RPC is the registry.


def shared_kv_class(class_type):
    return {KVClassType.KVARGS: KVArgs, KVClassType.MANAGER: SharedHeadwiseKVManager,
            KVClassType.SENDER: SharedHeadwiseKVSender, KVClassType.RECEIVER: SharedHeadwiseKVReceiver,
            KVClassType.BOOTSTRAP_SERVER: SharedHeadwiseKVBootstrapServer}[class_type]
