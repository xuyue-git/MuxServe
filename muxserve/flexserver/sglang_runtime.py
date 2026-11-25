"""LLMRuntime (llm_runtime.py) for the SGLang engine.

The methods below are LLMRuntime's, line for line: the C++ BatchScheduler builds
every step, FlexStore allocates and frees its KV blocks (cache_alloc /
free_cache), rank 0 hands each step to the other TP ranks (shared memory for the
batch shape, torch.distributed.broadcast for the tensors), warmup is the same.
Only the engine parts differ:

  * LLMEngine / vLLM configs        -> plain fields from the same CLI arguments;
  * PipeWorker (vLLM model)         -> SGLangWorker (sglang_worker.py);
  * vLLM's TP group                 -> SGLang's TP group (same ranks, NCCL);
  * CacheEngine's "cache_init"      -> the same FlexStore request, here.

Pipeline parallelism (send/recv between stages) is not ported: every
configuration in this repository uses pipeline_parallel_size 1.
"""
from __future__ import annotations

import itertools
import os
import time
from types import SimpleNamespace
from typing import List, Optional

import numpy as np
import torch
from torch.multiprocessing.reductions import rebuild_cuda_tensor
from transformers import AutoConfig

from muxserve.batch_scheduler import BatchScheduler
from muxserve.constants import PREEMPT_REQ_NAME_FMT
from muxserve.flexserver.partition import pipeline_split
from muxserve.flexserver.sglang_worker import SGLangWorker
from muxserve.logger import get_logger
from muxserve.shm_utils import dump_to_shared_var, load_from_shared_var

logger = get_logger()


def _get_max_len(hf_config) -> int:
    """vLLM 0.2.0 config._get_and_verify_max_len with max_model_len=None."""
    derived_max_model_len = float("inf")
    possible_keys = ["max_position_embeddings", "n_positions", "max_seq_len",
                     "max_sequence_length", "max_seq_length", "seq_len"]
    for key in possible_keys:
        max_len_key = getattr(hf_config, key, None)
        if max_len_key is not None:
            derived_max_model_len = min(derived_max_model_len, max_len_key)
    if derived_max_model_len == float("inf"):
        derived_max_model_len = 2048
    rope_scaling = getattr(hf_config, "rope_scaling", None)
    # Newer transformers fill rope_scaling with {"rope_type": "default", ...} where the
    # transformers vLLM 0.2.0 ran with left it None: both mean no scaling.
    if isinstance(rope_scaling, dict) and rope_scaling.get(
            "rope_type", rope_scaling.get("type")) in (None, "default"):
        rope_scaling = None
    if rope_scaling is not None:
        assert "factor" in rope_scaling
        derived_max_model_len *= rope_scaling["factor"]
    return derived_max_model_len


def _get_num_kv_heads(hf_config, tensor_parallel_size: int) -> int:
    """vLLM 0.2.0 ModelConfig.get_num_kv_heads for the LLaMA family."""
    if getattr(hf_config, "num_key_value_heads", None) is not None:
        return hf_config.num_key_value_heads // tensor_parallel_size
    return hf_config.num_attention_heads // tensor_parallel_size


class SGLangRuntime:

    def __init__(self, model_id: int, model_name: str, model: str,
                 tp_rank: int, stage_id: int, tensor_parallel_size: int,
                 pipeline_parallel_size: int, block_size: int,
                 mps_percentage: int, max_num_seqs: int,
                 max_num_batched_tokens: int, local_rank: int,
                 tcp_client=None, runtime_profile: bool = False,
                 is_prefill: bool = False) -> None:
        logger.info("Initializing an SGLang LLM runtime with config: "
                    f"model_id={model_id}, model_name={model_name!r}, "
                    f"model={model!r}, "
                    f"tensor_parallel_size={tensor_parallel_size}")
        if pipeline_parallel_size != 1:
            raise RuntimeError("The SGLang engine runs pipeline_parallel_size 1")
        if tcp_client is None:
            raise RuntimeError("The SGLang engine runs with FlexStore only")
        if runtime_profile:
            raise RuntimeError("--runtime-profile is not ported to SGLang")

        self.model_id = model_id
        self.model_name = model_name
        self.is_prefill = is_prefill
        self.hf_config = AutoConfig.from_pretrained(model)
        self.scheduler_config = SimpleNamespace(
            max_num_seqs=max_num_seqs,
            max_num_batched_tokens=max_num_batched_tokens)

        self.tp_rank = tp_rank
        self.stage_id = stage_id
        self.num_stages = pipeline_parallel_size
        self.tensor_parallel_size = tensor_parallel_size
        self.world_size = pipeline_parallel_size * tensor_parallel_size
        self.global_steps = 0

        mps = mps_percentage
        self.preempt_req_shm_name = PREEMPT_REQ_NAME_FMT.format(
            self.model_name, mps)
        if self.tp_rank == 0:
            self.tp_share_name = [
                f"m{self.model_id}_mps{mps}_r{i + 1}"
                for i in range(self.tensor_parallel_size - 1)
            ]
        else:
            self.tp_share_name = f"m{self.model_id}_mps{mps}_r{self.tp_rank}"
        self.block_size = block_size

        # tcp client for muxserve (the weights, the KV pool and every
        # cache_alloc / free_cache go through FlexStore, as in the original)
        self.tcp_client = tcp_client
        self.is_muxserve = (self.tcp_client is not None)

        self._init_workers(model, tensor_parallel_size, local_rank)
        self.num_layers = self.workers[0].num_hidden_layers
        self.num_heads = _get_num_kv_heads(self.hf_config,
                                           tensor_parallel_size)

        # The KV pool FlexStore allocated on this GPU.
        self._init_cache()

        max_seq_len = _get_max_len(self.hf_config)
        num_layers = max(pipeline_split(self.hf_config.num_hidden_layers,
                                        pipeline_parallel_size))
        self.batch_scheduler = BatchScheduler(num_layers, self.num_heads,
                                              max_seq_len, self.block_size)

        # warmup engine
        if self.tcp_client is not None and self.stage_id == 0 and self.tp_rank == 0:
            while True:
                self.tcp_client.send_pyobj(["start_warmup", None])
                status = self.tcp_client.recv_pyobj()
                if status:
                    break
                time.sleep(3)

        logger.info("Warmup engine...")
        self.unique_id = itertools.count(int(os.environ["MASTER_PORT"]) * 100)
        for _ in range(5):
            self.warmup()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

        if self.tcp_client is not None and self.stage_id == 0 and self.tp_rank == 0:
            self.tcp_client.send_pyobj(["warmup_ready", None])
            self.tcp_client.recv_pyobj()
        logger.info("LLM Engine begins serving...")

    def _init_workers(self, model: str, tensor_parallel_size: int,
                      local_rank: int):
        self.workers = [
            SGLangWorker(model, self.model_name, tensor_parallel_size,
                         local_rank, self.tcp_client,
                         decode_graphs=not self.is_prefill,
                         prefill_graphs=self.is_prefill)
        ]
        self.tp_group = self.workers[0].tp_group

    def _init_cache(self):
        """CacheEngine.get_gpu_cache_from_flexstore (MuxServe-vLLM)."""
        rank = torch.distributed.get_rank()
        self.tcp_client.send_pyobj(["cache_init", [rank, self.model_name]])
        kv_blocks = self.tcp_client.recv_pyobj()
        key_blocks = rebuild_cuda_tensor(torch.Tensor, **(kv_blocks[0]))
        value_blocks = rebuild_cuda_tensor(torch.Tensor, **(kv_blocks[1]))
        assert key_blocks.is_cuda
        assert value_blocks.is_cuda
        self.gpu_cache = (key_blocks, value_blocks)
        self.num_gpu_blocks = key_blocks.shape[0]
        self.workers[0].set_cache(key_blocks, value_blocks)
        logger.info(f"# GPU blocks: {self.num_gpu_blocks} from FlexStore")

    def pipeline_step(
        self,
        batch_request_ids: Optional[List[int]] = None,
        batch_output_tokens: Optional[List[int]] = None
    ):
        preempted_reqs = []
        if self.tp_rank == 0:
            batch = self.prepare_inputs(batch_request_ids,
                                        batch_output_tokens)
            (input_tensor, input_positions, input_metadata, batch_reqs,
             preempted_reqs) = batch
            logger.info(
                f"Step {self.global_steps} BS {len(batch_reqs)} scheduled "
                f"{input_tensor.size()} tokens, "
                f"requests: {batch_request_ids}, preempted: {preempted_reqs}"
            )

            if len(batch_reqs) == 0:
                return [], batch_reqs, batch_request_ids

            self.scatter_input(input_tensor, input_positions,
                               input_metadata, batch_reqs)
        else:
            batch = self.gather_input()
            input_tensor, input_positions, input_metadata, batch_reqs = batch

        output_tensor = self.workers[0].pure_forward(
            input_tensor=input_tensor,
            positions=input_positions,
            input_metadata=input_metadata)
        self.global_steps += 1

        torch.cuda.synchronize()
        return output_tensor, batch_reqs, preempted_reqs

    def prepare_inputs(self, batch_request_ids: List[int],
                       batch_output_tokens: List[int]):
        batch_block_request_info = self.batch_scheduler.try_batch(
            batch_request_ids, batch_output_tokens)
        total_blocks_needed = batch_block_request_info[-1]
        batch_block_request = batch_block_request_info[:-1]
        if total_blocks_needed > 0:
            self.tcp_client.send_pyobj(
                ["cache_alloc", [self.model_name, batch_block_request]])
            block_info = self.tcp_client.recv_pyobj()
        else:
            block_info = np.array([len(batch_request_ids)], dtype=np.int32)

        batch_info = self.batch_scheduler.get_batch_info(
            batch_block_request, block_info)
        num_tokens = batch_info[0]
        num_contexts = batch_info[1]
        max_context_len = batch_info[2]
        max_num_blocks_per_seq = batch_info[3]
        num_layers = batch_info[4]
        num_heads = batch_info[5]
        prompt_lens = batch_info[6:]

        padded_len = (num_tokens + 7) // 8 * 8
        slot_mapping = torch.empty((num_tokens, num_layers, num_heads),
                                   dtype=torch.int64)
        if max_context_len > 0:
            block_tables = torch.empty(
                (num_tokens, max_num_blocks_per_seq, num_layers, num_heads),
                dtype=torch.int32)
        else:
            block_tables = torch.empty((0, max_num_blocks_per_seq, 0, 0),
                                       dtype=torch.int32)
        input_tensor = torch.empty((padded_len, ), dtype=torch.int32)
        input_positions = torch.empty((padded_len, ), dtype=torch.int32)
        context_lens = torch.empty((num_contexts, ), dtype=torch.int32)

        self.batch_scheduler.get_batch(batch_block_request, block_info,
                                       input_tensor, input_positions,
                                       context_lens, block_tables,
                                       slot_mapping)

        preempted_reqs = self.batch_scheduler.get_preempt_requests()
        batch_reqs = self.batch_scheduler.get_batch_reqs()

        slot_mapping = slot_mapping.cuda(non_blocking=True)
        block_tables = block_tables.cuda(non_blocking=True)
        input_tensor = input_tensor.cuda(non_blocking=True).long()
        input_positions = input_positions.cuda(non_blocking=True).long()
        context_lens = context_lens.cuda(non_blocking=True)
        block_tables = block_tables.permute(2, 0, 3, 1).contiguous()

        input_metadata = self.workers[0]._prepare_inputs_metadata(
            prompt_lens=prompt_lens,
            slot_mapping=slot_mapping,
            context_lens=context_lens,
            max_context_len=max_context_len,
            block_tables=block_tables)

        if preempted_reqs:
            dump_to_shared_var(self.preempt_req_shm_name, preempted_reqs)

        return input_tensor, input_positions, input_metadata, batch_reqs, preempted_reqs

    def scatter_input(self, input_tensor: torch.Tensor,
                      positions: torch.Tensor, input_metadata,
                      batch_reqs: List[List[int]]) -> None:
        # walkaround for cuda synchronization
        num_tokens = input_metadata.num_valid_tokens
        num_generation_tokens = input_metadata.num_generation_tokens
        num_prompts = input_metadata.num_prompts
        max_context_len = input_metadata.max_context_len
        prompt_lens = input_metadata.prompt_lens

        data = [
            num_tokens, num_generation_tokens, num_prompts, max_context_len
        ]
        data += prompt_lens
        for req in batch_reqs:
            data += req

        src = self.tp_group.first_rank
        group = self.tp_group.device_group
        if self.world_size > 8:
            batch_info = torch.tensor(data,
                                      dtype=torch.int32,
                                      device=torch.cuda.current_device())
            shape_tensor = torch.tensor([len(data)],
                                        dtype=torch.int32,
                                        device=torch.cuda.current_device())
            torch.distributed.broadcast(shape_tensor, src=src, group=group)
            torch.distributed.broadcast(batch_info, src=src, group=group)
        else:
            for shm_name in self.tp_share_name:
                while True:
                    try:
                        dump_to_shared_var(shm_name, data)
                        break
                    except FileExistsError:
                        time.sleep(1 / 5e4)

        torch.distributed.broadcast(input_tensor, src=src, group=group)
        torch.distributed.broadcast(positions, src=src, group=group)
        torch.distributed.broadcast(input_metadata.slot_mapping,
                                    src=src,
                                    group=group)
        if input_metadata.num_generation_tokens > 0:
            torch.distributed.broadcast(input_metadata.context_lens,
                                        src=src,
                                        group=group)
            torch.distributed.broadcast(input_metadata.block_tables,
                                        src=src,
                                        group=group)

    def gather_input(self):
        # walkaround for cuda synchronization
        src = self.tp_group.first_rank
        group = self.tp_group.device_group
        if self.world_size > 8:
            shape_tensor = torch.empty((1, ),
                                       dtype=torch.int32,
                                       device=torch.cuda.current_device())
            torch.distributed.broadcast(shape_tensor, src=src, group=group)
            batch_info = torch.empty((shape_tensor[0], ),
                                     dtype=torch.int32,
                                     device=torch.cuda.current_device())
            torch.distributed.broadcast(batch_info, src=src, group=group)
            data = batch_info.tolist()
        else:
            while True:
                data = load_from_shared_var(self.tp_share_name)
                if data:
                    break
                time.sleep(1 / 5e4)
        num_tokens = data.pop(0)
        num_generation_tokens = data.pop(0)
        num_prompts = data.pop(0)
        max_context_len = data.pop(0)
        prompt_lens = data[:num_prompts]
        batch_reqs = [[data[i], data[i + 1]]
                      for i in range(num_prompts, len(data), 2)]

        num_requests = len(batch_reqs)
        num_blocks = (max_context_len + self.block_size - 1) // self.block_size

        multiple_of = 8
        padded_num_tokens = (num_tokens + multiple_of -
                             1) // multiple_of * multiple_of
        input_tensor_shape = (padded_num_tokens, )
        positions_shape = (padded_num_tokens, )
        context_lens_shape = (num_generation_tokens, )
        if self.is_muxserve:
            slot_mapping_shape = (num_tokens, self.num_layers, self.num_heads)
            block_tables_shape = (self.num_layers, num_generation_tokens,
                                  self.num_heads, num_blocks)
        else:
            slot_mapping_shape = (num_tokens, )
            block_tables_shape = (num_generation_tokens, num_blocks)

        device = torch.cuda.current_device()
        input_tensor = torch.empty(input_tensor_shape,
                                   dtype=torch.int64,
                                   device=device)
        positions = torch.empty(positions_shape,
                                dtype=torch.int64,
                                device=device)
        slot_mapping = torch.empty(slot_mapping_shape,
                                   dtype=torch.int64,
                                   device=device)
        context_lens = torch.empty(context_lens_shape,
                                   dtype=torch.int,
                                   device=device)
        block_tables = torch.empty(block_tables_shape,
                                   dtype=torch.int,
                                   device=device)

        torch.distributed.broadcast(input_tensor, src=src, group=group)
        torch.distributed.broadcast(positions, src=src, group=group)
        torch.distributed.broadcast(slot_mapping, src=src, group=group)
        if max_context_len > 0:
            torch.distributed.broadcast(context_lens, src=src, group=group)
            torch.distributed.broadcast(block_tables, src=src, group=group)

        input_metadata = self.workers[0]._prepare_inputs_metadata(
            prompt_lens=prompt_lens,
            slot_mapping=slot_mapping,
            context_lens=context_lens,
            max_context_len=max_context_len,
            block_tables=block_tables)

        return input_tensor, positions, input_metadata, batch_reqs

    def release_request(self, free_request_ids: List[int],
                        finished_request_ids: List[int]):
        if finished_request_ids:
            self.tcp_client.send_pyobj(
                ["free_cache", [self.model_name, finished_request_ids]])
        self.batch_scheduler.release_requests(free_request_ids)
        if finished_request_ids:
            _ = self.tcp_client.recv_pyobj()

    def warmup(self) -> None:
        """Warms up the engine and cuda context."""
        prompt_tokens = [100] * 50
        max_tokens = 5
        batch_request_ids, batch_output_tokens = [], []
        if self.stage_id == 0 and self.tp_rank == 0:
            for i in range(16):
                req_id = next(self.unique_id)
                batch_request_ids.append(req_id)
                self.batch_scheduler.add_request(prompt_tokens, req_id,
                                                 max_tokens)

        generated_len = {}
        for step in range(5):
            if self.stage_id == 0:
                outputs, batch_reqs, _ = self.pipeline_step(
                    batch_request_ids, batch_output_tokens)

                if self.tp_rank == 0:
                    free_request_ids = []
                    batch_request_ids = []
                    batch_output_tokens = []
                    for (req_id, max_token) in batch_reqs:
                        if req_id not in generated_len:
                            generated_len[req_id] = 0
                        generated_len[req_id] += 1
                        if generated_len[req_id] == max_token:
                            free_request_ids.append(req_id)
                        else:
                            batch_request_ids.append(req_id)
                            batch_output_tokens.append(20)  # a random token

                    if free_request_ids:
                        self.release_request(free_request_ids,
                                             free_request_ids)
            else:
                outputs, batch_reqs, _ = self.pipeline_step()

        # wait for all requests to finish
        torch.distributed.all_reduce(torch.zeros(1).cuda(),
                                     group=self.tp_group.device_group)
        torch.cuda.synchronize()
