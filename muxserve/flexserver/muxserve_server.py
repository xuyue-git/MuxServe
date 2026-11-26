import os
import argparse
import asyncio
import time
import numpy as np
from typing import Dict, Set

import torch

from muxserve.zmq_utils import ZMQClient

from typing import List, Optional, Union

from muxserve.constants import (SM_HOLD_NAME_FMT, ADD_REQ_NAME_FMT,
                              RET_REQ_NAME_FMT, PREEMPT_REQ_NAME_FMT)
from muxserve.flexserver.sglang_runtime import SGLangRuntime
from muxserve.muxsched.workload_utils import Workload, Request
from muxserve.shm_utils import (create_shared_var, read_shared_var,
                              write_shared_var, dump_to_shared_var,
                              load_from_shared_var, load_reqs_from_shared_var,
                              dump_reqs_to_shared_var)
from muxserve.logger import init_logger

logger = init_logger("MuxServe")


class MuxServeEngine:

    def __init__(self, llm_runtime: SGLangRuntime, workload: Workload,
                 mps_percentage: int, is_prefill: bool):
        self.llm_runtime = llm_runtime
        self.workload = workload
        self.mps_percentage = mps_percentage
        self.is_prefill = is_prefill
        self.model_name = self.llm_runtime.model_name

        self.rank = int(os.environ["RANK"])
        self.local_rank = int(os.environ["LOCAL_RANK"])
        self.world_size = int(os.environ["WORLD_SIZE"])

        self.lock = asyncio.Lock()

        # Create shared variables.
        if self.llm_runtime.tcp_client is not None:
            sm_hold_shm_name = SM_HOLD_NAME_FMT.format(self.model_name,
                                                       self.mps_percentage)
            self.sm_hold_shm = create_shared_var(sm_hold_shm_name,
                                                 create=False)
            self.add_req_shm_name = ADD_REQ_NAME_FMT.format(
                self.model_name, self.mps_percentage)
            self.ret_req_shm_name = RET_REQ_NAME_FMT.format(
                self.model_name, self.mps_percentage)
            self.preempt_req_shm_name = PREEMPT_REQ_NAME_FMT.format(
                self.model_name, self.mps_percentage)
        self.requests_in_queue: Set[int] = set()
        self.requests_max_tokens: Dict[int, int] = {}

        self.requests: Dict[int, Request] = {}

        self.enable_profiler = False
        self.start_batch = 100
        self.stop_batch = 120 if self.is_prefill else 150
        self.cur_batch = 0
        self.prof = torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA
            ],
            with_stack=True,
            with_modules=True) if self.enable_profiler else None
        model = self.model_name.split("/")[-1]
        self.prof_out_name = f"log/profiler_muxserve/profiler_{model}_mps{self.mps_percentage}_w{self.world_size}_rank{self.rank}.json"

    def add_requests(self):
        while True:
            batch_reqs = load_reqs_from_shared_var(self.add_req_shm_name)
            if batch_reqs:
                break
        # if not batch_reqs:
        #     return None, None

        if self.is_prefill:
            num_requests = len(batch_reqs)
        else:
            num_requests = len(batch_reqs) // 2

        batch_request_ids = []
        for req in batch_reqs[:num_requests]:
            if isinstance(req, Request):
                self.requests[req.idx] = req
                batch_request_ids.append(req.idx)
            else:
                batch_request_ids.append(req)

        for i in range(num_requests):
            req_id = batch_request_ids[i]
            req = self.requests[req_id]
            prompt_tokens = req.data[0]
            output_len = 1 if self.is_prefill else req.data[2]
            max_tokens = req.data[2]

            if self.is_prefill:
                self.llm_runtime.batch_scheduler.add_request(
                    prompt_tokens, req_id, output_len)
                self.requests_max_tokens[req_id] = max_tokens
            else:
                if req_id not in self.requests_in_queue:
                    self.llm_runtime.batch_scheduler.add_request(
                        prompt_tokens, req_id, output_len)
                    self.requests_in_queue.add(req_id)
                    self.requests_max_tokens[req_id] = max_tokens
                    # reset output idx for new requests
                    self.requests[req_id].output_idx = 0

                self.requests[req_id].output_idx += 1

        batch_output_tokens = batch_reqs[num_requests:]
        return batch_request_ids, batch_output_tokens

    def process_output(self, outputs: List[int],
                       batch_reqs: List[List[int]]) -> None:
        # logger.info(
        #     f"Final stage outputs: {outputs}, batch_reqs: {batch_reqs}")
        generated = [req[0] for req in batch_reqs] + outputs
        while True:
            try:
                dump_to_shared_var(self.ret_req_shm_name, generated)
                break
            except FileExistsError:
                time.sleep(1 / 5e4)

    def first_stage_loop(self):
        while True:
            num_iters = read_shared_var(self.sm_hold_shm)
            if num_iters != 1:
                time.sleep(0.00005)
                continue

            exec_bs = []
            while num_iters > 0:
                batch_request_ids, batch_output_tokens = self.add_requests()

                outputs, batch_reqs, preempted_reqs = self.llm_runtime.pipeline_step(
                    batch_request_ids, batch_output_tokens)
                exec_bs.append(len(batch_reqs))
                if len(batch_reqs) > 0:
                    self.cur_batch += 1

                    if self.cur_batch % 200 == 0:
                        self.memory_stats()

                if self.llm_runtime.stage_id == self.llm_runtime.num_stages - 1:
                    if len(batch_reqs) > 0:
                        self.process_output(outputs, batch_reqs)

                # free kv cache blocks for finished requests
                free_request_ids, finished_request_ids = [], []
                for (req_id, max_token) in batch_reqs:
                    req = self.requests[req_id]
                    # req = self.llm_runtime.batch_scheduler._requests[] # need to move the definition of request from python to cpp
                    if req.output_idx == max_token - 1:
                        free_request_ids.append(req_id)
                        if req.output_idx == req.data[2] - 1:
                            finished_request_ids.append(req_id)
                self.requests_in_queue.difference_update(preempted_reqs)
                if free_request_ids:
                    self.requests_in_queue.difference_update(free_request_ids)
                    self.llm_runtime.release_request(free_request_ids,
                                                     finished_request_ids)
                    # logger.info(
                    #     f"Requests {free_request_ids} free, {finished_request_ids} finished, "
                    #     f"KV cache are freed or stored!")

                num_iters -= 1

                if self.enable_profiler:
                    if self.cur_batch == self.start_batch:
                        self.prof.start()
                    elif self.cur_batch == self.stop_batch:
                        self.prof.stop()
                        logger.info(
                            f"Export profiler trace to {self.prof_out_name}")
                        self.prof.export_chrome_trace(self.prof_out_name)
                        self.enable_profiler = False

            avg_bs = int(np.mean(exec_bs)) if exec_bs else 0
            write_shared_var(self.sm_hold_shm, -avg_bs)

    def other_stage_loop(self):
        exec_bs = []
        while True:
            outputs, batch_reqs, _ = self.llm_runtime.pipeline_step()
            exec_bs.append(len(batch_reqs))
            if len(batch_reqs) > 0:
                self.cur_batch += 1

                if self.cur_batch % 200 == 0:
                    self.memory_stats()

            if self.llm_runtime.stage_id == self.llm_runtime.num_stages - 1 and self.llm_runtime.tp_rank == 0:
                if len(batch_reqs) > 0:
                    self.process_output(outputs, batch_reqs)

            if self.enable_profiler:
                if self.cur_batch == self.start_batch:
                    self.prof.start()
                elif self.cur_batch == self.stop_batch:
                    self.prof.stop()
                    logger.info(
                        f"Export profiler trace to {self.prof_out_name}")
                    self.prof.export_chrome_trace(self.prof_out_name)
                    self.enable_profiler = False

    def serve(self):
        logger.info(
            f"MuxServe engine started (MPS: {self.mps_percentage})! "
            f"max_num_seqs: {self.llm_runtime.scheduler_config.max_num_seqs} "
            f"max_batched_token: "
            f"{self.llm_runtime.scheduler_config.max_num_batched_tokens}")
        self.memory_stats()
        if self.llm_runtime.stage_id == 0 and self.llm_runtime.tp_rank == 0:
            self.first_stage_loop()
        else:
            self.other_stage_loop()

    def memory_stats(self):
        max_allocated_memory = torch.cuda.max_memory_allocated() / 1024**3
        allocated_memory = torch.cuda.memory_allocated() / 1024**3
        reserved_memory = torch.cuda.memory_reserved() / 1024**3
        max_reserved_memory = torch.cuda.max_memory_reserved() / 1024**3
        logger.info(f"Rank {self.rank} Memory Stats: "
                    f"Allocated {allocated_memory:.2f} GB, "
                    f"Max Allocated {max_allocated_memory:.2f} GB, "
                    f"Reserved {reserved_memory:.2f} GB "
                    f"Max Reserved {max_reserved_memory:.2f} GB")


def main(args: argparse.Namespace):
    # build zmq client
    tcp_client = ZMQClient('localhost', args.flexstore_port)
    local_rank = int(os.getenv("LOCAL_RANK", "0"))
    tcp_client.send_pyobj(["get_rank", local_rank])
    rank = tcp_client.recv_pyobj()
    os.environ["RANK"] = str(rank)
    if args.pipeline_parallel_size != 1:
        raise SystemExit("The SGLang engine runs pipeline_parallel_size 1")

    llm_runtime = SGLangRuntime(
        model_id=args.model_id,
        model_name=args.model_name,
        model=args.model,
        tp_rank=rank % args.tensor_parallel_size,
        stage_id=0,
        tensor_parallel_size=args.tensor_parallel_size,
        pipeline_parallel_size=args.pipeline_parallel_size,
        block_size=args.block_size,
        mps_percentage=args.mps_percentage,
        max_num_seqs=args.max_num_seqs,
        max_num_batched_tokens=args.max_num_batched_tokens,
        local_rank=local_rank,
        tcp_client=tcp_client,
        runtime_profile=args.runtime_profile,
        is_prefill=args.is_prefill)

    workload = None
    muxserve_engine = MuxServeEngine(llm_runtime, workload,
                                 args.mps_percentage,
                                 args.is_prefill)
    # Start the engine loop.
    muxserve_engine.serve()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='MuxServe Runtime Server')
    parser.add_argument("--model-id",
                        type=int,
                        default=0,
                        help="The index of served model.")
    parser.add_argument("--local-rank", "--local_rank", type=int)
    parser.add_argument('--runtime-profile', action='store_true')
    # The arguments muxsched/launcher.py passes.
    parser.add_argument("--model-name", type=str, required=True)
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--pipeline-parallel-size", type=int, default=1)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--swap-space", type=float, default=4)
    parser.add_argument("--max-num-batched-tokens", type=int, default=2048)
    parser.add_argument("--max-num-seqs", type=int, default=256)
    parser.add_argument("--flexstore-port", type=int, required=True)
    parser.add_argument("--workload-file", type=str, default=None)
    parser.add_argument("--mps-percentage", type=int, required=True)
    parser.add_argument("--is-prefill", action="store_true")
    parser.add_argument("--split-by-model", type=str, default=None)
    args = parser.parse_args()

    if args.local_rank is not None:
        os.environ["LOCAL_RANK"] = str(args.local_rank)

    main(args)
