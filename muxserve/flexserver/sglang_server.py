"""P/D rank entry point with the SGLang engine (MUXSERVE_ENGINE=sglang).

muxsched/launcher.py starts this file exactly as it starts muxserve_server.py:
torch.distributed.launch, the same command-line arguments, the same MPS
percentage. main() does what muxserve_server.main does -- ask FlexStore for the
rank, build the runtime, then run MuxServeEngine (imported unchanged from
muxserve_server.py) -- with SGLangRuntime in place of the vLLM LLMRuntime.
"""
import argparse
import os

from muxserve.flexserver.muxserve_server import MuxServeEngine
from muxserve.flexserver.sglang_runtime import SGLangRuntime
from muxserve.logger import init_logger
from muxserve.zmq_utils import ZMQClient

logger = init_logger("MuxServe")


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
                                     args.mps_percentage, args.is_prefill)
    # Start the engine loop.
    muxserve_engine.serve()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='MuxServe Runtime Server (SGLang engine)')
    parser.add_argument("--model-id", type=int, default=0,
                        help="The index of served model.")
    parser.add_argument("--local-rank", "--local_rank", type=int)
    parser.add_argument('--runtime-profile', action='store_true')
    # The arguments muxsched/launcher.py passes (vLLM EngineArgs names).
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
