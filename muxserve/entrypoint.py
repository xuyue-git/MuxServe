import os
import argparse
import asyncio

from muxserve.arg_utils import MuxServeArgs
from muxserve.config import MuxServeConfig
from muxserve.flexstore.manager import FlexStoreManager
from muxserve.muxsched.scheduler import MuxScheduler
from muxserve.logger import get_logger

logger = get_logger()


def main_flexstore(muxserve_config: MuxServeConfig):
    flexstore_manager = FlexStoreManager(muxserve_config)
    flexstore_manager.deploy()


def main_muxsched(muxserve_config: MuxServeConfig, http_port=None):
    if http_port is None:
        muxscheduler = MuxScheduler(muxserve_config)
    else:  # requests arrive over HTTP instead of from the workload file
        from muxserve.muxsched.http_scheduler import HttpMuxScheduler
        muxscheduler = HttpMuxScheduler(muxserve_config, http_port)
    muxscheduler.serve_models()
    asyncio.run(muxscheduler.schedule_loop())


def power_law_workload(args: argparse.Namespace) -> str:
    """Write the ADBS inputs for a Figure 5 load point and return the file.

    The rates follow the paper's generator (benchmark/end_to_end/
    bench_end_to_end_muxserve.py: get_real_rate x gen_power_law_dis): the model at
    popularity rank r (the order of `models` in the config) gets
    base_rate x rate_scale / (r + 1) ** alpha req/s. The file holds no requests;
    with --http-port they come over HTTP."""
    import getpass
    import json
    import tempfile
    import yaml

    with open(args.model_config) as f:
        names = [model["name"] for model in yaml.safe_load(f)["models"]]
    rates = sorted([name, args.base_rate * args.rate_scale / (rank + 1)**args.alpha]
                   for rank, name in enumerate(names))
    workload = {"arrivals": [], "requests": [],
                "info": {"alpha": args.alpha, "rate_scale": args.rate_scale,
                         "base_rate": args.base_rate, "rates": rates}}
    folder = os.path.join(tempfile.gettempdir(),
                          f"mxo-{getpass.getuser()}-load")
    os.makedirs(folder, mode=0o700, exist_ok=True)
    path = os.path.join(
        folder, f"workload-a{args.alpha:g}-s{args.rate_scale:g}"
        f"-b{args.base_rate:g}.json")
    # The FlexStore and MuxScheduler entries both write the same bytes.
    staging = f"{path}.{os.getpid()}"
    with open(staging, "w") as f:
        json.dump(workload, f)
    os.replace(staging, path)
    logger.info(f"Load point alpha {args.alpha:g}, rate scale "
                f"{args.rate_scale:g}, base rate {args.base_rate:g}: "
                f"rates {rates}")
    return path


def main(args: argparse.Namespace):
    if (args.alpha is None) != (args.rate_scale is None):
        raise SystemExit("--alpha and --rate-scale go together")
    if args.alpha is not None:
        if args.workload_file is not None:
            raise SystemExit("Use --alpha/--rate-scale or --workload-file, "
                             "not both")
        if min(args.alpha, args.rate_scale, args.base_rate) <= 0:
            raise SystemExit("--alpha, --rate-scale and --base-rate must be "
                             "positive")
        args.workload_file = power_law_workload(args)
    muxserve_args = MuxServeArgs.from_cli_args(args)
    muxserve_config = muxserve_args.create_muxserve_config()

    if args.flexstore:
        main_flexstore(muxserve_config)

    if args.muxscheduler:
        main_muxsched(muxserve_config, args.http_port)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='MuxServe Entry Point')
    parser.add_argument("--flexstore",
                        action="store_true",
                        help="Launch FlexStore process.")
    parser.add_argument("--muxscheduler",
                        action="store_true",
                        help="Launch MuxScheduler process.")
    parser.add_argument("--http-port",
                        type=int,
                        default=None,
                        help="Serve requests over HTTP on this port instead "
                        "of replaying the workload file.")
    parser.add_argument("--alpha",
                        type=float,
                        default=None,
                        help="Figure 5 load point: power-law exponent. With "
                        "--rate-scale it gives ADBS its arrival rates instead "
                        "of --workload-file.")
    parser.add_argument("--rate-scale", type=float, default=None,
                        help="Figure 5 load point: rate scale.")
    parser.add_argument("--base-rate",
                        type=float,
                        default=2.0,
                        help="req/s of the most popular model at rate scale "
                        "1 (default: 2.0).")
    parser = MuxServeArgs.add_cli_args(parser)
    args = parser.parse_args()
    main(args)
