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


def main(args: argparse.Namespace):
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
    parser = MuxServeArgs.add_cli_args(parser)
    args = parser.parse_args()
    main(args)
