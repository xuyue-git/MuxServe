"""Entry points the pinned SGLang fork (xuyue-git/sglang-engine, muxserve-v0.5.20) calls.

The fork imports ``muxserve.flexserver.sglang_hooks`` at its hook sites. Under the
original runtime only one hook is active: the model loader, which maps the weights
FlexStore lends (sglang_worker.FlexStoreModelLoader). SGLang's own scheduler, KV
pool and attention backends are never created here, so their hooks must not run.
"""
import os

_weight_source = None


def enabled() -> bool:
    from muxserve.flexserver.sglang_worker import EXECUTION_GROUP
    return os.environ.get("MUXSERVE_EXECUTION_GROUP") == EXECUTION_GROUP


def set_weight_source(factory) -> None:
    global _weight_source
    _weight_source = factory


def shared_weight_loader(load_config):
    if _weight_source is None:
        raise RuntimeError("SGLangWorker sets the FlexStore weight source first")
    return _weight_source(load_config)


def assert_forward_permitted(runner, forward_batch):
    """ModelRunner.forward is not used: sglang_worker runs the model itself."""


def finish_forward_timing(runner):
    """ModelRunner.forward is not used: sglang_worker runs the model itself."""


def _not_used(name):
    raise RuntimeError(f"{name} is not part of the original runtime's SGLang engine")


def allocate_shared_memory(runner, memory_pool_config=None):
    _not_used("SGLang's KV pool")


def run_controlled_scheduler(scheduler):
    _not_used("SGLang's scheduler")
