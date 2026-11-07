__version__ = "0.0.0"


def __getattr__(name):
    # PEP 562: import the vLLM-based entry points on first use, so that
    # vLLM-free submodules (muxsched.resource, used by the SGLang backend)
    # import on their own. `from muxserve import MuxScheduler` is unchanged.
    if name == "MuxServeArgs":
        from .arg_utils import MuxServeArgs
        return MuxServeArgs
    if name == "MuxScheduler":
        from .muxsched.scheduler import MuxScheduler
        return MuxScheduler
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
