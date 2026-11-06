"""Original MuxServe Head-wise kernels, independent of the legacy vLLM package."""

from importlib import import_module

from .contracts import IntegrationError


_KERNELS = None


def kernels():
    global _KERNELS
    if _KERNELS is not None:
        return _KERNELS
    try:
        _KERNELS = import_module("muxserve_sglang._headwise")
        return _KERNELS
    except ImportError as exc:
        raise IntegrationError(
            "KERNEL_UNAVAILABLE", "Build the Head-wise extension before starting execution groups"
        ) from exc


def store(key, value, pool_key, pool_value, slots):
    kernels().store(key, value, pool_key, pool_value, slots)


def decode(query, pool_key, pool_value, block_tables, lengths, scale, max_context):
    return kernels().decode(query.contiguous(), pool_key, pool_value,
                            block_tables, lengths, scale, max_context)
