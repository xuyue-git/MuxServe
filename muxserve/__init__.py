"""MuxServe: flexible spatial-temporal multiplexing for multiple LLM serving.

Execution runs on SGLang (pinned v0.5.20 with the patches under
``upstream-reference/``). The package keeps MuxServe's structure:

- ``muxserve.muxsched``: scheduling (controller, ADBS policy, driver, placement,
  SM accounting, workload generation);
- ``muxserve.flexstore``: unified memory (shared model weights, Head-wise KV
  store, CUDA IPC, owner RPC);
- ``muxserve.flexserver``: per-model prefill/decode execution on SGLang;
- ``muxserve.launch``: the deployment entry point
  (``python -m muxserve.launch CONFIG``).
"""
__version__ = "0.1.0"
