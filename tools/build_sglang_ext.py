"""Build the extensions the SGLang engine environment needs, in place.

  cd <repository root> && python tools/build_sglang_ext.py

Run it with the SGLang environment's Python and CUDA 13.2 nvcc on PATH. It builds
  muxserve/memory_manager*.so   csrc/memory_manager.cpp  (FlexStore's KVStorage)
  muxserve/batch_scheduler*.so  csrc/batch_scheduler.cpp (the runtime's BatchScheduler)
The C++ sources are the original ones, unchanged; setup.py builds the same two
for the vLLM environment. Only the C++ standard differs: torch 2.13's headers
need C++20. MuxServe-vLLM's head-wise KV kernels come with the SGLang fork
(sglang.kernels.ops.attention.headwise), which builds them itself.
"""
import os
import sys

os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "8.0")  # A100

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
CXX = ["-O2", "-std=c++20"]

setup(
    name="muxserve-sglang-extensions",
    script_args=["build_ext", "--inplace"] + sys.argv[1:],
    ext_modules=[
        CppExtension("muxserve.memory_manager", ["csrc/memory_manager.cpp"],
                     extra_compile_args=CXX),
        CppExtension("muxserve.batch_scheduler", ["csrc/batch_scheduler.cpp"],
                     extra_compile_args=CXX),
    ],
    cmdclass={"build_ext": BuildExtension},
)
