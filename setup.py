"""Build the Head-wise CUDA extension; package metadata lives in pyproject.toml.

The pinned SGLang fork comes from the sibling ``../MuxServe-SGLang-Engine``
repository (``uv sync``, or pip as in README). No vLLM is used or installed.
"""
import os
from pathlib import Path

# The acceptance target is A100. Never infer an architecture from whatever GPU
# happens to be visible during package installation.
os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "8.0")

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

ROOT = Path(__file__).resolve().parent
HEADWISE = "csrc/headwise"

setup(
    ext_modules=[
        CUDAExtension(
            "muxserve._headwise",
            [f"{HEADWISE}/bindings.cpp", f"{HEADWISE}/cache_headwise.cu",
             f"{HEADWISE}/attention_headwise_kernels.cu"],
            include_dirs=[str(ROOT / HEADWISE)],
            extra_compile_args={"cxx": ["-O2", "-std=c++20"],
                                "nvcc": ["-O2", "-std=c++20"]},
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
