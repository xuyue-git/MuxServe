"""MuxServe on SGLang: the Python package and its Head-wise CUDA extension.

Install the pinned, patched SGLang (``third_party/sglang/python``, see README)
in the same environment first. No vLLM is used or installed.
"""
import io
import os
import re
from pathlib import Path

# The acceptance target is A100. Never infer an architecture from whatever GPU
# happens to be visible during package installation.
os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "8.0")

from setuptools import find_packages, setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

ROOT = Path(__file__).resolve().parent
HEADWISE = "csrc/headwise"


def version() -> str:
    text = io.open(ROOT / "muxserve" / "__init__.py", encoding="utf-8").read()
    return re.search(r"^__version__ = ['\"]([^'\"]*)['\"]", text, re.M).group(1)


setup(
    name="muxserve",
    version=version(),
    author="MuxServe Team",
    license="Apache 2.0",
    description="Flexible spatial-temporal multiplexing for multiple LLM serving, on SGLang",
    long_description=io.open(ROOT / "README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    python_requires=">=3.10",
    packages=find_packages(include=["muxserve", "muxserve.*"]),
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
