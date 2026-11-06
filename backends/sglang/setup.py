"""Build only the copied project's original Head-wise CUDA operators."""
import os
from pathlib import Path

# The initial acceptance target is A100. Never infer an architecture from
# whatever GPU happens to be visible during package installation.
os.environ["TORCH_CUDA_ARCH_LIST"] = "8.0"

from setuptools import find_packages, setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

ROOT = Path(__file__).resolve().parent

setup(
    name="muxserve-sglang-headwise",
    version="0.1.0",
    description="Permit-controlled SGLang with original MuxServe Head-wise KV",
    python_requires=">=3.10",
    packages=find_packages(include=["muxserve_sglang", "muxserve_sglang.*"]),
    ext_modules=[
        CUDAExtension(
            "muxserve_sglang._headwise",
            ["native/headwise/bindings.cpp", "native/headwise/cache_headwise.cu",
             "native/headwise/attention_headwise_kernels.cu"],
            include_dirs=[str(ROOT / "native/headwise")],
            extra_compile_args={"cxx": ["-O2", "-std=c++20"],
                                "nvcc": ["-O2", "-std=c++20"]},
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
