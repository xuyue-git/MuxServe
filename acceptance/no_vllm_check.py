"""The repository neither imports nor requires vLLM.

    python acceptance/no_vllm_check.py

Scans every Python and shell file outside third_party/, the packaging files and
uv.lock, and the separate engine checkout's Python files, for a vLLM
import or install, and imports the MuxServe control and storage modules with
any `vllm` import made to fail, so that a hidden dependency cannot pass.
"""
from __future__ import annotations

import builtins
import importlib
from pathlib import Path
import re
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

IMPORT = re.compile(r"^\s*(from\s+vllm[\s.]|import\s+vllm\b)", re.M)
INSTALL = re.compile(r"^\s*vllm\b|pip\s+install[^\n]*\bvllm\b", re.M | re.I)
SKIP = {"third_party", ".git", "__pycache__"}
# Modules importable on a CPU-only machine (no CUDA, no SGLang runtime needed).
CPU_MODULES = ("muxserve", "muxserve.config", "muxserve.contracts", "muxserve.trace",
               "muxserve.muxsched.controller", "muxserve.muxsched.driver",
               "muxserve.muxsched.policy", "muxserve.muxsched.events",
               "muxserve.muxsched.resource", "muxserve.flexstore.storage",
               "muxserve.flexstore.ipc", "muxserve.gateway", "muxserve.gateway_proc")


def files(suffixes):
    for path in ROOT.rglob("*"):
        if path.is_file() and path.suffix in suffixes and not SKIP.intersection(path.relative_to(ROOT).parts):
            yield path


class NoVllmChecks(unittest.TestCase):
    def test_no_vllm_import_anywhere(self):
        offenders = [str(p.relative_to(ROOT)) for p in files({".py"})
                     if IMPORT.search(p.read_text(encoding="utf-8", errors="replace"))]
        self.assertEqual(offenders, [])

    def test_no_vllm_import_in_engine_repository(self):
        """The separate SGLang engine checkout imports no vLLM either."""
        from muxserve.engine_source import engine_root
        try:
            engine = engine_root(root=ROOT)
        except ValueError as exc:
            self.skipTest(str(exc))
        offenders = []
        for path in engine.rglob("*.py"):
            parts = path.relative_to(engine).parts
            if ".git" in parts or "__pycache__" in parts:
                continue
            if IMPORT.search(path.read_text(encoding="utf-8", errors="replace")):
                offenders.append(path.relative_to(engine).as_posix())
        self.assertEqual(offenders, [])

    def test_no_vllm_requirement_or_install(self):
        candidates = [ROOT / "setup.py", ROOT / "pyproject.toml", *files({".sh"})]
        offenders = [str(p.relative_to(ROOT)) for p in candidates
                     if p.is_file() and INSTALL.search(p.read_text(encoding="utf-8", errors="replace"))]
        # uv.lock resolves every installed package; vllm must not be one of them.
        lock = ROOT / "uv.lock"
        if lock.is_file() and re.search(r'^name = "vllm"$', lock.read_text(encoding="utf-8"), re.M):
            offenders.append("uv.lock")
        self.assertEqual(offenders, [])

    def test_modules_import_with_vllm_blocked(self):
        real_import = builtins.__import__

        def guarded(name, *args, **kwargs):
            if name == "vllm" or name.startswith("vllm."):
                raise ImportError("vllm is blocked by no_vllm_check")
            return real_import(name, *args, **kwargs)

        builtins.__import__ = guarded
        try:
            for module in CPU_MODULES:
                importlib.import_module(module)
        finally:
            builtins.__import__ = real_import
        self.assertNotIn("vllm", sys.modules)


if __name__ == "__main__":
    unittest.main(verbosity=2)
