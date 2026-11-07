"""MuxServe control and Head-wise storage integration for fixed SGLang.

The backend reuses original MuxServe code from this repository's root package
(``muxserve.muxsched.resource``: SM accounting and KV block sizing). The
repository root is appended to the import path here, so the launcher and
every spawned rank or owner process resolve ``muxserve`` to this checkout
without installing it. Only vLLM-free original modules are imported; see
acceptance/original_reuse_check.py.
"""
from pathlib import Path as _Path
import sys as _sys

__version__ = "0.1.0"

# backends/sglang/muxserve_sglang/__init__.py -> repository root
REPOSITORY_ROOT = _Path(__file__).resolve().parents[3]
if (REPOSITORY_ROOT / "muxserve" / "muxsched" / "resource.py").is_file():
    if str(REPOSITORY_ROOT) not in _sys.path:
        # Appended, not prepended: nothing already importable is shadowed.
        _sys.path.append(str(REPOSITORY_ROOT))


def original_muxserve_path() -> _Path:
    """Directory of the imported original ``muxserve`` package; it must be
    this checkout's (another installed copy would silently change behaviour)."""
    import muxserve
    location = _Path(muxserve.__file__).resolve().parent
    if location != REPOSITORY_ROOT / "muxserve":
        raise ImportError(f"muxserve_sglang needs this repository's muxserve package "
                          f"({REPOSITORY_ROOT / 'muxserve'}), not {location}")
    return location
