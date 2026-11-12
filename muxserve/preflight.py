"""Offline asset and pre-MPS GPU admission for the fixed A100 deployment.

This command inspects installed metadata/source, the host toolchain, configured
local assets and `nvidia-smi`. It never imports Torch/SGLang, downloads weights,
starts CUDA, or chooses GPUs on behalf of other users.
"""
from __future__ import annotations

import argparse
import csv
from email.parser import Parser
import hashlib
from importlib import machinery
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
from types import FunctionType
from urllib.parse import urlsplit
from urllib.request import url2pathname
from zipimport import zipimporter

from muxserve.config import (SGLANG_COMMIT, SGLANG_TAG, asset_lock_path, load_config,
                     read_asset_path, safe_path, visible_devices)
from muxserve.contracts import IntegrationError


def _tool_output(command: list[str], *, input_text: str | None = None) -> str:
    try:
        result = subprocess.run(command, input=input_text, capture_output=True,
                                text=True, check=True, timeout=15)
    except (OSError, subprocess.SubprocessError) as exc:
        raise IntegrationError("TOOLCHAIN_MISMATCH", f"Toolchain inspection failed: {command[0]}") from exc
    return result.stdout + result.stderr


def _distribution(name: str, search: list[Path]) -> dict | None:
    """Read filesystem metadata only, in sys.path order; never invoke finders."""
    normalize = lambda value: re.sub(r"[-_.]+", "-", value).lower()
    for root in search:
        if not root.is_dir():
            continue  # Zip distributions and arbitrary import hooks are unsupported.
        with os.scandir(root) as entries:
            names = sorted(entry.name for entry in entries)
        candidates = []
        for filename in names:
            suffix = next((s for s in (".dist-info", ".egg-info") if filename.endswith(s)), None)
            if suffix is None:
                continue
            stem = filename[:-len(suffix)]
            project = re.split(r"-(?=\d)", stem, maxsplit=1)[0]
            if normalize(project) != normalize(name):
                continue
            candidate = safe_path(root / filename)
            is_directory = candidate.is_dir()
            if suffix == ".dist-info" and not is_directory:
                raise IntegrationError("SOFTWARE_MISMATCH", f"Invalid metadata directory for {name}")
            metadata_file = (safe_path(candidate / ("METADATA" if suffix == ".dist-info" else "PKG-INFO"))
                             if is_directory else candidate)
            if not metadata_file.is_file():
                raise IntegrationError("SOFTWARE_MISMATCH", f"Missing metadata for {name}")
            headers = Parser().parsestr(metadata_file.read_text(encoding="utf-8"), headersonly=True)
            if (len(headers.get_all("Name", [])) != 1 or len(headers.get_all("Version", [])) != 1
                    or normalize(headers["Name"]) != normalize(name) or not headers["Version"].strip()):
                raise IntegrationError("SOFTWARE_MISMATCH", f"Invalid metadata identity for {name}")
            candidates.append({"version": headers["Version"].strip(), "distribution_root": str(root),
                               "metadata_path": str(metadata_file), "metadata_directory": str(candidate)})
        if len(candidates) > 1:
            raise IntegrationError("SOFTWARE_MISMATCH", f"Ambiguous installed metadata for {name}")
        if candidates:
            return candidates[0]
    return None


def _sglang_source(search: list[Path], distribution: dict) -> Path:
    """Support filesystem wheels and setuptools path-based compat editables."""
    unsupported = "Use a filesystem wheel or reinstall SGLang with --config-settings editable_mode=compat"
    for finder in sys.meta_path:
        # Setuptools' standard helper handles distutils only. No finder is called.
        if (finder is machinery.BuiltinImporter or finder is machinery.FrozenImporter
                or finder is machinery.PathFinder
                or (type(finder).__module__ == "_distutils_hack"
                    and type(finder).__name__ == "DistutilsMetaFinder")):
            continue
        raise IntegrationError("SOFTWARE_MISMATCH", f"Unsupported custom import finder. {unsupported}")
    for hook in sys.path_hooks:
        if hook is zipimporter or (type(hook) is FunctionType
                                  and hook.__module__ == "_frozen_importlib_external"
                                  and hook.__qualname__ == "FileFinder.path_hook.<locals>.path_hook_for_FileFinder"):
            continue
        raise IntegrationError("SOFTWARE_MISMATCH", f"Unsupported custom path hook. {unsupported}")
    origin = None
    suffixes = machinery.EXTENSION_SUFFIXES + machinery.SOURCE_SUFFIXES + machinery.BYTECODE_SUFFIXES
    for root in search:
        if root.is_file():
            raise IntegrationError("SOFTWARE_MISMATCH", f"Archive sys.path entries are unsupported. {unsupported}")
        cached = sys.path_importer_cache.get(str(root))
        if cached is not None and type(cached) is not machinery.FileFinder:
            raise IntegrationError("SOFTWARE_MISMATCH", f"Unsupported cached path importer. {unsupported}")
        # Validate each child before any probe; do not use PathFinder/path hooks.
        directory = safe_path(root / "sglang")
        if directory.is_dir():
            for suffix in suffixes:
                candidate = safe_path(directory / ("__init__" + suffix))
                if candidate.is_file():
                    if candidate.parent != directory:
                        raise IntegrationError("SOFTWARE_MISMATCH", "SGLang initializer points outside its package")
                    origin = candidate
                    break
        if origin is not None:
            break
        for suffix in suffixes:
            candidate = safe_path(root / ("sglang" + suffix))
            if candidate.is_file():
                raise IntegrationError("SOFTWARE_MISMATCH", "SGLang is shadowed by a top-level module")
    if origin is None or origin.name != "__init__.py":
        raise IntegrationError("SOFTWARE_MISMATCH", f"SGLang Python source is not resolvable. {unsupported}")
    package = origin.parent
    metadata_directory = safe_path(distribution["metadata_directory"])
    direct_url = safe_path(metadata_directory / "direct_url.json")
    editable_root = None
    if direct_url.is_file():
        record = _read_json(direct_url)
        if not isinstance(record.get("dir_info", {}), dict):
            raise IntegrationError("SOFTWARE_MISMATCH", "Invalid direct_url directory metadata")
        if record.get("dir_info", {}).get("editable") is True:
            if not isinstance(record.get("url"), str):
                raise IntegrationError("SOFTWARE_MISMATCH", "Editable source URL is missing")
            url = urlsplit(record["url"])
            if url.scheme != "file" or url.netloc not in ("", "localhost") or url.query or url.fragment:
                raise IntegrationError("SOFTWARE_MISMATCH", "Editable source must use a local file URL")
            target = Path(url2pathname(url.path))
            if not target.is_absolute():
                raise IntegrationError("SOFTWARE_MISMATCH", "Editable source URL must be absolute")
            editable_root = safe_path(target)
    expected_root = editable_root or safe_path(distribution["distribution_root"])
    if package.parent != expected_root:
        raise IntegrationError("SOFTWARE_MISMATCH", "SGLang metadata and resolved source belong to different installations")
    loaded = sys.modules.get("sglang")
    if loaded is not None:
        loaded_file = vars(loaded).get("__file__")
        if not loaded_file or safe_path(loaded_file) != origin:
            raise IntegrationError("SOFTWARE_MISMATCH", "Loaded SGLang differs from resolved source")
    return package


def _executable(executable: str) -> Path:
    """Resolve explicit/PATH executables without shutil.which's early probes."""
    if os.path.dirname(executable):
        candidates = [safe_path(executable)]
    else:
        roots = [safe_path(entry or os.getcwd()) for entry in os.get_exec_path()]
        candidates = [safe_path(root / executable) for root in roots]
    for candidate in candidates:
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return candidate
    raise IntegrationError("TOOLCHAIN_MISMATCH", f"Executable is absent from approved paths: {executable}")


def _compiler_command() -> list[str]:
    command = shlex.split(os.environ.get("CXX", "c++"))
    if len(command) != 1 or not command[0]:
        raise IntegrationError("TOOLCHAIN_MISMATCH", "CXX must name one compiler executable without wrapper/options")
    return [str(_executable(command[0]))]


def inspect_software() -> dict:
    """User-run pre-CUDA identity gate; no numerical/ABI claim from versions."""
    required = {"torch": "2.13.0", "transformers": "5.12.1",
                "sglang-kernel": "0.4.7", "sglang": SGLANG_TAG.removeprefix("v")}
    # Complete this validation before enumeration or metadata reads of any root.
    raw_search = [entry or os.getcwd() for entry in sys.path]
    checked_search = [safe_path(entry) for entry in raw_search]
    # Check both original and resolved cache keys: a symlinked sys.path entry
    # can have its own importer even when its resolved root is ordinary.
    for raw, checked in zip(raw_search, checked_search):
        for key in (raw, str(checked)):
            importer = sys.path_importer_cache.get(key)
            if importer is not None and (type(importer) is not machinery.FileFinder
                                        or safe_path(importer.path) != checked):
                raise IntegrationError("SOFTWARE_MISMATCH", "Unsupported cached path importer; use a filesystem installation")
    search = list(dict.fromkeys(checked_search))
    packages = {}
    for name, expected in required.items():
        distribution = _distribution(name, search)
        if distribution is None:
            raise IntegrationError("SOFTWARE_MISMATCH", f"Required installed filesystem distribution is absent: {name}")
        actual = distribution["version"]
        # Local CUDA wheel tags are reported, never used to guess a wheel index.
        if actual.split("+", 1)[0] != expected:
            raise IntegrationError("SOFTWARE_MISMATCH", f"{name}: expected {expected}, installed {actual}")
        packages[name] = distribution
    torchaudio_distribution = _distribution("torchaudio", search)
    package = _sglang_source(search, packages["sglang"])
    root = safe_path(Path(__file__).parent.parent)
    patches = _read_json(root / "upstream-reference/patches.json")
    extraction = _read_json(root / "upstream-reference/extraction.json")
    if extraction.get("commit") != SGLANG_COMMIT or extraction.get("tag") != SGLANG_TAG:
        raise IntegrationError("SOFTWARE_MISMATCH", "Bundled upstream identity differs from the deployment pin")
    inspected = {}
    for name, entry in patches.items():
        prefix = "python/sglang/"
        if not name.startswith(prefix):
            raise IntegrationError("SOFTWARE_MISMATCH", "Unexpected patched source path")
        relative = name[len(prefix):]
        actual = _sha256(_asset_file(package, relative))
        if actual != entry["patched_sha256"]:
            raise IntegrationError("SOFTWARE_MISMATCH", f"Installed SGLang patch mismatch: {relative}")
        inspected[relative] = actual
    if not inspected:
        raise IntegrationError("SOFTWARE_MISMATCH", "SGLang patch identity inventory is empty")
    cuda_home = safe_path(os.environ.get("CUDA_HOME", "/usr/local/cuda-13.2"))
    nvcc = safe_path(cuda_home / "bin/nvcc")
    cuda_version = _tool_output([str(nvcc), "--version"])
    match = re.search(r"release\s+(\d+\.\d+)", cuda_version)
    if match is None or match.group(1) != "13.2":
        raise IntegrationError("TOOLCHAIN_MISMATCH", "CUDA toolkit 13.2 is required")
    compiler = _compiler_command()
    compiler_version = _tool_output([*compiler, "--version"])
    # Preprocess only; create no binary. Match Torch 2.13 and setup.py.
    # This checks the language mode, not header, nvcc or extension ABI compatibility.
    language = _tool_output([*compiler, "-std=c++20", "-dM", "-E", "-x", "c++", "-"], input_text="")
    standard = re.search(r"^#define __cplusplus (\d+)L?\s*$", language, re.MULTILINE)
    if standard is None or int(standard.group(1)) < 202002:
        raise IntegrationError("TOOLCHAIN_MISMATCH", "The native extension requires C++20")
    torchaudio = torchaudio_distribution["version"] if torchaudio_distribution else None
    return {"packages": packages, "sglang_source_path": str(package),
            "sglang_commit": SGLANG_COMMIT, "sglang_tag": SGLANG_TAG,
            "sglang_patch_sha256": inspected, "patch_manifest_sha256": _sha256(root / "upstream-reference/patches.json"),
            "cuda_home": str(cuda_home), "nvcc_path": str(nvcc), "nvcc_version": cuda_version.strip(),
            "cxx_command": compiler, "cxx_version": compiler_version.strip(), "cxx_standard": "c++20",
            "extension_build_validation": "unexecuted",
            "torchaudio_version": torchaudio, "torchaudio_abi_validation": "unexecuted",
            "source_identity_scope": "installed patched files; upstream provenance follows bundled extraction record"}


def _read_json(path: Path) -> dict:
    checked = read_asset_path(path)
    if not checked.is_file():
        raise IntegrationError("ASSET_MISSING", f"Required asset file is missing: {path.name}")
    result = json.loads(checked.read_text(encoding="utf-8"))
    if not isinstance(result, dict):
        raise IntegrationError("ASSET_MISMATCH", "Asset metadata must be a JSON object")
    return result


def _asset_file(root: Path, name: str) -> Path:
    if (not isinstance(name, str) or not name or Path(name).is_absolute()
            or any(part in (".", "..") for part in Path(name).parts)
            or name.startswith(("/", "\\")) or "\\" in name):
        raise IntegrationError("ASSET_MISMATCH", "Asset index contains an unsafe file name")
    file = read_asset_path(root / name)
    if not file.is_file():
        raise IntegrationError("ASSET_MISSING", f"Indexed asset file is missing: {name}")
    return file


def _weight_files(root: Path) -> set[str]:
    for index_name in ("model.safetensors.index.json", "pytorch_model.bin.index.json"):
        if read_asset_path(root / index_name).is_file():
            index = _read_json(root / index_name)
            mapping = index.get("weight_map")
            if not isinstance(mapping, dict) or not mapping:
                raise IntegrationError("ASSET_MISMATCH", "Weight index has no weight map")
            if any(not isinstance(name, str) for name in mapping.values()):
                raise IntegrationError("ASSET_MISMATCH", "Weight index file names must be strings")
            names = set(mapping.values())
            for name in names:
                _asset_file(root, name)
            return names | {index_name}
    with os.scandir(root) as entries:
        names = {entry.name for entry in entries
                 if entry.name.endswith((".safetensors", ".bin"))}
    if not names:
        raise IntegrationError("ASSET_MISSING", "No local model weight files found")
    for name in names:
        _asset_file(root, name)
    return names


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with read_asset_path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _inventory_names(root: Path) -> set[str]:
    names = set()
    pending = [(root, "")]
    while pending:
        directory, prefix = pending.pop()
        with os.scandir(read_asset_path(directory)) as entries:
            for entry in entries:
                name = prefix + entry.name
                if name == ".muxserve-asset-lock.json":
                    continue
                target = read_asset_path(entry.path)
                if entry.is_dir(follow_symlinks=False):
                    pending.append((target, name + "/"))
                elif target.is_file():
                    names.add(name)
                else:
                    raise IntegrationError("ASSET_MISMATCH", "Unsupported asset directory link or file type")
    return names


def _check_lock(root: Path, revision: str, required: set[str], lock_dir: str = "") -> str:
    """A lock covers actual files; its revision must come from staged provenance."""
    root = read_asset_path(root)
    path = asset_lock_path(root, lock_dir)
    lock = _read_json(path)
    files = lock.get("files")
    if ((lock_dir and lock.get("asset_root") != str(root))
            or lock.get("revision") != revision or not isinstance(files, dict)
            or not required <= files.keys() or set(files) != _inventory_names(root)):
        raise IntegrationError("ASSET_NOT_FROZEN", "Asset lock lacks revision or required file hashes")
    for name, expected in files.items():
        if not isinstance(expected, str) or len(expected) != 64 or any(c not in "0123456789abcdef" for c in expected):
            raise IntegrationError("ASSET_NOT_FROZEN", "Asset lock contains an invalid SHA-256")
        if _sha256(_asset_file(root, name)) != expected:
            raise IntegrationError("ASSET_CHANGED", f"Staged asset changed: {name}")
    return _sha256(path)


def inspect_assets(config) -> list[dict]:
    inspected = []
    for model in config.models:
        root = read_asset_path(model.model_path)
        tokenizer = read_asset_path(model.tokenizer_path)
        required = {"config.json"} | _weight_files(root)
        tokenizer_names = {name for name in ("tokenizer.model", "tokenizer.json")
                           if read_asset_path(tokenizer / name).is_file()}
        if not tokenizer_names:
            raise IntegrationError("ASSET_MISSING", "A local Llama tokenizer file is required")
        tokenizer_names.add("tokenizer_config.json")
        for name in tokenizer_names:
            _asset_file(tokenizer, name)
        model_lock = _check_lock(root, model.revision,
                                 required | tokenizer_names if tokenizer == root else required,
                                 config.asset_lock_dir)
        tokenizer_lock = (_check_lock(tokenizer, model.tokenizer_revision, tokenizer_names,
                                      config.asset_lock_dir)
                          if tokenizer != root else model_lock)
        if tokenizer == root and model.tokenizer_revision != model.revision:
            raise IntegrationError("ASSET_NOT_FROZEN", "Shared model/tokenizer snapshot has different revisions")
        inspected.append({"model_id": model.model_id, "model_revision": model.revision,
                          "tokenizer_revision": model.tokenizer_revision,
                          "model_lock_sha256": model_lock,
                          "tokenizer_lock_sha256": tokenizer_lock,
                          "weight_files": sorted(required - {"config.json"})})
    return inspected


def inspect_gpus(tp_size: int) -> list[dict]:
    requested = visible_devices(tp_size)
    try:
        output = subprocess.run(
            [str(_executable("nvidia-smi")), "-i", ",".join(requested),
             "--query-gpu=index,uuid,name,compute_cap,memory.free,memory.total",
             "--format=csv,noheader,nounits"], check=True, capture_output=True,
            text=True, timeout=15, env={**os.environ, "CUDA_VISIBLE_DEVICES": ""},
        ).stdout
    except (OSError, subprocess.SubprocessError) as exc:
        raise IntegrationError("GPU_PREFLIGHT_FAILED", "nvidia-smi inventory failed") from exc
    cards = {}
    uuids = {}
    for row in csv.reader(output.splitlines()):
        if len(row) != 6:
            raise IntegrationError("GPU_PREFLIGHT_FAILED", "Unexpected nvidia-smi inventory format")
        index, uuid, name, capability, free_mib, total_mib = (cell.strip() for cell in row)
        card = {"index": int(index), "uuid": uuid, "name": name,
                "compute_capability": capability, "free_mib": int(free_mib),
                "total_mib": int(total_mib)}
        cards[str(card["index"])] = card
        uuids[uuid] = card
    selected = []
    for token in requested:
        card = cards.get(token) if token.isdecimal() else uuids.get(token)
        if card is None or "A100" not in card["name"] or card["compute_capability"] != "8.0":
            raise IntegrationError("GPU_MAPPING_REQUIRED", "Each coordinated device must resolve to a full A100 UUID")
        # nvidia-smi reports MiB. Keep every comparison in the design's
        # 40GB hardware class, allowing ordinary driver-reserved capacity.
        if not 38 * 1024 <= card["total_mib"] <= 41 * 1024:
            raise IntegrationError("GPU_MAPPING_REQUIRED", "The approved comparison requires four A100 40GB devices")
        if not 0 <= card["free_mib"] <= card["total_mib"]:
            raise IntegrationError("GPU_PREFLIGHT_FAILED", "GPU free-memory inventory is inconsistent")
        selected.append(card)
    if len({card["uuid"] for card in selected}) != tp_size:
        raise IntegrationError("GPU_MAPPING_REQUIRED", "Coordinated devices alias the same physical GPU")
    return selected


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", help="Frozen deployment JSON")
    args = parser.parse_args()
    config = load_config(args.config)
    software = inspect_software()
    assets = inspect_assets(config)
    cards = inspect_gpus(config.tp_size)
    print(json.dumps({"sglang_commit": SGLANG_COMMIT, "software": software,
                      "config_sha256": _sha256(safe_path(args.config)),
                      "assets": assets, "rank_gpus": cards,
                      "normalized_cuda_visible_devices": ",".join(card["uuid"] for card in cards),
                      "kv_bytes_per_rank": config.kv_bytes_per_rank,
                      "extra_headroom_bytes_per_rank": config.extra_headroom_bytes_per_rank},
                     sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
