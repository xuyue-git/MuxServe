"""Create personal integrity locks for shared, revision-pinned read-only assets.

The revision must be checked against the asset source before running this tool.
A local hash lock detects later file drift; it cannot prove a remote publisher.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re

from muxserve_sglang.config import asset_lock_path, load_config, read_asset_path
from muxserve_sglang.contracts import IntegrationError
from muxserve_sglang.preflight import _asset_file, _check_lock, _inventory_names, _sha256


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", help="Deployment JSON with staged model/tokenizer revisions")
    args = parser.parse_args()
    config = load_config(args.config, require_assets=False)
    if not config.asset_lock_dir:
        raise IntegrationError("INVALID_CONFIG", "Set asset_lock_dir to an absolute personal directory")
    roots: dict[Path, str] = {}
    for model in config.models:
        for path, revision in ((model.model_path, model.revision),
                               (model.tokenizer_path, model.tokenizer_revision)):
            if not re.fullmatch(r"[0-9a-f]{40}", revision):
                raise IntegrationError("ASSET_NOT_FROZEN", "An exact upstream revision SHA is required")
            root = read_asset_path(path)
            asset_lock_path(root, config.asset_lock_dir)
            if not root.is_dir() or (root in roots and roots[root] != revision):
                raise IntegrationError("ASSET_MISMATCH", "Asset directory or shared revision differs")
            roots[root] = revision
    for root, revision in roots.items():
        path = asset_lock_path(root, config.asset_lock_dir)
        if path.exists():
            digest = _check_lock(root, revision, set(), config.asset_lock_dir)
            print(f"{root}: existing personal lock verified, SHA-256 {digest}")
            continue
        names = _inventory_names(root)
        if not names:
            raise IntegrationError("ASSET_MISSING", "Cannot lock an empty asset directory")
        files = {name: _sha256(_asset_file(root, name)) for name in sorted(names)}
        payload = json.dumps({"asset_root": str(root), "revision": revision, "files": files}, sort_keys=True,
                             indent=2).encode("utf-8") + b"\n"
        path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(descriptor, "wb") as output:
            output.write(payload)
            output.flush()
            os.fsync(output.fileno())
        print(f"{root}: {len(files)} files locked at {path}, lock SHA-256 {_sha256(path)}")


if __name__ == "__main__":
    main()
