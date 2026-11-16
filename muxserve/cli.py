"""The ``muxserve`` command (``uv run muxserve ...``), shaped like CrossPool's ``xpool``.

    muxserve serve CONFIG.toml        launch the deployment (python -m muxserve.launch)
    muxserve config dump CONFIG.toml  print the effective deployment: the file with
                                      the local settings applied and every default

Both read the local settings file from --local-config or $MUXSERVE_LOCAL_CONFIG
(set in .env, loaded by ``export UV_ENV_FILE=$PWD/.env``).
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import os
import sys


def _local_option(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--local-config", default=os.environ.get("MUXSERVE_LOCAL_CONFIG") or None,
                        help="machine-local settings TOML (default: $MUXSERVE_LOCAL_CONFIG)")


def _without_none(value):
    if isinstance(value, dict):
        return {key: _without_none(item) for key, item in value.items() if item is not None}
    if isinstance(value, (list, tuple)):
        return [_without_none(item) for item in value]
    return value


def _dump(args: argparse.Namespace) -> int:
    from muxserve.config import config_to_toml, load_config
    config = load_config(args.config, require_assets=False, local=args.local_config)
    header = (f"Effective deployment (defaults included)\n"
              f"deployment:     {args.config}\n"
              f"local settings: {args.local_config or '(none)'}\n"
              f"fingerprint:    {config.fingerprint}")
    sys.stdout.write(config_to_toml(_without_none(asdict(config)), header))
    return 0


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else list(argv)
    if argv[:1] == ["serve"]:
        # Everything after `serve` belongs to the launcher, --help included.
        from muxserve.launch import main as launch
        launch(argv[1:])
        return 0
    parser = argparse.ArgumentParser(prog="muxserve", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("serve", help="launch a deployment (options: muxserve serve --help)")
    config = commands.add_parser("config", help="inspect a deployment")
    config_commands = config.add_subparsers(dest="action", required=True)
    dump = config_commands.add_parser("dump", help="print the effective deployment")
    dump.add_argument("config", help="deployment TOML")
    _local_option(dump)
    args = parser.parse_args(argv)
    return _dump(args)


if __name__ == "__main__":
    raise SystemExit(main())
