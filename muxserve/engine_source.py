"""Locate the separate engine checkout for source checks, never for imports.

Serving imports the installed ``sglang`` package normally. These helpers are
only for build tools and CPU checks which inspect the engine's source files.
"""
from __future__ import annotations

from pathlib import Path, PurePosixPath


ENGINE_DIRECTORY = "MuxServe-SGLang-Engine"


def _no_links(path: Path) -> None:
    for current in reversed((path, *path.parents)):
        junction = getattr(current, "is_junction", None)
        if current.is_symlink() or (junction is not None and junction()):
            raise ValueError(f"Engine source contains a link or junction: {current}")
        if current == Path("/home/incoming"):
            raise ValueError("Engine source excludes /home/incoming")


def engine_root(*, root: Path | None = None) -> Path:
    project = Path(root) if root is not None else Path(__file__).absolute().parents[1]
    project = project.absolute()
    if ".." in project.parts:
        raise ValueError("Main repository root must be canonical")
    _no_links(project)
    directory = project.parent / ENGINE_DIRECTORY
    _no_links(directory)
    if not directory.is_dir():
        raise ValueError(f"Separate engine repository is missing: {directory}")
    return directory


def engine_file(name: str, *, root: Path | None = None) -> Path:
    if not isinstance(name, str) or not name or "\\" in name or ":" in name:
        raise ValueError(f"Unsafe engine source path: {name!r}")
    relative = PurePosixPath(name)
    if relative.is_absolute() or relative.as_posix() != name or ".." in relative.parts:
        raise ValueError(f"Non-canonical engine source path: {name!r}")
    path = engine_root(root=root).joinpath(*relative.parts)
    _no_links(path)
    if not path.is_file():
        raise ValueError(f"Engine source is not a regular file: {name}")
    return path
