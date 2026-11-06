"""Offline safety and integrity tests for the source materializer (stdlib only)."""
from __future__ import annotations

import hashlib
import importlib.util
import io
import json
import marshal
import os
from pathlib import Path
import shutil
import ssl
import stat
import struct
import tarfile
import tempfile
import unittest
from types import SimpleNamespace
from unittest import mock


SCRIPT = Path(__file__).absolute().parents[1] / "tools/bootstrap_source.py"
SPEC = importlib.util.spec_from_file_location("bootstrap_source", SCRIPT)
bootstrap_source = importlib.util.module_from_spec(SPEC)
if SCRIPT.is_file():
    SPEC.loader.exec_module(bootstrap_source)


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


class BootstrapSourceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.sandbox = tempfile.TemporaryDirectory()
        self.addCleanup(self.sandbox.cleanup)
        self.root = Path(self.sandbox.name) / "repository with spaces"
        (self.root / "tools").mkdir(parents=True)
        (self.root / "upstream-reference").mkdir()
        self.files = {
            "README.md": b"Upstream README\n",
            "LICENSE": b"Upstream license\n",
            "python/sglang/worker.py": b"VERSION = 'pristine'\n",
            "scripts/run.sh": b"#!/bin/sh\nexit 0\n",
        }
        self.patched = b"VERSION = 'patched'\n"
        self.links = {
            ".dockerignore": ".gitignore",
            "python/sglang/srt/mem_cache/cpp_radix_tree/.clang-format": "../../../kernels/aot/.clang-format",
            "sgl-model-gateway/LICENSE": "../LICENSE",
        }
        self.prefix = "sglang-94602c9c2b7cbdb8efd5c52802dac6a1c180089e"
        self.archive = Path(self.sandbox.name) / "source.tar.gz"
        self.write_archive()
        self.extraction = {
            "commit": "94602c9c2b7cbdb8efd5c52802dac6a1c180089e",
            "tag": "v0.5.20",
            "archive_sha256": digest(self.archive.read_bytes()),
            "files": len(self.files),
            "files_sha256": {name: digest(data) for name, data in self.files.items()},
            "skipped_links": [{"path": name, "target": target} for name, target in self.links.items()],
        }
        self.patches = {
            "python/sglang/worker.py": {
                "source_sha256": digest(self.files["python/sglang/worker.py"]),
                "patched_sha256": digest(self.patched),
                "upgrade_from": {digest(b"obsolete"): [["obsolete", "patched"]]},
            }
        }
        self.packaging = {
            "sglang_commit": self.extraction["commit"],
            "additions": [
                {"path": "python/README.md", "source": "README.md", "sha256": digest(self.files["README.md"])},
                {"path": "python/LICENSE", "source": "LICENSE", "sha256": digest(self.files["LICENSE"])},
            ],
        }
        self.write_metadata()
        (self.root / "tools/patch_sglang.py").write_text(
            "from pathlib import Path\n"
            "ROOT = Path(__file__).absolute().parents[1]\n"
            "UPSTREAM = ROOT / 'third_party/sglang'\n"
            "def apply():\n"
            "    path = UPSTREAM / 'python/sglang/worker.py'\n"
            "    assert path.read_bytes() == b\"VERSION = 'pristine'\\n\"\n"
            "    path.write_bytes(b\"VERSION = 'patched'\\n\")\n",
            encoding="utf-8",
        )

    def write_archive(self, extra: tarfile.TarInfo | None = None) -> None:
        with tarfile.open(self.archive, "w:gz") as archive:
            directory = tarfile.TarInfo(self.prefix)
            directory.type = tarfile.DIRTYPE
            archive.addfile(directory)
            for name, data in self.files.items():
                member = tarfile.TarInfo(f"{self.prefix}/{name}")
                member.size = len(data)
                member.mode = 0o755 if name.endswith(".sh") else 0o644
                archive.addfile(member, io.BytesIO(data))
            for name, target in self.links.items():
                member = tarfile.TarInfo(f"{self.prefix}/{name}")
                member.type = tarfile.SYMTYPE
                member.linkname = target
                archive.addfile(member)
            if extra is not None:
                archive.addfile(extra, io.BytesIO(b"x" * extra.size) if extra.isfile() else None)

    def write_metadata(self) -> None:
        for name, data in (
            ("extraction.json", self.extraction),
            ("patches.json", self.patches),
            ("packaging-additions.json", self.packaging),
        ):
            (self.root / "upstream-reference" / name).write_text(json.dumps(data), encoding="utf-8")

    def invoke(self) -> dict:
        self.assertTrue(hasattr(bootstrap_source, "bootstrap"), "Source materializer is not implemented")
        with mock.patch.multiple(
            bootstrap_source,
            ARCHIVE_SHA256=digest(self.archive.read_bytes()),
            ARCHIVE_SIZE=self.archive.stat().st_size,
            SOURCE_FILE_COUNT=len(self.files),
        ), mock.patch.object(bootstrap_source, "_download_archive", side_effect=AssertionError("No network in offline tests")):
            return bootstrap_source.bootstrap(root=self.root, archive=self.archive)

    @property
    def destination(self) -> Path:
        return self.root / "third_party/sglang"

    def test_materializes_verified_patched_source_and_is_idempotent(self) -> None:
        report = self.invoke()
        self.assertEqual(report["status"], "materialized")
        self.assertEqual(len(report["skipped_links"]), 3)
        self.assertEqual((self.destination / "python/sglang/worker.py").read_bytes(), self.patched)
        self.assertEqual((self.destination / "python/README.md").read_bytes(), self.files["README.md"])
        self.assertEqual((self.destination / "python/LICENSE").read_bytes(), self.files["LICENSE"])
        self.assertFalse((self.destination / ".dockerignore").exists())
        original_stat = (self.destination / "README.md").stat().st_mtime_ns
        self.archive.unlink()
        with mock.patch.multiple(
            bootstrap_source,
            ARCHIVE_SHA256=self.extraction["archive_sha256"],
            SOURCE_FILE_COUNT=len(self.files),
        ), mock.patch.object(bootstrap_source, "_download_archive", side_effect=AssertionError("Already present")):
            self.assertEqual(bootstrap_source.bootstrap(root=self.root)["status"], "verified")
        self.assertEqual((self.destination / "README.md").stat().st_mtime_ns, original_stat)

    def test_refuses_existing_source_drift_and_obsolete_patch(self) -> None:
        self.invoke()
        path = self.destination / "python/sglang/worker.py"
        for content in (b"user edit", b"obsolete", self.files["python/sglang/worker.py"]):
            with self.subTest(content=content):
                path.write_bytes(content)
                with self.assertRaisesRegex(RuntimeError, "integrity|mismatch|drift"):
                    self.invoke()
                self.assertEqual(path.read_bytes(), content)

    def test_refuses_extra_or_missing_existing_files(self) -> None:
        self.invoke()
        extra = self.destination / "surprise.py"
        extra.write_bytes(b"unrecognized")
        with self.assertRaisesRegex(RuntimeError, "unexpected|unrecognized|extra"):
            self.invoke()
        extra.unlink()
        (self.destination / "LICENSE").unlink()
        with self.assertRaisesRegex(RuntimeError, "missing|integrity"):
            self.invoke()

    def test_refuses_unrecognized_empty_directory(self) -> None:
        self.invoke()
        (self.destination / "unrecognized").mkdir()
        with self.assertRaisesRegex(RuntimeError, "unexpected|unrecognized"):
            self.invoke()

    def add_installed_artifacts(self) -> None:
        cache = self.destination / "python/sglang/__pycache__"
        cache.mkdir(exist_ok=True)
        (cache / "worker.cpython-311.pyc").write_bytes(b"generated bytecode")
        metadata = self.destination / "python/sglang.egg-info"
        metadata.mkdir(exist_ok=True)
        for name in ("PKG-INFO", "SOURCES.txt", "dependency_links.txt", "entry_points.txt", "requires.txt", "top_level.txt"):
            (metadata / name).write_bytes(b"generated metadata\n")

    def test_accepts_only_standard_installed_artifacts(self) -> None:
        self.invoke()
        self.add_installed_artifacts()
        self.assertEqual(self.invoke()["status"], "verified")

    def test_installed_artifacts_do_not_relax_tracked_source_hashes(self) -> None:
        self.invoke()
        self.add_installed_artifacts()
        (self.destination / "python/sglang/worker.py").write_bytes(b"edited source")
        with self.assertRaisesRegex(RuntimeError, "integrity.*mismatch"):
            self.invoke()

    def test_rejects_unexpected_code_or_nested_generated_artifacts(self) -> None:
        self.invoke()
        self.add_installed_artifacts()
        for name in (
            "python/sglang/__pycache__/unexpected.py",
            "python/sglang.egg-info/unexpected.py",
            "python/sglang.egg-info/nested/PKG-INFO",
            "python/sglang/__pycache__/nested/worker.pyc",
        ):
            with self.subTest(name=name):
                path = self.destination / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"unrecognized")
                with self.assertRaisesRegex(RuntimeError, "unexpected"):
                    self.invoke()
                path.unlink()
                if path.parent.name == "nested":
                    path.parent.rmdir()

    def test_generated_artifacts_reject_links_reparse_points_and_special_files(self) -> None:
        self.invoke()
        self.add_installed_artifacts()
        original_lstat = Path.lstat
        for name, mode, attributes in (
            ("worker.cpython-311.pyc", stat.S_IFLNK, 0),
            ("worker.cpython-311.pyc", stat.S_IFREG, 0x400),
            ("worker.cpython-311.pyc", stat.S_IFIFO, 0),
            ("sglang.egg-info", stat.S_IFLNK, 0),
            ("sglang.egg-info", stat.S_IFDIR, 0x400),
        ):
            with self.subTest(name=name, mode=mode, attributes=attributes):
                def substituted_lstat(path):
                    if path.name == name:
                        return SimpleNamespace(st_mode=mode, st_file_attributes=attributes)
                    return original_lstat(path)

                with mock.patch.object(Path, "lstat", substituted_lstat), self.assertRaisesRegex(RuntimeError, "link|reparse|special"):
                    self.invoke()

    def test_rejects_archive_hash_or_size_before_extraction(self) -> None:
        self.assertTrue(hasattr(bootstrap_source, "bootstrap"), "Source materializer is not implemented")
        for checksum, size in (("0" * 64, self.archive.stat().st_size), (self.extraction["archive_sha256"], 1)):
            with self.subTest(checksum=checksum, size=size), mock.patch.multiple(
                bootstrap_source, ARCHIVE_SHA256=checksum, ARCHIVE_SIZE=size, SOURCE_FILE_COUNT=len(self.files)
            ):
                self.extraction["archive_sha256"] = checksum
                self.write_metadata()
                with self.assertRaisesRegex(RuntimeError, "archive|Archive"):
                    bootstrap_source.bootstrap(root=self.root, archive=self.archive)
                self.assertFalse(self.destination.exists())

    def test_rejects_unsafe_or_duplicate_archive_members(self) -> None:
        cases = [
            (f"{self.prefix}/../escape", tarfile.REGTYPE, ""),
            ("/absolute", tarfile.REGTYPE, ""),
            (f"{self.prefix}/a\\escape", tarfile.REGTYPE, ""),
            (f"{self.prefix}/C:escape", tarfile.REGTYPE, ""),
            (f"{self.prefix}/README.md", tarfile.REGTYPE, ""),
            (f"{self.prefix}/link", tarfile.SYMTYPE, "../../escape"),
            (f"{self.prefix}/hardlink", tarfile.LNKTYPE, f"{self.prefix}/README.md"),
            (f"{self.prefix}/fifo", tarfile.FIFOTYPE, ""),
            (f"{self.prefix}/device", tarfile.CHRTYPE, ""),
            (f"{self.prefix}/unknown-directory", tarfile.DIRTYPE, ""),
            (f"{self.prefix}/.dockerignore", tarfile.SYMTYPE, "../../escape"),
        ]
        for name, kind, target in cases:
            with self.subTest(name=name, kind=kind):
                member = tarfile.TarInfo(name)
                member.type = kind
                member.linkname = target
                self.write_archive(member)
                self.extraction["archive_sha256"] = digest(self.archive.read_bytes())
                self.write_metadata()
                with self.assertRaises(RuntimeError):
                    self.invoke()
                self.assertFalse(self.destination.exists())
                self.assertFalse((self.root / "escape").exists())
                self.assertEqual(list((self.root / "third_party").glob(".sglang-bootstrap-*")), [])

    def test_rejects_manifest_content_mismatch(self) -> None:
        self.extraction["files_sha256"]["README.md"] = "0" * 64
        self.write_metadata()
        with self.assertRaisesRegex(RuntimeError, "integrity|mismatch|hash"):
            self.invoke()
        self.assertFalse(self.destination.exists())

    def test_rejects_noncanonical_manifest_path(self) -> None:
        self.extraction["files_sha256"]["../escape"] = self.extraction["files_sha256"].pop("README.md")
        self.write_metadata()
        with self.assertRaisesRegex(RuntimeError, "path|Path"):
            self.invoke()
        self.assertFalse(self.destination.exists())

    def test_failed_patch_never_publishes_partial_source(self) -> None:
        (self.root / "tools/patch_sglang.py").write_text("def apply():\n    raise RuntimeError('patch failed')\n", encoding="utf-8")
        with self.assertRaisesRegex(RuntimeError, "patch failed"):
            self.invoke()
        self.assertFalse(self.destination.exists())
        self.assertEqual(list((self.root / "third_party").glob(".sglang-bootstrap-*")), [])

    def test_patcher_executes_source_instead_of_matching_cached_bytecode(self) -> None:
        patcher = self.root / "tools/patch_sglang.py"
        info = patcher.stat()
        poisoned = compile("raise RuntimeError('poisoned patcher cache executed')\n", str(patcher), "exec")
        # A normal timestamp-based .pyc header matches the trusted source exactly,
        # while the marshalled code body differs from that source.
        bytecode = (
            importlib.util.MAGIC_NUMBER
            + struct.pack("<III", 0, int(info.st_mtime) & 0xFFFFFFFF, info.st_size)
            + marshal.dumps(poisoned)
        )
        cache = Path(importlib.util.cache_from_source(str(patcher)))
        cache.parent.mkdir(exist_ok=True)
        cache.write_bytes(bytecode)
        self.assertEqual(self.invoke()["status"], "materialized")
        self.assertEqual(cache.read_bytes(), bytecode)

    def test_patcher_must_produce_every_manifest_target(self) -> None:
        self.patches["scripts/run.sh"] = {
            "source_sha256": digest(self.files["scripts/run.sh"]),
            "patched_sha256": digest(b"unimplemented patch"),
        }
        self.write_metadata()
        with self.assertRaisesRegex(RuntimeError, "integrity.*mismatch"):
            self.invoke()
        self.assertFalse(self.destination.exists())

    def test_download_keeps_tls_certificate_verification_enabled(self) -> None:
        response = io.BytesIO(self.archive.read_bytes())
        opener = mock.Mock()
        opener.open.return_value = response
        output = Path(self.sandbox.name) / "downloaded.tar.gz"
        with mock.patch.object(bootstrap_source, "build_opener", return_value=opener) as build:
            bootstrap_source._download_archive(output)
        https = build.call_args.args[0]
        self.assertEqual(https._context.verify_mode, ssl.CERT_REQUIRED)
        self.assertTrue(https._context.check_hostname)
        self.assertEqual(opener.open.call_args.args[0].full_url, bootstrap_source.ARCHIVE_URL)
        self.assertEqual(output.read_bytes(), self.archive.read_bytes())
        redirect = bootstrap_source._HTTPSOnlyRedirect()
        for url in ("http://codeload.github.com/archive", "https://example.org/archive"):
            with self.subTest(url=url), self.assertRaisesRegex(RuntimeError, "HTTPS"):
                redirect.redirect_request(None, None, 302, "Found", {}, url)

    def test_rejects_existing_directory_symlink(self) -> None:
        external = Path(self.sandbox.name) / "external"
        external.mkdir()
        (external / "untouched").write_bytes(b"keep")
        (self.root / "third_party").mkdir()
        try:
            self.destination.symlink_to(external, target_is_directory=True)
        except OSError:
            self.skipTest("Directory symlinks are unavailable on this platform")
        with self.assertRaisesRegex(RuntimeError, "link|reparse"):
            self.invoke()
        self.assertEqual((external / "untouched").read_bytes(), b"keep")

    def test_materializes_paths_longer_than_windows_legacy_limit(self) -> None:
        name = "deep/" * 55 + "source.py"
        self.files[name] = b"# Long upstream path\n"
        self.write_archive()
        self.extraction["archive_sha256"] = digest(self.archive.read_bytes())
        self.extraction["files"] = len(self.files)
        self.extraction["files_sha256"][name] = digest(self.files[name])
        self.write_metadata()
        if os.name == "nt":
            self.addCleanup(shutil.rmtree, "\\\\?\\" + str(self.root), True)
        self.assertEqual(self.invoke()["status"], "materialized")
        self.assertEqual(self.invoke()["status"], "verified")


if __name__ == "__main__":
    unittest.main()
