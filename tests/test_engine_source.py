"""Read the separately checked-out engine without falling back to a stale copy."""
from __future__ import annotations

import importlib.util
from pathlib import Path
import tempfile
import unittest


MODULE = Path(__file__).resolve().parents[1] / "muxserve/engine_source.py"


class EngineSourceTests(unittest.TestCase):
    def setUp(self):
        self.assertTrue(MODULE.is_file(), "External engine source locator is not implemented")
        spec = importlib.util.spec_from_file_location("engine_source_under_test", MODULE)
        self.module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.module)
        self.sandbox = tempfile.TemporaryDirectory()
        self.addCleanup(self.sandbox.cleanup)
        self.main = Path(self.sandbox.name) / "MuxServe with spaces"
        self.main.mkdir()
        self.engine = self.main.parent / "MuxServe-SGLang-Engine"
        (self.engine / "sglang/srt").mkdir(parents=True)
        (self.engine / "sglang/srt/worker.py").write_text("external\n", encoding="utf-8")

    def test_reads_sibling_engine_even_when_embedded_copy_is_present(self):
        embedded = self.main / "third_party/sglang/sglang/srt"
        embedded.mkdir(parents=True)
        (embedded / "worker.py").write_text("stale\n", encoding="utf-8")
        path = self.module.engine_file("sglang/srt/worker.py", root=self.main)
        self.assertEqual(path.read_text(encoding="utf-8"), "external\n")
        self.assertEqual(self.module.engine_root(root=self.main), self.engine)

    def test_rejects_traversal_and_absolute_names(self):
        for name in ("../secret", "python/../secret", "/etc/passwd", "C:/secret", "python\\worker.py"):
            with self.subTest(name=name), self.assertRaises(ValueError):
                self.module.engine_file(name, root=self.main)

    def test_missing_external_file_has_no_embedded_fallback(self):
        embedded = self.main / "third_party/sglang/sglang/srt"
        embedded.mkdir(parents=True)
        (embedded / "missing.py").write_text("stale\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "engine.*missing|not a regular file"):
            self.module.engine_file("sglang/srt/missing.py", root=self.main)


if __name__ == "__main__":
    unittest.main()
