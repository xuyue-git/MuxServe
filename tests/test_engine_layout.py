"""Static contracts for the cleaned source distribution; no GPU/model imports."""
from pathlib import Path
import ast
import json
import tempfile
try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib
import unittest
from unittest import mock
import sys

ROOT = Path(__file__).resolve().parents[1]
ENGINE = ROOT.parent / "MuxServe-SGLang-Engine"
sys.path.insert(0, str(ROOT))
from muxserve.config import read_config_file  # noqa: E402


class EngineLayoutTests(unittest.TestCase):
    def test_functional_directories_only(self):
        actual = {p.name for p in ENGINE.iterdir() if p.is_dir() and not p.name.startswith('.')}
        self.assertEqual(actual, {'csrc', 'docs', 'examples', 'tests', 'sglang'})
        self.assertTrue((ENGINE / 'sglang/srt/managers/scheduler.py').is_file())

    def test_package_identity_and_native_source_distribution(self):
        self.assertTrue((ENGINE / 'pyproject.toml').is_file(), 'Root package metadata not migrated')
        data = tomllib.loads((ENGINE / 'pyproject.toml').read_text(encoding='utf-8'))
        self.assertEqual(data['project']['name'], 'sglang')
        self.assertEqual(data['project']['version'], '0.5.20')
        self.assertEqual(data['tool']['setuptools']['packages']['find']['include'], ['sglang*'])
        self.assertIn('graft csrc', (ENGINE / 'MANIFEST.in').read_text())
        self.assertFalse((ENGINE / 'sglang/multimodal_gen').exists())

    def test_native_source_paths_resolve_after_relocation(self):
        checks = [('setup.py', '_RUST_WORKSPACE_DIR', ENGINE / 'csrc/rust', {'_PYTHON_DIR': ENGINE}),
                  ('sglang/srt/rust_extensions/loader.py', '_RUST_WORKSPACE', ENGINE / 'csrc/rust', {}),
                  ('sglang/srt/mem_cache/rust_tree_core/extension.py', '_CRATE_DIR', ENGINE / 'csrc/rust/sglang-radix-tree', {})]
        for relative, name, expected, extra in checks:
            path = ENGINE / relative
            self.assertTrue(path.is_file(), relative)
            tree = ast.parse(path.read_text(encoding='utf-8'))
            assignments = [n for n in tree.body if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == name for t in n.targets)]
            self.assertEqual(len(assignments), 1, relative)
            value = eval(compile(ast.Expression(assignments[0].value), relative, 'eval'), {'Path': Path, '__file__': str(path), **extra})
            self.assertEqual(value, expected)
            self.assertTrue(value.is_dir())
        self.assertTrue((ENGINE / 'csrc/proto/sglang/runtime/v1/sglang.proto').is_file())

    def test_unsupported_tensor_parallelism_is_rejected_at_config_boundary(self):
        from muxserve.config import load_config
        from muxserve.contracts import IntegrationError
        original = read_config_file(ROOT / 'configs/sglang_a100.toml')
        for tp in (1, 2):
            with self.subTest(tp=tp), tempfile.TemporaryDirectory() as directory:
                original['tp_size'] = tp
                path = Path(directory) / 'config.json'
                path.write_text(json.dumps(original), encoding='utf-8')
                with self.assertRaisesRegex(IntegrationError, 'TP4'):
                    load_config(path, require_assets=False)

    def test_preflight_rejects_a_mixed_vllm_environment_without_importing_it(self):
        from muxserve import preflight
        from muxserve.contracts import IntegrationError
        def distribution(name, search):
            return {'version': '0.2.1'} if name == 'vllm' else None
        with mock.patch.object(preflight, '_distribution', side_effect=distribution):
            with self.assertRaisesRegex(IntegrationError, 'vLLM'):
                preflight.inspect_software()


if __name__ == '__main__':
    unittest.main()
