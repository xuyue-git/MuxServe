"""Deployment and local settings files: TOML form, local overlay, `muxserve config dump`."""
from contextlib import redirect_stderr, redirect_stdout
import io
import os
from pathlib import Path
import sys
import tomllib
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from muxserve.cli import main as cli  # noqa: E402
from muxserve.config import config_to_toml, load_config, read_config_file  # noqa: E402

SHIPPED = sorted([*ROOT.glob("configs/*.toml"), *ROOT.glob("configs/xbench/*.toml"),
                  *ROOT.glob("configs/ablation/*.toml")])
LOCAL_TEMPLATE = ROOT / "configs/muxserve.example.toml"


class ConfigFileTests(unittest.TestCase):
    def test_every_shipped_deployment_loads_and_round_trips(self):
        deployments = [path for path in SHIPPED if path != LOCAL_TEMPLATE]
        self.assertEqual(len(deployments), 40)  # 3 configs/*.toml + 8 xbench + 29 ablation steps
        for path in deployments:
            with self.subTest(path=path.relative_to(ROOT).as_posix()):
                raw = read_config_file(path)
                self.assertEqual(tomllib.loads(config_to_toml(raw)), raw)
                load_config(path, require_assets=False)

    @unittest.skipUnless(os.name == "posix", "the template's lock directory is a POSIX path")
    def test_local_template_serves_every_deployment(self):
        for name in ("configs/a100_open_loop.toml", "configs/xbench/pair-7b-13b.toml",
                     "configs/xbench/isolated-30b.toml"):
            config = load_config(ROOT / name, require_assets=False, local=LOCAL_TEMPLATE)
            self.assertEqual(config.asset_lock_dir, "/absolute/path/to/your/personal/lock-directory")

    def test_config_dump_prints_the_effective_deployment(self):
        out = io.StringIO()
        with redirect_stdout(out):
            self.assertEqual(cli(["config", "dump", str(ROOT / "configs/a100_open_loop.toml")]), 0)
        text = out.getvalue()
        dumped = tomllib.loads(text)
        config = load_config(ROOT / "configs/a100_open_loop.toml", require_assets=False)
        self.assertIn(f"fingerprint:    {config.fingerprint}", text)
        self.assertEqual(dumped["kv_memory_fraction"], 0.05061)
        self.assertEqual(dumped["rpc_persistent_connections"], True)  # defaults are shown
        self.assertEqual([model["model_id"] for model in dumped["models"]], ["llama7b", "llama13b", "llama30b"])

    def test_load_point_options_set_the_power_law_rates(self):
        path = ROOT / "configs/a100_open_loop.toml"
        from_file = load_config(path, require_assets=False)
        # alpha 0.7 / scale 1.0 is the file's own point: same rates, same fingerprint.
        same = load_config(path, require_assets=False, arrival=(0.7, 1.0, 2.0))
        self.assertEqual(same.policy_arrival_rates, from_file.policy_arrival_rates)
        self.assertEqual(same.fingerprint, from_file.fingerprint)
        other = dict(load_config(path, require_assets=False, arrival=(2.1, 2.0, 2.0)).policy_arrival_rates)
        self.assertEqual(other["llama7b"], 4.0)
        self.assertAlmostEqual(other["llama13b"], 4.0 / 2 ** 2.1)
        self.assertAlmostEqual(other["llama30b"], 4.0 / 3 ** 2.1)
        out = io.StringIO()
        with redirect_stdout(out):
            cli(["config", "dump", str(path), "--alpha", "2.1", "--rate-scale", "2.0"])
        self.assertIn("load point:     alpha 2.1, rate scale 2, base rate 2 req/s", out.getvalue())
        self.assertEqual(dict(tomllib.loads(out.getvalue())["policy_arrival_rates"]), other)
        with self.assertRaises(SystemExit), redirect_stdout(io.StringIO()), \
                redirect_stderr(io.StringIO()):
            cli(["config", "dump", str(path), "--alpha", "2.1"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
