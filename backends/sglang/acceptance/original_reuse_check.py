"""The SGLang backend calls original MuxServe code, without vLLM or Torch.

- Importing the backend's controller/policy/config resolves ``muxserve`` to
  this repository and imports no vLLM, Torch or transformers module.
- CreditLedger allocates through one original SMResource per TP rank with the
  configured overload threshold; with the overloaded-part choice fixed it is
  step-for-step identical to the ledger it replaced.
- KV pool bytes from the shared original formula equal the previous
  arithmetic exactly (capacity and rounding unchanged).
- The original package keeps its behaviour: lazy entry points, the
  JobConfig default dtype, CacheResource.get_num_gpu_blocks.
"""
from __future__ import annotations

from pathlib import Path
import random
import subprocess
import sys
import textwrap
import unittest

ROOT = Path(__file__).resolve().parents[1]
REPOSITORY = ROOT.parents[1]
sys.path.insert(0, str(ROOT))

from muxserve_sglang.contracts import IntegrationError  # noqa: E402

HEAVY = ("vllm", "torch", "transformers")


def run_isolated(code: str) -> str:
    """Run code in a fresh interpreter (-I: no PYTHONPATH, no cwd on sys.path)
    that refuses to import any HEAVY package."""
    guard = textwrap.dedent(f"""
        import sys, importlib.abc
        class Refuse(importlib.abc.MetaPathFinder):
            def find_spec(self, name, path=None, target=None):
                if name.split('.')[0] in {HEAVY!r}:
                    raise ImportError('HEAVY_IMPORT ' + name)
                return None
        sys.meta_path.insert(0, Refuse())
        sys.path.insert(0, {str(ROOT)!r})
    """)
    result = subprocess.run([sys.executable, "-I", "-B", "-c", guard + textwrap.dedent(code)],
                            capture_output=True, text=True, timeout=120)
    if result.returncode != 0:
        raise AssertionError(result.stdout + result.stderr)
    return result.stdout


class ReferenceLedger:
    """The CreditLedger implementation this change replaced (verbatim)."""
    def __init__(self, multiplier=3, ranks=(0,)):
        if multiplier not in (1, 3) or not ranks or len(set(ranks)) != len(ranks):
            raise ValueError('Invalid software-credit profile')
        self.capacity, self.threshold = 10, multiplier
        self.refs = {r: [0] * 10 for r in ranks}
        self.free = {r: list(range(10)) for r in ranks}
        self.unoverloaded = {r: list(range(10)) for r in ranks}
        self.leases = {}

    @property
    def available(self):
        return min(len(self.free[r]) + min(self.threshold, len(self.unoverloaded[r])) for r in self.refs)

    def can_allocate(self, credits):
        return type(credits) is int and 0 < credits <= 10 and credits <= self.available

    def acquire(self, batch_id, model_id, stage, credits):
        signature = (model_id, stage, credits)
        if batch_id in self.leases:
            if self.leases[batch_id][0] != signature:
                raise IntegrationError('IDENTITY_CONFLICT', 'Credit lease changed contents')
            return True
        if any(item[0][:2] == signature[:2] for item in self.leases.values()) or not self.can_allocate(credits):
            return False
        acquired = {}
        for rank in self.refs:
            parts = []
            for _ in range(credits):
                part = self.free[rank].pop() if self.free[rank] else self.unoverloaded[rank].pop(0)
                self.refs[rank][part] += 1
                parts.append(part)
            acquired[rank] = tuple(parts)
        self.leases[batch_id] = (signature, acquired)
        return True

    def release(self, batch_id):
        lease = self.leases.pop(batch_id, None)
        if lease is None:
            return
        for rank, parts in lease[1].items():
            for part in parts:
                self.refs[rank][part] -= 1
                if self.refs[rank][part] == 0:
                    self.free[rank].append(part)
                if self.refs[rank][part] <= 1 and part not in self.unoverloaded[rank]:
                    self.unoverloaded[rank].append(part)


def previous_kv_bytes(fraction, total_memory_bytes):
    """kv_bytes_from_fraction before it called the original code (verbatim)."""
    head_size, block_size, element_size = 128, 32, 2
    available = round(fraction * total_memory_bytes) - 1
    block_mem = element_size * (head_size * block_size + head_size * block_size)
    blocks = (available // block_mem // 128) * 128
    return blocks * block_mem


class Imports(unittest.TestCase):
    def test_backend_uses_this_repositorys_muxserve_without_vllm_or_torch(self):
        out = run_isolated(f"""
            import muxserve_sglang, muxserve_sglang.controller, muxserve_sglang.policy as policy
            from muxserve_sglang.config import kv_bytes_from_fraction
            import muxserve, muxserve.muxsched.resource as resource
            assert muxserve_sglang.original_muxserve_path() == muxserve_sglang.REPOSITORY_ROOT / 'muxserve'
            assert str(muxserve_sglang.REPOSITORY_ROOT) == {str(REPOSITORY)!r}
            assert policy.SMResource is resource.SMResource
            assert kv_bytes_from_fraction(0.05, 42_505_273_344) > 0
            loaded = sorted(m for m in sys.modules if m.split('.')[0] in {HEAVY!r})
            assert not loaded, loaded
            assert 'muxserve.arg_utils' not in sys.modules and 'muxserve.muxsched.scheduler' not in sys.modules
            print('IMPORT_OK', muxserve.__file__)
        """)
        self.assertIn("IMPORT_OK " + str(REPOSITORY / "muxserve"), out)

    def test_original_entry_points_stay_lazy_and_importable_by_name(self):
        out = run_isolated("""
            import muxserve_sglang  # puts the repository root on the import path
            import muxserve
            assert 'muxserve.arg_utils' not in sys.modules
            # Stand-ins for the vLLM-based modules: `from muxserve import X`
            # must resolve X from exactly the original module, on first use.
            import types
            for name, module in (('MuxScheduler', 'muxserve.muxsched.scheduler'),
                                 ('MuxServeArgs', 'muxserve.arg_utils')):
                stub = types.ModuleType(module)
                setattr(stub, name, 'from ' + module)
                sys.modules[module] = stub
            from muxserve import MuxScheduler, MuxServeArgs
            assert MuxScheduler == 'from muxserve.muxsched.scheduler', MuxScheduler
            assert MuxServeArgs == 'from muxserve.arg_utils', MuxServeArgs
            try:
                muxserve.no_such_name
            except AttributeError:
                pass
            else:
                raise AssertionError('unknown attribute')
            print('LAZY_OK')
        """)
        self.assertIn("LAZY_OK", out)

    def test_a_copy_outside_the_repository_fails_clearly(self):
        # A regular (non-editable) install copies the package away from the
        # repository; the policy import must say why instead of a bare
        # "No module named 'muxserve'".
        import shutil
        import tempfile
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "site" / "lib" / "pkgs"
            shutil.copytree(ROOT / "muxserve_sglang", target / "muxserve_sglang",
                            ignore=shutil.ignore_patterns("__pycache__", "*.so"))
            code = textwrap.dedent(f"""
                import sys
                sys.path.insert(0, {str(target)!r})
                try:
                    import muxserve_sglang.policy
                except ImportError as exc:
                    assert "needs this repository's muxserve package" in str(exc), exc
                    print('CLEAR_ERROR_OK')
                else:
                    raise AssertionError('imported without the original package')
            """)
            result = subprocess.run([sys.executable, "-I", "-B", "-c", code], capture_output=True,
                                    text=True, timeout=120, cwd=directory)
        self.assertIn("CLEAR_ERROR_OK", result.stdout, result.stdout + result.stderr)

    def test_jobconfig_default_dtype_is_still_float16(self):
        out = run_isolated("""
            import types
            fake = types.ModuleType('torch'); fake.float16 = 'torch.float16'
            sys.meta_path.pop(0); sys.modules['torch'] = fake
            import muxserve_sglang  # puts the repository root on the import path
            from muxserve.config import JobConfig
            args = dict(name='m', model='p', pipeline_parallel_size=1, tensor_parallel_size=1,
                        placement=[[0]], mps_percentage=[100], max_num_seqs=1)
            assert JobConfig(**args).model_dtype == 'torch.float16'
            assert JobConfig(model_dtype=None, **args).model_dtype is None
            assert JobConfig(model_dtype='bf16', **args).model_dtype == 'bf16'
            print('DTYPE_OK')
        """)
        self.assertIn("DTYPE_OK", out)


class Ledger(unittest.TestCase):
    def setUp(self):
        from muxserve_sglang.policy import CreditLedger
        self.CreditLedger = CreditLedger

    def test_one_original_smresource_per_rank_with_the_configured_threshold(self):
        from muxserve.muxsched.resource import SMResource
        for multiplier in (1, 3):
            ledger = self.CreditLedger(multiplier, (0, 1, 2, 3))
            self.assertEqual(sorted(ledger.resources), [0, 1, 2, 3])
            for res in ledger.resources.values():
                self.assertIs(type(res), SMResource)
                self.assertEqual(res.overload_threshold, multiplier)  # never SMResource's default 4
        with self.assertRaises(ValueError):
            self.CreditLedger(4, (0,))

    def test_identical_to_the_replaced_ledger_when_the_overload_pick_is_fixed(self):
        # The original picks an overloaded part with random.choice; the replaced
        # ledger took the first unoverloaded part. Fixing the choice to the first
        # element makes both algorithms comparable step by step.
        original_choice = random.choice
        random.choice = lambda items: items[0]
        try:
            rng = random.Random(20261107)
            for trial in range(300):
                multiplier = rng.choice((1, 3))
                ranks = tuple(range(rng.choice((1, 2, 4))))
                new, old = self.CreditLedger(multiplier, ranks), ReferenceLedger(multiplier, ranks)
                live = []
                for step in range(60):
                    if live and rng.random() < 0.4:
                        batch = live.pop(rng.randrange(len(live)))
                        new.release(batch), old.release(batch)
                    else:
                        batch = f"b{trial}-{step}"
                        model, stage = rng.choice("abc"), rng.choice("PD")
                        credits = rng.choice((10, 4, 3, 1, 0, 11))
                        self.assertEqual(new.can_allocate(credits), old.can_allocate(credits))
                        got = new.acquire(batch, model, stage, credits)
                        self.assertEqual(got, old.acquire(batch, model, stage, credits))
                        if got:
                            live.append(batch)
                            # A repeated grant is a no-op; changed contents conflict.
                            self.assertTrue(new.acquire(batch, model, stage, credits))
                            with self.assertRaises(IntegrationError):
                                new.acquire(batch, model, stage, credits + 1)
                    self.assertEqual(new.available, old.available)
                    self.assertEqual(new.refs, old.refs)
                    self.assertEqual(new.leases, old.leases)
                    for rank in ranks:
                        res = new.resources[rank]
                        self.assertEqual([p.id for p in res.free_sms], old.free[rank])
                        self.assertEqual(res.unoverloaded_sms, old.unoverloaded[rank])
                for batch in live:
                    new.release(batch)
                self.assertEqual(new.available, 10 + multiplier)  # 10 free + threshold
                self.assertTrue(all(not any(refs) for refs in new.refs.values()))
        finally:
            random.choice = original_choice

    def test_paired_point_profile_with_random_overload_picks(self):
        # P needs all 10 parts, each D40 decode 4 (threshold 3): whatever part
        # random.choice overloads, admission matches the replaced ledger.
        for seed in range(50):
            random.seed(seed)
            new, old = self.CreditLedger(3, (0, 1, 2, 3)), ReferenceLedger(3, (0, 1, 2, 3))
            for ledger in (new, old):
                self.assertTrue(ledger.acquire("d1", "a", "D", 4))
                self.assertTrue(ledger.acquire("d2", "b", "D", 4))
                self.assertTrue(ledger.acquire("d3", "c", "D", 4))  # 2 free + 2 overloaded
                self.assertFalse(ledger.acquire("p", "a", "P", 10))
            self.assertEqual(new.available, old.available)
            for batch in ("d1", "d2", "d3"):
                new.release(batch), old.release(batch)
            self.assertTrue(new.acquire("p", "a", "P", 10))
            self.assertTrue(old.acquire("p", "a", "P", 10))
            self.assertEqual(new.available, old.available)


class KVCapacity(unittest.TestCase):
    def test_kv_bytes_are_unchanged(self):
        from muxserve_sglang.config import kv_bytes_from_fraction
        totals = (42_505_273_344, 42_949_672_960, 34_089_730_048, 85_899_345_920, 17_179_869_184, 81_920 * 1024**2)
        fractions = [round(0.001 * i, 3) for i in range(1, 951)] + [0.05, 0.4, 0.9]
        for total in totals:
            for fraction in fractions:
                got = kv_bytes_from_fraction(fraction, total)
                self.assertIs(type(got), int)
                self.assertEqual(got, previous_kv_bytes(fraction, total), (fraction, total))

    def test_original_cache_resource_formula_is_unchanged(self):
        import numpy as np
        import muxserve.muxsched.resource as resource
        old_get_gpu_memory = resource.get_gpu_memory
        try:
            for total in (42_505_273_344, 85_899_345_920):
                for block_size in (16, 32):
                    for utilization in (0.04, 0.05, 0.5, 0.9):
                        resource.get_gpu_memory = lambda gpu=0, total=total: total
                        cache = object.__new__(resource.CacheResource)
                        cache.config = type("C", (), {"gpu_memory_utilization": utilization})()
                        cache.block_size, cache.head_size = block_size, 128
                        # Verbatim body of the previous get_num_gpu_blocks.
                        available = round(utilization * total) - 1
                        block_mem = 2 * (np.prod(cache.get_key_block_shape()) +
                                         np.prod(cache.get_value_block_shape()))
                        expected = (available // block_mem // 128) * 128
                        got = cache.get_num_gpu_blocks()
                        self.assertEqual(got, expected)
                        self.assertEqual(type(got), type(expected))
        finally:
            resource.get_gpu_memory = old_get_gpu_memory


if __name__ == "__main__":
    unittest.main(verbosity=2)
