"""Portable asset paths use one protected resolver; no lab alias bypass."""
import errno
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from muxserve import config
from muxserve.contracts import IntegrationError


@unittest.skipUnless(os.name == 'posix', 'Linux path and symlink contract')
class AssetPathTests(unittest.TestCase):
    def test_explicit_alias_resolves_to_canonical_asset_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory).resolve()
            target, alias = base/'model', base/'alias'
            target.mkdir()
            (target/'weights.bin').write_bytes(b'fixture')
            alias.symlink_to(target, target_is_directory=True)
            # A historical alias table must not bypass canonical path checking.
            with patch.object(config, '_READ_ONLY_SHARED_MODELS', ((alias,target),), create=True):
                self.assertEqual(config.read_asset_path(alias/'weights.bin'), target/'weights.bin')

    def test_prohibited_path_is_rejected_before_readlink(self):
        called = []
        def readlink(path):
            path = Path(path)
            self.assertFalse(path == Path('/home/incoming') or Path('/home/incoming') in path.parents)
            called.append(path)
            raise OSError(errno.EINVAL, 'not a link')
        with patch.object(config.os, 'readlink', side_effect=readlink):
            with self.assertRaises(IntegrationError) as caught:
                config.read_asset_path('/home/incoming/model/weights.bin')
        self.assertEqual(caught.exception.code, 'FORBIDDEN_PATH')
        self.assertTrue(called)

    def test_alias_and_canonical_root_share_external_lock(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory).resolve()
            target, alias, locks = base/'model', base/'alias', base/'locks'
            target.mkdir()
            alias.symlink_to(target, target_is_directory=True)
            self.assertEqual(config.asset_lock_path(alias,str(locks)), config.asset_lock_path(target,str(locks)))
            with self.assertRaises(IntegrationError) as caught:
                config.asset_lock_path(alias,str(target/'locks'))
            self.assertEqual(caught.exception.code, 'INVALID_CONFIG')


if __name__ == '__main__':
    unittest.main()
