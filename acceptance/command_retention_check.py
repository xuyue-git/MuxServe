"""Retained large commands are immutable wire snapshots; polling stays detached."""
from collections import OrderedDict
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'acceptance')]
from muxserve.flexstore import ipc
from muxserve.contracts import IntegrationError
from command_snapshot_check import command, runtime_method
from controller_check import Fixture


class RetentionChecks(unittest.TestCase):
    def snapshot(self, value):
        self.assertTrue(hasattr(ipc, 'CommandSnapshot'), 'Immutable command retention missing')
        return ipc.CommandSnapshot.capture(value)

    def test_publication_captures_all_ranks_aliases_and_detached_reads(self):
        value = command()
        value['batch_id'] = 'batch'
        expected = ipc.copy_message(value)
        frozen = self.snapshot(value)
        original = value['requests'][0]['reservation_view']['block_tables']
        original[0][0][0][0] = 99
        one, two = frozen.detach(), frozen.detach()
        self.assertEqual(one, expected)
        self.assertEqual(one['requests'][0]['reservation_view']['block_tables'][0][0][0][0], 0)
        self.assertEqual(set(one['requests'][0]['reservation_view']['block_tables']), {0,1,2,3})
        self.assertIs(one['requests'][0]['rank_block_table'], one['requests'][0]['reservation_view']['block_tables'][0])
        one['requests'][0]['rank_block_table'][0][0][0] = 88
        self.assertEqual(two['requests'][0]['rank_block_table'][0][0][0], 0)
        with patch.object(type(frozen), 'detach', side_effect=AssertionError('Retention metadata decoded full KV tables')):
            self.assertEqual(frozen['kind'], 'PREPARE')
            self.assertEqual(frozen.get('batch_id'), 'batch')
            self.assertTrue(frozen.references('r'))
            self.assertFalse(frozen.references('other'))

    def test_controller_retains_bytes_and_poll_and_lifecycle_use_full_values(self):
        fixture = Fixture()
        group = fixture.controller.groups['m']
        fixture.controller._command(group, ('P', 0), 'PREPARE', command())
        retained = next(iter(group.registered[('P',0)].commands.values()))
        self.assertIsInstance(retained, getattr(ipc, 'CommandSnapshot', type(None)))
        first = fixture.controller.poll_commands(fixture.group, 'P', 0, 1001)
        self.assertIs(type(first[0]), dict)
        self.assertTrue(fixture.controller._command_refs_request(group, retained, 'r'))
        first[0]['requests'][0]['reservation_view']['block_tables'][3][0][0][0] = -1
        second = fixture.controller.poll_commands(fixture.group, 'P', 0, 1001)
        self.assertEqual(second[0]['requests'][0]['reservation_view']['block_tables'][3][0][0][0], 30)

    def test_worker_replay_compares_every_rank_and_is_independent_of_prepared(self):
        def require(ok, message, code=None):
            if not ok: raise IntegrationError(code or 'INVALID', message)
        from copy import deepcopy
        namespace = dict(require=require, IntegrationError=IntegrationError, deepcopy=deepcopy,
            copy_message=ipc.copy_message, CommandSnapshot=getattr(ipc, 'CommandSnapshot', None))
        accept = runtime_method('_accept', namespace)
        worker = SimpleNamespace(ctx=SimpleNamespace(group_id='g', stage='P', model=SimpleNamespace(model_id='m')),
            rank=0, cursor=0, commands=OrderedDict(), prepared={})
        received = []
        worker._prepare = lambda value: received.append(ipc.copy_message(value))
        value = command()
        accept(worker, value)
        self.assertIsInstance(worker.commands[1], getattr(ipc, 'CommandSnapshot', type(None)))
        accept(worker, value)
        self.assertEqual(len(received), 1)
        value['requests'][0]['reservation_view']['block_tables'][3][0][0][0] = -1
        with self.assertRaises(IntegrationError): accept(worker, value)
        self.assertEqual(received[0]['requests'][0]['reservation_view']['block_tables'][3][0][0][0], 30)

    def test_retention_keeps_size_bound_and_all_request_reference_forms(self):
        value = dict(kind='HANDOFF', batch_id='b', request_id='one', request_ids=['two'],
                     requests=[{'request_id':'three'}])
        frozen = self.snapshot(value)
        self.assertTrue(all(frozen.references(r) for r in ('one','two','three')))
        value['request_ids'].append('four')
        self.assertFalse(frozen.references('four'))
        with patch.object(ipc, '_MAX_MESSAGE_BYTES', 32), self.assertRaises(IntegrationError):
            self.snapshot(command())


if __name__ == '__main__': unittest.main(verbosity=2)
