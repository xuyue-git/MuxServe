"""The event pump must not copy cumulative diagnostic history per tick."""
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'acceptance')]
from controller_check import Fixture
from muxserve_sglang.driver import ControllerDriver


class CopyTrap(dict):
    def __deepcopy__(self, memo):
        raise AssertionError('driver copied unused diagnostic history')


class DriverHistoryChecks(unittest.TestCase):
    def test_event_pump_ignores_full_diagnostics_at_history_capacity(self):
        fixture = Fixture()
        driver = ControllerDriver(fixture.controller, lambda event: None, schedule=False)
        fixture.controller.policy.adjustments.extend(CopyTrap(index=i) for i in range(500))
        with patch.object(fixture.controller, 'snapshot', side_effect=AssertionError('full snapshot in hot path')):
            for _ in range(3):
                driver.step()
        self.assertEqual(fixture.controller._tick_count, 3)

    def test_default_diagnostics_keep_complete_independent_history(self):
        fixture = Fixture()
        fixture.controller.policy.adjustments.extend({'limits': {'m': i}} for i in range(500))
        tick = fixture.controller.tick()
        full = fixture.controller.snapshot()
        self.assertEqual(len(tick['adjustments']), 500)
        self.assertEqual(tick['adjustments'], full['policy_adjustments'])
        tick['adjustments'][0]['limits']['m'] = -1
        full['policy_adjustments'][-1]['limits']['m'] = -1
        self.assertEqual(fixture.controller.policy.adjustments[0]['limits']['m'], 0)
        self.assertEqual(fixture.controller.policy.adjustments[-1]['limits']['m'], 499)

    def test_batch_progress_is_detached_and_retains_every_ready_rank(self):
        fixture = Fixture()
        fixture.submit()
        plan, batch = fixture.prepare()
        fixture.report('P', plan.batch_id, 'READY', {'reservation_id': batch['reservation_id']})
        progress = fixture.controller.batch_progress()
        self.assertEqual(progress['m'][plan.batch_id], fixture.controller.snapshot()['groups']['m']['batches'][plan.batch_id])
        progress['m'][plan.batch_id]['ready'].append(('D', 0))
        progress['m'][plan.batch_id]['status'] = 'RUNNING'
        again = fixture.controller.batch_progress()['m'][plan.batch_id]
        self.assertEqual(again['ready'], [('P', 0)])
        self.assertEqual(again['status'], 'PREPARING')

    def test_driver_still_waits_for_all_ready_and_aborts_when_paused(self):
        for draining in (False, True):
            fixture = Fixture()
            fixture.submit()
            plan, batch = fixture.prepare()
            driver = ControllerDriver(fixture.controller, lambda event: None)
            driver.pending.add(('m', plan.batch_id))
            payload = {'reservation_id': batch['reservation_id']}
            fixture.report('P', plan.batch_id, 'READY', payload)
            driver.step()
            self.assertEqual(fixture.controller.batch_progress()['m'][plan.batch_id]['status'], 'PREPARING')
            fixture.report('D', plan.batch_id, 'READY', payload)
            if draining:
                driver.pause()
            driver.step()
            status = fixture.controller.batch_progress()['m'][plan.batch_id]['status']
            self.assertEqual(status, 'REVOKING' if draining else 'RUNNING')
            if draining:
                for stage in ('P', 'D'):
                    fixture.report(stage, plan.batch_id, 'QUIESCED', {
                        'reservation_id': batch['reservation_id'], 'request_ids': ('r',)})
                self.assertEqual(fixture.controller.batch_progress()['m'][plan.batch_id]['status'], 'ABORTED')


if __name__ == '__main__':
    unittest.main(verbosity=2)
