"""Real CPU watchdog threads; no Torch, SGLang or GPU imports."""
from __future__ import annotations

import ast
import gc
import logging
from pathlib import Path
import sys
import threading
import time
from types import SimpleNamespace
import unittest
from unittest.mock import Mock
import weakref

ROOT = Path(__file__).resolve().parents[1]


def functions(path, names, namespace):
    tree = ast.parse((ROOT / path).read_text(encoding='utf-8'))
    body = [node for node in tree.body if getattr(node, 'name', None) in names]
    assert len(body) == len(names)
    exec(compile(ast.Module(body=body, type_ignores=[]), path, 'exec'), namespace)
    return namespace


def native_watchdog():
    namespace = dict(threading=threading, time=time, logger=logging.getLogger(__name__),
        psutil=SimpleNamespace(Process=lambda: SimpleNamespace(parent=lambda: Mock())),
        pyspy_dump_schedulers=Mock(), sys=sys, signal=SimpleNamespace(SIGQUIT=3))
    functions('third_party/sglang/python/sglang/srt/utils/watchdog.py', {'WatchdogRaw'}, namespace)
    functions('third_party/sglang/python/sglang/srt/managers/scheduler_components/invariant_checker.py',
              {'create_scheduler_watchdog'}, namespace)
    return namespace


class Holder:
    pass


class WatchdogTeardownChecks(unittest.TestCase):
    def test_native_callbacks_pin_model_and_pool_until_stop(self):
        namespace = native_watchdog()
        scheduler = Holder()
        scheduler.is_initializing, scheduler.cur_batch_for_debug, scheduler.forward_ct = False, None, 0
        scheduler.model, scheduler.pool = Holder(), Holder()
        references = [weakref.ref(value) for value in (scheduler, scheduler.model, scheduler.pool)]
        watchdog = namespace['create_scheduler_watchdog'](scheduler, watchdog_timeout=90)
        del scheduler
        gc.collect()
        self.assertTrue(all(reference() is not None for reference in references),
                        'Reproduction must retain native callbacks through a live thread')
        self.assertTrue(callable(getattr(watchdog, 'stop', None)), 'Native watchdog has no stop API')
        start = time.monotonic()
        watchdog.stop(timeout=1)
        self.assertLess(time.monotonic() - start, 1, 'Shutdown waited for the 45-second watchdog tick')
        self.assertFalse(watchdog._thread.is_alive())
        gc.collect()
        self.assertTrue(all(reference() is None for reference in references))
        watchdog.stop(timeout=1)  # Idempotent normal shutdown.

    def test_stop_does_not_release_callbacks_while_callback_is_running(self):
        namespace = native_watchdog()
        entered, unblock = threading.Event(), threading.Event()
        model = Holder()
        reference = weakref.ref(model)
        def active(value=model):
            entered.set()
            unblock.wait(3)
            return False
        watchdog = namespace['WatchdogRaw']('test', lambda: 0, active, 90)
        del model, active
        self.assertTrue(entered.wait(1))
        try:
            with self.assertRaises(RuntimeError):
                watchdog.stop(timeout=0.01)
            self.assertIsNotNone(reference())
        finally:
            unblock.set()
            if hasattr(watchdog, 'stop'):
                watchdog.stop(timeout=1)
        gc.collect()
        self.assertIsNone(reference())

    def test_normal_controlled_drain_stops_hard_and_soft_watchdogs(self):
        self.controlled_path(fail=False)

    def test_failed_controlled_run_keeps_monitoring_and_refuses_cleanup(self):
        self.controlled_path(fail=True)

    def test_monitor_timeout_still_runs_while_serving(self):
        namespace = native_watchdog()
        timeout_seen = threading.Event()
        namespace['pyspy_dump_schedulers'].side_effect = timeout_seen.set
        watchdog = namespace['WatchdogRaw']('test', lambda: 0, lambda: True, 0.02, soft=True)
        try:
            self.assertTrue(timeout_seen.wait(1), 'Monitoring was disabled before DRAIN')
        finally:
            watchdog.stop(timeout=1)
        watchdog.parent_process.send_signal.assert_not_called()

    def test_monitor_stop_failure_does_not_certify_drain(self):
        self.controlled_path(fail='stop')

    def test_native_channel_failure_does_not_certify_drain(self):
        self.controlled_path(fail='channels')

    def controlled_path(self, fail):
        from types import ModuleType
        from unittest.mock import patch
        context = SimpleNamespace()
        trace = []
        scheduler = SimpleNamespace(watchdog=SimpleNamespace(stop=lambda: trace.append('hard')),
            soft_watchdog=SimpleNamespace(stop=lambda: trace.append('soft')),
            tp_worker=SimpleNamespace(model_runner=SimpleNamespace(model=Holder(),
                token_to_kv_pool=Holder(), _muxserve_imported_pool=object())))
        if fail == 'stop':
            scheduler.watchdog.stop = Mock(side_effect=RuntimeError('watchdog still running'))
        runtime = ModuleType('muxserve_sglang.sglang_runtime')
        def run():
            trace.append('drain')
            if fail is True: raise RuntimeError('uncertain native failure')
        runtime.ControlledScheduler = lambda *args: SimpleNamespace(run=run)
        loader = ModuleType('muxserve_sglang.weight_loader')
        loader.record_runtime_evidence = lambda *args: None
        namespace = dict(__package__='muxserve_sglang', context=lambda: context,
                         close_native_channels=lambda scheduler: trace.append('native-channels'))
        if fail == 'channels':
            namespace['close_native_channels'] = Mock(side_effect=RuntimeError('native socket still open'))
        functions('muxserve_sglang/sglang_hooks.py', {'run_controlled_scheduler'}, namespace)
        with patch.dict(sys.modules, {'muxserve_sglang.sglang_runtime': runtime,
                                     'muxserve_sglang.weight_loader': loader}):
            if fail:
                with self.assertRaises(RuntimeError): namespace['run_controlled_scheduler'](scheduler)
                self.assertEqual(trace, ['drain', 'hard', 'soft'] if fail == 'channels' else ['drain'])
                self.assertFalse(context._drain_complete)
            else:
                namespace['run_controlled_scheduler'](scheduler)
                self.assertEqual(trace, ['drain', 'hard', 'soft', 'native-channels'])
                self.assertTrue(context._drain_complete)


if __name__ == '__main__': unittest.main(verbosity=2)
