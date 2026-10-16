"""Hardware-independent regression checks; these do not validate RKNN inference."""
import importlib.util
from pathlib import Path
import queue
import threading
import types
import unittest
from unittest.mock import MagicMock, patch

import det


class RuntimeTests(unittest.TestCase):
    def setUp(self):
        det.stop_event.clear()
        while not det.worker_errors.empty():
            det.worker_errors.get_nowait()

    def tearDown(self):
        det.stop_event.clear()

    def load_base(self, factory):
        api = types.ModuleType('rknnlite.api')
        api.RKNNLite = factory
        spec = importlib.util.spec_from_file_location('base_under_test', Path(__file__).with_name('utils_base.py'))
        module = importlib.util.module_from_spec(spec)
        with patch.dict('sys.modules', {'cv2': MagicMock(), 'rknnlite': types.ModuleType('rknnlite'), 'rknnlite.api': api}):
            spec.loader.exec_module(module)
        return module

    def test_model_load_failure_releases_model(self):
        factory = MagicMock()
        model = factory.return_value
        model.load_rknn.return_value = -1
        base = self.load_base(factory)
        with self.assertRaisesRegex(RuntimeError, 'Cannot load'):
            base.load_rknn('bad.rknn')
        model.init_runtime.assert_not_called()
        model.release.assert_called_once()

    def test_partial_multi_core_failure_releases_all_models(self):
        first, second = MagicMock(), MagicMock()
        first.load_rknn.return_value = first.init_runtime.return_value = 0
        second.load_rknn.return_value = 0
        second.init_runtime.return_value = -1
        base = self.load_base(MagicMock(side_effect=[first, second]))
        with self.assertRaisesRegex(RuntimeError, 'Cannot initialize'):
            base.load_rknn_multi('model.rknn', [0, 1])
        first.release.assert_called_once()
        second.release.assert_called_once()

    def test_worker_failure_signals_shutdown(self):
        def fail():
            raise ValueError('inference failed')
        det.guarded(fail)
        self.assertTrue(det.stop_event.is_set())
        self.assertIsInstance(det.worker_errors.get_nowait(), ValueError)

    def test_full_queue_can_be_cancelled(self):
        pending = queue.Queue(maxsize=1)
        pending.put('occupied')
        worker = threading.Thread(target=det.send, args=(pending, None), daemon=True)
        worker.start()
        det.stop_event.set()
        worker.join(timeout=1)
        self.assertFalse(worker.is_alive())

    def test_renderer_eof_can_be_cancelled_when_display_is_full(self):
        base = self.load_base(MagicMock())
        incoming, display = queue.Queue(), queue.Queue(maxsize=1)
        incoming.put(None)
        display.put('occupied')
        stopped = threading.Event()
        args = [incoming] + [None] * 18 + [display]
        worker = threading.Thread(target=base.render_main, args=args,
                                  kwargs={'stop_event': stopped}, daemon=True)
        worker.start()
        stopped.set()
        worker.join(timeout=1)
        self.assertFalse(worker.is_alive())


if __name__ == '__main__':
    unittest.main()
