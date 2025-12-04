
import unittest
from unittest.mock import MagicMock, patch
import sys

import cv2
import numpy as np

from led_detection.main import X86CameraDriver, PeakMonitor, main

class TestX86CameraDriver(unittest.TestCase):
    @patch('cv2.VideoCapture')
    def test_init_linux(self, mock_capture):
        with patch('platform.system', return_value='Linux'):
            _ = X86CameraDriver(idx=0)
            mock_capture.assert_called_with(0, cv2.CAP_V4L2)

    @patch('cv2.VideoCapture')
    def test_init_windows(self, mock_capture):
        with patch('platform.system', return_value='Windows'), \
             patch('led_detection.main.SYS_OS', 'Windows'):
            _ = X86CameraDriver(idx=0)
            # On Linux cv2 might not have CAP_DSHOW, so check
            if not hasattr(cv2, 'CAP_DSHOW'):
                cv2.CAP_DSHOW = 700 # Mock it

            mock_capture.assert_called_with(0, cv2.CAP_DSHOW)

    @patch('cv2.VideoCapture')
    def test_start_linux(self, mock_capture):
        mock_cap_instance = mock_capture.return_value
        with patch('platform.system', return_value='Linux'):
            driver = X86CameraDriver()
            with patch('time.sleep'): # Speed up test
                driver.start()

            # Verify Linux specific settings
            # Auto exposure set to 0.25 multiple times
            self.assertTrue(mock_cap_instance.set.called)
            # Check for specific calls if needed, but general verification is good

    @patch('cv2.VideoCapture')
    def test_start_windows(self, mock_capture):
        mock_cap_instance = mock_capture.return_value
        with patch('platform.system', return_value='Windows'), \
             patch('led_detection.main.SYS_OS', 'Windows'):
            driver = X86CameraDriver()
            with patch('time.sleep'):
                driver.start()

            # Verify Windows specific settings
            mock_cap_instance.set.assert_any_call(cv2.CAP_PROP_AUTO_EXPOSURE, 0)
            mock_cap_instance.set.assert_any_call(cv2.CAP_PROP_EXPOSURE, -13)

    @patch('cv2.VideoCapture')
    def test_get_frame(self, mock_capture):
        mock_cap_instance = mock_capture.return_value
        # Mock successful frame read
        fake_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        mock_cap_instance.read.return_value = (True, fake_frame)

        driver = X86CameraDriver()
        frame = driver.get_frame()

        self.assertEqual(frame.shape, (480, 640)) # Converted to gray
        self.assertEqual(frame.dtype, np.uint8)

        # Mock failed frame read
        mock_cap_instance.read.return_value = (False, None)
        frame = driver.get_frame()
        self.assertEqual(frame.shape, (480, 640)) # Returns black frame
        self.assertEqual(np.sum(frame), 0)

    @patch('cv2.VideoCapture')
    def test_log_camera_info(self, mock_capture):
        mock_cap_instance = mock_capture.return_value
        mock_cap_instance.getBackendName.return_value = "V4L2"
        mock_cap_instance.get.return_value = 0

        driver = X86CameraDriver()
        with self.assertLogs(level='INFO') as cm:
            driver.log_camera_info()
        self.assertTrue(any("Camera Information" in o for o in cm.output))

class TestPeakMonitorMethods(unittest.TestCase):
    def setUp(self):
        self.mock_driver = MagicMock(spec=X86CameraDriver)
        self.mock_driver.get_frame.return_value = np.zeros((480, 640), dtype=np.uint8)
        self.mock_driver.cap = MagicMock() # Ensure cap exists for calibrate_exposure
        self.mock_driver.cap.get.return_value = 100.0 # Ensure get returns float

        # Patch get_driver to return our mock
        self.patcher = patch('led_detection.main.get_driver', return_value=self.mock_driver)
        self.patcher.start()

        self.monitor = PeakMonitor(interval=1, threshold=10)
        self.monitor.cam = self.mock_driver # Explicitly set

    def tearDown(self):
        self.patcher.stop()

    @patch('cv2.imshow')
    @patch('cv2.waitKey')
    @patch('time.time')
    def test_aim_camera(self, mock_time, mock_wait, mock_imshow):
        # Simulate time passing: start, loop once, end
        # time.time() is called:
        # 1. In aim_camera to set end time
        # 2. In while loop condition
        # 3. In putText
        # 4. In while loop condition (to exit)

        start_time = 1000.0
        mock_time.side_effect = [start_time, start_time, start_time, start_time + 6.0]

        self.monitor.aim_camera()

        self.assertTrue(mock_imshow.called)
        self.assertTrue(mock_wait.called)

    @patch('led_detection.main.PeakMonitor.wait_for_signal')
    @patch('led_detection.main.PeakMonitor.wait_for_led_off')
    @patch('led_detection.main.PeakMonitor.calibrate_exposure')
    def test_run_loop_structure(self, mock_cal, mock_wait_off, mock_wait_signal):
        # Verify the sequence of calls in run()
        # We need to break the infinite loop.
        # 1. wait_for_signal returns True (found signal)
        # 2. wait_for_led_off called
        # 3. calibrate_exposure called
        # 4. Calibration loop (mock get_frame)
        # 5. Monitoring loop (infinite) -> Raise exception to break

        mock_wait_signal.return_value = True

        # Mock get_frame to raise exception after a few calls to break the monitoring loop
        self.mock_driver.get_frame.side_effect = [
            np.zeros((480, 640), dtype=np.uint8), # Calibration 1
            *([np.zeros((480, 640), dtype=np.uint8)] * 40), # Calibration loop + some monitoring
            KeyboardInterrupt("Break loop")
        ]

        self.monitor.roi = (0, 10, 0, 10)
        self.monitor.preview = False

        try:
            self.monitor.run()
        except KeyboardInterrupt:
            pass

        self.assertTrue(mock_wait_signal.called)
        self.assertTrue(mock_wait_off.called)
        self.assertTrue(mock_cal.called)

class TestMain(unittest.TestCase):
    @patch('led_detection.main.PeakMonitor')
    @patch('led_detection.main.setup_logging')
    def test_main_args(self, mock_setup, mock_monitor_cls):
        test_args = ['prog', '--interval', '5', '--threshold', '20', '--debug']
        with patch.object(sys, 'argv', test_args):
            # Mock the monitor instance to prevent run() from blocking
            mock_monitor_instance = mock_monitor_cls.return_value

            main()

            # Verify arguments were parsed and passed to PeakMonitor
            mock_monitor_cls.assert_called_once()
            call_args = mock_monitor_cls.call_args

            # Check positional args
            self.assertEqual(call_args[0][0], 5.0) # interval
            self.assertEqual(call_args[0][1], 20.0) # threshold
            self.assertEqual(call_args[0][2], False) # preview (default)

            # Check kwargs
            self.assertEqual(call_args[1]['use_contrast'], True)
            self.assertEqual(call_args[1]['adaptive_roi'], True)

            # Verify run was called
            mock_monitor_instance.run.assert_called_once()

            # Verify logging setup
            mock_setup.assert_called_with(True)
