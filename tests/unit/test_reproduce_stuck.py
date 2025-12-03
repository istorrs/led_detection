import unittest
from unittest.mock import MagicMock, patch
import numpy as np
from led_detection.main import PeakMonitor, X86CameraDriver

class TestStuckNoiseFloor(unittest.TestCase):
    def setUp(self):
        self.mock_driver = MagicMock(spec=X86CameraDriver)
        self.mock_driver.get_frame.return_value = np.zeros((480, 640), dtype=np.uint8)
        self.mock_driver.start = MagicMock()
        self.mock_driver.stop = MagicMock()
        self.mock_driver.cap = MagicMock()
        self.mock_driver.cap.get.return_value = 50.0

        self.patcher = patch('led_detection.main.get_driver', return_value=self.mock_driver)
        self.patcher.start()

    def tearDown(self):
        self.patcher.stop()

    def test_high_contrast_noise_floor_update(self):
        """Test that noise floor updates correctly even when background has high contrast."""
        # 1. Initialize with HIGH CONTRAST noise floor
        # User case: Contrast ~100.
        # We create a frame with median 0 but some pixels at 100.
        high_contrast_frame = np.zeros((480, 640), dtype=np.uint8)
        # Set 10% of pixels to 100 to ensure max is 100 but median is 0
        # Actually just one pixel is enough for max, but let's do a block
        high_contrast_frame[0:10, 0:10] = 100

        # 2. Slightly higher contrast (drift)
        drift_frame = np.zeros((480, 640), dtype=np.uint8)
        drift_frame[0:10, 0:10] = 102

        monitor = PeakMonitor(interval=10, threshold=20, adaptive_exposure=False, log_saturation=True)
        monitor.cam = self.mock_driver
        monitor.roi = (0, 480, 0, 640)

        frames = [high_contrast_frame] * 40 # Calibration
        frames.extend([drift_frame] * 100) # Drift
        frames.append(None)

        iter_frames = iter(frames)
        self.mock_driver.get_frame.side_effect = lambda: next(iter_frames)

        # Simulate weak signal
        monitor.detected_peak_strength = 10.0
        monitor.detected_on_brightness = 110.0

        with patch.object(monitor, 'wait_for_signal', return_value=True), \
             patch.object(monitor, 'wait_for_led_off', return_value=True), \
             patch.object(monitor, 'calibrate_exposure'):

            monitor.run()

        # Check if noise floor updated to 102
        # With the bug, it stays at 100 (or whatever calibration set it to)
        # With the fix, it should track to 102

        self.assertGreater(monitor.current_noise_floor, 101,
                          f"Noise floor should have tracked to 102, but was {monitor.current_noise_floor}")

if __name__ == '__main__':
    unittest.main()
