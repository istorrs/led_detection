import unittest
from unittest.mock import MagicMock, patch
import time
import numpy as np
import cv2
from led_detection.main import PeakMonitor, X86CameraDriver

class MockVideoDriver(X86CameraDriver):
    def __init__(self, width=640, height=480):  # pylint: disable=super-init-not-called
        self.cap = MagicMock()
        self.w = width
        self.h = height
        self.frame_count = 0
        self.ambient_brightness = 50
        self.noise_level = 2
        self.leds = []
        self.background_pattern = None

    def get_frame(self):
        if self.background_pattern is not None:
            img = self.background_pattern.copy()
        else:
            img = np.full((self.h, self.w), self.ambient_brightness, dtype=np.uint8)

        # Add noise
        noise = np.random.normal(0, self.noise_level, (self.h, self.w))
        img = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)

        # Draw LEDs
        for led in self.leds:
            if led["start"] <= self.frame_count < led["start"] + led["duration"]:
                cv2.circle(img, led["pos"], led["radius"], led["brightness"], -1)

        self.frame_count += 1
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)


class TestScenarios(unittest.TestCase):
    def setUp(self):
        self.driver = MockVideoDriver()
        self.patcher = patch('led_detection.main.get_driver', return_value=self.driver)
        self.patcher.start()

    def tearDown(self):
        self.patcher.stop()

    def run_monitor(self, monitor, frames=100):
        # Mock time to advance
        start_time = time.time()
        def mock_time():
            return start_time + (self.driver.frame_count * 0.033) # 30fps

        with patch('time.time', side_effect=mock_time):
            # We need to break the infinite loop in run()
            # We can mock wait_for_signal to return False after N calls?
            # Or mock get_frame to return None after N frames?

            orig_get_frame = self.driver.get_frame
            def limited_get_frame():
                if self.driver.frame_count >= frames:
                    return None
                return orig_get_frame()

            self.driver.get_frame = limited_get_frame

            # We also need to bypass the initial "wait_for_signal" loop if we want to test the main loop
            # But the main loop IS wait_for_signal loop?
            # No, run() calls wait_for_signal() then wait_for_led_off() then enters monitoring loop.
            # If we want to test the monitoring loop (where exposure logic is), we need to get past startup.

            # Let's mock startup methods to return immediately
            with patch.object(monitor, 'wait_for_signal', return_value=True), \
                 patch.object(monitor, 'wait_for_led_off', return_value=True), \
                 patch.object(monitor, 'calibrate_exposure'):

                monitor.run()

    def test_scenario_high_contrast_background(self):
        """Test detection with high contrast background (noise floor update check)."""
        # High contrast background
        self.driver.background_pattern = np.zeros((480, 640), dtype=np.uint8)
        self.driver.background_pattern[0:10, 0:10] = 100 # Hot spot

        monitor = PeakMonitor(interval=10, threshold=20, adaptive_exposure=False, log_saturation=True)
        monitor.roi = (0, 480, 0, 640) # Set ROI
        # Simulate weak calibration
        monitor.detected_peak_strength = 10.0
        monitor.detected_on_brightness = 110.0

        self.run_monitor(monitor, frames=50)

        # Noise floor should have adapted to ~100
        self.assertGreater(monitor.current_noise_floor, 90)

    def test_scenario_dark_recovery(self):
        """Test exposure recovery from dark scene."""
        # Dark scene
        self.driver.ambient_brightness = 0
        self.driver.noise_level = 0

        monitor = PeakMonitor(interval=10, threshold=20, adaptive_exposure=True, log_saturation=True)
        monitor.roi = (0, 480, 0, 640) # Set ROI
        monitor.cam = self.driver
        # Mock cap.get/set
        self.driver.cap.get.return_value = 50.0

        self.run_monitor(monitor, frames=50)

        # Verify exposure increased
        sets = self.driver.cap.set.call_args_list
        exposure_sets = [call for call in sets if call[0][0] == cv2.CAP_PROP_EXPOSURE]
        self.assertTrue(any(call[0][1] > 50.0 for call in exposure_sets))

    def test_scenario_saturation_reduction(self):
        """Test exposure reduction on saturation."""
        # Saturated scene
        self.driver.ambient_brightness = 255
        self.driver.noise_level = 0

        monitor = PeakMonitor(interval=10, threshold=20, adaptive_exposure=True, log_saturation=True)
        monitor.roi = (0, 480, 0, 640) # Set ROI
        monitor.cam = self.driver
        self.driver.cap.get.return_value = 50.0

        self.run_monitor(monitor, frames=50)

        # Verify exposure decreased
        sets = self.driver.cap.set.call_args_list
        exposure_sets = [call for call in sets if call[0][0] == cv2.CAP_PROP_EXPOSURE]
        self.assertTrue(any(call[0][1] < 50.0 for call in exposure_sets))

if __name__ == '__main__':
    unittest.main()
