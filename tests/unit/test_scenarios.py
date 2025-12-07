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
        self.pre_frame_callback = None
        self.noise_pattern = None

    def get_frame(self):
        # Allow dynamic updates via callback
        if hasattr(self, 'pre_frame_callback'):
            self.pre_frame_callback(self)

        if self.background_pattern is not None:
            img = self.background_pattern.copy()
        else:
            img = np.full((self.h, self.w), self.ambient_brightness, dtype=np.uint8)

        # Handle Noise Patterns
        noise_pattern = getattr(self, 'noise_pattern', None)
        if noise_pattern == 'high_contrast':
            # Oscillating high contrast spot (simulating clutter/sun)
            if self.frame_count % 2 == 0:
                cv2.rectangle(img, (280, 200), (300, 220), 255, -1)
            else:
                cv2.rectangle(img, (280, 200), (300, 220), self.ambient_brightness + 12, -1)

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

    def test_scenario_adaptive_threshold(self):
        """
        Test that threshold adapts to ambient light transitions (Dark -> Bright -> Dark).
        """
        # Start dark
        self.driver.ambient_brightness = 20
        self.driver.noise_level = 2

        monitor = PeakMonitor(interval=10, threshold=20, adaptive_exposure=False, log_saturation=True)
        monitor.roi = (0, 480, 0, 640)
        # Simulate initial calibration
        monitor.current_noise_floor = 20.0
        monitor.calibrated_signal_strength = 50.0
        monitor.thresh_bright = 45.0 # 20 + 25

        # Define dynamic behavior
        def update_ambient(driver):
            if driver.frame_count == 50:
                # Transition to Bright
                driver.ambient_brightness = 150
            elif driver.frame_count == 100:
                # Transition back to Dark
                driver.ambient_brightness = 20

        self.driver.pre_frame_callback = update_ambient

        # Run for 150 frames (50 dark, 50 bright, 50 dark)
        # We need to ensure we don't exit early due to "no signal" if we mock wait_for_signal?
        # run_monitor bypasses wait_for_signal.
        # It runs the loop which calls wait_for_signal... wait.
        # run_monitor patches wait_for_signal to return True.
        # So it thinks signal is ALWAYS detected?
        # If wait_for_signal returns True, it enters the monitoring loop?
        # Yes, line 813 in main.py "while True".

        # But wait_for_signal returning True means it detected a PULSE.
        # The adaptive logic we want to test runs when signal is NOT active (OFF state).
        # In the monitoring loop:
        # if is_active: ... else: ... (adaptive logic)
        # is_active = val > thresh_bright.

        # If we have no LED, val = ambient.
        # Initial ambient 20 < 45. So is_active=False.
        # Then ambient 150 > 45. So is_active=True (Stuck ON).
        # It should adapt after 10s (300 frames? No, 10s is 300 frames at 30fps).
        # We are running 30fps.
        # 50 frames is 1.6s. Not enough to trigger 10s timeout.
        # We need more frames or mock time faster?
        # run_monitor mocks time based on frame count (30fps).

        # Let's adjust frame counts to be meaningful.
        # Dark: 20 frames (stabilize).
        # Bright: Need > 10s to force update if stuck.
        # 30fps * 11s = 330 frames.

        def update_ambient_long(driver):
            if driver.frame_count == 20:
                driver.ambient_brightness = 150
            elif driver.frame_count == 400: # After ~12s of bright
                driver.ambient_brightness = 20

        self.driver.pre_frame_callback = update_ambient_long

        self.run_monitor(monitor, frames=450)

        # Verification
        # 1. At end (Dark), threshold should be back to low.
        # Just checking the final state might be enough?
        # Or we can check logs/history if we really want.

        self.assertLess(monitor.current_noise_floor, 40.0,
                        f"Noise floor should return to low levels (was {monitor.current_noise_floor})")
        self.assertLess(monitor.thresh_bright, 60.0)

    def test_scenario_oscillating_background(self):
        """
        Test that threshold adapts to oscillating high contrast background (clutter).
        """
        self.driver.ambient_brightness = 20
        self.driver.noise_pattern = 'high_contrast' # Oscillates 20 <-> 255 (if drawn)

        monitor = PeakMonitor(interval=10, threshold=20, adaptive_exposure=False)
        monitor.roi = (0, 480, 0, 640)
        # Verify it updates even if signal is "stuck" high due to oscillation
        # High contrast spot: 255. Threshold: ~45.
        # It will be ON (255 > 45).
        # Next frame: 32 (20+12). 32 < 45. OFF.
        # So it's not stuck ON. It's fickering ON/OFF.
        # If flickering, it goes to ELSE block (is_active=False) every other frame.
        # It should adapt quickly because it sees "OFF" states.

        self.run_monitor(monitor, frames=100)

        # Threshold should rise to ignore the 32s?
        # Wait, if 32 is "OFF" (background), it should raise noise floor to ~32.
        # If 255 is "ON" (noise), it might ignore it as pulse.
        # But if it thinks 32 is noise floor...
        # Mean of (32, 32, 32...) is 32.
        # Threshold = 32 + 25 = 57.
        # 255 > 57 (ON). 32 < 57 (OFF).
        # So it stays flickering.

        # However, reproduction_complex said: "Threshold should have risen"
        # "Val oscillates 12 <-> ~235... Thr -> 17.75".
        # If Val=12 (OFF) and Val=235 (ON).
        # It sees enough OFF frames to update noise floor?
        # Yes, every other frame is OFF.
        # So noise floor should track the "OFF" values (the valleys).

        # The test expects it to NOT be stuck in state where everything is ON.
        # And threshold should be reasonable.

        self.assertGreater(monitor.current_noise_floor, 10.0)
        # Should not have exploded to 255
        self.assertLess(monitor.current_noise_floor, 100.0)

if __name__ == '__main__':
    unittest.main()
