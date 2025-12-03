import unittest
from unittest.mock import MagicMock, patch
import numpy as np
import cv2
from led_detection.main import PeakMonitor, X86CameraDriver

class TestExposureLogic(unittest.TestCase):
    def setUp(self):
        self.mock_cap = MagicMock()
        self.mock_cap.get.return_value = 50.0 # Default exposure

        self.mock_driver = MagicMock(spec=X86CameraDriver)
        self.mock_driver.cap = self.mock_cap
        self.mock_driver.get_frame.return_value = np.zeros((480, 640), dtype=np.uint8)
        self.mock_driver.start = MagicMock()
        self.mock_driver.stop = MagicMock()

        self.patcher = patch('led_detection.main.get_driver', return_value=self.mock_driver)
        self.patcher.start()
        self.time_val = 0.0

    def tearDown(self):
        self.patcher.stop()

    def test_exposure_recovery_from_darkness(self):
        """Test that exposure increases when image is too dark."""
        # Simulate dark frames
        dark_frame = np.full((480, 640), 10, dtype=np.uint8) # Very dark

        monitor = PeakMonitor(interval=10, threshold=20, adaptive_exposure=True, log_saturation=True)
        monitor.cam = self.mock_driver # Ensure it uses our mock
        monitor.roi = (0, 480, 0, 640)

        # Run for enough frames to trigger logic (need > 30 frames for window, > 10 for check)
        frames = [dark_frame] * 60
        frames.append(None) # Stop loop

        iter_frames = iter(frames)
        self.mock_driver.get_frame.side_effect = lambda: next(iter_frames)

        # Mock time to advance
        self.time_val = 1000.0
        def advance_time():
            self.time_val += 0.1
            return self.time_val

        with patch('time.time', side_effect=advance_time):
            # We need to bypass aim_camera and autofocus to get to the loop quickly
            monitor.preview = False
            monitor.autofocus = False
            # Also bypass wait_for_signal loop?
            # wait_for_signal calls get_frame. It returns False if no signal.
            # run() loops wait_for_signal.
            # We need wait_for_signal to return False (no signal) but consume frames.
            # wait_for_signal consumes frames until timeout.
            # We want to test the loop inside run() that calls wait_for_signal?
            # Wait, the exposure logic is in run() loop, AFTER wait_for_signal returns?
            # No, looking at main.py:
            # The exposure logic is inside `monitor.run()`?
            # Let's check main.py again.


        # Re-reading main.py structure:
        # run() calls wait_for_signal()
        # wait_for_signal() has a loop "while time.time() - start_scan < scan_duration:"
        # BUT the continuous exposure logic seems to be in... wait, where is it?
        # I modified lines 830+ in main.py.
        # That code is inside... let's check the indentation.
        # It looks like it is inside `run`? No, wait.
        # Line 813: while True:
        # This loop seems to be... inside `PeakMonitor`?
        # Ah, I need to check which method the loop belongs to.
        # Line 733: def run(self):
        # Line 745: while True:
        #     if self.wait_for_signal(): break
        # This loop in run() is high level.

        # The code I modified (lines 801+) seems to be part of a method that was NOT shown in the first view?
        # Or is it `wait_for_signal`?
        # Let's check line 248: def wait_for_signal(self, timeout=None):
        # It has a loop: while time.time() - start_scan < scan_duration:

        # The code I modified starts at line 813 "while True:".
        # This looks like `monitor_loop` or something?
        # Wait, I missed the method definition for the code block starting at 801.
        # Let's look at the file content again.
        # Line 798: logging.info("--- MONITORING STARTED ---")
        # Line 800: last_pulse = time.time()
        # Line 813: while True:

        # This code block (798-1080) seems to be at the end of `run()`?
        # Let's check `run()` again.
        # Line 733: def run(self):
        # ...
        # Line 746: if self.wait_for_signal(): break
        # ...
        # Line 751: self.wait_for_led_off()
        # ...
        # Line 754: self.calibrate_exposure()
        # ...
        # Line 798: logging.info("--- MONITORING STARTED ---")

        # Yes, it is the continuation of `run()`.
        # So after `wait_for_signal` returns True (signal detected), and `wait_for_led_off` returns,
        # it enters the main monitoring loop (line 813).

        # So to test this, I need `wait_for_signal` to return True immediately,
        # and `wait_for_led_off` to return True immediately.
        # Then it enters the loop where my logic is.

        # I can mock `wait_for_signal` and `wait_for_led_off` to return True.

        with patch.object(monitor, 'wait_for_signal', return_value=True), \
             patch.object(monitor, 'wait_for_led_off', return_value=True), \
             patch.object(monitor, 'calibrate_exposure'), \
             patch('time.time', side_effect=advance_time):

            monitor.run()

        # Verify exposure increased
        sets = self.mock_cap.set.call_args_list
        exposure_sets = [call for call in sets if call[0][0] == cv2.CAP_PROP_EXPOSURE]

        self.assertTrue(len(exposure_sets) > 0, "Should have attempted to change exposure")
        increased = any(call[0][1] > 50.0 for call in exposure_sets)
        self.assertTrue(increased, f"Should have increased exposure. Sets: {exposure_sets}")

    def test_exposure_reduction_prevention(self):
        """Test that exposure is NOT reduced when image is dark but has high contrast."""
        # Simulate dark but high contrast frames (floor=0, but some hot pixels)
        frame = np.zeros((480, 640), dtype=np.uint8)
        # Add 10% hot pixels
        num_pixels = 480 * 640
        num_hot = int(num_pixels * 0.10)
        flat = frame.flatten()
        flat[:num_hot] = 255
        np.random.shuffle(flat)
        frame = flat.reshape((480, 640))

        # Verify stats
        self.assertEqual(np.median(frame), 0)
        self.assertEqual(np.percentile(frame, 95), 255)

        monitor = PeakMonitor(interval=10, threshold=20, adaptive_exposure=True, log_saturation=True)
        monitor.cam = self.mock_driver
        monitor.roi = (0, 480, 0, 640)

        frames = [frame] * 60
        frames.append(None)
        iter_frames = iter(frames)
        self.mock_driver.get_frame.side_effect = lambda: next(iter_frames)

        self.time_val = 1000.0
        def advance_time():
            self.time_val += 0.1
            return self.time_val

        with patch.object(monitor, 'wait_for_signal', return_value=True), \
             patch.object(monitor, 'wait_for_led_off', return_value=True), \
             patch.object(monitor, 'calibrate_exposure'), \
             patch('time.time', side_effect=advance_time):

            monitor.run()

        # Verify exposure was NOT decreased
        sets = self.mock_cap.set.call_args_list
        exposure_sets = [call for call in sets if call[0][0] == cv2.CAP_PROP_EXPOSURE]

        # We start at 50. We should not see any set < 50.
        decreased = any(call[0][1] < 50.0 for call in exposure_sets)
        self.assertFalse(decreased, f"Should NOT have decreased exposure. Sets: {exposure_sets}")

if __name__ == '__main__':
    unittest.main()
