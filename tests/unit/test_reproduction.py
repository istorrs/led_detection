
import unittest
from unittest.mock import MagicMock, patch

import cv2
import numpy as np

from led_detection.main import PeakMonitor, X86CameraDriver

class MockVideoDriver(X86CameraDriver):
    def __init__(self):
        super().__init__()
        self.cap = MagicMock()
        self.frame_count = 0
        self.ambient_brightness = 20
        self.led_brightness = 0
        self.led_on = False

    def get_frame(self):
        # Create background with noise to simulate contrast
        img = np.full((480, 640), self.ambient_brightness, dtype=np.uint8)

        # Add noise (subtract to keep within range, or just use noise as texture)
        # Let's make it simple: base brightness + random noise
        # But we want to control the contrast.
        # Contrast = max - median.
        # If we want contrast X, we can have a pixel with val+X.

        # Let's just use a gradient or a fixed pattern to be deterministic.
        # Draw a "sun" or bright spot that is NOT the LED (e.g. in corner)
        # This creates high max, but median might stay low.

        # Draw a bright spot in the corner (within ROI?)
        # ROI is (200, 300, 280, 360).
        # Let's put a bright static reflection inside the ROI.
        # Scale spot brightness with ambient to simulate contrast
        # increasing with light
        spot_brightness = int(self.ambient_brightness * 0.5)
        cv2.rectangle(
            img, (280, 200), (300, 220),
            min(255, self.ambient_brightness + spot_brightness), -1)

        if self.led_on:
            # Add LED pulse in center (320, 240)
            cv2.circle(img, (320, 240), 10, (self.ambient_brightness + self.led_brightness), -1)

        return img


class TestReproduction(unittest.TestCase):
    def setUp(self):
        self.driver = MockVideoDriver()
        self.patcher = patch('led_detection.main.get_driver', return_value=self.driver)
        self.patcher.start()
        self.addCleanup(self.patcher.stop)

    @patch('time.time')
    def test_dark_to_light_transition(self, mock_time):
        # Start time
        start_time = 1000.0
        current_time = start_time
        mock_time.return_value = current_time

        # 1. Initialize in Dark
        self.driver.ambient_brightness = 20
        monitor = PeakMonitor(interval=10, threshold=20, preview=False)
        monitor.cam = self.driver

        # Initialize state as run() does
        monitor.roi = (200, 300, 280, 360) # Larger ROI to include background
        monitor.current_noise_floor = 20.0
        monitor.calibrated_signal_strength = 50.0
        monitor.thresh_bright = 20.0 + (50.0 * 0.5) # 45.0
        monitor.led_state = False
        monitor.last_exposure_adjust = start_time - 100 # Long ago

        # Helper to simulate one frame processing
        def process_frame():
            nonlocal current_time
            mock_time.return_value = current_time

            frame = monitor.cam.get_frame()
            roi = frame[monitor.roi[0]:monitor.roi[1], monitor.roi[2]:monitor.roi[3]]
            val = monitor._measure_roi(roi)
            now = current_time

            # Copy-paste relevant logic from run() loop for reproduction
            # We are testing the adaptive threshold logic

            is_active = val > monitor.thresh_bright

            # We need to track last_off_time for the fix to work
            if not hasattr(monitor, 'last_off_time'):
                monitor.last_off_time = now
                monitor.min_while_on = val

            # Adaptive Threshold Update Logic (simplified from main.py)
            if not is_active:
                monitor.last_off_time = now

                # In main.py there is a check for time_since_exposure_adjust > 3.0
                # We assume no exposure adjust happened recently

                max_reasonable_floor = monitor.calibrated_signal_strength * 0.6 # 30.0

                if val < max_reasonable_floor or len(monitor.noise_floor_history) == 0:
                    monitor.noise_floor_history.append(val)
                    if len(monitor.noise_floor_history) > monitor.noise_floor_window:
                        monitor.noise_floor_history.pop(0)

                    monitor.current_noise_floor = np.mean(monitor.noise_floor_history)
                    monitor.thresh_bright = (monitor.current_noise_floor +
                                           (monitor.calibrated_signal_strength * 0.5))
                    if monitor.current_noise_floor < monitor.calibrated_signal_strength * 0.3:
                        monitor.noise_floor_skip_count = 0
                else:
                    monitor.noise_floor_skip_count += 1

                    # FALLBACK logic from main.py
                    if monitor.noise_floor_skip_count > 200:
                        monitor.noise_floor_history.append(val)
                        if len(monitor.noise_floor_history) > monitor.noise_floor_window:
                            monitor.noise_floor_history.pop(0)
                        monitor.current_noise_floor = np.mean(monitor.noise_floor_history)
                        monitor.thresh_bright = (monitor.current_noise_floor +
                                             (monitor.calibrated_signal_strength * 0.5))
                        monitor.noise_floor_skip_count = 0
            else:
                # LED is ON
                time_since_off = now - monitor.last_off_time

                if time_since_off < 0.1:
                    monitor.min_while_on = val
                else:
                    monitor.min_while_on = min(monitor.min_while_on, val)

                max_reasonable_floor = monitor.calibrated_signal_strength * 0.6

                # The FIX: Force update if stuck ON for > 10s
                force_update = time_since_off > 10.0

                if (time_since_off > 5.0 and
                    monitor.min_while_on > (monitor.current_noise_floor +
                              monitor.calibrated_signal_strength * 0.3) and
                    (monitor.min_while_on < max_reasonable_floor or force_update)):

                    monitor.noise_floor_history.append(monitor.min_while_on)
                    if len(monitor.noise_floor_history) > monitor.noise_floor_window:
                        monitor.noise_floor_history.pop(0)

                    monitor.current_noise_floor = np.mean(monitor.noise_floor_history)
                    monitor.thresh_bright = (monitor.current_noise_floor +
                                         (monitor.calibrated_signal_strength * 0.5))
                    monitor.min_while_on = val

            return is_active, val, monitor.thresh_bright

        # 2. Verify Dark Detection
        # Pulse ON
        self.driver.led_on = True
        self.driver.led_brightness = 100 # Val = 120
        is_active, val, thr = process_frame()
        self.assertTrue(is_active, f"Should detect pulse in dark (Val={val}, Thr={thr})")

        # Pulse OFF
        self.driver.led_on = False
        current_time += 1.0
        is_active, val, thr = process_frame()
        self.assertFalse(is_active, "Should be OFF")

        # 3. Transition to Bright
        self.driver.ambient_brightness = 150

        # Run for 20 seconds (20 frames, 1s each)
        # It should be stuck ON initially, but then adapt after 10s.

        stuck_count = 0
        adapted = False

        for _ in range(20):
            current_time += 1.0
            is_active, val, thr = process_frame()
            if is_active:
                stuck_count += 1
            else:
                adapted = True
                break

        print(f"Stuck count: {stuck_count}")

        # Verify that it adapted
        self.assertTrue(adapted, "Should have adapted to bright ambient")
        self.assertFalse(is_active, "Should be OFF after adaptation")
        self.assertGreater(monitor.thresh_bright, 60.0, "Threshold should have risen")

        # 4. Verify Pulse Detection in Bright
        # Now that it adapted, a pulse should be detected if it's bright enough.
        # Pulse adds 100.
        # Ambient 150 + Spot 75 = 225.
        # Pulse -> 325 (clipped 255).
        # Contrast might be lost if clipped.
        # Let's see if 100 brightness is enough on top of 150.
        # Max 255. Median 150. Contrast 105.
        # Threshold should be around 75 + 25 = 100.
        # 105 > 100. Should detect!

        self.driver.led_on = True
        current_time += 1.0
        is_active, val, thr = process_frame()
        self.assertTrue(is_active, f"Should detect pulse in bright (Val={val}, Thr={thr})")

        # 5. Transition back to Dark
        self.driver.led_on = False
        self.driver.ambient_brightness = 20

        # Run for a few frames to allow adaptation
        # Threshold should drop.
        # Current threshold is high (~100).
        # New noise floor will be 20.
        # It should adapt quickly if not stuck.

        adapted_down = False
        for _ in range(20):
            current_time += 1.0
            is_active, val, thr = process_frame()
            if thr < 40.0: # Should drop back to near original ~35-45
                adapted_down = True
                break

        self.assertTrue(adapted_down,
                        f"Threshold should drop back to dark levels (Current: {thr})")
        self.assertFalse(is_active, "Should be OFF in dark")


if __name__ == '__main__':
    unittest.main()
