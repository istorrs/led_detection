
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
        self.noise_pattern = None

    def get_frame(self):
        img = np.full((480, 640), self.ambient_brightness, dtype=np.uint8)

        # Add noise/contrast
        if self.noise_pattern == 'high_contrast':
            # Simulate the "sun" spot or high contrast clutter
            # Oscillating or static? Logs showed oscillation 12 <-> 170
            # Let's simulate oscillation by using frame_count
            self.frame_count += 1
            if self.frame_count % 2 == 0:
                # High peak
                cv2.rectangle(img, (280, 200), (300, 220), 255, -1)
            else:
                # Low valley (but still above ambient?)
                # If ambient is 20, and we want val=12... wait.
                # Contrast = Max - Median.
                # If Median is 20.
                # We want Contrast 12 -> Max 32.
                cv2.rectangle(img, (280, 200), (300, 220), self.ambient_brightness + 12, -1)

        if self.led_on:
            cv2.circle(
                img, (320, 240), 10,
                min(255, self.ambient_brightness + self.led_brightness), -1)

        return img

class TestReproductionComplex(unittest.TestCase):
    def setUp(self):
        self.driver = MockVideoDriver()
        self.patcher = patch('led_detection.main.get_driver', return_value=self.driver)
        self.patcher.start()
        self.addCleanup(self.patcher.stop)

    @patch('time.time')
    def test_dark_bright_dark_bright(self, mock_time):
        start_time = 1000.0
        current_time = start_time
        mock_time.return_value = current_time

        # 1. Initialize in Dark (Very Dark)
        # Logs showed Floor=1, Signal=11.5.
        # So ambient ~1? Or contrast ~1?
        self.driver.ambient_brightness = 5 # Low ambient

        monitor = PeakMonitor(interval=10, threshold=20, preview=False)
        monitor.cam = self.driver

        # Manually set calibrated values to match logs
        monitor.roi = (200, 300, 280, 360)
        monitor.current_noise_floor = 1.0
        monitor.calibrated_signal_strength = 11.5
        monitor.thresh_bright = 1.0 + (11.5 * 0.5) # 6.75 -> 7ish
        monitor.led_state = False
        monitor.last_exposure_adjust = start_time - 100
        monitor.min_while_on = 5.0  # Initialize to dark ambient value
        monitor.last_off_time = start_time  # Initialize to prevent override

        # Helper to simulate one frame processing
        def process_frame():
            nonlocal current_time
            mock_time.return_value = current_time

            frame = monitor.cam.get_frame()
            roi = frame[monitor.roi[0]:monitor.roi[1], monitor.roi[2]:monitor.roi[3]]
            val = monitor._measure_roi(roi)
            now = current_time

            is_active = val > monitor.thresh_bright

            if not hasattr(monitor, 'last_off_time'):
                monitor.last_off_time = now
                monitor.min_while_on = val

            if not is_active:
                monitor.last_off_time = now
                max_reasonable_floor = monitor.calibrated_signal_strength * 0.6

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
                    if monitor.noise_floor_skip_count > 200:
                        monitor.noise_floor_history.append(val)
                        if len(monitor.noise_floor_history) > monitor.noise_floor_window:
                            monitor.noise_floor_history.pop(0)
                        monitor.current_noise_floor = np.mean(monitor.noise_floor_history)
                        monitor.thresh_bright = (monitor.current_noise_floor +
                                             (monitor.calibrated_signal_strength * 0.5))
                        monitor.noise_floor_skip_count = 0
            else:
                time_since_off = now - monitor.last_off_time
                if time_since_off < 0.1:
                    monitor.min_while_on = val
                else:
                    monitor.min_while_on = min(monitor.min_while_on, val)

                max_reasonable_floor = monitor.calibrated_signal_strength * 0.6
                force_update = time_since_off > 10.0

                if (time_since_off > 5.0 and
                    monitor.min_while_on > (monitor.current_noise_floor +
                              monitor.calibrated_signal_strength * 0.3) and
                    (monitor.min_while_on < max_reasonable_floor or force_update)):

                    print(f"DEBUG: Updating! Force={force_update}, "
                          f"Min={monitor.min_while_on}, Val={val}")
                    update_val = monitor.min_while_on
                    if force_update and val > monitor.min_while_on:
                        # Blend min and current to pull average up
                        update_val = (monitor.min_while_on + val) / 2.0

                    monitor.noise_floor_history.append(update_val)
                    if len(monitor.noise_floor_history) > monitor.noise_floor_window:
                        monitor.noise_floor_history.pop(0)

                    monitor.current_noise_floor = np.mean(monitor.noise_floor_history)
                    monitor.thresh_bright = (monitor.current_noise_floor +
                                         (monitor.calibrated_signal_strength * 0.5))
                    monitor.min_while_on = val
                else:
                    if i % 10 == 0:
                        check_val = (monitor.current_noise_floor +
                                     monitor.calibrated_signal_strength * 0.3)
                        print(f"DEBUG: No Update. T={time_since_off:.1f}, "
                              f"Min={monitor.min_while_on}, "
                              f"Floor={monitor.current_noise_floor}, "
                              f"Check={check_val}")

            return is_active, val, monitor.thresh_bright

        # 2. Verify Dark Phase
        is_active, val, thr = process_frame()
        self.assertFalse(is_active)
        self.assertLess(thr, 10.0)

        # 3. Transition to Bright (High Contrast / Oscillating)
        self.driver.ambient_brightness = 20  # Still low ambient,
        # but high contrast spot
        self.driver.noise_pattern = 'high_contrast'

        # Run for > 10s to trigger force update
        stuck_count = 0
        adapted = False

        print("\nStarting Bright Phase...")
        for i in range(40):  # 40 frames, 0.5s each -> 20s
            current_time += 0.5
            is_active, val, thr = process_frame()
            print(f"Frame {i}: Val={val}, Thr={thr}, Active={is_active}")

            if is_active:
                stuck_count += 1
            else:
                adapted = True
                # Once adapted, it might flicker ON/OFF if oscillation is huge
                # But we want to see if threshold rises.

        print(f"Stuck count: {stuck_count}")

        # It should adapt.
        # Val oscillates 12 <-> ~235.
        # Min while on -> 12.
        # Floor -> 12.
        # Thr -> 12 + 5.75 = 17.75.
        # When Val is 12, 12 < 17.75 -> OFF.

        self.assertTrue(adapted, "Should have adapted to high contrast")
        self.assertGreater(monitor.thresh_bright, 15.0, "Threshold should have risen")


if __name__ == '__main__':
    unittest.main()
