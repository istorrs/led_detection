
import unittest
from unittest.mock import MagicMock, patch
import cv2
import numpy as np
from led_detection.main import PeakMonitor, X86CameraDriver

class MockVideoDriver(X86CameraDriver):
    def __init__(self):
        super().__init__()
        self.cap = MagicMock()
        self.cap.get.return_value = 0.0
        self.w = 640
        self.h = 480
        self.frame_count = 0
        self.leds = []
        self.fps = 30.0

    def get_frame(self):
        self.frame_count += 1
        img = np.zeros((self.h, self.w), dtype=np.uint8)

        # Draw LEDs
        for led in self.leds:
            if led["start"] <= self.frame_count < led["start"] + led["duration"]:
                cv2.circle(img, led["pos"], 5, 255, -1)

        return img

    def configure_flash(self, start, duration):
        self.leds.append({
            "pos": (320, 240),
            "start": start,
            "duration": duration
        })

class TestPulseAccuracy(unittest.TestCase):
    def setUp(self):
        self.driver = MockVideoDriver()
        self.patcher = patch('led_detection.main.get_driver', return_value=self.driver)
        self.patcher.start()

    def tearDown(self):
        self.patcher.stop()

    def test_duration_calculation(self):
        # Setup: 5 frame flash at 30fps
        # Duration should be ~166ms
        fps = 30.0
        frame_time = 1.0 / fps
        flash_frames = 5
        expected_duration = flash_frames * frame_time * 1000 # ms

        self.driver.configure_flash(start=10, duration=flash_frames)

        # Mock time.time to advance with frames
        # We need to coordinate get_frame calls with time updates
        # PeakMonitor calls get_frame() then time.time()

        start_time = 1000.0

        def mock_time():
            # Return time corresponding to current frame count
            return start_time + (self.driver.frame_count * frame_time)

        monitor = PeakMonitor(interval=10, threshold=20, preview=False)

        detected_duration = [0]

        def on_pulse(_ts, duration, _gap):
            detected_duration[0] = duration

        with patch('time.time', side_effect=mock_time):
            # Run monitoring for enough frames
            # We need to call start_monitoring, but break out of it
            # We can use max_pulses=1

            # We need to bypass calibration and wait_for_signal loops if we call run()
            # Or just call start_monitoring directly.
            # But start_monitoring does calibration first.

            # Let's just manually feed frames to simulate the loop logic if needed,
            # or rely on start_monitoring working with our mock.

            # We need to ensure start_monitoring exits.
            # It exits if get_frame returns None.

            # Limit mock driver to N frames
            original_get_frame = self.driver.get_frame
            def limited_get_frame():
                if self.driver.frame_count > 30:
                    return None
                return original_get_frame()
            self.driver.get_frame = limited_get_frame

            # Skip calibration by mocking _measure_roi to return 0 initially?
            # Calibration runs for 30 frames.
            # We need more frames.

            # Let's just set flash to start after calibration (frame 30)
            self.driver.leds = []
            self.driver.configure_flash(start=40, duration=flash_frames)

            # Increase limit
            def limited_get_frame_2():
                if self.driver.frame_count > 60:
                    return None
                return original_get_frame()
            self.driver.get_frame = limited_get_frame_2

            monitor.roi = (0, 480, 0, 640) # Full screen ROI to skip detection
            # But start_monitoring uses ROI.
            # We need to ensure it detects the signal.

            # We can set monitor.roi manually.

            monitor.start_monitoring(max_pulses=1, on_pulse_callback=on_pulse)

        print(f"Expected: {expected_duration:.1f}ms")
        print(f"Detected: {detected_duration[0]:.1f}ms")

        # Current implementation likely returns (N-1)*interval
        # 5 frames -> 4 intervals = 133ms
        # Expected = 166ms

        # Assert that it is ACCURATE (within 10ms)
        self.assertAlmostEqual(detected_duration[0], expected_duration, delta=10, msg="Duration should be accurate with fix")

if __name__ == '__main__':
    unittest.main()
