import unittest
from unittest.mock import MagicMock, patch
import numpy as np
import cv2
from led_detection.main import PeakMonitor, X86CameraDriver

class MockVideoDriver(X86CameraDriver):
    def __init__(self, width=640, height=480):  # pylint: disable=super-init-not-called
        self.cap = MagicMock()
        self.w = width
        self.h = height
        self.frame_count = 0
        self.leds = []
        self.background_pattern = None

    def get_frame(self):
        img = np.zeros((self.h, self.w), dtype=np.uint8)
        # Add noise
        noise = np.random.normal(0, 2, (self.h, self.w))
        img = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)

        # Draw LEDs
        for led in self.leds:
            if led["start"] <= self.frame_count < led["start"] + led["duration"]:
                cv2.circle(img, led["pos"], led["radius"], led["brightness"], -1)

        self.frame_count += 1
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

class TestROIConsistencyMock(unittest.TestCase):
    def setUp(self):
        self.driver = MockVideoDriver()
        self.current_time = 0.0
        # self.patcher = patch('led_detection.main.get_driver', return_value=self.driver)
        # self.patcher.start()

    def tearDown(self):
        pass
        # self.patcher.stop()

    def test_roi_consistency_mock(self):
        monitor = PeakMonitor(interval=10, threshold=20, adaptive_exposure=False, log_saturation=False)
        monitor.cam = self.driver
        monitor.preview = False

        led_pos = (320, 240)
        centers = []

        # Mock time
        self.current_time = 1000.0

        def mock_time():
            return self.current_time

        def mock_sleep(duration):
            self.current_time += duration

        with patch('time.time', side_effect=mock_time), \
             patch('time.sleep', side_effect=mock_sleep), \
             patch('cv2.accumulateWeighted'):

            print("\nRunning Mock ROI Consistency Test (10 iterations)...")
            for i in range(10):
                self.driver.frame_count = 0
                self.driver.leds = [{
                    "pos": led_pos,
                    "radius": 5,
                    "brightness": 200,
                    "start": 20,
                    "duration": 300
                }]
                monitor.roi = None

                # Dwell time 2.0s is fine as we simulate enough frames (100 frames > 2.0s at 30fps)
                found = monitor.wait_for_signal(timeout=5.0, dwell_time=2.0)

                self.assertTrue(found, f"Failed iteration {i}")
                y1, y2, x1, x2 = monitor.roi
                cx = (x1 + x2) / 2
                cy = (y1 + y2) / 2
                centers.append((cx, cy))
                # print(f"Iter {i}: {cx},{cy}")

        centers = np.array(centers)
        std_dev = np.std(centers, axis=0)
        print(f"Mock Std Dev: {std_dev}")
        self.assertLess(np.max(std_dev), 2.0)

if __name__ == '__main__':
    unittest.main()
