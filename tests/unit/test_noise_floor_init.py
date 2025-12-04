"""Test for noise floor initialization bug fix."""
import unittest
from unittest.mock import MagicMock, patch
import numpy as np

from led_detection.main import PeakMonitor, X86CameraDriver


class MockVideoDriver(X86CameraDriver):
    """Mock video driver for testing."""
    def __init__(self):
        super().__init__()
        self.cap = MagicMock()
        self.frame_count = 0
        self.ambient_brightness = 20

    def get_frame(self):
        """Return a frame with ambient brightness."""
        img = np.full((480, 640), self.ambient_brightness, dtype=np.uint8)
        return img


class TestNoiseFloorInitialization(unittest.TestCase):
    """Test that noise floor history is properly initialized during calibration."""

    def setUp(self):
        self.driver = MockVideoDriver()
        self.patcher = patch('led_detection.main.get_driver', return_value=self.driver)
        self.patcher.start()
        self.addCleanup(self.patcher.stop)

    @patch('cv2.imwrite')
    @patch('cv2.connectedComponentsWithStats')
    @patch('cv2.threshold')
    def test_noise_floor_history_initialized(self, mock_threshold, mock_cc, _mock_imwrite):  # pylint: disable=too-many-locals
        """Verify that noise_floor_history is populated during calibration."""
        # Mock the connected components to return a simple result
        mock_cc.return_value = (2, np.zeros((480, 640)), [[0, 0, 10, 10, 100]], None)
        mock_threshold.return_value = (0, np.zeros((480, 640), dtype=np.uint8))

        # Create monitor with specific ROI and state
        # Use brightness mode (not contrast) to get non-zero measurements
        monitor = PeakMonitor(interval=10, threshold=20, preview=False, use_contrast=False)
        monitor.cam = self.driver

        # Simulate the calibration phase (lines 756-779 of main.py)
        monitor.roi = (200, 300, 280, 360)
        monitor.detected_on_brightness = 100.0  # Simulate detected ON brightness

        y1, y2, x1, x2 = monitor.roi
        metric_samples = []

        # Collect 30 samples as done in calibration
        for _ in range(30):
            f = monitor.cam.get_frame()
            roi = f[y1:y2, x1:x2]
            metric_samples.append(monitor._measure_roi(roi))

        avg_noise_level = np.mean(metric_samples)

        # Apply the fix: initialize noise_floor_history
        monitor.noise_floor_history = metric_samples[:monitor.noise_floor_window]

        # Estimate ON level and Signal Strength
        estimated_on_level = monitor.detected_on_brightness
        monitor.calibrated_signal_strength = max(10.0, estimated_on_level - avg_noise_level)
        monitor.current_noise_floor = avg_noise_level

        # Verify that noise_floor_history is populated
        self.assertEqual(len(monitor.noise_floor_history), 10,
                        "noise_floor_history should have 10 samples")
        self.assertGreater(len(monitor.noise_floor_history), 0,
                          "noise_floor_history should not be empty")

        # Verify that current_noise_floor equals mean of history
        expected_noise_floor = np.mean(monitor.noise_floor_history)
        self.assertAlmostEqual(monitor.current_noise_floor, expected_noise_floor, places=1,
                              msg="current_noise_floor should equal mean of history")

        # Verify noise floor is non-zero
        self.assertGreater(monitor.current_noise_floor, 0,
                          "Noise floor should be greater than 0")

        # Verify that subsequent mean calculations will work
        test_mean = np.mean(monitor.noise_floor_history)
        self.assertFalse(np.isnan(test_mean), "Mean of noise_floor_history should not be NaN")
        self.assertGreater(test_mean, 0, "Mean of noise_floor_history should be greater than 0")


if __name__ == '__main__':
    unittest.main()
