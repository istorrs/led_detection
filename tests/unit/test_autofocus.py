
import unittest
from unittest.mock import MagicMock, patch
import numpy as np
import cv2
from led_detection.main import PeakMonitor, X86CameraDriver

class MockCameraDriver(X86CameraDriver):
    def __init__(self, peak_focus=100, peak_width=50):
        super().__init__()
        self.cap = MagicMock()
        self.peak_focus = peak_focus
        self.peak_width = peak_width
        self.current_focus = 0

        # Mock cap.get/set for focus
        def get_prop(prop):
            if prop == cv2.CAP_PROP_FOCUS:
                return self.current_focus
            return 0

        def set_prop(prop, val):
            if prop == cv2.CAP_PROP_FOCUS:
                self.current_focus = val
                return True
            return True

        self.cap.get.side_effect = get_prop
        self.cap.set.side_effect = set_prop

    def get_frame(self):
        # Generate a frame where sharpness depends on current_focus
        # Gaussian curve: exp(-((x - mu)^2) / (2 * sigma^2))
        dist = abs(self.current_focus - self.peak_focus)

        # Create a dummy image with noise related to sharpness
        # Higher sharpness = more high frequency noise/edges
        img = np.zeros((480, 640), dtype=np.uint8)

        # Add some random noise scaled by sharpness
        cv2.circle(img, (320, 240), 50, (255, 255, 255), -1)

        # Blur based on distance from peak focus to simulate out of focus
        blur_ksize = int(dist / 10) * 2 + 1
        if blur_ksize > 1:
            img = cv2.GaussianBlur(img, (blur_ksize, blur_ksize), 0)

        return img

class TestAutofocus(unittest.TestCase):
    def test_autofocus_convergence(self):
        # Setup mock driver with peak at 150
        driver = MockCameraDriver(peak_focus=150, peak_width=30)

        with patch('led_detection.main.get_driver', return_value=driver):
            monitor = PeakMonitor(interval=1, threshold=10, autofocus=True)
        # monitor.cam = driver # No longer needed as get_driver returns it

        # Run autofocus
        monitor.autofocus_sweep()

        # Check if it converged near 150
        final_focus = driver.current_focus
        print(f"Final focus: {final_focus}")
        self.assertTrue(140 <= final_focus <= 160, f"Focus {final_focus} not close to peak 150")

if __name__ == '__main__':
    unittest.main()
