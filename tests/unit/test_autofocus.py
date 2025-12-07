
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

    def test_autofocus_reverts_to_initial_if_no_improvement(self):
        """
        Test that autofocus reverts to the initial position if the sweep
        does not find a significantly better focus.
        """
        # Setup
        initial_focus = 100
        initial_sharpness = 1000.0

        # We want the sweep to find something WORSE or Same, ensuring it reverts.
        # The logic in main.py likely seeks max sharpness.

        # Mock the camera methods using MagicMock for more control than the simple MockCameraDriver
        driver = MagicMock(spec=X86CameraDriver)
        driver.cap = MagicMock()

        camera_state = {'focus': initial_focus}

        def set_focus(prop, val):
            if prop == cv2.CAP_PROP_FOCUS:
                camera_state['focus'] = int(val)
                return True
            return True

        def get_focus(prop):
            if prop == cv2.CAP_PROP_FOCUS:
                return camera_state['focus']
            return 0

        driver.cap.set.side_effect = set_focus
        driver.cap.get.side_effect = get_focus

        # We also need get_frame to return something valid
        driver.get_frame.return_value = np.zeros((100, 100), dtype=np.uint8)

        # Mock PeakMonitor with this driver
        with patch('led_detection.main.get_driver', return_value=driver), \
             patch('cv2.Laplacian') as mock_laplacian:

            monitor = PeakMonitor(interval=10, threshold=50, autofocus=True)
            # Ensure the monitor uses our mock driver (get_driver patch handles init, but let's be safe)
            monitor.cam = driver

            # Counter to simulate noise/transient peaks
            call_counts = {}

            def get_sharpness(*_args, **_kwargs):
                focus = camera_state['focus']
                call_counts[focus] = call_counts.get(focus, 0) + 1

                mock_res = MagicMock()

                # Logic from original test:
                if focus == initial_focus: # 100
                    mock_res.var.return_value = initial_sharpness # 1000
                elif focus == 220:
                    # Simulating a transient peak that isn't sustained
                    if call_counts[focus] == 1:
                        mock_res.var.return_value = 2000.0
                    else:
                        mock_res.var.return_value = 500.0
                else:
                    mock_res.var.return_value = 0.0
                return mock_res

            mock_laplacian.side_effect = get_sharpness

            # Run autofocus
            monitor.autofocus_sweep()

            # Verify that the FINAL focus set was the initial focus
            # Get all calls to set focus
            focus_calls = [
                call.args[1]
                for call in driver.cap.set.mock_calls
                if call.args[0] == cv2.CAP_PROP_FOCUS
            ]

            final_set_focus = focus_calls[-1]

            self.assertEqual(final_set_focus, initial_focus,
                f"Autofocus should revert to initial {initial_focus}, but ended at {final_set_focus}")

if __name__ == '__main__':
    unittest.main()
