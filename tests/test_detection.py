
import time
import unittest
from unittest.mock import MagicMock, patch

import cv2
import numpy as np

from led_detection.main import PeakMonitor, X86CameraDriver

class MockVideoDriver(X86CameraDriver):
    def __init__(self, width=640, height=480, fps=30, show_preview=True):
        super().__init__()
        self.cap = MagicMock()
        self.w = width
        self.h = height
        self.fps = fps
        self.frame_count = 0
        self.show_preview = show_preview

        # Simulation parameters
        self.ambient_brightness = 50
        self.noise_level = 2

        # List of LEDs: each is a dict with properties
        self.leds = []

        # Default single LED config (for backward compatibility)
        self.leds.append({
            "pos": (width//2, height//2),
            "radius": 4,
            "brightness": 0,
            "start_frame": -1,
            "duration": 0
        })

        # Ambient transition support
        self.ambient_transition_start_frame = -1
        self.ambient_start_brightness = 50
        self.ambient_end_brightness = 50
        self.ambient_transition_duration = 0

        # Mock cap properties
        self.cap.get.return_value = 0 # Default

        self.background_pattern = None

    def configure_background(self, mode='uniform'):
        """Configure background pattern."""
        if mode == 'random':
            # Generate static random background
            self.background_pattern = np.random.randint(0, 255, (self.h, self.w), dtype=np.uint8)
        else:
            self.background_pattern = None

    @property
    def led_brightness(self):
        return self.leds[0]["brightness"]

    @led_brightness.setter
    def led_brightness(self, val):
        self.leds[0]["brightness"] = val

    @property
    def led_radius(self):
        return self.leds[0]["radius"]

    @led_radius.setter
    def led_radius(self, val):
        self.leds[0]["radius"] = val

    @property
    def led_position(self):
        return self.leds[0]["pos"]

    @led_position.setter
    def led_position(self, val):
        self.leds[0]["pos"] = val

    def configure_flash(self, start_frame, duration, brightness=200, led_index=0):
        if led_index >= len(self.leds):
            # Extend list if needed (simple case)
            self.leds.append({
                "pos": (self.w//2, self.h//2),
                "radius": 4,
                "brightness": 0,
                "start_frame": -1,
                "duration": 0
            })

        self.leds[led_index]["start_frame"] = start_frame
        self.leds[led_index]["duration"] = duration
        self.leds[led_index]["brightness"] = brightness

    def add_led(self, pos, radius=4):
        """Add a new LED to the simulation."""
        self.leds.append({
            "pos": pos,
            "radius": radius,
            "brightness": 0,
            "start_frame": -1,
            "duration": 0
        })
        return len(self.leds) - 1

    def configure_ambient_transition(self, start_frame, duration, start_brightness, end_brightness):
        """Configure a gradual ambient brightness transition."""
        self.ambient_transition_start_frame = start_frame
        self.ambient_transition_duration = duration
        self.ambient_start_brightness = start_brightness
        self.ambient_end_brightness = end_brightness
        self.ambient_brightness = start_brightness

    def get_frame(self):
        # Handle ambient transition if configured
        if (self.ambient_transition_start_frame >= 0 and
            self.ambient_transition_start_frame <= self.frame_count <
            self.ambient_transition_start_frame + self.ambient_transition_duration):
            # Linear interpolation
            progress = (self.frame_count - self.ambient_transition_start_frame) / \
                       self.ambient_transition_duration
            self.ambient_brightness = int(
                self.ambient_start_brightness +
                (self.ambient_end_brightness - self.ambient_start_brightness) * progress)

        # Create base image
        if self.background_pattern is not None:
            # Use static background pattern
            img = self.background_pattern.copy()
            # Apply ambient brightness scaling if needed, or just use as is
            # For simplicity, let's assume the pattern REPLACES the ambient brightness logic
            # or we can scale it. Let's just use it as base.
        else:
            # Create base image with ambient brightness (uniform gray)
            img = np.full((self.h, self.w), self.ambient_brightness, dtype=np.uint8)

        # Add noise
        noise = np.random.normal(0, self.noise_level, (self.h, self.w))
        img = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)

        # Draw all active LEDs
        led_layer = np.zeros_like(img)
        any_led_on = False

        for led in self.leds:
            led_on = (led["start_frame"] >= 0 and
                      led["start_frame"] <= self.frame_count < led["start_frame"] + led["duration"])

            if led_on:
                any_led_on = True
                cv2.circle(led_layer, led["pos"], led["radius"], (led["brightness"]), -1)

        if any_led_on:
            img = cv2.add(img, led_layer)

        # Convert to BGR to match camera output
        img_bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        # Show preview if enabled
        if self.show_preview:
            preview = img_bgr.copy()
            # Add status text
            status = "LED ON" if any_led_on else "LED OFF"
            color = (0, 255, 0) if any_led_on else (128, 128, 128)
            cv2.putText(preview, f"Frame: {self.frame_count} | {status}",
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            cv2.putText(preview, f"Ambient: {self.ambient_brightness} | LEDs: {len(self.leds)}",
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            # Draw crosshairs
            for i, led in enumerate(self.leds):
                cv2.drawMarker(preview, led["pos"], (0, 255, 255), cv2.MARKER_CROSS, 20, 1)
                cv2.putText(preview, str(i), (led["pos"][0]+10, led["pos"][1]),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

            cv2.imshow("Mock Camera Feed", preview)
            cv2.waitKey(1)

        self.frame_count += 1
        return img_bgr

class TestDetection(unittest.TestCase):
    def setUp(self):
        self.driver = MockVideoDriver(show_preview=False)
        # Patch get_driver to return our mock driver
        self.patcher = patch('led_detection.main.get_driver', return_value=self.driver)
        self.mock_get_driver = self.patcher.start()
        self.addCleanup(self.patcher.stop)

        # Use a fast interval for testing
        self.monitor = PeakMonitor(interval=10, threshold=20, autofocus=False, preview=True,
                                 window_name=self._testMethodName)
        # self.monitor.cam = self.driver # get_driver now returns it
        # Disable aiming to speed up test

    def tearDown(self):
        """Close preview window after each test."""
        cv2.destroyAllWindows()
        time.sleep(0.1)  # Brief pause to ensure window closes

    def test_standard_flash(self):
        """Test a clear, standard duration flash."""
        # Flash at frame 10 for 5 frames (~160ms at 30fps)
        self.driver.configure_flash(start_frame=10, duration=5, brightness=150)

        # Run detection
        # We need to override wait_for_signal to stop after some frames to avoid infinite loop
        # But wait_for_signal is the loop body. The loop is in run().
        # We can call wait_for_signal() once. It returns True if signal found.

        # We need to pump frames into it. wait_for_signal calls get_frame internally.
        # It establishes baseline (10 frames) then loops.

        # Let's just call wait_for_signal. It should return True when it sees the flash.
        # We need to ensure it doesn't loop forever if it misses.
        # The monitor has a timeout, but it's long (70s).

        # We can rely on the fact that our mock driver advances frames.
        # If we set the flash early enough, it should catch it.

        found = self.monitor.wait_for_signal()
        self.assertTrue(found, "Should detect standard flash")

    def test_short_flash(self):
        """Test a very short flash (1 frame)."""
        self.driver.configure_flash(start_frame=15, duration=1, brightness=150)
        found = self.monitor.wait_for_signal()
        self.assertTrue(found, "Should detect 1-frame flash")

    def test_low_contrast(self):
        """Test a dim flash against bright background."""
        self.driver.ambient_brightness = 150
        # Flash adds 30 brightness (150 -> 180). Threshold is 20.
        self.driver.configure_flash(start_frame=15, duration=5, brightness=30)

        # Ensure we use contrast mode if that's what helps, or brightness.
        # Default is contrast=True.
        # 180 (spot) - 150 (median) = 30 > 20 (threshold) -> Should detect.

        found = self.monitor.wait_for_signal()
        self.assertTrue(found, "Should detect low contrast flash")

    def test_no_flash(self):
        """Test false positive rejection."""
        self.driver.configure_flash(start_frame=-1, duration=0) # No flash

        # This will hang until timeout (70s) if we don't interrupt it.
        # We can mock time.time to simulate timeout, or just limit frames.
        # But wait_for_signal doesn't take a frame limit.

        # Hack: We can make the driver raise an exception after N frames to stop the loop?
        # Or better, we can subclass PeakMonitor to add a frame limit.

        # For now, let's just set a short timeout on the monitor?
        # Use the new timeout parameter
        start = time.time()
        found = self.monitor.wait_for_signal(timeout=2.0)
        duration = time.time() - start

        self.assertFalse(found, "Should not detect anything")
        # Should have waited roughly the timeout duration
        self.assertGreater(duration, 1.5)

    def test_white_background(self):
        """Test detection against bright white background."""
        self.driver.ambient_brightness = 220  # Very bright background
        self.driver.configure_flash(start_frame=15, duration=5, brightness=35)  # Challenging but detectable

        found = self.monitor.wait_for_signal()
        self.assertTrue(found, "Should detect LED even on white background")

    def test_black_background(self):
        """Test detection against dark black background."""
        self.driver.ambient_brightness = 10  # Very dark background
        self.driver.configure_flash(start_frame=15, duration=5, brightness=200)

        found = self.monitor.wait_for_signal()
        self.assertTrue(found, "Should detect LED on black background")

    def test_gray_background(self):
        """Test detection against medium gray background."""
        self.driver.ambient_brightness = 128  # Medium gray
        self.driver.configure_flash(start_frame=15, duration=5, brightness=100)

        found = self.monitor.wait_for_signal()
        self.assertTrue(found, "Should detect LED on gray background")

    def test_roi_detection(self):
        """Test that ROI is correctly detected and locked."""
        # Position LED off-center
        self.driver.led_position = (200, 150)
        self.driver.configure_flash(start_frame=15, duration=5, brightness=150)

        found = self.monitor.wait_for_signal()
        self.assertTrue(found, "Should detect LED")

        # Verify ROI was set
        self.assertIsNotNone(self.monitor.roi, "ROI should be set after detection")

        # Verify ROI contains the LED position
        y1, y2, x1, x2 = self.monitor.roi
        led_x, led_y = self.driver.led_position

        self.assertLessEqual(x1, led_x,
                             "ROI should contain LED x position (left bound)")
        self.assertGreaterEqual(x2, led_x,
                                "ROI should contain LED x position (right bound)")
        self.assertLessEqual(y1, led_y,
                             "ROI should contain LED y position (top bound)")
        self.assertGreaterEqual(y2, led_y,
                                "ROI should contain LED y position (bottom bound)")

        # Verify ROI is reasonably sized (not too small, not too large)
        roi_width = x2 - x1
        roi_height = y2 - y1
        self.assertGreater(roi_width, 10, "ROI should be at least 10px wide")
        self.assertGreater(roi_height, 10, "ROI should be at least 10px tall")
        self.assertLess(roi_width, 100, "ROI should not be excessively wide")
        self.assertLess(roi_height, 100, "ROI should not be excessively tall")

    def test_dark_to_bright_transition(self):
        """Test detection during ambient light transition from dark to bright."""
        # Configure ambient transition: dark (20) to bright (200) over 30 frames
        self.driver.configure_ambient_transition(
            start_frame=5,
            duration=30,
            start_brightness=20,
            end_brightness=200
        )
        # Flash occurs during the transition
        self.driver.configure_flash(start_frame=20, duration=5, brightness=50)

        found = self.monitor.wait_for_signal()
        self.assertTrue(found, "Should detect LED during dark-to-bright transition")

    def test_bright_to_dark_transition(self):
        """Test detection during ambient light transition from bright to dark."""
        # Configure ambient transition: bright (200) to dark (20) over 30 frames
        self.driver.configure_ambient_transition(
            start_frame=5,
            duration=30,
            start_brightness=200,
            end_brightness=20
        )
        # Flash occurs during the transition
        self.driver.configure_flash(start_frame=20, duration=5, brightness=100)

        found = self.monitor.wait_for_signal()
        self.assertTrue(found, "Should detect LED during bright-to-dark transition")

    def test_flash_detection_rate(self):
        """Test that flash detection rate is reliable across multiple attempts."""
        success_count = 0
        total_attempts = 5

        for attempt in range(total_attempts):
            # Reset driver for each attempt
            self.driver.frame_count = 0
            self.driver.configure_flash(start_frame=15, duration=5, brightness=150)

            # Create new monitor for each attempt to reset state
            # Note: get_driver is already patched in setUp, but we need to make sure
            # it returns the *current* driver if we replaced it?
            # Actually we just reset the driver state, we don't create a new driver object usually.
            # But here we are creating a new monitor.

            monitor = PeakMonitor(interval=10, threshold=20, autofocus=False, preview=True,
                                window_name=f"{self._testMethodName}_{attempt}")
            # monitor.cam = self.driver # Handled by patch
            # monitor.preview = False # Enable preview

            found = monitor.wait_for_signal()
            if found:
                success_count += 1

        detection_rate = success_count / total_attempts
        self.assertGreaterEqual(detection_rate, 0.8,
                                f"Detection rate {detection_rate:.1%}"
                                " should be >= 80%")

    def test_nighttime_conditions(self):
        """Test detection in very dark nighttime conditions."""
        self.driver.ambient_brightness = 2  # Near-black nighttime
        self.driver.configure_flash(start_frame=15, duration=5, brightness=100)

        found = self.monitor.wait_for_signal()
        self.assertTrue(found, "Should detect LED in nighttime conditions")

    def test_bright_daylight_conditions(self):
        """Test detection in very bright daylight conditions."""
        self.driver.ambient_brightness = 240  # Very bright daylight
        self.driver.configure_flash(start_frame=15, duration=5, brightness=15)  # Challenging but detectable

        # Saturation limits contrast to 15 (255-240). Threshold 20 is too high without noise.
        self.monitor.min_signal_strength = 10
        found = self.monitor.wait_for_signal(timeout=10.0)
        self.assertTrue(found, "Should detect LED in bright daylight (tests adaptive threshold)")

    def test_tiny_led(self):
        """Test detection of very small LED (2px radius)."""
        self.driver.led_radius = 2
        self.driver.configure_flash(start_frame=15, duration=5, brightness=200)

        found = self.monitor.wait_for_signal()
        self.assertTrue(found, "Should detect tiny 2px LED")

    def test_large_led(self):
        """Test detection of larger LED (8px radius)."""
        self.driver.led_radius = 8
        self.driver.configure_flash(start_frame=15, duration=5, brightness=150)

        found = self.monitor.wait_for_signal()
        self.assertTrue(found, "Should detect large 8px LED")

    def test_dim_led_dark_background(self):
        """Test dim LED on dark background."""
        self.driver.ambient_brightness = 10
        self.driver.configure_flash(start_frame=15, duration=5, brightness=100)  # Dim but detectable

        found = self.monitor.wait_for_signal()
        self.assertTrue(found, "Should detect dim LED on dark background")

    def test_bright_led_bright_background(self):
        """Test bright LED on bright background."""
        self.driver.ambient_brightness = 200
        self.driver.configure_flash(start_frame=15, duration=5, brightness=50)

        found = self.monitor.wait_for_signal()
        self.assertTrue(found, "Should detect bright LED on bright background")

    def test_adaptive_baseline_update(self):
        """Test that baseline adapts to slow ambient changes."""
        # Start dark, gradually increase ambient during baseline establishment
        self.driver.configure_ambient_transition(
            start_frame=0,
            duration=50,  # Slow transition during baseline
            start_brightness=10,
            end_brightness=100
        )
        # Flash after transition
        self.driver.configure_flash(start_frame=60, duration=5, brightness=100)

        found = self.monitor.wait_for_signal()
        self.assertTrue(found, "Should detect LED after baseline adapts to ambient change")

    def test_extreme_contrast_range(self):
        """Test detection across extreme contrast scenarios in one test."""
        test_scenarios = [
            {"ambient": 1, "led": 254, "name": "max contrast (night)"},
            {"ambient": 230, "led": 25, "name": "challenging daylight"},
            {"ambient": 128, "led": 127, "name": "medium contrast"},
        ]

        for scenario in test_scenarios:
            with self.subTest(scenario=scenario["name"]):
                # Reset driver
                self.driver.frame_count = 0
                self.driver.ambient_brightness = scenario["ambient"]
                self.driver.configure_flash(start_frame=15, duration=5, brightness=scenario["led"])

                # Create fresh monitor
                monitor = PeakMonitor(interval=10, threshold=20, autofocus=False, preview=True,
                                    window_name=f"{self._testMethodName}_{scenario['name']}")
                # monitor.cam = self.driver # Handled by patch
                # monitor.preview = False # Enable preview

                found = monitor.wait_for_signal(timeout=10.0)
                self.assertTrue(found,
                    f"Should detect in {scenario['name']} scenario "
                    f"(ambient={scenario['ambient']}, led={scenario['led']})")

    def test_multiple_leds(self):
        """Test detection with multiple flashing LEDs (should pick brightest)."""
        # LED 1: Dim, Top-Left
        self.driver.led_position = (150, 150)
        self.driver.led_brightness = 50
        self.driver.configure_flash(start_frame=15, duration=5, brightness=50, led_index=0)

        # LED 2: Bright, Bottom-Right
        led2_pos = (450, 350)
        self.driver.add_led(pos=led2_pos, radius=6)
        self.driver.configure_flash(start_frame=15, duration=5, brightness=200, led_index=1)

        found = self.monitor.wait_for_signal()
        self.assertTrue(found, "Should detect signal with multiple LEDs")

        # Verify ROI selected the brighter LED (LED 2)
        y1, y2, x1, x2 = self.monitor.roi
        led_x, led_y = led2_pos

        self.assertLessEqual(x1, led_x, "ROI should contain bright LED x")
        self.assertGreaterEqual(x2, led_x, "ROI should contain bright LED x")
        self.assertLessEqual(y1, led_y, "ROI should contain bright LED y")
        self.assertGreaterEqual(y2, led_y, "ROI should contain bright LED y")

    def test_high_temporal_noise(self):
        """Test detection with high temporal noise (dynamic grain) across various conditions."""
        brightness_levels = [50, 100, 200]
        radii = [2, 4, 8]

        for brightness in brightness_levels:
            for radius in radii:
                with self.subTest(brightness=brightness, radius=radius):
                    # Reset driver state
                    self.driver.frame_count = 0
                    self.driver.noise_level = 20  # High noise
                    self.driver.led_radius = radius
                    self.driver.configure_flash(start_frame=15, duration=5, brightness=brightness)

                    # Create fresh monitor for each subtest
                    monitor = PeakMonitor(interval=10, threshold=20, autofocus=False, preview=True,
                                        window_name=f"{self._testMethodName}_b{brightness}_r{radius}")
                    # monitor.cam = self.driver # Handled by patch

                    found = monitor.wait_for_signal()
                    self.assertTrue(found, f"Should detect LED (bright={brightness}, rad={radius}) despite high noise")

    def test_static_random_background(self):
        """Test detection against a static random background texture across various conditions."""
        brightness_levels = [50, 150, 250]
        radii = [2, 4, 8]

        for brightness in brightness_levels:
            for radius in radii:
                with self.subTest(brightness=brightness, radius=radius):
                    # Reset driver state
                    self.driver.frame_count = 0
                    self.driver.configure_background('random')
                    self.driver.led_radius = radius
                    self.driver.configure_flash(start_frame=15, duration=5, brightness=brightness)

                    # Create fresh monitor for each subtest
                    monitor = PeakMonitor(interval=10, threshold=20, autofocus=False, preview=True,
                                        window_name=f"{self._testMethodName}_b{brightness}_r{radius}")
                    # monitor.cam = self.driver # Handled by patch

                    found = monitor.wait_for_signal()
                    self.assertTrue(found, f"Should detect LED (bright={brightness}, rad={radius}) against static random background")

class TestFeatures(unittest.TestCase):
    """Test independent detection features controlled by flags."""

    def setUp(self):
        self.driver = MockVideoDriver(show_preview=False)
        self.patcher = patch('led_detection.main.get_driver', return_value=self.driver)
        self.mock_get_driver = self.patcher.start()
        self.addCleanup(self.patcher.stop)

    def tearDown(self):
        cv2.destroyAllWindows()
        time.sleep(0.1)

    def test_contrast_mode(self):
        """Verify use_contrast flag toggles between contrast and brightness measurement."""
        # Create a frame with high brightness but low contrast (flat bright)
        # e.g. all pixels = 200. Max=200, Median=200. Contrast=0. Brightness(90th)=200.
        frame = np.full((100, 100), 200, dtype=np.uint8)

        # 1. Contrast Mode (Default)
        monitor = PeakMonitor(interval=10, threshold=20, use_contrast=True)
        val = monitor._measure_roi(frame)
        self.assertEqual(val, 0.0, "Contrast mode should return 0 for flat image")

        # 2. Brightness Mode
        monitor = PeakMonitor(interval=10, threshold=20, use_contrast=False)
        val = monitor._measure_roi(frame)
        self.assertEqual(val, 200.0, "Brightness mode should return pixel value")

    def test_adaptive_roi_flag(self):
        """Verify adaptive_roi flag controls ROI sizing logic."""
        # Setup a small LED flash
        self.driver.led_radius = 2
        self.driver.configure_flash(start_frame=15, duration=5, brightness=200)

        # 1. Adaptive ROI Enabled
        monitor = PeakMonitor(interval=10, threshold=20, adaptive_roi=True, preview=False)
        # monitor.cam = self.driver # Handled by patch
        monitor.wait_for_signal()

        y1, y2, x1, x2 = monitor.roi
        w, h = x2-x1, y2-y1
        # Should be small (e.g. 24x24 min size)
        self.assertLess(w, 40, "Adaptive ROI should produce small box for small LED")

        # 2. Adaptive ROI Disabled
        self.driver.frame_count = 0
        self.driver.configure_flash(start_frame=15, duration=5, brightness=200)
        monitor = PeakMonitor(interval=10, threshold=20, adaptive_roi=False, preview=False)
        # monitor.cam = self.driver # Handled by patch
        monitor.wait_for_signal()

        y1, y2, x1, x2 = monitor.roi
        w, h = x2-x1, y2-y1
        # Should be fixed size (32 half-size -> 64x64)
        self.assertEqual(w, 64, "Fixed ROI should be 64px wide")
        self.assertEqual(h, 64, "Fixed ROI should be 64px tall")

    def test_adaptive_off_flag(self):
        """Verify adaptive_off flag changes OFF detection threshold."""
        # We need to simulate the wait_for_led_off logic.
        # It's hard to test full flow without mocking time or frames perfectly.
        # But we can check the logic inside wait_for_led_off if we could inspect it.
        # Alternatively, we can construct a scenario where fixed threshold fails but adaptive passes, or vice versa.

        # Fixed threshold is high_val * 0.6.
        # Adaptive threshold is usually mean - 2*std.

        # Scenario: Stable signal drops slightly (e.g. to 80%).
        # Fixed: 80% > 60%, so it thinks it's still ON.
        # Adaptive: If noise is low, mean-2*std will be close to mean (e.g. 98%).
        # So 80% < 98%, it thinks it's OFF.

        # Let's try to verify this behavior.

        # 1. Adaptive OFF (Enabled)
        monitor = PeakMonitor(interval=10, threshold=20, adaptive_off=True, use_contrast=False)
        # monitor.cam = self.driver # Handled by patch
        monitor.roi = (0, 100, 0, 100) # Dummy ROI

        # Mock get_frame to return stable high value then drop to 80%
        # We need to mock _measure_roi or control frame content.
        # Let's control frame content.

        # Sequence: 10 frames of 200 (est baseline), then frames of 160 (80%).
        # Adaptive logic: mean=200, std=0. Threshold ~200. 160 < 200 -> OFF detected.

        def frame_gen():
            # 10 frames high (200)
            for _ in range(12): # +2 for initial read
                yield np.full((480, 640), 200, dtype=np.uint8)
            # Then drop to 160
            while True:
                yield np.full((480, 640), 160, dtype=np.uint8)

        gen = frame_gen()
        self.driver.get_frame = lambda: next(gen)

        # Should return True (OFF detected)
        # We need to override wait_for_led_off's internal loop or just call it.
        # It waits up to 15s.
        start = time.time()
        is_off = monitor.wait_for_led_off()
        duration = time.time() - start

        self.assertTrue(is_off, "Adaptive OFF should detect 20% drop as OFF (stable signal)")
        self.assertLess(duration, 2.0, "Should detect quickly")

        # 2. Fixed OFF (Disabled)
        monitor = PeakMonitor(interval=10, threshold=20, adaptive_off=False, use_contrast=False)
        # monitor.cam = self.driver # Re-attach mocked driver (patch is still active)
        monitor.roi = (0, 100, 0, 100)

        gen = frame_gen() # Reset generator
        self.driver.get_frame = lambda: next(gen)

        # Threshold = 200 * 0.6 = 120.
        # Value = 160.
        # 160 > 120 -> Still ON.
        # Should timeout (return False)

        # Reduce timeout for test speed? wait_for_led_off has hardcoded 15s.
        # We can mock time.time to speed it up.

        orig_time = time.time
        start_time = orig_time()
        def mock_time():
            # Advance time by 1s each call to simulate timeout quickly
            nonlocal start_time
            start_time += 1.0
            return start_time

        with unittest.mock.patch('time.time', side_effect=mock_time):
            is_off = monitor.wait_for_led_off()

        self.assertFalse(is_off, "Fixed OFF should NOT detect 20% drop (threshold is 60%)")

    def test_adaptive_exposure_flag(self):
        """Verify adaptive_exposure flag controls calibration."""
        # 1. Enabled
        monitor = PeakMonitor(interval=10, threshold=20, adaptive_exposure=True)
        # Mock cam to be X86Driver (required for exposure)
        # We need to update the return value of our patch to be this specific mock
        mock_x86 = MagicMock(spec=X86CameraDriver)
        mock_x86.get_frame.return_value = np.zeros((100,100), dtype=np.uint8)
        self.mock_get_driver.return_value = mock_x86

        monitor.cam = mock_x86 # Explicitly set it too, though PeakMonitor init called get_driver which returned it
        monitor.roi = (0, 10, 0, 10)

        monitor.calibrate_exposure()
        # Should have called get_frame (calibration loop)
        self.assertTrue(monitor.cam.get_frame.called, "Should attempt calibration when enabled")

        # 2. Disabled
        monitor = PeakMonitor(interval=10, threshold=20, adaptive_exposure=False)
        # monitor.cam = MagicMock(spec=X86CameraDriver)
        # get_driver returns the mock_x86 from previous step or we reset it
        # Let's reset it to a generic mock
        mock_x86_2 = MagicMock(spec=X86CameraDriver)
        self.mock_get_driver.return_value = mock_x86_2
        monitor.cam = mock_x86_2

        monitor.calibrate_exposure()
        # Should NOT have called get_frame
        self.assertFalse(monitor.cam.get_frame.called, "Should skip calibration when disabled")

if __name__ == '__main__':
    unittest.main()
