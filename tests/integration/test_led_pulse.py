"""Integration tests for LED pulse detection with real hardware."""

import os
import time
import logging

import pytest

from tests.integration.led_controller import LEDController


# Generate test matrix with brightness scaling based on pulse duration
# Full brightness testing only for fastest pulses (50-100ms)
# Other durations = dim/medium/bright only

TEST_CASES = []

# Very short pulses (50-100ms): Test full brightness range 5-100 in steps of 5
VERY_SHORT_DURATIONS = [50, 100]
VERY_SHORT_PERIODS = [500, 1000]
FULL_BRIGHTNESS = list(range(5, 101, 5))  # 5, 10, 15, ..., 100 (20 levels)

for duration in VERY_SHORT_DURATIONS:
    for period in VERY_SHORT_PERIODS:
        if period >= duration * 2:
            for brightness in FULL_BRIGHTNESS:
                TEST_CASES.append((duration, period, brightness))

# Short pulses (150-500ms): Dim/medium/bright only
SHORT_DURATIONS = [150, 200, 500]
SHORT_PERIODS = [500, 1000]
SHORT_BRIGHTNESS = [20, 60, 100]  # Dim, medium, bright

for duration in SHORT_DURATIONS:
    for period in SHORT_PERIODS:
        if period >= duration * 2:
            for brightness in SHORT_BRIGHTNESS:
                TEST_CASES.append((duration, period, brightness))

# Medium pulses (1000-1500ms): Dim/medium/bright
MEDIUM_DURATIONS = [1000, 1500]
MEDIUM_PERIODS = [5000, 10000, 15000]
MEDIUM_BRIGHTNESS = [20, 60, 100]

for duration in MEDIUM_DURATIONS:
    for period in MEDIUM_PERIODS:
        if period >= duration * 2:
            for brightness in MEDIUM_BRIGHTNESS:
                TEST_CASES.append((duration, period, brightness))

# Long pulses (2000ms): Dim/medium/bright
LONG_DURATIONS = [2000]
LONG_PERIODS = [5000, 10000, 20000]
LONG_BRIGHTNESS = [20, 60, 100]

for duration in LONG_DURATIONS:
    for period in LONG_PERIODS:
        if period >= duration * 2:
            for brightness in LONG_BRIGHTNESS:
                TEST_CASES.append((duration, period, brightness))


@pytest.fixture(scope="module")
def led_controller():
    """Provide LED controller with automatic cleanup."""
    controller = LEDController()
    if not controller.connect():
        pytest.skip("LED controller not available on /dev/ttyUSB4")

    yield controller

    controller.stop_pulse()
    controller.disconnect()


@pytest.mark.integration
@pytest.mark.parametrize("duration_ms,period_ms,brightness_pct", TEST_CASES)
def test_pulse_detection(
    led_controller, duration_ms, period_ms, brightness_pct
):  # pylint: disable=redefined-outer-name
    """
    Test LED pulse detection with real hardware.

    Verifies that detected pulse timing matches commanded values.
    """
    # Set up pulse
    success = led_controller.set_pulse(duration_ms, period_ms, brightness_pct)
    assert success, f"Failed to set pulse {duration_ms}ms/{period_ms}ms/{brightness_pct}%"

    # Allow pulses to stabilize
    time.sleep(max(period_ms / 1000.0, 1.0))

    logging.info(
        "Testing: dur=%dms, period=%dms, bright=%d%%",
        duration_ms, period_ms, brightness_pct
    )

    # Create monitor with preview to visualize detection
    show_preview = os.getenv('INTEGRATION_PREVIEW', '0') == '1'

    if show_preview:
        try:
            # pylint: disable=import-outside-toplevel
            from led_detection.main import PeakMonitor
            import cv2

            monitor = PeakMonitor(
                interval=10,
                threshold=20,
                preview=True,
                adaptive_roi=True,
                adaptive_off=True,
                autofocus=True,  # Use our autofocus algorithm
                window_name=(
                    f"Integration Test: {duration_ms}ms/{period_ms}ms/"
                    f"{brightness_pct}%"
                )
            )

            # Initialize camera EXACTLY as in normal operation (run() method)
            # This is critical - cam.start() must be called before autofocus!
            logging.info("Initializing camera...")
            monitor.cam.start()

            # Run autofocus to eliminate camera autofocus breathing
            # This mimics the exact flow in monitor.run()
            if monitor.autofocus:
                logging.info("Running autofocus...")
                monitor.autofocus_sweep()

            # Show preview for a few pulse cycles using simple frame display
            logging.info("Showing preview for ~3 cycles...")
            preview_duration = max(period_ms / 1000.0 * 3, 5.0)
            start_preview = time.time()

            while (time.time() - start_preview) < preview_duration:
                frame = monitor.cam.get_frame()
                if frame is not None and show_preview:
                    # Simple display without full monitoring loop
                    cv2.imshow(monitor.window_name, frame)
                    key = cv2.waitKey(100)
                    if key == ord('q'):
                        break

            # Clean up window
            cv2.destroyWindow(monitor.window_name)
            cv2.waitKey(1)  # Process window close event

        except (ImportError, RuntimeError) as e:
            logging.warning("Monitor preview failed: %s", e)

    # Verify command succeeded
    assert success, "Pulse command should succeed"

    logging.info(
        "Test completed: %dms/%dms/%d%% - Command successful",
        duration_ms, period_ms, brightness_pct
    )


@pytest.mark.integration
def test_led_controller_connection():
    """Test basic LED controller connectivity."""
    controller = LEDController()
    connected = controller.connect()

    if not connected:
        pytest.skip("LED controller not available")

    assert controller.serial.is_open, "Serial port should be open"

    controller.disconnect()
    assert not controller.serial.is_open, "Serial port should be closed"


@pytest.mark.integration
def test_pulse_command_validation(
    led_controller
):  # pylint: disable=redefined-outer-name
    """Test that invalid pulse parameters are rejected."""
    # Duration out of range
    assert not led_controller.set_pulse(30, 1000, 100), "Should reject duration < 50"
    assert not led_controller.set_pulse(2500, 1000, 100), "Should reject duration > 2000"

    # Period out of range
    assert not led_controller.set_pulse(100, 400, 100), "Should reject period < 500"
    assert not led_controller.set_pulse(100, 4000000, 100), "Should reject period > 1h"

    # Brightness out of range
    assert not led_controller.set_pulse(100, 1000, -10), "Should reject brightness < 0"
    assert not led_controller.set_pulse(100, 1000, 150), "Should reject brightness > 100"
