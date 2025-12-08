import os
import time
import logging
import shutil
import pytest
import cv2

from tests.integration.led_controller import LEDController
from led_detection.main import PeakMonitor

@pytest.fixture(scope="module", autouse=True)
def clean_debug_dir():
    """Clean debug directory before test run."""
    debug_dir = "/tmp/oneshot_debug"
    if os.path.exists(debug_dir):
        shutil.rmtree(debug_dir)
    os.makedirs(debug_dir)
    logging.info("Cleaned debug directory: %s", debug_dir)
    yield

@pytest.fixture(scope="module")
def led_controller():
    """Provide LED controller with automatic cleanup."""
    controller = LEDController()
    if not controller.connect():
        pytest.skip("LED controller not available on /dev/ttyUSB4")

    yield controller

    # Leave LED pulsing: 100ms ON every 500ms at 100% brightness
    controller.set_pulse(100, 500, 100)
    time.sleep(0.1)
    controller.disconnect()

@pytest.fixture(scope="module")
def monitor():
    """Provide PeakMonitor with shared camera."""
    try:
        # Check if we should show preview
        show_preview = os.getenv('INTEGRATION_PREVIEW', '0') == '1'

        mon = PeakMonitor(
            interval=10,
            threshold=50,
            preview=show_preview,
            adaptive_roi=True,
            adaptive_off=True,
            autofocus=True,
            min_pulse_duration=25,
            window_name="OneShot Test"
        )

        logging.info("Initializing camera...")
        mon.cam.start()



        yield mon

        mon.cam.stop()
        if show_preview:
            cv2.destroyAllWindows()

    except Exception as e: # pylint: disable=broad-exception-caught
        logging.warning("Failed to initialize monitor/camera: %s", e)
        pytest.skip(f"Camera not available: {e}")

@pytest.mark.integration
@pytest.mark.parametrize("duration,period,brightness", [
    (147, 5.0, 100),   # Standard
    (50, 1.0, 50),     # Short & Dim
    (500, 2.0, 100)    # Long
])
def test_oneshot_success(led_controller, monitor, duration, period, brightness): # pylint: disable=redefined-outer-name
    """
    Test successful one-shot detection with various parameters.
    Cases:
    1. Standard: 147ms / 5s / 100%
    2. Short/Dim: 50ms / 1s / 50%
    3. Long: 500ms / 2s / 100%
    """
    logging.info("Testing One-Shot: Duration=%dms, Period=%.1fs, Brightness=%d%%",
                 duration, period, brightness)

    # Setup Pulse
    assert led_controller.set_pulse(duration, int(period * 1000), brightness)

    # Run one-shot
    # Note: Short pulses might need slightly more tolerance or careful sampling?
    # But 50ms is > 33ms (1 frame), so it should be caught.
    # We keep tolerance at default 0.2
    success = monitor.run_one_shot(expected_period=period, expected_duration=duration, num_pulses=6)

    assert success, f"One-shot detection failed for {duration}ms / {period}s / {brightness}%"

@pytest.mark.integration
def test_oneshot_fail_duration(led_controller, monitor): # pylint: disable=redefined-outer-name
    """Test one-shot detection failure due to duration mismatch."""
    # Setup: 50ms ON / 500ms Period
    assert led_controller.set_pulse(50, 500, 100)

    # Run one-shot with WRONG duration (expecting 100ms)
    success = monitor.run_one_shot(expected_period=0.5, expected_duration=100, tolerance=0.2)

    assert not success, "One-shot detection should fail for duration mismatch"

@pytest.mark.integration
def test_oneshot_fail_period(led_controller, monitor): # pylint: disable=redefined-outer-name
    """Test one-shot detection failure due to period mismatch."""
    # Setup: 50ms ON / 500ms Period
    assert led_controller.set_pulse(50, 500, 100)

    # Run one-shot with WRONG period (expecting 1.0s)
    success = monitor.run_one_shot(expected_period=1.0, expected_duration=50, tolerance=0.2)

    assert not success, "One-shot detection should fail for period mismatch"
