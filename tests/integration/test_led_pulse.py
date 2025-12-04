import os
import time
import logging
import shutil
import pytest
import cv2
import numpy as np

from tests.integration.led_controller import LEDController
from led_detection.main import PeakMonitor

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


@pytest.fixture(scope="module", autouse=True)
def clean_debug_dir():
    """Clean debug directory before test run."""
    debug_dir = "/tmp/led_pulse"
    if os.path.exists(debug_dir):
        shutil.rmtree(debug_dir)
    os.makedirs(debug_dir)
    logging.info("Cleaned debug directory: %s", debug_dir)
    yield
    # No cleanup after, so user can inspect


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
        show_preview = os.getenv('INTEGRATION_PREVIEW', '1') == '1'

        mon = PeakMonitor(
            interval=10,
            threshold=50,
            preview=show_preview,
            adaptive_roi=True,
            adaptive_off=True,
            autofocus=True,
            min_pulse_duration=25,
            window_name="Integration Test"
        )

        logging.info("Initializing camera...")
        mon.cam.start()

        if mon.autofocus:
            logging.info("Running initial autofocus...")
            mon.autofocus_sweep()

        yield mon

        mon.cam.stop()
        if show_preview:
            cv2.destroyAllWindows()

    except Exception as e:  # pylint: disable=broad-exception-caught
        logging.warning("Failed to initialize monitor/camera: %s", e)
        pytest.skip(f"Camera not available: {e}")



# Shared results storage
ROI_RESULTS = {
    "centers": [],
    "valid_iterations": 0
}

@pytest.fixture(scope="module", autouse=True)
def setup_roi_collection():
    """Initialize ROI collection."""
    ROI_RESULTS["centers"] = []
    ROI_RESULTS["valid_iterations"] = 0
    yield

@pytest.mark.integration
@pytest.mark.parametrize("duration_ms,period_ms,brightness_pct", TEST_CASES)
def test_pulse_detection(
    led_controller, monitor, duration_ms, period_ms, brightness_pct
):  # pylint: disable=redefined-outer-name, too-many-locals
    """
    Test LED pulse detection with real hardware.

    Verifies that detected pulse timing matches commanded values.
    """
    # Set up pulse
    success = led_controller.set_pulse(duration_ms, period_ms, brightness_pct)
    assert success, f"Failed to set pulse {duration_ms}ms/{period_ms}ms/{brightness_pct}%"

    # Immediate detection start - no stabilization needed
    logging.info(
        "Testing: dur=%dms, period=%dms, bright=%d%%",
        duration_ms, period_ms, brightness_pct
    )

    # Reset monitor state for new test
    monitor.roi = None

    # Run detection
    # Use a timeout relative to the period, but at least 5s (or 10s for long periods)
    timeout = max(10.0, period_ms / 1000.0 * 3)

    # Calculate dwell time:
    # For short pulses, we must dwell less than the pulse duration to capture it.
    # For long pulses/noise robustness, we want to dwell longer (up to 2.0s).
    # Use 50% of duration, capped at 2.0s.
    dwell_time = min(2.0, duration_ms / 1000.0 * 0.5)

    found = monitor.wait_for_signal(timeout=timeout, dwell_time=dwell_time)

    if found:
        # Collect ROI stats
        y1, y2, x1, x2 = monitor.roi
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        ROI_RESULTS["centers"].append((cx, cy))
        ROI_RESULTS["valid_iterations"] += 1

        # Save debug image
        debug_filename = f"detect_{duration_ms}ms_{period_ms}ms_{brightness_pct}pct.png"
        dest_path = os.path.join("/tmp/led_pulse", debug_filename)

        if os.path.exists("debug_detection.png"):
            shutil.copy("debug_detection.png", dest_path)
            logging.info("Saved debug image to %s", dest_path)
        else:
            logging.warning("debug_detection.png not found!")

        # Verify Pulse Reporting Logic
        # We want to detect at least 3 pulses to verify gap and duration
        collected_pulses = []
        def pulse_callback(_timestamp, duration, gap):
            logging.info("Callback: duration=%.0fms, gap=%.1fs", duration, gap)
            collected_pulses.append({'duration': duration, 'gap': gap})

        logging.info("Starting monitoring for 3 pulses...")
        monitor.wait_for_led_off()
        monitor.start_monitoring(max_pulses=3, on_pulse_callback=pulse_callback)

        assert len(collected_pulses) == 3, f"Expected 3 pulses, got {len(collected_pulses)}"

        # Verify durations and gaps
        # Allow 20% tolerance or 100ms, whichever is larger
        dur_tol = max(100, duration_ms * 0.2)
        gap_tol = max(100, (period_ms - duration_ms) / 1000.0 * 0.2) # Gap is in seconds

        logging.info("Collected Pulses: %s", collected_pulses)

        for i, p in enumerate(collected_pulses):
            # Duration check
            assert abs(p['duration'] - duration_ms) < dur_tol, \
                f"Pulse {i}: Duration {p['duration']}ms not within {dur_tol}ms of {duration_ms}ms"

            # Gap check (skip first pulse as gap might be irregular from startup)
            if i > 0:
                expected_gap = (period_ms - duration_ms) / 1000.0
                assert abs(p['gap'] - expected_gap) < gap_tol, \
                    f"Pulse {i}: Gap {p['gap']}s not within {gap_tol}s of {expected_gap}s"

    else:
        logging.warning("Signal NOT detected for %dms/%dms/%d%%",
                       duration_ms, period_ms, brightness_pct)
        pytest.fail(f"Signal not detected for {duration_ms}ms pulse")

    logging.info("Test completed: %dms/%dms/%d%% - Found: %s",
                 duration_ms, period_ms, brightness_pct, found)
    assert success, "Pulse command should succeed"

    logging.info(
        "Test completed: %dms/%dms/%d%% - Found: %s",
        duration_ms, period_ms, brightness_pct, found
    )

@pytest.mark.integration
def test_pulse_roi_analysis():
    """Analyze ROI consistency across all pulse detection tests."""
    valid_count = ROI_RESULTS["valid_iterations"]
    centers = np.array(ROI_RESULTS["centers"])

    logging.info("ROI Analysis: %d valid detections", valid_count)

    if valid_count < 5:
        # If we didn't detect enough pulses, we can't really analyze consistency.
        # But we shouldn't fail if the tests were just skipped or failed for other reasons?
        # The user wants to know if ROI is consistent when it IS detected.
        if valid_count == 0:
            pytest.skip("No valid detections to analyze.")
        else:
            logging.warning("Low number of detections (%d) for analysis.", valid_count)

    if valid_count > 1:
        mean_center = np.mean(centers, axis=0)
        std_dev = np.std(centers, axis=0)

        logging.info("Mean Center: %s", mean_center)
        logging.info("Std Dev: %s", std_dev)

        # Assert consistency
        # Since we are testing different brightnesses/durations, the ROI might shift slightly
        # due to blooming or different effective center of brightness.
        # So we allow a larger margin than the consistency test (which repeats the exact same condition).
        assert np.max(std_dev) < 10.0, f"ROI detection inconsistent across pulse tests! Std Dev: {std_dev}"


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
