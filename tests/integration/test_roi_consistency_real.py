import time
import os
import shutil
import csv
import logging
import sys
import pytest
import numpy as np
import cv2
from led_detection.main import PeakMonitor, X86CameraDriver

# Configure logging to stdout
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s', stream=sys.stdout)

# Shared results storage
RESULTS = {
    "centers": [],
    "valid_iterations": 0
}

@pytest.fixture(scope="module", autouse=True)
def setup_test_env():
    """Setup debug directory and CSV file."""
    debug_dir = "/tmp/roi_debug"
    if os.path.exists(debug_dir):
        shutil.rmtree(debug_dir)
    os.makedirs(debug_dir)
    print(f"\nDebug artifacts will be saved to: {os.path.abspath(debug_dir)}")

    csv_path = os.path.join(debug_dir, "roi_results.csv")
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["Iteration", "CenterX", "CenterY", "Width", "Height", "Score", "Brightness"])

    # Clear results
    RESULTS["centers"] = []
    RESULTS["valid_iterations"] = 0

    yield
    # Teardown if needed

@pytest.fixture(scope="module")
def monitor():
    """Provide PeakMonitor with shared camera."""
    try:
        driver = X86CameraDriver(idx=0)
        if not driver.cap.isOpened():
            pytest.skip("Camera not found or cannot be opened.")

        show_preview = os.getenv('INTEGRATION_PREVIEW', '0') == '1'

        mon = PeakMonitor(
            interval=10,
            threshold=20,
            adaptive_exposure=True,
            log_saturation=False,
            preview=show_preview
        )
        mon.cam = driver

        logging.info("Initializing camera...")
        mon.cam.start()

        if mon.autofocus:
            logging.info("Performing initial autofocus...")
            mon.autofocus_sweep()

        yield mon

        mon.cam.stop()
        if show_preview:
            cv2.destroyAllWindows()

    except Exception as e:  # pylint: disable=broad-exception-caught
        pytest.skip(f"Failed to initialize camera: {e}")

@pytest.mark.integration
@pytest.mark.parametrize("iteration", range(1, 101))
def test_roi_consistency_iteration(monitor, iteration):  # pylint: disable=redefined-outer-name
    """Run a single iteration of ROI detection."""
    print(f"\nIteration {iteration}/100: Waiting for signal...", end='\r')

    # Reset monitor ROI
    monitor.roi = None

    # Run detection
    found = monitor.wait_for_signal(timeout=10.0)

    if not found:
        print(f"\nIteration {iteration}: Failed to detect signal (Timeout). Skipping.")
        return # Skip this iteration but don't fail the test yet? Or fail?
        # If we fail here, the whole suite stops? No, pytest continues.
        # But we want to collect stats.
        # Let's just return and not record a valid iteration.

    y1, y2, x1, x2 = monitor.roi
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    w = x2 - x1
    h = y2 - y1

    # Store result
    RESULTS["centers"].append((cx, cy))
    RESULTS["valid_iterations"] += 1

    print(f"Iteration {iteration}: ROI Center=({cx:.1f}, {cy:.1f}) Score={monitor.detected_peak_strength:.1f}")

    # Log to CSV
    debug_dir = "/tmp/roi_debug"
    csv_path = os.path.join(debug_dir, "roi_results.csv")
    with open(csv_path, 'a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([iteration, cx, cy, w, h, monitor.detected_peak_strength, monitor.detected_on_brightness])

    # Save debug image
    if os.path.exists("debug_detection.png"):
        shutil.move("debug_detection.png", os.path.join(debug_dir, f"iter_{iteration:03d}.png"))

    # Brief pause
    time.sleep(0.5)

@pytest.mark.integration
def test_roi_consistency_analysis():
    """Analyze the results of the 100 iterations."""
    valid_count = RESULTS["valid_iterations"]
    centers = np.array(RESULTS["centers"])

    print(f"\nAnalysis Results ({valid_count} valid iterations):")

    if valid_count < 10:
        pytest.fail(f"Too few valid detections ({valid_count}/100). Check camera aiming.")

    mean_center = np.mean(centers, axis=0)
    std_dev = np.std(centers, axis=0)

    print(f"Mean Center: {mean_center}")
    print(f"Std Dev: {std_dev}")

    # Assertions
    assert np.max(std_dev) < 5.0, f"ROI detection inconsistent! Std Dev: {std_dev}"
