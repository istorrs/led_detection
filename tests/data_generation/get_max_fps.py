#!/usr/bin/env python3
"""Detect the maximum FPS supported by the connected camera."""

import sys
import time
import logging

import cv2

# Configure logging to stderr so it doesn't pollute stdout
logging.basicConfig(stream=sys.stderr, level=logging.INFO)


def get_max_fps(index=0):
    """Open the camera and measure its maximum frame rate."""
    cap = cv2.VideoCapture(index, cv2.CAP_V4L2)
    if not cap.isOpened():
        logging.error("Could not open camera %d", index)
        print("0")  # Return 0 on failure
        return

    # Set common format
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    # Request 30 FPS
    cap.set(cv2.CAP_PROP_FPS, 30)

    # Force low exposure to ensure we are testing max transport speed
    # Auto-Exposure Off
    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
    # Exposure = 10 (approx 1-3ms depending on scale, definitely fast enough for 30fps)
    cap.set(cv2.CAP_PROP_EXPOSURE, 10)

    # Warmup: Read frames for 2 seconds to let AE/AGC settle and flush buffer
    warmup_start = time.time()
    while time.time() - warmup_start < 2.0:
        cap.read()

    # Measure
    count = 50
    start = time.time()
    for _ in range(count):
        ret, _ = cap.read()
        if not ret:
            break
    end = time.time()
    cap.release()

    duration = end - start
    if duration > 0:
        fps = count / duration
    else:
        fps = 0.0

    logging.info("Measured Max FPS: %.2f", fps)

    # Output integer FPS to stdout (rounded, not truncated)
    print(round(fps))


if __name__ == "__main__":
    try:
        get_max_fps()
    except (OSError, cv2.error) as exc:
        logging.error("Error: %s", exc)
        print("0")
