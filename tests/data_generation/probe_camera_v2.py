"""Probe camera FPS at various manual exposure settings."""

import time
import logging

import cv2

logging.basicConfig(level=logging.INFO)


def probe_camera_fps_exposure(index=0):
    """Measure actual FPS at different manual exposure levels."""
    cap = cv2.VideoCapture(index, cv2.CAP_V4L2)
    if not cap.isOpened():
        logging.error("Could not open camera %d", index)
        return

    # Set common format
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    # Set FPS to 30
    cap.set(cv2.CAP_PROP_FPS, 30)

    # Test exposures
    exposures = [157, 100, 50, 10, 1]  # Decreasing exposure

    for exp in exposures:
        logging.info("--- Testing Manual Exposure: %d ---", exp)

        # Turn off Auto Exposure
        cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
        cap.set(cv2.CAP_PROP_EXPOSURE, exp)

        # Wait for settle
        time.sleep(2)

        # Measure FPS over 100 frames
        start = time.time()
        for _ in range(100):
            ret, _ = cap.read()
            if not ret:
                logging.error("Failed to read frame")
                break
        end = time.time()

        duration = end - start
        fps = 100 / duration
        logging.info("Target Exposure: %d -> Measured FPS: %.2f", exp, fps)

    cap.release()


if __name__ == "__main__":
    probe_camera_fps_exposure()
