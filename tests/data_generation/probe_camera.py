"""Probe camera to test supported FPS values."""

import logging

import cv2

logging.basicConfig(level=logging.INFO)


def probe_camera(index=0):
    """Open camera and test various FPS settings."""
    cap = cv2.VideoCapture(index, cv2.CAP_V4L2)
    if not cap.isOpened():
        logging.error("Could not open camera %d", index)
        return

    # Set common format first
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    # Test target FPS values
    test_fps = [11, 25, 30, 60]

    for fps in test_fps:
        logging.info("--- Testing FPS: %d ---", fps)

        success = cap.set(cv2.CAP_PROP_FPS, fps)
        actual = cap.get(cv2.CAP_PROP_FPS)
        logging.info("Requested: %d, Set Success: %s, Actual: %s",
                     fps, success, actual)

    cap.release()


if __name__ == "__main__":
    probe_camera()
