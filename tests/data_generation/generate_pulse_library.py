"""
Generate a library of LED pulse data for verification.
Controls LED hardware and captures video/frames for various pulse parameters.
"""

import os
import sys
import time
import json
import logging
import argparse
import shutil

import cv2

# Ensure we can import from src and tests
# This assumes we run from the repo root
sys.path.append(os.getcwd())

# pylint: disable=wrong-import-position  # Must be after sys.path.append
from tests.integration.led_controller import LEDController
from led_detection.main import get_driver

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Parameter Grid
PULSE_DURATIONS = [0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000] # ms
PULSE_PERIODS = [1000, 2000, 5000, 10000] # ms
PULSE_BRIGHTNESS = [100] # %

OUTPUT_DIR = "/tmp/pulse_data"

def generate_data(fps=30, color=False, dry_run=False, limit=None):
    # pylint: disable=too-many-locals,too-many-nested-blocks
    """
    Generate pulse data.

    Args:
        dry_run (bool): If True, simulate actions without hardware.
        limit (int): Maximum number of cases to process (for testing).
    """
    if os.path.exists(OUTPUT_DIR):
        logging.info("Clearing output directory: %s", OUTPUT_DIR)
        shutil.rmtree(OUTPUT_DIR)

    os.makedirs(OUTPUT_DIR)

    registry = []

    # Initialize Hardware
    if not dry_run:
        led = LEDController()
        if not led.connect():
            logging.error("Failed to connect to LED Controller!")
            sys.exit(1)

        cam = get_driver()
        cam.start()

        # Enable Hardware Auto-Focus and Auto-Exposure
        if hasattr(cam, 'cap'):
            logging.info("Enabling Hardware Auto-Focus and Auto-Exposure...")
            # Auto-Focus (V4L2 1=Manual, ? Actually standard varies, but for V4L2 usually 1=Auto in some contexts or separate ctrl)
            # cap.set(cv2.CAP_PROP_AUTOFOCUS, 1) # 1 is often auto
            # But let's check what main.py did. It used 0 for manual. So 1 is likely auto.
            cam.cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)

            # Auto-Exposure (V4L2: 1=Manual, 3=Auto)
            cam.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 3)

            # Wait for them to settle
            logging.info("Waiting 5 seconds for camera to settle...")
            time.sleep(5)
        else:
            time.sleep(2)
    else:
        logging.info("DRY RUN: Hardware initialization skipped.")

    count = 0

    try:
        min_safe_duration = (1000.0 / fps) * 1.5
        logging.info("Filtering pulses shorter than %.1fms (for %d FPS)", min_safe_duration, fps)

        for duration in PULSE_DURATIONS:
            if duration < min_safe_duration:
                continue

            for period in PULSE_PERIODS:
                # Skip invalid combinations (period must be > duration)
                # Enforce a reasonable duty cycle limit if needed, but hardware handles most.
                # Let's say period should be at least duration + 10ms
                if period <= duration + 10:
                    continue

                for brightness in PULSE_BRIGHTNESS:
                    if limit and count >= limit:
                        logging.info("Limit reached (%d). Stopping.", limit)
                        break

                    case_name = f"pulse_{duration}ms_{period}ms_{brightness}pct"
                    case_dir = os.path.join(OUTPUT_DIR, case_name)

                    logging.info("Processing Case %d: %s", count+1, case_name)

                    if not dry_run:
                        # 1. Setup LED
                        if not led.set_pulse(duration, period, brightness):
                            logging.error("Failed to set pulse %s", case_name)
                            continue

                        # Wait a bit for stable pulsing
                        time.sleep(1.0) # Wait 1 second (at least one period usually, or part of it)

                        # 2. Setup Recording
                        if not os.path.exists(case_dir):
                            os.makedirs(case_dir)

                        # We want to capture 3 cycles.
                        # capture_duration = 3 * period_ms / 1000.0
                        capture_duration_sec = (3 * period) / 1000.0
                        # Add a buffer
                        capture_duration_sec += 1.0

                        logging.info("Recording for %.1fs...", capture_duration_sec)

                        start_time = time.time()
                        frames = []

                        while time.time() - start_time < capture_duration_sec:
                            frame = cam.get_frame(color=color)
                            if frame is not None:
                                timestamp = time.time()
                                frames.append((timestamp, frame))
                            # 30 FPS approx
                            time.sleep(0.01) # Small delay to not busy loop too hard, though get_frame might block

                        logging.info("Captured %d frames.", len(frames))

                        # 3. Save Data
                        # Save Images
                        for i, (ts, img) in enumerate(frames):
                            # Relative time from start
                            rel_ts = ts - start_time
                            fname = f"LED-frame-{i:04d}_{rel_ts:.3f}.jpg"

                            # Overlay frame number
                            img_copy = img.copy()
                            cv2.putText(img_copy, f"Frame: {i}", (10, 30),
                                      cv2.FONT_HERSHEY_SIMPLEX, 1, (255), 2)

                            cv2.imwrite(os.path.join(case_dir, fname), img_copy)

                        # Save Video
                        if frames:
                            h, w = frames[0][1].shape[:2]
                            video_path = os.path.join(case_dir, "video.mp4")
                            # MP4V for compatibility
                            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                            # Estimate FPS
                            duration_actual = frames[-1][0] - frames[0][0]
                            fps_est = len(frames) / duration_actual if duration_actual > 0 else 30.0

                            is_color = len(frames[0][1].shape) == 3
                            out = cv2.VideoWriter(video_path, fourcc, fps_est, (w, h), isColor=is_color)
                            for _, img in frames:
                                # VideoWriter expects BGR usually, even if isColor=False?
                                # If isColor=False, it expects single channel.
                                # Our frames are grayscale (from get_frame() usually returning gray for analysis,
                                # but wait, X86CameraDriver.get_frame returns COLOR_BGR2GRAY).
                                # So they are 2D arrays.
                                out.write(img)
                            out.release()

                    # Add to registry
                    registry.append({
                        "name": case_name,
                        "directory": os.path.abspath(case_dir),
                        "expected_count": 3,
                        "expected_frames": [],
                        "repeat_interval_sec": period / 1000.0,
                        "aed_type": "SimulatedPulse",
                        # Metadata not in strict schema but useful
                        "effective_frame_rate": fps_est,
                        "duration_ms": duration,
                        "period_ms": period,
                        "brightness_pct": brightness
                    })

                    count += 1

                if limit and count >= limit:
                    break
            if limit and count >= limit:
                break

    except KeyboardInterrupt:
        logging.info("Interrupted by user.")
    finally:
        if not dry_run:
            led.stop_pulse()
            led.disconnect()
            cam.stop()

        # Save Registry
        with open(os.path.join(OUTPUT_DIR, "pulse_library.json"), "w", encoding="utf-8") as f:
            json.dump(registry, f, indent=4)

        logging.info("Generation complete. Saved %d entries to %s/pulse_library.json", len(registry), OUTPUT_DIR)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate LED pulse data library.")
    parser.add_argument("--dry-run", action="store_true", help="Simulate without hardware.")
    parser.add_argument("--fps", type=int, default=30, help="Camera frame rate.")
    parser.add_argument("--color", action="store_true", help="Capture color frames.")
    parser.add_argument("--limit", type=int, help="Limit number of cases.")
    args = parser.parse_args()

    generate_data(fps=args.fps, color=args.color, dry_run=args.dry_run, limit=args.limit)
