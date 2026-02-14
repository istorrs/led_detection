"""
Generate a library of LED pulse data for verification.
Controls LED hardware and captures video/frames for various pulse parameters.
"""

import os
import sys
import time
import json
import math
import logging
import argparse
import shutil
import uuid

import cv2

# Ensure we can import from src and tests
# This assumes we run from the repo root
sys.path.append(os.getcwd())
sys.path.append(os.path.join(os.getcwd(), 'src'))

# pylint: disable=wrong-import-position  # Must be after sys.path.append
from tests.integration.led_controller import LEDController  # pylint: disable=import-error
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

# Settle time in seconds to allow camera exposure to stabilise before
# pulse data is considered usable.
SETTLE_TIME_SECONDS = 2.0

# Default frame count used as fallback when no AED period is available.
DEFAULT_FRAME_COUNT = 360

# Maximum JPEG file size in bytes (128 KB).
MAX_JPEG_BYTES = 128 * 1024


def imwrite_capped(path, img, max_bytes=MAX_JPEG_BYTES):
    """Write a JPEG, iteratively reducing quality until it fits under *max_bytes*.

    Args:
        path: Destination file path.
        img: Image array (BGR or grayscale).
        max_bytes: Maximum file size in bytes.

    Returns:
        The JPEG quality value that was used.
    """
    for quality in range(95, 10, -5):
        success, buf = cv2.imencode(
            '.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, quality]
        )
        if success and len(buf) <= max_bytes:
            with open(path, 'wb') as fh:
                fh.write(buf)
            return quality
    # Last resort – write at minimum quality
    cv2.imwrite(path, img, [cv2.IMWRITE_JPEG_QUALITY, 10])
    return 10

def compute_frame_count(aed_type_id, fps, aed_types_path=None):
    """Compute the number of frames needed for 2 pulse periods plus settle time.

    Args:
        aed_type_id: Integer AED type identifier from aed_types.json.
        fps: Capture frame rate.
        aed_types_path: Optional path to aed_types.json.  When *None* the
            file is resolved relative to the repository root.

    Returns:
        Integer frame count.
    """
    if aed_types_path is None:
        aed_types_path = os.path.join(
            os.path.dirname(__file__), "..", "..", "aed_types.json"
        )
        aed_types_path = os.path.normpath(aed_types_path)

    try:
        with open(aed_types_path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
    except (OSError, json.JSONDecodeError) as exc:
        logging.warning(
            "Could not load %s: %s – using default %d frames",
            aed_types_path, exc, DEFAULT_FRAME_COUNT,
        )
        return DEFAULT_FRAME_COUNT

    for aed in data.get("aed_types", []):
        if aed.get("id") == aed_type_id:
            period = aed["params"].get("blink_period_seconds", -1)
            if period <= 0:
                logging.info(
                    "AED '%s' is aperiodic – using default %d frames",
                    aed.get("name", "?"), DEFAULT_FRAME_COUNT,
                )
                return DEFAULT_FRAME_COUNT
            total_seconds = SETTLE_TIME_SECONDS + 2.0 * period
            count = int(math.ceil(total_seconds * fps))
            logging.info(
                "AED '%s': period=%.2fs, settle=%.1fs, "
                "total=%.2fs -> %d frames @ %d FPS",
                aed.get("name", "?"), period,
                SETTLE_TIME_SECONDS, total_seconds, count, fps,
            )
            return count

    logging.warning(
        "AED type ID %s not found – using default %d frames",
        aed_type_id, DEFAULT_FRAME_COUNT,
    )
    return DEFAULT_FRAME_COUNT


def software_autofocus(cam, color=True):
    # pylint: disable=too-many-locals
    """
    Find optimal focus using hill-climbing algorithm with sharpness measurement.
    Adapted from main.py's autofocus_sweep method.
    """
    logging.info("Starting software autofocus...")

    if not hasattr(cam, 'cap'):
        logging.warning("Camera does not support focus control")
        return None

    # Disable hardware autofocus first
    cam.cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)

    # Detect camera's focus range
    initial_focus = int(cam.cap.get(cv2.CAP_PROP_FOCUS))
    logging.info("Detecting camera focus range (initial: %d)...", initial_focus)

    # Test upper boundary
    test_upper = initial_focus + 1000
    cam.cap.set(cv2.CAP_PROP_FOCUS, test_upper)
    time.sleep(0.1)
    actual_upper = int(cam.cap.get(cv2.CAP_PROP_FOCUS))

    # Test lower boundary
    cam.cap.set(cv2.CAP_PROP_FOCUS, 0)
    time.sleep(0.1)
    actual_lower = int(cam.cap.get(cv2.CAP_PROP_FOCUS))

    # Restore initial focus
    cam.cap.set(cv2.CAP_PROP_FOCUS, initial_focus)
    time.sleep(0.1)

    # Determine focus range
    focus_min = min(actual_lower, initial_focus)
    focus_max = max(actual_upper, initial_focus)
    logging.info("Camera focus range: %d to %d", focus_min, focus_max)

    def measure_sharpness(focus_pos, settle_time=0.1):
        """Set focus and measure sharpness after settling."""
        cam.cap.set(cv2.CAP_PROP_FOCUS, focus_pos)
        time.sleep(settle_time)

        frame = cam.get_frame(color=color)
        if frame is None:
            return 0

        # Convert to grayscale if color
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame

        # Measure sharpness in center region using Laplacian variance
        h, w = gray.shape[:2]
        center_h, center_w = h // 2, w // 2
        roi_size = min(h, w) // 3
        y1, y2 = center_h - roi_size // 2, center_h + roi_size // 2
        x1, x2 = center_w - roi_size // 2, center_w + roi_size // 2

        center_roi = gray[y1:y2, x1:x2]
        sharpness = cv2.Laplacian(center_roi, cv2.CV_64F).var()

        return sharpness

    # Measure initial sharpness
    initial_sharpness = measure_sharpness(initial_focus)
    logging.info("Initial focus: %d, sharpness: %.2f", initial_focus, initial_sharpness)

    # Phase 1: Coarse linear scan
    focus_range = focus_max - focus_min
    coarse_step = max(5, focus_range // 20)

    best_focus = initial_focus
    best_sharpness = initial_sharpness

    logging.info("Phase 1: Coarse scan (step=%d)...", coarse_step)
    for focus_pos in range(focus_min, focus_max + 1, coarse_step):
        sharpness = measure_sharpness(focus_pos)
        if sharpness > best_sharpness:
            best_focus = focus_pos
            best_sharpness = sharpness
            logging.info("  New best: focus=%d, sharpness=%.2f", best_focus, best_sharpness)

    # Phase 2: Binary search around best coarse focus
    left = max(focus_min, best_focus - coarse_step)
    right = min(focus_max, best_focus + coarse_step)
    logging.info("Phase 2: Fine search (%d to %d)...", left, right)

    while left <= right:
        mid = (left + right) // 2
        mid_sharp = measure_sharpness(mid)

        left_sharp = measure_sharpness(mid - 1) if mid - 1 >= focus_min else None
        right_sharp = measure_sharpness(mid + 1) if mid + 1 <= focus_max else None

        if left_sharp is not None and left_sharp > mid_sharp:
            right = mid - 1
        elif right_sharp is not None and right_sharp > mid_sharp:
            left = mid + 1
        else:
            best_focus = mid
            best_sharpness = mid_sharp
            logging.info("  Peak found: focus=%d, sharpness=%.2f", best_focus, best_sharpness)
            break

    # Apply and lock best focus
    improvement_pct = ((best_sharpness - initial_sharpness) / initial_sharpness * 100) \
                     if initial_sharpness > 0 else 0

    if best_sharpness > initial_sharpness * 1.05:
        logging.info("✓ Autofocus improved: %d→%d (+%.1f%%)",
                    initial_focus, best_focus, improvement_pct)
    else:
        best_focus = initial_focus
        logging.info("✗ No improvement, keeping initial focus: %d", initial_focus)

    # Lock focus (do it multiple times for reliability)
    for _ in range(3):
        cam.cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
        time.sleep(0.05)
        cam.cap.set(cv2.CAP_PROP_FOCUS, best_focus)
        time.sleep(0.05)

    final_focus = int(cam.cap.get(cv2.CAP_PROP_FOCUS))
    logging.info("Focus locked at: %d", final_focus)

    return best_focus


def generate_data(fps=30, color=False, dry_run=False, limit=None,
                  duration=None, period=None, brightness=None,
                  output_dir=OUTPUT_DIR, case_name=None, serial_port="/dev/ttyUSB1",
                  frame_count=None, aed_type_id=None, aed_name=None, run_id=None):
    # pylint: disable=too-many-locals,too-many-nested-blocks,too-many-arguments
    # pylint: disable=too-many-positional-arguments
    """
    Generate pulse data.

    Args:
        fps (int): Camera frame rate.
        color (bool): Capture color frames.
        dry_run (bool): If True, simulate actions without hardware.
        limit (int): Maximum number of cases to process (for testing).
        duration (float): Pulse duration in ms (optional override).
        period (float): Pulse period in ms (optional override).
        brightness (int): Pulse brightness % (optional override).
        output_dir (str): Directory to save data.
        case_name (str): Custom name for the case directory.
        serial_port (str): Serial port for LED controller.
        frame_count (int): Number of frames to capture.  When *None* the
            count is computed automatically from the AED pulse period.
        aed_type_id (int): ID of the AED type.
        aed_name (str): Name of the AED type.
    """
    # --- Auto-compute frame_count when not explicitly provided ---
    if frame_count is None:
        if aed_type_id is not None:
            frame_count = compute_frame_count(aed_type_id, fps)
        else:
            frame_count = DEFAULT_FRAME_COUNT
            logging.info(
                "No AED type supplied – using default %d frames",
                frame_count,
            )

    # Determine if we are running a single case or the grid
    single_case = duration is not None and period is not None and brightness is not None

    if single_case:
        logging.info("Generating single case: Duration=%.1fms, Period=%.1fms, Brightness=%d%%, Frames=%d",
                     duration, period, brightness, frame_count)
        # If running a single case, we might not want to clear the whole directory if it's the shared one,
        # unless it's a specific run. But safe bet is to just ensure the specific case dir exists.
        # However, the original logic cleared OUTPUT_DIR.
        # Let's preserve existing data if output_dir is provided or if we are doing single case,
        # essentially only clear if we are doing a full grid run and using default dir?
        # For safety/simplicity in this script's original design, it cleared everything.
        # But for the shell script usage, we want to accumulate.
        # Let's NOT clear the output directory if we are in single_case mode.
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
    else:
        # Full grid run - clear default output dir to be clean
        if output_dir == OUTPUT_DIR and os.path.exists(output_dir):
            logging.info("Clearing output directory: %s", output_dir)
            shutil.rmtree(output_dir)
        elif not os.path.exists(output_dir):
            os.makedirs(output_dir)

    registry = []
    registry_path = os.path.join(output_dir, "pulse_library.json")

    # If appending (single case) and registry exists, load it
    if single_case and os.path.exists(registry_path):
        try:
            with open(registry_path, "r", encoding="utf-8") as f:
                registry = json.load(f)
        except json.JSONDecodeError:
            logging.warning("Failed to load existing registry, starting fresh.")

    # Initialize Hardware
    if not dry_run:
        led = LEDController(port=serial_port)
        if not led.connect():
            logging.error("Failed to connect to LED Controller on %s!", serial_port)
            sys.exit(1)

        cam = get_driver()
        cam.start()

        # Enable Hardware Auto-Focus and Auto-Exposure
        if hasattr(cam, 'cap'):
            logging.info("Setting Camera FPS to %d", fps)
            cam.cap.set(cv2.CAP_PROP_FPS, fps)
            actual_fps = cam.cap.get(cv2.CAP_PROP_FPS)
            logging.info("Actual Camera FPS: %s", actual_fps)

            logging.info("Enabling Auto-Exposure to gauge scene...")
            cam.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 3) # 3 is Auto

            # Wait for exposure to settle
            logging.info("Waiting 3 seconds for camera to settle...")
            time.sleep(3)

            # Read current exposure
            current_exposure = cam.cap.get(cv2.CAP_PROP_EXPOSURE)
            logging.info("Camera settled at Exposure: %s", current_exposure)

            # Exposure Cap: If exposure is too high, it causes frame drops (integration > 33ms)
            # 157 was observed to cause ~25 FPS. We want to be safe for 30 FPS.
            # Heuristic: If > 120, reduce by 25% to prioritize FPS.
            if current_exposure > 120:
                new_exposure = int(current_exposure * 0.75)
                logging.warning("Exposure %s is too high for 30 FPS! reducing to %s", current_exposure, new_exposure)
                current_exposure = new_exposure

            # Lock to Manual Exposure
            logging.info("Locking Exposure to %s", current_exposure)
            cam.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25) # Manual
            cam.cap.set(cv2.CAP_PROP_EXPOSURE, current_exposure)

            # Software autofocus (hill-climbing algorithm)
            software_autofocus(cam, color=color)

            # Re-enforce FPS in case exposure change affected it
            logging.info("Re-enforcing FPS to %d", fps)
            cam.cap.set(cv2.CAP_PROP_FPS, fps)
            actual_fps_after = cam.cap.get(cv2.CAP_PROP_FPS)
            logging.info("Camera FPS after lock: %s", actual_fps_after)

        else:
            time.sleep(2)
    else:
        logging.info("DRY RUN: Hardware initialization skipped.")

    try:
        if single_case:
            # Create a single item list to iterate over once
            # We treat it as one item in the loops
            durations = [duration]
            periods = [period]
            brights = [brightness]
        else:
            durations = PULSE_DURATIONS
            periods = PULSE_PERIODS
            brights = PULSE_BRIGHTNESS

        count = 0
        min_safe_duration = (1000.0 / fps) * 1.5
        logging.info("Filtering pulses shorter than %.1fms (for %d FPS)", min_safe_duration, fps)

        for d in durations:
            if d < min_safe_duration and not single_case:
                logging.info("Skipping duration %.1fms (too short for FPS)", d)
                continue
            # If single_case is forced, we might want to warn but proceed?
            # Or just enforce physics.
            if single_case and d < min_safe_duration:
                logging.warning("WARNING: Duration %.1fms is very short for %d FPS! Detection may be unreliable.", d, fps)

            for p in periods:
                if p <= d + 10 and not single_case:
                    continue

                for b in brights:
                    if limit and count >= limit:
                        logging.info("Limit reached (%d). Stopping.", limit)
                        break

                    if single_case and case_name:
                        current_case_name = case_name
                    else:
                        current_case_name = f"pulse_{d}ms_{p}ms_{b}pct"

                    # Use provided run_id or generate one
                    unique_id = run_id if run_id else str(uuid.uuid4())[:8]
                    case_dir = os.path.join(output_dir, f"{current_case_name}_{unique_id}")

                    logging.info("Processing Case: %s", current_case_name)

                    # Initialize variables for metadata scope
                    frames = []
                    fps_est = 0.0

                    if not dry_run:
                        # 1. Setup LED
                        if not led.set_pulse(d, p, b):
                            logging.error("Failed to set pulse %s", current_case_name)
                            continue

                        # Wait a bit for stable pulsing
                        time.sleep(1.0) # Wait 1 second

                        # 2. Setup Recording
                        if not os.path.exists(case_dir):
                            os.makedirs(case_dir)

                        logging.info("Recording frames at native rate...")

                        start_time = time.time()
                        all_frames = []  # (timestamp, frame) tuples

                        logging.info(
                            "Capturing %d frames at native rate...",
                            frame_count,
                        )

                        # Timeout based on expected capture duration
                        timeout_sec = (frame_count / 10.0) + 30.0

                        while len(all_frames) < frame_count:
                            if time.time() - start_time > timeout_sec:
                                logging.warning(
                                    "Capture timed out! Got %d frames.",
                                    len(all_frames),
                                )
                                break

                            frame = cam.get_frame(color=color)
                            now = time.time()

                            if frame is not None:
                                all_frames.append((now, frame.copy()))

                                # Progress every 50 frames
                                if len(all_frames) % 50 == 0:
                                    sys.stdout.write(
                                        f"\rCaptured {len(all_frames)}"
                                        f"/{frame_count}"
                                    )
                                    sys.stdout.flush()
                            else:
                                time.sleep(0.001)

                        print("")  # Newline after progress

                        # Calculate native FPS from capture
                        if len(all_frames) >= 2:
                            duration_actual = all_frames[-1][0] - all_frames[0][0]
                            native_fps_measured = len(all_frames) / duration_actual if duration_actual > 0 else fps
                        else:
                            native_fps_measured = fps
                            duration_actual = 0

                        logging.info("Captured %d frames in %.2fs (Native FPS: %.2f)",
                                     len(all_frames), duration_actual, native_fps_measured)

                        # 3. Save Data - Create BOTH native FPS and 11 FPS datasets
                        import numpy as np  # pylint: disable=import-outside-toplevel

                        # --- Native FPS Dataset (first frame_count frames) ---
                        native_frames = all_frames[:frame_count]
                        native_dir = case_dir  # Use original case_dir for native

                        logging.info(
                            "Saving Native FPS dataset (%d frames) to %s",
                            len(native_frames), native_dir,
                        )

                        for i, (_, img) in enumerate(native_frames):
                            fname = f"LED-frame-{i:04d}.jpg"
                            imwrite_capped(os.path.join(native_dir, fname), img)

                        # Save timestamps for native dataset
                        csv_path = os.path.join(native_dir, "timestamps.csv")
                        with open(csv_path, "w", encoding="utf-8") as f:
                            f.write("frame_idx,timestamp,relative_time\n")
                            start_t = native_frames[0][0]
                            for idx, (t, _) in enumerate(native_frames):
                                f.write(f"{idx},{t:.6f},{t-start_t:.6f}\n")

                        # Save video for native dataset
                        if native_frames:
                            h, w = native_frames[0][1].shape[:2]
                            video_path = os.path.join(native_dir, "video.mp4")
                            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                            is_color = len(native_frames[0][1].shape) == 3
                            out = cv2.VideoWriter(video_path, fourcc, native_fps_measured, (w, h), isColor=is_color)
                            for _, img in native_frames:
                                out.write(img)
                            out.release()

                        # Native FPS metadata
                        native_metadata = {
                            "aed_type": aed_type_id,
                            "aed_name": aed_name,
                            "frame_count": len(native_frames),
                            "FPS": native_fps_measured
                        }
                        with open(os.path.join(native_dir, ".metadata"), "w", encoding="utf-8") as f:
                            json.dump(native_metadata, f, indent=4)

                        fps_est = native_fps_measured  # For registry entry

                        # --- Generate all applicable FPS datasets via temporal averaging ---
                        # Determine which FPS datasets we can produce (must be <= native FPS)
                        target_fps_list = []
                        if native_fps_measured >= 29:
                            target_fps_list = [30, 25, 11]
                        elif native_fps_measured >= 24:
                            target_fps_list = [25, 11]
                        elif native_fps_measured >= 10:
                            target_fps_list = [11]
                        else:
                            target_fps_list = []  # Can't produce any standard datasets
                            logging.warning("Native FPS (%.1f) too low for any target datasets!", native_fps_measured)

                        logging.info("Native FPS: %.2f -> Will create datasets for: %s",
                                     native_fps_measured, target_fps_list)

                        for target_fps in target_fps_list:
                            # Skip if target equals native (already saved above as native_dir)
                            if abs(target_fps - native_fps_measured) < 2:
                                logging.info("Skipping %d FPS (same as native)", target_fps)
                                continue

                            # Create directory for this FPS
                            fps_dir = case_dir.replace(f"_{fps}fps_", f"_{target_fps}fps_")
                            if fps_dir == case_dir:
                                fps_dir = case_dir + f"_{target_fps}fps"

                            if not os.path.exists(fps_dir):
                                os.makedirs(fps_dir)

                            # Compute target output frame count for this FPS
                            # using the same time-duration the native dataset
                            # covers.
                            target_output_frames = int(math.ceil(
                                frame_count * target_fps / fps
                            )) if fps > 0 else frame_count

                            # Frames from source = duration * native_fps
                            target_duration = float(target_output_frames) / target_fps
                            frames_needed = int(target_duration * native_fps_measured)
                            if frames_needed > len(all_frames):
                                frames_needed = len(all_frames)
                                logging.warning(
                                    "Not enough frames for %d FPS. "
                                    "Using %d frames instead of ideal.",
                                    target_fps, frames_needed,
                                )

                            source_subset = all_frames[:frames_needed]
                            ratio = (
                                len(source_subset) / float(target_output_frames)
                            )

                            logging.info(
                                "Creating %d FPS dataset (%d output frames, "
                                "ratio %.2f from %d source frames)",
                                target_fps, target_output_frames,
                                ratio, len(source_subset),
                            )

                            output_frames = []
                            for i in range(target_output_frames):
                                start_idx = int(i * ratio)
                                end_idx = int((i + 1) * ratio)
                                end_idx = min(end_idx, len(source_subset))

                                window_frames = [source_subset[j][1] for j in range(start_idx, end_idx)]

                                if len(window_frames) == 1:
                                    avg_frame = window_frames[0]
                                elif len(window_frames) > 1:
                                    stacked = np.stack(window_frames, axis=0).astype(np.float32)
                                    avg_frame = np.mean(stacked, axis=0).astype(np.uint8)
                                else:
                                    continue

                                output_frames.append((source_subset[start_idx][0], avg_frame))

                            # Save frames
                            for i, (_, img) in enumerate(output_frames):
                                fname = f"LED-frame-{i:04d}.jpg"
                                imwrite_capped(os.path.join(fps_dir, fname), img)

                            # Save timestamps
                            csv_path_fps = os.path.join(fps_dir, "timestamps.csv")
                            with open(csv_path_fps, "w", encoding="utf-8") as f:
                                f.write("frame_idx,timestamp,relative_time\n")
                                start_t = output_frames[0][0]
                                for idx, (t, _) in enumerate(output_frames):
                                    f.write(f"{idx},{t:.6f},{t-start_t:.6f}\n")

                            # Save video
                            if output_frames:
                                h, w = output_frames[0][1].shape[:2]
                                video_path_fps = os.path.join(fps_dir, "video.mp4")
                                is_color = len(output_frames[0][1].shape) == 3
                                out_fps = cv2.VideoWriter(video_path_fps, fourcc, float(target_fps), (w, h), isColor=is_color)
                                for _, img in output_frames:
                                    out_fps.write(img)
                                out_fps.release()

                            # Save metadata
                            fps_metadata = {
                                "aed_type": aed_type_id,
                                "aed_name": aed_name,
                                "frame_count": len(output_frames),
                                "FPS": float(target_fps)
                            }
                            with open(os.path.join(fps_dir, ".metadata"), "w", encoding="utf-8") as f:
                                json.dump(fps_metadata, f, indent=4)

                            logging.info("Saved %d FPS dataset (%d frames) to %s",
                                        target_fps, len(output_frames), fps_dir)

                        logging.info("All applicable datasets saved successfully.")

                    # Create .metadata file
                    if single_case and aed_type_id is not None and aed_name is not None:
                        metadata = {
                            "aed_type": aed_type_id,
                            "aed_name": aed_name,
                            "frame_count": len(frames), # Use actual captured count
                            "FPS": fps_est if fps_est > 0 else fps # Use effective FPS, fallback to target if 0 (failed)
                        }
                        # We are creating .metadata inside the case_dir (e.g. heartstart_frx_30fps/.metadata)
                        # Wait, user request said "default .metadata file in each pulseset".
                        # Assuming pulseset == case directory.
                        # Check where to save it. If case_dir is created, save it there.
                        if not os.path.exists(case_dir):
                            # In dry run, case_dir might not exist if logic was skipped,
                            # but we simulate creation in dry run usually or just skip I/O.
                            # In dry run, we should probably output log or skip file creation.
                            if dry_run:
                                logging.info("DRY RUN: Creating .metadata in %s", case_dir)
                            else:
                                os.makedirs(case_dir)

                        if not dry_run or (dry_run and os.path.exists(case_dir)):
                            with open(os.path.join(case_dir, ".metadata"), "w", encoding="utf-8") as f:
                                json.dump(metadata, f, indent=4)

                    # Add to registry (avoid duplicates if possible)
                     # Remove existing entry with same name if any
                    registry = [r for r in registry if r["name"] != current_case_name]

                    registry.append({
                        "name": current_case_name,
                        "directory": os.path.abspath(case_dir),
                        "expected_count": 3, # This might be inaccurate now if we rely on frames, but keeping legacy field
                        "expected_frames": [],
                        "repeat_interval_sec": p / 1000.0,
                        "aed_type": "SimulatedPulse",
                        "effective_frame_rate": 30.0, # Placeholder, or use fps_est if available
                        "duration_ms": d,
                        "period_ms": p,
                        "brightness_pct": b
                    })

                    count += 1

                    if limit and count >= limit:
                        break
                if limit and count >= limit:
                    break
            if limit and count >= limit:
                break

    except KeyboardInterrupt:
        logging.info("Interrupted by user.")
    finally:
        if not dry_run and 'led' in locals():
            led.stop_pulse()
            led.disconnect()
        if not dry_run and 'cam' in locals():
            cam.stop()

        # Save Registry
        with open(registry_path, "w", encoding="utf-8") as f:
            json.dump(registry, f, indent=4)

        logging.info("Generation complete. Saved %d entries to %s/pulse_library.json", len(registry), output_dir)


def align_camera(fps=30, color=True, serial_port="/dev/ttyUSB1"):
    # pylint: disable=too-many-locals
    """Open a preview window for camera alignment with LED flashing."""
    logging.info("Starting camera alignment mode...")

    # Initialize LED controller
    led = LEDController(port=serial_port)
    if led.connect():
        # Set a default pulse pattern: 100ms on, 1000ms period, 100% brightness
        logging.info("Starting LED pulse for alignment...")
        led.set_pulse(100, 1000, 100)
    else:
        logging.warning("Could not connect to LED controller, continuing without LED")

    cam = get_driver()
    cam.start()

    if hasattr(cam, 'cap'):
        logging.info("Setting Camera FPS to %d", fps)
        cam.cap.set(cv2.CAP_PROP_FPS, fps)

        # Enable auto-exposure first
        logging.info("Enabling Auto-Exposure...")
        cam.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 3)  # 3 = Auto

        # Warm up camera (flush buffer)
        logging.info("Warming up camera...")
        for _ in range(30):
            cam.get_frame(color=color)

        # Wait for auto-exposure to settle
        logging.info("Waiting 3 seconds for exposure to settle...")
        time.sleep(3)

        # Read and lock exposure
        current_exposure = cam.cap.get(cv2.CAP_PROP_EXPOSURE)
        logging.info("Camera settled at Exposure: %s", current_exposure)

        if current_exposure > 120:
            new_exposure = int(current_exposure * 0.75)
            logging.warning("Exposure %s too high, reducing to %s", current_exposure, new_exposure)
            current_exposure = new_exposure

        logging.info("Locking Exposure to %s", current_exposure)
        # V4L2 exposure modes: 1=Manual, 3=Auto (0.25 is OpenCV's attempt at 1)
        # Some cameras need the sequence: disable auto first, then set value
        for attempt in range(3):
            cam.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)  # 1=Manual in V4L2
            time.sleep(0.05)
            cam.cap.set(cv2.CAP_PROP_EXPOSURE, current_exposure)
            time.sleep(0.05)
            # Verify
            actual_ae = cam.cap.get(cv2.CAP_PROP_AUTO_EXPOSURE)
            if actual_ae != 3:  # No longer auto
                logging.info("Exposure locked on attempt %d", attempt + 1)
                break

        # Software autofocus (hill-climbing algorithm)
        software_autofocus(cam, color=color)

        # Re-enforce exposure lock after autofocus (autofocus may have changed settings)
        logging.info("Re-locking exposure after autofocus...")
        for attempt in range(3):
            cam.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)  # Manual
            time.sleep(0.05)
            cam.cap.set(cv2.CAP_PROP_EXPOSURE, current_exposure)
            time.sleep(0.05)

        # Lock gain to 0 (prevents auto-gain flickering)
        cam.cap.set(cv2.CAP_PROP_GAIN, 0)

        # Lock white balance (if supported) - disable auto
        cam.cap.set(cv2.CAP_PROP_AUTO_WB, 0)

        # Re-enforce FPS
        cam.cap.set(cv2.CAP_PROP_FPS, fps)

        # Verify all settings are locked
        actual_ae = cam.cap.get(cv2.CAP_PROP_AUTO_EXPOSURE)
        actual_exp = cam.cap.get(cv2.CAP_PROP_EXPOSURE)
        actual_af = cam.cap.get(cv2.CAP_PROP_AUTOFOCUS)
        actual_focus = cam.cap.get(cv2.CAP_PROP_FOCUS)
        actual_gain = cam.cap.get(cv2.CAP_PROP_GAIN)
        actual_wb = cam.cap.get(cv2.CAP_PROP_AUTO_WB)
        logging.info("Camera settings verified:")
        logging.info("  Auto-Exposure: %s (1=Manual)", actual_ae)
        logging.info("  Exposure: %s (target: %s)", actual_exp, current_exposure)
        logging.info("  Auto-Focus: %s (0=Off)", actual_af)
        logging.info("  Focus: %s", actual_focus)
        logging.info("  Gain: %s (0=locked)", actual_gain)
        logging.info("  Auto-WB: %s (0=Off)", actual_wb)

    logging.info("Press 'q' or ESC to close the preview window.")

    # Store the target focus we want to maintain
    target_focus = cam.cap.get(cv2.CAP_PROP_FOCUS) if hasattr(cam, 'cap') else 0
    frame_count = 0

    try:
        while True:
            frame = cam.get_frame(color=color)
            if frame is not None:
                # Monitor and re-lock focus if it drifts
                if hasattr(cam, 'cap') and frame_count % 30 == 0:  # Check every 30 frames
                    current_af = cam.cap.get(cv2.CAP_PROP_AUTOFOCUS)
                    current_focus = cam.cap.get(cv2.CAP_PROP_FOCUS)

                    # Re-lock if autofocus got re-enabled or focus drifted
                    if current_af != 0 or abs(current_focus - target_focus) > 5:
                        logging.warning("Focus drift detected! AF=%s, Focus=%s->%s. Re-locking...",
                                       current_af, target_focus, current_focus)
                        cam.cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
                        cam.cap.set(cv2.CAP_PROP_FOCUS, target_focus)

                # Add alignment crosshair
                h, w = frame.shape[:2]
                cv2.line(frame, (w//2, 0), (w//2, h), (0, 255, 0), 1)
                cv2.line(frame, (0, h//2), (w, h//2), (0, 255, 0), 1)

                # Show live camera status
                if hasattr(cam, 'cap'):
                    live_focus = int(cam.cap.get(cv2.CAP_PROP_FOCUS))
                    live_af = int(cam.cap.get(cv2.CAP_PROP_AUTOFOCUS))
                    live_exp = int(cam.cap.get(cv2.CAP_PROP_EXPOSURE))
                    live_ae = int(cam.cap.get(cv2.CAP_PROP_AUTO_EXPOSURE))
                    live_gain = int(cam.cap.get(cv2.CAP_PROP_GAIN))
                    live_wb = int(cam.cap.get(cv2.CAP_PROP_WB_TEMPERATURE))
                    live_awb = int(cam.cap.get(cv2.CAP_PROP_AUTO_WB))

                    status1 = f"Focus:{live_focus} AF:{live_af} | Exp:{live_exp} AE:{live_ae}"
                    status2 = f"Gain:{live_gain} | WB:{live_wb} AWB:{live_awb}"
                    cv2.putText(frame, status1, (10, h - 25),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
                    cv2.putText(frame, status2, (10, h - 10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

                cv2.putText(frame, "Align camera, then press 'q' to exit",
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                cv2.imshow("Camera Alignment", frame)
                frame_count += 1

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:  # q or ESC
                break
    finally:
        cv2.destroyAllWindows()
        cam.stop()

    logging.info("Alignment mode finished.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate LED pulse data library.")
    parser.add_argument("--dry-run", action="store_true", help="Simulate without hardware.")
    parser.add_argument("--fps", type=int, default=30, help="Camera frame rate.")
    parser.add_argument("--color", action="store_true", help="Capture color frames.")
    parser.add_argument("--limit", type=int, help="Limit number of cases.")
    parser.add_argument("--align", action="store_true", help="Open preview for camera alignment.")

    # New arguments for single case / specific control
    parser.add_argument("--duration", type=float, help="Pulse duration in ms")
    parser.add_argument("--period", type=float, help="Pulse period in ms")
    parser.add_argument("--brightness", type=int, help="Pulse brightness %")
    parser.add_argument("--output-dir", type=str, default=OUTPUT_DIR, help="Output directory")
    parser.add_argument("--case-name", type=str, help="Custom case name")
    parser.add_argument("--serial-port", type=str, default="/dev/ttyUSB1", help="Serial port for LED controller")
    parser.add_argument("--frames", type=int, default=None,
                        help="Number of frames to capture "
                             "(default: auto-compute from AED period)")
    parser.add_argument("--aed-type", type=int, help="AED Type ID")
    parser.add_argument("--aed-name", type=str, help="AED Name")
    parser.add_argument("--run-id", type=str, help="Unique run ID for dataset folder names")

    args = parser.parse_args()

    try:
        if args.align:
            align_camera(fps=args.fps, color=True, serial_port=args.serial_port)
        else:
            generate_data(fps=args.fps,
                          color=args.color,
                          dry_run=args.dry_run,
                          limit=args.limit,
                          duration=args.duration,
                          period=args.period,
                          brightness=args.brightness,
                          output_dir=args.output_dir,
                          case_name=args.case_name,
                          serial_port=args.serial_port,
                          frame_count=args.frames,
                          aed_type_id=args.aed_type,
                          aed_name=args.aed_name,
                          run_id=args.run_id)
    except KeyboardInterrupt:
        logging.info("Interrupted by user (Ctrl+C). Exiting...")
        sys.exit(130)
