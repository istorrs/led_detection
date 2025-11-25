# pylint: disable=no-member
import time
import sys
import platform
import signal
import logging
import argparse
import os
from datetime import datetime
import numpy as np
import cv2

# --- SYSTEM SETUP ---
SYS_OS = platform.system()
if SYS_OS == "Linux":
    os.environ["QT_QPA_PLATFORM"] = "xcb"

def get_timestamp():
    """Get formatted timestamp with milliseconds."""
    now = datetime.now()
    return now.strftime('%H:%M:%S.%f')[:-3]

def setup_logging(debug_mode):
    """Set up logging configuration."""
    level = logging.DEBUG if debug_mode else logging.INFO
    logging.basicConfig(level=level,
                        format='[%(asctime)s.%(msecs)03d] [%(levelname)s] %(message)s',
                        datefmt='%H:%M:%S')

# --- DRIVERS ---
class CameraDriver:
    """Base class for camera drivers."""
    def start(self):
        """Start the camera."""

    def stop(self):
        """Stop the camera."""

    def get_frame(self):
        """Get a frame from the camera."""
        raise NotImplementedError

class RPiCameraDriver(CameraDriver):
    """Driver for Raspberry Pi Camera."""
    def __init__(self, w=640, h=480):
        try:
            # pylint: disable=import-outside-toplevel
            from picamera2 import Picamera2
            self.p = Picamera2()
            self.h, self.w = h, w
            self.p.configure(self.p.create_configuration(main={"format": "YUV420", "size": (w, h)}))
        except ImportError as exc:
            raise RuntimeError("Picamera2 missing.") from exc

    def start(self):
        self.p.start()
        self.p.set_controls({
            "AeEnable": False,
            "AnalogueGain": 8.0,
            "AwbEnable": False,
            "AfMode": 0,
            "LensPosition": 0.0,
            "ExposureTime": 8333,
            "FrameDurationLimits": (16666, 16666)
        })
        time.sleep(1)

    def stop(self):
        self.p.stop()

    def get_frame(self):
        return self.p.capture_array("main")[:self.h, :self.w]

class X86CameraDriver(CameraDriver):
    """Driver for X86/USB Camera."""
    def __init__(self, idx=0, w=640, h=480):
        bk = cv2.CAP_DSHOW if SYS_OS == "Windows" else cv2.CAP_V4L2
        self.cap = cv2.VideoCapture(idx, bk)
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
        self.h, self.w = h, w

    def start(self):
        if SYS_OS == "Linux":
            for _ in range(3):
                self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
                time.sleep(0.1)
            self.cap.set(cv2.CAP_PROP_EXPOSURE, 83)
        else:
            self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0)
            self.cap.set(cv2.CAP_PROP_EXPOSURE, -13)

        self.cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
        self.cap.set(cv2.CAP_PROP_GAIN, 0)
        self.cap.set(cv2.CAP_PROP_FOCUS, 128)  # Mid-range focus as compromise

        # Debug: Print actual settings
        print(f"[Debug] Exposure: {self.cap.get(cv2.CAP_PROP_EXPOSURE)}")
        print(f"[Debug] Gain: {self.cap.get(cv2.CAP_PROP_GAIN)}")
        print(f"[Debug] Focus: {self.cap.get(cv2.CAP_PROP_FOCUS)}")
        time.sleep(0.5)

    def stop(self):
        self.cap.release()

    def get_frame(self):
        r, f = self.cap.read()
        if r:
            return cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
        return np.zeros((self.h, self.w), dtype=np.uint8)

def get_driver():
    """Factory method to get the appropriate camera driver."""
    try:
        # pylint: disable=import-outside-toplevel, unused-import
        import picamera2
        return RPiCameraDriver()
    except ImportError:
        return X86CameraDriver()

# --- PEAK MONITOR ---

class PeakMonitor:  # pylint: disable=too-many-instance-attributes
    """Monitor for LED detection."""
    # pylint: disable=too-many-arguments, too-many-positional-arguments
    def __init__(self, interval, threshold, preview=False,
                 use_contrast=True,      # 1a: Contrast-based detection
                 adaptive_roi=True,      # 1b: Adaptive ROI sizing
                 adaptive_off=True,      # 1c: Adaptive OFF detection
                 log_saturation=True,    # 2a: Saturation logging
                 adaptive_exposure=True, # 2b: Adaptive exposure (experimental)
                 autofocus=False,        # 2c: Autofocus sweep (experimental)
                 min_pulse_duration=100  # 3a: Minimum pulse duration (ms)
                ):
        self.cam = get_driver()
        self.interval = interval
        self.min_pulse_duration = min_pulse_duration
        self.preview = preview
        self.roi = None
        self.thresh_bright = 0.0
        self.min_signal_strength = threshold
        self.detected_peak_strength = 0.0
        self.detected_on_brightness = 0.0  # Actual brightness when ON

        # Feature flags
        self.use_contrast = use_contrast
        self.adaptive_roi = adaptive_roi
        self.adaptive_off = adaptive_off
        self.log_saturation = log_saturation
        self.adaptive_exposure = adaptive_exposure
        self.autofocus = autofocus

        # Adaptive Thresholding
        self.noise_floor_history = []
        self.noise_floor_window = 10
        self.current_noise_floor = 0.0
        self.calibrated_signal_strength = 0.0  # Initialize
        self.led_state = False  # Initialize
        self.led_on_time = 0  # Track when LED turned ON
        self.scan_timeout = 70.0 # Default scan timeout

        # Saturation tracking
        self.saturation_count = 0
        self.total_frames = 0
        self.min_while_on = 0  # Track minimum value during prolonged ON state

        # Continuous adaptive exposure
        self.saturation_window = []  # Rolling window of saturation percentages
        self.last_exposure_adjust = 0  # Time of last exposure adjustment

    def _measure_roi(self, roi):
        """Measure ROI using either contrast or brightness based on feature flag."""
        if self.use_contrast:
            # Contrast-based: max - median (robust against global illumination)
            return float(roi.max()) - float(np.median(roi))
        # Brightness-based: 90th percentile (original behavior)
        return float(np.percentile(roi, 90))

    def aim_camera(self):
        """Aiming phase helper."""
        if not isinstance(self.cam, X86CameraDriver):
            return
        print("\n[System] AIMING PHASE (5s) - Adjust your camera now...")
        end = time.time() + 5.0
        while time.time() < end:
            f = self.cam.get_frame()
            cv2.putText(f, f"AIMING... {end-time.time():.1f}s", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255), 2)
            cv2.imshow("Aiming", f)
            cv2.waitKey(1)
        cv2.destroyWindow("Aiming")

    def detect_signal(self):
        """Detects the LED signal by looking for bright AND changing pixels."""
        # pylint: disable=too-many-locals
        # 1. Establish baseline
        logging.info("Establishing baseline...")
        baseline_frames = []
        for _ in range(10):
            f = self.cam.get_frame()
            if f is not None:
                if len(f.shape) == 3 and f.shape[2] == 3:
                    baseline_frames.append(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY))
                else:
                    baseline_frames.append(f)
            time.sleep(0.05)

        if not baseline_frames:
            return False

        baseline = np.median(baseline_frames, axis=0).astype(np.uint8)

        # 2. Scan for signal
        accum_max_diff = np.zeros_like(baseline)
        accum_max_bright = np.zeros_like(baseline) # Track max brightness

        start_scan = time.time()
        while time.time() - start_scan < self.scan_timeout:
            frame = self.cam.get_frame()
            if frame is None:
                continue

            if len(frame.shape) == 3 and frame.shape[2] == 3:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            else:
                gray = frame

            # Update max difference
            diff = cv2.absdiff(gray, baseline)
            accum_max_diff = np.maximum(accum_max_diff, diff)

            # Update max brightness
            accum_max_bright = np.maximum(accum_max_bright, gray)

            # Check if we have a significant peak yet
            # We use the combined score for this check too
            combined_map = ((accum_max_diff.astype(np.float32) / 255.0) *
                          (accum_max_bright.astype(np.float32) / 255.0) * 255.0).astype(np.uint8)

            _, max_val, _, max_loc = cv2.minMaxLoc(combined_map)

            # Update baseline to adapt to slow lighting changes
            cv2.accumulateWeighted(gray, baseline.astype(float), 0.1)
            baseline = baseline.astype(np.uint8)

            sys.stdout.write(f"\r    Scanning... {time.time()-start_scan:.1f}s | "
                           f"Peak Score: {max_val:.0f} (Req: {self.min_signal_strength:.1f})   ")
            sys.stdout.flush()

            if max_val > self.min_signal_strength:
                # Wait a bit more to ensure we capture the full pulse
                time.sleep(0.5)
                logging.info("SIGNAL DETECTED! Score: %d", max_val)

                # Lock ROI based on the COMBINED map (Bright + Changed)
                self.lock_roi(max_loc, gray.shape, combined_map)

                # Measure the actual ON brightness using the same method as monitoring
                # This ensures threshold calculation matches actual measurements
                y1, y2, x1, x2 = self.roi
                roi = gray[y1:y2, x1:x2]
                self.detected_on_brightness = self._measure_roi(roi)
                self.detected_peak_strength = accum_max_diff[max_loc[1], max_loc[0]]

                logging.info("[Debug] Initial ROI brightness: %.1f", self.detected_on_brightness)
                return True

            time.sleep(0.05)

        print("")
        return False

    def wait_for_signal(self):
        """ Stares at scene using Peak Hold to find the LED """
        logging.info("Scanning for signal... (Max wait: %ss)", self.scan_timeout)
        print("    [Action] Waiting for LED to turn ON...\n")
        return self.detect_signal()



    def lock_roi(self, loc, shape, accum_max_diff=None):
        """Lock Region of Interest."""
        # pylint: disable=too-many-locals
        c_x, c_y = loc
        h, w = shape

        if self.adaptive_roi and accum_max_diff is not None:
            # Adaptive ROI: measure blob size
            threshold_val = self.min_signal_strength * 0.5
            _, binary = cv2.threshold(accum_max_diff, threshold_val, 255, cv2.THRESH_BINARY)
            binary = binary.astype(np.uint8)

            # Find connected components
            _, labels, stats, _ = cv2.connectedComponentsWithStats(binary)

            # Find the component containing the peak location
            peak_label = labels[c_y, c_x]

            if peak_label > 0:  # 0 is background
                bbox = stats[peak_label]  # x, y, width, height, area
                blob_w, blob_h = bbox[2], bbox[3]

                # Use blob size with 1.1x margin (0.55 * dimension for half-width)
                raw_size = int(max(blob_w, blob_h) * 0.55)
                size = max(12, min(48, raw_size))  # Constrain between 24x24 and 96x96

                # Use blob centroid for better centering
                centroid_x = int(bbox[0] + bbox[2] / 2)
                centroid_y = int(bbox[1] + bbox[3] / 2)

                c_x, c_y = centroid_x, centroid_y

                logging.info("Adaptive ROI: Blob=%dx%d, RawHalf=%d, FinalHalf=%d, Center=(%d,%d)",
                           blob_w, blob_h, raw_size, size, c_x, c_y)
            else:
                size = 32  # Fallback to default half-size (64x64)
                logging.warning("Peak not in connected component? Using default ROI size.")
        else:
            # Fixed ROI (original behavior)
            size = 32
            logging.info("Adaptive ROI disabled or no diff, using fixed size (32)")

        x1, x2 = max(0, c_x - size), min(w, c_x + size)
        y1, y2 = max(0, c_y - size), min(h, c_y + size)
        self.roi = (y1, y2, x1, x2)
        logging.info("ROI Locked: %s (size: %dx%d)", self.roi, x2-x1, y2-y1)

        # Save debug image
        debug_img = cv2.cvtColor(accum_max_diff, cv2.COLOR_GRAY2BGR)
        cv2.rectangle(debug_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.circle(debug_img, (c_x, c_y), 5, (0, 0, 255), -1)
        cv2.imwrite("debug_detection.png", debug_img)
        logging.info("Saved debug_detection.png")

    def calibrate_exposure(self):
        """Adaptively calibrate camera exposure to avoid saturation."""
        if not self.adaptive_exposure:
            return

        if not isinstance(self.cam, X86CameraDriver):
            logging.info("Adaptive exposure only supported for X86CameraDriver")
            return

        logging.info("Starting adaptive exposure calibration...")
        y1, y2, x1, x2 = self.roi
        # Target background brightness: 80
        # Target LED brightness (not saturated): 200

        for iteration in range(10):
            # Capture frame and measure
            f = self.cam.get_frame()
            roi = f[y1:y2, x1:x2]
            bg_brightness = np.median(roi)
            max_brightness = np.percentile(roi, 95)

            logging.info("Iteration %d: BG=%.1f, Peak=%.1f",
                        iteration, bg_brightness, max_brightness)

            # Check if we're in good range
            if bg_brightness < 150 and max_brightness < 250:
                logging.info("Exposure calibration complete: BG=%.1f, Peak=%.1f",
                            bg_brightness, max_brightness)
                return

            # Get current exposure
            current_exposure = self.cam.cap.get(cv2.CAP_PROP_EXPOSURE)

            # Adjust exposure based on brightness
            if bg_brightness > 150 or max_brightness > 250:
                # Too bright, reduce exposure
                new_exposure = current_exposure * 0.7
                logging.info("Too bright, reducing exposure: %.1f -> %.1f",
                            current_exposure, new_exposure)
            elif max_brightness < 150:
                # Too dim, increase exposure
                new_exposure = current_exposure * 1.3
                logging.info("Too dim, increasing exposure: %.1f -> %.1f",
                            current_exposure, new_exposure)
            else:
                # Good enough
                break

            # Apply new exposure
            self.cam.cap.set(cv2.CAP_PROP_EXPOSURE, new_exposure)
            time.sleep(0.5)  # Allow sensor to settle

        logging.info("Adaptive exposure complete after %d iterations", iteration + 1)

    def autofocus_sweep(self):
        """Automatically find optimal focus using bidirectional hill-climbing algorithm."""
        # pylint: disable=too-many-locals,too-many-branches,too-many-statements
        if not self.autofocus:
            return

        if not isinstance(self.cam, X86CameraDriver):
            logging.info("Autofocus only supported for X86CameraDriver")
            return

        logging.info("Starting autofocus sweep (hill-climbing algorithm)...")

        # Helper function to measure sharpness with proper settling
        def measure_sharpness(focus_pos, settle_time=0.5):
            """Set focus and measure sharpness after settling."""
            self.cam.cap.set(cv2.CAP_PROP_FOCUS, focus_pos)
            time.sleep(settle_time)  # Critical: wait for motor to settle

            frame = self.cam.get_frame()
            if frame is None:
                return 0

            # Measure sharpness in center region
            h, w = frame.shape[:2]
            center_h, center_w = h // 2, w // 2
            roi_size = min(h, w) // 3
            y1, y2 = center_h - roi_size // 2, center_h + roi_size // 2
            x1, x2 = center_w - roi_size // 2, center_w + roi_size // 2

            center_roi = frame[y1:y2, x1:x2]
            sharpness = cv2.Laplacian(center_roi, cv2.CV_64F).var()

            # Show preview if enabled
            if self.preview or self.autofocus:
                preview_frame = frame.copy()
                cv2.rectangle(preview_frame, (x1, y1), (x2, y2), (255, 255, 255), 2)
                cv2.putText(preview_frame, f"Focus: {focus_pos} | Sharp: {sharpness:.1f}",
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.imshow("Autofocus Sweep", preview_frame)
                cv2.waitKey(1)

            return sharpness, frame

        # Capture initial state
        initial_focus = int(self.cam.cap.get(cv2.CAP_PROP_FOCUS))
        initial_sharpness, initial_frame = measure_sharpness(initial_focus)

        if initial_frame is not None:
            cv2.imwrite("autofocus_initial.png", initial_frame)
            logging.info("Saved autofocus_initial.png")

        logging.info("Initial focus: %d, sharpness: %.2f", initial_focus, initial_sharpness)

        # Hill-climbing parameters
        coarse_step = 15  # Larger steps for coarse search
        fine_step = 3     # Smaller steps for fine tuning

        best_focus = initial_focus
        best_sharpness = initial_sharpness

        # Phase 1: Coarse search in both directions
        logging.info("Phase 1: Coarse bidirectional search (step=%d)...", coarse_step)

        for direction in [+1, -1]:  # Try both directions
            current_focus = initial_focus
            current_sharpness = initial_sharpness
            improving = True

            direction_name = "forward" if direction > 0 else "backward"
            logging.info("  Searching %s from %d...", direction_name, current_focus)

            while improving:
                # Try next position
                next_focus = current_focus + (direction * coarse_step)

                # Bounds check
                if next_focus < 0 or next_focus > 255:
                    logging.debug("    Hit boundary at %d", next_focus)
                    break

                next_sharpness, _ = measure_sharpness(next_focus)
                logging.debug("    Focus %d: sharpness %.2f", next_focus, next_sharpness)

                # Check if we're improving
                if next_sharpness > current_sharpness * 1.02:  # At least 2% improvement
                    current_focus = next_focus
                    current_sharpness = next_sharpness

                    # Update best if this is better
                    if current_sharpness > best_sharpness:
                        best_focus = current_focus
                        best_sharpness = current_sharpness
                        logging.info("    New best: focus=%d, sharpness=%.2f",
                                   best_focus, best_sharpness)
                else:
                    # Getting worse, stop this direction
                    logging.debug("    Stopped improving at %d", next_focus)
                    improving = False

        # Phase 2: Fine tuning around best position
        logging.info("Phase 2: Fine tuning around %d (step=%d)...", best_focus, fine_step)

        # Search ±2 steps around best position
        fine_start = max(0, best_focus - (fine_step * 2))
        fine_end = min(255, best_focus + (fine_step * 2))

        for focus_pos in range(fine_start, fine_end + 1, fine_step):
            if focus_pos == best_focus:
                continue  # Already measured

            sharpness, _ = measure_sharpness(focus_pos, settle_time=0.4)
            logging.debug("  Fine focus %d: sharpness %.2f", focus_pos, sharpness)

            if sharpness > best_sharpness:
                best_focus = focus_pos
                best_sharpness = sharpness
                logging.info("  Fine tuning improved: focus=%d, sharpness=%.2f",
                           best_focus, best_sharpness)

        # Decide whether to apply new focus
        improvement_pct = ((best_sharpness - initial_sharpness) / initial_sharpness * 100) \
                         if initial_sharpness > 0 else 0

        if best_sharpness > initial_sharpness * 1.05:  # At least 5% improvement
            # Apply best focus
            final_sharpness, final_frame = measure_sharpness(best_focus)

            logging.info("✓ Autofocus improved: %d→%d, Sharpness: %.2f→%.2f (+%.1f%%)",
                        initial_focus, best_focus, initial_sharpness, final_sharpness,
                        improvement_pct)
        else:
            # Revert to initial - it was better
            final_sharpness, final_frame = measure_sharpness(initial_focus)

            logging.warning("✗ Autofocus did not improve (%.1f%% change). "
                          "Keeping initial position %d",
                          improvement_pct, initial_focus)

        # Save final image
        if final_frame is not None:
            cv2.imwrite("autofocus_final.png", final_frame)
            logging.info("Saved autofocus_final.png")

        # Close preview window
        if self.preview or self.autofocus:
            cv2.destroyWindow("Autofocus Sweep")

    def wait_for_led_off(self):
        """ Monitors ROI brightness until it drops significanty """
        # pylint: disable=too-many-locals
        logging.info("Waiting for LED to turn OFF...")
        y1, y2, x1, x2 = self.roi

        # Measure "High" State
        f = self.cam.get_frame()
        roi = f[y1:y2, x1:x2]
        high_val = self._measure_roi(roi)

        if self.adaptive_off:
            # Adaptive: measure initial variance to set noise-based threshold
            samples = []
            for _ in range(10):
                f = self.cam.get_frame()
                roi = f[y1:y2, x1:x2]
                samples.append(self._measure_roi(roi))
                time.sleep(0.05)

            initial_mean = np.mean(samples)
            noise_std = np.std(samples)

            # Threshold: mean - 3*sigma (drop below noise floor)
            # If std is very high (e.g. flashing), 3*sigma might be huge, making threshold too low.
            # We clamp the std effect to avoid unreachable thresholds.
            effective_std = min(noise_std, initial_mean * 0.1)
            # Calculate threshold (Mean - 2*StdDev) - Relaxed from 3*StdDev
            # Also ensure it's at least 10% below the mean
            drop_threshold = min(initial_mean - 2.0 * effective_std, initial_mean * 0.9)

            logging.info("Adaptive OFF: Initial=%.1f, Std=%.1f, EffStd=%.1f, Threshold=%.1f",
                       initial_mean, noise_std, effective_std, drop_threshold)
        else:
            # Fixed threshold (original behavior)
            drop_threshold = high_val * 0.6

        start = time.time()
        while time.time() - start < 15.0: # 15s Timeout
            f = self.cam.get_frame()
            roi = f[y1:y2, x1:x2]
            curr_val = self._measure_roi(roi)

            # Condition 1: Drop below threshold
            drop_cond = curr_val < drop_threshold

            # Condition 2: Absolute dark (safe noise floor)
            # For contrast mode, this is different than brightness mode
            abs_threshold = 50 if self.use_contrast else 100
            abs_cond = curr_val < abs_threshold

            if drop_cond or abs_cond:
                # Found an OFF state - accept it immediately
                # This works even if LED is flashing, we just catch it during OFF phase
                logging.info("LED OFF Confirmed (Curr: %.1f < Threshold: %.1f)",
                           curr_val, drop_threshold)
                time.sleep(0.1) # Brief settling
                return True

            # Only log periodically to avoid spam
            if int(time.time() - start) % 2 == 0:  # Every 2 seconds
                logging.info("Waiting for drop... Curr: %.0f (Threshold: %.0f)",
                           curr_val, drop_threshold)
            time.sleep(0.1)

        logging.warning("Timeout waiting for LED off! Noise floor might be inaccurate.")
        return False

    def run(self):
        """Run the monitor loop."""
        # pylint: disable=too-many-locals, too-many-statements, too-many-branches
        self.cam.start()
        if self.preview:
            self.aim_camera()

        # 0. Autofocus (if enabled) - must happen before ROI detection
        self.autofocus_sweep()

        # 1. Search Loop
        print("[System] Waiting for device activity...")
        while True:
            if self.wait_for_signal():
                break
            print("\n[Retry] No signal seen. Clearing buffer and rescanning...\n")

        # 2. Wait for "OFF" State
        self.wait_for_led_off()

        # 2b. Adaptive Exposure (if enabled)
        self.calibrate_exposure()

        # 3. Calibrate Noise Floor (OFF State)
        logging.info("Calibrating OFF-state noise floor...")
        y1, y2, x1, x2 = self.roi
        metric_samples = []
        bg_brightness_samples = []  # Track background for saturation

        for _ in range(30):
            f = self.cam.get_frame()
            roi = f[y1:y2, x1:x2]
            metric_samples.append(self._measure_roi(roi))
            bg_brightness_samples.append(np.median(roi))  # Background brightness

        avg_noise_level = np.mean(metric_samples)
        # Estimate ON level and Signal Strength
        if hasattr(self, 'detected_on_brightness'):
            estimated_on_level = self.detected_on_brightness
        else:
            estimated_on_level = avg_noise_level + self.detected_peak_strength

        self.calibrated_signal_strength = max(10.0, estimated_on_level - avg_noise_level)
        self.current_noise_floor = avg_noise_level

        # Set threshold: Noise + 50% of Signal
        self.thresh_bright = self.current_noise_floor + (self.calibrated_signal_strength * 0.5)

        metric_name = "Contrast" if self.use_contrast else "Brightness"
        logging.info("Noise (%s): %.1f | Est. ON: %.1f | Signal: %.1f",
                    metric_name, avg_noise_level, estimated_on_level,
                    self.calibrated_signal_strength)
        logging.info("Set Initial Detection Threshold: %.1f", self.thresh_bright)

        # Saturation check
        if self.log_saturation and np.mean(bg_brightness_samples) > 240: # Use mean of samples
            logging.warning("High ambient brightness detected! Sensor may be saturating.")

        # --- Monitoring Loop ---
        logging.info("--- MONITORING STARTED ---")

        last_pulse = time.time()
        self.led_state = False  # Reset LED state
        last_off_time = last_pulse  # Track last time LED was truly OFF

        # Stats aggregation
        report_period = 1.0
        last_report = time.time()

        int_max = 0
        int_min = 255 if not self.use_contrast else 0 # Initialize int_min correctly for contrast
        t_max = last_pulse
        t_min = last_pulse

        while True:  # pylint: disable=too-many-nested-blocks
            # Capture frame
            frame = self.cam.get_frame() # Changed from read_frame() to get_frame()
            if frame is None:
                break

            # Process ROI
            if self.roi:
                y1, y2, x1, x2 = self.roi
                roi = frame[y1:y2, x1:x2]
            else:
                roi = frame

            # Measure signal
            val = self._measure_roi(roi) # Changed from get_signal_strength() to _measure_roi()
            now = time.time()

            # Saturation tracking and continuous adaptive exposure
            if self.log_saturation:
                self.total_frames += 1
                sat_pct = np.percentile(roi, 95) >= 250
                if sat_pct:
                    self.saturation_count += 1

                # Continuous adaptive exposure: track saturation over time
                if self.adaptive_exposure and isinstance(self.cam, X86CameraDriver):
                    # Calculate rolling saturation percentage
                    self.saturation_window.append(1 if sat_pct else 0)
                    if len(self.saturation_window) > 30:  # 30-frame window (~1 second)
                        self.saturation_window.pop(0)

                    # Bidirectional adaptive exposure
                    time_since_adjust = now - self.last_exposure_adjust
                    if len(self.saturation_window) >= 10 and time_since_adjust > 2.0:
                        recent_sat_pct = sum(self.saturation_window) / len(self.saturation_window)
                        current_exposure = self.cam.cap.get(cv2.CAP_PROP_EXPOSURE)

                        # Check if we need to reduce exposure (too saturated)
                        if recent_sat_pct > 0.5:
                            # More than 50% of recent frames saturated - reduce exposure
                            new_exposure = current_exposure * 0.7
                            self.cam.cap.set(cv2.CAP_PROP_EXPOSURE, new_exposure)
                            self.last_exposure_adjust = now
                            logging.info("Continuous Exposure: Saturation %.0f%%, "
                                       "reducing %.1f→%.1f",
                                       recent_sat_pct * 100, current_exposure, new_exposure)
                            self.saturation_window = []

                        # Check if we need to increase exposure (signal too weak)
                        elif recent_sat_pct < 0.05:  # Less than 5% saturation
                            # Check if signal strength is low
                            # Only increase if we're detecting pulses but they're weak
                            if val > self.thresh_bright:  # Currently in a pulse
                                signal_strength = val - self.current_noise_floor
                                # If signal is weak relative to calibrated strength
                                if signal_strength < self.calibrated_signal_strength * 0.5:
                                    # Cap at initial exposure
                                    new_exposure = min(current_exposure * 1.3, 83.0)
                                    self.cam.cap.set(cv2.CAP_PROP_EXPOSURE, new_exposure)
                                    self.last_exposure_adjust = now
                                    logging.info("Continuous Exposure: Signal weak (%.0f < %.0f), "
                                               "increasing %.1f→%.1f",
                                               signal_strength,
                                               self.calibrated_signal_strength * 0.5,
                                               current_exposure, new_exposure)
                                    self.saturation_window = []

            # Update Interval Stats
            if val > int_max:
                int_max = val
                t_max = now
            if val < int_min:
                int_min = val
                t_min = now

            # Check if LED is active (must be calculated before adaptive threshold update)
            is_active = val > self.thresh_bright
            gap = now - last_pulse

            # Adaptive Threshold Update (Bidirectional)
            # Track noise floor changes in both directions (brightening and dimming)
            # Freeze updates for 3s after exposure changes to avoid contamination
            time_since_exposure_adjust = now - self.last_exposure_adjust

            if not is_active and time_since_exposure_adjust > 3.0:
                # LED is OFF AND exposure has stabilized (3s since last adjustment)
                last_off_time = now  # Update last OFF time
                self.noise_floor_history.append(val)
                if len(self.noise_floor_history) > self.noise_floor_window:
                    self.noise_floor_history.pop(0)

                # Recalculate noise floor and threshold
                self.current_noise_floor = np.mean(self.noise_floor_history)
                self.thresh_bright = (self.current_noise_floor +
                                     (self.calibrated_signal_strength * 0.5))
            else:
                # LED is ON - check if we've been stuck in ON state for too long
                # Use time since last OFF state, not time since last pulse
                time_since_off = now - last_off_time

                # Track minimum value while in ON state (doesn't reset with report period)
                if time_since_off < 0.1:
                    self.min_while_on = val
                else:
                    self.min_while_on = min(self.min_while_on, val)

                if (time_since_off > 5.0 and
                    self.min_while_on > (self.current_noise_floor +
                              self.calibrated_signal_strength * 0.3)):
                    # The minimum value we're seeing is significantly higher than expected
                    # This suggests ambient light has increased
                    self.noise_floor_history.append(self.min_while_on)
                    if len(self.noise_floor_history) > self.noise_floor_window:
                        self.noise_floor_history.pop(0)

                    # Recalculate noise floor and threshold
                    self.current_noise_floor = np.mean(self.noise_floor_history)
                    self.thresh_bright = (self.current_noise_floor +
                                         (self.calibrated_signal_strength * 0.5))
                    # Reset min_while_on after updating
                    self.min_while_on = val

            if is_active:
                if not self.led_state:
                    # LED just turned ON
                    self.led_state = True
                    self.led_on_time = now
            else:
                if self.led_state:
                    # LED just turned OFF
                    self.led_state = False
                    duration = (now - self.led_on_time) * 1000 # ms

                    # Filter out short noise spikes
                    if duration >= self.min_pulse_duration:
                        last_pulse = now
                        if gap > 0.5: # Reduced from 2.0s to catch faster pulses
                            timestamp = get_timestamp()
                            sys.stdout.write(f"\033[96m[{timestamp}] [PULSE DETECTED] "
                                           f"Gap: {gap:.1f}s | Duration: {duration:.0f}ms\033[0m\n")
                    else:
                        logging.debug("Ignored short pulse: %.0fms", duration)

            # Report every second
            if now - last_report > report_period:
                limit = self.interval * 1.2
                if is_active:
                    status = "[LED ON]"
                    color = "\033[92m" # Green
                else:
                    status = f"[ALARM] ({gap:.1f}s)" if gap > limit else f"[WAITING] {gap:.1f}s"
                    color = "\033[91m" if gap > limit else "\033[92m"

                dt = abs(t_max - t_min)
                metric_label = "Contrast" if self.use_contrast else "Bright"
                timestamp = get_timestamp()
                status_line = (f"{color}[{timestamp}] {status} | {metric_label}: {val:.0f} | "
                              f"Thr: {self.thresh_bright:.0f} "
                              f"(Floor: {self.current_noise_floor:.0f}) | "
                              f"Min/Max: {int_min:.0f}/{int_max:.0f} | dT: {dt:.3f}s")

                # Add saturation indicator
                if self.log_saturation and self.total_frames > 0:
                    sat_pct = (self.saturation_count / self.total_frames) * 100
                    if sat_pct > 10:  # More than 10% saturation
                        status_line += f" [SAT: {sat_pct:.1f}%]"

                print(f"{status_line}\033[0m")

                # Reset stats
                last_report = now
                int_max = 0
                int_min = 255 if not self.use_contrast else 0
                if self.log_saturation:
                    self.saturation_count = 0
                    self.total_frames = 0

            if isinstance(self.cam, X86CameraDriver):
                debug = frame.copy()
                cv2.rectangle(debug, (x1, y1), (x2, y2), (255, 255, 255), 1)
                cv2.putText(debug, f"{val:.0f}", (x1, y1-5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            (0, 255, 0) if is_active else (0, 0, 255), 2)
                cv2.imshow("Monitor", debug)
                if cv2.waitKey(1) == ord('q'):
                    break

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="LED Detection and Monitoring System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Feature Flags (all enabled by default except adaptive-exposure and autofocus):
  --use-contrast        Use contrast (max-median) instead of brightness
  --adaptive-roi        Automatically size ROI based on LED blob size
  --adaptive-off        Use variance-based OFF detection
  --log-saturation      Track and log saturation events
  --adaptive-exposure   Automatically adjust camera exposure (experimental)
  --autofocus           Automatically find optimal focus (experimental, X86 only)

To disable a feature, use --no-<feature>, e.g., --no-use-contrast
        """)

    parser.add_argument("-i", "--interval", type=float, default=60.0,
                       help="Expected max time (s) between LED pulses")
    parser.add_argument("-t", "--threshold", type=float, default=50.0,
                       help="Minimum signal strength for LED detection")
    parser.add_argument("-d", "--debug", action="store_true",
                       help="Enable debug logging")
    parser.add_argument("-p", "--preview", action="store_true",
                       help="Enable aiming phase for camera positioning")

    # Feature flags (enabled by default)
    parser.add_argument("--use-contrast", dest="use_contrast", action="store_true", default=True,
                       help="Use contrast-based detection (default)")
    parser.add_argument("--no-use-contrast", dest="use_contrast", action="store_false",
                       help="Use brightness-based detection (original)")

    parser.add_argument("--adaptive-roi", dest="adaptive_roi", action="store_true", default=True,
                       help="Auto-size ROI based on LED blob (default)")
    parser.add_argument("--no-adaptive-roi", dest="adaptive_roi", action="store_false",
                       help="Use fixed 64x64 ROI")

    parser.add_argument("--adaptive-off", dest="adaptive_off", action="store_true", default=True,
                       help="Use variance-based OFF detection (default)")
    parser.add_argument("--no-adaptive-off", dest="adaptive_off", action="store_false",
                       help="Use fixed 60% drop threshold")

    parser.add_argument("--log-saturation", dest="log_saturation",
                       action="store_true", default=True,
                       help="Track saturation events (default)")
    parser.add_argument("--no-log-saturation", dest="log_saturation", action="store_false",
                       help="Disable saturation logging")

    parser.add_argument("--adaptive-exposure", dest="adaptive_exposure",
                       action="store_true", default=True,
                       help="Auto-adjust exposure (experimental, X86 only)")

    parser.add_argument("--autofocus", dest="autofocus",
                       action="store_true", default=False,
                       help="Auto-focus using Laplacian variance sweep (experimental, X86 only)")

    parser.add_argument("--min-pulse-duration", dest="min_pulse_duration",
                       type=int, default=25,
                       help="Minimum pulse duration in ms to count as valid (default: 25)")

    args = parser.parse_args()

    setup_logging(args.debug)

    # Log enabled features
    if args.debug:
        logging.debug("Feature flags:")
        logging.debug("  use_contrast: %s", args.use_contrast)
        logging.debug("  adaptive_roi: %s", args.adaptive_roi)
        logging.debug("  adaptive_off: %s", args.adaptive_off)
        logging.debug("  log_saturation: %s", args.log_saturation)
        logging.debug("  adaptive_exposure: %s", args.adaptive_exposure)
        logging.debug("  autofocus: %s", args.autofocus)

    monitor = PeakMonitor(
        args.interval,
        args.threshold,
        args.preview,
        use_contrast=args.use_contrast,
        adaptive_roi=args.adaptive_roi,
        adaptive_off=args.adaptive_off,
        log_saturation=args.log_saturation,
        adaptive_exposure=args.adaptive_exposure,
        autofocus=args.autofocus,
        min_pulse_duration=args.min_pulse_duration
    )

    def cleanup(_s, _f):
        """Signal handler for cleanup."""
        monitor.cam.stop()
        # pylint: disable=no-member
        cv2.destroyAllWindows()
        sys.exit(0)

    signal.signal(signal.SIGINT, cleanup)
    monitor.run()
