# pylint: disable=no-member
import time
import sys
import platform
import logging
import argparse
import os
import shutil
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
                        format='[%(asctime)s.%(msecs)03d] [%(levelname)s] %(filename)s:%(lineno)d %(message)s',
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

        # Get current focus and set it explicitly to ensure manual mode is active
        # Some cameras need an actual focus value set to leave autofocus mode
        current_focus = self.cap.get(cv2.CAP_PROP_FOCUS)
        self.cap.set(cv2.CAP_PROP_FOCUS, current_focus)
        logging.info("Set manual focus mode at position: %s", current_focus)

        # Optimize latency
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        # Log info
        self.log_camera_info()

        # Debug: Print actual settings
        logging.info("[Debug] Exposure: %s", self.cap.get(cv2.CAP_PROP_EXPOSURE))
        logging.info("[Debug] Gain: %s", self.cap.get(cv2.CAP_PROP_GAIN))
        logging.info("[Debug] Focus: %s", self.cap.get(cv2.CAP_PROP_FOCUS))
        time.sleep(0.5)

    def log_camera_info(self):
        """Log available camera information."""
        logging.info("--- Camera Information ---")
        try:
            backend = self.cap.getBackendName()
            logging.info("Backend: %s", backend)

            w = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
            h = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
            logging.info("Resolution: %sx%s", w, h)

            fps = self.cap.get(cv2.CAP_PROP_FPS)
            logging.info("FPS: %s", fps)

            fourcc = int(self.cap.get(cv2.CAP_PROP_FOURCC))
            # Handle case where fourcc might be 0 or invalid
            if fourcc != 0:
                fourcc_str = "".join([chr((fourcc >> 8 * i) & 0xFF) for i in range(4)])
                logging.info("FourCC: %s (%s)", fourcc, fourcc_str)
            else:
                logging.info("FourCC: %s", fourcc)

            buf_size = self.cap.get(cv2.CAP_PROP_BUFFERSIZE)
            logging.info("Buffer Size: %s", buf_size)

            # Try to get focus range (not all cameras support this)
            try:
                focus_val = self.cap.get(cv2.CAP_PROP_FOCUS)
                if hasattr(cv2, 'CAP_PROP_FOCUS'):
                    logging.info("Current Focus: %s", focus_val)
            except Exception:  # pylint: disable=broad-exception-caught
                pass

        except (AttributeError, cv2.error) as e:
            logging.error("Error reading camera info: %s", e)
        logging.info("--------------------------")

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
                 autofocus=True,         # 2c: Autofocus sweep
                 min_pulse_duration=100, # 3a: Minimum pulse duration (ms)
                 window_name="Detection Monitor" # 4a: Custom window name
                ):
        self.cam = get_driver()
        self.interval = interval
        self.min_pulse_duration = min_pulse_duration
        self.preview = preview
        self.window_name = window_name
        self.roi = None
        self.thresh_bright = 0.0
        self.min_signal_strength = threshold
        self.detected_peak_strength = 0.0
        self.detected_on_brightness = 0.0  # Actual brightness when ON
        self.calibrated_off_level = None   # Calibrated OFF level (from exposure calibration)

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
        self.noise_floor_skip_count = 0  # Track how many updates we've skipped

        self.frame_interval = 0.033

        # Initialized later
        self.thresh_high = 0.0
        self.thresh_low = 0.0
        self.verification_samples = []
        self.pulse_samples = []

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


    def _establish_baseline(self):
        """Capture baseline frames for signal detection."""
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
            logging.error("Failed to capture baseline frames.")
            return None

        return np.median(baseline_frames, axis=0).astype(np.uint8)

    def wait_for_signal(self, timeout=None, dwell_time=2.0):
        """
        Wait for a signal to be detected.

        Args:
            timeout: Maximum time to wait in seconds.
            dwell_time: Time the signal must be sustained to be considered valid (seconds).
        """
        # pylint: disable=too-many-locals, too-many-nested-blocks
        if not self.cam.cap.isOpened():
            return False

        # 1. Establish baseline
        baseline = self._establish_baseline()
        if baseline is None:
            return False

        # 2. Scan for signal
        accum_max_diff = np.zeros_like(baseline)
        accum_max_bright = np.zeros_like(baseline) # Track max brightness

        start_scan = time.time()
        scan_duration = timeout if timeout is not None else self.scan_timeout

        detection_start_time = None
        # DWELL_TIME is now a parameter

        if self.preview:
            logging.info("Preview enabled. Window name: %s", self.window_name)
            cv2.namedWindow(self.window_name, cv2.WINDOW_AUTOSIZE)

        while time.time() - start_scan < scan_duration:
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
            baseline_float = baseline.astype(float)
            cv2.accumulateWeighted(gray, baseline_float, 0.1)
            baseline = baseline_float.astype(np.uint8)

            # Debug
            # logging.info("Frame %d: Max Brightness %d, Max Diff %d, Baseline %d",
            #              self.cam.frame_count if hasattr(self.cam, 'frame_count') else -1,
            #              np.max(gray), np.max(diff), np.mean(baseline))

            sys.stdout.write(f"\r    Scanning... {time.time()-start_scan:.1f}s | "
                           f"Peak Score: {max_val:.0f} (Req: {self.min_signal_strength:.1f})   ")
            sys.stdout.flush()

            # Visualization
            if self.preview:
                vis_frame = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
                # Draw status
                status_color = (0, 255, 0) if max_val > self.min_signal_strength else (0, 0, 255)
                cv2.putText(vis_frame, f"Score: {max_val:.1f}/{self.min_signal_strength}",
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)

                # Draw peak location
                if max_val > 0:
                    cv2.circle(vis_frame, max_loc, 10, status_color, 2)

                # Draw ROI if we were to lock now
                if max_val > self.min_signal_strength:
                    cv2.circle(vis_frame, max_loc, 20, (0, 255, 255), 2)
                    if detection_start_time:
                        remaining = dwell_time - (time.time() - detection_start_time)
                        cv2.putText(vis_frame, f"Verifying... {remaining:.1f}s", (10, 60),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

                cv2.imshow(self.window_name, vis_frame)
                cv2.waitKey(1)

            if max_val > self.min_signal_strength:
                # Verify blob size to reject single-pixel noise
                # Threshold relative to peak to isolate the potential blob
                threshold_val = max(self.min_signal_strength * 0.5, max_val * 0.8)
                _, binary = cv2.threshold(accum_max_diff, threshold_val, 255, cv2.THRESH_BINARY)
                binary = binary.astype(np.uint8)

                # Check connected component at max_loc using floodFill
                # This is efficient and gives us the bounding box of the blob at max_loc
                mask = np.zeros((accum_max_diff.shape[0]+2, accum_max_diff.shape[1]+2), np.uint8)
                _, _, _, rect = cv2.floodFill(binary, mask, max_loc, 255)
                blob_w, blob_h = rect[2], rect[3]

                # Require at least 3x3 blob to consider it a valid signal source
                # UNLESS the signal is very strong (e.g. focused LED at distance)
                is_small = blob_w < 3 and blob_h < 3
                is_strong = max_val > self.min_signal_strength * 1.5

                if is_small and not is_strong:
                    # Too small and not strong enough. Likely noise.
                    logging.debug("Ignored small blob: %dx%d at %s (Score: %d)", blob_w, blob_h, max_loc, max_val)
                    detection_start_time = None
                else:
                    # Verify that the CURRENT signal is also strong, not just the accumulated history
                    # accum_max_diff remembers past peaks, so a transient noise spike would stick.
                    # We need to ensure the signal is SUSTAINED.
                    current_val = diff[max_loc[1], max_loc[0]]

                    # Use a relaxed threshold for sustaining (50% of trigger threshold)
                    # This handles slight fluctuations but rejects noise that has disappeared
                    if current_val > self.min_signal_strength * 0.5:
                        if detection_start_time is None:
                            detection_start_time = time.time()
                            self.verification_samples = [] # Initialize samples
                            logging.info("Potential signal (%.1f). Current: %.1f. Verifying...", max_val, current_val)

                        # Collect samples for stats
                        self.verification_samples.append(current_val)

                        # Debug logging
                        # logging.info("Time: %.2f, Start: %s, Diff: %.2f", time.time(), detection_start_time, time.time() - (detection_start_time or 0))

                        if time.time() - detection_start_time > dwell_time:
                            # Calculate stats
                            if self.verification_samples:
                                mean_val = np.mean(self.verification_samples)
                                peak_val = np.max(self.verification_samples)
                                std_val = np.std(self.verification_samples)
                                logging.info("SIGNAL DETECTED! Score: %d | Stats: Mean=%.1f, Peak=%.1f, Std=%.1f",
                                           max_val, mean_val, peak_val, std_val)
                            else:
                                logging.info("SIGNAL DETECTED! Score: %d", max_val)

                            # Lock ROI based on the COMBINED map (Bright + Changed)
                            self.lock_roi(max_loc, gray.shape, combined_map)

                            # Measure the actual ON brightness using the same method as monitoring
                            # This ensures threshold calculation matches actual measurements
                            y1, y2, x1, x2 = self.roi
                            roi = gray[y1:y2, x1:x2]
                            self.detected_on_brightness = self._measure_roi(roi)
                            self.detected_peak_strength = accum_max_diff[max_loc[1], max_loc[0]]

                            logging.info("[Debug] Peak signal strength: %.1f", self.detected_peak_strength)

                            # Visualize ROI if preview enabled
                            if self.preview:
                                # Re-draw frame with ROI
                                vis_frame = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
                                cv2.rectangle(vis_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                                cv2.putText(vis_frame, "ROI LOCKED", (x1, y1-10),
                                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                                cv2.imshow(self.window_name, vis_frame)
                                cv2.waitKey(1000) # Pause to let user see the ROI
                                # Close this window - main monitoring will open fresh window
                                cv2.destroyWindow(self.window_name)
                                cv2.waitKey(1)  # Process close event

                            return True
                    else:
                        # Signal dropped below sustaining threshold
                        detection_start_time = None
            else:
                # If signal drops below threshold, should we reset?
                # No, accum_max_diff retains the peak.
                # But if it was just noise and we want to wait for a REAL pulse...
                # accum_max_diff never drops. So max_val never drops.
                detection_start_time = None # Reset if signal drops below threshold

            time.sleep(0.05)

        print("")
        return False




    def lock_roi(self, loc, shape, accum_max_diff=None):
        """Lock Region of Interest."""
        # pylint: disable=too-many-locals
        c_x, c_y = loc
        h, w = shape

        if self.adaptive_roi and accum_max_diff is not None:
            # Adaptive ROI: measure blob size
            # Use threshold relative to peak to avoid large blobs from global changes
            peak_val = accum_max_diff[loc[1], loc[0]]
            threshold_val = max(self.min_signal_strength * 0.5, peak_val * 0.8)
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
                size = max(12, min(32, raw_size))  # Constrain between 24x24 and 64x64

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
            return None

        if not isinstance(self.cam, X86CameraDriver):
            logging.info("Adaptive exposure only supported for X86CameraDriver")
            return None

        logging.info("Starting adaptive exposure calibration...")

        y1, y2, x1, x2 = self.roi
        # Target background brightness: 80
        # Target LED brightness (not saturated): 200

        # Initialize current exposure once
        current_exposure = self.cam.cap.get(cv2.CAP_PROP_EXPOSURE)

        for iteration in range(10):
            # Capture frame and measure
            f = self.cam.get_frame()
            roi = f[y1:y2, x1:x2]
            bg_brightness = np.median(roi)
            max_brightness = np.percentile(roi, 95)

            logging.info("Iteration %d: BG=%.1f, Peak=%.1f",
                        iteration, bg_brightness, max_brightness)

            # Check if we're in good range
            # Target lower background to ensure good SNR and avoid noise floor issues
            if bg_brightness < 50 and max_brightness < 200:
                logging.info("Exposure calibration complete: BG=%.1f, Peak=%.1f",
                            bg_brightness, max_brightness)
                self.calibrated_off_level = bg_brightness
                # Get current exposure to return it
                return (bg_brightness, current_exposure)

            # Adjust exposure based on brightness
            if bg_brightness > 50 or max_brightness > 200:
                # Too bright, reduce exposure
                new_exposure = current_exposure * 0.7
                logging.info("Too bright, reducing exposure: %.1f -> %.1f",
                            current_exposure, new_exposure)
            elif max_brightness < 50:
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
            current_exposure = new_exposure # Update current_exposure for next iteration/return

        logging.info("Adaptive exposure complete after %d iterations", iteration + 1)
        return (bg_brightness, current_exposure)

    def lock_current_exposure(self):
        """Locks the current exposure settings."""
        logging.info("Locking current exposure...")
        try:
            # Get current values
            curr_exp = self.cam.cap.get(cv2.CAP_PROP_EXPOSURE)

            # Switch to Manual (1)
            self.cam.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)

            # Re-apply the read exposure to ensure it sticks
            self.cam.cap.set(cv2.CAP_PROP_EXPOSURE, curr_exp)

            logging.info("Exposure locked at %.1f", curr_exp)

            # Measure background now that exposure is locked
            time.sleep(0.5)
            f = self.cam.get_frame()

            if self.roi:
                y1, y2, x1, x2 = self.roi
                roi = f[y1:y2, x1:x2]
            else:
                roi = f

            self.calibrated_off_level = np.median(roi)
            logging.info("Calibrated OFF level set to: %.1f", self.calibrated_off_level)

            return True
        except Exception as e: # pylint: disable=broad-exception-caught
            logging.warning("Failed to lock exposure: %s", e)
            return False

    def autofocus_sweep(self):
        """Automatically find optimal focus using bidirectional hill-climbing algorithm."""
        # pylint: disable=too-many-locals,too-many-branches,too-many-statements
        if not self.autofocus:
            return

        if not isinstance(self.cam, X86CameraDriver):
            logging.info("Autofocus only supported for X86CameraDriver")
            return

        # Create/clean autofocus debug directory
        autofocus_dir = "/tmp/autofocus"
        if os.path.exists(autofocus_dir):
            shutil.rmtree(autofocus_dir)
        os.makedirs(autofocus_dir)
        logging.info("Created autofocus debug directory: %s", autofocus_dir)

        logging.info("Starting autofocus sweep (hill-climbing algorithm)...")

        # Detect camera's focus range by testing boundaries
        initial_focus = int(self.cam.cap.get(cv2.CAP_PROP_FOCUS))
        logging.info("Detecting camera focus range...")

        # Test upper boundary
        test_upper = initial_focus + 1000  # larger step to reach max focus range
        self.cam.cap.set(cv2.CAP_PROP_FOCUS, test_upper)
        time.sleep(0.1)
        actual_upper = int(self.cam.cap.get(cv2.CAP_PROP_FOCUS))

        # Test lower boundary
        self.cam.cap.set(cv2.CAP_PROP_FOCUS, 0)
        time.sleep(0.1)
        actual_lower = int(self.cam.cap.get(cv2.CAP_PROP_FOCUS))

        # Restore initial focus
        self.cam.cap.set(cv2.CAP_PROP_FOCUS, initial_focus)
        time.sleep(0.1)

        # Determine focus range without artificial clamping
        focus_min = min(actual_lower, initial_focus)
        focus_max = max(actual_upper, initial_focus)
        logging.info("Camera focus range: %d to %d (current: %d)",
                     focus_min, focus_max, initial_focus)
        # Counter for saved debug images
        measurement_counter = [0]

        def measure_sharpness(focus_pos, settle_time=0.1, save_image=True):
            """Set focus and measure sharpness after settling."""
            self.cam.cap.set(cv2.CAP_PROP_FOCUS, focus_pos)
            time.sleep(settle_time)  # Critical: wait for motor to settle

            frame = self.cam.get_frame()
            if frame is None:
                return 0, None

            # Measure sharpness in center region
            h, w = frame.shape[:2]
            center_h, center_w = h // 2, w // 2
            roi_size = min(h, w) // 3
            y1, y2 = center_h - roi_size // 2, center_h + roi_size // 2
            x1, x2 = center_w - roi_size // 2, center_w + roi_size // 2

            center_roi = frame[y1:y2, x1:x2]
            sharpness = cv2.Laplacian(center_roi, cv2.CV_64F).var()

            # Show preview if enabled
            if self.preview:
                preview_frame = frame.copy()
                cv2.rectangle(preview_frame, (x1, y1), (x2, y2), (255, 255, 255), 2)
                cv2.putText(preview_frame, f"Focus: {focus_pos} | Sharpness: {sharpness:.1f}",
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.imshow("Camera Preview", preview_frame)
                cv2.waitKey(1)

            # Log each measurement
            logging.info("  Focus position: %d, Sharpness quality: %.2f", focus_pos, sharpness)

            # Save image to debug directory with focus and score in filename
            if save_image:
                measurement_counter[0] += 1
                filename = (f"{measurement_counter[0]:03d}_focus{focus_pos:04d}_"
                           f"score{sharpness:07.2f}.png")
                filepath = os.path.join(autofocus_dir, filename)
                cv2.imwrite(filepath, frame)

            return sharpness, frame

        # Capture initial state (already have initial_focus from range detection)
        logging.info("Initial focus position: %d", initial_focus)
        initial_sharpness, _ = measure_sharpness(initial_focus)
        logging.info("Initial focus: %d, sharpness: %.2f", initial_focus, initial_sharpness)

        # Hill-climbing parameters (slower steps for better visibility)
        # Scale step sizes based on focus range
        focus_range = focus_max - focus_min
        coarse_step = max(5, focus_range // 20)  # ~5% of range, minimum 5
        fine_step = 1  # Maximum precision for fine tuning

        logging.info("Using step sizes: coarse=%d, fine=%d (range=%d)",
                    coarse_step, fine_step, focus_range)

        best_focus = initial_focus
        best_sharpness = initial_sharpness

        # Phase 1: Coarse linear scan across focus range
        logging.info(
            "Phase 1: Coarse linear scan from %d to %d with step %d",
            focus_min, focus_max, coarse_step)
        for focus_pos in range(focus_min, focus_max + 1, coarse_step):
            sharpness, _ = measure_sharpness(focus_pos)
            if sharpness > best_sharpness:
                best_focus = focus_pos
                best_sharpness = sharpness
                logging.info("    New best: focus=%d, sharpness=%.2f", best_focus, best_sharpness)
        # Phase 2: Binary search around the best coarse focus
        left = max(focus_min, best_focus - coarse_step)
        right = min(focus_max, best_focus + coarse_step)
        logging.info(
            "Phase 2: Binary search between %d and %d (initial best=%d)...",
            left, right, best_focus)
        while left <= right:
            mid = (left + right) // 2
            mid_sharp, _ = measure_sharpness(mid)
            logging.info("  Tested focus=%d, sharpness=%.2f", mid, mid_sharp)
            # Evaluate neighbors
            left_sharp = None
            right_sharp = None
            if mid - 1 >= focus_min:
                left_sharp, _ = measure_sharpness(mid - 1)
            if mid + 1 <= focus_max:
                right_sharp, _ = measure_sharpness(mid + 1)
            # Move towards higher sharpness
            if left_sharp is not None and left_sharp > mid_sharp:
                right = mid - 1
            elif right_sharp is not None and right_sharp > mid_sharp:
                left = mid + 1
            else:
                # Peak found at mid
                best_focus = mid
                best_sharpness = mid_sharp
                logging.info(
                    "    ✓ Binary search peak: focus=%d, sharpness=%.2f",
                    best_focus, best_sharpness)
                break

        # Decide whether to apply new focus
        improvement_pct = ((best_sharpness - initial_sharpness) / initial_sharpness * 100) \
                         if initial_sharpness > 0 else 0

        if best_sharpness > initial_sharpness * 1.05:  # At least 5% improvement
            # Apply best focus
            final_sharpness, _ = measure_sharpness(best_focus)

            logging.info("✓ Autofocus improved: %d→%d, Sharpness: %.2f→%.2f (+%.1f%%)",
                        initial_focus, best_focus, initial_sharpness, final_sharpness,
                        improvement_pct)
        else:
            # Revert to initial - it was better
            final_sharpness, _ = measure_sharpness(initial_focus)
            best_focus = initial_focus

            logging.warning("✗ Autofocus did not improve (%.1f%% change). "
                          "Keeping initial position %d",
                          improvement_pct, initial_focus)

        # Keep preview window open for 2 seconds to show final result
        logging.info("Autofocus complete.")

        # Explicitly disable camera autofocus and lock focus position
        # Some cameras need this done repeatedly to stick
        logging.info("Locking focus at position %d...", best_focus)
        for _ in range(3):
            self.cam.cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
            time.sleep(0.05)
            self.cam.cap.set(cv2.CAP_PROP_FOCUS, best_focus)
            time.sleep(0.05)

        # Verify focus is locked
        actual_focus = int(self.cam.cap.get(cv2.CAP_PROP_FOCUS))
        actual_autofocus = int(self.cam.cap.get(cv2.CAP_PROP_AUTOFOCUS))
        logging.info("Focus locked: position=%d, hw_autofocus=%s",
                     actual_focus, "DISABLED" if actual_autofocus == 0 else "ENABLED")

        if self.preview:
            logging.info("Showing final result for 2 seconds...")
            time.sleep(2.0)
            # Do NOT destroy window here, keep it open for seamless transition
            # cv2.destroyWindow("Autofocus Sweep")
            # cv2.waitKey(1)

    def wait_for_led_off(self):
        """ Monitors ROI brightness until it drops significanty """
        # pylint: disable=too-many-locals
        logging.info("Waiting for LED to turn OFF...")
        y1, y2, x1, x2 = self.roi

        # Measure "High" State
        f = self.cam.get_frame()
        roi = f[y1:y2, x1:x2]
        high_val = self._measure_roi(roi)

        if self.calibrated_off_level is not None:
            # Use calibrated level if available (most robust)
            # Allow some margin for noise (e.g. +30 or +50%)
            drop_threshold = self.calibrated_off_level + 30.0
            logging.info("Using Calibrated OFF Threshold: %.1f (Base: %.1f)",
                        drop_threshold, self.calibrated_off_level)
        elif self.adaptive_off:
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

            # We need BOTH conditions ideally, or at least a very strong drop
            # If we are using calibrated level, we trust it more.
            if self.calibrated_off_level is not None:
                if curr_val < (self.calibrated_off_level + 20.0):
                    logging.info("LED OFF Confirmed (Calibrated): %.1f < %.1f", curr_val, self.calibrated_off_level + 20.0)
                    time.sleep(0.1)
                    return True
            elif drop_cond or abs_cond:
                # Found an OFF state - accept it immediately
                # This works even if LED is flashing, we just catch it during OFF phase
                logging.info("LED OFF Confirmed (Curr: %.1f < Threshold: %.1f) [Drop: %s, Abs: %s]",
                           curr_val, drop_threshold, drop_cond, abs_cond)
                time.sleep(0.1) # Brief settling
                return True

            # Only log periodically to avoid spam
            if int(time.time() - start) % 2 == 0:  # Every 2 seconds
                logging.info("Waiting for drop... Curr: %.0f (Threshold: %.0f)",
                           curr_val, drop_threshold)
            time.sleep(0.1)

        logging.warning("Timeout waiting for LED off! Noise floor might be inaccurate.")
        return False


    def start_monitoring(self, max_pulses=None, on_pulse_callback=None):
        """
        Start the monitoring loop.

        Args:
            max_pulses: Optional limit on number of pulses to detect before exiting.
            on_pulse_callback: Optional callback function(timestamp, duration, gap) called on detection.
        """
        # pylint: disable=too-many-locals, too-many-statements, too-many-branches, too-many-nested-blocks
        # 3. Calibrate Noise Floor (OFF State)
        logging.info("Calibrating OFF-state noise floor...")

        # Ensure we are in OFF state before calibrating
        # We might have just come from a signal detection, so wait for OFF
        self.wait_for_led_off()

        # Debug: Check exposure
        curr_exp = self.cam.cap.get(cv2.CAP_PROP_EXPOSURE)
        logging.info("[Debug] Current Exposure: %.1f", curr_exp)

        # Flush buffer to ensure we get fresh frames with current exposure
        for _ in range(5):
            self.cam.get_frame()

        y1, y2, x1, x2 = self.roi
        metric_samples = []
        bg_brightness_samples = []  # Track background for saturation

        for _ in range(self.noise_floor_window):
            frame = self.cam.get_frame()
            if frame is None:
                continue
            roi = frame[y1:y2, x1:x2]
            metric_samples.append(self._measure_roi(roi))
            bg_brightness_samples.append(np.median(roi))  # Background brightness

        avg_noise_level = np.median(metric_samples)
        std_noise_level = np.std(metric_samples)

        # Initialize noise floor history with calibrated samples
        # Fill the history window with the measured noise level to establish baseline
        #logging.warning("CHANGE FOR NOISE FLOOR HISTORY")
        self.noise_floor_history = metric_samples[:self.noise_floor_window]

        # Estimate ON level and Signal Strength
        # Sanity check: detected_on_brightness must be significantly above noise floor
        # Otherwise, we likely measured the LED after it turned off
        # Estimate ON level and Signal Strength
        # Sanity check: detected_on_brightness must be significantly above noise floor
        # AND comparable to the peak strength we saw during detection
        use_on_brightness = False
        if hasattr(self, 'detected_on_brightness'):
            signal_from_brightness = self.detected_on_brightness - avg_noise_level
            # It must be > 10 units AND at least 50% of the peak strength we saw
            if (signal_from_brightness > 10.0 and
                signal_from_brightness > self.detected_peak_strength * 0.5):
                use_on_brightness = True

        if use_on_brightness:
            estimated_on_level = self.detected_on_brightness
            logging.info("[Debug] Using detected_on_brightness: %.1f", estimated_on_level)
        else:
            if hasattr(self, 'detected_on_brightness'):
                logging.warning("detected_on_brightness (%.1f) too close to noise (%.1f) or weak. Fallback to peak strength.",
                              self.detected_on_brightness, avg_noise_level)

            logging.info("[Debug] Using detected_peak_strength: %.1f", self.detected_peak_strength)
            estimated_on_level = min(255.0, avg_noise_level + self.detected_peak_strength)


        self.calibrated_signal_strength = max(10.0, estimated_on_level - avg_noise_level)
        self.current_noise_floor = avg_noise_level

        # Set thresholds for Hysteresis
        # High threshold to START detection (reject noise)
        # Low threshold to CONTINUE detection (capture tails)

        # High: Max of (Noise + 30% Signal) AND (Noise + 2 * StdDev)
        high_signal_margin = self.calibrated_signal_strength * 0.3
        high_noise_margin = 2.0 * std_noise_level
        self.thresh_high = min(253.0, self.current_noise_floor + max(high_signal_margin, high_noise_margin))

        # Low: Max of (Noise + 20% Signal) AND (Noise + 2 * StdDev)
        low_signal_margin = self.calibrated_signal_strength * 0.2
        low_noise_margin = 2.0 * std_noise_level
        self.thresh_low = min(253.0, self.current_noise_floor + max(low_signal_margin, low_noise_margin))

        metric_name = "Contrast" if self.use_contrast else "Brightness"
        logging.info("Noise (%s): %.1f (Std: %.1f) | Est. ON: %.1f | Signal: %.1f",
                    metric_name, avg_noise_level, std_noise_level, estimated_on_level,
                    self.calibrated_signal_strength)
        logging.info("Thresholds: High=%.1f (Start), Low=%.1f (Continue)",
                     self.thresh_high, self.thresh_low)

        # Saturation check
        if self.log_saturation and np.mean(bg_brightness_samples) > 240: # Use mean of samples
            logging.warning("High ambient brightness detected! Sensor may be saturating.")

        start_time = time.time()
        pulses_found = 0
        last_pulse = start_time
        last_off_time = start_time # Initialize to avoid UnboundLocalError

        # --- Monitoring Loop ---
        logging.info("--- MONITORING STARTED ---")

        self.led_state = False  # Reset LED state
        last_active_time = last_pulse # Track last time signal was above threshold

        # Estimate frame interval for duration correction
        # We can use the interval between the last few frames if available, or default to 33ms
        self.frame_interval = 0.033 # Default to 30fps
        last_frame_time = time.time()

        # Stats aggregation
        report_period = 1.0
        last_report = time.time()

        int_max = 0
        int_min = 255 if not self.use_contrast else 0 # Initialize int_min correctly for contrast
        t_max = last_pulse
        t_min = last_pulse

        pulse_count = 0

        while True:  # pylint: disable=too-many-nested-blocks
            if max_pulses is not None and pulses_found >= max_pulses:
                logging.info("Reached max pulses (%d). Exiting.", max_pulses)
                break
            # Capture frame
            frame = self.cam.get_frame() # Changed from read_frame() to get_frame()
            if frame is None:
                break

            now = time.time()
            # Update frame interval estimate
            current_interval = now - last_frame_time
            if 0.001 < current_interval < 0.2: # Filter reasonable values
                # Simple moving average
                self.frame_interval = (self.frame_interval * 0.9) + (current_interval * 0.1)
            last_frame_time = now

            # Process ROI
            if self.roi:
                y1, y2, x1, x2 = self.roi
                roi = frame[y1:y2, x1:x2]
            else:
                roi = frame

            # Measure signal
            val = self._measure_roi(roi) # Changed from get_signal_strength() to _measure_roi()

            # Calculate ROI stats for exposure logic
            roi_median = np.median(roi)
            roi_p95 = np.percentile(roi, 95)

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
                        # Only if the image is also reasonably bright (median > 100)
                        if recent_sat_pct > 0.5 and roi_median > 100:
                            # More than 50% of recent frames saturated - reduce exposure
                            new_exposure = current_exposure * 0.7
                            self.cam.cap.set(cv2.CAP_PROP_EXPOSURE, new_exposure)
                            self.last_exposure_adjust = now
                            logging.info("Continuous Exposure: Saturation %.0f%%, "
                                       "reducing %.1f→%.1f",
                                       recent_sat_pct * 100, current_exposure, new_exposure)
                            self.saturation_window = []

                        # Check for high ambient light via skip counter
                        # Only trigger if we're repeatedly hitting the noise floor cap
                        # AND the image is actually bright (median > 100)
                        # This prevents reducing exposure when we have high contrast but dark image (e.g. floor=0)
                        elif self.noise_floor_skip_count > 100 and roi_median > 100:  # Cap hit >100 times
                            # Ambient light has increased significantly
                            new_exposure = current_exposure * 0.7
                            self.cam.cap.set(cv2.CAP_PROP_EXPOSURE, new_exposure)
                            self.last_exposure_adjust = now
                            logging.info("Continuous Exposure: Noise floor capped %d times, "
                                       "reducing %.1f→%.1f",
                                       self.noise_floor_skip_count,
                                       current_exposure, new_exposure)
                            self.saturation_window = []
                            # Reset noise floor history to recalibrate
                            self.noise_floor_history = []
                            self.noise_floor_skip_count = 0

                        # Check if we need to increase exposure (signal too weak OR low ambient)
                        elif recent_sat_pct < 0.05:  # Less than 5% saturation
                            # Two scenarios for increasing exposure:
                            # 1. Signal is weak during pulse detection
                            # 2. Noise floor has dropped significantly (ambient light removed)

                            should_increase = False
                            reason = ""

                            # Scenario 1: Weak signal during pulse
                            if val > self.thresh_bright:  # Currently in a pulse
                                signal_strength = val - self.current_noise_floor
                                if signal_strength < self.calibrated_signal_strength * 0.5:
                                    should_increase = True
                                    reason = f"Signal weak ({signal_strength:.0f} < " \
                                           f"{self.calibrated_signal_strength * 0.5:.0f})"

                            # Scenario 2: Noise floor dropped (ambient light removed)
                            # Only if we have a stable noise floor history AND exposure is low
                            elif (len(self.noise_floor_history) >= 5 and
                                  self.current_noise_floor <
                                  self.calibrated_signal_strength * 0.1 and
                                  current_exposure < 70.0):
                                # Only if exposure is significantly reduced
                                # Noise floor is very low, can increase exposure
                                should_increase = True
                                reason = f"Low ambient (floor={self.current_noise_floor:.0f})"

                            # Scenario 3: Image is too dark (recovery mode)
                            # If the 95th percentile is very low, we are likely underexposed
                            elif roi_p95 < 50.0 and current_exposure < 83.0:
                                should_increase = True
                                reason = f"Image too dark (p95={roi_p95:.1f})"

                            if should_increase:
                                # Cap at initial exposure
                                new_exposure = min(current_exposure * 1.3, 83.0)
                                self.cam.cap.set(cv2.CAP_PROP_EXPOSURE, new_exposure)
                                self.last_exposure_adjust = now
                                logging.info("Continuous Exposure: %s, "
                                           "increasing %.1f→%.1f",
                                           reason, current_exposure, new_exposure)
                                self.saturation_window = []

            # Update Interval Stats
            if val > int_max:
                int_max = val
                t_max = now
            if val < int_min:
                int_min = val
                t_min = now

            # Reset min_while_on to current value to avoid getting stuck on a past low
            self.min_while_on = val

            # Check if LED is active (must be calculated before adaptive threshold update)
            # Hysteresis Thresholding
            if not self.led_state:
                # To turn ON, must exceed HIGH threshold
                is_active = val > self.thresh_high
            else:
                # To stay ON, must exceed LOW threshold
                is_active = val > self.thresh_low

            gap = now - last_pulse

            # Adaptive Threshold Update (Bidirectional)
            # Track noise floor changes in both directions (brightening and dimming)
            # Freeze updates for 3s after exposure changes to avoid contamination
            time_since_exposure_adjust = now - self.last_exposure_adjust

            if not is_active and time_since_exposure_adjust > 3.0:
                # LED is OFF AND exposure has stabilized (3s since last adjustment)
                last_off_time = now  # Update last OFF time

                # Cap noise floor updates to prevent runaway threshold increases
                # Only update if the new value is reasonable relative to calibrated baseline
                # Changed from absolute check to relative check to handle high noise floors
                max_reasonable_floor = self.current_noise_floor + self.calibrated_signal_strength
                if val < max_reasonable_floor or len(self.noise_floor_history) == 0:
                    self.noise_floor_history.append(val)
                    if len(self.noise_floor_history) > self.noise_floor_window:
                        self.noise_floor_history.pop(0)

                    # Recalculate noise floor and threshold
                    self.current_noise_floor = np.mean(self.noise_floor_history)
                    self.thresh_bright = (self.current_noise_floor +
                                         (self.calibrated_signal_strength * 0.5))
                    # Only reset skip counter if noise floor has dropped significantly
                    # This allows counter to accumulate during sustained high ambient
                    if self.current_noise_floor < self.calibrated_signal_strength * 0.3:
                        self.noise_floor_skip_count = 0
                else:
                    # Noise floor is unreasonably high - likely ambient light issue
                    # Don't update threshold, let exposure adjustment handle it
                    self.noise_floor_skip_count += 1
                    # FALLBACK: If we've skipped too many times
                    # (exposure adjustment failed or maxed out)
                    # We must accept the new reality to avoid being blind
                    if self.noise_floor_skip_count > 200:
                        logging.warning(
                            "Noise floor stuck high (skipped %d). Forcing update.",
                            self.noise_floor_skip_count)
                        self.noise_floor_history.append(val)
                        if len(self.noise_floor_history) > self.noise_floor_window:
                            self.noise_floor_history.pop(0)
                        self.current_noise_floor = np.mean(self.noise_floor_history)
                        self.thresh_bright = (self.current_noise_floor +
                                             (self.calibrated_signal_strength * 0.5))
                        self.noise_floor_skip_count = 0

                    # Only log occasionally to avoid spam (every 50 skips)
                    elif self.noise_floor_skip_count % 50 == 1:
                        logging.info("Noise floor capped: val=%.0f > max=%.0f (skipped %d times)",
                                   val, max_reasonable_floor, self.noise_floor_skip_count)
            else:
                # LED is ON - check if we've been stuck in ON state for too long
                # Use time since last OFF state, not time since last pulse
                time_since_off = now - last_off_time

                # Track minimum value while in ON state (doesn't reset with report period)
                if time_since_off < 0.1:
                    self.min_while_on = val
                else:
                    self.min_while_on = min(self.min_while_on, val)

                # Only update noise floor from "stuck ON" if reasonable
                max_reasonable_floor = self.current_noise_floor + self.calibrated_signal_strength

                # If stuck ON for > 10s, force update even if high
                force_update = time_since_off > 10.0

                if (time_since_off > 5.0 and
                    self.min_while_on > (self.current_noise_floor +
                              self.calibrated_signal_strength * 0.3) and
                    (self.min_while_on < max_reasonable_floor or
                     force_update)):

                    # If forced, use current if higher than min_while_on
                    # This helps when stuck on high-contrast clutter
                    update_val = self.min_while_on
                    if force_update and val > self.min_while_on:
                        # Blend min and current to pull average up
                        update_val = (self.min_while_on + val) / 2.0

                    self.noise_floor_history.append(update_val)
                    if len(self.noise_floor_history) > self.noise_floor_window:
                        self.noise_floor_history.pop(0)

                    self.current_noise_floor = np.mean(self.noise_floor_history)
                    self.thresh_bright = (self.current_noise_floor +
                                         (self.calibrated_signal_strength * 0.5))
                    # Reset min_while_on to current value to avoid getting stuck on a past low
                    self.min_while_on = val

            # Check if LED is active (must be calculated before adaptive threshold update)
            is_active = val > self.thresh_bright

            # Debounce Logic
            if is_active:
                last_active_time = now
                if not self.led_state:
                    # LED just turned ON
                    self.led_state = True
                    self.led_on_time = now
                    self.pulse_samples = [] # Initialize pulse samples

                # Collect samples while ON
                self.pulse_samples.append(val)
            else:
                # LED is currently below threshold
                if self.led_state:
                    # Check if it has been OFF for long enough to confirm
                    min_gap_duration = 0.1 # 100ms debounce
                    if now - last_active_time > min_gap_duration:
                        # Confirmed OFF
                        self.led_state = False
                        # Duration ends at the last known active time
                        # Duration ends at the last known active time
                        # Add one frame interval to account for the duration of the last frame itself
                        duration = (last_active_time - self.led_on_time + self.frame_interval) * 1000 # ms

                        # Calculate gap from the END of the previous pulse to the START of this one
                        # last_pulse tracks the END of the previous pulse
                        gap = self.led_on_time - last_pulse

                        timestamp = get_timestamp()
                        sys.stdout.write(f"\033[91m[{timestamp}] [LED OFF] Duration: {duration:.0f}ms\033[0m\n")

                        # Filter out short noise spikes
                        if duration >= self.min_pulse_duration:
                            last_pulse = last_active_time # Update last pulse end time
                            if gap > 0.1: # Reduced to 0.1s to catch faster pulses
                                # Calculate pulse stats
                                stats_str = ""
                                if hasattr(self, 'pulse_samples') and self.pulse_samples:
                                    p_mean = np.mean(self.pulse_samples)
                                    p_peak = np.max(self.pulse_samples)
                                    p_std = np.std(self.pulse_samples)
                                    stats_str = f" | Mean: {p_mean:.0f}, Peak: {p_peak:.0f}, Std: {p_std:.1f}"

                                timestamp = get_timestamp()
                                sys.stdout.write(f"\033[96m[{timestamp}] [PULSE DETECTED] "
                                               f"Gap: {gap:.1f}s | Duration: {duration:.0f}ms{stats_str}\033[0m\n")

                                if on_pulse_callback:
                                    on_pulse_callback(timestamp, duration, gap)

                                pulse_count += 1
                                if max_pulses and pulse_count >= max_pulses:
                                    logging.info("Reached max pulses (%d). Exiting.", max_pulses)
                                    return
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
                t_max = now
                t_min = now
                if self.log_saturation:
                    self.saturation_count = 0
                    self.total_frames = 0

            if self.preview and isinstance(self.cam, X86CameraDriver):
                debug = frame.copy()
                cv2.rectangle(debug, (x1, y1), (x2, y2), (255, 255, 255), 1)
                cv2.putText(debug, f"{val:.0f}", (x1, y1-5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            (0, 255, 0) if is_active else (0, 0, 255), 2)
                cv2.imshow("Monitor", debug)
                if cv2.waitKey(1) == ord('q'):
                    break

    def setup_camera_locked(self):
        """
        Sets up the camera for one-shot detection:
        1. Auto-Exposure to find scene.
        2. Autofocus sweep.
        3. Lock Exposure and Focus.
        """
        logging.info("Setting up camera (Locked Mode)...")

        # 1. Start with Auto-Exposure
        try:
            self.cam.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 3) # Auto
            logging.info("Enabled Auto-Exposure")
            time.sleep(1.0) # Settle
        except Exception as e: # pylint: disable=broad-exception-caught
            logging.warning("Failed to set Auto-Exposure: %s", e)

        # if self.preview:
        #     self.aim_camera()

        # 2. Autofocus
        self.autofocus_sweep()

        # 3. Lock Exposure
        self.lock_current_exposure()

        logging.info("Camera setup complete (Locked).")

    def capture_frame_buffer(self, duration):
        """
        Captures a sequence of frames for the specified duration.
        Returns a list of (timestamp, frame) tuples.
        """
        # pylint: disable=too-many-nested-blocks
        logging.info("Capturing frame buffer for %.1fs...", duration)
        frames = []
        start_time = time.time()

        # Visualization helper state
        viz_baseline = None

        while (time.time() - start_time) < duration:
            frame = self.cam.get_frame()
            if frame is not None:
                frames.append((time.time(), frame.copy()))

                if self.preview:
                    preview_frame = frame.copy()
                    if self.roi:
                        y1, y2, x1, x2 = self.roi
                        # Draw ROI box
                        cv2.rectangle(preview_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        # Draw label
                        cv2.putText(preview_frame, "ROI", (x1, y1-5),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                    else:
                        # Visualize potential ROI during scan
                        if len(frame.shape) == 3 and frame.shape[2] == 3:
                            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                        else:
                            gray = frame

                        if viz_baseline is None:
                            viz_baseline = gray
                        else:
                            diff = cv2.absdiff(gray, viz_baseline)
                            _, max_val, _, max_loc = cv2.minMaxLoc(diff)

                            # Draw circles to show we are scanning/detecting activity
                            if max_val > 10: # Minimum activity threshold
                                cv2.circle(preview_frame, max_loc, 10, (0, 255, 255), 1)
                                cv2.circle(preview_frame, max_loc, 20, (0, 255, 255), 1)
                                cv2.putText(preview_frame, f"Scanning... {max_val:.0f}", (10, 30),
                                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

                    cv2.imshow("Camera Preview", preview_frame)
                    cv2.waitKey(1)

        logging.info("Captured %d frames in %.1fs (%.1f FPS)",
                     len(frames), duration, len(frames)/duration)
        return frames

    def _find_roi_from_variance(self, frames):
        """Helper to find ROI based on variance map."""
        # pylint: disable=too-many-locals
        # Convert to grayscale first if needed
        gray_frames = []
        for _, f in frames:
            if len(f.shape) == 3 and f.shape[2] == 3:
                gray_frames.append(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY))
            else:
                gray_frames.append(f) # Already grayscale or single channel

        stack = np.stack(gray_frames, axis=0)

        # Calculate variance map
        variance_map = np.var(stack, axis=0)

        # Find the area with max variance
        # Blur slightly to reduce noise
        variance_map = cv2.GaussianBlur(variance_map, (5, 5), 0)

        # Find max location
        _, max_val, _, max_loc = cv2.minMaxLoc(variance_map)
        logging.info("Max variance: %.1f at %s", max_val, max_loc)

        if max_val < 10.0: # Threshold for "no activity"
            logging.warning("No significant activity detected.")
            return None

        # Define ROI around max location (e.g., 20x20)
        x, y = max_loc
        w, h = 20, 20
        x1 = max(0, x - w//2)
        y1 = max(0, y - h//2)
        x2 = min(stack.shape[2], x + w//2)
        y2 = min(stack.shape[1], y + h//2)

        roi = (y1, y2, x1, x2)
        logging.info("ROI found: %s", roi)
        return roi

    def analyze_frame_buffer(self, frames):
        """
        Analyzes the frame buffer to find the LED ROI and extract the signal.
        Returns (roi, signal_values, timestamps).
        """
        # pylint: disable=too-many-locals
        if not frames:
            return None, [], []

        logging.info("Analyzing %d frames...", len(frames))

        # 1. Compute Frame Differences to find ROI (if not already found)
        # Check self.roi explicitly in case it was set externally (e.g., by pre-scan)
        if self.roi is None:
            self.roi = self._find_roi_from_variance(frames)
            if self.roi is None:
                return None, [], []

        y1, y2, x1, x2 = self.roi

        # 2. Extract Signal from ROI
        signal_values = []
        timestamps = [ts for ts, _ in frames]

        # Need to ensure frames are grayscale for signal extraction
        gray_frames = []
        for _, f in frames:
            if len(f.shape) == 3 and f.shape[2] == 3:
                gray_frames.append(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY))
            else:
                gray_frames.append(f)

        for i in range(len(frames)):
            # Average brightness in ROI
            roi_frame = gray_frames[i][y1:y2, x1:x2]
            avg_val = np.mean(roi_frame)
            signal_values.append(avg_val)

        return self.roi, signal_values, timestamps

    def _detect_pulses(self, signal_values, timestamps, threshold, avg_frame_interval):
        """Helper to detect pulses from binary signal."""
        # pylint: disable=too-many-locals
        binary_signal = [1 if v > threshold else 0 for v in signal_values]
        pulses = []
        current_state = 0
        start_time = 0
        start_index = 0

        # We need to access previous values for interpolation
        for i, val in enumerate(binary_signal):
            if i == 0:
                current_state = val
                if val == 1:
                    start_time = timestamps[0]
                    start_index = 0
                continue

            state = val # This is binary_signal[i]
            t2 = timestamps[i]
            t1 = timestamps[i-1]
            v2 = signal_values[i]
            v1 = signal_values[i-1]

            # Rising Edge: 0 -> 1
            if current_state == 0 and state == 1:
                # Interpolate exact crossing time
                # v1 <= threshold < v2
                denom = v2 - v1
                fraction = (threshold - v1) / denom if denom != 0 else 0.5
                start_time = t1 + (t2 - t1) * fraction
                start_index = i
                current_state = 1

            # Falling Edge: 1 -> 0
            elif current_state == 1 and state == 0:
                # Interpolate exact crossing time
                # v1 > threshold >= v2
                denom = v2 - v1
                fraction = (threshold - v1) / denom if denom != 0 else 0.5
                end_time = t1 + (t2 - t1) * fraction

                duration = (end_time - start_time) * 1000.0 # ms

                # Calculate frame-based duration
                # Number of frames where signal was 1 (approximate)
                # This is just end_index - start_index?
                # binary_signal[start_index] is 1. binary_signal[i] is 0.
                # So frames are start_index to i-1.
                num_frames = i - start_index
                duration_frames = num_frames * avg_frame_interval * 1000.0

                pulses.append({
                    'start': start_time,
                    'end': end_time,
                    'duration': duration,
                    'num_frames': num_frames,
                    'duration_frames': duration_frames
                })
                current_state = 0
        return pulses

    def verify_signal(self, signal_values, timestamps, expected_period, expected_duration, tolerance=0.2, frames=None):
        """
        Verifies if the extracted signal matches the expected parameters.
        If frames is provided, saves frames for invalid pulses to /tmp/badpulses.
        """
        # pylint: disable=too-many-locals, too-many-branches, too-many-nested-blocks
        if not signal_values:
            return False

        # Normalize signal
        sig_min = np.min(signal_values)
        sig_max = np.max(signal_values)
        sig_range = sig_max - sig_min

        logging.info("Signal Range: %.1f - %.1f (Delta: %.1f)", sig_min, sig_max, sig_range)

        if sig_range < 10.0:
            logging.warning("Signal range too low (%.1f). Noise?", sig_range)
            return False

        # Thresholding
        # Use mid-point
        threshold = sig_min + (sig_range * 0.5)

        # Calculate average frame interval
        avg_frame_interval = 0.0
        if len(timestamps) > 1:
            avg_frame_interval = (timestamps[-1] - timestamps[0]) / (len(timestamps) - 1)

        # Detect pulses
        pulses = self._detect_pulses(signal_values, timestamps, threshold, avg_frame_interval)

        logging.info("Detected %d pulses", len(pulses))
        for p in pulses:
            logging.info("  Duration: %.1fms (Time) vs %.1fms (Frames: %d)",
                         p['duration'], p['duration_frames'], p['num_frames'])

        if len(pulses) < 2:
            logging.warning("Not enough pulses detected.")
            return False

        # Prepare bad pulses directory
        # User wants a library of bad pulses, so use a timestamped folder for each run
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        bad_pulse_base_dir = f"/tmp/bad_pulses/run_{timestamp}"

        if frames and not os.path.exists(bad_pulse_base_dir):
            os.makedirs(bad_pulse_base_dir)
            logging.info("Created bad pulse directory: %s", bad_pulse_base_dir)

        # Verify Duration and Period
        valid_pulses = 0
        for i, p in enumerate(pulses):
            is_pulse_valid = True

            # Period Check (if not first)
            period = 0.0
            if i > 0:
                period = p['start'] - pulses[i-1]['start']
                period_diff = abs(period - expected_period)
                if period_diff > (expected_period * tolerance):
                    logging.warning("Pulse %d: Duration=%.1fms, Period=%.3fs [INVALID PERIOD] (Exp: %.3fs)",
                                  i, p['duration'], period, expected_period)
                    is_pulse_valid = False

            # Duration Check
            dur_diff = abs(p['duration'] - expected_duration)
            if dur_diff > (expected_duration * tolerance) and dur_diff > 10.0:
                logging.warning("Pulse %d: Duration=%.1fms, Period=%.3fs [INVALID DURATION] (Exp: %dms)",
                              i, p['duration'], period, expected_duration)
                is_pulse_valid = False

            # Collect raw signal values for this pulse
            pulse_signal_values = []
            if frames:
                # Find frames within start and end time
                # Use slightly wider window for logging context
                p_start = p['start'] - 0.1
                p_end = p['end'] + 0.1
                for j, (t, _) in enumerate(frames): # Unused frame
                    if p_start <= t <= p_end:
                        if j < len(signal_values):
                            pulse_signal_values.append(f"{signal_values[j]:.1f}")

            if is_pulse_valid:
                logging.info("Pulse %d: Duration=%.1fms, Period=%.3fs [VALID]", i, p['duration'], period)
                logging.info("  Raw Signal: %s", pulse_signal_values)
                valid_pulses += 1
            else:
                # Save frames for bad pulse
                if frames:
                    # Create subdirectory for this pulse
                    pulse_dir = os.path.join(bad_pulse_base_dir, f"pulse_{i}")
                    if not os.path.exists(pulse_dir):
                        os.makedirs(pulse_dir)

                    # Find frames within start and end time
                    p_start = p['start'] - 0.1
                    p_end = p['end'] + 0.1

                    saved_count = 0
                    for j, (t, frame) in enumerate(frames):
                        if p_start <= t <= p_end:
                            # Append signal value to filename
                            sig_val_str = ""
                            if j < len(signal_values):
                                sig_val_str = f"_val_{signal_values[j]:.1f}"

                            fname = os.path.join(pulse_dir, f"frame_{j}_{t:.3f}{sig_val_str}.jpg")
                            cv2.imwrite(fname, frame)
                            saved_count += 1

                    logging.info("Saved %d frames for bad pulse %d to %s", saved_count, i, pulse_dir)
                    logging.info("Raw signal values for bad pulse %d: %s", i, pulse_signal_values)

        # Quality Check
        total_pulses = len(pulses)
        valid_ratio = valid_pulses / total_pulses if total_pulses > 0 else 0.0
        logging.info("Signal Quality: %d/%d valid pulses (%.1f%%)", valid_pulses, total_pulses, valid_ratio * 100)

        if valid_ratio < 0.5:
            logging.warning("Signal quality too low (<50%% valid).")
            return False

        if valid_pulses >= 2: # Require at least 2 valid pulses (1 period check)
            logging.info("Signal Verified! (%d valid pulses)", valid_pulses)
            return True

        return False

    def run_one_shot(self, expected_period, expected_duration, tolerance=0.2, num_pulses=4):
        """
        Run a one-shot detection sequence.

        Args:
            expected_period: Expected pulse period in seconds.
            expected_duration: Expected pulse duration in ms.
            num_pulses: Number of pulses to capture (default 4).
        Run a one-shot detection sequence using Frame Difference approach.
        """
        logging.info("Starting One-Shot Detection (Period: %.2fs, Duration: %dms, Tolerance: %.1f, Pulses: %d)",
                    expected_period, expected_duration, tolerance, num_pulses)

        self.cam.start()

        # 1. Setup Camera (Locked Mode)
        self.setup_camera_locked()

        # 2. Pre-capture ROI Scan
        # Scan for at least 2 pulses to ensure we catch one
        scan_duration = max(2.0, expected_period * 2.1)
        logging.info("Performing pre-capture ROI scan (%.1fs)...", scan_duration)

        pre_frames = self.capture_frame_buffer(scan_duration)
        self.roi, _, _ = self.analyze_frame_buffer(pre_frames)

        if self.roi:
            logging.info("Pre-capture ROI found: %s", self.roi)
        else:
            logging.warning("Pre-capture ROI scan failed. Will attempt post-capture.")

        # 3. Capture Frame Buffer
        # Capture for num_pulses periods + buffer
        capture_duration = max(5.0, expected_period * (num_pulses + 1.0))
        frames = self.capture_frame_buffer(capture_duration)

        self.cam.stop()

        if not frames:
            logging.error("Failed to capture frames.")
            return False

        logging.info("Analyzing %d frames...", len(frames))

        # 4. Analyze Frames (if ROI not found yet, it will be found here)
        # If ROI was found in pre-scan, analyze_frame_buffer should use it?
        # analyze_frame_buffer calculates ROI if self.roi is None.
        # So if we set self.roi above, it should use it?
        # Wait, analyze_frame_buffer implementation (lines 1400+) RE-CALCULATES ROI every time!
        # It doesn't check self.roi!
        # I need to modify analyze_frame_buffer to respect self.roi if set.
        _, signal_values, timestamps = self.analyze_frame_buffer(frames)

        # 5. Verify Signal
        success = self.verify_signal(signal_values, timestamps, expected_period, expected_duration, tolerance, frames)

        if self.preview:
            cv2.destroyAllWindows()

        if success:
            print("\n[SUCCESS] Signal Verified!")
            return True
        print("\n[FAILURE] Signal Verification Failed.")
        return False

    def run_offline_analysis(self, folder_path, expected_period, expected_duration, tolerance=0.2):
        """
        Run detection on a folder of images (offline mode).

        Args:
            folder_path: Path to folder containing .jpg images.
            expected_period: Expected pulse period in seconds.
            expected_duration: Expected pulse duration in ms.
            tolerance: Tolerance for timing validation.
        """
        # pylint: disable=too-many-locals
        logging.info("Starting Offline Analysis on folder: %s", folder_path)

        if not os.path.exists(folder_path):
            logging.error("Folder not found: %s", folder_path)
            return False

        # 1. Read images
        frames = []
        files = sorted([f for f in os.listdir(folder_path) if f.endswith(".jpg")])

        if not files:
            logging.error("No .jpg images found in %s", folder_path)
            return False

        logging.info("Found %d images", len(files))

        for f in files:
            # Parse timestamp from filename
            # Format: frame_INDEX_TIMESTAMP.jpg or frame_INDEX_TIMESTAMP_val_XXX.jpg
            # We need the timestamp part.
            # Example: frame_0_1.234.jpg -> 1.234
            try:
                parts = f.split('_')
                # parts[0] = frame
                # parts[1] = index
                # parts[2] = timestamp (maybe with .jpg or _val...)

                ts_part = parts[2]
                if ts_part.endswith(".jpg"):
                    ts_str = ts_part[:-4]
                else:
                    # Handle _val_XXX case
                    # If parts has more elements, timestamp is likely just parts[2]
                    # But we need to be careful if timestamp contains underscores (unlikely based on my code)
                    # My code uses f"frame_{j}_{t:.3f}" -> 1.234
                    # So it should be safe to take parts[2] and strip .jpg or stop at next _
                    ts_str = ts_part
                    if ".jpg" in ts_str:
                        ts_str = ts_str.replace(".jpg", "")

                t = float(ts_str)

                img_path = os.path.join(folder_path, f)
                img = cv2.imread(img_path)
                if img is not None:
                    frames.append((t, img))
                else:
                    logging.warning("Failed to read image: %s", f)

            except Exception as e: # pylint: disable=broad-exception-caught
                logging.warning("Skipping file %s: %s", f, e)
                continue

        if not frames:
            logging.error("No valid frames loaded.")
            return False

        # Sort by timestamp just in case
        frames.sort(key=lambda x: x[0])

        logging.info("Loaded %d frames. Duration: %.1fs", len(frames), frames[-1][0] - frames[0][0])

        # 2. Analyze Frames
        # We need to set self.roi if not set?
        # analyze_frame_buffer calls find_roi_from_variance if self.roi is None.
        # This should work fine.
        _, signal_values, timestamps = self.analyze_frame_buffer(frames)

        # 3. Verify Signal
        success = self.verify_signal(signal_values, timestamps, expected_period, expected_duration, tolerance, frames)

        return success

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

        # 3. Start Monitoring
        self.start_monitoring()

def main():
    parser = argparse.ArgumentParser(
        description="LED Detection and Monitoring System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Feature Flags (all enabled by default except adaptive-exposure):
  --use-contrast        Use contrast (max-median) instead of brightness
  --adaptive-roi        Automatically size ROI based on LED blob size
  --adaptive-off        Use variance-based OFF detection
  --log-saturation      Track and log saturation events
  --adaptive-exposure   Automatically adjust camera exposure (experimental)
  --autofocus           Automatically find optimal focus (X86 only)

To disable a feature, use --no-<feature>, e.g., --no-autofocus
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
                       help="Use fixed 60%% drop threshold")

    parser.add_argument("--log-saturation", dest="log_saturation",
                       action="store_true", default=True,
                       help="Track saturation events (default)")
    parser.add_argument("--no-log-saturation", dest="log_saturation", action="store_false",
                       help="Disable saturation logging")

    parser.add_argument("--adaptive-exposure", dest="adaptive_exposure",
                       action="store_true", default=True,
                       help="Auto-adjust exposure (experimental, X86 only)")

    parser.add_argument("--autofocus", dest="autofocus",
                       action="store_true", default=True,
                       help="Auto-focus using Laplacian variance sweep (default, X86 only)")
    parser.add_argument("--no-autofocus", dest="autofocus", action="store_false",
                       help="Disable autofocus")

    parser.add_argument("--min-pulse-duration", dest="min_pulse_duration",
                       type=int, default=25,
                       help="Minimum pulse duration in ms to count as valid (default: 25)")

    # One-shot mode arguments
    parser.add_argument("--expected-period", dest="expected_period", type=float,
                       help="Expected pulse period in seconds (enables one-shot mode)")
    parser.add_argument("--expected-duration", dest="expected_duration", type=int,
                       help="Expected pulse duration in ms (required for one-shot mode)")
    parser.add_argument("--tolerance", dest="tolerance", type=float, default=0.2,
                       help="Tolerance for timing validation (default: 0.2 = 20%%)")

    parser.add_argument("--offline", dest="offline_folder", type=str,
                       help="Run detection on a folder of images (offline mode)")

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

    try:
        monitor = PeakMonitor(
            interval=args.interval,
            threshold=args.threshold,
            preview=args.preview,
            use_contrast=args.use_contrast,
            adaptive_roi=args.adaptive_roi,
            adaptive_off=args.adaptive_off,
            log_saturation=args.log_saturation,
            adaptive_exposure=args.adaptive_exposure,
            autofocus=args.autofocus,
            min_pulse_duration=args.min_pulse_duration
        )

        if args.offline_folder:
            # Offline Mode
            if not args.expected_period or not args.expected_duration:
                logging.error("Offline mode requires --expected-period and --expected-duration")
                sys.exit(1)

            monitor.run_offline_analysis(
                args.offline_folder,
                args.expected_period,
                args.expected_duration,
                args.tolerance
            )

        elif args.expected_period and args.expected_duration:
            # One-Shot Mode
            monitor.run_one_shot(
                args.expected_period,
                args.expected_duration,
                args.tolerance
            )
        else:
            # Continuous Mode
            monitor.run()

    except KeyboardInterrupt:
        logging.info("Interrupted by user")
        sys.exit(0)
    except Exception as e: # pylint: disable=broad-exception-caught
        logging.exception("Fatal error: %s", e)
        sys.exit(1)

if __name__ == "__main__":
    main()
