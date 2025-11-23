# pylint: disable=no-member
import time
import sys
import platform
import signal
import logging
import argparse
import os
import numpy as np
import cv2

def add(a, b):
    """
    Add two numbers.
    """
    return a + b

# --- SYSTEM SETUP ---
SYS_OS = platform.system()
if SYS_OS == "Linux":
    os.environ["QT_QPA_PLATFORM"] = "xcb"

def setup_logging(debug_mode):
    """Set up logging configuration."""
    level = logging.DEBUG if debug_mode else logging.INFO
    logging.basicConfig(level=level, format='[%(levelname)s] %(message)s', datefmt='%H:%M:%S')

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
    def __init__(self, interval, threshold, preview=False,
                 use_contrast=True,      # 1a: Contrast-based detection
                 adaptive_roi=True,      # 1b: Adaptive ROI sizing
                 adaptive_off=True,      # 1c: Adaptive OFF detection
                 log_saturation=True,    # 2a: Saturation logging
                 adaptive_exposure=False # 2b: Adaptive exposure (experimental)
                ):
        self.cam = get_driver()
        self.interval = interval
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

        # Saturation tracking
        self.saturation_count = 0
        self.total_frames = 0

    def _measure_roi(self, roi):
        """Measure ROI using either contrast or brightness based on feature flag."""
        if self.use_contrast:
            # Contrast-based: max - median (robust against global illumination)
            return float(roi.max()) - float(np.median(roi))
        else:
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

    def wait_for_signal(self):
        """ Stares at scene using Peak Hold to find the LED """
        wait_time = self.interval + 10
        logging.info("Scanning for signal... (Max wait: %ss)", wait_time)
        print("    [Action] Waiting for LED to turn ON...\n")

        start_time = time.time()
        h, w = 480, 640
        accum_max_diff = np.zeros((h, w), dtype=np.float32)
        last_frame = None

        while time.time() - start_time < wait_time:
            frame = self.cam.get_frame()

            if last_frame is not None:
                # Detect change (Pixel-wise)
                diff = cv2.absdiff(frame, last_frame).astype(np.float32)
                np.maximum(accum_max_diff, diff, out=accum_max_diff)

                _, max_val, _, max_loc = cv2.minMaxLoc(accum_max_diff)

                if max_val > self.min_signal_strength:
                    self.detected_peak_strength = max_val
                    logging.info("SIGNAL DETECTED! Score: %.0f", max_val)
                    self.lock_roi(max_loc, frame.shape, accum_max_diff)
                    # Store actual brightness (not change score) for threshold calculation
                    roi_slice = frame[self.roi[0]:self.roi[1], self.roi[2]:self.roi[3]]
                    self.detected_on_brightness = self._measure_roi(roi_slice)
                    logging.info("[Debug] Initial ROI brightness: %.1f",
                                self.detected_on_brightness)
                    return True

            last_frame = frame

            peak = cv2.minMaxLoc(accum_max_diff)[1] if last_frame is not None else 0
            sys.stdout.write(f"\r    Scanning... {time.time()-start_time:.1f}s | "
                             f"Peak Change: {peak:.0f} (Req: {self.min_signal_strength})   ")
            sys.stdout.flush()

            if isinstance(self.cam, X86CameraDriver):
                cv2.imshow("Calibration", frame)
                if cv2.waitKey(1) == ord('q'):
                    sys.exit()

        print("")
        return False

    def lock_roi(self, loc, shape, accum_max_diff=None):
        """Lock Region of Interest."""
        c_x, c_y = loc
        h, w = shape

        if self.adaptive_roi and accum_max_diff is not None:
            # Adaptive ROI: measure blob size
            threshold_val = self.min_signal_strength * 0.5
            _, binary = cv2.threshold(accum_max_diff, threshold_val, 255, cv2.THRESH_BINARY)
            binary = binary.astype(np.uint8)

            # Find connected components
            num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary)

            # Find the component containing the peak location
            peak_label = labels[c_y, c_x]

            if peak_label > 0:  # 0 is background
                bbox = stats[peak_label]  # x, y, width, height, area
                blob_w, blob_h = bbox[2], bbox[3]

                # Use blob size with 1.5x margin, constrained to min/max
                size = int(max(blob_w, blob_h) * 0.75)  # 1.5x margin (size is half-width)
                size = max(16, min(64, size))  # Constrain between 32x32 and 128x128
                logging.info("Adaptive ROI: Blob size %dx%d -> ROI half-size %d",
                            blob_w, blob_h, size)
            else:
                size = 32  # Fallback
                logging.warning("Peak not in any blob, using default ROI size")
        else:
            # Fixed ROI (original behavior)
            size = 32

        x1, x2 = max(0, c_x - size), min(w, c_x + size)
        y1, y2 = max(0, c_y - size), min(h, c_y + size)
        self.roi = (y1, y2, x1, x2)
        logging.info("ROI Locked: %s (size: %dx%d)", self.roi, x2-x1, y2-y1)
        if isinstance(self.cam, X86CameraDriver):
            cv2.destroyWindow("Calibration")

    def calibrate_exposure(self):
        """Adaptively calibrate camera exposure to avoid saturation."""
        if not self.adaptive_exposure:
            return

        if not isinstance(self.cam, X86CameraDriver):
            logging.info("Adaptive exposure only supported for X86CameraDriver")
            return

        logging.info("Starting adaptive exposure calibration...")
        y1, y2, x1, x2 = self.roi
        target_bg = 80      # Target background brightness
        target_led = 200    # Target LED brightness (not saturated)

        for iteration in range(10):
            # Capture frame and measure
            f = self.cam.get_frame()
            roi = f[y1:y2, x1:x2]
            bg_brightness = np.median(roi)
            max_brightness = np.percentile(roi, 95)

            logging.info("Iteration %d: BG=%.1f, Peak=%.1f", iteration, bg_brightness, max_brightness)

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

    def wait_for_led_off(self):
        """ Monitors ROI brightness until it drops significanty """
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
            drop_threshold = max(initial_mean - (3 * noise_std), 10)  # Min 10 to avoid negatives
            logging.info("Adaptive OFF: Initial=%.1f, Std=%.1f, Threshold=%.1f",
                        initial_mean, noise_std, drop_threshold)
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
                logging.info("LED OFF Confirmed (High: %.1f -> Low: %.1f)", high_val, curr_val)
                time.sleep(0.5) # Allow settling
                return True

            sys.stdout.write(f"\r    Waiting for drop... Curr: {curr_val:.0f} "
                             f"(Threshold: {drop_threshold:.0f})   ")
            sys.stdout.flush()
            time.sleep(0.1)

        logging.warning("Timeout waiting for LED off! Noise floor might be inaccurate.")
        return False

    def run(self):
        """Run the monitor loop."""
        # pylint: disable=too-many-locals, too-many-statements, too-many-branches
        self.cam.start()
        if self.preview:
            self.aim_camera()

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
        avg_bg_brightness = np.mean(bg_brightness_samples)

        # DYNAMIC THRESHOLD using actual measured ON level
        if self.detected_on_brightness > 10:
            estimated_on_level = self.detected_on_brightness
        else:
            # Fallback if we didn't measure it properly
            estimated_on_level = avg_noise_level + self.detected_peak_strength

        # Set threshold halfway between OFF and ON
        self.thresh_bright = (avg_noise_level + estimated_on_level) / 2.0

        metric_name = "Contrast" if self.use_contrast else "Brightness"
        logging.info("Noise (%s): %.1f | Est. ON: %.1f", metric_name, avg_noise_level, estimated_on_level)
        logging.info("Set Detection Threshold: %.1f", self.thresh_bright)

        # Saturation check
        if self.log_saturation and avg_bg_brightness > 240:
            logging.warning("[SATURATION] Background is %.1f (near clipping!)",
                            avg_bg_brightness)
            logging.warning("Consider: 1) ND filter, 2) Reduce LED brightness, "
                            "3) Increase distance, 4) Enable --adaptive-exposure")

        # 4. Monitor
        logging.info("--- MONITORING STARTED ---")
        last_pulse = time.time()

        # Stats aggregation
        report_period = 1.0
        last_report = time.time()
        int_max = 0
        int_min = 255
        t_max = last_report
        t_min = last_report

        while True:
            frame = self.cam.get_frame()
            roi = frame[y1:y2, x1:x2]
            val = self._measure_roi(roi)
            now = time.time()

            # Saturation tracking
            if self.log_saturation:
                self.total_frames += 1
                bg_brightness = np.median(roi)
                if bg_brightness > 250:  # Nearly saturated
                    self.saturation_count += 1

            # Update Interval Stats
            if val > int_max:
                int_max = val
                t_max = now
            if val < int_min:
                int_min = val
                t_min = now

            is_active = val > self.thresh_bright
            gap = now - last_pulse

            if is_active:
                # If we see light, reset timer
                last_pulse = now
                if gap > 0.5: # Reduced from 2.0s to catch faster pulses
                    sys.stdout.write(f"\033[96m[PULSE DETECTED] Gap: {gap:.1f}s \033[0m\n")

            # Report every second
            if now - last_report > report_period:
                limit = self.interval * 1.2
                status = f"[ALARM] ({gap:.1f}s)" if gap > limit else f"[WAITING] {gap:.1f}s"
                color = "\033[91m" if gap > limit else "\033[92m"

                dt = abs(t_max - t_min)
                metric_label = "Contrast" if self.use_contrast else "Bright"
                status_line = (f"{color}{status} | {metric_label}: {val:.0f} / {self.thresh_bright:.0f} | "
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
Feature Flags (all enabled by default except adaptive-exposure):
  --use-contrast        Use contrast (max-median) instead of brightness
  --adaptive-roi        Automatically size ROI based on LED blob size
  --adaptive-off        Use variance-based OFF detection
  --log-saturation      Track and log saturation events
  --adaptive-exposure   Automatically adjust camera exposure (experimental)

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

    parser.add_argument("--log-saturation", dest="log_saturation", action="store_true", default=True,
                       help="Track saturation events (default)")
    parser.add_argument("--no-log-saturation", dest="log_saturation", action="store_false",
                       help="Disable saturation logging")

    parser.add_argument("--adaptive-exposure", dest="adaptive_exposure", action="store_true", default=False,
                       help="Auto-adjust camera exposure (experimental, X86 only)")

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

    monitor = PeakMonitor(
        args.interval,
        args.threshold,
        args.preview,
        use_contrast=args.use_contrast,
        adaptive_roi=args.adaptive_roi,
        adaptive_off=args.adaptive_off,
        log_saturation=args.log_saturation,
        adaptive_exposure=args.adaptive_exposure
    )

    def cleanup(_s, _f):
        """Signal handler for cleanup."""
        monitor.cam.stop()
        # pylint: disable=no-member
        cv2.destroyAllWindows()
        sys.exit(0)

    signal.signal(signal.SIGINT, cleanup)
    monitor.run()
