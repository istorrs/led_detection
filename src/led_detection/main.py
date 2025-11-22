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
    def __init__(self, interval, threshold, preview=False):
        self.cam = get_driver()
        self.interval = interval
        self.preview = preview
        self.roi = None
        self.thresh_bright = 0.0
        self.min_signal_strength = threshold
        self.detected_peak_strength = 0.0
        self.detected_on_brightness = 0.0  # Actual brightness when ON

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
                    self.lock_roi(max_loc, frame.shape)
                    # Store actual brightness (not change score) for threshold calculation
                    self.detected_on_brightness = np.percentile(
                        frame[self.roi[0]:self.roi[1], self.roi[2]:self.roi[3]], 90)
                    logging.info("[Debug] Initial ROI brightness (P90): %.1f",
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

    def lock_roi(self, loc, shape):
        """Lock Region of Interest."""
        c_x, c_y = loc
        h, w = shape
        size = 32
        x1, x2 = max(0, c_x - size), min(w, c_x + size)
        y1, y2 = max(0, c_y - size), min(h, c_y + size)
        self.roi = (y1, y2, x1, x2)
        logging.info("ROI Locked: %s", self.roi)
        if isinstance(self.cam, X86CameraDriver):
            cv2.destroyWindow("Calibration")

    def wait_for_led_off(self):
        """ Monitors ROI brightness until it drops significanty """
        logging.info("Waiting for LED to turn OFF...")
        y1, y2, x1, x2 = self.roi

        # Measure "High" State using percentile to ignore saturated center
        f = self.cam.get_frame()
        high_val = np.percentile(f[y1:y2, x1:x2], 90)

        # If we are already saturated or very bright, we assume ON.
        # But if detection was weak, high_val might be low.
        # We enforce a hard drop or a timeout.

        start = time.time()
        while time.time() - start < 15.0: # 15s Timeout
            f = self.cam.get_frame()
            curr_val = np.percentile(f[y1:y2, x1:x2], 90)

            # Condition 1: Significant drop relative to high
            drop_cond = curr_val < (high_val * 0.6)

            # Condition 2: Absolute dark (safe noise floor)
            # If we are below 100 (out of 255), it's likely OFF enough for calibration
            abs_cond = curr_val < 100

            if drop_cond or abs_cond:
                logging.info("LED OFF Confirmed (High: %s -> Low: %s)", high_val, curr_val)
                time.sleep(0.5) # Allow settling
                return True

            sys.stdout.write(f"\r    Waiting for drop... Curr: {curr_val:.0f} "
                             f"(High: {high_val:.0f})   ")
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

        # 3. Calibrate Noise Floor (OFF State)
        logging.info("Calibrating OFF-state noise floor...")
        y1, y2, x1, x2 = self.roi
        brightness_samples = []
        for _ in range(30):
            f = self.cam.get_frame()
            # Use 90th percentile instead of max to handle saturation
            brightness_samples.append(np.percentile(f[y1:y2, x1:x2], 90))

        avg_noise_brightness = np.mean(brightness_samples)

        # DYNAMIC THRESHOLD using actual measured ON brightness
        # Use the actual ON brightness we measured, not the change score
        if self.detected_on_brightness > 10:
            estimated_on_level = self.detected_on_brightness
        else:
            # Fallback if we didn't measure it properly
            estimated_on_level = avg_noise_brightness + self.detected_peak_strength

        # Set threshold halfway between OFF and ON
        self.thresh_bright = (avg_noise_brightness + estimated_on_level) / 2.0

        logging.info("Noise (P90): %.1f | Est. ON: %.1f", avg_noise_brightness, estimated_on_level)
        logging.info("Set Brightness Threshold: %.1f", self.thresh_bright)

        if avg_noise_brightness > 240:
            logging.warning("[SATURATION] Noise floor is %.1f (too bright!)",
                            avg_noise_brightness)
            logging.warning("Consider: 1) ND filter, 2) Reduce LED, "
                            "3) Increase distance")

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
            # Use 90th percentile instead of max to handle partial saturation
            val_bright = np.percentile(roi, 90)
            now = time.time()

            # Update Interval Stats
            if val_bright > int_max:
                int_max = val_bright
                t_max = now
            if val_bright < int_min:
                int_min = val_bright
                t_min = now

            is_active = val_bright > self.thresh_bright
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
                print(f"{color}{status} | Bright: {val_bright:.0f} / {self.thresh_bright:.0f} | "
                      f"Min/Max: {int_min:.0f}/{int_max:.0f} | dT: {dt:.3f}s\033[0m")

                # Reset stats
                last_report = now
                int_max = 0
                int_min = 255

            if isinstance(self.cam, X86CameraDriver):
                debug = frame.copy()
                cv2.rectangle(debug, (x1, y1), (x2, y2), (255, 255, 255), 1)
                cv2.putText(debug, f"{val_bright:.0f}", (x1, y1-5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            (0, 255, 0) if is_active else (0, 0, 255), 2)
                cv2.imshow("Monitor", debug)
                if cv2.waitKey(1) == ord('q'):
                    break

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--interval", type=float, default=60.0)
    parser.add_argument("-t", "--threshold", type=float, default=50.0)
    parser.add_argument("-d", "--debug", action="store_true")
    parser.add_argument("-p", "--preview", action="store_true", help="Enable aiming phase")
    args = parser.parse_args()

    setup_logging(args.debug)
    monitor = PeakMonitor(args.interval, args.threshold, args.preview)

    def cleanup(_s, _f):
        """Signal handler for cleanup."""
        monitor.cam.stop()
        # pylint: disable=no-member
        cv2.destroyAllWindows()
        sys.exit(0)

    signal.signal(signal.SIGINT, cleanup)
    monitor.run()
