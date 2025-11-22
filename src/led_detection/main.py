"""
Main module for LED detection.
"""

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

    def lock_settings(self):
        """Lock camera settings."""

class RPiCameraDriver(CameraDriver):
    """Driver for Raspberry Pi Camera."""
    def __init__(self, w=640, h=480):
        try:
            # pylint: disable=import-outside-toplevel
            from picamera2 import Picamera2
            self.p = Picamera2()
            self.h, self.w = h, w
            config = self.p.create_configuration(main={"format": "YUV420", "size": (w, h)})
            self.p.configure(config)
        except ImportError as exc:
            raise RuntimeError("Picamera2 missing.") from exc

    def start(self):
        self.p.start()
        self.lock_settings()
        time.sleep(1)

    def lock_settings(self):
        self.p.set_controls({
            "AeEnable": False,
            "AnalogueGain": 8.0,
            "AwbEnable": False,
            "ExposureTime": 8333,
            "FrameDurationLimits": (16666, 16666)
        })

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
        if SYS_OS == "Windows":
            self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)
        else:
            self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 3)

    def lock_settings(self):
        logging.info("Locking Camera Settings (Manual)...")
        if SYS_OS == "Linux":
            self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)
            self.cap.set(cv2.CAP_PROP_EXPOSURE, 83)
        else:
            self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0)
            self.cap.set(cv2.CAP_PROP_EXPOSURE, -5)
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

# --- MONITOR LOGIC ---

class ProductionMonitor:
    """Monitor for LED detection."""
    def __init__(self, interval, threshold):
        self.cam = get_driver()
        self.interval = interval
        self.roi = None
        self.thresh_contrast = 0.0
        self.min_search_score = threshold
        self.detected_flash_strength = 0.0

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

    def flush_camera_buffer(self):
        """Flush camera buffer to stabilize sensor."""
        logging.info("Stabilizing sensor (flushing buffer)...")
        for _ in range(30):
            self.cam.get_frame()
            time.sleep(0.03)

    def wait_for_signal(self):
        """Wait for the LED signal."""
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
                diff = cv2.absdiff(frame, last_frame).astype(np.float32)
                np.maximum(accum_max_diff, diff, out=accum_max_diff)
                _, max_val, _, max_loc = cv2.minMaxLoc(accum_max_diff)

                if max_val > self.min_search_score:
                    self.detected_flash_strength = max_val
                    logging.info("SIGNAL DETECTED! Score: %.0f", max_val)
                    self.lock_roi(max_loc, frame.shape)
                    return True

            last_frame = frame
            sys.stdout.write(f"\r    Scanning... {time.time()-start_time:.1f}s | "
                             f"Peak Change: {cv2.minMaxLoc(accum_max_diff)[1]:.0f}   ")
            sys.stdout.flush()

            if isinstance(self.cam, X86CameraDriver):
                cv2.imshow("Search", frame)
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
            cv2.destroyWindow("Search")

    def wait_for_led_off(self):
        """Blocks until the signal drops back down to near zero."""
        logging.info("Waiting for LED to turn OFF (drop in contrast)...")
        y1, y2, x1, x2 = self.roi

        # We expect the signal to drop below 25% of what we just detected
        target_level = max(10.0, self.detected_flash_strength * 0.25)
        logging.info("Detected Strength: %.0f -> Waiting for drop below: %.0f",
                     self.detected_flash_strength, target_level)

        start = time.time()
        while time.time() - start < 30.0: # 30s timeout
            f = self.cam.get_frame()
            roi = f[y1:y2, x1:x2]

            # Metric: Contrast
            curr_contrast = float(np.max(roi)) - float(np.median(roi))

            if curr_contrast < target_level:
                logging.info("LED OFF Confirmed. Current Contrast: %.1f", curr_contrast)
                time.sleep(0.5) # Extra settlement time
                return True

            sys.stdout.write(f"\r    Waiting... Contrast: {curr_contrast:.0f} "
                             f"(Target < {target_level:.0f})   ")
            sys.stdout.flush()
            time.sleep(0.1)

        logging.warning("Timeout waiting for LED off! Calibration might be inaccurate.")
        return False

    def calibrate_thresholds(self):
        """Calibrate contrast thresholds."""
        logging.info("Calibrating contrast noise floor...")
        y1, y2, x1, x2 = self.roi
        contrasts = []

        for _ in range(30):
            f = self.cam.get_frame()
            roi = f[y1:y2, x1:x2]
            val = float(np.max(roi)) - float(np.median(roi))
            contrasts.append(val)

        avg_noise_contrast = np.mean(contrasts)

        # THRESHOLD FORMULA:
        # Base Noise + (Half of the Signal Strength)
        # This ensures we are exactly in the middle of "OFF" and "ON".
        margin = max(15.0, self.detected_flash_strength * 0.5)
        self.thresh_contrast = avg_noise_contrast + margin

        logging.info("Noise: %.1f | Detected Flash: %.1f",
                     avg_noise_contrast, self.detected_flash_strength)
        logging.info("Calculated Threshold: %.1f", self.thresh_contrast)

    def run(self):
        """Run the monitor loop."""
        # pylint: disable=too-many-locals
        self.cam.start()
        self.aim_camera()
        self.cam.lock_settings()
        self.flush_camera_buffer()

        while True:
            # 1. Search
            print("\n[System] Waiting for device activity...")
            while True:
                if self.wait_for_signal():
                    break
                print("\n[Retry] No signal seen. Clearing buffer and rescanning...\n")

            # 2. Wait for LED OFF (Crucial Fix)
            self.wait_for_led_off()

            # 3. Calibrate
            self.calibrate_thresholds()

            # 4. Monitor
            logging.info("--- MONITORING STARTED ---")
            last_pulse = time.time()
            y1, y2, x1, x2 = self.roi

            while True:
                frame = self.cam.get_frame()
                roi = frame[y1:y2, x1:x2]

                bg_level = np.median(roi)
                peak_level = np.max(roi)
                val_contrast = peak_level - bg_level

                is_active = val_contrast > self.thresh_contrast

                now = time.time()
                gap = now - last_pulse

                if is_active:
                    last_pulse = now
                    if gap > 2.0:
                        sys.stdout.write(f"\r\033[96m[PULSE DETECTED] Gap: {gap:.1f}s "
                                         "\033[0m                                     \n")

                limit = self.interval * 1.2
                status = f"[ALARM] ({gap:.1f}s)" if gap > limit else f"[WAITING] {gap:.1f}s"
                color = "\033[91m" if gap > limit else "\033[92m"

                warning = " [SAT]" if bg_level > 240 else ""
                sys.stdout.write(f"\r{color}{status} | Contrast: {val_contrast:.0f} "
                                 f"(Th:{self.thresh_contrast:.0f}) | "
                                 f"Bg: {bg_level:.0f}{warning}\033[0m   ")
                sys.stdout.flush()

                if isinstance(self.cam, X86CameraDriver):
                    debug = frame.copy()
                    cv2.rectangle(debug, (x1, y1), (x2, y2), (255, 255, 255), 1)
                    cv2.putText(debug, f"C:{val_contrast:.0f}", (x1, y1-5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                (0, 255, 0) if is_active else (0, 0, 255), 2)
                    cv2.imshow("Monitor", debug)
                    if cv2.waitKey(1) == ord('q'):
                        break

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--interval", type=float, default=60.0)
    parser.add_argument("-t", "--threshold", type=float, default=40.0)
    parser.add_argument("-d", "--debug", action="store_true")
    args = parser.parse_args()

    setup_logging(args.debug)
    monitor = ProductionMonitor(args.interval, args.threshold)

    def cleanup(_s, _f):
        """Signal handler for cleanup."""
        monitor.cam.stop()
        # pylint: disable=no-member
        cv2.destroyAllWindows()
        sys.exit(0)

    signal.signal(signal.SIGINT, cleanup)
    monitor.run()
