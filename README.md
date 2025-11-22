# LED Detection

A best-in-class Python project for LED detection.

## Development

### Setup

1.  Create a virtual environment:
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```

2.  Install dependencies:
    ```bash
    pip install -e .[dev]
    ```

3.  Install pre-commit hooks:
    ```bash
    pre-commit install
    ```

### Testing

Run tests with pytest:

```bash
pytest
```
Here is a complete `README.md` documentation for the final solution we developed.

***

# Robust LED Heartbeat Monitor

A computer vision system designed to detect periodic LED flashes ("heartbeats") on embedded devices. It uses a **Contrast-Based Algorithm** to remain robust against ambient light changes, flashlights, and reflections, while running efficiently on both **Raspberry Pi 5** and **x86 Desktops**.

## 🚀 Features

*   **Hybrid Hardware Driver:** Automatically detects and loads the correct driver for **Raspberry Pi 5** (`picamera2` / `libcamera`) or **Standard USB Webcams** (OpenCV `v4l2`/`DSHOW`).
*   **Contrast-Based Detection:** Uses a `Peak - Median` metric instead of raw brightness. This allows detection of an LED even if a flashlight is shined on the device (Global Illumination Rejection).
*   **Auto-Calibration:**
    *   **Search Phase:** Uses "Peak Hold" logic to find the single pixel changing the most over time.
    *   **Smart Thresholding:** Dynamically sets triggers based on the specific signal strength of the detected LED ($Threshold = Noise + 0.5 \times Signal$).
    *   **Active OFF Detection:** Intelligently waits for the LED to turn off before measuring the noise floor, preventing false calibrations.
*   **Exposure Locking:** Forces the camera into Manual Exposure mode to prevent "breathing" (gain hunting) which causes false positives.

## 🛠️ Hardware Requirements

1.  **Camera:**
    *   **Raspberry Pi:** Module 3, HQ Camera, or Global Shutter (via CSI).
    *   **PC/Laptop:** Any standard USB Webcam (Logitech C270, ELP, etc).
2.  **Compute:**
    *   Raspberry Pi 5 (Recommended for embedded use).
    *   Any x86 Linux or Windows machine.

## 📦 Installation

### 1. Raspberry Pi 5 (Bookworm)
The Pi uses a system-managed Python environment. It is best to use the system packages.

```bash
# Update system
sudo apt update
sudo apt upgrade

# Install dependencies
sudo apt install python3-opencv python3-numpy python3-libcamera
```

### 2. x86 Linux (Ubuntu 24.04) / Windows
It is recommended to use a virtual environment to avoid system package conflicts.

```bash
# Create virtual env
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install opencv-python numpy
```

## 🏃 Usage

Save the final script as `led_monitor.py`.

### Basic Start
Run the monitor expecting a flash at least once every 60 seconds.

```bash
python3 led_monitor.py --interval 60
```

### Advanced Options
```bash
python3 led_monitor.py --interval 60 --threshold 40 --debug
```

*   `--interval`: The expected max time (seconds) between heartbeats. The system alarms if this is exceeded.
*   `--threshold`: The minimum raw signal score (0-255) required to initially lock onto the LED. Lower this for dim LEDs.
*   `--debug`: Shows the video feed with bounding boxes and real-time metrics.

## 🧠 How It Works (The Workflow)

The system operates in a strict state machine to ensure reliability:

### 1. Aiming Phase
The camera starts in **Auto-Exposure** mode for 5 seconds. A preview window appears (on Desktop) allowing you to position the camera.

### 2. Locking Phase
The system switches to **Manual Exposure**.
*   **Exposure Time:** ~8.3ms (1/120s). This is the "sweet spot" to reject 60Hz mains flicker while keeping the LED distinct.
*   **Buffer Flush:** The system discards frames for ~1 second to allow the sensor levels to stabilize.

### 3. Search Phase (Peak Hold)
The system stares at the scene. It calculates the `absdiff` (Absolute Difference) between frames. It maintains a **Peak History** buffer.
*   *Why?* If the LED flashes for only 10ms, a simple average would miss it. Peak Hold ensures we catch the transient event.
*   Once a pixel exceeds the `--threshold`, an **ROI (Region of Interest)** is locked around that coordinate.

### 4. Settlement Phase (Active OFF)
Before calibrating, the system must ensure the LED is **OFF**.
*   It watches the contrast in the ROI.
*   It waits for the signal to drop to <25% of the detected peak.
*   This prevents the "Calibration while ON" bug.

### 5. Calibration Phase
The system measures the **Local Contrast Noise Floor** inside the ROI.
*   **Metric:** `Contrast = Max_Pixel - Median_Pixel`.
*   **Threshold Calculation:** `Trigger = Noise_Floor + (Detected_Signal_Strength * 0.5)`.
*   This places the trigger line exactly halfway between the "OFF" state and the "ON" state.

### 6. Monitor Phase
The system runs the detection loop:
*   Calculates `Contrast` every frame.
*   If `Contrast > Trigger`, a pulse is recorded.
*   If `Time_Since_Pulse > Interval`, an **[ALARM]** is triggered.
*   If background brightness > 240, a **[SATURATION WARN]** is shown (signal clipping risk).

## ❓ Troubleshooting

**Q: The system scans forever but never finds the LED.**
*   **A:** Your LED might be too dim for the default threshold. Try running with `python3 led_monitor.py --threshold 20`.
*   **A:** Ensure the camera is focused. A blurry blob is harder to detect than a sharp point.

**Q: It detects a signal immediately at 0.0s, then fails.**
*   **A:** This usually happens if the camera is still adjusting exposure. The code includes a buffer flush to prevent this, but cheap webcams may need longer.

**Q: It works, but fails when I shine a flashlight.**
*   **A:** Check the console for `[SAT]`. If the background level hits 255 (Pure White), the sensor is saturated. Contrast cannot be calculated on a white pixel. You must physically reduce the light or modify the code to use a lower exposure time (`set_exposure_us`).

**Q: "QSocketNotifier: Can only be used with threads..." warning.**
*   **A:** This is a harmless Linux Wayland warning. The script automatically sets `QT_QPA_PLATFORM=xcb` to suppress it, but it may persist on some systems. It does not affect detection logic.
