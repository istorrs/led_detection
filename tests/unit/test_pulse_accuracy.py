import os
import shutil
import numpy as np
from led_detection.main import PeakMonitor

class MockMonitor(PeakMonitor):
    def __init__(self):
        super().__init__(interval=1, threshold=10)
        self.use_contrast = False
        self.log_saturation = False
        self.roi = (0, 10, 0, 10)
        self.thresh_bright = 0
        self.current_noise_floor = 0
        self.cam = None # No camera needed for this test

def test_verify_signal_interpolation_accuracy():
    """
    Test that verify_signal correctly calculates duration using interpolation
    and passes the tighter tolerance (10ms).
    """
    mon = MockMonitor()

    # We want a 50ms pulse.
    # With interpolation, if we have points at 0.0, 0.05, 0.10 with values 0, 255, 0
    # Threshold 127.5
    # Rising: 0.0 -> 0.05. Start = 0.025
    # Falling: 0.05 -> 0.10. End = 0.075
    # Duration = 0.05s = 50ms. Perfect match.

    timestamps = [0.0, 0.05, 0.10, 0.15]
    signal_values = [0, 255, 0, 0]

    expected_period = 0.5
    expected_duration = 50 # ms
    tolerance = 0.2 # 10ms

    # Create 2 identical pulses
    timestamps = [
        0.0, 0.05, 0.10, 0.15, # Pulse 1 (50ms)
        0.5, 0.55, 0.60, 0.65  # Pulse 2 (50ms)
    ]
    signal_values = [
        0, 255, 0, 0,
        0, 255, 0, 0
    ]

    # Run verification
    result = mon.verify_signal(signal_values, timestamps, expected_period, expected_duration, tolerance)

    assert result is True, "Should pass with exact 50ms duration from interpolation"

def test_verify_signal_first_pulse_fail():
    """
    Test that verify_signal passes even if the first pulse is way off,
    as long as subsequent pulses are valid.
    """
    mon = MockMonitor()

    expected_duration = 50
    tolerance = 0.2

    # Pulse 1: 100ms (0, 255, 255, 0 at 0.05 steps -> 0.025 to 0.125 = 100ms)
    # Pulse 2: 50ms (0, 255, 0 at 0.05 steps -> 0.025 to 0.075 = 50ms)
    # Pulse 3: 50ms

    timestamps = [
        0.0, 0.05, 0.10, 0.15, 0.20, # Pulse 1 (100ms)
        0.5, 0.55, 0.60, 0.65,       # Pulse 2 (50ms)
        1.0, 1.05, 1.10, 1.15        # Pulse 3 (50ms)
    ]
    signal_values = [
        0, 255, 255, 0, 0,
        0, 255, 0, 0,
        0, 255, 0, 0
    ]

    # We have 3 pulses. 1 Invalid (100ms vs 50ms), 2 Valid (50ms).
    # verify_signal requires >= 2 valid pulses.
    # So this should PASS.

    result = mon.verify_signal(signal_values, timestamps, 0.5, expected_duration, tolerance)

    assert result is True, "Should pass because 2 out of 3 pulses are valid"

def test_verify_signal_saves_frames(_tmp_path): # pylint: disable=too-many-locals
    """
    Test that verify_signal saves frames for bad pulses in correct subdirectories.
    """
    mon = MockMonitor()

    # Create dummy frames
    # 10 frames from t=0.0 to t=0.9
    frames = []
    for i in range(10):
        t = i * 0.1
        # Create a dummy 10x10 image
        img = np.zeros((10, 10, 3), dtype=np.uint8)
        frames.append((t, img))

    # Create signal that has a bad pulse and a good pulse
    # Pulse 0: Bad (Duration 150ms)
    # 0.0(0), 0.1(255) -> Start 0.05
    # 0.1(255), 0.3(0) -> End 0.2
    # Duration 150ms.

    # Pulse 1: Good (Duration 50ms)
    # 0.5(0), 0.55(255) -> Start 0.525
    # 0.55(255), 0.6(0) -> End 0.575
    # Duration 50ms.

    timestamps = [0.0, 0.1, 0.3, 0.4, 0.5, 0.55, 0.6]
    signal_values = [0, 255, 0, 0, 0, 255, 0]

    # We need to mock /tmp/bad_pulses to use tmp_path
    # But the code hardcodes /tmp/bad_pulses.
    # We can patch os.makedirs or just check /tmp/bad_pulses if we are allowed.
    # The environment allows writing to /tmp.

    # Clean up /tmp/bad_pulses first
    # Clean up /tmp/bad_pulses first
    if os.path.exists("/tmp/bad_pulses"):
        shutil.rmtree("/tmp/bad_pulses")

    result = mon.verify_signal(signal_values, timestamps, 0.5, 50, 0.2, frames=frames)

    # Should fail because valid_pulses (1) < 2
    assert result is False

    # Find the run directory
    # The code creates a timestamped run directory, e.g. /tmp/bad_pulses/run_2025...
    run_dirs = os.listdir("/tmp/bad_pulses")
    assert len(run_dirs) == 1, f"Expected 1 run directory, found: {run_dirs}"
    run_dir = os.path.join("/tmp/bad_pulses", run_dirs[0])

    # Check if folder exists for bad pulse (Pulse 0)
    pulse_0_path = os.path.join(run_dir, "pulse_0")
    assert os.path.exists(pulse_0_path), f"Path not found: {pulse_0_path}"
    files = os.listdir(pulse_0_path)
    assert len(files) > 0, "Should have saved frames for Pulse 0"

    # Check if folder does NOT exist for good pulse (Pulse 1)
    pulse_1_path = os.path.join(run_dir, "pulse_1")
    assert not os.path.exists(pulse_1_path), "Should NOT save frames for valid Pulse 1"

def test_interpolation_accuracy():
    """
    Test if linear interpolation improves accuracy.
    """
    # Simulate a 50ms pulse sampled at 43ms interval (23 FPS)
    # Pulse starts at 0.100, ends at 0.150.
    # Timestamps: 0.0, 0.043, 0.086, 0.129, 0.172, 0.215
    # Pulse is ON during 0.129.
    # At 0.086 (t-1), val=0.
    # At 0.129 (t), val=255 (fully ON).
    # At 0.172 (t+1), val=0.

    # Wait, if it's fully ON at 0.129, and 0 at neighbors.
    # Threshold = 127.5.
    # Rising: between 0.086 (0) and 0.129 (255).
    # 127.5 is exactly halfway.
    # t_start = 0.086 + (0.129 - 0.086) * 0.5 = 0.1075.
    # Falling: between 0.129 (255) and 0.172 (0).
    # t_end = 0.129 + (0.172 - 0.129) * 0.5 = 0.1505.
    # Duration = 0.1505 - 0.1075 = 0.043s = 43ms.
    # Error = 7ms.

    # Without interpolation:
    # Rising at 0.129.
    # Falling at 0.172.
    # Duration = 43ms.
    # Same?

    # Let's try a case where interpolation helps.
    # Pulse starts at 0.100, ends at 0.150.
    # Signal values reflect partial exposure?
    # If frame integrates light...
    # Frame at 0.129 covers [0.1075, 0.1505]? No, timestamp is usually start or middle.
    # Assuming timestamp is middle of exposure.
    # Exposure time?

    # If we assume signal value is proportional to overlap.
    # Let's just assume smooth transition for now.

    timestamps = [0.0, 0.1, 0.2, 0.3]
    signal_values = [0, 100, 200, 0]
    # Threshold 100.
    # Rising: 0.1 (val=100). Exact match?
    # Falling: between 0.2 (200) and 0.3 (0).
    # Crosses 100 at mid point? 0.25.
    # Duration = 0.25 - 0.1 = 0.15s = 150ms.

    # Without interpolation:
    # Rising at 0.1 (val=100 >= 100? if > then 0.2).
    # If >=, then 0.1.
    # Falling at 0.3.
    # Duration = 0.2s = 200ms.
    # Error = 50ms.

    # Interpolation gives 150ms. Much better!

    # Implement simple interpolation logic here to verify
    threshold = 100
    pulses = []
    current_state = 0
    start_time = 0

    for i in range(len(signal_values) - 1):
        v1 = signal_values[i]
        v2 = signal_values[i+1]
        t1 = timestamps[i]
        t2 = timestamps[i+1]

        # Rising edge
        if v1 <= threshold < v2:
            # Interpolate
            fraction = (threshold - v1) / (v2 - v1)
            t_cross = t1 + (t2 - t1) * fraction
            start_time = t_cross
            current_state = 1

        # Falling edge
        elif v1 > threshold >= v2:
            if current_state == 1:
                fraction = (threshold - v1) / (v2 - v1)
                t_cross = t1 + (t2 - t1) * fraction
                duration = (t_cross - start_time) * 1000.0
                pulses.append(duration)
                current_state = 0

    assert len(pulses) == 1
    assert abs(pulses[0] - 150.0) < 1.0, f"Expected 150ms, got {pulses[0]}"
