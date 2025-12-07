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

def test_verify_signal_false_positive():
    """
    Test that verify_signal passes even with bad pulses if they are not filtered out correctly.
    Based on the user's log:
    Pulses: ['86.0ms', '49.9ms', '67.4ms', '66.1ms', '64.4ms', '50.0ms', '50.6ms', '32.1ms', '50.1ms', '50.4ms']
    """
    mon = MockMonitor()

    expected_period = 0.5
    expected_duration = 50
    tolerance = 0.2

    # We will simulate the pulses by providing timestamps and signal values that produce these durations.
    # To simplify, we can mock the internal pulses list if we could, but verify_signal computes it.
    # So we must construct signal_values and timestamps.

    # Pulse 0: 86ms (Invalid)
    # Pulse 1: 49.9ms (Valid)
    # Pulse 2: 67.4ms (Invalid)
    # Pulse 3: 66.1ms (Invalid)
    # Pulse 4: 64.4ms (Invalid? Diff 14.4 > 10. Should fail.)
    # Pulse 5: 50.0ms (Valid)
    # ...

    # Let's focus on why Pulse 4 passed.
    # And why the period check passed for Pulse 4 (relative to Pulse 3).

    # Construct a signal with:
    # Pulse A: 66.1ms (Invalid)
    # Gap
    # Pulse B: 64.4ms (Invalid)

    # If Pulse B passes, verify_signal returns True (assuming we have another valid pulse somewhere).

    # Let's make Pulse B have 64.4ms duration.
    # Start: 1.0
    # End: 1.0644

    timestamps = [0.0, 0.05, 0.10, 1.0, 1.0644, 1.10]
    signal_values = [0, 255, 0, 0, 255, 0]
    # Pulse 1: 50ms (Valid)
    # Pulse 2: ~35ms? No, let's use exact points for interpolation.

    # We want to reproduce the exact scenario.
    # But constructing the signal is tedious.
    # Instead, let's subclass and override the pulse detection part? No, too complex.

    # Let's just create a test that feeds a signal producing a 64.4ms pulse.
    # 0 -> 255 at 1.0 -> 1.05. Start = 1.025.
    # 255 -> 0 at 1.0894 -> 1.1394. End = 1.0894 + 0.025 = 1.1144?
    # Duration = 1.1144 - 1.025 = 0.0894 = 89.4ms.

    # Let's just use the logic directly.
    # If we have a pulse of 64.4ms.
    # diff = 14.4.
    # 14.4 > 10.0 (tolerance) AND 14.4 > 10.0 (hardcoded).
    # It MUST fail.

    # Unless... expected_duration is not 50?
    # User log says (Exp: 50ms).

    # Unless... tolerance is not 0.2?
    # User calls run_one_shot with default tolerance=0.2.

    # Maybe floating point precision?
    # 14.4 is definitely > 10.0.

    # Let's verify the logic with a unit test.

    timestamps = [0.0, 0.05, 0.10, 0.15] # 50ms pulse
    signal_values = [0, 255, 0, 0]

    # Add a 64.4ms pulse.
    # Start at 0.5.
    # End at 0.5644.
    # We need points around start and end.
    # Start: 0.48 (0), 0.52 (255). Mid = 0.5.
    # End: 0.5644. We need mid point to be 0.5644.
    # 0.54 (255), 0.5888 (0). Mid = 0.5644.

    timestamps += [0.48, 0.52, 0.54, 0.5888]
    signal_values += [0, 255, 255, 0]

    # Pulse 1: 50ms.
    # Pulse 2: 64.4ms.

    # Run verify_signal.
    # We expect Pulse 2 to trigger a warning.
    # And valid_pulses to be 1.
    # So verify_signal should return False.

    result = mon.verify_signal(signal_values, timestamps, expected_period, expected_duration, tolerance)

    assert result is False, "Should fail because only 1 valid pulse (Pulse 2 should be invalid)"
