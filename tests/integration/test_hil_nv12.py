"""Hardware-in-the-loop tests for aedcamera_nv12.cam.

Programs an ESP32 LED pulse generator to simulate each AED type's blink
pattern, SSHes to the camera to run detection, then collects logs and
JPG artifacts back to the workstation.
"""

import json
import logging
import os
import time

import pytest  # pylint: disable=import-error

logger = logging.getLogger(__name__)

FPS = 30  # Camera capture rate


def _compute_pulse_params(aed_params):
    """Compute LED pulse duration_ms and period_ms from AED parameters.

    Returns:
        (duration_ms, period_ms, capture_period_s)
    """
    blink_frames_min = aed_params["blink_frames_min"]
    blink_frames_max = aed_params["blink_frames_max"]
    blink_period_s = aed_params["blink_period_seconds"]

    # Average frame count, converted to ms
    avg_frames = (blink_frames_min + blink_frames_max) / 2.0
    duration_ms = avg_frames * (1000 / FPS)

    period_ms = blink_period_s * 1000

    # Capture long enough for at least 2 full blink cycles plus overhead
    # Matches firmware: period*2.2 + margin, minimum 12s
    capture_period_s = max(12, int(2 + period_ms * 2.2 / 1000 + 1))

    return duration_ms, period_ms, capture_period_s


def _save_log(test_dir, result):
    """Save camera stdout/stderr to log.txt."""
    with open(os.path.join(test_dir, "log.txt"), "w", encoding="utf-8") as f:
        f.write("=== STDOUT ===\n")
        f.write(result.stdout)
        f.write("\n=== STDERR ===\n")
        f.write(result.stderr)


def _save_result_json(test_dir, aed_type_id, aed_name, brightness_pct,
                      duration_ms, period_ms, capture_period_s, result, artifacts):
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    """Save structured test result to result.json."""
    data = {
        "aed_type_id": aed_type_id,
        "aed_name": aed_name,
        "brightness_pct": brightness_pct,
        "pulse_duration_ms": duration_ms,
        "pulse_period_ms": period_ms,
        "capture_period_s": capture_period_s,
        "exit_code": result.exit_code,
        "detected": result.detected,
        "detection_code": result.detection_code,
        "total_time_ms": result.total_time_ms,
        "frames_processed": result.frames_processed,
        "frames_detected": result.frames_detected,
        "artifacts": list(artifacts.keys()),
    }
    with open(os.path.join(test_dir, "result.json"), "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


@pytest.mark.integration
class TestHILNv12:
    """Hardware-in-the-loop detection tests across AED types and brightness."""

    # Collect all results for summary generation
    _results = []

    def test_detection(
        self,
        camera_runner,
        led_controller,
        result_dir,
        aed_type_id,
        aed_name,
        brightness_pct,
        aed_params,
    ):
        # pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
        """Test detection for a specific AED type at a given brightness."""
        duration_ms, period_ms, capture_period_s = _compute_pulse_params(aed_params)
        test_label = f"{aed_name}-{brightness_pct}pct"
        test_dir = os.path.join(result_dir, test_label)
        os.makedirs(test_dir, exist_ok=True)

        logger.info(
            "--- %s: duration=%.0fms, period=%.0fms, capture=%ds ---",
            test_label, duration_ms, period_ms, capture_period_s,
        )

        # 1. Program the LED pulse generator
        ok = led_controller.set_pulse(duration_ms, period_ms, brightness_pct)
        assert ok, f"Failed to set LED pulse for {test_label}"

        # 2. Let the pulse stabilize
        time.sleep(2.0)

        # 3. Clean leftover artifacts on the camera
        camera_runner.clean_artifacts()

        # 4. Run detection
        result = camera_runner.run_detection(aed_type_id, capture_period_s)

        # 5. Stop the LED
        led_controller.stop_pulse()

        # 6. Fetch artifacts (non-fatal if some are missing)
        artifacts = camera_runner.fetch_artifacts(test_dir)
        logger.info("Fetched %d artifact(s): %s", len(artifacts), list(artifacts.keys()))

        # 7. Save logs and structured result
        _save_log(test_dir, result)
        _save_result_json(
            test_dir, aed_type_id, aed_name, brightness_pct,
            duration_ms, period_ms, capture_period_s, result, artifacts,
        )

        # 8. Record for session summary
        TestHILNv12._results.append({
            "test": test_label,
            "aed_type_id": aed_type_id,
            "brightness_pct": brightness_pct,
            "detected": result.detected,
            "detection_code": result.detection_code,
            "exit_code": result.exit_code,
            "frames_processed": result.frames_processed,
            "total_time_ms": result.total_time_ms,
        })

        # 9. Assert: 100% brightness must detect
        if brightness_pct == 100:
            assert result.detected, (
                f"{aed_name} not detected at 100% brightness "
                f"(exit_code={result.exit_code}, code={result.detection_code})"
            )
        else:
            # Lower brightness is characterization — warn but don't fail
            if not result.detected:
                logger.warning(
                    "%s NOT detected at %d%% brightness (characterization)",
                    aed_name, brightness_pct,
                )

    def test_no_pulse(
        self,
        camera_runner,
        led_controller,
        result_dir,
        aed_type_id_no_pulse,
        aed_name_no_pulse,
        aed_params_no_pulse,  # pylint: disable=unused-argument
    ):
        # pylint: disable=too-many-arguments,too-many-positional-arguments
        """Verify camera does NOT detect when LED is off."""
        led_controller.stop_pulse()
        time.sleep(2.0)
        camera_runner.clean_artifacts()

        capture_period_s = 12  # minimum capture
        result = camera_runner.run_detection(aed_type_id_no_pulse, capture_period_s)
        led_controller.stop_pulse()  # ensure off

        test_label = f"no_pulse-{aed_name_no_pulse}"
        test_dir = os.path.join(result_dir, test_label)
        os.makedirs(test_dir, exist_ok=True)

        # Fetch artifacts (JPGs help diagnose false positives)
        artifacts = camera_runner.fetch_artifacts(test_dir)
        logger.info("Fetched %d artifact(s): %s", len(artifacts), list(artifacts.keys()))

        _save_log(test_dir, result)
        _save_result_json(
            test_dir, aed_type_id_no_pulse, aed_name_no_pulse, 0,
            0, 0, capture_period_s, result, artifacts,
        )

        # Record for session summary
        TestHILNv12._results.append({
            "test": test_label,
            "aed_type_id": aed_type_id_no_pulse,
            "brightness_pct": 0,
            "detected": result.detected,
            "detection_code": result.detection_code,
            "exit_code": result.exit_code,
            "frames_processed": result.frames_processed,
            "total_time_ms": result.total_time_ms,
        })

        assert not result.detected, (
            f"{aed_name_no_pulse} falsely detected with LED off "
            f"(exit_code={result.exit_code}, code={result.detection_code})"
        )

    @pytest.fixture(autouse=True, scope="session")
    def write_summary(self, result_dir):
        """Write summary.json and print results table after all tests."""
        yield

        if not TestHILNv12._results:
            return

        # Write summary JSON
        summary_path = os.path.join(result_dir, "summary.json")
        summary = {
            "total": len(TestHILNv12._results),
            "detected": sum(1 for r in TestHILNv12._results if r["detected"]),
            "not_detected": sum(1 for r in TestHILNv12._results if not r["detected"]),
            "results": TestHILNv12._results,
        }
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        # Print results table
        header = f"{'Test':<35} {'Bright':>6} {'Det':>5} {'Code':>5} {'Frames':>7} {'Time':>8}"
        sep = "-" * len(header)
        logger.info("\n%s\n%s", header, sep)
        for r in TestHILNv12._results:
            det_str = "YES" if r["detected"] else "NO"
            logger.info(
                "%-35s %5d%% %5s %5d %7d %7dms",
                r["test"],
                r["brightness_pct"],
                det_str,
                r["detection_code"],
                r["frames_processed"],
                r["total_time_ms"],
            )
        logger.info(sep)
        logger.info(
            "Total: %d | Detected: %d | Not detected: %d",
            summary["total"], summary["detected"], summary["not_detected"],
        )
        logger.info("Summary written to %s", summary_path)
