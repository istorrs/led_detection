"""Shared fixtures and CLI options for HIL integration tests."""

import json
from datetime import datetime
from pathlib import Path

import pytest  # pylint: disable=import-error

from tests.integration.camera_runner import CameraRunner
from tests.integration.led_controller import LEDController

PROJECT_ROOT = Path(__file__).resolve().parents[2]
AED_TYPES_JSON = PROJECT_ROOT / "aed_types.json"
RESULT_BASE = PROJECT_ROOT / "test_results"


def pytest_addoption(parser):
    parser.addoption(
        "--camera-ip", default="192.168.1.221", help="Camera IP address"
    )
    parser.addoption(
        "--led-port", default="/dev/ttyUSB4", help="LED controller serial port"
    )
    parser.addoption(
        "--brightness-levels",
        default="1,33,66,100",
        help="Comma-separated brightness percentages to test",
    )


@pytest.fixture(scope="session")
def camera_runner(request):
    ip = request.config.getoption("--camera-ip")
    runner = CameraRunner(host=ip)
    if not runner.check_connectivity():
        pytest.skip(f"Camera at {ip} is unreachable")
    return runner


@pytest.fixture(scope="session")
def led_controller(request):
    port = request.config.getoption("--led-port")
    ctrl = LEDController(port=port)
    if not ctrl.connect():
        pytest.skip(f"LED controller on {port} is unavailable")
    yield ctrl
    ctrl.stop_pulse()
    ctrl.disconnect()


@pytest.fixture(scope="session")
def result_dir():
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    d = RESULT_BASE / f"hil_nv12_{stamp}"
    d.mkdir(parents=True, exist_ok=True)
    return d


@pytest.fixture(scope="session")
def aed_types():
    with open(AED_TYPES_JSON, encoding="utf-8") as f:
        data = json.load(f)
    return {entry["id"]: entry for entry in data["aed_types"]}


def pytest_generate_tests(metafunc):
    """Dynamically parametrize tests over AED types x brightness levels."""
    with open(AED_TYPES_JSON, encoding="utf-8") as f:
        data = json.load(f)

    if "aed_type_id" in metafunc.fixturenames:
        levels_str = metafunc.config.getoption("--brightness-levels", "33,66,100")
        brightness_levels = [int(x.strip()) for x in levels_str.split(",")]

        argvalues = []
        ids = []

        for entry in data["aed_types"]:
            # Skip LCD (id=0) — aperiodic, not suitable for pulse testing
            if entry["id"] == 0:
                continue
            for brightness in brightness_levels:
                argvalues.append((entry["id"], entry["name"], brightness, entry["params"]))
                ids.append(f"{entry['name']}-{brightness}pct")

        metafunc.parametrize(
            "aed_type_id,aed_name,brightness_pct,aed_params",
            argvalues,
            ids=ids,
        )

    if "aed_type_id_no_pulse" in metafunc.fixturenames:
        # Parametrize over AED types only (no brightness dimension)
        argvalues = []
        ids = []
        for entry in data["aed_types"]:
            if entry["id"] == 0:
                continue
            argvalues.append((entry["id"], entry["name"], entry["params"]))
            ids.append(f"no_pulse-{entry['name']}")
        metafunc.parametrize(
            "aed_type_id_no_pulse,aed_name_no_pulse,aed_params_no_pulse",
            argvalues,
            ids=ids,
        )
