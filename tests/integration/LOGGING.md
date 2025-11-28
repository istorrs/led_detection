# Integration Test Logging

## See Full Pytest Logs (LED Control Commands)

### Show All Logs (Including LED Controller)
```bash
./venv/bin/python -m pytest -m integration -v -s --log-cli-level=INFO
```

### Show Only Integration Test Logs
```bash
./venv/bin/python -m pytest -m integration -v -s --log-cli-level=INFO -k "test_pulse_detection"
```

### With Preview AND Full Logs
```bash
INTEGRATION_PREVIEW=1 ./venv/bin/python -m pytest -m integration -v -s --log-cli-level=INFO
```

### Test Single Case with Full Logs
```bash
./venv/bin/python -m pytest -m integration -v -s --log-cli-level=INFO -k "50-500-50"
```

## Log Output Includes

- LED controller connection status
- Serial commands sent: `led_pulse <duration> <period> <brightness>`
- Autofocus sweep progress
- Preview window status
- Test pass/fail status

## Example Output
```
INFO     tests.integration.led_controller:led_controller.py:35 Connected to LED controller on /dev/ttyUSB4
INFO     tests.integration.led_controller:led_controller.py:102 Sending: led_pulse 50 500 50
INFO     tests.integration.test_led_pulse:test_led_pulse.py:111 Running autofocus...
INFO     led_detection.main:main.py:649 Focus locked: position=123, hw_autofocus=DISABLED
```
