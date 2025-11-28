# Integration Tests

## Overview

Integration tests verify LED pulse detection with real hardware connected via serial port `/dev/ttyUSB-4`.

## Setup

1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Connect LED hardware** to `/dev/ttyUSB-4`

3. **Verify connection**:
   ```bash
   ls -l /dev/ttyUSB-4
   ```

## Running Tests

### Skip Integration Tests (Default)
```bash
pytest tests/
# OR explicitly
pytest -m "not integration"
```

### Run Only Integration Tests
```bash
pytest -m integration
```

### Run All Tests
```bash
pytest
```

## Test Matrix

Tests cover combinations of:
- **Durations**: 50ms, 100ms, 150ms, 200ms, 500ms, 1000ms, 1500ms, 2000ms
- **Periods**: 500ms, 1s, 2s, 5s, 10s, 15s, 20s
- **Brightness**: 20%, 40%, 60%, 80%, 100%

**Total**: 20 parametrized test cases

## Expected Behavior

- Tests control LED via `led_pulse <dur_ms> <period_ms> [brightness_%]` command
- Verify command succeeds
- Future: Verify detected timing within ±10% tolerance

## Troubleshooting

### Serial Port Not Found
```
pytest.skip: LED controller not available on /dev/ttyUSB-4
```

**Solution**: Check hardware connection and permissions:
```bash
sudo chmod 666 /dev/ttyUSB-4
```

### Import Error: No module 'serial'
```
ImportError: No module named 'serial'
```

**Solution**: Install pyserial:
```bash
pip install pyserial
```
