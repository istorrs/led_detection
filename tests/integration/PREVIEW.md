# Integration Test Preview

## Enabling Preview Window

To see the live camera preview during integration tests:

```bash
INTEGRATION_PREVIEW=1 ./venv/bin/python -m pytest -m integration -v -s
```

The preview window will:
- Show the live camera feed
- Display ROI with detected LED
- Show current threshold and detected pulses
- Window title shows test parameters (duration/period/brightness)

## Running Without Preview (Default)

Normal test run without preview:
```bash
./venv/bin/python -m pytest -m integration -v
```

## Single Test with Preview

Preview one specific test case:
```bash
INTEGRATION_PREVIEW=1 ./venv/bin/python -m pytest -m integration -v -s -k "50-500-100"
```

## Notes

- Preview shows ~3 pulse cycles per test
- Press 'q' to skip waiting and proceed to next test
- Each test gets its own window title showing parameters
- Useful for debugging and visual verification
