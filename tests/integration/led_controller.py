"""LED controller for integration testing via serial port."""

import re
import time
import logging

import serial  # pylint: disable=import-error


class LEDController:
    """Controls LED hardware via serial commands."""

    def __init__(self, port='/dev/ttyUSB4', baudrate=115200, timeout=2.0):
        """
        Initialize LED controller.

        Args:
            port: Serial port device path
            baudrate: Serial communication baud rate
            timeout: Read timeout in seconds
        """
        self.port = port
        self.baudrate = baudrate
        self.timeout = timeout
        self.serial = None

    def connect(self):
        """Open serial connection to LED controller."""
        try:
            self.serial = serial.Serial(
                port=self.port,
                baudrate=self.baudrate,
                timeout=self.timeout
            )
            time.sleep(0.5)  # Allow connection to stabilize
            logging.info("Connected to LED controller on %s", self.port)
            return True
        except (serial.SerialException, FileNotFoundError) as e:
            logging.error("Failed to connect to %s: %s", self.port, e)
            return False

    def disconnect(self):
        """Close serial connection."""
        if self.serial and self.serial.is_open:
            self.serial.close()
            logging.info("Disconnected from LED controller")

    def send_command(self, command):
        # pylint: disable=too-many-return-statements,too-many-nested-blocks
        """
        Send command to LED controller.

        Args:
            command: Command string to send

        Returns:
            Response from controller or None if error
        """
        if not self.serial or not self.serial.is_open:
            logging.error("Serial port not open")
            return None

        try:
            # Clear input buffer to remove any stale data from previous commands
            self.serial.reset_input_buffer()

            self.serial.write(f"{command}\n".encode())
            self.serial.flush()

            # Read until we see the prompt again, indicating command completion
            # ESP32 CLI format: "ESP32 CLI> "
            response_lines = []
            timeout = time.time() + 5.0  # Maximum 2 second timeout

            while time.time() < timeout:
                if self.serial.in_waiting:
                    try:
                        line = self.serial.readline().decode('utf-8', errors='replace').strip()
                        if line:
                            logging.info("Response from LED: %s", line)
                            response_lines.append(line)

                            # Check if this is the prompt indicating command is done
                            if line.startswith("ESP32 CLI>"):
                                break
                    except Exception as e:  # pylint: disable=broad-except
                        logging.warning("Could not decode response line: %s", e)
                        continue
                else:
                    time.sleep(0.05)  # Small delay to avoid busy-waiting

            if not response_lines:
                logging.warning("No response received from LED controller")
                return None

            # Check for error responses from ESP32
            for line in response_lines:
                if "Invalid command syntax:" in line or "Unknown command:" in line:
                    logging.error("ESP32 error: %s", line)
                    return None

            # Find the confirmation line that shows actual parameters
            # Format: "LED pulse set: XXms ON @ YY% / ZZZms period"
            confirmation = None
            for line in response_lines:
                if "LED pulse set:" in line:
                    confirmation = line
                    break

            if confirmation:
                return confirmation

            # If no confirmation line, return last non-prompt response
            for line in reversed(response_lines):
                if not line.startswith("ESP32 CLI>"):
                    return line

            return "OK"

        except serial.SerialException as e:
            logging.error("Serial communication error: %s", e)
            return None

    def _parse_time_str(self, time_str):
        """Parse time string with units to milliseconds."""
        # Handle decimal numbers
        value_match = re.match(r'([\d\.]+)(.*)', time_str)
        if not value_match:
            return 0

        value = float(value_match.group(1))
        unit = value_match.group(2).strip()

        if unit in ('us', 'μs'):
            return int(value / 1000) if value >= 1000 else value / 1000.0
        if unit == 's':
            return int(value * 1000)

        # Default to ms
        return int(value)

    def set_pulse(self, duration_ms, period_ms, brightness_pct=100):
        """
        Configure LED pulse parameters.

        Args:
            duration_ms: Pulse duration (50-2000ms)
            period_ms: Pulse period (500ms-3600000ms / 1 hour)
            brightness_pct: LED brightness (0-100%)

        Returns:
            True if command succeeded, False otherwise
        """
        # Validate parameters
        if not 50 <= duration_ms <= 2000:
            logging.error("Duration %dms out of range (50-2000)", duration_ms)
            return False

        if not 500 <= period_ms <= 3600000:
            logging.error("Period %dms out of range (500-3600000)", period_ms)
            return False

        if not 0 <= brightness_pct <= 100:
            logging.error("Brightness %d%% out of range (0-100)", brightness_pct)
            return False

        # Add units to command
        command = f"led_pulse {duration_ms}ms {period_ms}ms {brightness_pct}"
        logging.info("Sending: %s", command)
        response = self.send_command(command)

        if response is None:
            return False

        # Validate response matches command (if we got a confirmation line)
        # Format: "LED pulse set: 50ms ON @ 10% / 1000ms period"
        # Or: "LED pulse set: 500μs ON @ 20% / 5.0s period"
        if "LED pulse set:" in response:
            # Flexible regex to capture value and unit
            # Group 1: duration value+unit
            # Group 2: brightness
            # Group 3: period value+unit
            match = re.search(r'([\d\.]+(?:ms|us|μs|s)) ON @ (\d+)% / ([\d\.]+(?:ms|us|μs|s)) period', response)

            if match:
                actual_dur_str = match.group(1)
                actual_bright = int(match.group(2))
                actual_period_str = match.group(3)

                # Convert to ms for comparison
                # Note: floating point comparison might need tolerance, but let's see
                actual_dur_ms = self._parse_time_str(actual_dur_str)
                actual_period_ms = self._parse_time_str(actual_period_str)

                # Use a small tolerance for float conversions (e.g. 1ms)
                if (abs(actual_dur_ms - duration_ms) > 1 or
                    actual_bright != brightness_pct or
                    abs(actual_period_ms - period_ms) > 1):
                    logging.warning(
                        "LED response mismatch! Sent: %dms/%dms/%d%%, Got: %s/%s/%d%% (Parsed: %.1fms/%.1fms)",
                        duration_ms, period_ms, brightness_pct,
                        actual_dur_str, actual_period_str, actual_bright,
                        actual_dur_ms, actual_period_ms
                    )
                    return False
                logging.info("LED confirmed: %dms/%dms/%d%%",
                           duration_ms, period_ms, brightness_pct)
            else:
                logging.warning("Could not parse LED confirmation: %s", response)
                # We return True here because technically the command succeeded if we got here,
                # but we warn about parsing failure. This mimics previous behavior but with logging.

        return True

    def stop_pulse(self):
        """Stop LED pulsing."""
        # Use minimum valid duration (50ms) with 0% brightness to effectively stop
        command = "led_pulse 50 1000 0"
        return self.send_command(command) is not None

    def __enter__(self):
        """Context manager entry."""
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop_pulse()
        self.disconnect()
