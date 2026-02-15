"""SSH/SCP wrapper for running aedcamera_nv12.cam on a remote Wyze Cam V3."""

import dataclasses
import logging
import re
import subprocess

logger = logging.getLogger(__name__)

# Artifacts the camera may produce in /tmp/
CAMERA_ARTIFACTS = [
    "NV12_WholeRecording.jpg",
    "NV12_DetectionStack.jpg",
    "NV12_Captioned.jpg",
    "CapturedFrame.jpg",
]

SSH_OPTS = [
    "-o", "StrictHostKeyChecking=no",
    "-o", "UserKnownHostsFile=/dev/null",
    "-o", "ConnectTimeout=10",
    "-o", "LogLevel=ERROR",
]


@dataclasses.dataclass
class CameraResult:
    """Result of a single camera detection run."""

    exit_code: int
    stdout: str
    stderr: str
    detected: bool
    total_time_ms: int
    frames_processed: int
    frames_detected: int
    detection_code: int  # 6=DET_ANY, 2=DET_NON, 0=DET_GRN, 1=DET_RED

    @staticmethod
    def from_process(proc):
        """Parse a completed subprocess into a CameraResult."""
        stdout = proc.stdout or ""
        stderr = proc.stderr or ""

        total_time_ms = 0
        frames_processed = 0
        frames_detected = 0
        detection_code = -1

        # Parse "Total time: 12345 ms"
        m = re.search(r"Total time:\s*(\d+)\s*ms", stdout)
        if m:
            total_time_ms = int(m.group(1))

        # Parse "Frames: 360" or "Frames processed: 360"
        m = re.search(r"Frames(?:\s+processed)?:\s*(\d+)", stdout)
        if m:
            frames_processed = int(m.group(1))

        # Parse "Frames detected: 5" or "Events: 5"
        m = re.search(r"(?:Frames detected|Events):\s*(\d+)", stdout)
        if m:
            frames_detected = int(m.group(1))

        # Parse detection code from "Detection: DET_ANY (6)" or "result=6"
        m = re.search(r"(?:Detection:.*\((\d+)\)|result=(\d+))", stdout)
        if m:
            detection_code = int(m.group(1) or m.group(2))

        # Detection succeeded if exit code is 0
        detected = proc.returncode == 0

        return CameraResult(
            exit_code=proc.returncode,
            stdout=stdout,
            stderr=stderr,
            detected=detected,
            total_time_ms=total_time_ms,
            frames_processed=frames_processed,
            frames_detected=frames_detected,
            detection_code=detection_code,
        )


class CameraRunner:
    """Manages SSH sessions to a remote camera for detection runs."""

    BINARY = "/usr/bin/aedcamera_nv12.cam"

    def __init__(self, host="192.168.1.221", user="root"):
        self.host = host
        self.user = user
        self._target = f"{user}@{host}"

    def _ssh(self, command, timeout=None):
        """Run a command on the camera via SSH."""
        cmd = ["ssh"] + SSH_OPTS + [self._target, command]
        logger.debug("SSH: %s", command)
        return subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )

    def _scp_from(self, remote_path, local_path):
        """Copy a file from the camera to local. Uses -O for dropbear compat."""
        cmd = ["scp", "-O"] + SSH_OPTS + [
            f"{self._target}:{remote_path}",
            local_path,
        ]
        return subprocess.run(
            cmd, capture_output=True, text=True, timeout=30, check=False,
        )

    def check_connectivity(self):
        """Return True if the camera is reachable via SSH."""
        try:
            proc = self._ssh("echo ok", timeout=15)
            return proc.returncode == 0 and "ok" in proc.stdout
        except (subprocess.TimeoutExpired, OSError) as e:
            logger.error("Camera connectivity check failed: %s", e)
            return False

    def clean_artifacts(self):
        """Remove any leftover detection artifacts on the camera."""
        files = " ".join(f"/tmp/{a}" for a in CAMERA_ARTIFACTS)
        self._ssh(f"rm -f {files}", timeout=10)

    def run_detection(self, aed_type_id, capture_period_s, timeout_s=None):
        """
        Run aedcamera_nv12.cam on the camera and return parsed results.

        Args:
            aed_type_id: AED type integer (0-7)
            capture_period_s: Capture period in seconds (-2 arg)
            timeout_s: SSH timeout; defaults to capture_period_s + 30

        Returns:
            CameraResult with parsed output
        """
        if timeout_s is None:
            timeout_s = capture_period_s + 30

        # -u 1: CLI mode
        # -0 <id>: AED type
        # -2 <period>: capture period seconds
        # -3 3: debug caption ON (bit1) + custom thresholds ON (bit0)
        remote_cmd = (
            f"{self.BINARY} -u 1 -0 {aed_type_id} -2 {capture_period_s} -3 3"
        )
        logger.info("Running detection: %s (timeout=%ds)", remote_cmd, timeout_s)

        try:
            proc = self._ssh(remote_cmd, timeout=timeout_s)
        except subprocess.TimeoutExpired:
            logger.error("Detection timed out after %ds", timeout_s)
            return CameraResult(
                exit_code=-1,
                stdout="",
                stderr="Timed out",
                detected=False,
                total_time_ms=0,
                frames_processed=0,
                frames_detected=0,
                detection_code=-1,
            )

        result = CameraResult.from_process(proc)
        logger.info(
            "Detection complete: detected=%s, code=%d, frames=%d, time=%dms",
            result.detected,
            result.detection_code,
            result.frames_processed,
            result.total_time_ms,
        )
        return result

    def fetch_artifacts(self, local_dir):
        """
        SCP detection artifacts from /tmp/ to local_dir.

        Returns:
            dict mapping artifact name -> local path for files successfully fetched
        """
        fetched = {}
        for artifact in CAMERA_ARTIFACTS:
            remote = f"/tmp/{artifact}"
            local = f"{local_dir}/{artifact}"
            proc = self._scp_from(remote, local)
            if proc.returncode == 0:
                fetched[artifact] = local
                logger.info("Fetched %s", artifact)
            else:
                logger.debug("Artifact not available: %s", artifact)
        return fetched
