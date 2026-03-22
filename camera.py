"""Camera discovery, permission diagnostics, and hot-swap support.

On Windows, Python apps need explicit camera permission granted in:
  Settings > Privacy & Security > Camera > Let desktop apps access your camera
"""

import platform
import subprocess
import sys
import threading
from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import numpy as np

VIRTUAL_KEYWORDS = ["virtual", "nvidia", "obs", "snap", "manycam", "xsplit", "imaging edge"]


def _check_windows_camera_devices() -> list[dict]:
    """Use PowerShell to list camera devices registered in Windows."""
    if platform.system() != "Windows":
        return []
    try:
        result = subprocess.run(
            [
                "powershell", "-Command",
                "Get-PnpDevice -Class Camera -ErrorAction SilentlyContinue "
                "| Select-Object -Property Status, FriendlyName "
                "| ConvertTo-Json",
            ],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode != 0:
            return []
        import json
        data = json.loads(result.stdout)
        if isinstance(data, dict):
            data = [data]
        return [
            {"name": d.get("FriendlyName", "Unknown"), "status": d.get("Status", "Unknown")}
            for d in data
        ]
    except Exception:
        return []


def _open_windows_camera_settings():
    if platform.system() != "Windows":
        return
    try:
        subprocess.Popen(
            ["explorer", "ms-settings:privacy-webcam"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
    except Exception:
        pass


def _print_permission_help(hw_devices: list[dict]):
    print()
    print("=" * 70)
    print("CAMERA PERMISSION REQUIRED")
    print("=" * 70)
    print()
    print("Windows detected these camera devices:")
    for dev in hw_devices:
        marker = "  (virtual)" if any(
            kw in dev["name"].lower() for kw in VIRTUAL_KEYWORDS
        ) else ""
        print(f"  - {dev['name']} [{dev['status']}]{marker}")
    print()
    print("But Python cannot access the physical camera. To fix:")
    print()
    print("  1. Open Settings > Privacy & Security > Camera")
    print("  2. Ensure 'Camera access' is ON (master toggle)")
    print("  3. Scroll down to 'Let desktop apps access your camera'")
    print("  4. Toggle it ON")
    print(f"  5. Find 'python{sys.version_info.major}.{sys.version_info.minor}.exe' and enable it")
    print()
    print("=" * 70)
    print()


def _is_virtual_camera(frame) -> bool:
    if frame is None:
        return True
    std = frame.std()
    return std > 70 or std < 10


@dataclass
class CameraInfo:
    """Metadata about a discovered camera."""
    index: int
    backend: int
    backend_name: str
    width: int
    height: int
    is_virtual: bool
    label: str  # human-readable label for the trackbar


def _probe_camera(idx: int, backend: int, timeout_s: float = 3.0) -> Optional[CameraInfo]:
    """Probe a single camera index with a timeout. Returns CameraInfo or None."""
    result = [None]

    def _probe():
        cap = cv2.VideoCapture(idx, backend)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret:
                w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                backend_name = "DSHOW" if backend == cv2.CAP_DSHOW else "MSMF"
                is_virt = _is_virtual_camera(frame)
                tag = " (virtual)" if is_virt else ""
                result[0] = CameraInfo(
                    index=idx, backend=backend, backend_name=backend_name,
                    width=w, height=h, is_virtual=is_virt,
                    label=f"Cam {idx}{tag} [{w}x{h}]",
                )
            cap.release()

    t = threading.Thread(target=_probe, daemon=True)
    t.start()
    t.join(timeout=timeout_s)
    if t.is_alive():
        return None  # timed out (e.g. NVIDIA virtual cam hanging)
    return result[0]


def discover_cameras() -> List[CameraInfo]:
    """Scan for all available cameras with per-device timeout."""
    cameras = []

    for idx in range(10):
        print(f"  probing index {idx}...", end="", flush=True)
        info = _probe_camera(idx, cv2.CAP_DSHOW, timeout_s=3.0)
        if info:
            cameras.append(info)
            print(f" found [{info.width}x{info.height}]{' (virtual)' if info.is_virtual else ''}")
        else:
            print(" skip")

    return cameras


class CameraManager:
    """Manages camera discovery, selection, and hot-swapping.

    Integrates with the OpenCV window as a trackbar for live camera switching.
    """

    def __init__(self, width: int = 1280, height: int = 720):
        self.req_width = width
        self.req_height = height
        self.cameras: List[CameraInfo] = []
        self.current_idx: int = 0  # index into self.cameras list
        self.cap: Optional[cv2.VideoCapture] = None
        self._pending_switch: Optional[int] = None

    def discover(self) -> List[CameraInfo]:
        """Scan for cameras. Call before attach_to_window."""
        print("Scanning for cameras...")
        self.cameras = discover_cameras()

        if not self.cameras:
            hw_devices = _check_windows_camera_devices()
            physical = [d for d in hw_devices
                        if not any(kw in d["name"].lower() for kw in VIRTUAL_KEYWORDS)]
            if physical:
                _print_permission_help(hw_devices)
                _open_windows_camera_settings()
            else:
                print("\nERROR: No cameras found. Connect a webcam and try again.")
            sys.exit(1)

        real = [c for c in self.cameras if not c.is_virtual]
        print(f"Found {len(self.cameras)} camera(s) "
              f"({len(real)} real, {len(self.cameras) - len(real)} virtual)")
        for i, cam in enumerate(self.cameras):
            print(f"  [{i}] {cam.label}")

        return self.cameras

    def open_camera(self, list_index: int) -> bool:
        """Open camera by index into self.cameras list."""
        if list_index < 0 or list_index >= len(self.cameras):
            return False

        cam = self.cameras[list_index]

        # Release current camera
        if self.cap is not None:
            self.cap.release()
            self.cap = None

        cap = cv2.VideoCapture(cam.index, cam.backend)
        if not cap.isOpened():
            print(f"ERROR: Could not open camera {cam.index}")
            return False

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.req_width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.req_height)
        actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"Opened {cam.label} at {actual_w}x{actual_h}")

        self.cap = cap
        self.current_idx = list_index
        return True

    def open_default(self, preferred_index: Optional[int] = None) -> bool:
        """Open the best default camera (first non-virtual, or preferred)."""
        if preferred_index is not None:
            # Find it in our list
            for i, cam in enumerate(self.cameras):
                if cam.index == preferred_index:
                    return self.open_camera(i)
            # Not in list — try direct open
            print(f"Opening camera {preferred_index}...")
            cap = cv2.VideoCapture(preferred_index, cv2.CAP_DSHOW)
            if not cap.isOpened():
                cap = cv2.VideoCapture(preferred_index)
            if cap.isOpened():
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.req_width)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.req_height)
                self.cap = cap
                return True
            print(f"ERROR: Could not open camera {preferred_index}")
            sys.exit(1)

        # Auto-select first non-virtual
        for i, cam in enumerate(self.cameras):
            if not cam.is_virtual:
                return self.open_camera(i)
        # Fall back to first camera
        return self.open_camera(0)

    def _on_trackbar_change(self, val: int):
        """Called by OpenCV trackbar — schedule a camera switch."""
        if val != self.current_idx:
            self._pending_switch = val

    def attach_to_window(self, window_name: str):
        """Add a camera selector trackbar to an OpenCV window."""
        if len(self.cameras) <= 1:
            return  # no point showing selector for one camera

        cv2.createTrackbar(
            "Camera",
            window_name,
            self.current_idx,
            len(self.cameras) - 1,
            self._on_trackbar_change,
        )

    def check_switch(self) -> bool:
        """Process any pending camera switch. Call each frame.

        Returns True if camera was switched.
        """
        if self._pending_switch is not None:
            target = self._pending_switch
            self._pending_switch = None
            return self.open_camera(target)
        return False

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        """Read a frame from the current camera."""
        if self.cap is None:
            return False, None
        return self.cap.read()

    def get_camera_label(self) -> str:
        """Get the label of the currently active camera."""
        if 0 <= self.current_idx < len(self.cameras):
            return self.cameras[self.current_idx].label
        return "Unknown"

    def release(self):
        """Release the camera."""
        if self.cap is not None:
            self.cap.release()
            self.cap = None
