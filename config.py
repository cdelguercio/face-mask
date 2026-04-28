from dataclasses import dataclass, field
from typing import Optional

import numpy as np

# Camera settings
CAMERA_INDEX = 0
CAMERA_WIDTH = 1280
CAMERA_HEIGHT = 720

# Output resolution (should match your Resolume composition)
OUTPUT_WIDTH = 1920
OUTPUT_HEIGHT = 1080

# Sender names (visible in Resolume / NDI monitors)
SPOUT_SENDER_NAME = "FaceMaskSpout"
NDI_SENDER_NAME = "FaceMaskNDI"

# Mask defaults
DEFAULT_BLUR_RADIUS = 21  # must be odd
DEFAULT_DILATION_PIXELS = 10
MAX_FACES = 1

# Calibration
NUDGE_STEP = 5  # pixels per arrow-key press


DETECTOR_MODES = ["mp+scrfd+yolo", "mp+scrfd", "mp+yolo", "mp", "scrfd", "yolo"]


from detector import (
    FACE_OVAL_INDICES,
    LEFT_EYE_INDICES,
    LEFT_EYEBROW_INDICES,
    RIGHT_EYE_INDICES,
    RIGHT_EYEBROW_INDICES,
)

CONTOUR_MODES = {
    "eyes": [LEFT_EYE_INDICES, RIGHT_EYE_INDICES],
    "face": [FACE_OVAL_INDICES],
    "eyes_and_brows": [
        LEFT_EYE_INDICES,
        RIGHT_EYE_INDICES,
        LEFT_EYEBROW_INDICES,
        RIGHT_EYEBROW_INDICES,
    ],
}


@dataclass
class RuntimeConfig:
    """Mutable runtime configuration, updated by GUI trackbars."""
    mask_mode: str = "eyes"  # "eyes", "face", "eyes_and_brows"
    detector_mode: str = "mp+scrfd+yolo"
    blur_radius: int = DEFAULT_BLUR_RADIUS
    dilation_pixels: int = DEFAULT_DILATION_PIXELS
    # v2: camera-to-projector homography transform
    homography_matrix: Optional[np.ndarray] = field(default=None, repr=False)
    # Hold last mask position (ms) when all detectors lose the face
    hold_ms: int = 1000
