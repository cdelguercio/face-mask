"""YOLO-based face detector — fallback when MediaPipe loses tracking.

Uses ultralytics YOLOv12-face to detect face bounding boxes.
Returns BBoxResult (normalized 0-1) instead of landmarks.

Gracefully optional: if ultralytics is not installed, YOLO_AVAILABLE is False
and YoloFaceDetector.detect() returns an empty list.
"""

import os
from typing import List

import numpy as np

from detector_types import BBoxResult

_MODEL_DIR = os.path.dirname(os.path.abspath(__file__))

try:
    from ultralytics import YOLO
    import torch

    YOLO_AVAILABLE = True
    _DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
except ImportError:
    YOLO_AVAILABLE = False
    _DEVICE = "cpu"


class YoloFaceDetector:
    """Face detector using YOLO.  Intended as a fallback for MediaPipe.

    Parameters
    ----------
    model_name : str
        Ultralytics model identifier.  ``"yolo11n-face.pt"`` is a lightweight
        face-specific model; ``"yolov8n-face.pt"`` also works.  The model is
        auto-downloaded on first use.
    confidence : float
        Minimum detection confidence (0-1).
    enabled : bool
        Master switch — when False, detect() always returns [].
    """

    def __init__(
        self,
        model_name: str = "yolov12n-face.pt",
        confidence: float = 0.7,
        enabled: bool = True,
    ):
        self.enabled = enabled
        self.confidence = confidence
        self._model = None

        if not YOLO_AVAILABLE:
            if enabled:
                print(
                    "[YOLO] ultralytics is not installed. "
                    "YOLO fallback disabled.  pip install ultralytics"
                )
            self.enabled = False
            return

        if not enabled:
            return

        try:
            # Look for model in the project directory first
            local_path = os.path.join(_MODEL_DIR, model_name)
            model_path = local_path if os.path.exists(local_path) else model_name
            print(f"[YOLO] Loading model '{model_path}' on {_DEVICE}...")
            self._model = YOLO(model_path)
            self._model.to(_DEVICE)
            print(f"[YOLO] Model loaded on {_DEVICE}.")
        except Exception as e:
            print(f"[YOLO] Failed to load model: {e}")
            self.enabled = False

    def detect(self, frame_bgr: np.ndarray) -> List[BBoxResult]:
        """Run YOLO face detection on a BGR frame.

        Returns a list of BBoxResult with normalized (0-1) coordinates.
        Returns an empty list when disabled or if no faces found.
        """
        if not self.enabled or self._model is None:
            return []

        h, w = frame_bgr.shape[:2]
        if h == 0 or w == 0:
            return []

        results = self._model.predict(
            frame_bgr,
            conf=self.confidence,
            verbose=False,
        )

        faces: List[BBoxResult] = []
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                # Normalize to 0-1
                bx = float(x1 / w)
                by = float(y1 / h)
                bw = float((x2 - x1) / w)
                bh = float((y2 - y1) / h)
                faces.append(BBoxResult(x=bx, y=by, w=bw, h=bh, label="YOLO"))

        return faces
