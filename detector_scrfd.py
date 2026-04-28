"""SCRFD-based face detector.

Uses InsightFace's FaceAnalysis wrapper, whose detector is SCRFD. This module
is intentionally optional: if insightface/onnxruntime are not installed, the
detector disables itself and returns no boxes.
"""

from typing import List

import numpy as np

from detector_types import BBoxResult

try:
    from insightface.app import FaceAnalysis
    import onnxruntime as ort

    SCRFD_AVAILABLE = True
except ImportError:
    FaceAnalysis = None
    ort = None
    SCRFD_AVAILABLE = False


class ScrfdFaceDetector:
    """Face detector backed by InsightFace SCRFD."""

    def __init__(
        self,
        model_name: str = "buffalo_l",
        confidence: float = 0.7,
        det_size: tuple[int, int] = (640, 640),
        enabled: bool = True,
    ):
        self.enabled = enabled
        self.confidence = confidence
        self._app = None

        if not SCRFD_AVAILABLE:
            if enabled:
                print(
                    "[SCRFD] insightface is not installed. "
                    "SCRFD disabled. pip install insightface onnxruntime-gpu"
                )
            self.enabled = False
            return

        if not enabled:
            return

        try:
            available_providers = ort.get_available_providers()
            providers = [
                p for p in ["CUDAExecutionProvider", "CPUExecutionProvider"]
                if p in available_providers
            ]
            print(f"[SCRFD] Loading InsightFace model '{model_name}'...")
            self._app = FaceAnalysis(
                name=model_name,
                allowed_modules=["detection"],
                providers=providers,
            )
            ctx_id = 0
            self._app.prepare(ctx_id=ctx_id, det_size=det_size)
            print(f"[SCRFD] Model ready at det_size={det_size}.")
        except Exception as e:
            print(f"[SCRFD] Failed to load model: {e}")
            self.enabled = False

    def detect(self, frame_bgr: np.ndarray) -> List[BBoxResult]:
        """Run SCRFD face detection and return normalized bounding boxes."""
        if not self.enabled or self._app is None:
            return []

        h, w = frame_bgr.shape[:2]
        if h == 0 or w == 0:
            return []

        try:
            faces = self._app.get(frame_bgr)
        except Exception as e:
            print(f"[SCRFD] Detect error: {e}")
            return []

        results: List[BBoxResult] = []
        for face in faces:
            score = float(getattr(face, "det_score", 1.0))
            if score < self.confidence:
                continue
            x1, y1, x2, y2 = face.bbox.astype(float)
            x1 = min(max(x1, 0.0), float(w))
            y1 = min(max(y1, 0.0), float(h))
            x2 = min(max(x2, 0.0), float(w))
            y2 = min(max(y2, 0.0), float(h))
            bw = x2 - x1
            bh = y2 - y1
            if bw <= 0 or bh <= 0:
                continue
            results.append(BBoxResult(
                x=float(x1 / w),
                y=float(y1 / h),
                w=float(bw / w),
                h=float(bh / h),
                label="SCRFD",
            ))

        return results
