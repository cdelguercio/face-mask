from typing import Callable, Dict, List

import numpy as np

from detector import FaceDetector
from detector_scrfd import SCRFD_AVAILABLE, ScrfdFaceDetector
from detector_types import DetectionResult
from detector_yolo import YOLO_AVAILABLE, YoloFaceDetector


class FaceDetectionManager:
    """Owns detector instances and runs the selected fallback chain."""

    def __init__(self, max_faces: int):
        self.max_faces = max_faces
        self._bbox_confidence = 0.7
        self._mp: FaceDetector | None = None
        self._bbox_detectors: Dict[str, object] = {}
        self._bbox_factories: Dict[str, Callable[[], object]] = {
            "scrfd": lambda: ScrfdFaceDetector(
                confidence=self._bbox_confidence,
                enabled=SCRFD_AVAILABLE,
            ),
            "yolo": lambda: YoloFaceDetector(
                confidence=self._bbox_confidence,
                enabled=YOLO_AVAILABLE,
            ),
        }

    def detect(self, frame_bgr: np.ndarray, mode: str) -> DetectionResult:
        for backend in mode.split("+"):
            backend = backend.strip().lower()
            if backend == "mp":
                faces = self._detect_mp(frame_bgr)
                if faces:
                    return DetectionResult(faces=faces, source="MP")
            elif backend in self._bbox_factories:
                bboxes = self._detect_bbox(backend, frame_bgr)
                if bboxes:
                    return DetectionResult(
                        bboxes=bboxes,
                        source=backend.upper(),
                    )
            else:
                print(f"[Detector] Unknown backend '{backend}' in mode '{mode}'")

        return DetectionResult()

    def set_bbox_confidence(self, confidence: float):
        self._bbox_confidence = confidence
        for detector in self._bbox_detectors.values():
            if hasattr(detector, "confidence"):
                detector.confidence = confidence

    def get_bbox_confidence(self) -> float:
        return self._bbox_confidence

    def get_provider_label(self, name: str) -> str:
        detector = self._bbox_detectors.get(name)
        if detector is None:
            return "not loaded"
        return getattr(detector, "provider", "unknown")

    def close(self):
        if self._mp is not None:
            self._mp.close()

    def _detect_mp(self, frame_bgr: np.ndarray):
        if self._mp is None:
            print("[Detector] Loading MediaPipe face landmarker...")
            self._mp = FaceDetector(max_faces=self.max_faces)
        return self._mp.detect(frame_bgr)

    def _detect_bbox(self, name: str, frame_bgr: np.ndarray) -> List:
        detector = self._bbox_detectors.get(name)
        if detector is None:
            detector = self._bbox_factories[name]()
            self._bbox_detectors[name] = detector
        return detector.detect(frame_bgr)
