import os
from dataclasses import dataclass
from typing import List

import mediapipe as mp
import numpy as np
from mediapipe.tasks.python.vision import FaceLandmarksConnections as flc

BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
RunningMode = mp.tasks.vision.RunningMode

MODEL_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "face_landmarker_v2_with_blendshapes.task",
)


def _ordered_contour(connections):
    """Walk a list of Connection(start, end) to produce ordered vertex indices.

    Each contour is a single closed loop — we follow the chain from any start node.
    """
    adj = {}
    for c in connections:
        adj.setdefault(c.start, []).append(c.end)
        adj.setdefault(c.end, []).append(c.start)

    start = next(iter(adj))
    ordered = [start]
    visited = {start}

    current = start
    while True:
        next_node = None
        for n in adj[current]:
            if n not in visited:
                next_node = n
                break
        if next_node is None:
            break
        ordered.append(next_node)
        visited.add(next_node)
        current = next_node

    return ordered


# Pre-compute ordered contour index lists at import time
LEFT_EYE_INDICES = _ordered_contour(flc.FACE_LANDMARKS_LEFT_EYE)
RIGHT_EYE_INDICES = _ordered_contour(flc.FACE_LANDMARKS_RIGHT_EYE)
FACE_OVAL_INDICES = _ordered_contour(flc.FACE_LANDMARKS_FACE_OVAL)
LEFT_EYEBROW_INDICES = _ordered_contour(flc.FACE_LANDMARKS_LEFT_EYEBROW)
RIGHT_EYEBROW_INDICES = _ordered_contour(flc.FACE_LANDMARKS_RIGHT_EYEBROW)


@dataclass
class FaceResult:
    """Detection result for a single face with normalized (0-1) landmarks."""
    landmarks: np.ndarray  # shape (478, 2), values in [0, 1]


class FaceDetector:
    def __init__(self, max_faces: int = 1):
        options = FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=MODEL_PATH),
            running_mode=RunningMode.VIDEO,
            num_faces=max_faces,
            min_face_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self.landmarker = FaceLandmarker.create_from_options(options)
        self._frame_ts = 0

    def detect(self, frame_bgr: np.ndarray) -> List[FaceResult]:
        """Run face landmark detection. Returns list of FaceResult with normalized coords."""
        frame_rgb = frame_bgr[:, :, ::-1].copy()  # BGR -> RGB
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

        self._frame_ts += 33  # ~30fps, timestamps must be monotonically increasing
        result = self.landmarker.detect_for_video(mp_image, self._frame_ts)

        if not result.face_landmarks:
            return []

        faces = []
        for face_landmarks in result.face_landmarks:
            coords = np.array(
                [(lm.x, lm.y) for lm in face_landmarks],
                dtype=np.float32,
            )
            faces.append(FaceResult(landmarks=coords))
        return faces

    def close(self):
        self.landmarker.close()
