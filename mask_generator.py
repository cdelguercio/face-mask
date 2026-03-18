from typing import List

import cv2
import numpy as np

from config import RuntimeConfig
from detector import (
    FaceResult,
    FACE_OVAL_INDICES,
    LEFT_EYE_INDICES,
    LEFT_EYEBROW_INDICES,
    RIGHT_EYE_INDICES,
    RIGHT_EYEBROW_INDICES,
)
from detector_yolo import BBoxResult


class MaskGenerator:
    def __init__(self, output_width: int, output_height: int, config: RuntimeConfig):
        self.width = output_width
        self.height = output_height
        self.config = config

    def _get_contour_points(self, landmarks: np.ndarray, indices: list) -> np.ndarray:
        """Extract landmark points for given indices, scaled to output resolution."""
        pts = landmarks[indices]  # (N, 2) normalized
        pts[:, 0] *= self.width
        pts[:, 1] *= self.height
        return pts.astype(np.int32)

    def _post_process(self, mask: np.ndarray) -> np.ndarray:
        """Apply dilation, blur, homography warp, and convert to BGRA."""
        dilation = self.config.dilation_pixels
        if dilation > 0:
            # For large dilation, downscale first to reduce work
            if dilation > 30:
                scale = 0.25
                small = cv2.resize(mask, None, fx=scale, fy=scale,
                                   interpolation=cv2.INTER_NEAREST)
                k_size = max(3, int(dilation * scale) | 1)  # ensure odd
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                                   (k_size, k_size))
                small = cv2.dilate(small, kernel, iterations=1)
                mask = cv2.resize(small, (self.width, self.height),
                                  interpolation=cv2.INTER_LINEAR)
            else:
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
                iterations = max(1, dilation // 2)
                mask = cv2.dilate(mask, kernel, iterations=iterations)

        blur = self.config.blur_radius
        if blur > 0:
            if blur % 2 == 0:
                blur += 1
            mask = cv2.GaussianBlur(mask, (blur, blur), 0)

        if self.config.homography_matrix is not None:
            mask = cv2.warpPerspective(
                mask, self.config.homography_matrix, (self.width, self.height)
            )

        return cv2.merge([mask, mask, mask, np.full_like(mask, 255)])

    def generate(self, faces: List[FaceResult]) -> np.ndarray:
        """Generate a BGRA mask from MediaPipe face landmarks."""
        mask = np.zeros((self.height, self.width), dtype=np.uint8)

        for face in faces:
            lm = face.landmarks.copy()

            if self.config.mask_mode == "eyes":
                contours = [LEFT_EYE_INDICES, RIGHT_EYE_INDICES]
            elif self.config.mask_mode == "face":
                contours = [FACE_OVAL_INDICES]
            elif self.config.mask_mode == "eyes_and_brows":
                contours = [
                    LEFT_EYE_INDICES,
                    RIGHT_EYE_INDICES,
                    LEFT_EYEBROW_INDICES,
                    RIGHT_EYEBROW_INDICES,
                ]
            else:
                contours = [LEFT_EYE_INDICES, RIGHT_EYE_INDICES]

            polys = [self._get_contour_points(lm, idx) for idx in contours]
            cv2.fillPoly(mask, polys, 255)

        return self._post_process(mask)

    def generate_from_bbox(self, bboxes: List[BBoxResult]) -> np.ndarray:
        """Generate a BGRA mask from YOLO bounding boxes (head ellipse)."""
        mask = np.zeros((self.height, self.width), dtype=np.uint8)

        for bbox in bboxes:
            cx = int((bbox.x + bbox.w / 2) * self.width)
            cy = int((bbox.y + bbox.h / 2) * self.height)
            ax = int((bbox.w / 2) * self.width * 1.1)
            ay = int((bbox.h / 2) * self.height * 1.1)
            if ax > 0 and ay > 0:
                cv2.ellipse(mask, (cx, cy), (ax, ay), 0, 0, 360, 255, -1)

        return self._post_process(mask)
