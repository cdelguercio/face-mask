"""ArUco grid generation and detection for automatic calibration.

Generates a grid of ArUco markers at known projector coordinates.
When the camera sees markers (e.g. on a white board held in the performance area),
each detected marker ID maps to a known projector position, giving automatic
camera-to-projector point correspondences for homography computation.
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional

# Use 6x6 dictionary — 250 unique IDs, robust detection at distance
ARUCO_DICT = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
ARUCO_PARAMS = cv2.aruco.DetectorParameters()
ARUCO_DETECTOR = cv2.aruco.ArucoDetector(ARUCO_DICT, ARUCO_PARAMS)


class ArucoGrid:
    """Generates and detects an ArUco marker grid for projector calibration."""

    def __init__(self, proj_width: int, proj_height: int, grid_cols: int = 6, grid_rows: int = 4):
        self.proj_width = proj_width
        self.proj_height = proj_height
        self.set_grid_size(grid_cols, grid_rows)

    def set_grid_size(self, cols: int, rows: int):
        """Update grid dimensions and rebuild the marker-to-position lookup."""
        self.grid_cols = max(2, cols)
        self.grid_rows = max(2, rows)

        # Each marker ID maps to its center position in normalized projector coords (0-1)
        # Markers are placed with padding so they don't touch the edges
        self._id_to_proj: dict[int, Tuple[float, float]] = {}
        marker_id = 0
        for row in range(self.grid_rows):
            for col in range(self.grid_cols):
                # Center of each cell, with half-cell padding from edges
                x_norm = (col + 0.5) / self.grid_cols
                y_norm = (row + 0.5) / self.grid_rows
                self._id_to_proj[marker_id] = (x_norm, y_norm)
                marker_id += 1

        self._grid_image: Optional[np.ndarray] = None  # cached

    def get_grid_bgra(self) -> np.ndarray:
        """Generate the ArUco grid image at projector resolution (BGRA)."""
        if self._grid_image is not None:
            return self._grid_image

        w, h = self.proj_width, self.proj_height
        # White background — important so the markers have good contrast
        img = np.full((h, w, 3), 255, dtype=np.uint8)

        cell_w = w / self.grid_cols
        cell_h = h / self.grid_rows
        # Marker size: ~60% of the smaller cell dimension
        marker_px = int(min(cell_w, cell_h) * 0.55)
        # Ensure even size for clean marker generation
        marker_px = max(marker_px, 20) | 1 ^ 1  # make even

        for marker_id, (x_norm, y_norm) in self._id_to_proj.items():
            if marker_id >= 250:
                break  # dictionary limit

            # Generate marker image
            marker_img = cv2.aruco.generateImageMarker(ARUCO_DICT, marker_id, marker_px)
            marker_bgr = cv2.cvtColor(marker_img, cv2.COLOR_GRAY2BGR)

            # Center position in pixels
            cx = int(x_norm * w)
            cy = int(y_norm * h)

            # Top-left corner
            x1 = cx - marker_px // 2
            y1 = cy - marker_px // 2
            x2 = x1 + marker_px
            y2 = y1 + marker_px

            # Clip to image bounds
            if x1 < 0 or y1 < 0 or x2 > w or y2 > h:
                continue

            img[y1:y2, x1:x2] = marker_bgr

            # Draw ID label below marker
            label_y = min(y2 + 20, h - 5)
            cv2.putText(img, str(marker_id), (cx - 10, label_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (128, 128, 128), 1)

        # Draw subtle grid lines for reference
        for col in range(1, self.grid_cols):
            x = int(col * cell_w)
            cv2.line(img, (x, 0), (x, h), (220, 220, 220), 1)
        for row in range(1, self.grid_rows):
            y = int(row * cell_h)
            cv2.line(img, (0, y), (w, y), (220, 220, 220), 1)

        self._grid_image = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
        return self._grid_image

    def detect(self, frame_bgr: np.ndarray) -> List[Tuple[int, np.ndarray, Tuple[float, float]]]:
        """Detect ArUco markers in a camera frame.

        Returns list of (marker_id, corners, projector_norm_xy) for each
        marker whose ID is in our grid. corners is shape (4, 2) in pixel coords.
        """
        corners_list, ids, _ = ARUCO_DETECTOR.detectMarkers(frame_bgr)

        if ids is None or len(ids) == 0:
            return []

        results = []
        for i, marker_id in enumerate(ids.flatten()):
            marker_id = int(marker_id)
            if marker_id not in self._id_to_proj:
                continue
            corners = corners_list[i].reshape(4, 2)  # 4 corner points in pixel coords
            proj_pt = self._id_to_proj[marker_id]
            results.append((marker_id, corners, proj_pt))

        return results

    def get_correspondences(self, detections: list) -> Tuple[np.ndarray, np.ndarray]:
        """Convert detections to camera/projector point arrays for findHomography.

        Returns (camera_pts, projector_pts) both as Nx2 float32 arrays.
        Camera points are the center of each detected marker in pixel coords.
        Projector points are in normalized (0-1) coords.
        """
        cam_pts = []
        proj_pts = []

        for marker_id, corners, proj_pt in detections:
            # Use marker center as the camera point
            center = corners.mean(axis=0)
            cam_pts.append(center)
            proj_pts.append(proj_pt)

        return (
            np.array(cam_pts, dtype=np.float32),
            np.array(proj_pts, dtype=np.float32),
        )

    def draw_detections(self, preview: np.ndarray, detections: list,
                        cam_w: int, cam_h: int, preview_w: int, preview_h: int):
        """Draw detected markers on the preview image with ID labels."""
        sx = preview_w / cam_w
        sy = preview_h / cam_h

        for marker_id, corners, proj_pt in detections:
            # Scale corners to preview coordinates
            pts = corners.copy()
            pts[:, 0] *= sx
            pts[:, 1] *= sy
            pts_int = pts.astype(np.int32)

            # Draw marker outline
            cv2.polylines(preview, [pts_int], isClosed=True, color=(0, 255, 0), thickness=2)

            # Draw center dot
            cx = int(pts[:, 0].mean())
            cy = int(pts[:, 1].mean())
            cv2.circle(preview, (cx, cy), 4, (0, 255, 0), -1)

            # Label with ID and projector coords
            label = f"#{marker_id} ({proj_pt[0]:.2f},{proj_pt[1]:.2f})"
            cv2.putText(preview, label, (cx + 8, cy - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
