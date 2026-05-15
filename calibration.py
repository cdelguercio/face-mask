"""Camera-to-projector calibration using homography transform.

Workflow:
  1. Press 'c' to enter calibration mode
  2. Click a face in the camera preview -> snaps to nearest detected face center
  3. Click the corresponding position in the projector grid window
  4. Repeat 4+ times -> homography is auto-computed
  5. Press 'c' again to exit calibration mode and go live
  6. Arrow keys nudge the transform offset during live mode
  7. Press 'r' to reset all calibration data
"""

import json
import os
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import cv2
import numpy as np

from aruco_grid import ArucoGrid
from detector import FaceResult

# Path for persisting calibration data
CALIBRATION_FILE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "calibration.json"
)

GRID_WINDOW_NAME = "Projector Grid (click to place point)"

# Calibration sub-modes
CAL_MODE_MANUAL = "manual"
CAL_MODE_ARUCO = "aruco"


@dataclass
class PointPair:
    """A single camera -> projector correspondence point."""
    camera_pt: Tuple[float, float]      # normalized 0-1
    projector_pt: Tuple[float, float]    # normalized 0-1


class Calibration:
    """Manages camera-to-projector calibration via homography.

    Supports two calibration sub-modes:
      - manual: click face in camera, click position in projector grid
      - aruco: project ArUco grid, detect markers on a held-up board, capture

    Coordinates are stored normalized (0-1) and converted to pixel
    coordinates only when computing the homography for a specific
    output resolution.
    """

    def __init__(self, output_width: int, output_height: int):
        self.output_width = output_width
        self.output_height = output_height

        # Separate point lists per calibration mode
        self.manual_pairs: List[PointPair] = []
        self.aruco_pairs: List[PointPair] = []
        self.homography: Optional[np.ndarray] = None
        self.nudge_x: float = 0.0  # pixels in output space
        self.nudge_y: float = 0.0

        # State machine for calibration mode
        self.active: bool = False
        self.cal_mode: str = CAL_MODE_MANUAL  # "manual" or "aruco"
        self._waiting_for_projector: bool = False
        self._pending_camera_pt: Optional[Tuple[float, float]] = None

        # Current detected faces (updated each frame for snap-to-face)
        self._current_faces: List[FaceResult] = []

        # ArUco grid
        self.aruco = ArucoGrid(output_width, output_height)
        self._aruco_detections: list = []  # latest frame's detections

        # ROI for ArUco mode — normalized (0-1) coords, None = no ROI (use all)
        self.roi: Optional[Tuple[float, float, float, float]] = None  # (x1, y1, x2, y2)
        self._roi_dragging: bool = False
        self._roi_start: Optional[Tuple[float, float]] = None

        # Try to load saved calibration
        self._load()

    @property
    def pairs(self) -> List[PointPair]:
        """Return the active point list for the current calibration mode."""
        if self.cal_mode == CAL_MODE_ARUCO:
            return self.aruco_pairs
        return self.manual_pairs

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def toggle(self, show_projector_window: bool = True):
        """Toggle calibration mode on/off."""
        self.active = not self.active
        if self.active:
            self._waiting_for_projector = False
            self._pending_camera_pt = None
            self._aruco_detections = []
            if show_projector_window and self.cal_mode == CAL_MODE_MANUAL:
                cv2.namedWindow(GRID_WINDOW_NAME, cv2.WINDOW_NORMAL)
                cv2.resizeWindow(GRID_WINDOW_NAME, 960, 540)
                cv2.setMouseCallback(GRID_WINDOW_NAME, self._on_projector_click)
        else:
            # Clear pending state so overlays don't linger
            self._waiting_for_projector = False
            self._pending_camera_pt = None
            self._aruco_detections = []
            try:
                cv2.destroyWindow(GRID_WINDOW_NAME)
            except cv2.error:
                pass

    def set_cal_mode(self, mode: str, show_projector_window: bool = True):
        """Switch between 'manual' and 'aruco' calibration sub-modes."""
        if mode == self.cal_mode:
            return
        was_active = self.active
        if was_active:
            self.toggle(show_projector_window=show_projector_window)  # turn off
        self.cal_mode = mode
        # Invalidate cached aruco grid when switching
        self.aruco._grid_image = None
        if was_active:
            self.toggle(show_projector_window=show_projector_window)  # turn back on in new mode

    def set_aruco_grid_size(self, cols: int, rows: int):
        """Update ArUco grid density."""
        self.aruco.set_grid_size(cols, rows)

    def update_aruco(self, frame_bgr: np.ndarray, cam_w: int, cam_h: int):
        """Run ArUco detection on the current camera frame.

        Call this each frame when in aruco calibration mode.
        Applies ROI filter if a ROI is set.
        """
        if not self.active or self.cal_mode != CAL_MODE_ARUCO:
            self._aruco_detections = []
            return
        raw = self.aruco.detect(frame_bgr)
        self._aruco_detections = self._filter_by_roi(raw, cam_w, cam_h)

    def capture_aruco(self, cam_w: int, cam_h: int):
        """Capture current ArUco detections as calibration point pairs.

        Adds all currently detected markers as correspondences and recomputes
        the homography. Can be called multiple times to accumulate points
        as the board is moved to different positions.

        cam_w, cam_h: dimensions of the camera frame (for normalizing pixel coords).
        """
        if not self._aruco_detections:
            print("[Calibration] No ArUco markers detected — nothing to capture.")
            return 0

        cam_pts, proj_pts = self.aruco.get_correspondences(self._aruco_detections)

        added = 0
        for cam_pt, proj_pt in zip(cam_pts, proj_pts):
            # Normalize camera points to 0-1
            self.pairs.append(PointPair(
                camera_pt=(float(cam_pt[0] / cam_w), float(cam_pt[1] / cam_h)),
                projector_pt=(float(proj_pt[0]), float(proj_pt[1])),
            ))
            added += 1

        self._recompute()
        self._save()
        print(f"[Calibration] Captured {added} ArUco points (total: {len(self.pairs)} pts)")
        return added

    def get_aruco_detections(self) -> list:
        """Return the latest ArUco detections (filtered by ROI if set)."""
        return self._aruco_detections

    # ------------------------------------------------------------------
    # ROI for ArUco mode
    # ------------------------------------------------------------------

    def on_roi_mouse(self, event: int, x_norm: float, y_norm: float):
        """Handle mouse events for ROI drawing in ArUco mode.

        Called from the camera preview mouse handler.
        Returns True if the event was consumed (ArUco ROI mode).
        """
        if not self.active or self.cal_mode != CAL_MODE_ARUCO:
            return False

        if event == cv2.EVENT_LBUTTONDOWN:
            self._roi_dragging = True
            self._roi_start = (x_norm, y_norm)
            self.roi = None  # clear while dragging
            return True
        elif event == cv2.EVENT_MOUSEMOVE and self._roi_dragging:
            # Live preview of the ROI rectangle
            sx, sy = self._roi_start
            self.roi = (min(sx, x_norm), min(sy, y_norm),
                        max(sx, x_norm), max(sy, y_norm))
            return True
        elif event == cv2.EVENT_LBUTTONUP and self._roi_dragging:
            self._roi_dragging = False
            sx, sy = self._roi_start
            x1, y1 = min(sx, x_norm), min(sy, y_norm)
            x2, y2 = max(sx, x_norm), max(sy, y_norm)
            # Require minimum size to avoid accidental tiny boxes
            if (x2 - x1) > 0.02 and (y2 - y1) > 0.02:
                self.roi = (x1, y1, x2, y2)
                print(f"[Calibration] ROI set: ({x1:.2f},{y1:.2f}) -> ({x2:.2f},{y2:.2f})")
            else:
                self.roi = None  # too small, treat as click-to-clear
                print("[Calibration] ROI cleared (too small or single click)")
            return True

        return False

    def clear_roi(self):
        """Remove the ROI filter."""
        self.roi = None
        self._roi_dragging = False
        self._roi_start = None
        print("[Calibration] ROI cleared")

    def _filter_by_roi(self, detections: list, cam_w: int, cam_h: int) -> list:
        """Filter ArUco detections to only those whose center is inside the ROI."""
        if self.roi is None:
            return detections

        x1, y1, x2, y2 = self.roi
        # ROI is in normalized coords, marker centers are in pixel coords
        filtered = []
        for marker_id, corners, proj_pt in detections:
            center = corners.mean(axis=0)
            cx_norm = center[0] / cam_w
            cy_norm = center[1] / cam_h
            if x1 <= cx_norm <= x2 and y1 <= cy_norm <= y2:
                filtered.append((marker_id, corners, proj_pt))
        return filtered

    def draw_roi(self, preview: np.ndarray, preview_w: int, preview_h: int):
        """Draw the ROI rectangle on the preview if set."""
        if self.roi is None or not self.active or self.cal_mode != CAL_MODE_ARUCO:
            return
        x1, y1, x2, y2 = self.roi
        pt1 = (int(x1 * preview_w), int(y1 * preview_h))
        pt2 = (int(x2 * preview_w), int(y2 * preview_h))
        # Semi-transparent overlay outside ROI
        overlay = preview.copy()
        cv2.rectangle(overlay, (0, 0), (preview_w, preview_h), (0, 0, 0), -1)
        cv2.rectangle(overlay, pt1, pt2, (0, 0, 0), -1)  # cut out ROI area
        # Darken outside ROI
        mask = np.zeros((preview_h, preview_w), dtype=np.uint8)
        cv2.rectangle(mask, pt1, pt2, 255, -1)
        preview[mask == 0] = (preview[mask == 0] * 0.4).astype(np.uint8)
        # Draw ROI border
        cv2.rectangle(preview, pt1, pt2, (0, 255, 255), 2)
        cv2.putText(preview, "ROI (right-click to clear)",
                    (pt1[0], pt1[1] - 8), cv2.FONT_HERSHEY_SIMPLEX,
                    0.45, (0, 255, 255), 1)

    # ------------------------------------------------------------------
    # Face tracking
    # ------------------------------------------------------------------

    def update_faces(self, faces: List[FaceResult]):
        """Feed current frame's face detections for snap-to-face."""
        self._current_faces = faces

    def on_camera_click(self, x_norm: float, y_norm: float):
        """Handle a click on the camera preview (normalized 0-1 coords).

        Snaps to the nearest detected face center if any faces are present.
        Uses the raw click position if no face is found (for manual override).
        """
        if not self.active:
            return

        snapped = self._snap_to_face(x_norm, y_norm)
        self._pending_camera_pt = snapped if snapped is not None else (x_norm, y_norm)
        self._waiting_for_projector = True
        if snapped is None:
            print("[Calibration] No face detected — using raw click position.")

    def nudge(self, dx: float, dy: float):
        """Shift the transform by (dx, dy) pixels in output space."""
        self.nudge_x += dx
        self.nudge_y += dy
        self._recompute()
        self._save()

    def clear_points(self):
        """Clear calibration points for the current mode only."""
        mode = self.cal_mode
        if mode == CAL_MODE_ARUCO:
            self.aruco_pairs.clear()
            print("[Calibration] ArUco points cleared.")
        else:
            self.manual_pairs.clear()
            self._waiting_for_projector = False
            self._pending_camera_pt = None
            print("[Calibration] Manual points cleared.")
        self._recompute()
        self._save()

    def reset(self):
        """Clear all calibration data (both modes)."""
        self.manual_pairs.clear()
        self.aruco_pairs.clear()
        self.homography = None
        self.nudge_x = 0.0
        self.nudge_y = 0.0
        self._waiting_for_projector = False
        self._pending_camera_pt = None
        self.roi = None
        self._save()

    def get_homography(self) -> Optional[np.ndarray]:
        """Return the current homography matrix (with nudge applied), or None."""
        return self.homography

    def get_status(self) -> str:
        """Return a short status string for the preview overlay."""
        if self.active:
            if self.cal_mode == CAL_MODE_ARUCO:
                n_det = len(self._aruco_detections)
                roi_tag = " [ROI]" if self.roi else " (drag ROI to filter)"
                return f"CAL-ARUCO: {n_det} markers{roi_tag} | 's'=capture ({len(self.pairs)} pts)"
            if self._waiting_for_projector:
                return f"CAL: click projector grid ({len(self.pairs)} pts)"
            return f"CAL: click a face ({len(self.pairs)} pts)"
        if self.homography is not None:
            nudge = ""
            if self.nudge_x != 0 or self.nudge_y != 0:
                nudge = f" nudge({self.nudge_x:+.0f},{self.nudge_y:+.0f})"
            return f"LIVE {len(self.pairs)}pts{nudge}"
        return "NO CAL"

    def draw_camera_overlay(self, preview: np.ndarray, preview_w: int, preview_h: int):
        """Draw calibration points on the camera preview image."""
        # Show live nose tracking when in calibration mode
        if self.active:
            for face in self._current_faces:
                nx = int(face.landmarks[1, 0] * preview_w)
                ny = int(face.landmarks[1, 1] * preview_h)
                cv2.circle(preview, (nx, ny), 8, (255, 0, 255), 2)  # magenta crosshair
                cv2.line(preview, (nx - 12, ny), (nx + 12, ny), (255, 0, 255), 1)
                cv2.line(preview, (nx, ny - 12), (nx, ny + 12), (255, 0, 255), 1)

        for pair in self.pairs:
            cx = int(pair.camera_pt[0] * preview_w)
            cy = int(pair.camera_pt[1] * preview_h)
            cv2.circle(preview, (cx, cy), 6, (0, 255, 0), 2)  # green ring
            cv2.circle(preview, (cx, cy), 2, (0, 255, 0), -1)  # green dot

        # Highlight pending camera point
        if self._waiting_for_projector and self._pending_camera_pt is not None:
            cx = int(self._pending_camera_pt[0] * preview_w)
            cy = int(self._pending_camera_pt[1] * preview_h)
            cv2.circle(preview, (cx, cy), 10, (0, 255, 255), 2)  # yellow ring
            cv2.putText(
                preview, "-> click projector grid",
                (cx + 14, cy + 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1,
            )

        # Draw projector outline in camera space (inverse homography)
        if self.active and self.homography is not None:
            self._draw_projector_outline(preview, preview_w, preview_h)

    def _render_grid(self, w: int, h: int) -> np.ndarray:
        """Render the projector grid at the given resolution."""
        grid = np.zeros((h, w, 3), dtype=np.uint8)
        scale = w / 960.0  # scale factor vs the reference 960px width

        cols, rows = 8, 6
        for i in range(cols + 1):
            x = int(i * w / cols)
            color = (80, 80, 80) if i % 2 else (50, 50, 50)
            cv2.line(grid, (x, 0), (x, h), color, max(1, int(scale)))
            if i < cols:
                label = f"{i * self.output_width // cols}"
                cv2.putText(grid, label, (x + 4, int(20 * scale)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4 * scale, (120, 120, 120), max(1, int(scale)))

        for j in range(rows + 1):
            y = int(j * h / rows)
            color = (80, 80, 80) if j % 2 else (50, 50, 50)
            cv2.line(grid, (0, y), (w, y), color, max(1, int(scale)))
            if j < rows:
                label = f"{j * self.output_height // rows}"
                cv2.putText(grid, label, (4, y + int(20 * scale)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4 * scale, (120, 120, 120), max(1, int(scale)))

        cv2.rectangle(grid, (0, 0), (w - 1, h - 1), (100, 100, 100), max(1, int(scale)))

        for i, pair in enumerate(self.pairs):
            px = int(pair.projector_pt[0] * w)
            py = int(pair.projector_pt[1] * h)
            r = max(7, int(7 * scale))
            cv2.circle(grid, (px, py), r, (0, 0, 255), max(2, int(2 * scale)))
            cv2.circle(grid, (px, py), max(2, int(2 * scale)), (0, 0, 255), -1)
            cv2.putText(grid, str(i + 1), (px + r + 3, py + 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5 * scale, (0, 0, 255), max(1, int(scale)))

        title = f"Projector Space ({self.output_width}x{self.output_height})"
        cv2.putText(grid, title, (int(10 * scale), h - int(15 * scale)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5 * scale, (200, 200, 200), max(1, int(scale)))

        if self._waiting_for_projector:
            cv2.putText(grid, "CLICK where the face appears here",
                        (w // 2 - int(180 * scale), int(35 * scale)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6 * scale, (0, 255, 255), max(1, int(2 * scale)))

        return grid

    def get_projector_frame(self) -> Optional[np.ndarray]:
        """Return a full-resolution BGRA image for Spout output, or None if not calibrating."""
        if not self.active:
            return None
        if self.cal_mode == CAL_MODE_ARUCO:
            return self.aruco.get_grid_bgra()
        grid_bgr = self._render_grid(self.output_width, self.output_height)
        return cv2.cvtColor(grid_bgr, cv2.COLOR_BGR2BGRA)

    def get_projector_preview(self, width: int, height: int) -> np.ndarray:
        """Return a BGR projector preview for the control UI."""
        if self.active and self.cal_mode == CAL_MODE_ARUCO:
            return cv2.resize(self.aruco.get_grid_bgra()[:, :, :3], (width, height))
        return self._render_grid(width, height)

    def draw_projector_grid(self):
        """Draw and show the projector coordinate grid window (scaled for clicking).

        Only used in manual mode — aruco mode doesn't need a click target.
        """
        if not self.active or self.cal_mode != CAL_MODE_MANUAL:
            return
        grid = self._render_grid(960, 540)
        cv2.imshow(GRID_WINDOW_NAME, grid)

    # ------------------------------------------------------------------
    # Mouse callbacks
    # ------------------------------------------------------------------

    def _on_projector_click(self, event, x, y, flags, param):
        """Mouse callback for the projector grid window."""
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        if not self._waiting_for_projector or self._pending_camera_pt is None:
            return

        # Window is 960x540; normalize to 0-1
        proj_x = x / 960.0
        proj_y = y / 540.0

        self.pairs.append(PointPair(
            camera_pt=self._pending_camera_pt,
            projector_pt=(proj_x, proj_y),
        ))

        self._waiting_for_projector = False
        self._pending_camera_pt = None

        self._recompute()
        self._save()

    def on_projector_click(self, x_norm: float, y_norm: float):
        """Handle a normalized projector-space click from a GUI preview."""
        if not self._waiting_for_projector or self._pending_camera_pt is None:
            return

        self.pairs.append(PointPair(
            camera_pt=self._pending_camera_pt,
            projector_pt=(x_norm, y_norm),
        ))

        self._waiting_for_projector = False
        self._pending_camera_pt = None

        self._recompute()
        self._save()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _draw_projector_outline(self, preview: np.ndarray, preview_w: int, preview_h: int):
        """Draw the projector display boundary in camera space using the inverse homography."""
        try:
            H_inv = np.linalg.inv(self.homography)
        except np.linalg.LinAlgError:
            return

        # Projector corners in normalized coords (0-1)
        proj_corners = np.array([
            [0, 0], [1, 0], [1, 1], [0, 1]
        ], dtype=np.float32)

        # Convert to output pixel coords for the homography
        proj_px = proj_corners.copy()
        proj_px[:, 0] *= self.output_width
        proj_px[:, 1] *= self.output_height

        # Apply inverse homography to get camera pixel coords
        proj_px_h = np.hstack([proj_px, np.ones((4, 1), dtype=np.float32)])
        cam_px_h = (H_inv @ proj_px_h.T).T
        # Perspective divide
        cam_px = cam_px_h[:, :2] / cam_px_h[:, 2:3]

        # Convert from output-space to normalized, then to preview coords
        cam_norm = cam_px.copy()
        cam_norm[:, 0] /= self.output_width
        cam_norm[:, 1] /= self.output_height
        cam_preview = cam_norm.copy()
        cam_preview[:, 0] *= preview_w
        cam_preview[:, 1] *= preview_h

        pts = cam_preview.astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(preview, [pts], isClosed=True, color=(0, 0, 255), thickness=2)

        # Label corners
        labels = ["TL", "TR", "BR", "BL"]
        for i, label in enumerate(labels):
            x, y = int(cam_preview[i, 0]), int(cam_preview[i, 1])
            cv2.putText(preview, label, (x + 5, y - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)

    def _snap_to_face(self, click_x: float, click_y: float) -> Optional[Tuple[float, float]]:
        """Find the nearest detected face center to the click point.

        Returns normalized (x, y) of the face center, or None if no faces.
        """
        if not self._current_faces:
            return None

        best_dist = float("inf")
        best_center = None

        for face in self._current_faces:
            lm = face.landmarks  # (478, 2) normalized
            # Use nose tip (landmark 1) as a more intuitive face center
            cx = float(lm[1, 0])
            cy = float(lm[1, 1])
            dist = (cx - click_x) ** 2 + (cy - click_y) ** 2
            if dist < best_dist:
                best_dist = dist
                best_center = (cx, cy)

        return best_center

    def _recompute(self):
        """Recompute homography from current point pairs + nudge."""
        if len(self.pairs) < 4:
            self.homography = None
            return

        # Build source (camera) and destination (projector) point arrays
        # in output pixel coordinates
        src = np.array([
            [p.camera_pt[0] * self.output_width,
             p.camera_pt[1] * self.output_height]
            for p in self.pairs
        ], dtype=np.float32)

        dst = np.array([
            [p.projector_pt[0] * self.output_width,
             p.projector_pt[1] * self.output_height]
            for p in self.pairs
        ], dtype=np.float32)

        H, mask = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
        if H is None:
            self.homography = None
            return

        # Apply nudge as a translation composed with the homography
        if self.nudge_x != 0 or self.nudge_y != 0:
            T = np.array([
                [1, 0, self.nudge_x],
                [0, 1, self.nudge_y],
                [0, 0, 1],
            ], dtype=np.float64)
            H = T @ H

        self.homography = H

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _serialize_pairs(self, pairs: List[PointPair]) -> list:
        return [{"camera": list(p.camera_pt), "projector": list(p.projector_pt)} for p in pairs]

    def _deserialize_pairs(self, data: list) -> List[PointPair]:
        return [PointPair(camera_pt=tuple(p["camera"]), projector_pt=tuple(p["projector"])) for p in data]

    def _save(self):
        """Save calibration data to JSON (both modes stored separately)."""
        data = {
            "manual_pairs": self._serialize_pairs(self.manual_pairs),
            "aruco_pairs": self._serialize_pairs(self.aruco_pairs),
            "active_mode": self.cal_mode,
            "nudge_x": self.nudge_x,
            "nudge_y": self.nudge_y,
        }
        try:
            with open(CALIBRATION_FILE, "w") as f:
                json.dump(data, f, indent=2)
        except OSError as e:
            print(f"[Calibration] Could not save: {e}")

    def _load(self):
        """Load calibration data from JSON if it exists."""
        if not os.path.exists(CALIBRATION_FILE):
            return
        try:
            with open(CALIBRATION_FILE, "r") as f:
                data = json.load(f)

            # Support both old format (single "pairs") and new format (separate lists)
            if "manual_pairs" in data:
                self.manual_pairs = self._deserialize_pairs(data.get("manual_pairs", []))
                self.aruco_pairs = self._deserialize_pairs(data.get("aruco_pairs", []))
                saved_mode = data.get("active_mode", CAL_MODE_MANUAL)
                self.cal_mode = saved_mode
            else:
                # Legacy: single "pairs" list → treat as manual
                self.manual_pairs = self._deserialize_pairs(data.get("pairs", []))
                self.aruco_pairs = []

            self.nudge_x = data.get("nudge_x", 0.0)
            self.nudge_y = data.get("nudge_y", 0.0)

            self._recompute()
            active = self.pairs
            if self.homography is not None:
                print(f"[Calibration] Loaded {len(active)} {self.cal_mode} point pairs from disk.")
            elif active:
                print(f"[Calibration] Loaded {len(active)} {self.cal_mode} pairs (need 4+ for homography).")
        except (json.JSONDecodeError, KeyError, OSError) as e:
            print(f"[Calibration] Could not load: {e}")
