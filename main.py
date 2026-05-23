import os

# Silence MediaPipe's noisy native telemetry. MediaPipe's C++ pipeline ships
# with a Google Play Logs ("clearcut") uploader that fails when run outside
# an Android device and spams stderr every 60 seconds:
#   E0000 ... portable_clearcut_uploader.cc:90] Failed to send to clearcut...
# Suppressing absl/glog/TF logging below FATAL silences it. These MUST be
# set before mediapipe / any detector import.
os.environ.setdefault("GLOG_minloglevel", "3")
os.environ.setdefault("ABSL_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("GRPC_VERBOSITY", "NONE")

import argparse
import sys
import time

# Pin python.exe to the discrete GPU before anything touches OpenGL.
# This is a no-op after the first run; on the first run it writes a registry
# preference and re-execs the process so Spout shares textures on the right
# adapter. Must run before pygame / SpoutGL / cv2 import any GL.
from gpu_preference import ensure_high_performance_gpu

ensure_high_performance_gpu()

import cv2
import numpy as np

try:
    import dearpygui.dearpygui as dpg
except ImportError:
    dpg = None

from calibration import CAL_MODE_ARUCO, CAL_MODE_MANUAL, Calibration
from camera import CameraManager
from config import (
    CONTOUR_MODES,
    DETECTOR_MODES,
    MAX_FACES,
    NDI_SENDER_NAME,
    NUDGE_STEP,
    OUTPUT_HEIGHT,
    OUTPUT_WIDTH,
    SPOUT_SENDER_NAME,
    RuntimeConfig,
)
from detector_manager import FaceDetectionManager
from mask_generator import MaskGenerator
from output_ndi import NDIOutput
from output_spout import SpoutOutput

MASK_MODES = ["eyes", "face", "eyes_and_brows"]

PREVIEW_W, PREVIEW_H = 960, 540
GRID_W, GRID_H = 480, 270
MASK_INSET_W, MASK_INSET_H = 220, 124

PREVIEW_TEXTURE = "preview_texture"
GRID_TEXTURE = "grid_texture"
PREVIEW_IMAGE = "preview_image"
GRID_IMAGE = "grid_image"
STATUS_TEXT = "status_text"
DETAIL_TEXT = "detail_text"

CONTOUR_COLORS = {
    "eyes": [(0, 255, 0), (0, 255, 0)],
    "face": [(0, 255, 255)],
    "eyes_and_brows": [(0, 255, 0), (0, 255, 0), (255, 0, 255), (255, 0, 255)],
}


def parse_args():
    parser = argparse.ArgumentParser(description="Real-time face mask for projection")
    parser.add_argument("--camera", type=int, default=None,
                        help="Camera device index (auto-detect if omitted)")
    parser.add_argument("--cam-width", type=int, default=1280,
                        help="Camera capture width (default: 1280)")
    parser.add_argument("--cam-height", type=int, default=720,
                        help="Camera capture height (default: 720)")
    parser.add_argument("--out-width", type=int, default=OUTPUT_WIDTH,
                        help=f"Output/Spout width (default: {OUTPUT_WIDTH})")
    parser.add_argument("--out-height", type=int, default=OUTPUT_HEIGHT,
                        help=f"Output/Spout height (default: {OUTPUT_HEIGHT})")
    return parser.parse_args()


def _require_dearpygui():
    if dpg is not None:
        return

    print()
    print("=" * 70)
    print("Dear PyGui is not installed")
    print("=" * 70)
    print()
    print("Install it in this venv:")
    print("  pip install dearpygui")
    print()
    sys.exit(1)


def _blank_rgba(width: int, height: int) -> np.ndarray:
    return np.zeros((height, width, 4), dtype=np.float32)


def _bgr_to_texture(frame_bgr: np.ndarray, width: int, height: int) -> np.ndarray:
    if frame_bgr.shape[1] != width or frame_bgr.shape[0] != height:
        frame_bgr = cv2.resize(frame_bgr, (width, height))
    frame_rgba = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGBA)
    return (frame_rgba.astype(np.float32) / 255.0).ravel()


def _bgra_to_bgr(frame_bgra: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(frame_bgra, cv2.COLOR_BGRA2BGR)


def _draw_preview_overlays(display, detection, mask_mode, cam_w, cam_h):
    for bbox in detection.bboxes:
        x1 = int(bbox.x * cam_w)
        y1 = int(bbox.y * cam_h)
        x2 = int((bbox.x + bbox.w) * cam_w)
        y2 = int((bbox.y + bbox.h) * cam_h)
        cv2.rectangle(display, (x1, y1), (x2, y2), (255, 255, 0), 2)
        cv2.putText(display, bbox.label, (x1, y1 - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

    indices_list = CONTOUR_MODES[mask_mode]
    colors = CONTOUR_COLORS[mask_mode]

    for face in detection.faces:
        lm = face.landmarks

        for indices, color in zip(indices_list, colors):
            pts = lm[indices].copy()
            pts[:, 0] *= cam_w
            pts[:, 1] *= cam_h
            pts = pts.astype(np.int32).reshape(-1, 1, 2)
            cv2.polylines(display, [pts], isClosed=True, color=color, thickness=2)

        for idx in range(468, min(478, len(lm))):
            x = int(lm[idx, 0] * cam_w)
            y = int(lm[idx, 1] * cam_h)
            cv2.circle(display, (x, y), 2, (0, 0, 255), -1)


class FaceMaskApp:
    def __init__(self, args):
        self.args = args
        self.config = RuntimeConfig()
        self.running = True

        print("Initializing face detectors...")
        self.detection_manager = FaceDetectionManager(max_faces=MAX_FACES)

        self.out_w = args.out_width
        self.out_h = args.out_height
        self.mask_gen = MaskGenerator(self.out_w, self.out_h, self.config)

        print("Initializing outputs...")
        self.spout = SpoutOutput(SPOUT_SENDER_NAME, self.out_w, self.out_h)
        self.ndi = NDIOutput(NDI_SENDER_NAME)

        if not self.spout.enabled and not self.ndi.enabled:
            print("WARNING: No outputs available. Running in preview-only mode.")

        self.cam_manager = CameraManager(width=args.cam_width, height=args.cam_height)
        self.cam_manager.discover()
        self.cam_manager.open_default(preferred_index=args.camera)

        self.calibration = Calibration(self.out_w, self.out_h)
        self.last_valid_mask = None
        self.tracking_lost_time = None
        self.last_frame_shape = None

        self.fps_timer = time.perf_counter()
        self.frame_count = 0
        self.fps = 0.0
        self.detect_ms = 0.0
        self.total_ms = 0.0
        self.detect_src = "LOST"
        self.n_detect = 0

        self.preview_dragging = False
        self.latest_preview = np.zeros((PREVIEW_H, PREVIEW_W, 3), dtype=np.uint8)

    # ------------------------------------------------------------------
    # GUI setup
    # ------------------------------------------------------------------

    def setup_gui(self):
        dpg.create_context()

        with dpg.texture_registry(show=False):
            dpg.add_dynamic_texture(
                PREVIEW_W,
                PREVIEW_H,
                _blank_rgba(PREVIEW_W, PREVIEW_H).ravel(),
                tag=PREVIEW_TEXTURE,
            )
            dpg.add_dynamic_texture(
                GRID_W,
                GRID_H,
                _blank_rgba(GRID_W, GRID_H).ravel(),
                tag=GRID_TEXTURE,
            )

        with dpg.window(tag="main_window", label="Face Mask Projection Blocker"):
            with dpg.group(horizontal=True):
                with dpg.group():
                    dpg.add_text("", tag=STATUS_TEXT)
                    dpg.add_image(PREVIEW_TEXTURE, tag=PREVIEW_IMAGE)
                    dpg.add_spacer(height=8)
                    dpg.add_text("Projector grid")
                    dpg.add_image(GRID_TEXTURE, tag=GRID_IMAGE)
                    dpg.add_text("", tag=DETAIL_TEXT)

                with dpg.child_window(width=360, autosize_y=True, border=True):
                    dpg.add_text("Controls")
                    dpg.add_separator()

                    dpg.add_combo(
                        DETECTOR_MODES,
                        default_value=self.config.detector_mode,
                        label="Detector",
                        callback=self.on_detector_mode,
                    )
                    dpg.add_combo(
                        MASK_MODES,
                        default_value=self.config.mask_mode,
                        label="Mask Mode",
                        callback=self.on_mask_mode,
                    )
                    dpg.add_slider_int(
                        label="Box Confidence",
                        default_value=int(self.detection_manager.get_bbox_confidence() * 100),
                        min_value=0,
                        max_value=100,
                        callback=self.on_box_confidence,
                    )
                    dpg.add_slider_int(
                        label="Blur",
                        default_value=self.config.blur_radius,
                        min_value=1,
                        max_value=51,
                        callback=self.on_blur,
                    )
                    dpg.add_slider_int(
                        label="Dilation",
                        default_value=self.config.dilation_pixels,
                        min_value=0,
                        max_value=200,
                        callback=self.on_dilation,
                    )
                    dpg.add_slider_int(
                        label="Hold ms",
                        default_value=self.config.hold_ms,
                        min_value=0,
                        max_value=10000,
                        callback=self.on_hold,
                    )

                    camera_items = [
                        f"{i}: {cam.label}" for i, cam in enumerate(self.cam_manager.cameras)
                    ]
                    if camera_items:
                        dpg.add_combo(
                            camera_items,
                            default_value=camera_items[self.cam_manager.current_idx],
                            label="Camera",
                            callback=self.on_camera,
                        )

                    dpg.add_spacer(height=8)
                    dpg.add_text("Calibration")
                    dpg.add_separator()
                    dpg.add_combo(
                        [CAL_MODE_MANUAL, CAL_MODE_ARUCO],
                        default_value=self.calibration.cal_mode,
                        label="Mode",
                        callback=self.on_cal_mode,
                    )
                    dpg.add_slider_int(
                        label="ArUco Cols",
                        default_value=6,
                        min_value=2,
                        max_value=12,
                        callback=self.on_aruco_cols,
                    )
                    dpg.add_button(label="Start/Stop Calibration", callback=self.toggle_calibration)
                    dpg.add_button(label="Capture ArUco", callback=self.capture_aruco)
                    dpg.add_button(label="Clear Points", callback=self.clear_points)
                    dpg.add_button(label="Reset Calibration", callback=self.reset_calibration)

                    dpg.add_spacer(height=8)
                    dpg.add_text("Nudge")
                    with dpg.group(horizontal=True):
                        dpg.add_button(label="Left", width=78,
                                       callback=lambda *args: self.nudge(-NUDGE_STEP, 0))
                        dpg.add_button(label="Right", width=78,
                                       callback=lambda *args: self.nudge(NUDGE_STEP, 0))
                    with dpg.group(horizontal=True):
                        dpg.add_button(label="Up", width=78,
                                       callback=lambda *args: self.nudge(0, -NUDGE_STEP))
                        dpg.add_button(label="Down", width=78,
                                       callback=lambda *args: self.nudge(0, NUDGE_STEP))

                    dpg.add_spacer(height=8)
                    dpg.add_button(label="Quit", callback=self.quit)

        with dpg.handler_registry():
            dpg.add_mouse_click_handler(button=dpg.mvMouseButton_Left, callback=self.on_left_click)
            dpg.add_mouse_release_handler(button=dpg.mvMouseButton_Left, callback=self.on_left_release)
            dpg.add_mouse_drag_handler(button=dpg.mvMouseButton_Left, callback=self.on_left_drag)
            dpg.add_mouse_click_handler(button=dpg.mvMouseButton_Right, callback=self.on_right_click)
            dpg.add_key_press_handler(key=dpg.mvKey_Q, callback=lambda *args: self.quit())
            dpg.add_key_press_handler(key=dpg.mvKey_C, callback=lambda *args: self.toggle_calibration())
            dpg.add_key_press_handler(key=dpg.mvKey_S, callback=lambda *args: self.capture_aruco())
            dpg.add_key_press_handler(key=dpg.mvKey_X, callback=lambda *args: self.clear_points())
            dpg.add_key_press_handler(key=dpg.mvKey_R, callback=lambda *args: self.reset_calibration())
            dpg.add_key_press_handler(key=dpg.mvKey_Left, callback=lambda *args: self.nudge(-NUDGE_STEP, 0))
            dpg.add_key_press_handler(key=dpg.mvKey_Right, callback=lambda *args: self.nudge(NUDGE_STEP, 0))
            dpg.add_key_press_handler(key=dpg.mvKey_Up, callback=lambda *args: self.nudge(0, -NUDGE_STEP))
            dpg.add_key_press_handler(key=dpg.mvKey_Down, callback=lambda *args: self.nudge(0, NUDGE_STEP))

        dpg.create_viewport(title="Face Mask Projection Blocker", width=1360, height=900)
        dpg.setup_dearpygui()
        dpg.set_primary_window("main_window", True)
        dpg.show_viewport()

    # ------------------------------------------------------------------
    # GUI callbacks
    # ------------------------------------------------------------------

    def on_detector_mode(self, sender, app_data, user_data=None):
        self.config.detector_mode = app_data

    def on_mask_mode(self, sender, app_data, user_data=None):
        self.config.mask_mode = app_data

    def on_box_confidence(self, sender, app_data, user_data=None):
        self.detection_manager.set_bbox_confidence(app_data / 100.0)

    def on_blur(self, sender, app_data, user_data=None):
        self.config.blur_radius = max(1, int(app_data) | 1)

    def on_dilation(self, sender, app_data, user_data=None):
        self.config.dilation_pixels = int(app_data)

    def on_hold(self, sender, app_data, user_data=None):
        self.config.hold_ms = int(app_data)

    def on_camera(self, sender, app_data, user_data=None):
        try:
            list_index = int(app_data.split(":", 1)[0])
        except (ValueError, AttributeError):
            return
        # Defer the actual switch to the main loop — opening/releasing a
        # cv2.VideoCapture from a GUI callback while the main loop is in the
        # middle of cap.read() can crash inside flaky virtual camera drivers
        # (e.g. NVIDIA VCAM). The main loop drains _pending_switch in
        # check_switch() before its next read.
        self.cam_manager.request_switch(list_index)

    def on_cal_mode(self, sender, app_data, user_data=None):
        self.calibration.set_cal_mode(app_data, show_projector_window=False)

    def on_aruco_cols(self, sender, app_data, user_data=None):
        density = max(2, int(app_data))
        rows = max(2, int(density * 2 / 3))
        self.calibration.set_aruco_grid_size(density, rows)

    def toggle_calibration(self, *args):
        self.calibration.toggle(show_projector_window=False)
        mode = "ON" if self.calibration.active else "OFF"
        print(f"[Calibration] Mode {mode}")

    def capture_aruco(self, *args):
        if (
            self.calibration.active
            and self.calibration.cal_mode == CAL_MODE_ARUCO
            and self.last_frame_shape is not None
        ):
            cam_h, cam_w = self.last_frame_shape[:2]
            self.calibration.capture_aruco(cam_w, cam_h)

    def clear_points(self, *args):
        self.calibration.clear_points()

    def reset_calibration(self, *args):
        self.calibration.reset()
        print("[Calibration] Reset - all points and nudge cleared.")

    def nudge(self, dx: float, dy: float, *args):
        self.calibration.nudge(dx, dy)

    def quit(self, *args):
        self.running = False
        dpg.stop_dearpygui()

    # ------------------------------------------------------------------
    # Mouse interaction
    # ------------------------------------------------------------------

    def _norm_from_item(self, item, clamp: bool = False):
        item_min = dpg.get_item_rect_min(item)
        item_max = dpg.get_item_rect_max(item)
        if item_min is None or item_max is None:
            return None

        mx, my = dpg.get_mouse_pos(local=False)
        x = (mx - item_min[0]) / max(1, item_max[0] - item_min[0])
        y = (my - item_min[1]) / max(1, item_max[1] - item_min[1])

        if clamp:
            x = min(1.0, max(0.0, x))
            y = min(1.0, max(0.0, y))
        elif x < 0.0 or y < 0.0 or x > 1.0 or y > 1.0:
            return None

        return x, y

    def on_left_click(self, *args):
        preview_pt = self._norm_from_item(PREVIEW_IMAGE)
        if preview_pt is not None:
            x_norm, y_norm = preview_pt
            self.preview_dragging = True
            if self.calibration.on_roi_mouse(cv2.EVENT_LBUTTONDOWN, x_norm, y_norm):
                return
            self.calibration.on_camera_click(x_norm, y_norm)
            return

        grid_pt = self._norm_from_item(GRID_IMAGE)
        if grid_pt is not None and self.calibration.active:
            self.calibration.on_projector_click(*grid_pt)

    def on_left_drag(self, *args):
        if not self.preview_dragging:
            return
        preview_pt = self._norm_from_item(PREVIEW_IMAGE, clamp=True)
        if preview_pt is None:
            return
        self.calibration.on_roi_mouse(cv2.EVENT_MOUSEMOVE, *preview_pt)

    def on_left_release(self, *args):
        if not self.preview_dragging:
            return
        self.preview_dragging = False
        preview_pt = self._norm_from_item(PREVIEW_IMAGE, clamp=True)
        if preview_pt is not None:
            self.calibration.on_roi_mouse(cv2.EVENT_LBUTTONUP, *preview_pt)

    def on_right_click(self, *args):
        if self._norm_from_item(PREVIEW_IMAGE) is not None:
            self.calibration.clear_roi()

    # ------------------------------------------------------------------
    # Frame loop
    # ------------------------------------------------------------------

    def run(self):
        self.setup_gui()
        try:
            while self.running and dpg.is_dearpygui_running():
                self.process_frame()
                dpg.render_dearpygui_frame()
        except KeyboardInterrupt:
            print("\nInterrupted.")
        finally:
            self.cleanup()

    def process_frame(self):
        # Apply any pending camera switch requested via the GUI dropdown.
        # This must happen on the main loop thread, not the GUI callback,
        # to avoid racing cap.release()/cap.set() against an in-flight read().
        self.cam_manager.check_switch()

        ret, frame = self.cam_manager.read()
        if not ret:
            self.show_no_camera()
            return

        self.last_frame_shape = frame.shape
        frame_start = time.perf_counter()

        detect_start = time.perf_counter()
        detection = self.detection_manager.detect(frame, self.config.detector_mode)
        self.detect_ms = (time.perf_counter() - detect_start) * 1000.0

        self.calibration.update_faces(detection.faces)
        cam_h, cam_w = frame.shape[:2]
        self.calibration.update_aruco(frame, cam_w, cam_h)
        self.config.homography_matrix = self.calibration.get_homography()

        mask_bgra = self.make_mask(detection)
        self.send_outputs(mask_bgra)

        preview = self.make_preview(frame, detection, mask_bgra)
        grid = self.calibration.get_projector_preview(GRID_W, GRID_H)

        dpg.set_value(PREVIEW_TEXTURE, _bgr_to_texture(preview, PREVIEW_W, PREVIEW_H))
        dpg.set_value(GRID_TEXTURE, _bgr_to_texture(grid, GRID_W, GRID_H))

        self.total_ms = (time.perf_counter() - frame_start) * 1000.0
        self.update_fps()
        self.update_status()

    def make_mask(self, detection):
        if detection.faces:
            mask_bgra = self.mask_gen.generate(detection.faces)
            self.last_valid_mask = mask_bgra.copy()
            self.tracking_lost_time = None
            self.detect_src = detection.source
            self.n_detect = len(detection.faces)
            return mask_bgra

        if detection.bboxes:
            mask_bgra = self.mask_gen.generate_from_bbox(detection.bboxes)
            self.last_valid_mask = mask_bgra.copy()
            self.tracking_lost_time = None
            self.detect_src = detection.source
            self.n_detect = len(detection.bboxes)
            return mask_bgra

        now = time.perf_counter()
        if self.tracking_lost_time is None:
            self.tracking_lost_time = now

        elapsed_ms = (now - self.tracking_lost_time) * 1000.0
        if self.last_valid_mask is not None and elapsed_ms < self.config.hold_ms:
            fade = 1.0 - (elapsed_ms / self.config.hold_ms) if self.config.hold_ms > 0 else 0.0
            mask_bgra = self.last_valid_mask.copy()
            rgb = mask_bgra[:, :, :3].astype(np.float32)
            mask_bgra[:, :, :3] = (
                rgb * fade + 255.0 * (1.0 - fade)
            ).astype(np.uint8)
            self.detect_src = "HOLD"
            self.n_detect = 0
            return mask_bgra

        self.detect_src = "LOST"
        self.n_detect = 0
        return np.full((self.out_h, self.out_w, 4), 255, dtype=np.uint8)

    def send_outputs(self, mask_bgra):
        projector_grid = self.calibration.get_projector_frame()
        spout_frame = projector_grid if projector_grid is not None else mask_bgra

        try:
            self.spout.send(spout_frame)
        except Exception as e:
            print(f"[Spout] Send error: {e}")

        try:
            self.ndi.send(mask_bgra)
        except Exception as e:
            print(f"[NDI] Send error: {e}")

    def make_preview(self, frame, detection, mask_bgra):
        cam_h, cam_w = frame.shape[:2]
        display = frame.copy()
        _draw_preview_overlays(display, detection, self.config.mask_mode, cam_w, cam_h)

        status = self.status_line()
        cv2.rectangle(display, (0, 0), (cam_w, 35), (0, 0, 0), -1)
        cv2.putText(display, status, (10, 25), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (0, 255, 0), 2)

        preview = cv2.resize(display, (PREVIEW_W, PREVIEW_H))
        self.calibration.draw_camera_overlay(preview, PREVIEW_W, PREVIEW_H)
        self.calibration.draw_roi(preview, PREVIEW_W, PREVIEW_H)

        aruco_dets = self.calibration.get_aruco_detections()
        if aruco_dets:
            self.calibration.aruco.draw_detections(
                preview, aruco_dets, cam_w, cam_h, PREVIEW_W, PREVIEW_H
            )

        mask_inset = cv2.resize(_bgra_to_bgr(mask_bgra), (MASK_INSET_W, MASK_INSET_H))
        cv2.rectangle(mask_inset, (0, 0), (MASK_INSET_W - 1, MASK_INSET_H - 1), (0, 255, 0), 1)
        y1 = PREVIEW_H - MASK_INSET_H - 10
        y2 = PREVIEW_H - 10
        x1 = PREVIEW_W - MASK_INSET_W - 10
        x2 = PREVIEW_W - 10
        preview[y1:y2, x1:x2] = mask_inset
        return preview

    def update_fps(self):
        self.frame_count += 1
        elapsed = time.perf_counter() - self.fps_timer
        if elapsed >= 1.0:
            self.fps = self.frame_count / elapsed
            self.frame_count = 0
            self.fps_timer = time.perf_counter()

    def status_line(self):
        cal_status = self.calibration.get_status()
        return (f"{self.n_detect} [{self.detect_src}] | {self.config.mask_mode} | "
                f"det:{self.detect_ms:.0f}ms tot:{self.total_ms:.0f}ms | "
                f"{self.fps:.0f} FPS | {cal_status}")

    def update_status(self):
        dpg.set_value(STATUS_TEXT, self.status_line())
        dpg.set_value(
            DETAIL_TEXT,
            f"Camera: {self.cam_manager.get_camera_label()} | "
            f"SCRFD: {self.detection_manager.get_provider_label('scrfd')} | "
            f"YOLO: {self.detection_manager.get_provider_label('yolo')} | "
            f"Spout: {'on' if self.spout.enabled else 'off'} | "
            f"NDI: {'on' if self.ndi.enabled else 'off'}",
        )

    def show_no_camera(self):
        placeholder = np.zeros((PREVIEW_H, PREVIEW_W, 3), dtype=np.uint8)
        msg = "No camera - waiting for reconnect..."
        cv2.putText(placeholder, msg, (PREVIEW_W // 2 - 250, PREVIEW_H // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        dpg.set_value(PREVIEW_TEXTURE, _bgr_to_texture(placeholder, PREVIEW_W, PREVIEW_H))
        dpg.set_value(STATUS_TEXT, msg)

    def cleanup(self):
        print("Cleaning up...")
        self.cam_manager.release()
        self.detection_manager.close()
        self.spout.release()
        self.ndi.release()
        try:
            cv2.destroyAllWindows()
        except cv2.error:
            pass
        dpg.destroy_context()


def main():
    _require_dearpygui()
    args = parse_args()
    FaceMaskApp(args).run()


if __name__ == "__main__":
    main()
