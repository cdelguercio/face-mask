import argparse
import sys
import time

import cv2
import numpy as np

from calibration import Calibration
from camera import CameraManager
from config import (
    CAMERA_HEIGHT,
    CAMERA_WIDTH,
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
from detector import FaceDetector
from detector_yolo import BBoxResult, YoloFaceDetector, YOLO_AVAILABLE
from mask_generator import MaskGenerator
from output_ndi import NDIOutput
from output_spout import SpoutOutput

MASK_MODES = ["eyes", "face", "eyes_and_brows"]
WINDOW_NAME = "Face Mask Preview"


def on_mode_change(val, config):
    config.mask_mode = MASK_MODES[val]


def on_blur_change(val, config):
    config.blur_radius = max(1, val | 1)  # ensure odd and >= 1


def on_dilation_change(val, config):
    config.dilation_pixels = val


def on_hold_change(val, config):
    config.hold_ms = val


def on_detector_mode_change(val, config):
    config.detector_mode = DETECTOR_MODES[val]


def on_yolo_conf_change(val, yolo_detector):
    yolo_detector.confidence = val / 100.0


def window_closed() -> bool:
    """Check if the OpenCV window was closed by the user (X button)."""
    try:
        return cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1
    except cv2.error:
        return True


PREVIEW_W, PREVIEW_H = 1280, 720

CONTOUR_COLORS = {
    "eyes": [(0, 255, 0), (0, 255, 0)],
    "face": [(0, 255, 255)],
    "eyes_and_brows": [(0, 255, 0), (0, 255, 0), (255, 0, 255), (255, 0, 255)],
}


def _draw_preview_overlays(display, faces, yolo_bboxes, mask_mode, cam_w, cam_h):
    for bbox in yolo_bboxes:
        x1 = int(bbox.x * cam_w)
        y1 = int(bbox.y * cam_h)
        x2 = int((bbox.x + bbox.w) * cam_w)
        y2 = int((bbox.y + bbox.h) * cam_h)
        cv2.rectangle(display, (x1, y1), (x2, y2), (255, 255, 0), 2)
        cv2.putText(display, "YOLO", (x1, y1 - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

    indices_list = CONTOUR_MODES[mask_mode]
    colors = CONTOUR_COLORS[mask_mode]

    for face in faces:
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


def _make_camera_click_handler(calibration):
    """Return a mouse callback for the camera preview window."""
    def _on_mouse(event, x, y, flags, param):
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        x_norm = x / PREVIEW_W
        y_norm = y / PREVIEW_H
        calibration.on_camera_click(x_norm, y_norm)
    return _on_mouse


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


def main():
    args = parse_args()
    config = RuntimeConfig()

    # Initialize components
    print("Initializing face detector...")
    detector = FaceDetector(max_faces=MAX_FACES)

    # YOLO fallback detector (gracefully optional)
    yolo_detector = YoloFaceDetector(enabled=YOLO_AVAILABLE)

    out_w, out_h = args.out_width, args.out_height
    mask_gen = MaskGenerator(out_w, out_h, config)

    print("Initializing outputs...")
    spout = SpoutOutput(SPOUT_SENDER_NAME, out_w, out_h)
    ndi = NDIOutput(NDI_SENDER_NAME)

    if not spout.enabled and not ndi.enabled:
        print("WARNING: No outputs available. Running in preview-only mode.")

    # Camera setup — discover, open, and prepare selector
    cam_manager = CameraManager(width=args.cam_width, height=args.cam_height)
    cam_manager.discover()
    cam_manager.open_default(preferred_index=args.camera)

    # Calibration system
    calibration = Calibration(out_w, out_h)

    # AUTOSIZE prevents user from stretching the aspect ratio
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(WINDOW_NAME, _make_camera_click_handler(calibration))

    # Camera selector trackbar
    cam_manager.attach_to_window(WINDOW_NAME)

    cv2.createTrackbar("Mask Mode", WINDOW_NAME, 0, 2,
                        lambda v: on_mode_change(v, config))
    cv2.createTrackbar("Blur", WINDOW_NAME, config.blur_radius, 51,
                        lambda v: on_blur_change(v, config))
    cv2.createTrackbar("Dilation", WINDOW_NAME, config.dilation_pixels, 200,
                        lambda v: on_dilation_change(v, config))
    cv2.createTrackbar("Hold ms", WINDOW_NAME, config.hold_ms, 2000,
                        lambda v: on_hold_change(v, config))
    cv2.createTrackbar("Detector", WINDOW_NAME, DETECTOR_MODES.index(config.detector_mode), 2,
                        lambda v: on_detector_mode_change(v, config))
    cv2.createTrackbar("YOLO Conf", WINDOW_NAME, int(yolo_detector.confidence * 100), 100,
                        lambda v: on_yolo_conf_change(v, yolo_detector))

    # Print control legend to console
    print()
    print("=== Controls ===")
    print("Mask Mode:  0=eyes  1=face  2=eyes+brows")
    print("Detector:   0=MP+YOLO  1=MP only  2=YOLO only")
    if len(cam_manager.cameras) > 1:
        for i, cam in enumerate(cam_manager.cameras):
            print(f"Camera {i}:   {cam.label}")
    print()

    print("Running. Press 'q' to quit, 'c' for calibration, 'r' to reset cal.")
    print("Arrow keys nudge the calibration transform.")
    fps_timer = time.perf_counter()
    frame_count = 0
    fps = 0.0

    # Hold-last-position state
    last_valid_mask = None  # last successfully generated BGRA mask
    tracking_lost_time = None  # perf_counter timestamp when tracking was lost

    try:
        while True:
            # Check if window was closed
            if window_closed():
                print("Window closed.")
                break

            # Handle camera hot-swap from trackbar
            cam_manager.check_switch()

            ret, frame = cam_manager.read()
            if not ret:
                print("Camera read failed.")
                break

            frame_start = time.perf_counter()

            # Detect faces based on detector mode
            faces = []
            yolo_bboxes = []
            mode = config.detector_mode

            detect_start = time.perf_counter()
            if mode == "mp+yolo":
                faces = detector.detect(frame)
                if not faces:
                    yolo_bboxes = yolo_detector.detect(frame)
            elif mode == "mp":
                faces = detector.detect(frame)
            elif mode == "yolo":
                yolo_bboxes = yolo_detector.detect(frame)
            detect_ms = (time.perf_counter() - detect_start) * 1000.0

            # Feed faces to calibration for snap-to-face
            calibration.update_faces(faces)

            # Update the runtime config homography from calibration
            config.homography_matrix = calibration.get_homography()

            # Generate mask based on what was detected
            if faces:
                # MediaPipe landmarks available — use normal mode
                mask_bgra = mask_gen.generate(faces)
                last_valid_mask = mask_bgra.copy()
                tracking_lost_time = None
            elif yolo_bboxes:
                # YOLO fallback — head ellipse from bounding box
                mask_bgra = mask_gen.generate_from_bbox(yolo_bboxes)
                last_valid_mask = mask_bgra.copy()
                tracking_lost_time = None
            else:
                # No detections from either detector — hold logic
                now = time.perf_counter()
                if tracking_lost_time is None:
                    tracking_lost_time = now

                elapsed_ms = (now - tracking_lost_time) * 1000.0
                if last_valid_mask is not None and elapsed_ms < config.hold_ms:
                    # Within hold window: fade out as we approach timeout
                    if config.hold_ms > 0:
                        fade = 1.0 - (elapsed_ms / config.hold_ms)
                    else:
                        fade = 0.0
                    # Fade held mask toward white (pass-through)
                    mask_bgra = last_valid_mask.copy()
                    rgb = mask_bgra[:, :, :3].astype(np.float32)
                    mask_bgra[:, :, :3] = (
                        rgb * fade + 255.0 * (1.0 - fade)
                    ).astype(np.uint8)
                else:
                    # Past hold window or no previous mask — output white (pass-through)
                    mask_bgra = np.full(
                        (OUTPUT_HEIGHT, OUTPUT_WIDTH, 4), 255, dtype=np.uint8
                    )

            # Send to outputs — grid to projector during calibration, mask otherwise
            projector_grid = calibration.get_projector_frame()
            spout_frame = projector_grid if projector_grid is not None else mask_bgra
            try:
                spout.send(spout_frame)
            except Exception as e:
                print(f"[Spout] Send error: {e}")

            try:
                ndi.send(mask_bgra)
            except Exception as e:
                print(f"[NDI] Send error: {e}")

            # Build preview: camera feed with detection overlays + mask inset
            preview_w, preview_h = PREVIEW_W, PREVIEW_H
            cam_h, cam_w = frame.shape[:2]

            # Determine detection source label for status bar
            if faces:
                detect_src = "MP"
            elif yolo_bboxes:
                detect_src = "YOLO"
            elif last_valid_mask is not None and tracking_lost_time is not None:
                elapsed_ms = (time.perf_counter() - tracking_lost_time) * 1000.0
                if elapsed_ms < config.hold_ms:
                    detect_src = "HOLD"
                else:
                    detect_src = "LOST"
            else:
                detect_src = "LOST"

            # Draw detection contours on the camera frame
            display = frame.copy()
            _draw_preview_overlays(display, faces, yolo_bboxes,
                                   config.mask_mode, cam_w, cam_h)

            total_ms = (time.perf_counter() - frame_start) * 1000.0

            # Status bar at top
            cam_label = cam_manager.get_camera_label()
            cal_status = calibration.get_status()
            n_detect = len(faces) if faces else len(yolo_bboxes)
            status = (f"{n_detect} [{detect_src}] | {config.mask_mode} | "
                      f"det:{detect_ms:.0f}ms tot:{total_ms:.0f}ms | "
                      f"{fps:.0f} FPS | {cal_status}")
            # Dark background strip for readability
            cv2.rectangle(display, (0, 0), (cam_w, 35), (0, 0, 0), -1)
            cv2.putText(display, status, (10, 25), cv2.FONT_HERSHEY_SIMPLEX,
                        0.55, (0, 255, 0), 2)

            preview = cv2.resize(display, (preview_w, preview_h))

            # Draw calibration overlays on the preview
            calibration.draw_camera_overlay(preview, preview_w, preview_h)

            # Mask inset (bottom-right corner)
            inset_w, inset_h = 240, 135
            mask_inset = cv2.resize(mask_bgra[:, :, :3], (inset_w, inset_h))
            cv2.rectangle(mask_inset, (0, 0), (inset_w - 1, inset_h - 1), (0, 255, 0), 1)
            preview[preview_h - inset_h - 10 : preview_h - 10,
                    preview_w - inset_w - 10 : preview_w - 10] = mask_inset

            # FPS counter
            frame_count += 1
            elapsed = time.perf_counter() - fps_timer
            if elapsed >= 1.0:
                fps = frame_count / elapsed
                frame_count = 0
                fps_timer = time.perf_counter()

            cv2.imshow(WINDOW_NAME, preview)

            # Draw projector grid window when in calibration mode
            calibration.draw_projector_grid()

            # Keyboard handling
            key = cv2.waitKeyEx(1)
            if key != -1:
                print(f"[KEY] {key}")
            if key == ord("q"):
                break
            elif key == ord("c"):
                calibration.toggle()
                mode = "ON" if calibration.active else "OFF"
                print(f"[Calibration] Mode {mode}")
            elif key == ord("r"):
                calibration.reset()
                print("[Calibration] Reset — all points cleared.")
            elif key == 2424832:  # left arrow (Windows)
                calibration.nudge(-NUDGE_STEP, 0)
            elif key == 2555904:  # right arrow (Windows)
                calibration.nudge(NUDGE_STEP, 0)
            elif key == 2490368:  # up arrow (Windows)
                calibration.nudge(0, -NUDGE_STEP)
            elif key == 2621440:  # down arrow (Windows)
                calibration.nudge(0, NUDGE_STEP)

    except KeyboardInterrupt:
        print("\nInterrupted.")
    finally:
        print("Cleaning up...")
        cam_manager.release()
        detector.close()
        spout.release()
        ndi.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
