import argparse
import sys
import time

import cv2
import numpy as np

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


def on_bbox_conf_change(val, detection_manager):
    detection_manager.set_bbox_confidence(val / 100.0)


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


def _make_camera_click_handler(calibration):
    """Return a mouse callback for the camera preview window.

    In ArUco mode: left-click-drag draws a ROI, right-click clears it.
    In manual mode: left-click places a calibration point.
    """
    def _on_mouse(event, x, y, flags, param):
        x_norm = x / PREVIEW_W
        y_norm = y / PREVIEW_H

        # Right-click clears ROI in ArUco mode
        if event == cv2.EVENT_RBUTTONDOWN:
            calibration.clear_roi()
            return

        # Try ROI handling first (consumes left-click events in ArUco mode)
        if calibration.on_roi_mouse(event, x_norm, y_norm):
            return

        # Manual mode click-to-place
        if event == cv2.EVENT_LBUTTONDOWN:
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
    print("Initializing face detectors...")
    detection_manager = FaceDetectionManager(max_faces=MAX_FACES)

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
    cv2.createTrackbar("Hold ms", WINDOW_NAME, config.hold_ms, 10000,
                        lambda v: on_hold_change(v, config))
    cv2.createTrackbar("Detector", WINDOW_NAME, DETECTOR_MODES.index(config.detector_mode),
                        len(DETECTOR_MODES) - 1,
                        lambda v: on_detector_mode_change(v, config))
    cv2.createTrackbar("Box Conf", WINDOW_NAME,
                        int(detection_manager.get_bbox_confidence() * 100), 100,
                        lambda v: on_bbox_conf_change(v, detection_manager))

    # Calibration mode trackbar: 0=manual, 1=aruco — sync with loaded state
    CAL_MODES = [CAL_MODE_MANUAL, CAL_MODE_ARUCO]
    def on_cal_mode_change(val):
        calibration.set_cal_mode(CAL_MODES[val])
    initial_cal_idx = CAL_MODES.index(calibration.cal_mode) if calibration.cal_mode in CAL_MODES else 0
    cv2.createTrackbar("Cal Mode", WINDOW_NAME, initial_cal_idx, 1,
                        on_cal_mode_change)

    # ArUco grid density (cols x rows, linked slider for simplicity)
    def on_aruco_density_change(val):
        density = max(2, val)
        # Keep roughly 3:2 aspect ratio for the grid
        rows = max(2, int(density * 2 / 3))
        calibration.set_aruco_grid_size(density, rows)
    cv2.createTrackbar("ArUco Cols", WINDOW_NAME, 6, 12,
                        on_aruco_density_change)

    # Print control legend to console
    print()
    print("=== Controls ===")
    print("Mask Mode:  0=eyes  1=face  2=eyes+brows")
    print("Detector:   " + "  ".join(
        f"{i}={mode.upper()}" for i, mode in enumerate(DETECTOR_MODES)
    ))
    print("Cal Mode:   0=manual  1=aruco")
    if len(cam_manager.cameras) > 1:
        for i, cam in enumerate(cam_manager.cameras):
            print(f"Camera {i}:   {cam.label}")
    print()

    print("Running. Press 'q' to quit, 'c' for calibration.")
    print("  'x' = clear current mode's points, 'r' = reset all")
    print("  Arrow keys = nudge transform")
    print("  ArUco mode: 's' = capture, drag ROI, right-click = clear ROI")
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
                # Show "no camera" placeholder and keep looping
                placeholder = np.zeros((PREVIEW_H, PREVIEW_W, 3), dtype=np.uint8)
                msg = "No camera — waiting for reconnect..."
                cv2.putText(placeholder, msg, (PREVIEW_W // 2 - 250, PREVIEW_H // 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                cv2.imshow(WINDOW_NAME, placeholder)
                key = cv2.waitKeyEx(100)  # slower polling when no camera
                if key == ord("q") or window_closed():
                    break
                continue

            frame_start = time.perf_counter()

            detect_start = time.perf_counter()
            detection = detection_manager.detect(frame, config.detector_mode)
            detect_ms = (time.perf_counter() - detect_start) * 1000.0

            # Feed faces to calibration for snap-to-face
            calibration.update_faces(detection.faces)

            # Run ArUco detection when in aruco calibration mode
            cam_h, cam_w = frame.shape[:2]
            calibration.update_aruco(frame, cam_w, cam_h)

            # Update the runtime config homography from calibration
            config.homography_matrix = calibration.get_homography()

            # Generate mask based on what was detected
            if detection.faces:
                # MediaPipe landmarks available — use normal mode
                mask_bgra = mask_gen.generate(detection.faces)
                last_valid_mask = mask_bgra.copy()
                tracking_lost_time = None
            elif detection.bboxes:
                # YOLO fallback — head ellipse from bounding box
                mask_bgra = mask_gen.generate_from_bbox(detection.bboxes)
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
            if detection.found:
                detect_src = detection.source
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
            _draw_preview_overlays(display, detection, config.mask_mode, cam_w, cam_h)

            total_ms = (time.perf_counter() - frame_start) * 1000.0

            # Status bar at top
            cam_label = cam_manager.get_camera_label()
            cal_status = calibration.get_status()
            n_detect = len(detection.faces) if detection.faces else len(detection.bboxes)
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

            # Draw ROI and ArUco detections on the preview
            calibration.draw_roi(preview, preview_w, preview_h)
            aruco_dets = calibration.get_aruco_detections()
            if aruco_dets:
                calibration.aruco.draw_detections(
                    preview, aruco_dets, cam_w, cam_h, preview_w, preview_h
                )

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
            elif key == ord("s"):
                if calibration.active and calibration.cal_mode == CAL_MODE_ARUCO:
                    cam_h_cur, cam_w_cur = frame.shape[:2]
                    calibration.capture_aruco(cam_w_cur, cam_h_cur)
            elif key == ord("x"):
                calibration.clear_points()
            elif key == ord("r"):
                calibration.reset()
                print("[Calibration] Reset — all points and nudge cleared.")
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
        detection_manager.close()
        spout.release()
        ndi.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
