# Face Mask Projection Blocker

## What it does

Real-time face and eye detection that generates a white-on-black mask, sent via Spout (and optionally NDI) to Resolume Arena/Avenue. Resolume ingests the mask as a layer to block projection on a performer's face and eyes, preventing projectors from blinding them.

## Requirements

- Python 3.13+ (tested on Windows 11)
- NVIDIA GPU recommended (RTX series for CUDA acceleration)
- Spout (for Resolume integration) — https://spout.zeal.co
- Resolume Arena/Avenue

## Installation

```bash
# Create virtual environment
python -m venv .venv
.venv\Scripts\activate

# Install dependencies
pip install opencv-python mediapipe numpy ultralytics pygame PyOpenGL SpoutGL

# Install PyTorch with CUDA (for GPU-accelerated YOLO)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

# Download MediaPipe face landmarker model
curl -L -o face_landmarker_v2_with_blendshapes.task https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task
```

## Windows Camera Permissions

Settings > Privacy & Security > Camera:

1. "Camera access" master toggle — ON
2. "Let desktop apps access your camera" — ON
3. Find python3.XX.exe and enable it

## Usage

```bash
python main.py           # auto-detect camera
python main.py 3         # use camera index 3
```

## Controls

- UI trackbars: Mask Mode (eyes/face/eyes+brows), Blur, Dilation, Hold ms, Detector (MP+YOLO/MP/YOLO), YOLO Confidence, Camera selector
- `c` — toggle calibration mode
- `r` — reset calibration
- Arrow keys — nudge calibration transform
- `q` — quit

## Calibration (camera to projector transform)

1. Press `c` to enter calibration mode (grid appears on projector via Spout)
2. Click performer's face in camera preview
3. Click matching position on projector grid
4. Repeat 4+ times from different positions
5. Press `c` to go live — homography is computed
6. Arrow keys for fine-tuning nudge
7. Calibration auto-saves to calibration.json

## Architecture

- `main.py` — main loop, UI, preview rendering
- `detector.py` — MediaPipe face landmark detection (CPU)
- `detector_yolo.py` — YOLOv12 face detection (GPU/CUDA)
- `mask_generator.py` — generates white-on-black mask from detections
- `calibration.py` — camera-to-projector homography with persistence
- `camera.py` — camera discovery, permissions, hot-swap
- `output_spout.py` — Spout sender for Resolume
- `output_ndi.py` — NDI sender (optional)
- `config.py` — constants and runtime configuration
