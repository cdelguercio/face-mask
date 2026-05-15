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
pip install opencv-contrib-python mediapipe numpy ultralytics pygame PyOpenGL SpoutGL dearpygui

# Optional SCRFD backend
# First grab MSVC (latest) and Windows SDK from https://visualstudio.microsoft.com/visual-cpp-build-tools/
pip install insightface onnxruntime-gpu

# Native CUDA runtime for SCRFD GPU acceleration
# 1. Install NVIDIA CUDA Toolkit 12.x from https://developer.nvidia.com/cuda-downloads
# 2. Install NVIDIA cuDNN 9.x for CUDA 12.x from https://developer.nvidia.com/cudnn
# 3. Add these folders to your Windows PATH, adjusting version paths as needed:
#    C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.x\bin
#    <your cuDNN install>\bin
# 4. Restart the terminal, then verify the DLLs are discoverable:
#    where cublasLt64_12.dll
#    where cudnn64_9.dll
# 5. Verify ONNX Runtime can use CUDA:
#    python -c "import onnxruntime as ort; print(ort.get_available_providers())"
#
# Note: get_available_providers() only proves the CUDA provider is installed.
# It can still fail at runtime if CUDA/cuDNN DLLs are not loadable by Python.
# If SCRFD prints that cudnn64_9.dll is missing, install cuDNN 9.x and add its
# bin folder to PATH, then restart the terminal. NVIDIA's cuDNN installer may
# use a nested folder like C:\Program Files\NVIDIA\CUDNN\v9.21\bin\12.9\x64;
# the app will try to discover this layout automatically.

# Install PyTorch with CUDA (for GPU-accelerated YOLO)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

# If YOLO says ultralytics is not installed, restore it without replacing OpenCV:
pip install ultralytics --no-deps

# Download MediaPipe face landmarker model
curl -L -o face_landmarker_v2_with_blendshapes.task https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task
```

## Windows Camera Permissions

Settings > Privacy & Security > Camera:

1. "Camera access" master toggle — ON
2. "Let desktop apps access your camera" — ON
3. Find python3.XX.exe and enable it

## OpenCV GUI Troubleshooting

If startup fails at `cv2.namedWindow` with "The function is not implemented",
your environment is probably importing `opencv-python-headless`. Remove the
conflicting wheels and reinstall the GUI contrib build:

```bash
pip uninstall -y opencv-python-headless opencv-python
pip install --force-reinstall opencv-contrib-python
```

## Usage

```bash
python main.py           # auto-detect camera
python main.py 3         # use camera index 3
```

## Controls

- Dear PyGui controls: Detector, Mask Mode, Box Confidence, Blur, Dilation, Hold ms, Camera selector, Calibration mode, ArUco columns
- Buttons: Start/Stop Calibration, Capture ArUco, Clear Points, Reset, Quit
- Camera preview: click to place a camera calibration point; drag to draw an ArUco ROI; right-click to clear ROI
- Projector grid preview: click to place the matching projector point in manual calibration mode
- Keyboard: `c` toggles calibration, `s` captures ArUco, `x` clears current points, `r` resets, arrow keys nudge, `q` quits

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
- `detector_scrfd.py` — SCRFD face detection via InsightFace/ONNX Runtime
- `detector_manager.py` — detector routing and fallback chains
- `mask_generator.py` — generates white-on-black mask from detections
- `calibration.py` — camera-to-projector homography with persistence
- `camera.py` — camera discovery, permissions, hot-swap
- `output_spout.py` — Spout sender for Resolume
- `output_ndi.py` — NDI sender (optional)
- `config.py` — constants and runtime configuration
