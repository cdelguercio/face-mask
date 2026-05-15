"""SCRFD-based face detector.

Uses InsightFace's FaceAnalysis wrapper, whose detector is SCRFD. This module
is intentionally optional: if insightface/onnxruntime are not installed, the
detector disables itself and returns no boxes.
"""

import os
from typing import List

import numpy as np

from detector_types import BBoxResult

try:
    from insightface.app import FaceAnalysis
    import onnxruntime as ort

    SCRFD_AVAILABLE = True
except ImportError:
    FaceAnalysis = None
    ort = None
    SCRFD_AVAILABLE = False


_DLL_DIR_HANDLES = []


def _existing_bin_dir(root: str | None) -> str | None:
    if not root:
        return None
    path = os.path.join(root, "bin")
    return path if os.path.isdir(path) else None


def _candidate_cuda_bin_dirs() -> list[str]:
    candidates = [
        _existing_bin_dir(os.environ.get("CUDA_PATH")),
        _existing_bin_dir(os.environ.get("CUDA_HOME")),
    ]

    for key, value in os.environ.items():
        if key.startswith("CUDA_PATH_V"):
            candidates.append(_existing_bin_dir(value))

    roots = [
        os.path.join(
            os.environ.get("ProgramFiles", r"C:\Program Files"),
            "NVIDIA GPU Computing Toolkit",
            "CUDA",
        )
    ]
    for root in roots:
        if os.path.isdir(root):
            for version in os.listdir(root):
                candidates.append(os.path.join(root, version, "bin"))

    result = []
    for path in candidates:
        if path and os.path.isdir(path) and path not in result:
            result.append(path)
    return result


def _candidate_cudnn_bin_dirs() -> list[str]:
    candidates = [
        _existing_bin_dir(os.environ.get("CUDNN_PATH")),
        _existing_bin_dir(os.environ.get("CUDNN_HOME")),
        _existing_bin_dir(os.environ.get("CUDNN_ROOT")),
    ]

    for path in os.environ.get("PATH", "").split(os.pathsep):
        if path and os.path.isdir(path):
            candidates.append(path)

    nvidia_cudnn_root = os.path.join(
        os.environ.get("ProgramFiles", r"C:\Program Files"),
        "NVIDIA",
        "CUDNN",
    )
    cuda_version = None
    cuda_root = os.environ.get("CUDA_PATH") or os.environ.get("CUDA_HOME")
    if cuda_root:
        cuda_version = os.path.basename(cuda_root).lstrip("v")

    if os.path.isdir(nvidia_cudnn_root):
        for cudnn_version in os.listdir(nvidia_cudnn_root):
            cudnn_root = os.path.join(nvidia_cudnn_root, cudnn_version)
            if not os.path.isdir(cudnn_root):
                continue

            if cuda_version:
                candidates.append(os.path.join(cudnn_root, "bin", cuda_version, "x64"))

            bin_root = os.path.join(cudnn_root, "bin")
            if os.path.isdir(bin_root):
                for root, _, files in os.walk(bin_root):
                    if "cudnn64_9.dll" in files:
                        candidates.append(root)

    result = []
    for path in candidates:
        if path and os.path.isdir(path) and path not in result:
            result.append(path)
    return result


def _add_dll_dir(path: str):
    """Register a Windows DLL directory for this Python process."""
    if os.name != "nt" or not hasattr(os, "add_dll_directory"):
        return
    try:
        _DLL_DIR_HANDLES.append(os.add_dll_directory(path))
    except OSError:
        pass


def _find_dll(name: str, search_dirs: list[str]) -> str | None:
    for directory in search_dirs:
        candidate = os.path.join(directory, name)
        if os.path.exists(candidate):
            return candidate
    return None


def _configure_cuda_dll_search() -> tuple[bool, list[str]]:
    """Make CUDA/cuDNN DLLs visible before ONNX Runtime creates sessions."""
    cuda_dirs = _candidate_cuda_bin_dirs()
    cudnn_dirs = _candidate_cudnn_bin_dirs()
    for directory in cuda_dirs + cudnn_dirs:
        _add_dll_dir(directory)

    missing = []
    if not _find_dll("cublasLt64_12.dll", cuda_dirs + cudnn_dirs):
        missing.append("cublasLt64_12.dll")
    if not _find_dll("cudnn64_9.dll", cudnn_dirs + cuda_dirs):
        missing.append("cudnn64_9.dll")

    return not missing, missing


class ScrfdFaceDetector:
    """Face detector backed by InsightFace SCRFD."""

    def __init__(
        self,
        model_name: str = "buffalo_l",
        confidence: float = 0.7,
        det_size: tuple[int, int] = (640, 640),
        enabled: bool = True,
    ):
        self.enabled = enabled
        self.confidence = confidence
        self._app = None
        self.provider = "disabled"

        if not SCRFD_AVAILABLE:
            if enabled:
                print(
                    "[SCRFD] insightface is not installed. "
                    "SCRFD disabled. pip install insightface onnxruntime-gpu"
                )
            self.enabled = False
            return

        if not enabled:
            return

        try:
            available_providers = ort.get_available_providers()
            cuda_ready, missing_cuda_dlls = _configure_cuda_dll_search()
            use_cuda = (
                "CUDAExecutionProvider" in available_providers
                and cuda_ready
            )
            providers = ["CPUExecutionProvider"]
            ctx_id = -1
            if use_cuda:
                providers.insert(0, "CUDAExecutionProvider")
                ctx_id = 0
            elif "CUDAExecutionProvider" in available_providers:
                print(
                    "[SCRFD] CUDA provider is installed, but GPU runtime DLLs "
                    f"are missing: {', '.join(missing_cuda_dlls)}. "
                    "Install CUDA 12.x and cuDNN 9.x, then restart the terminal."
                )

            print(f"[SCRFD] Loading InsightFace model '{model_name}'...")
            self._app = FaceAnalysis(
                name=model_name,
                allowed_modules=["detection"],
                providers=providers,
            )
            self._app.prepare(ctx_id=ctx_id, det_size=det_size)
            self.provider = self._get_detection_provider()
            print(f"[SCRFD] Model ready at det_size={det_size}.")
            print(f"[SCRFD] Detection provider: {self.provider}")
        except Exception as e:
            print(f"[SCRFD] Failed to load model: {e}")
            self.enabled = False
            self.provider = "failed"

    def detect(self, frame_bgr: np.ndarray) -> List[BBoxResult]:
        """Run SCRFD face detection and return normalized bounding boxes."""
        if not self.enabled or self._app is None:
            return []

        h, w = frame_bgr.shape[:2]
        if h == 0 or w == 0:
            return []

        try:
            faces = self._app.get(frame_bgr)
        except Exception as e:
            print(f"[SCRFD] Detect error: {e}")
            return []

        results: List[BBoxResult] = []
        for face in faces:
            score = float(getattr(face, "det_score", 1.0))
            if score < self.confidence:
                continue
            x1, y1, x2, y2 = face.bbox.astype(float)
            x1 = min(max(x1, 0.0), float(w))
            y1 = min(max(y1, 0.0), float(h))
            x2 = min(max(x2, 0.0), float(w))
            y2 = min(max(y2, 0.0), float(h))
            bw = x2 - x1
            bh = y2 - y1
            if bw <= 0 or bh <= 0:
                continue
            results.append(BBoxResult(
                x=float(x1 / w),
                y=float(y1 / h),
                w=float(bw / w),
                h=float(bh / h),
                label="SCRFD",
            ))

        return results

    def _get_detection_provider(self) -> str:
        """Return the first ONNX provider used by the detection session."""
        try:
            model = self._app.models.get("detection")
            session = getattr(model, "session", None) or getattr(model, "sess", None)
            if session is None:
                return "unknown"
            providers = session.get_providers()
            return providers[0] if providers else "unknown"
        except Exception:
            return "unknown"
