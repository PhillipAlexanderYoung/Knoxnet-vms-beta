from __future__ import annotations

from typing import Optional, Sequence

from core.detection.backend_base import DetectorBackend
from core.detection.types import BackendInfo
from core.utils.detector_device import resolve_detector_device


def get_backend_info(backend_id: str) -> BackendInfo:
    backend_id = str(backend_id or "").strip().lower()
    if backend_id in ("auto", "default"):
        return BackendInfo(
            id="auto",
            display_name="Auto (ONNXRuntime YOLO → MobileNetSSD)",
            available=True,
        )
    if backend_id in ("mobilenet", "mobilenetssd", "opencv-mobilenetssd"):
        return BackendInfo(
            id="mobilenetssd",
            display_name="MobileNetSSD (fallback)",
            available=True,
        )
    if backend_id in ("onnx", "onnxruntime", "onnxruntime-yolo", "onnx_yolo"):
        try:
            import onnxruntime  # noqa: F401

            return BackendInfo(
                id="onnxruntime",
                display_name="ONNXRuntime (recommended)",
                available=True,
            )
        except Exception as e:
            return BackendInfo(
                id="onnxruntime",
                display_name="ONNXRuntime (recommended)",
                available=False,
                detail=str(e),
            )
    if backend_id in ("ultralytics", "yolo-ultralytics", "plugin-ultralytics"):
        try:
            import ultralytics  # noqa: F401

            return BackendInfo(
                id="ultralytics",
                display_name="Ultralytics plugin (optional)",
                available=True,
            )
        except Exception as e:
            return BackendInfo(
                id="ultralytics",
                display_name="Ultralytics plugin (optional)",
                available=False,
                detail=str(e),
            )

    return BackendInfo(id=backend_id or "onnxruntime", display_name=str(backend_id), available=False, detail="Unknown backend")


def create_backend(backend_id: str) -> DetectorBackend:
    backend_id = str(backend_id or "onnxruntime").strip().lower()
    if backend_id in ("mobilenet", "mobilenetssd", "opencv-mobilenetssd"):
        from core.detection.mobilenetssd_backend import MobileNetSSDBackend

        return MobileNetSSDBackend()
    if backend_id in ("auto", "default"):
        # created as onnxruntime first; caller can fallback if load fails
        from core.detection.yolo_onnxruntime import OnnxRuntimeYoloBackend

        return OnnxRuntimeYoloBackend()
    if backend_id in ("onnx", "onnxruntime", "onnxruntime-yolo", "onnx_yolo"):
        from core.detection.yolo_onnxruntime import OnnxRuntimeYoloBackend

        return OnnxRuntimeYoloBackend()
    if backend_id in ("ultralytics", "yolo-ultralytics", "plugin-ultralytics"):
        from core.detection.yolo_ultralytics_plugin import UltralyticsYoloBackend

        return UltralyticsYoloBackend()

    # default
    from core.detection.yolo_onnxruntime import OnnxRuntimeYoloBackend

    return OnnxRuntimeYoloBackend()


def resolve_device_for_backend(backend_id: str, requested: str) -> str:
    """
    Normalize device preference across backends.
    - returns "cpu" or "cuda" (best effort)
    """
    backend_id = str(backend_id or "onnxruntime").lower().strip()
    want = str(requested or "auto").lower().strip()
    if want in ("cpu", "cuda"):
        return want
    if want == "gpu":
        # Backend-specific canonical GPU token:
        # - ONNXRuntime/Ultralytics: "cuda"
        # - OpenCV/MobileNetSSD: "gpu"
        if backend_id in ("mobilenetssd", "mobilenet", "opencv-mobilenetssd"):
            return "gpu"
        return "cuda"
    if backend_id in ("auto", "default"):
        backend_id = "onnxruntime"

    # auto:
    if backend_id in ("onnxruntime", "onnx", "onnxruntime-yolo", "onnx_yolo"):
        try:
            import onnxruntime as ort

            prov = [p.lower() for p in (ort.get_available_providers() or [])]
            if any("cuda" in p for p in prov):
                return "cuda"
        except Exception:
            pass
        return "cpu"

    if backend_id in ("mobilenetssd", "mobilenet", "opencv-mobilenetssd"):
        cfg = resolve_detector_device("auto")
        # OpenCV DNN MobileNet can only run accelerated when OpenCV CUDA is available.
        if cfg.kind == "gpu" and bool(cfg.capabilities.cv_cuda):
            return "gpu"
        return "cpu"

    # ultralytics plugin: use torch if available, else cpu
    cfg = resolve_detector_device("auto")
    if cfg.kind == "gpu" and bool(cfg.capabilities.torch_cuda):
        return "cuda"
    return "cpu"


def normalize_allowed_classes(allowed: Optional[Sequence[str]]) -> Optional[list[str]]:
    if not allowed:
        return None
    out: list[str] = []
    for a in allowed:
        if a is None:
            continue
        s = str(a).strip()
        if not s:
            continue
        out.append(s)
    return out or None

