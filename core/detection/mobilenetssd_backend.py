from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

from core.detection.backend_base import DetectorBackend
from core.detection.types import DetectionDict, LoadedModelInfo


class MobileNetSSDBackend(DetectorBackend):
    """
    Lightweight fallback detector using OpenCV DNN MobileNet SSD (VOC classes).
    This is intended as a safe default when no BYO ONNX YOLO model is installed.
    """

    backend_id = "mobilenetssd"

    def __init__(self) -> None:
        self._det = None

    def load(
        self,
        *,
        model_path: str,
        labels: Optional[Sequence[str]] = None,
        device: str = "cpu",
        input_size: int = 300,
    ) -> LoadedModelInfo:
        # model_path is ignored; ObjectDetector finds repo models/ files.
        from core.object_detector import ObjectDetector

        requested = str(device or "cpu").strip().lower()
        # ObjectDetector device resolver expects auto/cpu/gpu.
        if requested == "cuda":
            requested = "gpu"
        self._det = ObjectDetector(model_type="mobilenet", model_path=None, device=requested)
        resolved = "cpu"
        try:
            info = self._det.get_model_info() or {}
            resolved = "gpu" if str(info.get("device") or "cpu").strip().lower() == "gpu" else "cpu"
        except Exception:
            resolved = "cpu"
        return LoadedModelInfo(
            backend_id=self.backend_id,
            model_id="mobilenetssd",
            model_path="models/mobilenet_iter_73000.caffemodel",
            device=resolved,
            input_size=int(input_size or 300),
            class_names=self.get_class_names(),
        )

    def get_class_names(self) -> Optional[list[str]]:
        try:
            if self._det is not None and hasattr(self._det, "mobilenet_classes"):
                return [str(x) for x in (self._det.mobilenet_classes or [])]
        except Exception:
            return None
        return None

    def close(self) -> None:
        self._det = None

    def detect(
        self,
        frame_bgr: np.ndarray,
        *,
        min_confidence: float = 0.25,
        max_det: int = 100,
        allowed_classes: Optional[Sequence[str]] = None,
    ) -> list[DetectionDict]:
        if self._det is None:
            raise RuntimeError("MobileNetSSD backend not loaded")
        dets = self._det.detect(frame_bgr, conf_threshold=float(min_confidence))
        if not dets:
            return []
        allow = [str(x).strip().lower() for x in (allowed_classes or []) if str(x).strip()]
        if not allow:
            return dets[: int(max_det or 100)]
        out: list[DetectionDict] = []
        for d in dets:
            try:
                lab = str(d.get("class") or "").strip().lower()
                if lab and lab in allow:
                    out.append(d)
            except Exception:
                continue
        return out[: int(max_det or 100)]

