from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import List

import numpy as np
from PIL import Image

try:
    import onnxruntime  # noqa: F401
    ONNX_AVAILABLE = True
except Exception:
    ONNX_AVAILABLE = False

try:
    from ultralytics import YOLO  # type: ignore
    ULTRALYTICS_AVAILABLE = True
except Exception:
    YOLO = None  # type: ignore
    ULTRALYTICS_AVAILABLE = False


@dataclass
class Detection:
    label: str
    confidence: float


class DetectionEngine:
    def __init__(self, model_path: str, device: str = "cpu") -> None:
        self.model_path = str(model_path)
        self.device = str(device or "cpu")
        self._backend = None

        # Preferred: ONNXRuntime
        if self.model_path.lower().endswith(".onnx"):
            if not ONNX_AVAILABLE:
                raise RuntimeError("onnxruntime is not installed; cannot run .onnx detector model.")
            from core.detection.registry import create_backend

            self._backend = create_backend("onnxruntime")
            self._backend.load(model_path=self.model_path, labels=None, device=self.device, input_size=640)
            return

        # Optional plugin: Ultralytics .pt
        if not ULTRALYTICS_AVAILABLE:
            raise RuntimeError(
                "Ultralytics plugin is not installed. "
                "Install requirements-ultralytics.txt to enable .pt models."
            )
        self.model = YOLO(self.model_path)
        self.model.to(self.device)

    def detect(self, image: Image.Image, confidence: float = 0.25) -> List[Detection]:
        array = np.array(image.convert("RGB"))
        # ONNXRuntime backend path
        if self._backend is not None:
            # backend expects BGR
            bgr = array[:, :, ::-1].copy()
            dets = self._backend.detect(bgr, min_confidence=float(confidence), max_det=50)
            out: List[Detection] = []
            for d in dets or []:
                out.append(Detection(label=str(d.get("class") or "object"), confidence=float(d.get("confidence", 0.0) or 0.0)))
            return _deduplicate(out)

        results = self.model.predict(array, conf=confidence, verbose=False)
        detections: List[Detection] = []
        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue
            names = result.names
            for cls_id, score in zip(boxes.cls.tolist(), boxes.conf.tolist()):
                detections.append(Detection(label=names[int(cls_id)], confidence=float(score)))
        return _deduplicate(detections)


def _deduplicate(detections: List[Detection], min_confidence: float = 0.2) -> List[Detection]:
    merged = {}
    for det in detections:
        if det.confidence < min_confidence:
            continue
        stored = merged.get(det.label)
        if not stored or det.confidence > stored.confidence:
            merged[det.label] = det
    return sorted(merged.values(), key=lambda d: d.confidence, reverse=True)


@lru_cache(maxsize=1)
def load_detector(model_path: str, device: str) -> DetectionEngine:
    return DetectionEngine(model_path=model_path, device=device)

