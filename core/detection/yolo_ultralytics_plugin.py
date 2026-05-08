from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

from core.detection.backend_base import DetectorBackend
from core.detection.types import DetectionDict, LoadedModelInfo

logger = logging.getLogger(__name__)


class UltralyticsYoloBackend(DetectorBackend):
    """
    Optional plugin backend.

    Policy:
    - Never required for the app to run.
    - Import is lazy and guarded.
    - Not installed/bundled by default.
    """

    backend_id = "ultralytics"

    def __init__(self) -> None:
        self._model = None
        self._labels: Optional[list[str]] = None
        self._model_path: Optional[str] = None
        self._device: str = "cpu"
        self._input_size: int = 640

    def load(
        self,
        *,
        model_path: str,
        labels: Optional[Sequence[str]] = None,
        device: str = "cpu",
        input_size: int = 640,
    ) -> LoadedModelInfo:
        try:
            from ultralytics import YOLO  # type: ignore
        except Exception as e:
            raise RuntimeError(
                "Ultralytics plugin is not installed. "
                "Install the optional plugin to use .pt models. "
                f"Details: {e}"
            )

        p = str(model_path)
        if not p:
            raise ValueError("Missing model_path")
        if not Path(p).exists():
            raise FileNotFoundError(f"Model not found: {p}")

        self._model = YOLO(p)
        self._model_path = p
        self._device = str(device or "cpu").lower().strip()
        self._input_size = int(input_size or 640)
        self._labels = [str(x) for x in (labels or []) if str(x).strip()] or None

        # best-effort labels from model.names
        try:
            names = getattr(self._model, "names", None) or {}
            if isinstance(names, dict) and not self._labels:
                out = []
                for k in sorted(names.keys()):
                    try:
                        out.append(str(names[k]))
                    except Exception:
                        continue
                self._labels = [x for x in out if x and x.strip()] or None
        except Exception:
            pass

        return LoadedModelInfo(
            backend_id=self.backend_id,
            model_id=f"ultralytics:{Path(p).stem}",
            model_path=p,
            device=self._device,
            input_size=self._input_size,
            class_names=self.get_class_names(),
        )

    def get_class_names(self) -> Optional[list[str]]:
        return list(self._labels) if self._labels else None

    def close(self) -> None:
        self._model = None

    def detect(
        self,
        frame_bgr: np.ndarray,
        *,
        min_confidence: float = 0.25,
        max_det: int = 100,
        allowed_classes: Optional[Sequence[str]] = None,
    ) -> list[DetectionDict]:
        if self._model is None:
            raise RuntimeError("Ultralytics backend not loaded")

        # Ultralytics expects numpy array; device can be "cpu" or "cuda"
        results = self._model.predict(
            frame_bgr,
            imgsz=int(self._input_size or 640),
            conf=float(min_confidence),
            device=str(self._device or "cpu"),
            max_det=int(max_det or 100),
            verbose=False,
        )

        dets: list[DetectionDict] = []
        r0 = results[0] if isinstance(results, (list, tuple)) and results else results
        boxes = getattr(r0, "boxes", None)
        if boxes is None:
            return []

        xyxy = getattr(boxes, "xyxy", None)
        confs = getattr(boxes, "conf", None)
        clss = getattr(boxes, "cls", None)
        if xyxy is None or confs is None or clss is None:
            return []

        xyxy_np = xyxy.detach().cpu().numpy()  # type: ignore[union-attr]
        conf_np = confs.detach().cpu().numpy()  # type: ignore[union-attr]
        cls_np = clss.detach().cpu().numpy()  # type: ignore[union-attr]

        labels = self._labels
        allow = [str(x).strip().lower() for x in (allowed_classes or []) if str(x).strip()]

        for i in range(int(xyxy_np.shape[0])):
            x1, y1, x2, y2 = [float(v) for v in xyxy_np[i].tolist()]
            w = max(0.0, x2 - x1)
            h = max(0.0, y2 - y1)
            if w < 1.0 or h < 1.0:
                continue
            cid = int(cls_np[i])
            label = str(cid)
            if labels and 0 <= cid < len(labels):
                label = str(labels[cid])
            if allow and str(label).strip().lower() not in allow:
                continue
            dets.append(
                {
                    "bbox": {"x": float(x1), "y": float(y1), "w": float(w), "h": float(h)},
                    "class": str(label),
                    "confidence": float(conf_np[i]),
                    "class_id": int(cid),
                    "backend": self.backend_id,
                    "model": str(self._model_path or ""),
                }
            )
        dets.sort(key=lambda d: float(d.get("confidence", 0.0)), reverse=True)
        return dets[: int(max_det or 100)]

