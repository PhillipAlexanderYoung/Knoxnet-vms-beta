from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence, Tuple

import numpy as np

from core.detection.backend_base import DetectorBackend
from core.detection.nms import nms_xyxy
from core.detection.types import DetectionDict, LoadedModelInfo

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class _LetterboxInfo:
    ratio: float
    pad_x: float
    pad_y: float
    in_size: int
    orig_w: int
    orig_h: int


def _letterbox(img_bgr: np.ndarray, new_size: int) -> Tuple[np.ndarray, _LetterboxInfo]:
    """
    Resize + pad to square (new_size x new_size) while preserving aspect ratio.
    """
    h, w = img_bgr.shape[:2]
    new_size = int(new_size)
    if new_size <= 0:
        new_size = 640

    r = min(new_size / max(1, w), new_size / max(1, h))
    new_w = int(round(w * r))
    new_h = int(round(h * r))

    # resize
    import cv2

    resized = cv2.resize(img_bgr, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    # pad to square
    pad_x = (new_size - new_w) / 2.0
    pad_y = (new_size - new_h) / 2.0
    left = int(np.floor(pad_x))
    right = int(np.ceil(pad_x))
    top = int(np.floor(pad_y))
    bottom = int(np.ceil(pad_y))

    out = cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))
    out = out[:new_size, :new_size]

    info = _LetterboxInfo(ratio=float(r), pad_x=float(left), pad_y=float(top), in_size=int(new_size), orig_w=int(w), orig_h=int(h))
    return out, info


def _xywh_to_xyxy(xywh: np.ndarray) -> np.ndarray:
    x = xywh[:, 0]
    y = xywh[:, 1]
    w = xywh[:, 2]
    h = xywh[:, 3]
    x1 = x - w / 2.0
    y1 = y - h / 2.0
    x2 = x + w / 2.0
    y2 = y + h / 2.0
    return np.stack([x1, y1, x2, y2], axis=1)


def _clip_xyxy(boxes: np.ndarray, w: float, h: float) -> np.ndarray:
    boxes[:, 0] = np.clip(boxes[:, 0], 0.0, w - 1.0)
    boxes[:, 1] = np.clip(boxes[:, 1], 0.0, h - 1.0)
    boxes[:, 2] = np.clip(boxes[:, 2], 0.0, w)
    boxes[:, 3] = np.clip(boxes[:, 3], 0.0, h)
    return boxes


def _hash_model(path: str) -> str:
    try:
        p = Path(path)
        st = p.stat()
        h = hashlib.sha256(f"{p.resolve()}|{st.st_size}|{int(st.st_mtime)}".encode("utf-8")).hexdigest()
        return h[:16]
    except Exception:
        return "unknown"


class OnnxRuntimeYoloBackend(DetectorBackend):
    backend_id = "onnxruntime"

    def __init__(self) -> None:
        self._session = None
        self._input_name: Optional[str] = None
        self._labels: Optional[list[str]] = None
        self._model_path: Optional[str] = None
        self._device: str = "cpu"
        self._input_size: int = 640
        self._model_id: str = ""

    def load(
        self,
        *,
        model_path: str,
        labels: Optional[Sequence[str]] = None,
        device: str = "cpu",
        input_size: int = 640,
    ) -> LoadedModelInfo:
        import onnxruntime as ort

        p = str(model_path)
        if not p:
            raise ValueError("Missing model_path")
        if not Path(p).exists():
            raise FileNotFoundError(f"ONNX model not found: {p}")

        self._labels = [str(x) for x in (labels or []) if str(x).strip()] or None
        self._model_path = p
        self._device = str(device or "cpu").lower().strip()
        self._input_size = int(input_size or 640)
        self._model_id = f"onnx:{Path(p).stem}:{_hash_model(p)}"

        providers = ["CPUExecutionProvider"]
        if self._device == "cuda":
            avail = [x for x in (ort.get_available_providers() or [])]
            if any("CUDAExecutionProvider" == x for x in avail):
                providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
            else:
                logger.info("onnxruntime CUDAExecutionProvider not available; falling back to CPU")
                self._device = "cpu"

        so = ort.SessionOptions()
        so.log_severity_level = 3
        self._session = ort.InferenceSession(p, sess_options=so, providers=providers)
        inp = self._session.get_inputs()[0]
        self._input_name = inp.name

        return LoadedModelInfo(
            backend_id=self.backend_id,
            model_id=self._model_id,
            model_path=p,
            device=self._device,
            input_size=self._input_size,
            class_names=self.get_class_names(),
        )

    def get_class_names(self) -> Optional[list[str]]:
        return list(self._labels) if self._labels else None

    def close(self) -> None:
        self._session = None
        self._input_name = None

    def _run(self, inp: np.ndarray) -> list[np.ndarray]:
        if self._session is None or not self._input_name:
            raise RuntimeError("ONNX backend not loaded")
        outs = self._session.run(None, {self._input_name: inp})
        return [np.asarray(x) for x in outs]

    def _decode_outputs(self, outputs: list[np.ndarray]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Return (boxes_xywh, scores, class_ids) all in model input pixel coordinates.

        Supports common Ultralytics exports:
          - (1, N, 4+nc)  (YOLOv8: no objectness)
          - (1, N, 5+nc)  (YOLOv5: objectness + class probs)
          - (1, 4+nc, N)  (sometimes transposed)
        """
        if not outputs:
            return np.zeros((0, 4), dtype=np.float32), np.zeros((0,), dtype=np.float32), np.zeros((0,), dtype=np.int32)

        y = outputs[0]
        y = np.asarray(y)

        # squeeze batch
        if y.ndim == 3 and int(y.shape[0]) == 1:
            y = y[0]

        # transpose if needed (C,N) -> (N,C)
        if y.ndim == 2 and y.shape[0] < y.shape[1] and y.shape[0] in (84, 85, 6, 7):
            # common: (84, 8400)
            y = y.T

        if y.ndim != 2 or y.shape[1] < 6:
            # Some models include NMS already and return (N,6) / (N,7)
            if y.ndim == 2 and y.shape[1] in (6, 7):
                pass
            else:
                raise RuntimeError(f"Unsupported ONNX output shape: {tuple(outputs[0].shape)}")

        n = int(y.shape[0])
        m = int(y.shape[1])
        if n <= 0:
            return np.zeros((0, 4), dtype=np.float32), np.zeros((0,), dtype=np.float32), np.zeros((0,), dtype=np.int32)

        # If the model already returns [x1,y1,x2,y2,score,class] (common NMS export)
        if m in (6, 7):
            boxes = y[:, 0:4].astype(np.float32, copy=False)
            scores = y[:, 4].astype(np.float32, copy=False)
            cls = y[:, 5].astype(np.int32, copy=False)
            # convert xyxy -> xywh for pipeline consistency
            xyxy = boxes
            xywh = np.stack(
                [
                    (xyxy[:, 0] + xyxy[:, 2]) / 2.0,
                    (xyxy[:, 1] + xyxy[:, 3]) / 2.0,
                    (xyxy[:, 2] - xyxy[:, 0]),
                    (xyxy[:, 3] - xyxy[:, 1]),
                ],
                axis=1,
            ).astype(np.float32, copy=False)
            return xywh, scores, cls

        nc_guess = m - 4
        # YOLOv5 style: 4 + obj + nc
        if (m >= 6) and (m - 5) >= 1 and (m - 5) <= 1000:
            # Heuristic: treat as YOLOv5 if column 4 looks like objectness in [0..1]
            obj = y[:, 4].astype(np.float32, copy=False)
            if float(np.nanmax(obj)) <= 1.0 + 1e-3:
                cls_probs = y[:, 5:].astype(np.float32, copy=False)
                cls_ids = np.argmax(cls_probs, axis=1).astype(np.int32)
                cls_scores = cls_probs[np.arange(n), cls_ids]
                scores = (obj * cls_scores).astype(np.float32)
                xywh = y[:, 0:4].astype(np.float32, copy=False)
                return xywh, scores, cls_ids

        # YOLOv8 style: 4 + nc (no obj)
        if nc_guess >= 1:
            cls_probs = y[:, 4:].astype(np.float32, copy=False)
            cls_ids = np.argmax(cls_probs, axis=1).astype(np.int32)
            scores = cls_probs[np.arange(n), cls_ids].astype(np.float32)
            xywh = y[:, 0:4].astype(np.float32, copy=False)
            return xywh, scores, cls_ids

        raise RuntimeError(f"Unable to decode ONNX output shape: {tuple(outputs[0].shape)}")

    def detect(
        self,
        frame_bgr: np.ndarray,
        *,
        min_confidence: float = 0.25,
        max_det: int = 100,
        allowed_classes: Optional[Sequence[str]] = None,
    ) -> list[DetectionDict]:
        if frame_bgr is None or not hasattr(frame_bgr, "shape"):
            return []
        if self._session is None:
            raise RuntimeError("ONNX detector not loaded")

        img, info = _letterbox(frame_bgr, int(self._input_size or 640))
        # BGR -> RGB, HWC -> CHW, normalize to 0..1
        rgb = img[:, :, ::-1].astype(np.float32) / 255.0
        chw = np.transpose(rgb, (2, 0, 1))
        inp = np.expand_dims(chw, axis=0).astype(np.float32, copy=False)

        outs = self._run(inp)
        xywh, scores, cls_ids = self._decode_outputs(outs)

        if xywh.size == 0:
            return []

        # confidence filter
        scores = scores.astype(np.float32, copy=False)
        keep = scores >= float(min_confidence)
        if not bool(np.any(keep)):
            return []

        xywh = xywh[keep]
        scores = scores[keep]
        cls_ids = cls_ids[keep]

        # optional class allowlist (by label)
        allowed = [str(x).strip().lower() for x in (allowed_classes or []) if str(x).strip()]
        if allowed and self._labels:
            # build label->id map
            lbl_to_id = {str(n).strip().lower(): i for i, n in enumerate(self._labels)}
            allowed_ids = set([int(lbl_to_id[a]) for a in allowed if a in lbl_to_id])
            if allowed_ids:
                mask = np.array([int(c) in allowed_ids for c in cls_ids], dtype=bool)
                if not bool(np.any(mask)):
                    return []
                xywh = xywh[mask]
                scores = scores[mask]
                cls_ids = cls_ids[mask]

        xyxy = _xywh_to_xyxy(xywh).astype(np.float32, copy=False)

        # NMS in model-input space
        keep_idx = nms_xyxy(xyxy, scores, iou_thresh=0.45, class_ids=cls_ids, class_aware=True, max_det=int(max_det or 100))
        if not keep_idx:
            return []

        xyxy = xyxy[keep_idx]
        scores = scores[keep_idx]
        cls_ids = cls_ids[keep_idx]

        # map back to original image coords
        # undo padding then divide by ratio
        xyxy[:, [0, 2]] -= float(info.pad_x)
        xyxy[:, [1, 3]] -= float(info.pad_y)
        xyxy /= float(info.ratio if info.ratio else 1.0)
        xyxy = _clip_xyxy(xyxy, float(info.orig_w), float(info.orig_h))

        dets: list[DetectionDict] = []
        for i in range(int(xyxy.shape[0])):
            x1, y1, x2, y2 = [float(v) for v in xyxy[i].tolist()]
            w = max(0.0, x2 - x1)
            h = max(0.0, y2 - y1)
            if w < 1.0 or h < 1.0:
                continue
            cid = int(cls_ids[i])
            label = str(cid)
            if self._labels and 0 <= cid < len(self._labels):
                label = str(self._labels[cid])
            dets.append(
                {
                    "bbox": {"x": float(x1), "y": float(y1), "w": float(w), "h": float(h)},
                    "class": label,
                    "confidence": float(scores[i]),
                    "class_id": cid,
                    "backend": self.backend_id,
                    "model": str(self._model_id or ""),
                }
            )

        dets.sort(key=lambda d: float(d.get("confidence", 0.0)), reverse=True)
        return dets[: int(max_det or 100)]

