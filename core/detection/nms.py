from __future__ import annotations

from typing import List, Optional

import numpy as np


def nms_xyxy(
    boxes_xyxy: np.ndarray,
    scores: np.ndarray,
    iou_thresh: float = 0.45,
    *,
    class_ids: Optional[np.ndarray] = None,
    class_aware: bool = True,
    max_det: int = 100,
) -> List[int]:
    """
    NMS for xyxy boxes.

    - boxes_xyxy: (N,4) float32
    - scores: (N,) float32
    - class_ids: (N,) int32 (optional)
    """
    if boxes_xyxy is None or scores is None:
        return []
    if int(boxes_xyxy.shape[0]) == 0:
        return []

    boxes = boxes_xyxy.astype(np.float32, copy=False)
    scores = scores.astype(np.float32, copy=False)
    if class_ids is not None:
        class_ids = class_ids.astype(np.int32, copy=False)

    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    areas = np.maximum(0.0, (x2 - x1)) * np.maximum(0.0, (y2 - y1))

    order = scores.argsort()[::-1]
    keep: List[int] = []
    while order.size > 0:
        i = int(order[0])
        keep.append(i)
        if len(keep) >= int(max_det):
            break
        if order.size == 1:
            break

        rest = order[1:]

        if class_aware and class_ids is not None:
            same = class_ids[rest] == class_ids[i]
            comp = rest[same]
            rest_keep = rest[~same]
        else:
            comp = rest
            rest_keep = np.empty((0,), dtype=rest.dtype)

        if comp.size == 0:
            order = rest_keep
            continue

        xx1 = np.maximum(x1[i], x1[comp])
        yy1 = np.maximum(y1[i], y1[comp])
        xx2 = np.minimum(x2[i], x2[comp])
        yy2 = np.minimum(y2[i], y2[comp])

        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        union = areas[i] + areas[comp] - inter
        # IMPORTANT: avoid RuntimeWarning from eager `inter / union` evaluation.
        iou = np.zeros_like(union, dtype=np.float32)
        np.divide(inter, union, out=iou, where=(union > 0))

        survivors = comp[iou <= float(iou_thresh)]
        order = np.concatenate([survivors, rest_keep])

    return keep

