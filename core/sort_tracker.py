"""Lightweight SORT tracker with EMA smoothing for stable IDs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict, Any, Tuple, Optional

import math

# Numerical guard to avoid divide-by-zero situations
_EPS = 1e-6


def _sanitize_bbox(bbox: Tuple[float, float, float, float]) -> Tuple[float, float, float, float]:
    """Ensure bbox components are finite and non-negative."""
    x, y, w, h = bbox
    if not math.isfinite(x):
        x = 0.0
    if not math.isfinite(y):
        y = 0.0
    if not math.isfinite(w) or w < 0.0:
        w = 0.0
    if not math.isfinite(h) or h < 0.0:
        h = 0.0
    return (float(x), float(y), float(w), float(h))


def iou_xywh(box_a: Tuple[float, float, float, float],
             box_b: Tuple[float, float, float, float]) -> float:
    """Compute IoU for boxes in (x, y, w, h) format (top-left origin)."""
    ax, ay, aw, ah = _sanitize_bbox(box_a)
    bx, by, bw, bh = _sanitize_bbox(box_b)

    if aw <= 0.0 or ah <= 0.0 or bw <= 0.0 or bh <= 0.0:
        return 0.0

    ax2 = ax + aw
    ay2 = ay + ah
    bx2 = bx + bw
    by2 = by + bh

    inter_x1 = max(ax, bx)
    inter_y1 = max(ay, by)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter = inter_w * inter_h

    area_a = aw * ah
    area_b = bw * bh
    union = area_a + area_b - inter
    if union <= 0.0:
        return 0.0
    return float(inter / union)


@dataclass
class _StickyLabelState:
    stable_label: str
    label_scores: Dict[str, float]
    label_confidence: float


class _Track:
    """Internal track that maintains an exponentially-smoothed bbox state."""

    def __init__(self,
                 tid: int,
                 bbox: Tuple[float, float, float, float],
                 cls: str,
                 conf: float,
                 confirmation_hits: int):
        self.id = int(tid)
        self.age = 0
        self.time_since_update = 0
        self.hits = 1
        self.hit_streak = 1
        self.confirmed = confirmation_hits <= 1
        self._confirmation_hits = max(1, int(confirmation_hits))

        self.class_name = str(cls)
        self.confidence = float(conf)

        # Sticky label smoothing
        normalized = self._normalize_label(cls)
        self._label_state = _StickyLabelState(
            stable_label=normalized,
            label_scores={normalized: float(conf)},
            label_confidence=float(conf)
        )

        self.bbox = _sanitize_bbox(bbox)
        self._smoothed_bbox: Tuple[float, float, float, float] = self.bbox
        self._ema_alpha_det = 0.45
        self._ema_alpha_pred = 0.25

    def _smooth_bbox(
        self,
        bbox: Tuple[float, float, float, float],
        alpha: float,
        clamp: bool = True,
    ) -> Tuple[float, float, float, float]:
        alpha = max(0.0, min(1.0, float(alpha)))
        target = _sanitize_bbox(bbox)
        if self._smoothed_bbox is None:
            self._smoothed_bbox = target
            return target

        old_x, old_y, old_w, old_h = self._smoothed_bbox
        new_x, new_y, new_w, new_h = target

        limit_xy = max(6.0, 0.35 * max(old_w, old_h, new_w, new_h))
        limit_wh = max(4.0, 0.3 * max(old_w, old_h, new_w, new_h))

        def _lerp(old: float, new: float, limit: float) -> float:
            blended = old + alpha * (new - old)
            delta = blended - old
            delta = max(-limit, min(limit, delta))
            blended = old + delta
            if clamp:
                lo = min(old, new)
                hi = max(old, new)
                blended = min(max(blended, lo), hi)
            return blended

        blended = (
            _lerp(old_x, new_x, limit_xy),
            _lerp(old_y, new_y, limit_xy),
            _lerp(old_w, new_w, limit_wh),
            _lerp(old_h, new_h, limit_wh),
        )

        self._smoothed_bbox = _sanitize_bbox(blended)
        return self._smoothed_bbox

    @staticmethod
    def _normalize_label(label: Optional[str]) -> str:
        name = (label or "object").strip().lower()
        if name in ("car", "truck", "bus", "van", "automobile", "pickup"):
            return "car"
        if name in ("person", "human"):
            return "person"
        return name or "object"

    def _decay_scores(self, decay: float) -> None:
        scores = self._label_state.label_scores
        for key in list(scores.keys()):
            scores[key] *= float(decay)
            if scores[key] < 1e-6:
                del scores[key]

    def _update_label(self, label: Optional[str], confidence: Optional[float]) -> None:
        observed_label = self._normalize_label(label)
        observed_conf = float(confidence or 0.0)

        self._decay_scores(decay=0.92)
        scores = self._label_state.label_scores
        scores[observed_label] = scores.get(observed_label, 0.0) + max(0.0, observed_conf)

        if scores:
            sorted_items = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
            top_label, top_score = sorted_items[0]
            prev_score = scores.get(self._label_state.stable_label, 0.0)
            margin_ratio = 1.4
            margin_delta = 0.05
            if (
                top_label != self._label_state.stable_label
                and (top_score >= prev_score * margin_ratio or (top_score - prev_score) >= margin_delta)
            ):
                self._label_state.stable_label = top_label
            total = float(sum(scores.values())) or 1.0
            self._label_state.label_confidence = float(max(0.0, min(1.0, top_score / total)))

    def predict(self) -> Tuple[float, float, float, float]:
        """Advance the state prediction one step ahead."""
        self.age += 1
        self.time_since_update += 1
        self._decay_scores(decay=0.995)

        return self._smoothed_bbox

    def update(self,
               bbox: Tuple[float, float, float, float],
               cls: Optional[str],
               conf: Optional[float]) -> None:
        """Correct the state with an observed detection."""
        measurement_bbox = _sanitize_bbox(bbox)
        self.bbox = measurement_bbox
        self._smooth_bbox(measurement_bbox, self._ema_alpha_det, clamp=True)
        self.time_since_update = 0
        self.hits += 1
        self.hit_streak += 1

        if cls is not None:
            self.class_name = str(cls)
        if conf is not None:
            self.confidence = float(conf)

        self._update_label(cls, conf)
        if self.hit_streak >= self._confirmation_hits:
            self.confirmed = True

    def mark_missed(self) -> None:
        self.hit_streak = 0
        # Gently follow the prediction when we have no measurement
        self._smooth_bbox(self._smoothed_bbox, self._ema_alpha_pred, clamp=False)

    @property
    def stable_label(self) -> str:
        return self._label_state.stable_label

    @property
    def label_confidence(self) -> float:
        return float(self._label_state.label_confidence)

    def to_dict(self) -> Dict[str, Any]:
        box_source = self._smoothed_bbox or self.bbox
        x, y, w, h = box_source
        return {
            'id': self.id,
            'bbox': {'x': float(x), 'y': float(y), 'w': float(w), 'h': float(h)},
            'class': self.stable_label,
            'confidence': float(self.label_confidence),
            'age': int(self.age),
            'hits': int(self.hits),
            'lost_frames': int(self.time_since_update),
            'predicted': bool(self.time_since_update > 0),
        }


class SortTracker:
    """SORT-inspired tracker with IoU association and EMA smoothing."""

    def __init__(self, max_age: int = 15, min_hits: int = 2, iou_threshold: float = 0.3):
        self.max_age = int(max_age)
        self.min_hits = max(1, int(min_hits))
        self.iou_threshold = float(iou_threshold)
        self._next_id = 1
        self._tracks: Dict[int, _Track] = {}

    def reset(self) -> None:
        self._tracks.clear()
        self._next_id = 1

    def update(self, detections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Update tracker with current detections.

        Args:
            detections: list of detection dicts with bbox{x,y,w,h}, class, confidence.

        Returns:
            List of confirmed tracks with persistent IDs.
        """
        detections = detections or []

        # 1) Predict track positions for the new frame
        predicted: Dict[int, Tuple[float, float, float, float]] = {}
        for tid, track in list(self._tracks.items()):
            predicted[tid] = track.predict()

        # 2) Prepare detection data
        det_boxes: List[Tuple[float, float, float, float]] = []
        det_meta: List[Tuple[str, float]] = []
        for det in detections:
            box_dict = det.get('bbox') or {}
            box = _sanitize_bbox((
                float(box_dict.get('x', 0.0)),
                float(box_dict.get('y', 0.0)),
                float(box_dict.get('w', 0.0)),
                float(box_dict.get('h', 0.0)),
            ))
            det_boxes.append(box)
            det_meta.append((str(det.get('class', 'object')), float(det.get('confidence', 0.0))))

        unmatched_trks = set(self._tracks.keys())
        unmatched_dets = set(range(len(det_boxes)))

        # 3) Build IoU matrix between predicted tracks and detections
        iou_matrix: Dict[Tuple[int, int], float] = {}
        for tid, pred_box in predicted.items():
            for didx, det_box in enumerate(det_boxes):
                iou = iou_xywh(pred_box, det_box)
                if iou >= self.iou_threshold:
                    iou_matrix[(tid, didx)] = iou

        # 4) Greedy matching of detections to tracks
        pairs: List[Tuple[int, int]] = []
        while iou_matrix:
            (best_tid, best_didx), best_iou = max(iou_matrix.items(), key=lambda kv: kv[1])
            if best_tid not in unmatched_trks or best_didx not in unmatched_dets:
                del iou_matrix[(best_tid, best_didx)]
                continue
            pairs.append((best_tid, best_didx))
            unmatched_trks.discard(best_tid)
            unmatched_dets.discard(best_didx)
            for key in list(iou_matrix.keys()):
                if key[0] == best_tid or key[1] == best_didx:
                    del iou_matrix[key]

        # 5) Update matched tracks
        for tid, didx in pairs:
            det_box = det_boxes[didx]
            cls, conf = det_meta[didx]
            self._tracks[tid].update(det_box, cls, conf)

        # Tracks without matches get marked as missed
        for tid in unmatched_trks:
            self._tracks[tid].mark_missed()

        # 6) Create new tracks for unmatched detections (avoid tiny degenerate boxes)
        for didx in sorted(unmatched_dets):
            det_box = det_boxes[didx]
            if det_box[2] <= 1.0 or det_box[3] <= 1.0:
                continue
            cls, conf = det_meta[didx]
            tid = self._next_id
            self._next_id += 1
            self._tracks[tid] = _Track(tid, det_box, cls, conf, self.min_hits)

        # 7) Drop stale tracks
        for tid in list(self._tracks.keys()):
            track = self._tracks[tid]
            if track.time_since_update > self.max_age:
                del self._tracks[tid]

        # 8) Export confirmed tracks (includes short-term predictions for continuity)
        output: List[Dict[str, Any]] = []
        for track in self._tracks.values():
            if track.confirmed:
                output.append(track.to_dict())
            elif track.hit_streak >= self.min_hits and track.time_since_update == 0:
                # Promote newly stable track immediately
                track.confirmed = True
                output.append(track.to_dict())

        output.sort(key=lambda t: t['bbox']['w'] * t['bbox']['h'], reverse=True)
        return output

