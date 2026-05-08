from __future__ import annotations

"""
YOLOX-style ByteTrack (vendored, minimal) for Desktop-local tracking.

Notes:
- Pure NumPy implementation (no Ultralytics internal imports).
- Uses a SORT-like Kalman filter and ByteTrack's two-stage association (high/low score).
- Operates on bboxes in (x, y, w, h) pixel coordinates (top-left origin).
"""

from dataclasses import dataclass
from enum import IntEnum
from typing import Dict, List, Optional, Tuple

import numpy as np

from .base import BaseTracker, Detection, Track, BBox


def _sanitize_tlwh(tlwh: Tuple[float, float, float, float]) -> Tuple[float, float, float, float]:
    x, y, w, h = tlwh
    if not np.isfinite(x):
        x = 0.0
    if not np.isfinite(y):
        y = 0.0
    if not np.isfinite(w) or w < 0.0:
        w = 0.0
    if not np.isfinite(h) or h < 0.0:
        h = 0.0
    return (float(x), float(y), float(w), float(h))


def tlwh_to_tlbr(tlwh: Tuple[float, float, float, float]) -> np.ndarray:
    x, y, w, h = _sanitize_tlwh(tlwh)
    return np.array([x, y, x + w, y + h], dtype=np.float32)


def iou_tlwh(a: Tuple[float, float, float, float], b: Tuple[float, float, float, float]) -> float:
    aa = tlwh_to_tlbr(a)
    bb = tlwh_to_tlbr(b)
    x1 = max(float(aa[0]), float(bb[0]))
    y1 = max(float(aa[1]), float(bb[1]))
    x2 = min(float(aa[2]), float(bb[2]))
    y2 = min(float(aa[3]), float(bb[3]))
    iw = max(0.0, x2 - x1)
    ih = max(0.0, y2 - y1)
    inter = iw * ih
    area_a = max(0.0, float(aa[2] - aa[0])) * max(0.0, float(aa[3] - aa[1]))
    area_b = max(0.0, float(bb[2] - bb[0])) * max(0.0, float(bb[3] - bb[1]))
    union = area_a + area_b - inter
    if union <= 1e-6:
        return 0.0
    return float(inter / union)


def _greedy_match_iou(
    tracks: List["STrack"],
    dets: List["STrack"],
    *,
    iou_thresh: float,
) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
    if not tracks or not dets:
        return [], list(range(len(tracks))), list(range(len(dets)))

    pairs: List[Tuple[float, int, int]] = []
    for ti, trk in enumerate(tracks):
        tb = trk.tlwh
        for di, det in enumerate(dets):
            iou = iou_tlwh(tb, det.tlwh)
            if iou >= iou_thresh:
                pairs.append((iou, ti, di))

    pairs.sort(key=lambda x: x[0], reverse=True)

    matched_t: set[int] = set()
    matched_d: set[int] = set()
    matches: List[Tuple[int, int]] = []
    for iou, ti, di in pairs:
        if ti in matched_t or di in matched_d:
            continue
        matched_t.add(ti)
        matched_d.add(di)
        matches.append((ti, di))

    u_trk = [i for i in range(len(tracks)) if i not in matched_t]
    u_det = [i for i in range(len(dets)) if i not in matched_d]
    return matches, u_trk, u_det


class TrackState(IntEnum):
    New = 0
    Tracked = 1
    Lost = 2
    Removed = 3


class KalmanFilter:
    """
    8D state: (cx, cy, a, h, vx, vy, va, vh)
    4D measurement: (cx, cy, a, h)
    """

    ndim, dt = 4, 1.0

    def __init__(self):
        self._motion_mat = np.eye(2 * self.ndim, dtype=np.float32)
        for i in range(self.ndim):
            self._motion_mat[i, self.ndim + i] = self.dt
        self._update_mat = np.eye(self.ndim, 2 * self.ndim, dtype=np.float32)

        # Tuned similar to SORT/ByteTrack defaults
        self._std_weight_position = 1.0 / 20
        self._std_weight_velocity = 1.0 / 160

    @staticmethod
    def tlwh_to_xyah(tlwh: Tuple[float, float, float, float]) -> np.ndarray:
        x, y, w, h = _sanitize_tlwh(tlwh)
        cx = x + w / 2.0
        cy = y + h / 2.0
        a = (w / h) if h > 1e-6 else 0.0
        return np.array([cx, cy, a, h], dtype=np.float32)

    @staticmethod
    def xyah_to_tlwh(xyah: np.ndarray) -> Tuple[float, float, float, float]:
        cx, cy, a, h = [float(v) for v in xyah[:4]]
        w = a * h
        x = cx - w / 2.0
        y = cy - h / 2.0
        return _sanitize_tlwh((x, y, w, h))

    def initiate(self, measurement: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        mean_pos = measurement
        mean_vel = np.zeros_like(mean_pos, dtype=np.float32)
        mean = np.r_[mean_pos, mean_vel].astype(np.float32)

        h = float(measurement[3])
        std = np.array(
            [
                2 * self._std_weight_position * h,
                2 * self._std_weight_position * h,
                1e-2,
                2 * self._std_weight_position * h,
                10 * self._std_weight_velocity * h,
                10 * self._std_weight_velocity * h,
                1e-5,
                10 * self._std_weight_velocity * h,
            ],
            dtype=np.float32,
        )
        covariance = np.diag(np.square(std)).astype(np.float32)
        return mean, covariance

    def predict(self, mean: np.ndarray, covariance: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        h = float(mean[3])
        std_pos = np.array(
            [
                self._std_weight_position * h,
                self._std_weight_position * h,
                1e-2,
                self._std_weight_position * h,
            ],
            dtype=np.float32,
        )
        std_vel = np.array(
            [
                self._std_weight_velocity * h,
                self._std_weight_velocity * h,
                1e-5,
                self._std_weight_velocity * h,
            ],
            dtype=np.float32,
        )
        motion_cov = np.diag(np.square(np.r_[std_pos, std_vel])).astype(np.float32)

        mean = (self._motion_mat @ mean).astype(np.float32)
        covariance = (self._motion_mat @ covariance @ self._motion_mat.T + motion_cov).astype(np.float32)
        return mean, covariance

    def update(self, mean: np.ndarray, covariance: np.ndarray, measurement: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        h = float(mean[3])
        std = np.array(
            [
                self._std_weight_position * h,
                self._std_weight_position * h,
                1e-1,
                self._std_weight_position * h,
            ],
            dtype=np.float32,
        )
        innovation_cov = np.diag(np.square(std)).astype(np.float32)

        projected_mean = (self._update_mat @ mean).astype(np.float32)
        projected_cov = (self._update_mat @ covariance @ self._update_mat.T + innovation_cov).astype(np.float32)

        # Kalman gain via Cholesky
        chol_factor = np.linalg.cholesky(projected_cov)
        kalman_gain = np.linalg.solve(chol_factor, (covariance @ self._update_mat.T).T).T
        kalman_gain = np.linalg.solve(chol_factor.T, kalman_gain.T).T

        innovation = (measurement - projected_mean).astype(np.float32)
        new_mean = (mean + kalman_gain @ innovation).astype(np.float32)
        new_cov = (covariance - kalman_gain @ projected_cov @ kalman_gain.T).astype(np.float32)
        return new_mean, new_cov


class STrack:
    def __init__(self, tlwh: Tuple[float, float, float, float], score: float, cls: str):
        self._tlwh = _sanitize_tlwh(tlwh)
        self.score = float(score)
        self.cls = str(cls or "object")

        self.track_id: int = 0
        self.state: TrackState = TrackState.New
        self.is_activated: bool = False

        self.mean: Optional[np.ndarray] = None
        self.covariance: Optional[np.ndarray] = None
        self.frame_id: int = 0
        self.start_frame: int = 0
        self.time_since_update: int = 0

        # Sticky label smoothing (simple score accumulation)
        self._label_scores: Dict[str, float] = {self.cls: max(0.0, float(score))}

    @property
    def tlwh(self) -> Tuple[float, float, float, float]:
        if self.mean is None:
            return self._tlwh
        xyah = self.mean[:4]
        return KalmanFilter.xyah_to_tlwh(xyah)

    def _decay_labels(self, decay: float = 0.92) -> None:
        for k in list(self._label_scores.keys()):
            self._label_scores[k] *= float(decay)
            if self._label_scores[k] < 1e-6:
                del self._label_scores[k]

    def _update_label(self, cls: str, score: float) -> None:
        self._decay_labels(0.92)
        key = str(cls or "object")
        self._label_scores[key] = self._label_scores.get(key, 0.0) + max(0.0, float(score))
        # set stable label
        if self._label_scores:
            self.cls = max(self._label_scores.items(), key=lambda kv: kv[1])[0]

    def activate(self, kf: KalmanFilter, frame_id: int, track_id: int) -> None:
        self.track_id = int(track_id)
        self.mean, self.covariance = kf.initiate(kf.tlwh_to_xyah(self._tlwh))
        self.frame_id = int(frame_id)
        self.start_frame = int(frame_id)
        self.state = TrackState.Tracked
        self.is_activated = True
        self.time_since_update = 0

    def re_activate(self, new_track: "STrack", kf: KalmanFilter, frame_id: int, new_id: bool = False) -> None:
        if self.mean is None or self.covariance is None:
            self.mean, self.covariance = kf.initiate(kf.tlwh_to_xyah(new_track._tlwh))
        else:
            self.mean, self.covariance = kf.update(self.mean, self.covariance, kf.tlwh_to_xyah(new_track._tlwh))
        self.frame_id = int(frame_id)
        self.state = TrackState.Tracked
        self.is_activated = True
        self.time_since_update = 0
        self.score = float(new_track.score)
        self._update_label(new_track.cls, new_track.score)
        if new_id:
            self.track_id = int(self.track_id)

    def update(self, new_track: "STrack", kf: KalmanFilter, frame_id: int) -> None:
        if self.mean is None or self.covariance is None:
            self.mean, self.covariance = kf.initiate(kf.tlwh_to_xyah(new_track._tlwh))
        else:
            self.mean, self.covariance = kf.update(self.mean, self.covariance, kf.tlwh_to_xyah(new_track._tlwh))
        self.frame_id = int(frame_id)
        self.state = TrackState.Tracked
        self.is_activated = True
        self.time_since_update = 0
        self.score = float(new_track.score)
        self._update_label(new_track.cls, new_track.score)

    def predict(self, kf: KalmanFilter) -> None:
        if self.mean is None or self.covariance is None:
            return
        self.mean, self.covariance = kf.predict(self.mean, self.covariance)
        self.time_since_update += 1

    def mark_lost(self) -> None:
        self.state = TrackState.Lost

    def mark_removed(self) -> None:
        self.state = TrackState.Removed

    def to_track(self) -> Track:
        x, y, w, h = self.tlwh
        predicted = bool(self.time_since_update > 0)
        lost_frames = int(self.time_since_update)
        age = int(max(0, self.frame_id - self.start_frame))
        # label confidence is normalized relative to current label scores sum
        conf = float(self.score)
        try:
            total = float(sum(self._label_scores.values())) or 1.0
            conf = float(max(0.0, min(1.0, self._label_scores.get(self.cls, 0.0) / total)))
        except Exception:
            pass
        return Track(
            track_id=int(self.track_id),
            bbox=BBox(x=float(x), y=float(y), w=float(w), h=float(h)),
            cls=str(self.cls or "object"),
            confidence=float(conf),
            age=age,
            lost_frames=lost_frames,
            predicted=predicted,
        )


@dataclass(frozen=True)
class ByteTrackConfig:
    track_thresh: float = 0.35
    low_thresh: float = 0.10
    match_thresh: float = 0.30
    track_buffer: int = 30
    min_box_area: float = 10.0


class _ByteTrackerCore:
    def __init__(self, cfg: ByteTrackConfig):
        self.cfg = cfg
        self.kf = KalmanFilter()
        self.frame_id = 0
        self._next_id = 1

        self.tracked: List[STrack] = []
        self.lost: List[STrack] = []
        self.removed: List[STrack] = []

    def reset(self) -> None:
        self.frame_id = 0
        self._next_id = 1
        self.tracked.clear()
        self.lost.clear()
        self.removed.clear()

    def _new_id(self) -> int:
        tid = self._next_id
        self._next_id += 1
        return tid

    def update(self, dets: List[STrack]) -> List[STrack]:
        self.frame_id += 1
        fid = self.frame_id

        # Split detections into high and low score pools
        high: List[STrack] = []
        low: List[STrack] = []
        for d in dets or []:
            x, y, w, h = d.tlwh
            if (w * h) < float(self.cfg.min_box_area):
                continue
            if d.score >= float(self.cfg.track_thresh):
                high.append(d)
            elif d.score >= float(self.cfg.low_thresh):
                low.append(d)

        # Predict existing tracks (tracked + lost)
        strack_pool: List[STrack] = []
        for t in self.tracked:
            t.predict(self.kf)
            strack_pool.append(t)
        for t in self.lost:
            t.predict(self.kf)
            strack_pool.append(t)

        # First association: high score dets with tracked+lost
        matches, u_trk, u_det = _greedy_match_iou(strack_pool, high, iou_thresh=float(self.cfg.match_thresh))

        activated: List[STrack] = []
        refind: List[STrack] = []
        lost: List[STrack] = []

        for ti, di in matches:
            trk = strack_pool[ti]
            det = high[di]
            if trk.state == TrackState.Tracked:
                trk.update(det, self.kf, fid)
                activated.append(trk)
            else:
                trk.re_activate(det, self.kf, fid, new_id=False)
                refind.append(trk)

        # Unmatched tracks become candidates for second stage or lost
        r_trk = [strack_pool[i] for i in u_trk]
        # Second association: remaining tracks with low score dets
        matches2, u_trk2, u_det2 = _greedy_match_iou(r_trk, low, iou_thresh=float(self.cfg.match_thresh))
        for ti, di in matches2:
            trk = r_trk[ti]
            det = low[di]
            if trk.state == TrackState.Tracked:
                trk.update(det, self.kf, fid)
                activated.append(trk)
            else:
                trk.re_activate(det, self.kf, fid, new_id=False)
                refind.append(trk)

        # Tracks still unmatched -> lost
        for i in u_trk2:
            trk = r_trk[i]
            if trk.state != TrackState.Removed:
                trk.mark_lost()
                lost.append(trk)

        # New tracks from unmatched high detections
        for i in u_det:
            det = high[i]
            det.activate(self.kf, fid, self._new_id())
            activated.append(det)

        # Manage state lists
        # 1) Update tracked list
        new_tracked: List[STrack] = []
        for t in self.tracked:
            if t.state == TrackState.Tracked:
                new_tracked.append(t)
        new_tracked.extend(activated)
        new_tracked.extend(refind)

        # 2) Update lost list (dedupe by id)
        new_lost: List[STrack] = []
        existing_lost = {t.track_id: t for t in self.lost if t.state == TrackState.Lost}
        for t in lost:
            existing_lost[t.track_id] = t
        for tid, t in existing_lost.items():
            if (fid - t.frame_id) > int(self.cfg.track_buffer):
                t.mark_removed()
                self.removed.append(t)
            else:
                new_lost.append(t)

        # Remove any lost tracks that were refound/activated
        active_ids = {t.track_id for t in new_tracked}
        new_lost = [t for t in new_lost if t.track_id not in active_ids]

        self.tracked = _dedupe_stracks(new_tracked)
        self.lost = _dedupe_stracks(new_lost)

        # Return active tracks only
        return [t for t in self.tracked if t.is_activated and t.state == TrackState.Tracked]


def _dedupe_stracks(stracks: List[STrack]) -> List[STrack]:
    seen: Dict[int, STrack] = {}
    for t in stracks or []:
        if not t.track_id:
            continue
        seen[int(t.track_id)] = t
    return list(seen.values())


class ByteTrackObjectTracker(BaseTracker):
    def __init__(self, cfg: Optional[ByteTrackConfig] = None):
        self.cfg = cfg or ByteTrackConfig()
        self._core = _ByteTrackerCore(self.cfg)

    def reset(self) -> None:
        self._core.reset()

    def update(self, detections: List[Detection], frame_ts: Optional[float] = None) -> List[Track]:
        det_tracks: List[STrack] = []
        for d in detections or []:
            tlwh = d.bbox.as_xywh()
            det_tracks.append(STrack(tlwh=tlwh, score=float(d.confidence), cls=str(d.cls)))

        active = self._core.update(det_tracks)
        return [t.to_track() for t in active]


