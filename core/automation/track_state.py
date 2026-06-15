"""
Track-centric scene state for Event Rules (Stage 1).

Maintains per-(camera_id, tracker_namespace, track_id) state and emits
edge-triggered semantic events from backend detection tracks.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Deque, Dict, List, Optional, Sequence, Set, Tuple

from .conditions import point_in_polygon, estimate_dominant_color_from_bgr

BACKEND_SORT_NAMESPACE = "backend_sort"
MOTION_BOX_NAMESPACE = "motion_box"

HYSTERESIS_FRAMES = 3
CENTROID_HISTORY_LEN = 16
LOST_GRACE_FRAMES = 2
REACQUIRE_IOU_THRESHOLD = 0.35
REACQUIRE_WINDOW_SEC = 2.0
DEFAULT_DWELL_SEC = 1.0
TAG_NEAR_RADIUS = 0.05
COLOR_EMA_ALPHA = 0.35


def _as_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return default


def _utc_iso(ts: Optional[float] = None) -> str:
    t = float(ts if ts is not None else datetime.now(tz=timezone.utc).timestamp())
    return datetime.fromtimestamp(t, tz=timezone.utc).isoformat()


def _bbox_iou(a: Dict[str, float], b: Dict[str, float]) -> float:
    ax, ay, aw, ah = _as_float(a.get("x")), _as_float(a.get("y")), _as_float(a.get("w")), _as_float(a.get("h"))
    bx, by, bw, bh = _as_float(b.get("x")), _as_float(b.get("y")), _as_float(b.get("w")), _as_float(b.get("h"))
    if aw <= 0 or ah <= 0 or bw <= 0 or bh <= 0:
        return 0.0
    x1 = max(ax, bx)
    y1 = max(ay, by)
    x2 = min(ax + aw, bx + bw)
    y2 = min(ay + ah, by + bh)
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    union = aw * ah + bw * bh - inter
    return float(inter / union) if union > 0 else 0.0


def normalize_shapes(shapes: Optional[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    """Normalize DB/desktop shape records to a common schema."""
    src = shapes if isinstance(shapes, dict) else {}
    zones_out: List[Dict[str, Any]] = []
    lines_out: List[Dict[str, Any]] = []
    for z in src.get("zones", []) or []:
        if not isinstance(z, dict):
            continue
        pts = z.get("points") or z.get("pts") or z.get("coordinates") or []
        if not isinstance(pts, list) or len(pts) < 3:
            continue
        points = []
        for p in pts:
            if not isinstance(p, dict):
                continue
            points.append({"x": _as_float(p.get("x")), "y": _as_float(p.get("y"))})
        if len(points) < 3:
            continue
        zones_out.append(
            {
                "id": str(z.get("id") or ""),
                "name": str(z.get("label") or z.get("name") or z.get("id") or "zone"),
                "enabled": bool(z.get("enabled", True)),
                "points": points,
            }
        )
    for ln in src.get("lines", []) or []:
        if not isinstance(ln, dict):
            continue
        p1 = ln.get("p1") or {}
        p2 = ln.get("p2") or {}
        if not isinstance(p1, dict) or not isinstance(p2, dict):
            continue
        lines_out.append(
            {
                "id": str(ln.get("id") or ""),
                "name": str(ln.get("label") or ln.get("name") or ln.get("id") or "line"),
                "enabled": bool(ln.get("enabled", True)),
                "p1": {"x": _as_float(p1.get("x")), "y": _as_float(p1.get("y"))},
                "p2": {"x": _as_float(p2.get("x")), "y": _as_float(p2.get("y"))},
            }
        )
    tags_out: List[Dict[str, Any]] = []
    for tg in src.get("tags", []) or []:
        if not isinstance(tg, dict):
            continue
        anchor = tg.get("anchor") if isinstance(tg.get("anchor"), dict) else {}
        ax = _as_float(anchor.get("x", tg.get("x", 0.5)), 0.5)
        ay = _as_float(anchor.get("y", tg.get("y", 0.5)), 0.5)
        tags_out.append(
            {
                "id": str(tg.get("id") or ""),
                "name": str(tg.get("label") or tg.get("name") or tg.get("id") or "tag"),
                "enabled": bool(tg.get("enabled", True)),
                "anchor": {"x": ax, "y": ay},
                "x": ax,
                "y": ay,
            }
        )
    return {"zones": zones_out, "lines": lines_out, "tags": tags_out}


def _track_centroid_norm(track: Dict[str, Any], frame_w: int, frame_h: int) -> Tuple[float, float]:
    center = track.get("center") if isinstance(track.get("center"), dict) else {}
    if "nx" in center and "ny" in center:
        return _as_float(center.get("nx"), 0.5), _as_float(center.get("ny"), 0.5)
    bb = track.get("bbox") if isinstance(track.get("bbox"), dict) else {}
    fw = max(1, int(frame_w))
    fh = max(1, int(frame_h))
    cx = _as_float(bb.get("x")) + _as_float(bb.get("w")) / 2.0
    cy = _as_float(bb.get("y")) + _as_float(bb.get("h")) / 2.0
    return cx / fw, cy / fh


def _line_cross_sign(px: float, py: float, p1: Dict[str, float], p2: Dict[str, float]) -> int:
    ax, ay = _as_float(p1.get("x")), _as_float(p1.get("y"))
    bx, by = _as_float(p2.get("x")), _as_float(p2.get("y"))
    cross = (bx - ax) * (py - ay) - (by - ay) * (px - ax)
    if abs(cross) <= 1e-6:
        return 0
    return 1 if cross > 0 else -1


def _movement_direction(history: Sequence[Dict[str, Any]]) -> Optional[str]:
    if len(history) < 2:
        return None
    a, b = history[-2], history[-1]
    dx = _as_float(b.get("x")) - _as_float(a.get("x"))
    if abs(dx) <= 1e-4:
        return None
    return "left_to_right" if dx > 0 else "right_to_left"


def _update_color_ema(rec: _TrackRecord, instant_color: Optional[str], *, alpha: float = COLOR_EMA_ALPHA) -> None:
    if not instant_color:
        return
    color = str(instant_color).strip().lower()
    if not color:
        return
    decay = max(0.0, min(1.0, 1.0 - float(alpha)))
    for key in list(rec.color_votes.keys()):
        rec.color_votes[key] *= decay
        if rec.color_votes[key] < 0.01:
            rec.color_votes.pop(key, None)
    rec.color_votes[color] = rec.color_votes.get(color, 0.0) + float(alpha)


def _dominant_color(rec: _TrackRecord) -> Optional[str]:
    if not rec.color_votes:
        return None
    return max(rec.color_votes, key=lambda k: rec.color_votes[k])


def _bbox_crop_box(bbox: Dict[str, float]) -> Tuple[int, int, int, int]:
    return (
        max(0, int(_as_float(bbox.get("x")))),
        max(0, int(_as_float(bbox.get("y")))),
        max(1, int(_as_float(bbox.get("w")))),
        max(1, int(_as_float(bbox.get("h")))),
    )


@dataclass
class _ZoneTrackState:
    inside: bool = False
    enter_streak: int = 0
    exit_streak: int = 0
    entered_at: Optional[float] = None
    dwell_met: bool = False


@dataclass
class _LineTrackState:
    side: int = 0
    pending_side: int = 0
    side_streak: int = 0


@dataclass
class _TagTrackState:
    near: bool = False
    near_streak: int = 0
    away_streak: int = 0


@dataclass
class _TrackRecord:
    track_id: int
    cls: str = "object"
    confidence: float = 0.0
    bbox: Dict[str, float] = field(default_factory=dict)
    centroid_history: Deque[Dict[str, Any]] = field(default_factory=lambda: deque(maxlen=CENTROID_HISTORY_LEN))
    zones: Dict[str, _ZoneTrackState] = field(default_factory=dict)
    lines: Dict[str, _LineTrackState] = field(default_factory=dict)
    tags: Dict[str, _TagTrackState] = field(default_factory=dict)
    missing_frames: int = 0
    lost: bool = False
    lost_at: Optional[float] = None
    last_bbox: Dict[str, float] = field(default_factory=dict)
    color_votes: Dict[str, float] = field(default_factory=dict)


@dataclass
class _LostTrackGhost:
    track_id: int
    bbox: Dict[str, float]
    lost_at: float
    cls: str
    confidence: float


class TrackSceneEngine:
    """
    Authoritative server-side track scene evaluator.

    State is keyed by (camera_id, tracker_namespace, track_id).
    """

    def __init__(
        self,
        *,
        hysteresis_frames: int = HYSTERESIS_FRAMES,
        dwell_sec: float = DEFAULT_DWELL_SEC,
        reacquire_iou: float = REACQUIRE_IOU_THRESHOLD,
        reacquire_window_sec: float = REACQUIRE_WINDOW_SEC,
    ) -> None:
        self.hysteresis_frames = max(1, int(hysteresis_frames))
        self.dwell_sec = max(0.0, float(dwell_sec))
        self.reacquire_iou = max(0.0, min(1.0, float(reacquire_iou)))
        self.reacquire_window_sec = max(0.0, float(reacquire_window_sec))
        self._tracks: Dict[Tuple[str, str, int], _TrackRecord] = {}
        self._lost_ghosts: Dict[str, List[_LostTrackGhost]] = {}

    def update(
        self,
        *,
        camera_id: str,
        tracks: Sequence[Dict[str, Any]],
        shapes: Optional[Dict[str, Any]],
        frame_w: int,
        frame_h: int,
        tracker_namespace: str = BACKEND_SORT_NAMESPACE,
        now: Optional[float] = None,
        dwell_sec: Optional[float] = None,
        frame_bgr: Any = None,
    ) -> List[Dict[str, Any]]:
        ts = float(now if now is not None else datetime.now(tz=timezone.utc).timestamp())
        dwell_threshold = self.dwell_sec if dwell_sec is None else max(0.0, float(dwell_sec))
        norm_shapes = normalize_shapes(shapes)
        events: List[Dict[str, Any]] = []

        seen: Set[int] = set()
        for tr in tracks or []:
            if not isinstance(tr, dict):
                continue
            try:
                tid = int(tr.get("id", tr.get("track_id", -1)))
            except Exception:
                continue
            if tid < 0:
                continue
            seen.add(tid)
            key = (str(camera_id), str(tracker_namespace), tid)
            rec = self._tracks.get(key)
            if rec is None:
                reacquired_from = self._match_reacquire(camera_id, tr, ts)
                rec = _TrackRecord(track_id=tid)
                if reacquired_from is not None:
                    old_key = (str(camera_id), str(tracker_namespace), int(reacquired_from))
                    old_rec = self._tracks.get(old_key)
                    if old_rec is not None:
                        self._inherit_track_scene_state(rec, old_rec)
                self._tracks[key] = rec
                if reacquired_from is not None:
                    events.append(
                        self._build_event(
                            event_type="track_reacquired",
                            camera_id=camera_id,
                            tracker_namespace=tracker_namespace,
                            track=tr,
                            frame_w=frame_w,
                            frame_h=frame_h,
                            timestamp=ts,
                            extra={
                                "previous_track_id": reacquired_from,
                                "reacquired_track_id": tid,
                            },
                            rec=rec,
                        )
                    )

            rec.missing_frames = 0
            rec.lost = False
            rec.lost_at = None
            rec.cls = str(tr.get("class") or tr.get("class_name") or rec.cls or "object")
            rec.confidence = _as_float(tr.get("confidence"), rec.confidence)
            bb = tr.get("bbox") if isinstance(tr.get("bbox"), dict) else {}
            rec.bbox = {
                "x": _as_float(bb.get("x")),
                "y": _as_float(bb.get("y")),
                "w": _as_float(bb.get("w")),
                "h": _as_float(bb.get("h")),
            }
            rec.last_bbox = dict(rec.bbox)
            nx, ny = _track_centroid_norm(tr, frame_w, frame_h)
            rec.centroid_history.append({"x": nx, "y": ny, "t": _utc_iso(ts)})

            if frame_bgr is not None:
                instant_color = estimate_dominant_color_from_bgr(frame_bgr, crop_box=_bbox_crop_box(rec.bbox))
                _update_color_ema(rec, instant_color)

            events.extend(
                self._eval_zones(
                    rec=rec,
                    camera_id=camera_id,
                    tracker_namespace=tracker_namespace,
                    track=tr,
                    zones=norm_shapes.get("zones", []),
                    frame_w=frame_w,
                    frame_h=frame_h,
                    timestamp=ts,
                    dwell_threshold=dwell_threshold,
                )
            )
            events.extend(
                self._eval_lines(
                    rec=rec,
                    camera_id=camera_id,
                    tracker_namespace=tracker_namespace,
                    track=tr,
                    lines=norm_shapes.get("lines", []),
                    frame_w=frame_w,
                    frame_h=frame_h,
                    timestamp=ts,
                )
            )
            events.extend(
                self._eval_tags(
                    rec=rec,
                    camera_id=camera_id,
                    tracker_namespace=tracker_namespace,
                    track=tr,
                    tags=norm_shapes.get("tags", []),
                    frame_w=frame_w,
                    frame_h=frame_h,
                    timestamp=ts,
                )
            )

        prefix = (str(camera_id), str(tracker_namespace))
        for key, rec in list(self._tracks.items()):
            if key[:2] != prefix:
                continue
            if rec.track_id in seen:
                continue
            rec.missing_frames += 1
            if rec.lost:
                continue
            if rec.missing_frames <= LOST_GRACE_FRAMES:
                continue
            rec.lost = True
            rec.lost_at = ts
            ghosts = self._lost_ghosts.setdefault(str(camera_id), [])
            ghosts.append(
                _LostTrackGhost(
                    track_id=rec.track_id,
                    bbox=dict(rec.last_bbox or rec.bbox),
                    lost_at=ts,
                    cls=rec.cls,
                    confidence=rec.confidence,
                )
            )
            events.append(
                self._build_event(
                    event_type="track_lost",
                    camera_id=camera_id,
                    tracker_namespace=tracker_namespace,
                    track={
                        "id": rec.track_id,
                        "class": rec.cls,
                        "confidence": rec.confidence,
                        "bbox": dict(rec.last_bbox or rec.bbox),
                    },
                    frame_w=frame_w,
                    frame_h=frame_h,
                    timestamp=ts,
                    rec=rec,
                )
            )

        self._prune_lost_ghosts(camera_id, ts)
        self._prune_stale_tracks(prefix, seen)
        zone_counts = self._zone_occupancy_counts(prefix)
        if zone_counts and events:
            for ev in events:
                zid = str(ev.get("shape_id") or "")
                if zid and zid in zone_counts:
                    total, by_class = zone_counts[zid]
                    ev["zone_track_count"] = total
                    ev["zone_track_counts"] = dict(by_class)
        return events

    @staticmethod
    def _inherit_track_scene_state(rec: _TrackRecord, old_rec: _TrackRecord) -> None:
        """Preserve zone/line occupancy across track ID reacquire to avoid duplicate enters."""
        rec.zones = {
            zid: _ZoneTrackState(
                inside=zs.inside,
                enter_streak=zs.enter_streak,
                exit_streak=zs.exit_streak,
                entered_at=zs.entered_at,
                dwell_met=zs.dwell_met,
            )
            for zid, zs in old_rec.zones.items()
        }
        rec.lines = {
            lid: _LineTrackState(
                side=ls.side,
                pending_side=ls.pending_side,
                side_streak=ls.side_streak,
            )
            for lid, ls in old_rec.lines.items()
        }
        rec.tags = {
            tid: _TagTrackState(
                near=ts.near,
                near_streak=ts.near_streak,
                away_streak=ts.away_streak,
            )
            for tid, ts in old_rec.tags.items()
        }
        if old_rec.centroid_history:
            rec.centroid_history = deque(old_rec.centroid_history, maxlen=CENTROID_HISTORY_LEN)

    def _zone_occupancy_counts(
        self,
        prefix: Tuple[str, str],
    ) -> Dict[str, Tuple[int, Dict[str, int]]]:
        """Per-zone occupancy: total tracks inside and counts by class."""
        out: Dict[str, Tuple[int, Dict[str, int]]] = {}
        for key, rec in self._tracks.items():
            if key[:2] != prefix:
                continue
            if rec.lost:
                continue
            cls = str(rec.cls or "object").strip().lower() or "object"
            for zid, zs in rec.zones.items():
                if not zs.inside:
                    continue
                total, by_class = out.get(zid, (0, {}))
                by_class = dict(by_class)
                by_class[cls] = by_class.get(cls, 0) + 1
                out[zid] = (total + 1, by_class)
        return out

    def _eval_zones(
        self,
        *,
        rec: _TrackRecord,
        camera_id: str,
        tracker_namespace: str,
        track: Dict[str, Any],
        zones: Sequence[Dict[str, Any]],
        frame_w: int,
        frame_h: int,
        timestamp: float,
        dwell_threshold: float,
    ) -> List[Dict[str, Any]]:
        events: List[Dict[str, Any]] = []
        nx, ny = _track_centroid_norm(track, frame_w, frame_h)
        for zone in zones:
            if not zone.get("enabled", True):
                continue
            zid = str(zone.get("id") or "")
            if not zid:
                continue
            zs = rec.zones.setdefault(zid, _ZoneTrackState())
            raw_inside = point_in_polygon(nx, ny, zone.get("points") or [])

            if raw_inside:
                zs.exit_streak = 0
                if zs.inside:
                    zs.enter_streak = self.hysteresis_frames
                else:
                    zs.enter_streak += 1
                    if zs.enter_streak >= self.hysteresis_frames:
                        zs.inside = True
                        zs.entered_at = timestamp
                        zs.dwell_met = False
                        events.append(
                            self._build_event(
                                event_type="zone_enter",
                                camera_id=camera_id,
                                tracker_namespace=tracker_namespace,
                                track=track,
                                frame_w=frame_w,
                                frame_h=frame_h,
                                timestamp=timestamp,
                                shape_id=zid,
                                shape_name=str(zone.get("name") or zid),
                                rec=rec,
                            )
                        )
            else:
                zs.enter_streak = 0
                if not zs.inside:
                    zs.exit_streak = 0
                else:
                    zs.exit_streak += 1
                    if zs.exit_streak >= self.hysteresis_frames:
                        dwell = 0.0
                        if zs.entered_at is not None:
                            dwell = max(0.0, timestamp - float(zs.entered_at))
                        events.append(
                            self._build_event(
                                event_type="zone_exit",
                                camera_id=camera_id,
                                tracker_namespace=tracker_namespace,
                                track=track,
                                frame_w=frame_w,
                                frame_h=frame_h,
                                timestamp=timestamp,
                                shape_id=zid,
                                shape_name=str(zone.get("name") or zid),
                                dwell_sec=dwell,
                                rec=rec,
                            )
                        )
                        zs.inside = False
                        zs.entered_at = None
                        zs.dwell_met = False
                        zs.exit_streak = 0

            if zs.inside and zs.entered_at is not None and not zs.dwell_met:
                elapsed = max(0.0, timestamp - float(zs.entered_at))
                if elapsed >= dwell_threshold:
                    zs.dwell_met = True
                    events.append(
                        self._build_event(
                            event_type="dwell_met",
                            camera_id=camera_id,
                            tracker_namespace=tracker_namespace,
                            track=track,
                            frame_w=frame_w,
                            frame_h=frame_h,
                            timestamp=timestamp,
                            shape_id=zid,
                            shape_name=str(zone.get("name") or zid),
                            dwell_sec=elapsed,
                            rec=rec,
                        )
                    )
        return events

    def _eval_lines(
        self,
        *,
        rec: _TrackRecord,
        camera_id: str,
        tracker_namespace: str,
        track: Dict[str, Any],
        lines: Sequence[Dict[str, Any]],
        frame_w: int,
        frame_h: int,
        timestamp: float,
    ) -> List[Dict[str, Any]]:
        events: List[Dict[str, Any]] = []
        nx, ny = _track_centroid_norm(track, frame_w, frame_h)
        for line in lines:
            if not line.get("enabled", True):
                continue
            lid = str(line.get("id") or "")
            if not lid:
                continue
            ls = rec.lines.setdefault(lid, _LineTrackState())
            sign = _line_cross_sign(nx, ny, line.get("p1") or {}, line.get("p2") or {})
            if sign == 0:
                continue
            if ls.side == 0:
                ls.side = sign
                ls.pending_side = sign
                ls.side_streak = self.hysteresis_frames
                continue
            if sign == ls.pending_side:
                ls.side_streak = min(self.hysteresis_frames, ls.side_streak + 1)
            else:
                ls.pending_side = sign
                ls.side_streak = 1
            if ls.side_streak >= self.hysteresis_frames and sign != ls.side:
                prev = ls.side
                direction = "positive" if sign > prev else "negative"
                move_dir = _movement_direction(list(rec.centroid_history))
                if move_dir:
                    direction = move_dir
                events.append(
                    self._build_event(
                        event_type="line_cross",
                        camera_id=camera_id,
                        tracker_namespace=tracker_namespace,
                        track=track,
                        frame_w=frame_w,
                        frame_h=frame_h,
                        timestamp=timestamp,
                        shape_id=lid,
                        shape_name=str(line.get("name") or lid),
                        direction=direction,
                        extra={"cross_sign_from": prev, "cross_sign_to": sign},
                        rec=rec,
                    )
                )
                ls.side = sign
        return events

    def _eval_tags(
        self,
        *,
        rec: _TrackRecord,
        camera_id: str,
        tracker_namespace: str,
        track: Dict[str, Any],
        tags: Sequence[Dict[str, Any]],
        frame_w: int,
        frame_h: int,
        timestamp: float,
    ) -> List[Dict[str, Any]]:
        events: List[Dict[str, Any]] = []
        nx, ny = _track_centroid_norm(track, frame_w, frame_h)
        for tag in tags:
            if not tag.get("enabled", True):
                continue
            tid = str(tag.get("id") or "")
            if not tid:
                continue
            anchor = tag.get("anchor") if isinstance(tag.get("anchor"), dict) else {}
            tx = _as_float(anchor.get("x", tag.get("x", 0.5)), 0.5)
            ty = _as_float(anchor.get("y", tag.get("y", 0.5)), 0.5)
            dist = math.hypot(nx - tx, ny - ty)
            ts = rec.tags.setdefault(tid, _TagTrackState())
            raw_near = dist <= TAG_NEAR_RADIUS
            if raw_near:
                ts.away_streak = 0
                if ts.near:
                    ts.near_streak = self.hysteresis_frames
                else:
                    ts.near_streak += 1
                    if ts.near_streak >= self.hysteresis_frames:
                        ts.near = True
                        events.append(
                            self._build_event(
                                event_type="near_tag",
                                camera_id=camera_id,
                                tracker_namespace=tracker_namespace,
                                track=track,
                                frame_w=frame_w,
                                frame_h=frame_h,
                                timestamp=timestamp,
                                shape_id=tid,
                                shape_name=str(tag.get("name") or tid),
                                rec=rec,
                            )
                        )
            else:
                ts.near_streak = 0
                if not ts.near:
                    ts.away_streak = 0
                else:
                    ts.away_streak += 1
                    if ts.away_streak >= self.hysteresis_frames:
                        ts.near = False
                        ts.away_streak = 0
        return events

    def _build_event(
        self,
        *,
        event_type: str,
        camera_id: str,
        tracker_namespace: str,
        track: Dict[str, Any],
        frame_w: int,
        frame_h: int,
        timestamp: float,
        rec: _TrackRecord,
        shape_id: Optional[str] = None,
        shape_name: Optional[str] = None,
        direction: Optional[str] = None,
        dwell_sec: Optional[float] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        nx, ny = _track_centroid_norm(track, frame_w, frame_h)
        bb = track.get("bbox") if isinstance(track.get("bbox"), dict) else {}
        try:
            tid = int(track.get("id", track.get("track_id", rec.track_id)))
        except Exception:
            tid = rec.track_id
        payload: Dict[str, Any] = {
            "event_type": event_type,
            "camera_id": str(camera_id),
            "tracker_namespace": str(tracker_namespace),
            "track_id": tid,
            "class": str(track.get("class") or track.get("class_name") or rec.cls or "object"),
            "confidence": _as_float(track.get("confidence"), rec.confidence),
            "bbox": {
                "x": _as_float(bb.get("x")),
                "y": _as_float(bb.get("y")),
                "w": _as_float(bb.get("w")),
                "h": _as_float(bb.get("h")),
            },
            "centroid_norm": {"x": nx, "y": ny},
            "timestamp": _utc_iso(timestamp),
            "centroid_history": list(rec.centroid_history),
        }
        if shape_id is not None:
            payload["shape_id"] = shape_id
        if shape_name is not None:
            payload["shape_name"] = shape_name
        if direction is not None:
            payload["direction"] = direction
        if dwell_sec is not None:
            payload["dwell_sec"] = round(float(dwell_sec), 3)
        dom = _dominant_color(rec)
        if dom:
            payload["dominant_color"] = dom
        if extra:
            payload.update(extra)
        return payload

    def _match_reacquire(
        self,
        camera_id: str,
        track: Dict[str, Any],
        now: float,
    ) -> Optional[int]:
        ghosts = self._lost_ghosts.get(str(camera_id)) or []
        if not ghosts:
            return None
        bb = track.get("bbox") if isinstance(track.get("bbox"), dict) else {}
        best_id: Optional[int] = None
        best_iou = 0.0
        for ghost in ghosts:
            if (now - ghost.lost_at) > self.reacquire_window_sec:
                continue
            iou = _bbox_iou(bb, ghost.bbox)
            if iou >= self.reacquire_iou and iou > best_iou:
                best_iou = iou
                best_id = ghost.track_id
        if best_id is not None:
            self._lost_ghosts[str(camera_id)] = [g for g in ghosts if g.track_id != best_id]
        return best_id

    def _prune_lost_ghosts(self, camera_id: str, now: float) -> None:
        ghosts = self._lost_ghosts.get(str(camera_id))
        if not ghosts:
            return
        kept = [g for g in ghosts if (now - g.lost_at) <= self.reacquire_window_sec]
        if kept:
            self._lost_ghosts[str(camera_id)] = kept
        else:
            self._lost_ghosts.pop(str(camera_id), None)

    def _prune_stale_tracks(self, prefix: Tuple[str, str], seen: Set[int]) -> None:
        drop: List[Tuple[str, str, int]] = []
        for key, rec in self._tracks.items():
            if key[:2] != prefix:
                continue
            if rec.track_id in seen:
                continue
            if rec.lost and rec.missing_frames > LOST_GRACE_FRAMES + 5:
                drop.append(key)
        for key in drop:
            self._tracks.pop(key, None)

    def reset_camera(self, camera_id: str, tracker_namespace: str = BACKEND_SORT_NAMESPACE) -> None:
        prefix = (str(camera_id), str(tracker_namespace))
        for key in list(self._tracks.keys()):
            if key[:2] == prefix:
                self._tracks.pop(key, None)
        self._lost_ghosts.pop(str(camera_id), None)
