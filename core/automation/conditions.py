from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence, Tuple


def _as_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return default


def _as_int(x: Any, default: int = 0) -> int:
    try:
        return int(x)
    except Exception:
        return default


def point_in_polygon(px: float, py: float, poly: Sequence[Dict[str, Any]]) -> bool:
    """
    Ray casting algorithm, expects poly points with {x,y} in normalized space (0..1).
    """
    if not poly or len(poly) < 3:
        return False
    inside = False
    j = len(poly) - 1
    for i in range(len(poly)):
        xi = _as_float(poly[i].get("x"))
        yi = _as_float(poly[i].get("y"))
        xj = _as_float(poly[j].get("x"))
        yj = _as_float(poly[j].get("y"))
        intersects = ((yi > py) != (yj > py)) and (px < (xj - xi) * (py - yi) / ((yj - yi) or 1e-9) + xi)
        if intersects:
            inside = not inside
        j = i
    return inside


def dist_point_to_segment(px: float, py: float, ax: float, ay: float, bx: float, by: float) -> float:
    """Distance from point P to line segment AB in normalized coordinate space."""
    abx = bx - ax
    aby = by - ay
    apx = px - ax
    apy = py - ay
    ab_len2 = abx * abx + aby * aby
    if ab_len2 <= 1e-12:
        return math.sqrt(apx * apx + apy * apy)
    t = max(0.0, min(1.0, (apx * abx + apy * aby) / ab_len2))
    cx = ax + t * abx
    cy = ay + t * aby
    dx = px - cx
    dy = py - cy
    return math.sqrt(dx * dx + dy * dy)


def parse_iso(ts: Optional[str]) -> Optional[datetime]:
    if not ts:
        return None
    try:
        return datetime.fromisoformat(ts.replace("Z", "+00:00"))
    except Exception:
        return None


def _time_to_minutes(hhmm: str) -> Optional[int]:
    try:
        parts = (hhmm or "").strip().split(":")
        if len(parts) != 2:
            return None
        h = int(parts[0])
        m = int(parts[1])
        if h < 0 or h > 23 or m < 0 or m > 59:
            return None
        return h * 60 + m
    except Exception:
        return None


def is_time_allowed(now: datetime, window: Optional[Dict[str, Any]]) -> bool:
    """
    window: { start: "HH:MM", end: "HH:MM", days: [0..6] } (0=Mon)
    If window omitted, allow.
    """
    if not window or not isinstance(window, dict):
        return True
    days = window.get("days")
    if isinstance(days, list) and days:
        # datetime.weekday(): Monday=0
        if now.weekday() not in [int(d) for d in days if str(d).isdigit()]:
            return False
    start_min = _time_to_minutes(str(window.get("start") or ""))
    end_min = _time_to_minutes(str(window.get("end") or ""))
    if start_min is None or end_min is None:
        return True
    cur = now.hour * 60 + now.minute
    if start_min <= end_min:
        return start_min <= cur <= end_min
    # overnight window (e.g., 22:00..06:00)
    return cur >= start_min or cur <= end_min


@dataclass
class EvalContext:
    kind: str
    camera_id: str
    timestamp: datetime
    frame_w: Optional[int]
    frame_h: Optional[int]
    detections: List[Dict[str, Any]]
    tracks: List[Dict[str, Any]]
    payload: Dict[str, Any]

    def normalized_centers_from_detections(self) -> List[Tuple[float, float, Dict[str, Any]]]:
        out: List[Tuple[float, float, Dict[str, Any]]] = []
        if not self.frame_w or not self.frame_h:
            return out
        fw = max(1, int(self.frame_w))
        fh = max(1, int(self.frame_h))
        for d in self.detections or []:
            b = d.get("bbox") or {}
            x = _as_float(b.get("x"))
            y = _as_float(b.get("y"))
            w = _as_float(b.get("w"))
            h = _as_float(b.get("h"))
            cx = (x + w / 2.0) / fw
            cy = (y + h / 2.0) / fh
            out.append((cx, cy, d))
        return out

    def normalized_centers_from_tracks(self) -> List[Tuple[float, float, Dict[str, Any]]]:
        out: List[Tuple[float, float, Dict[str, Any]]] = []
        if not self.frame_w or not self.frame_h:
            return out
        fw = max(1, int(self.frame_w))
        fh = max(1, int(self.frame_h))
        for t in self.tracks or []:
            b = t.get("bbox") or {}
            x = _as_float(b.get("x"))
            y = _as_float(b.get("y"))
            w = _as_float(b.get("w"))
            h = _as_float(b.get("h"))
            cx = (x + w / 2.0) / fw
            cy = (y + h / 2.0) / fh
            out.append((cx, cy, t))
        return out


def _class_name(obj: Dict[str, Any]) -> str:
    # detections use 'class'; tracks often use 'class'
    return str(obj.get("class") or obj.get("class_name") or obj.get("label") or "object").strip().lower()


def _confidence(obj: Dict[str, Any]) -> float:
    return _as_float(obj.get("confidence"), 0.0)


def filter_objects(
    objects: Sequence[Dict[str, Any]],
    *,
    classes: Optional[Sequence[str]] = None,
    min_confidence: Optional[float] = None,
) -> List[Dict[str, Any]]:
    allowed = [c.strip().lower() for c in (classes or []) if str(c).strip()]
    minc = _as_float(min_confidence, 0.0) if min_confidence is not None else None
    out: List[Dict[str, Any]] = []
    for obj in objects or []:
        cls = _class_name(obj)
        conf = _confidence(obj)
        if allowed and cls not in allowed:
            continue
        if minc is not None and conf < minc:
            continue
        out.append(obj)
    return out


def shape_match(
    *,
    shape: Dict[str, Any],
    ctx: EvalContext,
    prefer: str = "detections",
    line_threshold: float = 0.05,
    tag_radius: float = 0.10,
) -> bool:
    """
    shape: one of zone/line/tag from camera_shapes.
    - zone: points: [{x,y},...]
    - line: p1:{x,y}, p2:{x,y}
    - tag: x,y
    """
    kind = str(shape.get("kind") or shape.get("type") or "").lower().strip()
    # Infer by fields if no kind set
    if not kind:
        if "points" in shape:
            kind = "zone"
        elif "p1" in shape and "p2" in shape:
            kind = "line"
        elif "x" in shape and "y" in shape:
            kind = "tag"

    candidates: List[Tuple[float, float, Dict[str, Any]]] = []
    if prefer == "tracks":
        candidates = ctx.normalized_centers_from_tracks() or ctx.normalized_centers_from_detections()
    else:
        candidates = ctx.normalized_centers_from_detections() or ctx.normalized_centers_from_tracks()

    if not candidates:
        return False

    if kind == "zone":
        poly = shape.get("points") or []
        return any(point_in_polygon(cx, cy, poly) for (cx, cy, _obj) in candidates)

    if kind == "line":
        p1 = shape.get("p1") or {}
        p2 = shape.get("p2") or {}
        ax = _as_float(p1.get("x"))
        ay = _as_float(p1.get("y"))
        bx = _as_float(p2.get("x"))
        by = _as_float(p2.get("y"))
        thr = max(0.0, float(line_threshold))
        return any(dist_point_to_segment(cx, cy, ax, ay, bx, by) <= thr for (cx, cy, _obj) in candidates)

    if kind == "tag":
        tx = _as_float(shape.get("x"))
        ty = _as_float(shape.get("y"))
        rr = max(0.0, float(tag_radius))
        return any(math.sqrt((cx - tx) ** 2 + (cy - ty) ** 2) <= rr for (cx, cy, _obj) in candidates)

    return False


def build_eval_context(kind: str, camera_id: str, payload: Dict[str, Any]) -> EvalContext:
    ts = parse_iso(payload.get("timestamp") if isinstance(payload, dict) else None) or datetime.now()
    fw = payload.get("frame_width") if isinstance(payload, dict) else None
    fh = payload.get("frame_height") if isinstance(payload, dict) else None

    detections = []
    tracks = []
    if isinstance(payload, dict):
        if isinstance(payload.get("detections"), list):
            detections = payload.get("detections") or []
        if isinstance(payload.get("tracks"), list):
            tracks = payload.get("tracks") or []
        # motion payload sometimes nests tracks under motion
        if not tracks and isinstance(payload.get("motion"), dict) and isinstance(payload["motion"].get("tracks"), list):
            tracks = payload["motion"].get("tracks") or []
        # motion payload nests frame dims under motion
        if (fw is None or fh is None) and isinstance(payload.get("motion"), dict):
            fw = fw or payload["motion"].get("frame_width")
            fh = fh or payload["motion"].get("frame_height")

    return EvalContext(
        kind=str(kind),
        camera_id=str(camera_id),
        timestamp=ts,
        frame_w=_as_int(fw, 0) or None,
        frame_h=_as_int(fh, 0) or None,
        detections=detections if isinstance(detections, list) else [],
        tracks=tracks if isinstance(tracks, list) else [],
        payload=payload if isinstance(payload, dict) else {},
    )


def matches_rule(
    *,
    rule: Dict[str, Any],
    ctx: EvalContext,
    shape: Optional[Dict[str, Any]] = None,
) -> Tuple[bool, Dict[str, Any]]:
    """
    Safe rule evaluation.
    Returns (match, details) where details is suitable for logging/observability.
    """
    rule_id = str(rule.get("id") or "")
    trigger = str(rule.get("trigger") or "any").strip().lower()
    enabled = bool(rule.get("enabled", True))
    if not enabled:
        return False, {"reason": "disabled", "rule_id": rule_id}

    if trigger not in ("", "*", "any") and trigger != str(ctx.kind).lower():
        return False, {"reason": "trigger_mismatch", "rule_id": rule_id, "trigger": trigger, "event_kind": ctx.kind}

    # Optional camera filter
    rule_cam = rule.get("camera_id")
    if rule_cam and str(rule_cam) != str(ctx.camera_id):
        return False, {"reason": "camera_mismatch", "rule_id": rule_id}

    conditions = rule.get("conditions") if isinstance(rule.get("conditions"), dict) else {}

    # Time windows
    if not is_time_allowed(ctx.timestamp, conditions.get("time_window")):
        return False, {"reason": "time_window", "rule_id": rule_id}

    # Object filters (apply to detections if present, otherwise tracks)
    classes = conditions.get("classes") or conditions.get("object_classes")
    min_conf = conditions.get("min_confidence")
    # Support percent (0..100) by converting if needed
    try:
        if min_conf is not None and float(min_conf) > 1.0:
            min_conf = float(min_conf) / 100.0
    except Exception:
        pass

    objs = ctx.detections if ctx.detections else ctx.tracks
    filtered = filter_objects(objs, classes=classes if isinstance(classes, list) else None, min_confidence=min_conf)
    if (classes or min_conf is not None) and not filtered:
        return False, {"reason": "object_filter", "rule_id": rule_id}

    # Shape interaction (zone/line/tag)
    if shape:
        if not shape_match(
            shape=shape,
            ctx=ctx,
            prefer=str(conditions.get("shape_prefer") or "detections"),
            line_threshold=_as_float(conditions.get("line_threshold"), 0.05),
            tag_radius=_as_float(conditions.get("tag_radius"), 0.10),
        ):
            return False, {"reason": "shape_no_match", "rule_id": rule_id, "shape_id": shape.get("id")}

    return True, {
        "rule_id": rule_id,
        "camera_id": ctx.camera_id,
        "event_kind": ctx.kind,
        "filtered_object_count": len(filtered),
        "frame_w": ctx.frame_w,
        "frame_h": ctx.frame_h,
    }



