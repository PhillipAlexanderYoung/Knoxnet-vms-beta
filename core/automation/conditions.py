from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence, Tuple

BACKEND_SORT_NAMESPACE = "backend_sort"
MOTION_BOX_NAMESPACE = "motion_box"
ANY_INTERACTION_EVENTS = frozenset(
    {
        "zone_enter",
        "zone_exit",
        "dwell_met",
        "line_cross",
        "near_tag",
        "zone_touch",
        "track_present",
    }
)
DEFAULT_PATH_MATCH_TOLERANCE = 0.20
# Minimum direction dot product (0.35 ≈ 70° cone); rejects opposite direction (~180°).
DEFAULT_MIN_DIRECTION_DOT = 0.35
# Relaxed cone for auto/direction-only drawn-path rules (~78°).
AUTO_MIN_DIRECTION_DOT = 0.20
DEFAULT_PATH_MAX_ANGLE_RAD = math.acos(DEFAULT_MIN_DIRECTION_DOT)
MIN_DIRECTION_DISPLACEMENT = 0.006
PATH_DWELL_INSIDE_RATIO = 0.35
PATH_SPEED_RATIO_MIN = 0.1
PATH_SPEED_RATIO_MAX = 10.0
# Endpoint alignment only for very short drawn paths; longer paths use polyline distance.
ENDPOINT_ALIGNMENT_MAX_PATH_LEN = 0.20


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


def bucket_rgb_color(r: float, g: float, b: float) -> Optional[str]:
    """Map mean RGB to a coarse color bucket (shared with event index)."""
    yv = 0.2126 * r + 0.7152 * g + 0.0722 * b
    if yv >= 220:
        return "white"
    if yv <= 35:
        return "black"
    mx = max(r, g, b)
    mn = min(r, g, b)
    sat = 0 if mx == 0 else (mx - mn) / mx
    if sat < 0.22:
        return "white" if yv >= 125 else "gray"
    if r > g * 1.2 and r > b * 1.2:
        return "red" if r < 170 else "yellow"
    if g > r * 1.2 and g > b * 1.2:
        return "green"
    if b > r * 1.2 and b > g * 1.2:
        return "blue"
    if r > 120 and g > 90 and b < 90:
        return "brown"
    return None


def estimate_dominant_color_from_bgr(
    frame: Any,
    *,
    crop_box: Optional[Tuple[int, int, int, int]] = None,
) -> Optional[str]:
    """Estimate dominant color bucket from a BGR frame, optionally cropped to bbox."""
    try:
        import cv2  # type: ignore
    except Exception:
        return None
    try:
        img = frame
        if crop_box:
            x, y, w, h = crop_box
            x = max(0, int(x))
            y = max(0, int(y))
            w = max(1, int(w))
            h = max(1, int(h))
            img = img[y : y + h, x : x + w]
        if img is None or getattr(img, "size", 0) == 0:
            return None
        small = cv2.resize(img, (64, 64))
        b, g, r = cv2.split(small)
        return bucket_rgb_color(float(r.mean()), float(g.mean()), float(b.mean()))
    except Exception:
        return None


def _normalize_color_name(value: Any) -> Optional[str]:
    if value is None:
        return None
    s = str(value).strip().lower()
    return s or None


def _color_condition_matches(
    payload: Dict[str, Any],
    track_obj: Dict[str, Any],
    conditions: Dict[str, Any],
    *,
    tracker_namespace: str = "",
) -> bool:
    """Return True when no color filter is set or payload color matches."""
    want = _normalize_color_name(conditions.get("color") or conditions.get("dominant_color"))
    if not want:
        return True
    if tracker_namespace and not color_filter_applies_to_namespace(tracker_namespace, conditions):
        return True
    got = _normalize_color_name(
        payload.get("dominant_color") or payload.get("color") or track_obj.get("dominant_color")
    )
    return bool(got) and got == want


def _resolve_zone_track_count(payload: Dict[str, Any], classes: Any) -> Optional[int]:
    allowed = classes if isinstance(classes, list) else ([classes] if classes else [])
    allowed_norm = [str(c).strip().lower() for c in allowed if str(c).strip()]
    if allowed_norm:
        by_class = payload.get("zone_track_counts")
        if isinstance(by_class, dict):
            return sum(int(by_class.get(c, 0) or 0) for c in allowed_norm)
    raw = payload.get("zone_track_count")
    if raw is None:
        return None
    try:
        return int(raw)
    except Exception:
        return None


def _count_condition_matches(payload: Dict[str, Any], conditions: Dict[str, Any]) -> bool:
    count_min = conditions.get("count_min")
    count_max = conditions.get("count_max")
    if count_min is None and count_max is None:
        return True
    classes = conditions.get("classes") or conditions.get("object_classes")
    count = _resolve_zone_track_count(payload, classes)
    if count is None:
        return False
    if count_min is not None:
        try:
            if count < int(count_min):
                return False
        except Exception:
            return False
    if count_max is not None:
        try:
            if count > int(count_max):
                return False
        except Exception:
            return False
    return True


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

    detections: List[Dict[str, Any]] = []
    tracks: List[Dict[str, Any]] = []
    if isinstance(payload, dict):
        if str(kind).lower() == "track_event":
            track_obj = {
                "id": payload.get("track_id"),
                "track_id": payload.get("track_id"),
                "class": payload.get("class"),
                "class_name": payload.get("class"),
                "confidence": payload.get("confidence"),
                "bbox": payload.get("bbox") if isinstance(payload.get("bbox"), dict) else {},
                "direction": payload.get("direction"),
                "shape_id": payload.get("shape_id"),
                "shape_name": payload.get("shape_name"),
                "dwell_sec": payload.get("dwell_sec"),
                "tracker_namespace": payload.get("tracker_namespace"),
                "event_type": payload.get("event_type"),
                "dominant_color": payload.get("dominant_color") or payload.get("color"),
                "zone_track_count": payload.get("zone_track_count"),
                "zone_track_counts": payload.get("zone_track_counts"),
                "centroid_history": payload.get("centroid_history"),
            }
            tracks = [track_obj]
            detections = [track_obj]
            fw = fw or payload.get("frame_w")
            fh = fh or payload.get("frame_h")
        else:
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


def _normalize_trigger(value: Any) -> str:
    return str(value or "any").strip().lower()


def _extract_xy_points(points: Sequence[Any]) -> List[Tuple[float, float]]:
    out: List[Tuple[float, float]] = []
    for p in points or []:
        if isinstance(p, dict):
            out.append((_as_float(p.get("x")), _as_float(p.get("y"))))
        elif isinstance(p, (list, tuple)) and len(p) >= 2:
            out.append((_as_float(p[0]), _as_float(p[1])))
    return out


def _segment_direction(p0: Tuple[float, float], p1: Tuple[float, float]) -> Tuple[float, float]:
    dx = p1[0] - p0[0]
    dy = p1[1] - p0[1]
    length = math.hypot(dx, dy)
    if length <= 1e-9:
        return 0.0, 0.0
    return dx / length, dy / length


def _angle_between(v1: Tuple[float, float], v2: Tuple[float, float]) -> float:
    dot = max(-1.0, min(1.0, v1[0] * v2[0] + v1[1] * v2[1]))
    return math.acos(dot)


def _point_to_polyline_dist(point: Tuple[float, float], polyline: Sequence[Tuple[float, float]]) -> float:
    best = 999.0
    for i in range(len(polyline) - 1):
        ax, ay = polyline[i]
        bx, by = polyline[i + 1]
        d = dist_point_to_segment(point[0], point[1], ax, ay, bx, by)
        best = min(best, d)
    return best


def _avg_distance_track_to_path(
    track: Sequence[Tuple[float, float]],
    ref: Sequence[Tuple[float, float]],
) -> float:
    if not track or not ref:
        return 999.0
    total = sum(_point_to_polyline_dist(tp, ref) for tp in track)
    return total / len(track)


def _path_polyline_length(ref: Sequence[Tuple[float, float]]) -> float:
    total = 0.0
    for i in range(len(ref) - 1):
        ax, ay = ref[i]
        bx, by = ref[i + 1]
        total += math.hypot(bx - ax, by - ay)
    return total


def _track_displacement(track: Sequence[Tuple[float, float]]) -> float:
    if len(track) < 2:
        return 0.0
    return math.hypot(track[-1][0] - track[0][0], track[-1][1] - track[0][1])


def _points_inside_shape_ratio(
    points: Sequence[Tuple[float, float]],
    shape: Optional[Dict[str, Any]],
) -> float:
    if not shape or not points:
        return 0.0
    kind = str(shape.get("kind") or "").strip().lower()
    if kind != "zone":
        return 0.0
    poly = shape.get("pts") or shape.get("points") or []
    if len(poly) < 3:
        return 0.0
    inside = sum(1 for px, py in points if point_in_polygon(px, py, poly))
    return inside / len(points)


def _bbox_side(
    shape: Optional[Dict[str, Any]],
    px: float,
    py: float,
) -> str:
    """Classify a normalized point as left/right/top/bottom relative to shape bbox center."""
    min_x, min_y, max_x, max_y = _shape_bounds(shape or {})
    cx = (min_x + max_x) / 2.0
    cy = (min_y + max_y) / 2.0
    dx = px - cx
    dy = py - cy
    if abs(dx) >= abs(dy):
        return "right" if dx > 0 else "left"
    return "bottom" if dy > 0 else "top"


def compute_path_direction_gate(
    motion_path: Sequence[Any],
    shape: Optional[Dict[str, Any]] = None,
) -> Optional[Dict[str, Any]]:
    """
    Derive explicit direction intent from a user-drawn path.

    Stores a normalized travel vector plus entry/exit sides relative to the shape bbox
    so opposite-direction rules on the same zone can be distinguished robustly.
    """
    ref = _extract_xy_points(motion_path)
    if len(ref) < 2:
        return None
    start, end = ref[0], ref[-1]
    ref_dir = _segment_direction(start, end)
    if ref_dir == (0.0, 0.0):
        return None
    gate: Dict[str, Any] = {
        "path_direction": {"dx": round(ref_dir[0], 6), "dy": round(ref_dir[1], 6)},
    }
    if isinstance(shape, dict):
        entry = _bbox_side(shape, start[0], start[1])
        exit_side = _bbox_side(shape, end[0], end[1])
        gate["entry_side"] = entry
        gate["exit_side"] = exit_side
    return gate


def _direction_gate_from_conditions(
    conditions: Dict[str, Any],
    *,
    motion_path: Optional[Sequence[Any]] = None,
    shape: Optional[Dict[str, Any]] = None,
) -> Optional[Dict[str, Any]]:
    stored = conditions.get("path_direction_gate")
    if isinstance(stored, dict) and stored.get("path_direction"):
        return dict(stored)
    path = motion_path if motion_path is not None else conditions.get("motion_path")
    if isinstance(path, list) and len(path) >= 2:
        return compute_path_direction_gate(path, shape)
    return None


def _match_direction_gate(
    track: Sequence[Tuple[float, float]],
    *,
    gate: Dict[str, Any],
    shape: Optional[Dict[str, Any]],
    min_dot: float,
) -> Tuple[bool, Dict[str, Any]]:
    """Enforce stored path direction + optional entry/exit sides against track movement."""
    pd = gate.get("path_direction") if isinstance(gate.get("path_direction"), dict) else {}
    ref_dir = (_as_float(pd.get("dx")), _as_float(pd.get("dy")))
    if ref_dir == (0.0, 0.0):
        return False, {"reason": "degenerate_direction"}

    track_dir = _track_movement_direction(track)
    if track_dir is None or track_dir == (0.0, 0.0):
        return False, {"reason": "insufficient_movement"}

    dot = ref_dir[0] * track_dir[0] + ref_dir[1] * track_dir[1]
    if dot < min_dot:
        angle = _angle_between(ref_dir, track_dir)
        return False, {
            "reason": "direction_mismatch",
            "angle_rad": round(angle, 4),
            "dot": round(dot, 4),
            "gate": "path_direction",
        }

    entry_side = str(gate.get("entry_side") or "").strip()
    exit_side = str(gate.get("exit_side") or "").strip()
    if entry_side and exit_side and entry_side != exit_side and isinstance(shape, dict):
        track_entry = _bbox_side(shape, track[0][0], track[0][1])
        track_exit = _bbox_side(shape, track[-1][0], track[-1][1])
        if track_entry != entry_side or track_exit != exit_side:
            return False, {
                "reason": "direction_gate_mismatch",
                "expected_entry": entry_side,
                "expected_exit": exit_side,
                "track_entry": track_entry,
                "track_exit": track_exit,
            }

    return True, {
        "reason": "direction_gate_ok",
        "dot": round(dot, 4),
    }


def local_motion_matches_counter_rule(
    *,
    point: Dict[str, Any],
    centroid_history: Sequence[Any],
    conditions: Dict[str, Any],
    shape: Optional[Dict[str, Any]] = None,
    trigger: str = "",
) -> bool:
    """
    Return True when a desktop motion-box interaction should increment a directional counter rule.
    Skips direction checks for any_interaction rules or rules without a drawn path.
    """
    trig = _normalize_trigger(trigger)
    any_interaction = trig == "any_interaction" or bool(conditions.get("any_interaction"))
    if any_interaction:
        return True

    motion_path = conditions.get("motion_path")
    if not isinstance(motion_path, list) or len(motion_path) < 2:
        return True

    frame_path = _resolve_motion_path_for_frame(
        motion_path,
        shape,
        space=conditions.get("motion_path_space"),
        shape_ref=conditions.get("motion_path_shape_ref"),
        attach_to_shape=True,
    )
    if not frame_path:
        return False

    gate = _direction_gate_from_conditions(conditions, motion_path=frame_path, shape=shape)
    if not gate:
        return False

    cn = point if isinstance(point, dict) else {}
    centroid: Optional[Tuple[float, float]] = None
    if cn:
        centroid = (_as_float(cn.get("x")), _as_float(cn.get("y")))

    track = _augment_track_points(
        centroid_history if isinstance(centroid_history, list) else [],
        centroid,
    )
    if len(track) < 2:
        return False

    direction_only = trig != "path_match"
    min_dot = AUTO_MIN_DIRECTION_DOT if direction_only else DEFAULT_MIN_DIRECTION_DOT
    ok, _ = _match_direction_gate(track, gate=gate, shape=shape, min_dot=min_dot)
    return ok


def _centroid_in_shape_context(
    cx: float,
    cy: float,
    shape: Optional[Dict[str, Any]],
    *,
    line_pad: float = 0.06,
    tag_radius: float = 0.05,
) -> bool:
    """True when a normalized centroid lies on/near the bound shape (zone/line/tag)."""
    if not shape:
        return True
    kind = str(shape.get("kind") or "").strip().lower()
    if kind == "zone":
        poly = shape.get("pts") or shape.get("points") or []
        return len(poly) >= 3 and point_in_polygon(cx, cy, poly)
    if kind == "line":
        p1 = shape.get("p1") or {}
        p2 = shape.get("p2") or {}
        d = dist_point_to_segment(
            cx,
            cy,
            _as_float(p1.get("x")),
            _as_float(p1.get("y")),
            _as_float(p2.get("x")),
            _as_float(p2.get("y")),
        )
        return d <= line_pad
    if kind == "tag":
        anchor = shape.get("anchor") or {"x": 0.5, "y": 0.5}
        tx = _as_float(anchor.get("x"), 0.5)
        ty = _as_float(anchor.get("y"), 0.5)
        return math.hypot(cx - tx, cy - ty) <= tag_radius
    return True


def _endpoint_alignment_ok(
    track: Sequence[Tuple[float, float]],
    ref: Sequence[Tuple[float, float]],
    tolerance: float,
) -> bool:
    """Loose check that track motion spans path start→end regions (short paths only)."""
    if len(track) < 2 or len(ref) < 2:
        return True
    path_len = _path_polyline_length(ref)
    if path_len >= ENDPOINT_ALIGNMENT_MAX_PATH_LEN:
        return True
    tol = max(0.01, float(tolerance)) * 2.5
    start_dist = math.hypot(track[0][0] - ref[0][0], track[0][1] - ref[0][1])
    end_dist = math.hypot(track[-1][0] - ref[-1][0], track[-1][1] - ref[-1][1])
    return start_dist <= tol and end_dist <= tol


def _path_reference_direction(ref: Sequence[Tuple[float, float]]) -> Tuple[float, float]:
    """Overall direction of the user-drawn path (start→end)."""
    if len(ref) >= 2:
        return _segment_direction(ref[0], ref[-1])
    return 0.0, 0.0


def _augment_track_points(
    track_history: Sequence[Any],
    centroid: Optional[Tuple[float, float]],
) -> List[Tuple[float, float]]:
    """Build normalized track points; append current centroid when history has one point."""
    track = _extract_xy_points(track_history)
    if len(track) >= 2:
        return track
    if len(track) == 1 and centroid is not None:
        cx, cy = centroid
        if math.hypot(cx - track[0][0], cy - track[0][1]) > 1e-6:
            return [track[0], (cx, cy)]
    return track


def _track_movement_direction(
    track: Sequence[Tuple[float, float]],
    *,
    min_displacement: float = MIN_DIRECTION_DISPLACEMENT,
) -> Optional[Tuple[float, float]]:
    """
    Estimate track travel direction from centroid history.
    Prefers start→end, then the most recent segment, then the longest span.
    """
    if len(track) < 2:
        return None
    start, end = track[0], track[-1]
    dx = end[0] - start[0]
    dy = end[1] - start[1]
    if math.hypot(dx, dy) >= min_displacement:
        return _segment_direction(start, end)

    recent_a, recent_b = track[-2], track[-1]
    rdx = recent_b[0] - recent_a[0]
    rdy = recent_b[1] - recent_a[1]
    if math.hypot(rdx, rdy) >= min_displacement:
        return _segment_direction(recent_a, recent_b)

    best_len = 0.0
    best_dir: Optional[Tuple[float, float]] = None
    for i in range(len(track)):
        for j in range(i + 1, len(track)):
            ax, ay = track[i]
            bx, by = track[j]
            span = math.hypot(bx - ax, by - ay)
            if span >= min_displacement and span > best_len:
                best_len = span
                best_dir = _segment_direction((ax, ay), (bx, by))
    return best_dir


_MOTION_PATH_SPACE_FRAME = "frame"
_MOTION_PATH_SPACE_SHAPE = "shape"
_LINE_TAG_BOUNDS_PAD = 0.04


def _normalize_motion_path_space(value: Any) -> str:
    space = str(value or "").strip().lower()
    if space in (_MOTION_PATH_SPACE_FRAME, _MOTION_PATH_SPACE_SHAPE):
        return space
    return _MOTION_PATH_SPACE_FRAME


def _shape_bounds(shape: Dict[str, Any]) -> Tuple[float, float, float, float]:
    kind = str(shape.get("kind") or "").strip().lower()
    xs: List[float] = []
    ys: List[float] = []
    if kind == "zone":
        for p in shape.get("pts") or shape.get("points") or []:
            if isinstance(p, dict):
                xs.append(float(p.get("x", 0)))
                ys.append(float(p.get("y", 0)))
    elif kind == "line":
        for pt in (shape.get("p1"), shape.get("p2")):
            if isinstance(pt, dict):
                xs.append(float(pt.get("x", 0)))
                ys.append(float(pt.get("y", 0)))
        if xs and ys:
            pad = _LINE_TAG_BOUNDS_PAD
            if max(xs) - min(xs) < 1e-9 and max(ys) - min(ys) < 1e-9:
                ax, ay = xs[0], ys[0]
                return ax - pad, ay - pad, ax + pad, ay + pad
            return min(xs) - pad, min(ys) - pad, max(xs) + pad, max(ys) + pad
    elif kind == "tag":
        anchor = shape.get("anchor") or {"x": 0.5, "y": 0.5}
        ax = float(anchor.get("x", 0.5))
        ay = float(anchor.get("y", 0.5))
        pad = _LINE_TAG_BOUNDS_PAD
        return ax - pad, ay - pad, ax + pad, ay + pad
    if not xs or not ys:
        return 0.0, 0.0, 1.0, 1.0
    return min(xs), min(ys), max(xs), max(ys)


def _shape_relative_path_to_frame(
    path: Sequence[Any],
    shape: Dict[str, Any],
) -> List[Dict[str, float]]:
    min_x, min_y, max_x, max_y = _shape_bounds(shape)
    span_x = max(max_x - min_x, 1e-6)
    span_y = max(max_y - min_y, 1e-6)
    out: List[Dict[str, float]] = []
    for p in path or []:
        if not isinstance(p, dict):
            continue
        rx = float(p.get("x", 0))
        ry = float(p.get("y", 0))
        out.append({"x": min_x + rx * span_x, "y": min_y + ry * span_y})
    return out


def _parse_motion_path_shape_ref(value: Any) -> Optional[Dict[str, float]]:
    if not isinstance(value, dict):
        return None
    try:
        min_x = float(value.get("min_x"))
        min_y = float(value.get("min_y"))
        max_x = float(value.get("max_x"))
        max_y = float(value.get("max_y"))
    except (TypeError, ValueError):
        return None
    if max_x <= min_x or max_y <= min_y:
        return None
    return {"min_x": min_x, "min_y": min_y, "max_x": max_x, "max_y": max_y}


def _transform_frame_path_with_shape_ref(
    path: Sequence[Any],
    shape_ref: Dict[str, float],
    shape: Dict[str, Any],
) -> List[Dict[str, float]]:
    """Re-map frame path points from a saved shape bbox to the current shape bbox."""
    min_x_r = float(shape_ref["min_x"])
    min_y_r = float(shape_ref["min_y"])
    max_x_r = float(shape_ref["max_x"])
    max_y_r = float(shape_ref["max_y"])
    min_x_c, min_y_c, max_x_c, max_y_c = _shape_bounds(shape)
    span_x_r = max(max_x_r - min_x_r, 1e-6)
    span_y_r = max(max_y_r - min_y_r, 1e-6)
    span_x_c = max(max_x_c - min_x_c, 1e-6)
    span_y_c = max(max_y_c - min_y_c, 1e-6)
    out: List[Dict[str, float]] = []
    for p in path or []:
        if not isinstance(p, dict):
            continue
        fx = float(p.get("x", 0))
        fy = float(p.get("y", 0))
        rx = (fx - min_x_r) / span_x_r
        ry = (fy - min_y_r) / span_y_r
        out.append({"x": min_x_c + rx * span_x_c, "y": min_y_c + ry * span_y_c})
    return out


def _resolve_motion_path_for_frame(
    path: Sequence[Any],
    shape: Optional[Dict[str, Any]],
    *,
    space: Any = None,
    shape_ref: Any = None,
    attach_to_shape: bool = False,
) -> Optional[List[Dict[str, float]]]:
    parsed: List[Dict[str, float]] = []
    for p in path or []:
        if isinstance(p, dict):
            parsed.append({"x": float(p.get("x", 0)), "y": float(p.get("y", 0))})
    if len(parsed) < 2:
        return None
    path_space = _normalize_motion_path_space(space)
    if path_space == _MOTION_PATH_SPACE_SHAPE and isinstance(shape, dict):
        return _shape_relative_path_to_frame(parsed, shape)
    if (
        attach_to_shape
        and path_space == _MOTION_PATH_SPACE_FRAME
        and isinstance(shape, dict)
    ):
        ref = _parse_motion_path_shape_ref(shape_ref)
        if ref:
            transformed = _transform_frame_path_with_shape_ref(parsed, ref, shape)
            if len(transformed) >= 2:
                return transformed
    return parsed


def match_motion_path(
    *,
    motion_path: Sequence[Any],
    track_history: Sequence[Any],
    centroid: Optional[Tuple[float, float]] = None,
    tolerance: float = DEFAULT_PATH_MATCH_TOLERANCE,
    direction_only: bool = False,
    dwell_min: float = 0.0,
    shape: Optional[Dict[str, Any]] = None,
    dwell_sec: Optional[float] = None,
    check_speed: bool = False,
    shape_event_confirmed: bool = False,
    direction_gate: Optional[Dict[str, Any]] = None,
) -> Tuple[bool, Dict[str, Any]]:
    """
    Compare recent track centroid history to a user-drawn frame-normalized motion path.

    Tier 1 (always): general travel direction (path start→end vs track movement).
    Tier 2 (path_match only, when direction_only=False): polyline distance tolerance.
    Speed and dwell shape alignment are optional and only apply when explicitly enabled.
    """
    ref = _extract_xy_points(motion_path)
    if len(ref) < 2:
        return True, {"reason": "no_path_constraint"}

    min_dot = AUTO_MIN_DIRECTION_DOT if direction_only else DEFAULT_MIN_DIRECTION_DOT

    gate = direction_gate if isinstance(direction_gate, dict) else compute_path_direction_gate(ref, shape)

    track = _augment_track_points(track_history, centroid)
    if len(track) < 2:
        return False, {"reason": "insufficient_track_history"}

    ref_dir = _path_reference_direction(ref)
    if ref_dir == (0.0, 0.0):
        return False, {"reason": "degenerate_direction"}

    if gate:
        gate_ok, gate_details = _match_direction_gate(
            track,
            gate=gate,
            shape=shape,
            min_dot=min_dot,
        )
        if not gate_ok:
            return False, gate_details
        track_dir = _track_movement_direction(track) or ref_dir
        dot = ref_dir[0] * track_dir[0] + ref_dir[1] * track_dir[1]
        angle = _angle_between(ref_dir, track_dir)
    else:
        track_dir = _track_movement_direction(track)
        if track_dir is None or track_dir == (0.0, 0.0):
            return False, {"reason": "insufficient_movement"}
        dot = ref_dir[0] * track_dir[0] + ref_dir[1] * track_dir[1]
        if dot < min_dot:
            angle = _angle_between(ref_dir, track_dir)
            return False, {
                "reason": "direction_mismatch",
                "angle_rad": round(angle, 4),
                "dot": round(dot, 4),
            }
        angle = _angle_between(ref_dir, track_dir)

    path_inside = _points_inside_shape_ratio(ref, shape)
    track_inside = _points_inside_shape_ratio(track, shape)
    dwell_threshold = max(0.0, float(dwell_min))
    if dwell_threshold > 0 and path_inside >= PATH_DWELL_INSIDE_RATIO:
        dwell_ok = False
        if dwell_sec is not None and float(dwell_sec) >= dwell_threshold * 0.85:
            dwell_ok = True
        elif track_inside >= PATH_DWELL_INSIDE_RATIO:
            dwell_ok = True
        if not dwell_ok:
            return False, {
                "reason": "dwell_mismatch",
                "path_inside_ratio": round(path_inside, 4),
                "track_inside_ratio": round(track_inside, 4),
                "dwell_min": dwell_threshold,
            }

    path_len = _path_polyline_length(ref)
    track_disp = _track_displacement(track)
    if check_speed and path_len >= 0.05 and track_disp >= MIN_DIRECTION_DISPLACEMENT:
        path_speed = path_len / max(1, len(ref) - 1)
        track_speed = track_disp / max(1, len(track) - 1)
        if path_speed > 1e-4:
            speed_ratio = track_speed / path_speed
            if speed_ratio < PATH_SPEED_RATIO_MIN or speed_ratio > PATH_SPEED_RATIO_MAX:
                return False, {
                    "reason": "speed_mismatch",
                    "speed_ratio": round(speed_ratio, 4),
                }

    if direction_only:
        return True, {
            "reason": "direction_ok",
            "angle_rad": round(angle, 4),
            "path_inside_ratio": round(path_inside, 4),
        }

    tol = max(0.01, float(tolerance))
    avg_dist = _avg_distance_track_to_path(track, ref)
    endpoints_ok = _endpoint_alignment_ok(track, ref, tol)
    ok = avg_dist <= tol and endpoints_ok
    return (
        ok,
        {
            "reason": "path_distance_ok" if ok else "path_distance",
            "avg_dist": round(avg_dist, 4),
            "tolerance": tol,
            "angle_rad": round(angle, 4),
            "endpoints_ok": endpoints_ok,
            "path_inside_ratio": round(path_inside, 4),
            "track_inside_ratio": round(track_inside, 4),
        },
    )


def allowed_tracker_namespaces(conditions: Dict[str, Any]) -> set[str]:
    """Namespaces that may satisfy a rule's event source conditions."""
    namespaces = conditions.get("tracker_namespaces")
    if isinstance(namespaces, list) and namespaces:
        return {str(n).strip() for n in namespaces if str(n).strip()}
    expected = conditions.get("tracker_namespace")
    if expected is not None and str(expected).strip():
        exp = str(expected).strip()
        if exp == "any":
            return {MOTION_BOX_NAMESPACE, BACKEND_SORT_NAMESPACE}
        return {exp}
    require_detection = conditions.get("require_detection")
    if require_detection is False:
        return {MOTION_BOX_NAMESPACE, BACKEND_SORT_NAMESPACE}
    return {BACKEND_SORT_NAMESPACE}


def _is_dual_source_rule(conditions: Dict[str, Any]) -> bool:
    namespaces = conditions.get("tracker_namespaces")
    if isinstance(namespaces, list) and namespaces:
        ns_set = {str(n).strip() for n in namespaces if str(n).strip()}
        return MOTION_BOX_NAMESPACE in ns_set and BACKEND_SORT_NAMESPACE in ns_set
    return str(conditions.get("tracker_namespace") or "").strip() == "any"


def class_filter_applies_to_namespace(ns: str, conditions: Dict[str, Any]) -> bool:
    """Class/count filters never apply to motion_box; dual-source applies them to backend only."""
    if str(ns or "") == MOTION_BOX_NAMESPACE:
        return False
    if str(ns or "") != BACKEND_SORT_NAMESPACE:
        return False
    if _is_dual_source_rule(conditions):
        return True
    if conditions.get("require_detection") is False:
        return False
    return True


def color_filter_applies_to_namespace(ns: str, conditions: Dict[str, Any]) -> bool:
    """Color filters apply to backend tracks; skip motion_box only in dual-source rules."""
    if str(ns or "") == MOTION_BOX_NAMESPACE and _is_dual_source_rule(conditions):
        return False
    return True


def tracker_namespace_matches(ns: str, conditions: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
    """Return whether *ns* is allowed for rule *conditions* and the expected value(s)."""
    allowed = allowed_tracker_namespaces(conditions)
    ns_s = str(ns or "")
    if ns_s in allowed:
        return True, None
    expected = "|".join(sorted(allowed))
    return False, expected


def _class_matches(obj: Dict[str, Any], classes: Any, min_conf: Any) -> bool:
    allowed = classes if isinstance(classes, list) else ([classes] if classes else [])
    allowed_norm = [str(c).strip().lower() for c in allowed if str(c).strip()]
    cls = _class_name(obj)
    conf = _confidence(obj)
    minc = _as_float(min_conf, 0.0) if min_conf is not None else None
    try:
        if minc is not None and float(minc) > 1.0:
            minc = float(minc) / 100.0
    except Exception:
        pass
    if allowed_norm and cls not in allowed_norm:
        return False
    if minc is not None and conf < minc:
        return False
    return True


def matches_track_event(*, rule: Dict[str, Any], ctx: EvalContext, shape: Optional[Dict[str, Any]] = None) -> Tuple[bool, Dict[str, Any]]:
    """Evaluate a semantic track_event payload against rule trigger + conditions."""
    rule_id = str(rule.get("id") or "")
    payload = ctx.payload if isinstance(ctx.payload, dict) else {}
    event_type = _normalize_trigger(payload.get("event_type"))
    trigger = _normalize_trigger(rule.get("trigger"))

    if not bool(rule.get("enabled", True)):
        return False, {"reason": "disabled", "rule_id": rule_id}

    rule_cam = rule.get("camera_id")
    if rule_cam and str(rule_cam) != str(ctx.camera_id):
        return False, {"reason": "camera_mismatch", "rule_id": rule_id}

    conditions = rule.get("conditions") if isinstance(rule.get("conditions"), dict) else {}
    any_interaction = trigger == "any_interaction" or bool(conditions.get("any_interaction"))

    if any_interaction:
        if event_type not in ANY_INTERACTION_EVENTS:
            return False, {
                "reason": "trigger_mismatch",
                "rule_id": rule_id,
                "trigger": trigger,
                "event_type": event_type,
            }
    elif trigger == "path_match":
        allowed = ("zone_enter", "zone_exit", "line_cross", "dwell_met", "near_tag", "path_match")
        if event_type not in allowed:
            return False, {"reason": "trigger_mismatch", "rule_id": rule_id, "trigger": trigger, "event_type": event_type}
        derived = conditions.get("derived_trigger") or conditions.get("event_type")
        if derived and _normalize_trigger(derived) != event_type:
            return False, {"reason": "derived_trigger_mismatch", "rule_id": rule_id, "expected": derived, "event_type": event_type}
    elif trigger not in ("", "*", "any", "track_event", "auto_path") and trigger != event_type:
        return False, {"reason": "trigger_mismatch", "rule_id": rule_id, "trigger": trigger, "event_type": event_type}

    if not is_time_allowed(ctx.timestamp, conditions.get("time_window")):
        return False, {"reason": "time_window", "rule_id": rule_id}

    cond_event = conditions.get("event_type")
    if cond_event and _normalize_trigger(cond_event) != event_type:
        return False, {"reason": "event_type", "rule_id": rule_id}

    ns = str(payload.get("tracker_namespace") or "")
    ns_ok, ns_expected = tracker_namespace_matches(ns, conditions)
    if not ns_ok:
        return False, {
            "reason": "tracker_namespace",
            "rule_id": rule_id,
            "expected": ns_expected,
            "got": ns or None,
        }

    shape_id = rule.get("shape_id") or conditions.get("shape_id")
    if shape_id and str(payload.get("shape_id") or "") != str(shape_id):
        return False, {"reason": "shape_id", "rule_id": rule_id}

    shape_name = conditions.get("shape_name")
    if shape_name and str(payload.get("shape_name") or "").strip().lower() != str(shape_name).strip().lower():
        return False, {"reason": "shape_name", "rule_id": rule_id}

    track_obj = (ctx.tracks or ctx.detections or [{}])[0] if (ctx.tracks or ctx.detections) else payload
    if not isinstance(track_obj, dict):
        track_obj = payload

    classes = conditions.get("classes") or conditions.get("object_classes")
    min_conf = conditions.get("min_confidence")
    allowed_classes = classes if isinstance(classes, list) else ([classes] if classes else [])
    has_class_filter = any(str(c).strip() for c in allowed_classes if c is not None)
    if has_class_filter and class_filter_applies_to_namespace(ns, conditions):
        if not _class_matches(track_obj, classes, min_conf):
            return False, {"reason": "object_filter", "rule_id": rule_id}

    motion_path_cond = conditions.get("motion_path")
    has_drawn_path = isinstance(motion_path_cond, list) and len(motion_path_cond) >= 2
    path_gate = conditions.get("path_direction_gate")
    direction = conditions.get("direction")
    # Drawn-path rules use path_direction_gate for east/west discrimination (same as zones).
    # Legacy line rules may have stored positive/negative which conflicts with line_cross
    # payload direction (left_to_right/right_to_left) — ignore direction when gate is present.
    if direction and not (has_drawn_path and isinstance(path_gate, dict) and path_gate.get("path_direction")):
        allowed_dirs = direction if isinstance(direction, list) else [direction]
        cur_dir = str(payload.get("direction") or track_obj.get("direction") or "").strip().lower()
        if cur_dir and cur_dir not in [str(d).strip().lower() for d in allowed_dirs if str(d).strip()]:
            return False, {"reason": "direction", "rule_id": rule_id, "direction": cur_dir}

    dwell_val = payload.get("dwell_sec")
    if dwell_val is None:
        dwell_val = track_obj.get("dwell_sec")
    dwell_min = conditions.get("dwell_min") if conditions.get("dwell_min") is not None else conditions.get("dwell_min_sec")
    dwell_max = conditions.get("dwell_max") if conditions.get("dwell_max") is not None else conditions.get("dwell_max_sec")
    if dwell_min is not None and dwell_val is not None:
        try:
            if float(dwell_val) < float(dwell_min):
                return False, {"reason": "dwell_min", "rule_id": rule_id}
        except Exception:
            return False, {"reason": "dwell_min", "rule_id": rule_id}
    if dwell_max is not None and dwell_val is not None:
        try:
            if float(dwell_val) > float(dwell_max):
                return False, {"reason": "dwell_max", "rule_id": rule_id}
        except Exception:
            return False, {"reason": "dwell_max", "rule_id": rule_id}

    if not _color_condition_matches(payload, track_obj, conditions, tracker_namespace=ns):
        return False, {"reason": "color", "rule_id": rule_id}

    if class_filter_applies_to_namespace(ns, conditions) and not _count_condition_matches(payload, conditions):
        return False, {"reason": "count", "rule_id": rule_id}

    motion_path = conditions.get("motion_path")
    path_match_details: Dict[str, Any] = {}
    if isinstance(motion_path, list) and len(motion_path) >= 2 and not any_interaction:
        path_tol = _as_float(conditions.get("path_match_tolerance"), DEFAULT_PATH_MATCH_TOLERANCE)
        history = payload.get("centroid_history") or track_obj.get("centroid_history") or []
        centroid: Optional[Tuple[float, float]] = None
        cn = payload.get("centroid_norm")
        if isinstance(cn, dict):
            centroid = (_as_float(cn.get("x")), _as_float(cn.get("y")))
        direction_only = trigger != "path_match"
        frame_path = _resolve_motion_path_for_frame(
            motion_path,
            shape,
            space=conditions.get("motion_path_space"),
            shape_ref=conditions.get("motion_path_shape_ref"),
            attach_to_shape=True,
        )
        if not frame_path:
            return False, {"reason": "invalid_motion_path", "rule_id": rule_id}
        dwell_min_val = _as_float(conditions.get("dwell_min") or conditions.get("dwell_min_sec"), 0.0)
        check_speed = bool(conditions.get("path_speed_check"))
        direction_gate = _direction_gate_from_conditions(
            conditions,
            motion_path=frame_path,
            shape=shape,
        )
        ok_path, path_details = match_motion_path(
            motion_path=frame_path,
            track_history=history if isinstance(history, list) else [],
            centroid=centroid,
            tolerance=path_tol,
            direction_only=direction_only,
            dwell_min=dwell_min_val,
            shape=shape,
            dwell_sec=_as_float(dwell_val, 0.0) if dwell_val is not None else None,
            check_speed=check_speed,
            direction_gate=direction_gate,
        )
        if not ok_path:
            return False, {"rule_id": rule_id, **path_details}
        path_match_details = path_details

    try:
        track_id = int(payload.get("track_id"))
    except Exception:
        track_id = None

    return True, {
        "rule_id": rule_id,
        "camera_id": ctx.camera_id,
        "event_kind": ctx.kind,
        "event_type": event_type,
        "track_id": track_id,
        "shape_id": payload.get("shape_id"),
        "direction": payload.get("direction"),
        "dwell_sec": dwell_val,
        "dominant_color": payload.get("dominant_color") or payload.get("color"),
        "zone_track_count": _resolve_zone_track_count(
            payload, conditions.get("classes") or conditions.get("object_classes")
        ),
        "filtered_object_count": 1,
        "frame_w": ctx.frame_w,
        "frame_h": ctx.frame_h,
        **({"path_match": path_match_details} if path_match_details else {}),
    }


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

    if str(ctx.kind).lower() == "track_event":
        return matches_track_event(rule=rule, ctx=ctx, shape=shape)

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



