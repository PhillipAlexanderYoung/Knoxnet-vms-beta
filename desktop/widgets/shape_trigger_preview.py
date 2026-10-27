"""Pure helpers for shape trigger preview (no Qt dependencies)."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

from desktop.utils.event_rules_api import DEFAULT_RULE_COOLDOWN_SEC

COUNTER_MODES = ("off", "always", "on_trigger")
DEFAULT_COUNTER_MODE = "always"
COUNTER_COMBINE_MODES = ("none", "sum", "max", "min")
DEFAULT_COUNTER_PILL_BG = "#334155"
DEFAULT_COUNTER_PILL_TEXT = "#FFFFFF"
DEFAULT_COUNTER_PILL_HIGHLIGHT_BG = "#EF4444"
# counter_pill_anchor {x,y} is normalized within the shape bounding box (0=top/left, 1=bottom/right).
DEFAULT_COUNTER_PILL_ANCHOR: Dict[str, float] = {"x": 0.5, "y": 0.0}


def event_source_description(
    *,
    motion_enabled: bool = False,
    detection_enabled: bool = False,
    backend_status: str = "Unknown",
    detection_mode: Optional[bool] = None,
) -> str:
    """Human-readable summary of how this rule evaluates motion vs detections."""
    backend = str(backend_status or "Unknown").strip() or "Unknown"
    if detection_mode is not None:
        motion_enabled = not bool(detection_mode)
        detection_enabled = bool(detection_mode)
    if motion_enabled and detection_enabled:
        return (
            f"Dual source: lightweight motion boxes and backend YOLO/SORT tracks ({backend}). "
            "Either source can trigger; class/color filters apply to object detection events only."
        )
    if detection_enabled:
        return (
            f"Object detection uses backend YOLO/SORT tracks ({backend}). "
            "Rules match class, color, and confidence filters on tracked objects."
        )
    return (
        "Motion boxes use lightweight MOG2 motion tracks. "
        "Turn on the overlay below to preview desktop motion boxes on the live feed."
    )
LINE_TAG_BOUNDS_PAD = 0.04
# motion_path {x,y} stored relative to shape_bounds() (0=top/left, 1=bottom/right).
MOTION_PATH_SPACE_FRAME = "frame"
MOTION_PATH_SPACE_SHAPE = "shape"
PILL_SPREAD_EPS = 0.018
PILL_SPREAD_STEP = 0.12
PREVIEW_FIT_PADDING = 10.0
PREVIEW_MIN_SPAN = 0.06
PREVIEW_HIT_EXPAND = 0.12
GHOST_PATH_COLORS = ("#FFD74A", "#60A5FA", "#4ADE80", "#F472B6", "#A78BFA")
GHOST_TRIGGER_LABELS = {
    "zone_enter": "Enter",
    "zone_exit": "Exit",
    "dwell_met": "Dwell",
    "line_cross": "Cross",
    "near_tag": "Near tag",
    "path_match": "Path match",
}

_NEW_RULE_DIALOG_KEY = "__new__"


def shape_trigger_dialog_key(
    shape_id: str,
    existing_rule: Optional[Dict[str, Any]] = None,
) -> str:
    """Stable key for one Event Rule dialog per shape + rule slot."""
    sid = str(shape_id or "").strip()
    rid = str((existing_rule or {}).get("id") or "").strip()
    return f"{sid}:{rid or _NEW_RULE_DIALOG_KEY}"


@dataclass
class PreviewFit:
    """Maps frame-normalized coords into a fitted preview canvas rect."""

    min_x: float
    min_y: float
    span_x: float
    span_y: float
    x0: float
    y0: float
    w: float
    h: float

    def to_widget(self, nx: float, ny: float) -> Tuple[float, float]:
        sx = self.span_x or 1e-6
        sy = self.span_y or 1e-6
        return (
            self.x0 + (nx - self.min_x) / sx * self.w,
            self.y0 + (ny - self.min_y) / sy * self.h,
        )

    def to_norm(self, wx: float, wy: float) -> Tuple[float, float]:
        sx = self.span_x or 1e-6
        sy = self.span_y or 1e-6
        nx = self.min_x + (wx - self.x0) / max(1.0, self.w) * sx
        ny = self.min_y + (wy - self.y0) / max(1.0, self.h) * sy
        return max(0.0, min(1.0, nx)), max(0.0, min(1.0, ny))

    def hit_expanded(self, wx: float, wy: float) -> bool:
        pad_x = self.w * PREVIEW_HIT_EXPAND
        pad_y = self.h * PREVIEW_HIT_EXPAND
        return (
            self.x0 - pad_x <= wx <= self.x0 + self.w + pad_x
            and self.y0 - pad_y <= wy <= self.y0 + self.h + pad_y
        )


def normalize_counter_mode(value: Any) -> str:
    mode = str(value or "off").strip().lower()
    return mode if mode in COUNTER_MODES else "off"


def normalize_counter_combine(value: Any, *, group: str = "") -> str:
    mode = str(value or "").strip().lower()
    if mode in COUNTER_COMBINE_MODES:
        return mode
    return "sum" if str(group or "").strip() else "none"


def parse_counter_pill_color(value: Any, *, fallback: str = "") -> str:
    raw = str(value or "").strip()
    return raw if raw else str(fallback or "")


def counter_pill_label_for_rule(
    rule: Dict[str, Any],
    shape: Optional[Dict[str, Any]] = None,
    *,
    max_len: int = 12,
) -> str:
    cond = rule.get("conditions") if isinstance(rule.get("conditions"), dict) else {}
    custom = str(cond.get("counter_pill_label") or "").strip()
    if custom:
        return custom[:max_len]
    name = str(rule.get("name") or "").strip()
    if name:
        return name[:max_len]
    if isinstance(shape, dict):
        shape_label = str(shape.get("label") or shape.get("name") or "").strip()
        if shape_label:
            return shape_label[:max_len]
    return "Count"


def combine_trigger_counts(counts: List[int], mode: str) -> int:
    values = [max(0, int(c)) for c in counts]
    if not values:
        return 0
    combine = normalize_counter_combine(mode)
    if combine == "max":
        return max(values)
    if combine == "min":
        return min(values)
    return sum(values)


def default_counter_pill_anchor(shape: Dict[str, Any], slot: int = 0) -> Dict[str, float]:
    """Default pill anchor within shape_bounds so multiple rules do not stack."""
    kind = str(shape.get("kind") or "").strip().lower()
    slot = max(0, int(slot))
    if kind == "line":
        xs = [0.25, 0.5, 0.75, 0.25, 0.5, 0.75]
        ys = [0.0, 0.0, 0.0, 1.0, 1.0, 1.0]
        i = slot % len(xs)
        return {"x": xs[i], "y": ys[i]}
    if kind == "tag":
        xs = [0.5, 0.2, 0.8]
        ys = [0.0, 0.5, 1.0]
        i = slot % len(xs)
        return {"x": xs[i], "y": ys[i]}
    xs = [0.5, 0.2, 0.8, 0.5]
    ys = [0.0, 0.0, 0.0, 1.0]
    i = slot % len(xs)
    return {"x": xs[i], "y": ys[i]}


def resolve_counter_pill_anchor(
    shape: Optional[Dict[str, Any]],
    raw: Any,
    *,
    slot: int = 0,
) -> Dict[str, float]:
    """Parse stored anchor or assign a shape-specific default for *slot*."""
    if raw is None:
        return default_counter_pill_anchor(shape or {}, slot)
    return parse_counter_pill_anchor(raw)


def spread_overlapping_pill_items(
    items: List[Dict[str, Any]],
    shape: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """Offset pill anchors so items sharing the same frame point do not overlap."""
    if len(items) <= 1 or not isinstance(shape, dict):
        return items
    min_x, min_y, max_x, max_y = shape_bounds(shape)
    span_x = max(max_x - min_x, 1e-6)
    span_y = max(max_y - min_y, 1e-6)
    kind = str(shape.get("kind") or "").strip().lower()
    step_x = min(PILL_SPREAD_STEP, span_x * 0.35)
    step_y = min(PILL_SPREAD_STEP, span_y * 0.35)
    placed: List[Tuple[float, float]] = []
    for item in items:
        anchor = dict(item.get("anchor") or DEFAULT_COUNTER_PILL_ANCHOR)
        fx, fy = counter_pill_frame_coords(shape, anchor)
        guard = 0
        while any(math.hypot(fx - px, fy - py) < PILL_SPREAD_EPS for px, py in placed):
            if kind == "line":
                anchor["x"] = min(1.0, anchor.get("x", 0.5) + step_x)
                if anchor["x"] > 0.95:
                    anchor["x"] = max(0.0, anchor.get("x", 0.5) - step_x * (guard + 1))
            else:
                anchor["y"] = min(1.0, anchor.get("y", 0.0) + step_y)
            fx, fy = counter_pill_frame_coords(shape, anchor)
            guard += 1
            if guard > 12:
                break
        item["anchor"] = anchor
        placed.append((fx, fy))
    return items


def counter_pill_config_from_rule(
    rule: Dict[str, Any],
    *,
    shape: Optional[Dict[str, Any]] = None,
    slot: int = 0,
) -> Optional[Dict[str, Any]]:
    """Extract counter pill settings from an enabled event rule."""
    if not isinstance(rule, dict) or not rule.get("enabled", True):
        return None
    rule_id = str(rule.get("id") or "").strip()
    shape_id = str(rule.get("shape_id") or "").strip()
    if not rule_id or not shape_id:
        return None
    cond = rule.get("conditions") if isinstance(rule.get("conditions"), dict) else {}
    mode = normalize_counter_mode(cond.get("show_counter"))
    if mode == "off":
        return None
    group = str(cond.get("counter_group") or "").strip()
    motion_path = cond.get("motion_path") if isinstance(cond.get("motion_path"), list) else None
    path_direction_gate = (
        cond.get("path_direction_gate")
        if isinstance(cond.get("path_direction_gate"), dict)
        else None
    )
    return {
        "rule_id": rule_id,
        "shape_id": shape_id,
        "mode": mode,
        "require_detection": cond.get("require_detection", True) is not False,
        "trigger": str(rule.get("trigger") or "").strip(),
        "any_interaction": bool(cond.get("any_interaction")),
        "motion_path": motion_path,
        "motion_path_space": str(cond.get("motion_path_space") or MOTION_PATH_SPACE_FRAME),
        "path_direction_gate": path_direction_gate,
        "conditions": dict(cond),
        "anchor": resolve_counter_pill_anchor(shape, cond.get("counter_pill_anchor"), slot=slot),
        "group": group,
        "combine": normalize_counter_combine(cond.get("counter_combine"), group=group),
        "label": str(cond.get("counter_pill_label") or "").strip(),
        "bg_color": parse_counter_pill_color(cond.get("counter_pill_color")),
        "text_color": parse_counter_pill_color(cond.get("counter_pill_text_color")),
        "rule_name": str(rule.get("name") or "").strip(),
        "cooldown_sec": max(0.0, float(cond.get("cooldown_sec", cond.get("cooldown", DEFAULT_RULE_COOLDOWN_SEC)) or DEFAULT_RULE_COOLDOWN_SEC)),
    }


def counter_pill_configs_from_rules(
    rules: List[Dict[str, Any]],
    shapes: Optional[Dict[str, Dict[str, Any]]] = None,
) -> List[Dict[str, Any]]:
    shapes = shapes or {}
    shape_slots: Dict[str, int] = {}
    configs: List[Dict[str, Any]] = []
    for rule in rules or []:
        if not isinstance(rule, dict):
            continue
        sid = str(rule.get("shape_id") or "").strip()
        slot = shape_slots.get(sid, 0)
        shape_slots[sid] = slot + 1
        shape = shapes.get(sid)
        cfg = counter_pill_config_from_rule(rule, shape=shape, slot=slot)
        if cfg is not None:
            configs.append(cfg)
    return configs


def prune_trigger_counts(
    trigger_counts: Dict[str, int],
    active_rule_ids: List[str],
) -> Dict[str, int]:
    active = {str(rid) for rid in active_rule_ids if str(rid).strip()}
    return {rid: int(count) for rid, count in (trigger_counts or {}).items() if rid in active}


def resolve_counter_pill_label(cfg: Dict[str, Any], shape: Optional[Dict[str, Any]] = None) -> str:
    label = str(cfg.get("label") or "").strip()
    if label:
        return label[:12]
    rule_name = str(cfg.get("rule_name") or "").strip()
    if rule_name:
        return rule_name[:12]
    if isinstance(shape, dict):
        shape_label = str(shape.get("label") or shape.get("name") or "").strip()
        if shape_label:
            return shape_label[:12]
    group = str(cfg.get("group") or "").strip()
    if group:
        return group[:12]
    return "Count"


def build_counter_pill_render_items(
    configs: List[Dict[str, Any]],
    trigger_counts: Dict[str, int],
    *,
    shape: Optional[Dict[str, Any]] = None,
    now: float = 0.0,
    pulse_ts_by_rule: Optional[Dict[str, float]] = None,
) -> List[Dict[str, Any]]:
    """Build drawable counter pills for one shape, applying optional group aggregation."""
    pulses = pulse_ts_by_rule or {}
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    singles: List[Dict[str, Any]] = []

    for cfg in configs or []:
        group = str(cfg.get("group") or "").strip()
        combine = normalize_counter_combine(cfg.get("combine"), group=group)
        if group and combine != "none":
            grouped.setdefault(group, []).append(cfg)
        else:
            singles.append(cfg)

    items: List[Dict[str, Any]] = []

    def _count_for(cfg: Dict[str, Any]) -> int:
        rid = str(cfg.get("rule_id") or "")
        return max(0, int(trigger_counts.get(rid, 0)))

    def _visible(cfg: Dict[str, Any], count: int) -> bool:
        mode = normalize_counter_mode(cfg.get("mode"))
        if mode == "always":
            return True
        if mode == "on_trigger":
            rid = str(cfg.get("rule_id") or "")
            pulse_ts = pulses.get(rid)
            if pulse_ts and (now - float(pulse_ts)) <= 2.5:
                return True
            return count > 0
        return False

    def _highlight(rule_ids: List[str]) -> bool:
        for rid in rule_ids:
            pulse_ts = pulses.get(str(rid))
            if pulse_ts and (now - float(pulse_ts)) < 0.8:
                return True
        return False

    for cfg in singles:
        count = _count_for(cfg)
        if not _visible(cfg, count):
            continue
        rid = str(cfg.get("rule_id") or "")
        items.append(
            {
                "rule_ids": [rid],
                "count": count,
                "anchor": dict(cfg.get("anchor") or DEFAULT_COUNTER_PILL_ANCHOR),
                "label": resolve_counter_pill_label(cfg, shape),
                "bg_color": str(cfg.get("bg_color") or ""),
                "text_color": str(cfg.get("text_color") or ""),
                "highlight": _highlight([rid]),
            }
        )

    for group, members in grouped.items():
        members_sorted = sorted(members, key=lambda c: str(c.get("rule_id") or ""))
        rule_ids = [str(c.get("rule_id") or "") for c in members_sorted]
        counts = [_count_for(c) for c in members_sorted]
        combine = normalize_counter_combine(members_sorted[0].get("combine"), group=group)
        combined = combine_trigger_counts(counts, combine)
        visible = any(_visible(c, _count_for(c)) for c in members_sorted) or combined > 0
        if not visible:
            continue
        anchor_cfg = members_sorted[0]
        label = str(group)[:12] if group else resolve_counter_pill_label(anchor_cfg, shape)
        items.append(
            {
                "rule_ids": rule_ids,
                "count": combined,
                "anchor": dict(anchor_cfg.get("anchor") or DEFAULT_COUNTER_PILL_ANCHOR),
                "label": label,
                "bg_color": str(anchor_cfg.get("bg_color") or ""),
                "text_color": str(anchor_cfg.get("text_color") or ""),
                "highlight": _highlight(rule_ids),
                "group": group,
                "combine": combine,
            }
        )

    return spread_overlapping_pill_items(items, shape)


def parse_counter_pill_anchor(value: Any) -> Dict[str, float]:
    """Parse counter_pill_anchor stored relative to shape_bounds()."""
    if isinstance(value, dict):
        try:
            return {
                "x": max(0.0, min(1.0, float(value.get("x", DEFAULT_COUNTER_PILL_ANCHOR["x"])))),
                "y": max(0.0, min(1.0, float(value.get("y", DEFAULT_COUNTER_PILL_ANCHOR["y"])))),
            }
        except (TypeError, ValueError):
            pass
    return dict(DEFAULT_COUNTER_PILL_ANCHOR)


def counter_pill_frame_coords(
    shape: Dict[str, Any],
    anchor_bbox: Dict[str, float],
) -> Tuple[float, float]:
    """Map bbox-relative anchor to normalized frame coordinates."""
    min_x, min_y, max_x, max_y = shape_bounds(shape)
    ax = float(anchor_bbox.get("x", DEFAULT_COUNTER_PILL_ANCHOR["x"]))
    ay = float(anchor_bbox.get("y", DEFAULT_COUNTER_PILL_ANCHOR["y"]))
    return min_x + ax * (max_x - min_x), min_y + ay * (max_y - min_y)


def counter_pill_bbox_from_frame(
    shape: Dict[str, Any],
    frame_x: float,
    frame_y: float,
) -> Dict[str, float]:
    """Map normalized frame coordinates to bbox-relative anchor."""
    min_x, min_y, max_x, max_y = shape_bounds(shape)
    span_x = max(max_x - min_x, 1e-6)
    span_y = max(max_y - min_y, 1e-6)
    return {
        "x": max(0.0, min(1.0, (float(frame_x) - min_x) / span_x)),
        "y": max(0.0, min(1.0, (float(frame_y) - min_y) / span_y)),
    }


def normalize_motion_path_space(value: Any) -> str:
    space = str(value or "").strip().lower()
    if space in (MOTION_PATH_SPACE_FRAME, MOTION_PATH_SPACE_SHAPE):
        return space
    return MOTION_PATH_SPACE_FRAME


def motion_path_shape_ref_from_bounds(
    min_x: float,
    min_y: float,
    max_x: float,
    max_y: float,
) -> Dict[str, float]:
    """Snapshot of shape_bounds() stored alongside frame-space motion paths."""
    return {
        "min_x": float(min_x),
        "min_y": float(min_y),
        "max_x": float(max_x),
        "max_y": float(max_y),
    }


def motion_path_shape_ref_from_shape(shape: Dict[str, Any]) -> Dict[str, float]:
    min_x, min_y, max_x, max_y = shape_bounds(shape)
    return motion_path_shape_ref_from_bounds(min_x, min_y, max_x, max_y)


def parse_motion_path_shape_ref(value: Any) -> Optional[Dict[str, float]]:
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


def transform_frame_path_with_shape_ref(
    path: Sequence[Any],
    shape_ref: Dict[str, float],
    shape: Dict[str, Any],
) -> List[Dict[str, float]]:
    """Re-map frame path points from a saved shape bbox to the current shape bbox."""
    min_x_r = float(shape_ref["min_x"])
    min_y_r = float(shape_ref["min_y"])
    max_x_r = float(shape_ref["max_x"])
    max_y_r = float(shape_ref["max_y"])
    min_x_c, min_y_c, max_x_c, max_y_c = shape_bounds(shape)
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


def frame_path_to_shape_relative(
    path: Sequence[Any],
    shape: Dict[str, Any],
) -> List[Dict[str, float]]:
    """Convert frame-normalized path points to shape-bbox-relative coords (0-1)."""
    min_x, min_y, max_x, max_y = shape_bounds(shape)
    span_x = max(max_x - min_x, 1e-6)
    span_y = max(max_y - min_y, 1e-6)
    out: List[Dict[str, float]] = []
    for p in path or []:
        if not isinstance(p, dict):
            continue
        fx = float(p.get("x", 0))
        fy = float(p.get("y", 0))
        out.append(
            {
                "x": max(0.0, min(1.0, (fx - min_x) / span_x)),
                "y": max(0.0, min(1.0, (fy - min_y) / span_y)),
            }
        )
    return out


def shape_relative_path_to_frame(
    path: Sequence[Any],
    shape: Dict[str, Any],
) -> List[Dict[str, float]]:
    """Convert shape-bbox-relative path points to frame-normalized coords."""
    min_x, min_y, max_x, max_y = shape_bounds(shape)
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


def normalize_frame_motion_path(path: Sequence[Any]) -> List[Dict[str, float]]:
    """Clamp path points to the video frame (0-1), not to any shape bbox."""
    out: List[Dict[str, float]] = []
    for p in path or []:
        if not isinstance(p, dict):
            continue
        out.append(
            {
                "x": max(0.0, min(1.0, float(p.get("x", 0)))),
                "y": max(0.0, min(1.0, float(p.get("y", 0)))),
            }
        )
    return out


def resolve_motion_path_for_frame(
    path: Optional[Sequence[Any]],
    shape: Optional[Dict[str, Any]],
    *,
    space: Any = None,
    shape_ref: Any = None,
    attach_to_shape: bool = False,
) -> Optional[List[Dict[str, float]]]:
    """Return motion_path in frame-normalized coords for display or matching."""
    if not path or not isinstance(path, list) or len(path) < 2:
        return None
    parsed: List[Dict[str, float]] = []
    for p in path:
        if isinstance(p, dict):
            parsed.append({"x": float(p.get("x", 0)), "y": float(p.get("y", 0))})
    if len(parsed) < 2:
        return None
    path_space = normalize_motion_path_space(space)
    if path_space == MOTION_PATH_SPACE_SHAPE and isinstance(shape, dict):
        return shape_relative_path_to_frame(parsed, shape)
    if (
        attach_to_shape
        and path_space == MOTION_PATH_SPACE_FRAME
        and isinstance(shape, dict)
    ):
        ref = parse_motion_path_shape_ref(shape_ref)
        if ref:
            transformed = transform_frame_path_with_shape_ref(parsed, ref, shape)
            if len(transformed) >= 2:
                return transformed
    return parsed


def shape_bounds(shape: Dict[str, Any]) -> Tuple[float, float, float, float]:
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
            pad = LINE_TAG_BOUNDS_PAD
            if max(xs) - min(xs) < 1e-9 and max(ys) - min(ys) < 1e-9:
                ax, ay = xs[0], ys[0]
                return ax - pad, ay - pad, ax + pad, ay + pad
            return min(xs) - pad, min(ys) - pad, max(xs) + pad, max(ys) + pad
    elif kind == "tag":
        anchor = shape.get("anchor") or {"x": 0.5, "y": 0.5}
        ax = float(anchor.get("x", 0.5))
        ay = float(anchor.get("y", 0.5))
        pad = LINE_TAG_BOUNDS_PAD
        return ax - pad, ay - pad, ax + pad, ay + pad
    if not xs or not ys:
        return 0.0, 0.0, 1.0, 1.0
    return min(xs), min(ys), max(xs), max(ys)


def fit_shape_preview(
    shape: Dict[str, Any],
    canvas_w: float,
    canvas_h: float,
    *,
    padding: float = PREVIEW_FIT_PADDING,
    min_span: float = PREVIEW_MIN_SPAN,
) -> PreviewFit:
    min_x, min_y, max_x, max_y = shape_bounds(shape)
    span_x = max(max_x - min_x, min_span)
    span_y = max(max_y - min_y, min_span)
    cx = (min_x + max_x) / 2.0
    cy = (min_y + max_y) / 2.0
    min_x = cx - span_x / 2.0
    min_y = cy - span_y / 2.0
    max_x = cx + span_x / 2.0
    max_y = cy + span_y / 2.0
    span_x = max_x - min_x
    span_y = max_y - min_y

    avail_w = max(1.0, canvas_w - 2 * padding)
    avail_h = max(1.0, canvas_h - 2 * padding)
    shape_aspect = span_x / max(span_y, 1e-6)
    canvas_aspect = avail_w / avail_h
    if shape_aspect >= canvas_aspect:
        draw_w = avail_w
        draw_h = draw_w / shape_aspect
    else:
        draw_h = avail_h
        draw_w = draw_h * shape_aspect
    x0 = padding + (avail_w - draw_w) / 2.0
    y0 = padding + (avail_h - draw_h) / 2.0
    return PreviewFit(min_x, min_y, span_x, span_y, x0, y0, draw_w, draw_h)


def preview_animation_params(dwell_min: float, cooldown_sec: float) -> Tuple[float, float]:
    """Return phase increment per tick and dwell fraction of each animation cycle."""
    phase_inc = 0.04 / max(1.0, float(cooldown_sec) / DEFAULT_RULE_COOLDOWN_SEC)
    dwell = max(0.0, float(dwell_min))
    dwell_frac = min(0.55, dwell / max(dwell + 1.5, 1.5)) if dwell > 0 else 0.0
    return phase_inc, dwell_frac


def rules_for_shape(rules: List[Dict[str, Any]], shape_id: str) -> List[Dict[str, Any]]:
    """Return event rules bound to *shape_id*."""
    sid = str(shape_id or "").strip()
    if not sid:
        return []
    return [r for r in rules if isinstance(r, dict) and str(r.get("shape_id") or "") == sid]


def build_armed_rule_ghost_entries(
    rules: List[Dict[str, Any]],
    shapes: Sequence[Dict[str, Any]],
) -> Dict[str, List[Dict[str, Any]]]:
    """Build per-shape rule ghost overlays for armed motion-watch display."""
    out: Dict[str, List[Dict[str, Any]]] = {}
    for shape in shapes or []:
        if not isinstance(shape, dict):
            continue
        sid = str(shape.get("id") or "").strip()
        if not sid:
            continue
        entries: List[Dict[str, Any]] = []
        for idx, rule in enumerate(rules_for_shape(rules, sid)):
            if not rule.get("enabled", True):
                continue
            entry = rule_to_hover_ghost_entry(rule, color_index=idx, shape=shape)
            if entry:
                entries.append(entry)
        if entries:
            out[sid] = entries
    return out


def upsert_event_rule_in_cache(
    rules: List[Dict[str, Any]],
    saved_rule: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """Insert or replace *saved_rule* in a cached event-rules list (by rule id)."""
    rid = str(saved_rule.get("id") or "").strip()
    if not rid:
        return list(rules or [])
    kept = [
        r
        for r in (rules or [])
        if isinstance(r, dict) and str(r.get("id") or "").strip() != rid
    ]
    kept.append(dict(saved_rule))
    return kept


def rule_to_hover_ghost_entry(
    rule: Dict[str, Any],
    *,
    color_index: int = 0,
    shape: Optional[Dict[str, Any]] = None,
) -> Optional[Dict[str, Any]]:
    """Extract hover-ghost preview fields from an event rule."""
    if not isinstance(rule, dict):
        return None
    cond = rule.get("conditions") if isinstance(rule.get("conditions"), dict) else {}
    motion_path_raw = cond.get("motion_path")
    motion_path: Optional[List[Dict[str, float]]] = None
    if isinstance(motion_path_raw, list) and len(motion_path_raw) >= 2:
        parsed: List[Dict[str, float]] = []
        for p in motion_path_raw:
            if isinstance(p, dict):
                parsed.append({"x": float(p.get("x", 0)), "y": float(p.get("y", 0))})
        if len(parsed) >= 2:
            motion_path = parsed
    motion_path_space = normalize_motion_path_space(cond.get("motion_path_space"))
    motion_path_shape_ref = parse_motion_path_shape_ref(cond.get("motion_path_shape_ref"))

    classes_raw = cond.get("classes") or cond.get("object_classes") or []
    classes = [str(c) for c in classes_raw] if isinstance(classes_raw, list) else []

    trigger = str(rule.get("trigger") or cond.get("derived_trigger") or "")
    actions = rule.get("actions") if isinstance(rule.get("actions"), list) else []
    has_snapshot = any(
        isinstance(a, dict) and str(a.get("type") or "").strip().lower() == "snapshot"
        for a in actions
    )
    return {
        "name": str(rule.get("name") or ""),
        "trigger": trigger,
        "direction": str(cond.get("direction") or ""),
        "motion_path": motion_path,
        "motion_path_space": motion_path_space,
        "motion_path_shape_ref": motion_path_shape_ref,
        "color_bucket": str(cond.get("color") or cond.get("dominant_color") or ""),
        "classes": classes,
        "require_detection": cond.get("require_detection", True) is not False,
        "dwell_min": float(cond.get("dwell_min") or cond.get("dwell_min_sec") or 0.0),
        "cooldown_sec": float(cond.get("cooldown_sec", DEFAULT_RULE_COOLDOWN_SEC) or DEFAULT_RULE_COOLDOWN_SEC),
        "ghost_color_index": int(color_index),
        "has_snapshot": has_snapshot,
        "path_match_tolerance": float(cond.get("path_match_tolerance") or 0.0),
    }


def ghost_entry_hover_lines(entry: Dict[str, Any]) -> List[str]:
    """Multi-line hover tooltip for a configured event rule ghost."""
    lines: List[str] = []
    name = str(entry.get("name") or "").strip()
    if name:
        lines.append(name[:28] + ("…" if len(name) > 28 else ""))

    trigger = str(entry.get("trigger") or "").strip()
    if trigger:
        trig_label = GHOST_TRIGGER_LABELS.get(trigger, trigger.replace("_", " ").title())
        lines.append(f"Trigger: {trig_label}")

    direction = str(entry.get("direction") or "").strip()
    if direction:
        lines.append(f"Direction: {direction.replace('_', ' ')}")

    motion_path = entry.get("motion_path")
    if isinstance(motion_path, list) and len(motion_path) >= 2:
        lines.append(f"Path: {len(motion_path)} pts")
    elif trigger == "path_match":
        lines.append("Path: not set")

    tolerance = float(entry.get("path_match_tolerance") or 0.0)
    if tolerance > 0 and (
        trigger == "path_match"
        or (isinstance(motion_path, list) and len(motion_path) >= 2)
    ):
        lines.append(f"Tolerance: {tolerance:.2f}")

    classes = entry.get("classes") or []
    if isinstance(classes, list) and classes:
        cls = ", ".join(str(c) for c in classes[:3])
        if len(classes) > 3:
            cls += "…"
        lines.append(f"Class: {cls}")

    color = str(entry.get("color_bucket") or "").strip()
    if color:
        lines.append(f"Color: {color}")

    dwell = float(entry.get("dwell_min") or 0.0)
    if dwell > 0:
        lines.append(f"Dwell: {dwell:.1f}s")

    cooldown = float(entry.get("cooldown_sec") or 0.0)
    if cooldown > 0:
        lines.append(f"Cooldown: {cooldown:.0f}s")

    if entry.get("has_snapshot") is False:
        lines.append("No snapshot")
    elif entry.get("require_detection") is False:
        lines.append("Motion only")

    return lines or ["Rule"]


def ghost_entry_label(entry: Dict[str, Any]) -> str:
    """Compact label for a hover-ghost chip (rule name or trigger summary)."""
    name = str(entry.get("name") or "").strip()
    if name:
        label = name[:24]
        if entry.get("has_snapshot") is False:
            label = f"{label} · no snap"
        return label
    parts: List[str] = []
    trigger = str(entry.get("trigger") or "").strip()
    if trigger:
        parts.append(GHOST_TRIGGER_LABELS.get(trigger, trigger.replace("_", " ").title()))
    direction = str(entry.get("direction") or "").strip()
    if direction:
        parts.append(direction.replace("_", " "))
    classes = entry.get("classes") or []
    if isinstance(classes, list) and classes:
        parts.append(str(classes[0]))
    color = str(entry.get("color_bucket") or "").strip()
    if color:
        parts.append(color)
    label = " · ".join(parts) if parts else "Rule"
    if entry.get("has_snapshot") is False:
        label = f"{label} · no snap"
    return label


def motion_path_travel_t(phase: float, dwell_fraction: float) -> float:
    """Map animation phase (0..1) to path travel fraction (0..1), pausing at end during dwell."""
    dwell_fraction = max(0.0, min(0.85, float(dwell_fraction)))
    travel_portion = max(0.05, 1.0 - dwell_fraction)
    p = float(phase) % 1.0
    if p >= travel_portion:
        return 1.0
    return p / travel_portion
