"""Graphical shape-centric Event Rule / trigger configuration."""

from __future__ import annotations

import math
import uuid
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from PySide6.QtCore import QPoint, QPointF, QRectF, Qt, QTimer, QUrl
from PySide6.QtGui import QColor, QDesktopServices, QGuiApplication, QPainter, QPen, QPolygonF
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from core.automation.actions.script import DEFAULT_SCRIPT_TIMEOUT_SEC, normalize_runner
from desktop.utils.shape_trigger_helpers import (
    DEFAULT_SCRIPT_RUNNER,
    DEFAULT_TRIGGER_MODE,
    EXPLICIT_SHAPE_TRIGGERS,
    SCRIPT_RUNNER_OPTIONS,
    build_event_source_conditions,
    build_rule_actions,
    effective_trigger_from_mode,
    ensure_event_rules_scripts_dir,
    parse_event_source_flags,
    script_action_from_rule,
    trigger_mode_options_for_kind,
)

from core.automation.conditions import (
    DEFAULT_PATH_MATCH_TOLERANCE,
    compute_path_direction_gate,
    point_in_polygon,
)
from desktop.widgets.shape_trigger_preview import (
    DEFAULT_COUNTER_PILL_BG,
    DEFAULT_COUNTER_PILL_HIGHLIGHT_BG,
    DEFAULT_COUNTER_PILL_TEXT,
    GHOST_PATH_COLORS,
    MOTION_PATH_SPACE_FRAME,
    PreviewFit,
    build_counter_pill_render_items,
    counter_pill_bbox_from_frame,
    counter_pill_config_from_rule,
    counter_pill_frame_coords,
    default_counter_pill_anchor,
    DEFAULT_COUNTER_MODE,
    event_source_description,
    ghost_entry_label,
    motion_path_travel_t,
    normalize_counter_combine,
    normalize_counter_mode,
    normalize_frame_motion_path,
    parse_counter_pill_anchor,
    parse_counter_pill_color,
    preview_animation_params,
    resolve_counter_pill_anchor,
    resolve_counter_pill_label,
    resolve_motion_path_for_frame,
    rules_for_shape,
    motion_path_shape_ref_from_shape,
)
from desktop.utils.event_rules_api import (
    DEFAULT_API_BASE,
    DEFAULT_RULE_COOLDOWN_MS,
    DEFAULT_RULE_COOLDOWN_SEC,
    backend_detection_status_label,
    cooldown_ms_from_sec,
    cooldown_sec_from_ms,
    delete_rule,
    ensure_backend_detection_for_rules,
    get_backend_detection_enabled,
    list_rules,
    load_motion_watch_settings_from_disk,
    save_rule,
    set_rules_enabled,
    snapshot_action_from_motion_watch_settings,
    _api_get,
)
from desktop.widgets.event_rules_editor import COLOR_BUCKETS

ZONE_TRIGGERS = [
    ("Enter zone", "zone_enter"),
    ("Exit zone", "zone_exit"),
    ("Dwell met", "dwell_met"),
]

LINE_TRIGGERS = [
    ("Cross line", "line_cross"),
]

TAG_TRIGGERS = [
    ("Near tag", "near_tag"),
]

PATH_TRIGGER_MODES = [
    ("Auto (from path)", "auto_path"),
    ("Any interaction", "any_interaction"),
    ("Path match (directional)", "path_match"),
]

DEFAULT_PATH_TOLERANCE = DEFAULT_PATH_MATCH_TOLERANCE
MIN_PATH_POINT_DIST = 0.012

COLOR_BUCKET_QCOLORS: Dict[str, QColor] = {
    "": QColor("#FFD74A"),
    "white": QColor("#F8FAFC"),
    "black": QColor("#1E293B"),
    "gray": QColor("#94A3B8"),
    "red": QColor("#EF4444"),
    "green": QColor("#22C55E"),
    "blue": QColor("#3B82F6"),
    "yellow": QColor("#EAB308"),
    "brown": QColor("#92400E"),
}

COUNTER_PILL_COLOR_PRESETS: List[Tuple[str, str]] = [
    ("Default", ""),
    ("Red", "#EF4444"),
    ("Slate", "#334155"),
    ("Blue", "#3B82F6"),
    ("Green", "#22C55E"),
    ("Amber", "#F59E0B"),
    ("Purple", "#A855F7"),
]

COUNTER_PILL_TEXT_PRESETS: List[Tuple[str, str]] = [
    ("Default", ""),
    ("White", "#FFFFFF"),
    ("Black", "#0F172A"),
    ("Light gray", "#E2E8F0"),
]


def position_shape_trigger_dialog_beside(dialog: QDialog, camera_widget) -> None:
    """Place the Event Rule panel beside the camera so the video stays clickable."""
    try:
        anchor = camera_widget.window() if camera_widget is not None else dialog.parentWidget()
        if anchor is None:
            return
        screen = QGuiApplication.screenAt(anchor.mapToGlobal(QPoint(anchor.width() // 2, anchor.height() // 2)))
        if screen is None:
            screen = QGuiApplication.primaryScreen()
        if screen is None:
            return
        ag = screen.availableGeometry()
        top_left = anchor.mapToGlobal(QPoint(0, 0))
        margin = 8
        dlg_w = max(dialog.width(), dialog.sizeHint().width(), dialog.minimumWidth())
        dlg_h = max(dialog.height(), dialog.sizeHint().height())

        x = top_left.x() + anchor.width() + margin
        y = top_left.y() + margin
        if x + dlg_w > ag.right():
            x = top_left.x() - dlg_w - margin
        if x < ag.left():
            x = ag.left() + margin
        if y + dlg_h > ag.bottom():
            y = max(ag.top() + margin, ag.bottom() - dlg_h - margin)

        dialog.move(int(x), int(y))
    except Exception:
        pass


def color_bucket_to_qcolor(bucket: str, *, fallback: str = "#FFD74A") -> QColor:
    key = str(bucket or "").strip().lower()
    col = COLOR_BUCKET_QCOLORS.get(key)
    if col is not None:
        return QColor(col)
    try:
        parsed = QColor(str(bucket or fallback))
        return parsed if parsed.isValid() else QColor(fallback)
    except Exception:
        return QColor(fallback)


def counter_pill_qcolor(value: str, *, fallback: str) -> QColor:
    parsed = color_bucket_to_qcolor(value, fallback=fallback)
    return parsed if parsed.isValid() else QColor(fallback)


def _fit_to_qpoint(fit: PreviewFit, nx: float, ny: float) -> QPointF:
    wx, wy = fit.to_widget(nx, ny)
    return QPointF(wx, wy)


DIRECTION_LABELS = {
    "": "Any direction",
    "left_to_right": "Left → Right",
    "right_to_left": "Right → Left",
    "positive": "Positive side",
    "negative": "Negative side",
    "both": "Both directions",
}

TRIGGER_CHIP_LABELS = {
    "zone_enter": "Enter",
    "zone_exit": "Exit",
    "dwell_met": "Dwell",
    "line_cross": "Cross",
    "near_tag": "Near tag",
    "path_match": "Path match",
    "any_interaction": "Any interaction",
}

PreviewCallback = Callable[[Optional[Dict[str, Any]]], None]
ShapeUpdateCallback = Callable[[Dict[str, Any]], None]
PathDrawControlCallback = Callable[[str, List[Dict[str, float]]], None]
PillMoveControlCallback = Callable[[str], None]


def _triggers_for_kind(kind: str) -> List[tuple[str, str]]:
    k = str(kind or "").strip().lower()
    if k == "line":
        return list(LINE_TRIGGERS)
    if k == "tag":
        return list(TAG_TRIGGERS)
    return list(ZONE_TRIGGERS)


def _default_trigger(kind: str) -> str:
    opts = _triggers_for_kind(kind)
    return opts[0][1] if opts else "zone_enter"


def _shape_qcolor(shape: Dict[str, Any], fallback: str = "#24D1FF") -> QColor:
    try:
        c = QColor(str(shape.get("color") or fallback))
        return c if c.isValid() else QColor(fallback)
    except Exception:
        return QColor(fallback)


def _norm_pt(nx: float, ny: float, rect: QRectF) -> QPointF:
    return QPointF(rect.x() + nx * rect.width(), rect.y() + ny * rect.height())


def _line_geometry(p1: Dict[str, Any], p2: Dict[str, Any]) -> Tuple[float, float, float, float, float, float]:
    x1, y1 = float(p1.get("x", 0)), float(p1.get("y", 0))
    x2, y2 = float(p2.get("x", 1)), float(p2.get("y", 1))
    dx, dy = x2 - x1, y2 - y1
    length = math.hypot(dx, dy) or 1e-6
    ux, uy = dx / length, dy / length
    return x1, y1, x2, y2, ux, uy


def derive_line_direction_from_path(shape: Dict[str, Any], path: List[Dict[str, float]]) -> str:
    p1, p2 = shape.get("p1"), shape.get("p2")
    if not p1 or not p2 or len(path) < 2:
        return ""
    x1, y1, x2, y2, ux, uy = _line_geometry(p1, p2)
    perp_x, perp_y = -uy, ux
    mvx = float(path[-1]["x"]) - float(path[0]["x"])
    mvy = float(path[-1]["y"]) - float(path[0]["y"])
    dot_line = mvx * ux + mvy * uy
    dot_perp = mvx * perp_x + mvy * perp_y
    if abs(dot_line) >= abs(dot_perp):
        return "left_to_right" if dot_line > 0 else "right_to_left"
    return "positive" if dot_perp > 0 else "negative"


def derive_trigger_from_path(
    shape: Dict[str, Any],
    path: List[Dict[str, float]],
) -> Tuple[str, str, Optional[float]]:
    """Infer trigger semantics from a user-drawn normalized motion path."""
    kind = str(shape.get("kind") or "").strip().lower()
    if len(path) < 2:
        return _default_trigger(kind), "", None

    sx, sy = float(path[0]["x"]), float(path[0]["y"])
    ex, ey = float(path[-1]["x"]), float(path[-1]["y"])

    if kind == "zone":
        pts = shape.get("pts") or shape.get("points") or []
        start_in = point_in_polygon(sx, sy, pts) if len(pts) >= 3 else False
        end_in = point_in_polygon(ex, ey, pts) if len(pts) >= 3 else False
        inside_count = sum(
            1 for p in path if point_in_polygon(float(p["x"]), float(p["y"]), pts)
        ) if len(pts) >= 3 else 0
        inside_ratio = inside_count / max(1, len(path))

        if not start_in and end_in:
            return "zone_enter", "", None
        if start_in and not end_in:
            return "zone_exit", "", None
        if start_in and end_in and inside_ratio > 0.65 and len(path) >= 4:
            return "dwell_met", "", 1.0
        if not start_in and not end_in:
            return "zone_enter", "", None
        return "zone_enter", "", None

    if kind == "line":
        return "line_cross", derive_line_direction_from_path(shape, path), None

    if kind == "tag":
        return "near_tag", "", None

    return _default_trigger(kind), "", None


def _draw_shape_base(
    painter: QPainter,
    shape: Dict[str, Any],
    rect: QRectF,
    *,
    preview_fit: Optional[PreviewFit] = None,
) -> None:
    kind = str(shape.get("kind") or "").lower()
    col = _shape_qcolor(shape)
    alpha = float(shape.get("alpha", 0.65))
    col.setAlphaF(min(1.0, alpha))
    pen = QPen(col, 2)
    painter.setPen(pen)

    def _pt(nx: float, ny: float) -> QPointF:
        if preview_fit is not None:
            return _fit_to_qpoint(preview_fit, nx, ny)
        return _norm_pt(nx, ny, rect)

    if kind == "zone":
        pts = shape.get("pts") or shape.get("points") or []
        if len(pts) < 3:
            return
        poly = QPolygonF([_pt(float(p.get("x", 0)), float(p.get("y", 0))) for p in pts])
        fill = QColor(col)
        fill.setAlpha(int(alpha * 90))
        painter.setBrush(fill)
        painter.drawPolygon(poly)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawPolygon(poly)

    elif kind == "line":
        p1, p2 = shape.get("p1"), shape.get("p2")
        if not p1 or not p2:
            return
        pen.setStyle(Qt.PenStyle.DashLine)
        painter.setPen(pen)
        pt1 = _pt(float(p1.get("x", 0)), float(p1.get("y", 0)))
        pt2 = _pt(float(p2.get("x", 1)), float(p2.get("y", 1)))
        painter.drawLine(pt1, pt2)

    elif kind == "tag":
        anchor = shape.get("anchor") or {"x": 0.5, "y": 0.5}
        cx = float(anchor.get("x", 0.5))
        cy = float(anchor.get("y", 0.5))
        cpt = _pt(cx, cy)
        sz = 12.0
        painter.setPen(pen)
        painter.drawLine(QPointF(cpt.x() - sz, cpt.y()), QPointF(cpt.x() + sz, cpt.y()))
        painter.drawLine(QPointF(cpt.x(), cpt.y() - sz), QPointF(cpt.x(), cpt.y() + sz))


def draw_motion_path(
    painter: QPainter,
    motion_path: List[Dict[str, float]],
    *,
    phase: float,
    draw_rect: Tuple[float, float, float, float],
    preview_fit: Optional[PreviewFit] = None,
    color_bucket: str = "",
    classes: Optional[List[str]] = None,
    require_detection: bool = True,
    dwell_min: float = 0.0,
    cooldown_sec: float = DEFAULT_RULE_COOLDOWN_SEC,
) -> None:
    """Draw user-recorded motion path with animated dot."""
    if len(motion_path) < 2:
        return

    x0, y0, rw, rh = draw_rect
    rect = QRectF(x0, y0, rw, rh)

    def _pt(nx: float, ny: float) -> QPointF:
        if preview_fit is not None:
            return _fit_to_qpoint(preview_fit, nx, ny)
        return _norm_pt(nx, ny, rect)

    pts = [_pt(float(p["x"]), float(p["y"])) for p in motion_path]

    _, dwell_frac = preview_animation_params(dwell_min, cooldown_sec)
    travel_t = motion_path_travel_t(phase, dwell_frac)

    accent = color_bucket_to_qcolor(color_bucket)
    accent.setAlpha(220)
    dash_pen = QPen(accent, 2, Qt.PenStyle.CustomDashLine)
    dash_pen.setDashPattern([5, 4])
    dash_pen.setDashOffset(-phase * 20)
    painter.setPen(dash_pen)
    painter.setBrush(Qt.BrushStyle.NoBrush)
    for i in range(len(pts) - 1):
        painter.drawLine(pts[i], pts[i + 1])

    start_col = QColor("#4ADE80")
    end_col = QColor("#FF6B6B")
    painter.setPen(QPen(start_col, 4))
    painter.drawEllipse(pts[0], 4, 4)
    painter.setPen(QPen(end_col, 4))
    painter.drawEllipse(pts[-1], 4, 4)

    idx_f = travel_t * (len(pts) - 1)
    idx = min(len(pts) - 2, int(idx_f))
    frac = idx_f - idx
    mx = pts[idx].x() + (pts[idx + 1].x() - pts[idx].x()) * frac
    my = pts[idx].y() + (pts[idx + 1].y() - pts[idx].y()) * frac

    dot = QColor(accent)
    dot.setAlpha(240)
    painter.setPen(Qt.PenStyle.NoPen)
    painter.setBrush(dot)
    radius = 5 if require_detection else 7
    painter.drawEllipse(QPointF(mx, my), radius, radius)

    if require_detection:
        label = (classes or ["object"])[0] if classes else "object"
        chip = str(label)[:10]
        chip_font = painter.font()
        chip_font.setPointSize(8)
        painter.setFont(chip_font)
        metrics = painter.fontMetrics()
        tw = metrics.horizontalAdvance(chip) + 8
        th = metrics.height() + 4
        chip_rect = QRectF(mx + 8, my - th - 2, tw, th)
        chip_bg = QColor("#0F172A")
        chip_bg.setAlpha(210)
        painter.setBrush(chip_bg)
        painter.setPen(QPen(accent, 1))
        painter.drawRoundedRect(chip_rect, 4, 4)
        painter.setPen(QColor("#E2E8F0"))
        painter.drawText(chip_rect.adjusted(4, 2, -4, -2), Qt.AlignmentFlag.AlignCenter, chip)
    else:
        blob = QColor(accent)
        blob.setAlpha(120)
        painter.setBrush(blob)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawEllipse(QPointF(mx, my), radius + 4, radius + 4)


def draw_rule_ghost_overlay(
    painter: QPainter,
    entry: Dict[str, Any],
    *,
    phase: float,
    draw_rect: Tuple[float, float, float, float],
    shape: Optional[Dict[str, Any]] = None,
    preview_fit: Optional[PreviewFit] = None,
) -> None:
    """Semi-transparent hover hint for a configured event rule (no counter pill)."""
    x0, y0, rw, rh = draw_rect
    rect = QRectF(x0, y0, rw, rh)

    def _pt(nx: float, ny: float) -> QPointF:
        if preview_fit is not None:
            return _fit_to_qpoint(preview_fit, nx, ny)
        return _norm_pt(nx, ny, rect)

    idx = int(entry.get("ghost_color_index") or 0)
    accent = color_bucket_to_qcolor(
        str(entry.get("color_bucket") or ""),
        fallback=GHOST_PATH_COLORS[idx % len(GHOST_PATH_COLORS)],
    )
    accent.setAlpha(102)

    label = ghost_entry_label(entry)
    motion_path = resolve_motion_path_for_frame(
        entry.get("motion_path"),
        shape,
        space=entry.get("motion_path_space"),
        shape_ref=entry.get("motion_path_shape_ref"),
        attach_to_shape=True,
    )
    dwell_min = float(entry.get("dwell_min") or 0.0)
    cooldown_sec = float(entry.get("cooldown_sec") or DEFAULT_RULE_COOLDOWN_SEC)
    _, dwell_frac = preview_animation_params(dwell_min, cooldown_sec)
    travel_t = motion_path_travel_t(phase, dwell_frac)

    mx: Optional[float] = None
    my: Optional[float] = None
    label_y_offset = float(idx) * 16.0

    if isinstance(motion_path, list) and len(motion_path) >= 2:
        pts = [_pt(float(p["x"]), float(p["y"])) for p in motion_path if isinstance(p, dict)]
        if len(pts) >= 2:
            dash_pen = QPen(accent, 2, Qt.PenStyle.CustomDashLine)
            dash_pen.setDashPattern([6, 5])
            dash_pen.setDashOffset(-phase * 12)
            painter.setPen(dash_pen)
            painter.setBrush(Qt.BrushStyle.NoBrush)
            for i in range(len(pts) - 1):
                painter.drawLine(pts[i], pts[i + 1])

            idx_f = travel_t * (len(pts) - 1)
            seg = min(len(pts) - 2, int(idx_f))
            frac = idx_f - seg
            mx = pts[seg].x() + (pts[seg + 1].x() - pts[seg].x()) * frac
            my = pts[seg].y() + (pts[seg + 1].y() - pts[seg].y()) * frac

            dot = QColor(accent)
            dot.setAlpha(80)
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(dot)
            painter.drawEllipse(QPointF(mx, my), 4, 4)

    if mx is None and shape is not None:
        anchor = _shape_anchor_widget(shape, preview_fit=preview_fit, rect=rect)
        mx, my = anchor.x(), anchor.y()

    if mx is not None and label:
        chip_font = painter.font()
        chip_font.setPointSize(8)
        painter.setFont(chip_font)
        metrics = painter.fontMetrics()
        tw = min(metrics.horizontalAdvance(label) + 10, 180)
        th = metrics.height() + 4
        chip_rect = QRectF(mx + 8, my - th - 4 - label_y_offset, tw, th)
        chip_bg = QColor(accent)
        chip_bg.setAlpha(70)
        painter.setBrush(chip_bg)
        painter.setPen(QPen(accent, 1))
        painter.drawRoundedRect(chip_rect, 4, 4)
        text_col = QColor("#E2E8F0")
        text_col.setAlpha(180)
        painter.setPen(text_col)
        painter.drawText(
            chip_rect.adjusted(5, 2, -5, -2),
            Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft,
            label,
        )


def draw_trigger_animation(
    painter: QPainter,
    shape: Dict[str, Any],
    *,
    trigger: str,
    direction: str,
    phase: float,
    draw_rect: Tuple[float, float, float, float],
    motion_path: Optional[List[Dict[str, float]]] = None,
    preview_fit: Optional[PreviewFit] = None,
    color_bucket: str = "",
    classes: Optional[List[str]] = None,
    require_detection: bool = True,
    dwell_min: float = 0.0,
    cooldown_sec: float = DEFAULT_RULE_COOLDOWN_SEC,
) -> None:
    """Draw animated trigger overlay only when a user path exists."""
    if not motion_path or len(motion_path) < 2:
        return
    draw_motion_path(
        painter,
        motion_path,
        phase=phase,
        draw_rect=draw_rect,
        preview_fit=preview_fit,
        color_bucket=color_bucket,
        classes=classes,
        require_detection=require_detection,
        dwell_min=dwell_min,
        cooldown_sec=cooldown_sec,
    )


def draw_trigger_preview(
    painter: QPainter,
    shape: Dict[str, Any],
    *,
    trigger: str,
    direction: str,
    phase: float,
    draw_rect: Tuple[float, float, float, float],
    motion_path: Optional[List[Dict[str, float]]] = None,
    preview_fit: Optional[PreviewFit] = None,
    color_bucket: str = "",
    classes: Optional[List[str]] = None,
    require_detection: bool = True,
    dwell_min: float = 0.0,
    cooldown_sec: float = DEFAULT_RULE_COOLDOWN_SEC,
    show_counter: str = "off",
    counter_value: int = 0,
    counter_pill_anchor: Optional[Dict[str, float]] = None,
    counter_pill_label: str = "",
    counter_pill_color: str = "",
    counter_pill_text_color: str = "",
) -> None:
    """Draw shape plus animated trigger overlay on a preview canvas."""
    x0, y0, rw, rh = draw_rect
    rect = QRectF(x0, y0, rw, rh)
    _draw_shape_base(painter, shape, rect, preview_fit=preview_fit)
    draw_trigger_animation(
        painter,
        shape,
        trigger=trigger,
        direction=direction,
        phase=phase,
        draw_rect=draw_rect,
        motion_path=motion_path,
        preview_fit=preview_fit,
        color_bucket=color_bucket,
        classes=classes,
        require_detection=require_detection,
        dwell_min=dwell_min,
        cooldown_sec=cooldown_sec,
    )
    if normalize_counter_mode(show_counter) != "off" and counter_value >= 0:
        _draw_counter_pill_preview(
            painter,
            shape,
            counter_value,
            preview_fit=preview_fit,
            rect=rect,
            counter_pill_anchor=counter_pill_anchor,
            highlight=normalize_counter_mode(show_counter) == "on_trigger",
            label=counter_pill_label,
            bg_color=counter_pill_color,
            text_color=counter_pill_text_color,
        )


def _shape_anchor_widget(
    shape: Dict[str, Any],
    *,
    preview_fit: Optional[PreviewFit],
    rect: QRectF,
) -> QPointF:
    kind = str(shape.get("kind") or "").lower()
    if kind == "zone":
        pts = shape.get("pts") or shape.get("points") or []
        if len(pts) >= 3:
            cx = sum(float(p.get("x", 0)) for p in pts) / len(pts)
            cy = sum(float(p.get("y", 0)) for p in pts) / len(pts)
            if preview_fit is not None:
                return _fit_to_qpoint(preview_fit, cx, cy)
            return _norm_pt(cx, cy, rect)
    elif kind == "line":
        p1, p2 = shape.get("p1"), shape.get("p2")
        if p1 and p2:
            cx = (float(p1.get("x", 0)) + float(p2.get("x", 1))) / 2.0
            cy = (float(p1.get("y", 0)) + float(p2.get("y", 1))) / 2.0
            if preview_fit is not None:
                return _fit_to_qpoint(preview_fit, cx, cy)
            return _norm_pt(cx, cy, rect)
    anchor = shape.get("anchor") or {"x": 0.5, "y": 0.5}
    if preview_fit is not None:
        return _fit_to_qpoint(preview_fit, float(anchor.get("x", 0.5)), float(anchor.get("y", 0.5)))
    return _norm_pt(float(anchor.get("x", 0.5)), float(anchor.get("y", 0.5)), rect)


def _counter_pill_widget_pos(
    shape: Dict[str, Any],
    *,
    preview_fit: Optional[PreviewFit],
    rect: QRectF,
    counter_pill_anchor: Optional[Dict[str, float]] = None,
) -> QPointF:
    """Widget coordinates for the counter pill anchor (pill sits slightly above point)."""
    fx, fy = counter_pill_frame_coords(shape, parse_counter_pill_anchor(counter_pill_anchor))
    if preview_fit is not None:
        wx, wy = preview_fit.to_widget(fx, fy)
    else:
        pt = _norm_pt(fx, fy, rect)
        wx, wy = pt.x(), pt.y()
    return QPointF(wx, wy - 14.0 * 0.85)


def _draw_counter_pill_preview(
    painter: QPainter,
    shape: Dict[str, Any],
    count: int,
    *,
    preview_fit: Optional[PreviewFit],
    rect: QRectF,
    counter_pill_anchor: Optional[Dict[str, float]] = None,
    highlight: bool = False,
    label: str = "",
    bg_color: str = "",
    text_color: str = "",
) -> None:
    pos = _counter_pill_widget_pos(
        shape,
        preview_fit=preview_fit,
        rect=rect,
        counter_pill_anchor=counter_pill_anchor,
    )
    _draw_counter_pill(
        painter,
        pos.x(),
        pos.y(),
        count,
        highlight=highlight,
        scale=0.85,
        label=label,
        bg_color=bg_color,
        text_color=text_color,
    )


def _draw_counter_pill(
    painter: QPainter,
    x: float,
    y: float,
    count: int,
    *,
    highlight: bool = False,
    scale: float = 1.0,
    label: str = "",
    bg_color: str = "",
    text_color: str = "",
) -> None:
    text = str(max(0, int(count)))
    font = painter.font()
    font.setPointSize(max(7, int(9 * scale)))
    font.setBold(True)
    painter.setFont(font)
    metrics = painter.fontMetrics()
    tw = metrics.horizontalAdvance(text) + int(10 * scale)
    th = metrics.height() + int(4 * scale)
    pill = QRectF(x - tw / 2.0, y - th, tw, th)
    if bg_color:
        bg = counter_pill_qcolor(bg_color, fallback=DEFAULT_COUNTER_PILL_BG)
    else:
        bg = QColor(DEFAULT_COUNTER_PILL_HIGHLIGHT_BG if highlight else DEFAULT_COUNTER_PILL_BG)
    bg.setAlpha(230 if highlight else 200)
    if text_color:
        fg = counter_pill_qcolor(text_color, fallback=DEFAULT_COUNTER_PILL_TEXT)
        border = QColor(fg)
        border.setAlpha(180)
    else:
        fg = QColor(DEFAULT_COUNTER_PILL_TEXT)
        border = QColor("#F8FAFC" if highlight else "#CBD5E1")
    painter.setBrush(bg)
    painter.setPen(QPen(border, 1))
    painter.drawRoundedRect(pill, th / 2.0, th / 2.0)
    painter.setPen(fg)
    painter.drawText(pill, Qt.AlignmentFlag.AlignCenter, text)

    label_text = str(label or "").strip()
    if label_text:
        label_font = painter.font()
        label_font.setPointSize(max(6, int(7 * scale)))
        label_font.setBold(False)
        painter.setFont(label_font)
        label_metrics = painter.fontMetrics()
        label_col = counter_pill_qcolor(text_color, fallback="#CBD5E1") if text_color else QColor("#94A3B8")
        painter.setPen(label_col)
        lx = pill.right() + 4
        ly = pill.center().y() + label_metrics.ascent() / 2.0 - 1
        painter.drawText(QPointF(lx, ly), label_text[:12])


class ShapeTriggerDialog(QDialog):
    """Compact single-shape trigger editor with mouse-drawn motion path."""

    def __init__(
        self,
        *,
        camera_id: str,
        shape: Dict[str, Any],
        api_base: str = DEFAULT_API_BASE,
        existing_rule: Optional[Dict[str, Any]] = None,
        on_preview_changed: Optional[PreviewCallback] = None,
        on_shape_updated: Optional[ShapeUpdateCallback] = None,
        on_path_draw_control: Optional[PathDrawControlCallback] = None,
        on_pill_move_control: Optional[PillMoveControlCallback] = None,
        parent=None,
    ):
        super().__init__(parent)
        self.camera_id = str(camera_id)
        self.shape = dict(shape)
        self.api_base = api_base.rstrip("/")
        self.saved_rule: Optional[Dict[str, Any]] = None
        self._rule_id = str(existing_rule.get("id")) if existing_rule else None
        self._arm_on_accept = False
        self._on_preview_changed = on_preview_changed
        self._on_shape_updated = on_shape_updated
        self._on_path_draw_control = on_path_draw_control
        self._on_pill_move_control = on_pill_move_control
        self._auto_rule_name = not bool(existing_rule)
        self._syncing_name = False
        self._derived_trigger = _default_trigger(str(shape.get("kind") or ""))
        self._derived_direction = ""
        self._path_tolerance = DEFAULT_PATH_TOLERANCE
        self._motion_path: List[Dict[str, float]] = []
        self._preview_phase = 0.0
        self._path_draw_active = False
        self._pill_move_active = False
        shape_rules = rules_for_shape(
            list_rules(self.api_base, self.camera_id),
            str(self.shape.get("id") or ""),
        )
        self._counter_pill_anchor = default_counter_pill_anchor(self.shape, len(shape_rules))
        self._rule_enabled = True

        kind = str(self.shape.get("kind") or "shape").strip().lower()
        label = str(self.shape.get("label") or self.shape.get("id") or "Shape")
        sid = str(self.shape.get("id") or "")

        self.setWindowTitle(f"Event Rule — {label}")
        self.setMinimumWidth(420)
        self.setMaximumWidth(520)
        self.setModal(False)
        self.setWindowModality(Qt.WindowModality.NonModal)
        self.setWindowFlags(
            Qt.WindowType.Tool
            | Qt.WindowType.WindowStaysOnTopHint
            | Qt.WindowType.WindowCloseButtonHint
        )
        self._camera_widget = parent

        root = QVBoxLayout(self)
        root.setSpacing(6)
        root.setContentsMargins(8, 8, 8, 8)

        top_row = QHBoxLayout()
        top_row.setSpacing(8)

        path_col = QVBoxLayout()
        path_col.setSpacing(4)
        self.path_summary_label = QLabel("No path drawn")
        self.path_summary_label.setStyleSheet("color: #94a3b8; font-size: 11px;")
        self.path_summary_label.setWordWrap(True)
        path_col.addWidget(self.path_summary_label)
        path_btns = QHBoxLayout()
        path_btns.setSpacing(4)
        self.draw_path_btn = QPushButton("Draw path on camera")
        self.draw_path_btn.setFixedHeight(28)
        self.draw_path_btn.clicked.connect(self._toggle_camera_path_draw)
        path_btns.addWidget(self.draw_path_btn)
        self.clear_path_btn = QPushButton("Clear path")
        self.clear_path_btn.setFixedHeight(28)
        self.clear_path_btn.clicked.connect(self._clear_path)
        path_btns.addWidget(self.clear_path_btn)
        path_col.addLayout(path_btns)
        top_row.addLayout(path_col)

        controls_col = QVBoxLayout()
        controls_col.setSpacing(4)

        meta = QLabel(f"<b>{label}</b> <span style='color:#64748b'>{kind}</span>")
        meta.setTextFormat(Qt.TextFormat.RichText)
        controls_col.addWidget(meta)

        self.shape_name_edit = QLineEdit(label)
        self.shape_name_edit.setPlaceholderText("Shape name")
        self.shape_name_edit.textChanged.connect(self._on_shape_name_changed)
        controls_col.addWidget(self.shape_name_edit)

        trig_row = QHBoxLayout()
        trig_row.addWidget(QLabel("Trigger"))
        self.trigger_combo = QComboBox()
        for tlabel, tval in trigger_mode_options_for_kind(kind):
            self.trigger_combo.addItem(tlabel, tval)
        self.trigger_combo.currentIndexChanged.connect(self._on_trigger_mode_changed)
        trig_row.addWidget(self.trigger_combo, 1)
        if not existing_rule:
            auto_idx = self.trigger_combo.findData(DEFAULT_TRIGGER_MODE)
            if auto_idx >= 0:
                self.trigger_combo.setCurrentIndex(auto_idx)
        controls_col.addLayout(trig_row)

        self.inferred_label = QLabel("")
        self.inferred_label.setStyleSheet("color: #94a3b8; font-size: 10px;")
        self.inferred_label.setWordWrap(True)
        controls_col.addWidget(self.inferred_label)

        rule_row = QHBoxLayout()
        self.rule_combo = QComboBox()
        self.rule_combo.addItem("(New rule)", "")
        self._reload_rule_combo(select_id=self._rule_id)
        self.rule_combo.currentIndexChanged.connect(self._on_rule_selected)
        rule_row.addWidget(self.rule_combo, 1)
        self.name_edit = QLineEdit(str((existing_rule or {}).get("name") or f"{label} trigger"))
        self.name_edit.setPlaceholderText("Rule name")
        self.name_edit.textEdited.connect(lambda _t: setattr(self, "_auto_rule_name", False))
        rule_row.addWidget(self.name_edit, 1)
        controls_col.addLayout(rule_row)

        rule_manage_row = QHBoxLayout()
        self.enabled_check = QCheckBox("Rule enabled")
        self.enabled_check.setChecked(True)
        self.enabled_check.toggled.connect(self._emit_preview)
        rule_manage_row.addWidget(self.enabled_check)
        self.delete_rule_btn = QPushButton("Delete rule")
        self.delete_rule_btn.setFixedHeight(24)
        self.delete_rule_btn.clicked.connect(self._delete_rule)
        rule_manage_row.addWidget(self.delete_rule_btn)
        rule_manage_row.addStretch()
        controls_col.addLayout(rule_manage_row)

        top_row.addLayout(controls_col, 1)
        root.addLayout(top_row)

        event_source_box = QGroupBox("Event source")
        event_source_layout = QVBoxLayout(event_source_box)
        event_source_layout.setSpacing(6)
        event_source_layout.setContentsMargins(8, 8, 8, 8)

        self.event_source_label = QLabel("")
        self.event_source_label.setStyleSheet("color: #64748b; font-size: 10px;")
        self.event_source_label.setWordWrap(True)
        event_source_layout.addWidget(self.event_source_label)

        mode_row = QHBoxLayout()
        mode_row.setSpacing(12)
        self.motion_source_check = QCheckBox("Motion boxes (lightweight)")
        self.detection_source_check = QCheckBox("Object detection / tracking (classes/colors)")
        self.motion_source_check.toggled.connect(self._on_event_source_changed)
        self.detection_source_check.toggled.connect(self._on_event_source_changed)
        mode_row.addWidget(self.motion_source_check)
        mode_row.addWidget(self.detection_source_check)
        mode_row.addStretch()
        event_source_layout.addLayout(mode_row)

        self.show_motion_overlay_check = QCheckBox("Show motion boxes overlay")
        self.show_motion_overlay_check.toggled.connect(self._on_show_motion_overlay_toggled)
        event_source_layout.addWidget(self.show_motion_overlay_check)

        self.motion_box_settings_btn = QPushButton("Motion box settings…")
        self.motion_box_settings_btn.setFixedHeight(28)
        self.motion_box_settings_btn.clicked.connect(self._open_motion_box_settings)
        event_source_layout.addWidget(self.motion_box_settings_btn)

        self.backend_detection_widget = QWidget()
        backend_row = QHBoxLayout(self.backend_detection_widget)
        backend_row.setContentsMargins(0, 0, 0, 0)
        backend_row.setSpacing(8)
        self.backend_detection_check = QCheckBox("Backend detection / object tracking")
        self.backend_detection_check.toggled.connect(self._on_backend_detection_toggled)
        backend_row.addWidget(self.backend_detection_check)
        self.backend_detection_status = QLabel("")
        self.backend_detection_status.setStyleSheet("color: #94a3b8; font-size: 10px;")
        backend_row.addWidget(self.backend_detection_status)
        backend_row.addStretch()
        event_source_layout.addWidget(self.backend_detection_widget)
        self._backend_detection_syncing = False
        self._syncing_motion_overlay = False

        root.addWidget(event_source_box)

        camera_widget = getattr(self, "_camera_widget", None)
        if camera_widget is not None:
            visible = (
                camera_widget.get_motion_boxes_visible()
                if hasattr(camera_widget, "get_motion_boxes_visible")
                else bool(getattr(camera_widget, "motion_boxes_enabled", False))
            )
            self._syncing_motion_overlay = True
            try:
                self.show_motion_overlay_check.setChecked(visible)
            finally:
                self._syncing_motion_overlay = False

        self._build_collapsible_sections(root, kind)

        if kind == "tag":
            hint = QLabel("Tag rules use desktop motion capture.")
            hint.setStyleSheet("color: #64748b; font-size: 10px;")
            root.addWidget(hint)

        if existing_rule:
            self._apply_rule(existing_rule)
        else:
            self.motion_source_check.setChecked(True)
            self._set_counter_mode(DEFAULT_COUNTER_MODE)

        self.status_label = QLabel("")
        self.status_label.setStyleSheet("color: #94a3b8; font-size: 10px;")
        root.addWidget(self.status_label)

        buttons = QDialogButtonBox()
        self.save_btn = buttons.addButton("Save", QDialogButtonBox.ButtonRole.AcceptRole)
        self.arm_btn = buttons.addButton("Save && Arm", QDialogButtonBox.ButtonRole.ActionRole)
        buttons.addButton(QDialogButtonBox.StandardButton.Cancel)
        self.save_btn.clicked.connect(lambda: self._save(arm=False))
        self.arm_btn.clicked.connect(lambda: self._save(arm=True))
        buttons.rejected.connect(self.reject)
        root.addWidget(buttons)

        self._on_event_source_changed()
        self._on_trigger_mode_changed()
        self._apply_tooltips(kind)
        self._update_rule_action_state()
        self._on_screenshot_toggled(self.screenshot_check.isChecked())
        self._on_run_script_toggled(self.run_script_check.isChecked())
        self._refresh_backend_detection_status()
        self._emit_preview()

    def _set_counter_mode(self, mode: str) -> None:
        idx = self.counter_combo.findData(normalize_counter_mode(mode))
        if idx >= 0:
            self.counter_combo.setCurrentIndex(idx)

    def _build_collapsible_sections(self, root: QVBoxLayout, kind: str) -> None:
        filters_box, filters_inner = self._collapsible_group("Detection filters")
        filters_form = QFormLayout(filters_inner)
        filters_form.setContentsMargins(4, 4, 4, 4)
        filters_form.setSpacing(4)
        self.classes_edit = QLineEdit()
        self.classes_edit.setPlaceholderText("car, person")
        filters_form.addRow("Classes", self.classes_edit)
        self.confidence_spin = QDoubleSpinBox()
        self.confidence_spin.setRange(0.0, 1.0)
        self.confidence_spin.setSingleStep(0.05)
        self.confidence_spin.setValue(0.5)
        filters_form.addRow("Min conf.", self.confidence_spin)
        self.color_combo = QComboBox()
        for c in COLOR_BUCKETS:
            self.color_combo.addItem(c or "(any)", c)
        filters_form.addRow("Color", self.color_combo)
        self.color_combo.currentIndexChanged.connect(self._emit_preview)
        self.classes_edit.textChanged.connect(self._emit_preview)
        self.confidence_spin.valueChanged.connect(self._emit_preview)
        root.addWidget(filters_box)

        timing_box, timing_inner = self._collapsible_group("Timing")
        timing_form = QFormLayout(timing_inner)
        timing_form.setContentsMargins(4, 4, 4, 4)
        self.dwell_min_spin = QDoubleSpinBox()
        self.dwell_min_spin.setRange(0.0, 3600.0)
        self.dwell_min_spin.setSingleStep(0.5)
        self.dwell_min_spin.setSpecialValueText("(none)")
        self.dwell_min_spin.setEnabled(kind == "zone")
        timing_form.addRow("Dwell (s)", self.dwell_min_spin)
        self.dwell_min_spin.valueChanged.connect(self._emit_preview)
        self.cooldown_spin = QSpinBox()
        self.cooldown_spin.setRange(0, 600_000)
        self.cooldown_spin.setSingleStep(50)
        self.cooldown_spin.setSuffix(" ms")
        self.cooldown_spin.setValue(DEFAULT_RULE_COOLDOWN_MS)
        timing_form.addRow("Cooldown", self.cooldown_spin)
        self.cooldown_spin.valueChanged.connect(self._emit_preview)
        self.path_tol_spin = QDoubleSpinBox()
        self.path_tol_spin.setRange(0.02, 0.5)
        self.path_tol_spin.setSingleStep(0.01)
        self.path_tol_spin.setValue(DEFAULT_PATH_TOLERANCE)
        self.path_tol_spin.setToolTip("Normalized distance tolerance for path matching")
        timing_form.addRow("Path tol.", self.path_tol_spin)
        root.addWidget(timing_box)

        counter_box, counter_inner = self._collapsible_group("Counter display")
        counter_form = QFormLayout(counter_inner)
        counter_form.setContentsMargins(4, 4, 4, 4)
        counter_form.setSpacing(4)
        self.counter_combo = QComboBox()
        for label, mode in (
            ("Off", "off"),
            ("Always", "always"),
            ("On trigger", "on_trigger"),
        ):
            self.counter_combo.addItem(label, mode)
        counter_form.addRow("Counter pill", self.counter_combo)
        self.counter_combo.currentIndexChanged.connect(self._emit_preview)
        self.counter_label_edit = QLineEdit()
        self.counter_label_edit.setPlaceholderText("e.g. Cars")
        self.counter_label_edit.setMaxLength(12)
        self.counter_label_edit.textChanged.connect(self._emit_preview)
        counter_form.addRow("Pill label", self.counter_label_edit)
        self.counter_group_edit = QLineEdit()
        self.counter_group_edit.setPlaceholderText("Optional group id")
        self.counter_group_edit.textChanged.connect(self._emit_preview)
        counter_form.addRow("Counter group", self.counter_group_edit)
        self.counter_combine_combo = QComboBox()
        for label, mode in (
            ("None (individual)", "none"),
            ("Sum", "sum"),
            ("Max", "max"),
            ("Min", "min"),
        ):
            self.counter_combine_combo.addItem(label, mode)
        self.counter_combine_combo.currentIndexChanged.connect(self._emit_preview)
        counter_form.addRow("Group combine", self.counter_combine_combo)
        self.counter_pill_color_combo = QComboBox()
        for label, color in COUNTER_PILL_COLOR_PRESETS:
            self.counter_pill_color_combo.addItem(label, color)
        self.counter_pill_color_combo.currentIndexChanged.connect(self._emit_preview)
        counter_form.addRow("Pill color", self.counter_pill_color_combo)
        self.counter_pill_text_combo = QComboBox()
        for label, color in COUNTER_PILL_TEXT_PRESETS:
            self.counter_pill_text_combo.addItem(label, color)
        self.counter_pill_text_combo.currentIndexChanged.connect(self._emit_preview)
        counter_form.addRow("Pill text", self.counter_pill_text_combo)
        pill_move_row = QHBoxLayout()
        pill_move_row.setSpacing(4)
        self.move_pill_btn = QPushButton("Move counter pill")
        self.move_pill_btn.setFixedHeight(28)
        self.move_pill_btn.clicked.connect(self._toggle_camera_pill_move)
        pill_move_row.addWidget(self.move_pill_btn)
        pill_move_row.addStretch()
        counter_form.addRow("", pill_move_row)
        root.addWidget(counter_box)

        adv_box, adv_inner = self._collapsible_group("More options")
        adv_layout = QVBoxLayout(adv_inner)
        adv_layout.setContentsMargins(4, 4, 4, 4)
        adv_layout.setSpacing(6)

        capture_row = QHBoxLayout()
        self.screenshot_check = QCheckBox("Take screenshot on trigger")
        self.screenshot_check.setChecked(True)
        self.screenshot_check.toggled.connect(self._on_screenshot_toggled)
        capture_row.addWidget(self.screenshot_check)
        self.capture_settings_btn = QPushButton("Capture settings…")
        self.capture_settings_btn.setFixedHeight(24)
        self.capture_settings_btn.clicked.connect(self._open_capture_settings)
        capture_row.addWidget(self.capture_settings_btn)
        capture_row.addStretch()
        adv_layout.addLayout(capture_row)
        self.screenshot_warn_label = QLabel("Screenshot disabled — rule will not save captures.")
        self.screenshot_warn_label.setStyleSheet("color: #F59E0B; font-size: 11px;")
        self.screenshot_warn_label.setVisible(False)
        adv_layout.addWidget(self.screenshot_warn_label)

        self.run_script_check = QCheckBox("Run script on trigger")
        self.run_script_check.toggled.connect(self._on_run_script_toggled)
        adv_layout.addWidget(self.run_script_check)

        script_form = QFormLayout()
        script_form.setContentsMargins(0, 0, 0, 0)
        script_form.setSpacing(4)
        script_path_row = QHBoxLayout()
        self.script_path_edit = QLineEdit()
        self.script_path_edit.setPlaceholderText("script.py (relative to scripts/event_rules)")
        script_path_row.addWidget(self.script_path_edit, 1)
        self.script_browse_btn = QPushButton("Browse…")
        self.script_browse_btn.setFixedHeight(24)
        self.script_browse_btn.clicked.connect(self._browse_script_path)
        script_path_row.addWidget(self.script_browse_btn)
        script_path_widget = QWidget()
        script_path_widget.setLayout(script_path_row)
        script_form.addRow("Script", script_path_widget)

        self.script_runner_combo = QComboBox()
        for label, runner in SCRIPT_RUNNER_OPTIONS:
            self.script_runner_combo.addItem(label, runner)
        script_form.addRow("Runner", self.script_runner_combo)

        self.script_args_edit = QLineEdit()
        self.script_args_edit.setPlaceholderText("Optional args")
        script_form.addRow("Args", self.script_args_edit)

        self.script_timeout_spin = QSpinBox()
        self.script_timeout_spin.setRange(1, 600)
        self.script_timeout_spin.setSuffix(" s")
        self.script_timeout_spin.setValue(int(DEFAULT_SCRIPT_TIMEOUT_SEC))
        script_form.addRow("Timeout", self.script_timeout_spin)

        self.script_folder_btn = QPushButton("Open scripts folder")
        self.script_folder_btn.setFixedHeight(24)
        self.script_folder_btn.clicked.connect(self._open_scripts_folder)
        script_form.addRow("", self.script_folder_btn)

        self._script_fields: List[QWidget] = [
            self.script_path_edit,
            self.script_browse_btn,
            self.script_runner_combo,
            self.script_args_edit,
            self.script_timeout_spin,
            self.script_folder_btn,
        ]
        adv_layout.addLayout(script_form)

        adv_form = QFormLayout()
        adv_form.setContentsMargins(0, 0, 0, 0)
        self.count_min_spin = QDoubleSpinBox()
        self.count_min_spin.setRange(0.0, 999.0)
        self.count_min_spin.setDecimals(0)
        self.count_min_spin.setSpecialValueText("(none)")
        adv_form.addRow("Occupancy min", self.count_min_spin)
        self.count_max_spin = QDoubleSpinBox()
        self.count_max_spin.setRange(0.0, 999.0)
        self.count_max_spin.setDecimals(0)
        self.count_max_spin.setSpecialValueText("(none)")
        adv_form.addRow("Occupancy max", self.count_max_spin)
        self.per_track_check = QCheckBox("Per-track cooldown")
        self.per_track_check.setChecked(True)
        adv_form.addRow("", self.per_track_check)
        adv_layout.addLayout(adv_form)
        root.addWidget(adv_box)

    def _apply_tooltips(self, kind: str) -> None:
        self.draw_path_btn.setToolTip(
            "Draw the expected motion path directly on the live camera image. "
            "The path spans the full scene (not clipped to the shape). "
            "Drag on the video to sketch the route; drag the counter pill on-camera to reposition it."
        )
        self.path_summary_label.setToolTip(
            "Read-only summary of the drawn path: point count and inferred trigger semantics."
        )
        self.clear_path_btn.setToolTip(
            "Remove the drawn motion path so you can sketch a new one. "
            "Clears inferred trigger chips until you draw again."
        )
        self.shape_name_edit.setToolTip(
            "Label shown on the camera overlay and in rule lists. "
            "Renaming updates the live shape label immediately."
        )
        self.trigger_combo.setToolTip(
            "Choose how this rule fires. Auto infers enter/exit/dwell/cross from your drawn path. "
            "Any interaction counts any zone/line/tag touch without path filtering. "
            "Path match fires only when motion closely follows the recorded path. "
            "Explicit choices (Enter shape, Cross line, etc.) pin the backend event type."
        )
        self.inferred_label.setToolTip(
            "Summary of trigger semantics inferred from the motion path or path-match mode."
        )
        self.rule_combo.setToolTip(
            "Load an existing event rule for this shape or start a new one."
        )
        self.name_edit.setToolTip(
            "Saved event rule name in the rules list. "
            "Does not change the shape overlay label."
        )
        self.motion_source_check.setToolTip(
            "Evaluate rules using lightweight motion blobs (MOG2 / frame-diff). "
            "No AI object class required; lower CPU than object detection."
        )
        self.detection_source_check.setToolTip(
            "Evaluate rules using backend object tracking with class, color, and confidence filters."
        )
        self.show_motion_overlay_check.setToolTip(
            "Draw desktop motion boxes on the live camera feed. "
            "Recommended in Motion mode; also useful as a visual aid in Detection mode."
        )
        self.backend_detection_check.setToolTip(
            "Enables backend YOLO object tracking on this camera. Required for Detection-mode rules "
            "(class/color filters). Motion-mode rules use backend MOG2 motion boxes instead."
        )
        self.backend_detection_status.setToolTip(
            "Current backend detection state for this camera (On, Off, or Unknown)."
        )
        self.classes_edit.setToolTip(
            "Comma-separated COCO-style classes (e.g. person, car). "
            "Only objects matching these classes can satisfy the rule."
        )
        self.confidence_spin.setToolTip(
            "Minimum detector confidence (0–1) for a match. "
            "Higher values reduce false positives but may miss distant objects."
        )
        self.color_combo.setToolTip(
            "Optional dominant-color filter on detections. "
            "Preview path accent uses the selected color bucket."
        )
        dwell_tip = (
            "Seconds inside a zone before a dwell trigger can fire. "
            "Preview animation pauses longer at the path end when dwell is set."
        )
        if kind != "zone":
            dwell_tip += " Only applies to zone shapes."
        self.dwell_min_spin.setToolTip(dwell_tip)
        self.cooldown_spin.setToolTip(
            "Minimum time between rule firings for the same shape (milliseconds). "
            "Examples: 250 ms, 500 ms, 1000 ms (1 s). "
            "Higher cooldown slows the preview loop to reflect re-trigger spacing."
        )
        self.motion_box_settings_btn.setToolTip(
            "Open motion box detection tuning (sensitivity and merge/size) without leaving this dialog. "
            "Sensitivity controls how much pixel change counts as motion. "
            "Merge size joins nearby motion blobs before boxing."
        )
        self.path_tol_spin.setToolTip(
            "Normalized distance tolerance for path-match triggers. "
            "Lower values require motion to follow the drawn path more closely."
        )
        self.counter_combo.setToolTip(
            "Show a cumulative trigger-count pill on the shape overlay. "
            "Drag the pill on the camera image to set its position. "
            "Always keeps the pill visible; On trigger highlights briefly after each fire."
        )
        self.counter_label_edit.setToolTip(
            "Short label drawn beside the counter pill (defaults to rule or shape name)."
        )
        self.counter_group_edit.setToolTip(
            "Optional group id. Pills sharing a group can display a combined count."
        )
        self.counter_combine_combo.setToolTip(
            "How to aggregate trigger counts for rules in the same counter group."
        )
        self.counter_pill_color_combo.setToolTip("Background color for the counter pill.")
        self.counter_pill_text_combo.setToolTip("Text color for the counter pill and label.")
        self.move_pill_btn.setToolTip(
            "Drag the counter pill on the live camera overlay to set its position."
        )
        self.delete_rule_btn.setToolTip(
            "Delete the selected event rule from the server and clear the editor."
        )
        self.enabled_check.setToolTip(
            "When disabled, the rule is saved but does not fire until re-enabled."
        )
        self.count_min_spin.setToolTip(
            "Optional minimum occupancy / object count threshold before the rule can fire."
        )
        self.count_max_spin.setToolTip(
            "Optional maximum occupancy threshold; counts above this block the rule."
        )
        self.per_track_check.setToolTip(
            "Apply cooldown separately per tracked object instead of once per shape."
        )
        self.screenshot_check.setToolTip(
            "Save a JPEG when this rule fires or increments a counter."
        )
        self.capture_settings_btn.setToolTip(
            "Configure save folder, overlays, resize, clips, and retention for rule screenshots."
        )
        self.run_script_check.setToolTip(
            "Run a script from scripts/event_rules when this rule fires. "
            "Context is passed as JSON via KNOXNET_EVENT_JSON and stdin."
        )
        self.script_path_edit.setToolTip(
            "Script path relative to scripts/event_rules (recommended) or an allowed scripts directory."
        )
        self.script_browse_btn.setToolTip("Pick a script file from the allowed scripts folder.")
        self.script_runner_combo.setToolTip(
            "Python runs with the server interpreter. Shell requires .sh files and "
            "KNOXNET_EVENT_RULES_ALLOW_SHELL=1. Executable must be marked executable."
        )
        self.script_args_edit.setToolTip("Optional arguments appended to the script command.")
        self.script_timeout_spin.setToolTip("Maximum seconds before the script is terminated.")
        self.script_folder_btn.setToolTip(
            "Open the Event Rules scripts folder in your file manager."
        )
        self.save_btn.setToolTip(
            "Save the rule with the drawn path and current settings. "
            "Does not enable global event-rule arming."
        )
        self.arm_btn.setToolTip(
            "Save the rule and arm event rules for this camera so triggers are evaluated live."
        )

    def _collapsible_group(self, title: str) -> Tuple[QGroupBox, QWidget]:
        box = QGroupBox(title)
        box.setCheckable(True)
        box.setChecked(False)
        box.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)
        inner = QWidget()
        inner.setVisible(False)
        layout = QVBoxLayout(box)
        layout.setContentsMargins(6, 4, 6, 4)
        layout.addWidget(inner)
        box.toggled.connect(inner.setVisible)
        return box, inner

    def showEvent(self, event) -> None:
        super().showEvent(event)
        anchor = getattr(self, "_camera_widget", None)
        if anchor is not None:
            QTimer.singleShot(0, lambda: position_shape_trigger_dialog_beside(self, anchor))
        self._sync_motion_overlay_from_camera()
        self._emit_preview()

    def _sync_motion_overlay_from_camera(self) -> None:
        camera_widget = getattr(self, "_camera_widget", None)
        if camera_widget is None:
            return
        visible = (
            camera_widget.get_motion_boxes_visible()
            if hasattr(camera_widget, "get_motion_boxes_visible")
            else bool(getattr(camera_widget, "motion_boxes_enabled", False))
        )
        if self.show_motion_overlay_check.isChecked() == visible:
            return
        self._syncing_motion_overlay = True
        try:
            self.show_motion_overlay_check.setChecked(visible)
        finally:
            self._syncing_motion_overlay = False

    def closeEvent(self, event) -> None:
        self._stop_camera_path_draw()
        self._stop_camera_pill_move()
        if self._on_preview_changed:
            self._on_preview_changed(None)
        super().closeEvent(event)

    def reject(self) -> None:
        self._stop_camera_path_draw()
        self._stop_camera_pill_move()
        if self._on_preview_changed:
            self._on_preview_changed(None)
        super().reject()

    def set_motion_path_from_camera(self, path: List[Dict[str, float]]) -> None:
        """Called by CameraWidget when user draws or edits path on live video."""
        self._motion_path = normalize_frame_motion_path(path or [])
        self._on_motion_path_updated()

    def set_pill_anchor_from_camera(self, anchor: Dict[str, float]) -> None:
        self._counter_pill_anchor = parse_counter_pill_anchor(anchor)
        self._emit_preview()

    def motion_path_norm(self) -> List[Dict[str, float]]:
        return [{"x": p["x"], "y": p["y"]} for p in self._motion_path]

    def _on_motion_path_updated(self) -> None:
        path = self.motion_path_norm()
        trig, direction, dwell = derive_trigger_from_path(self.shape, path)
        self._derived_trigger = trig
        self._derived_direction = direction
        if dwell and self.dwell_min_spin.value() <= 0:
            self.dwell_min_spin.setValue(float(dwell))
        self._update_inferred_chips()
        self._emit_preview()

    def _toggle_camera_path_draw(self) -> None:
        if self._path_draw_active:
            self._stop_camera_path_draw()
            return
        self._stop_camera_pill_move()
        if self._on_path_draw_control:
            self._on_path_draw_control("start", self.motion_path_norm())
        self._path_draw_active = True
        self.draw_path_btn.setText("Stop drawing")
        self.status_label.setText("Draw the motion path on the live camera image.")

    def _stop_camera_path_draw(self) -> None:
        if not self._path_draw_active:
            return
        if self._on_path_draw_control:
            self._on_path_draw_control("stop", self.motion_path_norm())
        self._path_draw_active = False
        self.draw_path_btn.setText("Draw path on camera")

    def _toggle_camera_pill_move(self) -> None:
        if self._pill_move_active:
            self._stop_camera_pill_move()
            return
        self._stop_camera_path_draw()
        if self._on_pill_move_control:
            self._on_pill_move_control("start")
        self._pill_move_active = True
        self.move_pill_btn.setText("Stop moving pill")
        self.status_label.setText("Drag the counter pill on the live camera image.")

    def _stop_camera_pill_move(self) -> None:
        if not self._pill_move_active:
            return
        if self._on_pill_move_control:
            self._on_pill_move_control("stop")
        self._pill_move_active = False
        self.move_pill_btn.setText("Move counter pill")

    def _update_path_summary(self) -> None:
        path = self.motion_path_norm()
        if len(path) < 2:
            self.path_summary_label.setText("No path drawn — use Draw path on camera")
            return
        trig_label = TRIGGER_CHIP_LABELS.get(self._derived_trigger, self._derived_trigger.replace("_", " "))
        dir_label = DIRECTION_LABELS.get(self._derived_direction, "") if self._derived_direction else ""
        parts = [f"{len(path)} points", trig_label]
        if dir_label:
            parts.append(dir_label)
        self.path_summary_label.setText(" · ".join(parts))

    def _rule_slot_for_id(self, rule_id: Optional[str]) -> int:
        sid = str(self.shape.get("id") or "")
        rules = sorted(
            rules_for_shape(list_rules(self.api_base, self.camera_id), sid),
            key=lambda r: str(r.get("id") or ""),
        )
        for idx, rule in enumerate(rules):
            if str(rule.get("id") or "") == str(rule_id or ""):
                return idx
        return len(rules)

    def _sibling_pill_preview_items(self) -> List[Dict[str, Any]]:
        sid = str(self.shape.get("id") or "")
        shape_rules = sorted(
            rules_for_shape(list_rules(self.api_base, self.camera_id), sid),
            key=lambda r: str(r.get("id") or ""),
        )
        configs: List[Dict[str, Any]] = []
        for slot, rule in enumerate(shape_rules):
            rid = str(rule.get("id") or "")
            if rid == str(self._rule_id or ""):
                continue
            cfg = counter_pill_config_from_rule(rule, shape=self.shape, slot=slot)
            if cfg is not None:
                configs.append(cfg)
        if not configs:
            return []
        trigger_counts = {str(c.get("rule_id") or ""): 1 for c in configs}
        return build_counter_pill_render_items(
            configs,
            trigger_counts,
            shape=self.shape,
            now=0.0,
        )

    def _update_rule_action_state(self) -> None:
        has_rule = bool(self._rule_id)
        self.delete_rule_btn.setEnabled(has_rule)

    def _refresh_backend_detection_status(self) -> None:
        enabled = get_backend_detection_enabled(self.api_base, self.camera_id)
        self._backend_detection_syncing = True
        try:
            self.backend_detection_status.setText(backend_detection_status_label(enabled))
            self.backend_detection_check.setTristate(False)
            self.backend_detection_check.setChecked(enabled is True)
        finally:
            self._backend_detection_syncing = False
        self._update_event_source_label()

    def _on_backend_detection_toggled(self, checked: bool) -> None:
        if self._backend_detection_syncing:
            return
        if not checked:
            confirm = QMessageBox.question(
                self,
                "Backend detection",
                "Server-side path matching and counters require backend object tracking. "
                "Turn off backend detection for this camera anyway?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if confirm != QMessageBox.StandardButton.Yes:
                self._refresh_backend_detection_status()
                return
        ok = ensure_backend_detection_for_rules(
            self.api_base,
            self.camera_id,
            verification_enabled=checked,
        )
        if ok:
            self.backend_detection_status.setText(backend_detection_status_label(checked))
            self.status_label.setText(
                "Backend detection enabled." if checked else "Backend detection disabled."
            )
        else:
            self.status_label.setText("Failed to update backend detection. Is the API running?")
            self._refresh_backend_detection_status()

    def _delete_rule(self) -> None:
        rid = str(self._rule_id or "").strip()
        if not rid:
            return
        confirm = QMessageBox.question(
            self,
            "Delete rule",
            f"Delete rule \"{self.name_edit.text().strip() or rid}\"? This cannot be undone.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if confirm != QMessageBox.StandardButton.Yes:
            return
        if not delete_rule(self.api_base, rid):
            QMessageBox.warning(self, "Event Rule", "Failed to delete rule. Is the API running?")
            return
        self._rule_id = None
        self._reload_rule_combo()
        self.rule_combo.setCurrentIndex(0)
        self._auto_rule_name = True
        label = str(self.shape.get("label") or "Shape")
        self.name_edit.setText(f"{label} trigger")
        self.enabled_check.setChecked(True)
        self._rule_enabled = True
        shape_rules = rules_for_shape(
            list_rules(self.api_base, self.camera_id),
            str(self.shape.get("id") or ""),
        )
        self._counter_pill_anchor = default_counter_pill_anchor(self.shape, len(shape_rules))
        self._motion_path = []
        self._derived_trigger = _default_trigger(self.shape.get("kind"))
        self._derived_direction = ""
        self.status_label.setText("Rule deleted.")
        self._update_rule_action_state()
        self._emit_preview()

    def _clear_path(self) -> None:
        self._motion_path = []
        if self._on_path_draw_control:
            self._on_path_draw_control("clear", [])
        self._derived_trigger = _default_trigger(self.shape.get("kind"))
        self._derived_direction = ""
        self._update_inferred_chips()
        self._emit_preview()

    def _on_pill_anchor_changed(self, anchor: dict) -> None:
        self._counter_pill_anchor = parse_counter_pill_anchor(anchor)
        self._emit_preview()

    def _update_inferred_chips(self) -> None:
        path = self.motion_path_norm()
        mode = str(self.trigger_combo.currentData() or "auto_path")
        if mode in EXPLICIT_SHAPE_TRIGGERS:
            trig = mode
            direction = self._current_direction()
        elif len(path) >= 2:
            trig = self._derived_trigger
            direction = self._derived_direction
        else:
            trig = _default_trigger(self.shape.get("kind"))
            direction = ""
        trig_label = TRIGGER_CHIP_LABELS.get(trig, trig.replace("_", " "))
        dir_label = DIRECTION_LABELS.get(direction, "") if direction else ""
        parts = [f"Inferred: {trig_label}"]
        if dir_label:
            parts.append(dir_label)
        if len(path) >= 2:
            parts.append(f"{len(path)} pts")
        elif str(self.trigger_combo.currentData() or "") == "path_match":
            parts.append("path match")
        else:
            parts.append("draw path")
        self.inferred_label.setText(" · ".join(parts))
        self._update_path_summary()

    def _effective_trigger(self) -> str:
        mode = str(self.trigger_combo.currentData() or "auto_path")
        path = self.motion_path_norm()
        return effective_trigger_from_mode(
            mode=mode,
            shape_kind=str(self.shape.get("kind") or ""),
            derived_trigger=self._derived_trigger,
            has_path=len(path) >= 2,
        )

    def _current_direction(self) -> str:
        if len(self.motion_path_norm()) >= 2:
            return self._derived_direction
        return ""

    def _event_source_flags(self) -> Tuple[bool, bool]:
        return (
            bool(self.motion_source_check.isChecked()),
            bool(self.detection_source_check.isChecked()),
        )

    def _preview_settings(self) -> Dict[str, Any]:
        motion_enabled, detection_enabled = self._event_source_flags()
        classes = (
            [c.strip().lower() for c in self.classes_edit.text().split(",") if c.strip()]
            if detection_enabled
            else []
        )
        show_counter = normalize_counter_mode(self.counter_combo.currentData())
        label = self.counter_label_edit.text().strip()
        if not label:
            label = resolve_counter_pill_label(
                {"label": "", "rule_name": self.name_edit.text().strip()},
                self.shape,
            )
        return {
            "color_bucket": str(self.color_combo.currentData() or ""),
            "classes": classes,
            "require_detection": detection_enabled,
            "dwell_min": float(self.dwell_min_spin.value()),
            "cooldown_sec": cooldown_sec_from_ms(int(self.cooldown_spin.value())),
            "show_counter": show_counter,
            "counter_value": 1 if show_counter != "off" else 0,
            "counter_pill_anchor": dict(self._counter_pill_anchor),
            "counter_pill_label": label,
            "counter_pill_color": str(self.counter_pill_color_combo.currentData() or ""),
            "counter_pill_text_color": str(self.counter_pill_text_combo.currentData() or ""),
        }

    def _emit_preview(self) -> None:
        trigger = self._effective_trigger()
        direction = self._current_direction()
        path = self.motion_path_norm()
        settings = self._preview_settings()
        dwell = float(settings.get("dwell_min") or 0.0)
        cooldown = float(settings.get("cooldown_sec") or DEFAULT_RULE_COOLDOWN_SEC)
        phase_inc, _ = preview_animation_params(dwell, cooldown)
        self._preview_phase = (self._preview_phase + phase_inc) % 1.0
        self._update_inferred_chips()
        if self._on_preview_changed:
            preview_state: Dict[str, Any] = {
                "shape_id": str(self.shape.get("id") or ""),
                "trigger": trigger,
                "direction": direction,
                "phase": self._preview_phase,
                "path_draw_active": self._path_draw_active,
                "pill_move_active": self._pill_move_active,
                **settings,
            }
            if self._pill_move_active and normalize_counter_mode(settings.get("show_counter")) == "off":
                preview_state["show_counter"] = "always"
                preview_state["counter_value"] = 0
            if len(path) >= 2:
                preview_state["motion_path"] = path
            if self._path_draw_active:
                preview_state["extra_pill_items"] = self._sibling_pill_preview_items()
            self._on_preview_changed(preview_state)

    def _on_shape_name_changed(self, text: str) -> None:
        if self._syncing_name:
            return
        self.shape["label"] = text.strip() or "Shape"
        self.setWindowTitle(f"Event Rule — {self.shape['label']}")
        if self._auto_rule_name:
            self._syncing_name = True
            self.name_edit.setText(f"{self.shape['label']} trigger")
            self._syncing_name = False
        if self._on_shape_updated:
            self._on_shape_updated(dict(self.shape))
        self._emit_preview()

    def _on_trigger_mode_changed(self, _idx: int = 0) -> None:
        self._update_inferred_chips()
        self._emit_preview()

    def _on_event_source_changed(self, _checked: bool = False) -> None:
        motion_enabled, detection_enabled = self._event_source_flags()
        if not motion_enabled and not detection_enabled:
            sender = self.sender()
            if sender is self.motion_source_check:
                self.detection_source_check.setChecked(True)
            else:
                self.motion_source_check.setChecked(True)
            motion_enabled, detection_enabled = self._event_source_flags()
        self.classes_edit.setEnabled(detection_enabled)
        self.confidence_spin.setEnabled(detection_enabled)
        self.backend_detection_widget.setVisible(detection_enabled)
        self.motion_box_settings_btn.setVisible(motion_enabled)
        self._update_event_source_label()
        self._emit_preview()

    def _on_show_motion_overlay_toggled(self, checked: bool) -> None:
        if self._syncing_motion_overlay:
            return
        camera_widget = getattr(self, "_camera_widget", None)
        if camera_widget is not None and hasattr(camera_widget, "set_motion_boxes_visible"):
            camera_widget.set_motion_boxes_visible(bool(checked))

    def _get_motion_watch_settings(self) -> Dict[str, Any]:
        camera_widget = getattr(self, "_camera_widget", None)
        if camera_widget is not None:
            settings = getattr(camera_widget, "motion_watch_settings", None)
            if isinstance(settings, dict):
                return dict(settings)
        return load_motion_watch_settings_from_disk(self.camera_id)

    def _on_screenshot_toggled(self, _checked: bool) -> None:
        self.capture_settings_btn.setEnabled(self.screenshot_check.isChecked())
        if hasattr(self, "screenshot_warn_label"):
            self.screenshot_warn_label.setVisible(not self.screenshot_check.isChecked())

    def _on_run_script_toggled(self, checked: bool) -> None:
        for widget in self._script_fields:
            widget.setEnabled(bool(checked))

    def _browse_script_path(self) -> None:
        scripts_dir = ensure_event_rules_scripts_dir()
        selected, _ = QFileDialog.getOpenFileName(
            self,
            "Select Event Rule script",
            str(scripts_dir),
            "Scripts (*.py *.sh);;All files (*)",
        )
        if not selected:
            return
        selected_path = Path(selected)
        try:
            rel = selected_path.resolve().relative_to(scripts_dir.resolve())
            self.script_path_edit.setText(rel.as_posix())
        except ValueError:
            self.script_path_edit.setText(str(selected_path))

    def _open_scripts_folder(self) -> None:
        scripts_dir = ensure_event_rules_scripts_dir()
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(scripts_dir)))

    def _open_capture_settings(self) -> None:
        camera_widget = getattr(self, "_camera_widget", None)
        if camera_widget is not None and hasattr(camera_widget, "open_motion_watch_settings_dialog"):
            camera_widget.open_motion_watch_settings_dialog(settings_only=True)
            return
        QMessageBox.information(
            self,
            "Capture settings",
            "Open this Event Rule from a camera widget to edit capture settings.",
        )

    def _open_motion_box_settings(self) -> None:
        camera_widget = getattr(self, "_camera_widget", None)
        if camera_widget is not None and hasattr(camera_widget, "open_motion_settings"):
            camera_widget.open_motion_settings()
            return
        QMessageBox.information(
            self,
            "Motion box settings",
            "Open this Event Rule from a camera widget to tune motion sensitivity and merge size.",
        )

    def _update_event_source_label(self) -> None:
        motion_enabled, detection_enabled = self._event_source_flags()
        backend = backend_detection_status_label(
            get_backend_detection_enabled(self.api_base, self.camera_id)
        )
        self.event_source_label.setText(
            event_source_description(
                motion_enabled=motion_enabled,
                detection_enabled=detection_enabled,
                backend_status=backend,
            )
        )

    def _reload_rule_combo(self, select_id: Optional[str] = None) -> None:
        while self.rule_combo.count() > 1:
            self.rule_combo.removeItem(1)
        sid = str(self.shape.get("id") or "")
        for rule in rules_for_shape(list_rules(self.api_base, self.camera_id), sid):
            rid = str(rule.get("id") or "")
            rlabel = str(rule.get("name") or rid)
            self.rule_combo.addItem(rlabel, rid)
            if select_id and rid == select_id:
                self.rule_combo.setCurrentIndex(self.rule_combo.count() - 1)

    def _on_rule_selected(self, _idx: int) -> None:
        rid = str(self.rule_combo.currentData() or "")
        if not rid:
            self._rule_id = None
            self._auto_rule_name = True
            self.name_edit.setText(str(self.shape.get("label") or "Shape") + " trigger")
            self.enabled_check.setChecked(True)
            self._rule_enabled = True
            shape_rules = rules_for_shape(
                list_rules(self.api_base, self.camera_id),
                str(self.shape.get("id") or ""),
            )
            self._counter_pill_anchor = default_counter_pill_anchor(self.shape, len(shape_rules))
            auto_idx = self.trigger_combo.findData(DEFAULT_TRIGGER_MODE)
            if auto_idx >= 0:
                self.trigger_combo.setCurrentIndex(auto_idx)
            self._set_counter_mode(DEFAULT_COUNTER_MODE)
            self._update_rule_action_state()
            self._emit_preview()
            return
        try:
            data = _api_get(self.api_base, f"rules/{rid}")
            rule = data.get("data")
            if isinstance(rule, dict):
                self._apply_rule(rule)
                self._rule_id = rid
                self._update_rule_action_state()
        except Exception as e:
            self.status_label.setText(f"Failed to load rule: {e}")

    def _apply_rule(self, rule: Dict[str, Any]) -> None:
        self.name_edit.setText(str(rule.get("name") or ""))
        self._auto_rule_name = False
        self._rule_enabled = bool(rule.get("enabled", True))
        self.enabled_check.setChecked(self._rule_enabled)

        rule_trigger = str(rule.get("trigger") or "")
        cond = rule.get("conditions") if isinstance(rule.get("conditions"), dict) else {}

        trigger_idx = self.trigger_combo.findData(rule_trigger)
        if trigger_idx >= 0:
            self.trigger_combo.setCurrentIndex(trigger_idx)
        elif rule_trigger == "any_interaction" or bool(cond.get("any_interaction")):
            idx = self.trigger_combo.findData("any_interaction")
            if idx >= 0:
                self.trigger_combo.setCurrentIndex(idx)
        else:
            idx = self.trigger_combo.findData("auto_path")
            if idx >= 0:
                self.trigger_combo.setCurrentIndex(idx)

        motion_path = cond.get("motion_path")
        if isinstance(motion_path, list) and len(motion_path) >= 2:
            frame_path = resolve_motion_path_for_frame(
                motion_path,
                self.shape,
                space=cond.get("motion_path_space"),
            )
            if frame_path:
                self._motion_path = frame_path
            trig, direction, dwell = derive_trigger_from_path(self.shape, self.motion_path_norm())
            self._derived_trigger = str(cond.get("derived_trigger") or trig)
            self._derived_direction = str(cond.get("direction") or direction)
            if dwell and float(cond.get("dwell_min") or 0) <= 0:
                self.dwell_min_spin.setValue(float(dwell))

        motion_enabled, detection_enabled = parse_event_source_flags(cond)
        self.motion_source_check.setChecked(motion_enabled)
        self.detection_source_check.setChecked(detection_enabled)

        classes = cond.get("classes") or cond.get("object_classes") or []
        if isinstance(classes, list):
            self.classes_edit.setText(", ".join(str(c) for c in classes))
        self.confidence_spin.setValue(float(cond.get("min_confidence", 0.5) or 0.5))

        self._derived_direction = str(cond.get("direction") or self._derived_direction)

        cidx = self.color_combo.findData(str(cond.get("color") or cond.get("dominant_color") or ""))
        if cidx >= 0:
            self.color_combo.setCurrentIndex(cidx)

        self.dwell_min_spin.setValue(float(cond.get("dwell_min") or cond.get("dwell_min_sec") or 0.0))
        self.cooldown_spin.setValue(
            cooldown_ms_from_sec(float(cond.get("cooldown_sec", DEFAULT_RULE_COOLDOWN_SEC) or DEFAULT_RULE_COOLDOWN_SEC))
        )
        self.path_tol_spin.setValue(float(cond.get("path_match_tolerance") or DEFAULT_PATH_TOLERANCE))
        counter_idx = self.counter_combo.findData(normalize_counter_mode(cond.get("show_counter")))
        if counter_idx >= 0:
            self.counter_combo.setCurrentIndex(counter_idx)
        slot = self._rule_slot_for_id(str(rule.get("id") or ""))
        self._counter_pill_anchor = resolve_counter_pill_anchor(
            self.shape,
            cond.get("counter_pill_anchor"),
            slot=slot,
        )
        self.counter_label_edit.setText(str(cond.get("counter_pill_label") or ""))
        self.counter_group_edit.setText(str(cond.get("counter_group") or ""))
        combine_idx = self.counter_combine_combo.findData(
            normalize_counter_combine(cond.get("counter_combine"), group=str(cond.get("counter_group") or ""))
        )
        if combine_idx >= 0:
            self.counter_combine_combo.setCurrentIndex(combine_idx)
        pill_color = parse_counter_pill_color(cond.get("counter_pill_color"))
        pill_color_idx = self.counter_pill_color_combo.findData(pill_color)
        if pill_color_idx >= 0:
            self.counter_pill_color_combo.setCurrentIndex(pill_color_idx)
        pill_text = parse_counter_pill_color(cond.get("counter_pill_text_color"))
        pill_text_idx = self.counter_pill_text_combo.findData(pill_text)
        if pill_text_idx >= 0:
            self.counter_pill_text_combo.setCurrentIndex(pill_text_idx)
        self.count_min_spin.setValue(float(cond.get("count_min") or 0.0))
        self.count_max_spin.setValue(float(cond.get("count_max") or 0.0))

        self.per_track_check.setChecked(cond.get("cooldown_per_track", True) is not False)

        actions = rule.get("actions") if isinstance(rule.get("actions"), list) else []
        has_snapshot = any(
            isinstance(a, dict) and str(a.get("type")) == "snapshot" for a in actions
        )
        self.screenshot_check.setChecked(has_snapshot)
        self._on_screenshot_toggled(has_snapshot)

        script_cfg = script_action_from_rule(rule)
        has_script = bool(script_cfg)
        self.run_script_check.setChecked(has_script)
        self.script_path_edit.setText(str(script_cfg.get("path") or script_cfg.get("command") or ""))
        runner_idx = self.script_runner_combo.findData(
            normalize_runner(script_cfg.get("language") or script_cfg.get("runner") or DEFAULT_SCRIPT_RUNNER)
        )
        if runner_idx >= 0:
            self.script_runner_combo.setCurrentIndex(runner_idx)
        args = script_cfg.get("args")
        if isinstance(args, list):
            self.script_args_edit.setText(" ".join(str(a) for a in args))
        else:
            self.script_args_edit.setText(str(args or ""))
        timeout_val = script_cfg.get("timeout_sec", script_cfg.get("timeout", DEFAULT_SCRIPT_TIMEOUT_SEC))
        try:
            self.script_timeout_spin.setValue(max(1, int(timeout_val)))
        except Exception:
            self.script_timeout_spin.setValue(int(DEFAULT_SCRIPT_TIMEOUT_SEC))
        self._on_run_script_toggled(has_script)

        self._on_event_source_changed()
        self._update_inferred_chips()
        self._emit_preview()

    def build_rule_payload(self) -> Dict[str, Any]:
        shape_id = str(self.shape.get("id") or "").strip()
        motion_enabled, detection_enabled = self._event_source_flags()
        classes = (
            [c.strip().lower() for c in self.classes_edit.text().split(",") if c.strip()]
            if detection_enabled
            else []
        )
        direction = self._current_direction()
        path = self.motion_path_norm()
        effective_trigger = self._effective_trigger()
        trigger_mode = str(self.trigger_combo.currentData() or "auto_path")
        semantic_trigger = self._derived_trigger
        if trigger_mode in EXPLICIT_SHAPE_TRIGGERS:
            semantic_trigger = trigger_mode

        conditions: Dict[str, Any] = {
            **build_event_source_conditions(
                motion_enabled=motion_enabled,
                detection_enabled=detection_enabled,
                classes=classes or None,
                min_confidence=float(self.confidence_spin.value()) if classes else None,
            ),
            "cooldown_sec": cooldown_sec_from_ms(int(self.cooldown_spin.value())),
            "cooldown_per_track": bool(self.per_track_check.isChecked()),
        }
        if trigger_mode == "any_interaction":
            conditions["any_interaction"] = True
        if direction:
            conditions["direction"] = direction

        dwell_min = float(self.dwell_min_spin.value())
        if dwell_min > 0:
            conditions["dwell_min"] = dwell_min

        color = str(self.color_combo.currentData() or "").strip()
        if color:
            conditions["color"] = color

        show_counter = normalize_counter_mode(self.counter_combo.currentData())
        if show_counter != "off":
            conditions["show_counter"] = show_counter
            conditions["counter_pill_anchor"] = dict(self._counter_pill_anchor)
            pill_label = self.counter_label_edit.text().strip()
            if pill_label:
                conditions["counter_pill_label"] = pill_label
            group = self.counter_group_edit.text().strip()
            if group:
                conditions["counter_group"] = group
                combine = normalize_counter_combine(
                    self.counter_combine_combo.currentData(),
                    group=group,
                )
                if combine != "none":
                    conditions["counter_combine"] = combine
            pill_color = str(self.counter_pill_color_combo.currentData() or "").strip()
            if pill_color:
                conditions["counter_pill_color"] = pill_color
            pill_text = str(self.counter_pill_text_combo.currentData() or "").strip()
            if pill_text:
                conditions["counter_pill_text_color"] = pill_text

        count_min = int(self.count_min_spin.value())
        count_max = int(self.count_max_spin.value())
        if count_min > 0:
            conditions["count_min"] = count_min
        if count_max > 0:
            conditions["count_max"] = count_max

        if len(path) >= 2 and trigger_mode != "any_interaction":
            conditions["motion_path"] = path
            conditions["motion_path_space"] = MOTION_PATH_SPACE_FRAME
            conditions["motion_path_shape_ref"] = motion_path_shape_ref_from_shape(self.shape)
            conditions["path_match_tolerance"] = float(self.path_tol_spin.value())
            conditions["derived_trigger"] = semantic_trigger
            direction_gate = compute_path_direction_gate(path, self.shape)
            if direction_gate:
                conditions["path_direction_gate"] = direction_gate

        actions = build_rule_actions(
            take_screenshot=self.screenshot_check.isChecked(),
            motion_watch_settings=self._get_motion_watch_settings(),
            run_script=self.run_script_check.isChecked(),
            script_path=self.script_path_edit.text().strip(),
            script_runner=str(self.script_runner_combo.currentData() or DEFAULT_SCRIPT_RUNNER),
            script_args=self.script_args_edit.text().strip(),
            script_timeout_sec=int(self.script_timeout_spin.value()),
        )

        payload: Dict[str, Any] = {
            "name": self.name_edit.text().strip() or "Shape trigger",
            "camera_id": self.camera_id,
            "trigger": effective_trigger,
            "shape_id": shape_id,
            "conditions": conditions,
            "actions": actions,
            "enabled": bool(self.enabled_check.isChecked()),
        }
        if not self._rule_id:
            payload["id"] = f"rule_{uuid.uuid4().hex[:10]}"
        return payload

    def _save(self, *, arm: bool) -> None:
        self._stop_camera_path_draw()
        self._stop_camera_pill_move()
        path = self.motion_path_norm()
        trigger_mode = str(self.trigger_combo.currentData() or "auto_path")
        if trigger_mode != "any_interaction" and len(path) < 2:
            QMessageBox.information(
                self,
                "Event Rule",
                "Draw the expected motion path on the camera to define this trigger.",
            )
            return
        if self.run_script_check.isChecked() and not self.script_path_edit.text().strip():
            QMessageBox.information(
                self,
                "Event Rule",
                "Enter a script path or disable Run script on trigger.",
            )
            return
        motion_enabled, detection_enabled = self._event_source_flags()
        if not motion_enabled and not detection_enabled:
            QMessageBox.information(
                self,
                "Event Rule",
                "Select at least one event source (Motion boxes and/or Object detection).",
            )
            return
        if self._on_shape_updated:
            self._on_shape_updated(dict(self.shape))
        body = self.build_rule_payload()
        saved = save_rule(self.api_base, body, rule_id=self._rule_id)
        if not saved:
            QMessageBox.warning(self, "Event Rule", "Failed to save rule. Is the API running?")
            return
        self.saved_rule = saved
        self._rule_id = str(saved.get("id") or self._rule_id)
        self._rule_enabled = bool(saved.get("enabled", self.enabled_check.isChecked()))
        self._reload_rule_combo(select_id=self._rule_id)
        self._update_rule_action_state()
        if arm:
            set_rules_enabled(self.api_base, self.camera_id, True)
            if detection_enabled:
                ensure_backend_detection_for_rules(
                    self.api_base,
                    self.camera_id,
                    verification_enabled=True,
                )
            self._refresh_backend_detection_status()
        self.status_label.setText(f"Saved rule {self._rule_id}" + (" and armed." if arm else "."))
        self._arm_on_accept = arm
        self._stop_camera_path_draw()
        self._stop_camera_pill_move()
        if self._on_preview_changed:
            self._on_preview_changed(None)
        self.accept()

    def should_arm(self) -> bool:
        return bool(self._arm_on_accept)

    def saved_rule_id(self) -> Optional[str]:
        return self._rule_id
