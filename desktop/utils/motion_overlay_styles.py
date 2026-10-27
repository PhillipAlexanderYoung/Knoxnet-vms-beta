"""
Motion box styles and animations for the desktop camera overlay.

Styles are grouped by vibe (classic, modern, tactical, minimal, cosmetic,
funny, emoji). Animations are grouped similarly. Rendering is centralized
here so camera.py stays readable.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from PySide6.QtCore import Qt, QRectF, QPointF
from PySide6.QtGui import QBrush, QColor, QFont, QLinearGradient, QPainter, QPen, QPolygonF


# ---------------------------------------------------------------------------
# Catalog
# ---------------------------------------------------------------------------

MOTION_STYLE_CATEGORIES: Dict[str, List[str]] = {
    "Classic": [
        "Box", "Fill", "Corners", "Circle", "Bracket", "Underline", "Crosshair",
    ],
    "Modern": [
        "HUD", "Scanline", "Reticle", "Hexagon", "Diamond", "Pill", "Double Box", "Cross Grid",
    ],
    "Tactical": [
        "Threat Mark", "NATO Bracket", "Range Ring", "Vector Track", "Lock Box",
        "Compass", "Bearing", "IFF Box",
    ],
    "Minimal": [
        "Dot", "Wire", "Ghost", "Pin", "Dash", "Cross Dot",
    ],
    "Cosmetic": [
        "Neon", "Glass", "Hologram", "Aurora", "Sparkle Corner", "Ribbon", "Gradient Fill",
    ],
    "Funny": [
        "Confetti", "Cartoon", "Disco", "Zany", "Boing", "Party Frame",
    ],
    "Emoji": [
        "👀 Eyes", "👻 Ghost", "🔥 Fire", "🎯 Target", "🏃 Run", "⚠️ Alert", "💨 Speed", "🤖 Bot",
    ],
}

MOTION_ANIMATION_CATEGORIES: Dict[str, List[str]] = {
    "Basic": ["None", "Pulse", "Flash", "Glitch", "Rainbow"],
    "Modern": ["Glow", "Breathe", "Neon Pulse", "Float", "Scan"],
    "Tactical": ["Radar", "Strobe", "Tactical Lock", "Matrix", "Target Ping"],
    "Minimal": ["Flicker", "Fade", "Drift"],
    "Cosmetic": ["Sparkle", "Aurora Shift", "Shimmer", "Orbit"],
    "Funny": ["Shake", "Wiggle", "Bounce", "Party", "Dizzy"],
}

MOTION_STYLE_QUICK_PICK: List[str] = [
    "Box", "Corners", "HUD", "Reticle", "Neon", "👀 Eyes", "Threat Mark",
]

_EMOJI_STYLE_MAP: Dict[str, str] = {
    "👀 Eyes": "👀",
    "👻 Ghost": "👻",
    "🔥 Fire": "🔥",
    "🎯 Target": "🎯",
    "🏃 Run": "🏃",
    "⚠️ Alert": "⚠️",
    "💨 Speed": "💨",
    "🤖 Bot": "🤖",
}


def all_motion_styles() -> List[str]:
    out: List[str] = []
    for items in MOTION_STYLE_CATEGORIES.values():
        out.extend(items)
    return out


def all_motion_animations() -> List[str]:
    out: List[str] = []
    for items in MOTION_ANIMATION_CATEGORIES.values():
        out.extend(items)
    return out


def normalize_motion_style(style: str) -> str:
    s = str(style or "Box")
    if s in all_motion_styles():
        return s
    # Legacy / unknown → classic box
    return "Box"


def normalize_motion_animation(anim: str) -> str:
    a = str(anim or "None")
    if a in all_motion_animations():
        return a
    legacy = {"Glow": "Glow"}
    return legacy.get(a, "None")


def populate_categorized_combo(combo, categories: Dict[str, List[str]], current: str) -> None:
    """Fill a QComboBox with category headers (disabled) + items."""
    combo.blockSignals(True)
    combo.clear()
    for cat, items in categories.items():
        combo.addItem(f"— {cat} —")
        header_idx = combo.count() - 1
        try:
            combo.model().item(header_idx).setEnabled(False)
        except Exception:
            pass
        for item in items:
            combo.addItem(item)
    idx = combo.findText(current)
    if idx >= 0:
        combo.setCurrentIndex(idx)
    combo.blockSignals(False)


def combo_selection_is_valid(text: str) -> bool:
    return bool(text) and not str(text).startswith("—")


# ---------------------------------------------------------------------------
# Animation state
# ---------------------------------------------------------------------------

@dataclass
class MotionAnimState:
    color: QColor
    draw_x: float
    draw_y: float
    draw_w: float
    draw_h: float
    cx: float
    cy: float
    alpha_mult: float = 1.0
    rotation_deg: float = 0.0
    scan_y: Optional[float] = None
    radar_angle: Optional[float] = None
    extra_colors: List[QColor] = field(default_factory=list)
    draw_scanline: bool = False
    draw_orbit_dot: bool = False
    draw_sparkles: bool = False
    matrix_chars: bool = False
    lock_tightness: float = 0.0


def apply_motion_animation(
    anim: str,
    t: float,
    base_color: QColor,
    draw_x: float,
    draw_y: float,
    draw_w: float,
    draw_h: float,
    cx: float,
    cy: float,
    obj: dict,
    *,
    color_speed: bool = False,
) -> MotionAnimState:
    """Compute per-frame animation transforms and color shifts."""
    anim = normalize_motion_animation(anim)
    final_color = QColor(base_color)

    if color_speed:
        sp = min(float(obj.get("speed", 0) or 0), 20.0)
        hue = 120.0 - (sp / 20.0 * 120.0)
        final_color.setHslF(hue / 360.0, 1.0, 0.5)

    state = MotionAnimState(
        color=final_color,
        draw_x=draw_x,
        draw_y=draw_y,
        draw_w=draw_w,
        draw_h=draw_h,
        cx=cx,
        cy=cy,
    )

    if anim == "Rainbow" or anim == "Aurora Shift":
        hue = (t * (55 if anim == "Rainbow" else 25)) % 360.0
        final_color.setHslF(hue / 360.0, 1.0, 0.55)
        state.color = final_color

    alpha_mult = 1.0
    size_mult = 1.0

    if anim == "Pulse":
        val = (math.sin(t * 5) + 1) / 2
        alpha_mult = 0.4 + 0.6 * val
        size_mult = 1.0 + 0.05 * val
    elif anim == "Flash" or anim == "Strobe":
        rate = 8 if anim == "Flash" else 14
        alpha_mult = 1.0 if int(t * rate) % 2 == 0 else 0.15
    elif anim == "Glitch":
        if random.random() > 0.65:
            state.draw_x += random.randint(-6, 6)
            state.draw_y += random.randint(-6, 6)
            state.draw_w += random.randint(-4, 4)
            state.draw_h += random.randint(-4, 4)
            if random.random() > 0.5:
                state.color = QColor(0, 255, 255) if random.random() > 0.5 else QColor(255, 0, 255)
    elif anim == "Glow" or anim == "Neon Pulse":
        val = (math.sin(t * 3.5) + 1) / 2
        alpha_mult = 0.55 + 0.45 * val
        size_mult = 1.0 + (0.08 if anim == "Neon Pulse" else 0.04) * val
    elif anim == "Breathe":
        val = (math.sin(t * 2.2) + 1) / 2
        size_mult = 0.92 + 0.12 * val
        alpha_mult = 0.65 + 0.35 * val
    elif anim == "Float":
        state.draw_y += math.sin(t * 2.8) * 4.0
        state.cy = state.draw_y + state.draw_h / 2
    elif anim == "Bounce":
        bounce = abs(math.sin(t * 6)) * 8
        state.draw_y -= bounce
        state.cy = state.draw_y + state.draw_h / 2
    elif anim == "Shake" or anim == "Wiggle":
        amp = 4 if anim == "Shake" else 6
        state.draw_x += math.sin(t * 22) * amp
        state.draw_y += math.cos(t * 19) * amp
        state.cx = state.draw_x + state.draw_w / 2
        state.cy = state.draw_y + state.draw_h / 2
    elif anim == "Dizzy":
        state.rotation_deg = (t * 120) % 360
    elif anim == "Radar":
        state.radar_angle = (t * 2.5) % (2 * math.pi)
        state.draw_scanline = True
    elif anim == "Scan":
        frac = (t * 0.85) % 1.0
        state.scan_y = state.draw_y + state.draw_h * frac
    elif anim == "Flicker":
        alpha_mult = 0.5 + random.random() * 0.5
    elif anim == "Fade":
        alpha_mult = 0.35 + 0.65 * ((math.sin(t * 1.8) + 1) / 2)
    elif anim == "Drift":
        state.draw_x += math.sin(t * 1.2) * 3
        state.cx = state.draw_x + state.draw_w / 2
    elif anim == "Orbit":
        state.draw_orbit_dot = True
    elif anim == "Sparkle" or anim == "Shimmer" or anim == "Party":
        state.draw_sparkles = True
    elif anim == "Matrix":
        state.matrix_chars = True
        state.color = QColor(0, 255, 90)
    elif anim == "Tactical Lock" or anim == "Target Ping":
        val = (math.sin(t * 4) + 1) / 2
        state.lock_tightness = val
        size_mult = 1.0 - 0.12 * val if anim == "Tactical Lock" else 1.0 + 0.06 * val

    if size_mult != 1.0:
        state.draw_w *= size_mult
        state.draw_h *= size_mult
        state.draw_x = state.cx - state.draw_w / 2
        state.draw_y = state.cy - state.draw_h / 2
        state.cx = state.draw_x + state.draw_w / 2
        state.cy = state.draw_y + state.draw_h / 2

    c = QColor(state.color)
    c.setAlphaF(min(1.0, c.alphaF() * alpha_mult))
    state.color = c
    state.alpha_mult = alpha_mult
    return state


# ---------------------------------------------------------------------------
# Drawing helpers
# ---------------------------------------------------------------------------

def _corner_brackets(
    painter: QPainter,
    x: float, y: float, w: float, h: float,
    color: QColor,
    thickness: int,
    arm: float,
) -> None:
    pen = QPen(color)
    pen.setWidth(thickness)
    painter.setPen(pen)
    xi, yi, wi, hi = int(x), int(y), int(w), int(h)
    la, lb = int(arm), int(arm)
    # TL
    painter.drawLine(xi, yi, xi + la, yi)
    painter.drawLine(xi, yi, xi, yi + lb)
    # TR
    painter.drawLine(xi + wi, yi, xi + wi - la, yi)
    painter.drawLine(xi + wi, yi, xi + wi, yi + lb)
    # BL
    painter.drawLine(xi, yi + hi, xi + la, yi + hi)
    painter.drawLine(xi, yi + hi, xi, yi + hi - lb)
    # BR
    painter.drawLine(xi + wi, yi + hi, xi + wi - la, yi + hi)
    painter.drawLine(xi + wi, yi + hi, xi + wi, yi + hi - lb)


def _draw_emoji(
    painter: QPainter,
    emoji: str,
    cx: float,
    cy: float,
    box_size: float,
) -> None:
    size = max(14, min(box_size * 0.85, 48))
    font = QFont()
    font.setFamilies(["Noto Color Emoji", "Segoe UI Emoji", "Apple Color Emoji", "sans-serif"])
    font.setPixelSize(int(size))
    painter.setFont(font)
    painter.setPen(QColor(255, 255, 255))
    rect = QRectF(cx - size, cy - size, size * 2, size * 2)
    painter.drawText(rect, int(Qt.AlignmentFlag.AlignCenter), emoji)


def _draw_animation_fx(
    painter: QPainter,
    state: MotionAnimState,
    anim: str,
    t: float,
    thickness: int,
) -> None:
    anim = normalize_motion_animation(anim)
    cx, cy = state.cx, state.cy
    color = state.color

    if state.draw_scanline or anim == "Radar":
        pen = QPen(QColor(color.red(), color.green(), color.blue(), 140))
        pen.setWidth(max(1, thickness))
        painter.setPen(pen)
        angle = state.radar_angle if state.radar_angle is not None else 0
        r = max(state.draw_w, state.draw_h) * 0.55
        ex = cx + math.cos(angle) * r
        ey = cy + math.sin(angle) * r
        painter.drawLine(QPointF(cx, cy), QPointF(ex, ey))

    if state.scan_y is not None:
        pen = QPen(QColor(0, 255, 200, 180))
        pen.setWidth(max(2, thickness))
        painter.setPen(pen)
        sy = state.scan_y
        painter.drawLine(int(state.draw_x), int(sy), int(state.draw_x + state.draw_w), int(sy))

    if state.draw_orbit_dot:
        r = max(state.draw_w, state.draw_h) * 0.55
        ang = t * 4.5
        ox = cx + math.cos(ang) * r
        oy = cy + math.sin(ang) * r
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(color)
        painter.drawEllipse(QPointF(ox, oy), 4, 4)

    if state.draw_sparkles:
        rng = random.Random(int(t * 10))
        for _ in range(6):
            sx = state.draw_x + rng.random() * state.draw_w
            sy = state.draw_y + rng.random() * state.draw_h
            sc = QColor(color)
            sc.setAlpha(rng.randint(80, 220))
            painter.setPen(QPen(sc, 2))
            painter.drawLine(QPointF(sx - 3, sy), QPointF(sx + 3, sy))
            painter.drawLine(QPointF(sx, sy - 3), QPointF(sx, sy + 3))

    if state.matrix_chars:
        font = QFont("monospace")
        font.setPixelSize(max(8, thickness * 3))
        painter.setFont(font)
        cols = max(2, int(state.draw_w / 10))
        for i in range(cols):
            col_x = state.draw_x + i * (state.draw_w / cols)
            drop = (t * 40 + i * 17) % (state.draw_h + 20)
            ch = chr(0x30A0 + (i * 7 + int(t * 5)) % 96)
            painter.setPen(QColor(0, 255, 90, 160))
            painter.drawText(QPointF(col_x, state.draw_y + drop), ch)


def draw_motion_box_style(
    painter: QPainter,
    style: str,
    state: MotionAnimState,
    thickness: int,
    t: float,
    obj: dict,
    *,
    anim: str = "None",
) -> None:
    """Draw a single motion box using the selected style."""
    style = normalize_motion_style(style)
    anim = normalize_motion_animation(anim)

    x, y, w, h = state.draw_x, state.draw_y, state.draw_w, state.draw_h
    cx, cy = state.cx, state.cy
    color = state.color

    painter.save()
    if state.rotation_deg:
        painter.translate(cx, cy)
        painter.rotate(state.rotation_deg)
        painter.translate(-cx, -cy)

    lock_inset = state.lock_tightness * min(w, h) * 0.15
    if state.lock_tightness > 0:
        x += lock_inset
        y += lock_inset
        w -= lock_inset * 2
        h -= lock_inset * 2

    rect = QRectF(x, y, w, h)
    xi, yi, wi, hi = int(x), int(y), int(w), int(h)

    # --- Classic ---
    if style == "Box":
        pen = QPen(color)
        pen.setWidth(thickness)
        painter.setPen(pen)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawRect(rect)

    elif style == "Fill":
        painter.setPen(Qt.PenStyle.NoPen)
        fill = QColor(color)
        fill.setAlpha(min(fill.alpha(), 100))
        painter.setBrush(fill)
        painter.drawRect(rect)

    elif style == "Circle":
        pen = QPen(color)
        pen.setWidth(thickness)
        painter.setPen(pen)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawEllipse(rect)

    elif style == "Underline":
        pen = QPen(color)
        pen.setWidth(thickness)
        painter.setPen(pen)
        painter.drawLine(xi, yi + hi, xi + wi, yi + hi)

    elif style == "Crosshair":
        pen = QPen(color)
        pen.setWidth(thickness)
        painter.setPen(pen)
        len_c = min(w, h) / 2
        painter.drawLine(int(cx - len_c), int(cy), int(cx + len_c), int(cy))
        painter.drawLine(int(cx), int(cy - len_c), int(cx), int(cy + len_c))

    elif style == "Bracket":
        arm = 5
        pen = QPen(color)
        pen.setWidth(thickness)
        painter.setPen(pen)
        painter.drawLine(xi, yi, xi, yi + hi)
        painter.drawLine(xi, yi, xi + arm, yi)
        painter.drawLine(xi, yi + hi, xi + arm, yi + hi)
        painter.drawLine(xi + wi, yi, xi + wi, yi + hi)
        painter.drawLine(xi + wi, yi, xi + wi - arm, yi)
        painter.drawLine(xi + wi, yi + hi, xi + wi - arm, yi + hi)

    elif style == "Corners":
        len_x = min(w / 3, 20)
        len_y = min(h / 3, 20)
        pen = QPen(color)
        pen.setWidth(thickness)
        painter.setPen(pen)
        _corner_brackets(painter, x, y, w, h, color, thickness, max(len_x, len_y))

    # --- Modern ---
    elif style == "HUD":
        arm = min(w, h) * 0.22
        _corner_brackets(painter, x, y, w, h, color, thickness, arm)
        pen = QPen(QColor(color.red(), color.green(), color.blue(), 120))
        pen.setWidth(1)
        painter.setPen(pen)
        painter.drawLine(int(cx - 6), int(cy), int(cx + 6), int(cy))
        painter.drawLine(int(cx), int(cy - 6), int(cx), int(cy + 6))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(color)
        painter.drawEllipse(QPointF(cx, cy), 2, 2)

    elif style == "Scanline":
        pen = QPen(color)
        pen.setWidth(max(1, thickness))
        painter.setPen(pen)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawRect(rect)
        sy = y + ((t * 90) % max(h, 1))
        scan_pen = QPen(QColor(0, 255, 220, 200))
        scan_pen.setWidth(2)
        painter.setPen(scan_pen)
        painter.drawLine(xi, int(sy), xi + wi, int(sy))

    elif style == "Reticle":
        pen = QPen(color)
        pen.setWidth(thickness)
        painter.setPen(pen)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        r1 = min(w, h) * 0.35
        r2 = min(w, h) * 0.55
        painter.drawEllipse(QRectF(cx - r1, cy - r1, r1 * 2, r1 * 2))
        painter.drawEllipse(QRectF(cx - r2, cy - r2, r2 * 2, r2 * 2))
        gap = 6
        painter.drawLine(int(cx - r2), int(cy), int(cx - gap), int(cy))
        painter.drawLine(int(cx + gap), int(cy), int(cx + r2), int(cy))
        painter.drawLine(int(cx), int(cy - r2), int(cx), int(cy - gap))
        painter.drawLine(int(cx), int(cy + gap), int(cx), int(cy + r2))

    elif style == "Hexagon":
        r = min(w, h) / 2
        pts = []
        for i in range(6):
            ang = math.pi / 3 * i - math.pi / 6
            pts.append(QPointF(cx + r * math.cos(ang), cy + r * math.sin(ang)))
        pen = QPen(color)
        pen.setWidth(thickness)
        painter.setPen(pen)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawPolygon(QPolygonF(pts))

    elif style == "Diamond":
        pts = [
            QPointF(cx, y), QPointF(x + w, cy), QPointF(cx, y + h), QPointF(x, cy),
        ]
        pen = QPen(color)
        pen.setWidth(thickness)
        painter.setPen(pen)
        painter.drawPolygon(QPolygonF(pts))

    elif style == "Pill":
        pen = QPen(color)
        pen.setWidth(thickness)
        painter.setPen(pen)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawRoundedRect(rect, h / 2, h / 2)

    elif style == "Double Box":
        pen = QPen(color)
        pen.setWidth(thickness)
        painter.setPen(pen)
        painter.drawRect(rect)
        inset = max(3, thickness * 2)
        painter.drawRect(QRectF(x + inset, y + inset, w - inset * 2, h - inset * 2))

    elif style == "Cross Grid":
        pen = QPen(color)
        pen.setWidth(max(1, thickness))
        painter.setPen(pen)
        painter.drawRect(rect)
        painter.drawLine(int(cx), int(y), int(cx), int(y + h))
        painter.drawLine(int(x), int(cy), int(x + w), int(cy))
        for i in range(1, 3):
            fx = x + w * i / 3
            fy = y + h * i / 3
            pen.setStyle(Qt.PenStyle.DotLine)
            painter.setPen(pen)
            painter.drawLine(int(fx), yi, int(fx), yi + hi)
            painter.drawLine(xi, int(fy), xi + wi, int(fy))
            pen.setStyle(Qt.PenStyle.SolidLine)
            painter.setPen(pen)

    # --- Tactical ---
    elif style == "Threat Mark":
        pen = QPen(color)
        pen.setWidth(thickness)
        painter.setPen(pen)
        painter.drawLine(xi, yi, xi + wi, yi + hi)
        painter.drawLine(xi + wi, yi, xi, yi + hi)
        _corner_brackets(painter, x, y, w, h, color, max(1, thickness - 1), min(w, h) * 0.15)

    elif style == "NATO Bracket":
        pen = QPen(color)
        pen.setWidth(thickness)
        painter.setPen(pen)
        cut = min(w, h) * 0.2
        # angular cut corners
        painter.drawLine(xi, yi + cut, xi, yi)
        painter.drawLine(xi, yi, xi + cut, yi)
        painter.drawLine(xi + wi - cut, yi, xi + wi, yi)
        painter.drawLine(xi + wi, yi, xi + wi, yi + cut)
        painter.drawLine(xi + wi, yi + hi - cut, xi + wi, yi + hi)
        painter.drawLine(xi + wi, yi + hi, xi + wi - cut, yi + hi)
        painter.drawLine(xi + cut, yi + hi, xi, yi + hi)
        painter.drawLine(xi, yi + hi, xi, yi + hi - cut)

    elif style == "Range Ring":
        pen = QPen(color)
        pen.setWidth(thickness)
        painter.setPen(pen)
        for mult in (0.35, 0.55, 0.75):
            r = min(w, h) * mult
            painter.drawEllipse(QRectF(cx - r, cy - r, r * 2, r * 2))

    elif style == "Vector Track":
        pen = QPen(color)
        pen.setWidth(thickness)
        painter.setPen(pen)
        painter.drawRect(rect)
        hist = obj.get("history") or []
        if len(hist) >= 2:
            lx, ly = hist[-2]
            hx, hy = hist[-1]
            dx, dy = hx - lx, hy - ly
            dist = math.hypot(dx, dy)
            if dist > 1:
                ang = math.atan2(dy, dx)
                alen = min(w, h) * 0.4 + float(obj.get("speed", 0) or 0) * 2
                alen = max(alen, 8.0)
                ex = cx + math.cos(ang) * alen
                ey = cy + math.sin(ang) * alen
                painter.drawLine(QPointF(cx, cy), QPointF(ex, ey))
                a1, a2 = ang + 2.6, ang - 2.6
                painter.drawLine(QPointF(ex, ey), QPointF(ex + math.cos(a1) * 8, ey + math.sin(a1) * 8))
                painter.drawLine(QPointF(ex, ey), QPointF(ex + math.cos(a2) * 8, ey + math.sin(a2) * 8))

    elif style == "Lock Box":
        pen = QPen(color)
        pen.setStyle(Qt.PenStyle.DashLine)
        pen.setWidth(thickness)
        painter.setPen(pen)
        painter.drawRect(rect)
        pen.setStyle(Qt.PenStyle.SolidLine)
        _corner_brackets(painter, x, y, w, h, color, thickness + 1, min(w, h) * 0.18)

    elif style == "Compass":
        pen = QPen(color)
        pen.setWidth(max(1, thickness))
        painter.setPen(pen)
        painter.drawRect(rect)
        tick = max(4, thickness * 2)
        painter.drawLine(int(cx), int(y - tick), int(cx), int(y))
        painter.drawLine(int(cx), int(y + h), int(cx), int(y + h + tick))
        painter.drawLine(int(x - tick), int(cy), int(x), int(cy))
        painter.drawLine(int(x + w), int(cy), int(x + w + tick), int(cy))

    elif style == "Bearing":
        hist = obj.get("history") or []
        pen = QPen(color)
        pen.setWidth(thickness)
        painter.setPen(pen)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawEllipse(QPointF(cx, cy), 4, 4)
        if len(hist) >= 2:
            lx, ly = hist[-2]
            hx, hy = hist[-1]
            dx, dy = hx - lx, hy - ly
            dist = math.hypot(dx, dy)
            if dist > 1:
                ang = math.atan2(dy, dx)
                alen = min(w, h) * 0.45
                ex = cx + math.cos(ang) * alen
                ey = cy + math.sin(ang) * alen
                pen.setStyle(Qt.PenStyle.DashDotLine)
                painter.setPen(pen)
                painter.drawLine(QPointF(cx, cy), QPointF(ex, ey))

    elif style == "IFF Box":
        pen = QPen(color)
        pen.setWidth(thickness)
        painter.setPen(pen)
        painter.drawRect(rect)
        painter.drawLine(xi, yi, xi + wi, yi + hi)

    # --- Minimal ---
    elif style == "Dot":
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(color)
        r = max(3, thickness + 1)
        painter.drawEllipse(QPointF(cx, cy), r, r)

    elif style == "Wire":
        pen = QPen(color)
        pen.setWidth(1)
        painter.setPen(pen)
        painter.drawRect(rect)

    elif style == "Ghost":
        pen = QPen(QColor(color.red(), color.green(), color.blue(), 70))
        pen.setWidth(1)
        painter.setPen(pen)
        fill = QColor(color.red(), color.green(), color.blue(), 25)
        painter.setBrush(fill)
        painter.drawRect(rect)

    elif style == "Pin":
        pen = QPen(color)
        pen.setWidth(max(1, thickness))
        painter.setPen(pen)
        painter.drawLine(int(cx), int(y - min(h * 0.3, 18)), int(cx), int(cy))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(color)
        painter.drawEllipse(QPointF(cx, cy), 3, 3)

    elif style == "Dash":
        pen = QPen(color)
        pen.setWidth(thickness)
        pen.setStyle(Qt.PenStyle.DashLine)
        painter.setPen(pen)
        painter.drawRect(rect)

    elif style == "Cross Dot":
        pen = QPen(color)
        pen.setWidth(max(1, thickness))
        painter.setPen(pen)
        s = 5
        painter.drawLine(int(cx - s), int(cy), int(cx + s), int(cy))
        painter.drawLine(int(cx), int(cy - s), int(cx), int(cy + s))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(color)
        painter.drawEllipse(QPointF(cx, cy), 2, 2)

    # --- Cosmetic ---
    elif style == "Neon":
        for i, alpha in enumerate((40, 90, 180)):
            glow = QColor(color.red(), color.green(), color.blue(), alpha)
            pen = QPen(glow)
            pen.setWidth(thickness + (3 - i) * 2)
            painter.setPen(pen)
            painter.drawRect(rect)

    elif style == "Glass":
        fill = QColor(color.red(), color.green(), color.blue(), 45)
        painter.setPen(QPen(QColor(255, 255, 255, 140), max(1, thickness)))
        painter.setBrush(fill)
        painter.drawRect(rect)
        painter.drawLine(xi, yi + 2, xi + wi, yi + 2)

    elif style == "Hologram":
        painter.setPen(QPen(color, max(1, thickness)))
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawRect(rect)
        stripe_pen = QPen(QColor(color.red(), color.green(), color.blue(), 60))
        stripe_pen.setWidth(1)
        painter.setPen(stripe_pen)
        step = 6
        for i in range(int(x), int(x + w + h), step):
            painter.drawLine(i, yi, i - int(h), yi + hi)

    elif style == "Aurora":
        grad = QLinearGradient(x, y, x + w, y + h)
        hue = (t * 30) % 360
        c1 = QColor.fromHslF(hue / 360, 0.9, 0.55)
        c2 = QColor.fromHslF(((hue + 80) % 360) / 360, 0.9, 0.55)
        grad.setColorAt(0, c1)
        grad.setColorAt(1, c2)
        pen = QPen(QBrush(grad), thickness + 1)
        painter.setPen(pen)
        painter.drawRect(rect)

    elif style == "Sparkle Corner":
        _corner_brackets(painter, x, y, w, h, color, thickness, min(w, h) * 0.2)
        painter.setPen(QPen(color, 2))
        for px, py in ((x, y), (x + w, y), (x, y + h), (x + w, y + h)):
            painter.drawLine(int(px - 4), int(py), int(px + 4), int(py))
            painter.drawLine(int(px), int(py - 4), int(px), int(py + 4))

    elif style == "Ribbon":
        pen = QPen(color)
        pen.setWidth(thickness)
        painter.setPen(pen)
        painter.drawRect(rect)
        bow_y = y - 4
        painter.setBrush(QColor(color.red(), color.green(), color.blue(), 180))
        painter.drawEllipse(QRectF(cx - 8, bow_y - 6, 8, 8))
        painter.drawEllipse(QRectF(cx, bow_y - 6, 8, 8))

    elif style == "Gradient Fill":
        grad = QLinearGradient(x, y, x, y + h)
        top = QColor(color)
        top.setAlpha(160)
        bot = QColor(color.red(), color.green(), color.blue(), 30)
        grad.setColorAt(0, top)
        grad.setColorAt(1, bot)
        painter.setPen(QPen(color, max(1, thickness)))
        painter.setBrush(QBrush(grad))
        painter.drawRect(rect)

    # --- Funny ---
    elif style == "Confetti":
        pen = QPen(color)
        pen.setWidth(thickness)
        painter.setPen(pen)
        painter.drawRect(rect)
        rng = random.Random(int(cx + cy))
        for _ in range(10):
            px = x + rng.random() * w
            py = y + rng.random() * h
            cc = QColor.fromHslF(rng.random(), 0.9, 0.55)
            painter.setPen(QPen(cc, 3))
            painter.drawPoint(QPointF(px, py))

    elif style == "Cartoon":
        pen = QPen(QColor(20, 20, 20))
        pen.setWidth(thickness + 2)
        painter.setPen(pen)
        fill = QColor(255, 255, 100, 120)
        painter.setBrush(fill)
        painter.drawRoundedRect(rect, 6, 6)

    elif style == "Disco":
        segs = 8
        for i in range(segs):
            hue = ((t * 80) + i * (360 / segs)) % 360
            seg_color = QColor.fromHslF(hue / 360, 0.85, 0.55)
            pen = QPen(seg_color)
            pen.setWidth(thickness + 1)
            painter.setPen(pen)
            ang1 = 2 * math.pi * i / segs
            ang2 = 2 * math.pi * (i + 1) / segs
            r = min(w, h) / 2
            p1 = QPointF(cx + r * math.cos(ang1), cy + r * math.sin(ang1))
            p2 = QPointF(cx + r * math.cos(ang2), cy + r * math.sin(ang2))
            painter.drawLine(p1, p2)

    elif style == "Zany":
        pen = QPen(color)
        pen.setWidth(thickness)
        painter.setPen(pen)
        wobble = 3
        pts = [
            (x, y), (x + w * 0.3, y - wobble), (x + w * 0.7, y + wobble), (x + w, y),
            (x + w + wobble, y + h * 0.4), (x + w - wobble, y + h * 0.8), (x + w, y + h),
            (x + w * 0.6, y + h + wobble), (x + w * 0.2, y + h - wobble), (x, y + h),
            (x - wobble, y + h * 0.5), (x + wobble, y + h * 0.2), (x, y),
        ]
        for i in range(len(pts) - 1):
            painter.drawLine(int(pts[i][0]), int(pts[i][1]), int(pts[i + 1][0]), int(pts[i + 1][1]))

    elif style == "Boing":
        pen = QPen(color)
        pen.setWidth(thickness)
        painter.setPen(pen)
        painter.drawRect(rect)
        coils = 4
        base_y = y + h + 2
        coil_w = w / coils
        for i in range(coils):
            sx = x + i * coil_w
            painter.drawArc(QRectF(sx, base_y - 6, coil_w, 12), 0, 180 * 16)

    elif style == "Party Frame":
        pen = QPen(color)
        pen.setWidth(thickness)
        painter.setPen(pen)
        painter.drawRect(rect)
        hues = [0, 60, 120, 180, 240, 300]
        for i, (px, py) in enumerate(((x, y), (x + w, y), (x, y + h), (x + w, y + h))):
            c = QColor.fromHslF(hues[i % len(hues)] / 360, 1.0, 0.55)
            painter.setBrush(c)
            painter.setPen(Qt.PenStyle.NoPen)
            painter.drawEllipse(QPointF(px, py), 5, 5)

    # --- Emoji ---
    elif style in _EMOJI_STYLE_MAP:
        emoji = _EMOJI_STYLE_MAP[style]
        # faint bounding hint
        pen = QPen(QColor(color.red(), color.green(), color.blue(), 50))
        pen.setWidth(1)
        pen.setStyle(Qt.PenStyle.DotLine)
        painter.setPen(pen)
        painter.drawRect(rect)
        _draw_emoji(painter, emoji, cx, cy, min(w, h))

    else:
        pen = QPen(color)
        pen.setWidth(thickness)
        painter.setPen(pen)
        painter.drawRect(rect)

    _draw_animation_fx(painter, state, anim, t, thickness)
    painter.restore()


# ---------------------------------------------------------------------------
# Motion trails
# ---------------------------------------------------------------------------

TRAIL_STYLES: List[str] = ["Solid", "Dash", "Dot", "Fade"]
TRAIL_COLOR_MODES: List[str] = ["Match Box", "Custom", "Speed"]

MOTION_TRAIL_DEFAULTS: Dict[str, object] = {
    "trails": False,
    "trail_length": 20,
    "trail_width": 2,
    "trail_style": "Solid",
    "trail_color_mode": "Match Box",
    "trail_color": None,
    "trail_opacity": 180,
    "trail_fade": True,
}


def normalize_motion_trail_settings(motion_settings: Optional[dict]) -> dict:
    """Merge motion_settings with trail defaults and clamp values."""
    ms = dict(motion_settings or {})
    out = dict(MOTION_TRAIL_DEFAULTS)
    out["trails"] = bool(ms.get("trails", out["trails"]))
    out["trail_length"] = max(5, min(60, int(ms.get("trail_length", out["trail_length"]) or 20)))
    out["trail_width"] = max(1, min(8, int(ms.get("trail_width", out["trail_width"]) or 2)))
    style = str(ms.get("trail_style") or out["trail_style"])
    out["trail_style"] = style if style in TRAIL_STYLES else "Solid"
    mode = str(ms.get("trail_color_mode") or out["trail_color_mode"])
    out["trail_color_mode"] = mode if mode in TRAIL_COLOR_MODES else "Match Box"
    tc = ms.get("trail_color")
    if isinstance(tc, QColor):
        out["trail_color"] = tc
    elif isinstance(tc, str) and tc:
        c = QColor(tc)
        out["trail_color"] = c if c.isValid() else None
    else:
        out["trail_color"] = None
    out["trail_opacity"] = max(20, min(255, int(ms.get("trail_opacity", out["trail_opacity"]) or 180)))
    out["trail_fade"] = bool(ms.get("trail_fade", out["trail_fade"]))
    return out


def motion_trail_history_limit(motion_settings: Optional[dict]) -> int:
    ts = normalize_motion_trail_settings(motion_settings)
    if ts["trails"]:
        return int(ts["trail_length"])
    return 30


def _trail_segment_color(
    color_mode: str,
    base_color: QColor,
    trail_settings: dict,
    obj: dict,
    progress: float,
) -> QColor:
    """progress 0..1 along trail (older → newer)."""
    if color_mode == "Custom":
        c = trail_settings.get("trail_color")
        if isinstance(c, QColor) and c.isValid():
            return QColor(c)
    if color_mode == "Speed":
        sp = min(float(obj.get("speed", 0) or 0), 20.0)
        hue = 120.0 - (sp / 20.0 * 120.0)
        c = QColor()
        c.setHslF(hue / 360.0, 1.0, 0.5)
        return c
    return QColor(base_color)


def draw_motion_trail(
    painter: QPainter,
    widget_points: List[QPointF],
    motion_settings: dict,
    obj: dict,
    base_color: QColor,
) -> None:
    """Draw centroid path trail in widget coordinates."""
    ts = normalize_motion_trail_settings(motion_settings)
    if not ts["trails"] or len(widget_points) < 2:
        return

    length = int(ts["trail_length"])
    pts = list(widget_points[-length:])
    if len(pts) < 2:
        return

    width = int(ts["trail_width"])
    style = str(ts["trail_style"])
    opacity = int(ts["trail_opacity"])
    fade = bool(ts["trail_fade"]) or style == "Fade"
    color_mode = str(ts["trail_color_mode"])

    painter.save()
    painter.setBrush(Qt.BrushStyle.NoBrush)

    n_seg = len(pts) - 1
    if fade and n_seg >= 1:
        for i in range(n_seg):
            progress = (i + 1) / max(1, n_seg)
            alpha = int(opacity * (0.12 + 0.88 * progress))
            c = _trail_segment_color(color_mode, base_color, ts, obj, progress)
            c.setAlpha(max(0, min(255, alpha)))
            pen = QPen(c)
            pen.setWidth(width)
            pen.setCapStyle(Qt.PenCapStyle.RoundCap)
            if style == "Dash":
                pen.setStyle(Qt.PenStyle.DashLine)
            elif style == "Dot":
                pen.setStyle(Qt.PenStyle.DotLine)
            else:
                pen.setStyle(Qt.PenStyle.SolidLine)
            painter.setPen(pen)
            painter.drawLine(pts[i], pts[i + 1])
    else:
        c = _trail_segment_color(color_mode, base_color, ts, obj, 1.0)
        c.setAlpha(opacity)
        pen = QPen(c)
        pen.setWidth(width)
        pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        if style == "Dash":
            pen.setStyle(Qt.PenStyle.DashLine)
        elif style == "Dot":
            pen.setStyle(Qt.PenStyle.DotLine)
        painter.setPen(pen)
        painter.drawPolyline(pts)

    # Head marker on newest point
    head = pts[-1]
    hc = _trail_segment_color(color_mode, base_color, ts, obj, 1.0)
    hc.setAlpha(min(255, opacity + 40))
    painter.setPen(Qt.PenStyle.NoPen)
    painter.setBrush(hc)
    r = max(2.0, width * 0.9)
    painter.drawEllipse(head, r, r)
    painter.restore()


# ---------------------------------------------------------------------------
# Picker icons + labels (settings UI)
# ---------------------------------------------------------------------------

_ICON_BG = QColor(24, 28, 36)
_ICON_ACCENT = QColor(255, 72, 72)
_ICON_PREVIEW_T = 0.65

_style_icon_cache: Dict[str, "QPixmap"] = {}
_anim_icon_cache: Dict[str, "QPixmap"] = {}


def motion_style_short_label(style: str) -> str:
    style = normalize_motion_style(style)
    if style in _EMOJI_STYLE_MAP:
        return _EMOJI_STYLE_MAP[style]
    short_map = {
        "Double Box": "Double",
        "Cross Grid": "Grid",
        "Threat Mark": "Threat",
        "NATO Bracket": "NATO",
        "Range Ring": "Rings",
        "Vector Track": "Vector",
        "Lock Box": "Lock",
        "Cross Dot": "X Dot",
        "Sparkle Corner": "Spark",
        "Gradient Fill": "Grad",
        "Party Frame": "Party",
        "Crosshair": "Cross",
        "Underline": "Line",
    }
    return short_map.get(style, style if len(style) <= 10 else style.split()[0])


def motion_animation_short_label(anim: str) -> str:
    anim = normalize_motion_animation(anim)
    short_map = {
        "Neon Pulse": "Neon",
        "Aurora Shift": "Aurora",
        "Tactical Lock": "Lock",
        "Target Ping": "Ping",
    }
    return short_map.get(anim, anim if len(anim) <= 9 else anim.split()[0])


def motion_style_icon_pixmap(style: str, size: int = 36) -> "QPixmap":
    from PySide6.QtGui import QPixmap

    style = normalize_motion_style(style)
    key = f"{style}:{size}"
    cached = _style_icon_cache.get(key)
    if cached is not None and not cached.isNull():
        return cached

    pm = QPixmap(size, size)
    pm.fill(_ICON_BG)
    p = QPainter(pm)
    p.setRenderHint(QPainter.RenderHint.Antialiasing, True)

    pad = max(4, size // 8)
    w = h = float(size - pad * 2)
    x = y = float(pad)
    cx = x + w / 2
    cy = y + h / 2
    state = MotionAnimState(
        color=_ICON_ACCENT,
        draw_x=x,
        draw_y=y,
        draw_w=w,
        draw_h=h,
        cx=cx,
        cy=cy,
    )
    obj = {"history": [(cx - 4, cy), (cx + 4, cy)], "speed": 6}
    draw_motion_box_style(
        p, style, state, max(1, size // 16), _ICON_PREVIEW_T, obj, anim="None",
    )
    p.end()
    _style_icon_cache[key] = pm
    return pm


def motion_animation_icon_pixmap(anim: str, size: int = 36) -> "QPixmap":
    from PySide6.QtGui import QPixmap

    anim = normalize_motion_animation(anim)
    key = f"{anim}:{size}"
    cached = _anim_icon_cache.get(key)
    if cached is not None and not cached.isNull():
        return cached

    pm = QPixmap(size, size)
    pm.fill(_ICON_BG)
    p = QPainter(pm)
    p.setRenderHint(QPainter.RenderHint.Antialiasing, True)

    pad = max(5, size // 7)
    x, y = float(pad), float(pad)
    w = h = float(size - pad * 2)
    cx, cy = x + w / 2, y + h / 2
    t = _ICON_PREVIEW_T

    # Base motion box
    pen = QPen(QColor(120, 130, 150))
    pen.setWidth(max(1, size // 18))
    p.setPen(pen)
    p.setBrush(Qt.BrushStyle.NoBrush)
    p.drawRect(QRectF(x, y, w, h))

    accent = _ICON_ACCENT
    if anim == "None":
        p.setPen(QPen(QColor(90, 98, 110), 1, Qt.PenStyle.DashLine))
        p.drawLine(int(x + 2), int(y + 2), int(x + w - 2), int(y + h - 2))
    elif anim in ("Pulse", "Breathe", "Neon Pulse", "Target Ping"):
        rings = 3 if anim == "Target Ping" else 2
        for i in range(rings):
            inset = (i + 1) * (size // 10)
            c = QColor(accent)
            c.setAlpha(180 - i * 50)
            p.setPen(QPen(c, 1))
            p.drawRect(QRectF(x - inset / 2, y - inset / 2, w + inset, h + inset))
    elif anim in ("Flash", "Strobe", "Flicker"):
        p.fillRect(QRectF(x, y, w, h), QColor(accent.red(), accent.green(), accent.blue(), 100))
    elif anim == "Glitch":
        p.setPen(QPen(QColor(0, 255, 255), 2))
        p.drawRect(QRectF(x + 2, y - 1, w - 1, h))
        p.setPen(QPen(QColor(255, 0, 255), 1))
        p.drawRect(QRectF(x - 1, y + 2, w, h - 2))
    elif anim in ("Rainbow", "Aurora Shift"):
        for i, hue in enumerate((0, 80, 160)):
            c = QColor.fromHslF((hue / 360) % 1.0, 0.85, 0.55)
            p.setPen(QPen(c, 2))
            p.drawRect(QRectF(x + i, y + i, w - i * 2, h - i * 2))
    elif anim in ("Glow", "Shimmer"):
        for i in range(3):
            c = QColor(accent.red(), accent.green(), accent.blue(), 60 + i * 30)
            p.setPen(QPen(c, 2 + i))
            p.drawRect(QRectF(x, y, w, h))
    elif anim == "Radar":
        p.setPen(QPen(accent, 2))
        ang = t * 6.28
        p.drawLine(QPointF(cx, cy), QPointF(cx + math.cos(ang) * w * 0.45, cy + math.sin(ang) * h * 0.45))
    elif anim == "Scan":
        p.setPen(QPen(QColor(0, 255, 200), 2))
        sy = y + h * t
        p.drawLine(int(x), int(sy), int(x + w), int(sy))
    elif anim == "Matrix":
        font = QFont("monospace")
        font.setPixelSize(max(7, size // 5))
        p.setFont(font)
        p.setPen(QColor(0, 255, 90))
        p.drawText(QPointF(x + 2, y + h * 0.45), "01")
        p.drawText(QPointF(x + w * 0.35, y + h * 0.75), "10")
    elif anim == "Orbit":
        p.setBrush(accent)
        p.setPen(Qt.PenStyle.NoPen)
        ang = t * 6.28
        ox = cx + math.cos(ang) * w * 0.38
        oy = cy + math.sin(ang) * h * 0.38
        p.drawEllipse(QPointF(ox, oy), 2.5, 2.5)
    elif anim in ("Shake", "Wiggle", "Dizzy"):
        p.setPen(QPen(accent, 2))
        p.drawRect(QRectF(x + 2, y - 1, w - 4, h + 1))
        p.drawRect(QRectF(x - 1, y + 1, w + 1, h - 2))
    elif anim == "Bounce":
        p.setPen(QPen(accent, 2))
        p.drawRect(QRectF(x, y - 3, w, h))
    elif anim == "Float":
        p.setPen(QPen(accent, 2))
        p.drawRect(QRectF(x, y - 2, w, h))
    elif anim == "Drift":
        p.setPen(QPen(accent, 2))
        p.drawRect(QRectF(x + 2, y, w, h))
    elif anim == "Party":
        for i, (px, py) in enumerate(((x, y), (x + w, y), (x, y + h), (x + w, y + h))):
            c = QColor.fromHslF(((i * 90) % 360) / 360.0, 0.9, 0.55)
            p.setBrush(c)
            p.setPen(Qt.PenStyle.NoPen)
            p.drawEllipse(QPointF(px, py), 2.5, 2.5)
    elif anim in ("Sparkle",):
        p.setPen(QPen(accent, 2))
        p.drawLine(QPointF(cx - 4, cy), QPointF(cx + 4, cy))
        p.drawLine(QPointF(cx, cy - 4), QPointF(cx, cy + 4))
    elif anim == "Tactical Lock":
        inset = w * 0.15
        p.setPen(QPen(accent, 2))
        p.drawRect(QRectF(x + inset, y + inset, w - inset * 2, h - inset * 2))
    elif anim == "Fade":
        c = QColor(accent)
        c.setAlpha(100)
        p.fillRect(QRectF(x, y, w, h), c)
    else:
        state = apply_motion_animation(
            anim, t, accent, x, y, w, h, cx, cy, {"history": [], "speed": 0},
        )
        p.setPen(QPen(state.color, 2))
        p.drawRect(QRectF(state.draw_x, state.draw_y, state.draw_w, state.draw_h))

    p.end()
    _anim_icon_cache[key] = pm
    return pm

