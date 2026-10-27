import math
import time
from typing import Dict, List, Optional, Tuple

from PySide6.QtCore import Qt, QTimer, QPointF
from PySide6.QtGui import QColor, QPainter, QPen, QPolygonF, QFont
from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QLabel,
    QDialog,
    QFormLayout,
    QSlider,
    QCheckBox,
    QPushButton,
    QColorDialog,
    QComboBox,
    QHBoxLayout,
    QMenu,
)
from PySide6.QtGui import QPainterPath
from PySide6.QtCore import QRectF

from desktop.widgets.base import BaseDesktopWidget
from desktop.utils.qt_helpers import KnoxnetStyle


def _clamp01(v: float) -> float:
    return max(0.0, min(1.0, v))


class OverlaySettingsDialog(QDialog):
    """Lightweight settings panel for the overlay window."""

    def __init__(self, parent=None, initial=None):
        super().__init__(parent)
        self.setWindowTitle("Overlay Settings")
        self.setWindowFlags(Qt.WindowType.Tool | Qt.WindowType.WindowStaysOnTopHint)
        self.initial = initial or {}
        self.result_settings = dict(self.initial)

        layout = QVBoxLayout(self)
        form = QFormLayout()

        # Opacity
        self.opacity_slider = QSlider(Qt.Orientation.Horizontal)
        self.opacity_slider.setRange(30, 100)
        self.opacity_slider.setValue(int(self.initial.get("opacity", 0.85) * 100))
        form.addRow("Opacity", self.opacity_slider)

        # Scale
        self.scale_slider = QSlider(Qt.Orientation.Horizontal)
        self.scale_slider.setRange(50, 200)
        self.scale_slider.setValue(int(self.initial.get("scale", 1.0) * 100))
        form.addRow("Scale", self.scale_slider)

        # Rotation
        self.rotation_slider = QSlider(Qt.Orientation.Horizontal)
        self.rotation_slider.setRange(-180, 180)
        self.rotation_slider.setValue(int(self.initial.get("rotation_deg", 0)))
        form.addRow("Rotation (deg)", self.rotation_slider)

        # Lock aspect
        self.lock_aspect = QCheckBox("Lock aspect to camera frame")
        self.lock_aspect.setChecked(self.initial.get("lock_aspect", True))
        form.addRow(self.lock_aspect)

        # Show labels
        self.show_labels = QCheckBox("Show labels")
        self.show_labels.setChecked(self.initial.get("show_labels", True))
        form.addRow(self.show_labels)

        # Always on top
        self.always_on_top = QCheckBox("Keep on top (pin)")
        self.always_on_top.setChecked(self.initial.get("always_on_top", True))
        form.addRow(self.always_on_top)

        # Mouse passthrough
        self.pass_through = QCheckBox("Transparent to mouse (view-only)")
        self.pass_through.setChecked(self.initial.get("pass_through", False))
        form.addRow(self.pass_through)

        # Motion style
        self.motion_combo = QComboBox()
        for opt in ["Glow", "Pulse", "None"]:
            self.motion_combo.addItem(opt)
        cur_style = self.initial.get("motion_style", "Glow")
        idx = self.motion_combo.findText(cur_style)
        if idx >= 0:
            self.motion_combo.setCurrentIndex(idx)
        form.addRow("Motion interaction", self.motion_combo)

        # Poll interval
        self.poll_slider = QSlider(Qt.Orientation.Horizontal)
        self.poll_slider.setRange(15, 200)
        self.poll_slider.setValue(int(self.initial.get("poll_ms", 50)))
        form.addRow("Poll interval (ms)", self.poll_slider)

        # Accent color
        self.color_btn = QPushButton("Pick accent color")
        self.color = QColor(self.initial.get("accent_color", "#24D1FF"))
        self.color_btn.setStyleSheet(f"background-color: {self.color.name()};")
        self.color_btn.clicked.connect(self._pick_color)
        form.addRow(self.color_btn)

        # Zone fill controls (so zones don't block the view during snapshots)
        self.zone_base_fill = QCheckBox("Zones: show base fill (shaded interior)")
        self.zone_base_fill.setChecked(bool(self.initial.get("zone_base_fill", False)))
        form.addRow(self.zone_base_fill)

        self.zone_interaction_fill_slider = QSlider(Qt.Orientation.Horizontal)
        self.zone_interaction_fill_slider.setRange(0, 60)  # percent of max alpha
        self.zone_interaction_fill_slider.setValue(int(float(self.initial.get("zone_interaction_fill_pct", 18))))
        form.addRow("Zones: interaction fill (%)", self.zone_interaction_fill_slider)

        layout.addLayout(form)

        btn_row = QHBoxLayout()
        save_btn = QPushButton("Save")
        save_btn.clicked.connect(self._accept)
        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)
        btn_row.addStretch()
        btn_row.addWidget(save_btn)
        btn_row.addWidget(cancel_btn)
        layout.addLayout(btn_row)

    def _pick_color(self):
        c = QColorDialog.getColor(self.color, self, "Select Accent Color")
        if c.isValid():
            self.color = c
            self.color_btn.setStyleSheet(f"background-color: {c.name()};")

    def _accept(self):
        self.result_settings = {
            "opacity": self.opacity_slider.value() / 100.0,
            "scale": self.scale_slider.value() / 100.0,
            "rotation_deg": self.rotation_slider.value(),
            "lock_aspect": self.lock_aspect.isChecked(),
            "show_labels": self.show_labels.isChecked(),
            "always_on_top": self.always_on_top.isChecked(),
            "pass_through": self.pass_through.isChecked(),
            "motion_style": self.motion_combo.currentText(),
            "poll_ms": self.poll_slider.value(),
            "accent_color": self.color.name(),
            "zone_base_fill": self.zone_base_fill.isChecked(),
            "zone_interaction_fill_pct": int(self.zone_interaction_fill_slider.value()),
        }
        self.accept()


class CameraOverlayCanvas(QWidget):
    """Transparent canvas that renders camera overlay shapes + motion."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.snapshot: Optional[dict] = None
        self.settings: Dict[str, object] = {
            "opacity": 0.85,
            "scale": 1.0,
            "rotation_deg": 0,
            "lock_aspect": True,
            "show_labels": True,
            "motion_style": "Glow",
            "accent_color": "#24D1FF",
            # Zone fill defaults: keep interiors clear so we don't hide activity.
            "zone_base_fill": False,
            # Max alpha (%) for interaction fill pulses. Lower = more transparent.
            "zone_interaction_fill_pct": 18,
        }
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self.setAttribute(Qt.WidgetAttribute.WA_OpaquePaintEvent, False)
        self.setAutoFillBackground(False)
        self._selection_paths: Dict[str, QPainterPath] = {}

    def update_snapshot(self, snap: Optional[dict], settings: Dict[str, object]):
        self.snapshot = snap
        if settings:
            self.settings.update(settings)
        self._rebuild_selection_paths()
        self.update()

    def _rebuild_selection_paths(self):
        self._selection_paths = {}
        if not self.snapshot:
            return
        for sh in self.snapshot.get("shapes", []):
            if sh.get("kind") == "zone":
                pts = sh.get("pts") or sh.get("points") or []
                if len(pts) < 3:
                    continue
                path = QPainterPath()
                first = pts[0]
                path.moveTo(first.get("x", 0), first.get("y", 0))
                for p in pts[1:]:
                    path.lineTo(p.get("x", 0), p.get("y", 0))
                path.closeSubpath()
                self._selection_paths[sh.get("id")] = path

    def _map_rect(self, frame_dims: Tuple[int, int]) -> Tuple[float, float, float, float]:
        """Return x_off, y_off, w, h for drawing area respecting aspect if enabled."""
        fw, fh = frame_dims or (0, 0)
        if fw <= 0 or fh <= 0 or not self.settings.get("lock_aspect", True):
            return 0.0, 0.0, float(self.width()), float(self.height())
        widget_w = float(self.width())
        widget_h = float(self.height())
        target_w = widget_w
        target_h = widget_w * (fh / fw)
        if target_h > widget_h:
            target_h = widget_h
            target_w = widget_h * (fw / fh)
        x_off = (widget_w - target_w) / 2.0
        y_off = (widget_h - target_h) / 2.0
        return x_off, y_off, target_w, target_h

    def _apply_scale(self, x: float, y: float, scale: float) -> Tuple[float, float]:
        cx, cy = 0.5, 0.5
        return cx + (x - cx) * scale, cy + (y - cy) * scale

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        # Use explicit enum namespaces for compatibility (PySide6 / Qt6).
        painter.setCompositionMode(QPainter.CompositionMode.CompositionMode_SourceOver)
        painter.setOpacity(1.0)
        painter.fillRect(self.rect(), Qt.GlobalColor.transparent)

        if not self.snapshot:
            painter.setPen(QColor("#FFFFFF"))
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "No camera overlay data")
            return

        snap = self.snapshot
        now = time.time()
        fw, fh = snap.get("frame_dims", (0, 0))
        x_off, y_off, draw_w, draw_h = self._map_rect((fw, fh))
        scale = float(self.settings.get("scale", 1.0))
        rot_rad = math.radians(float(self.settings.get("rotation_deg", 0)))
        show_labels = bool(self.settings.get("show_labels", True)) and snap.get("show_shape_labels", True)
        motion_style = self.settings.get("motion_style", "Glow")
        accent = QColor(self.settings.get("accent_color", "#24D1FF"))
        zone_base_fill = bool(self.settings.get("zone_base_fill", False))
        try:
            zone_interaction_fill_pct = max(0, min(60, int(self.settings.get("zone_interaction_fill_pct", 18))))
        except Exception:
            zone_interaction_fill_pct = 18
        zone_interaction_alpha_max = int(255 * (zone_interaction_fill_pct / 100.0))

        def to_canvas(nx: float, ny: float) -> QPointF:
            sx, sy = self._apply_scale(nx, ny, scale)
            px = x_off + sx * draw_w
            py = y_off + sy * draw_h
            # rotation around center of draw area
            cx = x_off + draw_w / 2.0
            cy = y_off + draw_h / 2.0
            dx, dy = px - cx, py - cy
            rx = dx * math.cos(rot_rad) - dy * math.sin(rot_rad) + cx
            ry = dx * math.sin(rot_rad) + dy * math.cos(rot_rad) + cy
            return QPointF(rx, ry)

        painter.save()
        # Zones
        for sh in snap.get("shapes", []):
            if sh.get("kind") != "zone" or sh.get("hidden"):
                continue
            pts = sh.get("pts") or sh.get("points") or []
            if len(pts) < 3:
                continue
            color = QColor(str(sh.get("color", accent.name())))
            alpha = float(sh.get("alpha", 0.65))
            poly = QPolygonF([to_canvas(float(p.get("x", 0)), float(p.get("y", 0))) for p in pts])

            # Base fill is optional; default is transparent so overlays don't block snapshots.
            if zone_base_fill:
                fill = QColor(color)
                # Preserve per-shape alpha but keep it subtle.
                fill.setAlpha(int(max(0, min(90, alpha * 60))))
                painter.setBrush(fill)
            else:
                painter.setBrush(Qt.BrushStyle.NoBrush)
            pen = QPen(color)
            pen.setWidth(3)
            pen.setColor(color)
            pen.setCapStyle(Qt.PenCapStyle.RoundCap)
            pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
            painter.setPen(pen)
            painter.drawPolygon(poly)

            # Interaction pulse from motion hits
            pulse_ts = snap.get("zone_pulses", {}).get(sh.get("id"))
            if pulse_ts:
                age = now - pulse_ts
                if age < 0.8:
                    interaction_color = sh.get("interaction_color") or accent
                    try:
                        pulse_col = QColor(interaction_color)
                        if not pulse_col.isValid():
                            pulse_col = QColor(accent)
                    except Exception:
                        pulse_col = QColor(accent)
                    if motion_style == "Pulse":
                        # Use a configurable, transparent interaction fill so we don't hide activity.
                        pulse_col.setAlpha(int(zone_interaction_alpha_max * (1.0 - age / 0.8)))
                        painter.setBrush(pulse_col)
                        painter.setPen(Qt.PenStyle.NoPen)
                        painter.drawPolygon(poly)
                    elif motion_style == "Glow":
                        glow_pen = QPen(pulse_col)
                        glow_pen.setWidth(6)
                        alpha_val = int(zone_interaction_alpha_max * (1.0 - age / 0.8))
                        glow_pen.setColor(QColor(pulse_col.red(), pulse_col.green(), pulse_col.blue(), alpha_val))
                        painter.setPen(glow_pen)
                        painter.setBrush(Qt.BrushStyle.NoBrush)
                        painter.drawPolygon(poly)

            if show_labels and sh.get("show_label", True):
                cx = sum(float(p.get("x", 0)) for p in pts) / len(pts)
                cy = sum(float(p.get("y", 0)) for p in pts) / len(pts)
                cpt = to_canvas(cx, cy)
                font = QFont(painter.font())
                font.setPointSize(int(sh.get("text_size", 12)))
                painter.setFont(font)
                painter.setPen(QColor(sh.get("text_color", "#F0F0F0")))
                painter.drawText(cpt, sh.get("label") or "Zone")

        # Lines
        for sh in snap.get("shapes", []):
            if sh.get("kind") != "line" or sh.get("hidden"):
                continue
            p1 = sh.get("p1")
            p2 = sh.get("p2")
            if not p1 or not p2:
                continue
            color = QColor(str(sh.get("color", "#FF4D4D")))
            alpha = float(sh.get("alpha", 0.65))
            pen = QPen(color)
            pen.setWidth(int(max(2, sh.get("line_thickness", 2))))
            pen.setColor(QColor(color.red(), color.green(), color.blue(), int(alpha * 255)))
            pen.setStyle(Qt.PenStyle.SolidLine)
            painter.setPen(pen)
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawLine(to_canvas(p1.get("x", 0), p1.get("y", 0)), to_canvas(p2.get("x", 0), p2.get("y", 0)))

            pulse = snap.get("line_pulses", {}).get(sh.get("id"))
            if pulse:
                age = now - pulse.get("ts", now)
                if age < 0.8:
                    pt = pulse.get("pt", {})
                    cp = to_canvas(pt.get("x", 0), pt.get("y", 0))
                    icol = QColor(sh.get("interaction_color", "#FFD74A"))
                    rad = 12 + age * 40
                    icol.setAlpha(int(255 * (1.0 - age / 0.8)))
                    painter.setBrush(Qt.BrushStyle.NoBrush)
                    painter.setPen(QPen(icol, 3))
                    painter.drawEllipse(cp, rad, rad)

            if show_labels and sh.get("show_label", True):
                mid = to_canvas((p1.get("x", 0) + p2.get("x", 0)) / 2, (p1.get("y", 0) + p2.get("y", 0)) / 2)
                painter.setPen(QColor(sh.get("text_color", "#F0F0F0")))
                painter.drawText(mid + QPointF(6, -6), sh.get("label") or "Line")

        # Tags
        for sh in snap.get("shapes", []):
            if sh.get("kind") != "tag" or sh.get("hidden"):
                continue
            anchor = sh.get("anchor") or {"x": sh.get("x", 0.5), "y": sh.get("y", 0.5)}
            color = QColor(str(sh.get("color", "#00FFC6")))
            alpha = float(sh.get("alpha", 0.7))
            pen = QPen(color)
            pen.setWidth(max(2, int(sh.get("line_thickness", 2))))
            pen.setColor(QColor(color.red(), color.green(), color.blue(), int(alpha * 255)))
            painter.setPen(pen)
            painter.setBrush(Qt.BrushStyle.NoBrush)
            pt = to_canvas(anchor.get("x", 0.5), anchor.get("y", 0.5))
            size = max(8, int(sh.get("tag_size", 18)))
            painter.drawLine(QPointF(pt.x() - size, pt.y()), QPointF(pt.x() + size, pt.y()))
            painter.drawLine(QPointF(pt.x(), pt.y() - size), QPointF(pt.x(), pt.y() + size))

            pulse = snap.get("tag_pulses", {}).get(sh.get("id"))
            if pulse:
                age = now - pulse
                if age < 0.8:
                    icol = QColor(sh.get("interaction_color", "#FFD74A"))
                    icol.setAlpha(int(255 * (1.0 - age / 0.8)))
                    painter.setPen(QPen(icol, 3))
                    painter.setBrush(Qt.BrushStyle.NoBrush)
                    painter.drawEllipse(pt, size * 0.8 + age * 20, size * 0.8 + age * 20)

            if show_labels and sh.get("show_label", True):
                painter.setPen(QColor(sh.get("text_color", "#F0F0F0")))
                painter.drawText(pt + QPointF(10, -4), sh.get("label") or sh.get("name") or "Tag")

        # Motion boxes (normalized from source dims)
        boxes = snap.get("motion_boxes", [])
        motion_col = QColor(self.settings.get("accent_color", "#24D1FF"))
        motion_col.setAlpha(120)
        pen = QPen(motion_col)
        pen.setWidth(2)
        painter.setPen(pen)
        painter.setBrush(Qt.BrushStyle.NoBrush)

        def _point_in_polygon(x: float, y: float, poly_pts: List[Tuple[float, float]]) -> bool:
            inside = False
            j = len(poly_pts) - 1
            for i in range(len(poly_pts)):
                xi, yi = poly_pts[i]
                xj, yj = poly_pts[j]
                if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / ((yj - yi) or 1e-6) + xi):
                    inside = not inside
                j = i
            return inside

        def _dist_to_segment(px: float, py: float, ax: float, ay: float, bx: float, by: float) -> float:
            abx, aby = bx - ax, by - ay
            ab_len_sq = abx * abx + aby * aby
            if ab_len_sq <= 0:
                dx, dy = px - ax, py - ay
                return (dx * dx + dy * dy) ** 0.5
            t = ((px - ax) * abx + (py - ay) * aby) / ab_len_sq
            t = _clamp01(t)
            projx = ax + t * abx
            projy = ay + t * aby
            dx, dy = px - projx, py - projy
            return (dx * dx + dy * dy) ** 0.5

        shapes = snap.get("shapes", [])

        for bx, by, bw, bh in boxes:
            if fw <= 0 or fh <= 0:
                continue
            nx = _clamp01(bx / float(fw))
            ny = _clamp01(by / float(fh))
            nw = _clamp01(bw / float(fw))
            nh = _clamp01(bh / float(fh))
            p1 = to_canvas(nx, ny)
            p2 = to_canvas(nx + nw, ny + nh)
            rect_x = min(p1.x(), p2.x())
            rect_y = min(p1.y(), p2.y())
            rect_w = abs(p2.x() - p1.x())
            rect_h = abs(p2.y() - p1.y())

            # Skip boxes that do not intersect selected shapes when a target is set
            if shapes:
                hit = False
                for sh in shapes:
                    kind = sh.get("kind")
                    sid = sh.get("id")
                    if kind == "zone":
                        pts = sh.get("pts") or sh.get("points") or []
                        if len(pts) < 3:
                            continue
                        cx = nx + nw / 2.0
                        cy = ny + nh / 2.0
                        if _point_in_polygon(cx, cy, [(float(p.get("x", 0)), float(p.get("y", 0))) for p in pts]):
                            hit = True
                            break
                    elif kind == "line":
                        p1n = sh.get("p1") or {}
                        p2n = sh.get("p2") or {}
                        if p1n and p2n:
                            cx = nx + nw / 2.0
                            cy = ny + nh / 2.0
                            if _dist_to_segment(cx, cy, float(p1n.get("x", 0)), float(p1n.get("y", 0)), float(p2n.get("x", 0)), float(p2n.get("y", 0))) < 0.03:
                                hit = True
                                break
                    elif kind == "tag":
                        anchor = sh.get("anchor")
                        if anchor:
                            cx = nx + nw / 2.0
                            cy = ny + nh / 2.0
                            dx = cx - float(anchor.get("x", 0.5))
                            dy = cy - float(anchor.get("y", 0.5))
                            if (dx * dx + dy * dy) ** 0.5 < 0.04:
                                hit = True
                                break
                if not hit:
                    continue

            rect = QRectF(rect_x, rect_y, rect_w, rect_h)

            if motion_style == "Glow":
                glow = QPen(QColor(motion_col.red(), motion_col.green(), motion_col.blue(), 180))
                glow.setWidth(4)
                painter.setPen(glow)
                painter.drawRect(rect)
                painter.setPen(pen)
            elif motion_style == "Pulse":
                age = (now * 1000) % 600 / 600.0
                alpha = int(200 * abs(math.sin(age * math.pi)))
                pp = QPen(QColor(motion_col.red(), motion_col.green(), motion_col.blue(), alpha))
                pp.setWidth(3)
                painter.setPen(pp)
            # Clip motion to zone polygon if available for a "traverse through zone" effect
            clip_applied = False
            if shapes:
                for sh in shapes:
                    if sh.get("kind") == "zone" and sh.get("id") in self._selection_paths:
                        path = self._selection_paths[sh.get("id")]
                        # Build path in canvas space
                        poly_path = QPainterPath()
                        pts = sh.get("pts") or sh.get("points") or []
                        if pts:
                            first = pts[0]
                            poly_path.moveTo(to_canvas(first.get("x", 0), first.get("y", 0)))
                            for p in pts[1:]:
                                poly_path.lineTo(to_canvas(p.get("x", 0), p.get("y", 0)))
                            poly_path.closeSubpath()
                            painter.save()
                            painter.setClipPath(poly_path)
                            clip_applied = True
                            break

            painter.drawRect(rect)
            if clip_applied:
                painter.restore()

        painter.restore()


class CameraOverlayWindow(BaseDesktopWidget):
    """
    Transparent, frameless overlay that mirrors the React camera overlay widget.
    Polls the existing CameraWidget for motion + shape data with minimal overhead.
    """

    def __init__(self, camera_widget, target_ids: Optional[List[str]] = None):
        super().__init__(title=f"Overlay: {camera_widget.camera_id}", width=520, height=320)
        self.camera_widget = camera_widget
        self.target_ids: List[str] = target_ids or []
        self.settings = {
            "opacity": 0.85,
            "scale": 1.0,
            "rotation_deg": 0,
            "lock_aspect": True,
            "show_labels": True,
            "always_on_top": True,
            "pass_through": False,
            "motion_style": "Glow",
            "poll_ms": 50,
            "accent_color": "#24D1FF",
            # Zone fill behavior
            "zone_base_fill": False,
            "zone_interaction_fill_pct": 18,
        }

        self.setWindowFlags(self.windowFlags() | Qt.WindowType.FramelessWindowHint | Qt.WindowType.Tool | Qt.WindowType.WindowStaysOnTopHint)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self.setAttribute(Qt.WidgetAttribute.WA_OpaquePaintEvent, False)
        self.setAutoFillBackground(False)
        self.setWindowOpacity(self.settings["opacity"])
        self.apply_knox_style()

        self.canvas = CameraOverlayCanvas(self)
        self.set_content(self.canvas)
        self.title_bar.hide()
        self.main_layout.setContentsMargins(0, 0, 0, 0)
        self.content_layout.setContentsMargins(0, 0, 0, 0)

        # Poll loop
        self.poll_timer = QTimer(self)
        self.poll_timer.timeout.connect(self._pull_snapshot)
        self.poll_timer.start(self.settings["poll_ms"])
        self._drag_pos = None

    def apply_knox_style(self):
        self.central_widget.setStyleSheet("""
            QWidget#Central {
                background: transparent;
                border: none;
            }
        """)

    def _pull_snapshot(self):
        snap = self.camera_widget.get_overlay_snapshot() if self.camera_widget else None
        if snap:
            if self.target_ids:
                snap = self._filter_snapshot(snap, self.target_ids)
            self.canvas.update_snapshot(snap, self.settings)

    def _filter_snapshot(self, snap: dict, ids: List[str]) -> dict:
        allowed = set(ids or [])
        filtered_shapes = [s for s in snap.get("shapes", []) if s.get("id") in allowed]
        zp = {k: v for k, v in (snap.get("zone_pulses") or {}).items() if k in allowed}
        lp = {k: v for k, v in (snap.get("line_pulses") or {}).items() if k in allowed}
        tp = {k: v for k, v in (snap.get("tag_pulses") or {}).items() if k in allowed}
        selected = [sid for sid in snap.get("selected_shapes", []) if sid in allowed]
        return {
            **snap,
            "shapes": filtered_shapes,
            "zone_pulses": zp,
            "line_pulses": lp,
            "tag_pulses": tp,
            "selected_shapes": selected,
        }

    def set_target_ids(self, ids: List[str]):
        self.target_ids = ids or []
        self._pull_snapshot()

    def open_settings_dialog(self):
        dialog = OverlaySettingsDialog(self, initial=self.settings)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            self.settings.update(dialog.result_settings)
            self.setWindowOpacity(self.settings["opacity"])
            self._apply_on_top(self.settings["always_on_top"])
            self._apply_passthrough(self.settings["pass_through"])
            # Update poll rate
            self.poll_timer.stop()
            self.poll_timer.start(max(15, int(self.settings.get("poll_ms", 50))))
            self._pull_snapshot()

    def _apply_on_top(self, enabled: bool):
        flags = self.windowFlags()
        if enabled:
            flags |= Qt.WindowType.WindowStaysOnTopHint
        else:
            flags &= ~Qt.WindowType.WindowStaysOnTopHint
        self.setWindowFlags(flags)
        self.show()

    def _apply_passthrough(self, enabled: bool):
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, enabled)

    def contextMenuEvent(self, event):
        menu = QMenu(self)
        settings_action = menu.addAction("Settings…")
        labels_action = menu.addAction("Toggle Labels")
        labels_action.setCheckable(True)
        labels_action.setChecked(self.settings.get("show_labels", True))
        lock_ar_action = menu.addAction("Lock Aspect")
        lock_ar_action.setCheckable(True)
        lock_ar_action.setChecked(self.settings.get("lock_aspect", True))
        passthrough_action = menu.addAction("Mouse Passthrough")
        passthrough_action.setCheckable(True)
        passthrough_action.setChecked(self.settings.get("pass_through", False))
        pin_action = menu.addAction("Pin (Always on Top)")
        pin_action.setCheckable(True)
        pin_action.setChecked(self.settings.get("always_on_top", True))
        close_action = menu.addAction("Close Overlay")

        action = menu.exec(event.globalPos())
        if action == settings_action:
            self.open_settings_dialog()
        elif action == labels_action:
            self.settings["show_labels"] = not self.settings.get("show_labels", True)
            self._pull_snapshot()
        elif action == lock_ar_action:
            self.settings["lock_aspect"] = not self.settings.get("lock_aspect", True)
            self._pull_snapshot()
        elif action == passthrough_action:
            self.settings["pass_through"] = not self.settings.get("pass_through", False)
            self._apply_passthrough(self.settings["pass_through"])
        elif action == pin_action:
            self.settings["always_on_top"] = not self.settings.get("always_on_top", True)
            self._apply_on_top(self.settings["always_on_top"])
        elif action == close_action:
            self.close()

    def mousePressEvent(self, event):
        if self.settings.get("pass_through"):
            return super().mousePressEvent(event)
        if event.button() == Qt.MouseButton.LeftButton:
            self._drag_pos = event.globalPosition()
            event.accept()
        else:
            super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self.settings.get("pass_through"):
            return super().mouseMoveEvent(event)
        if self._drag_pos is not None and event.buttons() & Qt.MouseButton.LeftButton:
            delta = event.globalPosition() - self._drag_pos
            self.move(self.pos() + delta.toPoint())
            self._drag_pos = event.globalPosition()
            event.accept()
        else:
            super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        self._drag_pos = None
        super().mouseReleaseEvent(event)

    def closeEvent(self, event):
        if self.poll_timer.isActive():
            self.poll_timer.stop()
        super().closeEvent(event)

