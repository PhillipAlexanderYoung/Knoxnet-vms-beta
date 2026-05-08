import asyncio
import cv2
import json
import base64
import numpy as np
import threading
import random
import requests
import socket
import socketio
import time
from collections import deque
from pathlib import Path
import re as _re
import uuid
from typing import Dict, List, Optional, Tuple
from urllib.parse import urlparse
from PySide6.QtGui import QImage, QPainter, QColor, QAction, QPen, QPolygonF, QFont
from PySide6.QtWidgets import (QVBoxLayout, QLabel, QDialog, QFormLayout, QComboBox, QPushButton, 
                             QSpinBox, QDoubleSpinBox, QColorDialog, QCheckBox, QSlider, QTabWidget, QGroupBox, 
                             QRadioButton, QScrollArea, QWidget, QGridLayout, QHBoxLayout, QMessageBox,
                             QLineEdit, QListWidget, QListWidgetItem, QTableWidget, QTableWidgetItem,
                             QHeaderView, QProgressBar, QFileDialog)
# QOpenGLWidget replaced with QWidget to avoid GLX crashes in PyInstaller bundles.
# The paintGL() method only used QPainter (no raw GL), so QWidget+paintEvent is equivalent.
from PySide6.QtCore import Qt, QTimer, Slot, Signal, QPointF, QRectF, QThread, QBuffer, QIODevice
from desktop.widgets.base import BaseDesktopWidget
from desktop.utils.qt_helpers import KnoxnetStyle
from desktop.utils.depth_worker import DepthAnythingOverlayWorker, DepthOverlayConfig
from desktop.utils.detector_worker import YoloDetectorWorker, DetectorConfig
from desktop.widgets.depth_settings import DepthOverlaySettingsDialog
from desktop.widgets.model_library import ModelLibraryDialog
from desktop.widgets.camera_overlay import CameraOverlayWindow
from desktop.widgets.playback_overlay import PlaybackOverlayWidget
from desktop.widgets.ptz_overlay import (
    PTZOverlayWidget,
    PTZOverlaySettingsDialog,
    PTZTapoCloudPasswordPrompt,
    PTZCredentialsDialog,
    PTZControllerWindow,
    load_ptz_overlay_settings,
    save_ptz_overlay_settings,
)
from desktop.widgets.audio_eq import (
    AudioEQOverlayWidget,
    AudioEQSettings,
    AudioPlayback,
    AudioWHEPReceiver,
    AudioEQWindow,
)

# Shared geometry helpers and color palettes (mirrors React camera widget)
Pt = Dict[str, float]
Shape = Dict[str, object]
ZONE_COLORS = ['#24D1FF', '#7C4DFF', '#00FFA3', '#FF6B6B', '#FFD93D', '#FF5ED1', '#63F5FF', '#9EFF6B']
TAG_COLORS = ['#00FFC6', '#FF9F1C', '#5B8CFF', '#FF4D6D', '#8AFF80', '#FFDF6B', '#9C7BFF', '#40E0D0']
# Presets for motion + detection bounding box colors (fast picker, visible selected state)
# Keep these as *named* choices for better UX than raw hex.
BBOX_COLOR_CHOICES: List[Tuple[str, str]] = [
    ("Red", "#FF3B30"),
    ("Orange", "#FF9500"),
    ("Yellow", "#FFCC00"),
    ("Green", "#34C759"),
    ("Teal", "#00C7BE"),
    ("Cyan", "#24D1FF"),
    ("Blue", "#0A84FF"),
    ("Purple", "#7C4DFF"),
    ("Pink", "#FF5ED1"),
    ("White", "#FFFFFF"),
    ("Black", "#000000"),
]

def _sanitize_zone_dirname(name: str) -> str:
    """Sanitize a shape/zone name for use as a filesystem directory name."""
    s = str(name or "").strip()
    if not s:
        return "_unzoned"
    s = _re.sub(r'[<>:"/\\|?*\x00-\x1f]', '_', s)
    s = s.strip('. ')
    return s[:120] or "_unzoned"


# Persistence path for motion watch settings
def _motion_watch_settings_path() -> Path:
    """
    Persist under the per-user data dir in frozen builds.
    In dev/source, this resolves to <repo>/data/...
    """
    from core.paths import get_data_dir

    return get_data_dir() / "motion_watch_settings.json"


MOTION_WATCH_SETTINGS_PATH = _motion_watch_settings_path()

# Lightweight OUI + port hints for camera discovery (kept small for speed)
CAMERA_OUI_PREFIXES = {
    "44:2C:05": "Hikvision",
    "BC:51:FE": "Hikvision",
    "9C:8E:CD": "Hikvision",
    "64:16:7F": "Axis",
    "00:40:8C": "Axis",
    "24:A4:3C": "Ubiquiti",
    "44:D9:E7": "Ubiquiti",
    "B4:FB:E4": "Reolink",
    "54:62:66": "Reolink",
    "38:AF:29": "Dahua",
    "94:9F:3F": "Dahua",
    "FC:AD:0F": "Amcrest",
    "F0:9F:C2": "Wyze",
    "2C:AA:8E": "Wyze"
}

CAMERA_PORTS = [
    80,    # HTTP
    443,   # HTTPS
    554,   # RTSP
    8554,  # Alternate RTSP
    8000,  # ONVIF / vendor APIs
    8080,  # Alt HTTP
    7447,  # RTSP (some Reolink/UniFi)
    1935   # RTMP (some NVRs)
]


def clamp01(v: float) -> float:
    return max(0.0, min(1.0, v))


def uid(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:8]}"


def point_in_polygon(point: Pt, polygon: List[Pt]) -> bool:
    """Ray casting algorithm to test point within polygon (normalized coords)."""
    if len(polygon) < 3:
        return False
    inside = False
    j = len(polygon) - 1
    for i in range(len(polygon)):
        pi, pj = polygon[i], polygon[j]
        if ((pi['y'] > point['y']) != (pj['y'] > point['y'])):
            x_int = (pj['x'] - pi['x']) * (point['y'] - pi['y']) / ((pj['y'] - pi['y']) or 1e-6) + pi['x']
            if point['x'] < x_int:
                inside = not inside
        j = i
    return inside


def distance_to_line(point: Pt, p1: Pt, p2: Pt) -> float:
    """Distance from point to segment in normalized space."""
    ax, ay = point['x'], point['y']
    bx, by = p1['x'], p1['y']
    cx, cy = p2['x'], p2['y']
    abx, aby = cx - bx, cy - by
    ab_len_sq = abx * abx + aby * aby
    if ab_len_sq == 0:
        dx, dy = ax - bx, ay - by
        return (dx * dx + dy * dy) ** 0.5
    t = ((ax - bx) * abx + (ay - by) * aby) / ab_len_sq
    t = clamp01(t)
    proj_x = bx + t * abx
    proj_y = by + t * aby
    dx, dy = ax - proj_x, ay - proj_y
    return (dx * dx + dy * dy) ** 0.5


def color_from_shape(shape: Shape, fallback: str = '#24D1FF') -> QColor:
    try:
        return QColor(str(shape.get('color', fallback)))
    except Exception:
        return QColor(fallback)

class CameraOpenGLWidget(QWidget):
    """
    Camera video rendering widget. Uses QPainter for 2D rendering of QImage.
    Subclasses QWidget instead of QOpenGLWidget for PyInstaller compatibility.
    """
    shapes_changed = Signal(list)
    shape_triggered = Signal(dict)
    shape_double_clicked = Signal(str)

    def __init__(self, parent=None, camera_id: Optional[str] = None):
        super().__init__(parent)
        self.camera_id = camera_id
        self.image = None
        self.fps = 0
        self.last_frame_count = 0
        self.frame_count = 0
        self.aspect_ratio_mode = Qt.AspectRatioMode.KeepAspectRatioByExpanding

        # Per-widget paint-rate throttle.  When the widget is small (e.g.
        # in a multi-camera grid), we don't need to repaint at full
        # camera FPS -- ~15 fps is visually identical and roughly halves
        # CPU.  Frames arriving above the cap are dropped in update_frame.
        self._last_paint_ts: float = 0.0
        self._small_widget_paint_interval: float = 1.0 / 15.0  # 15 fps cap
        # Width below which we treat the widget as "small" for both
        # paint throttling and FastTransformation.
        self._small_widget_threshold: int = 480

        # When True, the debug overlay paint suppresses the always-on
        # base lines (FPS / Res / Motion / Detections) so only the
        # caller-supplied extra lines are visible.  Used by chord and
        # toggle feedback so the user sees ONLY their action and not
        # a wall of stream stats.
        self._debug_focus_mode: bool = False

        # Auto-protection (load shedder) overrides.  When > 0, these
        # caps apply regardless of widget size and are tighter than the
        # baseline small-widget throttle; the shedder sets them via
        # CameraWidget.apply_shed_level().  0 means "no override".
        self._shed_paint_fps_cap: int = 0
        self._shed_motion_fps_cap: int = 0
        self._last_motion_ts: float = 0.0
        
        # Debug / Motion state
        self.show_debug = False
        self.show_motion = False
        # Depth overlay state (e.g., DepthAnythingV2)
        self.depth_overlay_enabled: bool = False
        self.depth_overlay_opacity: float = 0.55
        self.depth_overlay_image: Optional[QImage] = None
        # When True, the depth overlay becomes the *primary* view (e.g. first-person point cloud),
        # rather than a translucent overlay on top of the video.
        self.depth_overlay_replace_base: bool = False
        # Overlay compositing controls
        self.depth_overlay_scale: float = 1.0
        self.depth_overlay_blackout_base: bool = False
        self.base_camera_opacity: float = 1.0
        self.depth_overlay_is_pointcloud: bool = False
        self._debug_extra_lines: List[str] = []
        self.prev_gray = None
        self.motion_boxes = [] # List of (x, y, w, h) relative to source frame
        self.tracked_objects = {} # { id: {box, history, speed, missing} }
        # Object detections (two independent sources):
        # - backend_detections: Socket.IO (/realtime) detections from backend/React pipeline
        # - desktop_detections: Desktop-local YOLO worker detections
        # When desktop_detections_active is True, we prefer desktop_detections for overlay rendering.
        # NOTE: *_detections are the *filtered* lists used for rendering (may be ROI-filtered).
        self.backend_detections_raw: List[dict] = []
        self.desktop_detections_raw: List[dict] = []
        self.backend_detections: List[dict] = []
        self.desktop_detections: List[dict] = []
        self.desktop_detections_active: bool = False
        # Desktop-local tracking output (stable IDs). Rendered in the same overlay layer as detections.
        self.desktop_tracks_raw: List[dict] = []
        self.desktop_tracks: List[dict] = []
        self.desktop_tracking_active: bool = False
        self.selected_track_id: Optional[int] = None
        # Historical detections from event index during recording playback
        self.playback_detections: List[dict] = []
        # When True, hide zone/line/tag shapes during playback
        self.playback_hide_shapes: bool = False
        # Per-camera per-track overrides (name/color/hide) supplied by CameraWidget
        self.object_overrides: Dict[int, dict] = {}
        self.tracker_next_id = 0
        self.frame_dims = (0, 0) # (w, h)
        self.motion_settings = {
            'style': 'Box', # Box, Fill, Corners, Circle, Bracket, Underline, Crosshair
            'color': QColor(255, 0, 0), # Red
            'thickness': 2,
            'animation': 'None', # None, Pulse, Flash, Glitch, Rainbow
            'trails': False,
            'color_speed': False,
            'sensitivity': 50,
            'merge_size': 0
        }

        # Object detection overlay style settings (separate from motion settings)
        self.detection_settings = {
            'style': 'Box',  # Box, Fill, Corners, Circle, Bracket, Underline, Crosshair
            'color': QColor(0, 255, 255),  # Cyan
            'thickness': 2,
            'animation': 'None',  # None, Pulse, Flash, Glitch, Rainbow, Glow
            'show_labels': True,
            'label_font_size': 10,
            'label_bg_alpha': 150,
            # Shape interaction (zones/lines/tags), mirrors motion box interactions
            'interact_with_shapes': True,
            'interact_zone': True,
            'interact_line': True,
            'interact_tag': True,
            # Interaction distances (normalized 0..1, relative to frame dims)
            'interact_line_dist': 0.02,
            'interact_tag_dist': 0.03,
            # Zone margin expands zones slightly (distance to zone edges) for interaction
            'interact_zone_margin': 0.0,
            # Emit shape_triggered events at most once per cooldown window (ms). Pulses still update.
            'interaction_cooldown_ms': 250,
            # ROI filter: restrict which detections are shown/used based on shapes.
            # If enabled, only detections whose *center* is in/near specified shapes will be kept.
            'roi_enabled': False,
            # If true, only selected shapes are considered ROI; otherwise all shapes.
            'roi_selected_only': False,
            'roi_zone': True,
            'roi_line': True,
            'roi_tag': True,
            # Distances are in normalized units (0..1) relative to frame dims.
            'roi_line_dist': 0.02,
            'roi_tag_dist': 0.03,
            # Zone margin expands zones slightly (distance to zone edges) to include "around zone".
            'roi_zone_margin': 0.0,
            # If enabled, Desktop YOLO inference will run on a cropped region around ROI shapes
            # (zones/lines/tags) instead of the whole frame (faster on large frames).
            # Note: backend detections are unaffected.
            'roi_crop_inference': False,
        }

        # Shapes state (zones/lines/tags) mirrors React implementation
        self.shapes: List[Shape] = []
        self.selected_shapes: List[str] = []
        self.draw_mode: str = 'idle'  # idle|zone|line|tag|drag_vertex|drag_line_end|drag_shape
        self.draw_temp: List[Pt] = []
        self.drag_meta: Optional[Dict[str, object]] = None
        self.show_shape_labels: bool = True
        self.last_draw_rect: Optional[Tuple[float, float, float, float]] = None  # x_off, y_off, w, h in widget coords
        self.zone_pulses: Dict[str, float] = {}
        self.line_pulses: Dict[str, Dict[str, float]] = {}
        self.tag_pulses: Dict[str, float] = {}
        self.cursor_norm: Optional[Pt] = None
        self.hover_shape: Optional[str] = None
        self.motion_hit_pulses: List[Dict[str, object]] = []  # recent motion hit boxes for visual feedback
        self.detection_hit_pulses: List[Dict[str, object]] = []  # recent detection hit boxes for visual feedback
        self._detection_last_emit_ts: float = 0.0

        # FPS Counter
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_fps)
        self.timer.start(1000)
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.FocusPolicy.ClickFocus)

    def set_aspect_ratio_mode(self, mode):
        self.aspect_ratio_mode = mode
        self.update()

    def set_overlay_settings(self, debug=False, motion=False):
        self.show_debug = debug
        self.show_motion = motion
        if not motion:
            self.motion_boxes = []
            self.tracked_objects = {}
            self.prev_gray = None
        self.update()

    def set_depth_overlay(
        self,
        image: Optional[QImage],
        *,
        enabled: Optional[bool] = None,
        opacity: Optional[float] = None,
        replace_base: Optional[bool] = None,
        overlay_scale: Optional[float] = None,
        blackout_base: Optional[bool] = None,
        camera_opacity: Optional[float] = None,
        is_pointcloud: Optional[bool] = None,
    ):
        if enabled is not None:
            self.depth_overlay_enabled = bool(enabled)
        if opacity is not None:
            try:
                self.depth_overlay_opacity = max(0.0, min(1.0, float(opacity)))
            except Exception:
                pass
        if replace_base is not None:
            self.depth_overlay_replace_base = bool(replace_base)
        if overlay_scale is not None:
            try:
                self.depth_overlay_scale = max(0.25, min(3.0, float(overlay_scale)))
            except Exception:
                self.depth_overlay_scale = 1.0
        if blackout_base is not None:
            self.depth_overlay_blackout_base = bool(blackout_base)
        if camera_opacity is not None:
            try:
                self.base_camera_opacity = max(0.0, min(1.0, float(camera_opacity)))
            except Exception:
                self.base_camera_opacity = 1.0
        if is_pointcloud is not None:
            self.depth_overlay_is_pointcloud = bool(is_pointcloud)
        self.depth_overlay_image = image
        self.update()

    def set_debug_extra_lines(self, lines: List[str]):
        """Attach extra debug lines (e.g., model status) to be shown in debug overlay."""
        try:
            self._debug_extra_lines = [str(x) for x in (lines or []) if x is not None and str(x).strip()]
        except Exception:
            self._debug_extra_lines = []
        self.update()

    def set_camera_id(self, camera_id: str):
        self.camera_id = camera_id

    def set_shapes(self, shapes: List[Shape]):
        """Replace current shapes with provided list (normalized coords)."""
        self.shapes = [self._coerce_shape(s) for s in (shapes or [])]
        self.zone_pulses.clear()
        self.line_pulses.clear()
        self.tag_pulses.clear()
        self.motion_hit_pulses.clear()
        self.update()

    def set_selected_shapes(self, ids: List[str]):
        self.selected_shapes = ids or []
        self.update()

    def start_draw_mode(self, mode: str):
        """Begin drawing a new shape (zone|line|tag)."""
        self.draw_mode = mode
        self.draw_temp = []
        self.drag_meta = None
        self.setCursor(Qt.CursorShape.CrossCursor)
        self.setFocus()
        self.update()

    def cancel_draw_mode(self):
        self.draw_mode = 'idle'
        self.draw_temp = []
        self.drag_meta = None
        self.unsetCursor()
        self.update()

    def clear_shapes(self):
        self.shapes = []
        self.selected_shapes = []
        self.draw_temp = []
        self.drag_meta = None
        self.zone_pulses.clear()
        self.line_pulses.clear()
        self.tag_pulses.clear()
        self.motion_hit_pulses.clear()
        self.shapes_changed.emit(self.shapes)
        self.update()

    def _emit_shapes(self):
        self.shapes_changed.emit(self.shapes)

    def _coerce_shape(self, shape: Shape) -> Shape:
        """Ensure shape dict has expected keys and clamped coordinates."""
        kind = shape.get('kind')
        base = {
            'id': shape.get('id') or uid(kind or 'shape'),
            'kind': kind,
            'label': shape.get('label') or shape.get('name') or (f"{kind.title()}"),
            'enabled': shape.get('enabled', True),
            'hidden': shape.get('hidden', False),
            'show_label': shape.get('show_label', True),
            'color': shape.get('color', '#24D1FF'),
            'alpha': shape.get('alpha', 0.65),
            'locked': shape.get('locked', False),
            'interaction_color': shape.get('interaction_color'), # None = default logic
            'interaction_animation': shape.get('interaction_animation', 'Pulse'),
            # New appearance settings
            'line_thickness': float(shape.get('line_thickness', 1.0)),
            'dot_size': float(shape.get('dot_size', 2.0)),
            'tag_size': float(shape.get('tag_size', 20)),
            'tag_style': shape.get('tag_style', 'Crosshair'),
            # Text settings
            'text_size': int(shape.get('text_size', 12)),
            'text_color': shape.get('text_color', '#F0F0F0')
        }
        if kind == 'zone':
            pts = shape.get('pts') or shape.get('points') or []
            base['pts'] = [{'x': clamp01(float(p.get('x', 0))), 'y': clamp01(float(p.get('y', 0)))} for p in pts if isinstance(p, dict)]
        elif kind == 'line':
            p1 = shape.get('p1') or (shape.get('points') or [None, None])[0]
            p2 = shape.get('p2') or (shape.get('points') or [None, None])[1]
            base['p1'] = {'x': clamp01(float(p1.get('x', 0))), 'y': clamp01(float(p1.get('y', 0)))} if p1 else {'x': 0.3, 'y': 0.3}
            base['p2'] = {'x': clamp01(float(p2.get('x', 1))), 'y': clamp01(float(p2.get('y', 1)))} if p2 else {'x': 0.7, 'y': 0.7}
        elif kind == 'tag':
            anchor = shape.get('anchor') or {'x': shape.get('x', 0.5), 'y': shape.get('y', 0.5)}
            base['anchor'] = {'x': clamp01(float(anchor.get('x', 0.5))), 'y': clamp01(float(anchor.get('y', 0.5)))}
        return base

    def _norm_to_widget(self, pt: Pt) -> Optional[Tuple[float, float]]:
        if not self.last_draw_rect:
            return None
        x_off, y_off, w, h = self.last_draw_rect
        return (x_off + pt['x'] * w, y_off + pt['y'] * h)

    def _widget_to_norm(self, pos: QPointF) -> Optional[Pt]:
        # Prefer normalized coords within the video draw rect; if unavailable, fall back to whole widget.
        if not self.last_draw_rect:
            # Fallback: map to full widget area
            w = float(self.width() or 1)
            h = float(self.height() or 1)
            nx = pos.x() / w
            ny = pos.y() / h
            return {'x': clamp01(nx), 'y': clamp01(ny)}

        x_off, y_off, w, h = self.last_draw_rect
        if w <= 0 or h <= 0:
            return None
        nx = (pos.x() - x_off) / w
        ny = (pos.y() - y_off) / h
        # Clamp instead of discarding; allows drawing even if slightly outside the video bounds.
        return {'x': clamp01(nx), 'y': clamp01(ny)}

    def _hit_test_vertex(self, pos: QPointF) -> Optional[Tuple[str, int]]:
        """Return (shape_id, index) if pointer is near a zone vertex."""
        for sh in self.shapes:
            if sh.get('kind') != 'zone' or sh.get('locked'):
                continue
            pts = sh.get('pts') or []
            for idx, pt in enumerate(pts):
                wpt = self._norm_to_widget(pt)
                if not wpt:
                    continue
                dx = pos.x() - wpt[0]
                dy = pos.y() - wpt[1]
                if (dx * dx + dy * dy) ** 0.5 <= 10.0:
                    return (sh['id'], idx)
        return None

    def _hit_test_line_end(self, pos: QPointF) -> Optional[Tuple[str, str]]:
        for sh in self.shapes:
            if sh.get('kind') != 'line' or sh.get('locked'):
                continue
            for end_key in ('p1', 'p2'):
                pt = sh.get(end_key)
                wpt = self._norm_to_widget(pt) if pt else None
                if not wpt:
                    continue
                dx = pos.x() - wpt[0]
                dy = pos.y() - wpt[1]
                if (dx * dx + dy * dy) ** 0.5 <= 10.0:
                    return (sh['id'], end_key)
        return None

    def _shape_contains(self, sh: Shape, norm_pt: Pt) -> bool:
        if sh.get('kind') == 'zone':
            pts = sh.get('pts') or []
            return len(pts) >= 3 and point_in_polygon(norm_pt, pts)
        if sh.get('kind') == 'line':
            p1, p2 = sh.get('p1'), sh.get('p2')
            if not p1 or not p2:
                return False
            return distance_to_line(norm_pt, p1, p2) <= 0.02
        if sh.get('kind') == 'tag':
            anchor = sh.get('anchor')
            if not anchor:
                return False
            dx = norm_pt['x'] - anchor['x']
            dy = norm_pt['y'] - anchor['y']
            return (dx * dx + dy * dy) ** 0.5 <= 0.03
        return False

    def _hit_test_shape(self, pos: QPointF) -> Optional[str]:
        norm = self._widget_to_norm(pos)
        if not norm:
            return None
        # Prefer existing selection ordering
        for sh in reversed(self.shapes):
            if sh.get('locked'):
                continue
            if self._shape_contains(sh, norm):
                return sh['id']
        return None

    def _add_zone(self, pts: List[Pt]):
        if len(pts) < 3:
            return
        idx = len([s for s in self.shapes if s.get('kind') == 'zone'])
        color = ZONE_COLORS[idx % len(ZONE_COLORS)]
        nz = {
            'id': uid('zone'),
            'kind': 'zone',
            'label': f"Zone {idx + 1}",
            'enabled': True,
            'pts': pts,
            'color': color,
            'alpha': 0.6,
            'show_label': True
        }
        self.shapes.append(nz)
        self._emit_shapes()
        self.update()

    def _add_line(self, p1: Pt, p2: Pt):
        idx = len([s for s in self.shapes if s.get('kind') == 'line'])
        nl = {
            'id': uid('line'),
            'kind': 'line',
            'label': f"Line {idx + 1}",
            'enabled': True,
            'p1': p1,
            'p2': p2,
            'alpha': 0.65,
            'show_label': True
        }
        self.shapes.append(nl)
        self._emit_shapes()
        self.update()

    def _add_tag(self, anchor: Pt):
        idx = len([s for s in self.shapes if s.get('kind') == 'tag'])
        color = TAG_COLORS[idx % len(TAG_COLORS)]
        nt = {
            'id': uid('tag'),
            'kind': 'tag',
            'label': f"Tag {idx + 1}",
            'enabled': True,
            'anchor': anchor,
            'color': color,
            'alpha': 0.7,
            'show_label': True
        }
        self.shapes.append(nt)
        self._emit_shapes()
        self.update()

    def update_detections(self, detections):
        """Backward-compatible alias: treat as backend detections."""
        self.update_backend_detections(detections)
        return

    def update_backend_detections(self, detections):
        """Update list of backend AI detections to render."""
        self.backend_detections_raw = list(detections or [])
        dets_filtered = self._apply_detection_roi_filter(self.backend_detections_raw)
        self.backend_detections = dets_filtered
        # Shape interactions for detections (optional)
        try:
            if bool((getattr(self, "detection_settings", {}) or {}).get("interact_with_shapes", True)):
                self._process_detection_shapes(self.backend_detections, source="backend")
        except Exception:
            pass
        self.update()

    def update_desktop_detections(self, detections):
        """Update list of Desktop-local AI detections to render."""
        self.desktop_detections_raw = list(detections or [])
        dets_filtered = self._apply_detection_roi_filter(self.desktop_detections_raw)
        self.desktop_detections = dets_filtered
        # Shape interactions for detections (optional)
        try:
            if bool((getattr(self, "detection_settings", {}) or {}).get("interact_with_shapes", True)):
                self._process_detection_shapes(self.desktop_detections, source="desktop")
        except Exception:
            pass
        self.update()

    def update_desktop_tracks(self, tracks):
        """Update list of Desktop-local tracked objects to render."""
        self.desktop_tracks_raw = list(tracks or [])
        try:
            # Reuse the same ROI filter logic (center-in-shape) as detections.
            self.desktop_tracks = self._apply_detection_roi_filter(self.desktop_tracks_raw)
        except Exception:
            self.desktop_tracks = list(self.desktop_tracks_raw or [])
        self.update()

    def set_desktop_tracking_active(self, active: bool):
        self.desktop_tracking_active = bool(active)
        if not self.desktop_tracking_active:
            self.desktop_tracks_raw = []
            self.desktop_tracks = []
            self.selected_track_id = None
        self.update()

    def set_object_overrides(self, overrides: Dict[int, dict]):
        """Set per-track override settings (name/color/hide)."""
        try:
            self.object_overrides = {int(k): (v if isinstance(v, dict) else {}) for k, v in (overrides or {}).items()}
        except Exception:
            self.object_overrides = {}
        self.update()

    def refresh_detection_roi(self):
        """
        Re-apply ROI filter to both detection layers using the last seen raw lists.
        Call this when ROI settings change.
        """
        try:
            self.backend_detections = self._apply_detection_roi_filter(self.backend_detections_raw)
        except Exception:
            self.backend_detections = list(self.backend_detections_raw or [])
        try:
            self.desktop_detections = self._apply_detection_roi_filter(self.desktop_detections_raw)
        except Exception:
            self.desktop_detections = list(self.desktop_detections_raw or [])
        self.update()

    def set_desktop_detection_active(self, active: bool):
        """Controls which detection layer is preferred for rendering."""
        self.desktop_detections_active = bool(active)
        self.update()

    def update_fps(self):
        self.fps = self.frame_count - self.last_frame_count
        self.last_frame_count = self.frame_count
        # Trigger update to redraw FPS if no new frames are coming
        self.update()

    @Slot(np.ndarray)
    def update_frame(self, frame):
        """Receive a numpy array (OpenCV frame), convert, and render."""
        try:
            # Auto-protection paint cap takes precedence over the
            # baseline small-widget throttle.  When the load shedder
            # has set a cap, drop frames arriving above that rate.
            try:
                shed_paint_cap = int(getattr(self, "_shed_paint_fps_cap", 0) or 0)
                if shed_paint_cap > 0:
                    interval = 1.0 / max(1, shed_paint_cap)
                    now_ts = time.time()
                    if (now_ts - self._last_paint_ts) < interval:
                        return
                    self._last_paint_ts = now_ts
                elif (self.width() < self._small_widget_threshold
                        and not self.show_motion):
                    # Throttle expensive paint work for small/grid tiles.
                    now_ts = time.time()
                    if (now_ts - self._last_paint_ts) < self._small_widget_paint_interval:
                        return
                    self._last_paint_ts = now_ts
            except Exception:
                pass

            # Optimization: Use BGR888 directly to avoid CPU-side cv2.cvtColor conversion
            # frame is already in BGR format from OpenCV
            h, w, ch = frame.shape
            bytes_per_line = frame.strides[0]
            
            # Create QImage from data (no copy, just wrapper)
            # QImage holds a reference to the data, so we must ensure frame stays alive 
            # (PyQt/Python ref counting usually handles this for the scope of the slot, 
            # but for safety in the widget we keep a reference via self.image)
            self.image = QImage(frame.data, w, h, bytes_per_line, QImage.Format.Format_BGR888)
            self.frame_dims = (w, h)

            # Simple Motion Detection (Frame Difference)
            if self.show_motion:
                # Auto-protection: throttle motion-detection rate when
                # the load shedder has imposed a cap.  Skipping motion
                # processing on the dropped ticks (instead of the whole
                # frame) keeps the live image fluid while CV2 work drops.
                shed_motion_cap = int(getattr(self, "_shed_motion_fps_cap", 0) or 0)
                _shed_skip_motion = False
                if shed_motion_cap > 0:
                    interval = 1.0 / max(1, shed_motion_cap)
                    now_ts = time.time()
                    if (now_ts - self._last_motion_ts) < interval:
                        _shed_skip_motion = True
                    else:
                        self._last_motion_ts = now_ts

                if _shed_skip_motion:
                    # Skip the heavy motion CV pipeline this frame.  The
                    # QImage above is already set, so just bump counters
                    # and trigger a repaint, then bail.
                    self.frame_count += 1
                    self.update()
                    return

                # Optimization: Downscale for faster motion processing
                motion_scale = 0.5
                small_frame = cv2.resize(frame, None, fx=motion_scale, fy=motion_scale, interpolation=cv2.INTER_LINEAR)
                
                gray = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)
                gray = cv2.GaussianBlur(gray, (21, 21), 0)
                
                if self.prev_gray is None or self.prev_gray.shape != gray.shape:
                    self.prev_gray = gray
                else:
                    sens = self.motion_settings.get('sensitivity', 50)
                    # Map 1-100 to Threshold 80-5 (Higher sens = lower threshold)
                    thresh_val = int(max(5, 80 - (sens * 0.75)))
                    # Map 1-100 to Min Area 2000-100 (Higher sens = smaller objects)
                    # Scale area threshold by scale^2
                    base_min_area = max(100, 2000 - (sens * 19))
                    min_area = int(base_min_area * (motion_scale * motion_scale))
                    
                    merge_sz = self.motion_settings.get('merge_size', 0)

                    frame_delta = cv2.absdiff(self.prev_gray, gray)
                    thresh = cv2.threshold(frame_delta, thresh_val, 255, cv2.THRESH_BINARY)[1]
                    
                    # Dynamic dilation based on merge_size
                    # Base (0 setting) = 3x3 kernel
                    extra_k = int(merge_sz * 0.3) 
                    k_dim = 3 + extra_k
                    kernel = np.ones((k_dim, k_dim), np.uint8)
                    
                    thresh = cv2.dilate(thresh, kernel, iterations=2)
                    contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    
                    current_centroids = []
                    current_boxes = []
                    
                    for c in contours:
                        if cv2.contourArea(c) < min_area:
                            continue
                        
                        # Get rect in small coordinates
                        sx, sy, sw, sh = cv2.boundingRect(c)
                        
                        # Scale back to original coordinates
                        x = int(sx / motion_scale)
                        y = int(sy / motion_scale)
                        w_box = int(sw / motion_scale)
                        h_box = int(sh / motion_scale)
                        
                        box = (x, y, w_box, h_box)
                        current_boxes.append(box)
                        
                        cx = x + w_box / 2
                        cy = y + h_box / 2
                        current_centroids.append((cx, cy))
                    
                    # Track objects
                    new_tracked = {}
                    used_centroids = set()
                    
                    # Match existing
                    for tid, tobj in self.tracked_objects.items():
                        last_cx, last_cy = tobj['history'][-1]
                        best_dist = 100.0 # Max match distance
                        best_idx = -1
                        
                        for idx, (cx, cy) in enumerate(current_centroids):
                            if idx in used_centroids: continue
                            dist = np.hypot(cx - last_cx, cy - last_cy)
                            if dist < best_dist:
                                best_dist = dist
                                best_idx = idx
                        
                        if best_idx != -1:
                            # Update existing
                            cx, cy = current_centroids[best_idx]
                            box = current_boxes[best_idx]
                            history = tobj['history']
                            history.append((cx, cy))
                            if len(history) > 30: history.pop(0)
                            
                            speed = best_dist # px per frame
                            # Smooth speed
                            speed = 0.7 * tobj.get('speed', 0) + 0.3 * speed
                            
                            new_tracked[tid] = {
                                'box': box,
                                'history': history,
                                'speed': speed,
                                'missing': 0
                            }
                            used_centroids.add(best_idx)
                        else:
                            # Keep lost objects briefly
                            if tobj['missing'] < 5:
                                tobj['missing'] += 1
                                new_tracked[tid] = tobj
                    
                    # Add new
                    for idx, (cx, cy) in enumerate(current_centroids):
                        if idx not in used_centroids:
                            self.tracker_next_id += 1
                            new_tracked[self.tracker_next_id] = {
                                'box': current_boxes[idx],
                                'history': [(cx, cy)],
                                'speed': 0,
                                'missing': 0
                            }
                    
                    self.tracked_objects = new_tracked
                    self.motion_boxes = current_boxes # For fallback/debug
                    self.prev_gray = gray
                    # Check for zone/line/tag interactions
                    self._process_motion_shapes(current_boxes)
            
            self.frame_count += 1
            self.update() # Trigger paintEvent
        except Exception as e:
            print(f"Frame error: {e}")

    def _process_motion_shapes(self, boxes: List[Tuple[int, int, int, int]]):
        """Check motion boxes against zones/lines/tags and emit triggers."""
        if not self.shapes or self.frame_dims[0] == 0:
            return
        src_w, src_h = self.frame_dims
        now = time.time()
        events = []
        for bx, by, bw, bh in boxes:
            cx = clamp01((bx + bw / 2) / src_w)
            cy = clamp01((by + bh / 2) / src_h)
            pt = {'x': cx, 'y': cy}
            for sh in self.shapes:
                if not sh.get('enabled', True) or sh.get('hidden', False):
                    continue
                kind = sh.get('kind')
                if kind == 'zone':
                    pts = sh.get('pts') or []
                    if len(pts) >= 3 and point_in_polygon(pt, pts):
                        self.zone_pulses[sh['id']] = now
                        events.append({
                            'shape_id': sh['id'], 'shape_type': 'zone',
                            'shape_name': sh.get('label') or sh.get('id', ''),
                            'interaction_type': 'entered_zone', 'point': pt,
                        })
                        self.motion_hit_pulses.append({'ts': now, 'box': (bx, by, bw, bh)})
                        break
                elif kind == 'line':
                    p1, p2 = sh.get('p1'), sh.get('p2')
                    if p1 and p2 and distance_to_line(pt, p1, p2) <= 0.02:
                        self.line_pulses[sh['id']] = {'ts': now, 'pt': pt}
                        events.append({
                            'shape_id': sh['id'], 'shape_type': 'line',
                            'shape_name': sh.get('label') or sh.get('id', ''),
                            'interaction_type': 'crossed_line', 'point': pt,
                        })
                        self.motion_hit_pulses.append({'ts': now, 'box': (bx, by, bw, bh)})
                        break
                elif kind == 'tag':
                    anchor = sh.get('anchor')
                    if anchor:
                        dx = pt['x'] - anchor['x']
                        dy = pt['y'] - anchor['y']
                        dist = (dx * dx + dy * dy) ** 0.5
                        if dist <= 0.03:
                            self.tag_pulses[sh['id']] = now
                            events.append({
                                'shape_id': sh['id'], 'shape_type': 'tag',
                                'shape_name': sh.get('label') or sh.get('id', ''),
                                'interaction_type': 'near_tag', 'point': pt,
                            })
                            self.motion_hit_pulses.append({'ts': now, 'box': (bx, by, bw, bh)})
                            break
        # prune old pulses and cap list
        self.motion_hit_pulses = [p for p in self.motion_hit_pulses if now - p.get('ts', 0) < 0.8]
        if len(self.motion_hit_pulses) > 50:
            self.motion_hit_pulses = self.motion_hit_pulses[-50:]
        if events and self.camera_id:
            self.shape_triggered.emit({'camera_id': self.camera_id, 'events': events})

    def _process_detection_shapes(self, dets: List[dict], *, source: str = "detection"):
        """
        Check detection boxes against zones/lines/tags and emit triggers.
        Mirrors `_process_motion_shapes`, but uses detection bboxes.
        """
        if not dets or not self.shapes or self.frame_dims[0] == 0:
            return

        ds = getattr(self, "detection_settings", None) or {}
        allow_zone = bool(ds.get("interact_zone", True))
        allow_line = bool(ds.get("interact_line", True))
        allow_tag = bool(ds.get("interact_tag", True))
        interact_line_dist = float(ds.get("interact_line_dist", 0.02) or 0.02)
        interact_tag_dist = float(ds.get("interact_tag_dist", 0.03) or 0.03)
        interact_zone_margin = float(ds.get("interact_zone_margin", 0.0) or 0.0)
        cooldown_ms = int(ds.get("interaction_cooldown_ms", 250) or 0)
        cooldown_s = max(0.0, float(cooldown_ms) / 1000.0)

        interact_line_dist = max(0.0, min(0.5, interact_line_dist))
        interact_tag_dist = max(0.0, min(0.5, interact_tag_dist))
        interact_zone_margin = max(0.0, min(0.5, interact_zone_margin))

        src_w, src_h = self.frame_dims
        now = time.time()
        events = []

        for det in dets:
            bbox = (det or {}).get("bbox") or {}
            # bbox can be dict or list/tuple
            try:
                if isinstance(bbox, dict):
                    bx = float(bbox.get("x", 0))
                    by = float(bbox.get("y", 0))
                    bw = float(bbox.get("w", 0))
                    bh = float(bbox.get("h", 0))
                elif isinstance(bbox, (list, tuple)) and len(bbox) >= 4:
                    bx, by, bw, bh = float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])
                else:
                    continue
            except Exception:
                continue

            cx = clamp01((bx + bw / 2) / src_w)
            cy = clamp01((by + bh / 2) / src_h)
            pt = {"x": cx, "y": cy}

            def _zone_near(pt0: dict, pts0: list) -> bool:
                if len(pts0) < 3:
                    return False
                if point_in_polygon(pt0, pts0):
                    return True
                if interact_zone_margin <= 0:
                    return False
                mind = 999.0
                for i in range(len(pts0)):
                    a = pts0[i]
                    b = pts0[(i + 1) % len(pts0)]
                    try:
                        d = distance_to_line(pt0, a, b)
                        if d < mind:
                            mind = d
                    except Exception:
                        continue
                return mind <= interact_zone_margin

            for sh in self.shapes:
                if not sh.get("enabled", True) or sh.get("hidden", False):
                    continue
                kind = sh.get("kind")

                if kind == "zone" and allow_zone:
                    pts = sh.get("pts") or []
                    if len(pts) >= 3 and _zone_near(pt, pts):
                        self.zone_pulses[sh["id"]] = now
                        events.append(
                            {
                                "shape_id": sh["id"],
                                "shape_type": "zone",
                                "shape_name": sh.get("label") or sh.get("id", ""),
                                "interaction_type": "entered_zone",
                                "point": pt,
                                "source": source,
                                "det_class": det.get("class"),
                                "confidence": det.get("confidence"),
                            }
                        )
                        self.detection_hit_pulses.append({"ts": now, "box": (bx, by, bw, bh), "source": source})
                        break

                if kind == "line" and allow_line:
                    p1, p2 = sh.get("p1"), sh.get("p2")
                    if p1 and p2 and distance_to_line(pt, p1, p2) <= interact_line_dist:
                        self.line_pulses[sh["id"]] = {"ts": now, "pt": pt}
                        events.append(
                            {
                                "shape_id": sh["id"],
                                "shape_type": "line",
                                "shape_name": sh.get("label") or sh.get("id", ""),
                                "interaction_type": "crossed_line",
                                "point": pt,
                                "source": source,
                                "det_class": det.get("class"),
                                "confidence": det.get("confidence"),
                            }
                        )
                        self.detection_hit_pulses.append({"ts": now, "box": (bx, by, bw, bh), "source": source})
                        break

                if kind == "tag" and allow_tag:
                    anchor = sh.get("anchor")
                    if anchor:
                        dx = pt["x"] - float(anchor.get("x", 0.5))
                        dy = pt["y"] - float(anchor.get("y", 0.5))
                        dist = (dx * dx + dy * dy) ** 0.5
                        if dist <= interact_tag_dist:
                            self.tag_pulses[sh["id"]] = now
                            events.append(
                                {
                                    "shape_id": sh["id"],
                                    "shape_type": "tag",
                                    "shape_name": sh.get("label") or sh.get("id", ""),
                                    "interaction_type": "near_tag",
                                    "point": pt,
                                    "source": source,
                                    "det_class": det.get("class"),
                                    "confidence": det.get("confidence"),
                                }
                            )
                            self.detection_hit_pulses.append({"ts": now, "box": (bx, by, bw, bh), "source": source})
                            break

        # prune old pulses and cap list
        self.detection_hit_pulses = [p for p in self.detection_hit_pulses if now - p.get("ts", 0) < 0.8]
        if len(self.detection_hit_pulses) > 50:
            self.detection_hit_pulses = self.detection_hit_pulses[-50:]

        # emit events with cooldown to avoid spamming downstream listeners
        if events and self.camera_id:
            last = float(getattr(self, "_detection_last_emit_ts", 0.0) or 0.0)
            if cooldown_s <= 0.0 or (now - last) >= cooldown_s:
                self._detection_last_emit_ts = now
                self.shape_triggered.emit({"camera_id": self.camera_id, "events": events, "source": source})

    def _apply_detection_roi_filter(self, dets: List[dict]) -> List[dict]:
        """
        ROI filter for detections using zones/lines/tags.
        Keeps detections whose center is in/near at least one ROI shape.
        """
        ds = getattr(self, "detection_settings", None) or {}
        if not bool(ds.get("roi_enabled", False)):
            return list(dets or [])
        if not dets or not self.shapes or self.frame_dims[0] == 0:
            return list(dets or [])

        try:
            roi_selected_only = bool(ds.get("roi_selected_only", False))
            roi_zone = bool(ds.get("roi_zone", True))
            roi_line = bool(ds.get("roi_line", True))
            roi_tag = bool(ds.get("roi_tag", True))
            roi_line_dist = float(ds.get("roi_line_dist", 0.02) or 0.02)
            roi_tag_dist = float(ds.get("roi_tag_dist", 0.03) or 0.03)
            roi_zone_margin = float(ds.get("roi_zone_margin", 0.0) or 0.0)
        except Exception:
            roi_selected_only = False
            roi_zone = True
            roi_line = True
            roi_tag = True
            roi_line_dist = 0.02
            roi_tag_dist = 0.03
            roi_zone_margin = 0.0

        roi_line_dist = max(0.0, min(0.5, roi_line_dist))
        roi_tag_dist = max(0.0, min(0.5, roi_tag_dist))
        roi_zone_margin = max(0.0, min(0.5, roi_zone_margin))

        shapes = self.shapes or []
        if roi_selected_only and getattr(self, "selected_shapes", None):
            sel = set(self.selected_shapes or [])
            shapes = [s for s in shapes if s.get("id") in sel]

        src_w, src_h = self.frame_dims

        def zone_near(pt: dict, pts: list) -> bool:
            if len(pts) < 3:
                return False
            if point_in_polygon(pt, pts):
                return True
            if roi_zone_margin <= 0:
                return False
            # Distance to polygon edges
            mind = 999.0
            for i in range(len(pts)):
                a = pts[i]
                b = pts[(i + 1) % len(pts)]
                try:
                    d = distance_to_line(pt, a, b)
                    if d < mind:
                        mind = d
                except Exception:
                    continue
            return mind <= roi_zone_margin

        kept: list[dict] = []
        for det in dets:
            bbox = (det or {}).get("bbox") or {}
            try:
                if isinstance(bbox, dict):
                    bx = float(bbox.get("x", 0))
                    by = float(bbox.get("y", 0))
                    bw = float(bbox.get("w", 0))
                    bh = float(bbox.get("h", 0))
                elif isinstance(bbox, (list, tuple)) and len(bbox) >= 4:
                    bx, by, bw, bh = float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])
                else:
                    continue
            except Exception:
                continue

            cx = clamp01((bx + bw / 2) / src_w)
            cy = clamp01((by + bh / 2) / src_h)
            pt = {"x": cx, "y": cy}

            ok = False
            for sh in shapes:
                if not sh.get("enabled", True) or sh.get("hidden", False):
                    continue
                kind = sh.get("kind")
                if kind == "zone" and roi_zone:
                    pts = sh.get("pts") or []
                    if zone_near(pt, pts):
                        ok = True
                        break
                elif kind == "line" and roi_line:
                    p1, p2 = sh.get("p1"), sh.get("p2")
                    if p1 and p2 and distance_to_line(pt, p1, p2) <= roi_line_dist:
                        ok = True
                        break
                elif kind == "tag" and roi_tag:
                    anchor = sh.get("anchor")
                    if anchor:
                        dx = pt["x"] - float(anchor.get("x", 0.5))
                        dy = pt["y"] - float(anchor.get("y", 0.5))
                        if (dx * dx + dy * dy) ** 0.5 <= roi_tag_dist:
                            ok = True
                            break

            if ok:
                kept.append(det)
        return kept

    def paintEvent(self, event):
        """Render the image using QPainter."""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        # Determine if playback is active and rewound (not at live edge).
        # When rewound, suppress live overlays and draw historical detections instead.
        _is_playback_rewound = False
        try:
            w_parent = self.parent()
            for _ in range(5):
                if w_parent is None:
                    break
                if hasattr(w_parent, '_playback_active'):
                    if getattr(w_parent, '_playback_active', False):
                        po = getattr(w_parent, 'playback_overlay', None)
                        eng = po._engine if po else None
                        if eng and not eng.is_at_live_edge:
                            _is_playback_rewound = True
                    break
                w_parent = w_parent.parent() if hasattr(w_parent, 'parent') else None
        except Exception:
            pass

        # Clear background
        painter.fillRect(self.rect(), Qt.GlobalColor.black)

        # Choose layout image (for sizing / draw rect). Prefer camera frame if present.
        base_img = None
        if self.image and (not self.image.isNull()):
            base_img = self.image
        elif self.depth_overlay_image and (not self.depth_overlay_image.isNull()):
            base_img = self.depth_overlay_image

        if base_img and not base_img.isNull():
            # On small tiles (<480px wide), FastTransformation is visually
            # indistinguishable from SmoothTransformation on moving video
            # but is significantly cheaper, freeing CPU for the other 8+
            # cameras in a multi-cam grid.
            if self.width() < self._small_widget_threshold:
                _xform = Qt.TransformationMode.FastTransformation
            else:
                _xform = Qt.TransformationMode.SmoothTransformation
            scaled_img = base_img.scaled(
                self.size(),
                self.aspect_ratio_mode,
                _xform,
            )
            
            # Center the image (crop the overflow or center fit)
            x_offset = (self.width() - scaled_img.width()) // 2
            y_offset = (self.height() - scaled_img.height()) // 2
            # Track draw rect for hit-testing and normalized conversions
            self.last_draw_rect = (float(x_offset), float(y_offset), float(scaled_img.width()), float(scaled_img.height()))
            is_pointcloud = bool(getattr(self, "depth_overlay_is_pointcloud", False))

            # POINTCLOUD MODE:
            # Draw visualization first (so camera can be a transparent overlay above it),
            # and use the camera's alpha to avoid "black void" in sparse areas.
            if (
                is_pointcloud
                and (not self.depth_overlay_replace_base)
                and self.depth_overlay_enabled
                and self.depth_overlay_image
                and (not self.depth_overlay_image.isNull())
            ):
                # 1) Draw point cloud visualization (scaled/centered)
                try:
                    painter.save()
                    painter.setOpacity(float(self.depth_overlay_opacity))
                    s = float(getattr(self, "depth_overlay_scale", 1.0) or 1.0)
                    s = max(0.25, min(3.0, s))
                    target_w = max(1, int(round(scaled_img.width() * s)))
                    target_h = max(1, int(round(scaled_img.height() * s)))
                    depth_scaled = self.depth_overlay_image.scaled(
                        target_w,
                        target_h,
                        Qt.AspectRatioMode.IgnoreAspectRatio,
                        Qt.TransformationMode.SmoothTransformation,
                    )
                    ox = int(x_offset + (scaled_img.width() - depth_scaled.width()) / 2)
                    oy = int(y_offset + (scaled_img.height() - depth_scaled.height()) / 2)
                    painter.drawImage(ox, oy, depth_scaled)
                finally:
                    painter.restore()

                # 2) Draw camera as a transparent overlay (optional)
                if self.image and (not self.image.isNull()) and (not bool(self.depth_overlay_blackout_base)):
                    try:
                        painter.save()
                        painter.setOpacity(float(getattr(self, "base_camera_opacity", 1.0) or 1.0))
                        cam_scaled = self.image.scaled(
                            scaled_img.size(),
                            Qt.AspectRatioMode.IgnoreAspectRatio,
                            Qt.TransformationMode.SmoothTransformation,
                        )
                        painter.drawImage(x_offset, y_offset, cam_scaled)
                    finally:
                        painter.restore()
            else:
                # DEFAULT MODE:
                # Draw camera image unless user blacked it out (viz-only mode).
                if not bool(self.depth_overlay_blackout_base) or base_img is not self.image:
                    painter.drawImage(x_offset, y_offset, scaled_img)
            scale_x = scaled_img.width() / self.frame_dims[0] if self.frame_dims[0] else 1.0
            scale_y = scaled_img.height() / self.frame_dims[1] if self.frame_dims[1] else 1.0

            # Depth overlay (drawn above base image, below motion/boxes/shapes)
            if (
                (not is_pointcloud)
                and (not self.depth_overlay_replace_base)
                and self.depth_overlay_enabled
                and self.depth_overlay_image
                and (not self.depth_overlay_image.isNull())
            ):
                try:
                    painter.save()
                    painter.setOpacity(float(self.depth_overlay_opacity))
                    # Apply overlay scaling around center of the base draw rect.
                    s = float(getattr(self, "depth_overlay_scale", 1.0) or 1.0)
                    s = max(0.25, min(3.0, s))
                    target_w = max(1, int(round(scaled_img.width() * s)))
                    target_h = max(1, int(round(scaled_img.height() * s)))
                    depth_scaled = self.depth_overlay_image.scaled(
                        target_w,
                        target_h,
                        Qt.AspectRatioMode.IgnoreAspectRatio,
                        Qt.TransformationMode.SmoothTransformation,
                    )
                    ox = int(x_offset + (scaled_img.width() - depth_scaled.width()) / 2)
                    oy = int(y_offset + (scaled_img.height() - depth_scaled.height()) / 2)
                    painter.drawImage(ox, oy, depth_scaled)
                finally:
                    painter.restore()

            # Draw Motion Boxes (suppress during playback rewind)
            if self.show_motion and self.frame_dims[0] > 0 and not _is_playback_rewound:
                src_w, src_h = self.frame_dims
                
                base_color = self.motion_settings['color']
                thickness = self.motion_settings['thickness']
                style = self.motion_settings['style']
                anim = self.motion_settings.get('animation', 'None')
                show_trails = self.motion_settings.get('trails', False)
                color_speed = self.motion_settings.get('color_speed', False)

                # Animation Logic
                t = time.time()
                
                # Determine which objects to draw
                # Prefer tracked objects for richer features, fallback to raw boxes if empty (rare)
                objects_to_draw = list(self.tracked_objects.values())
                if not objects_to_draw and self.motion_boxes:
                    # Fallback to raw boxes (wrap in dict to match structure)
                    objects_to_draw = [{'box': b, 'history': [], 'speed': 0} for b in self.motion_boxes]

                painter.setPen(Qt.PenStyle.NoPen)
                painter.setBrush(Qt.BrushStyle.NoBrush)

                for obj in objects_to_draw:
                    bx, by, bw, bh = obj['box']

                    # Determine if this motion box recently hit a shape for subtle glow
                    hit_strength = 0.0
                    now_ts = time.time()
                    cx_src = bx + bw / 2
                    cy_src = by + bh / 2
                    # Allow detection boxes to interact/animate with shapes similarly to motion boxes.
                    for pulse in list(self.motion_hit_pulses) + list(getattr(self, "detection_hit_pulses", []) or []):
                        age = now_ts - pulse.get('ts', 0)
                        if age > 0.8:
                            continue
                        pbx, pby, pbw, pbh = pulse.get('box', (0, 0, 0, 0))
                        pcx = pbx + pbw / 2
                        pcy = pby + pbh / 2
                        dist = np.hypot(cx_src - pcx, cy_src - pcy)
                        if dist < max(bw, bh) * 0.75:
                            hit_strength = max(hit_strength, 1.0 - age / 0.8)
                    
                    # Map source coords to widget coords
                    draw_x = x_offset + bx * scale_x
                    draw_y = y_offset + by * scale_y
                    draw_w = bw * scale_x
                    draw_h = bh * scale_y
                    
                    cx, cy = draw_x + draw_w/2, draw_y + draw_h/2

                    # --- Color Logic ---
                    final_color = QColor(base_color)
                    
                    if color_speed:
                        # Map speed (0-50) to Color (Green -> Red)
                        # Hue: 120 (Green) -> 0 (Red)
                        sp = min(obj.get('speed', 0), 20.0) # Cap at 20 px/frame
                        hue = 120 - (sp / 20.0 * 120)
                        final_color.setHslF(hue / 360.0, 1.0, 0.5)
                    
                    if anim == 'Rainbow':
                        # Rotate hue over time
                        hue = (t * 50) % 360
                        final_color.setHslF(hue / 360.0, 1.0, 0.5)

                    # --- Animation Transforms ---
                    alpha_mult = 1.0
                    size_mult = 1.0
                    
                    if anim == 'Pulse':
                        val = (np.sin(t * 5) + 1) / 2
                        alpha_mult = 0.4 + 0.6 * val
                        size_mult = 1.0 + 0.05 * val
                    elif anim == 'Flash':
                        val = int(t * 8) % 2
                        alpha_mult = 1.0 if val else 0.2
                    elif anim == 'Glitch':
                        # Random offset and size jitter
                        if random.random() > 0.7:
                            draw_x += random.randint(-5, 5)
                            draw_y += random.randint(-5, 5)
                            draw_w += random.randint(-5, 5)
                            draw_h += random.randint(-5, 5)
                            # Occasional color glitch
                            if random.random() > 0.5:
                                final_color = QColor(0, 255, 255) if random.random() > 0.5 else QColor(255, 0, 255)

                    final_color.setAlphaF(min(1.0, final_color.alphaF() * alpha_mult))

                    # Apply Size Multiplier
                    if size_mult != 1.0:
                        draw_w *= size_mult
                        draw_h *= size_mult
                        draw_x = cx - draw_w/2
                        draw_y = cy - draw_h/2

                    # Subtle interaction glow when motion intersects shapes
                    if hit_strength > 0:
                        glow_pen = QPen(QColor(0, 255, 200))
                        glow_alpha = int(180 * hit_strength)
                        glow_pen.setColor(QColor(0, 255, 200, glow_alpha))
                        glow_pen.setWidth(max(3, thickness + 1))
                        painter.setPen(glow_pen)
                        painter.setBrush(Qt.BrushStyle.NoBrush)
                        painter.drawRect(QRectF(draw_x, draw_y, draw_w, draw_h))
                        painter.fillRect(QRectF(draw_x, draw_y, draw_w, draw_h), QColor(0, 255, 200, int(40 * hit_strength)))
                        # Brief ripple
                        ripple_pen = QPen(QColor(0, 255, 180, int(120 * hit_strength)))
                        ripple_pen.setWidth(max(2, thickness))
                        painter.setPen(ripple_pen)
                        grow = 6 + 12 * hit_strength
                        painter.drawRect(QRectF(draw_x - grow/2, draw_y - grow/2, draw_w + grow, draw_h + grow))

                    # --- Draw Trails ---
                    if show_trails and len(obj.get('history', [])) > 1:
                        path_pen = QPen(final_color)
                        path_pen.setWidth(max(1, thickness // 2))
                        path_pen.setStyle(Qt.PenStyle.DotLine)
                        painter.setPen(path_pen)
                        
                        hist_points = []
                        for (hx, hy) in obj['history']:
                            px = x_offset + hx * scale_x
                            py = y_offset + hy * scale_y
                            hist_points.append(QPointF(px, py))
                        
                        painter.drawPolyline(hist_points)

                    # --- Draw Box/Shape ---
                    if style == 'Fill':
                        painter.setPen(Qt.PenStyle.NoPen)
                        fill_color = QColor(final_color)
                        fill_color.setAlpha(min(fill_color.alpha(), 100))
                        painter.setBrush(fill_color)
                    else:
                        pen = QPen(final_color)
                        pen.setWidth(thickness)
                        painter.setPen(pen)
                        painter.setBrush(Qt.BrushStyle.NoBrush)
                    
                    if style == 'Box':
                        painter.drawRect(QRectF(draw_x, draw_y, draw_w, draw_h))
                    elif style == 'Fill':
                        painter.drawRect(QRectF(draw_x, draw_y, draw_w, draw_h))
                    elif style == 'Circle':
                        painter.drawEllipse(QRectF(draw_x, draw_y, draw_w, draw_h))
                    elif style == 'Underline':
                        painter.drawLine(int(draw_x), int(draw_y + draw_h), int(draw_x + draw_w), int(draw_y + draw_h))
                    elif style == 'Crosshair':
                        len_c = min(draw_w, draw_h) / 2
                        # Horizontal
                        painter.drawLine(int(cx - len_c), int(cy), int(cx + len_c), int(cy))
                        # Vertical
                        painter.drawLine(int(cx), int(cy - len_c), int(cx), int(cy + len_c))
                    elif style == 'Bracket':
                        # ... existing bracket logic ...
                        painter.drawLine(int(draw_x), int(draw_y), int(draw_x), int(draw_y + draw_h))
                        painter.drawLine(int(draw_x), int(draw_y), int(draw_x + 5), int(draw_y))
                        painter.drawLine(int(draw_x), int(draw_y + draw_h), int(draw_x + 5), int(draw_y + draw_h))
                        painter.drawLine(int(draw_x + draw_w), int(draw_y), int(draw_x + draw_w), int(draw_y + draw_h))
                        painter.drawLine(int(draw_x + draw_w), int(draw_y), int(draw_x + draw_w - 5), int(draw_y))
                        painter.drawLine(int(draw_x + draw_w), int(draw_y + draw_h), int(draw_x + draw_w - 5), int(draw_y + draw_h))
                    elif style == 'Corners':
                        # ... existing corners logic ...
                        len_x = min(draw_w / 3, 20)
                        len_y = min(draw_h / 3, 20)
                        x, y, w, h = int(draw_x), int(draw_y), int(draw_w), int(draw_h)
                        painter.drawLine(x, y, int(x + len_x), y)
                        painter.drawLine(x, y, x, int(y + len_y))
                        painter.drawLine(x + w, y, int(x + w - len_x), y)
                        painter.drawLine(x + w, y, x + w, int(y + len_y))
                        painter.drawLine(x, y + h, int(x + len_x), y + h)
                        painter.drawLine(x, y + h, x, int(y + h - len_y))
                        painter.drawLine(x + w, y + h, int(x + w - len_x), y + h)
                        painter.drawLine(x + w, y + h, x + w, int(y + h - len_y))

            # Draw user-defined zones, lines, and tags (normalized coordinates)
            # Hide shapes when playback_hide_shapes is True (controlled by overlay filter toggle).
            if self.frame_dims[0] > 0 and not getattr(self, 'playback_hide_shapes', False):
                painter.save()
                now = time.time()
                font = QFont(painter.font())
                font.setPointSize(10)
                painter.setFont(font)

                view_w = float(scaled_img.width())
                view_h = float(scaled_img.height())

                for sh in self.shapes:
                    if sh.get('hidden'):
                        continue
                    sid = sh.get('id')
                    kind = sh.get('kind')
                    alpha = float(sh.get('alpha', 0.65)) * (1.0 if sh.get('enabled', True) else 0.55)
                    col = color_from_shape(sh)
                    col.setAlphaF(min(1.0, alpha))
                    
                    # Appearance Settings
                    line_thick = max(0.5, float(sh.get('line_thickness', 1.0)))
                    dot_sz = max(0.5, float(sh.get('dot_size', 2.0)))
                    tag_sz = max(10, sh.get('tag_size', 20))
                    tag_style = sh.get('tag_style', 'Crosshair')
                    line_w = max(1, int(round(line_thick)))

                    pen = QPen(col)
                    pen.setWidth(line_w)
                    painter.setPen(pen)
                    painter.setBrush(Qt.BrushStyle.NoBrush)
                    is_selected = sid in self.selected_shapes

                    if kind == 'zone':
                        pts = sh.get('pts') or []
                        if len(pts) < 3:
                            continue
                        poly = QPolygonF([QPointF(x_offset + p['x'] * view_w, y_offset + p['y'] * view_h) for p in pts])
                        fill = QColor(col)
                        fill.setAlpha(int(alpha * 90))
                        painter.setBrush(fill)
                        painter.drawPolygon(poly)
                        pen.setWidth(max(1, int(round(line_thick + (1 if is_selected else 0)))))
                        painter.setPen(pen)
                        painter.setBrush(Qt.BrushStyle.NoBrush)
                        painter.drawPolygon(poly)
                        
                        # Interaction effect (suppress live pulses during playback)
                        pulse_ts = None if _is_playback_rewound else self.zone_pulses.get(sid)
                        if pulse_ts:
                            age = now - pulse_ts
                            if age < 0.8:
                                # Custom interaction color or default gold
                                icol_val = sh.get('interaction_color') or '#FFD74A'
                                icol = QColor(icol_val)
                                anim = sh.get('interaction_animation', 'Pulse')
                                
                                if anim == 'Pulse':
                                    # Standard fill pulse
                                    icol.setAlpha(int(150 * (1.0 - age/0.8)))
                                    painter.setBrush(icol)
                                    painter.setPen(Qt.PenStyle.NoPen)
                                    painter.drawPolygon(poly)
                                    
                                elif anim == 'Ripple':
                                    # Expanding outlines
                                    painter.setBrush(Qt.BrushStyle.NoBrush)
                                    for i in range(3):
                                        prog = (age / 0.8 + i * 0.33) % 1.0
                                        alpha = int(255 * (1.0 - prog))
                                        icol.setAlpha(alpha)
                                        rpen = QPen(icol)
                                        rpen.setWidth(max(1, int(round(line_thick + int(6 * prog)))))
                                        painter.setPen(rpen)
                                        painter.drawPolygon(poly)

                                elif anim == 'Flash':
                                    # Bright flash then fade
                                    f_alpha = 200 if age < 0.1 else int(200 * (1.0 - (age-0.1)/0.7))
                                    icol.setAlpha(f_alpha)
                                    painter.setBrush(icol)
                                    painter.setPen(Qt.PenStyle.NoPen)
                                    painter.drawPolygon(poly)

                                elif anim == 'Outline':
                                    # Glowing border
                                    glow = int(127 + 127 * np.sin(age * 15))
                                    icol.setAlpha(glow)
                                    open = QPen(icol)
                                    open.setWidth(max(1, int(round(line_thick + 3))))
                                    painter.setPen(open)
                                    painter.setBrush(Qt.BrushStyle.NoBrush)
                                    painter.drawPolygon(poly)
                            else:
                                self.zone_pulses.pop(sid, None)
                                
                        if not sh.get('locked'):
                            painter.setBrush(QColor(col.red(), col.green(), col.blue(), 180))
                            handle_sz = dot_sz + 1 if is_selected else dot_sz
                            for p in pts:
                                painter.drawEllipse(QPointF(x_offset + p['x'] * view_w, y_offset + p['y'] * view_h),
                                                     handle_sz, handle_sz)
                        if self.show_shape_labels and sh.get('show_label', True):
                            cx = sum(p['x'] for p in pts) / len(pts)
                            cy = sum(p['y'] for p in pts) / len(pts)
                            tfont = QFont(painter.font())
                            tfont.setPointSize(int(sh.get('text_size', 12)))
                            painter.setFont(tfont)
                            painter.setPen(QColor(sh.get('text_color', '#F0F0F0')))
                            painter.drawText(QPointF(x_offset + cx * view_w + 8, y_offset + cy * view_h),
                                             f"{sh.get('label') or 'Zone'}{' (off)' if not sh.get('enabled', True) else ''}")

                    elif kind == 'line':
                        p1, p2 = sh.get('p1'), sh.get('p2')
                        if not p1 or not p2:
                            continue
                        x1 = x_offset + p1['x'] * view_w
                        y1 = y_offset + p1['y'] * view_h
                        x2 = x_offset + p2['x'] * view_w
                        y2 = y_offset + p2['y'] * view_h
                        pen.setWidth(max(1, int(round(line_thick + (1 if is_selected else 0)))))
                        pen.setStyle(Qt.PenStyle.DashLine)
                        painter.setPen(pen)
                        painter.drawLine(QPointF(x1, y1), QPointF(x2, y2))
                        
                        pulse = None if _is_playback_rewound else self.line_pulses.get(sid)
                        if pulse:
                            age = now - pulse.get('ts', 0)
                            if age < 0.8 and pulse.get('pt'):
                                px = x_offset + pulse['pt']['x'] * view_w
                                py = y_offset + pulse['pt']['y'] * view_h
                                icol_val = sh.get('interaction_color') or '#FFD74A'
                                icol = QColor(icol_val)
                                anim = sh.get('interaction_animation', 'Pulse')
                                
                                if anim == 'Pulse':
                                    rad = 12 + age * 40
                                    icol.setAlpha(int(255 * (1.0 - age/0.8)))
                                    painter.setPen(QPen(icol, 3))
                                    painter.setBrush(Qt.BrushStyle.NoBrush)
                                    painter.drawEllipse(QPointF(px, py), rad, rad)
                                    
                                elif anim == 'Ripple':
                                    for i in range(3):
                                        prog = (age / 0.8 + i * 0.33) % 1.0
                                        rad = 5 + prog * 60
                                        alpha = int(255 * (1.0 - prog))
                                        icol.setAlpha(alpha)
                                        painter.setPen(QPen(icol, 2))
                                        painter.setBrush(Qt.BrushStyle.NoBrush)
                                        painter.drawEllipse(QPointF(px, py), rad, rad)

                                elif anim == 'Flash':
                                    f_alpha = 255 if age < 0.1 else int(255 * (1.0 - (age-0.1)/0.7))
                                    icol.setAlpha(f_alpha)
                                    painter.setBrush(icol)
                                    painter.setPen(Qt.PenStyle.NoPen)
                                    painter.drawEllipse(QPointF(px, py), 30, 30)
                                    
                                elif anim == 'Outline':
                                    # Highlight the whole line
                                    glow = int(127 + 127 * np.sin(age * 15))
                                    icol.setAlpha(glow)
                                    lpen = QPen(icol)
                                    lpen.setWidth(max(1, int(round(line_thick + 4))))
                                    lpen.setCapStyle(Qt.PenCapStyle.RoundCap)
                                    painter.setPen(lpen)
                                    painter.drawLine(QPointF(x1, y1), QPointF(x2, y2))

                            else:
                                self.line_pulses.pop(sid, None)
                        if not sh.get('locked'):
                            handle_color = QColor(255, 154, 162)
                            painter.setBrush(handle_color)
                            painter.setPen(QPen(handle_color))
                            handle_sz = dot_sz + 1 if is_selected else dot_sz
                            painter.drawEllipse(QPointF(x1, y1), handle_sz, handle_sz)
                            painter.drawEllipse(QPointF(x2, y2), handle_sz, handle_sz)
                        if self.show_shape_labels and sh.get('show_label', True):
                            mx = (x1 + x2) / 2
                            my = (y1 + y2) / 2
                            tfont = QFont(painter.font())
                            tfont.setPointSize(int(sh.get('text_size', 12)))
                            painter.setFont(tfont)
                            painter.setPen(QColor(sh.get('text_color', '#F0F0F0')))
                            painter.drawText(QPointF(mx + 6, my - 6), sh.get('label') or 'Line')

                    elif kind == 'tag':
                        anchor = sh.get('anchor')
                        if not anchor:
                            continue
                        
                        # Floating animation
                        # 0.5Hz, 4px amplitude
                        float_y = np.sin(now * 3.0) * 4.0
                        
                        cx = x_offset + anchor['x'] * view_w
                        cy = y_offset + anchor['y'] * view_h + float_y
                        
                        painter.setPen(pen)
                        
                        # Render based on style
                        # tag_sz is full width/height approx
                        half_sz = tag_sz / 2.0
                        
                        if tag_style == 'Crosshair':
                            gap = tag_sz * 0.2
                            painter.drawLine(QPointF(cx - half_sz, cy), QPointF(cx - gap, cy))
                            painter.drawLine(QPointF(cx + gap, cy), QPointF(cx + half_sz, cy))
                            painter.drawLine(QPointF(cx, cy - half_sz), QPointF(cx, cy - gap))
                            painter.drawLine(QPointF(cx, cy + gap), QPointF(cx, cy + half_sz))
                            
                        elif tag_style == 'Target':
                            painter.setBrush(Qt.BrushStyle.NoBrush)
                            painter.drawEllipse(QPointF(cx, cy), half_sz, half_sz)
                            painter.setBrush(col)
                            painter.drawEllipse(QPointF(cx, cy), 3, 3)
                            
                        elif tag_style == 'Pin':
                            # Triangle pointing down
                            tri = QPolygonF([
                                QPointF(cx - half_sz*0.6, cy - half_sz), 
                                QPointF(cx + half_sz*0.6, cy - half_sz),
                                QPointF(cx, cy + half_sz)
                            ])
                            painter.setBrush(col)
                            painter.drawPolygon(tri)
                            painter.setBrush(QColor(0,0,0,100))
                            painter.drawEllipse(QPointF(cx, cy - half_sz), half_sz*0.3, half_sz*0.3)
                            
                        elif tag_style == 'Diamond':
                            path = QPolygonF([
                                QPointF(cx, cy - half_sz),
                                QPointF(cx + half_sz, cy),
                                QPointF(cx, cy + half_sz),
                                QPointF(cx - half_sz, cy)
                            ])
                            painter.setBrush(col) # Fill diamond
                            painter.drawPolygon(path)
                            
                        elif tag_style == 'Square':
                            painter.setBrush(Qt.BrushStyle.NoBrush)
                            painter.drawRect(QRectF(cx - half_sz, cy - half_sz, tag_sz, tag_sz))
                            painter.setBrush(col)
                            painter.drawRect(QRectF(cx - 2, cy - 2, 4, 4))
                            
                        elif tag_style == 'Triangle':
                            path = QPolygonF([
                                QPointF(cx, cy - half_sz),
                                QPointF(cx + half_sz, cy + half_sz),
                                QPointF(cx - half_sz, cy + half_sz)
                            ])
                            painter.setBrush(Qt.BrushStyle.NoBrush)
                            painter.drawPolygon(path)
                            
                        elif tag_style == 'Star':
                            # Simple star approach
                            path = QPolygonF()
                            for i in range(5):
                                angle = i * 4 * np.pi / 5 - np.pi / 2
                                path.append(QPointF(cx + half_sz * np.cos(angle), cy + half_sz * np.sin(angle)))
                            painter.setBrush(col)
                            painter.drawPolygon(path)
                            
                        elif tag_style == 'Shield':
                            path = QPolygonF([
                                QPointF(cx - half_sz, cy - half_sz),
                                QPointF(cx + half_sz, cy - half_sz),
                                QPointF(cx + half_sz, cy),
                                QPointF(cx, cy + half_sz),
                                QPointF(cx - half_sz, cy)
                            ])
                            painter.setBrush(Qt.BrushStyle.NoBrush)
                            painter.drawPolygon(path)
                            painter.drawText(QPointF(cx - 3, cy + 3), "S")
                            
                        elif tag_style == 'Warning':
                            # Yellow triangle with !
                            path = QPolygonF([
                                QPointF(cx, cy - half_sz),
                                QPointF(cx + half_sz, cy + half_sz),
                                QPointF(cx - half_sz, cy + half_sz)
                            ])
                            warn_col = QColor(255, 200, 0)
                            warn_col.setAlphaF(alpha)
                            painter.setBrush(warn_col)
                            painter.setPen(QPen(Qt.GlobalColor.black))
                            painter.drawPolygon(path)
                            painter.drawText(QPointF(cx - 3, cy + half_sz/2), "!")

                        elif tag_style == 'Person':
                             # Stick figure head + body
                             painter.setBrush(col)
                             painter.drawEllipse(QPointF(cx, cy - half_sz*0.6), half_sz*0.4, half_sz*0.4)
                             path = QPolygonF([
                                 QPointF(cx - half_sz*0.6, cy + half_sz),
                                 QPointF(cx, cy - half_sz*0.1),
                                 QPointF(cx + half_sz*0.6, cy + half_sz)
                             ])
                             painter.drawPolygon(path)
                             
                        elif tag_style == 'Car':
                            painter.setBrush(col)
                            painter.drawRect(QRectF(cx - half_sz, cy, tag_sz, half_sz/2))
                            painter.drawRect(QRectF(cx - half_sz*0.6, cy - half_sz*0.6, tag_sz*0.6, half_sz*0.6))

                        else: # Fallback to Crosshair
                            gap = tag_sz * 0.2
                            painter.drawLine(QPointF(cx - half_sz, cy), QPointF(cx - gap, cy))
                            painter.drawLine(QPointF(cx + gap, cy), QPointF(cx + half_sz, cy))
                            painter.drawLine(QPointF(cx, cy - half_sz), QPointF(cx, cy - gap))
                            painter.drawLine(QPointF(cx, cy + gap), QPointF(cx, cy + half_sz))

                        pulse_ts = None if _is_playback_rewound else self.tag_pulses.get(sid)
                        if pulse_ts:
                            age = now - pulse_ts
                            if age < 0.8:
                                icol_val = sh.get('interaction_color') or '#00FFC6'
                                icol = QColor(icol_val)
                                anim = sh.get('interaction_animation', 'Pulse')
                                
                                if anim == 'Pulse':
                                    rad = tag_sz/2 + 5 + age * 40
                                    icol.setAlpha(int(255 * (1.0 - age/0.8)))
                                    painter.setPen(QPen(icol, 3))
                                    painter.setBrush(Qt.BrushStyle.NoBrush)
                                    painter.drawEllipse(QPointF(cx, cy), rad, rad)
                                    
                                elif anim == 'Ripple':
                                    for i in range(3):
                                        prog = (age / 0.8 + i * 0.33) % 1.0
                                        rad = tag_sz/2 + prog * 50
                                        alpha = int(255 * (1.0 - prog))
                                        icol.setAlpha(alpha)
                                        painter.setPen(QPen(icol, 2))
                                        painter.setBrush(Qt.BrushStyle.NoBrush)
                                        painter.drawEllipse(QPointF(cx, cy), rad, rad)

                                elif anim == 'Flash':
                                    f_alpha = 255 if age < 0.1 else int(255 * (1.0 - (age-0.1)/0.7))
                                    icol.setAlpha(f_alpha)
                                    painter.setBrush(icol)
                                    painter.setPen(Qt.PenStyle.NoPen)
                                    painter.drawEllipse(QPointF(cx, cy), tag_sz, tag_sz)
                                    
                                elif anim == 'Outline':
                                    glow = int(127 + 127 * np.sin(age * 15))
                                    icol.setAlpha(glow)
                                    painter.setPen(QPen(icol, 4))
                                    painter.setBrush(Qt.BrushStyle.NoBrush)
                                    painter.drawEllipse(QPointF(cx, cy), tag_sz*0.7, tag_sz*0.7)

                            else:
                                self.tag_pulses.pop(sid, None)
                        if self.show_shape_labels and sh.get('show_label', True):
                            tfont = QFont(painter.font())
                            tfont.setPointSize(int(sh.get('text_size', 12)))
                            painter.setFont(tfont)
                            painter.setPen(QColor(sh.get('text_color', '#F0F0F0')))
                            painter.drawText(QPointF(cx + 10, cy - 10), sh.get('label') or 'Tag')

                # Ghost preview while drawing
                if self.draw_mode in ('zone', 'line', 'tag') and (self.draw_temp or self.cursor_norm):
                    ghost_pen = QPen(QColor(255, 255, 255, 180))
                    ghost_pen.setStyle(Qt.PenStyle.DashLine)
                    ghost_pen.setWidth(2)
                    painter.setPen(ghost_pen)
                    painter.setBrush(Qt.BrushStyle.NoBrush)
                    if self.draw_mode == 'zone':
                        pts = list(self.draw_temp)
                        if self.cursor_norm:
                            pts.append(self.cursor_norm)
                        if pts:
                            poly = QPolygonF([QPointF(x_offset + p['x'] * view_w, y_offset + p['y'] * view_h) for p in pts])
                            painter.drawPolyline(poly)
                    elif self.draw_mode == 'line':
                        if len(self.draw_temp) == 1 and self.cursor_norm:
                            p1 = self.draw_temp[0]
                            p2 = self.cursor_norm
                            painter.drawLine(QPointF(x_offset + p1['x'] * view_w, y_offset + p1['y'] * view_h),
                                             QPointF(x_offset + p2['x'] * view_w, y_offset + p2['y'] * view_h))
                    elif self.draw_mode == 'tag' and self.cursor_norm:
                        cx = x_offset + self.cursor_norm['x'] * view_w
                        cy = y_offset + self.cursor_norm['y'] * view_h
                        painter.drawLine(QPointF(cx - 16, cy), QPointF(cx + 16, cy))
                        painter.drawLine(QPointF(cx, cy - 16), QPointF(cx, cy + 16))

                painter.restore()

        # NOTE: Debug Overlay paint is intentionally MOVED to the very
        # end of paintEvent so it draws on top of every other layer
        # (motion boxes, zones, AI detection bounding boxes, REC pill).
        # Previously it was drawn before the AI overlay and got
        # painted-over whenever an object was detected near the
        # top-left corner of the frame, producing a "comes-and-goes"
        # flicker that looked like the overlay was tied to motion.

        # Draw REC indicator when continuous recording is active
        try:
            is_rec = getattr(self, '_camera_recording_flag', False)
            if not is_rec:
                w_parent = self.parent()
                for _ in range(5):
                    if w_parent is None:
                        break
                    if hasattr(w_parent, '_continuous_recording'):
                        is_rec = getattr(w_parent, '_continuous_recording', False)
                        break
                    w_parent = w_parent.parent() if hasattr(w_parent, 'parent') else None
            if is_rec:
                painter.save()
                # Scale indicator proportionally to widget size
                scale = max(0.5, min(2.0, min(self.width(), self.height()) / 360.0))
                font_size = max(7, int(11 * scale))
                dot_r = max(4, int(5 * scale))
                margin = max(6, int(10 * scale))
                text_gap = max(4, int(6 * scale))
                rec_font = QFont("Segoe UI", font_size)
                rec_font.setBold(True)
                painter.setFont(rec_font)
                fm = painter.fontMetrics()
                text_w = fm.horizontalAdvance("REC")
                total_w = dot_r * 2 + text_gap + text_w
                rx = self.width() - total_w - margin
                ry = margin
                painter.setPen(Qt.PenStyle.NoPen)
                painter.setBrush(QColor(220, 38, 38, 240))
                painter.drawEllipse(rx, ry + fm.ascent() // 2 - dot_r, dot_r * 2, dot_r * 2)
                painter.setPen(QColor(255, 80, 80))
                painter.drawText(rx + dot_r * 2 + text_gap, ry + fm.ascent(), "REC")
                painter.restore()
        except Exception:
            pass

        # Draw AI overlay (detections or tracks; style-configurable)
        # During playback rewind: draw historical (playback) detections instead of live ones.
        active_items: List[dict] = []
        active_is_tracks = False
        if _is_playback_rewound:
            active_items = list(getattr(self, 'playback_detections', None) or [])
        else:
            if bool(self.desktop_detections_active):
                if bool(self.desktop_tracking_active) and (self.desktop_tracks or []):
                    active_items = list(self.desktop_tracks or [])
                    active_is_tracks = True
                else:
                    active_items = list(self.desktop_detections or [])
            else:
                active_items = list(self.backend_detections or [])

        if active_items and self.frame_dims[0] > 0:
            painter.save()
            src_w, src_h = self.frame_dims
            scale_x = scaled_img.width() / src_w
            scale_y = scaled_img.height() / src_h

            ds = getattr(self, "detection_settings", None) or {}
            base_color = ds.get("color", QColor(0, 255, 255))
            thickness = int(ds.get("thickness", 2) or 2)
            style = str(ds.get("style", "Box") or "Box")
            anim = str(ds.get("animation", "None") or "None")
            show_labels = bool(ds.get("show_labels", True))
            label_font_size = int(ds.get("label_font_size", 10) or 10)
            label_bg_alpha = int(ds.get("label_bg_alpha", 150) or 150)
            label_bg_alpha = max(0, min(255, label_bg_alpha))

            t = time.time()

            painter.setBrush(Qt.BrushStyle.NoBrush)

            # Label styling
            font = painter.font()
            font.setPointSize(max(7, min(18, label_font_size)))
            font.setBold(True)
            painter.setFont(font)

            for det in active_items:
                bbox = (det or {}).get('bbox', {})
                if not bbox: continue
                
                # Handle dict or list format
                if isinstance(bbox, dict):
                    bx, by = float(bbox.get('x', 0)), float(bbox.get('y', 0))
                    bw, bh = float(bbox.get('w', 0)), float(bbox.get('h', 0))
                elif isinstance(bbox, (list, tuple)) and len(bbox) >= 4:
                    bx, by, bw, bh = float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])
                else:
                    continue

                # Map to widget coords
                draw_x = x_offset + bx * scale_x
                draw_y = y_offset + by * scale_y
                draw_w = bw * scale_x
                draw_h = bh * scale_y

                cx, cy = draw_x + draw_w / 2.0, draw_y + draw_h / 2.0

                # Apply per-track overrides (Desktop tracking only)
                ov = None
                if active_is_tracks:
                    try:
                        tid = int(det.get("track_id"))
                        ov = (getattr(self, "object_overrides", None) or {}).get(tid)
                    except Exception:
                        ov = None
                    try:
                        if isinstance(ov, dict) and bool(ov.get("hidden", False)):
                            continue
                    except Exception:
                        pass

                # --- Color + animation logic (mirrors motion box behavior) ---
                final_color = QColor(base_color)
                if active_is_tracks and isinstance(ov, dict) and ov.get("color"):
                    try:
                        final_color = QColor(str(ov.get("color")))
                    except Exception:
                        final_color = QColor(base_color)
                alpha_mult = 1.0
                size_mult = 1.0

                if anim == "Rainbow":
                    hue = (t * 50) % 360
                    final_color.setHslF(hue / 360.0, 1.0, 0.5)

                if anim == "Pulse":
                    val = (np.sin(t * 5) + 1) / 2
                    alpha_mult = 0.4 + 0.6 * val
                    size_mult = 1.0 + 0.05 * val
                elif anim == "Flash":
                    val = int(t * 8) % 2
                    alpha_mult = 1.0 if val else 0.2
                elif anim == "Glitch":
                    if random.random() > 0.7:
                        draw_x += random.randint(-5, 5)
                        draw_y += random.randint(-5, 5)
                        draw_w += random.randint(-5, 5)
                        draw_h += random.randint(-5, 5)
                        if random.random() > 0.5:
                            final_color = QColor(0, 255, 255) if random.random() > 0.5 else QColor(255, 0, 255)

                final_color.setAlphaF(min(1.0, final_color.alphaF() * alpha_mult))

                if size_mult != 1.0:
                    draw_w *= size_mult
                    draw_h *= size_mult
                    draw_x = cx - draw_w / 2.0
                    draw_y = cy - draw_h / 2.0

                # Optional glow effect
                if anim == "Glow":
                    glow_alpha = int(180 * ((np.sin(t * 4) + 1) / 2))
                    glow_pen = QPen(QColor(final_color.red(), final_color.green(), final_color.blue(), glow_alpha))
                    glow_pen.setWidth(max(3, thickness + 2))
                    painter.setPen(glow_pen)
                    painter.drawRect(QRectF(draw_x, draw_y, draw_w, draw_h))

                # --- Draw box/shape ---
                if style == 'Fill':
                    painter.setPen(Qt.PenStyle.NoPen)
                    fill_color = QColor(final_color)
                    fill_color.setAlpha(min(fill_color.alpha(), 100))
                    painter.setBrush(fill_color)
                else:
                    pen = QPen(final_color)
                    pen.setWidth(max(1, thickness))
                    painter.setPen(pen)
                    painter.setBrush(Qt.BrushStyle.NoBrush)

                # Selected track highlight (Desktop tracking only)
                sel_id = getattr(self, "selected_track_id", None)
                if active_is_tracks and sel_id is not None:
                    try:
                        tid = det.get("track_id")
                        if tid is not None and int(tid) == int(sel_id):
                            # Add a bright outer stroke to indicate selection
                            s_pen = QPen(QColor(255, 255, 255, 220))
                            s_pen.setWidth(max(2, thickness + 2))
                            painter.setPen(s_pen)
                            painter.setBrush(Qt.BrushStyle.NoBrush)
                            painter.drawRect(QRectF(draw_x - 2, draw_y - 2, draw_w + 4, draw_h + 4))
                    except Exception:
                        pass

                if style == 'Box':
                    painter.drawRect(QRectF(draw_x, draw_y, draw_w, draw_h))
                elif style == 'Fill':
                    painter.drawRect(QRectF(draw_x, draw_y, draw_w, draw_h))
                elif style == 'Circle':
                    painter.drawEllipse(QRectF(draw_x, draw_y, draw_w, draw_h))
                elif style == 'Underline':
                    painter.drawLine(int(draw_x), int(draw_y + draw_h), int(draw_x + draw_w), int(draw_y + draw_h))
                elif style == 'Crosshair':
                    len_c = min(draw_w, draw_h) / 2.0
                    painter.drawLine(int(cx - len_c), int(cy), int(cx + len_c), int(cy))
                    painter.drawLine(int(cx), int(cy - len_c), int(cx), int(cy + len_c))
                elif style == 'Bracket':
                    painter.drawLine(int(draw_x), int(draw_y), int(draw_x), int(draw_y + draw_h))
                    painter.drawLine(int(draw_x), int(draw_y), int(draw_x + 5), int(draw_y))
                    painter.drawLine(int(draw_x), int(draw_y + draw_h), int(draw_x + 5), int(draw_y + draw_h))
                    painter.drawLine(int(draw_x + draw_w), int(draw_y), int(draw_x + draw_w), int(draw_y + draw_h))
                    painter.drawLine(int(draw_x + draw_w), int(draw_y), int(draw_x + draw_w - 5), int(draw_y))
                    painter.drawLine(int(draw_x + draw_w), int(draw_y + draw_h), int(draw_x + draw_w - 5), int(draw_y + draw_h))
                elif style == 'Corners':
                    len_x = min(draw_w / 3.0, 20.0)
                    len_y = min(draw_h / 3.0, 20.0)
                    x, y, w, h = int(draw_x), int(draw_y), int(draw_w), int(draw_h)
                    painter.drawLine(x, y, int(x + len_x), y)
                    painter.drawLine(x, y, x, int(y + len_y))
                    painter.drawLine(x + w, y, int(x + w - len_x), y)
                    painter.drawLine(x + w, y, x + w, int(y + len_y))
                    painter.drawLine(x, y + h, int(x + len_x), y + h)
                    painter.drawLine(x, y + h, x, int(y + h - len_y))
                    painter.drawLine(x + w, y + h, int(x + w - len_x), y + h)
                    painter.drawLine(x + w, y + h, x + w, int(y + h - len_y))

                # --- Label ---
                if show_labels:
                    try:
                        base_lbl = str(det.get("class", "Object"))
                    except Exception:
                        base_lbl = "Object"
                    if active_is_tracks and isinstance(ov, dict) and ov.get("name"):
                        try:
                            base_lbl = str(ov.get("name"))
                        except Exception:
                            pass
                    try:
                        confv = float(det.get("confidence", 0) or 0)
                    except Exception:
                        confv = 0.0
                    if active_is_tracks and det.get("track_id") is not None:
                        label = f"{base_lbl} #{int(det.get('track_id'))} {confv:.2f}"
                    else:
                        label = f"{base_lbl} {confv:.2f}"
                    text_rect = painter.boundingRect(
                        QRectF(draw_x, draw_y - 20, 240, 20),
                        Qt.AlignmentFlag.AlignLeft,
                        label,
                    )
                    painter.fillRect(text_rect, QColor(0, 0, 0, label_bg_alpha))
                    painter.setPen(QColor(final_color.red(), final_color.green(), final_color.blue()))
                    painter.drawText(text_rect.topLeft() + QPointF(0, 14), label)

            painter.restore()

        # ── Debug Overlay (drawn LAST so nothing can cover it) ──
        if self.show_debug:
            painter.save()
            font = painter.font()
            font.setFamily("Consolas")
            font.setStyleHint(font.StyleHint.Monospace)
            font.setPointSize(10)
            painter.setFont(font)

            # Focus mode (set during chord/toggle feedback) hides the
            # always-on stream stats so the user sees ONLY the action
            # they're performing.  Outside focus mode we show the
            # standard base lines + any model/load-shed extras.
            if bool(getattr(self, "_debug_focus_mode", False)) and self._debug_extra_lines:
                lines = list(self._debug_extra_lines)
            else:
                lines = [
                    f"FPS: {self.fps}",
                    f"Res: {self.frame_dims[0]}x{self.frame_dims[1]}",
                    f"Motion: {'ON' if self.show_motion else 'OFF'}",
                    f"Detections (desktop/backend): {len(self.desktop_detections)}/{len(self.backend_detections)} (raw {len(getattr(self,'desktop_detections_raw',[]) or [])}/{len(getattr(self,'backend_detections_raw',[]) or [])})",
                ]
                if self._debug_extra_lines:
                    lines.append("")
                    lines.extend(self._debug_extra_lines)
            info_text = "\n".join(lines)

            # Belt-and-braces guard: never render a wordless overlay
            # box. If for any reason all lines stripped to empty,
            # skip the draw entirely (better than the cosmetic glitch
            # of a tiny empty black square).
            if info_text.strip():
                flags = Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop
                rect = painter.boundingRect(10, 20, 540, 240, flags, info_text)

                painter.setPen(Qt.PenStyle.NoPen)
                painter.setBrush(QColor(0, 0, 0, 180))
                painter.drawRect(rect.adjusted(-6, -4, 6, 4))

                painter.setPen(Qt.GlobalColor.green)
                painter.drawText(rect, flags, info_text)
            painter.restore()

        painter.end()

    def _in_parent_resize_zone(self, event) -> bool:
        """Check if the mouse position falls within the top-level widget's resize edge margin."""
        top = self.window()
        if top is None or top is self:
            return False
        margin = getattr(top, 'resize_margin', 0)
        if margin <= 0:
            return False
        win_pos = self.mapTo(top, event.position().toPoint())
        wrect = top.rect()
        px, py = win_pos.x(), win_pos.y()
        return (
            px <= margin or px >= wrect.width() - margin
            or py <= margin or py >= wrect.height() - margin
        )

    def mousePressEvent(self, event):
        if self._in_parent_resize_zone(event):
            event.ignore()
            return
        # Right-click while drawing: cancel and pass through so dragging still works
        if event.button() == Qt.MouseButton.RightButton and self.draw_mode != 'idle':
            self.cancel_draw_mode()
            return super().mousePressEvent(event)
        if event.button() != Qt.MouseButton.LeftButton:
            return super().mousePressEvent(event)
        norm = self._widget_to_norm(event.position())
        if norm:
            self.cursor_norm = norm
        else:
            return super().mousePressEvent(event)
        # Drawing modes
        if self.draw_mode == 'zone':
            if self.draw_temp:
                first = self.draw_temp[0]
                dx = norm['x'] - first['x']
                dy = norm['y'] - first['y']
                if len(self.draw_temp) >= 3 and (dx * dx + dy * dy) ** 0.5 < 0.03:
                    self._add_zone(self.draw_temp.copy())
                    self.cancel_draw_mode()
                    return
            self.draw_temp.append(norm)
            self.update()
            return
        if self.draw_mode == 'line':
            self.draw_temp.append(norm)
            if len(self.draw_temp) == 2:
                self._add_line(self.draw_temp[0], self.draw_temp[1])
                self.cancel_draw_mode()
            else:
                self.update()
            return
        if self.draw_mode == 'tag':
            self._add_tag(norm)
            self.cancel_draw_mode()
            return

        # Edit modes
        vertex_hit = self._hit_test_vertex(event.position())
        if vertex_hit:
            shape_id, idx = vertex_hit
            self.drag_meta = {'type': 'vertex', 'shape_id': shape_id, 'index': idx}
            return
        line_hit = self._hit_test_line_end(event.position())
        if line_hit:
            shape_id, end_key = line_hit
            self.drag_meta = {'type': 'line_end', 'shape_id': shape_id, 'end': end_key}
            return
        shape_hit = self._hit_test_shape(event.position())
        if shape_hit:
            self.drag_meta = {'type': 'shape', 'shape_id': shape_hit, 'last': norm}
            if event.modifiers() & Qt.KeyboardModifier.ShiftModifier:
                if shape_hit not in self.selected_shapes:
                    self.selected_shapes.append(shape_hit)
            else:
                self.selected_shapes = [shape_hit]
            self.update()
            return

        # Desktop tracking selection (click-to-select track box)
        track_hit = self._hit_test_track(event.position())
        if track_hit is not None:
            self.selected_track_id = int(track_hit)
            self.update()
            return

        # If nothing hit, clear selection
        self.selected_shapes = []
        self.selected_track_id = None
        self.update()
        return super().mousePressEvent(event)

    def _hit_test_track(self, pos) -> Optional[int]:
        """
        Hit test Desktop-local tracked boxes in widget coordinates.
        Returns track_id if a track bbox contains the click.
        """
        if not bool(getattr(self, "desktop_tracking_active", False)):
            return None
        if not (getattr(self, "desktop_tracks", None) or []):
            return None
        if not self.last_draw_rect or self.frame_dims[0] <= 0 or self.frame_dims[1] <= 0:
            return None
        try:
            x_off, y_off, draw_w, draw_h = self.last_draw_rect
            px = float(pos.x())
            py = float(pos.y())
            if px < x_off or py < y_off or px > (x_off + draw_w) or py > (y_off + draw_h):
                return None
            scale_x = float(draw_w) / float(self.frame_dims[0])
            scale_y = float(draw_h) / float(self.frame_dims[1])
            # Convert click to source coords
            sx = (px - x_off) / (scale_x if scale_x != 0 else 1.0)
            sy = (py - y_off) / (scale_y if scale_y != 0 else 1.0)
        except Exception:
            return None

        best_tid: Optional[int] = None
        best_area: float = 0.0
        for t in list(self.desktop_tracks or []):
            if not isinstance(t, dict):
                continue
            bbox = (t.get("bbox") or {})
            try:
                bx = float(bbox.get("x", 0.0))
                by = float(bbox.get("y", 0.0))
                bw = float(bbox.get("w", 0.0))
                bh = float(bbox.get("h", 0.0))
                tid = int(t.get("track_id"))
            except Exception:
                continue
            if bw <= 0 or bh <= 0:
                continue
            if (sx >= bx) and (sx <= bx + bw) and (sy >= by) and (sy <= by + bh):
                area = float(bw * bh)
                if best_tid is None or area < best_area:
                    best_tid = tid
                    best_area = area
        return best_tid

    def can_start_window_drag(self, event) -> bool:
        """Only allow window dragging from empty camera-space clicks."""
        if event.button() != Qt.MouseButton.LeftButton:
            return False
        if self.draw_mode != 'idle':
            return False
        pos = event.position()
        norm = self._widget_to_norm(pos)
        if not norm:
            return False
        if self._hit_test_vertex(pos):
            return False
        if self._hit_test_line_end(pos):
            return False
        if self._hit_test_shape(pos):
            return False
        if self._hit_test_track(pos) is not None:
            return False
        return True

    def mouseDoubleClickEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            sid = self._hit_test_shape(event.position())
            if sid:
                self.shape_double_clicked.emit(sid)
                return
        super().mouseDoubleClickEvent(event)

    def mouseMoveEvent(self, event):
        # When near the parent's resize edges, let the parent handle cursor and resize
        if not (hasattr(self, 'drag_meta') and self.drag_meta) and self._in_parent_resize_zone(event):
            self.cursor_norm = None
            event.ignore()
            return

        norm = self._widget_to_norm(event.position())
        if norm:
            self.cursor_norm = norm
        else:
            return super().mouseMoveEvent(event)

        # If we're in drawing mode, handle locally (don't bubble to allow precise cursor/ghost)
        if self.draw_mode != 'idle' and not self.drag_meta:
            self.update()
            return

        if not self.drag_meta:
            self.update()
            return super().mouseMoveEvent(event)
        meta = self.drag_meta
        if meta.get('type') == 'vertex':
            sid, idx = meta['shape_id'], meta['index']
            self.shapes = [
                ({**sh, 'pts': [norm if (i == idx and sh['id'] == sid) else p for i, p in enumerate(sh.get('pts', []))]})
                if sh.get('id') == sid and sh.get('kind') == 'zone' else sh
                for sh in self.shapes
            ]
            self._emit_shapes()
            self.update()
            return
        if meta.get('type') == 'line_end':
            sid, end_key = meta['shape_id'], meta['end']
            self.shapes = [
                ({**sh, end_key: norm} if sh.get('id') == sid and sh.get('kind') == 'line' else sh)
                for sh in self.shapes
            ]
            self._emit_shapes()
            self.update()
            return
        if meta.get('type') == 'shape':
            last = meta.get('last') or norm
            dx = norm['x'] - last['x']
            dy = norm['y'] - last['y']
            meta['last'] = norm
            updated = []
            for sh in self.shapes:
                if sh.get('id') != meta['shape_id']:
                    updated.append(sh)
                    continue
                kind = sh.get('kind')
                if kind == 'zone':
                    pts = [{'x': clamp01(p['x'] + dx), 'y': clamp01(p['y'] + dy)} for p in sh.get('pts', [])]
                    updated.append({**sh, 'pts': pts})
                elif kind == 'line':
                    p1 = sh.get('p1') or {'x': 0, 'y': 0}
                    p2 = sh.get('p2') or {'x': 0, 'y': 0}
                    updated.append({**sh,
                                    'p1': {'x': clamp01(p1['x'] + dx), 'y': clamp01(p1['y'] + dy)},
                                    'p2': {'x': clamp01(p2['x'] + dx), 'y': clamp01(p2['y'] + dy)}})
                elif kind == 'tag':
                    anchor = sh.get('anchor') or {'x': 0, 'y': 0}
                    updated.append({**sh, 'anchor': {'x': clamp01(anchor['x'] + dx), 'y': clamp01(anchor['y'] + dy)}})
            self.shapes = updated
            self._emit_shapes()
            self.update()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if self.drag_meta:
            self.drag_meta = None
        parent = self.parent()
        if parent and hasattr(parent, 'is_resizing') and getattr(parent, 'is_resizing', False):
            event.ignore()
            return
        super().mouseReleaseEvent(event)

    def leaveEvent(self, event):
        self.cursor_norm = None
        return super().leaveEvent(event)

    def keyPressEvent(self, event):
        if event.key() == Qt.Key.Key_Escape:
            self.cancel_draw_mode()
            self.drag_meta = None
            self.update()
            return
        if event.key() in (Qt.Key.Key_Backspace, Qt.Key.Key_Delete):
            if self.selected_shapes:
                self.shapes = [s for s in self.shapes if s.get('id') not in self.selected_shapes]
                self.selected_shapes = []
                self._emit_shapes()
                self.update()
                return
        return super().keyPressEvent(event)


class MotionSettingsDialog(QDialog):
    settings_changed = Signal(dict)

    def __init__(self, current_settings, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Motion Box Settings")
        self.setFixedSize(300, 250)
        # Ensure the dialog stays on top if the parent is pinned, and tool tool window style
        self.setWindowFlags(Qt.WindowType.Tool | Qt.WindowType.WindowStaysOnTopHint)
        
        self.settings = current_settings.copy()
        
        layout = QFormLayout(self)
        
        # Style
        self.style_combo = QComboBox()
        self.style_combo.addItems(["Box", "Fill", "Corners", "Circle", "Bracket", "Underline", "Crosshair"])
        self.style_combo.setCurrentText(self.settings['style'])
        self.style_combo.currentTextChanged.connect(self.update_style)
        layout.addRow("Style:", self.style_combo)
        
        # Animation
        self.anim_combo = QComboBox()
        self.anim_combo.addItems(["None", "Pulse", "Flash", "Glitch", "Rainbow"])
        self.anim_combo.setCurrentText(self.settings.get('animation', 'None'))
        self.anim_combo.currentTextChanged.connect(self.update_animation)
        layout.addRow("Animation:", self.anim_combo)

        # Options
        self.trails_check = QCheckBox("Show Trails")
        self.trails_check.setChecked(self.settings.get('trails', False))
        self.trails_check.toggled.connect(lambda x: self.update_bool('trails', x))
        layout.addRow("", self.trails_check)

        self.speed_color_check = QCheckBox("Color by Speed")
        self.speed_color_check.setChecked(self.settings.get('color_speed', False))
        self.speed_color_check.toggled.connect(lambda x: self.update_bool('color_speed', x))
        layout.addRow("", self.speed_color_check)

        # Sensitivity
        self.sens_slider = QSlider(Qt.Orientation.Horizontal)
        self.sens_slider.setRange(1, 100)
        self.sens_slider.setValue(self.settings.get('sensitivity', 50))
        self.sens_slider.valueChanged.connect(self.update_sensitivity)
        layout.addRow("Sensitivity:", self.sens_slider)

        # Merge / Size
        self.size_slider = QSlider(Qt.Orientation.Horizontal)
        self.size_slider.setRange(0, 100)
        self.size_slider.setValue(self.settings.get('merge_size', 0))
        self.size_slider.valueChanged.connect(self.update_merge_size)
        layout.addRow("Box Size/Merge:", self.size_slider)

        # Color
        self.color_btn = QPushButton()
        self._refresh_color_button()
        self.color_btn.clicked.connect(self.pick_color)
        layout.addRow("Color:", self.color_btn)
        
        # Thickness
        self.thickness_spin = QSpinBox()
        self.thickness_spin.setRange(1, 10)
        self.thickness_spin.setValue(self.settings['thickness'])
        self.thickness_spin.valueChanged.connect(self.update_thickness)
        layout.addRow("Thickness:", self.thickness_spin)
        
        # Close button
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.close)
        layout.addRow(close_btn)
        
    def update_style(self, text):
        self.settings['style'] = text
        self.settings_changed.emit(self.settings)

    def update_animation(self, text):
        self.settings['animation'] = text
        self.settings_changed.emit(self.settings)
        
    def update_bool(self, key, val):
        self.settings[key] = val
        self.settings_changed.emit(self.settings)

    def update_sensitivity(self, val):
        self.settings['sensitivity'] = val
        self.settings_changed.emit(self.settings)

    def update_merge_size(self, val):
        self.settings['merge_size'] = val
        self.settings_changed.emit(self.settings)

    def update_thickness(self, val):
        self.settings['thickness'] = val
        self.settings_changed.emit(self.settings)
        
    def _refresh_color_button(self) -> None:
        try:
            c = self.settings.get("color")
            if not isinstance(c, QColor):
                c = QColor(str(c or ""))
            if not c.isValid():
                c = QColor(255, 0, 0)
            self.settings["color"] = c
            # Prefer friendly name when it matches one of our presets.
            name = None
            try:
                cur = c.name().lower()
                for label, hx in (BBOX_COLOR_CHOICES or []):
                    if str(hx).lower() == cur:
                        name = str(label)
                        break
            except Exception:
                name = None
            self.color_btn.setText(f"{name} ({c.name().upper()})" if name else c.name().upper())
            self.color_btn.setStyleSheet(
                f"background-color: {c.name()};"
                "border: 1px solid #666;"
                "padding: 2px 6px;"
                "min-height: 18px;"
            )
        except Exception:
            # Best effort; don't crash the dialog if something is odd.
            try:
                self.color_btn.setText("Pick…")
            except Exception:
                pass

    def _set_color(self, c: QColor) -> None:
        try:
            if c and c.isValid():
                self.settings["color"] = c
                self._refresh_color_button()
                # Emit a copy so downstream consumers don't accidentally share our mutable dict.
                self.settings_changed.emit(dict(self.settings))
        except Exception:
            pass

    def _open_custom_color_picker(self) -> None:
        """
        Non-blocking custom picker that auto-closes shortly after the user stops changing the color.
        This avoids the 'must manually close' UX while still allowing slider-based picking.
        """
        try:
            # Local imports keep module import surface stable.
            from PySide6.QtWidgets import QColorDialog

            base = self.settings.get("color")
            if not isinstance(base, QColor):
                base = QColor(str(base or ""))
            if not base.isValid():
                base = QColor(255, 0, 0)

            dlg = QColorDialog(base, self)
            dlg.setWindowTitle("Select Motion Color")
            dlg.setWindowFlags(Qt.WindowType.Tool | Qt.WindowType.WindowStaysOnTopHint)
            # Don't require an OK button; we treat changes as the selection.
            try:
                dlg.setOption(QColorDialog.ColorDialogOption.NoButtons, True)
                dlg.setOption(QColorDialog.ColorDialogOption.DontUseNativeDialog, True)
            except Exception:
                pass

            # Hold a strong ref so it isn't GC'd.
            self._motion_color_dlg = dlg

            # Debounced auto-close after user stops changing for a moment.
            t = QTimer(dlg)
            t.setSingleShot(True)
            self._motion_color_close_timer = t

            def on_change(col: QColor):
                self._set_color(col)
                try:
                    # Reset debounce window.
                    self._motion_color_close_timer.start(350)
                except Exception:
                    pass

            def on_close():
                try:
                    self._motion_color_close_timer.stop()
                except Exception:
                    pass
                try:
                    self._motion_color_dlg = None
                except Exception:
                    pass

            dlg.currentColorChanged.connect(on_change)
            dlg.finished.connect(lambda _code=0: on_close())

            dlg.open()
        except Exception:
            # Fallback to blocking picker if anything goes wrong.
            try:
                color = QColorDialog.getColor(self.settings["color"], self, "Select Motion Color")
                if color.isValid():
                    self._set_color(color)
            except Exception:
                pass

    def pick_color(self):
        # Fast palette popup (shows selected color; auto-closes when chosen) + optional custom picker.
        try:
            from PySide6.QtWidgets import QMenu
            from PySide6.QtGui import QAction, QPixmap, QIcon
            from PySide6.QtCore import QPoint

            cur = self.settings.get("color")
            if not isinstance(cur, QColor):
                cur = QColor(str(cur or ""))
            cur_hex = cur.name().lower() if (cur and cur.isValid()) else ""

            # Ensure we don't leave multiple menus around (users reported stacks).
            try:
                old = getattr(self, "_color_menu", None)
                if old is not None:
                    old.close()
                    old.deleteLater()
            except Exception:
                pass

            menu = QMenu(self)
            # Popup menus should always disappear on click-out; don't force Tool windows.
            self._color_menu = menu
            menu.aboutToHide.connect(lambda: setattr(self, "_color_menu", None))

            # Presets
            for label, hx in (BBOX_COLOR_CHOICES or []):
                h = str(hx or "").strip()
                if not h:
                    continue
                c = QColor(h)
                if not c.isValid():
                    continue
                px = QPixmap(14, 14)
                px.fill(c)
                is_sel = (cur_hex == h.lower())
                text = f"✓ {label}" if is_sel else str(label)
                act = QAction(QIcon(px), text, menu)
                # Force-close on trigger (prevents sticky menus in some WM/Qt combos)
                act.triggered.connect(
                    lambda _chk=False, col=c, m=menu: (m.close(), self._set_color(QColor(col)))
                )
                menu.addAction(act)

            menu.addSeparator()
            custom = QAction("Custom…", menu)
            custom.triggered.connect(lambda _chk=False, m=menu: (m.close(), self._open_custom_color_picker()))
            menu.addAction(custom)

            anchor = self.color_btn.mapToGlobal(QPoint(0, self.color_btn.height()))
            menu.popup(anchor)
        except Exception:
            # Fall back to basic modal picker
            self._open_custom_color_picker()


class DetectionOverlaySettingsDialog(QDialog):
    settings_changed = Signal(dict)

    def __init__(self, current_settings, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Detection Bounding Box Settings")
        self.setFixedSize(360, 470)
        self.setWindowFlags(Qt.WindowType.Tool | Qt.WindowType.WindowStaysOnTopHint)

        self.settings = (current_settings or {}).copy()
        layout = QFormLayout(self)

        # Style
        self.style_combo = QComboBox()
        self.style_combo.addItems(["Box", "Fill", "Corners", "Circle", "Bracket", "Underline", "Crosshair"])
        self.style_combo.setCurrentText(str(self.settings.get("style", "Box")))
        self.style_combo.currentTextChanged.connect(lambda t: self._set("style", t))
        layout.addRow("Style:", self.style_combo)

        # Animation
        self.anim_combo = QComboBox()
        self.anim_combo.addItems(["None", "Pulse", "Flash", "Glitch", "Rainbow", "Glow"])
        self.anim_combo.setCurrentText(str(self.settings.get("animation", "None")))
        self.anim_combo.currentTextChanged.connect(lambda t: self._set("animation", t))
        layout.addRow("Animation:", self.anim_combo)

        # Color
        self.color_btn = QPushButton()
        col = self.settings.get("color", QColor(0, 255, 255))
        if not isinstance(col, QColor):
            col = QColor(0, 255, 255)
        self.settings["color"] = col
        self._refresh_color_button()
        self.color_btn.clicked.connect(self.pick_color)
        layout.addRow("Color:", self.color_btn)

        # Thickness
        self.thickness_spin = QSpinBox()
        self.thickness_spin.setRange(1, 10)
        self.thickness_spin.setValue(int(self.settings.get("thickness", 2) or 2))
        self.thickness_spin.valueChanged.connect(lambda v: self._set("thickness", int(v)))
        layout.addRow("Thickness:", self.thickness_spin)

        # Labels
        self.labels_check = QCheckBox("Show labels")
        self.labels_check.setChecked(bool(self.settings.get("show_labels", True)))
        self.labels_check.toggled.connect(lambda x: self._set("show_labels", bool(x)))
        layout.addRow("", self.labels_check)

        self.font_spin = QSpinBox()
        self.font_spin.setRange(7, 18)
        self.font_spin.setValue(int(self.settings.get("label_font_size", 10) or 10))
        self.font_spin.valueChanged.connect(lambda v: self._set("label_font_size", int(v)))
        layout.addRow("Label size:", self.font_spin)

        self.bg_slider = QSlider(Qt.Orientation.Horizontal)
        self.bg_slider.setRange(0, 255)
        self.bg_slider.setValue(int(self.settings.get("label_bg_alpha", 150) or 150))
        self.bg_slider.valueChanged.connect(lambda v: self._set("label_bg_alpha", int(v)))
        layout.addRow("Label bg α:", self.bg_slider)

        # Shape interaction (zones/lines/tags)
        self.interact_check = QCheckBox("Interact with zones/lines/tags")
        self.interact_check.setChecked(bool(self.settings.get("interact_with_shapes", True)))
        self.interact_check.toggled.connect(lambda x: self._set("interact_with_shapes", bool(x)))
        layout.addRow("", self.interact_check)

        self.zone_check = QCheckBox("Zones")
        self.zone_check.setChecked(bool(self.settings.get("interact_zone", True)))
        self.zone_check.toggled.connect(lambda x: self._set("interact_zone", bool(x)))
        self.line_check = QCheckBox("Lines")
        self.line_check.setChecked(bool(self.settings.get("interact_line", True)))
        self.line_check.toggled.connect(lambda x: self._set("interact_line", bool(x)))
        self.tag_check = QCheckBox("Tags")
        self.tag_check.setChecked(bool(self.settings.get("interact_tag", True)))
        self.tag_check.toggled.connect(lambda x: self._set("interact_tag", bool(x)))
        row = QHBoxLayout()
        row.addWidget(self.zone_check)
        row.addWidget(self.line_check)
        row.addWidget(self.tag_check)
        layout.addRow("Interact:", row)

        self.cooldown_spin = QSpinBox()
        self.cooldown_spin.setRange(0, 5000)
        self.cooldown_spin.setSingleStep(50)
        self.cooldown_spin.setSuffix(" ms")
        self.cooldown_spin.setValue(int(self.settings.get("interaction_cooldown_ms", 250) or 0))
        self.cooldown_spin.valueChanged.connect(lambda v: self._set("interaction_cooldown_ms", int(v)))
        layout.addRow("Cooldown:", self.cooldown_spin)

        # Interaction distances (normalized)
        self.interact_line_dist = QDoubleSpinBox()
        self.interact_line_dist.setRange(0.0, 0.5)
        self.interact_line_dist.setSingleStep(0.005)
        self.interact_line_dist.setDecimals(3)
        self.interact_line_dist.setValue(float(self.settings.get("interact_line_dist", 0.02) or 0.02))
        self.interact_line_dist.valueChanged.connect(lambda v: self._set("interact_line_dist", float(v)))
        layout.addRow("Interact line dist (norm):", self.interact_line_dist)

        self.interact_tag_dist = QDoubleSpinBox()
        self.interact_tag_dist.setRange(0.0, 0.5)
        self.interact_tag_dist.setSingleStep(0.005)
        self.interact_tag_dist.setDecimals(3)
        self.interact_tag_dist.setValue(float(self.settings.get("interact_tag_dist", 0.03) or 0.03))
        self.interact_tag_dist.valueChanged.connect(lambda v: self._set("interact_tag_dist", float(v)))
        layout.addRow("Interact tag dist (norm):", self.interact_tag_dist)

        self.interact_zone_margin = QDoubleSpinBox()
        self.interact_zone_margin.setRange(0.0, 0.5)
        self.interact_zone_margin.setSingleStep(0.005)
        self.interact_zone_margin.setDecimals(3)
        self.interact_zone_margin.setValue(float(self.settings.get("interact_zone_margin", 0.0) or 0.0))
        self.interact_zone_margin.valueChanged.connect(lambda v: self._set("interact_zone_margin", float(v)))
        layout.addRow("Interact zone margin (norm):", self.interact_zone_margin)

        # ROI filter
        self.roi_check = QCheckBox("ROI filter: only keep detections in/near shapes")
        self.roi_check.setChecked(bool(self.settings.get("roi_enabled", False)))
        self.roi_check.toggled.connect(lambda x: self._set("roi_enabled", bool(x)))
        layout.addRow("", self.roi_check)

        self.roi_selected_only = QCheckBox("ROI uses selected shapes only")
        self.roi_selected_only.setChecked(bool(self.settings.get("roi_selected_only", False)))
        self.roi_selected_only.toggled.connect(lambda x: self._set("roi_selected_only", bool(x)))
        layout.addRow("", self.roi_selected_only)

        self.roi_zone = QCheckBox("Zones (inside + margin)")
        self.roi_zone.setChecked(bool(self.settings.get("roi_zone", True)))
        self.roi_zone.toggled.connect(lambda x: self._set("roi_zone", bool(x)))
        self.roi_line = QCheckBox("Lines (within distance)")
        self.roi_line.setChecked(bool(self.settings.get("roi_line", True)))
        self.roi_line.toggled.connect(lambda x: self._set("roi_line", bool(x)))
        self.roi_tag = QCheckBox("Tags (within distance)")
        self.roi_tag.setChecked(bool(self.settings.get("roi_tag", True)))
        self.roi_tag.toggled.connect(lambda x: self._set("roi_tag", bool(x)))
        rrow = QHBoxLayout()
        rrow.addWidget(self.roi_zone)
        rrow.addWidget(self.roi_line)
        rrow.addWidget(self.roi_tag)
        layout.addRow("ROI shapes:", rrow)

        self.roi_line_dist = QDoubleSpinBox()
        self.roi_line_dist.setRange(0.0, 0.5)
        self.roi_line_dist.setSingleStep(0.005)
        self.roi_line_dist.setDecimals(3)
        self.roi_line_dist.setValue(float(self.settings.get("roi_line_dist", 0.02) or 0.02))
        self.roi_line_dist.valueChanged.connect(lambda v: self._set("roi_line_dist", float(v)))
        layout.addRow("Line dist (norm):", self.roi_line_dist)

        self.roi_tag_dist = QDoubleSpinBox()
        self.roi_tag_dist.setRange(0.0, 0.5)
        self.roi_tag_dist.setSingleStep(0.005)
        self.roi_tag_dist.setDecimals(3)
        self.roi_tag_dist.setValue(float(self.settings.get("roi_tag_dist", 0.03) or 0.03))
        self.roi_tag_dist.valueChanged.connect(lambda v: self._set("roi_tag_dist", float(v)))
        layout.addRow("Tag dist (norm):", self.roi_tag_dist)

        self.roi_zone_margin = QDoubleSpinBox()
        self.roi_zone_margin.setRange(0.0, 0.5)
        self.roi_zone_margin.setSingleStep(0.005)
        self.roi_zone_margin.setDecimals(3)
        self.roi_zone_margin.setValue(float(self.settings.get("roi_zone_margin", 0.0) or 0.0))
        self.roi_zone_margin.valueChanged.connect(lambda v: self._set("roi_zone_margin", float(v)))
        layout.addRow("Zone margin (norm):", self.roi_zone_margin)

        # ROI inference crop (Desktop-local YOLO only)
        self.roi_crop_infer = QCheckBox("Crop Desktop inference to ROI (faster)")
        self.roi_crop_infer.setChecked(bool(self.settings.get("roi_crop_inference", False)))
        self.roi_crop_infer.toggled.connect(lambda x: self._set("roi_crop_inference", bool(x)))
        layout.addRow("", self.roi_crop_infer)

        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.close)
        layout.addRow(close_btn)

        # Enable/disable per-type toggles based on main switch
        self.interact_check.toggled.connect(self._update_interaction_enabled)
        self._update_interaction_enabled(self.interact_check.isChecked())

        self.roi_check.toggled.connect(self._update_roi_enabled)
        self._update_roi_enabled(self.roi_check.isChecked())

    def _update_interaction_enabled(self, enabled: bool):
        try:
            self.zone_check.setEnabled(bool(enabled))
            self.line_check.setEnabled(bool(enabled))
            self.tag_check.setEnabled(bool(enabled))
            self.cooldown_spin.setEnabled(bool(enabled))
            self.interact_line_dist.setEnabled(bool(enabled) and bool(self.line_check.isChecked()))
            self.interact_tag_dist.setEnabled(bool(enabled) and bool(self.tag_check.isChecked()))
            self.interact_zone_margin.setEnabled(bool(enabled) and bool(self.zone_check.isChecked()))
        except Exception:
            pass

    def _update_roi_enabled(self, enabled: bool):
        try:
            self.roi_selected_only.setEnabled(bool(enabled))
            self.roi_zone.setEnabled(bool(enabled))
            self.roi_line.setEnabled(bool(enabled))
            self.roi_tag.setEnabled(bool(enabled))
            self.roi_line_dist.setEnabled(bool(enabled) and bool(self.roi_line.isChecked()))
            self.roi_tag_dist.setEnabled(bool(enabled) and bool(self.roi_tag.isChecked()))
            self.roi_zone_margin.setEnabled(bool(enabled) and bool(self.roi_zone.isChecked()))
        except Exception:
            pass

    def _set(self, key: str, val):
        self.settings[key] = val
        self.settings_changed.emit(self.settings)

    def pick_color(self):
        # Fast palette popup (shows selected color; auto-closes when chosen) + optional custom picker.
        try:
            from PySide6.QtWidgets import QMenu, QColorDialog
            from PySide6.QtGui import QAction, QPixmap, QIcon
            from PySide6.QtCore import QPoint

            cur = self.settings.get("color")
            if not isinstance(cur, QColor):
                cur = QColor(str(cur or ""))
            cur_hex = cur.name().lower() if (cur and cur.isValid()) else ""

            def set_color(c: QColor) -> None:
                try:
                    if c and c.isValid():
                        self.settings["color"] = c
                        self._refresh_color_button()
                        self.settings_changed.emit(dict(self.settings))
                except Exception:
                    pass

            def open_custom() -> None:
                try:
                    # Close any previous custom dialog to avoid stacks.
                    try:
                        old = getattr(self, "_det_color_dlg", None)
                        if old is not None:
                            old.close()
                            old.deleteLater()
                    except Exception:
                        pass

                    base = self.settings.get("color")
                    if not isinstance(base, QColor):
                        base = QColor(str(base or ""))
                    if not base.isValid():
                        base = QColor(0, 255, 255)

                    dlg = QColorDialog(base, self)
                    dlg.setWindowTitle("Select Detection Color")
                    dlg.setWindowFlags(Qt.WindowType.Tool | Qt.WindowType.WindowStaysOnTopHint)
                    try:
                        dlg.setOption(QColorDialog.ColorDialogOption.NoButtons, True)
                        dlg.setOption(QColorDialog.ColorDialogOption.DontUseNativeDialog, True)
                    except Exception:
                        pass

                    self._det_color_dlg = dlg
                    t = QTimer(dlg)
                    t.setSingleShot(True)
                    self._det_color_close_timer = t

                    def on_change(col: QColor):
                        set_color(col)
                        try:
                            self._det_color_close_timer.start(350)
                        except Exception:
                            pass

                    def on_close():
                        try:
                            self._det_color_close_timer.stop()
                        except Exception:
                            pass
                        try:
                            self._det_color_dlg = None
                        except Exception:
                            pass

                    dlg.currentColorChanged.connect(on_change)
                    dlg.finished.connect(lambda _code=0: on_close())
                    dlg.open()
                except Exception:
                    # last resort
                    try:
                        c = QColorDialog.getColor(self.settings.get("color", QColor(0, 255, 255)), self, "Select Detection Color")
                        if c.isValid():
                            set_color(c)
                    except Exception:
                        pass

            # Ensure we don't leave multiple menus around (users reported stacks).
            try:
                old_menu = getattr(self, "_color_menu", None)
                if old_menu is not None:
                    old_menu.close()
                    old_menu.deleteLater()
            except Exception:
                pass

            menu = QMenu(self)
            self._color_menu = menu
            menu.aboutToHide.connect(lambda: setattr(self, "_color_menu", None))

            for label, hx in (BBOX_COLOR_CHOICES or []):
                h = str(hx or "").strip()
                if not h:
                    continue
                c = QColor(h)
                if not c.isValid():
                    continue
                px = QPixmap(14, 14)
                px.fill(c)
                is_sel = (cur_hex == h.lower())
                text = f"✓ {label}" if is_sel else str(label)
                act = QAction(QIcon(px), text, menu)
                act.triggered.connect(
                    lambda _chk=False, col=c, m=menu: (m.close(), set_color(QColor(col)))
                )
                menu.addAction(act)

            menu.addSeparator()
            custom = QAction("Custom…", menu)
            custom.triggered.connect(lambda _chk=False, m=menu: (m.close(), open_custom()))
            menu.addAction(custom)

            anchor = self.color_btn.mapToGlobal(QPoint(0, self.color_btn.height()))
            menu.popup(anchor)
        except Exception:
            # Fallback to previous modal picker behavior
            try:
                color = QColorDialog.getColor(self.settings["color"], self, "Select Detection Color")
                if color.isValid():
                    self.settings["color"] = color
                    self._refresh_color_button()
                    self.settings_changed.emit(dict(self.settings))
            except Exception:
                pass

    def _refresh_color_button(self) -> None:
        try:
            c = self.settings.get("color")
            if not isinstance(c, QColor):
                c = QColor(str(c or ""))
            if not c.isValid():
                c = QColor(0, 255, 255)
            self.settings["color"] = c
            name = None
            try:
                cur = c.name().lower()
                for label, hx in (BBOX_COLOR_CHOICES or []):
                    if str(hx).lower() == cur:
                        name = str(label)
                        break
            except Exception:
                name = None
            self.color_btn.setText(f"{name} ({c.name().upper()})" if name else c.name().upper())
            self.color_btn.setStyleSheet(
                f"background-color: {c.name()};"
                "border: 1px solid #666;"
                "padding: 2px 6px;"
                "min-height: 18px;"
            )
        except Exception:
            try:
                self.color_btn.setText("Pick…")
            except Exception:
                pass


class TrackedObjectSettingsDialog(QDialog):
    """
    Rename/color/hide settings for a selected Desktop-local tracked object.

    Saved per-camera per-track_id to data/desktop_object_overrides.json.
    Track IDs may reset between sessions or when the tracker is reset/switched.
    """

    def __init__(self, camera_id: str, track_id: int, current_override: Dict[str, object], parent=None):
        super().__init__(parent)
        # Local import to avoid relying on large module-level Qt imports.
        from PySide6.QtWidgets import QDialogButtonBox

        self.setWindowTitle(f"Tracked Object Settings - {camera_id} #{int(track_id)}")
        self.setWindowFlags(Qt.WindowType.Tool | Qt.WindowType.WindowStaysOnTopHint)
        self.setFixedSize(360, 220)

        self.result_override: Optional[dict] = None

        ov = dict(current_override or {})
        name_val = str(ov.get("name") or "").strip()
        color_val = str(ov.get("color") or "").strip()
        hidden_val = bool(ov.get("hidden", False))

        root = QVBoxLayout(self)
        form = QFormLayout()
        root.addLayout(form)

        self.name_edit = QLineEdit()
        self.name_edit.setText(name_val)
        self.name_edit.setPlaceholderText("Optional friendly name (e.g. 'Bob')")
        form.addRow("Name", self.name_edit)

        self.hidden_check = QCheckBox("Hide this tracked object")
        self.hidden_check.setChecked(hidden_val)
        form.addRow("", self.hidden_check)

        self._color = QColor(color_val) if color_val else QColor()
        self.color_btn = QPushButton()
        self._refresh_color_button()
        self.color_btn.clicked.connect(self._pick_color)
        form.addRow("Color", self.color_btn)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Save | QDialogButtonBox.StandardButton.Cancel)
        root.addWidget(buttons)
        buttons.accepted.connect(self._on_save)
        buttons.rejected.connect(self.reject)

        # Extra action: let user forget this saved mapping when track IDs get reused.
        try:
            forget_btn = buttons.addButton("Forget saved settings", QDialogButtonBox.ButtonRole.ResetRole)
            if forget_btn is not None:
                forget_btn.clicked.connect(self._on_forget)
        except Exception:
            pass

    def _refresh_color_button(self):
        if self._color and self._color.isValid():
            self.color_btn.setText(self._color.name())
            self.color_btn.setStyleSheet(f"background-color: {self._color.name()};")
        else:
            self.color_btn.setText("Default (overlay color)")
            self.color_btn.setStyleSheet("")

    def _pick_color(self):
        base = self._color if (self._color and self._color.isValid()) else QColor(0, 255, 255)
        c = QColorDialog.getColor(base, self, "Select Track Color")
        if c.isValid():
            self._color = c
            self._refresh_color_button()

    def _on_save(self):
        name = str(self.name_edit.text() or "").strip()
        hidden = bool(self.hidden_check.isChecked())
        out: dict = {"hidden": hidden}
        if name:
            out["name"] = name
        if self._color and self._color.isValid():
            out["color"] = self._color.name()
        self.result_override = out
        self.accept()

    def _on_forget(self):
        # Signal caller to delete this override
        self.result_override = {"__delete__": True}
        self.accept()


class MotionWatchDialog(QDialog):
    """
    Lightweight motion watch configuration dialog.
    Allows user control over delay, crop, resolution, overlays, save path, and shape filters.
    """
    def __init__(self, current_settings: Dict[str, object], parent=None):
        super().__init__(parent)
        self.setWindowTitle("Motion Watch")
        self.setFixedSize(420, 700)
        self.setWindowFlags(Qt.WindowType.Tool | Qt.WindowType.WindowStaysOnTopHint)
        self.settings = current_settings.copy()
        duration_val = int(self.settings.get("duration_sec", 30))

        form = QFormLayout(self)

        self.duration_spin = QSpinBox()
        self.duration_spin.setRange(1, 36000)
        self.duration_unit = QComboBox()
        self.duration_unit.addItems(["Seconds", "Minutes", "Hours", "Infinite"])
        default_unit = self.settings.get("duration_unit", "Seconds")
        if duration_val < 0:
            default_unit = "Infinite"
        if default_unit not in {"Seconds", "Minutes", "Hours", "Infinite"}:
            default_unit = "Seconds"
        self.duration_unit.setCurrentText(default_unit)
        # Set spin based on stored seconds and unit
        if default_unit == "Seconds":
            spin_val = max(1, abs(duration_val) if duration_val > 0 else 30)
        elif default_unit == "Minutes":
            spin_val = max(1, max(1, abs(duration_val)) // 60 if duration_val > 0 else 5)
        elif default_unit == "Hours":
            spin_val = max(1, max(1, abs(duration_val)) // 3600 if duration_val > 0 else 1)
        else:  # Infinite
            spin_val = 1
        self.duration_spin.setValue(spin_val)
        self.duration_unit.currentTextChanged.connect(self._on_unit_change)
        # Apply initial enable/disable based on unit
        self._on_unit_change(default_unit)
        dur_row = QHBoxLayout()
        dur_row.addWidget(self.duration_spin)
        dur_row.addWidget(self.duration_unit)
        form.addRow("Watch duration", dur_row)

        self.delay_spin = QSpinBox()
        self.delay_spin.setRange(0, 5000)
        self.delay_spin.setSuffix(" ms")
        self.delay_spin.setValue(int(self.settings.get("trigger_delay_ms", 500)))
        form.addRow("Trigger delay", self.delay_spin)

        self.crop_w_spin = QSpinBox()
        self.crop_w_spin.setRange(0, 3840)
        self.crop_w_spin.setValue(int(self.settings.get("crop_w", 0)))
        self.crop_h_spin = QSpinBox()
        self.crop_h_spin.setRange(0, 2160)
        self.crop_h_spin.setValue(int(self.settings.get("crop_h", 0)))
        crop_row = QHBoxLayout()
        crop_row.addWidget(self.crop_w_spin)
        crop_row.addWidget(QLabel("x"))
        crop_row.addWidget(self.crop_h_spin)
        form.addRow("Crop size (px, 0=full)", crop_row)

        self.resize_w_spin = QSpinBox()
        self.resize_w_spin.setRange(0, 3840)
        self.resize_w_spin.setValue(int(self.settings.get("resize_w", 0)))
        form.addRow("Output width (0=keep)", self.resize_w_spin)

        self.overlay_check = QCheckBox("Include overlays (zones/lines/tags/motion)")
        self.overlay_check.setChecked(bool(self.settings.get("include_overlays", True)))
        form.addRow("", self.overlay_check)

        self.save_dir_edit = QLineEdit(self.settings.get("save_dir", "captures/motion_watch"))
        browse_btn = QPushButton("Browse…")
        browse_btn.clicked.connect(self._pick_dir)
        dir_row = QHBoxLayout()
        dir_row.addWidget(self.save_dir_edit)
        dir_row.addWidget(browse_btn)
        form.addRow("Save directory", dir_row)

        # Shape filters
        self.zone_check = QCheckBox("Zones")
        self.zone_check.setChecked(bool(self.settings.get("allow_zone", True)))
        self.line_check = QCheckBox("Lines")
        self.line_check.setChecked(bool(self.settings.get("allow_line", True)))
        self.tag_check = QCheckBox("Tags")
        self.tag_check.setChecked(bool(self.settings.get("allow_tag", True)))
        filter_row = QHBoxLayout()
        for w in (self.zone_check, self.line_check, self.tag_check):
            filter_row.addWidget(w)
        form.addRow("Trigger on", filter_row)

        self.cooldown_spin = QSpinBox()
        self.cooldown_spin.setRange(0, 60)
        self.cooldown_spin.setSuffix(" s")
        self.cooldown_spin.setValue(int(self.settings.get("cooldown_sec", 3)))
        form.addRow("Cooldown between shots", self.cooldown_spin)

        # Clip Recording group
        clip_group = QGroupBox("Clip Recording")
        clip_layout = QFormLayout()
        self.clip_enabled_check = QCheckBox("Enable video clip capture")
        self.clip_enabled_check.setChecked(bool(self.settings.get("clip_enabled", False)))
        clip_layout.addRow("", self.clip_enabled_check)

        self.clip_duration_spin = QSpinBox()
        self.clip_duration_spin.setRange(3, 60)
        self.clip_duration_spin.setSuffix(" s")
        self.clip_duration_spin.setValue(int(self.settings.get("clip_duration_sec", 10)))
        clip_layout.addRow("Clip duration", self.clip_duration_spin)

        self.clip_pre_roll_spin = QSpinBox()
        self.clip_pre_roll_spin.setRange(0, 10)
        self.clip_pre_roll_spin.setSuffix(" s")
        self.clip_pre_roll_spin.setValue(int(self.settings.get("clip_pre_roll_sec", 5)))
        clip_layout.addRow("Pre-roll", self.clip_pre_roll_spin)

        self.clip_resize_spin = QSpinBox()
        self.clip_resize_spin.setRange(0, 1920)
        self.clip_resize_spin.setValue(int(self.settings.get("clip_resize_w", 640)))
        clip_layout.addRow("Clip width (0=native)", self.clip_resize_spin)

        self.clip_quality_spin = QSpinBox()
        self.clip_quality_spin.setRange(18, 35)
        self.clip_quality_spin.setValue(int(self.settings.get("clip_quality", 23)))
        clip_layout.addRow("Quality (CRF, lower=better)", self.clip_quality_spin)

        self.clip_save_dir_edit = QLineEdit(self.settings.get("clip_save_dir", "") or "")
        self.clip_save_dir_edit.setPlaceholderText("Same as photo save directory")
        clip_browse_btn = QPushButton("Browse…")
        clip_browse_btn.clicked.connect(self._pick_clip_dir)
        clip_dir_row = QHBoxLayout()
        clip_dir_row.addWidget(self.clip_save_dir_edit)
        clip_dir_row.addWidget(clip_browse_btn)
        clip_layout.addRow("Clip save directory", clip_dir_row)

        clip_group.setLayout(clip_layout)
        self.clip_enabled_check.toggled.connect(self._on_clip_toggle)
        self._on_clip_toggle(self.clip_enabled_check.isChecked())
        form.addRow(clip_group)

        # Disk management group
        storage_group = QGroupBox("Disk Management")
        storage_layout = QFormLayout()
        self.storage_auto_check = QCheckBox("Auto-manage disk space (delete oldest when full)")
        self.storage_auto_check.setChecked(bool(self.settings.get("storage_auto_manage", False)))
        storage_layout.addRow("", self.storage_auto_check)

        self.storage_threshold_spin = QSpinBox()
        self.storage_threshold_spin.setRange(50, 98)
        self.storage_threshold_spin.setSuffix("%")
        self.storage_threshold_spin.setValue(int(self.settings.get("storage_max_pct", 85)))
        storage_layout.addRow("Max disk usage", self.storage_threshold_spin)

        storage_group.setLayout(storage_layout)
        self.storage_auto_check.toggled.connect(
            lambda on: self.storage_threshold_spin.setEnabled(on)
        )
        self.storage_threshold_spin.setEnabled(self.storage_auto_check.isChecked())
        form.addRow(storage_group)

        btn_row = QHBoxLayout()
        self.start_btn = QPushButton("Start")
        self.cancel_btn = QPushButton("Cancel")
        self.start_btn.clicked.connect(self.accept)
        self.cancel_btn.clicked.connect(self.reject)
        btn_row.addWidget(self.start_btn)
        btn_row.addWidget(self.cancel_btn)
        form.addRow(btn_row)

    def _pick_dir(self):
        target = QFileDialog.getExistingDirectory(self, "Select save directory", self.save_dir_edit.text() or ".")
        if target:
            self.save_dir_edit.setText(target)

    def _pick_clip_dir(self):
        start = self.clip_save_dir_edit.text().strip() or self.save_dir_edit.text().strip() or "."
        target = QFileDialog.getExistingDirectory(self, "Select clip save directory", start)
        if target:
            self.clip_save_dir_edit.setText(target)

    def get_settings(self) -> Dict[str, object]:
        unit = self.duration_unit.currentText()
        duration_sec = int(self.duration_spin.value())
        if unit == "Minutes":
            duration_sec *= 60
        elif unit == "Hours":
            duration_sec *= 3600
        elif unit == "Infinite":
            duration_sec = -1
        return {
            "duration_sec": duration_sec,
            "duration_unit": unit,
            "trigger_delay_ms": int(self.delay_spin.value()),
            "crop_w": int(self.crop_w_spin.value()),
            "crop_h": int(self.crop_h_spin.value()),
            "resize_w": int(self.resize_w_spin.value()),
            "include_overlays": bool(self.overlay_check.isChecked()),
            "save_dir": self.save_dir_edit.text().strip() or "captures/motion_watch",
            "allow_zone": bool(self.zone_check.isChecked()),
            "allow_line": bool(self.line_check.isChecked()),
            "allow_tag": bool(self.tag_check.isChecked()),
            "cooldown_sec": int(self.cooldown_spin.value()),
            "clip_enabled": bool(self.clip_enabled_check.isChecked()),
            "clip_duration_sec": int(self.clip_duration_spin.value()),
            "clip_pre_roll_sec": int(self.clip_pre_roll_spin.value()),
            "clip_resize_w": int(self.clip_resize_spin.value()),
            "clip_quality": int(self.clip_quality_spin.value()),
            "clip_save_dir": self.clip_save_dir_edit.text().strip(),
            "storage_auto_manage": bool(self.storage_auto_check.isChecked()),
            "storage_max_pct": int(self.storage_threshold_spin.value()),
        }

    def _on_unit_change(self, text: str):
        is_infinite = text == "Infinite"
        self.duration_spin.setEnabled(not is_infinite)

    def _on_clip_toggle(self, enabled: bool):
        self.clip_duration_spin.setEnabled(enabled)
        self.clip_pre_roll_spin.setEnabled(enabled)
        self.clip_resize_spin.setEnabled(enabled)
        self.clip_quality_spin.setEnabled(enabled)
        self.clip_save_dir_edit.setEnabled(enabled)


class ShapeSettingsDialog(QDialog):
    shape_updated = Signal(dict)

    def __init__(self, shape: Shape, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Shape Settings")
        self.resize(420, 820)
        self.setWindowFlags(Qt.WindowType.Tool | Qt.WindowType.WindowStaysOnTopHint)
        
        self.shape = shape.copy()
        kind = self.shape.get('kind')
        self._camera_id = None
        try:
            # Parent is typically CameraWidget
            self._camera_id = getattr(parent, "camera_id", None)
        except Exception:
            self._camera_id = None
        
        layout = QFormLayout(self)
        
        # Name
        self.name_input = QLineEdit(self.shape.get('label', ''))
        self.name_input.textChanged.connect(self._update_name)
        layout.addRow("Label:", self.name_input)
        
        # Color
        self.color_btn = QPushButton()
        self.color_btn.setFixedHeight(25)
        self.color = QColor(self.shape.get('color', '#24D1FF'))
        self.color_btn.setStyleSheet(f"background-color: {self.color.name()}; border: 1px solid #555;")
        self.color_btn.clicked.connect(self._pick_color)
        layout.addRow("Color:", self.color_btn)
        
        # Opacity
        self.alpha_slider = QSlider(Qt.Orientation.Horizontal)
        self.alpha_slider.setRange(10, 100)
        self.alpha_slider.setValue(int(float(self.shape.get('alpha', 0.65)) * 100))
        self.alpha_slider.valueChanged.connect(self._update_alpha)
        layout.addRow("Opacity:", self.alpha_slider)
        
        # --- Appearance ---
        layout.addRow(QLabel("<b>Appearance</b>"))
        
        if kind in ('zone', 'line'):
            # Line Thickness (0.5 - 10, slider stored as tenths)
            self.thick_slider = QSlider(Qt.Orientation.Horizontal)
            self.thick_slider.setRange(5, 100)  # 0.5 .. 10.0
            self.thick_slider.setValue(int(float(self.shape.get('line_thickness', 1.0)) * 10))
            self.thick_slider.valueChanged.connect(lambda v: self._update_float('line_thickness', v / 10.0))
            layout.addRow("Line Thickness:", self.thick_slider)
            
            # Dot Size (Vertex/Handle) 0.5 - 10
            self.dot_slider = QSlider(Qt.Orientation.Horizontal)
            self.dot_slider.setRange(5, 100)  # 0.5 .. 10.0
            self.dot_slider.setValue(int(float(self.shape.get('dot_size', 2.0)) * 10))
            self.dot_slider.valueChanged.connect(lambda v: self._update_float('dot_size', v / 10.0))
            layout.addRow("Vertex/End Size:", self.dot_slider)

        if kind == 'tag':
            # Tag Size
            self.size_slider = QSlider(Qt.Orientation.Horizontal)
            self.size_slider.setRange(10, 100)
            self.size_slider.setValue(int(self.shape.get('tag_size', 20)))
            self.size_slider.valueChanged.connect(lambda v: self._update_int('tag_size', v))
            layout.addRow("Tag Size:", self.size_slider)
            
            # Tag Style
            self.style_combo = QComboBox()
            self.style_combo.addItems(["Crosshair", "Target", "Pin", "Diamond", "Square", "Triangle", "Star", "Shield", "Person", "Car", "Warning"])
            self.style_combo.setCurrentText(self.shape.get('tag_style', 'Crosshair'))
            self.style_combo.currentTextChanged.connect(lambda t: self._update_str('tag_style', t))
            layout.addRow("Style:", self.style_combo)

        # Zone/Line text settings
        if kind in ('zone', 'line'):
            self.text_size_slider = QSlider(Qt.Orientation.Horizontal)
            self.text_size_slider.setRange(6, 48)
            self.text_size_slider.setValue(int(self.shape.get('text_size', 12)))
            self.text_size_slider.valueChanged.connect(lambda v: self._update_int('text_size', v))
            layout.addRow("Text Size:", self.text_size_slider)

            self.text_color_btn = QPushButton()
            self.text_color_btn.setFixedHeight(25)
            text_val = self.shape.get('text_color') or '#F0F0F0'
            self.text_color = QColor(text_val)
            self.text_color_btn.setStyleSheet(f"background-color: {self.text_color.name()}; border: 1px solid #555;")
            self.text_color_btn.clicked.connect(self._pick_text_color)
            layout.addRow("Text Color:", self.text_color_btn)

        # --- Interaction ---
        layout.addRow(QLabel("<b>Interaction</b>"))

        # Interaction Color
        self.int_color_btn = QPushButton()
        self.int_color_btn.setFixedHeight(25)
        default_int = '#FFD74A' if self.shape.get('kind') != 'tag' else '#00FFC6'
        int_val = self.shape.get('interaction_color') or default_int
        self.int_color = QColor(int_val)
        self.int_color_btn.setStyleSheet(f"background-color: {self.int_color.name()}; border: 1px solid #555;")
        self.int_color_btn.clicked.connect(self._pick_int_color)
        layout.addRow("Interaction Color:", self.int_color_btn)
        
        # Interaction Animation
        self.anim_combo = QComboBox()
        self.anim_combo.addItems(["Pulse", "Ripple", "Flash", "Outline"])
        self.anim_combo.setCurrentText(self.shape.get('interaction_animation', 'Pulse'))
        self.anim_combo.currentTextChanged.connect(self._update_anim)
        layout.addRow("Animation:", self.anim_combo)
        
        # Toggles
        layout.addRow(QLabel("<b>Options</b>"))
        self.show_label_chk = QCheckBox("Show Label")
        self.show_label_chk.setChecked(self.shape.get('show_label', True))
        self.show_label_chk.toggled.connect(lambda x: self._update_bool('show_label', x))
        layout.addRow("", self.show_label_chk)
        
        self.enabled_chk = QCheckBox("Enabled")
        self.enabled_chk.setChecked(self.shape.get('enabled', True))
        self.enabled_chk.toggled.connect(lambda x: self._update_bool('enabled', x))
        layout.addRow("", self.enabled_chk)
        
        self.locked_chk = QCheckBox("Locked (prevent drag)")
        self.locked_chk.setChecked(self.shape.get('locked', False))
        self.locked_chk.toggled.connect(lambda x: self._update_bool('locked', x))
        layout.addRow("", self.locked_chk)

        # --- Auto visibility ---
        try:
            if kind in ("zone", "line", "tag") and self._camera_id:
                layout.addRow(QLabel("<b>Auto visibility (this shape)</b>"))

                # Global focus-steal safety toggle (stored in desktop prefs)
                self.focus_steal_chk = QCheckBox("Bring to front / activate window when timed auto-show triggers (global)")
                self.focus_steal_chk.setChecked(bool(self._read_focus_steal_enabled()))
                layout.addRow("", self.focus_steal_chk)

                layout.addRow(QLabel("<b>Auto-show / Auto-hide</b>"))
                self.show_motion_chk = QCheckBox("Auto-show (sticky) on motion")
                self.show_det_chk = QCheckBox("Auto-show (sticky) on detections")
                self.hide_motion_chk = QCheckBox("Auto-hide on motion")
                self.hide_det_chk = QCheckBox("Auto-hide on detections")
                layout.addRow("", self.show_motion_chk)
                layout.addRow("", self.show_det_chk)
                layout.addRow("", self.hide_motion_chk)
                layout.addRow("", self.hide_det_chk)

                layout.addRow(QLabel("<b>Auto-show (timed)</b>"))
                self.timed_motion_chk = QCheckBox("Timed auto-show on motion")
                self.timed_det_chk = QCheckBox("Timed auto-show on detections")
                layout.addRow("", self.timed_motion_chk)
                layout.addRow("", self.timed_det_chk)

                self.timed_duration_spin = QSpinBox()
                self.timed_duration_spin.setRange(1, 3600)
                self.timed_duration_spin.setSuffix(" s")
                self.timed_cooldown_spin = QSpinBox()
                self.timed_cooldown_spin.setRange(0, 3600)
                self.timed_cooldown_spin.setSuffix(" s")
                layout.addRow("Duration:", self.timed_duration_spin)
                layout.addRow("Cooldown:", self.timed_cooldown_spin)

                # Hydrate from prefs
                self._hydrate_visibility_from_prefs()
        except Exception:
            pass
        
        # Buttons
        btn_box = QHBoxLayout()
        save_btn = QPushButton("Save")
        save_btn.clicked.connect(self.accept)
        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)
        btn_box.addWidget(save_btn)
        btn_box.addWidget(cancel_btn)
        layout.addRow(btn_box)

    def _update_name(self, text):
        self.shape['label'] = text

    def _update_anim(self, text):
        self.shape['interaction_animation'] = text

    def _update_bool(self, key, val):
        self.shape[key] = val
        
    def _update_int(self, key, val):
        self.shape[key] = val

    def _update_float(self, key, val):
        self.shape[key] = float(val)
        
    def _update_str(self, key, val):
        self.shape[key] = val

    def _update_alpha(self, val):
        self.shape['alpha'] = val / 100.0

    def _pick_color(self):
        c = QColorDialog.getColor(self.color, self, "Select Shape Color")
        if c.isValid():
            self.color = c
            self.shape['color'] = c.name()
            self.color_btn.setStyleSheet(f"background-color: {c.name()}; border: 1px solid #555;")

    def _pick_int_color(self):
        c = QColorDialog.getColor(self.int_color, self, "Select Interaction Color")
        if c.isValid():
            self.int_color = c
            self.shape['interaction_color'] = c.name()
            self.int_color_btn.setStyleSheet(f"background-color: {c.name()}; border: 1px solid #555;")

    def _pick_text_color(self):
        c = QColorDialog.getColor(self.text_color, self, "Select Text Color")
        if c.isValid():
            self.text_color = c
            self.shape['text_color'] = c.name()
            self.text_color_btn.setStyleSheet(f"background-color: {c.name()}; border: 1px solid #555;")

    def get_shape(self):
        return self.shape

    # ---- auto visibility helpers ----
    def _get_desktop_app(self):
        try:
            from PySide6.QtWidgets import QApplication
            app = QApplication.instance()
            if app and hasattr(app, "_load_prefs") and hasattr(app, "_save_prefs"):
                return app
        except Exception:
            pass
        return None

    def _read_focus_steal_enabled(self) -> bool:
        app = self._get_desktop_app()
        if not app:
            return True
        try:
            prefs = app._load_prefs()
            if not isinstance(prefs, dict):
                return True
            if "focus_steal_enabled" not in prefs:
                return True
            return bool(prefs.get("focus_steal_enabled"))
        except Exception:
            return True

    def _hydrate_visibility_from_prefs(self) -> None:
        """
        Populate checkboxes/spinboxes from visibility_rules matching this camera+shape.
        """
        app = self._get_desktop_app()
        if not app or not self._camera_id:
            return
        try:
            sid = str(self.shape.get("id") or "").strip()
            kind = str(self.shape.get("kind") or "").strip().lower()
            if not sid or kind not in ("zone", "line", "tag"):
                return
            widget_key = f"camera:{self._camera_id}"
            prefs = app._load_prefs()
            rules = prefs.get("visibility_rules") if isinstance(prefs, dict) else None
            rules = list(rules) if isinstance(rules, list) else []

            # Defaults
            show_motion = False
            show_det = False
            hide_motion = False
            hide_det = False
            timed_motion = False
            timed_det = False
            duration = 10
            cooldown = 10

            for r in rules:
                if not isinstance(r, dict):
                    continue
                trig = r.get("trigger") if isinstance(r.get("trigger"), dict) else {}
                tgt = r.get("target") if isinstance(r.get("target"), dict) else {}
                act = r.get("action") if isinstance(r.get("action"), dict) else {}

                if str(trig.get("type") or "") != "shape_trigger":
                    continue
                if str(trig.get("camera_id") or "") != str(self._camera_id):
                    continue
                if str(trig.get("shape_id") or "") != sid:
                    continue
                if str(trig.get("kind") or "").strip().lower() != kind:
                    continue
                if str(tgt.get("type") or "") != "widget":
                    continue
                if str(tgt.get("widget_key") or "") != widget_key:
                    continue

                src = str(trig.get("source") or "").strip().lower()
                act_type = str(act.get("type") or "").strip()

                if act_type == "show_persisted":
                    if src == "motion":
                        show_motion = True
                    elif src == "detection":
                        show_det = True
                elif act_type == "hide_persisted":
                    if src == "motion":
                        hide_motion = True
                    elif src == "detection":
                        hide_det = True
                elif act_type == "show_and_activate":
                    if src == "motion":
                        timed_motion = True
                    elif src == "detection":
                        timed_det = True
                    try:
                        duration = max(duration, int(float(act.get("duration_sec", 0) or 0)))
                    except Exception:
                        pass
                    try:
                        cooldown = max(cooldown, int(float(act.get("cooldown_sec", 0) or 0)))
                    except Exception:
                        pass

            self.show_motion_chk.setChecked(bool(show_motion))
            self.show_det_chk.setChecked(bool(show_det))
            self.hide_motion_chk.setChecked(bool(hide_motion))
            self.hide_det_chk.setChecked(bool(hide_det))
            self.timed_motion_chk.setChecked(bool(timed_motion))
            self.timed_det_chk.setChecked(bool(timed_det))
            self.timed_duration_spin.setValue(int(max(1, min(3600, duration))))
            self.timed_cooldown_spin.setValue(int(max(0, min(3600, cooldown))))
        except Exception:
            return

    def _write_visibility_to_prefs(self) -> None:
        """
        Upsert visibility_rules for this camera+shape based on current UI controls.
        """
        app = self._get_desktop_app()
        if not app or not self._camera_id:
            return
        try:
            sid = str(self.shape.get("id") or "").strip()
            kind = str(self.shape.get("kind") or "").strip().lower()
            if not sid or kind not in ("zone", "line", "tag"):
                return
            cam_id = str(self._camera_id)
            widget_key = f"camera:{cam_id}"

            prefs = app._load_prefs()
            if not isinstance(prefs, dict):
                prefs = {}

            # Global toggle
            try:
                prefs["focus_steal_enabled"] = bool(self.focus_steal_chk.isChecked())
            except Exception:
                pass

            rules = prefs.get("visibility_rules")
            rules = list(rules) if isinstance(rules, list) else []

            def _is_this_shape_rule(r: dict) -> bool:
                try:
                    if not isinstance(r, dict):
                        return False
                    trig = r.get("trigger") if isinstance(r.get("trigger"), dict) else {}
                    tgt = r.get("target") if isinstance(r.get("target"), dict) else {}
                    act = r.get("action") if isinstance(r.get("action"), dict) else {}
                    if str(trig.get("type") or "") != "shape_trigger":
                        return False
                    if str(trig.get("camera_id") or "") != cam_id:
                        return False
                    if str(trig.get("shape_id") or "") != sid:
                        return False
                    if str(trig.get("kind") or "").strip().lower() != kind:
                        return False
                    if str(trig.get("source") or "").strip().lower() not in ("motion", "detection"):
                        return False
                    if str(tgt.get("type") or "") != "widget":
                        return False
                    if str(tgt.get("widget_key") or "") != widget_key:
                        return False
                    if str(act.get("type") or "").strip() not in ("show_and_activate", "show_persisted", "hide_persisted"):
                        return False
                    return True
                except Exception:
                    return False

            kept = [r for r in rules if not _is_this_shape_rule(r)]
            new_rules = list(kept)

            def add_rule(source: str, action_type: str, action: dict):
                new_rules.append(
                    {
                        "trigger": {
                            "type": "shape_trigger",
                            "camera_id": cam_id,
                            "shape_id": sid,
                            "kind": kind,
                            "source": source,
                        },
                        "target": {"type": "widget", "widget_key": widget_key},
                        "action": {"type": action_type, **action},
                    }
                )

            duration = int(self.timed_duration_spin.value())
            cooldown = int(self.timed_cooldown_spin.value())
            duration = max(1, min(3600, duration))
            cooldown = max(0, min(3600, cooldown))

            # Sticky show/hide
            if bool(self.show_motion_chk.isChecked()):
                add_rule("motion", "show_persisted", {})
            if bool(self.show_det_chk.isChecked()):
                add_rule("detection", "show_persisted", {})
            if bool(self.hide_motion_chk.isChecked()):
                add_rule("motion", "hide_persisted", {})
            if bool(self.hide_det_chk.isChecked()):
                add_rule("detection", "hide_persisted", {})

            # Timed show
            if bool(self.timed_motion_chk.isChecked()):
                add_rule("motion", "show_and_activate", {"duration_sec": duration, "cooldown_sec": cooldown})
            if bool(self.timed_det_chk.isChecked()):
                add_rule("detection", "show_and_activate", {"duration_sec": duration, "cooldown_sec": cooldown})

            prefs["visibility_rules"] = new_rules
            app._save_prefs(prefs)
        except Exception:
            return

    def accept(self):
        # Persist auto visibility changes first (best-effort)
        try:
            if hasattr(self, "timed_duration_spin"):
                self._write_visibility_to_prefs()
        except Exception:
            pass
        super().accept()


class ObjectDetectionSettingsDialog(QDialog):
    API_BASE = "http://localhost:5000/api"
    COCO_CLASSES = [
        'person','bicycle','car','motorcycle','airplane','bus','train','truck','boat',
        'traffic light','fire hydrant','stop sign','parking meter','bench','bird','cat',
        'dog','horse','sheep','cow','elephant','bear','zebra','giraffe','backpack',
        'umbrella','handbag','tie','suitcase','frisbee','skis','snowboard','sports ball',
        'kite','baseball bat','baseball glove','skateboard','surfboard','tennis racket',
        'bottle','wine glass','cup','fork','knife','spoon','bowl','banana','apple',
        'sandwich','orange','broccoli','carrot','hot dog','pizza','donut','cake',
        'chair','couch','potted plant','bed','dining table','toilet','tv','laptop',
        'mouse','remote','keyboard','cell phone','microwave','oven','toaster','sink',
        'refrigerator','book','clock','vase','scissors','teddy bear','hair drier','toothbrush'
    ]
    MOBILENET_VOC = [
        'aeroplane','bicycle','bird','boat','bottle','bus','car','cat','chair','cow',
        'diningtable','dog','horse','motorbike','person','pottedplant','sheep','sofa','train','tvmonitor'
    ]
    ALL_CLASSES = sorted(list(set(COCO_CLASSES + MOBILENET_VOC)))

    def __init__(self, camera_id, parent=None):
        super().__init__(parent)
        self.camera_id = camera_id
        self.setWindowTitle(f"Object Detection - {camera_id}")
        self.resize(500, 600)
        self.setWindowFlags(Qt.WindowType.Tool | Qt.WindowType.WindowStaysOnTopHint)
        
        self.settings = {}
        self.loading = True

        # Layouts
        self.main_layout = QVBoxLayout(self)
        self.tabs = QTabWidget()
        self.main_layout.addWidget(self.tabs)

        # Setup Tabs
        self.setup_basics_tab()
        self.setup_classes_tab()

        # Footer Buttons
        footer = QHBoxLayout()
        self.apply_btn = QPushButton("Apply Settings")
        self.apply_btn.clicked.connect(self.apply_settings)
        self.close_btn = QPushButton("Close")
        self.close_btn.clicked.connect(self.close)
        
        footer.addStretch()
        footer.addWidget(self.apply_btn)
        footer.addWidget(self.close_btn)
        self.main_layout.addLayout(footer)

        # Load Data
        self.load_config()

    def setup_basics_tab(self):
        self.basics_tab = QWidget()
        layout = QVBoxLayout(self.basics_tab)
        
        # Enable Checkbox
        self.enable_check = QCheckBox("Enable Object Detection (Verification)")
        self.enable_check.setChecked(False) # Default to off until config loads
        layout.addWidget(self.enable_check)

        # Model Selection
        group = QGroupBox("Model Selection")
        vbox = QVBoxLayout()
        self.model_mobilenet = QRadioButton("MobileNet SSD (CPU Friendly)")
        self.model_yolo = QRadioButton("YOLO (High Accuracy)")
        self.model_mobilenet.setChecked(True) # Default to MobileNet
        self.model_group = [self.model_mobilenet, self.model_yolo]
        vbox.addWidget(self.model_mobilenet)
        vbox.addWidget(self.model_yolo)
        group.setLayout(vbox)
        layout.addWidget(group)

        # Confidence
        layout.addWidget(QLabel("Minimum Confidence:"))
        self.conf_slider = QSlider(Qt.Orientation.Horizontal)
        self.conf_slider.setRange(0, 100)
        self.conf_slider.setValue(35) # Default 0.35
        self.conf_slider.valueChanged.connect(lambda v: self.conf_label.setText(f"{v/100.0:.2f}"))
        
        h = QHBoxLayout()
        h.addWidget(self.conf_slider)
        self.conf_label = QLabel("0.35")
        h.addWidget(self.conf_label)
        layout.addLayout(h)

        # Verifier Behavior
        group_ver = QGroupBox("Verifier Behavior")
        vbox_ver = QVBoxLayout()
        self.chk_detect_without_motion = QCheckBox("Detect without motion (within shapes)")
        self.chk_debug_fullframe = QCheckBox("Force full-frame detection (debug)")
        self.chk_always_detect = QCheckBox("Always detect (ignore motion gating)")
        
        # Defaults
        self.chk_detect_without_motion.setChecked(True)
        self.chk_always_detect.setChecked(True)
        
        vbox_ver.addWidget(self.chk_detect_without_motion)
        vbox_ver.addWidget(self.chk_debug_fullframe)
        vbox_ver.addWidget(self.chk_always_detect)
        group_ver.setLayout(vbox_ver)
        layout.addWidget(group_ver)

        layout.addStretch()
        self.tabs.addTab(self.basics_tab, "Basics")

    def setup_classes_tab(self):
        self.classes_tab = QWidget()
        layout = QVBoxLayout(self.classes_tab)
        
        self.class_checks = {}
        
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        container = QWidget()
        grid = QGridLayout(container)
        
        # Populate Grid
        cols = 3
        # Default targets
        defaults = ['person', 'car', 'truck', 'bus', 'motorcycle']
        
        for i, cls in enumerate(self.ALL_CLASSES):
            chk = QCheckBox(cls)
            if cls in defaults:
                chk.setChecked(True)
            self.class_checks[cls] = chk
            grid.addWidget(chk, i // cols, i % cols)
            
        scroll.setWidget(container)
        layout.addWidget(scroll)
        
        # Quick Select Buttons
        btn_layout = QHBoxLayout()
        btn_all = QPushButton("All")
        btn_none = QPushButton("None")
        btn_rec = QPushButton("People + Vehicles")
        
        btn_all.clicked.connect(self.select_all_classes)
        btn_none.clicked.connect(self.select_no_classes)
        btn_rec.clicked.connect(self.select_rec_classes)
        
        btn_layout.addWidget(btn_all)
        btn_layout.addWidget(btn_none)
        btn_layout.addWidget(btn_rec)
        layout.addLayout(btn_layout)

        self.tabs.addTab(self.classes_tab, "Classes")

    def select_all_classes(self):
        for chk in self.class_checks.values(): chk.setChecked(True)

    def select_no_classes(self):
        for chk in self.class_checks.values(): chk.setChecked(False)

    def select_rec_classes(self):
        rec = ['person', 'car', 'truck', 'bus', 'motorcycle']
        for cls, chk in self.class_checks.items():
            chk.setChecked(cls in rec)

    def load_config(self):
        def worker():
            try:
                url = f"{self.API_BASE}/cameras/{self.camera_id}/detection-config"
                res = requests.get(url, timeout=3)
                if res.ok:
                    data = res.json().get('data', {})
                    # Post to main thread
                    QTimer.singleShot(0, lambda: self.populate_ui(data))
                else:
                    print(f"Failed to load config: {res.status_code}")
                    # If failed, defaults are already set in UI setup
            except Exception as e:
                print(f"Error loading config: {e}")
                # Defaults already set

        threading.Thread(target=worker, daemon=True).start()

    def populate_ui(self, cfg):
        if not cfg:
            self.loading = False
            return
            
        # Basics
        if 'verification_enabled' in cfg:
            self.enable_check.setChecked(bool(cfg['verification_enabled']))
        
        models = cfg.get('models', [])
        if models:
            model = models[0]
            if 'yolo' in model.lower():
                self.model_yolo.setChecked(True)
            else:
                self.model_mobilenet.setChecked(True)
            
        if 'min_confidence' in cfg:
            conf = float(cfg['min_confidence'])
            self.conf_slider.setValue(int(conf * 100))
        
        if 'detect_without_motion' in cfg:
            self.chk_detect_without_motion.setChecked(bool(cfg['detect_without_motion']))
        if 'debug_fullframe' in cfg:
            self.chk_debug_fullframe.setChecked(bool(cfg['debug_fullframe']))
        if 'always_detect' in cfg:
            self.chk_always_detect.setChecked(bool(cfg['always_detect']))

        # Classes
        targets = cfg.get('target_classes')
        # Only override if explicit list provided
        if targets is not None:
            for cls, chk in self.class_checks.items():
                chk.setChecked(cls in targets)
            
        self.loading = False

    def apply_settings(self):
        if self.loading: return
        
        # Gather settings
        model = 'yolo' if self.model_yolo.isChecked() else 'mobilenet'
        target_classes = [cls for cls, chk in self.class_checks.items() if chk.isChecked()]

        # If enabling detection with MobileNet, ensure the lightweight CPU model is installed.
        # This keeps "turn on detection and it works" as the default user experience.
        if bool(self.enable_check.isChecked()) and model == "mobilenet":
            try:
                from core.model_library.vision_detection import check_mobilenetssd, ensure_mobilenetssd

                st = check_mobilenetssd()
                if not st.ok:
                    resp = QMessageBox.question(
                        self,
                        "Install MobileNetSSD?",
                        "Object detection requires the MobileNetSSD model files.\n\n"
                        "Install now? (Recommended)\n\n"
                        "This downloads the model into your per-user models folder.",
                        QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                        QMessageBox.StandardButton.Yes,
                    )
                    if resp != QMessageBox.StandardButton.Yes:
                        return

                    # Download in a background thread; keep UI responsive.
                    self.apply_btn.setEnabled(False)
                    self.apply_btn.setText("Installing…")

                    def _bg_install():
                        try:
                            ensure_mobilenetssd(force_download=False)
                            QTimer.singleShot(0, lambda: self.apply_settings())
                        except Exception as e:
                            def _ui_err():
                                QMessageBox.warning(self, "Install failed", str(e))
                                self.apply_btn.setEnabled(True)
                                self.apply_btn.setText("Apply Settings")
                            QTimer.singleShot(0, _ui_err)

                    threading.Thread(target=_bg_install, daemon=True).start()
                    return
            except Exception:
                # Best-effort; if anything goes wrong here, backend will report missing files.
                pass
        
        payload = {
            "verification_enabled": self.enable_check.isChecked(),
            "models": [model],
            "min_confidence": self.conf_slider.value() / 100.0,
            "target_classes": target_classes,
            "detect_without_motion": self.chk_detect_without_motion.isChecked(),
            "debug_fullframe": self.chk_debug_fullframe.isChecked(),
            "always_detect": self.chk_always_detect.isChecked()
        }
        
        # Also handle Enable/Disable endpoints for immediate effect if needed
        # But detection-config usually handles params, enable/disable might be separate
        # React calls /detection/enable or /detection/disable
        
        def worker():
            try:
                # 1. Update Config
                url = f"{self.API_BASE}/cameras/{self.camera_id}/detection-config"
                res = requests.put(url, json=payload, timeout=3)
                if not res.ok:
                    print(f"Failed to save settings: {res.text}")
                    return

                # 2. Trigger Enable/Disable explicitly if changed
                # The React app does this separately. 
                # If enabled is checked, we hit /enable, else /disable
                if self.enable_check.isChecked():
                    requests.post(f"{self.API_BASE}/cameras/{self.camera_id}/detection/enable", timeout=3)
                else:
                    requests.post(f"{self.API_BASE}/cameras/{self.camera_id}/detection/disable", timeout=3)
                
                print("Settings applied successfully")
                
            except Exception as e:
                print(f"Error saving settings: {e}")

        threading.Thread(target=worker, daemon=True).start()
        self.close()


class CameraScanWorker(QThread):
    """Background worker that calls the API network scan then probes ports."""
    progress = Signal(str)
    finished = Signal(list)
    error = Signal(str)

    def __init__(self, api_base: str, cidr: str = "", interface_name: str = "", max_hosts: int = 512,
                 timeout_ms: int = 750, port_timeout: float = 0.45, include_unreachable: bool = False, parent=None):
        super().__init__(parent)
        self.api_base = api_base.rstrip("/")
        self.cidr = cidr.strip()
        self.interface_name = interface_name.strip()
        self.max_hosts = max(1, min(max_hosts, 4096))
        self.timeout_ms = max(100, min(timeout_ms, 5000))
        self.port_timeout = max(0.1, min(port_timeout, 2.0))
        self.include_unreachable = bool(include_unreachable)

    @staticmethod
    def _normalize_mac(mac: Optional[str]) -> str:
        if not mac:
            return ""
        return mac.upper().replace("-", ":")

    @staticmethod
    def _lookup_vendor(mac: str) -> Optional[str]:
        mac_norm = CameraScanWorker._normalize_mac(mac)
        if len(mac_norm) < 8:
            return None
        prefix = mac_norm[:8]
        return CAMERA_OUI_PREFIXES.get(prefix)

    @staticmethod
    def _scan_ports(ip: str, timeout: float) -> List[int]:
        open_ports: List[int] = []
        for port in CAMERA_PORTS:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(timeout)
            try:
                if sock.connect_ex((ip, port)) == 0:
                    open_ports.append(port)
            except Exception:
                # Ignore individual port failures to keep scan resilient
                pass
            finally:
                try:
                    sock.close()
                except Exception:
                    pass
        return open_ports

    @staticmethod
    def _score_device(open_ports: List[int], vendor: Optional[str]) -> Tuple[int, bool]:
        port_hits = len(set(open_ports) & set(CAMERA_PORTS))
        vendor_bonus = 2 if vendor else 0
        score = port_hits * 2 + vendor_bonus
        likely_camera = score >= 3 or 554 in open_ports or 8554 in open_ports
        return score, likely_camera

    def run(self):
        try:
            self.progress.emit("Running network sweep…")
            params = {
                "max_hosts": self.max_hosts,
                "timeout_ms": self.timeout_ms
            }
            if self.interface_name:
                params["interface"] = self.interface_name
            if self.cidr:
                params["cidr"] = self.cidr
            if self.include_unreachable:
                params["include_unreachable"] = "true"

            resp = requests.get(f"{self.api_base}/network/scan", params=params, timeout=35)
            resp.raise_for_status()
            payload = resp.json() or {}
            data = payload.get("data") or {}
            devices = data.get("devices") or []
            candidates = [d for d in devices if isinstance(d, dict) and d.get("ip")]
            if not self.include_unreachable:
                candidates = [d for d in candidates if d.get("reachable")]

            results = []
            total = len(candidates)
            for idx, dev in enumerate(candidates, start=1):
                ip = dev.get("ip")
                if not ip:
                    continue
                self.progress.emit(f"Probing {ip} ({idx}/{total})…")
                open_ports = self._scan_ports(ip, self.port_timeout)
                mac = dev.get("mac") or dev.get("mac_address")
                vendor = self._lookup_vendor(mac)
                score, likely = self._score_device(open_ports, vendor)
                results.append({
                    "ip": ip,
                    "mac": self._normalize_mac(mac),
                    "hostname": dev.get("hostname") or "",
                    "manufacturer": vendor or "",
                    "open_ports": sorted(open_ports),
                    "score": score,
                    "likely_camera": likely,
                    "latency_ms": dev.get("latency_ms"),
                })

            results.sort(key=lambda item: (-int(item.get("likely_camera", False)), -item.get("score", 0), item.get("ip", "")))
            self.finished.emit(results)
        except Exception as exc:
            self.error.emit(str(exc))


class CameraDiscoveryDialog(QDialog):
    """Simple IP camera finder dialog, reachable from the Add Camera menu."""
    camera_selected = Signal(dict)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Find IP Cameras")
        self.resize(760, 520)
        self.worker: Optional[CameraScanWorker] = None
        self.results: List[Dict[str, object]] = []
        self.interfaces: List[Dict[str, object]] = []

        layout = QVBoxLayout(self)

        form = QHBoxLayout()
        form.addWidget(QLabel("Interface:"))
        self.interface_combo = QComboBox()
        self.interface_combo.currentIndexChanged.connect(self._handle_interface_change)
        form.addWidget(self.interface_combo)
        self.refresh_interfaces_btn = QPushButton("Refresh")
        self.refresh_interfaces_btn.clicked.connect(self._load_interfaces)
        form.addWidget(self.refresh_interfaces_btn)

        form.addWidget(QLabel("CIDR / Range:"))
        self.cidr_input = QLineEdit()
        self.cidr_input.setPlaceholderText("auto (use active interface)")
        form.addWidget(self.cidr_input)

        form.addWidget(QLabel("Max hosts"))
        self.max_hosts = QSpinBox()
        self.max_hosts.setRange(1, 4096)
        self.max_hosts.setValue(512)
        form.addWidget(self.max_hosts)

        self.scan_btn = QPushButton("Scan")
        self.scan_btn.clicked.connect(self.start_scan)
        form.addWidget(self.scan_btn)
        layout.addLayout(form)

        self.include_unreachable = QCheckBox("Include unreachable hosts (scan all)")
        self.include_unreachable.setChecked(True)
        layout.addWidget(self.include_unreachable)

        self.progress = QProgressBar()
        self.progress.setRange(0, 0)
        self.progress.hide()
        layout.addWidget(self.progress)

        self.status_label = QLabel("")
        layout.addWidget(self.status_label)

        self.table = QTableWidget(0, 5)
        self.table.setHorizontalHeaderLabels(["IP", "MAC", "Manufacturer", "Ports", "Confidence"])
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.table.verticalHeader().setVisible(False)
        self.table.setSelectionBehavior(self.table.SelectionBehavior.SelectRows)
        self.table.setEditTriggers(self.table.EditTrigger.NoEditTriggers)
        self.table.itemDoubleClicked.connect(self._use_selected)
        layout.addWidget(self.table)

        btn_row = QHBoxLayout()
        self.use_btn = QPushButton("Use Selected")
        self.use_btn.clicked.connect(self._use_selected)
        btn_row.addWidget(self.use_btn)
        btn_row.addStretch()
        layout.addLayout(btn_row)

        self._load_interfaces()

    def _format_interface_label(self, iface: Dict[str, object]) -> str:
        parts = [str(iface.get("name") or "").strip()]
        ip = iface.get("ip")
        cidr = iface.get("cidr")
        if ip:
            parts.append(str(ip))
        if cidr:
            parts.append(str(cidr))
        return " • ".join([part for part in parts if part])

    def _load_interfaces(self):
        try:
            resp = requests.get(f"{ObjectDetectionSettingsDialog.API_BASE}/network/interfaces", timeout=5)
            payload = resp.json() if resp.ok else {}
        except Exception as exc:
            self.status_label.setText(f"Unable to load interfaces: {exc}")
            return

        data = payload.get("data") if isinstance(payload, dict) else {}
        interfaces = data.get("interfaces") if isinstance(data, dict) else []
        default_iface = data.get("defaultInterface") if isinstance(data, dict) else None
        if not isinstance(interfaces, list):
            interfaces = []

        self.interfaces = [iface for iface in interfaces if isinstance(iface, dict)]

        self.interface_combo.blockSignals(True)
        self.interface_combo.clear()
        self.interface_combo.addItem("Auto (default interface)", "")
        for iface in self.interfaces:
            name = str(iface.get("name") or "").strip()
            label = self._format_interface_label(iface)
            self.interface_combo.addItem(label or name, name)
        self.interface_combo.blockSignals(False)

        default_name = ""
        if isinstance(default_iface, dict):
            default_name = str(default_iface.get("name") or "").strip()

        if default_name:
            for idx in range(self.interface_combo.count()):
                if self.interface_combo.itemData(idx) == default_name:
                    self.interface_combo.setCurrentIndex(idx)
                    if not self.cidr_input.text().strip():
                        cidr_val = default_iface.get("cidr")
                        if isinstance(cidr_val, str) and cidr_val:
                            self.cidr_input.setText(cidr_val)
                    break
        elif self.interfaces and not self.cidr_input.text().strip():
            cidr_val = self.interfaces[0].get("cidr")
            if isinstance(cidr_val, str) and cidr_val:
                self.cidr_input.setText(cidr_val)

    def _handle_interface_change(self):
        idx = self.interface_combo.currentIndex()
        if idx < 0:
            return
        iface_name = self.interface_combo.itemData(idx)
        if not iface_name:
            return
        selected = next((iface for iface in self.interfaces if iface.get("name") == iface_name), None)
        if not selected:
            return
        cidr_val = selected.get("cidr")
        if isinstance(cidr_val, str) and cidr_val:
            self.cidr_input.setText(cidr_val)

    def start_scan(self):
        if self.worker and self.worker.isRunning():
            return

        cidr = self.cidr_input.text().strip()
        interface_name = self.interface_combo.currentData() if hasattr(self, "interface_combo") else ""
        self.status_label.setText("Starting scan…")
        self.progress.show()
        self.scan_btn.setEnabled(False)
        self.use_btn.setEnabled(False)
        self.table.setRowCount(0)

        self.worker = CameraScanWorker(
            api_base=ObjectDetectionSettingsDialog.API_BASE,
            cidr=cidr,
            interface_name=str(interface_name or ""),
            max_hosts=self.max_hosts.value(),
            include_unreachable=self.include_unreachable.isChecked()
        )
        self.worker.progress.connect(self.status_label.setText)
        self.worker.error.connect(self._handle_error)
        self.worker.finished.connect(self._handle_results)
        self.worker.start()

    def _handle_error(self, message: str):
        self.progress.hide()
        self.scan_btn.setEnabled(True)
        self.status_label.setText(f"Scan failed: {message}")

    def _handle_results(self, results: List[Dict[str, object]]):
        self.results = results
        self.progress.hide()
        self.scan_btn.setEnabled(True)
        self.use_btn.setEnabled(bool(results))
        self.status_label.setText(f"Found {len(results)} reachable devices")
        self._populate_table(results)

    def _populate_table(self, results: List[Dict[str, object]]):
        self.table.setRowCount(len(results))
        for row, item in enumerate(results):
            ip_item = QTableWidgetItem(item.get("ip", ""))
            ip_item.setData(Qt.ItemDataRole.UserRole, item)
            self.table.setItem(row, 0, ip_item)
            self.table.setItem(row, 1, QTableWidgetItem(item.get("mac", "")))
            self.table.setItem(row, 2, QTableWidgetItem(item.get("manufacturer", "")))
            ports_txt = ", ".join(str(p) for p in item.get("open_ports", []))
            self.table.setItem(row, 3, QTableWidgetItem(ports_txt))
            confidence = "Likely camera" if item.get("likely_camera") else "Uncertain"
            conf_item = QTableWidgetItem(confidence)
            if item.get("likely_camera"):
                conf_item.setForeground(QColor("#3dd598"))
            self.table.setItem(row, 4, conf_item)

    def _use_selected(self):
        row = self.table.currentRow()
        if row < 0:
            return
        ip_item = self.table.item(row, 0)
        if not ip_item:
            return
        data = ip_item.data(Qt.ItemDataRole.UserRole) or {}
        self.camera_selected.emit(data)
        self.accept()


class CameraConfigDialog(QDialog):
    """
    Compact camera configuration dialog that mirrors the React Camera Utility.
    Uses the shared /api/devices endpoints so React and PyQt stay in sync.
    """
    camera_added = Signal(dict)
    camera_deleted = Signal(list)
    camera_updated = Signal(dict)
    API_BASE = ObjectDetectionSettingsDialog.API_BASE

    def __init__(self, parent=None, camera_manager=None):
        super().__init__(parent)
        self.camera_manager = camera_manager
        self.setWindowTitle("Camera Configuration")
        self.setWindowFlags(Qt.WindowType.Tool | Qt.WindowType.WindowStaysOnTopHint)
        self.resize(460, 560)
        self.editing_camera_id: Optional[str] = None

        self.devices = []
        self.main_layout = QVBoxLayout(self)

        # Existing cameras (shared with React)
        existing_group = QGroupBox("Existing Cameras (shared with React)")
        existing_layout = QVBoxLayout(existing_group)
        self.devices_list = QListWidget()
        self.devices_list.setMinimumHeight(140)
        self.devices_list.setSelectionMode(QListWidget.SelectionMode.ExtendedSelection)
        self.refresh_btn = QPushButton("Refresh from /api/devices")
        self.refresh_btn.clicked.connect(self.load_devices)
        self.edit_btn = QPushButton("Edit Selected")
        self.edit_btn.clicked.connect(self.edit_selected_camera)
        self.delete_btn = QPushButton("Delete Selected")
        self.delete_btn.clicked.connect(self.delete_selected_cameras)
        existing_layout.addWidget(self.devices_list)
        btn_row = QHBoxLayout()
        btn_row.addWidget(self.refresh_btn)
        btn_row.addWidget(self.edit_btn)
        btn_row.addWidget(self.delete_btn)
        existing_layout.addLayout(btn_row)
        self.main_layout.addWidget(existing_group)

        discover_row = QHBoxLayout()
        self.smart_scan_btn = QPushButton("Scan for Cameras\u2026")
        self.smart_scan_btn.setToolTip("Open the encoder-aware camera scanner with inline editing")
        self.smart_scan_btn.clicked.connect(self._open_smart_scanner)
        discover_row.addWidget(self.smart_scan_btn)
        discover_row.addStretch()
        self.main_layout.addLayout(discover_row)

        # New camera form
        form_group = QGroupBox("Add Camera")
        form = QFormLayout(form_group)

        self.name_input = QLineEdit()
        self.name_input.setPlaceholderText("Front Gate")
        form.addRow("Name*", self.name_input)

        self.location_input = QLineEdit("Default")
        form.addRow("Location", self.location_input)

        self.use_full_url = QCheckBox("Use full RTSP URL")
        self.use_full_url.toggled.connect(self._toggle_full_url)
        form.addRow(self.use_full_url)

        self.full_url_input = QLineEdit()
        self.full_url_input.setPlaceholderText("rtsp://user:pass@10.0.0.12:554/media/video1")
        self.full_url_input.setEnabled(False)
        form.addRow("Full URL", self.full_url_input)

        self.ip_input = QLineEdit()
        self.ip_input.setPlaceholderText("192.168.1.100")
        form.addRow("IP Address*", self.ip_input)

        self.username_input = QLineEdit("admin")
        form.addRow("Username", self.username_input)

        self.password_input = QLineEdit()
        self.password_input.setEchoMode(QLineEdit.EchoMode.Password)
        form.addRow("Password", self.password_input)

        self.port_spin = QSpinBox()
        self.port_spin.setRange(1, 65535)
        self.port_spin.setValue(554)
        form.addRow("RTSP Port", self.port_spin)

        self.stream_path_input = QLineEdit("/media/video1")
        form.addRow("Stream Path", self.stream_path_input)

        self.substream_enabled = QCheckBox("Configure substream path")
        self.substream_enabled.setChecked(False)
        self.substream_enabled.toggled.connect(self._toggle_substream)
        form.addRow(self.substream_enabled)

        self.substream_path_input = QLineEdit("/media/video2")
        self.substream_path_input.setEnabled(False)
        form.addRow("Substream Path", self.substream_path_input)

        self.manufacturer_combo = QComboBox()
        self.manufacturer_combo.addItems([
            "auto", "hikvision", "dahua", "axis", "ubiquiti", "foscam", "amcrest", "generic", "custom"
        ])
        form.addRow("Manufacturer", self.manufacturer_combo)

        self.priority_combo = QComboBox()
        self.priority_combo.addItems(["sub", "main"])
        self.priority_combo.setCurrentText("main")
        form.addRow("Stream Priority", self.priority_combo)

        self.main_layout.addWidget(form_group)

        # Action buttons
        btn_row = QHBoxLayout()
        self.test_btn = QPushButton("Test Connection")
        self.test_btn.clicked.connect(self.test_connection)
        self.save_btn = QPushButton("Save Camera")
        self.save_btn.clicked.connect(self.save_camera)
        btn_row.addWidget(self.test_btn)
        btn_row.addWidget(self.save_btn)
        self.main_layout.addLayout(btn_row)

        self.status_label = QLabel("")
        self.status_label.setStyleSheet("color: #8ee3c6;")
        self.main_layout.addWidget(self.status_label)

        self.load_devices()

    def _toggle_full_url(self, checked: bool):
        self.full_url_input.setEnabled(checked)
        for widget in (self.ip_input, self.username_input, self.password_input, self.port_spin, self.stream_path_input):
            widget.setEnabled(not checked)

    def _toggle_substream(self, checked: bool):
        self.substream_path_input.setEnabled(checked)
        if not checked:
            self.priority_combo.setCurrentText("main")

    def _open_smart_scanner(self):
        """Open the encoder-aware smart camera scanner dialog."""
        try:
            from desktop.widgets.scanner_dialog import CameraScannerDialog
            from pathlib import Path
            cameras_path = Path(__file__).resolve().parent.parent.parent / "data" / "cameras.json"
            dlg = CameraScannerDialog(
                cameras_json_path=cameras_path,
                on_open_config=lambda: (self.show(), self.raise_(), self.load_devices()),
                parent=None,
            )
            dlg.exec()
            self.load_devices()
        except Exception as exc:
            logger.warning("Could not open smart scanner: %s", exc)

    def apply_discovered_camera(self, info: Dict[str, object]):
        ip = info.get("ip")
        if isinstance(ip, str) and ip:
            self.ip_input.setText(ip)
            if not self.name_input.text().strip():
                self.name_input.setText(f"Camera {ip}")

        manufacturer = info.get("manufacturer")
        if isinstance(manufacturer, str) and manufacturer:
            idx = self.manufacturer_combo.findText(manufacturer, Qt.MatchFlag.MatchContains)
            if idx >= 0:
                self.manufacturer_combo.setCurrentIndex(idx)

        open_ports = info.get("open_ports") or []
        if 554 in open_ports:
            self.port_spin.setValue(554)
        elif 8554 in open_ports:
            self.port_spin.setValue(8554)

        self.status_label.setText(f"Filled from discovery: {ip or 'device'}")

    def _build_rtsp_url(self) -> str:
        if self.use_full_url.isChecked():
            return self.full_url_input.text().strip()

        ip = self.ip_input.text().strip()
        if not ip:
            return ""

        username = self.username_input.text().strip()
        password = self.password_input.text()
        port = self.port_spin.value()
        path = self.stream_path_input.text().strip() or "/media/video1"

        if username:
            auth = f"{username}:{password}@" if password else f"{username}@"
            return f"rtsp://{auth}{ip}:{port}{path}"
        return f"rtsp://{ip}:{port}{path}"

    def _current_payload(self) -> dict:
        rtsp_url = self._build_rtsp_url()
        substream_path = self.substream_path_input.text().strip() if self.substream_enabled.isChecked() else None
        stream_priority = self.priority_combo.currentText()
        stream_quality = "low" if stream_priority == "sub" else "medium"

        return {
            "name": self.name_input.text().strip(),
            "rtsp_url": rtsp_url,
            "location": self.location_input.text().strip() or "Default",
            "enabled": True,
            "webrtc_enabled": True,
            "username": self.username_input.text().strip(),
            "password": self.password_input.text(),
            "ip_address": self.ip_input.text().strip(),
            "port": self.port_spin.value(),
            "type": "camera",
            "manufacturer": self.manufacturer_combo.currentText(),
            "stream_path": None if self.use_full_url.isChecked() else (self.stream_path_input.text().strip() or "/media/video1"),
            "custom_rtsp": self.use_full_url.isChecked(),
            "ai_analysis": False,
            "recording": False,
            "ptz_enabled": False,
            "stream_quality": stream_quality,
            "motion_detection": True,
            "audio_enabled": False,
            "night_vision": False,
            "privacy_mask": [],
            "protocol": "tcp",
            "substream_path": substream_path,
            "stream_priority": stream_priority
        }

    def _set_edit_mode(self, camera_id: Optional[str]):
        self.editing_camera_id = camera_id
        if camera_id:
            self.save_btn.setText("Update Camera")
            self.status_label.setText("Editing existing camera (Update Camera to save)")
        else:
            self.save_btn.setText("Save Camera")

    def _populate_form_from_camera(self, cam: dict):
        # Basic fields
        self.name_input.setText(str(cam.get("name") or ""))
        self.location_input.setText(str(cam.get("location") or "Default"))

        # Prefer explicit stored fields if present
        ip = cam.get("ip_address") or cam.get("ip") or ""
        username = cam.get("username") or ""
        password = cam.get("password") or ""
        port = cam.get("port") or 554
        stream_path = cam.get("stream_path") or "/media/video1"
        substream_path = cam.get("substream_path")

        manufacturer = cam.get("manufacturer") or "auto"
        priority = cam.get("stream_priority") or "main"

        # If the backend marked it as custom/full URL, respect it; else try to parse RTSP
        custom_rtsp = bool(cam.get("custom_rtsp"))
        rtsp_url = (cam.get("rtsp_url") or "").strip()
        if rtsp_url and not ip:
            try:
                parsed = urlparse(rtsp_url)
                if parsed.hostname:
                    ip = parsed.hostname
                if parsed.port:
                    port = parsed.port
                if parsed.username:
                    username = parsed.username
                if parsed.password is not None:
                    password = parsed.password
                if parsed.path:
                    stream_path = parsed.path
            except Exception:
                pass

        # Apply to UI
        self.use_full_url.setChecked(custom_rtsp)
        self.full_url_input.setText(rtsp_url if custom_rtsp else "")
        self.ip_input.setText(str(ip))
        self.username_input.setText(str(username))
        self.password_input.setText(str(password))
        try:
            self.port_spin.setValue(int(port) if port else 554)
        except Exception:
            self.port_spin.setValue(554)
        self.stream_path_input.setText(str(stream_path or "/Streaming/Channels/101"))
        self.substream_enabled.setChecked(bool(substream_path))
        self.substream_path_input.setText(str(substream_path or "/Streaming/Channels/102"))

        idx = self.manufacturer_combo.findText(str(manufacturer), Qt.MatchFlag.MatchFixedString)
        if idx >= 0:
            self.manufacturer_combo.setCurrentIndex(idx)
        else:
            # best-effort match
            idx = self.manufacturer_combo.findText(str(manufacturer), Qt.MatchFlag.MatchContains)
            if idx >= 0:
                self.manufacturer_combo.setCurrentIndex(idx)

        if priority in ("sub", "main"):
            self.priority_combo.setCurrentText(priority)

    def load_devices(self):
        try:
            resp = requests.get(f"{self.API_BASE}/devices", timeout=4)
            if not resp.ok:
                self.status_label.setText(f"Failed to load devices ({resp.status_code})")
                return

            data = resp.json()
            devices = data.get('data') or data.get('devices') or data
            if isinstance(devices, dict):
                devices = devices.get('devices', [])
            self.devices = [d for d in devices if isinstance(d, dict) and d.get('type') == 'camera']

            self.devices_list.clear()
            if not self.devices:
                self.devices_list.addItem(QListWidgetItem("No cameras returned from /api/devices"))
            else:
                for cam in self.devices:
                    name = cam.get('name') or cam.get('id') or 'Camera'
                    ip = cam.get('ip_address') or cam.get('ip') or cam.get('rtsp_url') or ''
                    item = QListWidgetItem(f"{name} • {ip}")
                    item.setToolTip(cam.get('id', ''))
                    item.setData(Qt.ItemDataRole.UserRole, cam.get('id'))
                    item.setData(Qt.ItemDataRole.UserRole + 1, cam)
                    self.devices_list.addItem(item)

            self.status_label.setText(f"Loaded {len(self.devices)} camera(s) from shared API")
        except Exception as e:
            self.status_label.setText(f"Error loading devices: {e}")

    def _sync_camera_manager(self):
        """
        Immediately push new/updated cameras into the live CameraManager so they
        appear in PyQt lists without waiting for the filesystem polling cycle.
        """
        if not self.camera_manager:
            return

        loop = getattr(self.camera_manager, "_loop", None)
        if not loop or not getattr(loop, "is_running", lambda: False)():
            return

        self.status_label.setText("Camera saved; syncing PyQt manager…")

        async def _do_sync():
            api_synced = 0
            try:
                api_synced = await self.camera_manager.sync_cameras_api_to_db(self.API_BASE, prune_missing=True)
            except Exception:
                api_synced = 0
            if not api_synced:
                await self.camera_manager.sync_cameras_json_to_db()
            # Desktop-light: do not auto-connect all cameras on config changes.

        try:
            future = asyncio.run_coroutine_threadsafe(_do_sync(), loop)
        except Exception as exc:
            QTimer.singleShot(0, lambda: self.status_label.setText(f"Camera saved; sync pending: {exc}"))
            return

        def _handle_result(fut):
            message = "Camera ready in PyQt"
            try:
                fut.result()
            except Exception as exc:  # pragma: no cover - UI hint only
                message = f"Camera saved; sync later: {exc}"
            QTimer.singleShot(0, lambda: self.status_label.setText(message))

        future.add_done_callback(_handle_result)

    def test_connection(self):
        rtsp_url = self._build_rtsp_url()
        if not rtsp_url:
            QMessageBox.warning(self, "Missing info", "Provide an IP or full RTSP URL before testing.")
            return
        try:
            resp = requests.post(f"{self.API_BASE}/test-rtsp", json={"rtsp_url": rtsp_url}, timeout=6)
            if resp.ok:
                QMessageBox.information(self, "Connection Test", "RTSP connection successful.")
            else:
                QMessageBox.warning(self, "Connection Test", f"Test failed ({resp.status_code}).")
        except Exception as e:
            QMessageBox.warning(self, "Connection Test", f"Test failed: {e}")

    def save_camera(self):
        payload = self._current_payload()
        if not payload["name"]:
            QMessageBox.warning(self, "Missing info", "Camera name is required.")
            return
        if not payload["rtsp_url"]:
            QMessageBox.warning(self, "Missing info", "Provide camera details to build an RTSP URL.")
            return
        if not self.editing_camera_id and not self._can_add_camera():
            return

        try:
            if self.editing_camera_id:
                resp = requests.put(f"{self.API_BASE}/devices/{self.editing_camera_id}", json=payload, timeout=8)
            else:
                resp = requests.post(f"{self.API_BASE}/devices", json=payload, timeout=8)
            if resp.ok:
                resp_json = {}
                try:
                    resp_json = resp.json() or {}
                except Exception:
                    pass
                camera_data = resp_json.get("data") or resp_json

                self.status_label.setText("Camera saved via /api/devices")
                self.load_devices()
                self._sync_camera_manager()
                if self.editing_camera_id:
                    self.camera_updated.emit(camera_data if isinstance(camera_data, dict) else {})
                    self._set_edit_mode(None)
                    QMessageBox.information(self, "Camera Updated", "Camera updated successfully. It will appear in both React and PyQt.")
                else:
                    self.camera_added.emit(camera_data if isinstance(camera_data, dict) else {})
                    QMessageBox.information(self, "Camera Saved", "Camera added successfully. It will appear in both React and PyQt.")
            else:
                error_txt = resp.text
                try:
                    j = resp.json()
                    error_txt = j.get('message', error_txt)
                except Exception:
                    pass
                QMessageBox.warning(self, "Save Failed", f"Unable to save camera: {error_txt}")
        except Exception as e:
            QMessageBox.critical(self, "Save Failed", f"Error saving camera: {e}")

    def _can_add_camera(self) -> bool:
        """Check server entitlement before allowing camera creation."""
        try:
            resp = requests.get(f"{self.API_BASE}/system/entitlement", timeout=5)
            if not resp.ok:
                return True
            data = resp.json().get("data", {})
            limit = int(data.get("camera_limit", 4))
            count = int(data.get("camera_count", 0))
            if count >= limit:
                msg = QMessageBox(self)
                msg.setIcon(QMessageBox.Icon.Warning)
                msg.setWindowTitle("Camera Limit Reached")
                msg.setText(
                    f"You've reached the {limit}-camera limit for the beta. Disable a camera to stay within the beta limit."
                )
                msg.addButton(QMessageBox.StandardButton.OK)
                msg.exec()
                return False
        except Exception:
            # If the entitlement check fails, don't block the user.
            return True
        return True

    def edit_selected_camera(self):
        items = self.devices_list.selectedItems()
        if not items:
            QMessageBox.information(self, "Edit Camera", "Select a camera to edit.")
            return
        if len(items) != 1:
            QMessageBox.information(self, "Edit Camera", "Select exactly one camera to edit.")
            return

        item = items[0]
        cid = item.data(Qt.ItemDataRole.UserRole) or item.toolTip() or ""
        cam = item.data(Qt.ItemDataRole.UserRole + 1) or {}
        if not cid:
            QMessageBox.warning(self, "Edit Camera", "Unable to determine camera ID.")
            return
        if not isinstance(cam, dict) or not cam:
            # Fallback: try to find it from the last loaded list
            cam = next((d for d in (self.devices or []) if isinstance(d, dict) and d.get("id") == cid), {})

        self._set_edit_mode(str(cid))
        if isinstance(cam, dict):
            self._populate_form_from_camera(cam)

    def delete_selected_cameras(self):
        items = self.devices_list.selectedItems()
        if not items:
            QMessageBox.information(self, "Delete Cameras", "Select one or more cameras to delete.")
            return

        cam_ids = []
        names = []
        for item in items:
            cid = item.data(Qt.ItemDataRole.UserRole) or item.toolTip() or ""
            if cid:
                cam_ids.append(cid)
                names.append(item.text())

        if not cam_ids:
            QMessageBox.warning(self, "Delete Cameras", "Unable to determine camera IDs for deletion.")
            return

        confirm = QMessageBox.question(
            self,
            "Delete Cameras",
            f"Delete {len(cam_ids)} camera(s)?\n\n" + "\n".join(names),
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No
        )
        if confirm != QMessageBox.StandardButton.Yes:
            return

        failures = []
        for cid in cam_ids:
            try:
                resp = requests.delete(f"{self.API_BASE}/devices/{cid}", timeout=6)
                if not resp.ok:
                    failures.append(f"{cid} ({resp.status_code})")
            except Exception as e:
                failures.append(f"{cid} ({e})")

        if failures:
            QMessageBox.warning(self, "Delete Cameras", f"Some deletions failed:\n" + "\n".join(failures))
        else:
            QMessageBox.information(self, "Delete Cameras", f"Deleted {len(cam_ids)} camera(s).")

        self.load_devices()
        self._sync_camera_manager()
        self.camera_deleted.emit(cam_ids)


class CameraWidget(BaseDesktopWidget):
    # Prevent motion-watch enrich/ingest work from piling up and stealing CPU over time.
    _motion_watch_ingest_sem = threading.Semaphore(1)
    # Global capture throttle: if many cameras trigger at once, cap concurrent capture workers
    # so live rendering stays responsive.
    _motion_watch_capture_sem = threading.Semaphore(2)
    """
    Desktop Widget wrapper for the OpenGL Camera view.
    Connects to the shared CameraManager to fetch frames.
    Also connects to Socket.IO for AI detections.
    """
    detections_signal = Signal(list)
    ptz_result_signal = Signal(dict)
    _playback_data_ready = Signal(list, bool, list)  # segments, use_mediamtx, extra_dirs
    _playback_events_ready = Signal(list)

    def __init__(self, camera_id, camera_manager=None):
        super().__init__(title=f"Camera: {camera_id}", width=640, height=360) # Default to 16:9
        self.camera_id = camera_id
        self.camera_manager = camera_manager
        self.camera_name = camera_id
        self.camera_ip = None
        self.running = True
        self.aspect_ratio_locked = True
        self.keep_aspect_ratio = True # Enforce window resize constraint by default
        self.has_set_initial_size = False
        self.debug_overlay_enabled = False
        # Debug overlay behavior:
        # - Users can toggle it manually via context menu.
        # - Depth/Object Detection toggles can auto-enable it temporarily to show status/errors.
        self._debug_overlay_user_forced: bool = False
        # If the user manually turns debug OFF, do not auto-reenable it until they
        # explicitly trigger a feature that needs it again (depth/detection toggles).
        self._debug_overlay_user_opt_out: bool = False
        self._debug_overlay_auto_reasons: set[str] = set()
        self._debug_overlay_auto_timers: Dict[str, QTimer] = {}
        # Auto-protection (load shedder) per-widget state.
        # `_shed_active_level` is the current load level being applied
        # to this widget (0=NORMAL = nothing).  `_shed_force_sub` is a
        # separate flag from the user's _quality_pinned so a manual
        # pin to MAIN can be restored cleanly when the shed mode exits.
        # `_shed_saved_state` snapshots the values we override on
        # first entry into shed mode so they can be restored exactly.
        # `_debug_load_shed_lines` feeds extra text into the debug
        # overlay aggregator (`_refresh_debug_extra_lines`).
        self._shed_active_level: int = 0
        self._shed_force_sub: bool = False
        self._shed_saved_state: dict = {}
        self._shed_workers_paused: bool = False
        self._shed_camera_released: bool = False
        self._debug_load_shed_lines: List[str] = []

        # ---- Chord-key quick-adjust ----
        # Hold a "chord modifier" key (e.g. M for motion, D for detection)
        # then press an arrow key to nudge the related setting in real
        # time without opening the full settings dialog.  Single-tapping
        # the modifier still performs its original toggle action; we
        # only intercept it as a chord when an arrow is pressed before
        # the modifier is released.
        self._chord_modifier_key: Optional[int] = None
        self._chord_used: bool = False
        # `_chord_saw_autorepeat`: True once we've received an
        # autorepeat press of the modifier. Distinguishes a quick
        # tap (no autorepeat -> trust the release event) from a
        # held interaction (autorepeat seen -> on X11 release
        # events between autorepeats may not carry isAutoRepeat=True,
        # so defer them briefly and cancel if more activity arrives).
        self._chord_saw_autorepeat: bool = False
        # Single deferred-release timer (re-used for the lifetime of
        # the widget). When it fires, the chord ends. Any keyboard
        # activity while it's pending CANCELS it -- the user is
        # clearly still engaged.
        self._chord_release_timer: Optional[QTimer] = None
        # How long after the LAST keyboard event to keep the chord
        # alive (only applies once an autorepeat has been seen).
        # Long enough to absorb X11 autorepeat alternation gaps and
        # the inter-press delay when the user hops between secondary
        # keys, short enough not to feel laggy on real release.
        self._CHORD_RELEASE_DEFER_MS = 220
        # Transient feedback line shown in the debug overlay while the
        # chord is active or for ~1.5s after a value change.
        self._debug_chord_lines: List[str] = []
        # Chord/toggle takes DIRECT ownership of the debug overlay
        # during user interaction (bypasses the auto_reasons /
        # user_opt_out machinery entirely so the user always sees
        # their feedback, even if they previously opted out of the
        # overlay).  These remember the prior overlay state so we
        # can restore it cleanly on chord/toggle release.
        self._chord_prev_show_debug: Optional[bool] = None
        self._chord_prev_focus_mode: Optional[bool] = None
        self._chord_prev_extra_lines: Optional[List[str]] = None
        self._chord_owns_overlay: bool = False
        # Pending auto-clear timer.  Cancelled whenever a new chord
        # interaction starts so a stale toggle/chord clear doesn't
        # close the overlay mid-action.
        self._chord_clear_timer: Optional[QTimer] = None
        self.motion_boxes_enabled = False
        self.object_detection_enabled = False  # Backend detection state
        # Desktop-local object detection state (Desktop-first)
        self.desktop_object_detection_enabled: bool = False
        self.desktop_detector_config = DetectorConfig(
            enabled=False,
            fps_limit=8,
            backend="mobilenetssd",
            model_variant="default",
            device="auto",
            imgsz=640,
            min_confidence=0.35,
            max_det=100,
            allowed_classes=None,
        )
        # Desktop tracking config is stored inside desktop_detector_config (tracking_enabled/tracker_type/tracker_params).
        self._desktop_detector_worker: Optional[YoloDetectorWorker] = None
        self._desktop_detector_status_text: Optional[str] = None
        self._desktop_detector_last_error: Optional[str] = None
        self._desktop_detector_last_stats: dict = {}
        self._desktop_tracker_last_stats: dict = {}
        # Cached from worker stats (populates Allowed Classes UI without loading Ultralytics in UI)
        self._desktop_detector_class_names: List[str] = []
        self._desktop_detector_first_ready_seen: bool = False
        self.stream_quality = 'medium'  # low (sub) or medium (main)
        # Auto stream-quality selection state.
        # - When the user toggles quality manually, _quality_pinned is set
        #   True and the auto policy is disabled for this widget until the
        #   user explicitly re-enables it.
        # - _auto_quality_target is the most recent computed target ('low'
        #   or 'medium') so we don't keep re-issuing the same switch.
        self._quality_pinned: bool = False
        self._auto_quality_enabled: bool = True
        self._auto_quality_target: Optional[str] = None
        self._auto_quality_timer: Optional[QTimer] = None
        # Hysteresis: switch to MAIN at >=this many pixels wide,
        # back to SUB at <=this many pixels wide.  Default tile size is
        # 640x360, and the user wants the default tile and anything
        # smaller to use the sub stream; only larger ("focused") tiles
        # and fullscreen get main.  Hysteresis gap of ~120px prevents
        # flap when a tile sits near the boundary.
        self._auto_quality_to_main_w: int = 800
        self._auto_quality_to_sub_w: int = 680
        # Depth overlay (DepthAnythingV2)
        self.depth_overlay_config = DepthOverlayConfig(enabled=False)
        self._depth_worker: Optional[DepthAnythingOverlayWorker] = None
        self._depth_last_error: Optional[str] = None
        self._depth_status_text: Optional[str] = None
        self._depth_last_stats: dict = {}
        self._depth_first_ready_seen: bool = False
        # Object detection debug status (Socket.IO + enable/disable)
        self._detector_status_text: Optional[str] = None
        self._detector_last_error: Optional[str] = None
        # Motion watch state
        self.motion_watch_active = False
        self.motion_watch_settings = {
            "duration_sec": 30,
            "duration_unit": "Seconds",
            "trigger_delay_ms": 500,
            "crop_w": 0,
            "crop_h": 0,
            "resize_w": 0,
            "include_overlays": True,
            "save_dir": "captures/motion_watch",
            "allow_zone": True,
            "allow_line": True,
            "allow_tag": True,
            "cooldown_sec": 3,
            "clip_enabled": False,
            "clip_duration_sec": 10,
            "clip_pre_roll_sec": 5,
            "clip_resize_w": 640,
            "clip_quality": 23,
            "clip_save_dir": "",
        }
        self.motion_watch_end_ts = 0.0
        self.motion_watch_last_trigger = 0.0
        self.motion_watch_timer = QTimer(self)
        self.motion_watch_timer.timeout.connect(self._tick_motion_watch)
        # Avoid UI stalls during motion-watch captures (zone/line/tag triggers) by doing heavy
        # image encode + disk + base64 work off the UI thread, and prevent overlapping captures.
        self._motion_watch_capture_lock = threading.Lock()
        self._motion_watch_capture_inflight = False
        
        # Setup OpenGL Widget
        self.shapes_by_camera: Dict[str, List[Shape]] = {}
        self.gl_widget = CameraOpenGLWidget(parent=self, camera_id=camera_id)
        self.gl_widget.shapes_changed.connect(self._on_shapes_changed)
        self.gl_widget.shape_triggered.connect(self._on_shape_triggered)
        self.gl_widget.shape_double_clicked.connect(self.edit_shape)
        self.register_drag_handle(self.gl_widget, self.gl_widget.can_start_window_drag)
        self.set_content(self.gl_widget)
        # Load per-camera tracked-object overrides (name/color/hide)
        self._object_overrides: Dict[int, dict] = {}
        self._load_object_overrides()
        try:
            self.gl_widget.set_object_overrides(self._object_overrides)
        except Exception:
            pass
        self.shapes_by_camera[self.camera_id] = self.shapes_by_camera.get(self.camera_id, [])
        self.gl_widget.set_shapes(self.shapes_by_camera[self.camera_id])
        # Overlay projections: allow multiple independent overlay windows per camera / per shape selection.
        # Keyed by shape id (single selection) or by a stable group key (multi-selection).
        self.overlay_windows: Dict[str, CameraOverlayWindow] = {}
        self.ptz_overlay: Optional[PTZOverlayWidget] = None
        self.ptz_controller_window: Optional[PTZControllerWindow] = None
        self.ptz_overlay_settings: Dict[str, object] = load_ptz_overlay_settings(self.camera_id)
        self._ptz_sweep_active: bool = False
        # Audio EQ overlay/window
        self.audio_eq_overlay: Optional[AudioEQOverlayWidget] = None
        self.audio_eq_window: Optional[AudioEQWindow] = None
        self.audio_eq_settings = AudioEQSettings()
        self._audio_receiver: Optional[AudioWHEPReceiver] = None
        self._audio_playback: Optional[AudioPlayback] = None
        self._audio_playing: bool = False
        # Used to avoid stopping/restarting audio when switching dock <-> undock.
        self._audio_eq_switching_ui: bool = False
        # Auto-reconnect for audio receiver
        self._audio_reconnect_attempts: int = 0
        self._audio_reconnect_scheduled: bool = False
        # Simple recording support (stores into data/audio_clips/)
        self._audio_ring: List[bytes] = []
        self._audio_ring_bytes: int = 0
        self._audio_ring_max_bytes: int = 48000 * 2 * 2 * 12  # ~12s at 48k stereo int16
        self._audio_recording: bool = False
        self._audio_record_deadline: float = 0.0
        self._audio_record_buf: List[bytes] = []

        # Continuous recording state (MediaMTX passthrough, zero CPU)
        self._continuous_recording: bool = False
        # Playback overlay
        self.playback_overlay: Optional[PlaybackOverlayWidget] = None
        self._playback_active: bool = False
        self._playback_loading: bool = False
        self._playback_custom_dir: Optional[str] = None
        self._sync_group_id: Optional[str] = None
        self._sync_start_ts: Optional[float] = None
        self._sync_start_speed: float = 1.0
        self._sync_broadcasting: bool = False  # guard against recursion
        self._playback_data_ready.connect(self._on_playback_data_loaded)
        self._playback_events_ready.connect(self._on_playback_events_loaded)

        # Video clip ring buffer (JPEG-compressed frames for pre-roll).
        # Only actively fed when clip recording is enabled in motion watch settings.
        self._clip_frame_ring: deque = deque(maxlen=150)  # ~10s at 15fps
        self._clip_ring_fps: float = 15.0
        self._clip_ring_last_push: float = 0.0
        self._clip_recording: bool = False
        self._clip_record_buf: list = []
        self._clip_record_deadline: float = 0.0
        self._clip_record_trigger: Optional[dict] = None
        self._clip_record_lock = threading.Lock()

        # Load persisted motion watch defaults (per camera if available)
        self._load_motion_watch_settings()

        # Pull friendly name/IP if available
        self._hydrate_camera_metadata()

        # Periodic recording state sync (every 10s)
        self._rec_sync_timer = QTimer(self)
        self._rec_sync_timer.timeout.connect(self._sync_recording_state)
        self._rec_sync_timer.start(10_000)

        # On-demand: acquire camera stream when widget is created (ensures RTSP capture starts only when needed).
        try:
            import asyncio
            cm = getattr(self, "camera_manager", None)
            loop = getattr(cm, "_loop", None) if cm else None
            if cm and loop and getattr(loop, "is_running", lambda: False)():
                asyncio.run_coroutine_threadsafe(cm.acquire_camera(self.camera_id), loop)
        except Exception:
            pass
        
        # Hide the title bar for borderless look
        self.title_bar.hide()
        self.main_layout.setContentsMargins(0,0,0,0)
        self.content_layout.setContentsMargins(0,0,0,0) # Ensure content area has no margins
        self.content_layout.setSpacing(0)

        # Loading indicator (discreet, lightweight)
        self.loading_label = QLabel("Loading")
        self.loading_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.loading_label.setStyleSheet("""
            QLabel {
                color: #d0e8ff;
                background-color: rgba(0, 0, 0, 0.45);
                border: 1px solid rgba(255, 255, 255, 0.15);
                padding: 6px 10px;
                border-radius: 6px;
                font-weight: 500;
            }
        """)
        self.loading_phase = 0
        self.loading_timer = QTimer(self)
        self.loading_timer.timeout.connect(self._tick_loading)
        self._start_loading()
        self.register_drag_handle(self.loading_label)
        self.content_layout.addWidget(self.loading_label)

        # Offline overlay
        self.offline_label = QLabel("Camera offline or unreachable")
        self.offline_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.offline_label.setStyleSheet("""
            QLabel {
                color: #ffb3b3;
                background-color: rgba(0, 0, 0, 0.6);
                border: 1px solid rgba(255, 255, 255, 0.2);
                padding: 8px 12px;
                border-radius: 6px;
            }
        """)
        self.offline_label.hide()
        self.register_drag_handle(self.offline_label)
        self.content_layout.addWidget(self.offline_label)
        self.last_frame_time = None
        self.offline_timer = QTimer(self)
        self.offline_timer.timeout.connect(self._update_offline_state)
        self.offline_timer.start(2000)
        
        self.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.customContextMenuRequested.connect(self.show_context_menu)
        
        # Connect signals
        self.detections_signal.connect(self.gl_widget.update_backend_detections)
        self.ptz_result_signal.connect(self._on_ptz_result)

        # Apply styles
        self.central_widget.setStyleSheet(f"""
            QWidget#Central {{
                background-color: black;
                border: none;
            }}
        """)

        # Check initial detection status
        self._check_object_detection_status()

        # Apply global detector defaults (from tray prefs) so new windows use the active model/backend.
        try:
            app = QApplication.instance()
            if app is not None and hasattr(app, "_get_detector_prefs"):
                det = app._get_detector_prefs()
                if isinstance(det, dict):
                    self.apply_global_detector_defaults(det)
        except Exception:
            pass
        
        # Start SocketIO client
        self.sio = None
        threading.Thread(target=self._run_socketio, daemon=True).start()

    def apply_global_detector_defaults(self, det: dict) -> None:
        """
        Called by tray menu when changing active backend/model.
        """
        try:
            backend = str(det.get("backend") or "mobilenetssd").strip()
            model = str(det.get("model") or "default").strip()
        except Exception:
            backend = "mobilenetssd"
            model = "default"

        try:
            cfg = getattr(self, "desktop_detector_config", None) or DetectorConfig()
            cfg.backend = backend  # type: ignore[attr-defined]
            cfg.model_variant = model
            self.desktop_detector_config = cfg
        except Exception:
            return

        try:
            if self._desktop_detector_worker is not None:
                self._desktop_detector_worker.update_config(self.desktop_detector_config)
        except Exception:
            pass
        self._update_debug_model_info()

    def _load_object_overrides(self):
        try:
            from desktop.utils.desktop_object_overrides import load_camera_overrides

            per = load_camera_overrides(self.camera_id)
            self._object_overrides = {int(tid): ov.to_dict() for tid, ov in (per or {}).items()}
        except Exception:
            self._object_overrides = {}

    def _save_object_override(self, track_id: int, override_dict: dict):
        try:
            from desktop.utils.desktop_object_overrides import ObjectOverride, set_track_override, delete_track_override

            if isinstance(override_dict, dict) and bool(override_dict.get("__delete__")):
                delete_track_override(self.camera_id, int(track_id))
            else:
                ov = ObjectOverride.from_dict(override_dict)
                set_track_override(self.camera_id, int(track_id), ov)
        except Exception:
            pass
        self._load_object_overrides()
        try:
            self.gl_widget.set_object_overrides(self._object_overrides)
        except Exception:
            pass

    # ==================== Debug Overlay Auto-Show Helpers ====================

    def _apply_overlay_settings(self):
        """Apply overlay settings to the GL widget.

        While the chord-key UI owns the debug overlay, force
        show_debug=True so unrelated callers can't make the chord
        menu vanish mid-interaction. The chord ownership snapshot
        already remembers the user's preferred show_debug state and
        will restore it correctly on chord release.
        """
        try:
            chord_owned = bool(getattr(self, "_chord_owns_overlay", False))
            debug_arg = True if chord_owned else self.debug_overlay_enabled
            self.gl_widget.set_overlay_settings(debug_arg, self.motion_boxes_enabled)
            if chord_owned:
                try:
                    self.gl_widget._debug_focus_mode = True
                except Exception:
                    pass
        except Exception:
            pass
        self._update_debug_model_info()

    def _cancel_auto_debug_timer(self, reason: str):
        try:
            t = (self._debug_overlay_auto_timers or {}).pop(str(reason), None)
            if t is not None:
                t.stop()
                t.deleteLater()
        except Exception:
            pass

    def _clear_auto_debug(self):
        try:
            for r in list((self._debug_overlay_auto_timers or {}).keys()):
                self._cancel_auto_debug_timer(r)
        except Exception:
            pass
        try:
            (self._debug_overlay_auto_reasons or set()).clear()
        except Exception:
            self._debug_overlay_auto_reasons = set()

    def _auto_show_debug_overlay(self, reason: str):
        """Enable debug overlay temporarily unless user explicitly forced it on."""
        if bool(getattr(self, "_debug_overlay_user_forced", False)):
            return
        if bool(getattr(self, "_debug_overlay_user_opt_out", False)):
            return
        try:
            self._debug_overlay_auto_reasons.add(str(reason))
        except Exception:
            self._debug_overlay_auto_reasons = {str(reason)}
        if not bool(self.debug_overlay_enabled):
            self.debug_overlay_enabled = True
            self._apply_overlay_settings()
        else:
            self._update_debug_model_info()

    def _auto_hide_debug_overlay_after_success(self, reason: str, delay_ms: int = 3000):
        """Schedule debug overlay auto-hide after a successful operation."""
        if bool(getattr(self, "_debug_overlay_user_forced", False)):
            return
        r = str(reason)
        self._cancel_auto_debug_timer(r)
        t = QTimer(self)
        t.setSingleShot(True)

        def _fire():
            try:
                self._debug_overlay_auto_reasons.discard(r)
            except Exception:
                pass
            self._cancel_auto_debug_timer(r)
            # Only disable debug overlay if no auto reasons remain and user didn't force it.
            if (not bool(getattr(self, "_debug_overlay_user_forced", False))) and (not bool(self._debug_overlay_auto_reasons)):
                if bool(self.debug_overlay_enabled):
                    self.debug_overlay_enabled = False
                    self._apply_overlay_settings()

        t.timeout.connect(_fire)
        self._debug_overlay_auto_timers[r] = t
        t.start(int(delay_ms))

    # ==================== Audio EQ (WHEP) ====================

    def _ensure_audio_pipeline(self):
        if self._audio_receiver is None:
            self._audio_receiver = AudioWHEPReceiver(bars=self.audio_eq_settings.bars, fft_size=self.audio_eq_settings.fft_size)
            self._audio_receiver.state_changed.connect(self._on_audio_state)
            self._audio_receiver.error.connect(self._on_audio_error)
            self._audio_receiver.pcm_ready.connect(self._on_audio_pcm)
            self._audio_receiver.spectrum_ready.connect(self._on_audio_spectrum)
            self._audio_receiver.waveform_ready.connect(self._on_audio_waveform)
        if self._audio_playback is None:
            self._audio_playback = AudioPlayback(self)

    def _audio_api_base(self) -> str:
        # Keep in sync with the rest of the desktop widgets (ObjectDetectionSettingsDialog.API_BASE)
        try:
            return str(ObjectDetectionSettingsDialog.API_BASE).rstrip("/")
        except Exception:
            return "http://localhost:5000/api"

    def start_audio_monitor(self):
        self._ensure_audio_pipeline()
        if self._audio_receiver is None:
            return
        self._audio_playing = True
        self._audio_reconnect_attempts = 0
        # Start receiver thread; it will emit pcm + spectrum
        self._audio_receiver.start(self._audio_api_base(), self.camera_id)
        # Start audio playback (best effort)
        try:
            if self._audio_playback is not None:
                self._audio_playback.start()
        except Exception:
            pass
        self._sync_audio_ui_state()

    def stop_audio_monitor(self):
        self._audio_playing = False
        self._audio_reconnect_attempts = 0
        self._audio_reconnect_scheduled = False
        try:
            if self._audio_receiver is not None:
                self._audio_receiver.stop()
        except Exception:
            pass
        try:
            if self._audio_playback is not None:
                self._audio_playback.stop()
        except Exception:
            pass
        self._sync_audio_ui_state()

    def _toggle_audio_play(self):
        if self._audio_playing:
            self.stop_audio_monitor()
        else:
            self.start_audio_monitor()

    def _sync_audio_ui_state(self):
        try:
            if self.audio_eq_overlay is not None:
                self.audio_eq_overlay.hud.set_playing(bool(self._audio_playing))
        except Exception:
            pass
        try:
            if self.audio_eq_window is not None:
                self.audio_eq_window.hud.set_playing(bool(self._audio_playing))
        except Exception:
            pass

    def _on_audio_state(self, state: str):
        # Update HUD title subtly
        try:
            if self.audio_eq_overlay is not None:
                suffix = ""
                if state == "connecting":
                    suffix = " • connecting…"
                elif state == "connected":
                    suffix = " • live"
                elif state == "error":
                    suffix = " • error"
                self.audio_eq_overlay.hud.title.setText(f"Audio{suffix}")
        except Exception:
            pass
        try:
            if self.audio_eq_window is not None:
                suffix = ""
                if state == "connecting":
                    suffix = " • connecting…"
                elif state == "connected":
                    suffix = " • live"
                elif state == "error":
                    suffix = " • error"
                self.audio_eq_window.hud.title.setText(f"Audio{suffix}")
        except Exception:
            pass

        # If receiver fell back to idle/error while user still wants audio, attempt auto-reconnect.
        try:
            if self._audio_playing and state in ("idle", "error"):
                self._schedule_audio_reconnect()
        except Exception:
            pass

    def _on_audio_error(self, msg: str):
        try:
            if self.audio_eq_overlay is not None:
                self.audio_eq_overlay.hud.title.setText("Audio • error")
        except Exception:
            pass
        try:
            if self.audio_eq_window is not None:
                self.audio_eq_window.hud.title.setText("Audio • error")
        except Exception:
            pass
        try:
            print(f"[{self.camera_id}] Audio error: {msg}")
        except Exception:
            pass

        # Auto-reconnect if user still has audio toggled on.
        try:
            if self._audio_playing:
                self._schedule_audio_reconnect()
        except Exception:
            pass

    def _schedule_audio_reconnect(self):
        """Backoff reconnect for audio receiver so transient ICE failures recover automatically."""
        try:
            if not self._audio_playing:
                return
            if self._audio_reconnect_scheduled:
                return
            self._audio_reconnect_scheduled = True
            self._audio_reconnect_attempts = int(self._audio_reconnect_attempts) + 1
            delay_ms = int(min(10_000, 800 + (self._audio_reconnect_attempts * 700)))

            def _do():
                try:
                    self._audio_reconnect_scheduled = False
                    if not self._audio_playing:
                        return
                    # Restart receiver thread (safe even if already running)
                    if self._audio_receiver is not None:
                        self._audio_receiver.stop()
                    if self._audio_receiver is not None:
                        self._audio_receiver.start(self._audio_api_base(), self.camera_id)
                except Exception:
                    self._audio_reconnect_scheduled = False

            QTimer.singleShot(delay_ms, _do)
        except Exception:
            return

    def _on_audio_pcm(self, pcm: bytes):
        # Feed Qt audio output (must be GUI thread; signal delivers queued)
        try:
            if self._audio_playback is not None:
                self._audio_playback.push_pcm(pcm, gain=1.0)
        except Exception:
            pass

        # Maintain ring buffer for recording
        try:
            if pcm:
                self._audio_ring.append(pcm)
                self._audio_ring_bytes += len(pcm)
                while self._audio_ring and self._audio_ring_bytes > self._audio_ring_max_bytes:
                    dropped = self._audio_ring.pop(0)
                    self._audio_ring_bytes -= len(dropped)
        except Exception:
            pass

        # Active recording capture
        try:
            if self._audio_recording:
                self._audio_record_buf.append(pcm)
                if time.time() >= self._audio_record_deadline:
                    self._finish_audio_recording()
        except Exception:
            pass

    def _on_audio_spectrum(self, bars):
        try:
            if self.audio_eq_overlay is not None:
                self.audio_eq_overlay.hud.set_spectrum(list(bars or []))
        except Exception:
            pass
        try:
            if self.audio_eq_window is not None:
                self.audio_eq_window.hud.set_spectrum(list(bars or []))
        except Exception:
            pass

    def _on_audio_waveform(self, values):
        try:
            if self.audio_eq_overlay is not None:
                self.audio_eq_overlay.hud.set_waveform(list(values or []))
        except Exception:
            pass
        try:
            if self.audio_eq_window is not None:
                self.audio_eq_window.hud.set_waveform(list(values or []))
        except Exception:
            pass

    def _start_audio_recording(self, seconds: float = 6.0, pre_roll_s: float = 2.0):
        """Record a short clip from the live PCM stream into data/audio_clips/."""
        try:
            if self._audio_recording:
                return
            secs = float(max(1.0, min(30.0, seconds)))
            pre = float(max(0.0, min(10.0, pre_roll_s)))
            # seed with pre-roll from ring
            if pre > 0 and self._audio_ring:
                # approximate: take last N bytes from ring corresponding to pre seconds
                bytes_per_sec = 48000 * 2 * 2
                need = int(pre * bytes_per_sec)
                collected: List[bytes] = []
                total = 0
                for chunk in reversed(self._audio_ring):
                    collected.append(chunk)
                    total += len(chunk)
                    if total >= need:
                        break
                self._audio_record_buf = list(reversed(collected))
            else:
                self._audio_record_buf = []
            self._audio_recording = True
            self._audio_record_deadline = time.time() + secs
            try:
                print(f"[{self.camera_id}] Audio recording started ({secs:.1f}s, pre={pre:.1f}s)")
            except Exception:
                pass
        except Exception:
            return

    def _finish_audio_recording(self):
        try:
            if not self._audio_recording:
                return
            self._audio_recording = False

            import os
            import wave
            from pathlib import Path
            clips_dir = Path("data") / "audio_clips"
            clips_dir.mkdir(parents=True, exist_ok=True)
            clip_id = f"clip_{int(time.time()*1000)}_{self.camera_id[:8]}.wav"
            out_path = clips_dir / clip_id
            pcm = b"".join(self._audio_record_buf)
            self._audio_record_buf = []

            # Write WAV: 48kHz stereo s16le
            with wave.open(str(out_path), "wb") as wf:
                wf.setnchannels(2)
                wf.setsampwidth(2)
                wf.setframerate(48000)
                wf.writeframes(pcm)

            try:
                print(f"[{self.camera_id}] Audio clip saved: {out_path}")
            except Exception:
                pass

            # Optional: tag immediately
            try:
                from PySide6.QtWidgets import QInputDialog
                txt, ok = QInputDialog.getText(self, "Tag Audio Clip", "Tags (comma-separated):")
                if ok and str(txt).strip():
                    tags = [t.strip() for t in str(txt).split(",") if t.strip()]
                    self._write_audio_clip_meta(clip_id, tags)
            except Exception:
                pass
        except Exception:
            return

    def _write_audio_clip_meta(self, clip_id: str, tags: List[str]):
        try:
            import json
            from pathlib import Path
            meta_path = Path("data") / "audio_clips_meta.json"
            try:
                existing = json.loads(meta_path.read_text()) if meta_path.exists() else {}
            except Exception:
                existing = {}
            rec = existing.get(clip_id, {}) if isinstance(existing, dict) else {}
            rec["clip_id"] = clip_id
            rec["camera_id"] = self.camera_id
            rec["camera_name"] = self.camera_name
            rec["tags"] = list(dict.fromkeys([str(t) for t in (tags or []) if str(t).strip()]))
            rec["updated_at"] = datetime.now().isoformat()
            existing[clip_id] = rec
            meta_path.write_text(json.dumps(existing, indent=2))
        except Exception:
            return

    def toggle_audio_eq_overlay(self, checked: Optional[bool] = None):
        want_show = bool(checked) if checked is not None else (self.audio_eq_overlay is None)
        if want_show:
            # If undocked is open, close it and dock instead.
            if self.audio_eq_window is not None:
                self._audio_eq_switching_ui = True
                try:
                    self.toggle_audio_eq_undocked(False)
                finally:
                    self._audio_eq_switching_ui = False
            if not self.audio_eq_overlay:
                self.audio_eq_overlay = AudioEQOverlayWidget(parent=self.gl_widget, settings=self.audio_eq_settings)
                self.audio_eq_overlay.setGeometry(self.gl_widget.rect())
                self.audio_eq_overlay.request_close.connect(lambda: self.toggle_audio_eq_overlay(False))
                # Wire controls
                self.audio_eq_overlay.hud.play_clicked.connect(self._toggle_audio_play)
                self.audio_eq_overlay.hud.mute_toggled.connect(lambda m: self._audio_playback.set_muted(m) if self._audio_playback else None)
                self.audio_eq_overlay.hud.volume_changed.connect(lambda v: self._audio_playback.set_volume(v) if self._audio_playback else None)
                self.audio_eq_overlay.hud.record_clicked.connect(lambda: self._start_audio_recording())
                self.audio_eq_overlay.hud.monitor_menu_requested.connect(self._open_audio_monitor_menu)
                # Auto-start audio when overlay is shown
                self.start_audio_monitor()
                self.audio_eq_overlay.show()
            self.audio_eq_overlay.raise_()
            return

        if self.audio_eq_overlay:
            try:
                # persist HUD state
                self.audio_eq_settings = self.audio_eq_overlay.settings
            except Exception:
                pass
            try:
                self.audio_eq_overlay.hide()
                self.audio_eq_overlay.setParent(None)
                self.audio_eq_overlay.deleteLater()
            except Exception:
                pass
            self.audio_eq_overlay = None
        # If nothing else uses audio, stop it
        if self.audio_eq_window is None and not self._audio_eq_switching_ui:
            self.stop_audio_monitor()

    def toggle_audio_eq_undocked(self, checked: Optional[bool] = None):
        want_undocked = bool(checked) if checked is not None else (self.audio_eq_window is None)
        if want_undocked:
            # Ensure docked overlay is closed so it doesn't obstruct the image
            if self.audio_eq_overlay:
                try:
                    self._audio_eq_switching_ui = True
                    try:
                        self.toggle_audio_eq_overlay(False)
                    finally:
                        self._audio_eq_switching_ui = False
                except Exception:
                    pass
            if self.audio_eq_window is None:
                self.audio_eq_window = AudioEQWindow(settings=self.audio_eq_settings, title=f"Audio EQ • {self.camera_name or self.camera_id}")
                self.audio_eq_window.closed.connect(lambda: self.toggle_audio_eq_undocked(False))
                # Wire controls
                self.audio_eq_window.hud.play_clicked.connect(self._toggle_audio_play)
                self.audio_eq_window.hud.mute_toggled.connect(lambda m: self._audio_playback.set_muted(m) if self._audio_playback else None)
                self.audio_eq_window.hud.volume_changed.connect(lambda v: self._audio_playback.set_volume(v) if self._audio_playback else None)
                self.audio_eq_window.hud.record_clicked.connect(lambda: self._start_audio_recording())
                self.audio_eq_window.hud.monitor_menu_requested.connect(self._open_audio_monitor_menu)
                self.audio_eq_window.show()
                self.start_audio_monitor()
            return

        # Closing undocked window
        if self.audio_eq_window is not None:
            try:
                # persist size/settings
                self.audio_eq_settings = self.audio_eq_window.settings
            except Exception:
                pass
            try:
                self.audio_eq_window.close()
            except Exception:
                pass
            self.audio_eq_window = None

        # If nothing else uses audio, stop it
        if self.audio_eq_overlay is None and not self._audio_eq_switching_ui:
            self.stop_audio_monitor()

    def _open_audio_monitor_menu(self):
        """
        Small menu/dialog to control backend audio monitoring strategies for this camera.
        Uses /api/audio/monitor/start|stop with config overrides.
        """
        try:
            from PySide6.QtWidgets import QDialog, QVBoxLayout, QFormLayout, QComboBox, QSlider, QLabel, QDialogButtonBox, QCheckBox
        except Exception:
            return

        # Lazy-load current status
        api = self._audio_api_base().rstrip("/")

        dlg = QDialog(self)
        dlg.setWindowTitle(f"Audio Monitor • {self.camera_name or self.camera_id}")
        root = QVBoxLayout(dlg)
        form = QFormLayout()
        root.addLayout(form)

        enabled = QCheckBox("Enable audio monitoring on server for this camera")
        enabled.setChecked(False)
        form.addRow("", enabled)

        strategy = QComboBox()
        strategy.addItem("Energy Gate (events)", "energy_gate")
        strategy.addItem("Energy Gate + Profile Matching", "energy+profiles")
        form.addRow("Strategy", strategy)

        thr = QSlider(Qt.Orientation.Horizontal)
        thr.setRange(1, 40)  # 0.01 .. 0.40
        thr.setValue(6)
        thr_lbl = QLabel("0.06")
        thr.valueChanged.connect(lambda v: thr_lbl.setText(f"{v/100.0:.2f}"))
        row = QHBoxLayout()
        row.addWidget(thr, 1)
        row.addWidget(thr_lbl)
        form.addRow("Amplitude threshold", row)

        sim = QSlider(Qt.Orientation.Horizontal)
        sim.setRange(50, 95)  # 0.50 .. 0.95
        sim.setValue(75)
        sim_lbl = QLabel("0.75")
        sim.valueChanged.connect(lambda v: sim_lbl.setText(f"{v/100.0:.2f}"))
        row2 = QHBoxLayout()
        row2.addWidget(sim, 1)
        row2.addWidget(sim_lbl)
        form.addRow("Match similarity", row2)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Save | QDialogButtonBox.StandardButton.Cancel)
        root.addWidget(buttons)
        buttons.accepted.connect(dlg.accept)
        buttons.rejected.connect(dlg.reject)

        if dlg.exec() != QDialog.DialogCode.Accepted:
            return

        # Apply in background so UI doesn't block
        want = bool(enabled.isChecked())
        amplitude_threshold = float(thr.value() / 100.0)
        min_similarity = float(sim.value() / 100.0)
        strat = str(strategy.currentData() or "energy_gate")

        def _apply():
            try:
                import requests
                if want:
                    payload = {
                        "camera_id": self.camera_id,
                        "config": {
                            "amplitude_threshold": amplitude_threshold,
                            "match_min_similarity": min_similarity,
                            # strategy is currently informational; backend always does events + optional matching if profiles exist
                            "strategy": strat,
                        },
                    }
                    requests.post(f"{api}/audio/monitor/start", json=payload, timeout=8)
                else:
                    requests.post(f"{api}/audio/monitor/stop", json={"camera_id": self.camera_id}, timeout=8)
            except Exception:
                pass

        threading.Thread(target=_apply, daemon=True).start()

    # ==================== Depth Overlay (DepthAnythingV2) ====================

    def _ensure_depth_worker(self):
        if self._depth_worker is not None:
            return
        self._depth_worker = DepthAnythingOverlayWorker(self.depth_overlay_config)
        self._depth_worker.depth_ready.connect(self._on_depth_ready)
        self._depth_worker.status.connect(self._on_depth_status)
        self._depth_worker.error.connect(self._on_depth_error)
        self._depth_worker.start()
        self._update_debug_model_info()

    def _on_depth_ready(self, qimg, stats):
        try:
            self._depth_last_stats = stats or {}
        except Exception:
            self._depth_last_stats = {}
        self._depth_status_text = "Depth: running"
        self.gl_widget.set_depth_overlay(
            qimg,
            enabled=bool(self.depth_overlay_config.enabled),
            opacity=float(self.depth_overlay_config.opacity),
            # For pointcloud we want the live camera visible behind the visualization.
            # The pointcloud QImage has transparency for "empty" pixels, and opacity controls blending.
            replace_base=False,
            overlay_scale=float(getattr(self.depth_overlay_config, "overlay_scale", 1.0) or 1.0),
            blackout_base=bool(getattr(self.depth_overlay_config, "blackout_base", False)),
            camera_opacity=float(getattr(self.depth_overlay_config, "camera_opacity", 1.0) or 1.0),
            is_pointcloud=bool(getattr(self.depth_overlay_config, "colormap", None) == "pointcloud"),
        )
        # Treat first successful frame as "loaded" for purposes of auto-hiding debug overlay.
        if not bool(getattr(self, "_depth_first_ready_seen", False)):
            self._depth_first_ready_seen = True
            self._auto_hide_debug_overlay_after_success("depth", 3000)
        self._update_debug_model_info()

    def _on_depth_status(self, msg: str):
        self._depth_last_error = None
        self._depth_status_text = msg
        # Only auto-show debug for transitional / actionable statuses.
        # Avoid re-enabling debug continuously while depth is already running.
        try:
            m = str(msg or "")
        except Exception:
            m = ""
        if m.startswith("Depth: enabling") or m.startswith("Depth: loading") or m.startswith("Depth: switching") or m.startswith("Depth: waiting") or m.startswith("Depth: warming") or m.startswith("Depth: GPU"):
            self._auto_show_debug_overlay("depth")
        try:
            if isinstance(msg, str) and msg.startswith("Depth: ready"):
                self._auto_hide_debug_overlay_after_success("depth", 3000)
        except Exception:
            pass
        self._update_debug_model_info()

    def _on_depth_error(self, msg: str):
        self._depth_last_error = msg
        self._depth_status_text = None
        self._auto_show_debug_overlay("depth")
        self._update_debug_model_info()
        try:
            print(f"[{self.camera_id}] {msg}")
        except Exception:
            pass

    def toggle_depth_overlay(self, checked: Optional[bool] = None):
        want = bool(checked) if checked is not None else (not bool(self.depth_overlay_config.enabled))
        self.depth_overlay_config.enabled = want
        if want:
            self._depth_first_ready_seen = False
            # Depth toggle implies user wants visibility into load/errors; allow auto-debug again.
            self._debug_overlay_user_opt_out = False
            self._ensure_depth_worker()
            self._auto_show_debug_overlay("depth")
            self._depth_status_text = f"Depth: enabling ({self.depth_overlay_config.model_size} on {self.depth_overlay_config.device})…"
        if self._depth_worker is not None:
            self._depth_worker.update_config(self.depth_overlay_config)
        if not want:
            self._depth_status_text = "Depth: disabled"
            try:
                self._debug_overlay_auto_reasons.discard("depth")
            except Exception:
                pass
            self._cancel_auto_debug_timer("depth")
            self.gl_widget.set_depth_overlay(
                None,
                enabled=False,
                replace_base=False,
                overlay_scale=1.0,
                blackout_base=False,
                camera_opacity=1.0,
                is_pointcloud=False,
            )
            # If debug overlay was only auto-enabled for depth, turn it off immediately.
            if (not bool(getattr(self, "_debug_overlay_user_forced", False))) and (not bool(self._debug_overlay_auto_reasons)):
                if bool(self.debug_overlay_enabled):
                    self.debug_overlay_enabled = False
                    self._apply_overlay_settings()
        self._update_debug_model_info()

    def open_depth_settings(self):
        dlg = DepthOverlaySettingsDialog(self.depth_overlay_config, self)
        if dlg.exec() == QDialog.DialogCode.Accepted and dlg.result_config:
            # Preserve enabled state
            enabled = bool(self.depth_overlay_config.enabled)
            self.depth_overlay_config = dlg.result_config
            self.depth_overlay_config.enabled = enabled
            if self._depth_worker is not None:
                self._depth_worker.update_config(self.depth_overlay_config)
            # Apply opacity immediately even before next depth frame
            self.gl_widget.set_depth_overlay(
                self.gl_widget.depth_overlay_image,
                enabled=bool(self.depth_overlay_config.enabled),
                opacity=float(self.depth_overlay_config.opacity),
                replace_base=False,
                overlay_scale=float(getattr(self.depth_overlay_config, "overlay_scale", 1.0) or 1.0),
                blackout_base=bool(getattr(self.depth_overlay_config, "blackout_base", False)),
                camera_opacity=float(getattr(self.depth_overlay_config, "camera_opacity", 1.0) or 1.0),
                is_pointcloud=bool(getattr(self.depth_overlay_config, "colormap", None) == "pointcloud"),
            )
            self._depth_status_text = f"Depth: updated settings (viz={self.depth_overlay_config.colormap})"
            self._update_debug_model_info()

    def _build_model_info_lines(self) -> List[str]:
        """Compose the depth/detector/backend status lines that the
        debug overlay shows.  Pulled out of `_update_debug_model_info`
        so the load shedder can append its own lines via
        `_refresh_debug_extra_lines` without losing this content.
        """
        lines: List[str] = []
        try:
            # Depth overlay summary
            if getattr(self, "depth_overlay_config", None) is not None:
                cfg = self.depth_overlay_config
                lines.append(
                    f"Depth: {'ON' if cfg.enabled else 'off'} | {cfg.model_size} | dev={cfg.device} | fp16={cfg.use_fp16}"
                )
                if getattr(cfg, "colormap", None) == "pointcloud":
                    step = int(getattr(cfg, "pointcloud_step", 3) or 3)
                    density = max(1, min(24, 25 - max(1, min(24, step))))
                    pc_color = str(getattr(cfg, "pointcloud_color", "camera") or "camera")
                    pc_zoom = float(getattr(cfg, "pointcloud_zoom", 1.0) or 1.0)
                    ov_scale = float(getattr(cfg, "overlay_scale", 1.0) or 1.0)
                    blackout = bool(getattr(cfg, "blackout_base", False))
                    cam_op = float(getattr(cfg, "camera_opacity", 1.0) or 1.0)
                    lines.append(
                        f"Viz: pointcloud (1st person) | zoom={pc_zoom:.1f}x | density={density}/24 | step={step}px | color={pc_color}"
                    )
                    lines.append(
                        f"Overlay: scale={ov_scale:.1f}x | vizOpacity={float(getattr(cfg,'opacity',0.55)):.2f} | camOpacity={cam_op:.2f} | blackout={'ON' if blackout else 'off'}"
                    )
                else:
                    lines.append(f"Viz: {getattr(cfg, 'colormap', 'turbo')} | opacity={cfg.opacity:.2f}")

            if self._depth_status_text:
                lines.append(f"Status: {self._depth_status_text}")
            if self._depth_last_error:
                lines.append(f"Error: {self._depth_last_error}")

            # Object detection status (Desktop-local + backend transport)
            try:
                cfg = getattr(self, "desktop_detector_config", None)
                if cfg is not None:
                    lines.append(
                        f"Desktop detections: {'ON' if bool(self.desktop_object_detection_enabled) else 'off'} | {cfg.model_variant} | dev={cfg.device} | conf={float(cfg.min_confidence):.2f} | fps≤{int(cfg.fps_limit)}"
                    )
                    try:
                        lines.append(
                            f"Tracking: {'ON' if bool(getattr(cfg,'tracking_enabled',False)) else 'off'} | tracker={str(getattr(cfg,'tracker_type','sort'))}"
                        )
                    except Exception:
                        pass
            except Exception:
                pass
            if getattr(self, "_desktop_detector_status_text", None):
                lines.append(f"Desktop status: {self._desktop_detector_status_text}")
            if getattr(self, "_desktop_detector_last_error", None):
                lines.append(f"Desktop error: {self._desktop_detector_last_error}")

            try:
                lines.append(f"Backend detection enabled: {'ON' if bool(self.object_detection_enabled) else 'off'}")
            except Exception:
                pass
            if getattr(self, "_detector_status_text", None):
                lines.append(f"{self._detector_status_text}")
            if getattr(self, "_detector_last_error", None):
                lines.append(f"Backend error: {self._detector_last_error}")

            # Stats (best-effort)
            st = self._depth_last_stats or {}
            if isinstance(st, dict):
                try:
                    if st.get("avg_fps") is not None:
                        lines.append(f"Depth FPS(avg): {float(st.get('avg_fps')):.1f}")
                except Exception:
                    pass
                try:
                    dev = st.get("device")
                    if dev:
                        lines.append(f"Depth device: {dev}")
                except Exception:
                    pass

        except Exception:
            pass
        return lines

    def _update_debug_model_info(self):
        """Push model/runtime status into the GL widget's debug overlay.

        Combines the model/detector/depth status with any active
        load-shed status lines so neither source clobbers the other.
        Cheap to call on every state change; only visible when the
        debug overlay is on.
        """
        try:
            self._refresh_debug_extra_lines()
        except Exception:
            pass

    def _refresh_debug_extra_lines(self):
        """Aggregate all sources of debug-overlay extra text and push
        them to the GL widget in a single call.

        Sources currently include:
          * `_build_model_info_lines()` -- depth / detector / backend
          * `_debug_load_shed_lines`    -- auto-protection status
          * `_debug_chord_lines`        -- chord-key quick-adjust hints

        Special case: when the user is actively chord-adjusting (chord
        modifier held OR chord lines pending auto-hide), suppress the
        verbose model/load-shed walls so the overlay is laser-focused
        on the value the user is changing.  This is the difference
        between 'I can see my change' and 'my change scrolled past
        20 lines of detector status'.
        """
        try:
            chord_lines = list(getattr(self, "_debug_chord_lines", None) or [])
            chord_active = bool(chord_lines)

            lines: List[str] = []
            if chord_active:
                # Focused mode: chord lines only.
                lines.extend(chord_lines)
            else:
                try:
                    lines.extend(self._build_model_info_lines())
                except Exception:
                    pass
                try:
                    shed_lines = list(getattr(self, "_debug_load_shed_lines", None) or [])
                    if shed_lines:
                        lines.extend(shed_lines)
                except Exception:
                    pass

            self.gl_widget.set_debug_extra_lines(lines)
        except Exception:
            pass

    def open_model_library(self):
        if not hasattr(self, "_model_library_dialog") or self._model_library_dialog is None:
            self._model_library_dialog = ModelLibraryDialog(self)
            self._model_library_dialog.finished.connect(lambda: setattr(self, "_model_library_dialog", None))
        self._model_library_dialog.show()
        self._model_library_dialog.raise_()
        self._model_library_dialog.activateWindow()

    def _run_socketio(self):
        """Connect to backend Socket.IO for realtime detections."""
        import weakref
        self_ref = weakref.ref(self)
        cam_id = str(getattr(self, "camera_id", ""))

        def _ui_safe(fn):
            w = self_ref()
            if w is None:
                return
            try:
                if not bool(getattr(w, "running", True)):
                    return
            except Exception:
                return
            try:
                fn(w)
            except RuntimeError:
                return
            except Exception:
                return

        try:
            w0 = self_ref()
            if w0 is None:
                return
            try:
                if not bool(getattr(w0, "running", True)):
                    return
            except Exception:
                return

            w0.sio = socketio.Client()
            
            @w0.sio.on('connect', namespace='/realtime')
            def on_connect():
                w = self_ref()
                if w is None:
                    return
                try:
                    if not bool(getattr(w, "running", True)):
                        return
                except Exception:
                    return
                try:
                    w.sio.emit('subscribe', {'camera_id': w.camera_id}, namespace='/realtime')
                except Exception:
                    return
                # Update debug overlay (hop to UI thread)
                QTimer.singleShot(0, lambda: _ui_safe(lambda ww: (
                    setattr(ww, "_detector_last_error", None),
                    setattr(ww, "_detector_status_text", "Backend detections: connected"),
                    ww._auto_show_debug_overlay("detector"),
                    ww._auto_hide_debug_overlay_after_success("detector", 3000),
                    ww._update_debug_model_info(),
                )))

            @w0.sio.on('detections_update', namespace='/realtime')
            def on_detections(data):
                w = self_ref()
                if w is None:
                    return
                try:
                    if not bool(getattr(w, "running", True)):
                        return
                except Exception:
                    return
                # Data payload: {'camera_id': ..., 'detections': [...]}
                # Filter for this camera just in case
                if data.get('camera_id') == w.camera_id or not data.get('camera_id'):
                    dets = data.get('detections', [])
                    try:
                        w.detections_signal.emit(dets)
                    except RuntimeError:
                        return
                    except Exception:
                        return
            
            # Connect
            # Use default backend port
            w0.sio.connect('http://localhost:5000', namespaces=['/realtime'], wait=False)
            w0.sio.wait()
            
        except Exception as e:
            # Backend may be offline (desktop-light). Avoid noisy logs unless debug overlay is enabled.
            try:
                w = self_ref()
                if w is not None and bool(getattr(w, "debug_overlay_enabled", False)):
                    print(f"SocketIO Error for {cam_id}: {e}")
            except Exception:
                pass
            QTimer.singleShot(0, lambda: _ui_safe(lambda ww: (
                setattr(ww, "_detector_status_text", None),
                setattr(ww, "_detector_last_error", f"Backend detections SocketIO error: {e}"),
                ww._auto_show_debug_overlay("detector"),
                ww._update_debug_model_info(),
            )))

    # ==================== Desktop Object Detection (YOLO) ====================

    def _ensure_desktop_detector_worker(self):
        if self._desktop_detector_worker is not None:
            return
        self._desktop_detector_worker = YoloDetectorWorker(self.desktop_detector_config)
        self._desktop_detector_worker.detections_ready.connect(self._on_desktop_detections_ready)
        self._desktop_detector_worker.tracks_ready.connect(self._on_desktop_tracks_ready)
        self._desktop_detector_worker.status.connect(self._on_desktop_detector_status)
        self._desktop_detector_worker.error.connect(self._on_desktop_detector_error)
        self._desktop_detector_worker.start()
        self._update_debug_model_info()

    def _on_desktop_detections_ready(self, dets, stats):
        try:
            self._desktop_detector_last_stats = stats or {}
        except Exception:
            self._desktop_detector_last_stats = {}
        try:
            cn = (stats or {}).get("class_names")
            if isinstance(cn, list) and cn:
                self._desktop_detector_class_names = [str(x) for x in cn if x is not None and str(x).strip()]
        except Exception:
            pass
        self._desktop_detector_last_error = None
        self._desktop_detector_status_text = "Detections: running"
        try:
            self.gl_widget.update_desktop_detections(dets or [])
        except Exception:
            pass
        if not bool(getattr(self, "_desktop_detector_first_ready_seen", False)):
            self._desktop_detector_first_ready_seen = True
            self._auto_hide_debug_overlay_after_success("desktop_detector", 3000)
        self._update_debug_model_info()

    def _on_desktop_tracks_ready(self, tracks, stats):
        try:
            self._desktop_tracker_last_stats = stats or {}
        except Exception:
            self._desktop_tracker_last_stats = {}
        try:
            cn = (stats or {}).get("class_names")
            if isinstance(cn, list) and cn:
                self._desktop_detector_class_names = [str(x) for x in cn if x is not None and str(x).strip()]
        except Exception:
            pass
        # Push tracks to GL widget; it will decide whether to render tracks (tracking_active + non-empty list)
        try:
            self.gl_widget.update_desktop_tracks(tracks or [])
            cfg = getattr(self, "desktop_detector_config", None)
            self.gl_widget.set_desktop_tracking_active(bool(getattr(cfg, "tracking_enabled", False)))
        except Exception:
            pass
        self._update_debug_model_info()

    def _on_desktop_detector_status(self, msg: str):
        self._desktop_detector_last_error = None
        self._desktop_detector_status_text = msg
        # Auto-show debug for transitional / actionable statuses.
        try:
            m = str(msg or "")
        except Exception:
            m = ""
        if m.startswith("Detections: loading") or m.startswith("Detections: waiting") or m.startswith("Detections: warming") or m.startswith("Detections: ready"):
            self._auto_show_debug_overlay("desktop_detector")
        try:
            if isinstance(msg, str) and msg.startswith("Detections: ready"):
                self._auto_hide_debug_overlay_after_success("desktop_detector", 3000)
        except Exception:
            pass
        self._update_debug_model_info()

    def _on_desktop_detector_error(self, msg: str):
        self._desktop_detector_last_error = msg
        self._desktop_detector_status_text = None
        self._auto_show_debug_overlay("desktop_detector")
        self._update_debug_model_info()
        try:
            print(f"[{self.camera_id}] {msg}")
        except Exception:
            pass

    def _check_object_detection_status(self):
        def worker():
            try:
                # We can check config or status. Config usually has 'verification_enabled'
                url = f"{ObjectDetectionSettingsDialog.API_BASE}/cameras/{self.camera_id}/detection-config"
                res = requests.get(url, timeout=2)
                if res.ok:
                    data = res.json().get('data', {})
                    enabled = data.get('verification_enabled', False)
                    # Update UI in main thread if we had a visual indicator (we use context menu state)
                    self.object_detection_enabled = enabled
            except:
                pass
        threading.Thread(target=worker, daemon=True).start()

    def toggle_object_detection(self):
        """
        Desktop-first object detection toggle (Desktop-local YOLO worker).

        Note: The backend-driven detection pipeline remains intact and continues to emit detections
        over Socket.IO (/realtime) when enabled/configured; Desktop-local detections are rendered
        independently and do not depend on backend health.
        """
        current = bool(self.desktop_object_detection_enabled)
        new_state = not current

        # If enabling Desktop detection and we don't have a BYO ONNX model, we will fall back to MobileNetSSD.
        # Prompt to install it if missing so the "turn on detection and it works" path holds.
        if bool(new_state):
            try:
                from PySide6.QtWidgets import QMessageBox
                from core.model_library.byo_models import list_installed_manifests
                from core.model_library.vision_detection import check_mobilenetssd, ensure_mobilenetssd

                have_byo = False
                try:
                    have_byo = bool(list_installed_manifests())
                except Exception:
                    have_byo = False

                st = check_mobilenetssd()
                if (not have_byo) and (not st.ok):
                    self._desktop_detector_last_error = None
                    self._desktop_detector_status_text = "Detections: installing MobileNetSSD…"
                    self._auto_show_debug_overlay("desktop_detector")
                    self._update_debug_model_info()

                    def _bg_install():
                        try:
                            ensure_mobilenetssd(force_download=False)
                            QTimer.singleShot(0, lambda: self.toggle_object_detection())
                        except Exception as e:
                            def _ui_err():
                                self._desktop_detector_last_error = str(e)
                                self._desktop_detector_status_text = None
                                self._auto_show_debug_overlay("desktop_detector")
                                self._update_debug_model_info()
                                try:
                                    QMessageBox.warning(self, "MobileNetSSD install failed", str(e))
                                except Exception:
                                    pass
                            QTimer.singleShot(0, _ui_err)

                    threading.Thread(target=_bg_install, daemon=True).start()
                    return
            except Exception:
                pass

        self.desktop_object_detection_enabled = bool(new_state)
        self.desktop_detector_config.enabled = bool(new_state)

        # Prefer Desktop layer for overlay when enabled
        try:
            self.gl_widget.set_desktop_detection_active(bool(self.desktop_object_detection_enabled))
        except Exception:
            pass

        if hasattr(self, 'obj_det_action'):
            self.obj_det_action.setChecked(bool(self.desktop_object_detection_enabled))

        # Detection toggle implies user wants visibility into status/errors; allow auto-debug again.
        self._debug_overlay_user_opt_out = False
        self._auto_show_debug_overlay("desktop_detector")

        self._desktop_detector_last_error = None
        state_txt = "enabling" if bool(self.desktop_object_detection_enabled) else "disabling"
        self._desktop_detector_status_text = f"Detections: {state_txt}…"

        if bool(self.desktop_object_detection_enabled):
            self._desktop_detector_first_ready_seen = False
            self._ensure_desktop_detector_worker()
        try:
            if self._desktop_detector_worker is not None:
                self._desktop_detector_worker.update_config(self.desktop_detector_config)
        except Exception:
            pass

        if not bool(self.desktop_object_detection_enabled):
            # Clear desktop overlay when disabled
            try:
                self.gl_widget.update_desktop_detections([])
            except Exception:
                pass
            try:
                self.gl_widget.set_desktop_tracking_active(False)
            except Exception:
                pass
            self._desktop_detector_status_text = "Detections: disabled"

        self._update_debug_model_info()
        return

    def open_desktop_tracking_settings(self):
        """
        Desktop-local tracking settings (S0RT/ByteTrack) for Desktop detections only.
        """
        from PySide6.QtWidgets import (
            QDialog,
            QVBoxLayout,
            QFormLayout,
            QComboBox,
            QDialogButtonBox,
            QCheckBox,
            QSpinBox,
            QDoubleSpinBox,
            QPushButton,
            QLabel,
            QHBoxLayout,
        )

        dlg = QDialog(self)
        dlg.setWindowTitle(f"Desktop Tracking Settings - {self.camera_id}")
        dlg.setWindowFlags(Qt.WindowType.Tool | Qt.WindowType.WindowStaysOnTopHint)

        root = QVBoxLayout(dlg)
        form = QFormLayout()
        root.addLayout(form)

        cfg = getattr(self, "desktop_detector_config", None) or DetectorConfig()
        params = dict(getattr(cfg, "tracker_params", None) or {})

        enabled = QCheckBox("Enable tracking (stable IDs)")
        enabled.setChecked(bool(getattr(cfg, "tracking_enabled", False)))
        form.addRow("", enabled)

        tracker_type = QComboBox()
        tracker_type.addItem("SORT", "sort")
        tracker_type.addItem("ByteTrack", "bytetrack")
        tidx = tracker_type.findData(str(getattr(cfg, "tracker_type", "sort") or "sort"))
        if tidx >= 0:
            tracker_type.setCurrentIndex(tidx)
        form.addRow("Tracker", tracker_type)

        # Parameter widgets (we swap which rows are visible based on tracker type)
        # SORT params
        sort_max_age = QSpinBox()
        sort_max_age.setRange(1, 200)
        sort_max_age.setValue(int(params.get("max_age", 15) or 15))

        sort_min_hits = QSpinBox()
        sort_min_hits.setRange(1, 20)
        sort_min_hits.setValue(int(params.get("min_hits", 2) or 2))

        sort_iou = QDoubleSpinBox()
        sort_iou.setRange(0.0, 1.0)
        sort_iou.setSingleStep(0.05)
        sort_iou.setDecimals(2)
        sort_iou.setValue(float(params.get("iou_threshold", 0.30) or 0.30))

        # ByteTrack params
        bt_track = QDoubleSpinBox()
        bt_track.setRange(0.0, 1.0)
        bt_track.setSingleStep(0.05)
        bt_track.setDecimals(2)
        bt_track.setValue(float(params.get("track_thresh", 0.35) or 0.35))

        bt_low = QDoubleSpinBox()
        bt_low.setRange(0.0, 1.0)
        bt_low.setSingleStep(0.05)
        bt_low.setDecimals(2)
        bt_low.setValue(float(params.get("low_thresh", 0.10) or 0.10))

        bt_match = QDoubleSpinBox()
        bt_match.setRange(0.0, 1.0)
        bt_match.setSingleStep(0.05)
        bt_match.setDecimals(2)
        bt_match.setValue(float(params.get("match_thresh", 0.30) or 0.30))

        bt_buffer = QSpinBox()
        bt_buffer.setRange(1, 300)
        bt_buffer.setValue(int(params.get("track_buffer", 30) or 30))

        bt_min_area = QSpinBox()
        bt_min_area.setRange(0, 20000)
        bt_min_area.setValue(int(params.get("min_box_area", 10) or 10))

        # Add rows (we'll show/hide later)
        sort_rows = []
        bt_rows = []

        form.addRow("SORT: max_age", sort_max_age); sort_rows.append(sort_max_age)
        form.addRow("SORT: min_hits", sort_min_hits); sort_rows.append(sort_min_hits)
        form.addRow("SORT: iou_threshold", sort_iou); sort_rows.append(sort_iou)

        form.addRow("ByteTrack: track_thresh", bt_track); bt_rows.append(bt_track)
        form.addRow("ByteTrack: low_thresh", bt_low); bt_rows.append(bt_low)
        form.addRow("ByteTrack: match_thresh", bt_match); bt_rows.append(bt_match)
        form.addRow("ByteTrack: track_buffer", bt_buffer); bt_rows.append(bt_buffer)
        form.addRow("ByteTrack: min_box_area", bt_min_area); bt_rows.append(bt_min_area)

        help_lbl = QLabel("Note: Track IDs may reset between sessions or when you reset the tracker.")
        help_lbl.setStyleSheet("color: #c9d6e2;")
        root.addWidget(help_lbl)

        # Reset button (increments tracker_reset_token)
        reset_row = QHBoxLayout()
        reset_btn = QPushButton("Reset tracker (clear IDs)")
        reset_row.addWidget(reset_btn)
        reset_row.addStretch(1)
        root.addLayout(reset_row)

        reset_clicked = {"v": False}

        def _do_reset():
            reset_clicked["v"] = True
            try:
                self.gl_widget.set_desktop_tracking_active(False)
            except Exception:
                pass
            try:
                self.gl_widget.update_desktop_tracks([])
            except Exception:
                pass

        reset_btn.clicked.connect(_do_reset)

        def _apply_visibility():
            t = str(tracker_type.currentData() or "sort")
            show_sort = (t == "sort")
            for w in sort_rows:
                w.setVisible(show_sort)
                # also hide the label by hiding the buddy's parent layout item best-effort
                try:
                    form.labelForField(w).setVisible(show_sort)
                except Exception:
                    pass
            for w in bt_rows:
                w.setVisible(not show_sort)
                try:
                    form.labelForField(w).setVisible(not show_sort)
                except Exception:
                    pass

        tracker_type.currentIndexChanged.connect(_apply_visibility)
        _apply_visibility()

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Save | QDialogButtonBox.StandardButton.Cancel)
        root.addWidget(buttons)
        buttons.accepted.connect(dlg.accept)
        buttons.rejected.connect(dlg.reject)

        if dlg.exec() != QDialog.DialogCode.Accepted:
            return

        # Build updated config (preserve detection enabled state)
        new_params: dict = {}
        ttype = str(tracker_type.currentData() or "sort")
        if ttype == "bytetrack":
            new_params = {
                "track_thresh": float(bt_track.value()),
                "low_thresh": float(bt_low.value()),
                "match_thresh": float(bt_match.value()),
                "track_buffer": int(bt_buffer.value()),
                "min_box_area": float(bt_min_area.value()),
            }
        else:
            new_params = {
                "max_age": int(sort_max_age.value()),
                "min_hits": int(sort_min_hits.value()),
                "iou_threshold": float(sort_iou.value()),
            }

        # Preserve existing DetectorConfig fields; override tracking-related ones
        cfg.tracking_enabled = bool(enabled.isChecked())
        cfg.tracker_type = ttype
        cfg.tracker_params = new_params
        if bool(reset_clicked["v"]):
            try:
                cfg.tracker_reset_token = int(getattr(cfg, "tracker_reset_token", 0) or 0) + 1
            except Exception:
                cfg.tracker_reset_token = 1

        # Apply to worker if running
        try:
            if self._desktop_detector_worker is not None:
                self._desktop_detector_worker.update_config(cfg)
        except Exception:
            pass

        # Tell GL widget whether tracking is active (render selection/hit-test)
        try:
            self.gl_widget.set_desktop_tracking_active(bool(cfg.tracking_enabled))
        except Exception:
            pass
        self._update_debug_model_info()

    def open_tracked_object_settings(self):
        """
        Open rename/color/hide settings for currently selected tracked object.
        Persists per-camera overrides to data/desktop_object_overrides.json.
        """
        tid = getattr(self.gl_widget, "selected_track_id", None)
        if tid is None:
            return
        try:
            tid = int(tid)
        except Exception:
            return

        from PySide6.QtWidgets import QDialog

        current = dict((self._object_overrides or {}).get(int(tid), {}) or {})
        dlg = TrackedObjectSettingsDialog(self.camera_id, int(tid), current, self)
        if dlg.exec() == QDialog.DialogCode.Accepted:
            new_ov = dlg.result_override
            if isinstance(new_ov, dict):
                self._save_object_override(int(tid), new_ov)
                # Force redraw
                try:
                    self.gl_widget.update()
                except Exception:
                    pass

    def clear_tracked_object_overrides(self):
        """Clear all saved tracked-object overrides for this camera."""
        from PySide6.QtWidgets import QMessageBox
        try:
            confirm = QMessageBox.question(
                self,
                "Clear saved tracked objects",
                "Clear all saved tracked-object names/colors/hide settings for this camera?\n\n"
                "This is useful if track IDs were reused and names got assigned to the wrong object.",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if confirm != QMessageBox.StandardButton.Yes:
                return
        except Exception:
            pass
        try:
            from desktop.utils.desktop_object_overrides import clear_camera_overrides

            clear_camera_overrides(self.camera_id)
        except Exception:
            pass
        self._load_object_overrides()
        try:
            self.gl_widget.set_object_overrides(self._object_overrides)
            self.gl_widget.update()
        except Exception:
            pass

    def open_desktop_detection_settings(self):
        """
        Lightweight Desktop-local detector settings (does NOT touch backend config).
        """
        from PySide6.QtWidgets import (
            QDialog,
            QVBoxLayout,
            QFormLayout,
            QComboBox,
            QSlider,
            QLabel,
            QDialogButtonBox,
            QSpinBox,
            QLineEdit,
            QListWidget,
            QListWidgetItem,
            QCheckBox,
            QHBoxLayout,
            QPushButton,
            QMessageBox,
        )
        from PySide6.QtCore import Qt
        import threading

        dlg = QDialog(self)
        dlg.setWindowTitle(f"Desktop Detection Settings - {self.camera_id}")
        dlg.setWindowFlags(Qt.WindowType.Tool | Qt.WindowType.WindowStaysOnTopHint)
        root = QVBoxLayout(dlg)
        form = QFormLayout()
        root.addLayout(form)

        cfg = self.desktop_detector_config
        enabled_state = bool(self.desktop_object_detection_enabled)

        backend = QComboBox()
        backend.addItem("auto (onnxruntime → mobilenetssd)", "auto")
        backend.addItem("onnxruntime (YOLO ONNX only)", "onnxruntime")
        backend.addItem("mobilenetssd (default, auto CPU/GPU)", "mobilenetssd")
        backend.addItem("ultralytics plugin (optional)", "ultralytics")
        bidx = backend.findData(str(getattr(cfg, "backend", "auto") or "auto"))
        if bidx >= 0:
            backend.setCurrentIndex(bidx)
        form.addRow("Backend", backend)

        model = QComboBox()
        model.addItem("default (first installed or first local .onnx)", "default")
        try:
            from core.model_library.byo_models import list_installed_manifests, list_local_onnx_models

            for m in list_installed_manifests():
                model.addItem(str(m.display_name), str(m.slug))
            for p in list_local_onnx_models(include_byo=False):
                model.addItem(f"Local ONNX: {p.name}", str(p))
        except Exception:
            pass
        idx = model.findData(str(getattr(cfg, "model_variant", "default")))
        if idx >= 0:
            model.setCurrentIndex(idx)
        form.addRow("Model", model)

        device = QComboBox()
        device.addItem("auto", "auto")
        device.addItem("cuda", "cuda")
        device.addItem("cpu", "cpu")
        didx = device.findData(str(getattr(cfg, "device", "auto")))
        if didx >= 0:
            device.setCurrentIndex(didx)
        form.addRow("Device", device)

        fps = QSpinBox()
        fps.setRange(1, 30)
        fps.setValue(int(getattr(cfg, "fps_limit", 8) or 8))
        form.addRow("Max FPS", fps)

        imgsz = QSpinBox()
        imgsz.setRange(256, 1280)
        imgsz.setSingleStep(32)
        imgsz.setValue(int(getattr(cfg, "imgsz", 640) or 640))
        form.addRow("Image size", imgsz)

        conf = QSlider(Qt.Orientation.Horizontal)
        conf.setRange(0, 100)
        conf_val = float(getattr(cfg, "min_confidence", 0.35) or 0.35)
        conf.setValue(int(max(0, min(100, round(conf_val * 100)))))
        conf_label = QLabel(f"{conf.value()/100.0:.2f}")
        conf.valueChanged.connect(lambda v: conf_label.setText(f"{v/100.0:.2f}"))
        conf_row = QVBoxLayout()
        conf_row.addWidget(conf)
        conf_row.addWidget(conf_label)
        form.addRow("Min confidence", conf_row)

        # Allowed classes: show all model classes with a checklist (optional filter)
        filter_enabled = QCheckBox("Enable class filter (unchecked = allow all classes)")
        filter_enabled.setChecked(bool(getattr(cfg, "allowed_classes", None)))
        form.addRow("", filter_enabled)

        search_row = QHBoxLayout()
        class_search = QLineEdit()
        class_search.setPlaceholderText("Search classes…")
        btn_all = QPushButton("All")
        btn_none = QPushButton("None")
        search_row.addWidget(class_search)
        search_row.addWidget(btn_all)
        search_row.addWidget(btn_none)
        root.addLayout(search_row)

        classes_list = QListWidget()
        classes_list.setMinimumHeight(220)
        root.addWidget(classes_list)

        classes_status = QLabel("")
        classes_status.setStyleSheet("color: #c9d6e2;")
        root.addWidget(classes_status)

        allowed_set = set([str(x).strip() for x in (getattr(cfg, "allowed_classes", None) or []) if str(x).strip()])

        def _set_all(checked: bool):
            for i in range(classes_list.count()):
                it = classes_list.item(i)
                it.setCheckState(Qt.CheckState.Checked if checked else Qt.CheckState.Unchecked)

        btn_all.clicked.connect(lambda: _set_all(True))
        btn_none.clicked.connect(lambda: _set_all(False))

        def _apply_filter():
            q = str(class_search.text() or "").strip().lower()
            for i in range(classes_list.count()):
                it = classes_list.item(i)
                name = str(it.text() or "").lower()
                it.setHidden(bool(q) and (q not in name))

        class_search.textChanged.connect(lambda _t: _apply_filter())

        def _set_enabled_state(on: bool):
            classes_list.setEnabled(bool(on))
            class_search.setEnabled(bool(on))
            btn_all.setEnabled(bool(on))
            btn_none.setEnabled(bool(on))

        filter_enabled.toggled.connect(_set_enabled_state)
        _set_enabled_state(filter_enabled.isChecked())

        def _populate_class_list(names: list[str]):
            classes_list.clear()
            for n in names:
                it = QListWidgetItem(str(n))
                it.setFlags(it.flags() | Qt.ItemFlag.ItemIsUserCheckable)
                if (not allowed_set) or (str(n) in allowed_set):
                    it.setCheckState(Qt.CheckState.Checked)
                else:
                    it.setCheckState(Qt.CheckState.Unchecked)
                classes_list.addItem(it)
            classes_status.setText(f"Loaded {len(names)} classes from model.")
            _apply_filter()

        # Populate immediately from cached worker stats (best UX; no model load needed)
        cached = list(getattr(self, "_desktop_detector_class_names", []) or [])
        if cached:
            _populate_class_list(cached)
        else:
            classes_status.setText("No class list yet. Turn on Desktop detections to populate, or install a model with labels.json.")

        def _refresh_classes_from_manifest():
            key = str(model.currentData() or "default")
            if key in ("", "default"):
                classes_status.setText("Classes: enable detections to populate, or select an installed model with labels.")
                return
            try:
                from core.model_library.byo_models import load_manifest, read_labels_file

                man = load_manifest(key)
                if man is None:
                    classes_status.setText("Classes: unknown model (install/manage via tray).")
                    classes_list.clear()
                    return
                lp = man.labels_abs_path()
                names = read_labels_file(lp) if lp else None
                if not names:
                    classes_status.setText("Classes: labels file not found. Provide labels.json/labels.txt in the model folder.")
                    classes_list.clear()
                    return
                _populate_class_list(list(names))
            except Exception as e:
                classes_status.setText(f"Classes: failed to load labels ({e})")
                classes_list.clear()

        model.currentIndexChanged.connect(lambda _i: _refresh_classes_from_manifest())
        _refresh_classes_from_manifest()

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Save | QDialogButtonBox.StandardButton.Cancel)
        root.addWidget(buttons)
        buttons.accepted.connect(dlg.accept)
        buttons.rejected.connect(dlg.reject)

        if dlg.exec() != QDialog.DialogCode.Accepted:
            return

        # Apply settings (preserve enabled state)
        new_allowed: list[str] = []
        if bool(filter_enabled.isChecked()):
            # Collect checked items; enforce at least 1 if filter is enabled.
            for i in range(classes_list.count()):
                it = classes_list.item(i)
                if it.checkState() == Qt.CheckState.Checked:
                    name = str(it.text() or "").strip()
                    if name:
                        new_allowed.append(name)
            if not new_allowed:
                QMessageBox.warning(dlg, "Allowed classes", "Select at least one class, or disable the class filter to allow all.")
                return
            # If user checked everything, treat as no filter (None) for performance/clarity.
            if classes_list.count() > 0 and len(new_allowed) == classes_list.count():
                new_allowed = []

        # Preserve tracking settings while updating detection model parameters.
        old_cfg = cfg
        self.desktop_detector_config = DetectorConfig(
            enabled=enabled_state,
            fps_limit=int(fps.value()),
            backend=str(backend.currentData() or "mobilenetssd"),
            model_variant=str(model.currentData() or "default"),
            device=str(device.currentData() or "auto"),
            imgsz=int(imgsz.value()),
            min_confidence=float(conf.value() / 100.0),
            max_det=int(getattr(cfg, "max_det", 100) or 100),
            allowed_classes=(new_allowed or None),
        )
        try:
            self.desktop_detector_config.tracking_enabled = bool(getattr(old_cfg, "tracking_enabled", False))
            self.desktop_detector_config.tracker_type = str(getattr(old_cfg, "tracker_type", "sort") or "sort")
            self.desktop_detector_config.tracker_params = dict(getattr(old_cfg, "tracker_params", None) or {})
            self.desktop_detector_config.emit_detections = bool(getattr(old_cfg, "emit_detections", True))
            self.desktop_detector_config.tracker_reset_token = int(getattr(old_cfg, "tracker_reset_token", 0) or 0)
        except Exception:
            pass

        if self._desktop_detector_worker is not None:
            try:
                self._desktop_detector_worker.update_config(self.desktop_detector_config)
            except Exception:
                pass

        # Best-effort UX status
        self._desktop_detector_status_text = "Detections: updated settings"
        self._auto_show_debug_overlay("desktop_detector")
        self._update_debug_model_info()

    def open_object_detection_settings(self):
        if not hasattr(self, 'obj_det_dialog') or self.obj_det_dialog is None:
            self.obj_det_dialog = ObjectDetectionSettingsDialog(self.camera_id, self)
            self.obj_det_dialog.finished.connect(lambda: setattr(self, 'obj_det_dialog', None))
            # Refresh status when dialog closes in case it changed there
            self.obj_det_dialog.finished.connect(self._check_object_detection_status)
            
        self.obj_det_dialog.show()
        self.obj_det_dialog.raise_()
        self.obj_det_dialog.activateWindow()

    def open_camera_config(self):
        if not hasattr(self, 'camera_config_dialog') or self.camera_config_dialog is None:
            self.camera_config_dialog = CameraConfigDialog(self, camera_manager=self.camera_manager)
            self.camera_config_dialog.finished.connect(lambda: setattr(self, 'camera_config_dialog', None))
        self.camera_config_dialog.show()
        self.camera_config_dialog.raise_()
        self.camera_config_dialog.activateWindow()

    def toggle_debug(self):
        self.debug_overlay_enabled = not self.debug_overlay_enabled
        # Manual toggle overrides auto-hide behavior.
        self._debug_overlay_user_forced = bool(self.debug_overlay_enabled)
        # If user turned it off manually, opt out of future auto-show until they toggle depth/detection.
        if not bool(self.debug_overlay_enabled):
            self._debug_overlay_user_opt_out = True
        else:
            self._debug_overlay_user_opt_out = False
        self._clear_auto_debug()
        self._apply_overlay_settings()

    def toggle_motion(self):
        self.motion_boxes_enabled = not self.motion_boxes_enabled
        self.gl_widget.set_overlay_settings(self.debug_overlay_enabled, self.motion_boxes_enabled)
        # Briefly surface a confirmation in the debug overlay so the
        # user has visible proof of the toggle action.  Takes direct
        # ownership of the overlay (bypasses user_opt_out) for the
        # 1.5s confirmation window, then restores the prior state.
        try:
            state = "ON" if self.motion_boxes_enabled else "OFF"
            self._debug_chord_lines = [f"Motion boxes: {state}"]
            self._chord_take_overlay()
            self._chord_push_lines(self._debug_chord_lines)
            # Only schedule the auto-clear when no chord modifier is
            # currently held -- otherwise the clear would fire mid-
            # chord and make the live value display vanish. When the
            # chord ends, _chord_finalize_release schedules its own
            # clear.
            if self._chord_modifier_key is None:
                self._chord_schedule_clear(1500, self._clear_toggle_overlay)
        except Exception:
            pass

    def _clear_toggle_overlay(self):
        """Drop the brief toggle-confirmation hint (used by M / D taps)
        and restore the prior debug-overlay state."""
        try:
            self._debug_chord_lines = []
            self._chord_release_overlay()
        except Exception:
            pass

    def open_motion_settings(self):
        if not hasattr(self, 'motion_settings_dialog') or self.motion_settings_dialog is None:
            self.motion_settings_dialog = MotionSettingsDialog(self.gl_widget.motion_settings, self)
            
            # Apply updates live
            def apply_settings(settings):
                self.gl_widget.motion_settings = settings
                self.gl_widget.update()
                
            self.motion_settings_dialog.settings_changed.connect(apply_settings)
            
            # Handle cleanup on close
            def on_dialog_close():
                self.motion_settings_dialog = None
                
            self.motion_settings_dialog.finished.connect(on_dialog_close)
            
        self.motion_settings_dialog.show()
        self.motion_settings_dialog.raise_()
        self.motion_settings_dialog.activateWindow()

    def open_detection_overlay_settings(self):
        if not hasattr(self, 'detection_settings_dialog') or self.detection_settings_dialog is None:
            self.detection_settings_dialog = DetectionOverlaySettingsDialog(self.gl_widget.detection_settings, self)

            def apply_settings(settings):
                try:
                    self.gl_widget.detection_settings = settings
                    # Re-apply ROI filter immediately when ROI settings change.
                    try:
                        self.gl_widget.refresh_detection_roi()
                    except Exception:
                        pass
                    self.gl_widget.update()
                except Exception:
                    pass

            self.detection_settings_dialog.settings_changed.connect(apply_settings)

            def on_dialog_close():
                self.detection_settings_dialog = None

            self.detection_settings_dialog.finished.connect(on_dialog_close)

        self.detection_settings_dialog.show()
        self.detection_settings_dialog.raise_()
        self.detection_settings_dialog.activateWindow()

    def open_motion_watch_dialog(self):
        dlg = MotionWatchDialog(self.motion_watch_settings, self)
        if dlg.exec() == QDialog.DialogCode.Accepted:
            self.motion_watch_settings.update(dlg.get_settings())
            self._persist_motion_watch_settings()
            self._sync_storage_settings()
            self.start_motion_watch()

    def _overlay_key(self, ids: List[str]) -> str:
        clean = [str(x) for x in (ids or []) if x]
        if not clean:
            return "__all__"
        if len(clean) == 1:
            return clean[0]
        return "group:" + "|".join(sorted(clean))

    def open_overlay_window(self, target_ids: Optional[List[str]] = None):
        """Launch or focus a transparent overlay projection window for this camera.

        Unlike the previous implementation, this supports *multiple* simultaneous overlay windows
        (e.g., projecting multiple zones separately from the same camera widget).
        """
        ids = [str(x) for x in (target_ids or []) if x]
        key = self._overlay_key(ids)

        win = (self.overlay_windows or {}).get(key)
        if win is not None:
            try:
                win.set_target_ids(ids)
            except Exception:
                pass
            try:
                win.show()
                win.raise_()
                win.activateWindow()
                return
            except Exception:
                # Fall through to recreate if the window is in a bad state.
                try:
                    self.overlay_windows.pop(key, None)
                except Exception:
                    pass

        win = CameraOverlayWindow(self, target_ids=ids)
        self.overlay_windows[key] = win
        win.destroyed.connect(lambda _obj=None, k=key: self.overlay_windows.pop(k, None))

        # Slightly cascade new overlays so multiple projections are visible immediately.
        try:
            base = self.pos()
            offset = 28 * max(0, len(self.overlay_windows) - 1)
            win.move(base.x() + 40 + offset, base.y() + 40 + offset)
        except Exception:
            pass

        win.show()
        try:
            win.raise_()
            win.activateWindow()
        except Exception:
            pass

    # ==================== PTZ Overlay + Commands ====================

    def _persist_ptz_overlay_settings(self):
        try:
            save_ptz_overlay_settings(self.camera_id, self.ptz_overlay_settings)
        except Exception:
            pass

    def open_ptz_overlay_settings(self):
        """Open PTZ overlay settings dialog (look & feel + connection + auto-pan)."""
        label = self.camera_name or self.camera_id
        dlg = PTZOverlaySettingsDialog(label, self.ptz_overlay_settings, self)
        if dlg.exec() == QDialog.DialogCode.Accepted:
            self.ptz_overlay_settings = dlg.result_settings
            self._persist_ptz_overlay_settings()
            if self.ptz_overlay:
                self.ptz_overlay.settings = self.ptz_overlay_settings
                self.ptz_overlay.apply_style()
                self.ptz_overlay.reposition_hud()
            if self.ptz_controller_window and getattr(self.ptz_controller_window, "panel", None):
                try:
                    self.ptz_controller_window.panel.settings = self.ptz_overlay_settings
                    self.ptz_controller_window.panel.apply_style()
                    self.ptz_controller_window.panel.reposition_hud()
                except Exception:
                    pass

    # ------------------------------------------------------------------ #
    # Floating PTZ controller flow
    #
    # The single entry point for PTZ. Probes the backend to learn which
    # protocol the camera speaks. If the backend asks for an extra
    # credential (today: TP-Link cloud password), shows a tiny prompt,
    # stores it via /api/cameras/<id>/ptz/credentials, and re-probes.
    # On success, brings the floating PTZControllerWindow forward.
    # ------------------------------------------------------------------ #

    def open_ptz_controller(self):
        """Open the floating PTZ controller (instant feedback) and probe in the background."""
        cam_id = str(self.camera_id)
        print(f"[PTZ] open_ptz_controller invoked for {cam_id}")

        # 1) Show the floating window IMMEDIATELY on the GUI thread.
        #    Probe and any credential prompt happen behind it.
        try:
            self._show_ptz_controller_window()
        except Exception as exc:
            print(f"[PTZ] failed to show floating controller for {cam_id}: {exc}")
            try:
                QMessageBox.warning(self, "PTZ", f"Could not open PTZ controller: {exc}")
            except Exception:
                pass
            return

        # 2) Kick the probe in the background; if creds are missing, prompt.
        self._ptz_probe_async()

    def _show_ptz_controller_window(self) -> None:
        """Create or focus the floating PTZ window. Must run on the GUI thread."""
        if self.ptz_controller_window is not None:
            try:
                self.ptz_controller_window.show()
                self.ptz_controller_window.raise_()
                self.ptz_controller_window.activateWindow()
            except Exception:
                pass
            return

        self.ptz_controller_window = PTZControllerWindow(self, self.ptz_overlay_settings)
        self.ptz_controller_window.destroyed.connect(
            lambda: setattr(self, "ptz_controller_window", None)
        )
        self.ptz_controller_window.show()
        self.ptz_controller_window.raise_()
        self.ptz_controller_window.activateWindow()

    def _ptz_probe_async(self) -> None:
        """Probe the backend; if it needs a Tapo cloud password, prompt for one."""
        url = f"{ObjectDetectionSettingsDialog.API_BASE}/cameras/{self.camera_id}/ptz/probe"
        body = self._build_ptz_command_body("__probe__", {})
        body.pop("action", None)
        cam_id = str(self.camera_id)

        def _handle_payload(payload: dict) -> None:
            # Runs on the GUI thread.
            try:
                if not bool(getattr(self, "running", True)):
                    return
                data = (payload or {}).get("data") or {}
                needs = data.get("needs_credentials") or []
                proto = data.get("protocol_resolved")
                print(f"[PTZ] probe result for {cam_id}: protocol={proto} needs={needs}")
                if "tapo_cloud_password" in needs:
                    self._prompt_tapo_cloud_password(
                        message=(payload or {}).get("data", {}).get("error"),
                        is_locked=bool((payload or {}).get("data", {}).get("is_locked")),
                    )
            except Exception as exc:
                print(f"[PTZ] probe handler error for {cam_id}: {exc}")

        def worker():
            payload: dict = {}
            try:
                # 25s: a fresh Tapo login (KLAP handshake + getBasicInfo)
                # can take 8-15s on first contact. Anything shorter
                # truncates a real login and burns an attempt.
                res = requests.post(url, json=body, timeout=25)
                try:
                    payload = res.json() or {}
                except Exception:
                    payload = {}
                print(f"[PTZ] probe HTTP {res.status_code} for {cam_id}")
            except Exception as exc:
                print(f"[PTZ] probe HTTP failed for {cam_id}: {exc}")
            try:
                QTimer.singleShot(0, lambda p=payload: _handle_payload(p))
            except Exception:
                _handle_payload(payload)

        threading.Thread(target=worker, daemon=True).start()

    def _prompt_tapo_cloud_password(self, message: Optional[str] = None,
                                     is_locked: bool = False) -> None:
        """Backwards-compatible alias: open the full credentials dialog."""
        self.open_ptz_credentials_dialog(message=message, is_locked=is_locked)

    def open_ptz_credentials_dialog(self, message: Optional[str] = None,
                                     is_locked: bool = False) -> None:
        """
        Open the PTZ credentials & protocol dialog for this camera.

        Always available from the floating PTZ window's gear button or the
        camera's right-click PTZ submenu. Auto-popped on first failure.
        """
        if getattr(self, "_ptz_cred_prompt_open", False):
            return
        self._ptz_cred_prompt_open = True

        label = self.camera_name or self.camera_id
        default_user = "admin"
        default_password = ""
        try:
            cfg = self.camera_manager.cameras.get(self.camera_id) if self.camera_manager else None
            if cfg:
                if getattr(cfg, "username", None):
                    default_user = str(cfg.username)
                if getattr(cfg, "password", None):
                    default_password = str(cfg.password)
        except Exception:
            pass

        # Pre-fill from any previously-stored credentials.
        existing = self._fetch_ptz_credentials_blocking()

        try:
            dlg = PTZCredentialsDialog(
                label,
                self,
                default_username=default_user,
                default_password=default_password,
                existing=existing,
                message=message,
                is_locked=bool(is_locked),
                test_callback=self._test_ptz_credentials_blocking,
            )
            if dlg.exec() != QDialog.DialogCode.Accepted:
                return
            if dlg.cleared:
                self._clear_ptz_credentials()
                return
            self._save_ptz_credentials(dlg.result_payload, bool(dlg.persist))
        except Exception as exc:
            print(f"[PTZ] credentials dialog failed: {exc}")
        finally:
            self._ptz_cred_prompt_open = False

    def _test_ptz_credentials_blocking(self, payload: Dict[str, str]):
        """
        Synchronously POST the candidate credentials to /ptz/test on the
        backend. Returns (ok: bool, message: str) for the dialog.
        """
        url = f"{ObjectDetectionSettingsDialog.API_BASE}/cameras/{self.camera_id}/ptz/test"
        try:
            res = requests.post(url, json=payload or {}, timeout=20)
            data = {}
            try:
                data = res.json() or {}
            except Exception:
                data = {}
            ok = bool(res.ok and data.get("success"))
            message = (data.get("message") or "").strip() or ("OK" if ok else "Test failed")
            return ok, message
        except Exception as exc:
            return False, f"Network error: {exc}"

    def _fetch_ptz_credentials_blocking(self) -> Dict[str, str]:
        url = f"{ObjectDetectionSettingsDialog.API_BASE}/cameras/{self.camera_id}/ptz/credentials"
        try:
            res = requests.get(url, timeout=3)
            if not res.ok:
                return {}
            data = (res.json() or {}).get("data") or {}
            return data if isinstance(data, dict) else {}
        except Exception as exc:
            print(f"[PTZ] credential fetch failed: {exc}")
            return {}

    def _save_ptz_credentials(self, payload: Dict[str, str], persist: bool) -> None:
        """Send each credential key to the backend, then re-probe."""
        if not payload:
            return
        base_url = f"{ObjectDetectionSettingsDialog.API_BASE}/cameras/{self.camera_id}/ptz/credentials"
        cam_id = str(self.camera_id)

        def worker():
            ok_any = False
            for key, value in payload.items():
                body = {"key": key, "value": value, "persist": bool(persist)}
                try:
                    res = requests.post(base_url, json=body, timeout=10)
                    ok = bool(res.ok and (res.json() or {}).get("success"))
                    print(f"[PTZ] credential save '{key}' HTTP {res.status_code} ok={ok} for {cam_id}")
                    ok_any = ok_any or ok
                except Exception as exc:
                    print(f"[PTZ] credential save '{key}' failed for {cam_id}: {exc}")
            if ok_any:
                # Reset throttle so a subsequent failure can re-prompt if needed.
                # We deliberately do NOT trigger an auto-probe here: the next
                # actual movement command will perform the one-and-only login
                # with the new credentials and stash the controller in the
                # manager. Triggering a probe now would burn an extra login
                # attempt against Tapo's 5/hour lockout window.
                self._ptz_last_cred_prompt_ts = 0.0
                self._ptz_first_failure_seen = False

        threading.Thread(target=worker, daemon=True).start()

    def _clear_ptz_credentials(self) -> None:
        url = f"{ObjectDetectionSettingsDialog.API_BASE}/cameras/{self.camera_id}/ptz/credentials"
        cam_id = str(self.camera_id)

        def worker():
            try:
                res = requests.delete(url, timeout=6)
                ok = bool(res.ok and (res.json() or {}).get("success"))
                print(f"[PTZ] credential clear HTTP {res.status_code} ok={ok} for {cam_id}")
            except Exception as exc:
                print(f"[PTZ] credential clear failed for {cam_id}: {exc}")
            self._ptz_last_cred_prompt_ts = 0.0
            self._ptz_first_failure_seen = False

        threading.Thread(target=worker, daemon=True).start()

    # Back-compat shims so existing IPC and AI tools continue to work even
    # though the docked overlay path is retired.

    def toggle_ptz_overlay(self, checked: Optional[bool] = None):
        """Back-compat: docked overlay is retired; open the floating controller instead."""
        want_show = bool(checked) if checked is not None else (self.ptz_controller_window is None)
        if want_show:
            self.open_ptz_controller()
        elif self.ptz_controller_window is not None:
            try:
                self.ptz_controller_window.close()
            except Exception:
                pass

    def toggle_ptz_undocked(self, checked: Optional[bool] = None):
        """Back-compat alias for the floating controller flow."""
        want_undocked = bool(checked) if checked is not None else (self.ptz_controller_window is None)
        if want_undocked:
            self.open_ptz_controller()
        elif self.ptz_controller_window is not None:
            try:
                self.ptz_controller_window.close()
            except Exception:
                pass
            self.ptz_controller_window = None

    def _build_ptz_command_body(self, action: str, payload: Dict[str, object]) -> Dict[str, object]:
        """
        Build the PTZ request body. The backend hydrates IP / username /
        password / Tapo cloud password from the camera record and the
        `ptz_credentials` store, so we only ship the action, params, and
        an optional brand override.
        """
        body: Dict[str, object] = {"action": action, "params": dict(payload or {})}

        conn = (self.ptz_overlay_settings or {}).get("connection") or {}
        brand = str(conn.get("brand_override") or "auto").strip().lower()
        if brand and brand != "auto":
            body["brand_hint"] = brand

        return body

    def ptz_send(self, action: str, payload: Dict[str, object], silent: bool = False):
        """Non-blocking PTZ command sender (uses backend /api/cameras/<id>/ptz)."""
        # If the widget is closing/closed, ignore background PTZ work.
        if not bool(getattr(self, "running", True)):
            return

        url = f"{ObjectDetectionSettingsDialog.API_BASE}/cameras/{self.camera_id}/ptz"
        body = self._build_ptz_command_body(action, payload)
        cam_id = str(self.camera_id)

        # Do NOT capture `self` in the thread; use a weakref so deletion is safe.
        import weakref
        self_ref = weakref.ref(self)

        def _safe_emit(payload_dict: dict) -> None:
            w = self_ref()
            if w is None:
                return
            # If widget has been closed/deleted, don't emit (prevents RuntimeError spam).
            try:
                if not bool(getattr(w, "running", True)):
                    return
            except Exception:
                return
            try:
                w.ptz_result_signal.emit(payload_dict)
            except RuntimeError:
                # wrapped C/C++ object deleted
                return
            except Exception:
                return

        def worker():
            try:
                # 12s timeout so the first PyTapo KLAP handshake (which can
                # take 5-8s end-to-end) doesn't get truncated and look like
                # an auth failure.
                res = requests.post(url, json=body, timeout=12)
                data = {}
                try:
                    data = res.json()
                except Exception:
                    data = {}
                ok = bool(data.get("success")) and res.ok
                result = {
                    "action": action,
                    "success": ok,
                    "status_code": int(res.status_code),
                    "data": data.get("data") if isinstance(data, dict) else None,
                    "message": data.get("message") if isinstance(data, dict) else None,
                }
                _safe_emit(result)
                if not silent and not ok:
                    msg = result.get("message") or "PTZ command failed"
                    w = self_ref()
                    if w is not None:
                        try:
                            w._ptz_log_failure(action, msg)
                        except Exception:
                            pass
            except Exception as e:
                _safe_emit({"action": action, "success": False, "status_code": 0, "data": None, "message": str(e)})
                if not silent:
                    w = self_ref()
                    if w is not None:
                        try:
                            w._ptz_log_failure(action, f"transport error: {e}")
                        except Exception:
                            pass

        threading.Thread(target=worker, daemon=True).start()

    def _ptz_log_failure(self, action: str, message: str) -> None:
        """Throttle the per-key-release PTZ failure spam to one line per ~5s per message."""
        import time as _time
        cache = getattr(self, "_ptz_log_cache", None)
        if cache is None:
            cache = {}
            self._ptz_log_cache = cache
        now = _time.monotonic()
        last = cache.get(message, 0.0)
        if now - last < 5.0:
            return
        cache[message] = now
        # Trim cache occasionally
        if len(cache) > 32:
            for key in list(cache.keys())[:16]:
                cache.pop(key, None)
        try:
            print(f"PTZ error for {self.camera_id} ({action}): {message}")
        except Exception:
            pass

    def _on_ptz_result(self, result: dict):
        # Keep overlay in sync (auto-pan state, etc.)
        try:
            data = result.get("data") or {}
            if isinstance(data, dict) and "sweep_active" in data:
                self._ptz_sweep_active = bool(data.get("sweep_active"))
        except Exception:
            pass
        if self.ptz_overlay:
            try:
                self.ptz_overlay.update_from_ptz_result(result)
            except Exception:
                pass
        if self.ptz_controller_window and getattr(self.ptz_controller_window, "panel", None):
            try:
                self.ptz_controller_window.panel.update_from_ptz_result(result)
            except Exception:
                pass

        # Auto-prompt for credentials when the backend tells us we need them.
        try:
            self._maybe_prompt_for_ptz_credentials(result)
        except Exception as exc:
            print(f"[PTZ] auto-prompt check failed: {exc}")

    def _maybe_prompt_for_ptz_credentials(self, result: dict) -> None:
        if bool(result.get("success")):
            self._ptz_first_failure_seen = False
            return

        data = result.get("data") if isinstance(result, dict) else None
        msg = (result.get("message") or "").lower()
        needs: list = []
        is_locked = False
        if isinstance(data, dict):
            raw_needs = data.get("needs_credentials") or []
            if isinstance(raw_needs, list):
                needs = raw_needs
            is_locked = bool(data.get("is_locked"))

        wants_prompt = (
            "tapo_cloud_password" in needs
            or is_locked
            or "missing credentials" in msg
            or "invalid camera credentials" in msg
            or "tp-link cloud password" in msg
            or "tapo ptz login failed" in msg
            or "tapo ptz needs credentials" in msg
            or "temporary suspension" in msg
            or "locked out" in msg
            or ("auth" in msg and "fail" in msg)
        )
        if not wants_prompt:
            return

        import time as _time
        first = not bool(getattr(self, "_ptz_first_failure_seen", False))
        last = float(getattr(self, "_ptz_last_cred_prompt_ts", 0.0) or 0.0)
        now = _time.monotonic()

        # Pop the dialog *immediately* on the first failure of a session; on
        # subsequent failures throttle to once per 30s to avoid spam.
        if not first and (now - last) < 30.0:
            return
        self._ptz_first_failure_seen = True
        self._ptz_last_cred_prompt_ts = now

        nice_message = result.get("message") or "PTZ credentials are required for this camera."
        cam_locked = is_locked

        def _go():
            self.open_ptz_credentials_dialog(message=str(nice_message), is_locked=cam_locked)

        try:
            QTimer.singleShot(0, _go)
        except Exception:
            _go()

    def edit_shape(self, shape_id: str):
        target = next((s for s in self.gl_widget.shapes if s['id'] == shape_id), None)
        if not target:
            return
            
        dialog = ShapeSettingsDialog(target, self)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            new_shape = dialog.get_shape()
            # Update list
            self.gl_widget.shapes = [new_shape if s['id'] == shape_id else s for s in self.gl_widget.shapes]
            self.gl_widget._emit_shapes()
            self.gl_widget.update()

    def toggle_aspect_ratio(self, checked):
        self.aspect_ratio_locked = checked
        self.keep_aspect_ratio = checked
        mode = Qt.AspectRatioMode.KeepAspectRatioByExpanding if checked else Qt.AspectRatioMode.IgnoreAspectRatio
        self.gl_widget.set_aspect_ratio_mode(mode)

    def toggle_stream_quality(self, checked):
        """User-initiated quality toggle from the context menu.

        Treats the user's choice as a pin: auto-switching is suspended
        for this widget until the user explicitly re-enables auto via
        the 'Auto Quality' menu item.
        """
        new_quality = 'medium' if checked else 'low'
        self._apply_stream_quality(new_quality, pinned=True)

    def set_auto_quality_enabled(self, enabled: bool):
        """Re-enable (or disable) the geometry-driven quality policy."""
        self._auto_quality_enabled = bool(enabled)
        if enabled:
            self._quality_pinned = False
            self._schedule_auto_quality_eval(delay_ms=100)

    def _switch_quality_async(self, quality):
        try:
            self.camera_manager.send_command(self.camera_id, "set_quality", {"quality": quality})
        except Exception as e:
            print(f"Failed to switch quality: {e}")

    def _retest_substream(self):
        """Re-run the substream capability probe in the background.

        Useful after fixing a substream URL or when a camera that
        previously failed becomes available again.  Refreshes the
        capability flag/resolution and re-evaluates auto quality.
        """
        if not self.camera_manager:
            return
        cam_id = self.camera_id
        cfg = None
        try:
            cfg = self.camera_manager.cameras.get(cam_id)
        except Exception:
            cfg = None
        if cfg is None:
            return

        import threading

        def _worker():
            try:
                import asyncio
                loop = getattr(self.camera_manager, "_loop", None)
                if loop and loop.is_running():
                    fut = asyncio.run_coroutine_threadsafe(
                        self.camera_manager._probe_substream(cfg, force=True), loop,
                    )
                    fut.result(timeout=15)
                else:
                    asyncio.run(
                        self.camera_manager._probe_substream(cfg, force=True)
                    )
                if cfg.substream_capable:
                    if loop and loop.is_running():
                        asyncio.run_coroutine_threadsafe(
                            self.camera_manager._ensure_mediamtx_sub_path(cfg), loop,
                        )
                self._schedule_auto_quality_eval(delay_ms=200)
            except Exception as exc:
                print(f"Substream retest failed for {cam_id}: {exc}")

        threading.Thread(target=_worker, daemon=True).start()

    def get_overlay_snapshot(self) -> Optional[dict]:
        """
        Lightweight pull API for overlay consumers (PyQt overlay window).
        Returns only inexpensive, already-available state; no extra network or allocation heavy work.
        """
        try:
            gl = self.gl_widget
            # Shallow copies to avoid mutating live state
            shapes = [dict(s) for s in (gl.shapes or [])]
            motion_boxes = list(gl.motion_boxes or [])
            tracked_objects = []
            for obj in (gl.tracked_objects or {}).values():
                tracked_objects.append({
                    'box': tuple(obj.get('box', (0, 0, 0, 0))),
                    'speed': float(obj.get('speed', 0)),
                    # Keep only a short history slice to keep payload light
                    'history': list(obj.get('history', [])[-10:])
                })
            snapshot = {
                'camera_id': self.camera_id,
                'frame_dims': tuple(gl.frame_dims or (0, 0)),
                'shapes': shapes,
                'selected_shapes': list(gl.selected_shapes or []),
                'show_shape_labels': bool(gl.show_shape_labels),
                'motion_boxes': motion_boxes,
                'tracked_objects': tracked_objects,
                'motion_hit_pulses': list(gl.motion_hit_pulses or []),
                'zone_pulses': dict(gl.zone_pulses or {}),
                'line_pulses': dict(gl.line_pulses or {}),
                'tag_pulses': dict(gl.tag_pulses or {}),
                'timestamp': time.time(),
            }
            return snapshot
        except Exception as e:
            print(f"overlay snapshot error for {self.camera_id}: {e}")
            return None

    def _on_shapes_changed(self, shapes: List[Shape]):
        """Persist latest shapes and keep OpenGL widget in sync."""
        self.shapes_by_camera[self.camera_id] = shapes
        # Shapes already applied inside widget; nothing else yet but reserved for persistence hooks.

    def _on_shape_triggered(self, payload: dict):
        """Receive motion-shape interactions from GL widget."""
        events = (payload or {}).get('events') or []
        src = (payload or {}).get("source") or None
        if self.motion_watch_active and events:
            allowed = {
                'zone': bool(self.motion_watch_settings.get("allow_zone", True)),
                'line': bool(self.motion_watch_settings.get("allow_line", True)),
                'tag': bool(self.motion_watch_settings.get("allow_tag", True)),
            }
            if any(allowed.get(ev.get("shape_type"), False) for ev in events):
                now = time.time()
                cooldown = float(self.motion_watch_settings.get("cooldown_sec", 3))
                if now - self.motion_watch_last_trigger >= cooldown:
                    self.motion_watch_last_trigger = now
                    delay_ms = int(self.motion_watch_settings.get("trigger_delay_ms", 0))
                    motion_box = None
                    try:
                        if src in ("desktop", "backend", "detection"):
                            if getattr(self.gl_widget, "detection_hit_pulses", None):
                                latest = self.gl_widget.detection_hit_pulses[-1]
                                motion_box = latest.get("box")
                        else:
                            if self.gl_widget.motion_hit_pulses:
                                latest = self.gl_widget.motion_hit_pulses[-1]
                                motion_box = latest.get("box")
                    except Exception:
                        motion_box = None

                    def do_capture():
                        self._capture_motion_watch_shot_async(
                            motion_box,
                            trigger_events=events,
                            trigger_source=src,
                            remaining_seconds=self._remaining_watch_seconds(),
                        )

                    if delay_ms > 0:
                        QTimer.singleShot(delay_ms, do_capture)
                    else:
                        do_capture()

                    # Trigger clip recording if enabled (independent of screenshot cooldown)
                    if self.motion_watch_settings.get("clip_enabled"):
                        try:
                            self._start_clip_recording(
                                trigger_events=events,
                                trigger_source=src,
                            )
                        except Exception:
                            pass
        try:
            print(f"[{self.camera_id}] shape interaction: {payload}")
        except Exception:
            pass

    def _toggle_shape_labels(self):
        self.gl_widget.show_shape_labels = not self.gl_widget.show_shape_labels
        self.gl_widget.update()

    def _hydrate_camera_metadata(self):
        """Fetch friendly camera name/IP for UI overlays and sync recording state."""
        if not self.camera_manager or not hasattr(self.camera_manager, "cameras"):
            return
        cfg = self.camera_manager.cameras.get(self.camera_id)
        if not cfg:
            return
        try:
            self.camera_name = getattr(cfg, "name", None) or self.camera_id
            self.camera_ip = getattr(cfg, "ip_address", None) or getattr(cfg, "ip", None)
            if not self.camera_ip and getattr(cfg, "rtsp_url", None):
                parsed = urlparse(cfg.rtsp_url)
                self.camera_ip = parsed.hostname
        except Exception:
            pass

        # Update window title with name if available
        if self.camera_name and self.camera_name != self.camera_id:
            self.setWindowTitle(f"Camera: {self.camera_name}")

        # Sync recording indicator from the backend (single source of truth)
        try:
            import requests as _req
            r = _req.get("http://localhost:5000/api/cameras/recording-status", timeout=2)
            data = r.json().get("data", {})
            self._continuous_recording = bool(data.get(self.camera_id, False))
            self.gl_widget._camera_recording_flag = self._continuous_recording
        except Exception:
            pass

    def _sync_recording_state(self):
        """Periodic sync of recording indicator from the backend."""
        def _fetch():
            try:
                import requests as _req
                r = _req.get("http://localhost:5000/api/cameras/recording-status", timeout=2)
                data = r.json().get("data", {})
                new_val = bool(data.get(self.camera_id, False))
                if new_val != self._continuous_recording:
                    self._continuous_recording = new_val
                    self.gl_widget._camera_recording_flag = new_val
                    self.gl_widget.update()
            except Exception:
                pass
        threading.Thread(target=_fetch, daemon=True).start()

    def _load_motion_watch_settings(self):
        """Best-effort load of persisted motion watch defaults."""
        try:
            if MOTION_WATCH_SETTINGS_PATH.exists():
                data = json.loads(MOTION_WATCH_SETTINGS_PATH.read_text())
                if isinstance(data, dict):
                    saved = data.get(self.camera_id) or data.get("default") or {}
                    if isinstance(saved, dict):
                        self.motion_watch_settings.update(saved)
        except Exception as e:
            print(f"Motion watch settings load error for {self.camera_id}: {e}")

    def _persist_motion_watch_settings(self):
        """Persist current motion watch config per camera."""
        try:
            MOTION_WATCH_SETTINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
            data = {}
            if MOTION_WATCH_SETTINGS_PATH.exists():
                try:
                    data = json.loads(MOTION_WATCH_SETTINGS_PATH.read_text())
                    if not isinstance(data, dict):
                        data = {}
                except Exception:
                    data = {}
            data[self.camera_id] = self.motion_watch_settings
            MOTION_WATCH_SETTINGS_PATH.write_text(json.dumps(data, indent=2))
        except Exception as e:
            print(f"Motion watch settings persist error for {self.camera_id}: {e}")

    def _playback_choose_folder(self):
        """Let the user pick a folder to scan for recordings, then open playback."""
        from PySide6.QtWidgets import QFileDialog
        start = getattr(self, '_playback_custom_dir', None) or str(Path.home())
        folder = QFileDialog.getExistingDirectory(
            self, "Select recordings folder", start,
        )
        if not folder:
            return
        self._playback_custom_dir = folder
        if self._playback_active or self._playback_loading:
            if self.playback_overlay:
                self.playback_overlay.stop_playback()
                self.playback_overlay.hide()
            self._playback_active = False
            self._playback_loading = False
        self.toggle_playback_overlay()

    def start_synced_playback(self, timestamp: float, speed: float = 1.0):
        """Open playback at a specific timestamp (called by the app sync system)."""
        if not self._playback_active and not self._playback_loading:
            self.toggle_playback_overlay()
        # Once loading finishes, the _on_playback_data_loaded handler will
        # start playback at the end of the last segment.  We override that
        # by storing the desired timestamp and seeking to it once ready.
        self._sync_start_ts = timestamp
        self._sync_start_speed = speed

    def toggle_playback_overlay(self):
        """Toggle the recording playback overlay on/off."""
        if self._playback_active or self._playback_loading:
            if self.playback_overlay:
                self.playback_overlay.stop_playback()
                self.playback_overlay.hide()
            self._playback_active = False
            self._playback_loading = False
            self._auto_unsync()
            self._reconnect_live_after_playback()
            return

        if self.playback_overlay is None:
            from desktop.widgets.playback_overlay import PlaybackOverlayWidget
            self.playback_overlay = PlaybackOverlayWidget(self.camera_id, parent=self.gl_widget)
            self.playback_overlay.camera_name = self.camera_name or ""
            self.playback_overlay.playback_closed.connect(self._on_playback_closed)
            self.playback_overlay.segments_loaded.connect(self._on_playback_segments_loaded)
            self.playback_overlay.seek_requested.connect(
                lambda ts: self._broadcast_sync('seek', ts=ts)
            )
            self.playback_overlay.speed_changed.connect(
                lambda spd: self._broadcast_sync('speed', speed=spd)
            )
            self.playback_overlay.play_toggled.connect(
                lambda playing: self._broadcast_sync('play' if playing else 'pause')
            )
            self.playback_overlay.go_live_requested.connect(
                lambda: self._broadcast_sync('go_live')
            )
            self.playback_overlay.step_requested.connect(
                lambda fwd: self._broadcast_sync('step', forward=fwd)
            )

        self.playback_overlay.setGeometry(self.gl_widget.rect())
        self.playback_overlay.show()
        self.playback_overlay.raise_()
        self.playback_overlay.setFocus()

        # Deferred geometry fix: the gl_widget may not have its final size
        # yet if this widget was just spawned (e.g. via sync).  Re-apply
        # geometry after the event loop has processed pending layout events.
        QTimer.singleShot(0, self._fix_playback_overlay_geometry)

        self._playback_loading = True

        self.loading_label.setText("Loading recordings...")
        self.loading_label.show()
        self.loading_timer.start(300)

        threading.Thread(target=self._load_playback_data, daemon=True).start()

    def _fix_playback_overlay_geometry(self):
        """Re-apply overlay geometry after the widget has been laid out."""
        if self.playback_overlay and self.playback_overlay.isVisible():
            self.playback_overlay.setGeometry(self.gl_widget.rect())
            self.playback_overlay.raise_()

    def _clear_no_video_warning(self):
        if self.playback_overlay:
            self.playback_overlay._no_video_at_time = False
            self.playback_overlay.update()

    def _on_playback_closed(self):
        self._playback_active = False
        self._playback_loading = False
        self._auto_unsync()
        self._reconnect_live_after_playback()

    def _on_playback_segments_loaded(self):
        """Hide loading indicator once segment scanning completes."""
        try:
            self._stop_loading()
        except Exception:
            pass

    def _load_playback_data(self):
        """Build segment timeline and fetch events in background.

        Runs on a worker thread.  Results are marshalled back to the GUI
        thread via signals so QTimers / QObjects are created safely.
        """
        try:
            from desktop.widgets.playback_overlay import build_segment_timeline

            extra_dirs: list = []

            cam_rec_dir = (self.motion_watch_settings.get("recording_dir") or "").strip()
            if cam_rec_dir:
                extra_dirs.append(cam_rec_dir)

            for key in ("save_dir", "clip_save_dir"):
                d = (self.motion_watch_settings.get(key) or "").strip()
                if d:
                    extra_dirs.append(d)

            try:
                app = __import__("PySide6.QtWidgets", fromlist=["QApplication"]).QApplication.instance()
                if app and hasattr(app, "_load_prefs"):
                    prefs = app._load_prefs() or {}
                    gd = (prefs.get("recording_dir") or "").strip()
                    if gd:
                        extra_dirs.append(gd)
            except Exception:
                pass

            custom = getattr(self, '_playback_custom_dir', None)
            if custom:
                extra_dirs.append(custom)

            use_mediamtx = self.motion_watch_settings.get("playback_source", "file") == "mediamtx"
            segments = build_segment_timeline(
                self.camera_id,
                self.camera_name or "",
                extra_dirs=extra_dirs,
            )
            self._playback_data_ready.emit(segments, use_mediamtx, extra_dirs)
        except Exception:
            self._playback_active = False
            self._playback_loading = False

        try:
            import requests as _req
            now = int(time.time())
            resp = _req.post(
                "http://localhost:5000/api/events/search",
                json={
                    "camera_name": self.camera_name or self.camera_id,
                    "start_ts": now - 86400,
                    "end_ts": now,
                    "limit": 500,
                },
                timeout=10,
            )
            data = resp.json()
            events = data.get("data", {}).get("timeline", [])
            self._playback_events_ready.emit(events)
        except Exception:
            pass

    def _on_playback_data_loaded(self, segments, use_mediamtx, extra_dirs):
        """Runs on the GUI thread -- safe to start timers and create QObjects."""
        if self.playback_overlay:
            self._playback_active = True
            self._playback_loading = False

            # Update sync indicator on the overlay
            is_synced = bool(getattr(self, '_sync_group_id', None))
            self.playback_overlay._synced = is_synced

            sync_ts = getattr(self, '_sync_start_ts', None)

            if sync_ts is not None:
                # Synced open: load segments WITHOUT auto-playing, then
                # seek directly to the requested timestamp so there is
                # no visible jump to the end-of-last-segment.
                self._sync_start_ts = None
                speed = getattr(self, '_sync_start_speed', 1.0)
                self._sync_start_speed = 1.0

                self.playback_overlay.load_segments_no_autoplay(
                    segments, use_mediamtx=use_mediamtx, extra_dirs=extra_dirs)

                self.playback_overlay._engine.speed = speed
                self.playback_overlay._sync_audio_to_speed()

                eng = self.playback_overlay._engine
                has_video = any(
                    seg.start_ts <= sync_ts <= seg.end_ts + 60 or seg.start_ts > sync_ts
                    for seg in (eng._segments or [])
                )
                self.playback_overlay._no_video_at_time = not has_video

                seek_target = sync_ts
                if not has_video:
                    nearest = None
                    nearest_dist = float('inf')
                    for seg in (eng._segments or []):
                        for t in (seg.start_ts, seg.end_ts):
                            d = abs(t - sync_ts)
                            if d < nearest_dist:
                                nearest_dist = d
                                nearest = seg.start_ts
                    if nearest is not None:
                        seek_target = nearest
                    QTimer.singleShot(5000, self._clear_no_video_warning)

                self.playback_overlay.start_playback(start_ts=seek_target)
            else:
                self.playback_overlay.load_segments(
                    segments, use_mediamtx=use_mediamtx, extra_dirs=extra_dirs)

            QTimer.singleShot(0, self._fix_playback_overlay_geometry)

    def _on_playback_events_loaded(self, events):
        """Runs on the GUI thread."""
        if self.playback_overlay:
            self.playback_overlay.load_events(events)

    def _request_sync_playback(self, camera_ids: list, timestamp: float, speed: float):
        """Ask the app to open other cameras at the same playback timestamp."""
        from PySide6.QtWidgets import QApplication
        app = QApplication.instance()
        if app and hasattr(app, 'sync_playback'):
            gid = app.sync_playback(self.camera_id, camera_ids, timestamp, speed)
            self._sync_group_id = gid

    def _request_unsync(self):
        """Remove this camera from its sync group."""
        self._auto_unsync()

    def _auto_unsync(self):
        """Leave the sync group (called when playback closes or user unsyncs)."""
        gid = getattr(self, '_sync_group_id', None)
        if not gid:
            return
        self._sync_group_id = None
        try:
            from PySide6.QtWidgets import QApplication
            app = QApplication.instance()
            if app and hasattr(app, 'unsync_playback'):
                app.unsync_playback(gid)
        except Exception:
            pass

    def _broadcast_sync(self, action: str, **kwargs):
        """Broadcast a playback action to other cameras in the sync group."""
        gid = getattr(self, '_sync_group_id', None)
        if not gid or self._sync_broadcasting:
            return
        try:
            from PySide6.QtWidgets import QApplication
            app = QApplication.instance()
            if app and hasattr(app, 'broadcast_playback_sync'):
                self._sync_broadcasting = True
                app.broadcast_playback_sync(gid, self.camera_id, action, **kwargs)
        except Exception:
            pass
        finally:
            self._sync_broadcasting = False

    def _reconnect_live_after_playback(self):
        """Re-acquire the live camera stream so live view resumes."""
        try:
            import asyncio
            cm = getattr(self, "camera_manager", None)
            loop = getattr(cm, "_loop", None) if cm else None
            if cm and loop and getattr(loop, "is_running", lambda: False)():
                asyncio.run_coroutine_threadsafe(cm.acquire_camera(self.camera_id), loop)
        except Exception:
            pass

    def _toggle_continuous_recording(self):
        """Toggle continuous recording for this camera via the backend API."""
        new_state = not self._continuous_recording
        url = f"http://localhost:5000/api/cameras/{self.camera_id}/recording"
        try:
            import requests as _req
            resp = _req.post(url, json={"record": new_state}, timeout=5)
            data = resp.json()
            if data.get("success"):
                self._continuous_recording = new_state
                self.gl_widget._camera_recording_flag = new_state
                self.gl_widget.update()
                state_str = "started" if new_state else "stopped"
                self._post_motion_watch_to_terminal(
                    f"Recording {state_str} for {self.camera_name or self.camera_id}",
                    suppress_log=False,
                )
            else:
                msg = data.get("message", "Unknown error")
                QMessageBox.warning(self, "Recording", f"Could not toggle recording:\n\n{msg}")
        except Exception as e:
            QMessageBox.warning(
                self, "Recording",
                f"Could not reach the backend to toggle recording.\n\n"
                f"Make sure the Flask backend is running (python app.py or run.py).\n\n"
                f"Error: {e}"
            )

    def _open_recording_settings(self):
        """Open a dialog to configure recording location and overwrite preferences."""
        dlg = QDialog(self)
        dlg.setWindowTitle("Recording Settings")
        dlg.setFixedSize(400, 300)
        dlg.setWindowFlags(Qt.WindowType.Tool | Qt.WindowType.WindowStaysOnTopHint)
        form = QFormLayout(dlg)

        # Recording location
        rec_dir_edit = QLineEdit(self.motion_watch_settings.get("recording_dir", "recordings"))
        browse_btn = QPushButton("Browse...")
        def _pick():
            target = QFileDialog.getExistingDirectory(dlg, "Select recording directory", rec_dir_edit.text() or ".")
            if target:
                rec_dir_edit.setText(target)
        browse_btn.clicked.connect(_pick)
        dir_row = QHBoxLayout()
        dir_row.addWidget(rec_dir_edit)
        dir_row.addWidget(browse_btn)
        form.addRow("Recording location", dir_row)

        # Overwrite preference
        overwrite_check = QCheckBox("Auto-delete oldest recordings when disk is full")
        overwrite_check.setChecked(bool(self.motion_watch_settings.get("storage_auto_manage", False)))
        form.addRow("", overwrite_check)

        threshold_spin = QSpinBox()
        threshold_spin.setRange(50, 98)
        threshold_spin.setSuffix("% disk usage")
        threshold_spin.setValue(int(self.motion_watch_settings.get("storage_max_pct", 85)))
        threshold_spin.setEnabled(overwrite_check.isChecked())
        overwrite_check.toggled.connect(threshold_spin.setEnabled)
        form.addRow("Cleanup threshold", threshold_spin)

        # Playback source
        from PySide6.QtWidgets import QComboBox
        playback_combo = QComboBox()
        playback_combo.addItem("Direct file (least CPU)", "file")
        playback_combo.addItem("MediaMTX API (remote-friendly)", "mediamtx")
        current_src = self.motion_watch_settings.get("playback_source", "file")
        idx = playback_combo.findData(current_src)
        if idx >= 0:
            playback_combo.setCurrentIndex(idx)
        form.addRow("Playback source", playback_combo)

        # Status
        status_label = QLabel("")
        status_label.setStyleSheet("color: #94a3b8; font-size: 11px;")
        try:
            import requests as _req
            st = _req.get("http://localhost:5000/api/storage/status", timeout=3).json()
            recs = st.get("data", {}).get("recordings", {})
            if recs and not recs.get("error"):
                status_label.setText(
                    f"Recordings: {recs.get('used_gb', '?')} GB used / {recs.get('total_gb', '?')} GB total "
                    f"({recs.get('used_percent', '?')}%)"
                )
        except Exception:
            pass
        form.addRow("Disk status", status_label)

        btn_row = QHBoxLayout()
        save_btn = QPushButton("Save")
        cancel_btn = QPushButton("Cancel")
        save_btn.clicked.connect(dlg.accept)
        cancel_btn.clicked.connect(dlg.reject)
        btn_row.addWidget(save_btn)
        btn_row.addWidget(cancel_btn)
        form.addRow(btn_row)

        if dlg.exec() == QDialog.DialogCode.Accepted:
            new_dir = rec_dir_edit.text().strip()
            self.motion_watch_settings["recording_dir"] = new_dir
            self.motion_watch_settings["storage_auto_manage"] = overwrite_check.isChecked()
            self.motion_watch_settings["storage_max_pct"] = threshold_spin.value()
            self.motion_watch_settings["playback_source"] = playback_combo.currentData() or "file"
            self._persist_motion_watch_settings()
            self._sync_storage_settings()
            # Persist per-camera recording directory to the backend
            try:
                import requests as _req
                _req.put(
                    f"http://localhost:5000/api/cameras/{self.camera_id}",
                    json={"recording_dir": new_dir},
                    timeout=3,
                )
            except Exception:
                pass

    def _sync_storage_settings(self):
        """Push storage-management prefs from the Motion Watch dialog to the backend."""
        try:
            import requests as _req
            enabled = bool(self.motion_watch_settings.get("storage_auto_manage", False))
            max_pct = int(self.motion_watch_settings.get("storage_max_pct", 85))
            _req.post(
                "http://localhost:5000/api/storage/settings",
                json={"enabled": enabled, "max_usage_percent": max_pct},
                timeout=3,
            )
        except Exception:
            pass

    def _open_global_recording_settings(self):
        """Open the System Management dialog so the user can manage global recording/disk settings."""
        from PySide6.QtWidgets import QApplication
        app = QApplication.instance()
        if app and hasattr(app, 'open_system_manager'):
            app.open_system_manager()

    def _post_motion_watch_to_terminal(
        self,
        text: str,
        countdown: Optional[int] = None,
        image_b64: Optional[str] = None,
        remaining_seconds: Optional[int] = None,
        stopped: bool = False,
        suppress_log: bool = False,
    ):
        """Send updates to all terminal widgets; best-effort and non-blocking."""
        try:
            from desktop.widgets.terminal import TerminalWidget
            TerminalWidget.broadcast_motion_watch(
                self.camera_id,
                text,
                countdown=countdown,
                image_b64=image_b64,
                remaining_seconds=remaining_seconds,
                stopped=stopped,
                camera_label=self._camera_label(),
                suppress_log=suppress_log,
            )
        except Exception as e:
            print(f"Motion watch terminal notify error for {self.camera_id}: {e}")

    def _camera_label(self) -> str:
        return self.camera_name or self.camera_id

    def _ensure_terminal_widget(self):
        """Spawn a terminal widget if none are currently active."""
        try:
            from desktop.widgets.terminal import TerminalWidget
            if TerminalWidget._instances:
                return
        except Exception:
            pass
        try:
            payload = json.dumps({"cmd": "spawn_terminal"}).encode("utf-8")
            with socket.create_connection(("127.0.0.1", 5555), timeout=1.5) as sock:
                sock.sendall(payload)
        except Exception as e:
            print(f"Motion watch terminal spawn error: {e}")

    def start_motion_watch(self):
        """Arm motion watch for the configured duration."""
        duration = int(self.motion_watch_settings.get("duration_sec", 30))
        # Make sure a terminal is up to display countdown/images
        self._ensure_terminal_widget()
        self.motion_watch_active = True
        if duration < 0:
            self.motion_watch_end_ts = None  # infinite
        else:
            duration = max(1, duration)
            self.motion_watch_end_ts = time.time() + duration
        self.motion_watch_last_trigger = 0.0
        if not self.motion_watch_timer.isActive():
            self.motion_watch_timer.start(1000)
        self._post_motion_watch_to_terminal(
            "Motion watch armed",
            countdown=duration if duration >= 0 else None,
            remaining_seconds=duration if duration >= 0 else None,
        )

    def stop_motion_watch(self, reason: str = "stopped"):
        if not self.motion_watch_active:
            return
        self.motion_watch_active = False
        self.motion_watch_timer.stop()
        self._post_motion_watch_to_terminal(f"Motion watch {reason}", stopped=True)

    def _remaining_watch_seconds(self) -> Optional[int]:
        if not self.motion_watch_active:
            return None
        if self.motion_watch_end_ts is None:
            return None  # infinite
        return max(0, int(self.motion_watch_end_ts - time.time()))

    def _tick_motion_watch(self):
        if not self.motion_watch_active:
            return
        if self.motion_watch_end_ts is not None:
            remaining = int(self.motion_watch_end_ts - time.time())
            if remaining <= 0:
                self.stop_motion_watch("complete")
                return
        else:
            remaining = None
        # Keep countdown visible inside terminal
        self._post_motion_watch_to_terminal(
            "Active",
            countdown=remaining if remaining is not None else None,
            remaining_seconds=remaining,
            suppress_log=True,
        )

    def _capture_motion_watch_shot_async(
        self,
        motion_box: Optional[Tuple[int, int, int, int]] = None,
        *,
        trigger_events: Optional[List[dict]] = None,
        trigger_source: Optional[str] = None,
        remaining_seconds: Optional[int] = None,
    ) -> None:
        """
        Capture a motion-watch screenshot without blocking the UI thread.

        Important: We only grab a copy of the latest frame on the UI thread. Everything else
        (overlay drawing, crop/resize, JPG encode, disk write, base64, sidecar/ingest) runs in a
        background worker.
        """
        try:
            acquired_global_capture = False
            with self._motion_watch_capture_lock:
                if self._motion_watch_capture_inflight:
                    return
                self._motion_watch_capture_inflight = True
            # Cross-camera throttle. If the system is already busy with other capture workers,
            # skip this trigger rather than stalling all live feeds.
            acquired_global_capture = CameraWidget._motion_watch_capture_sem.acquire(blocking=False)
            if not acquired_global_capture:
                with self._motion_watch_capture_lock:
                    self._motion_watch_capture_inflight = False
                return

            settings = dict(self.motion_watch_settings or {})
            include_overlays = bool(settings.get("include_overlays", True))
            non_blocking = bool(settings.get("non_blocking_capture", True))
            local_enrich = bool(settings.get("local_enrich", False))
            persist_overlay_snapshot = bool(settings.get("persist_overlay_snapshot", False))

            # Prefer the raw frame buffer (no GPU readback). If overlays are requested, we paint
            # them onto the frame in the worker using a snapshot of current overlay state.
            frame_img: Optional[QImage] = None
            overlay_snapshot: Optional[dict] = None
            if self.gl_widget.image and not self.gl_widget.image.isNull():
                frame_img = self.gl_widget.image.copy()
                if include_overlays:
                    overlay_snapshot = self.get_overlay_snapshot() or None
            else:
                # Fallback: last resort capture from the GL widget.
                if include_overlays and not non_blocking:
                    frame_img = self.gl_widget.grabFramebuffer()
                    overlay_snapshot = None
                else:
                    # In non-blocking mode, skip capture rather than forcing a GPU readback.
                    with self._motion_watch_capture_lock:
                        self._motion_watch_capture_inflight = False
                    if acquired_global_capture:
                        try:
                            CameraWidget._motion_watch_capture_sem.release()
                        except Exception:
                            pass
                    return

            if not frame_img or frame_img.isNull():
                with self._motion_watch_capture_lock:
                    self._motion_watch_capture_inflight = False
                if acquired_global_capture:
                    try:
                        CameraWidget._motion_watch_capture_sem.release()
                    except Exception:
                        pass
                return

            camera_id = self.camera_id
            camera_label = self._camera_label()
            api_base = getattr(self, "API_BASE", "http://localhost:5000/api")
            captured_ts = int(time.time())

            # Precompute trigger payload (cheap) on UI thread.
            trigger = {}
            ev0 = (trigger_events or [None])[0] if trigger_events else None
            if isinstance(ev0, dict):
                trigger = {
                    "interaction_type": ev0.get("interaction_type") or ev0.get("kind") or "motion_watch",
                    "shape_type": ev0.get("shape_type"),
                    "shape_id": ev0.get("shape_id") or ev0.get("id"),
                    "shape_name": ev0.get("shape_name") or ev0.get("name"),
                    "source": trigger_source,
                }
            motion_box_payload = None
            if motion_box and len(motion_box) >= 4:
                bx, by, bw, bh = motion_box
                motion_box_payload = {"x": int(bx), "y": int(by), "w": int(bw), "h": int(bh)}

            def _worker(img: QImage):
                try:
                    # Apply overlays in worker (if requested and we have a snapshot).
                    if include_overlays and isinstance(overlay_snapshot, dict):
                        try:
                            self._paint_overlay_snapshot(img, overlay_snapshot, motion_box=motion_box)
                        except Exception:
                            pass

                    # Crop/resize, then encode once.
                    img_w, img_h = img.width(), img.height()
                    crop_w = int(settings.get("crop_w", 0) or 0)
                    crop_h = int(settings.get("crop_h", 0) or 0)
                    if (crop_w > 0 or crop_h > 0) and motion_box:
                        bx, by, bw, bh = motion_box
                        cx = bx + bw / 2
                        cy = by + bh / 2
                        cw = crop_w or bw
                        ch = crop_h or bh
                        x = max(0, int(cx - cw / 2))
                        y = max(0, int(cy - ch / 2))
                        if x + cw > img_w:
                            x = max(0, img_w - cw)
                        if y + ch > img_h:
                            y = max(0, img_h - ch)
                        cw = min(int(cw), img_w)
                        ch = min(int(ch), img_h)
                        img = img.copy(x, y, cw, ch)

                    resize_w = int(settings.get("resize_w", 0) or 0)
                    if resize_w > 0 and img.width() > 0:
                        new_h = int(img.height() * (resize_w / img.width()))
                        new_h = max(1, new_h)
                        img = img.scaled(
                            resize_w,
                            new_h,
                            Qt.AspectRatioMode.KeepAspectRatio,
                            Qt.TransformationMode.SmoothTransformation,
                        )

                    from core.paths import get_data_dir

                    save_dir = Path(settings.get("save_dir", "") or "captures/motion_watch")
                    if not save_dir.is_absolute():
                        save_dir = (get_data_dir().parent / save_dir)
                    # Organize into zone-named subdirectory
                    shape_name_dir = _sanitize_zone_dirname(trigger.get("shape_name") or "")
                    save_dir = save_dir / shape_name_dir
                    save_dir.mkdir(parents=True, exist_ok=True)
                    fname = save_dir / f"{camera_id}_watch_{captured_ts}.jpg"

                    buf = QBuffer()
                    buf.open(QIODevice.OpenModeFlag.WriteOnly)
                    ok = img.save(buf, "JPG", quality=85)
                    if not ok:
                        return
                    jpg_bytes = bytes(buf.data())
                    try:
                        fname.write_bytes(jpg_bytes)
                    except Exception:
                        # best-effort: if disk write fails, still allow terminal preview
                        pass
                    # Send a small thumbnail to the terminal to keep UI fast, and include a link to the full file.
                    thumb_b64 = ""
                    try:
                        thumb = img
                        if thumb.width() > 360:
                            new_h = int(thumb.height() * (360 / max(1, thumb.width())))
                            new_h = max(1, new_h)
                            thumb = thumb.scaled(
                                360,
                                new_h,
                                Qt.AspectRatioMode.KeepAspectRatio,
                                Qt.TransformationMode.SmoothTransformation,
                            )
                        tbuf = QBuffer()
                        tbuf.open(QIODevice.OpenModeFlag.WriteOnly)
                        if thumb.save(tbuf, "JPG", quality=55):
                            thumb_b64 = base64.b64encode(bytes(tbuf.data())).decode("utf-8")
                    except Exception:
                        thumb_b64 = ""

                    file_url = ""
                    try:
                        from PySide6.QtCore import QUrl
                        file_url = QUrl.fromLocalFile(str(fname)).toString()
                    except Exception:
                        # Safe fallback: Windows-friendly file URL without backslashes.
                        file_url = "file:///" + fname.as_posix()

                    sidecar_payload = {
                        "file_path": str(fname),
                        "media_type": "image",
                        "captured_ts": captured_ts,
                        "camera_id": camera_id,
                        "camera_name": camera_label,
                        "enable_vision": False,
                        "enable_detections": False,
                        **({"trigger": trigger} if trigger else {}),
                        **({"motion_box": motion_box_payload} if motion_box_payload else {}),
                        "metadata": {
                            "media_type": "image",
                            "include_overlays": include_overlays,
                            "overlay_snapshot": (
                                overlay_snapshot
                                if (include_overlays and persist_overlay_snapshot)
                                else None
                            ),
                        },
                    }
                    sidecar_path = fname.with_suffix(".json")
                    try:
                        sidecar_path.write_text(json.dumps(sidecar_payload, separators=(",", ":")))
                    except Exception:
                        pass

                    def _enrich_and_ingest():
                        try:
                            # Don't let enrichment pile up; skip if another is running.
                            if not CameraWidget._motion_watch_ingest_sem.acquire(blocking=False):
                                return
                            # Optional: local enrich (can be CPU-heavy). Default OFF to keep live feeds smooth;
                            # backend post-processing can fill in detections later.
                            if local_enrich:
                                try:
                                    from core.object_detector import ObjectDetector

                                    img_cv = cv2.imread(str(fname))
                                    if img_cv is not None:
                                        det = getattr(CameraWidget, "_motion_watch_yolo", None)
                                        if det is None:
                                            det = ObjectDetector(model_type="yolo", device="auto")
                                            setattr(CameraWidget, "_motion_watch_yolo", det)
                                        dets = det.detect(img_cv, conf_threshold=0.25) or []
                                        labels = []
                                        detections_payload = []
                                        for d in dets:
                                            if not isinstance(d, dict):
                                                continue
                                            lab = d.get("class") or d.get("label") or d.get("class_name")
                                            if isinstance(lab, str) and lab.strip():
                                                labels.append(lab.strip().lower())
                                            try:
                                                bb = d.get("bbox")
                                                conf = float(d.get("confidence", 0.0) or 0.0)
                                                if isinstance(lab, str) and lab.strip() and isinstance(bb, dict):
                                                    detections_payload.append(
                                                        {
                                                            "class": lab.strip().lower(),
                                                            "confidence": conf,
                                                            "bbox": {
                                                                "x": float(bb.get("x", 0) or 0),
                                                                "y": float(bb.get("y", 0) or 0),
                                                                "w": float(bb.get("w", 0) or 0),
                                                                "h": float(bb.get("h", 0) or 0),
                                                            },
                                                        }
                                                    )
                                            except Exception:
                                                continue
                                        labels = list(dict.fromkeys([l for l in labels if l]))[:24]
                                        if labels:
                                            sidecar_payload["detection_classes"] = labels
                                            tags = sidecar_payload.get("tags") or []
                                            if not isinstance(tags, list):
                                                tags = []
                                            sidecar_payload["tags"] = list(dict.fromkeys([*(tags or []), *labels]))[:24]
                                        sidecar_payload.setdefault("metadata", {})
                                        if isinstance(sidecar_payload["metadata"], dict):
                                            sidecar_payload["metadata"]["yolo"] = {
                                                "detections": [
                                                    {
                                                        "class": (d.get("class") or d.get("label") or d.get("class_name")),
                                                        "confidence": float(d.get("confidence", 0.0) or 0.0),
                                                        "bbox": d.get("bbox"),
                                                    }
                                                    for d in dets[:15]
                                                    if isinstance(d, dict)
                                                ]
                                            }
                                        if detections_payload:
                                            sidecar_payload["detections"] = detections_payload[:50]
                                except Exception:
                                    pass

                            try:
                                sidecar_path.write_text(json.dumps(sidecar_payload, separators=(",", ":")))
                            except Exception:
                                pass

                            requests.post(f"{api_base}/events/ingest", json=sidecar_payload, timeout=20)
                        except Exception:
                            pass
                        finally:
                            try:
                                CameraWidget._motion_watch_ingest_sem.release()
                            except Exception:
                                pass

                    threading.Thread(target=_enrich_and_ingest, daemon=True).start()

                    # Notify terminal (thread-safe; uses Qt signals internally).
                    try:
                        from desktop.widgets.terminal import TerminalWidget

                        TerminalWidget.broadcast_motion_watch(
                            camera_id,
                            f"Captured {fname}",
                            image_b64=thumb_b64 or None,
                            link=file_url or None,
                            link_label="Open full image" if file_url else None,
                            remaining_seconds=remaining_seconds,
                            camera_label=camera_label,
                            suppress_log=False,
                        )
                    except Exception:
                        pass
                finally:
                    with self._motion_watch_capture_lock:
                        self._motion_watch_capture_inflight = False
                    if acquired_global_capture:
                        try:
                            CameraWidget._motion_watch_capture_sem.release()
                        except Exception:
                            pass

            threading.Thread(target=_worker, args=(frame_img,), daemon=True).start()
        except Exception as e:
            print(f"Motion watch capture error for {self.camera_id}: {e}")
            with self._motion_watch_capture_lock:
                self._motion_watch_capture_inflight = False
            try:
                if acquired_global_capture:
                    CameraWidget._motion_watch_capture_sem.release()
            except Exception:
                pass

    # ------------------------------------------------------------------
    # Clip recording: pre-roll drain, live accumulation, MP4 encode
    # ------------------------------------------------------------------

    def _start_clip_recording(
        self,
        trigger_events: Optional[List[dict]] = None,
        trigger_source: Optional[str] = None,
    ) -> None:
        """Begin recording a video clip with pre-roll from the frame ring buffer."""
        with self._clip_record_lock:
            if self._clip_recording:
                # Extend deadline instead of starting a new clip
                extra = float(self.motion_watch_settings.get("clip_duration_sec", 10))
                self._clip_record_deadline = time.time() + extra
                return

            settings = dict(self.motion_watch_settings or {})
            pre_roll_sec = float(settings.get("clip_pre_roll_sec", 5))
            duration_sec = float(settings.get("clip_duration_sec", 10))

            # Drain pre-roll from ring buffer
            now = time.time()
            cutoff = now - pre_roll_sec
            pre_frames = [(ts, jpg) for ts, jpg in self._clip_frame_ring if ts >= cutoff and jpg]

            # Build trigger metadata
            trigger = {}
            ev0 = (trigger_events or [None])[0] if trigger_events else None
            if isinstance(ev0, dict):
                trigger = {
                    "interaction_type": ev0.get("interaction_type") or ev0.get("kind") or "motion_watch",
                    "shape_type": ev0.get("shape_type"),
                    "shape_id": ev0.get("shape_id") or ev0.get("id"),
                    "shape_name": ev0.get("shape_name") or ev0.get("name"),
                    "source": trigger_source,
                }

            self._clip_record_buf = list(pre_frames)
            self._clip_recording = True
            self._clip_record_deadline = now + duration_sec
            self._clip_record_trigger = trigger

    def _finalize_clip_recording(self) -> None:
        """Encode accumulated frames to MP4 and write sidecar JSON."""
        with self._clip_record_lock:
            if not self._clip_recording:
                return
            self._clip_recording = False
            frames = [f for f in self._clip_record_buf if f and f[1]]
            self._clip_record_buf = []
            trigger = self._clip_record_trigger or {}
            self._clip_record_trigger = None

        if len(frames) < 2:
            return

        settings = dict(self.motion_watch_settings or {})
        camera_id = self.camera_id
        camera_label = self._camera_label()
        api_base = getattr(self, "API_BASE", "http://localhost:5000/api")
        captured_ts = int(time.time())
        clip_resize_w = int(settings.get("clip_resize_w", 640) or 0)
        clip_quality = int(settings.get("clip_quality", 23) or 23)
        clip_duration_sec = int(settings.get("clip_duration_sec", 10))
        clip_pre_roll_sec = int(settings.get("clip_pre_roll_sec", 5))

        def _encode_worker():
            try:
                from core.paths import get_data_dir
                import shutil

                # Use dedicated clip directory if set, otherwise fall back to photo save dir
                clip_dir_raw = str(settings.get("clip_save_dir", "") or "").strip()
                if clip_dir_raw:
                    save_dir = Path(clip_dir_raw)
                else:
                    save_dir = Path(settings.get("save_dir", "") or "captures/motion_watch")
                if not save_dir.is_absolute():
                    save_dir = get_data_dir().parent / save_dir

                # Zone-based subdirectory
                shape_name = trigger.get("shape_name") or ""
                zone_dir = _sanitize_zone_dirname(shape_name)
                save_dir = save_dir / zone_dir
                save_dir.mkdir(parents=True, exist_ok=True)

                fname = save_dir / f"{camera_id}_clip_{captured_ts}.mp4"

                # Decode first frame to get dimensions
                first_jpg = frames[0][1]
                first_arr = cv2.imdecode(np.frombuffer(first_jpg, dtype=np.uint8), cv2.IMREAD_COLOR)
                if first_arr is None:
                    return
                h, w = first_arr.shape[:2]

                if clip_resize_w > 0 and w > clip_resize_w:
                    scale = clip_resize_w / w
                    w = clip_resize_w
                    h = max(1, int(h * scale))
                    # Ensure even dimensions for codec compatibility
                    w = w if w % 2 == 0 else w + 1
                    h = h if h % 2 == 0 else h + 1

                # Estimate FPS from frame timestamps
                ts_list = [f[0] for f in frames]
                total_span = ts_list[-1] - ts_list[0]
                fps = len(frames) / total_span if total_span > 0.1 else self._clip_ring_fps
                fps = max(5.0, min(30.0, fps))

                # Try FFmpeg first for better compression, fall back to OpenCV
                ffmpeg_ok = False
                try:
                    ffmpeg_path = shutil.which("ffmpeg")
                    if ffmpeg_path:
                        import subprocess
                        cmd = [
                            ffmpeg_path, '-y',
                            '-f', 'rawvideo', '-pix_fmt', 'bgr24',
                            '-s', f'{w}x{h}', '-r', str(round(fps, 2)),
                            '-i', '-',
                            '-c:v', 'libx264', '-preset', 'fast',
                            '-crf', str(clip_quality),
                            '-pix_fmt', 'yuv420p',
                            '-movflags', '+faststart',
                            str(fname),
                        ]
                        proc = subprocess.Popen(
                            cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                        )
                        for _, jpg_data in frames:
                            arr = cv2.imdecode(np.frombuffer(jpg_data, dtype=np.uint8), cv2.IMREAD_COLOR)
                            if arr is None:
                                continue
                            if arr.shape[1] != w or arr.shape[0] != h:
                                arr = cv2.resize(arr, (w, h))
                            proc.stdin.write(arr.tobytes())
                        proc.stdin.close()
                        proc.wait(timeout=60)
                        ffmpeg_ok = proc.returncode == 0 and fname.exists() and fname.stat().st_size > 0
                except Exception:
                    ffmpeg_ok = False

                if not ffmpeg_ok:
                    # Try H.264 (avc1) first for browser compat, fall back to mp4v
                    out = None
                    for codec in ['avc1', 'mp4v']:
                        fourcc = cv2.VideoWriter_fourcc(*codec)
                        out = cv2.VideoWriter(str(fname), fourcc, fps, (w, h))
                        if out.isOpened():
                            break
                        out = None
                    if out is None:
                        return
                    for _, jpg_data in frames:
                        arr = cv2.imdecode(np.frombuffer(jpg_data, dtype=np.uint8), cv2.IMREAD_COLOR)
                        if arr is None:
                            continue
                        if arr.shape[1] != w or arr.shape[0] != h:
                            arr = cv2.resize(arr, (w, h))
                        out.write(arr)
                    out.release()

                if not fname.exists() or fname.stat().st_size == 0:
                    return

                # Generate a static thumbnail from the first frame
                thumb_path = fname.with_suffix(".thumb.jpg")
                try:
                    thumb_arr = cv2.imdecode(np.frombuffer(first_jpg, dtype=np.uint8), cv2.IMREAD_COLOR)
                    if thumb_arr is not None:
                        if thumb_arr.shape[1] > 360:
                            scale = 360 / thumb_arr.shape[1]
                            thumb_arr = cv2.resize(thumb_arr, (360, max(1, int(thumb_arr.shape[0] * scale))))
                        cv2.imwrite(str(thumb_path), thumb_arr, [cv2.IMWRITE_JPEG_QUALITY, 70])
                except Exception:
                    thumb_path = None

                # Generate an animated preview (short looping MP4, ~3s, small res)
                # for moving thumbnails in the live security report.
                preview_path = fname.with_suffix(".preview.mp4")
                try:
                    preview_w = min(320, w)
                    preview_scale = preview_w / w
                    preview_h = max(2, int(h * preview_scale))
                    preview_h = preview_h if preview_h % 2 == 0 else preview_h + 1
                    preview_w = preview_w if preview_w % 2 == 0 else preview_w + 1
                    # Sample evenly across all frames, cap at ~3s worth
                    max_preview_frames = int(min(fps, 10) * 3)
                    if len(frames) > max_preview_frames:
                        step = len(frames) / max_preview_frames
                        sampled = [frames[int(i * step)] for i in range(max_preview_frames)]
                    else:
                        sampled = frames
                    preview_fps = min(10.0, fps)

                    preview_ok = False
                    if ffmpeg_path:
                        pcmd = [
                            ffmpeg_path, '-y',
                            '-f', 'rawvideo', '-pix_fmt', 'bgr24',
                            '-s', f'{preview_w}x{preview_h}', '-r', str(round(preview_fps, 2)),
                            '-i', '-',
                            '-c:v', 'libx264', '-preset', 'ultrafast',
                            '-crf', '28', '-pix_fmt', 'yuv420p',
                            '-movflags', '+faststart',
                            '-an', str(preview_path),
                        ]
                        pproc = subprocess.Popen(
                            pcmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                        )
                        for _, jpg_data in sampled:
                            parr = cv2.imdecode(np.frombuffer(jpg_data, dtype=np.uint8), cv2.IMREAD_COLOR)
                            if parr is None:
                                continue
                            parr = cv2.resize(parr, (preview_w, preview_h))
                            pproc.stdin.write(parr.tobytes())
                        pproc.stdin.close()
                        pproc.wait(timeout=30)
                        preview_ok = pproc.returncode == 0 and preview_path.exists() and preview_path.stat().st_size > 0

                    if not preview_ok:
                        # Fallback: OpenCV with avc1 (H.264) or mp4v
                        for codec in ['avc1', 'mp4v']:
                            fourcc_p = cv2.VideoWriter_fourcc(*codec)
                            pout = cv2.VideoWriter(str(preview_path), fourcc_p, preview_fps, (preview_w, preview_h))
                            if pout.isOpened():
                                for _, jpg_data in sampled:
                                    parr = cv2.imdecode(np.frombuffer(jpg_data, dtype=np.uint8), cv2.IMREAD_COLOR)
                                    if parr is None:
                                        continue
                                    parr = cv2.resize(parr, (preview_w, preview_h))
                                    pout.write(parr)
                                pout.release()
                                if preview_path.exists() and preview_path.stat().st_size > 0:
                                    break
                except Exception:
                    pass

                # Sidecar JSON
                sidecar_payload = {
                    "file_path": str(fname),
                    "media_type": "clip",
                    "clip_duration_sec": clip_duration_sec,
                    "clip_pre_roll_sec": clip_pre_roll_sec,
                    "captured_ts": captured_ts,
                    "camera_id": camera_id,
                    "camera_name": camera_label,
                    "enable_vision": False,
                    "enable_detections": False,
                    **({"trigger": trigger} if trigger else {}),
                    "metadata": {
                        "media_type": "clip",
                        "clip_duration_sec": clip_duration_sec,
                        "clip_pre_roll_sec": clip_pre_roll_sec,
                        "frame_count": len(frames),
                        "fps": round(fps, 2),
                        "resolution": f"{w}x{h}",
                    },
                }
                sidecar_path = fname.with_suffix(".json")
                try:
                    sidecar_path.write_text(json.dumps(sidecar_payload, separators=(",", ":")))
                except Exception:
                    pass

                # Ingest into event index
                try:
                    if CameraWidget._motion_watch_ingest_sem.acquire(blocking=False):
                        try:
                            requests.post(f"{api_base}/events/ingest", json=sidecar_payload, timeout=20)
                        finally:
                            CameraWidget._motion_watch_ingest_sem.release()
                except Exception:
                    pass

                # Notify terminal
                try:
                    from desktop.widgets.terminal import TerminalWidget
                    TerminalWidget.broadcast_motion_watch(
                        camera_id,
                        f"Clip recorded: {fname.name} ({len(frames)} frames)",
                        remaining_seconds=self._remaining_watch_seconds(),
                        camera_label=camera_label,
                        suppress_log=False,
                    )
                except Exception:
                    pass
            except Exception as e:
                print(f"Clip encode error for {camera_id}: {e}")

        threading.Thread(target=_encode_worker, daemon=True).start()

    def _paint_overlay_snapshot(self, img: QImage, snapshot: dict, *, motion_box: Optional[Tuple[int, int, int, int]] = None) -> None:
        """
        Best-effort overlay renderer for motion-watch screenshots.
        Draws zones/lines/tags onto the raw frame in a worker thread to avoid UI stalls.
        """
        if img is None or img.isNull() or not isinstance(snapshot, dict):
            return
        w = int(img.width() or 0)
        h = int(img.height() or 0)
        if w <= 0 or h <= 0:
            return

        shapes = snapshot.get("shapes") or []
        show_labels = bool(snapshot.get("show_shape_labels"))

        painter = QPainter(img)
        try:
            painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)

            # Motion box highlight
            if motion_box and len(motion_box) >= 4:
                bx, by, bw, bh = motion_box
                pen = QPen(QColor("#ef4444"))
                pen.setWidth(2)
                painter.setPen(pen)
                painter.setBrush(Qt.BrushStyle.NoBrush)
                painter.drawRect(int(bx), int(by), int(bw), int(bh))

            font = QFont()
            font.setPointSize(11)
            painter.setFont(font)

            for sh in shapes:
                if not isinstance(sh, dict):
                    continue
                kind = (sh.get("kind") or "").lower()
                color = QColor(str(sh.get("color") or "#22d3ee"))
                thickness = int(float(sh.get("line_thickness", 2.0) or 2.0))
                thickness = max(1, min(8, thickness))
                pen = QPen(color)
                pen.setWidth(thickness)
                painter.setPen(pen)
                painter.setBrush(Qt.BrushStyle.NoBrush)

                label = sh.get("label") or sh.get("name") or ""

                if kind == "zone":
                    pts = sh.get("points") or sh.get("pts") or []
                    poly = QPolygonF()
                    for p in pts:
                        if isinstance(p, (list, tuple)) and len(p) >= 2:
                            nx, ny = float(p[0]), float(p[1])
                        elif isinstance(p, dict):
                            nx, ny = float(p.get("x", 0)), float(p.get("y", 0))
                        else:
                            continue
                        poly.append(QPointF(nx * w, ny * h))
                    if poly.count() >= 3:
                        painter.drawPolygon(poly)
                        if show_labels and isinstance(label, str) and label.strip():
                            p0 = poly.at(0)
                            painter.drawText(int(p0.x()) + 6, int(p0.y()) - 6, label.strip())

                elif kind == "line":
                    pts = sh.get("points") or sh.get("pts") or []
                    if isinstance(pts, list) and len(pts) >= 2:
                        def _pt(val):
                            if isinstance(val, (list, tuple)) and len(val) >= 2:
                                return float(val[0]), float(val[1])
                            if isinstance(val, dict):
                                return float(val.get("x", 0)), float(val.get("y", 0))
                            return None
                        a = _pt(pts[0])
                        b = _pt(pts[1])
                        if a and b:
                            ax, ay = a
                            bx, by = b
                            painter.drawLine(int(ax * w), int(ay * h), int(bx * w), int(by * h))
                            if show_labels and isinstance(label, str) and label.strip():
                                painter.drawText(int(ax * w) + 6, int(ay * h) - 6, label.strip())

                elif kind == "tag":
                    pts = sh.get("points") or sh.get("pts") or []
                    anchor = None
                    if isinstance(pts, list) and pts:
                        val = pts[0]
                        if isinstance(val, (list, tuple)) and len(val) >= 2:
                            anchor = (float(val[0]), float(val[1]))
                        elif isinstance(val, dict):
                            anchor = (float(val.get("x", 0)), float(val.get("y", 0)))
                    if anchor:
                        ax, ay = anchor
                        cx, cy = int(ax * w), int(ay * h)
                        r = int(float(sh.get("tag_size", 18) or 18))
                        r = max(6, min(80, r))
                        painter.drawEllipse(QPointF(cx, cy), r, r)
                        painter.drawLine(cx - r, cy, cx + r, cy)
                        painter.drawLine(cx, cy - r, cx, cy + r)
                        if show_labels and isinstance(label, str) and label.strip():
                            painter.drawText(cx + r + 6, cy - 6, label.strip())
        finally:
            painter.end()

    def _start_loading(self):
        self.loading_phase = 0
        self.loading_label.show()
        if not self.loading_timer.isActive():
            self.loading_timer.start(300)

    def _stop_loading(self):
        if self.loading_timer.isActive():
            self.loading_timer.stop()
        self.loading_label.hide()

    def _tick_loading(self):
        # Simple pulsing dots to keep it light and unobtrusive
        dots = "." * ((self.loading_phase % 3) + 1)
        self.loading_label.setText(f"Loading{dots}")
        self.loading_phase += 1

    def _is_simple_controls(self) -> bool:
        from PySide6.QtWidgets import QApplication
        app = QApplication.instance()
        if app and hasattr(app, '_load_prefs'):
            return bool((app._load_prefs() or {}).get('simple_controls', False))
        return False

    def _toggle_simple_controls(self, checked: bool):
        from PySide6.QtWidgets import QApplication
        app = QApplication.instance()
        if app and hasattr(app, '_load_prefs') and hasattr(app, '_save_prefs'):
            prefs = app._load_prefs() or {}
            prefs['simple_controls'] = bool(checked)
            app._save_prefs(prefs)

    def _toggle_hotkeys(self, checked: bool):
        from PySide6.QtWidgets import QApplication
        app = QApplication.instance()
        if app and hasattr(app, '_load_prefs') and hasattr(app, '_save_prefs'):
            prefs = app._load_prefs() or {}
            prefs['hotkeys_enabled'] = bool(checked)
            app._save_prefs(prefs)

    def _request_patrol(self):
        from PySide6.QtWidgets import QApplication
        app = QApplication.instance()
        if app and hasattr(app, 'toggle_patrol'):
            app.toggle_patrol()

    def _patrol_toggle_camera(self, camera_id: str, include: bool):
        """Add or remove a camera from the persisted patrol order."""
        from PySide6.QtWidgets import QApplication
        app = QApplication.instance()
        if not app or not hasattr(app, '_load_prefs'):
            return
        prefs = app._load_prefs() or {}
        order = list(prefs.get("patrol_order") or [])
        if include and camera_id not in order:
            order.append(camera_id)
        elif not include and camera_id in order:
            order.remove(camera_id)
        prefs["patrol_order"] = order
        app._save_prefs(prefs)

    def _patrol_move(self, camera_id: str, direction: int):
        """Move a camera earlier (-1) or later (+1) in the patrol order."""
        from PySide6.QtWidgets import QApplication
        app = QApplication.instance()
        if not app or not hasattr(app, '_load_prefs'):
            return
        prefs = app._load_prefs() or {}
        order = list(prefs.get("patrol_order") or [])
        if camera_id not in order:
            return
        idx = order.index(camera_id)
        new_idx = max(0, min(len(order) - 1, idx + direction))
        if new_idx != idx:
            order.insert(new_idx, order.pop(idx))
            prefs["patrol_order"] = order
            app._save_prefs(prefs)

    def _open_patrol_settings(self):
        from PySide6.QtWidgets import (
            QApplication, QDialog, QFormLayout, QSpinBox,
            QDialogButtonBox,
        )
        app = QApplication.instance()
        if not app or not hasattr(app, '_load_prefs'):
            return
        prefs = app._load_prefs() or {}
        cur_interval = int(prefs.get("patrol_interval_sec", 10))

        dlg = QDialog(self)
        dlg.setWindowTitle("Patrol Settings")
        dlg.setMinimumWidth(280)
        form = QFormLayout(dlg)

        sp_interval = QSpinBox()
        sp_interval.setRange(2, 300)
        sp_interval.setSuffix(" s")
        sp_interval.setValue(cur_interval)
        form.addRow("Cycle interval:", sp_interval)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(dlg.accept)
        buttons.rejected.connect(dlg.reject)
        form.addRow(buttons)

        if dlg.exec() != QDialog.DialogCode.Accepted:
            return
        new_interval = sp_interval.value()
        prefs["patrol_interval_sec"] = new_interval
        app._save_prefs(prefs)
        if getattr(app, '_patrol_active', False):
            app.start_patrol(interval_sec=new_interval)

    def _toggle_snap(self):
        from PySide6.QtWidgets import QApplication
        app = QApplication.instance()
        if not app or not hasattr(app, '_load_prefs'):
            return
        prefs = app._load_prefs() or {}
        prefs["snap_enabled"] = not bool(prefs.get("snap_enabled", True))
        app._save_prefs(prefs)

    def _toggle_snap_guides(self):
        from PySide6.QtWidgets import QApplication
        app = QApplication.instance()
        if not app or not hasattr(app, '_load_prefs'):
            return
        prefs = app._load_prefs() or {}
        prefs["snap_show_guides"] = not bool(prefs.get("snap_show_guides", True))
        app._save_prefs(prefs)

    def _request_snap_grid(self):
        from PySide6.QtWidgets import QApplication
        app = QApplication.instance()
        if app and hasattr(app, 'arrange_snap_grid'):
            app.arrange_snap_grid()

    def _open_grid_settings(self):
        from PySide6.QtWidgets import (
            QApplication, QDialog, QFormLayout, QSpinBox,
            QCheckBox, QDialogButtonBox,
        )
        app = QApplication.instance()
        if not app or not hasattr(app, '_load_prefs'):
            return
        prefs = app._load_prefs() or {}

        dlg = QDialog(self)
        dlg.setWindowTitle("Grid & Snap Settings")
        dlg.setMinimumWidth(300)
        form = QFormLayout(dlg)

        cb_enabled = QCheckBox("Enable snap to grid")
        cb_enabled.setChecked(bool(prefs.get("snap_enabled", True)))
        form.addRow(cb_enabled)

        cb_guides = QCheckBox("Show snap guide lines")
        cb_guides.setChecked(bool(prefs.get("snap_show_guides", True)))
        form.addRow(cb_guides)

        sp_edge = QSpinBox()
        sp_edge.setRange(4, 100)
        sp_edge.setSuffix(" px")
        sp_edge.setValue(int(prefs.get("snap_edge_threshold", 24)))
        sp_edge.setToolTip("How close (in pixels) a widget edge must be to another edge to snap together")
        form.addRow("Snap distance:", sp_edge)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(dlg.accept)
        buttons.rejected.connect(dlg.reject)
        form.addRow(buttons)

        if dlg.exec() != QDialog.DialogCode.Accepted:
            return
        prefs["snap_enabled"] = cb_enabled.isChecked()
        prefs["snap_show_guides"] = cb_guides.isChecked()
        prefs["snap_edge_threshold"] = sp_edge.value()
        app._save_prefs(prefs)

    def _hotkeys_enabled(self) -> bool:
        from PySide6.QtWidgets import QApplication
        app = QApplication.instance()
        if app and hasattr(app, '_load_prefs'):
            return bool((app._load_prefs() or {}).get('hotkeys_enabled', True))
        return True

    # ── Chord-key quick-adjust ──
    #
    # Hold one of the chord-modifier keys below, then press a
    # secondary key (arrow / number / letter) to adjust the related
    # setting in real time.  Single-tapping the modifier (no
    # secondary key pressed before release) keeps the original
    # toggle behaviour so the hotkey is non-disruptive.
    #
    # Modifiers:
    #   M (motion):    arrows=size,  1-7=style,  C=color,  N=animation,  Z/X=thickness
    #   D (detection): arrows=tune,  1-7=style,  C=color,  N=animation,  Z/X=thickness
    _MOTION_STYLES = ['Box', 'Fill', 'Corners', 'Circle', 'Bracket', 'Underline', 'Crosshair']
    _MOTION_ANIMATIONS = ['None', 'Pulse', 'Flash', 'Glitch', 'Rainbow']
    _DETECTION_ANIMATIONS = ['None', 'Pulse', 'Flash', 'Glitch', 'Rainbow', 'Glow']
    # (label, QColor) pairs.  Kept short enough to memorize; visually
    # distinct so cycling through them on a busy frame is obvious.
    _PRESET_COLORS = [
        ('Red',     (239,  68,  68)),
        ('Orange',  (249, 115,  22)),
        ('Yellow',  (250, 204,  21)),
        ('Green',   ( 34, 197,  94)),
        ('Cyan',    ( 34, 211, 238)),
        ('Blue',    ( 59, 130, 246)),
        ('Magenta', (217,  70, 239)),
        ('White',   (248, 250, 252)),
    ]

    @property
    def _chord_dispatch(self):
        # Build motion style number bindings (1..7)
        motion_style_keys = {
            getattr(Qt.Key, f"Key_{i+1}"): (
                "Style",
                (lambda s=name: self._set_motion_style(s)),
            )
            for i, name in enumerate(self._MOTION_STYLES)
        }
        detection_style_keys = {
            getattr(Qt.Key, f"Key_{i+1}"): (
                "Style",
                (lambda s=name: self._set_detection_style(s)),
            )
            for i, name in enumerate(self._MOTION_STYLES)
        }
        return {
            # Motion boxes
            Qt.Key.Key_M: {
                Qt.Key.Key_Right: ("Sensitivity", lambda: self._adjust_motion_setting("sensitivity", +5, 1, 100)),
                Qt.Key.Key_Left:  ("Sensitivity", lambda: self._adjust_motion_setting("sensitivity", -5, 1, 100)),
                Qt.Key.Key_Up:    ("Merge size",  lambda: self._adjust_motion_setting("merge_size", +5, 0, 100)),
                Qt.Key.Key_Down:  ("Merge size",  lambda: self._adjust_motion_setting("merge_size", -5, 0, 100)),
                Qt.Key.Key_C:     ("Color",       lambda: self._cycle_motion_color()),
                Qt.Key.Key_N:     ("Animation",   lambda: self._cycle_motion_animation()),
                Qt.Key.Key_X:     ("Thickness",   lambda: self._adjust_motion_setting("thickness", +1, 1, 12)),
                Qt.Key.Key_Z:     ("Thickness",   lambda: self._adjust_motion_setting("thickness", -1, 1, 12)),
                **motion_style_keys,
            },
            # Object detection (desktop)
            Qt.Key.Key_D: {
                Qt.Key.Key_Right: ("Confidence", lambda: self._adjust_detection_conf(+0.05)),
                Qt.Key.Key_Left:  ("Confidence", lambda: self._adjust_detection_conf(-0.05)),
                Qt.Key.Key_Up:    ("FPS limit",  lambda: self._adjust_detection_fps(+1)),
                Qt.Key.Key_Down:  ("FPS limit",  lambda: self._adjust_detection_fps(-1)),
                Qt.Key.Key_C:     ("Color",      lambda: self._cycle_detection_color()),
                Qt.Key.Key_N:     ("Animation",  lambda: self._cycle_detection_animation()),
                Qt.Key.Key_X:     ("Thickness",  lambda: self._adjust_detection_setting("thickness", +1, 1, 12)),
                Qt.Key.Key_Z:     ("Thickness",  lambda: self._adjust_detection_setting("thickness", -1, 1, 12)),
                **detection_style_keys,
            },
        }

    _CHORD_HINT_FORMATS = {
        Qt.Key.Key_M: (
            "MOTION quick-adjust  (release M to keep)\n"
            "  Left/Right  =  Sensitivity\n"
            "  Up/Down     =  Merge size\n"
            "  1..7        =  Style (Box, Fill, Corners, Circle,\n"
            "                       Bracket, Underline, Crosshair)\n"
            "  C           =  Cycle color\n"
            "  N           =  Cycle animation\n"
            "  Z / X       =  Thickness  -/+"
        ),
        Qt.Key.Key_D: (
            "DETECTION quick-adjust  (release D to keep)\n"
            "  Left/Right  =  Confidence\n"
            "  Up/Down     =  FPS limit\n"
            "  1..7        =  Style\n"
            "  C           =  Cycle color\n"
            "  N           =  Cycle animation\n"
            "  Z / X       =  Thickness  -/+"
        ),
    }

    _CHORD_TOGGLES_ON_TAP = {
        # Single-tap behavior preserved when no secondary key was pressed.
        Qt.Key.Key_M: lambda self: self.toggle_motion(),
        Qt.Key.Key_D: lambda self: self.toggle_object_detection(),
    }

    def keyPressEvent(self, event):
        if not self._hotkeys_enabled():
            return super().keyPressEvent(event)

        key = event.key()
        mods = event.modifiers()
        ctrl = bool(mods & Qt.KeyboardModifier.ControlModifier)
        is_repeat = False
        try:
            is_repeat = bool(event.isAutoRepeat())
        except Exception:
            is_repeat = False

        # ---- Chord-key UI (M = motion, D = detection) ----
        #
        # Design intent (kept deliberately simple):
        #   * Pressing M (or D) IMMEDIATELY shows the chord menu.
        #   * Holding M autorepeats; each autorepeat just keeps the
        #     chord alive (no flicker, no menu rebuild).
        #   * Pressing a secondary key (1-7, C, N, Z, X, arrows)
        #     dispatches the action and updates the menu in place.
        #   * Releasing M with no secondary keys pressed -> tap ->
        #     fire the original toggle action (toggle motion / detection).
        #   * Releasing M after using secondary keys -> leave the
        #     final value visible briefly (~1s) then clear.
        #
        # X11 autorepeat handling: on X11 each autorepeat sends a
        # paired Release+Press of the same key, and Qt's
        # isAutoRepeat() flag is unreliable on the synthetic
        # release. We defend by:
        #   1. Cancelling any pending deferred-release timer on EVERY
        #      keyboard press while a chord is active. So as long as
        #      ANY key activity is happening, the chord stays.
        #   2. Deferring the actual release by _CHORD_RELEASE_DEFER_MS
        #      (only when an autorepeat has been seen). If the next
        #      autorepeat press arrives in that window, the timer is
        #      cancelled and the chord persists.
        chord_dispatch = self._chord_dispatch

        # ANY press while chord is active cancels a pending release.
        # This is what keeps the chord alive while the user is
        # actively pressing secondary keys (X11 may stop autorepeating
        # M while another key is being processed).
        if self._chord_modifier_key is not None:
            self._chord_cancel_release_timer()

        # ---- Modifier key press (M / D) ----
        if key in chord_dispatch and not ctrl:
            if is_repeat and self._chord_modifier_key == key:
                # Autorepeat of the active modifier: keep chord alive.
                self._chord_saw_autorepeat = True
                return
            if is_repeat and self._chord_modifier_key is None:
                # Edge case: user was already holding M when the
                # widget gained focus -- the FIRST event we see is
                # an autorepeat. Treat it as activation so the chord
                # menu still appears instead of silently dropping.
                self._chord_modifier_key = key
                self._chord_used = False
                self._chord_saw_autorepeat = True
                self._show_chord_hint(key)
                return
            if is_repeat:
                # Autorepeat of a DIFFERENT modifier than is active.
                # Ignore -- the active chord owns the overlay.
                return
            # Real (non-autorepeat) press. Switch modifier or start
            # fresh chord.
            if self._chord_modifier_key is not None and self._chord_modifier_key != key:
                self._chord_end_no_toggle()
            if self._chord_modifier_key is None:
                self._chord_modifier_key = key
                self._chord_used = False
                self._chord_saw_autorepeat = False
                self._show_chord_hint(key)
            return

        # ---- Secondary key while chord active ----
        if (
            self._chord_modifier_key is not None
            and key in chord_dispatch.get(self._chord_modifier_key, {})
        ):
            label, action = chord_dispatch[self._chord_modifier_key][key]
            try:
                value_text = action()
            except Exception as exc:
                value_text = f"err: {exc}"
            self._chord_used = True
            self._show_chord_value(label, value_text)
            return

        # ---- Esc dismisses an active chord without toggling ----
        if key == Qt.Key.Key_Escape and self._chord_modifier_key is not None:
            self._chord_end_no_toggle()
            return

        # ---- Any other (non-dispatch) key while chord is active ----
        # End the chord silently and let the key fall through to its
        # normal handler below.
        if (
            self._chord_modifier_key is not None
            and not is_repeat
            and key not in chord_dispatch
        ):
            self._chord_end_no_toggle()
            # fall through

        # F / F11 -- toggle fullscreen (maximize / restore)
        if key in (Qt.Key.Key_F, Qt.Key.Key_F11) and not ctrl:
            if self.isMaximized():
                self.showNormal()
            else:
                self.showMaximized()
            return

        # Escape -- restore from maximized, or close
        if key == Qt.Key.Key_Escape:
            if self.isMaximized():
                self.showNormal()
            return

        # R -- toggle recording
        if key == Qt.Key.Key_R and not ctrl:
            self._toggle_continuous_recording()
            return

        # P -- toggle patrol
        if key == Qt.Key.Key_P and not ctrl:
            self._request_patrol()
            return

        # A -- toggle aspect ratio lock
        if key == Qt.Key.Key_A and not ctrl:
            self.toggle_aspect_ratio(not self.aspect_ratio_locked)
            return

        # S -- toggle snap
        if key == Qt.Key.Key_S and not ctrl:
            self._toggle_snap()
            return

        # T -- pin / always on top
        if key == Qt.Key.Key_T and not ctrl:
            self.toggle_pin()
            return

        # Space -- bring to front
        if key == Qt.Key.Key_Space and not ctrl:
            self.raise_()
            self.activateWindow()
            return

        # Ctrl+G -- auto-fit grid
        if key == Qt.Key.Key_G and ctrl:
            self._request_snap_grid()
            return

        return super().keyPressEvent(event)

    def keyReleaseEvent(self, event):
        try:
            key = event.key()
            if (
                self._chord_modifier_key is not None
                and key == self._chord_modifier_key
            ):
                if not self._chord_saw_autorepeat:
                    # No autorepeat press has been seen yet, so this
                    # release CAN'T be X11 autorepeat alternation
                    # (alternation only happens after the initial
                    # autorepeat-delay window). Fire immediately for
                    # snappy tap-to-toggle.
                    self._chord_finalize_release()
                    return
                # Defer the release by a small window. If the next
                # autorepeat press (or any other keyboard activity)
                # arrives, _chord_cancel_release_timer fires and we
                # stay in chord mode. If the timer expires, the user
                # really did release the key and we finalize.
                self._chord_start_release_timer()
                return
        except Exception:
            pass
        return super().keyReleaseEvent(event)

    def _chord_start_release_timer(self):
        """Defer the chord-release decision by a short window so we
        can absorb X11 autorepeat alternation events. Any keyboard
        press while this is pending cancels it."""
        try:
            if self._chord_release_timer is None:
                self._chord_release_timer = QTimer(self)
                self._chord_release_timer.setSingleShot(True)
                self._chord_release_timer.timeout.connect(self._chord_finalize_release)
            self._chord_release_timer.start(self._CHORD_RELEASE_DEFER_MS)
        except Exception:
            # If the timer machinery fails for any reason, fall back
            # to ending the chord immediately so we never get stuck.
            self._chord_finalize_release()

    def _chord_cancel_release_timer(self):
        """Cancel any pending deferred release. Called whenever ANY
        key press happens while chord is active -- the user is still
        engaged with the keyboard, keep the chord alive."""
        try:
            if self._chord_release_timer is not None and self._chord_release_timer.isActive():
                self._chord_release_timer.stop()
        except Exception:
            pass

    def _chord_finalize_release(self):
        """End the active chord session because the modifier is
        believed to be released. If the user never pressed a
        secondary key, fire the original toggle action (single-tap
        semantic). Otherwise leave the final value visible briefly
        then clear."""
        if self._chord_modifier_key is None:
            return
        key = self._chord_modifier_key
        used_chord = bool(self._chord_used)
        self._chord_modifier_key = None
        self._chord_used = False
        self._chord_saw_autorepeat = False
        self._chord_cancel_release_timer()

        if not used_chord:
            # Pure tap -> clear the chord HINT first, then fire the
            # toggle so its own brief 'Motion boxes: ON' confirmation
            # has clean ownership of the overlay.
            self._debug_chord_lines = []
            try:
                if "chord" in (self._debug_overlay_auto_reasons or set()):
                    self._debug_overlay_auto_reasons.discard("chord")
            except Exception:
                pass
            try:
                self._refresh_debug_extra_lines()
            except Exception:
                pass
            tap_action = self._CHORD_TOGGLES_ON_TAP.get(key)
            if tap_action is not None:
                try:
                    tap_action(self)
                except Exception:
                    pass
        else:
            # Chord was used; leave the final value visible briefly
            # (~1s) so the user can read it, then restore the normal
            # aggregator output.
            self._chord_schedule_clear(1000, self._clear_chord_overlay)

    def _chord_end_no_toggle(self):
        """End the chord session WITHOUT firing the tap toggle. Used
        by Esc, by switching from one modifier to another, and when a
        non-dispatch key is pressed mid-chord."""
        if self._chord_modifier_key is None:
            return
        self._chord_modifier_key = None
        self._chord_used = False
        self._chord_saw_autorepeat = False
        self._chord_cancel_release_timer()
        self._debug_chord_lines = []
        try:
            self._refresh_debug_extra_lines()
        except Exception:
            pass
        try:
            self._chord_release_overlay()
        except Exception:
            pass

    # ---- Chord-key helpers ----

    def _chord_cancel_clear_timer(self):
        """Cancel any pending toggle/chord auto-clear timer.  Called
        whenever a new chord interaction starts so an in-flight clear
        doesn't kill the overlay mid-action."""
        try:
            if self._chord_clear_timer is not None:
                try:
                    self._chord_clear_timer.stop()
                except Exception:
                    pass
                try:
                    self._chord_clear_timer.deleteLater()
                except Exception:
                    pass
                self._chord_clear_timer = None
        except Exception:
            pass

    def _chord_schedule_clear(self, delay_ms: int, callback):
        """Schedule a clear callback, cancelling any prior pending one
        so only the most recent action's clear fires."""
        self._chord_cancel_clear_timer()
        try:
            t = QTimer(self)
            t.setSingleShot(True)
            t.timeout.connect(callback)
            t.start(int(max(50, delay_ms)))
            self._chord_clear_timer = t
        except Exception:
            try:
                callback()
            except Exception:
                pass

    def _chord_take_overlay(self):
        """Snapshot the current debug overlay state and force the
        overlay into focus mode for chord/toggle feedback.

        This BYPASSES the existing auto-show / user_opt_out machinery
        entirely -- the user is actively pressing a key, so they
        absolutely want to see the feedback, even if they previously
        turned the debug overlay off.  We restore their original
        state in `_chord_release_overlay()`.

        Re-assertion: even if the overlay is already owned, we
        FORCE show_debug=True / focus_mode=True on every call.
        Helpers like `set_overlay_settings()` (called when the chord
        toggles motion ON via a secondary key) will otherwise clobber
        show_debug back to False mid-chord, making the menu vanish.
        """
        # Always cancel a pending clear when (re-)taking ownership;
        # the new interaction supersedes any in-flight cleanup.
        self._chord_cancel_clear_timer()
        if not self._chord_owns_overlay:
            try:
                self._chord_prev_show_debug = bool(getattr(self.gl_widget, "show_debug", False))
                self._chord_prev_focus_mode = bool(getattr(self.gl_widget, "_debug_focus_mode", False))
                self._chord_prev_extra_lines = list(getattr(self.gl_widget, "_debug_extra_lines", None) or [])
            except Exception:
                self._chord_prev_show_debug = False
                self._chord_prev_focus_mode = False
                self._chord_prev_extra_lines = []
            self._chord_owns_overlay = True
        # ALWAYS force the overlay visible while we own it -- this
        # is the line that prevents the menu from vanishing when
        # secondary chord actions call set_overlay_settings().
        try:
            self.gl_widget.show_debug = True
            self.gl_widget._debug_focus_mode = True
        except Exception:
            pass

    def _chord_release_overlay(self):
        """Restore the debug overlay to whatever state it was in
        before the chord/toggle took over."""
        if not self._chord_owns_overlay:
            return
        try:
            if self._chord_prev_show_debug is not None:
                self.gl_widget.show_debug = bool(self._chord_prev_show_debug)
            if self._chord_prev_focus_mode is not None:
                self.gl_widget._debug_focus_mode = bool(self._chord_prev_focus_mode)
            if self._chord_prev_extra_lines is not None:
                try:
                    self.gl_widget.set_debug_extra_lines(self._chord_prev_extra_lines)
                except Exception:
                    pass
        except Exception:
            pass
        self._chord_prev_show_debug = None
        self._chord_prev_focus_mode = None
        self._chord_prev_extra_lines = None
        self._chord_owns_overlay = False
        try:
            self.gl_widget.update()
        except Exception:
            pass

    def _chord_push_lines(self, lines: List[str]):
        """Push chord/toggle text into the debug overlay (focus mode
        suppresses the standard FPS/Res base lines so the user only
        sees their interaction).

        Re-asserts show_debug / focus_mode on every push so that any
        external code that touched those flags (e.g. the implicit
        toggle_motion() called from inside _adjust_motion_setting)
        can't make the menu vanish mid-chord.
        """
        if self._chord_owns_overlay:
            try:
                self.gl_widget.show_debug = True
                self.gl_widget._debug_focus_mode = True
            except Exception:
                pass
        try:
            self.gl_widget.set_debug_extra_lines(list(lines or []))
        except Exception:
            pass
        try:
            self.gl_widget.update()
        except Exception:
            pass

    def _show_chord_hint(self, modifier_key: int):
        """Display the chord hint in the debug overlay -- the user
        sees what each secondary key does in plain English.

        Takes direct ownership of the overlay (forced visible, focus
        mode on) regardless of auto_reasons / user_opt_out.  Restores
        the prior state on release.

        Always pushes SOMETHING into the overlay so the user never
        sees an empty/wordless menu box -- if the modifier doesn't
        have a registered hint we fall back to a generic label.
        """
        hint = self._CHORD_HINT_FORMATS.get(modifier_key, "")
        if not hint:
            try:
                from PySide6.QtGui import QKeySequence as _QKS
                key_name = _QKS(modifier_key).toString() or "?"
            except Exception:
                key_name = "?"
            hint = f"{key_name} quick-adjust  (release {key_name} to keep)"
        self._debug_chord_lines = [hint]
        self._chord_take_overlay()
        self._chord_push_lines(self._debug_chord_lines)

    def _show_chord_value(self, label: str, value_text: str):
        """Display the current chord value (e.g. 'sensitivity 60') in
        the debug overlay so the user sees changes live.  Format is
        intentionally sparse so the overlay is laser-focused on the
        single value being changed -- the aggregator suppresses the
        normal model-info noise while chord lines are present.
        """
        big_value = (value_text or "").strip()
        # Always rebuild the hint header from the canonical format
        # so it isn't lost when the secondary action implicitly
        # called toggle_motion() / toggle_object_detection() (which
        # overwrites _debug_chord_lines with their own confirmation).
        hint_text = ""
        if self._chord_modifier_key is not None:
            hint_text = self._CHORD_HINT_FORMATS.get(self._chord_modifier_key, "")
        if not hint_text and self._debug_chord_lines:
            hint_text = self._debug_chord_lines[0]
        body = f">> {label}: {big_value}" if big_value else label
        new_lines = []
        if hint_text:
            new_lines.append(hint_text)
            new_lines.append("")
        new_lines.append(body)
        self._debug_chord_lines = new_lines
        # Make sure we own the overlay (no-op if already owned, but
        # re-asserts show_debug=True in case toggle_motion's call
        # to set_overlay_settings clobbered it).
        self._chord_take_overlay()
        self._chord_push_lines(self._debug_chord_lines)

    def _clear_chord_overlay(self):
        """Drop chord overlay lines and restore the debug overlay to
        whatever state it was in before the chord took over.
        """
        try:
            self._debug_chord_lines = []
            self._chord_release_overlay()
        except Exception:
            pass

    def _adjust_motion_setting(self, key: str, delta: int,
                                min_val: int, max_val: int) -> str:
        """Nudge a motion-detection setting by *delta*, clamped to
        [min_val, max_val].  Auto-enables motion overlay if it was off
        so the user sees the effect immediately.  Returns the
        resulting value formatted for overlay feedback.
        """
        try:
            settings = dict(self.gl_widget.motion_settings or {})
            cur = int(settings.get(key, 0) or 0)
            new = max(min_val, min(max_val, cur + delta))
            settings[key] = new
            self.gl_widget.motion_settings = settings
            # Auto-enable motion boxes so the user sees the new value
            # take effect immediately on the next frame.
            if not bool(self.motion_boxes_enabled):
                try:
                    self.toggle_motion()
                except Exception:
                    pass
            try:
                self.gl_widget.update()
            except Exception:
                pass
            return f"{new}"
        except Exception as exc:
            return f"err: {exc}"

    def _set_motion_style(self, style: str) -> str:
        """Set motion box style directly (used by M+1..7).  Auto-enables
        motion overlay so the change is visible immediately."""
        try:
            settings = dict(self.gl_widget.motion_settings or {})
            settings['style'] = str(style)
            self.gl_widget.motion_settings = settings
            if not bool(self.motion_boxes_enabled):
                try:
                    self.toggle_motion()
                except Exception:
                    pass
            try:
                self.gl_widget.update()
            except Exception:
                pass
            return str(style)
        except Exception as exc:
            return f"err: {exc}"

    def _cycle_motion_color(self) -> str:
        """Step through the preset color palette for motion boxes."""
        try:
            settings = dict(self.gl_widget.motion_settings or {})
            current = settings.get('color')
            cur_rgb = (255, 0, 0)
            try:
                if current is not None:
                    cur_rgb = (int(current.red()), int(current.green()), int(current.blue()))
            except Exception:
                pass
            names = [n for n, _ in self._PRESET_COLORS]
            rgbs = [rgb for _, rgb in self._PRESET_COLORS]
            try:
                idx = rgbs.index(cur_rgb)
                idx = (idx + 1) % len(rgbs)
            except ValueError:
                idx = 0
            new_name, new_rgb = names[idx], rgbs[idx]
            settings['color'] = QColor(*new_rgb)
            self.gl_widget.motion_settings = settings
            if not bool(self.motion_boxes_enabled):
                try:
                    self.toggle_motion()
                except Exception:
                    pass
            try:
                self.gl_widget.update()
            except Exception:
                pass
            return new_name
        except Exception as exc:
            return f"err: {exc}"

    def _cycle_motion_animation(self) -> str:
        """Step through the motion-box animation styles."""
        try:
            settings = dict(self.gl_widget.motion_settings or {})
            cur = str(settings.get('animation') or 'None')
            options = self._MOTION_ANIMATIONS
            try:
                idx = options.index(cur)
                idx = (idx + 1) % len(options)
            except ValueError:
                idx = 0
            new = options[idx]
            settings['animation'] = new
            self.gl_widget.motion_settings = settings
            if not bool(self.motion_boxes_enabled):
                try:
                    self.toggle_motion()
                except Exception:
                    pass
            try:
                self.gl_widget.update()
            except Exception:
                pass
            return new
        except Exception as exc:
            return f"err: {exc}"

    def _adjust_detection_setting(self, key: str, delta: int,
                                  min_val: int, max_val: int) -> str:
        """Nudge a detection-overlay setting (thickness/font_size/etc.)."""
        try:
            settings = dict(getattr(self.gl_widget, 'detection_settings', None) or {})
            cur = int(settings.get(key, 0) or 0)
            new = max(min_val, min(max_val, cur + delta))
            settings[key] = new
            self.gl_widget.detection_settings = settings
            try:
                self.gl_widget.update()
            except Exception:
                pass
            return f"{new}"
        except Exception as exc:
            return f"err: {exc}"

    def _set_detection_style(self, style: str) -> str:
        """Set detection box style directly (D+1..7)."""
        try:
            settings = dict(getattr(self.gl_widget, 'detection_settings', None) or {})
            settings['style'] = str(style)
            self.gl_widget.detection_settings = settings
            try:
                self.gl_widget.update()
            except Exception:
                pass
            return str(style)
        except Exception as exc:
            return f"err: {exc}"

    def _cycle_detection_color(self) -> str:
        """Step through the preset color palette for detection boxes."""
        try:
            settings = dict(getattr(self.gl_widget, 'detection_settings', None) or {})
            current = settings.get('color')
            cur_rgb = (0, 255, 255)
            try:
                if current is not None:
                    cur_rgb = (int(current.red()), int(current.green()), int(current.blue()))
            except Exception:
                pass
            names = [n for n, _ in self._PRESET_COLORS]
            rgbs = [rgb for _, rgb in self._PRESET_COLORS]
            try:
                idx = rgbs.index(cur_rgb)
                idx = (idx + 1) % len(rgbs)
            except ValueError:
                idx = 0
            new_name, new_rgb = names[idx], rgbs[idx]
            settings['color'] = QColor(*new_rgb)
            self.gl_widget.detection_settings = settings
            try:
                self.gl_widget.update()
            except Exception:
                pass
            return new_name
        except Exception as exc:
            return f"err: {exc}"

    def _cycle_detection_animation(self) -> str:
        """Step through the detection animation styles."""
        try:
            settings = dict(getattr(self.gl_widget, 'detection_settings', None) or {})
            cur = str(settings.get('animation') or 'None')
            options = self._DETECTION_ANIMATIONS
            try:
                idx = options.index(cur)
                idx = (idx + 1) % len(options)
            except ValueError:
                idx = 0
            new = options[idx]
            settings['animation'] = new
            self.gl_widget.detection_settings = settings
            try:
                self.gl_widget.update()
            except Exception:
                pass
            return new
        except Exception as exc:
            return f"err: {exc}"

    def _adjust_detection_conf(self, delta: float) -> str:
        """Adjust desktop YOLO detector min_confidence in [0.05, 0.95]."""
        try:
            cfg = self.desktop_detector_config
            cur = float(getattr(cfg, "min_confidence", 0.35) or 0.35)
            new = max(0.05, min(0.95, cur + delta))
            cfg.min_confidence = new
            try:
                if self._desktop_detector_worker is not None:
                    self._desktop_detector_worker.update_config(cfg)
            except Exception:
                pass
            try:
                self._update_debug_model_info()
            except Exception:
                pass
            return f"{new:.2f}"
        except Exception as exc:
            return f"err: {exc}"

    def _adjust_detection_fps(self, delta: int) -> str:
        """Adjust desktop YOLO detector fps_limit in [1, 30]."""
        try:
            cfg = self.desktop_detector_config
            cur = int(getattr(cfg, "fps_limit", 8) or 8)
            new = max(1, min(30, cur + delta))
            cfg.fps_limit = new
            try:
                if self._desktop_detector_worker is not None:
                    self._desktop_detector_worker.set_fps_limit(new)
            except Exception:
                pass
            try:
                self._update_debug_model_info()
            except Exception:
                pass
            return f"{new}"
        except Exception as exc:
            return f"err: {exc}"

    def show_context_menu(self, pos):
        from PySide6.QtWidgets import QMenu, QApplication
        from PySide6.QtGui import QAction

        menu = QMenu(self)
        simple = self._is_simple_controls()

        # ── Window controls (always visible) ──

        min_action = QAction("— Minimize", self)
        min_action.triggered.connect(self.showMinimized)
        menu.addAction(min_action)

        expand_label = "⤢ Restore" if self.isMaximized() else "⤢ Expand"
        expand_action = QAction(f"{expand_label}\tF", self)
        if self.isMaximized():
            expand_action.triggered.connect(self.showNormal)
        else:
            expand_action.triggered.connect(self.showMaximized)
        menu.addAction(expand_action)

        close_action = QAction("✕ Close", self)
        close_action.triggered.connect(self.close)
        menu.addAction(close_action)

        menu.addSeparator()

        front_action = QAction("Bring to Front\tSpace", self)
        front_action.triggered.connect(self.raise_)
        menu.addAction(front_action)

        back_action = QAction("Send to Back", self)
        back_action.triggered.connect(self.lower)
        menu.addAction(back_action)

        pin_action = QAction("📌 Pin in place (Always on Top)\tT", self)
        pin_action.setCheckable(True)
        pin_action.setChecked(self.is_pinned)
        pin_action.triggered.connect(self.toggle_pin)
        menu.addAction(pin_action)

        menu.addSeparator()

        # ── Patrol (always visible) ──

        app = QApplication.instance()
        patrol_active = bool(getattr(app, '_patrol_active', False)) if app else False

        patrol_menu = menu.addMenu("👁 Patrol")
        patrol_toggle_label = "⏹ Stop Patrol" if patrol_active else "▶ Start Patrol"
        patrol_toggle_action = QAction(f"{patrol_toggle_label}\tP", self)
        patrol_toggle_action.triggered.connect(self._request_patrol)
        patrol_menu.addAction(patrol_toggle_action)

        patrol_settings_action = QAction("⚙️ Patrol Settings...", self)
        patrol_settings_action.triggered.connect(self._open_patrol_settings)
        patrol_menu.addAction(patrol_settings_action)

        patrol_menu.addSeparator()

        # Per-camera patrol include toggles + order controls
        prefs_patrol = {}
        if app and hasattr(app, '_load_prefs'):
            prefs_patrol = app._load_prefs() or {}
        patrol_order = list(prefs_patrol.get("patrol_order") or [])
        all_cam_widgets = []
        if app and hasattr(app, '_collect_target_widgets'):
            all_cam_widgets = app._collect_target_widgets("camera")

        my_cid = str(getattr(self, "camera_id", "") or "")

        if all_cam_widgets:
            for cw in all_cam_widgets:
                cid = str(getattr(cw, "camera_id", "") or "")
                cname = str(getattr(cw, "camera_name", "") or cid)
                in_patrol = (not patrol_order) or (cid in patrol_order)
                ca = QAction(cname, self)
                ca.setCheckable(True)
                ca.setChecked(in_patrol)
                ca.triggered.connect(lambda checked, _cid=cid: self._patrol_toggle_camera(_cid, checked))
                patrol_menu.addAction(ca)

            if my_cid and patrol_order and my_cid in patrol_order:
                patrol_menu.addSeparator()
                up_action = QAction("▲ Move earlier", self)
                up_action.triggered.connect(lambda: self._patrol_move(my_cid, -1))
                patrol_menu.addAction(up_action)
                down_action = QAction("▼ Move later", self)
                down_action.triggered.connect(lambda: self._patrol_move(my_cid, 1))
                patrol_menu.addAction(down_action)

        # ── Grid & Snap (always visible) ──

        snap_menu = menu.addMenu("📐 Grid & Snap")

        snap_cfg = {}
        if app and hasattr(app, 'get_snap_config'):
            snap_cfg = app.get_snap_config()
        snap_on = bool(snap_cfg.get("enabled", False))

        snap_toggle = QAction("Snap to Grid\tS", self)
        snap_toggle.setCheckable(True)
        snap_toggle.setChecked(snap_on)
        snap_toggle.triggered.connect(lambda: self._toggle_snap())
        snap_menu.addAction(snap_toggle)

        guides_on = bool(snap_cfg.get("show_guides", True))
        guides_toggle = QAction("Show Snap Guides", self)
        guides_toggle.setCheckable(True)
        guides_toggle.setChecked(guides_on)
        guides_toggle.triggered.connect(lambda: self._toggle_snap_guides())
        snap_menu.addAction(guides_toggle)

        snap_menu.addSeparator()

        snap_arrange = QAction("Auto-fit Grid (seamless)\tCtrl+G", self)
        snap_arrange.triggered.connect(lambda: self._request_snap_grid())
        snap_menu.addAction(snap_arrange)

        snap_settings_action = QAction("⚙️ Grid Settings...", self)
        snap_settings_action.triggered.connect(self._open_grid_settings)
        snap_menu.addAction(snap_settings_action)

        menu.addSeparator()

        # ── Recording submenu (visible in both modes) ──

        rec_menu = menu.addMenu("⏺ Recording")
        is_recording = getattr(self, '_continuous_recording', False)
        rec_toggle_label = "⏹ Stop Recording" if is_recording else "⏺ Start Recording"
        rec_toggle_action = QAction(f"{rec_toggle_label}\tR", self)
        rec_toggle_action.triggered.connect(lambda: self._toggle_continuous_recording())
        rec_menu.addAction(rec_toggle_action)

        rec_settings_action = QAction("⚙ Recording Settings...", self)
        rec_settings_action.triggered.connect(lambda: self._open_recording_settings())
        rec_menu.addAction(rec_settings_action)

        global_rec_action = QAction("🌐 Global Recording Settings...", self)
        global_rec_action.triggered.connect(lambda: self._open_global_recording_settings())
        rec_menu.addAction(global_rec_action)

        rec_menu.addSeparator()

        playback_label = "⏹ Close Playback" if (self._playback_active or self._playback_loading) else "▶ Recording Playback"
        playback_action = QAction(playback_label, self)
        playback_action.triggered.connect(lambda: self.toggle_playback_overlay())
        rec_menu.addAction(playback_action)

        playback_folder_action = QAction("📂 Playback from folder...", self)
        playback_folder_action.triggered.connect(lambda: self._playback_choose_folder())
        rec_menu.addAction(playback_folder_action)

        # Sync playback submenu
        sync_menu = rec_menu.addMenu("📎 Sync Playback")
        has_playback = self._playback_active and self.playback_overlay
        if has_playback:
            cur_ts = self.playback_overlay._engine.position
            cur_speed = self.playback_overlay._engine.speed
        else:
            cur_ts = 0
            cur_speed = 1.0

        all_cams = []
        if app and hasattr(app, 'get_all_camera_ids_and_names'):
            all_cams = app.get_all_camera_ids_and_names()

        if not has_playback:
            no_pb = QAction("(start playback first)", self)
            no_pb.setEnabled(False)
            sync_menu.addAction(no_pb)
        elif not all_cams:
            no_cams = QAction("(no other cameras)", self)
            no_cams.setEnabled(False)
            sync_menu.addAction(no_cams)
        else:
            sync_all_action = QAction("▶ Sync ALL cameras", self)
            other_ids = [cid for cid, _ in all_cams if cid != self.camera_id]
            sync_all_action.setEnabled(bool(other_ids))
            sync_all_action.triggered.connect(
                lambda: self._request_sync_playback(other_ids, cur_ts, cur_speed))
            sync_menu.addAction(sync_all_action)
            sync_menu.addSeparator()

            for cid, cname in all_cams:
                if cid == self.camera_id:
                    continue
                act = QAction(cname, self)
                act.triggered.connect(
                    lambda checked=False, _cid=cid: self._request_sync_playback([_cid], cur_ts, cur_speed))
                sync_menu.addAction(act)

            if getattr(self, '_sync_group_id', None):
                sync_menu.addSeparator()
                unsync_action = QAction("⏏ Unsync all", self)
                unsync_action.triggered.connect(self._request_unsync)
                sync_menu.addAction(unsync_action)

        # ── PTZ (floating controller is the primary surface) ──

        ptz_menu = menu.addMenu("🎮 PTZ")
        ptz_open_action = QAction("Open PTZ controller", self)
        ptz_open_action.setToolTip("Open the floating PTZ controller for this camera")
        ptz_open_action.triggered.connect(self.open_ptz_controller)
        ptz_menu.addAction(ptz_open_action)

        ptz_creds_action = QAction("Credentials && protocol...", self)
        ptz_creds_action.setToolTip("Set the PTZ connection type, Tapo cloud password, and overrides")
        ptz_creds_action.triggered.connect(lambda: self.open_ptz_credentials_dialog())
        ptz_menu.addAction(ptz_creds_action)

        # ── Advanced submenus (hidden in simple mode) ──

        if not simple:
            menu.addSeparator()

            # ── Stream & Display ──
            stream_menu = menu.addMenu("📺 Stream & Display")

            quality_label = "ᴴᴰ Main Stream"
            if self._quality_pinned:
                quality_label += "  (pinned)"
            elif self._auto_quality_enabled:
                quality_label += "  (auto)"
            quality_action = QAction(quality_label, self)
            quality_action.setCheckable(True)
            quality_action.setChecked(self.stream_quality == 'medium')
            quality_action.triggered.connect(self.toggle_stream_quality)
            stream_menu.addAction(quality_action)

            auto_q_action = QAction("🔄 Auto Quality (size-aware)", self)
            auto_q_action.setCheckable(True)
            auto_q_action.setChecked(self._auto_quality_enabled and not self._quality_pinned)
            auto_q_action.triggered.connect(
                lambda checked: self.set_auto_quality_enabled(checked))
            stream_menu.addAction(auto_q_action)

            retest_sub_action = QAction("🔍 Re-test Sub Stream", self)
            retest_sub_action.triggered.connect(self._retest_substream)
            stream_menu.addAction(retest_sub_action)

            ar_action = QAction("🔒 Lock Aspect Ratio\tA", self)
            ar_action.setCheckable(True)
            ar_action.setChecked(self.aspect_ratio_locked)
            ar_action.triggered.connect(self.toggle_aspect_ratio)
            stream_menu.addAction(ar_action)

            debug_action = QAction("🐞 Show Debug Info", self)
            debug_action.setCheckable(True)
            debug_action.setChecked(self.debug_overlay_enabled)
            debug_action.triggered.connect(self.toggle_debug)
            stream_menu.addAction(debug_action)

            # ── Motion ──
            motion_menu = menu.addMenu("🏃 Motion")

            motion_action = QAction("Show Motion Boxes\tM", self)
            motion_action.setCheckable(True)
            motion_action.setChecked(self.motion_boxes_enabled)
            motion_action.triggered.connect(self.toggle_motion)
            motion_menu.addAction(motion_action)

            motion_settings_action = QAction("⚙️ Motion Settings...", self)
            motion_settings_action.triggered.connect(self.open_motion_settings)
            motion_menu.addAction(motion_settings_action)

            motion_watch_action = QAction("📸 Motion Watch...", self)
            motion_watch_action.triggered.connect(self.open_motion_watch_dialog)
            motion_menu.addAction(motion_watch_action)

            if self.motion_watch_active:
                stop_watch_action = QAction("⏹ Stop Motion Watch", self)
                stop_watch_action.triggered.connect(lambda: self.stop_motion_watch("stopped"))
                motion_menu.addAction(stop_watch_action)

            # ── Audio ──
            audio_menu = menu.addMenu("🎵 Audio")

            audio_overlay_action = QAction("Audio EQ (Overlay)", self)
            audio_overlay_action.setCheckable(True)
            audio_overlay_action.setChecked(self.audio_eq_overlay is not None)
            if self.audio_eq_window is not None:
                audio_overlay_action.setEnabled(False)
            audio_overlay_action.triggered.connect(lambda checked: self.toggle_audio_eq_overlay(checked))
            audio_menu.addAction(audio_overlay_action)

            audio_undock_action = QAction("📎 Undock Audio EQ", self)
            audio_undock_action.setCheckable(True)
            audio_undock_action.setChecked(self.audio_eq_window is not None)
            audio_undock_action.triggered.connect(lambda checked: self.toggle_audio_eq_undocked(checked))
            audio_menu.addAction(audio_undock_action)

            audio_play_action = QAction("▶/⏸ Toggle Audio Monitor", self)
            audio_play_action.triggered.connect(self._toggle_audio_play)
            audio_menu.addAction(audio_play_action)

            # ── Zones / Lines / Tags ──
            shapes_menu = menu.addMenu("📐 Zones / Lines / Tags")

            if self.gl_widget.selected_shapes:
                sel_ids = list(self.gl_widget.selected_shapes or [])
                sel_id = sel_ids[0] if sel_ids else ""
                edit_action = QAction("✏️ Edit Shape", self)
                edit_action.triggered.connect(lambda: self.edit_shape(sel_id))
                shapes_menu.addAction(edit_action)

                def _projection_label():
                    sid = sel_ids[0] if sel_ids else ""
                    sh = next((s for s in self.gl_widget.shapes if s.get('id') == sid), None)
                    kind = (sh or {}).get('kind') or 'shape'
                    return f"🪟 Project {kind.title()}"

                project_action = QAction(_projection_label(), self)
                project_action.triggered.connect(lambda: self.open_overlay_window(sel_ids))
                shapes_menu.addAction(project_action)

                if len(sel_ids) > 1:
                    def _project_each_selected():
                        for sid in sel_ids:
                            try:
                                self.open_overlay_window([sid])
                            except Exception:
                                pass
                    project_each_action = QAction(f"🪟 Project Each Selected ({len(sel_ids)})", self)
                    project_each_action.triggered.connect(_project_each_selected)
                    shapes_menu.addAction(project_each_action)

                shapes_menu.addSeparator()

            zone_action = QAction("Draw Zone", self)
            zone_action.triggered.connect(lambda: self.gl_widget.start_draw_mode('zone'))
            shapes_menu.addAction(zone_action)

            line_action = QAction("Draw Line", self)
            line_action.triggered.connect(lambda: self.gl_widget.start_draw_mode('line'))
            shapes_menu.addAction(line_action)

            tag_action = QAction("Add Tag", self)
            tag_action.triggered.connect(lambda: self.gl_widget.start_draw_mode('tag'))
            shapes_menu.addAction(tag_action)

            finish_action = QAction("Finish/Cancel Drawing", self)
            finish_action.triggered.connect(self.gl_widget.cancel_draw_mode)
            shapes_menu.addAction(finish_action)

            clear_shapes_action = QAction("Clear Shapes", self)
            clear_shapes_action.triggered.connect(self.gl_widget.clear_shapes)
            shapes_menu.addAction(clear_shapes_action)

            toggle_labels_action = QAction("Show Shape Labels", self)
            toggle_labels_action.setCheckable(True)
            toggle_labels_action.setChecked(self.gl_widget.show_shape_labels)
            toggle_labels_action.triggered.connect(lambda _: self._toggle_shape_labels())
            shapes_menu.addAction(toggle_labels_action)

            # ── Object Detection (AI) ──
            det_menu = menu.addMenu("🔍 Object Detection (AI)")

            self.obj_det_action = QAction("Enable Detection\tD", self)
            self.obj_det_action.setCheckable(True)
            self.obj_det_action.setChecked(bool(self.desktop_object_detection_enabled))
            self.obj_det_action.triggered.connect(self.toggle_object_detection)
            det_menu.addAction(self.obj_det_action)

            overlay_settings_action = QAction("🎨 Detection Overlay Settings...", self)
            overlay_settings_action.triggered.connect(self.open_detection_overlay_settings)
            det_menu.addAction(overlay_settings_action)

            desktop_settings_action = QAction("⚙️ Desktop Detection Settings...", self)
            desktop_settings_action.triggered.connect(self.open_desktop_detection_settings)
            det_menu.addAction(desktop_settings_action)

            tracking_settings_action = QAction("🧭 Tracking Settings...", self)
            tracking_settings_action.triggered.connect(self.open_desktop_tracking_settings)
            det_menu.addAction(tracking_settings_action)

            clear_saved_tracks_action = QAction("🧽 Clear saved tracked-object names (this camera)", self)
            clear_saved_tracks_action.triggered.connect(self.clear_tracked_object_overrides)
            det_menu.addAction(clear_saved_tracks_action)

            selected_track = getattr(self.gl_widget, "selected_track_id", None)
            selected_obj_action = QAction("✏️ Selected object settings...", self)
            selected_obj_action.setEnabled(selected_track is not None)
            selected_obj_action.triggered.connect(lambda: self.open_tracked_object_settings())
            det_menu.addAction(selected_obj_action)

            obj_settings_action = QAction("⚙️ Backend Detection Settings...", self)
            obj_settings_action.triggered.connect(self.open_object_detection_settings)
            det_menu.addAction(obj_settings_action)

            # ── Depth Map ──
            depth_menu = menu.addMenu("🗺 Depth Map")

            depth_action = QAction("Enable Depth Map (DepthAnythingV2)", self)
            depth_action.setCheckable(True)
            depth_action.setChecked(bool(self.depth_overlay_config.enabled))
            depth_action.triggered.connect(lambda checked: self.toggle_depth_overlay(checked))
            depth_menu.addAction(depth_action)

            depth_settings_action = QAction("⚙️ Depth Settings...", self)
            depth_settings_action.triggered.connect(self.open_depth_settings)
            depth_menu.addAction(depth_settings_action)

            # ── Advanced ──
            adv_menu = menu.addMenu("🛠 Advanced")

            settings_action = QAction("⚙️ Settings", self)
            adv_menu.addAction(settings_action)

            model_lib_action = QAction("📦 Model Library...", self)
            model_lib_action.triggered.connect(self.open_model_library)
            adv_menu.addAction(model_lib_action)

            camera_config_action = QAction("🛠 Camera Config...", self)
            camera_config_action.triggered.connect(self.open_camera_config)
            adv_menu.addAction(camera_config_action)

            dup_action = QAction("📄 Duplicate", self)
            adv_menu.addAction(dup_action)

            map_pin_action = QAction("📍 Pin to Map", self)
            adv_menu.addAction(map_pin_action)

        # ── Toggles (always last) ──

        menu.addSeparator()
        simple_toggle = QAction("Simple Controls", self)
        simple_toggle.setCheckable(True)
        simple_toggle.setChecked(simple)
        simple_toggle.triggered.connect(lambda checked: self._toggle_simple_controls(checked))
        menu.addAction(simple_toggle)

        hotkeys_on = self._hotkeys_enabled()
        hotkey_toggle = QAction("Keyboard Shortcuts", self)
        hotkey_toggle.setCheckable(True)
        hotkey_toggle.setChecked(hotkeys_on)
        hotkey_toggle.triggered.connect(lambda checked: self._toggle_hotkeys(checked))
        menu.addAction(hotkey_toggle)

        # Show keyboard shortcut reference (chord-key quick adjust)
        if not simple:
            shortcuts_help_action = QAction("Show Hotkey Reference", self)
            shortcuts_help_action.triggered.connect(self._show_hotkey_reference)
            menu.addAction(shortcuts_help_action)

        menu.exec(self.mapToGlobal(pos))

    def _show_hotkey_reference(self):
        """Display a brief reference of all camera-widget hotkeys,
        including the chord-key quick-adjust shortcuts.
        """
        from PySide6.QtWidgets import QMessageBox
        text = (
            "Single-tap toggles:\n"
            "  F / F11 - Fullscreen\n"
            "  R - Recording\n"
            "  P - Patrol mode\n"
            "  M - Motion boxes\n"
            "  D - Object detection\n"
            "  A - Lock aspect ratio\n"
            "  S - Snap to grid\n"
            "  T - Pin / always on top\n"
            "  Space - Bring to front\n"
            "  Ctrl+G - Auto-fit grid\n"
            "  Esc - Exit fullscreen\n\n"
            "Hold M (motion) + secondary key:\n"
            "  Left / Right    - Sensitivity\n"
            "  Up / Down       - Merge size\n"
            "  1..7            - Style "
            "(Box / Fill / Corners / Circle / Bracket / Underline / Crosshair)\n"
            "  C               - Cycle color\n"
            "  N               - Cycle animation "
            "(None / Pulse / Flash / Glitch / Rainbow)\n"
            "  Z / X           - Thickness -/+\n\n"
            "Hold D (detection) + secondary key:\n"
            "  Left / Right    - Confidence\n"
            "  Up / Down       - FPS limit\n"
            "  1..7            - Style\n"
            "  C               - Cycle color\n"
            "  N               - Cycle animation\n"
            "  Z / X           - Thickness -/+\n\n"
            "Tip: tap the modifier key alone for the toggle; "
            "hold + secondary key to adjust without opening menus. "
            "The current value appears in the debug overlay."
        )
        QMessageBox.information(self, "Camera Hotkeys", text)

    def _ptz_toggle_autopan_headless(self):
        """Auto-pan toggle even if overlay isn't currently visible."""
        if self._ptz_sweep_active:
            self.ptz_send("stop_sweep", {})
            return

        # Mirror overlay logic using current settings
        ap = (self.ptz_overlay_settings or {}).get("autopan", {}) or {}
        angle = int(ap.get("sweep_angle", 120))
        half = max(0.1, min(1.0, angle / 180.0))
        start_pan = -half
        end_pan = half
        start_dir = -1 if str(ap.get("start_direction", "right")).lower() == "left" else 1
        seconds_per_side = max(2.0, min(180.0, float(ap.get("seconds_per_side", 8.0))))
        span_norm = max(0.1, min(1.8, abs(end_pan - start_pan)))
        speed = max(0.002, min(0.95, (span_norm * 2.25) / seconds_per_side))
        edge_pause = max(0.0, min(15.0, float(ap.get("edge_pause_seconds", 0.5))))
        smooth = max(0.0, min(0.9, float(ap.get("smooth_ratio", 0.25))))
        tilt = max(-1.0, min(1.0, float(ap.get("tilt", 0.0))))
        # We don't know current sweep state without calling status; try start_sweep and let backend treat it idempotently.
        self.ptz_send(
            "start_sweep",
            {
                "start_pan": start_pan,
                "end_pan": end_pan,
                "tilt": tilt,
                "speed": speed,
                "seconds_per_side": seconds_per_side,
                "edge_pause_seconds": edge_pause,
                "dwell_time": edge_pause,
                "smooth_ratio": smooth,
                "start_direction": start_dir,
                "loop": True,
                "sweep_angle": angle,
            },
        )

    def closeEvent(self, event):
        self.running = False

        # Stop playback overlay to kill its background thread
        try:
            if self.playback_overlay is not None:
                self.playback_overlay.stop_playback()
        except Exception:
            pass

        # Disconnect from the global app frame signal so a closed widget can't keep receiving frames.
        try:
            from PySide6.QtWidgets import QApplication
            app = QApplication.instance()
            if app is not None and hasattr(app, "frame_signal"):
                try:
                    app.frame_signal.disconnect(self.receive_frame)
                except Exception:
                    pass
        except Exception:
            pass

        # Stop periodic timers (in case the widget isn't deleted immediately on close).
        for tname in ["timer", "offline_timer", "motion_watch_timer", "loading_timer"]:
            try:
                t = getattr(self, tname, None)
                if t is not None and hasattr(t, "isActive") and t.isActive():
                    t.stop()
            except Exception:
                pass

        # Stop backend detections SocketIO client (it runs its own daemon thread + wait loop).
        try:
            if getattr(self, "sio", None) is not None:
                try:
                    self.sio.disconnect()
                except Exception:
                    pass
        except Exception:
            pass

        # Release camera stream (on-demand) when widget closes.
        try:
            import asyncio
            cm = getattr(self, "camera_manager", None)
            loop = getattr(cm, "_loop", None) if cm else None
            if cm and loop and getattr(loop, "is_running", lambda: False)():
                asyncio.run_coroutine_threadsafe(cm.release_camera(self.camera_id), loop)
        except Exception:
            pass
        # Safety: stop PTZ motion/sweeps on close to avoid leaving a camera drifting unattended.
        try:
            self.ptz_send("stop_sweep", {}, silent=True)
            self.ptz_send("stop", {}, silent=True)
        except Exception:
            pass
        # Close undocked PTZ controller if open
        try:
            if self.ptz_controller_window:
                self.ptz_controller_window.close()
        except Exception:
            pass
        self.stop_motion_watch("closed")
        # Stop audio monitor / overlay
        try:
            self.toggle_audio_eq_overlay(False)
        except Exception:
            pass
        try:
            self.toggle_audio_eq_undocked(False)
        except Exception:
            pass
        # Stop depth worker
        try:
            if self._depth_worker is not None:
                self._depth_worker.stop()
                self._depth_worker.wait(1500)
        except Exception:
            pass
        # Stop desktop detector worker
        try:
            if self._desktop_detector_worker is not None:
                self._desktop_detector_worker.stop()
                self._desktop_detector_worker.wait(1500)
        except Exception:
            pass
        self._stop_loading()
        # Close any active overlay projection windows owned by this camera widget.
        try:
            for w in list((getattr(self, "overlay_windows", None) or {}).values()):
                try:
                    w.close()
                    w.deleteLater()
                except Exception:
                    pass
            try:
                (getattr(self, "overlay_windows", None) or {}).clear()
            except Exception:
                pass
        except Exception:
            pass
        super().closeEvent(event)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self.ptz_overlay:
            try:
                self.ptz_overlay.setGeometry(self.gl_widget.rect())
                self.ptz_overlay.raise_()
            except Exception:
                pass
        if self.audio_eq_overlay:
            try:
                self.audio_eq_overlay.setGeometry(self.gl_widget.rect())
                self.audio_eq_overlay.raise_()
            except Exception:
                pass
        if self.playback_overlay:
            try:
                self.playback_overlay.setGeometry(self.gl_widget.rect())
                self.playback_overlay.raise_()
            except Exception:
                pass
        self._schedule_auto_quality_eval()

    def changeEvent(self, event):
        try:
            from PySide6.QtCore import QEvent
            if event.type() == QEvent.Type.WindowStateChange:
                self._schedule_auto_quality_eval(delay_ms=200)
        except Exception:
            pass
        super().changeEvent(event)

    def _schedule_auto_quality_eval(self, delay_ms: int = 600):
        """Debounce repeated resize events.  After the widget settles for
        ~600ms, run the auto-quality policy.

        Always reschedules when the load shedder is forcing sub mode so
        the override can switch the stream regardless of user pin.
        Also reschedules when pinned/auto-disabled so a recent shed
        clearing can trigger restoration on the next eval.
        """
        if not self.camera_manager:
            return
        # Always run if shed force-sub is engaged (it bypasses pin).
        # Otherwise honor the auto-enabled / pin state in _evaluate.
        try:
            if self._auto_quality_timer is None:
                self._auto_quality_timer = QTimer(self)
                self._auto_quality_timer.setSingleShot(True)
                self._auto_quality_timer.timeout.connect(self._evaluate_auto_quality)
            self._auto_quality_timer.start(int(max(50, delay_ms)))
        except Exception:
            pass

    def _is_widget_fullscreen(self) -> bool:
        try:
            if self.isFullScreen():
                return True
            win = self.window()
            if win is not None and win is not self and win.isFullScreen():
                return True
        except Exception:
            pass
        return False

    def _evaluate_auto_quality(self):
        """Pick MAIN or SUB based on current widget size, fullscreen
        state, substream capability, and any active load-shed override.

        Precedence:
          1. ``_shed_force_sub`` from the load shedder wins over user pin
             (system protection trumps user preference temporarily).
          2. ``_quality_pinned`` from the user wins over auto policy.
          3. Auto policy uses geometry + sub capability.
        """
        if not self.camera_manager or not self.camera_id:
            return
        try:
            if not self.isVisible():
                return
        except Exception:
            return

        cfg = None
        try:
            cfg = self.camera_manager.cameras.get(self.camera_id)
        except Exception:
            cfg = None

        sub_capable = True
        if cfg is not None:
            sub_capable = (
                bool(getattr(cfg, "substream_rtsp_url", None))
                and getattr(cfg, "substream_capable", None) is not False
            )

        # 1. Load-shed override -- force sub when capable (otherwise stay
        #    on main; we can't conjure a sub stream that doesn't exist).
        if getattr(self, "_shed_force_sub", False):
            target = "low" if sub_capable else "medium"
            if target != self.stream_quality:
                self._auto_quality_target = target
                self._switch_stream_quality_internal(target)
            return

        # 2. User pin wins over auto.
        if self._quality_pinned or not self._auto_quality_enabled:
            return

        if self._is_widget_fullscreen():
            target = "medium"
        elif not sub_capable:
            target = "medium"
        else:
            cur_w = self.width()
            cur = self.stream_quality or "medium"
            if cur == "low":
                target = "medium" if cur_w >= self._auto_quality_to_main_w else "low"
            else:
                target = "low" if cur_w <= self._auto_quality_to_sub_w else "medium"

        if target == self._auto_quality_target and target == self.stream_quality:
            return
        self._auto_quality_target = target

        if target == self.stream_quality:
            return

        self._apply_stream_quality(target, pinned=False)

    def _apply_stream_quality(self, quality: str, pinned: bool):
        """Apply a stream quality change.  When *pinned* is True, the
        widget remembers the user's choice and stops auto-switching."""
        if quality not in ("low", "medium"):
            return
        if pinned:
            self._quality_pinned = True
        if quality == self.stream_quality:
            return
        self._switch_stream_quality_internal(quality)

    def _switch_stream_quality_internal(self, quality: str):
        """Issue the actual stream switch on a background thread.

        Used by both the user-driven `_apply_stream_quality` and the
        load shedder's force-sub path (which must NOT touch the user's
        `_quality_pinned` flag).
        """
        if quality not in ("low", "medium"):
            return
        if quality == self.stream_quality:
            return
        self.stream_quality = quality
        if not self.camera_manager:
            return
        import threading
        threading.Thread(
            target=self._switch_quality_async, args=(quality,), daemon=True,
        ).start()

    # ---- Auto-protection (load shedder) integration ----

    def apply_shed_level(self, level: int, throttles: Optional[Dict[str, int]] = None,
                        is_primary: bool = True):
        """Apply a load-shed level to this widget.

        Idempotent: safe to call with the same level repeatedly.  When
        the level differs from the previously-applied one, it (a)
        snapshots the original values into ``_shed_saved_state`` if not
        already saved, then (b) applies the per-level effects on top of
        those originals.

        Args:
            level: integer LoadLevel (0=NORMAL, 1=ELEVATED, 2=HIGH,
                3=CRITICAL, 4=EMERGENCY).
            throttles: dict from the shedder with paint_fps, motion_fps,
                detector_fps, depth_fps for the active level.
            is_primary: True if this widget is one of the user's primary
                cameras (kept connected at CRITICAL).  Non-primary
                widgets get released_camera() at CRITICAL+.
        """
        try:
            level = int(level or 0)
        except Exception:
            level = 0
        if level <= 0:
            self.clear_shed_state()
            return

        # Snapshot baseline ONCE on first entry into shed mode so
        # multiple level transitions don't compound losses.
        if not self._shed_saved_state:
            try:
                self._shed_saved_state = {
                    "paint_cap": int(getattr(self.gl_widget, "_shed_paint_fps_cap", 0) or 0),
                    "motion_cap": int(getattr(self.gl_widget, "_shed_motion_fps_cap", 0) or 0),
                    "detector_fps": int(getattr(self.desktop_detector_config, "fps_limit", 8) or 8),
                    "depth_fps": int(getattr(self.depth_overlay_config, "fps_limit", 12) or 12),
                    "depth_enabled": bool(getattr(self.depth_overlay_config, "enabled", False)),
                    "detector_enabled": bool(self.desktop_object_detection_enabled),
                }
            except Exception:
                self._shed_saved_state = {}

        throttles = throttles or {}
        prev_level = int(self._shed_active_level or 0)
        self._shed_active_level = level

        # Apply throttles to the GL widget (paint cap, motion cap)
        try:
            paint_fps = int(throttles.get("paint_fps", 0) or 0)
            motion_fps = int(throttles.get("motion_fps", 0) or 0)
            self.gl_widget._shed_paint_fps_cap = paint_fps
            self.gl_widget._shed_motion_fps_cap = motion_fps
        except Exception:
            pass

        # Force substream regardless of user pin while shed is active
        if not self._shed_force_sub:
            self._shed_force_sub = True
            try:
                self._schedule_auto_quality_eval(delay_ms=50)
            except Exception:
                pass

        # Worker FPS / suspension
        try:
            det_fps = int(throttles.get("detector_fps", -1))
            depth_fps = int(throttles.get("depth_fps", -1))

            # CRITICAL or worse: stop the workers entirely.
            if level >= 3 or det_fps == 0:
                self._shed_pause_detector_worker()
            elif det_fps > 0 and self._desktop_detector_worker is not None:
                try:
                    self._desktop_detector_worker.set_fps_limit(det_fps)
                except Exception:
                    pass

            if level >= 3 or depth_fps == 0:
                self._shed_pause_depth_worker()
            elif depth_fps > 0 and self._depth_worker is not None:
                try:
                    self._depth_worker.set_fps_limit(depth_fps)
                except Exception:
                    pass
        except Exception:
            pass

        # CRITICAL+: release non-primary widgets from the camera manager
        # so their RTSP threads stop entirely.  HIGH releases hidden
        # widgets but keeps visible ones connected.
        try:
            if level >= 3 and not is_primary and not self._shed_camera_released:
                self._shed_release_camera()
            elif level >= 2 and not bool(self.isVisible()) and not self._shed_camera_released:
                self._shed_release_camera()
        except Exception:
            pass

        # Push status into the per-camera debug overlay using the
        # existing auto-show plumbing -- respects user_opt_out so we
        # never override their explicit OFF choice.
        try:
            self._update_shed_debug_lines(level, throttles)
            self._auto_show_debug_overlay("load_shed")
        except Exception:
            pass

        if prev_level != level:
            try:
                logger.info(
                    "Camera %s shed level %d -> %d (paint=%s motion=%s det=%s depth=%s)",
                    self.camera_id, prev_level, level,
                    throttles.get("paint_fps"), throttles.get("motion_fps"),
                    throttles.get("detector_fps"), throttles.get("depth_fps"),
                )
            except Exception:
                pass

    def clear_shed_state(self):
        """Restore everything the shedder changed back to its baseline.

        Safe to call when the widget was never in shed mode (no-op).
        """
        if not self._shed_active_level and not self._shed_saved_state:
            return

        saved = dict(self._shed_saved_state or {})

        try:
            self.gl_widget._shed_paint_fps_cap = int(saved.get("paint_cap", 0) or 0)
            self.gl_widget._shed_motion_fps_cap = int(saved.get("motion_cap", 0) or 0)
        except Exception:
            pass

        # Restore worker fps caps
        try:
            if self._desktop_detector_worker is not None:
                self._desktop_detector_worker.set_fps_limit(
                    int(saved.get("detector_fps", 8) or 8)
                )
        except Exception:
            pass
        try:
            if self._depth_worker is not None:
                self._depth_worker.set_fps_limit(
                    int(saved.get("depth_fps", 12) or 12)
                )
        except Exception:
            pass

        # Re-acquire the camera if shedder released it
        try:
            if self._shed_camera_released:
                self._shed_reacquire_camera()
        except Exception:
            pass

        # Restart workers if they were running before shed mode kicked
        # in.  The user's enable flags (desktop_object_detection_enabled,
        # depth_overlay_config.enabled) were never modified by the
        # shedder, so calling the existing _ensure_*_worker methods is
        # safe and respects the current user state.
        try:
            if saved.get("detector_enabled") and not self._desktop_detector_worker:
                if hasattr(self, "_ensure_desktop_detector_worker"):
                    self._ensure_desktop_detector_worker()
        except Exception:
            pass
        try:
            if saved.get("depth_enabled") and not self._depth_worker:
                if hasattr(self, "_ensure_depth_worker"):
                    self._ensure_depth_worker()
        except Exception:
            pass

        # Drop substream force; let auto-quality re-evaluate naturally
        if self._shed_force_sub:
            self._shed_force_sub = False
            try:
                self._schedule_auto_quality_eval(delay_ms=200)
            except Exception:
                pass

        # Drop debug overlay reason; the existing logic at
        # _auto_show_debug_overlay handles the user_forced/user_opt_out
        # case correctly when we discard a reason.
        try:
            self._debug_load_shed_lines = []
            self._refresh_debug_extra_lines()
            if "load_shed" in (self._debug_overlay_auto_reasons or set()):
                self._debug_overlay_auto_reasons.discard("load_shed")
                # Hide overlay only if the user didn't explicitly turn
                # it on AND there are no other auto-reasons left.
                if (not bool(getattr(self, "_debug_overlay_user_forced", False))
                        and not bool(self._debug_overlay_auto_reasons)
                        and bool(self.debug_overlay_enabled)):
                    self.debug_overlay_enabled = False
                    self._apply_overlay_settings()
        except Exception:
            pass

        prev = self._shed_active_level
        self._shed_active_level = 0
        self._shed_saved_state = {}
        self._shed_workers_paused = False
        if prev:
            try:
                logger.info("Camera %s shed cleared (was level %d)", self.camera_id, prev)
            except Exception:
                pass

    def _shed_pause_detector_worker(self):
        """Stop the desktop YOLO worker without losing config so the
        load shedder can suspend it under heavy load."""
        try:
            if self._desktop_detector_worker is not None:
                try:
                    self._desktop_detector_worker.stop()
                    self._desktop_detector_worker.wait(1000)
                except Exception:
                    pass
                self._desktop_detector_worker = None
                self._shed_workers_paused = True
        except Exception:
            pass

    def _shed_pause_depth_worker(self):
        """Stop the depth worker without losing config."""
        try:
            if self._depth_worker is not None:
                try:
                    self._depth_worker.stop()
                    self._depth_worker.wait(1000)
                except Exception:
                    pass
                self._depth_worker = None
                self._shed_workers_paused = True
        except Exception:
            pass

    def _shed_release_camera(self):
        """Release this widget's reference to the camera so the capture
        thread can stop.  Only called for non-primary or hidden widgets.
        """
        try:
            cm = getattr(self, "camera_manager", None)
            loop = getattr(cm, "_loop", None) if cm else None
            if cm and loop and getattr(loop, "is_running", lambda: False)():
                import asyncio as _aio
                _aio.run_coroutine_threadsafe(
                    cm.release_camera(self.camera_id), loop,
                )
                self._shed_camera_released = True
        except Exception:
            pass

    def _shed_reacquire_camera(self):
        """Re-acquire the camera that was released by the shedder."""
        try:
            cm = getattr(self, "camera_manager", None)
            loop = getattr(cm, "_loop", None) if cm else None
            if cm and loop and getattr(loop, "is_running", lambda: False)():
                import asyncio as _aio
                _aio.run_coroutine_threadsafe(
                    cm.acquire_camera(self.camera_id), loop,
                )
            self._shed_camera_released = False
        except Exception:
            pass

    def _update_shed_debug_lines(self, level: int, throttles: Dict[str, int]):
        """Populate `_debug_load_shed_lines` so the debug overlay shows
        the current protection state (level + active throttles)."""
        try:
            label_map = {1: "ELEVATED", 2: "HIGH", 3: "CRITICAL", 4: "EMERGENCY"}
            label = label_map.get(int(level), str(level))
            paint = throttles.get("paint_fps")
            motion = throttles.get("motion_fps")
            det = throttles.get("detector_fps")
            depth = throttles.get("depth_fps")

            def _fmt(v):
                if v is None:
                    return "-"
                if int(v) == 0:
                    return "off"
                return f"{int(v)}fps"

            self._debug_load_shed_lines = [
                f"[Auto-Protect] {label}",
                f"  paint:{_fmt(paint)}  motion:{_fmt(motion)}  det:{_fmt(det)}  depth:{_fmt(depth)}",
            ]
            self._refresh_debug_extra_lines()
        except Exception:
            pass

    def get_shed_state(self) -> dict:
        """Snapshot of the widget's current shed state for the UI."""
        return {
            "level": int(self._shed_active_level or 0),
            "force_sub": bool(self._shed_force_sub),
            "workers_paused": bool(self._shed_workers_paused),
            "camera_released": bool(self._shed_camera_released),
        }

    @Slot(str, np.ndarray)
    def receive_frame(self, camera_id, frame):
        """Slot to receive frames from the main thread/signal."""
        # Fast guard: don't process frames once closed/hidden (prevents CPU climb on layout switches).
        if not bool(getattr(self, "running", True)):
            return
        try:
            if not self.isVisible():
                return
        except Exception:
            pass
        if camera_id == self.camera_id:
            # Gate live frames during any playback state (loading or active).
            # During loading, freeze the last live frame so the UI doesn't
            # flash between live and recorded content.
            if getattr(self, '_playback_loading', False):
                return
            if getattr(self, '_playback_active', False):
                eng = self.playback_overlay._engine if self.playback_overlay else None
                if eng:
                    # While the engine is actively playing, ALWAYS suppress
                    # live frames -- the engine is the sole frame source.
                    # Only allow live frames when the engine has stopped/paused
                    # AND the position is at the live edge (DVR live mode).
                    if eng.playing or not eng.is_at_live_edge:
                        return
            self.last_frame_time = time.time()
            if self.offline_label.isVisible():
                self.offline_label.hide()
            if self.loading_label.isVisible():
                self._stop_loading()
            self.gl_widget.update_frame(frame)

            # Feed clip ring buffer when clip recording is enabled
            try:
                if self.motion_watch_settings.get("clip_enabled"):
                    now = time.time()
                    interval = 1.0 / max(1.0, self._clip_ring_fps)
                    if now - self._clip_ring_last_push >= interval:
                        self._clip_ring_last_push = now
                        ok, jpg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
                        if ok:
                            self._clip_frame_ring.append((now, bytes(jpg)))
                        # If actively recording a clip, also push to the record buffer
                        if self._clip_recording and self._clip_record_buf is not None:
                            if now < self._clip_record_deadline:
                                self._clip_record_buf.append((now, bytes(jpg) if ok else None))
                            else:
                                self._finalize_clip_recording()
            except Exception:
                pass

            # Feed depth worker (async) if enabled
            if bool(self.depth_overlay_config.enabled):
                try:
                    self._ensure_depth_worker()
                    if self._depth_worker is not None:
                        self._depth_worker.submit_frame(frame)
                except Exception:
                    pass

            # Feed desktop detector worker (async) if enabled
            if bool(self.desktop_object_detection_enabled):
                try:
                    self._ensure_desktop_detector_worker()
                    if self._desktop_detector_worker is not None:
                        crop_frame = frame
                        meta = None
                        try:
                            crop_frame, meta = self._roi_crop_for_inference(frame)
                        except Exception:
                            crop_frame, meta = frame, None
                        self._desktop_detector_worker.submit_frame(crop_frame, meta=meta)
                except Exception:
                    pass
            
            # Update aspect ratio for window resizing
            if self.gl_widget.image and self.gl_widget.image.height() > 0:
                new_ar = self.gl_widget.image.width() / self.gl_widget.image.height()
                self.aspect_ratio = new_ar
                
                # Auto-resize window on first frame to match content
                if not self.has_set_initial_size and self.keep_aspect_ratio:
                    current_w = self.width()
                    # Maintain width, adjust height to match video AR
                    new_h = int(current_w / new_ar)
                    self.resize(current_w, new_h)
                    self.has_set_initial_size = True

    def _roi_crop_for_inference(self, frame: np.ndarray):
        """
        If enabled in detection settings, crop inference to the union of ROI shape bounds
        (zones/lines/tags + their proximity distances), then map detections back via meta offsets.
        """
        ds = getattr(self.gl_widget, "detection_settings", None) or {}
        if not bool(ds.get("roi_crop_inference", False)):
            return frame, None
        if frame is None:
            return frame, None
        h, w = frame.shape[0], frame.shape[1]
        if h <= 0 or w <= 0:
            return frame, None

        # Determine shapes considered for ROI (reuse ROI filter knobs)
        shapes = list(getattr(self.gl_widget, "shapes", None) or [])
        if not shapes:
            return frame, None

        selected_only = bool(ds.get("roi_selected_only", False))
        sel_ids = set(getattr(self.gl_widget, "selected_shapes", None) or [])
        if selected_only and sel_ids:
            shapes = [s for s in shapes if (s or {}).get("id") in sel_ids]

        roi_zone = bool(ds.get("roi_zone", True))
        roi_line = bool(ds.get("roi_line", True))
        roi_tag = bool(ds.get("roi_tag", True))
        roi_line_dist = float(ds.get("roi_line_dist", 0.02) or 0.02)
        roi_tag_dist = float(ds.get("roi_tag_dist", 0.03) or 0.03)
        roi_zone_margin = float(ds.get("roi_zone_margin", 0.0) or 0.0)

        roi_line_dist = max(0.0, min(0.5, roi_line_dist))
        roi_tag_dist = max(0.0, min(0.5, roi_tag_dist))
        roi_zone_margin = max(0.0, min(0.5, roi_zone_margin))

        minx, miny, maxx, maxy = 1.0, 1.0, 0.0, 0.0
        any_roi = False

        def _expand(x0: float, y0: float, x1: float, y1: float, pad: float):
            nonlocal minx, miny, maxx, maxy, any_roi
            x0 = max(0.0, min(1.0, x0 - pad))
            y0 = max(0.0, min(1.0, y0 - pad))
            x1 = max(0.0, min(1.0, x1 + pad))
            y1 = max(0.0, min(1.0, y1 + pad))
            minx = min(minx, x0)
            miny = min(miny, y0)
            maxx = max(maxx, x1)
            maxy = max(maxy, y1)
            any_roi = True

        for sh in shapes:
            if not isinstance(sh, dict):
                continue
            if not sh.get("enabled", True) or sh.get("hidden", False):
                continue
            kind = sh.get("kind")
            if kind == "zone" and roi_zone:
                pts = sh.get("pts") or []
                if isinstance(pts, list) and len(pts) >= 3:
                    xs = [float(p.get("x", 0.5)) for p in pts if isinstance(p, dict)]
                    ys = [float(p.get("y", 0.5)) for p in pts if isinstance(p, dict)]
                    if xs and ys:
                        _expand(min(xs), min(ys), max(xs), max(ys), pad=roi_zone_margin)
            elif kind == "line" and roi_line:
                p1 = sh.get("p1") or {}
                p2 = sh.get("p2") or {}
                try:
                    x0 = min(float(p1.get("x", 0.5)), float(p2.get("x", 0.5)))
                    y0 = min(float(p1.get("y", 0.5)), float(p2.get("y", 0.5)))
                    x1 = max(float(p1.get("x", 0.5)), float(p2.get("x", 0.5)))
                    y1 = max(float(p1.get("y", 0.5)), float(p2.get("y", 0.5)))
                    _expand(x0, y0, x1, y1, pad=roi_line_dist)
                except Exception:
                    pass
            elif kind == "tag" and roi_tag:
                a = sh.get("anchor") or {}
                try:
                    cx = float(a.get("x", 0.5))
                    cy = float(a.get("y", 0.5))
                    _expand(cx, cy, cx, cy, pad=roi_tag_dist)
                except Exception:
                    pass

        if not any_roi or maxx <= minx or maxy <= miny:
            return frame, None

        x0 = int(max(0, min(w - 1, round(minx * w))))
        y0 = int(max(0, min(h - 1, round(miny * h))))
        x1 = int(max(1, min(w, round(maxx * w))))
        y1 = int(max(1, min(h, round(maxy * h))))
        # Ensure a minimum crop size to avoid degenerate crops
        min_px = 64
        if (x1 - x0) < min_px:
            cx = (x0 + x1) // 2
            x0 = max(0, cx - min_px // 2)
            x1 = min(w, x0 + min_px)
        if (y1 - y0) < min_px:
            cy = (y0 + y1) // 2
            y0 = max(0, cy - min_px // 2)
            y1 = min(h, y0 + min_px)

        cropped = frame[y0:y1, x0:x1]
        meta = {
            "offset_x": int(x0),
            "offset_y": int(y0),
            "roi_crop_rect": [int(x0), int(y0), int(x1 - x0), int(y1 - y0)],
            "orig_size": [int(w), int(h)],
        }
        return cropped, meta

    def _update_offline_state(self):
        """Show an unobtrusive overlay if no frames have been seen recently."""
        if getattr(self, '_playback_active', False) or getattr(self, '_playback_loading', False):
            if self.offline_label.isVisible():
                self.offline_label.hide()
            return
        now = time.time()
        stale = (self.last_frame_time is None) or (now - self.last_frame_time > 8)
        if stale and not self.offline_label.isVisible():
            name = self.camera_name or self.camera_id
            ip = self.camera_ip or "unknown IP"
            self.offline_label.setText(f"{name} ({ip}) offline or unreachable")
            self.offline_label.show()
        elif not stale and self.offline_label.isVisible():
            self.offline_label.hide()
