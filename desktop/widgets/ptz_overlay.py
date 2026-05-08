"""
PTZ overlay widgets

Provides:
- `default_ptz_overlay_settings()` / `load_ptz_overlay_settings()` /
  `save_ptz_overlay_settings()` for per-camera HUD preferences (look,
  control speeds, auto-pan, brand override).
- `PTZOverlaySettingsDialog`: small popover (Look / Controls / Auto Pan
  / Camera) — no connection / IP / port / credential fields. The
  backend infers everything else from the camera record.
- `PTZTapoCloudPasswordPrompt`: tiny modal that pops up only when the
  backend probe says a Tapo cloud password is needed. Defaults to
  session-only memory; the user can opt in to persistent storage with
  a visible warning.
- `PTZOverlayWidget`: translucent HUD with D-pad, zoom, home, presets,
  auto-pan, stop and gear (settings).
- `PTZControllerWindow`: floating window wrapper (the primary PTZ
  surface; the docked overlay path has been retired).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

from PySide6.QtCore import QPoint, Qt, QTimer, Signal
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QCheckBox,
    QColorDialog,
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QSlider,
    QSpinBox,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from desktop.utils.qt_helpers import KnoxnetStyle
from desktop.widgets.base import BaseDesktopWidget


# ---------------------------------------------------------------------------- #
# Settings persistence
# ---------------------------------------------------------------------------- #

def _ptz_settings_path() -> Path:
    """Persist under the per-user data dir (resolves to <repo>/data in dev)."""
    from core.paths import get_data_dir
    return get_data_dir() / "desktop_ptz_overlay_settings.json"


PTZ_OVERLAY_SETTINGS_PATH = _ptz_settings_path()


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def _as_float(v: Any, default: float) -> float:
    try:
        return float(v)
    except Exception:
        return default


def _as_int(v: Any, default: int) -> int:
    try:
        return int(v)
    except Exception:
        return default


def default_ptz_overlay_settings() -> Dict[str, Any]:
    return {
        "ui": {
            "opacity": 0.35,
            "accent_color": "#24D1FF",
            "bg_color": "#0b1220",
            "button_size": 44,
            "show_labels": False,
            "pos_norm": [0.5, 0.5],
        },
        # Brand override is the only "connection" thing the user touches;
        # anything truly required (camera IP / username / password) lives
        # in the camera record. The Tapo cloud password lives in the
        # backend `ptz_credentials` store, never on disk here.
        "connection": {
            "brand_override": "auto",  # auto|tapo|onvif|hikvision|dahua|axis|generic
        },
        "control": {
            "pan_speed": 0.5,
            "tilt_speed": 0.5,
            "zoom_speed": 0.3,
            "duration": 0.5,
        },
        "autopan": {
            "sweep_angle": 120,
            "seconds_per_side": 8.0,
            "edge_pause_seconds": 0.5,
            "smooth_ratio": 0.25,
            "start_direction": "right",
            "tilt": 0.0,
        },
    }


def _migrate_legacy_connection(camera_id: str, saved: Dict[str, Any]) -> Dict[str, Any]:
    """
    Old settings persisted IP/user/password/cloud-pw under `connection.*`.
    Move any cloud password into the session credential store and strip
    the rest so we never write secrets to this file again.
    """
    conn = dict(saved.get("connection") or {})
    legacy_cloud = (conn.pop("tapo_cloud_password", "") or "").strip()
    if legacy_cloud:
        try:
            from core import ptz_credentials
            ptz_credentials.set(camera_id, "tapo_cloud_password", legacy_cloud, persist=False)
        except Exception:
            pass
    for legacy_key in (
        "protocol", "ip_address", "onvif_port", "username", "password",
        "custom_ptz_url", "camera_brand", "camera_model", "use_pytapo",
        "cloud_password", "tapoCloudPassword",
    ):
        conn.pop(legacy_key, None)
    if "brand_override" not in conn:
        conn["brand_override"] = "auto"
    saved = dict(saved)
    saved["connection"] = conn
    return saved


def load_ptz_overlay_settings(camera_id: str) -> Dict[str, Any]:
    base = default_ptz_overlay_settings()
    needs_rewrite = False
    try:
        if PTZ_OVERLAY_SETTINGS_PATH.exists():
            data = json.loads(PTZ_OVERLAY_SETTINGS_PATH.read_text())
            if isinstance(data, dict):
                saved = data.get(camera_id) or data.get("default") or {}
                if isinstance(saved, dict):
                    cleaned = _migrate_legacy_connection(camera_id, saved)
                    if cleaned != saved:
                        needs_rewrite = True
                        data[camera_id] = cleaned
                        saved = cleaned
                    for key in ("ui", "connection", "control", "autopan"):
                        if isinstance(saved.get(key), dict):
                            base[key].update(saved[key])
                if needs_rewrite:
                    try:
                        PTZ_OVERLAY_SETTINGS_PATH.write_text(json.dumps(data, indent=2))
                    except Exception:
                        pass
    except Exception:
        pass
    return base


def save_ptz_overlay_settings(camera_id: str, settings: Dict[str, Any]) -> None:
    try:
        PTZ_OVERLAY_SETTINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
        existing: Dict[str, Any] = {}
        if PTZ_OVERLAY_SETTINGS_PATH.exists():
            try:
                existing = json.loads(PTZ_OVERLAY_SETTINGS_PATH.read_text())
                if not isinstance(existing, dict):
                    existing = {}
            except Exception:
                existing = {}
        existing[camera_id] = settings
        PTZ_OVERLAY_SETTINGS_PATH.write_text(json.dumps(existing, indent=2))
    except Exception:
        pass


# ---------------------------------------------------------------------------- #
# Tapo cloud password prompt
# ---------------------------------------------------------------------------- #

BRAND_CHOICES = [
    "auto", "tapo", "onvif", "hikvision", "dahua", "axis",
    "amcrest", "foscam", "ubiquiti", "generic",
]


class PTZCredentialsDialog(QDialog):
    """
    Always-accessible PTZ credentials & protocol dialog.

    The user can:
      - Pick the connection protocol explicitly: Auto / Tapo / ONVIF / Generic CGI.
        (Defaults to Auto, matching the manager's auto-detect.)
      - Set protocol-specific credentials (cloud password for Tapo, local
        username/password for any protocol, ONVIF port for ONVIF, custom URL
        template for Generic CGI).
      - Choose to use credentials once (session memory) or persist to
        `data/ptz_credentials.json` (chmod 0600) with a visible risk note.

    Result accessors:
      `dlg.result_payload` (Dict[str, str])  - keys to write to ptz_credentials
      `dlg.persist` (bool)                   - True for "Save & use"
      `dlg.cleared` (bool)                   - True if user clicked "Clear all"

    Stored credential keys:
        protocol_override            : 'auto' | 'tapo' | 'onvif' | 'generic'
        tapo_cloud_password          : str (Tapo only)
        tapo_local_username          : str (optional, Tapo override)
        tapo_local_password          : str (optional, Tapo override)
        onvif_port                   : str (optional, ONVIF override)
    """

    PROTOCOL_CHOICES = [
        ("auto", "Auto-detect (recommended)"),
        ("tapo", "TP-Link Tapo (pytapo)"),
        ("onvif", "ONVIF"),
        ("generic", "Generic / CGI"),
    ]

    def __init__(
        self,
        camera_label: str,
        parent=None,
        *,
        default_username: str = "admin",
        default_password: str = "",
        existing: Optional[Dict[str, str]] = None,
        message: Optional[str] = None,
        is_locked: bool = False,
        test_callback=None,
    ):
        super().__init__(parent)
        self.setWindowTitle(f"PTZ Credentials - {camera_label}")
        self.setWindowFlags(Qt.WindowType.Tool | Qt.WindowType.WindowStaysOnTopHint)
        self.setMinimumWidth(480)

        self.result_payload: Dict[str, str] = {}
        self.persist: bool = False
        self.cleared: bool = False
        self._default_username = default_username or "admin"
        self._default_password = default_password or ""
        self._test_callback = test_callback
        existing = dict(existing or {})

        root = QVBoxLayout(self)

        # Banner: contextual status (locked / auth failed / informational)
        if message:
            banner = QLabel(message)
            banner.setWordWrap(True)
            color = "#f87171" if is_locked else "#fbbf24"
            banner.setStyleSheet(
                f"padding: 8px; border-radius: 6px; "
                f"background: rgba(127,29,29,90); color: {color}; font-weight: 600;"
            )
            root.addWidget(banner)

        intro = QLabel(
            f"Configure how Knoxnet talks PTZ to <b>{camera_label}</b>. "
            "Knoxnet auto-detects the protocol from the camera record by default. "
            "Override here if auto-detect is wrong, or if you need to supply a "
            "Tapo cloud-account password."
        )
        intro.setWordWrap(True)
        root.addWidget(intro)

        # Protocol selector
        proto_form = QFormLayout()
        self.protocol_combo = QComboBox()
        for value, label in self.PROTOCOL_CHOICES:
            self.protocol_combo.addItem(label, value)
        cur_proto = (existing.get("protocol_override") or "auto").lower()
        for i in range(self.protocol_combo.count()):
            if self.protocol_combo.itemData(i) == cur_proto:
                self.protocol_combo.setCurrentIndex(i)
                break
        self.protocol_combo.currentIndexChanged.connect(self._update_visibility)
        proto_form.addRow("Connection type", self.protocol_combo)
        root.addLayout(proto_form)

        # Tapo section
        self._tapo_box = QWidget()
        tapo_form = QFormLayout(self._tapo_box)
        tapo_form.setContentsMargins(0, 0, 0, 0)
        self.tapo_cloud_pw = QLineEdit(existing.get("tapo_cloud_password", ""))
        self.tapo_cloud_pw.setEchoMode(QLineEdit.EchoMode.Password)
        self.tapo_cloud_pw.setPlaceholderText("TP-Link cloud-account password (Tapo phone app)")
        tapo_form.addRow("Cloud password", self.tapo_cloud_pw)

        self.tapo_local_user = QLineEdit(
            existing.get("tapo_local_username") or self._default_username
        )
        self.tapo_local_user.setPlaceholderText("admin")
        tapo_form.addRow("Local username", self.tapo_local_user)

        self.tapo_local_pw = QLineEdit(existing.get("tapo_local_password") or self._default_password)
        self.tapo_local_pw.setEchoMode(QLineEdit.EchoMode.Password)
        self.tapo_local_pw.setPlaceholderText("Camera local password (often same as RTSP)")
        tapo_form.addRow("Local password", self.tapo_local_pw)

        tapo_hint = QLabel(
            "How Tapo PTZ login works (per the official pytapo guidance):<br>"
            " &bull; <b>Camera Account</b> first: the user/password you set in "
            "the Tapo phone app under <i>Settings &rarr; Advanced Settings "
            "&rarr; Camera Account</i>.<br>"
            " &bull; <b>Admin fallback</b>: if the above fails, "
            "<code>admin</code> + your <i>TP-Link cloud account</i> password "
            "(the one you use to log into the Tapo phone app).<br>"
            "Knoxnet auto-tries both (capped at 2 attempts to avoid Tapo's "
            "5-attempt-per-hour lockout). You must also enable "
            "<i>Me &rarr; Third-Party Services &rarr; Third-Party "
            "Compatibility</i> in the Tapo app."
        )
        tapo_hint.setWordWrap(True)
        tapo_hint.setStyleSheet("color: #94a3b8;")
        tapo_form.addRow(tapo_hint)
        root.addWidget(self._tapo_box)

        # ONVIF section
        self._onvif_box = QWidget()
        onvif_form = QFormLayout(self._onvif_box)
        onvif_form.setContentsMargins(0, 0, 0, 0)
        self.onvif_port = QLineEdit(existing.get("onvif_port", ""))
        self.onvif_port.setPlaceholderText("80 (auto-probes 80/8080/2020/8000)")
        onvif_form.addRow("ONVIF port", self.onvif_port)
        onvif_hint = QLabel(
            "Username and password come from the camera record. ONVIF probes "
            "the supplied port first, then 80/8080/2020/8000."
        )
        onvif_hint.setWordWrap(True)
        onvif_hint.setStyleSheet("color: #94a3b8;")
        onvif_form.addRow(onvif_hint)
        root.addWidget(self._onvif_box)

        # Generic / CGI section
        self._generic_box = QWidget()
        generic_form = QFormLayout(self._generic_box)
        generic_form.setContentsMargins(0, 0, 0, 0)
        generic_hint = QLabel(
            "Generic / CGI tries Hikvision, Dahua, Axis and Amcrest URL "
            "patterns using the camera's stored credentials. To override the "
            "URL pattern, set 'PTZ URL' in the camera configuration screen "
            "(uses {CMD} and {SPEED} placeholders)."
        )
        generic_hint.setWordWrap(True)
        generic_hint.setStyleSheet("color: #94a3b8;")
        generic_form.addRow(generic_hint)
        root.addWidget(self._generic_box)

        # Persistence + warning
        self.remember = QCheckBox("Remember on this device")
        self.remember.setChecked(False)
        root.addWidget(self.remember)

        self.warning = QLabel(
            "Risk: when remembered, credentials are written unencrypted to "
            "<code>data/ptz_credentials.json</code> (file mode 0600). Anyone "
            "with file-system access to this machine can read them."
        )
        self.warning.setStyleSheet("color: #f87171;")
        self.warning.setWordWrap(True)
        self.warning.setVisible(False)
        root.addWidget(self.warning)
        self.remember.toggled.connect(self.warning.setVisible)

        # Test result line
        self.test_status = QLabel("")
        self.test_status.setWordWrap(True)
        self.test_status.setStyleSheet("color: #94a3b8;")
        root.addWidget(self.test_status)

        # Buttons
        button_row = QHBoxLayout()
        clear_btn = QPushButton("Clear stored")
        clear_btn.setToolTip("Forget all stored credentials for this camera (session and disk)")
        clear_btn.clicked.connect(self._on_clear)
        self.test_btn = QPushButton("Test connection")
        self.test_btn.setToolTip(
            "Try the entered credentials WITHOUT saving them. "
            "Each test counts as up to 2 attempts toward Tapo's lockout."
        )
        self.test_btn.clicked.connect(self._on_test)
        if self._test_callback is None:
            self.test_btn.setEnabled(False)
        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)
        use_once_btn = QPushButton("Use once")
        use_once_btn.setDefault(True)
        use_once_btn.clicked.connect(self._accept_once)
        save_btn = QPushButton("Save && use")
        save_btn.clicked.connect(self._accept_save)
        button_row.addWidget(clear_btn)
        button_row.addWidget(self.test_btn)
        button_row.addStretch()
        button_row.addWidget(cancel_btn)
        button_row.addWidget(use_once_btn)
        button_row.addWidget(save_btn)
        root.addLayout(button_row)

        self.setStyleSheet(f"QDialog {{ background: {KnoxnetStyle.BG_DARK}; }}")
        self._update_visibility()

    def _selected_protocol(self) -> str:
        return str(self.protocol_combo.currentData() or "auto").lower()

    def _update_visibility(self) -> None:
        proto = self._selected_protocol()
        # Tapo fields visible for both 'auto' (so the user can pre-fill) and 'tapo'
        self._tapo_box.setVisible(proto in ("auto", "tapo"))
        self._onvif_box.setVisible(proto in ("auto", "onvif"))
        self._generic_box.setVisible(proto in ("auto", "generic"))

    def _build_payload(self) -> Dict[str, str]:
        payload: Dict[str, str] = {}
        proto = self._selected_protocol()
        payload["protocol_override"] = proto

        cloud = self.tapo_cloud_pw.text().strip()
        if cloud:
            payload["tapo_cloud_password"] = cloud

        local_user = self.tapo_local_user.text().strip()
        if local_user and local_user != self._default_username:
            payload["tapo_local_username"] = local_user

        local_pw = self.tapo_local_pw.text()
        if local_pw and local_pw != self._default_password:
            payload["tapo_local_password"] = local_pw

        port = self.onvif_port.text().strip()
        if port and port.isdigit():
            payload["onvif_port"] = port

        return payload

    def _validate(self) -> bool:
        proto = self._selected_protocol()
        if proto == "tapo" and not self.tapo_cloud_pw.text().strip():
            QMessageBox.warning(
                self, "Cloud password required",
                "Tapo PTZ needs the TP-Link cloud-account password to authenticate.",
            )
            return False
        return True

    def _accept_once(self) -> None:
        if not self._validate():
            return
        self.result_payload = self._build_payload()
        self.persist = bool(self.remember.isChecked())
        self.accept()

    def _accept_save(self) -> None:
        if not self._validate():
            return
        self.result_payload = self._build_payload()
        self.persist = True
        self.accept()

    def _on_test(self) -> None:
        if not self._test_callback:
            return
        if not self._validate():
            return
        self.test_btn.setEnabled(False)
        self.test_status.setText("Testing...")
        self.test_status.setStyleSheet("color: #fbbf24;")

        try:
            payload = self._build_payload()
            ok, message = bool(False), "Test failed"
            try:
                ok, message = self._test_callback(payload)
            except Exception as exc:
                ok, message = False, f"Test error: {exc}"
            color = "#22c55e" if ok else "#f87171"
            prefix = "OK: " if ok else "FAIL: "
            self.test_status.setText(prefix + (message or ""))
            self.test_status.setStyleSheet(f"color: {color};")
        finally:
            self.test_btn.setEnabled(True)

    def _on_clear(self) -> None:
        ack = QMessageBox.question(
            self, "Clear stored credentials",
            "Forget all stored PTZ credentials for this camera (session + disk)?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if ack != QMessageBox.StandardButton.Yes:
            return
        self.cleared = True
        self.result_payload = {}
        self.accept()


# Back-compat alias for callers that imported the old name.
PTZTapoCloudPasswordPrompt = PTZCredentialsDialog


# ---------------------------------------------------------------------------- #
# Slim settings dialog (Look / Controls / Auto Pan / Camera)
# ---------------------------------------------------------------------------- #

class PTZOverlaySettingsDialog(QDialog):
    """Compact settings dialog for the PTZ HUD — no connection details."""

    def __init__(self, camera_label: str, initial: Dict[str, Any], parent=None):
        super().__init__(parent)
        self.setWindowTitle(f"PTZ Settings - {camera_label}")
        self.setWindowFlags(Qt.WindowType.Tool | Qt.WindowType.WindowStaysOnTopHint)
        self._initial = initial or default_ptz_overlay_settings()
        self.result_settings: Dict[str, Any] = json.loads(json.dumps(self._initial))

        root = QVBoxLayout(self)
        self.tabs = QTabWidget()
        root.addWidget(self.tabs)

        self._build_look_tab()
        self._build_controls_tab()
        self._build_autopan_tab()
        self._build_camera_tab()

        row = QHBoxLayout()
        row.addStretch()
        save_btn = QPushButton("Save")
        save_btn.setDefault(True)
        save_btn.clicked.connect(self._accept)
        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)
        row.addWidget(save_btn)
        row.addWidget(cancel_btn)
        root.addLayout(row)

        self.setStyleSheet(f"QDialog {{ background: {KnoxnetStyle.BG_DARK}; }}")

    # ---- tabs ----

    def _build_look_tab(self) -> None:
        tab = QWidget()
        form = QFormLayout(tab)
        ui_init = self._initial.get("ui", {})

        self.ui_opacity = QSlider(Qt.Orientation.Horizontal)
        self.ui_opacity.setRange(10, 90)
        self.ui_opacity.setValue(int(_clamp(_as_float(ui_init.get("opacity"), 0.35), 0.1, 0.9) * 100))
        form.addRow("Overlay opacity", self.ui_opacity)

        self.ui_btn_size = QSpinBox()
        self.ui_btn_size.setRange(28, 80)
        self.ui_btn_size.setValue(_as_int(ui_init.get("button_size"), 44))
        form.addRow("Button size", self.ui_btn_size)

        self.ui_show_labels = QCheckBox("Show labels on buttons")
        self.ui_show_labels.setChecked(bool(ui_init.get("show_labels", False)))
        form.addRow(self.ui_show_labels)

        self._accent = QColor(str(ui_init.get("accent_color", "#24D1FF")))
        self.ui_accent_btn = QPushButton("Pick accent color")
        self.ui_accent_btn.clicked.connect(self._pick_accent)
        self._sync_color_btn(self.ui_accent_btn, self._accent)
        form.addRow("Accent", self.ui_accent_btn)

        self._bg = QColor(str(ui_init.get("bg_color", "#0b1220")))
        self.ui_bg_btn = QPushButton("Pick background color")
        self.ui_bg_btn.clicked.connect(self._pick_bg)
        self._sync_color_btn(self.ui_bg_btn, self._bg)
        form.addRow("Background", self.ui_bg_btn)

        self.tabs.addTab(tab, "Look")

    def _build_controls_tab(self) -> None:
        tab = QWidget()
        form = QFormLayout(tab)
        ctl_init = self._initial.get("control", {})

        self.ctrl_pan = QDoubleSpinBox()
        self.ctrl_pan.setRange(0.05, 1.0)
        self.ctrl_pan.setSingleStep(0.05)
        self.ctrl_pan.setValue(_clamp(_as_float(ctl_init.get("pan_speed"), 0.5), 0.05, 1.0))
        form.addRow("Pan speed", self.ctrl_pan)

        self.ctrl_tilt = QDoubleSpinBox()
        self.ctrl_tilt.setRange(0.05, 1.0)
        self.ctrl_tilt.setSingleStep(0.05)
        self.ctrl_tilt.setValue(_clamp(_as_float(ctl_init.get("tilt_speed"), 0.5), 0.05, 1.0))
        form.addRow("Tilt speed", self.ctrl_tilt)

        self.ctrl_zoom = QDoubleSpinBox()
        self.ctrl_zoom.setRange(0.05, 1.0)
        self.ctrl_zoom.setSingleStep(0.05)
        self.ctrl_zoom.setValue(_clamp(_as_float(ctl_init.get("zoom_speed"), 0.3), 0.05, 1.0))
        form.addRow("Zoom speed", self.ctrl_zoom)

        self.ctrl_duration = QDoubleSpinBox()
        self.ctrl_duration.setRange(0.1, 5.0)
        self.ctrl_duration.setSingleStep(0.1)
        self.ctrl_duration.setValue(_clamp(_as_float(ctl_init.get("duration"), 0.5), 0.1, 5.0))
        form.addRow("Move duration (s)", self.ctrl_duration)

        self.tabs.addTab(tab, "Controls")

    def _build_autopan_tab(self) -> None:
        tab = QWidget()
        form = QFormLayout(tab)
        ap_init = self._initial.get("autopan", {})

        self.ap_start_dir = QComboBox()
        self.ap_start_dir.addItems(["left", "right"])
        cur_dir = str(ap_init.get("start_direction", "right")).lower()
        idx = self.ap_start_dir.findText(cur_dir)
        if idx >= 0:
            self.ap_start_dir.setCurrentIndex(idx)
        form.addRow("Start direction", self.ap_start_dir)

        self.ap_angle = QSpinBox()
        self.ap_angle.setRange(30, 170)
        self.ap_angle.setValue(_as_int(ap_init.get("sweep_angle"), 120))
        form.addRow("Sweep angle (°)", self.ap_angle)

        self.ap_seconds = QDoubleSpinBox()
        self.ap_seconds.setRange(2.0, 180.0)
        self.ap_seconds.setSingleStep(0.5)
        self.ap_seconds.setValue(_clamp(_as_float(ap_init.get("seconds_per_side"), 8.0), 2.0, 180.0))
        form.addRow("Seconds per side", self.ap_seconds)

        self.ap_pause = QDoubleSpinBox()
        self.ap_pause.setRange(0.0, 15.0)
        self.ap_pause.setSingleStep(0.25)
        self.ap_pause.setValue(_clamp(_as_float(ap_init.get("edge_pause_seconds"), 0.5), 0.0, 15.0))
        form.addRow("Edge pause (s)", self.ap_pause)

        self.ap_smooth = QDoubleSpinBox()
        self.ap_smooth.setRange(0.0, 0.9)
        self.ap_smooth.setSingleStep(0.05)
        self.ap_smooth.setValue(_clamp(_as_float(ap_init.get("smooth_ratio"), 0.25), 0.0, 0.9))
        form.addRow("Smooth ratio", self.ap_smooth)

        self.ap_tilt = QDoubleSpinBox()
        self.ap_tilt.setRange(-1.0, 1.0)
        self.ap_tilt.setSingleStep(0.05)
        self.ap_tilt.setValue(_clamp(_as_float(ap_init.get("tilt"), 0.0), -1.0, 1.0))
        form.addRow("Tilt during sweep", self.ap_tilt)

        self.tabs.addTab(tab, "Auto Pan")

    def _build_camera_tab(self) -> None:
        tab = QWidget()
        form = QFormLayout(tab)
        conn_init = self._initial.get("connection", {})

        self.brand_override = QComboBox()
        self.brand_override.addItems(BRAND_CHOICES)
        cur = str(conn_init.get("brand_override") or "auto").lower()
        idx = self.brand_override.findText(cur)
        if idx >= 0:
            self.brand_override.setCurrentIndex(idx)
        form.addRow("Brand override", self.brand_override)

        explainer = QLabel(
            "Knoxnet auto-detects PTZ protocol from the camera's IP and "
            "stored credentials. Override only if auto-detect picks the wrong "
            "stack. Camera username, password, and IP are managed in Camera "
            "Configuration; the Tapo cloud password is requested on first PTZ "
            "use and never stored on disk by default."
        )
        explainer.setWordWrap(True)
        explainer.setStyleSheet("color: #94a3b8;")
        form.addRow(explainer)

        self.tabs.addTab(tab, "Camera")

    # ---- helpers ----

    def _sync_color_btn(self, btn: QPushButton, color: QColor) -> None:
        try:
            btn.setStyleSheet(
                f"QPushButton {{ background-color: {color.name()}; border: 1px solid {KnoxnetStyle.BORDER}; }}"
            )
        except Exception:
            pass

    def _pick_accent(self) -> None:
        c = QColorDialog.getColor(self._accent, self, "Select Accent Color")
        if c.isValid():
            self._accent = c
            self._sync_color_btn(self.ui_accent_btn, c)

    def _pick_bg(self) -> None:
        c = QColorDialog.getColor(self._bg, self, "Select Background Color")
        if c.isValid():
            self._bg = c
            self._sync_color_btn(self.ui_bg_btn, c)

    def _accept(self) -> None:
        self.result_settings = {
            "ui": {
                "opacity": _clamp(self.ui_opacity.value() / 100.0, 0.1, 0.9),
                "accent_color": self._accent.name(),
                "bg_color": self._bg.name(),
                "button_size": int(self.ui_btn_size.value()),
                "show_labels": bool(self.ui_show_labels.isChecked()),
                "pos_norm": list(self._initial.get("ui", {}).get("pos_norm", [0.5, 0.5])),
                "window_geom": self._initial.get("ui", {}).get("window_geom"),
            },
            "connection": {
                "brand_override": self.brand_override.currentText().strip().lower() or "auto",
            },
            "control": {
                "pan_speed": float(self.ctrl_pan.value()),
                "tilt_speed": float(self.ctrl_tilt.value()),
                "zoom_speed": float(self.ctrl_zoom.value()),
                "duration": float(self.ctrl_duration.value()),
            },
            "autopan": {
                "sweep_angle": int(self.ap_angle.value()),
                "seconds_per_side": float(self.ap_seconds.value()),
                "edge_pause_seconds": float(self.ap_pause.value()),
                "smooth_ratio": float(self.ap_smooth.value()),
                "start_direction": self.ap_start_dir.currentText().strip().lower(),
                "tilt": float(self.ap_tilt.value()),
            },
        }
        # Drop None to keep the persisted JSON tidy.
        self.result_settings["ui"] = {k: v for k, v in self.result_settings["ui"].items() if v is not None}
        self.accept()


# ---------------------------------------------------------------------------- #
# HUD widget
# ---------------------------------------------------------------------------- #

class PTZOverlayWidget(QWidget):
    """Translucent PTZ HUD with D-pad, zoom, presets, home, autopan, stop, gear."""

    request_settings = Signal()
    request_credentials = Signal()
    PRESET_TOKENS = ("1", "2", "3", "4")
    PRESET_LONGPRESS_MS = 800

    def __init__(self, camera_widget, settings: Dict[str, Any], parent=None, mode: str = "overlay"):
        super().__init__(parent)
        self.camera_widget = camera_widget
        self.settings: Dict[str, Any] = settings or default_ptz_overlay_settings()
        self.mode = mode
        self._dragging = False
        self._drag_start: Optional[QPoint] = None
        self._hud_start: Optional[QPoint] = None
        self._sweep_active = False

        # Long-press tracking for preset buttons
        self._preset_timers: Dict[str, QTimer] = {}
        self._preset_longpress_fired: Dict[str, bool] = {}

        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self.setAttribute(Qt.WidgetAttribute.WA_OpaquePaintEvent, False)
        self.setMouseTracking(True)

        self.hud = QWidget(self)
        self.hud.setObjectName("PTZHUD")

        self._build_ui()
        self.apply_style()
        self.reposition_hud()
        self.show()

    # ---- styling ----

    def apply_style(self) -> None:
        ui = self.settings.get("ui", {})
        opacity = _clamp(_as_float(ui.get("opacity"), 0.35), 0.1, 0.9)
        bg = str(ui.get("bg_color", "#0b1220"))
        accent = str(ui.get("accent_color", "#24D1FF"))
        col = QColor(bg)
        if not col.isValid():
            col = QColor("#0b1220")
        rgba = f"rgba({col.red()}, {col.green()}, {col.blue()}, {int(opacity * 255)})"
        self.hud.setStyleSheet(
            f"""
            QWidget#PTZHUD {{
                background-color: {rgba};
                border: 1px solid rgba(255,255,255,30);
                border-radius: 10px;
            }}
            QPushButton {{
                background-color: rgba(0,0,0,70);
                color: #e5e7eb;
                border: 1px solid rgba(255,255,255,35);
                border-radius: 8px;
                padding: 0px;
                font-weight: 600;
            }}
            QPushButton:hover {{
                border: 1px solid {accent};
            }}
            QPushButton:pressed {{
                background-color: rgba(0,0,0,120);
            }}
            QPushButton[ptzPreset="true"] {{
                font-size: 11px;
            }}
            """
        )

        btn_size = int(_clamp(_as_int(ui.get("button_size"), 44), 28, 80))
        for b in self._all_buttons():
            b.setFixedSize(btn_size, btn_size)
        # Presets are slightly smaller for a denser row
        preset_size = max(24, btn_size - 12)
        for pb in self._preset_buttons.values():
            pb.setFixedSize(preset_size, preset_size)
        self.hud.adjustSize()

        show_labels = bool(ui.get("show_labels", False))
        self._set_labels_visible(show_labels)

    def _set_labels_visible(self, show: bool) -> None:
        # Glyphs only either way (kept for back-compat with the toggle).
        glyphs = {
            self.btn_up: "↑", self.btn_down: "↓", self.btn_left: "←", self.btn_right: "→",
            self.btn_home: "⌂", self.btn_zoom_in: "+", self.btn_zoom_out: "−",
            self.btn_stop: "■", self.btn_autopan: "⟲", self.btn_settings: "⚙",
        }
        for btn, glyph in glyphs.items():
            btn.setText(glyph)

    def _all_buttons(self):
        base = [
            self.btn_up, self.btn_down, self.btn_left, self.btn_right,
            self.btn_home, self.btn_zoom_in, self.btn_zoom_out,
            self.btn_stop, self.btn_autopan, self.btn_settings,
        ]
        return base

    # ---- layout ----

    def _build_ui(self) -> None:
        from PySide6.QtWidgets import QGridLayout

        layout = QGridLayout(self.hud)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setHorizontalSpacing(8)
        layout.setVerticalSpacing(8)

        self.btn_zoom_in = QPushButton("+")
        self.btn_zoom_out = QPushButton("−")
        self.btn_up = QPushButton("↑")
        self.btn_down = QPushButton("↓")
        self.btn_left = QPushButton("←")
        self.btn_right = QPushButton("→")
        self.btn_home = QPushButton("⌂")
        self.btn_stop = QPushButton("■")
        self.btn_autopan = QPushButton("⟲")
        self.btn_settings = QPushButton("⚙")

        # Column 0: zoom in / settings / zoom out
        layout.addWidget(self.btn_zoom_in, 0, 0)
        layout.addWidget(self.btn_settings, 1, 0)
        layout.addWidget(self.btn_zoom_out, 2, 0)

        # D-pad in columns 1..3
        layout.addWidget(self.btn_up, 0, 2)
        layout.addWidget(self.btn_left, 1, 1)
        layout.addWidget(self.btn_home, 1, 2)
        layout.addWidget(self.btn_right, 1, 3)
        layout.addWidget(self.btn_down, 2, 2)

        # Column 4: stop / autopan
        layout.addWidget(self.btn_stop, 0, 4)
        layout.addWidget(self.btn_autopan, 2, 4)

        # Presets row spanning columns 0..4
        from PySide6.QtWidgets import QHBoxLayout as _HBox, QFrame
        preset_row = QFrame()
        preset_layout = _HBox(preset_row)
        preset_layout.setContentsMargins(0, 0, 0, 0)
        preset_layout.setSpacing(6)
        preset_layout.addStretch()
        preset_label = QLabel("Presets")
        preset_label.setStyleSheet("color: rgba(229,231,235,160); font-size: 10px;")
        preset_layout.addWidget(preset_label)

        self._preset_buttons: Dict[str, QPushButton] = {}
        for token in self.PRESET_TOKENS:
            btn = QPushButton(token)
            btn.setProperty("ptzPreset", True)
            btn.setToolTip(f"Preset {token} (click = goto, hold = save current view)")
            btn.pressed.connect(lambda t=token: self._preset_pressed(t))
            btn.released.connect(lambda t=token: self._preset_released(t))
            preset_layout.addWidget(btn)
            self._preset_buttons[token] = btn
        preset_layout.addStretch()
        layout.addWidget(preset_row, 3, 0, 1, 5)

        # Wire press/release for movement buttons (press = move, release = stop)
        self.btn_up.pressed.connect(lambda: self._move("up"))
        self.btn_up.released.connect(self._stop)
        self.btn_down.pressed.connect(lambda: self._move("down"))
        self.btn_down.released.connect(self._stop)
        self.btn_left.pressed.connect(lambda: self._move("left"))
        self.btn_left.released.connect(self._stop)
        self.btn_right.pressed.connect(lambda: self._move("right"))
        self.btn_right.released.connect(self._stop)
        self.btn_zoom_in.pressed.connect(lambda: self._zoom("in"))
        self.btn_zoom_in.released.connect(self._stop)
        self.btn_zoom_out.pressed.connect(lambda: self._zoom("out"))
        self.btn_zoom_out.released.connect(self._stop)

        self.btn_home.clicked.connect(self._home)
        self.btn_stop.clicked.connect(self._stop)
        self.btn_autopan.clicked.connect(self._toggle_autopan)
        # Click  -> PTZ Settings (look / controls / autopan)
        # Right-click (or long-press) -> small menu with Credentials access
        self.btn_settings.clicked.connect(lambda: self.request_settings.emit())
        self.btn_settings.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.btn_settings.customContextMenuRequested.connect(self._show_settings_menu)
        self.btn_settings.setToolTip("PTZ Settings (right-click for credentials & protocol)")

    # ---- positioning ----

    def reposition_hud(self) -> None:
        ui = self.settings.get("ui", {})
        if self.mode == "panel":
            nx, ny = 0.5, 0.5
        else:
            pos_norm = ui.get("pos_norm", [0.5, 0.5])
            try:
                nx, ny = float(pos_norm[0]), float(pos_norm[1])
            except Exception:
                nx, ny = 0.5, 0.5
            nx = _clamp(nx, 0.05, 0.95)
            ny = _clamp(ny, 0.05, 0.95)

        self.hud.adjustSize()
        w = self.width()
        h = self.height()
        hud_w = self.hud.width()
        hud_h = self.hud.height()
        x = int(nx * w - hud_w / 2)
        y = int(ny * h - hud_h / 2)
        x = max(4, min(w - hud_w - 4, x))
        y = max(4, min(h - hud_h - 4, y))
        self.hud.move(x, y)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.reposition_hud()

    # ---- dragging (overlay mode only) ----

    def mousePressEvent(self, event):
        if (
            self.mode != "panel"
            and event.button() == Qt.MouseButton.LeftButton
            and self.hud.geometry().contains(event.position().toPoint())
        ):
            self._dragging = True
            self._drag_start = event.globalPosition().toPoint()
            self._hud_start = self.hud.pos()
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self._dragging and self._drag_start and self._hud_start:
            delta = event.globalPosition().toPoint() - self._drag_start
            new_pos = self._hud_start + delta
            x = max(4, min(self.width() - self.hud.width() - 4, new_pos.x()))
            y = max(4, min(self.height() - self.hud.height() - 4, new_pos.y()))
            self.hud.move(x, y)
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if self._dragging:
            self._dragging = False
            self._drag_start = None
            self._hud_start = None
            self._store_norm_position()
        super().mouseReleaseEvent(event)

    def _store_norm_position(self) -> None:
        if self.mode == "panel":
            return
        try:
            center = self.hud.geometry().center()
            nx = center.x() / max(1, self.width())
            ny = center.y() / max(1, self.height())
            self.settings.setdefault("ui", {})["pos_norm"] = [
                float(_clamp(nx, 0.05, 0.95)),
                float(_clamp(ny, 0.05, 0.95)),
            ]
        except Exception:
            pass

    # ---- PTZ commands (delegated to camera_widget) ----

    def _move(self, direction: str) -> None:
        ctrl = self.settings.get("control", {})
        pan = float(ctrl.get("pan_speed", 0.5))
        tilt = float(ctrl.get("tilt_speed", 0.5))
        dur = float(ctrl.get("duration", 0.5))
        pan_speed = 0.0
        tilt_speed = 0.0
        if direction == "left":
            pan_speed = -abs(pan)
        elif direction == "right":
            pan_speed = abs(pan)
        elif direction == "up":
            tilt_speed = abs(tilt)
        elif direction == "down":
            tilt_speed = -abs(tilt)
        self.camera_widget.ptz_send(
            "continuous_move",
            {"pan_speed": pan_speed, "tilt_speed": tilt_speed, "duration": dur},
        )

    def _zoom(self, direction: str) -> None:
        ctrl = self.settings.get("control", {})
        zoom = float(ctrl.get("zoom_speed", 0.3))
        dur = float(ctrl.get("duration", 0.5))
        z = abs(zoom) if direction == "in" else -abs(zoom)
        self.camera_widget.ptz_send(
            "continuous_move",
            {"zoom_speed": z, "duration": dur},
        )

    def _stop(self) -> None:
        self.camera_widget.ptz_send("stop", {})

    def _home(self) -> None:
        self.camera_widget.ptz_send("go_home", {})

    def _toggle_autopan(self) -> None:
        if self._sweep_active:
            self.camera_widget.ptz_send("stop_sweep", {})
            return
        ap = self.settings.get("autopan", {})
        angle = int(ap.get("sweep_angle", 120))
        half = _clamp(angle / 180.0, 0.1, 1.0)
        start_pan = -half
        end_pan = half
        start_dir = -1 if str(ap.get("start_direction", "right")).lower() == "left" else 1
        seconds_per_side = _clamp(float(ap.get("seconds_per_side", 8.0)), 2.0, 180.0)
        span_norm = max(0.1, min(1.8, abs(end_pan - start_pan)))
        speed = _clamp((span_norm * 2.25) / seconds_per_side, 0.002, 0.95)
        edge_pause = _clamp(float(ap.get("edge_pause_seconds", 0.5)), 0.0, 15.0)
        smooth = _clamp(float(ap.get("smooth_ratio", 0.25)), 0.0, 0.9)
        tilt = _clamp(float(ap.get("tilt", 0.0)), -1.0, 1.0)
        self.camera_widget.ptz_send(
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

    # ---- presets (click = goto, long-press = set) ----

    def _preset_pressed(self, token: str) -> None:
        self._preset_longpress_fired[token] = False
        timer = QTimer(self)
        timer.setSingleShot(True)
        timer.timeout.connect(lambda t=token: self._preset_longpress(t))
        timer.start(self.PRESET_LONGPRESS_MS)
        self._preset_timers[token] = timer

    def _preset_released(self, token: str) -> None:
        timer = self._preset_timers.pop(token, None)
        if timer:
            timer.stop()
        if self._preset_longpress_fired.pop(token, False):
            return  # long-press already saved the preset
        self.camera_widget.ptz_send("goto_preset", {"preset_token": token})

    def _show_settings_menu(self, pos) -> None:
        """Right-click on the gear button -> small menu (settings + credentials)."""
        from PySide6.QtWidgets import QMenu
        menu = QMenu(self.btn_settings)
        act_settings = menu.addAction("PTZ Settings...")
        act_creds = menu.addAction("Credentials & Protocol...")
        act_settings.triggered.connect(lambda: self.request_settings.emit())
        act_creds.triggered.connect(lambda: self.request_credentials.emit())
        try:
            global_pos = self.btn_settings.mapToGlobal(pos)
        except Exception:
            global_pos = self.btn_settings.mapToGlobal(self.btn_settings.rect().bottomLeft())
        menu.exec(global_pos)

    def _preset_longpress(self, token: str) -> None:
        self._preset_longpress_fired[token] = True
        # subtle visual feedback
        btn = self._preset_buttons.get(token)
        if btn is not None:
            try:
                btn.setStyleSheet("border: 1px solid #22c55e; font-size: 11px;")
                QTimer.singleShot(700, lambda b=btn: b.setStyleSheet(""))
            except Exception:
                pass
        self.camera_widget.ptz_send(
            "set_preset",
            {"preset_token": token, "preset_name": f"Preset {token}"},
        )

    # ---- result / state hook ----

    def update_from_ptz_result(self, result: Dict[str, Any]) -> None:
        try:
            data = result.get("data") or {}
            sweep_active = bool(data.get("sweep_active")) if isinstance(data, dict) else False
            self._sweep_active = sweep_active
            if sweep_active:
                self.btn_autopan.setStyleSheet("border: 1px solid #22c55e;")
            else:
                self.btn_autopan.setStyleSheet("")
        except Exception:
            pass


# ---------------------------------------------------------------------------- #
# Floating window
# ---------------------------------------------------------------------------- #

class PTZControllerWindow(BaseDesktopWidget):
    """Floating PTZ controller window (the primary PTZ surface)."""

    def __init__(self, camera_widget, settings: Dict[str, Any]):
        label = getattr(camera_widget, "camera_name", None) or getattr(camera_widget, "camera_id", "Camera")
        super().__init__(title=f"PTZ: {label}", width=300, height=300)
        self.camera_widget = camera_widget
        self.settings = settings

        try:
            self.title_bar.show()
        except Exception:
            pass

        self.keep_aspect_ratio = False

        self.panel = PTZOverlayWidget(camera_widget, settings, parent=self, mode="panel")
        # Gear (left-click) -> slim settings dialog.
        # Gear (right-click) -> PTZ credentials & protocol dialog.
        self.panel.request_settings.connect(camera_widget.open_ptz_overlay_settings)
        self.panel.request_credentials.connect(camera_widget.open_ptz_credentials_dialog)
        self.set_content(self.panel)

        self._apply_saved_geometry()

    def _apply_saved_geometry(self):
        try:
            ui = (self.settings or {}).get("ui", {}) or {}
            geom = ui.get("window_geom")
            if isinstance(geom, (list, tuple)) and len(geom) == 4:
                x, y, w, h = [int(v) for v in geom]
                if w > 120 and h > 120:
                    self.setGeometry(x, y, w, h)
        except Exception:
            pass

    def _persist_geometry(self):
        try:
            geo = self.geometry()
            self.settings.setdefault("ui", {})["window_geom"] = [
                geo.x(), geo.y(), geo.width(), geo.height()
            ]
        except Exception:
            pass

    def closeEvent(self, event):
        self._persist_geometry()
        super().closeEvent(event)
