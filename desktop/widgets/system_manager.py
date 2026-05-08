from __future__ import annotations

import os
import sys
import subprocess
import threading
import time
import requests
import shutil
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
from pathlib import Path

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QFormLayout,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSlider,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QMessageBox,
    QWidget,
    QCheckBox,
    QLineEdit,
    QTextEdit,
    QFileDialog,
)

def _qt_clipboard_set(text: str) -> None:
    try:
        from PySide6.QtGui import QGuiApplication
        cb = QGuiApplication.clipboard()
        if cb:
            cb.setText(text or "")
    except Exception:
        pass


def _qpixmap_from_base64_png(b64: str) -> Optional[QPixmap]:
    try:
        import base64
        raw = base64.b64decode((b64 or "").strip())
        pm = QPixmap()
        if pm.loadFromData(raw, "PNG"):
            return pm
        return None
    except Exception:
        return None


def _repo_root() -> Path:
    """
    Resolve repo root from this file location:
      <repo>/desktop/widgets/system_manager.py -> parents[2] == <repo>
    """
    try:
        return Path(__file__).resolve().parents[2]
    except Exception:
        return Path.cwd()


def _mediamtx_entrypoint() -> str:
    if os.name == "nt":
        return "mediamtx/mediamtx.exe"
    return "mediamtx/mediamtx"


def _vision_entrypoint() -> str:
    if os.name == "nt":
        return "services/vision_local/start_service.bat"
    # Use a shell script on Linux/macOS so this works in frozen builds (PyInstaller exe can't do `-m`).
    return "services/vision_local/start_service.sh"


def _llm_entrypoint() -> str:
    if os.name == "nt":
        return "services/llm_local/start_service.bat"
    # Use a shell script on Linux/macOS so this works in frozen builds (PyInstaller exe can't do `-m`).
    return "services/llm_local/start_service.sh"


def _default_state_dir() -> Path:
    """
    Per-user writable state directory.
    Linux:   $XDG_STATE_HOME/KnoxnetVMS (default: ~/.local/state/KnoxnetVMS)
    Other:   best-effort fallback to ~/.local/state/KnoxnetVMS
    """
    try:
        xdg = (os.environ.get("XDG_STATE_HOME") or "").strip()
        base = Path(xdg).expanduser() if xdg else (Path.home() / ".local" / "state")
        return (base / "KnoxnetVMS").resolve()
    except Exception:
        return (Path.home() / ".local" / "state" / "KnoxnetVMS")


def _writable_log_dir(*, repo_root: Path) -> Path:
    """
    Pick a writable log directory.
    - Dev mode: <repo>/logs
    - Frozen/AppImage: $XDG_STATE_HOME/KnoxnetVMS/logs
    - Always allow override via KNOXNET_LOG_DIR
    """
    env = (os.environ.get("KNOXNET_LOG_DIR") or "").strip()
    if env:
        try:
            return Path(env).expanduser().resolve()
        except Exception:
            return Path(env)
    if getattr(sys, "frozen", False):
        return _default_state_dir() / "logs"
    return (repo_root / "logs").resolve()


class _PeerConfigDialog(QDialog):
    def __init__(
        self,
        *,
        parent: QWidget,
        title: str,
        config_text: str,
        qr_png_base64: str,
        qr_error: Optional[str] = None,
        warnings: Optional[List[str]] = None,
        endpoint: Optional[str] = None,
    ):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.resize(820, 560)

        root = QHBoxLayout(self)
        root.setContentsMargins(12, 12, 12, 12)
        root.setSpacing(12)

        left = QVBoxLayout()
        right = QVBoxLayout()
        root.addLayout(left, stretch=0)
        root.addLayout(right, stretch=1)

        qr_lbl = QLabel()
        pm = _qpixmap_from_base64_png(qr_png_base64)
        if pm is not None and not pm.isNull():
            qr_lbl.setPixmap(pm)
        else:
            parts = ["QR unavailable."]
            if qr_error:
                parts.append(str(qr_error))
            else:
                parts.append("Backend did not return a valid PNG.")
            parts.append("You can still import the .conf file in the WireGuard app.")
            qr_lbl.setText("\n\n".join([p for p in parts if p]))
            qr_lbl.setStyleSheet("color: #ef4444;")
            qr_lbl.setWordWrap(True)
        left.addWidget(qr_lbl)
        left.addStretch(1)

        if warnings:
            warn_lbl = QLabel("\n".join([str(w) for w in warnings if str(w).strip()]))
            warn_lbl.setStyleSheet("color: #f59e0b; font-weight: bold;")
            warn_lbl.setWordWrap(True)
            right.addWidget(warn_lbl)
        if endpoint:
            ep_lbl = QLabel(f"Endpoint: {endpoint}")
            ep_lbl.setStyleSheet("color: #9aa4b2;")
            ep_lbl.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
            right.addWidget(ep_lbl)

        text = QTextEdit()
        text.setReadOnly(True)
        text.setPlainText(config_text or "")
        right.addWidget(text, stretch=1)

        btn_row = QHBoxLayout()
        copy_btn = QPushButton("Copy Config")
        copy_btn.clicked.connect(lambda: _qt_clipboard_set(config_text))
        btn_row.addWidget(copy_btn)

        save_btn = QPushButton("Save .conf…")
        def _save():
            try:
                path, _ = QFileDialog.getSaveFileName(self, "Save WireGuard Config", "peer.conf", "Config (*.conf);;All Files (*)")
                if not path:
                    return
                Path(path).write_text(config_text, encoding="utf-8")
            except Exception as e:
                QMessageBox.warning(self, "Save Failed", f"Failed to save file: {e}")
        save_btn.clicked.connect(_save)
        btn_row.addWidget(save_btn)

        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.accept)
        btn_row.addStretch(1)
        btn_row.addWidget(close_btn)

        right.addLayout(btn_row)

@dataclass
class _ProcStats:
    cpu_percent: Optional[float] = None
    mem_percent: Optional[float] = None
    rss_mb: Optional[float] = None


class _DiskStatusSignal(QWidget):
    """Thin QObject bridge to emit disk-status updates on the GUI thread."""
    updated = Signal(list)

    def __init__(self):
        super().__init__()
        self.setVisible(False)


class SystemManagerDialog(QDialog):
    """
    Compact System Management & Health dialog.

    Core services (Backend, MediaMTX) have one-click start/stop and
    persistent auto-start toggles.  Optional services are grouped
    separately to avoid clutter.
    """

    status_updated = Signal(str, str)
    _rec_toggle_signal = Signal(str)
    # Structured per-camera result for Record All / Stop All so the UI can
    # show users exactly which cameras succeeded, which were skipped because
    # the camera is offline, and which outright failed.
    _rec_result_signal = Signal(object)

    def __init__(self, parent=None, app=None):
        super().__init__(parent)
        self._app = app
        self.setWindowTitle("System Management")
        self.setWindowFlags(Qt.WindowType.Window | Qt.WindowType.WindowStaysOnTopHint)

        # Cap the dialog to ~80% of the available screen height so it
        # never runs off-screen on smaller displays / lower resolutions.
        # The scrollable content area below handles overflow gracefully.
        try:
            from PySide6.QtGui import QGuiApplication
            screen = QGuiApplication.primaryScreen()
            if screen is not None:
                avail = screen.availableGeometry()
                preferred_h = max(420, int(avail.height() * 0.80))
                preferred_w = int(min(620, max(540, avail.width() * 0.40)))
                self.setMaximumHeight(int(avail.height() - 40))
                self.setMaximumWidth(int(avail.width() - 40))
                self.resize(preferred_w, preferred_h)
            else:
                self.resize(580, 700)
        except Exception:
            self.resize(580, 700)

        try:
            from dotenv import load_dotenv
            load_dotenv()
        except Exception:
            pass

        self._processes: Dict[str, Any] = {}
        self._backend_origin = (os.environ.get("KNOXNET_BACKEND_ORIGIN") or "http://localhost:5000").rstrip("/")

        # WireGuard state (kept for compat with existing RA methods)
        self._ra_last_poll_ts = 0.0
        self._ra_last_status: Dict[str, Any] = {}
        self._ra_request_lock = threading.Lock()
        self._ra_last_peer_bundle: Optional[Dict[str, Any]] = None
        self._ra_peer_autocreate_attempted = False

        prefs = self._load_prefs()

        # Outer dialog layout holds only the scroll area so the dialog
        # itself never grows beyond its capped height; the inner content
        # widget can be arbitrarily tall and the user just scrolls.
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        scroll = QScrollArea(self)
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setFrameShape(QFrame.Shape.NoFrame)

        content = QWidget()
        scroll.setWidget(content)
        outer.addWidget(scroll)

        root = QVBoxLayout(content)
        root.setContentsMargins(14, 14, 14, 14)
        root.setSpacing(10)

        self.status_updated.connect(self._on_status_updated)

        # ── Header ──
        hdr = QHBoxLayout()
        self.status_lbl = QLabel("Checking…")
        self.status_lbl.setStyleSheet("font-weight: bold; font-size: 13px;")
        hdr.addWidget(self.status_lbl, stretch=1)
        self.proc_lbl = QLabel("")
        self.proc_lbl.setStyleSheet("color: #9aa4b2; font-size: 11px;")
        hdr.addWidget(self.proc_lbl)
        root.addLayout(hdr)

        # ── Core Services ──
        core_box = QGroupBox("Core Services")
        core_g = QGridLayout(core_box)
        core_g.setColumnStretch(0, 1)

        self.service_widgets: Dict[str, Dict[str, Any]] = {}

        core_services = [
            ("Backend",  "http://localhost:5000/",                          "app.py"),
            ("MediaMTX", "http://localhost:9997/v3/config/global/get",      _mediamtx_entrypoint()),
        ]

        for row, (name, health_url, entry_point) in enumerate(core_services):
            lbl = QLabel(name)
            lbl.setStyleSheet("font-weight: bold;")

            status_lbl = QLabel("…")
            status_lbl.setFixedWidth(80)

            auto_chk = QCheckBox("Auto-start")
            auto_chk.setToolTip(f"Start {name} automatically when the desktop app launches")
            auto_chk.setChecked(prefs.get(f"autostart_{name.lower()}", False))
            auto_chk.toggled.connect(lambda checked, n=name: self._save_autostart_pref(n, checked))

            start_btn = QPushButton("Start")
            start_btn.setFixedWidth(60)
            start_btn.clicked.connect(lambda _, n=name, e=entry_point: self._start_service(n, e))

            stop_btn = QPushButton("Stop")
            stop_btn.setFixedWidth(60)
            stop_btn.clicked.connect(lambda _, n=name: self._stop_service(n))

            core_g.addWidget(lbl,        row, 0)
            core_g.addWidget(status_lbl, row, 1)
            core_g.addWidget(auto_chk,   row, 2)
            core_g.addWidget(start_btn,  row, 3)
            core_g.addWidget(stop_btn,   row, 4)

            self.service_widgets[name] = {
                "status_lbl": status_lbl,
                "start_btn": start_btn,
                "stop_btn": stop_btn,
                "health_url": health_url,
                "entry_point": entry_point,
            }

        root.addWidget(core_box)

        # ── Recording snapshot ──
        rec_box = QGroupBox("Recording")
        rec_v = QVBoxLayout(rec_box)
        rec_v.setSpacing(6)

        # Directory row
        dir_row = QHBoxLayout()
        dir_label = QLabel("Save to:")
        dir_label.setStyleSheet("color: #94a3b8; font-size: 11px;")
        dir_row.addWidget(dir_label)

        from core.paths import get_recordings_dir
        saved_dir = prefs.get("recording_dir", "")
        self._rec_dir_edit = QLineEdit(saved_dir or str(get_recordings_dir()))
        self._rec_dir_edit.setPlaceholderText("Default recordings directory")
        self._rec_dir_edit.setStyleSheet("font-size: 11px;")
        # Don't let a long absolute path push the dialog wider; the
        # field still grows with the dialog because of stretch=1.
        self._rec_dir_edit.setMinimumWidth(120)
        from PySide6.QtWidgets import QSizePolicy
        self._rec_dir_edit.setSizePolicy(
            QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Fixed
        )
        dir_row.addWidget(self._rec_dir_edit, stretch=1)

        browse_btn = QPushButton("…")
        browse_btn.setFixedWidth(28)
        browse_btn.setToolTip("Browse for recording directory")
        browse_btn.clicked.connect(self._browse_rec_dir)
        dir_row.addWidget(browse_btn)

        apply_dir_btn = QPushButton("Apply")
        apply_dir_btn.setFixedWidth(50)
        apply_dir_btn.setToolTip("Set this directory for all cameras (or visible only)")
        apply_dir_btn.clicked.connect(self._apply_rec_dir_all)
        dir_row.addWidget(apply_dir_btn)
        rec_v.addLayout(dir_row)

        # Options row
        opt_row = QHBoxLayout()
        self._visible_only_chk = QCheckBox("Apply only to visible cameras")
        self._visible_only_chk.setStyleSheet("font-size: 11px; color: #94a3b8;")
        self._visible_only_chk.setToolTip("Only set the directory for cameras that currently have an open widget")
        opt_row.addWidget(self._visible_only_chk)
        opt_row.addStretch()
        rec_v.addLayout(opt_row)

        # Retention row
        retention_row = QHBoxLayout()
        ret_label = QLabel("Auto-delete after:")
        ret_label.setStyleSheet("color: #94a3b8; font-size: 11px;")
        retention_row.addWidget(ret_label)

        self._retention_combo = QComboBox()
        self._retention_combo.setStyleSheet("font-size: 11px;")
        for label, value in [
            ("Never (manual cleanup)", "0s"),
            ("1 day", "24h"),
            ("3 days", "72h"),
            ("7 days", "168h"),
            ("14 days", "336h"),
            ("30 days", "720h"),
            ("60 days", "1440h"),
            ("90 days", "2160h"),
        ]:
            self._retention_combo.addItem(label, value)
        saved_retention = prefs.get("record_delete_after", "168h")
        idx = self._retention_combo.findData(saved_retention)
        if idx >= 0:
            self._retention_combo.setCurrentIndex(idx)
        else:
            self._retention_combo.setCurrentIndex(3)  # default 7d
        self._retention_combo.setToolTip(
            "MediaMTX will automatically delete recording segments older than this.\n"
            "Set to 'Never' to manage disk space manually or via Disk Management below."
        )
        retention_row.addWidget(self._retention_combo)

        apply_retention_btn = QPushButton("Apply")
        apply_retention_btn.setFixedWidth(50)
        apply_retention_btn.setToolTip("Apply retention setting to MediaMTX config")
        apply_retention_btn.clicked.connect(self._apply_retention)
        retention_row.addWidget(apply_retention_btn)

        retention_row.addStretch()
        rec_v.addLayout(retention_row)

        # Status + buttons row
        rec_row = QHBoxLayout()
        self.rec_summary_lbl = QLabel("Recording: —")
        self.rec_summary_lbl.setStyleSheet("color: #94a3b8; font-size: 12px;")
        rec_row.addWidget(self.rec_summary_lbl, stretch=1)

        self.rec_all_btn = QPushButton("Record All")
        self.rec_all_btn.setFixedWidth(80)
        self.rec_all_btn.clicked.connect(lambda: self._toggle_recording_all(True))
        rec_row.addWidget(self.rec_all_btn)

        self.stop_rec_btn = QPushButton("Stop All")
        self.stop_rec_btn.setFixedWidth(80)
        self.stop_rec_btn.clicked.connect(lambda: self._toggle_recording_all(False))
        rec_row.addWidget(self.stop_rec_btn)
        rec_v.addLayout(rec_row)

        root.addWidget(rec_box)

        # ── Disk Management ──
        disk_box = QGroupBox("Disk Management")
        disk_v = QVBoxLayout(disk_box)
        disk_v.setSpacing(6)

        self._disk_protect_chk = QCheckBox("Protect OS disk (auto-cleanup old recordings)")
        self._disk_protect_chk.setToolTip(
            "When enabled, the storage manager automatically deletes the oldest "
            "recordings once disk usage exceeds the cleanup threshold, preventing "
            "the system drive from filling up and crashing the OS."
        )
        self._disk_protect_chk.setStyleSheet("font-weight: bold;")
        disk_v.addWidget(self._disk_protect_chk)

        disk_form = QFormLayout()
        disk_form.setContentsMargins(0, 4, 0, 0)

        self._cleanup_threshold_spin = QSpinBox()
        self._cleanup_threshold_spin.setRange(50, 98)
        self._cleanup_threshold_spin.setSuffix("% disk usage")
        self._cleanup_threshold_spin.setToolTip(
            "Start deleting old recordings once disk usage exceeds this percentage"
        )
        disk_form.addRow("Cleanup threshold:", self._cleanup_threshold_spin)

        self._cleanup_target_spin = QSpinBox()
        self._cleanup_target_spin.setRange(30, 95)
        self._cleanup_target_spin.setSuffix("% disk usage")
        self._cleanup_target_spin.setToolTip(
            "When bulk-cleaning, delete until disk usage drops below this level"
        )
        disk_form.addRow("Target after cleanup:", self._cleanup_target_spin)

        self._critical_threshold_spin = QSpinBox()
        self._critical_threshold_spin.setRange(80, 99)
        self._critical_threshold_spin.setSuffix("% disk usage")
        self._critical_threshold_spin.setToolTip(
            "EMERGENCY: if disk usage reaches this level, all recordings are "
            "paused immediately while aggressive cleanup runs to prevent OS crash"
        )
        disk_form.addRow("Emergency stop:", self._critical_threshold_spin)

        disk_v.addLayout(disk_form)

        # Dynamic container for per-drive usage bars
        self._disk_bars_container = QVBoxLayout()
        self._disk_bars_container.setSpacing(4)
        self._disk_bar_widgets: list[dict] = []
        disk_v.addLayout(self._disk_bars_container)

        self._disk_status_msg = QLabel("")
        self._disk_status_msg.setStyleSheet("color: #94a3b8; font-size: 11px;")
        self._disk_status_msg.setWordWrap(True)
        disk_v.addWidget(self._disk_status_msg)

        # Save / refresh buttons
        disk_btn_row = QHBoxLayout()
        disk_btn_row.addStretch()

        disk_refresh_btn = QPushButton("Refresh")
        disk_refresh_btn.setFixedWidth(70)
        disk_refresh_btn.setToolTip("Refresh disk usage information")
        disk_refresh_btn.clicked.connect(self._refresh_disk_status)
        disk_btn_row.addWidget(disk_refresh_btn)

        disk_save_btn = QPushButton("Save")
        disk_save_btn.setFixedWidth(70)
        disk_save_btn.setToolTip("Apply disk management settings")
        disk_save_btn.clicked.connect(self._save_disk_settings)
        disk_btn_row.addWidget(disk_save_btn)
        disk_v.addLayout(disk_btn_row)

        root.addWidget(disk_box)

        self._load_disk_settings()
        self._refresh_disk_status()

        # Wire up the protect checkbox to enable/disable the threshold controls
        self._disk_protect_chk.toggled.connect(self._on_disk_protect_toggled)
        self._on_disk_protect_toggled(self._disk_protect_chk.isChecked())

        # ── Optional Services (collapsed by default) ──
        opt_box = QGroupBox("Optional Services")
        opt_g = QGridLayout(opt_box)
        opt_g.setColumnStretch(0, 1)

        optional_services = [
            ("Local Vision", "http://localhost:8101/health", _vision_entrypoint()),
            ("Local LLM",    "http://localhost:8102/health", _llm_entrypoint()),
        ]

        for row, (name, health_url, entry_point) in enumerate(optional_services):
            lbl = QLabel(name)
            status_lbl = QLabel("…")
            status_lbl.setFixedWidth(80)

            start_btn = QPushButton("Start")
            start_btn.setFixedWidth(60)
            start_btn.clicked.connect(lambda _, n=name, e=entry_point: self._start_service(n, e))

            stop_btn = QPushButton("Stop")
            stop_btn.setFixedWidth(60)
            stop_btn.clicked.connect(lambda _, n=name: self._stop_service(n))

            opt_g.addWidget(lbl,        row, 0)
            opt_g.addWidget(status_lbl, row, 1)
            opt_g.addWidget(start_btn,  row, 3)
            opt_g.addWidget(stop_btn,   row, 4)

            self.service_widgets[name] = {
                "status_lbl": status_lbl,
                "start_btn": start_btn,
                "stop_btn": stop_btn,
                "health_url": health_url,
                "entry_point": entry_point,
            }

        root.addWidget(opt_box)

        # ── Auto Protection (load shedder) ──
        self._build_auto_protect_panel(root)

        # ── System health bar ──
        self.health_lbl = QLabel("")
        self.health_lbl.setStyleSheet("color: #64748b; font-size: 11px;")
        self.health_lbl.setWordWrap(True)
        root.addWidget(self.health_lbl)

        # ── Timer ──
        self._timer = QTimer(self)
        self._timer.timeout.connect(self.refresh)
        self._timer.start(3000)

        self._prime_psutil()
        self.refresh()

    # ── Preferences helpers ──

    def _load_prefs(self) -> dict:
        try:
            return self._app._load_prefs() if self._app else {}
        except Exception:
            return {}

    def _persist_rec_dir(self, path: str):
        """Save the recording directory to prefs so it persists across restarts."""
        if not self._app:
            return
        try:
            p = self._app._load_prefs()
            p["recording_dir"] = path
            self._app._save_prefs(p)
        except Exception:
            pass

    def _apply_retention(self):
        """Write the selected retention period to MediaMTX config and prefs."""
        value = self._retention_combo.currentData() or "168h"
        try:
            if self._app:
                p = self._app._load_prefs()
                p["record_delete_after"] = value
                self._app._save_prefs(p)
        except Exception:
            pass

        # Update the active mediamtx yml
        try:
            from app import _mediamtx_yml_path
            import re as _re
            yml = _mediamtx_yml_path()
            if yml.exists():
                text = yml.read_text(encoding="utf-8")
                new_text = _re.sub(
                    r'(recordDeleteAfter:\s*)\S+',
                    rf'\g<1>{value}',
                    text,
                )
                if new_text != text:
                    yml.write_text(new_text, encoding="utf-8")
            human = self._retention_combo.currentText()
            QMessageBox.information(
                self, "Retention",
                f"Recording retention set to: {human}\n\n"
                "MediaMTX will delete segments older than this automatically.\n"
                "Restart MediaMTX for changes to take effect.",
            )
        except Exception as exc:
            QMessageBox.warning(self, "Retention", f"Failed to update config:\n{exc}")

    def _get_visible_camera_ids(self) -> set:
        """Return camera IDs that currently have an open widget."""
        ids: set = set()
        try:
            if self._app:
                from desktop.widgets.camera import CameraWidget
                for w in list(self._app.active_widgets):
                    if isinstance(w, CameraWidget) and getattr(w, "camera_id", None):
                        ids.add(w.camera_id)
        except Exception:
            pass
        return ids

    def _browse_rec_dir(self):
        path = QFileDialog.getExistingDirectory(
            self, "Select recording directory",
            self._rec_dir_edit.text() or ".",
        )
        if path:
            self._rec_dir_edit.setText(path)
            self._persist_rec_dir(path)

    def _apply_rec_dir_all(self):
        new_dir = self._rec_dir_edit.text().strip()
        if not new_dir:
            return
        self._persist_rec_dir(new_dir)
        self._rec_dir_edit.setEnabled(False)
        self._rec_busy = True
        visible_only = self._visible_only_chk.isChecked()
        visible_ids = self._get_visible_camera_ids() if visible_only else None
        scope = "visible cameras" if visible_only else "all cameras"
        self.rec_summary_lbl.setText(f"Applying to {scope}…")
        self.rec_summary_lbl.setStyleSheet("color: #3b82f6; font-weight: bold; font-size: 12px;")

        def _worker():
            ok = 0
            re_rec = 0
            total = 0
            try:
                r = requests.get("http://localhost:5000/api/cameras", timeout=3)
                cameras = r.json().get("data", [])
                for cam in cameras:
                    cid = cam.get("id", "")
                    if not cid:
                        continue
                    if visible_ids is not None and cid not in visible_ids:
                        continue
                    total += 1
                    try:
                        resp = requests.put(
                            f"http://localhost:5000/api/cameras/{cid}",
                            json={"recording_dir": new_dir}, timeout=3,
                        )
                        if resp.status_code == 200:
                            ok += 1
                        if cam.get("recording"):
                            requests.post(
                                f"http://localhost:5000/api/cameras/{cid}/recording",
                                json={"record": True}, timeout=5,
                            )
                            re_rec += 1
                    except Exception:
                        pass
            except Exception:
                pass
            msg = f"Dir set for {ok}/{total} cameras"
            if re_rec:
                msg += f", recording restarted on {re_rec}"
            self._rec_toggle_signal.emit(msg)
            try:
                self._rec_dir_edit.setEnabled(True)
            except Exception:
                pass

        threading.Thread(target=_worker, daemon=True).start()

    def _save_autostart_pref(self, name: str, checked: bool):
        if not self._app:
            return
        try:
            prefs = self._app._load_prefs()
            prefs[f"autostart_{name.lower()}"] = bool(checked)
            self._app._save_prefs(prefs)
        except Exception:
            pass

    def _preflight_check_recording(self, rec_dir: str) -> tuple[bool, str]:
        """Validate the chosen recording directory before kicking off Record All.

        Returns ``(proceed, message)``:
          * ``proceed=False`` -> fatal, abort and show *message*.
          * ``proceed=True`` with non-empty *message* -> warning, ask user
            to confirm before continuing.
          * ``proceed=True`` with empty message -> no problems, continue
            silently.
        """
        from pathlib import Path as _P
        import shutil as _sh

        path_str = (rec_dir or "").strip()
        if not path_str:
            try:
                from core.paths import get_recordings_dir
                path_str = str(get_recordings_dir())
            except Exception:
                return False, "No recording directory is configured."

        p = _P(path_str).expanduser()
        if not p.exists():
            try:
                p.mkdir(parents=True, exist_ok=True)
            except Exception as exc:
                return False, (
                    f"Recording directory does not exist and cannot be "
                    f"created:\n{p}\n\n{exc}"
                )

        try:
            test = p / ".knoxnet_write_test"
            test.write_text("ok")
            test.unlink()
        except Exception as exc:
            return False, (
                f"Recording directory is not writable:\n{p}\n\n"
                f"{exc}\n\nFix the permissions or choose a different folder."
            )

        try:
            usage = _sh.disk_usage(str(p))
            free_gb = usage.free / (1024 ** 3)
            used_pct = usage.used / max(usage.total, 1) * 100
        except Exception:
            return True, ""

        if free_gb < 1.0:
            return False, (
                f"Less than 1 GB free on the recording drive — refusing "
                f"to start continuous recording.\n\n"
                f"Path: {p}\n"
                f"Free: {free_gb:.2f} GB · Used: {used_pct:.0f}%\n\n"
                f"Free up space (or enable Disk Management auto-cleanup "
                f"below) and try again."
            )
        if free_gb < 5.0 or used_pct > 90:
            return True, (
                f"Low disk space on the recording drive.\n\n"
                f"Path: {p}\n"
                f"Free: {free_gb:.2f} GB · Used: {used_pct:.0f}%\n\n"
                f"Continuous recording can fill this disk quickly. Make "
                f"sure Disk Management auto-cleanup is enabled, or pick "
                f"a larger drive."
            )

        return True, ""

    def _toggle_recording_all(self, enable: bool):
        action = "Starting" if enable else "Stopping"

        rec_dir = self._rec_dir_edit.text().strip() if enable else ""

        # Pre-flight validation -- only relevant when starting recording.
        if enable:
            try:
                proceed, message = self._preflight_check_recording(rec_dir)
            except Exception as exc:
                proceed, message = False, f"Pre-flight check failed: {exc}"

            if not proceed:
                QMessageBox.warning(self, "Cannot Start Recording", message)
                self.rec_summary_lbl.setText("Recording aborted — see warning")
                self.rec_summary_lbl.setStyleSheet(
                    "color: #ef4444; font-weight: bold; font-size: 12px;"
                )
                return
            if message:
                reply = QMessageBox.question(
                    self,
                    "Disk Space Warning",
                    f"{message}\n\nStart recording anyway?",
                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                    QMessageBox.StandardButton.No,
                )
                if reply != QMessageBox.StandardButton.Yes:
                    self.rec_summary_lbl.setText("Recording cancelled")
                    self.rec_summary_lbl.setStyleSheet(
                        "color: #94a3b8; font-size: 12px;"
                    )
                    return

        self._rec_busy = True
        self._rec_enable = enable
        self.rec_all_btn.setEnabled(False)
        self.stop_rec_btn.setEnabled(False)
        self.rec_summary_lbl.setText(f"{action} recording…")
        self.rec_summary_lbl.setStyleSheet(
            "color: #3b82f6; font-weight: bold; font-size: 12px;"
        )

        if not hasattr(self, "_rec_result_connected"):
            self._rec_result_signal.connect(self._on_rec_toggle_done)
            self._rec_result_connected = True

        def _worker():
            ok_live: List[str] = []      # request OK and camera looks live
            ok_offline: List[str] = []   # request OK but camera is offline
            failed: List[tuple] = []     # (name, reason)

            try:
                r = requests.get("http://localhost:5000/api/cameras", timeout=4)
                cameras = r.json().get("data", []) or []
            except Exception as exc:
                self._rec_result_signal.emit({
                    "ok_live": [],
                    "ok_offline": [],
                    "failed": [("(backend)", f"Could not list cameras: {exc}")],
                    "total": 0,
                    "rec_dir": rec_dir,
                })
                return

            for cam in cameras:
                cid = cam.get("id", "")
                if not cid:
                    continue
                name = cam.get("name") or cid
                status = str(cam.get("status") or "").strip().lower()
                publishers = cam.get("publishers_count")
                try:
                    publishers = int(publishers) if publishers is not None else None
                except Exception:
                    publishers = None

                # A camera is "live" for recording purposes if either the
                # backend marks it live/online OR MediaMTX reports an active
                # publisher. Any other state means no stream is flowing in,
                # so MediaMTX has nothing to record even if record=yes.
                is_live = status in {"live", "online", "ready"} or (
                    publishers is not None and publishers > 0
                )

                try:
                    payload: dict = {"record": enable}
                    if enable and rec_dir:
                        payload["recording_dir"] = rec_dir
                    resp = requests.post(
                        f"http://localhost:5000/api/cameras/{cid}/recording",
                        json=payload,
                        timeout=8,
                    )
                    body = {}
                    try:
                        body = resp.json() or {}
                    except Exception:
                        body = {}
                    if body.get("success"):
                        if enable and not is_live:
                            ok_offline.append(name)
                        else:
                            ok_live.append(name)
                    else:
                        reason = (
                            body.get("message")
                            or f"HTTP {resp.status_code}"
                        )
                        failed.append((name, str(reason)))
                except Exception as exc:
                    failed.append((name, str(exc)))

            self._rec_result_signal.emit({
                "ok_live": ok_live,
                "ok_offline": ok_offline,
                "failed": failed,
                "total": len(ok_live) + len(ok_offline) + len(failed),
                "rec_dir": rec_dir,
            })

        threading.Thread(target=_worker, daemon=True).start()

    def _on_rec_toggle_done(self, result):
        """Render the structured Record All / Stop All result in the UI."""
        try:
            ok_live = list(result.get("ok_live") or [])
            ok_offline = list(result.get("ok_offline") or [])
            failed = list(result.get("failed") or [])
            total = int(result.get("total") or 0)
        except Exception:
            ok_live, ok_offline, failed, total = [], [], [], 0

        enable = bool(getattr(self, "_rec_enable", True))

        if enable:
            recording = len(ok_live)
            skipped = len(ok_offline)
            failures = len(failed)
            self.rec_summary_lbl.setText(
                f"Recording: {recording}/{total} cameras"
            )
            if failures:
                self.rec_summary_lbl.setStyleSheet(
                    "color: #ef4444; font-weight: bold; font-size: 12px;"
                )
            elif skipped:
                self.rec_summary_lbl.setStyleSheet(
                    "color: #f59e0b; font-weight: bold; font-size: 12px;"
                )
            elif recording > 0:
                self.rec_summary_lbl.setStyleSheet(
                    "color: #10b981; font-weight: bold; font-size: 12px;"
                )
            else:
                self.rec_summary_lbl.setStyleSheet(
                    "color: #94a3b8; font-size: 12px;"
                )

            if skipped or failures:
                lines: List[str] = []
                if recording:
                    lines.append(
                        f"Recording started on {recording} camera"
                        f"{'s' if recording != 1 else ''}:"
                    )
                    lines.extend(f"  • {n}" for n in ok_live)
                    lines.append("")
                if skipped:
                    lines.append(
                        f"{skipped} camera{'s' if skipped != 1 else ''} "
                        f"could not be recorded because no live stream is "
                        f"reaching MediaMTX (camera is offline or "
                        f"unreachable):"
                    )
                    lines.extend(f"  • {n}" for n in ok_offline)
                    lines.append("")
                if failures:
                    lines.append(
                        f"{failures} camera{'s' if failures != 1 else ''} "
                        f"failed to start recording:"
                    )
                    lines.extend(f"  • {n}: {r}" for n, r in failed)
                    lines.append("")
                lines.append(
                    "Bring offline cameras back online (check power, "
                    "network, and credentials), then click Record All "
                    "again to capture them."
                )
                box = QMessageBox(self)
                box.setIcon(
                    QMessageBox.Icon.Critical if failures
                    else QMessageBox.Icon.Warning
                )
                box.setWindowTitle("Record All")
                box.setText(
                    f"Recording is now active on {recording} of {total} "
                    f"cameras."
                )
                box.setInformativeText("\n".join(lines))
                box.exec()
        else:
            stopped = len(ok_live) + len(ok_offline)
            self.rec_summary_lbl.setText(
                f"Stopped recording on {stopped} camera"
                f"{'s' if stopped != 1 else ''}"
            )
            self.rec_summary_lbl.setStyleSheet(
                "color: #94a3b8; font-size: 12px;"
            )
            if failed:
                box = QMessageBox(self)
                box.setIcon(QMessageBox.Icon.Warning)
                box.setWindowTitle("Stop All")
                box.setText(
                    f"Could not stop recording on "
                    f"{len(failed)} camera"
                    f"{'s' if len(failed) != 1 else ''}."
                )
                box.setInformativeText(
                    "\n".join(f"  • {n}: {r}" for n, r in failed)
                )
                box.exec()

        QTimer.singleShot(4000, self._rec_toggle_finished)

    def _rec_toggle_finished(self):
        """Re-enable buttons and resume normal refresh after the result is shown."""
        self._rec_busy = False
        self.rec_all_btn.setEnabled(True)
        self.stop_rec_btn.setEnabled(True)
        self._refresh_recording_summary()

    def _refresh_recording_summary(self):
        if getattr(self, '_rec_busy', False):
            return
        try:
            if not self._app:
                return
            with self._app._recording_status_lock:
                cache = dict(self._app._recording_status_cache)
            total = len(cache)
            rec = sum(1 for v in cache.values() if v)
            if total:
                self.rec_summary_lbl.setText(f"Recording: {rec}/{total} cameras")
                if rec > 0:
                    self.rec_summary_lbl.setStyleSheet("color: #ef4444; font-weight: bold; font-size: 12px;")
                else:
                    self.rec_summary_lbl.setStyleSheet("color: #94a3b8; font-size: 12px;")
            else:
                self.rec_summary_lbl.setText("Recording: — (backend offline)")
        except Exception:
            pass

    # ── Disk Management helpers ──

    def _on_disk_protect_toggled(self, checked: bool):
        self._cleanup_threshold_spin.setEnabled(checked)
        self._cleanup_target_spin.setEnabled(checked)
        self._critical_threshold_spin.setEnabled(checked)

    def _load_disk_settings(self):
        """Pull current storage-manager settings from the backend API."""
        try:
            r = requests.get("http://localhost:5000/api/storage/settings", timeout=3)
            data = r.json().get("data", {})
            self._disk_protect_chk.setChecked(bool(data.get("enabled", True)))
            self._cleanup_threshold_spin.setValue(int(data.get("max_usage_percent", 85)))
            self._cleanup_target_spin.setValue(int(data.get("target_usage_percent", 75)))
            self._critical_threshold_spin.setValue(int(data.get("critical_usage_percent", 95)))
        except Exception:
            self._disk_protect_chk.setChecked(True)
            self._cleanup_threshold_spin.setValue(85)
            self._cleanup_target_spin.setValue(75)
            self._critical_threshold_spin.setValue(95)

    def _save_disk_settings(self):
        """Push updated storage-manager settings to the backend API."""
        payload = {
            "enabled": self._disk_protect_chk.isChecked(),
            "max_usage_percent": self._cleanup_threshold_spin.value(),
            "target_usage_percent": self._cleanup_target_spin.value(),
            "critical_usage_percent": self._critical_threshold_spin.value(),
        }
        try:
            r = requests.post(
                "http://localhost:5000/api/storage/settings",
                json=payload, timeout=3,
            )
            if r.json().get("success"):
                self._disk_status_msg.setText("Settings saved ✓")
                self._disk_status_msg.setStyleSheet("color: #10b981; font-size: 11px;")
                QTimer.singleShot(3000, self._refresh_disk_status)
            else:
                self._disk_status_msg.setText("Save failed")
                self._disk_status_msg.setStyleSheet("color: #ef4444; font-size: 11px;")
        except Exception as exc:
            self._disk_status_msg.setText(f"Error: {exc}")
            self._disk_status_msg.setStyleSheet("color: #ef4444; font-size: 11px;")

    def _refresh_disk_status(self):
        """Fetch disk usage from the backend (or locally) and update the drive bars."""
        def _worker():
            drives = self._gather_drive_info()
            self._disk_status_signal.updated.emit(drives)

        if not hasattr(self, '_disk_status_signal'):
            self._disk_status_signal = _DiskStatusSignal()
            self._disk_status_signal.updated.connect(self._apply_disk_status)

        threading.Thread(target=_worker, daemon=True).start()

    def _gather_drive_info(self) -> list:
        """Collect per-partition disk info for every directory we care about.

        Tries the backend API first (it knows about custom per-camera dirs).
        Falls back to local ``shutil.disk_usage`` so we always have real
        numbers even when the backend is down.
        """
        partitions: Dict[str, dict] = {}

        try:
            r = requests.get("http://localhost:5000/api/storage/status", timeout=3)
            data = r.json().get("data", {})
            for label, info in data.items():
                if not isinstance(info, dict) or info.get("error"):
                    continue
                path = info.get("path", "")
                if not path:
                    continue
                mount = self._resolve_mount_point(path)
                if mount in partitions:
                    partitions[mount]["labels"].append(label)
                    continue
                partitions[mount] = {
                    "mount": mount,
                    "path": path,
                    "labels": [label],
                    "used_pct": float(info.get("used_percent", 0)),
                    "used_gb": info.get("used_gb", 0),
                    "total_gb": info.get("total_gb", 0),
                    "free_gb": info.get("free_gb", 0),
                }
        except Exception:
            pass

        rec_dir = self._rec_dir_edit.text().strip()
        if rec_dir:
            self._ensure_local_partition(partitions, rec_dir, "recordings (selected)")
        try:
            from core.paths import get_recordings_dir
            self._ensure_local_partition(partitions, str(get_recordings_dir()), "recordings (default)")
        except Exception:
            pass

        self._ensure_local_partition(partitions, "/", "system")

        return list(partitions.values())

    @staticmethod
    def _resolve_mount_point(path: str) -> str:
        """Return the mount point for *path* (best-effort, falls back to '/')."""
        try:
            p = Path(path).resolve()
            while not p.is_mount() and p != p.parent:
                p = p.parent
            return str(p)
        except Exception:
            return "/"

    @staticmethod
    def _ensure_local_partition(partitions: dict, path: str, label: str) -> None:
        """Add a local disk_usage entry if this partition isn't already tracked."""
        try:
            p = Path(path).resolve()
            mount = SystemManagerDialog._resolve_mount_point(str(p))
            if mount in partitions:
                if label not in partitions[mount]["labels"]:
                    partitions[mount]["labels"].append(label)
                return
            usage = shutil.disk_usage(str(p))
            partitions[mount] = {
                "mount": mount,
                "path": str(p),
                "labels": [label],
                "used_pct": round(usage.used / max(usage.total, 1) * 100, 1),
                "used_gb": round(usage.used / (1 << 30), 2),
                "total_gb": round(usage.total / (1 << 30), 2),
                "free_gb": round(usage.free / (1 << 30), 2),
            }
        except Exception:
            pass

    def _apply_disk_status(self, drives: list):
        critical = self._critical_threshold_spin.value()
        threshold = self._cleanup_threshold_spin.value()

        # Remove old bar widgets beyond what we need
        while len(self._disk_bar_widgets) > len(drives):
            old = self._disk_bar_widgets.pop()
            old["row"].setParent(None)
            old["bar"].deleteLater()
            old["label"].deleteLater()
            old["detail"].deleteLater()
            old["row"].deleteLater()

        # Create new bar widgets if needed
        while len(self._disk_bar_widgets) < len(drives):
            row_widget = QWidget()
            row_layout = QVBoxLayout(row_widget)
            row_layout.setContentsMargins(0, 0, 0, 0)
            row_layout.setSpacing(1)

            lbl = QLabel("")
            lbl.setStyleSheet("color: #94a3b8; font-size: 11px; font-weight: bold;")
            row_layout.addWidget(lbl)

            bar_row = QHBoxLayout()
            bar_row.setSpacing(6)
            bar = QProgressBar()
            bar.setRange(0, 100)
            bar.setFixedHeight(16)
            bar.setTextVisible(True)
            bar_row.addWidget(bar, stretch=1)

            detail = QLabel("")
            detail.setStyleSheet("color: #94a3b8; font-size: 11px;")
            detail.setMinimumWidth(160)
            bar_row.addWidget(detail)

            row_layout.addLayout(bar_row)
            self._disk_bars_container.addWidget(row_widget)
            self._disk_bar_widgets.append({
                "row": row_widget,
                "label": lbl,
                "bar": bar,
                "detail": detail,
            })

        if not drives:
            self._disk_status_msg.setText("No disk information available")
            self._disk_status_msg.setStyleSheet("color: #ef4444; font-size: 11px;")
            return
        self._disk_status_msg.setText("")

        for i, drv in enumerate(drives):
            w = self._disk_bar_widgets[i]
            pct = float(drv.get("used_pct", 0))
            used_gb = drv.get("used_gb", "?")
            total_gb = drv.get("total_gb", "?")
            free_gb = drv.get("free_gb", "?")
            mount = drv.get("mount", "?")
            labels = drv.get("labels", [])
            label_str = ", ".join(labels)
            display_name = f"{mount}  ({label_str})" if labels else mount

            w["label"].setText(display_name)
            w["bar"].setValue(int(pct))
            w["bar"].setFormat(f"{pct:.1f}%")
            w["detail"].setText(f"{used_gb} / {total_gb} GB  ({free_gb} GB free)")

            if pct >= critical:
                w["bar"].setStyleSheet("QProgressBar::chunk { background-color: #dc2626; }")
                w["detail"].setStyleSheet("color: #dc2626; font-weight: bold; font-size: 11px;")
            elif pct >= threshold:
                w["bar"].setStyleSheet("QProgressBar::chunk { background-color: #f59e0b; }")
                w["detail"].setStyleSheet("color: #f59e0b; font-size: 11px;")
            else:
                w["bar"].setStyleSheet("QProgressBar::chunk { background-color: #10b981; }")
                w["detail"].setStyleSheet("color: #94a3b8; font-size: 11px;")

    def _refresh_health_bar(self):
        parts: List[str] = []
        try:
            import psutil
            cpu = psutil.cpu_percent(interval=0)
            mem = psutil.virtual_memory()
            disk = shutil.disk_usage(".")
            parts.append(f"CPU {cpu:.0f}%")
            parts.append(f"RAM {mem.percent:.0f}%")
            parts.append(f"Disk {disk.free / (1024**3):.0f} GB free")
        except Exception:
            pass
        self.health_lbl.setText("  ·  ".join(parts) if parts else "")

    # ----------------------------------------------------------------
    # Auto Protection panel
    # ----------------------------------------------------------------

    # Color hints for the level pill
    _LEVEL_COLORS = {
        "Normal":    ("#10b981", "white"),
        "Elevated":  ("#fbbf24", "#1e293b"),
        "High":      ("#f97316", "white"),
        "Critical":  ("#ef4444", "white"),
        "Emergency": ("#7f1d1d", "white"),
    }

    def _build_auto_protect_panel(self, root):
        """Construct the Auto Protection group.  Compact layout:
          row 1: enable + protect-recording checkboxes
          row 2: small level pill + machine summary (wraps)
          row 3: throttles line (wraps)
          row 4: 3 short slider rows (CPU / RAM / Swap)
          row 5: events log
        """
        box = QGroupBox("Auto Protection")
        box.setToolTip(
            "Monitors system load and progressively reduces features "
            "(live FPS, AI, then recording as a last resort) so the "
            "host doesn't crash under high CPU/RAM/swap pressure."
        )
        v = QVBoxLayout(box)
        v.setSpacing(4)
        v.setContentsMargins(8, 6, 8, 6)

        prefs = self._load_prefs()
        ap = (prefs.get("auto_protect") or {}) if isinstance(prefs, dict) else {}

        # Row 1: enable + protect recording (compact)
        row1 = QHBoxLayout()
        row1.setSpacing(8)
        self._ap_enable_chk = QCheckBox("Enable")
        self._ap_enable_chk.setStyleSheet("font-size: 11px;")
        self._ap_enable_chk.setChecked(bool(ap.get("enabled", True)))
        self._ap_enable_chk.toggled.connect(self._ap_save_prefs)
        row1.addWidget(self._ap_enable_chk)

        self._ap_protect_rec_chk = QCheckBox("Protect recording")
        self._ap_protect_rec_chk.setStyleSheet("font-size: 11px;")
        self._ap_protect_rec_chk.setToolTip(
            "When ON (default), recording is only stopped at EMERGENCY "
            "level, after every other feature has been throttled or paused."
        )
        self._ap_protect_rec_chk.setChecked(bool(ap.get("protect_recording", True)))
        self._ap_protect_rec_chk.toggled.connect(self._ap_save_prefs)
        row1.addWidget(self._ap_protect_rec_chk)

        self._ap_level_pill = QLabel("Normal")
        self._ap_level_pill.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._ap_level_pill.setMinimumWidth(70)
        self._ap_level_pill.setStyleSheet(
            "background-color: #10b981; color: white; "
            "font-weight: bold; padding: 2px 6px; border-radius: 4px; font-size: 11px;"
        )
        row1.addStretch()
        row1.addWidget(self._ap_level_pill)
        v.addLayout(row1)

        # Row 1b: action buttons (Reset / Defaults)
        action_row = QHBoxLayout()
        action_row.setSpacing(6)
        action_row.setContentsMargins(0, 0, 0, 0)
        reset_btn = QPushButton("Force Normal")
        reset_btn.setFixedHeight(22)
        reset_btn.setStyleSheet("font-size: 10px;")
        reset_btn.setToolTip(
            "Immediately drop the protection level to Normal and "
            "restore all throttled features.  Use if you think the "
            "shedder fired by mistake."
        )
        reset_btn.clicked.connect(self._ap_force_normal)
        action_row.addWidget(reset_btn)

        defaults_btn = QPushButton("Reset Defaults")
        defaults_btn.setFixedHeight(22)
        defaults_btn.setStyleSheet("font-size: 10px;")
        defaults_btn.setToolTip(
            "Discard your slider overrides and restore the "
            "auto-tuned defaults for this machine."
        )
        defaults_btn.clicked.connect(self._ap_reset_defaults)
        action_row.addWidget(defaults_btn)
        action_row.addStretch()
        v.addLayout(action_row)

        # Row 2: machine class / current metrics (single wrapped line)
        self._ap_class_lbl = QLabel("Machine: …")
        self._ap_class_lbl.setStyleSheet("color: #94a3b8; font-size: 10px;")
        self._ap_class_lbl.setWordWrap(True)
        v.addWidget(self._ap_class_lbl)

        # Row 3: reason (only visible when not Normal)
        self._ap_reason_lbl = QLabel("")
        self._ap_reason_lbl.setStyleSheet("color: #cbd5e1; font-size: 10px;")
        self._ap_reason_lbl.setWordWrap(True)
        self._ap_reason_lbl.hide()
        v.addWidget(self._ap_reason_lbl)

        # Row 4: throttles (wrapped)
        self._ap_throttles_lbl = QLabel("Throttles: none")
        self._ap_throttles_lbl.setStyleSheet("color: #94a3b8; font-size: 10px;")
        self._ap_throttles_lbl.setWordWrap(True)
        v.addWidget(self._ap_throttles_lbl)

        # Row 5: 3 compact slider rows (CPU / RAM / Swap)
        existing_thresh = ap.get("thresholds") or {}

        def _slider_row(parent_layout, key: str, short_label: str, default_val: int):
            row = QHBoxLayout()
            row.setSpacing(6)
            row.setContentsMargins(0, 0, 0, 0)
            lbl = QLabel(short_label)
            lbl.setFixedWidth(34)
            lbl.setStyleSheet("color: #94a3b8; font-size: 10px;")
            row.addWidget(lbl)
            slider = QSlider(Qt.Orientation.Horizontal)
            slider.setMinimum(40)
            slider.setMaximum(99)
            initial = int(existing_thresh.get(key) or default_val)
            slider.setValue(int(max(40, min(99, initial))))
            slider.setMinimumWidth(80)
            row.addWidget(slider, stretch=1)
            val_lbl = QLabel(f"{slider.value()}%")
            val_lbl.setFixedWidth(36)
            val_lbl.setStyleSheet("color: #e2e8f0; font-size: 10px;")
            val_lbl.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            row.addWidget(val_lbl)
            slider.valueChanged.connect(
                lambda v, lbl=val_lbl: lbl.setText(f"{int(v)}%")
            )
            slider.sliderReleased.connect(self._ap_save_prefs)
            parent_layout.addLayout(row)
            return slider

        sliders_lbl = QLabel("Start protecting at…")
        sliders_lbl.setStyleSheet("color: #94a3b8; font-size: 10px;")
        v.addWidget(sliders_lbl)
        self._ap_slider_cpu = _slider_row(v, "elevated_cpu", "CPU", 80)
        self._ap_slider_ram = _slider_row(v, "elevated_ram", "RAM", 82)
        self._ap_slider_swap = _slider_row(v, "critical_swap", "Swap", 35)

        # Recent events log (smaller)
        evt_lbl = QLabel("Recent events:")
        evt_lbl.setStyleSheet("color: #94a3b8; font-size: 10px;")
        v.addWidget(evt_lbl)
        self._ap_events_text = QTextEdit()
        self._ap_events_text.setReadOnly(True)
        self._ap_events_text.setFixedHeight(56)
        self._ap_events_text.setLineWrapMode(QTextEdit.LineWrapMode.NoWrap)
        self._ap_events_text.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self._ap_events_text.setStyleSheet(
            "background: #0f172a; color: #cbd5e1; "
            "font-family: monospace; font-size: 9px;"
        )
        self._ap_events_text.setPlainText("(none)")
        v.addWidget(self._ap_events_text)

        root.addWidget(box)

    def _ap_force_normal(self):
        """User-triggered: immediately drop the shedder back to NORMAL
        and restore all features.  Used when the user thinks the
        shedder fired by mistake (e.g. transient stall)."""
        if not self._app:
            return
        try:
            if hasattr(self._app, "force_load_shedder_normal"):
                self._app.force_load_shedder_normal()
        except Exception:
            pass
        try:
            self._refresh_auto_protect_panel()
        except Exception:
            pass

    def _ap_reset_defaults(self):
        """Wipe user threshold overrides and re-apply auto-tier defaults."""
        if not self._app:
            return
        try:
            prefs = self._app._load_prefs()
            ap = dict(prefs.get("auto_protect") or {})
            ap["thresholds"] = {}
            ap["thresholds_version"] = 2
            prefs["auto_protect"] = ap
            self._app._save_prefs(prefs)
            try:
                if hasattr(self._app, "reload_load_shedder_prefs"):
                    self._app.reload_load_shedder_prefs()
            except Exception:
                pass
            # Refresh the slider positions to reflect the new defaults
            try:
                if hasattr(self._app, "get_load_shedder_summary"):
                    summary = self._app.get_load_shedder_summary()
                    thresh = summary.get("thresholds") or {}
                    self._ap_slider_cpu.setValue(int(thresh.get("elevated_cpu", 85)))
                    self._ap_slider_ram.setValue(int(thresh.get("elevated_ram", 85)))
                    self._ap_slider_swap.setValue(int(thresh.get("critical_swap", 40)))
            except Exception:
                pass
            QMessageBox.information(
                self, "Auto Protection",
                "Thresholds reset to auto-tuned defaults for your machine.",
            )
        except Exception as exc:
            QMessageBox.warning(
                self, "Auto Protection",
                f"Failed to reset defaults: {exc}",
            )

    def _ap_save_prefs(self):
        """Persist Auto Protection settings to prefs and notify the
        running shedder so it picks them up immediately.
        """
        if not self._app:
            return
        try:
            prefs = self._app._load_prefs()
            ap = dict(prefs.get("auto_protect") or {})
            ap["enabled"] = bool(self._ap_enable_chk.isChecked())
            ap["protect_recording"] = bool(self._ap_protect_rec_chk.isChecked())
            thresh = dict(ap.get("thresholds") or {})
            cpu_v = int(self._ap_slider_cpu.value())
            ram_v = int(self._ap_slider_ram.value())
            swap_v = int(self._ap_slider_swap.value())
            # User edits the "elevated" floors; auto-derive higher levels
            # at sensible deltas so users only see one slider per resource.
            # The deltas were tightened so users on weaker boxes who
            # raise the floor to 90% don't accidentally make EMERGENCY
            # unreachable -- it's clamped to 99 either way.
            thresh["elevated_cpu"] = cpu_v
            thresh["high_cpu"] = min(99, cpu_v + 8)
            thresh["critical_cpu"] = min(99, cpu_v + 14)
            thresh["elevated_ram"] = ram_v
            thresh["high_ram"] = min(99, ram_v + 8)
            thresh["critical_ram"] = min(99, ram_v + 12)
            thresh["emergency_ram"] = min(99, max(ram_v + 16, 96))
            thresh["critical_swap"] = swap_v
            thresh["emergency_swap"] = min(99, swap_v + 30)
            ap["thresholds"] = thresh
            prefs["auto_protect"] = ap
            self._app._save_prefs(prefs)
            try:
                if hasattr(self._app, "reload_load_shedder_prefs"):
                    self._app.reload_load_shedder_prefs()
            except Exception:
                pass
        except Exception:
            pass

    def _refresh_auto_protect_panel(self):
        """Periodic update of the level pill, throttles, and event log."""
        if not self._app or not hasattr(self._app, "get_load_shedder_summary"):
            return
        try:
            summary = self._app.get_load_shedder_summary()
        except Exception:
            return

        try:
            label = str(summary.get("level_label") or "Normal")
            self._ap_level_pill.setText(label)
            bg, fg = self._LEVEL_COLORS.get(label, ("#64748b", "white"))
            self._ap_level_pill.setStyleSheet(
                f"background-color: {bg}; color: {fg}; "
                "font-weight: bold; padding: 4px 8px; border-radius: 6px;"
            )
        except Exception:
            pass

        try:
            reason = str(summary.get("reason") or "")
            label = str(summary.get("level_label") or "Normal")
            warmup = float(summary.get("warmup_remaining_sec") or 0.0)
            cand = summary.get("candidate_level")
            gate = float(summary.get("candidate_gate_remaining") or 0.0)

            parts: List[str] = []
            if warmup > 0.5:
                parts.append(f"Warmup: {warmup:.0f}s remaining (no shedding yet)")
            elif cand is not None and hasattr(cand, "label") and cand.label != label:
                # Candidate differs from committed level -> show gate countdown
                if gate > 0.5:
                    parts.append(f"Evaluating: {cand.label} (commits in {gate:.0f}s)")
                else:
                    parts.append(f"Evaluating: {cand.label}")
            if label != "Normal" and reason:
                parts.append(reason)

            if parts:
                self._ap_reason_lbl.setText(" · ".join(parts))
                self._ap_reason_lbl.show()
            else:
                self._ap_reason_lbl.hide()
        except Exception:
            pass

        try:
            mc = str(summary.get("machine_class") or "mid")
            metrics = summary.get("metrics")
            mtxt = ""
            if metrics is not None:
                mtxt = (
                    f"  CPU {float(getattr(metrics,'cpu_percent',0)):.0f}%"
                    f" / RAM {float(getattr(metrics,'ram_percent',0)):.0f}%"
                    f" / Swap {float(getattr(metrics,'swap_percent',0)):.0f}%"
                )
            try:
                import psutil
                cores = psutil.cpu_count(logical=False) or psutil.cpu_count(logical=True) or 0
                ram_gb = float(psutil.virtual_memory().total) / (1024.0 ** 3)
                self._ap_class_lbl.setText(
                    f"{mc.title()} tier ({cores}c/{ram_gb:.0f}GB){mtxt}"
                )
            except Exception:
                self._ap_class_lbl.setText(f"{mc}{mtxt}")
        except Exception:
            pass

        try:
            throttles = summary.get("throttles") or {}
            if not throttles:
                self._ap_throttles_lbl.setText("Throttles: none")
            else:
                parts = []
                if "paint_fps" in throttles:
                    parts.append(f"paint {int(throttles['paint_fps'])}")
                if "motion_fps" in throttles:
                    parts.append(f"motion {int(throttles['motion_fps'])}")
                if "detector_fps" in throttles:
                    v = int(throttles["detector_fps"])
                    parts.append("det off" if v == 0 else f"det {v}")
                if "depth_fps" in throttles:
                    v = int(throttles["depth_fps"])
                    parts.append("depth off" if v == 0 else f"depth {v}")
                self._ap_throttles_lbl.setText("Throttles: " + " / ".join(parts))
        except Exception:
            pass

        try:
            events = summary.get("events") or []
            if not events:
                self._ap_events_text.setPlainText("(none)")
            else:
                lines = [e.format_line() for e in events if hasattr(e, "format_line")]
                self._ap_events_text.setPlainText("\n".join(lines) if lines else "(none)")
        except Exception:
            pass

    def _sync_updates_status_old(self) -> None:
        """Legacy updater sync -- widgets removed from compact view."""
        pass

    def _backend_python_cmd(self) -> List[str]:
        """
        Choose the interpreter used when the Desktop UI starts the backend.

        Many Windows installs have multiple Python versions; launching the backend with the
        wrong one can miss dependencies (e.g. qrcode for WireGuard QR generation).

        Priority:
        - KNOXNET_BACKEND_PYTHON: absolute path to python.exe to use
        - sys.executable if it looks like Python 3.11
        - Windows py launcher: py -3.11 (if available)
        - sys.executable (fallback)
        """
        override = str(os.environ.get("KNOXNET_BACKEND_PYTHON") or "").strip()
        if override:
            return [override]

        exe = str(getattr(sys, "executable", "") or "").strip()
        exe_l = exe.lower()
        if "python311" in exe_l:
            return [exe]

        if os.name == "nt":
            py = shutil.which("py")
            if py:
                return [py, "-3.11"]

        return [exe] if exe else ["python"]

    def _prime_psutil(self) -> None:
        if PSUTIL_AVAILABLE:
            try:
                psutil.Process().cpu_percent(interval=None)
            except Exception:
                pass

    def _read_proc_stats(self) -> _ProcStats:
        st = _ProcStats()
        if not PSUTIL_AVAILABLE:
            return st
        try:
            p = psutil.Process()
            st.cpu_percent = float(psutil.cpu_percent(interval=None))
            st.mem_percent = float(psutil.virtual_memory().percent)
            st.rss_mb = float(p.memory_info().rss) / (1024.0 * 1024.0)
        except Exception:
            return st
        return st

    def _check_service_health(self, name: str):
        widgets = self.service_widgets[name]
        url = widgets["health_url"]
        is_up = False
        
        # Try localhost first (most reliable on Windows for services), then 127.0.0.1
        urls = [url.replace("127.0.0.1", "localhost"), url]
        
        for check_url in urls:
            try:
                if "MediaMTX" in name:
                    from requests.auth import HTTPBasicAuth
                    username = (os.environ.get("MEDIAMTX_API_USERNAME") or "").strip()
                    password = (os.environ.get("MEDIAMTX_API_PASSWORD") or "").strip()
                    auth = HTTPBasicAuth(username, password) if username and password else None
                    res = requests.get(check_url, timeout=2.0, auth=auth)
                else:
                    res = requests.get(check_url, timeout=2.0)
                
                # Consider it "up" if we get any response (even 401/404 means server is alive)
                if res.status_code < 500:
                    is_up = True
                    break
            except:
                # Fallback to simple socket check if HTTP fails but server might be booting
                try:
                    from urllib.parse import urlparse
                    import socket
                    parsed = urlparse(check_url)
                    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                        s.settimeout(1.0)
                        if s.connect_ex((parsed.hostname, parsed.port)) == 0:
                            is_up = True
                            break
                except:
                    continue
        
        status_str = "ONLINE" if is_up else "OFFLINE"
        
        # If we have a tracked process but health check failed, it's still STARTING
        if not is_up and name in self._processes:
            # Check if process is actually still alive
            if self._processes[name].poll() is None:
                status_str = "STARTING"
            else:
                # Process crashed or stopped
                self._processes.pop(name)
        
        # Emit signal to update UI safely
        self.status_updated.emit(name, status_str)

    def _on_status_updated(self, name: str, status: str):
        widgets = self.service_widgets.get(name)
        if not widgets:
            return
            
        status_lbl = widgets["status_lbl"]
        start_btn = widgets["start_btn"]
        stop_btn = widgets["stop_btn"]
        
        if status == "ONLINE":
            status_lbl.setText("ONLINE")
            status_lbl.setStyleSheet("color: #10b981; font-weight: bold;")
            start_btn.setEnabled(False)
            stop_btn.setEnabled(True)
        elif status == "STARTING":
            status_lbl.setText("STARTING...")
            status_lbl.setStyleSheet("color: #3b82f6; font-weight: bold;")
            start_btn.setEnabled(False)
            stop_btn.setEnabled(True)
        else:
            status_lbl.setText("OFFLINE")
            status_lbl.setStyleSheet("color: #ef4444; font-weight: bold;")
            start_btn.setEnabled(widgets["entry_point"] is not None)
            # Enable stop button if we have a port or a recorded process
            stop_btn.setEnabled(True) 

        # Gate "Web UI (Paid)" behind entitlement (best-effort).
        # - If backend is online and tier != free: allow user to toggle.
        # - Otherwise: force OFF and disable the checkbox.
        try:
            if name == "Backend" and hasattr(self, "web_ui_toggle"):
                if status == "ONLINE":
                    threading.Thread(target=self._refresh_web_ui_entitlement, daemon=True).start()
                else:
                    self._set_web_ui_toggle_state(allowed=False, enabled=False)
        except Exception:
            pass

    def _set_web_ui_toggle_state(self, *, allowed: bool, enabled: bool) -> None:
        try:
            if not hasattr(self, "web_ui_toggle"):
                return
            # Avoid recursive toggled() firing when forcing state.
            self.web_ui_toggle.blockSignals(True)
            try:
                self.web_ui_toggle.setChecked(bool(enabled) if bool(allowed) else False)
            finally:
                self.web_ui_toggle.blockSignals(False)
            self.web_ui_toggle.setEnabled(bool(allowed))
        except Exception:
            pass

    def _refresh_web_ui_entitlement(self) -> None:
        """
        Query backend entitlement and (if supported) backend web-ui state.
        Safe default: Web UI OFF unless explicitly entitled.
        """
        try:
            # Entitlement: treat anything not explicitly paid as free.
            tier = "free"
            try:
                r = requests.get("http://localhost:5000/api/system/entitlement", timeout=1.0)
                if r.ok:
                    payload = r.json() or {}
                    data = payload.get("data") if isinstance(payload, dict) else {}
                    if isinstance(data, dict):
                        tier = str(data.get("tier") or "free").strip().lower() or "free"
            except Exception:
                tier = "free"

            allowed = tier not in {"free"}
            if not allowed:
                self._set_web_ui_toggle_state(allowed=False, enabled=False)
                return

            # If backend supports it, load current web-ui enabled state.
            enabled = bool(self.web_ui_toggle.isChecked())
            try:
                r2 = requests.get("http://localhost:5000/api/system/web-ui", timeout=1.0)
                if r2.ok:
                    payload2 = r2.json() or {}
                    data2 = payload2.get("data") if isinstance(payload2, dict) else {}
                    if isinstance(data2, dict):
                        enabled = bool(data2.get("enabled", enabled))
            except Exception:
                pass

            self._set_web_ui_toggle_state(allowed=True, enabled=enabled)
        except Exception:
            self._set_web_ui_toggle_state(allowed=False, enabled=False)

    def _updates_check(self):
        try:
            if self._app and hasattr(self._app, "_updates_check_now"):
                self._app._updates_check_now(user_initiated=True)
                self._sync_updates_status()
            else:
                self.update_status_lbl.setText("Status: updater unavailable")
        except Exception as e:
            self.update_status_lbl.setText("Status: failed")
            QMessageBox.warning(self, "Updates", f"Update check failed: {e}")

    def _updates_install(self):
        try:
            if self._app and hasattr(self._app, "_updates_install_latest"):
                self._app._updates_install_latest()
                self._sync_updates_status()
            else:
                self.update_status_lbl.setText("Status: updater unavailable")
        except Exception as e:
            self.update_status_lbl.setText("Status: failed")
            QMessageBox.warning(self, "Updates", f"Install failed: {e}")

    def _updates_apply(self):
        try:
            if self._app and hasattr(self._app, "_updates_apply_staged_and_restart"):
                self._app._updates_apply_staged_and_restart()
                self._sync_updates_status()
            else:
                self.update_status_lbl.setText("Status: updater unavailable")
        except Exception as e:
            self.update_status_lbl.setText("Status: failed")
            QMessageBox.warning(self, "Updates", f"Apply failed: {e}")

    def _start_service(self, name: str, entry_point: str):
        if not entry_point:
            return
            
        frozen = bool(getattr(sys, "frozen", False))

        # Repo root is meaningful only in dev mode. In frozen mode, prefer sys._MEIPASS to locate bundled resources.
        if frozen:
            repo_root = Path(sys.executable).resolve().parent
            internal_root = Path(getattr(sys, "_MEIPASS", repo_root / "_internal")).resolve()
            entry_base = internal_root
        else:
            repo_root = Path(self._app._repo_root() if self._app else ".").resolve()
            internal_root = repo_root / "_internal"
            entry_base = repo_root

        entry_path = entry_base / entry_point
        
        # In frozen builds, some "entry points" are logical (run via flags) rather than files.
        is_logical = frozen and entry_point in {"app.py"}
        if (
            not is_logical
            and entry_point != "__vite_dev__"
            and not entry_point.startswith("__py_module__:")
            and not entry_path.exists()
        ):
            QMessageBox.critical(self, "Error", f"Service entry point not found: {entry_path}")
            return

        # Pre-cleanup: ensure ports are clear before starting
        try:
            widgets = self.service_widgets.get(name)
            if widgets:
                from urllib.parse import urlparse
                port = urlparse(widgets["health_url"]).port
                if port:
                    self._cleanup_port(port)
                    if "Backend" in name:
                        self._cleanup_port(8765)
        except:
            pass

        try:
            env = dict(os.environ)
            env["PYTHONIOENCODING"] = "utf-8"
            
            # Match the terminal's backend startup environment
            if "Backend" in name:
                env["KNOXNET_SIMPLE_SERVER"] = "1"
                if hasattr(self, "web_ui_toggle") and not self.web_ui_toggle.isChecked():
                    env["KNOXNET_WEB_UI_DISABLED"] = "1"
            
            cmd = []
            # Never default to a read-only mount (AppImage). Use a per-user writable dir for cwd.
            cwd = str(_default_state_dir() if frozen else repo_root)
            
            # Prefer running packaged services via the frozen executable entrypoints.
            # This is more reliable than locating .py files or shell scripts under _internal.
            if frozen:
                if name == "Local Vision":
                    env.setdefault("LOCAL_VISION_HOST", env.get("LOCAL_VISION_HOST", "0.0.0.0"))
                    env.setdefault("LOCAL_VISION_PORT", env.get("LOCAL_VISION_PORT", "8101"))
                    cmd = [sys.executable, "--run-vision-local"]
                elif name == "Local LLM":
                    env.setdefault("LLM_HOST", env.get("LLM_HOST", "127.0.0.1"))
                    env.setdefault("LLM_PORT", env.get("LLM_PORT", "8102"))
                    cmd = [sys.executable, "--run-llm-local"]

            if cmd:
                # fall through to logging + Popen
                pass
            elif entry_point == "__vite_dev__":
                if frozen:
                    QMessageBox.critical(self, "Error", "Web UI (Vite) is disabled in packaged builds.")
                    return
                # Start Vite dev server (React UI)
                npm = shutil.which("npm") or shutil.which("npm.cmd") or shutil.which("pnpm") or shutil.which("yarn")
                if not npm:
                    QMessageBox.critical(self, "Error", "Node package manager not found (npm/pnpm/yarn). Install Node.js, then retry.")
                    return
                # Ensure port is clear (health check above already tries)
                env.setdefault("BROWSER", "none")
                cmd = [
                    npm,
                    "run",
                    "dev",
                    "--",
                    "--host",
                    "0.0.0.0",
                    "--port",
                    "5173",
                    "--strictPort",
                ]
            elif entry_point.startswith("__py_module__:"):
                module_name = entry_point.split(":", 1)[1].strip()
                if not module_name:
                    QMessageBox.critical(self, "Error", f"Missing python module for {name}")
                    return
                if frozen:
                    QMessageBox.critical(self, "Error", f"Cannot run python modules via -m in packaged builds: {module_name}")
                    return
                cmd = [sys.executable, "-m", module_name]
            elif entry_point.endswith(".py"):
                if frozen and entry_point == "app.py" and "Backend" in name:
                    # Run the backend from the frozen executable entrypoint.
                    cmd = [sys.executable, "--run-backend"]
                else:
                    if "Backend" in name:
                        cmd = [*self._backend_python_cmd(), str(entry_path)]
                    else:
                        cmd = [sys.executable, str(entry_path)]
            elif entry_point.endswith(".bat"):
                cmd = [str(entry_path)]
            elif entry_point.endswith(".sh"):
                # Always run via bash so the script doesn't need the executable bit set.
                cmd = ["bash", str(entry_path), "start"]
            elif entry_point.endswith(".exe"):
                if os.name != "nt":
                    QMessageBox.critical(self, "Error", f"Cannot run Windows .exe on this OS: {entry_path}")
                    return
                cmd = [str(entry_path)]
                if "mediamtx" in entry_point.lower():
                    cwd = str(entry_path.parent)
                    compat = entry_path.parent / "mediamtx_compat.yml"
                    cfg = compat if compat.exists() else entry_path.parent / "mediamtx.yml"
                    env["MTX_CONFIG_PATH"] = str(cfg)
                    cmd = [str(entry_path), str(cfg)]
            elif entry_path.is_file():
                # Allow running non-extension binaries on Linux (e.g., mediamtx)
                cmd = [str(entry_path)]
                if "mediamtx" in entry_point.lower():
                    cwd = str(entry_path.parent)
                    compat = entry_path.parent / "mediamtx_compat.yml"
                    cfg = compat if compat.exists() else entry_path.parent / "mediamtx.yml"
                    env["MTX_CONFIG_PATH"] = str(cfg)
                    cmd = [str(entry_path), str(cfg)]
            
            creationflags = 0
            if os.name == "nt":
                creationflags = subprocess.CREATE_NO_WINDOW
            
            if not cmd:
                QMessageBox.critical(self, "Error", f"Failed to build start command for {name} ({entry_path})")
                return

            # Log output to a file so we can debug failures
            log_dir = _writable_log_dir(repo_root=repo_root)
            log_dir.mkdir(parents=True, exist_ok=True)
            log_file = log_dir / f"service_{name.lower().replace(' ', '_').replace('(', '').replace(')', '')}.log"
            
            with open(log_file, "a", encoding="utf-8") as f:
                f.write(f"\n--- Starting {name} at {time.ctime()} ---\n")
                try:
                    f.write(f"CMD: {' '.join([str(x) for x in cmd])}\n")
                except Exception:
                    pass
                f.flush()
                
                proc = subprocess.Popen(
                    cmd,
                    cwd=cwd,
                    env=env,
                    stdout=f,
                    stderr=subprocess.STDOUT,
                    creationflags=creationflags
                )
            
            self._processes[name] = proc
            QMessageBox.information(self, "Service Started", f"Attempting to start {name}...\nLogs: {log_file}")

            # If the user wants the Web UI, bring up the Vite server as part of "Backend".
            try:
                if (
                    (not frozen)
                    and "Backend" in name
                    and hasattr(self, "web_ui_toggle")
                    and bool(self.web_ui_toggle.isChecked())
                ):
                    widgets2 = self.service_widgets.get("Web UI (Vite)")
                    if widgets2 and widgets2["status_lbl"].text() != "ONLINE":
                        # Best-effort; keep it non-blocking.
                        threading.Thread(
                            target=lambda: self._start_service("Web UI (Vite)", widgets2["entry_point"]),
                            daemon=True,
                        ).start()
            except Exception:
                pass
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to start {name}: {e}")

    def _stop_service(self, name: str):
        # If Backend is being stopped, also stop the Web UI dev server (best-effort).
        try:
            if "Backend" in name:
                self._stop_service("Web UI (Vite)")
        except Exception:
            pass

        # 1. If it's a process we started, terminate it
        if name in self._processes:
            proc = self._processes.pop(name)
            try:
                proc.terminate()
                proc.wait(timeout=2)
                QMessageBox.information(self, "Service Stopped", f"Stopped {name}.")
                return
            except:
                try:
                    proc.kill()
                    QMessageBox.information(self, "Service Stopped", f"Stopped {name} (forced).")
                    return
                except:
                    pass

        # 1b. If this is a script-managed service, attempt an explicit stop (best-effort).
        try:
            widgets0 = self.service_widgets.get(name)
            ep = str((widgets0 or {}).get("entry_point") or "")
            if ep.endswith(".sh"):
                frozen = bool(getattr(sys, "frozen", False))
                if frozen:
                    repo_root = Path(sys.executable).resolve().parent
                    internal_root = Path(getattr(sys, "_MEIPASS", repo_root / "_internal")).resolve()
                    ep_path = internal_root / ep
                else:
                    repo_root = Path(self._app._repo_root() if self._app else ".").resolve()
                    ep_path = repo_root / ep
                if ep_path.exists():
                    subprocess.Popen(["bash", str(ep_path), "stop"], cwd=str(repo_root))
                    # Fall through to port-based cleanup as a backup.
        except Exception:
            pass

        # 2. Try to find and kill by port if it's one of our known services
        widgets = self.service_widgets.get(name)
        if not widgets:
            return

        url = widgets["health_url"]
        try:
            from urllib.parse import urlparse
            parsed = urlparse(url)
            port = parsed.port
            if not port:
                return

            if not PSUTIL_AVAILABLE:
                QMessageBox.warning(self, "Notice", "psutil not available; cannot stop by port.")
                return

            # For the Backend, also clear the WebSocket port (8765)
            target_ports = [port]
            if "Backend" in name:
                target_ports.append(8765)

            found = False
            for p in target_ports:
                if self._cleanup_port(p):
                    found = True
            
            if found:
                QMessageBox.information(self, "Service Stopped", f"Stopped {name} by port(s) {', '.join(map(str, target_ports))}.")
            else:
                QMessageBox.warning(self, "Notice", f"Could not find process for {name} on port(s) {', '.join(map(str, target_ports))}.")
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Failed to stop {name}: {e}")

    def _cleanup_port(self, port: int) -> bool:
        """Find and kill any process using the specified port."""
        if not PSUTIL_AVAILABLE:
            return False
        
        import psutil
        found = False
        for proc in psutil.process_iter(['pid', 'name']):
            try:
                for conn in proc.connections(kind='inet'):
                    if conn.laddr.port == port:
                        proc.terminate()
                        try:
                            proc.wait(timeout=1)
                        except:
                            proc.kill()
                        found = True
                        break
                if found: break
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                continue
        return found

    def _start_docker_all(self):
        if self._app and hasattr(self._app, "_run_docker_compose_async"):
            self._app._run_docker_compose_async(
                ["up", "-d"],
                success_msg="Docker services starting...",
                fail_msg="Failed to start Docker services."
            )

    def _save_web_ui_pref(self, checked: bool):
        if not self._app: return
        try:
            prefs = self._app._load_prefs()
            prefs["web_ui_enabled"] = checked
            self._app._save_prefs(prefs)

            # If Web UI is enabled, ensure the frontend dev server is running.
            # If disabled, stop it (best-effort).
            try:
                if checked:
                    widgets = self.service_widgets.get("Web UI (Vite)")
                    if widgets and widgets["status_lbl"].text() != "ONLINE":
                        self._start_service("Web UI (Vite)", widgets["entry_point"])
                else:
                    self._stop_service("Web UI (Vite)")
            except Exception:
                pass
            
            # If backend is online, attempt live toggle
            widgets = self.service_widgets.get("Backend")
            if widgets:
                status_lbl = widgets["status_lbl"]
                if status_lbl.text() == "ONLINE":
                    def _live_toggle():
                        try:
                            requests.post("http://localhost:5000/api/system/web-ui", json={"enabled": checked}, timeout=1.0)
                        except:
                            pass
                    threading.Thread(target=_live_toggle, daemon=True).start()
        except Exception as e:
            print(f"DEBUG: Failed to save web UI pref: {e}")

    def _stop_docker_all(self):
        if self._app and hasattr(self._app, "_run_docker_compose_async"):
            self._app._run_docker_compose_async(
                ["down"],
                success_msg="Docker services stopping...",
                fail_msg="Failed to stop Docker services."
            )

    def refresh(self):
        for name in self.service_widgets:
            threading.Thread(target=self._check_service_health, args=(name,), daemon=True).start()

        self._refresh_recording_summary()
        self._refresh_disk_status()
        self._refresh_health_bar()
        try:
            self._refresh_auto_protect_panel()
        except Exception:
            pass

        # Update top-level status label
        online = sum(
            1 for w in self.service_widgets.values()
            if w["status_lbl"].text() == "ONLINE"
        )
        total = len(self.service_widgets)
        if online == total:
            self.status_lbl.setText("All services online")
            self.status_lbl.setStyleSheet("font-weight: bold; font-size: 13px; color: #10b981;")
        elif online == 0:
            self.status_lbl.setText("All services offline")
            self.status_lbl.setStyleSheet("font-weight: bold; font-size: 13px; color: #ef4444;")
        else:
            self.status_lbl.setText(f"{online}/{total} services online")
            self.status_lbl.setStyleSheet("font-weight: bold; font-size: 13px; color: #f59e0b;")

        # Stats
        try:
            st = self._read_proc_stats()
            if st.cpu_percent is not None:
                self.proc_lbl.setText(f"CPU {st.cpu_percent:.0f}%  RAM {st.mem_percent:.0f}%")
        except Exception:
            pass

    # ── Compat stubs for features moved out of the main view ──
    # These keep the rest of the methods (WireGuard, sessions, cameras) from
    # crashing if they reference widgets that no longer exist in the UI.

    def _sync_updates_status(self) -> None:
        pass

    # --- Remote Access (WireGuard) helpers ---------------------------------

    def _ra_api(self, path: str) -> str:
        p = path if path.startswith("/") else f"/{path}"
        return f"{self._backend_origin}{p}"

    def _ra_host_path(self, maybe_container_path: str) -> str:
        """
        If the backend is running in Docker it may return paths like /opt/knoxnet/data/...
        Translate those into a host path rooted at this repo so the Desktop app can open/run scripts.
        """
        p = str(maybe_container_path or "").strip()
        if not p:
            return ""
        # Normalize slashes for matching.
        p_norm = p.replace("\\", "/")
        if p_norm.startswith("/opt/knoxnet/"):
            rel = p_norm[len("/opt/knoxnet/") :].lstrip("/")
            return str((_repo_root() / rel).resolve())
        return p

    def _ra_headers(self) -> Dict[str, str]:
        return {"Content-Type": "application/json"}

    def _ra_poll_status(self) -> None:
        # Avoid overlapping polls (timer + manual refresh)
        if not self._ra_request_lock.acquire(blocking=False):
            return
        try:
            self._ra_last_poll_ts = time.time()
            url = self._ra_api("/api/remote-access/wireguard/status")
            res = requests.get(url, timeout=2.0)
            if not res.ok:
                raise RuntimeError(f"HTTP {res.status_code}")
            payload = res.json() or {}
            if not payload.get("success"):
                raise RuntimeError(payload.get("message") or "status failed")
            data = payload.get("data") or {}
            if isinstance(data, dict):
                self._ra_last_status = data
            # Update UI on main thread
            QTimer.singleShot(0, self._ra_apply_status)
        except Exception:
            self._ra_last_status = {}
            QTimer.singleShot(0, self._ra_apply_status)
        finally:
            try:
                self._ra_request_lock.release()
            except Exception:
                pass

    def _ra_apply_status(self) -> None:
        s = self._ra_last_status or {}
        enabled = bool(s.get("enabled"))
        iface_up = s.get("interface_up")
        peers = s.get("peers") if isinstance(s.get("peers"), list) else []
        any_handshake = False
        latest_hs = 0
        for p in peers:
            try:
                hs = int((p or {}).get("last_handshake") or 0)
                if hs > 0:
                    any_handshake = True
                    latest_hs = max(latest_hs, hs)
            except Exception:
                continue

        healthy = enabled and bool(iface_up) and any_handshake
        if not s:
            self.ra_status_lbl.setText("Status: Backend offline / unavailable")
            self.ra_status_lbl.setStyleSheet("font-weight: bold; color: #ef4444;")
        else:
            label = "Healthy" if healthy else ("Enabled" if enabled else "Disabled")
            color = "#10b981" if healthy else ("#3b82f6" if enabled else "#9aa4b2")
            self.ra_status_lbl.setText(f"Status: {label}")
            self.ra_status_lbl.setStyleSheet(f"font-weight: bold; color: {color};")

        # Avoid feedback loop: block signals while syncing checkbox
        self.ra_enable_chk.blockSignals(True)
        self.ra_enable_chk.setChecked(enabled)
        self.ra_enable_chk.blockSignals(False)

        vpn_url = str(s.get("vpn_ui_url") or "").strip()
        self.ra_open_btn.setEnabled(bool(vpn_url))
        self.ra_open_bundle_btn.setEnabled(bool(str(s.get("bundle_dir") or "").strip()))
        self.ra_add_peer_btn.setEnabled(bool(enabled))

        # Auto-load (and if needed auto-create once) the peer QR/config for this menu.
        if enabled:
            threading.Thread(target=self._ra_fetch_latest_peer_bundle, daemon=True).start()
        else:
            self._ra_last_peer_bundle = None
            self._ra_peer_autocreate_attempted = False
            # Ensure any previously rendered QR pixmap is cleared when disabling.
            self.ra_peer_qr_lbl.setPixmap(QPixmap())
            self.ra_peer_qr_lbl.setText("Enable WireGuard to generate the phone QR.")
            self.ra_peer_qr_lbl.setStyleSheet("color: #9aa4b2;")
            self.ra_peer_conf_txt.setPlainText("")
            self.ra_peer_hint_lbl.setText("Click “Enable WireGuard (guided)” above. Then re-open this menu or wait a few seconds.")

        details = []
        endpoint = str(s.get("endpoint") or "").strip()
        if endpoint:
            details.append(f"Endpoint: {endpoint}")
        bundle = str(s.get("bundle_dir") or "").strip()
        if bundle:
            details.append(f"Bundle: {bundle}")
        if latest_hs > 0:
            try:
                details.append(f"Last handshake: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(latest_hs))}")
            except Exception:
                pass
        if bool(s.get("likely_cgnat")):
            details.append("Likely CGNAT / VPN mode may not be possible without a public IPv4")
        self.ra_details_lbl.setText(" • ".join(details) if details else "—")

    def _ra_apply_peer_bundle(self) -> None:
        """
        Render the persistent peer QR/config panel from self._ra_last_peer_bundle.
        """
        b = self._ra_last_peer_bundle
        if not b:
            self.ra_peer_qr_lbl.setText("No phone peer yet.")
            self.ra_peer_qr_lbl.setStyleSheet("color: #f59e0b; font-weight: bold;")
            self.ra_peer_conf_txt.setPlainText("")
            self.ra_peer_hint_lbl.setText("Auto-generating a phone peer… (or click “Add Phone (QR)” to force a new one)")
            return

        conf = str(b.get("client_config_text") or "")
        qr_b64 = str(b.get("qr_png_base64") or "")
        qr_error = str(b.get("qr_error") or "").strip()
        endpoint = str(b.get("endpoint") or "").strip()
        warnings = b.get("warnings") if isinstance(b.get("warnings"), list) else []

        pm = _qpixmap_from_base64_png(qr_b64) if qr_b64 else None
        if pm is not None and not pm.isNull():
            pm2 = pm.scaled(
                160,
                160,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
            self.ra_peer_qr_lbl.setPixmap(pm2)
        else:
            msg = "QR unavailable.\n\n"
            if qr_error:
                msg += f"{qr_error}\n\n"
            msg += "You can still import the .conf file in the WireGuard app."
            self.ra_peer_qr_lbl.setText(msg)
            self.ra_peer_qr_lbl.setWordWrap(True)
            self.ra_peer_qr_lbl.setStyleSheet("color: #ef4444; font-weight: bold;")

        self.ra_peer_conf_txt.setPlainText(conf or "")

        hint_parts: List[str] = []
        if endpoint:
            hint_parts.append(f"Endpoint: {endpoint}")
        for w in warnings:
            try:
                ws = str(w).strip()
                if ws:
                    hint_parts.append(ws)
            except Exception:
                continue
        if not hint_parts:
            hint_parts.append("Scan the QR in the WireGuard app (or import the .conf).")
        self.ra_peer_hint_lbl.setText("\n".join(hint_parts))

    def _ra_fetch_latest_peer_bundle(self) -> None:
        """
        Best-effort fetch of the latest peer bundle for persistent display.
        If none exists yet, auto-create once per dialog-open (when enabled).
        """
        try:
            url = self._ra_api("/api/remote-access/wireguard/peers/latest")
            res = requests.get(url, timeout=4.0)
            if not res.ok:
                raise RuntimeError(f"HTTP {res.status_code}: {res.text}")
            payload = res.json() or {}
            if not payload.get("success"):
                raise RuntimeError(payload.get("message") or "latest peer failed")
            data = payload.get("data") or {}
            has_peer = bool(data.get("has_peer"))
            if has_peer and isinstance(data.get("bundle"), dict):
                self._ra_last_peer_bundle = data.get("bundle")
                QTimer.singleShot(0, self._ra_apply_peer_bundle)
                return

            # No peer yet: auto-create once
            self._ra_last_peer_bundle = None
            QTimer.singleShot(0, self._ra_apply_peer_bundle)
            if not self._ra_peer_autocreate_attempted:
                self._ra_peer_autocreate_attempted = True
                self._ra_add_peer()
                return
        except Exception as e:
            # Show error inline (no silent failures)
            self._ra_last_peer_bundle = {
                "client_config_text": "",
                "qr_png_base64": "",
                "qr_error": f"Failed to load peer bundle: {e}",
                "warnings": [],
                "endpoint": "",
            }
            QTimer.singleShot(0, self._ra_apply_peer_bundle)

    def _ra_on_toggle_enabled(self, checked: bool) -> None:
        threading.Thread(target=self._ra_set_enabled, args=(checked,), daemon=True).start()

    def _ra_set_enabled(self, checked: bool) -> None:
        try:
            if checked:
                url = self._ra_api("/api/remote-access/wireguard/enable")
                res = requests.post(url, json={}, headers=self._ra_headers(), timeout=4.0)
            else:
                url = self._ra_api("/api/remote-access/wireguard/disable")
                res = requests.post(url, json={}, headers=self._ra_headers(), timeout=4.0)
            if not res.ok:
                raise RuntimeError(f"HTTP {res.status_code}: {res.text}")
            # Show a clear "what to do next" message on enable.
            if checked:
                try:
                    payload = res.json() or {}
                    data = payload.get("data") or {}
                    bundle_dir = str((self._ra_last_status or {}).get("bundle_dir") or data.get("bundle_dir") or "").strip()
                    apply_res = data.get("apply_result") if isinstance(data, dict) else None
                    msg: List[str] = []
                    if bundle_dir:
                        msg.append(f"Setup bundle: {bundle_dir}")

                        # Attempt autostart where possible (Windows host).
                        if os.name == "nt":
                            ok, detail = self._ra_windows_try_autostart(bundle_dir=bundle_dir)
                            if ok:
                                msg.append("Tunnel: Started (or starting).")
                            else:
                                # If the user cancels UAC, ShellExecute returns <= 32.
                                msg.append("Tunnel: Not started automatically.")
                                if detail:
                                    msg.append(f"Reason: {detail}")
                                msg.append("Fix: Run install_windows.ps1 as Administrator from the setup bundle folder.")
                        else:
                            msg.append("Tunnel: Run install_linux.sh with sudo to bring the tunnel up.")

                    if isinstance(apply_res, dict) and apply_res.get("message"):
                        msg.append(f"Apply: {apply_res.get('message')}")
                    if msg:
                        QTimer.singleShot(0, lambda: QMessageBox.information(self, "WireGuard", "\n".join(msg)))
                except Exception:
                    pass
        except Exception as e:
            QTimer.singleShot(0, lambda: QMessageBox.warning(self, "Remote Access", f"Failed to update WireGuard state: {e}"))
        finally:
            self._ra_poll_status()

    def _ra_windows_try_autostart(self, *, bundle_dir: str) -> tuple[bool, str]:
        """
        Windows only: attempt to elevate and run install_windows.ps1, which installs/starts the tunnel service.
        Returns (ok, detail). ok=True means the elevation prompt was launched successfully (not that WG succeeded).
        """
        if os.name != "nt":
            return False, "Not Windows"
        try:
            bundle = self._ra_host_path(bundle_dir)
            if not bundle:
                return False, "Missing bundle dir"
            script = os.path.join(bundle, "install_windows.ps1")
            if not os.path.exists(script):
                return False, f"Missing {script}"

            # Use ShellExecuteW with "runas" to trigger UAC prompt.
            import ctypes

            params = f'-NoProfile -ExecutionPolicy Bypass -File "{script}"'
            rc = ctypes.windll.shell32.ShellExecuteW(None, "runas", "powershell.exe", params, None, 1)  # type: ignore[attr-defined]
            # Per docs: return value <= 32 indicates an error.
            if int(rc) <= 32:
                return False, f"UAC declined / ShellExecute error ({int(rc)})"
            return True, ""
        except Exception as e:
            return False, str(e)

    def _ra_open_bundle_dir(self) -> None:
        s = self._ra_last_status or {}
        p = self._ra_host_path(str(s.get("bundle_dir") or "").strip())
        if not p:
            return
        try:
            if os.name == "nt":
                os.startfile(p)  # type: ignore[attr-defined]
            else:
                subprocess.Popen(["xdg-open", p])
        except Exception as e:
            QMessageBox.warning(self, "Open Folder Failed", str(e))

    def _ra_open_knoxnet_vpn(self) -> None:
        s = self._ra_last_status or {}
        url = str(s.get("vpn_ui_url") or "").strip()
        if not url:
            return
        try:
            from PySide6.QtGui import QDesktopServices
            from PySide6.QtCore import QUrl
            QDesktopServices.openUrl(QUrl(url))
        except Exception:
            pass

    def _ra_add_peer(self) -> None:
        try:
            name = (self.ra_peer_name.text() or "").strip() or "Phone"
            url = self._ra_api("/api/remote-access/wireguard/peers")
            res = requests.post(url, json={"name": name}, headers=self._ra_headers(), timeout=6.0)
            if not res.ok:
                raise RuntimeError(f"HTTP {res.status_code}: {res.text}")
            payload = res.json() or {}
            if not payload.get("success"):
                raise RuntimeError(payload.get("message") or "peer create failed")
            data = payload.get("data") or {}
            conf = str(data.get("client_config_text") or "")
            qr_b64 = str(data.get("qr_png_base64") or "")
            qr_error = data.get("qr_error")
            warnings = data.get("warnings") if isinstance(data.get("warnings"), list) else None
            endpoint = str(data.get("endpoint") or "").strip() or None
            if not conf:
                raise RuntimeError("Peer config missing in response")

            # Update persistent panel immediately
            self._ra_last_peer_bundle = {
                "client_config_text": conf,
                "qr_png_base64": qr_b64,
                "qr_error": str(qr_error) if qr_error else None,
                "warnings": warnings or [],
                "endpoint": endpoint or "",
            }
            QTimer.singleShot(0, self._ra_apply_peer_bundle)

            def _show():
                dlg = _PeerConfigDialog(
                    parent=self,
                    title=f"WireGuard Peer: {name}",
                    config_text=conf,
                    qr_png_base64=qr_b64,
                    qr_error=str(qr_error) if qr_error else None,
                    warnings=warnings,
                    endpoint=endpoint,
                )
                dlg.exec()
            QTimer.singleShot(0, _show)
        except Exception as e:
            self._ra_last_peer_bundle = {
                "client_config_text": "",
                "qr_png_base64": "",
                "qr_error": str(e),
                "warnings": [],
                "endpoint": "",
            }
            QTimer.singleShot(0, self._ra_apply_peer_bundle)
            QTimer.singleShot(0, lambda: QMessageBox.warning(self, "Add Phone", str(e)))
        finally:
            self._ra_poll_status()

    def _safe_list_sessions(self) -> List[Any]:
        try:
            if self._app and hasattr(self._app, "session_manager") and self._app.session_manager:
                return self._app.session_manager.list_sessions()
        except: pass
        return []

    def _safe_session_widget_count(self, session_id: str) -> int:
        try:
            m = getattr(self._app, "_session_widgets", None)
            if isinstance(m, dict) and session_id in m:
                return len([w for w in m.get(session_id, []) if w is not None])
        except: pass
        return 0

    def _safe_widget_summary(self) -> str:
        try:
            if not self._app or not hasattr(self._app, "active_widgets"): return ""
            counts: Dict[str, int] = {}
            for w in list(self._app.active_widgets):
                t = type(w).__name__
                counts[t] = counts.get(t, 0) + 1
            if not counts: return "Active widgets: 0"
            return "Active widgets — " + ", ".join([f"{k}: {v}" for k, v in sorted(counts.items())])
        except: return ""

    def _safe_camera_snapshot(self) -> Dict[str, Dict[str, Any]]:
        try:
            cm = getattr(self._app, "camera_manager", None)
            if cm and hasattr(cm, "get_usage_snapshot"):
                return cm.get_usage_snapshot() or {}
        except: pass
        return {}

    def _set_tbl_item(self, tbl: QTableWidget, r: int, c: int, text: str):
        it = QTableWidgetItem(text)
        it.setFlags(it.flags() & ~Qt.ItemFlag.ItemIsEditable)
        tbl.setItem(r, c, it)

    def _stop_background_layouts(self):
        if self._app and hasattr(self._app, "stop_all_sessions"):
            self._app.stop_all_sessions()
        self.refresh()

    def _disconnect_unused_cameras(self):
        if self._app and hasattr(self._app, "disconnect_unused_cameras"):
            self._app.disconnect_unused_cameras()
        self.refresh()
