import base64
import json
import os
import re
import subprocess
import threading
import time
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, ClassVar
import socket
import sys
from collections import OrderedDict

import requests
from PySide6.QtCore import Qt, QTimer, Signal, QUrl
from PySide6.QtGui import QColor, QDesktopServices, QPainter
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QTextBrowser,
    QVBoxLayout,
    QWidget,
    QMenu,
    QApplication,
    QDialog,
    QFormLayout,
    QComboBox,
    QCheckBox,
    QMessageBox,
    QDateTimeEdit,
    QDialogButtonBox,
)

from desktop.widgets.base import BaseDesktopWidget
from desktop.utils.qt_helpers import KnoxnetStyle


class GhostLineEdit(QLineEdit):
    """QLineEdit that shows a faded inline ghost-text suggestion as you type.

    Press Tab to accept the current suggestion.  The ghost text is drawn
    directly inside the widget so it feels native and stays perfectly aligned
    with the typed characters.
    """

    COMMANDS = [
        "watch all",
        "watch ",
        "stop watch all",
        "stop watch ",
        "watch stop all",
        "watch stop ",
        "record all",
        "record ",
        "stop record all",
        "stop record ",
        "recording status",
        "recording dir ",
        "recording dir all ",
        "recording list ",
        "recording list",
        "recording paths",
        "motion box all",
        "motion box off all",
        "motion box off ",
        "motion box ",
        "detect all",
        "detect off all",
        "detect off ",
        "detect ",
        "show all",
        "status",
        "cameras",
        "camera ",
        "open camera ",
        "grid",
        "alerts",
        "tools",
        "events backfill",
        "events reindex",
        "events enrich",
        "events report ",
        "events live",
        "live report",
        "layouts",
        "layout run ",
        "layout stop ",
        "layout ",
        "sensitivity ",
        "events ",
        "clear",
        "help",
    ]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._ghost: str = ""
        self.textChanged.connect(self._update_ghost)

    def _update_ghost(self, text: str):
        typed = text.lstrip()
        if not typed:
            self._ghost = ""
            self.update()
            return
        low = typed.lower()
        for cmd in self.COMMANDS:
            if cmd.startswith(low) and cmd != low:
                self._ghost = cmd
                self.update()
                return
        self._ghost = ""
        self.update()

    def paintEvent(self, event):
        super().paintEvent(event)
        if not self._ghost or not self.text():
            return
        typed = self.text()
        suffix = self._ghost[len(typed):]
        if not suffix:
            return
        painter = QPainter(self)
        try:
            ghost_color = QColor(255, 255, 255, 50)
            painter.setPen(ghost_color)
            painter.setFont(self.font())
            cr = self.cursorRect()
            x = cr.right() + 1
            r = self.rect()
            painter.drawText(
                x, r.y(), r.width() - x, r.height(),
                Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
                suffix,
            )
        finally:
            painter.end()

    def keyPressEvent(self, event):
        if event.key() == Qt.Key.Key_Tab and self._ghost:
            self.setText(self._ghost)
            self.setCursorPosition(len(self._ghost))
            return
        super().keyPressEvent(event)


class TerminalWidget(BaseDesktopWidget):
    """
    Desktop Terminal widget that mirrors the React TerminalWidgetRefactored behavior:
    - AI agent chat with tool results (including inline base64 images)
    - Quick status/camera/alert helpers
    - System status header with uptime/CPU/Mem/GPU (when available)
    - Command history and feedback hooks
    - Bounded log persistence
    """

    ai_response_signal = Signal(dict)
    error_signal = Signal(str)
    status_signal = Signal(dict)
    log_signal = Signal(dict)
    watch_signal = Signal(dict)

    # Track live instances for cross-widget broadcasts (e.g., motion watch)
    _instances: ClassVar[List["TerminalWidget"]] = []
    _backend_lock: ClassVar[threading.Lock] = threading.Lock()
    _backend_process: ClassVar[Optional[subprocess.Popen]] = None

    # Prefer IPv4 loopback to avoid Windows IPv6 (::1) connection issues.
    API_BASE = "http://localhost:5000/api"
    # Persist user state under the per-user data dir in frozen builds.
    # In dev/source, this resolves to <repo>/data/... so existing workflows still work.
    try:
        from core.paths import get_data_dir as _get_data_dir
        _DATA_DIR = _get_data_dir()
    except Exception:
        _DATA_DIR = Path("data")
    HISTORY_PATH = _DATA_DIR / "desktop_terminal_history.json"
    AGENT_SETTINGS_PATH = _DATA_DIR / "desktop_agent_settings.json"
    APP_ROOT = _DATA_DIR.parent
    # Dev backend entry; in frozen builds we spawn `sys.executable --run-backend` instead.
    BACKEND_ENTRY = APP_ROOT / "app.py"
    BACKEND_LOG = APP_ROOT / "logs" / "desktop_backend.log"
    # Keep the QTextBrowser doc bounded; large inline base64 images will otherwise make UI slower over time.
    MAX_LOG_ENTRIES = 250
    # image:// click-to-open cache (LRU). The display itself uses inline base64; keep this bounded.
    IMAGE_CACHE_MAX_ITEMS = 64

    def __init__(self, title: str = "Terminal", width: int = 820, height: int = 560):
        super().__init__(title=title, width=width, height=height)

        # State
        self.widget_key = f"terminal_{uuid.uuid4().hex}"
        self.log: List[Dict[str, Any]] = []
        self.history: List[str] = []
        self.history_index = -1
        self.agent_active = False
        self.text_scale = 1.0
        self.text_color = "#22d3ee"  # default cyan
        self.start_time = time.time()
        self.image_cache: "OrderedDict[str, str]" = OrderedDict()
        self._detection_cache: Dict[str, Dict[str, Any]] = {}
        self.agent_settings: Dict[str, Any] = {
            "primary": "openai",
            "failover": [],
            "provider_priority": [],
            "use_local_vision": True,
            "api_keys": {},
        }
        # Prevent periodic status polling from ever blocking the UI thread or piling up.
        self._status_fetch_lock = threading.Lock()
        self._status_fetch_inflight = False

        # UI
        self.title_bar.show()  # show custom bar for minimize/close controls
        self.main_layout.setContentsMargins(0, 0, 0, 0)
        self.content_layout.setContentsMargins(0, 0, 0, 0)

        container = QWidget()
        container.setStyleSheet(
            f"""
            QWidget {{
                background-color: {KnoxnetStyle.BG_DARK};
                color: {KnoxnetStyle.TEXT_MAIN};
            }}
        """
        )
        layout = QVBoxLayout(container)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)

        # Header removed - functional labels/buttons kept hidden for logic compatibility
        self.header_label = QLabel()
        self.header_label.hide()
        self.status_label = QLabel()
        self.status_label.hide()
        self.agent_btn = QPushButton()
        self.agent_btn.hide()

        # Motion watch badges (multiple concurrent)
        self.motion_watch_status: Dict[str, Dict[str, Any]] = {}
        self.watch_badge = QLabel("")
        self.watch_badge.setStyleSheet(
            """
            QLabel {
                color: #0ea5e9;
                background-color: rgba(14,165,233,0.08);
                border: 1px solid rgba(14,165,233,0.35);
                border-radius: 6px;
                padding: 2px 6px;
                font-size: 11px;
            }
            """
        )
        self.watch_badge.setAlignment(Qt.AlignmentFlag.AlignRight)
        layout.addWidget(self.watch_badge)
        self.watch_badge.hide()

        # Log area (The "Slate")
        self.log_view = QTextBrowser()
        self.log_view.setOpenExternalLinks(False)
        # Critical: prevent QTextBrowser from navigating away (which would wipe the current terminal content)
        # when clicking custom links like image://... or file://...
        self.log_view.setOpenLinks(False)
        self.log_view.anchorClicked.connect(self._handle_anchor_click)
        self.log_view.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.log_view.customContextMenuRequested.connect(self._show_log_menu)
        self.log_view.setStyleSheet(
            f"""
            QTextBrowser {{
                background: #000000;
                border: none;
                color: {self.text_color};
                padding: 10px;
                font-family: 'Consolas', 'Monaco', monospace;
            }}
            """
        )
        self.log_view.setReadOnly(True)

        # Input row (Minimal profile)
        input_row = QHBoxLayout()
        input_row.setContentsMargins(10, 0, 10, 6)
        self.prompt = QLabel("$")
        self.prompt.setStyleSheet(f"color: {self.text_color}; font-weight: 700; font-family: monospace;")
        self.input = GhostLineEdit()
        self.input.setFrame(False)
        self.input.returnPressed.connect(self.handle_submit)
        self.input.setPlaceholderText("Type command or chat...")
        self.input.setStyleSheet(
            f"""
            QLineEdit {{
                background: transparent;
                border: none;
                color: {KnoxnetStyle.TEXT_MAIN};
                font-family: 'Consolas', 'Monaco', monospace;
                font-size: 13px;
            }}
            """
        )
        # Hidden combo boxes for logic compatibility
        self.provider_combo = QComboBox()
        self.provider_combo.hide()
        self.model_combo = QComboBox()
        self.model_combo.hide()
        self.send_btn = QPushButton() # Keep hidden for logic
        self.send_btn.hide()

        input_row.addWidget(self.prompt)
        input_row.addWidget(self.input, stretch=1)

        layout.addWidget(self.log_view, stretch=1)
        layout.addLayout(input_row)
        self.set_content(container)

        # Signals
        self.ai_response_signal.connect(self._on_ai_response)
        self.error_signal.connect(self._on_error)
        self.status_signal.connect(self._on_status_update)
        self.log_signal.connect(self._add_line_from_signal)
        self.watch_signal.connect(self._on_motion_watch_update)

        # Timers (Background status kept for internal logic, UI hidden)
        self.system_timer = QTimer(self)
        self.system_timer.timeout.connect(self._update_system_status)
        self.system_timer.start(1000)

        # Load persisted state
        self._load_agent_settings()
        self._rehydrate_state()
        
        # Fresh startup message
        self.clear_log()
        self._add_system("Terminal Ready. Right-click for menu.")

        # Populate provider/model dropdowns (best-effort)
        threading.Thread(target=self._load_provider_models, daemon=True).start()

        # Register instance for broadcast use-cases
        TerminalWidget._instances.append(self)

        # Start health check
        self.status_timer = QTimer(self)
        self.status_timer.timeout.connect(self._fetch_status_async)
        self.status_timer.start(10000)
        QTimer.singleShot(500, self._fetch_status_async)

    # UI helpers ---------------------------------------------------------
    def _mk_button(self, label: str, handler):
        btn = QPushButton(label)
        btn.clicked.connect(handler)
        btn.setCursor(Qt.CursorShape.PointingHandCursor)
        btn.setStyleSheet(
            """
            QPushButton {
                background-color: #374151;
                color: #e5e7eb;
                border: 1px solid #4b5563;
                padding: 4px 10px;
                border-radius: 4px;
            }
            QPushButton:hover { background-color: #4b5563; }
            """
        )
        return btn

    def _pill_style(self, active: bool) -> str:
        if active:
            return "background-color:#16a34a; color:#fff; border:1px solid #15803d; padding:4px 10px; border-radius:14px;"
        return "background-color:#374151; color:#e5e7eb; border:1px solid #4b5563; padding:4px 10px; border-radius:14px;"

    # Persistence --------------------------------------------------------
    def _rehydrate_state(self):
        try:
            if self.HISTORY_PATH.exists():
                data = json.loads(self.HISTORY_PATH.read_text())
                entry = data[-1] if isinstance(data, list) and data else None
                if entry and isinstance(entry, dict):
                    self.log = entry.get("log", [])[-200:]
                    self.text_color = entry.get("text_color", self.text_color)
                    self.text_scale = float(entry.get("text_scale", self.text_scale))
                    self.agent_active = bool(entry.get("agent_active", False))
                    self._refresh_log_view()
                    if self.agent_active:
                        self.agent_btn.setChecked(True)
                        self.agent_btn.setText("🤖 Agent ON")
                        self.agent_btn.setStyleSheet(self._pill_style(True))
        except Exception:
            # best-effort; ignore corrupt history
            pass

    def _persist_state(self):
        try:
            self.HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
            # Never persist base64 images: they bloat history and slow down rehydrate.
            sanitized_log = []
            for e in (self.log or [])[-200:]:
                if not isinstance(e, dict):
                    continue
                ee = dict(e)
                ee.pop("img", None)
                sanitized_log.append(ee)
            payload = {
                "log": sanitized_log,
                "text_color": self.text_color,
                "text_scale": self.text_scale,
                "agent_active": self.agent_active,
            }
            self.HISTORY_PATH.write_text(json.dumps([payload], indent=2))
        except Exception:
            pass

    def _load_agent_settings(self):
        try:
            if self.AGENT_SETTINGS_PATH.exists():
                data = json.loads(self.AGENT_SETTINGS_PATH.read_text())
                if isinstance(data, dict):
                    self.agent_settings.update(data)
        except Exception:
            pass

    def _persist_agent_settings(self):
        try:
            self.AGENT_SETTINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
            self.AGENT_SETTINGS_PATH.write_text(json.dumps(self.agent_settings, indent=2))
        except Exception:
            pass

    # Log handling -------------------------------------------------------
    def _add_line(
        self,
        text: str,
        kind: str = "output",
        image_b64: Optional[str] = None,
        link: Optional[str] = None,
        link_label: Optional[str] = None,
        tool: Optional[str] = None,
        model: Optional[str] = None,
    ):
        ts = time.strftime("%H:%M:%S")
        entry: Dict[str, Any] = {
            "t": ts,
            "k": kind,
            "text": text,
            "tool": tool,
            "model": model,
        }
        if image_b64:
            entry["img"] = image_b64
        if link:
            entry["link"] = link
        if link_label:
            entry["link_label"] = link_label
        self.log.append(entry)
        # Keep both the in-memory log and the QTextBrowser document bounded to prevent long-run UI degradation.
        if len(self.log) > self.MAX_LOG_ENTRIES:
            self.log = self.log[-self.MAX_LOG_ENTRIES :]
            self._refresh_log_view()
        else:
            self._append_to_view(entry)
        self._persist_state()

    def _add_line_from_signal(self, payload: Dict[str, Any]):
        # Ensures UI updates happen on the Qt main thread
        self._add_line(
            text=payload.get("text", ""),
            kind=payload.get("kind", "output"),
            image_b64=payload.get("image_b64"),
            link=payload.get("link"),
            link_label=payload.get("link_label"),
            tool=payload.get("tool"),
            model=payload.get("model"),
        )

    def _post_line(
        self,
        text: str,
        kind: str = "output",
        image_b64: Optional[str] = None,
        link: Optional[str] = None,
        link_label: Optional[str] = None,
        tool: Optional[str] = None,
        model: Optional[str] = None,
    ):
        # Safe to call from background threads
        self.log_signal.emit(
            {
                "text": text,
                "kind": kind,
                "image_b64": image_b64,
                "link": link,
                "link_label": link_label,
                "tool": tool,
                "model": model,
            }
        )

    def _append_to_view(self, entry: Dict[str, Any]):
        color = {
            "input": self.text_color,
            "system": "#60a5fa",
            "error": "#f87171",
            "success": "#34d399",
            "warning": "#f59e0b",
            "info": "#fbbf24",
            "output": "#e5e7eb",
        }.get(entry.get("k", "output"), "#e5e7eb")

        text = entry.get("text", "")
        text_html = self._escape(text).replace("\n", "<br>")
        if entry.get("tool"):
            tool = self._escape(entry.get("tool") or "")
            model = self._escape(entry.get("model") or "")
            meta_parts = [tool] if tool else []
            if model:
                meta_parts.append(model)
            if meta_parts:
                text_html += f' <span style="color:#9ca3af">[{", ".join(meta_parts)}]</span>'

        img_html = ""
        if entry.get("img"):
            key = uuid.uuid4().hex
            # LRU cache for click-to-open (image://)
            self.image_cache[key] = entry["img"]
            try:
                self.image_cache.move_to_end(key)
            except Exception:
                pass
            while len(self.image_cache) > self.IMAGE_CACHE_MAX_ITEMS:
                try:
                    self.image_cache.popitem(last=False)
                except Exception:
                    break
            img_html = (
                f'<div style="margin-top:6px;"><a href="image://{key}">'
                f'<img src="data:image/jpeg;base64,{entry["img"]}" style="max-width:100%;max-height:240px;border:1px solid #374151;border-radius:6px;" /></a></div>'
            )

        link_html = ""
        if entry.get("link"):
            link_href = self._escape(entry["link"])
            label = self._escape(entry.get("link_label") or "Open link")
            link_html = f'<div><a href="{link_href}" style="color:#93c5fd;">{label}</a></div>'

        html = f'<div style="color:{color}; font-size:{12 * self.text_scale}px;">[{self._escape(entry.get("t",""))}] {text_html}{img_html}{link_html}</div>'
        self.log_view.append(html)
        self.log_view.verticalScrollBar().setValue(self.log_view.verticalScrollBar().maximum())

    def _refresh_log_view(self):
        self.log_view.clear()
        try:
            self.image_cache.clear()
        except Exception:
            pass
        for entry in self.log:
            self._append_to_view(entry)

    # Event handlers -----------------------------------------------------
    def handle_submit(self):
        text = self.input.text().strip()
        if not text:
            return
        # Allow "run <command>" as an alias for built-in commands (common user expectation).
        if text.lower().startswith("run "):
            text = text[4:].strip()
        self._add_line(f"$ {text}", kind="input")
        self.history.append(text)
        self.history_index = -1
        self.input.clear()

        # Built-in commands
        lowered = text.lower()
        if lowered in {"clear", "cls"}:
            self.clear_log()
            return
        if lowered in {"help", "?"}:
            self._add_system(
                "Commands:\n"
                "  status, cameras, show all, alerts, tools, events, clear\n"
                "\n"
                "  Layout / Grid:\n"
                "  grid                     – smart auto grid (respects aspect ratios)\n"
                "  grid <N>                 – N columns, auto rows\n"
                "  grid <N>x<M>             – N columns × M rows\n"
                "  grid sort name           – auto grid sorted alphabetically\n"
                "  grid sort name desc      – reverse alphabetical\n"
                "  grid sort id             – sort by camera id\n"
                "  grid fill                – stretch to fill (ignores aspect ratio)\n"
                "  grid seamless            – tight packing, no gaps\n"
                "  grid focus <cam>         – one camera large + sidebar\n"
                "  cascade                  – cascade / stagger windows\n"
                "  tile                     – tile cameras horizontally\n"
                "  tile v                   – tile cameras vertically\n"
                "  fullscreen <cam>         – fullscreen one camera\n"
                "  patrol [sec]             – cycle cameras fullscreen (toggle)\n"
                "  snap                     – toggle snap-to-grid mode\n"
                "  snap grid                – auto-fit cameras into seamless grid\n"
                "  minimize all             – minimize all camera widgets\n"
                "  restore all              – restore & auto-grid all cameras\n"
                "  close all                – close all camera widgets\n"
                "\n"
                "  Recording:\n"
                "  record all             – start continuous recording on all cameras\n"
                "  record <cam>           – start continuous recording on a camera\n"
                "  stop record all        – stop continuous recording on all cameras\n"
                "  stop record <cam>      – stop continuous recording on a camera\n"
                "  recording status       – show recording state, paths, and disk usage\n"
                "  recording paths        – show recording directory for each camera\n"
                "  recording dir <cam> <path> – set recording directory for a camera\n"
                "  recording dir all <path>   – set recording directory for all cameras\n"
                "  recording list [cam]   – list recording files (all or one camera)\n"
                "\n"
                "  Motion / Detection:\n"
                "  watch all              – start motion watch on all cameras\n"
                "  watch <cam>            – start motion watch on a camera\n"
                "  stop watch all         – stop motion watch on all cameras\n"
                "  stop watch <cam>       – stop motion watch on a camera\n"
                "  motion box all         – enable motion boxes on all cameras\n"
                "  motion box off all     – disable motion boxes on all cameras\n"
                "  motion box <cam>       – enable motion boxes on a camera\n"
                "  motion box off <cam>   – disable motion boxes on a camera\n"
                "  detect all             – enable object detection on all cameras\n"
                "  detect off all         – disable object detection on all cameras\n"
                "  detect <cam>           – enable object detection on a camera\n"
                "  detect off <cam>       – disable object detection on a camera\n"
                "  sensitivity <1-100>    – set motion sensitivity for all cameras\n"
                "\n"
                "  Layouts:\n"
                "  layouts                – list all layouts with status\n"
                "  layout <name>          – load a layout (replaces current)\n"
                "  layout run <name>      – run layout in background\n"
                "  layout stop <name>     – stop a running layout\n"
                "\n"
                "  Events:\n"
                "  events enrich [N]      – classify unprocessed captures with YOLO\n"
                "  live report            – open live security report dashboard\n"
                "  events live            – open live security report dashboard"
            )
            return
        if lowered in ("live report", "live", "live dashboard"):
            self._open_live_report()
            return
        if lowered.startswith("events") or lowered.startswith("timeline"):
            # Syntax:
            #   events <free text query>
            #   events backfill [N]
            #   events reindex [max_files] [--force] [--cloud <max_calls>]
            #   events report <free text query>
            parts = text.split()
            if len(parts) >= 2 and parts[1].lower() == "backfill":
                n = 250
                if len(parts) >= 3 and parts[2].isdigit():
                    n = max(1, min(int(parts[2]), 5000))
                self._add_system(f"Backfilling events index (max {n})…")
                threading.Thread(target=self._events_backfill, args=(n,), daemon=True).start()
                return
            if len(parts) >= 2 and parts[1].lower() == "reindex":
                # defaults
                max_files = None
                force = False
                cloud_enrich = False
                cloud_max_calls = 25

                # Parse args
                for tok in parts[2:]:
                    t = tok.strip().lower()
                    if not t:
                        continue
                    if t in {"--force", "force"}:
                        force = True
                        continue
                    if t in {"--cloud", "cloud"}:
                        cloud_enrich = True
                        continue
                    if t.isdigit():
                        # first numeric token is max_files, unless cloud already enabled and max_files already set
                        if max_files is None and not cloud_enrich:
                            max_files = int(t)
                        else:
                            cloud_max_calls = max(1, min(int(t), 500))
                        continue

                # If cloud flag is present, warn explicitly before starting.
                if cloud_enrich:
                    msg = (
                        "Cloud enrichment will call your configured cloud vision provider (OpenAI/Grok/etc).\n\n"
                        "This may:\n"
                        "- incur cost\n"
                        "- hit rate limits\n"
                        "- take a long time on bulk batches\n\n"
                        f"Planned max cloud calls this run: {cloud_max_calls}\n\n"
                        "Proceed?"
                    )
                    if QMessageBox.question(self, "Bulk Cloud Enrichment Warning", msg) != QMessageBox.StandardButton.Yes:
                        self._add_warning("Reindex cancelled (cloud enrichment not approved).")
                        return

                self._add_system("Starting bulk reindex…")
                threading.Thread(
                    target=self._events_reindex,
                    args=(max_files, force, cloud_enrich, cloud_max_calls),
                    daemon=True,
                ).start()
                return
            if len(parts) >= 2 and parts[1].lower() == "enrich":
                max_files = None
                force = False
                for tok in parts[2:]:
                    t = tok.strip().lower()
                    if t in {"--force", "force"}:
                        force = True
                    elif t.isdigit() and max_files is None:
                        max_files = max(1, min(int(t), 5000))
                self._add_system(f"Enriching unclassified captures (detections=True, force={force}, max={max_files or 'all'})…")
                threading.Thread(
                    target=self._events_reindex,
                    args=(max_files, force, False, 0),
                    daemon=True,
                ).start()
                return
            if len(parts) >= 2 and parts[1].lower() == "report":
                report_query = text.split(" ", 2)[2].strip() if len(parts) >= 3 else ""
                if not report_query:
                    self._add_warning("Usage: events report <query>. Example: events report red cars yesterday produce stand")
                    return
                threading.Thread(target=self._events_report, args=({"query": report_query, "limit": 200},), daemon=True).start()
                return
            if len(parts) >= 2 and parts[1].lower() == "live":
                self._open_live_report()
                return
            query = text.split(" ", 1)[1].strip() if " " in text else ""
            self._add_system("Searching events…")
            threading.Thread(target=self._run_events_search, args=({"query": query},), daemon=True).start()
            return
        if lowered.startswith("stop watch"):
            ref = text.split(" ", 2)[-1].strip() if " " in text else ""
            if ref.lower() == "all":
                self._handle_stop_watch_all()
            else:
                self._handle_stop_watch(ref)
            return
        if lowered.startswith("watch stop"):
            ref = text.split(" ", 2)[-1].strip() if " " in text else ""
            if ref.lower() == "all":
                self._handle_stop_watch_all()
            else:
                self._handle_stop_watch(ref)
            return
        if lowered.startswith("watch all") or lowered == "watch":
            self._handle_watch_all(text)
            return
        if lowered.startswith("watch "):
            ref = text.split(" ", 1)[1].strip()
            self._handle_watch_camera(ref, text)
            return
        # -- recording management commands (must match before "record " prefix) --
        if lowered in ("recording status", "rec status"):
            self._handle_recording_status()
            return
        if lowered in ("recording paths", "rec paths", "recording path"):
            self._handle_recording_paths()
            return
        if lowered.startswith("recording dir ") or lowered.startswith("rec dir "):
            rest = text.split(None, 2)[2] if len(text.split(None, 2)) > 2 else ""
            self._handle_recording_dir(rest)
            return
        if lowered.startswith("recording list") or lowered.startswith("rec list"):
            parts = text.split(None, 2)
            ref = parts[2] if len(parts) > 2 else ""
            self._handle_recording_list(ref.strip())
            return
        # -- record / stop record --
        if lowered.startswith("stop record"):
            ref = text.split(" ", 2)[-1].strip() if len(text.split()) > 2 else ""
            if ref.lower() == "all":
                self._handle_record_all(False)
            elif ref:
                self._handle_record_camera(ref, False)
            else:
                self._add_warning("Usage: stop record <cam> | stop record all")
            return
        if lowered.startswith("record all"):
            self._handle_record_all(True)
            return
        if lowered.startswith("record "):
            ref = text.split(" ", 1)[1].strip()
            self._handle_record_camera(ref, True)
            return
        if lowered.startswith("motion box all") or lowered == "motion box":
            self._handle_motion_box_all(on=True)
            return
        if lowered.startswith("motion box off all"):
            self._handle_motion_box_all(on=False)
            return
        if lowered.startswith("motion box off "):
            ref = text.split(" ", 3)[-1].strip()
            self._handle_motion_box_camera(ref, on=False)
            return
        if lowered.startswith("motion box "):
            ref = text.split(" ", 2)[-1].strip()
            if ref.lower() == "off":
                self._handle_motion_box_all(on=False)
            else:
                self._handle_motion_box_camera(ref, on=True)
            return
        if lowered.startswith("detect all") or lowered == "detect":
            self._handle_detect_all(on=True)
            return
        if lowered.startswith("detect off all"):
            self._handle_detect_all(on=False)
            return
        if lowered.startswith("detect off "):
            ref = text.split(" ", 2)[-1].strip()
            self._handle_detect_camera(ref, on=False)
            return
        if lowered.startswith("detect "):
            ref = text.split(" ", 1)[1].strip()
            if ref.lower() == "off":
                self._handle_detect_all(on=False)
            else:
                self._handle_detect_camera(ref, on=True)
            return
        if lowered.startswith("sensitivity ") or lowered.startswith("sens "):
            val_str = text.split(" ", 1)[1].strip() if " " in text else ""
            self._handle_sensitivity(val_str)
            return
        if lowered == "layouts" or lowered == "layout list":
            self._handle_layout_list()
            return
        if lowered.startswith("layout stop "):
            ref = text.split(" ", 2)[-1].strip()
            self._handle_layout_stop(ref)
            return
        if lowered.startswith("layout run "):
            ref = text.split(" ", 2)[-1].strip()
            self._handle_layout_run(ref)
            return
        if lowered.startswith("layout "):
            ref = text.split(" ", 1)[1].strip()
            self._handle_layout_load(ref)
            return
        if lowered.startswith("status"):
            self.handle_quick_status()
            return
        if lowered.startswith("cameras"):
            self.handle_quick_cameras()
            return
        if lowered in {"show all", "showall", "show all cameras"}:
            self._handle_show_all()
            return
        if lowered.startswith("open camera") or lowered.startswith("camera "):
            ref = text.split(" ", 1)[1] if " " in text else ""
            self._open_camera_widget(ref.strip())
            return
        if lowered.startswith("grid"):
            self._arrange_grid(lowered)
            return
        if lowered.startswith("cascade"):
            self._handle_cascade(lowered)
            return
        if lowered.startswith("tile"):
            self._handle_tile(lowered)
            return
        if lowered.startswith("fullscreen"):
            self._handle_fullscreen(text)
            return
        if lowered.startswith("patrol"):
            self._handle_patrol(text)
            return
        if lowered == "snap grid":
            self._add_system("Auto-fitting cameras into seamless grid\u2026")
            self._send_ipc({"cmd": "snap_grid"})
            return
        if lowered == "snap":
            self._add_system("Toggling snap-to-grid\u2026")
            self._send_ipc({"cmd": "toggle_snap"})
            return
        if lowered in {"minimize all", "min all"}:
            self._add_system("Minimizing all camera widgets\u2026")
            self._send_ipc({"cmd": "minimize_all", "target": "camera"})
            return
        if lowered in {"restore all"}:
            self._add_system("Restoring all camera widgets\u2026")
            self._send_ipc({"cmd": "restore_all", "target": "camera"})
            return
        if lowered in {"close all", "close all cameras"}:
            self._add_system("Closing all camera widgets\u2026")
            self._send_ipc({"cmd": "close_all", "target": "camera"})
            return
        if lowered.startswith("alerts"):
            self.handle_quick_alerts()
            return
        if lowered.startswith("tools"):
            self.handle_quick_tools()
            return

        # Default: AI chat
        self._send_ai(text)

    def keyPressEvent(self, event):
        if event.key() == Qt.Key.Key_Up and self.history:
            if self.history_index + 1 < len(self.history):
                self.history_index += 1
                self.input.setText(self.history[-1 - self.history_index])
                self.input.setCursorPosition(len(self.input.text()))
                return
        if event.key() == Qt.Key.Key_Down and self.history:
            if self.history_index > 0:
                self.history_index -= 1
                self.input.setText(self.history[-1 - self.history_index])
                self.input.setCursorPosition(len(self.input.text()))
                return
            elif self.history_index == 0:
                self.history_index = -1
                self.input.clear()
                return
        super().keyPressEvent(event)

    def toggle_agent(self):
        self.agent_active = not self.agent_active
        self.agent_btn.setChecked(self.agent_active)
        self.agent_btn.setText("🤖 Agent ON" if self.agent_active else "🤖 Agent OFF")
        self.agent_btn.setStyleSheet(self._pill_style(self.agent_active))
        self._add_system("Agent activated" if self.agent_active else "Agent deactivated")
        # Show/hide overrides when agent is active
        try:
            self.provider_combo.setVisible(bool(self.agent_active) and self.provider_combo.count() > 0)
            self.model_combo.setVisible(bool(self.agent_active) and self.model_combo.count() > 0)
        except Exception:
            pass
        # If turning ON, ensure backend API is running (best-effort).
        if self.agent_active:
            def _ensure():
                try:
                    ok = self._ensure_backend_running()
                    if ok:
                        self._post_success("Backend API ready.", tool="backend")
                    else:
                        self._post_warning("Backend API not reachable. Try starting it with: python app.py")
                except Exception:
                    pass
            threading.Thread(target=_ensure, daemon=True).start()
        self._persist_state()

    def clear_log(self):
        self.log = []
        self.log_view.clear()
        self._add_system("Terminal cleared")
        self._persist_state()

    # Quick actions ------------------------------------------------------
    def handle_quick_status(self):
        # Local system stats
        uptime = int(time.time() - self.start_time)
        hours = uptime // 3600
        minutes = (uptime % 3600) // 60
        seconds = uptime % 60
        
        local_stats = f"Uptime: {hours}:{minutes:02d}:{seconds:02d}"
        try:
            import psutil
            local_stats += f" | CPU: {psutil.cpu_percent():.0f}% | MEM: {psutil.virtual_memory().percent:.0f}%"
        except:
            pass
            
        self._add_info(local_stats)

        # AI Agent status (from latest background fetch)
        if hasattr(self, "_last_status_data"):
            data = self._last_status_data.get("data") or self._last_status_data
            status_lines = []
            ai = data.get("ai") or data.get("agent") or {}
            if ai:
                online = ai.get("initialized") or ai.get("online") or ai.get("running")
                status_lines.append(f"AI Agent: {'ONLINE' if online else 'OFFLINE'}")
                if ai.get("provider"): status_lines.append(f"Provider: {ai.get('provider')}")
                if ai.get("model"): status_lines.append(f"Model: {ai.get('model')}")
            
            alerts = data.get("alerts") or {}
            if alerts:
                status_lines.append(f"Alerts: {alerts.get('active', 0)} active")
                
            if status_lines:
                self._add_info(" | ".join(status_lines))
        else:
            self._add_info("AI Status: Fetching...")
            self._fetch_status_async()

    def _fetch_status_async(self) -> None:
        """
        Schedule a status fetch without blocking the UI thread.
        Uses an in-flight guard so slow backends don't stack up requests and cause stutters.
        """
        with self._status_fetch_lock:
            if self._status_fetch_inflight:
                return
            self._status_fetch_inflight = True

        def worker():
            try:
                self._fetch_status()
            finally:
                with self._status_fetch_lock:
                    self._status_fetch_inflight = False

        threading.Thread(target=worker, daemon=True).start()

    def handle_quick_cameras(self):
        def worker():
            try:
                cams = self._fetch_camera_devices()
                if cams:
                    if not cams:
                        self._post_warning("No cameras available")
                        return
                    self._post_success(f"Available cameras ({len(cams)}):")
                    for cam in cams:
                        name = cam.get("name") or cam.get("id") or "Camera"
                        ip = cam.get("ip_address") or cam.get("ip") or cam.get("rtsp_url") or ""
                        self._post_output(f"• {name} {f'({ip})' if ip else ''}")
                else:
                    self._post_warning("No cameras available")
            except Exception as e:
                self._post_error(f"Camera list error: {e}")

        threading.Thread(target=worker, daemon=True).start()

    def handle_quick_alerts(self):
        # No dedicated alerts endpoint found; fall back to agent status alerts summary
        self._add_system("Checking alerts (agent status)…")
        self.handle_quick_status()

    def handle_quick_tools(self):
        self._add_system("Requesting tool list…")

        def worker():
            try:
                res = requests.get(f"{self.API_BASE}/ai/status", timeout=6)
                if res.ok:
                    data = res.json().get("data") or res.json()
                    tools = data.get("tools") or data.get("tool_schema", {}).get("tools") or []
                    if tools:
                        self._post_success(f"Tools ({len(tools)}):")
                        for tool in tools[:30]:
                            name = tool.get("name") or tool.get("id") or "tool"
                            desc = tool.get("description") or ""
                            self._post_output(f"• {name}: {desc}")
                        if len(tools) > 30:
                            self._post_output(f"... and {len(tools)-30} more")
                        return
                # Fallback: ask the AI agent to enumerate tools
                chat_payload = {
                    "message": "List the available AI tools with a short description.",
                    "context": {"source": "pyqt-terminal", "intent": "list_tools"},
                }
                chat_res = requests.post(f"{self.API_BASE}/ai/chat", json=chat_payload, timeout=20)
                if not chat_res.ok:
                    self._post_error(f"Tool list failed ({chat_res.status_code})")
                    return
                j = chat_res.json()
                data = j.get("data") or j
                tools = data.get("tools") or []
                msg = data.get("message") or j.get("message") or "AI responded."
                if tools:
                    self._post_success(f"Tools ({len(tools)}):")
                    for tool in tools[:30]:
                        name = tool.get("name") or tool.get("id") or "tool"
                        desc = tool.get("description") or ""
                        self._post_output(f"• {name}: {desc}")
                    if len(tools) > 30:
                        self._post_output(f"... and {len(tools)-30} more")
                self._post_success("AI tool summary:")
                self._post_output(msg)
            except Exception as e:
                self._post_error(f"Tool list error: {e}")

        threading.Thread(target=worker, daemon=True).start()

    # Agent settings ------------------------------------------------------
    def _fetch_llm_state(self):
        providers = []
        config = {}
        user_keys = {}
        try:
            res = requests.get(f"{self.API_BASE}/llm/providers", timeout=5)
            if res.ok:
                providers = res.json().get("providers", [])
        except Exception:
            pass
        try:
            res = requests.get(f"{self.API_BASE}/llm/config", timeout=5)
            if res.ok:
                config = res.json().get("config", {})
        except Exception:
            pass
        try:
            res = requests.get(f"{self.API_BASE}/llm/user-keys", timeout=5)
            if res.ok:
                user_keys = res.json().get("keys", {})
        except Exception:
            pass
        return providers, config, user_keys

    def _parse_priority_input(self, primary: str, failover_text: str) -> List[str]:
        order: List[str] = []
        if primary:
            order.append(primary)
        for item in (failover_text or "").split(","):
            pid = item.strip().lower()
            if pid and pid not in order:
                order.append(pid)
        return order

    def show_agent_settings_dialog(self):
        providers, config, user_keys = self._fetch_llm_state()
        provider_ids = [p.get("id") for p in providers if p.get("id")]
        if not provider_ids:
            provider_ids = ["openai", "grok", "anthropic", "huggingface_local"]

        current_primary = self.agent_settings.get("primary") or config.get("provider") or provider_ids[0]
        current_failover = self.agent_settings.get("failover") or config.get("provider_priority", [])[1:]

        dlg = QDialog(self)
        dlg.setWindowTitle("Agent Settings")
        form = QFormLayout(dlg)

        primary_combo = QComboBox()
        primary_combo.addItems(provider_ids)
        if current_primary in provider_ids:
            primary_combo.setCurrentText(current_primary)

        failover_input = QLineEdit(", ".join(current_failover or []))

        openai_key = QLineEdit(user_keys.get("openai", {}).get("api_key", ""))
        grok_key = QLineEdit(user_keys.get("grok", {}).get("api_key", ""))
        anthropic_key = QLineEdit(user_keys.get("anthropic", {}).get("api_key", ""))

        for field in (openai_key, grok_key, anthropic_key):
            field.setEchoMode(QLineEdit.EchoMode.Password)

        openai_status = QLabel("Not tested")
        grok_status = QLabel("Not tested")
        anthropic_status = QLabel("Not tested")
        for label in (openai_status, grok_status, anthropic_status):
            label.setStyleSheet("color: #9aa4b2;")

        openai_test = QPushButton("Test")
        grok_test = QPushButton("Test")
        anthropic_test = QPushButton("Test")

        use_local_vision = QCheckBox("Use local vision service")
        use_local_vision.setChecked(bool(self.agent_settings.get("use_local_vision", True)))

        form.addRow("Primary provider", primary_combo)
        form.addRow("Failover order (comma-separated)", failover_input)
        def _make_key_row(field: QLineEdit, button: QPushButton) -> QWidget:
            row = QWidget()
            layout = QHBoxLayout(row)
            layout.setContentsMargins(0, 0, 0, 0)
            layout.addWidget(field, stretch=1)
            layout.addWidget(button)
            return row

        form.addRow("OpenAI key", _make_key_row(openai_key, openai_test))
        form.addRow("", openai_status)
        form.addRow("Grok/xAI key", _make_key_row(grok_key, grok_test))
        form.addRow("", grok_status)
        form.addRow("Anthropic key", _make_key_row(anthropic_key, anthropic_test))
        form.addRow("", anthropic_status)
        form.addRow("", use_local_vision)

        save_btn = QPushButton("Save")
        cancel_btn = QPushButton("Cancel")
        btn_row = QHBoxLayout()
        btn_row.addWidget(save_btn)
        btn_row.addWidget(cancel_btn)
        form.addRow(btn_row)

        def save():
            primary = primary_combo.currentText().strip().lower()
            priority = self._parse_priority_input(primary, failover_input.text())

            # Merge config while preserving other fields
            cfg = config.copy() if isinstance(config, dict) else {}
            cfg["provider"] = primary
            cfg["provider_priority"] = priority
            cfg.setdefault("providers", {})
            # Ensure provider entries exist for edited keys
            for pid in ["openai", "grok", "anthropic", "huggingface_local"]:
                cfg["providers"].setdefault(pid, {"enabled": True})

            keys_payload = {
                "openai": {"api_key": openai_key.text().strip()},
                "grok": {"api_key": grok_key.text().strip()},
                "anthropic": {"api_key": anthropic_key.text().strip()},
            }

            def worker():
                try:
                    cfg_res = requests.put(f"{self.API_BASE}/llm/config", json=cfg, timeout=8)
                    if not cfg_res.ok:
                        self._post_error(f"Config save failed ({cfg_res.status_code})")
                        return

                    key_res = requests.put(f"{self.API_BASE}/llm/user-keys", json=keys_payload, timeout=8)
                    if not key_res.ok:
                        self._post_error(f"Key save failed ({key_res.status_code})")
                        return

                    reload_res = requests.post(f"{self.API_BASE}/llm/reload", timeout=8)
                    if not reload_res.ok:
                        self._post_warning(f"Reload warning ({reload_res.status_code})")

                    self.agent_settings.update(
                        {
                            "primary": primary,
                            "failover": priority[1:],
                            "provider_priority": priority,
                            "use_local_vision": use_local_vision.isChecked(),
                            "api_keys": keys_payload,
                        }
                    )
                    self._persist_agent_settings()
                    self._post_success(f"Agent settings saved: {', '.join(priority)}")
                except Exception as e:
                    self._post_error(f"Agent settings error: {e}")

            threading.Thread(target=worker, daemon=True).start()
            dlg.accept()

        def _set_status(label: QLabel, text: str, ok: Optional[bool] = None):
            def apply():
                label.setText(text)
                if ok is True:
                    label.setStyleSheet("color: #39d98a;")
                elif ok is False:
                    label.setStyleSheet("color: #f16063;")
                else:
                    label.setStyleSheet("color: #9aa4b2;")

            QTimer.singleShot(0, apply)

        def _test_key(provider_id: str, field: QLineEdit, label: QLabel):
            key_value = field.text().strip()
            if not key_value:
                _set_status(label, "Missing key", ok=False)
                return
            _set_status(label, "Testing…")

            def worker():
                try:
                    res = requests.post(
                        f"{self.API_BASE}/llm/test-key",
                        json={"provider": provider_id, "api_key": key_value},
                        timeout=12,
                    )
                    if not res.ok:
                        _set_status(label, f"HTTP {res.status_code}", ok=False)
                        return
                    payload = res.json() if isinstance(res.json(), dict) else {}
                    if not payload.get("success", True):
                        _set_status(label, payload.get("message") or "Test failed", ok=False)
                        return
                    valid = bool(payload.get("valid"))
                    msg = payload.get("message") or ("Valid" if valid else "Invalid")
                    models = payload.get("models") or []
                    if valid and models:
                        msg = f"{msg} ({len(models)} models)"
                    _set_status(label, msg, ok=valid)
                except Exception as exc:
                    _set_status(label, f"Error: {exc}", ok=False)

            threading.Thread(target=worker, daemon=True).start()

        save_btn.clicked.connect(save)
        cancel_btn.clicked.connect(dlg.reject)
        openai_test.clicked.connect(lambda: _test_key("openai", openai_key, openai_status))
        grok_test.clicked.connect(lambda: _test_key("grok", grok_key, grok_status))
        anthropic_test.clicked.connect(lambda: _test_key("anthropic", anthropic_key, anthropic_status))
        dlg.exec()

    # Networking ---------------------------------------------------------
    def _backend_is_responding(self) -> bool:
        try:
            r = requests.get(f"{self.API_BASE}/ai/status", timeout=(0.8, 1.2))
            return bool(r.ok)
        except Exception:
            return False

    def _ensure_backend_running(self) -> bool:
        """
        Ensure the backend API is reachable at API_BASE.
        If it isn't, attempt to start `app.py` automatically and wait for readiness.
        """
        if self._backend_is_responding():
            return True

        with TerminalWidget._backend_lock:
            if self._backend_is_responding():
                return True

            # If we already spawned it and it's still running, just wait.
            try:
                if TerminalWidget._backend_process and TerminalWidget._backend_process.poll() is None:
                    pass
                else:
                    is_frozen = bool(getattr(sys, "frozen", False))
                    if (not is_frozen) and (not self.BACKEND_ENTRY.exists()):
                        return False

                    try:
                        self.BACKEND_LOG.parent.mkdir(parents=True, exist_ok=True)
                    except Exception:
                        pass

                    env = dict(os.environ)
                    # Force reliable server mode for desktop-driven autostart.
                    env["KNOXNET_SIMPLE_SERVER"] = "1"
                    env.setdefault("PYTHONIOENCODING", "utf-8")

                    creationflags = 0
                    try:
                        if os.name == "nt" and hasattr(subprocess, "CREATE_NO_WINDOW"):
                            creationflags = subprocess.CREATE_NO_WINDOW
                    except Exception:
                        creationflags = 0

                    try:
                        log_f = open(self.BACKEND_LOG, "a", encoding="utf-8", errors="ignore")
                    except Exception:
                        log_f = subprocess.DEVNULL  # type: ignore

                    # Frozen builds should spawn the bundled backend via the same executable.
                    # Dev/source continues to spawn `python app.py`.
                    cmd = [sys.executable, str(self.BACKEND_ENTRY)]
                    cwd = str(self.APP_ROOT)
                    if is_frozen:
                        cmd = [sys.executable, "--run-backend"]
                        cwd = str(self.APP_ROOT)

                    TerminalWidget._backend_process = subprocess.Popen(
                        cmd,
                        cwd=cwd,
                        env=env,
                        stdout=log_f,
                        stderr=subprocess.STDOUT,
                        text=True,
                        creationflags=creationflags,
                    )
            except Exception:
                return False

        # Wait for readiness (outside lock)
        deadline = time.time() + 25.0
        while time.time() < deadline:
            if self._backend_is_responding():
                return True
            time.sleep(0.5)
        return False

    def _send_ai(self, message: str):
        self._add_system("Agent thinking…" if self.agent_active else "Processing…")

        def worker():
            try:
                if not self._ensure_backend_running():
                    self.error_signal.emit("Backend API is not running (http://localhost:5000). Start it from the app (or run: python app.py).")
                    return
                # Provide a minimal camera list so the agent can reference real camera names
                devices_payload: List[Dict[str, Any]] = []
                try:
                    cams = self._fetch_camera_devices()
                    for d in cams:
                        if not isinstance(d, dict):
                            continue
                        name = d.get("name")
                        cid = d.get("id") or d.get("camera_id")
                        if name or cid:
                            devices_payload.append(
                                {
                                    "type": "camera",
                                    "id": cid,
                                    "name": name,
                                    "ip": d.get("ip") or d.get("ip_address"),
                                }
                            )
                except Exception:
                    devices_payload = []

                payload = {
                    "message": message,
                    "context": {
                        "source": "pyqt-terminal",
                        # Provide stable identifiers so the backend (and logs) can correlate sessions.
                        # The backend currently doesn't require these, but they’re useful and forward-compatible.
                        "terminalId": self.widget_key,
                        "sessionId": f"pyqt_terminal_{self.widget_key}",
                        "devices": devices_payload,
                        "llm": self._build_llm_context_override(),
                    },
                }
                res = requests.post(f"{self.API_BASE}/ai/chat", json=payload, timeout=45)
                if not res.ok:
                    # Try to surface backend error details (critical for debugging 500s)
                    detail = ""
                    try:
                        j = res.json()
                        detail = (
                            j.get("message")
                            or (j.get("data") or {}).get("message")
                            or (j.get("data") or {}).get("error")
                            or j.get("error")
                            or ""
                        )
                    except Exception:
                        detail = (res.text or "").strip()
                    msg = f"AI error ({res.status_code})" + (f": {detail[:300]}" if detail else "")
                    self.error_signal.emit(msg)
                    return
                body = res.json()
                if body.get("success") is False:
                    msg = body.get("message") or "AI call failed"
                    self.error_signal.emit(msg)
                    return
                self.ai_response_signal.emit(body)
            except Exception as e:
                self.error_signal.emit(f"AI request failed: {e}")

        threading.Thread(target=worker, daemon=True).start()

    def _fetch_status(self):
        try:
            res = requests.get(f"{self.API_BASE}/ai/status", timeout=6)
            if res.ok:
                self.status_signal.emit(res.json())
            else:
                self.error_signal.emit(f"Status failed ({res.status_code})")
        except Exception as e:
            self.error_signal.emit(f"Status error: {e}")

    # IPC helpers --------------------------------------------------------
    def _send_ipc(self, payload: Dict[str, Any]):
        """
        Send a command to the local IPC server (desktop/ipc_server.py).
        """
        try:
            with socket.create_connection(("127.0.0.1", 5555), timeout=1.5) as sock:
                data = json.dumps(payload).encode("utf-8")
                sock.sendall(data)
        except Exception as e:
            self._post_error(f"IPC send failed: {e}")

    def _open_camera_widget(self, camera_ref: str):
        if not camera_ref:
            self._add_warning("Usage: camera <name/id/ip>")
            return
        self._add_system(f"Opening camera widget: {camera_ref}")
        self._send_ipc({"cmd": "spawn_camera", "camera_ref": camera_ref})

    def _arrange_grid(self, text: str):
        """Parse rich grid command syntax and dispatch via IPC.

        Examples::

            grid                          smart auto grid
            grid 3                        3 columns, auto rows
            grid 3x2                      3 columns, 2 rows
            grid sort name                auto grid sorted by name
            grid sort name desc           auto grid sorted by name descending
            grid sort id                  auto grid sorted by camera id
            grid fill                     stretch to fill cells (ignore AR)
            grid seamless                 tight packing, zero gaps
            grid compact                  alias for seamless
            grid focus <camera>           one camera large + sidebar
            grid 3 sort name              combine columns with sort
            grid 3x2 fill sort name       combine everything
        """
        parts = text.split()
        cols = None
        rows = None
        target = "camera"
        sort = None
        mode = "fit"
        focus_ref = None
        gap = 2

        i = 1  # skip "grid"
        while i < len(parts):
            tok = parts[i]

            if "x" in tok and tok[0].isdigit():
                try:
                    c, r = tok.split("x", 1)
                    cols = max(1, int(c))
                    rows = max(1, int(r))
                except Exception:
                    pass
                i += 1
                continue

            if tok.isdigit():
                cols = max(1, int(tok))
                i += 1
                continue

            if tok == "sort" and i + 1 < len(parts):
                sort = parts[i + 1]
                i += 2
                if i < len(parts) and parts[i] in ("desc", "asc"):
                    if parts[i] == "desc":
                        sort += " desc"
                    i += 1
                continue

            if tok in ("fill", "stretch"):
                mode = "fill"
                i += 1
                continue
            if tok in ("seamless", "tight", "compact"):
                mode = "seamless"
                gap = 0
                i += 1
                continue
            if tok == "fit":
                mode = "fit"
                i += 1
                continue

            if tok == "focus":
                mode = "focus"
                if i + 1 < len(parts):
                    focus_ref = " ".join(parts[i + 1:])
                    i = len(parts)
                else:
                    i += 1
                continue

            if tok in ("camera", "terminal", "all"):
                target = tok
                i += 1
                continue

            i += 1

        desc = []
        if cols and rows:
            desc.append(f"{cols}\u00d7{rows}")
        elif cols:
            desc.append(f"{cols} cols")
        else:
            desc.append("auto")
        if mode != "fit":
            desc.append(mode)
        if sort:
            desc.append(f"sort={sort}")
        if focus_ref:
            desc.append(f"focus={focus_ref}")

        self._add_system(f"Grid layout: {', '.join(desc)} ({target})")
        self._send_ipc({
            "cmd": "arrange_grid",
            "cols": cols,
            "rows": rows,
            "target": target,
            "sort": sort,
            "gap": gap,
            "mode": mode,
            "focus_ref": focus_ref,
        })

    # ---- Additional layout commands ----

    def _handle_cascade(self, text: str):
        """Parse ``cascade [sort <criterion>]`` and dispatch."""
        parts = text.split()
        sort = None
        if "sort" in parts:
            idx = parts.index("sort")
            if idx + 1 < len(parts):
                sort = parts[idx + 1]
        self._add_system("Cascading camera widgets\u2026")
        self._send_ipc({"cmd": "arrange_cascade", "target": "camera", "sort": sort})

    def _handle_tile(self, text: str):
        """Parse ``tile [h|v|horizontal|vertical]`` and dispatch."""
        parts = text.split()
        direction = "horizontal"
        for p in parts[1:]:
            if p in ("v", "vertical", "vert"):
                direction = "vertical"
                break
            if p in ("h", "horizontal", "horiz"):
                direction = "horizontal"
                break
        self._add_system(f"Tiling cameras {direction}ly\u2026")
        self._send_ipc({"cmd": "arrange_tile", "direction": direction, "target": "camera"})

    def _handle_fullscreen(self, text: str):
        """Parse ``fullscreen [<camera_ref>]`` and dispatch."""
        parts = text.split()
        camera_ref = " ".join(parts[1:]) if len(parts) > 1 else None
        if camera_ref:
            self._add_system(f"Fullscreen: {camera_ref}")
        else:
            self._add_system("Fullscreen: first camera")
        self._send_ipc({"cmd": "arrange_fullscreen", "camera_ref": camera_ref, "target": "camera"})

    def _handle_patrol(self, text: str):
        """Parse ``patrol [interval_sec]`` and toggle patrol mode."""
        parts = text.split()
        interval = None
        if len(parts) > 1:
            try:
                interval = float(parts[1])
            except ValueError:
                pass
        cmd: dict = {"cmd": "patrol_toggle"}
        if interval is not None:
            cmd["interval"] = interval
            self._add_system(f"Toggling patrol ({interval:.0f}s interval)\u2026")
        else:
            self._add_system("Toggling patrol\u2026")
        self._send_ipc(cmd)

    def _handle_show_all(self):
        """Offline 'show all' command — spawn every camera and tile them in a VMS grid.
        Reads camera list from local files (no LLM / no backend API required)."""
        self._add_system("Loading all cameras…")

        def worker():
            cams = self._fetch_camera_devices()
            if not cams:
                self._post_warning("No cameras found in system.")
                return

            cam_ids = []
            for cam in cams:
                cid = cam.get("id") or cam.get("name")
                if cid:
                    cam_ids.append(cid)

            if not cam_ids:
                self._post_warning("No valid camera IDs found.")
                return

            n = len(cam_ids)
            import math
            cols = math.ceil(math.sqrt(n))
            rows = math.ceil(n / cols)

            cam_names = [cam.get("name") or cam.get("id") for cam in cams if cam.get("id") or cam.get("name")]
            self._post_success(f"Showing all {n} cameras in {cols}x{rows} grid:")
            for name in cam_names:
                self._post_output(f"  • {name}")

            self._send_ipc({
                "cmd": "show_all_cameras",
                "camera_ids": cam_ids,
                "cols": cols,
                "rows": rows,
            })

        threading.Thread(target=worker, daemon=True).start()

    # Signal slots -------------------------------------------------------
    def _on_ai_response(self, payload: Dict[str, Any]):
        if isinstance(payload, str):
            self._add_success(f"🤖 {payload}")
            return
        data = payload.get("data") or payload
        resp = data.get("response") or data.get("message") or ""
        if resp:
            self._add_success(f"🤖 {resp}")

        # Execute high-signal UI actions returned by the agent (desktop app only).
        # Scoped to camera/grid/motion-watch to avoid surprising side effects.
        try:
            actions = data.get("actions") or []
            if isinstance(actions, list) and actions:
                for action in actions:
                    if not isinstance(action, dict):
                        continue
                    kind = (action.get("kind") or "").strip()
                    kind_l = kind.lower()
                    tool_id = (action.get("tool_id") or action.get("toolId") or "").strip()
                    params = action.get("parameters") if isinstance(action.get("parameters"), dict) else {}

                    if kind == "create_camera_widget":
                        camera_ref = (
                            (params or {}).get("cameraRef")
                            or (action.get("props") or {}).get("camera_name")
                            or (action.get("props") or {}).get("cameraRef")
                            or action.get("camera_id")
                            or action.get("cameraId")
                            or action.get("camera")
                        )
                        if camera_ref:
                            self._add_system(f"Opening camera: {camera_ref}")
                            self._open_camera_widget(str(camera_ref))
                    elif kind in {"create_camera_grid_layout", "create_all_cameras_grid", "reorganize_dashboard", "organize_widgets"}:
                        layout = (params or {}).get("layout") or (action.get("props") or {}).get("layout") or ""
                        cols = 2
                        rows = None
                        try:
                            if isinstance(layout, str) and "x" in layout:
                                c, r = layout.lower().split("x", 1)
                                cols = max(1, int(c.strip() or "2"))
                                rows = max(1, int(r.strip() or "2"))
                            elif isinstance(layout, str) and layout.strip().isdigit():
                                cols = max(1, int(layout.strip()))
                            elif isinstance(layout, int):
                                cols = max(1, layout)
                        except Exception:
                            cols, rows = 2, None
                        self._add_system(f"Arranging camera widgets into grid ({cols}x{rows or 'auto'})")
                        self._send_ipc({"cmd": "arrange_grid", "cols": cols, "rows": rows, "target": "camera"})
                    elif kind_l == "execute_tool":
                        # Support backend returning lowercase execute_tool + tool_id
                        if tool_id == "motion_detector_watch":
                            camera_ref = (params or {}).get("cameraRef") or (params or {}).get("camera_id") or (params or {}).get("cameraId")
                            duration_sec = (params or {}).get("durationSec") or (params or {}).get("duration_sec")
                            duration_min = (params or {}).get("durationMinutes") or (params or {}).get("duration_minutes")
                            if duration_sec is None and duration_min is not None:
                                try:
                                    duration_sec = int(float(duration_min) * 60)
                                except Exception:
                                    duration_sec = None
                            if camera_ref:
                                ipc_payload = {"cmd": "start_motion_watch", "camera_ref": str(camera_ref)}
                                if duration_sec is not None:
                                    ipc_payload["duration_sec"] = int(duration_sec)
                                self._add_system(f"Starting motion watch: {camera_ref}")
                                self._send_ipc(ipc_payload)
                            else:
                                self._add_warning("Motion watch requested but no camera was specified.")
                        elif tool_id == "stop_motion_detector_watch":
                            camera_ref = (params or {}).get("cameraRef") or (params or {}).get("camera_id") or (params or {}).get("cameraId")
                            if camera_ref:
                                self._add_system(f"Stopping motion watch: {camera_ref}")
                                self._send_ipc({"cmd": "stop_motion_watch", "camera_id": str(camera_ref)})
                            else:
                                self._add_warning("Stop motion watch requested but no camera was specified.")
                        elif tool_id == "set_motion_boxes":
                            camera_ref = (params or {}).get("cameraRef") or (params or {}).get("camera") or (params or {}).get("camera_id")
                            enabled = (params or {}).get("enabled")
                            if enabled is None:
                                enabled = True
                            if camera_ref:
                                self._add_system(f"Setting motion boxes on {camera_ref}: {'ON' if bool(enabled) else 'OFF'}")
                                self._send_ipc({"cmd": "set_motion_boxes", "camera_ref": str(camera_ref), "enabled": bool(enabled)})
                            else:
                                self._add_warning("Motion boxes requested but no camera was specified.")
                        elif tool_id == "snapshot_detect":
                            # Execute snapshot detection via backend API and print a human-readable count.
                            # This is critical for questions like "how many cars" in the PyQt terminal.
                            try:
                                camera_ref = (params or {}).get("cameraRef") or (params or {}).get("camera") or (params or {}).get("camera_id")
                                object_classes = (params or {}).get("objectClasses") or (params or {}).get("object_classes") or []
                                if isinstance(object_classes, str):
                                    object_classes = [object_classes]
                                if not isinstance(object_classes, list) or not object_classes:
                                    object_classes = ["car"]

                                if not camera_ref:
                                    self._add_warning("Snapshot detection requested but no camera was specified.")
                                else:
                                    cam = self._resolve_camera_ref(str(camera_ref))
                                    if not cam:
                                        self._add_error(f"Snapshot detection: camera not found: {camera_ref}")
                                    else:
                                        self._add_system(f"Running snapshot detection on {cam['name']}…")
                                        result = self._run_snapshot_detect(cam["id"], cam["name"], object_classes, params)
                                        if result.get("success"):
                                            summary = result.get("summary") or {}
                                            total = int(result.get("total_objects") or 0)
                                            if total <= 0 or not summary:
                                                self._add_success(f"🔍 {cam['name']}: no {', '.join(object_classes)} detected")
                                            else:
                                                parts = []
                                                try:
                                                    for cls, count in summary.items():
                                                        parts.append(f"{count} {cls}{'s' if int(count) != 1 else ''}")
                                                except Exception:
                                                    parts = [str(summary)]
                                                self._add_success(f"🔍 {cam['name']}: " + ", ".join(parts))
                                        else:
                                            err = result.get("error") or "Detection failed"
                                            self._add_error(f"Snapshot detection failed: {err}", tool="snapshot_detect")
                            except Exception as e:
                                self._add_error(f"Snapshot detection error: {e}", tool="snapshot_detect")
                        elif tool_id == "take_camera_snapshot":
                            # Execute snapshot + local vision captioning and render results.
                            try:
                                camera_ref = (params or {}).get("cameraRef") or (params or {}).get("camera") or (params or {}).get("camera_id")
                                analysis_prompt = (params or {}).get("analysisPrompt") or (params or {}).get("prompt") or ""
                                if not analysis_prompt:
                                    analysis_prompt = "Describe what you see."

                                if not camera_ref:
                                    self._add_warning("Snapshot requested but no camera was specified.")
                                else:
                                    cam = self._resolve_camera_ref(str(camera_ref))
                                    if not cam:
                                        self._add_error(f"Snapshot: camera not found: {camera_ref}")
                                    else:
                                        self._add_system(f"📸 Capturing snapshot from {cam['name']}…")
                                        img_b64 = self._fetch_snapshot_b64(cam["id"])
                                        if img_b64:
                                            # Show snapshot inline
                                            self._add_line(f"Snapshot: {cam['name']}", kind="system", image_b64=img_b64, tool="take_camera_snapshot")
                                            # Run vision captioning (best-effort)
                                            caption = self._run_vision_caption(img_b64, analysis_prompt)
                                            if caption:
                                                self._add_success(f"🧠 {cam['name']}: {caption}", tool="take_camera_snapshot")
                                            else:
                                                self._add_info(f"{cam['name']}: Snapshot captured (vision caption unavailable)")
                                        else:
                                            self._add_error(f"Failed to capture snapshot for {cam['name']}", tool="take_camera_snapshot")
                            except Exception as e:
                                self._add_error(f"Snapshot tool error: {e}", tool="take_camera_snapshot")
                        elif tool_id == "open_ptz_widget":
                            camera_ref = (params or {}).get("cameraRef") or (params or {}).get("camera") or (params or {}).get("camera_id")
                            undocked = bool((params or {}).get("undocked")) if (params or {}).get("undocked") is not None else False
                            if camera_ref:
                                self._add_system(f"Opening PTZ controls for {camera_ref}…")
                                self._send_ipc({"cmd": "open_ptz_widget", "camera_ref": str(camera_ref), "undocked": undocked})
                            else:
                                self._add_warning("PTZ requested but no camera was specified.")
                        elif tool_id == "open_audio_widget":
                            camera_ref = (params or {}).get("cameraRef") or (params or {}).get("camera") or (params or {}).get("camera_id")
                            undocked = bool((params or {}).get("undocked")) if (params or {}).get("undocked") is not None else False
                            if camera_ref:
                                self._add_system(f"Opening audio controls for {camera_ref}…")
                                self._send_ipc({"cmd": "open_audio_widget", "camera_ref": str(camera_ref), "undocked": undocked})
                            else:
                                self._add_warning("Audio requested but no camera was specified.")
                        elif tool_id == "open_depth_map_widget":
                            camera_ref = (params or {}).get("cameraRef") or (params or {}).get("camera") or (params or {}).get("camera_id")
                            color_scheme = (params or {}).get("colorScheme") or (params or {}).get("colormap")
                            mode = (params or {}).get("mode")
                            enabled = (params or {}).get("enabled")
                            if enabled is None:
                                enabled = True
                            if camera_ref:
                                self._add_system(f"Opening depth map for {camera_ref}…")
                                self._send_ipc({
                                    "cmd": "open_depth_map_widget",
                                    "camera_ref": str(camera_ref),
                                    "enabled": bool(enabled),
                                    **({"colorScheme": str(color_scheme)} if color_scheme else {}),
                                    **({"mode": str(mode)} if mode else {}),
                                })
                            else:
                                self._add_warning("Depth map requested but no camera was specified.")
                        elif tool_id in {"events_search", "events.search"}:
                            # Search the local event index and render a timeline with thumbnails
                            try:
                                self._add_system("Searching events…")
                                threading.Thread(target=self._run_events_search, args=(params or {},), daemon=True).start()
                            except Exception as e:
                                self._add_error(f"Events search error: {e}", tool="events_search")
                        elif tool_id in {"events_count", "events.count"}:
                            # Count/aggregate using the local event index (no timeline)
                            try:
                                self._add_system("Counting events…")
                                threading.Thread(target=self._run_events_count, args=(params or {},), daemon=True).start()
                            except Exception as e:
                                self._add_error(f"Events count error: {e}", tool="events_count")
                        elif tool_id in {"events_report", "events.report"}:
                            # Generate a coherent HTML report (timeline + crops + file links)
                            try:
                                threading.Thread(target=self._events_report, args=(params or {},), daemon=True).start()
                            except Exception as e:
                                self._add_error(f"Events report error: {e}", tool="events_report")
                    elif kind_l in {"take_camera_snapshot", "camera_snapshot"}:
                        # Some responses may encode snapshots as direct kinds rather than execute_tool.
                        try:
                            camera_ref = (params or {}).get("cameraRef") or action.get("camera_id") or action.get("cameraId") or (params or {}).get("camera_id")
                            analysis_prompt = (params or {}).get("analysisPrompt") or action.get("analysis_prompt") or action.get("analysisPrompt") or ""
                            if not analysis_prompt:
                                analysis_prompt = "Describe what you see."
                            if camera_ref:
                                cam = self._resolve_camera_ref(str(camera_ref))
                                if cam:
                                    self._add_system(f"📸 Capturing snapshot from {cam['name']}…")
                                    img_b64 = self._fetch_snapshot_b64(cam["id"])
                                    if img_b64:
                                        self._add_line(f"Snapshot: {cam['name']}", kind="system", image_b64=img_b64, tool="take_camera_snapshot")
                                        caption = self._run_vision_caption(img_b64, analysis_prompt)
                                        if caption:
                                            self._add_success(f"🧠 {cam['name']}: {caption}", tool="take_camera_snapshot")
                            else:
                                self._add_warning("Snapshot requested but no camera was specified.")
                        except Exception:
                            pass
        except Exception:
            # Best-effort: never let action execution break the terminal UI
            pass
        tools_list = data.get("tools") or []
        if isinstance(tools_list, list) and tools_list:
            self._add_success(f"Tools ({len(tools_list)}):")
            for tool in tools_list:
                if not isinstance(tool, dict):
                    continue
                name = tool.get("name") or tool.get("id") or "tool"
                desc = tool.get("description") or ""
                self._add_output(f"• {name}: {desc}")
        tools = data.get("toolsExecuted") or data.get("tools_executed") or []
        if isinstance(tools, list):
            for tool_exec in tools[:3]:
                tool_id = tool_exec.get("toolId") or tool_exec.get("tool") or tool_exec.get("name")
                result = tool_exec.get("result") or {}
                success = result.get("success", True)
                msg = result.get("message") or result.get("data", {}).get("apiAnalysis") or ""
                if success:
                    self._add_success(f"✅ {tool_id}: {msg or 'Completed'}", tool=tool_id, model=tool_exec.get("model"))
                    img_b64 = result.get("data", {}).get("imageBase64")
                    if img_b64 and len(img_b64) < 120000:
                        self._add_line(f"📸 {tool_id} snapshot", kind="system", image_b64=img_b64, tool=tool_id)
                else:
                    self._add_error(f"❌ {tool_id}: {result.get('error','failed')}", tool=tool_id)
        else:
            # No tools, still mark completion
            pass

    def _on_error(self, message: str):
        self._add_error(message)

    def _on_status_update(self, payload: Dict[str, Any]):
        # Store for explicitly requested status reports
        self._last_status_data = payload
        
        # Update hidden labels for logic compatibility
        data = payload.get("data") or payload
        ai = data.get("ai") or data.get("agent") or {}
        if ai:
            online = ai.get("initialized") or ai.get("online") or ai.get("running")
            if online:
                self.header_label.setText("● Online")
                self.header_label.setStyleSheet("color: #22c55e;")
            else:
                self.header_label.setText("● Offline")
                self.header_label.setStyleSheet("color: #ef4444;")

    # System status (local) ----------------------------------------------
    def _update_system_status(self):
        uptime = int(time.time() - self.start_time)
        hours = uptime // 3600
        minutes = (uptime % 3600) // 60
        seconds = uptime % 60

        cpu = mem = gpu = None
        try:
            import psutil

            cpu = f"{psutil.cpu_percent():.0f}%"
            mem = f"{psutil.virtual_memory().percent:.0f}%"
        except Exception:
            pass  # hide if unavailable

        parts = [f"Uptime {hours}:{minutes:02d}:{seconds:02d}"]
        if cpu:
            parts.append(f"CPU {cpu}")
        if mem:
            parts.append(f"Mem {mem}")
        if gpu:
            parts.append(f"GPU {gpu}")
        self.status_label.setText(" | ".join(parts))

    # Context menu -------------------------------------------------------
    def _show_log_menu(self, pos):
        menu = QMenu(self)
        menu.setStyleSheet(KnoxnetStyle.context_menu())

        toggle_ai = menu.addAction("🤖 Toggle AI Agent")
        toggle_ai.setCheckable(True)
        toggle_ai.setChecked(self.agent_active)
        
        menu.addSeparator()
        
        status_act = menu.addAction("📊 System Status")
        cams_act = menu.addAction("📷 List Cameras")
        alerts_act = menu.addAction("🔔 Recent Alerts")
        tools_act = menu.addAction("🛠️ Command Tools")
        
        # --- Event Reports Submenu ---
        menu.addSeparator()
        reports_act = menu.addAction("📅 Generate Custom Report...")
        
        menu.addSeparator()
        
        copy_all = menu.addAction("📋 Copy terminal")
        clear = menu.addAction("🧹 Clear terminal")
        
        menu.addSeparator()
        
        settings_act = menu.addAction("⚙️ Agent Settings")
        
        action = menu.exec(self.log_view.mapToGlobal(pos))
        
        if action == toggle_ai:
            self.toggle_agent()
        elif action == status_act:
            self.handle_quick_status()
        elif action == cams_act:
            self.handle_quick_cameras()
        elif action == alerts_act:
            self.handle_quick_alerts()
        elif action == tools_act:
            self.handle_quick_tools()
        elif action == reports_act:
            self._show_custom_report_dialog()
        elif action == copy_all:
            QApplication.clipboard().setText("\n".join([e.get("text", "") for e in self.log]))
        elif action == clear:
            self.clear_log()
        elif action == settings_act:
            self.show_agent_settings_dialog()

    # Anchor handling (images / links) -----------------------------------
    def _handle_anchor_click(self, url: QUrl):
        if url.scheme() == "image":
            key = url.toString().replace("image://", "")
            b64 = self.image_cache.get(key)
            if b64:
                self._open_image(b64)
            return
        # Always open local files (e.g., "Open full image") externally — never inside the terminal view.
        if url.scheme() == "file":
            QDesktopServices.openUrl(url)
            return
        if url.scheme() == "event":
            # event://<event_id>
            try:
                event_id = (url.host() or "").strip()
                if event_id:
                    self._post_system(f"Loading detections for event {event_id}…")
                    threading.Thread(target=self._load_event_detections, args=(event_id,), daemon=True).start()
                    return
            except Exception:
                return
        if url.scheme() == "override":
            # override://<event_id>?detection_id=<id>
            try:
                event_id = (url.host() or "").strip()
                q = url.query() or ""
                detection_id = ""
                for part in q.split("&"):
                    if part.startswith("detection_id="):
                        detection_id = part.split("=", 1)[1].strip()
                if event_id and detection_id:
                    self._open_override_dialog(event_id, detection_id)
                    return
            except Exception:
                return
        QDesktopServices.openUrl(url)

    # Provider/model override helpers ------------------------------------
    def _load_provider_models(self):
        try:
            res = requests.get(f"{self.API_BASE}/llm/providers", timeout=6)
            if not res.ok:
                return
            providers = res.json().get("providers", [])
            if not isinstance(providers, list) or not providers:
                return
            # Save a local map: provider_id -> models
            self._provider_models: Dict[str, List[str]] = {}
            provider_ids: List[str] = []
            for p in providers:
                if not isinstance(p, dict):
                    continue
                pid = p.get("id")
                if not isinstance(pid, str) or not pid.strip():
                    continue
                pid = pid.strip().lower()
                provider_ids.append(pid)
                models = p.get("models") if isinstance(p.get("models"), list) else []
                self._provider_models[pid] = [str(m) for m in models if isinstance(m, (str, int, float)) and str(m).strip()]

            # Update UI on main thread
            def _apply():
                try:
                    self.provider_combo.clear()
                    self.provider_combo.addItems(provider_ids)
                    # Default to OpenAI if present; otherwise use stored primary; otherwise first.
                    default_pid = "openai" if "openai" in provider_ids else (self.agent_settings.get("primary") or provider_ids[0])
                    if default_pid in provider_ids:
                        self.provider_combo.setCurrentText(default_pid)
                    self._on_provider_changed(self.provider_combo.currentText())
                    self.provider_combo.setVisible(bool(self.agent_active))
                except Exception:
                    pass

            QTimer.singleShot(0, _apply)
        except Exception:
            return

    def _on_provider_changed(self, provider_id: str):
        pid = (provider_id or "").strip().lower()
        models = []
        try:
            models = (getattr(self, "_provider_models", {}) or {}).get(pid, []) or []
        except Exception:
            models = []
        try:
            self.model_combo.clear()
            if models:
                self.model_combo.addItems(models)
                # Keep prior selection if possible
                self.model_combo.setVisible(bool(self.agent_active))
            else:
                self.model_combo.hide()
        except Exception:
            pass

    def _build_llm_context_override(self) -> Dict[str, Any]:
        """
        Build a per-request LLM override payload that the backend AI agent honors.
        """
        priority = list(self.agent_settings.get("provider_priority") or [])
        provider = (self.provider_combo.currentText() or "").strip().lower() if self.agent_active and self.provider_combo.isVisible() else ""
        model = (self.model_combo.currentText() or "").strip() if self.agent_active and self.model_combo.isVisible() else ""
        if provider:
            priority = [provider] + [p for p in priority if str(p).lower() != provider]
        out: Dict[str, Any] = {"provider_priority": priority}
        if provider:
            out["provider"] = provider
        if model:
            out["model"] = model
        return out

    # Events search tool --------------------------------------------------
    def _run_events_search(self, params: Dict[str, Any]):
        try:
            if not self._ensure_backend_running():
                self._post_error("Backend API is not running (http://localhost:5000). Start it with: python app.py", tool="events_search")
                return
            query = (params or {}).get("query") or (params or {}).get("text") or ""
            camera_ref = (params or {}).get("cameraRef") or (params or {}).get("camera") or (params or {}).get("camera_name")
            start = (params or {}).get("start") or (params or {}).get("start_ts") or (params or {}).get("startTs")
            end = (params or {}).get("end") or (params or {}).get("end_ts") or (params or {}).get("endTs")
            limit = (params or {}).get("limit") or 25

            payload: Dict[str, Any] = {"query": str(query or ""), "limit": int(limit)}
            if camera_ref:
                payload["cameraRef"] = str(camera_ref)
            if start is not None:
                payload["start"] = start
            if end is not None:
                payload["end"] = end

            res = requests.post(f"{self.API_BASE}/events/search", json=payload, timeout=25)
            if not res.ok:
                self._post_error(f"Events search failed ({res.status_code})", tool="events_search")
                return
            j = res.json()
            if not isinstance(j, dict) or not j.get("success"):
                self._post_error(j.get("message") or "Events search failed", tool="events_search")
                return
            data = j.get("data") or {}
            msg = data.get("message") or "Events search complete."
            self._post_success(msg, tool="events_search")
            timeline = data.get("timeline") if isinstance(data.get("timeline"), list) else []
            if not timeline:
                return
            self._render_timeline(timeline)
        except Exception as e:
            self._post_error(f"Events search error: {e}", tool="events_search")

    def _run_events_count(self, params: Dict[str, Any]):
        """
        Call backend /api/events/count and print aggregate totals.
        """
        try:
            if not self._ensure_backend_running():
                self._post_error("Backend API is not running (http://localhost:5000). Start it with: python app.py", tool="events_count")
                return
            camera_ref = (params or {}).get("cameraRef") or (params or {}).get("camera") or (params or {}).get("camera_name")
            start = (params or {}).get("start") or (params or {}).get("start_ts") or (params or {}).get("startTs")
            end = (params or {}).get("end") or (params or {}).get("end_ts") or (params or {}).get("endTs")
            classes = (params or {}).get("detection_classes") or (params or {}).get("classes") or (params or {}).get("objectClasses") or []
            color = (params or {}).get("detection_color") or (params or {}).get("color")

            if isinstance(classes, str):
                classes = [classes]
            if not isinstance(classes, list):
                classes = []

            payload: Dict[str, Any] = {"filters": {"detection": {}}}
            if camera_ref:
                payload["cameraRef"] = str(camera_ref)
            if start is not None:
                payload["start"] = start
            if end is not None:
                payload["end"] = end
            if classes:
                payload["filters"]["detection"]["classes"] = [str(c).strip() for c in classes if str(c).strip()]
            if isinstance(color, str) and color.strip():
                payload["filters"]["detection"]["color"] = color.strip().lower()

            res = requests.post(f"{self.API_BASE}/events/count", json=payload, timeout=20)
            if not res.ok:
                self._post_error(f"Events count failed ({res.status_code})", tool="events_count")
                return
            j = res.json() or {}
            if not j.get("success"):
                self._post_error(j.get("message") or "Events count failed", tool="events_count")
                return
            data = j.get("data") or {}

            ev_total = int(data.get("events_total_in_range") or 0)
            ev_det_indexed = int(data.get("events_detection_indexed_in_range") or 0)
            ev_det_missing = int(data.get("events_detection_not_indexed_in_range") or 0)
            ev_count = int(data.get("event_count") or 0)
            det_count = int(data.get("detection_count") or 0)

            filt = data.get("filters") or {}
            cls_list = filt.get("detection_classes") or []
            det_color = filt.get("detection_color")

            target = ", ".join([c for c in cls_list if c]) if isinstance(cls_list, list) and cls_list else "detections"
            if det_color:
                target = f"{det_color} {target}".strip()

            self._post_success(f"Total in window: {det_count} {target} across {ev_count} event(s).", tool="events_count")
            if ev_total:
                self._post_info(f"Coverage: detections indexed for {ev_det_indexed}/{ev_total} events in this window (missing {ev_det_missing}).")
        except Exception as e:
            self._post_error(f"Events count error: {e}", tool="events_count")

    def _render_timeline(self, timeline: List[Dict[str, Any]]):
        # Keep rendering bounded (avoid spamming the terminal)
        for item in timeline[:25]:
            try:
                if not isinstance(item, dict):
                    continue
                ts = item.get("captured_at") or item.get("captured_ts") or ""
                cam = item.get("camera_name") or ""
                caption = item.get("caption") or item.get("reason") or ""
                dom = item.get("dominant_color")
                parts = []
                if ts:
                    parts.append(str(ts))
                if cam:
                    parts.append(str(cam))
                title = " • ".join(parts) if parts else "Event"
                if dom:
                    title += f" [{dom}]"

                file_path = item.get("file_path") or ""
                thumb_b64 = item.get("thumb_base64")
                det_count = item.get("detections_count") or 0

                file_url = None
                folder_url = None
                try:
                    if file_path:
                        p = Path(str(file_path))
                        file_url = QUrl.fromLocalFile(str(p)).toString()
                        folder_url = QUrl.fromLocalFile(str(p.parent)).toString()
                except Exception:
                    file_url = None
                    folder_url = None

                self._post_line(title + (f"\n{caption}" if caption else ""), kind="output", image_b64=thumb_b64, link=file_url, tool="events_search")
                try:
                    eid = item.get("event_id") or item.get("id")
                    if eid and int(det_count or 0) > 0:
                        # Clickable expand: event://<event_id>
                        self._post_line(
                            f"Objects: {int(det_count)}",
                            kind="output",
                            link=f"event://{eid}",
                            link_label=f"Show objects ({int(det_count)})",
                            tool="events_search",
                        )
                except Exception:
                    pass
                if folder_url:
                    self._post_line("Open folder", kind="output", link=folder_url, tool="events_search")
            except Exception:
                continue

    def _events_backfill(self, max_items: int):
        try:
            if not self._ensure_backend_running():
                self._post_error("Backend API is not running (http://localhost:5000). Start it with: python app.py", tool="events_backfill")
                return
            res = requests.post(
                f"{self.API_BASE}/events/backfill",
                # Default to local detections (YOLO) for fast indexing; captions are optional and slower.
                json={"max_items": int(max_items), "include_vision": False, "include_detections": True},
                timeout=90,
            )
            if not res.ok:
                self._post_error(f"Events backfill failed ({res.status_code})", tool="events_backfill")
                return
            j = res.json()
            if not isinstance(j, dict) or not j.get("success"):
                self._post_error(j.get("message") or "Events backfill failed", tool="events_backfill")
                return
            data = j.get("data") or {}
            self._post_success(
                f"Backfill complete: processed={data.get('processed')}, skipped={data.get('skipped')}, errors={data.get('error_count')}",
                tool="events_backfill",
            )
        except Exception as e:
            self._post_error(f"Events backfill error: {e}", tool="events_backfill")

    def _events_reindex(self, max_files: Optional[int], force: bool, cloud_enrich: bool, cloud_max_calls: int):
        """
        Start bulk reindex and poll status until completion.
        """
        try:
            if not self._ensure_backend_running():
                self._post_error("Backend API is not running (http://localhost:5000). Start it with: python app.py", tool="events_reindex")
                return

            payload: Dict[str, Any] = {
                "force": bool(force),
                "include_detections": True,
                "include_vision": False,
            }
            if max_files is not None:
                payload["max_files"] = int(max_files)

            if cloud_enrich:
                payload.update(
                    {
                        "cloud_enrich": True,
                        "cloud_max_calls": int(max(1, cloud_max_calls)),
                        "cloud_ack": "I_UNDERSTAND_BULK_CLOUD_ENRICH_CAN_RATE_LIMIT",
                    }
                )

            start = requests.post(f"{self.API_BASE}/events/reindex", json=payload, timeout=20)
            if not start.ok:
                try:
                    j = start.json()
                    self._post_error(j.get("message") or f"Reindex failed ({start.status_code})", tool="events_reindex")
                except Exception:
                    # Special case: backend still running older routes (common after code updates)
                    if start.status_code == 405:
                        self._post_error(
                            "Reindex failed (405). Your backend needs a restart to pick up the new /api/events/reindex route. "
                            "Stop the running `python app.py` process and start it again, then retry `events reindex`.",
                            tool="events_reindex",
                        )
                    else:
                        self._post_error(f"Reindex failed ({start.status_code})", tool="events_reindex")
                return

            self._post_success("Reindex started. Polling progress…", tool="events_reindex")

            # Poll until finished (bounded)
            deadline = time.time() + 3600 * 6  # 6 hours safety cap
            last_line = 0.0
            while time.time() < deadline:
                time.sleep(1.5)
                res = requests.get(f"{self.API_BASE}/events/reindex", timeout=10)
                if not res.ok:
                    continue
                state = (res.json() or {}).get("data") or {}
                running = bool(state.get("running"))
                scanned = int(state.get("scanned") or 0)
                processed = int(state.get("processed") or 0)
                skipped = int(state.get("skipped") or 0)
                errs = int(state.get("error_count") or 0)
                eta_s = state.get("eta_seconds")
                total_target = state.get("total_target")
                cloud = state.get("cloud") or {}
                cloud_calls = int((cloud.get("calls") if isinstance(cloud, dict) else 0) or 0)
                cloud_max = int((cloud.get("max_calls") if isinstance(cloud, dict) else 0) or 0)

                # Don't spam the log: only print every ~6s unless finished.
                now = time.time()
                if now - last_line > 6.0 or not running:
                    line = f"Reindex: scanned={scanned} processed={processed} skipped={skipped} errors={errs}"
                    try:
                        if isinstance(total_target, int) and total_target:
                            line += f" target={int(total_target)}"
                        if isinstance(eta_s, int) and eta_s >= 0:
                            mins = int(eta_s) // 60
                            secs = int(eta_s) % 60
                            line += f" eta={mins}:{secs:02d}"
                    except Exception:
                        pass
                    if cloud_enrich:
                        line += f" cloud_calls={cloud_calls}/{cloud_max}"
                    self._post_info(line)
                    last_line = now

                if not running:
                    if errs:
                        self._post_warning(f"Reindex finished with errors={errs}. See /api/events/reindex for details.")
                    else:
                        self._post_success("Reindex finished successfully.", tool="events_reindex")
                    return

            self._post_warning("Reindex still running (poll timeout). You can check status via /api/events/reindex.")
        except Exception as e:
            self._post_error(f"Events reindex error: {e}", tool="events_reindex")

    def _open_live_report(self):
        """Open the live security report dashboard in the system browser."""
        try:
            url = f"{self.API_BASE.replace('/api', '')}/api/events/live-report"
            self._add_system(f"Opening live security report: {url}")
            from PySide6.QtCore import QUrl
            from PySide6.QtGui import QDesktopServices
            QDesktopServices.openUrl(QUrl(url))
        except Exception as e:
            self._add_error(f"Failed to open live report: {e}", tool="live_report")

    def _events_report(self, params: Dict[str, Any]):
        """
        Generate an HTML security report (timeline + crops + direct file links).
        """
        try:
            if not self._ensure_backend_running():
                self._post_error("Backend API is not running (http://localhost:5000). Start it with: python app.py", tool="events_report")
                return

            query = (params or {}).get("query") or ""
            limit = int((params or {}).get("limit") or 200)
            camera_ref = (params or {}).get("cameraRef") or (params or {}).get("camera") or (params or {}).get("camera_name")
            start = (params or {}).get("start") or (params or {}).get("start_ts") or (params or {}).get("startTs")
            end = (params or {}).get("end") or (params or {}).get("end_ts") or (params or {}).get("endTs")

            # Best-effort: infer a camera name from the query text (so "produce stand yesterday" narrows report).
            try:
                if not camera_ref and isinstance(query, str) and query.strip():
                    cams = self._fetch_camera_devices() or []
                    ql = query.lower()
                    # Longest name match wins (avoid matching "gate" inside "main gate" incorrectly).
                    matches = []
                    for c in cams:
                        name = str((c or {}).get("name") or "").strip()
                        if not name:
                            continue
                        nl = name.lower()
                        if nl and nl in ql:
                            matches.append(name)
                    if matches:
                        matches.sort(key=lambda s: len(s), reverse=True)
                        camera_ref = matches[0]
                        # Remove camera name from query to keep filtering cleaner.
                        try:
                            query = re.sub(re.escape(str(camera_ref)), "", query, flags=re.IGNORECASE).strip()
                        except Exception:
                            pass
            except Exception:
                pass

            payload: Dict[str, Any] = {"title": "Security Report", "query": str(query), "limit": max(10, min(limit, 5000))}
            if camera_ref:
                payload["cameraRef"] = str(camera_ref)
            if start is not None:
                payload["start"] = start
            if end is not None:
                payload["end"] = end

            res = requests.post(f"{self.API_BASE}/events/report", json=payload, timeout=60)
            if not res.ok:
                self._post_error(f"Report failed ({res.status_code})", tool="events_report")
                return
            j = res.json() or {}
            if not j.get("success"):
                self._post_error(j.get("message") or "Report failed", tool="events_report")
                return
            data = j.get("data") or {}
            url = data.get("report_url")
            path = data.get("report_path")
            n = data.get("events")
            
            # Auto-open without logging to terminal as requested
            if url:
                QDesktopServices.openUrl(QUrl(str(url)))
            elif path:
                try:
                    p = Path(str(path))
                    QDesktopServices.openUrl(QUrl.fromLocalFile(str(p)))
                except Exception:
                    pass
        except Exception as e:
            self._post_error(f"Events report error: {e}", tool="events_report")

    def _load_event_detections(self, event_id: str):
        try:
            if not self._ensure_backend_running():
                self._post_error("Backend API is not running (http://localhost:5000). Start it with: python app.py", tool="events_detections")
                return
            res = requests.get(
                f"{self.API_BASE}/events/detections",
                params={"event_id": str(event_id), "include_images": "1", "limit": "12"},
                timeout=20,
            )
            if not res.ok:
                self._post_error(f"Failed to load detections ({res.status_code})", tool="events_detections")
                return
            j = res.json() or {}
            if not j.get("success"):
                self._post_error(j.get("message") or "Failed to load detections", tool="events_detections")
                return
            dets = (j.get("data") or {}).get("detections") or []
            if not isinstance(dets, list) or not dets:
                self._post_info("No detections for this event.")
                return

            self._post_success(f"Objects for event {event_id}:", tool="events_detections")
            for d in dets[:12]:
                if not isinstance(d, dict):
                    continue
                det_id = str(d.get("detection_id") or "")
                if det_id:
                    self._detection_cache[det_id] = d
                cls = d.get("class") or d.get("class_raw") or "object"
                col = d.get("color") or d.get("color_raw") or ""
                conf = d.get("confidence")
                idx = d.get("det_idx")
                title = f"#{idx} {cls}" + (f" [{col}]" if col else "")
                try:
                    if conf is not None:
                        title += f" (conf {float(conf):.2f})"
                except Exception:
                    pass
                crop_b64 = d.get("crop_base64")
                self._post_line(title, kind="output", image_b64=crop_b64, tool="events_detections")
                if det_id:
                    self._post_line(
                        "Correct this object",
                        kind="output",
                        link=f"override://{event_id}?detection_id={det_id}",
                        link_label="Retag / correct",
                        tool="events_detections",
                    )
        except Exception as e:
            self._post_error(f"Detections load error: {e}", tool="events_detections")

    def _open_override_dialog(self, event_id: str, detection_id: str):
        det = self._detection_cache.get(str(detection_id)) or {}
        current_class = (det.get("class") or det.get("class_raw") or "").strip()
        current_color = (det.get("color") or det.get("color_raw") or "").strip()
        current_tags = []
        try:
            ov = det.get("override") or {}
            current_tags = ov.get("override_tags") or []
        except Exception:
            current_tags = []

        dlg = QDialog(self)
        dlg.setWindowTitle("Correct detection")
        form = QFormLayout(dlg)

        class_combo = QComboBox()
        class_options = ["car", "truck", "bus", "motorcycle", "person", "bicycle", "dog", "cat", "other"]
        class_combo.addItems(class_options)
        if current_class in class_options:
            class_combo.setCurrentText(current_class)

        color_combo = QComboBox()
        color_options = ["", "white", "black", "gray", "red", "blue", "green", "yellow", "brown", "unknown"]
        color_combo.addItems(color_options)
        if current_color in color_options:
            color_combo.setCurrentText(current_color)

        tags_input = QLineEdit(", ".join([str(t) for t in (current_tags or []) if str(t).strip()]))
        note_input = QLineEdit("")

        form.addRow("Class", class_combo)
        form.addRow("Color", color_combo)
        form.addRow("Tags (comma-separated)", tags_input)
        form.addRow("Note", note_input)

        save_btn = QPushButton("Save")
        cancel_btn = QPushButton("Cancel")
        btn_row = QHBoxLayout()
        btn_row.addWidget(save_btn)
        btn_row.addWidget(cancel_btn)
        form.addRow(btn_row)

        def _save():
            override_tags = [t.strip() for t in (tags_input.text() or "").split(",") if t.strip()]
            payload = {
                "event_id": str(event_id),
                "detection_id": str(detection_id),
                "override_class": class_combo.currentText().strip(),
                "override_color": color_combo.currentText().strip(),
                "override_tags": override_tags,
                "note": note_input.text().strip(),
                "updated_by": "pyqt-terminal",
            }

            def worker():
                try:
                    res = requests.post(f"{self.API_BASE}/events/override", json=payload, timeout=15)
                    if not res.ok:
                        self._post_error(f"Override save failed ({res.status_code})", tool="events_override")
                        return
                    j = res.json() or {}
                    if not j.get("success"):
                        self._post_error(j.get("message") or "Override save failed", tool="events_override")
                        return
                    self._post_success("Override saved. Reloading objects…", tool="events_override")
                    self._load_event_detections(str(event_id))
                except Exception as e:
                    self._post_error(f"Override save error: {e}", tool="events_override")

            threading.Thread(target=worker, daemon=True).start()
            dlg.accept()

        save_btn.clicked.connect(_save)
        cancel_btn.clicked.connect(dlg.reject)
        dlg.exec()

    def _show_custom_report_dialog(self):
        """
        Show a dialog to generate a custom security report with specific query and time window.
        """
        dlg = QDialog(self)
        dlg.setWindowTitle("Generate Custom Security Report")
        dlg.setMinimumWidth(450)
        form = QFormLayout(dlg)
        
        # Style the dialog
        dlg.setStyleSheet(KnoxnetStyle.context_menu())

        query_input = QLineEdit()
        query_input.setPlaceholderText("Search captures (e.g. 'red trucks')")
        form.addRow("Search Query:", query_input)

        all_time_check = QCheckBox("All Time (Search entire database)")
        form.addRow("", all_time_check)

        start_dt = QDateTimeEdit()
        start_dt.setCalendarPopup(True)
        # Default to 24h ago
        start_dt.setDateTime(datetime.now() - timedelta(days=1))
        form.addRow("Start Window:", start_dt)

        end_dt = QDateTimeEdit()
        end_dt.setCalendarPopup(True)
        end_dt.setDateTime(datetime.now())
        form.addRow("End Window:", end_dt)

        def _update_dt_state():
            is_all = all_time_check.isChecked()
            start_dt.setEnabled(not is_all)
            end_dt.setEnabled(not is_all)
        
        all_time_check.stateChanged.connect(_update_dt_state)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        form.addRow(buttons)

        buttons.accepted.connect(dlg.accept)
        buttons.rejected.connect(dlg.reject)

        if dlg.exec() == QDialog.DialogCode.Accepted:
            params = {
                "query": query_input.text().strip(),
                "limit": 5000,
            }
            
            if not all_time_check.isChecked():
                params["start"] = int(start_dt.dateTime().toSecsSinceEpoch())
                params["end"] = int(end_dt.dateTime().toSecsSinceEpoch())
            
            # Run report in background
            threading.Thread(target=self._events_report, args=(params,), daemon=True).start()

    def _open_image(self, b64: str):
        try:
            raw = base64.b64decode(b64)
        except Exception:
            return
        from PySide6.QtGui import QPixmap
        from PySide6.QtWidgets import QLabel, QDialog, QVBoxLayout

        dlg = QDialog(self)
        dlg.setWindowTitle("Image")
        dlg.setWindowFlag(Qt.WindowType.WindowStaysOnTopHint, True)
        v = QVBoxLayout(dlg)
        lbl = QLabel()
        pix = QPixmap()
        pix.loadFromData(raw)
        lbl.setPixmap(pix.scaled(900, 700, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))
        v.addWidget(lbl)
        dlg.resize(920, 720)
        dlg.exec()

    def closeEvent(self, event):
        try:
            TerminalWidget._instances = [w for w in TerminalWidget._instances if w is not self]
        except Exception:
            pass
        super().closeEvent(event)

    # Motion watch badge handling -----------------------------------------
    def _on_motion_watch_update(self, payload: Dict[str, Any]):
        cam_id = payload.get("camera_id") or payload.get("model")
        if not cam_id:
            return
        if payload.get("stopped"):
            self.motion_watch_status.pop(cam_id, None)
            self._refresh_watch_badge()
            return
        remaining = payload.get("remaining_seconds")
        self.motion_watch_status[cam_id] = {
            "remaining_seconds": remaining,
            "text": payload.get("text", ""),
            "label": payload.get("camera_label") or cam_id,
        }
        self._refresh_watch_badge()

    def _format_remaining(self, remaining: Optional[int]) -> str:
        if remaining is None:
            return "∞"
        if remaining < 0:
            return "∞"
        hrs = remaining // 3600
        mins = (remaining % 3600) // 60
        secs = remaining % 60
        if hrs > 0:
            return f"{hrs}:{mins:02d}:{secs:02d}"
        if mins > 0:
            return f"{mins}:{secs:02d}"
        return f"{secs}s"

    def _refresh_watch_badge(self):
        if not self.motion_watch_status:
            self.watch_badge.hide()
            return
        parts = []
        for cam_id, info in list(self.motion_watch_status.items()):
            rem = self._format_remaining(info.get("remaining_seconds"))
            label = info.get("label") or cam_id
            parts.append(f"{label}: {rem}")
        self.watch_badge.setText(" | ".join(parts))
        self.watch_badge.show()

    def _handle_stop_watch(self, camera_ref: str):
        if not camera_ref:
            self._add_warning("Usage: stop watch <camera_id>")
            return
        self._add_system(f"Stopping motion watch for {camera_ref}…")
        self._send_ipc({"cmd": "stop_motion_watch", "camera_id": camera_ref})

    def _handle_stop_watch_all(self):
        self._add_system("Stopping motion watch on all cameras…")
        self._send_ipc({"cmd": "stop_motion_watch_all"})

    def _handle_watch_all(self, text: str):
        duration = None
        parts = text.strip().split()
        for p in parts[1:]:
            if p.lower() == "all":
                continue
            try:
                duration = int(p)
            except ValueError:
                pass
        self._add_system("Starting motion watch on all cameras…")
        payload: Dict[str, Any] = {"cmd": "start_motion_watch_all"}
        if duration is not None:
            payload["duration_sec"] = duration
        self._send_ipc(payload)

    def _handle_watch_camera(self, ref: str, text: str):
        if not ref:
            self._add_warning("Usage: watch <camera_id>")
            return
        duration = None
        parts = text.strip().split()
        for p in parts[2:]:
            try:
                duration = int(p)
                break
            except ValueError:
                pass
        self._add_system(f"Starting motion watch for {ref}…")
        payload: Dict[str, Any] = {"cmd": "start_motion_watch", "camera_ref": ref}
        if duration is not None:
            payload["duration_sec"] = duration
        self._send_ipc(payload)

    # ------------------------------------------------------------------
    # Recording commands
    # ------------------------------------------------------------------

    def _fetch_cameras_json(self) -> list:
        """Fetch the camera list from the backend."""
        try:
            import requests as _req
            r = _req.get("http://localhost:5000/api/cameras", timeout=3)
            return r.json().get("data", [])
        except Exception:
            return []

    def _resolve_camera_ref_to_entry(self, ref: str, cameras: list) -> Optional[dict]:
        """Resolve a camera reference (name fragment, IP, or UUID) to a camera entry."""
        ref_l = ref.lower()
        for c in cameras:
            cid = c.get("id", "")
            name = c.get("name", "")
            ip = c.get("ip_address", "") or c.get("ip", "")
            if ref_l == cid.lower() or ref_l == name.lower() or ref_l == ip.lower():
                return c
            if cid.lower().startswith(ref_l) or ref_l in name.lower() or ref_l in ip.lower():
                return c
        return None

    def _handle_record_all(self, enable: bool):
        """Toggle recording on all cameras with per-camera feedback."""
        action = "Starting" if enable else "Stopping"
        self._add_system(f"{action} recording on all cameras…")

        def _worker():
            import requests as _req
            cameras = self._fetch_cameras_json()
            if not cameras:
                self._post_line("No cameras found.", kind="warning")
                return
            ok_count = 0
            for cam in cameras:
                cid = cam.get("id", "")
                label = cam.get("name") or cid[:8]
                try:
                    r = _req.post(
                        f"http://localhost:5000/api/cameras/{cid}/recording",
                        json={"record": enable}, timeout=5,
                    )
                    data = r.json()
                    if data.get("success"):
                        ok_count += 1
                    else:
                        self._post_line(f"  ✗ {label}: {data.get('message', 'failed')}", kind="warning")
                except Exception as e:
                    self._post_line(f"  ✗ {label}: {e}", kind="error")
            verb = "started" if enable else "stopped"
            self._post_line(f"Recording {verb} on {ok_count}/{len(cameras)} cameras.", kind="success")
        threading.Thread(target=_worker, daemon=True).start()

    def _handle_record_camera(self, ref: str, enable: bool):
        """Toggle recording on a single camera with feedback."""
        if not ref:
            self._add_warning("Usage: record <cam> | stop record <cam>")
            return
        action = "Starting" if enable else "Stopping"
        self._add_system(f"{action} recording for {ref}…")

        def _worker():
            import requests as _req
            cameras = self._fetch_cameras_json()
            cam = self._resolve_camera_ref_to_entry(ref, cameras)
            if not cam:
                self._post_line(f"Camera '{ref}' not found.", kind="warning")
                return
            cid = cam.get("id", "")
            label = cam.get("name") or cid[:8]
            try:
                r = _req.post(
                    f"http://localhost:5000/api/cameras/{cid}/recording",
                    json={"record": enable}, timeout=5,
                )
                data = r.json()
                if data.get("success"):
                    verb = "started" if enable else "stopped"
                    self._post_line(f"Recording {verb} for {label}.", kind="success")
                else:
                    self._post_line(f"Failed for {label}: {data.get('message', '?')}", kind="warning")
            except Exception as e:
                self._post_line(f"Error toggling recording for {label}: {e}", kind="error")
        threading.Thread(target=_worker, daemon=True).start()

    def _handle_recording_status(self):
        """Show recording state, directories, and disk usage for all cameras."""
        self._add_system("Fetching recording status…")

        def _worker():
            import requests as _req
            cameras = self._fetch_cameras_json()
            if not cameras:
                self._post_line("No cameras found.", kind="warning")
                return
            try:
                r = _req.get("http://localhost:5000/api/cameras/recording-status", timeout=3)
                rec_flags = r.json().get("data", {})
            except Exception:
                rec_flags = {}

            lines = ["Camera                 Recording   Directory"]
            lines.append("─" * 65)
            rec_count = 0
            for cam in cameras:
                cid = cam.get("id", "")
                name = cam.get("name") or cid[:8]
                is_rec = rec_flags.get(cid, cam.get("recording", False))
                rec_dir = cam.get("recording_dir", "").strip() or "(default)"
                status = "● REC" if is_rec else "  off"
                if is_rec:
                    rec_count += 1
                lines.append(f"  {name:20s} {status:11s} {rec_dir}")

            lines.append("")
            lines.append(f"Recording: {rec_count}/{len(cameras)} cameras")

            try:
                import shutil
                from core.paths import get_recordings_dir
                usage = shutil.disk_usage(str(get_recordings_dir()))
                used_gb = (usage.total - usage.free) / (1024 ** 3)
                total_gb = usage.total / (1024 ** 3)
                free_gb = usage.free / (1024 ** 3)
                lines.append(f"Disk: {used_gb:.1f}/{total_gb:.1f} GB used, {free_gb:.1f} GB free")
            except Exception:
                pass

            self._post_line("\n".join(lines), kind="system")
        threading.Thread(target=_worker, daemon=True).start()

    def _handle_recording_paths(self):
        """Show the resolved recording path for each camera."""
        self._add_system("Fetching recording paths…")

        def _worker():
            import requests as _req
            cameras = self._fetch_cameras_json()
            if not cameras:
                self._post_line("No cameras found.", kind="warning")
                return

            lines = ["Camera                 Recording Path"]
            lines.append("─" * 70)
            for cam in cameras:
                cid = cam.get("id", "")
                name = cam.get("name") or cid[:8]
                try:
                    r = _req.get(f"http://127.0.0.1:9997/v3/config/paths/get/{cid}", timeout=2)
                    if r.status_code == 200:
                        rp = r.json().get("recordPath", "(not configured)")
                    else:
                        rp = "(no MediaMTX path)"
                except Exception:
                    rp = "(unreachable)"
                lines.append(f"  {name:20s} {rp}")
            self._post_line("\n".join(lines), kind="system")
        threading.Thread(target=_worker, daemon=True).start()

    def _handle_recording_dir(self, args: str):
        """Set recording directory for a camera or all cameras.

        Usage:
            recording dir all /mnt/nas/recordings
            recording dir .103 /mnt/ssd/cam103
        """
        parts = args.strip().split(None, 1)
        if len(parts) < 2:
            self._add_warning("Usage: recording dir <cam|all> <path>")
            return
        target, new_dir = parts[0], parts[1]

        def _worker():
            import requests as _req
            cameras = self._fetch_cameras_json()
            if not cameras:
                self._post_line("No cameras found.", kind="warning")
                return

            if target.lower() == "all":
                ok = 0
                for cam in cameras:
                    cid = cam.get("id", "")
                    try:
                        r = _req.put(
                            f"http://localhost:5000/api/cameras/{cid}",
                            json={"recording_dir": new_dir}, timeout=3,
                        )
                        if r.status_code == 200:
                            ok += 1
                    except Exception:
                        pass
                self._post_line(f"Set recording directory to '{new_dir}' for {ok}/{len(cameras)} cameras.", kind="success")
                self._post_line("Restart recording to apply: stop record all && record all", kind="info")
            else:
                cam = self._resolve_camera_ref_to_entry(target, cameras)
                if not cam:
                    self._post_line(f"Camera '{target}' not found.", kind="warning")
                    return
                cid = cam.get("id", "")
                label = cam.get("name") or cid[:8]
                try:
                    r = _req.put(
                        f"http://localhost:5000/api/cameras/{cid}",
                        json={"recording_dir": new_dir}, timeout=3,
                    )
                    if r.status_code == 200:
                        self._post_line(f"Set recording directory for {label} to '{new_dir}'.", kind="success")
                        if cam.get("recording"):
                            self._post_line("Restart recording to apply: stop record {0} && record {0}".format(label), kind="info")
                    else:
                        self._post_line(f"Failed to update {label}: HTTP {r.status_code}", kind="warning")
                except Exception as e:
                    self._post_line(f"Error updating {label}: {e}", kind="error")
        threading.Thread(target=_worker, daemon=True).start()

    def _handle_recording_list(self, ref: str = ""):
        """List recording files for a camera or all cameras."""
        self._add_system("Listing recordings…")

        def _worker():
            import requests as _req
            cameras = self._fetch_cameras_json()
            if not cameras:
                self._post_line("No cameras found.", kind="warning")
                return

            targets = cameras
            if ref:
                cam = self._resolve_camera_ref_to_entry(ref, cameras)
                if not cam:
                    self._post_line(f"Camera '{ref}' not found.", kind="warning")
                    return
                targets = [cam]

            total_files = 0
            total_mb = 0.0
            for cam in targets:
                cid = cam.get("id", "")
                label = cam.get("name") or cid[:8]
                try:
                    r = _req.get(
                        f"http://localhost:5000/api/recordings/list?camera_id={cid}",
                        timeout=5,
                    )
                    segs = r.json().get("data", [])
                except Exception:
                    segs = []

                if not segs:
                    self._post_line(f"  {label}: no recordings", kind="system")
                    continue

                cam_mb = sum(s.get("size_mb", 0) for s in segs)
                total_files += len(segs)
                total_mb += cam_mb
                self._post_line(f"  {label}: {len(segs)} files, {cam_mb:.1f} MB", kind="system")
                for seg in segs[-5:]:
                    self._post_line(f"    {seg.get('name', '?'):40s}  {seg.get('size_mb', 0):.1f} MB", kind="output")
                if len(segs) > 5:
                    self._post_line(f"    … and {len(segs) - 5} more", kind="output")

            self._post_line(f"\nTotal: {total_files} files, {total_mb:.1f} MB", kind="success")
        threading.Thread(target=_worker, daemon=True).start()

    def _handle_motion_box_all(self, on: bool = True):
        state = "on" if on else "off"
        self._add_system(f"Turning motion boxes {state} for all cameras…")
        self._send_ipc({"cmd": "set_motion_boxes_all", "enabled": on})

    def _handle_motion_box_camera(self, ref: str, on: bool = True):
        if not ref:
            self._add_warning("Usage: motion box <camera_id>")
            return
        state = "on" if on else "off"
        self._add_system(f"Turning motion boxes {state} for {ref}…")
        self._send_ipc({"cmd": "set_motion_boxes", "camera_ref": ref, "enabled": on})

    def _handle_detect_all(self, on: bool = True):
        state = "on" if on else "off"
        self._add_system(f"Turning object detection {state} for all cameras…")
        self._send_ipc({"cmd": "set_object_detection_all", "enabled": on})

    def _handle_detect_camera(self, ref: str, on: bool = True):
        if not ref:
            self._add_warning("Usage: detect <camera_id>")
            return
        state = "on" if on else "off"
        self._add_system(f"Turning object detection {state} for {ref}…")
        self._send_ipc({"cmd": "set_object_detection", "camera_ref": ref, "enabled": on})

    def _handle_sensitivity(self, val_str: str):
        try:
            val = int(val_str)
        except (ValueError, TypeError):
            self._add_warning("Usage: sensitivity <1-100>   (e.g. sensitivity 60)")
            return
        val = max(1, min(100, val))
        self._add_system(f"Setting motion sensitivity to {val} for all cameras…")
        self._send_ipc({"cmd": "set_sensitivity_all", "value": val})

    def _handle_layout_list(self):
        self._add_system("Fetching layouts…")
        self._send_ipc({"cmd": "layout_list"})

    def _handle_layout_load(self, ref: str):
        if not ref:
            self._add_warning("Usage: layout <name>")
            return
        self._add_system(f"Loading layout '{ref}'…")
        self._send_ipc({"cmd": "layout_load", "layout_ref": ref})

    def _handle_layout_run(self, ref: str):
        if not ref:
            self._add_warning("Usage: layout run <name>")
            return
        self._add_system(f"Running layout '{ref}' in background…")
        self._send_ipc({"cmd": "layout_run", "layout_ref": ref})

    def _handle_layout_stop(self, ref: str):
        if not ref:
            self._add_warning("Usage: layout stop <name>")
            return
        self._add_system(f"Stopping layout '{ref}'…")
        self._send_ipc({"cmd": "layout_stop", "layout_ref": ref})

    # Tool execution helpers ---------------------------------------------
    def _resolve_camera_ref(self, camera_ref: str) -> Optional[Dict[str, str]]:
        """
        Resolve a camera reference (name or id) into a {id, name} dict using /api/devices.
        """
        ref = (camera_ref or "").strip()
        if not ref:
            return None
        try:
            devices = self._fetch_camera_devices()
            if not isinstance(devices, list):
                return None
            # Exact match by id first
            for d in devices:
                if not isinstance(d, dict):
                    continue
                if str(d.get("type", "")).lower() != "camera":
                    continue
                cid = str(d.get("id") or d.get("camera_id") or "").strip()
                if cid and cid == ref:
                    return {"id": cid, "name": str(d.get("name") or cid)}
            # Case-insensitive match by name
            ref_l = ref.lower()
            for d in devices:
                if not isinstance(d, dict):
                    continue
                if str(d.get("type", "")).lower() != "camera":
                    continue
                name = str(d.get("name") or "").strip()
                cid = str(d.get("id") or d.get("camera_id") or "").strip()
                if name and name.lower() == ref_l:
                    return {"id": cid or name, "name": name}
            return None
        except Exception:
            return None

    def _fetch_camera_devices(self) -> List[Dict[str, Any]]:
        """
        Fetch camera list from backend with robust fallbacks:
        - /api/cameras (blueprint; preferred)
        - /api/devices (legacy)
        - local data/cameras.json then cameras.json
        Returns a list of dicts with at least id/name/type fields when possible.
        """
        # 1) Preferred: /api/cameras (api/routes.py)
        try:
            r = requests.get(f"{self.API_BASE}/cameras", timeout=5)
            if r.ok:
                j = r.json()
                data = j.get("data") if isinstance(j, dict) else None
                # blueprint returns {"data": {"devices":[...]}} or sometimes {"data":[...]}
                if isinstance(data, dict) and isinstance(data.get("devices"), list):
                    cams = [c for c in data.get("devices") if isinstance(c, dict)]
                    if cams:
                        return cams
                if isinstance(data, list):
                    cams = [c for c in data if isinstance(c, dict)]
                    if cams:
                        return cams
        except Exception:
            pass

        # 2) Legacy: /api/devices (app.py)
        try:
            r = requests.get(f"{self.API_BASE}/devices", timeout=5)
            if r.ok:
                j = r.json()
                data = j.get("data") if isinstance(j, dict) else None
                if isinstance(data, list):
                    cams = [c for c in data if isinstance(c, dict)]
                    if cams:
                        return cams
        except Exception:
            pass

        # 3) Local file fallback (desktop always has access)
        try:
            from core.paths import get_data_dir
            data_dir = get_data_dir()
        except Exception:
            data_dir = Path("data")
        for p in [data_dir / "cameras.json", data_dir.parent / "cameras.json"]:
            try:
                if not p.exists():
                    continue
                raw = json.loads(p.read_text(encoding="utf-8", errors="ignore"))
                if isinstance(raw, list):
                    cams = []
                    for c in raw:
                        if not isinstance(c, dict):
                            continue
                        if str(c.get("type") or "").lower() not in {"camera", ""}:
                            continue
                        cams.append({**c, "type": "camera"})
                    if cams:
                        return cams
            except Exception:
                continue

        return []

    def _run_snapshot_detect(self, camera_id: str, camera_name: str, object_classes: List[str], params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Call backend /api/ai/snapshot-detect with resolved camera id + classes.
        Returns backend JSON dict.
        """
        model = (params or {}).get("model") or "auto"
        confidence = (params or {}).get("confidence")
        try:
            confidence = float(confidence) if confidence is not None else 0.25
        except Exception:
            confidence = 0.25
        payload = {
            "camera_id": camera_id,
            "camera_name": camera_name,
            "object_classes": object_classes,
            "model": model,
            "confidence": confidence,
        }
        res = requests.post(f"{self.API_BASE}/ai/snapshot-detect", json=payload, timeout=30)
        if not res.ok:
            return {"success": False, "error": f"HTTP {res.status_code}"}
        try:
            return res.json()
        except Exception:
            return {"success": False, "error": "Invalid JSON from snapshot-detect"}

    def _fetch_snapshot_b64(self, camera_id: str) -> Optional[str]:
        """
        Fetch /api/cameras/<id>/snapshot and return base64 (no data-url prefix).
        """
        try:
            res = requests.get(f"{self.API_BASE}/cameras/{camera_id}/snapshot", timeout=12)
            if not res.ok:
                return None
            # If it's already JSON with base64, accept it; otherwise treat as raw JPEG bytes.
            ctype = (res.headers.get("content-type") or "").lower()
            if "application/json" in ctype:
                j = res.json()
                data = j.get("data") or {}
                # tolerate a few common shapes
                img = data.get("imageBase64") or data.get("image_base64") or data.get("snapshot") or data.get("image")
                if isinstance(img, str) and img:
                    return img.split(",", 1)[1] if img.startswith("data:image/") else img
                return None
            # Raw bytes
            import base64 as _b64
            return _b64.b64encode(res.content).decode("utf-8")
        except Exception:
            return None

    def _run_vision_caption(self, image_b64: str, prompt: str) -> Optional[str]:
        """
        Call backend /api/ai/vision and return a short caption/analysis string.
        Uses local vision by default (BLIP). This is best-effort.
        """
        try:
            payload = {
                "image": f"data:image/jpeg;base64,{image_b64}",
                "prompt": str(prompt or "Describe what you see.").strip(),
                "source": "local",
                "model": "blip",
                "context": {"source": "pyqt-terminal"},
            }
            res = requests.post(f"{self.API_BASE}/ai/vision", json=payload, timeout=35)
            if not res.ok:
                return None
            j = res.json()
            if not isinstance(j, dict) or not j.get("success"):
                return None
            data = j.get("data") or {}
            caption = data.get("message") or data.get("caption") or data.get("analysis")
            if isinstance(caption, str):
                caption = caption.strip()
            return caption if caption else None
        except Exception:
            return None

    # Motion watch integration ----------------------------------------------
    @classmethod
    def broadcast_motion_watch(
        cls,
        camera_id: str,
        text: str,
        *,
        countdown: Optional[int] = None,
        image_b64: Optional[str] = None,
        link: Optional[str] = None,
        link_label: Optional[str] = None,
        remaining_seconds: Optional[int] = None,
        stopped: bool = False,
        camera_label: Optional[str] = None,
        suppress_log: bool = False,
    ):
        """
        Send motion watch updates (countdown ticks, captures) to all terminal widgets.
        Safe to call from any thread.
        """
        label = camera_label or camera_id
        payload = {
            "text": text,
            "kind": "info" if countdown is not None else "success",
            "image_b64": image_b64,
            "link": link,
            "link_label": link_label,
            "tool": "motion_watch",
            "model": camera_id,
            "camera_id": camera_id,
            "remaining_seconds": remaining_seconds,
            "stopped": stopped,
            "camera_label": label,
        }
        # Only send countdown text if provided
        if countdown is not None:
            payload["text"] = f"[Motion Watch] {label}: {countdown}s remaining — {text}"
        for inst in list(cls._instances):
            try:
                if not suppress_log:
                    inst.log_signal.emit(payload)
                inst.watch_signal.emit(payload)
            except Exception:
                continue

    # Convenience adders --------------------------------------------------
    def _add_system(self, text: str):
        self._add_line(text, kind="system")

    def _add_success(self, text: str, tool: Optional[str] = None, model: Optional[str] = None):
        self._add_line(text, kind="success", tool=tool, model=model)

    def _add_warning(self, text: str):
        self._add_line(text, kind="warning")

    def _add_info(self, text: str):
        self._add_line(text, kind="info")

    def _add_output(self, text: str):
        self._add_line(text, kind="output")

    def _add_error(self, text: str, tool: Optional[str] = None):
        self._add_line(text, kind="error", tool=tool)

    # Thread-safe variants (for background workers)
    def _post_system(self, text: str):
        self._post_line(text, kind="system")

    def _post_success(self, text: str, tool: Optional[str] = None, model: Optional[str] = None):
        self._post_line(text, kind="success", tool=tool, model=model)

    def _post_warning(self, text: str):
        self._post_line(text, kind="warning")

    def _post_info(self, text: str):
        self._post_line(text, kind="info")

    def _post_output(self, text: str):
        self._post_line(text, kind="output")

    def _post_error(self, text: str, tool: Optional[str] = None):
        self._post_line(text, kind="error", tool=tool)

    # Utils ---------------------------------------------------------------
    @staticmethod
    def _escape(text: str) -> str:
        return (
            text.replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace('"', "&quot;")
            .replace("'", "&#39;")
        )

