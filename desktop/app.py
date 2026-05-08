import sys
import os

# Fix for QWebEngineView crashes on some Linux configurations
# Disabling sandbox is often required when running inside containers or on certain kernels
os.environ["QTWEBENGINE_DISABLE_SANDBOX"] = "1"
# Force software rendering path to avoid Vulkan/GBM crashes on some GPUs/drivers
# Use software rendering for both WebEngine and QtQuick
# Note: swiftshader requires XCB GL integration to function properly for WebEngine
os.environ["QTWEBENGINE_CHROMIUM_FLAGS"] = "--no-sandbox --disable-gpu"
# --disable-software-rasterizer"
# os.environ["QT_QUICK_BACKEND"] = "software" 
# Remove QT_XCB_GL_INTEGRATION=none as it prevents even software GL contexts from forming
if "QT_XCB_GL_INTEGRATION" in os.environ:
    del os.environ["QT_XCB_GL_INTEGRATION"]


import threading
import logging
import json
import time
import asyncio
import requests
from datetime import datetime, timedelta
from pathlib import Path
# IMPORTANT: Import QWebEngineWidgets BEFORE QApplication is created to share OpenGL contexts properly
from PySide6.QtWebEngineWidgets import QWebEngineView
from PySide6.QtWidgets import (
    QApplication,
    QMainWindow,
    QSystemTrayIcon,
    QMenu,
    QInputDialog,
    QMessageBox,
    QWidget,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QWidgetAction,
)
from PySide6.QtGui import QIcon, QAction, QActionGroup, QColor
from PySide6.QtCore import Qt, QTimer, Slot, Signal
from PySide6.QtGui import QGuiApplication
import math
import numpy as np

# Import local modules
from desktop.ipc_server import IPCServer
from desktop.utils.app_icon import apply_app_icon, load_knoxnet_icon

# Import CameraManager from core
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.camera_manager import CameraManager
from core.layout_models import LayoutDefinition, WidgetDefinition
from core.layout_store import LayoutsAndProfilesStore
from core.session_manager import SessionManager
from core.load_shedder import (
    LoadShedder,
    LoadLevel,
    SystemMetrics,
    ShedEvent,
    ShedEventLog,
)
from run import ServerManager  # Still need this for Mediamtx/Flask process handling if we keep them separate?
import subprocess

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("DesktopApp")

class KnoxnetDesktopApp(QApplication):
    # Signal to bridge camera thread -> GUI thread
    # Arguments: camera_id, frame (numpy array)
    frame_signal = Signal(str, np.ndarray)
    # Thread-safe heartbeat ping: emitted from the load-shedder
    # heartbeat thread, slot runs on the GUI thread.  Using a Qt signal
    # is the only reliable way to schedule a callback from a worker
    # thread; QTimer.singleShot from non-GUI threads is racy.
    _shed_heartbeat_ping = Signal()

    def __init__(self, sys_argv):
        super().__init__(sys_argv)
        self._app_icon = apply_app_icon(self)
        self.setQuitOnLastWindowClosed(False)
        self.active_widgets: set = set()
        # Legacy (v1) desktop layouts file (kept for migration/backwards-compat)
        self.layout_file = Path("data/desktop_layouts.json")
        self.layout_file.parent.mkdir(parents=True, exist_ok=True)
        # v2 layouts + profiles store (local JSON source of truth)
        self.layouts_store = LayoutsAndProfilesStore()
        self.prefs_file = Path("data/desktop_prefs.json")
        self.prefs_file.parent.mkdir(parents=True, exist_ok=True)
        self.cameras_json_path = Path("cameras.json")
        self.cameras_json_mtime = self.cameras_json_path.stat().st_mtime if self.cameras_json_path.exists() else None
        self.camera_loop = None
        self.camera_config_dialog = None
        self._initial_sync_done = False
        self._startup_layout_loaded = False
        self.current_layout_name: str | None = None
        # Track which widgets were created by the most recently "loaded" layout (for backgrounding).
        self._current_layout_widgets: list = []
        # Map session_id -> list of widget instances created for that session (best-effort).
        self._session_widgets: dict[str, list] = {}
        # Map a single-layout "background run" to its session id (for quick tray controls).
        self._layout_sessions: dict[str, str] = {}
        # Track paused layouts (pause=disconnect cameras but keep widgets open).
        self._layout_paused: set[str] = set()
        # Track hidden layouts (hide windows but keep layout/session alive).
        self._layout_hidden: set[str] = set()
        self._system_manager_dialog = None
        self._tray_cam_cache: list[tuple[str, str]] = []
        self._tray_cam_cache_ts: float = 0.0
        self._tray_cam_cache_lock = threading.Lock()
        self._tray_cam_refresh_inflight = False

        # ---- Layout scheduler (time-of-day / uptime) ----
        self._scheduler_started_at_ts: float = time.time()
        self._scheduler_last_applied_ts: dict[str, float] = {}  # schedule_id -> epoch seconds
        self._scheduler_last_layout_id: str | None = None
        self._scheduler_manual_override_until_ts: float = 0.0  # epoch seconds; scheduler won't switch until then
        self._scheduler_timer = QTimer()
        self._scheduler_timer.timeout.connect(self._scheduler_tick)

        # ---- Playback sync groups ----
        # Maps group_id -> set of camera_id strings.  When any member seeks
        # or changes speed, the others follow.
        self._playback_sync_groups: dict[str, set[str]] = {}
        self._playback_sync_counter: int = 0

        # ---- Snap guide overlay ----
        self._snap_overlay = None

        # ---- Auto-protection (load shedder) ----
        # Background watchdog that monitors host CPU/RAM/swap and
        # progressively sheds features (live FPS throttle, worker
        # suspend, widget release, recording stop) under load to
        # prevent the host from locking up.  Initialized fully later
        # once prefs have been loaded.
        self._load_shedder: LoadShedder | None = None
        self._shed_event_log: ShedEventLog = ShedEventLog(capacity=20)
        self._shed_timer: QTimer | None = None
        self._shed_heartbeat_thread: threading.Thread | None = None
        self._shed_heartbeat_stop = threading.Event()
        # Last GUI-thread heartbeat tick (set from a singleShot callback).
        # The heartbeat thread reads it to detect stuck event loops.
        self._shed_heartbeat_last_seen_ts: float = time.time()
        self._shed_heartbeat_pending_since_ts: float = 0.0
        self._shed_event_loop_stuck_sec: float = 0.0
        # Cameras whose recording the shedder paused (so we know which
        # ones to re-enable on recovery, and not to fight the user if
        # they had a different set off intentionally).
        self._shed_recording_paused_cams: set[str] = set()
        self._shed_emergency_since_ts: float = 0.0
        self._shed_exit_initiated: bool = False
        # Recently-focused camera widgets (most recent first) for
        # selecting which cameras stay "primary" under CRITICAL load.
        self._recent_camera_focus: list = []  # list of CameraWidget refs

        # ---- Camera patrol mode ----
        self._patrol_active: bool = False
        self._patrol_timer = QTimer()
        self._patrol_timer.timeout.connect(self._patrol_tick)
        self._patrol_index: int = 0
        self._patrol_widgets: list = []

        # Timed auto-show (runtime only; does NOT persist)
        # widget_key -> {"timer": QTimer, "expire_at": float, "baseline_hidden": bool, "baseline_minimized": bool, "gen": int}
        self._timed_visibility: dict[str, dict] = {}
        # widget_key -> last focus timestamp (monotonic seconds)
        self._timed_visibility_last_focus: dict[str, float] = {}

        # Load persisted visibility flags
        try:
            prefs = self._load_prefs()
            vis = prefs.get("layout_visibility") if isinstance(prefs, dict) else {}
            if isinstance(vis, dict):
                for lid, cfg in vis.items():
                    try:
                        if isinstance(cfg, dict) and bool(cfg.get("hidden")):
                            self._layout_hidden.add(str(lid))
                    except Exception:
                        continue
        except Exception:
            pass

        # Use a hidden main window to keep the app alive and prevent taskbar clutter if desired, 
        # but for now, we just use the tray icon.
        # To hide the app from the taskbar (except when widgets are open), we don't show a main window.
        
        # 1. Start the IPC Server
        self.ipc_server = IPCServer()
        self.ipc_server.command_received.connect(self.handle_ipc_command)
        self.ipc_server.start()

        # 2. Initialize Shared CameraManager (desktop-light defaults: no auto-connect, no aiortc receiver)
        self.camera_manager = CameraManager(
            auto_connect_on_start=False,
            enable_webrtc_receiver=False,
            cleanup_mediamtx_paths_on_disconnect=False,
            idle_disconnect_seconds=15.0,
        )
        self.session_manager = SessionManager(store=self.layouts_store, camera_manager=self.camera_manager)

        # Best-effort: migrate legacy layouts into v2 store ONCE.
        # Re-running auto-migration can resurrect layouts the user deleted from v2 on restart.
        try:
            self._maybe_migrate_legacy_layouts(force=False)
        except Exception:
            pass
        
        # Start CameraManager async loop in a thread
        self.cm_thread = threading.Thread(target=self.run_camera_manager, daemon=True)
        self.cm_thread.start()

        # Watch for camera config changes so tray menus stay current
        self.camera_refresh_timer = QTimer()
        self.camera_refresh_timer.timeout.connect(self.check_camera_updates)
        self.camera_refresh_timer.start(5000)

        # Kick off an initial sync from cameras.json once the camera loop is ready
        self._initial_sync_timer = QTimer()
        self._initial_sync_timer.timeout.connect(self._maybe_initial_sync_from_json)
        self._initial_sync_timer.start(1000)
        QTimer.singleShot(1500, self._maybe_load_startup_layout)

        # 3. Auto-start core services if prefs say so and they aren't already running.
        self.server_manager = None
        self.core_thread = None
        self._auto_start_services()

        # Recording status cache (polled from backend)
        self._recording_status_cache: dict[str, bool] = {}
        self._recording_status_lock = threading.Lock()
        self._rec_status_timer = QTimer()
        self._rec_status_timer.timeout.connect(self._poll_recording_status)
        self._rec_status_timer.start(10_000)

        # 4. Setup System Tray
        self.setup_tray_icon()

        # 4b. First-run startup wizard (blocks until dismissed)
        try:
            prefs = self._load_prefs()
            if not prefs.get("setup_complete"):
                from desktop.widgets.startup_wizard import StartupWizard
                wiz = StartupWizard(
                    prefs_file=self.prefs_file,
                    cameras_json_path=self.cameras_json_path,
                )
                wiz.exec()
                self.refresh_cameras_from_json()
        except Exception as exc:
            logger.warning("Startup wizard could not run: %s", exc)

        # 5. Start scheduler loop (best-effort; fully driven by prefs)
        try:
            prefs = self._load_prefs()
            enabled = True if ("scheduler_enabled" not in prefs) else bool(prefs.get("scheduler_enabled"))
            if enabled:
                self._scheduler_timer.start(30_000)  # 30s
                QTimer.singleShot(3500, self._scheduler_tick)
        except Exception:
            # Never block startup on scheduler issues
            pass

        # 5b. Start auto-protection load shedder (best-effort)
        try:
            self._init_load_shedder()
        except Exception as exc:
            logger.warning("Auto-protection load shedder failed to start: %s", exc)

        # 5c. Catch ALL quit paths (tray quit, dialog quit, OS signal,
        # window close, etc) so background threads are torn down before
        # QApplication destruction.  Without this, RTSP capture threads
        # outlive the QApplication and crash with 'Signal source has
        # been deleted' / SIGABRT during process exit.
        try:
            self.aboutToQuit.connect(self._on_about_to_quit)
        except Exception:
            pass
        self._shutdown_done: bool = False
        
        # 6. Background lifecycle service (update checks + entitlement handshake)
        try:
            from desktop.services.lifecycle_service import LifecycleService
            self._lifecycle = LifecycleService(parent=self)
            self._lifecycle.update_available.connect(self._on_update_available)
            self._lifecycle.entitlement_changed.connect(self.rebuild_tray_menu)
            self._lifecycle.start()
        except Exception as exc:
            logger.warning("LifecycleService failed to start: %s", exc)
            self._lifecycle = None

        self._pending_update: tuple[str, str, str] | None = None  # (current, latest, channel)

        logger.info("Desktop Application Started.")

    def _auto_start_services(self):
        """Start Backend and/or MediaMTX if their auto-start prefs are enabled."""
        try:
            prefs = self._load_prefs()
        except Exception:
            prefs = {}

        backend_up = self.is_backend_running()
        mediamtx_up = self.is_backend_running(port=9997)

        if prefs.get("autostart_backend") and not backend_up:
            logger.info("Auto-starting Backend (pref enabled)…")
            threading.Thread(target=self._autostart_service, args=("Backend", "app.py"), daemon=True).start()
        elif backend_up:
            logger.info("Backend already running.")
        else:
            logger.info("Backend not running (auto-start disabled).")

        if prefs.get("autostart_mediamtx") and not mediamtx_up:
            logger.info("Auto-starting MediaMTX (pref enabled)…")
            threading.Thread(target=self._autostart_service, args=("MediaMTX", None), daemon=True).start()
        elif mediamtx_up:
            logger.info("MediaMTX already running.")
        else:
            logger.info("MediaMTX not running (auto-start disabled).")

        # Show system manager if both core services are down and no auto-start is configured
        if not backend_up and not mediamtx_up and not prefs.get("autostart_backend") and not prefs.get("autostart_mediamtx"):
            QTimer.singleShot(2000, self._prompt_services_down)

    def _prompt_services_down(self):
        """Show the system manager dialog when no core services are detected."""
        try:
            if not self.is_backend_running() and not self.is_backend_running(port=9997):
                self.tray_icon.showMessage(
                    "Knoxnet VMS Beta",
                    "Backend and MediaMTX are not running.\nOpen System Management to start services.",
                    QSystemTrayIcon.MessageIcon.Warning,
                    5000,
                )
                self.open_system_manager()
        except Exception:
            pass

    def _autostart_service(self, name: str, entry_point: str | None):
        """Start a service in the background (called from a thread)."""
        try:
            from desktop.widgets.system_manager import SystemManagerDialog
            dlg = SystemManagerDialog.__new__(SystemManagerDialog)
            dlg._app = self
            dlg._processes = {}
            dlg.service_widgets = {}
            from pathlib import Path

            if name == "MediaMTX":
                from desktop.widgets.system_manager import _mediamtx_entrypoint
                entry_point = _mediamtx_entrypoint()

            if not entry_point:
                return

            root = Path(__file__).resolve().parents[1]
            ep = root / entry_point

            if name == "Backend":
                import subprocess, sys as _sys
                env = dict(os.environ)
                env["KNOXNET_SIMPLE_SERVER"] = "1"
                env["PYTHONIOENCODING"] = "utf-8"
                subprocess.Popen(
                    [_sys.executable, str(ep)],
                    cwd=str(root), env=env,
                    stdout=open("/tmp/knoxnet_backend.log", "a"),
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
                logger.info("Backend auto-started.")
            elif name == "MediaMTX":
                import subprocess
                mtx_dir = ep.parent
                compat = mtx_dir / "mediamtx_compat.yml"
                cfg = compat if compat.exists() else mtx_dir / "mediamtx.yml"
                subprocess.Popen(
                    [str(ep), str(cfg)],
                    cwd=str(mtx_dir),
                    stdout=open("/tmp/knoxnet_mediamtx.log", "a"),
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
                logger.info("MediaMTX auto-started.")
        except Exception as e:
            logger.warning("Auto-start %s failed: %s", name, e)

    def is_backend_running(self, host="127.0.0.1", port=5000):
        """Check if the backend API is accessible."""
        import socket
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.settimeout(1)
                return s.connect_ex((host, port)) == 0
        except:
            return False

    def run_camera_manager(self):
        """Run the asyncio loop for CameraManager."""
        import asyncio
        loop = asyncio.new_event_loop()
        self.camera_loop = loop
        asyncio.set_event_loop(loop)
        
        # Bridge signals
        # We define a thread-safe callback here.  The shutdown guard
        # is critical: capture threads can call this fractionally
        # after the QApplication has started tearing down, and
        # emitting a signal on a deleted QObject causes a hard crash.
        def frame_callback(camera_id, frame):
            if getattr(self, "_shutdown_done", False):
                return
            try:
                self.frame_signal.emit(camera_id, frame)
            except RuntimeError:
                # "Signal source has been deleted" -- happens during
                # late shutdown; nothing to do, just drop the frame.
                return
            except Exception:
                return

        self.camera_manager.on_frame_received = frame_callback
        
        loop.run_until_complete(self.camera_manager.start())
        loop.run_forever()

    def setup_tray_icon(self):
        self.tray_icon = QSystemTrayIcon(self)
        self.tray_icon.setIcon(self._load_tray_icon())
        self.tray_icon.setToolTip(self._build_status_tooltip())

        menu = QMenu()
        self.tray_icon.setContextMenu(menu)
        menu.aboutToShow.connect(self.rebuild_tray_menu)
        self.rebuild_tray_menu()

        # Left-click and double-click both open the context menu
        self.tray_icon.activated.connect(self._on_tray_activated)
        self.tray_icon.show()

        # Refresh tooltip every 5 seconds for live hardware stats
        self._tooltip_timer = QTimer(self)
        self._tooltip_timer.timeout.connect(self._refresh_tooltip)
        self._tooltip_timer.start(5_000)

    def _on_tray_activated(self, reason):
        """Handle left-click and double-click on the tray icon."""
        trigger = QSystemTrayIcon.ActivationReason.Trigger
        double_click = QSystemTrayIcon.ActivationReason.DoubleClick
        if reason in (trigger, double_click):
            menu = self.tray_icon.contextMenu()
            if menu:
                self.rebuild_tray_menu()
                # Position the menu near the tray icon
                geo = self.tray_icon.geometry()
                if geo.isValid():
                    menu.popup(geo.topLeft())
                else:
                    from PySide6.QtGui import QCursor
                    menu.popup(QCursor.pos())

    def _load_tray_icon(self) -> QIcon:
        """Load the Knoxnet tray icon from bundled assets."""
        return load_knoxnet_icon(prefer_tray=True)

    def _build_status_tooltip(self) -> str:
        """Rich tooltip with plan, cameras, CPU, GPU, memory, and temps."""
        import psutil

        lines = ["Knoxnet VMS Beta"]

        # Plan & cameras
        try:
            from core.entitlements import load_entitlement, get_camera_limit, _effective_entitlement
            raw = load_entitlement()
            eff = _effective_entitlement(raw)
            tier = (eff.get("tier") or "free").upper()
            limit = get_camera_limit()
            try:
                from desktop.widgets.license_dialog import LicenseDialog
                count = LicenseDialog._count_cameras_from_disk()
            except Exception:
                count = 0
            lines.append(f"Plan: {tier}  \u2022  Cameras: {count}/{limit}")
        except Exception:
            lines.append("Plan: FREE  \u2022  Cameras: 0/4")

        # Active streams
        try:
            cams = getattr(self.camera_manager, "cameras", {}) if self.camera_manager else {}
            active = sum(1 for c in (cams or {}).values() if getattr(c, "enabled", False))
            lines.append(f"Active streams: {active}")
        except Exception:
            pass

        # Recording status
        try:
            with self._recording_status_lock:
                rec = dict(self._recording_status_cache)
            total = len(rec)
            recording = sum(1 for v in rec.values() if v)
            if total:
                lines.append(f"Recording: {recording}/{total} cameras")
        except Exception:
            pass

        lines.append("")

        # CPU
        try:
            cpu_pct = psutil.cpu_percent(interval=0)
            cpu_count = psutil.cpu_count(logical=True)
            lines.append(f"CPU: {cpu_pct:.0f}%  ({cpu_count} cores)")
        except Exception:
            pass

        # Memory
        try:
            mem = psutil.virtual_memory()
            used_gb = mem.used / (1024 ** 3)
            total_gb = mem.total / (1024 ** 3)
            lines.append(f"RAM: {used_gb:.1f}/{total_gb:.1f} GB  ({mem.percent:.0f}%)")
        except Exception:
            pass

        # GPU
        try:
            from core.utils.detector_device import probe_capabilities
            caps = probe_capabilities()
            if caps.has_gpu:
                gpu_info = self._gpu_stats()
                if gpu_info:
                    lines.append(gpu_info)
                else:
                    lines.append("GPU: CUDA available")
            else:
                lines.append("GPU: None (CPU mode)")
        except Exception:
            lines.append("GPU: N/A")

        # CPU temperature (best-effort, not available on all platforms)
        try:
            temps = psutil.sensors_temperatures()
            if temps:
                for name, entries in temps.items():
                    if entries:
                        t = entries[0].current
                        if t and t > 0:
                            lines.append(f"Temp: {t:.0f}\u00b0C ({name})")
                            break
        except Exception:
            pass

        # Disk
        try:
            disk = psutil.disk_usage(".")
            free_gb = disk.free / (1024 ** 3)
            total_gb = disk.total / (1024 ** 3)
            lines.append(f"Disk: {free_gb:.0f}/{total_gb:.0f} GB free")
        except Exception:
            pass

        # Auto-protection status (always visible, even when the per-camera
        # debug overlay is opted-out)
        try:
            if self._load_shedder is not None and self._load_shedder.enabled:
                lvl = self._load_shedder.current_level
                if lvl > LoadLevel.NORMAL:
                    throttles = self._load_shedder.get_throttles_for_level(lvl)
                    parts = [f"Auto-Protect: {lvl.label}"]
                    if "paint_fps" in throttles:
                        parts.append(f"live {throttles['paint_fps']}fps")
                    if "detector_fps" in throttles:
                        v = int(throttles["detector_fps"])
                        parts.append("AI off" if v == 0 else f"AI {v}fps")
                    lines.append(" · ".join(parts))
        except Exception:
            pass

        return "\n".join(lines)

    @staticmethod
    def _gpu_stats() -> str:
        """Try to get NVIDIA GPU utilisation, memory, and temp via nvidia-smi."""
        try:
            import subprocess as _sp
            out = _sp.check_output(
                ["nvidia-smi",
                 "--query-gpu=name,utilization.gpu,memory.used,memory.total,temperature.gpu",
                 "--format=csv,noheader,nounits"],
                timeout=3, text=True, stderr=_sp.DEVNULL,
                creationflags=_sp.CREATE_NO_WINDOW if sys.platform == "win32" else 0,
            ).strip()
            if not out:
                return ""
            parts = [p.strip() for p in out.split(",")]
            if len(parts) >= 5:
                name, util, mem_used, mem_total, temp = parts[:5]
                return (
                    f"GPU: {name}  {util}%  "
                    f"{mem_used}/{mem_total} MB  {temp}\u00b0C"
                )
            return f"GPU: {out}"
        except Exception:
            return ""

    def _refresh_tooltip(self):
        if self.tray_icon:
            self.tray_icon.setToolTip(self._build_status_tooltip())

    def _poll_recording_status(self):
        """Background poll of recording status from the backend (non-blocking)."""
        def _fetch():
            try:
                resp = requests.get("http://localhost:5000/api/cameras/recording-status", timeout=2)
                data = resp.json().get("data", {})
                with self._recording_status_lock:
                    self._recording_status_cache = {k: bool(v) for k, v in data.items()}
            except Exception:
                pass
        threading.Thread(target=_fetch, daemon=True).start()

    def rebuild_tray_menu(self):
        """Rebuild the tray menu to reflect current cameras and quick widget shortcuts."""
        menu = self.tray_icon.contextMenu()
        menu.clear()

        terminal_action = QAction("Open Terminal Widget", self)
        terminal_action.triggered.connect(lambda _: self.spawn_terminal_widget())
        menu.addAction(terminal_action)

        model_library_action = QAction("Model Library…", self)
        model_library_action.triggered.connect(self.open_model_library)
        menu.addAction(model_library_action)

        system_mgmt_action = QAction("System Management & Health…", self)
        system_mgmt_action.triggered.connect(self.open_system_manager)
        menu.addAction(system_mgmt_action)

        live_report_action = QAction("Live Security Report", self)
        live_report_action.triggered.connect(self._open_live_report)
        menu.addAction(live_report_action)

        layouts_menu = menu.addMenu("Layouts")
        save_layout_action = QAction("Save Current Layout…", self)
        save_layout_action.triggered.connect(self.prompt_save_layout)
        layouts_menu.addAction(save_layout_action)

        run_layouts_action = QAction("Run Layout(s)…", self)
        run_layouts_action.triggered.connect(self.prompt_run_layouts)
        layouts_menu.addAction(run_layouts_action)

        layout_settings_action = QAction("Layout Settings…", self)
        layout_settings_action.triggered.connect(self.edit_layout_settings)
        layout_settings_action.setEnabled(bool(self.current_layout_name))
        layouts_menu.addAction(layout_settings_action)

        manage_layouts_action = QAction("Manage Layouts…", self)
        manage_layouts_action.triggered.connect(self.open_layout_manager)
        layouts_menu.addAction(manage_layouts_action)

        # Scheduler quick controls
        try:
            prefs = self._load_prefs()
        except Exception:
            prefs = {}
        try:
            sched_enabled = True if ("scheduler_enabled" not in prefs) else bool(prefs.get("scheduler_enabled"))
        except Exception:
            sched_enabled = True
        sched_menu = layouts_menu.addMenu("Scheduler")
        toggle_sched = QAction("Enabled", self)
        toggle_sched.setCheckable(True)
        toggle_sched.setChecked(bool(sched_enabled))

        def _toggle_scheduler(val: bool):
            p = self._load_prefs()
            if not isinstance(p, dict):
                p = {}
            p["scheduler_enabled"] = bool(val)
            try:
                if bool(val):
                    if not self._scheduler_timer.isActive():
                        self._scheduler_timer.start(30_000)
                        QTimer.singleShot(1000, self._scheduler_tick)
                else:
                    if self._scheduler_timer.isActive():
                        self._scheduler_timer.stop()
            except Exception:
                pass
            self._save_prefs(p)

        toggle_sched.toggled.connect(_toggle_scheduler)
        sched_menu.addAction(toggle_sched)

        edit_sched = QAction("Edit schedules…", self)
        edit_sched.triggered.connect(self.open_layout_schedules)
        sched_menu.addAction(edit_sched)

        snooze_1h = QAction("Snooze 1 hour", self)

        def _snooze_1h():
            p = self._load_prefs()
            if not isinstance(p, dict):
                p = {}
            p["scheduler_snooze_until"] = (datetime.now() + timedelta(hours=1)).isoformat(timespec="seconds")
            self._save_prefs(p)

        snooze_1h.triggered.connect(_snooze_1h)
        sched_menu.addAction(snooze_1h)

        clear_snooze = QAction("Clear snooze", self)

        def _clear_snooze():
            p = self._load_prefs()
            if not isinstance(p, dict):
                p = {}
            p["scheduler_snooze_until"] = None
            self._save_prefs(p)

        clear_snooze.triggered.connect(_clear_snooze)
        sched_menu.addAction(clear_snooze)

        # --- Inline layout status rows (indicator + Start/Pause/Stop) ---
        layouts_menu.addSeparator()
        self._add_layout_status_rows(layouts_menu)
        layouts_menu.addSeparator()

        layouts = self._list_layouts_v2()
        prefs = self._load_prefs()
        startup_layout = prefs.get("startup_layout")
        if layouts:
            # Layout actions now shown as inline rows above.
            startup_menu = layouts_menu.addMenu("Load on Startup")
            startup_group = QActionGroup(self)
            startup_group.setExclusive(True)

            none_action = QAction("None", self)
            none_action.setCheckable(True)
            none_action.setChecked(startup_layout is None)
            none_action.triggered.connect(lambda checked: checked and self._set_startup_layout(None))
            startup_group.addAction(none_action)
            startup_menu.addAction(none_action)

            for layout in layouts:
                start_act = QAction(layout.name, self)
                start_act.setCheckable(True)
                start_act.setChecked(layout.id == startup_layout)
                start_act.triggered.connect(lambda checked, lid=layout.id: checked and self._set_startup_layout(lid))
                startup_group.addAction(start_act)
                startup_menu.addAction(start_act)
        else:
            act = QAction("No saved layouts", self)
            act.setEnabled(False)
            layouts_menu.addAction(act)

        profiles_menu = menu.addMenu("Profiles")
        save_profile_action = QAction("Save Profile from Camera…", self)
        save_profile_action.triggered.connect(self.prompt_save_profile_from_camera)
        profiles_menu.addAction(save_profile_action)
        apply_profile_action = QAction("Apply Profile to Cameras…", self)
        apply_profile_action.triggered.connect(self.prompt_apply_profile_to_cameras)
        profiles_menu.addAction(apply_profile_action)
        profiles_menu.addSeparator()
        global_overlay_action = QAction("Global Overlay Settings…", self)
        global_overlay_action.triggered.connect(self.prompt_global_overlay_settings)
        profiles_menu.addAction(global_overlay_action)
        profiles_menu.addSeparator()

        def _any_cam_has(attr: str) -> bool:
            try:
                from desktop.widgets.camera import CameraWidget
                return any(
                    bool(getattr(w, attr, False))
                    for w in list(self.active_widgets)
                    if isinstance(w, CameraWidget) and w.isVisible()
                )
            except Exception:
                return False

        motion_toggle = QAction("Motion Boxes (all cameras)", self)
        motion_toggle.setCheckable(True)
        motion_toggle.setChecked(_any_cam_has("motion_boxes_enabled"))
        motion_toggle.toggled.connect(lambda on: self._ipc_set_motion_boxes_all({"enabled": on}))
        profiles_menu.addAction(motion_toggle)

        detect_toggle = QAction("Object Detection (all cameras)", self)
        detect_toggle.setCheckable(True)
        detect_toggle.setChecked(_any_cam_has("desktop_object_detection_enabled"))
        detect_toggle.toggled.connect(lambda on: self._ipc_set_object_detection_all({"enabled": on}))
        profiles_menu.addAction(detect_toggle)

        watch_toggle = QAction("Motion Watch (all cameras)", self)
        watch_toggle.setCheckable(True)
        watch_toggle.setChecked(_any_cam_has("motion_watch_active"))
        def _toggle_watch(on):
            if on:
                self._ipc_start_motion_watch_all({})
            else:
                self._ipc_stop_motion_watch_all()
        watch_toggle.toggled.connect(_toggle_watch)
        profiles_menu.addAction(watch_toggle)

        record_toggle = QAction("Record All Cameras", self)
        record_toggle.setCheckable(True)
        record_toggle.setChecked(_any_cam_has("_continuous_recording"))
        def _toggle_record_all(on):
            self._ipc_toggle_recording_all({"record": on})
        record_toggle.toggled.connect(_toggle_record_all)
        profiles_menu.addAction(record_toggle)

        cam_menu = menu.addMenu("Open Camera")
        devices = self._load_menu_devices()
        with self._recording_status_lock:
            rec_cache = dict(self._recording_status_cache)
        if devices:
            for cam_id, label in devices:
                display = f"\U0001F534 {label}" if rec_cache.get(cam_id) else label
                act = QAction(display, self)
                act.triggered.connect(lambda _, cid=cam_id: self.spawn_camera_widget(cid))
                cam_menu.addAction(act)
        else:
            act = QAction("No cameras available", self)
            act.setEnabled(False)
            cam_menu.addAction(act)

        menu.addSeparator()

        # Common web widgets mirroring the browser UI
        cam_config_action = QAction("Camera Config", self)
        cam_config_action.triggered.connect(self.open_camera_config_dialog)
        menu.addAction(cam_config_action)

        discover_action = QAction("Discover Cameras…", self)
        discover_action.triggered.connect(self._open_camera_scanner)
        menu.addAction(discover_action)

        menu.addSeparator()

        tier_label, cam_info = self._get_tier_summary()
        license_action = QAction(f"Beta Plan: {tier_label}  ({cam_info})", self)
        license_action.triggered.connect(self._open_license_dialog)
        menu.addAction(license_action)


        menu.addSeparator()
        quit_action = QAction("Quit", self)
        quit_action.triggered.connect(self.quit_app)
        menu.addAction(quit_action)

    # -- Beta plan and scanner helpers --

    def _get_tier_summary(self) -> tuple[str, str]:
        """Return (tier_display, camera_info) for the tray menu label."""
        try:
            from core.entitlements import load_entitlement, get_camera_limit, _effective_entitlement
            raw = load_entitlement()
            eff = _effective_entitlement(raw)
            tier = (eff.get("tier") or "free").upper()
            limit = get_camera_limit()
            try:
                from desktop.widgets.license_dialog import LicenseDialog
                count = LicenseDialog._count_cameras_from_disk()
            except Exception:
                count = 0
            return tier, f"{count}/{limit} cameras"
        except Exception:
            return "BETA FREE", "0/4 cameras"

    def _open_license_dialog(self):
        try:
            from desktop.widgets.license_dialog import LicenseDialog
            dlg = LicenseDialog(parent=None)
            dlg.entitlement_changed.connect(self.rebuild_tray_menu)
            dlg.exec()
        except Exception as exc:
            logger.warning("Could not open license dialog: %s", exc)

    def _open_camera_scanner(self):
        try:
            from desktop.widgets.scanner_dialog import CameraScannerDialog
            dlg = CameraScannerDialog(
                cameras_json_path=self.cameras_json_path,
                parent=None,
            )
            dlg.exec()
            self.refresh_cameras_from_json()
        except Exception as exc:
            logger.warning("Could not open camera scanner: %s", exc)

    @Slot(str, str, str)
    def _on_update_available(self, current: str, latest: str, channel: str):
        self._pending_update = (current, latest, channel)
        self.rebuild_tray_menu()
        try:
            self.tray_icon.showMessage(
                "Knoxnet VMS Beta",
                f"Update available: version {latest}. Download the latest beta from GitHub.",
                QSystemTrayIcon.MessageIcon.Information,
                10000,
            )
        except Exception:
            pass
        self._show_update_dialog()

    def _show_update_dialog(self):
        if not self._pending_update:
            return
        current, latest, channel = self._pending_update
        try:
            from desktop.widgets.update_banner import UpdateAvailableDialog
            dlg = UpdateAvailableDialog(
                current_version=current,
                latest_version=latest,
                channel=channel,
                parent=None,
            )
            self._update_dialog = dlg
            dlg.show()
        except Exception as exc:
            logger.warning("Could not show update dialog: %s", exc)

    def _check_for_updates_manual(self):
        """Triggered from tray menu — force an immediate update check."""
        lc = getattr(self, "_lifecycle", None)
        if lc:
            lc.force_update_check()
        else:
            logger.info("LifecycleService not available for manual update check")

    def _refresh_camera_manager_from_api(self, prune_missing: bool = False, timeout_s: float = 2.0) -> bool:
        """Best-effort refresh of CameraManager from backend API."""
        cm = getattr(self, "camera_manager", None)
        if not cm:
            return False
        loop = getattr(cm, "_loop", None) or getattr(self, "camera_loop", None)
        if not loop or not getattr(loop, "is_running", lambda: False)():
            return False

        async def _do_sync():
            try:
                return await cm.sync_cameras_api_to_db("http://localhost:5000/api", prune_missing=prune_missing)
            except Exception:
                return 0

        try:
            future = asyncio.run_coroutine_threadsafe(_do_sync(), loop)
            future.result(timeout=timeout_s)
            return True
        except Exception:
            return False

    def _load_menu_devices(self) -> list[tuple[str, str]]:
        """
        Fetch devices for the tray menu without blocking UI.
        Uses a short-lived cache and refreshes in the background.
        """
        with self._tray_cam_cache_lock:
            cached = list(self._tray_cam_cache)
            cached_fresh = (time.time() - self._tray_cam_cache_ts) < 3.0 if self._tray_cam_cache_ts else False

        if not cached_fresh:
            self._refresh_tray_camera_cache_async()

        if cached:
            return cached

        devices: list[tuple[str, str]] = []
        cams = getattr(self.camera_manager, "cameras", {}) if self.camera_manager else {}
        for cam_id, cfg in (cams or {}).items():
            label = cfg.name or cam_id
            devices.append((cam_id, label))
        devices.sort(key=lambda item: item[1].lower())
        return devices

    def _refresh_tray_camera_cache_async(self) -> None:
        if self._tray_cam_refresh_inflight:
            return
        self._tray_cam_refresh_inflight = True

        def _worker():
            devices: list[tuple[str, str]] = []
            try:
                resp = requests.get("http://localhost:5000/api/devices", timeout=2)
                if getattr(resp, "ok", False):
                    payload = resp.json() or {}
                    api_devices = payload.get("data") if isinstance(payload, dict) else None
                    if api_devices is None:
                        api_devices = payload.get("devices") if isinstance(payload, dict) else None
                    if api_devices is None:
                        api_devices = payload if isinstance(payload, list) else []
                    seen: set[str] = set()
                    for cam in api_devices:
                        if not isinstance(cam, dict):
                            continue
                        if cam.get("type") not in (None, "camera") and cam.get("device_type") not in (None, "camera"):
                            continue
                        cam_id = str(cam.get("id") or "").strip()
                        if not cam_id or cam_id in seen:
                            continue
                        seen.add(cam_id)
                        name = str(cam.get("name") or cam_id).strip()
                        devices.append((cam_id, name))
            except Exception:
                devices = []

            devices.sort(key=lambda item: item[1].lower())
            with self._tray_cam_cache_lock:
                if devices:
                    self._tray_cam_cache = devices
                    self._tray_cam_cache_ts = time.time()
            self._tray_cam_refresh_inflight = False

            if devices:
                QTimer.singleShot(0, lambda: self._refresh_camera_manager_from_api(prune_missing=True))
                QTimer.singleShot(0, self.rebuild_tray_menu)

        threading.Thread(target=_worker, daemon=True).start()

    def open_model_library(self):
        """Open the desktop model library UI (download/manage model artifacts)."""
        try:
            from desktop.widgets.model_library import ModelLibraryDialog
        except Exception as e:
            logger.error(f"Failed to load ModelLibraryDialog: {e}")
            QMessageBox.warning(None, "Model Library", f"Unable to open model library: {e}")
            return

        if not hasattr(self, "_model_library_dialog") or self._model_library_dialog is None:
            self._model_library_dialog = ModelLibraryDialog(parent=None)
            self._model_library_dialog.finished.connect(lambda: setattr(self, "_model_library_dialog", None))

        self._model_library_dialog.show()
        self._model_library_dialog.raise_()
        self._model_library_dialog.activateWindow()

    def open_system_manager(self):
        """Open the consolidated System Management & Health UI."""
        try:
            from desktop.widgets.system_manager import SystemManagerDialog
        except Exception as e:
            logger.error(f"Failed to load SystemManagerDialog: {e}")
            QMessageBox.warning(None, "System Management", f"Unable to open System Management: {e}")
            return

        if self._system_manager_dialog is None:
            self._system_manager_dialog = SystemManagerDialog(parent=None, app=self)
            self._system_manager_dialog.finished.connect(lambda: setattr(self, "_system_manager_dialog", None))

        self._system_manager_dialog.show()
        self._system_manager_dialog.raise_()
        self._system_manager_dialog.activateWindow()

    def _open_live_report(self):
        """Open the live security report dashboard in the system browser."""
        try:
            url = "http://localhost:5000/api/events/live-report"
            from PySide6.QtCore import QUrl
            from PySide6.QtGui import QDesktopServices
            QDesktopServices.openUrl(QUrl(url))
        except Exception as e:
            logger.warning(f"Failed to open live report: {e}")

    def open_layout_manager(self):
        """Open the layout manager dialog (manage layouts individually and in bulk)."""
        try:
            from desktop.widgets.layout_manager import LayoutManagerDialog
        except Exception as e:
            logger.error(f"Failed to load LayoutManagerDialog: {e}")
            QMessageBox.warning(None, "Layout Manager", f"Unable to open Layout Manager: {e}")
            return

        if not hasattr(self, "_layout_manager_dialog") or self._layout_manager_dialog is None:
            self._layout_manager_dialog = LayoutManagerDialog(parent=None, app=self)
            self._layout_manager_dialog.finished.connect(lambda: setattr(self, "_layout_manager_dialog", None))

        self._layout_manager_dialog.show()
        self._layout_manager_dialog.raise_()
        self._layout_manager_dialog.activateWindow()

    def open_layout_schedules(self):
        """Open the layout schedules dialog."""
        try:
            from desktop.widgets.layout_manager import LayoutSchedulesDialog
        except Exception as e:
            logger.error(f"Failed to load LayoutSchedulesDialog: {e}")
            QMessageBox.warning(None, "Layout Schedules", f"Unable to open Layout Schedules: {e}")
            return
        try:
            dlg = LayoutSchedulesDialog(parent=None, app=self)
            dlg.exec()
        except Exception:
            return

    # ================== Layout quick controls (tray) ==================

    def _layout_state(self, layout_id: str) -> dict:
        """
        Returns a dict with:
          - current: bool
          - running: bool (current or background session running)
          - paused: bool
          - session_id: str|None
          - widgets: list
        """
        state = {
            "current": False,
            "running": False,
            "paused": False,
            "hidden": False,
            "session_id": None,
            "widgets": [],
        }
        if not layout_id:
            return state

        if self.current_layout_name and layout_id == self.current_layout_name:
            state["current"] = True
            state["running"] = True
            state["widgets"] = list(self._current_layout_widgets or [])
            state["paused"] = layout_id in (self._layout_paused or set())
            state["hidden"] = layout_id in (self._layout_hidden or set())
            return state

        sid = (self._layout_sessions or {}).get(layout_id)
        if sid:
            state["session_id"] = sid
            # Consider running if the session says so OR we still have widgets tracked for it.
            try:
                sess = self.session_manager.get_session(sid) if self.session_manager else None
                if sess and getattr(sess, "status", None) == "running":
                    state["running"] = True
            except Exception:
                pass
            widgets = list((self._session_widgets or {}).get(sid, []) or [])
            state["widgets"] = widgets
            if widgets:
                state["running"] = True
            state["paused"] = layout_id in (self._layout_paused or set())
            state["hidden"] = layout_id in (self._layout_hidden or set())

            # Prune stale mappings if nothing is actually alive anymore.
            if not state["running"]:
                try:
                    self._layout_sessions.pop(layout_id, None)
                except Exception:
                    pass
        return state

    def _get_layout_auto_hide(self, layout_id: str) -> dict:
        """Return auto-hide settings for a layout (with safe defaults)."""
        prefs = self._load_prefs()
        auto = prefs.get("layout_auto_hide") if isinstance(prefs, dict) else {}
        cfg = auto.get(layout_id) if isinstance(auto, dict) else None
        if not isinstance(cfg, dict):
            cfg = {}
        return {
            "on_layout_switch": bool(cfg.get("on_layout_switch", False)),
            "on_motion": bool(cfg.get("on_motion", False)),
            "on_detections": bool(cfg.get("on_detections", False)),
        }

    def _get_layout_auto_show(self, layout_id: str) -> dict:
        """Return auto-show settings for a layout (with safe defaults)."""
        prefs = self._load_prefs()
        auto = prefs.get("layout_auto_show") if isinstance(prefs, dict) else {}
        cfg = auto.get(layout_id) if isinstance(auto, dict) else None
        if not isinstance(cfg, dict):
            cfg = {}
        return {
            "on_motion": bool(cfg.get("on_motion", False)),
            "on_detections": bool(cfg.get("on_detections", False)),
        }

    def _camera_widget_key(self, camera_id: str) -> str:
        return f"camera:{str(camera_id)}"

    def _get_widget_auto_hide(self, widget_key: str) -> dict:
        prefs = self._load_prefs()
        auto = prefs.get("widget_auto_hide") if isinstance(prefs, dict) else {}
        cfg = auto.get(widget_key) if isinstance(auto, dict) else None
        if not isinstance(cfg, dict):
            cfg = {}
        return {
            "on_motion": bool(cfg.get("on_motion", False)),
            "on_detections": bool(cfg.get("on_detections", False)),
        }

    def _get_widget_auto_show(self, widget_key: str) -> dict:
        prefs = self._load_prefs()
        auto = prefs.get("widget_auto_show") if isinstance(prefs, dict) else {}
        cfg = auto.get(widget_key) if isinstance(auto, dict) else None
        if not isinstance(cfg, dict):
            cfg = {}
        return {
            "on_motion": bool(cfg.get("on_motion", False)),
            "on_detections": bool(cfg.get("on_detections", False)),
        }

    def _set_widget_hidden_persist(self, widget_key: str, hidden: bool) -> None:
        prefs = self._load_prefs()
        if not isinstance(prefs, dict):
            prefs = {}
        vis = prefs.get("widget_visibility")
        if not isinstance(vis, dict):
            vis = {}
        cfg = vis.get(widget_key)
        if not isinstance(cfg, dict):
            cfg = {}
        cfg["hidden"] = bool(hidden)
        vis[widget_key] = cfg
        prefs["widget_visibility"] = vis
        self._save_prefs(prefs)

    def _is_widget_hidden(self, widget_key: str) -> bool:
        try:
            prefs = self._load_prefs()
            vis = prefs.get("widget_visibility") if isinstance(prefs, dict) else {}
            cfg = vis.get(widget_key) if isinstance(vis, dict) else None
            return bool(cfg.get("hidden")) if isinstance(cfg, dict) else False
        except Exception:
            return False

    def _widgets_for_key(self, widget_key: str) -> list:
        # Today we only support camera widgets: "camera:<camera_id>"
        if not widget_key or ":" not in widget_key:
            return []
        kind, value = widget_key.split(":", 1)
        kind = kind.strip().lower()
        value = value.strip()
        if kind != "camera" or not value:
            return []
        try:
            from desktop.widgets.camera import CameraWidget
        except Exception:
            return []
        out = []
        for w in list(self.active_widgets):
            try:
                if isinstance(w, CameraWidget) and str(getattr(w, "camera_id", "")).lower() == value.lower():
                    out.append(w)
            except Exception:
                continue
        return out

    def _widget_hide(self, widget_key: str, persist: bool = True) -> None:
        for w in self._widgets_for_key(widget_key):
            try:
                w.hide()
            except Exception:
                pass
        if persist:
            self._set_widget_hidden_persist(widget_key, True)

    def _widget_show(self, widget_key: str, persist: bool = True) -> None:
        for w in self._widgets_for_key(widget_key):
            try:
                w.show()
                # If minimized, restore to normal so it actually becomes visible.
                try:
                    if bool(getattr(w, "windowState", None)) and (w.windowState() & Qt.WindowState.WindowMinimized):
                        if hasattr(w, "showNormal"):
                            w.showNormal()
                except Exception:
                    pass
                try:
                    w.raise_()
                except Exception:
                    pass
            except Exception:
                pass
        if persist:
            self._set_widget_hidden_persist(widget_key, False)

    def _set_layout_hidden_persist(self, layout_id: str, hidden: bool) -> None:
        """Persist hidden state into prefs + update in-memory set."""
        layout_id = str(layout_id)
        try:
            if hidden:
                self._layout_hidden.add(layout_id)
            else:
                self._layout_hidden.discard(layout_id)
        except Exception:
            pass

        prefs = self._load_prefs()
        if not isinstance(prefs, dict):
            prefs = {}
        vis = prefs.get("layout_visibility")
        if not isinstance(vis, dict):
            vis = {}
        cfg = vis.get(layout_id)
        if not isinstance(cfg, dict):
            cfg = {}
        cfg["hidden"] = bool(hidden)
        vis[layout_id] = cfg
        prefs["layout_visibility"] = vis
        self._save_prefs(prefs)

    def _layout_hide(self, layout_id: str, persist: bool = True) -> None:
        state = self._layout_state(layout_id)
        for w in list(state.get("widgets") or []):
            try:
                if w is None:
                    continue
                w.hide()
            except Exception:
                pass
        if persist:
            self._set_layout_hidden_persist(layout_id, True)

    def _layout_show(self, layout_id: str, persist: bool = True) -> None:
        state = self._layout_state(layout_id)
        for w in list(state.get("widgets") or []):
            try:
                if w is None:
                    continue
                w.show()
                w.raise_()
            except Exception:
                pass
        if persist:
            self._set_layout_hidden_persist(layout_id, False)

    def _layout_toggle_hidden(self, layout_id: str) -> None:
        if str(layout_id) in (self._layout_hidden or set()):
            self._layout_show(layout_id, persist=True)
        else:
            self._layout_hide(layout_id, persist=True)

    def _on_layout_shape_triggered(self, layout_id: str, payload: dict) -> None:
        """
        Auto-hide triggers based on GL widget shape events.
        Treat:
          - source in ("desktop","backend","detection") as detection-triggered
          - otherwise as motion-triggered

        Also supports timed auto-show/bring-to-front rules keyed by shape_id.
        """
        try:
            hide_cfg = self._get_layout_auto_hide(layout_id)
            show_cfg = self._get_layout_auto_show(layout_id)
            events = (payload or {}).get("events") or []
            if not events:
                return
            src = (payload or {}).get("source") or ""
            src = str(src).lower()
            is_detection = src in {"desktop", "backend", "detection"}

            # Auto-show has priority (only shows if configured)
            if is_detection and bool(show_cfg.get("on_detections")):
                self._layout_show(layout_id, persist=True)
            elif (not is_detection) and bool(show_cfg.get("on_motion")):
                self._layout_show(layout_id, persist=True)
            else:
                # Auto-hide if configured and no auto-show matched
                if is_detection and bool(hide_cfg.get("on_detections")):
                    self._layout_hide(layout_id, persist=True)
                if (not is_detection) and bool(hide_cfg.get("on_motion")):
                    self._layout_hide(layout_id, persist=True)

            # Widget-level auto show/hide (camera widgets) based on camera_id
            cam_id = (payload or {}).get("camera_id")
            if cam_id:
                key = self._camera_widget_key(str(cam_id))
                w_show = self._get_widget_auto_show(key)
                w_hide = self._get_widget_auto_hide(key)
                if is_detection and bool(w_show.get("on_detections")):
                    self._widget_show(key, persist=True)
                elif (not is_detection) and bool(w_show.get("on_motion")):
                    self._widget_show(key, persist=True)
                else:
                    if is_detection and bool(w_hide.get("on_detections")):
                        self._widget_hide(key, persist=True)
                    if (not is_detection) and bool(w_hide.get("on_motion")):
                        self._widget_hide(key, persist=True)

                # Timed auto-show rules: shape-specific (zones/lines/tags)
                try:
                    self._apply_visibility_rules(
                        camera_id=str(cam_id),
                        events=list(events or []),
                        source=("detection" if is_detection else "motion"),
                    )
                except Exception:
                    pass
        except Exception:
            return

    def _on_camera_shape_triggered(self, payload: dict) -> None:
        """
        Shape trigger handler for standalone camera widgets (not opened via layouts).
        Applies widget-level auto show/hide and visibility_rules.
        """
        try:
            events = (payload or {}).get("events") or []
            if not events:
                return
            src = (payload or {}).get("source") or ""
            src = str(src).lower()
            is_detection = src in {"desktop", "backend", "detection"}

            cam_id = (payload or {}).get("camera_id")
            if not cam_id:
                return
            key = self._camera_widget_key(str(cam_id))

            # Existing widget auto show/hide (persisted)
            w_show = self._get_widget_auto_show(key)
            w_hide = self._get_widget_auto_hide(key)
            if is_detection and bool(w_show.get("on_detections")):
                self._widget_show(key, persist=True)
            elif (not is_detection) and bool(w_show.get("on_motion")):
                self._widget_show(key, persist=True)
            else:
                if is_detection and bool(w_hide.get("on_detections")):
                    self._widget_hide(key, persist=True)
                if (not is_detection) and bool(w_hide.get("on_motion")):
                    self._widget_hide(key, persist=True)

            # New visibility rules (shape-specific)
            try:
                self._apply_visibility_rules(
                    camera_id=str(cam_id),
                    events=list(events or []),
                    source=("detection" if is_detection else "motion"),
                )
            except Exception:
                pass
        except Exception:
            return

    def _focus_steal_enabled(self) -> bool:
        try:
            prefs = self._load_prefs()
            val = prefs.get("focus_steal_enabled") if isinstance(prefs, dict) else None
            return True if val is None else bool(val)
        except Exception:
            return True

    def _list_assigned_shapes_for_camera(self, camera_id: str) -> list[dict]:
        """
        Return merged overlay shapes for the camera's assigned profile(s).
        Shape dicts typically include: {"id","kind","label",...}
        """
        camera_id = str(camera_id or "").strip()
        if not camera_id:
            return []
        try:
            assigns = self.layouts_store.get_assignments()
            val = assigns.get(camera_id)
            profile_ids: list[str] = []
            if isinstance(val, str) and val:
                profile_ids = [val]
            elif isinstance(val, list):
                profile_ids = [str(x) for x in val if str(x).strip()]
            if not profile_ids:
                return []

            shapes: list[dict] = []
            seen: set[str] = set()
            for pid in profile_ids:
                prof = self.layouts_store.get_profile(pid)
                if not prof:
                    continue
                ov = prof.overlays or {}
                s = ov.get("shapes")
                if not isinstance(s, list):
                    continue
                for sh in s:
                    try:
                        if not isinstance(sh, dict):
                            continue
                        sid = str(sh.get("id") or "").strip()
                        if not sid or sid in seen:
                            continue
                        seen.add(sid)
                        shapes.append(sh)
                    except Exception:
                        continue
            return shapes
        except Exception:
            return []

    def _get_visibility_rules(self) -> list[dict]:
        """
        Minimal rule model from prefs:
          visibility_rules: [{"trigger": {...}, "target": {...}, "action": {...}}, ...]
        """
        try:
            prefs = self._load_prefs()
            rules = prefs.get("visibility_rules") if isinstance(prefs, dict) else None
            return list(rules) if isinstance(rules, list) else []
        except Exception:
            return []

    def _apply_visibility_rules(self, camera_id: str, events: list[dict], source: str) -> None:
        """
        Evaluate persisted visibility_rules and apply actions.
        source: "motion" | "detection"
        """
        camera_id = str(camera_id or "").strip()
        if not camera_id or not events:
            return
        source = str(source or "").strip().lower()
        if source not in {"motion", "detection"}:
            return

        rules = self._get_visibility_rules()
        if not rules:
            return

        # Collect matches per target widget_key with precedence:
        #   show_and_activate (timed) > show_persisted > hide_persisted
        matched: dict[str, dict] = {}  # widget_key -> {"action": str, "duration": float, "cooldown": float}
        for ev in list(events or []):
            if not isinstance(ev, dict):
                continue
            ev_shape_id = str(ev.get("shape_id") or "").strip()
            ev_kind = str(ev.get("shape_type") or ev.get("kind") or "").strip().lower()
            if not ev_shape_id or ev_kind not in {"zone", "line", "tag"}:
                continue

            for r in rules:
                if not isinstance(r, dict):
                    continue
                trig = r.get("trigger") if isinstance(r.get("trigger"), dict) else {}
                tgt = r.get("target") if isinstance(r.get("target"), dict) else {}
                act = r.get("action") if isinstance(r.get("action"), dict) else {}

                if str(trig.get("type") or "") != "shape_trigger":
                    continue
                if str(tgt.get("type") or "") != "widget":
                    continue
                act_type = str(act.get("type") or "").strip()
                if act_type not in {"show_and_activate", "show_persisted", "hide_persisted", "bring_to_front"}:
                    continue

                if str(trig.get("camera_id") or "") != camera_id:
                    continue
                if str(trig.get("shape_id") or "") != ev_shape_id:
                    continue
                if str(trig.get("kind") or "").strip().lower() != ev_kind:
                    continue
                if str(trig.get("source") or "").strip().lower() != source:
                    continue

                widget_key = str(tgt.get("widget_key") or "").strip()
                if not widget_key:
                    continue
                # Resolve action precedence and parameters
                prev = matched.get(widget_key) or {}
                prev_type = str(prev.get("action") or "")
                _rank_map = {"hide_persisted": 1, "show_persisted": 2, "show_and_activate": 3, "bring_to_front": 4}
                prev_rank = _rank_map.get(prev_type, 0)
                cur_rank = _rank_map.get(act_type, 0)

                if act_type == "show_and_activate":
                    try:
                        duration = float(act.get("duration_sec", 0) or 0)
                    except Exception:
                        duration = 0.0
                    try:
                        cooldown = float(act.get("cooldown_sec", 0) or 0)
                    except Exception:
                        cooldown = 0.0
                    duration = max(0.0, min(3600.0, duration))
                    cooldown = max(0.0, min(3600.0, cooldown))
                    if duration <= 0:
                        continue
                    if cur_rank > prev_rank:
                        matched[widget_key] = {"action": act_type, "duration": duration, "cooldown": cooldown}
                    elif cur_rank == prev_rank:
                        # Same action type: extend duration/cooldown
                        matched[widget_key] = {
                            "action": act_type,
                            "duration": max(float(prev.get("duration") or 0.0), duration),
                            "cooldown": max(float(prev.get("cooldown") or 0.0), cooldown),
                        }
                else:
                    # show_persisted / hide_persisted: no extra params
                    if cur_rank > prev_rank:
                        matched[widget_key] = {"action": act_type}

        if not matched:
            return

        allow_focus = self._focus_steal_enabled()
        for widget_key, cfg in matched.items():
            act_type = str((cfg or {}).get("action") or "")
            if act_type == "show_and_activate":
                try:
                    self._timed_widget_show_and_focus(
                        widget_key=widget_key,
                        duration_sec=float(cfg.get("duration") or 0.0),
                        cooldown_sec=float(cfg.get("cooldown") or 0.0),
                        allow_focus=allow_focus,
                    )
                except Exception:
                    continue
            elif act_type == "show_persisted":
                try:
                    self._widget_show(widget_key, persist=True)
                except Exception:
                    continue
            elif act_type == "hide_persisted":
                try:
                    self._widget_hide(widget_key, persist=True)
                except Exception:
                    continue
            elif act_type == "bring_to_front":
                try:
                    self._widget_bring_to_front(widget_key)
                except Exception:
                    continue

    def _widget_bring_to_front(self, widget_key: str) -> None:
        """Permanently raise a widget without timed re-hide semantics."""
        widgets = self._widgets_for_key(widget_key)
        for w in list(widgets):
            try:
                if w is None:
                    continue
                w.show()
                try:
                    if hasattr(w, "windowState") and (w.windowState() & Qt.WindowState.WindowMinimized):
                        if hasattr(w, "showNormal"):
                            w.showNormal()
                except Exception:
                    pass
                w.raise_()
                w.activateWindow()
                if self._patrol_active:
                    cam_id = str(getattr(w, "camera_id", "") or "")
                    if cam_id:
                        self.patrol_bring_camera_to_front(cam_id)
            except Exception:
                continue

    def _timed_widget_show_and_focus(self, widget_key: str, duration_sec: float, cooldown_sec: float, allow_focus: bool) -> None:
        """
        Temporarily show a widget (without persisting visibility), optionally bring to front,
        and start/extend a timer. On expiry, re-hide only if persisted baseline is hidden.
        """
        widget_key = str(widget_key or "").strip()
        if not widget_key:
            return
        try:
            duration_sec = float(duration_sec or 0.0)
        except Exception:
            duration_sec = 0.0
        try:
            cooldown_sec = float(cooldown_sec or 0.0)
        except Exception:
            cooldown_sec = 0.0
        duration_sec = max(0.0, min(3600.0, duration_sec))
        cooldown_sec = max(0.0, min(3600.0, cooldown_sec))
        if duration_sec <= 0.0:
            return

        now = time.monotonic()
        widgets = self._widgets_for_key(widget_key)
        if not widgets:
            return

        # Capture baseline window state for restore semantics.
        baseline_minimized = False
        try:
            for w in list(widgets):
                try:
                    if w is None:
                        continue
                    if bool(getattr(w, "windowState", None)) and (w.windowState() & Qt.WindowState.WindowMinimized):
                        baseline_minimized = True
                        break
                except Exception:
                    continue
        except Exception:
            baseline_minimized = False

        # Decide whether to focus/activate based on cooldown and current visibility.
        focus_now = False
        if allow_focus:
            any_visible = False
            try:
                for w in list(widgets):
                    try:
                        if w is None:
                            continue
                        if not bool(w.isVisible()):
                            continue
                        # Treat minimized windows as "not visible" for focus/restore purposes.
                        try:
                            if bool(getattr(w, "windowState", None)) and (w.windowState() & Qt.WindowState.WindowMinimized):
                                continue
                        except Exception:
                            pass
                        any_visible = True
                        break
                    except Exception:
                        continue
            except Exception:
                any_visible = False
            last = float(self._timed_visibility_last_focus.get(widget_key, 0.0) or 0.0)
            if (not any_visible) or cooldown_sec <= 0.0 or (now - last) >= cooldown_sec:
                focus_now = True

        # Show without persisting visibility state.
        for w in list(widgets):
            try:
                if w is None:
                    continue
                w.show()
                # Always restore from minimized so it actually appears.
                try:
                    if bool(getattr(w, "windowState", None)) and (w.windowState() & Qt.WindowState.WindowMinimized):
                        if hasattr(w, "showNormal"):
                            w.showNormal()
                except Exception:
                    pass
                if focus_now:
                    try:
                        # On some platforms, clearing minimized and setting active helps.
                        try:
                            if hasattr(w, "setWindowState"):
                                w.setWindowState((w.windowState() & ~Qt.WindowState.WindowMinimized) | Qt.WindowState.WindowActive)
                        except Exception:
                            pass
                    except Exception:
                        pass
                    try:
                        w.raise_()
                    except Exception:
                        pass
                    try:
                        w.activateWindow()
                    except Exception:
                        pass
            except Exception:
                continue

        if focus_now:
            self._timed_visibility_last_focus[widget_key] = now

        # Start/extend timer and remember if this target was hidden in persisted baseline.
        baseline_hidden = bool(self._is_widget_hidden(widget_key))
        entry = self._timed_visibility.get(widget_key)
        if not isinstance(entry, dict):
            entry = {"timer": None, "expire_at": 0.0, "baseline_hidden": False, "baseline_minimized": False, "gen": 0}
            self._timed_visibility[widget_key] = entry
        entry["baseline_hidden"] = bool(entry.get("baseline_hidden")) or baseline_hidden
        entry["baseline_minimized"] = bool(entry.get("baseline_minimized")) or bool(baseline_minimized)

        desired_expire = now + duration_sec
        prev_expire = float(entry.get("expire_at") or 0.0)
        expire_at = max(prev_expire, desired_expire)
        entry["expire_at"] = expire_at

        remaining_ms = int(max(0.0, expire_at - now) * 1000.0)
        if remaining_ms <= 0:
            remaining_ms = 1

        # (Re)arm the timer with a generation guard.
        try:
            gen = int(entry.get("gen") or 0) + 1
        except Exception:
            gen = 1
        entry["gen"] = gen

        t = entry.get("timer")
        if not isinstance(t, QTimer):
            t = QTimer(self)
            t.setSingleShot(True)
            entry["timer"] = t

        def _timeout_closure(_wk=widget_key, _gen=gen):
            try:
                self._on_timed_widget_visibility_timeout(_wk, _gen)
            except Exception:
                return

        try:
            try:
                t.timeout.disconnect()
            except Exception:
                pass
            t.timeout.connect(_timeout_closure)
            t.start(remaining_ms)
        except Exception:
            return

    def _on_timed_widget_visibility_timeout(self, widget_key: str, gen: int) -> None:
        entry = self._timed_visibility.get(widget_key)
        if not isinstance(entry, dict):
            return
        try:
            if int(entry.get("gen") or 0) != int(gen or 0):
                return
        except Exception:
            return

        # Restore semantics:
        # - If the widget was persisted-hidden before timing, re-hide (but only if it still is persisted-hidden).
        # - Else if it was minimized before timing, minimize it again (best-effort), unless user is actively using it.
        try:
            baseline_hidden = bool(entry.get("baseline_hidden"))
        except Exception:
            baseline_hidden = False
        if not baseline_hidden:
            # Try to restore minimized state if that was the baseline.
            try:
                if bool(entry.get("baseline_minimized")):
                    for w in self._widgets_for_key(widget_key):
                        try:
                            if w is None:
                                continue
                            try:
                                if bool(getattr(w, "windowState", None)) and (w.windowState() & Qt.WindowState.WindowMinimized):
                                    continue
                            except Exception:
                                pass
                            # Minimize (best-effort across platforms)
                            try:
                                if hasattr(w, "showMinimized"):
                                    w.showMinimized()
                            except Exception:
                                pass
                            try:
                                if hasattr(w, "setWindowState"):
                                    w.setWindowState(w.windowState() | Qt.WindowState.WindowMinimized)
                            except Exception:
                                pass
                        except Exception:
                            continue
            except Exception:
                pass
            return

        # If user has since persisted it as visible, do not hide.
        if not bool(self._is_widget_hidden(widget_key)):
            return

        # Best-effort runtime hide without persisting.
        try:
            self._widget_hide(widget_key, persist=False)
        except Exception:
            return

    def _close_widget_list(self, widgets: list) -> None:
        """Best-effort close+delete a list of widgets."""
        if not widgets:
            return
        try:
            from desktop.widgets.camera import CameraWidget
        except Exception:
            CameraWidget = None

        for w in list(widgets):
            try:
                if w is None:
                    continue
                try:
                    if CameraWidget and isinstance(w, CameraWidget):
                        try:
                            self.frame_signal.disconnect(w.receive_frame)
                        except Exception:
                            pass
                except Exception:
                    pass
                try:
                    w.close()
                except Exception:
                    pass
                try:
                    w.deleteLater()
                except Exception:
                    pass
            except Exception:
                continue

    def _layout_pause(self, layout_id: str) -> None:
        """Pause layout by releasing cameras for its camera widgets."""
        state = self._layout_state(layout_id)
        widgets = state.get("widgets") or []
        if not widgets:
            return
        try:
            from desktop.widgets.camera import CameraWidget
        except Exception:
            return

        cm = getattr(self, "camera_manager", None)
        if not cm:
            return
        try:
            import asyncio
            if self.camera_loop and self.camera_loop.is_running():
                for w in list(widgets):
                    try:
                        if isinstance(w, CameraWidget):
                            asyncio.run_coroutine_threadsafe(cm.release_camera(str(w.camera_id)), self.camera_loop)
                    except Exception:
                        continue
        except Exception:
            pass
        try:
            self._layout_paused.add(layout_id)
        except Exception:
            pass

    def _layout_resume(self, layout_id: str) -> None:
        """Resume layout by acquiring cameras for its camera widgets."""
        state = self._layout_state(layout_id)
        widgets = state.get("widgets") or []
        if not widgets:
            return
        try:
            from desktop.widgets.camera import CameraWidget
        except Exception:
            return

        cm = getattr(self, "camera_manager", None)
        if not cm:
            return
        try:
            import asyncio
            if self.camera_loop and self.camera_loop.is_running():
                for w in list(widgets):
                    try:
                        if isinstance(w, CameraWidget):
                            self._sync_widget_quality_to_config(w)
                            asyncio.run_coroutine_threadsafe(cm.acquire_camera(str(w.camera_id)), self.camera_loop)
                    except Exception:
                        continue
        except Exception:
            pass
        try:
            self._layout_paused.discard(layout_id)
        except Exception:
            pass

    def _layout_stop(self, layout_id: str) -> None:
        """Stop a layout: close its widgets and stop any associated background session."""
        state = self._layout_state(layout_id)

        # If it's the current layout, close only current layout widgets (not all widgets).
        if bool(state.get("current")):
            self._close_widget_list(list(self._current_layout_widgets or []))
            self._current_layout_widgets = []
            self.current_layout_name = None
            try:
                self._layout_paused.discard(layout_id)
            except Exception:
                pass
            return

        sid = state.get("session_id")
        if sid:
            try:
                self.session_manager.stop_session(sid)
            except Exception:
                pass
            widgets = list((self._session_widgets or {}).get(sid, []) or [])
            self._close_widget_list(widgets)
            try:
                self._session_widgets.pop(sid, None)
            except Exception:
                pass
            try:
                self._layout_sessions.pop(layout_id, None)
            except Exception:
                pass
            try:
                self._layout_paused.discard(layout_id)
            except Exception:
                pass

    def _layout_start_in_background(self, layout_id: str) -> None:
        """
        Start a layout without switching away from the current layout:
          - opens its widgets
          - tracks a single-layout session for stop/pause controls
        """
        if not layout_id:
            return

        state = self._layout_state(layout_id)
        if state.get("paused"):
            self._layout_resume(layout_id)
            return
        if state.get("running"):
            # Bring its widgets to front
            for w in list(state.get("widgets") or []):
                try:
                    w.show()
                    w.raise_()
                    w.activateWindow()
                except Exception:
                    pass
            return

        try:
            sess = self.session_manager.create_session([layout_id], name=f"Background: {layout_id}", meta={"source": "tray_start"})
            self.session_manager.start_session(sess.id)
            created = self._load_layout_internal(layout_id, close_existing=False, set_current=False) or []
            self._session_widgets.setdefault(sess.id, [])
            self._session_widgets[sess.id].extend(list(created))
            self._layout_sessions[layout_id] = sess.id
            self._layout_paused.discard(layout_id)
        except Exception as e:
            try:
                self.tray_icon.showMessage("Knoxnet VMS Beta", f"Failed to start layout '{layout_id}': {e}", QSystemTrayIcon.MessageIcon.Warning)
            except Exception:
                pass

    def _add_layout_status_rows(self, layouts_menu: QMenu) -> None:
        """Add layout entries as native QAction / QMenu items (always visible on all platforms)."""
        layouts = self._list_layouts_v2()
        if not layouts:
            act = QAction("No saved layouts", self)
            act.setEnabled(False)
            layouts_menu.addAction(act)
            return

        def _sort_key(l):
            try:
                if self.current_layout_name and l.id == self.current_layout_name:
                    return (0, l.name.lower())
                return (1, l.name.lower())
            except Exception:
                return (1, str(getattr(l, "name", "")).lower())

        STATUS_ICONS = {
            "hidden": "\u25cb",   # ○
            "paused": "\u25d1",   # ◑
            "current": "\u25c9",  # ◉
            "running": "\u25cf",  # ●
            "stopped": "\u25cb",  # ○
        }

        for layout in sorted(layouts, key=_sort_key):
            lid = layout.id
            lname = layout.name
            state = self._layout_state(lid)
            current = bool(state.get("current"))
            running = bool(state.get("running"))
            paused = bool(state.get("paused"))
            hidden = bool(state.get("hidden"))

            if hidden:
                tip = "hidden"
            elif paused:
                tip = "paused"
            elif current:
                tip = "current"
            elif running:
                tip = "running"
            else:
                tip = "stopped"

            icon = STATUS_ICONS.get(tip, "\u25cb")
            label = f"{icon}  {lname}  [{tip}]"

            sub = layouts_menu.addMenu(label)

            load_act = QAction("Load layout", self)
            load_act.triggered.connect(lambda checked=False, x=lid: self.load_layout(x))
            sub.addAction(load_act)

            start_act = QAction("Start / Resume", self)
            start_act.setEnabled(not current)
            start_act.triggered.connect(lambda checked=False, x=lid: self._layout_start_in_background(x))
            sub.addAction(start_act)

            pause_act = QAction("Pause", self)
            pause_act.setEnabled(running and not paused)
            pause_act.triggered.connect(lambda checked=False, x=lid: self._layout_pause(x))
            sub.addAction(pause_act)

            stop_act = QAction("Stop", self)
            stop_act.setEnabled(running)
            stop_act.triggered.connect(lambda checked=False, x=lid: self._layout_stop(x))
            sub.addAction(stop_act)

            hide_act = QAction("Show" if hidden else "Hide", self)
            hide_act.triggered.connect(lambda checked=False, x=lid: self._layout_toggle_hidden(x))
            sub.addAction(hide_act)

    def open_camera_config_dialog(self):
        """Open the native PyQt camera config tool (shared with React via /api/devices)."""
        try:
            from desktop.widgets.camera import CameraConfigDialog
        except Exception as e:
            logger.error(f"Failed to load CameraConfigDialog: {e}")
            QMessageBox.warning(None, "Camera Config", f"Unable to open camera config: {e}")
            return

        if self.camera_config_dialog is None:
            self.camera_config_dialog = CameraConfigDialog(parent=None, camera_manager=self.camera_manager)
            try:
                self.camera_config_dialog.camera_added.connect(self._on_camera_added)
                self.camera_config_dialog.camera_deleted.connect(self._on_camera_deleted)
                self.camera_config_dialog.camera_updated.connect(self._on_camera_updated)
            except Exception as e:
                logger.warning(f"Failed to connect camera_added signal: {e}")
            self.camera_config_dialog.finished.connect(lambda: setattr(self, "camera_config_dialog", None))

        self.camera_config_dialog.show()
        self.camera_config_dialog.raise_()
        self.camera_config_dialog.activateWindow()

    def _on_camera_added(self, camera_data: dict):
        """
        When a camera is added via the PyQt dialog, immediately refresh the
        CameraManager and tray menu so it appears under Open Camera.
        """
        def _rebuild():
            try:
                self.rebuild_tray_menu()
            except Exception as exc:
                logger.warning(f"Tray rebuild after camera add failed: {exc}")

        try:
            future = self.refresh_cameras_from_json(on_complete=_rebuild)
            if future is None:
                # Fallback if loop not ready; still attempt rebuild soon.
                QTimer.singleShot(750, self.rebuild_tray_menu)
        except Exception as e:
            logger.warning(f"Failed to refresh cameras after add: {e}")
            QTimer.singleShot(750, self.rebuild_tray_menu)

    def _on_camera_deleted(self, camera_ids: list):
        """
        Refresh tray menu after cameras are deleted from the config dialog.
        """
        def _rebuild():
            try:
                self.rebuild_tray_menu()
            except Exception as exc:
                logger.warning(f"Tray rebuild after camera delete failed: {exc}")

        try:
            future = self.refresh_cameras_from_json(on_complete=_rebuild)
            if future is None:
                QTimer.singleShot(750, self.rebuild_tray_menu)
        except Exception as e:
            logger.warning(f"Failed to refresh cameras after delete: {e}")
            QTimer.singleShot(750, self.rebuild_tray_menu)

    def _on_camera_updated(self, camera_data: dict):
        """
        Refresh tray menu after cameras are edited from the config dialog.
        """
        def _rebuild():
            try:
                self.rebuild_tray_menu()
            except Exception as exc:
                logger.warning(f"Tray rebuild after camera update failed: {exc}")

        try:
            future = self.refresh_cameras_from_json(on_complete=_rebuild)
            if future is None:
                QTimer.singleShot(750, self.rebuild_tray_menu)
        except Exception as e:
            logger.warning(f"Failed to refresh cameras after update: {e}")
            QTimer.singleShot(750, self.rebuild_tray_menu)

    def prompt_and_spawn_camera(self):
        """
        Simple GUI prompt to add a camera widget by name, IP, or ID.
        """
        text, ok = QInputDialog.getText(None, "Add Camera Widget", "Camera name / IP / ID:")
        if ok and text.strip():
            self.spawn_camera_widget(text.strip())

    @Slot(dict)
    def handle_ipc_command(self, command):
        cmd_type = command.get("cmd")
        logger.info(f"Handling command: {cmd_type}")

        if cmd_type == "spawn_camera":
            camera_ref = command.get("camera_id") or command.get("camera_ref")
            self.spawn_camera_widget(camera_ref)
        elif cmd_type == "spawn_web":
            url = command.get("url", "")
            title = command.get("title", "Web Widget")
            self.spawn_web_widget(title, url)
        elif cmd_type == "spawn_terminal":
            self.spawn_terminal_widget()
        elif cmd_type == "stop_motion_watch":
            camera_id = command.get("camera_id")
            if not camera_id:
                logger.warning("stop_motion_watch missing camera_id")
                return
            from desktop.widgets.camera import CameraWidget
            for w in list(self.active_widgets):
                if isinstance(w, CameraWidget) and str(w.camera_id).lower() == str(camera_id).lower():
                    try:
                        w.stop_motion_watch("stopped via terminal")
                        logger.info(f"Stopped motion watch for {camera_id}")
                        return
                    except Exception as e:
                        logger.error(f"Failed to stop motion watch for {camera_id}: {e}")
                        return
            logger.warning(f"No camera widget found for {camera_id}")
        elif cmd_type == "start_motion_watch":
            camera_ref = command.get("camera_id") or command.get("camera_ref")
            if not camera_ref:
                logger.warning("start_motion_watch missing camera_ref")
                return
            duration = command.get("duration_sec")
            w = self._find_camera_widget(str(camera_ref))
            if w is None:
                w = self.spawn_camera_widget(str(camera_ref))
            if w is None:
                return
            try:
                if duration is not None:
                    try:
                        w.motion_watch_settings["duration_sec"] = int(duration)
                    except Exception:
                        pass
                w.start_motion_watch()
                logger.info(f"Started motion watch for {camera_ref}")
            except Exception as e:
                logger.error(f"Failed to start motion watch for {camera_ref}: {e}")
        elif cmd_type == "set_motion_boxes":
            # Toggle true motion boxes overlay in the desktop camera widget (NOT motion watch).
            camera_ref = command.get("camera_id") or command.get("camera_ref") or command.get("cameraRef")
            if not camera_ref:
                logger.warning("set_motion_boxes missing camera_ref")
                return
            enabled = command.get("enabled")
            if enabled is None:
                enabled = True
            enabled = bool(enabled)
            w = self._find_camera_widget(str(camera_ref))
            if w is None:
                w = self.spawn_camera_widget(str(camera_ref))
            if w is None:
                return
            try:
                # Apply without relying on QAction toggles
                w.motion_boxes_enabled = enabled
                try:
                    w.gl_widget.set_overlay_settings(w.debug_overlay_enabled, w.motion_boxes_enabled)
                except Exception:
                    pass
                logger.info(f"Set motion boxes for {camera_ref}: {enabled}")
            except Exception as e:
                logger.error(f"Failed to set motion boxes for {camera_ref}: {e}")
        elif cmd_type == "open_ptz_widget":
            camera_ref = command.get("camera_id") or command.get("camera_ref") or command.get("cameraRef")
            if not camera_ref:
                logger.warning("open_ptz_widget missing camera_ref")
                return
            w = self._find_camera_widget(str(camera_ref))
            if w is None:
                w = self.spawn_camera_widget(str(camera_ref))
            if w is None:
                return
            try:
                # The floating PTZ controller is the only PTZ surface now;
                # legacy `undocked` flag is accepted but ignored.
                w.open_ptz_controller()
                logger.info(f"Opened PTZ controller for {camera_ref}")
            except Exception as e:
                logger.error(f"Failed to open PTZ widget for {camera_ref}: {e}")
        elif cmd_type == "open_audio_widget":
            camera_ref = command.get("camera_id") or command.get("camera_ref") or command.get("cameraRef")
            if not camera_ref:
                logger.warning("open_audio_widget missing camera_ref")
                return
            undocked = bool(command.get("undocked")) if command.get("undocked") is not None else False
            w = self._find_camera_widget(str(camera_ref))
            if w is None:
                w = self.spawn_camera_widget(str(camera_ref))
            if w is None:
                return
            try:
                if undocked:
                    w.toggle_audio_eq_undocked(True)
                else:
                    w.toggle_audio_eq_overlay(True)
                logger.info(f"Opened audio widget for {camera_ref} (undocked={undocked})")
            except Exception as e:
                logger.error(f"Failed to open audio widget for {camera_ref}: {e}")
        elif cmd_type == "open_depth_map_widget":
            camera_ref = command.get("camera_id") or command.get("camera_ref") or command.get("cameraRef")
            if not camera_ref:
                logger.warning("open_depth_map_widget missing camera_ref")
                return
            # Best-effort: map React depth-map params to desktop DepthAnything overlay config.
            enabled = command.get("enabled")
            color_scheme = command.get("colorScheme") or command.get("colormap")
            mode = command.get("mode")  # pointcloud|slam|overlay
            w = self._find_camera_widget(str(camera_ref))
            if w is None:
                w = self.spawn_camera_widget(str(camera_ref))
            if w is None:
                return
            try:
                if color_scheme:
                    try:
                        w.depth_overlay_config.colormap = str(color_scheme)
                    except Exception:
                        pass
                if mode:
                    m = str(mode).lower().strip()
                    if m in {"pointcloud", "slam"}:
                        try:
                            w.depth_overlay_config.colormap = "pointcloud"
                            # In pointcloud mode, show camera behind visualization (better UX)
                            w.depth_overlay_config.blackout_base = False
                        except Exception:
                            pass
                if enabled is None:
                    # Default behavior: enable depth overlay
                    enabled = True
                w.toggle_depth_overlay(bool(enabled))
                logger.info(f"Depth overlay set for {camera_ref}: enabled={bool(enabled)} colormap={getattr(w.depth_overlay_config,'colormap',None)}")
            except Exception as e:
                logger.error(f"Failed to open depth map for {camera_ref}: {e}")
        elif cmd_type == "start_motion_watch_all":
            self._ipc_start_motion_watch_all(command)
        elif cmd_type == "stop_motion_watch_all":
            self._ipc_stop_motion_watch_all()
        elif cmd_type == "toggle_recording":
            self._ipc_toggle_recording(command)
        elif cmd_type == "toggle_recording_all":
            self._ipc_toggle_recording_all(command)
        elif cmd_type == "set_motion_boxes_all":
            self._ipc_set_motion_boxes_all(command)
        elif cmd_type == "set_object_detection_all":
            self._ipc_set_object_detection_all(command)
        elif cmd_type == "set_object_detection":
            self._ipc_set_object_detection(command)
        elif cmd_type == "set_sensitivity_all":
            self._ipc_set_sensitivity_all(command)
        elif cmd_type == "layout_list":
            self._ipc_layout_list()
        elif cmd_type == "layout_load":
            self._ipc_layout_load(command)
        elif cmd_type == "layout_run":
            self._ipc_layout_run(command)
        elif cmd_type == "layout_stop":
            self._ipc_layout_stop(command)
        elif cmd_type == "arrange_grid":
            cols = command.get("cols")
            cols = int(cols) if cols is not None else None
            rows = command.get("rows")
            rows = int(rows) if rows is not None else None
            target = command.get("target", "camera")
            sort = command.get("sort")
            gap = int(command.get("gap", 2) or 2)
            mode = command.get("mode", "fit") or "fit"
            focus_ref = command.get("focus_ref")
            self.arrange_grid(cols=cols, rows=rows, target=target,
                              sort=sort, gap=gap, mode=mode,
                              focus_ref=focus_ref)
        elif cmd_type == "arrange_cascade":
            self.arrange_cascade(
                target=command.get("target", "camera"),
                sort=command.get("sort"),
            )
        elif cmd_type == "arrange_tile":
            self.arrange_tile(
                direction=command.get("direction", "horizontal"),
                target=command.get("target", "camera"),
                sort=command.get("sort"),
            )
        elif cmd_type == "arrange_fullscreen":
            self.arrange_fullscreen(
                camera_ref=command.get("camera_ref"),
                target=command.get("target", "camera"),
            )
        elif cmd_type == "snap_grid":
            self.arrange_snap_grid(target=command.get("target", "camera"))
        elif cmd_type == "toggle_snap":
            prefs = self._load_prefs()
            prefs["snap_enabled"] = not bool(prefs.get("snap_enabled", False))
            self._save_prefs(prefs)
            state = "ON" if prefs["snap_enabled"] else "OFF"
            self.tray_icon.showMessage(
                "Knoxnet VMS Beta", f"Snap to grid: {state}",
                QSystemTrayIcon.MessageIcon.Information,
            )
        elif cmd_type == "patrol_start":
            self.start_patrol(interval_sec=command.get("interval"))
        elif cmd_type == "patrol_stop":
            self.stop_patrol()
        elif cmd_type == "patrol_toggle":
            self.toggle_patrol()
        elif cmd_type == "minimize_all":
            self.minimize_all_widgets(target=command.get("target", "camera"))
        elif cmd_type == "restore_all":
            self.restore_all_widgets(target=command.get("target", "camera"))
        elif cmd_type == "close_all":
            self.close_all_widgets(target=command.get("target", "camera"))
        elif cmd_type == "show_all_cameras":
            self._show_all_cameras(command)
        elif cmd_type == "status":
            self.show_status()
        elif cmd_type == "shutdown":
            self.quit_app()
        else:
            logger.warning(f"Unknown command: {cmd_type}")

    # ---- Bulk IPC handlers ----

    def _ipc_camera_widgets(self):
        """Return all visible CameraWidget instances."""
        try:
            from desktop.widgets.camera import CameraWidget
        except Exception:
            return []
        return [w for w in list(self.active_widgets) if isinstance(w, CameraWidget) and w.isVisible()]

    def _ipc_start_motion_watch_all(self, command):
        duration = command.get("duration_sec")
        widgets = self._ipc_camera_widgets()
        if not widgets:
            logger.warning("start_motion_watch_all: no camera widgets found")
            return
        count = 0
        for w in widgets:
            try:
                if duration is not None:
                    try:
                        w.motion_watch_settings["duration_sec"] = int(duration)
                    except Exception:
                        pass
                w.start_motion_watch()
                count += 1
            except Exception as e:
                logger.error(f"Failed to start motion watch for {getattr(w, 'camera_id', '?')}: {e}")
        logger.info(f"Started motion watch on {count} camera(s)")

    def _ipc_stop_motion_watch_all(self):
        widgets = self._ipc_camera_widgets()
        count = 0
        for w in widgets:
            try:
                if getattr(w, "motion_watch_active", False):
                    w.stop_motion_watch("stopped via terminal (all)")
                    count += 1
            except Exception as e:
                logger.error(f"Failed to stop motion watch for {getattr(w, 'camera_id', '?')}: {e}")
        logger.info(f"Stopped motion watch on {count} camera(s)")

    def _ipc_toggle_recording(self, command):
        camera_ref = command.get("camera_ref") or command.get("camera_id") or ""
        enable = bool(command.get("record", True))
        widgets = self._ipc_camera_widgets()
        for w in widgets:
            cid = getattr(w, "camera_id", "")
            cname = getattr(w, "camera_name", "")
            if cid == camera_ref or cname == camera_ref or (cname and camera_ref.lower() in cname.lower()):
                try:
                    w._toggle_continuous_recording() if w._continuous_recording != enable else None
                except Exception as e:
                    logger.error("Recording toggle IPC error for %s: %s", camera_ref, e)
                return
        logger.warning("toggle_recording: no widget matched %s", camera_ref)

    def _ipc_toggle_recording_all(self, command):
        enable = bool(command.get("record", True))
        widgets = self._ipc_camera_widgets()
        count = 0
        for w in widgets:
            try:
                if getattr(w, '_continuous_recording', False) != enable:
                    w._toggle_continuous_recording()
                    count += 1
            except Exception as e:
                logger.error("Recording toggle error for %s: %s", getattr(w, 'camera_id', '?'), e)
        logger.info("Recording %s on %d camera(s)", "started" if enable else "stopped", count)

    def _ipc_set_motion_boxes_all(self, command):
        enabled = command.get("enabled")
        if enabled is None:
            enabled = True
        enabled = bool(enabled)
        widgets = self._ipc_camera_widgets()
        count = 0
        for w in widgets:
            try:
                w.motion_boxes_enabled = enabled
                try:
                    w.gl_widget.set_overlay_settings(w.debug_overlay_enabled, w.motion_boxes_enabled)
                except Exception:
                    pass
                count += 1
            except Exception as e:
                logger.error(f"Failed to set motion boxes for {getattr(w, 'camera_id', '?')}: {e}")
        logger.info(f"Set motion boxes ({enabled}) on {count} camera(s)")

    def _ipc_set_object_detection_all(self, command):
        enabled = command.get("enabled")
        if enabled is None:
            enabled = True
        enabled = bool(enabled)
        widgets = self._ipc_camera_widgets()
        count = 0
        for w in widgets:
            try:
                current = bool(getattr(w, "desktop_object_detection_enabled", False))
                if current != enabled:
                    w.toggle_object_detection()
                count += 1
            except Exception as e:
                logger.error(f"Failed to set detection for {getattr(w, 'camera_id', '?')}: {e}")
        logger.info(f"Set object detection ({enabled}) on {count} camera(s)")

    def _ipc_set_object_detection(self, command):
        camera_ref = command.get("camera_id") or command.get("camera_ref") or command.get("cameraRef")
        if not camera_ref:
            logger.warning("set_object_detection missing camera_ref")
            return
        enabled = command.get("enabled")
        if enabled is None:
            enabled = True
        enabled = bool(enabled)
        w = self._find_camera_widget(str(camera_ref))
        if w is None:
            w = self.spawn_camera_widget(str(camera_ref))
        if w is None:
            return
        try:
            current = bool(getattr(w, "desktop_object_detection_enabled", False))
            if current != enabled:
                w.toggle_object_detection()
            logger.info(f"Set object detection for {camera_ref}: {enabled}")
        except Exception as e:
            logger.error(f"Failed to set object detection for {camera_ref}: {e}")

    def _ipc_set_sensitivity_all(self, command):
        val = command.get("value")
        if val is None:
            logger.warning("set_sensitivity_all missing value")
            return
        val = max(1, min(100, int(val)))
        widgets = self._ipc_camera_widgets()
        count = 0
        for w in widgets:
            try:
                gl = getattr(w, "gl_widget", None)
                if gl:
                    ms = getattr(gl, "motion_settings", {})
                    ms["sensitivity"] = val
                    gl.motion_settings = ms
                    count += 1
            except Exception as e:
                logger.error(f"Failed to set sensitivity for {getattr(w, 'camera_id', '?')}: {e}")
        logger.info(f"Set motion sensitivity to {val} on {count} camera(s)")

    def _resolve_layout_ref(self, ref: str):
        """Find a layout by id or name (case-insensitive)."""
        if not ref:
            return None
        layouts = self._list_layouts_v2()
        ref_l = ref.strip().lower()
        for l in layouts:
            if l.id.lower() == ref_l or l.name.lower() == ref_l:
                return l
        for l in layouts:
            if ref_l in l.name.lower():
                return l
        return None

    def _ipc_layout_list(self):
        layouts = self._list_layouts_v2()
        if not layouts:
            self._terminal_broadcast("No saved layouts.")
            return
        lines = []
        for l in layouts:
            state = self._layout_state(l.id)
            if state.get("hidden"):
                tag = "hidden"
            elif state.get("paused"):
                tag = "paused"
            elif state.get("current"):
                tag = "current"
            elif state.get("running"):
                tag = "running"
            else:
                tag = "stopped"
            lines.append(f"  {l.name}  [{tag}]")
        self._terminal_broadcast("Layouts:\n" + "\n".join(lines))

    def _ipc_layout_load(self, command):
        ref = str(command.get("layout_ref") or "").strip()
        layout = self._resolve_layout_ref(ref)
        if not layout:
            self._terminal_broadcast(f"Layout '{ref}' not found.")
            return
        self.load_layout(layout.id)
        self._terminal_broadcast(f"Loaded layout '{layout.name}'.")

    def _ipc_layout_run(self, command):
        ref = str(command.get("layout_ref") or "").strip()
        layout = self._resolve_layout_ref(ref)
        if not layout:
            self._terminal_broadcast(f"Layout '{ref}' not found.")
            return
        self._layout_start_in_background(layout.id)
        self._terminal_broadcast(f"Started layout '{layout.name}' in background.")

    def _ipc_layout_stop(self, command):
        ref = str(command.get("layout_ref") or "").strip()
        layout = self._resolve_layout_ref(ref)
        if not layout:
            self._terminal_broadcast(f"Layout '{ref}' not found.")
            return
        self._layout_stop(layout.id)
        self._terminal_broadcast(f"Stopped layout '{layout.name}'.")

    def _terminal_broadcast(self, message: str):
        """Best-effort send a system message to all open terminal widgets."""
        try:
            from desktop.widgets.terminal import TerminalWidget
            for tw in list(TerminalWidget._instances):
                try:
                    tw._add_system(message)
                except Exception:
                    pass
        except Exception:
            pass

    def spawn_web_widget(self, title, url):
        logger.info(f"Spawning web widget: {title} -> {url}")
        # WebWidget has been removed due to stability issues on this platform
        QMessageBox.information(None, title, f"Please view {title} in your browser at:\nhttp://localhost:5000{url}")

    # ==================== Docker toggles (desktop-light) ====================

    def _repo_root(self) -> str:
        try:
            return str(Path(__file__).resolve().parents[1])
        except Exception:
            return str(Path(".").resolve())

    def _run_docker_compose_async(self, args: list[str], success_msg: str, fail_msg: str):
        def _worker():
            try:
                proc = subprocess.run(
                    ["docker", "compose", *args],
                    cwd=self._repo_root(),
                    capture_output=True,
                    text=True,
                    check=False,
                )
                if proc.returncode == 0:
                    self.tray_icon.showMessage("Knoxnet VMS Beta", success_msg, QSystemTrayIcon.MessageIcon.Information)
                else:
                    details = (proc.stderr or proc.stdout or "").strip()
                    msg = fail_msg + (f"\n{details}" if details else "")
                    self.tray_icon.showMessage("Knoxnet VMS Beta", msg, QSystemTrayIcon.MessageIcon.Warning)
            except Exception as e:
                self.tray_icon.showMessage("Knoxnet VMS Beta", f"{fail_msg}\n{e}", QSystemTrayIcon.MessageIcon.Warning)

        threading.Thread(target=_worker, daemon=True).start()

    def start_backend_docker(self):
        # Starts the backend only (requires docker compose profiles in docker-compose.yml).
        self._run_docker_compose_async(
            ["--profile", "backend", "up", "-d"],
            success_msg="Web UI backend starting…",
            fail_msg="Failed to start Web UI backend.",
        )

    def stop_backend_docker(self):
        self._run_docker_compose_async(
            ["stop", "knoxnet"],
            success_msg="Web UI backend stopped.",
            fail_msg="Failed to stop Web UI backend.",
        )

    def start_ai_docker(self):
        # Starts AI services only (vision + llm).
        self._run_docker_compose_async(
            ["--profile", "ai", "up", "-d"],
            success_msg="AI services starting…",
            fail_msg="Failed to start AI services.",
        )

    def stop_ai_docker(self):
        self._run_docker_compose_async(
            ["stop", "llm-local", "vision-local"],
            success_msg="AI services stopped.",
            fail_msg="Failed to stop AI services.",
        )

    def spawn_terminal_widget(self, checked: bool = False, title: str = "Terminal"):
        # QAction.triggered passes a bool; ignore it and use the provided/default title.
        if isinstance(checked, str):
            title = checked
        logger.info(f"Spawning terminal widget: {title}")
        try:
            from desktop.widgets.terminal import TerminalWidget
        except Exception as e:
            logger.error(f"Failed to load TerminalWidget: {e}")
            QMessageBox.warning(None, "Terminal", f"Unable to open terminal widget: {e}")
            return

        widget = TerminalWidget(title=title)
        self._register_widget(widget)
        widget.show()

    # ----------------------------------------------------------------
    #  Smart Layout Engine
    # ----------------------------------------------------------------

    def _collect_target_widgets(self, target: str = "camera") -> list:
        """Return visible widgets matching *target* type."""
        from desktop.widgets.camera import CameraWidget
        try:
            from desktop.widgets.terminal import TerminalWidget
        except Exception:
            TerminalWidget = None

        widgets = []
        for w in list(self.active_widgets):
            if not w.isVisible():
                continue
            if target == "camera" and isinstance(w, CameraWidget):
                widgets.append(w)
            elif target == "terminal" and TerminalWidget and isinstance(w, TerminalWidget):
                widgets.append(w)
            elif target == "all":
                widgets.append(w)
        return widgets

    @staticmethod
    def _sort_widgets(widgets: list, sort_key: str | None) -> list:
        """Sort *widgets* by criterion.  Supports name, id, status and
        an optional ``desc`` suffix for reverse order."""
        if not sort_key or not widgets:
            return list(widgets)
        reverse = False
        key = sort_key.lower().strip()
        if key.endswith(" desc") or key.endswith("-desc"):
            reverse = True
            key = key.rsplit(" ", 1)[0].rsplit("-", 1)[0].strip()

        if key in ("name", "title"):
            return sorted(widgets,
                          key=lambda w: (getattr(w, "camera_name", "") or w.windowTitle() or "").lower(),
                          reverse=reverse)
        if key == "id":
            return sorted(widgets,
                          key=lambda w: str(getattr(w, "camera_id", "") or "").lower(),
                          reverse=reverse)
        if key == "status":
            return sorted(widgets,
                          key=lambda w: 0 if getattr(w, "running", False) else 1,
                          reverse=reverse)
        return list(widgets)

    @staticmethod
    def _optimal_grid_dims(count: int, geo, target_ar: float = 16 / 9) -> tuple:
        """Pick the (cols, rows) that maximise screen utilisation for
        *count* widgets whose average aspect ratio is *target_ar*."""
        if count <= 0:
            return (1, 1)
        if count == 1:
            return (1, 1)

        screen_w = geo.width()
        screen_h = geo.height()
        best_cols = 1
        best_score = -1.0

        for cols in range(1, count + 1):
            rows = math.ceil(count / cols)
            cell_w = screen_w / cols
            cell_h = screen_h / rows
            cell_ar = cell_w / cell_h if cell_h > 0 else target_ar

            if cell_ar > target_ar:
                widget_w = cell_h * target_ar
                widget_h = cell_h
            else:
                widget_w = cell_w
                widget_h = cell_w / target_ar if target_ar > 0 else cell_h

            video_area = widget_w * widget_h * count
            screen_area = screen_w * screen_h
            utilization = video_area / screen_area if screen_area > 0 else 0

            total_cells = cols * rows
            empty_frac = (total_cells - count) / total_cells
            score = utilization * (1.0 - 0.3 * empty_frac)

            if score > best_score:
                best_score = score
                best_cols = cols

        return (best_cols, math.ceil(count / best_cols))

    @staticmethod
    def _fit_to_cell(cell_w: int, cell_h: int, ar: float) -> tuple:
        """Return (w, h) that fits inside *cell_w*×*cell_h* while
        preserving aspect ratio *ar*."""
        if ar <= 0:
            ar = 16 / 9
        cell_ar = cell_w / cell_h if cell_h > 0 else ar

        if cell_ar > ar:
            fit_h = cell_h
            fit_w = int(cell_h * ar)
        else:
            fit_w = cell_w
            fit_h = int(cell_w / ar)

        return (max(160, fit_w), max(90, fit_h))

    # ---- Primary grid arranger ----

    def arrange_grid(self, cols: int | None = None, rows: int | None = None,
                     target: str = "camera", sort: str | None = None,
                     gap: int = 2, mode: str = "fit",
                     focus_ref: str | None = None):
        """Smart, aspect-ratio-aware grid layout for desktop widgets.

        Modes
        -----
        fit       Preserve widget aspect ratios, centre within cells (default).
        fill      Stretch widgets to fill each cell (ignores AR).
        seamless  Tight packing with AR preservation and zero gap.
        focus     One camera large on the left, rest stacked on the right.
        """
        widgets = self._collect_target_widgets(target)
        if not widgets:
            self.tray_icon.showMessage(
                "Knoxnet VMS Beta",
                f"No visible {target} widgets to arrange.",
                QSystemTrayIcon.MessageIcon.Warning,
            )
            return

        if sort:
            widgets = self._sort_widgets(widgets, sort)

        screen = QGuiApplication.primaryScreen()
        geo = screen.availableGeometry()
        count = len(widgets)

        if mode == "focus":
            self._arrange_focus_layout(widgets, geo, focus_ref, gap)
            return

        ars = [getattr(w, "aspect_ratio", 16 / 9) for w in widgets]
        avg_ar = sum(ars) / len(ars) if ars else 16 / 9

        if cols is None:
            cols, rows = self._optimal_grid_dims(count, geo, avg_ar)
        else:
            cols = max(1, cols)
            if rows is None:
                rows = math.ceil(count / cols)
            else:
                rows = max(1, rows)

        effective_gap = 0 if mode == "seamless" else gap
        usable_w = geo.width() - effective_gap * (cols + 1)
        usable_h = geo.height() - effective_gap * (rows + 1)
        cell_w = max(160, usable_w // cols)
        cell_h = max(90, usable_h // rows)

        for idx, w in enumerate(widgets):
            if idx >= cols * rows:
                w.hide()
                continue
            r = idx // cols
            c = idx % cols

            if mode == "fill":
                fit_w, fit_h = cell_w, cell_h
                offset_x = offset_y = 0
            else:
                ar = getattr(w, "aspect_ratio", 16 / 9)
                fit_w, fit_h = self._fit_to_cell(cell_w, cell_h, ar)
                offset_x = (cell_w - fit_w) // 2
                offset_y = (cell_h - fit_h) // 2

            x = geo.x() + effective_gap + c * (cell_w + effective_gap) + offset_x
            y = geo.y() + effective_gap + r * (cell_h + effective_gap) + offset_y
            w.setGeometry(x, y, fit_w, fit_h)
            w.show()

        mode_lbl = f" ({mode})" if mode != "fit" else ""
        sort_lbl = f", sorted by {sort}" if sort else ""
        self.tray_icon.showMessage(
            "Knoxnet VMS Beta",
            f"Grid {cols}\u00d7{rows}: {count} {target} widget(s){mode_lbl}{sort_lbl}",
            QSystemTrayIcon.MessageIcon.Information,
        )

    # ---- Focus layout ----

    def _arrange_focus_layout(self, widgets: list, geo, focus_ref: str | None, gap: int = 4):
        """One camera occupies ~70 % of the screen; the rest stack in a sidebar."""
        focus_widget = None
        others = []
        ref = (focus_ref or "").lower()

        for w in widgets:
            if ref and focus_widget is None:
                cid = str(getattr(w, "camera_id", "") or "").lower()
                cname = str(getattr(w, "camera_name", "") or "").lower()
                title = (w.windowTitle() or "").lower()
                if ref in cid or ref in cname or ref in title:
                    focus_widget = w
                    continue
            others.append(w)

        if focus_widget is None:
            focus_widget = widgets[0]
            others = widgets[1:]

        if not others:
            ar = getattr(focus_widget, "aspect_ratio", 16 / 9)
            fw, fh = self._fit_to_cell(geo.width(), geo.height(), ar)
            focus_widget.setGeometry(
                geo.x() + (geo.width() - fw) // 2,
                geo.y() + (geo.height() - fh) // 2,
                fw, fh,
            )
            focus_widget.show()
            return

        main_w = int(geo.width() * 0.70) - gap * 2
        sidebar_w = geo.width() - main_w - gap * 3
        main_h = geo.height() - gap * 2

        ar = getattr(focus_widget, "aspect_ratio", 16 / 9)
        fw, fh = self._fit_to_cell(main_w, main_h, ar)
        focus_widget.setGeometry(
            geo.x() + gap + (main_w - fw) // 2,
            geo.y() + gap + (main_h - fh) // 2,
            fw, fh,
        )
        focus_widget.show()

        sidebar_x = geo.x() + main_w + gap * 2
        n = len(others)
        cell_h = max(80, (geo.height() - gap * (n + 1)) // n)
        for idx, w in enumerate(others):
            ar = getattr(w, "aspect_ratio", 16 / 9)
            sw, sh = self._fit_to_cell(sidebar_w, cell_h, ar)
            w.setGeometry(
                sidebar_x + (sidebar_w - sw) // 2,
                geo.y() + gap + idx * (cell_h + gap) + (cell_h - sh) // 2,
                sw, sh,
            )
            w.show()

        self.tray_icon.showMessage(
            "Knoxnet VMS Beta",
            f"Focus: {getattr(focus_widget, 'camera_name', 'camera')} + {n} sidebar",
            QSystemTrayIcon.MessageIcon.Information,
        )

    # ---- Cascade ----

    def arrange_cascade(self, target: str = "camera", sort: str | None = None):
        """Cascade / stagger windows diagonally across the screen."""
        widgets = self._collect_target_widgets(target)
        if not widgets:
            self.tray_icon.showMessage(
                "Knoxnet VMS Beta", f"No visible {target} widgets.",
                QSystemTrayIcon.MessageIcon.Warning,
            )
            return
        if sort:
            widgets = self._sort_widgets(widgets, sort)

        screen = QGuiApplication.primaryScreen()
        geo = screen.availableGeometry()
        offset_step = 32
        base_w = min(640, int(geo.width() * 0.5))
        base_h = min(360, int(geo.height() * 0.5))

        for idx, w in enumerate(widgets):
            ar = getattr(w, "aspect_ratio", 16 / 9)
            fw, fh = self._fit_to_cell(base_w, base_h, ar)
            x = geo.x() + idx * offset_step
            y = geo.y() + idx * offset_step
            if x + fw > geo.x() + geo.width():
                x = geo.x() + (idx * offset_step) % max(1, geo.width() - fw)
            if y + fh > geo.y() + geo.height():
                y = geo.y() + (idx * offset_step) % max(1, geo.height() - fh)
            w.setGeometry(x, y, fw, fh)
            w.show()
            w.raise_()

        self.tray_icon.showMessage(
            "Knoxnet VMS Beta", f"Cascaded {len(widgets)} widget(s).",
            QSystemTrayIcon.MessageIcon.Information,
        )

    # ---- Tile (single row / column) ----

    def arrange_tile(self, direction: str = "horizontal",
                     target: str = "camera", sort: str | None = None,
                     gap: int = 2):
        """Tile widgets in a single row (horizontal) or column (vertical)."""
        widgets = self._collect_target_widgets(target)
        if not widgets:
            self.tray_icon.showMessage(
                "Knoxnet VMS Beta", f"No visible {target} widgets.",
                QSystemTrayIcon.MessageIcon.Warning,
            )
            return
        if sort:
            widgets = self._sort_widgets(widgets, sort)

        screen = QGuiApplication.primaryScreen()
        geo = screen.availableGeometry()
        count = len(widgets)

        if direction == "vertical":
            cell_w = geo.width() - gap * 2
            cell_h = max(80, (geo.height() - gap * (count + 1)) // count)
            for idx, w in enumerate(widgets):
                ar = getattr(w, "aspect_ratio", 16 / 9)
                fw, fh = self._fit_to_cell(cell_w, cell_h, ar)
                w.setGeometry(
                    geo.x() + gap + (cell_w - fw) // 2,
                    geo.y() + gap + idx * (cell_h + gap) + (cell_h - fh) // 2,
                    fw, fh,
                )
                w.show()
        else:
            cell_w = max(160, (geo.width() - gap * (count + 1)) // count)
            cell_h = geo.height() - gap * 2
            for idx, w in enumerate(widgets):
                ar = getattr(w, "aspect_ratio", 16 / 9)
                fw, fh = self._fit_to_cell(cell_w, cell_h, ar)
                w.setGeometry(
                    geo.x() + gap + idx * (cell_w + gap) + (cell_w - fw) // 2,
                    geo.y() + gap + (cell_h - fh) // 2,
                    fw, fh,
                )
                w.show()

        self.tray_icon.showMessage(
            "Knoxnet VMS Beta", f"Tiled {count} widget(s) {direction}ly.",
            QSystemTrayIcon.MessageIcon.Information,
        )

    # ---- Fullscreen (single widget) ----

    def arrange_fullscreen(self, camera_ref: str | None = None,
                           target: str = "camera"):
        """Maximise a single widget, minimising the rest."""
        widgets = self._collect_target_widgets(target)
        if not widgets:
            self.tray_icon.showMessage(
                "Knoxnet VMS Beta", "No visible widgets.",
                QSystemTrayIcon.MessageIcon.Warning,
            )
            return

        widget = widgets[0]
        if camera_ref:
            ref = camera_ref.lower()
            for w in widgets:
                cid = str(getattr(w, "camera_id", "") or "").lower()
                cname = str(getattr(w, "camera_name", "") or "").lower()
                if ref in cid or ref in cname:
                    widget = w
                    break

        screen = QGuiApplication.primaryScreen()
        geo = screen.availableGeometry()
        ar = getattr(widget, "aspect_ratio", 16 / 9)
        fw, fh = self._fit_to_cell(geo.width(), geo.height(), ar)
        widget.setGeometry(
            geo.x() + (geo.width() - fw) // 2,
            geo.y() + (geo.height() - fh) // 2,
            fw, fh,
        )
        widget.show()
        widget.raise_()

        for w in widgets:
            if w is not widget:
                w.showMinimized()

        name = getattr(widget, "camera_name", camera_ref or "widget")
        self.tray_icon.showMessage(
            "Knoxnet VMS Beta", f"Fullscreen: {name}",
            QSystemTrayIcon.MessageIcon.Information,
        )

    # ---- Camera Patrol Mode ----

    def toggle_patrol(self):
        if self._patrol_active:
            self.stop_patrol()
        else:
            self.start_patrol()

    def start_patrol(self, interval_sec: float | None = None):
        """Cycle through visible camera widgets, raising each on a timer.

        Respects ``patrol_order`` from prefs: only cameras in that list are
        included, in the stored order.  If the list is empty, all visible
        cameras are used.
        """
        try:
            prefs = self._load_prefs()
        except Exception:
            prefs = {}
        if interval_sec is None:
            interval_sec = float(prefs.get("patrol_interval_sec", 10))
        interval_sec = max(2.0, min(300.0, float(interval_sec)))

        all_widgets = self._collect_target_widgets("camera")
        order = list(prefs.get("patrol_order") or [])

        if order:
            by_id = {}
            for w in all_widgets:
                cid = str(getattr(w, "camera_id", "") or "")
                if cid:
                    by_id[cid] = w
            widgets = [by_id[cid] for cid in order if cid in by_id]
        else:
            widgets = list(all_widgets)

        if len(widgets) < 2:
            self.tray_icon.showMessage(
                "Knoxnet VMS Beta", "Need at least 2 cameras for patrol.",
                QSystemTrayIcon.MessageIcon.Warning,
            )
            return

        self._patrol_widgets = widgets
        self._patrol_index = 0
        self._patrol_active = True

        self._patrol_show_current()
        self._patrol_timer.start(int(interval_sec * 1000))

        self.tray_icon.showMessage(
            "Knoxnet VMS Beta",
            f"Patrol started: {len(widgets)} cameras, {interval_sec:.0f}s interval.",
            QSystemTrayIcon.MessageIcon.Information,
        )

    def stop_patrol(self):
        self._patrol_timer.stop()
        self._patrol_active = False
        self._patrol_widgets.clear()
        self._patrol_index = 0
        self.tray_icon.showMessage(
            "Knoxnet VMS Beta", "Patrol stopped.",
            QSystemTrayIcon.MessageIcon.Information,
        )

    def _patrol_tick(self):
        if not self._patrol_active or not self._patrol_widgets:
            self.stop_patrol()
            return
        # Prune destroyed widgets
        self._patrol_widgets = [w for w in self._patrol_widgets if w is not None and not getattr(w, '_closed', False)]
        if len(self._patrol_widgets) < 2:
            self.stop_patrol()
            return
        self._patrol_index = (self._patrol_index + 1) % len(self._patrol_widgets)
        self._patrol_show_current()

    def _patrol_show_current(self):
        """Bring the current patrol widget to front without altering its geometry."""
        if not self._patrol_widgets:
            return
        idx = self._patrol_index % len(self._patrol_widgets)
        widget = self._patrol_widgets[idx]
        widget.show()
        try:
            if hasattr(widget, "windowState") and (widget.windowState() & Qt.WindowState.WindowMinimized):
                widget.showNormal()
        except Exception:
            pass
        widget.raise_()
        widget.activateWindow()

    def patrol_bring_camera_to_front(self, camera_id: str):
        """Interrupt patrol to bring a specific camera to the front (e.g. from shape trigger)."""
        if not self._patrol_active:
            return
        camera_id = str(camera_id or "").strip().lower()
        if not camera_id:
            return
        for i, w in enumerate(self._patrol_widgets):
            cid = str(getattr(w, "camera_id", "") or "").lower()
            cname = str(getattr(w, "camera_name", "") or "").lower()
            if camera_id in cid or camera_id in cname:
                self._patrol_index = i
                self._patrol_show_current()
                self._patrol_timer.start()
                break

    # ---- Playback Sync ----

    def sync_playback(self, source_camera_id: str, target_camera_ids: list[str],
                      timestamp: float, speed: float = 1.0) -> str:
        """Open (or reuse) camera widgets for *target_camera_ids* and start
        playback at *timestamp*, synced together with the source camera.

        Returns a sync group id that can be used to unsync later.
        """
        from desktop.widgets.camera import CameraWidget

        self._playback_sync_counter += 1
        group_id = f"sync_{self._playback_sync_counter}"
        members: set[str] = {source_camera_id}

        for cam_id in target_camera_ids:
            resolved = self._resolve_camera_id(cam_id)
            if not resolved:
                continue
            members.add(resolved)
            widget = self._find_camera_widget(resolved)
            if widget is None:
                widget = self.spawn_camera_widget(resolved)
            if widget is None:
                continue
            widget.start_synced_playback(timestamp, speed)

        self._playback_sync_groups[group_id] = members

        # Tag every member widget with the group and update sync indicator
        for w in list(self.active_widgets):
            if isinstance(w, CameraWidget) and getattr(w, 'camera_id', '') in members:
                w._sync_group_id = group_id
                po = getattr(w, 'playback_overlay', None)
                if po:
                    po._synced = True
                    po.update()

        return group_id

    def broadcast_playback_sync(self, group_id: str, source_camera_id: str,
                                action: str, **kwargs):
        """Called by a camera widget to broadcast a seek/speed/stop to its sync group.

        Sets ``_sync_broadcasting`` on each receiver BEFORE calling the
        action so that the receiver's own signal emissions don't trigger
        another broadcast (which would cause infinite recursion).
        """
        members = self._playback_sync_groups.get(group_id)
        if not members:
            return
        from desktop.widgets.camera import CameraWidget
        for w in list(self.active_widgets):
            if not isinstance(w, CameraWidget):
                continue
            cid = getattr(w, 'camera_id', '')
            if cid not in members or cid == source_camera_id:
                continue
            po = getattr(w, 'playback_overlay', None)
            if po is None:
                continue
            w._sync_broadcasting = True
            try:
                if action == 'seek':
                    po._engine.seek(kwargs.get('ts', 0))
                    po.update()
                elif action == 'speed':
                    po._engine.speed = kwargs.get('speed', 1.0)
                    po._sync_audio_to_speed()
                    po.update()
                elif action == 'play':
                    if not po._engine.playing:
                        po._engine.play()
                        po._tick_timer.start()
                        po._replay_timer.start()
                        po.update()
                elif action == 'pause':
                    if po._engine.playing:
                        po._engine.pause()
                        po.update()
                elif action == 'stop':
                    po.stop_playback()
                    po.hide()
                    po.playback_closed.emit()
                elif action == 'go_live':
                    po._go_live()
                elif action == 'step':
                    po._engine.step_frame(forward=kwargs.get('forward', True))
                    po.update()
            except Exception:
                continue
            finally:
                w._sync_broadcasting = False

    def unsync_playback(self, group_id: str):
        """Remove a sync group and clear group tags on member widgets."""
        members = self._playback_sync_groups.pop(group_id, set())
        from desktop.widgets.camera import CameraWidget
        for w in list(self.active_widgets):
            if isinstance(w, CameraWidget) and getattr(w, '_sync_group_id', '') == group_id:
                w._sync_group_id = None
                po = getattr(w, 'playback_overlay', None)
                if po:
                    po._synced = False
                    po.update()

    def get_all_camera_ids_and_names(self) -> list[tuple[str, str]]:
        """Return [(camera_id, camera_name), ...] for all known cameras."""
        result = []
        try:
            if self.camera_manager:
                for cam_id, cfg in self.camera_manager.cameras.items():
                    name = getattr(cfg, 'name', '') or cam_id
                    result.append((cam_id, name))
        except Exception:
            pass
        return result

    # ---- Grid / Snap helpers ----

    def get_snap_config(self) -> dict:
        try:
            prefs = self._load_prefs()
        except Exception:
            prefs = {}
        return {
            "enabled": bool(prefs.get("snap_enabled", True)),
            "grid_size": max(4, int(prefs.get("snap_grid_size", 20))),
            "edge_threshold": max(1, int(prefs.get("snap_edge_threshold", 24))),
            "show_guides": bool(prefs.get("snap_show_guides", True)),
        }

    def _collect_snap_edges(self, widget):
        """Gather horizontal (x) and vertical (y) edges from screen bounds and sibling widgets."""
        screen = QGuiApplication.primaryScreen()
        sg = screen.availableGeometry()
        h_edges = {sg.x(), sg.x() + sg.width()}
        v_edges = {sg.y(), sg.y() + sg.height()}
        from desktop.widgets.camera import CameraWidget
        for aw in list(self.active_widgets):
            if aw is widget or not aw.isVisible():
                continue
            if not isinstance(aw, CameraWidget):
                continue
            ag = aw.geometry()
            h_edges.add(ag.x())
            h_edges.add(ag.x() + ag.width())
            v_edges.add(ag.y())
            v_edges.add(ag.y() + ag.height())
        return h_edges, v_edges

    def compute_snap_info(self, widget, geo) -> dict:
        """Compute snapped geometry and guide-line positions.

        The snap is "magnetic": when a widget edge is near a neighbor edge,
        the widget slides flush against it.  When both left *and* right (or
        top *and* bottom) edges are each near a different neighbor, the
        widget resizes to fill the gap exactly -- so the user can just drop
        a camera roughly between two others and it locks in seamlessly.

        Returns ``{"geo": QRect, "h_guides": [x, ...], "v_guides": [y, ...]}``.
        """
        from PySide6.QtCore import QRect
        empty = {"geo": geo, "h_guides": [], "v_guides": []}
        cfg = self.get_snap_config()
        if not cfg["enabled"]:
            return empty

        et = cfg["edge_threshold"]
        x, y, w, h = geo.x(), geo.y(), geo.width(), geo.height()
        h_edges, v_edges = self._collect_snap_edges(widget)

        def _nearest(val, edges):
            """Return (snapped_value, edge_value, distance) for the closest edge."""
            best_e = None
            best_d = float("inf")
            for e in edges:
                d = abs(val - e)
                if d < best_d:
                    best_d = d
                    best_e = e
            if best_e is not None and best_d <= et:
                return best_e, best_e, best_d
            return val, None, float("inf")

        # -- Horizontal axis (x / width) --
        sl, el, dl = _nearest(x, h_edges)
        sr, er, dr = _nearest(x + w, h_edges)

        h_guides = []
        new_x, new_w = x, w

        if el is not None and er is not None and el != er:
            # Both left and right edges near neighbors -- resize to fill gap
            new_x = el
            new_w = er - el
            h_guides.extend([el, er])
        elif el is not None and er is not None:
            # Both edges near the same neighbor edge -- just snap position
            if dl <= dr:
                new_x = el
                h_guides.append(el)
            else:
                new_x = er - w
                h_guides.append(er)
        elif el is not None:
            new_x = el
            h_guides.append(el)
        elif er is not None:
            new_x = er - w
            h_guides.append(er)

        # -- Vertical axis (y / height) --
        st, et_v, dt = _nearest(y, v_edges)
        sb, eb, db = _nearest(y + h, v_edges)

        v_guides = []
        new_y, new_h = y, h

        if et_v is not None and eb is not None and et_v != eb:
            new_y = et_v
            new_h = eb - et_v
            v_guides.extend([et_v, eb])
        elif et_v is not None and eb is not None:
            if dt <= db:
                new_y = et_v
                v_guides.append(et_v)
            else:
                new_y = eb - h
                v_guides.append(eb)
        elif et_v is not None:
            new_y = et_v
            v_guides.append(et_v)
        elif eb is not None:
            new_y = eb - h
            v_guides.append(eb)

        # Enforce a minimum size so we never collapse
        new_w = max(160, new_w)
        new_h = max(90, new_h)

        return {"geo": QRect(new_x, new_y, new_w, new_h), "h_guides": h_guides, "v_guides": v_guides}

    def snap_geometry(self, widget, geo) -> 'QRect':
        """Convenience wrapper -- returns only the snapped QRect."""
        return self.compute_snap_info(widget, geo)["geo"]

    # ── Snap guide overlay ──

    def _ensure_snap_overlay(self):
        from desktop.widgets.snap_overlay import SnapGuideOverlay
        if self._snap_overlay is None:
            self._snap_overlay = SnapGuideOverlay()
        return self._snap_overlay

    def show_snap_guides(self, h_guides: list, v_guides: list):
        cfg = self.get_snap_config()
        if not cfg.get("show_guides", True):
            return
        screen = QGuiApplication.primaryScreen()
        sg = screen.availableGeometry()
        overlay = self._ensure_snap_overlay()
        overlay.set_guides(h_guides, v_guides, sg)

    def hide_snap_guides(self):
        if self._snap_overlay is not None:
            self._snap_overlay.set_guides([], [], None)

    def arrange_snap_grid(self, target: str = "camera"):
        """Auto-arrange widgets into a seamless edge-to-edge grid, snapped to grid lines."""
        self.arrange_grid(target=target, mode="seamless", gap=0)

    # ---- Minimize / Restore / Close ----

    def minimize_all_widgets(self, target: str = "camera"):
        """Minimise all visible widgets of the given type."""
        widgets = self._collect_target_widgets(target)
        for w in widgets:
            w.showMinimized()
        if widgets:
            self.tray_icon.showMessage(
                "Knoxnet VMS Beta", f"Minimized {len(widgets)} widget(s).",
                QSystemTrayIcon.MessageIcon.Information,
            )

    def restore_all_widgets(self, target: str = "camera"):
        """Restore minimised widgets then auto-grid them."""
        from desktop.widgets.camera import CameraWidget
        try:
            from desktop.widgets.terminal import TerminalWidget
        except Exception:
            TerminalWidget = None

        restored = []
        for w in list(self.active_widgets):
            if target == "camera" and isinstance(w, CameraWidget):
                w.showNormal()
                restored.append(w)
            elif target == "terminal" and TerminalWidget and isinstance(w, TerminalWidget):
                w.showNormal()
                restored.append(w)
            elif target == "all":
                w.showNormal()
                restored.append(w)

        if restored:
            self.arrange_grid(target=target)
            self.tray_icon.showMessage(
                "Knoxnet VMS Beta", f"Restored {len(restored)} widget(s).",
                QSystemTrayIcon.MessageIcon.Information,
            )

    def close_all_widgets(self, target: str = "camera"):
        """Close all widgets of the given type."""
        widgets = self._collect_target_widgets(target)
        count = len(widgets)
        for w in list(widgets):
            w.close()
        if count:
            self.tray_icon.showMessage(
                "Knoxnet VMS Beta", f"Closed {count} widget(s).",
                QSystemTrayIcon.MessageIcon.Information,
            )

    def _show_all_cameras(self, command: dict):
        """Spawn widgets for every camera and tile them in a VMS-style grid.
        Reuses already-open widgets so duplicate windows aren't created."""
        camera_ids = command.get("camera_ids", [])
        cols = int(command.get("cols", 2) or 2)
        rows_hint = command.get("rows")
        rows = int(rows_hint) if rows_hint is not None else None

        if not camera_ids:
            self.tray_icon.showMessage("Knoxnet VMS Beta", "No cameras to show.", QSystemTrayIcon.MessageIcon.Warning)
            return

        from desktop.widgets.camera import CameraWidget

        widgets: list = []
        for cam_ref in camera_ids:
            existing = self._find_camera_widget(str(cam_ref))
            if existing is not None:
                widgets.append(existing)
            else:
                w = self.spawn_camera_widget(str(cam_ref))
                if w is not None:
                    widgets.append(w)

        if not widgets:
            self.tray_icon.showMessage("Knoxnet VMS Beta", "Could not open any cameras.", QSystemTrayIcon.MessageIcon.Warning)
            return

        count = len(widgets)

        screen = QGuiApplication.primaryScreen()
        geo = screen.availableGeometry()
        ars = [getattr(w, "aspect_ratio", 16 / 9) for w in widgets]
        avg_ar = sum(ars) / len(ars) if ars else 16 / 9

        if cols is None or cols < 1:
            cols, rows = self._optimal_grid_dims(count, geo, avg_ar)
        elif rows is None:
            rows = math.ceil(count / cols)
        else:
            rows = max(1, rows)

        gap = 2
        usable_w = geo.width() - gap * (cols + 1)
        usable_h = geo.height() - gap * (rows + 1)
        cell_w = max(160, usable_w // cols)
        cell_h = max(90, usable_h // rows)

        for idx, w in enumerate(widgets):
            r = idx // cols
            c = idx % cols
            ar = getattr(w, "aspect_ratio", 16 / 9)
            fw, fh = self._fit_to_cell(cell_w, cell_h, ar)
            offset_x = (cell_w - fw) // 2
            offset_y = (cell_h - fh) // 2
            x = geo.x() + gap + c * (cell_w + gap) + offset_x
            y = geo.y() + gap + r * (cell_h + gap) + offset_y
            w.setGeometry(x, y, fw, fh)
            w.show()

        self.tray_icon.showMessage(
            "Knoxnet VMS Beta",
            f"Showing all {count} cameras in {cols}\u00d7{rows} grid",
            QSystemTrayIcon.MessageIcon.Information,
        )
        logger.info(f"show_all_cameras: displayed {count} cameras in {cols}x{rows} grid")

    def _sync_widget_quality_to_config(self, widget) -> None:
        """Push the widget's stream_quality into CameraConfig so that
        acquire_camera / _start_rtsp_stream picks the correct URL."""
        try:
            from core.camera_manager import StreamQuality
            cam_id = str(getattr(widget, "camera_id", ""))
            cfg = self.camera_manager.cameras.get(cam_id) if self.camera_manager else None
            if cfg is None:
                return

            priority = getattr(cfg, "stream_priority", None) or ("sub" if cfg.stream_quality == StreamQuality.LOW else "main")
            if priority == "sub":
                widget.stream_quality = "low"
                desired_quality = StreamQuality.LOW
            else:
                widget.stream_quality = "medium"
                desired_quality = cfg.stream_quality if cfg.stream_quality != StreamQuality.LOW else StreamQuality.MEDIUM

            cfg.stream_priority = priority
            if cfg.stream_quality != desired_quality:
                cfg.stream_quality = desired_quality
        except Exception:
            pass

    def _apply_camera_settings(self, widget, settings: dict):
        """Apply persisted camera widget settings gracefully."""
        if not settings:
            return
        try:
            # Stream quality & flags
            widget.stream_quality = settings.get("stream_quality", widget.stream_quality)
            widget.aspect_ratio_locked = settings.get("aspect_ratio_locked", widget.aspect_ratio_locked)
            widget.debug_overlay_enabled = settings.get("debug_overlay_enabled", widget.debug_overlay_enabled)
            widget.motion_boxes_enabled = settings.get("motion_boxes_enabled", widget.motion_boxes_enabled)
            widget.object_detection_enabled = settings.get("object_detection_enabled", widget.object_detection_enabled)

            # Depth overlay settings (DepthAnythingV2)
            depth = settings.get("depth_overlay")
            if isinstance(depth, dict):
                try:
                    # Update config fields conservatively
                    cfg = getattr(widget, "depth_overlay_config", None)
                    if cfg is not None:
                        for k in [
                            "enabled",
                            "fps_limit",
                            "opacity",
                            "colormap",
                            "pointcloud_step",
                            "model_size",
                            "device",
                            "use_fp16",
                            "optimize",
                            "memory_fraction",
                        ]:
                            if k in depth:
                                try:
                                    setattr(cfg, k, depth.get(k))
                                except Exception:
                                    pass
                        # Apply to widget
                        try:
                            widget.depth_overlay_config = cfg
                        except Exception:
                            pass
                        try:
                            widget.gl_widget.set_depth_overlay(
                                getattr(widget.gl_widget, "depth_overlay_image", None),
                                enabled=bool(getattr(cfg, "enabled", False)),
                                opacity=float(getattr(cfg, "opacity", 0.55)),
                            )
                        except Exception:
                            pass
                        # Start if enabled
                        if bool(getattr(cfg, "enabled", False)):
                            try:
                                widget.toggle_depth_overlay(True)
                            except Exception:
                                pass
                except Exception:
                    pass
            # Shapes
            if "shapes" in settings:
                widget.gl_widget.set_shapes(settings.get("shapes") or [])
            if "show_shape_labels" in settings:
                widget.gl_widget.show_shape_labels = bool(settings.get("show_shape_labels"))

            # Detection overlay settings (bounding box/labels/ROI interaction UI)
            det_settings = settings.get("detection_overlay_settings")
            if isinstance(det_settings, dict):
                try:
                    current_ds = getattr(widget.gl_widget, "detection_settings", {}) or {}
                    merged_ds = dict(current_ds)
                    merged_ds.update(self._deserialize_detection_overlay_settings(det_settings))
                    widget.gl_widget.detection_settings = merged_ds
                    try:
                        widget.gl_widget.refresh_detection_roi()
                    except Exception:
                        pass
                except Exception:
                    pass

            # Desktop-local YOLO detector state/config (Desktop-first detections)
            desktop_cfg = settings.get("desktop_detector_config")
            desktop_enabled = settings.get("desktop_object_detection_enabled")
            if isinstance(desktop_cfg, dict) or isinstance(desktop_enabled, bool):
                try:
                    cfg_obj = None
                    if isinstance(desktop_cfg, dict):
                        cfg_obj = self._deserialize_desktop_detector_config(desktop_cfg)
                    self._apply_desktop_detector_restore(widget, enabled=bool(desktop_enabled), cfg=cfg_obj)
                except Exception:
                    pass

            # Audio/PTZ/overlay projections are restored after the widget is shown
            # (because they depend on geometry / parent rects being realized).

            # Apply overlays
            widget.gl_widget.set_aspect_ratio_mode(
                Qt.AspectRatioMode.KeepAspectRatioByExpanding if widget.aspect_ratio_locked
                else Qt.AspectRatioMode.IgnoreAspectRatio
            )
            widget.gl_widget.set_overlay_settings(widget.debug_overlay_enabled, widget.motion_boxes_enabled)

            # Motion settings
            motion_settings = settings.get("motion_settings")
            if motion_settings:
                saved_ms = self._deserialize_motion_settings(motion_settings)
                current_ms = getattr(widget.gl_widget, "motion_settings", {}) or {}
                merged_ms = dict(current_ms)
                merged_ms.update(saved_ms)
                widget.gl_widget.motion_settings = merged_ms
                widget.gl_widget.update()
        except Exception as e:
            logger.warning(f"Failed to apply camera settings: {e}")

    def _serialize_detection_overlay_settings(self, settings: dict) -> dict:
        if not settings:
            return {}
        safe: dict = {}
        for k, v in dict(settings).items():
            try:
                if isinstance(v, QColor):
                    safe[str(k)] = v.name()
                else:
                    safe[str(k)] = v
            except Exception:
                continue
        return safe

    def _deserialize_detection_overlay_settings(self, settings: dict) -> dict:
        if not settings:
            return {}
        ds = dict(settings)
        # Only 'color' is expected to be a QColor in the GL widget.
        try:
            c = ds.get("color")
            if isinstance(c, str) and c.strip():
                ds["color"] = QColor(c)
        except Exception:
            pass
        return ds

    def _serialize_desktop_detector_config(self, cfg) -> dict:
        """
        JSON-safe subset of desktop.utils.detector_worker.DetectorConfig.
        Stored inside layout view so detector settings don't leak across layouts.
        """
        if cfg is None:
            return {}
        keys = [
            "enabled",
            "fps_limit",
            "model_variant",
            "device",
            "imgsz",
            "min_confidence",
            "max_det",
            "allowed_classes",
            "tracking_enabled",
            "tracker_type",
            "tracker_params",
            "emit_detections",
            "tracker_reset_token",
        ]
        out: dict = {}
        for k in keys:
            try:
                out[k] = getattr(cfg, k)
            except Exception:
                continue
        return out

    def _deserialize_desktop_detector_config(self, d: dict):
        try:
            from desktop.utils.detector_worker import DetectorConfig
        except Exception:
            DetectorConfig = None
        if DetectorConfig is None:
            return None
        if not isinstance(d, dict):
            return None
        base = DetectorConfig()
        for k, v in dict(d).items():
            try:
                if hasattr(base, k):
                    setattr(base, k, v)
            except Exception:
                continue
        return base

    def _apply_desktop_detector_restore(self, widget, *, enabled: bool, cfg=None) -> None:
        """
        Restore Desktop-local YOLO detector state without "toggling" UI actions.
        """
        try:
            if cfg is not None:
                try:
                    widget.desktop_detector_config = cfg
                except Exception:
                    pass
            # Fall back to existing config if we couldn't deserialize
            cfg0 = getattr(widget, "desktop_detector_config", None)
            if cfg0 is None:
                return
            try:
                widget.desktop_object_detection_enabled = bool(enabled)
            except Exception:
                pass
            try:
                cfg0.enabled = bool(enabled)
            except Exception:
                pass
            # Prefer Desktop layer when enabled
            try:
                widget.gl_widget.set_desktop_detection_active(bool(enabled))
            except Exception:
                pass
            try:
                if hasattr(widget, "obj_det_action"):
                    widget.obj_det_action.setChecked(bool(enabled))
            except Exception:
                pass
            if bool(enabled):
                try:
                    widget._ensure_desktop_detector_worker()
                except Exception:
                    pass
                try:
                    if getattr(widget, "_desktop_detector_worker", None) is not None:
                        widget._desktop_detector_worker.update_config(cfg0)
                except Exception:
                    pass
                try:
                    widget.gl_widget.set_desktop_tracking_active(bool(getattr(cfg0, "tracking_enabled", False)))
                except Exception:
                    pass
            else:
                try:
                    widget.gl_widget.update_desktop_detections([])
                except Exception:
                    pass
                try:
                    widget.gl_widget.set_desktop_tracking_active(False)
                except Exception:
                    pass
        except Exception:
            return

    def _serialize_motion_settings(self, settings: dict) -> dict:
        if not settings:
            return {}
        safe = dict(settings)
        color = safe.get("color")
        if isinstance(color, QColor):
            safe["color"] = color.name()
        return safe

    def _deserialize_motion_settings(self, settings: dict) -> dict:
        if not settings:
            return {}
        ms = dict(settings)
        color = ms.get("color")
        if isinstance(color, str):
            ms["color"] = QColor(color)
        return ms

    def _resolve_camera_id(self, camera_ref: str) -> str | None:
        """
        Resolve camera reference by id, name (case-insensitive), or IP address.
        """
        if not camera_ref or not self.camera_manager:
            return None

        # Direct ID hit
        if camera_ref in self.camera_manager.cameras:
            return camera_ref

        ref_lower = camera_ref.lower()
        for cam_id, cfg in self.camera_manager.cameras.items():
            if cfg.name and cfg.name.lower() == ref_lower:
                return cam_id
            ip_addr = getattr(cfg, "ip_address", None) or getattr(cfg, "ip", None)
            if ip_addr and ip_addr.lower() == ref_lower:
                return cam_id
        return None

    def spawn_camera_widget(self, camera_ref):
        logger.info(f"Spawning camera widget for ref: {camera_ref}")

        cam_id = self._resolve_camera_id(camera_ref)
        if not cam_id:
            logger.error(f"Camera not found for ref: {camera_ref}")
            self.tray_icon.showMessage("Knoxnet VMS Beta", f"Camera not found: {camera_ref}", QSystemTrayIcon.MessageIcon.Warning)
            return
        
        from desktop.widgets.camera import CameraWidget
        
        # Pass our shared CameraManager instance!
        widget = CameraWidget(cam_id, camera_manager=self.camera_manager)
        # Use camera name as title if available
        cam_cfg = self.camera_manager.cameras.get(cam_id)
        if cam_cfg and getattr(cam_cfg, "name", None):
            widget.setWindowTitle(f"Camera: {cam_cfg.name}")
            
        # Connect the global frame signal to this widget's slot
        self.frame_signal.connect(widget.receive_frame)
        self._register_widget(widget)
        widget.show()
        # Ensure this camera is connected only when a widget is opened (on-demand).
        try:
            import asyncio
            if self.camera_loop and self.camera_loop.is_running():
                self._sync_widget_quality_to_config(widget)
                asyncio.run_coroutine_threadsafe(self.camera_manager.acquire_camera(cam_id), self.camera_loop)
        except Exception:
            pass
        return widget

    def _find_camera_widget(self, camera_ref: str):
        """Return an active CameraWidget matching the camera_ref (id/name/ip), if any."""
        try:
            from desktop.widgets.camera import CameraWidget
        except Exception:
            CameraWidget = None
        if not CameraWidget:
            return None
        cam_id = self._resolve_camera_id(camera_ref) if camera_ref else None
        for w in list(self.active_widgets):
            try:
                if not w.isVisible():
                    continue
                if isinstance(w, CameraWidget):
                    if cam_id and str(getattr(w, "camera_id", "")).lower() == str(cam_id).lower():
                        return w
                    title = (w.windowTitle() or "").lower()
                    if camera_ref and camera_ref.lower() in title:
                        return w
            except Exception:
                continue
        return None

    def show_status(self):
        logger.info("Status requested.")
        self.tray_icon.showMessage("Knoxnet VMS Beta", "System is running.")

    def quit_app(self):
        logger.info("Shutting down...")

        # ── Build the shutdown dialog ──
        from PySide6.QtWidgets import (
            QDialog, QVBoxLayout, QHBoxLayout, QCheckBox, QLabel,
            QPushButton, QFrame,
        )
        from PySide6.QtCore import Qt

        prefs = self._load_prefs()
        quit_prefs = prefs.get("quit_options") if isinstance(prefs, dict) else None
        if not isinstance(quit_prefs, dict):
            quit_prefs = {}

        # Detect current recording state
        rec_count = 0
        total_cams = 0
        try:
            with self._recording_status_lock:
                rec_cache = dict(self._recording_status_cache)
            total_cams = len(rec_cache)
            rec_count = sum(1 for v in rec_cache.values() if v)
        except Exception:
            pass

        dlg = QDialog()
        dlg.setWindowTitle("Quit Knoxnet VMS Beta")
        dlg.setFixedWidth(420)
        dlg.setWindowFlags(Qt.WindowType.Dialog | Qt.WindowType.WindowStaysOnTopHint)
        layout = QVBoxLayout(dlg)
        layout.setSpacing(12)

        # Header
        header = QLabel("Close the desktop app. Choose what to keep running.")
        header.setStyleSheet("font-size: 13px; color: #e2e8f0; padding-bottom: 4px;")
        header.setWordWrap(True)
        layout.addWidget(header)

        # Save layout
        save_cb = QCheckBox("Save current layout")
        save_cb.setChecked(quit_prefs.get("save_layout", True) and bool(self.active_widgets))
        save_cb.setEnabled(bool(self.active_widgets))
        layout.addWidget(save_cb)

        # Separator
        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.HLine)
        sep.setStyleSheet("color: #334155;")
        layout.addWidget(sep)

        # Recording options
        rec_label_text = f"Recording ({rec_count}/{total_cams} cameras)" if total_cams else "Recording"
        rec_label = QLabel(rec_label_text)
        rec_label.setStyleSheet("font-weight: bold; font-size: 12px; color: #94a3b8;")
        layout.addWidget(rec_label)

        continue_rec_cb = QCheckBox("Continue recording in background")
        continue_rec_cb.setChecked(quit_prefs.get("continue_recording", True))
        continue_rec_cb.setEnabled(rec_count > 0)
        if rec_count == 0:
            continue_rec_cb.setChecked(False)
            continue_rec_cb.setToolTip("No cameras are currently recording")
        layout.addWidget(continue_rec_cb)

        # Backend services
        sep2 = QFrame()
        sep2.setFrameShape(QFrame.Shape.HLine)
        sep2.setStyleSheet("color: #334155;")
        layout.addWidget(sep2)

        svc_label = QLabel("Services")
        svc_label.setStyleSheet("font-weight: bold; font-size: 12px; color: #94a3b8;")
        layout.addWidget(svc_label)

        leave_backend_cb = QCheckBox("Leave backend and MediaMTX running")
        leave_backend_cb.setChecked(quit_prefs.get("leave_backend", True))
        layout.addWidget(leave_backend_cb)

        # Auto-logic: recording requires backend
        def _sync_toggles():
            if continue_rec_cb.isChecked():
                leave_backend_cb.setChecked(True)
                leave_backend_cb.setEnabled(False)
            else:
                leave_backend_cb.setEnabled(True)
        continue_rec_cb.toggled.connect(_sync_toggles)
        _sync_toggles()

        hint = QLabel("")
        hint.setStyleSheet("font-size: 11px; color: #64748b;")
        hint.setWordWrap(True)

        def _update_hint():
            if continue_rec_cb.isChecked():
                hint.setText(f"Desktop closes. {rec_count} camera(s) keep recording via MediaMTX.")
            elif leave_backend_cb.isChecked():
                hint.setText("Desktop closes. Backend stays up (API, streaming). No active recordings.")
            else:
                hint.setText("Everything shuts down: desktop, backend, MediaMTX, and recordings.")
        continue_rec_cb.toggled.connect(lambda: _update_hint())
        leave_backend_cb.toggled.connect(lambda: _update_hint())
        _update_hint()
        layout.addWidget(hint)

        # Buttons
        btn_row = QHBoxLayout()
        btn_row.addStretch()
        cancel_btn = QPushButton("Cancel")
        cancel_btn.setFixedWidth(80)
        cancel_btn.clicked.connect(dlg.reject)
        quit_btn = QPushButton("Quit")
        quit_btn.setFixedWidth(80)
        quit_btn.setDefault(True)
        quit_btn.clicked.connect(dlg.accept)
        btn_row.addWidget(cancel_btn)
        btn_row.addWidget(quit_btn)
        layout.addLayout(btn_row)

        if dlg.exec() != QDialog.DialogCode.Accepted:
            return

        # ── Persist choices ──
        quit_prefs["save_layout"] = save_cb.isChecked()
        quit_prefs["continue_recording"] = continue_rec_cb.isChecked()
        quit_prefs["leave_backend"] = leave_backend_cb.isChecked()
        prefs["quit_options"] = quit_prefs
        self._save_prefs(prefs)

        # ── Execute choices ──

        # 1. Save layout
        if save_cb.isChecked() and self.active_widgets:
            self.prompt_save_layout()

        # 2. Stop recording if requested
        if not continue_rec_cb.isChecked() and rec_count > 0:
            logger.info("Stopping all recordings before quit…")
            try:
                with self._recording_status_lock:
                    rec_ids = [k for k, v in self._recording_status_cache.items() if v]
                for cid in rec_ids:
                    try:
                        requests.post(
                            f"http://localhost:5000/api/cameras/{cid}/recording",
                            json={"record": False}, timeout=3,
                        )
                    except Exception:
                        pass
            except Exception as e:
                logger.warning("Error stopping recordings on quit: %s", e)

        # 3. Close all camera widgets
        for w in list(self.active_widgets):
            try:
                if hasattr(w, 'running'):
                    w.running = False
                w.close()
            except Exception:
                pass
        self.active_widgets.clear()

        try:
            self.processEvents()
        except Exception:
            pass

        # 4. Tear down camera manager and load shedder so background
        # threads stop calling into Qt before we destroy the app.
        # Skipping this caused "Signal source has been deleted" errors
        # and intermittent core dumps because RTSP capture threads
        # outlive the QApplication and try to emit dead signals.
        try:
            self._shutdown_runtime_threads()
        except Exception as exc:
            logger.warning("Runtime shutdown error: %s", exc)

        # 5. Stop backend services if requested
        self.ipc_server.stop()
        if not leave_backend_cb.isChecked():
            logger.info("Shutting down backend services…")
            if self.server_manager:
                self.server_manager.stop_servers()
            # Kill Flask and MediaMTX if we started them
            self._kill_backend_services()

        self.quit()

    @Slot()
    def _on_about_to_quit(self) -> None:
        """Final-chance cleanup hook fired by QApplication.aboutToQuit.

        Idempotent: `_shutdown_runtime_threads` is safe to call twice
        but we still guard with `_shutdown_done` to avoid duplicate
        log lines and join attempts on already-stopped threads.
        """
        if getattr(self, "_shutdown_done", False):
            return
        self._shutdown_done = True
        try:
            self._shutdown_runtime_threads()
        except Exception as exc:
            logger.warning("aboutToQuit cleanup error: %s", exc)

    def _shutdown_runtime_threads(self) -> None:
        """Stop background workers that touch Qt objects BEFORE the
        QApplication is destroyed.

        Order matters:
          1. Detach the camera frame callback so capture threads stop
             trying to emit `frame_signal` (this is the actual cause
             of the 'Signal source has been deleted' crash).
          2. Stop the load shedder timers + heartbeat thread.
          3. Stop the camera manager (joins its capture threads).
          4. Stop the asyncio loop driving the camera manager.
        """
        # 1. Detach frame callback so capture threads stop poking Qt.
        try:
            if self.camera_manager is not None:
                self.camera_manager.on_frame_received = None
        except Exception:
            pass

        # 2. Stop the shedder pieces.
        try:
            if self._shed_timer is not None:
                self._shed_timer.stop()
        except Exception:
            pass
        try:
            self._shed_heartbeat_stop.set()
            if self._shed_heartbeat_thread is not None:
                self._shed_heartbeat_thread.join(timeout=2)
        except Exception:
            pass

        # 3. Stop the camera manager async loop.  We schedule its
        # `.stop()` coroutine on the loop it lives on, then ask the
        # loop itself to stop.  Bounded waits prevent hanging the
        # quit if a capture thread is wedged.
        try:
            cm = self.camera_manager
            loop = getattr(self, "camera_loop", None)
            if cm is not None and loop is not None and loop.is_running():
                fut = asyncio.run_coroutine_threadsafe(cm.stop(), loop)
                try:
                    fut.result(timeout=4)
                except Exception:
                    pass
                try:
                    loop.call_soon_threadsafe(loop.stop)
                except Exception:
                    pass
            # Best-effort: wait for the camera-manager thread itself
            # to finish so capture threads are fully gone before quit.
            try:
                if self.cm_thread is not None and self.cm_thread.is_alive():
                    self.cm_thread.join(timeout=3)
            except Exception:
                pass
        except Exception:
            pass

        # 4. Drain any pending Qt events triggered by the cleanup so
        # nothing tries to fire after we return.
        try:
            self.processEvents()
        except Exception:
            pass

    def _kill_backend_services(self):
        """Best-effort shutdown of Flask and MediaMTX processes."""
        for proc_name in ("app.py", "mediamtx"):
            try:
                import subprocess
                result = subprocess.run(
                    ["pkill", "-f", proc_name],
                    capture_output=True, timeout=5,
                )
                if result.returncode == 0:
                    logger.info("Stopped %s", proc_name)
            except Exception as e:
                logger.warning("Could not stop %s: %s", proc_name, e)

    def _register_widget(self, widget):
        """Track widget lifecycle so layouts can be captured accurately."""
        self.active_widgets.add(widget)
        widget.destroyed.connect(lambda _, w=widget: self._unregister_widget(w))
        # For standalone camera widgets, connect shape triggers to the app rule engine.
        try:
            from desktop.widgets.camera import CameraWidget
            if isinstance(widget, CameraWidget):
                if bool(getattr(widget, "_shape_triggered_to_app", False)):
                    return
                gl = getattr(widget, "gl_widget", None)
                if gl is not None and hasattr(gl, "shape_triggered"):
                    # If a layout already connected a layout-specific handler, it will set this flag.
                    gl.shape_triggered.connect(self._on_camera_shape_triggered)
                    setattr(widget, "_shape_triggered_to_app", True)
        except Exception:
            pass

    def _unregister_widget(self, widget):
        self.active_widgets.discard(widget)

    # --- Layout Persistence ---
    def _capture_current_layout_v2(self, layout_id: str, name: str) -> LayoutDefinition:
        from desktop.widgets.camera import CameraWidget
        # from desktop.widgets.web import WebWidget
        WebWidget = None
        try:
            from desktop.widgets.terminal import TerminalWidget
        except Exception:
            TerminalWidget = None

        widgets: list[WidgetDefinition] = []
        for w in list(self.active_widgets):
            if not w.isVisible():
                continue
            geo = w.geometry()
            wid = f"{layout_id}:{type(w).__name__}:{geo.x()}:{geo.y()}:{geo.width()}:{geo.height()}"

            # Detect X11 workspace (virtual desktop) for this widget.
            _widget_desktop: int | None = None
            try:
                import subprocess
                _xid = int(w.winId())
                _res = subprocess.run(
                    ["xdotool", "get-desktop-for-window", str(_xid)],
                    capture_output=True, text=True, timeout=2,
                )
                if _res.returncode == 0 and _res.stdout.strip().isdigit():
                    _widget_desktop = int(_res.stdout.strip())
            except Exception:
                pass

            entry = WidgetDefinition(
                id=wid,
                type="unknown",
                x=geo.x(),
                y=geo.y(),
                w=geo.width(),
                h=geo.height(),
                title=w.windowTitle() or "",
                pinned=bool(getattr(w, "is_pinned", False)),
                desktop=_widget_desktop,
            )
            if isinstance(w, CameraWidget):
                entry.type = "camera"
                entry.camera_id = str(w.camera_id)
                # Layouts persist camera view state *including* overlay shapes (zones/lines/tags),
                # so overlays do not leak across layouts.
                shapes: list[dict] = []
                try:
                    gl = getattr(w, "gl_widget", None)
                    raw_shapes = list(getattr(gl, "shapes", []) or [])
                    for sh in raw_shapes:
                        if not isinstance(sh, dict):
                            continue
                        ss = dict(sh)
                        # Best-effort make JSON-safe (avoid QColor objects if present).
                        for k in ("color", "text_color", "interaction_color"):
                            try:
                                v = ss.get(k)
                                if isinstance(v, QColor):
                                    ss[k] = v.name()
                            except Exception:
                                continue
                        shapes.append(ss)
                except Exception:
                    shapes = []
                entry.view = {
                    "aspect_ratio_locked": getattr(w, "aspect_ratio_locked", True),
                    "stream_quality": getattr(w, "stream_quality", "medium"),
                    "debug_overlay_enabled": getattr(w, "debug_overlay_enabled", False),
                    "motion_boxes_enabled": getattr(w, "motion_boxes_enabled", False),
                    "motion_settings": self._serialize_motion_settings(
                        getattr(getattr(w, "gl_widget", None), "motion_settings", {})
                    ),
                    "object_detection_enabled": getattr(w, "object_detection_enabled", False),
                    # Detection overlay settings (bbox/labels/ROI display)
                    "detection_overlay_settings": self._serialize_detection_overlay_settings(
                        getattr(getattr(w, "gl_widget", None), "detection_settings", {}) or {}
                    ),
                    # Desktop-local YOLO detector state/config
                    "desktop_object_detection_enabled": bool(getattr(w, "desktop_object_detection_enabled", False)),
                    "desktop_detector_config": self._serialize_desktop_detector_config(
                        getattr(w, "desktop_detector_config", None)
                    ),
                    "show_shape_labels": getattr(getattr(w, "gl_widget", None), "show_shape_labels", True),
                    "shapes": shapes,
                    # Audio EQ (dock/undock) + settings (layout-scoped)
                    "audio_eq": {
                        "docked": bool(getattr(w, "audio_eq_overlay", None) is not None),
                        "undocked": bool(getattr(w, "audio_eq_window", None) is not None),
                        "playing": bool(getattr(w, "_audio_playing", False)),
                        "settings": {
                            "bars": int(getattr(getattr(w, "audio_eq_settings", None), "bars", 48)),
                            "fft_size": int(getattr(getattr(w, "audio_eq_settings", None), "fft_size", 4096)),
                            "opacity": float(getattr(getattr(w, "audio_eq_settings", None), "opacity", 0.70)),
                            "accent_color": str(getattr(getattr(w, "audio_eq_settings", None), "accent_color", "#00ff8c")),
                            "bg_color": str(getattr(getattr(w, "audio_eq_settings", None), "bg_color", "#0b1220")),
                            "visual_mode": str(getattr(getattr(w, "audio_eq_settings", None), "visual_mode", "spectrum_bars")),
                            "borderless": bool(getattr(getattr(w, "audio_eq_settings", None), "borderless", False)),
                            "pos_norm": list(getattr(getattr(w, "audio_eq_settings", None), "pos_norm", (0.15, 0.85))),
                            "size_px": list(getattr(getattr(w, "audio_eq_settings", None), "size_px", (280, 140))),
                        },
                    },
                    # PTZ overlay (dock/undock) + settings (layout-scoped)
                    "ptz": {
                        "docked": bool(getattr(w, "ptz_overlay", None) is not None),
                        "undocked": bool(getattr(w, "ptz_controller_window", None) is not None),
                        "settings": dict(getattr(w, "ptz_overlay_settings", {}) or {}),
                    },
                    # Overlay projection windows for zones/lines/tags (layout-scoped)
                    "overlay_projections": [
                        {
                            "target_ids": list(getattr(win, "target_ids", []) or []),
                            "settings": dict(getattr(win, "settings", {}) or {}),
                            "geometry": {
                                "x": int(getattr(win.geometry(), "x", lambda: 0)()),
                                "y": int(getattr(win.geometry(), "y", lambda: 0)()),
                                "w": int(getattr(win.geometry(), "width", lambda: 520)()),
                                "h": int(getattr(win.geometry(), "height", lambda: 320)()),
                            },
                        }
                        for win in list((getattr(w, "overlay_windows", None) or {}).values())
                        if win is not None and bool(getattr(win, "isVisible", lambda: True)())
                    ],
                    "depth_overlay": {
                        "enabled": bool(getattr(getattr(w, "depth_overlay_config", None), "enabled", False)),
                        "fps_limit": int(getattr(getattr(w, "depth_overlay_config", None), "fps_limit", 12)),
                        "opacity": float(getattr(getattr(w, "depth_overlay_config", None), "opacity", 0.55)),
                        "colormap": str(getattr(getattr(w, "depth_overlay_config", None), "colormap", "turbo")),
                        "pointcloud_step": int(getattr(getattr(w, "depth_overlay_config", None), "pointcloud_step", 6)),
                        "model_size": str(getattr(getattr(w, "depth_overlay_config", None), "model_size", "vits")),
                        "device": str(getattr(getattr(w, "depth_overlay_config", None), "device", "cuda")),
                        "use_fp16": bool(getattr(getattr(w, "depth_overlay_config", None), "use_fp16", True)),
                        "optimize": bool(getattr(getattr(w, "depth_overlay_config", None), "optimize", True)),
                        "memory_fraction": getattr(getattr(w, "depth_overlay_config", None), "memory_fraction", None),
                    },
                }
            # elif isinstance(w, WebWidget):
            #    entry["type"] = "web"
            #    entry["url"] = w.browser.url().toString()
            elif TerminalWidget and isinstance(w, TerminalWidget):
                entry.type = "terminal"
                entry.view = {
                    "agent_active": getattr(w, "agent_active", False),
                    "text_color": getattr(w, "text_color", None),
                    "text_scale": getattr(w, "text_scale", None),
                }
            else:
                continue
            widgets.append(entry)
        return LayoutDefinition(id=layout_id, name=name or layout_id, widgets=widgets, meta={"source": "desktop_v2"})

    def _restore_camera_extras(self, widget, settings: dict) -> None:
        """
        Restore UI extras that depend on realized geometry:
          - Audio EQ overlay/window
          - PTZ overlay/window
          - Overlay projection windows
        """
        if not settings:
            return
        try:
            # --- Audio EQ ---
            audio = settings.get("audio_eq")
            if isinstance(audio, dict):
                # Apply saved settings to the existing dataclass instance (avoid heavy imports).
                try:
                    aes = getattr(widget, "audio_eq_settings", None)
                    saved = audio.get("settings") if isinstance(audio.get("settings"), dict) else {}
                    if aes is not None and isinstance(saved, dict):
                        for k in [
                            "bars",
                            "fft_size",
                            "opacity",
                            "accent_color",
                            "bg_color",
                            "visual_mode",
                            "borderless",
                            "pos_norm",
                            "size_px",
                        ]:
                            if k in saved:
                                try:
                                    setattr(aes, k, saved.get(k))
                                except Exception:
                                    pass
                except Exception:
                    pass

                want_docked = bool(audio.get("docked", False))
                want_undocked = bool(audio.get("undocked", False))
                want_playing = True if ("playing" not in audio) else bool(audio.get("playing"))

                # Only one mode at a time; prefer undocked if both are true.
                if want_undocked:
                    try:
                        widget.toggle_audio_eq_undocked(True)
                    except Exception:
                        pass
                elif want_docked:
                    try:
                        widget.toggle_audio_eq_overlay(True)
                    except Exception:
                        pass
                if (want_docked or want_undocked) and (not want_playing):
                    try:
                        widget.stop_audio_monitor()
                    except Exception:
                        pass

            # --- PTZ ---
            ptz = settings.get("ptz")
            if isinstance(ptz, dict):
                try:
                    if isinstance(ptz.get("settings"), dict):
                        widget.ptz_overlay_settings = dict(ptz.get("settings") or {})
                except Exception:
                    pass
                want_undocked = bool(ptz.get("undocked", False))
                want_docked = bool(ptz.get("docked", False))
                if want_undocked:
                    try:
                        widget.toggle_ptz_undocked(True)
                    except Exception:
                        pass
                elif want_docked:
                    try:
                        widget.toggle_ptz_overlay(True)
                    except Exception:
                        pass

            # --- Overlay projections ---
            projs = settings.get("overlay_projections")
            if isinstance(projs, list):
                for rec in projs:
                    if not isinstance(rec, dict):
                        continue
                    ids = rec.get("target_ids") if isinstance(rec.get("target_ids"), list) else []
                    ids = [str(x) for x in ids if str(x).strip()]
                    try:
                        widget.open_overlay_window(ids)
                    except Exception:
                        continue
                    try:
                        key = widget._overlay_key(ids)
                        win = (getattr(widget, "overlay_windows", None) or {}).get(key)
                        if win is None:
                            continue
                        geom = rec.get("geometry") if isinstance(rec.get("geometry"), dict) else {}
                        try:
                            win.setGeometry(
                                int(geom.get("x", win.x())),
                                int(geom.get("y", win.y())),
                                int(geom.get("w", win.width())),
                                int(geom.get("h", win.height())),
                            )
                        except Exception:
                            pass
                        s = rec.get("settings") if isinstance(rec.get("settings"), dict) else {}
                        if isinstance(s, dict) and s:
                            try:
                                win.settings.update(s)
                            except Exception:
                                pass
                            try:
                                win.setWindowOpacity(float(win.settings.get("opacity", 0.85)))
                            except Exception:
                                pass
                            # Best-effort apply pin/passthrough + poll interval
                            try:
                                if hasattr(win, "_apply_on_top"):
                                    win._apply_on_top(bool(win.settings.get("always_on_top", True)))
                            except Exception:
                                pass
                            try:
                                if hasattr(win, "_apply_passthrough"):
                                    win._apply_passthrough(bool(win.settings.get("pass_through", False)))
                            except Exception:
                                pass
                            try:
                                if hasattr(win, "poll_timer") and win.poll_timer is not None:
                                    win.poll_timer.stop()
                                    win.poll_timer.start(max(15, int(win.settings.get("poll_ms", 50))))
                            except Exception:
                                pass
                        try:
                            win.show()
                            win.raise_()
                        except Exception:
                            pass
                    except Exception:
                        continue
        except Exception:
            return

    def _list_layouts_v2(self) -> list[LayoutDefinition]:
        try:
            return self.layouts_store.list_layouts()
        except Exception:
            return []

    def _apply_assigned_profile_overlays(self, camera_id: str, widget) -> None:
        """
        Apply all profile features (shapes, overlay styles, toggles, monitoring)
        from the camera's assigned profile(s) to a CameraWidget.
        """
        try:
            assigns = self.layouts_store.get_assignments()
            val = assigns.get(camera_id)
            profile_ids: list = []
            if isinstance(val, str) and val:
                profile_ids = [val]
            elif isinstance(val, list):
                profile_ids = [str(x) for x in val if str(x).strip()]
            if not profile_ids:
                return

            shapes: list = []
            for pid in profile_ids:
                prof = self.layouts_store.get_profile(pid)
                if not prof:
                    continue
                ov = prof.overlays or {}

                # Shapes
                s = ov.get("shapes")
                if isinstance(s, list):
                    shapes.extend(s)

                gl = getattr(widget, "gl_widget", None)

                # Motion overlay style
                ms = ov.get("motion_settings")
                if isinstance(ms, dict) and gl:
                    merged = dict(getattr(gl, "motion_settings", {}) or {})
                    merged.update(self._deserialize_settings_dict(ms))
                    gl.motion_settings = merged

                # Motion boxes enabled
                if "motion_boxes_enabled" in ov:
                    try:
                        widget.motion_boxes_enabled = bool(ov["motion_boxes_enabled"])
                        widget._apply_overlay_settings()
                    except Exception:
                        pass

                # Detection overlay style
                ds = ov.get("detection_settings")
                if isinstance(ds, dict) and gl:
                    merged = dict(getattr(gl, "detection_settings", {}) or {})
                    merged.update(self._deserialize_settings_dict(ds))
                    gl.detection_settings = merged

                # Debug overlay
                if "debug_overlay_enabled" in ov:
                    try:
                        widget.debug_overlay_enabled = bool(ov["debug_overlay_enabled"])
                        widget._apply_overlay_settings()
                    except Exception:
                        pass

                # AI pipeline settings
                ai = prof.ai_pipeline or {}
                if "object_detection_enabled" in ai:
                    try:
                        if bool(ai["object_detection_enabled"]) and not widget.desktop_object_detection_enabled:
                            widget.toggle_object_detection()
                        elif not bool(ai["object_detection_enabled"]) and widget.desktop_object_detection_enabled:
                            widget.toggle_object_detection()
                    except Exception:
                        pass

                # Monitoring tools
                mt = prof.monitoring_tools or {}
                mw = mt.get("motion_watch_settings")
                if isinstance(mw, dict):
                    try:
                        widget.motion_watch_settings.update(mw)
                    except Exception:
                        pass

            if shapes:
                try:
                    widget.gl_widget.set_shapes(shapes)
                except Exception:
                    pass
        except Exception:
            return

    def _load_saved_layouts_legacy(self):
        """Legacy v1 layouts dict from data/desktop_layouts.json (for migration only)."""
        if not self.layout_file.exists():
            return {}
        try:
            with open(self.layout_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}

    def _load_prefs(self) -> dict:
        defaults = {
            "layout_switch_policy_default": "ask",  # ask|stop|keep
            "layout_switch_policy_overrides": {},   # {layout_id: "stop"|"keep"}
            # Layout visibility + auto-hide
            # layout_visibility: {layout_id: {"hidden": bool}}
            "layout_visibility": {},
            # layout_auto_hide: {layout_id: {"on_layout_switch": bool, "on_motion": bool, "on_detections": bool}}
            "layout_auto_hide": {},
            # layout_auto_show: {layout_id: {"on_motion": bool, "on_detections": bool}}
            "layout_auto_show": {},
            # widget_*: keyed by "camera:<camera_id>" for now
            "widget_visibility": {},
            "widget_auto_hide": {},
            "widget_auto_show": {},
            # Timed auto-show rules (shape-triggered)
            "visibility_rules": [],
            # Global safety: allow bring-to-front/activateWindow when auto-show triggers
            "focus_steal_enabled": True,
            # Legacy layouts migration guard: prevents deleted layouts from being re-imported on restart
            "legacy_layouts_migrated": False,
            # ---- Layout scheduler ----
            "scheduler_enabled": True,
            # list of schedule dicts (see _scheduler_* helpers)
            "layout_schedules": [],
            # ISO datetime string; when set, scheduler will not auto-switch until this time.
            "scheduler_snooze_until": None,
            # When user manually changes layouts, pause scheduler for N minutes.
            "scheduler_manual_override_minutes": 30,
            # Simple controls mode: hides advanced submenus in camera context menu
            "simple_controls": False,
            # Patrol mode default interval in seconds
            "patrol_interval_sec": 10,
            # Ordered list of camera_ids for patrol; empty = use all visible cameras
            "patrol_order": [],
            # Hotkeys
            "hotkeys_enabled": True,
            # Grid/snap settings
            "snap_enabled": True,
            "snap_grid_size": 20,
            "snap_edge_threshold": 24,
            "snap_show_guides": True,
            # Auto-protection / load shedder
            "auto_protect": {
                "enabled": True,
                "protect_recording": True,
                "machine_class": "auto",
                "thresholds": {},
                "throttles": {},
                "primary_widget_count": 2,
                "exit_on_emergency_after_sec": 60,
                "show_overlay": True,
            },
        }

        # Best-effort migrate legacy prefsX if desktop_prefs.json doesn't exist yet.
        if not self.prefs_file.exists():
            legacy = Path("data/desktop_prefsX.json")
            if legacy.exists():
                try:
                    with open(legacy, "r") as f:
                        raw = json.load(f)
                    if isinstance(raw, dict):
                        merged = dict(defaults)
                        merged.update(raw)
                        if not isinstance(merged.get("layout_switch_policy_overrides"), dict):
                            merged["layout_switch_policy_overrides"] = {}
                        return merged
                except Exception:
                    pass
            return dict(defaults)

        try:
            with open(self.prefs_file, "r") as f:
                raw = json.load(f)
            if not isinstance(raw, dict):
                raw = {}
            merged = dict(defaults)
            merged.update(raw)
            if not isinstance(merged.get("layout_switch_policy_overrides"), dict):
                merged["layout_switch_policy_overrides"] = {}
            if not isinstance(merged.get("layout_visibility"), dict):
                merged["layout_visibility"] = {}
            if not isinstance(merged.get("layout_auto_hide"), dict):
                merged["layout_auto_hide"] = {}
            if not isinstance(merged.get("layout_auto_show"), dict):
                merged["layout_auto_show"] = {}
            if not isinstance(merged.get("widget_visibility"), dict):
                merged["widget_visibility"] = {}
            if not isinstance(merged.get("widget_auto_hide"), dict):
                merged["widget_auto_hide"] = {}
            if not isinstance(merged.get("widget_auto_show"), dict):
                merged["widget_auto_show"] = {}
            if not isinstance(merged.get("visibility_rules"), list):
                merged["visibility_rules"] = []
            if "focus_steal_enabled" not in merged:
                merged["focus_steal_enabled"] = True
            else:
                merged["focus_steal_enabled"] = bool(merged.get("focus_steal_enabled"))
            if "legacy_layouts_migrated" not in merged:
                merged["legacy_layouts_migrated"] = False
            else:
                merged["legacy_layouts_migrated"] = bool(merged.get("legacy_layouts_migrated"))
            # Ensure auto_protect block exists with all required keys
            ap = merged.get("auto_protect")
            if not isinstance(ap, dict):
                ap = {}
            ap_defaults = defaults.get("auto_protect", {}) if isinstance(defaults.get("auto_protect"), dict) else {}
            for k, v in ap_defaults.items():
                ap.setdefault(k, v)
            merged["auto_protect"] = ap
            return merged
        except Exception as e:
            logger.warning(f"Failed to load desktop prefs: {e}")
            return dict(defaults)

    def _save_prefs(self, prefs: dict):
        try:
            if not isinstance(prefs, dict):
                prefs = {}
            with open(self.prefs_file, "w") as f:
                json.dump(prefs, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save desktop prefs: {e}")

    def _maybe_migrate_legacy_layouts(self, *, force: bool = False) -> None:
        """
        Migrate `data/desktop_layouts.json` into the v2 store once.

        Without a guard, startup migration can resurrect layouts the user deleted from v2.
        """
        try:
            prefs = self._load_prefs()
        except Exception:
            prefs = {}
        if not isinstance(prefs, dict):
            prefs = {}

        migrated_flag = bool(prefs.get("legacy_layouts_migrated", False))

        # If we already have v2 layouts, consider migration complete and never auto-run again.
        if not force:
            try:
                existing = self.layouts_store.list_layouts()
                if existing:
                    if not migrated_flag:
                        prefs["legacy_layouts_migrated"] = True
                        self._save_prefs(prefs)
                    return
            except Exception:
                pass
            if migrated_flag:
                return

        try:
            self.layouts_store.migrate_from_legacy_desktop_layouts(
                overwrite_layouts=False,
                create_profiles=True,
                overwrite_assignments=False,
            )
        finally:
            # Mark complete even if no legacy found, so we don't keep trying repeatedly.
            try:
                prefs = self._load_prefs()
                if not isinstance(prefs, dict):
                    prefs = {}
                prefs["legacy_layouts_migrated"] = True
                self._save_prefs(prefs)
            except Exception:
                pass

    def _set_startup_layout(self, name: str | None):
        prefs = self._load_prefs()
        if name:
            prefs["startup_layout"] = name
            self.tray_icon.showMessage("Knoxnet VMS Beta", f"Startup dashboard set to '{name}'", QSystemTrayIcon.MessageIcon.Information)
        else:
            prefs.pop("startup_layout", None)
            self.tray_icon.showMessage("Knoxnet VMS Beta", "Startup dashboard cleared", QSystemTrayIcon.MessageIcon.Information)
        self._save_prefs(prefs)

    def _maybe_load_startup_layout(self):
        if self._startup_layout_loaded:
            return
        prefs = self._load_prefs()
        startup_layout = prefs.get("startup_layout")
        if not startup_layout:
            return
        self._startup_layout_loaded = True
        self.load_layout(startup_layout)

    # ================== Layout scheduler ==================

    def _scheduler_note_manual_override(self) -> None:
        """
        When the user manually changes layouts, pause the scheduler so it doesn't
        immediately switch back due to time rules.
        """
        try:
            prefs = self._load_prefs()
            mins = prefs.get("scheduler_manual_override_minutes") if isinstance(prefs, dict) else None
            mins = 30 if mins is None else int(mins)
            mins = max(0, min(24 * 60, mins))
        except Exception:
            mins = 30
        if mins <= 0:
            return
        self._scheduler_manual_override_until_ts = time.time() + float(mins * 60)

    def _scheduler_snoozed(self) -> bool:
        try:
            prefs = self._load_prefs()
            until = prefs.get("scheduler_snooze_until") if isinstance(prefs, dict) else None
            if not until:
                return False
            dt = datetime.fromisoformat(str(until))
            return datetime.now() < dt
        except Exception:
            return False

    def _scheduler_list(self) -> list[dict]:
        try:
            prefs = self._load_prefs()
            schedules = prefs.get("layout_schedules") if isinstance(prefs, dict) else None
            return list(schedules) if isinstance(schedules, list) else []
        except Exception:
            return []

    def _parse_hhmm(self, s: str) -> tuple[int, int] | None:
        try:
            s = str(s or "").strip()
            if not s:
                return None
            parts = s.split(":")
            if len(parts) != 2:
                return None
            hh = int(parts[0])
            mm = int(parts[1])
            if hh < 0 or hh > 23 or mm < 0 or mm > 59:
                return None
            return hh, mm
        except Exception:
            return None

    def _time_window_match(self, now_dt: datetime, start: str, end: str) -> bool:
        """
        True if now_dt is within [start,end) in local time.
        Handles overnight windows where end < start (e.g. 22:00-06:00).
        """
        st = self._parse_hhmm(start)
        en = self._parse_hhmm(end)
        if not st or not en:
            return False
        start_min = st[0] * 60 + st[1]
        end_min = en[0] * 60 + en[1]
        now_min = now_dt.hour * 60 + now_dt.minute
        if start_min == end_min:
            # Treat "all day" explicitly
            return True
        if end_min > start_min:
            return start_min <= now_min < end_min
        # overnight
        return (now_min >= start_min) or (now_min < end_min)

    def _scheduler_match(self, sched: dict, now_dt: datetime, uptime_sec: float) -> bool:
        if not isinstance(sched, dict):
            return False
        if not bool(sched.get("enabled", True)):
            return False
        layout_id = str(sched.get("layout_id") or "").strip()
        if not layout_id:
            return False
        stype = str(sched.get("type") or "time_of_day").strip().lower()

        # Optional date window
        try:
            start_date = sched.get("start_date") or None  # YYYY-MM-DD
            end_date = sched.get("end_date") or None
            if start_date:
                sd = datetime.fromisoformat(str(start_date) + "T00:00:00")
                if now_dt < sd:
                    return False
            if end_date:
                ed = datetime.fromisoformat(str(end_date) + "T23:59:59")
                if now_dt > ed:
                    return False
        except Exception:
            pass

        if stype in {"time", "time_of_day", "timeofday"}:
            days = sched.get("days")
            if isinstance(days, list) and days:
                try:
                    day_set = set(int(x) for x in days)
                except Exception:
                    day_set = set()
                if day_set and now_dt.weekday() not in day_set:
                    return False
            start = str(sched.get("start_time") or "00:00")
            end = str(sched.get("end_time") or "24:00")
            # allow "24:00" as end sentinel
            if end == "24:00":
                end = "00:00"
                # this will be treated as "all day" by start==end if start is also 00:00
            return self._time_window_match(now_dt, start, end)

        if stype in {"uptime", "runtime"}:
            try:
                after_sec = float(sched.get("after_sec") or 0)
            except Exception:
                after_sec = 0.0
            if after_sec <= 0:
                return False
            if uptime_sec < after_sec:
                return False
            # once_per_launch: only match if we haven't fired this schedule in this launch
            if bool(sched.get("once_per_launch", True)):
                sid = str(sched.get("id") or "")
                if sid and sid in self._scheduler_last_applied_ts:
                    return False
            return True

        return False

    def _scheduler_pick(self, matches: list[dict]) -> dict | None:
        if not matches:
            return None
        # Highest priority wins; tie-break by name then id for determinism.
        def key(s: dict):
            try:
                pr = int(s.get("priority", 100))
            except Exception:
                pr = 100
            nm = str(s.get("name") or "")
            sid = str(s.get("id") or "")
            return (pr, nm, sid)
        return sorted(matches, key=key, reverse=True)[0]

    # ====================================================================
    # Auto-protection load shedder
    # ====================================================================

    def _init_load_shedder(self) -> None:
        """Construct the LoadShedder, start the periodic poll timer, and
        kick off the GUI-thread heartbeat so we can detect stalls."""
        # One-time migration: wipe stale low-threshold saves from
        # earlier versions so users on machines with old prefs files
        # pick up the new (more permissive) defaults automatically.
        # Users with intentional overrides have ap_thresholds_version
        # set; we only wipe if that marker is missing.
        try:
            prefs = self._load_prefs()
            ap = prefs.get("auto_protect") or {}
            if isinstance(ap, dict) and ap.get("thresholds_version") != 2:
                ap["thresholds"] = {}
                ap["thresholds_version"] = 2
                prefs["auto_protect"] = ap
                self._save_prefs(prefs)
                logger.info(
                    "Auto-protection: migrated thresholds to v2 defaults"
                )
        except Exception:
            pass

        prefs = self._load_prefs()
        self._load_shedder = LoadShedder(prefs=prefs)

        # Warm-up grace period: skip the shedder for the first 30s so
        # initial layout load, camera connects, and worker spin-ups
        # don't briefly trigger false positives.  Shorter than before
        # so testing / responsiveness feels immediate.
        self._shed_warmup_until_ts: float = time.time() + 30.0
        # Last candidate level we logged, so we emit ONE info line per
        # candidate transition (not on every 2s tick).
        self._shed_last_logged_candidate: int = -1
        # Last periodic state-snapshot log timestamp; we emit a heartbeat
        # log every 5 minutes so users can confirm the shedder is alive
        # in normal operation without spamming the log.
        self._shed_last_snapshot_log_ts: float = 0.0

        # Prime psutil so cpu_percent() returns meaningful values on
        # the next call (otherwise the first reading is 0% / 100%).
        try:
            import psutil
            psutil.cpu_percent(interval=None)
        except Exception:
            pass

        # 2s poll cadence is a good balance: fast enough to catch a
        # rising spike before it locks the box; slow enough not to
        # itself become a CPU drain.
        self._shed_timer = QTimer()
        self._shed_timer.timeout.connect(self._load_shed_tick)
        self._shed_timer.start(2000)

        # Heartbeat thread + GUI receiver pair.  The signal -> slot
        # connection is implicitly Qt.QueuedConnection across threads,
        # which is thread-safe; QTimer.singleShot from non-GUI threads
        # is NOT, and using it here previously caused the pong to
        # silently never fire (-> stuck counter grew unbounded).
        try:
            self._shed_heartbeat_ping.connect(
                self._shed_heartbeat_pong, Qt.ConnectionType.QueuedConnection,
            )
        except Exception:
            pass
        self._shed_heartbeat_stop.clear()
        self._shed_heartbeat_thread = threading.Thread(
            target=self._shed_heartbeat_loop,
            daemon=True,
            name="ShedHeartbeat",
        )
        self._shed_heartbeat_thread.start()

        logger.info(
            "Load shedder started (machine=%s, enabled=%s, protect_recording=%s)",
            self._load_shedder.machine_class,
            self._load_shedder.enabled,
            self._load_shedder.protect_recording,
        )

    def _shed_heartbeat_loop(self) -> None:
        """Background thread that pings the GUI thread roughly every
        second and tracks how long it has been since we last got a
        response.  When the GUI thread is healthy, the round-trip is
        well under 100ms; if it climbs past several seconds the GUI is
        actively hung (informational only; does not change shed level).

        Uses a Qt signal (not QTimer.singleShot) for the ping because
        signal->slot connections across threads are guaranteed
        thread-safe, while singleShot from a non-GUI thread is not.
        """
        while not self._shed_heartbeat_stop.is_set():
            try:
                now = time.time()
                if not self._shed_heartbeat_pending_since_ts:
                    self._shed_heartbeat_pending_since_ts = now
                    try:
                        self._shed_heartbeat_ping.emit()
                    except Exception:
                        # If the signal source has been deleted (during
                        # shutdown), bail out cleanly instead of
                        # spamming exceptions.
                        return
                else:
                    pending_for = now - self._shed_heartbeat_pending_since_ts
                    if pending_for > self._shed_event_loop_stuck_sec:
                        self._shed_event_loop_stuck_sec = pending_for
            except Exception:
                pass
            self._shed_heartbeat_stop.wait(1.0)

    @Slot()
    def _shed_heartbeat_pong(self) -> None:
        """Called on the GUI thread by the heartbeat ping signal.

        Resetting both the pending timestamp and the stuck counter
        here means a healthy GUI naturally keeps stuck_sec at 0; only
        a genuinely stalled event loop lets it climb.
        """
        self._shed_heartbeat_last_seen_ts = time.time()
        self._shed_heartbeat_pending_since_ts = 0.0
        self._shed_event_loop_stuck_sec = 0.0

    def _load_shed_tick(self) -> None:
        """Single tick of the load shedder.  Gathers metrics, runs the
        state machine, and dispatches actions to the GUI."""
        if self._load_shedder is None:
            return

        # Diagnostic-only long-stall logging.  Stalls no longer drive
        # the shed level, but a sustained one is still worth knowing
        # about (often points to a heavy operation that should be
        # moved to a worker thread).
        try:
            stall = float(self._shed_event_loop_stuck_sec or 0.0)
            if stall >= 10.0 and (time.time() - getattr(self, "_shed_last_stall_log_ts", 0.0)) > 30.0:
                logger.warning(
                    "Auto-protection: GUI thread stalled %.1fs (informational only; not throttling)",
                    stall,
                )
                self._shed_last_stall_log_ts = time.time()
        except Exception:
            pass

        # Warm-up grace period: don't shed during initial app startup,
        # but always gather metrics so the UI has something to show.
        in_warmup = time.time() < float(getattr(self, "_shed_warmup_until_ts", 0.0) or 0.0)

        try:
            metrics = self._gather_system_metrics()
        except Exception as exc:
            logger.warning("Shedder metrics gather failed: %s", exc)
            return

        if in_warmup:
            # Cache metrics for the UI summary even during warmup so
            # users see live CPU/RAM numbers immediately.
            self._shed_last_metrics = metrics
            return

        try:
            decision = self._load_shedder.update(metrics)
            self._shed_last_metrics = metrics
        except Exception as exc:
            logger.warning("Shedder update failed: %s", exc)
            return

        # Log when the candidate (target) level changes -- gives users
        # immediate feedback that the shedder is actively evaluating
        # without waiting for the rise gate to commit.
        try:
            cand = int(getattr(self._load_shedder, "candidate_level", LoadLevel.NORMAL))
            if cand != self._shed_last_logged_candidate:
                self._shed_last_logged_candidate = cand
                cand_lvl = LoadLevel(cand)
                if cand_lvl != decision.level:
                    gate = self._load_shedder.candidate_gate_remaining()
                    logger.info(
                        "Auto-protection: evaluating %s (would commit in %.0fs) "
                        "- CPU %.0f%% RAM %.0f%% Swap %.0f%%",
                        cand_lvl.label, gate,
                        metrics.cpu_percent, metrics.ram_percent, metrics.swap_percent,
                    )
                else:
                    logger.info(
                        "Auto-protection: candidate back to %s (committed)",
                        cand_lvl.label,
                    )
        except Exception:
            pass

        # Periodic heartbeat snapshot every 5 minutes
        try:
            now_ts = time.time()
            if (now_ts - self._shed_last_snapshot_log_ts) > 300.0:
                self._shed_last_snapshot_log_ts = now_ts
                logger.info(
                    "Auto-protection heartbeat: level=%s CPU=%.0f%% RAM=%.0f%% Swap=%.0f%%",
                    decision.level.label,
                    metrics.cpu_percent, metrics.ram_percent, metrics.swap_percent,
                )
        except Exception:
            pass

        if decision.level_changed:
            try:
                evt = ShedEvent(
                    ts=time.time(),
                    from_level=decision.previous_level,
                    to_level=decision.level,
                    reason=decision.reason,
                    summary=self._format_shed_action_summary(decision),
                )
                self._shed_event_log.append(evt)
                logger.info(
                    "Auto-protection level %s -> %s (%s)",
                    decision.previous_level.label, decision.level.label, decision.reason,
                )
            except Exception:
                pass

        try:
            self._apply_shed_decision(decision)
        except Exception as exc:
            logger.warning("Shedder action dispatch failed: %s", exc)

        # EMERGENCY persistence: if we've been at EMERGENCY long enough
        # and recovery hasn't happened, do a controlled exit.  Stall
        # detection no longer drives EMERGENCY (only true memory/swap
        # exhaustion does), so this path is only reached for genuine
        # resource crises.
        if decision.level == LoadLevel.EMERGENCY:
            if self._shed_emergency_since_ts <= 0.0:
                self._shed_emergency_since_ts = time.time()
            elapsed = time.time() - self._shed_emergency_since_ts
            limit = int(self._load_shedder.exit_on_emergency_after_sec or 60)
            if elapsed >= limit and not self._shed_exit_initiated:
                # Re-confirm memory is still genuinely critical before
                # quitting; if RAM/swap have eased even though level
                # hasn't dropped through the recovery gate yet, skip.
                metrics = decision.metrics
                t = self._load_shedder.thresholds
                ram_critical = (
                    getattr(metrics, "ram_percent", 0) >= t.get("emergency_ram", 98)
                    or getattr(metrics, "swap_percent", 0) >= t.get("emergency_swap", 65)
                )
                if ram_critical:
                    self._shed_exit_initiated = True
                    try:
                        self._shed_emergency_graceful_exit(elapsed)
                    except Exception as exc:
                        logger.error("Shedder graceful exit failed: %s", exc)
                else:
                    self._shed_emergency_since_ts = 0.0
        else:
            self._shed_emergency_since_ts = 0.0

    def _gather_system_metrics(self) -> SystemMetrics:
        """Read host CPU/RAM/swap with psutil; report event-loop stall
        sourced from the heartbeat thread."""
        m = SystemMetrics()
        try:
            import psutil
            m.cpu_percent = float(psutil.cpu_percent(interval=None) or 0.0)
            m.ram_percent = float(psutil.virtual_memory().percent or 0.0)
            try:
                m.swap_percent = float(psutil.swap_memory().percent or 0.0)
            except Exception:
                m.swap_percent = 0.0
        except Exception:
            pass
        try:
            m.event_loop_stuck_sec = float(self._shed_event_loop_stuck_sec or 0.0)
        except Exception:
            m.event_loop_stuck_sec = 0.0
        return m

    def _format_shed_action_summary(self, decision) -> str:
        """One-line description of what the shedder is doing."""
        if decision.level == LoadLevel.NORMAL:
            return "all features restored"
        t = decision.throttles or {}
        parts = []
        if "paint_fps" in t:
            parts.append(f"paint {t['paint_fps']}fps")
        if "motion_fps" in t:
            parts.append(f"motion {t['motion_fps']}fps")
        if "detector_fps" in t:
            v = int(t["detector_fps"])
            parts.append("det off" if v == 0 else f"det {v}fps")
        if "depth_fps" in t:
            v = int(t["depth_fps"])
            parts.append("depth off" if v == 0 else f"depth {v}fps")
        return ", ".join(parts)

    def _camera_widgets(self) -> list:
        """All currently-tracked CameraWidget instances (best-effort)."""
        from desktop.widgets.camera import CameraWidget
        result = []
        try:
            for w in list(self.active_widgets or set()):
                try:
                    if isinstance(w, CameraWidget):
                        result.append(w)
                except Exception:
                    continue
        except Exception:
            pass
        return result

    def _select_primary_widgets(self, n: int) -> set:
        """Pick the N widgets to keep fully connected at CRITICAL load.

        Strategy: use the most-recently focused widgets if we have
        enough history, otherwise fall back to "currently visible and
        in the foreground".
        """
        n = max(1, int(n or 1))
        cams = self._camera_widgets()
        primary: list = []
        # Most-recently-focused first
        try:
            for ref in (self._recent_camera_focus or []):
                w = ref() if callable(ref) else ref
                if w in cams and w not in primary:
                    primary.append(w)
                if len(primary) >= n:
                    break
        except Exception:
            pass
        # Then visible widgets to fill up
        if len(primary) < n:
            for w in cams:
                try:
                    if w not in primary and w.isVisible():
                        primary.append(w)
                        if len(primary) >= n:
                            break
                except Exception:
                    continue
        return set(primary[:n])

    def _apply_shed_decision(self, decision) -> None:
        """Push the decision out to all camera widgets and to recording."""
        if self._load_shedder is None:
            return
        cams = self._camera_widgets()
        primary = (
            self._select_primary_widgets(self._load_shedder.primary_widget_count)
            if decision.level >= LoadLevel.CRITICAL
            else set(cams)  # nothing released below CRITICAL
        )

        for w in cams:
            try:
                w.apply_shed_level(
                    int(decision.level),
                    throttles=decision.throttles,
                    is_primary=(w in primary),
                )
            except Exception as exc:
                logger.warning("apply_shed_level failed for widget: %s", exc)

        # Recording is only stopped at EMERGENCY (per policy) unless the
        # user opted out of "protect_recording", in which case we'll
        # also stop it at CRITICAL.
        try:
            should_stop_recording = (
                decision.level == LoadLevel.EMERGENCY
                or (
                    decision.level == LoadLevel.CRITICAL
                    and not self._load_shedder.protect_recording
                )
            )
        except Exception:
            should_stop_recording = (decision.level == LoadLevel.EMERGENCY)

        if should_stop_recording:
            self._shed_pause_recording()
        elif decision.level <= LoadLevel.HIGH and self._shed_recording_paused_cams:
            # Recovered enough -- re-enable any recordings we paused.
            self._shed_resume_recording()

    def _shed_pause_recording(self) -> None:
        """Disable recording on all currently-recording cameras and
        remember which ones we touched so we can restore them later.

        Performs the HTTP calls on a background thread so the GUI never
        blocks; with 9+ cameras a synchronous loop here could freeze
        the UI for tens of seconds at the worst possible moment.
        """
        try:
            with self._recording_status_lock:
                rec = dict(self._recording_status_cache)
        except Exception:
            rec = {}
        to_stop = [cid for cid, on in rec.items()
                   if on and cid not in self._shed_recording_paused_cams]
        if not to_stop:
            return

        # Reserve the cam IDs immediately so a fast bounce of the
        # shedder doesn't issue duplicate stops.
        for cid in to_stop:
            self._shed_recording_paused_cams.add(cid)

        threading.Thread(
            target=self._shed_recording_call_async,
            args=(list(to_stop), False),
            daemon=True,
            name="ShedRecordingPause",
        ).start()

    def _shed_resume_recording(self) -> None:
        """Re-enable recording on cameras the shedder previously paused."""
        cams = list(self._shed_recording_paused_cams)
        if not cams:
            return
        # Optimistically clear; failed re-enables get re-added below.
        self._shed_recording_paused_cams.clear()
        threading.Thread(
            target=self._shed_recording_call_async,
            args=(cams, True),
            daemon=True,
            name="ShedRecordingResume",
        ).start()

    def _shed_recording_call_async(self, cam_ids: list, enable: bool) -> None:
        """Background HTTP loop for pause/resume of recording.

        On failure, re-add to the paused-set so a future tick (or the
        recovery path) can retry.
        """
        for cid in cam_ids:
            ok = False
            try:
                resp = requests.post(
                    f"http://localhost:5000/api/cameras/{cid}/recording",
                    json={"record": bool(enable)}, timeout=5,
                )
                ok = (resp.status_code == 200)
            except Exception as exc:
                logger.warning(
                    "Auto-protection recording %s failed for %s: %s",
                    "resume" if enable else "pause", cid, exc,
                )
                ok = False

            if not enable:
                if ok:
                    logger.warning("Auto-protection: stopped recording on %s", cid)
                else:
                    # Failed to stop; allow a retry on the next critical tick.
                    self._shed_recording_paused_cams.discard(cid)
            else:
                if ok:
                    logger.info("Auto-protection: re-enabled recording on %s", cid)
                else:
                    # Failed to resume; remember so we try again.
                    self._shed_recording_paused_cams.add(cid)

    def _shed_emergency_graceful_exit(self, elapsed_sec: float) -> None:
        """Last-resort: log, save layout, briefly notify the user, then
        ask the app to quit so the OS can reclaim memory."""
        try:
            logger.error(
                "Auto-protection: EMERGENCY held for %.0fs; initiating graceful exit",
                elapsed_sec,
            )
            try:
                logf = Path("data/load_shedder.log")
                logf.parent.mkdir(parents=True, exist_ok=True)
                with open(logf, "a") as f:
                    f.write(
                        f"{datetime.now().isoformat()} EMERGENCY exit after {elapsed_sec:.0f}s "
                        f"reason={self._load_shedder.last_reason if self._load_shedder else 'unknown'}\n"
                    )
            except Exception:
                pass
            try:
                self.tray_icon.showMessage(
                    "Knoxnet VMS Beta - Auto Protection",
                    "Out of memory; restarting to recover. "
                    "Open the app again to continue.",
                    QSystemTrayIcon.MessageIcon.Critical,
                    5000,
                )
            except Exception:
                pass

            def _do_clean_exit():
                try:
                    self._shutdown_runtime_threads()
                except Exception:
                    pass
                try:
                    self.quit()
                except Exception:
                    pass

            QTimer.singleShot(2500, _do_clean_exit)
        except Exception as exc:
            logger.error("Graceful exit failed: %s", exc)
            try:
                self._shutdown_runtime_threads()
            except Exception:
                pass
            try:
                self.quit()
            except Exception:
                pass

    def reload_load_shedder_prefs(self) -> None:
        """Re-read prefs into the shedder (called when sliders change)."""
        if self._load_shedder is None:
            return
        try:
            prefs = self._load_prefs()
            self._load_shedder.reload_prefs(prefs)
        except Exception as exc:
            logger.warning("Failed to reload shedder prefs: %s", exc)

    def force_load_shedder_normal(self) -> None:
        """User-triggered override to immediately drop to NORMAL and
        clear all per-widget shed state.  Resets the warmup so the
        shedder won't immediately re-escalate, and resets the emergency
        exit timer.

        Intended as an escape hatch when the user thinks the shedder
        fired by mistake (e.g. transient stall, transient CPU spike).
        """
        if self._load_shedder is None:
            return
        try:
            self._load_shedder._level = LoadLevel.NORMAL
            self._load_shedder._candidate_level = LoadLevel.NORMAL
            self._load_shedder._candidate_since = time.time()
            self._load_shedder._last_reason = "Manually reset to Normal"
        except Exception:
            pass
        # Clear per-widget shed state on every camera widget.
        try:
            for w in self._camera_widgets():
                try:
                    w.clear_shed_state()
                except Exception:
                    continue
        except Exception:
            pass
        # Resume any recordings the shedder paused.
        try:
            if self._shed_recording_paused_cams:
                self._shed_resume_recording()
        except Exception:
            pass
        # Reset emergency timers and give a fresh warmup window so
        # whatever was driving the spike has time to settle.
        self._shed_emergency_since_ts = 0.0
        self._shed_exit_initiated = False
        self._shed_warmup_until_ts = time.time() + 30.0
        try:
            evt = ShedEvent(
                ts=time.time(),
                from_level=LoadLevel.EMERGENCY,
                to_level=LoadLevel.NORMAL,
                reason="manual reset",
                summary="user clicked Force Normal",
            )
            self._shed_event_log.append(evt)
        except Exception:
            pass
        logger.info("Auto-protection: manually forced to NORMAL by user")

    def get_load_shedder_summary(self) -> dict:
        """Snapshot used by the System Manager UI."""
        if self._load_shedder is None:
            return {
                "enabled": False,
                "level": LoadLevel.NORMAL,
                "level_label": "Normal",
                "machine_class": "mid",
                "thresholds": {},
                "throttles": {},
                "events": [],
                "metrics": SystemMetrics(),
                "protect_recording": True,
                "candidate_level": LoadLevel.NORMAL,
                "candidate_gate_remaining": 0.0,
                "warmup_remaining_sec": 0.0,
            }
        try:
            metrics = getattr(self, "_shed_last_metrics", None) or self._gather_system_metrics()
        except Exception:
            metrics = SystemMetrics()
        try:
            warmup_remaining = max(0.0, float(getattr(self, "_shed_warmup_until_ts", 0.0)) - time.time())
        except Exception:
            warmup_remaining = 0.0
        try:
            candidate = self._load_shedder.candidate_level
            gate_remaining = self._load_shedder.candidate_gate_remaining()
        except Exception:
            candidate = self._load_shedder.current_level
            gate_remaining = 0.0
        return {
            "enabled": self._load_shedder.enabled,
            "level": self._load_shedder.current_level,
            "level_label": self._load_shedder.current_level.label,
            "machine_class": self._load_shedder.machine_class,
            "thresholds": self._load_shedder.thresholds,
            "throttles": self._load_shedder.get_throttles_for_level(
                self._load_shedder.current_level
            ),
            "events": self._shed_event_log.recent(5),
            "metrics": metrics,
            "protect_recording": self._load_shedder.protect_recording,
            "reason": self._load_shedder.last_reason,
            "candidate_level": candidate,
            "candidate_gate_remaining": gate_remaining,
            "warmup_remaining_sec": warmup_remaining,
        }

    def note_camera_widget_focus(self, widget) -> None:
        """Called from CameraWidget on focus/raise to maintain the
        recently-focused list used to pick CRITICAL-mode 'primary'
        cameras.  Kept short to avoid retaining destroyed widgets.
        """
        try:
            if widget is None:
                return
            self._recent_camera_focus = [
                w for w in self._recent_camera_focus if w is not widget
            ]
            self._recent_camera_focus.insert(0, widget)
            self._recent_camera_focus = self._recent_camera_focus[:8]
        except Exception:
            pass

    # ====================================================================
    # End auto-protection load shedder
    # ====================================================================

    def _scheduler_tick(self) -> None:
        try:
            prefs = self._load_prefs()
            enabled = True if ("scheduler_enabled" not in prefs) else bool(prefs.get("scheduler_enabled"))
            if not enabled:
                return
        except Exception:
            return

        # Respect snooze + manual override
        try:
            if self._scheduler_snoozed():
                return
        except Exception:
            pass
        if time.time() < float(getattr(self, "_scheduler_manual_override_until_ts", 0.0) or 0.0):
            return

        now_dt = datetime.now()
        uptime_sec = time.time() - float(getattr(self, "_scheduler_started_at_ts", time.time()))
        schedules = self._scheduler_list()
        matches = []
        for s in schedules:
            try:
                if self._scheduler_match(s, now_dt, uptime_sec):
                    matches.append(s)
            except Exception:
                continue
        chosen = self._scheduler_pick(matches)
        if not chosen:
            return

        sid = str(chosen.get("id") or "").strip()
        layout_id = str(chosen.get("layout_id") or "").strip()
        if not layout_id:
            return

        # Cooldown / spam guard
        try:
            cooldown = int(chosen.get("cooldown_sec", 300) or 0)
        except Exception:
            cooldown = 300
        cooldown = max(0, min(24 * 3600, cooldown))
        last_ts = float((self._scheduler_last_applied_ts or {}).get(sid, 0.0) or 0.0)
        if cooldown and (time.time() - last_ts) < cooldown:
            return
        # Avoid re-applying same layout continuously
        if str(getattr(self, "current_layout_name", None) or "") == layout_id:
            # Still update last run timestamps only when we actually switch.
            return

        action = str(chosen.get("action") or "load").strip().lower()
        policy = str(chosen.get("switch_policy") or "stop").strip().lower()
        if policy not in {"stop", "keep"}:
            # Use default policy if invalid/missing
            try:
                d = prefs.get("layout_switch_policy_default") if isinstance(prefs, dict) else "ask"
                policy = "keep" if d == "keep" else "stop"
            except Exception:
                policy = "stop"

        try:
            if action in {"run", "run_background", "background"}:
                self._layout_start_in_background(layout_id)
            else:
                self._switch_layout_with_decision(layout_id, policy, source="scheduler")
        except Exception:
            return

        # Mark applied
        if sid:
            self._scheduler_last_applied_ts[sid] = time.time()
        self._scheduler_last_layout_id = layout_id

        # Persist last_run_at for UI visibility (best-effort)
        try:
            prefs = self._load_prefs()
            sch = prefs.get("layout_schedules") if isinstance(prefs, dict) else None
            if isinstance(prefs, dict) and isinstance(sch, list):
                out = []
                for s in sch:
                    if isinstance(s, dict) and str(s.get("id") or "") == sid:
                        ss = dict(s)
                        ss["last_run_at"] = datetime.now().isoformat(timespec="seconds")
                        out.append(ss)
                    else:
                        out.append(s)
                prefs["layout_schedules"] = out
                self._save_prefs(prefs)
        except Exception:
            pass

    def prompt_save_layout(self):
        # If we already have a layout name loaded, overwrite without prompting
        if self.current_layout_name:
            try:
                layout_def = self._capture_current_layout_v2(self.current_layout_name, self.current_layout_name)
                if not layout_def.widgets:
                    self.tray_icon.showMessage("Knoxnet VMS Beta", "No widgets to save in layout.", QSystemTrayIcon.MessageIcon.Information)
                    return
                self.layouts_store.upsert_layout(layout_def)
                self.tray_icon.showMessage("Knoxnet VMS Beta", f"Saved layout '{layout_def.name}'", QSystemTrayIcon.MessageIcon.Information)
                return
            except Exception as e:
                logger.warning(f"Failed to save current layout: {e}")

        # New save
        layout_id, ok = QInputDialog.getText(None, "Save Layout", "Layout name (new):")
        if not (ok and layout_id.strip()):
            return
        layout_id = layout_id.strip()
        layout_def = self._capture_current_layout_v2(layout_id, layout_id)
        if not layout_def.widgets:
            self.tray_icon.showMessage("Knoxnet VMS Beta", "No widgets to save in layout.", QSystemTrayIcon.MessageIcon.Information)
            return
        try:
            self.layouts_store.upsert_layout(layout_def)
            self.current_layout_name = layout_id
            self.tray_icon.showMessage("Knoxnet VMS Beta", f"Saved layout '{layout_id}'", QSystemTrayIcon.MessageIcon.Information)
        except Exception as e:
            logger.error(f"Failed to save layout '{layout_id}': {e}")
            self.tray_icon.showMessage("Knoxnet VMS Beta", f"Failed to save layout '{layout_id}'", QSystemTrayIcon.MessageIcon.Warning)

    def _prompt_layout_switch_policy(self, target_layout_id: str, target_layout_name: str | None = None) -> str:
        """
        Decide how to handle currently open widgets when switching layouts.

        Returns:
            "stop" | "keep" | "cancel"
        """
        # If nothing is open, there's nothing to stop/keep.
        if not self.active_widgets:
            return "stop"

        prefs = self._load_prefs()
        overrides = prefs.get("layout_switch_policy_overrides") if isinstance(prefs, dict) else {}
        if not isinstance(overrides, dict):
            overrides = {}

        override = overrides.get(target_layout_id)
        if override in ("stop", "keep"):
            return override

        default_policy = (prefs.get("layout_switch_policy_default") if isinstance(prefs, dict) else None) or "ask"
        if default_policy in ("stop", "keep"):
            return default_policy

        # Ask each time
        try:
            from PySide6.QtWidgets import QCheckBox
            box = QMessageBox()
            box.setWindowTitle("Switch layout")
            nm = target_layout_name or target_layout_id
            box.setText(f"Switch to layout '{nm}'?")
            box.setInformativeText("What should we do with the currently running layout/widgets?")

            remember = QCheckBox("Remember this choice for this layout")
            box.setCheckBox(remember)

            stop_btn = box.addButton("Stop previous", QMessageBox.ButtonRole.AcceptRole)
            keep_btn = box.addButton("Keep running in background", QMessageBox.ButtonRole.ActionRole)
            cancel_btn = box.addButton(QMessageBox.StandardButton.Cancel)
            box.setDefaultButton(stop_btn)
            box.exec()

            clicked = box.clickedButton()
            if clicked == cancel_btn:
                return "cancel"
            decision = "keep" if clicked == keep_btn else "stop"

            if remember.isChecked():
                prefs = self._load_prefs()
                if not isinstance(prefs, dict):
                    prefs = {}
                ov = prefs.get("layout_switch_policy_overrides")
                if not isinstance(ov, dict):
                    ov = {}
                ov[target_layout_id] = decision
                prefs["layout_switch_policy_overrides"] = ov
                self._save_prefs(prefs)

            return decision
        except Exception:
            return "stop"

    def _close_all_widgets(self) -> None:
        """Best-effort teardown of all active widgets."""
        try:
            from desktop.widgets.camera import CameraWidget
        except Exception:
            CameraWidget = None

        for w in list(self.active_widgets):
            try:
                # Best-effort disconnect from global frame signal
                try:
                    if CameraWidget and isinstance(w, CameraWidget):
                        try:
                            self.frame_signal.disconnect(w.receive_frame)
                        except Exception:
                            pass
                except Exception:
                    pass
                try:
                    w.close()
                except Exception:
                    pass
                try:
                    w.deleteLater()
                except Exception:
                    pass
            except Exception:
                continue

        self._current_layout_widgets = []
        self.current_layout_name = None

    def stop_all_sessions(self) -> None:
        """Stop all running sessions and close widgets associated with them (best-effort)."""
        try:
            sessions = self.session_manager.list_sessions() if self.session_manager else []
        except Exception:
            sessions = []

        for s in sessions:
            try:
                if getattr(s, "status", None) == "running":
                    self.session_manager.stop_session(s.id)
            except Exception:
                pass

        for sid, widgets in list((self._session_widgets or {}).items()):
            for w in list(widgets or []):
                try:
                    w.close()
                    w.deleteLater()
                except Exception:
                    pass
            try:
                self._session_widgets.pop(sid, None)
            except Exception:
                pass
        try:
            self._layout_sessions.clear()
        except Exception:
            pass
        try:
            self._layout_paused.clear()
        except Exception:
            pass

    def disconnect_unused_cameras(self) -> None:
        """Disconnect any connected cameras with viewer_refcount == 0 (best-effort)."""
        cm = getattr(self, "camera_manager", None)
        if not cm:
            return
        try:
            snapshot = cm.get_usage_snapshot() if hasattr(cm, "get_usage_snapshot") else {}
        except Exception:
            snapshot = {}
        to_disconnect: list[str] = []
        for cam_id, info in (snapshot or {}).items():
            try:
                if bool(info.get("connected")) and int(info.get("viewer_refcount") or 0) <= 0:
                    to_disconnect.append(str(cam_id))
            except Exception:
                continue
        if not to_disconnect:
            return
        try:
            import asyncio
            if self.camera_loop and self.camera_loop.is_running():
                for cam_id in to_disconnect:
                    asyncio.run_coroutine_threadsafe(cm.disconnect_camera(cam_id), self.camera_loop)
        except Exception:
            pass

    def load_layout(self, layout_id: str):
        """Load a single layout (replaces current open widgets)."""
        # User-initiated action should pause scheduler briefly to avoid "fighting" the user.
        try:
            self._scheduler_note_manual_override()
        except Exception:
            pass

        layout = self.layouts_store.get_layout(layout_id)
        layout_name = layout.name if layout else layout_id

        decision = self._prompt_layout_switch_policy(layout_id, layout_name)
        if decision == "cancel":
            return

        self._switch_layout_with_decision(layout_id, decision, source="user")

    def _switch_layout_with_decision(self, layout_id: str, decision: str, *, source: str = "user") -> None:
        """
        Shared implementation for switching layouts.
        decision: "stop" | "keep" | "cancel"
        source: "user" | "scheduler"
        """
        if decision == "cancel":
            return

        # Explicit load should always show the target layout (even if it was previously hidden).
        try:
            self._set_layout_hidden_persist(str(layout_id), False)
        except Exception:
            pass

        if decision == "keep":
            # Turn current layout widgets into a background session for visibility/control.
            try:
                prev_layout = self.current_layout_name
                prev_widgets = list(self._current_layout_widgets or [])
                if prev_widgets:
                    sess_name = f"Background: {prev_layout}" if prev_layout else "Background: Unsaved"
                    sess_layouts = [prev_layout] if prev_layout else []
                    sess = self.session_manager.create_session(sess_layouts, name=sess_name, meta={"source": "layout_switch_keep"})
                    self.session_manager.start_session(sess.id)
                    self._session_widgets[sess.id] = prev_widgets
                    if prev_layout:
                        self._layout_sessions[str(prev_layout)] = str(sess.id)
                        try:
                            self._layout_paused.discard(str(prev_layout))
                        except Exception:
                            pass
                        # Auto-hide previous layout windows on switch if configured
                        try:
                            ah = self._get_layout_auto_hide(str(prev_layout))
                            if bool(ah.get("on_layout_switch")):
                                self._layout_hide(str(prev_layout), persist=True)
                        except Exception:
                            pass
            except Exception:
                pass
            self._load_layout_internal(layout_id, close_existing=False, set_current=True)
            return

        # Stop previous (safe default): stop background sessions and close all widgets so nothing accumulates.
        try:
            self.stop_all_sessions()
        except Exception:
            pass
        self._close_all_widgets()
        self._load_layout_internal(layout_id, close_existing=False, set_current=True)

    def run_layout(self, layout_id: str):
        """Run a layout without closing existing widgets."""
        self._load_layout_internal(layout_id, close_existing=False, set_current=False)

    def _load_layout_internal(self, layout_id: str, close_existing: bool, set_current: bool):
        layout = self.layouts_store.get_layout(layout_id)
        if not layout:
            # Attempt legacy migration only if migration hasn't been completed yet.
            try:
                self._maybe_migrate_legacy_layouts(force=False)
                layout = self.layouts_store.get_layout(layout_id)
            except Exception:
                layout = None
        if not layout:
            self.tray_icon.showMessage("Knoxnet VMS Beta", f"Layout '{layout_id}' not found.", QSystemTrayIcon.MessageIcon.Warning)
            return

        self.tray_icon.showMessage("Knoxnet VMS Beta", f"Restoring layout '{layout.name}'…", QSystemTrayIcon.MessageIcon.Information)

        if close_existing:
            self._close_all_widgets()

        from desktop.widgets.camera import CameraWidget
        # from desktop.widgets.web import WebWidget
        WebWidget = None
        try:
            from desktop.widgets.terminal import TerminalWidget
        except Exception:
            TerminalWidget = None

        created_widgets = []
        for entry in layout.widgets:
            w = None
            if entry.type == "camera":
                cam_id = entry.camera_id
                if not cam_id or not self._resolve_camera_id(cam_id):
                    logger.warning(f"Camera {cam_id} not found; skipping in layout {layout_id}")
                    continue
                w = CameraWidget(cam_id, camera_manager=self.camera_manager)
                self.frame_signal.connect(w.receive_frame)
                # Tag widgets with layout id so we can apply auto-hide triggers.
                try:
                    setattr(w, "_layout_id", str(layout.id))
                except Exception:
                    pass
                # Auto-hide hooks (best-effort):
                # - shape_triggered carries "source" so we can treat motion vs detections
                try:
                    w.gl_widget.shape_triggered.connect(lambda payload, lid=str(layout.id): self._on_layout_shape_triggered(lid, payload))
                    try:
                        setattr(w, "_shape_triggered_to_app", True)
                    except Exception:
                        pass
                except Exception:
                    pass
                # Apply view-only settings
                self._apply_camera_settings(w, entry.view or {})
            elif entry.type == "web":
                # w = WebWidget(title=entry.get("title", "Web Widget"), url=entry.get("url", "/"))
                logger.warning("Web widgets are currently disabled on this platform.")
                continue
            elif entry.type == "terminal" and TerminalWidget:
                w = TerminalWidget(title=entry.title or "Terminal")
                state = entry.view or {}
                try:
                    if "agent_active" in state:
                        w.agent_active = bool(state.get("agent_active"))
                        w.agent_btn.setChecked(w.agent_active)
                        w.agent_btn.setText("🤖 Agent ON" if w.agent_active else "🤖 Agent OFF")
                        w.agent_btn.setStyleSheet(w._pill_style(w.agent_active))
                    if state.get("text_color"):
                        w.text_color = state["text_color"]
                        w.prompt.setStyleSheet(f"color: {w.text_color}; font-weight: 700;")
                    if state.get("text_scale"):
                        w.text_scale = float(state["text_scale"])
                        w._refresh_log_view()
                except Exception:
                    pass

            if w:
                geom_args = (entry.x or 100, entry.y or 100, entry.w or 640, entry.h or 360)
                w.setGeometry(*geom_args)
                if entry.pinned:
                    w.toggle_pin()
                self._register_widget(w)
                w.show()

                # Move widget to saved workspace/virtual desktop if stored.
                if entry.desktop is not None:
                    try:
                        import subprocess
                        _xid = int(w.winId())
                        subprocess.run(
                            ["xdotool", "set-desktop-for-window", str(_xid), str(entry.desktop)],
                            timeout=2,
                        )
                    except Exception:
                        pass

                # Restore UI extras that depend on realized geometry (audio/PTZ/projections).
                try:
                    if entry.type == "camera":
                        self._restore_camera_extras(w, entry.view or {})
                except Exception:
                    pass
                created_widgets.append(w)
                # On-demand connect for cameras opened from layouts
                try:
                    import asyncio
                    if entry.type == "camera" and self.camera_loop and self.camera_loop.is_running():
                        self._sync_widget_quality_to_config(w)
                        asyncio.run_coroutine_threadsafe(self.camera_manager.acquire_camera(str(entry.camera_id)), self.camera_loop)
                except Exception:
                    pass
        if set_current:
            self.current_layout_name = layout.id
            self._current_layout_widgets = list(created_widgets)
        self.tray_icon.showMessage("Knoxnet VMS Beta", f"Layout '{layout.name}' restored", QSystemTrayIcon.MessageIcon.Information)
        # Apply persisted hidden flag (if any)
        try:
            if str(layout.id) in (self._layout_hidden or set()):
                self._layout_hide(str(layout.id), persist=False)
        except Exception:
            pass
        return created_widgets

    def edit_layout_settings(self):
        """Allow renaming the current layout when one is loaded."""
        if not self.current_layout_name:
            QMessageBox.information(None, "Layout Settings", "No layout is currently active.")
            return

        new_name, ok = QInputDialog.getText(None, "Layout Settings", "Rename layout:", text=self.current_layout_name)
        if not (ok and new_name.strip()):
            return

        new_name = new_name.strip()
        try:
            existing = self.layouts_store.get_layout(self.current_layout_name)
            if not existing:
                QMessageBox.warning(None, "Layout Settings", "Current layout data was not found.")
                return
            # Rename by changing id + name (simple local store approach)
            self.layouts_store.delete_layout(existing.id)
            existing.id = new_name
            existing.name = new_name
            self.layouts_store.upsert_layout(existing)
            self.current_layout_name = new_name
            self.tray_icon.showMessage("Knoxnet VMS Beta", f"Renamed layout to '{new_name}'", QSystemTrayIcon.MessageIcon.Information)
        except Exception as e:
            logger.error(f"Failed to rename layout: {e}")
            QMessageBox.warning(None, "Layout Settings", f"Failed to rename layout: {e}")

    def prompt_run_layouts(self):
        """Create and start a session from one or more layouts, and open their widgets."""
        layouts = self._list_layouts_v2()
        if not layouts:
            QMessageBox.information(None, "Run Layouts", "No saved layouts found.")
            return
        try:
            from PySide6.QtWidgets import QDialog, QVBoxLayout, QListWidget, QListWidgetItem, QDialogButtonBox, QLabel, QLineEdit
            from PySide6.QtCore import Qt
        except Exception as e:
            QMessageBox.warning(None, "Run Layouts", f"UI unavailable: {e}")
            return

        dlg = QDialog()
        dlg.setWindowTitle("Run Layout(s)")
        v = QVBoxLayout(dlg)
        v.addWidget(QLabel("Select one or more layouts to run (opens without closing existing widgets):"))
        name_box = QLineEdit()
        name_box.setPlaceholderText("Optional session name…")
        v.addWidget(name_box)
        lst = QListWidget()
        lst.setSelectionMode(QListWidget.SelectionMode.MultiSelection)
        for l in layouts:
            item = QListWidgetItem(l.name)
            item.setData(Qt.ItemDataRole.UserRole, l.id)
            lst.addItem(item)
        v.addWidget(lst)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        v.addWidget(buttons)
        buttons.accepted.connect(dlg.accept)
        buttons.rejected.connect(dlg.reject)
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return
        selected = [i.data(Qt.ItemDataRole.UserRole) for i in lst.selectedItems()]
        selected = [str(x) for x in selected if x]
        if not selected:
            return
        sess_name = name_box.text().strip() or None
        sess = self.session_manager.create_session(selected, name=sess_name)
        self.session_manager.start_session(sess.id)
        for lid in selected:
            try:
                self._session_widgets.setdefault(sess.id, [])
                created = self._load_layout_internal(lid, close_existing=False, set_current=False) or []
                self._session_widgets[sess.id].extend(list(created))
            except Exception:
                continue

    # ---- Profile helpers: serialise QColor ↔ dict for JSON storage ----

    @staticmethod
    def _serialize_color(color):
        if isinstance(color, QColor):
            return {"r": color.red(), "g": color.green(), "b": color.blue(), "a": color.alpha()}
        return color

    @staticmethod
    def _deserialize_color(val):
        if isinstance(val, dict) and "r" in val:
            return QColor(int(val["r"]), int(val["g"]), int(val["b"]), int(val.get("a", 255)))
        return val

    @staticmethod
    def _serialize_settings_dict(settings: dict) -> dict:
        out = {}
        for k, v in (settings or {}).items():
            if isinstance(v, QColor):
                out[k] = KnoxnetDesktopApp._serialize_color(v)
            else:
                out[k] = v
        return out

    @staticmethod
    def _deserialize_settings_dict(settings: dict) -> dict:
        out = {}
        for k, v in (settings or {}).items():
            if isinstance(v, dict) and "r" in v and "g" in v and "b" in v:
                out[k] = KnoxnetDesktopApp._deserialize_color(v)
            else:
                out[k] = v
        return out

    # ---- Save Profile from Camera ----

    def prompt_save_profile_from_camera(self):
        """Capture a profile from a visible camera widget with fine-grained include/exclude options."""
        from PySide6.QtWidgets import (
            QDialog, QVBoxLayout, QHBoxLayout, QGroupBox, QCheckBox,
            QDialogButtonBox, QLabel, QLineEdit, QComboBox, QScrollArea, QFrame,
        )
        from PySide6.QtCore import Qt

        try:
            from desktop.widgets.camera import CameraWidget
        except Exception:
            QMessageBox.warning(None, "Profiles", "Camera widgets are unavailable.")
            return

        cameras = []
        for w in list(self.active_widgets):
            try:
                if w.isVisible() and isinstance(w, CameraWidget):
                    cameras.append(w)
            except Exception:
                continue
        if not cameras:
            QMessageBox.information(None, "Profiles", "No active camera widgets found.")
            return

        dlg = QDialog()
        dlg.setWindowTitle("Save Profile from Camera")
        dlg.setMinimumWidth(420)
        root = QVBoxLayout(dlg)

        # Camera selector
        root.addWidget(QLabel("Source camera:"))
        cam_combo = QComboBox()
        for w in cameras:
            title = w.windowTitle() or str(getattr(w, "camera_id", "camera"))
            cam_combo.addItem(title)
        root.addWidget(cam_combo)

        # Profile name
        root.addWidget(QLabel("Profile name:"))
        name_edit = QLineEdit()
        name_edit.setPlaceholderText("My Camera Profile")
        root.addWidget(name_edit)

        # Scrollable options area
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        opts_widget = QWidget()
        opts_layout = QVBoxLayout(opts_widget)
        opts_layout.setContentsMargins(0, 0, 0, 0)

        # --- Exclude section (shapes by kind) ---
        exclude_group = QGroupBox("Exclude shape types")
        eg_layout = QVBoxLayout(exclude_group)
        exclude_zones = QCheckBox("Zones")
        exclude_lines = QCheckBox("Lines")
        exclude_tags = QCheckBox("Tags")
        for cb in (exclude_zones, exclude_lines, exclude_tags):
            eg_layout.addWidget(cb)
        opts_layout.addWidget(exclude_group)

        # --- Include section (overlay & monitoring features) ---
        include_group = QGroupBox("Include features")
        ig_layout = QVBoxLayout(include_group)
        inc_motion_style = QCheckBox("Motion box overlay style")
        inc_motion_enabled = QCheckBox("Motion boxes enabled state")
        inc_detection_style = QCheckBox("Detection overlay style")
        inc_detection_enabled = QCheckBox("Object detection enabled state")
        inc_motion_watch = QCheckBox("Motion watch settings")
        inc_debug = QCheckBox("Debug overlay enabled state")
        inc_shapes = QCheckBox("Zones / Lines / Tags (shapes)")
        inc_shapes.setChecked(True)
        for cb in (inc_shapes, inc_motion_style, inc_motion_enabled,
                   inc_detection_style, inc_detection_enabled,
                   inc_motion_watch, inc_debug):
            ig_layout.addWidget(cb)
        opts_layout.addWidget(include_group)

        scroll.setWidget(opts_widget)
        root.addWidget(scroll)

        # Auto-assign checkbox
        auto_assign = QCheckBox("Assign this profile to the source camera immediately")
        auto_assign.setChecked(True)
        root.addWidget(auto_assign)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Save | QDialogButtonBox.StandardButton.Cancel)
        root.addWidget(buttons)
        buttons.accepted.connect(dlg.accept)
        buttons.rejected.connect(dlg.reject)

        if dlg.exec() != QDialog.DialogCode.Accepted:
            return

        name = name_edit.text().strip()
        if not name:
            QMessageBox.warning(None, "Save Profile", "Profile name cannot be empty.")
            return

        idx = cam_combo.currentIndex()
        if idx < 0 or idx >= len(cameras):
            return
        chosen = cameras[idx]
        gl = getattr(chosen, "gl_widget", None)

        # Build profile payload
        overlays: dict = {}
        ai_pipeline: dict = {}
        monitoring_tools: dict = {}

        # Shapes (with exclusion filtering)
        if inc_shapes.isChecked():
            all_shapes = list(getattr(gl, "shapes", []) or [])
            excluded_kinds = set()
            if exclude_zones.isChecked():
                excluded_kinds.add("zone")
            if exclude_lines.isChecked():
                excluded_kinds.add("line")
            if exclude_tags.isChecked():
                excluded_kinds.add("tag")
            if excluded_kinds:
                all_shapes = [s for s in all_shapes if (s.get("kind") or "") not in excluded_kinds]
            if all_shapes:
                overlays["shapes"] = all_shapes

        # Motion overlay style
        if inc_motion_style.isChecked() and gl:
            overlays["motion_settings"] = self._serialize_settings_dict(
                dict(getattr(gl, "motion_settings", {}) or {})
            )

        # Motion boxes enabled
        if inc_motion_enabled.isChecked():
            overlays["motion_boxes_enabled"] = bool(getattr(chosen, "motion_boxes_enabled", False))

        # Detection overlay style
        if inc_detection_style.isChecked() and gl:
            overlays["detection_settings"] = self._serialize_settings_dict(
                dict(getattr(gl, "detection_settings", {}) or {})
            )

        # Object detection enabled
        if inc_detection_enabled.isChecked():
            ai_pipeline["object_detection_enabled"] = bool(getattr(chosen, "desktop_object_detection_enabled", False))

        # Motion watch settings
        if inc_motion_watch.isChecked():
            monitoring_tools["motion_watch_settings"] = dict(getattr(chosen, "motion_watch_settings", {}) or {})

        # Debug overlay
        if inc_debug.isChecked():
            overlays["debug_overlay_enabled"] = bool(getattr(chosen, "debug_overlay_enabled", False))

        if not overlays and not ai_pipeline and not monitoring_tools:
            QMessageBox.information(None, "Save Profile", "No features selected – nothing to save.")
            return

        try:
            prof = self.layouts_store.create_profile(
                name=name,
                overlays=overlays,
                ai_pipeline=ai_pipeline,
                monitoring_tools=monitoring_tools,
                meta={"source": "desktop_camera_widget"},
            )
            if auto_assign.isChecked():
                self.layouts_store.set_assignment(str(chosen.camera_id), prof.id)
            self.tray_icon.showMessage(
                "Knoxnet VMS Beta", f"Saved profile '{name}'",
                QSystemTrayIcon.MessageIcon.Information,
            )
        except Exception as e:
            logger.error(f"Failed to save profile: {e}")
            QMessageBox.warning(None, "Save Profile", f"Failed to save profile: {e}")

    # ---- Apply Profile to Cameras ----

    def prompt_apply_profile_to_cameras(self):
        """Bulk-apply an existing profile to selected (or all) cameras."""
        from PySide6.QtWidgets import (
            QDialog, QVBoxLayout, QHBoxLayout, QListWidget, QListWidgetItem,
            QDialogButtonBox, QLabel, QComboBox, QPushButton,
        )
        from PySide6.QtCore import Qt

        profiles = []
        try:
            profiles = self.layouts_store.list_profiles()
        except Exception:
            profiles = []
        if not profiles:
            QMessageBox.information(None, "Apply Profile", "No profiles found. Save a profile first.")
            return

        dlg = QDialog()
        dlg.setWindowTitle("Apply Profile to Cameras")
        dlg.setMinimumWidth(400)
        v = QVBoxLayout(dlg)

        v.addWidget(QLabel("Profile:"))
        prof_combo = QComboBox()
        for p in profiles:
            prof_combo.addItem(p.name, p.id)
        v.addWidget(prof_combo)

        v.addWidget(QLabel("Select cameras:"))

        # Select All / Deselect All buttons
        btn_row = QHBoxLayout()
        select_all_btn = QPushButton("Select All")
        deselect_all_btn = QPushButton("Deselect All")
        btn_row.addWidget(select_all_btn)
        btn_row.addWidget(deselect_all_btn)
        btn_row.addStretch()
        v.addLayout(btn_row)

        lst = QListWidget()
        lst.setSelectionMode(QListWidget.SelectionMode.MultiSelection)
        cams = getattr(self.camera_manager, "cameras", {}) if self.camera_manager else {}
        for cam_id, cfg in (cams or {}).items():
            label = getattr(cfg, "name", None) or str(cam_id)
            item = QListWidgetItem(label)
            item.setData(Qt.ItemDataRole.UserRole, cam_id)
            lst.addItem(item)
        v.addWidget(lst)

        def _select_all():
            for i in range(lst.count()):
                lst.item(i).setSelected(True)

        def _deselect_all():
            for i in range(lst.count()):
                lst.item(i).setSelected(False)

        select_all_btn.clicked.connect(_select_all)
        deselect_all_btn.clicked.connect(_deselect_all)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        v.addWidget(buttons)
        buttons.accepted.connect(dlg.accept)
        buttons.rejected.connect(dlg.reject)

        if dlg.exec() != QDialog.DialogCode.Accepted:
            return

        profile_id = str(prof_combo.currentData() or "").strip()
        cam_ids = [i.data(Qt.ItemDataRole.UserRole) for i in lst.selectedItems()]
        cam_ids = [str(x) for x in cam_ids if x]
        if not profile_id or not cam_ids:
            return
        try:
            self.layouts_store.bulk_apply_profile(profile_id, cam_ids, mode="replace")
            self.tray_icon.showMessage(
                "Knoxnet VMS Beta",
                f"Applied profile to {len(cam_ids)} camera(s)",
                QSystemTrayIcon.MessageIcon.Information,
            )
            for w in list(self.active_widgets):
                try:
                    from desktop.widgets.camera import CameraWidget
                    if w.isVisible() and isinstance(w, CameraWidget) and str(w.camera_id) in cam_ids:
                        self._apply_assigned_profile_overlays(str(w.camera_id), w)
                except Exception:
                    continue
        except Exception as e:
            logger.error(f"Failed to apply profile: {e}")
            QMessageBox.warning(None, "Apply Profile", f"Failed to apply profile: {e}")

    # ---- Global Overlay Settings ----

    def prompt_global_overlay_settings(self):
        """Change motion and/or detection overlay style for all open camera widgets at once."""
        from PySide6.QtWidgets import (
            QDialog, QVBoxLayout, QHBoxLayout, QGroupBox, QCheckBox,
            QDialogButtonBox, QLabel, QComboBox, QSpinBox, QPushButton,
            QColorDialog, QTabWidget, QWidget, QFormLayout, QSlider,
        )
        from PySide6.QtCore import Qt

        try:
            from desktop.widgets.camera import CameraWidget
        except Exception:
            QMessageBox.warning(None, "Global Overlay", "Camera widgets are unavailable.")
            return

        visible_cameras = [
            w for w in list(self.active_widgets)
            if isinstance(w, CameraWidget) and w.isVisible()
        ]
        if not visible_cameras:
            QMessageBox.information(None, "Global Overlay", "No active camera widgets found.")
            return

        STYLES = ["Box", "Fill", "Corners", "Circle", "Bracket", "Underline", "Crosshair"]
        ANIMATIONS = ["None", "Pulse", "Flash", "Glitch", "Rainbow"]
        DETECTION_ANIMATIONS = ["None", "Pulse", "Flash", "Glitch", "Rainbow", "Glow"]

        dlg = QDialog()
        dlg.setWindowTitle("Global Overlay Settings")
        dlg.setMinimumWidth(440)
        root = QVBoxLayout(dlg)

        tabs = QTabWidget()
        root.addWidget(tabs)

        # Track user-chosen colours (mutable in closures)
        motion_color_holder = [QColor(255, 0, 0)]
        detection_color_holder = [QColor(0, 255, 255)]

        # Grab defaults from first visible camera
        ms0: dict = {}
        ds0: dict = {}
        if visible_cameras:
            gl0 = getattr(visible_cameras[0], "gl_widget", None)
            if gl0:
                ms0 = dict(getattr(gl0, "motion_settings", {}) or {})
                ds0 = dict(getattr(gl0, "detection_settings", {}) or {})
                if isinstance(ms0.get("color"), QColor):
                    motion_color_holder[0] = QColor(ms0["color"])
                if isinstance(ds0.get("color"), QColor):
                    detection_color_holder[0] = QColor(ds0["color"])

        # ---- Motion tab ----
        motion_tab = QWidget()
        mf = QFormLayout(motion_tab)

        motion_apply_cb = QCheckBox("Apply motion overlay changes")
        motion_apply_cb.setChecked(True)
        mf.addRow(motion_apply_cb)

        m_style = QComboBox()
        m_style.addItems(STYLES)
        cur_style = ms0.get("style", "Box")
        if cur_style in STYLES:
            m_style.setCurrentText(cur_style)
        mf.addRow("Style:", m_style)

        m_color_btn = QPushButton()
        m_color_btn.setStyleSheet(f"background-color: {motion_color_holder[0].name()};")
        m_color_btn.setFixedHeight(28)

        def pick_motion_color():
            c = QColorDialog.getColor(motion_color_holder[0], dlg, "Motion Overlay Color")
            if c.isValid():
                motion_color_holder[0] = c
                m_color_btn.setStyleSheet(f"background-color: {c.name()};")

        m_color_btn.clicked.connect(pick_motion_color)
        mf.addRow("Color:", m_color_btn)

        m_thickness = QSpinBox()
        m_thickness.setRange(1, 10)
        m_thickness.setValue(int(ms0.get("thickness", 2)))
        mf.addRow("Thickness:", m_thickness)

        m_anim = QComboBox()
        m_anim.addItems(ANIMATIONS)
        cur_anim = ms0.get("animation", "None")
        if cur_anim in ANIMATIONS:
            m_anim.setCurrentText(cur_anim)
        mf.addRow("Animation:", m_anim)

        m_trails = QCheckBox("Show trails")
        m_trails.setChecked(bool(ms0.get("trails", False)))
        mf.addRow(m_trails)

        m_color_speed = QCheckBox("Color by speed")
        m_color_speed.setChecked(bool(ms0.get("color_speed", False)))
        mf.addRow(m_color_speed)

        mf.addRow(QLabel(""))  # spacer

        m_sens_label = QLabel(f"Sensitivity: {int(ms0.get('sensitivity', 50))}")
        m_sens = QSlider(Qt.Orientation.Horizontal)
        m_sens.setRange(1, 100)
        m_sens.setValue(int(ms0.get("sensitivity", 50)))
        m_sens.valueChanged.connect(lambda v: m_sens_label.setText(f"Sensitivity: {v}"))
        mf.addRow(m_sens_label, m_sens)

        m_merge = QSpinBox()
        m_merge.setRange(0, 500)
        m_merge.setValue(int(ms0.get("merge_size", 0)))
        m_merge.setSuffix(" px")
        mf.addRow("Merge size:", m_merge)

        tabs.addTab(motion_tab, "Motion Overlay")

        # ---- Detection tab ----
        det_tab = QWidget()
        df = QFormLayout(det_tab)

        det_apply_cb = QCheckBox("Apply detection overlay changes")
        det_apply_cb.setChecked(True)
        df.addRow(det_apply_cb)

        d_style = QComboBox()
        d_style.addItems(STYLES)
        cur_ds = ds0.get("style", "Box")
        if cur_ds in STYLES:
            d_style.setCurrentText(cur_ds)
        df.addRow("Style:", d_style)

        d_color_btn = QPushButton()
        d_color_btn.setStyleSheet(f"background-color: {detection_color_holder[0].name()};")
        d_color_btn.setFixedHeight(28)

        def pick_det_color():
            c = QColorDialog.getColor(detection_color_holder[0], dlg, "Detection Overlay Color")
            if c.isValid():
                detection_color_holder[0] = c
                d_color_btn.setStyleSheet(f"background-color: {c.name()};")

        d_color_btn.clicked.connect(pick_det_color)
        df.addRow("Color:", d_color_btn)

        d_thickness = QSpinBox()
        d_thickness.setRange(1, 10)
        d_thickness.setValue(int(ds0.get("thickness", 2)))
        df.addRow("Thickness:", d_thickness)

        d_anim = QComboBox()
        d_anim.addItems(DETECTION_ANIMATIONS)
        cur_da = ds0.get("animation", "None")
        if cur_da in DETECTION_ANIMATIONS:
            d_anim.setCurrentText(cur_da)
        df.addRow("Animation:", d_anim)

        d_labels = QCheckBox("Show labels")
        d_labels.setChecked(bool(ds0.get("show_labels", True)))
        df.addRow(d_labels)

        d_label_size = QSpinBox()
        d_label_size.setRange(6, 32)
        d_label_size.setValue(int(ds0.get("label_font_size", 10)))
        df.addRow("Label font size:", d_label_size)

        tabs.addTab(det_tab, "Detection Overlay")

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Apply | QDialogButtonBox.StandardButton.Cancel)
        root.addWidget(buttons)
        buttons.rejected.connect(dlg.reject)

        def _apply():
            applied = 0
            for w in list(self.active_widgets):
                try:
                    if not isinstance(w, CameraWidget) or not w.isVisible():
                        continue
                    gl = getattr(w, "gl_widget", None)
                    if not gl:
                        continue

                    if motion_apply_cb.isChecked():
                        ms = getattr(gl, "motion_settings", {})
                        ms["style"] = m_style.currentText()
                        ms["color"] = QColor(motion_color_holder[0])
                        ms["thickness"] = m_thickness.value()
                        ms["animation"] = m_anim.currentText()
                        ms["trails"] = m_trails.isChecked()
                        ms["color_speed"] = m_color_speed.isChecked()
                        ms["sensitivity"] = m_sens.value()
                        ms["merge_size"] = m_merge.value()
                        gl.motion_settings = ms

                    if det_apply_cb.isChecked():
                        ds = getattr(gl, "detection_settings", {})
                        ds["style"] = d_style.currentText()
                        ds["color"] = QColor(detection_color_holder[0])
                        ds["thickness"] = d_thickness.value()
                        ds["animation"] = d_anim.currentText()
                        ds["show_labels"] = d_labels.isChecked()
                        ds["label_font_size"] = d_label_size.value()
                        gl.detection_settings = ds

                    try:
                        gl.update()
                    except Exception:
                        pass
                    applied += 1
                except Exception:
                    continue

            self.tray_icon.showMessage(
                "Knoxnet VMS Beta",
                f"Updated overlay settings on {applied} camera(s)",
                QSystemTrayIcon.MessageIcon.Information,
            )
            dlg.accept()

        buttons.button(QDialogButtonBox.StandardButton.Apply).clicked.connect(_apply)

        dlg.exec()

    # --- Camera config refresh ---
    def check_camera_updates(self):
        if not self.cameras_json_path.exists():
            return
        try:
            current_mtime = self.cameras_json_path.stat().st_mtime
            if self.cameras_json_mtime is None or current_mtime > self.cameras_json_mtime:
                self.cameras_json_mtime = current_mtime
                self.refresh_cameras_from_json()
        except Exception as e:
            logger.error(f"Failed to check camera updates: {e}")

    def _maybe_initial_sync_from_json(self):
        """
        Perform one-time sync from cameras.json to CameraManager DB when the loop is ready.
        This makes existing cameras (added via React/API) available immediately in the tray menu.
        """
        if self._initial_sync_done:
            self._initial_sync_timer.stop()
            return

        if not self.camera_loop:
            return

        try:
            self.refresh_cameras_from_json()
            self._initial_sync_done = True
            self._initial_sync_timer.stop()
        except Exception as e:
            logger.error(f"Initial camera sync failed: {e}")

    def refresh_cameras_from_json(self, on_complete=None):
        """
        Pull latest cameras from JSON/DB so UI reflects emulator changes.
        Optionally invoke a callback (on the Qt thread) once refresh is done.
        """
        if not self.camera_loop:
            return None

        import asyncio

        async def _do_refresh():
            # Prefer API -> DB sync (backend may be in Docker with its own cameras.json),
            # fall back to local JSON -> DB when API isn't reachable.
            api_synced = 0
            try:
                api_synced = await self.camera_manager.sync_cameras_api_to_db("http://localhost:5000/api")
            except Exception:
                api_synced = 0
            if not api_synced:
                await self.camera_manager.sync_cameras_json_to_db()
            # Desktop-light: do not auto-connect all cameras on refresh.
            # Active widgets will acquire cameras on-demand.

        future = asyncio.run_coroutine_threadsafe(_do_refresh(), self.camera_loop)
        future.add_done_callback(lambda f: logger.info("Camera list refreshed from JSON"))

        if on_complete:
            def _finish(_):
                try:
                    QTimer.singleShot(0, on_complete)
                except Exception as e:
                    logger.warning(f"refresh_cameras_from_json on_complete failed: {e}")
            future.add_done_callback(_finish)

        return future

def main():
    app = KnoxnetDesktopApp(sys.argv)
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
