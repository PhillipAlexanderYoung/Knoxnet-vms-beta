import os
import sys

# Eventlet greendns pulls dnspython async backends (e.g. dns.asyncbackend) which are
# easy to miss in frozen/PyInstaller builds and can crash the backend at import time.
# Knoxnet's backend workload is primarily local traffic (localhost/LAN), so we default
# to disabling greendns for reliability. Users can override by explicitly setting it.
os.environ.setdefault("EVENTLET_NO_GREENDNS", "yes")

# IMPORTANT (Windows/Python 3.13+):
# eventlet.monkey_patch() can cause HTTP servers to accept TCP connections but hang on responses.
# Default to a "simple server" mode on Python 3.13+, or when explicitly requested.
_FORCE_SIMPLE_SERVER = str(os.environ.get("KNOXNET_SIMPLE_SERVER", "")).lower() in {"1", "true", "yes"} or sys.version_info >= (3, 13)

if not _FORCE_SIMPLE_SERVER:
    import eventlet
    eventlet.monkey_patch()

from flask import Flask, request, jsonify, Response
from flask_cors import CORS
import json
import re
import requests
from datetime import datetime
import uuid
import logging
import asyncio
import time
from dotenv import load_dotenv
from typing import Any, Dict, Optional, List
from pathlib import Path
from urllib.parse import urlparse, urlunparse
from core.python_script_manager import PythonScriptManager
from core.entitlements import get_camera_limit, entitlement_summary
from core.version import get_version
from core.storage_manager import StorageManager
import threading
import socket

# SSH support
try:
    import paramiko  # type: ignore
    PARAMIKO_AVAILABLE = True
except Exception as _ssh_e:
    print(f"Paramiko not available: {_ssh_e}")
    PARAMIKO_AVAILABLE = False

# Import API routes
try:
    from api.routes import register_routes
    from api.proxy_routes import proxy_bp
    API_ROUTES_AVAILABLE = True
except ImportError as e:
    print(f"API routes not available: {e}")
    API_ROUTES_AVAILABLE = False

# Load environment variables from .env file
load_dotenv()

# Import AI agent
try:
    from core.ai_agent import AIAgent, AIContext
    from dataclasses import asdict
    AI_AGENT_AVAILABLE = True
except ImportError as e:
    print(f"AI Agent not available: {e}")
    AI_AGENT_AVAILABLE = False

# Import LLM service manager
try:
    from core.llm_service_manager import LocalLLMService
    LLM_SERVICE_AVAILABLE = True
except ImportError as e:
    print(f"LLM Service Manager not available: {e}")
    LLM_SERVICE_AVAILABLE = False

# Import camera auto-recovery
try:
    from core.camera_auto_recovery import CameraAutoRecovery
    from core.camera_bootstrap import CameraBootstrap
    AUTO_RECOVERY_AVAILABLE = True
except ImportError as e:
    print(f"Camera auto-recovery not available: {e}")
    AUTO_RECOVERY_AVAILABLE = False

# Import websocket connection management
try:
    from core.motion_detection_integration import MotionDetectionIntegration
    from api.websocket_routes import init_websocket_routes
    WEBSOCKET_MANAGEMENT_AVAILABLE = True
except ImportError as e:
    print(f"WebSocket management not available: {e}")
    WEBSOCKET_MANAGEMENT_AVAILABLE = False

# Shared background loop for async tasks
_background_loop = None
_background_loop_lock = threading.Lock()

def _ensure_background_loop():
    global _background_loop
    if _background_loop and _background_loop.is_running():
        return _background_loop

    with _background_loop_lock:
        if _background_loop and _background_loop.is_running():
            return _background_loop

        loop_ready = threading.Event()

        def _loop_worker():
            global _background_loop
            try:
                try:
                    loop = asyncio.get_running_loop()
                except RuntimeError:
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                
                _background_loop = loop
                loop_ready.set()
                
                if not loop.is_running():
                    try:
                        loop.run_forever()
                    except RuntimeError as e:
                        # If loop is already running, that's fine
                        logger.info(f"Background loop attached: {e}")
            except Exception as e:
                logger.error(f"Background loop failed: {e}")
                if not loop_ready.is_set():
                    loop_ready.set()

        # Start loop worker in a DAEMON thread so it doesn't block shutdown
        threading.Thread(target=_loop_worker, name="app-bg-loop", daemon=True).start()

        loop_ready.wait(timeout=5)
        return _background_loop

def _run_coro_safe(coro):
    """Helper to run a coroutine safely without nesting event loops."""
    try:
        try:
            current_loop = asyncio.get_running_loop()
        except RuntimeError:
            current_loop = None

        target_loop = None

        # Prefer the stream server loop if it exists and is running
        if 'STREAM_SERVER_GLOBAL' in globals() and STREAM_SERVER_GLOBAL:
            ss = STREAM_SERVER_GLOBAL
            if hasattr(ss, '_loop') and ss._loop and ss._loop.is_running():
                target_loop = ss._loop

        # If no target yet, use the shared background loop
        if target_loop is None:
            target_loop = _ensure_background_loop()

        # If we're already inside a running loop on this thread, we can't safely block waiting
        # for that same loop. Run the coroutine in a fresh OS thread with its own event loop.
        if current_loop and current_loop.is_running() and target_loop is current_loop:
            import eventlet
            real_threading = eventlet.patcher.original('threading')
            holder = {"result": None, "error": None}

            def _runner():
                try:
                    holder["result"] = asyncio.run(coro)
                except Exception as e:
                    holder["error"] = e

            t = real_threading.Thread(target=_runner, daemon=True)
            t.start()
            timeout_s = float(os.environ.get("AI_ROUTE_TIMEOUT", "35"))
            t.join(timeout=timeout_s + 2.0)
            if t.is_alive():
                raise TimeoutError(f"Async execution exceeded {timeout_s:.0f}s")
            if holder["error"]:
                raise holder["error"]
            return holder["result"]

        future = asyncio.run_coroutine_threadsafe(coro, target_loop)
        timeout_s = float(os.environ.get("AI_ROUTE_TIMEOUT", "35"))
        return future.result(timeout=timeout_s)
    except Exception as e:
        logger.error(f"Async execution failed: {e}")
        return None

app = Flask(__name__)
CORS(app)

# --- Realtime (Socket.IO) ---
try:
    from flask_socketio import SocketIO, Namespace, emit, join_room, leave_room
    # Allow best available async mode (eventlet/gevent) and configure engine options
    # Optimized for lowest latency real-time motion detection
    _async_mode = 'threading' if _FORCE_SIMPLE_SERVER else 'eventlet'
    _transports = ['polling'] if _FORCE_SIMPLE_SERVER else ['websocket', 'polling']
    socketio: Optional[SocketIO] = SocketIO(
        app,
        cors_allowed_origins="*",
        path="/socket.io",
        ping_interval=25,
        ping_timeout=60,
        max_http_buffer_size=1_000_000,
        logger=False,
        engineio_logger=False,
        async_mode=_async_mode,
        transports=_transports,
        allow_upgrades=(False if _FORCE_SIMPLE_SERVER else True),
        always_connect=True,
        manage_session=True,   # Enable session management
        cookie=None,
        cors_credentials=False
    )
    WS_NS = '/realtime'
except Exception as _e:
    print(f"Socket.IO not available: {_e}")
    socketio = None
else:
    # Make SocketIO accessible to blueprints via current_app.extensions["socketio"]
    try:
        app.extensions = getattr(app, "extensions", {}) or {}
        app.extensions["socketio"] = socketio
    except Exception:
        pass

# Configure logging - both file and console
import os
os.makedirs('logs', exist_ok=True)

# Create formatters and handlers
log_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# File handler with rotation (force UTF-8 to avoid Windows encoding issues)
from logging.handlers import RotatingFileHandler
file_handler = RotatingFileHandler(
    'logs/knoxnet.log', mode='a', encoding='utf-8',
    maxBytes=10 * 1024 * 1024, backupCount=5,
)
file_handler.setFormatter(log_formatter)
file_handler.setLevel(logging.DEBUG)

# Console handler with sanitizer filter to strip non-ASCII for Windows consoles
console_handler = logging.StreamHandler()
console_handler.setFormatter(log_formatter)
console_handler.setLevel(logging.INFO)

# Sanitize console logs to avoid UnicodeEncodeError on Windows cp1252 consoles
class _ConsoleSanitizerFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:  # type: ignore[override]
        try:
            # Ensure message is ASCII-safe for Windows console
            msg = str(record.getMessage())
            safe = msg.encode('cp1252', errors='ignore').decode('cp1252', errors='ignore')
            # Overwrite the message safely
            record.msg = safe
            record.args = ()
        except Exception:
            # Best-effort: drop through
            pass
        return True

# Suppress urllib3 connection pool warnings to reduce noise during startup
logging.getLogger("urllib3.connectionpool").setLevel(logging.ERROR)

console_handler.addFilter(_ConsoleSanitizerFilter())

# Configure root logger
logging.basicConfig(
    level=logging.DEBUG,
    handlers=[file_handler, console_handler]
)

logger = logging.getLogger(__name__)

# Simple in-memory storage (use a database in production)
cameras_db = []
# Prefer data/cameras.json (desktop edits) but keep cameras.json (repo root) in sync for legacy code paths.
LEGACY_CAMERAS_FILE = 'cameras.json'
cameras_file = 'data/cameras.json' if os.path.exists('data/cameras.json') else LEGACY_CAMERAS_FILE


def _enabled_cameras() -> List[Dict[str, Any]]:
    return [c for c in cameras_db if c.get("enabled", True)]


def _sanitize_fs_name(name: str) -> str:
    """Sanitize a string for safe use as a filesystem directory name."""
    import re
    s = str(name or "").strip()
    s = re.sub(r'[<>:"/\\|?*\x00-\x1f]', '_', s)
    s = s.rstrip('. ')
    if s.startswith('.'):
        s = '_' + s[1:]
    return s[:120] or "unknown"


def _build_record_path(camera: dict) -> str:
    """Build an absolute MediaMTX ``recordPath`` for a camera.

    Structure:  {base}/{CameraName}/%Y-%m-%d/%H-%M-%S-%f_%path

    MediaMTX requires ``%path`` in recordPath (it resolves to the stream path
    name, which is a UUID).  We use the human-readable camera name as the
    top-level directory so users see friendly folder names when browsing
    recordings, and append ``%path`` to the filename so MediaMTX's playback
    server can still locate segments for a given stream.

    Per-camera ``recording_dir`` overrides the global base directory.
    """
    cam_name = _sanitize_fs_name(camera.get('name') or camera.get('id', 'unknown'))

    custom_dir = (camera.get('recording_dir') or '').strip()
    if custom_dir:
        base = str(Path(custom_dir).expanduser().resolve()).replace("\\", "/")
    else:
        from core.paths import get_recordings_dir
        base = str(get_recordings_dir()).replace("\\", "/")

    return f"{base}/{cam_name}/%Y-%m-%d/%H-%M-%S-%f_%path"


def _build_recording_payload(camera: dict, enable: bool) -> dict:
    """Build the full MediaMTX path payload for a recording toggle.

    When **enabling** recording we must guarantee MediaMTX actually has a
    live stream to write to disk — otherwise ``record: yes`` is a no-op
    and the user gets no .mp4 file.  Two MediaMTX behaviours make this
    surprisingly easy to get wrong:

      * If ``source`` is omitted (or set to ``publisher``) MediaMTX waits
        for a client to push a stream in.  Nothing in the system pushes
        unless the user happens to have a camera widget open.
      * If ``sourceOnDemand: true`` MediaMTX only connects to the camera
        when there is an active *reader*.  ``record`` does NOT count as
        a reader, so an on-demand path with recording enabled stays cold.

    To make "Record All" actually work for every camera regardless of
    whether the user has its widget open, we always write back a full
    path config:  pull from the camera RTSP, keep the connection open
    (``sourceOnDemand: false``), and set ``recordPath``.

    When **disabling** recording we restore the on-demand defaults so we
    don't pin an RTSP session to every camera 24/7 just for live view.
    """
    rtsp_url = (camera.get("rtsp_url") or "").strip()
    payload: dict = {"record": bool(enable)}

    if enable:
        payload["recordPath"] = _build_record_path(camera)
        if rtsp_url:
            payload["source"] = rtsp_url
            payload["sourceProtocol"] = "tcp"
        payload["sourceOnDemand"] = False
    else:
        if rtsp_url:
            payload["source"] = rtsp_url
            payload["sourceProtocol"] = "tcp"
        payload["sourceOnDemand"] = True
        payload["sourceOnDemandStartTimeout"] = "10s"
        payload["sourceOnDemandCloseAfter"] = "30s"

    return payload


def _mediamtx_yml_path() -> Path:
    """Resolve the active MediaMTX config YAML path."""
    project_root = Path(__file__).resolve().parent
    mtx_dir = project_root / "mediamtx"
    compat = mtx_dir / "mediamtx_compat.yml"
    return compat if compat.exists() else mtx_dir / "mediamtx.yml"


def _toggle_recording_via_yml(camera_id: str, camera: dict, enable: bool) -> bool:
    """Toggle recording by writing directly to mediamtx.yml.

    MediaMTX v1.17 has an API bug where PATCH/POST always fails with
    ``'recordPath' must contain %path'`` even when the value is correct.
    This workaround edits the YAML config file directly; MediaMTX watches
    its config file and hot-reloads changes automatically.
    """
    import re

    yml = _mediamtx_yml_path()
    if not yml.exists():
        logger.error("MediaMTX config not found at %s", yml)
        return False

    try:
        text = yml.read_text(encoding="utf-8")
    except Exception as exc:
        logger.error("Failed to read %s: %s", yml, exc)
        return False

    # We need to add/update a path entry under the top-level ``paths:`` key.
    # The YAML structure is:
    #   paths:
    #     all_others:
    #       ...
    #     <camera_id>:
    #       source: rtsp://...
    #       sourceProtocol: tcp
    #       record: yes
    #       recordPath: <path>
    #
    # Rather than using a full YAML parser (which would strip comments and
    # reformat the file), we do a targeted text-based insertion/update.

    record_path = _build_record_path(camera)
    record_val = "yes" if enable else "no"
    rtsp_url = (camera.get("rtsp_url") or "").strip()

    # Build the path block.  We MUST include ``source`` here:  if it's
    # omitted MediaMTX falls back to the ``all_others`` default which is
    # ``source: publisher`` and just waits for someone to push a stream
    # in — nothing does that automatically, so ``record: yes`` would
    # silently produce zero recordings.
    #
    # We also pin ``sourceOnDemand`` based on whether recording is on:
    #   * recording on  -> sourceOnDemand: no   (keep the camera connection
    #                                            open so we always have data
    #                                            to record, regardless of
    #                                            whether anyone is viewing)
    #   * recording off -> sourceOnDemand: yes  (restore on-demand mode so
    #                                            we don't pin an RTSP session
    #                                            to every camera 24/7)
    lines = [f"  {camera_id}:"]
    if rtsp_url:
        lines.append(f"    source: {rtsp_url}")
        lines.append(f"    sourceProtocol: tcp")
    if enable:
        lines.append(f"    sourceOnDemand: no")
    else:
        lines.append(f"    sourceOnDemand: yes")
        lines.append(f"    sourceOnDemandStartTimeout: 10s")
        lines.append(f"    sourceOnDemandCloseAfter: 30s")
    lines.append(f"    record: {record_val}")
    if enable:
        lines.append(f"    recordPath: {record_path}")
    path_block = "\n".join(lines)

    # Check if this camera_id already exists in the paths section
    # Pattern: "  <camera_id>:" at the start of a line (2-space indent)
    cam_pattern = re.compile(
        rf'^(  {re.escape(camera_id)}:\s*\n)'     # header
        rf'((?:    .+\n)*)',                        # indented body lines
        re.MULTILINE,
    )
    match = cam_pattern.search(text)
    if match:
        new_text = text[:match.start()] + path_block + "\n" + text[match.end():]
    else:
        # Append after the last line of the file (which should be inside paths:)
        # Ensure there's a trailing newline before our block
        if not text.endswith("\n"):
            text += "\n"
        new_text = text + "\n" + path_block + "\n"

    try:
        yml.write_text(new_text, encoding="utf-8")
        logger.info("Wrote recording config for %s to %s (record=%s)", camera_id, yml, record_val)
        return True
    except Exception as exc:
        logger.error("Failed to write %s: %s", yml, exc)
        return False


def _camera_limit_state() -> Dict[str, Any]:
    enabled = _enabled_cameras()
    limit = get_camera_limit()
    allowed = enabled[:limit]
    return {
        "limit": limit,
        "enabled": enabled,
        "allowed": allowed,
        "allowed_ids": [c.get("id") for c in allowed if c.get("id")],
    }


def _is_camera_allowed(camera_id: str) -> bool:
    if not camera_id:
        return False
    return camera_id in _camera_limit_state()["allowed_ids"]

# Car counting rules storage
car_counting_rules = {}
car_count_history = {}

# AI Agent instance
ai_agent = None

_WEB_UI_DISABLED = True  # Desktop-only beta: web UI is disabled.

# MediaMTX Configuration
# Global reference to stream server for MJPEG endpoint
STREAM_SERVER_GLOBAL = None
# Global reference to camera manager (some endpoints expect it)
CAMERA_MANAGER_GLOBAL = None
MEDIAMTX_API_URL = os.environ.get('MEDIAMTX_API_URL', 'http://localhost:9997/v3')
MEDIAMTX_WEBRTC_URL = os.environ.get('MEDIAMTX_WEBRTC_URL', 'http://localhost:8889')
MEDIAMTX_HLS_URL = os.environ.get('MEDIAMTX_HLS_URL', 'http://localhost:8888')
MEDIAMTX_RTSP_URL = os.environ.get('MEDIAMTX_RTSP_URL', 'rtsp://localhost:8554')
MEDIAMTX_API_USERNAME = os.environ.get('MEDIAMTX_API_USERNAME', 'admin')
MEDIAMTX_API_PASSWORD = os.environ.get('MEDIAMTX_API_PASSWORD', '')
# Some shells/users escape "$" as "$$" when setting environment variables.
# If we detect a double-dollar prefix, normalize it back.
try:
    if isinstance(MEDIAMTX_API_PASSWORD, str) and MEDIAMTX_API_PASSWORD.startswith('$$'):
        MEDIAMTX_API_PASSWORD = MEDIAMTX_API_PASSWORD.lstrip('$')
except Exception:
    pass

# Ensure downstream modules that read directly from the environment (e.g. core.mediamtx_client)
# see the normalized password value.
try:
    os.environ['MEDIAMTX_API_PASSWORD'] = str(MEDIAMTX_API_PASSWORD)
except Exception:
    pass

python_script_manager: Optional[PythonScriptManager] = None


def _normalize_stream_path(value: Optional[str]) -> Optional[str]:
    if not value:
        return None
    value = str(value).strip()
    if not value:
        return None
    # Strip protocol portions if someone passed a full RTSP URL
    if value.lower().startswith('rtsp://'):
        try:
            parsed = urlparse(value)
            path = parsed.path or '/'
            if parsed.query:
                path += f"?{parsed.query}"
            return path
        except Exception:
            return None
    if not value.startswith('/'):
        value = f"/{value}"
    return value


def _extract_stream_path_from_rtsp(rtsp_url: Optional[str]) -> Optional[str]:
    if not rtsp_url:
        return None
    try:
        parsed = urlparse(rtsp_url)
        path = parsed.path or '/'
        if parsed.query:
            path += f"?{parsed.query}"
        return path
    except Exception:
        return None


def _transform_to_substream_path(path: Optional[str]) -> Optional[str]:
    if not path:
        return '/media/video2'
    normalized = _normalize_stream_path(path) or '/media/video2'
    if re.search(r'/media/video\d+', normalized, flags=re.IGNORECASE):
        return re.sub(r'/media/video\d+', '/media/video2', normalized, flags=re.IGNORECASE)
    if re.search(r'subtype=\d+', normalized, flags=re.IGNORECASE):
        return re.sub(r'subtype=\d+', 'subtype=1', normalized, flags=re.IGNORECASE)
    if re.search(r'stream1', normalized, flags=re.IGNORECASE):
        return re.sub(r'stream1', 'stream2', normalized, flags=re.IGNORECASE)
    if re.search(r'/Streaming/Channels/\d+', normalized, flags=re.IGNORECASE):
        return re.sub(r'/Streaming/Channels/\d+', '/Streaming/Channels/102', normalized, flags=re.IGNORECASE)
    if 'channel=1' in normalized.lower() and 'subtype=' not in normalized.lower():
        separator = '&' if '?' in normalized else '?'
        return f"{normalized}{separator}subtype=1"
    return '/media/video2'


def _build_rtsp_with_path(rtsp_url: Optional[str], new_path: Optional[str]) -> Optional[str]:
    if not rtsp_url or not new_path:
        return None
    try:
        parsed = urlparse(rtsp_url)
        path_only = new_path
        query = ''
        if '?' in new_path:
            path_only, query = new_path.split('?', 1)
        if not path_only:
            path_only = '/'
        updated = parsed._replace(path=path_only, query=query)
        return urlunparse(updated)
    except Exception:
        return None


def _normalize_stream_priority(value: Optional[str]) -> str:
    raw = str(value or "").strip().lower()
    if raw == "sub":
        return "sub"
    return "main"


def _normalize_stream_quality(priority: str, value: Optional[str]) -> str:
    raw = str(value or "").strip().lower()
    if priority == "sub":
        return "low"
    if raw in {"medium", "high", "ultra"}:
        return raw
    return "medium"


def _normalize_privacy_mask(value: Any) -> List[Dict[str, Any]]:
    return value if isinstance(value, list) else []


def _build_rtsp_url_from_camera_fields(camera: Dict[str, Any]) -> Optional[str]:
    rtsp_url = str(camera.get('rtsp_url') or '').strip()
    if camera.get('custom_rtsp'):
        return rtsp_url or None

    ip_address = str(camera.get('ip_address') or camera.get('ip') or '').strip()
    if not ip_address:
        return rtsp_url or None

    try:
        port = int(camera.get('port') or 554)
    except Exception:
        port = 554

    stream_path = (
        _normalize_stream_path(camera.get('stream_path')) or
        _extract_stream_path_from_rtsp(rtsp_url) or
        '/media/video1'
    )
    camera['stream_path'] = stream_path

    username = str(camera.get('username') or '').strip()
    password = camera.get('password')
    if username:
        auth = f"{username}:{password}@" if password not in (None, "") else f"{username}@"
        return f"rtsp://{auth}{ip_address}:{port}{stream_path}"
    return f"rtsp://{ip_address}:{port}{stream_path}"


def _apply_camera_payload(camera: Dict[str, Any], data: Dict[str, Any]) -> bool:
    connection_changed = False

    for field in (
        'name', 'location', 'manufacturer', 'protocol', 'ptz_url', 'backup_rtsp_url',
        'custom_rtsp', 'ai_analysis', 'recording', 'ptz_enabled', 'motion_detection',
        'audio_enabled', 'night_vision', 'webrtc_enabled', 'recording_dir',
    ):
        if field in data:
            camera[field] = data[field]

    if 'privacy_mask' in data:
        camera['privacy_mask'] = _normalize_privacy_mask(data.get('privacy_mask'))
    else:
        camera['privacy_mask'] = _normalize_privacy_mask(camera.get('privacy_mask'))

    if 'ip_address' in data or 'ip' in data:
        new_ip = data.get('ip_address') or data.get('ip')
        camera['ip_address'] = new_ip
        camera['ip'] = new_ip
        connection_changed = True

    if 'username' in data:
        camera['username'] = data['username']
        connection_changed = True

    if 'password' in data:
        camera['password'] = data['password']
        connection_changed = True

    if 'port' in data:
        camera['port'] = data['port']
        connection_changed = True

    if 'rtsp_url' in data:
        camera['rtsp_url'] = data.get('rtsp_url')
        connection_changed = True

    if 'stream_path' in data:
        camera['stream_path'] = _normalize_stream_path(data.get('stream_path'))
        connection_changed = True
    elif not camera.get('stream_path'):
        camera['stream_path'] = _extract_stream_path_from_rtsp(camera.get('rtsp_url')) or '/media/video1'

    if 'substream_path' in data:
        camera['substream_path'] = _normalize_stream_path(data.get('substream_path'))

    if 'substream_rtsp_url' in data:
        substream_rtsp_url = str(data.get('substream_rtsp_url') or '').strip()
        camera['substream_rtsp_url'] = substream_rtsp_url or None

    if 'mediamtx_path' in data:
        camera['mediamtx_path'] = data.get('mediamtx_path') or None

    if 'mediamtx_sub_path' in data:
        camera['mediamtx_sub_path'] = data.get('mediamtx_sub_path') or None

    if 'enabled' in data:
        camera['enabled'] = bool(data['enabled'])
        camera['status'] = 'live' if camera['enabled'] else 'offline'

    priority_source: Optional[str]
    if 'stream_priority' in data:
        priority_source = data.get('stream_priority')
    elif 'stream_quality' in data:
        raw_quality = str(data.get('stream_quality') or '').strip().lower()
        priority_source = 'sub' if raw_quality == 'low' else 'main'
    else:
        priority_source = camera.get('stream_priority')

    stream_priority = _normalize_stream_priority(priority_source)
    camera['stream_priority'] = stream_priority
    camera['stream_quality'] = _normalize_stream_quality(
        stream_priority,
        data.get('stream_quality', camera.get('stream_quality'))
    )

    if 'custom_rtsp' in data:
        connection_changed = True

    if connection_changed:
        rebuilt_rtsp_url = _build_rtsp_url_from_camera_fields(camera)
        if rebuilt_rtsp_url:
            camera['rtsp_url'] = rebuilt_rtsp_url

    camera['type'] = 'camera'
    camera['device_type'] = 'camera'
    camera['updated_at'] = datetime.now().isoformat()
    return connection_changed


def ensure_camera_stream_metadata(camera: Dict[str, Any]) -> None:
    if not isinstance(camera, dict) or not camera.get('id'):
        return

    stream_priority = _normalize_stream_priority(
        camera.get('stream_priority') or ('sub' if str(camera.get('stream_quality') or '').strip().lower() == 'low' else 'main')
    )
    camera['stream_priority'] = stream_priority
    camera['stream_quality'] = _normalize_stream_quality(stream_priority, camera.get('stream_quality'))

    base_stream_path = (
        _normalize_stream_path(camera.get('stream_path')) or
        _extract_stream_path_from_rtsp(camera.get('rtsp_url')) or
        '/media/video1'
    )
    camera['stream_path'] = base_stream_path
    camera.setdefault('mediamtx_path', camera.get('id'))

    explicit_substream_path = 'substream_path' in camera
    explicit_substream_rtsp_url = 'substream_rtsp_url' in camera

    substream_path = _normalize_stream_path(camera.get('substream_path'))
    substream_rtsp_url = str(camera.get('substream_rtsp_url') or '').strip() or None

    if not substream_path and substream_rtsp_url:
        substream_path = _normalize_stream_path(_extract_stream_path_from_rtsp(substream_rtsp_url))

    if (
        not substream_path and
        not substream_rtsp_url and
        not explicit_substream_path and
        not explicit_substream_rtsp_url
    ):
        substream_path = _transform_to_substream_path(base_stream_path)

    if substream_path and not substream_rtsp_url:
        substream_rtsp_url = _build_rtsp_with_path(camera.get('rtsp_url'), substream_path)

    if substream_path or substream_rtsp_url:
        camera['substream_path'] = substream_path
        camera['substream_rtsp_url'] = substream_rtsp_url
        if substream_rtsp_url:
            sub_path_name = camera.get('mediamtx_sub_path') or f"{camera.get('mediamtx_path') or camera['id']}_sub"
            camera['mediamtx_sub_path'] = sub_path_name
            camera['webrtc_sub_whep_url'] = f"/proxy/webrtc/{sub_path_name}/whep"
            camera['hls_sub_url'] = f"/proxy/hls/{sub_path_name}/index.m3u8"
        else:
            camera.pop('mediamtx_sub_path', None)
            camera.pop('webrtc_sub_whep_url', None)
            camera.pop('hls_sub_url', None)
    else:
        camera['substream_path'] = None
        camera['substream_rtsp_url'] = None
        camera.pop('mediamtx_sub_path', None)
        camera.pop('webrtc_sub_whep_url', None)
        camera.pop('hls_sub_url', None)


def _test_rtsp_stream(rtsp_url: str, timeout: int = 5) -> bool:
    if not rtsp_url:
        return False
    try:
        import cv2  # type: ignore
    except Exception:
        logger.warning("OpenCV not available for RTSP testing; returning optimistic success")
        return True
    cap = None
    try:
        cap = cv2.VideoCapture(rtsp_url)
        if not cap.isOpened():
            return False
        start = time.time()
        ok, _ = cap.read()
        while not ok and (time.time() - start) < timeout:
            ok, _ = cap.read()
        return bool(ok)
    except Exception as e:
        logger.error(f"RTSP test failed for {rtsp_url}: {e}")
        return False
    finally:
        try:
            if cap:
                cap.release()
        except Exception:
            pass

class MediaMTXClient:
    """Simple MediaMTX client for managing streams with connection pooling"""

    def __init__(self):
        self.api_url = MEDIAMTX_API_URL
        self.username = MEDIAMTX_API_USERNAME
        self.password = MEDIAMTX_API_PASSWORD
        self.auth = requests.auth.HTTPBasicAuth(self.username, self.password) if self.username and self.password else None
        
        # Use a session for connection pooling
        self.session = requests.Session()
        if self.auth:
            self.session.auth = self.auth
            
        # Configure retry strategy on the session
        from requests.adapters import HTTPAdapter
        from urllib3.util.retry import Retry
        
        retry_strategy = Retry(
            total=3,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["HEAD", "GET", "OPTIONS", "POST", "PATCH", "DELETE"]
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)

    def create_path(self, path_name: str, rtsp_source: str) -> bool:
        """Create or update a MediaMTX path with an RTSP source"""
        try:
            # When source is an RTSP URL, MediaMTX pulls from it (sourceOnDemand works)
            # When source is "publisher", something pushes TO MediaMTX (sourceOnDemand is invalid)
            path_config = {
                "name": path_name,
                "source": rtsp_source,
                "sourceProtocol": "tcp",
                "sourceOnDemand": True,
                "sourceOnDemandStartTimeout": "10s",
                "sourceOnDemandCloseAfter": "30s"
            }

            # Use session instead of direct requests
            response = self.session.post(
                f"{self.api_url}/config/paths/add/{path_name}",
                json=path_config,
                timeout=10
            )

            if response.status_code in [200, 201]:
                logger.info(f"[SUCCESS] Created MediaMTX path {path_name} -> {rtsp_source}")
                return True
            if response.status_code == 400 and 'already exists' in response.text.lower():
                patch_response = self.session.patch(
                    f"{self.api_url}/config/paths/patch/{path_name}",
                    json=path_config,
                    timeout=10
                )
                if patch_response.status_code == 200:
                    logger.info(f"[SUCCESS] Updated existing MediaMTX path {path_name}")
                    return True
                logger.warning(f"[WARN] Failed to update existing MediaMTX path {path_name}: {patch_response.status_code}")
                return False

            logger.error(f"[ERROR] Failed to create MediaMTX path {path_name}: {response.status_code} - {response.text}")
            return False

        except requests.exceptions.RequestException as e:
            # Log simple error message without full traceback for common connection errors
            logger.error(f"[ERROR] Error creating MediaMTX path {path_name}: {str(e)}")
            return False

    def delete_path(self, path_name: str) -> bool:
        """Delete a path from MediaMTX"""
        try:
            response = self.session.delete(
                f"{self.api_url}/config/paths/delete/{path_name}",
                timeout=10
            )

            if response.status_code in [200, 204]:
                logger.info(f"✅ Deleted MediaMTX path {path_name}")
                return True
            else:
                logger.warning(f"⚠️ Failed to delete MediaMTX path {path_name}: {response.status_code}")
                return False

        except requests.exceptions.RequestException as e:
            logger.error(f"❌ Error deleting MediaMTX path {path_name}: {e}")
            return False

    def get_path_info(self, camera_id: str) -> dict:
        """Get path information from MediaMTX"""
        try:
            response = self.session.get(
                f"{self.api_url}/paths/list",
                timeout=10
            )

            if response.status_code == 200:
                data = response.json()
                paths = data.get('items', [])
                for path in paths:
                    if path.get('name') == camera_id:
                        return path

            return {}

        except requests.exceptions.RequestException as e:
            logger.error(f"❌ Error getting MediaMTX path info: {e}")
            return {}

    def test_connection(self, silent: bool = False) -> bool:
        """Test connection to MediaMTX"""
        try:
            if silent:
                # Use a fresh session without retries for silent checks to avoid logging warnings
                # We purposefully do not use self.session here as it has retry logic attached
                with requests.Session() as s:
                    response = s.get(
                        f"{self.api_url}/paths/list",
                        timeout=2  # Short timeout for quick check
                    )
            else:
                response = self.session.get(
                    f"{self.api_url}/paths/list",
                    timeout=5
                )
            return response.status_code == 200
        except requests.exceptions.RequestException as e:
            if not silent:
                logger.error(f"❌ MediaMTX connection test failed: {e}")
            return False

    def get_ice_servers(self) -> list:
        """Get ICE servers for WebRTC"""
        # Return default STUN servers
        return [
            {'urls': 'stun:stun.l.google.com:19302'},
            {'urls': 'stun:stun1.l.google.com:19302'}
        ]


# Initialize MediaMTX client
mediamtx = MediaMTXClient()

# Storage manager (disk-space auto-cleanup)

def _pause_all_recordings():
    """Emergency callback: disable recording on all cameras via MediaMTX."""
    import requests as _req
    mtx_base = MEDIAMTX_API_URL.rstrip("/")
    for cam in cameras_db:
        if not cam.get("recording"):
            continue
        cam_id = cam.get("id", "")
        if not cam_id:
            continue
        try:
            _req.patch(
                f"{mtx_base}/config/paths/patch/{cam_id}",
                json={"record": False}, timeout=5,
            )
        except Exception:
            pass
    logger.warning("StorageManager: all recordings paused (emergency)")


def _resume_all_recordings():
    """Emergency callback: re-enable recording on cameras that were recording."""
    import requests as _req
    mtx_base = MEDIAMTX_API_URL.rstrip("/")
    for cam in cameras_db:
        if not cam.get("recording"):
            continue
        cam_id = cam.get("id", "")
        if not cam_id:
            continue
        try:
            payload = _build_recording_payload(cam, True)
            _req.patch(
                f"{mtx_base}/config/paths/patch/{cam_id}",
                json=payload, timeout=5,
            )
        except Exception:
            pass
    logger.info("StorageManager: recordings resumed after emergency cleanup")


def _get_custom_recording_dirs():
    """Return per-camera recording_dir overrides as Path objects."""
    from pathlib import Path as _P
    dirs = []
    seen = set()
    for cam in cameras_db:
        custom = (cam.get("recording_dir") or "").strip()
        if not custom:
            continue
        p = _P(custom).expanduser().resolve()
        key = str(p)
        if key not in seen and p.exists():
            dirs.append(p)
            seen.add(key)
    return dirs


storage_manager = StorageManager(
    pause_recordings_cb=_pause_all_recordings,
    resume_recordings_cb=_resume_all_recordings,
    get_recording_dirs_cb=_get_custom_recording_dirs,
)
storage_manager.start()


# ---------------------------------------------------------------------------
# Recording watchdog -- keeps MediaMTX alive and re-enables lost recordings
# ---------------------------------------------------------------------------
def _recording_watchdog_loop():
    """Background thread that monitors MediaMTX health and recording state."""
    import requests as _req
    import time as _t

    consecutive_failures = 0
    while True:
        _t.sleep(15)
        try:
            resp = _req.get("http://127.0.0.1:9997/v3/config/global/get", timeout=3)
            if resp.status_code == 200:
                consecutive_failures = 0
            else:
                consecutive_failures += 1
        except Exception:
            consecutive_failures += 1

        if consecutive_failures >= 2:
            logger.warning("MediaMTX unreachable for %d checks – attempting restart", consecutive_failures)
            if _try_restart_mediamtx():
                consecutive_failures = 0
            else:
                continue

        # Reconcile: ensure cameras flagged recording=True still have record=true
        # in MediaMTX (catches silent config loss after a MediaMTX restart).
        try:
            mtx_base = MEDIAMTX_API_URL.rstrip("/")
            for cam in cameras_db:
                if not cam.get('recording'):
                    continue
                cam_id = cam.get('id', '')
                if not cam_id:
                    continue
                try:
                    payload = _build_recording_payload(cam, True)
                    cfg_resp = _req.get(f"{mtx_base}/config/paths/get/{cam_id}", timeout=3)
                    if cfg_resp.status_code == 200:
                        path_cfg = cfg_resp.json()
                        if not path_cfg.get("record"):
                            logger.warning("Recording lost for %s in MediaMTX – re-enabling", cam_id)
                            _req.patch(f"{mtx_base}/config/paths/patch/{cam_id}", json=payload, timeout=5)
                    elif cfg_resp.status_code == 404:
                        logger.warning("Path missing for recording camera %s – re-creating", cam_id)
                        _req.post(f"{mtx_base}/config/paths/add/{cam_id}", json=payload, timeout=5)
                except Exception as inner_err:
                    logger.warning("Watchdog reconcile error for %s: %s", cam_id, inner_err)
        except Exception as e:
            logger.warning("Recording watchdog reconcile sweep failed: %s", e)


_recording_watchdog_thread = threading.Thread(target=_recording_watchdog_loop, daemon=True, name="RecordingWatchdog")
_recording_watchdog_thread.start()


# Optional: Optimized Stream Server (runs alongside the legacy StreamServer for migration)
try:
    from core.optimized_stream_server import OptimizedStreamServer  # type: ignore
    _OPTIMIZED_SERVER_AVAILABLE = True
except Exception as _e:
    print(f"OptimizedStreamServer not available: {_e}")
    _OPTIMIZED_SERVER_AVAILABLE = False


def _try_restart_mediamtx() -> bool:
    """Best-effort restart of MediaMTX when it's unreachable. Returns True if it came back up."""
    import shutil
    import subprocess as _sp
    import time as _t

    # First check if it's actually already running and reachable
    try:
        import requests as _req
        resp = _req.get("http://127.0.0.1:9997/v3/config/global/get", timeout=2)
        if resp.status_code == 200:
            return True
    except Exception:
        pass

    project_root = Path(__file__).resolve().parent
    mtx_bin = None
    for candidate in [
        project_root / "mediamtx" / "mediamtx",
        Path(shutil.which("mediamtx") or "/nonexistent"),
    ]:
        if candidate.exists() and candidate.is_file():
            mtx_bin = candidate.resolve()
            break

    if not mtx_bin:
        logger.warning("Cannot restart MediaMTX: binary not found")
        return False

    mtx_dir = mtx_bin.parent
    compat = mtx_dir / "mediamtx_compat.yml"
    cfg = compat if compat.exists() else mtx_dir / "mediamtx.yml"

    logger.info("Attempting MediaMTX restart: %s %s", mtx_bin, cfg)

    try:
        _sp.Popen(
            [str(mtx_bin), str(cfg)],
            stdout=open("/tmp/mediamtx.log", "a"),
            stderr=_sp.STDOUT,
            cwd=str(mtx_dir),
            start_new_session=True,
        )
        _t.sleep(3)
        import requests as _req
        resp = _req.get("http://127.0.0.1:9997/v3/config/global/get", timeout=2)
        if resp.status_code == 200:
            logger.info("MediaMTX restarted successfully")
            return True
    except Exception as e:
        logger.warning("MediaMTX restart failed: %s", e)
    return False


# Load cameras from file if it exists
def load_cameras():
    global cameras_db
    if os.path.exists(cameras_file):
        try:
            with open(cameras_file, 'r') as f:
                cameras_db = json.load(f)
                logger.info(f"📂 Loaded {len(cameras_db)} cameras from file")
        except Exception as e:
            logger.error(f"Error loading cameras: {e}")
            cameras_db = []

    # Keep legacy cameras.json in sync (many codepaths still read it directly).
    try:
        if cameras_file != LEGACY_CAMERAS_FILE and isinstance(cameras_db, list) and cameras_db:
            try:
                import shutil
                shutil.copy2(cameras_file, LEGACY_CAMERAS_FILE)
            except Exception:
                # fallback: rewrite JSON
                with open(LEGACY_CAMERAS_FILE, 'w', encoding='utf-8') as f:
                    json.dump(cameras_db, f, indent=2)
            logger.info(f"📂 Synced legacy camera file: {LEGACY_CAMERAS_FILE}")
    except Exception:
        pass


# Save cameras to file (atomic with backup)
def save_cameras():
    """Persist cameras_db to cameras.json safely with backup and atomic replace."""
    try:
        import tempfile
        import shutil

        # Create backup of current JSON file if it exists
        if os.path.exists(cameras_file):
            try:
                shutil.copy2(cameras_file, f"{cameras_file}.backup")
            except Exception as _e:
                logger.warning(f"⚠️ Failed to create backup of {cameras_file}: {_e}")

        # Write to a temp file first, then atomically replace
        dir_name = os.path.dirname(cameras_file) or "."
        fd, tmp_path = tempfile.mkstemp(prefix="cameras.", suffix=".json.tmp", dir=dir_name)
        try:
            with os.fdopen(fd, 'w', encoding='utf-8') as tmp_f:
                json.dump(cameras_db, tmp_f, indent=2)
                tmp_f.flush()
                os.fsync(tmp_f.fileno())
            os.replace(tmp_path, cameras_file)
        finally:
            try:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
            except Exception:
                pass

        logger.info(f"💾 Saved {len(cameras_db)} cameras to {cameras_file}")

        # Also keep legacy cameras.json updated for older code paths.
        try:
            if cameras_file != LEGACY_CAMERAS_FILE:
                try:
                    shutil.copy2(cameras_file, LEGACY_CAMERAS_FILE)
                except Exception:
                    with open(LEGACY_CAMERAS_FILE, 'w', encoding='utf-8') as f:
                        json.dump(cameras_db, f, indent=2)
        except Exception:
            pass
    except Exception as e:
        logger.error(f"Error saving cameras to {cameras_file}: {e}")


# Load cameras on startup
load_cameras()
for cam in cameras_db:
    ensure_camera_stream_metadata(cam)

# Sync with MediaMTX on startup to get active cameras
def initialize_camera_sync():
    """Initialize camera sync with MediaMTX on startup - disabled to prevent URL modifications"""
    logger.info("🔄 Camera sync disabled to prevent RTSP URL modifications")
    # Disabled to prevent any automatic URL changes
    # try:
    #     logger.info("🔄 Initializing camera sync with MediaMTX...")
    #     sync_mediamtx_cameras()
    #     logger.info("✅ Camera sync initialization completed")
    # except Exception as e:
    #     logger.warning(f"⚠️ Camera sync initialization failed: {e}")

# Initialize camera sync on startup
initialize_camera_sync()

# Resolve camera by id or name (case-insensitive, substring allowed)
def resolve_camera_ref(camera_ref: str) -> Optional[Dict[str, Any]]:
    try:
        if not camera_ref:
            return None
        # Exact ID match
        for cam in cameras_db:
            if cam.get('id') == camera_ref:
                return cam
        # Exact name (case-insensitive)
        lower_ref = str(camera_ref).strip().lower()
        for cam in cameras_db:
            name = str(cam.get('name', '')).strip().lower()
            if name == lower_ref:
                return cam
        # Substring match on name
        for cam in cameras_db:
            name = str(cam.get('name', '')).strip().lower()
            if lower_ref and lower_ref in name:
                return cam
        return None
    except Exception:
        return None


# Initialize Python automation script manager
if python_script_manager is None:
    try:
        script_dir = Path(os.environ.get('PYTHON_SCRIPT_PATH', 'data/python_scripts'))
        python_script_manager = PythonScriptManager(
            script_dir,
            camera_resolver=lambda camera_id: resolve_camera_ref(camera_id)
        )
        logger.info("✅ Python script manager initialized at %s", script_dir)
    except Exception as exc:
        python_script_manager = None
        logger.error("❌ Failed to initialize python script manager: %s", exc)

def ensure_cameras_connected_for_detection():
    """Ensure all cameras are connected to stream server for motion detection"""
    try:
        if 'STREAM_SERVER_GLOBAL' in globals() and STREAM_SERVER_GLOBAL is not None:
            stream_server = STREAM_SERVER_GLOBAL
            
            # Get all cameras from the database
            cameras_to_connect = []
            for camera in cameras_db:
                if camera.get('enabled', True) and camera.get('motion_detection', True):
                    cameras_to_connect.append(camera)
            
            logger.info(f"Ensuring {len(cameras_to_connect)} cameras are connected for motion detection...")
            
            import asyncio
            for camera in cameras_to_connect:
                camera_id = camera.get('id')
                rtsp_url = camera.get('rtsp_url')
                
                if not camera_id or not rtsp_url:
                    continue
                
                # Check if camera is already connected to stream server
                if hasattr(stream_server, 'active_streams') and camera_id not in stream_server.active_streams:
                    logger.info(f"Connecting camera {camera.get('name', camera_id)} to stream server for motion detection...")
                    
                    try:
                        # Start stream in background
                        success = _run_coro_safe(
                            stream_server.start_stream(camera_id, {
                                'rtsp_url': rtsp_url,
                                'webrtc_enabled': camera.get('webrtc_enabled', True),
                                'fps': 15  # Lower FPS for motion detection
                            })
                        )
                        if success:
                            logger.info(f"✅ Camera {camera.get('name', camera_id)} connected for motion detection")
                        else:
                            logger.warning(f"⚠️ Failed to connect camera {camera.get('name', camera_id)} for motion detection")
                    except Exception as e:
                        logger.error(f"❌ Error connecting camera {camera.get('name', camera_id)} for motion detection: {e}")
                else:
                    logger.debug(f"Camera {camera.get('name', camera_id)} already connected to stream server")
            
            logger.info("Camera connection check completed")
        else:
            logger.warning("Stream server not available for camera connection check")
            
    except Exception as e:
        logger.error(f"Error ensuring cameras are connected for detection: {e}")

# Ensure cameras are connected for motion detection on startup
# DISABLED: We don't want to eagerly connect all cameras on startup.
# Motion detection should be enabled on-demand or only for enabled cameras if configured explicitly.
# ensure_cameras_connected_for_detection()

def fetch_mediamtx_cameras():
    """Fetch real cameras from MediaMTX API"""
    try:
        logger.info("Fetching cameras from MediaMTX API...")
        
        # Get active paths from MediaMTX API
        response = requests.get(f"{MEDIAMTX_API_URL}/paths/list", timeout=10)
        
        if response.status_code == 200:
            paths_data = response.json()
            logger.info(f"Found MediaMTX response: {type(paths_data)}")
            
            cameras = []
            
            # Handle the correct MediaMTX v3 API response format
            if isinstance(paths_data, dict) and 'items' in paths_data:
                # New v3 format: {"itemCount": X, "pageCount": Y, "items": [...]}
                for path_info in paths_data.get('items', []):
                    if isinstance(path_info, dict):
                        path_name = path_info.get("name", "")
                        is_ready = path_info.get("ready", False)
                        
                        # Skip system paths and only include ready paths
                        if path_name and is_ready and path_name not in ['all_cameras', 'test', 'fallback']:
                            # Create camera object from MediaMTX path
                            camera = {
                                "id": path_name,
                                "name": path_name.replace('_', ' ').replace('-', ' ').title(),
                                "ip_address": "MediaMTX",
                                "port": 554,
                                "username": "admin",
                                "password": "",
                                "location": path_name.replace('_', ' ').replace('-', ' ').title(),
                                "status": "online" if is_ready else "offline",
                                "rtsp_url": f"{MEDIAMTX_RTSP_URL}/{path_name}",
                                "webrtc_enabled": True,
                                "hls_url": f"/proxy/hls/{path_name}/index.m3u8",
                                "webrtc_url": f"/proxy/webrtc/{path_name}",
                                "webrtc_whep_url": f"/proxy/webrtc/{path_name}/whep",
                                "created_at": "2024-01-01T00:00:00Z",
                                "updated_at": datetime.now().isoformat(),
                                "source": path_info.get("source", {}).get("type", "rtspSource"),
                                "mediamtx_path": path_name,
                                "ready": is_ready,
                                "readers_count": len(path_info.get("readers", [])),
                                "publishers_count": 1 if path_info.get("source", {}).get("type") else 0,
                                "mediamtx_ready": is_ready
                            }
                            cameras.append(camera)
                            logger.info(f"Found active camera: {camera['name']} (path: {path_name})")
            
            elif isinstance(paths_data, dict):
                # Legacy format: {path_name: path_info}
                for path_name, path_info in paths_data.items():
                    # Skip system paths
                    if path_name in ['all_cameras', 'test', 'fallback']:
                        continue
                    
                    # Only include paths that are ready/active
                    if isinstance(path_info, dict) and path_info.get("ready", False):
                        # Create camera object from MediaMTX path
                        camera = {
                            "id": path_name,
                            "name": path_name.replace('_', ' ').replace('-', ' ').title(),
                            "ip_address": "MediaMTX",
                            "port": 554,
                            "username": "admin",
                            "password": "",
                            "location": path_name.replace('_', ' ').replace('-', ' ').title(),
                            "status": "online",
                            "rtsp_url": f"{MEDIAMTX_RTSP_URL}/{path_name}",
                            "webrtc_enabled": True,
                            "hls_url": f"/proxy/hls/{path_name}/index.m3u8",
                            "webrtc_url": f"/proxy/webrtc/{path_name}",
                            "webrtc_whep_url": f"/proxy/webrtc/{path_name}/whep",
                            "created_at": "2024-01-01T00:00:00Z",
                            "updated_at": datetime.now().isoformat(),
                            "source": path_info.get("source", "publisher"),
                            "mediamtx_path": path_name,
                            "ready": path_info.get("ready", False),
                            "readers_count": path_info.get("readers_count", 0),
                            "publishers_count": path_info.get("publishers_count", 0)
                        }
                        cameras.append(camera)
                        logger.info(f"Found active camera: {camera['name']} (path: {path_name})")
            
            elif isinstance(paths_data, list):
                # Old format: [path_info]
                for path_info in paths_data:
                    if isinstance(path_info, dict):
                        path_name = path_info.get("name", "")
                        if path_name and path_info.get("ready", False):
                            camera = {
                                "id": path_name,
                                "name": path_name.replace('_', ' ').replace('-', ' ').title(),
                                "ip_address": "MediaMTX",
                                "port": 554,
                                "username": "admin",
                                "password": "",
                                "location": path_name.replace('_', ' ').replace('-', ' ').title(),
                                "status": "online",
                                "rtsp_url": f"{MEDIAMTX_RTSP_URL}/{path_name}",
                                "webrtc_enabled": True,
                                "hls_url": f"/proxy/hls/{path_name}/index.m3u8",
                                "webrtc_url": f"/proxy/webrtc/{path_name}",
                                "webrtc_whep_url": f"/proxy/webrtc/{path_name}/whep",
                                "created_at": "2024-01-01T00:00:00Z",
                                "updated_at": datetime.now().isoformat(),
                                "source": path_info.get("source", "publisher"),
                                "mediamtx_path": path_name,
                                "ready": path_info.get("ready", False),
                                "readers_count": path_info.get("readers_count", 0),
                                "publishers_count": path_info.get("publishers_count", 0)
                            }
                            cameras.append(camera)
                            logger.info(f"Found active camera: {camera['name']} (path: {path_name})")
            
            else:
                logger.warning(f"Unexpected MediaMTX response format: {type(paths_data)}")
            
            logger.info(f"✅ Successfully fetched {len(cameras)} active cameras from MediaMTX")
            return cameras
        else:
            logger.warning(f"Failed to fetch MediaMTX paths: {response.status_code} - {response.text}")
            return []
            
    except Exception as e:
        logger.error(f"Error fetching MediaMTX cameras: {e}")
        return []

def sync_mediamtx_cameras():
    """Sync cameras from MediaMTX to our database"""
    global cameras_db
    
    try:
        logger.info("Syncing cameras from MediaMTX...")
        mediamtx_cameras = fetch_mediamtx_cameras()
        
        if mediamtx_cameras:
            # Create a map of MediaMTX paths to camera info
            mediamtx_map = {cam['id']: cam for cam in mediamtx_cameras}
            
            # Update existing cameras with MediaMTX status
            for camera in cameras_db:
                if camera['id'] in mediamtx_map:
                    # Update status based on MediaMTX
                    camera['status'] = 'online' if mediamtx_map[camera['id']]['ready'] else 'offline'
                    camera['mediamtx_ready'] = mediamtx_map[camera['id']]['ready']
                    camera['readers_count'] = mediamtx_map[camera['id']]['readers_count']
                    camera['publishers_count'] = mediamtx_map[camera['id']]['publishers_count']
                    logger.info(f"✅ Camera {camera['name']} is {'online' if camera['status'] == 'online' else 'offline'} in MediaMTX")
                else:
                    # Camera not found in MediaMTX
                    camera['status'] = 'offline'
                    camera['mediamtx_ready'] = False
                    logger.info(f"⚠️ Camera {camera['name']} not found in MediaMTX")
            
            logger.info(f"✅ Updated {len(cameras_db)} cameras with MediaMTX status")
        else:
            logger.warning("No cameras found in MediaMTX, marking all cameras as offline")
            for camera in cameras_db:
                camera['status'] = 'offline'
                camera['mediamtx_ready'] = False
            
    except Exception as e:
        logger.error(f"Error syncing MediaMTX cameras: {e}")


# Add debugging middleware
@app.before_request
def log_request():
    # Skip logging for health, test, and high-frequency stream endpoints
    skip_endpoints = ['health', 'test', 'static']
    if request.endpoint in skip_endpoints:
        return
        
    # Skip logging for stream and snapshot requests to prevent spam
    if request.path.startswith('/api/cameras/') and ('/stream' in request.path or '/snapshot' in request.path):
        return
        
    if request.path.startswith('/assets/'):
        return

    logger.info(f"{request.method} {request.path}")


# Initialize AI Agent
def initialize_ai_agent():
    """Initialize the AI agent if available"""
    global ai_agent
    if AI_AGENT_AVAILABLE:
        try:
            ai_agent = AIAgent()
            logger.info(f"[SUCCESS] AI Agent initialized: {ai_agent.get_status()}")
            return True
        except Exception as e:
            logger.error(f"[ERROR] Failed to initialize AI Agent: {e}")
            ai_agent = None
            return False
    else:
        logger.warning("[WARNING] AI Agent not available")
        return False


# Initialize LLM Service
llm_service = None

def initialize_llm_service():
    """Initialize the local LLM service if available"""
    global llm_service
    if LLM_SERVICE_AVAILABLE:
        try:
            port = int(os.environ.get("LLM_SERVICE_PORT", "8102"))
            host = os.environ.get("LLM_SERVICE_HOST", "127.0.0.1")
            managed_mode = os.environ.get("LLM_SERVICE_MANAGED_MODE", "internal").lower()
            manage_process = managed_mode not in {"external", "remote"}
            llm_service = LocalLLMService(port=port, host=host, manage_process=manage_process)
            
            # Check if we should auto-start the service
            auto_start = os.environ.get("LLM_SERVICE_AUTO_START", "false").lower() in {"true", "1", "yes"}
            if auto_start and manage_process:
                default_model = os.environ.get("LLM_DEFAULT_MODEL")
                logger.info(f"Auto-starting LLM service...")
                success = llm_service.start(model_name=default_model, background=True)
                if success:
                    logger.info(f"[SUCCESS] LLM service started on {host}:{port}")
                else:
                    logger.warning("[WARNING] LLM service auto-start failed")
            elif not manage_process:
                logger.info("[INFO] LLM service is managed externally (Docker Compose)")
                reachable = llm_service.start(background=True)
                if reachable:
                    logger.info(f"[SUCCESS] Verified external LLM service at {host}:{port}")
                else:
                    logger.warning("[WARNING] External LLM service not reachable yet")
            else:
                logger.info(f"[INFO] LLM service initialized (manual start required)")
            
            return True
        except Exception as e:
            logger.error(f"[ERROR] Failed to initialize LLM service: {e}")
            llm_service = None
            return False
    else:
        logger.warning("[WARNING] LLM Service Manager not available")
        return False

# AI Agent endpoints
@app.route('/api/ai/chat', methods=['POST', 'OPTIONS'])
def ai_chat():
    """Handle AI chat requests"""
    if request.method == 'OPTIONS':
        # Handle preflight request
        response = jsonify({"status": "ok"})
        response.headers.add('Access-Control-Allow-Origin', '*')
        response.headers.add('Access-Control-Allow-Headers', 'Content-Type')
        response.headers.add('Access-Control-Allow-Methods', 'POST')
        return response
    
    if not ai_agent:
        return jsonify({
            'success': False, 
            'message': 'AI agent not available'
        }), 503
    
    try:
        data = request.get_json()
        message = data.get('message')
        context_data = data.get('context', {})
        image = data.get('image')
        
        if not message:
            return jsonify({
                'success': False, 
                'message': 'Message is required'
            }), 400
        
        # Create AI context from the provided data
        context = AIContext(
            devices=context_data.get('devices', []),
            connections=context_data.get('connections', []),
            layout=context_data.get('layout', []),
            overlays=context_data.get('overlays'),
            system_time=datetime.now().isoformat(),
            user_intent=context_data.get('user_intent'),
            permissions=context_data.get('permissions')
        )
        
        # Run the async chat method in a separate thread to avoid event loop conflicts
        import eventlet
        # Use original threading to escape eventlet greenlets
        real_threading = eventlet.patcher.original('threading')
        
        result_holder = {'response': None, 'error': None}
        
        def run_chat_thread():
            try:
                # Run inside a fresh event loop in this OS thread to avoid eventlet/background-loop stalls
                import asyncio as _asyncio
                result_holder['response'] = _asyncio.run(ai_agent.chat(message, context, image))
            except Exception as e:
                result_holder['error'] = e
        
        # Run in a real OS thread
        chat_thread = real_threading.Thread(target=run_chat_thread, daemon=True)
        chat_thread.start()
        join_timeout = float(os.environ.get("AI_ROUTE_TIMEOUT", "35")) + 2.0
        chat_thread.join(timeout=join_timeout)
        if chat_thread.is_alive():
            return jsonify({
                'success': False,
                'message': f'AI chat timed out after {join_timeout:.0f}s'
            }), 504
        
        if result_holder['error']:
            raise result_holder['error']
            
        response = result_holder['response']
        if response is None:
            logger.error("AI chat returned no response")
            return jsonify({
                'success': False,
                'message': 'AI chat failed to produce a response'
            }), 500
        
        # Normalize response to avoid NoneType errors
        actions_src = []
        if hasattr(response, "actions"):
            actions_src = response.actions or []
            message_text = getattr(response, "message", "")
            vision_analysis = getattr(response, "vision_analysis", None)
            resp_error = getattr(response, "error", None)
            provider = getattr(response, "provider", None)
            model = getattr(response, "model", None)
        elif isinstance(response, dict):
            actions_src = response.get("actions") or []
            message_text = response.get("message", "")
            vision_analysis = response.get("vision_analysis")
            resp_error = response.get("error")
            provider = response.get("provider")
            model = response.get("model")
        else:
            message_text = str(response)
            vision_analysis = None
            resp_error = None
            provider = None
            model = None

        # Extract optional tools list
        if hasattr(response, "tools"):
            tools_src = getattr(response, "tools") or []
        elif isinstance(response, dict):
            tools_src = response.get("tools") or []
        else:
            tools_src = []

        # Manually serialize actions to avoid recursion issues
        actions_data = []
        if isinstance(actions_src, list):
            for action in actions_src:
                if not action:
                    continue
                action_dict = {
                    'kind': getattr(action, 'kind', None) if not isinstance(action, dict) else action.get('kind'),
                    'widget_type': getattr(action, 'widget_type', None) if not isinstance(action, dict) else action.get('widget_type'),
                    'widget_id': getattr(action, 'widget_id', None) if not isinstance(action, dict) else action.get('widget_id'),
                    'props': getattr(action, 'props', None) if not isinstance(action, dict) else action.get('props'),
                    'position': getattr(action, 'position', None) if not isinstance(action, dict) else action.get('position'),
                    'size': getattr(action, 'size', None) if not isinstance(action, dict) else action.get('size'),
                    'rule_name': getattr(action, 'rule_name', None) if not isinstance(action, dict) else action.get('rule_name'),
                    'rule_when': getattr(action, 'rule_when', None) if not isinstance(action, dict) else action.get('rule_when'),
                    'rule_actions': getattr(action, 'rule_actions', None) if not isinstance(action, dict) else action.get('rule_actions'),
                    'detection_model': getattr(action, 'detection_model', None) if not isinstance(action, dict) else action.get('detection_model'),
                    'camera_id': getattr(action, 'camera_id', None) if not isinstance(action, dict) else action.get('camera_id'),
                    'zone_id': getattr(action, 'zone_id', None) if not isinstance(action, dict) else action.get('zone_id'),
                    'natural_language_rule': getattr(action, 'natural_language_rule', None) if not isinstance(action, dict) else action.get('natural_language_rule'),
                    'object_types': getattr(action, 'object_types', None) if not isinstance(action, dict) else action.get('object_types'),
                    'confidence_threshold': getattr(action, 'confidence_threshold', None) if not isinstance(action, dict) else action.get('confidence_threshold'),
                    'monitoring_enabled': getattr(action, 'monitoring_enabled', None) if not isinstance(action, dict) else action.get('monitoring_enabled'),
                    'context_query': getattr(action, 'context_query', None) if not isinstance(action, dict) else action.get('context_query'),
                    'car_counting_camera_id': getattr(action, 'car_counting_camera_id', None) if not isinstance(action, dict) else action.get('car_counting_camera_id'),
                    # Generic tool execution support
                    'tool_id': getattr(action, 'tool_id', None) if not isinstance(action, dict) else action.get('tool_id'),
                    'parameters': getattr(action, 'parameters', None) if not isinstance(action, dict) else action.get('parameters')
                }
                actions_data.append(action_dict)
        
        return jsonify({
            'success': True,
            'data': {
                'message': message_text,
                'actions': actions_data,
                'tools': tools_src,
                'vision_analysis': vision_analysis,
                'error': resp_error,
                'provider': provider,  # Include provider for feedback tracking
                'model': model         # Include model for feedback tracking
            }
        })
        
    except Exception as e:
        import traceback as _tb
        logger.error("AI chat error: %s\n%s", e, _tb.format_exc())
        return jsonify({
            'success': False, 
            'message': f'AI chat error ({type(e).__name__}): {str(e)}'
        }), 500

@app.route('/api/ai/status', methods=['GET'])
def ai_status():
    """Get AI agent status"""
    if not ai_agent:
        return jsonify({
            'success': False,
            'message': 'AI agent not available'
        }), 503
    
    try:
        status = ai_agent.get_status()
        return jsonify({
            'success': True,
            'data': status
        })
    except Exception as e:
        logger.error(f"AI status error: {e}")
        return jsonify({
            'success': False,
            'message': f'AI status error: {str(e)}'
        }), 500


USER_KEYS_PATH = Path('data/llm_user_keys.json')


def _sanitize_api_key(value: Optional[str]) -> str:
    key = (value or "").strip()
    if (key.startswith('"') and key.endswith('"')) or (key.startswith("'") and key.endswith("'")):
        key = key[1:-1].strip()
    return key


def _extract_error_message(resp) -> str:
    try:
        payload = resp.json()
        if isinstance(payload, dict):
            err = payload.get("error")
            if isinstance(err, dict):
                msg = err.get("message")
                if isinstance(msg, str) and msg.strip():
                    return msg.strip()
            msg = payload.get("message")
            if isinstance(msg, str) and msg.strip():
                return msg.strip()
    except Exception:
        pass
    return resp.text[:300] if hasattr(resp, "text") else "Unknown error"


def _validate_openai_compatible_key(api_key: str, base_url: str) -> Dict[str, Any]:
    # OpenAI-compatible validation: GET /v1/models with Authorization: Bearer <key>
    url = base_url.rstrip("/") + "/models"
    headers = {"Authorization": f"Bearer {api_key}"}
    resp = requests.get(url, headers=headers, timeout=8)
    if resp.ok:
        data = resp.json() if isinstance(resp.json(), dict) else {}
        models = []
        items = data.get("data") if isinstance(data, dict) else None
        if isinstance(items, list):
            for item in items:
                if isinstance(item, dict) and item.get("id"):
                    models.append(str(item.get("id")))
        return {
            "valid": True,
            "status_code": resp.status_code,
            "message": "API key valid",
            "models": models[:12],
        }
    return {
        "valid": False,
        "status_code": resp.status_code,
        "message": _extract_error_message(resp),
        "models": [],
    }


def _validate_anthropic_key(api_key: str, base_url: str, version: str) -> Dict[str, Any]:
    # Anthropic docs: x-api-key + anthropic-version required for requests.
    url = base_url.rstrip("/") + "/models"
    headers = {
        "x-api-key": api_key,
        "anthropic-version": version,
        "content-type": "application/json",
    }
    resp = requests.get(url, headers=headers, timeout=8)
    if resp.ok:
        data = resp.json() if isinstance(resp.json(), dict) else {}
        models = []
        items = data.get("data") if isinstance(data, dict) else None
        if isinstance(items, list):
            for item in items:
                if isinstance(item, dict) and item.get("id"):
                    models.append(str(item.get("id")))
        return {
            "valid": True,
            "status_code": resp.status_code,
            "message": "API key valid",
            "models": models[:12],
        }
    return {
        "valid": False,
        "status_code": resp.status_code,
        "message": _extract_error_message(resp),
        "models": [],
    }


@app.route('/api/llm/user-keys', methods=['GET'])
def llm_user_keys_get():
    """Get user-provided API keys for AI providers."""
    try:
        if USER_KEYS_PATH.exists():
            with open(USER_KEYS_PATH, 'r') as f:
                data = json.load(f)
        else:
            data = {}
        return jsonify({'success': True, 'keys': data})
    except Exception as e:
        logger.error(f"Error reading user keys: {e}")
        return jsonify({'success': False, 'message': str(e)}), 500


@app.route('/api/llm/user-keys', methods=['PUT'])
def llm_user_keys_put():
    """Persist user-provided API keys for AI providers."""
    try:
        data = request.get_json() or {}
        USER_KEYS_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(USER_KEYS_PATH, 'w') as f:
            json.dump(data, f, indent=2)
        logger.info("User API keys updated")

        if ai_agent:
            ai_agent.reload_providers()

        return jsonify({'success': True, 'message': 'User API keys saved'})
    except Exception as e:
        logger.error(f"Error saving user keys: {e}")
        return jsonify({'success': False, 'message': str(e)}), 500


@app.route('/api/llm/test-key', methods=['POST'])
def llm_test_key():
    """Validate provider API keys using official auth headers and model-list endpoints."""
    try:
        data = request.get_json() or {}
        provider = str(data.get("provider") or "").strip().lower()
        api_key = _sanitize_api_key(data.get("api_key"))
        base_url = str(data.get("base_url") or "").strip()

        if provider not in {"openai", "anthropic", "grok", "xai"}:
            return jsonify({"success": False, "message": "Unsupported provider"}), 400
        if not api_key or len(api_key) < 10:
            return jsonify({"success": False, "message": "API key is required"}), 400

        if provider in {"openai"}:
            base_url = base_url or "https://api.openai.com/v1"
            result = _validate_openai_compatible_key(api_key, base_url)
        elif provider in {"grok", "xai"}:
            base_url = base_url or "https://api.x.ai/v1"
            result = _validate_openai_compatible_key(api_key, base_url)
        else:
            base_url = base_url or "https://api.anthropic.com/v1"
            version = os.environ.get("ANTHROPIC_VERSION", "2023-06-01")
            result = _validate_anthropic_key(api_key, base_url, version)

        return jsonify({"success": True, **result})
    except Exception as e:
        logger.error(f"Error validating API key: {e}")
        return jsonify({"success": False, "message": str(e)}), 500


@app.route('/api/llm/reload', methods=['POST'])
def llm_reload():
    """Reload AI providers after configuration updates."""
    try:
        if not ai_agent:
            return jsonify({'success': False, 'message': 'AI agent not initialized'}), 503
        status = ai_agent.reload_providers()
        return jsonify({'success': True, 'data': status})
    except Exception as e:
        logger.error(f"Error reloading AI providers: {e}")
        return jsonify({'success': False, 'message': str(e)}), 500

@app.route('/api/ai/snapshot-detect', methods=['POST', 'OPTIONS'])
def ai_snapshot_detect():
    """Lightweight snapshot-based object detection for quick counts"""
    if request.method == 'OPTIONS':
        response = jsonify({"status": "ok"})
        response.headers.add('Access-Control-Allow-Origin', '*')
        response.headers.add('Access-Control-Allow-Headers', 'Content-Type')
        response.headers.add('Access-Control-Allow-Methods', 'POST')
        return response
    
    if not ai_agent:
        return jsonify({
            'success': False,
            'error': 'AI agent not available'
        }), 503
    
    try:
        data = request.get_json()
        camera_id = data.get('camera_id')
        camera_name = data.get('camera_name', 'Unknown')
        object_classes = data.get('object_classes', ['car', 'person'])
        model = data.get('model', 'mobilenet')
        confidence = data.get('confidence', 0.3)
        
        if not camera_id:
            return jsonify({
                'success': False,
                'error': 'camera_id is required'
            }), 400
        
        # Run async detection in a separate thread
        import eventlet
        real_threading = eventlet.patcher.original('threading')
        
        result_holder = {'response': None, 'error': None}
        
        def run_detect_thread():
            try:
                result_holder['response'] = _run_coro_safe(
                    ai_agent.snapshot_detect(
                        camera_id=camera_id,
                        camera_name=camera_name,
                        object_classes=object_classes,
                        model=model,
                        confidence=confidence
                    )
                )
            except Exception as e:
                result_holder['error'] = e
        
        detect_thread = real_threading.Thread(target=run_detect_thread)
        detect_thread.start()
        detect_thread.join()
        
        if result_holder['error']:
            raise result_holder['error']
            
        result = result_holder['response']
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Snapshot detection error: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/ai/clear-context', methods=['POST'])
def ai_clear_context():
    """Clear AI agent environment context"""
    if not ai_agent:
        return jsonify({
            'success': False,
            'message': 'AI agent not available'
        }), 503
    
    try:
        ai_agent.clear_environment_context()
        return jsonify({
            'success': True,
            'message': 'Environment context cleared'
        })
    except Exception as e:
        logger.error(f"AI clear context error: {e}")
        return jsonify({
            'success': False,
            'message': f'AI clear context error: {str(e)}'
        }), 500

@app.route('/api/ai/vision', methods=['POST'])
def ai_vision():
    """Handle AI vision requests with source routing (local vs API)"""
    if not ai_agent:
        return jsonify({
            'success': False,
            'message': 'AI agent not available'
        }), 503
    
    try:
        data = request.get_json()
        image = data.get('image')
        prompt = data.get('prompt', 'Analyze this image')
        source = data.get('source', 'local')  # 'local' or 'api'
        model = data.get('model', 'blip')
        context_data = data.get('context', {})
        
        if not image:
            return jsonify({
                'success': False,
                'message': 'Image is required'
            }), 400
        
        # Route based on source
        if source == 'api':
            # Use cloud API (OpenAI GPT-4 Vision) - call provider.vision() directly
            logger.info(f"Routing vision request to cloud API (GPT-4 Vision)")
            
            # Call OpenAI provider's vision method directly with source='cloud'
            import eventlet
            real_threading = eventlet.patcher.original('threading')
            
            result_holder = {'vision_result': None, 'error': None}
            
            def run_vision_thread():
                try:
                    result_holder['vision_result'] = _run_coro_safe(
                        ai_agent.provider.vision(
                            image=image,
                            prompt=prompt,
                            model='gpt-4o',
                            opts={
                                'source': 'cloud',  # Force cloud API
                                'temperature': 0.3,
                                'max_tokens': 800,
                                'timeout': 30
                            }
                        )
                    )
                except Exception as e:
                    result_holder['error'] = e
            
            vision_thread = real_threading.Thread(target=run_vision_thread)
            vision_thread.start()
            vision_thread.join()
            
            if result_holder['error']:
                raise result_holder['error']
            
            vision_result = result_holder['vision_result']
            
            caption = vision_result.get('content', '') or vision_result.get('analysis', {}).get('caption', '')
            
            return jsonify({
                'success': True,
                'data': {
                    'caption': caption,
                    'analysis': caption,
                    'message': caption,
                    'model': 'gpt-4o',
                    'source': 'cloud'
                }
            })
        else:
            # Use local vision service (BLIP)
            logger.info(f"Routing vision request to local BLIP")
            vision_endpoint = os.environ.get('LOCAL_VISION_ENDPOINT', 'http://127.0.0.1:8101')
            
            # Remove data URL prefix if present
            image_b64 = image.split(",", 1)[1] if image.startswith("data:image/") else image

            model_slug = (model or "blip").strip().lower()
            if model_slug not in {"blip", "git"}:
                model_slug = "blip"
            
            # Run blocking request in a separate thread to avoid blocking eventlet loop
            import eventlet
            real_threading = eventlet.patcher.original('threading')
            
            result_holder = {'response': None, 'error': None, 'status_code': 500}
            
            def run_local_vision_thread():
                try:
                    # vision-local (production mode) expects:
                    # { image: <base64 or data-url>, model?: "blip"|"git", include_detections?: bool }
                    result_holder['response'] = requests.post(
                        f'{vision_endpoint}/describe',
                        json={
                            'image': image_b64,
                            'model': model_slug,
                            'include_detections': True,
                        },
                        timeout=30
                    )
                    result_holder['status_code'] = result_holder['response'].status_code
                except Exception as e:
                    result_holder['error'] = e
            
            vision_thread = real_threading.Thread(target=run_local_vision_thread)
            vision_thread.start()
            vision_thread.join()

            if result_holder['error']:
                logger.error(f"Local vision service unavailable: {result_holder['error']}")
                return jsonify({
                    'success': False,
                    'message': f'Local vision service unavailable: {str(result_holder["error"])}'
                }), 503
                
            response = result_holder['response']
            
            if response and response.status_code == 200:
                result = response.json()
                # Extract caption from /describe response format: {"results": [{"caption": "...", "model": "blip"}]}
                results = result.get('results', [])
                caption = ''
                if results and len(results) > 0:
                    caption = results[0].get('caption', '')
                else:
                    # Fallback to direct fields
                    caption = result.get('caption') or result.get('analysis') or result.get('aggregate_caption', '')
                
                logger.info(f"Local BLIP responded: {caption[:100]}...")
                
                return jsonify({
                    'success': True,
                    'data': {
                        'caption': caption,
                        'analysis': caption,
                        'message': caption,
                        'model': model_slug,
                        'source': 'local'
                    }
                })
            else:
                status_code = response.status_code if response else 500
                detail = ""
                try:
                    detail = (response.text or "").strip() if response is not None else ""
                except Exception:
                    detail = ""
                logger.error(f"Local vision service error: {status_code} {detail[:300]}")
                return jsonify({
                    'success': False,
                    'message': f'Local vision service error: {status_code}' + (f' - {detail[:300]}' if detail else '')
                }), 500
        
    except Exception as e:
        logger.error(f"AI vision error: {e}")
        return jsonify({
            'success': False,
            'message': f'AI vision error: {str(e)}'
        }), 500

@app.route('/api/ai/detect-objects', methods=['POST'])
def ai_detect_objects():
    """Detect objects in an image using YOLOv8 or MobileNet"""
    try:
        if not ai_agent:
            return jsonify({
                'success': False,
                'message': 'AI agent not available'
            }), 503

        data = request.get_json()
        if not data:
            return jsonify({
                'success': False,
                'message': 'No data provided'
            }), 400

        image_base64 = data.get('image_base64')
        # Default to MobileNet SSD to avoid requiring ultralytics in local setups
        model_type = data.get('model_type', 'mobilenet')
        confidence_threshold = data.get('confidence_threshold', 0.5)

        if not image_base64:
            return jsonify({
                'success': False,
                'message': 'Image data is required'
            }), 400

        # Perform object detection
        # Run in separate thread to avoid event loop conflicts
        import eventlet
        real_threading = eventlet.patcher.original('threading')
        
        result_holder = {'detections': None, 'error': None}
        
        def run_detect_thread():
            try:
                result_holder['detections'] = ai_agent.detect_objects_in_image(image_base64, model_type)
            except Exception as e:
                result_holder['error'] = e
                
        detect_thread = real_threading.Thread(target=run_detect_thread)
        detect_thread.start()
        detect_thread.join()
        
        if result_holder['error']:
            raise result_holder['error']
            
        detections = result_holder['detections']
        
        # Filter by confidence threshold
        filtered_detections = [
            obj for obj in detections 
            if obj.get('confidence', 0) >= confidence_threshold
        ]

        return jsonify({
            'success': True,
            'detections': filtered_detections,
            'model_used': model_type,
            'confidence_threshold': confidence_threshold,
            'total_detections': len(detections),
            'filtered_detections': len(filtered_detections)
        })

    except Exception as e:
        logger.error(f"Object detection error: {e}")
        return jsonify({
            'success': False,
            'message': f'Object detection error: {str(e)}'
        }), 500

@app.route('/api/ai/detect-cars-efficiently', methods=['POST'])
def ai_detect_cars_efficiently():
    """Efficient car detection using motion detection first, then object detection."""
    try:
        if not ai_agent:
            return jsonify({
                'success': False,
                'message': 'AI agent not available'
            }), 503

        data = request.get_json()
        if not data:
            return jsonify({
                'success': False,
                'message': 'No data provided'
            }), 400

        image_base64 = data.get('image_base64')
        camera_id = data.get('camera_id')
        previous_frame_base64 = data.get('previous_frame_base64')

        if not image_base64:
            return jsonify({
                'success': False,
                'message': 'Image data is required'
            }), 400

        if not camera_id:
            return jsonify({
                'success': False,
                'message': 'Camera ID is required'
            }), 400

        # Perform efficient car detection
        # Run in separate thread to avoid event loop conflicts
        import eventlet
        real_threading = eventlet.patcher.original('threading')
        
        result_holder = {'result': None, 'error': None}
        
        def run_car_detect_thread():
            try:
                result_holder['result'] = ai_agent.detect_cars_efficiently(image_base64, camera_id, previous_frame_base64)
            except Exception as e:
                result_holder['error'] = e
                
        car_thread = real_threading.Thread(target=run_car_detect_thread)
        car_thread.start()
        car_thread.join()
        
        if result_holder['error']:
            raise result_holder['error']
            
        result = result_holder['result']
        
        # If cars were detected, send a chat message
        if result.get('cars_detected', False):
            chat_message = result.get('chat_message', '')
            if chat_message:
                # Add to context memory for future reference
                ai_agent.add_to_context_memory(chat_message, "car_detection")
                
                # You could also send this to a chat system here
                logger.info(f"Car detection alert: {chat_message}")

        return jsonify({
            'success': True,
            'data': result
        })

    except Exception as e:
        logger.error(f"Efficient car detection error: {e}")
        return jsonify({
            'success': False,
            'message': f'Efficient car detection error: {str(e)}'
        }), 500

@app.route('/api/ai/car-detection-history', methods=['GET'])
def ai_car_detection_history():
    """Get car detection history."""
    try:
        if not ai_agent:
            return jsonify({'success': False, 'message': 'AI agent not available'}), 503
        camera_id = request.args.get('camera_id')
        limit = request.args.get('limit', 20, type=int)
        history = ai_agent.get_car_detection_history(camera_id, limit)
        return jsonify({'success': True, 'data': {'history': history, 'total_count': len(history)}})
    except Exception as e:
        logger.error(f"Car detection history error: {e}")
        return jsonify({'success': False, 'message': f'Car detection error: {str(e)}'}), 500

@app.route('/api/ai/analyze-scene', methods=['POST'])
def ai_analyze_scene():
    """Analyze scene to create automatic focus areas using LLM."""
    try:
        if not ai_agent:
            return jsonify({'success': False, 'message': 'AI agent not available'}), 503
        data = request.get_json()
        image_base64 = data.get('image_base64')
        camera_id = data.get('camera_id')
        
        if not image_base64 or not camera_id:
            return jsonify({'success': False, 'message': 'Missing image_base64 or camera_id'}), 400
        
        # Run the async function in a separate thread
        import eventlet
        real_threading = eventlet.patcher.original('threading')
        
        result_holder = {'response': None, 'error': None}
        
        def run_analysis_thread():
            try:
                result_holder['response'] = _run_coro_safe(
                    ai_agent.analyze_scene_for_focus_areas(image_base64, camera_id)
                )
            except Exception as e:
                result_holder['error'] = e
        
        analysis_thread = real_threading.Thread(target=run_analysis_thread)
        analysis_thread.start()
        analysis_thread.join()
        
        if result_holder['error']:
            raise result_holder['error']
            
        result = result_holder['response']
        
        return jsonify({'success': True, 'data': result})
    except Exception as e:
        logger.error(f"Scene analysis error: {e}")
        return jsonify({'success': False, 'message': f'Scene analysis error: {str(e)}'}), 500

@app.route('/api/ai/available-models', methods=['GET'])
def ai_available_models():
    """Compatibility wrapper that delegates to the unified provider below.
    Ensures response includes 'id' fields required by the frontend dropdown.
    """
    try:
        # Reuse the consolidated implementation to guarantee shape
        return api_available_models()
    except Exception as e:
        logger.error(f"Available models error: {e}")
        return jsonify({'success': False, 'message': f'Available models error: {str(e)}'}), 500

@app.route('/api/ai/set-active-model', methods=['POST'])
def ai_set_active_model():
    """Set the active AI model for detection."""
    try:
        if not ai_analyzer:
            return jsonify({'success': False, 'message': 'AI analyzer not available'}), 503
        
        data = request.get_json()
        model_name = data.get('model_name')
        
        if not model_name:
            return jsonify({'success': False, 'message': 'Missing model_name'}), 400
        
        success = ai_analyzer.set_active_model(model_name)
        if success:
            return jsonify({'success': True, 'message': f'Switched to {model_name} model'})
        else:
            return jsonify({'success': False, 'message': f'Model {model_name} not available'}), 400
    except Exception as e:
        logger.error(f"Set active model error: {e}")
        return jsonify({'success': False, 'message': f'Set active model error: {str(e)}'}), 500

@app.route('/api/ai/detect-objects-config', methods=['POST'])
def ai_detect_objects_config():
    """Detect objects with configurable model and target objects."""
    try:
        if not ai_agent:
            return jsonify({'success': False, 'message': 'AI agent not available'}), 503
        data = request.get_json()
        image_base64 = data.get('image_base64')
        model_type = data.get('model_type', 'yolov8')
        target_objects = data.get('target_objects', ['person', 'car', 'truck', 'bus', 'motorcycle'])
        confidence_threshold = data.get('confidence_threshold', 0.5)
        
        if not image_base64:
            return jsonify({'success': False, 'message': 'Missing image_base64'}), 400
        
        detections = ai_agent.detect_objects_in_image_with_config(
            image_base64, model_type, target_objects, confidence_threshold
        )
        return jsonify({'success': True, 'data': {'detections': detections, 'count': len(detections)}})
    except Exception as e:
        logger.error(f"Configurable object detection error: {e}")
        return jsonify({'success': False, 'message': f'Object detection error: {str(e)}'}), 500

@app.route('/api/ai/check-user-detection-zones', methods=['POST'])
def ai_check_user_detection_zones():
    """Check if detected objects interact with user-defined zones, lines, or tags."""
    try:
        if not ai_agent:
            return jsonify({'success': False, 'message': 'AI agent not available'}), 503
        data = request.get_json()
        camera_id = data.get('camera_id')
        detected_objects = data.get('detected_objects', [])
        shapes = data.get('shapes', [])
        
        if not camera_id:
            return jsonify({'success': False, 'message': 'Missing camera_id'}), 400
        
        triggered_events = ai_agent.check_user_detection_zones(camera_id, detected_objects, shapes)
        return jsonify({'success': True, 'data': {'events': triggered_events, 'count': len(triggered_events)}})
    except Exception as e:
        logger.error(f"User detection zones error: {e}")
        return jsonify({'success': False, 'message': f'User detection zones error: {str(e)}'}), 500

# Test MediaMTX connection on startup
def test_mediamtx_connection():
    """Test MediaMTX connection and log status"""
    # Use silent=True to avoid double logging since we log the result here
    if mediamtx.test_connection(silent=True):
        logger.info("[SUCCESS] MediaMTX connection successful")
        return True
    else:
        logger.warning("[WARNING] MediaMTX connection failed - WebRTC features may not work")
        return False

def check_mediamtx_status():
    """Check MediaMTX status and provide helpful information"""
    try:
        logger.info("Checking MediaMTX status...")
        
        # Check if MediaMTX API is accessible
        response = requests.get(f"{MEDIAMTX_API_URL}/paths/list", timeout=5)
        
        if response.status_code == 200:
            paths_data = response.json()
            
            # Handle different response formats
            if isinstance(paths_data, dict) and 'items' in paths_data:
                # MediaMTX v3 format: {"itemCount": X, "pageCount": Y, "items": [...]}
                active_paths = [item.get("name", "") for item in paths_data.get('items', []) if isinstance(item, dict) and item.get("ready", False)]
            elif isinstance(paths_data, dict):
                # Legacy format: {path_name: path_info}
                active_paths = [name for name, info in paths_data.items() if isinstance(info, dict) and info.get("ready", False)]
            elif isinstance(paths_data, list):
                # Old format: [path_info]
                active_paths = [path.get("name", "") for path in paths_data if isinstance(path, dict) and path.get("ready", False)]
            else:
                logger.warning(f"Unexpected MediaMTX response format: {type(paths_data)}")
                active_paths = []
            
            logger.info(f"[SUCCESS] MediaMTX is running and accessible")
            logger.info(f"📊 Active paths: {len(active_paths)}")
            
            if active_paths:
                logger.info(f"🎥 Active cameras: {', '.join(active_paths)}")
            else:
                logger.info("⚠️  No active camera streams found")
                logger.info("💡 To connect cameras:")
                logger.info("   1. Use RTSP: rtsp://localhost:8554/[camera-name]")
                logger.info("   2. Use RTMP: rtmp://localhost:1935/[camera-name]")
                logger.info("   3. Use WebRTC: Connect to http://localhost:8889/[camera-name]/whep")
                logger.info("   4. Camera names can be: cam1, cam2, cam3, etc.")
            
            return True
        else:
            logger.error(f"[ERROR] MediaMTX API not accessible: {response.status_code}")
            return False
            
    except Exception as e:
        logger.error(f"[ERROR] Cannot connect to MediaMTX: {e}")
        logger.info("💡 Make sure MediaMTX is running in the mediamtx folder")
        return False

def connect_camera_to_mediamtx(camera_id: str) -> bool:
    """Connect a camera to MediaMTX using its RTSP URL"""
    try:
        if not _is_camera_allowed(camera_id):
            logger.warning(f"Camera {camera_id} blocked by 4-camera limit")
            return False
        # Find the camera in our database
        camera = next((c for c in cameras_db if c['id'] == camera_id), None)
        if not camera:
            logger.error(f"Camera {camera_id} not found in database")
            return False
        
        logger.info(f"Connecting camera {camera['name']} to MediaMTX...")
        ensure_camera_stream_metadata(camera)
        
        # Create MediaMTX path using the camera's RTSP URL
        success = mediamtx.create_path(camera.get('mediamtx_path', camera_id), camera['rtsp_url'])
        if success and camera.get('substream_rtsp_url') and camera.get('mediamtx_sub_path'):
            mediamtx.create_path(camera['mediamtx_sub_path'], camera['substream_rtsp_url'])
        if success:
            logger.info(f"✅ Successfully connected camera {camera['name']} to MediaMTX")
            return True
        logger.error(f"❌ Failed to connect camera {camera['name']} to MediaMTX")
        return False
            
    except Exception as e:
        logger.error(f"❌ Error connecting camera {camera_id} to MediaMTX: {e}")
        return False

def connect_all_cameras_to_mediamtx():
    """Connect all cameras to MediaMTX"""
    logger.info("Connecting all cameras to MediaMTX...")
    
    success_count = 0
    limit_state = _camera_limit_state()
    for camera in limit_state["allowed"]:
        if camera.get('enabled', True):
            if connect_camera_to_mediamtx(camera['id']):
                success_count += 1
    
    logger.info(f"✅ Connected {success_count}/{len(limit_state['allowed'])} cameras to MediaMTX")
    return success_count

_MEDIAMTX_READY_CACHE_SECONDS = 10.0

def ensure_camera_mediamtx_ready(camera: Dict[str, Any], force_check: bool = False) -> bool:
    """Ensure a camera has an active MediaMTX path ready for WebRTC/HLS usage."""
    if not camera or not camera.get('id'):
        return False

    try:
        ensure_camera_stream_metadata(camera)

        now = time.time()
        last_check = camera.get('_last_mediamtx_check', 0.0)
        if (
            not force_check
            and camera.get('mediamtx_ready')
            and (now - last_check) < _MEDIAMTX_READY_CACHE_SECONDS
        ):
            return True

        camera['_last_mediamtx_check'] = now

        path_name = camera.get('mediamtx_path') or camera['id']
        rtsp_url = camera.get('rtsp_url')
        if not path_name or not rtsp_url:
            camera['mediamtx_ready'] = False
            return False

        path_info = mediamtx.get_path_info(path_name)
        if path_info.get('ready'):
            camera['mediamtx_ready'] = True
            camera['ready'] = True
            camera['status'] = 'live'
            camera['last_seen'] = datetime.now().isoformat()
            return True

        connected = connect_camera_to_mediamtx(camera['id'])
        if connected:
            camera['mediamtx_ready'] = True
            camera['ready'] = True
            camera['status'] = 'live'
            camera['last_seen'] = datetime.now().isoformat()
            return True

        camera['mediamtx_ready'] = False
        return False
    except Exception as exc:
        logger.error(f"❌ Failed to ensure MediaMTX path for {camera.get('id')}: {exc}")
        camera['mediamtx_ready'] = False
        return False

def _normalize_camera_stream_urls(camera: Dict[str, Any]) -> None:
    """Ensure camera stream URLs always point to the proxy endpoints."""
    path_name = camera.get('mediamtx_path') or camera.get('id')
    if path_name:
        camera['hls_url'] = f"/proxy/hls/{path_name}/index.m3u8"
        camera['webrtc_url'] = f"/proxy/webrtc/{path_name}"
        camera['webrtc_whip_url'] = f"/proxy/webrtc/{path_name}/whip"
        camera['webrtc_whep_url'] = f"/proxy/webrtc/{path_name}/whep"

    sub_path = camera.get('mediamtx_sub_path')
    if sub_path:
        camera['hls_sub_url'] = f"/proxy/hls/{sub_path}/index.m3u8"
        camera['webrtc_sub_whep_url'] = f"/proxy/webrtc/{sub_path}/whep"

def serialize_camera(camera: Dict[str, Any], force_mediamtx_check: bool = False, include_mediamtx: bool = True) -> Dict[str, Any]:
    """Return a sanitized snapshot of the camera with guaranteed proxy URLs."""
    ensure_camera_stream_metadata(camera)
    if include_mediamtx and camera.get('webrtc_enabled', True):
        ensure_camera_mediamtx_ready(camera, force_check=force_mediamtx_check)
    _normalize_camera_stream_urls(camera)

    return {
        key: value
        for key, value in camera.items()
        if not str(key).startswith('_')
    }

def configure_mediamtx_paths_for_cameras():
    """Configure MediaMTX paths for all cameras in cameras.json - CRITICAL for camera streaming"""
    logger.info("=" * 80)
    logger.info("🔧 CONFIGURING MediaMTX PATHS FOR CAMERAS (CRITICAL SERVICE)")
    logger.info("=" * 80)
    
    # Wait for MediaMTX to be ready
    max_wait = 60  # Increased to 60s to allow MediaMTX time to start
    wait_count = 0
    while wait_count < max_wait:
        try:
            # Use mediamtx.test_connection() to include authentication
            if mediamtx.test_connection():
                logger.info("✅ MediaMTX API is ready")
                break
        except Exception as e:
            logger.warning(f"⚠️ MediaMTX API connection attempt failed: {e}")
            pass
        wait_count += 1
        if wait_count < max_wait:
            logger.info(f"⏳ Waiting for MediaMTX API... ({wait_count}/{max_wait})")
            time.sleep(1)
    
    if wait_count >= max_wait:
        logger.error("❌ CRITICAL: MediaMTX API not responding - cameras will not work!")
        return 0
    
    try:
        # Load cameras from cameras.json
        if not os.path.exists('cameras.json'):
            logger.warning("⚠️ cameras.json not found")
            return 0
            
        with open('cameras.json', 'r') as f:
            cameras = json.load(f)
        
        logger.info(f"📹 Found {len(cameras)} camera(s) in cameras.json")
        
        success_count = 0
        failed_cameras = []
        
        for camera in cameras:
            if not camera.get('enabled', True):
                logger.info(f"⏭️ Skipping disabled camera: {camera.get('name')}")
                continue
                
            camera_id = camera.get('id')
            camera_name = camera.get('name')
            rtsp_url = camera.get('rtsp_url')
            
            if not camera_id or not rtsp_url:
                logger.warning(f"⚠️ Skipping {camera_name}: missing ID or RTSP URL")
                failed_cameras.append(camera_name)
                continue
            
            logger.info(f"\n🎥 Configuring: {camera_name}")
            logger.info(f"   ID: {camera_id}")
            logger.info(f"   RTSP: {rtsp_url[:50]}...")  # Truncate for security

            ensure_camera_stream_metadata(camera)

            try:
                main_path = camera.get('mediamtx_path', camera_id)
                
                # Robust retry loop with exponential backoff
                max_retries = 3  # Reduced retries to prevent spam
                created_main = False
                
                for attempt in range(max_retries):
                    try:
                        # Initial backoff + exponential increase
                        backoff = 0.5 * (2 ** attempt)
                        time.sleep(backoff)
                        
                        created_main = mediamtx.create_path(main_path, camera['rtsp_url'])
                        if created_main:
                            break
                    except Exception:
                        pass # create_path handles logging
                
                if created_main:
                    logger.info(f"   ✅ Primary path ready: {main_path}")
                    success_count += 1
                else:
                    logger.error(f"   ❌ Failed to configure primary path for {camera_name}")
                    failed_cameras.append(camera_name)
                    continue

                if camera.get('substream_rtsp_url') and camera.get('mediamtx_sub_path'):
                    sub_path = camera['mediamtx_sub_path']
                    
                    # Retry logic for substream too
                    for attempt in range(max_retries):
                        try:
                            backoff = 0.5 * (2 ** attempt)
                            time.sleep(backoff)
                            if mediamtx.create_path(sub_path, camera['substream_rtsp_url']):
                                logger.info(f"   ✅ Substream path ready: {sub_path}")
                                break
                        except Exception:
                             pass


            except Exception as e:
                logger.error(f"   ❌ Unexpected error: {e}")
                failed_cameras.append(camera_name)

            # Sync metadata back to in-memory DB
            global_camera = next((c for c in cameras_db if c.get('id') == camera_id), None)
            if global_camera:
                global_camera.update({
                    'stream_path': camera.get('stream_path'),
                    'substream_path': camera.get('substream_path'),
                    'substream_rtsp_url': camera.get('substream_rtsp_url'),
                    'mediamtx_path': camera.get('mediamtx_path', camera_id),
                    'mediamtx_sub_path': camera.get('mediamtx_sub_path'),
                    'webrtc_sub_whep_url': camera.get('webrtc_sub_whep_url'),
                    'hls_sub_url': camera.get('hls_sub_url')
                })
                ensure_camera_stream_metadata(global_camera)
        
        logger.info("\n" + "=" * 80)
        logger.info(f"✅ MediaMTX CONFIGURATION COMPLETE")
        logger.info(f"   Configured: {success_count}/{len([c for c in cameras if c.get('enabled', True)])} cameras")
        if failed_cameras:
            logger.warning(f"   ⚠️ Failed: {', '.join(failed_cameras)}")
        logger.info("=" * 80)
        
        save_cameras()
        return success_count
        
    except Exception as e:
        logger.error(f"❌ CRITICAL ERROR configuring MediaMTX paths: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return 0

def auto_discover_cameras():
    """Automatically discover and add cameras from MediaMTX"""
    global cameras_db

    try:
        logger.info("🔍 Auto-discovering cameras from MediaMTX...")

        response = requests.get(f"{MEDIAMTX_API_URL}/paths/list", timeout=10)
        if response.status_code != 200:
            logger.warning(f"Failed to fetch MediaMTX paths for auto-discovery: {response.status_code}")
            return 0

        data = response.json() or {}
        logger.info(f"MediaMTX API response type: {type(data)}")
        logger.info(f"MediaMTX API response keys: {list(data.keys()) if isinstance(data, dict) else 'not a dict'}")

        items = []

        # Handle different MediaMTX API response formats
        if isinstance(data, dict):
            # MediaMTX v3 format: { items: [...] }
            if 'items' in data and isinstance(data['items'], list):
                items = data['items']
                logger.info(f"Found v3 format with {len(items)} items")
            # MediaMTX v2 format: { path_name: path_info }
            else:
                items = [{'name': k, **v} for k, v in data.items() if k not in ['all_cameras', 'test', 'fallback']]
                logger.info(f"Found v2 format with {len(items)} paths")
        elif isinstance(data, list):
            items = data
            logger.info(f"Found list format with {len(items)} items")
        else:
            logger.warning(f"Unexpected MediaMTX response format: {type(data)}")
            return 0

        discovered_cameras = []

        def extract_base_camera_id(path_name):
            """Extract the base camera ID from MediaMTX paths like camera_camera_363bdde1-7420-49d0-89ae-96bc9b67aebd"""
            if not path_name:
                return path_name
            
            # Remove repeated "camera_" prefixes to get the original ID
            while path_name.startswith('camera_'):
                path_name = path_name[7:]  # Remove "camera_" prefix
            
            return path_name

        def is_duplicate_mediamtx_path(path_name):
            """Check if this MediaMTX path is a duplicate of an existing camera"""
            if not path_name:
                return False
            
            # Check if this path name matches any existing camera ID
            for camera in cameras_db:
                if camera.get('id') == path_name:
                    return True
                if camera.get('mediamtx_path') == path_name:
                    return True
            
            # Only block known duplicate patterns, not legitimate UUID camera IDs
            # Block any path that looks like it might be a MediaMTX-generated duplicate
            if path_name.upper() == path_name and len(path_name) > 20:
                logger.warning(f"Blocking MediaMTX-generated duplicate path: {path_name}")
                return True
            
            return False

        for item in items:
            if not isinstance(item, dict):
                continue

            path_name = item.get('name') or item.get('id')
            if not path_name:
                continue

            # Skip reserved/system paths
            if path_name in ['all_cameras', 'test', 'fallback']:
                continue

            # Skip duplicate camera paths (camera_, camera_camera_, etc.)
            if path_name.startswith('camera_'):
                logger.debug(f"Skipping duplicate MediaMTX path: {path_name}")
                continue

            # Skip if this is a duplicate MediaMTX path
            if is_duplicate_mediamtx_path(path_name):
                logger.debug(f"Skipping duplicate MediaMTX path: {path_name}")
                continue

            # Check if camera is ready/active
            ready = bool(
                item.get('ready') or 
                item.get('sourceReady') or 
                (item.get('status') == 'ready') or
                item.get('source')  # If source exists, camera is active
            )
            
            readers_count = len(item.get('readers', []) if isinstance(item.get('readers'), list) else [])
            publishers_count = 1 if item.get('source') else item.get('publishers_count', 0)

            # Find existing camera by base ID (handle both original and MediaMTX paths)
            base_id = extract_base_camera_id(path_name)
            existing_camera = next(
                (c for c in cameras_db if c.get('id') == base_id or c.get('mediamtx_path') == path_name), 
                None
            )

            if not existing_camera:
                limit_state = _camera_limit_state()
                if len(limit_state["enabled"]) >= limit_state["limit"]:
                    logger.warning("Camera limit reached; skipping auto-discovered camera %s", path_name)
                    continue
                # Allow creation of new cameras from MediaMTX paths if they're not duplicates
                logger.info(f"Creating new camera from MediaMTX path: {path_name}")
                
                # Create new camera configuration
                camera = {
                    'id': base_id,
                    'name': base_id.replace('_', ' ').replace('-', ' ').title(),
                    'ip_address': 'MediaMTX',
                    'port': 554,
                    'username': 'admin',
                    'password': '',
                    'location': base_id.replace('_', ' ').replace('-', ' ').title(),
                    'status': 'online' if ready else 'offline',
                    'rtsp_url': f"{MEDIAMTX_RTSP_URL}/{path_name}",
                    'webrtc_enabled': True,
                    'hls_url': f"/proxy/hls/{path_name}/index.m3u8",
                    'webrtc_url': f"/proxy/webrtc/{path_name}",
                    'webrtc_whep_url': f"/proxy/webrtc/{path_name}/whep",
                    'created_at': datetime.now().isoformat(),
                    'updated_at': datetime.now().isoformat(),
                    'enabled': True,
                    'type': 'camera',
                    'device_type': 'camera',
                    'mediamtx_path': path_name,
                    'ready': ready,
                    'readers_count': readers_count,
                    'publishers_count': publishers_count,
                    'mediamtx_ready': ready,
                    'source': item.get('source'),
                    'motion_detection': True  # Enable motion detection by default
                }

                ensure_camera_stream_metadata(camera)
                cameras_db.append(camera)
                discovered_cameras.append(camera)
                logger.info(f"✅ Auto-discovered camera: {camera['name']} (path: {path_name}) - Status: {'online' if ready else 'offline'}")
                continue
            else:
                # Update existing camera with MediaMTX status
                existing_camera['status'] = 'online' if ready else 'offline'
                existing_camera['ready'] = ready
                existing_camera['readers_count'] = readers_count
                existing_camera['publishers_count'] = publishers_count
                existing_camera['mediamtx_ready'] = ready
                existing_camera['mediamtx_path'] = existing_camera.get('mediamtx_path') or path_name
                existing_camera['mediamtx_rtsp_url'] = f"{MEDIAMTX_RTSP_URL}/{path_name}"
                existing_camera['updated_at'] = datetime.now().isoformat()
                ensure_camera_stream_metadata(existing_camera)
                logger.info(f"✅ Updated camera: {existing_camera['name']} - Status: {existing_camera['status']}")

        if discovered_cameras or any(c.get('updated_at') for c in cameras_db):
            save_cameras()
            logger.info(f"💾 Saved {len(discovered_cameras)} new cameras and updated existing ones")

        logger.info(f"✅ Auto-discovery complete: {len(discovered_cameras)} new, {len(cameras_db)} total")
        return len(discovered_cameras)

    except Exception as e:
        logger.error(f"Error during auto-discovery: {e}")
        return 0


# Test route
@app.route('/test')
def test():
    mediamtx_status = mediamtx.test_connection()
    return jsonify({
        "status": "success",
        "message": "Flask is working!",
        "mediamtx_connected": mediamtx_status,
        "timestamp": datetime.now().isoformat()
    })


# Health check
@app.route('/api/health')
def health():
    mediamtx_status = mediamtx.test_connection()
    return jsonify({
        "status": "healthy",
        "cameras": len(cameras_db),
        "mediamtx_connected": mediamtx_status,
        "timestamp": datetime.now().isoformat()
    })


# ---- Storage management endpoints ----

@app.route('/api/storage/settings', methods=['GET'])
def get_storage_settings():
    return jsonify({"success": True, "data": storage_manager.settings})


@app.route('/api/storage/settings', methods=['POST'])
def update_storage_settings():
    patch = request.get_json(silent=True) or {}
    updated = storage_manager.update_settings(patch)
    return jsonify({"success": True, "data": updated})


@app.route('/api/storage/status', methods=['GET'])
def get_storage_status():
    return jsonify({"success": True, "data": storage_manager.get_status()})


# Sync cameras from MediaMTX
@app.route('/api/cameras/sync', methods=['POST'])
def sync_cameras():
    """Sync cameras from MediaMTX"""
    try:
        # Auto-discover new cameras and update existing ones
        discovered_count = auto_discover_cameras()
        return jsonify({
            "status": "success",
            "message": f"Synced {len(cameras_db)} cameras from MediaMTX (discovered {discovered_count} new)",
            "count": len(cameras_db),
            "discovered": discovered_count
        })
    except Exception as e:
        logger.error(f"Error syncing cameras: {e}")
        return jsonify({
            "status": "error",
            "message": f"Failed to sync cameras: {str(e)}"
        }), 500

# Comprehensive camera synchronization and detection setup
@app.route('/api/cameras/sync-comprehensive', methods=['POST'])
def sync_cameras_comprehensive():
    """Comprehensive synchronization between cameras.json and cameras.db with detection streaming setup"""
    try:
        # Check if camera manager is available
        if 'CAMERA_MANAGER_GLOBAL' in globals() and CAMERA_MANAGER_GLOBAL is not None:
            # Run the comprehensive sync safely on shared loop
            detection_enabled_count = _run_coro_safe(
                CAMERA_MANAGER_GLOBAL.sync_all_cameras_and_ensure_detection()
            )
            
            return jsonify({
                "status": "success",
                "message": f"Comprehensive sync completed. {detection_enabled_count} cameras have detection streaming enabled.",
                "detection_enabled_count": detection_enabled_count,
                "total_cameras": len(cameras_db)
            })
        else:
            # Fallback to basic sync if camera manager not available
            discovered_count = auto_discover_cameras()
            return jsonify({
                "status": "success",
                "message": f"Basic sync completed (camera manager not available). Synced {len(cameras_db)} cameras.",
                "detection_enabled_count": 0,
                "total_cameras": len(cameras_db),
                "discovered": discovered_count
            })
            
    except Exception as e:
        logger.error(f"Error in comprehensive camera sync: {e}")
        return jsonify({
            "status": "error",
            "message": f"Failed to perform comprehensive sync: {str(e)}"
        }), 500

# Get detection status for all cameras
@app.route('/api/cameras/detection-status', methods=['GET'])
def get_cameras_detection_status():
    """Get detection status for all cameras"""
    try:
        if 'CAMERA_MANAGER_GLOBAL' in globals() and CAMERA_MANAGER_GLOBAL is not None:
            # Get detection status safely on shared loop
            status_data = _run_coro_safe(
                CAMERA_MANAGER_GLOBAL.get_all_cameras_detection_status()
            )
            
            return jsonify({
                "status": "success",
                "data": status_data
            })
        else:
            # Fallback status without camera manager
            enabled_cameras = [c for c in cameras_db if c.get('enabled', True)]
            motion_enabled = [c for c in enabled_cameras if c.get('motion_detection', True)]
            
            return jsonify({
                "status": "success",
                "data": {
                    "total_cameras": len(cameras_db),
                    "enabled_cameras": len(enabled_cameras),
                    "detection_enabled": len(motion_enabled),
                    "connected_cameras": len([c for c in cameras_db if c.get('status') == 'live']),
                    "camera_statuses": {},
                    "note": "Camera manager not available - limited status information"
                }
            })
            
    except Exception as e:
        logger.error(f"Error getting cameras detection status: {e}")
        return jsonify({
            "status": "error",
            "message": f"Failed to get detection status: {str(e)}"
        }), 500

# Verify specific camera detection status
@app.route('/api/cameras/<camera_id>/detection-status', methods=['GET'])
def get_camera_detection_status(camera_id):
    """Get detection status for a specific camera"""
    try:
        if 'CAMERA_MANAGER_GLOBAL' in globals() and CAMERA_MANAGER_GLOBAL is not None:
            # Get specific camera status safely on shared loop
            status_data = _run_coro_safe(
                CAMERA_MANAGER_GLOBAL.verify_camera_detection_status(camera_id)
            )
            
            return jsonify({
                "status": "success",
                "data": status_data
            })
        else:
            # Fallback status without camera manager
            camera = next((c for c in cameras_db if c['id'] == camera_id), None)
            if camera:
                return jsonify({
                    "status": "success",
                    "data": {
                        "camera_id": camera_id,
                        "name": camera.get('name', 'Unknown'),
                        "enabled": camera.get('enabled', True),
                        "motion_detection_enabled": camera.get('motion_detection', True),
                        "connected": camera.get('status') == 'live',
                        "rtsp_url": camera.get('rtsp_url', ''),
                        "note": "Camera manager not available - limited status information"
                    }
                })
            else:
                return jsonify({
                    "status": "error",
                    "message": "Camera not found"
                }), 404
            
    except Exception as e:
        logger.error(f"Error getting camera {camera_id} detection status: {e}")
        return jsonify({
            "status": "error",
            "message": f"Failed to get detection status: {str(e)}"
        }), 500

# Ensure detection streaming for specific camera
@app.route('/api/cameras/<camera_id>/ensure-detection', methods=['POST'])
def ensure_camera_detection(camera_id):
    """Ensure detection streaming is working for a specific camera"""
    try:
        if not _is_camera_allowed(camera_id):
            return jsonify({
                "status": "error",
                "message": "Camera limit reached. This camera is not enabled for detection.",
                "camera_id": camera_id
            }), 403
        if 'CAMERA_MANAGER_GLOBAL' in globals() and CAMERA_MANAGER_GLOBAL is not None:
            # Ensure detection streaming
            success = _run_coro_safe(
                CAMERA_MANAGER_GLOBAL.ensure_detection_streaming(camera_id)
            )
            
            if success:
                return jsonify({
                    "status": "success",
                    "message": f"Detection streaming enabled for camera {camera_id}",
                    "camera_id": camera_id
                })
            else:
                return jsonify({
                    "status": "error",
                    "message": f"Failed to enable detection streaming for camera {camera_id}",
                    "camera_id": camera_id
                }), 500
        else:
            return jsonify({
                "status": "error",
                "message": "Camera manager not available",
                "camera_id": camera_id
            }), 503
            
    except Exception as e:
        logger.error(f"Error ensuring detection for camera {camera_id}: {e}")
        return jsonify({
            "status": "error",
            "message": f"Failed to ensure detection: {str(e)}",
            "camera_id": camera_id
        }), 500

# Connect camera to stream server for motion detection
@app.route('/api/cameras/<camera_id>/connect-stream', methods=['POST'])
def connect_camera_stream(camera_id):
    """Connect a camera to the stream server for motion detection"""
    try:
        if not _is_camera_allowed(camera_id):
            return jsonify({
                "status": "error",
                "message": "Camera limit reached. This camera is not enabled for streaming.",
                "camera_id": camera_id
            }), 403
        if 'STREAM_SERVER_GLOBAL' in globals() and STREAM_SERVER_GLOBAL is not None:
            stream_server = STREAM_SERVER_GLOBAL
            
            # Find camera in database
            camera = None
            for cam in cameras_db:
                if cam.get('id') == camera_id:
                    camera = cam
                    break
            
            if not camera:
                return jsonify({
                    "status": "error",
                    "message": f"Camera {camera_id} not found in database",
                    "camera_id": camera_id
                }), 404
            
            rtsp_url = camera.get('rtsp_url')
            if not rtsp_url:
                return jsonify({
                    "status": "error",
                    "message": f"Camera {camera_id} has no RTSP URL configured",
                    "camera_id": camera_id
                }), 400
            
            # Check if already connected
            if hasattr(stream_server, 'active_streams') and camera_id in stream_server.active_streams:
                return jsonify({
                    "status": "success",
                    "message": f"Camera {camera_id} already connected to stream server",
                    "camera_id": camera_id
                })
            
            # Connect camera to stream server
            success = _run_coro_safe(
                stream_server.start_stream(camera_id, {
                    'rtsp_url': rtsp_url,
                    'webrtc_enabled': camera.get('webrtc_enabled', True),
                    'fps': 15  # Lower FPS for motion detection
                })
            )
            
            if success:
                logger.info(f"✅ Camera {camera.get('name', camera_id)} connected to stream server for motion detection")
                return jsonify({
                    "status": "success",
                    "message": f"Camera {camera.get('name', camera_id)} connected to stream server for motion detection",
                    "camera_id": camera_id,
                    "camera_name": camera.get('name', 'Unknown')
                })
            else:
                logger.error(f"❌ Failed to connect camera {camera.get('name', camera_id)} to stream server")
                return jsonify({
                    "status": "error",
                    "message": f"Failed to connect camera {camera.get('name', camera_id)} to stream server",
                    "camera_id": camera_id
                }), 500
        else:
            return jsonify({
                "status": "error",
                "message": "Stream server not available",
                "camera_id": camera_id
            }), 503
            
    except Exception as e:
        logger.error(f"Error connecting camera {camera_id} to stream server: {e}")
        return jsonify({
            "status": "error",
            "message": f"Failed to connect camera to stream server: {str(e)}",
            "camera_id": camera_id
        }), 500

# Auto-discover cameras
@app.route('/api/cameras/discover', methods=['POST'])
def discover_cameras():
    """Auto-discover cameras from MediaMTX"""
    try:
        discovered_count = auto_discover_cameras()
        return jsonify({
            "status": "success",
            "message": f"Discovered {discovered_count} new cameras from MediaMTX",
            "count": len(cameras_db),
            "discovered": discovered_count
        })
    except Exception as e:
        logger.error(f"Error discovering cameras: {e}")
        return jsonify({
            "status": "error",
            "message": f"Failed to discover cameras: {str(e)}"
        }), 500

# Connect cameras to MediaMTX
@app.route('/api/cameras/connect', methods=['POST'])
def connect_cameras():
    """Connect all cameras to MediaMTX"""
    try:
        limit_state = _camera_limit_state()
        success_count = connect_all_cameras_to_mediamtx()
        return jsonify({
            "status": "success",
            "message": f"Connected {success_count} cameras to MediaMTX",
            "connected_count": success_count,
            "total_count": len(limit_state["allowed"]),
            "limit": limit_state["limit"],
            "enabled_count": len(limit_state["enabled"])
        })
    except Exception as e:
        logger.error(f"Error connecting cameras: {e}")
        return jsonify({
            "status": "error",
            "message": f"Failed to connect cameras: {str(e)}"
        }), 500

# Get all cameras (compatible with frontend)
@app.route('/api/cameras', methods=['GET'])
def get_cameras():
    # Keep list endpoint lightweight: do NOT block on MediaMTX readiness checks.
    camera_snapshots = [serialize_camera(camera, include_mediamtx=False) for camera in cameras_db]
    return jsonify({
        "status": "success",
        "data": camera_snapshots,
        "count": len(camera_snapshots)
    })


@app.route('/api/system/entitlement', methods=['GET'])
def get_entitlement():
    summary = entitlement_summary(len(_enabled_cameras()))
    return jsonify({
        "status": "success",
        "data": summary
    })


@app.route('/api/system/entitlement/refresh', methods=['POST'])
def refresh_entitlement():
    return jsonify({
        "status": "disabled",
        "message": "Cloud entitlement refresh is not included in the public beta. Knoxnet VMS Beta is free for up to 4 cameras."
    }), 404

@app.route('/api/system/feedback', methods=['POST'])
def submit_feedback():
    """
    Capture local beta feedback for support and product improvements.
    Public beta users should report bugs through GitHub Issues.
    """
    try:
        allow_remote = str(os.environ.get("KNOXNET_FEEDBACK_ALLOW_REMOTE", "")).lower() in {"1", "true", "yes"}
        remote = str(getattr(request, "remote_addr", "") or "")
        local_addrs = {"127.0.0.1", "::1"}
        is_local = remote in local_addrs
        if not allow_remote and not is_local:
            return jsonify({
                "status": "error",
                "message": "Feedback submission is local-only by default"
            }), 403

        data = request.get_json(silent=True) or {}
        message = str(data.get("message") or "").strip()
        if not message:
            return jsonify({
                "status": "error",
                "message": "message is required"
            }), 400

        from core.paths import get_data_dir
        from datetime import datetime
        from pathlib import Path
        import json as _json

        record = {
            "message": message,
            "email": str(data.get("email") or "").strip() or None,
            "context": data.get("context") if isinstance(data.get("context"), dict) else {},
            "rating": data.get("rating"),
            "created_at": datetime.utcnow().replace(microsecond=0).isoformat() + "Z",
            "remote_addr": remote,
            "user_agent": str(request.headers.get("User-Agent") or ""),
        }

        try:
            path = Path(get_data_dir()) / "feedback.jsonl"
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "a", encoding="utf-8") as f:
                f.write(_json.dumps(record) + "\n")
        except Exception:
            pass

        return jsonify({"status": "success"})
    except Exception as e:
        logger.error(f"/api/system/feedback error: {e}")
        return jsonify({
            "status": "error",
            "message": "Failed to submit feedback"
        }), 502


@app.route('/api/system/version', methods=['GET'])
def get_system_version():
    return jsonify({
        "status": "success",
        "data": {
            "version": get_version(),
        }
    })


# Lightweight RTSP connectivity check
@app.route('/api/test-rtsp', methods=['POST', 'OPTIONS'])
def test_rtsp_endpoint():
    if request.method == 'OPTIONS':
        return Response(status=204)

    try:
        data = request.get_json() or {}
        rtsp_url = data.get('rtsp_url') or data.get('url')
        if not rtsp_url:
            return jsonify({
                "success": False,
                "message": "rtsp_url is required"
            }), 400

        logger.info(f"🎯 Testing RTSP connectivity for {rtsp_url[:64]}...")
        ok = _test_rtsp_stream(rtsp_url)
        if ok:
            return jsonify({
                "success": True,
                "message": "RTSP connection successful"
            })
        return jsonify({
            "success": False,
            "message": "Failed to establish RTSP connection"
        }), 400
    except Exception as e:
        logger.error(f"❌ RTSP test error: {e}")
        return jsonify({
            "success": False,
            "message": f"RTSP test error: {str(e)}"
        }), 500


# Get all devices (alias for cameras for compatibility)
@app.route('/api/devices', methods=['GET'])
def get_devices():
    # Transform cameras to device format for compatibility
    devices = []
    for camera in cameras_db:
        # Keep list endpoint lightweight: do NOT block on MediaMTX readiness checks.
        camera_snapshot = serialize_camera(camera, include_mediamtx=False)
        device = {
            **camera_snapshot,
            "type": "camera",
            "device_type": "camera"
        }
        devices.append(device)

    return jsonify({
        "status": "success",
        "data": devices,
        "count": len(devices)
    })


# Get specific camera (compatible with frontend)
@app.route('/api/cameras/<camera_id>', methods=['GET'])
def get_camera(camera_id):
    camera = next((c for c in cameras_db if c['id'] == camera_id), None)
    if camera:
        return jsonify({
            "status": "success",
            "data": serialize_camera(camera, force_mediamtx_check=True)
        })
    return jsonify({
        "status": "error",
        "message": "Camera not found"
    }), 404


# Get specific device (alias for camera)
@app.route('/api/devices/<device_id>', methods=['GET'])
def get_device(device_id):
    camera = next((c for c in cameras_db if c['id'] == device_id), None)
    if camera:
        camera_snapshot = serialize_camera(camera)
        device = {
            **camera_snapshot,
            "type": "camera",
            "device_type": "camera"
        }
        return jsonify({
            "status": "success",
            "data": device
        })
    return jsonify({
        "status": "error",
        "message": "Device not found"
    }), 404


# Add new camera with MediaMTX integration (compatible with frontend)
@app.route('/api/cameras', methods=['POST'])
def add_camera():
    try:
        data = request.get_json()

        if not data:
            return jsonify({
                "status": "error",
                "message": "No data provided"
            }), 400

        limit_state = _camera_limit_state()
        if len(limit_state["enabled"]) >= limit_state["limit"]:
            return jsonify({
                "status": "error",
                "message": f"Camera limit reached ({limit_state['limit']}). Disable a camera to stay within the beta limit."
            }), 403

        # Handle different field name formats from frontend
        ip_address = data.get('ip_address') or data.get('ip') or data.get('ipAddress')
        username = data.get('username') or data.get('user') or 'admin'
        password = data.get('password') or data.get('pass') or ''
        name = data.get('name') or f"Camera {len(cameras_db) + 1}"
        location = data.get('location') or 'Default'

        # Validate required fields
        if not ip_address:
            return jsonify({
                "status": "error",
                "message": "IP address is required"
            }), 400

        # Generate camera ID
        camera_id = str(uuid.uuid4())

        camera = {
            "id": camera_id,
            "name": name,
            "ip_address": ip_address,
            "ip": ip_address,  # Keep both for compatibility
            "port": data.get('port', 554),
            "username": username,
            "password": password,
            "stream_path": _normalize_stream_path(data.get('stream_path')) or '/media/video1',
            "rtsp_url": str(data.get('rtsp_url') or '').strip(),
            "location": location,
            "enabled": True,
            "status": "offline",
            "type": "camera",
            "device_type": "camera",
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
            # Additional fields for compatibility
            "ai_analysis": data.get('ai_analysis', False),
            "recording": data.get('recording', False),
            "ptz_enabled": data.get('ptz_enabled', False),
            "stream_quality": data.get('stream_quality'),
            "motion_detection": data.get('motion_detection', True),
            "audio_enabled": data.get('audio_enabled', False),
            "night_vision": data.get('night_vision', False),
            "privacy_mask": _normalize_privacy_mask(data.get('privacy_mask')),
            "stream_priority": data.get('stream_priority'),
            "substream_path": _normalize_stream_path(data.get('substream_path')) if 'substream_path' in data else None,
            "substream_rtsp_url": str(data.get('substream_rtsp_url') or '').strip() or None,
            "manufacturer": data.get('manufacturer'),
            "protocol": data.get('protocol', 'tcp'),
            "custom_rtsp": bool(data.get('custom_rtsp', False))
        }

        _apply_camera_payload(camera, data)

        ensure_camera_stream_metadata(camera)

        # Try to create MediaMTX path(s)
        mediamtx_success = mediamtx.create_path(camera.get('mediamtx_path', camera_id), camera['rtsp_url'])
        if mediamtx_success and camera.get('substream_rtsp_url') and camera.get('mediamtx_sub_path'):
            mediamtx.create_path(camera['mediamtx_sub_path'], camera['substream_rtsp_url'])

        if mediamtx_success:
            # Generate stream URLs with MediaMTX integration
            active_path = camera.get('mediamtx_path', camera_id)
            camera['hls_url'] = f"/proxy/hls/{active_path}/index.m3u8"
            camera['webrtc_url'] = f"/proxy/webrtc/{active_path}"
            camera['webrtc_whip_url'] = f"/proxy/webrtc/{active_path}/whip"
            camera['webrtc_whep_url'] = f"/proxy/webrtc/{active_path}/whep"
            camera['webrtc_enabled'] = True
            camera['status'] = "live"
            logger.info(f"✅ MediaMTX path created successfully for camera {camera_id}")
        else:
            # Fallback if MediaMTX is not available
            camera['hls_url'] = f"http://localhost:8888/{camera_id}/index.m3u8"
            camera['webrtc_url'] = f"ws://localhost:8889/{camera_id}"
            camera['webrtc_whip_url'] = None
            camera['webrtc_whep_url'] = None
            camera['mediamtx_path'] = None
            camera['webrtc_enabled'] = False
            camera['status'] = "offline"
            logger.warning(f"⚠️ MediaMTX path creation failed for camera {camera_id}, WebRTC disabled")

        # Add to database
        cameras_db.append(camera)
        save_cameras()

        logger.info(f"✅ Camera added: {camera['name']} (ID: {camera_id}, WebRTC: {camera['webrtc_enabled']})")

        return jsonify({
            "status": "success",
            "data": camera,
            "message": "Camera added successfully"
        }), 201

    except Exception as e:
        logger.error(f"❌ Error adding camera: {e}")
        return jsonify({
            "status": "error",
            "message": f"Error adding camera: {str(e)}"
        }), 500


# Add device (alias for camera)
@app.route('/api/devices', methods=['POST'])
def add_device():
    # Route to add_camera for devices of type camera
    data = request.get_json()
    if data and data.get('type') == 'camera':
        return add_camera()

    return jsonify({
        "status": "error",
        "message": "Only camera devices are supported"
    }), 400


# Update camera
@app.route('/api/cameras/<camera_id>', methods=['PUT', 'PATCH'])
def update_camera(camera_id):
    try:
        camera = next((c for c in cameras_db if c['id'] == camera_id), None)
        if not camera:
            return jsonify({
                "status": "error",
                "message": "Camera not found"
            }), 404

        data = request.get_json()
        if not data:
            return jsonify({
                "status": "error",
                "message": "No data provided"
            }), 400

        connection_changed = _apply_camera_payload(camera, data)
        ensure_camera_stream_metadata(camera)

        # Update MediaMTX path if connection details changed
        if connection_changed:
            rebuilt_rtsp_url = _build_rtsp_url_from_camera_fields(camera)
            if rebuilt_rtsp_url:
                camera['rtsp_url'] = rebuilt_rtsp_url

            # Delete old path and create new one
            mediamtx.delete_path(camera.get('mediamtx_path', camera_id))
            mediamtx_success = mediamtx.create_path(camera.get('mediamtx_path', camera_id), camera['rtsp_url'])
            if camera.get('mediamtx_sub_path'):
                mediamtx.delete_path(camera['mediamtx_sub_path'])
                if camera.get('substream_rtsp_url'):
                    mediamtx.create_path(camera['mediamtx_sub_path'], camera['substream_rtsp_url'])

            if mediamtx_success:
                camera['webrtc_enabled'] = True
                camera['status'] = 'live' if camera['enabled'] else 'offline'
                logger.info(f"✅ Updated MediaMTX path for camera {camera_id}")
            else:
                camera['webrtc_enabled'] = False
                logger.warning(f"⚠️ Failed to update MediaMTX path for camera {camera_id}")

        save_cameras()

        return jsonify({
            "status": "success",
            "data": camera,
            "message": "Camera updated successfully"
        })

    except Exception as e:
        logger.error(f"❌ Error updating camera: {e}")
        return jsonify({
            "status": "error",
            "message": f"Error updating camera: {str(e)}"
        }), 500


# Update device (alias for camera)
@app.route('/api/devices/<device_id>', methods=['PUT', 'PATCH'])
def update_device(device_id):
    return update_camera(device_id)


# Delete camera
@app.route('/api/cameras/<camera_id>', methods=['DELETE'])
def delete_camera(camera_id):
    try:
        global cameras_db
        camera = next((c for c in cameras_db if c['id'] == camera_id), None)
        if not camera:
            return jsonify({
                "status": "error",
                "message": "Camera not found"
            }), 404

        # Delete from MediaMTX
        mediamtx.delete_path(camera.get('mediamtx_path', camera_id))
        if camera.get('mediamtx_sub_path'):
            mediamtx.delete_path(camera['mediamtx_sub_path'])

        # Remove from database
        cameras_db = [c for c in cameras_db if c['id'] != camera_id]
        save_cameras()

        logger.info(f"✅ Deleted camera {camera_id}")

        return jsonify({
            "status": "success",
            "message": "Camera deleted successfully"
        })

    except Exception as e:
        logger.error(f"❌ Error deleting camera: {e}")
        return jsonify({
            "status": "error",
            "message": f"Error deleting camera: {str(e)}"
        }), 500


# Delete device (alias for camera)
@app.route('/api/devices/<device_id>', methods=['DELETE'])
def delete_device(device_id):
    return delete_camera(device_id)


# Toggle continuous recording for a camera
@app.route('/api/cameras/<camera_id>/recording', methods=['POST'])
def toggle_camera_recording(camera_id):
    """Enable or disable continuous recording (MediaMTX passthrough)."""
    try:
        camera = next((c for c in cameras_db if c['id'] == camera_id), None)
        if not camera:
            return jsonify({"success": False, "message": "Camera not found"}), 404

        data = request.get_json(silent=True) or {}
        enable = data.get("record", not camera.get("recording", False))

        # Accept optional per-camera recording directory from client
        if "recording_dir" in data:
            camera["recording_dir"] = data["recording_dir"]

        ok = False
        path_payload = _build_recording_payload(camera, enable)

        if camera_manager:
            loop = getattr(camera_manager, '_event_loop', None)
            if loop and loop.is_running():
                future = asyncio.run_coroutine_threadsafe(
                    camera_manager.toggle_recording(camera_id, bool(enable)), loop
                )
                ok = future.result(timeout=10)
            else:
                ok = asyncio.get_event_loop().run_until_complete(
                    camera_manager.toggle_recording(camera_id, bool(enable))
                )
        else:
            try:
                import requests as _req
                mtx_base = MEDIAMTX_API_URL.rstrip("/")
                logger.info("Recording toggle: PATCH %s/config/paths/patch/%s payload=%s",
                            mtx_base, camera_id, path_payload)
                resp = _req.patch(
                    f"{mtx_base}/config/paths/patch/{camera_id}",
                    json=path_payload,
                    timeout=5,
                )
                if resp.status_code == 200:
                    ok = True
                elif resp.status_code == 404:
                    logger.info("Recording toggle: path not found, trying ADD")
                    resp2 = _req.post(
                        f"{mtx_base}/config/paths/add/{camera_id}",
                        json=path_payload,
                        timeout=5,
                    )
                    ok = resp2.status_code in (200, 201)
                    if not ok:
                        logger.error("MediaMTX add path returned %s: %s", resp2.status_code, resp2.text[:500] if resp2.text else "")
                else:
                    logger.error("MediaMTX PATCH returned %s: %s", resp.status_code, resp.text[:500] if resp.text else "")
            except Exception as mtx_err:
                logger.error("Direct MediaMTX recording toggle failed: %s", mtx_err)

        if not ok:
            # MediaMTX v1.17+ has an API bug where PATCH/POST always rejects
            # recordPath.  Fall back to editing the YAML config directly.
            logger.info("Recording toggle: API failed, falling back to YAML config write")
            ok = _toggle_recording_via_yml(camera_id, camera, enable)

        if not ok:
            restarted = _try_restart_mediamtx()
            if restarted:
                ok = _toggle_recording_via_yml(camera_id, camera, enable)

        if ok:
            camera['recording'] = bool(enable)
            save_cameras()
            return jsonify({"success": True, "recording": bool(enable)})

        detail = (
            "Could not toggle recording via MediaMTX API or YAML config. "
            "Ensure MediaMTX is running and the config file is writable."
        )
        return jsonify({"success": False, "message": detail}), 500
    except Exception as e:
        logger.error("Error toggling recording for %s: %s", camera_id, e)
        return jsonify({"success": False, "message": str(e)}), 500


@app.route('/api/cameras/recording-status', methods=['GET'])
def cameras_recording_status():
    """Lightweight endpoint returning recording flags and per-camera paths."""
    statuses = {}
    for cam in cameras_db:
        statuses[cam.get('id', '')] = cam.get('recording', False)
    return jsonify({"success": True, "data": statuses})


# Test camera connection
@app.route('/api/cameras/<camera_id>/test', methods=['POST'])
def test_camera(camera_id):
    camera = next((c for c in cameras_db if c['id'] == camera_id), None)
    if not camera:
        return jsonify({
            "status": "error",
            "message": "Camera not found"
        }), 404

    try:
        logger.info(f"🔧 Testing camera connection for {camera['name']} ({camera_id})")
        
        # Always try to connect/refresh camera in MediaMTX to ensure stream is ready
        logger.info(f"🔗 Connecting/refreshing {camera['name']} in MediaMTX...")
        connect_result = connect_camera_to_mediamtx(camera_id)
        if connect_result:
            logger.info(f"✅ Successfully connected {camera['name']} to MediaMTX")
        else:
            logger.warning(f"⚠️ Could not connect {camera['name']} to MediaMTX")
        
        # Test MediaMTX path
        path_info = mediamtx.get_path_info(camera_id)
        
        # Also try to test RTSP connection directly
        rtsp_test = False
        if camera.get('rtsp_url'):
            try:
                import cv2
                cap = cv2.VideoCapture(camera['rtsp_url'])
                if cap.isOpened():
                    ret, frame = cap.read()
                    if ret and frame is not None:
                        rtsp_test = True
                        logger.info(f"✅ RTSP test successful for {camera['name']}")
                    cap.release()
                else:
                    logger.warning(f"⚠️ Could not open RTSP stream for {camera['name']}")
            except Exception as e:
                logger.warning(f"⚠️ RTSP test failed for {camera['name']}: {e}")

        if path_info or rtsp_test:
            camera['status'] = 'live'
            camera['ready'] = True
            camera['mediamtx_ready'] = path_info is not None
            camera['last_seen'] = datetime.now().isoformat()
            message = f"Camera {camera['name']} connection test successful"
            success = True
            logger.info(f"✅ {message}")
        else:
            camera['status'] = 'offline'
            camera['ready'] = False
            camera['mediamtx_ready'] = False
            message = f"Camera {camera['name']} connection test failed - no active stream found"
            success = False
            logger.warning(f"❌ {message}")

        save_cameras()

        return jsonify({
            "status": "success" if success else "error",
            "message": message,
            "data": {
                "rtsp_url": camera['rtsp_url'],
                "mediamtx_path_info": path_info,
                "rtsp_test_passed": rtsp_test,
                "camera_status": camera['status']
            }
        })

    except Exception as e:
        camera['status'] = 'error'
        camera['ready'] = False
        save_cameras()
        error_msg = f"Camera {camera['name']} connection failed: {str(e)}"
        logger.error(f"❌ {error_msg}")
        return jsonify({
            "status": "error",
            "message": error_msg
        }), 500


# Get camera snapshot
@app.route('/api/cameras/<camera_id>/snapshot', methods=['GET'])
def get_camera_snapshot(camera_id):
    """Get camera snapshot"""
    camera = next((c for c in cameras_db if c['id'] == camera_id), None)
    if not camera:
        return jsonify({'success': False, 'message': 'Camera not found'}), 404
    
    try:
        from PIL import Image, ImageDraw, ImageFont
        import io
        import time
        
        # Prefer the stream server's latest frame if available. This avoids slow/unreliable
        # per-request RTSP opens (which often fall back to a gray status image).
        try:
            if stream_server and camera.get('rtsp_url'):
                # Start stream on-demand if needed
                if hasattr(stream_server, 'active_streams') and camera_id not in stream_server.active_streams:
                    try:
                        _run_coro_safe(stream_server.start_stream(camera_id, {
                            'rtsp_url': camera.get('rtsp_url'),
                            'webrtc_enabled': False,
                            'fps': 15
                        }))
                        # Give capture loop a moment to populate last_frame
                        time.sleep(0.25)
                    except Exception as e:
                        logger.warning(f"Failed to start stream_server stream for snapshot: {e}")

                frame_bytes = stream_server.get_frame(camera_id)
                if frame_bytes:
                    img_io = io.BytesIO(frame_bytes)
                    img_io.seek(0)
                    from flask import send_file
                    return send_file(img_io, mimetype='image/jpeg')
        except Exception as e:
            logger.warning(f"stream_server snapshot failed for camera {camera_id}: {e}")
        
        # Fallback: Create a status image showing camera info
        width, height = 800, 600
        image = Image.new('RGB', (width, height), color='#2a2a2a')
        draw = ImageDraw.Draw(image)
        
        try:
            font = ImageFont.load_default()
        except:
            font = None
        
        # Add camera info
        draw.rectangle([10, 10, width-10, 80], fill='#1a1a1a', outline='#444')
        draw.text((20, 20), f"Camera: {camera['name']}", fill='#fff', font=font)
        draw.text((20, 40), f"ID: {camera_id}", fill='#aaa', font=font)
        draw.text((20, 60), f"Status: {camera.get('status', 'Unknown')}", fill='#0f0', font=font)
        
        # Add connection info
        if camera.get('ready'):
            draw.text((20, 100), f"Stream: Ready", fill='#0f0', font=font)
        else:
            draw.text((20, 100), f"Stream: Not Ready", fill='#f00', font=font)
        
        if camera.get('readers_count', 0) > 0:
            draw.text((20, 120), f"Viewers: {camera.get('readers_count', 0)}", fill='#fff', font=font)
        
        # Add timestamp
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        draw.text((20, height-30), f"Snapshot: {timestamp}", fill='#888', font=font)
        
        # Convert to bytes
        img_io = io.BytesIO()
        image.save(img_io, 'JPEG', quality=85)
        img_io.seek(0)
        
        logger.info(f"Generated status snapshot for camera {camera_id}")
        from flask import send_file
        return send_file(img_io, mimetype='image/jpeg')
        
    except Exception as e:
        logger.error(f"Error generating snapshot for camera {camera_id}: {e}")
        return jsonify({'success': False, 'message': f'Failed to capture snapshot: {str(e)}'}), 500


# Get camera stream URLs with WebRTC support (and configure on-demand)
@app.route('/api/cameras/<camera_id>/stream', methods=['GET'])
def get_camera_stream(camera_id):
    if not _is_camera_allowed(camera_id):
        return jsonify({
            "status": "error",
            "message": "Camera limit reached. This camera is not enabled for streaming."
        }), 403
    camera = next((c for c in cameras_db if c['id'] == camera_id), None)
    if not camera:
        return jsonify({
            "status": "error",
            "message": "Camera not found"
        }), 404

    # Ensure MediaMTX path exists (On-Demand Configuration)
    try:
        rtsp_url = camera.get('rtsp_url')
        if rtsp_url:
            main_path = camera.get('mediamtx_path', camera_id)
            
            # Only try to create if it's likely not ready, or just always ensure (idempotent)
            # Since we have connection pooling now, a quick check/create is cheap
            created = mediamtx.create_path(main_path, rtsp_url)
            if created:
                logger.info(f"✅ [On-Demand] Configured MediaMTX path for {camera.get('name')}")
                # Update status immediately so frontend sees it as 'online'
                camera['status'] = 'online'
                camera['ready'] = True
            else:
                logger.warning(f"⚠️ [On-Demand] Failed to configure path for {camera.get('name')}")
                
            # Handle substream if present
            if camera.get('substream_rtsp_url') and camera.get('mediamtx_sub_path'):
                sub_path = camera['mediamtx_sub_path']
                mediamtx.create_path(sub_path, camera['substream_rtsp_url'])
                
    except Exception as e:
        logger.error(f"Error configuring on-demand path for {camera_id}: {e}")

    # Get ICE servers
    ice_servers = mediamtx.get_ice_servers()

    # Determine which stream to serve based on priority
    priority = _normalize_stream_priority(camera.get('stream_priority'))
    use_sub = (priority == 'sub' and camera.get('webrtc_sub_whep_url'))

    if use_sub:
        active_whep = camera.get('webrtc_sub_whep_url')
        active_hls = camera.get('hls_sub_url')
        active_path = camera.get('mediamtx_sub_path')
    else:
        active_whep = camera.get('webrtc_whep_url')
        active_hls = camera.get('hls_url')
        active_path = camera.get('mediamtx_path')

    return jsonify({
        "status": "success",
        "data": {
            "camera_id": camera_id,
            "name": camera['name'],
            "rtsp_url": camera['rtsp_url'],
            "hls_url": active_hls,
            "webrtc_url": camera.get('webrtc_url'),
            "webrtc_whip_url": camera.get('webrtc_whip_url'),
            "webrtc_whep_url": active_whep,
            "webrtc_enabled": camera.get('webrtc_enabled', False),
            "ice_servers": ice_servers,
            "status": camera.get('status', 'online'),
            "mediamtx_path": active_path,
            "stream_priority": priority,
            "main_whep_url": camera.get('webrtc_whep_url'),
            "sub_whep_url": camera.get('webrtc_sub_whep_url'),
            "main_hls_url": camera.get('hls_url'),
            "sub_hls_url": camera.get('hls_sub_url'),
        }
    })


# WebRTC offer endpoint for signaling
@app.route('/api/cameras/<camera_id>/webrtc/offer', methods=['POST'])
def webrtc_offer(camera_id):
    if not _is_camera_allowed(camera_id):
        return jsonify({
            "status": "error",
            "message": "Camera limit reached. This camera is not enabled for WebRTC."
        }), 403
    camera = next((c for c in cameras_db if c['id'] == camera_id), None)
    if not camera:
        return jsonify({
            "status": "error",
            "message": "Camera not found"
        }), 404

    if not camera.get('webrtc_enabled'):
        return jsonify({
            "status": "error",
            "message": "WebRTC not enabled for this camera"
        }), 400

    try:
        data = request.get_json()
        if not data or 'offer' not in data:
            return jsonify({
                "status": "error",
                "message": "WebRTC offer required"
            }), 400

        return jsonify({
            "status": "success",
            "data": {
                "camera_id": camera_id,
                "webrtc_url": f"/proxy/webrtc/{camera_id}",
                "whep_url": f"/proxy/webrtc/{camera_id}/whep",
                "ice_servers": mediamtx.get_ice_servers(),
                "message": "Use the whep_url for WebRTC connection"
            }
        })

    except Exception as e:
        logger.error(f"❌ WebRTC offer error: {e}")
        return jsonify({
            "status": "error",
            "message": f"WebRTC offer failed: {str(e)}"
        }), 500


# Desktop-only beta: web UI is not served from this process.
def _serve_knoxnet_ui():
    return jsonify({
        "status": "disabled",
        "message": "Web UI is disabled in the desktop-only beta."
    }), 404


@app.route('/')
def knoxnet_root():
    return _serve_knoxnet_ui()


@app.errorhandler(404)
def knoxnet_spa_fallback(error):
    return error


# Simple camera management page
@app.route('/mediamtx')
@app.route('/mediamtx-dashboard')
def mediamtx_dashboard():
    return jsonify({
        "status": "disabled",
        "message": "Web dashboards are disabled in the desktop-only beta."
    }), 404
    # Legacy dashboard HTML kept for reference; unreachable when disabled.
    if False:
        card_html = f"""
            <div class="camera">
                <div class="camera-header">
                    <h3>{camera["name"]}
                        {webrtc_badge}
                    </h3>
                    <span class="status {status}">{status}</span>
                </div>

                <div class="info-grid">
                    <div class="info-item"><strong>ID:</strong> {camera["id"][:8]}...</div>
                    <div class="info-item"><strong>IP:</strong> {ip_address}</div>
                    <div class="info-item"><strong>Port:</strong> {port}</div>
                    <div class="info-item"><strong>Location:</strong> {location}</div>
                </div>

                <div class="stream-urls">
                    <p><strong>RTSP:</strong> <code>{rtsp_url}</code></p>
                    {hls_row}
                    {whep_row}
                </div>

                <div class="button-group">
                    <button class="secondary" onclick="testCamera('{camera["id"]}')">Test Connection</button>
                    <button class="secondary" onclick="getStreamInfo('{camera["id"]}')">Stream Info</button>
                    <button class="danger" onclick="deleteCamera('{camera["id"]}')">Delete</button>
                </div>
            </div>
        """
        camera_cards.append(card_html.strip())

    cameras_html = "\n".join(camera_cards) if camera_cards else '<p style="text-align: center; color: #999;">No cameras configured yet.</p>'

    return f'''
    <!DOCTYPE html>
    <html>
    <head>
        <title>MediaMTX Camera Manager</title>
        <style>
            body {{ 
                font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
                margin: 0;
                padding: 20px;
                background: #f5f5f5;
            }}
            .container {{
                max-width: 1200px;
                margin: 0 auto;
            }}
            h1 {{
                color: #333;
                border-bottom: 3px solid #4CAF50;
                padding-bottom: 10px;
            }}
            .status-bar {{ 
                background: white;
                padding: 15px;
                margin-bottom: 20px;
                border-radius: 8px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }}
            .form-container {{
                background: white;
                padding: 20px;
                border-radius: 8px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                margin-bottom: 20px;
            }}
            .form-row {{
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                gap: 15px;
                margin-bottom: 15px;
            }}
            .form-group {{
                display: flex;
                flex-direction: column;
            }}
            label {{
                font-weight: 600;
                margin-bottom: 5px;
                color: #555;
                font-size: 14px;
            }}
            input {{
                padding: 10px;
                border: 1px solid #ddd;
                border-radius: 4px;
                font-size: 14px;
            }}
            input:focus {{
                outline: none;
                border-color: #4CAF50;
                box-shadow: 0 0 0 2px rgba(76, 175, 80, 0.1);
            }}
            button {{
                background: #4CAF50;
                color: white;
                border: none;
                padding: 10px 20px;
                border-radius: 4px;
                cursor: pointer;
                font-weight: 600;
                transition: background 0.2s;
            }}
            button:hover {{
                background: #45a049;
            }}
            button.danger {{
                background: #f44336;
            }}
            button.danger:hover {{
                background: #da190b;
            }}
            button.secondary {{
                background: #2196F3;
            }}
            button.secondary:hover {{
                background: #0b7dda;
            }}
            .camera {{
                background: white;
                padding: 20px;
                margin: 15px 0;
                border-radius: 8px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }}
            .camera-header {{
                display: flex;
                justify-content: space-between;
                align-items: center;
                margin-bottom: 15px;
            }}
            .camera h3 {{
                margin: 0;
                color: #333;
            }}
            .status {{
                padding: 4px 12px;
                border-radius: 20px;
                color: white;
                font-weight: 600;
                font-size: 12px;
                text-transform: uppercase;
            }}
            .status.live {{ background: #4CAF50; }}
            .status.offline {{ background: #f44336; }}
            .status.error {{ background: #ff9800; }}
            .webrtc-enabled {{
                background: #2196F3;
                color: white;
                padding: 4px 10px;
                border-radius: 20px;
                font-size: 11px;
                text-transform: uppercase;
                margin-left: 10px;
            }}
            .stream-urls {{
                background: #f9f9f9;
                padding: 15px;
                margin: 15px 0;
                border-radius: 4px;
                border-left: 4px solid #4CAF50;
            }}
            .stream-urls p {{
                margin: 8px 0;
                font-size: 13px;
            }}
            code {{
                background: #e8e8e8;
                padding: 3px 6px;
                border-radius: 3px;
                font-family: "Courier New", monospace;
                font-size: 12px;
            }}
            .button-group {{
                display: flex;
                gap: 10px;
                margin-top: 15px;
            }}
            .info-grid {{
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
                gap: 10px;
                margin: 10px 0;
            }}
            .info-item {{
                font-size: 14px;
            }}
            .info-item strong {{
                color: #666;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🎥 MediaMTX Camera Manager</h1>

            <div class="status-bar">
                <strong>System Status:</strong> Flask Running | 
                <strong>MediaMTX:</strong> {mediamtx_status} | 
                <strong>Cameras:</strong> {len(cameras_db)} | 
                <strong>Time:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
            </div>

            <div class="form-container">
                <h2>Add New Camera</h2>
                <form id="addForm">
                    <div class="form-row">
                        <div class="form-group">
                            <label for="name">Camera Name</label>
                            <input type="text" id="name" placeholder="Living Room Camera">
                        </div>
                        <div class="form-group">
                            <label for="ip">IP Address *</label>
                            <input type="text" id="ip" placeholder="192.168.1.100" required>
                        </div>
                    </div>
                    <div class="form-row">
                        <div class="form-group">
                            <label for="username">Username *</label>
                            <input type="text" id="username" value="admin" placeholder="admin" required>
                        </div>
                        <div class="form-group">
                            <label for="password">Password *</label>
                            <input type="password" id="password" placeholder="password" required>
                        </div>
                    </div>
                    <div class="form-row">
                        <div class="form-group">
                            <label for="port">Port</label>
                            <input type="number" id="port" value="554">
                        </div>
                        <div class="form-group">
                            <label for="stream_path">Stream Path</label>
                            <input type="text" id="stream_path" value="/Streaming/Channels/101">
                        </div>
                        <div class="form-group">
                            <label for="location">Location</label>
                            <input type="text" id="location" placeholder="Living Room">
                        </div>
                    </div>
                    <button type="submit">Add Camera</button>
                </form>
            </div>

            <h2>Cameras ({len(cameras_db)})</h2>
            <div id="cameras">
                {cameras_html}
            </div>
        </div>

        <script>
            document.getElementById('addForm').onsubmit = async (e) => {{
                e.preventDefault();

                const data = {{
                    name: document.getElementById('name').value || `Camera ${{Date.now()}}`,
                    ip_address: document.getElementById('ip').value,
                    username: document.getElementById('username').value,
                    password: document.getElementById('password').value,
                    port: parseInt(document.getElementById('port').value) || 554,
                    stream_path: document.getElementById('stream_path').value || '/Streaming/Channels/101',
                    location: document.getElementById('location').value || 'Default'
                }};

                try {{
                    const response = await fetch('/api/cameras', {{
                        method: 'POST',
                        headers: {{'Content-Type': 'application/json'}},
                        body: JSON.stringify(data)
                    }});

                    const result = await response.json();

                    if (result.status === 'success') {{
                        const webrtcStatus = result.data.webrtc_enabled ? ' (WebRTC enabled)' : ' (WebRTC disabled)';
                        alert('Camera added successfully!' + webrtcStatus);
                        location.reload();
                    }} else {{
                        alert('Error: ' + result.message);
                    }}
                }} catch (error) {{
                    alert('Error: ' + error.message);
                }}
            }};

            async function testCamera(id) {{
                try {{
                    const response = await fetch(`/api/cameras/${{id}}/test`, {{method: 'POST'}});
                    const result = await response.json();
                    alert(result.message);
                    if (result.status === 'success') location.reload();
                }} catch (error) {{
                    alert('Error: ' + error.message);
                }}
            }}

            async function getStreamInfo(id) {{
                try {{
                    const response = await fetch(`/api/cameras/${{id}}/stream`);
                    const result = await response.json();

                    if (result.status === 'success') {{
                        const info = result.data;
                        let message = `Stream Info for ${{info.name}}:\\n\\n`;
                        message += `Status: ${{info.status}}\\n`;
                        message += `WebRTC Enabled: ${{info.webrtc_enabled}}\\n\\n`;

                        if (info.webrtc_enabled) {{
                            message += `HLS URL: ${{info.hls_url}}\\n`;
                            message += `WebRTC URL: ${{info.webrtc_url}}\\n`;
                            message += `WHEP URL: ${{info.webrtc_whep_url}}\\n`;
                        }} else {{
                            message += `WebRTC is not enabled for this camera.\\n`;
                            message += `Please check MediaMTX connection.`;
                        }}

                        alert(message);
                    }} else {{
                        alert('Error: ' + result.message);
                    }}
                }} catch (error) {{
                    alert('Error: ' + error.message);
                }}
            }}

            async function deleteCamera(id) {{
                if (confirm('Are you sure you want to delete this camera?')) {{
                    try {{
                        const response = await fetch(`/api/cameras/${{id}}`, {{method: 'DELETE'}});
                        const result = await response.json();
                        alert(result.message);
                        if (result.status === 'success') location.reload();
                    }} catch (error) {{
                        alert('Error: ' + error.message);
                    }}
                }}
            }}

            // Auto-refresh status every 30 seconds
            setInterval(() => {{
                console.log('Auto-refreshing camera status...');
                // You could implement a more sophisticated status update here
            }}, 30000);
        </script>
    </body>
    </html>
    '''

@app.route('/api/ai/models/health', methods=['GET'])
def ai_models_health():
    """Get health status of AI models"""
    try:
        if not ai_analyzer:
            return jsonify({
                'success': False,
                'message': 'AI analyzer not available'
            }), 503
        
        # Get model health information
        health_data = {
            'status': 'healthy',
            'models': [],
            'total_models': 0,
            'active_models': 0,
            'last_check': datetime.now().isoformat()
        }
        
        try:
            if hasattr(ai_analyzer, 'get_models_health'):
                health_data = ai_analyzer.get_models_health()
            elif hasattr(ai_analyzer, 'get_available_models'):
                models = ai_analyzer.get_available_models()
                health_data['models'] = [{'name': m, 'status': 'available'} for m in models]
                health_data['total_models'] = len(models)
                health_data['active_models'] = len(models)
        except Exception as e:
            health_data['status'] = 'error'
            health_data['error'] = str(e)
        
        return jsonify({
            'success': True,
            'data': health_data
        })
    except Exception as e:
        logger.error(f"AI models health error: {e}")
        return jsonify({
            'success': False,
            'message': f'AI models health error: {str(e)}'
        }), 500

@app.route('/api/ai/available-models', methods=['GET'])
def api_available_models():
    try:
        models = []
        
        # Try new detector_manager first
        try:
            from core.detector_manager import get_detector_manager
            detector_mgr = get_detector_manager()
            available = detector_mgr.get_available_models()
            
            # Add MobileNet SSD as primary (fixed - now working correctly)
            models.append({'id': 'mobilenet', 'name': 'MobileNet SSD (Default - Fast & Lightweight)', 'type': 'built-in'})
            
            # Add YOLO models from ai_analyzer
            if 'ai_analyzer' in globals() and ai_analyzer is not None:
                try:
                    names = ai_analyzer.get_available_models() if hasattr(ai_analyzer, 'get_available_models') else []
                    for n in names:
                        if n not in ['mobilenet']:  # Don't duplicate
                            display_name = n.replace('yolov8', 'YOLOv8-').upper() if 'yolo' in n.lower() else n
                            models.append({'id': n, 'name': display_name, 'type': 'yolo'})
                except Exception as e:
                    logger.warning(f"AI analyzer models read failed: {e}")
            
            # Add custom uploaded models from detector_mgr (treat anything not default as custom)
            for model_name in available or []:
                try:
                    name_lc = str(model_name).strip()
                    # Skip obvious defaults
                    if 'default' in name_lc.lower() or name_lc.lower() in ['mobilenet', 'yolov8n', 'yolov8s', 'yolov8m', 'yolov8l', 'yolov8x']:
                        continue
                    clean_name = name_lc.replace('model_', '')
                    if not any(m.get('id') == clean_name for m in models):
                        models.append({'id': clean_name, 'name': f'{clean_name} (Custom)', 'type': 'custom'})
                except Exception:
                    continue
                        
        except Exception as e:
            logger.warning(f"Detector manager not available: {e}")
            # Fallback to ai_analyzer only
            if 'ai_analyzer' in globals() and ai_analyzer is not None:
                try:
                    names = ai_analyzer.get_available_models() if hasattr(ai_analyzer, 'get_available_models') else []
                    models = [{'id': n, 'name': n, 'type': 'yolo'} for n in names]
                except Exception:
                    pass
            
            # Always ensure mobilenet is available as fallback
            if not any(m.get('id') == 'mobilenet' for m in models):
                models.insert(0, {'id': 'mobilenet', 'name': 'MobileNet SSD (Default)', 'type': 'built-in'})
        
        return jsonify({'success': True, 'data': models})
    except Exception as e:
        logger.error(f"available-models error: {e}")
        return jsonify({'success': False, 'message': str(e)}), 500

@app.route('/api/cameras/<camera_id>/detection-config', methods=['GET'])
def api_get_detection_config(camera_id):
    try:
        if 'STREAM_SERVER_GLOBAL' in globals() and STREAM_SERVER_GLOBAL is not None:
            stream_server = STREAM_SERVER_GLOBAL
            if hasattr(stream_server, 'get_detection_config'):
                cfg = stream_server.get_detection_config(camera_id)
                return jsonify({'success': True, 'data': cfg})
        return jsonify({'success': False, 'message': 'Stream server not available'}), 503
    except Exception as e:
        logger.error(f"detection-config get error for {camera_id}: {e}")
        return jsonify({'success': False, 'message': str(e)}), 500

@app.route('/api/cameras/<camera_id>/detection-config', methods=['PUT'])
def api_put_detection_config(camera_id):
    try:
        data = request.get_json() or {}
        if 'STREAM_SERVER_GLOBAL' in globals() and STREAM_SERVER_GLOBAL is not None:
            stream_server = STREAM_SERVER_GLOBAL
            if hasattr(stream_server, 'update_detection_config'):
                ok = stream_server.update_detection_config(camera_id, data)
                # Return the updated config for convenience
                cfg = stream_server.get_detection_config(camera_id) if ok else None
                return jsonify({'success': bool(ok), 'data': cfg or {}})
        return jsonify({'success': False, 'message': 'Stream server not available'}), 503
    except Exception as e:
        logger.error(f"detection-config put error for {camera_id}: {e}")
        return jsonify({'success': False, 'message': str(e)}), 500

@app.route('/api/ai/yolo/params', methods=['GET'])
def api_get_yolo_params():
    """Get current YOLO runtime parameters"""
    try:
        if 'ai_analyzer' in globals() and ai_analyzer is not None:
            if hasattr(ai_analyzer, 'get_yolo_params'):
                params = ai_analyzer.get_yolo_params()
                return jsonify({'success': True, 'data': params})
        return jsonify({'success': False, 'message': 'AI analyzer not available'}), 503
    except Exception as e:
        logger.error(f"get_yolo_params error: {e}")
        return jsonify({'success': False, 'message': str(e)}), 500

@app.route('/api/ai/yolo/params', methods=['PUT'])
def api_set_yolo_params():
    """Update YOLO runtime parameters"""
    try:
        data = request.get_json() or {}
        if 'ai_analyzer' in globals() and ai_analyzer is not None:
            if hasattr(ai_analyzer, 'set_yolo_params'):
                updated_params = ai_analyzer.set_yolo_params(data)
                return jsonify({'success': True, 'data': updated_params})
        return jsonify({'success': False, 'message': 'AI analyzer not available'}), 503
    except Exception as e:
        logger.error(f"set_yolo_params error: {e}")
        return jsonify({'success': False, 'message': str(e)}), 500

@app.route('/api/ai/models/custom', methods=['POST', 'OPTIONS'])
def api_load_custom_model():
    """Load a custom detection model: .pt (YOLO) or .caffemodel (MobileNet SSD)"""
    # Handle OPTIONS preflight request
    if request.method == 'OPTIONS':
        response = jsonify({'status': 'ok'})
        response.headers.add('Access-Control-Allow-Origin', '*')
        response.headers.add('Access-Control-Allow-Headers', 'Content-Type')
        response.headers.add('Access-Control-Allow-Methods', 'POST, OPTIONS')
        return response
    
    try:
        # Try detector_manager first, fallback to ai_analyzer
        from core.detector_manager import get_detector_manager
        detector_mgr = get_detector_manager()
        
        # Check if file upload or path provided
        if 'file' in request.files:
            file = request.files['file']
            if not file.filename:
                return jsonify({'success': False, 'message': 'No file selected'}), 400
            
            # Validate extension
            allowed = (file.filename.lower().endswith('.pt') or file.filename.lower().endswith('.caffemodel'))
            if not allowed:
                return jsonify({'success': False, 'message': 'Supported: .pt (YOLO), .caffemodel (MobileNet SSD)'}), 400
            
            # Save to models directory
            models_dir = os.path.join(os.path.dirname(__file__), 'models')
            os.makedirs(models_dir, exist_ok=True)
            
            # Sanitize filename
            import re
            safe_filename = re.sub(r'[^a-zA-Z0-9_\-\.]', '_', file.filename)
            model_path = os.path.join(models_dir, safe_filename)
            
            file.save(model_path)
            logger.info(f"Saved custom model to {model_path}")
            
            # Extract model name from filename
            model_name = request.form.get('name') or os.path.splitext(safe_filename)[0]
        else:
            # Path-based loading
            data = request.get_json() or {}
            model_path = data.get('path')
            model_name = data.get('name')
            
            if not model_path:
                return jsonify({'success': False, 'message': 'model path required'}), 400
            
            if not os.path.exists(model_path):
                return jsonify({'success': False, 'message': f'Model file not found: {model_path}'}), 404
        
        # Load the model using detector_manager
        model_type = 'yolo' if model_path.lower().endswith('.pt') else 'custom'
        try:
            success = detector_mgr.load_custom_model(model_name, model_path, model_type=model_type)
        except Exception as load_err:
            # Provide clearer guidance for common upload issues
            msg = str(load_err)
            if '.pt' in model_path.lower() and ('ultralytics' in msg.lower() or 'not installed' in msg.lower()):
                msg = 'Ultralytics is required for YOLO .pt models. Install with: pip install ultralytics'
            elif model_path.lower().endswith('.caffemodel') and 'prototxt' in msg.lower():
                msg = 'Missing deploy prototxt for Caffe model. Place MobileNetSSD_deploy.prototxt or matching .prototxt next to the .caffemodel'
            logger.error(f"Custom model load failed: {msg}")
            response = jsonify({'success': False, 'message': msg})
            response.headers.add('Access-Control-Allow-Origin', '*')
            return response, 500
        
        if success:
            response = jsonify({
                'success': True,
                'data': {'name': model_name, 'path': model_path},
                'message': f'Custom model "{model_name}" loaded successfully'
            })
            response.headers.add('Access-Control-Allow-Origin', '*')
            return response
        else:
            return jsonify({'success': False, 'message': 'Failed to load custom model'}), 500
    except Exception as e:
        logger.error(f"load_custom_model error: {e}")
        response = jsonify({'success': False, 'message': str(e)})
        response.headers.add('Access-Control-Allow-Origin', '*')
        return response, 500

# Allow both with and without trailing slash to avoid method mismatch on some clients
@app.route('/api/ai/models/custom/', methods=['POST', 'OPTIONS'])
def api_load_custom_model_slash():
    return api_load_custom_model()

@app.route('/api/ai/models/list', methods=['GET'])
def api_list_models():
    """Get list of available detection models"""
    try:
        from core.detector_manager import get_detector_manager
        detector_mgr = get_detector_manager()
        
        models = detector_mgr.get_available_models()
        model_info = detector_mgr.get_model_info()
        
        response = jsonify({
            'success': True,
            'data': {
                'models': models,
                'current': model_info
            }
        })
        response.headers.add('Access-Control-Allow-Origin', '*')
        return response
    except Exception as e:
        logger.error(f"list_models error: {e}")
        return jsonify({'success': False, 'message': str(e)}), 500

@app.route('/api/cameras/<camera_id>/detector', methods=['GET', 'PUT'])
def api_camera_detector(camera_id):
    """Get or set detector for a specific camera"""
    try:
        from core.detector_manager import get_detector_manager
        detector_mgr = get_detector_manager()
        
        if request.method == 'GET':
            config = detector_mgr.get_detection_config(camera_id)
            model_info = detector_mgr.get_model_info(camera_id)
            
            response = jsonify({
                'success': True,
                'data': {
                    'config': config,
                    'model_info': model_info
                }
            })
            response.headers.add('Access-Control-Allow-Origin', '*')
            return response
        
        elif request.method == 'PUT':
            data = request.get_json() or {}
            
            model = data.get('model', 'default')
            model_path = data.get('model_path')
            confidence = data.get('confidence', 0.25)
            
            # Set detector
            detector_mgr.set_camera_detector(camera_id, model, model_path)
            
            # Update config
            config = {
                'enabled': data.get('enabled', True),
                'confidence': confidence,
                'model': model,
                'classes': data.get('classes', [])
            }
            detector_mgr.set_detection_config(camera_id, config)
            
            response = jsonify({
                'success': True,
                'data': {
                    'camera_id': camera_id,
                    'config': config
                },
                'message': f'Detector updated for camera {camera_id}'
            })
            response.headers.add('Access-Control-Allow-Origin', '*')
            return response
            
    except Exception as e:
        logger.error(f"camera_detector error for {camera_id}: {e}")
        return jsonify({'success': False, 'message': str(e)}), 500

@app.route('/api/cameras/<camera_id>/shapes', methods=['POST'])
def api_set_camera_shapes(camera_id):
    """Store user-drawn shapes (zones/lines/tags) for a camera to enable ROI-based detection"""
    try:
        data = request.get_json() or {}
        zones = data.get('zones', [])
        lines = data.get('lines', [])
        tags = data.get('tags', [])
        
        # Store in stream server for ROI cropping
        if 'STREAM_SERVER_GLOBAL' in globals() and STREAM_SERVER_GLOBAL is not None:
            stream_server = STREAM_SERVER_GLOBAL
            if hasattr(stream_server, 'set_camera_shapes'):
                stream_server.set_camera_shapes(camera_id, {
                    'zones': zones,
                    'lines': lines,
                    'tags': tags
                })
        
        logger.info(f"Stored shapes for camera {camera_id}: {len(zones)} zones, {len(lines)} lines, {len(tags)} tags")
        return jsonify({
            'success': True,
            'data': {'camera_id': camera_id, 'zones': len(zones), 'lines': len(lines), 'tags': len(tags)}
        })
    except Exception as e:
        logger.error(f"set_camera_shapes error for {camera_id}: {e}")
        return jsonify({'success': False, 'message': str(e)}), 500

@app.route('/api/ai/car-counting-rules', methods=['GET'])
def get_car_counting_rules():
    """Get all car counting rules"""
    try:
        return jsonify({
            'success': True,
            'data': {
                'rules': car_counting_rules
            }
        })
    except Exception as e:
        logger.error(f"Error getting car counting rules: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/ai/car-counting-rules', methods=['POST'])
def create_car_counting_rule():
    """Create a new car counting rule"""
    try:
        data = request.get_json()
        rule_id = data.get('rule_id')
        rule_data = data.get('rule_data', {})
        
        if not rule_id:
            return jsonify({
                'success': False,
                'error': 'Rule ID is required'
            }), 400
        
        car_counting_rules[rule_id] = {
            'id': rule_id,
            'name': rule_data.get('name', f'Car Count Rule {len(car_counting_rules) + 1}'),
            'camera_id': rule_data.get('camera_id'),
            'zone_id': rule_data.get('zone_id'),
            'enabled': rule_data.get('enabled', True),
            'direction': rule_data.get('direction', 'both'),  # 'in', 'out', 'both'
            'object_types': rule_data.get('object_types', ['car', 'truck', 'bus']),
            'confidence_threshold': rule_data.get('confidence_threshold', 0.5),
            'created_at': datetime.now().isoformat(),
            'count_in': 0,
            'count_out': 0,
            'last_count_time': None
        }
        
        # Initialize count history for this rule
        car_count_history[rule_id] = []
        
        logger.info(f"Created car counting rule: {rule_id}")
        
        return jsonify({
            'success': True,
            'data': {
                'rule': car_counting_rules[rule_id]
            }
        })
    except Exception as e:
        logger.error(f"Error creating car counting rule: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/ai/car-counting-rules/<rule_id>', methods=['PUT'])
def update_car_counting_rule(rule_id):
    """Update a car counting rule"""
    try:
        if rule_id not in car_counting_rules:
            return jsonify({
                'success': False,
                'error': 'Rule not found'
            }), 404
        
        data = request.get_json()
        
        # Update allowed fields
        allowed_fields = ['name', 'enabled', 'direction', 'object_types', 'confidence_threshold']
        for field in allowed_fields:
            if field in data:
                car_counting_rules[rule_id][field] = data[field]
        
        logger.info(f"Updated car counting rule: {rule_id}")
        
        return jsonify({
            'success': True,
            'data': {
                'rule': car_counting_rules[rule_id]
            }
        })
    except Exception as e:
        logger.error(f"Error updating car counting rule: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/ai/car-counting-rules/<rule_id>', methods=['DELETE'])
def delete_car_counting_rule(rule_id):
    """Delete a car counting rule"""
    try:
        if rule_id not in car_counting_rules:
            return jsonify({
                'success': False,
                'error': 'Rule not found'
            }), 404
        
        # Remove rule and its history
        del car_counting_rules[rule_id]
        if rule_id in car_count_history:
            del car_count_history[rule_id]
        
        logger.info(f"Deleted car counting rule: {rule_id}")
        
        return jsonify({
            'success': True,
            'message': 'Rule deleted successfully'
        })
    except Exception as e:
        logger.error(f"Error deleting car counting rule: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/ai/car-counting-rules/<rule_id>/counts', methods=['GET'])
def get_car_count_history(rule_id):
    """Get car count history for a specific rule"""
    try:
        if rule_id not in car_counting_rules:
            return jsonify({
                'success': False,
                'error': 'Rule not found'
            }), 404
        
        limit = request.args.get('limit', 50, type=int)
        history = car_count_history.get(rule_id, [])
        
        return jsonify({
            'success': True,
            'data': {
                'rule': car_counting_rules[rule_id],
                'history': history[-limit:],
                'total_count_in': car_counting_rules[rule_id]['count_in'],
                'total_count_out': car_counting_rules[rule_id]['count_out']
            }
        })
    except Exception as e:
        logger.error(f"Error getting car count history: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/ai/process-car-counting', methods=['POST'])
def process_car_counting():
    """Process car counting for all active rules"""
    try:
        data = request.get_json()
        camera_id = data.get('camera_id')
        detected_objects = data.get('detected_objects', [])
        current_frame_base64 = data.get('image_base64')
        
        if not camera_id:
            return jsonify({
                'success': False,
                'error': 'Camera ID is required'
            }), 400
        
        # Get active rules for this camera
        active_rules = [
            rule for rule in car_counting_rules.values()
            if rule['camera_id'] == camera_id and rule['enabled']
        ]
        
        counting_results = []
        
        for rule in active_rules:
            # Check if any detected objects match the rule criteria
            matching_objects = []
            for obj in detected_objects:
                if (obj.get('class_name') in rule['object_types'] and 
                    obj.get('confidence', 0) >= rule['confidence_threshold']):
                    matching_objects.append(obj)
            
            if matching_objects:
                # Check if objects are in the specified zone
                zone_id = rule['zone_id']
                if zone_id:
                    # This would require zone intersection logic
                    # For now, we'll count all matching objects
                    pass
                
                # Count objects (simplified logic - in production you'd track individual objects)
                count = len(matching_objects)
                
                # Update rule counts
                if rule['direction'] in ['in', 'both']:
                    car_counting_rules[rule['id']]['count_in'] += count
                if rule['direction'] in ['out', 'both']:
                    car_counting_rules[rule['id']]['count_out'] += count
                
                car_counting_rules[rule['id']]['last_count_time'] = datetime.now().isoformat()
                
                # Add to history
                history_entry = {
                    'timestamp': datetime.now().isoformat(),
                    'count': count,
                    'objects': matching_objects,
                    'direction': rule['direction']
                }
                car_count_history[rule['id']].append(history_entry)
                
                counting_results.append({
                    'rule_id': rule['id'],
                    'rule_name': rule['name'],
                    'count': count,
                    'total_in': car_counting_rules[rule['id']]['count_in'],
                    'total_out': car_counting_rules[rule['id']]['count_out'],
                    'objects': matching_objects
                })
        
        return jsonify({
            'success': True,
            'data': {
                'counting_results': counting_results,
                'active_rules_count': len(active_rules)
            }
        })
    except Exception as e:
        logger.error(f"Error processing car counting: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/ai/create-counting-rule-from-chat', methods=['POST'])
def create_counting_rule_from_chat():
    """Create a car counting rule from AI chat interaction"""
    try:
        data = request.get_json()
        user_message = data.get('user_message', '')
        camera_id = data.get('camera_id')
        zone_id = data.get('zone_id')
        
        if not AI_AGENT_AVAILABLE:
            return jsonify({
                'success': False,
                'error': 'AI Agent not available'
            }), 500
        
        # Initialize AI agent if not already done
        if ai_agent is None:
            initialize_ai_agent()
        
        # Create a rule based on the user message
        rule_id = f"rule_{int(time.time())}"
        
        # Parse user message to extract rule parameters
        rule_name = f"Car Count Rule {len(car_counting_rules) + 1}"
        direction = 'both'
        object_types = ['car', 'truck', 'bus']
        confidence_threshold = 0.5
        
        # Simple parsing logic - in production, use AI to parse more intelligently
        if 'incoming' in user_message.lower() or 'entering' in user_message.lower():
            direction = 'in'
        elif 'outgoing' in user_message.lower() or 'leaving' in user_message.lower():
            direction = 'out'
        
        if 'truck' in user_message.lower():
            object_types = ['truck']
        elif 'bus' in user_message.lower():
            object_types = ['bus']
        elif 'vehicle' in user_message.lower():
            object_types = ['car', 'truck', 'bus']
        
        # Create the rule
        car_counting_rules[rule_id] = {
            'id': rule_id,
            'name': rule_name,
            'camera_id': camera_id,
            'zone_id': zone_id,
            'enabled': True,
            'direction': direction,
            'object_types': object_types,
            'confidence_threshold': confidence_threshold,
            'created_at': datetime.now().isoformat(),
            'count_in': 0,
            'count_out': 0,
            'last_count_time': None,
            'created_from_chat': True,
            'user_message': user_message
        }
        
        # Initialize count history
        car_count_history[rule_id] = []
        
        # Generate AI response
        ai_response = f"I've created a car counting rule for you! 🚗\n\n" \
                     f"**Rule: {rule_name}**\n" \
                     f"• Direction: {direction.title()}\n" \
                     f"• Objects: {', '.join(object_types)}\n" \
                     f"• Confidence: {confidence_threshold * 100}%\n" \
                     f"• Status: Active ✅\n\n" \
                     f"The system will now count vehicles in the specified zone and report back to you in the chat."
        
        logger.info(f"Created car counting rule from chat: {rule_id}")
        
        return jsonify({
            'success': True,
            'data': {
                'rule': car_counting_rules[rule_id],
                'ai_response': ai_response
            }
        })
    except Exception as e:
        logger.error(f"Error creating counting rule from chat: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/ai/car-counting-tool', methods=['POST'])
def ai_car_counting_tool():
    """AI agent tool for handling car counting requests"""
    try:
        logger.info("Car counting tool endpoint called")
        data = request.get_json()
        logger.info(f"Received data: {data}")
        
        camera_id = data.get('camera_id')
        user_message = data.get('user_message', '')
        image_base64 = data.get('image_base64')
        
        logger.info(f"Camera ID: {camera_id}")
        logger.info(f"User message: {user_message}")
        logger.info(f"Image provided: {image_base64 is not None}")
        
        if not camera_id:
            logger.error("Camera ID is required")
            return jsonify({
                'success': False,
                'error': 'Camera ID is required'
            }), 400
        
        if not AI_AGENT_AVAILABLE:
            logger.error("AI Agent not available")
            return jsonify({
                'success': False,
                'error': 'AI Agent not available'
            }), 500
        
        # Initialize AI agent if not already done
        if ai_agent is None:
            logger.info("Initializing AI agent")
            initialize_ai_agent()
        
        if ai_agent is None:
            logger.error("Failed to initialize AI agent")
            return jsonify({
                'success': False,
                'error': 'Failed to initialize AI agent'
            }), 500
        
        logger.info("Running car counting request")
        # Run the async function in a separate thread
        import eventlet
        real_threading = eventlet.patcher.original('threading')
        
        result_holder = {'response': None, 'error': None}
        
        def run_counting_thread():
            try:
                result_holder['response'] = _run_coro_safe(
                    ai_agent.handle_car_counting_request(camera_id, user_message, image_base64)
                )
            except Exception as e:
                result_holder['error'] = e
        
        counting_thread = real_threading.Thread(target=run_counting_thread)
        counting_thread.start()
        counting_thread.join()
        
        if result_holder['error']:
            logger.error(f"Async error: {result_holder['error']}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            raise result_holder['error']
            
        result = result_holder['response']
        logger.info(f"Car counting result: {result}")
        
        if result.get('success'):
            # Store the created rules
            rules_created = result['data'].get('rules_created', [])
            for rule in rules_created:
                car_counting_rules[rule['id']] = rule
                car_count_history[rule['id']] = []
            
            logger.info(f"AI agent created {len(rules_created)} car counting rules for camera {camera_id}")
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Error in AI car counting tool: {e}")
        import traceback
        logger.error(f"Full traceback: {traceback.format_exc()}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500



@app.route('/api/test/motion', methods=['POST'])
def test_motion_injection():
    """Test endpoint to inject motion detection data for WebSocket testing"""
    try:
        data = request.get_json()
        if not data:
            return jsonify({
                'success': False,
                'message': 'No data provided'
            }), 400
        
        camera_id = data.get('camera_id')
        motion_data = data.get('motion')
        
        if not camera_id or not motion_data:
            return jsonify({
                'success': False,
                'message': 'Missing camera_id or motion data'
            }), 400
        
        # Emit motion data via WebSocket if available - ULTRA FAST
        if socketio is not None:
            try:
                # Use fast background task for context-free emission
                def emit_test_data():
                    # Ultra-fast broadcast to all clients
                    socketio.emit('motion_update', data)
                    # Also emit to specific room for compatibility
                    socketio.emit('motion_update', data, room=f"camera:{camera_id}")
                
                # Execute emission in background task immediately
                socketio.start_background_task(emit_test_data)
                logger.info(f"✅ Test motion data emitted for camera {camera_id} (ultra-fast)")
            except Exception as e:
                logger.error(f"❌ Failed to emit test motion data: {e}")
        
        # Also try to emit via stream server if available
        try:
            if 'STREAM_SERVER_GLOBAL' in globals() and STREAM_SERVER_GLOBAL is not None:
                if hasattr(STREAM_SERVER_GLOBAL, 'on_motion_update') and STREAM_SERVER_GLOBAL.on_motion_update:
                    STREAM_SERVER_GLOBAL.on_motion_update(camera_id, data)
                    logger.info(f"✅ Test motion data sent via stream server for camera {camera_id}")
        except Exception as e:
            logger.error(f"❌ Failed to send via stream server: {e}")
        
        return jsonify({
            'success': True,
            'message': f'Test motion data injected for camera {camera_id}',
            'data': data
        })
        
    except Exception as e:
        logger.error(f"❌ Test motion injection error: {e}")
        return jsonify({
            'success': False,
            'message': f'Test motion injection error: {str(e)}'
        }), 500

@app.route('/api/test/tracks', methods=['POST'])
def test_tracks_injection():
    """Test endpoint to inject tracking data for WebSocket testing"""
    try:
        data = request.get_json()
        if not data:
            return jsonify({
                'success': False,
                'message': 'No data provided'
            }), 400
        
        camera_id = data.get('camera_id')
        tracks_data = data.get('tracks', [])
        
        if not camera_id:
            return jsonify({
                'success': False,
                'message': 'Missing camera_id'
            }), 400
        
        # Emit tracks data via WebSocket if available
        if socketio is not None:
            try:
                payload = {
                    'camera_id': camera_id,
                    'tracks': tracks_data,
                    'timestamp': datetime.now().isoformat()
                }
                # Emit to both namespaces for maximum compatibility
                socketio.emit('tracks_update', payload, to=f"camera:{camera_id}", include_self=False)
                socketio.emit('tracks_update', payload, namespace=WS_NS, to=f"camera:{camera_id}", include_self=False)
                logger.info(f"✅ Test tracks data emitted for camera {camera_id}")
            except Exception as e:
                logger.error(f"❌ Failed to emit test tracks data: {e}")
        
        return jsonify({
            'success': True,
            'message': f'Test tracks data injected for camera {camera_id}',
            'data': payload
        })
        
    except Exception as e:
        logger.error(f"❌ Test tracks injection error: {e}")
        return jsonify({
            'success': False,
            'message': f'Test tracks injection error: {str(e)}'
        }), 500

@app.route('/api/test/detections', methods=['POST'])
def test_detections_injection():
    """Test endpoint to inject object detection data for WebSocket testing"""
    try:
        data = request.get_json()
        if not data:
            return jsonify({
                'success': False,
                'message': 'No data provided'
            }), 400
        
        camera_id = data.get('camera_id')
        detections_data = data.get('detections', [])
        
        if not camera_id:
            return jsonify({
                'success': False,
                'message': 'Missing camera_id'
            }), 400
        
        # Emit detections data via WebSocket if available
        if socketio is not None:
            try:
                payload = {
                    'camera_id': camera_id,
                    'detections': detections_data,
                    'timestamp': datetime.now().isoformat()
                }
                # Emit to both namespaces for maximum compatibility
                socketio.emit('detections_update', payload, to=f"camera:{camera_id}", include_self=False)
                socketio.emit('detections_update', payload, namespace=WS_NS, to=f"camera:{camera_id}", include_self=False)
                logger.info(f"✅ Test detections data emitted for camera {camera_id}")
            except Exception as e:
                logger.error(f"❌ Failed to emit test detections data: {e}")
        
        return jsonify({
            'success': True,
            'message': f'Test detections data injected for camera {camera_id}',
            'data': payload
        })
        
    except Exception as e:
        logger.error(f"❌ Test detections injection error: {e}")
        return jsonify({
            'success': False,
            'message': f'Test detections injection error: {str(e)}'
        }), 500

@app.route('/api/test/emit', methods=['POST'])
def test_websocket_emission():
    """Test endpoint to directly emit WebSocket data for testing"""
    try:
        data = request.get_json()
        if not data:
            return jsonify({
                'success': False,
                'message': 'No data provided'
            }), 400
        
        camera_id = data.get('camera_id')
        motion_data = data.get('motion')
        
        if not camera_id or not motion_data:
            return jsonify({
                'success': False,
                'message': 'Missing camera_id or motion data'
            }), 400
        
        # Emit motion data via WebSocket if available
        if socketio is not None:
            try:
                # Use the global stream server's motion emission function if available
                if 'STREAM_SERVER_GLOBAL' in globals() and STREAM_SERVER_GLOBAL is not None:
                    if hasattr(STREAM_SERVER_GLOBAL, 'on_motion_update') and STREAM_SERVER_GLOBAL.on_motion_update:
                        STREAM_SERVER_GLOBAL.on_motion_update(camera_id, data)
                        logger.info(f"✅ Test WebSocket emission successful for camera {camera_id}")
                        return jsonify({
                            'success': True,
                            'message': f'WebSocket emission successful for camera {camera_id}',
                            'data': data
                        })
                
                # If stream server is not available, just return success for now
                # The actual motion detection should work through the stream server
                logger.info(f"✅ Test endpoint called for camera {camera_id} - motion detection should work through stream server")
                return jsonify({
                    'success': True,
                    'message': f'Test endpoint called for camera {camera_id} - motion detection should work through stream server',
                    'data': data
                })
            except Exception as e:
                logger.error(f"❌ Test WebSocket emission failed: {e}")
                return jsonify({
                    'success': False,
                    'message': f'WebSocket emission failed: {str(e)}'
                }), 500
        else:
            return jsonify({
                'success': False,
                'message': 'Socket.IO not available'
            }), 500
            
    except Exception as e:
        logger.error(f"❌ Test emission endpoint error: {e}")
        return jsonify({
            'success': False,
            'message': f'Server error: {str(e)}'
        }), 500

# Ensure all cameras are connected for motion detection
def ensure_all_cameras_motion_detection():
    """Ensure all cameras are properly connected to motion detection system"""
    try:
        if 'STREAM_SERVER_GLOBAL' in globals() and STREAM_SERVER_GLOBAL is not None:
            stream_server = STREAM_SERVER_GLOBAL
            
            # Check which cameras are active in the stream server
            active_cameras = list(getattr(stream_server, 'active_streams', {}).keys())
            logger.info(f"Active cameras in stream server: {active_cameras}")
            
            # Ensure motion detection is working for all active cameras
            for camera_id in active_cameras:
                if hasattr(stream_server, '_motion_detectors') and camera_id in stream_server._motion_detectors:
                    logger.info(f"✅ Motion detection active for camera {camera_id}")
                else:
                    logger.warning(f"⚠️ Motion detection not active for camera {camera_id}")
                    
            # Check if motion detection callbacks are properly wired
            if hasattr(stream_server, 'on_motion_update') and stream_server.on_motion_update:
                logger.info("✅ Motion detection callbacks are properly wired")
            else:
                logger.warning("⚠️ Motion detection callbacks are not wired")
                
    except Exception as e:
        logger.error(f"Error ensuring camera motion detection: {e}")

# Call this function after stream server initialization
ensure_all_cameras_motion_detection()

# Test motion detection for specific camera
@app.route('/api/cameras/<camera_id>/test-motion', methods=['POST'])
def test_camera_motion_detection(camera_id):
    """Test motion detection for a specific camera"""
    try:
        if 'STREAM_SERVER_GLOBAL' in globals() and STREAM_SERVER_GLOBAL is not None:
            stream_server = STREAM_SERVER_GLOBAL
            
            # Check if camera is active
            if camera_id not in getattr(stream_server, 'active_streams', {}):
                return jsonify({
                    'success': False,
                    'message': f'Camera {camera_id} is not active in stream server'
                }), 404
            
            # Check if motion detection is active
            if not hasattr(stream_server, '_motion_detectors') or camera_id not in stream_server._motion_detectors:
                return jsonify({
                    'success': False,
                    'message': f'Motion detection not active for camera {camera_id}'
                }), 400
            
            # Create a test motion event
            test_motion_data = {
                'camera_id': camera_id,
                'motion': {
                    'has_motion': True,
                    'score': 0.8,
                    'regions': [
                        {'x': 100, 'y': 100, 'w': 200, 'h': 150, 'area': 30000}
                    ],
                    'frame_width': 1920,
                    'frame_height': 1080
                },
                'timestamp': datetime.now().isoformat()
            }
            
            # Emit the test motion event
            if hasattr(stream_server, 'on_motion_update') and stream_server.on_motion_update:
                stream_server.on_motion_update(camera_id, test_motion_data)
                logger.info(f"✅ Test motion event emitted for camera {camera_id}")
                
                return jsonify({
                    'success': True,
                    'message': f'Test motion event emitted for camera {camera_id}',
                    'data': test_motion_data
                })
            else:
                return jsonify({
                    'success': False,
                    'message': 'Motion detection callbacks not wired'
                }), 500
        else:
            return jsonify({
                'success': False,
                'message': 'Stream server not available'
            }), 503
            
    except Exception as e:
        logger.error(f"Error testing motion detection for camera {camera_id}: {e}")
        return jsonify({
            'success': False,
            'message': f'Error testing motion detection: {str(e)}'
        }), 500

# Diagnostic endpoint to check camera status
@app.route('/api/cameras/diagnostic', methods=['GET'])
def camera_diagnostic():
    """Get diagnostic information about all cameras"""
    try:
        diagnostic_data = {
            'timestamp': datetime.now().isoformat(),
            'stream_server_available': 'STREAM_SERVER_GLOBAL' in globals() and STREAM_SERVER_GLOBAL is not None,
            'cameras': {}
        }
        
        if 'STREAM_SERVER_GLOBAL' in globals() and STREAM_SERVER_GLOBAL is not None:
            stream_server = STREAM_SERVER_GLOBAL
            
            # Get active streams
            active_streams = list(getattr(stream_server, 'active_streams', {}).keys())
            diagnostic_data['active_streams'] = active_streams
            
            # Get motion detectors
            motion_detectors = list(getattr(stream_server, '_motion_detectors', {}).keys())
            diagnostic_data['motion_detectors'] = motion_detectors
            
            # Get motion state
            motion_state = list(getattr(stream_server, '_motion_state', {}).keys())
            diagnostic_data['motion_state'] = motion_state
            
            # Check callbacks
            diagnostic_data['callbacks'] = {
                'on_motion_update': hasattr(stream_server, 'on_motion_update') and stream_server.on_motion_update is not None,
                'on_tracks_update': hasattr(stream_server, 'on_tracks_update') and stream_server.on_tracks_update is not None,
                'on_detection_update': hasattr(stream_server, 'on_detection_update') and stream_server.on_detection_update is not None
            }
            
            # Check configured cameras (or fall back to active ones)
            camera_ids = []
            try:
                with open('cameras.json', 'r') as f:
                    configured = json.load(f)
                    camera_ids = [cam.get('id') for cam in configured if cam.get('id')]
            except Exception:
                camera_ids = []

            if not camera_ids:
                unified_ids = set(active_streams) | set(motion_detectors) | set(motion_state)
                camera_ids = list(unified_ids)

            for camera_id in camera_ids:
                camera_info = {
                    'active_in_streams': camera_id in active_streams,
                    'has_motion_detector': camera_id in motion_detectors,
                    'has_motion_state': camera_id in motion_state,
                    'stream_info': None
                }
                
                if camera_id in active_streams:
                    stream_info = stream_server.active_streams[camera_id]
                    camera_info['stream_info'] = {
                        'active': stream_info.get('active', False),
                        'webrtc_enabled': stream_info.get('webrtc_enabled', False),
                        'fps': stream_info.get('fps', 0),
                        'quality': str(stream_info.get('quality', 'unknown')),
                        'clients_count': len(stream_info.get('clients', set())),
                        'capture_url': stream_info.get('capture_url', 'unknown')
                    }
                
                diagnostic_data['cameras'][camera_id] = camera_info
        
        return jsonify({
            'success': True,
            'data': diagnostic_data
        })
        
    except Exception as e:
        logger.error(f"Error getting camera diagnostic: {e}")
        return jsonify({
            'success': False,
            'message': f'Error getting diagnostic: {str(e)}'
        }), 500

# Force start all camera streams for motion detection
@app.route('/api/cameras/start-all-streams', methods=['POST'])
def start_all_camera_streams():
    """Force start streams for all cameras to enable motion detection"""
    try:
        if 'STREAM_SERVER_GLOBAL' in globals() and STREAM_SERVER_GLOBAL is not None:
            stream_server = STREAM_SERVER_GLOBAL
            
            # Load cameras from cameras.json
            try:
                with open('cameras.json', 'r') as f:
                    cameras = json.load(f)
                logger.info(f"📂 Loaded {len(cameras)} cameras from cameras.json")
            except Exception as e:
                logger.error(f"Failed to load cameras.json: {e}")
                # Try to use global cameras_db as fallback
                cameras = cameras_db if cameras_db else []
                logger.info(f"📂 Using global cameras_db: {len(cameras)} cameras")
            
            if not cameras:
                return jsonify({
                    'success': False,
                    'message': 'No cameras found in cameras.json or cameras_db'
                }), 404
            
            limit_state = _camera_limit_state()
            allowed_ids = set(limit_state["allowed_ids"])
            results = []
            logger.info(f"🚀 Starting streams for {len(allowed_ids)} cameras (limit {limit_state['limit']})")
            
            for camera in cameras:
                camera_id = camera.get('id')
                camera_name = camera.get('name')
                rtsp_url = camera.get('rtsp_url')
                
                if not camera.get('enabled', True):
                    continue
                if camera_id not in allowed_ids:
                    continue
                
                # Check if stream is already active
                if camera_id in getattr(stream_server, 'active_streams', {}):
                    results.append({
                        'camera_id': camera_id,
                        'camera_name': camera_name,
                        'status': 'already_active',
                        'motion_detection': True
                    })
                    continue
                
                # Force start the stream
                try:
                    success = _run_coro_safe(
                        stream_server.start_stream(camera_id, {
                            'rtsp_url': rtsp_url,
                            'webrtc_enabled': camera.get('webrtc_enabled', True),
                            'motion_detection': True
                        })
                    )
                    
                    results.append({
                        'camera_id': camera_id,
                        'camera_name': camera_name,
                        'status': 'started' if success else 'failed',
                        'motion_detection': success
                    })
                            
                except Exception as e:
                    results.append({
                        'camera_id': camera_id,
                        'camera_name': camera_name,
                        'status': 'error',
                        'error': str(e),
                        'motion_detection': False
                    })
            
            return jsonify({
                'success': True,
                'message': f'Stream startup attempted for {len(results)} cameras',
                'data': {
                    'results': results,
                    'active_streams': len([r for r in results if r['status'] in ['started', 'already_active']])
                }
            })
        else:
            return jsonify({
                'success': False,
                'message': 'Stream server not available'
            }), 503
            
    except Exception as e:
        logger.error(f"Error starting camera streams: {e}")
        return jsonify({
            'success': False,
            'message': f'Error starting streams: {str(e)}'
        }), 500

# Simple camera status check
@app.route('/api/cameras/status', methods=['GET'])
def camera_status():
    """Get simple status of all cameras"""
    try:
        # Load from cameras.json
        try:
            with open('cameras.json', 'r') as f:
                cameras = json.load(f)
        except Exception as e:
            cameras = cameras_db if cameras_db else []
        
        # Check stream server status
        stream_server_available = 'STREAM_SERVER_GLOBAL' in globals() and STREAM_SERVER_GLOBAL is not None
        active_streams = []
        
        if stream_server_available:
            stream_server = STREAM_SERVER_GLOBAL
            active_streams = list(getattr(stream_server, 'active_streams', {}).keys())
        
        camera_status = []
        for camera in cameras:
            camera_status.append({
                'id': camera.get('id'),
                'name': camera.get('name'),
                'enabled': camera.get('enabled', True),
                'motion_detection': camera.get('motion_detection', False),
                'stream_active': camera.get('id') in active_streams,
                'rtsp_url': camera.get('rtsp_url', '')[:50] + '...' if camera.get('rtsp_url', '') else 'None'
            })
        
        return jsonify({
            'success': True,
            'data': {
                'total_cameras': len(cameras),
                'stream_server_available': stream_server_available,
                'active_streams_count': len(active_streams),
                'cameras': camera_status
            }
        })
        
    except Exception as e:
        logger.error(f"Camera status error: {e}")
        return jsonify({
            'success': False,
            'message': f'Status error: {str(e)}'
        }), 500

# Test camera connectivity
@app.route('/api/cameras/test-connectivity', methods=['POST'])
def test_camera_connectivity():
    """Test direct connectivity to all cameras"""
    try:
        import socket
        from urllib.parse import urlparse
        
        results = []
        
        # Load cameras from cameras.json
        with open('cameras.json', 'r') as f:
            cameras = json.load(f)
        
        for camera in cameras:
            camera_id = camera.get('id')
            camera_name = camera.get('name')
            ip_address = camera.get('ip_address')
            port = camera.get('port', 554)
            rtsp_url = camera.get('rtsp_url')
            
            result = {
                'camera_id': camera_id,
                'camera_name': camera_name,
                'ip_address': ip_address,
                'port': port,
                'rtsp_url': rtsp_url,
                'network_reachable': False,
                'rtsp_port_open': False
            }
            
            # Test network connectivity
            try:
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(5)
                result_code = sock.connect_ex((ip_address, port))
                sock.close()
                
                result['network_reachable'] = (result_code == 0)
                result['rtsp_port_open'] = (result_code == 0)
                
            except Exception as e:
                result['network_error'] = str(e)
            
            results.append(result)
            logger.info(f"🔍 Camera {camera_name}: Network={result['network_reachable']}, RTSP Port={result['rtsp_port_open']}")
        
        return jsonify({
            'success': True,
            'message': 'Camera connectivity test completed',
            'data': {
                'results': results,
                'summary': {
                    'total_cameras': len(results),
                    'reachable_cameras': len([r for r in results if r['network_reachable']]),
                    'unreachable_cameras': len([r for r in results if not r['network_reachable']])
                }
            }
        })
        
    except Exception as e:
        logger.error(f"Camera connectivity test error: {e}")
        return jsonify({
            'success': False,
            'message': f'Connectivity test error: {str(e)}'
        }), 500

# Force start a camera in the stream server
@app.route('/api/cameras/<camera_id>/force-start', methods=['POST'])
def force_start_camera(camera_id):
    """Force start a specific camera in the stream server"""
    try:
        if 'STREAM_SERVER_GLOBAL' in globals() and STREAM_SERVER_GLOBAL is not None:
            stream_server = STREAM_SERVER_GLOBAL
            
            # Check if already active
            if camera_id in getattr(stream_server, 'active_streams', {}):
                return jsonify({
                    'success': True,
                    'message': 'Camera already active in stream server'
                })
            
            # Get camera config from cameras.json (fallback to in-memory db)
            try:
                with open('cameras.json', 'r') as f:
                    cameras = json.load(f)
            except Exception:
                cameras = cameras_db if cameras_db else []
            
            camera_config = next((cam for cam in cameras if cam.get('id') == camera_id), None)
            
            if not camera_config:
                return jsonify({
                    'success': False,
                    'message': 'Camera not found in configuration'
                }), 404

            # Ensure MediaMTX path exists (needed for /proxy/webrtc/* and /proxy/hls/* to stop 404'ing).
            # In SIMPLE_SERVER mode, CameraManager can be disabled, so we configure via StreamServer.mediamtx_client.
            try:
                rtsp_url = camera_config.get('rtsp_url')
                mediamtx_path = camera_config.get('mediamtx_path') or camera_config.get('mediamtxPath') or camera_id
                if rtsp_url and mediamtx_path:
                    # Use a fresh MediaMTX client with explicit credentials to avoid any runtime misconfiguration.
                    from urllib.parse import urlparse
                    from core.mediamtx_client import MediaMTXWebRTCClient
                    parsed = urlparse(MEDIAMTX_API_URL)
                    host = parsed.hostname or '127.0.0.1'
                    port = int(parsed.port or 9997)
                    mt = MediaMTXWebRTCClient(
                        mediamtx_host=host,
                        mediamtx_api_port=port,
                        api_username=MEDIAMTX_API_USERNAME,
                        api_password=MEDIAMTX_API_PASSWORD
                    )
                    _run_coro_safe(mt.configure_stream_source(str(mediamtx_path), str(rtsp_url), force_recreate=False))
            except Exception as e:
                logger.warning(f"⚠️ Failed to configure MediaMTX path for {camera_id}: {e}")
            
            # Start stream
            success = _run_coro_safe(
                stream_server.start_stream(camera_id, {
                    'rtsp_url': camera_config.get('rtsp_url'),
                    'webrtc_enabled': camera_config.get('webrtc_enabled', True),
                    'mediamtx_path': camera_config.get('mediamtx_path') or camera_config.get('mediamtxPath') or camera_id,
                    'fps': 15,
                    # Keep force-start lightweight: this endpoint is for stream availability (MediaMTX WHEP/HLS),
                    # not for enabling motion detection pipelines.
                    'motion_detection': False
                })
            )
            
            if success:
                logger.info(f"✅ Force-started camera {camera_id} in stream server")
                return jsonify({
                    'success': True,
                    'message': 'Camera force-started in stream server'
                })
            else:
                logger.error(f"❌ Failed to force-start camera {camera_id} in stream server")
                return jsonify({
                    'success': False,
                    'message': 'Failed to force-start camera in stream server'
                }), 500
        else:
            return jsonify({
                'success': False,
                'message': 'Stream server not available'
            }), 503
            
    except Exception as e:
        logger.error(f"Error force-starting camera {camera_id}: {e}")
        return jsonify({
            'success': False,
            'message': f'Error force-starting camera: {str(e)}'
        }), 500

# Clean up duplicate cameras
@app.route('/api/cameras/cleanup-duplicates', methods=['POST'])
def cleanup_duplicate_cameras():
    """Clean up duplicate cameras created by auto-discovery"""
    try:
        global cameras_db
        
        # Find duplicates by checking for cameras with same mediamtx_path or similar IDs
        duplicates_to_remove = []
        seen_paths = set()
        seen_ids = set()
        
        for i, camera in enumerate(cameras_db):
            camera_id = camera.get('id', '')
            mediamtx_path = camera.get('mediamtx_path', '')
            
            # Check for duplicate mediamtx_path
            if mediamtx_path and mediamtx_path in seen_paths:
                duplicates_to_remove.append(i)
                logger.info(f"Found duplicate mediamtx_path: {mediamtx_path}")
                continue
            
            # Check for duplicate camera IDs
            if camera_id and camera_id in seen_ids:
                duplicates_to_remove.append(i)
                logger.info(f"Found duplicate camera ID: {camera_id}")
                continue
            
            # Check for UUID-style duplicates that might be MediaMTX artifacts
            if len(camera_id) == 36 and camera_id.count('-') == 4:
                # Check if this UUID matches any existing camera's mediamtx_path
                for existing_camera in cameras_db:
                    if existing_camera.get('mediamtx_path') == camera_id:
                        duplicates_to_remove.append(i)
                        logger.info(f"Found UUID duplicate: {camera_id}")
                        break
            
            seen_paths.add(mediamtx_path)
            seen_ids.add(camera_id)
        
        # Remove duplicates in reverse order to maintain indices
        removed_count = 0
        for index in sorted(duplicates_to_remove, reverse=True):
            removed_camera = cameras_db.pop(index)
            removed_count += 1
            logger.info(f"Removed duplicate camera: {removed_camera.get('name', 'Unknown')} (ID: {removed_camera.get('id', 'Unknown')})")
        
        # Save the cleaned up camera list
        if removed_count > 0:
            save_cameras()
            logger.info(f"✅ Cleaned up {removed_count} duplicate cameras")
        
        return jsonify({
            'success': True,
            'message': f'Cleaned up {removed_count} duplicate cameras',
            'removed_count': removed_count,
            'remaining_count': len(cameras_db)
        })
        
    except Exception as e:
        logger.error(f"Error cleaning up duplicate cameras: {e}")
        return jsonify({
            'success': False,
            'message': f'Error cleaning up duplicates: {str(e)}'
        }), 500

if __name__ == '__main__':
    print("Starting MediaMTX Camera Manager")
    print("=" * 50)
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    # Do not log developer personal identifiers in production builds.
    print("=" * 50)
    
    # CRITICAL: Check CUDA availability for GPU acceleration
    try:
        import torch
        print("\nGPU Acceleration Check:")
        print("-" * 50)
        if torch.cuda.is_available():
            print(f"[OK] CUDA Available: YES")
            print(f"[OK] GPU: {torch.cuda.get_device_name(0)}")
            print(f"[OK] Memory: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.1f} GB")
            print(f"[OK] PyTorch CUDA: {torch.version.cuda}")
            print(f"[OK] Expected Performance: 30-60 FPS")
        else:
            print(f"[X] CUDA Available: NO")
            print(f"[X] Expected Performance: 3-5 FPS (CPU only)")
            print(f"\n[WARNING] TO ENABLE GPU ACCELERATION:")
            print(f"   pip uninstall torch torchvision")
            print(f"   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118")
        print("-" * 50)
    except Exception as e:
        print(f"[X] Could not check CUDA: {e}")
    
    print("\n" + "=" * 50)
    print("Web Interface: http://localhost:5000")
    print("API Endpoints:")
    print("   GET  /api/cameras - List all cameras")
    print("   POST /api/cameras - Add new camera")
    print("   GET  /api/cameras/{id} - Get camera details")
    print("   PUT  /api/cameras/{id} - Update camera")
    print("   DELETE /api/cameras/{id} - Delete camera")
    print("   GET  /api/cameras/{id}/stream - Get stream URLs")
    print("   POST /api/cameras/{id}/test - Test connection")
    print("   POST /api/ai/chat - AI chat endpoint")
    print("   GET  /api/ai/status - AI agent status")
    print("   POST /api/ai/vision - AI vision endpoint")
    print("=" * 50)

    # Initialize AI Agent
    initialize_ai_agent()
    
    # Initialize LLM Service
    initialize_llm_service()

    # Network Diagnostics
    logger.info("=" * 40)
    logger.info("🔍 NETWORK DIAGNOSTICS")
    try:
        # Check MediaMTX Host Resolution
        parsed = urlparse(MEDIAMTX_API_URL)
        host = parsed.hostname or "localhost"
        ip = socket.gethostbyname(host)
        logger.info(f"  ✅ DNS Resolve: {host} -> {ip}")
    except Exception as e:
        logger.error(f"  ❌ DNS Resolve Failed for MediaMTX: {e}")

    try:
        # Check LLM Host Resolution
        host = os.environ.get("LLM_SERVICE_HOST", "127.0.0.1")
        ip = socket.gethostbyname(host)
        logger.info(f"  ✅ DNS Resolve: {host} -> {ip}")
    except Exception as e:
        logger.error(f"  ❌ DNS Resolve Failed for LLM Host: {e}")
    logger.info("=" * 40)

    # Skip auto-discovery to prevent duplicate cameras
    logger.info("Auto-discovery disabled to prevent duplicate cameras")
    logger.info("Using existing camera database")
    
    # Configure MediaMTX paths for cameras that need them (On-demand optimization)
    # We skip the heavy startup loop to prevent flooding MediaMTX and delaying startup
    # Paths will be created on-demand when accessed via /api/cameras/<id>/stream
    logger.info("Skipping initial MediaMTX path configuration (will configure on-demand)")
    # try:
    #     configure_mediamtx_paths_for_cameras()
    # except Exception as e:
    #     logger.error(f"Error configuring MediaMTX paths: {e}")

    # Test MediaMTX connection on startup
    test_mediamtx_connection()

    # Restore recording for cameras that were recording before restart
    try:
        import requests as _req
        mtx_base = MEDIAMTX_API_URL.rstrip("/")
        restored = 0
        for cam in cameras_db:
            if cam.get('recording') and cam.get('rtsp_url'):
                cam_id = cam.get('id', '')
                if not cam_id:
                    continue
                payload = _build_recording_payload(cam, True)
                try:
                    resp = _req.patch(f"{mtx_base}/config/paths/patch/{cam_id}", json=payload, timeout=5)
                    if resp.status_code == 404:
                        _req.post(f"{mtx_base}/config/paths/add/{cam_id}", json=payload, timeout=5)
                    restored += 1
                except Exception as re_err:
                    logger.warning("Could not restore recording for %s: %s", cam_id, re_err)
        if restored:
            logger.info("Restored recording for %d camera(s) from persisted state", restored)
    except Exception as e:
        logger.warning("Recording restoration on startup failed: %s", e)

    # Register comprehensive API routes if available
    if API_ROUTES_AVAILABLE:
        try:
            # Prefer real services where available (DB and Scheduler)
            from core.database import DatabaseManager
            from core.scheduler import Scheduler
            from core.stream_server import StreamServer
            
            # Optional real services
            try:
                from core.ai_analyzer import AIAnalyzer
            except Exception:
                AIAnalyzer = None  # type: ignore
            try:
                from core.alert_system import AlertSystem
            except Exception:
                AlertSystem = None  # type: ignore
            try:
                from core.camera_manager import CameraManager
            except Exception:
                CameraManager = None  # type: ignore
            
            # Initialize real database manager for rules CRUD
            db_manager = DatabaseManager(db_path='data/sentry.db')
            try:
                db_manager.initialize()
            except Exception as e:
                logger.error(f"❌ Failed to initialize DatabaseManager: {e}")
                # Fallback to a tiny shim that at least responds to status
                class ShimDB:
                    def get_status(self):
                        return {'status': 'error'}
                db_manager = ShimDB()
            
            # Use real scheduler (lightweight, in-memory)
            scheduler = Scheduler()

            # Server-side automation engine (rules -> actions)
            automation_engine = None
            try:
                from core.automation import AutomationEngine
                dry = str(os.environ.get("AUTOMATION_DRY_RUN", "false")).strip().lower() in ("1", "true", "yes", "on")
                automation_engine = AutomationEngine(db_manager=db_manager, stream_server=None, socketio=socketio, dry_run=dry)
                automation_engine.start()
                globals()["AUTOMATION_ENGINE_GLOBAL"] = automation_engine
                logger.info("✅ Automation engine initialized")
            except Exception as e:
                automation_engine = None
                logger.warning(f"⚠️ Automation engine unavailable: {e}")

            # Audio monitoring (server-side) for event detection + similarity matching
            try:
                from core.audio_monitor import AudioMonitorManager
                audio_monitor = AudioMonitorManager(
                    mediamtx_webrtc_url=MEDIAMTX_WEBRTC_URL,
                    db_manager=db_manager,
                    socketio=socketio,
                    python_script_manager=python_script_manager,
                    automation_engine=automation_engine,
                )
                globals()['AUDIO_MONITOR_GLOBAL'] = audio_monitor
                logger.info("✅ Audio monitor manager initialized")
            except Exception as e:
                audio_monitor = None
                logger.warning(f"⚠️ Audio monitor manager unavailable: {e}")
            
            # Initialize services (real implementations) - disable camera manager to use cameras.json only
            camera_manager = None  # Disabled to use cameras.json as single source of truth
            
            # Parse MediaMTX host from environment variables (default to configured values)
            try:
                parsed = urlparse(MEDIAMTX_API_URL)
                mediamtx_host = parsed.hostname or "localhost"
                mediamtx_api_port = parsed.port or 9997
            except Exception:
                mediamtx_host = "localhost"
                mediamtx_api_port = 9997
            
            mediamtx_webrtc_port = 8889
            if MEDIAMTX_WEBRTC_URL:
                try:
                    parsed = urlparse(MEDIAMTX_WEBRTC_URL)
                    if parsed.port:
                        mediamtx_webrtc_port = parsed.port
                except Exception:
                    pass

            stream_server = StreamServer(
                mediamtx_host=mediamtx_host,
                mediamtx_api_port=mediamtx_api_port,
                mediamtx_webrtc_port=mediamtx_webrtc_port,
                ai_agent=ai_agent
            )
            # expose globally for MJPEG streaming endpoint
            globals()['STREAM_SERVER_GLOBAL'] = stream_server

            # Connect stream server to automation engine + register actions
            try:
                if automation_engine is not None:
                    automation_engine.stream_server = stream_server
                    from core.automation.actions.email import EmailAction
                    email_action = EmailAction(db_manager=db_manager, stream_server=stream_server)
                    automation_engine.action_handlers["email"] = email_action.handler()
            except Exception as e:
                logger.warning(f"⚠️ Failed to register automation actions: {e}")
            
            # Link depth processor to stream server for direct frame access (GPU optimization)
            try:
                from core.depth_processor import get_depth_processor
                depth_processor = get_depth_processor(stream_server=stream_server)
                logger.info("✓ Depth processor linked to stream server for direct frame access")
            except Exception as e:
                logger.warning(f"Could not link depth processor to stream server: {e}")
            
            # Initialize auto-recovery system
            auto_recovery = None
            if AUTO_RECOVERY_AVAILABLE and camera_manager and stream_server:
                try:
                    auto_recovery = CameraAutoRecovery(
                        camera_manager=camera_manager,
                        stream_server=stream_server,
                        mediamtx_client=getattr(camera_manager, 'mediamtx_client', None)
                    )
                    logger.info("🔄 Camera auto-recovery system initialized")
                except Exception as e:
                    logger.warning(f"⚠️ Failed to initialize camera auto-recovery: {e}")
            
            ai_analyzer = (AIAnalyzer() if AIAnalyzer else None)
            # Make ai_analyzer globally accessible
            globals()['ai_analyzer'] = ai_analyzer
            alert_system = (AlertSystem() if AlertSystem else None)
            
            # Set stream server reference in AI agent for detection queries
            if ai_agent and hasattr(ai_agent, 'set_stream_server'):
                ai_agent.set_stream_server(stream_server)
            
            # Always register robust WebSocket event handlers if Socket.IO is available
            if socketio is not None:
                class RealtimeNamespace(Namespace):
                    def on_connect(self):
                        try:
                            emit('connected', { 'message': 'socket connected', 'namespace': WS_NS })
                            logger.info("✅ WS /realtime connected")
                        except Exception as e:
                            logger.error(f"❌ WS /realtime connect error: {e}")

                    def on_disconnect(self):
                        logger.info("🔌 WS /realtime disconnected")

                    def on_subscribe(self, data):
                        try:
                            cam = (data or {}).get('camera_id')
                            if cam:
                                try:
                                    cam_obj = resolve_camera_ref(cam)
                                    canonical_id = cam_obj.get('id') if cam_obj else cam
                                except Exception:
                                    canonical_id = cam
                                join_room(f"camera:{canonical_id}")
                                emit('subscribed', { 'camera_id': canonical_id })
                                logger.info(f"WS /realtime subscribed -> camera:{canonical_id}")
                            else:
                                emit('error', { 'message': 'No camera_id provided' })
                        except Exception as e:
                            logger.error(f"❌ WS /realtime subscribe error: {e}")
                            emit('error', { 'message': str(e) })

                    def on_unsubscribe(self, data):
                        try:
                            cam = (data or {}).get('camera_id')
                            if cam:
                                leave_room(f"camera:{cam}")
                                emit('unsubscribed', { 'camera_id': cam })
                                logger.info(f"✅ WS /realtime unsubscribed <- camera:{cam}")
                            else:
                                emit('error', { 'message': 'No camera_id provided' })
                        except Exception as e:
                            logger.error(f"❌ WS /realtime unsubscribe error: {e}")
                            emit('error', { 'message': str(e) })

                # Default namespace handlers with improved error handling
                @socketio.on('connect')
                def _on_connect_default(auth=None):
                    try:
                        # Use safe emit with proper context
                        socketio.emit('connected', { 'message': 'socket connected', 'namespace': '/' })
                        logger.info("WS default connected")
                    except Exception as e:
                        logger.error(f"❌ WS default connect error: {e}")

                @socketio.on('disconnect')
                def _on_disconnect_default():
                    logger.info("WS default disconnected")

                @socketio.on('subscribe')
                def _on_subscribe_default(data):
                    try:
                        cam = (data or {}).get('camera_id')
                        if cam:
                            try:
                                cam_obj = resolve_camera_ref(cam)
                                canonical_id = cam_obj.get('id') if cam_obj else cam
                            except Exception:
                                canonical_id = cam
                            join_room(f"camera:{canonical_id}")
                            # Use safe emit with proper context
                            socketio.emit('subscribed', { 'camera_id': canonical_id })
                            logger.info(f"WS default subscribed -> camera:{canonical_id}")
                        else:
                            socketio.emit('error', { 'message': 'No camera_id provided' })
                    except Exception as e:
                        logger.error(f"❌ WS default subscribe error: {e}")
                        try:
                            socketio.emit('error', { 'message': str(e) })
                        except:
                            pass

                @socketio.on('unsubscribe')
                def _on_unsubscribe_default(data):
                    try:
                        cam = (data or {}).get('camera_id')
                        if cam:
                            leave_room(f"camera:{cam}")
                            # Use safe emit with proper context
                            socketio.emit('unsubscribed', { 'camera_id': cam })
                            logger.info(f"WS default unsubscribed <- camera:{cam}")
                        else:
                            socketio.emit('error', { 'message': 'No camera_id provided' })
                    except Exception as e:
                        logger.error(f"❌ WS default unsubscribe error: {e}")
                        try:
                            socketio.emit('error', { 'message': str(e) })
                        except:
                            pass

                # Register namespace
                try:
                    socketio.on_namespace(RealtimeNamespace(WS_NS))
                    logger.info("✅ WebSocket event handlers registered")
                except Exception as e:
                    logger.error(f"❌ Failed to register WS namespace: {e}")

                # ==============================
                # Sessions Namespace (/sessions)
                # ==============================
                try:
                    SESSIONS_NS = "/sessions"

                    class SessionsNamespace(Namespace):
                        def on_connect(self):
                            try:
                                emit("connected", {"message": "sessions socket connected", "namespace": SESSIONS_NS})
                                logger.info("✅ WS /sessions connected")
                            except Exception as e:
                                logger.error(f"❌ WS /sessions connect error: {e}")

                        def on_disconnect(self):
                            logger.info("🔌 WS /sessions disconnected")

                        def on_join_session(self, data):
                            try:
                                session_id = str((data or {}).get("session_id") or "").strip()
                                if not session_id:
                                    emit("error", {"message": "session_id is required"})
                                    return
                                join_room(f"session:{session_id}")
                                emit("joined_session", {"session_id": session_id})
                            except Exception as e:
                                logger.error(f"❌ WS /sessions join_session error: {e}")
                                emit("error", {"message": str(e)})

                        def on_leave_session(self, data):
                            try:
                                session_id = str((data or {}).get("session_id") or "").strip()
                                if not session_id:
                                    emit("error", {"message": "session_id is required"})
                                    return
                                leave_room(f"session:{session_id}")
                                emit("left_session", {"session_id": session_id})
                            except Exception as e:
                                logger.error(f"❌ WS /sessions leave_session error: {e}")
                                emit("error", {"message": str(e)})

                    socketio.on_namespace(SessionsNamespace(SESSIONS_NS))
                    logger.info("✅ Sessions namespace registered at /sessions")
                except Exception as e:
                    logger.warning(f"Sessions namespace not registered: {e}")

                # ==============================
                # SSH Namespace (/ssh)
                # ==============================
                if PARAMIKO_AVAILABLE:
                    SSH_NS = '/ssh'
                    _SSH_SESSIONS: Dict[str, Dict[str, Any]] = {}

                    def _cleanup_ssh_session(sid: str) -> None:
                        try:
                            sess = _SSH_SESSIONS.pop(sid, None)
                            if not sess:
                                return
                            try:
                                chan = sess.get('channel')
                                if chan:
                                    try:
                                        chan.close()
                                    except Exception:
                                        pass
                            finally:
                                cli = sess.get('client')
                                if cli:
                                    try:
                                        cli.close()
                                    except Exception:
                                        pass
                        except Exception as _e:
                            logger.debug(f"SSH cleanup error for {sid}: {_e}")

                    class SSHNamespace(Namespace):
                        def on_connect(self):
                            try:
                                emit('connected', { 'message': 'ssh socket connected', 'namespace': SSH_NS })
                                logger.info("✅ WS /ssh connected")
                            except Exception as e:
                                logger.error(f"❌ WS /ssh connect error: {e}")

                        def on_disconnect(self):
                            try:
                                sid = request.sid
                                _cleanup_ssh_session(sid)
                                logger.info("🔌 WS /ssh disconnected")
                            except Exception as e:
                                logger.debug(f"WS /ssh disconnect cleanup error: {e}")

                        def on_ssh_connect(self, data):
                            sid = request.sid
                            if not PARAMIKO_AVAILABLE:
                                socketio.emit('ssh_error', { 'message': 'SSH not available on server' }, to=sid, namespace=SSH_NS)
                                return
                            try:
                                host = (data or {}).get('host')
                                port = int((data or {}).get('port') or 22)
                                username = (data or {}).get('username')
                                password = (data or {}).get('password')
                                timeout = int((data or {}).get('timeout') or 10)
                                if not host or not username:
                                    socketio.emit('ssh_error', { 'message': 'host and username are required' }, to=sid, namespace=SSH_NS)
                                    return

                                # Close any existing session first
                                _cleanup_ssh_session(sid)

                                client = paramiko.SSHClient()
                                client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
                                client.connect(hostname=host, port=port, username=username, password=password, timeout=timeout, banner_timeout=timeout, auth_timeout=timeout)
                                transport = client.get_transport()
                                if transport is None:
                                    raise RuntimeError('No SSH transport')
                                chan = transport.open_session()
                                chan.get_pty(term='xterm', width=120, height=32)
                                chan.invoke_shell()

                                _SSH_SESSIONS[sid] = {
                                    'client': client,
                                    'channel': chan,
                                    'running': True
                                }

                                # Reader loop in background thread
                                def _reader_loop(_sid: str, _chan: Any):
                                    try:
                                        while True:
                                            if _chan.recv_ready():
                                                try:
                                                    data = _chan.recv(4096)
                                                except Exception:
                                                    break
                                                if not data:
                                                    break
                                                try:
                                                    text = data.decode('utf-8', errors='ignore')
                                                except Exception:
                                                    text = ''
                                                try:
                                                    socketio.emit('ssh_output', { 'data': text }, to=_sid, namespace=SSH_NS)
                                                except Exception as _emit_e:
                                                    logger.debug(f"/ssh emit error: {_emit_e}")
                                            else:
                                                if _chan.closed or _chan.exit_status_ready():
                                                    break
                                                time.sleep(0.02)
                                    except Exception as _e:
                                        logger.debug(f"SSH reader loop error: {_e}")
                                    finally:
                                        socketio.emit('ssh_closed', { 'message': 'SSH session closed' }, to=_sid, namespace=SSH_NS)
                                        _cleanup_ssh_session(_sid)

                                t = threading.Thread(target=_reader_loop, args=(sid, chan), daemon=True)
                                t.start()

                                socketio.emit('ssh_connected', { 'message': f'Connected to {host}:{port} as {username}' }, to=sid, namespace=SSH_NS)
                            except Exception as e:
                                logger.error(f"SSH connect error: {e}")
                                try:
                                    _cleanup_ssh_session(sid)
                                finally:
                                    socketio.emit('ssh_error', { 'message': str(e) }, to=sid, namespace=SSH_NS)

                        def on_ssh_input(self, data):
                            sid = request.sid
                            try:
                                sess = _SSH_SESSIONS.get(sid)
                                if not sess:
                                    socketio.emit('ssh_error', { 'message': 'SSH session not established' }, to=sid, namespace=SSH_NS)
                                    return
                                chan = sess.get('channel')
                                if not chan:
                                    socketio.emit('ssh_error', { 'message': 'SSH channel missing' }, to=sid, namespace=SSH_NS)
                                    return
                                payload = (data or {}).get('data') or ''
                                if isinstance(payload, str):
                                    chan.send(payload)
                                else:
                                    socketio.emit('ssh_error', { 'message': 'invalid input' }, to=sid, namespace=SSH_NS)
                            except Exception as e:
                                logger.debug(f"SSH input error: {e}")
                                socketio.emit('ssh_error', { 'message': str(e) }, to=sid, namespace=SSH_NS)

                        def on_ssh_disconnect(self, data=None):
                            sid = request.sid
                            try:
                                _cleanup_ssh_session(sid)
                                socketio.emit('ssh_closed', { 'message': 'SSH session closed' }, to=sid, namespace=SSH_NS)
                            except Exception as e:
                                logger.debug(f"SSH disconnect error: {e}")

                    try:
                        socketio.on_namespace(SSHNamespace('/ssh'))
                        logger.info("✅ SSH namespace registered at /ssh")
                    except Exception as e:
                        logger.error(f"❌ Failed to register SSH namespace: {e}")

            # Start optimized stream server (non-blocking) if available
            optimized_stream_server = None
            if _OPTIMIZED_SERVER_AVAILABLE:
                try:
                    optimized_stream_server = OptimizedStreamServer(
                        mediamtx_host=mediamtx_host,
                        mediamtx_api_port=mediamtx_api_port,
                        mediamtx_webrtc_port=mediamtx_webrtc_port
                    )
                    # Start in background task/thread to avoid blocking Flask
                    def _start_opt_server():
                        try:
                            # Use the safe runner to start it on the background loop
                            _run_coro_safe(optimized_stream_server.start())
                        except Exception as _err:
                            logger.error(f"❌ Failed to start OptimizedStreamServer: {_err}")
                    
                    import threading as _threading
                    _t = _threading.Thread(target=_start_opt_server, daemon=True)
                    _t.start()
                    logger.info("✅ OptimizedStreamServer started in background")
                except Exception as _err:
                    logger.error(f"❌ OptimizedStreamServer init error: {_err}")

            # Wire realtime callbacks to Socket.IO rooms per camera (if both available)
            # Optimized for lowest latency real-time motion detection with improved error handling
            if socketio is not None and stream_server is not None:
                def _emit_motion(camera_id: str, payload: Dict[str, Any]) -> None:
                    """Emit motion detection data with ultra-low latency"""
                    try:
                        if python_script_manager:
                            try:
                                python_script_manager.handle_event('motion', camera_id, payload)
                            except Exception as script_exc:
                                logger.debug(f"Python script motion handler error for {camera_id}: {script_exc}")
                        if automation_engine:
                            try:
                                automation_engine.submit("motion", camera_id, payload)
                            except Exception:
                                pass
                        # Use safe emission with proper error handling
                        def emit_data():
                            try:
                                # Ultra-fast broadcast to all clients
                                socketio.emit('motion_update', payload)
                                # Also emit to specific room for compatibility
                                socketio.emit('motion_update', payload, room=f"camera:{camera_id}")
                            except Exception as emit_error:
                                logger.debug(f"Motion emit context error for {camera_id}: {emit_error}")
                        
                        # Execute emission in background task immediately
                        socketio.start_background_task(emit_data)
                    except Exception as e:
                        logger.debug(f"Motion emit error for {camera_id}: {e}")

                def _emit_tracks(camera_id: str, payload: Dict[str, Any]) -> None:
                    """Emit tracking data with ultra-low latency"""
                    try:
                        if automation_engine:
                            try:
                                automation_engine.submit("tracks", camera_id, payload)
                            except Exception:
                                pass
                        # Use safe emission with proper error handling
                        def emit_data():
                            try:
                                # Ultra-fast broadcast to all clients
                                socketio.emit('tracks_update', payload)
                                # Also emit to specific room for compatibility
                                socketio.emit('tracks_update', payload, room=f"camera:{camera_id}")
                            except Exception as emit_error:
                                logger.debug(f"Tracks emit context error for {camera_id}: {emit_error}")
                        
                        # Execute emission in background task immediately
                        socketio.start_background_task(emit_data)
                    except Exception as e:
                        logger.debug(f"Tracks emit error for {camera_id}: {e}")

                def _emit_detections(camera_id: str, payload: Dict[str, Any]) -> None:
                    """Emit object detection data with ultra-low latency"""
                    try:
                        if python_script_manager:
                            try:
                                python_script_manager.handle_event('detection', camera_id, payload)
                            except Exception as script_exc:
                                logger.debug(f"Python script detection handler error for {camera_id}: {script_exc}")
                        if automation_engine:
                            try:
                                automation_engine.submit("detections", camera_id, payload)
                            except Exception:
                                pass
                        # Use safe emission with proper error handling
                        def emit_data():
                            try:
                                # Ultra-fast broadcast to all clients
                                socketio.emit('detections_update', payload)
                                # Also emit to specific room for compatibility
                                socketio.emit('detections_update', payload, room=f"camera:{camera_id}")
                            except Exception as emit_error:
                                logger.debug(f"Detections emit context error for {camera_id}: {emit_error}")
                        
                        # Execute emission in background task immediately
                        socketio.start_background_task(emit_data)
                    except Exception as e:
                        logger.debug(f"Detections emit error for {camera_id}: {e}")

                stream_server.on_motion_update = _emit_motion
                stream_server.on_tracks_update = _emit_tracks
                stream_server.on_detection_update = _emit_detections

                # Attach AI verifier (Tier-2) if available
                if ai_analyzer is not None:
                    try:
                        stream_server.ai_verifier = ai_analyzer
                    except Exception:
                        pass

                # Attach AI agent (Tier-3) if available
                if ai_agent is not None:
                    try:
                        stream_server.set_ai_agent(ai_agent)
                    except Exception:
                        pass

                logger.info("✅ Stream server callbacks wired to WebSocket")

            # Wire optimized stream server callbacks to WebSocket (FIX FOR MOTION DETECTION)
            if socketio is not None and optimized_stream_server is not None:
                def _emit_motion_optimized(camera_id: str, payload: Dict[str, Any]) -> None:
                    """Emit motion detection data from optimized stream server with ultra-low latency"""
                    try:
                        if python_script_manager:
                            try:
                                python_script_manager.handle_event('motion', camera_id, payload)
                            except Exception as script_exc:
                                logger.debug(f"Python script motion handler error (optimized) for {camera_id}: {script_exc}")
                        if automation_engine:
                            try:
                                automation_engine.submit("motion", camera_id, payload)
                            except Exception:
                                pass
                        # Use safe emission with proper error handling
                        def emit_data():
                            try:
                                # Ultra-fast broadcast to all clients
                                socketio.emit('motion_update', payload)
                                # Also emit to specific room for compatibility
                                socketio.emit('motion_update', payload, room=f"camera:{camera_id}")
                                logger.debug(f"Optimized stream motion emitted for camera {camera_id}")
                            except Exception as emit_error:
                                logger.debug(f"Optimized motion emit context error for {camera_id}: {emit_error}")
                        
                        # Execute emission in background task immediately
                        socketio.start_background_task(emit_data)
                    except Exception as e:
                        logger.debug(f"Optimized motion emit error for {camera_id}: {e}")

                def _emit_tracks_optimized(camera_id: str, payload: Dict[str, Any]) -> None:
                    """Emit tracking data from optimized stream server with ultra-low latency"""
                    try:
                        if automation_engine:
                            try:
                                automation_engine.submit("tracks", camera_id, payload)
                            except Exception:
                                pass
                        # Use safe emission with proper error handling
                        def emit_data():
                            try:
                                # Ultra-fast broadcast to all clients
                                socketio.emit('tracks_update', payload)
                                # Also emit to specific room for compatibility
                                socketio.emit('tracks_update', payload, room=f"camera:{camera_id}")
                            except Exception as emit_error:
                                logger.debug(f"Optimized tracks emit context error for {camera_id}: {emit_error}")
                        
                        # Execute emission in background task immediately
                        socketio.start_background_task(emit_data)
                    except Exception as e:
                        logger.debug(f"Optimized tracks emit error for {camera_id}: {e}")

                def _emit_detections_optimized(camera_id: str, payload: Dict[str, Any]) -> None:
                    """Emit object detection data from optimized stream server with ultra-low latency"""
                    try:
                        if python_script_manager:
                            try:
                                python_script_manager.handle_event('detection', camera_id, payload)
                            except Exception as script_exc:
                                logger.debug(f"Python script detection handler error (optimized) for {camera_id}: {script_exc}")
                        if automation_engine:
                            try:
                                automation_engine.submit("detections", camera_id, payload)
                            except Exception:
                                pass
                        # Use safe emission with proper error handling
                        def emit_data():
                            try:
                                # Ultra-fast broadcast to all clients
                                socketio.emit('detections_update', payload)
                                # Also emit to specific room for compatibility
                                socketio.emit('detections_update', payload, room=f"camera:{camera_id}")
                            except Exception as emit_error:
                                logger.debug(f"Optimized detections emit context error for {camera_id}: {emit_error}")
                        
                        # Execute emission in background task immediately
                        socketio.start_background_task(emit_data)
                    except Exception as e:
                        logger.debug(f"Optimized detections emit error for {camera_id}: {e}")

                # Wire optimized stream server callbacks
                optimized_stream_server.on_motion_update = _emit_motion_optimized
                optimized_stream_server.on_tracks_update = _emit_tracks_optimized
                optimized_stream_server.on_detection_update = _emit_detections_optimized

                logger.info("✅ Optimized stream server callbacks wired to WebSocket")

            # Store auto-recovery in app context and globally for API access
            app.auto_recovery = auto_recovery
            globals()['AUTO_RECOVERY_GLOBAL'] = auto_recovery
            
    # Register the API routes with real services
            register_routes(app, camera_manager, stream_server, ai_analyzer, alert_system, ai_agent, db_manager, scheduler, python_script_manager, llm_service, audio_monitor)
            
            # Register Proxy Routes (MediaMTX)
            try:
                app.register_blueprint(proxy_bp)
                logger.info("✅ Proxy routes registered")
            except Exception as e:
                logger.error(f"❌ Failed to register proxy routes: {e}")

            logger.info("✅ API routes registered with real services (StreamServer, DB, Scheduler)")

            # Skip automatic stream bootstrap to prevent startup delays
            # Streams will be started on-demand when motion detection is enabled
            logger.info("Stream server ready - streams will start on-demand for motion detection")
            
            # Bootstrap all cameras on startup
            if AUTO_RECOVERY_AVAILABLE and camera_manager and stream_server:
                bootstrap = CameraBootstrap(camera_manager, stream_server)
                logger.info("🚀 Starting camera bootstrap sequence...")
                
                # Run bootstrap in background thread to not block startup
                import threading
                def bootstrap_thread():
                    try:
                        time.sleep(5)  # Wait for MediaMTX to be fully ready
                        results = bootstrap.bootstrap_cameras_sync(delay_between_cameras=1.5)
                        successful = sum(1 for success in results.values() if success)
                        total = len(results)
                        logger.info(f"🎯 Camera bootstrap completed: {successful}/{total} cameras started")
                    except Exception as e:
                        logger.error(f"❌ Error in camera bootstrap: {e}")
                        
                bootstrap_thread = threading.Thread(target=bootstrap_thread, daemon=True)
                bootstrap_thread.start()
            
            # Start auto-recovery system
            if auto_recovery:
                auto_recovery.start()
                logger.info("🔄 Camera auto-recovery monitoring started")
            
            # Initialize WebSocket connection management
            motion_detection_integration = None
            if WEBSOCKET_MANAGEMENT_AVAILABLE and socketio is not None:
                try:
                    motion_detection_integration = MotionDetectionIntegration(
                        socketio_instance=socketio,
                        camera_manager=camera_manager,
                        stream_server=stream_server,
                        ai_agent=ai_agent
                    )
                    
                    # Start the integration
                    motion_detection_integration.start()
                    logger.info("🔌 WebSocket connection management started")
                    
                    # Register websocket management routes
                    init_websocket_routes(app, motion_detection_integration)
                    logger.info("📡 WebSocket management API routes registered")
                    
                except Exception as e:
                    logger.error(f"❌ Failed to initialize WebSocket connection management: {e}")
            else:
                logger.warning("⚠️ WebSocket connection management not available")
                
        except Exception as e:
            logger.error(f"❌ Failed to register API routes: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
    else:
        logger.warning("⚠️ Using basic API endpoints only")

    # Fallback motion routes (ensure production-ready motion control even if blueprint fails)
    try:
        def _has_rule(path: str) -> bool:
            try:
                for rule in app.url_map.iter_rules():
                    if str(rule.rule) == path:
                        return True
            except Exception:
                pass
            return False

        # Enable motion (fallback)
        if not _has_rule('/api/cameras/<camera_id>/motion/enable'):
            @app.route('/api/cameras/<camera_id>/motion/enable', methods=['POST'])
            def _fallback_enable_motion(camera_id):
                try:
                    if 'STREAM_SERVER_GLOBAL' not in globals() or STREAM_SERVER_GLOBAL is None:
                        return jsonify({"success": False, "message": "Stream server not available"}), 503
                    # Accept either id or name
                    camera = resolve_camera_ref(camera_id)
                    if not camera:
                        return jsonify({"success": False, "message": f"Camera {camera_id} not found"}), 404
                    rtsp_url = camera.get('rtsp_url')
                    if not rtsp_url:
                        return jsonify({"success": False, "message": "RTSP URL missing"}), 400
                    
                    # NOTE: Motion detection does NOT auto-enable object detection
                    # User must explicitly enable via /detection/enable endpoint
                    
                    ok = _run_coro_safe(
                        STREAM_SERVER_GLOBAL.start_stream(camera.get('id'), {
                            'rtsp_url': rtsp_url,
                            'webrtc_enabled': False,
                            'fps': 15
                        })
                    )
                    if ok:
                        return jsonify({
                            "success": True,
                            "message": "Motion detection enabled",
                            "data": {"camera_id": camera.get('id'), "camera_name": camera.get('name'), "motion_detection_active": True, "stream_active": True}
                        })
                    return jsonify({"success": False, "message": "Failed to start stream"}), 500
                except Exception as e:
                    logger.error(f"Fallback enable motion failed: {e}")
                    return jsonify({"success": False, "message": str(e)}), 500

        # Disable motion (fallback)
        if not _has_rule('/api/cameras/<camera_id>/motion/disable'):
            @app.route('/api/cameras/<camera_id>/motion/disable', methods=['POST'])
            def _fallback_disable_motion(camera_id):
                try:
                    # NOTE: Motion disable does NOT auto-disable object detection
                    # User must explicitly disable via /detection/disable endpoint
                    return jsonify({"success": True, "message": "Motion detection disabled"})
                except Exception as e:
                    logger.error(f"Fallback disable motion failed: {e}")
                    return jsonify({"success": False, "message": str(e)}), 500
        
        # Add new dedicated endpoint for object detection control
        @app.route('/api/cameras/<camera_id>/detection/enable', methods=['POST'])
        def enable_object_detection(camera_id):
            """Enable object detection for a specific camera"""
            try:
                if 'STREAM_SERVER_GLOBAL' not in globals() or STREAM_SERVER_GLOBAL is None:
                    return jsonify({"success": False, "message": "Stream server not available"}), 503
                camera = resolve_camera_ref(camera_id)
                if not camera:
                    return jsonify({"success": False, "message": f"Camera {camera_id} not found"}), 404
                
                success = STREAM_SERVER_GLOBAL.enable_detection(camera.get('id'))
                stream_ok = False
                if success:
                    # Ensure the stream is running so detection has frames to process
                    try:
                        stream_ok = hasattr(STREAM_SERVER_GLOBAL, 'active_streams') and camera.get('id') in STREAM_SERVER_GLOBAL.active_streams
                    except Exception:
                        stream_ok = False

                    if not stream_ok:
                        try:
                            rtsp_url = camera.get('rtsp_url')
                            if rtsp_url:
                                # Use safe async wrapper to ensure we use the stream server's loop
                                stream_ok = _run_coro_safe(STREAM_SERVER_GLOBAL.start_stream(camera.get('id'), {
                                    'rtsp_url': rtsp_url,
                                    'webrtc_enabled': False,
                                    'fps': 15
                                }))
                            else:
                                stream_ok = False
                        except Exception as e:
                            logger.error(f"Failed to start stream for detection on {camera_id}: {e}")
                            stream_ok = False

                    if stream_ok:
                        logger.info(f"✅ Object detection enabled and stream active for camera {camera_id}")
                    else:
                        logger.warning(f"⚠️ Object detection enabled but stream not active for camera {camera_id}")

                    return jsonify({
                        "success": True,
                        "message": "Object detection enabled",
                        "data": {"camera_id": camera.get('id'), "detection_enabled": True, "stream_active": bool(stream_ok)}
                    })
                return jsonify({"success": False, "message": "Failed to enable detection"}), 500
            except Exception as e:
                logger.error(f"Failed to enable detection: {e}")
                return jsonify({"success": False, "message": str(e)}), 500
        
        @app.route('/api/cameras/<camera_id>/detection/disable', methods=['POST'])
        def disable_object_detection(camera_id):
            """Disable object detection for a specific camera"""
            try:
                if 'STREAM_SERVER_GLOBAL' not in globals() or STREAM_SERVER_GLOBAL is None:
                    return jsonify({"success": False, "message": "Stream server not available"}), 503
                camera = resolve_camera_ref(camera_id)
                if not camera:
                    return jsonify({"success": False, "message": f"Camera {camera_id} not found"}), 404
                
                success = STREAM_SERVER_GLOBAL.disable_detection(camera.get('id'))
                if success:
                    logger.info(f"🛑 Object detection disabled for camera {camera_id}")
                    return jsonify({
                        "success": True,
                        "message": "Object detection disabled",
                        "data": {"camera_id": camera.get('id'), "detection_enabled": False}
                    })
                return jsonify({"success": False, "message": "Failed to disable detection"}), 500
            except Exception as e:
                logger.error(f"Failed to disable detection: {e}")
                return jsonify({"success": False, "message": str(e)}), 500
        
        @app.route('/api/cameras/<camera_id>/detection/status', methods=['GET'])
        def get_detection_status(camera_id):
            """Get object detection status for a specific camera"""
            try:
                if 'STREAM_SERVER_GLOBAL' not in globals() or STREAM_SERVER_GLOBAL is None:
                    return jsonify({"success": False, "message": "Stream server not available"}), 503
                camera = resolve_camera_ref(camera_id)
                if not camera:
                    return jsonify({"success": False, "message": f"Camera {camera_id} not found"}), 404
                
                enabled = STREAM_SERVER_GLOBAL.is_detection_enabled(camera.get('id'))
                return jsonify({
                    "success": True,
                    "data": {"camera_id": camera.get('id'), "detection_enabled": enabled}
                })
            except Exception as e:
                logger.error(f"Failed to get detection status: {e}")
                return jsonify({"success": False, "message": str(e)}), 500
        
        # Legacy fallback (keeping original behavior for compatibility)
        if not _has_rule('/api/cameras/<camera_id>/motion/disable'):
            @app.route('/api/cameras/<camera_id>/motion/disable', methods=['POST'])
            def _fallback_disable_motion_legacy(camera_id):
                try:
                    if 'STREAM_SERVER_GLOBAL' not in globals() or STREAM_SERVER_GLOBAL is None:
                        return jsonify({"success": False, "message": "Stream server not available"}), 503
                    camera = resolve_camera_ref(camera_id)
                    if not camera:
                        return jsonify({"success": False, "message": f"Camera {camera_id} not found"}), 404
                    
                    ok = _run_coro_safe(STREAM_SERVER_GLOBAL.stop_stream(camera.get('id')))

                    if ok:
                        return jsonify({"success": True, "message": "Motion detection disabled", "data": {"camera_id": camera.get('id')}})
                    return jsonify({"success": False, "message": "Failed to stop stream"}), 500
                except Exception as e:
                    logger.error(f"Fallback disable motion failed: {e}")
                    return jsonify({"success": False, "message": str(e)}), 500

        # Motion status (fallback)
        if not _has_rule('/api/cameras/<camera_id>/motion/status'):
            @app.route('/api/cameras/<camera_id>/motion/status', methods=['GET'])
            def _fallback_motion_status(camera_id):
                try:
                    active = False
                    resolved_id = None
                    cam = resolve_camera_ref(camera_id)
                    if cam:
                        resolved_id = cam.get('id')
                    if 'STREAM_SERVER_GLOBAL' in globals() and STREAM_SERVER_GLOBAL is not None:
                        try:
                            cid = resolved_id or camera_id
                            active = hasattr(STREAM_SERVER_GLOBAL, 'active_streams') and cid in STREAM_SERVER_GLOBAL.active_streams
                        except Exception:
                            active = False
                    return jsonify({
                        "success": True,
                        "data": {"camera_id": (resolved_id or camera_id), "stream_active": bool(active)}
                    })
                except Exception as e:
                    return jsonify({"success": False, "message": str(e)}), 500
    except Exception as _fallback_e:
        logger.warning(f"Fallback motion routes setup skipped: {_fallback_e}")

    # NOTE: On some Windows/Python builds (notably Python 3.13+), Flask-SocketIO/eventlet stacks can
    # accept TCP connections but hang on HTTP responses. For reliability of the REST API endpoints,
    # allow forcing a plain Werkzeug server.
    import sys as _sys
    force_simple = str(os.environ.get("KNOXNET_SIMPLE_SERVER", "")).lower() in {"1", "true", "yes"} or _sys.version_info >= (3, 13)

    if socketio is not None and not force_simple:
        socketio.run(
            app,
            host='0.0.0.0',
            port=5000,
            debug=False,
            allow_unsafe_werkzeug=True,
        )
    else:
        app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)