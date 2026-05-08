import os
import json
import concurrent.futures
import ipaddress
import platform
import socket
import subprocess
import psutil
import re
import shutil
import threading
from typing import Any, Dict, List, Optional, Set
# Avoid aggressive monkey-patching at import time; only enable when explicitly requested
if os.environ.get('USE_EVENTLET', '0') == '1':
    import eventlet
    eventlet.monkey_patch()

from flask import Blueprint, jsonify, request, send_file, Response, current_app, redirect
import logging
import numpy as np
import uuid
from datetime import datetime
from datetime import datetime
import io
import traceback
import time
import asyncio
import concurrent.futures

_background_loop_lock = threading.Lock()
_background_loop: Optional[asyncio.AbstractEventLoop] = None


def _ensure_background_loop() -> asyncio.AbstractEventLoop:
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
                        # If loop is already running (e.g. attached to thread by something else),
                        # just log and continue. The loop object is still valid.
                        logger.info(f"Background loop thread found running loop: {e}")
            except Exception as e:
                logger.error(f"Critical error in background loop worker: {e}")
                # Ensure we don't hang the main thread waiting
                if not loop_ready.is_set():
                    loop_ready.set()

        threading.Thread(target=_loop_worker, name="api-bg-loop", daemon=True).start()
        loop_ready.wait(timeout=5)

        # Even if run_forever failed, if we have a valid loop object, we might be okay 
        # (e.g. if it was already running). Double check is_running() only if we started it fresh.
        # If we just attached to an existing one, we assume it's good.
        if _background_loop and not _background_loop.is_running():
             # Try one more check - maybe it started after our check?
             time.sleep(0.1)
             if not _background_loop.is_running():
                 # Last resort: try to get ANY loop for this thread
                 try:
                     _background_loop = asyncio.get_event_loop()
                 except RuntimeError:
                     pass

        if not _background_loop:
             raise RuntimeError("Background asyncio loop failed to initialize")

        return _background_loop


def _run_coro_safe(coro):
    """Helper to run a coroutine safely without nesting event loops."""
    try:
        try:
            current_loop = asyncio.get_running_loop()
        except RuntimeError:
            current_loop = None

        target_loop = None

        # Prefer stream_server loop if available and running
        if stream_server and getattr(stream_server, 'running', False) and hasattr(stream_server, '_loop'):
            try:
                if stream_server._loop and stream_server._loop.is_running():
                    target_loop = stream_server._loop
            except Exception:
                target_loop = None

        # Otherwise use the shared background loop
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

import uuid
import cv2
from dataclasses import asdict
import base64
import uuid as _uuid
import os as _os
from pathlib import Path
from core.python_script_manager import PythonScriptManager, ScriptExecutionResult
from core.vehicle_counter import VehicleCounter
from core.entitlements import get_camera_limit



# Set up logger
logger = logging.getLogger(__name__)


def _file_uri(p: str) -> str:
    """Convert a filesystem path to a file:// URI (best-effort)."""
    try:
        from pathlib import Path as _P
        return _P(p).resolve().as_uri()
    except Exception:
        return str(p)


# Create the blueprint
api_bp = Blueprint('api', __name__)

# Global references to services (initialized by app.py)
camera_manager = None
stream_server = None
ai_analyzer = None
alert_system = None
ai_agent = None  # Add AI agent reference
db_manager = None
scheduler = None
python_script_manager = None
llm_service = None  # Add LLM service
audio_monitor = None  # AudioMonitorManager (core/audio_monitor.py)
event_index_service = None  # Event index/search service (captures)
_events_reindex_lock = threading.Lock()
_events_reindex_state: Dict[str, Any] = {
    "running": False,
    "job_id": None,
    "started_at": None,
    "finished_at": None,
    "updated_at": None,
    "total_target": None,  # best-effort: number of items we intend to process (after filters/max_files)
    "scanned": 0,
    "processed": 0,
    "skipped": 0,
    "errors": [],
    "error_count": 0,
    "eta_seconds": None,
    "cloud": {"enabled": False, "max_calls": 0, "calls": 0, "provider": None, "model": None},
    "config": {},
}

# Background indexer to keep Motion Watch captures searchable even after restarts.
_events_bg_ingest_started = False


def _events_background_ingest_loop() -> None:
    """
    Periodically ingest new Motion Watch captures into the local event index.

    This is intentionally bounded and light-weight: we index events (metadata + thumbnails)
    without forcing heavy detections runs. Queries that require detections can still trigger
    an on-demand ingest with detections enabled.
    """
    while True:
        try:
            svc = event_index_service
            if svc is not None:
                # Keep this modest; it runs repeatedly.
                svc.backfill(max_items=80, include_vision=False, include_detections=True)
        except Exception:
            # Best-effort; never crash the server thread.
            pass
        time.sleep(30)

# Layouts / Profiles / Sessions (local-file source of truth)
layouts_store = None
session_manager = None

__all__ = ['api_bp', 'register_routes']


def register_routes(app, cm=None, ss=None, ai=None, alerts=None, agent=None, db=None, sched=None, scripts=None, llm=None, audio=None):
    """Register all routes and initialize service references"""
    global camera_manager, stream_server, ai_analyzer, alert_system, ai_agent, db_manager, scheduler, python_script_manager, llm_service, audio_monitor, event_index_service
    global layouts_store, session_manager

    # Set global service references
    camera_manager = cm
    stream_server = ss
    ai_analyzer = ai
    alert_system = alerts
    ai_agent = agent  # Add AI agent
    db_manager = db
    scheduler = sched
    python_script_manager = scripts
    llm_service = llm  # Add LLM service
    audio_monitor = audio

    # Initialize event index service (captures/motion_watch -> searchable timeline)
    if event_index_service is None:
        try:
            from core.event_index_service import EventIndexService

            event_index_service = EventIndexService()
            logger.info("Initialized EventIndexService at %s", getattr(event_index_service, "db_path", "data/events_index.sqlite"))
        except Exception as exc:
            event_index_service = None
            logger.warning("EventIndexService unavailable: %s", exc)

    # Start a background ingester so new Motion Watch captures appear without manual backfill.
    global _events_bg_ingest_started
    if (not _events_bg_ingest_started) and (event_index_service is not None):
        try:
            t = threading.Thread(target=_events_background_ingest_loop, daemon=True)
            t.start()
            _events_bg_ingest_started = True
            logger.info("Started background events ingester")
        except Exception as exc:
            logger.warning("Failed to start background events ingester: %s", exc)

    if python_script_manager is None:
        try:
            script_dir = app.config.get('PYTHON_SCRIPT_PATH', 'data/python_scripts')
            resolver = None
            if camera_manager is not None:
                def _resolve(camera_id: str) -> Optional[Dict[str, Any]]:
                    cam = safe_service_call(camera_manager, 'get_camera', None, camera_id)
                    if cam:
                        return cam
                    return safe_service_call(camera_manager, 'get_camera_by_name', None, camera_id)
                resolver = _resolve
            python_script_manager = PythonScriptManager(Path(script_dir), camera_resolver=resolver)
            logger.info("Initialized default PythonScriptManager at %s", script_dir)
        except Exception as exc:
            python_script_manager = None
            logger.error("Failed to initialize PythonScriptManager: %s", exc)

    # Register the blueprint
    app.register_blueprint(api_bp, url_prefix='/api')

    logger.info("API routes registered successfully")
    logger.info(
        f"Services initialized: CM={cm is not None}, SS={ss is not None}, AI={ai is not None}, Alerts={alerts is not None}, Agent={agent is not None}, Audio={audio is not None}")

    # Register Layouts/Profiles/Sessions routes (safe to load even if cm is None)
    try:
        from core.layout_store import LayoutsAndProfilesStore
        from core.session_manager import SessionManager
        from api.layouts_profiles_routes import init_layouts_profiles_routes
        from api.sessions_routes import init_sessions_routes

        if layouts_store is None:
            layouts_store = LayoutsAndProfilesStore()
        if session_manager is None:
            session_manager = SessionManager(store=layouts_store, camera_manager=camera_manager)

        init_layouts_profiles_routes(app, layouts_store)
        init_sessions_routes(app, session_manager)
    except Exception as e:
        logger.warning(f"Layouts/Profiles/Sessions routes not registered: {e}")

    # Register System resource telemetry routes
    try:
        from api.system_routes import init_system_routes
        init_system_routes(app)
    except Exception as e:
        logger.warning(f"System routes not registered: {e}")

    # Register Remote Access routes (WireGuard VPN)
    try:
        from api.remote_access_routes import init_remote_access_routes
        init_remote_access_routes(app)
    except Exception as e:
        logger.warning(f"Remote access routes not registered: {e}")


def safe_service_call(service, method_name, default_return=None, *args, **kwargs):
    """Safely call a service method with error handling"""
    try:
        if service is None:
            logger.debug(f"Service not available for {method_name}")
            return default_return

        method = getattr(service, method_name, None)
        if method is None:
            logger.debug(f"Method {method_name} not found on service")
            return default_return

        if not callable(method):
            logger.debug(f"Method {method_name} is not callable")
            return default_return

        return method(*args, **kwargs)
    except Exception as e:
        logger.warning(f"Error calling {method_name}: {e}")
        return default_return


# ==================== EVENTS (CAPTURE INDEX) ====================

@api_bp.route('/events/status', methods=['GET'])
def events_status():
    try:
        if not event_index_service:
            return jsonify({'success': False, 'message': 'Event index not available'}), 503
        return jsonify({'success': True, 'data': event_index_service.status()})
    except Exception as e:
        logger.error(f"Events status error: {e}")
        return jsonify({'success': False, 'message': 'Failed to get events status'}), 500


@api_bp.route('/events/ingest', methods=['POST'])
def events_ingest():
    """
    Ingest an event capture into the local event index.
    Intended caller: Desktop Motion Watch capture (sidecar JSON + image path).
    Emits a ``new_capture`` Socket.IO event on ``/realtime`` for live report consumers.
    """
    try:
        if not event_index_service:
            return jsonify({'success': False, 'message': 'Event index not available'}), 503
        data = request.get_json() or {}
        result = event_index_service.ingest(data)

        # Emit real-time event for live security report consumers
        try:
            sio = current_app.extensions.get("socketio")
            if sio is not None:
                file_p = str(result.get("file_path") or "")
                thumb_p = str(result.get("thumb_path") or "")
                def _file_uri_local(p: str) -> str:
                    if not p:
                        return ""
                    from pathlib import Path as _P
                    return _P(p).as_uri() if _P(p).is_absolute() else f"file:///{_P(p).as_posix()}"
                sio.emit('new_capture', {
                    'event_id': str(result.get("event_id") or ""),
                    'captured_ts': int(result.get("captured_ts") or 0),
                    'camera_name': str(result.get("camera_name") or ""),
                    'caption': str(result.get("caption") or ""),
                    'file_uri': _file_uri_local(file_p),
                    'thumb_uri': _file_uri_local(thumb_p) if thumb_p else _file_uri_local(file_p),
                    'shape_name': str(result.get("shape_name") or ""),
                    'trigger_type': str(result.get("trigger_type") or ""),
                    'media_type': str(result.get("media_type") or "image"),
                    'tags': [str(t) for t in (result.get("tags") or [])],
                    'detection_classes': [str(c) for c in (result.get("detection_classes") or [])],
                }, namespace='/realtime')
        except Exception:
            pass

        return jsonify({'success': True, 'data': result})
    except FileNotFoundError as e:
        return jsonify({'success': False, 'message': f'File not found: {e}'}), 404
    except ValueError as e:
        return jsonify({'success': False, 'message': str(e)}), 400
    except Exception as e:
        logger.error(f"Events ingest error: {e}")
        return jsonify({'success': False, 'message': 'Failed to ingest event'}), 500


@api_bp.route('/events/search', methods=['POST'])
def events_search():
    """
    Search indexed captures and return a timeline for the desktop terminal.
    """
    try:
        if not event_index_service:
            return jsonify({'success': False, 'message': 'Event index not available'}), 503
        data = request.get_json() or {}
        query = str(data.get("query") or data.get("text") or "").strip()
        limit = int(data.get("limit", 25) or 25)
        camera_ref = data.get("camera_name") or data.get("cameraName") or data.get("cameraRef") or data.get("camera")
        filters = data.get("filters") if isinstance(data.get("filters"), dict) else {}
        dominant_color = (filters.get("dominant_color") or filters.get("color") or data.get("dominant_color") or data.get("color"))
        trigger_type = (filters.get("trigger_type") or data.get("trigger_type"))
        shape_name = (filters.get("shape_name") or data.get("shape_name"))
        detection_classes = (filters.get("detection_classes") or filters.get("classes") or data.get("detection_classes") or data.get("classes"))
        if isinstance(detection_classes, str):
            detection_classes = [detection_classes]
        if not isinstance(detection_classes, list):
            detection_classes = []
        # New: detection-level filters (object-level index)
        det_filter = filters.get("detection") if isinstance(filters.get("detection"), dict) else {}
        detection_color = (
            det_filter.get("color")
            or det_filter.get("dominant_color")
            or filters.get("detection_color")
            or filters.get("detectionColor")
            or data.get("detection_color")
            or data.get("detectionColor")
        )
        min_confidence = det_filter.get("min_confidence") or det_filter.get("minConfidence") or filters.get("min_confidence") or data.get("min_confidence")
        min_area = det_filter.get("min_area") or det_filter.get("minArea") or filters.get("min_area") or data.get("min_area")
        include_detections = bool(data.get("include_detections", data.get("includeDetections", False)))

        # Heuristic parsing: allow plain-text queries like "red cars" to drive detection filters.
        # Only applies when the caller did not explicitly set detection filters.
        try:
            if query and not detection_color and not detection_classes:
                tokens = re.findall(r"[a-z0-9]+", query.lower())
                color_map = {
                    "grey": "gray",
                    "silver": "gray",
                    "dark": None,   # ignore vague modifiers
                    "light": None,
                }
                color_set = {"white", "black", "gray", "grey", "red", "blue", "green", "yellow", "brown", "silver"}
                class_map = {
                    "car": "car",
                    "cars": "car",
                    "truck": "truck",
                    "trucks": "truck",
                    "bus": "bus",
                    "buses": "bus",
                    "motorcycle": "motorcycle",
                    "motorcycles": "motorcycle",
                    "bike": "bicycle",
                    "bikes": "bicycle",
                    "person": "person",
                    "people": "person",
                    "dog": "dog",
                    "dogs": "dog",
                    "cat": "cat",
                    "cats": "cat",
                }

                found_color = None
                found_class = None
                for t in tokens:
                    if t in color_set and not found_color:
                        found_color = color_map.get(t, t)
                    if t in class_map and not found_class:
                        found_class = class_map[t]

                # Apply as detection-level filters
                if found_color:
                    detection_color = found_color
                if found_class:
                    detection_classes = [found_class]

                # Remove matched tokens from free-text query so FTS doesn't broaden results.
                if found_color or found_class:
                    drop = set([x for x in [found_color, found_class] if x])
                    # also drop plural forms if present
                    if found_class:
                        drop.update({found_class + "s"})
                    q2 = [t for t in tokens if t not in drop and t not in {"grey", "silver"}]
                    query = " ".join(q2).strip()
        except Exception:
            pass

        # Time filtering: accept ISO or unix seconds
        start_ts = data.get("start_ts") or data.get("startTs") or data.get("start")
        end_ts = data.get("end_ts") or data.get("endTs") or data.get("end")
        def _to_ts(v):
            if v is None:
                return None
            if isinstance(v, (int, float)):
                return int(v)
            if isinstance(v, str) and v.strip():
                s = v.strip()
                if s.isdigit():
                    return int(s)
                try:
                    dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
                    return int(dt.timestamp())
                except Exception:
                    return None
            return None
        start_ts_i = _to_ts(start_ts)
        end_ts_i = _to_ts(end_ts)

        min_conf_f = None
        min_area_f = None
        try:
            if min_confidence is not None and str(min_confidence).strip() != "":
                min_conf_f = float(min_confidence)
        except Exception:
            min_conf_f = None
        try:
            if min_area is not None and str(min_area).strip() != "":
                min_area_f = float(min_area)
        except Exception:
            min_area_f = None

        hits = event_index_service.search(
            query=query,
            camera_name=str(camera_ref).strip() if isinstance(camera_ref, str) and str(camera_ref).strip() else None,
            trigger_type=str(trigger_type).strip() if isinstance(trigger_type, str) and trigger_type.strip() else None,
            shape_name=str(shape_name).strip() if isinstance(shape_name, str) and shape_name.strip() else None,
            dominant_color=str(dominant_color).strip() if isinstance(dominant_color, str) and dominant_color.strip() else None,
            detection_classes=[str(c).strip() for c in detection_classes if isinstance(c, (str, int, float)) and str(c).strip()],
            detection_color=str(detection_color).strip() if isinstance(detection_color, str) and str(detection_color).strip() else None,
            min_confidence=min_conf_f,
            min_area=min_area_f,
            start_ts=start_ts_i,
            end_ts=end_ts_i,
            limit=limit,
        )

        # Auto-refresh (bounded): if 0 hits, try ingesting the most recent captures for this camera
        # and retry once. This fixes the "just captured but not indexed yet" case (e.g., backend restart).
        try:
            auto_refresh = data.get("auto_refresh")
            if auto_refresh is None:
                auto_refresh = True
            if auto_refresh and not hits and isinstance(camera_ref, str) and str(camera_ref).strip():
                cam_ref = str(camera_ref).strip()
                cam_id = None
                # Resolve camera name -> id from cameras.json
                try:
                    root = Path(getattr(event_index_service, "project_root", Path.cwd()))
                    # Prefer per-user data dir (portable/frozen builds), fall back to repo-root paths.
                    try:
                        from core.paths import get_data_dir
                        data_dir = get_data_dir()
                        candidates = [data_dir / "cameras.json", data_dir.parent / "cameras.json"]
                    except Exception:
                        candidates = []
                    candidates.extend([root / "data" / "cameras.json", root / "cameras.json"])
                    for p in candidates:
                        if not p.exists():
                            continue
                        raw = json.loads(p.read_text(encoding="utf-8", errors="ignore"))
                        if isinstance(raw, list):
                            for item in raw:
                                if not isinstance(item, dict):
                                    continue
                                cid = str(item.get("id") or "").strip()
                                nm = str(item.get("name") or "").strip()
                                if nm and nm.lower() == cam_ref.lower() and cid:
                                    cam_id = cid
                                    break
                        if cam_id:
                            break
                except Exception:
                    cam_id = None

                # If cam_ref looks like an id already, accept it
                if cam_id is None and "-" in cam_ref and len(cam_ref) >= 16:
                    cam_id = cam_ref

                if cam_id and hasattr(event_index_service, "ingest_recent_for_camera"):
                    event_index_service.ingest_recent_for_camera(
                        str(cam_id),
                        max_items=25,
                        include_detections=True,
                        include_vision=False,
                    )
                    hits = event_index_service.search(
                        query=query,
                        camera_name=str(camera_ref).strip() if isinstance(camera_ref, str) and str(camera_ref).strip() else None,
                        trigger_type=str(trigger_type).strip() if isinstance(trigger_type, str) and trigger_type.strip() else None,
                        shape_name=str(shape_name).strip() if isinstance(shape_name, str) and shape_name.strip() else None,
                        dominant_color=str(dominant_color).strip() if isinstance(dominant_color, str) and dominant_color.strip() else None,
                        detection_classes=[str(c).strip() for c in detection_classes if isinstance(c, (str, int, float)) and str(c).strip()],
                        detection_color=str(detection_color).strip() if isinstance(detection_color, str) and str(detection_color).strip() else None,
                        min_confidence=min_conf_f,
                        min_area=min_area_f,
                        start_ts=start_ts_i,
                        end_ts=end_ts_i,
                        limit=limit,
                    )
        except Exception:
            pass

        # Batch detection counts for UI expansion links
        det_counts: Dict[str, int] = {}
        if hits:
            try:
                event_ids = [str(h.id) for h in hits if getattr(h, "id", None)]
                if event_ids:
                    placeholders = ",".join(["?"] * len(event_ids))
                    with event_index_service._connect() as conn:  # pylint: disable=protected-access
                        rows = conn.execute(
                            f"SELECT event_id, COUNT(1) AS c FROM event_detections WHERE event_id IN ({placeholders}) GROUP BY event_id;",
                            event_ids,
                        ).fetchall()
                    for r in rows:
                        det_counts[str(r["event_id"])] = int(r["c"] or 0)
            except Exception:
                det_counts = {}

        timeline = []
        for h in hits:
            thumb_b64 = event_index_service.read_thumb_base64(h.thumb_path) if h.thumb_path else None
            file_p = str(h.file_path) if h.file_path else ""
            thumb_p = str(h.thumb_path) if h.thumb_path else ""
            eid = str(h.id)
            api_file_url = f"/api/events/file?id={eid}"
            api_thumb_url = f"/api/events/file?id={eid}&kind=thumb"
            timeline.append(
                {
                    "event_id": eid,
                    "id": eid,
                    "captured_at": h.captured_at,
                    "captured_ts": h.captured_ts,
                    "camera_name": h.camera_name,
                    "file_path": file_p,
                    "file_uri": _file_uri(file_p) if file_p else "",
                    "api_file_url": api_file_url,
                    "thumb_path": thumb_p,
                    "thumb_uri": _file_uri(thumb_p) if thumb_p else (_file_uri(file_p) if file_p else ""),
                    "api_thumb_url": api_thumb_url,
                    "thumb_base64": thumb_b64,
                    "caption": h.caption,
                    "tags": h.tags,
                    "detection_classes": h.detection_classes or [],
                    "dominant_color": h.dominant_color,
                    "trigger_type": h.trigger_type,
                    "shape_type": h.shape_type,
                    "shape_name": h.shape_name,
                    "media_type": getattr(h, "media_type", None) or "image",
                    "reason": h.caption or "",
                    "detections_count": int(det_counts.get(str(h.id), 0)),
                }
            )
            if include_detections:
                try:
                    timeline[-1]["detections"] = event_index_service.list_detections(str(h.id))[:24]
                except Exception:
                    timeline[-1]["detections"] = []

        summary = f"Found {len(timeline)} event(s)."
        return jsonify({'success': True, 'data': {'message': summary, 'timeline': timeline}})
    except Exception as e:
        logger.error(f"Events search error: {e}")
        return jsonify({'success': False, 'message': 'Failed to search events'}), 500


@api_bp.route('/events/thumb', methods=['GET'])
def events_thumb():
    """
    Safely serve an indexed thumbnail by event_id.
    """
    try:
        if not event_index_service:
            return jsonify({'success': False, 'message': 'Event index not available'}), 503
        event_id = request.args.get("event_id") or request.args.get("id")
        if not event_id:
            return jsonify({'success': False, 'message': 'event_id is required'}), 400

        with event_index_service._connect() as conn:  # pylint: disable=protected-access
            row = conn.execute("SELECT thumb_path FROM events WHERE id = ? LIMIT 1;", (str(event_id),)).fetchone()
        if not row or not row["thumb_path"]:
            return jsonify({'success': False, 'message': 'Thumbnail not found'}), 404

        p = Path(str(row["thumb_path"]))
        if not p.is_absolute():
            p = (Path.cwd() / p).resolve()
        if not p.exists():
            return jsonify({'success': False, 'message': 'Thumbnail missing on disk'}), 404

        thumbs_root = Path(getattr(event_index_service, "thumbs_dir", Path("data/events_thumbs")))
        thumbs_root = (Path.cwd() / thumbs_root).resolve() if not thumbs_root.is_absolute() else thumbs_root.resolve()
        try:
            p.relative_to(thumbs_root)
        except Exception:
            return jsonify({'success': False, 'message': 'Invalid thumbnail path'}), 400

        return send_file(str(p), mimetype="image/jpeg", conditional=True)
    except Exception as e:
        logger.error(f"Events thumb error: {e}")
        return jsonify({'success': False, 'message': 'Failed to serve thumbnail'}), 500


@api_bp.route('/events/file', methods=['GET'])
def events_file():
    """
    Serve an event capture file (image or clip) by event_id.
    ``kind`` param: ``file`` (full capture), ``thumb`` (static JPG thumbnail),
    ``preview`` (animated MP4 preview for clip moving thumbnails).
    Used by the live report dashboard where file:// URIs are not accessible.
    """
    try:
        if not event_index_service:
            return jsonify({'success': False, 'message': 'Event index not available'}), 503
        event_id = request.args.get("event_id") or request.args.get("id")
        kind = request.args.get("kind", "file")
        if not event_id:
            return jsonify({'success': False, 'message': 'event_id is required'}), 400

        col = "thumb_path" if kind == "thumb" else "file_path"
        with event_index_service._connect() as conn:
            row = conn.execute(f"SELECT file_path, thumb_path FROM events WHERE id = ? LIMIT 1;", (str(event_id),)).fetchone()
        if not row:
            return jsonify({'success': False, 'message': 'Event not found'}), 404

        from core.paths import get_project_root

        def _resolve(raw):
            if not raw:
                return None
            rp = Path(str(raw))
            if rp.is_absolute():
                return rp.resolve() if rp.exists() else None
            # Try project root first
            candidate = (get_project_root() / rp).resolve()
            if candidate.exists():
                return candidate
            # Check each capture root (handles custom save directories)
            for cr in getattr(event_index_service, "capture_roots", []):
                try:
                    candidate = (cr / rp).resolve()
                    if candidate.exists():
                        return candidate
                except Exception:
                    continue
            return None

        if kind == "preview":
            # Serve the animated .preview.mp4 if it exists, fall back to static thumb
            fp = _resolve(row["file_path"])
            if fp:
                preview_p = fp.with_suffix(".preview.mp4")
                if preview_p.exists():
                    return send_file(str(preview_p), mimetype="video/mp4", conditional=True)
            # Fall back to static thumb
            tp = _resolve(row["thumb_path"])
            if tp:
                return send_file(str(tp), mimetype="image/jpeg", conditional=True)
            return jsonify({'success': False, 'message': 'Preview not found'}), 404

        target_col = "thumb_path" if kind == "thumb" else "file_path"
        p = _resolve(row[target_col])
        if not p:
            return jsonify({'success': False, 'message': 'File not found'}), 404

        mime_map = {
            '.jpg': 'image/jpeg', '.jpeg': 'image/jpeg', '.png': 'image/png',
            '.mp4': 'video/mp4', '.avi': 'video/x-msvideo', '.mkv': 'video/x-matroska',
            '.webm': 'video/webm',
        }
        mime = mime_map.get(p.suffix.lower(), 'application/octet-stream')
        return send_file(str(p), mimetype=mime, conditional=True)
    except Exception as e:
        logger.error(f"Events file serve error: {e}")
        return jsonify({'success': False, 'message': 'Failed to serve file'}), 500


@api_bp.route('/events/backfill', methods=['POST'])
def events_backfill():
    """
    Bounded backfill to index existing captures under captures/motion_watch.
    """
    try:
        if not event_index_service:
            return jsonify({'success': False, 'message': 'Event index not available'}), 503
        data = request.get_json() or {}
        max_items = int(data.get("max_items", data.get("maxItems", 250)) or 250)
        include_vision = bool(data.get("include_vision", data.get("includeVision", False)))
        include_detections = bool(data.get("include_detections", data.get("includeDetections", True)))
        result = event_index_service.backfill(
            max_items=max_items,
            include_vision=include_vision,
            include_detections=include_detections,
        )
        return jsonify({'success': True, 'data': result})
    except Exception as e:
        logger.error(f"Events backfill error: {e}")
        return jsonify({'success': False, 'message': 'Failed to backfill events'}), 500


@api_bp.route('/events/detections', methods=['GET'])
def events_detections():
    """
    Fetch object detections for an event (with overrides applied). Optionally include small crop images.
    """
    try:
        if not event_index_service:
            return jsonify({'success': False, 'message': 'Event index not available'}), 503
        event_id = request.args.get("event_id") or request.args.get("id")
        if not event_id:
            return jsonify({'success': False, 'message': 'event_id is required'}), 400
        include_images = str(request.args.get("include_images") or request.args.get("includeImages") or "").lower() in {"1", "true", "yes"}
        limit = request.args.get("limit")
        try:
            limit_i = max(1, min(int(limit or 50), 100))
        except Exception:
            limit_i = 50

        dets = event_index_service.list_detections(str(event_id))[:limit_i]
        if include_images:
            crops_root = Path(getattr(event_index_service, "crops_dir", Path("data/events_crops")))
            crops_root = (Path.cwd() / crops_root).resolve() if not crops_root.is_absolute() else crops_root.resolve()
            for d in dets:
                try:
                    cp = d.get("crop_path")
                    if not cp:
                        continue
                    p = Path(str(cp))
                    if not p.is_absolute():
                        p = (Path.cwd() / p).resolve()
                    else:
                        p = p.resolve()
                    try:
                        p.relative_to(crops_root)
                    except Exception:
                        continue
                    b64 = event_index_service.read_file_base64(str(p), max_bytes=240_000)
                    if b64:
                        d["crop_base64"] = b64
                except Exception:
                    continue

        return jsonify({'success': True, 'data': {'event_id': str(event_id), 'detections': dets}})
    except Exception as e:
        logger.error(f"Events detections error: {e}")
        return jsonify({'success': False, 'message': 'Failed to load detections'}), 500


@api_bp.route('/events/count', methods=['POST'])
def events_count():
    """
    Aggregate counts over the event index without returning a full timeline.

    Intended for questions like:
      - "how many cars passed the produce stand yesterday"
      - "total red cars today"
    """
    try:
        if not event_index_service:
            return jsonify({'success': False, 'message': 'Event index not available'}), 503
        data = request.get_json() or {}

        camera_ref = data.get("camera_name") or data.get("cameraName") or data.get("cameraRef") or data.get("camera")
        filters = data.get("filters") if isinstance(data.get("filters"), dict) else {}

        # Time filtering: accept ISO or unix seconds
        start_ts = data.get("start_ts") or data.get("startTs") or data.get("start")
        end_ts = data.get("end_ts") or data.get("endTs") or data.get("end")

        def _to_ts(v):
            if v is None:
                return None
            if isinstance(v, (int, float)):
                return int(v)
            if isinstance(v, str) and v.strip():
                s = v.strip()
                if s.isdigit():
                    return int(s)
                try:
                    dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
                    return int(dt.timestamp())
                except Exception:
                    return None
            return None

        start_ts_i = _to_ts(start_ts)
        end_ts_i = _to_ts(end_ts)

        det_filter = filters.get("detection") if isinstance(filters.get("detection"), dict) else {}
        detection_classes = (
            det_filter.get("classes")
            or filters.get("detection_classes")
            or filters.get("classes")
            or data.get("detection_classes")
            or data.get("classes")
        )
        if isinstance(detection_classes, str):
            detection_classes = [detection_classes]
        if not isinstance(detection_classes, list):
            detection_classes = []

        detection_color = (
            det_filter.get("color")
            or det_filter.get("dominant_color")
            or filters.get("detection_color")
            or data.get("detection_color")
        )
        min_confidence = det_filter.get("min_confidence") or det_filter.get("minConfidence") or data.get("min_confidence")
        min_area = det_filter.get("min_area") or det_filter.get("minArea") or data.get("min_area")

        min_conf_f = None
        min_area_f = None
        try:
            if min_confidence is not None and str(min_confidence).strip() != "":
                min_conf_f = float(min_confidence)
        except Exception:
            min_conf_f = None
        try:
            if min_area is not None and str(min_area).strip() != "":
                min_area_f = float(min_area)
        except Exception:
            min_area_f = None

        result = event_index_service.count(
            camera_name=str(camera_ref).strip() if isinstance(camera_ref, str) and str(camera_ref).strip() else None,
            start_ts=start_ts_i,
            end_ts=end_ts_i,
            detection_classes=[str(c).strip() for c in detection_classes if str(c).strip()],
            detection_color=str(detection_color).strip() if isinstance(detection_color, str) and str(detection_color).strip() else None,
            min_confidence=min_conf_f,
            min_area=min_area_f,
        )

        return jsonify({'success': True, 'data': result})
    except Exception as e:
        logger.error(f"Events count error: {e}")
        return jsonify({'success': False, 'message': 'Failed to count events'}), 500


@api_bp.route('/events/vehicle_count', methods=['POST'])
def events_vehicle_count():
    """
    Count unique vehicles by linking detections across captures into simple tracks.

    Intended for questions like:
      - "how many cars passed the produce stand today"
    """
    try:
        if not event_index_service:
            return jsonify({'success': False, 'message': 'Event index not available'}), 503
        data = request.get_json() or {}

        camera_ref = data.get("camera_name") or data.get("cameraName") or data.get("cameraRef") or data.get("camera")
        filters = data.get("filters") if isinstance(data.get("filters"), dict) else {}

        # Time filtering: accept ISO or unix seconds
        start_ts = data.get("start_ts") or data.get("startTs") or data.get("start")
        end_ts = data.get("end_ts") or data.get("endTs") or data.get("end")

        def _to_ts(v):
            if v is None:
                return None
            if isinstance(v, (int, float)):
                return int(v)
            if isinstance(v, str) and v.strip():
                s = v.strip()
                if s.isdigit():
                    return int(s)
                try:
                    dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
                    return int(dt.timestamp())
                except Exception:
                    return None
            return None

        start_ts_i = _to_ts(start_ts)
        end_ts_i = _to_ts(end_ts)

        # detection filters
        det_filter = filters.get("detection") if isinstance(filters.get("detection"), dict) else {}
        detection_classes = (
            det_filter.get("classes")
            or filters.get("detection_classes")
            or filters.get("classes")
            or data.get("detection_classes")
            or data.get("classes")
        )
        if isinstance(detection_classes, str):
            detection_classes = [detection_classes]
        if not isinstance(detection_classes, list):
            detection_classes = []

        min_confidence = (
            det_filter.get("min_confidence")
            or det_filter.get("minConfidence")
            or filters.get("min_confidence")
            or data.get("min_confidence")
        )
        max_gap_seconds = data.get("max_gap_seconds") or data.get("maxGapSeconds") or filters.get("max_gap_seconds")
        max_distance_px = data.get("max_distance_px") or data.get("maxDistancePx") or filters.get("max_distance_px")
        auto_refresh = data.get("auto_refresh")
        if auto_refresh is None:
            auto_refresh = True

        try:
            min_conf_f = float(min_confidence) if min_confidence is not None and str(min_confidence).strip() != "" else 0.25
        except Exception:
            min_conf_f = 0.25
        try:
            max_gap_i = int(max_gap_seconds) if max_gap_seconds is not None and str(max_gap_seconds).strip() != "" else 12
        except Exception:
            max_gap_i = 12
        try:
            max_dist_f = float(max_distance_px) if max_distance_px is not None and str(max_distance_px).strip() != "" else 140.0
        except Exception:
            max_dist_f = 140.0

        def _compute() -> Dict[str, Any]:
            return event_index_service.count_unique_vehicles(
                camera_name=str(camera_ref).strip() if isinstance(camera_ref, str) and str(camera_ref).strip() else None,
                start_ts=start_ts_i,
                end_ts=end_ts_i,
                detection_classes=[str(c).strip() for c in detection_classes if str(c).strip()],
                min_confidence=min_conf_f,
                max_gap_seconds=max_gap_i,
                max_distance_px=max_dist_f,
            )

        result = _compute()

        # Auto-refresh detections for this camera if we got nothing (common after backend restart).
        try:
            if auto_refresh and (int(result.get("unique_vehicle_count") or 0) == 0) and isinstance(camera_ref, str) and str(camera_ref).strip():
                cam_ref = str(camera_ref).strip()
                cam_id = None
                # Resolve camera name -> id from cameras.json
                try:
                    root = Path(getattr(event_index_service, "project_root", Path.cwd()))
                    for p in [root / "data" / "cameras.json", root / "cameras.json"]:
                        if not p.exists():
                            continue
                        raw = json.loads(p.read_text(encoding="utf-8", errors="ignore"))
                        if isinstance(raw, list):
                            for item in raw:
                                if not isinstance(item, dict):
                                    continue
                                cid = str(item.get("id") or "").strip()
                                nm = str(item.get("name") or "").strip()
                                if nm and nm.lower() == cam_ref.lower() and cid:
                                    cam_id = cid
                                    break
                        if cam_id:
                            break
                except Exception:
                    cam_id = None
                # If cam_ref looks like an id already, accept it
                if cam_id is None and "-" in cam_ref and len(cam_ref) >= 16:
                    cam_id = cam_ref
                if cam_id and hasattr(event_index_service, "ingest_recent_for_camera"):
                    event_index_service.ingest_recent_for_camera(
                        str(cam_id),
                        max_items=50,
                        include_detections=True,
                        include_vision=False,
                    )
                    result = _compute()
        except Exception:
            pass

        return jsonify({'success': True, 'data': result})
    except Exception as e:
        logger.error(f"Events vehicle_count error: {e}")
        return jsonify({'success': False, 'message': 'Failed to count vehicles'}), 500


@api_bp.route('/events/report', methods=['POST'])
def events_report():
    """
    Generate a small HTML "security report" timeline:
    - timestamped rows
    - crop thumbnails (file:// links)
    - direct file links to the original capture and its folder

    This is intended for coherent review + sharing locally (no cloud calls).
    """
    try:
        if not event_index_service:
            return jsonify({'success': False, 'message': 'Event index not available'}), 503
        data = request.get_json() or {}

        title = str(data.get("title") or "Security Report").strip() or "Security Report"
        query = str(data.get("query") or data.get("text") or "").strip()
        limit = int(data.get("limit", 200) or 200)
        # Reports are meant for "everything in one view". Keep a hard cap to avoid huge HTML.
        limit = max(10, min(limit, 5000))
        camera_name = data.get("camera_name") or data.get("cameraName") or data.get("cameraRef") or data.get("camera")
        filters = data.get("filters") if isinstance(data.get("filters"), dict) else {}

        # Reuse the same filter parsing as events_search
        dominant_color = (filters.get("dominant_color") or filters.get("color") or data.get("dominant_color") or data.get("color"))
        trigger_type = (filters.get("trigger_type") or data.get("trigger_type"))
        shape_name = (filters.get("shape_name") or data.get("shape_name"))

        det_filter = filters.get("detection") if isinstance(filters.get("detection"), dict) else {}
        detection_color = (
            det_filter.get("color")
            or det_filter.get("dominant_color")
            or filters.get("detection_color")
            or filters.get("detectionColor")
            or data.get("detection_color")
            or data.get("detectionColor")
        )
        detection_classes = (
            det_filter.get("classes")
            or filters.get("detection_classes")
            or filters.get("classes")
            or data.get("detection_classes")
            or data.get("classes")
        )
        if isinstance(detection_classes, str):
            detection_classes = [detection_classes]
        if not isinstance(detection_classes, list):
            detection_classes = []

        min_confidence = det_filter.get("min_confidence") or det_filter.get("minConfidence") or filters.get("min_confidence") or data.get("min_confidence")
        min_area = det_filter.get("min_area") or det_filter.get("minArea") or filters.get("min_area") or data.get("min_area")

        min_conf_f = None
        min_area_f = None
        try:
            if min_confidence is not None and str(min_confidence).strip() != "":
                min_conf_f = float(min_confidence)
        except Exception:
            min_conf_f = None
        try:
            if min_area is not None and str(min_area).strip() != "":
                min_area_f = float(min_area)
        except Exception:
            min_area_f = None

        # Time filtering: accept ISO or unix seconds
        start_ts = data.get("start_ts") or data.get("startTs") or data.get("start")
        end_ts = data.get("end_ts") or data.get("endTs") or data.get("end")

        def _to_ts(v):
            if v is None:
                return None
            if isinstance(v, (int, float)):
                return int(v)
            if isinstance(v, str) and v.strip():
                s = v.strip()
                if s.isdigit():
                    return int(s)
                try:
                    dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
                    return int(dt.timestamp())
                except Exception:
                    return None
            return None

        start_ts_i = _to_ts(start_ts)
        end_ts_i = _to_ts(end_ts)

        auto_refresh = data.get("auto_refresh")
        if auto_refresh is None:
            auto_refresh = True

        # Heuristic parsing: allow plain-text queries like "red cars yesterday" to drive filters.
        # Applies only when caller did not explicitly set detection filters / time range.
        try:
            text_l = (query or "").lower()
            if text_l and not detection_color and not detection_classes:
                tokens = re.findall(r"[a-z0-9]+", text_l)
                cam_tokens = set()
                try:
                    if isinstance(camera_name, str) and camera_name.strip():
                        cam_tokens = set(re.findall(r"[a-z0-9]+", camera_name.lower()))
                except Exception:
                    cam_tokens = set()
                color_map = {"grey": "gray", "silver": "gray"}
                color_set = {"white", "black", "gray", "grey", "red", "blue", "green", "yellow", "brown", "silver"}
                class_map = {
                    "car": "car",
                    "cars": "car",
                    # YOLO doesn't have an explicit "suv" class; approximate as car for search/reporting.
                    "suv": "car",
                    "suvs": "car",
                    "truck": "truck",
                    "trucks": "truck",
                    "pickup": "truck",
                    "pickups": "truck",
                    "bus": "bus",
                    "buses": "bus",
                    "motorcycle": "motorcycle",
                    "motorcycles": "motorcycle",
                    "bike": "bicycle",
                    "bikes": "bicycle",
                    "person": "person",
                    "people": "person",
                }
                found_color = None
                found_class = None
                matched_class_token = None
                matched_color_token = None
                for t in tokens:
                    if t in color_set and not found_color:
                        found_color = color_map.get(t, t)
                        matched_color_token = t
                    if t in class_map and not found_class:
                        found_class = class_map[t]
                        matched_class_token = t
                if found_color:
                    detection_color = found_color
                if found_class:
                    detection_classes = [found_class]
                # remove matched tokens from query so FTS doesn't broaden
                if found_color or found_class:
                    drop = set()
                    if found_color:
                        drop.update({found_color, "grey", "silver"})
                        if matched_color_token:
                            drop.add(str(matched_color_token))
                    if found_class:
                        drop.update({found_class, found_class + "s"})
                        # Also drop the original user token (e.g., "suv"/"suvs") so we don't
                        # end up running an FTS query for an unsupported class name.
                        if matched_class_token:
                            drop.add(str(matched_class_token))
                    stop = {
                        "all",
                        "any",
                        "the",
                        "a",
                        "an",
                        "for",
                        "from",
                        "to",
                        "of",
                        "in",
                        "on",
                        "at",
                        "that",
                        "which",
                        "who",
                        "with",
                        "passed",
                        "pass",
                        "yesterday",
                        "today",
                        "report",
                        "suv",
                        "suvs",
                    }
                    # If a camera is provided separately, drop its name tokens from the free-text query
                    # to avoid over-filtering (camera filter already narrows results).
                    stop = stop.union(cam_tokens)
                    q2 = [t for t in tokens if t not in drop and t not in stop]
                    query = " ".join(q2).strip()
                    # If only stopwords remain, make it empty so detection filters drive matches.
                    if not query:
                        query = ""

            # Time keywords ("today"/"yesterday") if explicit range not provided
            if (start_ts_i is None and end_ts_i is None) and any(k in text_l for k in ["today", "yesterday"]):
                try:
                    from datetime import datetime as _dt
                    now = _dt.now()
                    start_today = _dt(year=now.year, month=now.month, day=now.day)
                    if "yesterday" in text_l:
                        start_ts_i = int((start_today.timestamp() - 86400))
                        end_ts_i = int(start_today.timestamp())
                    elif "today" in text_l:
                        start_ts_i = int(start_today.timestamp())
                        end_ts_i = int(_dt.now().timestamp())
                except Exception:
                    pass
        except Exception:
            pass

        # Auto-refresh (bounded): if user is asking for a report for a camera+day/class,
        # ingest the most recent sidecar events for that camera so we include *all* today's captures.
        try:
            if auto_refresh and isinstance(camera_name, str) and camera_name.strip() and hasattr(event_index_service, "ingest_recent_for_camera"):
                cam_ref = camera_name.strip()
                cam_id = None
                try:
                    root = Path(getattr(event_index_service, "project_root", Path.cwd()))
                    for p in [root / "data" / "cameras.json", root / "cameras.json"]:
                        if not p.exists():
                            continue
                        raw = json.loads(p.read_text(encoding="utf-8", errors="ignore"))
                        if isinstance(raw, list):
                            for item in raw:
                                if not isinstance(item, dict):
                                    continue
                                cid = str(item.get("id") or "").strip()
                                nm = str(item.get("name") or "").strip()
                                if nm and nm.lower() == cam_ref.lower() and cid:
                                    cam_id = cid
                                    break
                        if cam_id:
                            break
                except Exception:
                    cam_id = None
                if cam_id is None and "-" in cam_ref and len(cam_ref) >= 16:
                    cam_id = cam_ref

                if cam_id:
                    # Do NOT force detector runs here; we'll persist sidecar detections when present.
                    event_index_service.ingest_recent_for_camera(
                        str(cam_id),
                        # Cover a full day of Motion Watch bursts.
                        max_items=2500,
                        include_detections=False,
                        include_vision=False,
                    )
        except Exception:
            pass

        hits = event_index_service.search(
            query=query,
            camera_name=str(camera_name).strip() if isinstance(camera_name, str) and camera_name.strip() else None,
            trigger_type=str(trigger_type).strip() if isinstance(trigger_type, str) and trigger_type.strip() else None,
            shape_name=str(shape_name).strip() if isinstance(shape_name, str) and shape_name.strip() else None,
            dominant_color=str(dominant_color).strip() if isinstance(dominant_color, str) and str(dominant_color).strip() else None,
            detection_classes=[str(c).strip() for c in detection_classes if str(c).strip()],
            detection_color=str(detection_color).strip() if isinstance(detection_color, str) and str(detection_color).strip() else None,
            min_confidence=min_conf_f,
            min_area=min_area_f,
            start_ts=start_ts_i,
            end_ts=end_ts_i,
            limit=limit,
        )

        relaxed_note = None
        # If the user asked for a very specific color and we got 0 matches, still generate a useful report:
        # show "near matches" (same class/time/camera, any color) so the user can verify whether
        # cars were classified as gray/white/etc and then retag via overrides if needed.
        if (not hits) and isinstance(detection_color, str) and detection_color.strip() and detection_classes:
            try:
                relaxed = event_index_service.search(
                    query=query,
                    camera_name=str(camera_name).strip() if isinstance(camera_name, str) and camera_name.strip() else None,
                    trigger_type=str(trigger_type).strip() if isinstance(trigger_type, str) and trigger_type.strip() else None,
                    shape_name=str(shape_name).strip() if isinstance(shape_name, str) and shape_name.strip() else None,
                    dominant_color=str(dominant_color).strip() if isinstance(dominant_color, str) and str(dominant_color).strip() else None,
                    detection_classes=[str(c).strip() for c in detection_classes if str(c).strip()],
                    detection_color=None,
                    min_confidence=min_conf_f,
                    min_area=min_area_f,
                    start_ts=start_ts_i,
                    end_ts=end_ts_i,
                    limit=limit,
                )
                if relaxed:
                    hits = relaxed
                    relaxed_note = f"No exact '{detection_color}' matches; showing {', '.join([str(c) for c in detection_classes])} events with any color."
            except Exception:
                relaxed_note = None

        # If still empty and a class filter was specified, fall back to a camera/time timeline report
        # (no detection filters) so the user can still review what happened and retag if needed.
        if (not hits) and detection_classes:
            try:
                relaxed2 = event_index_service.search(
                    query=query,
                    camera_name=str(camera_name).strip() if isinstance(camera_name, str) and camera_name.strip() else None,
                    trigger_type=str(trigger_type).strip() if isinstance(trigger_type, str) and trigger_type.strip() else None,
                    shape_name=str(shape_name).strip() if isinstance(shape_name, str) and shape_name.strip() else None,
                    dominant_color=str(dominant_color).strip() if isinstance(dominant_color, str) and str(dominant_color).strip() else None,
                    detection_classes=[],
                    detection_color=None,
                    min_confidence=min_conf_f,
                    min_area=min_area_f,
                    start_ts=start_ts_i,
                    end_ts=end_ts_i,
                    limit=limit,
                )
                if relaxed2:
                    hits = relaxed2
                    relaxed_note = (
                        relaxed_note
                        or f"No exact matches for class filter ({', '.join([str(c) for c in detection_classes])}); showing camera timeline without detection filters."
                    )
            except Exception:
                pass

        # Oldest-first timeline for "report" readability
        hits_sorted = sorted(hits, key=lambda h: int(h.captured_ts or 0))

        # Create report file (persist under per-user data dir in frozen builds)
        try:
            from core.paths import get_data_dir
            reports_dir = get_data_dir() / "reports"
        except Exception:
            reports_dir = Path("data/reports")
        reports_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_id = uuid.uuid4().hex[:10]
        out_path = reports_dir / f"security_report_{stamp}_{report_id}.html"

        def _esc(s: Any) -> str:
            try:
                return (
                    str(s or "")
                    .replace("&", "&amp;")
                    .replace("<", "&lt;")
                    .replace(">", "&gt;")
                    .replace('"', "&quot;")
                )
            except Exception:
                return ""

        def _file_uri(p: str) -> str:
            try:
                return Path(p).resolve().as_uri()
            except Exception:
                # fallback: raw string
                return str(p)

        # Build a compact JSON payload and render a zoomable timeline in-browser.
        # Avoid per-event detection queries here (too slow for large days).
        events_payload: List[Dict[str, Any]] = []
        for h in hits_sorted:
            try:
                file_p = str(h.file_path)
                thumb_p = str(h.thumb_path) if getattr(h, "thumb_path", None) else ""
                eid = str(h.id)
                events_payload.append(
                    {
                        "event_id": eid,
                        "captured_ts": int(h.captured_ts or 0),
                        "captured_at": str(h.captured_at or ""),
                        "camera_name": str(h.camera_name or ""),
                        "caption": str(h.caption or ""),
                        "file_uri": _file_uri(file_p),
                        "api_file_url": f"/api/events/file?id={eid}",
                        "folder_uri": _file_uri(str(Path(file_p).parent)),
                        "thumb_uri": _file_uri(thumb_p) if thumb_p else _file_uri(file_p),
                        "api_thumb_url": f"/api/events/file?id={eid}&kind=thumb",
                        "dominant_color": str(h.dominant_color or ""),
                        "tags": [str(t) for t in (h.tags or [])],
                        "detection_classes": [str(c) for c in (h.detection_classes or [])],
                        "shape_name": str(h.shape_name or ""),
                        "media_type": str(getattr(h, "media_type", None) or "image"),
                    }
                )
            except Exception:
                continue

        header_bits = []
        if camera_name:
            header_bits.append(f"Camera: {_esc(camera_name)}")
        if start_ts_i or end_ts_i:
            header_bits.append(f"Window: {_esc(start_ts_i)} → {_esc(end_ts_i)}")
        if query:
            header_bits.append(f"Query: {_esc(query)}")
        if detection_color:
            header_bits.append(f"Detection color: {_esc(detection_color)}")
        if detection_classes:
            header_bits.append(f"Detection classes: {_esc(', '.join([str(c) for c in detection_classes]))}")
        if relaxed_note:
            header_bits.append(f"Note: {_esc(relaxed_note)}")

        # Serialize the payload once for embedding into the report HTML.
        # NOTE: This must exist before we do placeholder substitution below.
        events_json = json.dumps(events_payload, default=str)

        # Build the HTML template
        # We use a literal string and replace placeholders to avoid f-string escaping nightmares with JS/CSS
        html_template = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>{{TITLE}}</title>
  <style>
    :root {
      --bg: #0f172a;
      --panel: rgba(30, 41, 59, 0.85);
      --border: rgba(148, 163, 184, 0.15);
      --muted: #94a3b8;
      --text: #f1f5f9;
      --accent: #38bdf8;
      --accent-glow: rgba(56, 189, 248, 0.2);
      --warn: #fbbf24;
      --cardW: 100%;
      --sidebarW: 280px;
      --tileSize: 260px;
    }
    * { box-sizing: border-box; }
    body { 
      font-family: 'Inter', system-ui, -apple-system, sans-serif; 
      background: var(--bg); 
      color: var(--text); 
      margin: 0; 
      height: 100vh;
      overflow: hidden;
      display: flex;
    }
    
    /* Sidebar */
    .sidebar {
      width: var(--sidebarW);
      border-right: 1px solid var(--border);
      background: rgba(15, 23, 42, 0.98);
      display: flex;
      flex-direction: column;
      z-index: 50;
    }
    .sidebar-header { padding: 20px; border-bottom: 1px solid var(--border); }
    .sidebar-content { flex: 1; overflow-y: auto; padding: 12px; }
    .sidebar-label { font-size: 11px; text-transform: uppercase; letter-spacing: 0.1em; color: var(--muted); font-weight: 700; margin: 16px 12px 8px; }
    
    .filter-group { padding: 0 12px 12px; display: flex; flex-direction: column; gap: 6px; }
    .filter-select { background: rgba(30, 41, 59, 0.5); border: 1px solid var(--border); color: var(--text); padding: 8px; border-radius: 8px; font-size: 13px; outline: none; width: 100%; }
    
    .view-mode-toggle { display: flex; gap: 4px; background: rgba(0,0,0,0.2); padding: 4px; border-radius: 10px; margin: 0 12px 16px; border: 1px solid var(--border); }
    .mode-btn { flex: 1; border: none; background: transparent; color: var(--muted); padding: 6px; border-radius: 6px; font-size: 12px; font-weight: 600; cursor: pointer; transition: all 0.2s; }
    .mode-btn.active { background: var(--accent); color: #000; }

    .day-jump {
      display: block; padding: 8px 12px; margin-bottom: 2px; border-radius: 6px; color: var(--muted);
      text-decoration: none; font-size: 12px; transition: all 0.2s;
    }
    .day-jump:hover { background: rgba(56, 189, 248, 0.1); color: var(--accent); }

    /* Main Area */
    .main { flex: 1; display: flex; flex-direction: column; min-width: 0; background: #0b0f19; }
    
    .header-panel {
      padding: 16px 24px; background: var(--panel); backdrop-filter: blur(16px);
      border-bottom: 1px solid var(--border); z-index: 40;
    }
    .header-top { display: flex; justify-content: space-between; align-items: center; }
    h1 { font-size: 18px; margin: 0; font-weight: 800; }
    
    .pagination-controls { display: flex; align-items: center; gap: 12px; }
    .page-info { font-size: 12px; color: var(--muted); font-weight: 600; }
    
    .controls { display: flex; align-items: center; gap: 16px; margin-top: 12px; }
    .search-box { 
      flex: 1; display: flex; align-items: center; gap: 10px; background: rgba(15, 23, 42, 0.5); 
      padding: 6px 14px; border-radius: 10px; border: 1px solid var(--border);
    }
    .search-box input { background: transparent; border: none; color: var(--text); font-size: 13px; width: 100%; outline: none; }
    
    .btn { 
      background: rgba(56, 189, 248, 0.1); border: 1px solid var(--accent-glow); color: var(--accent);
      padding: 6px 12px; border-radius: 6px; cursor: pointer; font-size: 12px; font-weight: 600; transition: all 0.2s;
    }
    .btn:hover:not(:disabled) { background: var(--accent); color: #000; }
    .btn:disabled { opacity: 0.2; cursor: not-allowed; }

    /* Content Containers */
    .viewport { flex: 1; overflow-y: auto; position: relative; }
    /* "Wall" view is implemented by the list container as a responsive image grid. */
    .list-view { 
      display: grid; 
      grid-template-columns: repeat(auto-fill, minmax(var(--tileSize), 1fr));
      gap: 10px; 
      padding: 16px;
      max-width: none;
      margin: 0;
      width: 100%;
      align-content: start;
    }
    .timeline-view { height: 100%; width: 100%; display: none; flex-direction: column; }

    /* List Item Styles */
    .time-divider { padding: 20px 0 8px; color: var(--accent); font-size: 11px; font-weight: 800; text-transform: uppercase; letter-spacing: 0.1em; display: flex; align-items: center; gap: 10px; }
    .time-divider::after { content: ''; flex: 1; height: 1px; background: var(--border); }
    
    .gap-indicator { padding: 8px; text-align: center; border: 1px dashed var(--border); border-radius: 8px; margin: 4px 0; color: var(--muted); font-size: 11px; background: rgba(255,255,255,0.01); }

    .event-item {
      display: flex; flex-direction: column; gap: 0; background: var(--panel); border: 1px solid var(--border);
      border-radius: 12px; padding: 0; transition: all 0.2s; cursor: pointer; overflow: hidden;
    }
    .event-item:hover { transform: scale(1.01); border-color: var(--accent); background: rgba(30, 41, 59, 0.95); }
    
    .item-thumb { width: 100%; height: 180px; overflow: hidden; background: #000; flex-shrink: 0; position: relative; }
    .item-thumb img, .item-thumb video { width: 100%; height: 100%; object-fit: cover; }
    .play-overlay { position: absolute; top: 50%; left: 50%; transform: translate(-50%,-50%); font-size: 36px; opacity: 0.85; text-shadow: 0 2px 8px rgba(0,0,0,0.7); pointer-events: none; }
    
    .item-info { flex: 1; min-width: 0; display: flex; flex-direction: column; padding: 10px 10px 12px; }
    .item-header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 4px; }
    .item-time { font-size: 14px; font-weight: 700; color: var(--accent); }
    .item-camera { font-size: 11px; font-weight: 700; color: var(--warn); padding: 2px 6px; background: rgba(251, 191, 36, 0.1); border-radius: 4px; }
    .item-caption { font-size: 13px; color: var(--text); line-height: 1.35; margin-bottom: 8px; overflow: hidden; display: -webkit-box; -webkit-line-clamp: 2; -webkit-box-orient: vertical; }
    .tag-cloud { display: flex; flex-wrap: wrap; gap: 4px; }
    .tag { font-size: 10px; padding: 1px 6px; border-radius: 4px; background: rgba(255,255,255,0.05); border: 1px solid var(--border); color: var(--muted); }

    /* Timeline Explorer Styles */
    /* Scrub rail is horizontal; big preview is the floating card in the center. */
    .timeline-content { flex: 1; position: relative; overflow: auto; padding: 16px; display: flex; align-items: center; justify-content: center; }
    .timeline-rail { height: 90px; border-top: 1px solid var(--border); background: rgba(0,0,0,0.2); position: relative; overflow: hidden; cursor: crosshair; }
    .timeline-scrubber { position: absolute; top: 0; bottom: 0; width: 2px; background: var(--accent); z-index: 20; pointer-events: none; }
    .timeline-scrubber::after { content: ''; position: absolute; left: -4px; top: -4px; width: 10px; height: 10px; background: var(--accent); border-radius: 50%; box-shadow: 0 0 10px var(--accent); }
    
    .activity-marker { position: absolute; top: 12px; bottom: 12px; width: 4px; border-radius: 4px; opacity: 0.55; transition: opacity 0.2s, width 0.2s; }
    .activity-marker:hover { opacity: 1; width: 7px; z-index: 15; }
    
    .floating-card { 
      position: relative; left: auto; width: min(1100px, 100%); 
      background: var(--panel); border: 1px solid var(--accent); border-radius: 12px; padding: 12px; 
      box-shadow: 0 20px 50px rgba(0,0,0,0.5); z-index: 100; pointer-events: auto; display: none; 
      backdrop-filter: blur(10px);
    }
    .floating-card img { width: 100%; max-height: calc(100vh - 340px); object-fit: contain; border-radius: 8px; margin-bottom: 10px; background: #000; }

    .zoom-overlay { position: absolute; bottom: 20px; right: 30px; background: rgba(0,0,0,0.7); padding: 8px 16px; border-radius: 20px; font-size: 11px; color: var(--muted); pointer-events: none; z-index: 100; border: 1px solid var(--border); }
  </style>
</head>
<body>
  <aside class="sidebar">
    <div class="sidebar-header">
      <div style="font-size: 11px; text-transform: uppercase; letter-spacing: 0.1em; color: var(--muted); font-weight: 700;">Control Center</div>
    </div>
    <div class="sidebar-content">
      <div class="sidebar-label">View Mode</div>
      <div class="view-mode-toggle">
        <button class="mode-btn active" id="list-mode-btn">Wall</button>
        <button class="mode-btn" id="timeline-mode-btn">Scrub</button>
      </div>

      <div class="sidebar-label">Camera Filter</div>
      <div class="filter-group">
        <select id="camera-filter" class="filter-select">
          <option value="all">All Cameras</option>
        </select>
      </div>

      <div class="sidebar-label">Zone / Shape</div>
      <div class="filter-group">
        <select id="zone-filter" class="filter-select">
          <option value="all">All Zones</option>
        </select>
      </div>

      <div class="sidebar-label">Media Type</div>
      <div class="filter-group">
        <select id="media-filter" class="filter-select">
          <option value="all">All</option>
          <option value="image">Images</option>
          <option value="clip">Clips</option>
        </select>
      </div>

      <div class="sidebar-label">Object Class</div>
      <div class="filter-group">
        <select id="class-filter" class="filter-select"><option value="all">All Objects</option></select>
      </div>

      <div class="sidebar-label">Color</div>
      <div class="filter-group">
        <select id="color-filter" class="filter-select"><option value="all">All Colors</option></select>
      </div>

      <div class="sidebar-label">Organize</div>
      <div class="filter-group">
        <select id="sort-mode" class="filter-select">
          <option value="newest" selected>Newest first</option>
          <option value="oldest">Oldest first</option>
          <option value="camera">Camera (A&rarr;Z)</option>
          <option value="zone">Zone (A&rarr;Z)</option>
        </select>
      </div>
      
      <div class="sidebar-label">Items per page</div>
      <div class="filter-group">
        <select id="per-page" class="filter-select">
          <option value="5">5 items</option>
          <option value="10">10 items</option>
          <option value="25">25 items</option>
          <option value="50">50 items</option>
          <option value="5000" selected>All items</option>
        </select>
      </div>

      <div class="sidebar-label">Tile size</div>
      <div class="filter-group">
        <input id="tile-size" type="range" min="160" max="420" value="260" />
      </div>

      <div class="sidebar-label">Jump to Date</div>
      <div id="date-jumps" class="filter-group">
        <!-- Generated -->
      </div>
    </div>
  </aside>

  <main class="main">
    <div class="header-panel">
      <div class="header-top">
        <h1>{{TITLE}}</h1>
        <div class="pagination-controls" id="pag-controls">
          <button class="btn" id="prev-page" disabled>Prev</button>
          <span class="page-info" id="page-display">Page 1 / 1</span>
          <button class="btn" id="next-page" disabled>Next</button>
        </div>
      </div>

      <div class="controls">
        <div class="search-box">
          <input id="search-input" type="text" placeholder="Filter by text, tags, or objects..." />
        </div>
        <span id="total-stats" style="font-size: 12px; color: var(--muted); font-weight: 600;"></span>
      </div>
    </div>

    <div class="viewport" id="viewport">
      <div class="list-view" id="list-view">
        <!-- Items rendered here -->
      </div>
      
      <div class="timeline-view" id="timeline-view">
        <div class="timeline-content" id="timeline-content">
          <div class="floating-card" id="floating-card">
            <img id="floating-img" src="">
            <div id="floating-info"></div>
          </div>
        </div>
        <div class="timeline-rail" id="timeline-rail">
          <div class="timeline-scrubber" id="timeline-scrubber"></div>
          <!-- Markers rendered here -->
        </div>
        <div class="zoom-overlay"><b>Move</b> on bar to scrub &bull; <b>Click image</b> to open</div>
      </div>

      <div id="empty-state" style="display:none; padding: 100px; text-align: center; color: var(--muted);">
        No events found for current selection.
      </div>
    </div>
  </main>

  <script>
    const RAW_DATA = {{EVENTS_JSON}};
    
    // UI Elements
    const listView = document.getElementById('list-view');
    const timelineView = document.getElementById('timeline-view');
    const listBtn = document.getElementById('list-mode-btn');
    const timelineBtn = document.getElementById('timeline-mode-btn');
    const pagControls = document.getElementById('pag-controls');
    const searchInput = document.getElementById('search-input');
    const cameraFilter = document.getElementById('camera-filter');
    const classFilter = document.getElementById('class-filter');
    const colorFilter = document.getElementById('color-filter');
    const sortMode = document.getElementById('sort-mode');
    const perPageSelect = document.getElementById('per-page');
    const tileSize = document.getElementById('tile-size');
    const prevBtn = document.getElementById('prev-page');
    const nextBtn = document.getElementById('next-page');
    const pageDisplay = document.getElementById('page-display');
    const dateJumps = document.getElementById('date-jumps');
    const totalStats = document.getElementById('total-stats');
    const emptyState = document.getElementById('empty-state');
    const timelineRail = document.getElementById('timeline-rail');
    const timelineScrubber = document.getElementById('timeline-scrubber');
    const timelineContent = document.getElementById('timeline-content');
    const timelineAxis = document.getElementById('timeline-axis');
    const floatCard = document.getElementById('floating-card');
    const floatImg = document.getElementById('floating-img');
    const floatInfo = document.getElementById('floating-info');

    // Application State
    let viewMode = 'list';
    let currentPage = 1;
    let filteredEvents = [];
    let pxPerSec = 0.5;

    function pad2(n) { return String(n).padStart(2, '0'); }
    function tsToDate(ts) { return new Date(Number(ts) * 1000); }
    function isoDate(ts) {
      try { return tsToDate(ts).toISOString().slice(0, 10); } catch { return ''; }
    }
    function localTime(ts) {
      try {
        const d = tsToDate(ts);
        return `${pad2(d.getHours())}:${pad2(d.getMinutes())}:${pad2(d.getSeconds())}`;
      } catch { return ''; }
    }
    function localDateTime(ts) {
      const d = isoDate(ts);
      const t = localTime(ts);
      return d && t ? `${d} ${t}` : (d || t || '');
    }
    
    function normalize(e) {
      const ts = Number(e.captured_ts || 0);
      return {
        ...e,
        captured_ts: ts,
        file_uri: e.api_file_url || e.file_uri || '',
        thumb_uri: e.api_thumb_url || e.thumb_uri || e.api_file_url || e.file_uri || '',
        _date: isoDate(ts),
        _time: localTime(ts),
        _dt: localDateTime(ts),
        _searchKey: (
          String(e.caption || '') + ' ' +
          (e.tags || []).join(' ') + ' ' +
          (e.detection_classes || []).join(' ') + ' ' +
          String(e.camera_name || '') + ' ' +
          String(e.shape_name || '') + ' ' +
          String(e.media_type || 'image') + ' ' +
          String(e.dominant_color || '') + ' ' +
          isoDate(ts) + ' ' + localDateTime(ts)
        ).toLowerCase()
      };
    }

    const ALL_EVENTS = (Array.isArray(RAW_DATA) ? RAW_DATA : []).map(normalize);
    const CAMERAS = [...new Set(ALL_EVENTS.map(e => e.camera_name))].sort();
    const ZONES = [...new Set(ALL_EVENTS.map(e => e.shape_name).filter(z => z))].sort();
    const CLASSES = [...new Set(ALL_EVENTS.flatMap(e => e.detection_classes || []).filter(c => c))].sort();
    const COLORS = [...new Set(ALL_EVENTS.map(e => e.dominant_color).filter(c => c))].sort();
    
    const zoneFilter = document.getElementById('zone-filter');
    const mediaFilter = document.getElementById('media-filter');

    function _populateSelect(sel, items) {
      items.forEach(v => { const o = document.createElement('option'); o.value = v; o.textContent = v; sel.appendChild(o); });
    }
    _populateSelect(cameraFilter, CAMERAS);
    _populateSelect(zoneFilter, ZONES);
    _populateSelect(classFilter, CLASSES);
    _populateSelect(colorFilter, COLORS);

    function applyTileSize() {
      const v = Math.max(160, Math.min(420, parseInt(tileSize.value || '260')));
      document.documentElement.style.setProperty('--tileSize', v + 'px');
    }

    function sortEvents(arr) {
      const mode = (sortMode && sortMode.value) ? sortMode.value : 'newest';
      const copy = arr.slice();
      if (mode === 'camera') {
        copy.sort((a, b) => {
          const ca = String(a.camera_name || '').toLowerCase();
          const cb = String(b.camera_name || '').toLowerCase();
          if (ca < cb) return -1;
          if (ca > cb) return 1;
          return (b.captured_ts || 0) - (a.captured_ts || 0);
        });
        return copy;
      }
      if (mode === 'zone') {
        copy.sort((a, b) => {
          const za = String(a.shape_name || '').toLowerCase();
          const zb = String(b.shape_name || '').toLowerCase();
          if (za < zb) return -1;
          if (za > zb) return 1;
          return (b.captured_ts || 0) - (a.captured_ts || 0);
        });
        return copy;
      }
      copy.sort((a, b) => (a.captured_ts || 0) - (b.captured_ts || 0));
      if (mode === 'newest') copy.reverse();
      return copy;
    }

    function update() {
      const q = searchInput.value.toLowerCase().trim();
      const cam = cameraFilter.value;
      const zone = zoneFilter ? zoneFilter.value : 'all';
      const media = mediaFilter ? mediaFilter.value : 'all';
      const cls = classFilter ? classFilter.value : 'all';
      const col = colorFilter ? colorFilter.value : 'all';
      
      filteredEvents = sortEvents(ALL_EVENTS.filter(e => {
        const matchesSearch = !q || e._searchKey.includes(q);
        const matchesCam = cam === 'all' || e.camera_name === cam;
        const matchesZone = zone === 'all' || (e.shape_name || '') === zone;
        const matchesMedia = media === 'all' || (e.media_type || 'image') === media;
        const matchesClass = cls === 'all' || (e.detection_classes || []).includes(cls);
        const matchesColor = col === 'all' || (e.dominant_color || '') === col;
        return matchesSearch && matchesCam && matchesZone && matchesMedia && matchesClass && matchesColor;
      }));

      if (viewMode === 'list') {
        // "Wall": show all matches on one page (no pagination).
        renderList(filteredEvents);
        pagControls.style.display = 'none';
      } else {
        pagControls.style.display = 'none';
        renderTimeline();
      }

      totalStats.textContent = `${filteredEvents.length} matches found`;
      emptyState.style.display = filteredEvents.length === 0 ? 'block' : 'none';
      generateDateJumps();
    }

    function renderList(items) {
      listView.innerHTML = '';
      items.forEach(e => {
        const item = document.createElement('div');
        item.className = 'event-item';
        item.onclick = () => window.open(e.file_uri, '_blank');
        const isClip = (e.media_type === 'clip');
        const zoneBadge = e.shape_name ? `<span class="tag" style="background:rgba(56,189,248,0.15);color:var(--accent);border-color:var(--accent);">${e.shape_name}</span>` : '';
        const mediaBadge = isClip ? '<span class="tag" style="background:rgba(251,191,36,0.15);color:var(--warn);border-color:var(--warn);">clip</span>' : '';
        const colorBadge = e.dominant_color ? `<span class="tag" style="background:rgba(148,163,184,0.12);color:var(--muted);border-color:var(--border);">${e.dominant_color}</span>` : '';
        let thumbHtml;
        if (isClip) {
          const previewUri = e.file_uri.replace(/[.]mp4$/i, '.preview.mp4');
          thumbHtml = `<div class="item-thumb"><video src="${previewUri}" autoplay muted loop playsinline poster="${e.thumb_uri}" preload="metadata"></video><div class="play-overlay">&#9654;</div></div>`;
        } else {
          thumbHtml = `<div class="item-thumb"><img src="${e.thumb_uri}" loading="lazy"></div>`;
        }
        item.innerHTML = `
          ${thumbHtml}
          <div class="item-info">
            <div class="item-header">
              <div class="item-time">${e._time || ''}</div>
              <div class="item-camera">${e.camera_name}</div>
            </div>
            <div class="item-caption">${e.caption || 'Event detected'}<span style="color: var(--muted); font-weight: 800;"> &middot; ${e._date || ''}</span></div>
            <div class="tag-cloud">
              ${zoneBadge}${mediaBadge}${colorBadge}${(e.detection_classes || []).map(t => `<span class="tag">${t}</span>`).join('')}
            </div>
          </div>
        `;
        listView.appendChild(item);
      });
    }

    function renderTimeline() {
      // Clear markers
      timelineRail.querySelectorAll('.activity-marker').forEach(m => m.remove());
      
      if (!filteredEvents.length) return;

      // Make the preview always visible in scrub mode
      floatCard.style.display = 'block';
      
      // Compute timeline bounds (oldest -> newest)
      const byTime = filteredEvents.slice().sort((a, b) => (a.captured_ts || 0) - (b.captured_ts || 0));
      const start = byTime[0].captured_ts;
      const end = byTime[byTime.length - 1].captured_ts;
      const span = (end - start) || 1;
      
      // Rail Markers (Summary)
      const railW = timelineRail.clientWidth;
      byTime.forEach(e => {
        const m = document.createElement('div');
        m.className = 'activity-marker';
        const left = ((e.captured_ts - start) / span) * railW;
        m.style.left = Math.max(0, Math.min(railW - 1, left)) + 'px';
        m.style.background = '#38bdf8';
        timelineRail.appendChild(m);
      });

      // Initialize selection to the newest event
      const newest = byTime[byTime.length - 1];
      showScrubSelection(newest);
    }

    function showScrubSelection(e) {
      if (!e) return;
      // Big, visible image (use full frame when possible)
      floatImg.src = e.file_uri || e.thumb_uri;
      floatImg.style.cursor = 'pointer';
      floatImg.onclick = () => window.open(e.file_uri, '_blank');
      floatInfo.innerHTML = `<div style="font-weight:900; color: var(--warn);">${e.camera_name || ''}</div><div style="color: var(--muted); font-weight:800;">${e._dt || ''}</div>`;
    }

    // Interaction logic
    function setMode(m) {
      viewMode = m;
      listBtn.classList.toggle('active', m === 'list');
      timelineBtn.classList.toggle('active', m === 'timeline');
      listView.style.display = m === 'list' ? 'grid' : 'none';
      timelineView.style.display = m === 'timeline' ? 'flex' : 'none';
      floatCard.style.display = 'none';
      update();
    }

    listBtn.onclick = () => setMode('list');
    timelineBtn.onclick = () => setMode('timeline');

    timelineRail.onmousemove = (e) => {
      const rect = timelineRail.getBoundingClientRect();
      const x = e.clientX - rect.left;
      timelineScrubber.style.left = x + 'px';
      
      const byTime = filteredEvents.slice().sort((a, b) => (a.captured_ts || 0) - (b.captured_ts || 0));
      const start = byTime[0].captured_ts;
      const end = byTime[byTime.length - 1].captured_ts;
      const targetTs = start + (x / rect.width) * (end - start);
      
      // Find closest event
      const closest = byTime.reduce((prev, curr) => 
        Math.abs(curr.captured_ts - targetTs) < Math.abs(prev.captured_ts - targetTs) ? curr : prev
      );
      
      if (closest) {
        showScrubSelection(closest);
      }
    };
    timelineRail.onmouseleave = () => { /* keep selection visible */ };
    timelineRail.onclick = (e) => {
      const rect = timelineRail.getBoundingClientRect();
      const x = e.clientX - rect.left;
      timelineScrubber.style.left = x + 'px';
    };

    // Tile size zoom (Ctrl+Wheel) while in Wall mode
    window.addEventListener('wheel', (e) => {
      if (viewMode === 'list' && e.ctrlKey) {
        e.preventDefault();
        const cur = parseInt(tileSize.value || '260');
        const next = Math.max(160, Math.min(420, cur + (e.deltaY < 0 ? 20 : -20)));
        tileSize.value = String(next);
        applyTileSize();
      }
    }, { passive: false });

    function generateDateJumps() {
      dateJumps.innerHTML = '';
      const days = new Map();
      ALL_EVENTS.forEach(e => {
        const d = e._date || '';
        if (d && !days.has(d)) days.set(d, e.captured_ts);
      });
      [...days.entries()].slice(0, 10).forEach(([date, ts]) => {
        const a = document.createElement('a');
        a.className = 'day-jump';
        a.href = '#';
        a.textContent = date;
        a.onclick = (ev) => {
          ev.preventDefault();
          cameraFilter.value = 'all';
          searchInput.value = date;
          currentPage = 1;
          update();
        };
        dateJumps.appendChild(a);
      });
    }

    searchInput.oninput = () => { currentPage = 1; update(); };
    cameraFilter.onchange = () => { currentPage = 1; update(); };
    if (zoneFilter) zoneFilter.onchange = () => { currentPage = 1; update(); };
    if (mediaFilter) mediaFilter.onchange = () => { currentPage = 1; update(); };
    if (classFilter) classFilter.onchange = () => { currentPage = 1; update(); };
    if (colorFilter) colorFilter.onchange = () => { currentPage = 1; update(); };
    if (sortMode) sortMode.onchange = () => { currentPage = 1; update(); };
    perPageSelect.onchange = () => { currentPage = 1; update(); };
    if (tileSize) tileSize.oninput = () => applyTileSize();
    prevBtn.onclick = () => { currentPage--; update(); };
    nextBtn.onclick = () => { currentPage++; update(); };

    applyTileSize();
    update();
  </script>
</body>
</html>"""

        # Replace placeholders with Python values
        html = (
            html_template.replace("{{TITLE}}", _esc(title))
            .replace("{{META}}", _esc(' | '.join(header_bits)))
            .replace("{{COUNT}}", str(len(hits_sorted)))
            .replace("{{EVENTS_JSON}}", events_json)
        )

        out_path.write_text(html, encoding="utf-8", errors="ignore")

        return jsonify(
            {
                'success': True,
                'data': {
                    'report_path': str(out_path),
                    'report_url': out_path.resolve().as_uri(),
                    'events': len(hits_sorted),
                },
            }
        )
    except Exception:
        # Include traceback in logs for debugging
        logger.exception("Events report error")
        return jsonify({'success': False, 'message': 'Failed to generate report'}), 500


@api_bp.route('/events/live-report', methods=['GET'])
def events_live_report():
    """
    Serve a live security report dashboard that connects to Socket.IO for
    real-time event streaming. The page fetches initial data from /api/events/search
    and then subscribes to ``new_capture`` events on the ``/realtime`` namespace.
    """
    try:
        live_html = _build_live_report_html()
        return Response(live_html, mimetype='text/html')
    except Exception:
        logger.exception("Live report error")
        return jsonify({'success': False, 'message': 'Failed to generate live report'}), 500


def _build_live_report_html() -> str:
    """Build the self-contained live security report HTML page."""
    return """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Live Security Report</title>
  <style>
    :root {
      --bg: #0f172a;
      --panel: rgba(30, 41, 59, 0.85);
      --border: rgba(148, 163, 184, 0.15);
      --muted: #94a3b8;
      --text: #f1f5f9;
      --accent: #38bdf8;
      --accent-glow: rgba(56, 189, 248, 0.2);
      --warn: #fbbf24;
      --success: #34d399;
      --tileSize: 260px;
      --sidebarW: 280px;
    }
    * { box-sizing: border-box; }
    body {
      font-family: 'Inter', system-ui, -apple-system, sans-serif;
      background: var(--bg); color: var(--text);
      margin: 0; height: 100vh; overflow: hidden; display: flex;
    }

    .sidebar {
      width: var(--sidebarW);
      border-right: 1px solid var(--border);
      background: rgba(15, 23, 42, 0.98);
      display: flex; flex-direction: column; z-index: 50;
    }
    .sidebar-header { padding: 20px; border-bottom: 1px solid var(--border); }
    .sidebar-content { flex: 1; overflow-y: auto; padding: 12px; }
    .sidebar-label { font-size: 11px; text-transform: uppercase; letter-spacing: 0.1em; color: var(--muted); font-weight: 700; margin: 16px 12px 8px; }

    .filter-group { padding: 0 12px 12px; display: flex; flex-direction: column; gap: 6px; }
    .filter-select { background: rgba(30, 41, 59, 0.5); border: 1px solid var(--border); color: var(--text); padding: 8px; border-radius: 8px; font-size: 13px; outline: none; width: 100%; }

    .view-mode-toggle { display: flex; gap: 4px; background: rgba(0,0,0,0.2); padding: 4px; border-radius: 10px; margin: 0 12px 16px; border: 1px solid var(--border); }
    .mode-btn { flex: 1; border: none; background: transparent; color: var(--muted); padding: 6px; border-radius: 6px; font-size: 12px; font-weight: 600; cursor: pointer; transition: all 0.2s; }
    .mode-btn.active { background: var(--accent); color: #000; }

    .live-indicator {
      display: flex; align-items: center; gap: 8px; padding: 8px 12px; margin: 0 12px 12px;
      background: rgba(52, 211, 153, 0.08); border: 1px solid rgba(52, 211, 153, 0.3);
      border-radius: 8px; font-size: 12px; font-weight: 600; color: var(--success);
    }
    .live-dot {
      width: 8px; height: 8px; border-radius: 50%; background: var(--success);
      animation: pulse-dot 2s ease-in-out infinite;
    }
    @keyframes pulse-dot { 0%, 100% { opacity: 1; } 50% { opacity: 0.3; } }

    .day-jump {
      display: block; padding: 8px 12px; margin-bottom: 2px; border-radius: 6px; color: var(--muted);
      text-decoration: none; font-size: 12px; transition: all 0.2s; cursor: pointer;
    }
    .day-jump:hover { background: rgba(56, 189, 248, 0.1); color: var(--accent); }

    .main { flex: 1; display: flex; flex-direction: column; min-width: 0; background: #0b0f19; }

    .header-panel {
      padding: 16px 24px; background: var(--panel); backdrop-filter: blur(16px);
      border-bottom: 1px solid var(--border); z-index: 40;
    }
    .header-top { display: flex; justify-content: space-between; align-items: center; }
    h1 { font-size: 18px; margin: 0; font-weight: 800; }

    .controls { display: flex; align-items: center; gap: 16px; margin-top: 12px; }
    .search-box {
      flex: 1; display: flex; align-items: center; gap: 10px; background: rgba(15, 23, 42, 0.5);
      padding: 6px 14px; border-radius: 10px; border: 1px solid var(--border);
    }
    .search-box input { background: transparent; border: none; color: var(--text); font-size: 13px; width: 100%; outline: none; }

    .btn {
      background: rgba(56, 189, 248, 0.1); border: 1px solid var(--accent-glow); color: var(--accent);
      padding: 6px 12px; border-radius: 6px; cursor: pointer; font-size: 12px; font-weight: 600; transition: all 0.2s;
    }
    .btn:hover:not(:disabled) { background: var(--accent); color: #000; }
    .btn:disabled { opacity: 0.2; cursor: not-allowed; }

    .viewport { flex: 1; overflow-y: auto; position: relative; }
    .list-view {
      display: grid;
      grid-template-columns: repeat(auto-fill, minmax(var(--tileSize), 1fr));
      gap: 10px; padding: 16px; max-width: none; margin: 0; width: 100%; align-content: start;
    }
    .timeline-view { height: 100%; width: 100%; display: none; flex-direction: column; }

    .event-item {
      display: flex; flex-direction: column; gap: 0; background: var(--panel); border: 1px solid var(--border);
      border-radius: 12px; padding: 0; transition: all 0.2s; cursor: pointer; overflow: hidden;
    }
    .event-item:hover { transform: scale(1.01); border-color: var(--accent); background: rgba(30, 41, 59, 0.95); }
    .event-item.new-event { animation: flash-in 1.5s ease-out; }
    @keyframes flash-in { 0% { border-color: var(--success); box-shadow: 0 0 20px rgba(52, 211, 153, 0.3); } 100% { border-color: var(--border); box-shadow: none; } }

    .item-thumb { width: 100%; height: 180px; overflow: hidden; background: #000; flex-shrink: 0; position: relative; }
    .item-thumb img, .item-thumb video { width: 100%; height: 100%; object-fit: cover; }
    .play-overlay { position: absolute; top: 50%; left: 50%; transform: translate(-50%,-50%); font-size: 36px; opacity: 0.85; text-shadow: 0 2px 8px rgba(0,0,0,0.7); pointer-events: none; }

    .item-info { flex: 1; min-width: 0; display: flex; flex-direction: column; padding: 10px 10px 12px; }
    .item-header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 4px; }
    .item-time { font-size: 14px; font-weight: 700; color: var(--accent); }
    .item-camera { font-size: 11px; font-weight: 700; color: var(--warn); padding: 2px 6px; background: rgba(251, 191, 36, 0.1); border-radius: 4px; }
    .item-caption { font-size: 13px; color: var(--text); line-height: 1.35; margin-bottom: 8px; overflow: hidden; display: -webkit-box; -webkit-line-clamp: 2; -webkit-box-orient: vertical; }
    .tag-cloud { display: flex; flex-wrap: wrap; gap: 4px; }
    .tag { font-size: 10px; padding: 1px 6px; border-radius: 4px; background: rgba(255,255,255,0.05); border: 1px solid var(--border); color: var(--muted); }
    .tag.zone-tag { background: rgba(56,189,248,0.15); color: var(--accent); border-color: var(--accent); }
    .tag.clip-tag { background: rgba(251,191,36,0.15); color: var(--warn); border-color: var(--warn); }
    .extract-clip-btn { font-size: 10px; padding: 1px 8px; border-radius: 4px; background: rgba(16,185,129,0.15); border: 1px solid rgba(16,185,129,0.4); color: #10b981; cursor: pointer; margin-left: auto; }
    .extract-clip-btn:hover { background: rgba(16,185,129,0.3); }

    .timeline-content { flex: 1; position: relative; overflow: auto; padding: 16px; display: flex; align-items: center; justify-content: center; }
    .timeline-rail { height: 90px; border-top: 1px solid var(--border); background: rgba(0,0,0,0.2); position: relative; overflow: hidden; cursor: crosshair; }
    .timeline-scrubber { position: absolute; top: 0; bottom: 0; width: 2px; background: var(--accent); z-index: 20; pointer-events: none; }
    .timeline-scrubber::after { content: ''; position: absolute; left: -4px; top: -4px; width: 10px; height: 10px; background: var(--accent); border-radius: 50%; box-shadow: 0 0 10px var(--accent); }

    .activity-marker { position: absolute; top: 12px; bottom: 12px; width: 4px; border-radius: 4px; opacity: 0.55; transition: opacity 0.2s, width 0.2s; }
    .activity-marker:hover { opacity: 1; width: 7px; z-index: 15; }

    .floating-card {
      position: relative; left: auto; width: min(1100px, 100%);
      background: var(--panel); border: 1px solid var(--accent); border-radius: 12px; padding: 12px;
      box-shadow: 0 20px 50px rgba(0,0,0,0.5); z-index: 100; pointer-events: auto; display: none;
      backdrop-filter: blur(10px);
    }
    .floating-card img { width: 100%; max-height: calc(100vh - 340px); object-fit: contain; border-radius: 8px; margin-bottom: 10px; background: #000; }

    .zoom-overlay { position: absolute; bottom: 20px; right: 30px; background: rgba(0,0,0,0.7); padding: 8px 16px; border-radius: 20px; font-size: 11px; color: var(--muted); pointer-events: none; z-index: 100; border: 1px solid var(--border); }

    #loading-state { display: none; padding: 100px; text-align: center; color: var(--muted); }
    #empty-state { display: none; padding: 100px; text-align: center; color: var(--muted); }
  </style>
</head>
<body>
  <aside class="sidebar">
    <div class="sidebar-header">
      <div style="font-size: 11px; text-transform: uppercase; letter-spacing: 0.1em; color: var(--muted); font-weight: 700;">Live Control Center</div>
    </div>
    <div class="sidebar-content">
      <div class="live-indicator"><div class="live-dot"></div><span id="live-status">Connecting...</span></div>

      <div class="sidebar-label">View Mode</div>
      <div class="view-mode-toggle">
        <button class="mode-btn active" id="list-mode-btn">Wall</button>
        <button class="mode-btn" id="timeline-mode-btn">Scrub</button>
      </div>

      <div class="sidebar-label">Camera Filter</div>
      <div class="filter-group">
        <select id="camera-filter" class="filter-select"><option value="all">All Cameras</option></select>
      </div>

      <div class="sidebar-label">Zone / Shape</div>
      <div class="filter-group">
        <select id="zone-filter" class="filter-select"><option value="all">All Zones</option></select>
      </div>

      <div class="sidebar-label">Media Type</div>
      <div class="filter-group">
        <select id="media-filter" class="filter-select">
          <option value="all">All</option>
          <option value="image">Images</option>
          <option value="clip">Clips</option>
        </select>
      </div>

      <div class="sidebar-label">Object Class</div>
      <div class="filter-group">
        <select id="class-filter" class="filter-select"><option value="all">All Objects</option></select>
      </div>

      <div class="sidebar-label">Color</div>
      <div class="filter-group">
        <select id="color-filter" class="filter-select"><option value="all">All Colors</option></select>
      </div>

      <div class="sidebar-label">Time Range</div>
      <div class="filter-group">
        <input id="start-time" type="datetime-local" class="filter-select" title="Start time" />
        <input id="end-time" type="datetime-local" class="filter-select" title="End time" style="margin-top:4px;" />
        <button class="btn" id="time-filter-btn" style="width:100%;margin-top:4px;">Apply Time Range</button>
      </div>

      <div class="sidebar-label">Organize</div>
      <div class="filter-group">
        <select id="sort-mode" class="filter-select">
          <option value="newest" selected>Newest first</option>
          <option value="oldest">Oldest first</option>
          <option value="camera">Camera (A-Z)</option>
          <option value="zone">Zone (A-Z)</option>
        </select>
      </div>

      <div class="sidebar-label">Tile size</div>
      <div class="filter-group">
        <input id="tile-size" type="range" min="160" max="420" value="260" />
      </div>

      <div class="sidebar-label">Jump to Date</div>
      <div id="date-jumps" class="filter-group"></div>

      <div class="sidebar-label">Fetch Limit</div>
      <div class="filter-group">
        <select id="fetch-limit" class="filter-select">
          <option value="50">Last 50</option>
          <option value="200" selected>Last 200</option>
          <option value="500">Last 500</option>
          <option value="2000">Last 2000</option>
        </select>
      </div>
      <div class="filter-group">
        <button class="btn" id="refresh-btn" style="width:100%;">Refresh Now</button>
      </div>

      <div class="sidebar-label">Post-Processing</div>
      <div class="filter-group">
        <button class="btn" id="enrich-btn" style="width:100%;">Enrich Unclassified</button>
        <div id="enrich-status" style="font-size:11px;color:var(--muted);margin-top:4px;display:none;"></div>
      </div>
    </div>
  </aside>

  <main class="main">
    <div class="header-panel">
      <div class="header-top">
        <h1>Live Security Report</h1>
        <span id="total-stats" style="font-size: 12px; color: var(--muted); font-weight: 600;"></span>
      </div>
      <div class="controls">
        <div class="search-box">
          <input id="search-input" type="text" placeholder="Filter by text, tags, zone, or objects..." />
        </div>
      </div>
    </div>

    <div class="viewport" id="viewport">
      <div class="list-view" id="list-view"></div>

      <div class="timeline-view" id="timeline-view">
        <div class="timeline-content" id="timeline-content">
          <div class="floating-card" id="floating-card">
            <img id="floating-img" src="">
            <video id="floating-video" style="display:none;width:100%;max-height:60vh;background:#000;border-radius:8px;" controls></video>
            <div id="floating-info"></div>
            <div id="playback-controls" style="display:none;padding:6px 0;display:flex;gap:6px;align-items:center;justify-content:center;">
              <button onclick="seekPlayback(-30)" style="font-size:12px;padding:2px 8px;border-radius:4px;border:1px solid var(--border);background:var(--panel);color:var(--text);cursor:pointer;">-30s</button>
              <button onclick="togglePlayback()" id="play-pause-btn" style="font-size:12px;padding:2px 12px;border-radius:4px;border:1px solid var(--accent);background:rgba(56,189,248,0.15);color:var(--accent);cursor:pointer;">Play</button>
              <button onclick="seekPlayback(30)" style="font-size:12px;padding:2px 8px;border-radius:4px;border:1px solid var(--border);background:var(--panel);color:var(--text);cursor:pointer;">+30s</button>
              <select id="speed-select" onchange="setPlaybackSpeed(this.value)" style="font-size:11px;padding:2px 4px;border-radius:4px;border:1px solid var(--border);background:var(--panel);color:var(--text);">
                <option value="0.5">0.5x</option><option value="1" selected>1x</option><option value="2">2x</option><option value="4">4x</option>
              </select>
            </div>
          </div>
        </div>
        <div class="timeline-rail" id="timeline-rail">
          <div class="timeline-scrubber" id="timeline-scrubber"></div>
        </div>
        <div class="zoom-overlay"><b>Move</b> on bar to scrub &bull; <b>Click</b> to load recording &bull; <b>Scroll</b> on marker for video</div>
      </div>

      <div id="loading-state" style="display:none;">Loading events...</div>
      <div id="empty-state" style="display:none;">No events found for current selection.</div>
    </div>
  </main>

  <script src="https://cdn.socket.io/4.7.5/socket.io.min.js"></script>
  <script>
    // DOM refs
    const listView = document.getElementById('list-view');
    const timelineView = document.getElementById('timeline-view');
    const listBtn = document.getElementById('list-mode-btn');
    const timelineBtn = document.getElementById('timeline-mode-btn');
    const searchInput = document.getElementById('search-input');
    const cameraFilter = document.getElementById('camera-filter');
    const zoneFilter = document.getElementById('zone-filter');
    const mediaFilter = document.getElementById('media-filter');
    const classFilter = document.getElementById('class-filter');
    const colorFilter = document.getElementById('color-filter');
    const startTimeInput = document.getElementById('start-time');
    const endTimeInput = document.getElementById('end-time');
    const timeFilterBtn = document.getElementById('time-filter-btn');
    const sortMode = document.getElementById('sort-mode');
    const tileSize = document.getElementById('tile-size');
    const dateJumps = document.getElementById('date-jumps');
    const totalStats = document.getElementById('total-stats');
    const emptyState = document.getElementById('empty-state');
    const loadingState = document.getElementById('loading-state');
    const timelineRail = document.getElementById('timeline-rail');
    const timelineScrubber = document.getElementById('timeline-scrubber');
    const timelineContent = document.getElementById('timeline-content');
    const floatCard = document.getElementById('floating-card');
    const floatImg = document.getElementById('floating-img');
    const floatInfo = document.getElementById('floating-info');
    const liveStatus = document.getElementById('live-status');
    const fetchLimit = document.getElementById('fetch-limit');
    const refreshBtn = document.getElementById('refresh-btn');
    const enrichBtn = document.getElementById('enrich-btn');
    const enrichStatus = document.getElementById('enrich-status');

    const MAX_EVENTS = 2000;
    let viewMode = 'list';
    let ALL_EVENTS = [];
    let filteredEvents = [];
    let newEventIds = new Set();

    function pad2(n) { return String(n).padStart(2, '0'); }
    function tsToDate(ts) { return new Date(Number(ts) * 1000); }
    function isoDate(ts) { try { return tsToDate(ts).toISOString().slice(0, 10); } catch { return ''; } }
    function localTime(ts) {
      try { const d = tsToDate(ts); return pad2(d.getHours()) + ':' + pad2(d.getMinutes()) + ':' + pad2(d.getSeconds()); } catch { return ''; }
    }
    function localDateTime(ts) { const d = isoDate(ts); const t = localTime(ts); return d && t ? d + ' ' + t : (d || t || ''); }

    function normalize(e) {
      const ts = Number(e.captured_ts || 0);
      return {
        ...e,
        captured_ts: ts,
        dominant_color: e.dominant_color || '',
        detection_classes: e.detection_classes || [],
        _date: isoDate(ts),
        _time: localTime(ts),
        _dt: localDateTime(ts),
        _searchKey: (
          String(e.caption || '') + ' ' + (e.tags || []).join(' ') + ' ' +
          (e.detection_classes || []).join(' ') + ' ' + String(e.camera_name || '') + ' ' +
          String(e.shape_name || '') + ' ' + String(e.media_type || 'image') + ' ' +
          String(e.dominant_color || '') + ' ' +
          isoDate(ts) + ' ' + localDateTime(ts)
        ).toLowerCase()
      };
    }

    function rebuildFilters() {
      const cameras = [...new Set(ALL_EVENTS.map(e => e.camera_name).filter(c => c))].sort();
      const zones = [...new Set(ALL_EVENTS.map(e => e.shape_name).filter(z => z))].sort();
      const classes = [...new Set(ALL_EVENTS.flatMap(e => e.detection_classes || []).filter(c => c))].sort();
      const colors = [...new Set(ALL_EVENTS.map(e => e.dominant_color).filter(c => c))].sort();

      function repopulate(sel, items, allLabel, preserveVal) {
        const prev = sel.value;
        sel.innerHTML = '<option value="all">' + allLabel + '</option>';
        items.forEach(v => { const o = document.createElement('option'); o.value = v; o.textContent = v; sel.appendChild(o); });
        if (items.includes(prev) || (preserveVal && prev !== 'all')) sel.value = prev;
      }
      repopulate(cameraFilter, cameras, 'All Cameras', true);
      repopulate(zoneFilter, zones, 'All Zones', true);
      repopulate(classFilter, classes, 'All Objects', true);
      repopulate(colorFilter, colors, 'All Colors', true);
    }

    function sortEvents(arr) {
      const mode = sortMode.value || 'newest';
      const copy = arr.slice();
      if (mode === 'camera') {
        copy.sort((a, b) => { const x = String(a.camera_name||'').localeCompare(String(b.camera_name||'')); return x || (b.captured_ts - a.captured_ts); });
        return copy;
      }
      if (mode === 'zone') {
        copy.sort((a, b) => { const x = String(a.shape_name||'').localeCompare(String(b.shape_name||'')); return x || (b.captured_ts - a.captured_ts); });
        return copy;
      }
      copy.sort((a, b) => (a.captured_ts || 0) - (b.captured_ts || 0));
      if (mode === 'newest') copy.reverse();
      return copy;
    }

    function update() {
      const q = searchInput.value.toLowerCase().trim();
      const cam = cameraFilter.value;
      const zone = zoneFilter.value;
      const media = mediaFilter.value;
      const cls = classFilter.value;
      const col = colorFilter.value;

      filteredEvents = sortEvents(ALL_EVENTS.filter(e => {
        const matchQ = !q || e._searchKey.includes(q);
        const matchCam = cam === 'all' || e.camera_name === cam;
        const matchZone = zone === 'all' || (e.shape_name || '') === zone;
        const matchMedia = media === 'all' || (e.media_type || 'image') === media;
        const matchClass = cls === 'all' || (e.detection_classes || []).includes(cls);
        const matchColor = col === 'all' || (e.dominant_color || '') === col;
        return matchQ && matchCam && matchZone && matchMedia && matchClass && matchColor;
      }));

      if (viewMode === 'list') {
        renderList(filteredEvents);
      } else {
        renderTimeline();
      }
      totalStats.textContent = filteredEvents.length + ' matches / ' + ALL_EVENTS.length + ' total';
      emptyState.style.display = filteredEvents.length === 0 && ALL_EVENTS.length > 0 ? 'block' : 'none';
      generateDateJumps();
    }

    function renderList(items) {
      listView.innerHTML = '';
      items.forEach(e => {
        const item = document.createElement('div');
        item.className = 'event-item' + (newEventIds.has(e.event_id) ? ' new-event' : '');
        item.onclick = () => window.open(e.file_uri, '_blank');
        const isClip = e.media_type === 'clip';
        const zoneBadge = e.shape_name ? '<span class="tag zone-tag">' + e.shape_name + '</span>' : '';
        const clipBadge = isClip ? '<span class="tag clip-tag">clip</span>' : '';
        const colorBadge = e.dominant_color ? '<span class="tag" style="background:rgba(148,163,184,0.12);color:var(--muted);border-color:var(--border);">' + e.dominant_color + '</span>' : '';
        let thumbHtml;
        if (isClip) {
          const previewUrl = e.preview_uri || (e.file_uri.split('?')[0] + '?id=' + encodeURIComponent(e.event_id) + '&kind=preview');
          thumbHtml =
            '<div class="item-thumb">' +
              '<video src="' + previewUrl + '" autoplay muted loop playsinline ' +
                'poster="' + (e.thumb_uri || '') + '" preload="metadata" ' +
                'style="width:100%;height:100%;object-fit:cover;"></video>' +
              '<div class="play-overlay">&#9654;</div>' +
            '</div>';
        } else {
          thumbHtml =
            '<div class="item-thumb">' +
              '<img src="' + (e.thumb_uri || e.file_uri) + '" loading="lazy">' +
            '</div>';
        }
        const extractBtn = '<button class="extract-clip-btn" onclick="event.stopPropagation();extractClip(\'' + (e.camera_name||'') + '\',' + (e.captured_ts||0) + ')" title="Extract clip from recording">&#9986; Clip</button>';
        item.innerHTML = thumbHtml +
          '<div class="item-info">' +
            '<div class="item-header">' +
              '<div class="item-time">' + (e._time || '') + '</div>' +
              '<div class="item-camera">' + (e.camera_name || '') + '</div>' +
            '</div>' +
            '<div class="item-caption">' + (e.caption || 'Event detected') +
              '<span style="color:var(--muted);font-weight:800;"> &middot; ' + (e._date || '') + '</span>' +
            '</div>' +
            '<div class="tag-cloud">' + zoneBadge + clipBadge + colorBadge +
              (e.detection_classes || []).map(t => '<span class="tag">' + t + '</span>').join('') +
              extractBtn +
            '</div>' +
          '</div>';
        listView.appendChild(item);
      });
      setTimeout(() => { newEventIds.clear(); }, 2000);
    }

    function extractClip(cameraId, ts) {
      const dur = 30;
      const url = '/api/recordings/extract-clip';
      fetch(url, {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({camera_id: cameraId, start_ts: ts, duration: dur})
      }).then(r => {
        if (!r.ok) return r.json().then(d => alert(d.message || 'Extract failed'));
        return r.blob();
      }).then(blob => {
        if (!blob) return;
        const a = document.createElement('a');
        a.href = URL.createObjectURL(blob);
        a.download = cameraId + '_' + ts + '_' + dur + 's.mp4';
        document.body.appendChild(a); a.click(); a.remove();
      }).catch(err => alert('Clip extraction error: ' + err));
    }

    function renderTimeline() {
      timelineRail.querySelectorAll('.activity-marker').forEach(m => m.remove());
      if (!filteredEvents.length) return;
      floatCard.style.display = 'block';
      const byTime = filteredEvents.slice().sort((a, b) => (a.captured_ts || 0) - (b.captured_ts || 0));
      const start = byTime[0].captured_ts;
      const end = byTime[byTime.length - 1].captured_ts;
      const span = (end - start) || 1;
      const railW = timelineRail.clientWidth;
      byTime.forEach(e => {
        const m = document.createElement('div');
        m.className = 'activity-marker';
        const left = ((e.captured_ts - start) / span) * railW;
        m.style.left = Math.max(0, Math.min(railW - 1, left)) + 'px';
        m.style.background = '#38bdf8';
        timelineRail.appendChild(m);
      });
      showScrubSelection(byTime[byTime.length - 1]);
    }

    const floatVideo = document.getElementById('floating-video');
    const playbackControls = document.getElementById('playback-controls');
    const playPauseBtn = document.getElementById('play-pause-btn');
    let currentScrubEvent = null;
    let playbackActive = false;

    function showScrubSelection(e) {
      if (!e) return;
      currentScrubEvent = e;
      floatImg.src = e.file_uri || e.thumb_uri;
      floatImg.style.display = '';
      floatImg.style.cursor = 'pointer';
      floatImg.onclick = () => loadRecordingForEvent(e);
      if (!playbackActive) {
        floatVideo.style.display = 'none';
        playbackControls.style.display = 'none';
      }
      floatInfo.innerHTML =
        '<div style="font-weight:900;color:var(--warn);">' + (e.camera_name || '') + '</div>' +
        '<div style="color:var(--muted);font-weight:800;">' + (e._dt || '') + '</div>' +
        '<div style="font-size:11px;color:var(--accent);cursor:pointer;margin-top:4px;" onclick="loadRecordingForEvent(currentScrubEvent)">&#9654; Load recording</div>';
    }

    function loadRecordingForEvent(e) {
      if (!e || !e.camera_name || !e.captured_ts) return;
      const playbackPort = 9996;
      const startDt = new Date(e.captured_ts * 1000);
      const iso = startDt.toISOString();
      const videoUrl = 'http://localhost:' + playbackPort + '/get?path=' + encodeURIComponent(e.camera_name) + '&start=' + encodeURIComponent(iso) + '&duration=300';
      floatImg.style.display = 'none';
      floatVideo.style.display = '';
      floatVideo.src = videoUrl;
      floatVideo.play().catch(() => {});
      playbackControls.style.display = 'flex';
      playbackActive = true;
      playPauseBtn.textContent = 'Pause';
    }

    function togglePlayback() {
      if (!floatVideo.src) return;
      if (floatVideo.paused) { floatVideo.play(); playPauseBtn.textContent = 'Pause'; }
      else { floatVideo.pause(); playPauseBtn.textContent = 'Play'; }
    }

    function seekPlayback(delta) {
      if (!floatVideo.src) return;
      floatVideo.currentTime = Math.max(0, floatVideo.currentTime + delta);
    }

    function setPlaybackSpeed(rate) {
      floatVideo.playbackRate = parseFloat(rate) || 1;
    }

    function setMode(m) {
      viewMode = m;
      listBtn.classList.toggle('active', m === 'list');
      timelineBtn.classList.toggle('active', m === 'timeline');
      listView.style.display = m === 'list' ? 'grid' : 'none';
      timelineView.style.display = m === 'timeline' ? 'flex' : 'none';
      floatCard.style.display = 'none';
      if (m !== 'timeline') {
        playbackActive = false;
        floatVideo.pause();
        floatVideo.src = '';
      }
      update();
    }
    listBtn.onclick = () => setMode('list');
    timelineBtn.onclick = () => setMode('timeline');

    timelineRail.onmousemove = (ev) => {
      if (playbackActive) return;
      const rect = timelineRail.getBoundingClientRect();
      const x = ev.clientX - rect.left;
      timelineScrubber.style.left = x + 'px';
      const byTime = filteredEvents.slice().sort((a, b) => (a.captured_ts || 0) - (b.captured_ts || 0));
      if (!byTime.length) return;
      const start = byTime[0].captured_ts;
      const end = byTime[byTime.length - 1].captured_ts;
      const targetTs = start + (x / rect.width) * (end - start);
      const closest = byTime.reduce((prev, curr) => Math.abs(curr.captured_ts - targetTs) < Math.abs(prev.captured_ts - targetTs) ? curr : prev);
      if (closest) showScrubSelection(closest);
    };

    timelineRail.onclick = (ev) => {
      const rect = timelineRail.getBoundingClientRect();
      const x = ev.clientX - rect.left;
      const byTime = filteredEvents.slice().sort((a, b) => (a.captured_ts || 0) - (b.captured_ts || 0));
      if (!byTime.length) return;
      const start = byTime[0].captured_ts;
      const end = byTime[byTime.length - 1].captured_ts;
      const targetTs = start + (x / rect.width) * (end - start);
      const closest = byTime.reduce((prev, curr) => Math.abs(curr.captured_ts - targetTs) < Math.abs(prev.captured_ts - targetTs) ? curr : prev);
      if (closest) loadRecordingForEvent(closest);
    };

    function applyTileSize() {
      const v = Math.max(160, Math.min(420, parseInt(tileSize.value || '260')));
      document.documentElement.style.setProperty('--tileSize', v + 'px');
    }

    function generateDateJumps() {
      dateJumps.innerHTML = '';
      const days = new Map();
      ALL_EVENTS.forEach(e => { const d = e._date; if (d && !days.has(d)) days.set(d, e.captured_ts); });
      [...days.entries()].slice(0, 10).forEach(([date]) => {
        const a = document.createElement('a');
        a.className = 'day-jump';
        a.textContent = date;
        a.onclick = () => { cameraFilter.value = 'all'; searchInput.value = date; update(); };
        dateJumps.appendChild(a);
      });
    }

    window.addEventListener('wheel', (ev) => {
      if (viewMode === 'list' && ev.ctrlKey) {
        ev.preventDefault();
        const cur = parseInt(tileSize.value || '260');
        tileSize.value = String(Math.max(160, Math.min(420, cur + (ev.deltaY < 0 ? 20 : -20))));
        applyTileSize();
      }
    }, { passive: false });

    searchInput.oninput = () => update();
    cameraFilter.onchange = () => update();
    zoneFilter.onchange = () => update();
    mediaFilter.onchange = () => update();
    classFilter.onchange = () => update();
    colorFilter.onchange = () => update();
    sortMode.onchange = () => update();
    tileSize.oninput = () => applyTileSize();
    refreshBtn.onclick = () => fetchEvents();
    timeFilterBtn.onclick = () => fetchEvents();

    // ------ Data fetching ------
    async function fetchEvents() {
      loadingState.style.display = 'block';
      try {
        const limit = parseInt(fetchLimit.value || '200');
        const q = searchInput.value.trim();
        const payload = { query: q, limit: limit };
        if (startTimeInput.value) payload.start_ts = Math.floor(new Date(startTimeInput.value).getTime() / 1000);
        if (endTimeInput.value) payload.end_ts = Math.floor(new Date(endTimeInput.value).getTime() / 1000);
        const res = await fetch('/api/events/search', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(payload)
        });
        const json = await res.json();
        const tl = (json.success && json.data) ? (json.data.timeline || json.data) : [];
        if (Array.isArray(tl)) {
          ALL_EVENTS = tl.map(h => {
            const eid = h.id || h.event_id || '';
            return normalize({
              event_id: eid,
              captured_ts: h.captured_ts,
              captured_at: h.captured_at,
              camera_name: h.camera_name,
              caption: h.caption,
              file_uri: '/api/events/file?id=' + encodeURIComponent(eid),
              thumb_uri: '/api/events/file?id=' + encodeURIComponent(eid) + '&kind=thumb',
              dominant_color: h.dominant_color,
              tags: h.tags || [],
              detection_classes: h.detection_classes || [],
              shape_name: h.shape_name || '',
              media_type: h.media_type || 'image',
            });
          });
          rebuildFilters();
          update();
        }
      } catch (err) {
        console.error('Fetch error:', err);
      }
      loadingState.style.display = 'none';
    }

    // ------ Socket.IO real-time ------
    let socket = null;
    function connectSocket() {
      try {
        socket = io('/realtime', { transports: ['polling', 'websocket'], reconnection: true, reconnectionDelay: 2000, upgrade: true });
        socket.on('connect', () => { liveStatus.textContent = 'Live'; });
        socket.on('disconnect', () => { liveStatus.textContent = 'Reconnecting...'; });
        socket.on('connect_error', () => { liveStatus.textContent = 'Connection error'; });

        socket.on('new_capture', (data) => {
          if (!data) return;
          const eid = data.event_id || '';
          const evt = normalize({
            event_id: eid,
            captured_ts: data.captured_ts || Math.floor(Date.now() / 1000),
            camera_name: data.camera_name || '',
            caption: data.caption || '',
            file_uri: eid ? '/api/events/file?id=' + encodeURIComponent(eid) : '',
            thumb_uri: eid ? '/api/events/file?id=' + encodeURIComponent(eid) + '&kind=thumb' : '',
            shape_name: data.shape_name || '',
            trigger_type: data.trigger_type || '',
            media_type: data.media_type || 'image',
            tags: data.tags || [],
            detection_classes: data.detection_classes || [],
            dominant_color: data.dominant_color || '',
          });
          // Dedup
          if (ALL_EVENTS.some(e => e.event_id === evt.event_id)) return;
          ALL_EVENTS.unshift(evt);
          if (ALL_EVENTS.length > MAX_EVENTS) ALL_EVENTS.pop();
          newEventIds.add(evt.event_id);
          rebuildFilters();
          update();
        });
      } catch (err) {
        console.error('Socket init error:', err);
        liveStatus.textContent = 'Socket error';
      }
    }

    // ------ Enrich button ------
    enrichBtn.onclick = async () => {
      enrichBtn.disabled = true;
      enrichBtn.textContent = 'Enriching...';
      enrichStatus.style.display = 'block';
      enrichStatus.textContent = 'Starting...';
      try {
        const startRes = await fetch('/api/events/reindex', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ include_detections: true, max_files: 500 })
        });
        const startJson = await startRes.json();
        if (!startJson.success) { enrichStatus.textContent = startJson.message || 'Failed'; enrichBtn.disabled = false; enrichBtn.textContent = 'Enrich Unclassified'; return; }

        const poll = async () => {
          for (let i = 0; i < 600; i++) {
            await new Promise(r => setTimeout(r, 2000));
            try {
              const r = await fetch('/api/events/reindex');
              const j = await r.json();
              const s = (j.data || j);
              enrichStatus.textContent = 'Processed ' + (s.processed || 0) + ' / scanned ' + (s.scanned || 0) + (s.eta_seconds ? ' (ETA ' + Math.ceil(s.eta_seconds) + 's)' : '');
              if (!s.running) { enrichStatus.textContent = 'Done: ' + (s.processed || 0) + ' enriched'; break; }
            } catch { break; }
          }
          enrichBtn.disabled = false;
          enrichBtn.textContent = 'Enrich Unclassified';
          fetchEvents();
        };
        poll();
      } catch (err) {
        enrichStatus.textContent = 'Error: ' + err.message;
        enrichBtn.disabled = false;
        enrichBtn.textContent = 'Enrich Unclassified';
      }
    };

    applyTileSize();
    fetchEvents();
    connectSocket();
  </script>
</body>
</html>"""


@api_bp.route('/events/override', methods=['POST'])
def events_override():
    """
    Persist user overrides for a detection (class/color/tags) and refresh event aggregates.
    """
    try:
        if not event_index_service:
            return jsonify({'success': False, 'message': 'Event index not available'}), 503
        data = request.get_json() or {}
        event_id = data.get("event_id") or data.get("eventId") or data.get("id")
        detection_id = data.get("detection_id") or data.get("detectionId")
        det_idx = data.get("det_idx") or data.get("detIdx")
        override_class = data.get("override_class") or data.get("class") or data.get("overrideClass")
        override_color = data.get("override_color") or data.get("color") or data.get("overrideColor")
        override_tags = data.get("override_tags") or data.get("tags") or data.get("overrideTags")
        note = data.get("note")
        updated_by = data.get("updated_by") or data.get("updatedBy") or data.get("user") or data.get("source")

        if isinstance(override_tags, str):
            override_tags = [t.strip() for t in override_tags.split(",") if t.strip()]
        if override_tags is not None and not isinstance(override_tags, list):
            override_tags = None

        if not event_id:
            return jsonify({'success': False, 'message': 'event_id is required'}), 400
        if not detection_id and det_idx is None:
            return jsonify({'success': False, 'message': 'detection_id or det_idx is required'}), 400

        try:
            det_idx_i = int(det_idx) if det_idx is not None and str(det_idx).strip() != "" else None
        except Exception:
            det_idx_i = None

        result = event_index_service.set_detection_override(
            event_id=str(event_id),
            detection_id=str(detection_id) if isinstance(detection_id, str) and detection_id.strip() else None,
            det_idx=det_idx_i,
            override_class=str(override_class) if override_class is not None else None,
            override_color=str(override_color) if override_color is not None else None,
            override_tags=override_tags,
            note=str(note) if note is not None else None,
            updated_by=str(updated_by) if updated_by is not None else None,
        )
        return jsonify({'success': True, 'data': result})
    except ValueError as e:
        return jsonify({'success': False, 'message': str(e)}), 400
    except Exception as e:
        logger.error(f"Events override error: {e}")
        return jsonify({'success': False, 'message': 'Failed to save override'}), 500


def _events_reindex_status_payload() -> Dict[str, Any]:
    with _events_reindex_lock:
        return dict(_events_reindex_state)


@api_bp.route('/events/reindex', methods=['GET', 'POST'])
def events_reindex():
    """
    Bulk reindex Motion Watch captures in stable order.
    Cloud enrichment is NEVER run unless explicitly enabled and acknowledged.
    """
    try:
        if not event_index_service:
            return jsonify({'success': False, 'message': 'Event index not available'}), 503

        if request.method == 'GET':
            return jsonify({'success': True, 'data': _events_reindex_status_payload()})

        data = request.get_json() or {}
        force = bool(data.get("force", False))
        max_files = data.get("max_files") or data.get("maxFiles")
        since_ts = data.get("since_ts") or data.get("sinceTs") or data.get("start_ts") or data.get("startTs")
        until_ts = data.get("until_ts") or data.get("untilTs") or data.get("end_ts") or data.get("endTs")
        include_detections = bool(data.get("include_detections", data.get("includeDetections", True)))
        include_vision = bool(data.get("include_vision", data.get("includeVision", False)))

        # Cloud enrichment gating (hard default OFF)
        cloud_enrich = bool(data.get("cloud_enrich", data.get("cloudEnrich", False)))
        cloud_ack = str(data.get("cloud_ack") or data.get("cloudAck") or "").strip()
        cloud_max_calls = data.get("cloud_max_calls") or data.get("cloudMaxCalls") or 0
        cloud_model = data.get("cloud_model") or data.get("cloudModel")
        cloud_min_interval_ms = data.get("cloud_min_interval_ms") or data.get("cloudMinIntervalMs") or 1500

        if cloud_enrich:
            # Require explicit acknowledgement to protect against accidental bulk spend/rate-limit.
            if cloud_ack != "I_UNDERSTAND_BULK_CLOUD_ENRICH_CAN_RATE_LIMIT":
                return jsonify(
                    {
                        'success': False,
                        'message': (
                            "Cloud enrichment is disabled by default. "
                            "To enable for a bulk batch, you must pass "
                            "`cloud_ack: I_UNDERSTAND_BULK_CLOUD_ENRICH_CAN_RATE_LIMIT` "
                            "and set `cloud_max_calls` to a small number (e.g., 25)."
                        ),
                        'data': {'requires_ack': True},
                    }
                ), 400
            try:
                cloud_max_calls = int(cloud_max_calls)
            except Exception:
                cloud_max_calls = 0
            if cloud_max_calls <= 0:
                return jsonify({'success': False, 'message': 'cloud_max_calls must be > 0 when cloud_enrich=true'}), 400
        else:
            cloud_max_calls = 0

        # Normalize time bounds (seconds)
        def _to_ts(v):
            if v is None:
                return None
            if isinstance(v, (int, float)):
                return int(v)
            if isinstance(v, str) and v.strip():
                s = v.strip()
                if s.isdigit():
                    return int(s)
                try:
                    dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
                    return int(dt.timestamp())
                except Exception:
                    return None
            return None

        since_i = _to_ts(since_ts)
        until_i = _to_ts(until_ts)
        try:
            max_files_i = max(1, int(max_files)) if max_files is not None else None
        except Exception:
            max_files_i = None

        reset = bool(data.get("reset", False))
        with _events_reindex_lock:
            if _events_reindex_state.get("running") and not reset:
                return jsonify({'success': True, 'data': dict(_events_reindex_state)})

            job_id = uuid.uuid4().hex
            _events_reindex_state.update(
                {
                    "running": True,
                    "job_id": job_id,
                    "started_at": int(time.time()),
                    "finished_at": None,
                    "updated_at": int(time.time()),
                    "total_target": None,
                    "scanned": 0,
                    "processed": 0,
                    "skipped": 0,
                    "errors": [],
                    "error_count": 0,
                    "eta_seconds": None,
                    "cloud": {
                        "enabled": bool(cloud_enrich),
                        "max_calls": int(cloud_max_calls or 0),
                        "calls": 0,
                        "provider": getattr(getattr(ai_agent, "provider", None), "__class__", type("x", (), {})).__name__ if ai_agent else None,
                        "model": str(cloud_model) if cloud_model else None,
                    },
                    "config": {
                        "force": force,
                        "max_files": max_files_i,
                        "since_ts": since_i,
                        "until_ts": until_i,
                        "include_detections": include_detections,
                        "include_vision": include_vision,
                    },
                }
            )

        def _worker():
            errors: List[str] = []
            try:
                # Build a stable candidate list (oldest-first) so we don't keep reprocessing "newest" forever.
                roots = getattr(event_index_service, "capture_roots", [Path("captures/motion_watch")]) or [Path("captures/motion_watch")]
                candidates: List[Path] = []
                for root in roots:
                    try:
                        root_abs = (Path.cwd() / Path(root)).resolve() if not Path(root).is_absolute() else Path(root).resolve()
                        if not root_abs.exists():
                            continue
                        for ext in ("*.jpg", "*.jpeg", "*.png"):
                            candidates.extend(list(root_abs.glob(ext)))
                    except Exception:
                        continue

                def _infer_ts(p: Path) -> int:
                    try:
                        name = p.name
                        m = re.search(r"_watch_(\d+)\.(?:jpg|jpeg|png)$", name, re.IGNORECASE)
                        if m:
                            return int(m.group(1))
                    except Exception:
                        pass
                    try:
                        return int(p.stat().st_mtime)
                    except Exception:
                        return 0

                candidates.sort(key=lambda p: (_infer_ts(p), str(p).lower()))

                # Best-effort compute total_target after time filtering and skip logic.
                total_target = None
                try:
                    # Apply time filter first
                    filtered = []
                    for p in candidates:
                        ts = _infer_ts(p)
                        if since_i is not None and ts < int(since_i):
                            continue
                        if until_i is not None and ts > int(until_i):
                            continue
                        filtered.append(p)
                    total_target = len(filtered)
                    if max_files_i is not None:
                        total_target = min(int(max_files_i), int(total_target))
                    with _events_reindex_lock:
                        _events_reindex_state["total_target"] = int(total_target)
                        _events_reindex_state["updated_at"] = int(time.time())
                except Exception:
                    total_target = None

                # Load already-indexed paths (absolute) + whether detections have been indexed.
                # This is crucial: most installs already have `events` rows, but `event_detections` starts empty.
                known: Dict[str, Dict[str, Any]] = {}
                try:
                    with event_index_service._connect() as conn:  # pylint: disable=protected-access
                        rows = conn.execute("SELECT file_path, detections_indexed_at FROM events;").fetchall()
                    for r in rows:
                        fp = r["file_path"]
                        if isinstance(fp, str) and fp:
                            try:
                                known[str(Path(fp).resolve())] = {
                                    "detections_indexed_at": r["detections_indexed_at"],
                                }
                            except Exception:
                                continue
                except Exception:
                    known = {}

                cloud_calls = 0
                last_cloud_call = 0.0

                processed = scanned = skipped = 0
                start_time = time.time()
                for p in candidates:
                    with _events_reindex_lock:
                        if not _events_reindex_state.get("running") or _events_reindex_state.get("job_id") != job_id:
                            break
                    if max_files_i is not None and processed >= int(max_files_i):
                        break

                    scanned += 1
                    try:
                        p_abs = p.resolve()
                        ts = _infer_ts(p_abs)
                        if since_i is not None and ts < int(since_i):
                            continue
                        if until_i is not None and ts > int(until_i):
                            continue

                        existing = known.get(str(p_abs))
                        if existing is not None and not force:
                            # If detections are requested, only skip when we've already attempted detection indexing.
                            dets_indexed = bool(existing.get("detections_indexed_at"))
                            if include_detections and not dets_indexed:
                                # Need to populate per-object detections/crops/colors for this event.
                                pass
                            else:
                                skipped += 1
                                continue

                        # Prefer relative paths for portability and capture-root validation.
                        try:
                            rel_str = str(p_abs.relative_to(Path.cwd().resolve()))
                        except Exception:
                            rel_str = str(p_abs)

                        # Load sidecar if present (camera_id/name/trigger/motion_box)
                        payload: Dict[str, Any] = {"file_path": rel_str}
                        try:
                            sidecar = p_abs.with_suffix(".json")
                            if sidecar.exists() and sidecar.is_file():
                                j = json.loads(sidecar.read_text(encoding="utf-8"))
                                if isinstance(j, dict):
                                    payload = {**j}
                        except Exception:
                            payload = {"file_path": rel_str}

                        payload["enable_vision"] = bool(include_vision)
                        payload["enable_detections"] = bool(include_detections)
                        payload["replace_detections"] = True

                        # Local ingest (YOLO + crops + color); always safe
                        event_index_service.ingest(payload)
                        processed += 1

                        # Optional cloud enrichment (strictly gated)
                        if cloud_enrich and cloud_calls < int(cloud_max_calls or 0):
                            try:
                                if not ai_agent or not getattr(ai_agent, "provider", None):
                                    raise RuntimeError("AI provider not available for cloud enrichment")
                                # Pick at most one ambiguous vehicle crop per event
                                dets = event_index_service.list_detections(payload.get("event_id") or payload.get("id") or "")
                                if not dets:
                                    # list_detections expects event_id; fetch by file_path instead
                                    dets = []
                                # Workaround: look up event_id from DB if not in payload
                                event_id = payload.get("event_id") or payload.get("id")
                                if not event_id:
                                    try:
                                        with event_index_service._connect() as conn:  # pylint: disable=protected-access
                                            r = conn.execute("SELECT id FROM events WHERE file_path = ? LIMIT 1;", (str(p_abs),)).fetchone()
                                        if r:
                                            event_id = r["id"]
                                    except Exception:
                                        event_id = None
                                if event_id:
                                    dets = event_index_service.list_detections(str(event_id))
                                veh = [d for d in dets if (d.get("class") in {"car", "truck", "bus", "motorcycle"})]
                                veh.sort(key=lambda d: float(d.get("area") or 0.0), reverse=True)
                                target = None
                                for d in veh[:5]:
                                    col = (d.get("color") or "").strip().lower()
                                    conf = float(d.get("confidence") or 0.0)
                                    area = float(d.get("area") or 0.0)
                                    if area < 900.0:
                                        continue
                                    if (not col) or col in {"gray"} or conf < 0.30:
                                        if d.get("crop_path"):
                                            target = d
                                            break
                                if target:
                                    # simple rate limiter
                                    now_t = time.time()
                                    try:
                                        min_interval = max(0.2, float(cloud_min_interval_ms or 1500) / 1000.0)
                                    except Exception:
                                        min_interval = 1.5
                                    if now_t - last_cloud_call < min_interval:
                                        time.sleep(min_interval - (now_t - last_cloud_call))

                                    crop_b64 = event_index_service.read_file_base64(str(target["crop_path"]), max_bytes=240_000)
                                    if crop_b64:
                                        prompt = (
                                            "You are labeling a single vehicle crop image.\n"
                                            "Return JSON ONLY with no extra text:\n"
                                            '{"class":"car|truck|bus|motorcycle|other","color":"white|black|gray|red|blue|green|yellow|brown|unknown"}\n'
                                            "If unsure, set color to unknown and class to other."
                                        )
                                        coro = ai_agent.provider.vision(
                                            image=f"data:image/jpeg;base64,{crop_b64}",
                                            prompt=prompt,
                                            model=str(cloud_model) if cloud_model else getattr(ai_agent, "vision_model", None) or getattr(ai_agent, "model", ""),
                                            opts={
                                                "temperature": 0.0,
                                                "max_tokens": 180,
                                                "timeout": 45,
                                                "source": "cloud",
                                                "include_detections": False,
                                                "use_cache": True,
                                            },
                                        )
                                        res = _run_coro_safe(coro)
                                        content = ""
                                        if isinstance(res, dict):
                                            content = str(res.get("content") or res.get("analysis", {}).get("caption") or "")
                                        parsed = None
                                        try:
                                            parsed = json.loads(content) if content.strip().startswith("{") else None
                                        except Exception:
                                            parsed = None
                                        if isinstance(parsed, dict):
                                            new_color = str(parsed.get("color") or "").strip().lower()
                                            new_class = str(parsed.get("class") or "").strip().lower()
                                            # Only apply if it improves ambiguity; store as computed overrides? no - update detection row fields
                                            if new_color and new_color not in {"unknown"}:
                                                try:
                                                    with event_index_service._connect() as conn:  # pylint: disable=protected-access
                                                        conn.execute(
                                                            "UPDATE event_detections SET color = ?, source = COALESCE(source,'') || '+cloud', updated_at = ? WHERE detection_id = ?;",
                                                            (new_color, int(time.time()), str(target["detection_id"])),
                                                        )
                                                except Exception:
                                                    pass
                                            if new_class and new_class in {"car", "truck", "bus", "motorcycle"}:
                                                try:
                                                    with event_index_service._connect() as conn:  # pylint: disable=protected-access
                                                        conn.execute(
                                                            "UPDATE event_detections SET class = ?, source = COALESCE(source,'') || '+cloud', updated_at = ? WHERE detection_id = ?;",
                                                            (new_class, int(time.time()), str(target["detection_id"])),
                                                        )
                                                except Exception:
                                                    pass
                                            try:
                                                event_index_service._refresh_event_aggregates(str(event_id))  # pylint: disable=protected-access
                                            except Exception:
                                                pass

                                            cloud_calls += 1
                                            last_cloud_call = time.time()
                                            with _events_reindex_lock:
                                                _events_reindex_state["cloud"]["calls"] = int(cloud_calls)
                            except Exception as cloud_err:
                                errors.append(f"{p_abs} cloud: {cloud_err}")

                        with _events_reindex_lock:
                            _events_reindex_state["scanned"] = int(scanned)
                            _events_reindex_state["processed"] = int(processed)
                            _events_reindex_state["skipped"] = int(skipped)
                            _events_reindex_state["errors"] = errors[-25:]
                            _events_reindex_state["error_count"] = len(errors)
                            _events_reindex_state["updated_at"] = int(time.time())
                            # ETA based on processed rate and total_target (best-effort)
                            try:
                                tt = _events_reindex_state.get("total_target")
                                elapsed = max(0.001, time.time() - start_time)
                                rate = (float(processed) / elapsed) if processed > 0 else 0.0
                                if isinstance(tt, int) and tt > 0 and rate > 0:
                                    remaining = max(0, int(tt) - int(processed))
                                    _events_reindex_state["eta_seconds"] = int(remaining / rate)
                                else:
                                    _events_reindex_state["eta_seconds"] = None
                            except Exception:
                                _events_reindex_state["eta_seconds"] = None

                    except Exception as e:
                        errors.append(f"{p}: {e}")
                        with _events_reindex_lock:
                            _events_reindex_state["errors"] = errors[-25:]
                            _events_reindex_state["error_count"] = len(errors)
                            _events_reindex_state["updated_at"] = int(time.time())
                        continue

                with _events_reindex_lock:
                    _events_reindex_state["scanned"] = int(scanned)
                    _events_reindex_state["processed"] = int(processed)
                    _events_reindex_state["skipped"] = int(skipped)
                    _events_reindex_state["updated_at"] = int(time.time())
            finally:
                with _events_reindex_lock:
                    if _events_reindex_state.get("job_id") == job_id:
                        _events_reindex_state["running"] = False
                        _events_reindex_state["finished_at"] = int(time.time())
                        _events_reindex_state["updated_at"] = int(time.time())

        threading.Thread(target=_worker, daemon=True).start()
        return jsonify({'success': True, 'data': _events_reindex_status_payload()})

    except Exception as e:
        logger.error(f"Events reindex error: {e}")
        return jsonify({'success': False, 'message': 'Failed to start reindex'}), 500


# ==================== RECORDING PLAYBACK HELPERS ====================


@api_bp.route('/recordings/extract-clip', methods=['POST'])
def recordings_extract_clip():
    """
    Extract a clip from continuous MediaMTX recordings using ffmpeg stream-copy
    (no re-encoding, near-instant).  Returns the clip as a downloadable file.

    Body JSON:
        camera_id  – camera / MediaMTX path name
        start_ts   – UNIX timestamp (seconds) for clip start
        end_ts     – (optional) UNIX timestamp for clip end; defaults to start_ts + duration
        duration   – (optional) seconds, default 30
        event_id   – (optional) resolve start_ts from this event
    """
    from flask import send_file
    data = request.get_json(silent=True) or {}
    camera_id = data.get("camera_id", "")
    duration = int(data.get("duration", 30))

    start_ts = data.get("start_ts")
    end_ts = data.get("end_ts")
    event_id = data.get("event_id")

    if event_id and not start_ts:
        if event_index_service:
            hits = event_index_service.search(query="", limit=1, camera_name=camera_id)
            for h in (hits or []):
                if str(h.id) == str(event_id):
                    start_ts = h.captured_ts
                    break
            if not start_ts:
                try:
                    import sqlite3 as _sql
                    conn = _sql.connect(str(event_index_service.db_path))
                    row = conn.execute("SELECT captured_ts FROM events WHERE id = ?", (str(event_id),)).fetchone()
                    conn.close()
                    if row:
                        start_ts = int(row[0])
                except Exception:
                    pass

    if not start_ts:
        return jsonify({"success": False, "message": "start_ts or event_id required"}), 400

    start_ts = int(start_ts)
    if end_ts:
        duration = max(1, int(end_ts) - start_ts)

    from core.paths import get_recordings_dir
    rec_base = get_recordings_dir()
    rec_dir = rec_base / camera_id
    # Also try camera-name directory (new naming scheme)
    if not rec_dir.exists():
        try:
            from app import cameras_db
            import re as _re
            cam = next((c for c in cameras_db if c.get('id') == camera_id), None)
            if cam:
                safe = _re.sub(r'[<>:"/\\|?*\x00-\x1f]', '_', cam.get('name', '')).strip('. ')[:120]
                if safe:
                    alt = rec_base / safe
                    if alt.exists():
                        rec_dir = alt
                custom = (cam.get('recording_dir') or '').strip()
                if custom and not rec_dir.exists():
                    from pathlib import Path
                    alt2 = Path(custom).expanduser().resolve() / (safe or camera_id)
                    if alt2.exists():
                        rec_dir = alt2
        except Exception:
            pass
    if not rec_dir.exists():
        return jsonify({"success": False, "message": f"No recordings found for {camera_id}"}), 404

    from datetime import datetime as _dt, timezone as _tz
    target_dt = _dt.fromtimestamp(start_ts, tz=_tz.utc)

    segments = sorted(rec_dir.rglob("*.mp4"))
    if not segments:
        return jsonify({"success": False, "message": "No recording segments found"}), 404

    best = None
    for seg in segments:
        try:
            name = seg.stem
            seg_dt = _dt.strptime(name[:19], "%Y-%m-%d_%H-%M-%S").replace(tzinfo=_tz.utc)
            if seg_dt <= target_dt:
                best = seg
        except Exception:
            continue
    if best is None:
        best = segments[0]

    import subprocess, tempfile, shutil
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        return jsonify({"success": False, "message": "ffmpeg not found on system"}), 500

    out_file = Path(tempfile.mktemp(suffix=".mp4", prefix="knoxnet_clip_"))
    offset_s = max(0, start_ts - int(best.stem[:19].replace("-", "").replace("_", "")[:14] if len(best.stem) >= 19 else "0") if False else 0)

    cmd = [
        ffmpeg, "-y",
        "-ss", str(offset_s),
        "-i", str(best),
        "-t", str(duration),
        "-c", "copy",
        "-movflags", "+faststart",
        str(out_file),
    ]
    try:
        subprocess.run(cmd, timeout=30, check=True, capture_output=True)
    except Exception as e:
        return jsonify({"success": False, "message": f"ffmpeg error: {e}"}), 500

    if not out_file.exists() or out_file.stat().st_size == 0:
        return jsonify({"success": False, "message": "Clip extraction produced empty file"}), 500

    return send_file(
        str(out_file),
        mimetype="video/mp4",
        as_attachment=True,
        download_name=f"{camera_id}_{start_ts}_{duration}s.mp4",
    )


@api_bp.route('/recordings/list', methods=['GET'])
def recordings_list():
    """List recording segments for a camera path.

    Searches both UUID-based dirs (legacy) and camera-name dirs (new naming),
    including date subdirectories.
    """
    camera_id = request.args.get("camera_id", "")
    if not camera_id:
        return jsonify({"success": False, "message": "camera_id required"}), 400

    from core.paths import get_recordings_dir
    rec_base = get_recordings_dir()

    # Also check a per-camera custom recording_dir if set
    cam = None
    try:
        from app import cameras_db
        cam = next((c for c in cameras_db if c.get('id') == camera_id), None)
    except Exception:
        pass

    search_dirs = []
    # UUID-based dir (legacy)
    search_dirs.append(rec_base / camera_id)
    # Camera-name dir (new naming)
    if cam:
        import re
        cam_name = cam.get('name', '')
        if cam_name:
            safe = re.sub(r'[<>:"/\\|?*\x00-\x1f]', '_', cam_name).strip('. ')[:120]
            if safe:
                search_dirs.append(rec_base / safe)
        custom = (cam.get('recording_dir') or '').strip()
        if custom:
            from pathlib import Path
            custom_base = Path(custom).expanduser().resolve()
            if safe:
                search_dirs.append(custom_base / safe)

    segments = []
    seen_paths: set = set()
    for rec_dir in search_dirs:
        if not rec_dir.exists():
            continue
        # Search top-level and date subdirectories
        for seg in sorted(rec_dir.rglob("*.mp4")):
            real = str(seg.resolve())
            if real in seen_paths:
                continue
            seen_paths.add(real)
            try:
                st = seg.stat()
                rel = seg.relative_to(rec_dir)
                segments.append({
                    "name": str(rel),
                    "path": str(seg),
                    "size_mb": round(st.st_size / (1 << 20), 2),
                    "mtime": int(st.st_mtime),
                })
            except (OSError, ValueError):
                continue

    segments.sort(key=lambda s: s.get("mtime", 0))
    return jsonify({"success": True, "data": segments})


# ==================== NETWORK UTILITY HELPERS ====================

_MAC_REGEX = re.compile(r'([0-9A-Fa-f]{2}(?:[:-][0-9A-Fa-f]{2}){5})')


def _is_ipv4_address(value: str) -> bool:
    try:
        ipaddress.IPv4Address(value)
        return True
    except (ipaddress.AddressValueError, ValueError):
        return False


def _normalize_mac(mac: str) -> str:
    normalized = mac.replace('-', ':').lower()
    parts = [segment.zfill(2) for segment in normalized.split(':') if segment]
    if len(parts) == 6:
        return ':'.join(parts)
    return normalized


def _read_arp_table() -> Dict[str, str]:
    table: Dict[str, str] = {}
    try:
        system = platform.system().lower()
        command = ['arp', '-a'] if system == 'windows' else ['arp', '-n']
        creationflags = subprocess.CREATE_NO_WINDOW if system == 'windows' and hasattr(subprocess, 'CREATE_NO_WINDOW') else 0
        result = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=3,
            creationflags=creationflags
        )
        if result.returncode != 0:
            return table
        for raw_line in result.stdout.splitlines():
            line = raw_line.strip()
            if not line:
                continue
            lower = line.lower()
            if lower.startswith('interface:') or 'internet address' in lower:
                continue
            tokens = line.split()
            candidate_ip = None
            if tokens and _is_ipv4_address(tokens[0]):
                candidate_ip = tokens[0]
            elif len(tokens) > 1 and _is_ipv4_address(tokens[1]):
                candidate_ip = tokens[1]
            if not candidate_ip:
                continue
            mac_match = _MAC_REGEX.search(line)
            if not mac_match:
                continue
            table[candidate_ip] = _normalize_mac(mac_match.group(1))
    except Exception as exc:
        logger.debug(f"Failed to read ARP table: {exc}")
    return table


def _list_ipv4_interfaces() -> List[Dict[str, object]]:
    interfaces: List[Dict[str, object]] = []
    try:
        for iface_name, addr_list in psutil.net_if_addrs().items():
            for addr in addr_list:
                if addr.family != socket.AF_INET:
                    continue
                ip_addr = addr.address
                netmask = getattr(addr, 'netmask', None)
                if not ip_addr or not netmask:
                    continue
                if ip_addr.startswith('127.') or ip_addr.startswith('169.254.'):
                    continue
                try:
                    network = ipaddress.IPv4Network((ip_addr, netmask), strict=False)
                except Exception:
                    continue
                if network.prefixlen >= 31:
                    host_count = network.num_addresses
                else:
                    host_count = max(network.num_addresses - 2, 0)
                interfaces.append({
                    'name': iface_name,
                    'ip': ip_addr,
                    'netmask': netmask,
                    'broadcast': getattr(addr, 'broadcast', None),
                    'cidr': str(network),
                    'prefix': network.prefixlen,
                    'host_count': host_count
                })
    except Exception as exc:
        logger.warning(f"Unable to enumerate network interfaces: {exc}")
        return interfaces

    interfaces.sort(key=lambda item: (-int(item['host_count']), item['name']))
    for index, interface in enumerate(interfaces):
        interface['is_default'] = index == 0
    return interfaces


def _ping_host(ip: str, timeout_ms: int = 750) -> Dict[str, Optional[object]]:
    system = platform.system().lower()
    timeout_ms = max(100, min(timeout_ms, 5000))
    if system == 'windows':
        command = ['ping', '-n', '1', '-w', str(timeout_ms), ip]
        run_timeout = max(timeout_ms / 1000.0 + 1.0, 1.0)
    else:
        wait_arg = max(1, int(timeout_ms / 1000) or 1)
        command = ['ping', '-c', '1', '-W', str(wait_arg), ip]
        run_timeout = max(timeout_ms / 1000.0 + 1.0, 1.5)

    creationflags = subprocess.CREATE_NO_WINDOW if system == 'windows' and hasattr(subprocess, 'CREATE_NO_WINDOW') else 0
    started_at = time.perf_counter()
    try:
        completed = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=run_timeout,
            creationflags=creationflags
        )
        elapsed_ms = (time.perf_counter() - started_at) * 1000.0
        reachable = completed.returncode == 0
    except subprocess.TimeoutExpired:
        return {
            'ip': ip,
            'reachable': False,
            'latency_ms': None,
            'hostname': None,
            'error': 'timeout'
        }
    except FileNotFoundError:
        return {
            'ip': ip,
            'reachable': False,
            'latency_ms': None,
            'hostname': None,
            'error': 'ping_not_available'
        }
    except Exception as exc:
        logger.debug(f"Ping failed for {ip}: {exc}")
        return {
            'ip': ip,
            'reachable': False,
            'latency_ms': None,
            'hostname': None,
            'error': str(exc)
        }

    hostname = None
    if reachable:
        try:
            hostname = socket.gethostbyaddr(ip)[0]
        except (socket.herror, socket.gaierror, TimeoutError):
            hostname = None
        except Exception as exc:
            logger.debug(f"Reverse lookup failed for {ip}: {exc}")
            hostname = None

    return {
        'ip': ip,
        'reachable': reachable,
        'latency_ms': round(elapsed_ms, 2) if reachable else None,
        'hostname': hostname
    }


# ==================== HEALTH AND STATUS ENDPOINTS ====================
# ==================== HEALTH AND STATUS ENDPOINTS ====================

@api_bp.route('/health', methods=['GET'])
def health_check():
    """Comprehensive API health check"""
    try:
        services = {
            'camera_manager': camera_manager is not None,
            'stream_server': stream_server is not None,
            'ai_analyzer': ai_analyzer is not None,
            'alert_system': alert_system is not None
        }

        # Get service statuses
        service_statuses = {}

        if camera_manager:
            service_statuses['camera_manager'] = {
                'available': True,
                'running': getattr(camera_manager, '_running', False),
                'cameras_count': len(safe_service_call(camera_manager, 'get_all_cameras', []))
            }
        else:
            service_statuses['camera_manager'] = {'available': False}

        if stream_server:
            service_statuses['stream_server'] = {
                'available': True,
                'running': getattr(stream_server, 'running', False),
                'connections': len(safe_service_call(stream_server, 'get_client_connections', {}))
            }
        else:
            service_statuses['stream_server'] = {'available': False}

        if ai_analyzer:
            service_statuses['ai_analyzer'] = {
                'available': True,
                'running': getattr(ai_analyzer, 'running', True)
            }
        else:
            service_statuses['ai_analyzer'] = {'available': False}

        if alert_system:
            service_statuses['alert_system'] = {
                'available': True,
                'running': getattr(alert_system, 'running', True)
            }
        else:
            service_statuses['alert_system'] = {'available': False}

        return jsonify({
            "success": True,
            "data": {
                "status": "healthy",
                "message": "Knoxnet VMS Beta API is running",
                "timestamp": datetime.now().isoformat(),
                "version": "2.1.4",
                "services": services,
                "service_statuses": service_statuses
            }
        })

    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return jsonify({
            "success": False,
            "message": f"Health check failed: {str(e)}",
            "timestamp": datetime.now().isoformat()
        }), 500


# ==================== AUDIO ENDPOINTS ====================

@api_bp.route('/audio/tracks', methods=['GET'])
def list_audio_tracks():
    try:
        storage_dir = _os.path.join('data', 'audio')
        _os.makedirs(storage_dir, exist_ok=True)
        items = []
        for name in _os.listdir(storage_dir):
            path = _os.path.join(storage_dir, name)
            if not _os.path.isfile(path):
                continue
            stat = _os.stat(path)
            items.append({
                'id': name,
                'filename': name,
                'original_name': name,
                'size_bytes': stat.st_size,
                'mime_type': 'audio/mpeg' if name.lower().endswith('.mp3') else 'audio/wav' if name.lower().endswith('.wav') else 'application/octet-stream',
                'uploaded_at': datetime.fromtimestamp(stat.st_mtime).isoformat()
            })
        return jsonify({ 'success': True, 'data': { 'tracks': sorted(items, key=lambda x: x['uploaded_at'], reverse=True) } })
    except Exception as e:
        logger.error(f"list_audio_tracks error: {e}")
        return jsonify({ 'success': False, 'message': 'Failed to list tracks' }), 500


@api_bp.route('/audio/upload', methods=['POST'])
def upload_audio_track():
    try:
        if 'file' not in request.files:
            return jsonify({ 'success': False, 'message': 'No file provided' }), 400
        file = request.files['file']
        if not file or file.filename.strip() == '':
            return jsonify({ 'success': False, 'message': 'Empty file' }), 400
        storage_dir = _os.path.join('data', 'audio')
        _os.makedirs(storage_dir, exist_ok=True)
        ext = _os.path.splitext(file.filename)[1]
        file_id = f"{_uuid.uuid4().hex}{ext}"
        out_path = _os.path.join(storage_dir, file_id)
        file.save(out_path)
        stat = _os.stat(out_path)
        meta = {
            'id': file_id,
            'filename': file_id,
            'original_name': file.filename,
            'size_bytes': stat.st_size,
            'mime_type': file.mimetype or 'application/octet-stream',
            'uploaded_at': datetime.fromtimestamp(stat.st_mtime).isoformat()
        }
        return jsonify({ 'success': True, 'data': { 'track': meta } })
    except Exception as e:
        logger.error(f"upload_audio_track error: {e}")
        return jsonify({ 'success': False, 'message': 'Failed to upload track' }), 500


@api_bp.route('/audio/tracks/<track_id>', methods=['DELETE'])
def delete_audio_track(track_id):
    try:
        storage_dir = _os.path.join('data', 'audio')
        path = _os.path.join(storage_dir, track_id)
        if not _os.path.exists(path):
            return jsonify({ 'success': False, 'message': 'Not found' }), 404
        _os.remove(path)
        return jsonify({ 'success': True })
    except Exception as e:
        logger.error(f"delete_audio_track error: {e}")
        return jsonify({ 'success': False, 'message': 'Failed to delete' }), 500


@api_bp.route('/audio/tracks/<track_id>/stream', methods=['GET'])
def stream_audio_track(track_id):
    try:
        storage_dir = _os.path.join('data', 'audio')
        path = _os.path.join(storage_dir, track_id)
        if not _os.path.exists(path):
            return jsonify({ 'success': False, 'message': 'Not found' }), 404
        def generate():
            with open(path, 'rb') as f:
                while True:
                    chunk = f.read(8192)
                    if not chunk:
                        break
                    yield chunk
        mime = 'audio/mpeg' if track_id.lower().endswith('.mp3') else 'audio/wav' if track_id.lower().endswith('.wav') else 'application/octet-stream'
        return Response(generate(), mimetype=mime)
    except Exception as e:
        logger.error(f"stream_audio_track error: {e}")
        return jsonify({ 'success': False, 'message': 'Failed to stream' }), 500


@api_bp.route('/audio/cameras/<camera_id>/whep', methods=['GET'])
def audio_whep_proxy(camera_id):
    """Return a MediaMTX WHEP URL for the camera's audio path (client plays natively).
    This keeps implementation simple and production-friendly without SDP handling here.
    """
    try:
        # Use MediaMTX WebRTC port env or default
        mediamtx_webrtc = _os.environ.get('MEDIAMTX_WEBRTC_URL', 'http://localhost:8889')

        # Fix hostname for external access (Docker networking vs Browser access)
        try:
            from urllib.parse import urlparse
            parsed_mtx = urlparse(mediamtx_webrtc)
            # If using internal docker name or localhost, replace with request hostname to ensure reachability from browser
            # This handles the common case where backend is in Docker (seeing 'mediamtx') but browser is on host/LAN
            if parsed_mtx.hostname in ['mediamtx', 'mediamtx_server', 'localhost', '127.0.0.1']:
                # Get the hostname the client used to access the API
                client_host = request.host.split(':')[0]
                if client_host:
                    # Reconstruct URL with client hostname but keep the configured port
                    mediamtx_webrtc = mediamtx_webrtc.replace(parsed_mtx.hostname, client_host)
        except Exception as e:
            logger.debug(f"Failed to adjust WHEP hostname: {e}")

        # Try to resolve configured MediaMTX path for this camera if available
        stream_path = None
        cam_obj: Optional[Dict[str, Any]] = None
        try:
            # Prefer camera_manager if present
            if camera_manager is not None:
                cam = camera_manager.get_camera(camera_id)
                if cam is not None:
                    stream_path = getattr(cam, 'mediamtx_path', None) or getattr(cam, 'id', None)
            if not stream_path:
                # Fallback to in-memory/app cameras_db or file
                try:
                    from app import cameras_db  # type: ignore
                except Exception:
                    import json as _json
                    with open('cameras.json', 'r') as _f:
                        cameras_db = _json.load(_f)
                for cam in cameras_db:
                    if cam.get('id') == camera_id:
                        cam_obj = cam
                        stream_path = cam.get('mediamtx_path') or cam.get('id')
                        break
        except Exception:
            pass

        # Ensure the MediaMTX path exists (important after MediaMTX restarts).
        # This is best-effort; we still return a URL even if creation fails.
        try:
            if cam_obj and cam_obj.get("rtsp_url"):
                from app import ensure_camera_mediamtx_ready  # type: ignore

                ensure_camera_mediamtx_ready(cam_obj, force_check=True)
        except Exception as e:
            logger.debug(f"ensure_camera_mediamtx_ready failed in audio_whep_proxy: {e}")

        # Use resolved mediamtx path directly (no implicit prefixes)
        if not stream_path:
            stream_path = camera_id
        whep_path = stream_path

        whep = f"{mediamtx_webrtc}/{whep_path}/whep"
        return jsonify({ 'success': True, 'data': { 'whep': whep } })
    except Exception as e:
        logger.error(f"audio_whep_proxy error: {e}")
        return jsonify({ 'success': False, 'message': 'Failed to resolve WHEP' }), 500


@api_bp.route('/audio/cameras/<camera_id>/whip', methods=['GET'])
def audio_whip_proxy(camera_id):
    """Return a MediaMTX WHIP URL for sending audio TO the camera (two-way audio).
    """
    try:
        mediamtx_webrtc = _os.environ.get('MEDIAMTX_WEBRTC_URL', 'http://localhost:8889')

        # Fix hostname for external access (Docker networking vs Browser access)
        try:
            from urllib.parse import urlparse
            parsed_mtx = urlparse(mediamtx_webrtc)
            # If using internal docker name or localhost, replace with request hostname to ensure reachability from browser
            if parsed_mtx.hostname in ['mediamtx', 'mediamtx_server', 'localhost', '127.0.0.1']:
                # Get the hostname the client used to access the API
                client_host = request.host.split(':')[0]
                if client_host:
                    # Reconstruct URL with client hostname but keep the configured port
                    mediamtx_webrtc = mediamtx_webrtc.replace(parsed_mtx.hostname, client_host)
        except Exception as e:
            logger.debug(f"Failed to adjust WHIP hostname: {e}")
            
        # Resolve stream path
        stream_path = None
        try:
            if camera_manager is not None:
                cam = camera_manager.get_camera(camera_id)
                if cam is not None:
                    stream_path = getattr(cam, 'mediamtx_path', None) or getattr(cam, 'id', None)
            if not stream_path:
                try:
                    from app import cameras_db
                except Exception:
                    import json as _json
                    with open('cameras.json', 'r') as _f:
                        cameras_db = _json.load(_f)
                for cam in cameras_db:
                    if cam.get('id') == camera_id:
                        stream_path = cam.get('mediamtx_path') or cam.get('id')
                        break
        except Exception:
            pass

        if not stream_path:
            stream_path = camera_id
        
        # WHIP endpoint for publishing audio
        whip = f"{mediamtx_webrtc}/{stream_path}_backchannel/whip"
        
        return jsonify({ 
            'success': True, 
            'data': { 
                'whip': whip,
                'iceServers': [
                    { 'urls': 'stun:stun.l.google.com:19302' },
                    { 'urls': 'stun:stun1.l.google.com:19302' }
                ]
            } 
        })
    except Exception as e:
        logger.error(f"audio_whip_proxy error: {e}")
        return jsonify({ 'success': False, 'message': 'Failed to resolve WHIP' }), 500


@api_bp.route('/audio/clips', methods=['GET'])
def list_audio_clips():
    """List all uploaded audio clips."""
    try:
        clips_dir = _os.path.join('data', 'audio_clips')
        _os.makedirs(clips_dir, exist_ok=True)
        meta_path = _os.path.join('data', 'audio_clips_meta.json')
        meta: Dict[str, Any] = {}
        try:
            if _os.path.exists(meta_path):
                with open(meta_path, 'r') as _mf:
                    meta = json.load(_mf) or {}
        except Exception:
            meta = {}
        
        clips = []
        for filename in _os.listdir(clips_dir):
            if filename.endswith(('.mp3', '.wav', '.ogg', '.m4a')):
                filepath = _os.path.join(clips_dir, filename)
                stat = _os.stat(filepath)
                m = (meta.get(filename) if isinstance(meta, dict) else None) or {}
                clips.append({
                    'id': filename,
                    'name': filename.rsplit('.', 1)[0],
                    'filename': filename,
                    'size_bytes': stat.st_size,
                    'created_at': stat.st_mtime,
                    'tags': m.get('tags', []),
                    'camera_id': m.get('camera_id'),
                    'camera_name': m.get('camera_name'),
                })
        
        return jsonify({ 'success': True, 'clips': clips })
    except Exception as e:
        logger.error(f"list_audio_clips error: {e}")
        return jsonify({ 'success': False, 'message': str(e) }), 500


@api_bp.route('/audio/clips/<clip_id>/meta', methods=['GET'])
def get_audio_clip_meta(clip_id):
    try:
        meta_path = _os.path.join('data', 'audio_clips_meta.json')
        if not _os.path.exists(meta_path):
            return jsonify({ 'success': True, 'data': { 'clip_id': clip_id, 'tags': [] } })
        with open(meta_path, 'r') as f:
            meta = json.load(f) or {}
        rec = (meta.get(clip_id) if isinstance(meta, dict) else None) or { 'clip_id': clip_id, 'tags': [] }
        return jsonify({ 'success': True, 'data': rec })
    except Exception as e:
        logger.error(f"get_audio_clip_meta error: {e}")
        return jsonify({ 'success': False, 'message': 'Failed to get clip meta' }), 500


@api_bp.route('/audio/clips/<clip_id>/meta', methods=['PUT'])
def update_audio_clip_meta(clip_id):
    try:
        data = request.get_json() or {}
        tags = data.get('tags', [])
        if tags is None:
            tags = []
        tags = [str(t).strip() for t in (tags or []) if str(t).strip()]
        # de-dupe preserve order
        dedup = []
        seen = set()
        for t in tags:
            if t not in seen:
                dedup.append(t)
                seen.add(t)
        meta_path = _os.path.join('data', 'audio_clips_meta.json')
        _os.makedirs(_os.path.dirname(meta_path), exist_ok=True)
        try:
            meta = json.load(open(meta_path, 'r')) if _os.path.exists(meta_path) else {}
        except Exception:
            meta = {}
        if not isinstance(meta, dict):
            meta = {}
        rec = meta.get(clip_id, {}) if isinstance(meta.get(clip_id), dict) else {}
        rec['clip_id'] = clip_id
        rec['tags'] = dedup
        rec['updated_at'] = datetime.now().isoformat()
        meta[clip_id] = rec
        with open(meta_path, 'w') as f:
            json.dump(meta, f, indent=2)
        return jsonify({ 'success': True, 'data': rec })
    except Exception as e:
        logger.error(f"update_audio_clip_meta error: {e}")
        return jsonify({ 'success': False, 'message': 'Failed to update clip meta' }), 500


@api_bp.route('/audio/profiles', methods=['GET'])
def list_audio_profiles():
    try:
        if not audio_monitor:
            return jsonify({ 'success': False, 'message': 'Audio monitor not available' }), 503
        profiles = audio_monitor.profiles.list_profiles(include_embedding=False)
        return jsonify({ 'success': True, 'data': { 'profiles': profiles, 'count': len(profiles) } })
    except Exception as e:
        logger.error(f"list_audio_profiles error: {e}")
        return jsonify({ 'success': False, 'message': 'Failed to list profiles' }), 500


@api_bp.route('/audio/profiles', methods=['POST'])
def create_audio_profile():
    try:
        if not audio_monitor:
            return jsonify({ 'success': False, 'message': 'Audio monitor not available' }), 503
        data = request.get_json() or {}
        name = str(data.get('name') or '').strip()
        clip_id = str(data.get('clip_id') or '').strip()
        tags = data.get('tags', []) or []
        if not name:
            return jsonify({ 'success': False, 'message': 'name is required' }), 400
        if not clip_id:
            return jsonify({ 'success': False, 'message': 'clip_id is required' }), 400
        clip_path = _os.path.join('data', 'audio_clips', clip_id)
        if not _os.path.exists(clip_path):
            return jsonify({ 'success': False, 'message': 'clip not found' }), 404
        prof = audio_monitor.profiles.create_profile_from_clip(name=name, tags=tags, clip_path=clip_path, clip_id=clip_id)
        return jsonify({ 'success': True, 'data': prof })
    except Exception as e:
        logger.error(f"create_audio_profile error: {e}")
        return jsonify({ 'success': False, 'message': 'Failed to create profile' }), 500


@api_bp.route('/audio/profiles/<profile_id>', methods=['DELETE'])
def delete_audio_profile(profile_id):
    try:
        if not audio_monitor:
            return jsonify({ 'success': False, 'message': 'Audio monitor not available' }), 503
        ok = audio_monitor.profiles.delete_profile(profile_id)
        return jsonify({ 'success': ok, 'data': { 'deleted': ok, 'id': profile_id } })
    except Exception as e:
        logger.error(f"delete_audio_profile error: {e}")
        return jsonify({ 'success': False, 'message': 'Failed to delete profile' }), 500


@api_bp.route('/audio/monitor/status', methods=['GET'])
def audio_monitor_status():
    try:
        if not audio_monitor:
            return jsonify({ 'success': False, 'message': 'Audio monitor not available' }), 503
        return jsonify({ 'success': True, 'data': audio_monitor.status() })
    except Exception as e:
        logger.error(f"audio_monitor_status error: {e}")
        return jsonify({ 'success': False, 'message': 'Failed to get status' }), 500


@api_bp.route('/audio/monitor/start', methods=['POST'])
def audio_monitor_start():
    try:
        if not audio_monitor:
            return jsonify({ 'success': False, 'message': 'Audio monitor not available' }), 503
        data = request.get_json() or {}
        camera_id = str(data.get('camera_id') or '').strip()
        if not camera_id:
            return jsonify({ 'success': False, 'message': 'camera_id is required' }), 400
        # Optional config overrides
        cfg = data.get("config") or {}
        mon_cfg = None
        try:
            from core.audio_monitor import AudioMonitorConfig
            if isinstance(cfg, dict) and cfg:
                mon_cfg = AudioMonitorConfig(
                    amplitude_threshold=float(cfg.get("amplitude_threshold", 0.06)),
                    silence_timeout_s=float(cfg.get("silence_timeout_s", 0.8)),
                    min_event_s=float(cfg.get("min_event_s", 0.4)),
                    max_event_s=float(cfg.get("max_event_s", 12.0)),
                    pre_roll_s=float(cfg.get("pre_roll_s", 1.5)),
                    post_roll_s=float(cfg.get("post_roll_s", 0.6)),
                    match_min_similarity=float(cfg.get("match_min_similarity", 0.75)),
                    match_top_k=int(cfg.get("match_top_k", 3)),
                )
        except Exception:
            mon_cfg = None

        ok = audio_monitor.start_camera(camera_id, config=mon_cfg)
        return jsonify({ 'success': True, 'data': { 'started': ok, 'camera_id': camera_id } })
    except Exception as e:
        logger.error(f"audio_monitor_start error: {e}")
        return jsonify({ 'success': False, 'message': 'Failed to start monitor' }), 500


@api_bp.route('/audio/monitor/stop', methods=['POST'])
def audio_monitor_stop():
    try:
        if not audio_monitor:
            return jsonify({ 'success': False, 'message': 'Audio monitor not available' }), 503
        data = request.get_json() or {}
        camera_id = str(data.get('camera_id') or '').strip()
        if not camera_id:
            return jsonify({ 'success': False, 'message': 'camera_id is required' }), 400
        ok = audio_monitor.stop_camera(camera_id)
        return jsonify({ 'success': True, 'data': { 'stopped': ok, 'camera_id': camera_id } })
    except Exception as e:
        logger.error(f"audio_monitor_stop error: {e}")
        return jsonify({ 'success': False, 'message': 'Failed to stop monitor' }), 500


@api_bp.route('/audio/clips/upload', methods=['POST'])
def upload_audio_clip():
    """Upload an audio clip for playback."""
    try:
        if 'file' not in request.files:
            return jsonify({ 'success': False, 'message': 'No file provided' }), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({ 'success': False, 'message': 'No file selected' }), 400
        
        # Validate file extension
        allowed_extensions = {'.mp3', '.wav', '.ogg', '.m4a'}
        ext = _os.path.splitext(file.filename)[1].lower()
        if ext not in allowed_extensions:
            return jsonify({ 'success': False, 'message': 'Invalid file type' }), 400
        
        clips_dir = _os.path.join('data', 'audio_clips')
        _os.makedirs(clips_dir, exist_ok=True)
        
        # Save file
        filepath = _os.path.join(clips_dir, file.filename)
        file.save(filepath)
        
        stat = _os.stat(filepath)
        clip = {
            'id': file.filename,
            'name': file.filename.rsplit('.', 1)[0],
            'filename': file.filename,
            'size_bytes': stat.st_size,
            'created_at': stat.st_mtime
        }
        
        return jsonify({ 'success': True, 'clip': clip })
    except Exception as e:
        logger.error(f"upload_audio_clip error: {e}")
        return jsonify({ 'success': False, 'message': str(e) }), 500


@api_bp.route('/audio/clips/<clip_id>', methods=['DELETE'])
def delete_audio_clip(clip_id):
    """Delete an audio clip."""
    try:
        clips_dir = _os.path.join('data', 'audio_clips')
        filepath = _os.path.join(clips_dir, clip_id)
        
        if not _os.path.exists(filepath):
            return jsonify({ 'success': False, 'message': 'Clip not found' }), 404
        
        _os.remove(filepath)
        return jsonify({ 'success': True })
    except Exception as e:
        logger.error(f"delete_audio_clip error: {e}")
        return jsonify({ 'success': False, 'message': str(e) }), 500


@api_bp.route('/audio/clips/<clip_id>/stream', methods=['GET'])
def stream_audio_clip(clip_id):
    """Stream an audio clip."""
    try:
        clips_dir = _os.path.join('data', 'audio_clips')
        filepath = _os.path.join(clips_dir, clip_id)
        
        if not _os.path.exists(filepath):
            return jsonify({ 'success': False, 'message': 'Clip not found' }), 404
        
        return send_file(filepath)
    except Exception as e:
        logger.error(f"stream_audio_clip error: {e}")
        return jsonify({ 'success': False, 'message': str(e) }), 500


@api_bp.route('/status', methods=['GET'])
def system_status():
    """Get detailed system status"""
    try:
        status_data = {
            "system": {
                "status": "running",
                "uptime": "0s",  # Would calculate actual uptime
                "version": "2.1.4",
                "timestamp": datetime.now().isoformat()
            },
            "services": {},
            "statistics": {
                "cameras": {"total": 0, "online": 0, "offline": 0},
                "streams": {"active": 0, "total_connections": 0},
                "ai": {"models_loaded": 0, "processing_queue": 0},
                "alerts": {"total": 0, "unread": 0}
            }
        }

        # Service statuses
        services = status_data["services"]

        services["camera_manager"] = {
            "status": "available" if camera_manager else "unavailable",
            "running": getattr(camera_manager, '_running', False) if camera_manager else False
        }

        services["stream_server"] = {
            "status": "available" if stream_server else "unavailable",
            "running": getattr(stream_server, 'running', False) if stream_server else False
        }

        services["ai_analyzer"] = {
            "status": "available" if ai_analyzer else "unavailable",
            "running": getattr(ai_analyzer, 'running', True) if ai_analyzer else False
        }

        services["alert_system"] = {
            "status": "available" if alert_system else "unavailable",
            "running": getattr(alert_system, 'running', True) if alert_system else False
        }

        # Get statistics
        if camera_manager:
            all_cameras = safe_service_call(camera_manager, 'get_all_cameras', [])
            connected_cameras = safe_service_call(camera_manager, 'get_connected_cameras', [])

            status_data["statistics"]["cameras"] = {
                "total": len(all_cameras),
                "online": len(connected_cameras),
                "offline": len(all_cameras) - len(connected_cameras)
            }

        if stream_server:
            connections = safe_service_call(stream_server, 'get_client_connections', {})
            streams = safe_service_call(stream_server, 'get_all_streams', [])

            status_data["statistics"]["streams"] = {
                "active": len(streams),
                "total_connections": len(connections)
            }

        return jsonify({
            "success": True,
            "data": status_data
        })

    except Exception as e:
        logger.error(f"System status failed: {e}")
        return jsonify({
            "success": False,
            "message": f"Failed to get system status: {str(e)}"
        }), 500


# ==================== IMAGE GENERATION (OpenAI) ====================

@api_bp.route('/image/generate', methods=['POST'])
def generate_image():
    """Generate a transparent PNG birds-eye image using OpenAI Images API.

    Request JSON:
      { prompt: str, style?: str, width?: int, height?: int }
    Response JSON:
      { success: true, data: { imageDataUrl: 'data:image/png;base64,...' } }
    """
    try:
        data = request.get_json() or {}
        user_prompt = (data.get('prompt') or '').strip()
        style = (data.get('style') or '').strip().lower() or 'photoreal'
        width = int(data.get('width') or 1024)
        height = int(data.get('height') or 1024)

        if not user_prompt:
            return jsonify({ 'success': False, 'message': 'prompt is required' }), 400

        # Clamp to provider-allowed sizes (OpenAI supports squares and some sizes; we build WxH string)
        def clamp_size(w: int, h: int) -> str:
            try:
                w = max(64, min(2048, int(w)))
                h = max(64, min(2048, int(h)))
            except Exception:
                w, h = 1024, 1024
            return f"{w}x{h}"

        size = clamp_size(width, height)

        # Build style modifiers line
        style_map = {
            'vibrant': 'vibrant, bold colors, high contrast, saturated',
            'photoreal': 'photorealistic, orthophoto-like realism',
            'blueprint': 'technical drawing, blueprint style, white lines on blue, flat shading',
            'minimal': 'minimalist vector style, flat shading',
            'monochrome': 'grayscale, clean outlines',
            'night': 'nighttime, cool tones, emissive lights',
            'high-contrast': 'graphic, stark black/white, strong outlines',
            'default': ''
        }
        style_line = style_map.get(style, style_map.get('photoreal'))
        style_section = ("\n" + style_line) if style_line else ''

        # Prompt builder
        built_prompt = (
            "top-down, aerial, orthographic, overhead view of: " + user_prompt + "\n"
            "photorealistic, realistic materials and textures (metal, rubber, glass reflections), "
            "fine surface details (bolts, seams, tire treads), accurate scale and proportions, "
            "soft realistic shadows, ambient occlusion at contact points, crisp edges, high detail\n"
            "transparent background, clean cutout" + style_section
        )

        # Negative prompt (OpenAI Images API does not accept a separate negative prompt parameter; omit)

        # OpenAI client
        import os as _os
        from openai import OpenAI  # type: ignore

        api_key = _os.environ.get('OPENAI_API_KEY')
        if not api_key:
            return jsonify({ 'success': False, 'message': 'OPENAI_API_KEY not configured' }), 503

        client = OpenAI(api_key=api_key)

        # Call Images API
        try:
            result = client.images.generate(
                model='gpt-image-1',
                prompt=built_prompt,
                size=size,
                background='transparent',
                quality='high',
                n=1
            )
        except Exception as e:
            logger.error(f"OpenAI image generation failed: {e}")
            return jsonify({ 'success': False, 'message': f'Image generation failed: {str(e)}' }), 502

        # Extract base64 PNG
        try:
            image_b64 = result.data[0].b64_json  # type: ignore[attr-defined]
            if not image_b64:
                return jsonify({ 'success': False, 'message': 'No image data returned' }), 502
        except Exception:
            return jsonify({ 'success': False, 'message': 'Unexpected response from provider' }), 502

        data_url = f"data:image/png;base64,{image_b64}"
        return jsonify({ 'success': True, 'data': { 'imageDataUrl': data_url } })

    except Exception as e:
        logger.error(f"/image/generate error: {e}")
        return jsonify({ 'success': False, 'message': 'Failed to generate image' }), 500


# ==================== CAMERA MANAGEMENT ENDPOINTS ====================

@api_bp.route('/cameras', methods=['GET'])
def get_cameras():
    """Get all cameras with filtering options - simplified to use cameras.json as single source of truth"""
    try:
        # Get query parameters
        camera_type = request.args.get('type')
        status = request.args.get('status')
        location = request.args.get('location')
        include_health = request.args.get('include_health', 'false').lower() == 'true'

        cameras_data = []

        # Prefer data/cameras.json (desktop edits) and fall back to cameras.json (repo root)
        try:
            import json
            cameras_path = 'data/cameras.json' if os.path.exists('data/cameras.json') else 'cameras.json'
            with open(cameras_path, 'r') as f:
                all_cameras = json.load(f)
            
            logger.info(f"📂 Loaded {len(all_cameras)} cameras from {cameras_path}")
            
            for camera in all_cameras:
                # Apply filters
                if camera_type and camera.get('type') != camera_type:
                    continue
                if status and camera.get('status') != status:
                    continue
                if location and camera.get('location') != location:
                    continue

                cameras_data.append(camera)
                
        except Exception as e:
            logger.error(f"Error loading cameras from cameras.json: {e}")
            cameras_data = []

        # Return response
        response_data = {
            "devices": cameras_data,
            "total_count": len(cameras_data),
            "by_type": {"camera": len(cameras_data), "controller": 0, "sensor": 0},
            "by_status": {
                "online": len([c for c in cameras_data if c.get('status') == 'online']),
                "offline": len([c for c in cameras_data if c.get('status') == 'offline']),
                "error": 0,
                "maintenance": 0
            }
        }

        return jsonify({
            "success": True,
            "data": response_data,
            "message": f"Retrieved {len(cameras_data)} cameras"
        })

    except Exception as e:
        logger.error(f"Failed to get cameras: {e}")
        return jsonify({
            "success": False,
            "message": f"Failed to get cameras: {str(e)}"
        }), 500


@api_bp.route('/cameras/<camera_id>', methods=['GET'])
def get_camera(camera_id):
    """Get specific camera details"""
    try:
        if not camera_manager:
            return jsonify({
                "success": False,
                "message": "Camera manager not available"
            }), 503

        camera = safe_service_call(camera_manager, 'get_camera', None, camera_id)

        if not camera:
            return jsonify({
                "success": False,
                "message": f"Camera {camera_id} not found"
            }), 404

        # Convert camera object to dict if needed
        camera_dict = camera if isinstance(camera, dict) else {
            'id': getattr(camera, 'id', camera_id),
            'name': getattr(camera, 'name', f'Camera {camera_id[:8]}'),
            'type': 'camera',
            'location': getattr(camera, 'location', 'Unknown'),
            'enabled': getattr(camera, 'enabled', True),
            'rtsp_url': getattr(camera, 'rtsp_url', ''),
            'webrtc_enabled': getattr(camera, 'webrtc_enabled', False),
            'ip_address': getattr(camera, 'ip_address', ''),
            'username': getattr(camera, 'username', ''),
            'created_at': getattr(camera, 'created_at', datetime.now().isoformat()),
            'updated_at': getattr(camera, 'updated_at', datetime.now().isoformat())
        }

        # Get connection status
        connected_cameras = safe_service_call(camera_manager, 'get_connected_cameras', [])
        connected_ids = [c.get('id', c) for c in connected_cameras] if connected_cameras else []
        camera_dict['status'] = 'online' if camera_id in connected_ids else 'offline'

        return jsonify({
            "success": True,
            "data": camera_dict
        })

    except Exception as e:
        logger.error(f"Failed to get camera {camera_id}: {e}")
        return jsonify({
            "success": False,
            "message": f"Failed to get camera: {str(e)}"
        }), 500


@api_bp.route('/cameras', methods=['OPTIONS'])
def handle_preflight():
    """Handle CORS preflight requests"""
    response = Response()
    response.headers.add("Access-Control-Allow-Origin", "*")
    response.headers.add("Access-Control-Allow-Headers", "*")
    response.headers.add("Access-Control-Allow-Methods", "*")
    return response


@api_bp.route('/cameras', methods=['POST'])
def add_camera():
    """Add a new camera"""
    try:
        if not request.json:
            return jsonify({
                "success": False,
                "message": "JSON data required"
            }), 400

        camera_data = request.json

        # Validate required fields
        required_fields = ['name', 'rtsp_url']
        for field in required_fields:
            if field not in camera_data:
                return jsonify({
                    "success": False,
                    "message": f"Missing required field: {field}"
                }), 400

        if not camera_manager:
            return jsonify({
                "success": False,
                "message": "Camera manager not available"
            }), 503

        # Add default values
        camera_data.setdefault('type', 'camera')
        camera_data.setdefault('location', 'Default')
        camera_data.setdefault('enabled', True)
        camera_data.setdefault('webrtc_enabled', True)
        camera_data.setdefault('ai_analysis', False)
        camera_data.setdefault('recording', False)
        camera_data.setdefault('ptz_enabled', False)
        camera_data.setdefault('stream_quality', 'medium')
        camera_data.setdefault('motion_detection', True)
        camera_data.setdefault('audio_enabled', False)
        camera_data.setdefault('night_vision', False)
        camera_data.setdefault('privacy_mask', False)
        camera_data.setdefault('created_at', datetime.now().isoformat())
        camera_data.setdefault('updated_at', datetime.now().isoformat())

        # Enforce entitlement limit on enabled cameras
        if camera_data.get("enabled", True):
            try:
                current_enabled = 0
                if camera_manager and hasattr(camera_manager, "cameras"):
                    current_enabled = sum(
                        1 for cfg in camera_manager.cameras.values() if getattr(cfg, "enabled", True)
                    )
                limit = int(get_camera_limit())
                if current_enabled >= limit:
                    return jsonify({
                        "success": False,
                        "message": f"Camera limit reached ({limit}). Disable a camera to stay within the beta limit."
                    }), 403
            except Exception:
                pass

        # Add camera via camera manager
        added_camera = safe_service_call(camera_manager, 'add_camera', None, camera_data)

        if added_camera is None:
            # Fallback: create a camera dict with generated ID
            import uuid
            added_camera = {
                **camera_data,
                'id': str(uuid.uuid4())[:8]
            }

        return jsonify({
            "success": True,
            "data": added_camera,
            "message": f"Camera '{camera_data['name']}' added successfully"
        })

    except Exception as e:
        logger.error(f"Failed to add camera: {e}")
        return jsonify({
            "success": False,
            "message": f"Failed to add camera: {str(e)}"
        }), 500


@api_bp.route('/cameras/<camera_id>/auto-detect', methods=['POST'])
def auto_detect_camera(camera_id):
    """Auto-detect camera configuration"""
    try:
        if not request.json:
            return jsonify({
                "success": False,
                "message": "JSON data required with IP address"
            }), 400

        ip_address = request.json.get('ip_address')
        if not ip_address:
            return jsonify({
                "success": False,
                "message": "IP address is required"
            }), 400

        # Extract IP if it's a URL
        actual_ip = extract_ip_from_url(ip_address)

        # Detect manufacturer
        manufacturer = detect_manufacturer(actual_ip)

        # Test RTSP connection
        username = request.json.get('username', 'admin')
        password = request.json.get('password', '')
        rtsp_working = test_rtsp_connection(actual_ip, username, password)

        # Check PTZ support
        ptz_support = check_ptz_support(actual_ip, username, password)

        detection_result = {
            "ip_address": actual_ip,
            "manufacturer": manufacturer,
            "rtsp_working": rtsp_working,
            "ptz_support": ptz_support,
            "suggested_config": {
                "name": f"{manufacturer.title()} Camera ({actual_ip})",
                "rtsp_url": f"rtsp://{username}:{password}@{actual_ip}:554/Streaming/Channels/101",
                "manufacturer": manufacturer,
                "webrtc_enabled": True,
                "ptz_enabled": ptz_support
            }
        }

        return jsonify({
            "success": True,
            "data": detection_result,
            "message": "Camera auto-detection completed"
        })

    except Exception as e:
        logger.error(f"Auto-detection failed: {e}")
        return jsonify({
            "success": False,
            "message": f"Auto-detection failed: {str(e)}"
        }), 500


def extract_ip_from_url(url_or_ip):
    """Extract IP address from URL or return IP if already an IP"""
    if url_or_ip.startswith('http'):
        from urllib.parse import urlparse
        parsed = urlparse(url_or_ip)
        return parsed.hostname
    return url_or_ip


def detect_manufacturer(ip_address):
    """Detect camera manufacturer based on IP probing"""
    try:
        import requests

        # Try common manufacturer detection endpoints
        manufacturers = {
            'hikvision': ['digest', 'web', 'ISAPI'],
            'dahua': ['cgi-bin', 'cam'],
            'axis': ['axis-cgi', 'vapix']
        }

        for manufacturer, endpoints in manufacturers.items():
            for endpoint in endpoints:
                try:
                    response = requests.get(
                        f"http://{ip_address}/{endpoint}",
                        timeout=2
                    )
                    if response.status_code != 404:
                        return manufacturer
                except:
                    continue

        return 'generic'

    except Exception:
        return 'unknown'


def test_rtsp_connection(ip_address, username, password):
    """Test RTSP connection to camera"""
    try:
        import socket

        # Test if RTSP port (554) is open
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(3)
        result = sock.connect_ex((ip_address, 554))
        sock.close()

        return result == 0

    except Exception:
        return False


def check_ptz_support(ip_address, username, password):
    """Check if camera supports PTZ controls"""
    try:
        import requests

        # Try common PTZ endpoints
        ptz_endpoints = [
            f"http://{ip_address}/cgi-bin/ptz.cgi",
            f"http://{ip_address}/axis-cgi/com/ptz.cgi",
            f"http://{ip_address}/ISAPI/PTZCtrl"
        ]

        for endpoint in ptz_endpoints:
            try:
                response = requests.get(
                    endpoint,
                    auth=(username, password),
                    timeout=2
                )
                if response.status_code in [200, 401]:  # 401 means endpoint exists but needs auth
                    return True
            except:
                continue

        return False

    except Exception:
        return False


@api_bp.route('/cameras/<camera_id>', methods=['PUT'])
def update_camera(camera_id):
    """Update camera configuration"""
    try:
        if not request.json:
            return jsonify({
                "success": False,
                "message": "JSON data required"
            }), 400

        if not camera_manager:
            return jsonify({
                "success": False,
                "message": "Camera manager not available"
            }), 503

        update_data = request.json
        update_data['updated_at'] = datetime.now().isoformat()

        # Update camera via camera manager
        updated_camera = safe_service_call(camera_manager, 'update_camera', None, camera_id, update_data)

        if updated_camera is None:
            return jsonify({
                "success": False,
                "message": f"Camera {camera_id} not found or update failed"
            }), 404

        return jsonify({
            "success": True,
            "data": updated_camera,
            "message": f"Camera {camera_id} updated successfully"
        })

    except Exception as e:
        logger.error(f"Failed to update camera {camera_id}: {e}")
        return jsonify({
            "success": False,
            "message": f"Failed to update camera: {str(e)}"
        }), 500


@api_bp.route('/cameras/<camera_id>', methods=['DELETE'])
def delete_camera(camera_id):
    """Delete a camera"""
    try:
        if not camera_manager:
            return jsonify({
                "success": False,
                "message": "Camera manager not available"
            }), 503

        # Delete camera via camera manager
        success = safe_service_call(camera_manager, 'delete_camera', False, camera_id)

        if not success:
            return jsonify({
                "success": False,
                "message": f"Camera {camera_id} not found or delete failed"
            }), 404

        return jsonify({
            "success": True,
            "message": f"Camera {camera_id} deleted successfully"
        })

    except Exception as e:
        logger.error(f"Failed to delete camera {camera_id}: {e}")
        return jsonify({
            "success": False,
            "message": f"Failed to delete camera: {str(e)}"
        }), 500


# ==================== CAMERA CONTROL ENDPOINTS ====================

@api_bp.route('/cameras/<camera_id>/connect', methods=['POST'])
def connect_camera(camera_id):
    """Connect to a camera"""
    try:
        if not camera_manager:
            return jsonify({
                "success": False,
                "message": "Camera manager not available"
            }), 503

        # Connect camera
        success = safe_service_call(camera_manager, 'connect_camera', False, camera_id)

        if not success:
            return jsonify({
                "success": False,
                "message": f"Failed to connect to camera {camera_id}"
            }), 400

        return jsonify({
            "success": True,
            "message": f"Camera {camera_id} connected successfully"
        })

    except Exception as e:
        logger.error(f"Failed to connect camera {camera_id}: {e}")
        return jsonify({
            "success": False,
            "message": f"Failed to connect camera: {str(e)}"
        }), 500


@api_bp.route('/cameras/<camera_id>/disconnect', methods=['POST'])
def disconnect_camera(camera_id):
    """Disconnect from a camera"""
    try:
        if not camera_manager:
            return jsonify({
                "success": False,
                "message": "Camera manager not available"
            }), 503

        # Disconnect camera
        success = safe_service_call(camera_manager, 'disconnect_camera', False, camera_id)

        return jsonify({
            "success": True,
            "message": f"Camera {camera_id} disconnected successfully"
        })

    except Exception as e:
        logger.error(f"Failed to disconnect camera {camera_id}: {e}")
        return jsonify({
            "success": False,
            "message": f"Failed to disconnect camera: {str(e)}"
        }), 500


@api_bp.route('/cameras/<camera_id>/test', methods=['POST'])
def test_camera_connection(camera_id):
    """Test camera connection"""
    try:
        if not camera_manager:
            return jsonify({
                "success": False,
                "message": "Camera manager not available"
            }), 503

        # Test camera connection
        test_result = safe_service_call(camera_manager, 'test_camera_connection',
                                        {'success': False, 'message': 'Test failed'}, camera_id)

        if isinstance(test_result, bool):
            test_result = {
                'success': test_result,
                'message': 'Connection test passed' if test_result else 'Connection test failed'
            }

        return jsonify({
            "success": test_result.get('success', False),
            "data": test_result,
            "message": test_result.get('message', 'Connection test completed')
        })

    except Exception as e:
        logger.error(f"Failed to test camera {camera_id}: {e}")
        return jsonify({
            "success": False,
            "message": f"Failed to test camera: {str(e)}"
        }), 500


def _resolve_camera_for_ptz(camera_id: str) -> Dict[str, Any]:
    """
    Build a PTZ config dict from the existing camera record.

    The desktop client only sends `{action, params, brand_hint?}`; the
    backend hydrates IP, username, password, manufacturer (and the
    Tapo cloud password from `core.ptz_credentials`) here.
    """
    cfg: Dict[str, Any] = {'camera_id': camera_id}

    cam = None
    if camera_manager is not None:
        try:
            cam = camera_manager.get_camera(camera_id)
        except Exception:
            cam = None

    if cam is not None:
        for src_attr, dst_key in (
            ('ip_address', 'ip_address'),
            ('username', 'username'),
            ('password', 'password'),
            ('manufacturer', 'manufacturer'),
            ('rtsp_url', 'rtsp_url'),
            ('ptz_url', 'custom_ptz_url'),
            ('port', 'port'),
        ):
            value = getattr(cam, src_attr, None)
            if value is not None:
                cfg[dst_key] = value
    else:
        try:
            try:
                from app import cameras_db  # type: ignore
            except Exception:
                with open('cameras.json', 'r') as fh:
                    cameras_db = json.load(fh)
            for entry in cameras_db or []:
                if entry.get('id') == camera_id or entry.get('camera_id') == camera_id:
                    cfg['ip_address'] = entry.get('ip_address') or entry.get('ip')
                    cfg['username'] = entry.get('username')
                    cfg['password'] = entry.get('password')
                    cfg['manufacturer'] = entry.get('manufacturer')
                    cfg['rtsp_url'] = entry.get('rtsp_url')
                    if entry.get('ptz_url'):
                        cfg['custom_ptz_url'] = entry.get('ptz_url')
                    if entry.get('port'):
                        cfg['port'] = entry.get('port')
                    break
        except Exception as exc:
            logger.debug("Could not load camera %s from cameras.json: %s", camera_id, exc)

    if not cfg.get('ip_address') and cfg.get('rtsp_url'):
        try:
            from urllib.parse import urlparse
            parsed = urlparse(str(cfg['rtsp_url']))
            if parsed.hostname:
                cfg['ip_address'] = parsed.hostname
        except Exception:
            pass

    try:
        from core import ptz_credentials
        creds = ptz_credentials.get(camera_id) or {}
        if creds.get('tapo_cloud_password'):
            cfg['tapo_cloud_password'] = creds['tapo_cloud_password']
    except Exception:
        pass

    return cfg


@api_bp.route('/cameras/<camera_id>/ptz', methods=['POST'])
def control_ptz(camera_id):
    """
    Control camera PTZ. The body is `{action, params?, brand_hint?}`.
    All credentials and connection info are resolved server-side from
    the camera record + `core.ptz_credentials`.
    """
    try:
        if not request.json:
            return jsonify({"success": False, "message": "JSON data required"}), 400

        body = request.json
        action = body.get('action') or ''
        if not action:
            return jsonify({"success": False, "message": "action is required"}), 400

        params = body.get('params')
        if not isinstance(params, dict):
            # Back-compat: a few callers still send flat top-level params.
            params = {k: v for k, v in body.items()
                      if k not in ('action', 'params', 'brand_hint')}

        config = _resolve_camera_for_ptz(camera_id)
        brand_hint = body.get('brand_hint') or body.get('brand') or body.get('camera_brand')
        if brand_hint:
            config['brand_hint'] = brand_hint

        from core.ptz_manager import get_ptz_manager
        ptz_manager = get_ptz_manager()
        result = _run_coro_safe(
            ptz_manager.execute_command(camera_id, config, action, params or {})
        )

        return jsonify({
            "success": bool(result.get('success', False)),
            "data": result,
            "message": result.get('message') or result.get('error') or 'PTZ command executed',
        })

    except Exception as e:
        logger.error(f"PTZ control failed for camera {camera_id}: {e}")
        return jsonify({
            "success": False,
            "message": f"PTZ control failed: {str(e)}"
        }), 500


@api_bp.route('/cameras/<camera_id>/ptz/probe', methods=['POST'])
def probe_ptz(camera_id):
    """
    Detect which PTZ protocol a camera speaks. Returns
    `{protocol_resolved, needs_credentials, brand_guess}` so the desktop
    client knows whether to prompt for a Tapo cloud password before
    showing the controller.
    """
    try:
        body = request.json or {}
        config = _resolve_camera_for_ptz(camera_id)
        brand_hint = body.get('brand_hint') or body.get('brand') or body.get('camera_brand')
        if brand_hint:
            config['brand_hint'] = brand_hint

        from core.ptz_manager import get_ptz_manager
        ptz_manager = get_ptz_manager()
        result = _run_coro_safe(ptz_manager.probe(camera_id, config))

        return jsonify({
            "success": True,
            "data": result,
            "message": result.get('error') or 'PTZ probe completed',
        })
    except Exception as e:
        logger.error(f"PTZ probe failed for camera {camera_id}: {e}")
        return jsonify({"success": False, "message": f"PTZ probe failed: {str(e)}"}), 500


@api_bp.route('/cameras/<camera_id>/ptz/test', methods=['POST'])
def test_ptz(camera_id):
    """
    Run a one-shot PTZ probe with optional ad-hoc credentials WITHOUT
    saving them. Used by the credentials dialog's "Test connection"
    button so the user can validate inputs before saving.

    Body (all optional): {
        protocol_override, tapo_cloud_password,
        tapo_local_username, tapo_local_password, onvif_port, brand_hint
    }
    """
    try:
        body = request.json or {}
        config = _resolve_camera_for_ptz(camera_id)

        # Apply ad-hoc overrides from the body for THIS test only.
        if body.get('tapo_cloud_password'):
            config['tapo_cloud_password'] = body['tapo_cloud_password']
        if body.get('tapo_local_username'):
            config['username'] = body['tapo_local_username']
        if body.get('tapo_local_password'):
            config['password'] = body['tapo_local_password']
        if body.get('brand_hint'):
            config['brand_hint'] = body['brand_hint']
        protocol_override = (body.get('protocol_override') or '').lower().strip()

        # Force the manager to honor the override without writing creds.
        if protocol_override == 'tapo':
            config['brand_hint'] = 'tapo'
        elif protocol_override == 'onvif':
            # Bypass tapo by clearing the cloud pw for this probe.
            config.pop('tapo_cloud_password', None)
            config['brand_hint'] = 'onvif'
        elif protocol_override == 'generic':
            config.pop('tapo_cloud_password', None)
            config['brand_hint'] = 'generic'

        from core.ptz_manager import get_ptz_manager
        ptz_manager = get_ptz_manager()
        # Test must bypass any sticky 'auth-failed' / lockout state, AND
        # any cached controller from previous wrong creds — but we don't
        # touch _resolution_cache (it's just a protocol hint). Each test
        # is a single login attempt + getBasicInfo.
        try:
            ptz_manager._stuck_state.pop(camera_id, None)
            old = ptz_manager.controllers.pop(camera_id, None)
            ptz_manager.protocols.pop(camera_id, None)
            if old is not None:
                try:
                    disc = getattr(old, 'disconnect', None)
                    if disc and not asyncio.iscoroutinefunction(disc):
                        disc()
                except Exception:
                    pass
        except Exception:
            pass

        result = _run_coro_safe(ptz_manager.probe(camera_id, config))

        # Translate result into a simple OK/error for the dialog.
        ok = bool(result.get('protocol_resolved')) and not result.get('error') and not result.get('needs_credentials')
        return jsonify({
            "success": ok,
            "data": result,
            "message": result.get('error')
                or (f"OK: connected via {result.get('protocol_resolved')}" if ok
                    else "Test failed"),
        })
    except Exception as e:
        logger.error(f"PTZ test failed for camera {camera_id}: {e}")
        return jsonify({"success": False, "message": f"PTZ test failed: {str(e)}"}), 500


@api_bp.route('/cameras/<camera_id>/ptz/credentials', methods=['GET'])
def get_ptz_credentials(camera_id):
    """Return any stored PTZ credentials so the credentials dialog can pre-fill."""
    try:
        from core import ptz_credentials
        creds = ptz_credentials.get(camera_id) or {}
        return jsonify({"success": True, "data": creds})
    except Exception as e:
        logger.error(f"PTZ credentials fetch failed for camera {camera_id}: {e}")
        return jsonify({"success": False, "message": f"Fetch failed: {str(e)}"}), 500


@api_bp.route('/cameras/<camera_id>/ptz/credentials', methods=['POST'])
def set_ptz_credentials(camera_id):
    """
    Store an extra PTZ credential for a camera (e.g. Tapo cloud password).
    Body: `{key, value, persist}`. By default secrets live in process
    memory only; `persist=true` writes to `data/ptz_credentials.json`
    (chmod 0600). The desktop prompt surfaces the trade-off.
    """
    try:
        if not request.json:
            return jsonify({"success": False, "message": "JSON data required"}), 400

        body = request.json
        key = (body.get('key') or '').strip()
        value = body.get('value', '')
        persist = bool(body.get('persist', False))

        if not key:
            return jsonify({"success": False, "message": "key is required"}), 400

        from core import ptz_credentials
        from core.ptz_manager import get_ptz_manager

        ptz_credentials.set(camera_id, key, value, persist=persist)
        # Discard any stale Tapo controller / stuck state so the new creds
        # take effect on the very next command — but DO NOT invalidate
        # the resolution cache (that would force a fresh probe and
        # therefore an extra login, eating into Tapo's 5-attempts-per-hour
        # budget). The cached resolution still says 'tapo'; the next
        # command will build one fresh controller with the new creds.
        try:
            mgr = get_ptz_manager()
            mgr._stuck_state.pop(camera_id, None)
            old = mgr.controllers.pop(camera_id, None)
            mgr.protocols.pop(camera_id, None)
            if old is not None:
                try:
                    disc = getattr(old, 'disconnect', None)
                    if disc and not asyncio.iscoroutinefunction(disc):
                        disc()
                except Exception:
                    pass
        except Exception:
            pass

        return jsonify({
            "success": True,
            "message": "Credentials saved (session)" if not persist else "Credentials saved (disk)",
            "data": {"persisted": persist},
        })
    except Exception as e:
        logger.error(f"PTZ credentials save failed for camera {camera_id}: {e}")
        return jsonify({"success": False, "message": f"Save failed: {str(e)}"}), 500


@api_bp.route('/cameras/<camera_id>/ptz/credentials', methods=['DELETE'])
def clear_ptz_credentials(camera_id):
    """Forget all stored PTZ credentials for a camera (session + disk)."""
    try:
        from core import ptz_credentials
        from core.ptz_manager import get_ptz_manager

        ptz_credentials.clear(camera_id)
        try:
            get_ptz_manager().invalidate_resolution(camera_id)
        except Exception:
            pass
        return jsonify({"success": True, "message": "Credentials cleared"})
    except Exception as e:
        logger.error(f"PTZ credentials clear failed for camera {camera_id}: {e}")
        return jsonify({"success": False, "message": f"Clear failed: {str(e)}"}), 500


# ==================== AUTO-RECOVERY ENDPOINTS ====================

@api_bp.route('/recovery/status', methods=['GET'])
def get_recovery_status():
    """Get camera auto-recovery status"""
    try:
        # Get auto-recovery instance from app context
        auto_recovery = getattr(current_app, 'auto_recovery', None)
        
        # If not in app context, try to get from app module
        if not auto_recovery:
            try:
                import app
                auto_recovery = getattr(app, 'AUTO_RECOVERY_GLOBAL', None)
            except:
                pass
        
        if not auto_recovery:
            return jsonify({
                "success": False,
                "message": "Auto-recovery system not available"
            }), 503
            
        status = auto_recovery.get_recovery_status()
        
        return jsonify({
            "success": True,
            "data": status,
            "message": "Recovery status retrieved"
        })
        
    except Exception as e:
        logger.error(f"Failed to get recovery status: {e}")
        return jsonify({
            "success": False,
            "message": f"Failed to get recovery status: {str(e)}"
        }), 500

@api_bp.route('/recovery/trigger/<camera_id>', methods=['POST'])
def trigger_manual_recovery(camera_id):
    """Manually trigger recovery for a specific camera"""
    try:
        auto_recovery = getattr(current_app, 'auto_recovery', None)
        
        # If not in app context, try to get from app module
        if not auto_recovery:
            try:
                import app
                auto_recovery = getattr(app, 'AUTO_RECOVERY_GLOBAL', None)
            except:
                pass
        
        if not auto_recovery:
            return jsonify({
                "success": False,
                "message": "Auto-recovery system not available"
            }), 503
            
        # Get camera config
        camera_config = safe_service_call(camera_manager, 'get_camera', None, camera_id) if camera_manager else None
        
        if not camera_config:
            return jsonify({
                "success": False,
                "message": f"Camera {camera_id} not found"
            }), 404
            
        # Force a recovery attempt
        auto_recovery._attempt_camera_recovery(camera_id, camera_config)
        
        return jsonify({
            "success": True,
            "message": f"Manual recovery triggered for camera {camera_id}"
        })
        
    except Exception as e:
        logger.error(f"Failed to trigger manual recovery for {camera_id}: {e}")
        return jsonify({
            "success": False,
            "message": f"Failed to trigger recovery: {str(e)}"
        }), 500

# ==================== STREAMING ENDPOINTS ====================

@api_bp.route('/cameras/<camera_id>/stream', methods=['GET'])
def get_video_stream(camera_id):
    """Get video stream for camera"""
    try:
        if not camera_manager:
            return jsonify({
                "success": False,
                "message": "Camera manager not available"
            }), 503

        stream_format = request.args.get('format', 'mjpeg')
        quality = request.args.get('quality', 'medium')

        # Get stream info
        # Start stream if needed and return an MJPEG endpoint if requested
        stream_info = safe_service_call(camera_manager, 'get_stream_info', None, camera_id, stream_format, quality)

        if not stream_info:
            return jsonify({
                "success": False,
                "message": f"Stream not available for camera {camera_id}"
            }), 404

        # If MJPEG requested, provide direct endpoint
        if stream_format.lower() == 'mjpeg':
            # Ensure underlying capture is running
            if camera_manager and hasattr(camera_manager, 'get_camera'):
                cam = safe_service_call(camera_manager, 'get_camera', None, camera_id)
                # Support object or dict
                def _get(cam_obj, key, default=None):
                    if isinstance(cam_obj, dict):
                        return cam_obj.get(key, default)
                    try:
                        return getattr(cam_obj, key)
                    except Exception:
                        return default
                rtsp_url = _get(cam, 'rtsp_url')
                if rtsp_url:
                    try:
                        # Use globally-initialized stream_server from this module
                        if stream_server and camera_id not in getattr(stream_server, 'active_streams', {}):
                            _run_coro_safe(stream_server.start_stream(camera_id, { 'rtsp_url': rtsp_url, 'webrtc_enabled': False }))
                    except Exception:
                        pass

                # Best-effort: wait briefly for first frame to be available to avoid black image on first paint
                try:
                    import time
                    from core.stream_server import StreamQuality
                    start = time.time()
                    while time.time() - start < 1.0:
                        b = stream_server.get_frame(camera_id, quality=StreamQuality.MEDIUM, format='jpeg') if stream_server else None
                        if b:
                            break
                        time.sleep(0.05)
                except Exception:
                    pass

            # Return an MJPEG endpoint URL to consume
            return jsonify({
                "success": True,
                "data": {
                    **(stream_info or {}),
                    "stream_url": f"/api/cameras/{camera_id}/stream/mjpeg"
                },
                "message": "Stream info retrieved"
            })

        return jsonify({
            "success": True,
            "data": stream_info,
            "message": "Stream info retrieved"
        })
    except Exception as e:
        logger.error(f"Failed to get stream for camera {camera_id}: {e}")
        return jsonify({
            "success": False,
            "message": f"Failed to get stream: {str(e)}"
        }), 500
@api_bp.route('/cameras/<camera_id>/stream/mjpeg', methods=['GET'])
def mjpeg_stream(camera_id: str):
    """Serve simple MJPEG stream from latest processed frames in StreamServer."""
    
    def create_placeholder(text):
        try:
            import cv2
            import numpy as np
            placeholder = np.zeros((360, 640, 3), dtype=np.uint8)
            
            # Add red border for error
            cv2.rectangle(placeholder, (0,0), (640,360), (0,0,255), 4)
            
            # Split text into lines
            y = 180
            for line in text.split('\n')[:3]: # Max 3 lines
                cv2.putText(placeholder, line, (50, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                y += 40
            
            timestamp = datetime.now().strftime("%H:%M:%S")
            cv2.putText(placeholder, timestamp, (50, 340), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
            
            _, jpeg = cv2.imencode('.jpg', placeholder)
            return jpeg.tobytes()
        except Exception:
            return b''

    def error_response(message):
        logger.error(f"MJPEG Error for {camera_id}: {message}")
        def generate_error():
            frame = create_placeholder(f"Stream Error:\n{message}")
            if frame:
                # Yield single frame then sleep to keep connection open (prevents spam)
                yield (b"--frame\r\n"
                       b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")
                while True:
                    time.sleep(30)
        return Response(generate_error(), mimetype='multipart/x-mixed-replace; boundary=frame')

    try:
        if not stream_server:
            return error_response("Stream server not available")

        # Ensure the stream is started for this camera
        if camera_id not in getattr(stream_server, 'active_streams', {}):
            # Simplified camera lookup - use cameras.json as primary source
            try:
                resolved_id = camera_id
                rtsp_url = None
                cam = None

                # Load directly from cameras.json (primary source)
                try:
                    import json
                    with open('cameras.json', 'r') as f:
                        cameras_db = json.load(f)
                    
                    for camera in cameras_db:
                        if camera.get('id') == camera_id:
                            cam = camera
                            resolved_id = camera_id
                            logger.info(f"🚀 Auto-starting stream for MJPEG request: {camera_id}")
                            logger.info(f"📂 Found camera {camera_id} in cameras.json")
                            break
                except Exception as e:
                    logger.warning(f"Failed to load cameras.json: {e}")

                # If not found in cameras.json, try camera manager as fallback
                if not cam and camera_manager:
                    cam = safe_service_call(camera_manager, 'get_camera', None, resolved_id)
                    if cam:
                        logger.info(f"📡 Found camera {camera_id} via camera manager")

                # Extract RTSP URL and MediaMTX path
                path_name = None
                if cam is not None:
                    rtsp_url = cam.get('rtsp_url') if isinstance(cam, dict) else getattr(cam, 'rtsp_url', None)
                    path_name = cam.get('mediamtx_path', camera_id) if isinstance(cam, dict) else getattr(cam, 'mediamtx_path', camera_id)
                    logger.info(f"🔗 RTSP URL for {camera_id}: {rtsp_url}")

                if not rtsp_url:
                    # Only log errors for known cameras to reduce spam
                    if cam:
                        logger.error(f"❌ No RTSP URL available for camera '{camera_id}'")
                        # Try to get MediaMTX RTSP URL as fallback
                        mediamtx_rtsp = cam.get('mediamtx_rtsp_url') if isinstance(cam, dict) else getattr(cam, 'mediamtx_rtsp_url', None)
                        if mediamtx_rtsp:
                            rtsp_url = mediamtx_rtsp
                            logger.info(f"🔄 Using MediaMTX RTSP fallback for {camera_id}: {rtsp_url}")
                        else:
                            return jsonify({ 'success': False, 'message': f'Camera {camera_id} not configured or missing rtsp_url' }), 404
                    else:
                        # Silently reject unknown camera IDs to reduce log spam
                        return jsonify({ 'success': False, 'message': f'Camera not found' }), 404

                # Try to proxy through MediaMTX HLS first (much more efficient than MJPEG)
                # This redirects the client browser to the HLS proxy endpoint
                # which is handled by proxy_routes.py
                try:
                    from app import mediamtx  # type: ignore
                    # Verify MediaMTX path exists/is ready - silent check
                    # If path_name exists, we assume it's configured or will be auto-configured
                    if path_name and mediamtx.test_connection(silent=True):
                        # Check if path is actually ready or create it
                        path_info = mediamtx.get_path_info(path_name)
                        if not path_info.get('ready'):
                             # Create/Update path if not ready
                             mediamtx.create_path(path_name, rtsp_url)
                        
                        # Redirect to HLS proxy
                        return redirect(f"/proxy/hls/{path_name}/index.m3u8")
                except Exception:
                    pass

                # Fallback to internal MJPEG generation
                try:
                    success = _run_coro_safe(
                        stream_server.start_stream(resolved_id, {
                            'rtsp_url': rtsp_url,
                            'webrtc_enabled': False
                        })
                    )
                    if success:
                        logger.info(f"✅ Stream started for {resolved_id}")
                    else:
                        logger.error(f"❌ Failed to start stream for {resolved_id}")
                        return error_response(f"Stream start failed for {resolved_id}")
                except Exception as e:
                    return error_response(f"Start exception: {str(e)}")
            except Exception as e:
                return error_response(f"Setup exception: {str(e)}")
        
        import time
        from core.stream_server import StreamQuality

        def generate():
            boundary = 'frame'
            frame_count = 0
            max_retries = 300  # 10 seconds at 30fps
            consecutive_placeholders = 0
            last_log_time = 0
            
            # Create a placeholder image (black frame)
            try:
                import cv2
                import numpy as np
                placeholder = np.zeros((360, 640, 3), dtype=np.uint8)
                cv2.putText(placeholder, "Stream Loading...", (50, 180), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                _, placeholder_jpeg = cv2.imencode('.jpg', placeholder)
                placeholder_bytes = placeholder_jpeg.tobytes()
            except Exception:
                # Fallback if cv2/numpy issues
                placeholder_bytes = b''

            last_frame_time = time.time()

            while True:
                try:
                    frame_bytes = stream_server.get_frame(camera_id, quality=StreamQuality.MEDIUM, format='jpeg')
                    if frame_bytes:
                        frame_count += 1
                        consecutive_placeholders = 0
                        last_frame_time = time.time()
                        yield (b"--" + boundary.encode('utf-8') + b"\r\n"
                               b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n")
                        time.sleep(0.033) # Cap at ~30fps
                    else:
                        # If stream is starting, yield placeholder instead of breaking immediately
                        # But only if we have a placeholder and it's been a while since last frame
                        consecutive_placeholders += 1
                        current_time = time.time()
                        
                        # Exponential backoff for logging: 1s, 2s, 4s, 8s... max 60s
                        log_interval = min(60, 2 ** (consecutive_placeholders // 30))
                        
                        if frame_count == 0 and max_retries > 0:
                             max_retries -= 1
                             if placeholder_bytes and max_retries % 30 == 0: # Send placeholder every second while waiting
                                 yield (b"--" + boundary.encode('utf-8') + b"\r\n"
                                        b"Content-Type: image/jpeg\r\n\r\n" + placeholder_bytes + b"\r\n")
                             time.sleep(0.033)
                        elif max_retries <= 0:
                            # Timeout waiting for first frame
                            if placeholder_bytes:
                                # Only log if enough time has passed since last log
                                if current_time - last_log_time > log_interval:
                                    logger.warning(f"Stream timeout for {camera_id}, sending placeholder (interval {log_interval}s)")
                                    last_log_time = current_time
                                    
                                yield (b"--" + boundary.encode('utf-8') + b"\r\n"
                                       b"Content-Type: image/jpeg\r\n\r\n" + placeholder_bytes + b"\r\n")
                                time.sleep(1.0) # Slow down updates to 1 FPS when showing placeholder
                            else:
                                break
                        else:
                            # Stream was running but now no frames?
                            time.sleep(0.05)
                            
                except GeneratorExit:
                    # Log only once on exit
                    if frame_count > 0:
                        logger.info(f"🛑 MJPEG stream closed for {camera_id}")
                    break
                except Exception as e:
                    logger.error(f"❌ MJPEG stream error for {camera_id}: {e}")
                    time.sleep(0.1)

        return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')
    except Exception as e:
        return error_response(f"Stream error: {str(e)}")

    # (no outer try here; handled above)


@api_bp.route('/cameras/<camera_id>/snapshot', methods=['GET'])
def get_snapshot(camera_id):
    """Get camera snapshot"""
    try:
        # Prefer a real frame from stream_server when available (avoids returning "status snapshot"
        # placeholders from camera_manager when RTSP capture fails but the stream is otherwise running).
        snapshot_data = None

        if stream_server:
            # Ensure stream is running, then pull latest frame bytes from stream_server
            try:
                # Attempt to start stream if needed (mirrors logic used in get_motion)
                if hasattr(stream_server, 'active_streams') and camera_id not in stream_server.active_streams:
                    # Load camera RTSP info from cameras.json
                    try:
                        from app import cameras_db  # type: ignore
                    except Exception:
                        import json as _json
                        with open('cameras.json', 'r') as _f:
                            cameras_db = _json.load(_f)

                    camera = None
                    for cam in cameras_db:
                        if cam.get('id') == camera_id:
                            camera = cam
                            break

                    if camera and camera.get('rtsp_url'):
                        try:
                            _run_coro_safe(
                                stream_server.start_stream(camera_id, {
                                    'rtsp_url': camera.get('rtsp_url'),
                                    'webrtc_enabled': False,
                                    'fps': 15
                                })
                            )
                        except Exception as e:
                            logger.warning(f"Failed to start stream for snapshot: {e}")

                # Get frame bytes
                frame_bytes = stream_server.get_frame(camera_id)
                if frame_bytes:
                    return Response(frame_bytes, mimetype='image/jpeg')
            except Exception as _e:
                logger.warning(f"Snapshot fallback via stream_server failed for {camera_id}: {_e}")

        # Fallback: camera_manager snapshot (may generate a status/placeholder image)
        if camera_manager:
            snapshot_data = safe_service_call(camera_manager, 'get_snapshot', None, camera_id)

        if snapshot_data is None:
            return jsonify({
                "success": False,
                "message": f"Snapshot not available for camera {camera_id}"
            }), 404

        if isinstance(snapshot_data, bytes):
            return Response(snapshot_data, mimetype='image/jpeg')
        else:
            return jsonify({
                "success": True,
                "data": snapshot_data,
                "message": "Snapshot retrieved"
            })

    except Exception as e:
        logger.error(f"Failed to get snapshot for camera {camera_id}: {e}")
        return jsonify({
            "success": False,
            "message": f"Failed to get snapshot: {str(e)}"
        }), 500


@api_bp.route('/cameras/<camera_id>/motion', methods=['GET'])
def get_motion(camera_id):
    """Get current motion status for a camera with timeout handling"""
    try:
        if not stream_server:
            return jsonify({
                "success": False,
                "message": "Stream server not available"
            }), 503

        # Get cameras database
        try:
            from app import cameras_db
        except:
            # Load from file if import fails
            import json
            with open('cameras.json', 'r') as f:
                cameras_db = json.load(f)
        
        # Try to start stream if not already active (with timeout)
        if hasattr(stream_server, 'active_streams') and camera_id not in stream_server.active_streams:
            # Find camera and start stream
            camera = None
            for cam in cameras_db:
                if cam.get('id') == camera_id:
                    camera = cam
                    break
            
            if camera and camera.get('rtsp_url'):
                logger.info(f"Auto-starting stream for motion detection: {camera_id}")
                
                try:
                    success = _run_coro_safe(
                        stream_server.start_stream(camera_id, {
                            'rtsp_url': camera.get('rtsp_url'),
                            'webrtc_enabled': False,  # Direct RTSP for motion detection
                            'fps': 15
                        })
                    )
                    
                    if success:
                        logger.info(f"Stream auto-started for camera {camera_id}")
                    else:
                        logger.warning(f"Failed to auto-start stream for {camera_id}")
                            
                except Exception as e:
                    logger.warning(f"Stream start error for camera {camera_id}: {e}")
                except Exception as e:
                    logger.warning(f"Failed to auto-start stream for {camera_id}: {e}")

        # Get motion status with WebSocket-first approach
        try:
            # Try to get motion status from stream server (WebSocket source)
            # Check if this camera uses MediaMTX URL and adjust timeout accordingly
            is_mediamtx_camera = False
            try:
                from app import cameras_db
            except:
                import json
                with open('cameras.json', 'r') as f:
                    cameras_db = json.load(f)

            for cam in cameras_db:
                if cam.get('id') == camera_id and ':8554' in cam.get('rtsp_url', ''):
                    is_mediamtx_camera = True
                    break

            # Use longer timeout for MediaMTX cameras
            motion_timeout = 60 if is_mediamtx_camera else 3  # 60 seconds for MediaMTX, 3 for direct

            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(safe_service_call, stream_server, 'get_motion_status', None, camera_id)
                status = future.result(timeout=motion_timeout)

                if status is not None:
                    # WebSocket motion detection is working
                    return jsonify({
                        "success": True,
                        "data": {
                            "camera_id": camera_id,
                            "motion": status,
                            "timestamp": datetime.now().isoformat(),
                            "source": "websocket"
                        }
                    })
                else:
                    # Fallback to empty status
                    status = {
                        "has_motion": False,
                        "score": 0.0,
                        "regions": [],
                        "frame_width": 0,
                        "frame_height": 0,
                        "bbox": None,
                        "bbox_norm": None,
                        "fallback": True,
                        "fallback_reason": "no_websocket_status"
                    }

        except concurrent.futures.TimeoutError:
            logger.warning(f"WebSocket motion detection timeout for camera {camera_id}")
            status = {
                "has_motion": False,
                "score": 0.0,
                "regions": [],
                "frame_width": 0,
                "frame_height": 0,
                "bbox": None,
                "bbox_norm": None,
                "timeout": True,
                "fallback": True,
                "fallback_reason": "websocket_timeout"
            }
        except Exception as e:
            logger.error(f"Error getting WebSocket motion status for camera {camera_id}: {e}")
            status = {
                "has_motion": False,
                "score": 0.0,
                "regions": [],
                "frame_width": 0,
                "frame_height": 0,
                "bbox": None,
                "bbox_norm": None,
                "error": str(e),
                "fallback": True,
                "fallback_reason": "websocket_error"
            }
        
        # If status isn't ready yet, return a stable empty payload instead of 404
        if status is None:
            status = {
                "has_motion": False,
                "score": 0.0,
                "regions": [],
                "frame_width": 0,
                "frame_height": 0,
                "bbox": None,
                "bbox_norm": None,
            }

        return jsonify({
            "success": True,
            "data": {
                "camera_id": camera_id,
                "motion": status,
                "timestamp": datetime.now().isoformat()
            }
        })

    except Exception as e:
        logger.error(f"Failed to get motion for camera {camera_id}: {e}")
        return jsonify({
            "success": False,
            "message": f"Failed to get motion: {str(e)}"
        }), 500





@api_bp.route('/cameras/<camera_id>/stream/start', methods=['POST'])
def start_camera_stream(camera_id):
    """Start stream for a camera to enable motion detection"""
    try:
        if not stream_server:
            return jsonify({
                "success": False,
                "message": "Stream server not available"
            }), 503

        # Find camera in database - import from main app module
        try:
            import app
            cameras_db = getattr(app, 'cameras_db', [])
        except:
            # Load from file if import fails
            import json
            with open('cameras.json', 'r') as f:
                cameras_db = json.load(f)
        
        # Find the camera
        camera = None
        for cam in cameras_db:
            if cam.get('id') == camera_id:
                camera = cam
                break
        
        if not camera:
            return jsonify({
                "success": False,
                "message": f"Camera {camera_id} not found"
            }), 404

        # Check if stream is already active
        if hasattr(stream_server, 'active_streams') and camera_id in stream_server.active_streams:
            return jsonify({
                "success": True,
                "message": f"Stream already active for camera {camera_id}",
                "data": {
                    "camera_id": camera_id,
                    "status": "already_active"
                }
            })

        # Start the stream
        # Prepare camera config for stream server
        stream_config = {
            'rtsp_url': camera.get('rtsp_url'),
            'webrtc_enabled': camera.get('webrtc_enabled', True),
            'mediamtx_path': camera.get('mediamtx_path') or camera.get('mediamtxPath') or camera_id
        }
        
        # Start the stream using shared infrastructure
        success = _run_coro_safe(stream_server.start_stream(camera_id, stream_config))
        
        if success:
            return jsonify({
                "success": True,
                "message": f"Stream started for camera {camera_id}",
                "data": {
                    "camera_id": camera_id,
                    "status": "started",
                    "motion_detection_enabled": camera.get('motion_detection', False)
                }
            })
        else:
            return jsonify({
                "success": False,
                "message": f"Failed to start stream for camera {camera_id}"
            }), 500

    except Exception as e:
        logger.error(f"Failed to start stream for camera {camera_id}: {e}")
        return jsonify({
            "success": False,
            "message": f"Failed to start stream: {str(e)}"
        }), 500


@api_bp.route('/cameras/<camera_id>/force-start', methods=['POST'])
def force_start_camera_stream(camera_id):
    """
    Best-effort: deterministically start (or ensure) a camera stream so MediaMTX paths exist before WHEP/HLS.
    This is intentionally idempotent and safe to call repeatedly.
    """
    try:
        if not camera_manager and not stream_server:
            return jsonify({
                "success": False,
                "message": "Camera manager / Stream server not available"
            }), 503

        # Resolve camera config from camera_manager first (preferred), then from JSON file.
        # IMPORTANT: do NOT call camera_manager.acquire_camera() here; this endpoint is used for warmup
        # and must not increment viewer refcounts.
        camera_obj = None
        if camera_manager:
            try:
                camera_obj = safe_service_call(camera_manager, 'get_camera', None, camera_id)
            except Exception:
                camera_obj = None

        camera = None
        if isinstance(camera_obj, dict):
            camera = camera_obj
        elif camera_obj:
            camera = {
                'id': getattr(camera_obj, 'id', camera_id),
                'name': getattr(camera_obj, 'name', ''),
                'rtsp_url': getattr(camera_obj, 'rtsp_url', ''),
                'webrtc_enabled': getattr(camera_obj, 'webrtc_enabled', True),
                'mediamtx_path': getattr(camera_obj, 'mediamtx_path', None),
            }

        if not camera:
            try:
                import json
                cameras_path = 'data/cameras.json' if os.path.exists('data/cameras.json') else 'cameras.json'
                with open(cameras_path, 'r') as f:
                    cameras_db = json.load(f)
                for cam in cameras_db:
                    if cam.get('id') == camera_id:
                        camera = cam
                        break
            except Exception:
                camera = None

        if not camera:
            return jsonify({
                "success": False,
                "message": f"Camera {camera_id} not found"
            }), 404

        rtsp_url = camera.get('rtsp_url') if isinstance(camera, dict) else None
        if not rtsp_url:
            return jsonify({
                "success": False,
                "message": f"Camera {camera_id} missing rtsp_url"
            }), 400

        mediamtx_path = (camera.get('mediamtx_path') or camera.get('mediamtxPath') or camera_id) if isinstance(camera, dict) else camera_id

        stream_config = {
            'rtsp_url': rtsp_url,
            'webrtc_enabled': bool(camera.get('webrtc_enabled', True)) if isinstance(camera, dict) else True,
            'mediamtx_path': mediamtx_path
        }

        # 1) Ensure MediaMTX is configured so /proxy/webrtc/* and /proxy/hls/* stop returning 404.
        # Prefer a mediamtx-only warmup path that doesn't touch viewer refcounts.
        mediamtx_ok = False
        try:
            mt = None
            if stream_server:
                mt = getattr(stream_server, 'mediamtx_client', None)
            if not mt and camera_manager:
                mt = getattr(camera_manager, 'mediamtx_client', None)

            if mt and stream_config.get('webrtc_enabled') and mediamtx_path and rtsp_url:
                mediamtx_ok = bool(_run_coro_safe(mt.configure_stream_source(str(mediamtx_path), str(rtsp_url), force_recreate=False)))
                if not mediamtx_ok:
                    logger.warning(f"force-start: mediamtx_client.configure_stream_source returned False for {camera_id} -> {mediamtx_path}")
            else:
                # Fallback: use app.ensure_camera_mediamtx_ready() if available.
                try:
                    from app import ensure_camera_mediamtx_ready  # type: ignore
                    cam_payload = dict(camera) if isinstance(camera, dict) else {"id": camera_id, "rtsp_url": rtsp_url, "mediamtx_path": mediamtx_path}
                    mediamtx_ok = bool(ensure_camera_mediamtx_ready(cam_payload, force_check=True))
                except Exception as e:
                    logger.debug(f"force-start: ensure_camera_mediamtx_ready unavailable/failed for {camera_id}: {e}")
        except Exception as e:
            logger.warning(f"force-start: mediamtx warmup failed for {camera_id}: {e}")

        # 2) Optional: kick off StreamServer capture/motion/detection pipelines in the background.
        # Keep the HTTP request fast and idempotent.
        capture_start_requested = False
        if stream_server:
            capture_start_requested = True
            def _bg_start():
                try:
                    ok = _run_coro_safe(stream_server.start_stream(camera_id, stream_config))
                    if not ok:
                        logger.warning(f"force-start(bg): stream_server.start_stream returned False for {camera_id}")
                except Exception as e:
                    logger.warning(f"force-start(bg): stream_server.start_stream failed for {camera_id}: {e}")
            try:
                threading.Thread(target=_bg_start, daemon=True).start()
            except Exception:
                # Worst case: skip background kick.
                capture_start_requested = False

        return jsonify({
            "success": True,
            "message": f"Stream started for camera {camera_id}",
            "data": {
                "camera_id": camera_id,
                "status": "started",
                "mediamtx_path": mediamtx_path,
                "mediamtx_configured": bool(mediamtx_ok),
                "capture_start_requested": bool(capture_start_requested)
            }
        })

    except Exception as e:
        logger.error(f"Failed to force-start stream for camera {camera_id}: {e}")
        return jsonify({
            "success": False,
            "message": f"Failed to force-start stream: {str(e)}"
        }), 500


@api_bp.route('/cameras/<camera_id>/stream/ready', methods=['GET'])
def get_camera_stream_ready(camera_id):
    """
    Lightweight readiness endpoint for UI warmups.
    Reports MediaMTX path readiness and (optionally) whether the HLS playlist is reachable.

    Query:
      - probe_hls=1 to perform a best-effort playlist fetch (can be disabled to avoid extra work)
    """
    try:
        # Resolve camera config (best-effort; do not connect).
        camera_obj = None
        if camera_manager:
            try:
                camera_obj = safe_service_call(camera_manager, 'get_camera', None, camera_id)
            except Exception:
                camera_obj = None

        camera = None
        if isinstance(camera_obj, dict):
            camera = camera_obj
        elif camera_obj:
            camera = {
                'id': getattr(camera_obj, 'id', camera_id),
                'name': getattr(camera_obj, 'name', ''),
                'rtsp_url': getattr(camera_obj, 'rtsp_url', ''),
                'webrtc_enabled': getattr(camera_obj, 'webrtc_enabled', True),
                'mediamtx_path': getattr(camera_obj, 'mediamtx_path', None),
            }

        if not camera:
            try:
                import json
                cameras_path = 'data/cameras.json' if os.path.exists('data/cameras.json') else 'cameras.json'
                with open(cameras_path, 'r') as f:
                    cameras_db = json.load(f)
                for cam in cameras_db:
                    if cam.get('id') == camera_id:
                        camera = cam
                        break
            except Exception:
                camera = None

        if not camera:
            return jsonify({"success": False, "message": f"Camera {camera_id} not found"}), 404

        mediamtx_path = (camera.get('mediamtx_path') or camera.get('mediamtxPath') or camera_id) if isinstance(camera, dict) else camera_id
        mediamtx_path = str(mediamtx_path or camera_id).lstrip("/").rstrip("/")
        hls_url = f"/proxy/hls/{mediamtx_path}/index.m3u8"

        # MediaMTX readiness
        path_info: Dict[str, Any] = {"ready": False}
        try:
            mt = None
            if stream_server:
                mt = getattr(stream_server, 'mediamtx_client', None)
            if not mt and camera_manager:
                mt = getattr(camera_manager, 'mediamtx_client', None)
            if mt and hasattr(mt, "get_path_info"):
                path_info = _run_coro_safe(mt.get_path_info(str(mediamtx_path))) or {"ready": False}
            else:
                # Fallback to app.mediamtx wrapper if present
                try:
                    from app import mediamtx  # type: ignore
                    path_info = mediamtx.get_path_info(str(mediamtx_path)) or {"ready": False}
                except Exception:
                    path_info = {"ready": False}
        except Exception:
            path_info = {"ready": False}

        mediamtx_ready = bool((path_info or {}).get("ready", False))

        # Optional HLS probe (disabled by default)
        probe_hls = str(request.args.get("probe_hls", "")).lower() in ("1", "true", "yes", "y")
        hls_ready: Optional[bool] = None
        if probe_hls:
            # Best-effort: avoid long hangs.
            hls_ready = False
            try:
                import urllib.request
                base = (request.host_url or "").rstrip("/")
                url = f"{base}{hls_url}"
                req = urllib.request.Request(url, method="GET", headers={"Cache-Control": "no-cache"})
                with urllib.request.urlopen(req, timeout=2.0) as resp:
                    hls_ready = (getattr(resp, "status", 0) == 200)
            except Exception:
                hls_ready = False

        return jsonify({
            "success": True,
            "data": {
                "camera_id": camera_id,
                "mediamtx_path": mediamtx_path,
                "mediamtx_ready": mediamtx_ready,
                "hls_url": hls_url,
                "hls_ready": hls_ready,
                "checked_at": datetime.now().isoformat()
            }
        })

    except Exception as e:
        logger.error(f"Failed to get stream readiness for camera {camera_id}: {e}")
        return jsonify({"success": False, "message": f"Failed to get stream readiness: {str(e)}"}), 500


@api_bp.route('/cameras/<camera_id>/stream/status', methods=['GET'])
def get_camera_stream_status(camera_id):
    """Get stream status for a camera"""
    try:
        if not stream_server:
            return jsonify({
                "success": False,
                "message": "Stream server not available"
            }), 503

        # Check if stream is active
        is_active = False
        if hasattr(stream_server, 'active_streams'):
            is_active = camera_id in stream_server.active_streams

        # Get motion detection status
        motion_enabled = False
        try:
            import app
            cameras_db = getattr(app, 'cameras_db', [])
            for cam in cameras_db:
                if cam.get('id') == camera_id:
                    motion_enabled = cam.get('motion_detection', False)
                    break
        except:
            pass

        return jsonify({
            "success": True,
            "data": {
                "camera_id": camera_id,
                "stream_active": is_active,
                "motion_detection_enabled": motion_enabled,
                "motion_detection_working": is_active and motion_enabled
            }
        })

    except Exception as e:
        logger.error(f"Failed to get stream status for camera {camera_id}: {e}")
        return jsonify({
            "success": False,
            "message": f"Failed to get stream status: {str(e)}"
        }), 500


@api_bp.route('/cameras/<camera_id>/motion/enable', methods=['POST'])
def enable_motion_detection(camera_id):
    """Enable motion detection for a camera by starting its stream"""
    try:
        if not stream_server:
            return jsonify({
                "success": False,
                "message": "Stream server not available"
            }), 503

        # Find camera by ID
        try:
            from app import cameras_db
        except Exception:
            import json
            with open('cameras.json', 'r') as f:
                cameras_db = json.load(f)
        camera = None
        for cam in cameras_db:
            if cam.get('id') == camera_id:
                camera = cam
                break
        
        if not camera:
            return jsonify({
                "success": False,
                "message": f"Camera {camera_id} not found in database"
            }), 404

        # Force start the stream with direct RTSP URL (bypass MediaMTX issues)
        def start_stream_sync():
            try:
                # Use the original RTSP URL directly
                rtsp_url = camera.get('rtsp_url')
                if not rtsp_url:
                    return False
                    
                return _run_coro_safe(
                    stream_server.start_stream(camera_id, {
                        'rtsp_url': rtsp_url,
                        'webrtc_enabled': False,  # Disable WebRTC to avoid MediaMTX issues
                        'fps': 15  # Lower FPS for motion detection
                    })
                )
            except Exception as e:
                logger.error(f"Error starting stream for {camera_id}: {e}")
                return False
        
        success = start_stream_sync()
        
        if success:
            return jsonify({
                "success": True,
                "message": f"Motion detection enabled for camera {camera_id}",
                "data": {
                    "camera_id": camera_id,
                    "camera_name": camera.get('name'),
                    "motion_detection_active": True,
                    "stream_active": True
                }
            })
        else:
            return jsonify({
                "success": False,
                "message": f"Failed to start motion detection for camera {camera_id}"
            }), 500

    except Exception as e:
        logger.error(f"Failed to enable motion detection for camera {camera_id}: {e}")
        return jsonify({
            "success": False,
            "message": f"Failed to enable motion detection: {str(e)}"
        }), 500


@api_bp.route('/cameras/<camera_id>/motion/settings', methods=['GET'])
def get_motion_settings(camera_id):
    """Get current motion detector settings for a camera."""
    try:
        if not stream_server:
            return jsonify({
                "success": False,
                "message": "Stream server not available"
            }), 503

        # Get current motion detector parameters
        motion_detector = safe_service_call(stream_server, 'get_motion_detector', None, camera_id)
        if motion_detector is None:
            # Lazily create the detector by updating params with no-op, then re-fetch
            try:
                safe_service_call(stream_server, 'update_motion_params', None, camera_id, {})
                motion_detector = safe_service_call(stream_server, 'get_motion_detector', None, camera_id)
            except Exception:
                motion_detector = None
        if motion_detector is None:
            return jsonify({
                "success": False,
                "message": f"Motion detector not available for camera {camera_id}"
            }), 404

        # Extract current parameters
        # Map internal SimpleMotionDetector fields to UI-friendly names
        current_settings = {
            'diff_threshold': int(getattr(motion_detector, 'mog2_var_threshold', 16)),
            'min_area': int(getattr(motion_detector, 'min_area', 1000)),
            'min_area_norm': float(getattr(motion_detector, 'min_area_norm', 0.001)),
            'learning_rate': float(getattr(motion_detector, 'learning_rate', 0.02)),
            'dilate_iterations': int(getattr(motion_detector, 'dilate_iterations', 1)),
            'motion_history': int(getattr(motion_detector, 'mog2_history', 500) if hasattr(motion_detector, 'mog2_history') else 500),
            'downscale_width': int(getattr(motion_detector, 'downscale_width', 320)),
            'car_person_sensitivity': float(getattr(motion_detector, '_car_person_sensitivity', 1.0)),
            'motion_point_timeout': int(getattr(motion_detector, 'motion_point_timeout', 5000))
        }

        return jsonify({
            "success": True,
            "data": {
                "camera_id": camera_id,
                "motion_params": current_settings
            }
        })

    except Exception as e:
        logger.error(f"Failed to get motion settings for camera {camera_id}: {e}")
        return jsonify({
            "success": False,
            "message": f"Failed to get motion settings: {str(e)}"
        }), 500


@api_bp.route('/cameras/<camera_id>/motion/settings', methods=['PUT'])
def update_motion_settings(camera_id):
    """Update motion detector sensitivity parameters for a camera."""
    try:
        if not stream_server:
            return jsonify({
                "success": False,
                "message": "Stream server not available"
            }), 503

        data = request.get_json() or {}
        
        # Support both legacy single sensitivity and new multi-parameter approach
        if 'sensitivity' in data and len(data) == 1:
            # Legacy mode - single sensitivity slider
            sensitivity = float(data.get('sensitivity', 1.0))
            sensitivity = max(0.1, min(10.0, sensitivity))
            
            # Balanced sensitivity mapping for persistent object detection
            base_min_area_norm = 0.001
            base_var_threshold = 16
            
            target_min_area_norm = max(1e-5, base_min_area_norm / sensitivity)
            target_var_threshold = max(4, int(base_var_threshold / sensitivity))
            
            motion_params = {
                'min_area_norm': target_min_area_norm,
                'diff_threshold': target_var_threshold
            }
        else:
            # New multi-parameter mode
            motion_params = {
                'diff_threshold': int(data.get('diff_threshold', 20)),
                'min_area': int(data.get('min_area', 500)),
                'min_area_norm': float(data.get('min_area_norm', 0.0005)),
                'learning_rate': float(data.get('learning_rate', 0.02)),
                'dilate_iterations': int(data.get('dilate_iterations', 1)),
                'motion_history': int(data.get('motion_history', 3)),
                'downscale_width': int(data.get('downscale_width', 320)),
                'car_person_sensitivity': float(data.get('car_person_sensitivity', 1.0)),
                'motion_point_timeout': int(data.get('motion_point_timeout', 5000))
            }
            
            # Validate parameter ranges
            motion_params['diff_threshold'] = max(4, min(100, motion_params['diff_threshold']))
            motion_params['min_area'] = max(10, min(10000, motion_params['min_area']))
            motion_params['min_area_norm'] = max(1e-6, min(0.1, motion_params['min_area_norm']))
            motion_params['learning_rate'] = max(0.001, min(0.5, motion_params['learning_rate']))
            motion_params['dilate_iterations'] = max(0, min(5, motion_params['dilate_iterations']))
            motion_params['motion_history'] = max(1, min(10, motion_params['motion_history']))
            motion_params['downscale_width'] = max(64, min(640, motion_params['downscale_width']))
            motion_params['car_person_sensitivity'] = max(0.1, min(5.0, motion_params['car_person_sensitivity']))
            motion_params['motion_point_timeout'] = max(1000, min(15000, motion_params['motion_point_timeout']))

        ok = safe_service_call(stream_server, 'update_motion_params', None, camera_id, motion_params)
        if ok is False:
            return jsonify({ 'success': False, 'message': 'Failed to update motion settings' }), 400

        return jsonify({ 
            'success': True, 
            'data': { 
                'camera_id': camera_id, 
                'motion_params': motion_params 
            } 
        })
    except Exception as e:
        logger.error(f"Failed to update motion settings for camera {camera_id}: {e}")
        return jsonify({ 'success': False, 'message': f'Failed to update motion settings: {str(e)}' }), 500


@api_bp.route('/cameras/<camera_id>/tracks', methods=['GET'])
def get_tracks(camera_id):
    """Get current tracking information for a camera"""
    try:
        if not stream_server:
            return jsonify({
                "success": False,
                "message": "Stream server not available"
            }), 503

        data = safe_service_call(stream_server, 'get_tracks', None, camera_id)
        if data is None:
            return jsonify({
                "success": False,
                "message": f"Tracks not available for camera {camera_id}"
            }), 404

        data.update({"timestamp": datetime.now().isoformat()})
        return jsonify({
            "success": True,
            "data": data
        })

    except Exception as e:
        logger.error(f"Failed to get tracks for camera {camera_id}: {e}")
        return jsonify({
            "success": False,
            "message": f"Failed to get tracks: {str(e)}"
        }), 500


@api_bp.route('/cameras/<camera_id>/tracks/lifecycle', methods=['GET'])
def get_tracks_lifecycle(camera_id):
    """Get track birth/death events and active IDs for a camera"""
    try:
        if not stream_server:
            return jsonify({
                "success": False,
                "message": "Stream server not available"
            }), 503

        data = safe_service_call(stream_server, 'get_track_lifecycle', None, camera_id)
        if data is None:
            return jsonify({
                "success": False,
                "message": f"Lifecycle not available for camera {camera_id}"
            }), 404

        data.update({"timestamp": datetime.now().isoformat()})
        return jsonify({
            "success": True,
            "data": data
        })

    except Exception as e:
        logger.error(f"Failed to get track lifecycle for camera {camera_id}: {e}")
        return jsonify({
            "success": False,
            "message": f"Failed to get track lifecycle: {str(e)}"
        }), 500


@api_bp.route('/cameras/<camera_id>/tracks/trajectories', methods=['GET'])
def get_tracks_trajectories(camera_id):
    """Get track trajectory points for a camera"""
    try:
        if not stream_server:
            return jsonify({
                "success": False,
                "message": "Stream server not available"
            }), 503

        data = safe_service_call(stream_server, 'get_track_trajectories', None, camera_id)
        if data is None:
            # Return empty trajectories to avoid 404 noise
            data = {
                "camera_id": camera_id,
                "trajectories": [],
                "frame_width": 0,
                "frame_height": 0
            }

        data.update({"timestamp": datetime.now().isoformat()})
        return jsonify({
            "success": True,
            "data": data
        })

    except Exception as e:
        logger.error(f"Failed to get track trajectories for camera {camera_id}: {e}")
        return jsonify({
            "success": False,
            "message": f"Failed to get track trajectories: {str(e)}"
        }), 500


@api_bp.route('/cameras/<camera_id>/tracks/trajectories/store', methods=['POST'])
def store_tracks_trajectories(camera_id):
    """Persist current in-memory trajectories to DB for later analysis."""
    try:
        if not stream_server:
            return jsonify({
                "success": False,
                "message": "Stream server not available"
            }), 503
        if not db_manager or not hasattr(db_manager, 'store_track_trajectory'):
            return jsonify({
                "success": False,
                "message": "Database not available"
            }), 503

        data = safe_service_call(stream_server, 'get_track_trajectories', None, camera_id)
        if not data or not data.get('trajectories'):
            return jsonify({
                "success": False,
                "message": "No trajectories to store"
            }), 400
        saved = 0
        for t in data['trajectories']:
            ok = db_manager.store_track_trajectory(camera_id, {
                'id': f"{camera_id}-{t.get('id')}",
                'track_id': t.get('id'),
                'short_id': t.get('short_id'),
                'started_at': t.get('started_at'),
                'ended_at': t.get('ended_at'),
                'active': t.get('active'),
                'points': t.get('points', [])
            })
            if ok:
                saved += 1
        return jsonify({
            'success': True,
            'data': {'saved': saved}
        })
    except Exception as e:
        logger.error(f"Failed to store trajectories for camera {camera_id}: {e}")
        return jsonify({
            'success': False,
            'message': f'Failed to store trajectories: {str(e)}'
        }), 500


@api_bp.route('/cameras/<camera_id>/tracks/trajectories/list', methods=['GET'])
def get_tracks_trajectories_list(camera_id):
    """Get list of stored track trajectories for a camera"""
    try:
        if not stream_server:
            return jsonify({
                "success": False,
                "message": "Stream server not available"
            }), 503

        data = safe_service_call(stream_server, 'get_stored_trajectories', None, camera_id)
        if data is None:
            return jsonify({
                "success": False,
                "message": f"Stored trajectories not available for camera {camera_id}"
            }), 404

        data.update({"timestamp": datetime.now().isoformat()})
        return jsonify({
            "success": True,
            "data": data
        })

    except Exception as e:
        logger.error(f"Failed to get stored trajectories for camera {camera_id}: {e}")
        return jsonify({
            "success": False,
            "message": f"Failed to get stored trajectories: {str(e)}"
        }), 500


@api_bp.route('/cameras/<camera_id>/zones', methods=['GET'])
def get_camera_zones(camera_id):
    """Get zones configuration for a camera"""
    try:
        # Prefer DB-backed shapes so server-side automations can evaluate shapes reliably.
        data = None
        if db_manager and hasattr(db_manager, "get_camera_shapes"):
            try:
                rec = db_manager.get_camera_shapes(camera_id)
                if rec:
                    data = {
                        "camera_id": camera_id,
                        "zones": rec.get("zones", []) or [],
                        "lines": rec.get("lines", []) or [],
                        "tags": rec.get("tags", []) or [],
                        "rules": [],
                    }
            except Exception:
                data = None

        # Fallback: StreamServer in-memory shapes (if any)
        if data is None and stream_server:
            try:
                shapes = safe_service_call(stream_server, "get_camera_shapes", None, camera_id)
                if isinstance(shapes, dict):
                    data = {
                        "camera_id": camera_id,
                        "zones": shapes.get("zones", []) or [],
                        "lines": shapes.get("lines", []) or [],
                        "tags": shapes.get("tags", []) or [],
                        "rules": [],
                    }
            except Exception:
                data = None

        if data is None:
            # Return empty shapes to avoid 404 noise
            data = {
                "camera_id": camera_id,
                "zones": [],
                "lines": [],
                "tags": [],
                "rules": []
            }

        data.update({"timestamp": datetime.now().isoformat()})
        return jsonify({
            "success": True,
            "data": data
        })

    except Exception as e:
        logger.error(f"Failed to get zones for camera {camera_id}: {e}")
        return jsonify({
            "success": False,
            "message": f"Failed to get zones: {str(e)}"
        }), 500


# Persist camera shapes (zones/lines/tags) for server-side rules/automation
@api_bp.route('/cameras/<camera_id>/zones', methods=['PUT'])
def put_camera_zones(camera_id):
    """Persist zones/lines/tags for a camera (DB-backed) and sync StreamServer in-memory shapes."""
    try:
        if not db_manager or not hasattr(db_manager, "upsert_camera_shapes"):
            return jsonify({
                "success": False,
                "message": "Database not available"
            }), 503

        data = request.get_json() or {}
        zones = data.get("zones", []) or []
        lines = data.get("lines", []) or []
        tags = data.get("tags", []) or []

        # Normalize shapes into the schema StreamServer expects:
        # - zones: {id, enabled, points:[{x,y}], ...}
        # - lines: {id, enabled, p1:{x,y}, p2:{x,y}, ...}
        # - tags:  {id, enabled, x, y, ...}
        def _norm_zone(z):
            if not isinstance(z, dict):
                return None
            pts = z.get("points") or z.get("pts") or z.get("coordinates") or []
            if not isinstance(pts, list):
                pts = []
            # ensure x/y numeric
            points = []
            for p in pts:
                if not isinstance(p, dict):
                    continue
                try:
                    points.append({"x": float(p.get("x", 0.0)), "y": float(p.get("y", 0.0))})
                except Exception:
                    continue
            out = dict(z)
            out["enabled"] = bool(out.get("enabled", True))
            out["points"] = points
            out.pop("pts", None)
            out.pop("coordinates", None)
            return out

        def _norm_line(l):
            if not isinstance(l, dict):
                return None
            p1 = l.get("p1") or (l.get("points")[0] if isinstance(l.get("points"), list) and len(l.get("points")) > 0 else None) or {}
            p2 = l.get("p2") or (l.get("points")[1] if isinstance(l.get("points"), list) and len(l.get("points")) > 1 else None) or {}
            out = dict(l)
            out["enabled"] = bool(out.get("enabled", True))
            try:
                out["p1"] = {"x": float(p1.get("x", 0.0)), "y": float(p1.get("y", 0.0))}
            except Exception:
                out["p1"] = {"x": 0.0, "y": 0.0}
            try:
                out["p2"] = {"x": float(p2.get("x", 1.0)), "y": float(p2.get("y", 1.0))}
            except Exception:
                out["p2"] = {"x": 1.0, "y": 1.0}
            return out

        def _norm_tag(t):
            if not isinstance(t, dict):
                return None
            out = dict(t)
            out["enabled"] = bool(out.get("enabled", True))
            # accept either {x,y} or {anchor:{x,y}}
            anchor = out.get("anchor") if isinstance(out.get("anchor"), dict) else {}
            x = out.get("x", anchor.get("x", 0.0))
            y = out.get("y", anchor.get("y", 0.0))
            try:
                out["x"] = float(x)
                out["y"] = float(y)
            except Exception:
                out["x"] = 0.0
                out["y"] = 0.0
            return out

        norm = {
            "zones": [z for z in (_norm_zone(z) for z in zones) if z is not None],
            "lines": [l for l in (_norm_line(l) for l in lines) if l is not None],
            "tags": [t for t in (_norm_tag(t) for t in tags) if t is not None],
        }

        ok = db_manager.upsert_camera_shapes(camera_id, norm)
        if not ok:
            return jsonify({ "success": False, "message": "Failed to save zones" }), 500

        # Sync StreamServer in-memory shapes for ROI detection
        if stream_server:
            try:
                safe_service_call(stream_server, "set_camera_shapes", None, camera_id, norm)
            except Exception:
                pass

        return jsonify({
            "success": True,
            "data": {
                "camera_id": camera_id,
                "zones": norm["zones"],
                "lines": norm["lines"],
                "tags": norm["tags"],
                "timestamp": datetime.now().isoformat()
            }
        })
    except Exception as e:
        logger.error(f"Failed to save zones for camera {camera_id}: {e}")
        return jsonify({
            "success": False,
            "message": "Failed to save zones"
        }), 500


# ==================== NETWORK UTILITY ENDPOINTS ====================


@api_bp.route('/network/interfaces', methods=['GET'])
def list_network_interfaces():
    interfaces = _list_ipv4_interfaces()
    ping_available = shutil.which('ping') is not None

    if not interfaces:
        return jsonify({
            'success': False,
            'message': 'No active IPv4 interfaces detected',
            'data': {
                'interfaces': [],
                'pingAvailable': ping_available
            }
        }), 503

    return jsonify({
        'success': True,
        'data': {
            'interfaces': interfaces,
            'defaultInterface': interfaces[0],
            'pingAvailable': ping_available
        }
    })


@api_bp.route('/network/scan', methods=['GET'])
def network_scan():
    if shutil.which('ping') is None:
        return jsonify({
            'success': False,
            'message': 'Ping utility is not available on this system. Install ping to enable network scanning.'
        }), 501

    interfaces = _list_ipv4_interfaces()
    requested_interface = request.args.get('interface')
    cidr = request.args.get('cidr')
    timeout_ms = request.args.get('timeout_ms', type=int) or 750
    max_hosts = request.args.get('max_hosts', type=int) or 512
    include_unreachable = request.args.get('include_unreachable', 'false').lower() in ('1', 'true', 'yes', 'on')

    max_hosts = max(1, min(max_hosts, 4096))
    timeout_ms = max(100, min(timeout_ms, 5000))

    selected_interface: Optional[Dict[str, object]] = None
    if requested_interface:
        selected_interface = next((iface for iface in interfaces if iface['name'] == requested_interface), None)
        if selected_interface is None and not cidr:
            return jsonify({
                'success': False,
                'message': f'Interface "{requested_interface}" not found'
            }), 404

    network = None
    if cidr:
        try:
            network = ipaddress.ip_network(cidr, strict=False)
        except ValueError:
            return jsonify({
                'success': False,
                'message': f'Invalid CIDR value: {cidr}'
            }), 400
    elif selected_interface:
        network = ipaddress.ip_network(selected_interface['cidr'], strict=False)
    elif interfaces:
        selected_interface = interfaces[0]
        network = ipaddress.ip_network(selected_interface['cidr'], strict=False)
    else:
        return jsonify({
            'success': False,
            'message': 'No active IPv4 interfaces detected'
        }), 503

    if network.version != 4:
        return jsonify({
            'success': False,
            'message': 'Only IPv4 networks are supported at this time'
        }), 400

    total_hosts = network.num_addresses if network.prefixlen >= 31 else max(network.num_addresses - 2, 0)

    local_ip = None
    if selected_interface and selected_interface.get('ip'):
        local_ip = selected_interface['ip']  # type: ignore[index]
        try:
            ipaddress.IPv4Address(local_ip)
        except Exception:
            local_ip = None
        else:
            if ipaddress.IPv4Address(local_ip) not in network:
                local_ip = None

    ordered_hosts: List[str] = []
    if local_ip:
        ordered_hosts.append(local_ip)

    for host in network.hosts():
        host_str = str(host)
        if host_str == local_ip:
            continue
        ordered_hosts.append(host_str)
        if len(ordered_hosts) >= max_hosts * 2:
            # Keep a buffer to allow deduplication while respecting max_hosts later
            break

    seen: Set[str] = set()
    hosts_to_scan: List[str] = []
    for host_ip in ordered_hosts:
        if host_ip in seen:
            continue
        seen.add(host_ip)
        hosts_to_scan.append(host_ip)
        if len(hosts_to_scan) >= max_hosts:
            break

    if not hosts_to_scan:
        return jsonify({
            'success': True,
            'data': {
                'network': str(network),
                'devices': [],
                'host_count': total_hosts,
                'scanned_host_count': 0,
                'reachable_count': 0,
                'interface': selected_interface,
                'parameters': {
                    'timeout_ms': timeout_ms,
                    'max_hosts': max_hosts,
                    'include_unreachable': include_unreachable,
                    'requested_interface': requested_interface,
                    'requested_cidr': cidr
                },
                'elapsed_ms': 0.0,
                'pingAvailable': True
            }
        })

    started_at = time.perf_counter()
    worker_count = min(64, max(1, len(hosts_to_scan)))
    results: List[Dict[str, object]] = []
    reachable_count = 0

    with concurrent.futures.ThreadPoolExecutor(max_workers=worker_count) as executor:
        future_map = {executor.submit(_ping_host, host_ip, timeout_ms): host_ip for host_ip in hosts_to_scan}
        for future in concurrent.futures.as_completed(future_map):
            ip_addr = future_map[future]
            try:
                result = future.result()
            except Exception as exc:
                logger.debug(f"Ping worker failed for {ip_addr}: {exc}")
                continue

            if not isinstance(result, dict):
                continue

            if result.get('reachable'):
                reachable_count += 1

            if include_unreachable or result.get('reachable'):
                results.append(result)

    arp_map = _read_arp_table()
    for entry in results:
        mac = arp_map.get(entry['ip'])
        if mac:
            entry['mac'] = mac

    def _ip_sort_key(value: str):
        try:
            return tuple(int(part) for part in value.split('.'))
        except Exception:
            return (value,)

    results.sort(key=lambda item: _ip_sort_key(item.get('ip', '999.999.999.999')))

    elapsed_ms = round((time.perf_counter() - started_at) * 1000.0, 2)
    scan_window = {
        'start': hosts_to_scan[0],
        'end': hosts_to_scan[-1],
        'count': len(hosts_to_scan)
    }

    if selected_interface is not None:
        selected_interface['is_selected'] = True

    return jsonify({
        'success': True,
        'data': {
            'network': str(network),
            'prefix': network.prefixlen,
            'host_count': total_hosts,
            'scanned_host_count': len(hosts_to_scan),
            'reachable_count': reachable_count,
            'devices': results,
            'interface': selected_interface,
            'interfaces': interfaces,
            'scan_window': scan_window,
            'parameters': {
                'timeout_ms': timeout_ms,
                'max_hosts': max_hosts,
                'include_unreachable': include_unreachable,
                'requested_interface': requested_interface,
                'requested_cidr': cidr
            },
            'arp_entries': len(arp_map),
            'elapsed_ms': elapsed_ms,
            'pingAvailable': True
        }
    })


# ==================== STATISTICS ENDPOINTS ====================

@api_bp.route('/statistics', methods=['GET'])
def get_system_statistics():
    """Get comprehensive system statistics"""
    try:
        stats = {
            "timestamp": datetime.now().isoformat(),
            "system": {
                "uptime": "0s",
                "memory_usage": 0,
                "cpu_usage": 0,
                "disk_usage": 0
            },
            "cameras": {
                "total": 0,
                "online": 0,
                "offline": 0,
                "recording": 0
            },
            "streaming": {
                "active_streams": 0,
                "total_connections": 0,
                "webrtc_connections": 0,
                "bandwidth_usage": 0
            },
            "ai": {
                "models_loaded": 0,
                "processing_queue": 0,
                "detections_today": 0
            },
            "alerts": {
                "total": 0,
                "unread": 0,
                "critical": 0
            }
        }

        # Get camera statistics
        if camera_manager:
            all_cameras = safe_service_call(camera_manager, 'get_all_cameras', [])
            connected_cameras = safe_service_call(camera_manager, 'get_connected_cameras', [])

            stats["cameras"]["total"] = len(all_cameras)
            stats["cameras"]["online"] = len(connected_cameras)
            stats["cameras"]["offline"] = len(all_cameras) - len(connected_cameras)

        # Get streaming statistics
        if stream_server:
            connections = safe_service_call(stream_server, 'get_client_connections', {})
            streams = safe_service_call(stream_server, 'get_all_streams', [])

            stats["streaming"]["active_streams"] = len(streams)
            stats["streaming"]["total_connections"] = len(connections)
            stats["streaming"]["webrtc_connections"] = len([c for c in connections.values()
                                                            if c.get('type') == 'webrtc']) if connections else 0

        # Get AI statistics
        if ai_analyzer:
            models = safe_service_call(ai_analyzer, 'get_loaded_models', [])
            queue_size = safe_service_call(ai_analyzer, 'get_queue_size', 0)

            stats["ai"]["models_loaded"] = len(models) if models else 0
            stats["ai"]["processing_queue"] = queue_size

        # Get alert statistics
        if alert_system:
            alert_counts = safe_service_call(alert_system, 'get_alert_counts', {})
            stats["alerts"].update(alert_counts)

        return jsonify({
            "success": True,
            "data": stats
        })

    except Exception as e:
        logger.error(f"Failed to get statistics: {e}")
        return jsonify({
            "success": False,
            "message": f"Failed to get statistics: {str(e)}"
        }), 500


# ==================== SCHEDULER ENDPOINTS ====================

@api_bp.route('/scheduler/status', methods=['GET'])
def scheduler_status():
    try:
        if not scheduler:
            return jsonify({ 'success': False, 'message': 'Scheduler not available' }), 503
        status = scheduler.get_status()
        return jsonify({ 'success': True, 'data': status })
    except Exception as e:
        logger.error(f"Scheduler status error: {e}")
        return jsonify({ 'success': False, 'message': 'Failed to get scheduler status' }), 500


@api_bp.route('/scheduler/enqueue', methods=['POST'])
def scheduler_enqueue():
    try:
        if not scheduler:
            return jsonify({ 'success': False, 'message': 'Scheduler not available' }), 503
        data = request.get_json() or {}
        from core.scheduler import EventTask
        task = EventTask(
            priority=int(data.get('priority', 5)),
            created_at=time.time(),
            camera_id=data['camera_id'],
            kind=data.get('kind', 'generic'),
            payload=data.get('payload', {}),
            id=data.get('id'),
            not_before=float(data['not_before']) if data.get('not_before') else None
        )
        ok = scheduler.enqueue(task)
        return jsonify({ 'success': ok, 'data': {'enqueued': ok} }), (200 if ok else 429)
    except Exception as e:
        logger.error(f"Scheduler enqueue error: {e}")
        return jsonify({ 'success': False, 'message': 'Failed to enqueue task' }), 500


@api_bp.route('/scheduler/next', methods=['POST'])
def scheduler_next():
    try:
        if not scheduler:
            return jsonify({ 'success': False, 'message': 'Scheduler not available' }), 503
        task = scheduler.get_next_task()
        if not task:
            return jsonify({ 'success': True, 'data': None })
        return jsonify({ 'success': True, 'data': {
            'priority': task.priority,
            'created_at': task.created_at,
            'camera_id': task.camera_id,
            'kind': task.kind,
            'payload': task.payload,
            'id': task.id,
            'not_before': task.not_before,
        } })
    except Exception as e:
        logger.error(f"Scheduler next error: {e}")
        return jsonify({ 'success': False, 'message': 'Failed to get next task' }), 500


@api_bp.route('/scheduler/config', methods=['POST'])
def scheduler_config():
    try:
        if not scheduler:
            return jsonify({ 'success': False, 'message': 'Scheduler not available' }), 503
        data = request.get_json() or {}
        action = data.get('action')
        if action == 'set_cooldown':
            scheduler.set_cooldown(data['camera_id'], data['kind'], float(data['seconds']))
        elif action == 'configure_quota':
            scheduler.configure_quota(data['camera_id'], data['kind'], float(data['window_seconds']), int(data['max_count']))
        else:
            return jsonify({ 'success': False, 'message': 'Unknown action' }), 400
        return jsonify({ 'success': True })
    except Exception as e:
        logger.error(f"Scheduler config error: {e}")
        return jsonify({ 'success': False, 'message': 'Failed to update scheduler config' }), 500


# ==================== AI AGENT ENDPOINTS ====================

@api_bp.route('/ai/chat', methods=['POST'])
def ai_chat():
    """Send a chat message to the AI agent"""
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

        message = data.get('message')
        context_data = data.get('context', {})
        image = data.get('image')  # Optional base64 image

        if not message:
            return jsonify({
                'success': False,
                'message': 'Message is required'
            }), 400

        # Create AI context
        from core.ai_agent import AIContext
        context = AIContext(
            devices=context_data.get('devices', []),
            connections=context_data.get('connections', []),
            layout=context_data.get('layout', []),
            overlays=context_data.get('overlays'),
            system_time=context_data.get('system_time'),
            user_intent=context_data.get('user_intent'),
            permissions=context_data.get('permissions'),
            llm=context_data.get('llm'),
            vision=context_data.get('vision'),
        )

        # Run async chat using shared infrastructure
        response = _run_coro_safe(ai_agent.chat(message, context, image))

        return jsonify({
            'success': True,
            'data': {
                'message': response.message,
                'actions': [asdict(action) for action in response.actions],
                'vision_analysis': response.vision_analysis,
                'vision_analysis_data': response.vision_analysis_data,
                'error': response.error,
                **({'provider': response.provider} if getattr(response, 'provider', None) else {}),
                **({'model': response.model} if getattr(response, 'model', None) else {}),
            }
        })

    except Exception as e:
        logger.error(f"AI chat error: {e}")
        return jsonify({
            'success': False,
            'message': f'AI chat error: {str(e)}'
        }), 500


@api_bp.route('/ai/status', methods=['GET'])
def ai_status():
    """Get AI agent status"""
    try:
        if not ai_agent:
            return jsonify({
                'success': False,
                'message': 'AI agent not available'
            }), 503

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


@api_bp.route('/ai/analysis-memories', methods=['GET'])
def list_analysis_memories():
    try:
        if not db_manager:
            return jsonify({ 'success': False, 'message': 'Database not available' }), 503
        camera_id = request.args.get('camera_id')
        limit = int(request.args.get('limit', '50'))
        items = db_manager.list_analysis_memories(camera_id=camera_id, limit=limit)
        return jsonify({ 'success': True, 'data': { 'memories': items, 'count': len(items) } })
    except Exception as e:
        logger.error(f"List analysis memories error: {e}")
        return jsonify({ 'success': False, 'message': 'Failed to list analysis memories' }), 500


@api_bp.route('/ai/violations', methods=['GET'])
def list_zone_violations():
    try:
        if not db_manager:
            return jsonify({ 'success': False, 'message': 'Database not available' }), 503
        camera_id = request.args.get('camera_id')
        zone_id = request.args.get('zone_id')
        limit = int(request.args.get('limit', '100'))
        items = db_manager.list_zone_violations(camera_id=camera_id, zone_id=zone_id, limit=limit)
        return jsonify({ 'success': True, 'data': { 'violations': items, 'count': len(items) } })
    except Exception as e:
        logger.error(f"List zone violations error: {e}")
        return jsonify({ 'success': False, 'message': 'Failed to list zone violations' }), 500


@api_bp.route('/ai/vision', methods=['POST'])
def ai_vision():
    """Analyze an image with the configured vision provider or local service."""
    try:
        if not ai_agent or not getattr(ai_agent, "provider", None):
            return jsonify({'success': False, 'message': 'AI provider not available'}), 503

        data = request.get_json() or {}
        image = data.get('image')
        prompt = data.get('prompt') or 'Analyze this image'
        if not image:
            return jsonify({'success': False, 'message': 'Image is required'}), 400

        model_override = data.get('model')
        composition = data.get('composition')
        source = data.get('source')
        criteria = data.get('criteria') or []
        include_detections = data.get('include_detections', True)
        use_cache = data.get('use_cache', True)
        
        # User-provided API settings
        cloud_endpoint = data.get('cloudEndpoint')
        cloud_api_key = data.get('cloudApiKey')
        local_endpoint = data.get('localEndpoint')
        max_tokens = data.get('maxTokens', 400)
        temperature = data.get('temperature', 0.3)

        # Execute provider vision call inside a dedicated thread to avoid interference
        import eventlet
        real_threading = eventlet.patcher.original('threading')
        
        result_holder = {'result': None, 'error': None}
        
        def run_vision_thread():
            try:
                result_holder['result'] = _run_coro_safe(
                    ai_agent.provider.vision(
                        image=image,
                        prompt=prompt,
                        model=model_override or ai_agent.vision_model,
                        opts={
                            "temperature": temperature,
                            "max_tokens": max_tokens,
                            "timeout": data.get('timeout', ai_agent.timeout),
                            "source": source,
                            "composition": composition,
                            "criteria": criteria,
                            "include_detections": include_detections,
                            "use_cache": use_cache,
                            # User overrides
                            "cloud_endpoint": cloud_endpoint,
                            "cloud_api_key": cloud_api_key,
                            "local_endpoint": local_endpoint,
                        },
                    )
                )
            except Exception as e:
                result_holder['error'] = e
                
        vision_thread = real_threading.Thread(target=run_vision_thread)
        vision_thread.start()
        vision_thread.join()
        
        if result_holder['error']:
            raise result_holder['error']
            
        result = result_holder['result']

        analysis = result.get('analysis') if isinstance(result, dict) else {}
        caption = analysis.get('caption') if isinstance(analysis, dict) else None
        objects = analysis.get('objects') if isinstance(analysis, dict) else None
        by_model = analysis.get('by_model') if isinstance(analysis, dict) else None
        verdict = analysis.get('verdict') if isinstance(analysis, dict) else None
        caption = caption or result.get('content', '')

        response_payload = {
            'caption': caption,
            'objects': objects or [],
            'models': by_model or [],
            'verdict': verdict,
            'raw': result,
        }

        return jsonify({'success': True, 'data': response_payload})

    except Exception as e:
        logger.error(f"AI vision error: {e}")
        return jsonify({'success': False, 'message': f'AI vision error: {str(e)}'}), 500


@api_bp.route('/ai/actions/execute', methods=['POST'])
def execute_ai_actions():
    """Execute AI actions (widget creation, movement, etc.)"""
    try:
        data = request.get_json()
        if not data:
            return jsonify({
                'success': False,
                'message': 'No data provided'
            }), 400

        actions = data.get('actions', [])
        if not actions:
            return jsonify({
                'success': False,
                'message': 'No actions provided'
            }), 400

        results = []
        for action in actions:
            try:
                action_kind = action.get('kind')
                
                if action_kind == 'create_widget':
                    # This would be handled by the frontend
                    results.append({
                        'action': action_kind,
                        'success': True,
                        'message': 'Widget creation queued for frontend'
                    })
                    
                elif action_kind == 'create_rule':
                    # Create rule in the backend
                    rule_data = {
                        'name': action.get('rule_name', 'AI Generated Rule'),
                        'when': action.get('rule_when', ''),
                        'actions': action.get('rule_actions', [])
                    }
                    
                    # Add rule to the system (you'll need to implement this)
                    # rule_id = create_rule(rule_data)
                    
                    results.append({
                        'action': action_kind,
                        'success': True,
                        'message': 'Rule created successfully'
                    })
                    
                else:
                    results.append({
                        'action': action_kind,
                        'success': False,
                        'message': f'Unknown action type: {action_kind}'
                    })
                    
            except Exception as e:
                results.append({
                    'action': action.get('kind', 'unknown'),
                    'success': False,
                    'message': f'Action execution error: {str(e)}'
                })

        return jsonify({
            'success': True,
            'data': {
                'results': results
            }
        })

    except Exception as e:
        logger.error(f"AI actions execution error: {e}")
        return jsonify({
            'success': False,
            'message': f'AI actions execution error: {str(e)}'
        }), 500


# ==================== ENHANCED AI ENDPOINTS ====================

@api_bp.route('/ai/detect-objects', methods=['POST'])
def ai_detect_objects():
    """Detect objects in an image using MobileNet SSD (default) or YOLOv8"""
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
        model_type = data.get('model_type', 'mobilenet')
        confidence_threshold = data.get('confidence_threshold', 0.5)

        if not image_base64:
            return jsonify({
                'success': False,
                'message': 'Image data is required'
            }), 400

        # Perform object detection
        detections = ai_agent.detect_objects_in_image(image_base64, model_type)
        
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


@api_bp.route('/ai/vehicle-count', methods=['POST'])
def ai_vehicle_count():
    """Count moving vehicles on a roadway using lightweight motion + detector fusion."""
    try:
        if stream_server is None:
            return jsonify({
                'success': False,
                'message': 'Stream server not available'
            }), 503

        data = request.get_json()
        if not data:
            return jsonify({
                'success': False,
                'message': 'No data provided'
            }), 400

        camera_id = data.get('camera_id')
        if not camera_id:
            return jsonify({
                'success': False,
                'message': 'camera_id is required'
            }), 400

        duration_seconds = float(data.get('duration_seconds', 10.0))
        sample_fps = float(data.get('sample_fps', 5.0))
        min_confidence = float(data.get('min_confidence', 0.4))
        use_high_precision = bool(data.get('use_high_precision', True))

        def ensure_stream_running(cam_id: str) -> bool:
            if cam_id in getattr(stream_server, 'active_streams', {}):
                return True

            try:
                from app import cameras_db  # type: ignore
            except Exception:
                cameras_db = []

            camera = next((cam for cam in cameras_db if cam.get('id') == cam_id), None)
            if not camera:
                return False

            rtsp_url = camera.get('rtsp_url')
            if not rtsp_url:
                return False

            config = {
                'rtsp_url': rtsp_url,
                'webrtc_enabled': False,
                'fps': 15
            }
            return _run_coro_safe(stream_server.start_stream(cam_id, config))

        if not ensure_stream_running(camera_id):
            return jsonify({
                'success': False,
                'message': f'Unable to start stream for camera {camera_id}'
            }), 500

        counter = VehicleCounter(
            stream_server=stream_server,
            camera_id=camera_id,
            duration_seconds=duration_seconds,
            sample_fps=sample_fps,
            min_confidence=min_confidence,
            use_high_precision=use_high_precision,
        )

        result = counter.run()

        return jsonify({
            'success': True,
            'data': result
        })

    except Exception as e:
        logger.error(f"Vehicle counting error: {e}", exc_info=True)
        return jsonify({
            'success': False,
            'message': f'Vehicle counting error: {str(e)}'
        }), 500


# ==================== SERVER-SIDE SCREENSHOTS ====================

@api_bp.route('/screenshots', methods=['POST'])
def upload_screenshot():
    """Save a screenshot server-side with unified timestamp naming and metadata.

    Expects JSON body:
    {
      image_base64: string (data URL or raw base64),
      camera_id?: string,
      camera_name?: string,
      motion_type?: string,
      motion_confidence?: number,
      motion_size?: number,
      scene_description?: string,
      analysis_keywords?: string[],
      metadata?: object,
      created_at?: string (ISO)
    }
    Returns: { success, data: { id, file_path, created_at } }
    """
    try:
        if not db_manager:
            return jsonify({ 'success': False, 'message': 'Database not available' }), 503

        data = request.get_json() or {}
        image_b64 = data.get('image_base64')
        if not image_b64 or not isinstance(image_b64, str):
            return jsonify({ 'success': False, 'message': 'image_base64 is required' }), 400

        # Normalize base64 (strip data URL prefix if present)
        if image_b64.startswith('data:image'):
            try:
                image_b64 = image_b64.split(',', 1)[1]
            except Exception:
                pass

        # Decode image
        try:
            img_bytes = base64.b64decode(image_b64)
        except Exception:
            return jsonify({ 'success': False, 'message': 'Invalid base64 image' }), 400

        # Determine image format and dimensions
        np_arr = np.frombuffer(img_bytes, dtype=np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if img is None:
            return jsonify({ 'success': False, 'message': 'Failed to decode image bytes' }), 400
        height, width = img.shape[:2]

        # Ensure storage directory exists
        storage_dir = Path(current_app.config.get('SCREENSHOTS_DIR', 'data/screenshots'))
        storage_dir.mkdir(parents=True, exist_ok=True)

        # Unified timestamp name: YYYYMMDD_HHMMSSmmm_<uuid>.jpg
        created_at = data.get('created_at') or datetime.now().isoformat()
        ts_label = datetime.fromisoformat(created_at.replace('Z','')).strftime('%Y%m%d_%H%M%S%f')[:-3]
        rec_id = str(uuid.uuid4())
        filename = f"{ts_label}_{rec_id}.jpg"
        file_path = storage_dir / filename

        # Re-encode to JPEG for consistency
        ok, enc = cv2.imencode('.jpg', img, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
        if not ok:
            return jsonify({ 'success': False, 'message': 'Failed to encode image' }), 500
        with open(file_path, 'wb') as f:
            f.write(bytearray(enc))

        # Build record
        record = {
            'id': rec_id,
            'created_at': created_at,
            'camera_id': data.get('camera_id'),
            'camera_name': data.get('camera_name'),
            'file_path': str(file_path).replace('\\', '/'),
            'width': int(width),
            'height': int(height),
            'size_bytes': int(file_path.stat().st_size),
            'format': 'jpeg',
            'motion_type': data.get('motion_type'),
            'motion_confidence': data.get('motion_confidence'),
            'motion_size': data.get('motion_size'),
            'scene_description': data.get('scene_description'),
            'local_vision_analysis': data.get('local_vision_analysis'),  # BLIP analysis
            'api_vision_analysis': data.get('api_vision_analysis'),      # GPT-4 analysis
            'analysis_keywords': data.get('analysis_keywords') or [],
            'metadata_json': data.get('metadata') or {}
        }

        rec_id = db_manager.store_server_screenshot(record)
        if not rec_id:
            try: file_path.unlink(missing_ok=True)
            except Exception: pass
            return jsonify({ 'success': False, 'message': 'Failed to store screenshot record' }), 500

        return jsonify({ 'success': True, 'data': { 'id': rec_id, 'file_path': str(file_path), 'created_at': created_at } })
    except Exception as e:
        logger.error(f"Upload screenshot error: {e}")
        return jsonify({ 'success': False, 'message': 'Failed to upload screenshot' }), 500


@api_bp.route('/screenshots', methods=['GET'])
def search_screenshots():
    """Search server-side screenshots.
    Query params: camera_id, date_start, date_end, keyword, limit, offset
    """
    try:
        if not db_manager:
            return jsonify({ 'success': False, 'message': 'Database not available' }), 503
        camera_id = request.args.get('camera_id')
        date_start = request.args.get('date_start')
        date_end = request.args.get('date_end')
        keyword = request.args.get('keyword')
        limit = int(request.args.get('limit', '50'))
        offset = int(request.args.get('offset', '0'))
        items = db_manager.search_server_screenshots(camera_id=camera_id,
                                                     date_start=date_start,
                                                     date_end=date_end,
                                                     keyword=keyword,
                                                     limit=limit,
                                                     offset=offset)
        return jsonify({ 'success': True, 'data': { 'screenshots': items, 'count': len(items) } })
    except Exception as e:
        logger.error(f"Search screenshots error: {e}")
        return jsonify({ 'success': False, 'message': 'Failed to search screenshots' }), 500


@api_bp.route('/screenshots/<shot_id>', methods=['GET'])
def get_screenshot_record(shot_id: str):
    try:
        if not db_manager:
            return jsonify({ 'success': False, 'message': 'Database not available' }), 503
        rec = db_manager.get_server_screenshot(shot_id)
        if not rec:
            return jsonify({ 'success': False, 'message': 'Not found' }), 404
        return jsonify({ 'success': True, 'data': rec })
    except Exception as e:
        logger.error(f"Get screenshot error: {e}")
        return jsonify({ 'success': False, 'message': 'Failed to get screenshot' }), 500


@api_bp.route('/screenshots/<shot_id>/image', methods=['GET'])
def get_screenshot_image(shot_id: str):
    try:
        if not db_manager:
            return jsonify({ 'success': False, 'message': 'Database not available' }), 503
        rec = db_manager.get_server_screenshot(shot_id)
        if not rec:
            return jsonify({ 'success': False, 'message': 'Not found' }), 404
        path = rec.get('file_path')
        if not path or not os.path.exists(path):
            return jsonify({ 'success': False, 'message': 'Image file missing' }), 404
        return send_file(path, mimetype='image/jpeg')
    except Exception as e:
        logger.error(f"Get screenshot image error: {e}")
        return jsonify({ 'success': False, 'message': 'Failed to get screenshot image' }), 500


@api_bp.route('/screenshots/<shot_id>', methods=['DELETE'])
def delete_screenshot_record(shot_id: str):
    try:
        if not db_manager:
            return jsonify({ 'success': False, 'message': 'Database not available' }), 503
        rec = db_manager.get_server_screenshot(shot_id)
        ok = db_manager.delete_server_screenshot(shot_id)
        if ok and rec and rec.get('file_path') and os.path.exists(rec['file_path']):
            try:
                os.remove(rec['file_path'])
            except Exception:
                pass
        return jsonify({ 'success': ok, 'data': { 'deleted': ok, 'id': shot_id } })
    except Exception as e:
        logger.error(f"Delete screenshot error: {e}")
        return jsonify({ 'success': False, 'message': 'Failed to delete screenshot' }), 500

@api_bp.route('/events/bundle', methods=['POST'])
def create_event_bundle():
    """Create an EventBundle with optional snapshot annotation."""
    try:
        data = request.get_json() or {}
        from core.events import EventDetection, EventTrack, EventOverlay, build_event_bundle
        from core.snapshots import draw_overlays, encode_image_base64

        bundle_id = data.get('id') or str(uuid.uuid4())
        camera_id = data['camera_id']
        kind = data.get('kind', 'generic')

        detections = [
            EventDetection(
                class_name=d.get('class_name','object'),
                confidence=float(d.get('confidence',0.0)),
                bbox=d.get('bbox',{})
            ) for d in data.get('detections', [])
        ]

        tracks = [
            EventTrack(
                id=int(t.get('id',0)),
                bbox=t.get('bbox',{}),
                history=t.get('history',[])
            ) for t in data.get('tracks', [])
        ]

        overlays = None
        if data.get('overlays'):
            ov = data['overlays']
            overlays = EventOverlay(
                zones=ov.get('zones',[]),
                lines=ov.get('lines',[]),
                tags=ov.get('tags',[])
            )

        snapshot_b64 = None
        if data.get('draw_snapshot'):
            # Pull latest frame from stream_server
            if not stream_server:
                return jsonify({ 'success': False, 'message': 'Stream server not available for snapshot' }), 503
            # Use medium quality
            frame_bytes = stream_server.get_frame(camera_id)
            if frame_bytes:
                import numpy as np
                arr = np.frombuffer(frame_bytes, dtype=np.uint8)
                frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                zones_px = overlays.zones if overlays else []
                lines_px = overlays.lines if overlays else []
                tracks_px = [t.__dict__ for t in tracks]
                detections_px = [d.__dict__ for d in detections]
                annotated = draw_overlays(frame, zones_px, lines_px, tracks_px, detections_px)
                snapshot_b64 = encode_image_base64(annotated)

        bundle = build_event_bundle(
            bundle_id=bundle_id,
            camera_id=camera_id,
            kind=kind,
            detections=detections,
            tracks=tracks,
            overlays=overlays,
            snapshot_base64=snapshot_b64,
            metadata=data.get('metadata', {})
        )

        # Optional persistence
        if data.get('store', True):
            if not db_manager:
                return jsonify({ 'success': False, 'message': 'Database not available' }), 503
            try:
                import json as pyjson
                ok = db_manager.store_event_bundle(
                    bundle_id=bundle.id,
                    camera_id=bundle.camera_id,
                    kind=bundle.kind,
                    created_at=bundle.created_at,
                    bundle_json=pyjson.dumps(bundle.to_dict())
                )
                if not ok:
                    return jsonify({ 'success': False, 'message': 'Failed to store bundle' }), 500
            except Exception as e:
                logger.error(f"Store event bundle error: {e}")
                return jsonify({ 'success': False, 'message': 'Failed to store bundle' }), 500

        return jsonify({ 'success': True, 'data': bundle.to_dict() })

    except Exception as e:
        logger.error(f"Event bundle error: {e}")
        return jsonify({ 'success': False, 'message': 'Failed to create event bundle' }), 500


@api_bp.route('/events/bundles', methods=['GET'])
def list_event_bundles():
    try:
        if not db_manager:
            return jsonify({ 'success': False, 'message': 'Database not available' }), 503
        camera_id = request.args.get('camera_id')
        kind = request.args.get('kind')
        limit = int(request.args.get('limit', '50'))
        items = db_manager.list_event_bundles(camera_id=camera_id, kind=kind, limit=limit)
        return jsonify({ 'success': True, 'data': { 'bundles': items, 'count': len(items) } })
    except Exception as e:
        logger.error(f"List bundles error: {e}")
        return jsonify({ 'success': False, 'message': 'Failed to list event bundles' }), 500


@api_bp.route('/events/bundles/<bundle_id>', methods=['GET'])
def get_event_bundle(bundle_id: str):
    try:
        if not db_manager:
            return jsonify({ 'success': False, 'message': 'Database not available' }), 503
        rec = db_manager.get_event_bundle(bundle_id)
        if not rec:
            return jsonify({ 'success': False, 'message': 'Not found' }), 404
        return jsonify({ 'success': True, 'data': rec })
    except Exception as e:
        logger.error(f"Get bundle error: {e}")
        return jsonify({ 'success': False, 'message': 'Failed to get event bundle' }), 500


@api_bp.route('/events/bundles/<bundle_id>', methods=['DELETE'])
def delete_event_bundle(bundle_id: str):
    try:
        if not db_manager:
            return jsonify({ 'success': False, 'message': 'Database not available' }), 503
        ok = db_manager.delete_event_bundle(bundle_id)
        return jsonify({ 'success': ok, 'data': { 'deleted': ok, 'id': bundle_id } })
    except Exception as e:
        logger.error(f"Delete bundle error: {e}")
        return jsonify({ 'success': False, 'message': 'Failed to delete event bundle' }), 500


@api_bp.route('/ai/verify', methods=['POST'])
def ai_verify_bundle():
    """Verify an event bundle with AI provider and store AnalysisMemory and ZoneViolations if applicable."""
    try:
        if not ai_agent:
            return jsonify({ 'success': False, 'message': 'AI agent not available' }), 503
        if not db_manager:
            return jsonify({ 'success': False, 'message': 'Database not available' }), 503

        data = request.get_json() or {}
        bundle = data.get('bundle')
        prompt = data.get('prompt', 'Analyze and verify this event.')
        if not bundle:
            return jsonify({ 'success': False, 'message': 'bundle is required' }), 400

        # Serialize context for the agent
        context = {
            'camera_id': bundle.get('camera_id'),
            'kind': bundle.get('kind'),
            'detections': bundle.get('detections', []),
            'tracks': bundle.get('tracks', []),
            'overlays': bundle.get('overlays'),
            'metadata': bundle.get('metadata', {})
        }

        # If snapshot exists, pass it as image
        image_b64 = bundle.get('snapshot_base64')

        from core.ai_agent import AIContext
        ai_context = AIContext(devices=[], connections=[], layout=[], overlays=context.get('overlays'))

        response = _run_coro_safe(ai_agent.chat(prompt, ai_context, image_b64))

        # Store AnalysisMemory
        memory_id = str(uuid.uuid4())
        db_manager.store_analysis_memory(
            memory_id=memory_id,
            bundle_id=bundle.get('id'),
            camera_id=bundle.get('camera_id'),
            created_at=datetime.now().isoformat(),
            message=response.message,
            vision_analysis=(response.vision_analysis or ''),
            verdict_json={ 'actions': [a.__dict__ for a in response.actions], 'error': response.error }
        )

        # Parse structured violations from actions if present
        try:
            for act in (response.actions or []):
                kind = getattr(act, 'kind', None) or act.get('kind') if isinstance(act, dict) else None
                props = getattr(act, 'props', None) or act.get('props') if isinstance(act, dict) else None
                if kind == 'zone_violation' and isinstance(props, dict):
                    violation = {
                        'id': str(uuid.uuid4()),
                        'camera_id': bundle.get('camera_id'),
                        'zone_id': props.get('zone_id'),
                        'zone_name': props.get('zone_name'),
                        'violation_type': props.get('violation_type', 'violation'),
                        'confidence': float(props.get('confidence', 0.5)),
                        'description': props.get('description', ''),
                        'timestamp': datetime.now().isoformat(),
                        'bundle_id': bundle.get('id')
                    }
                    db_manager.store_zone_violation(violation)
        except Exception as e:
            logger.debug(f"No structured violations parsed: {e}")

        # Optionally attach AI verdict summary back to bundle for quick UI
        try:
            if bundle.get('id') and db_manager:
                rec = db_manager.get_event_bundle(bundle.get('id'))
                if rec and rec.get('bundle'):
                    b = rec['bundle']
                    b['ai_verdict'] = {
                        'message': response.message,
                        'vision_analysis': response.vision_analysis,
                        'vision_analysis_data': response.vision_analysis_data,
                        'analysis_memory_id': memory_id
                    }
                    db_manager.update_event_bundle_json(bundle.get('id'), b)
        except Exception as e:
            logger.debug(f"Could not update bundle with AI verdict: {e}")

        return jsonify({
            'success': True,
            'data': {
                'message': response.message,
                'vision_analysis': response.vision_analysis,
                'vision_analysis_data': response.vision_analysis_data,
                'actions': [a.__dict__ for a in response.actions],
                'analysis_memory_id': memory_id
            }
        })

    except Exception as e:
        logger.error(f"AI verify error: {e}")
        return jsonify({ 'success': False, 'message': 'Failed to verify bundle' }), 500


@api_bp.route('/ai/setup-monitoring', methods=['POST'])
def ai_setup_monitoring():
    """Setup monitoring for a specific zone"""
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

        zone_id = data.get('zone_id')
        camera_id = data.get('camera_id')
        enabled = data.get('enabled', True)

        if not zone_id or not camera_id:
            return jsonify({
                'success': False,
                'message': 'Zone ID and camera ID are required'
            }), 400

        success = ai_agent.setup_zone_monitoring(zone_id, camera_id, enabled)

        return jsonify({
            'success': success,
            'data': {
                'zone_id': zone_id,
                'camera_id': camera_id,
                'enabled': enabled,
                'message': 'Zone monitoring setup complete' if success else 'Zone monitoring setup failed'
            }
        })

    except Exception as e:
        logger.error(f"Zone monitoring setup error: {e}")
        return jsonify({
            'success': False,
            'message': f'Zone monitoring setup error: {str(e)}'
        }), 500


@api_bp.route('/ai/interpret-rules', methods=['POST'])
def ai_interpret_rules():
    """Interpret natural language rules for a zone"""
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

        zone_id = data.get('zone_id')
        natural_language_rule = data.get('natural_language_rule')

        if not zone_id or not natural_language_rule:
            return jsonify({
                'success': False,
                'message': 'Zone ID and natural language rule are required'
            }), 400

        # For now, we'll simulate detected objects
        # In a real implementation, you'd get these from actual object detection
        simulated_objects = [
            {'class_name': 'person', 'confidence': 0.85, 'bbox': [100, 100, 50, 100]},
            {'class_name': 'car', 'confidence': 0.75, 'bbox': [200, 150, 80, 60]}
        ]

        triggered_actions = ai_agent.interpret_zone_rules(zone_id, simulated_objects)

        return jsonify({
            'success': True,
            'data': {
                'zone_id': zone_id,
                'rule': natural_language_rule,
                'triggered_actions': triggered_actions,
                'message': f'Interpreted rule for zone {zone_id}'
            }
        })

    except Exception as e:
        logger.error(f"Rule interpretation error: {e}")
        return jsonify({
            'success': False,
            'message': f'Rule interpretation error: {str(e)}'
        }), 500


@api_bp.route('/ai/recall-context', methods=['POST'])
def ai_recall_context():
    """Recall relevant context from memory"""
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

        query = data.get('query')

        if not query:
            return jsonify({
                'success': False,
                'message': 'Query is required'
            }), 400

        relevant_context = ai_agent.recall_context(query)

        return jsonify({
            'success': True,
            'data': {
                'query': query,
                'context': relevant_context,
                'context_count': len(relevant_context),
                'message': f'Found {len(relevant_context)} relevant context entries'
            }
        })

    except Exception as e:
        logger.error(f"Context recall error: {e}")
        return jsonify({
            'success': False,
            'message': f'Context recall error: {str(e)}'
        }), 500


@api_bp.route('/ai/add-rule', methods=['POST'])
def ai_add_rule():
    """Add a natural language rule for a zone"""
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

        zone_id = data.get('zone_id')
        natural_language_rule = data.get('natural_language_rule')
        camera_id = data.get('camera_id')

        if not zone_id or not natural_language_rule:
            return jsonify({
                'success': False,
                'message': 'Zone ID and natural language rule are required'
            }), 400

        rule_id = ai_agent.add_natural_language_rule(zone_id, natural_language_rule, camera_id)

        return jsonify({
            'success': True,
            'data': {
                'rule_id': rule_id,
                'zone_id': zone_id,
                'rule': natural_language_rule,
                'camera_id': camera_id,
                'message': f'Natural language rule added successfully'
            }
        })

    except Exception as e:
        logger.error(f"Add rule error: {e}")
        return jsonify({
            'success': False,
            'message': f'Add rule error: {str(e)}'
        }), 500


@api_bp.route('/ai/rules', methods=['GET'])
def ai_get_rules():
    """Get all natural language rules"""
    try:
        if not ai_agent:
            return jsonify({
                'success': False,
                'message': 'AI agent not available'
            }), 503

        rules = ai_agent.get_natural_language_rules()

        return jsonify({
            'success': True,
            'data': {
                'rules': rules,
                'rule_count': len(rules)
            }
        })

    except Exception as e:
        logger.error(f"Get rules error: {e}")
        return jsonify({
            'success': False,
            'message': f'Get rules error: {str(e)}'
        }), 500


# ==================== AI MODELS ENDPOINTS ====================

@api_bp.route('/ai/models', methods=['GET'])
def ai_get_models():
    """Get all AI models with optional filtering"""
    try:
        if not ai_analyzer:
            return jsonify({
                'success': False,
                'message': 'AI analyzer not available'
            }), 503

        # Get query parameters
        provider = request.args.get('provider')
        status = request.args.get('status')
        capability = request.args.get('capability')
        include_metrics = request.args.get('include_metrics', 'false').lower() == 'true'

        # Get models from AI analyzer
        models = safe_service_call(ai_analyzer, 'get_available_models', [], provider, status, capability)
        
        # Add metrics if requested
        if include_metrics and models:
            for model in models:
                if isinstance(model, dict):
                    model['metrics'] = {
                        'response_time': 0.5,  # Placeholder
                        'success_rate': 0.95,  # Placeholder
                        'last_used': datetime.now().isoformat()
                    }

        return jsonify({
            'success': True,
            'data': {
                'models': models,
                'total_count': len(models),
                'filters_applied': {
                    'provider': provider,
                    'status': status,
                    'capability': capability
                }
            }
        })

    except Exception as e:
        logger.error(f"Get AI models error: {e}")
        return jsonify({
            'success': False,
            'message': f'Get AI models error: {str(e)}'
        }), 500


@api_bp.route('/ai/models/health', methods=['GET'])
def ai_get_models_health():
    """Get health status of all AI models"""
    try:
        if not ai_analyzer:
            return jsonify({
                'success': False,
                'message': 'AI analyzer not available'
            }), 503

        # Get health status from AI analyzer
        health_data = safe_service_call(ai_analyzer, 'get_models_health', {
            'overall_status': 'healthy',
            'models': [],
            'last_check': datetime.now().isoformat()
        })

        return jsonify({
            'success': True,
            'data': health_data
        })

    except Exception as e:
        logger.error(f"Get AI models health error: {e}")
        return jsonify({
            'success': False,
            'message': f'Get AI models health error: {str(e)}'
        }), 500


@api_bp.route('/ai/models/<model_id>', methods=['GET'])
def ai_get_model(model_id):
    """Get a specific AI model by ID"""
    try:
        if not ai_analyzer:
            return jsonify({
                'success': False,
                'message': 'AI analyzer not available'
            }), 503

        model = safe_service_call(ai_analyzer, 'get_model', None, model_id)
        if not model:
            return jsonify({
                'success': False,
                'message': f'Model {model_id} not found'
            }), 404

        return jsonify({
            'success': True,
            'data': model
        })

    except Exception as e:
        logger.error(f"Get AI model error: {e}")
        return jsonify({
            'success': False,
            'message': f'Get AI model error: {str(e)}'
        }), 500


@api_bp.route('/ai/models/<model_id>/health', methods=['GET'])
def ai_get_model_health(model_id):
    """Get health status of a specific model"""
    try:
        if not ai_analyzer:
            return jsonify({
                'success': False,
                'message': 'AI analyzer not available'
            }), 503

        health_data = safe_service_call(ai_analyzer, 'get_model_health', {
            'status': 'healthy',
            'last_check': datetime.now().isoformat(),
            'response_time': 0.5,
            'error_rate': 0.05,
            'uptime': 99.9
        }, model_id)

        return jsonify({
            'success': True,
            'data': health_data
        })

    except Exception as e:
        logger.error(f"Get AI model health error: {e}")
        return jsonify({
            'success': False,
            'message': f'Get AI model health error: {str(e)}'
        }), 500


# ==================== PYTHON SCRIPT AUTOMATION ====================

def _python_scripts_unavailable():
    return jsonify({'success': False, 'message': 'Python script manager not available'}), 503


def _coerce_truthy(value, default=False):
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in ('1', 'true', 'yes', 'on')


@api_bp.route('/python-scripts', methods=['GET'])
def list_python_scripts():
    if not python_script_manager:
        return _python_scripts_unavailable()
    try:
        include_code = _coerce_truthy(request.args.get('include_code'), False)
        scripts = python_script_manager.list_scripts()
        if include_code:
            detailed: List[Dict[str, Any]] = []
            for item in scripts:
                detail = python_script_manager.get_script(item['id'], include_code=True)
                if detail:
                    detailed.append(detail)
            scripts = detailed
        return jsonify({'success': True, 'data': {'scripts': scripts}})
    except Exception as exc:
        logger.error("Python script list error: %s", exc)
        return jsonify({'success': False, 'message': 'Failed to list python scripts'}), 500


@api_bp.route('/python-scripts', methods=['POST'])
def create_python_script():
    if not python_script_manager:
        return _python_scripts_unavailable()
    try:
        data = request.get_json() or {}
        script = python_script_manager.create_script(data)
        return jsonify({'success': True, 'data': script}), 201
    except ValueError as exc:
        return jsonify({'success': False, 'message': str(exc)}), 400
    except Exception as exc:
        logger.error("Python script create error: %s", exc)
        return jsonify({'success': False, 'message': 'Failed to create python script'}), 500


@api_bp.route('/python-scripts/<script_id>', methods=['GET'])
def get_python_script(script_id: str):
    if not python_script_manager:
        return _python_scripts_unavailable()
    try:
        include_code = _coerce_truthy(request.args.get('include_code'), True)
        script = python_script_manager.get_script(script_id, include_code=include_code)
        if not script:
            return jsonify({'success': False, 'message': 'Script not found'}), 404
        return jsonify({'success': True, 'data': script})
    except Exception as exc:
        logger.error("Python script get error: %s", exc)
        return jsonify({'success': False, 'message': 'Failed to load python script'}), 500


@api_bp.route('/python-scripts/<script_id>', methods=['PUT'])
def update_python_script(script_id: str):
    if not python_script_manager:
        return _python_scripts_unavailable()
    try:
        data = dict(request.get_json() or {})
        include_code = _coerce_truthy(data.get('include_code'), True)
        data['include_code'] = include_code
        script = python_script_manager.update_script(script_id, data)
        return jsonify({'success': True, 'data': script})
    except KeyError:
        return jsonify({'success': False, 'message': 'Script not found'}), 404
    except ValueError as exc:
        return jsonify({'success': False, 'message': str(exc)}), 400
    except Exception as exc:
        logger.error("Python script update error: %s", exc)
        return jsonify({'success': False, 'message': 'Failed to update python script'}), 500


@api_bp.route('/python-scripts/<script_id>', methods=['DELETE'])
def delete_python_script(script_id: str):
    if not python_script_manager:
        return _python_scripts_unavailable()
    try:
        deleted = python_script_manager.delete_script(script_id)
        if not deleted:
            return jsonify({'success': False, 'message': 'Script not found'}), 404
        return jsonify({'success': True})
    except Exception as exc:
        logger.error("Python script delete error: %s", exc)
        return jsonify({'success': False, 'message': 'Failed to delete python script'}), 500


@api_bp.route('/python-scripts/<script_id>/run', methods=['POST'])
def run_python_script(script_id: str):
    if not python_script_manager:
        return _python_scripts_unavailable()
    try:
        body = request.get_json() or {}
        event_type = str(body.get('event_type') or 'manual')
        camera_id = body.get('camera_id')
        payload = body.get('payload') if isinstance(body.get('payload'), dict) else {}
        timeout = float(body.get('timeout', 20.0))
        capture_output = _coerce_truthy(body.get('capture_output'), True)

        execution = python_script_manager.run_script(
            script_id,
            event_type=event_type,
            camera_id=camera_id,
            payload=payload,
            timeout=timeout,
            capture_output=capture_output,
        )
        return jsonify({'success': True, 'data': asdict(execution)})
    except KeyError:
        return jsonify({'success': False, 'message': 'Script not found'}), 404
    except Exception as exc:
        logger.error("Python script run error: %s", exc)
        return jsonify({'success': False, 'message': 'Failed to execute python script'}), 500


@api_bp.route('/python-scripts/generate', methods=['POST'])
def generate_python_script():
    if not ai_agent:
        return jsonify({'success': False, 'message': 'AI agent not available'}), 503
    try:
        data = request.get_json() or {}
        prompt = (data.get('prompt') or '').strip()
        if not prompt:
            return jsonify({'success': False, 'message': 'Prompt is required'}), 400

        context_payload = data.get('context') or {}
        from core.ai_agent import AIContext  # Local import to avoid circular dependency

        ai_context = AIContext(
            devices=context_payload.get('devices', []),
            connections=context_payload.get('connections', []),
            layout=context_payload.get('layout', []),
            overlays=context_payload.get('overlays'),
            system_time=datetime.now().isoformat(),
            user_intent={'automation': True, 'script_generation': True},
            permissions=['widget_management', 'rule_creation', 'script_generation']
        )

        requirements = [prompt]
        events = data.get('events')
        if events:
            requirements.append(f"Target events: {', '.join(events)}")
        cameras = data.get('cameras')
        if cameras:
            requirements.append(f"Relevant cameras: {', '.join(cameras)}")

        instruction = (
            "You are an automation engineer for the Knoxnet VMS Beta surveillance platform. "
            "Generate a production-ready Python 3 script that will run inside the Knoxnet VMS Beta automation runtime. "
            "The script must rely only on the Python standard library. "
            "Respond with a single JSON object containing the following keys: "
            "\"code\" (string, full script), \"summary\" (string), and \"notes\" (string). "
            "Do not include Markdown or code fences. The code must be fully runnable without placeholders.\n\n"
            "Requirements:\n- "
        ) + "\n- ".join(requirements)

        import eventlet
        real_threading = eventlet.patcher.original('threading')
        
        result_holder = {'response': None, 'error': None}
        
        def run_chat_thread():
            try:
                result_holder['response'] = _run_coro_safe(ai_agent.chat(instruction, ai_context))
            except Exception as e:
                result_holder['error'] = e

        chat_thread = real_threading.Thread(target=run_chat_thread)
        chat_thread.start()
        chat_thread.join()
        
        if result_holder['error']:
             raise result_holder['error']
             
        response = result_holder['response']

        raw_message = response.message or ""
        parsed = None
        try:
            parsed = json.loads(raw_message)
        except json.JSONDecodeError:
            match = re.search(r'\{.*\}', raw_message, re.DOTALL)
            if match:
                try:
                    parsed = json.loads(match.group(0))
                except json.JSONDecodeError:
                    parsed = None

        if not parsed:
            code_match = re.search(r'```(?:python)?\s*(.*?)```', raw_message, re.DOTALL)
            code = code_match.group(1).strip() if code_match else raw_message.strip()
            parsed = {
                'code': code,
                'summary': response.vision_analysis or 'Generated automation script',
                'notes': response.error or ''
            }

        parsed.setdefault('summary', 'Generated automation script')
        parsed.setdefault('notes', '')

        return jsonify({'success': True, 'data': parsed})
    except Exception as exc:
        logger.error("Python script generation error: %s", exc)
        return jsonify({'success': False, 'message': 'Failed to generate python script'}), 500


# ==================== RULES CRUD ENDPOINTS ====================

@api_bp.route('/rules', methods=['GET'])
def rules_list():
    try:
        if not db_manager:
            return jsonify({'success': False, 'message': 'Database not available'}), 503
        camera_id = request.args.get('camera_id')
        shape_id = request.args.get('shape_id')
        enabled = request.args.get('enabled')
        enabled_bool = None if enabled is None else enabled.lower() == 'true'
        rules = db_manager.list_rules(camera_id=camera_id, shape_id=shape_id, enabled=enabled_bool)
        return jsonify({'success': True, 'data': {'rules': rules, 'count': len(rules)}})
    except Exception as e:
        logger.error(f"Rules list error: {e}")
        return jsonify({'success': False, 'message': 'Failed to list rules'}), 500


@api_bp.route('/rules', methods=['POST'])
def rules_create():
    try:
        if not db_manager:
            return jsonify({'success': False, 'message': 'Database not available'}), 503
        data = request.get_json() or {}
        rule_id = db_manager.create_rule(data)
        if not rule_id:
            return jsonify({'success': False, 'message': 'Failed to create rule'}), 500
        rule = db_manager.get_rule(rule_id)
        return jsonify({'success': True, 'data': rule}), 201
    except Exception as e:
        logger.error(f"Rules create error: {e}")
        return jsonify({'success': False, 'message': 'Failed to create rule'}), 500


@api_bp.route('/rules/<rule_id>', methods=['GET'])
def rules_get(rule_id: str):
    try:
        if not db_manager:
            return jsonify({'success': False, 'message': 'Database not available'}), 503
        rule = db_manager.get_rule(rule_id)
        if not rule:
            return jsonify({'success': False, 'message': 'Rule not found'}), 404
        return jsonify({'success': True, 'data': rule})
    except Exception as e:
        logger.error(f"Rules get error: {e}")
        return jsonify({'success': False, 'message': 'Failed to get rule'}), 500


@api_bp.route('/rules/<rule_id>', methods=['PUT'])
def rules_update(rule_id: str):
    try:
        if not db_manager:
            return jsonify({'success': False, 'message': 'Database not available'}), 503
        updates = request.get_json() or {}
        ok = db_manager.update_rule(rule_id, updates)
        if not ok:
            return jsonify({'success': False, 'message': 'Failed to update rule'}), 400
        rule = db_manager.get_rule(rule_id)
        return jsonify({'success': True, 'data': rule})
    except Exception as e:
        logger.error(f"Rules update error: {e}")
        return jsonify({'success': False, 'message': 'Failed to update rule'}), 500


@api_bp.route('/rules/<rule_id>', methods=['DELETE'])
def rules_delete(rule_id: str):
    try:
        if not db_manager:
            return jsonify({'success': False, 'message': 'Database not available'}), 503
        ok = db_manager.delete_rule(rule_id)
        if not ok:
            return jsonify({'success': False, 'message': 'Failed to delete rule'}), 400
        return jsonify({'success': True, 'data': {'deleted': True, 'id': rule_id}})
    except Exception as e:
        logger.error(f"Rules delete error: {e}")
        return jsonify({'success': False, 'message': 'Failed to delete rule'}), 500


# ==================== DETECTION CONFIGURATION ENDPOINTS ====================

@api_bp.route('/config/detection', methods=['GET'])
def get_detection_config():
    """Get current detection configuration"""
    try:
        # Default configuration
        config = {
            'tier1_enabled': True,
            'tier2_enabled': True,
            'tier2_min_confidence': 0.35,
            'tier2_max_models': 1,
            'tier3_enabled': False,
            'prefer_webrtc': True
        }
        
        # Try to load from file if it exists
        try:
            import json
            config_path = 'data/detection_config.json'
            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    file_config = json.load(f)
                    config.update(file_config)
        except Exception as e:
            logger.warning(f"Failed to load detection config from file: {e}")
        
        return jsonify({'success': True, 'data': config})
    except Exception as e:
        logger.error(f"Get detection config error: {e}")
        return jsonify({'success': False, 'message': 'Failed to get detection config'}), 500


@api_bp.route('/config/detection', methods=['POST'])
def update_detection_config():
    """Update detection configuration"""
    try:
        data = request.get_json() or {}
        
        # Validate required fields
        required_fields = ['tier1_enabled', 'tier2_enabled', 'tier2_min_confidence', 'tier2_max_models', 'tier3_enabled']
        for field in required_fields:
            if field not in data:
                return jsonify({'success': False, 'message': f'Missing required field: {field}'}), 400
        
        # Validate values
        if not isinstance(data['tier1_enabled'], bool):
            return jsonify({'success': False, 'message': 'tier1_enabled must be boolean'}), 400
        if not isinstance(data['tier2_enabled'], bool):
            return jsonify({'success': False, 'message': 'tier2_enabled must be boolean'}), 400
        if not isinstance(data['tier3_enabled'], bool):
            return jsonify({'success': False, 'message': 'tier3_enabled must be boolean'}), 400
        if not isinstance(data['tier2_min_confidence'], (int, float)) or data['tier2_min_confidence'] < 0 or data['tier2_min_confidence'] > 1:
            return jsonify({'success': False, 'message': 'tier2_min_confidence must be between 0 and 1'}), 400
        if not isinstance(data['tier2_max_models'], int) or data['tier2_max_models'] < 1 or data['tier2_max_models'] > 3:
            return jsonify({'success': False, 'message': 'tier2_max_models must be between 1 and 3'}), 400
        
        # Save to file
        try:
            import json
            config_path = 'data/detection_config.json'
            os.makedirs(os.path.dirname(config_path), exist_ok=True)
            with open(config_path, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save detection config to file: {e}")
            return jsonify({'success': False, 'message': 'Failed to save configuration'}), 500
        
        # Update runtime configuration if services are available
        if stream_server and hasattr(stream_server, 'set_detection_config'):
            try:
                stream_server.set_detection_config(data)
            except Exception as e:
                logger.warning(f"Failed to update runtime detection config: {e}")
        
        if ai_analyzer and hasattr(ai_analyzer, 'set_detection_config'):
            try:
                ai_analyzer.set_detection_config(data)
            except Exception as e:
                logger.warning(f"Failed to update AI analyzer detection config: {e}")
        
        return jsonify({'success': True, 'data': data})
    except Exception as e:
        logger.error(f"Update detection config error: {e}")
        return jsonify({'success': False, 'message': 'Failed to update detection config'}), 500


# ==================== ERROR HANDLERS ====================

@api_bp.errorhandler(400)
def bad_request(error):
    return jsonify({
        "success": False,
        "message": "Bad request"
    }), 400


@api_bp.errorhandler(404)
def not_found(error):
    return jsonify({
        "success": False,
        "message": "Resource not found"
    }), 404


@api_bp.errorhandler(500)
def internal_error(error):
    return jsonify({
        "success": False,
        "message": "Internal server error"
    }), 500


@api_bp.route('/detection/status', methods=['GET'])
def get_detection_status():
    """Get comprehensive detection system status across all cameras"""
    try:
        if not stream_server:
            return jsonify({
                "success": False,
                "message": "Stream server not available"
            }), 503

        data = safe_service_call(stream_server, 'get_detection_status', None)
        if data is None:
            return jsonify({
                "success": False,
                "message": "Detection status not available"
            }), 404

        data.update({"timestamp": datetime.now().isoformat()})
        return jsonify({
            "success": True,
            "data": data
        })

    except Exception as e:
        logger.error(f"Failed to get detection status: {e}")
        return jsonify({
            "success": False,
            "message": f"Failed to get detection status: {str(e)}"
        }), 500


@api_bp.route('/detection/logs', methods=['GET'])
def get_detection_logs():
    """Get recent detection logs with optional filtering"""
    try:
        if not stream_server:
            return jsonify({
                "success": False,
                "message": "Stream server not available"
            }), 503

        # Get query parameters
        camera_id = request.args.get('camera_id')
        limit = int(request.args.get('limit', 100))
        since = request.args.get('since')  # ISO timestamp
        object_type = request.args.get('object_type')  # person, car, etc.

        data = safe_service_call(stream_server, 'get_detection_logs', None, 
                               camera_id=camera_id, limit=limit, since=since, object_type=object_type)
        if data is None:
            return jsonify({
                "success": False,
                "message": "Detection logs not available"
            }), 404

        data.update({"timestamp": datetime.now().isoformat()})
        return jsonify({
            "success": True,
            "data": data
        })

    except Exception as e:
        logger.error(f"Failed to get detection logs: {e}")
        return jsonify({
            "success": False,
            "message": f"Failed to get detection logs: {str(e)}"
        }), 500


@api_bp.route('/detection/composites/<track_id>', methods=['GET'])
def get_track_composite(track_id):
    """Get composite image for a specific track"""
    try:
        if not stream_server:
            return jsonify({
                "success": False,
                "message": "Stream server not available"
            }), 503

        camera_id = request.args.get('camera_id')
        if not camera_id:
            return jsonify({
                "success": False,
                "message": "camera_id parameter required"
            }), 400

        data = safe_service_call(stream_server, 'get_track_composite', None, camera_id, track_id)
        if data is None:
            return jsonify({
                "success": False,
                "message": f"Composite not available for track {track_id}"
            }), 404

        return jsonify({
            "success": True,
            "data": data
        })

    except Exception as e:
        logger.error(f"Failed to get track composite for {track_id}: {e}")
        return jsonify({
            "success": False,
            "message": f"Failed to get track composite: {str(e)}"
        }), 500


@api_bp.route('/detection/active-objects', methods=['GET'])
def get_active_objects():
    """Get all currently tracked objects across all cameras"""
    try:
        if not stream_server:
            return jsonify({
                "success": False,
                "message": "Stream server not available"
            }), 503

        data = safe_service_call(stream_server, 'get_active_objects', None)
        if data is None:
            return jsonify({
                "success": False,
                "message": "Active objects data not available"
            }), 404

        data.update({"timestamp": datetime.now().isoformat()})
        return jsonify({
            "success": True,
            "data": data
        })

    except Exception as e:
        logger.error(f"Failed to get active objects: {e}")
        return jsonify({
            "success": False,
            "message": f"Failed to get active objects: {str(e)}"
        }), 500

@api_bp.route('/cameras/recover', methods=['POST'])
def recover_cameras():
    """Recover camera configurations and auto-reconnect"""
    try:
        if not camera_manager:
            return jsonify({
                "success": False,
                "message": "Camera manager not available"
            }), 503

        # Run recovery in background using a real thread
        import eventlet
        real_threading = eventlet.patcher.original('threading')
        
        result_holder = {'recovered_count': 0, 'error': None}
        
        def run_recovery_thread():
            try:
                # Recover configurations
                result_holder['recovered_count'] = _run_coro_safe(
                    camera_manager.recover_camera_configurations()
                ) or 0
                # Auto-reconnect cameras
                _run_coro_safe(camera_manager.auto_reconnect_cameras())
            except Exception as e:
                result_holder['error'] = e
                
        recovery_thread = real_threading.Thread(target=run_recovery_thread)
        recovery_thread.start()
        recovery_thread.join()
        
        if result_holder['error']:
            raise result_holder['error']
            
        recovered_count = result_holder['recovered_count']
            
        return jsonify({
            "success": True,
            "data": {
                "recovered_cameras": recovered_count,
                "message": f"Recovered {recovered_count} camera configurations and initiated reconnection"
            }
        })

    except Exception as e:
        logger.error(f"Failed to recover cameras: {e}")
        return jsonify({
            "success": False,
            "message": f"Failed to recover cameras: {str(e)}"
        }), 500

@api_bp.route('/cameras/<camera_id>/validate', methods=['POST'])
def validate_camera(camera_id):
    """Validate camera connectivity and configuration"""
    try:
        if not camera_manager:
            return jsonify({
                "success": False,
                "message": "Camera manager not available"
            }), 503

        # Run validation in background using a real thread
        import eventlet
        real_threading = eventlet.patcher.original('threading')
        
        result_holder = {'is_valid': False, 'error': None}
        
        def run_validation_thread():
            try:
                result_holder['is_valid'] = bool(
                    _run_coro_safe(camera_manager.validate_camera_connectivity(camera_id))
                )
            except Exception as e:
                result_holder['error'] = e
            finally:
                # _run_coro_safe owns event loop lifecycle; nothing to close here.
                pass
                
        validation_thread = real_threading.Thread(target=run_validation_thread)
        validation_thread.start()
        validation_thread.join()
        
        if result_holder['error']:
            raise result_holder['error']
            
        is_valid = result_holder['is_valid']
        
        return jsonify({
            "success": True,
            "data": {
                "camera_id": camera_id,
                "is_valid": is_valid,
                "message": f"Camera {camera_id} validation {'passed' if is_valid else 'failed'}"
            }
        })

    except Exception as e:
        logger.error(f"Failed to validate camera {camera_id}: {e}")
        return jsonify({
            "success": False,
            "message": f"Failed to validate camera: {str(e)}"
        }), 500


# Adaptive Motion Detection Learning API Endpoints

@api_bp.route('/cameras/<camera_id>/motion/learning/status', methods=['GET'])
def get_motion_learning_status(camera_id):
    """Get adaptive learning status for a specific camera"""
    try:
        if not stream_server:
            return jsonify({
                "success": False,
                "message": "Stream server not available"
            }), 503

        status = stream_server.get_motion_learning_status(camera_id)
        
        if status is None:
            return jsonify({
                "success": False,
                "message": f"No learning status available for camera {camera_id}. Camera may not be active or learning may be disabled."
            }), 404

        return jsonify({
            "success": True,
            "data": status
        })

    except Exception as e:
        logger.error(f"Failed to get learning status for camera {camera_id}: {e}")
        return jsonify({
            "success": False,
            "message": f"Failed to get learning status: {str(e)}"
        }), 500


@api_bp.route('/cameras/motion/learning/status', methods=['GET'])
def get_all_motion_learning_status():
    """Get adaptive learning status for all cameras"""
    try:
        if not stream_server:
            return jsonify({
                "success": False,
                "message": "Stream server not available"
            }), 503

        status = stream_server.get_all_motion_learning_status()
        
        return jsonify({
            "success": True,
            "data": {
                "cameras": status,
                "total_cameras": len(status),
                "learning_enabled_count": sum(1 for s in status.values() if s.get('learning_enabled', False)),
                "timestamp": datetime.now().isoformat()
            }
        })

    except Exception as e:
        logger.error(f"Failed to get all learning status: {e}")
        return jsonify({
            "success": False,
            "message": f"Failed to get learning status: {str(e)}"
        }), 500


@api_bp.route('/cameras/<camera_id>/motion/learning/force-analysis', methods=['POST'])
def force_motion_analysis(camera_id):
    """Force immediate LLM analysis for a camera"""
    try:
        if not stream_server:
            return jsonify({
                "success": False,
                "message": "Stream server not available"
            }), 503

        success = stream_server.force_motion_analysis(camera_id)
        
        if success:
            return jsonify({
                "success": True,
                "message": f"LLM analysis scheduled for camera {camera_id}",
                "data": {
                    "camera_id": camera_id,
                    "analysis_scheduled": True,
                    "timestamp": datetime.now().isoformat()
                }
            })
        else:
            return jsonify({
                "success": False,
                "message": f"Failed to schedule analysis for camera {camera_id}. Camera may not be active or learning may be disabled."
            }), 400

    except Exception as e:
        logger.error(f"Failed to force analysis for camera {camera_id}: {e}")
        return jsonify({
            "success": False,
            "message": f"Failed to force analysis: {str(e)}"
        }), 500


@api_bp.route('/cameras/<camera_id>/motion/learning/enable', methods=['POST'])
def enable_motion_learning(camera_id):
    """Enable or disable adaptive learning for a camera"""
    try:
        if not stream_server:
            return jsonify({
                "success": False,
                "message": "Stream server not available"
            }), 503

        data = request.get_json() or {}
        enabled = data.get('enabled', True)
        
        success = stream_server.enable_motion_learning(camera_id, enabled)
        
        if success:
            return jsonify({
                "success": True,
                "message": f"Adaptive learning {'enabled' if enabled else 'disabled'} for camera {camera_id}",
                "data": {
                    "camera_id": camera_id,
                    "learning_enabled": enabled,
                    "timestamp": datetime.now().isoformat()
                }
            })
        else:
            return jsonify({
                "success": False,
                "message": f"Failed to {'enable' if enabled else 'disable'} learning for camera {camera_id}. Camera may not be active."
            }), 400

    except Exception as e:
        logger.error(f"Failed to set learning state for camera {camera_id}: {e}")
        return jsonify({
            "success": False,
            "message": f"Failed to set learning state: {str(e)}"
        }), 500


@api_bp.route('/cameras/<camera_id>/motion/parameters', methods=['GET'])
def get_motion_parameters(camera_id):
    """Get current motion detection parameters for a camera"""
    try:
        if not stream_server:
            return jsonify({
                "success": False,
                "message": "Stream server not available"
            }), 503

        detector = stream_server.get_motion_detector(camera_id)
        
        if not detector:
            return jsonify({
                "success": False,
                "message": f"No motion detector found for camera {camera_id}. Camera may not be active."
            }), 404

        # Get current parameters
        parameters = {
            "min_area": getattr(detector, 'min_area', 1000),
            "kernel_size": getattr(detector, 'kernel_size', 3),
            "mog2_history": getattr(detector, 'mog2_history', 500),
            "mog2_var_threshold": getattr(detector, 'mog2_var_threshold', 16),
        }
        
        # Get learning status if available
        learning_status = None
        if hasattr(detector, 'get_learning_status'):
            learning_status = detector.get_learning_status()

        return jsonify({
            "success": True,
            "data": {
                "camera_id": camera_id,
                "parameters": parameters,
                "learning_status": learning_status,
                "timestamp": datetime.now().isoformat()
            }
        })

    except Exception as e:
        logger.error(f"Failed to get parameters for camera {camera_id}: {e}")
        return jsonify({
            "success": False,
            "message": f"Failed to get parameters: {str(e)}"
        }), 500


@api_bp.route('/cameras/<camera_id>/motion/parameters', methods=['PUT'])
def update_motion_parameters(camera_id):
    """Manually update motion detection parameters for a camera"""
    try:
        if not stream_server:
            return jsonify({
                "success": False,
                "message": "Stream server not available"
            }), 503

        data = request.get_json() or {}
        
        detector = stream_server.get_motion_detector(camera_id)
        if not detector:
            return jsonify({
                "success": False,
                "message": f"No motion detector found for camera {camera_id}. Camera may not be active."
            }), 404

        # Update parameters with validation
        changes_made = []
        
        if 'min_area' in data:
            new_value = max(500, min(10000, int(data['min_area'])))
            old_value = getattr(detector, 'min_area', 1000)
            detector.min_area = new_value
            changes_made.append(f"min_area: {old_value} -> {new_value}")
        
        if 'mog2_var_threshold' in data:
            new_value = max(8, min(50, int(data['mog2_var_threshold'])))
            old_value = getattr(detector, 'mog2_var_threshold', 16)
            detector.mog2_var_threshold = new_value
            changes_made.append(f"mog2_var_threshold: {old_value} -> {new_value}")
            
            # Recreate background subtractor
            import cv2
            detector.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
                history=getattr(detector, 'mog2_history', 500),
                varThreshold=new_value,
                detectShadows=False
            )
        
        if 'mog2_history' in data:
            new_value = max(100, min(1000, int(data['mog2_history'])))
            old_value = getattr(detector, 'mog2_history', 500)
            detector.mog2_history = new_value
            changes_made.append(f"mog2_history: {old_value} -> {new_value}")
            
            # Recreate background subtractor
            import cv2
            detector.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
                history=new_value,
                varThreshold=getattr(detector, 'mog2_var_threshold', 16),
                detectShadows=False
            )
        
        if 'kernel_size' in data:
            new_value = max(1, min(7, int(data['kernel_size'])))
            old_value = getattr(detector, 'kernel_size', 3)
            detector.kernel_size = new_value
            changes_made.append(f"kernel_size: {old_value} -> {new_value}")

        return jsonify({
            "success": True,
            "message": f"Parameters updated for camera {camera_id}",
            "data": {
                "camera_id": camera_id,
                "changes_made": changes_made,
                "timestamp": datetime.now().isoformat()
            }
        })

    except Exception as e:
        logger.error(f"Failed to update parameters for camera {camera_id}: {e}")
        return jsonify({
            "success": False,
            "message": f"Failed to update parameters: {str(e)}"
        }), 500


# Scene Analysis API Endpoints

@api_bp.route('/cameras/<camera_id>/scene-analysis/enable', methods=['POST'])
def enable_scene_analysis(camera_id):
    """Enable or disable scene analysis for a camera"""
    try:
        if not stream_server:
            return jsonify({
                "success": False,
                "message": "Stream server not available"
            }), 503

        data = request.get_json() or {}
        enabled = data.get('enabled', True)
        
        success = stream_server.enable_scene_analysis(camera_id, enabled)
        
        if success:
            return jsonify({
                "success": True,
                "message": f"Scene analysis {'enabled' if enabled else 'disabled'} for camera {camera_id}",
                "data": {
                    "camera_id": camera_id,
                    "scene_analysis_enabled": enabled,
                    "timestamp": datetime.now().isoformat()
                }
            })
        else:
            return jsonify({
                "success": False,
                "message": f"Failed to {'enable' if enabled else 'disable'} scene analysis for camera {camera_id}. Camera may not be active."
            }), 400

    except Exception as e:
        logger.error(f"Failed to set scene analysis state for camera {camera_id}: {e}")
        return jsonify({
            "success": False,
            "message": f"Failed to set scene analysis state: {str(e)}"
        }), 500


@api_bp.route('/cameras/<camera_id>/scene-analysis/status', methods=['GET'])
def get_scene_analysis_status(camera_id):
    """Get scene analysis status for a specific camera"""
    try:
        if not stream_server:
            return jsonify({
                "success": False,
                "message": "Stream server not available"
            }), 503

        status = stream_server.get_scene_analysis_status(camera_id)
        
        if status is None:
            return jsonify({
                "success": False,
                "message": f"No scene analysis status available for camera {camera_id}. Camera may not be active or scene analysis may be disabled."
            }), 404

        return jsonify({
            "success": True,
            "data": status
        })

    except Exception as e:
        logger.error(f"Failed to get scene analysis status for camera {camera_id}: {e}")
        return jsonify({
            "success": False,
            "message": f"Failed to get scene analysis status: {str(e)}"
        }), 500


@api_bp.route('/cameras/<camera_id>/scene-analysis/history', methods=['GET'])
def get_scene_history(camera_id):
    """Get scene analysis history for a specific camera"""
    try:
        if not stream_server:
            return jsonify({
                "success": False,
                "message": "Stream server not available"
            }), 503

        limit = request.args.get('limit', 10, type=int)
        limit = max(1, min(50, limit))  # Clamp between 1 and 50
        
        history = stream_server.get_scene_history(camera_id, limit)
        
        if history is None:
            return jsonify({
                "success": False,
                "message": f"No scene history available for camera {camera_id}. Camera may not be active or scene analysis may be disabled."
            }), 404

        return jsonify({
            "success": True,
            "data": {
                "camera_id": camera_id,
                "history": history,
                "count": len(history),
                "limit": limit,
                "timestamp": datetime.now().isoformat()
            }
        })

    except Exception as e:
        logger.error(f"Failed to get scene history for camera {camera_id}: {e}")
        return jsonify({
            "success": False,
            "message": f"Failed to get scene history: {str(e)}"
        }), 500


@api_bp.route('/cameras/<camera_id>/scene-analysis/force', methods=['POST'])
def force_scene_analysis(camera_id):
    """Force immediate scene analysis for a camera"""
    try:
        if not stream_server:
            return jsonify({
                "success": False,
                "message": "Stream server not available"
            }), 503

        success = stream_server.force_scene_analysis(camera_id)
        
        if success:
            return jsonify({
                "success": True,
                "message": f"Scene analysis scheduled for camera {camera_id}",
                "data": {
                    "camera_id": camera_id,
                    "analysis_scheduled": True,
                    "timestamp": datetime.now().isoformat()
                }
            })
        else:
            return jsonify({
                "success": False,
                "message": f"Failed to schedule scene analysis for camera {camera_id}. Camera may not be active or scene analysis may be disabled."
            }), 400

    except Exception as e:
        logger.error(f"Failed to force scene analysis for camera {camera_id}: {e}")
        return jsonify({
            "success": False,
            "message": f"Failed to force scene analysis: {str(e)}"
        }), 500


@api_bp.route('/cameras/<camera_id>/motion/tuner/preview', methods=['GET'])
def get_motion_tuner_preview(camera_id):
    """Return a simple preview for the motion tuner: frame, mask, overlay as base64 strings."""
    try:
        if not stream_server:
            return jsonify({
                "success": False,
                "message": "Stream server not available"
            }), 503

        preview = safe_service_call(stream_server, 'get_motion_tuner_preview', None, camera_id)
        if not preview:
            return jsonify({
                "success": False,
                "message": "Preview not available"
            }), 404

        return jsonify({
            "success": True,
            "data": preview
        })
    except Exception as e:
        logger.error(f"Failed to get motion tuner preview for camera {camera_id}: {e}")
        return jsonify({
            "success": False,
            "message": f"Failed to get preview: {str(e)}"
        }), 500

@api_bp.route('/cameras/<camera_id>/detection-config', methods=['GET'])
def get_camera_detection_config(camera_id: str):
    try:
        if not stream_server:
            return jsonify({ 'success': False, 'message': 'Stream server not available' }), 503
        cfg = stream_server.get_detection_config(camera_id)
        return jsonify({ 'success': True, 'data': cfg })
    except Exception as e:
        logger.error(f"Failed to get detection config for camera {camera_id}: {e}")
        return jsonify({ 'success': False, 'message': 'Failed to get detection config' }), 500


@api_bp.route('/cameras/<camera_id>/detection-config', methods=['PUT'])
def update_camera_detection_config(camera_id: str):
    try:
        if not stream_server:
            return jsonify({ 'success': False, 'message': 'Stream server not available' }), 503
        updates = request.get_json() or {}
        ok = stream_server.update_detection_config(camera_id, updates)
        if not ok:
            return jsonify({ 'success': False, 'message': 'Failed to update detection config' }), 400
        cfg = stream_server.get_detection_config(camera_id)
        return jsonify({ 'success': True, 'data': cfg })
    except Exception as e:
        logger.error(f"Failed to update detection config for camera {camera_id}: {e}")
        return jsonify({ 'success': False, 'message': 'Failed to update detection config' }), 500


# ==================== Detection Feedback & Hints ====================

@api_bp.route('/detections/feedback', methods=['POST'])
def submit_detection_feedback():
    try:
        data = request.get_json(force=True) or {}
        camera_id = str(data.get('camera_id') or data.get('cameraId') or '')
        if not camera_id:
            return jsonify({'success': False, 'error': 'camera_id is required'}), 400
        kind = str(data.get('kind') or 'correction')
        if kind not in ('correction','hint'):
            kind = 'correction'
        rec = {
            'id': str(uuid.uuid4()),
            'camera_id': camera_id,
            'timestamp': datetime.now().isoformat(),
            'kind': kind,
            'object_class': (data.get('object_class') or data.get('label') or data.get('class_name') or '').strip().lower() or None,
            'correct': bool(data.get('correct')) if kind == 'correction' else None,
            'bbox': data.get('bbox') or {},
            'detection_meta': data.get('meta') or {},
            'note': data.get('note')
        }
        ok = False
        try:
            if db_manager is not None and hasattr(db_manager, 'store_detection_feedback'):
                ok = db_manager.store_detection_feedback(rec)
        except Exception as e:
            logging.getLogger(__name__).error(f"Failed to store feedback: {e}")
        return jsonify({'success': bool(ok), 'id': rec['id']})
    except Exception as e:
        logging.getLogger(__name__).error(f"feedback error: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


@api_bp.route('/detections/hint', methods=['POST'])
def hint_object_search():
    try:
        data = request.get_json(force=True) or {}
        camera_id = str(data.get('camera_id') or data.get('cameraId') or '')
        if not camera_id:
            return jsonify({'success': False, 'error': 'camera_id is required'}), 400
        target_class = (data.get('object_class') or data.get('target_class') or data.get('label') or '').strip().lower() or None
        bbox = data.get('bbox') or {}

        # Run a focused detection to respond to the hint
        detections = []
        try:
            ss = current_app.config.get('stream_server') if current_app else None
        except Exception:
            ss = None
        if ss is None:
            ss = stream_server
        try:
            if ss is not None:
                frame_bytes = ss.get_frame(camera_id)
                if frame_bytes is not None:
                    import cv2
                    arr = np.frombuffer(frame_bytes, dtype=np.uint8)
                    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                    from core.detector_manager import get_detector_manager
                    dm = get_detector_manager()
                    h, w = frame.shape[:2]
                    region = None
                    if isinstance(bbox, dict) and all(k in bbox for k in ('x','y','w','h')):
                        region = {
                            'x': int(max(0, min(1.0, float(bbox.get('x',0.0)))) * w),
                            'y': int(max(0, min(1.0, float(bbox.get('y',0.0)))) * h),
                            'w': int(max(0, min(1.0, float(bbox.get('w',1.0)))) * w),
                            'h': int(max(0, min(1.0, float(bbox.get('h',1.0)))) * h),
                        }
                    if region:
                        detections = dm.detect_in_region(camera_id, frame, region, conf_threshold=0.2) or []
                    else:
                        detections = dm.detect_and_track(camera_id, frame, conf_threshold=0.2).get('detections', [])
                    if target_class:
                        detections = [d for d in detections if str(d.get('class','')).lower() == target_class]
        except Exception as e:
            logging.getLogger(__name__).warning(f"hint detection failed: {e}")

        # Store hint feedback
        try:
            if db_manager is not None and hasattr(db_manager, 'store_detection_feedback'):
                db_manager.store_detection_feedback({
                    'id': str(uuid.uuid4()),
                    'camera_id': camera_id,
                    'timestamp': datetime.now().isoformat(),
                    'kind': 'hint',
                    'object_class': target_class,
                    'bbox': bbox,
                    'detection_meta': {'source': 'hint_api'},
                })
        except Exception:
            pass

        return jsonify({'success': True, 'detections': detections})
    except Exception as e:
        logging.getLogger(__name__).error(f"hint error: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


# ==================== SMART DETECTION / CPU OPTIMIZATION ENDPOINTS ====================

@api_bp.route('/detector/smart-detection/config', methods=['GET', 'PUT'])
def smart_detection_config():
    """Get or update smart detection configuration (CPU optimization settings)"""
    try:
        from core.detector_manager import get_detector_manager
        dm = get_detector_manager()
        
        if request.method == 'GET':
            config = dm.get_smart_detection_config()
            return jsonify({
                "success": True,
                "data": config,
                "message": "Smart detection config retrieved"
            })
        
        elif request.method == 'PUT':
            data = request.get_json()
            if not data:
                return jsonify({
                    "success": False,
                    "message": "No configuration data provided"
                }), 400
            
            dm.set_smart_detection_config(data)
            return jsonify({
                "success": True,
                "message": "Smart detection config updated",
                "data": dm.get_smart_detection_config()
            })
            
    except Exception as e:
        logger.error(f"Error managing smart detection config: {e}")
        return jsonify({
            "success": False,
            "message": str(e)
        }), 500


@api_bp.route('/detector/skip-stats', methods=['GET'])
@api_bp.route('/detector/skip-stats/<camera_id>', methods=['GET'])
def detection_skip_stats(camera_id=None):
    """Get detection skip statistics for CPU optimization monitoring"""
    try:
        from core.detector_manager import get_detector_manager
        dm = get_detector_manager()
        
        stats = dm.get_detection_skip_stats(camera_id)
        
        return jsonify({
            "success": True,
            "data": stats,
            "message": "Detection skip statistics retrieved"
        })
            
    except Exception as e:
        logger.error(f"Error getting detection skip stats: {e}")
        return jsonify({
            "success": False,
            "message": str(e)
        }), 500


# ==================== DEPTH MAP ENDPOINTS ====================

@api_bp.route('/depth/start/<camera_id>', methods=['POST'])
def start_depth_processing(camera_id):
    """Start depth map processing for a camera"""
    try:
        from core.depth_processor import get_depth_processor, DepthConfig, DepthMode, ColorMap
        
        depth_processor = get_depth_processor()
        
        # Parse configuration from request
        data = request.get_json() or {}
        
        config = DepthConfig()
        
        # Update config from request
        if 'mode' in data:
            mode_str = data['mode'].upper()
            if hasattr(DepthMode, mode_str):
                config.mode = getattr(DepthMode, mode_str)
        
        if 'color_map' in data:
            cmap_str = data['color_map'].upper()
            if hasattr(ColorMap, cmap_str):
                config.color_map = getattr(ColorMap, cmap_str)
        
        if 'fps_limit' in data:
            config.fps_limit = int(data['fps_limit'])
        
        if 'enable_orb' in data:
            config.enable_orb = bool(data['enable_orb'])
        
        if 'orb_features' in data:
            config.orb_features = int(data['orb_features'])
        
        if 'num_disparities' in data:
            config.num_disparities = int(data['num_disparities'])
        
        if 'block_size' in data:
            config.block_size = int(data['block_size'])
        
        # DepthAnythingV2 specific settings
        if 'model_size' in data:
            config.model_size = str(data['model_size'])
        
        if 'device' in data:
            config.device = str(data['device'])
        
        if 'use_fp16' in data:
            config.use_fp16 = bool(data['use_fp16'])
        
        if 'optimize' in data:
            config.optimize = bool(data['optimize'])
        
        # Start processing (will automatically restart if already active)
        # The depth_processor.start_processing handles duplicate checks internally
        success = depth_processor.start_processing(camera_id, config)
        
        if success:
            return jsonify({
                "success": True,
                "message": f"Depth processing started for camera {camera_id}",
                "data": {
                    "camera_id": camera_id,
                    "mode": config.mode.value,
                    "color_map": config.color_map.name,
                    "fps_limit": config.fps_limit
                }
            })
        else:
            return jsonify({
                "success": False,
                "message": "Failed to start depth processing after retry"
            }), 400
            
    except Exception as e:
        logger.error(f"Error starting depth processing: {e}")
        traceback.print_exc()
        return jsonify({
            "success": False,
            "message": str(e)
        }), 500


@api_bp.route('/depth/stop/<camera_id>', methods=['POST'])
def stop_depth_processing(camera_id):
    """
    Stop depth map processing for a camera
    Idempotent operation - safe to call multiple times
    """
    try:
        from core.depth_processor import get_depth_processor
        
        depth_processor = get_depth_processor()
        
        # Check if processing is active
        was_active = camera_id in depth_processor.active_processors
        
        # Stop processing (idempotent - safe to call even if not running)
        success = depth_processor.stop_processing(camera_id)
        
        # Always return success for idempotent behavior
        return jsonify({
            "success": True,
            "message": f"Depth processing stopped for camera {camera_id}" if was_active else f"Depth processing was not active for camera {camera_id}",
            "was_active": was_active
        })
            
    except Exception as e:
        logger.error(f"Error stopping depth processing: {e}")
        traceback.print_exc()
        return jsonify({
            "success": False,
            "message": str(e)
        }), 500


@api_bp.route('/depth/config/<camera_id>', methods=['PUT'])
def update_depth_config(camera_id):
    """Update depth processing configuration"""
    try:
        from core.depth_processor import get_depth_processor
        
        depth_processor = get_depth_processor()
        data = request.get_json() or {}
        
        success = depth_processor.update_config(camera_id, data)
        
        if success:
            return jsonify({
                "success": True,
                "message": "Depth configuration updated"
            })
        else:
            return jsonify({
                "success": False,
                "message": "Camera not found in active processors"
            }), 404
            
    except Exception as e:
        logger.error(f"Error updating depth config: {e}")
        return jsonify({
            "success": False,
            "message": str(e)
        }), 500


@api_bp.route('/depth/active', methods=['GET'])
def get_active_depth_processors():
    """Get list of active depth processors"""
    try:
        from core.depth_processor import get_depth_processor
        import torch
        
        depth_processor = get_depth_processor()
        active = depth_processor.get_active_processors()
        
        # Get GPU info if available
        gpu_info = None
        if torch.cuda.is_available():
            gpu_info = {
                "available": True,
                "device_name": torch.cuda.get_device_name(0),
                "device_count": torch.cuda.device_count(),
                "memory_allocated": torch.cuda.memory_allocated(0) / (1024**3),
                "memory_reserved": torch.cuda.memory_reserved(0) / (1024**3),
                "memory_total": torch.cuda.get_device_properties(0).total_memory / (1024**3)
            }
        else:
            gpu_info = {
                "available": False,
                "message": "CUDA not available - install CUDA-enabled PyTorch"
            }
        
        # Get current device being used
        device_info = None
        if depth_processor.depth_anything:
            device_info = {
                "device": depth_processor.depth_anything.device,
                "model_size": depth_processor.depth_anything.model_size,
                "use_fp16": depth_processor.depth_anything.use_fp16,
                "optimize": depth_processor.depth_anything.optimize
            }
        
        return jsonify({
            "success": True,
            "data": {
                "active_cameras": active,
                "count": len(active),
                "gpu_info": gpu_info,
                "depth_anything_info": device_info
            }
        })
            
    except Exception as e:
        logger.error(f"Error getting active depth processors: {e}")
        return jsonify({
            "success": False,
            "message": str(e)
        }), 500


@api_bp.route('/depth/status/<camera_id>', methods=['GET'])
def get_depth_status(camera_id):
    """Get depth processing status for a camera"""
    try:
        from core.depth_processor import get_depth_processor
        
        depth_processor = get_depth_processor()
        active = depth_processor.get_active_processors()
        is_active = camera_id in active
        
        config = None
        if is_active and camera_id in depth_processor.depth_configs:
            cfg = depth_processor.depth_configs[camera_id]
            config = {
                "mode": cfg.mode.value,
                "color_map": cfg.color_map.name,
                "fps_limit": cfg.fps_limit,
                "enable_orb": cfg.enable_orb,
                "orb_features": cfg.orb_features,
                "model_size": cfg.model_size,
                "device": cfg.device,
                "use_fp16": cfg.use_fp16,
                "optimize": cfg.optimize
            }
        
        return jsonify({
            "success": True,
            "data": {
                "camera_id": camera_id,
                "is_active": is_active,
                "config": config
            }
        })
            
    except Exception as e:
        logger.error(f"Error getting depth status: {e}")
        return jsonify({
            "success": False,
            "message": str(e)
        }), 500


@api_bp.route('/depth/process', methods=['POST'])
def process_depth_frame():
    """
    Process a frame for depth estimation
    OPTIMIZED: Returns latest frame from stream server (no decode overhead)
    FALLBACK: Processes uploaded frame if stream server not available
    """
    try:
        from core.depth_processor import get_depth_processor
        import base64
        
        data = request.get_json() or {}
        camera_id = data.get('camera_id')
        
        if not camera_id:
            return jsonify({
                "success": False,
                "message": "Missing camera_id"
            }), 400
        
        depth_processor = get_depth_processor()
        
        # Do not auto-start depth processing here to avoid unexpected conflicts
        if camera_id not in depth_processor.active_processors:
            return jsonify({
                "success": True,
                "skipped": True,
                "message": "Depth processing is not active for this camera",
                "hint": "Call /depth/start/<camera_id> first"
            })
        
        # FAST PATH: If stream server is linked, return latest processed frame
        # This avoids ALL encoding/decoding overhead!
        if depth_processor.stream_server:
            latest_frame = depth_processor.get_latest_depth_frame(camera_id)
            
            if latest_frame:
                # Encode and return latest frame (already processed from stream)
                encoded = depth_processor.encode_depth_frame(latest_frame)
                
                # Add performance info
                processor_data = depth_processor.active_processors.get(camera_id, {})
                encoded['fps'] = processor_data.get('fps', 0)
                encoded['source'] = 'stream_server'  # Fast path indicator
                encoded['mode'] = 'optimized'
                
                return jsonify({
                    "success": True,
                    "data": encoded
                })
            else:
                # No frame yet - stream starting up
                return jsonify({
                    "success": True,
                    "skipped": True,
                    "message": "Waiting for stream frames...",
                    "source": "stream_server"
                })
        
        # SLOW FALLBACK PATH: Manual frame processing (requires decode)
        frame_data = data.get('frame', '')
        if not frame_data:
            return jsonify({
                "success": True,
                "skipped": True,
                "message": "No frame data (stream server not linked)"
            })
        
        # Decode base64 image
        try:
            if 'base64,' in frame_data:
                frame_data = frame_data.split('base64,')[1]
            
            img_bytes = base64.b64decode(frame_data)
            nparr = np.frombuffer(img_bytes, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if frame is None:
                return jsonify({
                    "success": False,
                    "message": "Failed to decode frame"
                }), 400
        except Exception as e:
            logger.error(f"Error decoding frame: {e}")
            return jsonify({
                "success": False,
                "message": f"Failed to decode frame: {str(e)}"
            }), 400
        
        # Process frame manually
        depth_frame = depth_processor.process_frame(camera_id, frame)
        
        if depth_frame is None:
            return jsonify({
                "success": True,
                "skipped": True,
                "message": "Frame skipped (FPS limiting)"
            })
        
        # Encode result
        encoded = depth_processor.encode_depth_frame(depth_frame)
        encoded['source'] = 'manual'  # Slow fallback indicator
        
        return jsonify({
            "success": True,
            "data": encoded
        })
        
    except Exception as e:
        logger.error(f"Error processing depth frame: {e}")
        traceback.print_exc()
        return jsonify({
            "success": False,
            "message": str(e)
        }), 500


@api_bp.route('/depth/frame/<camera_id>', methods=['GET'])
def get_depth_frame(camera_id):
    """Get a single depth frame for a camera"""
    try:
        from core.depth_processor import get_depth_processor
        
        # Get services
        depth_processor = get_depth_processor()
        frame = None
        
        # Try multiple sources for frames
        # 1. Try camera_manager active streams
        if camera_manager and camera_id in camera_manager.active_streams:
            try:
                cap = camera_manager.active_streams[camera_id]
                ret, frame = cap.read()
                if not ret or frame is None:
                    logger.debug(f"Failed to read from active_streams for {camera_id}")
                    frame = None
            except Exception as e:
                logger.debug(f"Error reading from active_streams: {e}")
                frame = None
        
        # 2. Try to open camera directly via RTSP
        if frame is None and camera_manager:
            try:
                camera_config = camera_manager.get_camera(camera_id)
                if camera_config and camera_config.rtsp_url:
                    logger.debug(f"Attempting direct RTSP capture for {camera_id}")
                    cap = cv2.VideoCapture(camera_config.rtsp_url)
                    if cap.isOpened():
                        ret, frame = cap.read()
                        cap.release()
                        if not ret or frame is None:
                            logger.debug(f"Failed to read from direct RTSP for {camera_id}")
                            frame = None
            except Exception as e:
                logger.debug(f"Error reading from direct RTSP: {e}")
                frame = None
        
        # 3. Check if camera is configured but not connected
        if frame is None:
            if camera_manager:
                camera_config = camera_manager.get_camera(camera_id)
                if camera_config:
                    return jsonify({
                        "success": False,
                        "message": "Camera not actively streaming. Please ensure the camera is connected and streaming.",
                        "hint": f"RTSP URL: {camera_config.rtsp_url}"
                    }), 404
            
            return jsonify({
                "success": False,
                "message": "Camera not found or not streaming"
            }), 404
        
        # Process frame
        depth_frame = depth_processor.process_frame(camera_id, frame)
        
        if depth_frame is None:
            return jsonify({
                "success": False,
                "message": "Failed to process depth frame. Check if depth processing is started."
            }), 500
        
        # Encode frame
        encoded = depth_processor.encode_depth_frame(depth_frame)
        
        return jsonify({
            "success": True,
            "data": encoded
        })
            
    except Exception as e:
        logger.error(f"Error getting depth frame for {camera_id}: {e}")
        traceback.print_exc()
        return jsonify({
            "success": False,
            "message": str(e)
        }), 500


# ============================================================
# LLM Service Management Endpoints
# ============================================================

@api_bp.route('/llm/service/status', methods=['GET'])
def llm_service_status():
    """Get LLM service status"""
    try:
        if not llm_service:
            return jsonify({
                'success': False,
                'message': 'LLM service not initialized',
                'status': 'not_initialized'
            }), 503
        
        status = llm_service.get_status()
        return jsonify({
            'success': True,
            'status': status
        })
    except Exception as e:
        logger.error(f"Error getting LLM service status: {e}")
        return jsonify({
            'success': False,
            'message': str(e)
        }), 500


@api_bp.route('/llm/service/start', methods=['POST'])
def llm_service_start():
    """Start the LLM service"""
    try:
        if not llm_service:
            return jsonify({
                'success': False,
                'message': 'LLM service not initialized'
            }), 503
        
        if not getattr(llm_service, "manage_process", True):
            return jsonify({
                'success': False,
                'message': 'LLM service is managed by Docker Compose; use docker compose to control it.'
            }), 405
        
        data = request.get_json() or {}
        model_name = data.get('model_name')
        
        success = llm_service.start(model_name=model_name, background=True)
        
        if success:
            return jsonify({
                'success': True,
                'message': 'LLM service started successfully',
                'status': llm_service.get_status()
            })
        else:
            return jsonify({
                'success': False,
                'message': 'Failed to start LLM service'
            }), 500
            
    except Exception as e:
        logger.error(f"Error starting LLM service: {e}")
        return jsonify({
            'success': False,
            'message': str(e)
        }), 500


@api_bp.route('/llm/service/stop', methods=['POST'])
def llm_service_stop():
    """Stop the LLM service"""
    try:
        if not llm_service:
            return jsonify({
                'success': False,
                'message': 'LLM service not initialized'
            }), 503
        
        if not getattr(llm_service, "manage_process", True):
            return jsonify({
                'success': False,
                'message': 'LLM service is managed externally and cannot be stopped from the API.'
            }), 405
        
        success = llm_service.stop()
        
        if success:
            return jsonify({
                'success': True,
                'message': 'LLM service stopped successfully'
            })
        else:
            return jsonify({
                'success': False,
                'message': 'Failed to stop LLM service'
            }), 500
            
    except Exception as e:
        logger.error(f"Error stopping LLM service: {e}")
        return jsonify({
            'success': False,
            'message': str(e)
        }), 500


@api_bp.route('/llm/models/available', methods=['GET'])
def llm_models_available():
    """List available/cached models"""
    try:
        if not llm_service:
            return jsonify({
                'success': False,
                'message': 'LLM service not initialized',
                'models': []
            }), 503
        
        models = llm_service.list_models()
        
        # Add recommended models
        recommended = {
            "tinyllama": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            "phi-3-mini": "microsoft/Phi-3-mini-4k-instruct",
            "stablelm": "stabilityai/stablelm-3b-4e1t",
            "mistral-7b": "mistralai/Mistral-7B-Instruct-v0.2",
        }
        
        return jsonify({
            'success': True,
            'models': models,
            'recommended': recommended
        })
        
    except Exception as e:
        logger.error(f"Error listing models: {e}")
        return jsonify({
            'success': False,
            'message': str(e),
            'models': []
        }), 500


@api_bp.route('/llm/models/download', methods=['POST'])
def llm_model_download():
    """Download a model from HuggingFace"""
    try:
        if not llm_service:
            return jsonify({
                'success': False,
                'message': 'LLM service not initialized'
            }), 503
        
        data = request.get_json()
        if not data or 'model_id' not in data:
            return jsonify({
                'success': False,
                'message': 'model_id is required'
            }), 400
        
        model_id = data['model_id']
        revision = data.get('revision', 'main')
        
        result = llm_service.download_model(model_id=model_id, revision=revision)
        
        if result.get('success'):
            return jsonify(result)
        else:
            return jsonify(result), 500
            
    except Exception as e:
        logger.error(f"Error downloading model: {e}")
        return jsonify({
            'success': False,
            'message': str(e)
        }), 500


@api_bp.route('/llm/models/load', methods=['POST'])
def llm_model_load():
    """Load a model into memory"""
    try:
        if not llm_service:
            return jsonify({
                'success': False,
                'message': 'LLM service not initialized'
            }), 503
        
        data = request.get_json()
        if not data or 'model_id' not in data:
            return jsonify({
                'success': False,
                'message': 'model_id is required'
            }), 400
        
        model_id = data['model_id']
        
        result = llm_service.load_model(model_id=model_id)
        
        if result.get('success'):
            return jsonify(result)
        else:
            return jsonify(result), 500
            
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        return jsonify({
            'success': False,
            'message': str(e)
        }), 500


@api_bp.route('/llm/config', methods=['GET'])
def llm_get_config():
    """Get LLM configuration"""
    try:
        config_path = Path('data/llm_config.json')
        
        if config_path.exists():
            with open(config_path, 'r') as f:
                config = json.load(f)
        else:
            # Return default configuration
            config = {
                'provider': os.environ.get('AI_PROVIDER', 'openai'),
                'provider_priority': [os.environ.get('AI_PROVIDER', 'openai')],
                'fallback_mode': os.environ.get('AI_FALLBACK_MODE', 'hybrid'),
                'providers': {
                    'openai': {
                        'enabled': bool(os.environ.get('OPENAI_API_KEY')),
                        'model': os.environ.get('AI_CHAT_MODEL', 'gpt-3.5-turbo')
                    },
                    'huggingface_local': {
                        'enabled': llm_service is not None,
                        'model': os.environ.get('LLM_DEFAULT_MODEL', ''),
                        'service_url': f"http://{os.environ.get('LLM_SERVICE_HOST', '127.0.0.1')}:{os.environ.get('LLM_SERVICE_PORT', '8102')}"
                    }
                },
                'parameters': {
                    'temperature': float(os.environ.get('AI_TEMPERATURE', '0.7')),
                    'max_tokens': int(os.environ.get('AI_MAX_TOKENS', '512')),
                    'top_p': float(os.environ.get('AI_TOP_P', '0.9'))
                }
            }
        
        return jsonify({
            'success': True,
            'config': config
        })
        
    except Exception as e:
        logger.error(f"Error getting LLM config: {e}")
        return jsonify({
            'success': False,
            'message': str(e)
        }), 500


@api_bp.route('/llm/config', methods=['PUT'])
def llm_update_config():
    """Update LLM configuration"""
    try:
        data = request.get_json()
        if not data:
            return jsonify({
                'success': False,
                'message': 'No configuration provided'
            }), 400
        
        config_path = Path('data/llm_config.json')
        config_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save configuration
        with open(config_path, 'w') as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"LLM configuration updated")

        # Reload AI providers to apply new configuration if agent is available
        try:
            if ai_agent:
                ai_agent.reload_providers()
        except Exception as reload_err:
            logger.warning(f"Could not reload AI agent after config update: {reload_err}")
        
        return jsonify({
            'success': True,
            'message': 'Configuration updated successfully'
        })
        
    except Exception as e:
        logger.error(f"Error updating LLM config: {e}")
        return jsonify({
            'success': False,
            'message': str(e)
        }), 500


@api_bp.route('/llm/providers', methods=['GET'])
def llm_list_providers():
    """List available AI providers"""
    try:
        providers = []
        user_keys = {}
        try:
            uk_path = Path('data/llm_user_keys.json')
            if uk_path.exists():
                user_keys = json.load(open(uk_path, 'r'))
        except Exception:
            user_keys = {}

        def has_key(pid: str, env_var: str) -> bool:
            return bool(os.environ.get(env_var) or (user_keys.get(pid, {}) if isinstance(user_keys, dict) else {}).get("api_key"))
        
        # OpenAI
        if has_key('openai', 'OPENAI_API_KEY'):
            providers.append({
                'id': 'openai',
                'name': 'OpenAI',
                'status': 'available',
                'models': ['gpt-3.5-turbo', 'gpt-4', 'gpt-4-turbo']
            })
        
        # Azure OpenAI
        if os.environ.get('AZURE_OPENAI_KEY'):
            providers.append({
                'id': 'azure_openai',
                'name': 'Azure OpenAI',
                'status': 'available',
                'models': []
            })
        
        # Anthropic
        if has_key('anthropic', 'ANTHROPIC_API_KEY'):
            providers.append({
                'id': 'anthropic',
                'name': 'Anthropic',
                'status': 'available',
                'models': ['claude-3-opus', 'claude-3-sonnet', 'claude-3-haiku']
            })

        # Grok / xAI
        if has_key('grok', 'GROK_API_KEY'):
            providers.append({
                'id': 'grok',
                'name': 'Grok (xAI)',
                'status': 'available',
                'models': ['grok-2', 'grok-2-vision']
            })
        
        # Local HuggingFace
        if llm_service:
            status = llm_service.get_status()
            providers.append({
                'id': 'huggingface_local',
                'name': 'Local HuggingFace',
                'status': 'running' if status.get('running') else 'available',
                'models': [m['id'] for m in llm_service.list_models()]
            })
        
        return jsonify({
            'success': True,
            'providers': providers,
            'active': os.environ.get('AI_PROVIDER', 'openai')
        })
        
    except Exception as e:
        logger.error(f"Error listing providers: {e}")
        return jsonify({
            'success': False,
            'message': str(e)
        }), 500


# ============================================================
# Intent Learning System Endpoints
# ============================================================

@api_bp.route('/llm/patterns', methods=['GET'])
def llm_get_patterns():
    """Get learned intent patterns"""
    try:
        if not ai_agent:
            return jsonify({
                'success': False,
                'message': 'AI agent not initialized'
            }), 503
        
        stats = ai_agent.get_intent_learner_stats()
        return jsonify({
            'success': True,
            'stats': stats
        })
        
    except Exception as e:
        logger.error(f"Error getting patterns: {e}")
        return jsonify({
            'success': False,
            'message': str(e)
        }), 500


@api_bp.route('/llm/feedback', methods=['POST'])
def llm_provide_feedback():
    """Provide feedback on a tool execution (supports model-specific feedback)"""
    try:
        if not ai_agent:
            return jsonify({
                'success': False,
                'message': 'AI agent not initialized'
            }), 503
        
        data = request.get_json()
        if not data:
            return jsonify({
                'success': False,
                'message': 'No data provided'
            }), 400
        
        user_query = data.get('query')
        tool = data.get('tool')
        rating = float(data.get('rating', 1.0))
        model = data.get('model', 'unknown')  # Track which model the feedback is for
        
        if not user_query or not tool:
            return jsonify({
                'success': False,
                'message': 'query and tool are required'
            }), 400
        
        # Log model-specific feedback
        logger.info(f"Feedback received for {model} model: {tool} (rating: {rating})")
        
        ai_agent.provide_feedback(user_query, tool, rating)
        
        return jsonify({
            'success': True,
            'message': f'Feedback recorded for {model}'
        })
        
    except Exception as e:
        logger.error(f"Error providing feedback: {e}")
        return jsonify({
            'success': False,
            'message': str(e)
        }), 500


@api_bp.route('/llm/patterns/all', methods=['GET'])
def llm_get_all_patterns():
    """Get all learned patterns for management"""
    try:
        if not ai_agent:
            return jsonify({
                'success': False,
                'message': 'AI agent not initialized'
            }), 503
        
        patterns = ai_agent.intent_learner.get_all_patterns()
        
        return jsonify({
            'success': True,
            'patterns': patterns,
            'count': len(patterns)
        })
        
    except Exception as e:
        logger.error(f"Error getting all patterns: {e}")
        return jsonify({
            'success': False,
            'message': str(e)
        }), 500


@api_bp.route('/llm/patterns/<pattern_id>', methods=['DELETE'])
def llm_delete_pattern(pattern_id):
    """Delete a specific learned pattern"""
    try:
        if not ai_agent:
            return jsonify({
                'success': False,
                'message': 'AI agent not initialized'
            }), 503
        
        success = ai_agent.intent_learner.delete_pattern(pattern_id)
        
        if success:
            return jsonify({
                'success': True,
                'message': f'Pattern {pattern_id} deleted'
            })
        else:
            return jsonify({
                'success': False,
                'message': 'Pattern not found'
            }), 404
        
    except Exception as e:
        logger.error(f"Error deleting pattern: {e}")
        return jsonify({
            'success': False,
            'message': str(e)
        }), 500


@api_bp.route('/llm/patterns/<pattern_id>/score', methods=['PUT'])
def llm_update_pattern_score(pattern_id):
    """Update feedback score for a pattern"""
    try:
        if not ai_agent:
            return jsonify({
                'success': False,
                'message': 'AI agent not initialized'
            }), 503
        
        data = request.get_json()
        if not data or 'score' not in data:
            return jsonify({
                'success': False,
                'message': 'score is required'
            }), 400
        
        score = float(data['score'])
        success = ai_agent.intent_learner.update_pattern_score(pattern_id, score)
        
        if success:
            return jsonify({
                'success': True,
                'message': f'Pattern score updated to {score}'
            })
        else:
            return jsonify({
                'success': False,
                'message': 'Pattern not found'
            }), 404
        
    except Exception as e:
        logger.error(f"Error updating pattern score: {e}")
        return jsonify({
            'success': False,
            'message': str(e)
        }), 500


@api_bp.route('/llm/patterns/clear', methods=['POST'])
def llm_clear_all_patterns():
    """Clear all learned patterns"""
    try:
        if not ai_agent:
            return jsonify({
                'success': False,
                'message': 'AI agent not initialized'
            }), 503
        
        count = ai_agent.intent_learner.clear_all_patterns()
        
        return jsonify({
            'success': True,
            'message': f'Cleared {count} patterns',
            'count': count
        })
        
    except Exception as e:
        logger.error(f"Error clearing patterns: {e}")
        return jsonify({
            'success': False,
            'message': str(e)
        }), 500