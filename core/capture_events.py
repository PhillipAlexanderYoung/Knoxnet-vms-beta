"""Shared helpers for capture ingest notifications (Socket.IO + terminal)."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


def file_uri_local(path: str) -> str:
    """Return a ``file://`` URI for a local filesystem path."""
    if not path:
        return ""
    p = Path(path)
    try:
        return p.resolve().as_uri()
    except Exception:
        return f"file:///{p.as_posix()}"


def build_new_capture_payload(
    result: Dict[str, Any],
    *,
    camera_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Build the ``new_capture`` Socket.IO payload from an ingest result."""
    file_p = str(result.get("file_path") or "")
    thumb_p = str(result.get("thumb_path") or "")
    payload = {
        "event_id": str(result.get("event_id") or ""),
        "camera_id": str(camera_id or result.get("camera_id") or ""),
        "captured_ts": int(result.get("captured_ts") or 0),
        "camera_name": str(result.get("camera_name") or ""),
        "caption": str(result.get("caption") or ""),
        "file_uri": file_uri_local(file_p),
        "thumb_uri": file_uri_local(thumb_p) if thumb_p else file_uri_local(file_p),
        "shape_name": str(result.get("shape_name") or ""),
        "trigger_type": str(result.get("trigger_type") or ""),
        "media_type": str(result.get("media_type") or "image"),
        "tags": [str(t) for t in (result.get("tags") or [])],
        "detection_classes": [str(c) for c in (result.get("detection_classes") or [])],
        "dominant_color": str(result.get("dominant_color") or ""),
        "file_path": file_p,
    }
    thumb_b64 = str(result.get("thumb_b64") or "").strip()
    if thumb_b64:
        payload["thumb_b64"] = thumb_b64
    return payload


def capture_failure_reason_label(reason: str) -> str:
    """User-facing label for a snapshot failure reason code."""
    key = str(reason or "").strip().lower()
    labels = {
        "missing_camera_id": "camera id missing",
        "stream_server_unavailable": "stream server unavailable",
        "stream_not_active": "camera stream not active",
        "no_frame_available": "no video frame available",
        "write_failed": "failed to save capture file",
        "handler_exception": "snapshot handler error",
        "snapshot_not_executed": "snapshot action did not run",
    }
    return labels.get(key, key.replace("_", " ") or "unknown error")


def build_terminal_capture_failure_payload(
    *,
    error: Dict[str, Any],
    camera_id: str,
    camera_label: Optional[str] = None,
    rule_name: Optional[str] = None,
) -> Dict[str, Any]:
    """Keyword args for ``TerminalWidget.broadcast_motion_watch`` on capture failure."""
    reason = capture_failure_reason_label(str(error.get("reason") or "unknown"))
    detail = str(error.get("detail") or "").strip()
    rule_part = f" ({rule_name})" if rule_name else ""
    text = f"Screenshot failed{rule_part}: {reason}"
    if detail:
        text = f"{text} — {detail}"
    return {
        "camera_id": camera_id,
        "text": text,
        "image_b64": None,
        "link": None,
        "link_label": None,
        "camera_label": camera_label or camera_id,
        "kind": "warning",
    }


def build_motion_watch_terminal_payload(
    *,
    result: Dict[str, Any],
    camera_id: str,
    camera_label: Optional[str] = None,
    thumb_b64: Optional[str] = None,
) -> Dict[str, Any]:
    """Keyword args for ``TerminalWidget.broadcast_motion_watch``."""
    file_path = str(result.get("file_path") or "")
    file_url = file_uri_local(file_path)
    shape_name = str(result.get("shape_name") or "").strip()
    trigger_type = str(result.get("trigger_type") or "").strip()
    detail = shape_name or trigger_type or "event rule"
    return {
        "camera_id": camera_id,
        "text": f"Captured {detail} — {file_path}" if file_path else f"Captured {detail}",
        "image_b64": thumb_b64,
        "link": file_url or None,
        "link_label": "Open full image" if file_url else None,
        "camera_label": camera_label or result.get("camera_name") or camera_id,
    }


def emit_new_capture(
    socketio: Any,
    result: Dict[str, Any],
    *,
    camera_id: Optional[str] = None,
    room: Optional[str] = None,
) -> None:
    """Emit ``new_capture`` on ``/realtime`` for desktop terminal and live report UIs."""
    if socketio is None:
        return
    payload = build_new_capture_payload(result, camera_id=camera_id)
    cam = str(camera_id or payload.get("camera_id") or "").strip()
    try:
        if room:
            socketio.emit("new_capture", payload, namespace="/realtime", room=room)
        elif cam:
            socketio.emit("new_capture", payload, namespace="/realtime", room=f"camera:{cam}")
            socketio.emit("new_capture", payload, namespace="/realtime")
        else:
            socketio.emit("new_capture", payload, namespace="/realtime")
    except TypeError:
        try:
            socketio.emit("new_capture", payload, namespace="/realtime")
        except Exception as e:
            logger.debug("emit_new_capture failed: %s", e)
    except Exception as e:
        logger.debug("emit_new_capture failed: %s", e)
