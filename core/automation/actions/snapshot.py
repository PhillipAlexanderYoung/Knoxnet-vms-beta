from __future__ import annotations

import logging
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple

try:
    import cv2  # type: ignore
    import numpy as np  # type: ignore
except Exception:  # pragma: no cover
    cv2 = None
    np = None

from core.capture_events import emit_new_capture
from core.paths import get_motion_watch_dir, get_project_root

try:
    from core.snapshots import draw_overlays
except Exception:  # pragma: no cover
    draw_overlays = None

logger = logging.getLogger(__name__)

_CAPTURE_FRAME_WAIT_SEC = 1.5
_CAPTURE_FRAME_POLL_SEC = 0.1


def _sanitize_zone_dirname(name: str) -> str:
    s = str(name or "").strip()
    if not s:
        return "_unzoned"
    s = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", s).strip(". ")
    return (s[:120] or "_unzoned")


def _decode_jpeg(frame_bytes: bytes):
    if cv2 is None or np is None:
        return None
    try:
        arr = np.frombuffer(frame_bytes, dtype=np.uint8)
        return cv2.imdecode(arr, cv2.IMREAD_COLOR)
    except Exception:
        return None


def _encode_jpeg(img, quality: int = 85) -> Optional[bytes]:
    if cv2 is None:
        return None
    try:
        ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
        if not ok:
            return None
        return buf.tobytes()
    except Exception:
        return None


def _to_pixel_shapes(shapes: Dict[str, Any], frame_w: int, frame_h: int):
    zones_px = []
    lines_px = []
    tags_px = []
    for z in (shapes.get("zones") or []):
        if not isinstance(z, dict) or z.get("enabled") is False:
            continue
        pts = z.get("points") or z.get("pts") or []
        if not isinstance(pts, list) or len(pts) < 3:
            continue
        poly = []
        for p in pts:
            if not isinstance(p, dict):
                continue
            try:
                poly.append({"x": float(p.get("x", 0.0)) * frame_w, "y": float(p.get("y", 0.0)) * frame_h})
            except Exception:
                continue
        if len(poly) >= 3:
            zones_px.append(poly)

    for ln in (shapes.get("lines") or []):
        if not isinstance(ln, dict) or ln.get("enabled") is False:
            continue
        p1 = ln.get("p1") or {}
        p2 = ln.get("p2") or {}
        try:
            lines_px.append(
                {
                    "p1": {"x": float(p1.get("x", 0.0)) * frame_w, "y": float(p1.get("y", 0.0)) * frame_h},
                    "p2": {"x": float(p2.get("x", 1.0)) * frame_w, "y": float(p2.get("y", 1.0)) * frame_h},
                }
            )
        except Exception:
            continue

    for t in (shapes.get("tags") or []):
        if not isinstance(t, dict) or t.get("enabled") is False:
            continue
        anchor = t.get("anchor") if isinstance(t.get("anchor"), dict) else {}
        try:
            x = float(t.get("x", anchor.get("x", 0.0)))
            y = float(t.get("y", anchor.get("y", 0.0)))
            tags_px.append({"x": x * frame_w, "y": y * frame_h})
        except Exception:
            continue

    return zones_px, lines_px, tags_px


def _capture_failure(
    *,
    camera_id: str,
    rule_id: Any,
    reason: str,
    detail: str = "",
) -> Dict[str, Any]:
    """Structured snapshot failure for automation_alert / terminal observability."""
    out: Dict[str, Any] = {
        "capture_failed": True,
        "reason": str(reason or "unknown"),
        "camera_id": str(camera_id or ""),
        "rule_id": rule_id,
    }
    if detail:
        out["detail"] = str(detail)
    return out


def _resolve_camera_stream_config(camera_id: str) -> Optional[Dict[str, Any]]:
    """Best-effort RTSP config for on-demand stream start before snapshot."""
    cam_id = str(camera_id or "").strip()
    if not cam_id:
        return None
    try:
        from app import resolve_camera_ref  # type: ignore

        cam = resolve_camera_ref(cam_id)
        if isinstance(cam, dict):
            rtsp = str(cam.get("rtsp_url") or cam.get("stream_url") or "").strip()
            if rtsp:
                return {
                    "rtsp_url": rtsp,
                    "webrtc_enabled": bool(cam.get("webrtc_enabled", False)),
                    "fps": 15,
                }
    except Exception:
        pass
    try:
        import json

        root = get_project_root()
        for rel in ("data/cameras.json", "cameras.json"):
            path = root / rel
            if not path.is_file():
                continue
            with path.open("r", encoding="utf-8") as fh:
                cameras = json.load(fh)
            if not isinstance(cameras, list):
                continue
            for cam in cameras:
                if not isinstance(cam, dict):
                    continue
                if str(cam.get("id") or "") != cam_id:
                    continue
                rtsp = str(cam.get("rtsp_url") or cam.get("stream_url") or "").strip()
                if rtsp:
                    return {
                        "rtsp_url": rtsp,
                        "webrtc_enabled": bool(cam.get("webrtc_enabled", False)),
                        "fps": 15,
                    }
    except Exception:
        pass
    return None


def acquire_snapshot_frame_bytes(stream_server: Any, camera_id: str) -> Tuple[Optional[bytes], Optional[str]]:
    """
    Return JPEG bytes for a snapshot, starting the stream on demand when needed.

    Returns (frame_bytes, error_reason). error_reason is None on success.
    """
    cam_id = str(camera_id or "").strip()
    if not cam_id:
        return None, "missing_camera_id"
    if stream_server is None or not hasattr(stream_server, "get_frame"):
        return None, "stream_server_unavailable"

    def _try_get() -> Optional[bytes]:
        try:
            frame_bytes = stream_server.get_frame(cam_id)
            return frame_bytes if frame_bytes else None
        except Exception as e:
            logger.warning("SnapshotAction get_frame failed for %s: %s", cam_id, e)
            return None

    frame_bytes = _try_get()
    if frame_bytes:
        return frame_bytes, None

    stream_cfg = _resolve_camera_stream_config(cam_id)
    if stream_cfg and hasattr(stream_server, "start_stream_sync"):
        try:
            stream_server.start_stream_sync(cam_id, stream_cfg)
        except Exception as e:
            logger.warning("SnapshotAction start_stream_sync failed for %s: %s", cam_id, e)

    deadline = time.time() + _CAPTURE_FRAME_WAIT_SEC
    while time.time() < deadline:
        frame_bytes = _try_get()
        if frame_bytes:
            return frame_bytes, None
        time.sleep(_CAPTURE_FRAME_POLL_SEC)

    active = getattr(stream_server, "active_streams", None)
    if isinstance(active, dict) and cam_id not in active:
        return None, "stream_not_active"
    return None, "no_frame_available"


class SnapshotAction:
    """
    Server-side snapshot capture for Event Rules.

    Config (action dict):
      type: "snapshot"
      include_overlays: true     (default true; alias overlay)
      save_dir: "captures/motion_watch"  (optional)
      resize_w: 0               (optional output width)
      quality: 85               (jpeg quality)
    """

    def __init__(
        self,
        *,
        db_manager: Any,
        stream_server: Any,
        event_index: Any = None,
        socketio: Any = None,
    ) -> None:
        self.db_manager = db_manager
        self.stream_server = stream_server
        self._event_index = event_index
        self.socketio = socketio

    def _index(self):
        if self._event_index is not None:
            return self._event_index
        try:
            from core.event_index_service import EventIndexService

            self._event_index = EventIndexService()
        except Exception as e:
            logger.debug("EventIndexService unavailable: %s", e)
            self._event_index = None
        return self._event_index

    def handler(self) -> Callable[..., Optional[Dict[str, Any]]]:
        def _handler(*, rule: Dict[str, Any], ctx: Any, details: Dict[str, Any], action: Dict[str, Any], event: Any) -> Optional[Dict[str, Any]]:
            return self.capture(rule=rule, ctx=ctx, details=details, action=action, event=event)

        return _handler

    def capture(
        self,
        *,
        rule: Dict[str, Any],
        ctx: Any,
        details: Dict[str, Any],
        action: Dict[str, Any],
        event: Any,
    ) -> Optional[Dict[str, Any]]:
        camera_id = str(getattr(ctx, "camera_id", None) or getattr(event, "camera_id", "") or "")
        rule_id = rule.get("id")
        if not camera_id:
            logger.warning("SnapshotAction skipped: missing camera_id rule=%s", rule_id)
            return _capture_failure(camera_id="", rule_id=rule_id, reason="missing_camera_id")

        frame_bytes, frame_err = acquire_snapshot_frame_bytes(self.stream_server, camera_id)
        if not frame_bytes:
            reason = frame_err or "no_frame_available"
            logger.warning(
                "SnapshotAction skipped: no frame for camera=%s rule=%s reason=%s",
                camera_id,
                rule_id,
                reason,
            )
            return _capture_failure(camera_id=camera_id, rule_id=rule_id, reason=reason)

        include_overlays = bool(action.get("include_overlays", action.get("overlay", True)))
        quality = int(action.get("quality", 85) or 85)
        jpg_bytes = frame_bytes

        payload = getattr(ctx, "payload", {}) if hasattr(ctx, "payload") else {}
        if not isinstance(payload, dict):
            payload = getattr(event, "payload", {}) or {}
        if not isinstance(payload, dict):
            payload = {}

        motion_box = payload.get("bbox") if isinstance(payload.get("bbox"), dict) else None
        tracks = []
        dets = []
        if isinstance(payload.get("bbox"), dict):
            tr = {
                "id": payload.get("track_id"),
                "track_id": payload.get("track_id"),
                "class": payload.get("class"),
                "class_name": payload.get("class"),
                "confidence": payload.get("confidence"),
                "bbox": payload.get("bbox"),
            }
            tracks = [tr]
            dets = [tr]

        if include_overlays and draw_overlays is not None and cv2 is not None:
            img = _decode_jpeg(frame_bytes)
            if img is not None:
                h, w = img.shape[:2]
                shapes = {}
                if self.db_manager and hasattr(self.db_manager, "get_camera_shapes"):
                    shapes = self.db_manager.get_camera_shapes(camera_id) or {}
                elif hasattr(self.stream_server, "get_camera_shapes"):
                    shapes = self.stream_server.get_camera_shapes(camera_id) or {}
                zones_px, lines_px, tags_px = _to_pixel_shapes(shapes or {}, w, h)
                try:
                    img2 = draw_overlays(img, zones_px, lines_px, tracks, dets)
                    enc = _encode_jpeg(img2, quality=quality)
                    if enc:
                        jpg_bytes = enc
                except Exception as e:
                    logger.debug("SnapshotAction overlay paint failed: %s", e)

        img_for_resize = _decode_jpeg(jpg_bytes)
        resize_w = int(action.get("resize_w", 0) or 0)
        if img_for_resize is not None and resize_w > 0 and img_for_resize.shape[1] > resize_w:
            scale = resize_w / float(img_for_resize.shape[1])
            new_h = max(1, int(img_for_resize.shape[0] * scale))
            img_for_resize = cv2.resize(img_for_resize, (resize_w, new_h))
            enc = _encode_jpeg(img_for_resize, quality=quality)
            if enc:
                jpg_bytes = enc

        captured_ts = int(time.time())
        save_dir_raw = str(action.get("save_dir") or "captures/motion_watch").strip()
        save_dir = Path(save_dir_raw)
        if not save_dir.is_absolute():
            save_dir = (get_project_root() / save_dir).resolve()
        shape_name = str(payload.get("shape_name") or details.get("shape_id") or "")
        save_dir = save_dir / _sanitize_zone_dirname(shape_name)
        save_dir.mkdir(parents=True, exist_ok=True)
        fname = save_dir / f"{camera_id}_watch_{captured_ts}.jpg"
        try:
            fname.write_bytes(jpg_bytes)
        except Exception as e:
            logger.warning("SnapshotAction failed to write %s: %s", fname, e)
            return _capture_failure(
                camera_id=camera_id,
                rule_id=rule_id,
                reason="write_failed",
                detail=str(e),
            )

        event_type = str(payload.get("event_type") or details.get("event_type") or "event_rule")
        dominant_color = payload.get("dominant_color") or payload.get("color")
        shape_name_value = str(payload.get("shape_name") or details.get("shape_name") or shape_name or "")
        thumb_b64 = None
        try:
            import base64

            thumb_b64 = base64.b64encode(jpg_bytes).decode("utf-8")
        except Exception:
            thumb_b64 = None

        fallback_result: Dict[str, Any] = {
            "file_path": str(fname),
            "camera_id": camera_id,
            "captured_ts": captured_ts,
            "shape_name": shape_name_value,
            "trigger_type": event_type,
            "ingested": False,
        }
        if thumb_b64:
            fallback_result["thumb_b64"] = thumb_b64
        ingest_payload = {
            "file_path": str(fname),
            "camera_id": camera_id,
            "enable_vision": False,
            "enable_detections": False,
            "captured_ts": captured_ts,
            "captured_at": datetime.fromtimestamp(captured_ts, tz=timezone.utc).isoformat(),
            "trigger": {
                "interaction_type": event_type,
                "trigger_type": event_type,
                "shape_id": payload.get("shape_id") or details.get("shape_id"),
                "shape_name": payload.get("shape_name") or shape_name,
                "source": "event_rules",
                "rule_id": rule.get("id"),
                "rule_name": rule.get("name"),
                "track_id": payload.get("track_id") or details.get("track_id"),
            },
            "motion_box": motion_box,
            "detection_classes": [str(payload.get("class"))] if payload.get("class") else [],
            "dominant_color": str(dominant_color).strip().lower() if isinstance(dominant_color, str) and str(dominant_color).strip() else None,
            "metadata": {
                "rule": {"id": rule.get("id"), "name": rule.get("name")},
                "details": details,
                "event_kind": getattr(event, "kind", None),
            },
        }

        idx = self._index()
        if idx is not None and hasattr(idx, "ingest"):
            try:
                result = idx.ingest(ingest_payload)
                result = dict(result or {})
                result.setdefault("camera_id", camera_id)
                logger.info(
                    "SnapshotAction indexed event %s for camera=%s rule=%s file=%s",
                    result.get("event_id"),
                    camera_id,
                    rule.get("id"),
                    fname,
                )
                emit_new_capture(self.socketio, result, camera_id=camera_id)
                return result
            except Exception as e:
                logger.warning(
                    "SnapshotAction ingest failed for camera=%s rule=%s file=%s: %s",
                    camera_id,
                    rule.get("id"),
                    fname,
                    e,
                )
        else:
            logger.warning(
                "SnapshotAction ingest unavailable for camera=%s rule=%s file=%s",
                camera_id,
                rule.get("id"),
                fname,
            )
        emit_new_capture(self.socketio, fallback_result, camera_id=camera_id)
        return fallback_result
