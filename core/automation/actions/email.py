from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional

import base64

try:
    import cv2  # type: ignore
    import numpy as np  # type: ignore
except Exception:  # pragma: no cover
    cv2 = None
    np = None

from core.email_client import EmailClient

try:
    from core.snapshots import draw_overlays
except Exception:  # pragma: no cover
    draw_overlays = None

logger = logging.getLogger(__name__)


def _safe_format(template: str, values: Dict[str, Any]) -> str:
    try:
        return str(template).format_map({k: ("" if v is None else v) for k, v in values.items()})
    except Exception:
        return str(template)


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
        pts = z.get("points") or []
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

    for l in (shapes.get("lines") or []):
        if not isinstance(l, dict) or l.get("enabled") is False:
            continue
        p1 = l.get("p1") or {}
        p2 = l.get("p2") or {}
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
        try:
            tags_px.append({"x": float(t.get("x", 0.0)) * frame_w, "y": float(t.get("y", 0.0)) * frame_h})
        except Exception:
            continue

    return zones_px, lines_px, tags_px


class EmailAction:
    """
    Email action implementation using existing SMTP client.

    Config (action dict):
      type: "email"
      to: ["a@b.com"]            (optional; falls back to EMAIL_DEFAULT_TO)
      subject: "..."             (optional)
      body: "..."                (optional)
      include_snapshot: true     (optional, default false)
      overlay: true              (optional, default true) overlays zones/lines + detections/tracks if possible
      attach_as: "jpg"           (optional, default "jpg")
    """

    def __init__(self, *, db_manager: Any, stream_server: Any) -> None:
        self.db_manager = db_manager
        self.stream_server = stream_server
        self._client: Optional[EmailClient] = None

    def _client_from_env(self) -> Optional[EmailClient]:
        if self._client is not None:
            return self._client
        try:
            self._client = EmailClient.from_env()
        except Exception as e:
            logger.warning("Email client init failed: %s", e)
            self._client = None
        return self._client

    def handler(self) -> Callable[..., None]:
        def _handler(*, rule: Dict[str, Any], ctx: Any, details: Dict[str, Any], action: Dict[str, Any], event: Any) -> None:
            self.send(rule=rule, ctx=ctx, details=details, action=action, event=event)

        return _handler

    def send(self, *, rule: Dict[str, Any], ctx: Any, details: Dict[str, Any], action: Dict[str, Any], event: Any) -> None:
        client = self._client_from_env()
        if not client:
            logger.warning("EmailAction skipped: email not configured via environment.")
            return

        camera_id = getattr(ctx, "camera_id", None) or getattr(event, "camera_id", None) or ""
        event_kind = getattr(ctx, "kind", None) or getattr(event, "kind", None) or ""
        rule_name = str(rule.get("name") or rule.get("id") or "Rule")
        ts = datetime.now().isoformat()

        fmt_vals = {
            "camera_id": camera_id,
            "event_kind": event_kind,
            "rule_id": rule.get("id"),
            "rule_name": rule_name,
            "timestamp": ts,
        }

        subject = action.get("subject") or action.get("subject_template") or "[Knoxnet VMS Beta] {rule_name} ({camera_id})"
        body = action.get("body") or action.get("body_template") or (
            "Automation rule triggered.\n\n"
            "Rule: {rule_name}\n"
            "Camera: {camera_id}\n"
            "Event: {event_kind}\n"
            "Time: {timestamp}\n"
        )
        subject_s = _safe_format(str(subject), fmt_vals)
        body_s = _safe_format(str(body), fmt_vals)

        to = action.get("to")
        recipients: Optional[List[str]] = None
        if isinstance(to, str) and to.strip():
            recipients = [p.strip() for p in to.split(",") if p.strip()]
        elif isinstance(to, list):
            recipients = [str(p).strip() for p in to if str(p).strip()]

        attachments = []
        include_snapshot = bool(action.get("include_snapshot", False))
        overlay = bool(action.get("overlay", True))

        if include_snapshot and self.stream_server and hasattr(self.stream_server, "get_frame"):
            try:
                frame_bytes = self.stream_server.get_frame(str(camera_id))
                if frame_bytes:
                    att_bytes = frame_bytes

                    if overlay and draw_overlays is not None and cv2 is not None and np is not None:
                        img = _decode_jpeg(frame_bytes)
                        if img is not None:
                            h, w = img.shape[:2]
                            shapes = None
                            if self.db_manager and hasattr(self.db_manager, "get_camera_shapes"):
                                shapes = self.db_manager.get_camera_shapes(str(camera_id)) or {}
                            zones_px, lines_px, _tags_px = _to_pixel_shapes(shapes or {}, w, h)

                            # Normalize detections/tracks into snapshot renderer schema
                            dets = []
                            for d in getattr(ctx, "detections", []) or []:
                                if not isinstance(d, dict):
                                    continue
                                dd = dict(d)
                                dd["class_name"] = dd.get("class_name") or dd.get("class") or "object"
                                dets.append(dd)
                            trks = []
                            for t in getattr(ctx, "tracks", []) or []:
                                if isinstance(t, dict):
                                    trks.append(t)

                            try:
                                img2 = draw_overlays(img, zones_px, lines_px, trks, dets)
                                enc = _encode_jpeg(img2, quality=85)
                                if enc:
                                    att_bytes = enc
                            except Exception:
                                pass

                    attachments.append(
                        {
                            "filename": f"knoxnet_{camera_id}_{event_kind}_{rule.get('id')}.jpg",
                            "mime_type": "image/jpeg",
                            "data": att_bytes,
                        }
                    )
            except Exception as e:
                logger.debug("Snapshot capture failed for email action: %s", e)

        client.send(subject=subject_s, body=body_s, to=recipients, attachments=attachments)



