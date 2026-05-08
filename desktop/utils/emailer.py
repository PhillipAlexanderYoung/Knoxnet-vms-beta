import base64
import logging
import os
import threading
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

from core.email_client import EmailClient

logger = logging.getLogger(__name__)


def _as_bool(value: Optional[str], default: bool = False) -> bool:
    if value is None:
        return default
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


class EmailNotifier:
    """
    Thin helper around EmailClient for desktop/terminal use-cases.
    Keeps a shared client instance and throttles motion email bursts.
    """

    _client: Optional[EmailClient] = None
    _client_lock = threading.Lock()
    _last_motion_sent: Dict[str, float] = {}
    _motion_cooldown_sec = int(os.getenv("EMAIL_MOTION_COOLDOWN_SEC", "90") or 90)
    _motion_autosend = _as_bool(os.getenv("EMAIL_MOTION_AUTOSEND"), False)

    @classmethod
    def client(cls) -> Optional[EmailClient]:
        if cls._client:
            return cls._client
        with cls._client_lock:
            if cls._client:
                return cls._client
            cls._client = EmailClient.from_env()
            return cls._client

    @classmethod
    def send(
        cls,
        subject: str,
        body: str,
        *,
        to: Optional[Sequence[str]] = None,
        attachments: Optional[List[Dict[str, Any]]] = None,
        html_body: Optional[str] = None,
    ) -> Tuple[bool, Optional[str]]:
        client = cls.client()
        if not client:
            return False, "Email is not configured. Set EMAIL_NOTIFICATIONS_ENABLED and SMTP credentials."
        try:
            client.send(subject=subject, body=body, to=to, attachments=attachments, html_body=html_body)
            return True, None
        except Exception as e:
            logger.error("Email send failed: %s", e, exc_info=True)
            return False, str(e)

    @classmethod
    def maybe_send_motion(
        cls,
        camera_label: str,
        text: str,
        *,
        image_b64: Optional[str] = None,
        remaining_seconds: Optional[int] = None,
    ) -> Tuple[bool, Optional[str]]:
        """
        Optional auto-email when motion watch produces a capture. Controlled by
        EMAIL_MOTION_AUTOSEND and throttled by EMAIL_MOTION_COOLDOWN_SEC.
        """
        if not cls._motion_autosend:
            return False, "Auto email for motion is disabled"
        if not image_b64:
            return False, "No image to attach"

        now = time.time()
        last = cls._last_motion_sent.get(camera_label)
        if last and (now - last) < cls._motion_cooldown_sec:
            return False, "Cooldown active"

        attachments: List[Dict[str, Any]] = []
        try:
            attachments.append(
                {
                    "filename": f"{camera_label}-motion.jpg",
                    "mime_type": "image/jpeg",
                    "data": base64.b64decode(image_b64),
                }
            )
        except Exception as e:
            logger.warning("Failed to decode motion image for email: %s", e)

        body = (
            f"{text}\n"
            f"Camera: {camera_label}\n"
            f"Time: {time.strftime('%Y-%m-%d %H:%M:%S')}"
        )
        if remaining_seconds is not None:
            body += f"\nRemaining watch time: {remaining_seconds}s"

        ok, err = cls.send(
            subject=f"Motion alert - {camera_label}",
            body=body,
            attachments=attachments,
        )
        if ok:
            cls._last_motion_sent[camera_label] = now
        return ok, err










