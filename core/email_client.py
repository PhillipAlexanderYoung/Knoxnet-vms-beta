import logging
import os
import smtplib
import ssl
from dataclasses import dataclass, field
from email.message import EmailMessage
from typing import Any, Dict, List, Optional, Sequence

try:
    from dotenv import load_dotenv
except Exception:  # pragma: no cover - optional dependency
    load_dotenv = None

logger = logging.getLogger(__name__)


def _as_bool(value: Optional[str], default: bool = False) -> bool:
    if value is None:
        return default
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _split_csv(value: Optional[str]) -> List[str]:
    if not value:
        return []
    return [part.strip() for part in value.split(",") if part.strip()]


@dataclass
class EmailSettings:
    host: str
    port: int
    username: str
    password: str
    sender: str
    sender_name: Optional[str] = None
    use_tls: bool = True
    use_ssl: bool = False
    recipients: List[str] = field(default_factory=list)
    reply_to: Optional[str] = None


class EmailClient:
    """
    Minimal, provider-agnostic SMTP client with TLS/SSL support and attachments.
    """

    def __init__(self, settings: EmailSettings):
        self.settings = settings

    # ------------------------------------------------------------------ #
    # Factory helpers
    # ------------------------------------------------------------------ #
    @classmethod
    def from_env(cls) -> Optional["EmailClient"]:
        """
        Build an EmailClient from .env / environment variables.
        Required:
          EMAIL_NOTIFICATIONS_ENABLED=true
          EMAIL_HOST, EMAIL_PORT, EMAIL_USERNAME, EMAIL_PASSWORD, EMAIL_FROM
        Optional:
          EMAIL_USE_TLS (default true), EMAIL_USE_SSL (default false)
          EMAIL_DEFAULT_TO (comma-separated), EMAIL_SENDER_NAME, EMAIL_REPLY_TO
        """
        if load_dotenv:
            try:
                load_dotenv()
            except Exception:
                # best-effort; don't fail startup because of dotenv issues
                pass

        if not _as_bool(os.getenv("EMAIL_NOTIFICATIONS_ENABLED") or os.getenv("EMAIL_ENABLED")):
            logger.info("Email notifications are disabled via environment.")
            return None

        host = os.getenv("EMAIL_HOST") or os.getenv("SMTP_HOST")
        username = os.getenv("EMAIL_USERNAME") or os.getenv("SMTP_USERNAME")
        password = os.getenv("EMAIL_PASSWORD") or os.getenv("SMTP_PASSWORD")
        sender = os.getenv("EMAIL_FROM") or username
        sender_name = os.getenv("EMAIL_SENDER_NAME")
        reply_to = os.getenv("EMAIL_REPLY_TO")
        use_ssl = _as_bool(os.getenv("EMAIL_USE_SSL"), False)
        use_tls = _as_bool(os.getenv("EMAIL_USE_TLS"), not use_ssl)
        port = int(os.getenv("EMAIL_PORT") or os.getenv("SMTP_PORT") or (465 if use_ssl else 587))
        recipients = _split_csv(os.getenv("EMAIL_DEFAULT_TO") or os.getenv("EMAIL_RECIPIENTS"))

        if not host or not port or not username or not password or not sender:
            logger.warning("Email configuration is incomplete; skipping email setup.")
            return None

        settings = EmailSettings(
            host=host,
            port=port,
            username=username,
            password=password,
            sender=sender,
            sender_name=sender_name,
            use_tls=use_tls,
            use_ssl=use_ssl,
            recipients=recipients,
            reply_to=reply_to,
        )
        return cls(settings)

    @classmethod
    def from_dict(cls, config: Dict[str, Any]) -> Optional["EmailClient"]:
        """
        Backward-compatible factory for existing alert configuration schema.
        Expected keys: smtp_server, smtp_port, username, password, recipients, sender_name(optional)
        """
        if not config:
            return None

        host = config.get("smtp_server") or config.get("host")
        port = int(config.get("smtp_port") or config.get("port") or 587)
        username = config.get("username") or config.get("user")
        password = config.get("password") or config.get("pass")
        sender = config.get("from") or username
        sender_name = config.get("sender_name")
        use_ssl = _as_bool(str(config.get("use_ssl")) if "use_ssl" in config else None, False)
        use_tls = _as_bool(str(config.get("use_tls")) if "use_tls" in config else None, not use_ssl)
        recipients_raw = config.get("recipients") or config.get("to") or []
        if isinstance(recipients_raw, str):
            recipients = _split_csv(recipients_raw)
        else:
            recipients = [r for r in recipients_raw if isinstance(r, str)]

        if not host or not username or not password or not sender:
            logger.warning("Email config missing required fields; host/user/pass/sender are required.")
            return None

        settings = EmailSettings(
            host=host,
            port=port,
            username=username,
            password=password,
            sender=sender,
            sender_name=sender_name,
            use_tls=use_tls,
            use_ssl=use_ssl,
            recipients=recipients,
            reply_to=config.get("reply_to"),
        )
        return cls(settings)

    # ------------------------------------------------------------------ #
    # Send
    # ------------------------------------------------------------------ #
    def send(
        self,
        subject: str,
        body: str,
        *,
        to: Optional[Sequence[str]] = None,
        html_body: Optional[str] = None,
        attachments: Optional[Sequence[Dict[str, Any]]] = None,
        headers: Optional[Dict[str, str]] = None,
    ) -> None:
        recipients = [r for r in (to or self.settings.recipients) if r]
        if not recipients:
            raise ValueError("No email recipients provided.")

        msg = EmailMessage()
        from_addr = self.settings.sender
        if self.settings.sender_name:
            msg["From"] = f"{self.settings.sender_name} <{from_addr}>"
        else:
            msg["From"] = from_addr
        msg["To"] = ", ".join(recipients)
        msg["Subject"] = subject
        if self.settings.reply_to:
            msg["Reply-To"] = self.settings.reply_to
        if headers:
            for key, value in headers.items():
                msg[key] = value

        msg.set_content(body or "")
        if html_body:
            msg.add_alternative(html_body, subtype="html")

        for att in attachments or []:
            if att is None:
                continue
            data = att.get("data")
            if data is None:
                continue
            filename = att.get("filename") or "attachment"
            mime_type = att.get("mime_type") or "application/octet-stream"
            maintype, _, subtype = mime_type.partition("/")
            if not subtype:
                subtype = "octet-stream"
            msg.add_attachment(data, maintype=maintype, subtype=subtype, filename=filename)

        self._deliver(msg, recipients)

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #
    def _deliver(self, msg: EmailMessage, recipients: Sequence[str]) -> None:
        context = ssl.create_default_context()
        if self.settings.use_ssl:
            with smtplib.SMTP_SSL(self.settings.host, self.settings.port, context=context) as server:
                self._login_and_send(server, msg, recipients)
        else:
            with smtplib.SMTP(self.settings.host, self.settings.port) as server:
                server.ehlo()
                if self.settings.use_tls:
                    server.starttls(context=context)
                self._login_and_send(server, msg, recipients)

    def _login_and_send(self, server: smtplib.SMTP, msg: EmailMessage, recipients: Sequence[str]) -> None:
        if self.settings.username and self.settings.password:
            server.login(self.settings.username, self.settings.password)
        server.send_message(msg, to_addrs=list(recipients))
        logger.info("Email sent to %s", ", ".join(recipients))










