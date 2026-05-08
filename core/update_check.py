from __future__ import annotations

import os
import platform
import re
from dataclasses import dataclass
from typing import Any, Tuple

import requests

DEFAULT_CLOUD_API_BASE = "https://api.knoxnetvms.com"
DEFAULT_CHANNEL = "beta"
GITHUB_REPO_URL = "https://github.com/PhillipAlexanderYoung/Knoxnet-VMS"


@dataclass(frozen=True)
class UpdateInfo:
    current_version: str
    latest_version: str
    channel: str
    url: str
    repo_url: str = GITHUB_REPO_URL


def os_param() -> str:
    system = platform.system().lower()
    if system.startswith("win"):
        return "win"
    if system == "darwin":
        return "mac"
    return "linux"


def update_url(channel: str = DEFAULT_CHANNEL) -> str:
    override = (os.environ.get("KNOXNET_UPDATE_MANIFEST_URL") or "").strip()
    if override:
        return override.format(channel=channel, os=os_param())
    cloud_base = (os.environ.get("KNOXNET_CLOUD_API_BASE") or DEFAULT_CLOUD_API_BASE).strip()
    return cloud_base.rstrip("/") + f"/v1/updates/latest?channel={channel}&os={os_param()}"


def version_sort_key(version: str) -> Tuple[int, int, int, int, Tuple[Tuple[int, object], ...]]:
    text = str(version or "").strip()
    match = re.match(r"^v?(\d+)(?:\.(\d+))?(?:\.(\d+))?(?:[-+]([0-9A-Za-z.\-]+))?$", text)
    if not match:
        return (0, 0, 0, 0, ((1, text.lower()),))
    major = int(match.group(1) or 0)
    minor = int(match.group(2) or 0)
    patch = int(match.group(3) or 0)
    prerelease = (match.group(4) or "").strip()
    if not prerelease:
        return (major, minor, patch, 1, ())
    tokens: list[Tuple[int, object]] = []
    for token in re.split(r"[.\-]", prerelease):
        t = token.strip().lower()
        if not t:
            continue
        tokens.append((0, int(t)) if t.isdigit() else (1, t))
    return (major, minor, patch, 0, tuple(tokens))


def is_newer(latest: str, current: str) -> bool:
    return version_sort_key(latest) > version_sort_key(current)


def _extract_version(payload: Any) -> tuple[str, str]:
    if not isinstance(payload, dict):
        return "", DEFAULT_CHANNEL
    data = payload.get("data") if isinstance(payload.get("data"), dict) else payload
    latest = str(data.get("latest_version") or data.get("version") or "").strip()
    channel = str(data.get("channel") or payload.get("channel") or DEFAULT_CHANNEL).strip() or DEFAULT_CHANNEL
    return latest, channel


def check_for_update(current_version: str, *, timeout_s: float = 5.0) -> UpdateInfo | None:
    channel = (os.environ.get("KNOXNET_UPDATE_CHANNEL") or DEFAULT_CHANNEL).strip().lower() or DEFAULT_CHANNEL
    url = update_url(channel)
    response = requests.get(url, timeout=timeout_s)
    response.raise_for_status()
    latest, response_channel = _extract_version(response.json())
    if latest and is_newer(latest, current_version):
        return UpdateInfo(
            current_version=current_version,
            latest_version=latest,
            channel=response_channel,
            url=url,
        )
    return None
