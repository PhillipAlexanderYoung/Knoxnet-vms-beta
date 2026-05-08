"""
PTZ credential store

Per-camera secrets that the camera record itself does not hold (e.g. the
TP-Link cloud-account password required by `pytapo`).

Default behaviour is **session-only** (process memory). The user can
explicitly opt in to persisting the value to `data/ptz_credentials.json`
(file mode 0600) via the `persist=True` flag on `set()`. The desktop
prompt surfaces this trade-off with a visible warning.
"""

from __future__ import annotations

import json
import logging
import os
import threading
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


def _disk_path() -> Path:
    from core.paths import get_data_dir
    return get_data_dir() / "ptz_credentials.json"


_LOCK = threading.RLock()
_SESSION: Dict[str, Dict[str, Any]] = {}
_DISK_CACHE: Optional[Dict[str, Dict[str, Any]]] = None


# ---------------------------------------------------------------------------- #
# Disk persistence (opt-in)
# ---------------------------------------------------------------------------- #

def _load_disk() -> Dict[str, Dict[str, Any]]:
    global _DISK_CACHE
    if _DISK_CACHE is not None:
        return _DISK_CACHE
    path = _disk_path()
    if not path.exists():
        _DISK_CACHE = {}
        return _DISK_CACHE
    try:
        data = json.loads(path.read_text())
        if not isinstance(data, dict):
            data = {}
    except Exception as err:
        logger.warning("ptz_credentials: failed to read %s (%s); ignoring", path, err)
        data = {}
    _DISK_CACHE = data
    return _DISK_CACHE


def _save_disk(data: Dict[str, Dict[str, Any]]) -> None:
    global _DISK_CACHE
    path = _disk_path()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, indent=2))
        try:
            os.chmod(tmp, 0o600)
        except Exception:
            pass
        tmp.replace(path)
        try:
            os.chmod(path, 0o600)
        except Exception:
            pass
        _DISK_CACHE = data
    except Exception as err:
        logger.error("ptz_credentials: failed to write %s (%s)", path, err)


# ---------------------------------------------------------------------------- #
# Public API
# ---------------------------------------------------------------------------- #

def get(camera_id: str) -> Dict[str, Any]:
    """Return merged disk + session credentials for a camera (session wins)."""
    if not camera_id:
        return {}
    with _LOCK:
        merged: Dict[str, Any] = {}
        disk = _load_disk()
        if isinstance(disk.get(camera_id), dict):
            merged.update(disk[camera_id])
        if isinstance(_SESSION.get(camera_id), dict):
            merged.update(_SESSION[camera_id])
        return merged


def set(camera_id: str, key: str, value: Any, persist: bool = False) -> None:
    """
    Store a credential for `camera_id`.

    Always writes to the in-memory session store. When `persist=True`,
    additionally writes to `data/ptz_credentials.json` (chmod 0600).
    Set `value=None` (and persist=True) to remove a key from disk.
    """
    if not camera_id or not key:
        return
    with _LOCK:
        bucket = _SESSION.setdefault(camera_id, {})
        if value is None:
            bucket.pop(key, None)
        else:
            bucket[key] = value

        if persist:
            disk = _load_disk()
            cam_bucket = dict(disk.get(camera_id) or {})
            if value is None:
                cam_bucket.pop(key, None)
            else:
                cam_bucket[key] = value
            if cam_bucket:
                disk[camera_id] = cam_bucket
            else:
                disk.pop(camera_id, None)
            _save_disk(disk)


def clear(camera_id: str) -> None:
    """Forget a camera's credentials in both session and disk stores."""
    if not camera_id:
        return
    with _LOCK:
        _SESSION.pop(camera_id, None)
        disk = _load_disk()
        if camera_id in disk:
            disk.pop(camera_id, None)
            _save_disk(disk)


def has_persisted(camera_id: str, key: Optional[str] = None) -> bool:
    """True if the camera (and optionally a specific key) is on disk."""
    if not camera_id:
        return False
    with _LOCK:
        disk = _load_disk()
        bucket = disk.get(camera_id) or {}
        if key is None:
            return bool(bucket)
        return key in bucket
