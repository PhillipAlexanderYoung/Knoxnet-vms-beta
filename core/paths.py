from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Optional


def _frozen_bundle_dir() -> Path | None:
    """Return the PyInstaller extraction root when running as a frozen exe."""
    if getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS"):
        return Path(sys._MEIPASS)
    return None


def get_project_root() -> Path:
    """
    Return the repo/project root directory as an absolute Path.

    In a frozen PyInstaller build the bundled data lives under sys._MEIPASS.
    Primary strategy: derive from this file's location (core/paths.py -> repo root).
    Fallback: current working directory.
    """
    bundle = _frozen_bundle_dir()
    if bundle is not None:
        return bundle

    try:
        here = Path(__file__).resolve()
        # .../<repo>/core/paths.py -> parents[1] is <repo>
        root = here.parents[1]
        if root.exists():
            return root
    except Exception:
        pass
    return Path.cwd().resolve()


def get_models_dir() -> Path:
    """Return the directory containing bundled model files (MobileNetSSD, etc.)."""
    bundle = _frozen_bundle_dir()
    if bundle is not None:
        return bundle / "models"
    return get_project_root() / "models"


def get_data_dir() -> Path:
    """
    Return the data directory as an absolute Path.

    Allows override via KNOXNET_DATA_DIR.
    """
    env = (os.environ.get("KNOXNET_DATA_DIR") or "").strip()
    if env:
        try:
            p = Path(env).expanduser()
            return p.resolve() if p.is_absolute() else (get_project_root() / p).resolve()
        except Exception:
            # fall back below
            pass
    return (get_project_root() / "data").resolve()


def get_motion_watch_dir() -> Path:
    """
    Default Motion Watch capture root (absolute).

    Allows override via KNOXNET_MOTION_WATCH_DIR.
    """
    env = (os.environ.get("KNOXNET_MOTION_WATCH_DIR") or "").strip()
    if env:
        try:
            p = Path(env).expanduser()
            return p.resolve() if p.is_absolute() else (get_project_root() / p).resolve()
        except Exception:
            pass
    return (get_project_root() / "captures" / "motion_watch").resolve()


def get_motion_watch_zone_dir(zone_name: str) -> Path:
    """
    Return the capture directory for a specific zone/shape, creating it if needed.
    Names are sanitized for filesystem safety; empty names map to ``_unzoned``.
    """
    import re
    name = str(zone_name or "").strip()
    if not name:
        name = "_unzoned"
    name = re.sub(r'[<>:"/\\|?*\x00-\x1f]', '_', name).strip('. ')
    name = name[:120] or "_unzoned"
    d = get_motion_watch_dir() / name
    d.mkdir(parents=True, exist_ok=True)
    return d


def get_recordings_dir() -> Path:
    """
    Return the continuous-recording storage directory (absolute).

    Allows override via KNOXNET_RECORDINGS_DIR.  Falls back to
    ``<project_root>/recordings`` which matches the default MediaMTX recordPath.
    """
    env = (os.environ.get("KNOXNET_RECORDINGS_DIR") or "").strip()
    if env:
        try:
            p = Path(env).expanduser()
            return p.resolve() if p.is_absolute() else (get_project_root() / p).resolve()
        except Exception:
            pass
    return (get_project_root() / "recordings").resolve()


def resolve_under_root(p: Path, *, root: Optional[Path] = None) -> Path:
    """
    Resolve `p` as absolute. If relative, treat it as relative to `root` (default: project root).
    """
    root = root or get_project_root()
    try:
        if p.is_absolute():
            return p.resolve()
        return (root / p).resolve()
    except Exception:
        # best-effort
        return p if p.is_absolute() else (root / p)

