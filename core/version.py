from __future__ import annotations

import os
from pathlib import Path

DEFAULT_VERSION = "1.0.0-beta1"


def get_version() -> str:
    """Return the public beta version from VERSION or KNOXNET_VERSION."""
    override = (os.environ.get("KNOXNET_VERSION") or "").strip()
    if override:
        return override

    version_file = Path(__file__).resolve().parents[1] / "VERSION"
    try:
        version = version_file.read_text(encoding="utf-8").strip()
        return version or DEFAULT_VERSION
    except Exception:
        return DEFAULT_VERSION
