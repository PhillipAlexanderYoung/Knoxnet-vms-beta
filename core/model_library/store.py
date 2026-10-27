from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional


def _state_path() -> Path:
    from core.paths import get_data_dir

    return get_data_dir() / "model_library.json"


@dataclass(frozen=True)
class InstalledArtifact:
    key: str
    source: str  # e.g. "huggingface"
    repo_id: str
    filename: str
    local_path: str
    installed_at: float
    revision: Optional[str] = None
    files: Optional[list[str]] = None


def load_model_library_state() -> dict[str, Any]:
    p = _state_path()
    if not p.exists():
        return {"installed": {}, "defaults": {}}
    try:
        data = json.loads(p.read_text())
        if not isinstance(data, dict):
            return {"installed": {}, "defaults": {}}
        data.setdefault("installed", {})
        data.setdefault("defaults", {})
        return data
    except Exception:
        return {"installed": {}, "defaults": {}}


def save_model_library_state(state: dict[str, Any]) -> None:
    p = _state_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(state, indent=2))


def record_hf_install(
    *,
    key: str,
    repo_id: str,
    filename: str,
    local_path: Path,
    revision: Optional[str] = None,
) -> None:
    state = load_model_library_state()
    installed = state.get("installed")
    if not isinstance(installed, dict):
        installed = {}
        state["installed"] = installed
    installed[key] = {
        "source": "huggingface",
        "repo_id": repo_id,
        "filename": filename,
        "revision": revision,
        "local_path": str(local_path),
        "installed_at": time.time(),
    }
    save_model_library_state(state)


def record_hf_snapshot_install(
    *,
    key: str,
    repo_id: str,
    local_path: Path,
    revision: Optional[str] = None,
    files: Optional[list[str]] = None,
) -> None:
    state = load_model_library_state()
    installed = state.get("installed")
    if not isinstance(installed, dict):
        installed = {}
        state["installed"] = installed
    installed[key] = {
        "source": "huggingface_snapshot",
        "repo_id": repo_id,
        "revision": revision,
        "local_path": str(local_path),
        "installed_at": time.time(),
        "files": files or [],
    }
    save_model_library_state(state)


def get_installed_record(key: str) -> Optional[dict[str, Any]]:
    state = load_model_library_state()
    installed = state.get("installed")
    if not isinstance(installed, dict):
        return None
    record = installed.get(key)
    return record if isinstance(record, dict) else None

def record_local_install(*, key: str, local_path: Path) -> None:
    state = load_model_library_state()
    installed = state.get("installed")
    if not isinstance(installed, dict):
        installed = {}
        state["installed"] = installed
    installed[key] = {
        "source": "local",
        "local_path": str(local_path),
        "installed_at": time.time(),
    }
    save_model_library_state(state)


