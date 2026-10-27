from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional


def _overrides_path() -> Path:
    """
    Persist under the per-user data dir in frozen builds.
    In dev/source, this resolves to <repo>/data/...
    """
    from core.paths import get_data_dir

    return get_data_dir() / "desktop_object_overrides.json"


OVERRIDES_PATH = _overrides_path()


@dataclass
class ObjectOverride:
    """
    Per-camera per-track override settings.

    IMPORTANT: track_ids are NOT guaranteed stable across sessions; they may reset when
    Desktop restarts or when the tracker is reset/switched.
    """

    name: Optional[str] = None
    color: Optional[str] = None  # hex string like "#RRGGBB"
    hidden: bool = False

    def to_dict(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {"hidden": bool(self.hidden)}
        if self.name:
            out["name"] = str(self.name)
        if self.color:
            out["color"] = str(self.color)
        return out

    @staticmethod
    def from_dict(d: Optional[dict]) -> "ObjectOverride":
        dd = d or {}
        name = dd.get("name")
        color = dd.get("color")
        hidden = bool(dd.get("hidden", False))
        return ObjectOverride(
            name=str(name).strip() if isinstance(name, str) and name.strip() else None,
            color=str(color).strip() if isinstance(color, str) and color.strip() else None,
            hidden=hidden,
        )


def _load_all() -> Dict[str, Dict[str, dict]]:
    try:
        if not OVERRIDES_PATH.exists():
            return {}
        with OVERRIDES_PATH.open("r", encoding="utf-8") as f:
            raw = json.load(f)
        if not isinstance(raw, dict):
            return {}
        # schema: { camera_id: { track_id: {...} } }
        out: Dict[str, Dict[str, dict]] = {}
        for cam_id, per_cam in raw.items():
            if not isinstance(cam_id, str) or not isinstance(per_cam, dict):
                continue
            out[cam_id] = {str(k): (v if isinstance(v, dict) else {}) for k, v in per_cam.items()}
        return out
    except Exception:
        return {}


def _atomic_write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    tmp.replace(path)


def load_camera_overrides(camera_id: str) -> Dict[int, ObjectOverride]:
    all_data = _load_all()
    per_cam = all_data.get(str(camera_id), {}) if isinstance(all_data, dict) else {}
    out: Dict[int, ObjectOverride] = {}
    if isinstance(per_cam, dict):
        for tid_s, v in per_cam.items():
            try:
                tid = int(tid_s)
            except Exception:
                continue
            out[tid] = ObjectOverride.from_dict(v if isinstance(v, dict) else {})
    return out


def set_track_override(camera_id: str, track_id: int, override: ObjectOverride) -> None:
    camera_id = str(camera_id)
    track_id_s = str(int(track_id))
    all_data = _load_all()
    per_cam = all_data.get(camera_id)
    if not isinstance(per_cam, dict):
        per_cam = {}
        all_data[camera_id] = per_cam
    per_cam[track_id_s] = override.to_dict()
    _atomic_write_json(OVERRIDES_PATH, all_data)


def delete_track_override(camera_id: str, track_id: int) -> None:
    camera_id = str(camera_id)
    track_id_s = str(int(track_id))
    all_data = _load_all()
    per_cam = all_data.get(camera_id)
    if isinstance(per_cam, dict) and track_id_s in per_cam:
        del per_cam[track_id_s]
        if not per_cam:
            del all_data[camera_id]
        _atomic_write_json(OVERRIDES_PATH, all_data)


def clear_camera_overrides(camera_id: str) -> None:
    """Remove all saved overrides for a given camera."""
    camera_id = str(camera_id)
    all_data = _load_all()
    if camera_id in all_data:
        try:
            del all_data[camera_id]
        except Exception:
            pass
        _atomic_write_json(OVERRIDES_PATH, all_data)


