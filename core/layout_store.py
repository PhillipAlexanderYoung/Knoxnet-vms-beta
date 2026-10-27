from __future__ import annotations

import json
import os
import threading
import uuid
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .layout_models import CameraProfile, CameraProfileAssignmentValue, LayoutDefinition, WidgetDefinition


def _utc_now_iso() -> str:
    return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def _atomic_write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    os.replace(tmp, path)


class LayoutsAndProfilesStore:
    """
    Local-file JSON source of truth for:
      - layouts
      - camera profiles
      - camera->profile assignments
    """

    def __init__(
        self,
        layouts_path: str = "data/layouts.json",
        profiles_path: str = "data/camera_profiles.json",
        assignments_path: str = "data/camera_profile_assignments.json",
        legacy_desktop_layouts_path: str = "data/desktop_layouts.json",
    ):
        self.layouts_path = Path(layouts_path)
        self.profiles_path = Path(profiles_path)
        self.assignments_path = Path(assignments_path)
        self.legacy_desktop_layouts_path = Path(legacy_desktop_layouts_path)
        self._lock = threading.Lock()

    # ---- layouts ----
    def list_layouts(self) -> List[LayoutDefinition]:
        with self._lock:
            raw = self._read_json(self.layouts_path, default={"layouts": {}})
        layouts = raw.get("layouts") if isinstance(raw, dict) else {}
        out: List[LayoutDefinition] = []
        if isinstance(layouts, dict):
            for _, v in layouts.items():
                if isinstance(v, dict):
                    out.append(LayoutDefinition.from_dict(v))
        out.sort(key=lambda x: (x.name or x.id))
        return out

    def get_layout(self, layout_id: str) -> Optional[LayoutDefinition]:
        if not layout_id:
            return None
        with self._lock:
            raw = self._read_json(self.layouts_path, default={"layouts": {}})
        layouts = raw.get("layouts") if isinstance(raw, dict) else {}
        v = layouts.get(layout_id) if isinstance(layouts, dict) else None
        if isinstance(v, dict):
            return LayoutDefinition.from_dict(v)
        # fallback: allow lookup by name
        if isinstance(layouts, dict):
            for _, entry in layouts.items():
                if isinstance(entry, dict) and str(entry.get("name") or "").lower() == str(layout_id).lower():
                    return LayoutDefinition.from_dict(entry)
        return None

    def upsert_layout(self, layout: LayoutDefinition) -> LayoutDefinition:
        if not layout.id:
            raise ValueError("layout.id is required")
        if not layout.name:
            layout.name = layout.id
        now = _utc_now_iso()
        layout.updated_at = now
        if not layout.created_at:
            layout.created_at = now
        with self._lock:
            raw = self._read_json(self.layouts_path, default={"layouts": {}})
            if not isinstance(raw, dict):
                raw = {"layouts": {}}
            raw.setdefault("layouts", {})
            raw["layouts"][layout.id] = layout.to_dict()
            _atomic_write_json(self.layouts_path, raw)
        return layout

    def delete_layout(self, layout_id: str) -> bool:
        if not layout_id:
            return False
        with self._lock:
            raw = self._read_json(self.layouts_path, default={"layouts": {}})
            if not isinstance(raw, dict) or not isinstance(raw.get("layouts"), dict):
                return False
            existed = layout_id in raw["layouts"]
            raw["layouts"].pop(layout_id, None)
            _atomic_write_json(self.layouts_path, raw)
        return existed

    # ---- profiles ----
    def list_profiles(self) -> List[CameraProfile]:
        with self._lock:
            raw = self._read_json(self.profiles_path, default={"profiles": {}})
        profiles = raw.get("profiles") if isinstance(raw, dict) else {}
        out: List[CameraProfile] = []
        if isinstance(profiles, dict):
            for _, v in profiles.items():
                if isinstance(v, dict):
                    out.append(CameraProfile.from_dict(v))
        out.sort(key=lambda x: (x.name or x.id))
        return out

    def get_profile(self, profile_id: str) -> Optional[CameraProfile]:
        if not profile_id:
            return None
        with self._lock:
            raw = self._read_json(self.profiles_path, default={"profiles": {}})
        profiles = raw.get("profiles") if isinstance(raw, dict) else {}
        v = profiles.get(profile_id) if isinstance(profiles, dict) else None
        if isinstance(v, dict):
            return CameraProfile.from_dict(v)
        # fallback: allow lookup by name
        if isinstance(profiles, dict):
            for _, entry in profiles.items():
                if isinstance(entry, dict) and str(entry.get("name") or "").lower() == str(profile_id).lower():
                    return CameraProfile.from_dict(entry)
        return None

    def upsert_profile(self, profile: CameraProfile) -> CameraProfile:
        if not profile.id:
            raise ValueError("profile.id is required")
        if not profile.name:
            profile.name = profile.id
        now = _utc_now_iso()
        profile.updated_at = now
        if not profile.created_at:
            profile.created_at = now
        with self._lock:
            raw = self._read_json(self.profiles_path, default={"profiles": {}})
            if not isinstance(raw, dict):
                raw = {"profiles": {}}
            raw.setdefault("profiles", {})
            raw["profiles"][profile.id] = profile.to_dict()
            _atomic_write_json(self.profiles_path, raw)
        return profile

    def create_profile(
        self,
        name: str,
        overlays: Optional[Dict[str, Any]] = None,
        ai_pipeline: Optional[Dict[str, Any]] = None,
        monitoring_tools: Optional[Dict[str, Any]] = None,
        meta: Optional[Dict[str, Any]] = None,
        profile_id: Optional[str] = None,
    ) -> CameraProfile:
        pid = profile_id or str(uuid.uuid4())
        p = CameraProfile(
            id=pid,
            name=name or pid,
            overlays=overlays or {},
            ai_pipeline=ai_pipeline or {},
            monitoring_tools=monitoring_tools or {},
            meta=meta or {},
        )
        return self.upsert_profile(p)

    def delete_profile(self, profile_id: str, remove_assignments: bool = True) -> bool:
        if not profile_id:
            return False
        with self._lock:
            raw = self._read_json(self.profiles_path, default={"profiles": {}})
            if not isinstance(raw, dict) or not isinstance(raw.get("profiles"), dict):
                return False
            existed = profile_id in raw["profiles"]
            raw["profiles"].pop(profile_id, None)
            _atomic_write_json(self.profiles_path, raw)

            if remove_assignments:
                assigns = self._read_json(self.assignments_path, default={"assignments": {}})
                if isinstance(assigns, dict) and isinstance(assigns.get("assignments"), dict):
                    dirty = False
                    for cam_id, val in list(assigns["assignments"].items()):
                        if val == profile_id:
                            assigns["assignments"].pop(cam_id, None)
                            dirty = True
                        elif isinstance(val, list) and profile_id in val:
                            assigns["assignments"][cam_id] = [x for x in val if x != profile_id]
                            dirty = True
                    if dirty:
                        _atomic_write_json(self.assignments_path, assigns)

        return existed

    # ---- assignments ----
    def get_assignments(self) -> Dict[str, CameraProfileAssignmentValue]:
        with self._lock:
            raw = self._read_json(self.assignments_path, default={"assignments": {}})
        assigns = raw.get("assignments") if isinstance(raw, dict) else {}
        return assigns if isinstance(assigns, dict) else {}

    def set_assignment(self, camera_id: str, profile_id_or_list: CameraProfileAssignmentValue) -> None:
        if not camera_id:
            raise ValueError("camera_id is required")
        with self._lock:
            raw = self._read_json(self.assignments_path, default={"assignments": {}})
            if not isinstance(raw, dict):
                raw = {"assignments": {}}
            raw.setdefault("assignments", {})
            raw["assignments"][camera_id] = profile_id_or_list
            _atomic_write_json(self.assignments_path, raw)

    def bulk_apply_profile(
        self,
        profile_id: str,
        camera_ids: List[str],
        mode: str = "replace",  # replace|append
    ) -> Dict[str, Any]:
        if not profile_id:
            raise ValueError("profile_id is required")
        camera_ids = [str(c).strip() for c in (camera_ids or []) if str(c).strip()]
        if not camera_ids:
            return {"applied": 0, "camera_ids": []}

        with self._lock:
            raw = self._read_json(self.assignments_path, default={"assignments": {}})
            if not isinstance(raw, dict):
                raw = {"assignments": {}}
            raw.setdefault("assignments", {})

            applied = 0
            for cam_id in camera_ids:
                if mode == "append":
                    existing = raw["assignments"].get(cam_id)
                    if existing is None:
                        raw["assignments"][cam_id] = [profile_id]
                        applied += 1
                    elif isinstance(existing, str):
                        if existing != profile_id:
                            raw["assignments"][cam_id] = [existing, profile_id]
                            applied += 1
                    elif isinstance(existing, list):
                        if profile_id not in existing:
                            raw["assignments"][cam_id] = existing + [profile_id]
                            applied += 1
                else:
                    raw["assignments"][cam_id] = profile_id
                    applied += 1

            _atomic_write_json(self.assignments_path, raw)

        return {"applied": applied, "camera_ids": camera_ids}

    # ---- migration ----
    def migrate_from_legacy_desktop_layouts(
        self,
        overwrite_layouts: bool = False,
        create_profiles: bool = True,
        overwrite_assignments: bool = False,
    ) -> Dict[str, Any]:
        """
        Convert `data/desktop_layouts.json` (legacy desktop) into:
          - layouts.json (layout definitions; NO embedded shapes)
          - camera_profiles.json (generated profiles from embedded shapes, optional)
          - camera_profile_assignments.json (assign cameras to generated profiles, optional)
        """
        legacy = self._read_json(self.legacy_desktop_layouts_path, default={})
        if not isinstance(legacy, dict) or not legacy:
            return {"migrated_layouts": 0, "migrated_profiles": 0, "note": "no_legacy_layouts_found"}

        existing_layout_ids = {l.id for l in self.list_layouts()}
        existing_profiles = {p.id for p in self.list_profiles()}
        assignments = self.get_assignments()

        migrated_layouts = 0
        migrated_profiles = 0

        for legacy_name, entries in legacy.items():
            if not isinstance(entries, list):
                continue
            layout_id = str(legacy_name)
            if (layout_id in existing_layout_ids) and not overwrite_layouts:
                continue

            widgets: List[WidgetDefinition] = []
            for idx, entry in enumerate(entries):
                if not isinstance(entry, dict):
                    continue
                wtype = str(entry.get("type") or "unknown")
                wid = str(entry.get("id") or f"{layout_id}:{wtype}:{idx}")
                widget = WidgetDefinition(
                    id=wid,
                    type=wtype if wtype in ("camera", "terminal", "overlay", "web") else "unknown",  # type: ignore[arg-type]
                    x=int(entry.get("x", 0)),
                    y=int(entry.get("y", 0)),
                    w=int(entry.get("w", 640)),
                    h=int(entry.get("h", 360)),
                    title=str(entry.get("title") or ""),
                    pinned=bool(entry.get("pinned", False)),
                    camera_id=str(entry.get("camera_id") or "") if entry.get("camera_id") else None,
                    view={},
                )

                if widget.type == "camera":
                    cam_settings = entry.get("camera_settings") or {}
                    if isinstance(cam_settings, dict):
                        # Keep view-only toggles; do NOT keep shapes here.
                        widget.view = {
                            "aspect_ratio_locked": cam_settings.get("aspect_ratio_locked"),
                            "stream_quality": cam_settings.get("stream_quality"),
                            "debug_overlay_enabled": cam_settings.get("debug_overlay_enabled"),
                            "motion_boxes_enabled": cam_settings.get("motion_boxes_enabled"),
                            "object_detection_enabled": cam_settings.get("object_detection_enabled"),
                            "show_shape_labels": cam_settings.get("show_shape_labels"),
                            "depth_overlay": cam_settings.get("depth_overlay"),
                            "motion_settings": cam_settings.get("motion_settings"),
                        }

                        shapes = cam_settings.get("shapes")
                        if create_profiles and widget.camera_id and isinstance(shapes, list) and shapes:
                            # Deterministic profile id so repeated migrations are stable.
                            pid = str(uuid.uuid5(uuid.NAMESPACE_URL, f"legacy:{layout_id}:{widget.camera_id}"))
                            if pid not in existing_profiles:
                                profile_name = f"{layout_id}:{(widget.title or widget.camera_id)}"
                                self.create_profile(
                                    name=profile_name,
                                    overlays={"shapes": shapes},
                                    meta={"source": "legacy_desktop_layouts", "layout_id": layout_id, "camera_id": widget.camera_id},
                                    profile_id=pid,
                                )
                                existing_profiles.add(pid)
                                migrated_profiles += 1

                            if overwrite_assignments or (widget.camera_id not in assignments):
                                self.set_assignment(widget.camera_id, pid)
                                assignments[widget.camera_id] = pid

                widgets.append(widget)

            layout = LayoutDefinition(
                id=layout_id,
                name=layout_id,
                widgets=widgets,
                meta={"source": "legacy_desktop_layouts"},
            )
            self.upsert_layout(layout)
            migrated_layouts += 1

        return {
            "migrated_layouts": migrated_layouts,
            "migrated_profiles": migrated_profiles,
        }

    # ---- helpers ----
    def _read_json(self, path: Path, default: Any) -> Any:
        try:
            if not path.exists():
                return default
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return default


