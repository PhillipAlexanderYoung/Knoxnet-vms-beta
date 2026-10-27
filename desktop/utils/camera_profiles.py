"""Camera profile export/import helpers (Qt-free).

Profiles capture reusable per-camera configuration: shapes, event rules,
motion/capture settings, overlay toggles, and backend detection config.
"""

from __future__ import annotations

import copy
import uuid
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Set

PROFILE_SCHEMA_VERSION = 2

# Keys stripped from exported/imported event rules (server-assigned or transient).
RULE_STRIP_KEYS = frozenset(
    {
        "id",
        "created_at",
        "updated_at",
        "last_triggered_at",
        "trigger_count",
        "runtime",
    }
)

# Counter runtime values — copy config but reset live counts when applying.
COUNTER_RUNTIME_KEYS = frozenset(
    {
        "counter_value",
        "count_current",
        "count_total",
        "last_count_at",
    }
)

COLOR_KEYS = frozenset({"color", "text_color", "interaction_color", "trail_color"})


def _json_safe(value: Any) -> Any:
    """Best-effort conversion to JSON-serializable values."""
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    # QColor-like objects expose name()
    try:
        name_fn = getattr(value, "name", None)
        if callable(name_fn):
            return str(name_fn())
    except Exception:
        pass
    return str(value)


def sanitize_shape(shape: Mapping[str, Any]) -> Dict[str, Any]:
    """Return a JSON-safe copy of a GL widget shape dict."""
    out = _json_safe(dict(shape))
    if not isinstance(out, dict):
        return {}
    for key in COLOR_KEYS:
        if key in out:
            out[key] = _json_safe(out[key])
    return out


def sanitize_shapes(
    shapes: Sequence[Mapping[str, Any]],
    *,
    exclude_kinds: Optional[Set[str]] = None,
) -> List[Dict[str, Any]]:
    excluded = {str(k).strip().lower() for k in (exclude_kinds or set()) if str(k).strip()}
    out: List[Dict[str, Any]] = []
    for sh in shapes or []:
        if not isinstance(sh, Mapping):
            continue
        kind = str(sh.get("kind") or "").strip().lower()
        if excluded and kind in excluded:
            continue
        sid = str(sh.get("id") or "").strip()
        if not sid:
            continue
        out.append(sanitize_shape(sh))
    return out


def export_rule_for_profile(
    rule: Mapping[str, Any],
    *,
    include_legacy: bool = False,
    source_camera_id: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """Normalize a server rule dict for profile storage."""
    if not isinstance(rule, Mapping):
        return None
    from desktop.utils.event_rules_api import is_legacy_rule

    if not include_legacy and is_legacy_rule(rule):
        return None
    out = copy.deepcopy(dict(rule))
    for key in RULE_STRIP_KEYS:
        out.pop(key, None)
    if source_camera_id:
        out["camera_id"] = str(source_camera_id)
    cond = out.get("conditions")
    if isinstance(cond, dict):
        cleaned = copy.deepcopy(cond)
        for rk in COUNTER_RUNTIME_KEYS:
            cleaned.pop(rk, None)
        out["conditions"] = cleaned
    return out


def export_rules_for_profile(
    rules: Sequence[Mapping[str, Any]],
    *,
    include_legacy: bool = False,
    source_camera_id: Optional[str] = None,
) -> List[Dict[str, Any]]:
    exported: List[Dict[str, Any]] = []
    for rule in rules or []:
        normalized = export_rule_for_profile(
            rule,
            include_legacy=include_legacy,
            source_camera_id=source_camera_id,
        )
        if normalized:
            exported.append(normalized)
    return exported


def build_shape_id_map(
    profile_shapes: Sequence[Mapping[str, Any]],
    *,
    preserve_ids: bool = True,
) -> Dict[str, str]:
    """Map source shape ids to target ids (identity or regenerated UUIDs)."""
    mapping: Dict[str, str] = {}
    for sh in profile_shapes or []:
        if not isinstance(sh, Mapping):
            continue
        sid = str(sh.get("id") or "").strip()
        if not sid:
            continue
        if preserve_ids:
            mapping[sid] = sid
        else:
            mapping[sid] = str(uuid.uuid4())
    return mapping


def remap_shape_ids(
    shapes: Sequence[Mapping[str, Any]],
    id_map: Mapping[str, str],
) -> List[Dict[str, Any]]:
    """Return shapes with ids rewritten using *id_map*."""
    out: List[Dict[str, Any]] = []
    for sh in shapes or []:
        if not isinstance(sh, Mapping):
            continue
        ss = copy.deepcopy(dict(sh))
        old_id = str(ss.get("id") or "").strip()
        if old_id and old_id in id_map:
            ss["id"] = id_map[old_id]
        out.append(sanitize_shape(ss))
    return out


def _rewrite_shape_refs_in_value(value: Any, id_map: Mapping[str, str]) -> Any:
    if isinstance(value, str) and value in id_map:
        return id_map[value]
    if isinstance(value, list):
        return [_rewrite_shape_refs_in_value(v, id_map) for v in value]
    if isinstance(value, dict):
        out = {}
        for k, v in value.items():
            if k in ("shape_id", "motion_path_shape_ref") and isinstance(v, str) and v in id_map:
                out[k] = id_map[v]
            else:
                out[k] = _rewrite_shape_refs_in_value(v, id_map)
        return out
    return value


def rewrite_rule_for_camera(
    rule: Mapping[str, Any],
    target_camera_id: str,
    shape_id_map: Optional[Mapping[str, str]] = None,
    *,
    reset_counter_runtime: bool = True,
) -> Dict[str, Any]:
    """Prepare a profile rule for persistence on *target_camera_id*."""
    out = copy.deepcopy(dict(rule))
    for key in RULE_STRIP_KEYS:
        out.pop(key, None)
    out["camera_id"] = str(target_camera_id)
    sid = str(out.get("shape_id") or "").strip()
    if shape_id_map and sid and sid in shape_id_map:
        out["shape_id"] = shape_id_map[sid]
    if shape_id_map:
        out = _rewrite_shape_refs_in_value(out, shape_id_map)
    if reset_counter_runtime:
        cond = out.get("conditions")
        if isinstance(cond, dict):
            for rk in COUNTER_RUNTIME_KEYS:
                cond.pop(rk, None)
    return out


def rewrite_rules_for_camera(
    rules: Sequence[Mapping[str, Any]],
    target_camera_id: str,
    shape_id_map: Optional[Mapping[str, str]] = None,
    *,
    reset_counter_runtime: bool = True,
) -> List[Dict[str, Any]]:
    return [
        rewrite_rule_for_camera(
            r,
            target_camera_id,
            shape_id_map,
            reset_counter_runtime=reset_counter_runtime,
        )
        for r in (rules or [])
        if isinstance(r, Mapping)
    ]


def build_overlays_payload(
    *,
    shapes: Optional[Sequence[Mapping[str, Any]]] = None,
    motion_settings: Optional[Mapping[str, Any]] = None,
    detection_settings: Optional[Mapping[str, Any]] = None,
    motion_boxes_enabled: Optional[bool] = None,
    debug_overlay_enabled: Optional[bool] = None,
    show_shape_labels: Optional[bool] = None,
    aspect_ratio_locked: Optional[bool] = None,
    stream_quality: Optional[str] = None,
) -> Dict[str, Any]:
    overlays: Dict[str, Any] = {}
    if shapes is not None:
        overlays["shapes"] = sanitize_shapes(shapes)
    if motion_settings is not None:
        overlays["motion_settings"] = _json_safe(dict(motion_settings))
    if detection_settings is not None:
        overlays["detection_settings"] = _json_safe(dict(detection_settings))
    if motion_boxes_enabled is not None:
        overlays["motion_boxes_enabled"] = bool(motion_boxes_enabled)
    if debug_overlay_enabled is not None:
        overlays["debug_overlay_enabled"] = bool(debug_overlay_enabled)
    if show_shape_labels is not None:
        overlays["show_shape_labels"] = bool(show_shape_labels)
    if aspect_ratio_locked is not None:
        overlays["aspect_ratio_locked"] = bool(aspect_ratio_locked)
    if stream_quality is not None:
        overlays["stream_quality"] = str(stream_quality)
    return overlays


def build_ai_pipeline_payload(
    *,
    object_detection_enabled: Optional[bool] = None,
    desktop_object_detection_enabled: Optional[bool] = None,
    desktop_detector_config: Optional[Mapping[str, Any]] = None,
    backend_detection: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    ai: Dict[str, Any] = {}
    if object_detection_enabled is not None:
        ai["object_detection_enabled"] = bool(object_detection_enabled)
    if desktop_object_detection_enabled is not None:
        ai["desktop_object_detection_enabled"] = bool(desktop_object_detection_enabled)
    if desktop_detector_config is not None:
        ai["desktop_detector_config"] = _json_safe(dict(desktop_detector_config))
    if backend_detection is not None:
        ai["backend_detection"] = _json_safe(dict(backend_detection))
    return ai


def build_monitoring_tools_payload(
    *,
    motion_watch_settings: Optional[Mapping[str, Any]] = None,
    event_rules: Optional[Sequence[Mapping[str, Any]]] = None,
) -> Dict[str, Any]:
    mt: Dict[str, Any] = {}
    if motion_watch_settings is not None:
        mt["motion_watch_settings"] = _json_safe(dict(motion_watch_settings))
    if event_rules is not None:
        mt["event_rules"] = list(event_rules)
    return mt


def build_camera_profile_payload(
    *,
    name: str,
    source_camera_id: str,
    overlays: Optional[Mapping[str, Any]] = None,
    ai_pipeline: Optional[Mapping[str, Any]] = None,
    monitoring_tools: Optional[Mapping[str, Any]] = None,
    profile_id: Optional[str] = None,
    extra_meta: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Assemble a profile dict compatible with :class:`core.layout_models.CameraProfile`."""
    meta = {
        "schema_version": PROFILE_SCHEMA_VERSION,
        "source_camera_id": str(source_camera_id),
        "source": "desktop_camera_profile",
    }
    if extra_meta:
        meta.update(dict(extra_meta))
    return {
        "id": profile_id or str(uuid.uuid4()),
        "name": name,
        "overlays": dict(overlays or {}),
        "ai_pipeline": dict(ai_pipeline or {}),
        "monitoring_tools": dict(monitoring_tools or {}),
        "meta": meta,
    }


def profile_to_widget_view(profile: Mapping[str, Any]) -> Dict[str, Any]:
    """Convert a stored profile into layout-style camera view settings."""
    overlays = profile.get("overlays") if isinstance(profile.get("overlays"), dict) else {}
    ai = profile.get("ai_pipeline") if isinstance(profile.get("ai_pipeline"), dict) else {}
    mt = profile.get("monitoring_tools") if isinstance(profile.get("monitoring_tools"), dict) else {}

    view: Dict[str, Any] = {}
    for key in (
        "aspect_ratio_locked",
        "stream_quality",
        "debug_overlay_enabled",
        "motion_boxes_enabled",
        "show_shape_labels",
    ):
        if key in overlays:
            view[key] = overlays[key]
    if "shapes" in overlays:
        view["shapes"] = overlays["shapes"]
    if "motion_settings" in overlays:
        view["motion_settings"] = overlays["motion_settings"]
    if "detection_settings" in overlays:
        view["detection_overlay_settings"] = overlays["detection_settings"]
    if "object_detection_enabled" in ai:
        view["object_detection_enabled"] = ai["object_detection_enabled"]
    if "desktop_object_detection_enabled" in ai:
        view["desktop_object_detection_enabled"] = ai["desktop_object_detection_enabled"]
    if "desktop_detector_config" in ai:
        view["desktop_detector_config"] = ai["desktop_detector_config"]
    if "backend_detection" in ai:
        view["backend_detection"] = ai["backend_detection"]
    if "motion_watch_settings" in mt:
        view["motion_watch_settings"] = mt["motion_watch_settings"]
    if "event_rules" in mt:
        view["event_rules"] = mt["event_rules"]
    return view


def merge_profile_dicts(profiles: Iterable[Mapping[str, Any]]) -> Dict[str, Any]:
    """Merge multiple profile dicts (later entries override earlier ones)."""
    prof_list = [p for p in (profiles or []) if isinstance(p, Mapping)]
    merged: Dict[str, Any] = {
        "overlays": {},
        "ai_pipeline": {},
        "monitoring_tools": {},
        "meta": {},
    }
    all_shapes: List[Dict[str, Any]] = []
    all_rules: List[Dict[str, Any]] = []
    for prof in prof_list:
        for section in ("overlays", "ai_pipeline", "monitoring_tools", "meta"):
            chunk = prof.get(section)
            if isinstance(chunk, dict):
                if section == "overlays":
                    for k, v in chunk.items():
                        if k == "shapes" and isinstance(v, list):
                            all_shapes.extend(v)
                        else:
                            merged[section][k] = v
                elif section == "monitoring_tools":
                    for k, v in chunk.items():
                        if k == "event_rules" and isinstance(v, list):
                            all_rules.extend(v)
                        else:
                            merged[section][k] = v
                else:
                    merged[section].update(chunk)
    if all_shapes:
        merged["overlays"]["shapes"] = sanitize_shapes(all_shapes)
    if all_rules:
        merged["monitoring_tools"]["event_rules"] = list(all_rules)
    return merged
