"""Event Rules API helpers (no Qt dependencies).

Legacy Motion Watch migration (``data/motion_watch_settings.json`` per camera):

- ``migrate_motion_watch_settings`` creates or updates a single *Legacy Motion Watch*
  server rule from desktop capture settings when arming Event Rules.
- Idempotent: repeated calls refresh cooldown, save_dir, overlays, and trigger mode
  (zone vs line) from the current settings blob.
- Skipped when the camera already has user-defined (non-legacy) Event Rules so
  custom rules are never overwritten.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional

import requests

from core.paths import get_data_dir

logger = logging.getLogger(__name__)

LEGACY_RULE_NAME = "Legacy Motion Watch"
LEGACY_RULE_ID_PREFIX = "legacy_mw_"
DEFAULT_API_BASE = "http://localhost:5000/api"

# Event rule timing defaults (new rules / missing persisted values only).
DEFAULT_RULE_COOLDOWN_SEC = 1.0
DEFAULT_RULE_COOLDOWN_MS = 1000

# Desktop motion-box detection tuning defaults (GL overlay / MOG2 preview).
DEFAULT_MOTION_SENSITIVITY = 70
DEFAULT_MOTION_MERGE_SIZE = 50


def cooldown_sec_from_ms(cooldown_ms: int) -> float:
    """Convert UI milliseconds to stored ``cooldown_sec`` float seconds."""
    return max(0.0, float(cooldown_ms) / 1000.0)


def cooldown_ms_from_sec(cooldown_sec: float) -> int:
    """Convert stored ``cooldown_sec`` to UI milliseconds."""
    return max(0, int(round(float(cooldown_sec) * 1000)))


def default_motion_detection_tuning() -> Dict[str, int]:
    """Baseline motion sensitivity / merge values for new camera widgets."""
    return {
        "sensitivity": DEFAULT_MOTION_SENSITIVITY,
        "merge_size": DEFAULT_MOTION_MERGE_SIZE,
    }


def _api_get(api_base: str, path: str, *, timeout: float = 8.0) -> Dict[str, Any]:
    url = f"{api_base.rstrip('/')}/{path.lstrip('/')}"
    resp = requests.get(url, timeout=timeout)
    resp.raise_for_status()
    data = resp.json()
    return data if isinstance(data, dict) else {}


def _api_post(api_base: str, path: str, body: Dict[str, Any], *, timeout: float = 12.0) -> Dict[str, Any]:
    url = f"{api_base.rstrip('/')}/{path.lstrip('/')}"
    resp = requests.post(url, json=body, timeout=timeout)
    resp.raise_for_status()
    data = resp.json()
    return data if isinstance(data, dict) else {}


def _api_put(api_base: str, path: str, body: Dict[str, Any], *, timeout: float = 12.0) -> Dict[str, Any]:
    url = f"{api_base.rstrip('/')}/{path.lstrip('/')}"
    resp = requests.put(url, json=body, timeout=timeout)
    resp.raise_for_status()
    data = resp.json()
    return data if isinstance(data, dict) else {}


def _api_delete(api_base: str, path: str, *, timeout: float = 12.0) -> Dict[str, Any]:
    url = f"{api_base.rstrip('/')}/{path.lstrip('/')}"
    resp = requests.delete(url, timeout=timeout)
    resp.raise_for_status()
    data = resp.json()
    return data if isinstance(data, dict) else {}


def list_rules(api_base: str, camera_id: str) -> List[Dict[str, Any]]:
    try:
        data = _api_get(api_base, f"rules?camera_id={camera_id}")
        rules = (data.get("data") or {}).get("rules") or []
        return [r for r in rules if isinstance(r, dict)]
    except Exception as e:
        logger.debug("list_rules failed for %s: %s", camera_id, e)
        return []


def save_rule(api_base: str, rule: Dict[str, Any], *, rule_id: Optional[str] = None) -> Optional[Dict[str, Any]]:
    try:
        if rule_id:
            data = _api_put(api_base, f"rules/{rule_id}", rule)
        else:
            data = _api_post(api_base, "rules", rule)
        saved = data.get("data")
        return saved if isinstance(saved, dict) else None
    except Exception as e:
        logger.warning("save_rule failed: %s", e)
        return None


def delete_rule(api_base: str, rule_id: str) -> bool:
    rid = str(rule_id or "").strip()
    if not rid:
        return False
    try:
        data = _api_delete(api_base, f"rules/{rid}")
        return bool(data.get("success"))
    except Exception as e:
        logger.warning("delete_rule failed for %s: %s", rid, e)
        return False


def set_rules_enabled(api_base: str, camera_id: str, enabled: bool) -> int:
    updated = 0
    for rule in list_rules(api_base, camera_id):
        rid = rule.get("id")
        if not rid:
            continue
        try:
            _api_put(api_base, f"rules/{rid}", {"enabled": bool(enabled)})
            updated += 1
        except Exception:
            continue
    return updated


def is_legacy_rule(rule: Dict[str, Any]) -> bool:
    if rule.get("name") == LEGACY_RULE_NAME:
        return True
    return str(rule.get("id", "")).startswith(LEGACY_RULE_ID_PREFIX)


def has_custom_event_rules(rules: List[Dict[str, Any]]) -> bool:
    return any(is_legacy_rule(r) is False for r in rules if isinstance(r, dict))


def has_custom_rules(rules: List[Dict[str, Any]]) -> bool:
    """Alias for :func:`has_custom_event_rules`."""
    return has_custom_event_rules(rules)


def load_motion_watch_settings_from_disk(camera_id: str) -> Dict[str, Any]:
    """Load per-camera Motion Watch settings from ``data/motion_watch_settings.json``."""
    defaults: Dict[str, Any] = {
        "cooldown_ms": DEFAULT_RULE_COOLDOWN_MS,
        "include_overlays": True,
        "save_dir": "captures/motion_watch",
        "allow_zone": True,
        "allow_line": True,
    }
    path = get_data_dir() / "motion_watch_settings.json"
    if not path.is_file():
        return dict(defaults)
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        saved = raw.get(camera_id) if isinstance(raw, dict) else None
        if not isinstance(saved, dict):
            return dict(defaults)
        settings = {**defaults, **saved}
        if "cooldown_ms" not in saved and "cooldown_sec" in saved:
            try:
                settings["cooldown_ms"] = int(float(saved["cooldown_sec"]) * 1000)
            except Exception:
                settings["cooldown_ms"] = defaults["cooldown_ms"]
        return settings
    except Exception as e:
        logger.debug("load_motion_watch_settings_from_disk failed for %s: %s", camera_id, e)
        return dict(defaults)


def snapshot_action_from_motion_watch_settings(settings: Dict[str, Any]) -> Dict[str, Any]:
    """Build a server ``snapshot`` action dict from Motion Watch / Capture Settings."""
    action: Dict[str, Any] = {
        "type": "snapshot",
        "include_overlays": bool(settings.get("include_overlays", True)),
        "save_dir": str(settings.get("save_dir") or "captures/motion_watch"),
        "resize_w": int(settings.get("resize_w", 0) or 0),
    }
    quality = settings.get("quality")
    if quality is not None:
        try:
            action["quality"] = int(quality)
        except (TypeError, ValueError):
            pass
    return action


def legacy_rule_from_motion_watch_settings(camera_id: str, settings: Dict[str, Any]) -> Dict[str, Any]:
    cooldown_ms = settings.get("cooldown_ms")
    if cooldown_ms is None:
        cooldown_sec = float(settings.get("cooldown_sec", DEFAULT_RULE_COOLDOWN_SEC) or DEFAULT_RULE_COOLDOWN_SEC)
    else:
        cooldown_sec = max(0.0, float(cooldown_ms) / 1000.0)

    allow_zone = bool(settings.get("allow_zone", True))
    allow_line = bool(settings.get("allow_line", True))
    trigger = "zone_enter"
    if allow_line and not allow_zone:
        trigger = "line_cross"
    elif allow_line and allow_zone:
        trigger = "zone_enter"

    return {
        "id": f"{LEGACY_RULE_ID_PREFIX}{camera_id}",
        "name": LEGACY_RULE_NAME,
        "camera_id": camera_id,
        "trigger": trigger,
        "conditions": {
            "cooldown_sec": cooldown_sec,
            "cooldown_per_track": True,
            "tracker_namespace": "backend_sort",
            "allow_zone": allow_zone,
            "allow_line": allow_line,
        },
        "actions": [snapshot_action_from_motion_watch_settings(settings)],
        "enabled": True,
    }


def ensure_legacy_rule(api_base: str, camera_id: str, settings: Dict[str, Any]) -> Optional[str]:
    rules = list_rules(api_base, camera_id)
    for rule in rules:
        if is_legacy_rule(rule):
            rid = str(rule.get("id"))
            body = legacy_rule_from_motion_watch_settings(camera_id, settings)
            body.pop("id", None)
            save_rule(api_base, body, rule_id=rid)
            return rid
    body = legacy_rule_from_motion_watch_settings(camera_id, settings)
    saved = save_rule(api_base, body)
    return str(saved.get("id")) if saved else None


# Desktop Motion Watch captures for these shape types are handled server-side
# via track_event rules (zone_enter, line_cross, dwell_met) when Event Rules
# are armed. Tags remain desktop-only unless a server rule is added later.
SERVER_AUTHORITATIVE_SHAPE_TYPES = frozenset({"zone", "line"})


def shapes_to_api_payload(shapes: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    """Convert desktop GL widget shapes into API/stream-server zones payload."""
    zones: List[Dict[str, Any]] = []
    lines: List[Dict[str, Any]] = []
    tags: List[Dict[str, Any]] = []
    for sh in shapes or []:
        if not isinstance(sh, dict):
            continue
        kind = str(sh.get("kind") or "").strip().lower()
        sid = str(sh.get("id") or "").strip()
        if not sid:
            continue
        base = {
            "id": sid,
            "label": str(sh.get("label") or sh.get("name") or sid),
            "enabled": bool(sh.get("enabled", True)),
        }
        if kind == "zone":
            pts = sh.get("pts") or sh.get("points") or []
            points = []
            for p in pts:
                if isinstance(p, dict):
                    points.append({"x": float(p.get("x", 0)), "y": float(p.get("y", 0))})
            if len(points) >= 3:
                zones.append({**base, "points": points})
        elif kind == "line":
            p1 = sh.get("p1") or {}
            p2 = sh.get("p2") or {}
            if isinstance(p1, dict) and isinstance(p2, dict):
                lines.append(
                    {
                        **base,
                        "p1": {"x": float(p1.get("x", 0)), "y": float(p1.get("y", 0))},
                        "p2": {"x": float(p2.get("x", 1)), "y": float(p2.get("y", 1))},
                    }
                )
        elif kind == "tag":
            anchor = sh.get("anchor") if isinstance(sh.get("anchor"), dict) else {}
            ax = float(anchor.get("x", sh.get("x", 0.5)))
            ay = float(anchor.get("y", sh.get("y", 0.5)))
            tags.append({**base, "anchor": {"x": ax, "y": ay}, "x": ax, "y": ay})
    return {"zones": zones, "lines": lines, "tags": tags}


def sync_camera_shapes(api_base: str, camera_id: str, shapes: List[Dict[str, Any]]) -> bool:
    """Persist overlay shapes to the server so TrackSceneEngine can evaluate them."""
    body = shapes_to_api_payload(shapes)
    try:
        _api_put(api_base, f"cameras/{camera_id}/zones", body)
        return True
    except Exception as e:
        logger.warning("sync_camera_shapes failed for %s: %s", camera_id, e)
        return False


def get_backend_detection_config(api_base: str, camera_id: str) -> Dict[str, Any]:
    """Fetch full backend detection config for *camera_id* (empty dict on failure)."""
    try:
        data = _api_get(api_base, f"cameras/{camera_id}/detection-config")
        cfg = data.get("data") if isinstance(data.get("data"), dict) else data
        return dict(cfg) if isinstance(cfg, dict) else {}
    except Exception as e:
        logger.debug("get_backend_detection_config failed for %s: %s", camera_id, e)
        return {}


def apply_backend_detection_config(api_base: str, camera_id: str, config: Dict[str, Any]) -> bool:
    """Apply backend detection config fields to *camera_id*."""
    if not isinstance(config, dict) or not config:
        return False
    try:
        _api_put(api_base, f"cameras/{camera_id}/detection-config", dict(config))
        return True
    except Exception as e:
        logger.warning("apply_backend_detection_config failed for %s: %s", camera_id, e)
        return False


def replace_camera_rules(
    api_base: str,
    camera_id: str,
    rules: List[Dict[str, Any]],
    *,
    delete_existing: bool = True,
) -> int:
    """Replace event rules on *camera_id* with *rules*. Returns count saved."""
    saved = 0
    if delete_existing:
        for existing in list_rules(api_base, camera_id):
            rid = existing.get("id")
            if rid:
                delete_rule(api_base, str(rid))
    for rule in rules or []:
        if not isinstance(rule, dict):
            continue
        body = dict(rule)
        body.pop("id", None)
        body["camera_id"] = str(camera_id)
        if save_rule(api_base, body):
            saved += 1
    return saved


def get_backend_detection_enabled(api_base: str, camera_id: str) -> Optional[bool]:
    """Return ``verification_enabled`` for *camera_id*, or ``None`` when unknown."""
    try:
        data = _api_get(api_base, f"cameras/{camera_id}/detection-config")
        cfg = data.get("data") if isinstance(data.get("data"), dict) else data
        if isinstance(cfg, dict) and "verification_enabled" in cfg:
            return bool(cfg["verification_enabled"])
        return None
    except Exception as e:
        logger.debug("get_backend_detection_enabled failed for %s: %s", camera_id, e)
        return None


def backend_detection_status_label(enabled: Optional[bool]) -> str:
    """Human-readable backend detection state for UI labels."""
    if enabled is True:
        return "On"
    if enabled is False:
        return "Off"
    return "Unknown"


def ensure_backend_detection_for_rules(
    api_base: str,
    camera_id: str,
    *,
    verification_enabled: bool = True,
) -> bool:
    """Enable or disable backend object detection for server-side track events."""
    try:
        _api_put(
            api_base,
            f"cameras/{camera_id}/detection-config",
            {"verification_enabled": bool(verification_enabled)},
        )
        return True
    except Exception as e:
        logger.warning("ensure_backend_detection_for_rules failed for %s: %s", camera_id, e)
        return False


def filter_desktop_capture_events(
    events: List[Dict[str, Any]],
    *,
    motion_watch_active: bool,
    server_event_rules_active: bool,
) -> List[Dict[str, Any]]:
    """Return shape-trigger events that should still drive desktop snapshot capture."""
    if not events:
        return []
    if not motion_watch_active or not server_event_rules_active:
        return list(events)
    return [
        ev
        for ev in events
        if isinstance(ev, dict) and ev.get("shape_type") not in SERVER_AUTHORITATIVE_SHAPE_TYPES
    ]


def should_suppress_desktop_track_capture(
    events: List[Dict[str, Any]],
    *,
    motion_watch_active: bool,
    server_event_rules_active: bool,
) -> bool:
    """True when armed server rules own all events (no desktop snapshot needed)."""
    if not events or not motion_watch_active or not server_event_rules_active:
        return False
    return len(filter_desktop_capture_events(
        events,
        motion_watch_active=motion_watch_active,
        server_event_rules_active=server_event_rules_active,
    )) == 0


def migrate_motion_watch_settings(
    api_base: str,
    camera_id: str,
    settings: Optional[Dict[str, Any]] = None,
) -> Optional[str]:
    """
    Import or refresh the legacy Motion Watch rule for *camera_id*.

    Returns the legacy rule id when created/updated, or ``None`` when skipped
    because custom Event Rules already exist for the camera.
    """
    if settings is None:
        settings = load_motion_watch_settings_from_disk(camera_id)
    rules = list_rules(api_base, camera_id)
    if has_custom_event_rules(rules):
        return None
    return ensure_legacy_rule(api_base, camera_id, settings)
