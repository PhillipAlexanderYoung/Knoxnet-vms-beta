"""Pure helpers for shape Event Rule dialog (no Qt dependency)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from core.automation.actions.script import DEFAULT_SCRIPT_TIMEOUT_SEC, normalize_args, normalize_runner
from core.automation.conditions import BACKEND_SORT_NAMESPACE, MOTION_BOX_NAMESPACE
from core.paths import get_event_rules_scripts_dir
from desktop.utils.event_rules_api import snapshot_action_from_motion_watch_settings

DEFAULT_TRIGGER_MODE = "auto_path"

PATH_TRIGGER_MODES = [
    ("Auto (from path)", "auto_path"),
    ("Any interaction", "any_interaction"),
    ("Path match (directional)", "path_match"),
]

ZONE_TRIGGERS = [
    ("Enter zone", "zone_enter"),
    ("Exit zone", "zone_exit"),
    ("Dwell met", "dwell_met"),
]

LINE_TRIGGERS = [
    ("Cross line", "line_cross"),
]

TAG_TRIGGERS = [
    ("Near tag", "near_tag"),
]

EXPLICIT_SHAPE_TRIGGERS = frozenset(
    {"zone_enter", "zone_exit", "dwell_met", "line_cross", "near_tag"}
)

SCRIPT_RUNNER_OPTIONS = [
    ("Python", "python"),
    ("Shell script (.sh)", "shell"),
    ("Executable", "executable"),
]

DEFAULT_SCRIPT_RUNNER = "python"


def _default_trigger(kind: str) -> str:
    k = str(kind or "").strip().lower()
    if k == "line":
        return LINE_TRIGGERS[0][1]
    if k == "tag":
        return TAG_TRIGGERS[0][1]
    return ZONE_TRIGGERS[0][1]


def trigger_mode_options_for_kind(kind: str) -> List[Tuple[str, str]]:
    """User-facing trigger choices for the shape Event Rule dialog."""
    k = str(kind or "").strip().lower()
    options = list(PATH_TRIGGER_MODES)
    if k == "zone":
        options.extend(
            [
                ("Enter shape", "zone_enter"),
                ("Exit shape", "zone_exit"),
                ("Dwell in shape", "dwell_met"),
            ]
        )
    elif k == "line":
        options.append(("Cross line", "line_cross"))
    elif k == "tag":
        options.append(("Near tag", "near_tag"))
    return options


def effective_trigger_from_mode(
    *,
    mode: str,
    shape_kind: str,
    derived_trigger: str,
    has_path: bool,
) -> str:
    trigger_mode = str(mode or "auto_path").strip().lower()
    if trigger_mode == "path_match":
        return "path_match"
    if trigger_mode == "any_interaction":
        return "any_interaction"
    if trigger_mode in EXPLICIT_SHAPE_TRIGGERS:
        return trigger_mode
    if trigger_mode == "auto_path" and has_path:
        return str(derived_trigger or _default_trigger(shape_kind))
    return _default_trigger(shape_kind)


def script_action_from_settings(
    *,
    enabled: bool,
    path: str,
    runner: str,
    args_text: str,
    timeout_sec: int,
) -> Optional[Dict[str, Any]]:
    if not enabled:
        return None
    script_path = str(path or "").strip()
    if not script_path:
        return None
    action: Dict[str, Any] = {
        "type": "script",
        "language": normalize_runner(runner or DEFAULT_SCRIPT_RUNNER),
        "path": script_path,
        "timeout_sec": max(1, int(timeout_sec or DEFAULT_SCRIPT_TIMEOUT_SEC)),
    }
    args = normalize_args(args_text)
    if args:
        action["args"] = args
    return action


def build_fresh_shape_rule_conditions(
    *,
    trigger_mode: str = DEFAULT_TRIGGER_MODE,
    motion_path: Optional[List[Dict[str, float]]] = None,
    derived_trigger: str = "zone_enter",
    motion_enabled: bool = True,
    detection_enabled: bool = False,
    show_counter: str = "always",
    take_screenshot: bool = True,
) -> Dict[str, Any]:
    """Build conditions/actions like a new shape Event Rule dialog save (for tests)."""
    conditions: Dict[str, Any] = {
        **build_event_source_conditions(
            motion_enabled=motion_enabled,
            detection_enabled=detection_enabled,
        ),
        "cooldown_sec": 2.0,
        "cooldown_per_track": True,
    }
    mode = str(trigger_mode or DEFAULT_TRIGGER_MODE).strip().lower()
    if mode == "any_interaction":
        conditions["any_interaction"] = True
    path = list(motion_path or [])
    if len(path) >= 2 and mode != "any_interaction":
        conditions["motion_path"] = path
        conditions["motion_path_space"] = "frame"
        conditions["derived_trigger"] = derived_trigger
    if show_counter and str(show_counter).strip().lower() != "off":
        conditions["show_counter"] = show_counter
    actions = build_rule_actions(
        take_screenshot=take_screenshot,
        motion_watch_settings={"save_dir": "captures/motion_watch"},
        run_script=False,
        script_path="",
        script_runner=DEFAULT_SCRIPT_RUNNER,
        script_args="",
        script_timeout_sec=int(DEFAULT_SCRIPT_TIMEOUT_SEC),
    )
    effective = effective_trigger_from_mode(
        mode=mode,
        shape_kind="zone",
        derived_trigger=derived_trigger,
        has_path=len(path) >= 2,
    )
    return {
        "trigger_mode": mode,
        "trigger": effective,
        "conditions": conditions,
        "actions": actions,
    }


def build_rule_actions(
    *,
    take_screenshot: bool,
    motion_watch_settings: Dict[str, Any],
    run_script: bool,
    script_path: str,
    script_runner: str,
    script_args: str,
    script_timeout_sec: int,
) -> List[Dict[str, Any]]:
    actions: List[Dict[str, Any]] = []
    if take_screenshot:
        actions.append(snapshot_action_from_motion_watch_settings(motion_watch_settings))
    script_action = script_action_from_settings(
        enabled=run_script,
        path=script_path,
        runner=script_runner,
        args_text=script_args,
        timeout_sec=script_timeout_sec,
    )
    if script_action:
        actions.append(script_action)
    return actions


def rule_has_script_action(rule: dict) -> bool:
    actions = rule.get("actions") if isinstance(rule.get("actions"), list) else []
    return any(
        isinstance(a, dict) and str(a.get("type") or "").strip().lower() in {"script", "run_script"}
        for a in actions
    )


def rule_has_snapshot_action(rule: dict) -> bool:
    """True when a rule has an enabled snapshot/screenshot action.

    This is the single source of truth used by the desktop trigger pipeline to
    decide whether a counter increment should also dispatch a screenshot.
    """
    if not isinstance(rule, dict):
        return False
    actions = rule.get("actions") if isinstance(rule.get("actions"), list) else []
    for action in actions:
        if not isinstance(action, dict):
            continue
        if str(action.get("type") or "").strip().lower() != "snapshot":
            continue
        if action.get("enabled") is False:
            continue
        return True
    return False


def script_action_from_rule(rule: dict) -> Dict[str, Any]:
    actions = rule.get("actions") if isinstance(rule.get("actions"), list) else []
    for action in actions:
        if not isinstance(action, dict):
            continue
        if str(action.get("type") or "").strip().lower() not in {"script", "run_script"}:
            continue
        return dict(action)
    return {}


def parse_event_source_flags(conditions: Dict[str, Any]) -> Tuple[bool, bool]:
    """Return (motion_boxes_enabled, object_detection_enabled) from rule conditions."""
    cond = conditions if isinstance(conditions, dict) else {}
    namespaces = cond.get("tracker_namespaces")
    if isinstance(namespaces, list) and namespaces:
        ns_set = {str(n).strip() for n in namespaces if str(n).strip()}
        return (
            MOTION_BOX_NAMESPACE in ns_set,
            BACKEND_SORT_NAMESPACE in ns_set,
        )
    ns = str(cond.get("tracker_namespace") or "").strip()
    if ns == "any":
        return True, True
    require_det = cond.get("require_detection")
    if ns == BACKEND_SORT_NAMESPACE or (require_det is True and ns != MOTION_BOX_NAMESPACE):
        return False, True
    if ns == MOTION_BOX_NAMESPACE or require_det is False:
        return True, False
    return False, True


def build_event_source_conditions(
    *,
    motion_enabled: bool,
    detection_enabled: bool,
    classes: Optional[List[str]] = None,
    min_confidence: Optional[float] = None,
) -> Dict[str, Any]:
    """Build tracker namespace fields for shape event rule conditions."""
    if not motion_enabled and not detection_enabled:
        raise ValueError("At least one event source must be selected")
    cond: Dict[str, Any] = {}
    if motion_enabled and detection_enabled:
        cond["tracker_namespaces"] = [MOTION_BOX_NAMESPACE, BACKEND_SORT_NAMESPACE]
        cond["require_detection"] = False
    elif motion_enabled:
        cond["tracker_namespace"] = MOTION_BOX_NAMESPACE
        cond["require_detection"] = False
    else:
        cond["tracker_namespace"] = BACKEND_SORT_NAMESPACE
        cond["require_detection"] = True
    if detection_enabled and classes:
        cond["classes"] = list(classes)
        if min_confidence is not None:
            cond["min_confidence"] = float(min_confidence)
    return cond


def ensure_event_rules_scripts_dir() -> Path:
    scripts_dir = get_event_rules_scripts_dir()
    scripts_dir.mkdir(parents=True, exist_ok=True)
    readme = scripts_dir / "README.txt"
    if not readme.exists():
        readme.write_text(
            "Place Event Rule scripts here.\n"
            "Scripts receive JSON context via KNOXNET_EVENT_JSON env var and stdin.\n",
            encoding="utf-8",
        )
    return scripts_dir
