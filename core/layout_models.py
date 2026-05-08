from __future__ import annotations

from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Any, Dict, List, Literal, Optional, Union


def _utc_now_iso() -> str:
    return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


WidgetType = Literal["camera", "terminal", "overlay", "web", "unknown"]


@dataclass
class WidgetDefinition:
    id: str
    type: WidgetType
    x: int
    y: int
    w: int
    h: int
    title: str = ""
    pinned: bool = False

    # Workspace / virtual desktop index (X11 _NET_WM_DESKTOP)
    desktop: Optional[int] = None

    # Camera widget fields
    camera_id: Optional[str] = None
    view: Dict[str, Any] = field(default_factory=dict)  # view-only toggles/overrides

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "WidgetDefinition":
        raw_desktop = d.get("desktop")
        desktop_val = int(raw_desktop) if raw_desktop is not None else None
        return WidgetDefinition(
            id=str(d.get("id") or ""),
            type=str(d.get("type") or "unknown"),  # type: ignore[arg-type]
            x=int(d.get("x", 0)),
            y=int(d.get("y", 0)),
            w=int(d.get("w", 640)),
            h=int(d.get("h", 360)),
            title=str(d.get("title") or ""),
            pinned=bool(d.get("pinned", False)),
            desktop=desktop_val,
            camera_id=(str(d["camera_id"]) if d.get("camera_id") else None),
            view=dict(d.get("view") or {}),
        )


@dataclass
class LayoutDefinition:
    id: str
    name: str
    widgets: List[WidgetDefinition] = field(default_factory=list)
    created_at: str = field(default_factory=_utc_now_iso)
    updated_at: str = field(default_factory=_utc_now_iso)
    meta: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        out = asdict(self)
        out["widgets"] = [w.to_dict() for w in self.widgets]
        return out

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "LayoutDefinition":
        widgets = [WidgetDefinition.from_dict(x) for x in (d.get("widgets") or []) if isinstance(x, dict)]
        return LayoutDefinition(
            id=str(d.get("id") or ""),
            name=str(d.get("name") or d.get("id") or ""),
            widgets=widgets,
            created_at=str(d.get("created_at") or _utc_now_iso()),
            updated_at=str(d.get("updated_at") or _utc_now_iso()),
            meta=dict(d.get("meta") or {}),
        )


@dataclass
class CameraProfile:
    id: str
    name: str
    overlays: Dict[str, Any] = field(default_factory=dict)  # e.g. {"shapes":[...]} normalized
    ai_pipeline: Dict[str, Any] = field(default_factory=dict)
    monitoring_tools: Dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=_utc_now_iso)
    updated_at: str = field(default_factory=_utc_now_iso)
    meta: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "CameraProfile":
        return CameraProfile(
            id=str(d.get("id") or ""),
            name=str(d.get("name") or d.get("id") or ""),
            overlays=dict(d.get("overlays") or {}),
            ai_pipeline=dict(d.get("ai_pipeline") or {}),
            monitoring_tools=dict(d.get("monitoring_tools") or {}),
            created_at=str(d.get("created_at") or _utc_now_iso()),
            updated_at=str(d.get("updated_at") or _utc_now_iso()),
            meta=dict(d.get("meta") or {}),
        )


CameraProfileAssignmentValue = Union[str, List[str]]


