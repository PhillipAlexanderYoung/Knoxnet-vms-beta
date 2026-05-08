from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional
from datetime import datetime


@dataclass
class EventDetection:
    class_name: str
    confidence: float
    bbox: Dict[str, float]  # x, y, w, h in pixels


@dataclass
class EventTrack:
    id: int
    bbox: Dict[str, float]  # x, y, w, h in pixels
    history: List[Dict[str, float]]  # list of {x, y} in pixels


@dataclass
class EventOverlay:
    zones: List[List[Dict[str, float]]]  # list of polygons (points: {x,y} pixels)
    lines: List[Dict[str, Dict[str, float]]]  # list of {p1:{x,y}, p2:{x,y}}
    tags: List[Dict[str, float]]  # list of {x,y}


@dataclass
class EventBundle:
    id: str
    camera_id: str
    kind: str
    created_at: str
    detections: List[EventDetection]
    tracks: List[EventTrack]
    overlays: Optional[EventOverlay]
    snapshot_base64: Optional[str]
    metadata: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def build_event_bundle(bundle_id: str,
                       camera_id: str,
                       kind: str,
                       detections: List[EventDetection],
                       tracks: List[EventTrack],
                       overlays: Optional[EventOverlay],
                       snapshot_base64: Optional[str],
                       metadata: Optional[Dict[str, Any]] = None) -> EventBundle:
    return EventBundle(
        id=bundle_id,
        camera_id=camera_id,
        kind=kind,
        created_at=datetime.now().isoformat(),
        detections=detections,
        tracks=tracks,
        overlays=overlays,
        snapshot_base64=snapshot_base64,
        metadata=metadata or {}
    )


