from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Protocol, Iterable, Dict, Any, Tuple, List


@dataclass(frozen=True)
class BBox:
    """Bounding box in source-frame pixel coordinates."""

    x: float
    y: float
    w: float
    h: float

    def as_xywh(self) -> Tuple[float, float, float, float]:
        return (float(self.x), float(self.y), float(self.w), float(self.h))


@dataclass(frozen=True)
class Detection:
    bbox: BBox
    cls: str
    confidence: float
    timestamp: Optional[float] = None

    def to_sort_dict(self) -> Dict[str, Any]:
        return {
            "bbox": {"x": float(self.bbox.x), "y": float(self.bbox.y), "w": float(self.bbox.w), "h": float(self.bbox.h)},
            "class": str(self.cls),
            "confidence": float(self.confidence),
        }


@dataclass(frozen=True)
class Track:
    track_id: int
    bbox: BBox
    cls: str
    confidence: float
    age: int = 0
    lost_frames: int = 0
    predicted: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "track_id": int(self.track_id),
            "bbox": {"x": float(self.bbox.x), "y": float(self.bbox.y), "w": float(self.bbox.w), "h": float(self.bbox.h)},
            "class": str(self.cls),
            "confidence": float(self.confidence),
            "age": int(self.age),
            "lost_frames": int(self.lost_frames),
            "predicted": bool(self.predicted),
        }


class BaseTracker(Protocol):
    def update(self, detections: List[Detection], frame_ts: Optional[float] = None) -> List[Track]:
        ...

    def reset(self) -> None:
        ...


def coerce_bbox(obj: object) -> Optional[BBox]:
    """
    Accept bbox formats:
    - dict: {x,y,w,h}
    - list/tuple: [x,y,w,h]
    """
    if obj is None:
        return None
    try:
        if isinstance(obj, dict):
            return BBox(
                x=float(obj.get("x", 0.0)),
                y=float(obj.get("y", 0.0)),
                w=float(obj.get("w", 0.0)),
                h=float(obj.get("h", 0.0)),
            )
        if isinstance(obj, (list, tuple)) and len(obj) >= 4:
            return BBox(x=float(obj[0]), y=float(obj[1]), w=float(obj[2]), h=float(obj[3]))
    except Exception:
        return None
    return None


def detections_from_dicts(dets: Iterable[dict], *, ts: Optional[float] = None) -> List[Detection]:
    out: List[Detection] = []
    for d in dets or []:
        if not isinstance(d, dict):
            continue
        bbox = coerce_bbox(d.get("bbox"))
        if bbox is None:
            continue
        cls = d.get("class")
        conf = d.get("confidence")
        try:
            c = str(cls if cls is not None else "object")
        except Exception:
            c = "object"
        try:
            cf = float(conf if conf is not None else 0.0)
        except Exception:
            cf = 0.0
        out.append(Detection(bbox=bbox, cls=c, confidence=cf, timestamp=ts))
    return out


