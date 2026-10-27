from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from core.sort_tracker import SortTracker as CoreSortTracker

from .base import BaseTracker, Detection, Track, BBox


@dataclass(frozen=True)
class SortTrackerConfig:
    max_age: int = 15
    min_hits: int = 2
    iou_threshold: float = 0.3


class SortObjectTracker(BaseTracker):
    def __init__(self, cfg: Optional[SortTrackerConfig] = None):
        self.cfg = cfg or SortTrackerConfig()
        self._tracker = CoreSortTracker(
            max_age=int(self.cfg.max_age),
            min_hits=int(self.cfg.min_hits),
            iou_threshold=float(self.cfg.iou_threshold),
        )

    def reset(self) -> None:
        self._tracker.reset()

    def update(self, detections: List[Detection], frame_ts: Optional[float] = None) -> List[Track]:
        sort_in = [d.to_sort_dict() for d in (detections or [])]
        out = self._tracker.update(sort_in)
        tracks: List[Track] = []
        for t in out or []:
            bbox = (t or {}).get("bbox") or {}
            try:
                bb = BBox(
                    x=float(bbox.get("x", 0.0)),
                    y=float(bbox.get("y", 0.0)),
                    w=float(bbox.get("w", 0.0)),
                    h=float(bbox.get("h", 0.0)),
                )
            except Exception:
                continue
            try:
                tid = int(t.get("id"))
            except Exception:
                continue
            tracks.append(
                Track(
                    track_id=tid,
                    bbox=bb,
                    cls=str(t.get("class") or "object"),
                    confidence=float(t.get("confidence") or 0.0),
                    age=int(t.get("age") or 0),
                    lost_frames=int(t.get("lost_frames") or 0),
                    predicted=bool(t.get("predicted") or False),
                )
            )
        return tracks


