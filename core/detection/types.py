from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, TypedDict


class BBoxDict(TypedDict):
    x: float
    y: float
    w: float
    h: float


class DetectionDict(TypedDict, total=False):
    """
    Unified detection format used across Desktop overlays and backend.

    Required by Desktop overlay code:
      - bbox: {x,y,w,h} in pixel coords (top-left origin)
      - class: label/name string
      - confidence: float [0..1]
    """

    bbox: BBoxDict
    class_: str  # NOTE: key is "class" at runtime; TypedDict uses class_ to avoid keyword
    confidence: float

    # Optional metadata
    class_id: int
    backend: str
    model: str
    raw: Any


@dataclass(frozen=True)
class BackendInfo:
    id: str
    display_name: str
    available: bool
    detail: str = ""


@dataclass(frozen=True)
class LoadedModelInfo:
    backend_id: str
    model_id: str
    model_path: str
    device: str
    input_size: int
    class_names: Optional[list[str]] = None

