from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from PIL import Image


@dataclass
class RunContext:
    detections: List[str] = field(default_factory=list)
    prompt: Optional[str] = None
    timeout_s: Optional[float] = None
    device: str = "cpu"
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DescribeOutput:
    caption: str
    tags: List[str]
    raw: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MatchOutput:
    verdict: Dict[str, bool]
    reasoning: str
    raw: Dict[str, Any] = field(default_factory=dict)


class VisionRunner(ABC):
    name: str

    def __init__(self, device: str = "cpu") -> None:
        self.device = device

    @abstractmethod
    def describe(self, image: Image.Image, context: RunContext) -> DescribeOutput:
        ...

    @abstractmethod
    def match_criteria(
        self,
        image: Image.Image,
        criteria: List[str],
        context: RunContext,
    ) -> MatchOutput:
        ...

