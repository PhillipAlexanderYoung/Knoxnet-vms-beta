from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional, Sequence

import numpy as np

from core.detection.types import DetectionDict, LoadedModelInfo


class DetectorBackend(ABC):
    """
    A small interface to support multiple detector implementations (commercial-safe primary
    path: ONNXRuntime) while keeping optional "plugin" backends (Ultralytics) behind guards.
    """

    backend_id: str

    @abstractmethod
    def load(
        self,
        *,
        model_path: str,
        labels: Optional[Sequence[str]] = None,
        device: str = "cpu",
        input_size: int = 640,
    ) -> LoadedModelInfo:
        raise NotImplementedError

    @abstractmethod
    def detect(
        self,
        frame_bgr: np.ndarray,
        *,
        min_confidence: float = 0.25,
        max_det: int = 100,
        allowed_classes: Optional[Sequence[str]] = None,
    ) -> list[DetectionDict]:
        raise NotImplementedError

    @abstractmethod
    def get_class_names(self) -> Optional[list[str]]:
        raise NotImplementedError

    @abstractmethod
    def close(self) -> None:
        raise NotImplementedError

