from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from typing import Optional, Literal

from core.model_library.depth_anything_v2 import (
    DepthAnythingSize,
    ensure_depth_anything_v2_weights,
)

logger = logging.getLogger(__name__)


@dataclass
class DepthAnythingRuntimeConfig:
    model_size: DepthAnythingSize = "vits"
    device: str = "auto"  # "cuda"|"cpu"|"auto" or "cuda:0"
    use_fp16: bool = True
    optimize: bool = True
    memory_fraction: Optional[float] = None


class _DepthAnythingRunnerHandle:
    def __init__(self, cfg: DepthAnythingRuntimeConfig):
        self.cfg = cfg
        self.lock = threading.Lock()
        self._runner = None
        self._init_error: Optional[BaseException] = None

    def get(self):
        # If initialization already failed once, don't spam reload attempts.
        if self._init_error is not None:
            raise self._init_error

        if self._runner is not None:
            return self._runner

        with self.lock:
            if self._init_error is not None:
                raise self._init_error
            if self._runner is not None:
                return self._runner

            try:
                ensure_depth_anything_v2_weights(model_size=self.cfg.model_size)

                from core.depth_anything_estimator import DepthAnythingEstimator

                device = self.cfg.device
                if device == "auto":
                    device = None

                logger.info(
                    "[ModelLibrary] Loading DepthAnythingV2 runner "
                    "(size=%s, device=%s, fp16=%s)",
                    self.cfg.model_size, device, self.cfg.use_fp16,
                )
                self._runner = DepthAnythingEstimator(
                    model_size=self.cfg.model_size,
                    device=device,
                    use_fp16=self.cfg.use_fp16,
                    optimize=self.cfg.optimize,
                    memory_fraction=self.cfg.memory_fraction,
                )

                actual = getattr(self._runner, "device", "unknown")
                if str(actual) == "cpu":
                    logger.warning(
                        "[ModelLibrary] DepthAnythingV2 is running on CPU. "
                        "Inference will be slow – a CUDA-capable GPU is strongly recommended."
                    )

                return self._runner
            except BaseException as exc:
                self._init_error = exc
                raise


class ModelRuntime:
    """
    Process-wide runtime cache for loaded models.
    Keeps heavyweight model instances shared across widgets.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._depth_anything_by_key: dict[str, _DepthAnythingRunnerHandle] = {}

    def get_depth_anything(self, cfg: DepthAnythingRuntimeConfig) -> _DepthAnythingRunnerHandle:
        key = f"{cfg.model_size}|{cfg.device}|{int(cfg.use_fp16)}|{int(cfg.optimize)}|{cfg.memory_fraction}"
        with self._lock:
            handle = self._depth_anything_by_key.get(key)
            if handle is None:
                handle = _DepthAnythingRunnerHandle(cfg)
                self._depth_anything_by_key[key] = handle
            return handle


_RUNTIME: Optional[ModelRuntime] = None


def get_model_runtime() -> ModelRuntime:
    global _RUNTIME
    if _RUNTIME is None:
        _RUNTIME = ModelRuntime()
    return _RUNTIME


