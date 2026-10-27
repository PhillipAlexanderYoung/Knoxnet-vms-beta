"""Device resolution helpers for detector backends."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from functools import lru_cache
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DeviceCapabilities:
    """Captured capability probes for the runtimes we depend on."""

    cv_cuda: bool
    torch_cuda: bool
    reason: str = ""

    @property
    def has_gpu(self) -> bool:
        return bool(self.cv_cuda or self.torch_cuda)


@dataclass(frozen=True)
class DetectorDeviceConfig:
    """Resolved device configuration for detectors."""

    preference: str
    kind: str  # "cpu" or "gpu"
    torch_device: str
    opencv_backend: int
    opencv_target: int
    capabilities: DeviceCapabilities

    def summary(self) -> str:
        """Human readable representation for logging."""

        cap = self.capabilities
        return (
            f"preference={self.preference} resolved={self.kind} "
            f"(torch_cuda={cap.torch_cuda}, cv_cuda={cap.cv_cuda})"
        )


def _safe_import_cv2():
    try:
        import cv2  # type: ignore

        return cv2
    except Exception as exc:  # pragma: no cover - only hits when cv2 missing
        logger.debug("OpenCV unavailable: %s", exc)
        return None


def _safe_import_torch():
    try:
        import torch  # type: ignore

        return torch
    except Exception as exc:  # pragma: no cover - only hits when torch missing
        logger.debug("PyTorch unavailable: %s", exc)
        return None


@lru_cache(maxsize=1)
def probe_capabilities() -> DeviceCapabilities:
    """Detect GPU support for both OpenCV DNN and PyTorch."""

    cv2 = _safe_import_cv2()
    torch = _safe_import_torch()

    cv_cuda = False
    torch_cuda = False
    reasons = []

    if cv2 is not None:
        try:
            cv_cuda = bool(getattr(cv2.cuda, "getCudaEnabledDeviceCount", lambda: 0)() > 0)
        except Exception as exc:  # pragma: no cover - defensive
            reasons.append(f"cv_cuda probe failed: {exc}")
    else:
        reasons.append("cv2 not importable")

    if torch is not None:
        try:
            torch_cuda = bool(torch.cuda.is_available())
        except Exception as exc:  # pragma: no cover - defensive
            reasons.append(f"torch cuda probe failed: {exc}")
    else:
        reasons.append("torch not importable")

    return DeviceCapabilities(cv_cuda=cv_cuda, torch_cuda=torch_cuda, reason="; ".join(reasons))


def _normalise_preference(preference: Optional[str]) -> str:
    pref = (preference or os.getenv("DETECTOR_DEVICE", "auto")).strip().lower()
    if pref not in {"auto", "gpu", "cpu"}:
        logger.warning("Unknown detector device preference '%s', falling back to auto", pref)
        return "auto"
    return pref


def resolve_detector_device(preference: Optional[str] = None) -> DetectorDeviceConfig:
    """Resolve the concrete device selection for detectors.

    Returns a configuration that downstream components can consume.
    """

    pref = _normalise_preference(preference)
    caps = probe_capabilities()

    cv2 = _safe_import_cv2()
    backend_cpu = getattr(cv2.dnn, "DNN_BACKEND_OPENCV", 0) if cv2 else 0
    target_cpu = getattr(cv2.dnn, "DNN_TARGET_CPU", 0) if cv2 else 0
    backend_cuda = getattr(cv2.dnn, "DNN_BACKEND_CUDA", backend_cpu) if cv2 else backend_cpu
    target_cuda = getattr(cv2.dnn, "DNN_TARGET_CUDA", target_cpu) if cv2 else target_cpu
    target_cuda_fp16 = getattr(cv2.dnn, "DNN_TARGET_CUDA_FP16", target_cuda)

    chosen_kind = "cpu"
    torch_device = "cpu"
    backend = backend_cpu
    target = target_cpu

    def _enable_gpu():
        nonlocal chosen_kind, torch_device, backend, target
        chosen_kind = "gpu"
        torch_device = "cuda"
        backend = backend_cuda
        target = target_cuda_fp16 if target_cuda_fp16 != target_cpu else target_cuda

    if pref == "gpu" and caps.has_gpu:
        _enable_gpu()
    elif pref == "gpu":
        logger.warning("GPU requested but not available. Falling back to CPU")
    elif pref == "auto" and caps.has_gpu:
        _enable_gpu()

    if chosen_kind == "gpu" and not caps.cv_cuda:
        # If CUDA backend missing for OpenCV we still keep GPU for torch but stick to CPU backend for cv2.
        backend = backend_cpu
        target = target_cpu
        if not caps.cv_cuda:
            logger.debug("Using GPU for PyTorch but OpenCV DNN lacks CUDA support; keeping CPU backend")

    if chosen_kind == "gpu" and not caps.torch_cuda:
        torch_device = "cpu"
        logger.info("GPU selected but PyTorch CUDA unavailable; YOLO will stay on CPU")

    config = DetectorDeviceConfig(
        preference=pref,
        kind=chosen_kind,
        torch_device=torch_device,
        opencv_backend=backend,
        opencv_target=target,
        capabilities=caps,
    )

    logger.debug("Detector device resolved: %s", config.summary())
    return config


def detector_runs_on_gpu(config: DetectorDeviceConfig) -> bool:
    """Convenience predicate for callers."""

    return config.kind == "gpu" and config.torch_device == "cuda"


