from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List

try:
    import torch  # type: ignore
    TORCH_AVAILABLE = True
except ImportError:
    torch = None  # type: ignore
    TORCH_AVAILABLE = False


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if getattr(sys, "frozen", False):
    # In frozen builds (PyInstaller/AppImage), the bundled app directory is read-only.
    # Use per-user XDG cache/state locations for any runtime model/cached data.
    xdg_cache = (os.environ.get("XDG_CACHE_HOME") or "").strip()
    cache_base = Path(xdg_cache).expanduser() if xdg_cache else (Path.home() / ".cache")
    MODELS_ROOT = (cache_base / "KnoxnetVMS" / "models" / "vision").resolve()

    xdg_state = (os.environ.get("XDG_STATE_HOME") or "").strip()
    state_base = Path(xdg_state).expanduser() if xdg_state else (Path.home() / ".local" / "state")
    DEFAULT_RESULTS_DIR = (state_base / "KnoxnetVMS" / "benchmarks" / "results").resolve()
else:
    MODELS_ROOT = PROJECT_ROOT / "models" / "vision"
    DEFAULT_RESULTS_DIR = PROJECT_ROOT / "benchmarks" / "results"


def _split_csv(value: str) -> List[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


@dataclass
class ServiceSettings:
    host: str = os.getenv("LOCAL_VISION_HOST", "0.0.0.0")
    port: int = int(os.getenv("LOCAL_VISION_PORT", "8101"))
    default_model: str = os.getenv("LOCAL_VISION_MODEL", "blip2")
    default_composition: List[str] = field(
        default_factory=lambda: _split_csv(os.getenv("LOCAL_VISION_COMPOSITION", "blip2,llava,minicpm"))
    )
    device: str = os.getenv(
        "LOCAL_VISION_DEVICE",
        "cuda" if TORCH_AVAILABLE and torch.cuda.is_available() else "cpu",  # type: ignore[arg-type]
    )
    cache_dir: Path = Path(os.getenv("LOCAL_VISION_CACHE", MODELS_ROOT / "cache"))
    detector_model: Path = Path(os.getenv("LOCAL_VISION_DETECTOR_MODEL", PROJECT_ROOT / "yolov8n.pt"))
    enable_detector: bool = os.getenv("LOCAL_VISION_ENABLE_DETECTOR", "1").lower() not in {"0", "false", "no"}
    prometheus_namespace: str = os.getenv("LOCAL_VISION_PROM_NAMESPACE", "vision_local")
    metrics_bucket_ms: List[float] = field(
        default_factory=lambda: [50, 150, 300, 600, 1200, 2400, 4800]
    )
    max_new_tokens: int = int(os.getenv("LOCAL_VISION_MAX_TOKENS", "128"))
    timeout_s: float = float(os.getenv("LOCAL_VISION_TIMEOUT_S", "45"))
    batch_size: int = int(os.getenv("LOCAL_VISION_BATCH_SIZE", "1"))
    llava_endpoint: str = os.getenv("LOCAL_VISION_LLAVA_ENDPOINT", "")
    minicpm_endpoint: str = os.getenv("LOCAL_VISION_MINICPM_ENDPOINT", "")

    @property
    def model_paths(self) -> Dict[str, Path]:
        return {
            "blip2": Path(os.getenv("LOCAL_MODEL_PATH_BLIP2", MODELS_ROOT / "blip2")),
            "llava": Path(os.getenv("LOCAL_MODEL_PATH_LLAVA", MODELS_ROOT / "llava")),
            "minicpm": Path(os.getenv("LOCAL_MODEL_PATH_MINICPM", MODELS_ROOT / "minicpm")),
        }


def get_settings() -> ServiceSettings:
    return ServiceSettings()

