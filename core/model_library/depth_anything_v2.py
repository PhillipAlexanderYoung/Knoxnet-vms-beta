from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Literal

from core.model_library.huggingface import HFFileRef, hf_download_file
from core.model_library.store import record_hf_install
from core.paths import get_models_dir

logger = logging.getLogger(__name__)

DepthAnythingSize = Literal["vits", "vitb", "vitl"]


@dataclass(frozen=True)
class DepthAnythingWeightsSpec:
    model_size: DepthAnythingSize
    repo_id: str
    filename: str


_DEPTH_ANYTHING_V2_SPECS: dict[DepthAnythingSize, DepthAnythingWeightsSpec] = {
    "vits": DepthAnythingWeightsSpec(
        model_size="vits",
        repo_id="depth-anything/Depth-Anything-V2-Small",
        filename="depth_anything_v2_vits.pth",
    ),
    "vitb": DepthAnythingWeightsSpec(
        model_size="vitb",
        repo_id="depth-anything/Depth-Anything-V2-Base",
        filename="depth_anything_v2_vitb.pth",
    ),
    "vitl": DepthAnythingWeightsSpec(
        model_size="vitl",
        repo_id="depth-anything/Depth-Anything-V2-Large",
        filename="depth_anything_v2_vitl.pth",
    ),
}


def depth_anything_weights_path(*, repo_root: Optional[Path] = None, model_size: DepthAnythingSize) -> Path:
    """
    Path expected by `core.depth_anything_estimator.DepthAnythingEstimator`.
    """
    if repo_root is None:
        return get_models_dir() / "depth_anything" / f"depth_anything_v2_{model_size}.pth"
    return Path(repo_root) / "models" / "depth_anything" / f"depth_anything_v2_{model_size}.pth"


def ensure_depth_anything_v2_weights(
    *,
    model_size: DepthAnythingSize,
    repo_root: Optional[Path] = None,
    revision: Optional[str] = None,
    force_download: bool = False,
) -> Path:
    """
    Ensure DepthAnythingV2 weights exist locally (downloads from HF if missing).
    """
    if model_size not in _DEPTH_ANYTHING_V2_SPECS:
        raise ValueError(f"Unsupported DepthAnythingV2 model_size: {model_size}")

    target = depth_anything_weights_path(repo_root=repo_root, model_size=model_size)
    if target.exists() and not force_download:
        return target

    spec = _DEPTH_ANYTHING_V2_SPECS[model_size]
    local_dir = target.parent
    logger.info(f"[ModelLibrary] Downloading DepthAnythingV2 weights: {spec.repo_id}/{spec.filename} -> {local_dir}")

    downloaded = hf_download_file(
        HFFileRef(repo_id=spec.repo_id, filename=spec.filename, revision=revision),
        local_dir=local_dir,
        force_download=force_download,
    )

    # Normalize the filename to what our estimator expects.
    if downloaded.name != target.name:
        try:
            downloaded.replace(target)
        except Exception:
            # If replace fails (cross-device), fall back to copy via rename semantics.
            target.write_bytes(downloaded.read_bytes())

    # Record install for UI/library tooling
    try:
        record_hf_install(
            key=f"depth_anything_v2_{model_size}",
            repo_id=spec.repo_id,
            filename=spec.filename,
            local_path=target,
            revision=revision,
        )
    except Exception:
        pass
    return target


