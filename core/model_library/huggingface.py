from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional


@dataclass(frozen=True)
class HFFileRef:
    repo_id: str
    filename: str
    revision: Optional[str] = None


def hf_download_file(
    ref: HFFileRef,
    *,
    local_dir: Path,
    force_download: bool = False,
) -> Path:
    """
    Download a single artifact from Hugging Face Hub into `local_dir`.

    Returns the local path to the downloaded file.
    """
    # Lazy import so desktop can start even if extras aren't installed in some environments.
    from huggingface_hub import hf_hub_download

    local_dir = Path(local_dir)
    local_dir.mkdir(parents=True, exist_ok=True)

    out = hf_hub_download(
        repo_id=ref.repo_id,
        filename=ref.filename,
        revision=ref.revision,
        local_dir=str(local_dir),
        local_dir_use_symlinks=False,
        force_download=force_download,
    )
    return Path(out)


def hf_snapshot_download(
    *,
    repo_id: str,
    local_dir: Path,
    revision: Optional[str] = None,
    token: Optional[str] = None,
    allow_patterns: Optional[Iterable[str]] = None,
    force_download: bool = False,
) -> Path:
    """
    Download an entire repository snapshot (or subset) from Hugging Face Hub into `local_dir`.

    Returns the local path to the snapshot directory.
    """
    # Lazy import so desktop can start even if extras aren't installed in some environments.
    from huggingface_hub import snapshot_download

    local_dir = Path(local_dir)
    local_dir.mkdir(parents=True, exist_ok=True)

    out = snapshot_download(
        repo_id=repo_id,
        revision=revision,
        local_dir=str(local_dir),
        local_dir_use_symlinks=False,
        resume_download=True,
        force_download=force_download,
        allow_patterns=list(allow_patterns) if allow_patterns else None,
        token=token,
    )
    return Path(out)

