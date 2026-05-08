from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

from core.model_library.store import record_hf_snapshot_install
from core.paths import get_data_dir, get_models_dir


@dataclass(frozen=True)
class ModelCatalogEntry:
    id: str
    display_name: str
    modality: str  # e.g. "image_to_text"
    repo_id: str
    revision: Optional[str]
    local_dir: Path
    required_files: tuple[str, ...]
    license_url: str
    size_gb: Optional[float] = None
    requires_gpu: bool = False
    min_vram_gb: Optional[float] = None
    notes: Optional[str] = None

    def is_installed(self) -> bool:
        if not self.local_dir.exists():
            return False
        for name in self.required_files:
            if not (self.local_dir / name).exists():
                return False
        return True

    def missing_files(self) -> list[str]:
        if not self.local_dir.exists():
            return list(self.required_files)
        missing = []
        for name in self.required_files:
            if not (self.local_dir / name).exists():
                missing.append(name)
        return missing


def _models_root() -> Path:
    return get_models_dir() / "vision"


def vision_catalog() -> list[ModelCatalogEntry]:
    root = _models_root()
    return [
        ModelCatalogEntry(
            id="blip2",
            display_name="BLIP-2 OPT-6.7B",
            modality="image_to_text",
            repo_id="Salesforce/blip2-opt-6.7b",
            revision=None,
            local_dir=root / "blip2",
            required_files=("pytorch_model.bin", "config.json", "spiece.model"),
            license_url="https://github.com/salesforce/LAVIS/blob/main/LICENSE",
            size_gb=13.0,
            requires_gpu=True,
            min_vram_gb=16.0,
        ),
        ModelCatalogEntry(
            id="llava",
            display_name="LLaVA v1.5 13B",
            modality="image_to_text",
            repo_id="liuhaotian/llava-v1.5-13b",
            revision=None,
            local_dir=root / "llava",
            required_files=(
                "llava-v1.5-13b.safetensors",
                "config.json",
                "tokenizer.model",
                "mm_projector.bin",
                "preprocessor_config.json",
            ),
            license_url="https://github.com/haotian-liu/LLaVA/blob/main/LICENSE",
            size_gb=26.0,
            requires_gpu=True,
            min_vram_gb=20.0,
        ),
        ModelCatalogEntry(
            id="minicpm",
            display_name="MiniCPM-V 2.6",
            modality="image_to_text",
            repo_id="openbmb/MiniCPM-V-2_6",
            revision=None,
            local_dir=root / "minicpm",
            required_files=(
                "minicpm-v-2_6.bin",
                "config.json",
                "tokenizer.model",
                "vision_config.json",
                "mm_projector.bin",
            ),
            license_url="https://github.com/OpenBMB/MiniCPM-V/blob/main/LICENSE",
            size_gb=8.5,
            requires_gpu=False,
            min_vram_gb=8.0,
        ),
    ]


def get_catalog_entry(model_id: str) -> Optional[ModelCatalogEntry]:
    for entry in vision_catalog():
        if entry.id == model_id:
            return entry
    return None


def list_catalog_entries(modality: Optional[str] = None) -> list[ModelCatalogEntry]:
    entries = vision_catalog()
    if modality:
        entries = [e for e in entries if e.modality == modality]
    return entries


def resolve_vision_model_slug(model_id: str) -> str:
    """
    Map catalog IDs to local vision service model slugs.
    """
    entry = get_catalog_entry(model_id)
    return entry.id if entry else model_id


def _read_hf_token_from_user_keys() -> Optional[str]:
    """
    Best-effort token lookup from data/llm_user_keys.json for HF downloads.
    """
    p = get_data_dir() / "llm_user_keys.json"
    if not p.exists():
        return None
    try:
        import json

        data = json.loads(p.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            hf = data.get("huggingface") or data.get("hf") or {}
            if isinstance(hf, dict):
                token = hf.get("token") or hf.get("api_key")
                if isinstance(token, str) and token.strip():
                    return token.strip()
    except Exception:
        return None
    return None


def _resolve_hf_token(token: Optional[str]) -> Optional[str]:
    if token and token.strip():
        return token.strip()
    env_token = os.environ.get("HF_TOKEN")
    if env_token and env_token.strip():
        return env_token.strip()
    return _read_hf_token_from_user_keys()


def ensure_model_installed(
    model_id: str,
    *,
    force_download: bool = False,
    token: Optional[str] = None,
) -> Path:
    entry = get_catalog_entry(model_id)
    if entry is None:
        raise KeyError(f"Unknown model catalog entry: {model_id}")

    missing = entry.missing_files()
    if not missing and not force_download:
        return entry.local_dir

    resolved_token = _resolve_hf_token(token)
    allow_patterns: Iterable[str] = entry.required_files

    entry.local_dir.mkdir(parents=True, exist_ok=True)
    try:
        from core.model_library import huggingface as hf
    except Exception as exc:
        raise RuntimeError("Hugging Face client not available. Install requirements.txt.") from exc

    hf.hf_snapshot_download(
        repo_id=entry.repo_id,
        revision=entry.revision,
        local_dir=entry.local_dir,
        allow_patterns=allow_patterns,
        token=resolved_token,
        force_download=force_download,
    )

    missing = entry.missing_files()
    if missing:
        missing_str = ", ".join(missing)
        raise FileNotFoundError(f"Model '{entry.id}' missing required files after download: {missing_str}")

    try:
        record_hf_snapshot_install(
            key=entry.id,
            repo_id=entry.repo_id,
            local_path=entry.local_dir,
            revision=entry.revision,
            files=list(entry.required_files),
        )
    except Exception:
        pass

    return entry.local_dir
