from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from core.model_library.byo_models import (
    BYOModelManifest,
    ModelLicense,
    ensure_model_dir,
    save_manifest,
    suggest_slug_from_file,
    suggest_slug_from_hf,
)
from core.model_library.huggingface import HFFileRef, hf_download_file


@dataclass(frozen=True)
class InstallResult:
    slug: str
    model_path: Path
    manifest_path: Path


def _copy_file(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(str(src), str(dst))


def import_local_onnx(
    *,
    onnx_path: str,
    labels_path: Optional[str] = None,
    slug: Optional[str] = None,
    display_name: Optional[str] = None,
    license_spdx: Optional[str] = None,
    license_url: Optional[str] = None,
    license_text: Optional[str] = None,
) -> InstallResult:
    src = Path(str(onnx_path))
    if not src.exists():
        raise FileNotFoundError(f"ONNX file not found: {src}")
    if src.suffix.lower() != ".onnx":
        raise ValueError("Only .onnx models are supported for BYO import.")

    slug = str(slug or "").strip() or suggest_slug_from_file(str(src))
    display_name = str(display_name or "").strip() or src.stem

    d = ensure_model_dir(slug)
    dst_model = d / "model.onnx"
    _copy_file(src, dst_model)

    labels_rel = None
    if labels_path:
        lp = Path(str(labels_path))
        if not lp.exists():
            raise FileNotFoundError(f"Labels file not found: {lp}")
        dst_lbl = d / ("labels.json" if lp.suffix.lower() == ".json" else "labels.txt")
        _copy_file(lp, dst_lbl)
        labels_rel = dst_lbl.name

    lic = ModelLicense(spdx=license_spdx, url=license_url, text=license_text, source="user") if any([license_spdx, license_url, license_text]) else None
    man = BYOModelManifest(
        slug=slug,
        display_name=display_name,
        backend="onnxruntime",
        source="local_file",
        model_path="model.onnx",
        labels_path=labels_rel,
        input_size=640,
        license=lic,
        acceptance_required=True,
    )
    mp = save_manifest(man)
    return InstallResult(slug=slug, model_path=dst_model, manifest_path=mp)


def install_from_huggingface(
    *,
    repo_id: str,
    model_filename: str,
    labels_filename: Optional[str] = None,
    revision: Optional[str] = None,
    slug: Optional[str] = None,
    display_name: Optional[str] = None,
    license_spdx: Optional[str] = None,
    license_url: Optional[str] = None,
    license_text: Optional[str] = None,
) -> InstallResult:
    repo_id = str(repo_id or "").strip()
    model_filename = str(model_filename or "").strip()
    if not repo_id:
        raise ValueError("repo_id is required.")

    # Allow blank filename and auto-detect a likely ONNX artifact.
    if not model_filename:
        try:
            from huggingface_hub import list_repo_files  # type: ignore

            files = list_repo_files(repo_id=repo_id, revision=revision)
            onnx = [f for f in (files or []) if isinstance(f, str) and f.lower().endswith(".onnx")]
            # Prefer common names if present
            preferred = [
                "model.onnx",
                "yolov8n.onnx",
                "yolov8s.onnx",
                "yolov8m.onnx",
                "yolov8l.onnx",
                "yolov8x.onnx",
            ]
            chosen = None
            for p in preferred:
                for f in onnx:
                    if f.replace("\\", "/").lower().endswith(p):
                        chosen = f
                        break
                if chosen:
                    break
            model_filename = chosen or (onnx[0] if onnx else "")
        except Exception:
            model_filename = ""

    if not model_filename:
        raise ValueError("No .onnx file specified and auto-detect could not find one in the repo.")
    if not model_filename.lower().endswith(".onnx"):
        raise ValueError("Only .onnx artifacts are supported for BYO install.")

    # If labels filename not provided, try auto-detect common label files.
    if not labels_filename:
        try:
            from huggingface_hub import list_repo_files  # type: ignore

            files = list_repo_files(repo_id=repo_id, revision=revision)
            cand = []
            for f in (files or []):
                if not isinstance(f, str):
                    continue
                low = f.lower()
                if low.endswith("labels.json") or low.endswith("labels.txt") or low.endswith("classes.txt"):
                    cand.append(f)
            labels_filename = cand[0] if cand else None
        except Exception:
            labels_filename = None

    slug = str(slug or "").strip() or suggest_slug_from_hf(repo_id, model_filename)
    display_name = str(display_name or "").strip() or f"{repo_id}:{model_filename}"

    d = ensure_model_dir(slug)
    local_model = hf_download_file(
        HFFileRef(repo_id=repo_id, filename=model_filename, revision=revision),
        local_dir=d,
        force_download=False,
    )
    # normalize name
    dst_model = d / "model.onnx"
    if local_model.resolve() != dst_model.resolve():
        _copy_file(local_model, dst_model)

    labels_rel = None
    if labels_filename:
        local_lbl = hf_download_file(
            HFFileRef(repo_id=repo_id, filename=str(labels_filename), revision=revision),
            local_dir=d,
            force_download=False,
        )
        dst_lbl = d / ("labels.json" if local_lbl.suffix.lower() == ".json" else "labels.txt")
        if local_lbl.resolve() != dst_lbl.resolve():
            _copy_file(local_lbl, dst_lbl)
        labels_rel = dst_lbl.name

    lic = ModelLicense(spdx=license_spdx, url=license_url, text=license_text, source="huggingface" if any([license_spdx, license_url, license_text]) else None) if any([license_spdx, license_url, license_text]) else None
    man = BYOModelManifest(
        slug=slug,
        display_name=display_name,
        backend="onnxruntime",
        source="huggingface",
        model_path="model.onnx",
        labels_path=labels_rel,
        input_size=640,
        revision=revision,
        repo_id=repo_id,
        license=lic,
        acceptance_required=True,
    )
    mp = save_manifest(man)
    return InstallResult(slug=slug, model_path=dst_model, manifest_path=mp)

