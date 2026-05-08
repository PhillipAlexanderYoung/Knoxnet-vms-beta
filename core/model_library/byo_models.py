from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional


def byo_models_dir() -> Path:
    from core.paths import get_models_dir

    return get_models_dir() / "byo"


def _slugify(text: str) -> str:
    s = (text or "").strip().lower()
    s = re.sub(r"[^a-z0-9._-]+", "-", s)
    s = re.sub(r"-{2,}", "-", s).strip("-")
    return s or "model"


@dataclass(frozen=True)
class ModelLicense:
    spdx: Optional[str] = None
    url: Optional[str] = None
    text: Optional[str] = None
    source: Optional[str] = None  # e.g. "huggingface" | "user"


@dataclass(frozen=True)
class BYOModelManifest:
    slug: str
    display_name: str
    backend: str  # "onnxruntime" (primary) or "ultralytics" plugin
    source: str  # "huggingface" | "local_file"
    model_path: str  # relative path inside model dir (e.g. "model.onnx")
    labels_path: Optional[str] = None  # relative path inside model dir
    input_size: int = 640
    revision: Optional[str] = None
    repo_id: Optional[str] = None
    license: Optional[ModelLicense] = None
    acceptance_required: bool = True

    def model_abs_path(self) -> Path:
        return byo_models_dir() / self.slug / self.model_path

    def labels_abs_path(self) -> Optional[Path]:
        if not self.labels_path:
            return None
        return byo_models_dir() / self.slug / self.labels_path


def manifest_path(slug: str) -> Path:
    return byo_models_dir() / str(slug) / "manifest.json"


def load_manifest(slug: str) -> Optional[BYOModelManifest]:
    p = manifest_path(slug)
    if not p.exists():
        return None
    data = json.loads(p.read_text(encoding="utf-8"))
    lic = None
    if isinstance(data.get("license"), dict):
        d = data["license"]
        lic = ModelLicense(
            spdx=d.get("spdx"),
            url=d.get("url"),
            text=d.get("text"),
            source=d.get("source"),
        )
    return BYOModelManifest(
        slug=str(data.get("slug") or slug),
        display_name=str(data.get("display_name") or data.get("slug") or slug),
        backend=str(data.get("backend") or "onnxruntime"),
        source=str(data.get("source") or "local_file"),
        model_path=str(data.get("model_path") or "model.onnx"),
        labels_path=data.get("labels_path"),
        input_size=int(data.get("input_size") or 640),
        revision=data.get("revision"),
        repo_id=data.get("repo_id"),
        license=lic,
        acceptance_required=bool(data.get("acceptance_required", True)),
    )


def list_installed_manifests() -> list[BYOModelManifest]:
    out: list[BYOModelManifest] = []
    root = byo_models_dir()
    if not root.exists():
        return out
    for child in root.iterdir():
        if not child.is_dir():
            continue
        mp = child / "manifest.json"
        if not mp.exists():
            continue
        try:
            m = load_manifest(child.name)
            if m is not None:
                out.append(m)
        except Exception:
            continue
    out.sort(key=lambda m: m.display_name.lower())
    return out


def list_local_onnx_models(*, include_byo: bool = False) -> list[Path]:
    """
    Return ONNX model files discovered under models/.

    By default, excludes models/byo/* because those are already represented via manifests.
    """
    from core.paths import get_models_dir

    root = get_models_dir()
    if not root.exists():
        return []

    out: list[Path] = []
    for p in root.rglob("*.onnx"):
        try:
            if not p.is_file():
                continue
            if not include_byo:
                rel_parts = [part.lower() for part in p.relative_to(root).parts]
                if rel_parts and rel_parts[0] == "byo":
                    continue
            out.append(p.resolve())
        except Exception:
            continue
    out.sort(key=lambda x: str(x).lower())
    return out


def read_labels_file(path: Path) -> Optional[list[str]]:
    """
    Supported label formats:
    - labels.json: ["person","car",...]
    - labels.txt: one label per line
    """
    if not path or not path.exists():
        return None
    try:
        if path.suffix.lower() == ".json":
            obj = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(obj, list):
                out = [str(x).strip() for x in obj if str(x).strip()]
                return out or None
        # txt fallback
        lines = [ln.strip() for ln in path.read_text(encoding="utf-8", errors="ignore").splitlines()]
        out = [ln for ln in lines if ln]
        return out or None
    except Exception:
        return None


def ensure_model_dir(slug: str) -> Path:
    d = byo_models_dir() / str(slug)
    d.mkdir(parents=True, exist_ok=True)
    return d


def save_manifest(manifest: BYOModelManifest) -> Path:
    d = ensure_model_dir(manifest.slug)
    p = d / "manifest.json"
    data: dict[str, Any] = {
        "slug": manifest.slug,
        "display_name": manifest.display_name,
        "backend": manifest.backend,
        "source": manifest.source,
        "model_path": manifest.model_path,
        "labels_path": manifest.labels_path,
        "input_size": int(manifest.input_size or 640),
        "revision": manifest.revision,
        "repo_id": manifest.repo_id,
        "acceptance_required": bool(manifest.acceptance_required),
    }
    if manifest.license:
        data["license"] = {
            "spdx": manifest.license.spdx,
            "url": manifest.license.url,
            "text": manifest.license.text,
            "source": manifest.license.source,
        }
    p.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")
    return p


def suggest_slug_from_hf(repo_id: str, filename: Optional[str] = None) -> str:
    base = str(repo_id or "hf-model").strip()
    if filename:
        base = f"{base}-{Path(filename).stem}"
    return _slugify(base)


def suggest_slug_from_file(path: str) -> str:
    return _slugify(Path(path).stem)

