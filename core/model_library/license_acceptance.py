from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from core.model_library.byo_models import BYOModelManifest


def acceptance_store_path() -> Path:
    return Path("data") / "model_licenses_accepted.json"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _license_fingerprint(manifest: BYOModelManifest) -> str:
    lic = manifest.license
    raw = "|".join(
        [
            str(manifest.repo_id or ""),
            str(manifest.revision or ""),
            str(lic.spdx if lic else ""),
            str(lic.url if lic else ""),
            str(lic.text if lic else ""),
        ]
    )
    return hashlib.sha256(raw.encode("utf-8", errors="ignore")).hexdigest()


@dataclass(frozen=True)
class AcceptanceRecord:
    slug: str
    accepted_at: str
    license_fingerprint: str
    license_url: Optional[str] = None


def load_acceptance_store() -> dict:
    p = acceptance_store_path()
    if not p.exists():
        return {}
    try:
        obj = json.loads(p.read_text(encoding="utf-8"))
        return obj if isinstance(obj, dict) else {}
    except Exception:
        return {}


def save_acceptance_store(store: dict) -> None:
    p = acceptance_store_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(store, indent=2, sort_keys=True), encoding="utf-8")


def is_license_accepted(manifest: BYOModelManifest) -> bool:
    if not bool(getattr(manifest, "acceptance_required", True)):
        return True
    store = load_acceptance_store()
    rec = store.get(manifest.slug)
    if not isinstance(rec, dict):
        return False
    want = _license_fingerprint(manifest)
    return str(rec.get("license_fingerprint") or "") == str(want)


def record_license_acceptance(manifest: BYOModelManifest) -> AcceptanceRecord:
    store = load_acceptance_store()
    fp = _license_fingerprint(manifest)
    rec = {
        "slug": manifest.slug,
        "accepted_at": _now_iso(),
        "license_fingerprint": fp,
        "license_url": (manifest.license.url if manifest.license else None),
    }
    store[manifest.slug] = rec
    save_acceptance_store(store)
    return AcceptanceRecord(**rec)

