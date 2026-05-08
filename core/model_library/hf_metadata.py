from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional


@dataclass(frozen=True)
class HFLicenseInfo:
    license_spdx: Optional[str] = None
    license_url: Optional[str] = None
    raw: Optional[dict[str, Any]] = None


def try_get_hf_license(repo_id: str, revision: Optional[str] = None) -> HFLicenseInfo:
    """
    Best-effort: fetch license metadata from Hugging Face.

    Notes:
    - HF license information is not guaranteed to exist.
    - The exact fields vary; we probe a few common keys.
    """
    repo_id = str(repo_id or "").strip()
    if not repo_id:
        return HFLicenseInfo()
    try:
        from huggingface_hub import model_info  # type: ignore
    except Exception:
        return HFLicenseInfo()

    try:
        info = model_info(repo_id=repo_id, revision=revision)
        card = getattr(info, "cardData", None)
        raw = card if isinstance(card, dict) else None
        spdx = None
        url = None
        if isinstance(raw, dict):
            # Most common: "license": "apache-2.0" etc
            lic = raw.get("license") or raw.get("licenses")
            if isinstance(lic, str):
                spdx = lic
            elif isinstance(lic, list) and lic and isinstance(lic[0], str):
                spdx = str(lic[0])
            # Occasionally: "license_name" or "license_link"
            url = raw.get("license_link") or raw.get("license_url")
            if isinstance(url, list) and url:
                url = url[0]
            if not isinstance(url, str):
                url = None
        return HFLicenseInfo(license_spdx=spdx, license_url=url, raw=raw)
    except Exception:
        return HFLicenseInfo()

