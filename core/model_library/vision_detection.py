from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from core.model_library.store import record_local_install
from core.paths import get_models_dir

logger = logging.getLogger(__name__)


def models_dir() -> Path:
    return get_models_dir()


def mobilenetssd_paths() -> dict[str, Path]:
    d = models_dir()
    # The codebase prefers deploy1.prototxt (300x300) when present.
    return {
        "caffemodel": d / "mobilenet_iter_73000.caffemodel",
        "prototxt": d / "deploy1.prototxt",
        "prototxt_fallback": d / "MobileNetSSD_deploy.prototxt",
    }


@dataclass(frozen=True)
class InstallStatus:
    ok: bool
    status: str  # "installed" | "missing" | "partial"
    detail: str = ""


def check_mobilenetssd() -> InstallStatus:
    p = mobilenetssd_paths()
    if p["caffemodel"].exists() and (p["prototxt"].exists() or p["prototxt_fallback"].exists()):
        return InstallStatus(ok=True, status="installed")
    if p["caffemodel"].exists() or p["prototxt"].exists() or p["prototxt_fallback"].exists():
        return InstallStatus(ok=False, status="partial", detail="Missing either .caffemodel or .prototxt")
    return InstallStatus(ok=False, status="missing")


def _download_to(url: str, dst: Path, *, timeout_s: float = 30.0) -> None:
    """
    Download URL to dst (atomic-ish).
    """
    import requests

    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(dst.suffix + ".part")
    try:
        with requests.get(url, stream=True, timeout=timeout_s) as r:
            r.raise_for_status()
            with open(tmp, "wb") as f:
                for chunk in r.iter_content(chunk_size=1024 * 256):
                    if chunk:
                        f.write(chunk)
        os.replace(str(tmp), str(dst))
    finally:
        try:
            if tmp.exists():
                tmp.unlink()
        except Exception:
            pass


def ensure_mobilenetssd(*, force_download: bool = False) -> dict[str, Path]:
    """
    Ensure MobileNetSSD Caffe artifacts exist in the writable models directory.

    This does NOT bundle any model files into the installer; it downloads on demand.
    Users can also install manually by placing the files into the models directory.
    """
    p = mobilenetssd_paths()
    st = check_mobilenetssd()
    if st.ok and not force_download:
        return p

    # Prefer repo-provided deploy1.prototxt if present (some environments ship it),
    # otherwise download the widely used MobileNetSSD_deploy.prototxt.
    prototxt_target = p["prototxt_fallback"]
    caffemodel_target = p["caffemodel"]

    # If user already has deploy1.prototxt, keep it.
    if not p["prototxt"].exists() and (force_download or not prototxt_target.exists()):
        prototxt_urls = [
            # OpenCV extra testdata (commonly used for CI/test assets)
            "https://raw.githubusercontent.com/opencv/opencv_extra/master/testdata/dnn/MobileNetSSD_deploy.prototxt",
            # Fallback mirrors
            "https://raw.githubusercontent.com/chuanqi305/MobileNet-SSD/master/MobileNetSSD_deploy.prototxt",
        ]
        last_err: Optional[BaseException] = None
        for url in prototxt_urls:
            try:
                _download_to(url, prototxt_target)
                break
            except BaseException as e:
                last_err = e
                continue
        if not prototxt_target.exists():
            raise RuntimeError(
                f"Failed to download MobileNetSSD prototxt.\n"
                f"Target: {prototxt_target}\n"
                f"Last error: {last_err!r}\n\n"
                f"Manual install: place MobileNetSSD_deploy.prototxt into:\n  {models_dir()}"
            )

    if force_download or not caffemodel_target.exists():
        caffemodel_urls = [
            "https://raw.githubusercontent.com/opencv/opencv_extra/master/testdata/dnn/MobileNetSSD_deploy.caffemodel",
            "https://raw.githubusercontent.com/chuanqi305/MobileNet-SSD/master/MobileNetSSD_deploy.caffemodel",
        ]
        last_err = None
        for url in caffemodel_urls:
            try:
                # Download under its common name, then normalize to our expected filename.
                common = models_dir() / "MobileNetSSD_deploy.caffemodel"
                _download_to(url, common, timeout_s=60.0)
                if common.resolve() != caffemodel_target.resolve():
                    try:
                        common.replace(caffemodel_target)
                    except Exception:
                        caffemodel_target.write_bytes(common.read_bytes())
                        common.unlink(missing_ok=True)
                break
            except BaseException as e:
                last_err = e
                continue
        if not caffemodel_target.exists():
            raise RuntimeError(
                f"Failed to download MobileNetSSD caffemodel.\n"
                f"Target: {caffemodel_target}\n"
                f"Last error: {last_err!r}\n\n"
                f"Manual install: place mobilenet_iter_73000.caffemodel into:\n  {models_dir()}"
            )

    # Record for UI tracking (best-effort)
    try:
        record_local_install(key="mobilenetssd", local_path=models_dir())
    except Exception:
        pass

    st2 = check_mobilenetssd()
    if not st2.ok:
        raise RuntimeError(f"MobileNetSSD install incomplete: {st2.status} ({st2.detail})")
    return p


def check_yolov8(_variant: str) -> InstallStatus:
    """
    Legacy helper retained for backwards-compat, but YOLO `.pt` weights are no longer
    managed/installed by default builds.
    """
    return InstallStatus(ok=False, status="missing", detail="Use tray: Models ▸ Install/Import ONNX…")


def ensure_yolov8_weights(_variant: str, *, force_download: bool = False) -> Path:
    """
    Deprecated: default distributions do NOT include Ultralytics nor YOLO `.pt` weight download.
    Use BYO ONNX models via the tray menu instead.
    """
    raise RuntimeError("Deprecated. Use tray: Models ▸ Install from Hugging Face… / Import local ONNX…")


