"""
Capture-time object labeling for Motion Watch events.

Operators choose a labeling model in the desktop UI; this module resolves
hardware-aware defaults and runs detection on capture snapshots before ingest.
"""

from __future__ import annotations

import json
import logging
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Canonical model ids persisted in motion_watch_settings / desktop_prefs.
CAPTURE_LABEL_OFF = "off"
CAPTURE_LABEL_AUTO = "auto"
CAPTURE_LABEL_MOBILENET = "mobilenet"
CAPTURE_LABEL_YOLO = "yolo"
CAPTURE_LABEL_YOLO_NANO = "yolo-nano"
CAPTURE_LABEL_YOLO_SMALL = "yolo-small"
CAPTURE_LABEL_ULTRALYTICS = "ultralytics"

CAPTURE_LABEL_MODELS: Tuple[Tuple[str, str], ...] = (
    (CAPTURE_LABEL_OFF, "Off (index metadata only)"),
    (CAPTURE_LABEL_AUTO, "Auto (GPU YOLO if available, else MobileNetSSD)"),
    (CAPTURE_LABEL_MOBILENET, "MobileNetSSD (CPU-friendly)"),
    (CAPTURE_LABEL_YOLO_NANO, "YOLO nano (yolov8n)"),
    (CAPTURE_LABEL_YOLO_SMALL, "YOLO small (yolov8s)"),
    (CAPTURE_LABEL_YOLO, "YOLO (first available weights)"),
    (CAPTURE_LABEL_ULTRALYTICS, "Ultralytics YOLO (GPU if available)"),
)

_YOLO_WEIGHTS = {
    CAPTURE_LABEL_YOLO_NANO: "yolov8n.pt",
    CAPTURE_LABEL_YOLO_SMALL: "yolov8s.pt",
}

_detector_cache: Dict[str, Any] = {}
_detector_lock = threading.Lock()


def normalize_capture_label_model(value: Optional[str]) -> str:
    raw = str(value or CAPTURE_LABEL_AUTO).strip().lower()
    if raw in ("none", "false", "0", "disabled"):
        return CAPTURE_LABEL_OFF
    # Legacy boolean local_enrich=true maps to auto.
    if raw in ("true", "1", "yes"):
        return CAPTURE_LABEL_AUTO
    valid = {mid for mid, _ in CAPTURE_LABEL_MODELS}
    return raw if raw in valid else CAPTURE_LABEL_AUTO


def capture_labeling_enabled(model_id: Optional[str]) -> bool:
    return normalize_capture_label_model(model_id) != CAPTURE_LABEL_OFF


def probe_hardware() -> Dict[str, Any]:
    """Return hardware probe summary and recommended default model id."""
    from core.utils.detector_device import probe_capabilities

    caps = probe_capabilities()
    recommended = CAPTURE_LABEL_MOBILENET
    detail_parts: List[str] = []

    if caps.torch_cuda:
        recommended = CAPTURE_LABEL_YOLO_NANO
        detail_parts.append("CUDA available → YOLO nano recommended")
    elif caps.cv_cuda:
        recommended = CAPTURE_LABEL_YOLO_NANO
        detail_parts.append("OpenCV CUDA available → YOLO nano recommended")
    else:
        detail_parts.append("CPU-only → MobileNetSSD recommended")

    if caps.torch_cuda:
        detail_parts.append("PyTorch CUDA: yes")
    else:
        detail_parts.append("PyTorch CUDA: no")
    if caps.cv_cuda:
        detail_parts.append("OpenCV CUDA: yes")
    else:
        detail_parts.append("OpenCV CUDA: no")

    return {
        "recommended": recommended,
        "has_gpu": bool(caps.has_gpu),
        "torch_cuda": bool(caps.torch_cuda),
        "cv_cuda": bool(caps.cv_cuda),
        "detail": "; ".join(detail_parts),
    }


def resolve_effective_model(model_id: Optional[str]) -> str:
    """Resolve auto/legacy values to a concrete model id."""
    mid = normalize_capture_label_model(model_id)
    if mid == CAPTURE_LABEL_AUTO:
        return probe_hardware()["recommended"]
    return mid


def _resolve_model_path(model_id: str) -> Optional[str]:
    from core.paths import get_models_dir

    weights = _YOLO_WEIGHTS.get(model_id)
    if not weights:
        return None
    path = get_models_dir() / weights
    return str(path) if path.exists() else None


def _object_detector_config(model_id: str) -> Tuple[str, Optional[str]]:
    """Return (model_type, model_path) for ObjectDetector."""
    mid = resolve_effective_model(model_id)
    if mid == CAPTURE_LABEL_MOBILENET:
        return "mobilenet", None
    if mid in (CAPTURE_LABEL_YOLO_NANO, CAPTURE_LABEL_YOLO_SMALL):
        return "yolo", _resolve_model_path(mid)
    if mid in (CAPTURE_LABEL_YOLO, CAPTURE_LABEL_ULTRALYTICS, CAPTURE_LABEL_AUTO):
        return "yolo", None
    return "mobilenet", None


def get_cached_detector(model_id: str):
    """Return a cached ObjectDetector for the resolved model."""
    effective = resolve_effective_model(model_id)
    cache_key = f"{effective}|{_resolve_model_path(effective) or ''}"
    with _detector_lock:
        det = _detector_cache.get(cache_key)
        if det is not None:
            return det
    from core.object_detector import ObjectDetector

    model_type, model_path = _object_detector_config(model_id)
    try:
        det = ObjectDetector(model_type=model_type, model_path=model_path, device="auto")
    except Exception as exc:
        logger.warning("Capture label model %s failed (%s); falling back to MobileNetSSD", model_id, exc)
        det = ObjectDetector(model_type="mobilenet", device="auto")
        cache_key = "mobilenet|"
    with _detector_lock:
        _detector_cache[cache_key] = det
    return det


def detections_to_sidecar_fields(dets: List[Dict[str, Any]]) -> Tuple[List[str], List[str], List[Dict[str, Any]]]:
    """Convert raw detector output to detection_classes, tags, and sidecar detections list."""
    labels: List[str] = []
    detections_payload: List[Dict[str, Any]] = []
    for d in dets or []:
        if not isinstance(d, dict):
            continue
        lab = d.get("class") or d.get("label") or d.get("class_name")
        if not isinstance(lab, str) or not lab.strip():
            continue
        label = lab.strip().lower()
        labels.append(label)
        try:
            bb = d.get("bbox") or {}
            conf = float(d.get("confidence", 0.0) or 0.0)
            detections_payload.append(
                {
                    "class": label,
                    "confidence": conf,
                    "bbox": {
                        "x": float(bb.get("x", 0) or 0),
                        "y": float(bb.get("y", 0) or 0),
                        "w": float(bb.get("w", 0) or 0),
                        "h": float(bb.get("h", 0) or 0),
                    },
                }
            )
        except Exception:
            continue
    labels = list(dict.fromkeys([l for l in labels if l]))[:24]
    return labels, list(dict.fromkeys([*(labels or [])]))[:24], detections_payload[:50]


def run_capture_labeling(
    image_path: Path,
    model_id: Optional[str],
    *,
    conf_threshold: float = 0.25,
) -> Dict[str, Any]:
    """
    Run the selected detector on a capture image.
    Returns dict with detection_classes, tags, detections, metadata.capture_label.
    """
    import cv2

    mid = normalize_capture_label_model(model_id)
    if mid == CAPTURE_LABEL_OFF:
        return {}

    img = cv2.imread(str(image_path))
    if img is None:
        return {}

    effective = resolve_effective_model(mid)
    try:
        det = get_cached_detector(mid)
        dets = det.detect(img, conf_threshold=conf_threshold) or []
    except Exception as exc:
        logger.warning("Capture labeling failed for %s: %s", image_path, exc)
        return {"metadata": {"capture_label": {"model": mid, "effective": effective, "error": str(exc)}}}

    classes, tags, detections_payload = detections_to_sidecar_fields(dets)
    out: Dict[str, Any] = {
        "metadata": {
            "capture_label": {
                "model": mid,
                "effective": effective,
                "detector_type": getattr(det, "model_type", None),
                "count": len(detections_payload),
            }
        }
    }
    if classes:
        out["detection_classes"] = classes
    if tags:
        out["tags"] = tags
    if detections_payload:
        out["detections"] = detections_payload
    return out


def merge_shape_name_tags(sidecar: Dict[str, Any]) -> None:
    """Promote operator zone/line/tag names into searchable tags (in-place)."""
    from core.events_search import merge_operator_shape_tags

    trigger = sidecar.get("trigger") if isinstance(sidecar.get("trigger"), dict) else {}
    shape_name = trigger.get("shape_name") or sidecar.get("shape_name")
    tags = sidecar.get("tags") or []
    if not isinstance(tags, list):
        tags = [t.strip() for t in str(tags).split(",") if t.strip()]
    merged = merge_operator_shape_tags(
        [str(t).strip().lower() for t in tags if str(t).strip()],
        shape_name=str(shape_name) if isinstance(shape_name, str) else None,
    )
    if merged:
        sidecar["tags"] = merged
        sidecar.setdefault("operator_aliases", merged[:24])


def load_desktop_prefs() -> Dict[str, Any]:
    """Best-effort load of data/desktop_prefs.json."""
    try:
        from core.paths import get_data_dir

        prefs_path = get_data_dir() / "desktop_prefs.json"
        if prefs_path.exists():
            raw = json.loads(prefs_path.read_text(encoding="utf-8", errors="ignore"))
            if isinstance(raw, dict):
                return raw
    except Exception:
        pass
    return {}


def apply_enrichment_to_sidecar(
    sidecar: Dict[str, Any],
    enrich: Dict[str, Any],
    *,
    enable_detections: bool = True,
) -> None:
    """Merge labeling output into a Motion Watch sidecar payload (in-place)."""
    if not isinstance(enrich, dict) or not enrich:
        return
    for key in ("detection_classes", "tags", "detections"):
        if enrich.get(key) is not None:
            existing = sidecar.get(key)
            if key == "tags" and isinstance(existing, list) and isinstance(enrich.get(key), list):
                sidecar[key] = list(dict.fromkeys([*(existing or []), *(enrich[key] or [])]))[:32]
            else:
                sidecar[key] = enrich[key]
    meta = sidecar.setdefault("metadata", {})
    if isinstance(meta, dict) and isinstance(enrich.get("metadata"), dict):
        meta.update(enrich["metadata"])
    if enable_detections and (enrich.get("detections") or enrich.get("detection_classes")):
        sidecar["enable_detections"] = True


def run_capture_labeling_on_frame(
    frame: Any,
    model_id: Optional[str],
    *,
    conf_threshold: float = 0.25,
) -> Dict[str, Any]:
    """Run the selected detector on an in-memory BGR frame (for clips)."""
    mid = normalize_capture_label_model(model_id)
    if mid == CAPTURE_LABEL_OFF or frame is None:
        return {}

    effective = resolve_effective_model(mid)
    try:
        det = get_cached_detector(mid)
        dets = det.detect(frame, conf_threshold=conf_threshold) or []
    except Exception as exc:
        logger.warning("Capture labeling on frame failed: %s", exc)
        return {"metadata": {"capture_label": {"model": mid, "effective": effective, "error": str(exc)}}}

    classes, tags, detections_payload = detections_to_sidecar_fields(dets)
    out: Dict[str, Any] = {
        "metadata": {
            "capture_label": {
                "model": mid,
                "effective": effective,
                "detector_type": getattr(det, "model_type", None),
                "count": len(detections_payload),
                "source": "frame",
            }
        }
    }
    if classes:
        out["detection_classes"] = classes
    if tags:
        out["tags"] = tags
    if detections_payload:
        out["detections"] = detections_payload
    return out


def enrich_sidecar_for_index(
    sidecar: Dict[str, Any],
    *,
    image_path: Optional[Path] = None,
    frame_bgr: Any = None,
    per_camera_settings: Optional[Dict[str, Any]] = None,
) -> bool:
    """
    Label a capture sidecar before ingest. Returns True if labeling produced detections.
    Provide image_path (still) or frame_bgr (clip frame).
    """
    mid = resolve_capture_label_model(per_camera_settings, load_desktop_prefs())
    if not capture_labeling_enabled(mid):
        merge_shape_name_tags(sidecar)
        return False

    enrich: Dict[str, Any] = {}
    if frame_bgr is not None:
        enrich = run_capture_labeling_on_frame(frame_bgr, mid)
    elif image_path is not None:
        enrich = run_capture_labeling(Path(image_path), mid)

    apply_enrichment_to_sidecar(sidecar, enrich)
    sidecar.setdefault("metadata", {})
    if isinstance(sidecar["metadata"], dict):
        sidecar["metadata"]["capture_label_model"] = mid
    merge_shape_name_tags(sidecar)
    if enrich.get("detections") or enrich.get("detection_classes"):
        sidecar["enable_detections"] = True
    return bool(enrich.get("detections") or enrich.get("detection_classes"))


def resolve_capture_label_model(
    per_camera: Optional[Dict[str, Any]] = None,
    global_prefs: Optional[Dict[str, Any]] = None,
) -> str:
    """Per-camera override wins; then global desktop_prefs; default auto."""
    cam_val = (per_camera or {}).get("capture_label_model")
    if cam_val is not None and str(cam_val).strip():
        return normalize_capture_label_model(str(cam_val))
    # Legacy local_enrich boolean
    if (per_camera or {}).get("local_enrich") is True:
        return CAPTURE_LABEL_AUTO
    gp = global_prefs or {}
    events_cfg = gp.get("events_index") if isinstance(gp.get("events_index"), dict) else {}
    global_val = events_cfg.get("capture_label_model") or gp.get("capture_label_model")
    if global_val is not None and str(global_val).strip():
        return normalize_capture_label_model(str(global_val))
    return CAPTURE_LABEL_AUTO
