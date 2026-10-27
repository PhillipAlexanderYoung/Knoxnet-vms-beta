from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Literal, Iterable, Any

import numpy as np
from PySide6.QtCore import QThread, Signal

from core.detection.registry import create_backend, normalize_allowed_classes, resolve_device_for_backend
from core.model_library.byo_models import load_manifest, read_labels_file, list_local_onnx_models
from core.paths import get_models_dir


DetectorDevice = Literal["auto", "cpu", "cuda", "gpu"]
TrackerType = Literal["sort", "bytetrack"]
DetectorBackendId = Literal["auto", "onnxruntime", "mobilenetssd", "ultralytics"]


@dataclass
class DetectorConfig:
    enabled: bool = False
    fps_limit: int = 8
    # Preferred detector backend (primary = onnxruntime). Ultralytics is optional plugin.
    backend: DetectorBackendId = "mobilenetssd"
    # Model identifier:
    # - BYO manifest slug under models/byo/<slug>/manifest.json
    # - or a direct local path to a .onnx model (advanced)
    # (legacy: yolov8n/yolov8s etc are treated as slugs; no longer auto-downloaded)
    model_variant: str = "default"
    device: DetectorDevice = "auto"
    imgsz: int = 640
    min_confidence: float = 0.35
    max_det: int = 100
    # Optional class allowlist by label (e.g. ["person","car"]). If empty/None: allow all.
    allowed_classes: Optional[list[str]] = None

    # Desktop-local tracking (applies to Desktop YOLO detections only)
    tracking_enabled: bool = False
    tracker_type: TrackerType = "sort"
    # Tracker parameters (interpreted by the chosen tracker)
    tracker_params: dict[str, Any] = field(default_factory=dict)
    # If True, also emit raw detections_ready (in addition to tracks_ready when tracking enabled)
    emit_detections: bool = True
    # Increment to request an explicit tracker reset (IDs will restart)
    tracker_reset_token: int = 0


class YoloDetectorWorker(QThread):
    """
    Desktop-local object detector worker (commercial-safe primary path: ONNXRuntime YOLO).

    - Latest-frame only (drops intermediate frames).
    - Throttled inference (fps_limit).
    - Emits detections formatted for CameraOpenGLWidget.update_detections():
        [{'bbox': {'x':..,'y':..,'w':..,'h':..}, 'class': 'person', 'confidence': 0.83}, ...]
    """

    detections_ready = Signal(object, object)  # dets(list[dict]), stats(dict)
    tracks_ready = Signal(object, object)  # tracks(list[dict]), stats(dict)
    status = Signal(str)
    error = Signal(str)

    def __init__(self, config: DetectorConfig):
        super().__init__()
        self._cfg = config
        self._lock = threading.Lock()
        self._latest_frame: Optional[np.ndarray] = None
        self._latest_ts: float = 0.0
        self._latest_meta: Optional[dict] = None
        self._stop = threading.Event()
        self._last_infer_ts: float = 0.0

        # Model/runtime state
        self._backend = None
        self._backend_id: Optional[str] = None
        self._model_variant: Optional[str] = None
        self._model_path: Optional[str] = None
        self._device_resolved: Optional[str] = None
        self._class_name_to_id: dict[str, int] = {}
        self._last_cfg_sig: Optional[str] = None
        self._first_ready_emitted: bool = False
        self._last_error_text: Optional[str] = None
        self._last_error_emit_ts: float = 0.0

        # Tracking state (lives in worker thread)
        self._tracker = None
        self._tracker_sig: Optional[str] = None

    def update_config(self, config: DetectorConfig):
        with self._lock:
            self._cfg = config

    def set_fps_limit(self, fps: int) -> None:
        """Live-update the inference rate cap.

        Used by the auto-protection load shedder to throttle this
        worker without restarting it.  ``fps <= 0`` is clamped to 1
        (callers that want to fully stop should call ``stop()``).
        """
        try:
            new_fps = max(1, int(fps))
        except Exception:
            return
        with self._lock:
            try:
                if hasattr(self._cfg, "fps_limit"):
                    self._cfg.fps_limit = new_fps
            except Exception:
                pass

    def submit_frame(self, frame: np.ndarray, meta: Optional[dict] = None):
        # IMPORTANT: avoid per-frame .copy() allocations; keep only the latest.
        if frame is None:
            return
        with self._lock:
            self._latest_frame = frame
            self._latest_ts = time.time()
            self._latest_meta = dict(meta or {}) if meta else None

    def stop(self):
        self._stop.set()

    def _resolve_device(self, cfg: DetectorConfig) -> str:
        return resolve_device_for_backend(str(getattr(cfg, "backend", "onnxruntime")), str(getattr(cfg, "device", "auto")))

    def _build_allowed_class_ids(self, allowed_names: Optional[Iterable[str]]) -> Optional[list[int]]:
        if not allowed_names:
            return None
        ids: list[int] = []
        for n in allowed_names:
            if not n:
                continue
            key = str(n).strip().lower()
            cid = self._class_name_to_id.get(key)
            if cid is not None:
                ids.append(int(cid))
        # If nothing matched, treat as "no filter" rather than filtering everything out.
        return ids or None

    def _resolve_model(self, cfg: DetectorConfig) -> tuple[str, Optional[list[str]]]:
        """
        Return (model_path, labels_list).

        Supports:
        - BYO slug (models/byo/<slug>/manifest.json)
        - direct .onnx path
        """
        key = str(getattr(cfg, "model_variant", "") or "").strip()
        # Legacy migration: old saved configs used yolov8*.pt variants. Treat as "default" BYO.
        if key.lower() in {"yolov8n", "yolov8s", "yolov8m", "yolov8l", "yolov8x"}:
            key = "default"
        if not key or key == "default":
            # prefer the first installed BYO model
            try:
                from core.model_library.byo_models import list_installed_manifests

                ms = list_installed_manifests()
                if ms:
                    m0 = ms[0]
                    labels = read_labels_file(m0.labels_abs_path()) if m0.labels_abs_path() else None
                    return str(m0.model_abs_path()), labels
            except Exception:
                pass
            # Seamless fallback: use any local ONNX dropped under models/ (except models/byo/*).
            try:
                local_onnx = list_local_onnx_models(include_byo=False)
                if local_onnx:
                    return str(local_onnx[0]), None
            except Exception:
                pass
            raise RuntimeError("No detector model configured. Use tray: Models → Install/Import, then Enable detector.")

        # direct ONNX path
        if key.lower().endswith(".onnx"):
            p = Path(key)
            if p.is_absolute():
                return str(p), None
            # Relative paths are resolved under models/.
            models_dir = get_models_dir()
            cands = [
                models_dir / p,
                Path.cwd() / p,
            ]
            for cand in cands:
                try:
                    if cand.exists():
                        return str(cand.resolve()), None
                except Exception:
                    continue
            return key, None

        # BYO manifest slug
        man = load_manifest(key)
        if man is None:
            # Also allow selecting by basename/stem for ad-hoc ONNX files dropped into models/.
            try:
                models_dir = get_models_dir()
                needle = key.strip().lower()
                local_onnx = list_local_onnx_models(include_byo=False)
                for p in local_onnx:
                    stem = p.stem.strip().lower()
                    name = p.name.strip().lower()
                    rel = str(p.relative_to(models_dir)).replace("\\", "/").strip().lower()
                    if needle in {stem, name, rel}:
                        return str(p), None
            except Exception:
                pass
            raise RuntimeError(f"Unknown model '{key}'. Use tray: Models → Manage installed models…")
        labels = read_labels_file(man.labels_abs_path()) if man.labels_abs_path() else None
        return str(man.model_abs_path()), labels

    def _ensure_model_loaded(self, cfg: DetectorConfig) -> None:
        requested_backend = str(getattr(cfg, "backend", "auto") or "auto").strip().lower()
        variant = str(cfg.model_variant or "default").strip()
        device = self._resolve_device(cfg)

        cfg_sig = (
            f"{requested_backend}|{variant}|{device}|{int(cfg.imgsz)}|{float(cfg.min_confidence):.3f}|"
            f"{int(cfg.max_det)}|{','.join((cfg.allowed_classes or []))}"
        )

        if self._backend is not None and self._last_cfg_sig == cfg_sig:
            return

        # AUTO: prefer ONNX YOLO if a BYO model exists; otherwise fall back to MobileNetSSD.
        backend = None
        info = None
        backend_id_effective = requested_backend
        model_path = None
        labels = None

        def _load_mobilenet() -> None:
            nonlocal backend, info, backend_id_effective, model_path, labels
            backend_id_effective = "mobilenetssd"
            backend = create_backend("mobilenetssd")
            mobilenet_device = str(device or "auto").strip().lower()
            info = backend.load(model_path="", labels=None, device=mobilenet_device, input_size=300)
            model_path = str(getattr(info, "model_path", "") or "")
            labels = backend.get_class_names()

        try:
            if requested_backend in ("mobilenetssd", "mobilenet", "opencv-mobilenetssd"):
                _load_mobilenet()
            else:
                # onnxruntime or auto
                model_path, labels = self._resolve_model(cfg)
                backend_id_effective = "onnxruntime" if requested_backend in ("auto", "default", "onnxruntime", "onnx") else requested_backend
                if self._backend is None or self._model_variant != variant or self._backend_id != backend_id_effective:
                    self.status.emit(f"Detections: loading model ({variant})…")
                if device.startswith("cuda"):
                    self.status.emit("Detections: warming up GPU…")
                backend = create_backend(backend_id_effective)
                info = backend.load(
                    model_path=str(model_path),
                    labels=labels,
                    device=str(device),
                    input_size=int(cfg.imgsz or 640),
                )
        except Exception:
            # Hard requirement: never fail detection at startup — fall back to MobileNetSSD.
            _load_mobilenet()
            try:
                self.status.emit("Detections: ONNX model unavailable → using MobileNetSSD (fallback)")
            except Exception:
                pass

        self._backend = backend
        self._backend_id = str(getattr(info, "backend_id", "") or backend_id_effective)
        self._model_variant = variant
        self._model_path = str(model_path)
        self._device_resolved = str(info.device or device)

        # Build name->id mapping for class filtering (label -> index)
        self._class_name_to_id = {}
        try:
            names = backend.get_class_names() or []
            for i, nm in enumerate(names):
                key = str(nm).strip().lower()
                if key:
                    self._class_name_to_id[key] = int(i)
        except Exception:
            self._class_name_to_id = {}

        self._last_cfg_sig = cfg_sig
        self.status.emit(f"Detections: ready ({self._device_resolved})")
        self._first_ready_emitted = True

    def run(self):
        last_wait_status_ts: float = 0.0
        while not self._stop.is_set():
            with self._lock:
                cfg = self._cfg
                frame = self._latest_frame
                frame_ts = self._latest_ts
                meta = self._latest_meta
                # Consume latest frame (drop older ones)
                self._latest_frame = None
                self._latest_meta = None

            if not bool(cfg.enabled):
                time.sleep(0.05)
                continue

            if frame is None:
                now = time.time()
                if now - last_wait_status_ts > 1.0:
                    last_wait_status_ts = now
                    try:
                        self.status.emit("Detections: waiting for frames…")
                    except Exception:
                        pass
                time.sleep(0.01)
                continue

            # Basic FPS limit (wall-clock)
            min_dt = 1.0 / max(1, int(cfg.fps_limit))
            now = time.time()
            if now - self._last_infer_ts < min_dt:
                time.sleep(max(0.0, min_dt - (now - self._last_infer_ts)))

            try:
                self._ensure_model_loaded(cfg)
                backend = self._backend
                if backend is None:
                    time.sleep(0.1)
                    continue

                device = self._device_resolved or self._resolve_device(cfg)
                allowed = normalize_allowed_classes(cfg.allowed_classes)

                t0 = time.perf_counter()
                dets_out = backend.detect(
                    frame,
                    min_confidence=float(cfg.min_confidence),
                    max_det=int(cfg.max_det),
                    allowed_classes=allowed,
                )
                dt_ms = (time.perf_counter() - t0) * 1000.0

                # If inference was run on a crop, map back to original frame coords.
                if isinstance(meta, dict) and (meta.get("offset_x") or meta.get("offset_y")):
                    try:
                        ox = float(meta.get("offset_x", 0.0) or 0.0)
                        oy = float(meta.get("offset_y", 0.0) or 0.0)
                        for d in dets_out or []:
                            bb = d.get("bbox") if isinstance(d, dict) else None
                            if isinstance(bb, dict):
                                bb["x"] = float(bb.get("x", 0.0) or 0.0) + ox
                                bb["y"] = float(bb.get("y", 0.0) or 0.0) + oy
                    except Exception:
                        pass

                stats = {
                    "timestamp": float(frame_ts or time.time()),
                    "device": str(device),
                    "backend": str(self._backend_id or ""),
                    "model": str(self._model_variant or ""),
                    "inference_ms": float(dt_ms),
                    "fps_limit": int(cfg.fps_limit),
                    "num_detections": int(len(dets_out)),
                }
                try:
                    if isinstance(meta, dict) and meta:
                        stats["roi_crop"] = True
                        stats["roi_crop_rect"] = meta.get("roi_crop_rect")
                    else:
                        stats["roi_crop"] = False
                except Exception:
                    pass

                # Expose class names (best-effort) so UI can populate class filters without
                # needing to import/instantiate Ultralytics models in the UI thread.
                try:
                    # _class_name_to_id is normalized name->id; invert to id->name
                    inv = {}
                    for nm, cid in (self._class_name_to_id or {}).items():
                        try:
                            inv[int(cid)] = str(nm)
                        except Exception:
                            continue
                    if inv:
                        stats["num_classes"] = int(len(inv))
                        stats["class_names"] = [inv[k] for k in sorted(inv.keys())]
                except Exception:
                    pass

                # Optional tracking
                track_stats = {}
                if bool(getattr(cfg, "tracking_enabled", False)):
                    try:
                        t_track0 = time.perf_counter()
                        tracks_out = self._update_tracker(cfg, dets_out, float(stats["timestamp"]))
                        t_track_ms = (time.perf_counter() - t_track0) * 1000.0
                        track_stats = {
                            **stats,
                            "tracker_type": str(getattr(cfg, "tracker_type", "sort")),
                            "tracking_enabled": True,
                            "tracking_ms": float(t_track_ms),
                            "num_tracks": int(len(tracks_out or [])),
                        }
                        self.tracks_ready.emit(tracks_out, track_stats)
                    except Exception as e:
                        # Tracking errors should not kill detections
                        self.error.emit(f"Tracking error: {e}")
                else:
                    # If tracking was turned off, discard tracker state so future enable is clean.
                    try:
                        if self._tracker is not None:
                            self._tracker.reset()
                    except Exception:
                        pass
                    self._tracker = None
                    self._tracker_sig = None

                # Keep detections_ready optional (useful for debugging and shape interactions)
                if bool(getattr(cfg, "emit_detections", True)):
                    self.detections_ready.emit(dets_out, stats)
                self._last_infer_ts = time.time()
            except Exception as e:
                # Never spam logs/UI: rate-limit repeated errors.
                msg = f"Detections error: {e}"
                now = time.time()
                should_emit = (msg != self._last_error_text) or ((now - float(self._last_error_emit_ts or 0.0)) > 5.0)
                if should_emit:
                    self._last_error_text = msg
                    self._last_error_emit_ts = now
                    try:
                        self.error.emit(msg)
                    except Exception:
                        pass
                time.sleep(1.0)

    def _tracker_signature(self, cfg: DetectorConfig) -> str:
        # Any change should rebuild/reset tracker
        try:
            params = cfg.tracker_params or {}
            # stable-ish string
            items = ",".join([f"{k}={params[k]}" for k in sorted(params.keys())])
        except Exception:
            items = ""
        return f"{str(getattr(cfg,'tracker_type','sort'))}|{items}|{int(getattr(cfg,'tracker_reset_token',0) or 0)}"

    def _ensure_tracker(self, cfg: DetectorConfig):
        sig = self._tracker_signature(cfg)
        if self._tracker is not None and self._tracker_sig == sig:
            return self._tracker

        # (Re)build tracker
        ttype = str(getattr(cfg, "tracker_type", "sort") or "sort").lower().strip()
        params = getattr(cfg, "tracker_params", None) or {}

        if ttype == "bytetrack":
            from desktop.utils.object_tracker.bytetrack_tracker import ByteTrackObjectTracker, ByteTrackConfig

            btcfg = ByteTrackConfig(
                track_thresh=float(params.get("track_thresh", 0.35)),
                low_thresh=float(params.get("low_thresh", 0.10)),
                match_thresh=float(params.get("match_thresh", 0.30)),
                track_buffer=int(params.get("track_buffer", 30)),
                min_box_area=float(params.get("min_box_area", 10.0)),
            )
            self._tracker = ByteTrackObjectTracker(btcfg)
        else:
            from desktop.utils.object_tracker.sort_tracker import SortObjectTracker, SortTrackerConfig

            scfg = SortTrackerConfig(
                max_age=int(params.get("max_age", 15)),
                min_hits=int(params.get("min_hits", 2)),
                iou_threshold=float(params.get("iou_threshold", 0.30)),
            )
            self._tracker = SortObjectTracker(scfg)

        self._tracker_sig = sig
        try:
            self._tracker.reset()
        except Exception:
            pass
        return self._tracker

    def _update_tracker(self, cfg: DetectorConfig, dets_out: list[dict], ts: float) -> list[dict]:
        from desktop.utils.object_tracker.base import detections_from_dicts

        tracker = self._ensure_tracker(cfg)
        dets = detections_from_dicts(dets_out or [], ts=ts)
        tracks = tracker.update(dets, frame_ts=ts) or []
        # export as dicts for easy UI consumption
        return [t.to_dict() for t in tracks]


