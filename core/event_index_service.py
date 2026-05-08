from __future__ import annotations

import base64
import hashlib
import json
import math
import os
import re
import sqlite3
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import requests
import cv2
import numpy as np

from .paths import get_data_dir, get_motion_watch_dir, get_project_root, get_recordings_dir, resolve_under_root

try:
    from PIL import Image
except Exception:  # pragma: no cover
    Image = None  # type: ignore


_WATCH_TS_RE = re.compile(r"_watch_(\d+)\.(?:jpg|jpeg|png)$", re.IGNORECASE)
_CLIP_TS_RE = re.compile(r"_clip_(\d+)\.(?:mp4|avi|mkv)$", re.IGNORECASE)


def _discover_custom_capture_dirs() -> List[Path]:
    """
    Read persisted Motion Watch settings and collect any custom save_dir /
    clip_save_dir paths that users have configured.  This lets the
    EventIndexService auto-discover captures stored outside the default
    ``captures/motion_watch`` directory.
    """
    settings_path = get_data_dir() / "motion_watch_settings.json"
    dirs: List[Path] = []
    try:
        if not settings_path.exists():
            return dirs
        data = json.loads(settings_path.read_text())
        if not isinstance(data, dict):
            return dirs
        seen: set = set()
        for _cam_id, cam_settings in data.items():
            if not isinstance(cam_settings, dict):
                continue
            for key in ("save_dir", "clip_save_dir"):
                raw = (cam_settings.get(key) or "").strip()
                if not raw:
                    continue
                p = Path(raw).expanduser()
                if not p.is_absolute():
                    p = (get_project_root() / p).resolve()
                else:
                    p = p.resolve()
                if p.exists() and p.is_dir() and str(p) not in seen:
                    dirs.append(p)
                    seen.add(str(p))
    except Exception:
        pass
    return dirs


def _utc_iso(ts: int) -> str:
    return datetime.fromtimestamp(int(ts), tz=timezone.utc).isoformat()


def _safe_int(v: Any, default: Optional[int] = None) -> Optional[int]:
    try:
        if v is None:
            return default
        return int(float(v))
    except Exception:
        return default


def _safe_float(v: Any, default: Optional[float] = None) -> Optional[float]:
    try:
        if v is None:
            return default
        return float(v)
    except Exception:
        return default


def _sha1(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def _safe_slug(text: str) -> str:
    s = str(text or "").strip().lower()
    s = re.sub(r"[^a-z0-9]+", "_", s)
    s = s.strip("_")
    return s or "obj"


@dataclass
class EventSearchResult:
    id: str
    file_path: str
    thumb_path: Optional[str]
    camera_id: Optional[str]
    camera_name: Optional[str]
    captured_ts: Optional[int]
    captured_at: Optional[str]
    trigger_type: Optional[str]
    shape_type: Optional[str]
    shape_name: Optional[str]
    dominant_color: Optional[str]
    caption: Optional[str]
    tags: List[str]
    detection_classes: List[str]
    metadata: Dict[str, Any]
    media_type: Optional[str] = "image"


class EventIndexService:
    """
    Production-safe, disk-backed index for Motion Watch and other capture events.
    Stores:
      - source file path (JPG)
      - derived thumbnail
      - metadata JSON
      - searchable text (caption/tags/classes) via SQLite FTS5 when available
    """

    def __init__(
        self,
        *,
        db_path: Optional[Path] = None,
        thumbs_dir: Optional[Path] = None,
        crops_dir: Optional[Path] = None,
        capture_roots: Optional[Sequence[Path]] = None,
        local_vision_endpoint: Optional[str] = None,
        local_vision_model: str = "blip2",
    ):
        # Anchor paths to the project root so indexing/search is not sensitive to process CWD.
        self.project_root = get_project_root()

        db_path_is_default = db_path is None
        thumbs_dir_is_default = thumbs_dir is None
        crops_dir_is_default = crops_dir is None
        capture_roots_is_default = capture_roots is None

        if db_path is None:
            db_path = get_data_dir() / "events_index.sqlite"
        if thumbs_dir is None:
            thumbs_dir = get_data_dir() / "events_thumbs"
        if crops_dir is None:
            crops_dir = get_data_dir() / "events_crops"
        if capture_roots is None:
            capture_roots = [get_motion_watch_dir()]

        # Allow env overrides without forcing callers to plumb config.
        # Apply env overrides only when caller did not explicitly pass a custom path.
        env_db = (os.environ.get("EVENTS_INDEX_DB") or "").strip()
        env_thumbs = (os.environ.get("EVENTS_THUMBS_DIR") or "").strip()
        env_crops = (os.environ.get("EVENTS_CROPS_DIR") or "").strip()
        if db_path_is_default and env_db:
            db_path = Path(env_db)
        if thumbs_dir_is_default and env_thumbs:
            thumbs_dir = Path(env_thumbs)
        if crops_dir_is_default and env_crops:
            crops_dir = Path(env_crops)

        self.db_path = resolve_under_root(Path(db_path), root=self.project_root)
        self.thumbs_dir = resolve_under_root(Path(thumbs_dir), root=self.project_root)
        self.crops_dir = resolve_under_root(Path(crops_dir), root=self.project_root)
        # capture roots are treated as directories; allow env-relative or repo-relative inputs.
        self.capture_roots = [resolve_under_root(Path(p), root=self.project_root) for p in (capture_roots or [])]
        if capture_roots_is_default and not self.capture_roots:
            self.capture_roots = [get_motion_watch_dir()]
        # Auto-discover custom save directories from per-camera Motion Watch settings
        # so backfill and search find captures the user stored on external drives.
        self._merge_custom_capture_dirs()
        # Default to the standard local vision port so the index works out-of-the-box.
        # If the service isn't running, requests will fail and we fall back to detectors.
        self.local_vision_endpoint = (
            local_vision_endpoint
            or os.environ.get("LOCAL_VISION_ENDPOINT")
            or "http://127.0.0.1:8101"
        ).strip()
        # Prefer production service default ("blip2"); dev service accepts "blip"/"git"
        self.local_vision_model_id = (local_vision_model or "blip2").strip()
        self.local_vision_model = self.local_vision_model_id
        try:
            from core.model_library.catalog import resolve_vision_model_slug

            self.local_vision_model = resolve_vision_model_slug(self.local_vision_model_id)
        except Exception:
            self.local_vision_model = self.local_vision_model_id

        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.thumbs_dir.mkdir(parents=True, exist_ok=True)
        self.crops_dir.mkdir(parents=True, exist_ok=True)
        self._init_db()
        self._camera_name_cache: Optional[Dict[str, str]] = None

    def _rooted(self, p: Path) -> Path:
        return resolve_under_root(Path(p), root=self.project_root)

    def _merge_custom_capture_dirs(self) -> None:
        """Merge user-configured capture directories into capture_roots (deduplicated)."""
        existing = {str(r.resolve()) for r in self.capture_roots}
        for d in _discover_custom_capture_dirs():
            key = str(d.resolve())
            if key not in existing:
                self.capture_roots.append(d)
                existing.add(key)

    def refresh_capture_roots(self) -> None:
        """Re-scan Motion Watch settings and add any newly configured directories."""
        self._merge_custom_capture_dirs()

    @staticmethod
    def _safe_row_get(row, col: str, default: str = "") -> str:
        """Safely read a column from a sqlite3.Row that may not have the column (schema evolution)."""
        try:
            val = row[col]
            return str(val) if val is not None else default
        except (IndexError, KeyError):
            return default

    def _load_camera_name_map(self) -> Dict[str, str]:
        """Best-effort mapping of camera_id -> camera_name from cameras.json/data/cameras.json."""
        if isinstance(self._camera_name_cache, dict) and self._camera_name_cache:
            return self._camera_name_cache
        out: Dict[str, str] = {}
        try:
            # Prefer per-user data dir cameras config (portable/frozen builds), then fall back to repo-root paths.
            try:
                data_dir = get_data_dir()
                candidates = [data_dir / "cameras.json", data_dir.parent / "cameras.json"]
            except Exception:
                candidates = []
            candidates.extend([self._rooted(Path("cameras.json")), self._rooted(Path("data/cameras.json"))])
            for p in candidates:
                try:
                    if not p.exists():
                        continue
                    j = json.loads(p.read_text(encoding="utf-8"))
                    if isinstance(j, list):
                        for item in j:
                            if not isinstance(item, dict):
                                continue
                            cid = str(item.get("id") or "").strip()
                            name = str(item.get("name") or "").strip()
                            if cid and name:
                                out[cid] = name
                except Exception:
                    continue
        except Exception:
            out = {}
        self._camera_name_cache = out
        return out

    # ------------------------ DB ------------------------
    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=NORMAL;")
        return conn

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS events (
                  id TEXT PRIMARY KEY,
                  file_path TEXT NOT NULL UNIQUE,
                  thumb_path TEXT,
                  camera_id TEXT,
                  camera_name TEXT,
                  captured_ts INTEGER,
                  captured_at TEXT,
                  trigger_type TEXT,
                  shape_type TEXT,
                  shape_name TEXT,
                  shape_id TEXT,
                  dominant_color TEXT,
                  caption TEXT,
                  tags TEXT,
                  detection_classes TEXT,
                  json_metadata TEXT,
                  created_at INTEGER,
                  updated_at INTEGER
                );
                """
            )
            conn.execute("CREATE INDEX IF NOT EXISTS idx_events_captured_ts ON events(captured_ts);")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_events_camera_name ON events(camera_name);")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_events_camera_id ON events(camera_id);")

            # Lightweight schema evolution for existing installs
            try:
                cols = [r["name"] for r in conn.execute("PRAGMA table_info(events);").fetchall() if r and r["name"]]
            except Exception:
                cols = []
            if "detections_indexed_at" not in cols:
                try:
                    conn.execute("ALTER TABLE events ADD COLUMN detections_indexed_at INTEGER;")
                except Exception:
                    pass
            if "media_type" not in cols:
                try:
                    conn.execute("ALTER TABLE events ADD COLUMN media_type TEXT DEFAULT 'image';")
                except Exception:
                    pass

            # Object-level detections (YOLO/local vision, etc.)
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS event_detections (
                  detection_id TEXT PRIMARY KEY,
                  event_id TEXT NOT NULL,
                  det_idx INTEGER NOT NULL,
                  class TEXT,
                  confidence REAL,
                  bbox_x REAL,
                  bbox_y REAL,
                  bbox_w REAL,
                  bbox_h REAL,
                  crop_path TEXT,
                  color TEXT,
                  area REAL,
                  source TEXT,
                  created_at INTEGER,
                  updated_at INTEGER
                );
                """
            )
            conn.execute("CREATE INDEX IF NOT EXISTS idx_event_detections_event_id ON event_detections(event_id);")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_event_detections_class ON event_detections(class);")
            conn.execute("CREATE UNIQUE INDEX IF NOT EXISTS uq_event_detections_event_detidx ON event_detections(event_id, det_idx);")

            # Per-detection overrides (user corrections)
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS event_detection_overrides (
                  detection_id TEXT PRIMARY KEY,
                  event_id TEXT,
                  det_idx INTEGER,
                  override_class TEXT,
                  override_color TEXT,
                  override_tags TEXT,
                  note TEXT,
                  updated_at INTEGER,
                  updated_by TEXT
                );
                """
            )
            conn.execute("CREATE INDEX IF NOT EXISTS idx_event_detection_overrides_event_id ON event_detection_overrides(event_id);")

            # If we just added detections_indexed_at, best-effort seed it for events that already have detections.
            try:
                conn.execute(
                    """
                    UPDATE events
                    SET detections_indexed_at = COALESCE(detections_indexed_at, updated_at, created_at)
                    WHERE detections_indexed_at IS NULL
                      AND EXISTS (SELECT 1 FROM event_detections d WHERE d.event_id = events.id);
                    """
                )
            except Exception:
                pass

            # FTS5 is optional; build best-effort.
            try:
                conn.execute(
                    """
                    CREATE VIRTUAL TABLE IF NOT EXISTS events_fts USING fts5(
                      caption,
                      tags,
                      detection_classes,
                      camera_name,
                      shape_name,
                      content='events',
                      content_rowid='rowid'
                    );
                    """
                )
                conn.execute(
                    """
                    CREATE TRIGGER IF NOT EXISTS events_ai AFTER INSERT ON events BEGIN
                      INSERT INTO events_fts(rowid, caption, tags, detection_classes, camera_name, shape_name)
                      VALUES (new.rowid, new.caption, new.tags, new.detection_classes, new.camera_name, new.shape_name);
                    END;
                    """
                )
                conn.execute(
                    """
                    CREATE TRIGGER IF NOT EXISTS events_ad AFTER DELETE ON events BEGIN
                      INSERT INTO events_fts(events_fts, rowid, caption, tags, detection_classes, camera_name, shape_name)
                      VALUES('delete', old.rowid, old.caption, old.tags, old.detection_classes, old.camera_name, old.shape_name);
                    END;
                    """
                )
                conn.execute(
                    """
                    CREATE TRIGGER IF NOT EXISTS events_au AFTER UPDATE ON events BEGIN
                      INSERT INTO events_fts(events_fts, rowid, caption, tags, detection_classes, camera_name, shape_name)
                      VALUES('delete', old.rowid, old.caption, old.tags, old.detection_classes, old.camera_name, old.shape_name);
                      INSERT INTO events_fts(rowid, caption, tags, detection_classes, camera_name, shape_name)
                      VALUES (new.rowid, new.caption, new.tags, new.detection_classes, new.camera_name, new.shape_name);
                    END;
                    """
                )
            except Exception:
                # No FTS available; fallback to LIKE search.
                pass

    # ------------------------ Detections / overrides ------------------------
    @staticmethod
    def _normalize_bbox(det: Dict[str, Any]) -> Optional[Tuple[float, float, float, float]]:
        bb = det.get("bbox") or det.get("box") or det.get("rect") or {}
        if isinstance(bb, dict):
            x = _safe_float(bb.get("x"))
            y = _safe_float(bb.get("y"))
            w = _safe_float(bb.get("w"))
            h = _safe_float(bb.get("h"))
        elif isinstance(bb, (list, tuple)) and len(bb) >= 4:
            x = _safe_float(bb[0])
            y = _safe_float(bb[1])
            w = _safe_float(bb[2])
            h = _safe_float(bb[3])
        else:
            # tolerate xyxy formats
            x1 = _safe_float(det.get("x1"))
            y1 = _safe_float(det.get("y1"))
            x2 = _safe_float(det.get("x2"))
            y2 = _safe_float(det.get("y2"))
            if None not in (x1, y1, x2, y2):
                x = float(x1 or 0)
                y = float(y1 or 0)
                w = float((x2 or 0) - (x1 or 0))
                h = float((y2 or 0) - (y1 or 0))
            else:
                return None
        if x is None or y is None or w is None or h is None:
            return None
        if w <= 0 or h <= 0:
            return None
        return float(x), float(y), float(w), float(h)

    @staticmethod
    def _estimate_color_from_bgr(crop_bgr: Optional[np.ndarray]) -> Optional[str]:
        """
        Fast color bucket on a BGR crop. Returns a small set of human-searchable labels.
        """
        try:
            if crop_bgr is None:
                return None
            if not isinstance(crop_bgr, np.ndarray) or crop_bgr.size == 0:
                return None
            # Downsample for speed/stability
            h, w = crop_bgr.shape[:2]
            if h <= 0 or w <= 0:
                return None
            small = crop_bgr
            if h > 80 or w > 80:
                small = cv2.resize(crop_bgr, (64, 64), interpolation=cv2.INTER_AREA)
            b, g, r = [float(x) for x in np.mean(small, axis=(0, 1))[:3]]

            # Luma-ish (same coefficients as the PIL path)
            yv = 0.2126 * r + 0.7152 * g + 0.0722 * b
            if yv >= 220:
                return "white"
            if yv <= 35:
                return "black"

            mx = max(r, g, b)
            mn = min(r, g, b)
            sat = 0.0 if mx <= 1e-6 else (mx - mn) / mx
            if sat < 0.22:
                return "white" if yv >= 125 else "gray"

            if r > g * 1.2 and r > b * 1.2:
                return "red" if r < 170 else "yellow"
            if g > r * 1.2 and g > b * 1.2:
                return "green"
            if b > r * 1.2 and b > g * 1.2:
                return "blue"
            if r > 120 and g > 90 and b < 90:
                return "brown"
            return None
        except Exception:
            return None

    @staticmethod
    def _compute_detection_id(event_id: str, cls: str, bbox: Tuple[float, float, float, float]) -> str:
        # Quantize slightly so stable-ish across tiny bbox jitter.
        x, y, w, h = bbox
        key = f"{event_id}|{(cls or '').strip().lower()}|{round(x,1)}|{round(y,1)}|{round(w,1)}|{round(h,1)}"
        return _sha1(key)

    def _save_crop(self, *, event_id: str, det_idx: int, cls: str, detection_id: str, image_bgr: np.ndarray, bbox: Tuple[float, float, float, float]) -> Optional[Path]:
        try:
            x, y, w, h = bbox
            ih, iw = image_bgr.shape[:2]
            if ih <= 0 or iw <= 0:
                return None
            x0 = max(0, int(x))
            y0 = max(0, int(y))
            x1 = min(iw, int(x + w))
            y1 = min(ih, int(y + h))
            if x1 <= x0 or y1 <= y0:
                return None

            crop = image_bgr[y0:y1, x0:x1]
            if crop.size == 0:
                return None

            out_dir = (self.crops_dir / str(event_id))
            out_dir.mkdir(parents=True, exist_ok=True)
            out_name = f"det_{int(det_idx)}_{_safe_slug(cls)}_{detection_id[:10]}.jpg"
            out_path = out_dir / out_name
            # JPEG write
            cv2.imwrite(str(out_path), crop, [int(cv2.IMWRITE_JPEG_QUALITY), 88])
            return out_path
        except Exception:
            return None

    def _replace_event_detections(self, event_id: str) -> None:
        with self._connect() as conn:
            conn.execute("DELETE FROM event_detections WHERE event_id = ?;", (str(event_id),))
            # keep overrides; they may still match on detection_id after reindex

    def _upsert_detections(
        self,
        *,
        event_id: str,
        file_path: Path,
        detections: List[Dict[str, Any]],
        source: str = "yolo",
        replace_existing: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        Persist detections to event_detections, generate crops, and return normalized rows.
        """
        now = int(time.time())
        out_rows: List[Dict[str, Any]] = []

        if replace_existing:
            try:
                self._replace_event_detections(event_id)
            except Exception:
                pass

        img = None
        try:
            img = cv2.imread(str(file_path))
        except Exception:
            img = None

        with self._connect() as conn:
            for idx, d in enumerate(detections or []):
                if not isinstance(d, dict):
                    continue
                cls = (d.get("class") or d.get("label") or d.get("class_name") or "").strip().lower()
                conf = _safe_float(d.get("confidence"), 0.0) or 0.0
                bbox = self._normalize_bbox(d)
                if not bbox:
                    continue
                x, y, w, h = bbox
                area = float(max(0.0, w) * max(0.0, h))
                detection_id = self._compute_detection_id(str(event_id), cls, bbox)

                crop_path = None
                color = None
                if img is not None:
                    crop_path = self._save_crop(
                        event_id=str(event_id),
                        det_idx=int(idx),
                        cls=cls,
                        detection_id=detection_id,
                        image_bgr=img,
                        bbox=bbox,
                    )
                    try:
                        if crop_path and crop_path.exists():
                            # Re-read crop to estimate color quickly (avoids PIL overhead)
                            crop_img = cv2.imread(str(crop_path))
                            color = self._estimate_color_from_bgr(crop_img)
                    except Exception:
                        color = None

                row = {
                    "detection_id": detection_id,
                    "event_id": str(event_id),
                    "det_idx": int(idx),
                    "class": cls or None,
                    "confidence": float(conf),
                    "bbox_x": float(x),
                    "bbox_y": float(y),
                    "bbox_w": float(w),
                    "bbox_h": float(h),
                    "crop_path": str(crop_path) if crop_path else None,
                    "color": str(color).strip().lower() if isinstance(color, str) and color.strip() else None,
                    "area": float(area),
                    "source": str(source or "yolo"),
                    "created_at": now,
                    "updated_at": now,
                }

                conn.execute(
                    """
                    INSERT INTO event_detections(
                      detection_id, event_id, det_idx,
                      class, confidence,
                      bbox_x, bbox_y, bbox_w, bbox_h,
                      crop_path, color, area,
                      source,
                      created_at, updated_at
                    ) VALUES (
                      :detection_id, :event_id, :det_idx,
                      :class, :confidence,
                      :bbox_x, :bbox_y, :bbox_w, :bbox_h,
                      :crop_path, :color, :area,
                      :source,
                      :created_at, :updated_at
                    )
                    ON CONFLICT(detection_id) DO UPDATE SET
                      event_id=excluded.event_id,
                      det_idx=excluded.det_idx,
                      class=excluded.class,
                      confidence=excluded.confidence,
                      bbox_x=excluded.bbox_x,
                      bbox_y=excluded.bbox_y,
                      bbox_w=excluded.bbox_w,
                      bbox_h=excluded.bbox_h,
                      crop_path=excluded.crop_path,
                      color=excluded.color,
                      area=excluded.area,
                      source=excluded.source,
                      updated_at=excluded.updated_at;
                    """,
                    row,
                )
                out_rows.append(row)
        return out_rows

    def list_detections(self, event_id: str) -> List[Dict[str, Any]]:
        """
        Return detections for an event with effective (override-aware) class/color/tags.
        """
        event_id = str(event_id or "").strip()
        if not event_id:
            return []
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT
                  d.detection_id,
                  d.event_id,
                  d.det_idx,
                  d.class AS class_raw,
                  d.confidence,
                  d.bbox_x, d.bbox_y, d.bbox_w, d.bbox_h,
                  d.crop_path,
                  d.color AS color_raw,
                  d.area,
                  d.source,
                  o.override_class,
                  o.override_color,
                  o.override_tags,
                  o.note,
                  o.updated_at,
                  o.updated_by
                FROM event_detections d
                LEFT JOIN event_detection_overrides o ON o.detection_id = d.detection_id
                WHERE d.event_id = ?
                ORDER BY d.det_idx ASC;
                """,
                (event_id,),
            ).fetchall()
        out: List[Dict[str, Any]] = []
        for r in rows:
            eff_class = (r["override_class"] or r["class_raw"] or None)
            eff_color = (r["override_color"] or r["color_raw"] or None)
            tags: List[str] = []
            try:
                if r["override_tags"]:
                    raw = str(r["override_tags"])
                    if raw.strip().startswith("["):
                        arr = json.loads(raw)
                        if isinstance(arr, list):
                            tags = [str(t).strip().lower() for t in arr if str(t).strip()]
                    else:
                        tags = [t.strip().lower() for t in raw.split(",") if t.strip()]
            except Exception:
                tags = []

            out.append(
                {
                    "detection_id": r["detection_id"],
                    "event_id": r["event_id"],
                    "det_idx": int(r["det_idx"]),
                    "class": str(eff_class).strip().lower() if eff_class else None,
                    "class_raw": str(r["class_raw"]).strip().lower() if r["class_raw"] else None,
                    "confidence": float(r["confidence"] or 0.0),
                    "bbox": {
                        "x": float(r["bbox_x"] or 0.0),
                        "y": float(r["bbox_y"] or 0.0),
                        "w": float(r["bbox_w"] or 0.0),
                        "h": float(r["bbox_h"] or 0.0),
                    },
                    "crop_path": str(r["crop_path"]) if r["crop_path"] else None,
                    "color": str(eff_color).strip().lower() if eff_color else None,
                    "color_raw": str(r["color_raw"]).strip().lower() if r["color_raw"] else None,
                    "area": float(r["area"] or 0.0),
                    "source": str(r["source"] or ""),
                    "override": {
                        "override_class": str(r["override_class"]).strip().lower() if r["override_class"] else None,
                        "override_color": str(r["override_color"]).strip().lower() if r["override_color"] else None,
                        "override_tags": tags,
                        "note": str(r["note"]) if r["note"] else None,
                        "updated_at": int(r["updated_at"]) if r["updated_at"] is not None else None,
                        "updated_by": str(r["updated_by"]) if r["updated_by"] else None,
                    },
                }
            )
        return out

    def set_detection_override(
        self,
        *,
        event_id: str,
        detection_id: Optional[str] = None,
        det_idx: Optional[int] = None,
        override_class: Optional[str] = None,
        override_color: Optional[str] = None,
        override_tags: Optional[Sequence[str]] = None,
        note: Optional[str] = None,
        updated_by: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Persist a user correction and refresh event-level aggregates so subsequent searches reflect it.
        """
        event_id = str(event_id or "").strip()
        if not event_id:
            raise ValueError("event_id is required")

        det_id = (str(detection_id).strip() if isinstance(detection_id, str) and detection_id.strip() else None)
        det_idx_i = int(det_idx) if det_idx is not None else None

        if not det_id:
            if det_idx_i is None:
                raise ValueError("detection_id or det_idx is required")
            with self._connect() as conn:
                r = conn.execute(
                    "SELECT detection_id FROM event_detections WHERE event_id = ? AND det_idx = ? LIMIT 1;",
                    (event_id, int(det_idx_i)),
                ).fetchone()
            if not r:
                raise ValueError("Detection not found")
            det_id = str(r["detection_id"])

        now = int(time.time())
        ocls = str(override_class).strip().lower() if isinstance(override_class, str) and override_class.strip() else None
        ocol = str(override_color).strip().lower() if isinstance(override_color, str) and override_color.strip() else None
        otags = None
        if override_tags is not None:
            try:
                otags = json.dumps([str(t).strip().lower() for t in override_tags if str(t).strip()][:24])
            except Exception:
                otags = None

        with self._connect() as conn:
            # Best-effort: capture det_idx for UI convenience
            if det_idx_i is None:
                try:
                    r = conn.execute(
                        "SELECT det_idx FROM event_detections WHERE detection_id = ? LIMIT 1;",
                        (det_id,),
                    ).fetchone()
                    if r:
                        det_idx_i = int(r["det_idx"])
                except Exception:
                    det_idx_i = None

            payload = {
                "detection_id": det_id,
                "event_id": event_id,
                "det_idx": det_idx_i,
                "override_class": ocls,
                "override_color": ocol,
                "override_tags": otags,
                "note": str(note) if isinstance(note, str) and note.strip() else None,
                "updated_at": now,
                "updated_by": str(updated_by) if isinstance(updated_by, str) and updated_by.strip() else None,
            }
            conn.execute(
                """
                INSERT INTO event_detection_overrides(
                  detection_id, event_id, det_idx,
                  override_class, override_color, override_tags,
                  note, updated_at, updated_by
                ) VALUES (
                  :detection_id, :event_id, :det_idx,
                  :override_class, :override_color, :override_tags,
                  :note, :updated_at, :updated_by
                )
                ON CONFLICT(detection_id) DO UPDATE SET
                  event_id=excluded.event_id,
                  det_idx=excluded.det_idx,
                  override_class=excluded.override_class,
                  override_color=excluded.override_color,
                  override_tags=excluded.override_tags,
                  note=excluded.note,
                  updated_at=excluded.updated_at,
                  updated_by=excluded.updated_by;
                """,
                payload,
            )

        # Refresh aggregated fields (so FTS + filters reflect overrides)
        self._refresh_event_aggregates(event_id)
        return {"event_id": event_id, "detection_id": det_id, "det_idx": det_idx_i, "updated_at": now}

    def _refresh_event_aggregates(self, event_id: str) -> None:
        """
        Update `events.tags`, `events.detection_classes`, and (best-effort) `events.dominant_color`
        based on effective detections (overrides applied).
        """
        event_id = str(event_id or "").strip()
        if not event_id:
            return
        dets = self.list_detections(event_id)
        classes: List[str] = []
        tags: List[str] = []
        vehicle_candidates: List[Tuple[float, float, Optional[str]]] = []  # score, area, color

        for d in dets:
            cls = (d.get("class") or "").strip().lower()
            if cls:
                classes.append(cls)
                tags.append(cls)
            col = (d.get("color") or "").strip().lower()
            if col:
                tags.append(col)
            try:
                conf = float(d.get("confidence") or 0.0)
                area = float(d.get("area") or 0.0)
            except Exception:
                conf, area = 0.0, 0.0
            if cls in {"car", "truck", "bus", "motorcycle"}:
                vehicle_candidates.append((area * (0.25 + conf), area, col or None))

        classes = list(dict.fromkeys([c for c in classes if c]))[:24]
        tags = list(dict.fromkeys([t for t in tags if t]))[:24]

        dominant = None
        try:
            vehicle_candidates.sort(key=lambda t: t[0], reverse=True)
            if vehicle_candidates and vehicle_candidates[0][2]:
                dominant = vehicle_candidates[0][2]
        except Exception:
            dominant = None

        with self._connect() as conn:
            conn.execute(
                """
                UPDATE events
                SET detection_classes = ?,
                    tags = ?,
                    dominant_color = COALESCE(?, dominant_color),
                    updated_at = ?
                WHERE id = ?;
                """,
                (", ".join(classes) if classes else "", ", ".join(tags) if tags else "", dominant, int(time.time()), event_id),
            )

    # ------------------------ Path safety ------------------------
    def _resolve_capture_path(self, file_path: str) -> Path:
        p = Path(file_path)
        p = self._rooted(p) if not p.is_absolute() else p.resolve()
        for root in self.capture_roots:
            try:
                root_abs = root.resolve()
                p.relative_to(root_abs)
                return p
            except Exception:
                continue
        # Allow absolute paths outside capture_roots when the file actually exists
        # (e.g. clips saved to a user-chosen external drive).
        if p.is_absolute() and p.exists():
            return p
        raise ValueError("File path is outside configured capture roots")

    # ------------------------ Ingest helpers ------------------------
    def _infer_captured_ts(self, file_path: Path, payload: Dict[str, Any]) -> Optional[int]:
        # Prefer payload
        ts = _safe_int(payload.get("captured_ts") or payload.get("capturedTs"))
        if ts:
            return ts

        captured_at = payload.get("captured_at") or payload.get("capturedAt")
        if isinstance(captured_at, str) and captured_at.strip():
            try:
                dt = datetime.fromisoformat(captured_at.replace("Z", "+00:00"))
                return int(dt.timestamp())
            except Exception:
                pass

        m = _WATCH_TS_RE.search(file_path.name)
        if m:
            return _safe_int(m.group(1))
        m2 = _CLIP_TS_RE.search(file_path.name)
        if m2:
            return _safe_int(m2.group(1))
        try:
            return int(file_path.stat().st_mtime)
        except Exception:
            return None

    def _encode_image_b64(self, file_path: Path) -> str:
        raw = file_path.read_bytes()
        return base64.b64encode(raw).decode("utf-8")

    def _make_thumb(self, file_path: Path, *, width: int = 320) -> Optional[Path]:
        key = _sha1(str(file_path))
        out = self.thumbs_dir / f"{key}.jpg"

        suffix = file_path.suffix.lower()
        if suffix in (".mp4", ".avi", ".mkv", ".webm"):
            # For clips: prefer co-located .thumb.jpg, else extract first frame via OpenCV
            co_thumb = file_path.with_suffix(".thumb.jpg")
            if co_thumb.exists():
                try:
                    if Image is not None:
                        im = Image.open(co_thumb).convert("RGB")
                        if width > 0 and im.width > width:
                            h = max(1, int(im.height * (width / im.width)))
                            im = im.resize((width, h))
                        im.save(out, "JPEG", quality=82, optimize=True)
                        return out
                except Exception:
                    pass
            try:
                cap = cv2.VideoCapture(str(file_path))
                ret, frame = cap.read()
                cap.release()
                if ret and frame is not None:
                    if width > 0 and frame.shape[1] > width:
                        scale = width / frame.shape[1]
                        frame = cv2.resize(frame, (width, max(1, int(frame.shape[0] * scale))))
                    cv2.imwrite(str(out), frame, [cv2.IMWRITE_JPEG_QUALITY, 82])
                    return out
            except Exception:
                pass
            return None

        if Image is None:
            return None
        try:
            im = Image.open(file_path).convert("RGB")
            if width > 0 and im.width > width:
                h = max(1, int(im.height * (width / im.width)))
                im = im.resize((width, h))
            im.save(out, "JPEG", quality=82, optimize=True)
            return out
        except Exception:
            return None

    def _estimate_dominant_color(self, file_path: Path, *, crop_box: Optional[Tuple[int, int, int, int]] = None) -> Optional[str]:
        """
        Very lightweight color estimate (white/black/gray/red/green/blue/yellow/brown).
        If crop_box is provided, it should be (x, y, w, h) in pixels.
        """
        suffix = file_path.suffix.lower()
        if suffix in (".mp4", ".avi", ".mkv", ".webm"):
            return None
        if Image is None:
            return None
        try:
            im = Image.open(file_path).convert("RGB")
            if crop_box:
                x, y, w, h = crop_box
                x = max(0, int(x))
                y = max(0, int(y))
                w = max(1, int(w))
                h = max(1, int(h))
                im = im.crop((x, y, x + w, y + h))
            im = im.resize((64, 64))
            px = list(im.getdata())
            if not px:
                return None
            r = sum(p[0] for p in px) / len(px)
            g = sum(p[1] for p in px) / len(px)
            b = sum(p[2] for p in px) / len(px)

            # Luma-like intensity
            yv = 0.2126 * r + 0.7152 * g + 0.0722 * b
            # White vehicles at dusk often land in the 125-200 luma range; treat low-saturation,
            # higher-luma regions as "white" rather than "gray" to support queries like "white truck".
            if yv >= 220:
                return "white"
            if yv <= 35:
                return "black"
            # saturation-ish
            mx = max(r, g, b)
            mn = min(r, g, b)
            sat = 0 if mx == 0 else (mx - mn) / mx
            if sat < 0.22:
                return "white" if yv >= 125 else "gray"
            # dominant hue bucket via channel dominance
            if r > g * 1.2 and r > b * 1.2:
                return "red" if r < 170 else "yellow"
            if g > r * 1.2 and g > b * 1.2:
                return "green"
            if b > r * 1.2 and b > g * 1.2:
                return "blue"
            if r > 120 and g > 90 and b < 90:
                return "brown"
            return None
        except Exception:
            return None

    def _yolo_detect(self, file_path: Path) -> List[Dict[str, Any]]:
        """
        Run YOLO detection (cached) on a file and return detections with bbox + label + confidence.
        """
        try:
            frame = cv2.imread(str(file_path))
            if frame is None:
                return []
            # Cache YOLO detector on the instance (avoid repeated model loads).
            try:
                from .object_detector import ObjectDetector
                if getattr(self, "_yolo_detector", None) is None:
                    self._yolo_detector = ObjectDetector(model_type="yolo", device="auto")  # type: ignore[attr-defined]
                det = getattr(self, "_yolo_detector", None)
                if det is None:
                    return []
                dets = det.detect(frame, conf_threshold=0.25) or []
                return [d for d in dets if isinstance(d, dict)]
            except Exception:
                # Fallback: MobileNet SSD (keeps indexing functional even if Ultralytics isn't installed).
                try:
                    from .object_detector import ObjectDetector
                    if getattr(self, "_mobilenet_detector", None) is None:
                        self._mobilenet_detector = ObjectDetector(model_type="mobilenet", device="auto")  # type: ignore[attr-defined]
                    det = getattr(self, "_mobilenet_detector", None)
                    if det is None:
                        return []
                    dets = det.detect(frame, conf_threshold=0.25) or []
                    return [d for d in dets if isinstance(d, dict)]
                except Exception:
                    return []
        except Exception:
            return []

    def _describe_local_vision(self, file_path: Path) -> Tuple[Optional[str], List[str], List[str]]:
        """
        Best-effort caption/tags via local vision service (/describe).
        Returns (caption, tags, detection_classes).
        """
        if not self.local_vision_endpoint:
            return None, [], []
        try:
            image_b64 = self._encode_image_b64(file_path)
            # The service expects raw base64 (no data-url prefix).
            payload = {
                "image": image_b64,
                "model": self.local_vision_model,
                "use_cache": True,
                "include_detections": True,
            }
            url = self.local_vision_endpoint.rstrip("/") + "/describe"
            # Keep this bounded; if the vision service is slow/unavailable we fall back to detectors.
            res = requests.post(url, json=payload, timeout=8)
            if not res.ok:
                return None, [], []
            j = res.json() if isinstance(res.json(), dict) else {}
            caption = None
            tags: List[str] = []
            classes: List[str] = []

            # production service shape: {"results":[{caption, objects:[{label,...}]}], "aggregate_caption":...}
            if isinstance(j.get("aggregate_caption"), str) and j.get("aggregate_caption"):
                caption = j.get("aggregate_caption")
            results = j.get("results")
            if isinstance(results, list) and results:
                first = results[0] if isinstance(results[0], dict) else {}
                if not caption and isinstance(first.get("caption"), str):
                    caption = first.get("caption")
                objs = first.get("objects") or []
                if isinstance(objs, list):
                    for o in objs:
                        if not isinstance(o, dict):
                            continue
                        lab = o.get("label")
                        if isinstance(lab, str) and lab:
                            tags.append(lab)
                            classes.append(lab)
            # Dedup
            tags = list(dict.fromkeys([t.strip().lower() for t in tags if t and isinstance(t, str)]))[:24]
            classes = list(dict.fromkeys([c.strip().lower() for c in classes if c and isinstance(c, str)]))[:24]
            return caption.strip() if isinstance(caption, str) and caption.strip() else None, tags, classes
        except Exception:
            return None, [], []

    def _detect_with_detectors(
        self,
        file_path: Path,
        *,
        camera_id: Optional[str] = None,
        crop_box: Optional[Tuple[int, int, int, int]] = None,
    ) -> List[str]:
        """
        Best-effort object detection fallback using the built-in detector stack.
        Returns a lowercase, de-duplicated list of class labels.
        """
        try:
            frame = cv2.imread(str(file_path))
            if frame is None:
                return []

            if crop_box:
                x, y, w, h = crop_box
                x = max(0, int(x))
                y = max(0, int(y))
                w = max(1, int(w))
                h = max(1, int(h))
                frame = frame[y : y + h, x : x + w]

            # Prefer YOLO for rich classes (includes "truck"). Cache the detector on the instance.
            dets: List[Dict[str, Any]] = []
            try:
                from .object_detector import ObjectDetector  # local import (heavy)

                if getattr(self, "_yolo_detector", None) is None:
                    self._yolo_detector = ObjectDetector(model_type="yolo", device="auto")  # type: ignore[attr-defined]
                det = getattr(self, "_yolo_detector", None)
                if det is not None:
                    dets = det.detect(frame, conf_threshold=0.25) or []
            except Exception:
                dets = []

            # Fallback: use the global detector manager (usually MobileNet) if YOLO isn't available.
            if not dets:
                try:
                    from .detector_manager import get_detector_manager

                    cam = str(camera_id) if camera_id is not None else "event"
                    dm = get_detector_manager()
                    out = dm.detect_and_track(cam, frame, conf_threshold=0.25, force_detection=True) or {}
                    dets = (out.get("detections") or []) if isinstance(out, dict) else []
                    if not dets:
                        dets = (out.get("tracks") or []) if isinstance(out, dict) else []
                except Exception:
                    dets = []

            classes: List[str] = []
            for d in dets or []:
                if not isinstance(d, dict):
                    continue
                lab = d.get("class") or d.get("label") or d.get("class_name")
                if isinstance(lab, str) and lab.strip():
                    classes.append(lab.strip().lower())

            # Dedup + cap
            return list(dict.fromkeys([c for c in classes if c]))[:24]
        except Exception:
            return []

    # ------------------------ Public API ------------------------
    def ingest(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        Ingest a capture + metadata into the index.

        Required:
          - file_path: absolute or relative path to an image under capture_roots

        Optional:
          - event_id / id
          - camera_id, camera_name
          - captured_ts / captured_at
          - trigger: {interaction_type, shape_type, shape_id, shape_name, ...}
          - motion_box: {x,y,w,h} or [x,y,w,h]
          - metadata: arbitrary dict (will be stored)
        """
        file_path_raw = payload.get("file_path") or payload.get("path") or payload.get("image_path")
        if not isinstance(file_path_raw, str) or not file_path_raw.strip():
            raise ValueError("file_path is required")

        file_path = self._resolve_capture_path(file_path_raw.strip())
        if not file_path.exists():
            raise FileNotFoundError(str(file_path))

        captured_ts = self._infer_captured_ts(file_path, payload)
        captured_at = payload.get("captured_at") or payload.get("capturedAt") or (_utc_iso(captured_ts) if captured_ts else None)

        event_id = payload.get("event_id") or payload.get("id")
        if not isinstance(event_id, str) or not event_id.strip():
            stable = f"{file_path}|{captured_ts or ''}"
            event_id = _sha1(stable)
        event_id = str(event_id)

        camera_id = payload.get("camera_id") or payload.get("cameraId") or payload.get("camera")
        camera_name = payload.get("camera_name") or payload.get("cameraName") or payload.get("camera")

        trigger = payload.get("trigger") if isinstance(payload.get("trigger"), dict) else {}
        trigger_type = trigger.get("interaction_type") or trigger.get("trigger_type") or payload.get("trigger_type")
        shape_type = trigger.get("shape_type") or payload.get("shape_type")
        shape_name = trigger.get("shape_name") or payload.get("shape_name")
        shape_id = trigger.get("shape_id") or payload.get("shape_id")

        motion_box = payload.get("motion_box") or payload.get("motionBox") or payload.get("motion_bbox")
        crop_tuple: Optional[Tuple[int, int, int, int]] = None
        if isinstance(motion_box, dict):
            crop_tuple = (
                _safe_int(motion_box.get("x"), 0) or 0,
                _safe_int(motion_box.get("y"), 0) or 0,
                _safe_int(motion_box.get("w"), 0) or 0,
                _safe_int(motion_box.get("h"), 0) or 0,
            )
        elif isinstance(motion_box, list) and len(motion_box) >= 4:
            crop_tuple = (_safe_int(motion_box[0], 0) or 0, _safe_int(motion_box[1], 0) or 0, _safe_int(motion_box[2], 0) or 0, _safe_int(motion_box[3], 0) or 0)

        dominant_color = self._estimate_dominant_color(file_path, crop_box=crop_tuple)
        thumb_path = self._make_thumb(file_path)

        # Allow caller-provided enrichment (e.g., desktop Motion Watch sidecar writes).
        pre_caption = payload.get("caption")
        pre_tags = payload.get("tags")
        pre_classes = payload.get("detection_classes") or payload.get("classes")
        pre_dom = payload.get("dominant_color") or payload.get("color")
        try:
            if isinstance(pre_tags, str):
                pre_tags_list = [t.strip().lower() for t in pre_tags.split(",") if t.strip()]
            elif isinstance(pre_tags, list):
                pre_tags_list = [str(t).strip().lower() for t in pre_tags if str(t).strip()]
            else:
                pre_tags_list = []
        except Exception:
            pre_tags_list = []
        try:
            if isinstance(pre_classes, str):
                pre_classes_list = [c.strip().lower() for c in pre_classes.split(",") if c.strip()]
            elif isinstance(pre_classes, list):
                pre_classes_list = [str(c).strip().lower() for c in pre_classes if str(c).strip()]
            else:
                pre_classes_list = []
        except Exception:
            pre_classes_list = []
        if isinstance(pre_dom, str) and pre_dom.strip():
            dominant_color = pre_dom.strip().lower()

        payload_metadata = payload.get("metadata") if isinstance(payload.get("metadata"), dict) else {}
        meta = payload_metadata if isinstance(payload_metadata, dict) else {}

        enable_vision = bool(payload.get("enable_vision", True))
        enable_detections = bool(payload.get("enable_detections", payload.get("include_detections", enable_vision)))

        # Start from caller-provided info
        caption = pre_caption.strip() if isinstance(pre_caption, str) and pre_caption.strip() else None
        tags = list(dict.fromkeys(pre_tags_list))[:24]
        classes = list(dict.fromkeys(pre_classes_list))[:24]

        # Optional caption/tags/classes from local vision (/describe)
        if enable_vision:
            cap2, tags2, classes2 = self._describe_local_vision(file_path)
            if cap2 and not caption:
                caption = cap2
            tags = list(dict.fromkeys([*(tags or []), *(tags2 or [])]))[:24]
            classes = list(dict.fromkeys([*(classes or []), *(classes2 or [])]))[:24]
            try:
                from core.model_library.store import get_installed_record

                record = get_installed_record(self.local_vision_model_id)
                meta["vision_source"] = "local"
                meta["vision_model_id"] = self.local_vision_model_id
                meta["vision_model_slug"] = self.local_vision_model
                if isinstance(record, dict):
                    if record.get("revision"):
                        meta["vision_model_revision"] = record.get("revision")
                    if record.get("repo_id"):
                        meta["vision_model_repo"] = record.get("repo_id")
            except Exception:
                pass

        # Optional local detections (YOLO) - needed for "truck/car" searches
        yolo_dets: List[Dict[str, Any]] = []
        vehicle_color = None
        vehicle_box = None
        vehicle_label = None
        if enable_detections:
            # Prefer caller-provided detections (Motion Watch sidecar), else run YOLO.
            yolo_dets = []
            provided_dets = payload.get("detections") or payload_metadata.get("detections")
            if isinstance(provided_dets, list) and provided_dets:
                yolo_dets = [d for d in provided_dets if isinstance(d, dict)]
            else:
                yolo_dets = self._yolo_detect(file_path)
            yolo_classes = []
            for d in yolo_dets:
                lab = d.get("class") or d.get("label") or d.get("class_name")
                if isinstance(lab, str) and lab.strip():
                    yolo_classes.append(lab.strip().lower())
            yolo_classes = list(dict.fromkeys([c for c in yolo_classes if c]))[:24]
            if yolo_classes:
                classes = list(dict.fromkeys([*(classes or []), *yolo_classes]))[:24]
                tags = list(dict.fromkeys([*(tags or []), *yolo_classes]))[:24]

            # Vehicle color: estimate from the largest detected vehicle box (full-frame YOLO).
            try:
                veh = []
                for d in yolo_dets or []:
                    lab = str(d.get("class") or d.get("label") or "").strip().lower()
                    if lab not in {"car", "truck", "bus", "motorcycle"}:
                        continue
                    bb = d.get("bbox") or {}
                    x = float(bb.get("x", 0) or 0)
                    y = float(bb.get("y", 0) or 0)
                    w = float(bb.get("w", 0) or 0)
                    h = float(bb.get("h", 0) or 0)
                    score = float(d.get("confidence", 0.0) or 0.0)
                    area = max(0.0, w) * max(0.0, h)
                    veh.append((area * (0.25 + score), lab, (x, y, w, h), score))
                veh.sort(key=lambda t: t[0], reverse=True)
                if veh:
                    _, vehicle_label, (x, y, w, h), _ = veh[0]
                    vehicle_box = (int(x), int(y), int(w), int(h))
                    vehicle_color = self._estimate_dominant_color(file_path, crop_box=vehicle_box)
            except Exception:
                vehicle_color = None

            if isinstance(vehicle_color, str) and vehicle_color:
                # Promote vehicle color into dominant_color so existing filters ("white") work.
                dominant_color = vehicle_color
                tags = list(dict.fromkeys([*(tags or []), vehicle_color]))[:24]

        # Persist per-object detections + crops (object-level index)
        det_rows: List[Dict[str, Any]] = []
        try:
            if enable_detections:
                # Persist detections after event_id is finalized; replace_existing keeps the event consistent.
                det_rows = self._upsert_detections(
                    event_id=str(event_id),
                    file_path=file_path,
                    detections=yolo_dets or [],
                    source="yolo",
                    replace_existing=bool(payload.get("replace_detections", True)),
                )
        except Exception:
            pass

        # Preserve trigger/motion_box at top level too for debugging portability
        meta = {
            **meta,
            **({"trigger": trigger} if trigger else {}),
            **({"motion_box": motion_box} if motion_box else {}),
        }
        # Store compact YOLO summary for LLM recall / debugging
        try:
            if enable_detections:
                if isinstance(yolo_dets, list) and yolo_dets:
                    meta["yolo"] = {
                        "detections": [
                            {
                                "class": d.get("class") or d.get("label") or d.get("class_name"),
                                "confidence": float(d.get("confidence", 0.0) or 0.0),
                                "bbox": d.get("bbox"),
                            }
                            for d in yolo_dets[:15]
                            if isinstance(d, dict)
                        ]
                    }
                if vehicle_color:
                    meta["vehicle_color"] = vehicle_color
                if vehicle_label:
                    meta["vehicle_label"] = vehicle_label
                if vehicle_box:
                    meta["vehicle_box"] = vehicle_box
                # Mark detection indexing attempt so reindex can skip even when there are zero objects.
                meta["detections_indexed_at"] = now
                meta["detections_count"] = int(len(det_rows or []))
        except Exception:
            pass

        media_type = payload.get("media_type") or "image"
        if isinstance(meta, dict) and meta.get("media_type"):
            media_type = meta["media_type"]

        now = int(time.time())
        row = {
            "id": event_id,
            "file_path": str(file_path),
            "thumb_path": str(thumb_path) if thumb_path else None,
            "camera_id": str(camera_id) if camera_id is not None else None,
            "camera_name": str(camera_name) if camera_name is not None else None,
            "captured_ts": int(captured_ts) if captured_ts is not None else None,
            "captured_at": str(captured_at) if captured_at is not None else None,
            "trigger_type": str(trigger_type) if trigger_type is not None else None,
            "shape_type": str(shape_type) if shape_type is not None else None,
            "shape_name": str(shape_name) if shape_name is not None else None,
            "shape_id": str(shape_id) if shape_id is not None else None,
            "dominant_color": dominant_color,
            "caption": caption,
            "tags": ", ".join(tags) if tags else "",
            "detection_classes": ", ".join(classes) if classes else "",
            "json_metadata": json.dumps(meta, default=str),
            "media_type": str(media_type),
            "created_at": now,
            "updated_at": now,
            "detections_indexed_at": now if enable_detections else None,
        }

        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO events(
                  id, file_path, thumb_path, camera_id, camera_name,
                  captured_ts, captured_at,
                  trigger_type, shape_type, shape_name, shape_id,
                  dominant_color,
                  caption, tags, detection_classes,
                  json_metadata, media_type,
                  created_at, updated_at,
                  detections_indexed_at
                ) VALUES (
                  :id, :file_path, :thumb_path, :camera_id, :camera_name,
                  :captured_ts, :captured_at,
                  :trigger_type, :shape_type, :shape_name, :shape_id,
                  :dominant_color,
                  :caption, :tags, :detection_classes,
                  :json_metadata, :media_type,
                  :created_at, :updated_at,
                  :detections_indexed_at
                )
                ON CONFLICT(file_path) DO UPDATE SET
                  id=excluded.id,
                  thumb_path=excluded.thumb_path,
                  camera_id=excluded.camera_id,
                  camera_name=excluded.camera_name,
                  captured_ts=excluded.captured_ts,
                  captured_at=excluded.captured_at,
                  trigger_type=excluded.trigger_type,
                  shape_type=excluded.shape_type,
                  shape_name=excluded.shape_name,
                  shape_id=excluded.shape_id,
                  dominant_color=excluded.dominant_color,
                  caption=excluded.caption,
                  tags=excluded.tags,
                  detection_classes=excluded.detection_classes,
                  json_metadata=excluded.json_metadata,
                  media_type=excluded.media_type,
                  detections_indexed_at=COALESCE(excluded.detections_indexed_at, events.detections_indexed_at),
                  updated_at=excluded.updated_at;
                """,
                row,
            )

        # Update event aggregates based on detections/overrides (keeps searches consistent).
        try:
            if enable_detections:
                self._refresh_event_aggregates(str(event_id))
        except Exception:
            pass

        return {
            "event_id": event_id,
            "file_path": str(file_path),
            "thumb_path": row["thumb_path"],
            "media_type": str(media_type),
            "camera_name": str(camera_name) if camera_name else None,
            "shape_name": str(shape_name) if shape_name else None,
            "trigger_type": str(trigger_type) if trigger_type else None,
            "captured_ts": int(captured_ts) if captured_ts else None,
            "caption": caption,
            "tags": tags or [],
            "detection_classes": classes or [],
        }

    def backfill(
        self,
        *,
        max_items: int = 250,
        include_vision: bool = False,
        include_detections: bool = True,
    ) -> Dict[str, Any]:
        """
        Scan capture_roots and ingest items not yet indexed.
        This is intentionally bounded and best-effort (safe for production).
        """
        self.refresh_capture_roots()
        max_items = max(1, min(int(max_items or 250), 5000))
        processed = 0
        scanned = 0
        skipped = 0
        errors: List[str] = []

        # Build a quick in-memory map of known paths + enrichment flags.
        known: Dict[str, Dict[str, Any]] = {}
        try:
            with self._connect() as conn:
                for row in conn.execute("SELECT file_path, camera_name, caption, tags, detection_classes FROM events;").fetchall():
                    fp = row["file_path"]
                    if isinstance(fp, str) and fp:
                        cam_name = row["camera_name"]
                        cap = row["caption"]
                        tags = row["tags"]
                        det = row["detection_classes"]
                        has_vision = bool((cap and str(cap).strip()) or (tags and str(tags).strip()) or (det and str(det).strip()))
                        has_camera = bool(cam_name and str(cam_name).strip())
                        has_classes = bool(det and str(det).strip())
                        known[str(Path(fp).resolve())] = {"has_vision": has_vision, "has_camera": has_camera, "has_classes": has_classes}
        except Exception:
            known = {}

        candidates: List[Path] = []
        for root in self.capture_roots:
            try:
                root_abs = root.resolve()
                if not root_abs.exists():
                    continue
                # Most recent first
                for p in root_abs.glob("*.jpg"):
                    candidates.append(p)
                for p in root_abs.glob("*.jpeg"):
                    candidates.append(p)
                for p in root_abs.glob("*.png"):
                    candidates.append(p)
                for p in root_abs.glob("*.mp4"):
                    candidates.append(p)
                # Also scan zone subdirectories
                for p in root_abs.rglob("*.jpg"):
                    if p not in candidates:
                        candidates.append(p)
                for p in root_abs.rglob("*.mp4"):
                    if p not in candidates:
                        candidates.append(p)
            except Exception:
                continue
        candidates.sort(key=lambda p: p.stat().st_mtime if p.exists() else 0, reverse=True)

        cam_map = self._load_camera_name_map()

        for p in candidates:
            # Stop once we've actually processed/enriched max_items (skips don't consume the budget)
            if processed >= max_items:
                break
            try:
                scanned += 1
                p_abs = p.resolve()
                existing = known.get(str(p_abs))
                if existing is not None:
                    needs_camera = not bool(existing.get("has_camera"))
                    needs_classes = not bool(existing.get("has_classes"))
                    # If include_vision=True, we also want to fill missing detection_classes (vehicle search depends on it).
                    needs_vision = bool(include_vision) and (not bool(existing.get("has_vision")) or not bool(existing.get("has_classes")))
                    # Skip only when we don't need to enrich anything.
                    # Even when include_vision=False, we still want to fill missing classes when a sidecar provides detections.
                    if not (needs_camera or needs_vision or needs_classes):
                        skipped += 1
                        continue

                # Prefer relative-to-project paths for portability and safety checks.
                try:
                    rel_str = str(p_abs.relative_to(self.project_root.resolve()))
                except Exception:
                    rel_str = str(p_abs)

                # If a sidecar JSON exists, ingest it so we capture camera_name, trigger_type, motion_box, etc.
                payload: Dict[str, Any] = {"file_path": rel_str}
                try:
                    sidecar = p_abs.with_suffix(".json")
                    if sidecar.exists() and sidecar.is_file():
                        j = json.loads(sidecar.read_text(encoding="utf-8"))
                        if isinstance(j, dict):
                            payload = {**j}
                except Exception:
                    payload = {"file_path": rel_str}

                # If there is no sidecar, infer camera_id + timestamp from filename (legacy captures).
                try:
                    if not payload.get("camera_id") and "_watch_" in p_abs.name:
                        cam_id_guess = p_abs.name.split("_watch_", 1)[0]
                        if cam_id_guess:
                            payload["camera_id"] = cam_id_guess
                            if not payload.get("camera_name"):
                                payload["camera_name"] = cam_map.get(cam_id_guess)
                    if not payload.get("captured_ts"):
                        m = _WATCH_TS_RE.search(p_abs.name)
                        if m:
                            payload["captured_ts"] = _safe_int(m.group(1))
                except Exception:
                    pass

                # Respect backfill flags (override any stored value).
                # Important: if the Motion Watch sidecar already contains detections, enable detection persistence
                # even when include_detections=False (so we don't run YOLO, but we DO store/aggregate provided results).
                sidecar_dets = payload.get("detections") or (payload.get("metadata") or {}).get("detections")
                has_sidecar_dets = isinstance(sidecar_dets, list) and any(isinstance(d, dict) for d in sidecar_dets)
                payload["enable_vision"] = bool(include_vision)
                payload["enable_detections"] = bool(include_detections) or bool(has_sidecar_dets)
                self.ingest(payload)
                processed += 1
            except Exception as e:
                errors.append(f"{p}: {e}")
                continue

        return {
            "processed": processed,
            "scanned": scanned,
            "skipped": skipped,
            "errors": errors[:25],
            "error_count": len(errors),
        }

    def ingest_recent_for_camera(
        self,
        camera_id: str,
        *,
        max_items: int = 25,
        include_detections: bool = True,
        include_vision: bool = False,
    ) -> Dict[str, Any]:
        """
        Best-effort: ingest the most recent Motion Watch captures for a camera_id.
        This helps when a capture was just written but the desktop ingest POST failed
        (e.g., backend restart).
        """
        cam_id = str(camera_id or "").strip()
        if not cam_id:
            return {"processed": 0, "scanned": 0, "skipped": 0, "errors": [], "error_count": 0}
        max_items = max(1, min(int(max_items or 25), 200))

        processed = 0
        scanned = 0
        skipped = 0
        errors: List[str] = []

        # Known paths (absolute) + whether they're already enriched.
        known: Dict[str, Dict[str, Any]] = {}
        try:
            with self._connect() as conn:
                for row in conn.execute("SELECT file_path, camera_name, caption, tags, detection_classes FROM events;").fetchall():
                    fp = row["file_path"]
                    if not (isinstance(fp, str) and fp):
                        continue
                    try:
                        fp_abs = str(Path(fp).resolve())
                    except Exception:
                        continue
                    cam_name = row["camera_name"]
                    cap = row["caption"]
                    tags = row["tags"]
                    det = row["detection_classes"]
                    has_vision = bool((cap and str(cap).strip()) or (tags and str(tags).strip()) or (det and str(det).strip()))
                    has_camera = bool(cam_name and str(cam_name).strip())
                    has_classes = bool(det and str(det).strip())
                    known[fp_abs] = {"has_vision": has_vision, "has_camera": has_camera, "has_classes": has_classes}
        except Exception:
            known = {}

        candidates: List[Path] = []
        for root in self.capture_roots:
            try:
                root_abs = root.resolve()
                if not root_abs.exists():
                    continue
                candidates.extend(list(root_abs.glob(f"{cam_id}_watch_*.jpg")))
                candidates.extend(list(root_abs.glob(f"{cam_id}_watch_*.jpeg")))
                candidates.extend(list(root_abs.glob(f"{cam_id}_watch_*.png")))
                candidates.extend(list(root_abs.glob(f"{cam_id}_clip_*.mp4")))
                # Also scan zone subdirectories
                candidates.extend(list(root_abs.rglob(f"{cam_id}_watch_*.jpg")))
                candidates.extend(list(root_abs.rglob(f"{cam_id}_clip_*.mp4")))
            except Exception:
                continue

        # Most recent first
        candidates.sort(key=lambda p: p.stat().st_mtime if p.exists() else 0, reverse=True)

        cam_map = self._load_camera_name_map()

        for p in candidates:
            if processed >= max_items:
                break
            scanned += 1
            try:
                p_abs = p.resolve()
                existing = known.get(str(p_abs))
                if existing is not None:
                    needs_camera = not bool(existing.get("has_camera"))
                    needs_classes = not bool(existing.get("has_classes"))
                    needs_vision = bool(include_vision) and (not bool(existing.get("has_vision")) or not bool(existing.get("has_classes")))
                    if not (needs_camera or needs_classes or needs_vision):
                        skipped += 1
                        continue

                try:
                    rel_str = str(p_abs.relative_to(self.project_root.resolve()))
                except Exception:
                    rel_str = str(p_abs)

                payload: Dict[str, Any] = {"file_path": rel_str}
                try:
                    sidecar = p_abs.with_suffix(".json")
                    if sidecar.exists() and sidecar.is_file():
                        j = json.loads(sidecar.read_text(encoding="utf-8"))
                        if isinstance(j, dict):
                            payload = {**j}
                except Exception:
                    payload = {"file_path": rel_str}

                # Fill camera_name if missing
                try:
                    payload.setdefault("camera_id", cam_id)
                    if not payload.get("camera_name"):
                        payload["camera_name"] = cam_map.get(cam_id)
                except Exception:
                    pass

                sidecar_dets = payload.get("detections") or (payload.get("metadata") or {}).get("detections")
                has_sidecar_dets = isinstance(sidecar_dets, list) and any(isinstance(d, dict) for d in sidecar_dets)
                payload["enable_vision"] = bool(include_vision)
                payload["enable_detections"] = bool(include_detections) or bool(has_sidecar_dets)
                self.ingest(payload)
                processed += 1
            except Exception as e:
                errors.append(f"{p}: {e}")

        return {
            "processed": processed,
            "scanned": scanned,
            "skipped": skipped,
            "errors": errors[:25],
            "error_count": len(errors),
        }

    def status(self) -> Dict[str, Any]:
        with self._connect() as conn:
            try:
                total = int(conn.execute("SELECT COUNT(1) AS c FROM events;").fetchone()["c"])
            except Exception:
                total = 0
            latest = conn.execute("SELECT id, camera_name, captured_at, file_path FROM events ORDER BY captured_ts DESC NULLS LAST, updated_at DESC LIMIT 1;").fetchone()
            try:
                det_total = int(conn.execute("SELECT COUNT(1) AS c FROM event_detections;").fetchone()["c"])
            except Exception:
                det_total = 0
            try:
                events_with_det_rows = int(conn.execute("SELECT COUNT(1) AS c FROM events e WHERE EXISTS (SELECT 1 FROM event_detections d WHERE d.event_id=e.id);").fetchone()["c"])
            except Exception:
                events_with_det_rows = 0
            try:
                events_det_indexed = int(conn.execute("SELECT COUNT(1) AS c FROM events WHERE detections_indexed_at IS NOT NULL;").fetchone()["c"])
            except Exception:
                events_det_indexed = 0
        return {
            "cwd": os.getcwd(),
            "project_root": str(self.project_root),
            "db_path": str(self.db_path),
            "thumbs_dir": str(self.thumbs_dir),
            "crops_dir": str(self.crops_dir),
            "capture_roots": [str(p) for p in self.capture_roots],
            "local_vision_endpoint": self.local_vision_endpoint or None,
            "total_events": total,
            "detections": {
                "total_rows": det_total,
                "events_with_detection_rows": events_with_det_rows,
                "events_detection_indexed": events_det_indexed,
                "events_detection_not_indexed": max(0, total - events_det_indexed),
            },
            "latest": dict(latest) if latest else None,
        }

    def search(
        self,
        *,
        query: str = "",
        camera_name: Optional[str] = None,
        trigger_type: Optional[str] = None,
        shape_name: Optional[str] = None,
        dominant_color: Optional[str] = None,
        detection_classes: Optional[Sequence[str]] = None,
        detection_color: Optional[str] = None,
        min_confidence: Optional[float] = None,
        min_area: Optional[float] = None,
        start_ts: Optional[int] = None,
        end_ts: Optional[int] = None,
        limit: int = 25,
    ) -> List[EventSearchResult]:
        q = (query or "").strip()
        limit = max(1, min(int(limit or 25), 5000))
        start_ts_i = _safe_int(start_ts)
        end_ts_i = _safe_int(end_ts)
        cam = (camera_name or "").strip()
        cam_id_guess: Optional[str] = None
        if cam:
            # If the caller passed a camera name, try to map it to an id so we still match rows
            # that only have camera_id populated. If they passed an id already, keep it.
            if "-" in cam and len(cam) >= 16:
                cam_id_guess = cam
            else:
                try:
                    for cid, nm in (self._load_camera_name_map() or {}).items():
                        if isinstance(nm, str) and nm.strip().lower() == cam.lower():
                            cam_id_guess = str(cid)
                            break
                except Exception:
                    cam_id_guess = None
        trig = (trigger_type or "").strip().lower()
        shape = (shape_name or "").strip().lower()
        color = (dominant_color or "").strip().lower()
        det_color = (detection_color or "").strip().lower()
        min_conf = _safe_float(min_confidence)
        min_ar = _safe_float(min_area)
        cls_list = [str(c).strip().lower() for c in (detection_classes or []) if isinstance(c, (str, int, float)) and str(c).strip()]

        where: List[str] = []
        params: List[Any] = []
        if start_ts_i is not None:
            where.append("e.captured_ts >= ?")
            params.append(int(start_ts_i))
        if end_ts_i is not None:
            where.append("e.captured_ts <= ?")
            params.append(int(end_ts_i))
        if cam:
            # Accept either camera name or camera id (uuid) in the same field.
            where.append("(LOWER(COALESCE(e.camera_name,'')) = ? OR COALESCE(e.camera_id,'') = ?)")
            params.append(cam.lower())
            params.append(cam_id_guess or cam)
        if trig:
            where.append("LOWER(COALESCE(e.trigger_type,'')) = ?")
            params.append(trig)
        if shape:
            where.append("LOWER(COALESCE(e.shape_name,'')) = ?")
            params.append(shape)
        if color:
            where.append("LOWER(COALESCE(e.dominant_color,'')) = ?")
            params.append(color)
        # Detection-level filters (override-aware). For class-only filters, keep a fallback to
        # the legacy event-level `events.detection_classes` field so older indexed rows still match.
        if cls_list:
            for c in cls_list:
                where.append(
                    "("
                    "EXISTS ("
                    "  SELECT 1 FROM event_detections d "
                    "  LEFT JOIN event_detection_overrides o ON o.detection_id = d.detection_id "
                    "  WHERE d.event_id = e.id AND LOWER(COALESCE(o.override_class, d.class, '')) = ?"
                    ")"
                    " OR (',' || LOWER(COALESCE(e.detection_classes,'')) || ',') LIKE ?"
                    ")"
                )
                params.append(c)
                params.append(f"%,{c},%")

        if det_color or (min_conf is not None) or (min_ar is not None):
            det_where: List[str] = []
            det_params: List[Any] = []
            if det_color:
                det_where.append("LOWER(COALESCE(o.override_color, d.color, '')) = ?")
                det_params.append(det_color)
            if min_conf is not None:
                det_where.append("COALESCE(d.confidence, 0) >= ?")
                det_params.append(float(min_conf))
            if min_ar is not None:
                det_where.append("COALESCE(d.area, 0) >= ?")
                det_params.append(float(min_ar))
            where.append(
                "EXISTS (SELECT 1 FROM event_detections d "
                "LEFT JOIN event_detection_overrides o ON o.detection_id = d.detection_id "
                "WHERE d.event_id = e.id AND " + " AND ".join(det_where) + ")"
            )
            params.extend(det_params)

        where_sql = ("WHERE " + " AND ".join(where)) if where else ""

        with self._connect() as conn:
            rows: List[sqlite3.Row] = []
            if q:
                # Try FTS first
                try:
                    # NOTE: When there are no filters, we must still include a WHERE clause.
                    # The previous version emitted "... JOIN ... AND events_fts MATCH ?" which is invalid SQL,
                    # causing silent fallback to LIKE and returning 0 when captions/classes are empty.
                    where_fts = (where_sql + " AND events_fts MATCH ?") if where_sql else "WHERE events_fts MATCH ?"
                    sql = f"""
                    SELECT e.*
                    FROM events_fts
                    JOIN events e ON e.rowid = events_fts.rowid
                    {where_fts}
                    ORDER BY e.captured_ts DESC
                    LIMIT ?;
                    """
                    rows = list(conn.execute(sql, [*params, q, limit]).fetchall())
                except Exception:
                    # LIKE fallback
                    like = f"%{q.lower()}%"
                    sql = f"""
                    SELECT e.*
                    FROM events e
                    {where_sql}
                    {"AND" if where_sql else "WHERE"} (
                      LOWER(COALESCE(e.caption,'')) LIKE ?
                      OR LOWER(COALESCE(e.tags,'')) LIKE ?
                      OR LOWER(COALESCE(e.detection_classes,'')) LIKE ?
                      OR LOWER(COALESCE(e.shape_name,'')) LIKE ?
                    )
                    ORDER BY e.captured_ts DESC
                    LIMIT ?;
                    """
                    rows = list(conn.execute(sql, [*params, like, like, like, like, limit]).fetchall())
            else:
                sql = f"""
                SELECT e.*
                FROM events e
                {where_sql}
                ORDER BY e.captured_ts DESC
                LIMIT ?;
                """
                rows = list(conn.execute(sql, [*params, limit]).fetchall())

        out: List[EventSearchResult] = []
        for r in rows:
            meta: Dict[str, Any] = {}
            try:
                if r["json_metadata"]:
                    meta = json.loads(r["json_metadata"])
            except Exception:
                meta = {}
            tags = [t.strip() for t in (r["tags"] or "").split(",") if t.strip()]
            classes = [t.strip() for t in (r["detection_classes"] or "").split(",") if t.strip()]
            out.append(
                EventSearchResult(
                    id=str(r["id"]),
                    file_path=str(r["file_path"]),
                    thumb_path=str(r["thumb_path"]) if r["thumb_path"] else None,
                    camera_id=str(r["camera_id"]) if r["camera_id"] else None,
                    camera_name=str(r["camera_name"]) if r["camera_name"] else None,
                    captured_ts=int(r["captured_ts"]) if r["captured_ts"] is not None else None,
                    captured_at=str(r["captured_at"]) if r["captured_at"] else None,
                    trigger_type=str(r["trigger_type"]) if r["trigger_type"] else None,
                    shape_type=str(r["shape_type"]) if r["shape_type"] else None,
                    shape_name=str(r["shape_name"]) if r["shape_name"] else None,
                    dominant_color=str(r["dominant_color"]) if r["dominant_color"] else None,
                    caption=str(r["caption"]) if r["caption"] else None,
                    tags=tags,
                    detection_classes=classes,
                    metadata=meta,
                    media_type=self._safe_row_get(r, "media_type", "image"),
                )
            )
        return out

    def count(
        self,
        *,
        camera_name: Optional[str] = None,
        start_ts: Optional[int] = None,
        end_ts: Optional[int] = None,
        detection_classes: Optional[Sequence[str]] = None,
        detection_color: Optional[str] = None,
        min_confidence: Optional[float] = None,
        min_area: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Return aggregate counts without returning a timeline.

        - event_count: number of distinct events matching detection filters
        - detection_count: number of detections matching detection filters
        - events_total_in_range: total events matching camera/time window regardless of detections
        - events_detection_indexed_in_range: events in range where detections indexing was attempted
        """
        start_ts_i = _safe_int(start_ts)
        end_ts_i = _safe_int(end_ts)
        cam = (camera_name or "").strip()
        cam_id_guess: Optional[str] = None
        if cam:
            if "-" in cam and len(cam) >= 16:
                cam_id_guess = cam
            else:
                try:
                    for cid, nm in (self._load_camera_name_map() or {}).items():
                        if isinstance(nm, str) and nm.strip().lower() == cam.lower():
                            cam_id_guess = str(cid)
                            break
                except Exception:
                    cam_id_guess = None
        cls_list = [str(c).strip().lower() for c in (detection_classes or []) if str(c).strip()]
        det_color = (detection_color or "").strip().lower()
        min_conf = _safe_float(min_confidence)
        min_ar = _safe_float(min_area)

        # Base event WHERE (camera/time)
        ev_where: List[str] = []
        ev_params: List[Any] = []
        if start_ts_i is not None:
            ev_where.append("e.captured_ts >= ?")
            ev_params.append(int(start_ts_i))
        if end_ts_i is not None:
            ev_where.append("e.captured_ts <= ?")
            ev_params.append(int(end_ts_i))
        if cam:
            # Accept either camera name or camera id (uuid) in the same field.
            ev_where.append("(LOWER(COALESCE(e.camera_name,'')) = ? OR COALESCE(e.camera_id,'') = ?)")
            ev_params.append(cam.lower())
            ev_params.append(cam_id_guess or cam)
        ev_where_sql = ("WHERE " + " AND ".join(ev_where)) if ev_where else ""

        # Detection WHERE (effective override-aware)
        det_where: List[str] = []
        det_params: List[Any] = []
        if cls_list:
            det_where.append("(" + " OR ".join(["LOWER(COALESCE(o.override_class, d.class, '')) = ?"] * len(cls_list)) + ")")
            det_params.extend(cls_list)
        if det_color:
            det_where.append("LOWER(COALESCE(o.override_color, d.color, '')) = ?")
            det_params.append(det_color)
        if min_conf is not None:
            det_where.append("COALESCE(d.confidence, 0) >= ?")
            det_params.append(float(min_conf))
        if min_ar is not None:
            det_where.append("COALESCE(d.area, 0) >= ?")
            det_params.append(float(min_ar))
        det_where_sql = (" AND " + " AND ".join(det_where)) if det_where else ""

        with self._connect() as conn:
            # Totals in range
            try:
                events_total_in_range = int(conn.execute(f"SELECT COUNT(1) AS c FROM events e {ev_where_sql};", ev_params).fetchone()["c"])
            except Exception:
                events_total_in_range = 0
            try:
                events_detection_indexed_in_range = int(
                    conn.execute(
                        f"SELECT COUNT(1) AS c FROM events e {ev_where_sql} {'AND' if ev_where_sql else 'WHERE'} e.detections_indexed_at IS NOT NULL;",
                        ev_params,
                    ).fetchone()["c"]
                )
            except Exception:
                events_detection_indexed_in_range = 0

            # Detection aggregates
            if det_where:
                sql = (
                    "SELECT COUNT(1) AS detection_count, COUNT(DISTINCT d.event_id) AS event_count "
                    "FROM event_detections d "
                    "JOIN events e ON e.id = d.event_id "
                    "LEFT JOIN event_detection_overrides o ON o.detection_id = d.detection_id "
                    f"{ev_where_sql}"
                    f"{det_where_sql};"
                )
                row = conn.execute(sql, [*ev_params, *det_params]).fetchone()
                detection_count = int(row["detection_count"] or 0) if row else 0
                event_count = int(row["event_count"] or 0) if row else 0
            else:
                # No detection filters: just count events in range.
                detection_count = 0
                event_count = events_total_in_range

        return {
            "events_total_in_range": events_total_in_range,
            "events_detection_indexed_in_range": events_detection_indexed_in_range,
            "events_detection_not_indexed_in_range": max(0, int(events_total_in_range) - int(events_detection_indexed_in_range)),
            "event_count": event_count,
            "detection_count": detection_count,
            "filters": {
                "camera_name": cam or None,
                "start_ts": start_ts_i,
                "end_ts": end_ts_i,
                "detection_classes": cls_list,
                "detection_color": det_color or None,
                "min_confidence": min_conf,
                "min_area": min_ar,
            },
        }

    def count_unique_vehicles(
        self,
        *,
        camera_name: Optional[str] = None,
        start_ts: Optional[int] = None,
        end_ts: Optional[int] = None,
        detection_classes: Optional[Sequence[str]] = None,
        min_confidence: float = 0.25,
        max_gap_seconds: int = 12,
        max_distance_px: float = 140.0,
        limit_detections: int = 50_000,
    ) -> Dict[str, Any]:
        """
        Count unique vehicles by linking per-capture detections into simple tracks.

        This is a best-effort cross-capture tracker intended for Motion Watch style bursts.
        It is not a full SORT/DeepSORT pipeline; it is tuned for robustness and simplicity.
        """
        start_ts_i = _safe_int(start_ts)
        end_ts_i = _safe_int(end_ts)
        cam_ref = (camera_name or "").strip()

        # Normalize class filter; default to common vehicle classes
        cls_list = [
            str(c).strip().lower()
            for c in (detection_classes or ["car", "truck", "bus", "motorcycle"])
            if isinstance(c, (str, int, float)) and str(c).strip()
        ]
        cls_list = list(dict.fromkeys([c for c in cls_list if c]))[:16]
        min_conf = float(min_confidence or 0.0)
        max_gap = max(1, int(max_gap_seconds or 12))
        max_dist = float(max_distance_px or 140.0)
        limit_dets = max(1, min(int(limit_detections or 50_000), 250_000))

        # Reuse existing coverage counters
        coverage = self.count(
            camera_name=cam_ref or None,
            start_ts=start_ts_i,
            end_ts=end_ts_i,
            detection_classes=cls_list,
            min_confidence=min_conf,
        )

        cam_id_guess: Optional[str] = None
        if cam_ref:
            if "-" in cam_ref and len(cam_ref) >= 16:
                cam_id_guess = cam_ref
            else:
                try:
                    for cid, nm in (self._load_camera_name_map() or {}).items():
                        if isinstance(nm, str) and nm.strip().lower() == cam_ref.lower():
                            cam_id_guess = str(cid)
                            break
                except Exception:
                    cam_id_guess = None

        # Query detections in time order
        where: List[str] = []
        params: List[Any] = []
        if start_ts_i is not None:
            where.append("e.captured_ts >= ?")
            params.append(int(start_ts_i))
        if end_ts_i is not None:
            where.append("e.captured_ts <= ?")
            params.append(int(end_ts_i))
        if cam_ref:
            where.append("(LOWER(COALESCE(e.camera_name,'')) = ? OR COALESCE(e.camera_id,'') = ?)")
            params.append(cam_ref.lower())
            params.append(cam_id_guess or cam_ref)

        # Class filter (override-aware)
        if cls_list:
            where.append("(" + " OR ".join(["LOWER(COALESCE(o.override_class, d.class, '')) = ?"] * len(cls_list)) + ")")
            params.extend(cls_list)

        where_sql = ("WHERE " + " AND ".join(where)) if where else ""
        sql = f"""
        SELECT
          e.id AS event_id,
          e.captured_ts AS captured_ts,
          d.detection_id AS detection_id,
          LOWER(COALESCE(o.override_class, d.class, '')) AS class,
          COALESCE(d.confidence, 0) AS confidence,
          COALESCE(d.bbox_x, 0) AS bbox_x,
          COALESCE(d.bbox_y, 0) AS bbox_y,
          COALESCE(d.bbox_w, 0) AS bbox_w,
          COALESCE(d.bbox_h, 0) AS bbox_h
        FROM event_detections d
        JOIN events e ON e.id = d.event_id
        LEFT JOIN event_detection_overrides o ON o.detection_id = d.detection_id
        {where_sql}
          AND COALESCE(d.confidence, 0) >= ?
        ORDER BY e.captured_ts ASC, d.event_id ASC, d.det_idx ASC
        LIMIT ?;
        """
        rows: List[sqlite3.Row] = []
        with self._connect() as conn:
            rows = list(conn.execute(sql, [*params, float(min_conf), int(limit_dets)]).fetchall())

        # Track structure
        class Track:
            __slots__ = ("track_id", "last_ts", "last_cx", "last_cy", "classes", "events")

            def __init__(self, track_id: int, ts: int, cx: float, cy: float, cls: str, event_id: str):
                self.track_id = track_id
                self.last_ts = ts
                self.last_cx = cx
                self.last_cy = cy
                self.classes: Dict[str, int] = {cls: 1} if cls else {}
                self.events: set = {event_id} if event_id else set()

            def add(self, ts: int, cx: float, cy: float, cls: str, event_id: str) -> None:
                self.last_ts = ts
                self.last_cx = cx
                self.last_cy = cy
                if cls:
                    self.classes[cls] = self.classes.get(cls, 0) + 1
                if event_id:
                    self.events.add(event_id)

            def primary_class(self) -> Optional[str]:
                if not self.classes:
                    return None
                return max(self.classes.items(), key=lambda kv: kv[1])[0]

        active: List[Track] = []
        all_tracks: List[Track] = []
        next_id = 1

        def _dist(a: Track, cx: float, cy: float) -> float:
            dx = float(a.last_cx - cx)
            dy = float(a.last_cy - cy)
            return math.sqrt(dx * dx + dy * dy)

        for r in rows:
            try:
                ts = int(r["captured_ts"] or 0)
            except Exception:
                ts = 0
            try:
                x = float(r["bbox_x"] or 0.0)
                y = float(r["bbox_y"] or 0.0)
                w = float(r["bbox_w"] or 0.0)
                h = float(r["bbox_h"] or 0.0)
            except Exception:
                x, y, w, h = 0.0, 0.0, 0.0, 0.0
            cx = x + max(0.0, w) / 2.0
            cy = y + max(0.0, h) / 2.0
            cls = str(r["class"] or "").strip().lower()
            event_id = str(r["event_id"] or "")

            # Drop stale tracks
            if ts and active:
                active = [t for t in active if (ts - int(t.last_ts or 0)) <= max_gap]

            # Adaptive threshold: allow larger moves for larger boxes
            area = max(0.0, w) * max(0.0, h)
            thresh = max_dist + min(120.0, 0.15 * math.sqrt(area) if area > 0 else 0.0)

            best: Optional[Track] = None
            best_d = thresh
            for t in active:
                d = _dist(t, cx, cy)
                if d <= best_d:
                    best = t
                    best_d = d

            if best is None:
                tr = Track(next_id, ts, cx, cy, cls, event_id)
                next_id += 1
                active.append(tr)
                all_tracks.append(tr)
            else:
                best.add(ts, cx, cy, cls, event_id)

        breakdown: Dict[str, int] = {}
        for t in all_tracks:
            pc = t.primary_class()
            if pc:
                breakdown[pc] = breakdown.get(pc, 0) + 1

        return {
            "unique_vehicle_count": len(all_tracks),
            "tracks": len(all_tracks),
            "class_breakdown": breakdown,
            "detections_used": len(rows),
            "filters": {
                "camera_name": cam_ref or None,
                "start_ts": start_ts_i,
                "end_ts": end_ts_i,
                "detection_classes": cls_list,
                "min_confidence": min_conf,
                "max_gap_seconds": max_gap,
                "max_distance_px": max_dist,
                "limit_detections": limit_dets,
            },
            "coverage": coverage,
        }

    def read_file_base64(self, file_path: str, *, max_bytes: int = 350_000) -> Optional[str]:
        """
        Safe base64 file reader for small JPEG crops/thumbnails.
        """
        try:
            p = Path(str(file_path))
            if not p.is_absolute():
                p = (self.project_root / p).resolve()
            if not p.exists() or not p.is_file():
                return None
            raw = p.read_bytes()
            if len(raw) > int(max_bytes or 350_000):
                return None
            return base64.b64encode(raw).decode("utf-8")
        except Exception:
            return None

    def read_thumb_base64(self, thumb_path: str) -> Optional[str]:
        try:
            p = Path(thumb_path)
            if not p.is_absolute():
                p = (self.project_root / p).resolve()
            if not p.exists() or not p.is_file():
                return None
            raw = p.read_bytes()
            # Hard cap (avoid huge responses)
            if len(raw) > 350_000:
                return None
            return base64.b64encode(raw).decode("utf-8")
        except Exception:
            return None


