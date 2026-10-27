"""
Disk-space manager for KnoxNet VMS.

Monitors recording and capture directories and prunes the oldest files
when usage exceeds a configurable threshold.  Operates in two modes:

* **Rolling delete** (steady-state): once usage exceeds
  ``max_usage_percent``, the oldest single file is deleted each poll
  cycle so that new recordings replace old ones at roughly the same
  pace -- like a ring buffer.

* **Bulk cleanup** (catch-up): used at startup and when settings
  change to bring usage from wherever it is down to
  ``target_usage_percent`` quickly.

An emergency threshold (``critical_usage_percent``, default 95%)
triggers recording pause, aggressive bulk prune, then resume.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import threading
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from .paths import get_data_dir, get_motion_watch_dir, get_recordings_dir

logger = logging.getLogger(__name__)

_SETTINGS_FILE = "storage_settings.json"

_MIN_FILE_AGE_SECONDS = 60

# Only delete files with these extensions.  This is a hard safety guard:
# no matter what directory the StorageManager scans, it will NEVER touch
# files that aren't video clips, image captures, or related metadata.
_SAFE_EXTENSIONS: frozenset = frozenset({
    ".mp4", ".mkv", ".avi", ".mov", ".ts", ".m4v", ".flv", ".webm",  # video
    ".fmp4",                                                          # fragmented mp4
    ".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff",               # images
    ".m4s", ".mpd",                                                   # DASH segments
})

_DEFAULT_SETTINGS: Dict[str, Any] = {
    "enabled": True,
    "max_usage_percent": 85,
    "target_usage_percent": 75,
    "critical_usage_percent": 95,
    "check_interval_seconds": 300,
    "recordings_dir": "",
    "captures_dir": "",
}


class StorageManager:
    """Background service that keeps disk usage within configured limits."""

    def __init__(
        self,
        *,
        settings_path: Optional[Path] = None,
        pause_recordings_cb: Optional[Callable[[], None]] = None,
        resume_recordings_cb: Optional[Callable[[], None]] = None,
        get_recording_dirs_cb: Optional[Callable[[], List[Path]]] = None,
    ):
        self._settings_path = settings_path or (get_data_dir() / _SETTINGS_FILE)
        self._settings: Dict[str, Any] = dict(_DEFAULT_SETTINGS)
        self._load_settings()

        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._last_usage_pct: float = 0.0

        self._pause_recordings_cb = pause_recordings_cb
        self._resume_recordings_cb = resume_recordings_cb
        self._get_recording_dirs_cb = get_recording_dirs_cb

    # ------------------------------------------------------------------
    # Settings persistence
    # ------------------------------------------------------------------

    def _load_settings(self) -> None:
        try:
            if self._settings_path.exists():
                with open(self._settings_path, "r") as fh:
                    data = json.load(fh)
                if isinstance(data, dict):
                    for k in _DEFAULT_SETTINGS:
                        if k in data:
                            self._settings[k] = data[k]
        except Exception as exc:
            logger.warning("Could not load storage settings: %s", exc)

    def _save_settings(self) -> None:
        try:
            self._settings_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self._settings_path, "w") as fh:
                json.dump(self._settings, fh, indent=2)
        except Exception as exc:
            logger.warning("Could not save storage settings: %s", exc)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def settings(self) -> Dict[str, Any]:
        return dict(self._settings)

    def update_settings(self, patch: Dict[str, Any]) -> Dict[str, Any]:
        was_enabled = self._settings.get("enabled")
        for k, v in patch.items():
            if k in _DEFAULT_SETTINGS:
                self._settings[k] = v
        self._save_settings()

        if not was_enabled and self._settings.get("enabled"):
            threading.Thread(
                target=self._bulk_cleanup, daemon=True, name="StorageManager-catchup"
            ).start()

        return self.settings

    def get_status(self) -> Dict[str, Any]:
        """Return current disk usage for the managed directories."""
        rec_dir = self._recordings_dir()
        cap_dir = self._captures_dir()

        result: Dict[str, Any] = {"enabled": self._settings["enabled"]}

        dirs_to_check = [("recordings", rec_dir), ("captures", cap_dir)]
        self._add_custom_capture_dirs(dirs_to_check)
        self._add_custom_recording_dirs(dirs_to_check)

        for label, d in dirs_to_check:
            try:
                usage = shutil.disk_usage(d)
                result[label] = {
                    "path": str(d),
                    "total_gb": round(usage.total / (1 << 30), 2),
                    "used_gb": round(usage.used / (1 << 30), 2),
                    "free_gb": round(usage.free / (1 << 30), 2),
                    "used_percent": round(usage.used / max(usage.total, 1) * 100, 1),
                }
            except Exception:
                result[label] = {"path": str(d), "error": "unavailable"}

        return result

    # ------------------------------------------------------------------
    # Background thread
    # ------------------------------------------------------------------

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, daemon=True, name="StorageManager")
        self._thread.start()
        logger.info("StorageManager background thread started")

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=5)
        logger.info("StorageManager stopped")

    def _run(self) -> None:
        try:
            if self._settings.get("enabled"):
                self._bulk_cleanup()
        except Exception as exc:
            logger.error("StorageManager startup cleanup error: %s", exc)

        while not self._stop_event.is_set():
            try:
                if self._settings.get("enabled"):
                    self._rolling_cycle()
            except Exception as exc:
                logger.error("StorageManager cycle error: %s", exc)
            self._stop_event.wait(timeout=self._adaptive_interval())

    def _adaptive_interval(self) -> float:
        """Return poll interval in seconds, shorter when usage is near threshold."""
        max_pct = float(self._settings.get("max_usage_percent", 85))
        pct = self._last_usage_pct

        if pct >= max_pct:
            return 10
        if pct >= max_pct - 5:
            return 30
        if pct >= max_pct - 15:
            return 60
        return max(30, int(self._settings.get("check_interval_seconds", 300)))

    # ------------------------------------------------------------------
    # Directory helpers
    # ------------------------------------------------------------------

    def _recordings_dir(self) -> Path:
        custom = (self._settings.get("recordings_dir") or "").strip()
        if custom:
            p = Path(custom).expanduser()
            return p.resolve() if p.is_absolute() else (get_recordings_dir().parent / p).resolve()
        return get_recordings_dir()

    def _captures_dir(self) -> Path:
        custom = (self._settings.get("captures_dir") or "").strip()
        if custom:
            p = Path(custom).expanduser()
            return p.resolve() if p.is_absolute() else (get_motion_watch_dir().parent / p).resolve()
        return get_motion_watch_dir()

    def _managed_dirs(self) -> List[tuple]:
        """Build the list of (label, Path) directories to manage."""
        dirs: List[tuple] = [
            ("recordings", self._recordings_dir()),
            ("captures", self._captures_dir()),
        ]
        self._add_custom_capture_dirs(dirs)
        self._add_custom_recording_dirs(dirs)
        return dirs

    @staticmethod
    def _add_custom_capture_dirs(dirs: List[tuple]) -> None:
        try:
            from .event_index_service import _discover_custom_capture_dirs
            seen = {str(d.resolve()) for _, d in dirs}
            for custom in _discover_custom_capture_dirs():
                key = str(custom.resolve())
                if key not in seen:
                    dirs.append((f"custom:{custom.name}", custom))
                    seen.add(key)
        except Exception:
            pass

    def _add_custom_recording_dirs(self, dirs: List[tuple]) -> None:
        if not self._get_recording_dirs_cb:
            return
        try:
            seen = {str(d.resolve()) for _, d in dirs}
            for rdir in self._get_recording_dirs_cb():
                resolved = rdir.resolve()
                # Never add a bare drive root (e.g. D:\) as a managed
                # directory — rglob on a drive root scans EVERYTHING.
                # Instead, scan only subdirectories that look like
                # recording output (camera-name or UUID directories).
                if resolved.parent == resolved:
                    # This IS a drive root.  Scan its immediate children
                    # for directories that contain .mp4 files.
                    try:
                        for child in resolved.iterdir():
                            if not child.is_dir():
                                continue
                            # Quick sniff: does it have any mp4 files?
                            has_media = any(child.rglob("*.mp4"))
                            if not has_media:
                                continue
                            ckey = str(child.resolve())
                            if ckey not in seen:
                                dirs.append((f"recording:{child.name}", child))
                                seen.add(ckey)
                    except Exception:
                        pass
                    continue

                key = str(resolved)
                if key not in seen:
                    dirs.append((f"recording:{rdir.name}", resolved))
                    seen.add(key)
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Rolling cleanup (steady-state)
    # ------------------------------------------------------------------

    def _rolling_cycle(self) -> None:
        """One poll cycle: check usage and delete the single oldest file if over threshold."""
        max_pct = float(self._settings.get("max_usage_percent", 85))
        critical_pct = float(self._settings.get("critical_usage_percent", 95))

        for label, directory in self._managed_dirs():
            try:
                if not directory.exists():
                    continue
                usage = shutil.disk_usage(directory)
                used_pct = usage.used / max(usage.total, 1) * 100
                self._last_usage_pct = max(self._last_usage_pct, used_pct)

                if used_pct >= critical_pct:
                    self._handle_critical(label, directory, usage, used_pct)
                elif used_pct > max_pct:
                    self._delete_oldest_file(directory, label, used_pct)
            except Exception as exc:
                logger.error("StorageManager rolling cleanup error for %s: %s", label, exc)

        self._last_usage_pct = self._current_max_usage()

    def _current_max_usage(self) -> float:
        """Return the highest partition usage across all managed dirs."""
        worst = 0.0
        for _, directory in self._managed_dirs():
            try:
                if not directory.exists():
                    continue
                usage = shutil.disk_usage(directory)
                pct = usage.used / max(usage.total, 1) * 100
                worst = max(worst, pct)
            except Exception:
                pass
        return worst

    @staticmethod
    def _is_managed_file(path: Path) -> bool:
        """Return True only if the file is a video/image type safe to delete."""
        return path.suffix.lower() in _SAFE_EXTENSIONS

    def _delete_oldest_file(self, directory: Path, label: str, used_pct: float) -> bool:
        """Delete the single oldest eligible file in *directory*. Returns True if a file was deleted."""
        now = time.time()
        oldest_mtime = None
        oldest_path = None

        for entry in directory.rglob("*"):
            if not entry.is_file():
                continue
            if not self._is_managed_file(entry):
                continue
            try:
                st = entry.stat()
                if (now - st.st_mtime) < _MIN_FILE_AGE_SECONDS:
                    continue
                if oldest_mtime is None or st.st_mtime < oldest_mtime:
                    oldest_mtime = st.st_mtime
                    oldest_path = entry
            except OSError:
                continue

        if oldest_path is None:
            return False

        try:
            oldest_path.unlink()
            logger.info(
                "StorageManager: rolling delete in %s (usage %.1f%%): %s",
                label, used_pct, oldest_path,
            )
            self._cleanup_empty_parents(oldest_path.parent, directory)
            return True
        except OSError as exc:
            logger.debug("Could not delete %s: %s", oldest_path, exc)
            return False

    # ------------------------------------------------------------------
    # Emergency handling
    # ------------------------------------------------------------------

    def _handle_critical(self, label: str, directory: Path, usage, used_pct: float) -> None:
        """Usage is at critical level -- pause recordings, bulk prune aggressively, resume."""
        logger.critical(
            "StorageManager: %s at %.1f%% CRITICAL (threshold %.0f%%), "
            "pausing recordings and pruning aggressively",
            label, used_pct, float(self._settings.get("critical_usage_percent", 95)),
        )

        if self._pause_recordings_cb:
            try:
                self._pause_recordings_cb()
            except Exception as exc:
                logger.error("StorageManager: failed to pause recordings: %s", exc)

        emergency_target = min(
            float(self._settings.get("target_usage_percent", 75)),
            70.0,
        )
        self._prune_oldest(directory, usage.total, emergency_target)

        if self._resume_recordings_cb:
            try:
                self._resume_recordings_cb()
            except Exception as exc:
                logger.error("StorageManager: failed to resume recordings: %s", exc)

    # ------------------------------------------------------------------
    # Bulk cleanup (catch-up at startup or settings change)
    # ------------------------------------------------------------------

    def _bulk_cleanup(self) -> None:
        """Prune all managed directories down to target_usage_percent."""
        max_pct = float(self._settings.get("max_usage_percent", 85))
        target_pct = float(self._settings.get("target_usage_percent", 75))

        for label, directory in self._managed_dirs():
            try:
                if not directory.exists():
                    continue
                usage = shutil.disk_usage(directory)
                used_pct = usage.used / max(usage.total, 1) * 100
                if used_pct <= max_pct:
                    continue
                logger.info(
                    "StorageManager: %s at %.1f%% (threshold %.0f%%), bulk pruning to %.0f%%",
                    label, used_pct, max_pct, target_pct,
                )
                self._prune_oldest(directory, usage.total, target_pct)
            except Exception as exc:
                logger.error("StorageManager bulk cleanup error for %s: %s", label, exc)

    # ------------------------------------------------------------------
    # Prune logic (shared by bulk and emergency)
    # ------------------------------------------------------------------

    @staticmethod
    def _prune_oldest(directory: Path, total_bytes: int, target_pct: float) -> int:
        """Delete oldest files until partition usage drops below *target_pct*.

        Only deletes video/image files (see ``_SAFE_EXTENSIONS``).
        Skips files modified within the last ``_MIN_FILE_AGE_SECONDS`` to
        protect actively-written segments.  Stops early if all eligible
        files are exhausted (external data may be consuming the disk).

        Returns count of deleted files.
        """
        now = time.time()
        files: List[tuple] = []
        for entry in directory.rglob("*"):
            if entry.is_file() and entry.suffix.lower() in _SAFE_EXTENSIONS:
                try:
                    st = entry.stat()
                    if (now - st.st_mtime) < _MIN_FILE_AGE_SECONDS:
                        continue
                    files.append((st.st_mtime, entry))
                except OSError:
                    continue
        files.sort()

        deleted = 0
        for _mtime, fpath in files:
            usage = shutil.disk_usage(directory)
            if usage.used / max(total_bytes, 1) * 100 <= target_pct:
                break
            try:
                fpath.unlink()
                deleted += 1
                StorageManager._cleanup_empty_parents(fpath.parent, directory)
            except OSError as exc:
                logger.debug("Could not delete %s: %s", fpath, exc)

        if deleted:
            logger.info("StorageManager: pruned %d file(s) from %s", deleted, directory)

        final_usage = shutil.disk_usage(directory)
        final_pct = final_usage.used / max(total_bytes, 1) * 100
        if final_pct > target_pct:
            logger.warning(
                "StorageManager: all eligible files in %s deleted but partition "
                "usage is still %.1f%% (target %.0f%%) -- external data is "
                "consuming disk space",
                directory, final_pct, target_pct,
            )

        return deleted

    @staticmethod
    def _cleanup_empty_parents(parent: Path, root: Path) -> None:
        """Remove empty date-based parent directories up to *root*."""
        try:
            while parent != root:
                if not any(parent.iterdir()):
                    parent.rmdir()
                    parent = parent.parent
                else:
                    break
        except OSError:
            pass
