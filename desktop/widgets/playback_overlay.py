"""
Full video-player overlay for KnoxNet VMS camera widgets.

Renders a translucent bottom dock with:
- Filled progress bar with draggable scrubber handle
- Elapsed / total time display
- Play / pause / skip / speed / frame-step controls
- Keyboard shortcuts (space, arrows, comma/period)
- Thumbnail preview on hover (background-decoded, cached)
- Color-coded event markers on the timeline
- Click-on-video to toggle pause

Playback engine reads MP4 segments directly from disk (lowest CPU).
Falls back to MediaMTX playback API when configured.
"""

from __future__ import annotations

import calendar as _cal_mod
import math
import os
import re
import select
import subprocess
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt, QTimer, Signal
from PySide6.QtGui import QColor, QFont, QImage, QPainter, QPen, QPainterPath, QPixmap
from PySide6.QtWidgets import QWidget

_PLAYBACK_PORT = 9996

_EVENT_COLORS = {
    "zone": QColor(56, 189, 248),
    "line": QColor(74, 222, 128),
    "tag": QColor(251, 191, 36),
    "detection": QColor(248, 113, 113),
}

_DOCK_H = 78
_DOCK_MARGIN = 10
_DOCK_RADIUS = 12
_RAIL_H = 14
_RAIL_Y_OFFSET = 48
_HANDLE_R = 9
_THUMB_W = 200
_THUMB_H = 112


# ── Segment timeline ─────────────────────────────────────────────────

@dataclass
class Segment:
    path: str
    start_ts: float
    end_ts: float
    duration_s: float
    fps: float = 25.0


def _parse_segment_ts(filepath: Path) -> Optional[float]:
    """Extract a local-time timestamp from recording path structure: .../YYYY-MM-DD/HH-MM-SS-ffffff.mp4

    MediaMTX writes filenames in local time, so we interpret them as such.
    """
    try:
        date_part = filepath.parent.name
        time_part = filepath.stem
        dt_str = f"{date_part} {time_part}"
        for fmt in ("%Y-%m-%d %H-%M-%S-%f", "%Y-%m-%d %H-%M-%S", "%Y-%m-%d_%H-%M-%S-%f"):
            try:
                dt = datetime.strptime(dt_str, fmt)
                return dt.timestamp()
            except ValueError:
                continue
        # Legacy flat naming: YYYY-MM-DD_HH-MM-SS-ffffff.mp4
        for fmt in ("%Y-%m-%d_%H-%M-%S-%f", "%Y-%m-%d_%H-%M-%S"):
            try:
                dt = datetime.strptime(filepath.stem, fmt)
                return dt.timestamp()
            except ValueError:
                continue
    except Exception:
        pass
    return None


def _fast_duration(path: Path, start_ts: float = 0) -> Optional[float]:
    """Get duration via ffprobe (fast, no decode).

    Falls back to ``mtime - start_ts`` (very accurate for recordings
    whose filename encodes the start time and mtime reflects the end).
    Last resort: estimate from file size.
    """
    try:
        result = subprocess.run(
            ["ffprobe", "-v", "quiet", "-show_entries", "format=duration",
             "-of", "csv=p=0", str(path)],
            capture_output=True, text=True, timeout=3,
        )
        val = result.stdout.strip()
        if val:
            return float(val)
    except Exception:
        pass
    # Best fallback: mtime is when the file finished writing (= segment end).
    # duration = mtime - start_ts gives an accurate result when start_ts
    # was parsed from the filename.
    if start_ts > 0:
        try:
            mtime = path.stat().st_mtime
            dur = mtime - start_ts
            if dur > 0:
                return dur
        except Exception:
            pass
    # Last resort: rough bitrate estimate (assumes ~1.5 Mbps)
    try:
        size = path.stat().st_size
        return size / 187_500
    except Exception:
        return None


def _file_belongs_to_camera(filepath: Path, camera_id: str, safe_name: str) -> bool:
    """Check whether an MP4 file can be attributed to a specific camera.

    Matches if camera_id or safe_name appears as:
    - a directory component anywhere in the path, OR
    - a prefix/substring of the filename (motion-watch clips embed camera_id).
    """
    parts = filepath.parts
    fname = filepath.name
    if camera_id:
        if camera_id in parts or camera_id in fname:
            return True
    if safe_name:
        if safe_name in parts:
            return True
    return False


def build_segment_timeline(
    camera_id: str,
    camera_name: str = "",
    extra_dirs: Optional[List[str]] = None,
) -> List[Segment]:
    """Scan recording directories and build a sorted list of segments.

    Handles both continuous recording segments and motion-watch clips,
    even if they live in the same folder or overlap in time.
    Uses ffprobe for fast duration probing (no OpenCV decode).

    ``extra_dirs`` accepts additional absolute paths to scan (e.g. per-camera
    recording_dir, global prefs recording_dir, user-chosen folder).
    Only files whose path contains the camera_id or camera_name are included,
    preventing cross-camera mixing when scanning shared directories.
    """
    from core.paths import get_recordings_dir

    base = get_recordings_dir()

    safe_name = re.sub(r'[<>:"/\\|?*\x00-\x1f]', '_', camera_name or "").rstrip('. ')
    if safe_name.startswith('.'):
        safe_name = '_' + safe_name[1:]

    # Camera-scoped directories (guaranteed to belong to this camera)
    candidates: List[Path] = [base / camera_id]
    if safe_name:
        candidates.append(base / safe_name)

    # Motion watch: scan all zone subdirectories (shared across cameras).
    # Files will be filtered by camera_id/name below.
    try:
        from core.paths import get_motion_watch_dir
        mw = get_motion_watch_dir()
        if mw.exists():
            for sub in mw.iterdir():
                if sub.is_dir():
                    candidates.append(sub)
    except Exception:
        pass

    # Additional directories: only scan camera-specific subdirectories
    # to avoid pulling in other cameras' recordings from a shared root.
    for raw in (extra_dirs or []):
        if not raw:
            continue
        p = Path(raw).expanduser().resolve()
        if p.is_dir():
            if safe_name:
                candidates.append(p / safe_name)
            candidates.append(p / camera_id)
            # If the extra dir itself IS a camera-scoped folder (its name
            # matches camera_id or safe_name), scan it directly.
            if p.name == camera_id or (safe_name and p.name == safe_name):
                candidates.append(p)

    mp4s: List[Path] = []
    seen: set = set()
    for d in candidates:
        if not d.exists():
            continue
        for f in d.rglob("*.mp4"):
            real = str(f.resolve())
            if real not in seen:
                seen.add(real)
                mp4s.append(f)

    # Filter: only keep files that belong to this camera.
    # Camera-scoped dirs (base/id, base/name) pass because camera_id or
    # safe_name is a path component.  Motion-watch clips pass because
    # the filename embeds camera_id.  Files from unrelated cameras in
    # shared directories are excluded.
    mp4s = [f for f in mp4s if _file_belongs_to_camera(f, camera_id, safe_name)]

    segments: List[Segment] = []
    for f in sorted(mp4s):
        start = _parse_segment_ts(f)
        if start is None:
            try:
                start = f.stat().st_mtime
            except Exception:
                continue

        dur = _fast_duration(f, start_ts=start)
        if dur is None or dur <= 0:
            continue

        segments.append(Segment(
            path=str(f),
            start_ts=start,
            end_ts=start + dur,
            duration_s=dur,
            fps=25.0,  # actual FPS resolved when playing the segment
        ))

    segments.sort(key=lambda s: s.start_ts)

    if len(segments) > 1:
        merged: List[Segment] = [segments[0]]
        for seg in segments[1:]:
            prev = merged[-1]
            if seg.start_ts <= prev.end_ts and seg.path == prev.path:
                continue
            merged.append(seg)
        segments = merged

    return segments


# ── Thumbnail cache ──────────────────────────────────────────────────

class _ThumbnailCache:
    """LRU cache of decoded thumbnails keyed by (segment_path, offset_s)."""

    def __init__(self, max_size: int = 12):
        self._cache: OrderedDict[Tuple[str, int], QPixmap] = OrderedDict()
        self._max = max_size
        self._lock = threading.Lock()

    def get(self, key: Tuple[str, int]) -> Optional[QPixmap]:
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
                return self._cache[key]
        return None

    def put(self, key: Tuple[str, int], pm: QPixmap) -> None:
        with self._lock:
            self._cache[key] = pm
            if len(self._cache) > self._max:
                self._cache.popitem(last=False)


# ── Audio pipe ───────────────────────────────────────────────────────

_AUDIO_RATE = 48000
_AUDIO_CH = 2
_AUDIO_SAMPLE_FMT = "s16le"
# Bytes per second of raw PCM: 48000 * 2ch * 2 bytes = 192000
_AUDIO_BPS = _AUDIO_RATE * _AUDIO_CH * 2


class _AudioPipe:
    """Runs ffmpeg to extract audio from an MP4 as raw PCM, streaming to a pipe."""

    def __init__(self):
        self._proc: Optional[subprocess.Popen] = None

    def start(self, path: str, offset_s: float = 0, speed: float = 1.0) -> None:
        self.stop()
        atempo = max(0.5, min(2.0, speed))
        cmd = [
            "ffmpeg", "-hide_banner", "-loglevel", "error",
            "-ss", f"{offset_s:.3f}",
            "-i", path,
            "-vn",
            "-af", f"atempo={atempo:.2f}",
            "-f", _AUDIO_SAMPLE_FMT,
            "-acodec", f"pcm_{_AUDIO_SAMPLE_FMT}",
            "-ar", str(_AUDIO_RATE),
            "-ac", str(_AUDIO_CH),
            "pipe:1",
        ]
        try:
            self._proc = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                bufsize=_AUDIO_BPS,
            )
        except FileNotFoundError:
            self._proc = None

    def read(self, num_bytes: int) -> bytes:
        if self._proc is None or self._proc.stdout is None:
            return b""
        if self._proc.poll() is not None:
            return b""
        try:
            r, _, _ = select.select([self._proc.stdout], [], [], 0.005)
            if not r:
                return b""
            return self._proc.stdout.read(num_bytes) or b""
        except Exception:
            return b""

    def stop(self) -> None:
        if self._proc is not None:
            try:
                self._proc.kill()
                self._proc.wait(timeout=1)
            except Exception:
                pass
            self._proc = None

    @property
    def active(self) -> bool:
        return self._proc is not None and self._proc.poll() is None


# ── Playback engine ──────────────────────────────────────────────────

class _PlaybackEngine:
    """Reads frames from MP4 segments on a background thread."""

    frame_ready = None  # set by overlay to a callable(np.ndarray)
    audio_ready = None  # set by overlay to a callable(bytes)

    def __init__(self):
        self._segments: List[Segment] = []
        self._seg_idx: int = 0
        self._cap: Optional[cv2.VideoCapture] = None
        self._playing = False
        self._speed: float = 1.0
        self._position_ts: float = 0
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        self._use_mediamtx = False
        self._camera_id = ""
        self._audio_pipe = _AudioPipe()
        self._audio_enabled = True

    @property
    def position(self) -> float:
        return self._position_ts

    @property
    def playing(self) -> bool:
        return self._playing

    @property
    def speed(self) -> float:
        return self._speed

    @speed.setter
    def speed(self, v: float):
        self._speed = max(0.25, v)

    @property
    def total_duration(self) -> float:
        if not self._segments:
            return 0
        return self._segments[-1].end_ts - self._segments[0].start_ts

    @property
    def time_start(self) -> float:
        return self._segments[0].start_ts if self._segments else 0

    @property
    def time_end(self) -> float:
        if not self._segments:
            return 0
        seg_end = self._segments[-1].end_ts
        now = time.time()
        # Extend timeline to "now" when the latest segment is recent,
        # so dragging all the way right reaches live video.
        if now - seg_end < 3600:
            return now
        return seg_end

    @property
    def is_at_live_edge(self) -> bool:
        """True when playback position is within 2s of real-time (i.e. 'live')."""
        return abs(time.time() - self._position_ts) < 2.0

    def load(self, segments: List[Segment], camera_id: str = "", use_mediamtx: bool = False):
        self.stop()
        self._segments = segments
        self._camera_id = camera_id
        self._use_mediamtx = use_mediamtx
        if segments:
            self._position_ts = segments[0].start_ts
            self._seg_idx = 0

    def play(self):
        if not self._segments:
            return
        self._playing = True
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def pause(self):
        self._playing = False
        self._stop.set()
        self._audio_pipe.stop()

    def stop(self):
        self.pause()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2)
        self._thread = None
        with self._lock:
            if self._cap:
                try:
                    self._cap.release()
                except Exception:
                    pass
                self._cap = None
        self._audio_pipe.stop()

    def seek(self, ts: float):
        was_playing = self._playing
        self.pause()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=1)

        ts = max(self.time_start, min(self.time_end, ts))

        seg, offset = self._find_segment(ts)
        if seg:
            offset = max(0, min(offset, seg.duration_s - 0.1))
            # Set position to where video actually exists (not the raw
            # click time, which may be in a gap between segments).
            self._position_ts = seg.start_ts + offset
            with self._lock:
                if self._cap:
                    try:
                        self._cap.release()
                    except Exception:
                        pass
                cap = cv2.VideoCapture(seg.path)
                self._cap = cap
                if cap.isOpened():
                    cap.set(cv2.CAP_PROP_POS_MSEC, offset * 1000)
                    ok, frame = cap.read()
                    if ok and self.frame_ready:
                        self.frame_ready(frame)
        else:
            self._position_ts = ts

        if was_playing:
            self.play()

    def step_frame(self, forward: bool = True):
        """Advance or rewind by exactly one frame (while paused)."""
        if self._playing:
            return
        seg, offset = self._find_segment(self._position_ts)
        if not seg:
            return
        with self._lock:
            if self._cap is None or not self._cap.isOpened():
                self._cap = cv2.VideoCapture(seg.path)
                if self._cap.isOpened():
                    self._cap.set(cv2.CAP_PROP_POS_MSEC, offset * 1000)

            if self._cap and self._cap.isOpened():
                if not forward:
                    pos = self._cap.get(cv2.CAP_PROP_POS_MSEC)
                    new_pos = max(0, pos - 2000 / max(1, seg.fps))
                    self._cap.set(cv2.CAP_PROP_POS_MSEC, new_pos)

                ok, frame = self._cap.read()
                if ok:
                    pos_ms = self._cap.get(cv2.CAP_PROP_POS_MSEC)
                    self._position_ts = seg.start_ts + pos_ms / 1000
                    if self.frame_ready:
                        self.frame_ready(frame)

    def _find_segment(self, ts: float) -> Tuple[Optional[Segment], float]:
        # Exact match
        for i, seg in enumerate(self._segments):
            if seg.start_ts <= ts <= seg.end_ts + 0.5:
                self._seg_idx = i
                return seg, ts - seg.start_ts

        if not self._segments:
            return None, 0

        # Before all segments
        if ts < self._segments[0].start_ts:
            self._seg_idx = 0
            return self._segments[0], 0

        # After all segments -- clamp to end of last segment.
        # Use a generous tolerance: the last segment may still be
        # actively recording, so its file is growing beyond end_ts.
        if ts > self._segments[-1].end_ts:
            self._seg_idx = len(self._segments) - 1
            last = self._segments[-1]
            return last, max(0, last.duration_s - 0.5)

        # In a gap between segments -- snap to the nearest segment boundary
        best_i = 0
        best_dist = abs(ts - self._segments[0].start_ts)
        for i, seg in enumerate(self._segments):
            d_start = abs(ts - seg.start_ts)
            d_end = abs(ts - seg.end_ts)
            d = min(d_start, d_end)
            if d < best_dist:
                best_dist = d
                best_i = i
        self._seg_idx = best_i
        seg = self._segments[best_i]
        if ts < seg.start_ts:
            return seg, 0
        return seg, min(ts - seg.start_ts, seg.duration_s)

    def _run(self):
        """Main playback loop running on a background thread."""
        try:
            if self._use_mediamtx:
                self._run_mediamtx()
                return
            self._run_direct()
        except Exception:
            pass
        finally:
            self._playing = False

    def _start_audio_for_segment(self, seg: Segment, offset: float) -> None:
        if self._audio_enabled and self._speed <= 2.0:
            self._audio_pipe.start(seg.path, offset, self._speed)

    def _run_direct(self):
        seg, offset = self._find_segment(self._position_ts)
        if not seg:
            return

        offset = max(0, min(offset, seg.duration_s - 0.1))

        with self._lock:
            cap = self._cap
            if cap is None or not cap.isOpened():
                cap = cv2.VideoCapture(seg.path)
                self._cap = cap
                if not cap.isOpened():
                    return
                cap.set(cv2.CAP_PROP_POS_MSEC, offset * 1000)

        fps = cap.get(cv2.CAP_PROP_FPS) or seg.fps or 25

        if self._speed <= 2.0:
            self._start_audio_for_segment(seg, offset)

        audio_bytes_per_frame = int(_AUDIO_BPS / max(1, fps))

        _TARGET_DISPLAY_FPS = 25.0

        # Wall-clock anchor for drift-free pacing across synced cameras.
        # Maps a real-time moment to a playback-timeline moment so we can
        # compute exactly how long to sleep between frames.
        _wall_anchor = time.monotonic()
        _ts_anchor = self._position_ts

        while not self._stop.is_set():
            speed = self._speed
            display_interval = 1.0 / _TARGET_DISPLAY_FPS

            # ── Timelapse seek mode (speed >= 64x) ──
            if speed >= 64:
                jump_s = speed * display_interval
                target_ts = self._position_ts + jump_s
                new_seg, new_offset = self._find_segment(target_ts)
                if not new_seg:
                    self._position_ts = time.time()
                    break
                if new_seg is not seg or cap is None or not cap.isOpened():
                    if cap is not None:
                        cap.release()
                    cap = cv2.VideoCapture(new_seg.path)
                    with self._lock:
                        self._cap = cap
                    if not cap.isOpened():
                        break
                    seg = new_seg
                    fps = cap.get(cv2.CAP_PROP_FPS) or seg.fps or 25
                new_offset = max(0, min(new_offset, seg.duration_s - 0.1))
                cap.set(cv2.CAP_PROP_POS_MSEC, new_offset * 1000)
                ok, frame = cap.read()
                if ok:
                    self._position_ts = seg.start_ts + new_offset
                    if self.frame_ready:
                        self.frame_ready(frame)
                else:
                    self._position_ts = target_ts
                _wall_anchor = time.monotonic()
                _ts_anchor = self._position_ts
                time.sleep(display_interval)
                continue

            # ── Frame-skip mode (2x < speed < 64x) ──
            if speed > 2.0:
                skip_n = max(1, int(speed / 2)) - 1
            else:
                skip_n = 0

            for _ in range(skip_n):
                if self._stop.is_set():
                    break
                if not cap.grab():
                    break

            ok, frame = cap.read()
            if not ok:
                self._audio_pipe.stop()
                next_idx = self._seg_idx + 1
                if next_idx >= len(self._segments):
                    self._position_ts = time.time()
                    break
                next_seg = self._segments[next_idx]
                gap = next_seg.start_ts - seg.end_ts
                if speed < 4.0 and gap > 30:
                    self._position_ts = seg.end_ts
                    break
                self._seg_idx = next_idx
                seg = next_seg
                cap.release()
                cap = cv2.VideoCapture(seg.path)
                with self._lock:
                    self._cap = cap
                if not cap.isOpened():
                    break
                fps = cap.get(cv2.CAP_PROP_FPS) or seg.fps or 25
                audio_bytes_per_frame = int(_AUDIO_BPS / max(1, fps))
                if speed <= 2.0:
                    self._start_audio_for_segment(seg, 0)
                continue

            pos_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
            self._position_ts = seg.start_ts + pos_ms / 1000

            if self.frame_ready:
                self.frame_ready(frame)

            if speed <= 2.0 and self._audio_enabled and self._audio_pipe.active and self.audio_ready:
                pcm = self._audio_pipe.read(audio_bytes_per_frame)
                if pcm:
                    self.audio_ready(pcm)

            # Wall-clock-anchored pacing: compute how far ahead we are
            # relative to where we should be, and sleep only the
            # remaining time.  This eliminates drift between synced
            # cameras caused by variable decode times.
            wall_elapsed = time.monotonic() - _wall_anchor
            target_elapsed = (self._position_ts - _ts_anchor) / max(0.25, speed)
            sleep_needed = target_elapsed - wall_elapsed
            if sleep_needed > 0:
                time.sleep(min(sleep_needed, 0.1))
            elif sleep_needed < -0.5:
                # We've fallen behind by more than 500ms (slow decode);
                # re-anchor to prevent a permanent sprint to catch up.
                _wall_anchor = time.monotonic()
                _ts_anchor = self._position_ts

        self._audio_pipe.stop()
        with self._lock:
            if self._cap:
                try:
                    self._cap.release()
                except Exception:
                    pass
                self._cap = None

    def _run_mediamtx(self):
        dt = datetime.fromtimestamp(self._position_ts, tz=timezone.utc)
        url = (
            f"http://localhost:{_PLAYBACK_PORT}/get"
            f"?path={self._camera_id}"
            f"&start={dt.isoformat()}"
            f"&duration=3600"
        )
        cap = cv2.VideoCapture(url)
        with self._lock:
            if self._cap:
                try:
                    self._cap.release()
                except Exception:
                    pass
            self._cap = cap

        if not cap.isOpened():
            return

        fps = cap.get(cv2.CAP_PROP_FPS) or 25
        while not self._stop.is_set() and cap.isOpened():
            interval = 1.0 / max(1, fps) / max(0.25, self._speed)
            ok, frame = cap.read()
            if not ok:
                break
            self._position_ts += interval * self._speed
            if self.frame_ready:
                self.frame_ready(frame)
            time.sleep(interval)

        with self._lock:
            if self._cap:
                try:
                    self._cap.release()
                except Exception:
                    pass
                self._cap = None


# ── Helper ───────────────────────────────────────────────────────────


def _format_speed(speed: float) -> str:
    """Human-readable speed label.  At extreme multipliers, show the
    real-world time covered per second of playback (e.g. '1 min/s')."""
    if speed <= 0:
        return "0x"
    if speed < 1:
        return f"{speed:.2g}x"
    if speed <= 999:
        return f"{speed:g}x"
    secs_per_sec = speed
    if secs_per_sec < 3600:
        mins = secs_per_sec / 60
        return f"{mins:.0f}m/s"
    if secs_per_sec < 86400:
        hrs = secs_per_sec / 3600
        return f"{hrs:.1g}h/s"
    days = secs_per_sec / 86400
    return f"{days:.1g}d/s"


def _strftime(fmt: str, dt: datetime) -> str:
    """Cross-platform strftime that handles ``%-I``, ``%-d``, etc.

    The ``%-`` prefix (no zero-padding) works on Linux/macOS but crashes on
    Windows.  We normalise to standard ``%I``/``%d`` and strip leading zeros
    manually so the result is identical on every platform.
    """
    import sys
    if sys.platform == "win32":
        fmt = fmt.replace("%-", "%#")
    try:
        return dt.strftime(fmt)
    except ValueError:
        # Ultimate fallback: strip the modifier entirely and lstrip zeros.
        clean = fmt.replace("%-", "%").replace("%#", "%")
        return dt.strftime(clean)


def _format_time(seconds: float) -> str:
    s = max(0, int(seconds))
    h, s = divmod(s, 3600)
    m, s = divmod(s, 60)
    if h:
        return f"{h}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"


# ── Overlay widget ───────────────────────────────────────────────────

class PlaybackOverlayWidget(QWidget):
    """Full video-player overlay with progress bar, controls, and thumbnail preview."""

    seek_requested = Signal(float)
    speed_changed = Signal(float)
    play_toggled = Signal(bool)   # True = playing, False = paused
    go_live_requested = Signal()
    step_requested = Signal(bool)  # True = forward
    playback_closed = Signal()
    segments_loaded = Signal()
    _thumb_ready = Signal(object, object)  # (cache_key, QImage) -- thread-safe
    _audio_chunk = Signal(bytes)  # PCM from engine thread -> GUI thread

    def __init__(self, camera_id: str, parent: QWidget | None = None):
        super().__init__(parent)
        self.camera_id = camera_id
        self.camera_name = ""
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setAttribute(Qt.WidgetAttribute.WA_OpaquePaintEvent, False)
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

        self._events: List[Dict[str, Any]] = []
        self._dragging = False
        self._hovered = False
        self._hover_ts: float = 0
        self._hover_pixmap: Optional[QPixmap] = None
        self._thumb_cache = _ThumbnailCache()
        self._thumb_debounce_timer = QTimer(self)
        self._thumb_debounce_timer.setSingleShot(True)
        self._thumb_debounce_timer.setInterval(100)
        self._thumb_debounce_timer.timeout.connect(self._decode_hover_thumbnail)
        self._thumb_ready.connect(self._on_thumb_decoded)

        self._filter_zones = True
        self._filter_lines = True
        self._filter_tags = True
        self._filter_detections = True
        self._filter_shapes = False  # shapes hidden by default during playback
        self._filter_panel_visible = False

        self._engine = _PlaybackEngine()
        self._engine.frame_ready = self._on_engine_frame
        self._engine.audio_ready = self._on_engine_audio

        # Audio playback (reuse AudioPlayback from audio_eq module)
        self._audio_playback = None  # lazy init
        self._volume: float = 0.8
        self._muted: bool = False
        self._audio_chunk.connect(self._push_audio_pcm)

        self._tick_timer = QTimer(self)
        self._tick_timer.setInterval(40)
        self._tick_timer.timeout.connect(self._on_tick)

        # Event-replay: periodically fetch historical detections near playback position
        self._replay_timer = QTimer(self)
        self._replay_timer.setInterval(500)
        self._replay_timer.timeout.connect(self._fetch_playback_detections)
        self._last_replay_fetch_ts: float = 0

        # Periodic segment refresh so new recordings appear on the timeline
        self._refresh_timer = QTimer(self)
        self._refresh_timer.setInterval(15000)  # every 15 seconds
        self._refresh_timer.timeout.connect(self._refresh_segments)
        self._refresh_extra_dirs: list = []

        # Arrow-key seek acceleration state
        self._seek_last_key_ts: float = 0
        self._seek_repeat_count: int = 0
        self._seek_direction: int = 0  # -1 = left, +1 = right

        # Zoomable view window (timestamps defining the visible portion of the timeline)
        self._view_start: float = 0
        self._view_end: float = 0
        self._zoom_min_span: float = 30.0  # minimum visible window = 30s

        # Sync indicator
        self._synced: bool = False
        self._no_video_at_time: bool = False

        # Auto-hide dock: fade out after inactivity so the video is unobstructed.
        self._dock_opacity: float = 1.0
        self._dock_hide_timer = QTimer(self)
        self._dock_hide_timer.setSingleShot(True)
        self._dock_hide_timer.setInterval(2500)
        self._dock_hide_timer.timeout.connect(self._fade_dock_out)
        self._dock_fade_timer = QTimer(self)
        self._dock_fade_timer.setInterval(30)
        self._dock_fade_timer.timeout.connect(self._dock_fade_step)
        self._dock_target_opacity: float = 1.0
        self._dock_hide_timer.start()

        # Calendar popup state
        self._cal_visible: bool = False
        self._cal_year: int = 0
        self._cal_month: int = 0
        self._cal_hour: int = 12
        self._cal_minute: int = 0
        self._cal_rec_days: set = set()   # days in current month with recordings
        self._cal_evt_days: Dict[int, List[str]] = {}  # day -> list of event types

    # ── Dock auto-hide ────────────────────────────────────────────────

    def _show_dock(self):
        """Reveal the dock immediately and restart the hide timer."""
        self._dock_target_opacity = 1.0
        if self._dock_opacity < 1.0:
            self._dock_fade_timer.start()
        self._dock_hide_timer.start()

    def _fade_dock_out(self):
        """Begin fading the dock to near-invisible."""
        if self._dragging or self._cal_visible or self._filter_panel_visible:
            self._dock_hide_timer.start()
            return
        self._dock_target_opacity = 0.12
        self._dock_fade_timer.start()

    def _dock_fade_step(self):
        """Animate one step toward target opacity."""
        diff = self._dock_target_opacity - self._dock_opacity
        if abs(diff) < 0.02:
            self._dock_opacity = self._dock_target_opacity
            self._dock_fade_timer.stop()
        else:
            self._dock_opacity += diff * 0.25
        self.update()

    # ── Geometry ─────────────────────────────────────────────────────

    def _dock_rect(self) -> QRectF:
        w = self.width()
        h = self.height()
        return QRectF(_DOCK_MARGIN, h - _DOCK_H - _DOCK_MARGIN, w - 2 * _DOCK_MARGIN, _DOCK_H)

    def _rail_rect(self) -> QRectF:
        d = self._dock_rect()
        pad_l = 8
        pad_r = 8
        return QRectF(d.x() + pad_l, d.y() + _RAIL_Y_OFFSET, d.width() - pad_l - pad_r, _RAIL_H)

    # ── Public API ───────────────────────────────────────────────────

    def load_segments(self, segments: List[Segment], use_mediamtx: bool = False,
                      extra_dirs: Optional[List[str]] = None):
        self._engine.load(segments, camera_id=self.camera_id, use_mediamtx=use_mediamtx)
        if extra_dirs is not None:
            self._refresh_extra_dirs = list(extra_dirs)
        eng = self._engine
        self._view_end = eng.time_end
        self._view_start = max(eng.time_start, self._view_end - 300)
        self.segments_loaded.emit()
        self._filter_shapes = False
        self._sync_shapes_visibility()
        self._refresh_timer.start()
        if segments:
            start_at = max(segments[-1].start_ts, segments[-1].end_ts - 30)
            self.start_playback(start_ts=start_at)
        else:
            self.update()

    def load_segments_no_autoplay(self, segments: List[Segment],
                                  use_mediamtx: bool = False,
                                  extra_dirs: Optional[List[str]] = None):
        """Load segments and set up the timeline but do NOT auto-start playback.

        Used by sync-playback so the caller can seek to a specific timestamp
        before starting, avoiding a visible jump to the wrong position.
        """
        self._engine.load(segments, camera_id=self.camera_id, use_mediamtx=use_mediamtx)
        if extra_dirs is not None:
            self._refresh_extra_dirs = list(extra_dirs)
        eng = self._engine
        self._view_end = eng.time_end
        self._view_start = max(eng.time_start, self._view_end - 300)
        self.segments_loaded.emit()
        self._filter_shapes = False
        self._sync_shapes_visibility()
        self._refresh_timer.start()
        self.update()

    def load_events(self, events: List[Dict[str, Any]]) -> None:
        self._events = events or []
        self.update()

    def start_playback(self, start_ts: float | None = None) -> None:
        if start_ts is not None:
            self._engine.seek(start_ts)
        self._ensure_audio_playback()
        self._engine.play()
        self._tick_timer.start()
        self._replay_timer.start()
        self.update()

    def stop_playback(self) -> None:
        self._engine.stop()
        self._tick_timer.stop()
        self._replay_timer.stop()
        self._refresh_timer.stop()
        self._clear_playback_detections()
        self._stop_audio()
        self._filter_shapes = True
        self._sync_shapes_visibility()
        self.update()

    def toggle_play_pause(self):
        if self._engine.playing:
            self._engine.pause()
        else:
            self._engine.play()
            self._tick_timer.start()
            self._replay_timer.start()
        self.play_toggled.emit(self._engine.playing)
        self.update()

    def seek(self, ts: float) -> None:
        self._engine.seek(ts)
        self.seek_requested.emit(ts)
        self.update()

    # ── Engine callbacks ─────────────────────────────────────────────

    def _on_engine_frame(self, frame: np.ndarray):
        parent = self.parent()
        if parent and hasattr(parent, "update_frame"):
            parent.update_frame(frame)

    def _on_tick(self):
        eng = self._engine
        if not eng.playing:
            if eng.is_at_live_edge:
                # Track real-time so live frames keep flowing via receive_frame
                eng._position_ts = time.time()
                # Keep the view window's right edge at "now"
                now = time.time()
                span = self._view_span()
                self._view_end = now
                self._view_start = now - span
            else:
                self._tick_timer.stop()
        else:
            self._ensure_position_visible()
        self.update()

    # ── Segment refresh ───────────────────────────────────────────────

    def _refresh_segments(self):
        """Rescan recording directories for new/extended segments."""
        def _worker():
            try:
                from desktop.widgets.playback_overlay import build_segment_timeline
                segments = build_segment_timeline(
                    self.camera_id,
                    self.camera_name or "",
                    extra_dirs=self._refresh_extra_dirs or None,
                )
                if segments:
                    eng = self._engine
                    old_pos = eng._position_ts
                    was_playing = eng.playing
                    # Update segment list without disrupting current playback
                    eng._segments = segments
                    # If the last segment grew or a new one appeared, the
                    # timeline now covers more recent time.
                    self.update()
            except Exception:
                pass
        threading.Thread(target=_worker, daemon=True).start()

    # ── Audio playback ───────────────────────────────────────────────

    def _ensure_audio_playback(self):
        if self._audio_playback is not None:
            return
        try:
            from desktop.widgets.audio_eq import AudioPlayback
            self._audio_playback = AudioPlayback(parent=self)
            self._audio_playback.set_volume(self._volume)
            self._audio_playback.set_muted(self._muted)
            self._audio_playback.start()
        except Exception:
            self._audio_playback = None

    def _on_engine_audio(self, pcm: bytes):
        """Called from the engine background thread -- marshal to GUI thread."""
        if pcm and not self._muted:
            self._audio_chunk.emit(pcm)

    def _push_audio_pcm(self, pcm: bytes):
        """Runs on the GUI thread -- safe to push to QAudioSink."""
        self._ensure_audio_playback()
        if self._audio_playback:
            self._audio_playback.push_pcm(pcm)

    def _stop_audio(self):
        if self._audio_playback:
            try:
                self._audio_playback.stop()
            except Exception:
                pass

    def _sync_audio_to_speed(self):
        """Restart (or stop) the audio pipe to match the current playback speed."""
        eng = self._engine
        if eng.speed > 2.0:
            eng._audio_pipe.stop()
            return
        if eng.playing and eng._audio_enabled:
            seg, offset = eng._find_segment(eng.position)
            if seg:
                eng._audio_pipe.start(seg.path, offset, eng.speed)

    # ── Event-replay detections ──────────────────────────────────────

    def _clear_playback_detections(self):
        """Clear historical detections from the GL widget."""
        parent = self.parent()
        if parent and hasattr(parent, 'playback_detections'):
            parent.playback_detections = []

    def _fetch_playback_detections(self):
        """Query event index for detections near the current playback position."""
        eng = self._engine
        if eng.is_at_live_edge:
            self._clear_playback_detections()
            return

        pos = eng.position
        # Avoid redundant fetches for the same time window
        if abs(pos - self._last_replay_fetch_ts) < 2.0:
            return
        self._last_replay_fetch_ts = pos

        # Extract detections from already-loaded events within +-5s window
        dets: List[Dict[str, Any]] = []
        for ev in self._events:
            ev_ts = ev.get("captured_ts", 0)
            if abs(ev_ts - pos) <= 5:
                ev_dets = ev.get("detections") or []
                for d in ev_dets:
                    if isinstance(d, dict) and d.get("bbox"):
                        dets.append(d)

        parent = self.parent()
        if parent and hasattr(parent, 'playback_detections'):
            parent.playback_detections = dets

    # ── Thumbnail preview ────────────────────────────────────────────

    def _decode_hover_thumbnail(self):
        if not self._hovered or self._hover_ts <= 0:
            return
        ts = self._hover_ts
        seg, offset = self._engine._find_segment(ts)
        if not seg:
            return
        cache_key = (seg.path, int(offset))
        cached = self._thumb_cache.get(cache_key)
        if cached:
            self._hover_pixmap = cached
            self.update()
            return

        def _decode():
            try:
                cap = cv2.VideoCapture(seg.path)
                if not cap.isOpened():
                    return
                cap.set(cv2.CAP_PROP_POS_MSEC, offset * 1000)
                ok, frame = cap.read()
                cap.release()
                if not ok or frame is None:
                    return
                thumb = cv2.resize(frame, (_THUMB_W, _THUMB_H), interpolation=cv2.INTER_AREA)
                h, w, ch = thumb.shape
                qimg = QImage(thumb.data.tobytes(), w, h, w * ch, QImage.Format.Format_BGR888)
                self._thumb_ready.emit(cache_key, qimg)
            except Exception:
                pass
        threading.Thread(target=_decode, daemon=True).start()

    def _on_thumb_decoded(self, cache_key, qimg):
        """Runs on GUI thread -- safe to create QPixmap here."""
        try:
            pm = QPixmap.fromImage(qimg)
            self._thumb_cache.put(cache_key, pm)
            self._hover_pixmap = pm
            self.update()
        except Exception:
            pass

    # ── Filter helpers ───────────────────────────────────────────────

    def _sync_shapes_visibility(self):
        """Push the shapes toggle to the GL widget so paintEvent can skip shapes."""
        parent = self.parent()
        if parent and hasattr(parent, 'playback_hide_shapes'):
            parent.playback_hide_shapes = not self._filter_shapes

    def _visible_events(self) -> List[Dict[str, Any]]:
        out = []
        for e in self._events:
            st = (e.get("shape_type") or e.get("trigger_type") or "").lower()
            if st == "zone" and not self._filter_zones:
                continue
            if st == "line" and not self._filter_lines:
                continue
            if st == "tag" and not self._filter_tags:
                continue
            if st == "detection" and not self._filter_detections:
                continue
            out.append(e)
        return out

    # ── Paint ────────────────────────────────────────────────────────

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        try:
            self._do_paint(painter)
        except Exception:
            pass
        finally:
            try:
                painter.end()
            except Exception:
                pass

    def _do_paint(self, p: QPainter) -> None:
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        dock = self._dock_rect()
        rail = self._rail_rect()
        eng = self._engine

        # Apply dock auto-hide opacity
        p.setOpacity(self._dock_opacity)

        # Dock background
        path = QPainterPath()
        path.addRoundedRect(dock, _DOCK_RADIUS, _DOCK_RADIUS)
        p.setPen(QPen(QColor(51, 65, 85, 120), 1))
        p.setBrush(QColor(15, 23, 42, 210))
        p.drawPath(path)

        font = QFont("Segoe UI", 10)
        font.setBold(True)
        p.setFont(font)

        ty = dock.y() + 6
        btn_h = 26
        lx = dock.x() + 10

        # ── Transport row ────────────────────────────────────────────

        # Play / Pause
        p.setPen(QColor(56, 189, 248))
        p.setFont(QFont("Segoe UI", 12, QFont.Weight.Bold))
        play_label = "⏸" if eng.playing else "▶"
        p.drawText(QRectF(lx, ty, 28, btn_h), Qt.AlignmentFlag.AlignCenter, play_label)

        # Rewind / Fast-forward (30s jumps)
        p.setPen(QColor(148, 163, 184))
        p.setFont(QFont("Segoe UI", 11))
        p.drawText(QRectF(lx + 30, ty, 28, btn_h), Qt.AlignmentFlag.AlignCenter, "⏪")
        p.drawText(QRectF(lx + 60, ty, 28, btn_h), Qt.AlignmentFlag.AlignCenter, "⏩")

        # Frame step
        p.setFont(QFont("Segoe UI", 9))
        p.drawText(QRectF(lx + 92, ty, 20, btn_h), Qt.AlignmentFlag.AlignCenter, ",")
        p.drawText(QRectF(lx + 112, ty, 20, btn_h), Qt.AlignmentFlag.AlignCenter, ".")

        # Zoom out / in buttons
        p.setPen(QColor(148, 163, 184))
        p.setFont(QFont("Segoe UI", 11))
        p.drawText(QRectF(lx + 136, ty, 20, btn_h), Qt.AlignmentFlag.AlignCenter, "−")
        p.drawText(QRectF(lx + 158, ty, 20, btn_h), Qt.AlignmentFlag.AlignCenter, "+")

        # Calendar button
        cal_x = lx + 182
        p.setPen(QColor(56, 189, 248) if self._cal_visible else QColor(148, 163, 184))
        p.setFont(QFont("Segoe UI", 10))
        p.drawText(QRectF(cal_x, ty, 22, btn_h), Qt.AlignmentFlag.AlignCenter, "📅")

        # ── Context label: friendly date/time, camera name, shapes ───
        if eng.position > 0:
            pos_dt = datetime.fromtimestamp(eng.position)
            now_dt = datetime.now()
            time_part = _strftime("%-I:%M %p", pos_dt).lstrip("0")
            if pos_dt.date() == now_dt.date():
                date_part = "Today"
            elif pos_dt.date() == (now_dt - timedelta(days=1)).date():
                date_part = "Yesterday"
            else:
                date_part = _strftime("%a %b %-d", pos_dt)
            ctx = f"{date_part} {time_part}"
            if self.camera_name:
                ctx += f"  ·  {self.camera_name}"
            # Summarize active shapes from the GL widget
            parent = self.parent()
            shapes = getattr(parent, 'shapes', None) or []
            if shapes:
                counts = {}
                for s in shapes:
                    k = s.get('kind', '')
                    if k and s.get('enabled', True) and not s.get('hidden'):
                        counts[k] = counts.get(k, 0) + 1
                parts = []
                for k in ('zone', 'line', 'tag'):
                    n = counts.get(k, 0)
                    if n:
                        parts.append(f"{n} {k}{'s' if n > 1 else ''}")
                if parts:
                    ctx += f"  ·  {', '.join(parts)}"
        else:
            ctx = self.camera_name or "Playback"
        p.setFont(QFont("Segoe UI", 9))
        p.setPen(QColor(148, 163, 184))
        ctx_w = min(dock.width() - 420, max(200, len(ctx) * 7))
        ctx_x = dock.x() + dock.width() / 2 - ctx_w / 2
        p.drawText(QRectF(ctx_x, ty, ctx_w, btn_h), Qt.AlignmentFlag.AlignCenter, ctx)

        # ── Right-side controls (right to left: Close, LIVE, Filter, Speed, Volume)
        close_x = dock.right() - 30
        live_x = close_x - 50
        filter_x = live_x - 28
        spd_x = filter_x - 42
        vol_bar_right = spd_x - 8
        vol_bar_w = 50
        vol_bar_x = vol_bar_right - vol_bar_w
        vol_icon_x = vol_bar_x - 22

        # Volume icon (speaker / muted)
        p.setFont(QFont("Segoe UI", 10))
        if self._muted:
            p.setPen(QColor(248, 113, 113))
            p.drawText(QRectF(vol_icon_x, ty, 22, btn_h), Qt.AlignmentFlag.AlignCenter, "🔇")
        else:
            p.setPen(QColor(148, 163, 184))
            p.drawText(QRectF(vol_icon_x, ty, 22, btn_h), Qt.AlignmentFlag.AlignCenter, "🔊")

        # Volume bar (thin track with filled portion)
        vol_cy = ty + btn_h / 2
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QColor(51, 65, 85))
        p.drawRoundedRect(QRectF(vol_bar_x, vol_cy - 2, vol_bar_w, 4), 2, 2)
        if not self._muted:
            fill_w = vol_bar_w * self._volume
            p.setBrush(QColor(56, 189, 248))
            p.drawRoundedRect(QRectF(vol_bar_x, vol_cy - 2, fill_w, 4), 2, 2)
            # Knob
            p.setBrush(QColor(226, 232, 240))
            p.drawEllipse(QPointF(vol_bar_x + fill_w, vol_cy), 4, 4)

        # Speed
        p.setPen(QColor(148, 163, 184))
        p.setFont(QFont("Segoe UI", 9))
        spd_text = _format_speed(eng.speed)
        spd_w = max(40, len(spd_text) * 7 + 8)
        p.drawText(QRectF(spd_x - (spd_w - 40), ty, spd_w, btn_h), Qt.AlignmentFlag.AlignCenter, spd_text)

        # Filter
        p.setFont(QFont("Segoe UI", 10))
        p.drawText(QRectF(filter_x, ty, 26, btn_h), Qt.AlignmentFlag.AlignCenter, "⚙")

        # LIVE badge / Go Live
        if eng.is_at_live_edge:
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(QColor(239, 68, 68, 200))
            p.drawRoundedRect(QRectF(live_x, ty + 4, 38, btn_h - 8), 4, 4)
            p.setPen(QColor(255, 255, 255))
            p.setFont(QFont("Segoe UI", 8, QFont.Weight.Bold))
            p.drawText(QRectF(live_x, ty + 4, 38, btn_h - 8), Qt.AlignmentFlag.AlignCenter, "LIVE")
        else:
            p.setPen(QColor(100, 116, 139))
            p.setFont(QFont("Segoe UI", 8))
            p.drawText(QRectF(live_x, ty, 48, btn_h), Qt.AlignmentFlag.AlignCenter, "Go Live")

        # Close
        p.setPen(QColor(248, 113, 113))
        p.setFont(QFont("Segoe UI", 11))
        p.drawText(QRectF(close_x, ty, 24, btn_h), Qt.AlignmentFlag.AlignCenter, "✕")

        # SYNC badge (compact, next to LIVE badge)
        if self._synced:
            sync_x = live_x - 38
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(QColor(74, 222, 128, 180))
            p.drawRoundedRect(QRectF(sync_x, ty + 6, 32, btn_h - 12), 3, 3)
            p.setPen(QColor(255, 255, 255))
            p.setFont(QFont("Segoe UI", 6, QFont.Weight.Bold))
            p.drawText(QRectF(sync_x, ty + 6, 32, btn_h - 12), Qt.AlignmentFlag.AlignCenter, "SYNC")

        # No-video-at-time indicator
        if self._no_video_at_time:
            nv_w = 180
            nv_x = dock.x() + dock.width() / 2 - nv_w / 2
            nv_y = dock.y() - 28
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(QColor(245, 158, 11, 210))
            p.drawRoundedRect(QRectF(nv_x, nv_y, nv_w, 20), 4, 4)
            p.setPen(QColor(255, 255, 255))
            p.setFont(QFont("Segoe UI", 8, QFont.Weight.Bold))
            p.drawText(QRectF(nv_x, nv_y, nv_w, 20), Qt.AlignmentFlag.AlignCenter,
                       "No video at this time")

        # ── Zoomable progress bar ─────────────────────────────────────

        # Track background
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QColor(30, 41, 59, 220))
        p.drawRoundedRect(rail, _RAIL_H / 2, _RAIL_H / 2)

        v_start = self._view_start
        v_end = self._view_end
        v_span = v_end - v_start if v_end > v_start else 1

        def _ts_to_x(ts: float) -> float:
            return rail.x() + (ts - v_start) / v_span * rail.width()

        # ── Time-of-day tick marks (clean 12-hour format) ─────────────

        # Pick a tick interval that yields ~4-10 labels across the visible span
        _tick_steps = [30, 60, 300, 600, 900, 1800, 3600, 10800, 21600]
        tick_interval = _tick_steps[-1]
        for iv in _tick_steps:
            if 3 <= v_span / iv <= 12:
                tick_interval = iv
                break

        first_tick = math.ceil(v_start / tick_interval) * tick_interval
        p.setFont(QFont("Segoe UI", 6))
        tick_ts = first_tick
        while tick_ts <= v_end:
            tx = _ts_to_x(tick_ts)
            if rail.x() + 10 <= tx <= rail.right() - 10:
                p.setPen(QPen(QColor(100, 116, 139, 80), 1))
                p.drawLine(QPointF(tx, rail.y() + 1), QPointF(tx, rail.y() + rail.height() - 1))
                t_dt = datetime.fromtimestamp(tick_ts)
                if tick_interval >= 3600:
                    lbl = _strftime("%-I %p", t_dt).lstrip("0")
                else:
                    lbl = _strftime("%-I:%M", t_dt).lstrip("0")
                # Show day prefix when the view spans multiple days
                if v_span > 43200:
                    lbl = t_dt.strftime("%a ") + lbl
                lbl_w = max(34, len(lbl) * 6)
                lbl_rect = QRectF(tx - lbl_w / 2, rail.y() - 11, lbl_w, 10)
                p.setPen(QColor(100, 116, 139))
                p.drawText(lbl_rect, Qt.AlignmentFlag.AlignCenter, lbl)
            tick_ts += tick_interval

        # Segment coverage bars (clearly show where recordings exist)
        for seg in (eng._segments or []):
            seg_x0 = max(rail.x(), _ts_to_x(seg.start_ts))
            seg_x1 = min(rail.right(), _ts_to_x(seg.end_ts))
            if seg_x1 > seg_x0:
                p.setPen(Qt.PenStyle.NoPen)
                p.setBrush(QColor(56, 189, 248, 35))
                p.drawRoundedRect(QRectF(seg_x0, rail.y(), seg_x1 - seg_x0, rail.height()), 2, 2)
                # Segment boundary tick
                if seg_x0 > rail.x() + 2:
                    p.setPen(QPen(QColor(56, 189, 248, 80), 1))
                    p.drawLine(QPointF(seg_x0, rail.y()), QPointF(seg_x0, rail.y() + rail.height()))

        # Filled portion (played progress)
        if eng.position > v_start and v_span > 0:
            frac = max(0, min(1, (eng.position - v_start) / v_span))
            filled = QRectF(rail.x(), rail.y(), rail.width() * frac, rail.height())
            p.setBrush(QColor(56, 189, 248, 100))
            p.setPen(Qt.PenStyle.NoPen)
            p.drawRoundedRect(filled, _RAIL_H / 2, _RAIL_H / 2)

        # Event markers (visible within the view window, grouped when dense)
        visible = self._visible_events()
        _MARKER_W = 5
        _GROUP_PX = 8
        marker_slots: List[Tuple[float, List[Dict[str, Any]]]] = []
        for e in visible:
            ts = int(e.get("captured_ts", 0))
            if ts < v_start or ts > v_end:
                continue
            mx = _ts_to_x(ts)
            if marker_slots and abs(mx - marker_slots[-1][0]) < _GROUP_PX:
                marker_slots[-1][1].append(e)
            else:
                marker_slots.append((mx, [e]))

        for mx, group in marker_slots:
            st = (group[0].get("shape_type") or group[0].get("trigger_type") or "zone").lower()
            color = _EVENT_COLORS.get(st, _EVENT_COLORS["zone"])
            glow = QColor(color)
            glow.setAlpha(60)
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(glow)
            p.drawRoundedRect(QRectF(mx - _MARKER_W / 2 - 1, rail.y(), _MARKER_W + 2, rail.height()), 2, 2)
            p.setBrush(color)
            p.drawRoundedRect(QRectF(mx - _MARKER_W / 2, rail.y() + 1, _MARKER_W, rail.height() - 2), 2, 2)
            if len(group) > 1:
                badge_r = QRectF(mx - 6, rail.y() - 12, 12, 12)
                p.setBrush(QColor(15, 23, 42, 220))
                p.setPen(QPen(color, 1))
                p.drawRoundedRect(badge_r, 3, 3)
                p.setPen(QColor(226, 232, 240))
                p.setFont(QFont("Segoe UI", 6))
                p.drawText(badge_r, Qt.AlignmentFlag.AlignCenter, str(len(group)))

        # Scrubber handle (positioned within the view window)
        if v_span > 0 and v_start <= eng.position <= v_end:
            sx = _ts_to_x(eng.position)
            cy = rail.y() + rail.height() / 2
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(QColor(0, 0, 0, 80))
            p.drawEllipse(QPointF(sx + 1, cy + 1), _HANDLE_R, _HANDLE_R)
            p.setBrush(QColor(56, 189, 248))
            p.drawEllipse(QPointF(sx, cy), _HANDLE_R, _HANDLE_R)
            p.setBrush(QColor(255, 255, 255))
            p.drawEllipse(QPointF(sx, cy), 3, 3)

        # ── Hover tooltip with thumbnail ─────────────────────────────

        if self._hovered and self._hover_ts > 0 and v_span > 0:
            hx = _ts_to_x(self._hover_ts)
            hx = max(rail.x(), min(rail.right(), hx))

            if self._hover_pixmap and not self._hover_pixmap.isNull():
                tw, th = _THUMB_W, _THUMB_H
                tx = max(dock.x(), min(dock.right() - tw, hx - tw / 2))
                ty_thumb = rail.y() - th - 28
                p.setPen(QPen(QColor(51, 65, 85, 180), 1))
                p.setBrush(QColor(15, 23, 42, 240))
                p.drawRoundedRect(QRectF(tx - 2, ty_thumb - 2, tw + 4, th + 4), 4, 4)
                p.drawPixmap(int(tx), int(ty_thumb), self._hover_pixmap)

            # Time label below thumbnail (friendly format)
            h_dt = datetime.fromtimestamp(self._hover_ts)
            if h_dt.date() == datetime.now().date():
                tip = _strftime("%-I:%M:%S %p", h_dt).lstrip("0")
            else:
                tip = _strftime("%a %-I:%M %p", h_dt).lstrip("0")
            nearest_ev = None
            nearest_dist = float('inf')
            for e in visible:
                ev_ts = int(e.get("captured_ts", 0))
                if ev_ts < v_start or ev_ts > v_end:
                    continue
                d = abs(_ts_to_x(ev_ts) - hx)
                if d < 12 and d < nearest_dist:
                    nearest_dist = d
                    nearest_ev = e
            if nearest_ev:
                ev_type = (nearest_ev.get("shape_type") or nearest_ev.get("trigger_type") or "event").capitalize()
                ev_name = nearest_ev.get("shape_name") or nearest_ev.get("trigger_name") or ""
                tip = f"{tip}  {ev_type}: {ev_name}" if ev_name else f"{tip}  {ev_type}"
            tip_w = max(56, len(tip) * 6 + 12)
            tip_rect = QRectF(hx - tip_w / 2, rail.y() - 18, tip_w, 14)
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(QColor(15, 23, 42, 230))
            p.drawRoundedRect(tip_rect, 3, 3)
            p.setPen(QColor(226, 232, 240))
            p.setFont(QFont("Segoe UI", 7))
            p.drawText(tip_rect, Qt.AlignmentFlag.AlignCenter, tip)

        # Filter panel
        if self._filter_panel_visible:
            self._paint_filter_panel(p, dock)

        # Calendar popup
        if self._cal_visible:
            self._paint_calendar(p)

    def _paint_filter_panel(self, p: QPainter, dock: QRectF) -> None:
        pw, ph = 150, 136
        panel = QRectF(dock.right() - pw - 8, dock.y() - ph - 6, pw, ph)
        p.setPen(QPen(QColor(51, 65, 85, 150), 1))
        p.setBrush(QColor(15, 23, 42, 230))
        p.drawRoundedRect(panel, 6, 6)
        font = QFont("Segoe UI", 9)
        p.setFont(font)
        items = [
            ("Shapes", self._filter_shapes, QColor(168, 162, 255)),
            ("Zones", self._filter_zones, _EVENT_COLORS["zone"]),
            ("Lines", self._filter_lines, _EVENT_COLORS["line"]),
            ("Tags", self._filter_tags, _EVENT_COLORS["tag"]),
            ("Detections", self._filter_detections, _EVENT_COLORS["detection"]),
        ]
        y = panel.y() + 10
        for label, enabled, color in items:
            check_color = color if enabled else QColor(100, 116, 139, 80)
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(check_color)
            p.drawRoundedRect(QRectF(panel.x() + 10, y, 10, 10), 2, 2)
            p.setPen(QColor(226, 232, 240) if enabled else QColor(100, 116, 139))
            p.drawText(QRectF(panel.x() + 26, y - 2, 110, 16), Qt.AlignmentFlag.AlignVCenter, label)
            y += 24

    # ── Calendar popup ─────────────────────────────────────────────

    _CAL_W = 224
    _CAL_H = 240
    _CAL_CELL = 30
    _CAL_ROW_H = 24

    def _toggle_calendar(self):
        self._cal_visible = not self._cal_visible
        if self._cal_visible:
            now = datetime.fromtimestamp(self._engine.position) if self._engine.position > 0 else datetime.now()
            self._cal_year = now.year
            self._cal_month = now.month
            self._cal_hour = now.hour
            self._cal_minute = now.minute
            self._rebuild_cal_data()
        self.update()

    def _rebuild_cal_data(self):
        """Precompute which days in _cal_month have recordings or events."""
        y, m = self._cal_year, self._cal_month
        month_start = datetime(y, m, 1).timestamp()
        _, days_in_month = _cal_mod.monthrange(y, m)
        month_end = datetime(y, m, days_in_month, 23, 59, 59).timestamp()

        rec_days: set = set()
        for seg in (self._engine._segments or []):
            if seg.end_ts < month_start or seg.start_ts > month_end:
                continue
            d0 = max(1, datetime.fromtimestamp(max(seg.start_ts, month_start)).day)
            d1 = min(days_in_month, datetime.fromtimestamp(min(seg.end_ts, month_end)).day)
            for d in range(d0, d1 + 1):
                rec_days.add(d)
        self._cal_rec_days = rec_days

        evt_days: Dict[int, List[str]] = {}
        for e in self._events:
            ts = e.get("captured_ts", 0)
            if ts < month_start or ts > month_end:
                continue
            day = datetime.fromtimestamp(ts).day
            st = (e.get("shape_type") or e.get("trigger_type") or "zone").lower()
            evt_days.setdefault(day, []).append(st)
        self._cal_evt_days = evt_days

    def _cal_rect(self) -> QRectF:
        dock = self._dock_rect()
        cx = dock.x() + dock.width() / 2
        return QRectF(cx - self._CAL_W / 2, dock.y() - self._CAL_H - 8,
                      self._CAL_W, self._CAL_H)

    def _paint_calendar(self, p: QPainter) -> None:
        cr = self._cal_rect()
        p.setPen(QPen(QColor(51, 65, 85, 180), 1))
        p.setBrush(QColor(15, 23, 42, 240))
        p.drawRoundedRect(cr, 8, 8)

        x0, y0 = cr.x(), cr.y()
        cw = self._CAL_CELL
        rh = self._CAL_ROW_H

        # ── Header: < Month Year >
        header_y = y0 + 6
        p.setFont(QFont("Segoe UI", 9, QFont.Weight.Bold))
        p.setPen(QColor(226, 232, 240))
        month_name = datetime(self._cal_year, self._cal_month, 1).strftime("%B %Y")
        p.drawText(QRectF(x0, header_y, self._CAL_W, 18), Qt.AlignmentFlag.AlignCenter, month_name)

        # Nav arrows
        p.setFont(QFont("Segoe UI", 11))
        p.setPen(QColor(148, 163, 184))
        p.drawText(QRectF(x0 + 6, header_y, 20, 18), Qt.AlignmentFlag.AlignCenter, "◀")
        p.drawText(QRectF(x0 + self._CAL_W - 26, header_y, 20, 18), Qt.AlignmentFlag.AlignCenter, "▶")

        # ── Weekday headers
        wk_y = header_y + 22
        p.setFont(QFont("Segoe UI", 7))
        p.setPen(QColor(100, 116, 139))
        for i, d in enumerate("SMTWTFS"):
            p.drawText(QRectF(x0 + 4 + i * cw, wk_y, cw, 14), Qt.AlignmentFlag.AlignCenter, d)

        # ── Day grid
        first_dow, days_in_month = _cal_mod.monthrange(self._cal_year, self._cal_month)
        first_dow = (first_dow + 1) % 7  # calendar module: Mon=0, we want Sun=0
        grid_y = wk_y + 16
        today = datetime.now()
        today_day = today.day if today.year == self._cal_year and today.month == self._cal_month else -1

        p.setFont(QFont("Segoe UI", 8))
        for day in range(1, days_in_month + 1):
            cell_idx = first_dow + day - 1
            col = cell_idx % 7
            row = cell_idx // 7
            cx = x0 + 4 + col * cw
            cy = grid_y + row * rh
            cell_r = QRectF(cx, cy, cw, rh)

            # Recording highlight
            has_rec = day in self._cal_rec_days
            if has_rec:
                p.setPen(Qt.PenStyle.NoPen)
                p.setBrush(QColor(56, 189, 248, 40))
                p.drawRoundedRect(cell_r.adjusted(2, 1, -2, -1), 3, 3)

            # Today ring
            if day == today_day:
                p.setPen(QPen(QColor(56, 189, 248), 1.5))
                p.setBrush(Qt.BrushStyle.NoBrush)
                p.drawRoundedRect(cell_r.adjusted(3, 2, -3, -2), 3, 3)

            # Day number
            p.setPen(QColor(226, 232, 240) if has_rec else QColor(100, 116, 139))
            p.drawText(cell_r.adjusted(0, 0, 0, -4), Qt.AlignmentFlag.AlignCenter, str(day))

            # Event dots (up to 3 tiny colored dots below the number)
            evt_types = self._cal_evt_days.get(day)
            if evt_types:
                unique = list(dict.fromkeys(evt_types))[:3]
                dot_y = cy + rh - 6
                dot_start_x = cx + cw / 2 - len(unique) * 3
                for j, st in enumerate(unique):
                    color = _EVENT_COLORS.get(st, _EVENT_COLORS["zone"])
                    p.setPen(Qt.PenStyle.NoPen)
                    p.setBrush(color)
                    p.drawEllipse(QPointF(dot_start_x + j * 6 + 2, dot_y), 2, 2)

        # ── Time picker row
        time_y = grid_y + 6 * rh + 4
        p.setPen(QPen(QColor(51, 65, 85, 120), 1))
        p.drawLine(QPointF(x0 + 10, time_y), QPointF(x0 + self._CAL_W - 10, time_y))

        time_row_y = time_y + 4
        p.setFont(QFont("Segoe UI", 8))
        p.setPen(QColor(148, 163, 184))
        p.drawText(QRectF(x0 + 8, time_row_y, 34, 18), Qt.AlignmentFlag.AlignVCenter, "Time:")

        # Hour -/+
        p.setPen(QColor(100, 116, 139))
        p.drawText(QRectF(x0 + 44, time_row_y, 16, 18), Qt.AlignmentFlag.AlignCenter, "−")
        p.setPen(QColor(226, 232, 240))
        p.setFont(QFont("Segoe UI", 9, QFont.Weight.Bold))
        hr_str = f"{self._cal_hour:02d}"
        p.drawText(QRectF(x0 + 60, time_row_y, 24, 18), Qt.AlignmentFlag.AlignCenter, hr_str)
        p.setPen(QColor(100, 116, 139))
        p.setFont(QFont("Segoe UI", 8))
        p.drawText(QRectF(x0 + 84, time_row_y, 16, 18), Qt.AlignmentFlag.AlignCenter, "+")

        p.setPen(QColor(226, 232, 240))
        p.drawText(QRectF(x0 + 100, time_row_y, 10, 18), Qt.AlignmentFlag.AlignCenter, ":")

        # Minute -/+
        p.setPen(QColor(100, 116, 139))
        p.setFont(QFont("Segoe UI", 8))
        p.drawText(QRectF(x0 + 110, time_row_y, 16, 18), Qt.AlignmentFlag.AlignCenter, "−")
        p.setPen(QColor(226, 232, 240))
        p.setFont(QFont("Segoe UI", 9, QFont.Weight.Bold))
        mn_str = f"{self._cal_minute:02d}"
        p.drawText(QRectF(x0 + 126, time_row_y, 24, 18), Qt.AlignmentFlag.AlignCenter, mn_str)
        p.setPen(QColor(100, 116, 139))
        p.setFont(QFont("Segoe UI", 8))
        p.drawText(QRectF(x0 + 150, time_row_y, 16, 18), Qt.AlignmentFlag.AlignCenter, "+")

        # Go button
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QColor(56, 189, 248, 180))
        go_r = QRectF(x0 + self._CAL_W - 42, time_row_y + 1, 32, 16)
        p.drawRoundedRect(go_r, 3, 3)
        p.setPen(QColor(255, 255, 255))
        p.setFont(QFont("Segoe UI", 7, QFont.Weight.Bold))
        p.drawText(go_r, Qt.AlignmentFlag.AlignCenter, "GO")

    def _handle_cal_click(self, x: float, y: float) -> bool:
        """Handle a click inside the calendar popup. Returns True if consumed."""
        cr = self._cal_rect()
        if not cr.contains(QPointF(x, y)):
            return False

        x0, y0 = cr.x(), cr.y()
        cw = self._CAL_CELL
        rh = self._CAL_ROW_H
        header_y = y0 + 6

        # Month nav arrows
        if header_y <= y <= header_y + 18:
            if x < x0 + 30:
                self._cal_month -= 1
                if self._cal_month < 1:
                    self._cal_month = 12
                    self._cal_year -= 1
                self._rebuild_cal_data()
                self.update()
                return True
            if x > x0 + self._CAL_W - 30:
                self._cal_month += 1
                if self._cal_month > 12:
                    self._cal_month = 1
                    self._cal_year += 1
                self._rebuild_cal_data()
                self.update()
                return True

        # Day grid click
        wk_y = header_y + 22
        grid_y = wk_y + 16
        first_dow, days_in_month = _cal_mod.monthrange(self._cal_year, self._cal_month)
        first_dow = (first_dow + 1) % 7

        if grid_y <= y <= grid_y + 6 * rh:
            col = int((x - x0 - 4) / cw)
            row = int((y - grid_y) / rh)
            if 0 <= col <= 6:
                day = row * 7 + col - first_dow + 1
                if 1 <= day <= days_in_month:
                    target = datetime(self._cal_year, self._cal_month, day,
                                      self._cal_hour, self._cal_minute)
                    self._seek_and_zoom_to(target.timestamp())
                    return True

        # Time picker row
        time_y = grid_y + 6 * rh + 4
        time_row_y = time_y + 4
        if time_row_y <= y <= time_row_y + 18:
            # Hour minus
            if x0 + 44 <= x <= x0 + 60:
                self._cal_hour = (self._cal_hour - 1) % 24
                self.update()
                return True
            # Hour plus
            if x0 + 84 <= x <= x0 + 100:
                self._cal_hour = (self._cal_hour + 1) % 24
                self.update()
                return True
            # Minute minus
            if x0 + 110 <= x <= x0 + 126:
                self._cal_minute = (self._cal_minute - 5) % 60
                self.update()
                return True
            # Minute plus
            if x0 + 150 <= x <= x0 + 166:
                self._cal_minute = (self._cal_minute + 5) % 60
                self.update()
                return True
            # GO button
            if x >= x0 + self._CAL_W - 42:
                target = datetime(self._cal_year, self._cal_month, 1,
                                  self._cal_hour, self._cal_minute)
                # Use the first day that has recordings, or day 1
                for d in range(1, 32):
                    if d in self._cal_rec_days:
                        try:
                            target = datetime(self._cal_year, self._cal_month, d,
                                              self._cal_hour, self._cal_minute)
                        except ValueError:
                            pass
                        break
                self._seek_and_zoom_to(target.timestamp())
                return True

        return True  # consumed (inside calendar rect)

    def _go_live(self):
        """Jump to the live edge and let live camera frames show."""
        eng = self._engine
        eng.stop()
        eng._position_ts = time.time()
        self._clear_playback_detections()
        self._refresh_segments()
        self._view_end = eng.time_end
        self._view_start = max(eng.time_start, self._view_end - 300)
        self._tick_timer.start()
        self.go_live_requested.emit()
        self.update()

    def _seek_and_zoom_to(self, ts: float):
        """Seek to a timestamp and center the view window around it."""
        eng = self._engine
        self._engine.seek(ts)
        self.seek_requested.emit(ts)
        half = min(900, self._view_span() / 2)
        self._view_start = ts - half
        self._view_end = ts + half
        if self._view_end - self._view_start < self._zoom_min_span:
            self._view_end = self._view_start + self._zoom_min_span
        self._cal_visible = False
        self.update()

    # ── Mouse interaction ────────────────────────────────────────────

    def _in_dock(self, y: float) -> bool:
        return y >= self._dock_rect().y() - 20

    def mousePressEvent(self, event) -> None:
        pos = event.position() if hasattr(event, "position") else event.pos()
        x, y = pos.x(), pos.y()
        dock = self._dock_rect()
        rail = self._rail_rect()

        if self._in_dock(y):
            self._show_dock()

        # Calendar popup clicks (highest priority when visible, above dock)
        if self._cal_visible:
            if self._handle_cal_click(x, y):
                return
            # Click outside calendar and outside dock dismisses it
            if not self._in_dock(y):
                self._cal_visible = False
                self.update()
                return

        # Click on video area (above dock) -- pass through to parent
        # so the widget can be dragged.  Double-click toggles pause.
        if not self._in_dock(y):
            event.ignore()
            return

        ty = dock.y() + 6
        btn_h = 26
        lx = dock.x() + 10

        # ── Right-side button hit tests (same layout as paint) ─────────
        close_x = dock.right() - 30
        live_x = close_x - 50
        filter_x = live_x - 28
        spd_x = filter_x - 42
        vol_bar_right = spd_x - 8
        vol_bar_w = 50
        vol_bar_x = vol_bar_right - vol_bar_w
        vol_icon_x = vol_bar_x - 22

        # Close
        if ty <= y <= ty + btn_h and close_x <= x <= dock.right():
            self.stop_playback()
            self.hide()
            self.playback_closed.emit()
            return

        # Calendar button
        cal_x = lx + 182
        if ty <= y <= ty + btn_h and cal_x <= x <= cal_x + 22:
            self._toggle_calendar()
            return

        # Context label click (opens calendar)
        ctx_w = max(200, dock.width() - 420)
        ctx_x = dock.x() + dock.width() / 2 - ctx_w / 2
        if ty <= y <= ty + btn_h and ctx_x <= x <= ctx_x + ctx_w:
            self._toggle_calendar()
            return

        # LIVE / Go Live
        if ty <= y <= ty + btn_h and live_x <= x <= live_x + 48:
            self._go_live()
            return

        # Filter
        if ty <= y <= ty + btn_h and filter_x <= x <= filter_x + 26:
            self._filter_panel_visible = not self._filter_panel_visible
            self.update()
            return

        # Filter panel clicks
        if self._filter_panel_visible:
            pw, ph = 150, 136
            panel = QRectF(dock.right() - pw - 8, dock.y() - ph - 6, pw, ph)
            if panel.contains(QPointF(x, y)):
                idx = int((y - panel.y() - 10) / 24)
                if 0 <= idx <= 4:
                    if idx == 0:
                        self._filter_shapes = not self._filter_shapes
                        self._sync_shapes_visibility()
                    elif idx == 1: self._filter_zones = not self._filter_zones
                    elif idx == 2: self._filter_lines = not self._filter_lines
                    elif idx == 3: self._filter_tags = not self._filter_tags
                    elif idx == 4: self._filter_detections = not self._filter_detections
                    self.update()
                return

        # Volume icon click (toggle mute)
        if ty <= y <= ty + btn_h and vol_icon_x <= x <= vol_icon_x + 22:
            self._muted = not self._muted
            if self._audio_playback:
                self._audio_playback.set_muted(self._muted)
            self.update()
            return

        # Volume bar click (set volume)
        if ty <= y <= ty + btn_h and vol_bar_x <= x <= vol_bar_x + vol_bar_w:
            self._volume = max(0.0, min(1.0, (x - vol_bar_x) / vol_bar_w))
            self._muted = False
            if self._audio_playback:
                self._audio_playback.set_volume(self._volume)
                self._audio_playback.set_muted(False)
            self.update()
            return

        # Play/Pause
        if ty <= y <= ty + btn_h and lx <= x <= lx + 28:
            self.toggle_play_pause()
            return

        # Rewind (-30s)
        if ty <= y <= ty + btn_h and lx + 30 <= x <= lx + 58:
            self.seek(self._engine.position - 30)
            return

        # Fast-forward (+30s)
        if ty <= y <= ty + btn_h and lx + 60 <= x <= lx + 88:
            self.seek(self._engine.position + 30)
            return

        # Frame step back (,)
        if ty <= y <= ty + btn_h and lx + 92 <= x <= lx + 112:
            self._engine.step_frame(forward=False)
            self.step_requested.emit(False)
            self.update()
            return

        # Frame step forward (.)
        if ty <= y <= ty + btn_h and lx + 112 <= x <= lx + 132:
            self._engine.step_frame(forward=True)
            self.step_requested.emit(True)
            self.update()
            return

        # Zoom out (−)
        if ty <= y <= ty + btn_h and lx + 136 <= x <= lx + 156:
            self._zoom(1.0 / 0.7, 0.5)
            return

        # Zoom in (+)
        if ty <= y <= ty + btn_h and lx + 158 <= x <= lx + 178:
            self._zoom(0.7, 0.5)
            return

        # Speed cycle (spd_x already computed above)
        if ty <= y <= ty + btn_h and spd_x <= x <= spd_x + 40:
            speeds = [0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0,
                      128.0, 256.0, 512.0, 1024.0, 3600.0, 86400.0]
            cur = self._engine.speed
            nxt = speeds[0]
            for i, s in enumerate(speeds):
                if cur < s - 0.01:
                    nxt = s
                    break
                if i == len(speeds) - 1:
                    nxt = speeds[0]
            self._engine.speed = nxt
            self._sync_audio_to_speed()
            self.speed_changed.emit(self._engine.speed)
            self.update()
            return

        # Rail click / drag (generous hit area)
        if rail.y() - 10 <= y <= rail.y() + rail.height() + 10:
            self._drag_seek_rail(x, rail)
            self._dragging = True
            return

    def mouseMoveEvent(self, event) -> None:
        pos = event.position() if hasattr(event, "position") else event.pos()
        x, y = pos.x(), pos.y()
        rail = self._rail_rect()

        if self._dragging:
            self._show_dock()
            self._drag_seek_rail(x, rail)
            if self._view_end > self._view_start:
                frac = max(0.0, min(1.0, (x - rail.x()) / max(1, rail.width())))
                self._hover_ts = self._view_start + frac * self._view_span()
                self._hovered = True
                self._thumb_debounce_timer.start()
            return

        # Reveal dock when the mouse is near the bottom of the widget
        if self._in_dock(y):
            self._show_dock()
        else:
            event.ignore()
            return

        if rail.y() - 12 <= y <= rail.y() + rail.height() + 12 and self._view_end > self._view_start:
            frac = max(0.0, min(1.0, (x - rail.x()) / max(1, rail.width())))
            self._hover_ts = self._view_start + frac * self._view_span()
            self._hovered = True
            self._thumb_debounce_timer.start()
            self.update()
        elif self._hovered:
            self._hovered = False
            self._hover_pixmap = None
            self.update()

    def mouseDoubleClickEvent(self, event) -> None:
        pos = event.position() if hasattr(event, "position") else event.pos()
        if not self._in_dock(pos.y()):
            self.toggle_play_pause()
            return
        super().mouseDoubleClickEvent(event)

    def mouseReleaseEvent(self, event) -> None:
        was_dragging = self._dragging
        self._dragging = False
        if was_dragging:
            self._hovered = False
            self._hover_pixmap = None
            # Do the real seek now that the user has released
            ts = self._engine._position_ts
            self._engine.seek(ts)
            self.seek_requested.emit(ts)
            # Auto-play from wherever the user released the slider
            if not self._engine.playing:
                self._engine.play()
                self._tick_timer.start()
                self._replay_timer.start()
            self.update()

    def _drag_seek_rail(self, x: float, rail: QRectF) -> None:
        eng = self._engine
        if eng.time_end <= eng.time_start:
            return
        # Pause on first drag event only
        if eng.playing and not self._dragging:
            eng.pause()
            if eng._thread and eng._thread.is_alive():
                eng._thread.join(timeout=1)

        frac = max(0.0, min(1.0, (x - rail.x()) / max(1, rail.width())))
        ts = self._view_start + frac * self._view_span()

        # Auto-expand the view window when dragging near the edges so
        # the user can seamlessly scroll into older or newer recordings.
        edge_zone = 0.05
        span = self._view_span()
        if frac < edge_zone:
            shift = span * 0.15
            self._view_start -= shift
            self._view_end -= shift
        elif frac > 1.0 - edge_zone:
            shift = span * 0.15
            self._view_start += shift
            self._view_end += shift

        seg, offset = eng._find_segment(ts)
        if seg:
            eng._position_ts = seg.start_ts + max(0, min(offset, seg.duration_s - 0.1))
        else:
            eng._position_ts = ts
        self.update()

    # ── Zoom / pan ─────────────────────────────────────────────────

    def _view_span(self) -> float:
        return max(1.0, self._view_end - self._view_start)

    def _zoom(self, factor: float, anchor_frac: float = 0.5):
        """Zoom in (factor < 1) or out (factor > 1) around *anchor_frac* (0..1 across the view)."""
        eng = self._engine
        full_end = eng.time_end
        if full_end <= eng.time_start:
            return
        old_span = self._view_span()
        new_span = max(self._zoom_min_span, old_span * factor)
        anchor_ts = self._view_start + anchor_frac * old_span
        self._view_start = anchor_ts - anchor_frac * new_span
        self._view_end = self._view_start + new_span
        # Only clamp the right edge to "now"; the left edge is unbounded
        # so the user can scroll back through all historical recordings.
        if self._view_end > full_end:
            self._view_end = full_end
            self._view_start = full_end - new_span
        self.update()

    def _pan(self, delta_s: float):
        """Shift the view window by *delta_s* seconds.

        The right edge is clamped to 'now' but the left edge is
        unbounded so the user can pan back through all history.
        """
        eng = self._engine
        span = self._view_span()
        new_start = self._view_start + delta_s
        # Don't let the right edge go past the live edge
        if new_start + span > eng.time_end:
            new_start = eng.time_end - span
        self._view_start = new_start
        self._view_end = new_start + span
        self.update()

    def _ensure_position_visible(self):
        """Auto-scroll the view window so the playback position stays visible."""
        pos = self._engine.position
        margin = self._view_span() * 0.05
        if pos < self._view_start:
            self._pan(pos - self._view_start - margin)
        elif pos > self._view_end:
            self._pan(pos - self._view_end + margin)

    def wheelEvent(self, event) -> None:
        """Scroll wheel: zoom in/out. Shift+wheel: pan."""
        rail = self._rail_rect()
        pos = event.position() if hasattr(event, "position") else event.pos()
        delta = event.angleDelta().y()
        if delta == 0:
            return

        mods = event.modifiers()
        shift = bool(mods & Qt.KeyboardModifier.ShiftModifier)

        if shift:
            # Pan: shift+scroll moves the view window
            pan_s = self._view_span() * 0.1 * (-1 if delta > 0 else 1)
            self._pan(pan_s)
        else:
            # Zoom centered on cursor position along the rail
            anchor = max(0.0, min(1.0, (pos.x() - rail.x()) / max(1, rail.width())))
            factor = 0.7 if delta > 0 else 1.0 / 0.7
            self._zoom(factor, anchor)
        event.accept()

    # ── Keyboard ─────────────────────────────────────────────────────

    def _accelerated_seek_delta(self, direction: int) -> float:
        """Return a seek delta in seconds that ramps up with rapid/held presses.

        First press:   5 s
        Rapid taps:    10 → 30 → 60 → 120 → 300 s  (ramps over ~1s of tapping)
        Held key:      auto-repeat accelerates the same way
        Direction change resets the ramp.
        """
        now = time.time()
        gap = now - self._seek_last_key_ts

        if direction != self._seek_direction or gap > 0.6:
            self._seek_repeat_count = 0
            self._seek_direction = direction
        else:
            self._seek_repeat_count += 1

        self._seek_last_key_ts = now

        _RAMP = [5, 10, 30, 60, 120, 300, 600]
        idx = min(self._seek_repeat_count, len(_RAMP) - 1)
        return _RAMP[idx] * direction

    def keyPressEvent(self, event) -> None:
        key = event.key()
        mods = event.modifiers()
        shift = bool(mods & Qt.KeyboardModifier.ShiftModifier)

        self._show_dock()

        if key == Qt.Key.Key_Space:
            self.toggle_play_pause()
        elif key == Qt.Key.Key_Left:
            if shift:
                self.seek(self._engine.position - 1)
            else:
                delta = self._accelerated_seek_delta(-1)
                self.seek(self._engine.position + delta)
        elif key == Qt.Key.Key_Right:
            if shift:
                self.seek(self._engine.position + 1)
            else:
                delta = self._accelerated_seek_delta(+1)
                self.seek(self._engine.position + delta)
        elif key == Qt.Key.Key_Period:
            self._engine.step_frame(forward=True)
            self.step_requested.emit(True)
            self.update()
        elif key == Qt.Key.Key_Comma:
            self._engine.step_frame(forward=False)
            self.step_requested.emit(False)
            self.update()
        elif key == Qt.Key.Key_Plus or key == Qt.Key.Key_Equal:
            self._zoom(0.7, 0.5)
        elif key == Qt.Key.Key_Minus:
            self._zoom(1.0 / 0.7, 0.5)
        elif key == Qt.Key.Key_Escape:
            self.stop_playback()
            self.hide()
            self.playback_closed.emit()
        elif key == Qt.Key.Key_BracketRight:
            self._cycle_speed(+1)
        elif key == Qt.Key.Key_BracketLeft:
            self._cycle_speed(-1)
        else:
            super().keyPressEvent(event)

    def _cycle_speed(self, direction: int):
        """Double (+1) or halve (-1) the playback speed."""
        if direction > 0:
            self._engine.speed = self._engine.speed * 2
        else:
            self._engine.speed = self._engine.speed / 2
        self._sync_audio_to_speed()
        self.speed_changed.emit(self._engine.speed)
        self.update()
