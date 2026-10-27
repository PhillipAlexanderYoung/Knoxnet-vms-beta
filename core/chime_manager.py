"""Chime management: built-in tone generation, file/recorded chimes, and trigger engine."""

from __future__ import annotations

import json
import logging
import struct
import subprocess
import tempfile
import threading
import time
import uuid
import wave
from pathlib import Path
from typing import Dict, List, Optional

try:
    import numpy as np  # type: ignore
except Exception:
    np = None  # type: ignore

try:
    import sounddevice as sd  # type: ignore
    _SD_AVAILABLE = True
except Exception:
    sd = None  # type: ignore
    _SD_AVAILABLE = False

from core.paths import get_data_dir

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Built-in chime definitions
# ---------------------------------------------------------------------------

_SAMPLE_RATE = 44100


def _make_tone(
    freq_hz: float,
    duration_sec: float,
    sample_rate: int = _SAMPLE_RATE,
    amplitude: float = 0.6,
) -> bytes:
    """Generate a single sine-wave tone as 16-bit PCM mono bytes."""
    if np is not None:
        t = np.linspace(0, duration_sec, int(sample_rate * duration_sec), endpoint=False, dtype=np.float32)
        envelope = np.ones_like(t)
        # simple ADSR-like fade-in / fade-out
        fade = int(sample_rate * 0.02)
        envelope[:fade] = np.linspace(0, 1, fade)
        envelope[-fade:] = np.linspace(1, 0, fade)
        samples = (np.sin(2 * np.pi * freq_hz * t) * amplitude * envelope * 32767).astype(np.int16)
        return samples.tobytes()
    else:
        import math
        n = int(sample_rate * duration_sec)
        fade = int(sample_rate * 0.02)
        data = bytearray(n * 2)
        for i in range(n):
            fenv = 1.0
            if i < fade:
                fenv = i / fade
            elif i >= n - fade:
                fenv = (n - i) / fade
            v = math.sin(2 * math.pi * freq_hz * i / sample_rate) * amplitude * fenv
            s = max(-32768, min(32767, int(v * 32767)))
            struct.pack_into("<h", data, i * 2, s)
        return bytes(data)


def _make_sweep(
    freq_start: float,
    freq_end: float,
    duration_sec: float,
    sample_rate: int = _SAMPLE_RATE,
    amplitude: float = 0.6,
) -> bytes:
    """Generate a linear frequency sweep as 16-bit PCM mono bytes."""
    if np is not None:
        t = np.linspace(0, duration_sec, int(sample_rate * duration_sec), endpoint=False, dtype=np.float32)
        freq_t = np.linspace(freq_start, freq_end, len(t), dtype=np.float32)
        phase = np.cumsum(2 * np.pi * freq_t / sample_rate)
        envelope = np.ones_like(t)
        fade = int(sample_rate * 0.02)
        envelope[:fade] = np.linspace(0, 1, fade)
        envelope[-fade:] = np.linspace(1, 0, fade)
        samples = (np.sin(phase) * amplitude * envelope * 32767).astype(np.int16)
        return samples.tobytes()
    else:
        import math
        n = int(sample_rate * duration_sec)
        fade = int(sample_rate * 0.02)
        data = bytearray(n * 2)
        phase = 0.0
        for i in range(n):
            fenv = 1.0
            if i < fade:
                fenv = i / fade
            elif i >= n - fade:
                fenv = (n - i) / fade
            freq = freq_start + (freq_end - freq_start) * i / max(1, n - 1)
            phase += 2 * math.pi * freq / sample_rate
            v = math.sin(phase) * amplitude * fenv
            s = max(-32768, min(32767, int(v * 32767)))
            struct.pack_into("<h", data, i * 2, s)
        return bytes(data)


def _concat_pcm(*parts: bytes) -> bytes:
    """Concatenate PCM byte sequences with a short silence gap between them."""
    silence = bytes(int(_SAMPLE_RATE * 0.08) * 2)
    result = bytearray()
    for i, part in enumerate(parts):
        if i > 0:
            result.extend(silence)
        result.extend(part)
    return bytes(result)


_BUILTIN_SPECS: List[Dict] = [
    {
        "id": "builtin_ding",
        "name": "Ding",
        "type": "builtin",
        "volume": 0.8,
        "fn": lambda: _make_tone(880, 0.5),
    },
    {
        "id": "builtin_double_ding",
        "name": "Double Ding",
        "type": "builtin",
        "volume": 0.8,
        "fn": lambda: _concat_pcm(_make_tone(880, 0.35), _make_tone(1100, 0.35)),
    },
    {
        "id": "builtin_beep",
        "name": "Beep",
        "type": "builtin",
        "volume": 0.8,
        "fn": lambda: _make_tone(1000, 0.25),
    },
    {
        "id": "builtin_alert",
        "name": "Alert",
        "type": "builtin",
        "volume": 0.8,
        "fn": lambda: _make_sweep(440, 880, 0.45),
    },
    {
        "id": "builtin_triple",
        "name": "Triple Beep",
        "type": "builtin",
        "volume": 0.8,
        "fn": lambda: _concat_pcm(
            _make_tone(1000, 0.15),
            _make_tone(1000, 0.15),
            _make_tone(1200, 0.2),
        ),
    },
]


def _write_wav(pcm_mono_s16: bytes, path: Path, sample_rate: int = _SAMPLE_RATE) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm_mono_s16)


# ---------------------------------------------------------------------------
# ChimeStore
# ---------------------------------------------------------------------------

class ChimeStore:
    """Persist chime definitions and drive playback."""

    def __init__(self) -> None:
        self._data_dir = get_data_dir()
        self._chimes_dir = self._data_dir / "chimes"
        self._json_path = self._data_dir / "chimes.json"
        self._builtin_wav_dir = self._chimes_dir / "_builtin"
        self._builtin_wav_dir.mkdir(parents=True, exist_ok=True)
        self._chimes_dir.mkdir(parents=True, exist_ok=True)
        self._user_chimes: List[Dict] = []
        self._load()

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _load(self) -> None:
        try:
            if self._json_path.exists():
                data = json.loads(self._json_path.read_text(encoding="utf-8"))
                self._user_chimes = list(data.get("chimes") or [])
        except Exception:
            self._user_chimes = []

    def _save(self) -> None:
        try:
            self._json_path.parent.mkdir(parents=True, exist_ok=True)
            self._json_path.write_text(
                json.dumps({"chimes": self._user_chimes}, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
        except Exception:
            log.warning("ChimeStore: failed to save chimes.json", exc_info=True)

    # ------------------------------------------------------------------
    # Built-in WAV cache
    # ------------------------------------------------------------------

    def _builtin_wav_path(self, chime_id: str) -> Path:
        return self._builtin_wav_dir / f"{chime_id}.wav"

    def _ensure_builtin_wav(self, spec: Dict) -> Optional[Path]:
        path = self._builtin_wav_path(spec["id"])
        if not path.exists():
            try:
                pcm = spec["fn"]()
                _write_wav(pcm, path)
            except Exception:
                log.warning("ChimeStore: failed to generate built-in WAV %s", spec["id"], exc_info=True)
                return None
        return path

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def list_chimes(self) -> List[Dict]:
        """Return all chimes: built-ins first, then user-defined."""
        builtins = [
            {
                "id": s["id"],
                "name": s["name"],
                "type": "builtin",
                "path": None,
                "volume": s["volume"],
                "created_at": None,
                "readonly": True,
            }
            for s in _BUILTIN_SPECS
        ]
        return builtins + [dict(c) for c in self._user_chimes]

    def get_chime(self, chime_id: str) -> Optional[Dict]:
        for c in self.list_chimes():
            if c["id"] == chime_id:
                return c
        return None

    def add_chime(self, name: str, path: str, chime_type: str = "file") -> Dict:
        chime = {
            "id": str(uuid.uuid4()),
            "name": str(name).strip() or "Untitled",
            "type": chime_type,
            "path": str(path),
            "volume": 0.8,
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "readonly": False,
        }
        self._user_chimes.append(chime)
        self._save()
        return chime

    def delete_chime(self, chime_id: str) -> bool:
        before = len(self._user_chimes)
        self._user_chimes = [c for c in self._user_chimes if c.get("id") != chime_id]
        if len(self._user_chimes) < before:
            self._save()
            return True
        return False

    def update_volume(self, chime_id: str, volume: float) -> bool:
        for c in self._user_chimes:
            if c.get("id") == chime_id:
                c["volume"] = max(0.0, min(1.0, float(volume)))
                self._save()
                return True
        return False

    # ------------------------------------------------------------------
    # Playback
    # ------------------------------------------------------------------

    def play_chime(self, chime_id: str, output_device=None, volume_override: Optional[float] = None) -> None:
        """Fire-and-forget chime playback in a daemon thread."""
        t = threading.Thread(
            target=self._play_chime_sync,
            args=(chime_id, output_device, volume_override),
            daemon=True,
        )
        t.start()

    def _resolve_wav_path(self, chime_id: str) -> Optional[Path]:
        """Return a filesystem path to a WAV file for the given chime id."""
        # Built-in?
        for spec in _BUILTIN_SPECS:
            if spec["id"] == chime_id:
                return self._ensure_builtin_wav(spec)

        # User chime
        chime = next((c for c in self._user_chimes if c.get("id") == chime_id), None)
        if chime is None:
            return None
        path_str = chime.get("path") or ""
        if not path_str:
            return None
        return Path(path_str)

    def _apply_volume(self, pcm_s16: bytes, volume: float) -> bytes:
        """Scale 16-bit PCM by volume factor [0,1]."""
        v = max(0.0, min(1.0, float(volume)))
        if abs(v - 1.0) < 1e-3:
            return pcm_s16
        if np is not None:
            try:
                a = np.frombuffer(pcm_s16, dtype=np.int16).astype(np.float32) / 32768.0
                a = np.clip(a * v, -1.0, 1.0)
                return (a * 32767).astype(np.int16).tobytes()
            except Exception:
                pass
        # Fallback: scale each sample
        import array as _array
        arr = _array.array("h", pcm_s16)
        for i in range(len(arr)):
            arr[i] = max(-32768, min(32767, int(arr[i] * v)))
        return bytes(arr)

    def _play_chime_sync(self, chime_id: str, output_device, volume_override: Optional[float]) -> None:
        """Blocking playback — called in daemon thread."""
        try:
            chime = self.get_chime(chime_id)
            if chime is None:
                log.debug("ChimeStore.play: unknown chime id %s", chime_id)
                return

            volume = float(volume_override if volume_override is not None else chime.get("volume", 0.8))
            wav_path = self._resolve_wav_path(chime_id)

            if wav_path is None or not Path(wav_path).exists():
                log.warning("ChimeStore.play: WAV not found for %s", chime_id)
                return

            wav_path = Path(wav_path)

            # Read WAV
            try:
                with wave.open(str(wav_path), "rb") as wf:
                    sample_rate = wf.getframerate()
                    n_channels = wf.getnchannels()
                    raw = wf.readframes(wf.getnframes())
            except Exception:
                # Non-WAV file: try sounddevice's read
                if _SD_AVAILABLE and sd is not None:
                    try:
                        import soundfile as sf  # type: ignore
                        audio_data, sr = sf.read(str(wav_path), dtype="float32")
                        audio_data = (audio_data * volume).clip(-1.0, 1.0)
                        kwargs: dict = {"samplerate": sr, "blocking": True}
                        if output_device is not None:
                            kwargs["device"] = output_device
                        sd.play(audio_data, **kwargs)
                        return
                    except Exception:
                        pass
                # Give up for non-WAV without soundfile
                log.warning("ChimeStore.play: cannot decode non-WAV %s (no soundfile/sounddevice)", wav_path)
                return

            # Apply volume to raw PCM
            scaled = self._apply_volume(raw, volume)

            if _SD_AVAILABLE and sd is not None:
                self._play_via_sounddevice(scaled, sample_rate, n_channels, output_device)
            else:
                self._play_via_subprocess(scaled, sample_rate, n_channels)
        except Exception:
            log.warning("ChimeStore.play: unhandled error", exc_info=True)

    def _play_via_sounddevice(
        self, pcm_s16: bytes, sample_rate: int, n_channels: int, output_device
    ) -> None:
        try:
            if np is None:
                raise RuntimeError("numpy required for sounddevice playback")
            arr = np.frombuffer(pcm_s16, dtype=np.int16).astype(np.float32) / 32768.0
            if n_channels == 1:
                arr = arr.reshape(-1, 1)
            else:
                arr = arr.reshape(-1, n_channels)
            kwargs: dict = {"samplerate": sample_rate, "blocking": True}
            if output_device is not None:
                kwargs["device"] = output_device
            sd.play(arr, **kwargs)
        except Exception:
            log.warning("ChimeStore: sounddevice playback failed, falling back to subprocess", exc_info=True)
            self._play_via_subprocess(pcm_s16, sample_rate, n_channels)

    def _play_via_subprocess(self, pcm_s16: bytes, sample_rate: int, n_channels: int) -> None:
        """Fallback: write temp WAV and call aplay/paplay."""
        tmp_path: Optional[Path] = None
        try:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                tmp_path = Path(f.name)
            _write_wav(pcm_s16, tmp_path, sample_rate)
            # Try aplay first (ALSA), then paplay (PulseAudio)
            for cmd in (
                ["aplay", "-q", str(tmp_path)],
                ["paplay", str(tmp_path)],
            ):
                try:
                    result = subprocess.run(cmd, timeout=10, capture_output=True)
                    if result.returncode == 0:
                        return
                except (FileNotFoundError, subprocess.TimeoutExpired):
                    continue
            log.warning("ChimeStore: no audio player found (aplay/paplay)")
        except Exception:
            log.warning("ChimeStore: subprocess playback failed", exc_info=True)
        finally:
            if tmp_path is not None:
                try:
                    tmp_path.unlink(missing_ok=True)
                except Exception:
                    pass


# Singleton instance (lazy)
_store_lock = threading.Lock()
_store: Optional[ChimeStore] = None


def get_chime_store() -> ChimeStore:
    global _store
    if _store is None:
        with _store_lock:
            if _store is None:
                _store = ChimeStore()
    return _store


# ---------------------------------------------------------------------------
# ChimeTriggerEngine
# ---------------------------------------------------------------------------

_TRIGGERS_DIR_NAME = "chime_triggers"


class ChimeTriggerEngine:
    """
    Loads per-camera chime trigger configurations and fires chimes on events.

    A trigger config is a list of records, each with:
      - chime_id: str
      - event_types: list[str]  e.g. ["zone", "line", "tag", "detection"]
      - shape_ids: list[str]  empty = any shape
      - cooldown_sec: float
      - output_device: str|None
      - volume: float|None  (overrides chime default if set)
      - enabled: bool
    """

    def __init__(self, camera_id: str) -> None:
        self._camera_id = str(camera_id)
        self._store = get_chime_store()
        self._triggers: List[Dict] = []
        self._last_fired: Dict[str, float] = {}  # trigger_id -> timestamp
        self._lock = threading.Lock()
        self.load()

    # ------------------------------------------------------------------
    # Config I/O
    # ------------------------------------------------------------------

    def _triggers_path(self) -> Path:
        return get_data_dir() / _TRIGGERS_DIR_NAME / f"{self._camera_id}.json"

    def load(self) -> None:
        try:
            path = self._triggers_path()
            if path.exists():
                data = json.loads(path.read_text(encoding="utf-8"))
                with self._lock:
                    self._triggers = list(data.get("triggers") or [])
            else:
                with self._lock:
                    self._triggers = []
        except Exception:
            log.warning("ChimeTriggerEngine: failed to load %s", self._camera_id, exc_info=True)
            with self._lock:
                self._triggers = []

    def save(self) -> None:
        try:
            path = self._triggers_path()
            path.parent.mkdir(parents=True, exist_ok=True)
            with self._lock:
                triggers = list(self._triggers)
            path.write_text(
                json.dumps({"triggers": triggers}, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
        except Exception:
            log.warning("ChimeTriggerEngine: failed to save %s", self._camera_id, exc_info=True)

    def get_triggers(self) -> List[Dict]:
        with self._lock:
            return [dict(t) for t in self._triggers]

    def add_trigger(self, trigger: Dict) -> Dict:
        t = dict(trigger)
        if not t.get("id"):
            t["id"] = str(uuid.uuid4())
        t.setdefault("enabled", True)
        t.setdefault("cooldown_sec", 10.0)
        t.setdefault("event_types", [])
        t.setdefault("shape_ids", [])
        t.setdefault("output_device", None)
        t.setdefault("volume", None)
        with self._lock:
            self._triggers.append(t)
        self.save()
        return t

    def remove_trigger(self, trigger_id: str) -> bool:
        with self._lock:
            before = len(self._triggers)
            self._triggers = [t for t in self._triggers if t.get("id") != trigger_id]
            changed = len(self._triggers) < before
        if changed:
            self.save()
        return changed

    def update_trigger(self, trigger_id: str, updates: Dict) -> bool:
        with self._lock:
            for t in self._triggers:
                if t.get("id") == trigger_id:
                    t.update(updates)
                    break
            else:
                return False
        self.save()
        return True

    # ------------------------------------------------------------------
    # Event dispatch
    # ------------------------------------------------------------------

    def on_event(self, event_type: str, shape_id: Optional[str] = None, camera_id: Optional[str] = None) -> None:
        """
        Call this whenever a motion/detection event fires.

        event_type: one of "zone", "line", "tag", "detection", or any custom string
        shape_id:   the ID of the shape that triggered, or None
        camera_id:  optional camera_id for cross-camera filtering (unused for now)
        """
        now = time.monotonic()
        etype = str(event_type or "").lower().strip()
        sid = str(shape_id or "").strip()

        with self._lock:
            triggers = [dict(t) for t in self._triggers]

        for trig in triggers:
            try:
                if not bool(trig.get("enabled", True)):
                    continue

                # Match event type
                event_types = [str(e).lower().strip() for e in (trig.get("event_types") or [])]
                if event_types and etype not in event_types:
                    continue

                # Match shape_id
                shape_ids = [str(s).strip() for s in (trig.get("shape_ids") or [])]
                if shape_ids and sid and sid not in shape_ids:
                    continue

                # Cooldown
                tid = str(trig.get("id") or "")
                cooldown = float(trig.get("cooldown_sec") or 10.0)
                last = self._last_fired.get(tid, 0.0)
                if now - last < cooldown:
                    continue

                # Fire
                self._last_fired[tid] = now
                chime_id = str(trig.get("chime_id") or "")
                if not chime_id:
                    continue
                output_device = trig.get("output_device")
                volume_override = trig.get("volume")  # may be None
                self._store.play_chime(chime_id, output_device=output_device, volume_override=volume_override)
            except Exception:
                log.warning("ChimeTriggerEngine: error processing trigger", exc_info=True)


# Per-camera engine cache
_engines: Dict[str, ChimeTriggerEngine] = {}
_engines_lock = threading.Lock()


def get_chime_trigger_engine(camera_id: str) -> ChimeTriggerEngine:
    """Return (or create) the ChimeTriggerEngine for a given camera."""
    with _engines_lock:
        if camera_id not in _engines:
            _engines[camera_id] = ChimeTriggerEngine(camera_id)
        return _engines[camera_id]
