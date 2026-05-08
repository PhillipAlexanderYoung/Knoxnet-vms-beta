import asyncio
import threading
import time
import math
from collections import deque
from dataclasses import dataclass
from typing import Deque, List, Optional, Tuple

try:
    # NOTE: audioop was removed in Python 3.13. We only used it as a small fallback
    # for gain scaling, so keep it optional.
    import audioop  # type: ignore
except Exception:  # pragma: no cover
    audioop = None  # type: ignore

try:
    import numpy as np  # type: ignore
except Exception:  # pragma: no cover
    np = None  # type: ignore

from PySide6.QtCore import Qt, QTimer, Signal, QObject, QIODevice, QSize
from PySide6.QtGui import QColor, QPainter, QPen, QLinearGradient, QAction
from PySide6.QtWidgets import (
    QWidget,
    QHBoxLayout,
    QVBoxLayout,
    QPushButton,
    QSlider,
    QLabel,
    QSizeGrip,
    QMenu,
    QDialog,
    QFormLayout,
    QComboBox,
    QDialogButtonBox,
    QCheckBox,
)

import logging

# Reduce log spam from aiortc/aioice in desktop logs (keep errors visible).
try:
    logging.getLogger("aioice").setLevel(logging.WARNING)
    logging.getLogger("aioice.ice").setLevel(logging.WARNING)
    logging.getLogger("aiortc").setLevel(logging.WARNING)
except Exception:
    pass


class _HUDResizeHandle(QWidget):
    """
    Custom resize handle for the audio HUD.
    QSizeGrip targets the top-level window; when docked that is the whole camera widget.
    This handle resizes the HUD itself when docked, and resizes the window when undocked.
    """

    def __init__(self, hud: "AudioEQHUD"):
        super().__init__(hud)
        self._hud = hud
        self._dragging = False
        self._start_pos = None
        self._start_size = None
        self.setFixedSize(16, 16)
        self.setCursor(Qt.CursorShape.SizeFDiagCursor)
        # Prevent events from bubbling up and moving the camera widget.
        try:
            self.setAttribute(Qt.WidgetAttribute.WA_NoMousePropagation, True)
        except Exception:
            pass

    def paintEvent(self, event):
        try:
            p = QPainter(self)
            p.setRenderHint(QPainter.RenderHint.Antialiasing)
            p.fillRect(self.rect(), Qt.GlobalColor.transparent)
            pen = QPen(QColor(255, 255, 255, 65))
            pen.setWidth(1)
            p.setPen(pen)
            w = self.width()
            h = self.height()
            # diagonal ticks
            p.drawLine(w - 5, h - 1, w - 1, h - 5)
            p.drawLine(w - 9, h - 1, w - 1, h - 9)
            p.drawLine(w - 13, h - 1, w - 1, h - 13)
        except Exception:
            return

    def mousePressEvent(self, event):
        try:
            if event.button() == Qt.MouseButton.LeftButton:
                self._dragging = True
                self._start_pos = event.globalPosition().toPoint()
                self._start_size = self._hud.size()
                event.accept()
                return
        except Exception:
            pass
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        try:
            if not (self._dragging and self._start_pos is not None and self._start_size is not None):
                return super().mouseMoveEvent(event)

            delta = event.globalPosition().toPoint() - self._start_pos
            new_w = max(180, int(self._start_size.width() + delta.x()))
            new_h = max(110, int(self._start_size.height() + delta.y()))

            # Docked overlay: parent is AudioEQOverlayWidget
            overlay_cls = globals().get("AudioEQOverlayWidget")
            is_docked = bool(overlay_cls) and isinstance(self._hud.parent(), overlay_cls)
            if is_docked:
                self._hud.resize(new_w, new_h)
            else:
                # Undocked: resize the top-level window and let layout stretch HUD.
                try:
                    win = self._hud.window()
                    win.resize(new_w, new_h)
                except Exception:
                    self._hud.resize(new_w, new_h)

            event.accept()
            return
        except Exception:
            return super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        try:
            if self._dragging:
                self._dragging = False
                self._start_pos = None
                self._start_size = None
                try:
                    self._hud.settings.size_px = (int(self._hud.width()), int(self._hud.height()))
                except Exception:
                    pass
                event.accept()
                return
        except Exception:
            pass
        super().mouseReleaseEvent(event)


def _gradient_add_stop(grad: QLinearGradient, pos: float, color: QColor) -> None:
    """
    Compatibility helper: Qt gradients expose `setColorAt()`; some older code may call `addColorStop()`.
    This ensures we never crash in paint paths if an API mismatch slips in.
    """
    try:
        if hasattr(grad, "setColorAt"):
            grad.setColorAt(float(pos), color)
            return
        if hasattr(grad, "addColorStop"):
            # type: ignore[attr-defined]
            grad.addColorStop(float(pos), color)  # pragma: no cover
    except Exception:
        return

try:
    from PySide6.QtMultimedia import QAudioFormat, QAudioSink, QMediaDevices
except Exception:  # pragma: no cover
    QAudioFormat = None  # type: ignore
    QAudioSink = None  # type: ignore
    QMediaDevices = None  # type: ignore

try:
    from aiortc import RTCPeerConnection, RTCSessionDescription, RTCIceServer, RTCConfiguration
except Exception:  # pragma: no cover
    RTCPeerConnection = None  # type: ignore
    RTCSessionDescription = None  # type: ignore
    RTCIceServer = None  # type: ignore
    RTCConfiguration = None  # type: ignore

try:
    import aiohttp
except Exception:  # pragma: no cover
    aiohttp = None  # type: ignore

try:
    from av.audio.resampler import AudioResampler
except Exception:  # pragma: no cover
    AudioResampler = None  # type: ignore


@dataclass
class AudioEQSettings:
    bars: int = 48
    fft_size: int = 4096
    opacity: float = 0.70
    accent_color: str = "#00ff8c"
    bg_color: str = "#0b1220"
    visual_mode: str = "spectrum_bars"  # spectrum_bars|spectrum_line|waveform|radial|vu
    borderless: bool = False
    # overlay HUD size + position
    pos_norm: Tuple[float, float] = (0.15, 0.85)  # bottom-left-ish
    size_px: Tuple[int, int] = (280, 140)


class PCMBufferIODevice(QIODevice):
    """
    A simple ring-buffer QIODevice that feeds PCM bytes to QAudioSink.
    All writes must occur in the Qt (GUI) thread.
    """

    def __init__(self, parent=None, max_bytes: int = 48000 * 2 * 2 * 2):
        super().__init__(parent)
        self._buf = bytearray()
        self._max = int(max_bytes)
        self.open(QIODevice.OpenModeFlag.ReadOnly)

    def clear(self) -> None:
        self._buf.clear()

    def append_pcm(self, data: bytes) -> None:
        if not data:
            return
        self._buf.extend(data)
        # keep buffer bounded (drop oldest)
        if len(self._buf) > self._max:
            drop = len(self._buf) - self._max
            del self._buf[:drop]

    def bytesAvailable(self) -> int:  # type: ignore[override]
        return len(self._buf) + super().bytesAvailable()

    def readData(self, maxlen: int) -> bytes:  # type: ignore[override]
        if maxlen <= 0 or not self._buf:
            return b""
        n = min(int(maxlen), len(self._buf))
        out = bytes(self._buf[:n])
        del self._buf[:n]
        return out

    def writeData(self, data: bytes) -> int:  # type: ignore[override]
        # Read-only device for QAudioSink
        return 0


class AudioPlayback(QObject):
    """QtMultimedia-backed audio playback for s16le PCM."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.enabled = (QAudioSink is not None and QMediaDevices is not None and QAudioFormat is not None)
        self._sink = None
        self._device = None
        self._format = None
        self._volume = 0.8
        self._muted = False
        self._target_rate = 48000
        self._target_channels = 2

    @property
    def target_format(self) -> Tuple[int, int]:
        return self._target_rate, self._target_channels

    def start(self) -> None:
        if not self.enabled:
            return
        if self._sink is not None:
            return

        fmt = QAudioFormat()
        fmt.setSampleRate(self._target_rate)
        fmt.setChannelCount(self._target_channels)
        fmt.setSampleFormat(QAudioFormat.SampleFormat.Int16)

        dev = QMediaDevices.defaultAudioOutput()
        if not dev.isFormatSupported(fmt):
            # best effort: still attempt to start, Qt will do its best or fail gracefully
            pass

        self._format = fmt
        self._device = PCMBufferIODevice()
        self._sink = QAudioSink(dev, fmt)
        self._sink.setVolume(0.0 if self._muted else float(self._volume))
        self._sink.start(self._device)

    def stop(self) -> None:
        try:
            if self._sink is not None:
                self._sink.stop()
        except Exception:
            pass
        self._sink = None
        try:
            if self._device is not None:
                self._device.close()
        except Exception:
            pass
        self._device = None

    def set_volume(self, v01: float) -> None:
        self._volume = max(0.0, min(1.0, float(v01)))
        try:
            if self._sink is not None and not self._muted:
                self._sink.setVolume(float(self._volume))
        except Exception:
            pass

    def set_muted(self, muted: bool) -> None:
        self._muted = bool(muted)
        try:
            if self._sink is not None:
                self._sink.setVolume(0.0 if self._muted else float(self._volume))
        except Exception:
            pass

    def push_pcm(self, pcm_s16le: bytes, gain: float = 1.0) -> None:
        """
        Append PCM to the internal QIODevice. Must be called in GUI thread.
        gain is applied in int16 domain (best-effort).
        """
        if not pcm_s16le:
            return
        if not self.enabled:
            return
        if self._sink is None or self._device is None:
            self.start()
        if self._device is None:
            return
        if self._muted:
            return

        g = max(0.0, min(2.0, float(gain)))
        if abs(g - 1.0) < 1e-3:
            self._device.append_pcm(pcm_s16le)
            return

        # scale int16 samples (prefer numpy; fallback to stdlib audioop)
        if np is not None:
            try:
                a = np.frombuffer(pcm_s16le, dtype=np.int16).astype(np.float32) * g
                a = np.clip(a, -32768, 32767).astype(np.int16)
                self._device.append_pcm(a.tobytes())
                return
            except Exception:
                pass
        try:
            if audioop is not None:
                self._device.append_pcm(audioop.mul(pcm_s16le, 2, g))
            else:
                self._device.append_pcm(pcm_s16le)
        except Exception:
            self._device.append_pcm(pcm_s16le)


class AudioWHEPReceiver(QObject):
    """
    WHEP (recvonly) audio receiver.
    Runs aiortc in a dedicated asyncio loop thread and emits:
    - pcm_ready(bytes): resampled PCM s16le stereo @ 48k
    - spectrum_ready(list[float]): bars normalized 0..1
    """

    state_changed = Signal(str)        # "idle" | "connecting" | "connected" | "error"
    error = Signal(str)
    pcm_ready = Signal(bytes)
    spectrum_ready = Signal(object)    # List[float]
    waveform_ready = Signal(object)    # List[float] in [-1..1]

    def __init__(self, parent=None, bars: int = 48, fft_size: int = 4096):
        super().__init__(parent)
        self._bars = int(max(8, min(128, bars)))
        self._fft_size = int(max(512, min(16384, fft_size)))

        self._thread: Optional[threading.Thread] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._stop_evt = threading.Event()
        self._pc = None
        self._task: Optional[asyncio.Task] = None

        self._mono_buf: Deque[float] = deque(maxlen=self._fft_size * 2)
        self._last_fft_ts = 0.0
        self._smoothed = None
        self._wave_smoothed: Optional[List[float]] = None

        # For numpy-less spectrum fallback (Goertzel), we use a smaller fixed window
        self._fallback_n = 1024
        self._fallback_fs = 48000

        self._api_base: Optional[str] = None
        self._camera_id: Optional[str] = None

    @property
    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive() and not self._stop_evt.is_set()

    def start(self, api_base: str, camera_id: str) -> None:
        if not api_base or not camera_id:
            return
        self._api_base = str(api_base).rstrip("/")
        self._camera_id = str(camera_id)
        # If an old thread is still shutting down, wait briefly so we can restart cleanly.
        try:
            if self._thread is not None and self._thread.is_alive():
                if self._stop_evt.is_set():
                    try:
                        self._thread.join(timeout=1.0)
                    except Exception:
                        pass
                # If it's still alive, don't try to start a second loop.
                if self._thread.is_alive():
                    return
        except Exception:
            pass

        if self.is_running:
            return

        self._stop_evt.clear()
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_evt.set()
        try:
            if self._loop and self._loop.is_running():
                self._loop.call_soon_threadsafe(lambda: None)
        except Exception:
            pass
        # Best-effort join to avoid restart races when switching dock/undock quickly.
        try:
            if self._thread is not None and self._thread.is_alive():
                self._thread.join(timeout=1.0)
        except Exception:
            pass

    def _run_loop(self) -> None:
        self.state_changed.emit("connecting")
        loop = asyncio.new_event_loop()
        self._loop = loop
        asyncio.set_event_loop(loop)
        try:
            self._task = loop.create_task(self._connect_and_consume())
            loop.run_until_complete(self._task)
        except Exception as e:
            try:
                self.error.emit(str(e))
            except Exception:
                pass
        finally:
            try:
                pending = asyncio.all_tasks(loop)
                for t in pending:
                    t.cancel()
                if pending:
                    loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
            except Exception:
                pass
            try:
                loop.stop()
            except Exception:
                pass
            try:
                loop.close()
            except Exception:
                pass
            self._loop = None
            self._task = None
            self._pc = None
            self.state_changed.emit("idle")

    async def _connect_and_consume(self) -> None:
        if RTCPeerConnection is None or aiohttp is None or AudioResampler is None:
            self.state_changed.emit("error")
            self.error.emit("Missing dependencies (aiortc/aiohttp/av). Install requirements.txt.")
            return
        api_base = self._api_base or ""
        cam_id = self._camera_id or ""
        if not api_base or not cam_id:
            self.state_changed.emit("error")
            self.error.emit("Invalid API base or camera id")
            return

        # Resolve WHEP URL from backend (same as React)
        whep_url = ""
        try:
            async with aiohttp.ClientSession() as s:
                async with s.get(f"{api_base}/audio/cameras/{cam_id}/whep") as r:
                    body = await r.json(content_type=None)
                    if not r.ok:
                        raise RuntimeError(f"WHEP resolver failed ({r.status})")
                    whep_url = ((body or {}).get("data") or {}).get("whep") or ""
        except Exception as e:
            self.state_changed.emit("error")
            self.error.emit(f"Failed to resolve WHEP endpoint: {e}")
            return

        if not whep_url:
            self.state_changed.emit("error")
            self.error.emit("Backend returned empty WHEP URL")
            return

        cfg = RTCConfiguration(iceServers=[RTCIceServer(urls=["stun:stun.l.google.com:19302"])])
        pc = RTCPeerConnection(configuration=cfg)
        self._pc = pc

        # Ensure we negotiate an audio m-line
        pc.addTransceiver("audio", direction="recvonly")

        @pc.on("connectionstatechange")
        async def _on_state():
            try:
                if pc.connectionState == "connected":
                    self.state_changed.emit("connected")
                elif pc.connectionState == "failed":
                    self.state_changed.emit("error")
                    try:
                        self.error.emit("ICE connection failed (no reachable candidate pair).")
                    except Exception:
                        pass
                elif pc.connectionState == "closed":
                    self.state_changed.emit("idle")
            except Exception:
                pass

        track_holder = {"track": None}

        @pc.on("track")
        def _on_track(track):
            if track.kind == "audio":
                track_holder["track"] = track

        offer = await pc.createOffer()
        await pc.setLocalDescription(offer)

        # Wait for ICE gather for non-trickle WHEP. Some environments need >1.5s.
        try:
            start = time.time()
            while pc.iceGatheringState != "complete" and (time.time() - start) < 4.0:
                await asyncio.sleep(0.05)
        except Exception:
            pass

        local_sdp = (pc.localDescription.sdp if pc.localDescription else "") or ""
        if not local_sdp:
            self.state_changed.emit("error")
            self.error.emit("Failed to create local SDP offer")
            await pc.close()
            return

        # Defensive: ensure ICE credentials are present in SDP (some servers reject otherwise)
        if "a=ice-ufrag" not in local_sdp:
            try:
                start = time.time()
                while "a=ice-ufrag" not in local_sdp and (time.time() - start) < 2.0:
                    await asyncio.sleep(0.05)
                    local_sdp = (pc.localDescription.sdp if pc.localDescription else "") or ""
            except Exception:
                pass

        answer_sdp = ""
        try:
            async with aiohttp.ClientSession() as s:
                async with s.post(
                    whep_url,
                    data=local_sdp,
                    headers={"Content-Type": "application/sdp", "Accept": "application/sdp"},
                ) as r:
                    if not r.ok:
                        txt = await r.text()
                        raise RuntimeError(f"WHEP POST failed ({r.status}): {txt}")
                    answer_sdp = await r.text()
        except Exception as e:
            self.state_changed.emit("error")
            self.error.emit(str(e))
            await pc.close()
            return

        await pc.setRemoteDescription(RTCSessionDescription(sdp=answer_sdp, type="answer"))

        # Wait a bit for audio track
        for _ in range(40):
            if self._stop_evt.is_set():
                await pc.close()
                return
            if track_holder["track"] is not None:
                break
            await asyncio.sleep(0.05)

        track = track_holder["track"]
        if track is None:
            self.state_changed.emit("error")
            self.error.emit("No audio track received (camera may not publish audio)")
            await pc.close()
            return

        self.state_changed.emit("connected")

        # Resample to 48k stereo s16 for playback + visualization consistency
        resampler = AudioResampler(format="s16", layout="stereo", rate=48000)

        try:
            while not self._stop_evt.is_set():
                frame = await track.recv()
                if frame is None:
                    await asyncio.sleep(0.01)
                    continue
                for out in resampler.resample(frame):
                    try:
                        pcm = out.planes[0].to_bytes()
                    except Exception:
                        # fallback via numpy
                        arr = out.to_ndarray()
                        if arr.ndim == 2 and arr.shape[0] == 2:  # planar
                            arr = arr.T
                        pcm = arr.astype(np.int16).tobytes()

                    # Push PCM for playback
                    self.pcm_ready.emit(pcm)

                    # Update spectrum at ~30fps
                    now = time.time()
                    if (now - self._last_fft_ts) < (1.0 / 30.0):
                        continue
                    self._last_fft_ts = now
                    self._update_spectrum_from_pcm(pcm)
                    self._update_waveform_from_pcm(pcm)
        except Exception as e:
            if not self._stop_evt.is_set():
                self.error.emit(f"Audio receive error: {e}")
        finally:
            try:
                await pc.close()
            except Exception:
                pass

    def _update_spectrum_from_pcm(self, pcm_s16le_stereo: bytes) -> None:
        # Prefer numpy FFT if available; otherwise use a lightweight Goertzel filterbank.
        if np is not None:
            try:
                a = np.frombuffer(pcm_s16le_stereo, dtype=np.int16)
                if a.size < 8:
                    return
                # interleaved stereo -> mono
                a = a.reshape((-1, 2)).mean(axis=1).astype(np.float32) / 32768.0
                for v in a.tolist():
                    self._mono_buf.append(float(v))
                if len(self._mono_buf) < self._fft_size:
                    return
                buf = np.array(list(self._mono_buf)[-self._fft_size :], dtype=np.float32)
                window = np.hanning(buf.size).astype(np.float32)
                spec = np.abs(np.fft.rfft(buf * window))
                if spec.size < 4:
                    return
                # log scaling + normalize
                spec = np.log10(1.0 + spec)
                spec = spec / (np.max(spec) + 1e-9)

                # map to bars using a gentle log index curve similar to React's
                bars = np.zeros((self._bars,), dtype=np.float32)
                n = spec.size
                for i in range(self._bars):
                    t = i / max(1, (self._bars - 1))
                    idx = int(min(n - 1, max(0, (t**2.2) * (n - 1))))
                    j0 = max(0, idx - 1)
                    j1 = min(n - 1, idx + 1)
                    bars[i] = float(np.mean(spec[j0 : j1 + 1]))

                if self._smoothed is None or not isinstance(self._smoothed, np.ndarray) or self._smoothed.shape != bars.shape:
                    self._smoothed = bars
                else:
                    self._smoothed = self._smoothed * 0.70 + bars * 0.30
                out = np.clip(self._smoothed, 0.0, 1.0).tolist()
                self.spectrum_ready.emit(out)
                return
            except Exception:
                # fall through to Goertzel
                pass

        try:
            # Decode interleaved int16 stereo to mono float samples
            from array import array
            arr = array("h")
            arr.frombytes(pcm_s16le_stereo)
            if len(arr) < 8:
                return
            # downmix to mono
            for i in range(0, len(arr) - 1, 2):
                self._mono_buf.append(((arr[i] + arr[i + 1]) * 0.5) / 32768.0)

            n = min(self._fallback_n, len(self._mono_buf))
            if n < 256:
                return
            xs = list(self._mono_buf)[-n:]

            fs = self._fallback_fs
            f_min = 60.0
            f_max = 10000.0

            powers: List[float] = []
            max_p = 0.0
            for bi in range(self._bars):
                t = bi / max(1.0, (self._bars - 1))
                freq = f_min * ((f_max / f_min) ** t)
                k = int(0.5 + (n * freq / fs))
                if k <= 0:
                    k = 1
                if k >= n // 2:
                    k = max(1, (n // 2) - 1)
                omega = (2.0 * math.pi * k) / n
                coeff = 2.0 * math.cos(omega)
                s0 = s1 = s2 = 0.0
                for x in xs:
                    s0 = x + coeff * s1 - s2
                    s2 = s1
                    s1 = s0
                pwr = max(0.0, (s1 * s1 + s2 * s2 - coeff * s1 * s2))
                powers.append(pwr)
                if pwr > max_p:
                    max_p = pwr

            if max_p <= 1e-12:
                out = [0.0] * self._bars
            else:
                # log-ish scaling + normalize
                out = [min(1.0, math.log10(1.0 + (p / max_p) * 50.0)) for p in powers]
                # normalize again
                m2 = max(out) if out else 1.0
                if m2 > 1e-9:
                    out = [v / m2 for v in out]

            # smooth
            if self._smoothed is None or not isinstance(self._smoothed, list) or len(self._smoothed) != len(out):
                self._smoothed = list(out)
            else:
                self._smoothed = [self._smoothed[i] * 0.70 + out[i] * 0.30 for i in range(len(out))]
            self.spectrum_ready.emit([max(0.0, min(1.0, float(v))) for v in self._smoothed])
        except Exception:
            return

    def _update_waveform_from_pcm(self, pcm_s16le_stereo: bytes, points: int = 256) -> None:
        """
        Compute a downsampled mono waveform ([-1..1]) for visualization.
        Emitted at the same cadence as spectrum (~30fps).
        """
        try:
            if not pcm_s16le_stereo:
                return
            pts = int(max(64, min(1024, points)))

            # Prefer numpy for speed
            if np is not None:
                a = np.frombuffer(pcm_s16le_stereo, dtype=np.int16)
                if a.size < 8:
                    return
                a = a.reshape((-1, 2)).mean(axis=1).astype(np.float32) / 32768.0  # mono
                n = int(a.size)
                if n <= pts:
                    wave = a
                else:
                    # bucket-average to pts points (more stable than point sampling)
                    edges = np.linspace(0, n, num=pts + 1).astype(np.int32)
                    outw = np.zeros((pts,), dtype=np.float32)
                    for i in range(pts):
                        s0 = int(edges[i])
                        s1 = int(edges[i + 1])
                        if s1 <= s0 + 1:
                            outw[i] = float(a[min(n - 1, s0)])
                        else:
                            outw[i] = float(np.mean(a[s0:s1]))
                    wave = outw
                out = np.clip(wave, -1.0, 1.0).astype(np.float32).tolist()
            else:
                # Fallback: cheap int16 stepping
                from array import array
                arr = array("h")
                arr.frombytes(pcm_s16le_stereo)
                if len(arr) < 8:
                    return
                # interleaved stereo -> mono sampled
                mono = []
                step = max(1, int((len(arr) // 2) / pts))
                i = 0
                while i < len(arr) - 1 and len(mono) < pts:
                    m = (arr[i] + arr[i + 1]) * 0.5 / 32768.0
                    mono.append(max(-1.0, min(1.0, float(m))))
                    i += step * 2
                out = mono

            # Smooth a bit to avoid jitter
            if self._wave_smoothed is None or len(self._wave_smoothed) != len(out):
                self._wave_smoothed = list(out)
            else:
                self._wave_smoothed = [self._wave_smoothed[i] * 0.55 + out[i] * 0.45 for i in range(len(out))]
            self.waveform_ready.emit(self._wave_smoothed)
        except Exception:
            return


class AudioEQVisualizer(QWidget):
    """Transparent bar visualizer."""

    def __init__(self, parent=None, bars: int = 48, accent: str = "#00ff8c"):
        super().__init__(parent)
        self._bars = int(max(8, min(128, bars)))
        self._values: List[float] = [0.0] * self._bars
        self._accent = QColor(accent)
        self.setMinimumSize(QSize(140, 60))
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self.setAttribute(Qt.WidgetAttribute.WA_OpaquePaintEvent, False)
        self.setAutoFillBackground(False)

    def set_accent(self, color: str) -> None:
        c = QColor(str(color))
        if c.isValid():
            self._accent = c
            self.update()

    def set_values(self, values: List[float]) -> None:
        if not values:
            return
        if len(values) != self._bars:
            self._bars = len(values)
        self._values = [max(0.0, min(1.0, float(v))) for v in values]
        self.update()

    def paintEvent(self, event):
        # NOTE: Never allow exceptions to escape a paintEvent; Qt may abort the process.
        try:
            p = QPainter(self)
            p.setRenderHint(QPainter.RenderHint.Antialiasing)
            p.fillRect(self.rect(), Qt.GlobalColor.transparent)

            w = max(1, self.width())
            h = max(1, self.height())

            # subtle grid
            grid_pen = QPen(QColor(0, 255, 128, 26))
            grid_pen.setWidth(1)
            p.setPen(grid_pen)
            for r in range(1, 4):
                yy = int((h / 4) * r)
                p.drawLine(0, yy, w, yy)

            bars = max(1, len(self._values))
            gap = max(1, int(w / (bars * 18)))
            bar_w = max(2, int((w - gap * (bars - 1)) / bars))

            # Auto-gain so peaks reach full height (fixes "half height" look).
            peak = max(self._values) if self._values else 0.0
            if peak <= 1e-6:
                peak = 1.0

            for i, v in enumerate(self._values):
                mag = max(0.0, min(1.0, float(v) / peak))
                bh = max(2, int(mag * h))
                x = int(i * (bar_w + gap))
                y = int(h - bh)

                grad = QLinearGradient(0, y, 0, h)
                a = self._accent
                _gradient_add_stop(grad, 0.0, QColor(a.red(), a.green(), a.blue(), 235))
                _gradient_add_stop(grad, 0.7, QColor(int(a.red() * 0.7), int(a.green() * 0.7), int(a.blue() * 0.7), 210))
                _gradient_add_stop(grad, 1.0, QColor(int(a.red() * 0.45), int(a.green() * 0.45), int(a.blue() * 0.45), 180))
                p.fillRect(x, y, bar_w, bh, grad)
        except Exception:
            return


class AudioEQHUD(QWidget):
    """Movable/resizable HUD container used by overlay and undocked window."""

    play_clicked = Signal()
    mute_toggled = Signal(bool)
    volume_changed = Signal(float)  # 0..1
    close_clicked = Signal()
    record_clicked = Signal()
    monitor_menu_requested = Signal()

    def __init__(self, parent=None, settings: Optional[AudioEQSettings] = None):
        super().__init__(parent)
        self.settings = settings or AudioEQSettings()
        self._playing = False
        self._muted = False
        self._volume = 0.80

        self.setObjectName("AudioEQHUD")
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self.setAttribute(Qt.WidgetAttribute.WA_OpaquePaintEvent, False)
        self.setAutoFillBackground(False)

        self.setMouseTracking(True)
        self._hovering = False
        self._root = QVBoxLayout(self)
        self._root.setContentsMargins(10, 10, 10, 10)
        self._root.setSpacing(8)

        self._top_row = QWidget(self)
        top = QHBoxLayout(self._top_row)
        top.setContentsMargins(0, 0, 0, 0)
        top.setSpacing(8)

        self.title = QLabel("Audio")
        self.title.setStyleSheet("color: rgba(255,255,255,0.85); font-weight: 600;")
        top.addWidget(self.title)
        top.addStretch()

        # Minimal controls; adjustments live in the settings menu.
        self.btn_settings = QPushButton("⚙")
        self.btn_settings.setFixedSize(28, 24)
        self.btn_settings.clicked.connect(self._open_settings_menu)
        top.addWidget(self.btn_settings)

        self.btn_play = QPushButton("▶")
        self.btn_play.setFixedSize(28, 24)
        self.btn_play.clicked.connect(lambda: self.play_clicked.emit())
        top.addWidget(self.btn_play)

        # Mute moved into settings menu to keep overlay compact.

        self.btn_close = QPushButton("×")
        self.btn_close.setFixedSize(28, 24)
        self.btn_close.clicked.connect(lambda: self.close_clicked.emit())
        top.addWidget(self.btn_close)

        self._root.addWidget(self._top_row)

        self.viz = AudioEQVisualizer(self, bars=self.settings.bars, accent=self.settings.accent_color)
        self.line = AudioSpectrumLineVisualizer(self, bars=self.settings.bars, accent=self.settings.accent_color)
        self.wave = AudioWaveformVisualizer(self, accent=self.settings.accent_color)
        self.radial = AudioRadialWaveformVisualizer(self, accent=self.settings.accent_color)
        self.vu = AudioVUMeterVisualizer(self, accent=self.settings.accent_color)
        self._root.addWidget(self.viz, 1)
        self._root.addWidget(self.line, 1)
        self._root.addWidget(self.wave, 1)
        self._root.addWidget(self.radial, 1)
        self._root.addWidget(self.vu, 1)
        self._sync_mode_widgets()

        # Custom handle so docked resizing doesn't resize/move the whole camera widget.
        self.grip = _HUDResizeHandle(self)

        self._apply_style()

    def _apply_style(self) -> None:
        bg = QColor(self.settings.bg_color)
        if not bg.isValid():
            bg = QColor("#0b1220")
        # Opacity is intended to make the HUD *nearly transparent* if desired.
        # 0.00 = fully transparent, 0.95 = near-opaque.
        op = max(0.0, min(0.95, float(self.settings.opacity)))
        bg_a = int(op * 255)
        rgba = f"rgba({bg.red()}, {bg.green()}, {bg.blue()}, {bg_a})"
        accent = QColor(self.settings.accent_color)
        if not accent.isValid():
            accent = QColor("#00ff8c")
        # Fade borders with opacity so the widget can become truly subtle.
        border_a = int(max(0, min(60, 50 * op)))  # ~=35 alpha at op=0.70
        border_css = "none" if self.settings.borderless else f"1px solid rgba(255,255,255,{border_a})"
        bg_css = "transparent" if self.settings.borderless else rgba
        self.setStyleSheet(
            f"""
            QWidget#AudioEQHUD {{
                background-color: {bg_css};
                border: {border_css};
                border-radius: 10px;
            }}
            QPushButton {{
                background-color: rgba(0,0,0,0);
                color: rgba(255,255,255,0.85);
                border: {"none" if self.settings.borderless else f"1px solid rgba(255,255,255,{border_a})"};
                border-radius: 6px;
                font-weight: 700;
            }}
            QPushButton:hover {{
                border: 1px solid rgba({accent.red()},{accent.green()},{accent.blue()},220);
            }}
            QSlider::groove:horizontal {{
                height: 6px;
                background: rgba(255,255,255,20);
                border-radius: 3px;
            }}
            QSlider::handle:horizontal {{
                width: 10px;
                margin: -6px 0;
                border-radius: 5px;
                background: rgba({accent.red()},{accent.green()},{accent.blue()},220);
            }}
            """
        )
        # Borderless: visuals should truly fill the HUD and controls appear only on hover.
        try:
            self._root.setContentsMargins(0, 0, 0, 0) if self.settings.borderless else self._root.setContentsMargins(10, 10, 10, 10)
        except Exception:
            pass
        self._sync_borderless_controls()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        try:
            self.grip.move(self.width() - self.grip.width(), self.height() - self.grip.height())
        except Exception:
            pass

    def set_playing(self, playing: bool) -> None:
        self._playing = bool(playing)
        self.btn_play.setText("⏸" if self._playing else "▶")

    def _toggle_mute(self) -> None:
        self._muted = not self._muted
        self.mute_toggled.emit(self._muted)

    def _set_volume(self, v01: float) -> None:
        self._volume = max(0.0, min(1.0, float(v01)))
        self.volume_changed.emit(self._volume)

    def set_spectrum(self, values: List[float]) -> None:
        try:
            self.viz.set_values(values)
            self.line.set_values(values)
        except Exception:
            pass

    def set_waveform(self, values: List[float]) -> None:
        try:
            self.wave.set_values(values)
            self.radial.set_values(values)
            self.vu.set_values(values)
        except Exception:
            pass

    def _sync_mode_widgets(self) -> None:
        mode = (self.settings.visual_mode or "spectrum_bars").strip().lower()
        if mode not in ("spectrum_bars", "spectrum_line", "waveform", "radial", "vu"):
            mode = "spectrum_bars"
            self.settings.visual_mode = mode
        try:
            self.viz.setVisible(mode == "spectrum_bars")
            self.line.setVisible(mode == "spectrum_line")
            self.wave.setVisible(mode == "waveform")
            self.radial.setVisible(mode == "radial")
            self.vu.setVisible(mode == "vu")
        except Exception:
            pass

    def _open_settings_menu(self) -> None:
        try:
            menu = QMenu(self)

            # Visualization submenu
            viz_menu = menu.addMenu("Visual")
            modes = [
                ("Spectrum Bars", "spectrum_bars"),
                ("Spectrum Line", "spectrum_line"),
                ("Waveform", "waveform"),
                ("Radial", "radial"),
                ("VU Meter", "vu"),
            ]
            for label, key in modes:
                act = QAction(label, self)
                act.setCheckable(True)
                act.setChecked((self.settings.visual_mode or "") == key)
                act.triggered.connect(lambda _=False, k=key: self._set_visual_mode(k))
                viz_menu.addAction(act)

            menu.addSeparator()

            border_act = QAction("Borderless", self)
            border_act.setCheckable(True)
            border_act.setChecked(bool(self.settings.borderless))
            border_act.triggered.connect(lambda checked: self._set_borderless(bool(checked)))
            menu.addAction(border_act)

            menu.addSeparator()

            rec_act = QAction("Record Clip…", self)
            rec_act.triggered.connect(lambda: self.record_clicked.emit())
            menu.addAction(rec_act)

            mon_act = QAction("Audio Monitor…", self)
            mon_act.triggered.connect(lambda: self.monitor_menu_requested.emit())
            menu.addAction(mon_act)

            menu.addSeparator()

            settings_act = QAction("Audio Settings…", self)
            settings_act.triggered.connect(self._open_settings_dialog)
            menu.addAction(settings_act)

            menu.exec(self.btn_settings.mapToGlobal(self.btn_settings.rect().bottomLeft()))
        except Exception:
            return

    def _set_visual_mode(self, mode: str) -> None:
        self.settings.visual_mode = str(mode)
        self._sync_mode_widgets()

    def _set_borderless(self, enabled: bool) -> None:
        self.settings.borderless = bool(enabled)
        self._apply_style()
        self._sync_borderless_controls()

    def _sync_borderless_controls(self) -> None:
        """Hide overlay buttons unless hovering in borderless mode."""
        try:
            if not bool(self.settings.borderless):
                self._top_row.setVisible(True)
                self.grip.setVisible(True)
                self.title.setVisible(True)
                return
            # borderless: controls appear only on hover
            show = bool(self._hovering)
            self._top_row.setVisible(show)
            self.grip.setVisible(show)
            self.title.setVisible(False)
        except Exception:
            return

    def enterEvent(self, event):  # type: ignore[override]
        try:
            self._hovering = True
            self._sync_borderless_controls()
        except Exception:
            pass
        return super().enterEvent(event)

    def leaveEvent(self, event):  # type: ignore[override]
        try:
            self._hovering = False
            self._sync_borderless_controls()
        except Exception:
            pass
        return super().leaveEvent(event)

    def _open_settings_dialog(self) -> None:
        """
        Small settings dialog for volume + appearance.
        Kept minimal for production use.
        """
        try:
            dlg = QDialog(self)
            dlg.setWindowTitle("Audio Settings")
            layout = QVBoxLayout(dlg)
            form = QFormLayout()
            layout.addLayout(form)

            vol = QSlider(Qt.Orientation.Horizontal)
            vol.setRange(0, 100)
            vol.setValue(int(self._volume * 100))
            form.addRow("Volume", vol)

            mute = QCheckBox("Mute")
            mute.setChecked(bool(self._muted))
            form.addRow("", mute)

            opacity = QSlider(Qt.Orientation.Horizontal)
            # Allow nearly transparent panels
            opacity.setRange(0, 95)
            opacity.setValue(int(max(0.0, min(0.95, float(self.settings.opacity))) * 100))
            form.addRow("Opacity", opacity)

            buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
            layout.addWidget(buttons)
            buttons.accepted.connect(dlg.accept)
            buttons.rejected.connect(dlg.reject)

            if dlg.exec() == QDialog.DialogCode.Accepted:
                self._set_volume(vol.value() / 100.0)
                if bool(mute.isChecked()) != bool(self._muted):
                    self._toggle_mute()
                self.settings.opacity = float(opacity.value() / 100.0)
                self._apply_style()
        except Exception:
            return


class AudioWaveformVisualizer(QWidget):
    """Transparent oscilloscope-style waveform visualizer."""

    def __init__(self, parent=None, accent: str = "#00ff8c"):
        super().__init__(parent)
        self._values: List[float] = [0.0] * 256
        self._accent = QColor(accent)
        self.setMinimumSize(QSize(140, 60))
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self.setAttribute(Qt.WidgetAttribute.WA_OpaquePaintEvent, False)
        self.setAutoFillBackground(False)

    def set_accent(self, color: str) -> None:
        c = QColor(str(color))
        if c.isValid():
            self._accent = c
            self.update()

    def set_values(self, values: List[float]) -> None:
        if not values:
            return
        self._values = [max(-1.0, min(1.0, float(v))) for v in values]
        self.update()

    def paintEvent(self, event):
        try:
            p = QPainter(self)
            p.setRenderHint(QPainter.RenderHint.Antialiasing)
            p.fillRect(self.rect(), Qt.GlobalColor.transparent)

            w = max(1, self.width())
            h = max(1, self.height())
            mid = h / 2.0

            # baseline
            base_pen = QPen(QColor(255, 255, 255, 28))
            base_pen.setWidth(1)
            p.setPen(base_pen)
            p.drawLine(0, int(mid), w, int(mid))

            vals = self._values or [0.0]
            n = max(2, len(vals))
            # auto-gain to use full height
            peak = max(abs(float(v)) for v in vals) if vals else 0.0
            if peak <= 1e-6:
                peak = 1.0

            a = self._accent
            wave_pen = QPen(QColor(a.red(), a.green(), a.blue(), 220))
            wave_pen.setWidth(2)
            p.setPen(wave_pen)

            prev_x = 0
            prev_y = int(mid - (float(vals[0]) / peak) * (mid - 3))
            for i in range(1, n):
                x = int((i / (n - 1)) * (w - 1))
                y = int(mid - (float(vals[i]) / peak) * (mid - 3))
                p.drawLine(prev_x, prev_y, x, y)
                prev_x, prev_y = x, y
        except Exception:
            return


class AudioSpectrumLineVisualizer(QWidget):
    """Line plot of spectrum bars."""

    def __init__(self, parent=None, bars: int = 48, accent: str = "#00ff8c"):
        super().__init__(parent)
        self._bars = int(max(8, min(128, bars)))
        self._values: List[float] = [0.0] * self._bars
        self._accent = QColor(accent)
        self.setMinimumSize(QSize(140, 60))
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self.setAttribute(Qt.WidgetAttribute.WA_OpaquePaintEvent, False)
        self.setAutoFillBackground(False)

    def set_values(self, values: List[float]) -> None:
        if not values:
            return
        self._values = [max(0.0, min(1.0, float(v))) for v in values]
        self.update()

    def paintEvent(self, event):
        try:
            p = QPainter(self)
            p.setRenderHint(QPainter.RenderHint.Antialiasing)
            p.fillRect(self.rect(), Qt.GlobalColor.transparent)
            w = max(1, self.width())
            h = max(1, self.height())
            vals = self._values or [0.0]
            n = max(2, len(vals))
            peak = max(vals) if vals else 0.0
            if peak <= 1e-6:
                peak = 1.0
            a = self._accent
            pen = QPen(QColor(a.red(), a.green(), a.blue(), 220))
            pen.setWidth(2)
            p.setPen(pen)
            prev_x = 0
            prev_y = int(h - (float(vals[0]) / peak) * (h - 3))
            for i in range(1, n):
                x = int((i / (n - 1)) * (w - 1))
                y = int(h - (float(vals[i]) / peak) * (h - 3))
                p.drawLine(prev_x, prev_y, x, y)
                prev_x, prev_y = x, y
        except Exception:
            return


class AudioRadialWaveformVisualizer(QWidget):
    """Radial waveform ring."""

    def __init__(self, parent=None, accent: str = "#00ff8c"):
        super().__init__(parent)
        self._values: List[float] = [0.0] * 256
        self._accent = QColor(accent)
        self.setMinimumSize(QSize(140, 60))
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self.setAttribute(Qt.WidgetAttribute.WA_OpaquePaintEvent, False)
        self.setAutoFillBackground(False)

    def set_values(self, values: List[float]) -> None:
        if not values:
            return
        self._values = [max(-1.0, min(1.0, float(v))) for v in values]
        self.update()

    def paintEvent(self, event):
        try:
            p = QPainter(self)
            p.setRenderHint(QPainter.RenderHint.Antialiasing)
            p.fillRect(self.rect(), Qt.GlobalColor.transparent)
            w = max(1, self.width())
            h = max(1, self.height())
            cx = w / 2.0
            cy = h / 2.0
            r0 = max(6.0, min(w, h) * 0.22)
            r1 = max(r0 + 2.0, min(w, h) * 0.42)
            vals = self._values or [0.0]
            n = max(16, len(vals))
            a = self._accent
            pen = QPen(QColor(a.red(), a.green(), a.blue(), 210))
            pen.setWidth(2)
            p.setPen(pen)
            # draw radial segments
            for i in range(n):
                t = (i / n) * (2.0 * math.pi)
                v = abs(float(vals[i % len(vals)]))
                rr = r0 + (r1 - r0) * v
                x0 = cx + math.cos(t) * r0
                y0 = cy + math.sin(t) * r0
                x1 = cx + math.cos(t) * rr
                y1 = cy + math.sin(t) * rr
                p.drawLine(int(x0), int(y0), int(x1), int(y1))
        except Exception:
            return


class AudioVUMeterVisualizer(QWidget):
    """Simple VU meter derived from waveform energy."""

    def __init__(self, parent=None, accent: str = "#00ff8c"):
        super().__init__(parent)
        self._level = 0.0
        # Adaptive reference (AGC-ish) so the meter uses the full scale with real-world audio.
        self._ref_rms = 0.03
        self._accent = QColor(accent)
        self.setMinimumSize(QSize(140, 60))
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self.setAttribute(Qt.WidgetAttribute.WA_OpaquePaintEvent, False)
        self.setAutoFillBackground(False)

    def set_values(self, values: List[float]) -> None:
        try:
            if not values:
                return
            # RMS-ish
            s = 0.0
            for v in values:
                fv = float(v)
                s += fv * fv
            rms = math.sqrt(s / max(1, len(values)))
            # Adaptive normalization:
            # - ref rises quickly on peaks
            # - ref decays slowly so quiet audio still shows movement
            ref = float(self._ref_rms)
            # slow decay
            ref = max(0.01, ref * 0.995)
            # fast attack
            if rms > ref:
                ref = rms
            self._ref_rms = ref

            # Normalize and apply gentle curve to make low levels more visible.
            norm = rms / max(1e-6, ref)
            norm = max(0.0, min(1.0, norm))
            # gamma (<1) expands low values
            target = float(norm ** 0.6)

            # meter smoothing (fast attack, slower release)
            if target >= self._level:
                self._level = max(0.0, min(1.0, self._level * 0.40 + target * 0.60))
            else:
                self._level = max(0.0, min(1.0, self._level * 0.75 + target * 0.25))
            self.update()
        except Exception:
            return

    def paintEvent(self, event):
        try:
            p = QPainter(self)
            p.setRenderHint(QPainter.RenderHint.Antialiasing)
            p.fillRect(self.rect(), Qt.GlobalColor.transparent)
            w = max(1, self.width())
            h = max(1, self.height())
            pad = 6
            bar_w = int((w - pad * 2) * float(self._level))
            a = self._accent
            grad = QLinearGradient(pad, 0, w - pad, 0)
            _gradient_add_stop(grad, 0.0, QColor(a.red(), a.green(), a.blue(), 220))
            _gradient_add_stop(grad, 1.0, QColor(255, 80, 80, 220))
            p.fillRect(pad, int(h * 0.35), bar_w, int(h * 0.3), grad)
            # outline
            pen = QPen(QColor(255, 255, 255, 35))
            pen.setWidth(1)
            p.setPen(pen)
            p.drawRect(pad, int(h * 0.35), w - pad * 2, int(h * 0.3))
        except Exception:
            return


class AudioEQOverlayWidget(QWidget):
    """Full-size transparent overlay (over the camera view) with a movable/resizable HUD."""

    request_close = Signal()

    def __init__(self, parent=None, settings: Optional[AudioEQSettings] = None):
        super().__init__(parent)
        self.settings = settings or AudioEQSettings()
        self._dragging = False
        self._drag_start = None
        self._hud_start = None

        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self.setAttribute(Qt.WidgetAttribute.WA_OpaquePaintEvent, False)
        self.setAutoFillBackground(False)
        self.setMouseTracking(True)

        self.hud = AudioEQHUD(self, settings=self.settings)
        w, h = self.settings.size_px
        self.hud.resize(max(180, int(w)), max(110, int(h)))
        self.hud.close_clicked.connect(lambda: self.request_close.emit())

        self._reposition()
        self.show()

    def _reposition(self) -> None:
        try:
            nx, ny = self.settings.pos_norm
            nx = max(0.05, min(0.95, float(nx)))
            ny = max(0.05, min(0.95, float(ny)))
            w = self.width()
            h = self.height()
            hud_w = self.hud.width()
            hud_h = self.hud.height()
            x = int(nx * w - hud_w / 2)
            y = int(ny * h - hud_h / 2)
            x = max(4, min(w - hud_w - 4, x))
            y = max(4, min(h - hud_h - 4, y))
            self.hud.move(x, y)
        except Exception:
            pass

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._reposition()

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton and self.hud.geometry().contains(event.position().toPoint()):
            self._dragging = True
            self._drag_start = event.globalPosition().toPoint()
            self._hud_start = self.hud.pos()
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self._dragging and self._drag_start is not None and self._hud_start is not None:
            delta = event.globalPosition().toPoint() - self._drag_start
            new_pos = self._hud_start + delta
            x = max(4, min(self.width() - self.hud.width() - 4, new_pos.x()))
            y = max(4, min(self.height() - self.hud.height() - 4, new_pos.y()))
            self.hud.move(x, y)
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if self._dragging:
            self._dragging = False
            self._drag_start = None
            self._hud_start = None
            self._store_norm()
        super().mouseReleaseEvent(event)

    def _store_norm(self) -> None:
        try:
            c = self.hud.geometry().center()
            nx = c.x() / max(1, self.width())
            ny = c.y() / max(1, self.height())
            self.settings.pos_norm = (float(max(0.05, min(0.95, nx))), float(max(0.05, min(0.95, ny))))
            self.settings.size_px = (int(self.hud.width()), int(self.hud.height()))
        except Exception:
            pass


class AudioEQWindowBaseMixin:
    """Small helper to reuse the same HUD control wiring for floating windows."""

    def get_hud(self) -> AudioEQHUD:  # pragma: no cover
        raise NotImplementedError


class AudioEQWindow(QObject):
    """
    Lightweight floating window wrapper around AudioEQHUD.
    Implemented as a QObject that owns a BaseDesktopWidget instance to avoid import cycles at module import time.
    """

    closed = Signal()

    def __init__(self, settings: Optional[AudioEQSettings] = None, title: str = "Audio EQ", parent=None):
        super().__init__(parent)
        from desktop.widgets.base import BaseDesktopWidget  # local import to keep this module standalone

        self.settings = settings or AudioEQSettings()
        w, h = self.settings.size_px
        self.window = BaseDesktopWidget(title=title, width=max(240, int(w)), height=max(140, int(h)))
        self.window.setWindowFlag(Qt.WindowType.WindowStaysOnTopHint, True)
        self.window.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        try:
            self.window.central_widget.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        except Exception:
            pass
        try:
            self.window.central_widget.setStyleSheet("QWidget#Central { background-color: transparent; border: none; }")
        except Exception:
            pass
        try:
            self.window.title_bar.hide()
        except Exception:
            pass

        self.hud = AudioEQHUD(settings=self.settings)
        # Use HUD close button to close floating window
        self.hud.close_clicked.connect(self.window.close)
        self.window.set_content(self.hud)

        # Patch closeEvent to emit signal + persist size
        old_close = self.window.closeEvent

        def _close_event(ev):
            try:
                self.settings.size_px = (int(self.hud.width()), int(self.hud.height()))
            except Exception:
                pass
            try:
                self.closed.emit()
            except Exception:
                pass
            try:
                old_close(ev)
            except Exception:
                ev.accept()

        self.window.closeEvent = _close_event  # type: ignore[assignment]

    def show(self):
        self.window.show()
        try:
            self.window.raise_()
            self.window.activateWindow()
        except Exception:
            pass

    def close(self):
        try:
            self.window.close()
        except Exception:
            pass


