import asyncio
import json
import logging
import os
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    import numpy as np  # type: ignore
except Exception:  # pragma: no cover
    np = None  # type: ignore

try:
    import aiohttp
except Exception:  # pragma: no cover
    aiohttp = None  # type: ignore

try:
    from aiortc import RTCPeerConnection, RTCSessionDescription, RTCIceServer, RTCConfiguration
except Exception:  # pragma: no cover
    RTCPeerConnection = None  # type: ignore
    RTCSessionDescription = None  # type: ignore
    RTCIceServer = None  # type: ignore
    RTCConfiguration = None  # type: ignore

try:
    from av.audio.resampler import AudioResampler
except Exception:  # pragma: no cover
    AudioResampler = None  # type: ignore

from core.audio_embedding import compute_embedding_from_mono, pcm_s16le_to_mono_float
from core.audio_profiles import AudioProfileStore
from core.events import build_event_bundle

logger = logging.getLogger(__name__)


def _resolve_camera_stream_path(camera_id: str) -> str:
    """
    Resolve mediamtx_path from cameras.json (preferred when CameraManager is disabled).
    Falls back to camera_id.
    """
    try:
        with open("cameras.json", "r") as f:
            cams = json.load(f)
        for c in cams or []:
            if str(c.get("id")) == str(camera_id):
                return str(c.get("mediamtx_path") or c.get("id") or camera_id)
    except Exception:
        pass
    return str(camera_id)


def _build_whep_url(mediamtx_webrtc_url: str, camera_id: str) -> str:
    base = str(mediamtx_webrtc_url or "").rstrip("/")
    if not base:
        base = "http://localhost:8889"
    stream_path = _resolve_camera_stream_path(camera_id)
    return f"{base}/{stream_path}/whep"


@dataclass
class AudioMonitorConfig:
    amplitude_threshold: float = 0.06  # RMS threshold on mono float [-1..1]
    silence_timeout_s: float = 0.8     # end event after this much silence
    min_event_s: float = 0.4
    max_event_s: float = 12.0
    pre_roll_s: float = 1.5
    post_roll_s: float = 0.6
    match_min_similarity: float = 0.75
    match_top_k: int = 3


class AudioCameraMonitor:
    def __init__(
        self,
        *,
        camera_id: str,
        mediamtx_webrtc_url: str,
        profile_store: AudioProfileStore,
        db_manager: Any = None,
        socketio: Any = None,
        python_script_manager: Any = None,
        automation_engine: Any = None,
        config: Optional[AudioMonitorConfig] = None,
    ):
        self.camera_id = str(camera_id)
        self.mediamtx_webrtc_url = str(mediamtx_webrtc_url or "")
        self.profile_store = profile_store
        self.db_manager = db_manager
        self.socketio = socketio
        self.python_script_manager = python_script_manager
        self.automation_engine = automation_engine
        self.cfg = config or AudioMonitorConfig()

        self._stop_evt = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None

        # event state
        self._in_event = False
        self._event_started = 0.0
        self._last_loud = 0.0
        self._event_buf: List[bytes] = []

        # ring buffer for pre-roll
        self._ring: List[bytes] = []
        self._ring_bytes = 0
        self._ring_max_bytes = int(48000 * 2 * 2 * 15)  # ~15s

        self._last_error: Optional[str] = None

    @property
    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive() and not self._stop_evt.is_set()

    @property
    def last_error(self) -> Optional[str]:
        return self._last_error

    def start(self) -> None:
        if self.is_running:
            return
        self._stop_evt.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_evt.set()
        try:
            if self._loop and self._loop.is_running():
                self._loop.call_soon_threadsafe(lambda: None)
        except Exception:
            pass

    def _run(self) -> None:
        loop = asyncio.new_event_loop()
        self._loop = loop
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(self._connect_and_consume())
        except Exception as e:
            self._last_error = str(e)
            logger.error("Audio monitor error for %s: %s", self.camera_id, e)
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

    async def _connect_and_consume(self) -> None:
        if RTCPeerConnection is None or aiohttp is None or AudioResampler is None:
            raise RuntimeError("Missing dependencies (aiortc/aiohttp/av)")
        if RTCConfiguration is None or RTCIceServer is None or RTCSessionDescription is None:
            raise RuntimeError("Missing aiortc SDP dependencies")

        whep_url = _build_whep_url(self.mediamtx_webrtc_url, self.camera_id)

        cfg = RTCConfiguration(iceServers=[RTCIceServer(urls=["stun:stun.l.google.com:19302"])])
        pc = RTCPeerConnection(configuration=cfg)
        pc.addTransceiver("audio", direction="recvonly")

        track_holder: Dict[str, Any] = {"track": None}

        @pc.on("track")
        def _on_track(track):
            if track.kind == "audio":
                track_holder["track"] = track

        offer = await pc.createOffer()
        await pc.setLocalDescription(offer)

        # ICE gather for non-trickle WHEP
        try:
            start = time.time()
            while pc.iceGatheringState != "complete" and (time.time() - start) < 4.0:
                await asyncio.sleep(0.05)
        except Exception:
            pass

        local_sdp = (pc.localDescription.sdp if pc.localDescription else "") or ""
        if "a=ice-ufrag" not in local_sdp:
            # retry a bit
            try:
                start = time.time()
                while "a=ice-ufrag" not in local_sdp and (time.time() - start) < 2.0:
                    await asyncio.sleep(0.05)
                    local_sdp = (pc.localDescription.sdp if pc.localDescription else "") or ""
            except Exception:
                pass

        if not local_sdp:
            await pc.close()
            raise RuntimeError("Failed to create local SDP offer")

        answer_sdp = ""
        async with aiohttp.ClientSession() as s:
            async with s.post(
                whep_url,
                data=local_sdp,
                headers={"Content-Type": "application/sdp", "Accept": "application/sdp"},
            ) as r:
                if not r.ok:
                    txt = await r.text()
                    await pc.close()
                    raise RuntimeError(f"WHEP POST failed ({r.status}): {txt}")
                answer_sdp = await r.text()

        await pc.setRemoteDescription(RTCSessionDescription(sdp=answer_sdp, type="answer"))

        # Wait for audio track
        for _ in range(60):
            if self._stop_evt.is_set():
                await pc.close()
                return
            if track_holder["track"] is not None:
                break
            await asyncio.sleep(0.05)

        track = track_holder["track"]
        if track is None:
            await pc.close()
            raise RuntimeError("No audio track received (camera may not publish audio)")

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
                        arr = out.to_ndarray()
                        pcm = arr.astype(np.int16).tobytes() if np is not None else bytes(arr)
                    self._handle_pcm(pcm)
        finally:
            try:
                await pc.close()
            except Exception:
                pass

    def _handle_pcm(self, pcm: bytes) -> None:
        if not pcm:
            return

        # ring buffer
        self._ring.append(pcm)
        self._ring_bytes += len(pcm)
        while self._ring and self._ring_bytes > self._ring_max_bytes:
            d = self._ring.pop(0)
            self._ring_bytes -= len(d)

        now = time.time()

        # compute RMS quickly
        rms = 0.0
        try:
            if np is not None:
                x = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0
                x = x.reshape((-1, 2)).mean(axis=1)  # mono
                rms = float(np.sqrt(np.mean(x * x) + 1e-12))
            else:
                # coarse fallback
                import audioop

                rms = float(audioop.rms(pcm, 2) / 32768.0)
        except Exception:
            rms = 0.0

        thr = float(max(0.001, min(0.95, self.cfg.amplitude_threshold)))
        loud = rms >= thr

        if not self._in_event:
            if loud:
                self._begin_event(now)
                self._event_buf.append(pcm)
                self._last_loud = now
            return

        # in event
        self._event_buf.append(pcm)
        if loud:
            self._last_loud = now

        dur = now - self._event_started
        # stop if too long
        if dur >= float(self.cfg.max_event_s):
            self._finalize_event(now, rms)
            return

        # stop after silence
        if (now - self._last_loud) >= float(self.cfg.silence_timeout_s):
            # include post-roll, then finalize
            if (now - self._last_loud) >= float(self.cfg.post_roll_s):
                self._finalize_event(now, rms)

    def _begin_event(self, now: float) -> None:
        self._in_event = True
        self._event_started = now
        self._last_loud = now
        self._event_buf = []

        # pre-roll from ring
        try:
            bytes_per_sec = 48000 * 2 * 2
            need = int(float(self.cfg.pre_roll_s) * bytes_per_sec)
            collected: List[bytes] = []
            total = 0
            for chunk in reversed(self._ring):
                collected.append(chunk)
                total += len(chunk)
                if total >= need:
                    break
            self._event_buf.extend(reversed(collected))
        except Exception:
            pass

    def _finalize_event(self, now: float, rms_last: float) -> None:
        try:
            self._in_event = False
            dur = now - self._event_started
            if dur < float(self.cfg.min_event_s):
                self._event_buf = []
                return

            clips_dir = Path("data") / "audio_clips"
            clips_dir.mkdir(parents=True, exist_ok=True)
            clip_id = f"evt_{int(now*1000)}_{self.camera_id[:8]}.wav"
            out_path = clips_dir / clip_id
            pcm = b"".join(self._event_buf)
            self._event_buf = []

            import wave

            with wave.open(str(out_path), "wb") as wf:
                wf.setnchannels(2)
                wf.setsampwidth(2)
                wf.setframerate(48000)
                wf.writeframes(pcm)

            matches: List[Dict[str, Any]] = []
            emb: Optional[List[float]] = None
            try:
                if np is not None:
                    mono = pcm_s16le_to_mono_float(pcm, 2)
                    emb = compute_embedding_from_mono(mono, sample_rate=48000)
                    matches = self.profile_store.match_embedding(
                        emb,
                        top_k=int(self.cfg.match_top_k),
                        min_similarity=float(self.cfg.match_min_similarity),
                    )
            except Exception as e:
                logger.debug("Audio embedding/match failed: %s", e)

            payload = {
                "id": f"audio_evt_{uuid.uuid4().hex[:10]}",
                "camera_id": self.camera_id,
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "type": "audio_event",
                "clip_id": clip_id,
                "duration_s": float(dur),
                "rms_last": float(rms_last),
                "matches": matches,
            }

            # emit realtime
            try:
                if self.socketio is not None:
                    self.socketio.emit("audio_event", payload, room=f"camera_{self.camera_id}")
                    self.socketio.emit("audio_event", payload)
            except Exception:
                pass

            # trigger python automations
            try:
                if self.python_script_manager is not None and hasattr(self.python_script_manager, "handle_event"):
                    self.python_script_manager.handle_event("audio_event", self.camera_id, payload)
                    if matches:
                        self.python_script_manager.handle_event("audio_match", self.camera_id, payload)
            except Exception:
                pass

            # trigger server-side automation engine (rules/actions)
            try:
                if self.automation_engine is not None and hasattr(self.automation_engine, "submit"):
                    self.automation_engine.submit("audio_event", self.camera_id, payload)
                    if matches:
                        self.automation_engine.submit("audio_match", self.camera_id, payload)
            except Exception:
                pass

            # store event bundle (reuses unified events store)
            try:
                if self.db_manager is not None and hasattr(self.db_manager, "store_event_bundle"):
                    import json as pyjson

                    bundle = build_event_bundle(
                        bundle_id=str(uuid.uuid4()),
                        camera_id=self.camera_id,
                        kind="audio_event",
                        detections=[],
                        tracks=[],
                        overlays=None,
                        snapshot_base64=None,
                        metadata=payload,
                    )
                    self.db_manager.store_event_bundle(
                        bundle_id=bundle.id,
                        camera_id=bundle.camera_id,
                        kind=bundle.kind,
                        created_at=bundle.created_at,
                        bundle_json=pyjson.dumps(bundle.to_dict()),
                    )
            except Exception as e:
                logger.debug("Store audio bundle failed: %s", e)
        except Exception:
            self._in_event = False
            self._event_buf = []


class AudioMonitorManager:
    def __init__(
        self,
        *,
        mediamtx_webrtc_url: str,
        db_manager: Any = None,
        socketio: Any = None,
        python_script_manager: Any = None,
        automation_engine: Any = None,
        profile_store: Optional[AudioProfileStore] = None,
    ):
        self.mediamtx_webrtc_url = str(mediamtx_webrtc_url or "")
        self.db_manager = db_manager
        self.socketio = socketio
        self.python_script_manager = python_script_manager
        self.automation_engine = automation_engine
        self.profiles = profile_store or AudioProfileStore()
        self._monitors: Dict[str, AudioCameraMonitor] = {}
        self._lock = threading.RLock()

    def start_camera(self, camera_id: str, config: Optional[AudioMonitorConfig] = None) -> bool:
        with self._lock:
            cam_id = str(camera_id)
            if cam_id in self._monitors and self._monitors[cam_id].is_running:
                return True
            mon = AudioCameraMonitor(
                camera_id=cam_id,
                mediamtx_webrtc_url=self.mediamtx_webrtc_url,
                profile_store=self.profiles,
                db_manager=self.db_manager,
                socketio=self.socketio,
                python_script_manager=self.python_script_manager,
                automation_engine=self.automation_engine,
                config=config,
            )
            self._monitors[cam_id] = mon
            mon.start()
            return True

    def stop_camera(self, camera_id: str) -> bool:
        with self._lock:
            cam_id = str(camera_id)
            mon = self._monitors.get(cam_id)
            if not mon:
                return False
            try:
                mon.stop()
            except Exception:
                pass
            return True

    def status(self) -> Dict[str, Any]:
        with self._lock:
            out: Dict[str, Any] = {}
            for cam_id, mon in self._monitors.items():
                out[cam_id] = {
                    "camera_id": cam_id,
                    "running": bool(mon.is_running),
                    "last_error": mon.last_error,
                }
            return out


