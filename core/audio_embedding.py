import math
import os
import subprocess
import tempfile
import wave
from dataclasses import dataclass
from typing import List, Optional, Tuple

try:
    import numpy as np  # type: ignore
except Exception:  # pragma: no cover
    np = None  # type: ignore


@dataclass
class AudioClipPCM:
    sample_rate: int
    channels: int
    pcm_s16le: bytes


def _read_wav_pcm(path: str) -> AudioClipPCM:
    with wave.open(path, "rb") as wf:
        sr = int(wf.getframerate())
        ch = int(wf.getnchannels())
        sampwidth = int(wf.getsampwidth())
        if sampwidth != 2:
            raise ValueError(f"Unsupported WAV sample width: {sampwidth} bytes (need 16-bit)")
        frames = wf.readframes(wf.getnframes())
        return AudioClipPCM(sample_rate=sr, channels=ch, pcm_s16le=frames)


def _ffmpeg_to_wav_s16le(path: str) -> str:
    """
    Convert an arbitrary audio file to wav (s16le) using ffmpeg.
    Returns a temp wav path. Caller must delete it.
    """
    if not shutil_which("ffmpeg"):
        raise RuntimeError("ffmpeg not found (required to read non-WAV clips)")
    fd, out = tempfile.mkstemp(prefix="knoxnet_audio_", suffix=".wav")
    os.close(fd)
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        path,
        "-acodec",
        "pcm_s16le",
        "-ar",
        "48000",
        "-ac",
        "2",
        out,
    ]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if proc.returncode != 0:
        try:
            os.remove(out)
        except Exception:
            pass
        raise RuntimeError(f"ffmpeg conversion failed: {proc.stderr[-500:]}")
    return out


def shutil_which(cmd: str) -> Optional[str]:
    try:
        import shutil

        return shutil.which(cmd)
    except Exception:
        return None


def load_audio_clip_pcm(path: str) -> AudioClipPCM:
    """
    Load a clip into PCM s16le. WAV is supported directly. Other formats use ffmpeg if available.
    """
    p = str(path)
    if p.lower().endswith(".wav"):
        return _read_wav_pcm(p)
    tmp = None
    try:
        tmp = _ffmpeg_to_wav_s16le(p)
        return _read_wav_pcm(tmp)
    finally:
        if tmp:
            try:
                os.remove(tmp)
            except Exception:
                pass


def pcm_s16le_to_mono_float(pcm_s16le: bytes, channels: int) -> "np.ndarray":
    if np is None:
        raise RuntimeError("numpy is required for embeddings")
    a = np.frombuffer(pcm_s16le, dtype=np.int16).astype(np.float32) / 32768.0
    if channels <= 1:
        return a
    try:
        a = a.reshape((-1, channels)).mean(axis=1)
    except Exception:
        # fallback: assume stereo
        a = a.reshape((-1, 2)).mean(axis=1)
    return a


def _l2_normalize(v: "np.ndarray") -> "np.ndarray":
    n = float(np.linalg.norm(v) + 1e-12)
    return (v / n).astype(np.float32)


def compute_embedding_from_mono(
    x: "np.ndarray",
    *,
    sample_rate: int,
    n_fft: int = 2048,
    hop: int = 512,
    bins: int = 64,
) -> List[float]:
    """
    Lightweight, production-friendly embedding:
    - log power spectrum sampled into `bins` log-spaced bands
    - aggregate over time with mean + std
    - L2 normalize (cosine similarity friendly)
    """
    if np is None:
        raise RuntimeError("numpy is required for embeddings")
    sr = int(sample_rate)
    if sr <= 0:
        sr = 48000

    n_fft = int(max(256, min(8192, n_fft)))
    hop = int(max(64, min(n_fft, hop)))
    bins = int(max(16, min(128, bins)))

    if x.size < n_fft:
        # pad small clips
        pad = n_fft - int(x.size)
        x = np.pad(x, (0, pad), mode="constant")

    window = np.hanning(n_fft).astype(np.float32)

    # build log-spaced band indices
    f_min = 60.0
    f_max = min(12000.0, (sr / 2.0) - 50.0)
    freqs = [f_min * ((f_max / f_min) ** (i / max(1, bins - 1))) for i in range(bins)]
    k_idx = [max(1, min((n_fft // 2) - 2, int(0.5 + (n_fft * f / sr)))) for f in freqs]

    feats: List["np.ndarray"] = []
    # frame loop (short clips; loop is fine)
    for start in range(0, int(x.size) - n_fft + 1, hop):
        frame = x[start : start + n_fft] * window
        spec = np.fft.rfft(frame)
        mag = (spec.real * spec.real + spec.imag * spec.imag).astype(np.float32)
        # sample bands with small neighborhood
        band = np.zeros((bins,), dtype=np.float32)
        for bi, k in enumerate(k_idx):
            j0 = max(1, k - 1)
            j1 = min(mag.size - 1, k + 1)
            band[bi] = float(np.mean(mag[j0 : j1 + 1]))
        feats.append(band)

    if not feats:
        vec = np.zeros((bins * 2,), dtype=np.float32)
        return _l2_normalize(vec).tolist()

    M = np.stack(feats, axis=0)
    M = np.log10(1.0 + M)
    mean = np.mean(M, axis=0).astype(np.float32)
    std = np.std(M, axis=0).astype(np.float32)
    vec = np.concatenate([mean, std], axis=0)
    return _l2_normalize(vec).tolist()


def cosine_similarity(a: List[float], b: List[float]) -> float:
    if np is None:
        # fallback pure python
        dot = 0.0
        na = 0.0
        nb = 0.0
        n = min(len(a), len(b))
        for i in range(n):
            dot += float(a[i]) * float(b[i])
            na += float(a[i]) * float(a[i])
            nb += float(b[i]) * float(b[i])
        return float(dot / (math.sqrt(na) * math.sqrt(nb) + 1e-12))
    va = np.array(a, dtype=np.float32)
    vb = np.array(b, dtype=np.float32)
    return float(np.dot(va, vb) / (float(np.linalg.norm(va)) * float(np.linalg.norm(vb)) + 1e-12))


