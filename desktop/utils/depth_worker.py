from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Optional, Literal

import cv2
import numpy as np
from PySide6.QtCore import QThread, Signal
from PySide6.QtGui import QImage

from core.model_library.runtime import DepthAnythingRuntimeConfig, get_model_runtime


DepthColorMap = Literal["turbo", "jet", "viridis", "plasma", "inferno", "magma", "bone", "ocean", "pointcloud"]
PointCloudColorMode = Literal["camera", "depth_turbo", "depth_viridis", "depth_inferno", "white"]


_CMAP_TO_CV2 = {
    "jet": cv2.COLORMAP_JET,
    "viridis": cv2.COLORMAP_VIRIDIS,
    "plasma": cv2.COLORMAP_PLASMA,
    "turbo": cv2.COLORMAP_TURBO,
    "inferno": cv2.COLORMAP_INFERNO,
    "magma": cv2.COLORMAP_MAGMA,
    "bone": cv2.COLORMAP_BONE,
    "ocean": cv2.COLORMAP_OCEAN,
}


@dataclass
class DepthOverlayConfig:
    enabled: bool = False
    fps_limit: int = 12
    opacity: float = 0.55
    colormap: DepthColorMap = "turbo"
    # Point cloud visualization (used when colormap == "pointcloud")
    pointcloud_step: int = 3  # pixel stride; smaller = denser
    pointcloud_color: PointCloudColorMode = "camera"
    pointcloud_zoom: float = 1.0  # matches React point cloud zoom behavior
    # Overlay compositing controls (applied in the PyQt renderer)
    overlay_scale: float = 1.0   # scales visualization relative to the camera image
    blackout_base: bool = False  # if True, don't draw camera frame (show viz only)
    camera_opacity: float = 1.0  # base camera layer opacity (lets camera be a transparent overlay under viz)
    # DepthAnythingV2
    model_size: Literal["vits", "vitb", "vitl"] = "vits"
    device: str = "auto"  # "cuda"|"cpu"|"auto"
    use_fp16: bool = True
    optimize: bool = True
    memory_fraction: Optional[float] = None


class DepthAnythingOverlayWorker(QThread):
    """
    Background depth inference worker.

    Accepts frames via `submit_frame` and emits a colorized QImage overlay.
    """

    depth_ready = Signal(object, object)  # QImage, stats(dict)
    status = Signal(str)
    error = Signal(str)

    def __init__(self, config: DepthOverlayConfig):
        super().__init__()
        self._cfg = config
        self._lock = threading.Lock()
        self._latest_frame: Optional[np.ndarray] = None
        self._latest_ts: float = 0.0
        self._stop = threading.Event()
        self._last_infer_ts: float = 0.0
        self._last_cfg_sig: Optional[str] = None

    def update_config(self, config: DepthOverlayConfig):
        with self._lock:
            self._cfg = config

    def set_fps_limit(self, fps: int) -> None:
        """Live-update the depth inference rate cap.

        Used by the auto-protection load shedder to throttle the depth
        worker under load without stopping it.  ``fps <= 0`` is clamped
        to 1; callers that want to fully stop should call ``stop()``.
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

    def submit_frame(self, frame: np.ndarray):
        # IMPORTANT: avoid per-frame .copy() allocations (can cause large RAM spikes and fragmentation
        # at high FPS / multiple cameras). Frames coming from OpenCV are fresh numpy arrays; we can
        # safely keep a reference and only process the latest one.
        fr = frame
        if fr is None:
            return
        with self._lock:
            self._latest_frame = fr
            self._latest_ts = time.time()

    def stop(self):
        self._stop.set()

    def run(self):
        last_wait_status_ts: float = 0.0
        while not self._stop.is_set():
            with self._lock:
                cfg = self._cfg
                frame = self._latest_frame
                frame_ts = self._latest_ts
                self._latest_frame = None

            if not cfg.enabled:
                time.sleep(0.05)
                continue

            if frame is None:
                # Provide feedback when enabled but no frames are arriving.
                now = time.time()
                if now - last_wait_status_ts > 1.0:
                    last_wait_status_ts = now
                    try:
                        self.status.emit("Depth: waiting for frames…")
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
                cfg_sig = (
                    f"{cfg.model_size}|{cfg.device}|{int(cfg.use_fp16)}|{int(cfg.optimize)}|"
                    f"{cfg.colormap}|{int(cfg.pointcloud_step)}|{getattr(cfg, 'pointcloud_color', 'camera')}|{float(getattr(cfg, 'pointcloud_zoom', 1.0)):.2f}"
                )
                if self._last_cfg_sig != cfg_sig:
                    self._last_cfg_sig = cfg_sig
                    self.status.emit(
                        f"Depth: switching → {cfg.model_size} on {cfg.device} (fp16={cfg.use_fp16}) | viz={cfg.colormap}"
                    )

                runtime_cfg = DepthAnythingRuntimeConfig(
                    model_size=cfg.model_size,
                    device=cfg.device,
                    use_fp16=cfg.use_fp16,
                    optimize=cfg.optimize,
                    memory_fraction=cfg.memory_fraction,
                )
                handle = get_model_runtime().get_depth_anything(runtime_cfg)
                # NOTE: Do NOT hold handle.lock while calling handle.get()/estimate_depth().
                # The runtime handle is internally synchronized; holding it here can deadlock.
                need_load = getattr(handle, "_runner", None) is None
                if need_load:
                    self.status.emit(f"Depth: loading model ({cfg.model_size} on {cfg.device})…")
                    # Best-effort UX: show GPU warmup phase (even though warmup occurs inside model init).
                    try:
                        if str(cfg.device).startswith("cuda") or str(cfg.device) == "auto":
                            self.status.emit("Depth: warming up GPU…")
                    except Exception:
                        pass
                runner = handle.get()
                if need_load:
                    try:
                        actual_device = getattr(runner, "device", None) or cfg.device
                        self.status.emit(f"Depth: ready ({cfg.model_size} on {actual_device})")
                        if str(actual_device).startswith("cuda"):
                            self.status.emit("Depth: GPU ready")
                        elif str(actual_device) == "cpu":
                            self.status.emit(
                                "Depth: running on CPU (slow) – "
                                "a CUDA-capable GPU is strongly recommended"
                            )
                    except Exception:
                        self.status.emit("Depth: ready")

                depth_u8 = runner.estimate_depth(frame)
                stats = runner.get_stats()

                colored_bgr = None
                if cfg.colormap == "pointcloud":
                    colored_bgr = _render_pointcloud_first_person(
                        frame,
                        depth_u8,
                        step=int(cfg.pointcloud_step),
                        color_mode=str(getattr(cfg, "pointcloud_color", "camera")),
                        zoom=float(getattr(cfg, "pointcloud_zoom", 1.0) or 1.0),
                    )
                else:
                    cmap = _CMAP_TO_CV2.get(cfg.colormap, cv2.COLORMAP_TURBO)
                    colored_bgr = cv2.applyColorMap(depth_u8, cmap)
                h, w = colored_bgr.shape[:2]
                # Pointcloud returns BGRA with transparency so camera can show through "empty" pixels.
                if len(colored_bgr.shape) == 3 and colored_bgr.shape[2] == 4:
                    qimg = QImage(colored_bgr.data, w, h, colored_bgr.strides[0], QImage.Format.Format_ARGB32).copy()
                else:
                    qimg = QImage(colored_bgr.data, w, h, colored_bgr.strides[0], QImage.Format.Format_BGR888).copy()
                out_stats = {
                    **(stats or {}),
                    "timestamp": float(frame_ts or time.time()),
                    "algorithm": "depth-anything-v2",
                }
                self.depth_ready.emit(qimg, out_stats)
                self._last_infer_ts = time.time()
            except Exception as e:
                self.error.emit(f"DepthAnythingV2 error: {e}")
                time.sleep(0.25)


def _render_pointcloud_first_person(
    frame_bgr: np.ndarray,
    depth_u8: np.ndarray,
    *,
    step: int = 6,
    color_mode: str = "camera",
    zoom: float = 1.0,
) -> np.ndarray:
    """
    Render a simple first-person point cloud image from a depth map.

    This is not metric-accurate (DepthAnythingV2 output is relative), but it provides an
    intuitive 3D visualization that users can tune via density (step size).
    """
    # This is intended to match the "3D Point Cloud Visualization" path in the React frontend:
    # - sample depth map (u8)
    # - build simple camera-relative 3D points from pixel coords + depth
    # - perspective project to screen
    # - render as colored splats
    h, w = depth_u8.shape[:2]
    step = max(1, int(step))
    try:
        zoom = float(zoom)
    except Exception:
        zoom = 1.0
    zoom = max(0.25, min(5.0, zoom))

    # Sample grid
    ys = np.arange(0, h, step, dtype=np.int32)
    xs = np.arange(0, w, step, dtype=np.int32)
    if ys.size == 0 or xs.size == 0:
        return np.zeros((h, w, 3), dtype=np.uint8)
    grid_y, grid_x = np.meshgrid(ys, xs, indexing="ij")

    # Sample depth values
    depth = depth_u8[grid_y, grid_x].astype(np.float32)

    # Skip very dark/far points (mirrors React `if (depth < 15) continue`)
    m_depth = depth >= 15.0
    if not np.any(m_depth):
        return np.zeros((h, w, 3), dtype=np.uint8)

    gx = grid_x[m_depth].astype(np.float32)
    gy = grid_y[m_depth].astype(np.float32)
    depth = depth[m_depth]

    # React math:
    # baseScale = pointCloudZoom * 1.2
    # perspective = 500 / pointCloudZoom
    # x3d = (x - w/2) * baseScale
    # y3d = (y - h/2) * baseScale
    # z3d = (255 - depth) * baseScale * 1.5
    # point.z = z3d + 100
    base_scale = float(zoom) * 1.2
    perspective = 500.0 / max(0.01, float(zoom))
    cx = (w - 1) * 0.5
    cy = (h - 1) * 0.5
    x3d = (gx - cx) * base_scale
    y3d = (gy - cy) * base_scale
    z3d = (255.0 - depth) * base_scale * 1.5 + 100.0

    # Perspective projection
    scale = perspective / (perspective + z3d)
    sx = (cx + x3d * scale).astype(np.int32)
    sy = (cy + y3d * scale).astype(np.int32)

    # Build output as BGRA.
    # Alpha=0 for empty pixels (so the camera frame remains visible),
    # Alpha=255 where points are drawn.
    out = np.zeros((h, w, 4), dtype=np.uint8)

    # Colors per point (match React defaults: use source image colors by default)
    if color_mode == "camera":
        cols = frame_bgr[gy.astype(np.int32), gx.astype(np.int32)]
    elif color_mode == "white":
        cols = np.full((gx.shape[0], 3), 255, dtype=np.uint8)
    elif color_mode.startswith("depth_"):
        cmap_name = color_mode.split("_", 1)[1].strip().lower()
        cmap = _CMAP_TO_CV2.get(cmap_name, cv2.COLORMAP_TURBO)
        depth_colored = cv2.applyColorMap(depth_u8, cmap)
        cols = depth_colored[gy.astype(np.int32), gx.astype(np.int32)]
    else:
        cols = frame_bgr[gy.astype(np.int32), gx.astype(np.int32)]

    # Mask valid
    m = (sx >= 0) & (sx < w) & (sy >= 0) & (sy < h)
    sxv = sx[m]
    syv = sy[m]
    colv = cols[m]

    # Plot points. Use a tiny splat footprint for visibility (especially at low density).
    out[syv, sxv, :3] = colv
    out[syv, sxv, 3] = 255
    # 2x2 splat (clipped)
    sx1 = np.clip(sxv + 1, 0, w - 1)
    sy1 = np.clip(syv + 1, 0, h - 1)
    out[syv, sx1, :3] = colv
    out[syv, sx1, 3] = 255
    out[sy1, sxv, :3] = colv
    out[sy1, sxv, 3] = 255
    out[sy1, sx1, :3] = colv
    out[sy1, sx1, 3] = 255

    return out


