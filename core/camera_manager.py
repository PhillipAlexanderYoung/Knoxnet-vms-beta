import asyncio
import json
import logging
import threading
import time
import cv2
import numpy as np
from typing import Dict, List, Optional, Any, Callable, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from enum import Enum
import os
import sqlite3
from contextlib import contextmanager
import aiohttp
import re
import requests
from urllib.parse import urlparse, urlunparse

from .mediamtx_client import MediaMTXWebRTCClient, StreamConfig, ConnectionStats
from .entitlements import get_camera_limit

logger = logging.getLogger(__name__)


class CameraStatus(Enum):
    """Camera connection status"""
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    ERROR = "error"
    MAINTENANCE = "maintenance"


class StreamQuality(Enum):
    """Stream quality presets"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    ULTRA = "ultra"


@dataclass
class CameraConfig:
    """Configuration for a camera"""
    id: str
    name: str
    rtsp_url: str
    location: str = ""
    enabled: bool = True
    ai_analysis: bool = True
    recording: bool = False
    ptz_enabled: bool = False
    ptz_url: Optional[str] = None
    ptz_protocol: Optional[str] = None  # auto-resolved by PTZManager probe (tapo|onvif|generic)
    username: Optional[str] = None
    password: Optional[str] = None
    port: int = 554
    stream_path: Optional[str] = None
    stream_priority: str = "main"
    stream_quality: StreamQuality = StreamQuality.MEDIUM
    webrtc_enabled: bool = True
    substream_path: Optional[str] = None
    substream_rtsp_url: Optional[str] = None
    # Substream capability detection.  None = never tested, True = verified
    # working, False = verified broken (auto-switch will not try sub for
    # this camera until manually re-tested).  Width/height captured from
    # the probe frame so the auto-switch policy can pick wisely.
    substream_capable: Optional[bool] = None
    substream_width: Optional[int] = None
    substream_height: Optional[int] = None
    substream_last_check: float = 0.0
    backup_rtsp_url: Optional[str] = None
    ip_address: Optional[str] = None
    mediamtx_path: Optional[str] = None
    mediamtx_sub_path: Optional[str] = None
    custom_rtsp: bool = False
    manufacturer: Optional[str] = None
    protocol: Optional[str] = None
    status: str = "offline"
    motion_detection: bool = True
    audio_enabled: bool = False
    night_vision: bool = False
    privacy_mask: List[Dict[str, int]] = None
    extra_config: Dict[str, Any] = None
    created_at: datetime = None
    updated_at: datetime = None

    def __post_init__(self):
        if self.privacy_mask is None:
            self.privacy_mask = []
        if self.extra_config is None:
            self.extra_config = {}
        if self.created_at is None:
            self.created_at = datetime.now()
        if self.updated_at is None:
            self.updated_at = datetime.now()


@dataclass
class CameraHealth:
    """Camera health and performance metrics"""
    camera_id: str
    status: CameraStatus
    last_frame_time: Optional[datetime] = None
    frame_rate: float = 0.0
    connection_uptime: Optional[timedelta] = None
    total_frames: int = 0
    dropped_frames: int = 0
    error_count: int = 0
    last_error: Optional[str] = None
    bandwidth_usage: float = 0.0  # Mbps
    latency: float = 0.0  # seconds
    webrtc_connected: bool = False
    stream_quality_actual: Optional[StreamQuality] = None


@dataclass
class PTZCapabilities:
    """PTZ (Pan-Tilt-Zoom) capabilities"""
    pan_range: Tuple[float, float] = (-180.0, 180.0)
    tilt_range: Tuple[float, float] = (-90.0, 90.0)
    zoom_range: Tuple[float, float] = (1.0, 10.0)
    presets: List[Dict[str, Any]] = None
    auto_tracking: bool = False
    tour_support: bool = False

    def __post_init__(self):
        if self.presets is None:
            self.presets = []


class CameraManager:
    """
    Enhanced camera management with MediaMTX WebRTC integration
    """

    def __init__(self,
                 db_path: str = "data/cameras.db",
                 mediamtx_host: str = "localhost",
                 mediamtx_webrtc_port: int = 8889,
                 mediamtx_api_port: int = 9997,
                 auto_connect_on_start: bool = False,
                 enable_webrtc_receiver: bool = False,
                 cleanup_mediamtx_paths_on_disconnect: bool = False,
                 idle_disconnect_seconds: float = 15.0):
        """
        Initialize Camera Manager

        Args:
            db_path: SQLite database path for camera configurations
            mediamtx_host: MediaMTX server host
            mediamtx_webrtc_port: MediaMTX WebRTC port
            mediamtx_api_port: MediaMTX API port
        """
        self.db_path = db_path
        self.cameras: Dict[str, CameraConfig] = {}
        self.camera_health: Dict[str, CameraHealth] = {}
        self.ptz_capabilities: Dict[str, PTZCapabilities] = {}

        # MediaMTX integration
        self.mediamtx_client = MediaMTXWebRTCClient(
            mediamtx_host=mediamtx_host,
            mediamtx_webrtc_port=mediamtx_webrtc_port,
            mediamtx_api_port=mediamtx_api_port
        )

        # Desktop-light runtime flags
        self.auto_connect_on_start = bool(auto_connect_on_start)
        self.enable_webrtc_receiver = bool(enable_webrtc_receiver)
        self.cleanup_mediamtx_paths_on_disconnect = bool(cleanup_mediamtx_paths_on_disconnect)
        self.idle_disconnect_seconds = float(idle_disconnect_seconds)

        # Viewer reference counting (camera widgets / consumers).
        # When refcount hits 0, we schedule a disconnect after idle_disconnect_seconds.
        self._viewer_counts: Dict[str, int] = {}
        self._idle_disconnect_handles: Dict[str, Any] = {}

        # Stream management
        self.active_streams: Dict[str, cv2.VideoCapture] = {}
        self.stream_threads: Dict[str, threading.Thread] = {}
        self.webrtc_streams: Dict[str, bool] = {}
        self._stream_stop_events: Dict[str, threading.Event] = {}

        # Event callbacks
        self.on_camera_connected: Optional[Callable] = None
        self.on_camera_disconnected: Optional[Callable] = None
        self.on_frame_received: Optional[Callable] = None
        self.on_motion_detected: Optional[Callable] = None
        self.on_camera_error: Optional[Callable] = None

        # Background monitoring
        self._running = False
        self._monitor_thread: Optional[threading.Thread] = None
        self._health_thread: Optional[threading.Thread] = None
        # Guards to prevent runaway concurrent connect/reconnect attempts (can leak resources / OOM).
        self._connect_guard = threading.Lock()
        self._connect_inflight: set[str] = set()
        self._reconnect_inflight: set[str] = set()
        self._last_connect_attempt: Dict[str, float] = {}

        # Best-effort JSON mirror. In some setups the backend runs in Docker and
        # bind-mounts `cameras.json` as root-owned; avoid repeated permission errors.
        self._json_sync_disabled = False
        self._json_read_disabled = False
        self._json_path_candidates = ["data/cameras.json", "cameras.json"]

        # Per-camera lock around substream capability probing to avoid
        # multiple simultaneous probes of the same URL (which would each
        # consume an RTSP session).
        self._substream_probe_locks: Dict[str, asyncio.Lock] = {}

        # Quality settings mapping
        self.quality_settings = {
            StreamQuality.LOW: {"width": 640, "height": 480, "fps": 15, "bitrate": "500k"},
            StreamQuality.MEDIUM: {"width": 1280, "height": 720, "fps": 25, "bitrate": "2M"},
            StreamQuality.HIGH: {"width": 1920, "height": 1080, "fps": 30, "bitrate": "4M"},
            StreamQuality.ULTRA: {"width": 3840, "height": 2160, "fps": 30, "bitrate": "8M"}
        }

        # Initialize database and MediaMTX client
        self._init_database()
        self._load_cameras_from_db()

    async def start(self):
        """Start the camera manager and all services"""
        if self._running:
            return

        self._running = True
        self._loop = asyncio.get_running_loop()
        logger.info("Starting Camera Manager with MediaMTX integration")

        # Start MediaMTX client
        await self.mediamtx_client.start()

        # Set up MediaMTX callbacks
        self.mediamtx_client.on_stream_connected = self._on_webrtc_connected
        self.mediamtx_client.on_stream_disconnected = self._on_webrtc_disconnected
        self.mediamtx_client.on_stream_error = self._on_webrtc_error

        # Recover camera configurations
        logger.info("Recovering camera configurations...")
        recovered_count = await self.recover_camera_configurations()
        logger.info(f"Recovered {recovered_count} camera configurations")

        # Start background monitoring
        self._monitor_thread = threading.Thread(target=self._monitor_cameras, daemon=True)
        self._monitor_thread.start()

        self._health_thread = threading.Thread(target=self._update_health_metrics, daemon=True)
        self._health_thread.start()

        # Auto-connect cameras only when explicitly configured (desktop-light defaults to off).
        if self.auto_connect_on_start:
            logger.info("Auto-connecting enabled cameras (auto_connect_on_start=true)...")
            await self.auto_reconnect_cameras()
        else:
            logger.info("Skipping auto-connect on start (desktop-light mode)")

        logger.info("Camera Manager started successfully")

    async def stop(self):
        """Stop the camera manager and cleanup resources"""
        if not self._running:
            return

        logger.info("Stopping Camera Manager")
        self._running = False

        # Disconnect all cameras
        for camera_id in list(self.cameras.keys()):
            await self.disconnect_camera(camera_id)

        # Stop MediaMTX client
        await self.mediamtx_client.stop()

        # Wait for threads to finish
        if self._monitor_thread and self._monitor_thread.is_alive():
            self._monitor_thread.join(timeout=5)
        if self._health_thread and self._health_thread.is_alive():
            self._health_thread.join(timeout=5)

        logger.info("Camera Manager stopped")

    # Camera Management Methods

    def _allowed_camera_ids(self) -> set[str]:
        limit = get_camera_limit()
        enabled = [cid for cid, cfg in self.cameras.items() if getattr(cfg, "enabled", True)]
        return set(enabled[:limit])

    async def add_camera(self, config: CameraConfig) -> bool:
        """
        Add a new camera configuration

        Args:
            config: Camera configuration

        Returns:
            True if camera added successfully
        """
        try:
            # Validate configuration
            if not self._validate_camera_config(config):
                return False
            if getattr(config, "enabled", True):
                enabled_count = sum(1 for cfg in self.cameras.values() if getattr(cfg, "enabled", True))
                if enabled_count >= get_camera_limit():
                    logger.warning("Camera limit reached; refusing to add enabled camera")
                    return False

            config.updated_at = datetime.now()
            self.cameras[config.id] = config

            # Initialize health tracking
            self.camera_health[config.id] = CameraHealth(
                camera_id=config.id,
                status=CameraStatus.DISCONNECTED
            )

            # Save to database
            self._save_camera_to_db(config)

            # Connect if enabled
            if config.enabled:
                await self.connect_camera(config.id)

            logger.info(f"Added camera: {config.id} ({config.name})")
            return True

        except Exception as e:
            logger.error(f"Failed to add camera {config.id}: {e}")
            return False

    async def remove_camera(self, camera_id: str) -> bool:
        """
        Remove a camera configuration

        Args:
            camera_id: Camera ID to remove

        Returns:
            True if camera removed successfully
        """
        try:
            if camera_id not in self.cameras:
                logger.warning(f"Camera {camera_id} not found")
                return False

            # Disconnect camera first
            await self.disconnect_camera(camera_id)

            # Remove from collections
            del self.cameras[camera_id]
            if camera_id in self.camera_health:
                del self.camera_health[camera_id]
            if camera_id in self.ptz_capabilities:
                del self.ptz_capabilities[camera_id]

            # Remove from database
            self._delete_camera_from_db(camera_id)

            logger.info(f"Removed camera: {camera_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to remove camera {camera_id}: {e}")
            return False

    async def update_camera(self, camera_id: str, updates: Dict[str, Any]) -> bool:
        """
        Update camera configuration

        Args:
            camera_id: Camera ID to update
            updates: Dictionary of updates to apply

        Returns:
            True if camera updated successfully
        """
        try:
            if camera_id not in self.cameras:
                logger.warning(f"Camera {camera_id} not found")
                return False

            config = self.cameras[camera_id]
            was_enabled = config.enabled

            # Apply updates
            for key, value in updates.items():
                if hasattr(config, key):
                    setattr(config, key, value)

            config.updated_at = datetime.now()

            # Validate updated configuration
            if not self._validate_camera_config(config):
                return False

            # Save to database and sync to JSON
            self._save_camera_to_db(config)

            # Handle connection state changes
            if was_enabled and not config.enabled:
                await self.disconnect_camera(camera_id)
            elif not was_enabled and config.enabled:
                await self.connect_camera(camera_id)
            elif config.enabled:
                # Reconnect to apply changes
                await self.reconnect_camera(camera_id)

            logger.info(f"Updated camera: {camera_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to update camera {camera_id}: {e}")
            return False

    async def toggle_recording(self, camera_id: str, enable: bool) -> bool:
        """Toggle continuous recording for a camera via MediaMTX passthrough."""
        if camera_id not in self.cameras:
            logger.warning("toggle_recording: camera %s not found", camera_id)
            return False

        config = self.cameras[camera_id]
        stream_path = camera_id

        if enable:
            ok = await self.mediamtx_client.enable_recording(stream_path, source=config.rtsp_url)
        else:
            ok = await self.mediamtx_client.disable_recording(stream_path)

        if ok:
            config.recording = enable
            config.updated_at = datetime.now()
            self._save_camera_to_db(config)
            logger.info("Recording %s for camera %s", "enabled" if enable else "disabled", camera_id)
        return ok

    async def get_recording_status(self, camera_id: str) -> Optional[bool]:
        """Check whether continuous recording is active for a camera."""
        return await self.mediamtx_client.get_recording_status(camera_id)

    async def recover_camera_configurations(self):
        """Recover and validate all camera configurations"""
        try:
            logger.info("Starting camera configuration recovery...")
            
            # First, try to load from database
            self._load_cameras_from_db()
            
            # If no cameras loaded, try JSON fallback
            if not self.cameras:
                logger.warning("No cameras found in database, trying JSON fallback...")
                self._load_cameras_from_json_fallback()
            
            # Validate and fix any corrupted configurations
            cameras_to_remove = []
            for camera_id, config in self.cameras.items():
                try:
                    # Extract IP address from RTSP URL if missing
                    if not hasattr(config, 'ip_address') or not config.ip_address:
                        from urllib.parse import urlparse
                        parsed = urlparse(config.rtsp_url)
                        config.ip_address = parsed.hostname
                        logger.info(f"Recovered IP address for {camera_id}: {config.ip_address}")
                    
                    # Validate RTSP URL
                    if not config.rtsp_url or not self._validate_camera_config(config):
                        logger.warning(f"Invalid configuration for camera {camera_id}, marking for removal")
                        cameras_to_remove.append(camera_id)
                        continue
                    
                    # Ensure all required fields are present
                    if not config.name:
                        config.name = f"Camera {camera_id[:8]}"
                    
                    if not config.location:
                        config.location = "Default"
                    
                    # Update timestamps
                    config.updated_at = datetime.now()
                    
                except Exception as e:
                    logger.error(f"Error validating camera {camera_id}: {e}")
                    cameras_to_remove.append(camera_id)
            
            # Remove invalid cameras
            for camera_id in cameras_to_remove:
                await self.remove_camera(camera_id)
            
            # Save all valid configurations
            for config in self.cameras.values():
                self._save_camera_to_db(config)
            
            logger.info(f"Camera configuration recovery completed. {len(self.cameras)} cameras loaded.")
            return len(self.cameras)
            
        except Exception as e:
            logger.error(f"Failed to recover camera configurations: {e}")
            return 0

    async def auto_reconnect_cameras(self):
        """Automatically reconnect all enabled cameras"""
        try:
            logger.info("Starting automatic camera reconnection...")
            
            for camera_id, config in self.cameras.items():
                if config.enabled:
                    try:
                        # Check if camera is already connected
                        if camera_id in self.active_streams:
                            continue
                        
                        logger.info(f"Auto-reconnecting camera: {camera_id}")
                        await self.connect_camera(camera_id)
                        
                        # Wait a bit between connections to avoid overwhelming the system
                        await asyncio.sleep(1)
                        
                    except Exception as e:
                        logger.error(f"Failed to auto-reconnect camera {camera_id}: {e}")
            
            logger.info("Automatic camera reconnection completed")
            
        except Exception as e:
            logger.error(f"Failed to auto-reconnect cameras: {e}")

    async def validate_camera_connectivity(self, camera_id: str) -> bool:
        """Validate camera connectivity and configuration"""
        try:
            if camera_id not in self.cameras:
                return False
            
            config = self.cameras[camera_id]
            
            # Test RTSP connection
            if not await self._test_rtsp_connection(config):
                logger.warning(f"RTSP connection failed for camera {camera_id}")
                return False
            
            # Test MediaMTX integration if enabled
            if config.webrtc_enabled:
                try:
                    await self.mediamtx_client.test_stream_availability(camera_id)
                except Exception as e:
                    logger.warning(f"MediaMTX test failed for camera {camera_id}: {e}")
                    # Don't fail completely, just log the warning
            
            logger.info(f"Camera {camera_id} connectivity validated successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to validate camera {camera_id} connectivity: {e}")
            return False

    async def connect_camera(self, camera_id: str) -> bool:
        """
        Connect to a camera and start streaming

        Args:
            camera_id: Camera ID to connect

        Returns:
            True if connection successful
        """
        try:
            config = self.cameras.get(camera_id)
            if not config:
                logger.error(f"Camera configuration not found: {camera_id}")
                return False
            if camera_id not in self._allowed_camera_ids():
                logger.warning(f"Camera {camera_id} blocked by 4-camera limit")
                return False

            if not config.enabled:
                logger.info(f"Camera {camera_id} is disabled")
                return False

            logger.info(f"Connecting to camera: {camera_id} ({config.name})")

            # Prevent duplicate connect storms (especially when monitor + auto-reconnect overlap).
            now_ts = time.time()
            with self._connect_guard:
                # Cooldown: don't attempt connects too frequently for a flapping camera.
                last = float(self._last_connect_attempt.get(camera_id, 0.0))
                if (now_ts - last) < 5.0:
                    logger.info("Skipping connect for %s (cooldown %.1fs)", camera_id, 5.0 - (now_ts - last))
                    return False
                self._last_connect_attempt[camera_id] = now_ts

                if camera_id in self._connect_inflight:
                    logger.info("Connect already in progress for %s; skipping duplicate", camera_id)
                    return True
                self._connect_inflight.add(camera_id)

            # Update status
            self.camera_health[camera_id].status = CameraStatus.CONNECTING

            # Test RTSP connection
            if not await self._test_rtsp_connection(config):
                logger.error(f"Failed to connect to RTSP stream: {camera_id}")
                self.camera_health[camera_id].status = CameraStatus.ERROR
                return False

            # Configure MediaMTX stream source (for browser/WebRTC/HLS delivery), but do not
            # start a Python-side WebRTC receiver unless explicitly enabled.
            if config.webrtc_enabled:
                stream_path = camera_id  # Use camera_id directly as path for consistency
                # Use force_recreate=False to allow existing stable streams to persist
                # Only if connection fails will we try again with force_recreate=True
                success = await self.mediamtx_client.configure_stream_source(
                    stream_path, 
                    config.rtsp_url,
                    force_recreate=False
                )

                # Also publish the substream as {id}_sub so widgets can
                # auto-switch to low-res without hammering the camera.
                # Probe first if we don't already know whether it works.
                if config.substream_rtsp_url and config.substream_capable is not False:
                    try:
                        if config.substream_capable is None:
                            await self._probe_substream(config)
                        if config.substream_capable:
                            await self._ensure_mediamtx_sub_path(config)
                    except Exception as exc:
                        logger.warning(
                            "Sub stream setup failed for %s: %s", camera_id, exc,
                        )

                if success:
                    if self.enable_webrtc_receiver:
                        # Optional: Connect Python-side WebRTC receiver (aiortc). Not used in desktop-light mode.
                        stream_config = StreamConfig(
                            stream_path=stream_path,
                            rtsp_url=config.rtsp_url,
                            quality=config.stream_quality.value,
                            enable_audio=config.audio_enabled
                        )

                        webrtc_success = await self.mediamtx_client.connect_stream(stream_config)
                        if webrtc_success:
                            self.webrtc_streams[camera_id] = True
                            logger.info(f"WebRTC receiver connected for camera: {camera_id}")
                        else:
                            logger.warning(f"WebRTC receiver connection failed for camera: {camera_id}")
                            self.webrtc_streams[camera_id] = False
                    else:
                        # We still consider WebRTC available if MediaMTX path is configured.
                        self.webrtc_streams[camera_id] = True
                else:
                    logger.warning(f"Failed to configure MediaMTX path for {camera_id}")
                    self.webrtc_streams[camera_id] = False

            # Enable continuous recording if the camera config requests it
            if config.recording and config.webrtc_enabled:
                try:
                    rec_path = getattr(config, 'record_path', '') or ''
                    await self.mediamtx_client.enable_recording(camera_id, record_path=rec_path)
                    logger.info("Continuous recording enabled for %s", camera_id)
                except Exception as rec_err:
                    logger.warning("Could not enable recording for %s: %s", camera_id, rec_err)

            # Start traditional RTSP stream for fallback and processing
            await self._start_rtsp_stream(camera_id, config)

            # Detect PTZ capabilities
            if config.ptz_enabled:
                await self._detect_ptz_capabilities(camera_id, config)

            # Update status
            self.camera_health[camera_id].status = CameraStatus.CONNECTED
            self.camera_health[camera_id].connection_uptime = timedelta(0)

            if self.on_camera_connected:
                await self._safe_callback(self.on_camera_connected, camera_id)

            logger.info(f"Successfully connected to camera: {camera_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to connect to camera {camera_id}: {e}")
            self.camera_health[camera_id].status = CameraStatus.ERROR
            self.camera_health[camera_id].last_error = str(e)
            return False
        finally:
            try:
                with self._connect_guard:
                    self._connect_inflight.discard(camera_id)
            except Exception:
                pass

    async def disconnect_camera(self, camera_id: str) -> bool:
        """
        Disconnect from a camera

        Args:
            camera_id: Camera ID to disconnect

        Returns:
            True if disconnection successful
        """
        try:
            logger.info(f"Disconnecting camera: {camera_id}")

            # Disconnect optional Python-side WebRTC receiver (if it was used).
            if camera_id in self.webrtc_streams:
                stream_path = camera_id  # Use camera_id directly
                if self.enable_webrtc_receiver:
                    await self.mediamtx_client.disconnect_stream(stream_path)

                # Do not delete MediaMTX paths by default (reduces flapping, enables on-demand pulls).
                if self.cleanup_mediamtx_paths_on_disconnect:
                    await self.mediamtx_client.delete_stream_path(stream_path)

                del self.webrtc_streams[camera_id]

            # Stop RTSP stream
            await self._stop_rtsp_stream(camera_id)

            # Update status
            if camera_id in self.camera_health:
                self.camera_health[camera_id].status = CameraStatus.DISCONNECTED
                self.camera_health[camera_id].webrtc_connected = False

            if self.on_camera_disconnected:
                await self._safe_callback(self.on_camera_disconnected, camera_id)

            logger.info(f"Disconnected camera: {camera_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to disconnect camera {camera_id}: {e}")
            return False

    # ===========================
    # Desktop viewer refcounting (on-demand connect/disconnect)
    # ===========================

    async def acquire_camera(self, camera_id: str) -> bool:
        """
        Indicate a UI consumer wants frames for this camera. Ensures it is connected.
        """
        # Cancel any pending idle disconnect
        try:
            h = self._idle_disconnect_handles.pop(camera_id, None)
            if h is not None:
                try:
                    h.cancel()
                except Exception:
                    pass
        except Exception:
            pass

        self._viewer_counts[camera_id] = int(self._viewer_counts.get(camera_id, 0)) + 1
        if camera_id in self.active_streams:
            return True
        return await self.connect_camera(camera_id)

    async def release_camera(self, camera_id: str) -> None:
        """
        Indicate a UI consumer is done with this camera. Disconnects after idle timeout if no viewers remain.
        """
        current = int(self._viewer_counts.get(camera_id, 0))
        if current <= 1:
            self._viewer_counts.pop(camera_id, None)
        else:
            self._viewer_counts[camera_id] = current - 1

        if self._viewer_counts.get(camera_id, 0) > 0:
            return

        # Schedule idle disconnect (on the asyncio loop that owns this CameraManager).
        loop = getattr(self, "_loop", None)
        if not loop or not getattr(loop, "is_running", lambda: False)():
            return

        def _schedule():
            async def _do():
                try:
                    # If a viewer re-acquired the camera, skip.
                    if int(self._viewer_counts.get(camera_id, 0)) > 0:
                        return
                    await self.disconnect_camera(camera_id)
                finally:
                    self._idle_disconnect_handles.pop(camera_id, None)

            return asyncio.create_task(_do())

        try:
            handle = loop.call_later(float(self.idle_disconnect_seconds), _schedule)
            self._idle_disconnect_handles[camera_id] = handle
        except Exception:
            return

    async def reconnect_camera(self, camera_id: str) -> bool:
        """
        Reconnect to a camera

        Args:
            camera_id: Camera ID to reconnect

        Returns:
            True if reconnection successful
        """
        # Guard against runaway overlapping reconnects.
        with self._connect_guard:
            if camera_id in self._reconnect_inflight:
                logger.info("Reconnect already in progress for %s; skipping duplicate", camera_id)
                return True
            self._reconnect_inflight.add(camera_id)
        try:
            logger.info(f"Reconnecting camera: {camera_id}")
            await self.disconnect_camera(camera_id)
            # Async sleep: do NOT block the event loop.
            await asyncio.sleep(2)
            return await self.connect_camera(camera_id)
        finally:
            try:
                with self._connect_guard:
                    self._reconnect_inflight.discard(camera_id)
            except Exception:
                pass

    # Stream Management Methods

    async def get_webrtc_stream_url(self, camera_id: str) -> Optional[str]:
        """
        Get WebRTC stream URL for a camera

        Args:
            camera_id: Camera ID

        Returns:
            WebRTC stream URL if available
        """
        if camera_id not in self.cameras:
            return None

        config = self.cameras[camera_id]
        if not config.webrtc_enabled or not self.webrtc_streams.get(camera_id, False):
            return None

        stream_path = camera_id  # Use camera_id directly
        # Return the backend proxy WHEP endpoint (browser-safe; avoids direct :8889 access).
        return f"/proxy/webrtc/{stream_path}/whep"

    async def set_stream_quality(self, camera_id: str, quality: StreamQuality) -> bool:
        """Change stream quality for a camera (fast RTSP-only switch).

        For LOW (sub) quality, prefers MediaMTX's local re-stream of the
        sub path so the swap to substream is fast and the camera's RTSP
        session count isn't increased.  Falls back to direct sub URL,
        then to main if the sub stream is not capable.
        """
        try:
            if camera_id not in self.cameras:
                return False

            config = self.cameras[camera_id]
            if config.stream_quality == quality:
                return True

            old_quality = config.stream_quality
            config.stream_quality = quality
            config.stream_priority = "sub" if quality == StreamQuality.LOW else "main"
            config.updated_at = datetime.now()
            self._save_camera_to_db(config)

            if config.enabled and self.camera_health.get(camera_id, CameraHealth(
                camera_id=camera_id, status=CameraStatus.DISCONNECTED
            )).status == CameraStatus.CONNECTED:
                want_sub = (
                    quality == StreamQuality.LOW
                    and bool(config.substream_rtsp_url)
                    and config.substream_capable is not False
                )

                if want_sub:
                    if config.substream_capable is None:
                        await self._probe_substream(config)

                    if config.substream_capable is False:
                        logger.info("Substream probe failed for %s; staying on main", camera_id)
                        new_url = config.rtsp_url
                    else:
                        await self._ensure_mediamtx_sub_path(config)
                        sub_path = (config.mediamtx_sub_path
                                    or f"{config.mediamtx_path or camera_id}_sub")
                        mtx_host = self.mediamtx_client.mediamtx_host or "localhost"
                        local_sub_url = f"rtsp://{mtx_host}:8554/{sub_path}"
                        new_url = local_sub_url
                else:
                    new_url = config.rtsp_url

                ok = await self._switch_rtsp_stream(camera_id, new_url)
                if not ok and want_sub and config.substream_capable is not False:
                    logger.warning(
                        "Local sub-stream switch failed for %s; trying direct sub URL",
                        camera_id,
                    )
                    ok = await self._switch_rtsp_stream(
                        camera_id, config.substream_rtsp_url
                    )
                if not ok:
                    logger.warning(
                        "Fast-switch failed for %s; reverting quality to %s",
                        camera_id, old_quality.value,
                    )
                    config.stream_quality = old_quality
                    config.stream_priority = "sub" if old_quality == StreamQuality.LOW else "main"
                    if want_sub:
                        config.substream_capable = False
                    config.updated_at = datetime.now()
                    self._save_camera_to_db(config)
                    return False

            logger.info(f"Changed stream quality for camera {camera_id} to {quality.value}")
            return True

        except Exception as e:
            logger.error(f"Failed to set stream quality for camera {camera_id}: {e}")
            return False

    async def _probe_substream(self, config: CameraConfig, force: bool = False) -> bool:
        """Probe the substream URL to verify it works and capture its
        resolution.  Persists the result on the config (substream_capable,
        substream_width, substream_height, substream_last_check).

        Args:
            config: Camera config (must have substream_rtsp_url).
            force: When True, re-test even if a previous result is cached.

        Returns:
            True if the sub stream produced a valid frame within the timeout.
        """
        if not config.substream_rtsp_url:
            config.substream_capable = False
            config.substream_last_check = time.time()
            self._save_camera_to_db(config)
            return False
        if not force and config.substream_capable is not None:
            return bool(config.substream_capable)

        lock = self._substream_probe_locks.get(config.id)
        if lock is None:
            lock = asyncio.Lock()
            self._substream_probe_locks[config.id] = lock

        async with lock:
            if not force and config.substream_capable is not None:
                return bool(config.substream_capable)

            url = config.substream_rtsp_url
            logger.info("Probing substream for %s: %s", config.id, url)

            def _probe_sync() -> Tuple[bool, Optional[int], Optional[int]]:
                cap = None
                try:
                    os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = (
                        "rtsp_transport;tcp|buffer_size;1048576|max_delay;500000|stimeout;3000000"
                    )
                    cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
                    if not cap.isOpened():
                        return False, None, None
                    ok, frame = cap.read()
                    if not ok or frame is None:
                        return False, None, None
                    h, w = frame.shape[:2]
                    return True, int(w), int(h)
                except Exception as exc:
                    logger.warning("Substream probe error for %s: %s", config.id, exc)
                    return False, None, None
                finally:
                    try:
                        if cap is not None:
                            cap.release()
                    except Exception:
                        pass

            try:
                ok, w, h = await asyncio.wait_for(
                    asyncio.to_thread(_probe_sync), timeout=5.0
                )
            except asyncio.TimeoutError:
                logger.warning("Substream probe timed out for %s", config.id)
                ok, w, h = False, None, None

            config.substream_capable = bool(ok)
            config.substream_width = w
            config.substream_height = h
            config.substream_last_check = time.time()
            try:
                self._save_camera_to_db(config)
            except Exception:
                pass

            if ok:
                logger.info("Substream OK for %s: %dx%d", config.id, w or 0, h or 0)
            else:
                logger.info("Substream NOT capable for %s; will use main only", config.id)

            return ok

    async def _ensure_mediamtx_sub_path(self, config: CameraConfig) -> bool:
        """Make sure MediaMTX is publishing the sub stream as {id}_sub so
        any number of viewers can pull it from localhost without hitting
        the camera's session limit."""
        if not config.substream_rtsp_url or config.substream_capable is False:
            return False
        sub_path = (config.mediamtx_sub_path
                    or f"{config.mediamtx_path or config.id}_sub")
        config.mediamtx_sub_path = sub_path
        try:
            ok = await self.mediamtx_client.configure_stream_source(
                sub_path, config.substream_rtsp_url, force_recreate=False,
            )
            if ok:
                logger.info("MediaMTX sub path published for %s as %s",
                            config.id, sub_path)
            return ok
        except Exception as exc:
            logger.warning("Failed to publish sub path for %s: %s",
                           config.id, exc)
            return False

    # PTZ control lives in core.ptz_manager.PTZManager (HTTP route: /api/cameras/<id>/ptz).
    # The legacy CGI-only `send_ptz_command` here was duplicate and divergent;
    # it has been removed. Use `core.ptz_manager.get_ptz_manager().execute_command(...)`.

    # Information and Status Methods

    def get_camera(self, camera_id: str) -> Optional[CameraConfig]:
        """Get camera configuration"""
        return self.cameras.get(camera_id)

    def get_all_cameras(self) -> Dict[str, CameraConfig]:
        """Get all camera configurations"""
        return self.cameras.copy()

    def get_camera_health(self, camera_id: str) -> Optional[CameraHealth]:
        """Get camera health status"""
        return self.camera_health.get(camera_id)

    def get_all_camera_health(self) -> Dict[str, CameraHealth]:
        """Get health status for all cameras"""
        return self.camera_health.copy()

    def get_connected_cameras(self) -> List[str]:
        """Get list of connected camera IDs"""
        return [
            camera_id for camera_id, health in self.camera_health.items()
            if health.status == CameraStatus.CONNECTED
        ]

    def get_webrtc_enabled_cameras(self) -> List[str]:
        """Get list of cameras with WebRTC enabled"""
        return [
            camera_id for camera_id, config in self.cameras.items()
            if config.webrtc_enabled and self.webrtc_streams.get(camera_id, False)
        ]

    def is_camera_connected(self, camera_id: str) -> bool:
        """Check if camera is connected"""
        health = self.camera_health.get(camera_id)
        return health is not None and health.status == CameraStatus.CONNECTED

    def get_ptz_capabilities(self, camera_id: str) -> Optional[PTZCapabilities]:
        """Get PTZ capabilities for camera"""
        return self.ptz_capabilities.get(camera_id)

    def get_usage_snapshot(self) -> Dict[str, Dict[str, Any]]:
        """
        Lightweight introspection snapshot for Desktop runtime/health UI.

        Returns:
            Dict[camera_id, {
              name: str,
              viewer_refcount: int,
              connected: bool,
              status: str,
              frame_rate: float,
              webrtc_connected: bool,
              idle_disconnect_scheduled: bool,
            }]
        """
        out: Dict[str, Dict[str, Any]] = {}
        try:
            camera_ids = list(getattr(self, "cameras", {}).keys())
        except Exception:
            camera_ids = []

        for camera_id in camera_ids:
            try:
                cfg = (self.cameras or {}).get(camera_id)
                health = (self.camera_health or {}).get(camera_id)
                out[camera_id] = {
                    "name": getattr(cfg, "name", None) or str(camera_id),
                    "viewer_refcount": int((self._viewer_counts or {}).get(camera_id, 0)),
                    "connected": bool(camera_id in (self.active_streams or {})),
                    "status": getattr(getattr(health, "status", None), "value", None)
                    or str(getattr(health, "status", CameraStatus.DISCONNECTED).value),
                    "frame_rate": float(getattr(health, "frame_rate", 0.0) or 0.0),
                    "webrtc_connected": bool(getattr(health, "webrtc_connected", False)),
                    "idle_disconnect_scheduled": bool(camera_id in (self._idle_disconnect_handles or {})),
                }
            except Exception:
                # Never break the UI on snapshot errors; return best-effort per-camera.
                out[camera_id] = {
                    "name": str(camera_id),
                    "viewer_refcount": int((self._viewer_counts or {}).get(camera_id, 0)) if hasattr(self, "_viewer_counts") else 0,
                    "connected": bool(camera_id in (self.active_streams or {})) if hasattr(self, "active_streams") else False,
                    "status": "unknown",
                    "frame_rate": 0.0,
                    "webrtc_connected": False,
                    "idle_disconnect_scheduled": bool(camera_id in (self._idle_disconnect_handles or {})) if hasattr(self, "_idle_disconnect_handles") else False,
                }
        return out

    # Private Methods

    def _init_database(self):
        """Initialize SQLite database for camera configurations"""
        try:
            os.makedirs(os.path.dirname(self.db_path), exist_ok=True)

            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                             CREATE TABLE IF NOT EXISTS cameras
                             (
                                 id
                                 TEXT
                                 PRIMARY
                                 KEY,
                                 name
                                 TEXT
                                 NOT
                                 NULL,
                                 rtsp_url
                                 TEXT
                                 NOT
                                 NULL,
                                 location
                                 TEXT,
                                 enabled
                                 BOOLEAN
                                 DEFAULT
                                 1,
                                 ai_analysis
                                 BOOLEAN
                                 DEFAULT
                                 1,
                                 recording
                                 BOOLEAN
                                 DEFAULT
                                 0,
                                 ptz_enabled
                                 BOOLEAN
                                 DEFAULT
                                 0,
                                 ptz_url
                                 TEXT,
                                 username
                                 TEXT,
                                 password
                                 TEXT,
                                 port
                                 INTEGER
                                 DEFAULT
                                 554,
                                 stream_path
                                 TEXT,
                                 stream_priority
                                 TEXT
                                 DEFAULT
                                 'main',
                                 stream_quality
                                 TEXT
                                 DEFAULT
                                 'medium',
                                 webrtc_enabled
                                 BOOLEAN
                                 DEFAULT
                                 1,
                                 substream_path
                                 TEXT,
                                 substream_rtsp_url
                                 TEXT,
                                 backup_rtsp_url
                                 TEXT,
                                 ip_address
                                 TEXT,
                                 mediamtx_path
                                 TEXT,
                                 mediamtx_sub_path
                                 TEXT,
                                 custom_rtsp
                                 BOOLEAN
                                 DEFAULT
                                 0,
                                 manufacturer
                                 TEXT,
                                 protocol
                                 TEXT,
                                 status
                                 TEXT
                                 DEFAULT
                                 'offline',
                                 motion_detection
                                 BOOLEAN
                                 DEFAULT
                                 1,
                                 audio_enabled
                                 BOOLEAN
                                 DEFAULT
                                 0,
                                 night_vision
                                 BOOLEAN
                                 DEFAULT
                                 0,
                                 privacy_mask
                                 TEXT,
                                 extra_config
                                 TEXT,
                                 created_at
                                 TIMESTAMP,
                                 updated_at
                                 TIMESTAMP
                             )
                             """)
                columns = {
                    row[1]
                    for row in conn.execute("PRAGMA table_info(cameras)").fetchall()
                }
                column_migrations = {
                    "port": "ALTER TABLE cameras ADD COLUMN port INTEGER DEFAULT 554",
                    "stream_path": "ALTER TABLE cameras ADD COLUMN stream_path TEXT",
                    "stream_priority": "ALTER TABLE cameras ADD COLUMN stream_priority TEXT DEFAULT 'main'",
                    "substream_path": "ALTER TABLE cameras ADD COLUMN substream_path TEXT",
                    "substream_rtsp_url": "ALTER TABLE cameras ADD COLUMN substream_rtsp_url TEXT",
                    "ip_address": "ALTER TABLE cameras ADD COLUMN ip_address TEXT",
                    "mediamtx_path": "ALTER TABLE cameras ADD COLUMN mediamtx_path TEXT",
                    "mediamtx_sub_path": "ALTER TABLE cameras ADD COLUMN mediamtx_sub_path TEXT",
                    "substream_capable": "ALTER TABLE cameras ADD COLUMN substream_capable INTEGER",
                    "substream_width": "ALTER TABLE cameras ADD COLUMN substream_width INTEGER",
                    "substream_height": "ALTER TABLE cameras ADD COLUMN substream_height INTEGER",
                    "substream_last_check": "ALTER TABLE cameras ADD COLUMN substream_last_check REAL DEFAULT 0",
                    "custom_rtsp": "ALTER TABLE cameras ADD COLUMN custom_rtsp BOOLEAN DEFAULT 0",
                    "manufacturer": "ALTER TABLE cameras ADD COLUMN manufacturer TEXT",
                    "protocol": "ALTER TABLE cameras ADD COLUMN protocol TEXT",
                    "status": "ALTER TABLE cameras ADD COLUMN status TEXT DEFAULT 'offline'",
                    "extra_config": "ALTER TABLE cameras ADD COLUMN extra_config TEXT",
                }
                for column_name, ddl in column_migrations.items():
                    if column_name not in columns:
                        conn.execute(ddl)
                conn.commit()

        except Exception as e:
            logger.error(f"Failed to initialize database: {e}")

    def _load_cameras_from_db(self):
        """Load camera configurations from database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.execute("SELECT * FROM cameras")

                for row in cursor:
                    try:
                        privacy_mask = json.loads(row['privacy_mask']) if row['privacy_mask'] else []
                        extra_config = json.loads(row['extra_config']) if row['extra_config'] else {}
                        stream_priority = self._resolve_stream_priority(dict(row))
                        stream_path = row['stream_path'] or self._extract_stream_path(row['rtsp_url'])
                        substream_rtsp_url = row['substream_rtsp_url']
                        substream_path = row['substream_path'] or self._extract_stream_path(substream_rtsp_url)

                        row_keys = set(row.keys())

                        def _row_or(key: str, default):
                            return row[key] if key in row_keys and row[key] is not None else default

                        sub_capable_raw = _row_or('substream_capable', None)
                        sub_capable = None
                        if sub_capable_raw is not None:
                            sub_capable = bool(int(sub_capable_raw))

                        config = CameraConfig(
                            id=row['id'],
                            name=row['name'],
                            rtsp_url=row['rtsp_url'],
                            location=row['location'] or "",
                            enabled=bool(row['enabled']),
                            ai_analysis=bool(row['ai_analysis']),
                            recording=bool(row['recording']),
                            ptz_enabled=bool(row['ptz_enabled']),
                            ptz_url=row['ptz_url'],
                            username=row['username'],
                            password=row['password'],
                            port=int(row['port'] or 554),
                            stream_path=stream_path,
                            stream_priority=stream_priority,
                            stream_quality=self._resolve_stream_quality(dict(row)),
                            webrtc_enabled=bool(row['webrtc_enabled']),
                            substream_path=substream_path,
                            substream_rtsp_url=substream_rtsp_url,
                            backup_rtsp_url=row['backup_rtsp_url'],
                            ip_address=row['ip_address'],
                            mediamtx_path=row['mediamtx_path'],
                            mediamtx_sub_path=row['mediamtx_sub_path'],
                            substream_capable=sub_capable,
                            substream_width=_row_or('substream_width', None),
                            substream_height=_row_or('substream_height', None),
                            substream_last_check=float(_row_or('substream_last_check', 0.0) or 0.0),
                            custom_rtsp=bool(row['custom_rtsp']),
                            manufacturer=row['manufacturer'],
                            protocol=row['protocol'],
                            status=row['status'] or "offline",
                            motion_detection=bool(row['motion_detection']),
                            audio_enabled=bool(row['audio_enabled']),
                            night_vision=bool(row['night_vision']),
                            privacy_mask=privacy_mask,
                            extra_config=extra_config,
                            created_at=datetime.fromisoformat(row['created_at']) if row[
                                'created_at'] else datetime.now(),
                            updated_at=datetime.fromisoformat(row['updated_at']) if row[
                                'updated_at'] else datetime.now()
                        )
                        config.stream_priority = stream_priority
                        config.stream_quality = self._resolve_stream_quality({
                            "stream_priority": stream_priority,
                            "stream_quality": row['stream_quality'],
                        })
                        if not config.substream_path and config.substream_rtsp_url:
                            config.substream_path = self._extract_stream_path(config.substream_rtsp_url)

                        self.cameras[config.id] = config
                        self.camera_health[config.id] = CameraHealth(
                            camera_id=config.id,
                            status=CameraStatus.DISCONNECTED
                        )

                    except Exception as e:
                        logger.error(f"Failed to load camera configuration {row['id']}: {e}")

        except Exception as e:
            logger.error(f"Failed to load cameras from database: {e}")
            # Fallback to JSON file if database fails
            self._load_cameras_from_json_fallback()

    def _sync_cameras_to_json(self):
        """Sync camera configurations from database to JSON file with enhanced validation"""
        if self._json_sync_disabled:
            return
        try:
            # Prefer writing into data/ (works for Desktop + avoids root-owned bind mounts)
            json_path = "data/cameras.json" if os.access("data", os.W_OK) else "cameras.json"
            if os.path.exists(json_path):
                if not os.access(json_path, os.W_OK):
                    logger.warning(f"{json_path} is not writable; disabling JSON sync")
                    self._json_sync_disabled = True
                    return
            else:
                parent_dir = os.path.dirname(json_path) or "."
                if not os.access(parent_dir, os.W_OK):
                    logger.warning("Working directory is not writable; disabling cameras.json sync")
                    self._json_sync_disabled = True
                    return

            cameras_data = []
            for config in self.cameras.values():
                # Validate and fix RTSP URL if needed
                if hasattr(config, 'rtsp_url') and config.rtsp_url:
                    config.rtsp_url = self._validate_and_fix_rtsp_url(config.rtsp_url, config)

                cam_dict = dict(config.extra_config or {})
                cam_dict.update({
                    'id': config.id,
                    'name': config.name,
                    'rtsp_url': config.rtsp_url,
                    'location': config.location,
                    'enabled': config.enabled,
                    'ai_analysis': config.ai_analysis,
                    'recording': config.recording,
                    'ptz_enabled': config.ptz_enabled,
                    'ptz_url': config.ptz_url,
                    'username': config.username,
                    'password': config.password,
                    'port': getattr(config, 'port', 554) or 554,
                    'stream_path': getattr(config, 'stream_path', None) or self._extract_stream_path(config.rtsp_url),
                    'stream_quality': config.stream_quality.value,
                    'stream_priority': getattr(config, 'stream_priority', None) or self._resolve_stream_priority({
                        'stream_quality': config.stream_quality.value,
                    }),
                    'webrtc_enabled': config.webrtc_enabled,
                    'substream_path': getattr(config, 'substream_path', None),
                    'backup_rtsp_url': config.backup_rtsp_url,
                    'substream_rtsp_url': config.substream_rtsp_url,
                    'custom_rtsp': bool(getattr(config, 'custom_rtsp', False)),
                    'manufacturer': getattr(config, 'manufacturer', None),
                    'protocol': getattr(config, 'protocol', None),
                    'motion_detection': config.motion_detection,
                    'audio_enabled': config.audio_enabled,
                    'night_vision': config.night_vision,
                    'privacy_mask': config.privacy_mask,
                    'created_at': config.created_at.isoformat(),
                    'updated_at': config.updated_at.isoformat(),
                    'type': 'camera',
                    'device_type': 'camera',
                    'status': getattr(config, 'status', None) or 'offline'
                })

                if getattr(config, 'ip_address', None):
                    cam_dict['ip_address'] = config.ip_address
                    cam_dict['ip'] = config.ip_address

                if getattr(config, 'mediamtx_path', None):
                    cam_dict['mediamtx_path'] = config.mediamtx_path
                    cam_dict['hls_url'] = f"http://localhost:8888/{config.mediamtx_path}/index.m3u8"
                    cam_dict['webrtc_url'] = f"http://localhost:8889/{config.mediamtx_path}"
                    cam_dict['webrtc_whip_url'] = f"http://localhost:8889/{config.mediamtx_path}/whip"
                    cam_dict['webrtc_whep_url'] = f"http://localhost:8889/{config.mediamtx_path}/whep"
                    cam_dict['mediamtx_rtsp_url'] = f"rtsp://localhost:8554/{config.mediamtx_path}"
                    cam_dict['ready'] = True
                    cam_dict['mediamtx_ready'] = True
                    cam_dict['readers_count'] = 0
                    cam_dict['publishers_count'] = 1

                if getattr(config, 'mediamtx_sub_path', None):
                    cam_dict['mediamtx_sub_path'] = config.mediamtx_sub_path
                    cam_dict['hls_sub_url'] = f"http://localhost:8888/{config.mediamtx_sub_path}/index.m3u8"
                    cam_dict['webrtc_sub_whep_url'] = f"http://localhost:8889/{config.mediamtx_sub_path}/whep"
                
                cameras_data.append(cam_dict)
            
            # Create backup of current JSON file
            import shutil
            if os.path.exists(json_path):
                shutil.copy2(json_path, f"{json_path}.backup")
            
            with open(json_path, 'w') as f:
                json.dump(cameras_data, f, indent=2)
                
            logger.info(f"Synced {len(cameras_data)} cameras to JSON file with enhanced validation")
            
        except Exception as e:
            logger.error(f"Failed to sync cameras to JSON: {e}")

    @staticmethod
    def _resolve_stream_priority(data: dict) -> str:
        priority = (data.get("stream_priority") or "").strip().lower()
        if priority == "sub":
            return "sub"
        if priority == "main":
            return "main"

        raw_q = (data.get("stream_quality") or "").strip().lower()
        if raw_q == "low":
            return "sub"
        if raw_q in {"medium", "high", "ultra"}:
            return "main"
        return "main"

    @staticmethod
    def _resolve_stream_quality(data: dict) -> StreamQuality:
        """Determine the effective StreamQuality from camera config data.

        ``stream_priority`` (backend convention: ``"sub"`` / ``"main"``) takes
        precedence over ``stream_quality`` so the desktop honours the same
        preference the user set via the scanner / API.
        """
        priority = CameraManager._resolve_stream_priority(data)
        if priority == "sub":
            return StreamQuality.LOW

        raw_q = (data.get("stream_quality") or "").strip().lower()
        if raw_q:
            try:
                quality = StreamQuality(raw_q)
                if quality != StreamQuality.LOW:
                    return quality
            except ValueError:
                pass
        return StreamQuality.MEDIUM

    @staticmethod
    def _extract_stream_path(rtsp_url: Optional[str]) -> Optional[str]:
        if not rtsp_url:
            return None
        try:
            parsed = urlparse(rtsp_url)
            path = parsed.path or '/'
            if parsed.query:
                path += f"?{parsed.query}"
            return path
        except Exception:
            return None

    @staticmethod
    def _build_rtsp_with_path(rtsp_url: Optional[str], new_path: Optional[str]) -> Optional[str]:
        if not rtsp_url or not new_path:
            return None
        try:
            parsed = urlparse(rtsp_url)
            path_only = new_path
            query = ""
            if "?" in new_path:
                path_only, query = new_path.split("?", 1)
            if not path_only:
                path_only = "/"
            return urlunparse(parsed._replace(path=path_only, query=query))
        except Exception:
            return None

    @staticmethod
    def _extract_extra_config(data: dict) -> Dict[str, Any]:
        known_fields = {
            "id", "name", "rtsp_url", "location", "enabled", "ai_analysis", "recording",
            "ptz_enabled", "ptz_url", "username", "password", "port", "stream_path",
            "stream_priority", "stream_quality", "webrtc_enabled", "substream_path",
            "substream_rtsp_url", "backup_rtsp_url", "ip_address", "ip", "mediamtx_path",
            "mediamtx_sub_path", "custom_rtsp", "manufacturer", "protocol", "status",
            "motion_detection", "audio_enabled", "night_vision", "privacy_mask",
            "created_at", "updated_at", "type", "device_type"
        }
        return {
            key: value
            for key, value in (data or {}).items()
            if key not in known_fields
        }

    def _resolve_substream_rtsp_url(self, data: dict, rtsp_url: str) -> Optional[str]:
        explicit_substream_path = "substream_path" in data
        explicit_substream_rtsp_url = "substream_rtsp_url" in data
        substream_rtsp_url = str(data.get("substream_rtsp_url") or "").strip() or None
        substream_path = data.get("substream_path")
        if not substream_path and substream_rtsp_url:
            substream_path = self._extract_stream_path(substream_rtsp_url)

        if not substream_rtsp_url and substream_path:
            substream_rtsp_url = self._build_rtsp_with_path(rtsp_url, str(substream_path))

        if (
            substream_rtsp_url is None and
            not explicit_substream_path and
            not explicit_substream_rtsp_url
        ):
            return self._compute_substream_url(rtsp_url)
        return substream_rtsp_url

    def _apply_camera_data_to_config(self, config: CameraConfig, data: dict):
        config.name = str(data.get("name") or config.name or f"Camera {config.id[:8]}").strip()
        config.location = str(data.get("location") or config.location or "Default").strip()
        config.enabled = bool(data.get("enabled", config.enabled))
        config.ai_analysis = bool(data.get("ai_analysis", config.ai_analysis))
        config.recording = bool(data.get("recording", config.recording))
        config.ptz_enabled = bool(data.get("ptz_enabled", config.ptz_enabled))
        config.ptz_url = data.get("ptz_url", config.ptz_url)
        config.username = data.get("username", config.username)
        config.password = data.get("password", config.password)

        try:
            config.port = int(data.get("port", config.port or 554) or 554)
        except Exception:
            config.port = 554

        config.stream_priority = self._resolve_stream_priority(data)
        config.stream_quality = self._resolve_stream_quality(data)
        config.webrtc_enabled = bool(data.get("webrtc_enabled", config.webrtc_enabled))
        config.backup_rtsp_url = data.get("backup_rtsp_url", config.backup_rtsp_url)
        config.motion_detection = bool(data.get("motion_detection", config.motion_detection))
        config.audio_enabled = bool(data.get("audio_enabled", config.audio_enabled))
        config.night_vision = bool(data.get("night_vision", config.night_vision))
        config.privacy_mask = data.get("privacy_mask") if isinstance(data.get("privacy_mask"), list) else (config.privacy_mask or [])

        config.ip_address = data.get("ip_address") or data.get("ip") or config.ip_address
        config.mediamtx_path = data.get("mediamtx_path", config.mediamtx_path)
        config.mediamtx_sub_path = data.get("mediamtx_sub_path", config.mediamtx_sub_path)
        config.custom_rtsp = bool(data.get("custom_rtsp", config.custom_rtsp))
        config.manufacturer = data.get("manufacturer", config.manufacturer)
        config.protocol = data.get("protocol", config.protocol)
        config.status = data.get("status", config.status or "offline")

        rtsp_url = str(data.get("rtsp_url") or config.rtsp_url or "").strip()
        config.rtsp_url = self._validate_and_fix_rtsp_url(rtsp_url, config)
        config.stream_path = data.get("stream_path") or self._extract_stream_path(config.rtsp_url)

        if "substream_path" in data:
            config.substream_path = data.get("substream_path") or None
        elif not config.substream_path and data.get("substream_rtsp_url"):
            config.substream_path = self._extract_stream_path(data.get("substream_rtsp_url"))

        config.substream_rtsp_url = self._resolve_substream_rtsp_url(data, config.rtsp_url)
        if not config.substream_path and config.substream_rtsp_url:
            config.substream_path = self._extract_stream_path(config.substream_rtsp_url)

        if data.get("created_at"):
            try:
                config.created_at = datetime.fromisoformat(data["created_at"])
            except Exception:
                pass
        if data.get("updated_at"):
            try:
                config.updated_at = datetime.fromisoformat(data["updated_at"])
            except Exception:
                config.updated_at = datetime.now()
        else:
            config.updated_at = datetime.now()

        config.extra_config = self._extract_extra_config(data)

    @staticmethod
    def _compute_substream_url(rtsp_url: str) -> Optional[str]:
        """Derive a substream RTSP URL from the main stream URL.

        Applies common IP camera path conventions (Onvif /media/video2,
        Hikvision /Streaming/Channels/102, Dahua subtype=1, etc.).
        Returns None if the URL cannot be parsed.
        """
        if not rtsp_url:
            return None
        try:
            parsed = urlparse(rtsp_url)
            path = parsed.path or '/'
            query = parsed.query or ''
            full_path = f"{path}?{query}" if query else path

            if re.search(r'/media/video\d+', full_path, flags=re.IGNORECASE):
                full_path = re.sub(r'/media/video\d+', '/media/video2', full_path, flags=re.IGNORECASE)
            elif re.search(r'subtype=\d+', full_path, flags=re.IGNORECASE):
                full_path = re.sub(r'subtype=\d+', 'subtype=1', full_path, flags=re.IGNORECASE)
            elif re.search(r'stream1', full_path, flags=re.IGNORECASE):
                full_path = re.sub(r'stream1', 'stream2', full_path, flags=re.IGNORECASE)
            elif re.search(r'/Streaming/Channels/\d+', full_path, flags=re.IGNORECASE):
                full_path = re.sub(r'/Streaming/Channels/\d+', '/Streaming/Channels/102', full_path, flags=re.IGNORECASE)
            elif 'channel=1' in full_path.lower() and 'subtype=' not in full_path.lower():
                sep = '&' if '?' in full_path else '?'
                full_path = f"{full_path}{sep}subtype=1"
            else:
                full_path = '/media/video2'

            new_path = full_path
            new_query = ''
            if '?' in full_path:
                new_path, new_query = full_path.split('?', 1)
            if not new_path:
                new_path = '/'

            return urlunparse(parsed._replace(path=new_path, query=new_query))
        except Exception:
            return None

    def _validate_and_fix_rtsp_url(self, rtsp_url: str, config: CameraConfig) -> str:
        """Validate and fix RTSP URL issues"""
        try:
            from urllib.parse import urlparse, parse_qs
            
            parsed = urlparse(rtsp_url)
            
            # Check for common issues
            if not parsed.scheme or parsed.scheme != 'rtsp':
                logger.warning(f"Invalid RTSP scheme for camera {config.id}: {rtsp_url}")
                return rtsp_url
            
            # Fix stream path inconsistencies (subtype mismatch)
            if hasattr(config, 'stream_path') and config.stream_path:
                query_params = parse_qs(parsed.query)
                stream_path_params = parse_qs(config.stream_path.lstrip('/'))
                
                # Ensure subtype consistency between stream_path and rtsp_url
                if 'subtype' in stream_path_params and 'subtype' in query_params:
                    if stream_path_params['subtype'] != query_params['subtype']:
                        # Use the stream_path subtype
                        query_params['subtype'] = stream_path_params['subtype']
                        # Rebuild query string
                        new_query = '&'.join([f"{k}={v[0]}" for k, v in query_params.items()])
                        fixed_url = f"rtsp://{parsed.netloc}{parsed.path}?{new_query}"
                        logger.info(f"Fixed subtype inconsistency in RTSP URL for camera {config.id}")
                        return fixed_url
            
            return rtsp_url
            
        except Exception as e:
            logger.error(f"Error validating RTSP URL for camera {config.id}: {e}")
            return rtsp_url

    def _load_cameras_from_json_fallback(self):
        """Load camera configurations from JSON file as fallback with enhanced validation"""
        try:
            if self._json_read_disabled:
                return

            json_path = None
            for candidate in self._json_path_candidates:
                if os.path.exists(candidate) and os.access(candidate, os.R_OK):
                    json_path = candidate
                    break
            if not json_path:
                logger.warning("No readable cameras.json found for fallback loading")
                self._json_read_disabled = True
                return

            with open(json_path, 'r') as f:
                cameras_data = json.load(f)
                
            for cam_data in cameras_data:
                try:
                    # Validate required fields
                    if not cam_data.get('id') or not cam_data.get('name') or not cam_data.get('rtsp_url'):
                        logger.warning(f"Skipping camera with missing required fields: {cam_data.get('id', 'unknown')}")
                        continue
                    
                    # Create CameraConfig object
                    config = CameraConfig(
                        id=cam_data['id'],
                        name=cam_data['name'],
                        rtsp_url=cam_data['rtsp_url'],
                        location=cam_data.get('location', 'Default')
                    )
                    self._apply_camera_data_to_config(config, cam_data)
                    
                    # Validate configuration
                    if self._validate_camera_config(config):
                        self.cameras[config.id] = config
                        logger.info(f"Loaded camera from JSON: {config.name} ({config.id})")
                    else:
                        logger.warning(f"Invalid camera configuration from JSON: {config.name} ({config.id})")
                        
                except Exception as e:
                    logger.error(f"Failed to load camera from JSON {cam_data.get('id', 'unknown')}: {e}")
                    
            logger.info(f"Loaded {len(self.cameras)} cameras from JSON fallback with enhanced validation")
            
        except Exception as e:
            logger.error(f"Failed to load cameras from JSON fallback: {e}")

    async def sync_cameras_json_to_db(self):
        """Sync cameras from JSON to database ensuring consistency"""
        try:
            logger.info("Starting JSON to database synchronization...")
            
            # Load current JSON data
            if self._json_read_disabled:
                logger.warning("JSON-to-DB sync skipped (JSON read disabled)")
                return 0

            json_path = None
            for candidate in self._json_path_candidates:
                if os.path.exists(candidate) and os.access(candidate, os.R_OK):
                    json_path = candidate
                    break
            if not json_path:
                logger.warning("JSON-to-DB sync skipped (no readable cameras.json)")
                self._json_read_disabled = True
                return 0

            with open(json_path, 'r') as f:
                json_cameras = json.load(f)
            
            synced_count = 0
            for cam_data in json_cameras:
                try:
                    camera_id = cam_data.get('id')
                    if not camera_id:
                        continue
                    
                    # Check if camera exists in database
                    if camera_id in self.cameras:
                        # Update existing camera
                        config = self.cameras[camera_id]
                        self._apply_camera_data_to_config(config, cam_data)
                        self._save_camera_to_db(config)
                        synced_count += 1
                        logger.info(f"Updated camera in database: {config.name} ({camera_id})")
                    else:
                        # Create new camera from JSON
                        config = CameraConfig(
                            id=camera_id,
                            name=cam_data.get('name', f"Camera {camera_id[:8]}"),
                            rtsp_url=cam_data.get('rtsp_url', ''),
                            location=cam_data.get('location', 'Default'),
                        )
                        self._apply_camera_data_to_config(config, cam_data)
                        
                        if self._validate_camera_config(config):
                            self.cameras[camera_id] = config
                            self._save_camera_to_db(config)
                            synced_count += 1
                            logger.info(f"Added camera from JSON to database: {config.name} ({camera_id})")
                        
                except Exception as e:
                    logger.error(f"Failed to sync camera {cam_data.get('id', 'unknown')}: {e}")
            
            logger.info(f"JSON to database synchronization completed. {synced_count} cameras synced.")
            return synced_count
            
        except Exception as e:
            logger.error(f"Failed to sync cameras from JSON to database: {e}")
            return 0

    async def sync_cameras_api_to_db(self, api_base: str = "http://localhost:5000/api", prune_missing: bool = False) -> int:
        """
        Sync cameras from the backend API (/api/devices) into this CameraManager's
        DB + in-memory map. This is the preferred sync path for the Desktop UI
        because the backend may be running in Docker with its own cameras.json.
        """
        try:
            logger.info("Starting API to database synchronization...")

            try:
                resp = requests.get(f"{api_base}/devices", timeout=5)
            except Exception as e:
                logger.warning(f"API sync skipped (cannot reach {api_base}/devices): {e}")
                return 0

            if not getattr(resp, "ok", False):
                logger.warning(f"API sync skipped ({api_base}/devices returned {resp.status_code})")
                return 0

            payload = {}
            try:
                payload = resp.json() or {}
            except Exception:
                payload = {}

            devices = payload.get("data") if isinstance(payload, dict) else None
            if devices is None:
                devices = payload.get("devices") if isinstance(payload, dict) else None
            if devices is None:
                devices = payload if isinstance(payload, list) else []
            if isinstance(devices, dict):
                devices = devices.get("devices", [])

            cameras = [
                d for d in (devices or [])
                if isinstance(d, dict) and (d.get("type") == "camera" or d.get("device_type") == "camera")
            ]
            api_ids = {str(cam.get("id") or "").strip() for cam in cameras if isinstance(cam, dict)}
            api_ids.discard("")

            def _to_bool(v, default=False):
                if v is None:
                    return default
                return bool(v)

            synced = 0
            for cam in cameras:
                camera_id = str(cam.get("id") or "").strip()
                if not camera_id:
                    continue

                name = str(cam.get("name") or f"Camera {camera_id[:8]}").strip()
                rtsp_url = (cam.get("rtsp_url") or cam.get("url") or "").strip()
                if not rtsp_url:
                    continue

                location = str(cam.get("location") or "Default").strip()

                enabled = _to_bool(cam.get("enabled"), True)
                webrtc_enabled = _to_bool(cam.get("webrtc_enabled"), True)
                motion_detection = _to_bool(cam.get("motion_detection"), True)
                ai_analysis = _to_bool(cam.get("ai_analysis"), True)
                recording = _to_bool(cam.get("recording"), False)
                ptz_enabled = _to_bool(cam.get("ptz_enabled"), False)
                audio_enabled = _to_bool(cam.get("audio_enabled"), False)
                night_vision = _to_bool(cam.get("night_vision"), False)
                stream_quality = self._resolve_stream_quality(cam)

                username = cam.get("username")
                password = cam.get("password")
                ptz_url = cam.get("ptz_url")

                ip_address = cam.get("ip_address") or cam.get("ip")
                mediamtx_path = cam.get("mediamtx_path")

                if camera_id in self.cameras:
                    cfg = self.cameras[camera_id]
                    self._apply_camera_data_to_config(cfg, cam)
                    self._save_camera_to_db(cfg)
                    synced += 1
                else:
                    cfg = CameraConfig(
                        id=camera_id,
                        name=name,
                        rtsp_url=rtsp_url,
                        location=location,
                        enabled=enabled,
                    )
                    self._apply_camera_data_to_config(cfg, cam)

                    self.cameras[camera_id] = cfg
                    self.camera_health[camera_id] = CameraHealth(
                        camera_id=camera_id,
                        status=CameraStatus.DISCONNECTED
                    )
                    self._save_camera_to_db(cfg)
                    synced += 1

            if prune_missing:
                removed = 0
                for existing_id in list(self.cameras.keys()):
                    if existing_id not in api_ids:
                        try:
                            await self.remove_camera(existing_id)
                            removed += 1
                        except Exception as e:
                            logger.warning(f"Failed to prune missing camera {existing_id}: {e}")
                if removed:
                    logger.info(f"Pruned {removed} camera(s) missing from API during sync.")

            logger.info(f"API to database synchronization completed. {synced} cameras synced.")
            return synced

        except Exception as e:
            logger.error(f"Failed to sync cameras from API to database: {e}")
            return 0

    async def ensure_detection_streaming(self, camera_id: str) -> bool:
        """Ensure detection data is properly streaming for a camera"""
        try:
            if camera_id not in self.cameras:
                logger.warning(f"Camera {camera_id} not found for detection streaming setup")
                return False
            
            config = self.cameras[camera_id]
            
            # Ensure motion detection is enabled
            if not config.motion_detection:
                logger.info(f"Motion detection disabled for camera {camera_id}, enabling...")
                config.motion_detection = True
                config.updated_at = datetime.now()
                self._save_camera_to_db(config)
            
            # Ensure camera is connected and streaming
            if camera_id not in self.active_streams:
                logger.info(f"Connecting camera {camera_id} for detection streaming...")
                await self.connect_camera(camera_id)
            
            # Verify WebSocket emission is working
            if hasattr(self, 'on_motion_detected') and self.on_motion_detected:
                logger.info(f"Detection streaming verified for camera {camera_id}")
                return True
            else:
                logger.warning(f"Motion detection callback not set for camera {camera_id}")
                return False
                
        except Exception as e:
            logger.error(f"Failed to ensure detection streaming for camera {camera_id}: {e}")
            return False

    async def sync_all_cameras_and_ensure_detection(self):
        """Comprehensive synchronization and detection streaming setup"""
        try:
            logger.info("Starting comprehensive camera synchronization and detection setup...")
            
            # Step 1: Sync JSON to database
            json_synced = await self.sync_cameras_json_to_db()
            logger.info(f"Synced {json_synced} cameras from JSON to database")
            
            # Step 2: Sync database to JSON
            self._sync_cameras_to_json()
            logger.info("Synced database to JSON file")
            
            # Step 3: Ensure all enabled cameras have detection streaming
            detection_enabled_count = 0
            for camera_id, config in self.cameras.items():
                if config.enabled and config.motion_detection:
                    try:
                        if await self.ensure_detection_streaming(camera_id):
                            detection_enabled_count += 1
                            logger.info(f"✅ Detection streaming enabled for {config.name} ({camera_id})")
                        else:
                            logger.warning(f"⚠️ Failed to enable detection streaming for {config.name} ({camera_id})")
                    except Exception as e:
                        logger.error(f"❌ Error setting up detection for {config.name} ({camera_id}): {e}")
            
            logger.info(f"Comprehensive sync completed. {detection_enabled_count}/{len(self.cameras)} cameras have detection streaming enabled.")
            return detection_enabled_count
            
        except Exception as e:
            logger.error(f"Failed to perform comprehensive camera sync: {e}")
            return 0

    async def verify_camera_detection_status(self, camera_id: str) -> Dict[str, Any]:
        """Verify the detection status of a specific camera"""
        try:
            if camera_id not in self.cameras:
                return {"status": "not_found", "camera_id": camera_id}
            
            config = self.cameras[camera_id]
            health = self.camera_health.get(camera_id)
            
            status = {
                "camera_id": camera_id,
                "name": config.name,
                "enabled": config.enabled,
                "motion_detection_enabled": config.motion_detection,
                "connected": camera_id in self.active_streams,
                "webrtc_connected": self.webrtc_streams.get(camera_id, False),
                "health_status": health.status.value if health else "unknown",
                "rtsp_url": config.rtsp_url,
                "mediamtx_ready": hasattr(config, 'mediamtx_path') and config.mediamtx_path is not None
            }
            
            # Test RTSP connection
            if config.enabled:
                rtsp_test = await self._test_rtsp_connection(config)
                status["rtsp_working"] = rtsp_test
            
            return status
            
        except Exception as e:
            logger.error(f"Error verifying camera detection status for {camera_id}: {e}")
            return {"status": "error", "camera_id": camera_id, "error": str(e)}

    async def get_all_cameras_detection_status(self) -> Dict[str, Any]:
        """Get detection status for all cameras"""
        try:
            statuses = {}
            for camera_id in self.cameras.keys():
                statuses[camera_id] = await self.verify_camera_detection_status(camera_id)
            
            return {
                "total_cameras": len(self.cameras),
                "enabled_cameras": len([c for c in self.cameras.values() if c.enabled]),
                "detection_enabled": len([c for c in self.cameras.values() if c.enabled and c.motion_detection]),
                "connected_cameras": len([c for c in self.cameras.keys() if c in self.active_streams]),
                "camera_statuses": statuses
            }
            
        except Exception as e:
            logger.error(f"Error getting all cameras detection status: {e}")
            return {"error": str(e)}

    def _save_camera_to_db(self, config: CameraConfig):
        """Save camera configuration to database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                sub_capable = getattr(config, 'substream_capable', None)
                sub_capable_int = None if sub_capable is None else int(bool(sub_capable))

                conn.execute("""
                    INSERT OR REPLACE INTO cameras (
                        id, name, rtsp_url, location, enabled, ai_analysis, recording,
                        ptz_enabled, ptz_url, username, password, port, stream_path,
                        stream_priority, stream_quality, webrtc_enabled, substream_path,
                        substream_rtsp_url, backup_rtsp_url, ip_address, mediamtx_path,
                        mediamtx_sub_path, substream_capable, substream_width,
                        substream_height, substream_last_check,
                        custom_rtsp, manufacturer, protocol, status,
                        motion_detection, audio_enabled, night_vision, privacy_mask,
                        extra_config, created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    config.id, config.name, config.rtsp_url, config.location,
                    config.enabled, config.ai_analysis, config.recording,
                    config.ptz_enabled, config.ptz_url, config.username, config.password,
                    getattr(config, 'port', 554) or 554,
                    getattr(config, 'stream_path', None),
                    getattr(config, 'stream_priority', None) or self._resolve_stream_priority({
                        'stream_quality': config.stream_quality.value,
                    }),
                    config.stream_quality.value, config.webrtc_enabled,
                    getattr(config, 'substream_path', None),
                    config.substream_rtsp_url, config.backup_rtsp_url,
                    getattr(config, 'ip_address', None),
                    getattr(config, 'mediamtx_path', None),
                    getattr(config, 'mediamtx_sub_path', None),
                    sub_capable_int,
                    getattr(config, 'substream_width', None),
                    getattr(config, 'substream_height', None),
                    float(getattr(config, 'substream_last_check', 0.0) or 0.0),
                    int(bool(getattr(config, 'custom_rtsp', False))),
                    getattr(config, 'manufacturer', None),
                    getattr(config, 'protocol', None),
                    getattr(config, 'status', None) or 'offline',
                    config.motion_detection, config.audio_enabled, config.night_vision,
                    json.dumps(config.privacy_mask),
                    json.dumps(getattr(config, 'extra_config', {}) or {}),
                    config.created_at.isoformat(), config.updated_at.isoformat()
                ))
                conn.commit()

            # Also sync to JSON file for backup
            self._sync_cameras_to_json()

        except Exception as e:
            logger.error(f"Failed to save camera {config.id} to database: {e}")

    def _delete_camera_from_db(self, camera_id: str):
        """Delete camera configuration from database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("DELETE FROM cameras WHERE id = ?", (camera_id,))
                conn.commit()

        except Exception as e:
            logger.error(f"Failed to delete camera {camera_id} from database: {e}")

    def _validate_camera_config(self, config: CameraConfig) -> bool:
        """Validate camera configuration"""
        if not config.id or not config.name or not config.rtsp_url:
            logger.error("Camera configuration missing required fields")
            return False

        # Validate RTSP URL format
        parsed = urlparse(config.rtsp_url)
        if not parsed.scheme or parsed.scheme not in ['rtsp', 'rtmp', 'http', 'https']:
            logger.error(f"Invalid stream URL format: {config.rtsp_url}")
            return False

        return True

    async def _test_rtsp_connection(self, config: CameraConfig) -> bool:
        """Test RTSP connection to camera.

        If MediaMTX already has a ready path for this camera (e.g. from a
        prior connect or ongoing recording), skip the expensive OpenCV probe
        to avoid conflicting with the existing RTSP pull.
        """
        try:
            path_info = await self.mediamtx_client.get_path_info(config.id)
            if path_info.get("ready"):
                logger.info("MediaMTX path already ready for %s; skipping OpenCV probe", config.id)
                return True
        except Exception:
            pass

        def _probe_sync(url: str) -> bool:
            cap = None
            try:
                os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp|buffer_size;1048576|max_delay;500000|stimeout;5000000"
                cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
                if not cap.isOpened():
                    return False
                ret, frame = cap.read()
                return bool(ret and frame is not None)
            finally:
                try:
                    if cap is not None:
                        cap.release()
                except Exception:
                    pass

        try:
            return bool(await asyncio.to_thread(_probe_sync, config.rtsp_url))
        except Exception as e:
            logger.error(f"RTSP connection test failed for {config.id}: {e}")
            return False

    async def _start_rtsp_stream(self, camera_id: str, config: CameraConfig):
        """Start RTSP stream capture thread.

        When MediaMTX already has a ready path for this camera (e.g. from an
        active recording), read from MediaMTX's local RTSP re-stream instead
        of opening a second direct connection to the camera.  Most IP cameras
        only allow 1-2 concurrent RTSP sessions, so a duplicate pull would
        fail and kill the recording.

        For LOW quality, prefers the substream:
          1) Local MediaMTX sub path  (rtsp://localhost:8554/{id}_sub) if ready
          2) Direct substream_rtsp_url
          3) Falls back to main if no sub URL is available
        """
        try:
            if camera_id in self.active_streams:
                return  # Already streaming

            want_sub = (
                config.stream_quality == StreamQuality.LOW
                and bool(config.substream_rtsp_url)
                and config.substream_capable is not False
            )

            if want_sub:
                direct_url = config.substream_rtsp_url
                mtx_path = (config.mediamtx_sub_path
                            or f"{config.mediamtx_path or camera_id}_sub")
            else:
                direct_url = config.rtsp_url
                mtx_path = camera_id

            use_local = False
            try:
                path_info = await self.mediamtx_client.get_path_info(mtx_path)
                if path_info.get("ready"):
                    use_local = True
            except Exception:
                pass

            mtx_host = self.mediamtx_client.mediamtx_host or "localhost"
            local_url = f"rtsp://{mtx_host}:8554/{mtx_path}"

            if use_local:
                stream_url = local_url
                logger.info(
                    "MediaMTX path %s ready - reading from local re-stream %s",
                    mtx_path, local_url,
                )
            else:
                stream_url = direct_url

            logger.info(f"Starting RTSP stream capture from: {stream_url} (quality={config.stream_quality.value})")

            def _open_cap_sync(url: str) -> cv2.VideoCapture:
                os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp|buffer_size;1048576|max_delay;500000|stimeout;5000000"
                cap_local = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
                try:
                    cap_local.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                except Exception:
                    pass
                if cap_local.isOpened():
                    return cap_local

                if "?" in url:
                    tcp_url = f"{url}&transport=tcp"
                else:
                    tcp_url = f"{url}?transport=tcp"
                logger.info(f"Retrying with explicit TCP: {tcp_url}")
                try:
                    cap_local.release()
                except Exception:
                    pass
                cap_local = cv2.VideoCapture(tcp_url, cv2.CAP_FFMPEG)
                try:
                    cap_local.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                except Exception:
                    pass
                return cap_local

            cap = await asyncio.to_thread(_open_cap_sync, stream_url)

            # If the local re-stream failed, fall back to the direct camera URL.
            if use_local and (not cap or not cap.isOpened()):
                logger.warning("Local re-stream failed for %s – falling back to direct camera URL", camera_id)
                try:
                    if cap:
                        cap.release()
                except Exception:
                    pass
                cap = await asyncio.to_thread(_open_cap_sync, direct_url)

            if not cap or not cap.isOpened():
                try:
                    if cap:
                        cap.release()
                except Exception:
                    pass
                raise Exception(f"Failed to open RTSP stream: {stream_url}")

            # Don't try to set resolution - accept what the camera sends
            # Setting resolution on RTSP streams can cause issues
            
            self.active_streams[camera_id] = cap

            stop_event = threading.Event()
            self._stream_stop_events[camera_id] = stop_event

            thread = threading.Thread(
                target=self._capture_frames,
                args=(camera_id, cap, stop_event),
                daemon=True,
            )
            thread.start()
            self.stream_threads[camera_id] = thread

            logger.info(f"Successfully started RTSP stream for {camera_id}")

        except Exception as e:
            logger.error(f"Failed to start RTSP stream for {camera_id}: {e}")
            raise

    async def _stop_rtsp_stream(self, camera_id: str):
        """Stop RTSP stream capture safely.

        Signals the capture thread to exit via a stop event, waits for it
        to finish, then cleans up.  The thread itself owns cap.release()
        so we never race with a concurrent cap.grab()/cap.retrieve().
        """
        try:
            stop_ev = self._stream_stop_events.pop(camera_id, None)
            if stop_ev is not None:
                stop_ev.set()

            thread = self.stream_threads.pop(camera_id, None)
            if thread is not None and thread.is_alive():
                thread.join(timeout=6)

            cap = self.active_streams.pop(camera_id, None)
            if cap is not None:
                try:
                    cap.release()
                except Exception:
                    pass
        except Exception as e:
            logger.error(f"Failed to stop RTSP stream for {camera_id}: {e}")

    @staticmethod
    def _is_local_mediamtx_url(url: str) -> bool:
        """Return True for an ``rtsp://localhost:8554/...`` style re-stream."""
        if not url:
            return False
        u = url.lower()
        return ("://localhost:8554/" in u) or ("://127.0.0.1:8554/" in u)

    async def _wait_for_mediamtx_path_ready(
        self, mtx_path: str, *, total_timeout: float = 2.0
    ) -> bool:
        """Poll MediaMTX briefly for path readiness.

        When we fast-switch to MediaMTX's local re-stream the publisher may
        still be attaching (the underlying camera RTSP pull is in progress).
        Opening the local URL too eagerly returns 404 and the switch fails
        for no good reason.  This helper polls ``/v3/paths/get`` for up to
        ``total_timeout`` seconds and returns True once the path reports
        ``ready``.
        """
        if not mtx_path:
            return False
        deadline = time.time() + max(0.0, total_timeout)
        while True:
            try:
                info = await self.mediamtx_client.get_path_info(mtx_path)
                if info and info.get("ready"):
                    return True
            except Exception:
                pass
            if time.time() >= deadline:
                return False
            await asyncio.sleep(0.1)

    @staticmethod
    def _open_capture_with_retry(
        url: str,
        *,
        max_attempts: int = 3,
    ) -> Optional[cv2.VideoCapture]:
        """Open an OpenCV capture with retries, backoff, and explicit TCP.

        FFmpeg/OpenCV's ``VideoCapture`` is finicky against IP cameras.
        Common transient failures include the camera not having released
        the prior RTSP session yet, MediaMTX not yet serving the path,
        and SDP negotiation race conditions.  Retrying with a small
        backoff fixes the vast majority of those cases.

        The first attempt uses the URL as-is; subsequent attempts force
        ``?transport=tcp`` so we never silently fall back to UDP (which
        most cameras refuse).  Setting ``OPENCV_FFMPEG_CAPTURE_OPTIONS``
        once at module import would be cleaner, but plenty of code paths
        in the codebase do it ad-hoc, so we keep it idempotent here.
        """
        if not os.environ.get("OPENCV_FFMPEG_CAPTURE_OPTIONS"):
            os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = (
                "rtsp_transport;tcp|buffer_size;1048576|"
                "max_delay;500000|stimeout;5000000|timeout;5000000"
            )

        def _force_tcp(u: str) -> str:
            if "transport=tcp" in u:
                return u
            return f"{u}{'&' if '?' in u else '?'}transport=tcp"

        last_err: Optional[str] = None
        for attempt in range(max(1, int(max_attempts))):
            try_url = url if attempt == 0 else _force_tcp(url)
            cap_local: Optional[cv2.VideoCapture] = None
            try:
                cap_local = cv2.VideoCapture(try_url, cv2.CAP_FFMPEG)
                try:
                    cap_local.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                except Exception:
                    pass
                if cap_local.isOpened():
                    if attempt > 0:
                        logger.info(
                            "Capture opened on attempt %d/%d for %s",
                            attempt + 1, max_attempts, try_url,
                        )
                    return cap_local
            except Exception as exc:
                last_err = str(exc)

            try:
                if cap_local is not None:
                    cap_local.release()
            except Exception:
                pass

            # Backoff:  0.4s, 1.0s, 1.8s ... before next attempt.  Long
            # enough for the camera to drop the prior RTSP session and
            # for MediaMTX to finish negotiating an in-flight publish.
            if attempt + 1 < max_attempts:
                time.sleep(0.4 + 0.6 * attempt)

        if last_err:
            logger.error(
                "Failed to open capture after %d attempts: %s (last err: %s)",
                max_attempts, url, last_err,
            )
        else:
            logger.error(
                "Failed to open capture after %d attempts: %s",
                max_attempts, url,
            )
        return None

    async def _switch_rtsp_stream(self, camera_id: str, new_url: str) -> bool:
        """Fast-switch the RTSP capture to a different URL without
        tearing down MediaMTX / WebRTC paths.  Stops the current
        capture thread, opens a new VideoCapture, and starts a fresh
        capture thread.

        This is the main culprit for the "sometimes works, sometimes
        doesn't" complaint: opening a new VideoCapture immediately after
        closing the previous one races against the camera's RTSP session
        release (most IP cameras hold the prior session for 200–800ms
        after the TCP socket closes and only allow 1–2 concurrent
        sessions).  We mitigate this by:

          1. Waiting briefly for the camera to release the old session.
          2. For MediaMTX local re-streams, waiting for the path to
             actually be ``ready`` so we don't open against a 404.
          3. Opening with retries + exponential backoff and an explicit
             ``transport=tcp`` retry.
        """
        logger.info(f"Fast-switching stream for {camera_id} to: {new_url}")
        await self._stop_rtsp_stream(camera_id)

        # Give the camera RTSP server time to release the prior session.
        # 300ms is short enough to feel snappy and long enough to clear
        # the most common single-session-camera contention.
        await asyncio.sleep(0.3)

        # When swapping to a MediaMTX local re-stream, wait for the path
        # to be publishing frames before we try to read it.  This costs
        # nothing when the path is already up.
        is_local = self._is_local_mediamtx_url(new_url)
        if is_local:
            try:
                mtx_path = new_url.rsplit("/", 1)[-1].split("?", 1)[0]
                ready = await self._wait_for_mediamtx_path_ready(
                    mtx_path, total_timeout=2.0
                )
                if not ready:
                    logger.info(
                        "MediaMTX path %s not ready yet; opening anyway",
                        mtx_path,
                    )
            except Exception:
                pass

        cap = await asyncio.to_thread(
            self._open_capture_with_retry, new_url, max_attempts=3
        )
        if not cap or not cap.isOpened():
            try:
                if cap:
                    cap.release()
            except Exception:
                pass
            logger.error(
                "Fast-switch failed to open stream after retries: %s",
                new_url,
            )
            return False

        self.active_streams[camera_id] = cap

        stop_event = threading.Event()
        self._stream_stop_events[camera_id] = stop_event

        thread = threading.Thread(
            target=self._capture_frames,
            args=(camera_id, cap, stop_event),
            daemon=True,
        )
        thread.start()
        self.stream_threads[camera_id] = thread
        logger.info(f"Fast-switch complete for {camera_id}")
        return True

    def _capture_frames(self, camera_id: str, cap: cv2.VideoCapture,
                        stop_event: threading.Event | None = None):
        """Frame capture thread with drift compensation and auto-recovery"""
        health = self.camera_health[camera_id]
        frame_count = 0
        total_frames = 0
        last_fps_calc_time = time.time()
        consecutive_failures = 0
        max_consecutive_failures = 10

        last_read_time = time.time()
        watchdog_timeout = 10.0

        logger.info(f"Started capture loop for {camera_id}")
        health.status = CameraStatus.CONNECTED

        _stopped = stop_event.is_set if stop_event else (lambda: False)

        while self._running and camera_id in self.active_streams and not _stopped():
            try:
                current_time = time.time()
                
                # Watchdog check: If we haven't read a frame in watchdog_timeout seconds
                if current_time - last_read_time > watchdog_timeout:
                    logger.warning(f"Stream watchdog: No frames for {watchdog_timeout}s from {camera_id}")
                    health.status = CameraStatus.ERROR
                    health.last_error = "Stream stalled (watchdog)"
                    break

                # Grab frame (non-blocking check if available)
                ret = cap.grab()
                
                if not ret:
                    consecutive_failures += 1
                    if consecutive_failures >= max_consecutive_failures:
                        logger.warning(f"Too many consecutive failures ({consecutive_failures}) for camera {camera_id}")
                        health.status = CameraStatus.ERROR
                        health.last_error = f"Consecutive read failures: {consecutive_failures}"
                        break
                    time.sleep(0.01)  # Brief pause before retry
                    continue
                
                # Retrieve the frame
                ret, frame = cap.retrieve()
                if not ret or frame is None:
                    consecutive_failures += 1
                    continue
                
                # Success - reset failure counter
                consecutive_failures = 0
                last_read_time = time.time()
                frame_count += 1
                total_frames += 1

                # Update health metrics
                health.last_frame_time = datetime.now()
                health.total_frames = total_frames
                health.status = CameraStatus.CONNECTED

                # Calculate frame rate every second
                elapsed = current_time - last_fps_calc_time
                if elapsed >= 1.0:
                    health.frame_rate = frame_count / elapsed
                    frame_count = 0
                    last_fps_calc_time = current_time

                # Call frame callback (non-blocking)
                if self.on_frame_received:
                    try:
                        if asyncio.iscoroutinefunction(self.on_frame_received):
                            if hasattr(self, '_loop') and self._loop and self._loop.is_running():
                                asyncio.run_coroutine_threadsafe(
                                    self.on_frame_received(camera_id, frame),
                                    self._loop
                                )
                        else:
                            # Synchronous callback - call directly
                            self.on_frame_received(camera_id, frame)
                    except Exception as e:
                        logger.error(f"Error in frame callback: {e}")

            except Exception as e:
                logger.error(f"Error capturing frame from camera {camera_id}: {e}")
                health.error_count += 1
                consecutive_failures += 1
                if consecutive_failures >= max_consecutive_failures:
                    health.status = CameraStatus.ERROR
                    health.last_error = str(e)
                    break
                time.sleep(0.1)  # Brief pause before retry
        
        logger.info(f"Capture loop ended for {camera_id}")
        # If the loop exited on its own (watchdog / failures), clean up.
        # If _stop_rtsp_stream requested the stop it will handle release.
        if stop_event is None or not stop_event.is_set():
            try:
                cap.release()
            except Exception:
                pass
            self.active_streams.pop(camera_id, None)

    async def _detect_ptz_capabilities(self, camera_id: str, config: CameraConfig):
        """Detect PTZ capabilities for camera"""
        try:
            if not config.ptz_enabled or not config.ptz_url:
                return

            # Default PTZ capabilities - could be enhanced with actual detection
            capabilities = PTZCapabilities(
                pan_range=(-180.0, 180.0),
                tilt_range=(-90.0, 90.0),
                zoom_range=(1.0, 10.0),
                presets=[],
                auto_tracking=False,
                tour_support=False
            )

            self.ptz_capabilities[camera_id] = capabilities
            logger.info(f"PTZ capabilities detected for camera {camera_id}")

        except Exception as e:
            logger.error(f"Failed to detect PTZ capabilities for {camera_id}: {e}")

    def _monitor_cameras(self):
        """Background camera monitoring thread"""
        while self._running:
            try:
                current_time = datetime.now()

                for camera_id, health in self.camera_health.items():
                    config = self.cameras.get(camera_id)
                    if not config or not config.enabled:
                        continue

                    # Check for stale frames
                    if health.last_frame_time:
                        time_since_frame = current_time - health.last_frame_time
                        if time_since_frame > timedelta(seconds=30):
                            logger.warning(f"Camera {camera_id} appears stale, attempting reconnection")
                            # We're in a background thread; schedule the coroutine on the main loop.
                            try:
                                if hasattr(self, "_loop") and self._loop and self._loop.is_running():
                                    asyncio.run_coroutine_threadsafe(self.reconnect_camera(camera_id), self._loop)
                                else:
                                    logger.error("Cannot reconnect %s: no running event loop", camera_id)
                            except Exception as e:
                                logger.error("Failed scheduling reconnect for %s: %s", camera_id, e)

                time.sleep(30)  # Check every 30 seconds

            except Exception as e:
                logger.error(f"Error in camera monitoring: {e}")
                time.sleep(60)

    def _update_health_metrics(self):
        """Update health metrics for all cameras"""
        while self._running:
            try:
                current_time = datetime.now()

                for camera_id, health in self.camera_health.items():
                    if camera_id in self.webrtc_streams:
                        health.webrtc_connected = self.webrtc_streams[camera_id]

                    # Update connection uptime
                    if health.status == CameraStatus.CONNECTED and health.last_frame_time:
                        health.connection_uptime = current_time - health.last_frame_time

                time.sleep(10)  # Update every 10 seconds

            except Exception as e:
                logger.error(f"Error updating health metrics: {e}")
                time.sleep(30)

    # MediaMTX Event Callbacks

    async def _on_webrtc_connected(self, stream_path: str):
        """Handle WebRTC stream connected event"""
        camera_id = stream_path  # stream_path is now camera_id
        if camera_id in self.camera_health:
            self.camera_health[camera_id].webrtc_connected = True
            logger.info(f"WebRTC stream connected for camera: {camera_id}")

    async def _on_webrtc_disconnected(self, stream_path: str):
        """Handle WebRTC stream disconnected event"""
        camera_id = stream_path  # stream_path is now camera_id
        if camera_id in self.camera_health:
            self.camera_health[camera_id].webrtc_connected = False
            logger.info(f"WebRTC stream disconnected for camera: {camera_id}")

    async def _on_webrtc_error(self, stream_path: str, error: str):
        """Handle WebRTC stream error event"""
        camera_id = stream_path  # stream_path is now camera_id
        if camera_id in self.camera_health:
            self.camera_health[camera_id].webrtc_connected = False
            logger.error(f"WebRTC error for camera {camera_id}: {error}")

            if self.on_camera_error:
                await self._safe_callback(self.on_camera_error, camera_id, error)

    async def _safe_callback(self, callback: Callable, *args, **kwargs):
        """Safely execute callback without breaking the main flow"""
        try:
            if asyncio.iscoroutinefunction(callback):
                await callback(*args, **kwargs)
            else:
                callback(*args, **kwargs)
        except Exception as e:
            logger.error(f"Error in callback: {e}")

    # Public Utility Methods

    def get_snapshot(self, camera_id: str) -> Optional[bytes]:
        """
        Get a snapshot from a camera by capturing from the video stream
        
        Args:
            camera_id: Camera ID
            
        Returns:
            JPEG image bytes or None if failed
        """
        try:
            import cv2
            import numpy as np
            from PIL import Image, ImageDraw, ImageFont
            import io
            
            # First try to get the camera info to find the stream URL
            camera = self.get_camera(camera_id)
            if not camera:
                logger.warning(f"Camera {camera_id} not found")
                return None
            
            # Try to capture from RTSP stream
            if camera.rtsp_url:
                try:
                    cap = cv2.VideoCapture(camera.rtsp_url)
                    if cap.isOpened():
                        ret, frame = cap.read()
                        cap.release()
                        
                        if ret:
                            # Convert BGR to RGB
                            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                            
                            # Convert to PIL Image
                            pil_image = Image.fromarray(frame_rgb)
                            
                            # Add camera info overlay
                            draw = ImageDraw.Draw(pil_image)
                            try:
                                # Try to use a default font
                                font = ImageFont.load_default()
                            except:
                                font = None
                            
                            # Add camera name and timestamp
                            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                            text = f"{camera.name} - {timestamp}"
                            
                            # Draw background rectangle for text
                            bbox = draw.textbbox((10, 10), text, font=font)
                            draw.rectangle([bbox[0]-5, bbox[1]-5, bbox[2]+5, bbox[3]+5], fill='rgba(0,0,0,0.7)')
                            draw.text((10, 10), text, fill='white', font=font)
                            
                            # Convert to JPEG bytes
                            img_io = io.BytesIO()
                            pil_image.save(img_io, 'JPEG', quality=85)
                            img_io.seek(0)
                            
                            logger.info(f"Successfully captured snapshot from camera {camera_id} via RTSP")
                            return img_io.getvalue()
                            
                except Exception as e:
                    logger.warning(f"Failed to capture from RTSP for camera {camera_id}: {e}")
            
            # Fallback: Create a status image showing camera info
            width, height = 800, 600
            image = Image.new('RGB', (width, height), color='#2a2a2a')
            draw = ImageDraw.Draw(image)
            
            try:
                font = ImageFont.load_default()
            except:
                font = None
            
            # Add camera info
            draw.rectangle([10, 10, width-10, 80], fill='#1a1a1a', outline='#444')
            draw.text((20, 20), f"Camera: {camera.name}", fill='#fff', font=font)
            draw.text((20, 40), f"ID: {camera_id}", fill='#aaa', font=font)
            draw.text((20, 60), f"Status: {camera.status if hasattr(camera, 'status') else 'Unknown'}", fill='#0f0', font=font)
            
            # Add connection info
            health = self.camera_health.get(camera_id)
            if health:
                draw.text((20, 100), f"Frame Rate: {health.frame_rate:.1f} fps", fill='#fff', font=font)
                draw.text((20, 120), f"Connection: {'Connected' if health.status == CameraStatus.CONNECTED else 'Disconnected'}", fill='#fff', font=font)
                if health.last_frame_time:
                    draw.text((20, 140), f"Last Frame: {health.last_frame_time.strftime('%H:%M:%S')}", fill='#fff', font=font)
            
            # Add timestamp
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            draw.text((20, height-30), f"Snapshot: {timestamp}", fill='#888', font=font)
            
            # Convert to bytes
            img_io = io.BytesIO()
            image.save(img_io, 'JPEG', quality=85)
            img_io.seek(0)
            
            logger.info(f"Generated status snapshot for camera {camera_id}")
            return img_io.getvalue()
                
        except Exception as e:
            logger.error(f"Error getting snapshot for camera {camera_id}: {e}")
            return None

    def send_command(self, camera_id: str, command: str, params: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Send command to camera (synchronous wrapper for PTZ commands)

        Args:
            camera_id: Camera ID
            command: Command to send
            params: Command parameters

        Returns:
            Command result
        """
        try:
            # PTZ commands are routed via core.ptz_manager + /api/cameras/<id>/ptz.
            # Anything starting with 'ptz_' arriving here is from a deprecated path.
            if command.startswith('ptz_'):
                logger.warning(
                    "Legacy PTZ command '%s' received via send_command; "
                    "use the /api/cameras/<id>/ptz endpoint or core.ptz_manager directly.",
                    command,
                )
                return {
                    "success": False,
                    "command": command,
                    "error": "PTZ commands are no longer handled here; use the PTZ HTTP route.",
                }

            # Handle other commands
            elif command == "reconnect":
                # Prefer scheduling on the existing asyncio loop when available.
                try:
                    if hasattr(self, "_loop") and self._loop and self._loop.is_running():
                        fut = asyncio.run_coroutine_threadsafe(self.reconnect_camera(camera_id), self._loop)
                        return {"success": bool(fut.result(timeout=30)), "command": command}
                except Exception:
                    pass
                # Fallback: run in a temporary loop (best effort).
                result = asyncio.run(self.reconnect_camera(camera_id))
                return {"success": result, "command": command}

            elif command == "set_quality":
                quality = StreamQuality(params.get("quality", "medium"))
                try:
                    if hasattr(self, "_loop") and self._loop and self._loop.is_running():
                        fut = asyncio.run_coroutine_threadsafe(
                            self.set_stream_quality(camera_id, quality), self._loop
                        )
                        return {"success": bool(fut.result(timeout=30)), "command": command}
                except Exception:
                    pass
                result = asyncio.run(self.set_stream_quality(camera_id, quality))
                return {"success": result, "command": command}

            else:
                logger.warning(f"Unknown command: {command}")
                return {"success": False, "error": "Unknown command"}

        except Exception as e:
            logger.error(f"Error executing command {command} for camera {camera_id}: {e}")
            return {"success": False, "error": str(e)}