import asyncio
import json
import logging
import threading
import time
import queue
import cv2
import numpy as np
import websockets
import aiohttp
from typing import Dict, Any, Optional, Set, List, Callable, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from enum import Enum
import base64
import uuid
from urllib.parse import urlparse
import socket
from concurrent.futures import ThreadPoolExecutor
import weakref
import os
import re

from .mediamtx_client import MediaMTXWebRTCClient, StreamConfig
from .motion import MotionDetector, MotionResult, SimpleMotionDetector

from .snapshots import build_track_composite
import heapq

logger = logging.getLogger(__name__)
class SimpleCentroidTracker:
    """Very fast centroid tracker: stable IDs for moving blobs with minimal latency."""
    def __init__(self, max_distance: float = 60.0, max_age: int = 12):
        self.max_distance = float(max_distance)
        self.max_age = int(max_age)
        self._next_id = 1
        # id -> {x, y, age}
        self._tracks: Dict[int, Dict[str, float]] = {}

    def _distance(self, a: Tuple[float, float], b: Tuple[float, float]) -> float:
        dx = a[0] - b[0]
        dy = a[1] - b[1]
        return float((dx*dx + dy*dy) ** 0.5)

    def update(self, centroids: List[Tuple[float, float]]) -> List[Dict[str, float]]:
        # Age existing tracks
        for tid in list(self._tracks.keys()):
            self._tracks[tid]['age'] = self._tracks[tid].get('age', 0) + 1

        # Greedy nearest-neighbor assignment
        unmatched = set(range(len(centroids)))
        # Build cost matrix (id -> list of (dist, idx))
        for tid, t in list(self._tracks.items()):
            best_idx = None
            best_dist = 1e9
            for i in unmatched:
                d = self._distance((t['x'], t['y']), centroids[i])
                if d < best_dist:
                    best_dist = d
                    best_idx = i
            if best_idx is not None and best_dist <= self.max_distance:
                cx, cy = centroids[best_idx]
                self._tracks[tid]['x'] = float(cx)
                self._tracks[tid]['y'] = float(cy)
                self._tracks[tid]['age'] = 0
                unmatched.discard(best_idx)

        # Create new tracks for unmatched detections
        for i in unmatched:
            cx, cy = centroids[i]
            tid = self._next_id
            self._next_id += 1
            self._tracks[tid] = {'x': float(cx), 'y': float(cy), 'age': 0}

        # Remove stale tracks
        for tid in list(self._tracks.keys()):
            if self._tracks[tid].get('age', 0) > self.max_age:
                del self._tracks[tid]

        # Return list of points
        return [
            {'id': int(tid), 'x': float(t['x']), 'y': float(t['y'])}
            for tid, t in self._tracks.items()
        ]








class ConnectionType(Enum):
    """Client connection types"""
    WEBRTC = "webrtc"
    WEBSOCKET = "websocket"
    HTTP = "http"
    MJPEG = "mjpeg"


class StreamQuality(Enum):
    """Stream quality levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    ULTRA = "ultra"


@dataclass
class ClientConnection:
    """Client connection information"""
    id: str
    type: ConnectionType
    camera_id: str
    quality: StreamQuality
    connected_at: datetime
    last_activity: datetime
    bytes_sent: int = 0
    frames_sent: int = 0
    bandwidth_limit: Optional[int] = None  # bytes per second
    websocket: Optional[websockets.WebSocketServerProtocol] = None
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    message_queue: List[Tuple[int, Any]] = None  # Priority queue for messages
    last_message_time: float = 0.0  # For rate limiting

    def __post_init__(self):
        if self.message_queue is None:
            self.message_queue = []


@dataclass
class StreamStats:
    """Stream statistics"""
    camera_id: str
    active_clients: int = 0
    total_clients_served: int = 0
    frames_processed: int = 0
    bytes_transmitted: int = 0
    average_fps: float = 0.0
    current_bitrate: float = 0.0  # Mbps
    webrtc_connections: int = 0
    websocket_connections: int = 0
    http_connections: int = 0
    last_updated: datetime = None

    def __post_init__(self):
        if self.last_updated is None:
            self.last_updated = datetime.now()


@dataclass
class BandwidthManager:
    """Bandwidth management configuration"""
    max_total_bandwidth: int = 100 * 1024 * 1024  # 100 Mbps in bytes/sec
    max_client_bandwidth: int = 10 * 1024 * 1024  # 10 Mbps per client
    adaptive_quality: bool = True
    quality_thresholds: Dict[str, int] = None

    def __post_init__(self):
        if self.quality_thresholds is None:
            self.quality_thresholds = {
                "ultra": 8 * 1024 * 1024,  # 8 Mbps
                "high": 4 * 1024 * 1024,  # 4 Mbps
                "medium": 2 * 1024 * 1024,  # 2 Mbps
                "low": 500 * 1024  # 500 Kbps
            }


class StreamServer:
    """
    Enhanced stream server with WebRTC capabilities and MediaMTX integration
    """

    def __init__(self,
                 mediamtx_host: str = "localhost",
                 mediamtx_webrtc_port: int = 8889,
                 mediamtx_api_port: int = 9997,
                 websocket_port: int = 8765,
                 http_port: int = 8080,
                 ai_agent=None):
        """
        Initialize Stream Server

        Args:
            mediamtx_host: MediaMTX server host
            mediamtx_webrtc_port: MediaMTX WebRTC port
            mediamtx_api_port: MediaMTX API port
            websocket_port: WebSocket signaling port
            http_port: HTTP streaming port
        """
        # MediaMTX integration
        self.mediamtx_client = MediaMTXWebRTCClient(
            mediamtx_host=mediamtx_host,
            mediamtx_webrtc_port=mediamtx_webrtc_port,
            mediamtx_api_port=mediamtx_api_port
        )

        # Server configuration
        self.websocket_port = websocket_port
        self.http_port = http_port
        self.running = False
        
        # AI Agent for adaptive learning
        self.ai_agent = ai_agent

        # Stream management
        self.active_streams: Dict[str, Dict[str, Any]] = {}
        self._starting_streams: Set[str] = set()
        self._starting_lock = threading.Lock()
        self.client_connections: Dict[str, ClientConnection] = {}
        self.stream_stats: Dict[str, StreamStats] = {}
        self.signaling_server: Optional[websockets.WebSocketServer] = None

        # Enhanced motion detection per camera with AI learning
        self._motion_detectors: Dict[str, SimpleMotionDetector] = {}
        self._motion_state: Dict[str, MotionResult] = {}
        # Initialize additional state containers referenced by helper methods
        self._motion_params: Dict[str, Dict[str, float]] = {}
        # Box smoothing state per camera and track id
        self._bbox_ema: Dict[str, Dict[int, Dict[str, float]]] = {}
        # Detector bbox smoothing state per camera and track id (for SORT tracks)
        self._detector_bbox_ema: Dict[str, Dict[int, Dict[str, float]]] = {}
        self._tracks_state: Dict[str, List[Dict[str, Any]]] = {}
        # SORT-style tracker instances per camera (lazy init)
        self._sort_trackers: Dict[str, Any] = {}
        # Dedicated detection worker threads per camera
        self._detector_threads: Dict[str, threading.Thread] = {}
        # In-flight guard and worker pacing
        self._detector_inflight: Dict[str, bool] = {}
        # Camera movement suppression window (epoch seconds until which overlays are suppressed)
        self._camera_move_until: Dict[str, float] = {}
        
        # NEW: Detector manager for unified detection/tracking
        # LAZY LOAD - Only initialize when first detection is requested
        self._detector_manager = None
        self._detector_manager_initialized = False
        logger.info("✓ Detector manager will be lazy-loaded on first detection request")
        self._track_prev_ids: Dict[str, Set[int]] = {}
        self._track_events: Dict[str, List[Dict[str, Any]]] = {}
        self._track_trajectories: Dict[str, Dict[int, Dict[str, Any]]] = {}
        self._detection_logs: List[Dict[str, Any]] = []

        # Per-camera detector configuration and rate limiting for tier-2 verification
        self._detector_config: Dict[str, Dict[str, Any]] = {}
        # Per-camera detection enabled state - OFF by default until user enables
        self._detection_enabled: Dict[str, bool] = {}
        # Cache of last-known verification_enabled flag per camera to avoid redundant detector work
        self._verification_state: Dict[str, bool] = {}
        # Per-camera shapes (zones/lines/tags) for ROI-based detection
        self._camera_shapes: Dict[str, Dict[str, List[Any]]] = {}
        self._last_verification_time: Dict[str, float] = {}

        # Bandwidth management
        self.bandwidth_manager = BandwidthManager()
        
        # Connection rate limiting
        self.max_concurrent_recoveries = 2  # Max simultaneous recovery attempts
        self.active_recoveries: Set[str] = set()
        self.recovery_lock = threading.Lock()
        self.current_bandwidth_usage = 0

        # Quality settings
        self.quality_settings = {
            StreamQuality.LOW: {
                "width": 640, "height": 480, "fps": 15, "bitrate": 500000,
                "jpeg_quality": 60, "h264_crf": 28
            },
            StreamQuality.MEDIUM: {
                "width": 1280, "height": 720, "fps": 25, "bitrate": 2000000,
                "jpeg_quality": 75, "h264_crf": 23
            },
            StreamQuality.HIGH: {
                "width": 1920, "height": 1080, "fps": 30, "bitrate": 4000000,
                "jpeg_quality": 85, "h264_crf": 18
            },
            StreamQuality.ULTRA: {
                "width": 3840, "height": 2160, "fps": 30, "bitrate": 8000000,
                "jpeg_quality": 95, "h264_crf": 15
            }
        }

        # Event callbacks
        self.on_client_connected: Optional[Callable] = None
        self.on_client_disconnected: Optional[Callable] = None
        self.on_stream_started: Optional[Callable] = None
        self.on_stream_stopped: Optional[Callable] = None
        self.on_bandwidth_exceeded: Optional[Callable] = None
        # Simple motion detection callback
        self.on_motion_update: Optional[Callable[[str, Dict[str, Any]], None]] = None

        # Background tasks
        self.executor = ThreadPoolExecutor(max_workers=20)
        self._stats_task: Optional[asyncio.Task] = None
        self._bandwidth_task: Optional[asyncio.Task] = None
        self._cleanup_task: Optional[asyncio.Task] = None

    async def start(self):
        """Start the stream server and all services"""
        if self.running:
            return

        self.running = True
        # Store the current event loop for thread-safe operations
        self._loop = asyncio.get_running_loop()
        logger.info("Starting Enhanced Stream Server with WebRTC support")

        # Start MediaMTX client
        await self.mediamtx_client.start()

        # Start WebSocket signaling server
        await self._start_signaling_server()

        # Start background tasks
        self._stats_task = asyncio.create_task(self._update_statistics())
        self._bandwidth_task = asyncio.create_task(self._manage_bandwidth())
        self._cleanup_task = asyncio.create_task(self._cleanup_stale_connections())
        self._websocket_check_task = asyncio.create_task(self._websocket_connection_monitor())
        self._message_queue_task = asyncio.create_task(self._process_message_queues())

        logger.info(f"Stream Server started - SignalingWS:{self.websocket_port} HTTP:{self.http_port}")

    async def stop(self):
        """Stop the stream server and cleanup resources"""
        if not self.running:
            return

        logger.info("Stopping Stream Server")
        self.running = False

        # Stop signaling server
        if self.signaling_server:
            self.signaling_server.close()
            await self.signaling_server.wait_closed()

        # Cancel background tasks
        for task in [self._stats_task, self._bandwidth_task, self._cleanup_task, self._websocket_check_task, self._message_queue_task]:
            if task:
                task.cancel()

        # Disconnect all clients
        for client_id in list(self.client_connections.keys()):
            await self.disconnect_client(client_id)

        # Stop all streams
        for camera_id in list(self.active_streams.keys()):
            await self.stop_stream(camera_id)

        # Stop MediaMTX client
        await self.mediamtx_client.stop()

        # Shutdown executor
        self.executor.shutdown(wait=True)

        logger.info("Stream Server stopped")

    # Stream Management

    def _stream_worker(self, camera_id: str, camera_config: Dict[str, Any]) -> None:
        """Background worker to connect and start stream"""
        try:
            logger.info(f"Starting stream worker for camera {camera_id}")
            
            stream_url = camera_config.get('rtsp_url') or camera_config.get('stream_url')
            motion_enabled = bool(camera_config.get('motion_detection', False))
            
            # Simple motion detection setup - use direct camera RTSP
            capture_url = stream_url
            
            logger.info(
                "🎯 Connecting stream for %s (motion_detection=%s): %s",
                camera_id,
                motion_enabled,
                capture_url,
            )
            
            # Connection logic
            # Force TCP transport to avoid H.264 artifacts/UDP packet loss
            if '?' in capture_url:
                if 'rtsp_transport' not in capture_url:
                    capture_url += "&rtsp_transport=tcp"
            else:
                capture_url += "?rtsp_transport=tcp"
            
            # Set environment variable for OpenCV backends that check it
            os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"
            
            cap = cv2.VideoCapture(capture_url, cv2.CAP_FFMPEG)
            try:
                cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 5000)
                cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, 3000)
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            except Exception:
                pass

            if not cap.isOpened():
                # Fallback logic - retry with UDP if TCP fails (rare but possible)
                try: cap.release()
                except: pass
                
                # Try original URL (likely UDP)
                fallback_url = stream_url
                logger.warning(f"Primary TCP open failed for {camera_id}. Retrying with default transport: {fallback_url}")
                
                # Reset env var
                if "OPENCV_FFMPEG_CAPTURE_OPTIONS" in os.environ:
                    del os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"]
                    
                cap = cv2.VideoCapture(fallback_url, cv2.CAP_FFMPEG)
                try:
                    cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 5000)
                    cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, 3000)
                    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                except: pass
            
            if not cap.isOpened():
                # Final fallback
                try: cap.release()
                except: pass
                logger.warning(f"Retrying with CAP_ANY for {camera_id}")
                cap = cv2.VideoCapture(capture_url)
            
            if not cap.isOpened():
                logger.error(f"Failed to open RTSP stream for {camera_id}")
                try: cap.release()
                except: pass
                
                # Cleanup
                with self._starting_lock:
                    self._starting_streams.discard(camera_id)
                if camera_id in self.active_streams:
                    del self.active_streams[camera_id]
                if camera_id in self.stream_stats:
                    del self.stream_stats[camera_id]
                return

            logger.info(f"✅ Motion detection stream connected for {camera_id}")

            # Update stream info with active capture
            if camera_id in self.active_streams:
                stream_info = self.active_streams[camera_id]
                stream_info['capture'] = cap
                stream_info['active'] = True
                stream_info['capture_url'] = capture_url
                stream_info['motion_detection_enabled'] = motion_enabled
                
                # Setup detectors if enabled
                enable_learning = True
                
                # Ensure detection is enabled in the global map for the worker to pick it up
                if motion_enabled:
                    self._detection_enabled[camera_id] = True
                    logger.info(f"Enabled detection worker flag for {camera_id}")
                    self._motion_detectors[camera_id] = SimpleMotionDetector(
                        camera_id=camera_id,
                        ai_agent=self.ai_agent,
                        enable_learning=enable_learning
                    )
                else:
                    self._detection_enabled[camera_id] = False
                    self._motion_detectors.pop(camera_id, None)

                # Start frame capture loop (in this thread)
                # We use this thread for capture instead of spawning another one
                with self._starting_lock:
                    self._starting_streams.discard(camera_id)
                
                logger.info(f"Stream fully active for {camera_id}")
                self._capture_frames(camera_id)
                
            else:
                # Stream was stopped while connecting
                cap.release()
                with self._starting_lock:
                    self._starting_streams.discard(camera_id)

        except Exception as e:
            logger.error(f"Error in stream worker for {camera_id}: {e}")
            with self._starting_lock:
                self._starting_streams.discard(camera_id)
            if camera_id in self.active_streams:
                del self.active_streams[camera_id]

    def _start_stream_core(self, camera_id: str, camera_config: Dict[str, Any]) -> bool:
        """Core logic for starting a stream (synchronous)"""
        try:
            if not self.running:
                logger.info("Enabling StreamServer runtime for motion capture (light mode)")
                self.running = True
                
            if camera_id in self.active_streams:
                return True

            # Check if currently starting to prevent race conditions
            with self._starting_lock:
                if camera_id in self._starting_streams:
                    return True
                self._starting_streams.add(camera_id)

            stream_url = camera_config.get('rtsp_url') or camera_config.get('stream_url')
            if not stream_url:
                logger.error(f"No stream URL configured for camera {camera_id}")
                with self._starting_lock:
                    self._starting_streams.discard(camera_id)
                return False

            logger.info(f"Stream start requested for {camera_id}")

            # Initialize placeholder info so 'active_streams' check passes
            self.active_streams[camera_id] = {
                'camera_id': camera_id,
                'capture': None, 
                'config': camera_config,
                'active': False, # Will be set to True when connection succeeds
                'clients': set(),
                'frame_queue': queue.Queue(maxsize=3),
                'last_frame': None,
                'fps': 15,
                'quality': StreamQuality.MEDIUM,
                'stats': StreamStats(camera_id=camera_id),
                'capture_url': stream_url,
                'motion_detection_enabled': bool(camera_config.get('motion_detection', False)),
            }
            self.stream_stats[camera_id] = self.active_streams[camera_id]['stats']

            # Inline logic from _stream_worker to simplify and fix capture init
            # Initialize capture with optimized settings and TCP preference
            capture_url = stream_url
            # Force TCP transport for reliability if not specified
            if "rtsp_transport" not in capture_url and "rtsp://" in capture_url:
                sep = "&" if "?" in capture_url else "?"
                capture_url = f"{capture_url}{sep}rtsp_transport=tcp"
            
            # Set FFMPEG flags for low latency
            # os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp" # Global env var is risky, stick to URL params
            
            cap = cv2.VideoCapture(capture_url, cv2.CAP_FFMPEG)
            
            # Tune buffer settings
            try:
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            except Exception:
                pass
            
            if not cap.isOpened():
                # Retry with UDP if TCP fails
                logger.warning(f"Failed to open stream {camera_id} with TCP, retrying UDP...")
                if "rtsp_transport=tcp" in capture_url:
                    capture_url = capture_url.replace("rtsp_transport=tcp", "rtsp_transport=udp")
                else:
                    # If it wasn't added, it might be non-RTSP, just retry as is
                    pass
                cap = cv2.VideoCapture(capture_url, cv2.CAP_FFMPEG)
                
            if not cap.isOpened():
                logger.error(f"Failed to open stream capture for {camera_id}")
                with self._starting_lock:
                    self._starting_streams.discard(camera_id)
                if camera_id in self.active_streams:
                    del self.active_streams[camera_id]
                return False

            logger.info(f"Stream capture started successfully for {camera_id}")
            self.active_streams[camera_id]['capture'] = cap
            self.active_streams[camera_id]['active'] = True
            
            # Start frame capture loop
            threading.Thread(
                target=self._capture_frames,
                args=(camera_id,),
                daemon=True
            ).start()

            # If WebRTC is enabled, we need to connect to MediaMTX too - handled separately if needed
            # But for object detection backend, we just need frames.
            
            return True

        except Exception as e:
            logger.error(f"Error starting stream for camera {camera_id}: {e}")
            with self._starting_lock:
                self._starting_streams.discard(camera_id)
            if camera_id in self.active_streams:
                del self.active_streams[camera_id]
            return False

    async def start_stream(self, camera_id: str, camera_config: Dict[str, Any]) -> bool:
        """
        Start video stream for a camera with WebRTC support
        Returns True immediately to indicate request accepted (async start)
        """
        if self._start_stream_core(camera_id, camera_config):
            if self.on_stream_started:
                await self._safe_callback(self.on_stream_started, camera_id)
            return True
        return False

    def start_stream_sync(self, camera_id: str, camera_config: Dict[str, Any]) -> bool:
        """Synchronous version of start_stream for non-async contexts"""
        return self._start_stream_core(camera_id, camera_config)

    async def stop_stream(self, camera_id: str) -> bool:
        """
        Stop video stream for a camera

        Args:
            camera_id: Camera identifier

        Returns:
            True if stream stopped successfully
        """
        try:
            if camera_id not in self.active_streams:
                return True

            logger.info(f"Stopping stream for camera {camera_id}")

            stream_info = self.active_streams[camera_id]
            stream_info['active'] = False

            # Disconnect all clients
            clients_to_disconnect = list(stream_info['clients'])
            for client_id in clients_to_disconnect:
                await self.disconnect_client(client_id)

            # Stop WebRTC stream
            if stream_info.get('webrtc_enabled') and stream_info.get('stream_path'):
                await self.mediamtx_client.disconnect_stream(stream_info['stream_path'])

            # Release capture
            if 'capture' in stream_info:
                stream_info['capture'].release()

            # Cleanup
            del self.active_streams[camera_id]
            if camera_id in self.stream_stats:
                del self.stream_stats[camera_id]

            # Cleanup motion detector/state
            if camera_id in self._motion_detectors:
                del self._motion_detectors[camera_id]
            if camera_id in self._motion_state:
                del self._motion_state[camera_id]

            if self.on_stream_stopped:
                await self._safe_callback(self.on_stream_stopped, camera_id)

            logger.info(f"Stopped stream for camera {camera_id}")
            return True

        except Exception as e:
            logger.error(f"Error stopping stream for camera {camera_id}: {e}")
            return False

    # Client Management

    async def connect_client(self, camera_id: str, connection_type: ConnectionType,
                             quality: StreamQuality = StreamQuality.MEDIUM,
                             websocket: Optional[websockets.WebSocketServerProtocol] = None,
                             client_info: Optional[Dict[str, Any]] = None) -> Optional[str]:
        """
        Connect a client to a camera stream

        Args:
            camera_id: Camera to connect to
            connection_type: Type of connection
            quality: Stream quality
            websocket: WebSocket connection if applicable
            client_info: Additional client information

        Returns:
            Client ID if connection successful
        """
        try:
            if camera_id not in self.active_streams:
                logger.error(f"Camera stream {camera_id} not active")
                return None

            # Check bandwidth limits
            if not await self._check_bandwidth_availability(quality):
                logger.warning(f"Bandwidth limit exceeded for new client")
                return None

            # Generate client ID
            client_id = str(uuid.uuid4())

            # Create client connection
            client_info = client_info or {}
            connection = ClientConnection(
                id=client_id,
                type=connection_type,
                camera_id=camera_id,
                quality=quality,
                connected_at=datetime.now(),
                last_activity=datetime.now(),
                websocket=websocket,
                ip_address=client_info.get('ip_address'),
                user_agent=client_info.get('user_agent'),
                bandwidth_limit=self._get_client_bandwidth_limit(quality)
            )

            self.client_connections[client_id] = connection
            self.active_streams[camera_id]['clients'].add(client_id)

            # Update statistics
            stats = self.stream_stats[camera_id]
            stats.active_clients += 1
            stats.total_clients_served += 1

            if connection_type == ConnectionType.WEBRTC:
                stats.webrtc_connections += 1
            elif connection_type == ConnectionType.WEBSOCKET:
                stats.websocket_connections += 1
            elif connection_type == ConnectionType.HTTP:
                stats.http_connections += 1

            if self.on_client_connected:
                await self._safe_callback(self.on_client_connected, client_id, camera_id)

            logger.info(f"Connected client {client_id} to camera {camera_id} ({connection_type.value})")
            return client_id

        except Exception as e:
            logger.error(f"Error connecting client to camera {camera_id}: {e}")
            return None

    async def disconnect_client(self, client_id: str) -> bool:
        """
        Disconnect a client

        Args:
            client_id: Client to disconnect

        Returns:
            True if disconnection successful
        """
        try:
            connection = self.client_connections.get(client_id)
            if not connection:
                return True

            camera_id = connection.camera_id

            # Remove from stream clients
            if camera_id in self.active_streams:
                self.active_streams[camera_id]['clients'].discard(client_id)

            # Close WebSocket if applicable
            if connection.websocket:
                try:
                    await connection.websocket.close()
                except:
                    pass

            # Update statistics
            if camera_id in self.stream_stats:
                stats = self.stream_stats[camera_id]
                stats.active_clients = max(0, stats.active_clients - 1)

                if connection.type == ConnectionType.WEBRTC:
                    stats.webrtc_connections = max(0, stats.webrtc_connections - 1)
                elif connection.type == ConnectionType.WEBSOCKET:
                    stats.websocket_connections = max(0, stats.websocket_connections - 1)
                elif connection.type == ConnectionType.HTTP:
                    stats.http_connections = max(0, stats.http_connections - 1)

            # Remove connection
            del self.client_connections[client_id]

            if self.on_client_disconnected:
                await self._safe_callback(self.on_client_disconnected, client_id, camera_id)

            logger.info(f"Disconnected client {client_id} from camera {camera_id}")
            return True

        except Exception as e:
            logger.error(f"Error disconnecting client {client_id}: {e}")
            return False

    # WebRTC Signaling

    async def _start_signaling_server(self):
        """Start WebSocket signaling server for WebRTC"""
        try:
            self.signaling_server = await websockets.serve(
                self._handle_signaling_client,
                "0.0.0.0",
                self.websocket_port
            )
            logger.info(f"WebRTC signaling server started on port {self.websocket_port}")

        except Exception as e:
            logger.error(f"Failed to start signaling server: {e}")

    async def _handle_signaling_client(self, websocket, path):
        """Handle WebRTC signaling client"""
        client_id = None
        try:
            logger.info(f"New signaling client connected: {websocket.remote_address}")

            async for message in websocket:
                try:
                    data = json.loads(message)
                    response = await self._process_signaling_message(data, websocket)

                    if response:
                        await websocket.send(json.dumps(response))

                except json.JSONDecodeError:
                    await websocket.send(json.dumps({
                        "type": "error",
                        "message": "Invalid JSON message"
                    }))
                except Exception as e:
                    logger.error(f"Error processing signaling message: {e}")
                    await websocket.send(json.dumps({
                        "type": "error",
                        "message": str(e)
                    }))

        except websockets.exceptions.ConnectionClosed:
            logger.info("Signaling client disconnected")
        except Exception as e:
            logger.error(f"Error in signaling client handler: {e}")
        finally:
            if client_id:
                await self.disconnect_client(client_id)

    async def _process_signaling_message(self, data: Dict[str, Any],
                                         websocket: websockets.WebSocketServerProtocol) -> Optional[Dict[str, Any]]:
        """Process WebRTC signaling messages"""
        message_type = data.get("type")

        if message_type == "join":
            # Client wants to join a camera stream
            camera_id = data.get("camera_id")
            quality = StreamQuality(data.get("quality", "medium"))

            if camera_id not in self.active_streams:
                return {
                    "type": "error",
                    "message": f"Camera {camera_id} not available"
                }

            # Connect client
            client_id = await self.connect_client(
                camera_id=camera_id,
                connection_type=ConnectionType.WEBRTC,
                quality=quality,
                websocket=websocket,
                client_info={
                    "ip_address": websocket.remote_address[0] if websocket.remote_address else None
                }
            )

            if not client_id:
                return {
                    "type": "error",
                    "message": "Failed to connect to stream"
                }

            # Get WebRTC stream URL
            stream_info = self.active_streams[camera_id]
            if stream_info.get('webrtc_enabled'):
                # WHEP is HTTP-based (POST SDP offer -> SDP answer), not WebSocket.
                # Return the backend proxy endpoint so browser clients don't need direct :8889 access.
                webrtc_url = f"/proxy/webrtc/{stream_info['stream_path']}/whep"

                return {
                    "type": "joined",
                    "client_id": client_id,
                    "camera_id": camera_id,
                    "webrtc_url": webrtc_url,
                    "ice_servers": self.mediamtx_client.ice_servers
                }
            else:
                return {
                    "type": "error",
                    "message": "WebRTC not available for this camera"
                }

        elif message_type == "leave":
            # Client wants to leave
            client_id = data.get("client_id")
            if client_id:
                await self.disconnect_client(client_id)

            return {
                "type": "left",
                "client_id": client_id
            }

        elif message_type == "quality_change":
            # Client wants to change quality
            client_id = data.get("client_id")
            new_quality = StreamQuality(data.get("quality", "medium"))

            if client_id in self.client_connections:
                connection = self.client_connections[client_id]
                connection.quality = new_quality
                connection.bandwidth_limit = self._get_client_bandwidth_limit(new_quality)

                return {
                    "type": "quality_changed",
                    "client_id": client_id,
                    "quality": new_quality.value
                }

        return None

    # Frame Processing and Delivery

    def _capture_frames(self, camera_id: str):
        """Capture frames from camera stream"""
        logger.debug(f"Frame capture thread started for camera {camera_id}")
        stream_info = self.active_streams.get(camera_id)
        if not stream_info:
            logger.error(f"❌ No stream info found for camera {camera_id}")
            return

        cap = stream_info['capture']
        fps = stream_info['fps']
        stats = stream_info['stats']

        while stream_info['active'] and self.running:
            try:
                # Ultra-fast frame capture - minimal buffering for real-time
                ret, frame = cap.read()
                if ret:
                    # Ultra-fast drain - maximum 1ms to prevent blocking
                    start_drain = time.time()
                    while True:
                        ret2, frame2 = cap.read()
                        if not ret2 or (time.time() - start_drain) > 0.001:
                            break
                        ret, frame = ret2, frame2
                if not ret:
                    # Track consecutive failures and attempt recovery/fallback quickly
                    try:
                        stream_info['consec_fail'] = int(stream_info.get('consec_fail', 0)) + 1
                    except Exception:
                        stream_info['consec_fail'] = 1
                    
                    # Only log warning every 10 failures to reduce spam
                    if int(stream_info.get('consec_fail', 0)) % 10 == 0:
                        logger.warning(f"Failed to read frame from camera {camera_id} (attempt {stream_info.get('consec_fail', 0)})")
                    
                    # Attempt recovery after 5 failures with exponential backoff
                    if int(stream_info.get('consec_fail', 0)) >= 5:
                        current_time = time.time()
                        last_recovery = stream_info.get('last_recovery_attempt', 0)
                        recovery_count = stream_info.get('recovery_attempts', 0)
                        
                        # Exponential backoff: 30s, 60s, 120s, 240s, then 300s max
                        min_interval = min(30 * (2 ** recovery_count), 300)
                        
                        if current_time - last_recovery >= min_interval:
                            # Check if we can start a new recovery (rate limiting)
                            with self.recovery_lock:
                                if len(self.active_recoveries) < self.max_concurrent_recoveries and camera_id not in self.active_recoveries:
                                    self.active_recoveries.add(camera_id)
                                    should_recover = True
                                else:
                                    should_recover = False
                            
                            if should_recover:
                                try:
                                    logger.info(f"Attempting capture recovery for camera {camera_id} (attempt {recovery_count + 1})")
                                    self._attempt_recover_capture(camera_id)
                                    stream_info['consec_fail'] = 0
                                    stream_info['last_recovery_attempt'] = current_time
                                    stream_info['recovery_attempts'] = recovery_count + 1
                                except Exception as rec_e:
                                    logger.warning(f"Capture recover attempt failed for {camera_id}: {rec_e}")
                                    stream_info['recovery_attempts'] = recovery_count + 1
                                finally:
                                    # Always remove from active recoveries
                                    with self.recovery_lock:
                                        self.active_recoveries.discard(camera_id)
                            else:
                                logger.debug(f"Recovery rate limited for {camera_id}, {len(self.active_recoveries)} active recoveries")
                        else:
                            # Wait before next recovery attempt
                            remaining = min_interval - (current_time - last_recovery)
                            logger.debug(f"Recovery for {camera_id} on cooldown, {remaining:.1f}s remaining")
                    
                    # Exponential backoff for sleep time to prevent overwhelming the camera
                    sleep_time = min(0.1 * (2 ** min(stream_info.get('consec_fail', 0) // 5, 3)), 2.0)
                    time.sleep(sleep_time)
                    continue

                # Update statistics
                stats.frames_processed += 1
                stream_info['frame_count'] = stream_info.get('frame_count', 0) + 1

                # Process frame for different qualities
                processed_frames = {}
                for quality in StreamQuality:
                    if self._has_clients_with_quality(camera_id, quality):
                        processed_frames[quality] = self._process_frame(frame, quality)

                # Update stream info
                stream_info['last_frame'] = frame

                # self-test for blank frames; attempt recovery if too many
                try:
                    small = cv2.resize(frame, (64, 36), interpolation=cv2.INTER_LINEAR)
                    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
                    mean_val = float(gray.mean())
                    std_val = float(gray.std())
                    is_blank = (mean_val < 4.0 and std_val < 2.0)
                    if is_blank:
                        stream_info['consec_blank'] = int(stream_info.get('consec_blank', 0)) + 1
                    else:
                        stream_info['consec_blank'] = 0
                    if int(stream_info.get('consec_blank', 0)) >= 8:
                        logger.warning(f"Detected consecutive blank frames on {camera_id} (mean={mean_val:.2f}, std={std_val:.2f}) – attempting capture recovery")
                        try:
                            self._attempt_recover_capture(camera_id)
                            stream_info['consec_blank'] = 0
                        except Exception as rec2_e:
                            logger.warning(f"Blank-frame recover attempt failed for {camera_id}: {rec2_e}")
                except Exception:
                    pass

                # OBJECT DETECTION - Offload to background worker with rate limiting (non-blocking)
                try:
                    # Always keep video path fast; only enqueue latest frame for detection
                    detection_enabled = self._detection_enabled.get(camera_id, False)
                    if detection_enabled:
                        # Lazy-load detector manager once
                        if not self._detector_manager_initialized:
                            try:
                                from .detector_manager import get_detector_manager
                                self._detector_manager = get_detector_manager()
                                self._detector_manager_initialized = True
                                logger.info("✅ Detector manager lazy-loaded successfully")
                            except Exception as e:
                                logger.error(f"❌ Failed to lazy-load detector manager: {e}")
                                self._detector_manager = None
                        
                        if self._detector_manager is not None:
                            # Create a per-camera detection worker thread on demand
                            if camera_id not in self._detector_threads or not self._detector_threads[camera_id].is_alive():
                                t = threading.Thread(target=self._detection_worker, args=(camera_id,), daemon=True)
                                self._detector_threads[camera_id] = t
                                t.start()

                            # Update latest frame snapshot and timestamp for the worker
                            stream_info['latest_frame'] = frame
                            stream_info['latest_frame_ts'] = time.time()
                    else:
                        # If detection is disabled, clear any smoothed state and skip work
                        try:
                            self._detector_bbox_ema.pop(camera_id, None)
                            self._tracks_state.pop(camera_id, None)
                        except Exception:
                            pass
                except Exception as e:
                    logger.debug(f"Detection enqueue error: {e}")

                # Ultra-fast motion detection with minimal processing
                try:
                    if not stream_info.get('motion_detection_enabled', False):
                        detector = None
                    else:
                        detector = self._motion_detectors.get(camera_id)
                    if detector is not None:
                        # Enhanced motion detection with learning capabilities
                        mres = detector.detect(frame)
                        overlay_frame = getattr(mres, 'overlay_frame', None)
                        
                        # Only log when motion is detected to reduce spam
                        if mres.has_motion:
                            logger.info(f"🎯 Motion detected for camera {camera_id}: score={mres.score:.3f}, regions={len(mres.regions)}")
                        
                        # Update motion state
                        self._motion_state[camera_id] = mres

                        # Heuristic camera-movement detection: if total motion area is very large or many regions
                        try:
                            total_area = 0.0
                            num_regions = 0
                            frame_h, frame_w = frame.shape[:2]
                            frame_area = float(max(1, frame_w) * max(1, frame_h))
                            for r in (mres.regions or []):
                                total_area += float(getattr(r, 'area', r.w * r.h))
                                num_regions += 1
                            large_global_motion = (total_area / max(1.0, frame_area)) > 0.35 or num_regions >= 12
                            if large_global_motion:
                                # Start/extend suppression window (750ms)
                                until = time.time() + 0.75
                                prev_until = float(self._camera_move_until.get(camera_id, 0.0))
                                if until > prev_until:
                                    self._camera_move_until[camera_id] = until
                                # Reset smoothing caches and simple centroid tracker to avoid trails
                                try:
                                    self._bbox_ema.pop(camera_id, None)
                                except Exception:
                                    pass
                                try:
                                    if hasattr(self, '_centroid_trackers'):
                                        self._centroid_trackers.pop(camera_id, None)
                                except Exception:
                                    pass
                                # Consumers will hide overlays during suppression window via flag in payload
                            # Attach movement flag for clients
                            moving_flag = time.time() < float(self._camera_move_until.get(camera_id, 0.0))
                        except Exception:
                            moving_flag = False
                        
                        # Emit motion update if callback is set
                        if self.on_motion_update and mres.has_motion:
                            try:
                                frame_h, frame_w = frame.shape[:2]
                                # Simple centroid tracker over motion regions
                                if not hasattr(self, '_centroid_trackers'):
                                    self._centroid_trackers = {}
                                tracker = self._centroid_trackers.get(camera_id)
                                if tracker is None:
                                    tracker = SimpleCentroidTracker(max_distance=64.0, max_age=12)
                                    self._centroid_trackers[camera_id] = tracker
                                centroids: List[Tuple[float, float]] = []
                                region_centers: List[Tuple[float, float]] = []
                                for r in mres.regions:
                                    cx = r.x + r.w / 2.0
                                    cy = r.y + r.h / 2.0
                                    centroids.append((cx, cy))
                                    region_centers.append((cx, cy))
                                tracked = tracker.update(centroids) if centroids else []
                                # Store tracks state for UI retrieval APIs if needed
                                tracks_out: List[Dict[str, Any]] = []
                                # EMA smoothing parameters (lower alpha = smoother)
                                # Increase alpha for snappier lock-on during rover movement
                                ema_alpha = 0.60
                                cam_smooth = self._bbox_ema.setdefault(camera_id, {})
                                for t in tracked:
                                    # Find closest current region to provide a bbox for UI
                                    bx = 0; by = 0; bw = 0; bh = 0
                                    try:
                                        best_i = -1
                                        best_d = 1e9
                                        for i, rc in enumerate(region_centers):
                                            dx = float(t['x'] - rc[0])
                                            dy = float(t['y'] - rc[1])
                                            d = (dx*dx + dy*dy)
                                            if d < best_d:
                                                best_d = d
                                                best_i = i
                                        if best_i >= 0:
                                            r = mres.regions[best_i]
                                            bx, by, bw, bh = int(r.x), int(r.y), int(r.w), int(r.h)
                                    except Exception:
                                        pass
                                    # Smooth bbox via EMA for stability
                                    tid = int(t['id'])
                                    prev = cam_smooth.get(tid)
                                    if prev:
                                        bx = int(prev['x'] * (1-ema_alpha) + bx * ema_alpha)
                                        by = int(prev['y'] * (1-ema_alpha) + by * ema_alpha)
                                        bw = int(prev['w'] * (1-ema_alpha) + bw * ema_alpha)
                                        bh = int(prev['h'] * (1-ema_alpha) + bh * ema_alpha)
                                    # Compute normalized center and velocity for predictive UI
                                    cx = float(bx) + float(bw) / 2.0
                                    cy = float(by) + float(bh) / 2.0
                                    nx = float(cx) / max(1.0, float(frame_w))
                                    ny = float(cy) / max(1.0, float(frame_h))
                                    now_secs = time.time()
                                    prev_nx = float(prev.get('nx', nx)) if prev else nx
                                    prev_ny = float(prev.get('ny', ny)) if prev else ny
                                    prev_ts = float(prev.get('ts', now_secs)) if prev else now_secs
                                    dt = max(1e-3, now_secs - prev_ts)
                                    vx_nps = (nx - prev_nx) / dt
                                    vy_nps = (ny - prev_ny) / dt
                                    cam_smooth[tid] = {
                                        'x': float(bx), 'y': float(by), 'w': float(bw), 'h': float(bh),
                                        'nx': float(nx), 'ny': float(ny), 'ts': float(now_secs),
                                        'vx_nps': float(vx_nps), 'vy_nps': float(vy_nps),
                                    }

                                    tracks_out.append({
                                        'id': tid,
                                        'x': float(t['x']),
                                        'y': float(t['y']),
                                        'bbox': {'x': bx, 'y': by, 'w': bw, 'h': bh},
                                        'center': {'nx': float(nx), 'ny': float(ny)},
                                        'velocity': {'vx_norm_per_sec': float(vx_nps), 'vy_norm_per_sec': float(vy_nps)},
                                        'class': 'object',
                                        'confidence': 0.99,
                                    })
                                self._tracks_state[camera_id] = tracks_out
                                motion_payload = {
                                    "camera_id": camera_id,
                                    "motion": {
                                        "has_motion": mres.has_motion,
                                        "score": mres.score,
                                        "regions": [
                                            {"x": r.x, "y": r.y, "w": r.w, "h": r.h, "area": r.area}
                                            for r in mres.regions
                                        ],
                                        "frame_width": frame_w,
                                        "frame_height": frame_h,
                                        "tracks": self._tracks_state.get(camera_id, [])
                                    },
                                    "timestamp": datetime.now().isoformat(),
                                    "camera_moving": moving_flag,
                                }
                                
                                # Emit motion data
                                self.on_motion_update(camera_id, motion_payload)
                                
                                # Broadcast to WebSocket clients
                                if hasattr(self, '_loop') and self._loop and self._loop.is_running():
                                    self._loop.call_soon_threadsafe(
                                        lambda: asyncio.create_task(self._broadcast_detection_data(camera_id, motion_payload))
                                    )
                                
                            except Exception as e:
                                logger.warning(f"Motion emit error for {camera_id}: {e}")
                    elif stream_info.get('motion_detection_enabled', False):
                        # Avoid log storms: this loop runs at camera FPS, and detection is lazy-loaded.
                        # If motion detection is enabled but the detector hasn't been created yet,
                        # warn at most once every ~10s per camera.
                        try:
                            now_ts = time.time()
                            last_ts = float(getattr(self, "_last_no_detector_warn", {}).get(camera_id, 0.0))
                            if (now_ts - last_ts) >= 10.0:
                                if not hasattr(self, "_last_no_detector_warn"):
                                    self._last_no_detector_warn = {}
                                self._last_no_detector_warn[camera_id] = now_ts
                                logger.warning(f"No motion detector found for camera {camera_id}")
                        except Exception:
                            # Best-effort; never break streaming because of logging state.
                            pass
                        continue

                    # Simple motion detection only - no tracking needed
                except Exception as err:
                    # Do not silently swallow detection errors; report and emit a safe heartbeat
                    logger.error(f"Motion pipeline error on {camera_id}: {err}")
                    try:
                        if self.on_motion_update and frame is not None:
                            fh, fw = frame.shape[:2]
                            safe_payload = {
                                "camera_id": camera_id,
                                "motion": {
                                    "has_motion": False,
                                    "score": 0.0,
                                    "regions": [],
                                    "frame_width": fw,
                                    "frame_height": fh,
                                },
                                "timestamp": datetime.now().isoformat(),
                            }
                            self.on_motion_update(camera_id, safe_payload)
                    except Exception:
                        pass

                # Skip complex AI processing for speed - focus on fast motion detection only

                # Distribute frames to clients with timestamp for latency tracking
                frame_timestamp = time.time()
                self._distribute_frames(camera_id, processed_frames, frame_timestamp)

                # Minimal yield for maximum speed
                time.sleep(0.0001)  # Almost no delay

            except Exception as e:
                logger.error(f"Error capturing frame for camera {camera_id}: {e}")
                time.sleep(1)

        # Cleanup
        cap.release()
        logger.info(f"Frame capture stopped for camera {camera_id}")

    def _attempt_recover_capture(self, camera_id: str) -> None:
        """Try to recover capture by toggling RTSP transport or falling back to MediaMTX RTSP."""
        stream_info = self.active_streams.get(camera_id)
        if not stream_info:
            return
        cap: cv2.VideoCapture = stream_info.get('capture')
        current_url = stream_info.get('capture_url')
        webrtc_enabled = bool(stream_info.get('webrtc_enabled'))
        stream_path = stream_info.get('stream_path') or str(camera_id)
        mediamtx_rtsp = f"rtsp://{self.mediamtx_client.mediamtx_host}:8554/{stream_path}"

        def _with_opts(url: str, transport: str) -> str:
            try:
                sep = "&" if "?" in url else "?"
                return (
                    f"{url}{sep}rtsp_transport={transport}"
                    "&fflags=nobuffer&flags=low_delay&reorder_queue_size=0&buffer_size=0&max_delay=0"
                )
            except Exception:
                return url

        try:
            try:
                cap.release()
            except Exception:
                pass
            # toggle transport
            use_tcp = ("rtsp_transport=tcp" in (current_url or "").lower())
            next_transport = "udp" if use_tcp else "tcp"
            logger.debug(f"Attempting capture transport switch for {camera_id}: {next_transport}")
            new_cap = cv2.VideoCapture(_with_opts(current_url, next_transport), cv2.CAP_FFMPEG)
            new_cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 6000)
            new_cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, 3000)
            try:
                new_cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            except Exception:
                pass
            if new_cap.isOpened():
                stream_info['capture'] = new_cap
                return
            # fallback to mediamtx
            if webrtc_enabled:
                logger.info(f"Falling back capture to MediaMTX RTSP for {camera_id}: {mediamtx_rtsp}")
                new_cap = cv2.VideoCapture(_with_opts(mediamtx_rtsp, "udp"), cv2.CAP_FFMPEG)
                new_cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 6000)
                new_cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, 3000)
                try:
                    new_cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                except Exception:
                    pass
                if new_cap.isOpened():
                    stream_info['capture'] = new_cap
                    stream_info['capture_url'] = mediamtx_rtsp
                    return
            # last resort reopen original url with udp
            logger.info(f"Reopening original capture URL for {camera_id} with UDP...")
            new_cap = cv2.VideoCapture(_with_opts(current_url, "udp"), cv2.CAP_FFMPEG)
            new_cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 6000)
            new_cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, 3000)
            if new_cap.isOpened():
                stream_info['capture'] = new_cap
                return
            logger.error(f"All capture recovery attempts failed for {camera_id}")
        except Exception as e:
            logger.warning(f"Capture recovery error for {camera_id}: {e}")

    # Public motion accessors
    def get_motion_status(self, camera_id: str) -> Optional[Dict[str, Any]]:
        try:
            state = self._motion_state.get(camera_id)
            if not state:
                return None
            # Try to include the frame dimensions to allow the client to normalize coordinates correctly
            frame_w = 0
            frame_h = 0
            try:
                stream_info = self.active_streams.get(camera_id) if hasattr(self, 'active_streams') else None
                if stream_info is not None:
                    last_frame = stream_info.get('last_frame')
                    if last_frame is not None:
                        frame_h, frame_w = last_frame.shape[:2]
            except Exception:
                frame_w = 0
                frame_h = 0

            # Aggregate regions into a single bounding box (pixel space)
            bbox_px = None
            bbox_norm = None
            try:
                if state.regions:
                    min_x = min(r.x for r in state.regions)
                    min_y = min(r.y for r in state.regions)
                    max_x = max(r.x + r.w for r in state.regions)
                    max_y = max(r.y + r.h for r in state.regions)
                    bw = max(0, max_x - min_x)
                    bh = max(0, max_y - min_y)
                    bbox_px = {"x": min_x, "y": min_y, "w": bw, "h": bh}
                    if frame_w > 0 and frame_h > 0:
                        bbox_norm = {
                            "x": float(min_x) / float(frame_w),
                            "y": float(min_y) / float(frame_h),
                            "w": float(bw) / float(frame_w),
                            "h": float(bh) / float(frame_h),
                        }
            except Exception:
                bbox_px = None
                bbox_norm = None

            return {
                "has_motion": state.has_motion,
                "score": state.score,
                "regions": [
                    {"x": r.x, "y": r.y, "w": r.w, "h": r.h, "area": r.area}
                    for r in state.regions
                ],
                "frame_width": frame_w,
                "frame_height": frame_h,
                "bbox": bbox_px,
                "bbox_norm": bbox_norm,
            }
        except Exception:
            return None

    # Detector configuration (per camera)
    def get_detection_config(self, camera_id: str) -> Dict[str, Any]:
        try:
            # Defaults: out-of-the-box tracking of people and cars via MobileNet SSD (default detector)
            defaults = {
                'verification_enabled': False,  # OFF BY DEFAULT - user must enable
                'models': ['mobilenet'],        # MobileNet SSD is the default detector
                'min_confidence': 0.20,        # Lower threshold for better detection (reference uses 0.25 after 0.01-0.99 filter)
                'mobilenet_nms_iou': 0.45,     # Advanced: NMS IoU for MobileNet
                'mobilenet_max_det': 50,       # Advanced: max detections
                'max_detections_per_frame': 25, # Hard guard against UI/object spam
                'roi_mode': 'full',            # 'full' | 'zones' | 'full_then_zones'
                'target_classes': ['person', 'car'],  # Default to essential classes only
                'max_rois': 2,                 # Check up to 2 motion regions
                'rate_limit_ms': 500,          # More frequent checks (faster)
                'debug_fullframe': False,      # When true, run detector on full frame regardless of motion
                'detect_without_motion': True, # When shapes exist, allow detection even without motion
                'always_detect': True,         # Always run detection regardless of motion
            }
            cur = self._detector_config.get(camera_id, {})
            out = defaults.copy()
            out.update({k: v for k, v in cur.items() if v is not None})
            return out
        except Exception:
            return {
                'verification_enabled': False,  # OFF BY DEFAULT - user must enable
                'models': ['mobilenet'],        # MobileNet SSD default
                'min_confidence': 0.20,         # Lower threshold
                'target_classes': ['person', 'car'],
                'max_rois': 2,
                'rate_limit_ms': 500,
                'debug_fullframe': False,
                'detect_without_motion': True,
                'always_detect': True,
            }

    def _detection_worker(self, camera_id: str) -> None:
        """Background worker that pulls latest frames and runs detection at a controlled rate.

        Non-blocking to the capture loop. Only processes when detection is enabled, with
        per-camera rate limiting based on `rate_limit_ms` from `get_detection_config`.
        """
        try:
            while camera_id in self.active_streams and self.running:
                try:
                    if not self._detection_enabled.get(camera_id, False):
                        time.sleep(0.1)
                        continue
                    stream_info = self.active_streams.get(camera_id)
                    if not stream_info:
                        time.sleep(0.1)
                        continue
                    cfg = self.get_detection_config(camera_id)
                    now_ts = time.time()
                    verification_enabled = bool(cfg.get('verification_enabled', False))
                    prev_ver = self._verification_state.get(camera_id)
                    if not verification_enabled:
                        if prev_ver is not False:
                            logger.info(f"🔕 Skipping detector for {camera_id} — verification disabled in config")
                            try:
                                self._detector_bbox_ema.pop(camera_id, None)
                            except Exception:
                                pass
                            try:
                                self._tracks_state.pop(camera_id, None)
                            except Exception:
                                pass
                            try:
                                if self._detector_manager and hasattr(self._detector_manager, 'reset_tracker'):
                                    self._detector_manager.reset_tracker(camera_id)
                            except Exception:
                                pass
                        self._verification_state[camera_id] = False
                        self._detector_inflight[camera_id] = False
                        self._last_verification_time[camera_id] = now_ts
                        time.sleep(0.1)
                        continue
                    if prev_ver is not True:
                        logger.info(f"🎯 Detector verification enabled for {camera_id}")
                        self._verification_state[camera_id] = True
                    rate_limit_ms = int(cfg.get('rate_limit_ms', 500))
                    last_ts = float(self._last_verification_time.get(camera_id, 0))
                    if (now_ts - last_ts) * 1000.0 < rate_limit_ms:
                        time.sleep(0.01)
                        continue
                    frame = stream_info.get('latest_frame')
                    if frame is None:
                        time.sleep(0.01)
                        continue
                    if self._detector_inflight.get(camera_id, False):
                        time.sleep(0.005)
                        continue
                    self._detector_inflight[camera_id] = True
                    try:
                        frame_h, frame_w = frame.shape[:2]
                        min_conf = float(cfg.get('min_confidence', 0.25))
                        try:
                            dm_cfg_existing = self._detector_manager.get_detection_config(camera_id)
                            dm_cfg_update: Dict[str, Any] = {}
                            if float(dm_cfg_existing.get('confidence', -1)) != float(min_conf):
                                dm_cfg_update['confidence'] = min_conf
                            nms = cfg.get('mobilenet_nms_iou')
                            if nms is not None and float(dm_cfg_existing.get('nms', -1)) != float(nms):
                                dm_cfg_update['nms'] = float(nms)
                            max_det = cfg.get('mobilenet_max_det')
                            if max_det is not None and int(dm_cfg_existing.get('max_det', -1)) != int(max_det):
                                dm_cfg_update['max_det'] = int(max_det)
                            target_classes = list(cfg.get('target_classes') or ['person', 'car'])
                            if list(dm_cfg_existing.get('classes') or []) != target_classes:
                                dm_cfg_update['classes'] = target_classes
                            if dm_cfg_update:
                                self._detector_manager.set_detection_config(camera_id, {
                                    **dm_cfg_existing,
                                    **dm_cfg_update,
                                })
                        except Exception:
                            pass
                        roi_mode = str(cfg.get('roi_mode') or 'full').lower()
                        if roi_mode == 'zones':
                            rois = self._compute_shape_rois(camera_id, frame_w, frame_h)
                            detections_all: List[Dict[str, Any]] = []
                            for (rx, ry, rw, rh) in rois[: int(cfg.get('max_rois', 2))]:
                                dets = self._detector_manager.detect_in_region(
                                    camera_id, frame, {'x': rx, 'y': ry, 'w': rw, 'h': rh}, conf_threshold=min_conf
                                )
                                detections_all.extend(dets or [])
                            tracks_tmp = self._detector_manager.get_tracker(camera_id).update(detections_all)
                            result = {'detections': detections_all, 'tracks': tracks_tmp}
                        elif roi_mode == 'full_then_zones':
                            base = self._detector_manager.detect_and_track(camera_id, frame, conf_threshold=min_conf)
                            rois = self._compute_shape_rois(camera_id, frame_w, frame_h)
                            extra: List[Dict[str, Any]] = []
                            for (rx, ry, rw, rh) in rois[: int(cfg.get('max_rois', 2))]:
                                dets = self._detector_manager.detect_in_region(
                                    camera_id, frame, {'x': rx, 'y': ry, 'w': rw, 'h': rh}, conf_threshold=min_conf
                                )
                                extra.extend(dets or [])
                            if extra:
                                base['detections'] = (base.get('detections') or []) + extra
                            result = base
                        else:
                            result = self._detector_manager.detect_and_track(camera_id, frame, conf_threshold=min_conf)
                        detections = result.get('detections', [])
                        tracks = result.get('tracks', [])
                        target_classes = cfg.get('target_classes') or ['person', 'car']
                        filtered_detections: List[Dict[str, Any]] = []
                        for det in detections:
                            cls_name = det.get('class', 'object')
                            if (not target_classes) or (cls_name in target_classes):
                                bbox = det.get('bbox', {})
                                filtered_detections.append({
                                    'bbox': {
                                        'x': int(bbox.get('x', 0)),
                                        'y': int(bbox.get('y', 0)),
                                        'w': int(bbox.get('w', 0)),
                                        'h': int(bbox.get('h', 0)),
                                    },
                                    'class': cls_name,
                                    'confidence': float(det.get('confidence', 0.0)),
                                })
                        try:
                            max_per_frame = int(cfg.get('max_detections_per_frame', 25))
                            if len(filtered_detections) > max_per_frame:
                                filtered_detections = filtered_detections[:max_per_frame]
                        except Exception:
                            pass
                        # Suppress track/detection emits during camera movement suppression window
                        suppress_outputs = time.time() < float(self._camera_move_until.get(camera_id, 0.0))
                        if filtered_detections and not suppress_outputs:
                            detections_payload = {
                                'camera_id': camera_id,
                                'detections': filtered_detections,
                                'timestamp': datetime.now().isoformat(),
                                'frame_width': frame_w,
                                'frame_height': frame_h,
                                'camera_moving': False,
                            }
                            if self.on_detection_update:
                                try:
                                    self.on_detection_update(camera_id, detections_payload)
                                except Exception:
                                    pass
                            if hasattr(self, '_loop') and self._loop and self._loop.is_running():
                                self._loop.call_soon_threadsafe(
                                    lambda: asyncio.create_task(self._broadcast_detection_data(camera_id, detections_payload))
                                )
                        elif filtered_detections and suppress_outputs:
                            # Do not emit; also clear smoothing state aggressively
                            try:
                                self._detector_bbox_ema.pop(camera_id, None)
                                if self._detector_manager:
                                    try:
                                        self._detector_manager.reset_tracker(camera_id)
                                    except Exception:
                                        pass
                            except Exception:
                                pass
                        if tracks and not suppress_outputs:
                            # Slightly higher alpha for snappier overlay response from detector tracks
                            ema_alpha = 0.45
                            cam_smooth = self._detector_bbox_ema.setdefault(camera_id, {})
                            smoothed_tracks: List[Dict[str, Any]] = []
                            for tr in tracks:
                                try:
                                    tid = int(tr.get('id', 0))
                                    bb = tr.get('bbox') or {}
                                    bx = int(bb.get('x', 0))
                                    by = int(bb.get('y', 0))
                                    bw = int(bb.get('w', 0))
                                    bh = int(bb.get('h', 0))
                                    prev = cam_smooth.get(tid)
                                    if prev:
                                        bx = int(prev['x'] * (1 - ema_alpha) + bx * ema_alpha)
                                        by = int(prev['y'] * (1 - ema_alpha) + by * ema_alpha)
                                        bw = int(prev['w'] * (1 - ema_alpha) + bw * ema_alpha)
                                        bh = int(prev['h'] * (1 - ema_alpha) + bh * ema_alpha)
                                    # Update normalized center and velocity for predictive client-side smoothing
                                    cx = float(bx) + float(bw) / 2.0
                                    cy = float(by) + float(bh) / 2.0
                                    nx = float(cx) / max(1.0, float(frame_w))
                                    ny = float(cy) / max(1.0, float(frame_h))
                                    now_secs = time.time()
                                    prev_nx = float(prev.get('nx', nx)) if prev else nx
                                    prev_ny = float(prev.get('ny', ny)) if prev else ny
                                    prev_ts = float(prev.get('ts', now_secs)) if prev else now_secs
                                    dt = max(1e-3, now_secs - prev_ts)
                                    vx_nps = (nx - prev_nx) / dt
                                    vy_nps = (ny - prev_ny) / dt
                                    cam_smooth[tid] = {
                                        'x': float(bx), 'y': float(by), 'w': float(bw), 'h': float(bh),
                                        'nx': float(nx), 'ny': float(ny), 'ts': float(now_secs),
                                        'vx_nps': float(vx_nps), 'vy_nps': float(vy_nps),
                                    }
                                    st = dict(tr)
                                    st['bbox'] = {'x': bx, 'y': by, 'w': bw, 'h': bh}
                                    st['center'] = {'nx': float(nx), 'ny': float(ny)}
                                    st['velocity'] = {'vx_norm_per_sec': float(vx_nps), 'vy_norm_per_sec': float(vy_nps)}
                                    smoothed_tracks.append(st)
                                except Exception:
                                    smoothed_tracks.append(tr)
                            try:
                                if target_classes:
                                    smoothed_tracks = [t for t in smoothed_tracks if str(t.get('class','')) in target_classes]
                            except Exception:
                                pass
                            tracks_payload = {
                                'camera_id': camera_id,
                                'tracks': smoothed_tracks,
                                'timestamp': datetime.now().isoformat(),
                                'frame_width': frame_w,
                                'frame_height': frame_h,
                                'camera_moving': False,
                            }
                            if self.on_tracks_update:
                                try:
                                    self.on_tracks_update(camera_id, tracks_payload)
                                except Exception:
                                    pass
                            if hasattr(self, '_loop') and self._loop and self._loop.is_running():
                                self._loop.call_soon_threadsafe(
                                    lambda: asyncio.create_task(self._broadcast_tracks_data(camera_id, tracks_payload))
                                )
                        elif tracks and suppress_outputs:
                            # When suppressing, immediately clear smoothing so no stale boxes persist
                            try:
                                self._detector_bbox_ema.pop(camera_id, None)
                                if self._detector_manager:
                                    try:
                                        self._detector_manager.reset_tracker(camera_id)
                                    except Exception:
                                        pass
                            except Exception:
                                pass
                        self._last_verification_time[camera_id] = now_ts
                    finally:
                        self._detector_inflight[camera_id] = False
                    time.sleep(0.001)
                except Exception as _dw_e:
                    logger.debug(f"Detection worker error for {camera_id}: {_dw_e}")
                    time.sleep(0.02)
        except Exception as _e:
            logger.debug(f"Detection worker exited for {camera_id}: {_e}")

    def enable_detection(self, camera_id: str) -> bool:
        """Enable object detection for a specific camera"""
        try:
            self._detection_enabled[camera_id] = True
            logger.info(f"✅ Object detection ENABLED for camera {camera_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to enable detection for {camera_id}: {e}")
            return False
    
    def disable_detection(self, camera_id: str) -> bool:
        """Disable object detection for a specific camera"""
        try:
            self._detection_enabled[camera_id] = False
            logger.info(f"🛑 Object detection DISABLED for camera {camera_id}")
            # Clear smoothed state and suppress further emits
            try:
                self._detector_bbox_ema.pop(camera_id, None)
                self._tracks_state.pop(camera_id, None)
                # Also reset worker-specific timestamps so it idles immediately
                self._last_verification_time[camera_id] = 0.0
                # Mark worker as not in-flight to avoid lingering busy state
                self._detector_inflight[camera_id] = False
            except Exception:
                pass
            return True
        except Exception as e:
            logger.error(f"Failed to disable detection for {camera_id}: {e}")
            return False
    
    def is_detection_enabled(self, camera_id: str) -> bool:
        """Check if detection is enabled for a camera"""
        return self._detection_enabled.get(camera_id, False)

    def update_detection_config(self, camera_id: str, updates: Dict[str, Any]) -> bool:
        try:
            cur = self._detector_config.get(camera_id, {})
            if not isinstance(cur, dict):
                cur = {}
            # Sanitize inputs
            cleaned: Dict[str, Any] = {}
            if 'verification_enabled' in updates:
                cleaned['verification_enabled'] = bool(updates.get('verification_enabled'))
            if 'models' in updates and isinstance(updates.get('models'), list):
                cleaned['models'] = [str(m) for m in updates.get('models') or []]
            if 'min_confidence' in updates:
                try:
                    cleaned['min_confidence'] = max(0.0, min(1.0, float(updates.get('min_confidence'))))
                except Exception:
                    pass
            # Advanced MobileNet params
            if 'mobilenet_nms_iou' in updates:
                try:
                    cleaned['mobilenet_nms_iou'] = max(0.0, min(1.0, float(updates.get('mobilenet_nms_iou'))))
                except Exception:
                    pass
            if 'mobilenet_max_det' in updates:
                try:
                    cleaned['mobilenet_max_det'] = max(1, int(updates.get('mobilenet_max_det')))
                except Exception:
                    pass
            if 'target_classes' in updates and isinstance(updates.get('target_classes'), list):
                cleaned['target_classes'] = [str(c) for c in updates.get('target_classes') or []]
                # Reset tracker to purge stale tracks with old labels when class filter changes
                try:
                    if self._detector_manager is not None and hasattr(self._detector_manager, 'reset_tracker'):
                        self._detector_manager.reset_tracker(camera_id)
                except Exception:
                    pass
            if 'max_detections_per_frame' in updates:
                try:
                    cleaned['max_detections_per_frame'] = max(1, int(updates.get('max_detections_per_frame')))
                except Exception:
                    pass
            if 'roi_mode' in updates:
                try:
                    mode = str(updates.get('roi_mode') or 'full').lower()
                    if mode not in ('full','zones','full_then_zones'):
                        mode = 'full'
                    cleaned['roi_mode'] = mode
                except Exception:
                    pass
            if 'max_rois' in updates:
                try:
                    cleaned['max_rois'] = max(1, int(updates.get('max_rois')))
                except Exception:
                    pass
            if 'rate_limit_ms' in updates:
                try:
                    cleaned['rate_limit_ms'] = max(100, int(updates.get('rate_limit_ms')))
                except Exception:
                    pass
            if 'debug_fullframe' in updates:
                try:
                    cleaned['debug_fullframe'] = bool(updates.get('debug_fullframe'))
                except Exception:
                    pass
            if 'detect_without_motion' in updates:
                try:
                    cleaned['detect_without_motion'] = bool(updates.get('detect_without_motion'))
                except Exception:
                    pass
            if 'always_detect' in updates:
                try:
                    cleaned['always_detect'] = bool(updates.get('always_detect'))
                except Exception:
                    pass

            cur.update(cleaned)
            self._detector_config[camera_id] = cur

            # Propagate relevant overrides to the detector manager so they take effect immediately
            try:
                if self._detector_manager is not None:
                    # Switch camera detector if model selection provided
                    if 'models' in cleaned and isinstance(cleaned.get('models'), list):
                        models = [str(m).lower() for m in cleaned.get('models')]
                        model_choice = None
                        if any('yolo' in m for m in models):
                            model_choice = 'yolo'
                        elif any('mobilenet' in m for m in models):
                            model_choice = 'mobilenet'
                        if model_choice:
                            try:
                                self._detector_manager.set_camera_detector(camera_id, model_choice)
                            except Exception:
                                pass
                    dm_cfg: Dict[str, Any] = {}
                    if 'min_confidence' in cleaned:
                        dm_cfg['confidence'] = cleaned['min_confidence']
                    if 'mobilenet_nms_iou' in cleaned:
                        dm_cfg['nms'] = cleaned['mobilenet_nms_iou']
                    if 'mobilenet_max_det' in cleaned:
                        dm_cfg['max_det'] = cleaned['mobilenet_max_det']
                    # Propagate class filter to detector manager so filtering happens before tracking
                    if 'target_classes' in cleaned:
                        try:
                            dm_cfg['classes'] = list(cleaned.get('target_classes') or [])
                        except Exception:
                            dm_cfg['classes'] = []
                    if dm_cfg:
                        self._detector_manager.set_detection_config(camera_id, {
                            **self._detector_manager.get_detection_config(camera_id),
                            **dm_cfg
                        })
            except Exception:
                pass
            return True
        except Exception:
            return False

    def get_motion_detector(self, camera_id: str) -> Optional[SimpleMotionDetector]:
        """Get the motion detector instance for a camera"""
        try:
            return self._motion_detectors.get(camera_id)
        except Exception:
            return None
    
    def get_motion_learning_status(self, camera_id: str) -> Optional[Dict[str, Any]]:
        """Get adaptive learning status for a camera"""
        try:
            detector = self._motion_detectors.get(camera_id)
            if detector and hasattr(detector, 'get_learning_status'):
                return detector.get_learning_status()
            return None
        except Exception as e:
            logger.error(f"Failed to get learning status for camera {camera_id}: {e}")
            return None
    
    def get_all_motion_learning_status(self) -> Dict[str, Dict[str, Any]]:
        """Get adaptive learning status for all cameras"""
        status = {}
        for camera_id in self._motion_detectors:
            camera_status = self.get_motion_learning_status(camera_id)
            if camera_status:
                status[camera_id] = camera_status
        return status
    
    def force_motion_analysis(self, camera_id: str) -> bool:
        """Force immediate LLM analysis for a camera"""
        try:
            detector = self._motion_detectors.get(camera_id)
            if detector and hasattr(detector, 'force_analysis'):
                detector.force_analysis()
                return True
            return False
        except Exception as e:
            logger.error(f"Failed to force analysis for camera {camera_id}: {e}")
            return False
    
    def enable_motion_learning(self, camera_id: str, enabled: bool) -> bool:
        """Enable/disable adaptive learning for a camera"""
        try:
            detector = self._motion_detectors.get(camera_id)
            if detector and hasattr(detector, 'enable_adaptive_learning'):
                detector.enable_adaptive_learning(enabled)
                return True
            return False
        except Exception as e:
            logger.error(f"Failed to set learning state for camera {camera_id}: {e}")
            return False
    
    def enable_scene_analysis(self, camera_id: str, enabled: bool) -> bool:
        """Enable/disable scene analysis for a camera"""
        try:
            detector = self._motion_detectors.get(camera_id)
            if detector and hasattr(detector, 'enable_scene_analysis'):
                detector.enable_scene_analysis(enabled)
                return True
            return False
        except Exception as e:
            logger.error(f"Failed to set scene analysis state for camera {camera_id}: {e}")
            return False
    
    def set_camera_shapes(self, camera_id: str, shapes: Dict[str, List[Any]]) -> None:
        """Store user-drawn shapes (zones/lines/tags) for a camera to enable ROI-based detection."""
        try:
            self._camera_shapes[camera_id] = {
                'zones': shapes.get('zones', []),
                'lines': shapes.get('lines', []),
                'tags': shapes.get('tags', [])
            }
            logger.info(f"Stored shapes for camera {camera_id}: {len(shapes.get('zones', []))} zones, {len(shapes.get('lines', []))} lines, {len(shapes.get('tags', []))} tags")
        except Exception as e:
            logger.error(f"Failed to set shapes for camera {camera_id}: {e}")
    
    def get_camera_shapes(self, camera_id: str) -> Dict[str, List[Any]]:
        """Retrieve user-drawn shapes for a camera."""
        return self._camera_shapes.get(camera_id, {'zones': [], 'lines': [], 'tags': []})
    
    def _compute_shape_rois(self, camera_id: str, frame_w: int, frame_h: int) -> List[Tuple[int, int, int, int]]:
        """Compute ROI bounding boxes from user shapes (zones/lines/tags). Returns list of (x, y, w, h) in pixels."""
        try:
            shapes = self.get_camera_shapes(camera_id)
            rois: List[Tuple[int, int, int, int]] = []
            
            # Process zones - compute bounding box for each zone polygon
            for zone in shapes.get('zones', []):
                if not zone.get('enabled', True):
                    continue
                points = zone.get('points', [])
                if len(points) < 3:
                    continue
                xs = [p['x'] * frame_w for p in points if 'x' in p]
                ys = [p['y'] * frame_h for p in points if 'y' in p]
                if not xs or not ys:
                    continue
                x_min, x_max = int(min(xs)), int(max(xs))
                y_min, y_max = int(min(ys)), int(max(ys))
                w = max(1, x_max - x_min)
                h = max(1, y_max - y_min)
                rois.append((x_min, y_min, w, h))
            
            # Process lines - create small ROI around each line
            margin = 0.05  # 5% margin around lines
            for line in shapes.get('lines', []):
                if not line.get('enabled', True):
                    continue
                p1 = line.get('p1', {})
                p2 = line.get('p2', {})
                if 'x' not in p1 or 'x' not in p2:
                    continue
                x1, y1 = p1['x'] * frame_w, p1['y'] * frame_h
                x2, y2 = p2['x'] * frame_w, p2['y'] * frame_h
                x_min = int(min(x1, x2) - margin * frame_w)
                x_max = int(max(x1, x2) + margin * frame_w)
                y_min = int(min(y1, y2) - margin * frame_h)
                y_max = int(max(y1, y2) + margin * frame_h)
                w = max(1, x_max - x_min)
                h = max(1, y_max - y_min)
                rois.append((x_min, y_min, w, h))
            
            # Process tags - create small ROI around each tag
            tag_size = 0.15  # 15% of frame around each tag
            for tag in shapes.get('tags', []):
                if not tag.get('enabled', True):
                    continue
                if 'x' not in tag or 'y' not in tag:
                    continue
                cx, cy = tag['x'] * frame_w, tag['y'] * frame_h
                half_w = (tag_size * frame_w) / 2
                half_h = (tag_size * frame_h) / 2
                x_min = int(max(0, cx - half_w))
                y_min = int(max(0, cy - half_h))
                w = int(min(frame_w - x_min, tag_size * frame_w))
                h = int(min(frame_h - y_min, tag_size * frame_h))
                rois.append((x_min, y_min, w, h))
            
            return rois
        except Exception as e:
            logger.error(f"Error computing shape ROIs for camera {camera_id}: {e}")
            return []
    
    def get_scene_analysis_status(self, camera_id: str) -> Optional[Dict[str, Any]]:
        """Get scene analysis status for a camera"""
        try:
            detector = self._motion_detectors.get(camera_id)
            if detector and hasattr(detector, 'get_scene_analysis_status'):
                return detector.get_scene_analysis_status()
            return None
        except Exception as e:
            logger.error(f"Failed to get scene analysis status for camera {camera_id}: {e}")
            return None
    
    def get_scene_history(self, camera_id: str, limit: int = 10) -> Optional[List[Dict[str, Any]]]:
        """Get scene analysis history for a camera"""
        try:
            detector = self._motion_detectors.get(camera_id)
            if detector and hasattr(detector, 'get_scene_history'):
                return detector.get_scene_history(limit)
            return None
        except Exception as e:
            logger.error(f"Failed to get scene history for camera {camera_id}: {e}")
            return None
    
    def force_scene_analysis(self, camera_id: str) -> bool:
        """Force immediate scene analysis for a camera"""
        try:
            detector = self._motion_detectors.get(camera_id)
            if detector and hasattr(detector, 'force_scene_analysis'):
                detector.force_scene_analysis()
                return True
            return False
        except Exception as e:
            logger.error(f"Failed to force scene analysis for camera {camera_id}: {e}")
            return False

    def update_motion_params(self, camera_id: str, params: Dict[str, float]) -> bool:
        try:
            det = self._motion_detectors.get(camera_id)
            if det is None:
                # lazily create enhanced motion detector
                self._motion_detectors[camera_id] = SimpleMotionDetector(
                    camera_id=camera_id,
                    ai_agent=self.ai_agent,
                    enable_learning=True
                )
                det = self._motion_detectors[camera_id]
            
            # Update all supported parameters
            if hasattr(det, 'enable_learning'):
                det.enable_learning = True
            
            if 'min_area_norm' in params:
                det.min_area_norm = max(1e-6, float(params['min_area_norm']))
            if 'diff_threshold' in params:
                # Map diff_threshold to MOG2 variance threshold used by SimpleMotionDetector
                try:
                    det.mog2_var_threshold = int(max(4, min(100, params['diff_threshold'])))
                except Exception:
                    pass
            if 'min_area' in params:
                det.min_area = int(max(1, params['min_area']))
            if 'learning_rate' in params:
                det.learning_rate = float(max(0.001, min(0.5, params['learning_rate'])))
            if 'dilate_iterations' in params:
                det.dilate_iterations = int(max(0, min(5, params['dilate_iterations'])))
            if 'motion_history' in params:
                det._motion_history_size = int(max(1, min(10, params['motion_history'])))
            if 'downscale_width' in params:
                det.downscale_width = int(max(64, min(320, params['downscale_width'])))
            if 'car_person_sensitivity' in params:
                det._car_person_sensitivity = float(max(0.1, min(5.0, params['car_person_sensitivity'])))
            if 'motion_point_timeout' in params:
                det.motion_point_timeout = int(max(1000, min(15000, params['motion_point_timeout'])))
            
            # Legacy support for var_threshold
            if 'var_threshold' in params:
                try:
                    det.mog2_var_threshold = int(max(4, min(100, params['var_threshold'])))
                except Exception:
                    pass
            
            # Reset background when changing critical parameters
            if any(key in params for key in ['diff_threshold', 'var_threshold', 'learning_rate', 'downscale_width']):
                det.reset()
            
            # Store updated parameters
            self._motion_params[camera_id] = {
                'min_area_norm': float(getattr(det, 'min_area_norm', 0.0)),
                'diff_threshold': float(getattr(det, 'mog2_var_threshold', 0)),
                'min_area': int(getattr(det, 'min_area', 0)),
                'learning_rate': float(getattr(det, 'learning_rate', 0.0)),
                'dilate_iterations': int(getattr(det, 'dilate_iterations', 0)),
                'motion_history': int(getattr(det, '_motion_history_size', 0)),
                'downscale_width': int(getattr(det, 'downscale_width', 0)),
                'car_person_sensitivity': float(getattr(det, '_car_person_sensitivity', 0.0)),
                'motion_point_timeout': int(getattr(det, 'motion_point_timeout', 5000))
            }
            return True
        except Exception:
            return False

    def get_motion_tuner_preview(self, camera_id: str) -> Optional[Dict[str, Any]]:
        """Return latest frame, foreground mask, and overlay for simple tuner preview."""
        try:
            stream_info = self.active_streams.get(camera_id)
            if not stream_info:
                return None
            frame = stream_info.get('last_frame')
            if frame is None:
                return None
            det = self._motion_detectors.get(camera_id)
            fg_mask = getattr(det, '_last_fg_mask', None) if det is not None else None

            overlay = None
            if fg_mask is not None:
                try:
                    mask_bin = (fg_mask > 0).astype(np.uint8)
                    red = frame.copy()
                    red[:, :, 0] = (red[:, :, 0] * (1 - 0.35)).astype(np.uint8)
                    red[:, :, 1] = (red[:, :, 1] * (1 - 0.35)).astype(np.uint8)
                    red[:, :, 2] = np.maximum(red[:, :, 2], (mask_bin * 255)).astype(np.uint8)
                    overlay = np.where(mask_bin[..., None] > 0, red, frame)
                except Exception:
                    overlay = frame

            def enc(img, ext):
                ok, buf = cv2.imencode(ext, img)
                if not ok:
                    return None
                return base64.b64encode(buf.tobytes()).decode('utf-8')

            out = {'frame': enc(frame, '.jpg')}
            if fg_mask is not None:
                out['mask'] = enc(fg_mask, '.png')
            if overlay is not None:
                out['overlay'] = enc(overlay, '.jpg')
            return out
        except Exception:
            return None

    def get_tracks(self, camera_id: str) -> Optional[Dict[str, Any]]:
        try:
            tracks = self._tracks_state.get(camera_id)
            if tracks is None:
                return None
            # Include frame dimensions for proper normalization on the client
            frame_w = 0
            frame_h = 0
            try:
                stream_info = self.active_streams.get(camera_id) if hasattr(self, 'active_streams') else None
                if stream_info is not None:
                    last_frame = stream_info.get('last_frame')
                    if last_frame is not None:
                        frame_h, frame_w = last_frame.shape[:2]
            except Exception:
                frame_w = 0
                frame_h = 0

            return {
                "camera_id": camera_id,
                "tracks": tracks,
                "frame_width": frame_w,
                "frame_height": frame_h,
            }
        except Exception:
            return None

    def get_track_lifecycle(self, camera_id: str) -> Optional[Dict[str, Any]]:
        try:
            return {
                "camera_id": camera_id,
                "events": self._track_events.get(camera_id, []),
                "active_ids": list(self._track_prev_ids.get(camera_id, set())),
            }
        except Exception:
            return None

    def get_track_trajectories(self, camera_id: str) -> Optional[Dict[str, Any]]:
        try:
            trajs = self._track_trajectories.get(camera_id)
            if trajs is None:
                return None
            # Convert dict keyed by id to list
            return {
                "camera_id": camera_id,
                "trajectories": list(trajs.values()),
            }
        except Exception:
            return None

    def get_stored_trajectories(self, camera_id: str) -> Optional[Dict[str, Any]]:
        """Get stored trajectories from database if available"""
        try:
            if hasattr(self, 'db_manager') and self.db_manager:
                limit = 200
                recs = self.db_manager.list_track_trajectories(camera_id, limit=limit)
                return {
                    "camera_id": camera_id,
                    "trajectories": recs,
                    "limit": limit
                }
            return None
        except Exception:
            return None

    def get_detection_status(self) -> Optional[Dict[str, Any]]:
        """Get comprehensive detection system status across all cameras"""
        try:
            status = {
                "total_cameras": len(self.active_streams) if hasattr(self, 'active_streams') else 0,
                "cameras_with_motion": 0,
                "total_active_tracks": 0,
                "total_detections_last_hour": 0,
                "ai_analyzer_status": "unknown",
                "detection_models": [],
                "system_health": "unknown"
            }
            
            # Count active tracks and motion cameras
            for camera_id in self.active_streams.keys() if hasattr(self, 'active_streams') else []:
                tracks = self._tracks_state.get(camera_id, [])
                status["total_active_tracks"] += len(tracks)
                if tracks:
                    status["cameras_with_motion"] += 1
            
            # AI analyzer status
            if hasattr(self, 'ai_verifier') and self.ai_verifier:
                status["ai_analyzer_status"] = "active"
                if hasattr(self.ai_verifier, 'models'):
                    status["detection_models"] = list(self.ai_verifier.models.keys())
            else:
                status["ai_analyzer_status"] = "inactive"
            
            # System health based on recent activity
            if status["total_active_tracks"] > 0:
                status["system_health"] = "active"
            elif status["total_cameras"] > 0:
                status["system_health"] = "idle"
            else:
                status["system_health"] = "inactive"
                
            return status
        except Exception:
            return None

    def get_detection_logs(self, camera_id: str = None, limit: int = 100, 
                          since: str = None, object_type: str = None) -> Optional[Dict[str, Any]]:
        """Get recent detection logs with optional filtering"""
        try:
            logs = []
            current_time = datetime.now()
            
            # Get logs from detection events
            if hasattr(self, '_detection_logs'):
                for log_entry in self._detection_logs[-limit:]:
                    # Apply filters
                    if camera_id and log_entry.get('camera_id') != camera_id:
                        continue
                    if since:
                        try:
                            since_time = datetime.fromisoformat(since.replace('Z', '+00:00'))
                            log_time = datetime.fromisoformat(log_entry.get('timestamp', '').replace('Z', '+00:00'))
                            if log_time < since_time:
                                continue
                        except:
                            pass
                    if object_type and log_entry.get('object_type') != object_type:
                        continue
                    
                    logs.append(log_entry)
            
            return {
                "logs": logs,
                "total_count": len(logs),
                "filters": {
                    "camera_id": camera_id,
                    "limit": limit,
                    "since": since,
                    "object_type": object_type
                }
            }
        except Exception:
            return None

    def get_track_composite(self, camera_id: str, track_id: str) -> Optional[Dict[str, Any]]:
        """Get composite image for a specific track"""
        try:
            # Find the track
            tracks = self._tracks_state.get(camera_id, [])
            track = None
            for t in tracks:
                if str(t.get('id')) == str(track_id):
                    track = t
                    break
            
            if not track:
                return None
            
            # Get composite if available
            composite_key = f"{camera_id}_{track_id}"
            if hasattr(self, '_track_composites') and composite_key in self._track_composites:
                composite_data = self._track_composites[composite_key]
                return {
                    "track_id": track_id,
                    "camera_id": camera_id,
                    "composite_image": composite_data.get('image'),
                    "created_at": composite_data.get('timestamp'),
                    "track_info": {
                        "bbox": track.get('bbox'),
                        "label": track.get('label', ''),
                        "confidence": track.get('label_confidence', 0.0),
                        "age": track.get('age', 0)
                    }
                }
            
            return None
        except Exception:
            return None

    def get_active_objects(self) -> Optional[Dict[str, Any]]:
        """Get all currently tracked objects across all cameras"""
        try:
            active_objects = []
            
            for camera_id in self.active_streams.keys() if hasattr(self, 'active_streams') else []:
                tracks = self._tracks_state.get(camera_id, [])
                for track in tracks:
                    if track.get('age', 0) > 0:  # Only include tracks that have been updated
                        obj_info = {
                            "camera_id": camera_id,
                            "track_id": track.get('id'),
                            "short_id": track.get('short_id', ''),
                            "bbox": track.get('bbox'),
                            "label": track.get('label', ''),
                            "confidence": track.get('label_confidence', 0.0),
                            "age": track.get('age', 0),
                            "center": track.get('center', {}),
                            "velocity": track.get('velocity', {}),
                            "last_update": track.get('last_update')
                        }
                        active_objects.append(obj_info)
            
            return {
                "active_objects": active_objects,
                "total_count": len(active_objects),
                "cameras_with_objects": len(set(obj['camera_id'] for obj in active_objects))
            }
        except Exception:
            return None

    def _log_detection_event(self, camera_id: str, track_id: str, label: str, 
                           confidence: float, bbox: tuple, composite_available: bool = False):
        """Log a detection event for API access"""
        try:
            if not hasattr(self, '_detection_logs'):
                self._detection_logs = []
            
            log_entry = {
                "timestamp": datetime.now().isoformat(),
                "camera_id": camera_id,
                "track_id": track_id,
                "object_type": label,
                "confidence": confidence,
                "bbox": bbox,
                "composite_available": composite_available
            }
            
            self._detection_logs.append(log_entry)
            
            # Keep only last 1000 entries
            if len(self._detection_logs) > 1000:
                self._detection_logs = self._detection_logs[-1000:]
                
        except Exception as e:
            logger.debug(f"Failed to log detection event: {e}")

    def _store_track_composite(self, camera_id: str, track_id: str, composite_image: str):
        """Store track composite for API access"""
        try:
            if not hasattr(self, '_track_composites'):
                self._track_composites = {}
            
            composite_key = f"{camera_id}_{track_id}"
            self._track_composites[composite_key] = {
                "image": composite_image,
                "timestamp": datetime.now().isoformat()
            }
            
            # Keep only last 100 composites
            if len(self._track_composites) > 100:
                # Remove oldest entries
                keys_to_remove = list(self._track_composites.keys())[:-100]
                for key in keys_to_remove:
                    del self._track_composites[key]
                    
        except Exception as e:
            logger.debug(f"Failed to store track composite: {e}")

    def _process_frame(self, frame: np.ndarray, quality: StreamQuality) -> np.ndarray:
        """Process frame based on quality setting"""
        try:
            settings = self.quality_settings[quality]

            # Resize frame
            target_width = settings["width"]
            target_height = settings["height"]

            current_height, current_width = frame.shape[:2]
            if current_width != target_width or current_height != target_height:
                frame = cv2.resize(frame, (target_width, target_height))

            return frame

        except Exception as e:
            logger.error(f"Error processing frame: {e}")
            return frame

    def _distribute_frames(self, camera_id: str, processed_frames: Dict[StreamQuality, np.ndarray], frame_timestamp: float = None):
        """Distribute frames to connected clients with ultra-low latency"""
        try:
            stream_info = self.active_streams.get(camera_id)
            if not stream_info:
                return

            # Get current time for latency calculations
            current_time = time.time()
            frame_timestamp = frame_timestamp or current_time

            for client_id in list(stream_info['clients']):
                connection = self.client_connections.get(client_id)
                if not connection:
                    continue

                # Skip WebRTC clients (handled by MediaMTX)
                if connection.type == ConnectionType.WEBRTC:
                    continue

                # Get appropriate frame quality
                frame = processed_frames.get(connection.quality)
                if frame is None:
                    continue

                # Send frame based on connection type
                if connection.type == ConnectionType.WEBSOCKET:
                    asyncio.create_task(self._send_websocket_frame(connection, frame, frame_timestamp))
                elif connection.type == ConnectionType.HTTP:
                    # HTTP streaming handled separately
                    pass

                # Update client statistics
                connection.last_activity = datetime.now()
                connection.frames_sent += 1

        except Exception as e:
            logger.error(f"Error distributing frames for camera {camera_id}: {e}")

    async def _send_websocket_frame(self, connection: ClientConnection, frame: np.ndarray, frame_timestamp: float = None):
        """Send frame via WebSocket with ultra-low latency optimizations"""
        await self._send_websocket_message(connection, "frame", {
            "camera_id": connection.camera_id,
            "data": self._encode_frame_to_base64(frame, connection.quality),
            "timestamp": datetime.now().isoformat(),
            "frame_timestamp": frame_timestamp or time.time(),
            "latency_ms": int((time.time() - (frame_timestamp or time.time())) * 1000)
        }, priority=2)  # Lower priority for regular frames

    async def _send_detection_data(self, connection: ClientConnection, detection_data: Dict[str, Any]):
        """Send detection data with highest priority for minimal latency"""
        # Determine message type based on data structure
        if 'detections' in detection_data:
            message_type = "detections_update"
        elif 'tracks' in detection_data:
            message_type = "tracks_update"
        else:
            message_type = "detection"
            
        await self._send_websocket_message(connection, message_type, detection_data, priority=1)  # Highest priority

    def _encode_frame_to_base64(self, frame: np.ndarray, quality: StreamQuality) -> str:
        """Encode frame to base64 with optimized settings"""
        quality_settings = self.quality_settings[quality]
        jpeg_quality = quality_settings["jpeg_quality"]

        # Use faster encoding settings
        encode_params = [
            cv2.IMWRITE_JPEG_QUALITY, jpeg_quality,
            cv2.IMWRITE_JPEG_OPTIMIZE, 0,  # Disable optimization for speed
            cv2.IMWRITE_JPEG_PROGRESSIVE, 0  # Disable progressive for speed
        ]
        
        _, buffer = cv2.imencode('.jpg', frame, encode_params)
        frame_bytes = buffer.tobytes()
        return f"data:image/jpeg;base64,{base64.b64encode(frame_bytes).decode('utf-8')}"

    async def _send_websocket_message(self, connection: ClientConnection, message_type: str, data: Dict[str, Any], priority: int = 3):
        """Send WebSocket message with priority queue and rate limiting"""
        try:
            if not connection.websocket or connection.websocket.closed:
                return

            # Rate limiting - max 60 messages per second
            current_time = time.time()
            if current_time - connection.last_message_time < 0.016:  # ~60 FPS max
                return

            # Frame age check for frame messages
            if message_type == "frame" and "frame_timestamp" in data:
                if (current_time - data["frame_timestamp"]) > 0.1:
                    return  # Skip old frames

            # Create message
            message = {
                "type": message_type,
                **data,
                "priority": priority,
                "send_timestamp": current_time
            }

            # Check bandwidth limit
            message_size = len(json.dumps(message))
            if connection.bandwidth_limit and (connection.bytes_sent + message_size) > connection.bandwidth_limit:
                return  # Skip message to maintain bandwidth limit

            # Send message immediately for high priority (detection data)
            if priority <= 2:
                await connection.websocket.send(json.dumps(message))
                connection.bytes_sent += message_size
                connection.last_activity = datetime.now()
                connection.last_message_time = current_time
            else:
                # Queue lower priority messages
                heapq.heappush(connection.message_queue, (priority, message))

        except websockets.exceptions.ConnectionClosed:
            logger.warning(f"WebSocket connection {connection.id} closed during message send")
            await self._handle_websocket_disconnection(connection.id)
        except websockets.exceptions.WebSocketException as e:
            logger.error(f"WebSocket error for connection {connection.id}: {e}")
            await self._handle_websocket_disconnection(connection.id)
        except Exception as e:
            logger.error(f"Error sending WebSocket message for connection {connection.id}: {e}")

    async def _broadcast_detection_data(self, camera_id: str, detection_data: Dict[str, Any]):
        """Broadcast detection data to all WebSocket clients for this camera with highest priority"""
        try:
            if camera_id not in self.active_streams:
                return
                
            stream_info = self.active_streams[camera_id]
            for client_id in list(stream_info['clients']):
                connection = self.client_connections.get(client_id)
                if connection and connection.type == ConnectionType.WEBSOCKET:
                    await self._send_detection_data(connection, detection_data)
                    
        except Exception as e:
            logger.error(f"Error broadcasting detection data for camera {camera_id}: {e}")

    def _format_detection_for_frontend(self, camera_id: str, tracks: List[Any]) -> Dict[str, Any]:
        """Format detection data for frontend consumption"""
        detections = []
        for track in tracks:
            if hasattr(track, 'label') and track.label:
                detection = {
                    'id': track.id,
                    'class': track.label,
                    'confidence': getattr(track, 'label_confidence', 0.8),
                    'bbox': {
                        'x': track.bbox[0],
                        'y': track.bbox[1],
                        'w': track.bbox[2],
                        'h': track.bbox[3]
                    },
                    'age': track.age,
                    'center': {
                        'x': track.bbox[0] + track.bbox[2] / 2,
                        'y': track.bbox[1] + track.bbox[3] / 2
                    }
                }
                detections.append(detection)
        
        return {
            'camera_id': camera_id,
            'detections': detections,
            'timestamp': datetime.now().isoformat(),
            'count': len(detections)
        }

    async def _process_message_queues(self):
        """Process queued messages for all connections"""
        while self.running:
            try:
                for connection in self.client_connections.values():
                    if not connection.websocket or connection.websocket.closed:
                        continue

                    # Process up to 5 queued messages per connection
                    messages_sent = 0
                    while connection.message_queue and messages_sent < 5:
                        try:
                            priority, message = heapq.heappop(connection.message_queue)
                            
                            # Check if message is still relevant (not too old)
                            if time.time() - message.get("send_timestamp", 0) > 0.5:
                                continue  # Skip old messages

                            await connection.websocket.send(json.dumps(message))
                            connection.bytes_sent += len(json.dumps(message))
                            connection.last_activity = datetime.now()
                            messages_sent += 1

                        except Exception as e:
                            logger.error(f"Error processing queued message: {e}")
                            break

                await asyncio.sleep(0.01)  # Process every 10ms

            except Exception as e:
                logger.error(f"Error in message queue processor: {e}")
                await asyncio.sleep(0.1)

    async def _handle_websocket_disconnection(self, client_id: str):
        """Handle WebSocket disconnection and cleanup"""
        try:
            if client_id in self.client_connections:
                connection = self.client_connections[client_id]
                logger.info(f"Cleaning up disconnected WebSocket client {client_id}")
                
                # Remove from camera's client list
                if connection.camera_id in self.active_streams:
                    if client_id in self.active_streams[connection.camera_id]['clients']:
                        self.active_streams[connection.camera_id]['clients'].remove(client_id)
                
                # Remove from client connections
                del self.client_connections[client_id]
                
                # Update stats
                if connection.camera_id in self.stream_stats:
                    self.stream_stats[connection.camera_id].websocket_connections = max(0, 
                        self.stream_stats[connection.camera_id].websocket_connections - 1)
                
        except Exception as e:
            logger.error(f"Error handling WebSocket disconnection for {client_id}: {e}")

    async def _check_websocket_connections(self):
        """Periodically check WebSocket connections and clean up dead ones"""
        try:
            dead_connections = []
            for client_id, connection in self.client_connections.items():
                if connection.type == ConnectionType.WEBSOCKET and connection.websocket:
                    try:
                        # Check if connection is still alive
                        if connection.websocket.closed:
                            dead_connections.append(client_id)
                        else:
                            # Send ping to check connection
                            pong_waiter = await connection.websocket.ping()
                            await asyncio.wait_for(pong_waiter, timeout=1.0)
                    except (asyncio.TimeoutError, websockets.exceptions.WebSocketException):
                        dead_connections.append(client_id)
            
            # Clean up dead connections
            for client_id in dead_connections:
                await self._handle_websocket_disconnection(client_id)
                
        except Exception as e:
            logger.error(f"Error checking WebSocket connections: {e}")

    async def _websocket_connection_monitor(self):
        """Background task to monitor WebSocket connections and clean up dead ones"""
        while self.running:
            try:
                await self._check_websocket_connections()
                await asyncio.sleep(30)  # Check every 30 seconds
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in WebSocket connection monitor: {e}")
                await asyncio.sleep(30)  # Wait before retrying

    # HTTP Streaming Support

    def get_frame(self, camera_id: str, quality: StreamQuality = StreamQuality.MEDIUM,
                  format: str = 'jpeg') -> Optional[bytes]:
        """Get latest frame for HTTP streaming"""
        try:
            if camera_id not in self.active_streams:
                return None

            stream_info = self.active_streams[camera_id]
            frame = stream_info['last_frame']

            if frame is None:
                return None

            # Process frame
            processed_frame = self._process_frame(frame, quality)

            # Encode frame
            if format.lower() == 'jpeg':
                quality_settings = self.quality_settings[quality]
                jpeg_quality = quality_settings["jpeg_quality"]
                _, buffer = cv2.imencode('.jpg', processed_frame, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])
                return buffer.tobytes()
            elif format.lower() == 'png':
                _, buffer = cv2.imencode('.png', processed_frame)
                return buffer.tobytes()
            else:
                return None

        except Exception as e:
            logger.error(f"Error getting frame for camera {camera_id}: {e}")
            return None

    def get_frame_base64(self, camera_id: str, quality: StreamQuality = StreamQuality.MEDIUM,
                         format: str = 'jpeg') -> Optional[str]:
        """Get latest frame as base64 encoded string"""
        try:
            frame_bytes = self.get_frame(camera_id, quality, format)
            if frame_bytes:
                base64_string = base64.b64encode(frame_bytes).decode('utf-8')
                return f"data:image/{format};base64,{base64_string}"
            return None

        except Exception as e:
            logger.error(f"Error getting base64 frame for camera {camera_id}: {e}")
            return None

    # Information Methods

    def get_stream_info(self, camera_id: str) -> Optional[Dict[str, Any]]:
        """Get detailed stream information"""
        try:
            if camera_id not in self.active_streams:
                return None

            stream_info = self.active_streams[camera_id]
            stats = self.stream_stats[camera_id]

            return {
                'camera_id': camera_id,
                'active': stream_info['active'],
                'webrtc_enabled': stream_info.get('webrtc_enabled', False),
                'client_count': len(stream_info['clients']),
                'fps': stream_info['fps'],
                'quality': stream_info['quality'].value,
                'queue_size': stream_info['frame_queue'].qsize(),
                'stats': asdict(stats),
                'webrtc_url': self._get_webrtc_url(camera_id) if stream_info.get('webrtc_enabled') else None
            }

        except Exception as e:
            logger.error(f"Error getting stream info for camera {camera_id}: {e}")
            return None

    def get_all_streams(self) -> Dict[str, Dict[str, Any]]:
        """Get information about all active streams"""
        try:
            streams = {}
            for camera_id in self.active_streams:
                stream_info = self.get_stream_info(camera_id)
                if stream_info:
                    streams[camera_id] = stream_info
            return streams

        except Exception as e:
            logger.error(f"Error getting all streams info: {e}")
            return {}

    def get_client_connections(self) -> Dict[str, Dict[str, Any]]:
        """Get information about all client connections"""
        try:
            connections = {}
            for client_id, connection in self.client_connections.items():
                connections[client_id] = {
                    'id': connection.id,
                    'type': connection.type.value,
                    'camera_id': connection.camera_id,
                    'quality': connection.quality.value,
                    'connected_at': connection.connected_at.isoformat(),
                    'last_activity': connection.last_activity.isoformat(),
                    'bytes_sent': connection.bytes_sent,
                    'frames_sent': connection.frames_sent,
                    'ip_address': connection.ip_address,
                    'user_agent': connection.user_agent
                }
            return connections

        except Exception as e:
            logger.error(f"Error getting client connections: {e}")
            return {}

    def get_bandwidth_usage(self) -> Dict[str, Any]:
        """Get current bandwidth usage information"""
        try:
            total_usage = sum(conn.bytes_sent for conn in self.client_connections.values())

            return {
                'total_bandwidth_used': total_usage,
                'max_bandwidth': self.bandwidth_manager.max_total_bandwidth,
                'bandwidth_percentage': (total_usage / self.bandwidth_manager.max_total_bandwidth) * 100,
                'active_clients': len(self.client_connections),
                'client_breakdown': {
                    conn_type.value: len([c for c in self.client_connections.values() if c.type == conn_type])
                    for conn_type in ConnectionType
                }
            }

        except Exception as e:
            logger.error(f"Error getting bandwidth usage: {e}")
            return {}

    def set_detection_config(self, config: Dict[str, Any]) -> bool:
        """Set detection configuration for the stream server"""
        try:
            # Validate required fields
            required_fields = ['tier1_enabled', 'tier2_enabled', 'tier2_min_confidence', 'tier2_max_models', 'tier3_enabled']
            for field in required_fields:
                if field not in config:
                    logger.error(f"Missing required field in detection config: {field}")
                    return False
            
            # Update configuration
            self.detection_config.update(config)
            logger.info(f"Detection configuration updated: {config}")
            return True
            
        except Exception as e:
            logger.error(f"Error setting detection config: {e}")
            return False

    def set_ai_agent(self, ai_agent: Any) -> None:
        """Set AI agent for Tier-3 LLM analysis"""
        self.ai_agent = ai_agent
        logger.info("AI agent set for Tier-3 LLM analysis")

    # Private Helper Methods

    def _get_webrtc_url(self, camera_id: str) -> Optional[str]:
        """Get WebRTC URL for camera"""
        stream_info = self.active_streams.get(camera_id)
        if not stream_info or not stream_info.get('webrtc_enabled'):
            return None

        stream_path = stream_info.get('stream_path')
        if not stream_path:
            return None

        # Return the backend proxy WHEP endpoint (browser-safe; avoids direct :8889 access).
        return f"/proxy/webrtc/{stream_path}/whep"

    def _has_clients_with_quality(self, camera_id: str, quality: StreamQuality) -> bool:
        """Check if any clients need this quality level"""
        stream_info = self.active_streams.get(camera_id)
        if not stream_info:
            return False

        for client_id in stream_info['clients']:
            connection = self.client_connections.get(client_id)
            if connection and connection.quality == quality:
                return True

        return False

    async def _check_bandwidth_availability(self, quality: StreamQuality) -> bool:
        """Check if bandwidth is available for new client"""
        try:
            required_bandwidth = self.bandwidth_manager.quality_thresholds.get(quality.value, 2000000)
            current_usage = sum(conn.bytes_sent for conn in self.client_connections.values())

            if current_usage + required_bandwidth > self.bandwidth_manager.max_total_bandwidth:
                if self.on_bandwidth_exceeded:
                    await self._safe_callback(self.on_bandwidth_exceeded, current_usage, required_bandwidth)
                return False

            return True

        except Exception as e:
            logger.error(f"Error checking bandwidth availability: {e}")
            return True  # Allow connection on error

    def _get_client_bandwidth_limit(self, quality: StreamQuality) -> int:
        """Get bandwidth limit for client based on quality"""
        return self.bandwidth_manager.quality_thresholds.get(
            quality.value,
            self.bandwidth_manager.max_client_bandwidth
        )

    # Background Tasks

    async def _update_statistics(self):
        """Update stream statistics"""
        while self.running:
            try:
                current_time = datetime.now()

                for camera_id, stats in self.stream_stats.items():
                    stats.last_updated = current_time

                    # Calculate current bitrate
                    if camera_id in self.active_streams:
                        stream_info = self.active_streams[camera_id]
                        total_bytes = sum(
                            conn.bytes_sent for conn in self.client_connections.values()
                            if conn.camera_id == camera_id
                        )
                        stats.bytes_transmitted = total_bytes
                        stats.current_bitrate = total_bytes * 8 / (1024 * 1024)  # Mbps

                await asyncio.sleep(10)  # Update every 10 seconds

            except Exception as e:
                logger.error(f"Error updating statistics: {e}")
                await asyncio.sleep(30)

    async def _manage_bandwidth(self):
        """Manage bandwidth usage and quality adaptation"""
        while self.running:
            try:
                if self.bandwidth_manager.adaptive_quality:
                    total_usage = sum(conn.bytes_sent for conn in self.client_connections.values())

                    if total_usage > self.bandwidth_manager.max_total_bandwidth * 0.8:
                        # Reduce quality for some clients
                        await self._adapt_client_quality()

                await asyncio.sleep(5)  # Check every 5 seconds

            except Exception as e:
                logger.error(f"Error managing bandwidth: {e}")
                await asyncio.sleep(30)

    async def _adapt_client_quality(self):
        """Adapt client quality based on bandwidth usage"""
        try:
            # Sort clients by bytes sent (highest first)
            sorted_clients = sorted(
                self.client_connections.values(),
                key=lambda c: c.bytes_sent,
                reverse=True
            )

            for connection in sorted_clients[:5]:  # Adapt top 5 bandwidth users
                if connection.quality == StreamQuality.ULTRA:
                    connection.quality = StreamQuality.HIGH
                elif connection.quality == StreamQuality.HIGH:
                    connection.quality = StreamQuality.MEDIUM
                elif connection.quality == StreamQuality.MEDIUM:
                    connection.quality = StreamQuality.LOW

                connection.bandwidth_limit = self._get_client_bandwidth_limit(connection.quality)

                # Notify client of quality change
                if connection.websocket:
                    try:
                        await connection.websocket.send(json.dumps({
                            "type": "quality_adapted",
                            "new_quality": connection.quality.value,
                            "reason": "bandwidth_management"
                        }))
                    except:
                        pass

        except Exception as e:
            logger.error(f"Error adapting client quality: {e}")

    async def _cleanup_stale_connections(self):
        """Clean up stale connections"""
        while self.running:
            try:
                current_time = datetime.now()
                stale_clients = []

                for client_id, connection in self.client_connections.items():
                    time_since_activity = current_time - connection.last_activity
                    if time_since_activity > timedelta(minutes=5):
                        stale_clients.append(client_id)

                for client_id in stale_clients:
                    logger.info(f"Cleaning up stale client: {client_id}")
                    await self.disconnect_client(client_id)

                await asyncio.sleep(60)  # Check every minute

            except Exception as e:
                logger.error(f"Error cleaning up stale connections: {e}")
                await asyncio.sleep(300)

    async def _safe_callback(self, callback: Callable, *args, **kwargs):
        """Safely execute callback without breaking the main flow"""
        try:
            if asyncio.iscoroutinefunction(callback):
                await callback(*args, **kwargs)
            else:
                callback(*args, **kwargs)
        except Exception as e:
            logger.error(f"Error in callback: {e}")

    # Quality and Configuration Methods

    async def update_stream_quality(self, camera_id: str, quality: StreamQuality) -> bool:
        """Update default stream quality for a camera"""
        try:
            if camera_id not in self.active_streams:
                return False

            self.active_streams[camera_id]['quality'] = quality
            logger.info(f"Updated stream quality for camera {camera_id} to {quality.value}")
            return True

        except Exception as e:
            logger.error(f"Error updating stream quality: {e}")
            return False

    def update_bandwidth_settings(self, settings: Dict[str, Any]) -> bool:
        """Update bandwidth management settings"""
        try:
            if 'max_total_bandwidth' in settings:
                self.bandwidth_manager.max_total_bandwidth = settings['max_total_bandwidth']

            if 'max_client_bandwidth' in settings:
                self.bandwidth_manager.max_client_bandwidth = settings['max_client_bandwidth']

            if 'adaptive_quality' in settings:
                self.bandwidth_manager.adaptive_quality = settings['adaptive_quality']

            if 'quality_thresholds' in settings:
                self.bandwidth_manager.quality_thresholds.update(settings['quality_thresholds'])

            logger.info("Updated bandwidth management settings")
            return True

        except Exception as e:
            logger.error(f"Error updating bandwidth settings: {e}")
            return False

