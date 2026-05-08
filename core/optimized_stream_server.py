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
import heapq
from collections import defaultdict

from .mediamtx_client import MediaMTXWebRTCClient, StreamConfig
from .motion import MotionDetector, MotionResult
from .sort_tracker import SortTracker as SimpleSortTracker
from .snapshots import build_track_composite

logger = logging.getLogger(__name__)


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
class OptimizedClientConnection:
    """Optimized client connection information"""
    id: str
    type: ConnectionType
    camera_id: str
    quality: StreamQuality
    connected_at: datetime
    last_activity: datetime
    bytes_sent: int = 0
    frames_sent: int = 0
    bandwidth_limit: Optional[int] = None
    websocket: Optional[websockets.WebSocketServerProtocol] = None
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    message_queue: List[Tuple[int, Any]] = None
    last_message_time: float = 0.0
    reconnect_attempts: int = 0
    max_reconnect_attempts: int = 5
    is_reconnecting: bool = False
    connection_start_time: float = 0.0
    latency_ms: float = 0.0
    packet_loss: float = 0.0
    bitrate: float = 0.0

    def __post_init__(self):
        if self.message_queue is None:
            self.message_queue = []
        if self.connection_start_time == 0.0:
            self.connection_start_time = time.time()


@dataclass
class OptimizedStreamStats:
    """Optimized stream statistics"""
    camera_id: str
    active_clients: int = 0
    total_clients_served: int = 0
    frames_processed: int = 0
    bytes_transmitted: int = 0
    average_fps: float = 0.0
    current_bitrate: float = 0.0
    webrtc_connections: int = 0
    websocket_connections: int = 0
    http_connections: int = 0
    last_updated: datetime = None
    average_latency: float = 0.0
    total_packet_loss: float = 0.0
    connection_uptime: float = 0.0

    def __post_init__(self):
        if self.last_updated is None:
            self.last_updated = datetime.now()


@dataclass
class OptimizedBandwidthManager:
    """Optimized bandwidth management"""
    max_total_bandwidth: int = 100 * 1024 * 1024  # 100 Mbps
    max_client_bandwidth: int = 10 * 1024 * 1024  # 10 Mbps per client
    adaptive_quality: bool = True
    quality_thresholds: Dict[str, int] = None
    current_usage: int = 0
    client_usage: Dict[str, int] = None
    bandwidth_history: List[Tuple[float, int]] = None

    def __post_init__(self):
        if self.quality_thresholds is None:
            self.quality_thresholds = {
                "ultra": 8 * 1024 * 1024,  # 8 Mbps
                "high": 4 * 1024 * 1024,  # 4 Mbps
                "medium": 2 * 1024 * 1024,  # 2 Mbps
                "low": 500 * 1024  # 500 Kbps
            }
        if self.client_usage is None:
            self.client_usage = {}
        if self.bandwidth_history is None:
            self.bandwidth_history = []


class OptimizedStreamServer:
    """
    Optimized stream server with enhanced WebRTC capabilities and connection management
    """

    def __init__(self,
                 mediamtx_host: str = "localhost",
                 mediamtx_webrtc_port: int = 8889,
                 mediamtx_api_port: int = 9997,
                 websocket_port: int = 8765,
                 http_port: int = 8080):
        """
        Initialize Optimized Stream Server
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

        # Optimized stream management
        self.active_streams: Dict[str, Dict[str, Any]] = {}
        self.client_connections: Dict[str, OptimizedClientConnection] = {}
        self.stream_stats: Dict[str, OptimizedStreamStats] = {}
        self.signaling_server: Optional[websockets.WebSocketServer] = None

        # Connection pooling and management
        self.connection_pool: Dict[str, Set[str]] = defaultdict(set)  # camera_id -> client_ids
        self.client_priorities: Dict[str, int] = {}  # client_id -> priority
        self.connection_queue: List[Tuple[str, str, int, float]] = []  # (client_id, camera_id, priority, timestamp)
        self.max_concurrent_connections = 50
        self.current_connections = 0

        # Motion detection per camera
        self._motion_detectors: Dict[str, MotionDetector] = {}
        self._motion_state: Dict[str, MotionResult] = {}
        self._motion_params: Dict[str, Dict[str, float]] = {}
        self._trackers: Dict[str, SimpleSortTracker] = {}
        self._tracks_state: Dict[str, List[Dict[str, Any]]] = {}
        self._track_prev_ids: Dict[str, Set[int]] = {}
        self._track_events: Dict[str, List[Dict[str, Any]]] = {}
        self._track_trajectories: Dict[str, Dict[int, Dict[str, Any]]] = {}
        self._llm_cache: Dict[str, List[Dict[str, Any]]] = {}

        # Optimized bandwidth management
        self.bandwidth_manager = OptimizedBandwidthManager()
        self.bandwidth_monitor_interval = 5.0  # seconds
        self.last_bandwidth_check = time.time()

        # Quality settings with optimized parameters
        self.quality_settings = {
            StreamQuality.LOW: {
                "width": 640, "height": 480, "fps": 15, "bitrate": 500000,
                "jpeg_quality": 60, "h264_crf": 28, "keyframe_interval": 2
            },
            StreamQuality.MEDIUM: {
                "width": 1280, "height": 720, "fps": 25, "bitrate": 2000000,
                "jpeg_quality": 75, "h264_crf": 23, "keyframe_interval": 2
            },
            StreamQuality.HIGH: {
                "width": 1920, "height": 1080, "fps": 30, "bitrate": 4000000,
                "jpeg_quality": 85, "h264_crf": 18, "keyframe_interval": 1
            },
            StreamQuality.ULTRA: {
                "width": 3840, "height": 2160, "fps": 30, "bitrate": 8000000,
                "jpeg_quality": 95, "h264_crf": 15, "keyframe_interval": 1
            }
        }

        # Event callbacks
        self.on_client_connected: Optional[Callable] = None
        self.on_client_disconnected: Optional[Callable] = None
        self.on_stream_started: Optional[Callable] = None
        self.on_stream_stopped: Optional[Callable] = None
        self.on_bandwidth_exceeded: Optional[Callable] = None
        self.on_motion_update: Optional[Callable[[str, Dict[str, Any]], None]] = None
        self.on_tracks_update: Optional[Callable[[str, Dict[str, Any]], None]] = None
        self.on_detection_update: Optional[Callable[[str, Dict[str, Any]], None]] = None

        # AI components
        self.ai_verifier: Optional[Any] = None
        self.ai_agent: Optional[Any] = None

        # Detection configuration
        self.detection_config: Dict[str, Any] = {
            "motion_enabled": True,
            "motion_sensitivity": 0.3,
            "motion_threshold": 25,
            "tracking_enabled": True,
            "tracking_confidence": 0.5,
            "detection_classes": ["person", "car", "truck", "bicycle", "motorcycle"],
            "max_tracks": 50,
            "track_timeout": 30.0
        }

        # Performance monitoring
        self.performance_stats = {
            "total_connections": 0,
            "active_connections": 0,
            "failed_connections": 0,
            "average_latency": 0.0,
            "total_bandwidth_usage": 0,
            "peak_bandwidth_usage": 0,
            "connection_errors": 0
        }

        # Background tasks
        self._monitor_task: Optional[asyncio.Task] = None
        self._cleanup_task: Optional[asyncio.Task] = None
        self._stats_task: Optional[asyncio.Task] = None

    async def start(self):
        """Start the optimized stream server"""
        if self.running:
            return

        logger.info("Starting Optimized Stream Server...")
        self.running = True

        # Start MediaMTX client
        await self.mediamtx_client.start()

        # Start background tasks
        self._monitor_task = asyncio.create_task(self._monitor_connections())
        self._cleanup_task = asyncio.create_task(self._cleanup_inactive_connections())
        self._stats_task = asyncio.create_task(self._update_performance_stats())

        # Start WebSocket server
        await self._start_websocket_server()

        logger.info("Optimized Stream Server started successfully")

    async def stop(self):
        """Stop the optimized stream server"""
        if not self.running:
            return

        logger.info("Stopping Optimized Stream Server...")
        self.running = False

        # Cancel background tasks
        if self._monitor_task:
            self._monitor_task.cancel()
        if self._cleanup_task:
            self._cleanup_task.cancel()
        if self._stats_task:
            self._stats_task.cancel()

        # Stop WebSocket server
        if self.signaling_server:
            self.signaling_server.close()
            await self.signaling_server.wait_closed()

        # Stop MediaMTX client
        await self.mediamtx_client.stop()

        # Clean up connections
        await self._cleanup_all_connections()

        logger.info("Optimized Stream Server stopped")

    async def _start_websocket_server(self):
        """Start the WebSocket signaling server"""
        try:
            self.signaling_server = await websockets.serve(
                self._handle_websocket_connection,
                "0.0.0.0",
                self.websocket_port,
                ping_interval=10,
                ping_timeout=5,
                max_size=1024 * 1024,  # 1MB max message size
                max_queue=1000
            )
            logger.info(f"WebSocket server started on port {self.websocket_port}")
        except Exception as e:
            logger.error(f"Failed to start WebSocket server: {e}")
            raise

    async def _handle_websocket_connection(self, websocket: websockets.WebSocketServerProtocol, path: str):
        """Handle WebSocket connection with optimized connection management"""
        client_id = str(uuid.uuid4())
        client_info = {
            "ip_address": websocket.remote_address[0] if websocket.remote_address else "unknown",
            "user_agent": "WebRTC Client",
            "connected_at": datetime.now()
        }

        logger.info(f"New WebSocket connection: {client_id} from {client_info['ip_address']}")

        try:
            async for message in websocket:
                try:
                    data = json.loads(message)
                    response = await self._process_optimized_message(data, websocket, client_id, client_info)
                    
                    if response:
                        await websocket.send(json.dumps(response))
                        
                except json.JSONDecodeError:
                    logger.warning(f"Invalid JSON from client {client_id}")
                    await websocket.send(json.dumps({"type": "error", "message": "Invalid JSON"}))
                except Exception as e:
                    logger.error(f"Error processing message from {client_id}: {e}")
                    await websocket.send(json.dumps({"type": "error", "message": "Internal server error"}))

        except websockets.exceptions.ConnectionClosed:
            logger.info(f"WebSocket connection closed: {client_id}")
        except Exception as e:
            logger.error(f"WebSocket error for {client_id}: {e}")
        finally:
            await self._disconnect_client(client_id)

    async def _process_optimized_message(self, data: Dict[str, Any], 
                                       websocket: websockets.WebSocketServerProtocol,
                                       client_id: str, client_info: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Process WebSocket message with optimized handling"""
        message_type = data.get("type")

        if message_type == "webrtc_offer":
            return await self._handle_webrtc_offer(data, websocket, client_id, client_info)
        elif message_type == "webrtc_ice_candidate":
            return await self._handle_webrtc_ice_candidate(data, client_id)
        elif message_type == "disconnect_webrtc":
            return await self._handle_disconnect_webrtc(data, client_id)
        elif message_type == "heartbeat":
            return await self._handle_heartbeat(data, client_id)
        elif message_type == "get_stats":
            return await self._handle_get_stats(data, client_id)
        else:
            return {"type": "error", "message": f"Unknown message type: {message_type}"}

    async def _handle_webrtc_offer(self, data: Dict[str, Any], 
                                 websocket: websockets.WebSocketServerProtocol,
                                 client_id: str, client_info: Dict[str, Any]) -> Dict[str, Any]:
        """Handle WebRTC offer with optimized connection management"""
        camera_id = data.get("camera_id")
        offer = data.get("offer")
        quality = StreamQuality(data.get("quality", "medium"))

        if not camera_id or not offer:
            return {"type": "error", "message": "Missing camera_id or offer"}

        # Check if camera is available
        if camera_id not in self.active_streams:
            return {"type": "error", "message": f"Camera {camera_id} not available"}

        # Check connection limits
        if not await self._can_accept_connection(camera_id, quality):
            return {"type": "error", "message": "Connection limit reached or bandwidth exceeded"}

        try:
            # Create optimized client connection
            connection = OptimizedClientConnection(
                id=client_id,
                type=ConnectionType.WEBRTC,
                camera_id=camera_id,
                quality=quality,
                connected_at=datetime.now(),
                last_activity=datetime.now(),
                websocket=websocket,
                ip_address=client_info["ip_address"],
                user_agent=client_info["user_agent"],
                bandwidth_limit=self._get_bandwidth_limit_for_quality(quality)
            )

            self.client_connections[client_id] = connection
            self.connection_pool[camera_id].add(client_id)
            self.current_connections += 1

            # Update statistics
            if camera_id not in self.stream_stats:
                self.stream_stats[camera_id] = OptimizedStreamStats(camera_id=camera_id)
            
            stats = self.stream_stats[camera_id]
            stats.active_clients += 1
            stats.total_clients_served += 1
            stats.webrtc_connections += 1

            # Process WebRTC offer
            answer = await self._process_webrtc_offer(camera_id, offer, quality, client_id=client_id)

            # Update connection activity
            connection.last_activity = datetime.now()
            connection.connection_start_time = time.time()

            logger.info(f"WebRTC connection established: {client_id} -> {camera_id}")

            return {
                "type": "webrtc_answer",
                "camera_id": camera_id,
                "answer": answer,
                "client_id": client_id
            }

        except Exception as e:
            logger.error(f"Failed to handle WebRTC offer for {camera_id}: {e}")
            await self._disconnect_client(client_id)
            return {"type": "error", "message": "Failed to establish WebRTC connection"}

    async def _can_accept_connection(self, camera_id: str, quality: StreamQuality) -> bool:
        """Check if we can accept a new connection"""
        # Check concurrent connection limit
        if self.current_connections >= self.max_concurrent_connections:
            logger.warning(f"Max concurrent connections reached: {self.current_connections}")
            return False

        # Check bandwidth limits
        required_bandwidth = self._get_bandwidth_for_quality(quality)
        if self.bandwidth_manager.current_usage + required_bandwidth > self.bandwidth_manager.max_total_bandwidth:
            logger.warning(f"Bandwidth limit exceeded: {self.bandwidth_manager.current_usage + required_bandwidth}")
            return False

        return True

    def _get_bandwidth_for_quality(self, quality: StreamQuality) -> int:
        """Get bandwidth requirement for quality level"""
        return self.bandwidth_manager.quality_thresholds.get(quality.value, 2 * 1024 * 1024)

    def _get_bandwidth_limit_for_quality(self, quality: StreamQuality) -> int:
        """Get bandwidth limit for quality level"""
        return min(
            self._get_bandwidth_for_quality(quality),
            self.bandwidth_manager.max_client_bandwidth
        )

    async def _process_webrtc_offer(self, camera_id: str, offer: Dict[str, Any], quality: StreamQuality, client_id: str) -> Dict[str, Any]:
        """Process WebRTC offer and create answer"""
        try:
            # Get camera stream info
            stream_info = self.active_streams[camera_id]
            
            # Create WebRTC answer using MediaMTX
            answer = await self.mediamtx_client.create_webrtc_answer(
                stream_path=stream_info.get('stream_path', camera_id),
                offer=offer,
                client_id=client_id
            )
            
            return answer
        except Exception as e:
            logger.error(f"Failed to process WebRTC offer for {camera_id}: {e}")
            raise

    async def _handle_webrtc_ice_candidate(self, data: Dict[str, Any], client_id: str) -> Dict[str, Any]:
        """Handle WebRTC ICE candidate"""
        try:
            camera_id = data.get("camera_id")
            candidate = data.get("candidate")

            if not camera_id or not candidate:
                return {"type": "error", "message": "Missing camera_id or candidate"}

            # Forward ICE candidate to MediaMTX
            await self.mediamtx_client.add_ice_candidate(
                stream_path=camera_id,
                candidate=candidate
                ,client_id=client_id
            )

            return {"type": "success", "message": "ICE candidate processed"}

        except Exception as e:
            logger.error(f"Failed to handle ICE candidate: {e}")
            return {"type": "error", "message": "Failed to process ICE candidate"}

    async def _handle_disconnect_webrtc(self, data: Dict[str, Any], client_id: str) -> Dict[str, Any]:
        """Handle WebRTC disconnect"""
        camera_id = data.get("camera_id")
        
        if camera_id:
            # Best-effort close WHEP session on MediaMTX (browser viewer session)
            try:
                await self.mediamtx_client.close_webrtc_session(stream_path=camera_id, client_id=client_id)
            except Exception:
                pass
            await self._disconnect_client_from_camera(client_id, camera_id)
        
        return {"type": "success", "message": "Disconnected"}

    async def _handle_heartbeat(self, data: Dict[str, Any], client_id: str) -> Dict[str, Any]:
        """Handle client heartbeat"""
        connection = self.client_connections.get(client_id)
        if connection:
            connection.last_activity = datetime.now()
            
            # Update latency if provided
            if "latency" in data:
                connection.latency_ms = data["latency"]
            
            # Update packet loss if provided
            if "packet_loss" in data:
                connection.packet_loss = data["packet_loss"]
            
            # Update bitrate if provided
            if "bitrate" in data:
                connection.bitrate = data["bitrate"]

        return {"type": "heartbeat_ack", "timestamp": time.time()}

    async def _handle_get_stats(self, data: Dict[str, Any], client_id: str) -> Dict[str, Any]:
        """Handle stats request"""
        camera_id = data.get("camera_id")
        
        if camera_id and camera_id in self.stream_stats:
            stats = self.stream_stats[camera_id]
            return {
                "type": "stats",
                "camera_id": camera_id,
                "stats": {
                    "active_clients": stats.active_clients,
                    "total_clients_served": stats.total_clients_served,
                    "average_fps": stats.average_fps,
                    "current_bitrate": stats.current_bitrate,
                    "average_latency": stats.average_latency,
                    "total_packet_loss": stats.total_packet_loss,
                    "connection_uptime": stats.connection_uptime
                }
            }
        
        return {"type": "error", "message": "Camera not found"}

    async def _disconnect_client(self, client_id: str):
        """Disconnect a client"""
        connection = self.client_connections.get(client_id)
        if connection:
            camera_id = connection.camera_id
            
            # Remove from connection pool
            if camera_id in self.connection_pool:
                self.connection_pool[camera_id].discard(client_id)
            
            # Update statistics
            if camera_id in self.stream_stats:
                stats = self.stream_stats[camera_id]
                stats.active_clients = max(0, stats.active_clients - 1)
                if connection.type == ConnectionType.WEBRTC:
                    stats.webrtc_connections = max(0, stats.webrtc_connections - 1)
            
            # Update bandwidth usage
            bandwidth = self._get_bandwidth_for_quality(connection.quality)
            self.bandwidth_manager.current_usage = max(0, self.bandwidth_manager.current_usage - bandwidth)
            
            # Remove client connection
            del self.client_connections[client_id]
            self.current_connections = max(0, self.current_connections - 1)
            
            logger.info(f"Client disconnected: {client_id} from {camera_id}")

    async def _disconnect_client_from_camera(self, client_id: str, camera_id: str):
        """Disconnect a client from a specific camera"""
        connection = self.client_connections.get(client_id)
        if connection and connection.camera_id == camera_id:
            await self._disconnect_client(client_id)

    async def _monitor_connections(self):
        """Monitor active connections for health and performance"""
        while self.running:
            try:
                current_time = time.time()
                
                # Check for inactive connections
                inactive_clients = []
                for client_id, connection in self.client_connections.items():
                    if current_time - connection.last_activity.timestamp() > 30:  # 30 second timeout
                        inactive_clients.append(client_id)
                
                # Disconnect inactive clients
                for client_id in inactive_clients:
                    await self._disconnect_client(client_id)
                
                # Update bandwidth usage
                await self._update_bandwidth_usage()
                
                # Process connection queue
                await self._process_connection_queue()
                
                await asyncio.sleep(10)  # Check every 10 seconds
                
            except Exception as e:
                logger.error(f"Error in connection monitor: {e}")
                await asyncio.sleep(5)

    async def _cleanup_inactive_connections(self):
        """Clean up inactive connections"""
        while self.running:
            try:
                current_time = time.time()
                
                # Clean up connections inactive for more than 5 minutes
                inactive_clients = []
                for client_id, connection in self.client_connections.items():
                    if current_time - connection.last_activity.timestamp() > 300:  # 5 minutes
                        inactive_clients.append(client_id)
                
                for client_id in inactive_clients:
                    await self._disconnect_client(client_id)
                
                await asyncio.sleep(60)  # Check every minute
                
            except Exception as e:
                logger.error(f"Error in cleanup task: {e}")
                await asyncio.sleep(30)

    async def _update_performance_stats(self):
        """Update performance statistics"""
        while self.running:
            try:
                # Calculate average latency
                total_latency = 0
                active_connections = 0
                
                for connection in self.client_connections.values():
                    if connection.latency_ms > 0:
                        total_latency += connection.latency_ms
                        active_connections += 1
                
                if active_connections > 0:
                    self.performance_stats["average_latency"] = total_latency / active_connections
                
                # Update bandwidth peak
                if self.bandwidth_manager.current_usage > self.performance_stats["peak_bandwidth_usage"]:
                    self.performance_stats["peak_bandwidth_usage"] = self.bandwidth_manager.current_usage
                
                # Update connection stats
                self.performance_stats["active_connections"] = self.current_connections
                
                await asyncio.sleep(30)  # Update every 30 seconds
                
            except Exception as e:
                logger.error(f"Error updating performance stats: {e}")
                await asyncio.sleep(30)

    async def _update_bandwidth_usage(self):
        """Update bandwidth usage statistics"""
        current_time = time.time()
        
        # Add current usage to history
        self.bandwidth_manager.bandwidth_history.append((current_time, self.bandwidth_manager.current_usage))
        
        # Keep only last hour of history
        cutoff_time = current_time - 3600
        self.bandwidth_manager.bandwidth_history = [
            (t, usage) for t, usage in self.bandwidth_manager.bandwidth_history 
            if t > cutoff_time
        ]
        
        # Calculate average usage
        if self.bandwidth_manager.bandwidth_history:
            total_usage = sum(usage for _, usage in self.bandwidth_manager.bandwidth_history)
            avg_usage = total_usage / len(self.bandwidth_manager.bandwidth_history)
            self.performance_stats["total_bandwidth_usage"] = avg_usage

    async def _process_connection_queue(self):
        """Process queued connections"""
        if not self.connection_queue:
            return
        
        # Sort queue by priority and timestamp
        self.connection_queue.sort(key=lambda x: (-x[2], x[3]))  # Priority desc, timestamp asc
        
        # Process available slots
        available_slots = self.max_concurrent_connections - self.current_connections
        if available_slots <= 0:
            return
        
        processed = 0
        remaining_queue = []
        
        for client_id, camera_id, priority, timestamp in self.connection_queue:
            if processed >= available_slots:
                remaining_queue.append((client_id, camera_id, priority, timestamp))
                continue
            
            # Check if we can still accept this connection
            if await self._can_accept_connection(camera_id, StreamQuality.MEDIUM):
                # Re-establish connection (this would need to be implemented based on your specific needs)
                processed += 1
            else:
                remaining_queue.append((client_id, camera_id, priority, timestamp))
        
        self.connection_queue = remaining_queue

    async def _cleanup_all_connections(self):
        """Clean up all active connections"""
        client_ids = list(self.client_connections.keys())
        for client_id in client_ids:
            await self._disconnect_client(client_id)

    def get_performance_stats(self) -> Dict[str, Any]:
        """Get current performance statistics"""
        return {
            **self.performance_stats,
            "bandwidth_usage": self.bandwidth_manager.current_usage,
            "bandwidth_limit": self.bandwidth_manager.max_total_bandwidth,
            "connection_pool_size": len(self.connection_pool),
            "queue_size": len(self.connection_queue)
        }

    def get_stream_stats(self, camera_id: str) -> Optional[OptimizedStreamStats]:
        """Get statistics for a specific camera stream"""
        return self.stream_stats.get(camera_id)

    def get_all_stream_stats(self) -> Dict[str, OptimizedStreamStats]:
        """Get statistics for all camera streams"""
        return self.stream_stats.copy()
