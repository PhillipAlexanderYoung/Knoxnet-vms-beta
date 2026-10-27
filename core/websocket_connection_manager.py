"""
WebSocket Connection Manager for Camera Motion Detection

This module provides robust websocket connection management for cameras with motion detection enabled.
It includes auto-reconnection, health monitoring, activity tracking, and resource prioritization.
"""

import asyncio
import threading
import time
import logging
from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import json
import queue
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger(__name__)

class ConnectionStatus(Enum):
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    RECONNECTING = "reconnecting"
    ERROR = "error"

class CameraPriority(Enum):
    HIGH = "high"      # Active motion detection, multiple viewers
    MEDIUM = "medium"  # Motion detection enabled, some activity
    LOW = "low"        # Motion detection enabled, minimal activity
    IDLE = "idle"      # No motion detection or very low activity

@dataclass
class ConnectionMetrics:
    """Connection health and performance metrics"""
    last_connected: Optional[datetime] = None
    last_disconnected: Optional[datetime] = None
    connection_attempts: int = 0
    successful_connections: int = 0
    failed_connections: int = 0
    total_uptime: float = 0.0  # seconds
    last_heartbeat: Optional[datetime] = None
    latency_ms: float = 0.0
    packet_loss_rate: float = 0.0
    bandwidth_usage: float = 0.0  # kbps
    motion_events_count: int = 0
    last_motion_event: Optional[datetime] = None
    viewer_count: int = 0
    error_count: int = 0
    last_error: Optional[str] = None

@dataclass
class CameraConnection:
    """Individual camera connection state"""
    camera_id: str
    camera_name: str
    status: ConnectionStatus = ConnectionStatus.DISCONNECTED
    priority: CameraPriority = CameraPriority.LOW
    metrics: ConnectionMetrics = field(default_factory=ConnectionMetrics)
    websocket: Optional[Any] = None
    reconnect_attempts: int = 0
    max_reconnect_attempts: int = 10
    reconnect_delay: float = 1.0
    max_reconnect_delay: float = 60.0
    last_reconnect_attempt: Optional[datetime] = None
    connection_timeout: float = 30.0
    heartbeat_interval: float = 30.0
    motion_detection_enabled: bool = True
    auto_reconnect: bool = True
    is_active: bool = False
    last_activity: Optional[datetime] = None
    connection_start_time: Optional[datetime] = None

class WebSocketConnectionManager:
    """
    Manages robust websocket connections for cameras with motion detection.
    
    Features:
    - Auto-reconnection with exponential backoff
    - Connection health monitoring
    - Activity-based resource prioritization
    - Motion detection event tracking
    - Connection pooling and load balancing
    """
    
    def __init__(self, 
                 socketio_instance,
                 max_concurrent_connections: int = 50,
                 health_check_interval: float = 60.0,
                 activity_update_interval: float = 30.0):
        """
        Initialize the WebSocket Connection Manager
        
        Args:
            socketio_instance: SocketIO instance for websocket operations
            max_concurrent_connections: Maximum number of concurrent connections
            health_check_interval: Interval for health checks (seconds)
            activity_update_interval: Interval for activity updates (seconds)
        """
        self.socketio = socketio_instance
        self.max_concurrent_connections = max_concurrent_connections
        self.health_check_interval = health_check_interval
        self.activity_update_interval = activity_update_interval
        
        # Connection storage
        self.connections: Dict[str, CameraConnection] = {}
        self.connection_queue = queue.PriorityQueue()
        self.active_connections: List[str] = []
        
        # Threading and async
        self.running = False
        self.manager_thread: Optional[threading.Thread] = None
        self.health_check_thread: Optional[threading.Thread] = None
        self.activity_thread: Optional[threading.Thread] = None
        self.executor = ThreadPoolExecutor(max_workers=10, thread_name_prefix="WSManager")
        
        # Event callbacks
        self.on_connection_established: Optional[Callable] = None
        self.on_connection_lost: Optional[Callable] = None
        self.on_motion_detected: Optional[Callable] = None
        self.on_health_status: Optional[Callable] = None
        
        # Statistics
        self.total_connections = 0
        self.successful_reconnections = 0
        self.failed_reconnections = 0
        self.start_time = datetime.now()
        
        logger.info("WebSocket Connection Manager initialized")
    
    def start(self):
        """Start the connection manager"""
        if self.running:
            logger.warning("Connection manager already running")
            return
        
        self.running = True
        
        # Start management threads
        self.manager_thread = threading.Thread(target=self._connection_manager_loop, daemon=True)
        self.manager_thread.start()
        
        self.health_check_thread = threading.Thread(target=self._health_check_loop, daemon=True)
        self.health_check_thread.start()
        
        self.activity_thread = threading.Thread(target=self._activity_monitor_loop, daemon=True)
        self.activity_thread.start()
        
        logger.info("WebSocket Connection Manager started")
    
    def stop(self):
        """Stop the connection manager"""
        self.running = False
        
        # Close all connections
        for camera_id in list(self.connections.keys()):
            self.disconnect_camera(camera_id)
        
        # Shutdown executor
        self.executor.shutdown(wait=True)
        
        logger.info("WebSocket Connection Manager stopped")
    
    def add_camera(self, 
                   camera_id: str, 
                   camera_name: str, 
                   motion_detection_enabled: bool = True,
                   priority: CameraPriority = CameraPriority.MEDIUM) -> bool:
        """
        Add a camera to the connection manager
        
        Args:
            camera_id: Unique camera identifier
            camera_name: Human-readable camera name
            motion_detection_enabled: Whether motion detection is enabled
            priority: Initial priority level
            
        Returns:
            True if camera was added successfully
        """
        try:
            if camera_id in self.connections:
                logger.warning(f"Camera {camera_id} already exists in connection manager")
                return False
            
            connection = CameraConnection(
                camera_id=camera_id,
                camera_name=camera_name,
                motion_detection_enabled=motion_detection_enabled,
                priority=priority
            )
            
            self.connections[camera_id] = connection
            
            # Add to connection queue if motion detection is enabled
            if motion_detection_enabled:
                self._add_to_connection_queue(camera_id, priority)
            
            logger.info(f"Added camera {camera_name} ({camera_id}) to connection manager")
            return True
            
        except Exception as e:
            logger.error(f"Error adding camera {camera_id}: {e}")
            return False
    
    def remove_camera(self, camera_id: str) -> bool:
        """Remove a camera from the connection manager"""
        try:
            if camera_id not in self.connections:
                return False
            
            # Disconnect if connected
            self.disconnect_camera(camera_id)
            
            # Remove from storage
            del self.connections[camera_id]
            
            # Remove from active connections
            if camera_id in self.active_connections:
                self.active_connections.remove(camera_id)
            
            logger.info(f"Removed camera {camera_id} from connection manager")
            return True
            
        except Exception as e:
            logger.error(f"Error removing camera {camera_id}: {e}")
            return False
    
    def connect_camera(self, camera_id: str) -> bool:
        """Connect a specific camera"""
        try:
            if camera_id not in self.connections:
                logger.error(f"Camera {camera_id} not found in connection manager")
                return False
            
            connection = self.connections[camera_id]
            
            if connection.status == ConnectionStatus.CONNECTED:
                logger.debug(f"Camera {camera_id} already connected")
                return True
            
            # Update status
            connection.status = ConnectionStatus.CONNECTING
            connection.reconnect_attempts += 1
            connection.last_reconnect_attempt = datetime.now()
            
            # Attempt connection
            success = self._establish_connection(connection)
            
            if success:
                connection.status = ConnectionStatus.CONNECTED
                connection.metrics.successful_connections += 1
                connection.connection_start_time = datetime.now()
                connection.metrics.last_connected = datetime.now()
                connection.reconnect_attempts = 0
                
                # Add to active connections
                if camera_id not in self.active_connections:
                    self.active_connections.append(camera_id)
                
                # Notify callback
                if self.on_connection_established:
                    self.on_connection_established(camera_id, connection)
                
                logger.info(f"Successfully connected camera {connection.camera_name} ({camera_id})")
                return True
            else:
                connection.status = ConnectionStatus.ERROR
                connection.metrics.failed_connections += 1
                connection.metrics.last_error = "Connection failed"
                
                logger.error(f"Failed to connect camera {connection.camera_name} ({camera_id})")
                return False
                
        except Exception as e:
            logger.error(f"Error connecting camera {camera_id}: {e}")
            if camera_id in self.connections:
                self.connections[camera_id].status = ConnectionStatus.ERROR
                self.connections[camera_id].last_error = str(e)
            return False
    
    def disconnect_camera(self, camera_id: str) -> bool:
        """Disconnect a specific camera"""
        try:
            if camera_id not in self.connections:
                return False
            
            connection = self.connections[camera_id]
            
            if connection.status == ConnectionStatus.DISCONNECTED:
                return True
            
            # Close websocket if exists
            if connection.websocket:
                try:
                    connection.websocket.close()
                except:
                    pass
                connection.websocket = None
            
            # Update status
            connection.status = ConnectionStatus.DISCONNECTED
            connection.metrics.last_disconnected = datetime.now()
            
            # Calculate uptime
            if connection.connection_start_time:
                uptime = (datetime.now() - connection.connection_start_time).total_seconds()
                connection.metrics.total_uptime += uptime
            
            # Remove from active connections
            if camera_id in self.active_connections:
                self.active_connections.remove(camera_id)
            
            # Notify callback
            if self.on_connection_lost:
                self.on_connection_lost(camera_id, connection)
            
            logger.info(f"Disconnected camera {connection.camera_name} ({camera_id})")
            return True
            
        except Exception as e:
            logger.error(f"Error disconnecting camera {camera_id}: {e}")
            return False
    
    def update_camera_activity(self, camera_id: str, activity_data: Dict[str, Any]):
        """
        Update camera activity metrics
        
        Args:
            camera_id: Camera identifier
            activity_data: Activity data including motion events, viewer count, etc.
        """
        try:
            if camera_id not in self.connections:
                return
            
            connection = self.connections[camera_id]
            connection.last_activity = datetime.now()
            
            # Update metrics
            if 'motion_detected' in activity_data and activity_data['motion_detected']:
                connection.metrics.motion_events_count += 1
                connection.metrics.last_motion_event = datetime.now()
                connection.is_active = True
            
            if 'viewer_count' in activity_data:
                connection.metrics.viewer_count = activity_data['viewer_count']
            
            if 'latency_ms' in activity_data:
                connection.metrics.latency_ms = activity_data['latency_ms']
            
            # Update priority based on activity
            self._update_camera_priority(connection)
            
        except Exception as e:
            logger.error(f"Error updating activity for camera {camera_id}: {e}")
    
    def get_connection_status(self, camera_id: str) -> Optional[Dict[str, Any]]:
        """Get detailed connection status for a camera"""
        try:
            if camera_id not in self.connections:
                return None
            
            connection = self.connections[camera_id]
            
            return {
                'camera_id': camera_id,
                'camera_name': connection.camera_name,
                'status': connection.status.value,
                'priority': connection.priority.value,
                'motion_detection_enabled': connection.motion_detection_enabled,
                'auto_reconnect': connection.auto_reconnect,
                'is_active': connection.is_active,
                'metrics': {
                    'connection_attempts': connection.metrics.connection_attempts,
                    'successful_connections': connection.metrics.successful_connections,
                    'failed_connections': connection.metrics.failed_connections,
                    'total_uptime': connection.metrics.total_uptime,
                    'motion_events_count': connection.metrics.motion_events_count,
                    'viewer_count': connection.metrics.viewer_count,
                    'error_count': connection.metrics.error_count,
                    'last_motion_event': connection.metrics.last_motion_event.isoformat() if connection.metrics.last_motion_event else None,
                    'last_connected': connection.metrics.last_connected.isoformat() if connection.metrics.last_connected else None,
                    'last_disconnected': connection.metrics.last_disconnected.isoformat() if connection.metrics.last_disconnected else None,
                    'latency_ms': connection.metrics.latency_ms,
                    'packet_loss_rate': connection.metrics.packet_loss_rate,
                    'bandwidth_usage': connection.metrics.bandwidth_usage
                }
            }
            
        except Exception as e:
            logger.error(f"Error getting status for camera {camera_id}: {e}")
            return None
    
    def get_all_connections_status(self) -> Dict[str, Any]:
        """Get status of all connections"""
        try:
            status = {
                'total_cameras': len(self.connections),
                'active_connections': len(self.active_connections),
                'connected_cameras': [],
                'disconnected_cameras': [],
                'high_priority_cameras': [],
                'medium_priority_cameras': [],
                'low_priority_cameras': [],
                'idle_cameras': [],
                'statistics': {
                    'total_connections': self.total_connections,
                    'successful_reconnections': self.successful_reconnections,
                    'failed_reconnections': self.failed_reconnections,
                    'uptime': (datetime.now() - self.start_time).total_seconds()
                }
            }
            
            for camera_id, connection in self.connections.items():
                camera_status = self.get_connection_status(camera_id)
                if camera_status:
                    if connection.status == ConnectionStatus.CONNECTED:
                        status['connected_cameras'].append(camera_status)
                    else:
                        status['disconnected_cameras'].append(camera_status)
                    
                    # Categorize by priority
                    if connection.priority == CameraPriority.HIGH:
                        status['high_priority_cameras'].append(camera_status)
                    elif connection.priority == CameraPriority.MEDIUM:
                        status['medium_priority_cameras'].append(camera_status)
                    elif connection.priority == CameraPriority.LOW:
                        status['low_priority_cameras'].append(camera_status)
                    else:
                        status['idle_cameras'].append(camera_status)
            
            return status
            
        except Exception as e:
            logger.error(f"Error getting all connections status: {e}")
            return {}
    
    def _establish_connection(self, connection: CameraConnection) -> bool:
        """Establish websocket connection for a camera"""
        try:
            # This would integrate with your existing Socket.IO implementation
            # For now, we'll simulate a connection
            
            # In a real implementation, you would:
            # 1. Create Socket.IO connection to the camera
            # 2. Set up event handlers for motion detection
            # 3. Configure reconnection parameters
            # 4. Set up heartbeat monitoring
            
            # Simulate connection delay
            time.sleep(0.1)
            
            # For now, we'll just mark as connected
            # In reality, you'd need to integrate with your existing websocket infrastructure
            connection.websocket = "connected"  # Placeholder
            
            return True
            
        except Exception as e:
            logger.error(f"Error establishing connection for {connection.camera_id}: {e}")
            return False
    
    def _add_to_connection_queue(self, camera_id: str, priority: CameraPriority):
        """Add camera to connection queue with priority"""
        try:
            # Priority queue: lower number = higher priority
            priority_value = {
                CameraPriority.HIGH: 1,
                CameraPriority.MEDIUM: 2,
                CameraPriority.LOW: 3,
                CameraPriority.IDLE: 4
            }.get(priority, 3)
            
            self.connection_queue.put((priority_value, camera_id))
            
        except Exception as e:
            logger.error(f"Error adding camera {camera_id} to connection queue: {e}")
    
    def _update_camera_priority(self, connection: CameraConnection):
        """Update camera priority based on activity"""
        try:
            old_priority = connection.priority
            
            # Determine new priority based on activity
            if connection.metrics.viewer_count > 2 or connection.metrics.motion_events_count > 10:
                connection.priority = CameraPriority.HIGH
            elif connection.metrics.viewer_count > 0 or connection.metrics.motion_events_count > 0:
                connection.priority = CameraPriority.MEDIUM
            elif connection.motion_detection_enabled:
                connection.priority = CameraPriority.LOW
            else:
                connection.priority = CameraPriority.IDLE
            
            # If priority changed, update queue
            if old_priority != connection.priority:
                self._add_to_connection_queue(connection.camera_id, connection.priority)
                
        except Exception as e:
            logger.error(f"Error updating priority for camera {connection.camera_id}: {e}")
    
    def _connection_manager_loop(self):
        """Main connection management loop"""
        while self.running:
            try:
                # Process connection queue
                while not self.connection_queue.empty() and len(self.active_connections) < self.max_concurrent_connections:
                    try:
                        priority, camera_id = self.connection_queue.get_nowait()
                        
                        if camera_id in self.connections:
                            connection = self.connections[camera_id]
                            
                            # Only connect if not already connected and motion detection is enabled
                            if (connection.status != ConnectionStatus.CONNECTED and 
                                connection.motion_detection_enabled):
                                self.connect_camera(camera_id)
                        
                    except queue.Empty:
                        break
                    except Exception as e:
                        logger.error(f"Error processing connection queue: {e}")
                
                # Handle reconnections
                self._handle_reconnections()
                
                # Sleep before next iteration
                time.sleep(5)
                
            except Exception as e:
                logger.error(f"Error in connection manager loop: {e}")
                time.sleep(10)
    
    def _handle_reconnections(self):
        """Handle automatic reconnections for failed connections"""
        try:
            current_time = datetime.now()
            
            for camera_id, connection in self.connections.items():
                if not connection.auto_reconnect:
                    continue
                
                # Check if connection needs reconnection
                if (connection.status == ConnectionStatus.ERROR or 
                    connection.status == ConnectionStatus.DISCONNECTED):
                    
                    # Check if enough time has passed since last attempt
                    if (connection.last_reconnect_attempt is None or
                        (current_time - connection.last_reconnect_attempt).total_seconds() > connection.reconnect_delay):
                        
                        # Check if we haven't exceeded max attempts
                        if connection.reconnect_attempts < connection.max_reconnect_attempts:
                            logger.info(f"Attempting to reconnect camera {connection.camera_name} ({camera_id})")
                            
                            if self.connect_camera(camera_id):
                                self.successful_reconnections += 1
                            else:
                                connection.reconnect_attempts += 1
                                # Exponential backoff
                                connection.reconnect_delay = min(
                                    connection.reconnect_delay * 2,
                                    connection.max_reconnect_delay
                                )
                                self.failed_reconnections += 1
                        
        except Exception as e:
            logger.error(f"Error handling reconnections: {e}")
    
    def _health_check_loop(self):
        """Health check loop for active connections"""
        while self.running:
            try:
                current_time = datetime.now()
                
                for camera_id in list(self.active_connections):
                    if camera_id not in self.connections:
                        continue
                    
                    connection = self.connections[camera_id]
                    
                    # Check connection health
                    if connection.websocket:
                        # In a real implementation, you'd check websocket health
                        # For now, we'll just update heartbeat
                        connection.metrics.last_heartbeat = current_time
                    
                    # Check for stale connections
                    if (connection.last_activity and 
                        (current_time - connection.last_activity).total_seconds() > 300):  # 5 minutes
                        logger.warning(f"Camera {camera_id} appears stale, checking connection")
                        # You could implement connection refresh here
                
                # Sleep before next health check
                time.sleep(self.health_check_interval)
                
            except Exception as e:
                logger.error(f"Error in health check loop: {e}")
                time.sleep(30)
    
    def _activity_monitor_loop(self):
        """Activity monitoring loop"""
        while self.running:
            try:
                current_time = datetime.now()
                
                for camera_id, connection in self.connections.items():
                    # Check if camera has been inactive for too long
                    if (connection.last_activity and 
                        (current_time - connection.last_activity).total_seconds() > 600):  # 10 minutes
                        connection.is_active = False
                    
                    # Update priority based on activity
                    self._update_camera_priority(connection)
                
                # Sleep before next activity check
                time.sleep(self.activity_update_interval)
                
            except Exception as e:
                logger.error(f"Error in activity monitor loop: {e}")
                time.sleep(60)
