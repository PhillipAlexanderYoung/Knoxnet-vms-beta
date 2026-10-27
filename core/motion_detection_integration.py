"""
Real Motion Detection Integration Module

This module provides production-ready motion detection integration with CameraManager and SocketIO.
"""

import asyncio
import threading
import time
import logging
from typing import Dict, List, Optional, Callable, Any
from datetime import datetime, timedelta
import json

logger = logging.getLogger(__name__)

class MotionDetectionIntegration:
    """
    Production-ready motion detection integration.
    Connects CameraManager events to SocketIO for real-time updates.
    """
    
    def __init__(self, 
                 socketio_instance,
                 camera_manager=None,
                 stream_server=None,
                 ai_agent=None):
        """
        Initialize the motion detection integration
        
        Args:
            socketio_instance: Socket.IO instance for websocket operations
            camera_manager: Camera manager instance for camera operations
            stream_server: Stream server instance for motion detection
            ai_agent: AI agent instance
        """
        self.socketio = socketio_instance
        self.camera_manager = camera_manager
        self.stream_server = stream_server
        self.ai_agent = ai_agent
        
        # State tracking
        self.running = False
        self.camera_activity: Dict[str, Dict[str, Any]] = {}
        self.motion_event_history: Dict[str, List[Dict[str, Any]]] = {}
        self._lock = threading.RLock()
        
        logger.info("Motion Detection Integration initialized")
    
    def start(self):
        """Start the motion detection integration"""
        if self.running:
            return
        
        self.running = True
        
        # Hook into CameraManager events if available
        if self.camera_manager:
            self.camera_manager.on_motion_detected = self._on_motion_detected
            self.camera_manager.on_camera_connected = self._on_camera_connected
            self.camera_manager.on_camera_disconnected = self._on_camera_disconnected
            self.camera_manager.on_camera_error = self._on_camera_error
            
        logger.info("Motion Detection Integration started")
    
    def stop(self):
        """Stop the motion detection integration"""
        self.running = False
        logger.info("Motion Detection Integration stopped")
        
    async def _on_motion_detected(self, camera_id: str, metadata: Dict[str, Any]):
        """Handle motion detected event from CameraManager"""
        try:
            timestamp = datetime.now().isoformat()
            
            with self._lock:
                # Update activity state
                if camera_id not in self.camera_activity:
                    self.camera_activity[camera_id] = {
                        "camera_id": camera_id,
                        "motion_count": 0,
                        "first_seen": timestamp
                    }
                
                self.camera_activity[camera_id]["last_motion"] = timestamp
                self.camera_activity[camera_id]["motion_count"] += 1
                self.camera_activity[camera_id]["status"] = "active"
                
                # Add to history
                event = {
                    "id": f"evt_{int(time.time()*1000)}",
                    "camera_id": camera_id,
                    "timestamp": timestamp,
                    "type": "motion",
                    "metadata": metadata
                }
                
                if camera_id not in self.motion_event_history:
                    self.motion_event_history[camera_id] = []
                
                self.motion_event_history[camera_id].insert(0, event)
                # Keep history limited
                if len(self.motion_event_history[camera_id]) > 100:
                    self.motion_event_history[camera_id].pop()
            
            # Broadcast to SocketIO clients
            if self.socketio:
                self.socketio.emit('motion_event', event, room=f"camera_{camera_id}")
                self.socketio.emit('camera_activity', self.camera_activity[camera_id])
                
        except Exception as e:
            logger.error(f"Error handling motion event for {camera_id}: {e}")

    async def _on_camera_connected(self, camera_id: str):
        """Handle camera connected event"""
        if self.socketio:
            self.socketio.emit('camera_status', {
                'camera_id': camera_id,
                'status': 'connected',
                'timestamp': datetime.now().isoformat()
            })

    async def _on_camera_disconnected(self, camera_id: str):
        """Handle camera disconnected event"""
        if self.socketio:
            self.socketio.emit('camera_status', {
                'camera_id': camera_id,
                'status': 'disconnected',
                'timestamp': datetime.now().isoformat()
            })

    async def _on_camera_error(self, camera_id: str, error: str):
        """Handle camera error event"""
        if self.socketio:
            self.socketio.emit('camera_error', {
                'camera_id': camera_id,
                'error': error,
                'timestamp': datetime.now().isoformat()
            })
    
    def get_connection_status(self, camera_id: Optional[str] = None) -> Dict[str, Any]:
        """Get real connection status from CameraManager"""
        if not self.camera_manager:
            return {'status': 'unavailable', 'error': 'Camera manager not initialized'}
            
        if camera_id:
            health = self.camera_manager.get_camera_health(camera_id)
            status = health.status.value if health else 'unknown'
            return {
                'camera_id': camera_id,
                'status': status,
                'motion_detection_enabled': True, # TODO: Read from config
                'timestamp': datetime.now().isoformat()
            }
        else:
            connected_cameras = self.camera_manager.get_connected_cameras()
            return {
                'status': 'active',
                'total_cameras': len(self.camera_manager.cameras),
                'active_connections': len(connected_cameras),
                'connected_ids': connected_cameras,
                'timestamp': datetime.now().isoformat()
            }
    
    def get_camera_activity(self, camera_id: str) -> Optional[Dict[str, Any]]:
        """Get camera activity from local state"""
        return self.camera_activity.get(camera_id, {
            'camera_id': camera_id,
            'last_motion': None,
            'motion_count': 0,
            'status': 'idle'
        })
    
    def get_motion_event_history(self, camera_id: str) -> List[Dict[str, Any]]:
        """Get motion event history from local state"""
        return self.motion_event_history.get(camera_id, [])
    
    def force_reconnect_camera(self, camera_id: str) -> bool:
        """Force reconnect camera via CameraManager"""
        if self.camera_manager:
            logger.info(f"Force reconnect requested for camera {camera_id}")
            # Run async method in background
            asyncio.create_task(self.camera_manager.reconnect_camera(camera_id))
            return True
        return False
    
    def update_camera_motion_detection(self, camera_id: str, enabled: bool) -> bool:
        """Update camera motion detection settings"""
        logger.info(f"Motion detection {'enabled' if enabled else 'disabled'} for camera {camera_id}")
        if self.camera_manager:
            asyncio.create_task(
                self.camera_manager.update_camera(camera_id, {"motion_detection": enabled})
            )
            return True
        return False
