#!/usr/bin/env python3
"""
Automatic Camera Recovery System
Monitors camera health and automatically recovers failed streams
"""

import asyncio
import time
import threading
import logging
import os
from typing import Dict, Set, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import requests
import json

logger = logging.getLogger(__name__)

@dataclass
class CameraRecoveryState:
    """Track recovery state for each camera"""
    camera_id: str
    last_check: datetime = field(default_factory=datetime.now)
    last_success: datetime = field(default_factory=datetime.now)
    consecutive_failures: int = 0
    recovery_attempts: int = 0
    last_recovery_attempt: Optional[datetime] = None
    is_recovering: bool = False
    mediamtx_ready: bool = False
    stream_active: bool = False
    rtsp_reachable: bool = False
    
    def should_attempt_recovery(self) -> bool:
        """Determine if recovery should be attempted based on exponential backoff"""
        if not self.last_recovery_attempt:
            return True
            
        # Exponential backoff: 30s, 60s, 120s, 240s, then 300s max
        interval = min(30 * (2 ** self.recovery_attempts), 300)
        return (datetime.now() - self.last_recovery_attempt).total_seconds() >= interval

class CameraAutoRecovery:
    """Automatic camera recovery system"""
    
    def __init__(self, camera_manager=None, stream_server=None, mediamtx_client=None):
        self.camera_manager = camera_manager
        self.stream_server = stream_server
        self.mediamtx_client = mediamtx_client
        
        # Load MediaMTX configuration
        self.api_url = os.environ.get('MEDIAMTX_API_URL', 'http://localhost:9997/v3').rstrip('/')
        self.api_user = os.environ.get('MEDIAMTX_API_USERNAME', 'admin')
        self.api_pass = os.environ.get('MEDIAMTX_API_PASSWORD', '')
        
        self.auth = None
        if self.api_user and self.api_pass:
            self.auth = (self.api_user, self.api_pass)
        
        # Recovery state tracking
        self.recovery_states: Dict[str, CameraRecoveryState] = {}
        self.recovery_lock = threading.Lock()
        
        # Configuration
        self.check_interval = 30  # Check every 30 seconds
        self.max_concurrent_recoveries = 3
        self.active_recoveries: Set[str] = set()
        
        # Control
        self.running = False
        self.monitor_thread: Optional[threading.Thread] = None
        
        # Callbacks
        self.on_camera_recovered: Optional[Callable] = None
        self.on_camera_failed: Optional[Callable] = None
        
    def start(self):
        """Start the auto-recovery monitoring"""
        if self.running:
            logger.warning("Camera auto-recovery already running")
            return
            
        self.running = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
        logger.info("🔄 Camera auto-recovery system started")
        
    def stop(self):
        """Stop the auto-recovery monitoring"""
        self.running = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)
        logger.info("🛑 Camera auto-recovery system stopped")
        
    def _monitor_loop(self):
        """Main monitoring loop"""
        while self.running:
            try:
                self._check_all_cameras()
                time.sleep(self.check_interval)
            except Exception as e:
                logger.error(f"Error in camera monitor loop: {e}")
                time.sleep(5)  # Brief pause on error
                
    def _check_all_cameras(self):
        """Check health of all cameras and trigger recovery if needed"""
        if not self.camera_manager:
            return
            
        try:
            # Get all cameras from camera manager
            cameras = self.camera_manager.get_all_cameras()
            if not cameras:
                return
                
            for camera_id, camera_config in cameras.items():
                if not getattr(camera_config, 'enabled', True):
                    continue
                    
                # Initialize recovery state if needed
                if camera_id not in self.recovery_states:
                    self.recovery_states[camera_id] = CameraRecoveryState(camera_id)
                    
                state = self.recovery_states[camera_id]
                state.last_check = datetime.now()
                
                # Check camera health
                health_status = self._check_camera_health(camera_id, camera_config)
                
                if health_status['healthy']:
                    # Camera is healthy - reset failure counters
                    if state.consecutive_failures > 0:
                        logger.info(f"✅ Camera {camera_id} recovered after {state.consecutive_failures} failures")
                        if self.on_camera_recovered:
                            self.on_camera_recovered(camera_id)
                            
                    state.consecutive_failures = 0
                    state.recovery_attempts = 0
                    state.last_success = datetime.now()
                    state.is_recovering = False
                    state.mediamtx_ready = health_status.get('mediamtx_ready', False)
                    state.stream_active = health_status.get('stream_active', False)
                    state.rtsp_reachable = health_status.get('rtsp_reachable', False)
                else:
                    # Camera is unhealthy - consider recovery
                    state.consecutive_failures += 1
                    logger.debug(f"🔍 Camera {camera_id} health check failed (attempt {state.consecutive_failures})")
                    
                    # Trigger recovery if needed
                    if state.consecutive_failures >= 3 and state.should_attempt_recovery():
                        self._attempt_camera_recovery(camera_id, camera_config)
                        
        except Exception as e:
            logger.error(f"Error checking camera health: {e}")
            
    def _check_camera_health(self, camera_id: str, camera_config) -> dict:
        """Check comprehensive health of a camera"""
        health = {
            'healthy': True,
            'mediamtx_ready': False,
            'stream_active': False,
            'rtsp_reachable': False,
            'issues': []
        }
        
        try:
            # 1. Check MediaMTX path status
            mediamtx_status = self._check_mediamtx_path(camera_id)
            health['mediamtx_ready'] = mediamtx_status
            if not mediamtx_status:
                health['healthy'] = False
                health['issues'].append('mediamtx_not_ready')
                
            # 2. Check if stream is active in stream server
            if self.stream_server:
                stream_active = self._check_stream_server_status(camera_id)
                health['stream_active'] = stream_active
                if not stream_active:
                    health['healthy'] = False
                    health['issues'].append('stream_not_active')
                    
            # 3. Check RTSP reachability (optional, might be slow)
            # Only check if other checks pass to avoid overwhelming
            if health['healthy']:
                rtsp_url = getattr(camera_config, 'rtsp_url', None)
                if rtsp_url:
                    rtsp_reachable = self._check_rtsp_reachability(rtsp_url)
                    health['rtsp_reachable'] = rtsp_reachable
                    if not rtsp_reachable:
                        health['healthy'] = False
                        health['issues'].append('rtsp_unreachable')
                        
        except Exception as e:
            logger.debug(f"Error checking health for camera {camera_id}: {e}")
            health['healthy'] = False
            health['issues'].append(f'health_check_error: {e}')
            
        return health
        
    def _check_mediamtx_path(self, camera_id: str) -> bool:
        """Check if camera path exists and is ready in MediaMTX"""
        try:
            response = requests.get(
                f"{self.api_url}/paths/get/{camera_id}",
                auth=self.auth,
                timeout=5
            )
            if response.status_code == 200:
                data = response.json()
                return data.get('ready', False)
            return False
        except Exception:
            return False
            
    def _check_stream_server_status(self, camera_id: str) -> bool:
        """Check if camera stream is active in stream server"""
        try:
            if hasattr(self.stream_server, 'active_streams'):
                return camera_id in self.stream_server.active_streams
            return False
        except Exception:
            return False
            
    def _check_rtsp_reachability(self, rtsp_url: str) -> bool:
        """Quick check if RTSP stream is reachable (basic connectivity)"""
        try:
            # Simple TCP connection test to RTSP port
            import socket
            from urllib.parse import urlparse
            
            parsed = urlparse(rtsp_url)
            host = parsed.hostname or 'localhost'
            port = parsed.port or 554
            
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(3)
            result = sock.connect_ex((host, port))
            sock.close()
            
            return result == 0
        except Exception:
            return False
            
    def _attempt_camera_recovery(self, camera_id: str, camera_config):
        """Attempt to recover a failed camera"""
        with self.recovery_lock:
            # Check if we can start a new recovery
            if len(self.active_recoveries) >= self.max_concurrent_recoveries:
                logger.debug(f"Recovery rate limited for {camera_id}")
                return
                
            if camera_id in self.active_recoveries:
                logger.debug(f"Recovery already active for {camera_id}")
                return
                
            self.active_recoveries.add(camera_id)
            
        state = self.recovery_states[camera_id]
        state.is_recovering = True
        state.last_recovery_attempt = datetime.now()
        state.recovery_attempts += 1
        
        # Start recovery in background thread
        recovery_thread = threading.Thread(
            target=self._recover_camera_thread,
            args=(camera_id, camera_config),
            daemon=True
        )
        recovery_thread.start()
        
    def _recover_camera_thread(self, camera_id: str, camera_config):
        """Recovery thread for a specific camera"""
        try:
            logger.info(f"🔄 Attempting recovery for camera {camera_id} (attempt {self.recovery_states[camera_id].recovery_attempts})")
            
            # Step 1: Stop existing stream if any
            self._stop_camera_stream(camera_id)
            
            # Step 2: Wait a moment for cleanup
            time.sleep(2)
            
            # Step 3: Restart MediaMTX path if needed
            self._restart_mediamtx_path(camera_id, camera_config)
            
            # Step 4: Wait for MediaMTX to be ready
            time.sleep(3)
            
            # Step 5: Restart stream server stream
            self._restart_stream_server_stream(camera_id, camera_config)
            
            # Step 6: Wait and verify recovery
            time.sleep(5)
            health = self._check_camera_health(camera_id, camera_config)
            
            if health['healthy']:
                logger.info(f"✅ Successfully recovered camera {camera_id}")
            else:
                logger.warning(f"⚠️ Recovery attempt for {camera_id} completed but camera still unhealthy: {health['issues']}")
                
        except Exception as e:
            logger.error(f"❌ Error during recovery of camera {camera_id}: {e}")
        finally:
            # Always clean up
            with self.recovery_lock:
                self.active_recoveries.discard(camera_id)
            
            if camera_id in self.recovery_states:
                self.recovery_states[camera_id].is_recovering = False
                
    def _stop_camera_stream(self, camera_id: str):
        """Stop existing camera stream"""
        try:
            if self.stream_server and hasattr(self.stream_server, 'stop_stream'):
                # Only run in new loop if no loop is running, or if thread safety is ensured
                # asyncio.run might fail if there's a running loop in this thread?
                # This runs in a thread.
                asyncio.run(self.stream_server.stop_stream(camera_id))
        except Exception as e:
            logger.debug(f"Error stopping stream for {camera_id}: {e}")
            
    def _restart_mediamtx_path(self, camera_id: str, camera_config):
        """Restart MediaMTX path for camera"""
        try:
            # Get RTSP URL
            rtsp_url = getattr(camera_config, 'rtsp_url', None)
            if not rtsp_url:
                logger.warning(f"No RTSP URL for camera {camera_id}")
                return
                
            # Update MediaMTX path configuration
            path_config = {
                "source": rtsp_url,
                "sourceProtocol": "automatic",
                "sourceAnyPortEnable": False,
                "runOnInit": "",
                "runOnInitRestart": False,
                "runOnDemand": "",
                "runOnDemandRestart": False,
                "runOnDemandStartTimeout": "10s",
                "runOnDemandCloseAfter": "10s",
                "runOnUnDemand": ""
            }
            
            # Apply configuration to MediaMTX
            # Use POST if it might be missing, or PATCH if updating
            response = requests.patch(
                f"{self.api_url}/config/paths/patch/{camera_id}",
                json=path_config,
                auth=self.auth,
                timeout=10
            )
            
            if response.status_code == 404:
                response = requests.post(
                    f"{self.api_url}/config/paths/add/{camera_id}",
                    json=path_config,
                    auth=self.auth,
                    timeout=10
                )
            
            if response.status_code in (200, 201):
                logger.debug(f"Updated MediaMTX path config for {camera_id}")
            else:
                logger.warning(f"Failed to update MediaMTX path for {camera_id}: {response.status_code}")
                
        except Exception as e:
            logger.error(f"Error restarting MediaMTX path for {camera_id}: {e}")
            
    def _restart_stream_server_stream(self, camera_id: str, camera_config):
        """Restart stream in stream server"""
        try:
            if not self.stream_server:
                return
                
            # Start stream with proper configuration
            stream_config = {
                'rtsp_url': getattr(camera_config, 'rtsp_url', None),
                'webrtc_enabled': getattr(camera_config, 'webrtc_enabled', True),
                'mediamtx_path': camera_id,
                'motion_detection': getattr(camera_config, 'motion_detection', False),
            }
            
            asyncio.run(self.stream_server.start_stream(camera_id, stream_config))
            logger.debug(f"Restarted stream server stream for {camera_id}")
            
        except Exception as e:
            logger.error(f"Error restarting stream server for {camera_id}: {e}")
            
    def get_recovery_status(self) -> dict:
        """Get current recovery status for all cameras"""
        status = {
            'running': self.running,
            'active_recoveries': len(self.active_recoveries),
            'cameras': {}
        }
        
        for camera_id, state in self.recovery_states.items():
            status['cameras'][camera_id] = {
                'consecutive_failures': state.consecutive_failures,
                'recovery_attempts': state.recovery_attempts,
                'is_recovering': state.is_recovering,
                'last_success': state.last_success.isoformat() if state.last_success else None,
                'last_check': state.last_check.isoformat() if state.last_check else None,
                'mediamtx_ready': state.mediamtx_ready,
                'stream_active': state.stream_active,
                'rtsp_reachable': state.rtsp_reachable
            }
            
        return status
