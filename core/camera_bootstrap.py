#!/usr/bin/env python3
"""
Camera Bootstrap System
Automatically starts all cameras on server startup and after MediaMTX restarts
"""

import asyncio
import time
import logging
import os
import socket
from typing import Optional, Dict, Any
import requests

logger = logging.getLogger(__name__)

class CameraBootstrap:
    """Bootstrap all cameras on server startup"""
    
    def __init__(self, camera_manager=None, stream_server=None):
        self.camera_manager = camera_manager
        self.stream_server = stream_server
        
        # Load MediaMTX configuration
        env_url = os.environ.get('MEDIAMTX_API_URL')
        logger.info(f"DEBUG: MEDIAMTX_API_URL env var: {env_url}")
        
        # Default to localhost if not set
        self.api_url = (env_url or 'http://localhost:9997/v3').rstrip('/')
        
        # Failsafe: If we are in Docker (can resolve 'mediamtx') and URL is localhost, fix it
        if 'localhost' in self.api_url or '127.0.0.1' in self.api_url:
            try:
                # Try to resolve mediamtx hostname
                socket.gethostbyname('mediamtx')
                logger.warning("⚠️ Detected Docker environment but API URL is localhost. Switching to http://mediamtx:9997/v3")
                self.api_url = 'http://mediamtx:9997/v3'
            except socket.gaierror:
                pass  # Not in docker or mediamtx not resolvable
        
        self.api_user = os.environ.get('MEDIAMTX_API_USERNAME', 'admin')
        self.api_pass = os.environ.get('MEDIAMTX_API_PASSWORD', '')
        
        self.auth = None
        if self.api_user and self.api_pass:
            self.auth = (self.api_user, self.api_pass)
            
        logger.info(f"CameraBootstrap initialized with MediaMTX URL: {self.api_url}")
        
    async def bootstrap_all_cameras(self, delay_between_cameras: float = 2.0) -> Dict[str, bool]:
        """Bootstrap all enabled cameras with delays to prevent overwhelming"""
        results = {}
        
        if not self.camera_manager:
            logger.warning("No camera manager available for bootstrap")
            return results
            
        try:
            # Get all cameras
            cameras = self.camera_manager.get_all_cameras()
            if not cameras:
                logger.info("No cameras found to bootstrap")
                return results
                
            logger.info(f"🚀 Bootstrapping {len(cameras)} cameras...")
            
            for camera_id, camera_config in cameras.items():
                if not getattr(camera_config, 'enabled', True):
                    logger.debug(f"Skipping disabled camera: {camera_id}")
                    results[camera_id] = False
                    continue
                    
                try:
                    success = await self._bootstrap_single_camera(camera_id, camera_config)
                    results[camera_id] = success
                    
                    if success:
                        logger.info(f"✅ Successfully bootstrapped camera: {getattr(camera_config, 'name', camera_id)}")
                    else:
                        logger.warning(f"⚠️ Failed to bootstrap camera: {getattr(camera_config, 'name', camera_id)}")
                        
                except Exception as e:
                    logger.error(f"❌ Error bootstrapping camera {camera_id}: {e}")
                    results[camera_id] = False
                
                # Delay between cameras to prevent overwhelming the system
                if delay_between_cameras > 0:
                    await asyncio.sleep(delay_between_cameras)
                    
            successful = sum(1 for success in results.values() if success)
            logger.info(f"🎯 Bootstrap complete: {successful}/{len(cameras)} cameras started successfully")
            
        except Exception as e:
            logger.error(f"Error during camera bootstrap: {e}")
            
        return results
        
    async def _bootstrap_single_camera(self, camera_id: str, camera_config) -> bool:
        """Bootstrap a single camera"""
        try:
            rtsp_url = getattr(camera_config, 'rtsp_url', None)
            if not rtsp_url:
                logger.warning(f"No RTSP URL for camera {camera_id}")
                return False
                
            # Check if MediaMTX path already exists and is ready
            if await self._check_mediamtx_path_ready(camera_id):
                logger.debug(f"Camera {camera_id} already active in MediaMTX")
                
                # Still need to start stream server if not active
                if self.stream_server and camera_id not in getattr(self.stream_server, 'active_streams', {}):
                    await self._start_stream_server_stream(camera_id, camera_config)
                    
                return True
                
            # Configure MediaMTX path
            if not await self._configure_mediamtx_path(camera_id, camera_config):
                return False
                
            # Wait for MediaMTX to be ready
            await asyncio.sleep(3)
            
            # Start stream server stream
            if self.stream_server:
                await self._start_stream_server_stream(camera_id, camera_config)
                
            # Verify the camera is working
            await asyncio.sleep(2)
            return await self._verify_camera_working(camera_id)
            
        except Exception as e:
            logger.error(f"Error bootstrapping camera {camera_id}: {e}")
            return False
            
    async def _check_mediamtx_path_ready(self, camera_id: str) -> bool:
        """Check if MediaMTX path is ready"""
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
            
    async def _configure_mediamtx_path(self, camera_id: str, camera_config) -> bool:
        """Configure MediaMTX path for camera"""
        try:
            rtsp_url = getattr(camera_config, 'rtsp_url', None)
            if not rtsp_url:
                return False
                
            # MediaMTX path configuration
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
                "runOnUnDemand": "",
                "runOnReady": "",
                "runOnNotReady": "",
                "readTimeout": "10s",
                "writeTimeout": "10s"
            }
            
            # Apply configuration
            response = requests.patch(
                f"{self.api_url}/config/paths/patch/{camera_id}",
                json=path_config,
                auth=self.auth,
                timeout=10
            )
            
            if response.status_code == 200:
                logger.debug(f"✅ Configured MediaMTX path for {camera_id}")
                return True
            elif response.status_code == 404:
                # Path might not exist, try adding it
                response = requests.post(
                    f"{self.api_url}/config/paths/add/{camera_id}",
                    json=path_config,
                    auth=self.auth,
                    timeout=10
                )
                if response.status_code in (200, 201):
                     logger.debug(f"✅ Added MediaMTX path for {camera_id}")
                     return True
                     
            logger.warning(f"⚠️ Failed to configure MediaMTX path for {camera_id}: HTTP {response.status_code}")
            return False
                
        except Exception as e:
            logger.error(f"❌ Error configuring MediaMTX path for {camera_id}: {e}")
            return False
            
    async def _start_stream_server_stream(self, camera_id: str, camera_config):
        """Start stream in stream server"""
        try:
            if not self.stream_server:
                return
                
            stream_config = {
                'rtsp_url': getattr(camera_config, 'rtsp_url', None),
                'webrtc_enabled': getattr(camera_config, 'webrtc_enabled', True),
                'mediamtx_path': camera_id,
                'motion_detection': getattr(camera_config, 'motion_detection', False),
            }
            
            await self.stream_server.start_stream(camera_id, stream_config)
            logger.debug(f"✅ Started stream server stream for {camera_id}")
            
        except Exception as e:
            logger.error(f"❌ Error starting stream server for {camera_id}: {e}")
            
    async def _verify_camera_working(self, camera_id: str) -> bool:
        """Verify camera is working properly"""
        try:
            # Check MediaMTX status
            mediamtx_ready = await self._check_mediamtx_path_ready(camera_id)
            
            # Check stream server status
            stream_active = False
            if self.stream_server and hasattr(self.stream_server, 'active_streams'):
                stream_active = camera_id in self.stream_server.active_streams
                
            return mediamtx_ready or stream_active
            
        except Exception as e:
            logger.error(f"Error verifying camera {camera_id}: {e}")
            return False
            
    def bootstrap_cameras_sync(self, delay_between_cameras: float = 2.0) -> Dict[str, bool]:
        """Synchronous wrapper for bootstrap_all_cameras"""
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                return loop.run_until_complete(self.bootstrap_all_cameras(delay_between_cameras))
            finally:
                loop.close()
        except Exception as e:
            logger.error(f"Error in synchronous camera bootstrap: {e}")
            return {}
