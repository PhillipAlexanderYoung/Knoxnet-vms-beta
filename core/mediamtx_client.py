import asyncio
import base64
import json
import logging
import websockets
import aiohttp
from typing import Dict, Optional, Callable, Any, List, Tuple
from datetime import datetime, timedelta
import threading
import time
from urllib.parse import urljoin
from dataclasses import dataclass
import os
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from aiortc import (  # pragma: no cover
        RTCPeerConnection,
        RTCSessionDescription,
        VideoStreamTrack,
        RTCIceCandidate,
        RTCIceServer,
        RTCConfiguration,
    )
else:
    # Runtime imports are done lazily inside methods to keep desktop/light mode
    # from importing aiortc/av (and triggering decoder churn) unless explicitly needed.
    RTCPeerConnection = Any  # type: ignore
    RTCSessionDescription = Any  # type: ignore
    VideoStreamTrack = Any  # type: ignore
    RTCIceCandidate = Any  # type: ignore
    RTCIceServer = Any  # type: ignore
    RTCConfiguration = Any  # type: ignore


def _require_aiortc():
    """
    Lazily import aiortc symbols.
    This prevents importing aiortc/av at module import time (important for desktop-light mode).
    """
    try:
        from aiortc import (  # type: ignore
            RTCPeerConnection as _RTCPeerConnection,
            RTCSessionDescription as _RTCSessionDescription,
            VideoStreamTrack as _VideoStreamTrack,
            RTCIceCandidate as _RTCIceCandidate,
            RTCIceServer as _RTCIceServer,
            RTCConfiguration as _RTCConfiguration,
        )
        return (
            _RTCPeerConnection,
            _RTCSessionDescription,
            _VideoStreamTrack,
            _RTCIceCandidate,
            _RTCIceServer,
            _RTCConfiguration,
        )
    except Exception as e:
        raise RuntimeError("Missing dependency 'aiortc' (required only for WebRTC receiving).") from e

logger = logging.getLogger(__name__)


@dataclass
class StreamConfig:
    """Configuration for a MediaMTX stream"""
    stream_path: str
    rtsp_url: Optional[str] = None
    quality: str = "medium"  # low, medium, high
    enable_audio: bool = False
    reconnect_attempts: int = 5
    reconnect_delay: float = 2.0


@dataclass
class ConnectionStats:
    """Statistics for a WebRTC connection"""
    connected_at: datetime
    bytes_received: int = 0
    bytes_sent: int = 0
    packets_received: int = 0
    packets_sent: int = 0
    connection_state: str = "new"
    last_activity: datetime = None


class MediaMTXWebRTCClient:
    """
    Complete MediaMTX WebRTC client implementation for receiving video streams
    """

    def __init__(self,
                 mediamtx_host: str = "localhost",
                 mediamtx_webrtc_port: int = 8889,
                 mediamtx_api_port: int = 9997,
                 ice_servers: Optional[List[Dict[str, Any]]] = None,
                 api_username: Optional[str] = None,
                 api_password: Optional[str] = None):
        """
        Initialize MediaMTX WebRTC client

        Args:
            mediamtx_host: MediaMTX server hostname
            mediamtx_webrtc_port: WebRTC port
            mediamtx_api_port: API port for configuration
            ice_servers: List of ICE servers for NAT traversal
        """
        self.mediamtx_host = mediamtx_host
        self.mediamtx_webrtc_port = mediamtx_webrtc_port
        self.mediamtx_api_port = mediamtx_api_port
        self.api_username = api_username or os.getenv('MEDIAMTX_API_USERNAME', 'admin')
        self.api_password = api_password or os.getenv('MEDIAMTX_API_PASSWORD', '')
        # Some shells escape "$" as "$$" when setting env vars; normalize for MediaMTX auth.
        try:
            if isinstance(self.api_password, str) and self.api_password.startswith('$$'):
                self.api_password = self.api_password.lstrip('$')
        except Exception:
            pass
        self._basic_auth_header: Optional[Dict[str, str]] = None
        if self.api_username and self.api_password:
            token = base64.b64encode(f"{self.api_username}:{self.api_password}".encode("utf-8")).decode("ascii")
            self._basic_auth_header = {"Authorization": f"Basic {token}"}

        # Default ICE servers
        self.ice_servers = ice_servers or [
            {"urls": "stun:stun.l.google.com:19302"},
            {"urls": "stun:stun1.l.google.com:19302"},
        ]

        # Connection management
        self.connections: Dict[str, RTCPeerConnection] = {}
        self.websockets: Dict[str, websockets.WebSocketServerProtocol] = {}
        # WHEP sessions (HTTP). MediaMTX WHEP is HTTP-based; WebSocket attempts will 405.
        self.whep_session_urls: Dict[str, str] = {}
        # WHEP viewer sessions created from browser offers (WHEP POST). Keyed by (stream_path, client_id).
        self.whep_viewer_session_urls: Dict[Tuple[str, str], str] = {}
        self.streams: Dict[str, VideoStreamTrack] = {}
        self.stats: Dict[str, ConnectionStats] = {}
        self.stream_configs: Dict[str, StreamConfig] = {}

        # Event callbacks
        self.on_stream_connected: Optional[Callable] = None
        self.on_stream_disconnected: Optional[Callable] = None
        self.on_stream_error: Optional[Callable] = None
        self.on_frame_received: Optional[Callable] = None

        # Background tasks
        self._running = False
        self._monitor_task: Optional[asyncio.Task] = None
        self._stats_task: Optional[asyncio.Task] = None

    async def start(self):
        """Start the MediaMTX client"""
        if self._running:
            return

        self._running = True
        logger.info("Starting MediaMTX WebRTC client")

        # Start background monitoring tasks
        self._monitor_task = asyncio.create_task(self._monitor_connections())
        self._stats_task = asyncio.create_task(self._update_stats())

        logger.info(f"MediaMTX client started - connecting to {self.mediamtx_host}:{self.mediamtx_webrtc_port}")

    async def stop(self):
        """Stop the MediaMTX client and cleanup all connections"""
        if not self._running:
            return

        logger.info("Stopping MediaMTX WebRTC client")
        self._running = False

        # Cancel background tasks
        if self._monitor_task:
            self._monitor_task.cancel()
        if self._stats_task:
            self._stats_task.cancel()

        # Close all connections
        for stream_path in list(self.connections.keys()):
            await self.disconnect_stream(stream_path)

        logger.info("MediaMTX client stopped")

    async def connect_stream(self, config: StreamConfig) -> bool:
        """
        Connect to a MediaMTX stream via WebRTC

        Args:
            config: Stream configuration

        Returns:
            True if connection successful, False otherwise
        """
        stream_path = config.stream_path

        if stream_path in self.connections:
            logger.warning(f"Stream {stream_path} already connected")
            return True

        logger.info(f"Connecting to stream: {stream_path}")

        try:
            (
                RTCPeerConnection_t,
                RTCSessionDescription_t,
                _VideoStreamTrack_t,
                RTCIceCandidate_t,
                RTCIceServer_t,
                RTCConfiguration_t,
            ) = _require_aiortc()

            # Store configuration
            self.stream_configs[stream_path] = config

            # Create peer connection (aiortc expects RTCConfiguration/RTCIceServer).
            # Passing a plain dict can lead to incomplete ICE setup and SDPs missing ice-ufrag/pwd.
            ice_objs: List[Any] = []
            try:
                for s in (self.ice_servers or []):
                    urls = None
                    if isinstance(s, dict):
                        urls = s.get("urls") or s.get("url")
                    elif isinstance(s, str):
                        urls = s
                    if not urls:
                        continue
                    if isinstance(urls, (list, tuple)):
                        ice_objs.append(RTCIceServer_t(urls=list(urls)))
                    else:
                        ice_objs.append(RTCIceServer_t(urls=[str(urls)]))
            except Exception:
                ice_objs = [RTCIceServer_t(urls=["stun:stun.l.google.com:19302"])]

            pc = RTCPeerConnection_t(configuration=RTCConfiguration_t(iceServers=ice_objs))

            # Ensure the offer contains real media sections.
            # Without transceivers/tracks, some WHEP servers (and aiortc) can produce an SDP
            # that is missing ICE ufrag/pwd, causing MediaMTX to reject it.
            try:
                pc.addTransceiver("video", direction="recvonly")
            except Exception:
                pass
            if getattr(config, "enable_audio", False):
                try:
                    pc.addTransceiver("audio", direction="recvonly")
                except Exception:
                    pass

            # Set up event handlers
            self._setup_peer_connection_handlers(pc, stream_path)

            # Create offer
            offer = await pc.createOffer()
            await pc.setLocalDescription(offer)

            # Wait for ICE gathering to complete (non-trickle WHEP). We also need ICE ufrag/pwd
            # present in the SDP; otherwise MediaMTX may reject the offer with:
            #   "SetRemoteDescription called with no ice-ufrag"
            try:
                start = time.time()
                while pc.iceGatheringState != "complete" and (time.time() - start) < 4.0:
                    await asyncio.sleep(0.05)
            except Exception:
                pass

            local_sdp = (pc.localDescription.sdp if pc.localDescription else "") or ""
            if not local_sdp:
                raise Exception("Missing local SDP for WHEP offer")
            if "a=ice-ufrag" not in local_sdp:
                # Give it a bit longer; on some systems ufrag appears slightly after gather completion.
                try:
                    start = time.time()
                    while "a=ice-ufrag" not in local_sdp and (time.time() - start) < 2.0:
                        await asyncio.sleep(0.05)
                        local_sdp = (pc.localDescription.sdp if pc.localDescription else "") or ""
                except Exception:
                    pass
            if "a=ice-ufrag" not in local_sdp:
                raise Exception("Local SDP missing ICE ufrag; cannot negotiate WHEP offer")

            # WHEP is HTTP-based (POST offer SDP -> answer SDP). WebSocket attempts will 405.
            whep_url = f"http://{self.mediamtx_host}:{self.mediamtx_webrtc_port}/{stream_path}/whep"
            logger.debug(f"Connecting via WHEP (HTTP) to: {whep_url}")

            headers = {
                "Content-Type": "application/sdp",
                "Accept": "application/sdp",
                **(self._basic_auth_header or {})
            }

            async with aiohttp.ClientSession() as session:
                async with session.post(whep_url, data=local_sdp, headers=headers) as resp:
                    if resp.status < 200 or resp.status >= 300:
                        try:
                            body = await resp.text()
                        except Exception:
                            body = ""
                        raise Exception(f"WHEP HTTP error {resp.status}: {body}")

                    answer_sdp = await resp.text()
                    if not answer_sdp or not answer_sdp.lstrip().startswith("v="):
                        raise Exception("Invalid WHEP answer SDP")

                    # Store session URL for future DELETE/PATCH if server provides it
                    loc = resp.headers.get("Location") or resp.headers.get("location")
                    if loc:
                        try:
                            self.whep_session_urls[stream_path] = urljoin(whep_url, loc)
                        except Exception:
                            self.whep_session_urls[stream_path] = loc

            # Set remote description
            await pc.setRemoteDescription(RTCSessionDescription_t(sdp=answer_sdp, type="answer"))
            logger.debug(f"Set remote description for stream: {stream_path}")

            # Store connection
            self.connections[stream_path] = pc
            self.stats[stream_path] = ConnectionStats(connected_at=datetime.now())

            logger.info(f"Successfully connected to stream: {stream_path}")

            if self.on_stream_connected:
                await self._safe_callback(self.on_stream_connected, stream_path)

            return True

        except Exception as e:
            logger.error(f"Failed to connect to stream {stream_path}: {e}")
            await self._cleanup_stream(stream_path)

            if self.on_stream_error:
                await self._safe_callback(self.on_stream_error, stream_path, str(e))

            return False

    async def disconnect_stream(self, stream_path: str):
        """
        Disconnect from a MediaMTX stream

        Args:
            stream_path: Path of the stream to disconnect
        """
        if stream_path not in self.connections:
            logger.warning(f"Stream {stream_path} not connected")
            return

        logger.info(f"Disconnecting from stream: {stream_path}")

        try:
            # Best-effort: close WHEP session on server if available
            session_url = self.whep_session_urls.get(stream_path)
            if session_url:
                try:
                    headers = {**(self._basic_auth_header or {})}
                    async with aiohttp.ClientSession() as session:
                        async with session.delete(session_url, headers=headers) as resp:
                            if resp.status not in (200, 202, 204, 404):
                                try:
                                    body = await resp.text()
                                except Exception:
                                    body = ""
                                logger.debug(f"WHEP DELETE {resp.status} for {stream_path}: {body}")
                except Exception as e:
                    logger.debug(f"WHEP session delete failed for {stream_path}: {e}")

            await self._cleanup_stream(stream_path)

            if self.on_stream_disconnected:
                await self._safe_callback(self.on_stream_disconnected, stream_path)

        except Exception as e:
            logger.error(f"Error disconnecting stream {stream_path}: {e}")

    async def get_stream_track(self, stream_path: str) -> Optional[VideoStreamTrack]:
        """
        Get the video track for a connected stream

        Args:
            stream_path: Path of the stream

        Returns:
            VideoStreamTrack if available, None otherwise
        """
        return self.streams.get(stream_path)

    # ------------------------------------------------------------------
    # Recording control (continuous recording via MediaMTX passthrough)
    # ------------------------------------------------------------------

    async def enable_recording(self, stream_path: str, source: str = "", record_path: str = "") -> bool:
        """Enable continuous recording for a stream path (zero-CPU passthrough)."""
        return await self._patch_path_recording(stream_path, True, source=source, record_path=record_path)

    async def disable_recording(self, stream_path: str) -> bool:
        """Disable continuous recording for a stream path."""
        return await self._patch_path_recording(stream_path, False)

    async def _patch_path_recording(self, stream_path: str, enable: bool, source: str = "", record_path: str = "") -> bool:
        base = f"http://{self.mediamtx_host}:{self.mediamtx_api_port}"
        patch_url = f"{base}/v3/config/paths/patch/{stream_path}"
        add_url = f"{base}/v3/config/paths/add/{stream_path}"
        payload: Dict[str, Any] = {"record": enable}
        if source:
            payload["source"] = source
            payload["sourceProtocol"] = "tcp"
        if record_path and enable:
            payload["recordPath"] = record_path
        try:
            async with aiohttp.ClientSession() as session:
                auth = aiohttp.BasicAuth(self.api_username, self.api_password)
                async with session.patch(patch_url, json=payload, auth=auth) as resp:
                    if resp.status == 200:
                        logger.info("Recording %s for path %s", "enabled" if enable else "disabled", stream_path)
                        return True
                    if resp.status != 404:
                        body = ""
                        try:
                            body = await resp.text()
                        except Exception:
                            pass
                        logger.error("Failed to set recording for %s (%s): %s", stream_path, resp.status, body)
                        return False

                # Path doesn't exist in config yet (camera uses wildcard).
                # Create it explicitly so we can control recording per-path.
                async with session.post(add_url, json=payload, auth=auth) as resp2:
                    if resp2.status in (200, 201):
                        logger.info("Created config path and %s recording for %s",
                                    "enabled" if enable else "disabled", stream_path)
                        return True
                    body = ""
                    try:
                        body = await resp2.text()
                    except Exception:
                        pass
                    logger.error("Failed to add path config for %s (%s): %s", stream_path, resp2.status, body)
                    return False
        except Exception as e:
            logger.error("Error toggling recording for %s: %s", stream_path, e)
            return False

    async def get_recording_status(self, stream_path: str) -> Optional[bool]:
        """Return True if recording is enabled for *stream_path*, None on error."""
        api_url = (
            f"http://{self.mediamtx_host}:{self.mediamtx_api_port}"
            f"/v3/config/paths/get/{stream_path}"
        )
        try:
            async with aiohttp.ClientSession() as session:
                auth = aiohttp.BasicAuth(self.api_username, self.api_password)
                async with session.get(api_url, auth=auth) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        return bool(data.get("record", False))
                    return None
        except Exception as e:
            logger.error("Error checking recording status for %s: %s", stream_path, e)
            return None

    async def list_recordings(self, stream_path: str) -> List[Dict[str, Any]]:
        """List available recording segments for a stream path via the playback API."""
        playback_port = int(os.getenv("MEDIAMTX_PLAYBACK_PORT", "9996"))
        url = f"http://{self.mediamtx_host}:{playback_port}/list?path={stream_path}"
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        return data if isinstance(data, list) else []
                    return []
        except Exception as e:
            logger.error("Error listing recordings for %s: %s", stream_path, e)
            return []

    async def delete_stream_path(self, stream_path: str) -> bool:
        """
        Delete a stream path configuration from MediaMTX
        
        Args:
            stream_path: Stream path to delete
            
        Returns:
            True if deleted successfully
        """
        try:
            api_url_delete = f"http://{self.mediamtx_host}:{self.mediamtx_api_port}/v3/config/paths/delete/{stream_path}"
            
            async with aiohttp.ClientSession() as session:
                auth = aiohttp.BasicAuth(self.api_username, self.api_password)
                async with session.delete(api_url_delete, auth=auth) as response:
                    if response.status in (200, 204):
                        logger.info(f"Deleted stream path: {stream_path}")
                        return True
                    elif response.status == 404:
                        logger.info(f"Stream path {stream_path} not found (already deleted)")
                        return True
                    else:
                        try:
                            body = await response.text()
                        except Exception:
                            body = ''
                        logger.error(f"Failed to delete stream path {stream_path} ({response.status}): {body}")
                        return False
                        
        except Exception as e:
            logger.error(f"Error deleting stream path {stream_path}: {e}")
            return False

    async def configure_stream_source(self, stream_path: str, rtsp_url: str, force_recreate: bool = False) -> bool:
        """
        Configure MediaMTX to use an RTSP source for a stream path.

        Preserves the current ``record`` flag when PATCHing an existing path
        so that starting/reconnecting a live stream never silently disables
        an ongoing recording.
        """
        try:
            if force_recreate:
                await self.delete_stream_path(stream_path)

            base = f"http://{self.mediamtx_host}:{self.mediamtx_api_port}/v3"
            api_url_patch = f"{base}/config/paths/patch/{stream_path}"
            api_url_add = f"{base}/config/paths/add/{stream_path}"
            api_url_get = f"{base}/config/paths/get/{stream_path}"

            config_data: Dict[str, Any] = {
                "source": rtsp_url,
                "sourceProtocol": "tcp",
                "runOnInit": f"echo 'Stream {stream_path} initialized'",
                "runOnInitRestart": True,
            }

            async with aiohttp.ClientSession() as session:
                auth = aiohttp.BasicAuth(self.api_username, self.api_password)

                # Before PATCHing, read the existing path config so we can
                # carry forward the current ``record`` value.
                existing_record: Optional[bool] = None
                try:
                    async with session.get(api_url_get, auth=auth) as get_resp:
                        if get_resp.status == 200:
                            existing = await get_resp.json()
                            existing_record = bool(existing.get("record", False))
                except Exception:
                    pass

                if existing_record is not None:
                    config_data["record"] = existing_record

                # Try PATCH first
                async with session.patch(api_url_patch, json=config_data, auth=auth) as response:
                    if response.status == 200:
                        logger.info(f"Updated stream source: {stream_path} -> {rtsp_url}")
                        return True
                    elif response.status != 404:
                        try:
                            body = await response.text()
                        except Exception:
                            body = ''
                        logger.warning(f"Failed to patch stream source ({response.status}): {body}")

                # PATCH returned 404 -- path doesn't exist yet, so ADD it.
                async with session.post(api_url_add, json=config_data, auth=auth) as response:
                    if response.status in (200, 201):
                        logger.info(f"Added stream source: {stream_path} -> {rtsp_url}")
                        return True
                    elif response.status == 400:
                        try:
                            body = await response.text()
                        except Exception:
                            body = ''
                        if "path already exists" in (body or "").lower() and not force_recreate:
                            logger.info(f"Stream path already exists for {stream_path}; treating as configured")
                            return True
                        logger.error(f"Failed to add stream source ({response.status}): {body}")
                        return False
                    else:
                        try:
                            body = await response.text()
                        except Exception:
                            body = ''
                        logger.error(f"Failed to add stream source ({response.status}): {body}")
                        return False

        except Exception as e:
            logger.error(f"Error configuring stream source: {e}")
            return False

    async def get_path_info(self, stream_path: str) -> Dict[str, Any]:
        """
        Fetch runtime path info from the MediaMTX API.

        Returns a dict that includes at least:
          - ready: bool (best-effort)
        """
        try:
            stream_path = str(stream_path or "").lstrip("/").rstrip("/")
            if not stream_path:
                return {"ready": False}

            api_url = f"http://{self.mediamtx_host}:{self.mediamtx_api_port}/v3/paths/get/{stream_path}"
            async with aiohttp.ClientSession() as session:
                auth = aiohttp.BasicAuth(self.api_username, self.api_password)
                async with session.get(api_url, auth=auth) as response:
                    if response.status == 200:
                        try:
                            data = await response.json()
                        except Exception:
                            txt = await response.text()
                            return {"ready": False, "error": "Invalid JSON from MediaMTX", "raw": txt}
                        if isinstance(data, dict) and "ready" in data:
                            return data
                        # Be defensive: some versions might not include "ready" in the payload.
                        return {**(data if isinstance(data, dict) else {}), "ready": bool((data or {}).get("ready", False))}
                    if response.status == 404:
                        return {"ready": False, "not_found": True}
                    try:
                        body = await response.text()
                    except Exception:
                        body = ""
                    return {"ready": False, "status": response.status, "error": body}
        except Exception as e:
            logger.error(f"Error fetching MediaMTX path info for {stream_path}: {e}")
            return {"ready": False, "error": str(e)}

    def get_connection_stats(self, stream_path: str) -> Optional[ConnectionStats]:
        """Get connection statistics for a stream"""
        return self.stats.get(stream_path)

    def get_all_streams(self) -> List[str]:
        """Get list of all connected stream paths"""
        return list(self.connections.keys())

    def is_stream_connected(self, stream_path: str) -> bool:
        """Check if a stream is connected"""
        if stream_path not in self.connections:
            return False

        pc = self.connections[stream_path]
        return pc.connectionState in ["connected", "connecting"]

    async def reconnect_stream(self, stream_path: str) -> bool:
        """
        Reconnect to a stream using stored configuration

        Args:
            stream_path: Stream path to reconnect

        Returns:
            True if reconnection successful
        """
        config = self.stream_configs.get(stream_path)
        if not config:
            logger.error(f"No configuration found for stream: {stream_path}")
            return False

        # Disconnect existing connection
        await self.disconnect_stream(stream_path)

        # Wait before reconnecting
        await asyncio.sleep(config.reconnect_delay)

        # Attempt reconnection
        return await self.connect_stream(config)

    def _setup_peer_connection_handlers(self, pc: RTCPeerConnection, stream_path: str):
        """Set up event handlers for peer connection"""

        @pc.on("connectionstatechange")
        async def on_connectionstatechange():
            logger.info(f"Connection state changed for {stream_path}: {pc.connectionState}")

            if stream_path in self.stats:
                self.stats[stream_path].connection_state = pc.connectionState
                self.stats[stream_path].last_activity = datetime.now()

            if pc.connectionState == "failed":
                logger.warning(f"Connection failed for stream: {stream_path}")
                await self._handle_connection_failure(stream_path)

        @pc.on("track")
        def on_track(track):
            logger.info(f"Received {track.kind} track for stream: {stream_path}")

            if track.kind == "video":
                self.streams[stream_path] = track

                # Set up frame callback if provided
                if self.on_frame_received:
                    @track.on("frame")
                    def on_frame(frame):
                        asyncio.create_task(
                            self._safe_callback(self.on_frame_received, stream_path, frame)
                        )

        @pc.on("datachannel")
        def on_datachannel(channel):
            logger.info(f"Data channel received for stream: {stream_path}")

    async def _handle_websocket_messages(self, websocket, stream_path: str):
        """Handle incoming WebSocket messages"""
        try:
            async for message in websocket:
                try:
                    data = json.loads(message)
                    await self._process_websocket_message(stream_path, data)
                except json.JSONDecodeError:
                    logger.warning(f"Invalid JSON received for stream {stream_path}: {message}")
                except Exception as e:
                    logger.error(f"Error processing WebSocket message for {stream_path}: {e}")

        except websockets.exceptions.ConnectionClosed:
            logger.info(f"WebSocket connection closed for stream: {stream_path}")
        except Exception as e:
            logger.error(f"WebSocket error for stream {stream_path}: {e}")

    async def _process_websocket_message(self, stream_path: str, data: dict):
        """Process WebSocket messages from MediaMTX"""
        message_type = data.get("type")

        if message_type == "ice-candidate":
            (_RTCPeerConnection_t,
             _RTCSessionDescription_t,
             _VideoStreamTrack_t,
             RTCIceCandidate_t,
             _RTCIceServer_t,
             _RTCConfiguration_t) = _require_aiortc()
            # Handle ICE candidate
            pc = self.connections.get(stream_path)
            if pc:
                candidate = RTCIceCandidate_t(
                    candidate=data["candidate"],
                    sdpMid=data.get("sdpMid"),
                    sdpMLineIndex=data.get("sdpMLineIndex")
                )
                await pc.addIceCandidate(candidate)
                logger.debug(f"Added ICE candidate for stream: {stream_path}")

        elif message_type == "error":
            error_msg = data.get("message", "Unknown error")
            logger.error(f"MediaMTX error for stream {stream_path}: {error_msg}")

            if self.on_stream_error:
                await self._safe_callback(self.on_stream_error, stream_path, error_msg)

    async def _handle_connection_failure(self, stream_path: str):
        """Handle connection failure and attempt reconnection"""
        config = self.stream_configs.get(stream_path)
        if not config:
            return

        logger.info(f"Attempting to reconnect stream: {stream_path}")

        for attempt in range(config.reconnect_attempts):
            logger.info(f"Reconnection attempt {attempt + 1}/{config.reconnect_attempts} for {stream_path}")

            if await self.reconnect_stream(stream_path):
                logger.info(f"Successfully reconnected stream: {stream_path}")
                return

            if attempt < config.reconnect_attempts - 1:
                await asyncio.sleep(config.reconnect_delay * (attempt + 1))

        logger.error(f"Failed to reconnect stream after {config.reconnect_attempts} attempts: {stream_path}")
        await self._cleanup_stream(stream_path)

    async def _cleanup_stream(self, stream_path: str):
        """Clean up resources for a stream"""
        try:
            # Close peer connection
            if stream_path in self.connections:
                pc = self.connections[stream_path]
                await pc.close()
                del self.connections[stream_path]

            # Close WebSocket
            if stream_path in self.websockets:
                websocket = self.websockets[stream_path]
                await websocket.close()
                del self.websockets[stream_path]

            # Remove WHEP session
            if stream_path in self.whep_session_urls:
                del self.whep_session_urls[stream_path]

            # Remove stream track
            if stream_path in self.streams:
                del self.streams[stream_path]

            # Clean up stats but keep config for potential reconnection
            if stream_path in self.stats:
                del self.stats[stream_path]

        except Exception as e:
            logger.error(f"Error cleaning up stream {stream_path}: {e}")

    async def _monitor_connections(self):
        """Monitor connection health"""
        while self._running:
            try:
                current_time = datetime.now()

                for stream_path, stats in list(self.stats.items()):
                    # Check for stale connections
                    if stats.last_activity:
                        time_since_activity = current_time - stats.last_activity
                        if time_since_activity > timedelta(minutes=5):
                            logger.warning(f"Stream {stream_path} appears stale, attempting reconnection")
                            await self._handle_connection_failure(stream_path)

                await asyncio.sleep(30)  # Check every 30 seconds

            except Exception as e:
                logger.error(f"Error in connection monitoring: {e}")
                await asyncio.sleep(60)

    async def _update_stats(self):
        """Update connection statistics"""
        while self._running:
            try:
                for stream_path, pc in self.connections.items():
                    if stream_path in self.stats:
                        # Update basic stats
                        self.stats[stream_path].connection_state = pc.connectionState

                        # Get detailed stats if available
                        try:
                            stats_report = await pc.getStats()
                            # Process WebRTC stats here if needed
                        except Exception:
                            pass  # Stats not available

                await asyncio.sleep(10)  # Update every 10 seconds

            except Exception as e:
                logger.error(f"Error updating stats: {e}")
                await asyncio.sleep(30)

    async def _safe_callback(self, callback: Callable, *args, **kwargs):
        """Safely execute callback without breaking the main flow"""
        try:
            if asyncio.iscoroutinefunction(callback):
                await callback(*args, **kwargs)
            else:
                callback(*args, **kwargs)
        except Exception as e:
            logger.error(f"Error in callback: {e}")

    # ===========================
    # HTTP-only WHEP helpers (browser viewer sessions)
    # ===========================

    async def create_webrtc_answer(self, stream_path: str, offer: Dict[str, Any], client_id: str) -> Dict[str, Any]:
        """
        Create a WebRTC answer for a browser offer by proxying the offer SDP to MediaMTX WHEP.
        This path is HTTP-only (no aiortc) and is preferred for the browser-side WebRTC flow.
        """
        if not isinstance(offer, dict) or not offer.get("sdp"):
            raise ValueError("Invalid offer (expected dict with 'sdp').")

        offer_sdp = str(offer.get("sdp") or "")
        whep_url = f"http://{self.mediamtx_host}:{self.mediamtx_webrtc_port}/{stream_path}/whep"
        headers = {
            "Content-Type": "application/sdp",
            "Accept": "application/sdp",
            **(self._basic_auth_header or {})
        }

        async with aiohttp.ClientSession() as session:
            async with session.post(whep_url, data=offer_sdp, headers=headers) as resp:
                if resp.status < 200 or resp.status >= 300:
                    body = ""
                    try:
                        body = await resp.text()
                    except Exception:
                        body = ""
                    raise RuntimeError(f"WHEP HTTP error {resp.status}: {body}")

                answer_sdp = await resp.text()
                if not answer_sdp or not answer_sdp.lstrip().startswith("v="):
                    raise RuntimeError("Invalid WHEP answer SDP")

                loc = resp.headers.get("Location") or resp.headers.get("location")
                if loc:
                    try:
                        self.whep_viewer_session_urls[(stream_path, client_id)] = urljoin(whep_url, loc)
                    except Exception:
                        self.whep_viewer_session_urls[(stream_path, client_id)] = loc

        return {"type": "answer", "sdp": answer_sdp}

    async def add_ice_candidate(self, stream_path: str, candidate: Dict[str, Any], client_id: str) -> bool:
        """
        Forward a trickle ICE candidate to MediaMTX for an existing WHEP session via HTTP PATCH.
        """
        session_url = self.whep_viewer_session_urls.get((stream_path, client_id))
        if not session_url:
            # Without a session URL, we can't PATCH candidates.
            raise RuntimeError("WHEP session URL not found for this client; cannot add ICE candidate.")

        cand_str = (candidate or {}).get("candidate")
        sdp_mid = (candidate or {}).get("sdpMid")

        if not cand_str:
            return True

        # Build minimal SDP fragment (sdpfrag) for trickle ICE.
        # MediaMTX supports 'application/trickle-ice-sdpfrag' PATCH for WHIP/WHEP sessions.
        lines = []
        if sdp_mid is not None:
            lines.append(f"a=mid:{sdp_mid}")
        lines.append(f"a={cand_str}")
        body = "\r\n".join(lines) + "\r\n"

        headers = {
            "Content-Type": "application/trickle-ice-sdpfrag",
            "Accept": "*/*",
            **(self._basic_auth_header or {})
        }

        async with aiohttp.ClientSession() as session:
            async with session.patch(session_url, data=body, headers=headers) as resp:
                if resp.status in (200, 204):
                    return True
                # Some servers return 404 if session already closed; treat as not fatal.
                if resp.status == 404:
                    return False
                body_txt = ""
                try:
                    body_txt = await resp.text()
                except Exception:
                    body_txt = ""
                raise RuntimeError(f"WHEP ICE PATCH error {resp.status}: {body_txt}")

    async def close_webrtc_session(self, stream_path: str, client_id: str) -> None:
        """Best-effort close a WHEP viewer session created by create_webrtc_answer()."""
        session_url = self.whep_viewer_session_urls.get((stream_path, client_id))
        if not session_url:
            return
        headers = {**(self._basic_auth_header or {})}
        try:
            async with aiohttp.ClientSession() as session:
                async with session.delete(session_url, headers=headers) as resp:
                    _ = resp.status
        finally:
            try:
                del self.whep_viewer_session_urls[(stream_path, client_id)]
            except Exception:
                pass


# Convenience factory function
def create_mediamtx_client(host: str = "localhost",
                           webrtc_port: int = 8889,
                           api_port: int = 9997) -> MediaMTXWebRTCClient:
    """
    Create a MediaMTX WebRTC client with default configuration

    Args:
        host: MediaMTX server host
        webrtc_port: WebRTC port
        api_port: API port

    Returns:
        Configured MediaMTXWebRTCClient instance
    """
    return MediaMTXWebRTCClient(
        mediamtx_host=host,
        mediamtx_webrtc_port=webrtc_port,
        mediamtx_api_port=api_port
    )