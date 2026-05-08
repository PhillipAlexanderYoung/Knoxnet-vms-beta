import asyncio
import json
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List
from flask import Blueprint, request, jsonify, Response
from flask_cors import cross_origin
import uuid
import websockets
from dataclasses import asdict

from .routes import _run_coro_safe
from core.camera_manager import CameraManager, StreamQuality, CameraStatus
from core.stream_server import StreamServer, ConnectionType
from core.mediamtx_client import MediaMTXWebRTCClient, StreamConfig

logger = logging.getLogger(__name__)

# Create Blueprint for WebRTC routes
webrtc_bp = Blueprint('webrtc_api', __name__, url_prefix='/api/webrtc')

# Global references (will be injected by main app)
camera_manager: Optional[CameraManager] = None
stream_server: Optional[StreamServer] = None
mediamtx_client: Optional[MediaMTXWebRTCClient] = None


def init_webrtc_routes(cam_manager: CameraManager, strm_server: StreamServer):
    """Initialize WebRTC routes with required components"""
    global camera_manager, stream_server, mediamtx_client
    camera_manager = cam_manager
    stream_server = strm_server
    mediamtx_client = cam_manager.mediamtx_client if cam_manager else None


@webrtc_bp.route('/offer', methods=['POST', 'OPTIONS'])
@cross_origin()
def handle_webrtc_offer():
    """
    Handle WebRTC offer from client

    Expected payload:
    {
        "camera_id": "camera_001",
        "offer": {
            "type": "offer",
            "sdp": "v=0\r\no=- ..."
        },
        "quality": "medium",
        "client_info": {
            "user_agent": "...",
            "ip_address": "..."
        }
    }

    Returns:
    {
        "success": true,
        "answer": {
            "type": "answer",
            "sdp": "v=0\r\no=- ..."
        },
        "session_id": "uuid",
        "ice_servers": [...],
        "stream_url": "ws://..."
    }
    """
    try:
        if request.method == 'OPTIONS':
            return '', 200

        if not camera_manager or not stream_server:
            return jsonify({
                "success": False,
                "error": "WebRTC services not initialized"
            }), 500

        data = request.get_json()
        if not data:
            return jsonify({
                "success": False,
                "error": "Invalid JSON payload"
            }), 400

        camera_id = data.get('camera_id')
        offer = data.get('offer')
        quality = data.get('quality', 'medium')
        client_info = data.get('client_info', {})

        # Validate required fields
        if not camera_id or not offer:
            return jsonify({
                "success": False,
                "error": "Missing required fields: camera_id, offer"
            }), 400

        # Validate camera exists and is connected
        if not camera_manager.is_camera_connected(camera_id):
            return jsonify({
                "success": False,
                "error": f"Camera {camera_id} not available"
            }), 404

        # Validate quality
        try:
            stream_quality = StreamQuality(quality)
        except ValueError:
            return jsonify({
                "success": False,
                "error": f"Invalid quality: {quality}. Must be one of: low, medium, high, ultra"
            }), 400

        # Get WebRTC stream URL
        webrtc_url = _run_coro_safe(camera_manager.get_webrtc_stream_url(camera_id))
        if not webrtc_url:
            return jsonify({
                "success": False,
                "error": f"WebRTC not available for camera {camera_id}"
            }), 503

        # Generate session ID
        session_id = str(uuid.uuid4())

        # Connect client to stream server
        client_info_extended = {
            **client_info,
            "ip_address": request.remote_addr,
            "user_agent": request.headers.get('User-Agent'),
            "session_id": session_id
        }

        client_id = _run_coro_safe(stream_server.connect_client(
            camera_id=camera_id,
            connection_type=ConnectionType.WEBRTC,
            quality=stream_quality,
            client_info=client_info_extended
        ))

        if not client_id:
            return jsonify({
                "success": False,
                "error": "Failed to establish WebRTC connection"
            }), 503

        # Return WebRTC connection details
        response_data = {
            "success": True,
            "session_id": session_id,
            "client_id": client_id,
            "camera_id": camera_id,
            "webrtc_url": webrtc_url,
            "quality": quality,
            "ice_servers": mediamtx_client.ice_servers if mediamtx_client else [],
            "connection_info": {
                "signaling_url": f"ws://{request.host}/ws/webrtc/signaling",
                "stream_path": camera_id,
                "whep_endpoint": webrtc_url
            },
            "timestamp": datetime.now().isoformat()
        }

        logger.info(f"WebRTC offer processed for camera {camera_id}, session {session_id}")

        return jsonify(response_data), 200

    except Exception as e:
        logger.error(f"Error handling WebRTC offer: {e}")
        return jsonify({
            "success": False,
            "error": "Internal server error processing WebRTC offer"
        }), 500


@webrtc_bp.route('/ice-candidate', methods=['POST', 'OPTIONS'])
@cross_origin()
def handle_ice_candidate():
    """
    Handle ICE candidate from client

    Expected payload:
    {
        "session_id": "uuid",
        "candidate": {
            "candidate": "candidate:...",
            "sdpMid": "0",
            "sdpMLineIndex": 0
        }
    }

    Returns:
    {
        "success": true,
        "message": "ICE candidate processed"
    }
    """
    try:
        if request.method == 'OPTIONS':
            return '', 200

        data = request.get_json()
        if not data:
            return jsonify({
                "success": False,
                "error": "Invalid JSON payload"
            }), 400

        session_id = data.get('session_id')
        candidate = data.get('candidate')

        if not session_id or not candidate:
            return jsonify({
                "success": False,
                "error": "Missing required fields: session_id, candidate"
            }), 400

        # ICE candidates are handled directly by MediaMTX via WebSocket
        # This endpoint mainly serves as a validation/logging point

        logger.debug(f"ICE candidate received for session {session_id}")

        return jsonify({
            "success": True,
            "message": "ICE candidate processed",
            "session_id": session_id
        }), 200

    except Exception as e:
        logger.error(f"Error handling ICE candidate: {e}")
        return jsonify({
            "success": False,
            "error": "Internal server error processing ICE candidate"
        }), 500


@webrtc_bp.route('/disconnect', methods=['POST', 'OPTIONS'])
@cross_origin()
def disconnect_webrtc():
    """
    Disconnect WebRTC session

    Expected payload:
    {
        "session_id": "uuid",
        "client_id": "uuid"
    }

    Returns:
    {
        "success": true,
        "message": "Session disconnected"
    }
    """
    try:
        if request.method == 'OPTIONS':
            return '', 200

        data = request.get_json()
        if not data:
            return jsonify({
                "success": False,
                "error": "Invalid JSON payload"
            }), 400

        session_id = data.get('session_id')
        client_id = data.get('client_id')

        if not session_id and not client_id:
            return jsonify({
                "success": False,
                "error": "Either session_id or client_id must be provided"
            }), 400

        if not stream_server:
            return jsonify({
                "success": False,
                "error": "Stream server not initialized"
            }), 500

        # Disconnect client
        if client_id:
            success = _run_coro_safe(stream_server.disconnect_client(client_id))
            if success:
                logger.info(f"WebRTC client {client_id} disconnected")
                return jsonify({
                    "success": True,
                    "message": "Client disconnected",
                    "client_id": client_id
                }), 200
            else:
                return jsonify({
                    "success": False,
                    "error": "Failed to disconnect client"
                }), 500

        return jsonify({
            "success": True,
            "message": "Disconnect request processed"
        }), 200

    except Exception as e:
        logger.error(f"Error disconnecting WebRTC session: {e}")
        return jsonify({
            "success": False,
            "error": "Internal server error disconnecting session"
        }), 500


@webrtc_bp.route('/streams', methods=['GET'])
@cross_origin()
def get_webrtc_streams():
    """
    Get list of available WebRTC streams

    Returns:
    {
        "success": true,
        "streams": [
            {
                "camera_id": "camera_001",
                "name": "Gate",
                "location": "Entrance",
                "webrtc_url": "ws://...",
                "status": "connected",
                "quality_levels": ["low", "medium", "high"],
                "current_clients": 3,
                "webrtc_enabled": true
            }
        ]
    }
    """
    try:
        if not camera_manager or not stream_server:
            return jsonify({
                "success": False,
                "error": "Services not initialized"
            }), 500

        streams = []
        cameras = camera_manager.get_all_cameras()

        for camera_id, config in cameras.items():
            if not config.webrtc_enabled:
                continue

            # Get camera health
            health = camera_manager.get_camera_health(camera_id)
            if not health or health.status != CameraStatus.CONNECTED:
                continue

            # Get stream info
            stream_info = stream_server.get_stream_info(camera_id)
            webrtc_url = _run_coro_safe(camera_manager.get_webrtc_stream_url(camera_id))

            stream_data = {
                "camera_id": camera_id,
                "name": config.name,
                "location": config.location,
                "webrtc_url": webrtc_url,
                "status": health.status.value if health else "unknown",
                "quality_levels": [q.value for q in StreamQuality],
                "current_clients": stream_info.get('client_count', 0) if stream_info else 0,
                "webrtc_enabled": config.webrtc_enabled,
                "webrtc_connected": health.webrtc_connected if health else False,
                "stream_quality": config.stream_quality.value,
                "audio_enabled": config.audio_enabled,
                "last_frame_time": health.last_frame_time.isoformat() if health and health.last_frame_time else None
            }

            streams.append(stream_data)

        return jsonify({
            "success": True,
            "streams": streams,
            "total_streams": len(streams),
            "timestamp": datetime.now().isoformat()
        }), 200

    except Exception as e:
        logger.error(f"Error getting WebRTC streams: {e}")
        return jsonify({
            "success": False,
            "error": "Internal server error getting streams"
        }), 500


@webrtc_bp.route('/stream/<camera_id>', methods=['GET'])
@cross_origin()
def get_webrtc_stream_info(camera_id: str):
    """
    Get detailed WebRTC stream information for specific camera

    Returns:
    {
        "success": true,
        "camera_id": "camera_001",
        "stream_info": {
            "webrtc_url": "ws://...",
            "ice_servers": [...],
            "connection_stats": {...},
            "quality_options": [...],
            "current_quality": "medium"
        }
    }
    """
    try:
        if not camera_manager or not stream_server:
            return jsonify({
                "success": False,
                "error": "Services not initialized"
            }), 500

        # Validate camera exists
        camera = camera_manager.get_camera(camera_id)
        if not camera:
            return jsonify({
                "success": False,
                "error": f"Camera {camera_id} not found"
            }), 404

        # Check if WebRTC is enabled
        if not camera.webrtc_enabled:
            return jsonify({
                "success": False,
                "error": f"WebRTC not enabled for camera {camera_id}"
            }), 403

        # Get WebRTC URL
        webrtc_url = _run_coro_safe(camera_manager.get_webrtc_stream_url(camera_id))
        if not webrtc_url:
            return jsonify({
                "success": False,
                "error": f"WebRTC stream not available for camera {camera_id}"
            }), 503

        # Get connection stats
        health = camera_manager.get_camera_health(camera_id)
        stream_info = stream_server.get_stream_info(camera_id)

        # Get MediaMTX connection stats
        mediamtx_stats = None
        if mediamtx_client:
            stream_path = camera_id
            mediamtx_stats = mediamtx_client.get_connection_stats(stream_path)

        response_data = {
            "success": True,
            "camera_id": camera_id,
            "camera_name": camera.name,
            "stream_info": {
                "webrtc_url": webrtc_url,
                "ice_servers": mediamtx_client.ice_servers if mediamtx_client else [],
                "signaling_url": f"ws://{request.host}/ws/webrtc/signaling",
                "stream_path": camera_id,
                "quality_options": [q.value for q in StreamQuality],
                "current_quality": camera.stream_quality.value,
                "audio_enabled": camera.audio_enabled,
                "connection_stats": {
                    "status": health.status.value if health else "unknown",
                    "webrtc_connected": health.webrtc_connected if health else False,
                    "frame_rate": health.frame_rate if health else 0,
                    "total_frames": health.total_frames if health else 0,
                    "error_count": health.error_count if health else 0,
                    "last_error": health.last_error if health else None,
                    "uptime": str(health.connection_uptime) if health and health.connection_uptime else "0:00:00"
                },
                "stream_stats": stream_info if stream_info else {},
                "mediamtx_stats": asdict(mediamtx_stats) if mediamtx_stats else None
            },
            "timestamp": datetime.now().isoformat()
        }

        return jsonify(response_data), 200

    except Exception as e:
        logger.error(f"Error getting WebRTC stream info for {camera_id}: {e}")
        return jsonify({
            "success": False,
            "error": "Internal server error getting stream info"
        }), 500


@webrtc_bp.route('/quality/<camera_id>', methods=['PUT', 'OPTIONS'])
@cross_origin()
def update_stream_quality(camera_id: str):
    """
    Update stream quality for WebRTC stream

    Expected payload:
    {
        "quality": "high",
        "apply_to_clients": true
    }

    Returns:
    {
        "success": true,
        "camera_id": "camera_001",
        "old_quality": "medium",
        "new_quality": "high"
    }
    """
    try:
        if request.method == 'OPTIONS':
            return '', 200

        if not camera_manager or not stream_server:
            return jsonify({
                "success": False,
                "error": "Services not initialized"
            }), 500

        data = request.get_json()
        if not data:
            return jsonify({
                "success": False,
                "error": "Invalid JSON payload"
            }), 400

        new_quality = data.get('quality')
        apply_to_clients = data.get('apply_to_clients', False)

        if not new_quality:
            return jsonify({
                "success": False,
                "error": "Missing required field: quality"
            }), 400

        # Validate quality
        try:
            stream_quality = StreamQuality(new_quality)
        except ValueError:
            return jsonify({
                "success": False,
                "error": f"Invalid quality: {new_quality}. Must be one of: low, medium, high, ultra"
            }), 400

        # Get current camera
        camera = camera_manager.get_camera(camera_id)
        if not camera:
            return jsonify({
                "success": False,
                "error": f"Camera {camera_id} not found"
            }), 404

        old_quality = camera.stream_quality.value

        # Update camera quality
        success = _run_coro_safe(camera_manager.set_stream_quality(camera_id, stream_quality))
        if not success:
            return jsonify({
                "success": False,
                "error": "Failed to update camera stream quality"
            }), 500

        # Update stream server quality
        _run_coro_safe(stream_server.update_stream_quality(camera_id, stream_quality))

        response_data = {
            "success": True,
            "camera_id": camera_id,
            "old_quality": old_quality,
            "new_quality": new_quality,
            "apply_to_clients": apply_to_clients,
            "message": f"Stream quality updated from {old_quality} to {new_quality}",
            "timestamp": datetime.now().isoformat()
        }

        logger.info(f"Updated stream quality for camera {camera_id}: {old_quality} -> {new_quality}")

        return jsonify(response_data), 200

    except Exception as e:
        logger.error(f"Error updating stream quality for {camera_id}: {e}")
        return jsonify({
            "success": False,
            "error": "Internal server error updating stream quality"
        }), 500


@webrtc_bp.route('/connections', methods=['GET'])
@cross_origin()
def get_webrtc_connections():
    """
    Get active WebRTC connections information

    Returns:
    {
        "success": true,
        "connections": [
            {
                "client_id": "uuid",
                "camera_id": "camera_001",
                "quality": "medium",
                "connected_at": "2024-01-01T12:00:00Z",
                "bytes_sent": 1024000,
                "frames_sent": 300
            }
        ],
        "total_connections": 5,
        "bandwidth_usage": {...}
    }
    """
    try:
        if not stream_server:
            return jsonify({
                "success": False,
                "error": "Stream server not initialized"
            }), 500

        # Get all client connections
        all_connections = stream_server.get_client_connections()

        # Filter WebRTC connections
        webrtc_connections = []
        for client_id, conn_info in all_connections.items():
            if conn_info['type'] == 'webrtc':
                webrtc_connections.append({
                    "client_id": client_id,
                    "camera_id": conn_info['camera_id'],
                    "quality": conn_info['quality'],
                    "connected_at": conn_info['connected_at'],
                    "last_activity": conn_info['last_activity'],
                    "bytes_sent": conn_info['bytes_sent'],
                    "frames_sent": conn_info['frames_sent'],
                    "ip_address": conn_info.get('ip_address'),
                    "user_agent": conn_info.get('user_agent')
                })

        # Get bandwidth usage
        bandwidth_info = stream_server.get_bandwidth_usage()

        return jsonify({
            "success": True,
            "connections": webrtc_connections,
            "total_connections": len(webrtc_connections),
            "bandwidth_usage": bandwidth_info,
            "timestamp": datetime.now().isoformat()
        }), 200

    except Exception as e:
        logger.error(f"Error getting WebRTC connections: {e}")
        return jsonify({
            "success": False,
            "error": "Internal server error getting connections"
        }), 500


@webrtc_bp.route('/connection/<client_id>', methods=['GET'])
@cross_origin()
def get_webrtc_connection_status(client_id: str):
    """
    Get specific WebRTC connection status

    Returns:
    {
        "success": true,
        "connection": {
            "client_id": "uuid",
            "camera_id": "camera_001",
            "status": "connected",
            "quality": "medium",
            "stats": {...}
        }
    }
    """
    try:
        if not stream_server:
            return jsonify({
                "success": False,
                "error": "Stream server not initialized"
            }), 500

        # Get connection info
        connections = stream_server.get_client_connections()
        connection = connections.get(client_id)

        if not connection:
            return jsonify({
                "success": False,
                "error": f"Connection {client_id} not found"
            }), 404

        # Only return WebRTC connections
        if connection['type'] != 'webrtc':
            return jsonify({
                "success": False,
                "error": f"Connection {client_id} is not a WebRTC connection"
            }), 400

        return jsonify({
            "success": True,
            "connection": connection,
            "timestamp": datetime.now().isoformat()
        }), 200

    except Exception as e:
        logger.error(f"Error getting WebRTC connection status for {client_id}: {e}")
        return jsonify({
            "success": False,
            "error": "Internal server error getting connection status"
        }), 500


@webrtc_bp.route('/stats', methods=['GET'])
@cross_origin()
def get_webrtc_statistics():
    """
    Get comprehensive WebRTC statistics

    Returns:
    {
        "success": true,
        "statistics": {
            "total_streams": 5,
            "active_connections": 12,
            "bandwidth_usage": {...},
            "stream_breakdown": {...},
            "performance_metrics": {...}
        }
    }
    """
    try:
        if not camera_manager or not stream_server:
            return jsonify({
                "success": False,
                "error": "Services not initialized"
            }), 500

        # Get stream statistics
        all_streams = stream_server.get_all_streams()
        webrtc_enabled_cameras = camera_manager.get_webrtc_enabled_cameras()

        # Get connection statistics
        all_connections = stream_server.get_client_connections()
        webrtc_connections = {
            k: v for k, v in all_connections.items()
            if v['type'] == 'webrtc'
        }

        # Calculate statistics
        total_webrtc_streams = len([s for s in all_streams.values() if s.get('webrtc_enabled')])
        total_webrtc_connections = len(webrtc_connections)

        # Bandwidth usage
        bandwidth_info = stream_server.get_bandwidth_usage()

        # Stream breakdown by quality
        quality_breakdown = {}
        for conn in webrtc_connections.values():
            quality = conn['quality']
            quality_breakdown[quality] = quality_breakdown.get(quality, 0) + 1

        # Camera breakdown
        camera_breakdown = {}
        for conn in webrtc_connections.values():
            camera_id = conn['camera_id']
            camera_breakdown[camera_id] = camera_breakdown.get(camera_id, 0) + 1

        # Performance metrics
        total_bytes_sent = sum(conn['bytes_sent'] for conn in webrtc_connections.values())
        total_frames_sent = sum(conn['frames_sent'] for conn in webrtc_connections.values())

        statistics = {
            "webrtc_streams": {
                "total_streams": total_webrtc_streams,
                "enabled_cameras": len(webrtc_enabled_cameras),
                "active_connections": total_webrtc_connections
            },
            "bandwidth_usage": bandwidth_info,
            "quality_breakdown": quality_breakdown,
            "camera_breakdown": camera_breakdown,
            "performance_metrics": {
                "total_bytes_sent": total_bytes_sent,
                "total_frames_sent": total_frames_sent,
                "average_bytes_per_connection": total_bytes_sent / max(total_webrtc_connections, 1),
                "average_frames_per_connection": total_frames_sent / max(total_webrtc_connections, 1)
            },
            "mediamtx_status": {
                "connected_streams": len(mediamtx_client.get_all_streams()) if mediamtx_client else 0,
                "client_running": mediamtx_client._running if mediamtx_client else False
            }
        }

        return jsonify({
            "success": True,
            "statistics": statistics,
            "timestamp": datetime.now().isoformat()
        }), 200

    except Exception as e:
        logger.error(f"Error getting WebRTC statistics: {e}")
        return jsonify({
            "success": False,
            "error": "Internal server error getting statistics"
        }), 500


@webrtc_bp.route('/health', methods=['GET'])
@cross_origin()
def webrtc_health_check():
    """
    WebRTC service health check

    Returns:
    {
        "success": true,
        "health": {
            "mediamtx_client": "connected",
            "stream_server": "running",
            "active_streams": 5,
            "active_connections": 12
        }
    }
    """
    try:
        health_status = {
            "mediamtx_client": "disconnected",
            "stream_server": "stopped",
            "active_streams": 0,
            "active_connections": 0,
            "services_initialized": bool(camera_manager and stream_server)
        }

        if mediamtx_client:
            health_status["mediamtx_client"] = "connected" if mediamtx_client._running else "disconnected"
            health_status["mediamtx_streams"] = len(mediamtx_client.get_all_streams())

        if stream_server:
            health_status["stream_server"] = "running" if stream_server.running else "stopped"
            all_streams = stream_server.get_all_streams()
            health_status["active_streams"] = len(all_streams)

            all_connections = stream_server.get_client_connections()
            webrtc_connections = [c for c in all_connections.values() if c['type'] == 'webrtc']
            health_status["active_connections"] = len(webrtc_connections)

        # Determine overall health
        overall_healthy = (
                health_status["services_initialized"] and
                health_status["mediamtx_client"] == "connected" and
                health_status["stream_server"] == "running"
        )

        status_code = 200 if overall_healthy else 503

        return jsonify({
            "success": overall_healthy,
            "health": health_status,
            "overall_status": "healthy" if overall_healthy else "unhealthy",
            "timestamp": datetime.now().isoformat()
        }), status_code

    except Exception as e:
        logger.error(f"Error checking WebRTC health: {e}")
        return jsonify({
            "success": False,
            "error": "Internal server error checking health",
            "health": {"status": "error"}
        }), 500


# Error handlers
@webrtc_bp.errorhandler(404)
def not_found(error):
    return jsonify({
        "success": False,
        "error": "WebRTC endpoint not found"
    }), 404


@webrtc_bp.errorhandler(500)
def internal_error(error):
    return jsonify({
        "success": False,
        "error": "Internal server error in WebRTC service"
    }), 500


# Export blueprint
__all__ = ['webrtc_bp', 'init_webrtc_routes']