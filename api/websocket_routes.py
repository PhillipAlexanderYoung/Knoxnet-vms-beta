"""
WebSocket Management API Routes

This module provides API endpoints for managing websocket connections,
monitoring connection health, and controlling the connection manager.
"""

from flask import Blueprint, jsonify, request
import logging
from datetime import datetime
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

# Global reference to motion detection integration
motion_detection_integration = None

def init_websocket_routes(app, integration_instance=None):
    """Initialize websocket management routes"""
    global motion_detection_integration
    motion_detection_integration = integration_instance
    
    websocket_bp = Blueprint('websocket', __name__, url_prefix='/api/websocket')
    
    @websocket_bp.route('/status', methods=['GET'])
    def get_websocket_status():
        """Get overall websocket connection status"""
        try:
            if not motion_detection_integration:
                return jsonify({
                    'success': False,
                    'message': 'WebSocket integration not available'
                }), 503
            
            status = motion_detection_integration.get_connection_status()
            
            return jsonify({
                'success': True,
                'data': status
            })
            
        except Exception as e:
            logger.error(f"Error getting websocket status: {e}")
            return jsonify({
                'success': False,
                'message': f'Error getting websocket status: {str(e)}'
            }), 500
    
    @websocket_bp.route('/cameras/<camera_id>/status', methods=['GET'])
    def get_camera_connection_status(camera_id):
        """Get connection status for a specific camera"""
        try:
            if not motion_detection_integration:
                return jsonify({
                    'success': False,
                    'message': 'WebSocket integration not available'
                }), 503
            
            status = motion_detection_integration.get_connection_status(camera_id)
            
            if not status:
                return jsonify({
                    'success': False,
                    'message': f'Camera {camera_id} not found in connection manager'
                }), 404
            
            return jsonify({
                'success': True,
                'data': status
            })
            
        except Exception as e:
            logger.error(f"Error getting camera {camera_id} connection status: {e}")
            return jsonify({
                'success': False,
                'message': f'Error getting camera connection status: {str(e)}'
            }), 500
    
    @websocket_bp.route('/cameras/<camera_id>/activity', methods=['GET'])
    def get_camera_activity(camera_id):
        """Get activity data for a specific camera"""
        try:
            if not motion_detection_integration:
                return jsonify({
                    'success': False,
                    'message': 'WebSocket integration not available'
                }), 503
            
            activity = motion_detection_integration.get_camera_activity(camera_id)
            
            if not activity:
                return jsonify({
                    'success': False,
                    'message': f'Camera {camera_id} not found in activity tracking'
                }), 404
            
            return jsonify({
                'success': True,
                'data': activity
            })
            
        except Exception as e:
            logger.error(f"Error getting camera {camera_id} activity: {e}")
            return jsonify({
                'success': False,
                'message': f'Error getting camera activity: {str(e)}'
            }), 500
    
    @websocket_bp.route('/cameras/<camera_id>/motion-history', methods=['GET'])
    def get_camera_motion_history(camera_id):
        """Get motion event history for a specific camera"""
        try:
            if not motion_detection_integration:
                return jsonify({
                    'success': False,
                    'message': 'WebSocket integration not available'
                }), 503
            
            # Get limit parameter
            limit = request.args.get('limit', 50, type=int)
            
            history = motion_detection_integration.get_motion_event_history(camera_id)
            
            if not history:
                return jsonify({
                    'success': True,
                    'data': {
                        'camera_id': camera_id,
                        'events': [],
                        'total_events': 0
                    }
                })
            
            # Limit the number of events returned
            if limit and len(history) > limit:
                history = history[-limit:]
            
            return jsonify({
                'success': True,
                'data': {
                    'camera_id': camera_id,
                    'events': history,
                    'total_events': len(history)
                }
            })
            
        except Exception as e:
            logger.error(f"Error getting camera {camera_id} motion history: {e}")
            return jsonify({
                'success': False,
                'message': f'Error getting camera motion history: {str(e)}'
            }), 500
    
    @websocket_bp.route('/cameras/<camera_id>/reconnect', methods=['POST'])
    def force_reconnect_camera(camera_id):
        """Force reconnection of a specific camera"""
        try:
            if not motion_detection_integration:
                return jsonify({
                    'success': False,
                    'message': 'WebSocket integration not available'
                }), 503
            
            success = motion_detection_integration.force_reconnect_camera(camera_id)
            
            if success:
                return jsonify({
                    'success': True,
                    'message': f'Successfully initiated reconnection for camera {camera_id}'
                })
            else:
                return jsonify({
                    'success': False,
                    'message': f'Failed to reconnect camera {camera_id}'
                }), 400
            
        except Exception as e:
            logger.error(f"Error force reconnecting camera {camera_id}: {e}")
            return jsonify({
                'success': False,
                'message': f'Error force reconnecting camera: {str(e)}'
            }), 500
    
    @websocket_bp.route('/cameras/<camera_id>/motion-detection', methods=['PUT'])
    def update_motion_detection(camera_id):
        """Update motion detection setting for a camera"""
        try:
            if not motion_detection_integration:
                return jsonify({
                    'success': False,
                    'message': 'WebSocket integration not available'
                }), 503
            
            data = request.get_json()
            if not data:
                return jsonify({
                    'success': False,
                    'message': 'No data provided'
                }), 400
            
            enabled = data.get('enabled')
            if enabled is None:
                return jsonify({
                    'success': False,
                    'message': 'enabled field is required'
                }), 400
            
            success = motion_detection_integration.update_camera_motion_detection(camera_id, enabled)
            
            if success:
                return jsonify({
                    'success': True,
                    'message': f'Motion detection {"enabled" if enabled else "disabled"} for camera {camera_id}'
                })
            else:
                return jsonify({
                    'success': False,
                    'message': f'Failed to update motion detection for camera {camera_id}'
                }), 400
            
        except Exception as e:
            logger.error(f"Error updating motion detection for camera {camera_id}: {e}")
            return jsonify({
                'success': False,
                'message': f'Error updating motion detection: {str(e)}'
            }), 500
    
    @websocket_bp.route('/health', methods=['GET'])
    def get_websocket_health():
        """Get websocket health status"""
        try:
            if not motion_detection_integration:
                return jsonify({
                    'success': False,
                    'message': 'WebSocket integration not available'
                }), 503
            
            return jsonify({
                'success': True,
                'data': {
                    'status': 'healthy',
                    'websocket_enabled': True,
                    'timestamp': datetime.now().isoformat()
                }
            })
            
        except Exception as e:
            logger.error(f"Error getting websocket health: {e}")
            return jsonify({
                'success': False,
                'message': f'Error getting websocket health: {str(e)}'
            }), 500
    
    # Register the blueprint
    app.register_blueprint(websocket_bp)
    logger.info("WebSocket management routes registered")
