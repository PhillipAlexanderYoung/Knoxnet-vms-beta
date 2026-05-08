from __future__ import annotations

import logging
from flask import Blueprint, jsonify, request, current_app

from core.session_manager import SessionManager

logger = logging.getLogger(__name__)


def _emit_session_update(session_id: str):
    """
    Best-effort realtime push using the app's SocketIO instance (if available).
    """
    try:
        sio = None
        try:
            sio = current_app.extensions.get("socketio") if current_app else None
        except Exception:
            sio = None
        if not sio:
            return
        # room per session id
        sio.emit("session_update", {"session_id": session_id}, namespace="/sessions", room=f"session:{session_id}")
    except Exception:
        return


def init_sessions_routes(app, session_manager: SessionManager):
    sessions_bp = Blueprint("sessions_bp", __name__, url_prefix="/api/sessions")

    @sessions_bp.route("", methods=["GET"])
    def list_sessions():
        try:
            sessions = [s.to_dict() for s in session_manager.list_sessions()]
            return jsonify({"success": True, "data": {"sessions": sessions, "count": len(sessions)}})
        except Exception as e:
            logger.error(f"List sessions error: {e}")
            return jsonify({"success": False, "message": "Failed to list sessions"}), 500

    @sessions_bp.route("", methods=["POST"])
    def create_session():
        try:
            data = request.get_json() or {}
            layout_ids = data.get("layout_ids") or []
            if not isinstance(layout_ids, list) or not layout_ids:
                return jsonify({"success": False, "message": "layout_ids (list) is required"}), 400
            name = data.get("name")
            meta = data.get("meta") if isinstance(data.get("meta"), dict) else {}
            sess = session_manager.create_session(layout_ids=layout_ids, name=name, meta=meta)
            return jsonify({"success": True, "data": sess.to_dict()})
        except Exception as e:
            logger.error(f"Create session error: {e}")
            return jsonify({"success": False, "message": "Failed to create session"}), 500

    @sessions_bp.route("/<session_id>", methods=["GET"])
    def get_session(session_id: str):
        try:
            sess = session_manager.get_session(session_id)
            if not sess:
                return jsonify({"success": False, "message": "Session not found"}), 404
            return jsonify({"success": True, "data": sess.to_dict()})
        except Exception as e:
            logger.error(f"Get session error: {e}")
            return jsonify({"success": False, "message": "Failed to get session"}), 500

    @sessions_bp.route("/<session_id>", methods=["DELETE"])
    def delete_session(session_id: str):
        try:
            deleted = session_manager.delete_session(session_id)
            _emit_session_update(session_id)
            return jsonify({"success": True, "data": {"deleted": bool(deleted)}})
        except Exception as e:
            logger.error(f"Delete session error: {e}")
            return jsonify({"success": False, "message": "Failed to delete session"}), 500

    @sessions_bp.route("/<session_id>/start", methods=["POST"])
    def start_session(session_id: str):
        try:
            sess = session_manager.start_session(session_id)
            if not sess:
                return jsonify({"success": False, "message": "Session not found"}), 404
            _emit_session_update(session_id)
            return jsonify({"success": True, "data": sess.to_dict()})
        except Exception as e:
            logger.error(f"Start session error: {e}")
            return jsonify({"success": False, "message": "Failed to start session"}), 500

    @sessions_bp.route("/<session_id>/stop", methods=["POST"])
    def stop_session(session_id: str):
        try:
            sess = session_manager.stop_session(session_id)
            if not sess:
                return jsonify({"success": False, "message": "Session not found"}), 404
            _emit_session_update(session_id)
            return jsonify({"success": True, "data": sess.to_dict()})
        except Exception as e:
            logger.error(f"Stop session error: {e}")
            return jsonify({"success": False, "message": "Failed to stop session"}), 500

    @sessions_bp.route("/<session_id>/layouts", methods=["POST"])
    def attach_layouts(session_id: str):
        try:
            data = request.get_json() or {}
            layout_ids = data.get("layout_ids") or []
            action = str(data.get("action") or "attach").lower().strip()
            if not isinstance(layout_ids, list) or not layout_ids:
                return jsonify({"success": False, "message": "layout_ids (list) is required"}), 400
            if action == "detach":
                sess = session_manager.detach_layouts(session_id, layout_ids)
            else:
                sess = session_manager.attach_layouts(session_id, layout_ids)
            if not sess:
                return jsonify({"success": False, "message": "Session not found"}), 404
            _emit_session_update(session_id)
            return jsonify({"success": True, "data": sess.to_dict()})
        except Exception as e:
            logger.error(f"Update session layouts error: {e}")
            return jsonify({"success": False, "message": "Failed to update session layouts"}), 500

    app.register_blueprint(sessions_bp)
    logger.info("✅ Sessions routes registered")


