from __future__ import annotations

from flask import Blueprint, jsonify, request
import logging
from typing import Any, Dict, List

from core.layout_models import CameraProfile, LayoutDefinition
from core.layout_store import LayoutsAndProfilesStore

logger = logging.getLogger(__name__)


def init_layouts_profiles_routes(app, store: LayoutsAndProfilesStore):
    """
    Register CRUD routes for layouts + camera profiles + assignments.
    Local-file JSON is the source-of-truth.
    """

    layouts_bp = Blueprint("layouts_bp", __name__, url_prefix="/api/layouts")
    profiles_bp = Blueprint("camera_profiles_bp", __name__, url_prefix="/api/camera-profiles")

    @layouts_bp.route("", methods=["GET"])
    def list_layouts():
        try:
            layouts = [l.to_dict() for l in store.list_layouts()]
            return jsonify({"success": True, "data": {"layouts": layouts, "count": len(layouts)}})
        except Exception as e:
            logger.error(f"List layouts error: {e}")
            return jsonify({"success": False, "message": "Failed to list layouts"}), 500

    @layouts_bp.route("/<layout_id>", methods=["GET"])
    def get_layout(layout_id: str):
        try:
            layout = store.get_layout(layout_id)
            if not layout:
                return jsonify({"success": False, "message": "Layout not found"}), 404
            return jsonify({"success": True, "data": layout.to_dict()})
        except Exception as e:
            logger.error(f"Get layout error: {e}")
            return jsonify({"success": False, "message": "Failed to get layout"}), 500

    @layouts_bp.route("", methods=["PUT"])
    def upsert_layout():
        try:
            data = request.get_json() or {}
            layout = LayoutDefinition.from_dict(data)
            if not layout.id:
                return jsonify({"success": False, "message": "layout.id is required"}), 400
            if not layout.name:
                layout.name = layout.id
            saved = store.upsert_layout(layout)
            return jsonify({"success": True, "data": saved.to_dict()})
        except Exception as e:
            logger.error(f"Upsert layout error: {e}")
            return jsonify({"success": False, "message": "Failed to upsert layout"}), 500

    @layouts_bp.route("/<layout_id>", methods=["DELETE"])
    def delete_layout(layout_id: str):
        try:
            existed = store.delete_layout(layout_id)
            return jsonify({"success": True, "data": {"deleted": bool(existed)}})
        except Exception as e:
            logger.error(f"Delete layout error: {e}")
            return jsonify({"success": False, "message": "Failed to delete layout"}), 500

    @layouts_bp.route("/migrate-legacy", methods=["POST"])
    def migrate_legacy_layouts():
        """
        Convert existing data/desktop_layouts.json into layouts/profiles/assignments.
        Safe to run multiple times (deterministic profile ids).
        """
        try:
            payload = request.get_json() or {}
            res = store.migrate_from_legacy_desktop_layouts(
                overwrite_layouts=bool(payload.get("overwrite_layouts", False)),
                create_profiles=bool(payload.get("create_profiles", True)),
                overwrite_assignments=bool(payload.get("overwrite_assignments", False)),
            )
            return jsonify({"success": True, "data": res})
        except Exception as e:
            logger.error(f"Legacy migration error: {e}")
            return jsonify({"success": False, "message": "Failed to migrate legacy layouts"}), 500

    # ---- camera profiles ----
    @profiles_bp.route("", methods=["GET"])
    def list_profiles():
        try:
            profiles = [p.to_dict() for p in store.list_profiles()]
            return jsonify({"success": True, "data": {"profiles": profiles, "count": len(profiles)}})
        except Exception as e:
            logger.error(f"List profiles error: {e}")
            return jsonify({"success": False, "message": "Failed to list profiles"}), 500

    @profiles_bp.route("/<profile_id>", methods=["GET"])
    def get_profile(profile_id: str):
        try:
            prof = store.get_profile(profile_id)
            if not prof:
                return jsonify({"success": False, "message": "Profile not found"}), 404
            return jsonify({"success": True, "data": prof.to_dict()})
        except Exception as e:
            logger.error(f"Get profile error: {e}")
            return jsonify({"success": False, "message": "Failed to get profile"}), 500

    @profiles_bp.route("", methods=["PUT"])
    def upsert_profile():
        try:
            data = request.get_json() or {}
            prof = CameraProfile.from_dict(data)
            if not prof.id:
                return jsonify({"success": False, "message": "profile.id is required"}), 400
            if not prof.name:
                prof.name = prof.id
            saved = store.upsert_profile(prof)
            return jsonify({"success": True, "data": saved.to_dict()})
        except Exception as e:
            logger.error(f"Upsert profile error: {e}")
            return jsonify({"success": False, "message": "Failed to upsert profile"}), 500

    @profiles_bp.route("", methods=["POST"])
    def create_profile():
        try:
            data = request.get_json() or {}
            name = str(data.get("name") or "").strip()
            if not name:
                return jsonify({"success": False, "message": "name is required"}), 400
            prof = store.create_profile(
                name=name,
                overlays=(data.get("overlays") if isinstance(data.get("overlays"), dict) else {}),
                ai_pipeline=(data.get("ai_pipeline") if isinstance(data.get("ai_pipeline"), dict) else {}),
                monitoring_tools=(data.get("monitoring_tools") if isinstance(data.get("monitoring_tools"), dict) else {}),
                meta=(data.get("meta") if isinstance(data.get("meta"), dict) else {}),
            )
            return jsonify({"success": True, "data": prof.to_dict()})
        except Exception as e:
            logger.error(f"Create profile error: {e}")
            return jsonify({"success": False, "message": "Failed to create profile"}), 500

    @profiles_bp.route("/<profile_id>", methods=["DELETE"])
    def delete_profile(profile_id: str):
        try:
            existed = store.delete_profile(profile_id, remove_assignments=True)
            return jsonify({"success": True, "data": {"deleted": bool(existed)}})
        except Exception as e:
            logger.error(f"Delete profile error: {e}")
            return jsonify({"success": False, "message": "Failed to delete profile"}), 500

    @profiles_bp.route("/apply", methods=["POST"])
    def apply_profile_bulk():
        try:
            data = request.get_json() or {}
            profile_id = str(data.get("profile_id") or "").strip()
            camera_ids = data.get("camera_ids") or []
            mode = str(data.get("mode") or "replace").strip().lower()
            if not profile_id:
                return jsonify({"success": False, "message": "profile_id is required"}), 400
            if not isinstance(camera_ids, list):
                return jsonify({"success": False, "message": "camera_ids must be a list"}), 400
            if mode not in ("replace", "append"):
                mode = "replace"
            res = store.bulk_apply_profile(profile_id, camera_ids, mode=mode)
            return jsonify({"success": True, "data": res})
        except Exception as e:
            logger.error(f"Bulk apply profile error: {e}")
            return jsonify({"success": False, "message": "Failed to apply profile"}), 500

    @profiles_bp.route("/assignments", methods=["GET"])
    def get_assignments():
        try:
            assigns = store.get_assignments()
            return jsonify({"success": True, "data": {"assignments": assigns}})
        except Exception as e:
            logger.error(f"Get assignments error: {e}")
            return jsonify({"success": False, "message": "Failed to get assignments"}), 500

    @profiles_bp.route("/assignments", methods=["PUT"])
    def set_assignment():
        try:
            data = request.get_json() or {}
            camera_id = str(data.get("camera_id") or "").strip()
            value = data.get("profile_id") if "profile_id" in data else data.get("profile_ids")
            if not camera_id:
                return jsonify({"success": False, "message": "camera_id is required"}), 400
            if isinstance(value, str):
                store.set_assignment(camera_id, value)
            elif isinstance(value, list):
                store.set_assignment(camera_id, [str(x) for x in value if str(x).strip()])
            else:
                return jsonify({"success": False, "message": "profile_id (string) or profile_ids (list) required"}), 400
            return jsonify({"success": True})
        except Exception as e:
            logger.error(f"Set assignment error: {e}")
            return jsonify({"success": False, "message": "Failed to set assignment"}), 500

    app.register_blueprint(layouts_bp)
    app.register_blueprint(profiles_bp)
    logger.info("✅ Layouts + CameraProfiles routes registered")


