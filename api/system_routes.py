from __future__ import annotations

import logging
from flask import Blueprint, jsonify

logger = logging.getLogger(__name__)


def init_system_routes(app):
    """
    Lightweight system resource telemetry for the host machine.
    Uses psutil (already in requirements.txt).
    """
    bp = Blueprint("system_bp", __name__, url_prefix="/api/system")

    @bp.route("/resources", methods=["GET"])
    def get_system_resources():
        try:
            import os
            import psutil  # type: ignore
            p = psutil.Process(os.getpid())

            # cpu_percent(None) is last interval; call once to prime could return 0 on first call.
            cpu_percent = psutil.cpu_percent(interval=0.1)
            vm = psutil.virtual_memory()
            sm = psutil.swap_memory()

            mem_info = p.memory_info()
            num_threads = getattr(p, "num_threads", lambda: None)()

            return jsonify(
                {
                    "success": True,
                    "data": {
                        "cpu": {
                            "percent": float(cpu_percent),
                            "count_logical": int(psutil.cpu_count(logical=True) or 0),
                        },
                        "memory": {
                            "percent": float(vm.percent),
                            "total_bytes": int(vm.total),
                            "available_bytes": int(vm.available),
                            "used_bytes": int(vm.used),
                        },
                        "swap": {
                            "percent": float(sm.percent),
                            "total_bytes": int(sm.total),
                            "used_bytes": int(sm.used),
                        },
                        "process": {
                            "rss_bytes": int(getattr(mem_info, "rss", 0)),
                            "vms_bytes": int(getattr(mem_info, "vms", 0)),
                            "threads": int(num_threads) if isinstance(num_threads, int) else None,
                        },
                    },
                }
            )
        except Exception as e:
            logger.error(f"/api/system/resources error: {e}")
            return jsonify({"success": False, "message": "Failed to get system resources"}), 500

    app.register_blueprint(bp)
    logger.info("✅ System routes registered (/api/system)")


