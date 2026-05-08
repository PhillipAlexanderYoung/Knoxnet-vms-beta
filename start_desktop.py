#!/usr/bin/env python3
"""
Knoxnet VMS Beta Desktop Entry Point
Launches the PySide6 Desktop Application.
"""
import sys
import os

try:
    from dotenv import load_dotenv
except Exception:
    load_dotenv = None  # type: ignore

# Load project .env for desktop update/public-key settings.
try:
    if load_dotenv:
        load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"), override=False)
except Exception:
    pass


def configure_rendering_env():
    """
    Force QtWebEngine/QtQuick to use software rendering when GPU/GL causes
    crashes (e.g., Unrecognized OpenGL version or GPU driver faults).
    Only sets values if the user has not already provided overrides.
    """
    env_defaults = {
        "LIBGL_ALWAYS_SOFTWARE": "1",
        "QT_QUICK_BACKEND": "software",
        "QTWEBENGINE_CHROMIUM_FLAGS": (
            "--disable-gpu --disable-gpu-compositing "
            "--disable-gpu-sandbox --disable-zero-copy "
            "--use-gl=angle --use-angle=swiftshader"
        ),
        "QTWEBENGINE_DISABLE_SANDBOX": "1",
        # Let Qt pick GLX/EGL; do not force-disable integration.
    }

    for key, value in env_defaults.items():
        os.environ.setdefault(key, value)

def _parse_iso_maybe_z(s: str):
    from datetime import datetime, timezone
    try:
        txt = (s or "").strip()
        if not txt:
            return None
        if txt.endswith("Z"):
            txt = txt[:-1] + "+00:00"
        dt = datetime.fromisoformat(txt)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except Exception:
        return None

def _maybe_rollback_failed_update():
    """
    Best-effort rollback on next launch if the last applied update never reached boot_ok.

    Strategy:
    - When applying an update we write update_pending.json with the expected version.
    - The desktop app writes boot_ok.json once it reaches "ready" state.
    - On future startups, if update_pending exists and is older than a window
      and boot_ok is still not the expected version, rollback to old/.
    """
    try:
        from datetime import datetime, timezone
        from core.updater import Updater, read_update_pending, read_boot_ok_version, clear_update_pending
    except Exception:
        return

    pending = read_update_pending()
    if not pending:
        return

    expected = str(pending.get("expected_version") or pending.get("version") or "").strip()
    prev_ok = str(pending.get("previous_boot_ok_version") or "").strip()
    boot_ver = (read_boot_ok_version() or "").strip()

    # Success path: new version reached boot_ok.
    if expected and boot_ver and boot_ver == expected:
        clear_update_pending()
        return

    # If we didn't know the expected version, treat "boot_ok still equals previous"
    # as a signal that the new version never fully booted.
    if not expected and prev_ok and boot_ver and boot_ver != prev_ok:
        clear_update_pending()
        return

    # Don't rollback immediately; allow time for the new version to boot and write boot_ok.
    window_s = int(os.environ.get("KNOXNET_UPDATE_ROLLBACK_WINDOW_SEC", "900") or "900")  # 15 min
    window_s = max(60, min(24 * 60 * 60, window_s))

    ts = _parse_iso_maybe_z(str(pending.get("timestamp") or ""))
    if not ts:
        return
    age_s = (datetime.now(tz=timezone.utc) - ts).total_seconds()
    if age_s < float(window_s):
        return

    try:
        u = Updater(manifest_url="", signature_url="", public_keys={}, channel="")
        u.rollback()
        clear_update_pending()
    except Exception:
        return

def main():
    # Support running individual packaged services from the same frozen executable.
    # This is used by `services/*/start_service.sh`.
    try:
        args = list(sys.argv[1:])
    except Exception:
        args = []

    if args:
        if "--run-vision-local" in args:
            from services.vision_local.__main__ import main as _vision_main
            _vision_main()
            return
        if "--run-llm-local" in args:
            from services.llm_local.__main__ import main as _llm_main
            _llm_main()
            return
        if "--run-backend" in args:
            # Execute `app.py` under __main__ so its existing startup block runs.
            import runpy
            runpy.run_module("app", run_name="__main__")
            return

    # Ensure we can import from the desktop package
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))

    # Configure safe software rendering before Qt libraries load
    configure_rendering_env()
    from desktop.app import main as app_main
    app_main()

if __name__ == "__main__":
    main()
