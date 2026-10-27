from __future__ import annotations

import os
import sys
import uvicorn

try:
    from .config import get_settings
    # Production app (auto-downloads lighter HF models into models/vision/cache)
    from .app_production import app as production_app
except ModuleNotFoundError as e:
    # Most common missing dep in fresh installs is torch.
    missing = str(getattr(e, "name", "") or "")
    if missing:
        print(f"[Local Vision] Missing dependency: {missing}", file=sys.stderr)
    else:
        print(f"[Local Vision] Import error: {e}", file=sys.stderr)
    print("[Local Vision] Install required deps (see requirements.txt / services/vision_local/requirements*.txt).", file=sys.stderr)
    raise


def main() -> None:
    settings = get_settings()
    # Default to the production app which works out-of-the-box (auto-downloads smaller models).
    # The advanced app requires pre-downloaded model directories (blip2/llava/minicpm), which are huge.
    mode = os.getenv("VISION_LOCAL_MODE", "production").strip().lower()
    if mode in {"advanced", "multi", "ensemble"}:
        # Import on-demand to avoid heavy deps unless explicitly requested.
        try:
            from .app import create_app
        except ImportError:
            from app import create_app
        app = create_app(settings)
    else:
        app = production_app
    uvicorn.run(app, host=settings.host, port=settings.port)


if __name__ == "__main__":
    main()

