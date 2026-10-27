"""
Entry point for running the Local LLM Service.
"""

from __future__ import annotations

import sys

import uvicorn


def main() -> None:
    try:
        from .config import get_settings
    except ModuleNotFoundError as e:
        missing = str(getattr(e, "name", "") or "")
        if missing:
            print(f"[Local LLM] Missing dependency: {missing}", file=sys.stderr)
        else:
            print(f"[Local LLM] Import error: {e}", file=sys.stderr)
        print(
            "[Local LLM] Install required deps (see requirements.txt / services/llm_local/requirements*.txt).",
            file=sys.stderr,
        )
        raise

    settings = get_settings()

    uvicorn.run(
        "services.llm_local.app:app",
        host=settings.host,
        port=settings.port,
        log_level=str(getattr(settings, "log_level", "info")).lower(),
        reload=False,
    )


if __name__ == "__main__":
    main()

