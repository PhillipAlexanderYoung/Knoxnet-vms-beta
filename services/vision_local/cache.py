from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any, Dict, Optional


class ResultCache:
    """Disk-backed cache keyed by SHA256 + model + action."""

    def __init__(self, base_dir: Path):
        self.base_dir = base_dir
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def _path_for(self, model: str, digest: str, action: str) -> Path:
        return self.base_dir / model / action / f"{digest}.json"

    def get(self, model: str, digest: str, action: str) -> Optional[Dict[str, Any]]:
        path = self._path_for(model, digest, action)
        if not path.exists():
            return None
        with self._lock:
            try:
                with path.open("r", encoding="utf-8") as fh:
                    return json.load(fh)
            except json.JSONDecodeError:
                path.unlink(missing_ok=True)
                return None

    def set(self, model: str, digest: str, action: str, payload: Dict[str, Any]) -> None:
        path = self._path_for(model, digest, action)
        path.parent.mkdir(parents=True, exist_ok=True)
        with self._lock:
            with path.open("w", encoding="utf-8") as fh:
                json.dump(payload, fh)
                fh.write("\n")

