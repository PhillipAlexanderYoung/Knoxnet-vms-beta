from __future__ import annotations

import time
import threading
from dataclasses import dataclass
from typing import Dict, Optional, Tuple


@dataclass
class CooldownKey:
    rule_id: str
    camera_id: str

    def as_tuple(self) -> Tuple[str, str]:
        return (self.rule_id, self.camera_id)


class AutomationState:
    """
    In-memory state for automation execution.

    This is intentionally ephemeral; durable history is handled via event bundles / DB in
    the observability step.
    """

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._last_triggered: Dict[Tuple[str, str], float] = {}
        self._last_signature: Dict[Tuple[str, str], Tuple[str, float]] = {}

    def is_in_cooldown(self, *, rule_id: str, camera_id: str, cooldown_sec: float) -> bool:
        if cooldown_sec <= 0:
            return False
        key = (str(rule_id), str(camera_id))
        now = time.time()
        with self._lock:
            last = self._last_triggered.get(key)
        return bool(last and (now - last) < cooldown_sec)

    def mark_triggered(self, *, rule_id: str, camera_id: str) -> None:
        key = (str(rule_id), str(camera_id))
        now = time.time()
        with self._lock:
            self._last_triggered[key] = now

    def is_duplicate(self, *, rule_id: str, camera_id: str, signature: Optional[str], window_sec: float = 2.0) -> bool:
        """
        Best-effort dedupe: if the same signature fired very recently, skip.
        """
        if not signature:
            return False
        key = (str(rule_id), str(camera_id))
        now = time.time()
        with self._lock:
            prev = self._last_signature.get(key)
            if prev and prev[0] == signature and (now - prev[1]) < window_sec:
                return True
            self._last_signature[key] = (signature, now)
        return False



