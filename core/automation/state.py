from __future__ import annotations

import time
import threading
from typing import Dict, Optional, Tuple


class AutomationState:
    """
    In-memory state for automation execution.

    Cooldowns are scoped per (rule_id, camera_id) by default, and optionally
    per (rule_id, camera_id, track_id) for track_event rules.
    """

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._last_triggered: Dict[Tuple[str, str], float] = {}
        self._last_triggered_track: Dict[Tuple[str, str, str], float] = {}
        self._last_signature: Dict[Tuple[str, str], Tuple[str, float]] = {}

    def is_in_cooldown(self, *, rule_id: str, camera_id: str, cooldown_sec: float) -> bool:
        if cooldown_sec <= 0:
            return False
        key = (str(rule_id), str(camera_id))
        now = time.time()
        with self._lock:
            last = self._last_triggered.get(key)
        return bool(last and (now - last) < cooldown_sec)

    def is_in_track_cooldown(
        self,
        *,
        rule_id: str,
        camera_id: str,
        track_id: int,
        cooldown_sec: float,
    ) -> bool:
        if cooldown_sec <= 0:
            return False
        key = (str(rule_id), str(camera_id), str(int(track_id)))
        now = time.time()
        with self._lock:
            last = self._last_triggered_track.get(key)
        return bool(last and (now - last) < cooldown_sec)

    def mark_triggered(self, *, rule_id: str, camera_id: str, track_id: Optional[int] = None) -> None:
        now = time.time()
        with self._lock:
            self._last_triggered[(str(rule_id), str(camera_id))] = now
            if track_id is not None:
                self._last_triggered_track[(str(rule_id), str(camera_id), str(int(track_id)))] = now

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
