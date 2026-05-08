"""Public beta lifecycle service.

Runs one delayed update check on startup and emits a UI signal only when the
public beta endpoint reports a newer version. There is no entitlement refresh,
paid activation, staging, download, or auto-update behavior in this repo.
"""
from __future__ import annotations

import logging
import threading
from typing import Optional

from PySide6.QtCore import QObject, QTimer, Signal, Slot

logger = logging.getLogger("LifecycleService")

_STARTUP_DELAY_MS = 8_000


class LifecycleService(QObject):
    update_available = Signal(str, str, str)  # current_version, latest_version, channel
    entitlement_changed = Signal()

    def __init__(self, parent: Optional[QObject] = None):
        super().__init__(parent)
        self._started = False
        self._update_emitted = False

    def start(self):
        if self._started:
            return
        self._started = True
        QTimer.singleShot(_STARTUP_DELAY_MS, self._on_startup_tick)
        logger.info("LifecycleService started (startup beta update check only)")

    def stop(self):
        self._started = False

    def force_update_check(self):
        self._do_update_check()

    def force_entitlement_refresh(self):
        self.entitlement_changed.emit()

    @Slot()
    def _on_startup_tick(self):
        self._do_update_check()

    @Slot()
    def _do_update_check(self):
        threading.Thread(target=self._bg_update_check, daemon=True).start()

    def _bg_update_check(self):
        try:
            from core.version import get_version
            from core.update_check import check_for_update

            current = get_version()
            info = check_for_update(current)
            if info and not self._update_emitted:
                self._update_emitted = True
                logger.info("Beta update available: %s -> %s", current, info.latest_version)
                self.update_available.emit(current, info.latest_version, info.channel)
            elif info:
                logger.info("Beta update available but notification already shown")
            else:
                logger.info("Knoxnet VMS Beta is up to date")
        except Exception as exc:
            logger.debug("Beta update check skipped/failed: %s", exc)
