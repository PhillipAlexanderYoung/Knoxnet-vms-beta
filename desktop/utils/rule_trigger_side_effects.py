"""Shared coupling point between rule-counter increments and screenshot side-effects.

This module exists to enforce a single invariant that previous bandaid patches
kept violating:

    Any code path that increments a rule counter must also dispatch the rule's
    configured side-effects (screenshot capture + terminal notification), or at
    minimum surface a terminal failure when capture is impossible.

The desktop ``CameraWidget`` increments rule counters from two places:

* the *server* path (``automation_alert`` socket event), where the snapshot is
  produced server-side; successful captures are shown in the terminal via the
  separate ``new_capture`` socket event (not duplicated here); and
* the *local* fallback path (``shape_triggered`` from the GL widget), where
  motion-mode rules are counted on the desktop and the server never runs
  a ``SnapshotAction`` for those triggers.

The local path historically incremented the counter but never captured a
screenshot, so users saw pill counters move with no terminal screenshot and no
error.  ``RuleTriggerSideEffects`` is a Qt-free object (so it is unit testable)
that both paths funnel through, guaranteeing the counter and the screenshot stay
coupled and de-duplicated against each other.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional

from desktop.utils.shape_trigger_helpers import rule_has_snapshot_action


@dataclass
class TriggerOutcome:
    """Result of routing a single rule trigger through the shared dispatcher."""

    rule_id: str
    shape_id: str
    counter_incremented: bool = False
    screenshot_dispatched: bool = False
    failure_notified: bool = False
    reason: str = ""


class RuleTriggerSideEffects:
    """Couple rule-counter increments with screenshot/terminal side-effects.

    Callables are injected so this object has no Qt dependency:

    * ``increment_counter(shape_id, rule_id) -> bool`` returns ``True`` when the
      counter actually advanced (i.e. it was not suppressed by cooldown/dedupe).
    * ``dispatch_local_screenshot(rule_id, shape_id, event_ctx) -> bool`` takes a
      local screenshot and shows it in the terminal, returning ``True`` on
      success.
    * ``notify_terminal_capture(capture, rule_id)`` is available for injection but
      server success captures are shown via ``new_capture`` (see
      ``handle_server_alert``).
    * ``notify_terminal_failure(error, rule_name)`` shows a capture failure in the
      terminal.
    * ``rule_lookup(rule_id) -> Optional[dict]`` resolves the full rule (with its
      ``actions``) so we can tell whether a screenshot is configured.
    """

    def __init__(
        self,
        *,
        increment_counter: Callable[[str, Optional[str]], bool],
        dispatch_local_screenshot: Callable[[str, str, Optional[dict]], bool],
        notify_terminal_capture: Callable[[dict, str], None],
        notify_terminal_failure: Callable[[dict, str], None],
        rule_lookup: Callable[[str], Optional[Dict[str, Any]]],
        capture_dedupe_sec: float = 3.0,
        time_fn: Callable[[], float] = time.time,
    ) -> None:
        self._increment_counter = increment_counter
        self._dispatch_local_screenshot = dispatch_local_screenshot
        self._notify_terminal_capture = notify_terminal_capture
        self._notify_terminal_failure = notify_terminal_failure
        self._rule_lookup = rule_lookup
        self._capture_dedupe_sec = max(0.0, float(capture_dedupe_sec))
        self._time_fn = time_fn
        # rule_id -> monotonic-ish timestamp of the last shown capture (server OR
        # local).  Shared so a server alert and a local fallback for the same
        # rule never produce two screenshots back-to-back.
        self._capture_last_shown: Dict[str, float] = {}

    def _recently_captured(self, rule_id: str, now: float) -> bool:
        if self._capture_dedupe_sec <= 0:
            return False
        last = float(self._capture_last_shown.get(rule_id, 0.0) or 0.0)
        return (now - last) < self._capture_dedupe_sec

    def handle_local_trigger(
        self,
        rule_id: str,
        shape_id: str,
        event_ctx: Optional[dict] = None,
    ) -> TriggerOutcome:
        """Local (desktop motion/tag) counter increment + coupled screenshot."""
        rid = str(rule_id or "").strip()
        sid = str(shape_id or "").strip()
        out = TriggerOutcome(rule_id=rid, shape_id=sid)

        incremented = bool(self._increment_counter(sid, rid))
        out.counter_incremented = incremented
        if not incremented:
            # Counter was suppressed (cooldown/dedupe); keep screenshots coupled
            # by *not* capturing — this is what de-dupes us against the server
            # alert that may have just advanced the same counter.
            out.reason = "counter_suppressed"
            return out

        rule = self._rule_lookup(rid)
        if rule is None or not rule_has_snapshot_action(rule):
            out.reason = "no_snapshot_action"
            return out

        now = self._time_fn()
        if self._recently_captured(rid, now):
            out.reason = "dedupe_recent_capture"
            return out

        self._capture_last_shown[rid] = now
        ok = bool(self._dispatch_local_screenshot(rid, sid, event_ctx))
        out.screenshot_dispatched = ok
        if ok:
            out.reason = "local_capture_dispatched"
            return out

        # Capture impossible (no frame, stream offline, etc.) — the user must
        # see *something* on the same trigger that moved the counter.
        self._notify_terminal_failure(
            {
                "capture_failed": True,
                "reason": "local_capture_unavailable",
                "rule_id": rid,
            },
            str((rule or {}).get("name") or ""),
        )
        out.failure_notified = True
        out.reason = "local_capture_failed"
        return out

    def handle_server_alert(
        self,
        *,
        rule_id: str,
        shape_id: str,
        capture: Optional[dict] = None,
        capture_error: Optional[dict] = None,
        rule_name: str = "",
    ) -> TriggerOutcome:
        """Server ``automation_alert`` counter increment + capture/error display."""
        rid = str(rule_id or "").strip()
        sid = str(shape_id or "").strip()
        out = TriggerOutcome(rule_id=rid, shape_id=sid)

        if sid:
            out.counter_incremented = bool(self._increment_counter(sid, rid or None))

        now = self._time_fn()
        if isinstance(capture, dict) and str(
            capture.get("file_path") or capture.get("file_uri") or capture.get("thumb_uri") or ""
        ).strip():
            # Terminal success is handled by ``new_capture`` (SnapshotAction emits
            # after ingest). Record dedupe state here so local fallback does not
            # double-capture; do not notify terminal again from this path.
            self._capture_last_shown[rid] = now
            out.screenshot_dispatched = True
            out.reason = "server_capture_deferred_to_new_capture"
            return out

        if isinstance(capture_error, dict) and capture_error.get("capture_failed"):
            self._notify_terminal_failure(capture_error, rule_name)
            out.failure_notified = True
            out.reason = "server_capture_error"
            return out

        out.reason = "no_capture_payload"
        return out
