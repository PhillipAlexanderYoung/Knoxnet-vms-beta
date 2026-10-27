"""Tests for the shared rule-trigger side-effect coupler.

These lock in the invariant that *any* path which increments a rule counter also
dispatches the rule's screenshot side-effect (or surfaces a terminal failure),
de-duplicated between the server ``automation_alert`` path and the local motion
fallback path.
"""

from __future__ import annotations

import unittest

from desktop.utils.rule_trigger_side_effects import RuleTriggerSideEffects
from desktop.utils.shape_trigger_helpers import (
    build_fresh_shape_rule_conditions,
    rule_has_snapshot_action,
)


SNAPSHOT_RULE = {
    "id": "rule_snap",
    "name": "Driveway",
    "shape_id": "zone_1",
    "actions": [{"type": "snapshot", "save_dir": "captures/x"}],
}
NO_SNAPSHOT_RULE = {
    "id": "rule_nosnap",
    "name": "Quiet",
    "shape_id": "zone_2",
    "actions": [{"type": "script", "path": "x.py"}],
}


class _Recorder:
    """Collects callback invocations for assertions."""

    def __init__(self, *, increment_returns=True):
        self.increment_returns = increment_returns
        self.increment_calls = []
        self.local_screenshot_calls = []
        self.terminal_capture_calls = []
        self.terminal_failure_calls = []
        self.local_screenshot_ok = True
        self.now = 1000.0

    def increment(self, shape_id, rule_id):
        self.increment_calls.append((shape_id, rule_id))
        return self.increment_returns

    def dispatch_local(self, rule_id, shape_id, event_ctx):
        self.local_screenshot_calls.append((rule_id, shape_id, event_ctx))
        return self.local_screenshot_ok

    def notify_capture(self, capture, rule_id):
        self.terminal_capture_calls.append((capture, rule_id))

    def notify_failure(self, error, rule_name):
        self.terminal_failure_calls.append((error, rule_name))

    def time_fn(self):
        return self.now

    def build(self, rules=None, **kwargs):
        rules = rules or {SNAPSHOT_RULE["id"]: SNAPSHOT_RULE, NO_SNAPSHOT_RULE["id"]: NO_SNAPSHOT_RULE}
        return RuleTriggerSideEffects(
            increment_counter=self.increment,
            dispatch_local_screenshot=self.dispatch_local,
            notify_terminal_capture=self.notify_capture,
            notify_terminal_failure=self.notify_failure,
            rule_lookup=lambda rid: rules.get(rid),
            time_fn=self.time_fn,
            **kwargs,
        )


class RuleHasSnapshotActionTests(unittest.TestCase):
    def test_true_for_snapshot_action(self):
        self.assertTrue(rule_has_snapshot_action(SNAPSHOT_RULE))

    def test_false_without_snapshot(self):
        self.assertFalse(rule_has_snapshot_action(NO_SNAPSHOT_RULE))
        self.assertFalse(rule_has_snapshot_action({"actions": []}))
        self.assertFalse(rule_has_snapshot_action({}))

    def test_false_when_snapshot_disabled(self):
        self.assertFalse(
            rule_has_snapshot_action({"actions": [{"type": "snapshot", "enabled": False}]})
        )

    def test_fresh_rule_payload_has_snapshot_on_by_default(self):
        built = build_fresh_shape_rule_conditions(
            motion_path=[{"x": 0.1, "y": 0.5}, {"x": 0.9, "y": 0.5}],
        )
        self.assertTrue(rule_has_snapshot_action(built))


class LocalTriggerTests(unittest.TestCase):
    def test_local_counter_with_snapshot_dispatches_screenshot(self):
        rec = _Recorder()
        dispatcher = rec.build()
        out = dispatcher.handle_local_trigger("rule_snap", "zone_1", {"shape_id": "zone_1"})
        self.assertTrue(out.counter_incremented)
        self.assertTrue(out.screenshot_dispatched)
        self.assertEqual(len(rec.increment_calls), 1)
        self.assertEqual(len(rec.local_screenshot_calls), 1)
        self.assertEqual(rec.local_screenshot_calls[0][0], "rule_snap")

    def test_local_counter_without_snapshot_action_skips_screenshot(self):
        rec = _Recorder()
        dispatcher = rec.build()
        out = dispatcher.handle_local_trigger("rule_nosnap", "zone_2")
        self.assertTrue(out.counter_incremented)
        self.assertFalse(out.screenshot_dispatched)
        self.assertEqual(out.reason, "no_snapshot_action")
        self.assertEqual(len(rec.local_screenshot_calls), 0)

    def test_counter_suppressed_means_no_screenshot(self):
        # Coupling guarantee: if the counter did not advance (cooldown), the
        # screenshot must not fire either.
        rec = _Recorder(increment_returns=False)
        dispatcher = rec.build()
        out = dispatcher.handle_local_trigger("rule_snap", "zone_1")
        self.assertFalse(out.counter_incremented)
        self.assertFalse(out.screenshot_dispatched)
        self.assertEqual(len(rec.local_screenshot_calls), 0)

    def test_local_capture_failure_notifies_terminal(self):
        rec = _Recorder()
        rec.local_screenshot_ok = False
        dispatcher = rec.build()
        out = dispatcher.handle_local_trigger("rule_snap", "zone_1")
        self.assertTrue(out.counter_incremented)
        self.assertFalse(out.screenshot_dispatched)
        self.assertTrue(out.failure_notified)
        self.assertEqual(len(rec.terminal_failure_calls), 1)
        error, rule_name = rec.terminal_failure_calls[0]
        self.assertTrue(error.get("capture_failed"))
        self.assertEqual(rule_name, "Driveway")


class ServerAlertTests(unittest.TestCase):
    def test_server_capture_defers_terminal_to_new_capture(self):
        rec = _Recorder()
        dispatcher = rec.build()
        out = dispatcher.handle_server_alert(
            rule_id="rule_snap",
            shape_id="zone_1",
            capture={"file_path": "/tmp/a.jpg"},
        )
        self.assertTrue(out.counter_incremented)
        self.assertTrue(out.screenshot_dispatched)
        self.assertEqual(out.reason, "server_capture_deferred_to_new_capture")
        self.assertEqual(len(rec.terminal_capture_calls), 0)

    def test_server_capture_error_shows_warning(self):
        rec = _Recorder()
        dispatcher = rec.build()
        out = dispatcher.handle_server_alert(
            rule_id="rule_snap",
            shape_id="zone_1",
            capture_error={"capture_failed": True, "reason": "snapshot_not_executed"},
            rule_name="Driveway",
        )
        self.assertTrue(out.failure_notified)
        self.assertEqual(len(rec.terminal_failure_calls), 1)
        self.assertEqual(rec.terminal_failure_calls[0][0]["reason"], "snapshot_not_executed")

    def test_server_alert_without_payload_only_counts(self):
        rec = _Recorder()
        dispatcher = rec.build()
        out = dispatcher.handle_server_alert(rule_id="rule_snap", shape_id="zone_1")
        self.assertTrue(out.counter_incremented)
        self.assertFalse(out.screenshot_dispatched)
        self.assertFalse(out.failure_notified)
        self.assertEqual(len(rec.terminal_capture_calls), 0)


class DedupeTests(unittest.TestCase):
    def test_local_skips_when_server_capture_just_shown(self):
        rec = _Recorder()
        dispatcher = rec.build(capture_dedupe_sec=3.0)
        dispatcher.handle_server_alert(
            rule_id="rule_snap",
            shape_id="zone_1",
            capture={"file_path": "/tmp/a.jpg"},
        )
        self.assertEqual(len(rec.terminal_capture_calls), 0)
        # A local trigger 1s later for the same rule must not double-capture.
        rec.now += 1.0
        out = dispatcher.handle_local_trigger("rule_snap", "zone_1")
        self.assertTrue(out.counter_incremented)
        self.assertFalse(out.screenshot_dispatched)
        self.assertEqual(out.reason, "dedupe_recent_capture")
        self.assertEqual(len(rec.local_screenshot_calls), 0)

    def test_local_captures_after_dedupe_window(self):
        rec = _Recorder()
        dispatcher = rec.build(capture_dedupe_sec=3.0)
        dispatcher.handle_server_alert(
            rule_id="rule_snap",
            shape_id="zone_1",
            capture={"file_path": "/tmp/a.jpg"},
        )
        rec.now += 5.0
        out = dispatcher.handle_local_trigger("rule_snap", "zone_1")
        self.assertTrue(out.screenshot_dispatched)
        self.assertEqual(len(rec.local_screenshot_calls), 1)


if __name__ == "__main__":
    unittest.main()
