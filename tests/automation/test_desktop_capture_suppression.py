"""Stage 6: desktop capture suppression when server Event Rules are armed."""

from __future__ import annotations

import unittest

from desktop.utils.event_rules_api import (
    filter_desktop_capture_events,
    filter_desktop_counter_events,
    should_suppress_desktop_track_capture,
)


def _zone_event(**extra):
    return {
        "shape_id": "zone_1",
        "shape_type": "zone",
        "interaction_type": "entered_zone",
        **extra,
    }


def _line_event(**extra):
    return {
        "shape_id": "line_1",
        "shape_type": "line",
        "interaction_type": "crossed_line",
        **extra,
    }


def _tag_event(**extra):
    return {
        "shape_id": "tag_1",
        "shape_type": "tag",
        "interaction_type": "near_tag",
        **extra,
    }


class DesktopCaptureSuppressionTests(unittest.TestCase):
    def test_passes_all_events_when_not_armed(self):
        events = [_zone_event(), _line_event()]
        self.assertEqual(
            filter_desktop_capture_events(
                events,
                motion_watch_active=False,
                server_event_rules_active=True,
            ),
            events,
        )
        self.assertFalse(
            should_suppress_desktop_track_capture(
                events,
                motion_watch_active=False,
                server_event_rules_active=True,
            )
        )

    def test_passes_all_events_when_server_rules_inactive(self):
        events = [_zone_event()]
        self.assertEqual(
            filter_desktop_capture_events(
                events,
                motion_watch_active=True,
                server_event_rules_active=False,
            ),
            events,
        )

    def test_suppresses_zone_line_and_tag_legacy_capture_when_server_active(self):
        events = [_zone_event(), _line_event(), _tag_event()]
        self.assertEqual(
            filter_desktop_capture_events(
                events,
                motion_watch_active=True,
                server_event_rules_active=True,
            ),
            [],
        )
        self.assertTrue(
            should_suppress_desktop_track_capture(
                events,
                motion_watch_active=True,
                server_event_rules_active=True,
            )
        )

    def test_mixed_events_suppress_all_legacy_capture_when_server_active(self):
        events = [_zone_event(), _tag_event()]
        filtered = filter_desktop_capture_events(
            events,
            motion_watch_active=True,
            server_event_rules_active=True,
        )
        self.assertEqual(filtered, [])
        self.assertTrue(
            should_suppress_desktop_track_capture(
                events,
                motion_watch_active=True,
                server_event_rules_active=True,
            )
        )


class DesktopCounterSuppressionTests(unittest.TestCase):
    def test_detection_line_excluded_when_server_active(self):
        events = [_line_event(source="backend")]
        filtered = filter_desktop_counter_events(
            events,
            motion_watch_active=True,
            server_event_rules_active=True,
            trigger_source="backend",
            uses_local_motion_counter=lambda _sid: True,
        )
        self.assertEqual(filtered, [])

    def test_detection_zone_excluded_when_server_active(self):
        events = [_zone_event(source="detection")]
        filtered = filter_desktop_counter_events(
            events,
            motion_watch_active=True,
            server_event_rules_active=True,
            trigger_source="detection",
            uses_local_motion_counter=lambda _sid: True,
        )
        self.assertEqual(filtered, [])

    def test_motion_line_kept_when_shape_uses_local_motion_counter(self):
        events = [_line_event()]
        filtered = filter_desktop_counter_events(
            events,
            motion_watch_active=True,
            server_event_rules_active=True,
            trigger_source=None,
            uses_local_motion_counter=lambda sid: sid == "line_1",
        )
        self.assertEqual(filtered, events)

    def test_motion_line_excluded_without_local_motion_counter(self):
        events = [_line_event()]
        filtered = filter_desktop_counter_events(
            events,
            motion_watch_active=True,
            server_event_rules_active=True,
            trigger_source=None,
            uses_local_motion_counter=lambda _sid: False,
        )
        self.assertEqual(filtered, [])

    def test_tag_events_always_kept_for_local_counter_path(self):
        events = [_tag_event(source="backend")]
        filtered = filter_desktop_counter_events(
            events,
            motion_watch_active=True,
            server_event_rules_active=True,
            trigger_source="backend",
            uses_local_motion_counter=lambda _sid: False,
        )
        self.assertEqual(filtered, events)

    def test_passes_all_events_when_not_armed(self):
        events = [_line_event(source="backend")]
        self.assertEqual(
            filter_desktop_counter_events(
                events,
                motion_watch_active=False,
                server_event_rules_active=True,
                trigger_source="backend",
                uses_local_motion_counter=lambda _sid: False,
            ),
            events,
        )


if __name__ == "__main__":
    unittest.main()
