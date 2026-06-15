"""Stage 6: desktop capture suppression when server Event Rules are armed."""

from __future__ import annotations

import unittest

from desktop.utils.event_rules_api import (
    filter_desktop_capture_events,
    should_suppress_desktop_track_capture,
)


def _zone_event():
    return {
        "shape_id": "zone_1",
        "shape_type": "zone",
        "interaction_type": "entered_zone",
    }


def _line_event():
    return {
        "shape_id": "line_1",
        "shape_type": "line",
        "interaction_type": "crossed_line",
    }


def _tag_event():
    return {
        "shape_id": "tag_1",
        "shape_type": "tag",
        "interaction_type": "near_tag",
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

    def test_suppresses_zone_and_line_when_server_active(self):
        events = [_zone_event(), _line_event()]
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

    def test_keeps_tag_events_when_server_active(self):
        events = [_tag_event()]
        filtered = filter_desktop_capture_events(
            events,
            motion_watch_active=True,
            server_event_rules_active=True,
        )
        self.assertEqual(filtered, events)
        self.assertFalse(
            should_suppress_desktop_track_capture(
                events,
                motion_watch_active=True,
                server_event_rules_active=True,
            )
        )

    def test_mixed_events_keep_tags_only(self):
        events = [_zone_event(), _tag_event()]
        filtered = filter_desktop_capture_events(
            events,
            motion_watch_active=True,
            server_event_rules_active=True,
        )
        self.assertEqual(filtered, [_tag_event()])
        self.assertFalse(
            should_suppress_desktop_track_capture(
                events,
                motion_watch_active=True,
                server_event_rules_active=True,
            )
        )


if __name__ == "__main__":
    unittest.main()
