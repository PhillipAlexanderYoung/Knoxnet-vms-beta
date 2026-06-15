"""Tests for capture notification helpers."""

from __future__ import annotations

import unittest
from unittest.mock import MagicMock

from core.capture_events import (
    build_motion_watch_terminal_payload,
    build_new_capture_payload,
    build_terminal_capture_failure_payload,
    capture_failure_reason_label,
    emit_new_capture,
)


class CaptureEventsHelperTests(unittest.TestCase):
    def test_build_new_capture_payload_includes_camera_id(self):
        payload = build_new_capture_payload(
            {
                "event_id": "ev1",
                "file_path": "/tmp/cam1_watch_1.jpg",
                "thumb_path": "/tmp/thumb.jpg",
                "captured_ts": 1700000000,
                "shape_name": "Driveway",
                "trigger_type": "zone_enter",
            },
            camera_id="cam1",
        )
        self.assertEqual(payload["event_id"], "ev1")
        self.assertEqual(payload["camera_id"], "cam1")
        self.assertEqual(payload["shape_name"], "Driveway")
        self.assertTrue(payload["file_uri"].startswith("file:"))

    def test_build_motion_watch_terminal_payload(self):
        payload = build_motion_watch_terminal_payload(
            result={
                "file_path": "/tmp/cam1_watch_1.jpg",
                "shape_name": "Driveway",
                "trigger_type": "zone_enter",
            },
            camera_id="cam1",
            camera_label="Front Cam",
            thumb_b64="abc123",
        )
        self.assertEqual(payload["camera_id"], "cam1")
        self.assertEqual(payload["camera_label"], "Front Cam")
        self.assertIn("Driveway", payload["text"])
        self.assertEqual(payload["image_b64"], "abc123")
        self.assertTrue(str(payload["link"]).startswith("file:"))

    def test_emit_new_capture_targets_camera_room(self):
        sio = MagicMock()
        result = {
            "event_id": "ev2",
            "file_path": "/tmp/cam2_watch_2.jpg",
            "captured_ts": 1700000001,
        }
        emit_new_capture(sio, result, camera_id="cam2")
        self.assertTrue(sio.emit.called)
        calls = [c.args[0] for c in sio.emit.call_args_list]
        self.assertIn("new_capture", calls)
        kwargs_list = [c.kwargs for c in sio.emit.call_args_list]
        self.assertTrue(any(k.get("room") == "camera:cam2" for k in kwargs_list))

    def test_build_new_capture_payload_from_snapshot_fallback(self):
        payload = build_new_capture_payload(
            {
                "file_path": "/tmp/cam1_watch_99.jpg",
                "camera_id": "cam1",
                "captured_ts": 1700000099,
                "shape_name": "Lane",
                "trigger_type": "line_cross",
                "ingested": False,
            },
            camera_id="cam1",
        )
        self.assertEqual(payload["shape_name"], "Lane")
        self.assertEqual(payload["trigger_type"], "line_cross")
        self.assertTrue(payload["file_uri"].startswith("file:"))

    def test_build_new_capture_payload_includes_inline_thumb(self):
        payload = build_new_capture_payload(
            {
                "file_path": "/tmp/cam1_watch_1.jpg",
                "thumb_b64": "inline-thumb",
            },
            camera_id="cam1",
        )
        self.assertEqual(payload.get("thumb_b64"), "inline-thumb")

    def test_build_terminal_capture_failure_payload(self):
        payload = build_terminal_capture_failure_payload(
            error={"reason": "no_frame_available", "capture_failed": True},
            camera_id="cam1",
            camera_label="Front Cam",
            rule_name="Driveway rule",
        )
        self.assertEqual(payload["camera_id"], "cam1")
        self.assertEqual(payload["kind"], "warning")
        self.assertIn("no video frame available", payload["text"])
        self.assertIn("Driveway rule", payload["text"])

    def test_capture_failure_reason_label(self):
        self.assertEqual(
            capture_failure_reason_label("stream_not_active"),
            "camera stream not active",
        )


if __name__ == "__main__":
    unittest.main()
