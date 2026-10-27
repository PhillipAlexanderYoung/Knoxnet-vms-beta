"""Focused tests for camera online/last-seen status sync."""

from __future__ import annotations

import unittest


class BackendRuntimeStatusTests(unittest.TestCase):
    def test_runtime_stream_frame_marks_snapshot_live(self):
        import app

        original = getattr(app, "STREAM_SERVER_GLOBAL", None)

        class StreamServer:
            active_streams = {"cam-live:sub": {"last_frame": object()}}

        try:
            app.STREAM_SERVER_GLOBAL = StreamServer()
            camera = {"id": "cam-live", "status": "offline"}
            app._apply_runtime_camera_seen_status(camera)
        finally:
            app.STREAM_SERVER_GLOBAL = original

        self.assertTrue(camera.get("online"))
        self.assertTrue(camera.get("ready"))
        self.assertEqual(camera.get("status"), "live")
        self.assertTrue(camera.get("last_seen"))

    def test_runtime_stream_without_frame_does_not_mark_seen(self):
        import app

        original = getattr(app, "STREAM_SERVER_GLOBAL", None)

        class StreamServer:
            active_streams = {"cam-id": {"last_frame": None}}

        try:
            app.STREAM_SERVER_GLOBAL = StreamServer()
            camera = {"id": "cam-id", "status": "offline"}
            app._apply_runtime_camera_seen_status(camera)
        finally:
            app.STREAM_SERVER_GLOBAL = original

        self.assertNotIn("last_seen", camera)
        self.assertNotEqual(camera.get("status"), "live")


if __name__ == "__main__":
    unittest.main()
