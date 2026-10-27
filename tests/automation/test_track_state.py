"""Unit tests for TrackSceneEngine (Event Rules Stage 1)."""

from __future__ import annotations

import unittest

from core.automation.track_state import BACKEND_SORT_NAMESPACE, MOTION_BOX_NAMESPACE, TrackSceneEngine


def _square_zone(zid: str = "zone_1", margin: float = 0.2):
    m = margin
    return {
        "zones": [
            {
                "id": zid,
                "name": "Zone 1",
                "enabled": True,
                "points": [
                    {"x": m, "y": m},
                    {"x": 1.0 - m, "y": m},
                    {"x": 1.0 - m, "y": 1.0 - m},
                    {"x": m, "y": 1.0 - m},
                ],
            }
        ],
        "lines": [],
    }


def _horizontal_line(lid: str = "line_1", y: float = 0.5):
    return {
        "zones": [],
        "lines": [
            {
                "id": lid,
                "name": "Line 1",
                "enabled": True,
                "p1": {"x": 0.0, "y": y},
                "p2": {"x": 1.0, "y": y},
            }
        ],
    }


def _track(tid: int, nx: float, ny: float, *, cls: str = "car", conf: float = 0.9):
    fw, fh = 1920, 1080
    w, h = 80, 60
    cx = nx * fw
    cy = ny * fh
    return {
        "id": tid,
        "class": cls,
        "confidence": conf,
        "bbox": {"x": cx - w / 2, "y": cy - h / 2, "w": w, "h": h},
        "center": {"nx": nx, "ny": ny},
    }


class TrackSceneEngineTests(unittest.TestCase):
    def setUp(self):
        self.engine = TrackSceneEngine(hysteresis_frames=3, dwell_sec=1.0)
        self.cam = "cam_test"
        self.fw = 1920
        self.fh = 1080

    def _update(self, tracks, shapes, *, now=1000.0, dwell_sec=None):
        return self.engine.update(
            camera_id=self.cam,
            tracks=tracks,
            shapes=shapes,
            frame_w=self.fw,
            frame_h=self.fh,
            tracker_namespace=BACKEND_SORT_NAMESPACE,
            now=now,
            dwell_sec=dwell_sec,
        )

    def test_zone_enter_after_hysteresis(self):
        shapes = _square_zone()
        events = []
        for i in range(3):
            events = self._update([_track(1, 0.5, 0.5)], shapes, now=1000.0 + i * 0.04)
        types = [e["event_type"] for e in events]
        self.assertIn("zone_enter", types)
        ev = next(e for e in events if e["event_type"] == "zone_enter")
        self.assertEqual(ev["track_id"], 1)
        self.assertEqual(ev["shape_id"], "zone_1")
        self.assertEqual(ev["tracker_namespace"], BACKEND_SORT_NAMESPACE)

    def test_zone_enter_includes_centroid_history(self):
        shapes = _square_zone()
        events = []
        for i in range(3):
            nx = 0.25 + i * 0.1
            events = self._update([_track(1, nx, 0.5)], shapes, now=1000.0 + i * 0.04)
        enter = next(e for e in events if e["event_type"] == "zone_enter")
        hist = enter.get("centroid_history") or []
        self.assertGreaterEqual(len(hist), 3)
        self.assertAlmostEqual(float(hist[-1]["x"]), 0.45, places=2)

    def test_zone_exit_and_jitter(self):
        shapes = _square_zone()
        # Enter
        for i in range(3):
            self._update([_track(1, 0.5, 0.5)], shapes, now=1000.0 + i * 0.04)
        # Boundary jitter (outside 1-2 frames should not exit)
        self._update([_track(1, 0.05, 0.5)], shapes, now=1001.0)
        self._update([_track(1, 0.5, 0.5)], shapes, now=1001.04)
        events = self._update([_track(1, 0.05, 0.5)], shapes, now=1001.08)
        self.assertFalse(any(e["event_type"] == "zone_exit" for e in events))
        # Sustained outside -> exit
        all_events = []
        for i in range(4):
            all_events.extend(self._update([_track(1, 0.05, 0.5)], shapes, now=1002.0 + i * 0.04))
        self.assertTrue(any(e["event_type"] == "zone_exit" for e in all_events))

    def test_dwell_met(self):
        shapes = _square_zone()
        for i in range(3):
            self._update([_track(1, 0.5, 0.5)], shapes, now=1000.0 + i * 0.04)
        events = self._update([_track(1, 0.5, 0.5)], shapes, now=1001.5, dwell_sec=1.0)
        self.assertTrue(any(e["event_type"] == "dwell_met" for e in events))
        dwell = next(e for e in events if e["event_type"] == "dwell_met")
        self.assertGreaterEqual(float(dwell["dwell_sec"]), 1.0)

    def test_line_cross_both_directions(self):
        shapes = _horizontal_line()
        engine = TrackSceneEngine(hysteresis_frames=2, dwell_sec=1.0)
        # Establish below line (ny=0.4 -> negative side)
        for i in range(2):
            engine.update(
                camera_id=self.cam,
                tracks=[_track(1, 0.5, 0.40)],
                shapes=shapes,
                frame_w=self.fw,
                frame_h=self.fh,
                now=1000.0 + i * 0.1,
            )
        events = []
        for i in range(2):
            events = engine.update(
                camera_id=self.cam,
                tracks=[_track(1, 0.5, 0.60)],
                shapes=shapes,
                frame_w=self.fw,
                frame_h=self.fh,
                now=1001.0 + i * 0.1,
            )
        cross_up = [e for e in events if e["event_type"] == "line_cross"]
        self.assertTrue(cross_up)
        self.assertIn(cross_up[0]["direction"], ("left_to_right", "positive", "negative"))

        # Cross back below
        for i in range(2):
            engine.update(
                camera_id=self.cam,
                tracks=[_track(1, 0.5, 0.40)],
                shapes=shapes,
                frame_w=self.fw,
                frame_h=self.fh,
                now=1002.0 + i * 0.1,
            )
        events = []
        for i in range(2):
            events = engine.update(
                camera_id=self.cam,
                tracks=[_track(1, 0.5, 0.40)],
                shapes=shapes,
                frame_w=self.fw,
                frame_h=self.fh,
                now=1003.0 + i * 0.1,
            )
        # May not re-fire if already on same side; move above again from below
        for i in range(2):
            engine.update(
                camera_id=self.cam,
                tracks=[_track(1, 0.5, 0.35)],
                shapes=shapes,
                frame_w=self.fw,
                frame_h=self.fh,
                now=1004.0 + i * 0.1,
            )
        events = []
        for i in range(2):
            events = engine.update(
                camera_id=self.cam,
                tracks=[_track(1, 0.5, 0.65)],
                shapes=shapes,
                frame_w=self.fw,
                frame_h=self.fh,
                now=1005.0 + i * 0.1,
            )
        self.assertTrue(any(e["event_type"] == "line_cross" for e in events))

    def test_two_vehicles_independent(self):
        shapes = _square_zone()
        for i in range(3):
            self._update(
                [_track(1, 0.3, 0.5), _track(2, 0.7, 0.5)],
                shapes,
                now=1000.0 + i * 0.04,
            )
        events = self._update([_track(2, 0.7, 0.5)], shapes, now=1001.0)
        enter_ids = {e["track_id"] for e in events if e["event_type"] == "zone_enter"}
        self.assertNotIn(1, enter_ids)  # track 1 already inside; no duplicate enter

    def test_track_lost(self):
        shapes = _square_zone()
        for i in range(3):
            self._update([_track(1, 0.5, 0.5)], shapes, now=1000.0 + i * 0.04)
        events = self._update([], shapes, now=1001.0)
        # LOST_GRACE_FRAMES=2, need missing > 2
        events = self._update([], shapes, now=1001.1)
        events = self._update([], shapes, now=1001.2)
        self.assertTrue(any(e["event_type"] == "track_lost" for e in events))

    def test_track_reacquired_by_iou(self):
        shapes = _square_zone()
        for i in range(3):
            self._update([_track(1, 0.5, 0.5)], shapes, now=1000.0 + i * 0.04)
        for _ in range(4):
            self._update([], shapes, now=1001.5)
        # New ID, overlapping bbox -> reacquired
        events = self._update([_track(99, 0.5, 0.5)], shapes, now=1002.0)
        self.assertTrue(any(e["event_type"] == "track_reacquired" for e in events))

    def test_reacquire_does_not_emit_duplicate_zone_enter(self):
        shapes = _square_zone()
        for i in range(3):
            self._update([_track(1, 0.5, 0.5)], shapes, now=1000.0 + i * 0.04)
        for _ in range(4):
            self._update([], shapes, now=1001.5)
        events = self._update([_track(99, 0.5, 0.5)], shapes, now=1002.0)
        self.assertFalse(any(e["event_type"] == "zone_enter" for e in events))

    def test_pts_alias_normalized(self):
        shapes = {
            "zones": [
                {
                    "id": "z_pts",
                    "label": "Pts Zone",
                    "enabled": True,
                    "pts": [
                        {"x": 0.2, "y": 0.2},
                        {"x": 0.8, "y": 0.2},
                        {"x": 0.8, "y": 0.8},
                        {"x": 0.2, "y": 0.8},
                    ],
                }
            ]
        }
        events = []
        for i in range(3):
            events = self._update([_track(5, 0.5, 0.5)], shapes, now=2000.0 + i * 0.04)
        self.assertTrue(any(e["event_type"] == "zone_enter" for e in events))

    def test_near_tag_emits_after_hysteresis(self):
        engine = TrackSceneEngine(hysteresis_frames=2, dwell_sec=1.0)
        shapes = {
            "zones": [],
            "lines": [],
            "tags": [
                {
                    "id": "tag_1",
                    "name": "Tag 1",
                    "enabled": True,
                    "anchor": {"x": 0.5, "y": 0.5},
                }
            ],
        }
        track = {
            "id": 3,
            "class": "car",
            "confidence": 0.9,
            "bbox": {"x": 300, "y": 220, "w": 80, "h": 60},
            "center": {"nx": 0.5, "ny": 0.5},
        }
        events = []
        for i in range(3):
            events.extend(
                engine.update(
                    camera_id="cam1",
                    tracks=[track],
                    shapes=shapes,
                    frame_w=640,
                    frame_h=480,
                    now=1000.0 + i * 0.05,
                )
            )
        near = [e for e in events if e.get("event_type") == "near_tag"]
        self.assertTrue(near)
        self.assertEqual(near[0].get("shape_id"), "tag_1")

    def test_motion_box_namespace_zone_enter(self):
        engine = TrackSceneEngine(hysteresis_frames=2, dwell_sec=1.0)
        shapes = _square_zone()
        track = _track(42, 0.5, 0.5)
        events = []
        for i in range(3):
            events.extend(
                engine.update(
                    camera_id="cam1",
                    tracks=[track],
                    shapes=shapes,
                    frame_w=self.fw,
                    frame_h=self.fh,
                    tracker_namespace=MOTION_BOX_NAMESPACE,
                    now=1000.0 + i * 0.05,
                )
            )
        enter = [e for e in events if e.get("event_type") == "zone_enter"]
        self.assertTrue(enter)
        self.assertEqual(enter[-1].get("tracker_namespace"), MOTION_BOX_NAMESPACE)


if __name__ == "__main__":
    unittest.main()
