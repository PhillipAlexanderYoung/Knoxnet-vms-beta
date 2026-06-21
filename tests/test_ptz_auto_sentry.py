from __future__ import annotations

import time
import unittest

from core.ptz_auto_sentry import (
    AutoSentryConfig,
    SmoothTargetTracker,
    observations_from_payload,
)


def _payload(x, y, w=80, h=120, cls="person", conf=0.9, det_id="a"):
    return {
        "camera_id": "cam1",
        "frame_width": 640,
        "frame_height": 480,
        "detections": [
            {
                "id": det_id,
                "class": cls,
                "confidence": conf,
                "bbox": {"x": x, "y": y, "w": w, "h": h},
            }
        ],
    }


class AutoSentryTrackerTests(unittest.TestCase):
    def test_click_lock_persists_target_state(self):
        cfg = AutoSentryConfig()
        tracker = SmoothTargetTracker()
        obs = observations_from_payload({"target": _payload(420, 170)["detections"][0], "frame_width": 640, "frame_height": 480})

        tracker.lock(obs[0], cfg, label="person")
        target = tracker.target_dict()

        self.assertTrue(target["locked"])
        self.assertEqual(target["class"], "person")
        self.assertIn("bbox", target)
        self.assertIn("predicted_center", target)

    def test_locked_target_reassociates_by_position_and_class(self):
        cfg = AutoSentryConfig()
        tracker = SmoothTargetTracker()
        first = observations_from_payload(_payload(320, 180, det_id="old"))[0]
        tracker.lock(first, cfg, label="person")

        # Same object gets a new detector id but remains close enough to the lock.
        second = observations_from_payload(_payload(332, 184, det_id="new"))[0]
        selected = tracker.ingest([second], cfg)

        self.assertIsNotNone(selected)
        self.assertEqual(tracker.target_id, first.target_id)
        self.assertGreater(tracker.last_confirmed_ts, 0)

    def test_prediction_continues_during_short_dropout(self):
        cfg = AutoSentryConfig.from_params({
            "lostTargetTtl": 1.0,
            "scanAfterSeconds": 3.0,
            "predictionSeconds": 0.5,
        })
        tracker = SmoothTargetTracker()
        obs1 = observations_from_payload(_payload(300, 180, det_id="p"))[0]
        tracker.lock(obs1, cfg, label="person")
        time.sleep(0.01)
        obs2 = observations_from_payload(_payload(340, 180, det_id="p"))[0]
        tracker.ingest([obs2], cfg)
        tracker.last_seen = time.time() - 1.2

        cmd = tracker.compute(cfg, now=time.time())

        self.assertEqual(cmd.mode, "predictive_search")
        self.assertIsNotNone(cmd.target)
        self.assertEqual(cmd.target["state"], "lost")
        self.assertIn("predicted_bbox", cmd.target)

    def test_ramp_limits_speed_changes(self):
        cfg = AutoSentryConfig.from_params({
            "maxSpeed": 0.8,
            "maxAccelPerSec": 0.5,
            "commandCooldown": 0.1,
        })
        tracker = SmoothTargetTracker()
        obs = observations_from_payload(_payload(600, 180, det_id="fast"))[0]
        tracker.lock(obs, cfg, label="person")

        cmd1 = tracker.compute(cfg, now=100.0)
        cmd2 = tracker.compute(cfg, now=100.11)

        self.assertLessEqual(abs(cmd2.pan_speed - cmd1.pan_speed), 0.06)


if __name__ == "__main__":
    unittest.main()
