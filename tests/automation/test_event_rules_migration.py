"""Tests for Event Rules legacy migration helper."""

import unittest

from desktop.utils.event_rules_api import legacy_rule_from_motion_watch_settings


class LegacyMigrationTests(unittest.TestCase):
    def test_builds_snapshot_rule_from_settings(self):
        rule = legacy_rule_from_motion_watch_settings(
            "cam_a",
            {
                "cooldown_ms": 5000,
                "include_overlays": False,
                "save_dir": "captures/custom",
            },
        )
        self.assertEqual(rule["camera_id"], "cam_a")
        self.assertEqual(rule["trigger"], "zone_enter")
        self.assertEqual(rule["conditions"]["cooldown_sec"], 5.0)
        self.assertEqual(rule["actions"][0]["type"], "snapshot")
        self.assertFalse(rule["actions"][0]["include_overlays"])
        self.assertEqual(rule["actions"][0]["save_dir"], "captures/custom")


if __name__ == "__main__":
    unittest.main()
