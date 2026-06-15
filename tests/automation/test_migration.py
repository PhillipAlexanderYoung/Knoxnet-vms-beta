"""Stage 5 tests: Motion Watch → Event Rules migration."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from desktop.utils.event_rules_api import (
    LEGACY_RULE_NAME,
    ensure_legacy_rule,
    has_custom_rules,
    is_legacy_rule,
    legacy_rule_from_motion_watch_settings,
    load_motion_watch_settings_from_disk,
    migrate_motion_watch_settings,
    snapshot_action_from_motion_watch_settings,
)


class LegacyRuleMappingTests(unittest.TestCase):
    def test_snapshot_action_from_motion_watch_settings(self):
        action = snapshot_action_from_motion_watch_settings(
            {
                "include_overlays": False,
                "save_dir": "captures/custom",
                "resize_w": 1280,
                "quality": 90,
            }
        )
        self.assertEqual(action["type"], "snapshot")
        self.assertFalse(action["include_overlays"])
        self.assertEqual(action["save_dir"], "captures/custom")
        self.assertEqual(action["resize_w"], 1280)
        self.assertEqual(action["quality"], 90)

    def test_snapshot_action_omits_quality_when_unset(self):
        action = snapshot_action_from_motion_watch_settings({})
        self.assertEqual(action["type"], "snapshot")
        self.assertTrue(action["include_overlays"])
        self.assertEqual(action["save_dir"], "captures/motion_watch")
        self.assertEqual(action["resize_w"], 0)
        self.assertNotIn("quality", action)

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
        self.assertTrue(rule["conditions"]["allow_zone"])
        self.assertTrue(rule["conditions"]["allow_line"])
        self.assertEqual(rule["actions"][0]["type"], "snapshot")
        self.assertFalse(rule["actions"][0]["include_overlays"])
        self.assertEqual(rule["actions"][0]["save_dir"], "captures/custom")

    def test_cooldown_sec_fallback(self):
        rule = legacy_rule_from_motion_watch_settings("cam_b", {"cooldown_sec": 2.5})
        self.assertEqual(rule["conditions"]["cooldown_sec"], 2.5)

    def test_line_only_trigger_when_zone_disabled(self):
        rule = legacy_rule_from_motion_watch_settings(
            "cam_c",
            {"allow_zone": False, "allow_line": True},
        )
        self.assertEqual(rule["trigger"], "line_cross")
        self.assertFalse(rule["conditions"]["allow_zone"])
        self.assertTrue(rule["conditions"]["allow_line"])

    def test_zone_trigger_when_both_enabled(self):
        rule = legacy_rule_from_motion_watch_settings(
            "cam_d",
            {"allow_zone": True, "allow_line": True},
        )
        self.assertEqual(rule["trigger"], "zone_enter")


class LegacyRuleDetectionTests(unittest.TestCase):
    def test_is_legacy_rule_by_name_and_id(self):
        self.assertTrue(is_legacy_rule({"name": LEGACY_RULE_NAME}))
        self.assertTrue(is_legacy_rule({"id": "legacy_mw_cam1"}))
        self.assertFalse(is_legacy_rule({"name": "Custom Alert", "id": "r1"}))

    def test_has_custom_rules(self):
        self.assertFalse(has_custom_rules([{"name": LEGACY_RULE_NAME, "id": "legacy_mw_x"}]))
        self.assertTrue(
            has_custom_rules(
                [
                    {"name": LEGACY_RULE_NAME, "id": "legacy_mw_x"},
                    {"name": "Person in zone", "id": "custom_1"},
                ]
            )
        )


class DiskSettingsLoadTests(unittest.TestCase):
    def test_loads_per_camera_settings(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "motion_watch_settings.json"
            path.write_text(
                json.dumps(
                    {
                        "cam1": {
                            "cooldown_ms": 4000,
                            "save_dir": "captures/test",
                            "include_overlays": False,
                        }
                    }
                )
            )
            with patch("desktop.utils.event_rules_api.get_data_dir", return_value=Path(tmp)):
                settings = load_motion_watch_settings_from_disk("cam1")
            self.assertEqual(settings["cooldown_ms"], 4000)
            self.assertEqual(settings["save_dir"], "captures/test")
            self.assertFalse(settings["include_overlays"])

    def test_disk_defaults_use_one_second_cooldown(self):
        with tempfile.TemporaryDirectory() as tmp:
            with patch("desktop.utils.event_rules_api.get_data_dir", return_value=Path(tmp)):
                settings = load_motion_watch_settings_from_disk("missing_cam")
            self.assertEqual(settings["cooldown_ms"], 1000)

    def test_migrates_cooldown_sec_to_ms(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "motion_watch_settings.json"
            path.write_text(json.dumps({"cam1": {"cooldown_sec": 6}}))
            with patch("desktop.utils.event_rules_api.get_data_dir", return_value=Path(tmp)):
                settings = load_motion_watch_settings_from_disk("cam1")
            self.assertEqual(settings["cooldown_ms"], 6000)


class MigrateMotionWatchSettingsTests(unittest.TestCase):
    API = "http://localhost:5000/api"

    @patch("desktop.utils.event_rules_api.save_rule")
    @patch("desktop.utils.event_rules_api.list_rules")
    def test_creates_legacy_rule_when_none_exist(self, mock_list, mock_save):
        mock_list.return_value = []
        mock_save.return_value = {"id": "legacy_mw_cam1"}

        rid = migrate_motion_watch_settings(
            self.API,
            "cam1",
            {"cooldown_ms": 3000, "save_dir": "captures/mw", "include_overlays": True},
        )
        self.assertEqual(rid, "legacy_mw_cam1")
        mock_save.assert_called_once()
        body = mock_save.call_args[0][1]
        self.assertEqual(body["name"], LEGACY_RULE_NAME)
        self.assertEqual(body["conditions"]["cooldown_sec"], 3.0)

    @patch("desktop.utils.event_rules_api.save_rule")
    @patch("desktop.utils.event_rules_api.list_rules")
    def test_idempotent_updates_existing_legacy_rule(self, mock_list, mock_save):
        mock_list.return_value = [{"id": "legacy_mw_cam1", "name": LEGACY_RULE_NAME}]
        mock_save.return_value = {"id": "legacy_mw_cam1"}

        rid = migrate_motion_watch_settings(
            self.API,
            "cam1",
            {"cooldown_ms": 8000, "save_dir": "captures/updated", "include_overlays": False},
        )
        self.assertEqual(rid, "legacy_mw_cam1")
        mock_save.assert_called_once()
        _, kwargs = mock_save.call_args
        self.assertEqual(kwargs.get("rule_id"), "legacy_mw_cam1")
        body = mock_save.call_args[0][1]
        self.assertEqual(body["conditions"]["cooldown_sec"], 8.0)
        self.assertFalse(body["actions"][0]["include_overlays"])
        self.assertEqual(body["actions"][0]["save_dir"], "captures/updated")

    @patch("desktop.utils.event_rules_api.save_rule")
    @patch("desktop.utils.event_rules_api.list_rules")
    def test_skips_when_custom_rules_exist(self, mock_list, mock_save):
        mock_list.return_value = [
            {"id": "custom_1", "name": "Person alert"},
        ]

        rid = migrate_motion_watch_settings(self.API, "cam1", {"cooldown_ms": 3000})
        self.assertIsNone(rid)
        mock_save.assert_not_called()

    @patch("desktop.utils.event_rules_api.ensure_legacy_rule")
    @patch("desktop.utils.event_rules_api.load_motion_watch_settings_from_disk")
    def test_loads_from_disk_when_settings_omitted(self, mock_load, mock_ensure):
        mock_load.return_value = {"cooldown_ms": 2000}
        mock_ensure.return_value = "legacy_mw_cam2"

        rid = migrate_motion_watch_settings(self.API, "cam2")
        self.assertEqual(rid, "legacy_mw_cam2")
        mock_load.assert_called_once_with("cam2")
        mock_ensure.assert_called_once_with(self.API, "cam2", {"cooldown_ms": 2000})

    @patch("desktop.utils.event_rules_api.save_rule")
    @patch("desktop.utils.event_rules_api.list_rules")
    def test_double_migrate_does_not_create_duplicate(self, mock_list, mock_save):
        stored = []

        def _save(_api, body, *, rule_id=None):
            if rule_id:
                for r in stored:
                    if r["id"] == rule_id:
                        r.update(body)
                        return r
            else:
                rid = body.get("id") or "legacy_mw_cam1"
                entry = {"id": rid, **body}
                stored.append(entry)
                return entry
            return None

        mock_save.side_effect = _save
        mock_list.side_effect = lambda *_a, **_k: list(stored)

        settings = {"cooldown_ms": 5000, "save_dir": "captures/a"}
        rid1 = migrate_motion_watch_settings(self.API, "cam1", settings)
        rid2 = migrate_motion_watch_settings(self.API, "cam1", settings)

        self.assertEqual(rid1, rid2)
        self.assertEqual(len(stored), 1)
        self.assertEqual(mock_save.call_count, 2)


class EnsureLegacyRuleTests(unittest.TestCase):
    API = "http://localhost:5000/api"

    @patch("desktop.utils.event_rules_api.save_rule")
    @patch("desktop.utils.event_rules_api.list_rules")
    def test_syncs_allow_zone_line_on_update(self, mock_list, mock_save):
        mock_list.return_value = [{"id": "legacy_mw_cam1", "name": LEGACY_RULE_NAME}]
        mock_save.return_value = {"id": "legacy_mw_cam1"}

        ensure_legacy_rule(
            self.API,
            "cam1",
            {"allow_zone": False, "allow_line": True, "cooldown_ms": 1000},
        )
        body = mock_save.call_args[0][1]
        self.assertEqual(body["trigger"], "line_cross")
        self.assertFalse(body["conditions"]["allow_zone"])
        self.assertTrue(body["conditions"]["allow_line"])


if __name__ == "__main__":
    unittest.main()
