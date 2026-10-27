"""Tests for camera profile export/import helpers."""

from __future__ import annotations

import unittest
from unittest.mock import patch

from desktop.utils.camera_profiles import (
    PROFILE_SCHEMA_VERSION,
    build_camera_profile_payload,
    build_shape_id_map,
    export_rule_for_profile,
    export_rules_for_profile,
    merge_profile_dicts,
    profile_to_widget_view,
    remap_shape_ids,
    rewrite_rule_for_camera,
    rewrite_rules_for_camera,
    sanitize_shapes,
)
from desktop.utils.event_rules_api import LEGACY_RULE_NAME, replace_camera_rules


def _sample_shape(sid: str, kind: str = "zone") -> dict:
    return {
        "id": sid,
        "kind": kind,
        "label": f"Shape {sid}",
        "pts": [{"x": 0.1, "y": 0.1}, {"x": 0.9, "y": 0.1}, {"x": 0.9, "y": 0.9}],
    }


def _sample_rule(*, rid: str = "rule_1", camera_id: str = "cam_a", shape_id: str = "zone_1") -> dict:
    return {
        "id": rid,
        "name": "East cars",
        "enabled": True,
        "camera_id": camera_id,
        "trigger": "zone_enter",
        "shape_id": shape_id,
        "conditions": {
            "cooldown_sec": 1.0,
            "motion_path": [{"x": 0.05, "y": 0.5}, {"x": 0.95, "y": 0.48}],
            "motion_path_space": "frame",
            "motion_path_shape_ref": shape_id,
            "show_counter": "increment",
            "counter_value": 42,
            "counter_pill_label": "Cars",
        },
        "actions": [
            {"type": "snapshot", "include_overlays": True, "save_dir": "captures/test"},
            {
                "type": "script",
                "path": "/tmp/alert.sh",
                "runner": "bash",
                "args": ["--camera", camera_id],
            },
        ],
        "trigger_count": 99,
    }


class CameraProfileExportTests(unittest.TestCase):
    def test_sanitize_shapes_excludes_kinds(self):
        shapes = [
            _sample_shape("z1", "zone"),
            _sample_shape("l1", "line"),
            _sample_shape("t1", "tag"),
        ]
        out = sanitize_shapes(shapes, exclude_kinds={"line"})
        kinds = {s["kind"] for s in out}
        self.assertEqual(kinds, {"zone", "tag"})

    def test_export_rule_strips_runtime_and_preserves_payload(self):
        exported = export_rule_for_profile(_sample_rule(), source_camera_id="cam_src")
        self.assertIsNotNone(exported)
        assert exported is not None
        self.assertNotIn("id", exported)
        self.assertNotIn("trigger_count", exported)
        self.assertEqual(exported["camera_id"], "cam_src")
        self.assertEqual(exported["actions"][0]["type"], "snapshot")
        self.assertEqual(exported["actions"][1]["type"], "script")
        self.assertEqual(exported["conditions"]["motion_path"][0]["x"], 0.05)
        self.assertNotIn("counter_value", exported["conditions"])

    def test_export_skips_legacy_rules_by_default(self):
        legacy = {"id": "legacy_mw_cam_a", "name": LEGACY_RULE_NAME, "camera_id": "cam_a"}
        custom = _sample_rule()
        out = export_rules_for_profile([legacy, custom], source_camera_id="cam_a")
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0]["name"], "East cars")

    def test_build_profile_payload_schema(self):
        payload = build_camera_profile_payload(
            name="Test",
            source_camera_id="cam1",
            overlays={"shapes": [_sample_shape("zone_1")]},
            monitoring_tools={"event_rules": export_rules_for_profile([_sample_rule()])},
        )
        self.assertEqual(payload["meta"]["schema_version"], PROFILE_SCHEMA_VERSION)
        self.assertEqual(payload["meta"]["source_camera_id"], "cam1")


class CameraProfileRewriteTests(unittest.TestCase):
    def test_rewrite_rule_camera_and_shape_ids(self):
        id_map = {"zone_1": "zone_99"}
        rewritten = rewrite_rule_for_camera(_sample_rule(), "cam_b", id_map)
        self.assertEqual(rewritten["camera_id"], "cam_b")
        self.assertEqual(rewritten["shape_id"], "zone_99")
        self.assertEqual(rewritten["conditions"]["motion_path_shape_ref"], "zone_99")
        self.assertNotIn("counter_value", rewritten["conditions"])

    def test_rewrite_rules_batch(self):
        rules = [_sample_rule(rid="r1"), _sample_rule(rid="r2", shape_id="zone_2")]
        out = rewrite_rules_for_camera(rules, "cam_target")
        self.assertEqual(len(out), 2)
        self.assertTrue(all(r["camera_id"] == "cam_target" for r in out))

    def test_remap_shape_ids(self):
        id_map = build_shape_id_map([_sample_shape("old")], preserve_ids=False)
        new_id = id_map["old"]
        self.assertNotEqual(new_id, "old")
        remapped = remap_shape_ids([_sample_shape("old")], id_map)
        self.assertEqual(remapped[0]["id"], new_id)


class CameraProfileViewTests(unittest.TestCase):
    def test_profile_to_widget_view_maps_sections(self):
        profile = {
            "overlays": {
                "shapes": [_sample_shape("z1")],
                "motion_boxes_enabled": True,
                "detection_settings": {"style": "Box"},
                "show_shape_labels": False,
            },
            "ai_pipeline": {
                "desktop_object_detection_enabled": True,
                "backend_detection": {"verification_enabled": True},
            },
            "monitoring_tools": {
                "motion_watch_settings": {"cooldown_ms": 2000},
                "event_rules": [_sample_rule()],
            },
        }
        view = profile_to_widget_view(profile)
        self.assertEqual(view["detection_overlay_settings"]["style"], "Box")
        self.assertTrue(view["motion_boxes_enabled"])
        self.assertFalse(view["show_shape_labels"])
        self.assertEqual(view["backend_detection"]["verification_enabled"], True)
        self.assertEqual(len(view["event_rules"]), 1)


class CameraProfileMergeTests(unittest.TestCase):
    def test_merge_profile_dicts_appends_shapes_and_rules(self):
        p1 = {
            "overlays": {"shapes": [_sample_shape("a")], "motion_boxes_enabled": True},
            "monitoring_tools": {"event_rules": [_sample_rule(rid="r1")]},
        }
        p2 = {
            "overlays": {"shapes": [_sample_shape("b")], "debug_overlay_enabled": True},
            "monitoring_tools": {"event_rules": [_sample_rule(rid="r2")]},
        }
        merged = merge_profile_dicts([p1, p2])
        self.assertEqual(len(merged["overlays"]["shapes"]), 2)
        self.assertTrue(merged["overlays"]["motion_boxes_enabled"])
        self.assertTrue(merged["overlays"]["debug_overlay_enabled"])
        self.assertEqual(len(merged["monitoring_tools"]["event_rules"]), 2)


class ReplaceCameraRulesTests(unittest.TestCase):
    @patch("desktop.utils.event_rules_api.save_rule")
    @patch("desktop.utils.event_rules_api.delete_rule")
    @patch("desktop.utils.event_rules_api.list_rules")
    def test_replace_camera_rules_deletes_then_saves(self, mock_list, mock_delete, mock_save):
        mock_list.return_value = [{"id": "old1"}, {"id": "old2"}]
        mock_save.return_value = {"id": "new1"}

        count = replace_camera_rules(
            "http://localhost:5000/api",
            "cam_x",
            [rewrite_rule_for_camera(_sample_rule(), "cam_x")],
        )
        self.assertEqual(mock_delete.call_count, 2)
        self.assertEqual(mock_save.call_count, 1)
        self.assertEqual(count, 1)
        saved_body = mock_save.call_args[0][1]
        self.assertEqual(saved_body["camera_id"], "cam_x")
        self.assertNotIn("id", saved_body)


if __name__ == "__main__":
    unittest.main()
