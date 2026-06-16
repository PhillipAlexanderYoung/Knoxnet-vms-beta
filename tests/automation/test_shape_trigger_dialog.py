"""Unit tests for shape trigger dialog helpers."""

from __future__ import annotations

import unittest

from desktop.utils.event_rules_api import snapshot_action_from_motion_watch_settings
from desktop.utils.shape_trigger_helpers import (
    DEFAULT_TRIGGER_MODE,
    EXPLICIT_SHAPE_TRIGGERS,
    build_event_source_conditions,
    build_fresh_shape_rule_conditions,
    build_rule_actions,
    effective_trigger_from_mode,
    parse_event_source_flags,
    rule_has_script_action,
    script_action_from_settings,
    trigger_mode_options_for_kind,
)
from desktop.widgets.shape_trigger_preview import event_source_description, shape_trigger_dialog_key
from desktop.utils.shape_trigger_helpers import (
    path_tolerance_from_slider,
    path_tolerance_slider_value,
)
from core.automation.conditions import BACKEND_SORT_NAMESPACE, MOTION_BOX_NAMESPACE


def rule_has_snapshot_action(rule: dict) -> bool:
    actions = rule.get("actions") if isinstance(rule.get("actions"), list) else []
    return any(isinstance(a, dict) and str(a.get("type")) == "snapshot" for a in actions)


def build_rule_snapshot_actions(*, take_screenshot: bool, motion_watch_settings: dict) -> list:
    if not take_screenshot:
        return []
    return [snapshot_action_from_motion_watch_settings(motion_watch_settings)]


class ShapeTriggerDialogHelperTests(unittest.TestCase):
    def test_dialog_key_new_rule(self):
        self.assertEqual(shape_trigger_dialog_key("zone_abc"), "zone_abc:__new__")
        self.assertEqual(shape_trigger_dialog_key("zone_abc", None), "zone_abc:__new__")
        self.assertEqual(shape_trigger_dialog_key("zone_abc", {}), "zone_abc:__new__")

    def test_dialog_key_existing_rule(self):
        self.assertEqual(
            shape_trigger_dialog_key("zone_abc", {"id": "rule_1"}),
            "zone_abc:rule_1",
        )

    def test_dialog_key_different_shapes(self):
        self.assertNotEqual(
            shape_trigger_dialog_key("zone_a"),
            shape_trigger_dialog_key("zone_b"),
        )

    def test_rule_has_snapshot_action_when_present(self):
        self.assertTrue(
            rule_has_snapshot_action({"actions": [{"type": "snapshot", "save_dir": "captures/x"}]})
        )

    def test_rule_has_snapshot_action_when_absent(self):
        self.assertFalse(rule_has_snapshot_action({"actions": []}))
        self.assertFalse(rule_has_snapshot_action({}))

    def test_build_rule_snapshot_actions_off(self):
        self.assertEqual(
            build_rule_snapshot_actions(
                take_screenshot=False,
                motion_watch_settings={"save_dir": "captures/x"},
            ),
            [],
        )

    def test_build_rule_snapshot_actions_on_uses_motion_watch_settings(self):
        actions = build_rule_actions(
            take_screenshot=True,
            motion_watch_settings={
                "include_overlays": False,
                "save_dir": "captures/custom",
                "resize_w": 960,
            },
            run_script=False,
            script_path="",
            script_runner="python",
            script_args="",
            script_timeout_sec=30,
        )
        self.assertEqual(len(actions), 1)
        self.assertEqual(actions[0]["type"], "snapshot")
        self.assertFalse(actions[0]["include_overlays"])
        self.assertEqual(actions[0]["save_dir"], "captures/custom")
        self.assertEqual(actions[0]["resize_w"], 960)

    def test_default_trigger_mode_is_auto_path(self):
        options = trigger_mode_options_for_kind("zone")
        self.assertEqual(options[0][1], DEFAULT_TRIGGER_MODE)
        self.assertEqual(DEFAULT_TRIGGER_MODE, "auto_path")

    def test_fresh_rule_payload_defaults_auto_path_not_any_interaction(self):
        east_path = [{"x": 0.05, "y": 0.5}, {"x": 0.95, "y": 0.48}]
        built = build_fresh_shape_rule_conditions(
            motion_path=east_path,
            derived_trigger="zone_enter",
        )
        self.assertEqual(built["trigger_mode"], "auto_path")
        self.assertEqual(built["trigger"], "zone_enter")
        self.assertNotIn("any_interaction", built["conditions"])
        self.assertIn("motion_path", built["conditions"])
        self.assertEqual(len(built["actions"]), 1)
        self.assertEqual(built["actions"][0]["type"], "snapshot")

    def test_fresh_rule_any_interaction_sets_flag(self):
        built = build_fresh_shape_rule_conditions(trigger_mode="any_interaction", motion_path=[])
        self.assertTrue(built["conditions"].get("any_interaction"))
        self.assertEqual(built["trigger"], "any_interaction")

    def test_effective_trigger_from_mode_path_match(self):
        self.assertEqual(
            effective_trigger_from_mode(
                mode="path_match",
                shape_kind="zone",
                derived_trigger="zone_enter",
                has_path=True,
            ),
            "path_match",
        )

    def test_trigger_options_for_tag_shape(self):
        values = [v for _, v in trigger_mode_options_for_kind("tag")]
        self.assertIn("near_tag", values)

    def test_rule_has_script_action(self):
        self.assertTrue(
            rule_has_script_action({"actions": [{"type": "script", "path": "x.py"}]})
        )
        self.assertFalse(rule_has_script_action({"actions": [{"type": "snapshot"}]}))

    def test_script_action_from_settings_disabled(self):
        self.assertIsNone(
            script_action_from_settings(
                enabled=False,
                path="x.py",
                runner="python",
                args_text="",
                timeout_sec=30,
            )
        )


class EventSourceDescriptionTests(unittest.TestCase):
    def test_motion_mode_description(self):
        text = event_source_description(motion_enabled=True, detection_enabled=False, backend_status="Off")
        self.assertIn("Motion boxes", text)
        self.assertIn("overlay", text.lower())

    def test_detection_mode_description(self):
        text = event_source_description(motion_enabled=False, detection_enabled=True, backend_status="On")
        self.assertIn("Object detection", text)
        self.assertIn("YOLO", text)
        self.assertIn("On", text)

    def test_dual_source_description(self):
        text = event_source_description(motion_enabled=True, detection_enabled=True, backend_status="On")
        self.assertIn("Dual source", text)
        self.assertIn("class/color filters", text.lower())


class EventSourcePayloadHelperTests(unittest.TestCase):
    def test_build_motion_only_conditions(self):
        cond = build_event_source_conditions(motion_enabled=True, detection_enabled=False)
        self.assertEqual(cond["tracker_namespace"], MOTION_BOX_NAMESPACE)
        self.assertFalse(cond["require_detection"])
        self.assertNotIn("tracker_namespaces", cond)

    def test_build_detection_only_conditions(self):
        cond = build_event_source_conditions(
            motion_enabled=False,
            detection_enabled=True,
            classes=["person"],
            min_confidence=0.6,
        )
        self.assertEqual(cond["tracker_namespace"], BACKEND_SORT_NAMESPACE)
        self.assertTrue(cond["require_detection"])
        self.assertEqual(cond["classes"], ["person"])
        self.assertEqual(cond["min_confidence"], 0.6)

    def test_build_dual_source_conditions(self):
        cond = build_event_source_conditions(
            motion_enabled=True,
            detection_enabled=True,
            classes=["car"],
            min_confidence=0.5,
        )
        self.assertEqual(cond["tracker_namespaces"], [MOTION_BOX_NAMESPACE, BACKEND_SORT_NAMESPACE])
        self.assertFalse(cond["require_detection"])
        self.assertEqual(cond["classes"], ["car"])

    def test_build_requires_at_least_one_source(self):
        with self.assertRaises(ValueError):
            build_event_source_conditions(motion_enabled=False, detection_enabled=False)

    def test_parse_motion_only_legacy(self):
        motion, detection = parse_event_source_flags(
            {"require_detection": False, "tracker_namespace": MOTION_BOX_NAMESPACE}
        )
        self.assertTrue(motion)
        self.assertFalse(detection)

    def test_parse_detection_only_legacy(self):
        motion, detection = parse_event_source_flags(
            {"require_detection": True, "tracker_namespace": BACKEND_SORT_NAMESPACE}
        )
        self.assertFalse(motion)
        self.assertTrue(detection)

    def test_parse_dual_source_namespaces(self):
        motion, detection = parse_event_source_flags(
            {"tracker_namespaces": [MOTION_BOX_NAMESPACE, BACKEND_SORT_NAMESPACE]}
        )
        self.assertTrue(motion)
        self.assertTrue(detection)


class PathToleranceSliderTests(unittest.TestCase):
    def test_slider_round_trip_default(self):
        tol = path_tolerance_from_slider(path_tolerance_slider_value(0.20))
        self.assertAlmostEqual(tol, 0.20, places=2)

    def test_slider_clamps_extremes(self):
        self.assertEqual(path_tolerance_from_slider(1), 0.02)
        self.assertEqual(path_tolerance_from_slider(99), 0.5)
        self.assertEqual(path_tolerance_slider_value(0.01), 2)
        self.assertEqual(path_tolerance_slider_value(0.99), 50)


if __name__ == "__main__":
    unittest.main()
