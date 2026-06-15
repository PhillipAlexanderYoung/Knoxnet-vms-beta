"""Stage 2–4 tests: track_event rule matching, cooldowns, extended conditions."""

from __future__ import annotations

import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

from core.automation.conditions import (
    ANY_INTERACTION_EVENTS,
    BACKEND_SORT_NAMESPACE,
    MOTION_BOX_NAMESPACE,
    allowed_tracker_namespaces,
    bucket_rgb_color,
    build_eval_context,
    compute_path_direction_gate,
    local_motion_matches_counter_rule,
    match_motion_path,
    matches_rule,
    matches_track_event,
    class_filter_applies_to_namespace,
    color_filter_applies_to_namespace,
    tracker_namespace_matches,
)
from core.automation.engine import AutomationEngine, AutomationEvent
from core.automation.state import AutomationState
from core.automation.track_state import TrackSceneEngine
from desktop.widgets.shape_trigger_preview import DEFAULT_COUNTER_MODE, counter_pill_config_from_rule
from desktop.utils.event_rules_api import shapes_to_api_payload
from desktop.utils.shape_trigger_helpers import build_rule_actions


def _fresh_drawn_path_zone_rule(*, trigger: str = "zone_enter", direction_path=None, **cond_overrides):
    east_path = direction_path or [{"x": 0.05, "y": 0.5}, {"x": 0.95, "y": 0.48}]
    conditions = {
        "require_detection": False,
        "show_counter": DEFAULT_COUNTER_MODE,
        "motion_path": east_path,
        "motion_path_space": "frame",
        "derived_trigger": trigger,
        "cooldown_sec": 1.0,
        "cooldown_per_track": True,
    }
    conditions.update(cond_overrides)
    return {
        "id": "rule_fresh",
        "name": "East cars",
        "enabled": True,
        "camera_id": "cam1",
        "trigger": trigger,
        "shape_id": "zone_1",
        "conditions": conditions,
        "actions": [{"type": "snapshot"}],
    }


def _track_event_payload(**overrides):
    base = {
        "event_type": "zone_enter",
        "camera_id": "cam1",
        "tracker_namespace": BACKEND_SORT_NAMESPACE,
        "track_id": 7,
        "class": "car",
        "confidence": 0.92,
        "bbox": {"x": 100, "y": 100, "w": 80, "h": 60},
        "centroid_norm": {"x": 0.5, "y": 0.5},
        "shape_id": "zone_1",
        "shape_name": "Zone 1",
        "timestamp": datetime.now(tz=timezone.utc).isoformat(),
    }
    base.update(overrides)
    return base


class TrackEventRuleMatchingTests(unittest.TestCase):
    def test_zone_enter_class_filter(self):
        rule = {
            "id": "r1",
            "enabled": True,
            "trigger": "zone_enter",
            "shape_id": "zone_1",
            "conditions": {"classes": ["car"], "min_confidence": 0.5},
        }
        ctx = build_eval_context("track_event", "cam1", _track_event_payload())
        ok, details = matches_rule(rule=rule, ctx=ctx)
        self.assertTrue(ok)
        self.assertEqual(details.get("track_id"), 7)

    def test_class_mismatch(self):
        rule = {
            "id": "r1",
            "enabled": True,
            "trigger": "zone_enter",
            "conditions": {"classes": ["person"]},
        }
        ctx = build_eval_context("track_event", "cam1", _track_event_payload())
        ok, _ = matches_rule(rule=rule, ctx=ctx)
        self.assertFalse(ok)

    def test_dwell_min_condition(self):
        rule = {
            "id": "r1",
            "enabled": True,
            "trigger": "dwell_met",
            "conditions": {"dwell_min": 1.0},
        }
        ctx = build_eval_context("track_event", "cam1", _track_event_payload(event_type="dwell_met", dwell_sec=1.2))
        ok, _ = matches_track_event(rule=rule, ctx=ctx)
        self.assertTrue(ok)

        ctx2 = build_eval_context("track_event", "cam1", _track_event_payload(event_type="dwell_met", dwell_sec=0.5))
        ok2, _ = matches_track_event(rule=rule, ctx=ctx2)
        self.assertFalse(ok2)

    def test_direction_condition(self):
        rule = {
            "id": "r1",
            "enabled": True,
            "trigger": "line_cross",
            "conditions": {"direction": "left_to_right"},
        }
        ctx = build_eval_context(
            "track_event",
            "cam1",
            _track_event_payload(event_type="line_cross", direction="left_to_right", shape_id="line_1"),
        )
        ok, _ = matches_track_event(rule=rule, ctx=ctx)
        self.assertTrue(ok)

        ctx2 = build_eval_context(
            "track_event",
            "cam1",
            _track_event_payload(event_type="line_cross", direction="right_to_left", shape_id="line_1"),
        )
        ok2, _ = matches_track_event(rule=rule, ctx=ctx2)
        self.assertFalse(ok2)

    def test_require_detection_false_skips_class_filter(self):
        rule = {
            "id": "r_motion",
            "enabled": True,
            "trigger": "zone_enter",
            "conditions": {"require_detection": False, "classes": ["person"]},
        }
        ctx = build_eval_context(
            "track_event",
            "cam1",
            _track_event_payload(**{"class": "car"}),
        )
        ok, _ = matches_track_event(rule=rule, ctx=ctx)
        self.assertTrue(ok)

        rule_det = {
            "id": "r_det",
            "enabled": True,
            "trigger": "zone_enter",
            "conditions": {"require_detection": True, "classes": ["person"]},
        }
        ok2, details2 = matches_track_event(rule=rule_det, ctx=ctx)
        self.assertFalse(ok2)
        self.assertEqual(details2.get("reason"), "object_filter")

    def test_motion_only_color_filter_when_color_present(self):
        rule = {
            "id": "r_motion_color",
            "enabled": True,
            "trigger": "zone_enter",
            "conditions": {"require_detection": False, "color": "white"},
        }
        ctx_match = build_eval_context(
            "track_event",
            "cam1",
            _track_event_payload(dominant_color="white"),
        )
        ok, _ = matches_track_event(rule=rule, ctx=ctx_match)
        self.assertTrue(ok)

        ctx_miss = build_eval_context(
            "track_event",
            "cam1",
            _track_event_payload(dominant_color="red"),
        )
        ok2, details2 = matches_track_event(rule=rule, ctx=ctx_miss)
        self.assertFalse(ok2)
        self.assertEqual(details2.get("reason"), "color")


class ExtendedConditionTests(unittest.TestCase):
    def test_color_condition_match(self):
        rule = {
            "id": "r_color",
            "enabled": True,
            "trigger": "zone_enter",
            "conditions": {"color": "white"},
        }
        ctx = build_eval_context(
            "track_event",
            "cam1",
            _track_event_payload(dominant_color="white"),
        )
        ok, _ = matches_track_event(rule=rule, ctx=ctx)
        self.assertTrue(ok)

        ctx2 = build_eval_context(
            "track_event",
            "cam1",
            _track_event_payload(dominant_color="red"),
        )
        ok2, details2 = matches_track_event(rule=rule, ctx=ctx2)
        self.assertFalse(ok2)
        self.assertEqual(details2.get("reason"), "color")

    def test_count_min_max(self):
        rule_min = {
            "id": "r_count_min",
            "enabled": True,
            "trigger": "zone_enter",
            "conditions": {"count_min": 2, "classes": ["car"]},
        }
        ctx_low = build_eval_context(
            "track_event",
            "cam1",
            _track_event_payload(zone_track_count=1, zone_track_counts={"car": 1}),
        )
        ok, details = matches_track_event(rule=rule_min, ctx=ctx_low)
        self.assertFalse(ok)
        self.assertEqual(details.get("reason"), "count")

        ctx_ok = build_eval_context(
            "track_event",
            "cam1",
            _track_event_payload(zone_track_count=2, zone_track_counts={"car": 2}),
        )
        ok2, _ = matches_track_event(rule=rule_min, ctx=ctx_ok)
        self.assertTrue(ok2)

        rule_max = {
            "id": "r_count_max",
            "enabled": True,
            "trigger": "zone_enter",
            "conditions": {"count_max": 1},
        }
        ctx_high = build_eval_context(
            "track_event",
            "cam1",
            _track_event_payload(zone_track_count=3),
        )
        ok3, _ = matches_track_event(rule=rule_max, ctx=ctx_high)
        self.assertFalse(ok3)

    def test_schedule_time_window(self):
        rule = {
            "id": "r_sched",
            "enabled": True,
            "trigger": "zone_enter",
            "conditions": {"time_window": {"start": "09:00", "end": "17:00", "days": [0, 1, 2, 3, 4]}},
        }
        monday_noon = datetime(2026, 6, 15, 12, 0, tzinfo=timezone.utc)  # Monday
        ctx = build_eval_context(
            "track_event",
            "cam1",
            _track_event_payload(timestamp=monday_noon.isoformat()),
        )
        ok, _ = matches_track_event(rule=rule, ctx=ctx)
        self.assertTrue(ok)

        monday_night = datetime(2026, 6, 15, 22, 0, tzinfo=timezone.utc)
        ctx2 = build_eval_context(
            "track_event",
            "cam1",
            _track_event_payload(timestamp=monday_night.isoformat()),
        )
        ok2, details2 = matches_track_event(rule=rule, ctx=ctx2)
        self.assertFalse(ok2)
        self.assertEqual(details2.get("reason"), "time_window")

        saturday_noon = datetime(2026, 6, 20, 12, 0, tzinfo=timezone.utc)  # Saturday
        ctx3 = build_eval_context(
            "track_event",
            "cam1",
            _track_event_payload(timestamp=saturday_noon.isoformat()),
        )
        ok3, details3 = matches_track_event(rule=rule, ctx=ctx3)
        self.assertFalse(ok3)
        self.assertEqual(details3.get("reason"), "time_window")

    def test_rejects_non_backend_sort_namespace(self):
        rule = {
            "id": "r_ns",
            "enabled": True,
            "trigger": "zone_enter",
            "conditions": {},
        }
        ctx = build_eval_context(
            "track_event",
            "cam1",
            _track_event_payload(tracker_namespace="motion_contour"),
        )
        ok, details = matches_track_event(rule=rule, ctx=ctx)
        self.assertFalse(ok)
        self.assertEqual(details.get("reason"), "tracker_namespace")

    def test_motion_mode_accepts_motion_box_namespace(self):
        rule = {
            "id": "r_motion_ns",
            "enabled": True,
            "trigger": "zone_enter",
            "shape_id": "zone_1",
            "conditions": {"require_detection": False, "tracker_namespace": MOTION_BOX_NAMESPACE},
        }
        ctx = build_eval_context(
            "track_event",
            "cam1",
            _track_event_payload(tracker_namespace=MOTION_BOX_NAMESPACE, **{"class": "object"}),
        )
        ok, details = matches_track_event(rule=rule, ctx=ctx)
        self.assertTrue(ok, details)

    def test_detection_mode_rejects_motion_box_namespace(self):
        rule = {
            "id": "r_det_ns",
            "enabled": True,
            "trigger": "zone_enter",
            "shape_id": "zone_1",
            "conditions": {"require_detection": True, "tracker_namespace": BACKEND_SORT_NAMESPACE},
        }
        ctx = build_eval_context(
            "track_event",
            "cam1",
            _track_event_payload(tracker_namespace=MOTION_BOX_NAMESPACE),
        )
        ok, details = matches_track_event(rule=rule, ctx=ctx)
        self.assertFalse(ok)
        self.assertEqual(details.get("reason"), "tracker_namespace")

    def test_motion_mode_rule_without_object_track_payload(self):
        """Motion-mode zone_enter with motion_box track should match without YOLO class."""
        rule = {
            "id": "r_motion_only",
            "enabled": True,
            "trigger": "zone_enter",
            "shape_id": "zone_1",
            "conditions": {
                "require_detection": False,
                "tracker_namespace": MOTION_BOX_NAMESPACE,
            },
        }
        ctx = build_eval_context(
            "track_event",
            "cam1",
            _track_event_payload(
                tracker_namespace=MOTION_BOX_NAMESPACE,
                **{"class": "object", "confidence": 0.1},
            ),
        )
        ok, details = matches_track_event(rule=rule, ctx=ctx)
        self.assertTrue(ok, details)

    def test_tracker_namespace_matches_helper(self):
        motion_cond = {"require_detection": False}
        self.assertTrue(tracker_namespace_matches(MOTION_BOX_NAMESPACE, motion_cond)[0])
        self.assertTrue(tracker_namespace_matches(BACKEND_SORT_NAMESPACE, motion_cond)[0])
        self.assertFalse(tracker_namespace_matches("motion_contour", motion_cond)[0])

    def test_dual_source_accepts_either_namespace(self):
        rule = {
            "id": "r_dual",
            "enabled": True,
            "trigger": "zone_enter",
            "shape_id": "zone_1",
            "conditions": {
                "tracker_namespaces": [MOTION_BOX_NAMESPACE, BACKEND_SORT_NAMESPACE],
                "require_detection": False,
            },
        }
        for ns in (MOTION_BOX_NAMESPACE, BACKEND_SORT_NAMESPACE):
            ctx = build_eval_context(
                "track_event",
                "cam1",
                _track_event_payload(tracker_namespace=ns),
            )
            ok, details = matches_track_event(rule=rule, ctx=ctx)
            self.assertTrue(ok, details)

    def test_dual_source_class_filter_skips_motion_box(self):
        rule = {
            "id": "r_dual_class",
            "enabled": True,
            "trigger": "zone_enter",
            "shape_id": "zone_1",
            "conditions": {
                "tracker_namespaces": [MOTION_BOX_NAMESPACE, BACKEND_SORT_NAMESPACE],
                "require_detection": False,
                "classes": ["person"],
            },
        }
        motion_ctx = build_eval_context(
            "track_event",
            "cam1",
            _track_event_payload(
                tracker_namespace=MOTION_BOX_NAMESPACE,
                **{"class": "object", "confidence": 0.1},
            ),
        )
        ok, details = matches_track_event(rule=rule, ctx=motion_ctx)
        self.assertTrue(ok, details)

        det_ctx = build_eval_context(
            "track_event",
            "cam1",
            _track_event_payload(
                tracker_namespace=BACKEND_SORT_NAMESPACE,
                **{"class": "car", "confidence": 0.9},
            ),
        )
        ok, details = matches_track_event(rule=rule, ctx=det_ctx)
        self.assertFalse(ok)
        self.assertEqual(details.get("reason"), "object_filter")

    def test_allowed_tracker_namespaces_dual(self):
        allowed = allowed_tracker_namespaces(
            {"tracker_namespaces": [MOTION_BOX_NAMESPACE, BACKEND_SORT_NAMESPACE]}
        )
        self.assertEqual(allowed, {MOTION_BOX_NAMESPACE, BACKEND_SORT_NAMESPACE})

    def test_object_filters_apply_only_to_backend_in_dual_mode(self):
        dual = {"tracker_namespaces": [MOTION_BOX_NAMESPACE, BACKEND_SORT_NAMESPACE]}
        self.assertFalse(class_filter_applies_to_namespace(MOTION_BOX_NAMESPACE, dual))
        self.assertTrue(class_filter_applies_to_namespace(BACKEND_SORT_NAMESPACE, dual))
        self.assertFalse(color_filter_applies_to_namespace(MOTION_BOX_NAMESPACE, dual))
        self.assertTrue(color_filter_applies_to_namespace(BACKEND_SORT_NAMESPACE, dual))

    def test_counter_pill_config_carries_require_detection(self):
        rule = {
            "id": "rule_motion_pill",
            "name": "Motion pill",
            "enabled": True,
            "camera_id": "cam1",
            "trigger": "zone_enter",
            "shape_id": "zone_1",
            "conditions": {"show_counter": "always", "require_detection": False},
            "actions": [{"type": "snapshot"}],
        }
        cfg = counter_pill_config_from_rule(rule, shape={"kind": "zone", "pts": []})
        self.assertIsNotNone(cfg)
        assert cfg is not None
        self.assertFalse(cfg.get("require_detection"))

    def test_bucket_rgb_color_white(self):
        self.assertEqual(bucket_rgb_color(240.0, 240.0, 240.0), "white")


class MotionPathMatchingTests(unittest.TestCase):
    def test_match_motion_path_direction_ok(self):
        motion_path = [{"x": 0.1, "y": 0.5}, {"x": 0.9, "y": 0.5}]
        history = [{"x": 0.15, "y": 0.52}, {"x": 0.85, "y": 0.48}]
        ok, details = match_motion_path(motion_path=motion_path, track_history=history, tolerance=0.15)
        self.assertTrue(ok)
        self.assertEqual(details.get("reason"), "path_distance_ok")

    def test_match_motion_path_direction_mismatch(self):
        motion_path = [{"x": 0.1, "y": 0.5}, {"x": 0.9, "y": 0.5}]
        history = [{"x": 0.5, "y": 0.1}, {"x": 0.5, "y": 0.9}]
        ok, details = match_motion_path(motion_path=motion_path, track_history=history, tolerance=0.15)
        self.assertFalse(ok)
        self.assertEqual(details.get("reason"), "direction_mismatch")

    def test_path_match_trigger_with_motion_path(self):
        rule = {
            "id": "r_path",
            "enabled": True,
            "trigger": "path_match",
            "shape_id": "zone_1",
            "conditions": {
                "motion_path": [{"x": 0.1, "y": 0.5}, {"x": 0.6, "y": 0.5}],
                "path_match_tolerance": 0.12,
                "derived_trigger": "zone_enter",
            },
        }
        ctx = build_eval_context(
            "track_event",
            "cam1",
            _track_event_payload(
                event_type="zone_enter",
                centroid_history=[{"x": 0.12, "y": 0.51}, {"x": 0.55, "y": 0.49}],
            ),
        )
        ok, _ = matches_track_event(rule=rule, ctx=ctx)
        self.assertTrue(ok)

        ctx_bad = build_eval_context(
            "track_event",
            "cam1",
            _track_event_payload(
                event_type="zone_enter",
                centroid_history=[{"x": 0.5, "y": 0.1}, {"x": 0.5, "y": 0.9}],
            ),
        )
        ok2, details2 = matches_track_event(rule=rule, ctx=ctx_bad)
        self.assertFalse(ok2)
        self.assertEqual(details2.get("reason"), "direction_mismatch")

    def test_auto_path_rule_direction_filter_on_motion_path(self):
        rule = {
            "id": "r_auto",
            "enabled": True,
            "trigger": "zone_enter",
            "shape_id": "zone_1",
            "conditions": {
                "motion_path": [{"x": 0.2, "y": 0.5}, {"x": 0.8, "y": 0.5}],
                "path_match_tolerance": 0.15,
            },
        }
        ctx = build_eval_context(
            "track_event",
            "cam1",
            _track_event_payload(
                centroid_history=[{"x": 0.25, "y": 0.5}, {"x": 0.75, "y": 0.5}],
            ),
        )
        ok, _ = matches_track_event(rule=rule, ctx=ctx)
        self.assertTrue(ok)

        ctx_wrong = build_eval_context(
            "track_event",
            "cam1",
            _track_event_payload(
                centroid_history=[{"x": 0.75, "y": 0.5}, {"x": 0.25, "y": 0.5}],
            ),
        )
        ok2, details2 = matches_track_event(rule=rule, ctx=ctx_wrong)
        self.assertFalse(ok2)
        self.assertEqual(details2.get("reason"), "direction_mismatch")

    def test_eastbound_track_rejects_westbound_drawn_path(self):
        westbound_path = [{"x": 0.9, "y": 0.5}, {"x": 0.1, "y": 0.5}]
        eastbound_history = [{"x": 0.2, "y": 0.5}, {"x": 0.3, "y": 0.5}, {"x": 0.8, "y": 0.5}]
        ok, details = match_motion_path(
            motion_path=westbound_path,
            track_history=eastbound_history,
            tolerance=0.15,
        )
        self.assertFalse(ok)
        self.assertEqual(details.get("reason"), "direction_mismatch")

    def test_westbound_track_rejects_eastbound_drawn_path(self):
        eastbound_path = [{"x": 0.1, "y": 0.5}, {"x": 0.9, "y": 0.5}]
        westbound_history = [{"x": 0.8, "y": 0.5}, {"x": 0.7, "y": 0.5}, {"x": 0.2, "y": 0.5}]
        ok, details = match_motion_path(
            motion_path=eastbound_path,
            track_history=westbound_history,
            tolerance=0.15,
        )
        self.assertFalse(ok)
        self.assertEqual(details.get("reason"), "direction_mismatch")

    def test_single_history_with_reliable_centroid_matches(self):
        eastbound_path = [{"x": 0.1, "y": 0.5}, {"x": 0.9, "y": 0.5}]
        ok, details = match_motion_path(
            motion_path=eastbound_path,
            track_history=[{"x": 0.25, "y": 0.5}],
            centroid=(0.75, 0.5),
            tolerance=0.15,
        )
        self.assertTrue(ok)
        self.assertEqual(details.get("reason"), "path_distance_ok")

    def test_single_history_without_centroid_rejects(self):
        westbound_path = [{"x": 0.9, "y": 0.5}, {"x": 0.1, "y": 0.5}]
        ok, details = match_motion_path(
            motion_path=westbound_path,
            track_history=[{"x": 0.85, "y": 0.5}],
            tolerance=0.15,
        )
        self.assertFalse(ok)
        self.assertEqual(details.get("reason"), "insufficient_track_history")

    def test_single_history_same_centroid_rejects(self):
        """Do not fabricate movement from a duplicate centroid."""
        eastbound_path = [{"x": 0.1, "y": 0.5}, {"x": 0.9, "y": 0.5}]
        ok, details = match_motion_path(
            motion_path=eastbound_path,
            track_history=[{"x": 0.5, "y": 0.5}],
            centroid=(0.5, 0.5),
            tolerance=0.15,
        )
        self.assertFalse(ok)
        self.assertIn(details.get("reason"), ("insufficient_track_history", "insufficient_movement"))

    def test_path_start_to_end_direction_not_first_segment(self):
        """Multi-click paths may start with a short vertical segment; overall direction should win."""
        user_path = [
            {"x": 0.10, "y": 0.45},
            {"x": 0.10, "y": 0.55},
            {"x": 0.90, "y": 0.55},
        ]
        eastbound_history = [{"x": 0.20, "y": 0.54}, {"x": 0.50, "y": 0.55}, {"x": 0.75, "y": 0.55}]
        ok, details = match_motion_path(
            motion_path=user_path,
            track_history=eastbound_history,
            tolerance=0.15,
        )
        self.assertTrue(ok)
        self.assertEqual(details.get("reason"), "path_distance_ok")

    def test_full_scene_path_dwell_alignment(self):
        zone_shape = {
            "kind": "zone",
            "pts": [
                {"x": 0.4, "y": 0.4},
                {"x": 0.6, "y": 0.4},
                {"x": 0.5, "y": 0.6},
            ],
        }
        dwell_path = [
            {"x": 0.2, "y": 0.5},
            {"x": 0.45, "y": 0.5},
            {"x": 0.5, "y": 0.5},
            {"x": 0.55, "y": 0.5},
            {"x": 0.8, "y": 0.5},
        ]
        track_in_zone = [
            {"x": 0.42, "y": 0.5},
            {"x": 0.48, "y": 0.5},
            {"x": 0.52, "y": 0.5},
            {"x": 0.58, "y": 0.5},
        ]
        ok, details = match_motion_path(
            motion_path=dwell_path,
            track_history=track_in_zone,
            tolerance=0.15,
            dwell_min=1.0,
            shape=zone_shape,
            dwell_sec=1.2,
        )
        self.assertTrue(ok)
        self.assertGreater(details.get("path_inside_ratio", 0), 0.3)

    def test_speed_mismatch_rejects_wrong_pace(self):
        slow_path = [
            {"x": 0.1, "y": 0.5},
            {"x": 0.12, "y": 0.5},
            {"x": 0.14, "y": 0.5},
            {"x": 0.16, "y": 0.5},
            {"x": 0.18, "y": 0.5},
        ]
        fast_history = [{"x": 0.1, "y": 0.5}, {"x": 0.5, "y": 0.5}, {"x": 0.9, "y": 0.5}]
        ok, details = match_motion_path(
            motion_path=slow_path,
            track_history=fast_history,
            tolerance=0.2,
            direction_only=True,
            check_speed=True,
        )
        self.assertFalse(ok)
        self.assertEqual(details.get("reason"), "speed_mismatch")

    def test_speed_check_disabled_by_default_for_zone_enter(self):
        """Full-scene path + short in-zone track history should still match on direction."""
        slow_path = [{"x": 0.05, "y": 0.5}, {"x": 0.95, "y": 0.5}]
        in_zone_history = [{"x": 0.42, "y": 0.5}, {"x": 0.48, "y": 0.5}]
        rule = {
            "id": "r_speed_off",
            "enabled": True,
            "trigger": "zone_enter",
            "shape_id": "zone_1",
            "conditions": {"motion_path": slow_path},
        }
        ctx = build_eval_context(
            "track_event",
            "cam1",
            _track_event_payload(centroid_history=in_zone_history),
        )
        ok, details = matches_track_event(rule=rule, ctx=ctx)
        self.assertTrue(ok, details)

    def test_roughly_east_track_matches_east_drawn_path(self):
        east_path = [{"x": 0.1, "y": 0.5}, {"x": 0.9, "y": 0.5}]
        angled_history = [{"x": 0.15, "y": 0.48}, {"x": 0.45, "y": 0.54}, {"x": 0.75, "y": 0.56}]
        ok, details = match_motion_path(
            motion_path=east_path,
            track_history=angled_history,
            tolerance=0.20,
            direction_only=True,
        )
        self.assertTrue(ok)
        self.assertEqual(details.get("reason"), "direction_ok")

    def test_curved_path_similar_track_matches(self):
        curved_path = [
            {"x": 0.1, "y": 0.5},
            {"x": 0.35, "y": 0.45},
            {"x": 0.65, "y": 0.55},
            {"x": 0.9, "y": 0.5},
        ]
        similar_track = [
            {"x": 0.12, "y": 0.51},
            {"x": 0.38, "y": 0.46},
            {"x": 0.62, "y": 0.54},
            {"x": 0.88, "y": 0.49},
        ]
        ok, details = match_motion_path(
            motion_path=curved_path,
            track_history=similar_track,
            tolerance=0.20,
        )
        self.assertTrue(ok)
        self.assertEqual(details.get("reason"), "path_distance_ok")

    def test_eastbound_path_one_crossing_one_trigger(self):
        """Drawn east path + eastbound track matches; westbound rejects; engine fires once."""
        east_path = [{"x": 0.1, "y": 0.5}, {"x": 0.9, "y": 0.5}]
        east_rule = {
            "id": "east",
            "enabled": True,
            "trigger": "zone_enter",
            "shape_id": "zone_1",
            "camera_id": "cam1",
            "conditions": {
                "motion_path": east_path,
                "cooldown_sec": 5,
                "cooldown_per_track": True,
            },
            "actions": [{"type": "snapshot"}],
        }
        payload = _track_event_payload(
            centroid_history=[{"x": 0.15, "y": 0.5}, {"x": 0.35, "y": 0.5}, {"x": 0.75, "y": 0.5}],
            centroid_norm={"x": 0.75, "y": 0.5},
        )
        ctx = build_eval_context("track_event", "cam1", payload)
        east_ok, _ = matches_track_event(rule=east_rule, ctx=ctx)
        self.assertTrue(east_ok)

        west_rule = dict(east_rule)
        west_rule["id"] = "west"
        west_rule["conditions"] = dict(east_rule["conditions"])
        west_rule["conditions"]["motion_path"] = [{"x": 0.9, "y": 0.5}, {"x": 0.1, "y": 0.5}]
        west_ok, details = matches_track_event(rule=west_rule, ctx=ctx)
        self.assertFalse(west_ok)
        self.assertEqual(details.get("reason"), "direction_mismatch")

        db = MagicMock()
        db.list_rules.return_value = [east_rule]
        engine = AutomationEngine(db_manager=db, stream_server=MagicMock(), dry_run=False)
        handler = MagicMock()
        engine.action_handlers["snapshot"] = handler
        evt = AutomationEvent(
            id="e1",
            kind="track_event",
            camera_id="cam1",
            created_at=datetime.now().isoformat(),
            payload=payload,
        )
        for _ in range(14):
            engine._process(evt)
        self.assertEqual(handler.call_count, 1)

    def test_opposite_direction_rules_on_same_zone(self):
        west_rule = {
            "id": "west",
            "enabled": True,
            "trigger": "zone_enter",
            "shape_id": "zone_1",
            "conditions": {
                "motion_path": [{"x": 0.9, "y": 0.5}, {"x": 0.1, "y": 0.5}],
            },
        }
        east_rule = {
            "id": "east",
            "enabled": True,
            "trigger": "zone_enter",
            "shape_id": "zone_1",
            "conditions": {
                "motion_path": [{"x": 0.1, "y": 0.5}, {"x": 0.9, "y": 0.5}],
            },
        }
        payload = _track_event_payload(
            centroid_history=[
                {"x": 0.15, "y": 0.5},
                {"x": 0.35, "y": 0.5},
                {"x": 0.75, "y": 0.5},
            ],
        )
        ctx = build_eval_context("track_event", "cam1", payload)
        west_ok, _ = matches_track_event(rule=west_rule, ctx=ctx)
        east_ok, _ = matches_track_event(rule=east_rule, ctx=ctx)
        self.assertFalse(west_ok)
        self.assertTrue(east_ok)


    def test_new_rule_default_show_counter_always(self):
        """New shape rules should enable counter pills by default."""
        self.assertEqual(DEFAULT_COUNTER_MODE, "always")
        rule = {
            "id": "r_new",
            "enabled": True,
            "shape_id": "zone_1",
            "conditions": {"show_counter": DEFAULT_COUNTER_MODE},
        }
        cfg = counter_pill_config_from_rule(rule, shape={"kind": "zone", "pts": []})
        self.assertIsNotNone(cfg)
        self.assertEqual(cfg.get("mode"), "always")

    def test_full_scene_drawn_path_zone_enter_similar_vehicle(self):
        """Full-scene eastbound path + similar vehicle track + zone_enter should match."""
        zone_shape = {
            "kind": "zone",
            "id": "zone_1",
            "points": [
                {"x": 0.4, "y": 0.35},
                {"x": 0.6, "y": 0.35},
                {"x": 0.6, "y": 0.65},
                {"x": 0.4, "y": 0.65},
            ],
        }
        east_path = [
            {"x": 0.05, "y": 0.5},
            {"x": 0.45, "y": 0.52},
            {"x": 0.95, "y": 0.48},
        ]
        rule = {
            "id": "r_scene",
            "enabled": True,
            "trigger": "zone_enter",
            "shape_id": "zone_1",
            "conditions": {
                "motion_path": east_path,
                "motion_path_space": "frame",
                "require_detection": False,
            },
        }
        ctx = build_eval_context(
            "track_event",
            "cam1",
            _track_event_payload(
                centroid_history=[
                    {"x": 0.20, "y": 0.49},
                    {"x": 0.38, "y": 0.51},
                    {"x": 0.52, "y": 0.50},
                ],
                centroid_norm={"x": 0.52, "y": 0.50},
            ),
        )
        ok, details = matches_track_event(rule=rule, ctx=ctx, shape=zone_shape)
        self.assertTrue(ok, details)

    def test_short_history_without_movement_rejects_both_directions(self):
        """Centroid-only events must not match auto path rules (avoids dual east/west counts)."""
        zone_shape = {
            "kind": "zone",
            "id": "zone_1",
            "points": [
                {"x": 0.4, "y": 0.35},
                {"x": 0.6, "y": 0.35},
                {"x": 0.6, "y": 0.65},
                {"x": 0.4, "y": 0.65},
            ],
        }
        east_path = [{"x": 0.1, "y": 0.5}, {"x": 0.9, "y": 0.5}]
        west_path = [{"x": 0.9, "y": 0.5}, {"x": 0.1, "y": 0.5}]
        for path in (east_path, west_path):
            ok, details = match_motion_path(
                motion_path=path,
                track_history=[],
                centroid=(0.55, 0.5),
                direction_only=True,
                shape=zone_shape,
            )
            self.assertFalse(ok, details)
            self.assertEqual(details.get("reason"), "insufficient_track_history")

        rule = {
            "id": "r_short_hist",
            "enabled": True,
            "trigger": "zone_enter",
            "shape_id": "zone_1",
            "conditions": {
                "motion_path": east_path,
                "motion_path_space": "frame",
                "require_detection": False,
            },
        }
        ctx = build_eval_context(
            "track_event",
            "cam1",
            _track_event_payload(
                centroid_history=[],
                centroid_norm={"x": 0.55, "y": 0.5},
            ),
        )
        ok_rule, details = matches_track_event(rule=rule, ctx=ctx, shape=zone_shape)
        self.assertFalse(ok_rule)
        self.assertEqual(details.get("reason"), "insufficient_track_history")

    def test_path_direction_gate_rejects_opposite_traversal(self):
        zone_shape = {
            "kind": "zone",
            "id": "zone_1",
            "points": [
                {"x": 0.4, "y": 0.35},
                {"x": 0.6, "y": 0.35},
                {"x": 0.6, "y": 0.65},
                {"x": 0.4, "y": 0.65},
            ],
        }
        east_path = [{"x": 0.1, "y": 0.5}, {"x": 0.9, "y": 0.5}]
        gate = compute_path_direction_gate(east_path, zone_shape)
        self.assertIsNotNone(gate)
        self.assertEqual(gate.get("entry_side"), "left")
        self.assertEqual(gate.get("exit_side"), "right")

        east_history = [{"x": 0.15, "y": 0.5}, {"x": 0.35, "y": 0.5}, {"x": 0.75, "y": 0.5}]
        ok, details = match_motion_path(
            motion_path=east_path,
            track_history=east_history,
            direction_only=True,
            shape=zone_shape,
            direction_gate=gate,
        )
        self.assertTrue(ok, details)

        west_history = [{"x": 0.85, "y": 0.5}, {"x": 0.65, "y": 0.5}, {"x": 0.25, "y": 0.5}]
        ok2, details2 = match_motion_path(
            motion_path=east_path,
            track_history=west_history,
            direction_only=True,
            shape=zone_shape,
            direction_gate=gate,
        )
        self.assertFalse(ok2)
        self.assertIn(details2.get("reason"), ("direction_mismatch", "direction_gate_mismatch"))

    def test_east_track_east_rule_west_rule_rejects(self):
        zone_shape = {
            "kind": "zone",
            "id": "zone_1",
            "points": [
                {"x": 0.4, "y": 0.35},
                {"x": 0.6, "y": 0.35},
                {"x": 0.6, "y": 0.65},
                {"x": 0.4, "y": 0.65},
            ],
        }
        east_path = [{"x": 0.1, "y": 0.5}, {"x": 0.9, "y": 0.5}]
        west_path = [{"x": 0.9, "y": 0.5}, {"x": 0.1, "y": 0.5}]
        payload = _track_event_payload(
            centroid_history=[{"x": 0.15, "y": 0.5}, {"x": 0.35, "y": 0.5}, {"x": 0.75, "y": 0.5}],
            centroid_norm={"x": 0.75, "y": 0.5},
        )
        ctx = build_eval_context("track_event", "cam1", payload)
        east_rule = {
            "id": "east",
            "enabled": True,
            "trigger": "zone_enter",
            "shape_id": "zone_1",
            "conditions": {
                "motion_path": east_path,
                "path_direction_gate": compute_path_direction_gate(east_path, zone_shape),
            },
        }
        west_rule = {
            "id": "west",
            "enabled": True,
            "trigger": "zone_enter",
            "shape_id": "zone_1",
            "conditions": {
                "motion_path": west_path,
                "path_direction_gate": compute_path_direction_gate(west_path, zone_shape),
            },
        }
        east_ok, _ = matches_track_event(rule=east_rule, ctx=ctx, shape=zone_shape)
        west_ok, west_details = matches_track_event(rule=west_rule, ctx=ctx, shape=zone_shape)
        self.assertTrue(east_ok)
        self.assertFalse(west_ok)
        self.assertIn(west_details.get("reason"), ("direction_mismatch", "direction_gate_mismatch"))

    def test_opposite_rules_one_traversal_one_handler(self):
        zone_shape = {
            "kind": "zone",
            "id": "zone_1",
            "points": [
                {"x": 0.4, "y": 0.35},
                {"x": 0.6, "y": 0.35},
                {"x": 0.6, "y": 0.65},
                {"x": 0.4, "y": 0.65},
            ],
        }
        east_path = [{"x": 0.1, "y": 0.5}, {"x": 0.9, "y": 0.5}]
        west_path = [{"x": 0.9, "y": 0.5}, {"x": 0.1, "y": 0.5}]
        payload = _track_event_payload(
            centroid_history=[{"x": 0.15, "y": 0.5}, {"x": 0.35, "y": 0.5}, {"x": 0.75, "y": 0.5}],
            centroid_norm={"x": 0.75, "y": 0.5},
        )
        east_rule = {
            "id": "east",
            "enabled": True,
            "trigger": "zone_enter",
            "shape_id": "zone_1",
            "camera_id": "cam1",
            "conditions": {
                "motion_path": east_path,
                "path_direction_gate": compute_path_direction_gate(east_path, zone_shape),
                "cooldown_sec": 5,
                "cooldown_per_track": True,
            },
            "actions": [{"type": "snapshot"}],
        }
        west_rule = dict(east_rule)
        west_rule["id"] = "west"
        west_rule["conditions"] = dict(east_rule["conditions"])
        west_rule["conditions"]["motion_path"] = west_path
        west_rule["conditions"]["path_direction_gate"] = compute_path_direction_gate(west_path, zone_shape)

        db = MagicMock()
        db.list_rules.return_value = [east_rule, west_rule]
        engine = AutomationEngine(db_manager=db, stream_server=MagicMock(), dry_run=False)
        handler = MagicMock()
        engine.action_handlers["snapshot"] = handler
        evt = AutomationEvent(
            id="e1",
            kind="track_event",
            camera_id="cam1",
            created_at=datetime.now().isoformat(),
            payload=payload,
        )
        for _ in range(14):
            engine._process(evt)
        self.assertEqual(handler.call_count, 1)

    def test_local_motion_counter_respects_direction(self):
        zone_shape = {
            "kind": "zone",
            "id": "zone_1",
            "pts": [
                {"x": 0.4, "y": 0.35},
                {"x": 0.6, "y": 0.35},
                {"x": 0.6, "y": 0.65},
                {"x": 0.4, "y": 0.65},
            ],
        }
        east_path = [{"x": 0.1, "y": 0.5}, {"x": 0.9, "y": 0.5}]
        west_path = [{"x": 0.9, "y": 0.5}, {"x": 0.1, "y": 0.5}]
        east_cond = {
            "motion_path": east_path,
            "path_direction_gate": compute_path_direction_gate(east_path, zone_shape),
            "require_detection": False,
        }
        west_cond = {
            "motion_path": west_path,
            "path_direction_gate": compute_path_direction_gate(west_path, zone_shape),
            "require_detection": False,
        }
        point = {"x": 0.55, "y": 0.5}
        east_history = [{"x": 0.45, "y": 0.5}, {"x": 0.52, "y": 0.5}]
        west_history = [{"x": 0.65, "y": 0.5}, {"x": 0.45, "y": 0.5}]

        self.assertTrue(
            local_motion_matches_counter_rule(
                point=point,
                centroid_history=east_history,
                conditions=east_cond,
                shape=zone_shape,
                trigger="zone_enter",
            )
        )
        self.assertFalse(
            local_motion_matches_counter_rule(
                point=point,
                centroid_history=east_history,
                conditions=west_cond,
                shape=zone_shape,
                trigger="zone_enter",
            )
        )
        self.assertTrue(
            local_motion_matches_counter_rule(
                point=point,
                centroid_history=west_history,
                conditions=west_cond,
                shape=zone_shape,
                trigger="zone_enter",
            )
        )
        self.assertFalse(
            local_motion_matches_counter_rule(
                point=point,
                centroid_history=[],
                conditions=east_cond,
                shape=zone_shape,
                trigger="zone_enter",
            )
        )

    def test_min_confidence_ignored_without_classes(self):
        """Empty class list should not block motion tracks via min_confidence alone."""
        rule = {
            "id": "r_motion_conf",
            "enabled": True,
            "trigger": "zone_enter",
            "shape_id": "zone_1",
            "conditions": {
                "require_detection": True,
                "classes": [],
                "min_confidence": 0.5,
            },
        }
        ctx = build_eval_context(
            "track_event",
            "cam1",
            _track_event_payload(confidence=0.1, **{"class": "object"}),
        )
        ok, details = matches_track_event(rule=rule, ctx=ctx)
        self.assertTrue(ok, details)


class TrackSceneColorCountTests(unittest.TestCase):
    def test_rolling_color_in_event_payload(self):
        engine = TrackSceneEngine(hysteresis_frames=2, dwell_sec=1.0)
        shapes = {
            "zones": [
                {
                    "id": "zone_1",
                    "name": "Zone 1",
                    "enabled": True,
                    "points": [
                        {"x": 0.2, "y": 0.2},
                        {"x": 0.8, "y": 0.2},
                        {"x": 0.8, "y": 0.8},
                        {"x": 0.2, "y": 0.8},
                    ],
                }
            ]
        }
        track = {
            "id": 1,
            "class": "car",
            "confidence": 0.9,
            "bbox": {"x": 280, "y": 200, "w": 80, "h": 60},
            "center": {"nx": 0.5, "ny": 0.5},
        }
        with patch("core.automation.track_state.estimate_dominant_color_from_bgr", return_value="white"):
            all_events = []
            for i in range(3):
                all_events.extend(
                    engine.update(
                        camera_id="cam1",
                        tracks=[track],
                        shapes=shapes,
                        frame_w=640,
                        frame_h=480,
                        frame_bgr=object(),
                        now=1000.0 + i * 0.05,
                    )
                )
        enter = [e for e in all_events if e.get("event_type") == "zone_enter"]
        self.assertTrue(enter)
        self.assertEqual(enter[-1].get("dominant_color"), "white")

    def test_zone_track_count_on_enter(self):
        engine = TrackSceneEngine(hysteresis_frames=2, dwell_sec=1.0)
        shapes = {
            "zones": [
                {
                    "id": "zone_1",
                    "name": "Zone 1",
                    "enabled": True,
                    "points": [
                        {"x": 0.2, "y": 0.2},
                        {"x": 0.8, "y": 0.2},
                        {"x": 0.8, "y": 0.8},
                        {"x": 0.2, "y": 0.8},
                    ],
                }
            ]
        }
        t1 = {
            "id": 1,
            "class": "car",
            "confidence": 0.9,
            "bbox": {"x": 280, "y": 200, "w": 80, "h": 60},
            "center": {"nx": 0.35, "ny": 0.5},
        }
        t2 = {
            "id": 2,
            "class": "car",
            "confidence": 0.9,
            "bbox": {"x": 380, "y": 200, "w": 80, "h": 60},
            "center": {"nx": 0.65, "ny": 0.5},
        }
        for i in range(2):
            engine.update(
                camera_id="cam1",
                tracks=[t1],
                shapes=shapes,
                frame_w=640,
                frame_h=480,
                now=1000.0 + i * 0.05,
            )
        events = []
        for i in range(2):
            events = engine.update(
                camera_id="cam1",
                tracks=[t1, t2],
                shapes=shapes,
                frame_w=640,
                frame_h=480,
                now=1001.0 + i * 0.05,
            )
        enter = [e for e in events if e.get("event_type") == "zone_enter" and e.get("track_id") == 2]
        self.assertTrue(enter)
        self.assertGreaterEqual(int(enter[-1].get("zone_track_count", 0)), 2)
        counts = enter[-1].get("zone_track_counts") or {}
        self.assertGreaterEqual(int(counts.get("car", 0)), 2)


class TrackCooldownTests(unittest.TestCase):
    def test_per_track_cooldown_independent(self):
        state = AutomationState()
        rule_id = "r1"
        cam = "cam1"
        cooldown = 5.0
        self.assertFalse(state.is_in_track_cooldown(rule_id=rule_id, camera_id=cam, track_id=1, cooldown_sec=cooldown))
        state.mark_triggered(rule_id=rule_id, camera_id=cam, track_id=1)
        self.assertTrue(state.is_in_track_cooldown(rule_id=rule_id, camera_id=cam, track_id=1, cooldown_sec=cooldown))
        self.assertFalse(state.is_in_track_cooldown(rule_id=rule_id, camera_id=cam, track_id=2, cooldown_sec=cooldown))


class AutomationEngineTrackEventTests(unittest.TestCase):
    def test_engine_fires_snapshot_for_matching_rule(self):
        rules = [
            {
                "id": "snap1",
                "name": "Zone enter snap",
                "enabled": True,
                "trigger": "zone_enter",
                "shape_id": "zone_1",
                "camera_id": "cam1",
                "conditions": {"classes": ["car"], "cooldown_sec": 10, "cooldown_per_track": True},
                "actions": [{"type": "snapshot", "include_overlays": False}],
            }
        ]
        db = MagicMock()
        db.list_rules.return_value = rules

        engine = AutomationEngine(db_manager=db, stream_server=MagicMock(), dry_run=False)
        handler = MagicMock()
        engine.action_handlers["snapshot"] = handler

        evt = AutomationEvent(
            id="e1",
            kind="track_event",
            camera_id="cam1",
            created_at=datetime.now().isoformat(),
            payload=_track_event_payload(),
        )
        engine._process(evt)
        handler.assert_called_once()

        # Same track within cooldown should not fire again
        handler.reset_mock()
        engine._process(evt)
        handler.assert_not_called()

        # Different track should fire
        handler.reset_mock()
        evt2 = AutomationEvent(
            id="e2",
            kind="track_event",
            camera_id="cam1",
            created_at=datetime.now().isoformat(),
            payload=_track_event_payload(track_id=99),
        )
        engine._process(evt2)
        handler.assert_called_once()


class SimultaneousVehicleCooldownTests(unittest.TestCase):
    def test_two_vehicles_fire_independently(self):
        rules = [
            {
                "id": "snap1",
                "name": "Zone enter snap",
                "enabled": True,
                "trigger": "zone_enter",
                "shape_id": "zone_1",
                "camera_id": "cam1",
                "conditions": {"classes": ["car"], "cooldown_sec": 30, "cooldown_per_track": True},
                "actions": [{"type": "snapshot"}],
            }
        ]
        db = MagicMock()
        db.list_rules.return_value = rules
        engine = AutomationEngine(db_manager=db, stream_server=MagicMock(), dry_run=False)
        handler = MagicMock()
        engine.action_handlers["snapshot"] = handler

        for tid in (10, 20):
            evt = AutomationEvent(
                id=f"e{tid}",
                kind="track_event",
                camera_id="cam1",
                created_at=datetime.now().isoformat(),
                payload=_track_event_payload(track_id=tid),
            )
            engine._process(evt)

        self.assertEqual(handler.call_count, 2)


class LostTrackNewIdTests(unittest.TestCase):
    def test_new_track_id_not_blocked_by_prior_track_cooldown(self):
        state = AutomationState()
        rule_id = "r1"
        cam = "cam1"
        cooldown = 60.0
        state.mark_triggered(rule_id=rule_id, camera_id=cam, track_id=7)
        self.assertTrue(
            state.is_in_track_cooldown(rule_id=rule_id, camera_id=cam, track_id=7, cooldown_sec=cooldown)
        )
        self.assertFalse(
            state.is_in_track_cooldown(rule_id=rule_id, camera_id=cam, track_id=42, cooldown_sec=cooldown)
        )

    def test_engine_fires_for_reassigned_track_id(self):
        rules = [
            {
                "id": "snap1",
                "enabled": True,
                "trigger": "zone_enter",
                "camera_id": "cam1",
                "conditions": {"cooldown_sec": 60, "cooldown_per_track": True},
                "actions": [{"type": "snapshot"}],
            }
        ]
        db = MagicMock()
        db.list_rules.return_value = rules
        engine = AutomationEngine(db_manager=db, stream_server=MagicMock(), dry_run=False)
        handler = MagicMock()
        engine.action_handlers["snapshot"] = handler

        engine._process(
            AutomationEvent(
                id="e1",
                kind="track_event",
                camera_id="cam1",
                created_at=datetime.now().isoformat(),
                payload=_track_event_payload(track_id=7),
            )
        )
        handler.assert_called_once()

        handler.reset_mock()
        engine._process(
            AutomationEvent(
                id="e2",
                kind="track_event",
                camera_id="cam1",
                created_at=datetime.now().isoformat(),
                payload=_track_event_payload(track_id=99),
            )
        )
        handler.assert_called_once()


class BoundaryJitterDedupeTests(unittest.TestCase):
    def test_track_scene_hysteresis_avoids_spurious_reenter(self):
        engine = TrackSceneEngine(hysteresis_frames=3, dwell_sec=1.0)
        shapes = {
            "zones": [
                {
                    "id": "zone_1",
                    "name": "Zone 1",
                    "enabled": True,
                    "points": [
                        {"x": 0.2, "y": 0.2},
                        {"x": 0.8, "y": 0.2},
                        {"x": 0.8, "y": 0.8},
                        {"x": 0.2, "y": 0.8},
                    ],
                }
            ]
        }
        inside = {
            "id": 1,
            "class": "car",
            "confidence": 0.9,
            "bbox": {"x": 280, "y": 200, "w": 80, "h": 60},
            "center": {"nx": 0.5, "ny": 0.5},
        }
        outside = {
            "id": 1,
            "class": "car",
            "confidence": 0.9,
            "bbox": {"x": 10, "y": 200, "w": 80, "h": 60},
            "center": {"nx": 0.05, "ny": 0.5},
        }
        enter_count = 0
        for i in range(3):
            for e in engine.update(
                camera_id="cam1",
                tracks=[inside],
                shapes=shapes,
                frame_w=640,
                frame_h=480,
                now=1000.0 + i * 0.04,
            ):
                if e.get("event_type") == "zone_enter":
                    enter_count += 1
        self.assertEqual(enter_count, 1)

        # Brief boundary jitter should not produce another zone_enter
        engine.update(
            camera_id="cam1", tracks=[outside], shapes=shapes, frame_w=640, frame_h=480, now=1001.0
        )
        jitter_events = engine.update(
            camera_id="cam1", tracks=[inside], shapes=shapes, frame_w=640, frame_h=480, now=1001.04
        )
        reenters = [e for e in jitter_events if e.get("event_type") == "zone_enter"]
        self.assertFalse(reenters)

    def test_engine_dedupes_identical_track_events(self):
        rules = [
            {
                "id": "snap1",
                "enabled": True,
                "trigger": "zone_enter",
                "shape_id": "zone_1",
                "camera_id": "cam1",
                "conditions": {"cooldown_sec": 0},
                "actions": [{"type": "snapshot"}],
            }
        ]
        db = MagicMock()
        db.list_rules.return_value = rules
        engine = AutomationEngine(db_manager=db, stream_server=MagicMock(), dry_run=False)
        handler = MagicMock()
        engine.action_handlers["snapshot"] = handler

        payload = _track_event_payload()
        for i in range(3):
            engine._process(
                AutomationEvent(
                    id=f"e{i}",
                    kind="track_event",
                    camera_id="cam1",
                    created_at=datetime.now().isoformat(),
                    payload=payload,
                )
            )
        self.assertEqual(handler.call_count, 1)


class EngineSingleTraversalTests(unittest.TestCase):
    def test_one_zone_enter_fires_rule_once(self):
        rules = [
            {
                "id": "count_rule",
                "enabled": True,
                "trigger": "zone_enter",
                "shape_id": "zone_1",
                "camera_id": "cam1",
                "conditions": {
                    "motion_path": [{"x": 0.1, "y": 0.5}, {"x": 0.9, "y": 0.5}],
                    "cooldown_sec": 5,
                    "cooldown_per_track": True,
                },
                "actions": [{"type": "snapshot"}],
            }
        ]
        db = MagicMock()
        db.list_rules.return_value = rules
        engine = AutomationEngine(db_manager=db, stream_server=MagicMock(), dry_run=False)
        handler = MagicMock()
        engine.action_handlers["snapshot"] = handler

        payload = _track_event_payload(
            centroid_history=[{"x": 0.2, "y": 0.5}, {"x": 0.8, "y": 0.5}],
        )
        evt = AutomationEvent(
            id="e1",
            kind="track_event",
            camera_id="cam1",
            created_at=datetime.now().isoformat(),
            payload=payload,
        )
        for _ in range(14):
            engine._process(evt)
        self.assertEqual(handler.call_count, 1)


class HeadlessTrackEventEvaluationTests(unittest.TestCase):
    def test_submit_track_event_invokes_snapshot_handler(self):
        rules = [
            {
                "id": "headless1",
                "name": "Headless snap",
                "enabled": True,
                "trigger": "zone_enter",
                "shape_id": "zone_1",
                "camera_id": "cam1",
                "conditions": {"classes": ["car"], "cooldown_sec": 0},
                "actions": [{"type": "snapshot", "include_overlays": False}],
            }
        ]
        db = MagicMock()
        db.list_rules.return_value = rules
        stream = MagicMock()
        engine = AutomationEngine(db_manager=db, stream_server=stream, dry_run=False)

        captured = []

        def _snap_handler(*, rule, ctx, details, action, event):
            captured.append(
                {
                    "rule_id": rule.get("id"),
                    "track_id": details.get("track_id"),
                    "event_type": details.get("event_type"),
                    "action_type": action.get("type"),
                }
            )

        engine.action_handlers["snapshot"] = _snap_handler

        evt = AutomationEvent(
            id="headless_evt",
            kind="track_event",
            camera_id="cam1",
            created_at=datetime.now().isoformat(),
            payload=_track_event_payload(event_type="zone_enter", shape_id="zone_1"),
        )
        engine._process(evt)

        self.assertEqual(len(captured), 1)
        self.assertEqual(captured[0]["rule_id"], "headless1")
        self.assertEqual(captured[0]["track_id"], 7)
        self.assertEqual(captured[0]["event_type"], "zone_enter")
        self.assertEqual(captured[0]["action_type"], "snapshot")


class SnapshotActionTests(unittest.TestCase):
    def test_snapshot_writes_and_ingests(self):
        from core.automation.actions.snapshot import SnapshotAction

        fake_jpg = b"\xff\xd8\xff\xe0" + b"0" * 100
        stream = MagicMock()
        stream.get_frame.return_value = fake_jpg
        stream.get_camera_shapes.return_value = {"zones": [], "lines": [], "tags": []}

        idx = MagicMock()
        idx.ingest.return_value = {"event_id": "ev1", "file_path": "/tmp/x.jpg"}

        with tempfile.TemporaryDirectory() as tmp:
            action = SnapshotAction(db_manager=MagicMock(), stream_server=stream, event_index=idx)
            rule = {"id": "r1", "name": "Test"}
            ctx = build_eval_context("track_event", "cam1", _track_event_payload())
            details = {"event_type": "zone_enter", "track_id": 7}
            act_cfg = {"type": "snapshot", "include_overlays": False, "save_dir": tmp}

            with patch("core.automation.actions.snapshot._decode_jpeg", return_value=None):
                result = action.capture(rule=rule, ctx=ctx, details=details, action=act_cfg, event=MagicMock(kind="track_event"))

            self.assertIsNotNone(result)
            idx.ingest.assert_called_once()
            written = list(Path(tmp).rglob("*.jpg"))
            self.assertTrue(written)

    def test_snapshot_ingest_includes_dominant_color(self):
        from core.automation.actions.snapshot import SnapshotAction

        fake_jpg = b"\xff\xd8\xff\xe0" + b"0" * 100
        stream = MagicMock()
        stream.get_frame.return_value = fake_jpg
        stream.get_camera_shapes.return_value = {"zones": [], "lines": [], "tags": []}

        idx = MagicMock()
        idx.ingest.return_value = {"event_id": "ev1", "file_path": "/tmp/x.jpg"}

        with tempfile.TemporaryDirectory() as tmp:
            action = SnapshotAction(db_manager=MagicMock(), stream_server=stream, event_index=idx)
            rule = {"id": "r1", "name": "White car"}
            ctx = build_eval_context(
                "track_event",
                "cam1",
                _track_event_payload(dominant_color="white"),
            )
            details = {"event_type": "zone_enter", "track_id": 7}
            act_cfg = {"type": "snapshot", "include_overlays": False, "save_dir": tmp}

            with patch("core.automation.actions.snapshot._decode_jpeg", return_value=None):
                action.capture(
                    rule=rule,
                    ctx=ctx,
                    details=details,
                    action=act_cfg,
                    event=MagicMock(kind="track_event"),
                )

            ingest_payload = idx.ingest.call_args[0][0]
            self.assertEqual(ingest_payload.get("dominant_color"), "white")

    def test_snapshot_emits_new_capture_for_terminal(self):
        from core.automation.actions.snapshot import SnapshotAction

        fake_jpg = b"\xff\xd8\xff\xe0" + b"0" * 100
        stream = MagicMock()
        stream.get_frame.return_value = fake_jpg
        stream.get_camera_shapes.return_value = {"zones": [], "lines": [], "tags": []}

        idx = MagicMock()
        idx.ingest.return_value = {
            "event_id": "ev_terminal",
            "file_path": "/tmp/cam1_watch_99.jpg",
            "thumb_path": "/tmp/thumb.jpg",
            "captured_ts": 1700000099,
            "shape_name": "Zone A",
            "trigger_type": "zone_enter",
        }
        sio = MagicMock()

        with tempfile.TemporaryDirectory() as tmp:
            action = SnapshotAction(
                db_manager=MagicMock(),
                stream_server=stream,
                event_index=idx,
                socketio=sio,
            )
            rule = {"id": "r1", "name": "Snap rule"}
            ctx = build_eval_context("track_event", "cam1", _track_event_payload())
            details = {"event_type": "zone_enter", "track_id": 7}
            act_cfg = {"type": "snapshot", "include_overlays": False, "save_dir": tmp}

            with patch("core.automation.actions.snapshot._decode_jpeg", return_value=None):
                action.capture(
                    rule=rule,
                    ctx=ctx,
                    details=details,
                    action=act_cfg,
                    event=MagicMock(kind="track_event"),
                )

        self.assertTrue(sio.emit.called)
        emit_calls = [c for c in sio.emit.call_args_list if c.args and c.args[0] == "new_capture"]
        self.assertTrue(emit_calls)
        payload = emit_calls[0].args[1]
        self.assertEqual(payload.get("camera_id"), "cam1")
        self.assertEqual(payload.get("event_id"), "ev_terminal")

    def test_snapshot_emits_new_capture_when_ingest_unavailable(self):
        from core.automation.actions.snapshot import SnapshotAction

        fake_jpg = b"\xff\xd8\xff\xe0" + b"0" * 100
        stream = MagicMock()
        stream.get_frame.return_value = fake_jpg
        stream.get_camera_shapes.return_value = {"zones": [], "lines": [], "tags": []}
        sio = MagicMock()

        with tempfile.TemporaryDirectory() as tmp:
            action = SnapshotAction(
                db_manager=MagicMock(),
                stream_server=stream,
                event_index=None,
                socketio=sio,
            )
            action._index = lambda: None
            rule = {"id": "r1", "name": "Snap rule"}
            ctx = build_eval_context("track_event", "cam1", _track_event_payload())
            details = {"event_type": "zone_enter", "track_id": 7, "shape_name": "Zone A"}
            act_cfg = {"type": "snapshot", "include_overlays": False, "save_dir": tmp}

            with patch("core.automation.actions.snapshot._decode_jpeg", return_value=None):
                result = action.capture(
                    rule=rule,
                    ctx=ctx,
                    details=details,
                    action=act_cfg,
                    event=MagicMock(kind="track_event"),
                )

        self.assertIsInstance(result, dict)
        self.assertFalse(result.get("ingested", True))
        emit_calls = [c for c in sio.emit.call_args_list if c.args and c.args[0] == "new_capture"]
        self.assertTrue(emit_calls, "expected new_capture even when ingest unavailable")
        payload = emit_calls[0].args[1]
        self.assertEqual(payload.get("camera_id"), "cam1")
        self.assertTrue(str(payload.get("file_path") or "").endswith(".jpg"))

    def test_no_snapshot_action_when_toggle_off(self):
        actions = build_rule_actions(
            take_screenshot=False,
            motion_watch_settings={"save_dir": "captures/motion_watch"},
            run_script=False,
            script_path="",
            script_runner="python",
            script_args="",
            script_timeout_sec=30,
        )
        self.assertEqual(actions, [])


class SnapshotFailureTests(unittest.TestCase):
    def test_snapshot_returns_capture_failed_when_no_frame(self):
        from core.automation.actions.snapshot import SnapshotAction

        stream = MagicMock()
        stream.get_frame.return_value = None
        stream.active_streams = {"cam1": {"last_frame": None}}
        stream.start_stream_sync = MagicMock(return_value=True)
        action = SnapshotAction(db_manager=MagicMock(), stream_server=stream, socketio=MagicMock())
        rule = {"id": "r_fail", "name": "Fail rule"}
        ctx = build_eval_context("track_event", "cam1", _track_event_payload())
        details = {"event_type": "zone_enter", "track_id": 7}
        act_cfg = {"type": "snapshot", "include_overlays": False, "save_dir": "captures/motion_watch"}

        result = action.capture(
            rule=rule,
            ctx=ctx,
            details=details,
            action=act_cfg,
            event=MagicMock(kind="track_event"),
        )
        self.assertIsInstance(result, dict)
        self.assertTrue(result.get("capture_failed"))
        self.assertEqual(result.get("reason"), "no_frame_available")

    def test_engine_automation_alert_includes_capture_error_on_snapshot_failure(self):
        handler = MagicMock(
            return_value={
                "capture_failed": True,
                "reason": "no_frame_available",
                "camera_id": "cam1",
                "rule_id": "r_cap",
            }
        )
        engine = AutomationEngine(db_manager=MagicMock(), socketio=MagicMock(), dry_run=False)
        engine.action_handlers["snapshot"] = handler
        rule = {
            "id": "r_cap",
            "name": "Cap rule",
            "enabled": True,
            "camera_id": "cam1",
            "trigger": "zone_enter",
            "shape_id": "zone_1",
            "conditions": {"require_detection": False},
            "actions": [{"type": "snapshot"}],
        }
        ctx = build_eval_context("track_event", "cam1", _track_event_payload())
        evt = AutomationEvent(
            id="evt_cap_fail",
            kind="track_event",
            camera_id="cam1",
            created_at=datetime.now(tz=timezone.utc).isoformat(),
            payload=_track_event_payload(),
        )
        details = {"event_type": "zone_enter", "shape_id": "zone_1"}
        engine._on_rule_triggered(rule, details, evt, ctx)
        emit_calls = [
            c
            for c in engine.socketio.emit.call_args_list
            if c.args and c.args[0] == "automation_alert"
        ]
        self.assertTrue(emit_calls)
        alert_payload = emit_calls[-1].args[1]
        err = alert_payload.get("capture_error")
        self.assertIsInstance(err, dict)
        self.assertEqual(err.get("reason"), "no_frame_available")
        self.assertNotIn("capture", alert_payload)


class FreshShapeRulePayloadTests(unittest.TestCase):
    def test_engine_automation_alert_includes_capture_after_snapshot(self):
        handler = MagicMock(
            return_value={
                "event_id": "ev1",
                "file_path": "/tmp/cam1_watch_1.jpg",
                "thumb_path": "/tmp/thumb.jpg",
                "captured_ts": 1700000000,
                "shape_name": "Driveway",
                "trigger_type": "zone_enter",
                "camera_id": "cam1",
            }
        )
        engine = AutomationEngine(db_manager=MagicMock(), socketio=MagicMock(), dry_run=False)
        engine.action_handlers["snapshot"] = handler
        rule = {
            "id": "r_cap",
            "name": "Cap rule",
            "enabled": True,
            "camera_id": "cam1",
            "trigger": "zone_enter",
            "shape_id": "zone_1",
            "conditions": {"require_detection": False},
            "actions": [{"type": "snapshot"}],
        }
        ctx = build_eval_context("track_event", "cam1", _track_event_payload())
        evt = AutomationEvent(
            id="evt_cap",
            kind="track_event",
            camera_id="cam1",
            created_at=datetime.now(tz=timezone.utc).isoformat(),
            payload=_track_event_payload(),
        )
        details = {"event_type": "zone_enter", "shape_id": "zone_1"}
        engine._on_rule_triggered(rule, details, evt, ctx)
        handler.assert_called_once()
        emit_calls = [
            c
            for c in engine.socketio.emit.call_args_list
            if c.args and c.args[0] == "automation_alert"
        ]
        self.assertTrue(emit_calls)
        alert_payload = emit_calls[-1].args[1]
        capture = alert_payload.get("capture")
        self.assertIsInstance(capture, dict)
        self.assertEqual(capture.get("camera_id"), "cam1")
        self.assertTrue(str(capture.get("file_path") or "").endswith(".jpg"))

    def test_fresh_rule_has_counter_and_zone_enter_trigger(self):
        rule = _fresh_drawn_path_zone_rule()
        self.assertEqual(rule["trigger"], "zone_enter")
        self.assertEqual(rule["conditions"]["show_counter"], "always")
        self.assertNotIn("tracker_namespace", rule["conditions"])
        self.assertFalse(rule["conditions"]["require_detection"])
        cfg = counter_pill_config_from_rule(rule, shape={"kind": "zone", "pts": []})
        self.assertIsNotNone(cfg)

    def test_any_interaction_rule_matches_enter_and_exit(self):
        rule = {
            "id": "rule_any",
            "enabled": True,
            "trigger": "any_interaction",
            "shape_id": "zone_1",
            "conditions": {"any_interaction": True, "require_detection": False},
        }
        for event_type in ("zone_enter", "zone_exit"):
            ctx = build_eval_context(
                "track_event",
                "cam1",
                _track_event_payload(event_type=event_type),
            )
            ok, details = matches_track_event(rule=rule, ctx=ctx)
            self.assertTrue(ok, details)
        self.assertIn("zone_enter", ANY_INTERACTION_EVENTS)
        self.assertIn("zone_exit", ANY_INTERACTION_EVENTS)

    def test_any_interaction_skips_motion_path_filter(self):
        rule = {
            "id": "rule_any_path",
            "enabled": True,
            "trigger": "any_interaction",
            "shape_id": "zone_1",
            "conditions": {
                "any_interaction": True,
                "motion_path": [{"x": 0.1, "y": 0.5}, {"x": 0.9, "y": 0.5}],
            },
        }
        ctx = build_eval_context(
            "track_event",
            "cam1",
            _track_event_payload(
                event_type="zone_enter",
                centroid_history=[{"x": 0.5, "y": 0.1}, {"x": 0.5, "y": 0.9}],
            ),
        )
        ok, details = matches_track_event(rule=rule, ctx=ctx)
        self.assertTrue(ok, details)

    def test_explicit_zone_exit_rejects_zone_enter(self):
        rule = {
            "id": "rule_exit",
            "enabled": True,
            "camera_id": "cam1",
            "trigger": "zone_exit",
            "shape_id": "zone_1",
            "conditions": {"require_detection": False},
        }
        enter_ctx = build_eval_context(
            "track_event",
            "cam1",
            _track_event_payload(event_type="zone_enter"),
        )
        ok, _ = matches_track_event(rule=rule, ctx=enter_ctx)
        self.assertFalse(ok)
        exit_ctx = build_eval_context(
            "track_event",
            "cam1",
            _track_event_payload(event_type="zone_exit"),
        )
        ok, details = matches_track_event(rule=rule, ctx=exit_ctx)
        self.assertTrue(ok, details)


class EndToEndTraversalTests(unittest.TestCase):
    def test_synthetic_zone_traversal_fires_once(self):
        zone_shape = {
            "kind": "zone",
            "id": "zone_1",
            "points": [
                {"x": 0.4, "y": 0.35},
                {"x": 0.6, "y": 0.35},
                {"x": 0.6, "y": 0.65},
                {"x": 0.4, "y": 0.65},
            ],
        }
        engine_scene = TrackSceneEngine(hysteresis_frames=2, dwell_sec=1.0)
        shapes = {"zones": [dict(zone_shape, enabled=True)], "lines": [], "tags": []}
        rule = _fresh_drawn_path_zone_rule()
        rule["conditions"]["motion_path"] = [
            {"x": 0.05, "y": 0.5},
            {"x": 0.5, "y": 0.5},
            {"x": 0.95, "y": 0.5},
        ]

        db = MagicMock()
        db.list_rules.return_value = [rule]
        db.get_camera_shapes.return_value = shapes
        auto = AutomationEngine(db_manager=db, stream_server=MagicMock(), dry_run=False)
        handler = MagicMock()
        auto.action_handlers["snapshot"] = handler

        track = {
            "id": 42,
            "class": "car",
            "confidence": 0.9,
            "bbox": {"x": 50, "y": 200, "w": 80, "h": 60},
        }
        xs = [0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75]
        fired = 0
        for i, nx in enumerate(xs):
            track["center"] = {"nx": nx, "ny": 0.5}
            track["bbox"] = {"x": int(nx * 640), "y": 200, "w": 80, "h": 60}
            events = engine_scene.update(
                camera_id="cam1",
                tracks=[track],
                shapes=shapes,
                frame_w=640,
                frame_h=480,
                now=1000.0 + i * 0.05,
            )
            for ev in events:
                if ev.get("event_type") != "zone_enter":
                    continue
                auto._process(
                    AutomationEvent(
                        id=f"evt_{i}",
                        kind="track_event",
                        camera_id="cam1",
                        created_at=datetime.now().isoformat(),
                        payload=ev,
                    )
                )
                fired += 1
        self.assertEqual(handler.call_count, 1)
        self.assertGreaterEqual(fired, 1)

    def test_east_rule_rejects_westbound_traversal(self):
        zone_shape = {
            "kind": "zone",
            "id": "zone_1",
            "points": [
                {"x": 0.4, "y": 0.35},
                {"x": 0.6, "y": 0.35},
                {"x": 0.6, "y": 0.65},
                {"x": 0.4, "y": 0.65},
            ],
        }
        east_rule = _fresh_drawn_path_zone_rule()
        west_rule = _fresh_drawn_path_zone_rule(
            direction_path=[{"x": 0.95, "y": 0.5}, {"x": 0.05, "y": 0.5}],
        )
        west_rule["id"] = "rule_west"
        payload = _track_event_payload(
            centroid_history=[
                {"x": 0.75, "y": 0.5},
                {"x": 0.55, "y": 0.5},
                {"x": 0.35, "y": 0.5},
            ],
            centroid_norm={"x": 0.35, "y": 0.5},
        )
        ctx = build_eval_context("track_event", "cam1", payload)
        east_ok, _ = matches_track_event(rule=east_rule, ctx=ctx, shape=zone_shape)
        west_ok, _ = matches_track_event(rule=west_rule, ctx=ctx, shape=zone_shape)
        self.assertFalse(east_ok)
        self.assertTrue(west_ok)


class ShapeSyncPayloadTests(unittest.TestCase):
    def test_shapes_to_api_payload_normalizes_desktop_shapes(self):
        payload = shapes_to_api_payload(
            [
                {
                    "id": "z1",
                    "kind": "zone",
                    "label": "Road",
                    "pts": [{"x": 0.1, "y": 0.2}, {"x": 0.9, "y": 0.2}, {"x": 0.9, "y": 0.8}, {"x": 0.1, "y": 0.8}],
                },
                {
                    "id": "l1",
                    "kind": "line",
                    "p1": {"x": 0.2, "y": 0.5},
                    "p2": {"x": 0.8, "y": 0.5},
                },
                {
                    "id": "t1",
                    "kind": "tag",
                    "anchor": {"x": 0.5, "y": 0.5},
                },
            ]
        )
        self.assertEqual(len(payload["zones"]), 1)
        self.assertEqual(len(payload["lines"]), 1)
        self.assertEqual(len(payload["tags"]), 1)
        self.assertEqual(payload["zones"][0]["id"], "z1")
        self.assertGreaterEqual(len(payload["zones"][0]["points"]), 3)


class EventRuleCooldownDefaultTests(unittest.TestCase):
    def test_default_rule_cooldown_constant(self):
        from desktop.utils.event_rules_api import DEFAULT_RULE_COOLDOWN_SEC

        self.assertEqual(DEFAULT_RULE_COOLDOWN_SEC, 1.0)

    def test_cooldown_ms_round_trip(self):
        from desktop.utils.event_rules_api import cooldown_ms_from_sec, cooldown_sec_from_ms

        for ms in (250, 500, 1000, 1500):
            self.assertEqual(cooldown_ms_from_sec(cooldown_sec_from_ms(ms)), ms)

    def test_subsecond_cooldown_preserved_in_legacy_rule(self):
        from desktop.utils.event_rules_api import legacy_rule_from_motion_watch_settings

        rule = legacy_rule_from_motion_watch_settings("cam1", {"cooldown_ms": 500})
        self.assertEqual(rule["conditions"]["cooldown_sec"], 0.5)

    def test_motion_detection_defaults(self):
        from desktop.utils.event_rules_api import (
            DEFAULT_MOTION_MERGE_SIZE,
            DEFAULT_MOTION_SENSITIVITY,
            default_motion_detection_tuning,
        )

        tuning = default_motion_detection_tuning()
        self.assertEqual(tuning["sensitivity"], DEFAULT_MOTION_SENSITIVITY)
        self.assertEqual(tuning["merge_size"], DEFAULT_MOTION_MERGE_SIZE)
        self.assertEqual(DEFAULT_MOTION_SENSITIVITY, 70)
        self.assertEqual(DEFAULT_MOTION_MERGE_SIZE, 50)


class ExplicitSubsecondCooldownEngineTests(unittest.TestCase):
    @patch("core.automation.state.time.time")
    def test_engine_honors_one_second_cooldown_not_two(self, mock_time):
        mock_time.return_value = 1000.0
        rules = [
            {
                "id": "snap1",
                "enabled": True,
                "trigger": "zone_enter",
                "shape_id": "zone_1",
                "camera_id": "cam1",
                "conditions": {"cooldown_sec": 1.0, "cooldown_per_track": True},
                "actions": [{"type": "snapshot"}],
            }
        ]
        db = MagicMock()
        db.list_rules.return_value = rules
        engine = AutomationEngine(db_manager=db, stream_server=MagicMock(), dry_run=False)
        handler = MagicMock()
        engine.action_handlers["snapshot"] = handler

        payload = _track_event_payload()
        evt1 = AutomationEvent(
            id="e1",
            kind="track_event",
            camera_id="cam1",
            created_at=datetime.now().isoformat(),
            payload=payload,
        )
        engine._process(evt1)
        handler.assert_called_once()

        handler.reset_mock()
        mock_time.return_value = 1001.4
        evt2 = AutomationEvent(
            id="e2",
            kind="track_event",
            camera_id="cam1",
            created_at=datetime.now().isoformat(),
            payload=payload,
        )
        engine._process(evt2)
        handler.assert_called_once()

    @patch("core.automation.state.time.time")
    def test_half_second_cooldown_blocks_within_window(self, mock_time):
        mock_time.return_value = 2000.0
        rules = [
            {
                "id": "snap_half",
                "enabled": True,
                "trigger": "zone_enter",
                "shape_id": "zone_1",
                "camera_id": "cam1",
                "conditions": {"cooldown_sec": 0.5, "cooldown_per_track": True},
                "actions": [{"type": "snapshot"}],
            }
        ]
        db = MagicMock()
        db.list_rules.return_value = rules
        engine = AutomationEngine(db_manager=db, stream_server=MagicMock(), dry_run=False)
        handler = MagicMock()
        engine.action_handlers["snapshot"] = handler

        payload = _track_event_payload()
        evt = AutomationEvent(
            id="e1",
            kind="track_event",
            camera_id="cam1",
            created_at=datetime.now().isoformat(),
            payload=payload,
        )
        engine._process(evt)
        handler.assert_called_once()

        handler.reset_mock()
        mock_time.return_value = 2000.3
        engine._process(
            AutomationEvent(
                id="e2",
                kind="track_event",
                camera_id="cam1",
                created_at=datetime.now().isoformat(),
                payload=payload,
            )
        )
        handler.assert_not_called()


if __name__ == "__main__":
    unittest.main()
