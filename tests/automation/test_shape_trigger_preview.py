"""Unit tests for shape trigger preview helpers."""

from __future__ import annotations

import unittest

from desktop.widgets.shape_trigger_preview import (
    COUNTER_COMBINE_MODES,
    COUNTER_MODES,
    DEFAULT_COUNTER_MODE,
    DEFAULT_COUNTER_PILL_ANCHOR,
    MOTION_PATH_SPACE_FRAME,
    MOTION_PATH_SPACE_SHAPE,
    build_counter_pill_render_items,
    combine_trigger_counts,
    counter_pill_bbox_from_frame,
    counter_pill_config_from_rule,
    counter_pill_frame_coords,
    counter_pill_label_for_rule,
    default_counter_pill_anchor,
    fit_shape_preview,
    frame_path_to_shape_relative,
    ghost_entry_hover_lines,
    ghost_entry_label,
    motion_path_travel_t,
    motion_path_shape_ref_from_shape,
    normalize_counter_combine,
    normalize_counter_mode,
    normalize_frame_motion_path,
    normalize_motion_path_space,
    parse_counter_pill_anchor,
    preview_animation_params,
    prune_trigger_counts,
    resolve_counter_pill_anchor,
    resolve_counter_pill_label,
    resolve_motion_path_for_frame,
    rule_to_hover_ghost_entry,
    build_armed_rule_ghost_entries,
    rules_for_shape,
    upsert_event_rule_in_cache,
    shape_bounds,
    shape_relative_path_to_frame,
    spread_overlapping_pill_items,
)


from core.automation.conditions import BACKEND_SORT_NAMESPACE, MOTION_BOX_NAMESPACE


class ShapeTriggerPreviewTests(unittest.TestCase):
    def test_counter_mode_enum(self):
        self.assertEqual(COUNTER_MODES, ("off", "always", "on_trigger"))
        self.assertEqual(DEFAULT_COUNTER_MODE, "always")
        self.assertEqual(normalize_counter_mode("always"), "always")
        self.assertEqual(normalize_counter_mode("ON_TRIGGER"), "on_trigger")
        self.assertEqual(normalize_counter_mode("bogus"), "off")
        self.assertEqual(normalize_counter_mode(None), "off")

    def test_shape_bounds_zone(self):
        shape = {
            "kind": "zone",
            "pts": [
                {"x": 0.8, "y": 0.8},
                {"x": 0.82, "y": 0.8},
                {"x": 0.81, "y": 0.83},
            ],
        }
        min_x, min_y, max_x, max_y = shape_bounds(shape)
        self.assertAlmostEqual(min_x, 0.8)
        self.assertAlmostEqual(max_x, 0.82)
        self.assertAlmostEqual(min_y, 0.8)
        self.assertAlmostEqual(max_y, 0.83)

    def test_shape_bounds_line_has_perpendicular_pad(self):
        shape = {
            "kind": "line",
            "p1": {"x": 0.2, "y": 0.5},
            "p2": {"x": 0.8, "y": 0.5},
        }
        min_x, min_y, max_x, max_y = shape_bounds(shape)
        self.assertLess(min_y, 0.5)
        self.assertGreater(max_y, 0.5)
        self.assertAlmostEqual(min_x, 0.16)
        self.assertAlmostEqual(max_x, 0.84)

    def test_shape_bounds_tag_pads_anchor(self):
        shape = {"kind": "tag", "anchor": {"x": 0.4, "y": 0.6}}
        min_x, min_y, max_x, max_y = shape_bounds(shape)
        self.assertAlmostEqual(min_x, 0.36)
        self.assertAlmostEqual(max_x, 0.44)
        self.assertAlmostEqual(min_y, 0.56)
        self.assertAlmostEqual(max_y, 0.64)

    def test_default_counter_pill_anchor_offsets_line_and_tag_slots(self):
        line = {"kind": "line", "p1": {"x": 0.1, "y": 0.2}, "p2": {"x": 0.9, "y": 0.2}}
        tag = {"kind": "tag", "anchor": {"x": 0.5, "y": 0.5}}
        a0 = default_counter_pill_anchor(line, 0)
        a1 = default_counter_pill_anchor(line, 1)
        self.assertNotEqual(a0, a1)
        t0 = default_counter_pill_anchor(tag, 0)
        t1 = default_counter_pill_anchor(tag, 1)
        self.assertNotEqual(t0, t1)

    def test_counter_pill_anchor_roundtrip_line_and_tag(self):
        line = {
            "kind": "line",
            "p1": {"x": 0.15, "y": 0.35},
            "p2": {"x": 0.85, "y": 0.35},
        }
        tag = {"kind": "tag", "anchor": {"x": 0.55, "y": 0.45}}
        for shape in (line, tag):
            anchor = default_counter_pill_anchor(shape, 1)
            fx, fy = counter_pill_frame_coords(shape, anchor)
            roundtrip = counter_pill_bbox_from_frame(shape, fx, fy)
            self.assertAlmostEqual(roundtrip["x"], anchor["x"], places=4)
            self.assertAlmostEqual(roundtrip["y"], anchor["y"], places=4)

    def test_resolve_counter_pill_anchor_uses_defaults_when_missing(self):
        line = {"kind": "line", "p1": {"x": 0.0, "y": 0.5}, "p2": {"x": 1.0, "y": 0.5}}
        resolved = resolve_counter_pill_anchor(line, None, slot=2)
        self.assertEqual(resolved, default_counter_pill_anchor(line, 2))

    def test_spread_overlapping_pill_items_separates_stacked_anchors(self):
        line = {"kind": "line", "p1": {"x": 0.2, "y": 0.5}, "p2": {"x": 0.8, "y": 0.5}}
        items = [
            {"anchor": {"x": 0.5, "y": 0.0}, "count": 1},
            {"anchor": {"x": 0.5, "y": 0.0}, "count": 2},
        ]
        spread = spread_overlapping_pill_items(items, line)
        self.assertNotEqual(spread[0]["anchor"], spread[1]["anchor"])

    def test_fit_shape_preview_expands_small_zone(self):
        shape = {
            "kind": "zone",
            "pts": [
                {"x": 0.45, "y": 0.45},
                {"x": 0.47, "y": 0.45},
                {"x": 0.46, "y": 0.47},
            ],
        }
        fit = fit_shape_preview(shape, 168.0, 168.0)
        self.assertGreater(fit.w, 40.0)
        self.assertGreater(fit.h, 40.0)
        wx, wy = fit.to_widget(0.46, 0.46)
        nx, ny = fit.to_norm(wx, wy)
        self.assertAlmostEqual(nx, 0.46, places=3)
        self.assertAlmostEqual(ny, 0.46, places=3)

    def test_motion_path_travel_pauses_at_end(self):
        self.assertAlmostEqual(motion_path_travel_t(0.0, 0.5), 0.0)
        self.assertAlmostEqual(motion_path_travel_t(0.25, 0.5), 0.5)
        self.assertAlmostEqual(motion_path_travel_t(0.5, 0.5), 1.0)
        self.assertAlmostEqual(motion_path_travel_t(0.9, 0.5), 1.0)

    def test_preview_animation_params_slow_with_cooldown(self):
        fast_inc, fast_dwell = preview_animation_params(0.0, 3.0)
        slow_inc, slow_dwell = preview_animation_params(0.0, 12.0)
        self.assertLess(slow_inc, fast_inc)
        self.assertAlmostEqual(fast_dwell, 0.0)
        self.assertAlmostEqual(slow_dwell, 0.0)

    def test_preview_animation_params_dwell_increases_pause_fraction(self):
        _, no_dwell = preview_animation_params(0.0, 3.0)
        _, with_dwell = preview_animation_params(4.0, 3.0)
        self.assertAlmostEqual(no_dwell, 0.0)
        self.assertGreater(with_dwell, no_dwell)
        self.assertLessEqual(with_dwell, 0.55)

    def test_counter_pill_anchor_defaults_top_center(self):
        self.assertEqual(parse_counter_pill_anchor(None), DEFAULT_COUNTER_PILL_ANCHOR)
        self.assertEqual(parse_counter_pill_anchor({}), DEFAULT_COUNTER_PILL_ANCHOR)
        parsed = parse_counter_pill_anchor({"x": 0.25, "y": 0.75})
        self.assertAlmostEqual(parsed["x"], 0.25)
        self.assertAlmostEqual(parsed["y"], 0.75)

    def test_counter_pill_anchor_maps_through_shape_bounds(self):
        shape = {
            "kind": "zone",
            "pts": [
                {"x": 0.2, "y": 0.2},
                {"x": 0.8, "y": 0.2},
                {"x": 0.5, "y": 0.8},
            ],
        }
        fx, fy = counter_pill_frame_coords(shape, DEFAULT_COUNTER_PILL_ANCHOR)
        min_x, min_y, max_x, max_y = shape_bounds(shape)
        self.assertAlmostEqual(fx, (min_x + max_x) / 2.0)
        self.assertAlmostEqual(fy, min_y)
        roundtrip = counter_pill_bbox_from_frame(shape, fx, fy)
        self.assertAlmostEqual(roundtrip["x"], DEFAULT_COUNTER_PILL_ANCHOR["x"])
        self.assertAlmostEqual(roundtrip["y"], DEFAULT_COUNTER_PILL_ANCHOR["y"])

    def test_rules_for_shape_filters_by_shape_id(self):
        rules = [
            {"id": "r1", "shape_id": "zone_a", "name": "A"},
            {"id": "r2", "shape_id": "zone_b", "name": "B"},
            {"id": "r3", "shape_id": "zone_a", "name": "A2"},
        ]
        matched = rules_for_shape(rules, "zone_a")
        self.assertEqual(len(matched), 2)
        self.assertEqual(rules_for_shape(rules, ""), [])
        self.assertEqual(rules_for_shape(rules, "missing"), [])

    def test_build_armed_rule_ghost_entries_groups_by_shape(self):
        shape_a = {
            "id": "zone_a",
            "kind": "zone",
            "pts": [
                {"x": 0.2, "y": 0.2},
                {"x": 0.6, "y": 0.2},
                {"x": 0.4, "y": 0.5},
            ],
        }
        shape_b = {
            "id": "line_b",
            "kind": "line",
            "p1": {"x": 0.1, "y": 0.5},
            "p2": {"x": 0.9, "y": 0.5},
        }
        rules = [
            {
                "id": "r1",
                "shape_id": "zone_a",
                "name": "Zone path",
                "trigger": "path_match",
                "enabled": True,
                "conditions": {
                    "motion_path": [{"x": 0.3, "y": 0.35}, {"x": 0.5, "y": 0.35}],
                    "motion_path_space": MOTION_PATH_SPACE_FRAME,
                },
            },
            {
                "id": "r2",
                "shape_id": "line_b",
                "name": "Line path",
                "trigger": "path_match",
                "enabled": True,
                "conditions": {
                    "motion_path": [{"x": 0.2, "y": 0.5}, {"x": 0.8, "y": 0.5}],
                    "motion_path_space": MOTION_PATH_SPACE_FRAME,
                },
            },
            {
                "id": "r3",
                "shape_id": "zone_a",
                "name": "Disabled",
                "enabled": False,
                "conditions": {"motion_path": [{"x": 0.1, "y": 0.1}, {"x": 0.2, "y": 0.2}]},
            },
        ]
        armed = build_armed_rule_ghost_entries(rules, [shape_a, shape_b])
        self.assertIn("zone_a", armed)
        self.assertIn("line_b", armed)
        self.assertEqual(len(armed["zone_a"]), 1)
        self.assertEqual(len(armed["line_b"]), 1)
        self.assertEqual(armed["zone_a"][0]["name"], "Zone path")

    def test_rule_to_hover_ghost_entry(self):
        rule = {
            "name": "Front door",
            "trigger": "zone_enter",
            "shape_id": "z1",
            "conditions": {
                "motion_path": [{"x": 0.1, "y": 0.2}, {"x": 0.3, "y": 0.4}],
                "motion_path_space": MOTION_PATH_SPACE_SHAPE,
                "color": "red",
                "classes": ["person"],
                "cooldown_sec": 5,
            },
        }
        entry = rule_to_hover_ghost_entry(rule, color_index=1)
        self.assertIsNotNone(entry)
        assert entry is not None
        self.assertEqual(entry["name"], "Front door")
        self.assertEqual(len(entry["motion_path"]), 2)
        self.assertEqual(entry["motion_path_space"], MOTION_PATH_SPACE_SHAPE)
        self.assertEqual(entry["color_bucket"], "red")
        self.assertEqual(entry["ghost_color_index"], 1)

    def test_motion_path_frame_shape_roundtrip(self):
        shape = {
            "kind": "zone",
            "pts": [
                {"x": 0.2, "y": 0.3},
                {"x": 0.8, "y": 0.3},
                {"x": 0.5, "y": 0.7},
            ],
        }
        frame_path = [{"x": 0.5, "y": 0.3}, {"x": 0.5, "y": 0.6}]
        rel = frame_path_to_shape_relative(frame_path, shape)
        roundtrip = shape_relative_path_to_frame(rel, shape)
        for orig, back in zip(frame_path, roundtrip):
            self.assertAlmostEqual(orig["x"], back["x"], places=5)
            self.assertAlmostEqual(orig["y"], back["y"], places=5)

    def test_motion_path_moves_with_shape_bbox(self):
        shape_a = {
            "kind": "line",
            "p1": {"x": 0.1, "y": 0.4},
            "p2": {"x": 0.5, "y": 0.4},
        }
        shape_b = {
            "kind": "line",
            "p1": {"x": 0.5, "y": 0.6},
            "p2": {"x": 0.9, "y": 0.6},
        }
        frame_path = [{"x": 0.2, "y": 0.4}, {"x": 0.4, "y": 0.4}]
        rel = frame_path_to_shape_relative(frame_path, shape_a)
        moved = resolve_motion_path_for_frame(rel, shape_b, space=MOTION_PATH_SPACE_SHAPE)
        assert moved is not None
        self.assertAlmostEqual(moved[0]["x"], 0.6, places=4)
        self.assertAlmostEqual(moved[0]["y"], 0.6, places=4)
        self.assertAlmostEqual(moved[1]["x"], 0.8, places=4)
        self.assertAlmostEqual(moved[1]["y"], 0.6, places=4)

    def test_legacy_frame_motion_path_space_default(self):
        self.assertEqual(normalize_motion_path_space(None), MOTION_PATH_SPACE_FRAME)
        self.assertEqual(normalize_motion_path_space(""), MOTION_PATH_SPACE_FRAME)
        path = [{"x": 0.1, "y": 0.5}, {"x": 0.9, "y": 0.5}]
        resolved = resolve_motion_path_for_frame(path, None, space=None)
        self.assertEqual(resolved, path)

    def test_shape_relative_motion_path_resolves_for_matching(self):
        from core.automation.conditions import match_motion_path

        shape = {
            "kind": "zone",
            "pts": [
                {"x": 0.0, "y": 0.4},
                {"x": 1.0, "y": 0.4},
                {"x": 0.5, "y": 0.6},
            ],
        }
        rel_path = [{"x": 0.1, "y": 0.5}, {"x": 0.9, "y": 0.5}]
        frame_path = resolve_motion_path_for_frame(rel_path, shape, space=MOTION_PATH_SPACE_SHAPE)
        assert frame_path is not None
        history = [{"x": 0.15, "y": 0.52}, {"x": 0.85, "y": 0.48}]
        ok, details = match_motion_path(motion_path=frame_path, track_history=history, tolerance=0.15)
        self.assertTrue(ok)
        self.assertEqual(details.get("reason"), "path_distance_ok")

    def test_ghost_entry_label_prefers_rule_name(self):
        entry = {"name": "My Rule", "trigger": "zone_enter"}
        self.assertEqual(ghost_entry_label(entry), "My Rule")

    def test_ghost_entry_label_builds_summary(self):
        entry = {
            "trigger": "line_cross",
            "direction": "left_to_right",
            "classes": ["car"],
            "color_bucket": "blue",
        }
        label = ghost_entry_label(entry)
        self.assertIn("Cross", label)
        self.assertIn("car", label)
        self.assertIn("blue", label)

    def test_ghost_entry_label_marks_missing_snapshot(self):
        entry = {"name": "Lane rule", "has_snapshot": False}
        self.assertIn("no snap", ghost_entry_label(entry))

    def test_ghost_entry_hover_lines_multiline(self):
        entry = {
            "name": "Front gate rule",
            "trigger": "line_cross",
            "direction": "left_to_right",
            "classes": ["car", "truck"],
            "color_bucket": "blue",
            "motion_path": [{"x": 0.1, "y": 0.5}, {"x": 0.9, "y": 0.5}],
            "path_match_tolerance": 0.12,
            "dwell_min": 2.0,
            "cooldown_sec": 5.0,
        }
        lines = ghost_entry_hover_lines(entry)
        self.assertGreater(len(lines), 1)
        joined = "\n".join(lines)
        self.assertIn("Front gate rule", joined)
        self.assertIn("Trigger:", joined)
        self.assertIn("Cross", joined)
        self.assertIn("Direction:", joined)
        self.assertIn("Path:", joined)
        self.assertIn("Tolerance:", joined)
        self.assertIn("Class:", joined)
        self.assertIn("Color:", joined)
        self.assertIn("Dwell:", joined)
        self.assertIn("Cooldown:", joined)

    def test_ghost_entry_hover_lines_fallback(self):
        self.assertEqual(ghost_entry_hover_lines({}), ["Rule"])

    def test_counter_combine_modes(self):
        self.assertEqual(COUNTER_COMBINE_MODES, ("none", "sum", "max", "min"))
        self.assertEqual(normalize_counter_combine("sum"), "sum")
        self.assertEqual(normalize_counter_combine("MAX"), "max")
        self.assertEqual(normalize_counter_combine("", group="cars"), "sum")
        self.assertEqual(normalize_counter_combine("", group=""), "none")

    def test_combine_trigger_counts(self):
        self.assertEqual(combine_trigger_counts([2, 5, 1], "sum"), 8)
        self.assertEqual(combine_trigger_counts([2, 5, 1], "max"), 5)
        self.assertEqual(combine_trigger_counts([2, 5, 1], "min"), 1)
        self.assertEqual(combine_trigger_counts([], "sum"), 0)

    def test_prune_trigger_counts_drops_inactive_rules(self):
        counts = {"r1": 3, "r2": 1, "r3": 7}
        pruned = prune_trigger_counts(counts, ["r1", "r3"])
        self.assertEqual(pruned, {"r1": 3, "r3": 7})

    def test_counter_pill_config_from_rule(self):
        rule = {
            "id": "rule_a",
            "shape_id": "zone_1",
            "enabled": True,
            "name": "Cars",
            "conditions": {
                "show_counter": "always",
                "counter_group": "entry",
                "counter_combine": "max",
                "counter_pill_label": "In",
                "counter_pill_color": "#EF4444",
            },
        }
        cfg = counter_pill_config_from_rule(rule)
        self.assertIsNotNone(cfg)
        assert cfg is not None
        self.assertEqual(cfg["rule_id"], "rule_a")
        self.assertEqual(cfg["combine"], "max")
        self.assertEqual(cfg["label"], "In")
        self.assertEqual(cfg["bg_color"], "#EF4444")

    def test_counter_pill_label_defaults(self):
        rule = {"name": "Long rule name here", "conditions": {}}
        shape = {"label": "Front zone"}
        self.assertEqual(counter_pill_label_for_rule(rule, shape), "Long rule na")
        self.assertEqual(resolve_counter_pill_label({"label": "", "rule_name": ""}, shape), "Front zone")

    def test_build_counter_pill_render_items_groups_sum(self):
        configs = [
            {
                "rule_id": "r1",
                "shape_id": "z1",
                "mode": "always",
                "anchor": {"x": 0.5, "y": 0.0},
                "group": "cars",
                "combine": "sum",
                "label": "",
                "bg_color": "",
                "text_color": "",
                "rule_name": "A",
            },
            {
                "rule_id": "r2",
                "shape_id": "z1",
                "mode": "always",
                "anchor": {"x": 0.2, "y": 0.0},
                "group": "cars",
                "combine": "sum",
                "label": "",
                "bg_color": "",
                "text_color": "",
                "rule_name": "B",
            },
        ]
        items = build_counter_pill_render_items(
            configs,
            {"r1": 2, "r2": 3},
            shape={"label": "Zone"},
        )
        self.assertEqual(len(items), 1)
        self.assertEqual(items[0]["count"], 5)
        self.assertEqual(items[0]["label"], "cars")
        self.assertEqual(sorted(items[0]["rule_ids"]), ["r1", "r2"])

    def test_build_counter_pill_render_items_persists_on_trigger_count(self):
        configs = [
            {
                "rule_id": "r1",
                "shape_id": "z1",
                "mode": "on_trigger",
                "anchor": {"x": 0.5, "y": 0.0},
                "group": "",
                "combine": "none",
                "label": "Cars",
                "bg_color": "",
                "text_color": "",
                "rule_name": "Cars",
            }
        ]
        items = build_counter_pill_render_items(
            configs,
            {"r1": 4},
            now=100.0,
            pulse_ts_by_rule={},
        )
        self.assertEqual(len(items), 1)
        self.assertEqual(items[0]["count"], 4)


    def test_frame_motion_path_keeps_points_outside_shape_bbox(self):
        shape = {
            "kind": "zone",
            "pts": [
                {"x": 0.4, "y": 0.4},
                {"x": 0.6, "y": 0.4},
                {"x": 0.5, "y": 0.6},
            ],
        }
        raw_path = [
            {"x": 0.05, "y": 0.5},
            {"x": 0.35, "y": 0.52},
            {"x": 0.65, "y": 0.48},
            {"x": 0.95, "y": 0.5},
        ]
        path = normalize_frame_motion_path(raw_path)
        min_x, _, max_x, _ = shape_bounds(shape)
        self.assertLess(path[0]["x"], min_x)
        self.assertGreater(path[-1]["x"], max_x)
        resolved = resolve_motion_path_for_frame(path, shape, space=MOTION_PATH_SPACE_FRAME)
        self.assertEqual(resolved, path)

    def test_frame_motion_path_stored_roundtrip(self):
        frame_path = [
            {"x": 0.05, "y": 0.2},
            {"x": 0.5, "y": 0.5},
            {"x": 0.95, "y": 0.8},
        ]
        resolved = resolve_motion_path_for_frame(frame_path, None, space=MOTION_PATH_SPACE_FRAME)
        self.assertEqual(resolved, frame_path)
        for p in resolved or []:
            self.assertGreaterEqual(p["x"], 0.0)
            self.assertLessEqual(p["x"], 1.0)

    def test_full_scene_path_matching_with_track_history(self):
        from core.automation.conditions import match_motion_path

        full_scene_path = [
            {"x": 0.05, "y": 0.5},
            {"x": 0.35, "y": 0.52},
            {"x": 0.55, "y": 0.48},
            {"x": 0.95, "y": 0.5},
        ]
        track_history = [
            {"x": 0.30, "y": 0.51},
            {"x": 0.40, "y": 0.52},
            {"x": 0.50, "y": 0.49},
            {"x": 0.60, "y": 0.48},
        ]
        ok, details = match_motion_path(
            motion_path=full_scene_path,
            track_history=track_history,
            tolerance=0.15,
        )
        self.assertTrue(ok)
        self.assertEqual(details.get("reason"), "path_distance_ok")

    def test_legacy_shape_space_path_still_resolves(self):
        shape = {
            "kind": "zone",
            "pts": [
                {"x": 0.4, "y": 0.4},
                {"x": 0.6, "y": 0.4},
                {"x": 0.5, "y": 0.6},
            ],
        }
        rel = [{"x": 0.2, "y": 0.5}, {"x": 0.8, "y": 0.5}]
        frame = resolve_motion_path_for_frame(rel, shape, space=MOTION_PATH_SPACE_SHAPE)
        assert frame is not None
        self.assertGreater(frame[0]["x"], 0.4)
        self.assertLess(frame[1]["x"], 0.6)

    def test_frame_path_with_shape_ref_moves_with_shape_bbox(self):
        shape_a = {
            "kind": "line",
            "p1": {"x": 0.1, "y": 0.4},
            "p2": {"x": 0.5, "y": 0.4},
        }
        shape_b = {
            "kind": "line",
            "p1": {"x": 0.5, "y": 0.6},
            "p2": {"x": 0.9, "y": 0.6},
        }
        frame_path = [{"x": 0.2, "y": 0.4}, {"x": 0.4, "y": 0.4}]
        shape_ref = motion_path_shape_ref_from_shape(shape_a)
        moved = resolve_motion_path_for_frame(
            frame_path,
            shape_b,
            space=MOTION_PATH_SPACE_FRAME,
            shape_ref=shape_ref,
            attach_to_shape=True,
        )
        assert moved is not None
        self.assertAlmostEqual(moved[0]["x"], 0.6, places=4)
        self.assertAlmostEqual(moved[0]["y"], 0.6, places=4)
        self.assertAlmostEqual(moved[1]["x"], 0.8, places=4)
        self.assertAlmostEqual(moved[1]["y"], 0.6, places=4)

    def test_shape_relative_path_moves_with_current_shape(self):
        shape_a = {
            "kind": "line",
            "p1": {"x": 0.1, "y": 0.4},
            "p2": {"x": 0.5, "y": 0.4},
        }
        shape_b = {
            "kind": "line",
            "p1": {"x": 0.5, "y": 0.6},
            "p2": {"x": 0.9, "y": 0.6},
        }
        frame_path = [{"x": 0.2, "y": 0.4}, {"x": 0.4, "y": 0.4}]
        rel = frame_path_to_shape_relative(frame_path, shape_a)
        moved = resolve_motion_path_for_frame(
            rel,
            shape_b,
            space=MOTION_PATH_SPACE_SHAPE,
            attach_to_shape=True,
        )
        assert moved is not None
        self.assertAlmostEqual(moved[0]["x"], 0.6, places=4)
        self.assertAlmostEqual(moved[0]["y"], 0.6, places=4)
        self.assertAlmostEqual(moved[1]["x"], 0.8, places=4)
        self.assertAlmostEqual(moved[1]["y"], 0.6, places=4)

    def test_legacy_frame_path_without_ref_stays_frame_stable(self):
        shape = {
            "kind": "line",
            "p1": {"x": 0.5, "y": 0.6},
            "p2": {"x": 0.9, "y": 0.6},
        }
        frame_path = [{"x": 0.1, "y": 0.5}, {"x": 0.9, "y": 0.5}]
        resolved = resolve_motion_path_for_frame(
            frame_path,
            shape,
            space=MOTION_PATH_SPACE_FRAME,
            attach_to_shape=True,
        )
        self.assertEqual(resolved, frame_path)

    def test_rule_to_hover_ghost_entry_includes_shape_ref(self):
        shape = {
            "kind": "line",
            "p1": {"x": 0.2, "y": 0.3},
            "p2": {"x": 0.6, "y": 0.3},
        }
        shape_ref = motion_path_shape_ref_from_shape(shape)
        rule = {
            "name": "Lane",
            "trigger": "path_match",
            "shape_id": "l1",
            "conditions": {
                "motion_path": [{"x": 0.1, "y": 0.5}, {"x": 0.9, "y": 0.5}],
                "motion_path_space": MOTION_PATH_SPACE_FRAME,
                "motion_path_shape_ref": shape_ref,
            },
        }
        entry = rule_to_hover_ghost_entry(rule, shape=shape)
        assert entry is not None
        self.assertEqual(entry["motion_path_shape_ref"], shape_ref)

    def test_new_frame_path_rule_payload_produces_hover_ghost(self):
        """Regression: newly saved frame-space path rules must render hover ghosts."""
        shape = {
            "kind": "zone",
            "pts": [
                {"x": 0.2, "y": 0.2},
                {"x": 0.6, "y": 0.2},
                {"x": 0.4, "y": 0.5},
            ],
        }
        path = normalize_frame_motion_path([{"x": 0.25, "y": 0.35}, {"x": 0.55, "y": 0.35}])
        shape_ref = motion_path_shape_ref_from_shape(shape)
        saved_rule = {
            "id": "rule_new_abc",
            "name": "New path rule",
            "camera_id": "cam_1",
            "shape_id": "zone_new_1",
            "trigger": "path_match",
            "enabled": True,
            "conditions": {
                "tracker_namespaces": [MOTION_BOX_NAMESPACE, BACKEND_SORT_NAMESPACE],
                "require_detection": False,
                "motion_path": path,
                "motion_path_space": MOTION_PATH_SPACE_FRAME,
                "motion_path_shape_ref": shape_ref,
                "derived_trigger": "zone_enter",
                "cooldown_sec": 2.0,
            },
            "actions": [{"type": "snapshot", "save_dir": "captures/motion_watch"}],
        }
        stale_cache = [{"id": "rule_old", "shape_id": "zone_other", "name": "Old"}]
        cache = upsert_event_rule_in_cache(stale_cache, saved_rule)
        self.assertEqual(len(cache), 2)
        matched = rules_for_shape(cache, "zone_new_1")
        self.assertEqual(len(matched), 1)
        entry = rule_to_hover_ghost_entry(matched[0], shape=shape)
        assert entry is not None
        self.assertEqual(entry["name"], "New path rule")
        self.assertEqual(len(entry["motion_path"]), 2)
        self.assertEqual(entry["motion_path_space"], MOTION_PATH_SPACE_FRAME)
        self.assertEqual(entry["motion_path_shape_ref"], shape_ref)
        resolved = resolve_motion_path_for_frame(
            entry["motion_path"],
            shape,
            space=entry["motion_path_space"],
            shape_ref=entry["motion_path_shape_ref"],
            attach_to_shape=True,
        )
        assert resolved is not None
        self.assertEqual(len(resolved), 2)
        self.assertTrue(ghost_entry_label(entry))

    def test_upsert_event_rule_in_cache_replaces_by_id(self):
        original = [
            {"id": "rule_a", "shape_id": "z1", "name": "A v1"},
            {"id": "rule_b", "shape_id": "z2", "name": "B"},
        ]
        updated = upsert_event_rule_in_cache(
            original,
            {"id": "rule_a", "shape_id": "z1", "name": "A v2", "conditions": {"motion_path": []}},
        )
        self.assertEqual(len(updated), 2)
        by_id = {r["id"]: r for r in updated}
        self.assertEqual(by_id["rule_a"]["name"], "A v2")
        self.assertEqual(by_id["rule_b"]["name"], "B")


if __name__ == "__main__":
    unittest.main()
