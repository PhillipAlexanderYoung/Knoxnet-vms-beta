"""Unit tests for capture labeling helpers and FTS query building."""

from __future__ import annotations

import unittest


class CaptureLabelModelTests(unittest.TestCase):
    def test_normalize_legacy_local_enrich(self):
        from core.capture_label_model import CAPTURE_LABEL_AUTO, CAPTURE_LABEL_OFF, normalize_capture_label_model

        self.assertEqual(normalize_capture_label_model("off"), CAPTURE_LABEL_OFF)
        self.assertEqual(normalize_capture_label_model("true"), CAPTURE_LABEL_AUTO)
        self.assertEqual(normalize_capture_label_model("yolo-nano"), "yolo-nano")

    def test_merge_shape_name_tags(self):
        from core.capture_label_model import merge_shape_name_tags

        sidecar = {"trigger": {"shape_name": "Zone 1 - Left Lane - East Bound"}}
        merge_shape_name_tags(sidecar)
        tags = sidecar.get("tags") or []
        self.assertIn("left", tags)
        self.assertIn("lane", tags)
        self.assertIn("east", tags)
        self.assertIn("zone 1 - left lane - east bound", tags)

    def test_resolve_capture_label_model_precedence(self):
        from core.capture_label_model import resolve_capture_label_model

        per_cam = {"capture_label_model": "mobilenet"}
        global_prefs = {"events_index": {"capture_label_model": "yolo-nano"}}
        self.assertEqual(resolve_capture_label_model(per_cam, global_prefs), "mobilenet")
        self.assertEqual(resolve_capture_label_model({}, global_prefs), "yolo-nano")
        self.assertEqual(resolve_capture_label_model({"local_enrich": True}, {}), "auto")

    def test_apply_enrichment_to_sidecar(self):
        from core.capture_label_model import apply_enrichment_to_sidecar

        sidecar: dict = {"tags": ["left", "lane"], "metadata": {}}
        enrich = {
            "detection_classes": ["truck"],
            "tags": ["truck"],
            "detections": [{"class": "truck", "confidence": 0.9, "bbox": {"x": 1, "y": 2, "w": 3, "h": 4}}],
            "metadata": {"capture_label": {"model": "auto"}},
        }
        apply_enrichment_to_sidecar(sidecar, enrich)
        self.assertTrue(sidecar.get("enable_detections"))
        self.assertIn("truck", sidecar.get("tags") or [])
        self.assertIn("left", sidecar.get("tags") or [])


class EventIndexSearchTests(unittest.TestCase):
    def test_fts_and_query_multi_token(self):
        from core.events_search import fts_and_query, merge_operator_shape_tags

        q = fts_and_query("left lane east")
        self.assertEqual(q, '"left" AND "lane" AND "east"')

        tags = merge_operator_shape_tags(["truck"], shape_name="Main Gate - East")
        self.assertIn("east", tags)
        self.assertIn("truck", tags)
        self.assertIn("main gate - east", tags)


if __name__ == "__main__":
    unittest.main()
