"""Tests for backend detection API helpers used by the shape Event Rule dialog."""

from __future__ import annotations

import unittest
from unittest.mock import patch

from desktop.utils.event_rules_api import (
    backend_detection_status_label,
    ensure_backend_detection_for_rules,
    get_backend_detection_enabled,
)


class BackendDetectionStatusLabelTests(unittest.TestCase):
    def test_on_off_unknown_labels(self):
        self.assertEqual(backend_detection_status_label(True), "On")
        self.assertEqual(backend_detection_status_label(False), "Off")
        self.assertEqual(backend_detection_status_label(None), "Unknown")


class GetBackendDetectionEnabledTests(unittest.TestCase):
    @patch("desktop.utils.event_rules_api._api_get")
    def test_reads_verification_enabled_from_data(self, mock_get):
        mock_get.return_value = {"success": True, "data": {"verification_enabled": True}}
        self.assertTrue(get_backend_detection_enabled("http://localhost:5000/api", "cam1"))

    @patch("desktop.utils.event_rules_api._api_get")
    def test_returns_none_when_key_missing(self, mock_get):
        mock_get.return_value = {"success": True, "data": {}}
        self.assertIsNone(get_backend_detection_enabled("http://localhost:5000/api", "cam1"))

    @patch("desktop.utils.event_rules_api._api_get")
    def test_returns_none_on_api_error(self, mock_get):
        mock_get.side_effect = RuntimeError("connection refused")
        self.assertIsNone(get_backend_detection_enabled("http://localhost:5000/api", "cam1"))


class EnsureBackendDetectionForRulesTests(unittest.TestCase):
    @patch("desktop.utils.event_rules_api._api_put")
    def test_enable_puts_verification_enabled_true(self, mock_put):
        mock_put.return_value = {"success": True}
        ok = ensure_backend_detection_for_rules("http://localhost:5000/api", "cam1", verification_enabled=True)
        self.assertTrue(ok)
        mock_put.assert_called_once_with(
            "http://localhost:5000/api",
            "cameras/cam1/detection-config",
            {"verification_enabled": True},
        )

    @patch("desktop.utils.event_rules_api._api_put")
    def test_disable_puts_verification_enabled_false(self, mock_put):
        mock_put.return_value = {"success": True}
        ok = ensure_backend_detection_for_rules("http://localhost:5000/api", "cam1", verification_enabled=False)
        self.assertTrue(ok)
        mock_put.assert_called_once_with(
            "http://localhost:5000/api",
            "cameras/cam1/detection-config",
            {"verification_enabled": False},
        )

    @patch("desktop.utils.event_rules_api._api_put")
    def test_returns_false_on_failure(self, mock_put):
        mock_put.side_effect = RuntimeError("timeout")
        ok = ensure_backend_detection_for_rules("http://localhost:5000/api", "cam1")
        self.assertFalse(ok)


if __name__ == "__main__":
    unittest.main()
