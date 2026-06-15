"""Tests for MotionWatchDialog settings-only mode."""

from __future__ import annotations

import sys
import unittest

try:
    from PySide6.QtWidgets import QApplication

    from desktop.widgets.camera import MotionWatchDialog

    _QT_AVAILABLE = True
except ModuleNotFoundError:
    _QT_AVAILABLE = False


def _ensure_qapp() -> "QApplication":
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
    return app


@unittest.skipUnless(_QT_AVAILABLE, "PySide6 is required for MotionWatchDialog tests")
class MotionWatchDialogSettingsOnlyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        _ensure_qapp()

    def test_settings_only_preserves_legacy_session_fields(self) -> None:
        base = {
            "duration_sec": -1,
            "duration_unit": "Infinite",
            "allow_zone": False,
            "allow_line": True,
            "allow_tag": False,
            "cooldown_ms": 2500,
            "save_dir": "captures/legacy",
            "resize_w": 1280,
        }
        dlg = MotionWatchDialog(base, settings_only=True)
        dlg.save_dir_edit.setText("captures/event_rules")
        dlg.resize_w_spin.setValue(960)
        out = dlg.get_settings()
        self.assertEqual(out["save_dir"], "captures/event_rules")
        self.assertEqual(out["resize_w"], 960)
        self.assertEqual(out["duration_sec"], -1)
        self.assertEqual(out["duration_unit"], "Infinite")
        self.assertFalse(out["allow_zone"])
        self.assertTrue(out["allow_line"])
        self.assertFalse(out["allow_tag"])
        self.assertEqual(out["cooldown_ms"], 2500)

    def test_advanced_mode_reads_legacy_session_fields(self) -> None:
        base = {
            "duration_sec": 120,
            "duration_unit": "Seconds",
            "allow_zone": True,
            "allow_line": False,
            "allow_tag": True,
            "cooldown_ms": 500,
            "save_dir": "captures/legacy",
        }
        dlg = MotionWatchDialog(base, settings_only=False)
        dlg.zone_check.setChecked(False)
        dlg.line_check.setChecked(True)
        dlg.cooldown_spin.setValue(1500)
        out = dlg.get_settings()
        self.assertFalse(out["allow_zone"])
        self.assertTrue(out["allow_line"])
        self.assertEqual(out["cooldown_ms"], 1500)


if __name__ == "__main__":
    unittest.main()
