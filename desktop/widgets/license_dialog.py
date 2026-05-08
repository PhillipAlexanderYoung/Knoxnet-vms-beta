"""Public beta plan dialog."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from PySide6.QtCore import Signal
from PySide6.QtWidgets import QDialog, QHBoxLayout, QLabel, QPushButton, QVBoxLayout, QWidget


class LicenseDialog(QDialog):
    """Shows the fixed public beta entitlement without activation UI."""

    entitlement_changed = Signal()

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setWindowTitle("Knoxnet VMS Beta Plan")
        self.resize(440, 260)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 24, 24, 20)
        layout.setSpacing(12)

        title = QLabel("Knoxnet VMS Beta")
        title.setStyleSheet("font-size: 18px; font-weight: 700;")
        layout.addWidget(title)

        count = self._count_cameras_from_disk()
        body = QLabel(
            "This public beta includes free local use for up to 4 cameras. "
            "Activation and cloud entitlement flows are not included in this public beta.\n\n"
            f"Current configured cameras: {count}/4"
        )
        body.setWordWrap(True)
        layout.addWidget(body)

        warning = QLabel(
            "Early technical beta: intended for testers and demo feedback, "
            "not production-critical security use."
        )
        warning.setWordWrap(True)
        warning.setStyleSheet("color: #a15c00;")
        layout.addWidget(warning)

        row = QHBoxLayout()
        row.addStretch()
        close = QPushButton("Close")
        close.clicked.connect(self.accept)
        row.addWidget(close)
        layout.addLayout(row)

    @staticmethod
    def _candidate_camera_files() -> list[Path]:
        root = Path(__file__).resolve().parents[2]
        try:
            from core.paths import get_data_dir
            data_dir = get_data_dir()
            return [data_dir / "cameras.json", root / "data" / "cameras.json", root / "cameras.json"]
        except Exception:
            return [root / "data" / "cameras.json", root / "cameras.json"]

    @staticmethod
    def _count_cameras_from_disk() -> int:
        for path in LicenseDialog._candidate_camera_files():
            try:
                if not path.exists():
                    continue
                data = json.loads(path.read_text(encoding="utf-8", errors="ignore"))
                if isinstance(data, list):
                    return len(data)
                if isinstance(data, dict):
                    cameras = data.get("cameras") or data.get("data")
                    if isinstance(cameras, list):
                        return len(cameras)
            except Exception:
                continue
        return 0
