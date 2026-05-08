"""Non-blocking public beta update notification."""
from __future__ import annotations

from PySide6.QtCore import Qt, QUrl
from PySide6.QtGui import QDesktopServices, QFont
from PySide6.QtWidgets import QDialog, QHBoxLayout, QLabel, QPushButton, QVBoxLayout

GITHUB_REPO_URL = "https://github.com/PhillipAlexanderYoung/Knoxnet-vms-beta"


class UpdateAvailableDialog(QDialog):
    def __init__(
        self,
        current_version: str,
        latest_version: str,
        channel: str = "beta",
        download_url: str = GITHUB_REPO_URL,
        parent=None,
    ):
        super().__init__(parent)
        self._download_url = download_url
        self.setWindowTitle("Knoxnet VMS Beta Update")
        self.setFixedSize(500, 240)
        self.setWindowFlags(
            Qt.WindowType.Tool
            | Qt.WindowType.WindowStaysOnTopHint
            | Qt.WindowType.WindowTitleHint
            | Qt.WindowType.WindowCloseButtonHint
        )

        layout = QVBoxLayout(self)
        layout.setContentsMargins(28, 24, 28, 20)
        layout.setSpacing(12)

        title = QLabel("Update available")
        title.setFont(QFont("Segoe UI", 15, QFont.Weight.Bold))
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title)

        message = QLabel(
            f"Update available: version {latest_version}. "
            "Download the latest beta from GitHub."
        )
        message.setWordWrap(True)
        message.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(message)

        details = QLabel(f"Installed: {current_version}    Channel: {channel or 'beta'}")
        details.setAlignment(Qt.AlignmentFlag.AlignCenter)
        details.setStyleSheet("color: #666;")
        layout.addWidget(details)

        buttons = QHBoxLayout()
        buttons.addStretch()

        later = QPushButton("Dismiss")
        later.clicked.connect(self.accept)
        buttons.addWidget(later)

        github = QPushButton("Open GitHub")
        github.clicked.connect(self._open_download)
        buttons.addWidget(github)

        buttons.addStretch()
        layout.addLayout(buttons)

    def _open_download(self):
        QDesktopServices.openUrl(QUrl(self._download_url))
        self.accept()
