"""Standalone Camera Discovery dialog with inline editing.

Features:
- 11-column table with inline-editable fields per discovered device
- Per-row Test button (OpenCV RTSP probe with resolution/FPS feedback)
- Bulk credential/port toolbar for setting values across checked rows
- Encoder/multi-channel auto-detection with grouped channel rows
- Visual IP grouping with "Apply to Same IP" convenience
- Manual camera entry
"""

from __future__ import annotations

import json
import uuid
import logging
import threading
from pathlib import Path
from typing import List, Optional

from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, QPushButton,
    QCheckBox, QTableWidget, QTableWidgetItem, QHeaderView, QProgressBar,
    QFrame, QWidget, QAbstractItemView,
)
from PySide6.QtCore import Qt, QTimer, Slot, QUrl, Signal, QObject
from PySide6.QtGui import QFont, QDesktopServices, QColor, QBrush

from desktop.utils.network_scanner import (
    NetworkCameraScanner, build_rtsp_url, get_vendor_defaults,
    test_rtsp_connection, enumerate_channels, VENDOR_ENCODER_HINTS,
    VENDOR_RTSP_TEMPLATES,
)

logger = logging.getLogger(__name__)


def _get_camera_slots() -> tuple[int, int, int]:
    """Return (existing_count, camera_limit, remaining_slots)."""
    try:
        from core.entitlements import get_camera_limit
        limit = get_camera_limit()
    except Exception:
        limit = 4
    existing = 0
    try:
        from desktop.widgets.license_dialog import LicenseDialog
        existing = LicenseDialog._count_cameras_from_disk()
    except Exception:
        pass
    remaining = max(0, limit - existing)
    return existing, limit, remaining


# ═══════════════════════════════════════════════════════════════════════════
#  Theme
# ═══════════════════════════════════════════════════════════════════════════

_ACCENT = "#a855f7"
_BG_DARK = "#1a1a1a"
_BG_LIGHT = "#2d2d2d"
_BG_GROUP_A = "#1e1e2a"
_BG_GROUP_B = "#1a1a1a"
_BORDER = "#3f3f3f"
_TEXT_MAIN = "#ffffff"
_TEXT_DIM = "#a0a0a0"
_GREEN = "#22c55e"
_RED = "#ef4444"
_YELLOW = "#f59e0b"
_CYAN = "#22d3ee"

_DIALOG_SS = f"""
QDialog {{ background-color: {_BG_DARK}; color: {_TEXT_MAIN}; }}
QLabel {{ color: {_TEXT_MAIN}; border: none; }}
QLineEdit {{
    background-color: {_BG_LIGHT}; color: {_TEXT_MAIN};
    border: 1px solid {_BORDER}; border-radius: 3px; padding: 3px 5px; font-size: 12px;
}}
QLineEdit:focus {{ border: 1px solid {_ACCENT}; }}
QCheckBox {{ color: {_TEXT_MAIN}; spacing: 6px; }}
QCheckBox::indicator {{ width: 16px; height: 16px; }}
QProgressBar {{
    background-color: {_BG_LIGHT}; border: 1px solid {_BORDER};
    border-radius: 4px; height: 6px; text-align: center;
}}
QProgressBar::chunk {{ background-color: {_ACCENT}; border-radius: 3px; }}
QTableWidget {{
    background-color: {_BG_DARK}; color: {_TEXT_MAIN};
    border: 1px solid {_BORDER}; gridline-color: {_BORDER};
    selection-background-color: {_BG_LIGHT}; font-size: 12px;
}}
QTableWidget::item {{ padding: 2px 4px; }}
QHeaderView::section {{
    background-color: {_BG_LIGHT}; color: {_TEXT_DIM}; border: none;
    border-bottom: 1px solid {_BORDER}; padding: 4px 6px;
    font-size: 11px; font-weight: bold;
}}
"""

_BTN_PRIMARY = f"""
QPushButton {{
    background-color: {_ACCENT}; color: {_TEXT_MAIN}; border: none;
    border-radius: 5px; padding: 8px 20px; font-size: 13px; font-weight: bold;
}}
QPushButton:hover {{ background-color: #9333ea; }}
QPushButton:pressed {{ background-color: #7e22ce; }}
QPushButton:disabled {{ background-color: {_BG_LIGHT}; color: {_TEXT_DIM}; }}
"""

_BTN_SECONDARY = f"""
QPushButton {{
    background-color: transparent; color: {_TEXT_DIM};
    border: 1px solid {_BORDER}; border-radius: 5px; padding: 8px 20px; font-size: 13px;
}}
QPushButton:hover {{ color: {_TEXT_MAIN}; border-color: {_TEXT_DIM}; }}
"""

_BTN_SMALL = f"""
QPushButton {{
    background-color: {_BG_LIGHT}; color: {_TEXT_MAIN};
    border: 1px solid {_BORDER}; border-radius: 3px; padding: 3px 10px; font-size: 11px;
}}
QPushButton:hover {{ border-color: {_ACCENT}; }}
QPushButton:disabled {{ color: {_TEXT_DIM}; }}
"""

_BTN_TEST = f"""
QPushButton {{
    background-color: {_BG_LIGHT}; color: {_ACCENT};
    border: 1px solid {_BORDER}; border-radius: 3px; padding: 2px 8px;
    font-size: 11px; font-weight: bold;
}}
QPushButton:hover {{ border-color: {_ACCENT}; background-color: #2d1f4e; }}
QPushButton:disabled {{ color: {_TEXT_DIM}; }}
"""

_BTN_PROBE = f"""
QPushButton {{
    background-color: {_BG_LIGHT}; color: {_CYAN};
    border: 1px solid {_BORDER}; border-radius: 3px; padding: 2px 6px;
    font-size: 10px;
}}
QPushButton:hover {{ border-color: {_CYAN}; }}
QPushButton:disabled {{ color: {_TEXT_DIM}; }}
"""

_BTN_WEB = f"""
QPushButton {{
    background-color: transparent; color: {_TEXT_DIM};
    border: 1px solid {_BORDER}; border-radius: 3px; padding: 2px 4px;
    font-size: 12px;
}}
QPushButton:hover {{ color: {_ACCENT}; border-color: {_ACCENT}; }}
"""

_LINK_BTN = f"""
QPushButton {{
    background: transparent; border: none; color: {_ACCENT};
    font-size: 12px; text-decoration: underline; padding: 2px;
}}
QPushButton:hover {{ color: #c084fc; }}
"""

_INLINE_EDIT = f"""
QLineEdit {{
    background-color: {_BG_LIGHT}; color: {_TEXT_MAIN};
    border: 1px solid {_BORDER}; border-radius: 2px; padding: 2px 4px; font-size: 11px;
}}
QLineEdit:focus {{ border: 1px solid {_ACCENT}; }}
"""

# Column indices
COL_CHK = 0
COL_IP = 1
COL_MFG = 2
COL_CH = 3
COL_NAME = 4
COL_RTSP = 5
COL_PORT = 6
COL_USER = 7
COL_PASS = 8
COL_TEST = 9
COL_STATUS = 10
NUM_COLS = 11

HEADER_LABELS = [
    "", "IP Address", "Mfg", "Ch", "Name", "RTSP Path", "Port",
    "User", "Pass", "", "Status",
]

_GROUP_COLORS = [QColor(_BG_GROUP_A), QColor(_BG_GROUP_B)]


class CameraScannerDialog(QDialog):
    """Standalone camera discovery dialog with inline editing per row."""

    def __init__(self, cameras_json_path: Path, on_open_config=None, parent=None):
        super().__init__(parent)
        self.cameras_json_path = cameras_json_path
        self._on_open_config = on_open_config
        self._scanner: Optional[NetworkCameraScanner] = None
        self._scan_complete = False
        self._discovered: List[dict] = []
        self._row_map: dict[str, int] = {}
        self._ip_group_index: dict[str, int] = {}
        self._test_threads: dict[int, threading.Thread] = {}

        self.setWindowTitle("Discover Cameras")
        self.setWindowFlags(Qt.WindowType.Tool | Qt.WindowType.WindowStaysOnTopHint)
        self.setMinimumSize(1100, 620)
        self.resize(1200, 660)
        self.setStyleSheet(_DIALOG_SS)

        self._build_ui()

    # ═══════════════════════════════════════════════════════════════════
    #  UI construction
    # ═══════════════════════════════════════════════════════════════════

    def _build_ui(self):
        lay = QVBoxLayout(self)
        lay.setContentsMargins(18, 14, 18, 14)
        lay.setSpacing(6)

        title = QLabel("Camera Discovery")
        title.setFont(QFont("Segoe UI", 16, QFont.Weight.Bold))
        lay.addWidget(title)

        subtitle = QLabel(
            "Scanning your local network for IP cameras and encoders. "
            "Multi-channel devices are auto-detected and expanded."
        )
        subtitle.setFont(QFont("Segoe UI", 10))
        subtitle.setStyleSheet(f"color: {_TEXT_DIM};")
        lay.addWidget(subtitle)

        # Progress
        self.scan_progress = QProgressBar()
        self.scan_progress.setRange(0, 0)
        self.scan_progress.setFixedHeight(6)
        self.scan_progress.setTextVisible(False)
        lay.addWidget(self.scan_progress)

        self.scan_status = QLabel("Initializing\u2026")
        self.scan_status.setFont(QFont("Segoe UI", 10))
        self.scan_status.setStyleSheet(f"color: {_TEXT_DIM};")
        lay.addWidget(self.scan_status)

        # ── Bulk operations toolbar ──────────────────────────────────
        bulk_frame = QFrame()
        bulk_frame.setStyleSheet(f"background: {_BG_LIGHT}; border-radius: 4px;")
        bulk_lay = QHBoxLayout(bulk_frame)
        bulk_lay.setContentsMargins(10, 6, 10, 6)
        bulk_lay.setSpacing(6)

        bulk_lay.addWidget(QLabel("Bulk:"))

        bulk_lay.addWidget(QLabel("User"))
        self.bulk_user = QLineEdit()
        self.bulk_user.setFixedWidth(90)
        self.bulk_user.setPlaceholderText("admin")
        self.bulk_user.setStyleSheet(_INLINE_EDIT)
        bulk_lay.addWidget(self.bulk_user)

        bulk_lay.addWidget(QLabel("Pass"))
        self.bulk_pass = QLineEdit()
        self.bulk_pass.setFixedWidth(90)
        self.bulk_pass.setPlaceholderText("password")
        self.bulk_pass.setEchoMode(QLineEdit.EchoMode.Password)
        self.bulk_pass.setStyleSheet(_INLINE_EDIT)
        bulk_lay.addWidget(self.bulk_pass)

        bulk_lay.addWidget(QLabel("Port"))
        self.bulk_port = QLineEdit()
        self.bulk_port.setFixedWidth(55)
        self.bulk_port.setPlaceholderText("554")
        self.bulk_port.setStyleSheet(_INLINE_EDIT)
        bulk_lay.addWidget(self.bulk_port)

        apply_btn = QPushButton("Apply to Checked")
        apply_btn.setStyleSheet(_BTN_SMALL)
        apply_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        apply_btn.clicked.connect(self._on_bulk_apply)
        bulk_lay.addWidget(apply_btn)

        same_ip_btn = QPushButton("Apply to Same IP")
        same_ip_btn.setToolTip("Copy credentials from the first checked row to all rows sharing the same IP")
        same_ip_btn.setStyleSheet(_BTN_SMALL)
        same_ip_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        same_ip_btn.clicked.connect(self._on_apply_same_ip)
        bulk_lay.addWidget(same_ip_btn)

        bulk_lay.addStretch()

        test_all_btn = QPushButton("Test All Checked")
        test_all_btn.setStyleSheet(_BTN_SMALL)
        test_all_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        test_all_btn.clicked.connect(self._on_test_all_checked)
        bulk_lay.addWidget(test_all_btn)

        lay.addWidget(bulk_frame)

        # ── Table ────────────────────────────────────────────────────
        self.table = QTableWidget(0, NUM_COLS)
        self.table.setHorizontalHeaderLabels(HEADER_LABELS)
        hdr = self.table.horizontalHeader()
        hdr.setSectionResizeMode(COL_CHK, QHeaderView.ResizeMode.Fixed)
        hdr.setSectionResizeMode(COL_IP, QHeaderView.ResizeMode.ResizeToContents)
        hdr.setSectionResizeMode(COL_MFG, QHeaderView.ResizeMode.ResizeToContents)
        hdr.setSectionResizeMode(COL_CH, QHeaderView.ResizeMode.Fixed)
        hdr.setSectionResizeMode(COL_NAME, QHeaderView.ResizeMode.Stretch)
        hdr.setSectionResizeMode(COL_RTSP, QHeaderView.ResizeMode.Stretch)
        hdr.setSectionResizeMode(COL_PORT, QHeaderView.ResizeMode.Fixed)
        hdr.setSectionResizeMode(COL_USER, QHeaderView.ResizeMode.Interactive)
        hdr.setSectionResizeMode(COL_PASS, QHeaderView.ResizeMode.Interactive)
        hdr.setSectionResizeMode(COL_TEST, QHeaderView.ResizeMode.Fixed)
        hdr.setSectionResizeMode(COL_STATUS, QHeaderView.ResizeMode.Stretch)
        self.table.setColumnWidth(COL_CHK, 30)
        self.table.setColumnWidth(COL_CH, 32)
        self.table.setColumnWidth(COL_PORT, 55)
        self.table.setColumnWidth(COL_USER, 80)
        self.table.setColumnWidth(COL_PASS, 80)
        self.table.setColumnWidth(COL_TEST, 80)
        self.table.verticalHeader().setVisible(False)
        self.table.verticalHeader().setDefaultSectionSize(30)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.NoSelection)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.table.setMinimumHeight(200)
        lay.addWidget(self.table, stretch=1)

        # ── Manual entry ─────────────────────────────────────────────
        self._manual_frame = QFrame()
        self._manual_frame.setVisible(False)
        m_lay = QVBoxLayout(self._manual_frame)
        m_lay.setContentsMargins(0, 6, 0, 0)
        m_lay.setSpacing(6)

        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.HLine)
        sep.setStyleSheet(f"color: {_BORDER};")
        m_lay.addWidget(sep)

        row1 = QHBoxLayout()
        for lbl_text, attr, placeholder, width in [
            ("Name", "manual_name", "e.g. Front Door", 130),
            ("RTSP", "manual_url", "rtsp://user:pass@ip:554/stream", 320),
        ]:
            lbl = QLabel(lbl_text)
            lbl.setFont(QFont("Segoe UI", 10, QFont.Weight.DemiBold))
            lbl.setFixedWidth(42)
            row1.addWidget(lbl)
            le = QLineEdit()
            le.setPlaceholderText(placeholder)
            le.setMinimumWidth(width)
            setattr(self, attr, le)
            row1.addWidget(le)
        m_lay.addLayout(row1)

        self.manual_error = QLabel("")
        self.manual_error.setStyleSheet("color: #ef4444; font-size: 11px;")
        self.manual_error.setWordWrap(True)
        m_lay.addWidget(self.manual_error)

        add_manual_btn = QPushButton("Add This Camera")
        add_manual_btn.setStyleSheet(_BTN_PRIMARY)
        add_manual_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        add_manual_btn.clicked.connect(self._on_add_manual)
        add_row = QHBoxLayout()
        add_row.addStretch()
        add_row.addWidget(add_manual_btn)
        m_lay.addLayout(add_row)

        lay.addWidget(self._manual_frame)

        manual_link = QPushButton("Enter a camera manually")
        manual_link.setStyleSheet(_LINK_BTN)
        manual_link.setCursor(Qt.CursorShape.PointingHandCursor)
        manual_link.clicked.connect(lambda: self._manual_frame.setVisible(not self._manual_frame.isVisible()))
        lay.addWidget(manual_link, alignment=Qt.AlignmentFlag.AlignLeft)

        # ── Limit banner ─────────────────────────────────────────────
        self.limit_banner = QLabel()
        self.limit_banner.setFont(QFont("Segoe UI", 10))
        self.limit_banner.setWordWrap(True)
        self.limit_banner.setVisible(False)
        lay.addWidget(self.limit_banner)

        self.beta_limit_note = QLabel("Public beta limit: 4 cameras")
        self.beta_limit_note.setFont(QFont("Segoe UI", 10))
        self.beta_limit_note.setStyleSheet(f"color: {_TEXT_DIM};")
        self.beta_limit_note.setVisible(False)
        lay.addWidget(self.beta_limit_note, alignment=Qt.AlignmentFlag.AlignLeft)

        # ── Bottom buttons ───────────────────────────────────────────
        btn_row = QHBoxLayout()

        self.btn_rescan = QPushButton("Re-scan")
        self.btn_rescan.setStyleSheet(_BTN_SECONDARY)
        self.btn_rescan.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_rescan.setEnabled(False)
        self.btn_rescan.clicked.connect(self._on_rescan)
        btn_row.addWidget(self.btn_rescan)

        if self._on_open_config:
            config_btn = QPushButton("Camera Config\u2026")
            config_btn.setStyleSheet(_BTN_SECONDARY)
            config_btn.setCursor(Qt.CursorShape.PointingHandCursor)
            config_btn.setToolTip("Open the camera configuration panel to edit or delete existing cameras")
            config_btn.clicked.connect(self._go_to_config)
            btn_row.addWidget(config_btn)

        btn_row.addStretch()

        self.btn_add_selected = QPushButton("Add Selected")
        self.btn_add_selected.setStyleSheet(_BTN_PRIMARY)
        self.btn_add_selected.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_add_selected.setEnabled(False)
        self.btn_add_selected.clicked.connect(self._on_add_selected)
        btn_row.addWidget(self.btn_add_selected)

        close_btn = QPushButton("Close")
        close_btn.setStyleSheet(_BTN_SECONDARY)
        close_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        close_btn.clicked.connect(self.accept)
        btn_row.addWidget(close_btn)

        lay.addLayout(btn_row)

    # ═══════════════════════════════════════════════════════════════════
    #  Row helpers
    # ═══════════════════════════════════════════════════════════════════

    def _make_inline_edit(self, text: str = "", placeholder: str = "",
                          width: int = 0, echo_password: bool = False) -> QLineEdit:
        le = QLineEdit(text)
        le.setPlaceholderText(placeholder)
        le.setStyleSheet(_INLINE_EDIT)
        if width:
            le.setFixedWidth(width)
        if echo_password:
            le.setEchoMode(QLineEdit.EchoMode.Password)
        return le

    def _get_row_edit(self, row: int, col: int) -> Optional[QLineEdit]:
        w = self.table.cellWidget(row, col)
        return w if isinstance(w, QLineEdit) else None

    def _get_row_checkbox(self, row: int) -> Optional[QCheckBox]:
        w = self.table.cellWidget(row, COL_CHK)
        if w:
            return w.findChild(QCheckBox)
        return None

    def _get_ip_group_color(self, ip: str) -> QColor:
        """Assign alternating background tint per unique encoder IP."""
        if ip not in self._ip_group_index:
            self._ip_group_index[ip] = len(self._ip_group_index)
        return _GROUP_COLORS[self._ip_group_index[ip] % 2]

    def _tint_row(self, row: int, color: QColor):
        """Apply a subtle background tint to all non-widget cells in a row."""
        brush = QBrush(color)
        for col in range(NUM_COLS):
            item = self.table.item(row, col)
            if item:
                item.setBackground(brush)

    def _build_rtsp_url_from_row(self, row: int) -> str:
        """Build full RTSP URL from inline fields for a given row."""
        ip_item = self.table.item(row, COL_IP)
        ip = ip_item.text() if ip_item else ""
        mfg_item = self.table.item(row, COL_MFG)
        vendor = mfg_item.text() if mfg_item else ""
        ch_item = self.table.item(row, COL_CH)
        channel = 1
        try:
            channel = int(ch_item.text()) if ch_item else 1
        except (ValueError, TypeError):
            channel = 1

        port_edit = self._get_row_edit(row, COL_PORT)
        port = 554
        try:
            port = int(port_edit.text()) if port_edit else 554
        except (ValueError, TypeError):
            port = 554

        user_edit = self._get_row_edit(row, COL_USER)
        user = user_edit.text().strip() if user_edit else ""
        pass_edit = self._get_row_edit(row, COL_PASS)
        pw = pass_edit.text().strip() if pass_edit else ""

        rtsp_edit = self._get_row_edit(row, COL_RTSP)
        rtsp_path = rtsp_edit.text().strip() if rtsp_edit else ""

        if rtsp_path.lower().startswith("rtsp://") or rtsp_path.lower().startswith("rtsps://"):
            return rtsp_path

        return build_rtsp_url(vendor if vendor and vendor != "Unknown" else "default",
                              ip, port, channel, user, pw)

    def _insert_device_row(self, device: dict, channel: int = 1) -> int:
        """Insert a single row for a device/channel and return the row index."""
        row = self.table.rowCount()
        self.table.insertRow(row)

        ip = device.get("ip", "")
        vendor = device.get("manufacturer", "")
        likely = device.get("likely_camera", False)
        is_encoder = device.get("is_encoder", False)
        encoder_ch_count = device.get("encoder_channels", 1)
        defaults = get_vendor_defaults(vendor or "default")
        group_color = self._get_ip_group_color(ip)

        # Col 0: checkbox
        chk = QCheckBox()
        chk.setChecked(likely)
        chk.stateChanged.connect(self._update_add_button)
        chk_w = QWidget()
        chk_lay = QHBoxLayout(chk_w)
        chk_lay.addWidget(chk)
        chk_lay.setAlignment(Qt.AlignmentFlag.AlignCenter)
        chk_lay.setContentsMargins(0, 0, 0, 0)
        self.table.setCellWidget(row, COL_CHK, chk_w)

        # Col 1: IP (with encoder badge on first channel)
        ip_display = ip
        if is_encoder and channel == 1:
            ip_display = f"\U0001F4E1 {ip}  [{encoder_ch_count}ch]"
        elif is_encoder:
            ip_display = f"  \u2514 {ip}"
        ip_item = QTableWidgetItem(ip_display)
        ip_item.setData(Qt.ItemDataRole.UserRole, ip)
        self.table.setItem(row, COL_IP, ip_item)

        # Col 2: Manufacturer
        mfg_text = vendor or "Unknown"
        if is_encoder and channel == 1:
            mfg_text = f"{vendor} Encoder" if vendor else "Encoder"
        mfg_item = QTableWidgetItem(mfg_text)
        if not vendor:
            mfg_item.setForeground(Qt.GlobalColor.darkGray)
        self.table.setItem(row, COL_MFG, mfg_item)

        # Col 3: Channel
        self.table.setItem(row, COL_CH, QTableWidgetItem(str(channel)))

        # Col 4: Name (editable)
        if is_encoder:
            auto_name = f"{vendor + ' ' if vendor else ''}{ip} Ch{channel}"
        else:
            auto_name = f"{vendor + ' ' if vendor else ''}Camera ({ip})"
        self.table.setCellWidget(row, COL_NAME, self._make_inline_edit(auto_name, "Camera name"))

        # Col 5: RTSP path (editable)
        templates = VENDOR_RTSP_TEMPLATES.get(vendor, VENDOR_RTSP_TEMPLATES["default"])
        tmpl = templates[0] if templates else "rtsp://{ip}:{port}"
        try:
            path_preview = tmpl.format(
                ip=ip, port=device.get("rtsp_port", defaults["port"]),
                channel=channel, channel_hex=f"{channel:x}",
                user=device.get("default_user", defaults["user"]),
                password=device.get("default_password", defaults["password"]),
            )
        except (KeyError, ValueError):
            path_preview = f"rtsp://{ip}:554"
        self.table.setCellWidget(row, COL_RTSP, self._make_inline_edit(path_preview, "rtsp://..."))

        # Col 6: Port (editable)
        port_val = str(device.get("rtsp_port", defaults["port"]))
        self.table.setCellWidget(row, COL_PORT, self._make_inline_edit(port_val, "554"))

        # Col 7: User (editable)
        self.table.setCellWidget(row, COL_USER,
            self._make_inline_edit(device.get("default_user", defaults["user"]), "admin"))

        # Col 8: Pass (editable)
        self.table.setCellWidget(row, COL_PASS,
            self._make_inline_edit(device.get("default_password", defaults["password"]), "pass", echo_password=True))

        # Col 9: Test + Web buttons
        btn_widget = QWidget()
        btn_lay = QHBoxLayout(btn_widget)
        btn_lay.setContentsMargins(0, 0, 0, 0)
        btn_lay.setSpacing(2)

        test_btn = QPushButton("Test")
        test_btn.setStyleSheet(_BTN_TEST)
        test_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        test_btn.clicked.connect(lambda _, r=row: self._on_test_row(r))
        btn_lay.addWidget(test_btn)

        web_btn = QPushButton("\U0001F310")
        web_btn.setToolTip(f"Open {ip} management page")
        web_btn.setStyleSheet(_BTN_WEB)
        web_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        web_btn.setFixedWidth(26)
        web_btn.clicked.connect(lambda _, addr=ip: self._open_device_web(addr))
        btn_lay.addWidget(web_btn)

        self.table.setCellWidget(row, COL_TEST, btn_widget)

        # Col 10: Status
        if is_encoder:
            status_item = QTableWidgetItem(f"\U0001F4E1 Encoder Ch{channel}")
            status_item.setForeground(QColor(_CYAN))
        elif likely and device.get("rtsp_ports"):
            status_item = QTableWidgetItem("\u2022 Ready to test")
            status_item.setForeground(QColor(_YELLOW))
        else:
            status_item = QTableWidgetItem("\u2022 Discovered")
            status_item.setForeground(Qt.GlobalColor.darkGray)
        self.table.setItem(row, COL_STATUS, status_item)

        self._tint_row(row, group_color)

        return row

    # ═══════════════════════════════════════════════════════════════════
    #  Scan lifecycle
    # ═══════════════════════════════════════════════════════════════════

    def showEvent(self, event):
        super().showEvent(event)
        if self._scanner is None and not self._scan_complete:
            QTimer.singleShot(300, self._start_scan)

    def _start_scan(self):
        self._scanner = NetworkCameraScanner(parent=self)
        self._scanner.progress.connect(self._on_scan_progress)
        self._scanner.device_found.connect(self._on_device_found)
        self._scanner.finished_all.connect(self._on_scan_finished)
        self._scanner.error.connect(self._on_scan_error)
        self._scanner.start()
        self.btn_rescan.setEnabled(False)
        self.scan_progress.setRange(0, 0)

    def _on_rescan(self):
        self._stop_scan()
        self._scan_complete = False
        self._discovered.clear()
        self._row_map.clear()
        self._ip_group_index.clear()
        self.table.setRowCount(0)
        self.scan_status.setText("Rescanning\u2026")
        self._update_add_button()
        QTimer.singleShot(200, self._start_scan)

    @Slot(str)
    def _on_scan_progress(self, msg: str):
        self.scan_status.setText(msg)

    @Slot(dict)
    def _on_device_found(self, device: dict):
        ip = device.get("ip", "")
        ch = device.get("channel", 1)
        key = f"{ip}:{ch}"
        if key in self._row_map:
            return

        self._discovered.append(device)
        row = self._insert_device_row(device, ch)
        self._row_map[key] = row
        self._update_add_button()

    @Slot(list)
    def _on_scan_finished(self, _results: list):
        self._scan_complete = True
        self.scan_progress.setRange(0, 1)
        self.scan_progress.setValue(1)
        self.btn_rescan.setEnabled(True)
        count = self.table.rowCount()
        encoder_ips = set()
        for d in self._discovered:
            if d.get("is_encoder"):
                encoder_ips.add(d.get("ip"))
        if count == 0:
            self.scan_status.setText("Scan complete \u2014 no cameras found. Try adding one manually.")
        else:
            enc_msg = ""
            if encoder_ips:
                enc_msg = f"  ({len(encoder_ips)} encoder{'s' if len(encoder_ips) != 1 else ''} auto-expanded)"
            self.scan_status.setText(f"Scan complete \u2014 {count} channel(s) found.{enc_msg}")
        self._scanner = None

    @Slot(str)
    def _on_scan_error(self, msg: str):
        self._scan_complete = True
        self.scan_progress.setRange(0, 1)
        self.scan_progress.setValue(1)
        self.btn_rescan.setEnabled(True)
        self.scan_status.setText(f"Scan error: {msg}")
        self._scanner = None

    # ═══════════════════════════════════════════════════════════════════
    #  Bulk operations
    # ═══════════════════════════════════════════════════════════════════

    def _on_bulk_apply(self):
        """Apply bulk user/pass/port to all checked rows."""
        bulk_u = self.bulk_user.text().strip()
        bulk_p = self.bulk_pass.text().strip()
        bulk_port = self.bulk_port.text().strip()

        for row in range(self.table.rowCount()):
            chk = self._get_row_checkbox(row)
            if not chk or not chk.isChecked():
                continue
            if bulk_u:
                e = self._get_row_edit(row, COL_USER)
                if e:
                    e.setText(bulk_u)
            if bulk_p:
                e = self._get_row_edit(row, COL_PASS)
                if e:
                    e.setText(bulk_p)
            if bulk_port:
                e = self._get_row_edit(row, COL_PORT)
                if e:
                    e.setText(bulk_port)

    def _on_apply_same_ip(self):
        """For each checked row, propagate its user/pass/port to all other rows with the same IP."""
        source_rows: dict[str, int] = {}
        for row in range(self.table.rowCount()):
            chk = self._get_row_checkbox(row)
            if not chk or not chk.isChecked():
                continue
            ip_item = self.table.item(row, COL_IP)
            ip = ip_item.data(Qt.ItemDataRole.UserRole) if ip_item else ""
            if ip and ip not in source_rows:
                source_rows[ip] = row

        for ip, src_row in source_rows.items():
            src_user = self._get_row_edit(src_row, COL_USER)
            src_pass = self._get_row_edit(src_row, COL_PASS)
            src_port = self._get_row_edit(src_row, COL_PORT)
            u = src_user.text() if src_user else ""
            p = src_pass.text() if src_pass else ""
            pt = src_port.text() if src_port else ""

            for row in range(self.table.rowCount()):
                if row == src_row:
                    continue
                ip_item = self.table.item(row, COL_IP)
                row_ip = ip_item.data(Qt.ItemDataRole.UserRole) if ip_item else ""
                if row_ip != ip:
                    continue
                if u:
                    e = self._get_row_edit(row, COL_USER)
                    if e:
                        e.setText(u)
                if p:
                    e = self._get_row_edit(row, COL_PASS)
                    if e:
                        e.setText(p)
                if pt:
                    e = self._get_row_edit(row, COL_PORT)
                    if e:
                        e.setText(pt)
                chk = self._get_row_checkbox(row)
                if chk:
                    chk.setChecked(True)

    # ═══════════════════════════════════════════════════════════════════
    #  Connection testing
    # ═══════════════════════════════════════════════════════════════════

    def _find_test_btn(self, row: int) -> Optional[QPushButton]:
        w = self.table.cellWidget(row, COL_TEST)
        if w:
            for child in w.findChildren(QPushButton):
                if child.text() == "Test":
                    return child
        return None

    @staticmethod
    def _open_device_web(ip: str):
        QDesktopServices.openUrl(QUrl(f"http://{ip}"))

    def _go_to_config(self):
        if callable(self._on_open_config):
            self.accept()
            self._on_open_config()

    def _on_test_row(self, row: int):
        url = self._build_rtsp_url_from_row(row)

        status_item = QTableWidgetItem("\u23F3 Testing\u2026")
        status_item.setForeground(QColor(_CYAN))
        self.table.setItem(row, COL_STATUS, status_item)

        btn = self._find_test_btn(row)
        if btn:
            btn.setEnabled(False)

        def _worker():
            result = test_rtsp_connection(url, timeout_ms=3000)
            QTimer.singleShot(0, lambda: self._on_test_result(row, result))

        t = threading.Thread(target=_worker, daemon=True)
        self._test_threads[row] = t
        t.start()

    def _on_test_result(self, row: int, result: dict):
        self._test_threads.pop(row, None)

        btn = self._find_test_btn(row)
        if btn:
            btn.setEnabled(True)

        if result.get("ok"):
            w = result.get("width", 0)
            h = result.get("height", 0)
            fps = result.get("fps", 0)
            res_text = f"{w}x{h}" if w and h else ""
            fps_text = f" {fps}fps" if fps else ""
            item = QTableWidgetItem(f"\u2713 {res_text}{fps_text}")
            item.setForeground(QColor(_GREEN))
        else:
            err = result.get("error", "Failed")
            short = err[:40]
            if "401" in err or "auth" in err.lower():
                short = "Auth required"
            item = QTableWidgetItem(f"\u2717 {short}")
            item.setForeground(QColor(_RED))

        self.table.setItem(row, COL_STATUS, item)

    def _on_test_all_checked(self):
        for row in range(self.table.rowCount()):
            chk = self._get_row_checkbox(row)
            if chk and chk.isChecked():
                self._on_test_row(row)

    # ═══════════════════════════════════════════════════════════════════
    #  Add button / limit tracking
    # ═══════════════════════════════════════════════════════════════════

    def _update_add_button(self, _=None):
        checked = self._count_checked()
        existing, limit, remaining = _get_camera_slots()

        if remaining <= 0:
            self.btn_add_selected.setEnabled(False)
            self.btn_add_selected.setText("Camera limit reached")
            self.limit_banner.setText(
                f"\u26a0  The public beta supports {limit} cameras. "
                f"All slots are in use."
            )
            self.limit_banner.setStyleSheet(f"color: {_YELLOW}; font-size: 11px;")
            self.limit_banner.setVisible(True)
            self.beta_limit_note.setVisible(True)
        elif checked > remaining:
            self.btn_add_selected.setEnabled(True)
            self.btn_add_selected.setText(f"Add {remaining} of {checked}")
            self.limit_banner.setText(
                f"\u26a0  Your plan allows {limit} cameras ({remaining} slot{'s' if remaining != 1 else ''} left). "
                f"Only the first {remaining} will be added."
            )
            self.limit_banner.setStyleSheet(f"color: {_YELLOW}; font-size: 11px;")
            self.limit_banner.setVisible(True)
            self.beta_limit_note.setVisible(True)
        else:
            if checked > 0:
                self.btn_add_selected.setEnabled(True)
                self.btn_add_selected.setText(f"Add {checked} Camera{'s' if checked != 1 else ''}")
            else:
                self.btn_add_selected.setEnabled(False)
                self.btn_add_selected.setText("Add Selected")
            self.limit_banner.setVisible(False)
            self.beta_limit_note.setVisible(False)

    def _count_checked(self) -> int:
        count = 0
        for row in range(self.table.rowCount()):
            chk = self._get_row_checkbox(row)
            if chk and chk.isChecked():
                count += 1
        return count

    # ═══════════════════════════════════════════════════════════════════
    #  Save cameras
    # ═══════════════════════════════════════════════════════════════════

    def _on_add_selected(self):
        entries = []
        for row in range(self.table.rowCount()):
            chk = self._get_row_checkbox(row)
            if not chk or not chk.isChecked():
                continue

            ip_item = self.table.item(row, COL_IP)
            ip = ip_item.data(Qt.ItemDataRole.UserRole) if ip_item else ""
            if not ip:
                ip = ip_item.text().strip().split()[-1] if ip_item else ""
            if not ip:
                continue

            ch_item = self.table.item(row, COL_CH)
            ch = 1
            try:
                ch = int(ch_item.text()) if ch_item else 1
            except (ValueError, TypeError):
                ch = 1

            name_edit = self._get_row_edit(row, COL_NAME)
            name = name_edit.text().strip() if name_edit else f"Camera ({ip})"

            user_edit = self._get_row_edit(row, COL_USER)
            user = user_edit.text().strip() if user_edit else ""
            pass_edit = self._get_row_edit(row, COL_PASS)
            pw = pass_edit.text().strip() if pass_edit else ""

            rtsp_url = self._build_rtsp_url_from_row(row)

            entries.append({
                "id": str(uuid.uuid4()),
                "name": name,
                "rtsp_url": rtsp_url,
                "username": user,
                "password": pw,
                "ip_address": ip,
                "ip": ip,
                "location": "",
                "enabled": True,
                "motion_detection": True,
                "record_motion": False,
            })

        existing, limit, remaining = _get_camera_slots()
        if remaining <= 0:
            entries = []
        elif len(entries) > remaining:
            entries = entries[:remaining]

        saved = self._save_cameras(entries)
        if saved:
            self.scan_status.setStyleSheet(f"color: {_GREEN};")
            self.scan_status.setText(f"\u2713 {saved} camera{'s' if saved != 1 else ''} added successfully.")
            self._update_add_button()
            for row in range(self.table.rowCount()):
                chk = self._get_row_checkbox(row)
                if chk:
                    chk.setChecked(False)

    def _on_add_manual(self):
        name = self.manual_name.text().strip()
        url = self.manual_url.text().strip()
        if not name:
            self.manual_error.setText("Please enter a camera name.")
            return
        if not url:
            self.manual_error.setText("Please enter an RTSP URL.")
            return
        if not (url.lower().startswith("rtsp://") or url.lower().startswith("rtsps://")):
            self.manual_error.setText("URL must start with rtsp:// or rtsps://")
            return

        self.manual_error.setText("")
        existing, limit, remaining = _get_camera_slots()
        if remaining <= 0:
            self.manual_error.setText(f"Camera limit reached ({limit}/{limit}).")
            return

        entry = {
            "id": str(uuid.uuid4()),
            "name": name,
            "rtsp_url": url,
            "location": "",
            "enabled": True,
            "motion_detection": True,
            "record_motion": False,
        }

        saved = self._save_cameras([entry])
        if saved:
            self.manual_name.clear()
            self.manual_url.clear()
            self.manual_error.setStyleSheet(f"color: {_GREEN}; font-size: 11px;")
            self.manual_error.setText(f"\u2713 \"{name}\" added.")
            self._update_add_button()

    def _save_cameras(self, entries: list) -> int:
        if not entries:
            return 0
        try:
            existing = []
            if self.cameras_json_path.exists():
                with open(self.cameras_json_path, "r") as f:
                    raw = json.load(f)
                if isinstance(raw, list):
                    existing = raw
            existing.extend(entries)
            self.cameras_json_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.cameras_json_path, "w") as f:
                json.dump(existing, f, indent=2)
            for e in entries:
                logger.info("Scanner: saved camera '%s' (id=%s)", e.get("name"), e.get("id"))
            return len(entries)
        except Exception as exc:
            logger.error("Scanner: failed to save cameras: %s", exc)
            self.scan_status.setStyleSheet(f"color: {_RED};")
            self.scan_status.setText(f"Error saving: {exc}")
            return 0

    # ═══════════════════════════════════════════════════════════════════
    #  Cleanup
    # ═══════════════════════════════════════════════════════════════════

    def _stop_scan(self):
        if self._scanner and self._scanner.isRunning():
            self._scanner.request_stop()
            self._scanner.quit()
            self._scanner.wait(3000)
            self._scanner = None

    def closeEvent(self, event):
        self._stop_scan()
        super().closeEvent(event)
