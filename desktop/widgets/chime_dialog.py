"""Chime manager dialog and trigger editor for the camera widget."""

from __future__ import annotations

import logging
import os
import time
import tempfile
import threading
import wave
from pathlib import Path
from typing import Dict, List, Optional

from PySide6.QtCore import Qt, QTimer, Signal, QObject
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSlider,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from core.chime_manager import ChimeStore, get_chime_store, get_chime_trigger_engine

log = logging.getLogger(__name__)

try:
    import sounddevice as sd  # type: ignore
    _SD_AVAILABLE = True
except Exception:
    sd = None  # type: ignore
    _SD_AVAILABLE = False

try:
    import numpy as np  # type: ignore
except Exception:
    np = None  # type: ignore


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _list_output_devices() -> List[str]:
    """Return a list of sounddevice output device name strings."""
    if not _SD_AVAILABLE or sd is None:
        return []
    try:
        devices = sd.query_devices()
        return [
            d["name"]
            for d in devices
            if d.get("max_output_channels", 0) > 0
        ]
    except Exception:
        return []


def _type_badge(chime_type: str) -> str:
    return {"builtin": "[built-in]", "file": "[file]", "recorded": "[rec]"}.get(chime_type, f"[{chime_type}]")


# ---------------------------------------------------------------------------
# Recording helper
# ---------------------------------------------------------------------------

class _RecordDialog(QDialog):
    """Capture up to `max_sec` seconds from the microphone and save as WAV."""

    def __init__(self, save_dir: Path, parent=None, max_sec: int = 10):
        super().__init__(parent)
        self.setWindowTitle("Record Chime")
        self._save_dir = save_dir
        self._max_sec = int(max_sec)
        self._recorded_path: Optional[Path] = None
        self._recording = False
        self._audio_data = None

        layout = QVBoxLayout(self)

        self._status_label = QLabel("Press Record to start (up to 10 s).")
        layout.addWidget(self._status_label)

        name_row = QHBoxLayout()
        name_row.addWidget(QLabel("Name:"))
        self._name_edit = QLineEdit()
        self._name_edit.setPlaceholderText("My Chime")
        name_row.addWidget(self._name_edit, 1)
        layout.addLayout(name_row)

        btn_row = QHBoxLayout()
        self._rec_btn = QPushButton("⏺ Record")
        self._rec_btn.clicked.connect(self._toggle_record)
        btn_row.addWidget(self._rec_btn)
        self._stop_btn = QPushButton("⏹ Stop")
        self._stop_btn.setEnabled(False)
        self._stop_btn.clicked.connect(self._stop)
        btn_row.addWidget(self._stop_btn)
        layout.addLayout(btn_row)

        self._timer = QTimer(self)
        self._timer.setInterval(100)
        self._timer.timeout.connect(self._tick)
        self._start_time: float = 0.0

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self._on_accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _toggle_record(self):
        if not _SD_AVAILABLE or sd is None:
            QMessageBox.warning(self, "Not Available", "sounddevice is not installed. Cannot record audio.")
            return
        if self._recording:
            self._stop()
            return
        try:
            if np is None:
                QMessageBox.warning(self, "Not Available", "numpy is not installed. Cannot record audio.")
                return
            self._audio_data = sd.rec(
                int(self._max_sec * 44100),
                samplerate=44100,
                channels=1,
                dtype="int16",
            )
            self._recording = True
            self._start_time = time.monotonic()
            self._rec_btn.setText("⏹ Recording…")
            self._stop_btn.setEnabled(True)
            self._timer.start()
        except Exception as exc:
            QMessageBox.warning(self, "Record Error", str(exc))

    def _stop(self):
        if not self._recording:
            return
        self._recording = False
        self._timer.stop()
        self._rec_btn.setText("⏺ Record")
        self._stop_btn.setEnabled(False)
        if _SD_AVAILABLE and sd is not None:
            try:
                sd.stop()
            except Exception:
                pass
        self._status_label.setText("Recording stopped. Click OK to save.")

    def _tick(self):
        elapsed = time.monotonic() - self._start_time
        remaining = max(0.0, self._max_sec - elapsed)
        self._status_label.setText(f"Recording… {remaining:.1f}s remaining")
        if elapsed >= self._max_sec:
            self._stop()

    def _on_accept(self):
        if self._audio_data is None:
            QMessageBox.warning(self, "No Recording", "Please record audio first.")
            return
        name = self._name_edit.text().strip() or f"Recording {int(time.time())}"
        self._save_dir.mkdir(parents=True, exist_ok=True)
        safe_name = "".join(c if c.isalnum() or c in " _-" else "_" for c in name)
        out_path = self._save_dir / f"{safe_name}_{int(time.time())}.wav"
        try:
            if np is not None:
                data = np.trim_zeros(self._audio_data.flatten())
            else:
                data = self._audio_data.flatten()
            with wave.open(str(out_path), "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(44100)
                if np is not None:
                    wf.writeframes(data.astype(np.int16).tobytes())
                else:
                    wf.writeframes(bytes(data))
        except Exception as exc:
            QMessageBox.warning(self, "Save Error", str(exc))
            return
        self._recorded_path = out_path
        self._chime_name = name
        self.accept()

    @property
    def recorded_path(self) -> Optional[Path]:
        return self._recorded_path

    @property
    def chime_name(self) -> str:
        return getattr(self, "_chime_name", "")


# ---------------------------------------------------------------------------
# Chime list item widget
# ---------------------------------------------------------------------------

class _ChimeRowWidget(QWidget):
    """A single row in the chime list: name, badge, volume slider, play/delete."""

    play_requested = Signal(str)
    delete_requested = Signal(str)

    def __init__(self, chime: Dict, parent=None):
        super().__init__(parent)
        self._chime_id = chime["id"]
        self._readonly = bool(chime.get("readonly", False))

        layout = QHBoxLayout(self)
        layout.setContentsMargins(4, 2, 4, 2)
        layout.setSpacing(6)

        badge = QLabel(_type_badge(chime.get("type", "file")))
        badge.setFixedWidth(62)
        badge.setStyleSheet("color: #7ec8e3; font-size: 11px;")
        layout.addWidget(badge)

        name_lbl = QLabel(chime.get("name", ""))
        name_lbl.setMinimumWidth(80)
        name_lbl.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        layout.addWidget(name_lbl, 1)

        play_btn = QPushButton("▶")
        play_btn.setFixedWidth(28)
        play_btn.setToolTip("Preview chime")
        play_btn.clicked.connect(lambda: self.play_requested.emit(self._chime_id))
        layout.addWidget(play_btn)

        if not self._readonly:
            del_btn = QPushButton("🗑")
            del_btn.setFixedWidth(28)
            del_btn.setToolTip("Delete chime")
            del_btn.clicked.connect(lambda: self.delete_requested.emit(self._chime_id))
            layout.addWidget(del_btn)


# ---------------------------------------------------------------------------
# ChimeManagerDialog
# ---------------------------------------------------------------------------

class ChimeManagerDialog(QDialog):
    """
    Dialog for managing chimes: listing, adding (file/recorded), and previewing.
    """

    def __init__(self, store: Optional[ChimeStore] = None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Chime Manager")
        self.setMinimumSize(480, 400)
        self._store = store or get_chime_store()

        self._selected_device: Optional[str] = None

        root = QVBoxLayout(self)
        root.setSpacing(8)

        # Output device selector
        dev_row = QHBoxLayout()
        dev_row.addWidget(QLabel("Output device:"))
        self._dev_combo = QComboBox()
        self._dev_combo.addItem("System default", None)
        for dev_name in _list_output_devices():
            self._dev_combo.addItem(dev_name, dev_name)
        self._dev_combo.currentIndexChanged.connect(self._on_device_changed)
        dev_row.addWidget(self._dev_combo, 1)
        root.addLayout(dev_row)

        # Chime list
        list_group = QGroupBox("Chimes")
        list_layout = QVBoxLayout(list_group)
        self._list_widget = QListWidget()
        self._list_widget.setAlternatingRowColors(True)
        list_layout.addWidget(self._list_widget)
        root.addWidget(list_group, 1)

        # Add chime controls
        add_group = QGroupBox("Add Chime")
        add_layout = QVBoxLayout(add_group)

        name_row = QHBoxLayout()
        name_row.addWidget(QLabel("Name:"))
        self._name_edit = QLineEdit()
        self._name_edit.setPlaceholderText("Chime name")
        name_row.addWidget(self._name_edit, 1)
        add_layout.addLayout(name_row)

        btn_row = QHBoxLayout()
        add_file_btn = QPushButton("📂 Add from file…")
        add_file_btn.clicked.connect(self._add_from_file)
        btn_row.addWidget(add_file_btn)
        record_btn = QPushButton("🎤 Record chime…")
        record_btn.clicked.connect(self._record_chime)
        btn_row.addWidget(record_btn)
        add_layout.addLayout(btn_row)

        root.addWidget(add_group)

        # OK/Cancel
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        buttons.rejected.connect(self.reject)
        root.addWidget(buttons)

        self._refresh_list()

    def _on_device_changed(self, _idx: int) -> None:
        self._selected_device = self._dev_combo.currentData()

    def _refresh_list(self) -> None:
        self._list_widget.clear()
        for chime in self._store.list_chimes():
            item = QListWidgetItem(self._list_widget)
            row_widget = _ChimeRowWidget(chime)
            row_widget.play_requested.connect(self._play_chime)
            row_widget.delete_requested.connect(self._delete_chime)
            item.setSizeHint(row_widget.sizeHint())
            self._list_widget.addItem(item)
            self._list_widget.setItemWidget(item, row_widget)

    def _play_chime(self, chime_id: str) -> None:
        self._store.play_chime(chime_id, output_device=self._selected_device)

    def _delete_chime(self, chime_id: str) -> None:
        reply = QMessageBox.question(
            self,
            "Delete Chime",
            "Delete this chime?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if reply == QMessageBox.StandardButton.Yes:
            self._store.delete_chime(chime_id)
            self._refresh_list()

    def _add_from_file(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Audio File",
            "",
            "Audio Files (*.wav *.mp3 *.ogg *.flac *.aiff);;All Files (*)",
        )
        if not path:
            return
        name = self._name_edit.text().strip() or Path(path).stem
        self._store.add_chime(name, path, chime_type="file")
        self._name_edit.clear()
        self._refresh_list()

    def _record_chime(self) -> None:
        from core.paths import get_data_dir
        chimes_dir = get_data_dir() / "chimes"
        dlg = _RecordDialog(chimes_dir, parent=self)
        if dlg.exec() == QDialog.DialogCode.Accepted and dlg.recorded_path:
            name = self._name_edit.text().strip() or dlg.chime_name or "Recorded Chime"
            self._store.add_chime(name, str(dlg.recorded_path), chime_type="recorded")
            self._name_edit.clear()
            self._refresh_list()

    def selected_device(self) -> Optional[str]:
        return self._selected_device


# ---------------------------------------------------------------------------
# ChimeTriggerEditor  (embeddable widget)
# ---------------------------------------------------------------------------

class ChimeTriggerEditor(QWidget):
    """
    Embeddable widget for configuring a single chime trigger rule.
    Can be used stand-alone (in a dialog) or embedded into another widget.
    """

    def __init__(self, camera_id: str, store: Optional[ChimeStore] = None, parent=None):
        super().__init__(parent)
        self._camera_id = camera_id
        self._store = store or get_chime_store()
        self._engine = get_chime_trigger_engine(camera_id)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        # --- Existing triggers list ---
        existing_group = QGroupBox("Active Chime Triggers")
        existing_layout = QVBoxLayout(existing_group)
        self._triggers_list = QListWidget()
        self._triggers_list.setAlternatingRowColors(True)
        existing_layout.addWidget(self._triggers_list)
        remove_btn = QPushButton("🗑 Remove Selected")
        remove_btn.clicked.connect(self._remove_selected_trigger)
        existing_layout.addWidget(remove_btn)
        layout.addWidget(existing_group)

        # --- New trigger form ---
        new_group = QGroupBox("Add New Trigger")
        form = QFormLayout(new_group)

        # Chime picker
        self._chime_combo = QComboBox()
        self._refresh_chimes()
        form.addRow("Chime:", self._chime_combo)

        # Preview
        preview_btn = QPushButton("▶ Preview")
        preview_btn.clicked.connect(self._preview_chime)
        form.addRow("", preview_btn)

        # Output device
        self._device_combo = QComboBox()
        self._device_combo.addItem("System default", None)
        for dev in _list_output_devices():
            self._device_combo.addItem(dev, dev)
        form.addRow("Output device:", self._device_combo)

        # Event types checkboxes
        evt_widget = QWidget()
        evt_layout = QHBoxLayout(evt_widget)
        evt_layout.setContentsMargins(0, 0, 0, 0)
        self._evt_checks: Dict[str, QCheckBox] = {}
        for etype in ("zone", "line", "tag", "detection"):
            cb = QCheckBox(etype.capitalize())
            cb.setChecked(True)
            self._evt_checks[etype] = cb
            evt_layout.addWidget(cb)
        evt_layout.addStretch()
        form.addRow("Event types:", evt_widget)

        # Shape ID filter
        self._shape_edit = QLineEdit()
        self._shape_edit.setPlaceholderText("(blank = any shape)")
        self._shape_edit.setToolTip("Comma-separated shape IDs to match, or leave blank to match all shapes.")
        form.addRow("Shape IDs:", self._shape_edit)

        # Cooldown
        self._cooldown_spin = QDoubleSpinBox()
        self._cooldown_spin.setRange(0.5, 3600.0)
        self._cooldown_spin.setValue(10.0)
        self._cooldown_spin.setSuffix(" s")
        self._cooldown_spin.setDecimals(1)
        form.addRow("Cooldown:", self._cooldown_spin)

        # Volume override
        vol_widget = QWidget()
        vol_layout = QHBoxLayout(vol_widget)
        vol_layout.setContentsMargins(0, 0, 0, 0)
        self._vol_override_check = QCheckBox("Override volume")
        self._vol_override_check.stateChanged.connect(self._on_vol_override_toggle)
        self._vol_spin = QDoubleSpinBox()
        self._vol_spin.setRange(0.0, 1.0)
        self._vol_spin.setValue(0.8)
        self._vol_spin.setSingleStep(0.05)
        self._vol_spin.setDecimals(2)
        self._vol_spin.setEnabled(False)
        vol_layout.addWidget(self._vol_override_check)
        vol_layout.addWidget(self._vol_spin, 1)
        form.addRow("Volume:", vol_widget)

        add_btn = QPushButton("➕ Add Trigger")
        add_btn.clicked.connect(self._add_trigger)
        form.addRow("", add_btn)

        layout.addWidget(new_group)

        self._refresh_triggers_list()

    def _on_vol_override_toggle(self, state: int) -> None:
        self._vol_spin.setEnabled(state == Qt.CheckState.Checked.value)

    def _refresh_chimes(self) -> None:
        self._chime_combo.clear()
        for chime in self._store.list_chimes():
            label = f"{_type_badge(chime.get('type', ''))} {chime['name']}"
            self._chime_combo.addItem(label, chime["id"])

    def _refresh_triggers_list(self) -> None:
        self._triggers_list.clear()
        for trig in self._engine.get_triggers():
            chime = self._store.get_chime(trig.get("chime_id", ""))
            chime_name = chime["name"] if chime else "(unknown)"
            etypes = ", ".join(trig.get("event_types") or ["any"])
            cooldown = float(trig.get("cooldown_sec") or 10.0)
            enabled = "✔" if trig.get("enabled", True) else "✘"
            label = f"{enabled} {chime_name} | {etypes} | cooldown {cooldown:.0f}s"
            item = QListWidgetItem(label)
            item.setData(Qt.ItemDataRole.UserRole, trig.get("id"))
            self._triggers_list.addItem(item)

    def _preview_chime(self) -> None:
        chime_id = self._chime_combo.currentData()
        if chime_id:
            device = self._device_combo.currentData()
            self._store.play_chime(chime_id, output_device=device)

    def _add_trigger(self) -> None:
        chime_id = self._chime_combo.currentData()
        if not chime_id:
            QMessageBox.warning(self, "No Chime", "Please select a chime.")
            return
        event_types = [k for k, cb in self._evt_checks.items() if cb.isChecked()]
        if not event_types:
            QMessageBox.warning(self, "No Events", "Please select at least one event type.")
            return
        shape_ids_raw = self._shape_edit.text().strip()
        shape_ids = [s.strip() for s in shape_ids_raw.split(",") if s.strip()] if shape_ids_raw else []
        device = self._device_combo.currentData()
        volume = self._vol_spin.value() if self._vol_override_check.isChecked() else None
        trigger = {
            "chime_id": chime_id,
            "event_types": event_types,
            "shape_ids": shape_ids,
            "cooldown_sec": self._cooldown_spin.value(),
            "output_device": device,
            "volume": volume,
            "enabled": True,
        }
        self._engine.add_trigger(trigger)
        self._refresh_triggers_list()

    def _remove_selected_trigger(self) -> None:
        item = self._triggers_list.currentItem()
        if item is None:
            return
        trigger_id = item.data(Qt.ItemDataRole.UserRole)
        if trigger_id:
            self._engine.remove_trigger(trigger_id)
            self._refresh_triggers_list()


# ---------------------------------------------------------------------------
# ChimeTriggerDialog  (standalone dialog wrapping ChimeTriggerEditor)
# ---------------------------------------------------------------------------

class ChimeTriggerDialog(QDialog):
    """Standalone dialog for adding/removing chime triggers for a camera."""

    def __init__(self, camera_id: str, store: Optional[ChimeStore] = None, parent=None):
        super().__init__(parent)
        self.setWindowTitle(f"Chime Triggers — {camera_id}")
        self.setMinimumSize(460, 540)

        layout = QVBoxLayout(self)
        self._editor = ChimeTriggerEditor(camera_id, store=store, parent=self)
        layout.addWidget(self._editor)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)


# ---------------------------------------------------------------------------
# Shape-type metadata used by the two-step "Add Chime Trigger" flow
# ---------------------------------------------------------------------------

_KIND_ICONS: Dict[str, str] = {
    "zone": "🔷",
    "line": "📏",
    "tag": "🏷️",
}

_KIND_LABELS: Dict[str, str] = {
    "zone": "Zone",
    "line": "Line",
    "tag": "Tag",
}

# (event_type_key, display_label) per shape kind
_KIND_EVENTS: Dict[str, List[tuple]] = {
    "zone": [
        ("enter", "Enter"),
        ("exit", "Exit"),
        ("any_motion", "Any motion"),
    ],
    "line": [
        ("cross", "Cross (any direction)"),
        ("cross_lr", "Cross left→right"),
        ("cross_rl", "Cross right→left"),
    ],
    "tag": [
        ("tag_match", "Detection match"),
        ("tag_any", "Any detection"),
    ],
}


# ---------------------------------------------------------------------------
# ChimeShapePickerDialog  (Step 1 of 2-step "Add Chime Trigger" flow)
# ---------------------------------------------------------------------------

class ChimeShapePickerDialog(QDialog):
    """
    Step 1 — pick which shape (zone/line/tag) to configure a chime trigger for.

    Pass in the camera's current shapes list.  If the list is empty the dialog
    shows a greyed-out hint and disables the Select button.
    """

    def __init__(self, shapes: List[Dict], parent=None):
        super().__init__(parent)
        self.setWindowTitle("Add Chime Trigger — Select Shape")
        self.setMinimumSize(380, 320)
        self._selected_shape: Optional[Dict] = None

        layout = QVBoxLayout(self)
        layout.setSpacing(8)

        layout.addWidget(QLabel("Select a zone, line, or tag to trigger a chime:"))

        self._list = QListWidget()
        self._list.setAlternatingRowColors(True)
        layout.addWidget(self._list, 1)

        self._empty_label = QLabel(
            "No zones, lines, or tags configured —\nright-click to add shapes first."
        )
        self._empty_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._empty_label.setStyleSheet("color: grey; font-style: italic;")
        layout.addWidget(self._empty_label)

        self._buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        self._ok_btn = self._buttons.button(QDialogButtonBox.StandardButton.Ok)
        self._ok_btn.setText("Select")
        self._buttons.accepted.connect(self._on_accept)
        self._buttons.rejected.connect(self.reject)
        layout.addWidget(self._buttons)

        self._populate(shapes)
        self._list.itemDoubleClicked.connect(lambda _item: self._on_accept())
        self._list.currentRowChanged.connect(self._on_row_changed)

    def _populate(self, shapes: List[Dict]) -> None:
        has_shapes = bool(shapes)
        self._list.setVisible(has_shapes)
        self._empty_label.setVisible(not has_shapes)
        self._ok_btn.setEnabled(False)

        for shape in shapes:
            kind = shape.get("kind", "")
            label = shape.get("label") or shape.get("name") or shape.get("id", "")
            icon = _KIND_ICONS.get(kind, "◼")
            kind_text = _KIND_LABELS.get(kind, kind.capitalize() if kind else "Shape")
            item = QListWidgetItem(f"{icon} {kind_text}: {label}")
            item.setData(Qt.ItemDataRole.UserRole, shape)
            self._list.addItem(item)

        if has_shapes:
            self._list.setCurrentRow(0)
            self._ok_btn.setEnabled(True)

    def _on_row_changed(self, row: int) -> None:
        self._ok_btn.setEnabled(row >= 0)

    def _on_accept(self) -> None:
        item = self._list.currentItem()
        if item is None:
            return
        self._selected_shape = item.data(Qt.ItemDataRole.UserRole)
        self.accept()

    def selected_shape(self) -> Optional[Dict]:
        return self._selected_shape


# ---------------------------------------------------------------------------
# ChimeTriggerConfigDialog  (Step 2 of 2-step "Add Chime Trigger" flow)
# ---------------------------------------------------------------------------

class ChimeTriggerConfigDialog(QDialog):
    """
    Step 2 — configure and save a chime trigger for a specific shape.

    Shows event-type checkboxes appropriate to the shape kind, a chime picker,
    cooldown, volume override, and output device.  On OK the trigger is saved
    via the camera's ChimeTriggerEngine.
    """

    def __init__(
        self,
        shape: Dict,
        camera_id: str,
        store: Optional[ChimeStore] = None,
        parent=None,
    ):
        super().__init__(parent)
        self._shape = shape
        self._camera_id = camera_id
        self._store = store or get_chime_store()
        self._engine = get_chime_trigger_engine(camera_id)

        kind = shape.get("kind", "")
        shape_label = shape.get("label") or shape.get("name") or shape.get("id", "")
        kind_text = _KIND_LABELS.get(kind, kind.capitalize() if kind else "Shape")

        self.setWindowTitle(f"Add Chime Trigger — {kind_text}: {shape_label}")
        self.setMinimumSize(440, 420)

        layout = QVBoxLayout(self)
        layout.setSpacing(10)

        # Header
        header = QLabel(f"Configuring trigger for: {kind_text} \"{shape_label}\"")
        header.setStyleSheet("font-weight: bold; font-size: 13px;")
        layout.addWidget(header)

        form = QFormLayout()
        form.setSpacing(8)

        # Event-type checkboxes (kind-specific)
        events = _KIND_EVENTS.get(kind, [])
        evt_widget = QWidget()
        evt_layout = QVBoxLayout(evt_widget)
        evt_layout.setContentsMargins(0, 0, 0, 0)
        evt_layout.setSpacing(3)
        self._evt_checks: Dict[str, QCheckBox] = {}
        for idx, (key, text) in enumerate(events):
            cb = QCheckBox(text)
            cb.setChecked(idx == 0)  # default: first option only
            self._evt_checks[key] = cb
            evt_layout.addWidget(cb)
        if not events:
            evt_layout.addWidget(QLabel("(no specific event types for this shape kind)"))
        form.addRow("Events:", evt_widget)

        # Chime picker
        self._chime_combo = QComboBox()
        for chime in self._store.list_chimes():
            badge = _type_badge(chime.get("type", ""))
            self._chime_combo.addItem(f"{badge} {chime['name']}", chime["id"])
        if self._chime_combo.count() == 0:
            self._chime_combo.addItem("(no chimes — add via Manage Chimes…)", None)
        form.addRow("Chime:", self._chime_combo)

        # Preview
        preview_btn = QPushButton("▶ Preview")
        preview_btn.clicked.connect(self._preview_chime)
        form.addRow("", preview_btn)

        # Output device
        self._device_combo = QComboBox()
        self._device_combo.addItem("System default", None)
        for dev in _list_output_devices():
            self._device_combo.addItem(dev, dev)
        form.addRow("Output device:", self._device_combo)

        # Cooldown
        self._cooldown_spin = QSpinBox()
        self._cooldown_spin.setRange(1, 300)
        self._cooldown_spin.setValue(5)
        self._cooldown_spin.setSuffix(" s")
        form.addRow("Cooldown:", self._cooldown_spin)

        # Volume override
        vol_widget = QWidget()
        vol_layout = QHBoxLayout(vol_widget)
        vol_layout.setContentsMargins(0, 0, 0, 0)
        vol_layout.setSpacing(6)
        self._vol_override_check = QCheckBox("Override volume")
        self._vol_override_check.stateChanged.connect(self._on_vol_override_toggle)
        self._vol_slider = QSlider(Qt.Orientation.Horizontal)
        self._vol_slider.setRange(0, 100)
        self._vol_slider.setValue(80)
        self._vol_slider.setEnabled(False)
        self._vol_pct_label = QLabel("80%")
        self._vol_pct_label.setFixedWidth(38)
        self._vol_pct_label.setEnabled(False)
        self._vol_slider.valueChanged.connect(
            lambda v: self._vol_pct_label.setText(f"{v}%")
        )
        vol_layout.addWidget(self._vol_override_check)
        vol_layout.addWidget(self._vol_slider, 1)
        vol_layout.addWidget(self._vol_pct_label)
        form.addRow("Volume:", vol_widget)

        layout.addLayout(form)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self._on_accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _on_vol_override_toggle(self, state: int) -> None:
        enabled = state == Qt.CheckState.Checked.value
        self._vol_slider.setEnabled(enabled)
        self._vol_pct_label.setEnabled(enabled)

    def _preview_chime(self) -> None:
        chime_id = self._chime_combo.currentData()
        if not chime_id:
            return
        device = self._device_combo.currentData()
        try:
            self._store.play_chime(chime_id, output_device=device)
        except Exception as exc:
            log.warning("Chime preview failed: %s", exc)

    def _on_accept(self) -> None:
        chime_id = self._chime_combo.currentData()
        if not chime_id:
            QMessageBox.warning(self, "No Chime", "Please select a chime first.")
            return

        event_types = [k for k, cb in self._evt_checks.items() if cb.isChecked()]
        if self._evt_checks and not event_types:
            QMessageBox.warning(self, "No Events", "Please select at least one event type.")
            return

        volume: Optional[float] = None
        if self._vol_override_check.isChecked():
            volume = self._vol_slider.value() / 100.0

        shape_id = self._shape.get("id", "")
        trigger: Dict = {
            "chime_id": chime_id,
            "event_types": event_types if event_types else ["any"],
            "shape_ids": [shape_id] if shape_id else [],
            "cooldown_sec": float(self._cooldown_spin.value()),
            "output_device": self._device_combo.currentData(),
            "volume": volume,
            "enabled": True,
        }
        try:
            self._engine.add_trigger(trigger)
        except Exception as exc:
            QMessageBox.warning(self, "Save Error", f"Could not save trigger:\n{exc}")
            return
        self.accept()


# ---------------------------------------------------------------------------
# Quick "Test Chime" dialog
# ---------------------------------------------------------------------------

class TestChimeDialog(QDialog):
    """Play a chime immediately for testing — simple chime + device picker."""

    def __init__(self, store: Optional[ChimeStore] = None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Test Chime")
        self.setMinimumWidth(320)
        self._store = store or get_chime_store()

        layout = QVBoxLayout(self)

        form = QFormLayout()
        self._chime_combo = QComboBox()
        for chime in self._store.list_chimes():
            label = f"{_type_badge(chime.get('type', ''))} {chime['name']}"
            self._chime_combo.addItem(label, chime["id"])
        form.addRow("Chime:", self._chime_combo)

        self._device_combo = QComboBox()
        self._device_combo.addItem("System default", None)
        for dev in _list_output_devices():
            self._device_combo.addItem(dev, dev)
        form.addRow("Device:", self._device_combo)

        layout.addLayout(form)

        play_btn = QPushButton("▶ Play")
        play_btn.clicked.connect(self._play)
        layout.addWidget(play_btn)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _play(self) -> None:
        chime_id = self._chime_combo.currentData()
        device = self._device_combo.currentData()
        if chime_id:
            self._store.play_chime(chime_id, output_device=device)
