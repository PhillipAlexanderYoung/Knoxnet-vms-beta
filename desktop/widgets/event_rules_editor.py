"""Event Rules editor dialog (Stage 3 UI)."""

from __future__ import annotations

import uuid
from typing import Any, Dict, List, Optional

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from desktop.utils.event_rules_api import (
    DEFAULT_API_BASE,
    DEFAULT_RULE_COOLDOWN_MS,
    DEFAULT_RULE_COOLDOWN_SEC,
    cooldown_ms_from_sec,
    cooldown_sec_from_ms,
    list_rules,
    save_rule,
    set_rules_enabled,
    _api_get,
)

TRIGGER_TYPES = [
    ("Zone enter", "zone_enter"),
    ("Zone exit", "zone_exit"),
    ("Dwell met", "dwell_met"),
    ("Line cross", "line_cross"),
    ("Track lost", "track_lost"),
    ("Track reacquired", "track_reacquired"),
]

DIRECTIONS = ["", "left_to_right", "right_to_left", "positive", "negative"]

COLOR_BUCKETS = ["", "white", "black", "gray", "red", "green", "blue", "yellow", "brown"]

DAY_LABELS = [
    ("Mon", 0),
    ("Tue", 1),
    ("Wed", 2),
    ("Thu", 3),
    ("Fri", 4),
    ("Sat", 5),
    ("Sun", 6),
]


class EventRulesEditorDialog(QDialog):
    """Staged Event Rules editor: Trigger → Conditions → Actions."""

    def __init__(
        self,
        *,
        camera_id: str,
        api_base: str = DEFAULT_API_BASE,
        shapes: Optional[List[Dict[str, Any]]] = None,
        existing_rule: Optional[Dict[str, Any]] = None,
        parent=None,
    ):
        super().__init__(parent)
        self.camera_id = str(camera_id)
        self.api_base = api_base.rstrip("/")
        self.shapes = list(shapes or [])
        self.saved_rule: Optional[Dict[str, Any]] = None
        self._rule_id = str(existing_rule.get("id")) if existing_rule else None
        self._arm_on_accept = False

        self.setWindowTitle(f"Event Rules — {self.camera_id}")
        self.setMinimumSize(460, 520)
        self.setWindowFlags(Qt.WindowType.Tool | Qt.WindowType.WindowStaysOnTopHint)

        root = QVBoxLayout(self)
        top = QFormLayout()
        self.name_edit = QLineEdit(str((existing_rule or {}).get("name") or "New Event Rule"))
        self.rule_combo = QComboBox()
        self.rule_combo.addItem("(New rule)", "")
        self._reload_rule_combo(select_id=self._rule_id)
        self.rule_combo.currentIndexChanged.connect(self._on_rule_selected)
        top.addRow("Rule", self.rule_combo)
        top.addRow("Name", self.name_edit)
        root.addLayout(top)

        tabs = QTabWidget()
        tabs.addTab(self._build_trigger_tab(existing_rule), "Trigger")
        tabs.addTab(self._build_conditions_tab(existing_rule), "Conditions")
        tabs.addTab(self._build_actions_tab(existing_rule), "Actions")
        root.addWidget(tabs)

        self.status_label = QLabel("")
        self.status_label.setStyleSheet("color: #94a3b8; font-size: 11px;")
        root.addWidget(self.status_label)

        buttons = QDialogButtonBox()
        self.save_btn = buttons.addButton("Save Rule", QDialogButtonBox.ButtonRole.AcceptRole)
        self.arm_btn = buttons.addButton("Save && Arm", QDialogButtonBox.ButtonRole.ActionRole)
        buttons.addButton(QDialogButtonBox.StandardButton.Cancel)
        self.save_btn.clicked.connect(lambda: self._save(arm=False))
        self.arm_btn.clicked.connect(lambda: self._save(arm=True))
        buttons.rejected.connect(self.reject)
        root.addWidget(buttons)

    def _reload_rule_combo(self, select_id: Optional[str] = None) -> None:
        while self.rule_combo.count() > 1:
            self.rule_combo.removeItem(1)
        for rule in list_rules(self.api_base, self.camera_id):
            rid = str(rule.get("id") or "")
            label = str(rule.get("name") or rid)
            self.rule_combo.addItem(label, rid)
            if select_id and rid == select_id:
                self.rule_combo.setCurrentIndex(self.rule_combo.count() - 1)

    def _on_rule_selected(self, _idx: int) -> None:
        rid = str(self.rule_combo.currentData() or "")
        if not rid:
            self._rule_id = None
            return
        try:
            data = _api_get(self.api_base, f"rules/{rid}")
            rule = data.get("data")
            if isinstance(rule, dict):
                self._apply_rule(rule)
                self._rule_id = rid
        except Exception as e:
            self.status_label.setText(f"Failed to load rule: {e}")

    def _build_trigger_tab(self, rule: Optional[Dict[str, Any]]) -> QWidget:
        w = QWidget()
        form = QFormLayout(w)
        self.trigger_combo = QComboBox()
        for label, val in TRIGGER_TYPES:
            self.trigger_combo.addItem(label, val)
        cur = str((rule or {}).get("trigger") or "zone_enter")
        idx = self.trigger_combo.findData(cur)
        if idx >= 0:
            self.trigger_combo.setCurrentIndex(idx)
        form.addRow("Event", self.trigger_combo)

        self.shape_combo = QComboBox()
        self.shape_combo.addItem("(Any shape)", "")
        for sh in self.shapes:
            sid = str(sh.get("id") or "")
            if not sid:
                continue
            kind = str(sh.get("kind") or "shape")
            label = str(sh.get("label") or sid)
            self.shape_combo.addItem(f"{label} ({kind})", sid)
        shape_id = str((rule or {}).get("shape_id") or "")
        sidx = self.shape_combo.findData(shape_id)
        if sidx >= 0:
            self.shape_combo.setCurrentIndex(sidx)
        form.addRow("Shape", self.shape_combo)
        return w

    def _build_conditions_tab(self, rule: Optional[Dict[str, Any]]) -> QWidget:
        w = QWidget()
        form = QFormLayout(w)
        cond = (rule or {}).get("conditions") if isinstance((rule or {}).get("conditions"), dict) else {}
        classes = cond.get("classes") or cond.get("object_classes") or []
        classes_s = ", ".join(str(c) for c in classes) if isinstance(classes, list) else str(classes or "")
        self.classes_edit = QLineEdit(classes_s)
        self.classes_edit.setPlaceholderText("car, truck, person")
        form.addRow("Classes", self.classes_edit)

        self.confidence_spin = QDoubleSpinBox()
        self.confidence_spin.setRange(0.0, 1.0)
        self.confidence_spin.setSingleStep(0.05)
        self.confidence_spin.setValue(float(cond.get("min_confidence", 0.5) or 0.5))
        form.addRow("Min confidence", self.confidence_spin)

        self.direction_combo = QComboBox()
        for d in DIRECTIONS:
            self.direction_combo.addItem(d or "(any)", d)
        cur_dir = str(cond.get("direction") or "")
        didx = self.direction_combo.findData(cur_dir)
        if didx >= 0:
            self.direction_combo.setCurrentIndex(didx)
        form.addRow("Direction", self.direction_combo)

        self.dwell_min_spin = QDoubleSpinBox()
        self.dwell_min_spin.setRange(0.0, 3600.0)
        self.dwell_min_spin.setSingleStep(0.5)
        self.dwell_min_spin.setValue(float(cond.get("dwell_min") or cond.get("dwell_min_sec") or 0.0))
        form.addRow("Dwell min (sec)", self.dwell_min_spin)

        self.color_combo = QComboBox()
        for c in COLOR_BUCKETS:
            self.color_combo.addItem(c or "(any color)", c)
        cur_color = str(cond.get("color") or cond.get("dominant_color") or "")
        cidx = self.color_combo.findData(cur_color)
        if cidx >= 0:
            self.color_combo.setCurrentIndex(cidx)
        form.addRow("Object color", self.color_combo)

        self.count_min_spin = QDoubleSpinBox()
        self.count_min_spin.setRange(0.0, 999.0)
        self.count_min_spin.setDecimals(0)
        self.count_min_spin.setSingleStep(1.0)
        self.count_min_spin.setSpecialValueText("(none)")
        self.count_min_spin.setValue(float(cond.get("count_min") or 0.0))
        form.addRow("Count min (in zone)", self.count_min_spin)

        self.count_max_spin = QDoubleSpinBox()
        self.count_max_spin.setRange(0.0, 999.0)
        self.count_max_spin.setDecimals(0)
        self.count_max_spin.setSingleStep(1.0)
        self.count_max_spin.setSpecialValueText("(none)")
        self.count_max_spin.setValue(float(cond.get("count_max") or 0.0))
        form.addRow("Count max (in zone)", self.count_max_spin)

        self.schedule_start_edit = QLineEdit(str((cond.get("time_window") or {}).get("start") or ""))
        self.schedule_start_edit.setPlaceholderText("HH:MM")
        form.addRow("Schedule start", self.schedule_start_edit)

        self.schedule_end_edit = QLineEdit(str((cond.get("time_window") or {}).get("end") or ""))
        self.schedule_end_edit.setPlaceholderText("HH:MM")
        form.addRow("Schedule end", self.schedule_end_edit)

        self.day_checks: List[QCheckBox] = []
        tw_days = (cond.get("time_window") or {}).get("days") if isinstance(cond.get("time_window"), dict) else []
        tw_days_set = {int(d) for d in (tw_days or []) if str(d).isdigit()}
        days_row = QHBoxLayout()
        for label, day_idx in DAY_LABELS:
            cb = QCheckBox(label)
            cb.setChecked(day_idx in tw_days_set if tw_days_set else True)
            self.day_checks.append(cb)
            days_row.addWidget(cb)
        days_wrap = QWidget()
        days_wrap.setLayout(days_row)
        form.addRow("Schedule days", days_wrap)

        self.cooldown_spin = QSpinBox()
        self.cooldown_spin.setRange(0, 600_000)
        self.cooldown_spin.setSingleStep(50)
        self.cooldown_spin.setSuffix(" ms")
        self.cooldown_spin.setValue(
            cooldown_ms_from_sec(
                float(cond.get("cooldown_sec", cond.get("cooldown", DEFAULT_RULE_COOLDOWN_SEC)) or DEFAULT_RULE_COOLDOWN_SEC)
            )
        )
        self.cooldown_spin.setToolTip(
            "Minimum time between rule firings (milliseconds). "
            "Stored as fractional seconds (e.g. 500 ms → 0.5 s)."
        )
        form.addRow("Cooldown", self.cooldown_spin)

        self.per_track_check = QCheckBox("Per-track cooldown")
        self.per_track_check.setChecked(cond.get("cooldown_per_track", True) is not False)
        form.addRow("", self.per_track_check)
        return w

    def _build_actions_tab(self, rule: Optional[Dict[str, Any]]) -> QWidget:
        w = QWidget()
        form = QFormLayout(w)
        actions = (rule or {}).get("actions") if isinstance((rule or {}).get("actions"), list) else []
        snap = next((a for a in actions if isinstance(a, dict) and str(a.get("type")) == "snapshot"), {})
        self.snapshot_check = QCheckBox("Take snapshot on trigger")
        self.snapshot_check.setChecked(bool(snap) or not actions)
        form.addRow("", self.snapshot_check)

        self.overlay_check = QCheckBox("Include overlays")
        self.overlay_check.setChecked(bool(snap.get("include_overlays", snap.get("overlay", True))))
        form.addRow("", self.overlay_check)

        self.save_dir_edit = QLineEdit(str(snap.get("save_dir") or "captures/motion_watch"))
        browse = QPushButton("Browse…")
        browse.clicked.connect(self._pick_save_dir)
        row = QHBoxLayout()
        row.addWidget(self.save_dir_edit)
        row.addWidget(browse)
        form.addRow("Save directory", row)
        return w

    def _pick_save_dir(self) -> None:
        target = QFileDialog.getExistingDirectory(self, "Select save directory", self.save_dir_edit.text() or ".")
        if target:
            self.save_dir_edit.setText(target)

    def _apply_rule(self, rule: Dict[str, Any]) -> None:
        self.name_edit.setText(str(rule.get("name") or ""))
        idx = self.trigger_combo.findData(str(rule.get("trigger") or "zone_enter"))
        if idx >= 0:
            self.trigger_combo.setCurrentIndex(idx)
        sidx = self.shape_combo.findData(str(rule.get("shape_id") or ""))
        if sidx >= 0:
            self.shape_combo.setCurrentIndex(sidx)
        cond = rule.get("conditions") if isinstance(rule.get("conditions"), dict) else {}
        classes = cond.get("classes") or cond.get("object_classes") or []
        if isinstance(classes, list):
            self.classes_edit.setText(", ".join(str(c) for c in classes))
        self.confidence_spin.setValue(float(cond.get("min_confidence", 0.5) or 0.5))
        didx = self.direction_combo.findData(str(cond.get("direction") or ""))
        if didx >= 0:
            self.direction_combo.setCurrentIndex(didx)
        self.dwell_min_spin.setValue(float(cond.get("dwell_min") or cond.get("dwell_min_sec") or 0.0))
        cidx = self.color_combo.findData(str(cond.get("color") or cond.get("dominant_color") or ""))
        if cidx >= 0:
            self.color_combo.setCurrentIndex(cidx)
        self.count_min_spin.setValue(float(cond.get("count_min") or 0.0))
        self.count_max_spin.setValue(float(cond.get("count_max") or 0.0))
        tw = cond.get("time_window") if isinstance(cond.get("time_window"), dict) else {}
        self.schedule_start_edit.setText(str(tw.get("start") or ""))
        self.schedule_end_edit.setText(str(tw.get("end") or ""))
        tw_days = tw.get("days") if isinstance(tw.get("days"), list) else []
        tw_days_set = {int(d) for d in tw_days if str(d).isdigit()}
        for i, cb in enumerate(self.day_checks):
            cb.setChecked(i in tw_days_set if tw_days_set else True)
        self.cooldown_spin.setValue(
            cooldown_ms_from_sec(float(cond.get("cooldown_sec", DEFAULT_RULE_COOLDOWN_SEC) or DEFAULT_RULE_COOLDOWN_SEC))
        )
        self.per_track_check.setChecked(cond.get("cooldown_per_track", True) is not False)
        actions = rule.get("actions") if isinstance(rule.get("actions"), list) else []
        snap = next((a for a in actions if isinstance(a, dict) and str(a.get("type")) == "snapshot"), {})
        self.snapshot_check.setChecked(bool(snap) or bool(actions))
        self.overlay_check.setChecked(bool(snap.get("include_overlays", True)))
        self.save_dir_edit.setText(str(snap.get("save_dir") or "captures/motion_watch"))

    def build_rule_payload(self) -> Dict[str, Any]:
        classes = [c.strip().lower() for c in self.classes_edit.text().split(",") if c.strip()]
        direction = str(self.direction_combo.currentData() or "").strip()
        shape_id = str(self.shape_combo.currentData() or "").strip() or None
        conditions: Dict[str, Any] = {
            "classes": classes,
            "min_confidence": float(self.confidence_spin.value()),
            "cooldown_sec": cooldown_sec_from_ms(int(self.cooldown_spin.value())),
            "cooldown_per_track": bool(self.per_track_check.isChecked()),
            "tracker_namespace": "backend_sort",
        }
        if direction:
            conditions["direction"] = direction
        dwell_min = float(self.dwell_min_spin.value())
        if dwell_min > 0:
            conditions["dwell_min"] = dwell_min

        color = str(self.color_combo.currentData() or "").strip()
        if color:
            conditions["color"] = color

        count_min = int(self.count_min_spin.value())
        count_max = int(self.count_max_spin.value())
        if count_min > 0:
            conditions["count_min"] = count_min
        if count_max > 0:
            conditions["count_max"] = count_max

        start = self.schedule_start_edit.text().strip()
        end = self.schedule_end_edit.text().strip()
        days = [i for i, cb in enumerate(self.day_checks) if cb.isChecked()]
        if start or end or days:
            time_window: Dict[str, Any] = {}
            if start:
                time_window["start"] = start
            if end:
                time_window["end"] = end
            if days:
                time_window["days"] = days
            conditions["time_window"] = time_window

        actions: List[Dict[str, Any]] = []
        if self.snapshot_check.isChecked():
            actions.append(
                {
                    "type": "snapshot",
                    "include_overlays": bool(self.overlay_check.isChecked()),
                    "save_dir": self.save_dir_edit.text().strip() or "captures/motion_watch",
                }
            )

        payload: Dict[str, Any] = {
            "name": self.name_edit.text().strip() or "Event Rule",
            "camera_id": self.camera_id,
            "trigger": str(self.trigger_combo.currentData() or "zone_enter"),
            "conditions": conditions,
            "actions": actions,
            "enabled": True,
        }
        if shape_id:
            payload["shape_id"] = shape_id
        if not self._rule_id:
            payload["id"] = f"rule_{uuid.uuid4().hex[:10]}"
        return payload

    def _save(self, *, arm: bool) -> None:
        body = self.build_rule_payload()
        saved = save_rule(self.api_base, body, rule_id=self._rule_id)
        if not saved:
            QMessageBox.warning(self, "Event Rules", "Failed to save rule. Is the API running?")
            return
        self.saved_rule = saved
        self._rule_id = str(saved.get("id") or self._rule_id)
        if arm:
            set_rules_enabled(self.api_base, self.camera_id, True)
        self.status_label.setText(f"Saved rule {self._rule_id}" + (" and armed." if arm else "."))
        self._arm_on_accept = arm
        self.accept()

    def should_arm(self) -> bool:
        return bool(self._arm_on_accept)

    def saved_rule_id(self) -> Optional[str]:
        return self._rule_id
