from __future__ import annotations

from dataclasses import replace
from typing import Dict, List, Optional, Tuple
import uuid
from datetime import datetime, timedelta

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QAbstractItemView,
    QComboBox,
    QCheckBox,
    QDialog,
    QDialogButtonBox,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QTimeEdit,
    QVBoxLayout,
    QInputDialog,
    QWidget,
    QFormLayout,
)


class LayoutManagerDialog(QDialog):
    """
    Layout Manager:
      - list layouts with running/paused status
      - load/run/pause/resume/stop
      - rename/duplicate/delete
      - per-layout and bulk switch policy overrides
      - set startup layout
      - save current layout / save-as
    """

    def __init__(self, parent=None, app=None):
        super().__init__(parent)
        self._app = app
        self.setWindowTitle("Layout Manager")
        self.setWindowFlags(Qt.WindowType.Tool | Qt.WindowType.WindowStaysOnTopHint)
        self.resize(980, 560)

        root = QVBoxLayout(self)

        # Header
        header = QHBoxLayout()
        self.current_lbl = QLabel("")
        self.current_lbl.setStyleSheet("color: #e5e7eb; font-weight: 700;")
        header.addWidget(self.current_lbl, stretch=1)

        self.refresh_btn = QPushButton("Refresh")
        self.refresh_btn.clicked.connect(self.refresh)
        header.addWidget(self.refresh_btn)
        root.addLayout(header)

        # Search + bulk policy
        controls = QGridLayout()
        root.addLayout(controls)

        controls.addWidget(QLabel("Search:"), 0, 0)
        self.search = QLineEdit()
        self.search.setPlaceholderText("type to filter layouts…")
        self.search.textChanged.connect(self.refresh)
        controls.addWidget(self.search, 0, 1, 1, 3)

        controls.addWidget(QLabel("Bulk switch policy:"), 1, 0)
        self.bulk_policy = QComboBox()
        self.bulk_policy.addItem("Ask (clear override)", "ask")
        self.bulk_policy.addItem("Stop previous", "stop")
        self.bulk_policy.addItem("Keep running", "keep")
        controls.addWidget(self.bulk_policy, 1, 1)

        self.apply_bulk_policy_btn = QPushButton("Apply to selected")
        self.apply_bulk_policy_btn.clicked.connect(self._apply_bulk_policy)
        controls.addWidget(self.apply_bulk_policy_btn, 1, 2)

        self.clear_bulk_policy_btn = QPushButton("Clear overrides (selected)")
        self.clear_bulk_policy_btn.clicked.connect(self._clear_overrides_selected)
        controls.addWidget(self.clear_bulk_policy_btn, 1, 3)

        # Table
        self.tbl = QTableWidget(0, 9)
        self.tbl.setHorizontalHeaderLabels(["Layout", "Widgets", "State", "Paused", "Hidden", "Startup", "Switch policy", "Auto visibility", "Id"])
        self.tbl.horizontalHeader().setStretchLastSection(True)
        self.tbl.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.tbl.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.tbl.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.tbl.setColumnHidden(8, True)  # internal id
        root.addWidget(self.tbl, stretch=1)

        # Actions row
        row = QHBoxLayout()
        root.addLayout(row)

        self.load_btn = QPushButton("Load")
        self.load_btn.clicked.connect(self._load_selected)
        row.addWidget(self.load_btn)

        self.run_bg_btn = QPushButton("Run in background")
        self.run_bg_btn.clicked.connect(self._run_selected_background)
        row.addWidget(self.run_bg_btn)

        self.pause_btn = QPushButton("Pause")
        self.pause_btn.clicked.connect(self._pause_selected)
        row.addWidget(self.pause_btn)

        self.resume_btn = QPushButton("Resume")
        self.resume_btn.clicked.connect(self._resume_selected)
        row.addWidget(self.resume_btn)

        self.stop_btn = QPushButton("Stop")
        self.stop_btn.clicked.connect(self._stop_selected)
        row.addWidget(self.stop_btn)

        row.addSpacing(8)
        self.hide_btn = QPushButton("Hide")
        self.hide_btn.clicked.connect(self._hide_selected)
        row.addWidget(self.hide_btn)
        self.show_btn = QPushButton("Show")
        self.show_btn.clicked.connect(self._show_selected)
        row.addWidget(self.show_btn)
        self.auto_hide_btn = QPushButton("Auto-hide…")
        self.auto_hide_btn.clicked.connect(self._edit_auto_hide_selected)
        row.addWidget(self.auto_hide_btn)

        self.widget_auto_btn = QPushButton("Widget auto show/hide…")
        self.widget_auto_btn.clicked.connect(self._edit_widget_auto_selected)
        row.addWidget(self.widget_auto_btn)

        self.schedules_btn = QPushButton("Schedules…")
        self.schedules_btn.clicked.connect(self._open_schedules)
        row.addWidget(self.schedules_btn)

        row.addSpacing(14)

        self.rename_btn = QPushButton("Rename…")
        self.rename_btn.clicked.connect(self._rename_selected)
        row.addWidget(self.rename_btn)

        self.dup_btn = QPushButton("Duplicate…")
        self.dup_btn.clicked.connect(self._duplicate_selected)
        row.addWidget(self.dup_btn)

        self.delete_btn = QPushButton("Delete…")
        self.delete_btn.clicked.connect(self._delete_selected)
        row.addWidget(self.delete_btn)

        row.addStretch()

        self.startup_btn = QPushButton("Set as startup")
        self.startup_btn.clicked.connect(self._set_startup_selected)
        row.addWidget(self.startup_btn)

        self.clear_startup_btn = QPushButton("Clear startup")
        self.clear_startup_btn.clicked.connect(self._clear_startup)
        row.addWidget(self.clear_startup_btn)

        # Save row
        save_row = QHBoxLayout()
        root.addLayout(save_row)
        self.save_current_btn = QPushButton("Save current layout")
        self.save_current_btn.clicked.connect(self._save_current)
        save_row.addWidget(self.save_current_btn)
        self.save_as_btn = QPushButton("Save current as…")
        self.save_as_btn.clicked.connect(self._save_current_as)
        save_row.addWidget(self.save_as_btn)
        save_row.addStretch()

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        buttons.rejected.connect(self.reject)
        buttons.accepted.connect(self.accept)
        root.addWidget(buttons)

        self.refresh()

    def _open_schedules(self):
        try:
            dlg = LayoutSchedulesDialog(self, app=self._app)
            dlg.exec()
        except Exception:
            return
        self.refresh()

    # ---- helpers ----
    def _selected_layout_ids(self) -> List[str]:
        ids = []
        for idx in self.tbl.selectionModel().selectedRows():
            try:
                ridx = idx.row()
                lid = str(self.tbl.item(ridx, 8).text())
                if lid and lid not in ids:
                    ids.append(lid)
            except Exception:
                continue
        return ids

    def _prefs(self) -> Dict:
        try:
            return self._app._load_prefs() if self._app else {}
        except Exception:
            return {}

    def _save_prefs(self, prefs: Dict) -> None:
        try:
            if self._app:
                self._app._save_prefs(prefs)
        except Exception:
            pass

    def _layout_policy_label(self, prefs: Dict, layout_id: str) -> str:
        try:
            ov = prefs.get("layout_switch_policy_overrides") if isinstance(prefs, dict) else {}
            if isinstance(ov, dict):
                v = ov.get(layout_id)
                if v == "stop":
                    return "Stop"
                if v == "keep":
                    return "Keep"
            # No override: show default
            d = prefs.get("layout_switch_policy_default") if isinstance(prefs, dict) else None
            if d == "stop":
                return "Stop (default)"
            if d == "keep":
                return "Keep (default)"
            return "Ask (default)"
        except Exception:
            return ""

    def refresh(self):
        layouts = []
        try:
            layouts = self._app._list_layouts_v2() if self._app else []
        except Exception:
            layouts = []

        prefs = self._prefs()
        startup = prefs.get("startup_layout") if isinstance(prefs, dict) else None
        query = (self.search.text() or "").strip().lower()

        # Header current layout
        cur = getattr(self._app, "current_layout_name", None) if self._app else None
        self.current_lbl.setText(f"Current layout: {cur}" if cur else "Current layout: (none)")

        # Filter
        if query:
            layouts = [l for l in layouts if query in (l.name or l.id).lower() or query in (l.id or "").lower()]

        self.tbl.setRowCount(0)
        for l in layouts:
            lid = str(l.id)
            name = str(l.name or l.id)
            wcount = str(len(l.widgets or []))

            # state from app quick control helpers (if available)
            state = {}
            try:
                state = self._app._layout_state(lid) if self._app else {}
            except Exception:
                state = {}
            running = bool(state.get("running"))
            paused = bool(state.get("paused"))
            current = bool(state.get("current"))
            hidden = bool(state.get("hidden"))

            state_txt = "current" if current else ("running" if running else "stopped")
            paused_txt = "yes" if paused else ""
            hidden_txt = "yes" if hidden else ""
            startup_txt = "yes" if (startup and str(startup) == lid) else ""
            policy_txt = self._layout_policy_label(prefs, lid)
            auto_txt = ""
            try:
                cfg = self._app._get_layout_auto_hide(lid) if self._app else {}
                parts = []
                if bool(cfg.get("on_layout_switch")):
                    parts.append("switch")
                if bool(cfg.get("on_motion")):
                    parts.append("motion")
                if bool(cfg.get("on_detections")):
                    parts.append("detections")
                auto_txt = ", ".join(parts)
            except Exception:
                auto_txt = ""

            r = self.tbl.rowCount()
            self.tbl.insertRow(r)
            self.tbl.setItem(r, 0, QTableWidgetItem(name))
            self.tbl.setItem(r, 1, QTableWidgetItem(wcount))
            self.tbl.setItem(r, 2, QTableWidgetItem(state_txt))
            self.tbl.setItem(r, 3, QTableWidgetItem(paused_txt))
            self.tbl.setItem(r, 4, QTableWidgetItem(hidden_txt))
            self.tbl.setItem(r, 5, QTableWidgetItem(startup_txt))
            self.tbl.setItem(r, 6, QTableWidgetItem(policy_txt))
            self.tbl.setItem(r, 7, QTableWidgetItem(auto_txt))
            self.tbl.setItem(r, 8, QTableWidgetItem(lid))

    # ---- actions ----
    def _load_selected(self):
        ids = self._selected_layout_ids()
        if not ids:
            return
        self._app.load_layout(ids[0])
        self.refresh()

    def _run_selected_background(self):
        ids = self._selected_layout_ids()
        for lid in ids:
            try:
                self._app._layout_start_in_background(lid)
            except Exception:
                continue
        self.refresh()

    def _pause_selected(self):
        ids = self._selected_layout_ids()
        for lid in ids:
            try:
                self._app._layout_pause(lid)
            except Exception:
                continue
        self.refresh()

    def _resume_selected(self):
        ids = self._selected_layout_ids()
        for lid in ids:
            try:
                self._app._layout_resume(lid)
            except Exception:
                continue
        self.refresh()

    def _stop_selected(self):
        ids = self._selected_layout_ids()
        for lid in ids:
            try:
                self._app._layout_stop(lid)
            except Exception:
                continue
        self.refresh()

    def _hide_selected(self):
        ids = self._selected_layout_ids()
        for lid in ids:
            try:
                self._app._layout_hide(lid, persist=True)
            except Exception:
                continue
        self.refresh()

    def _show_selected(self):
        ids = self._selected_layout_ids()
        for lid in ids:
            try:
                self._app._layout_show(lid, persist=True)
            except Exception:
                continue
        self.refresh()

    def _edit_auto_hide_selected(self):
        ids = self._selected_layout_ids()
        if len(ids) != 1:
            return
        lid = ids[0]
        try:
            cfg = self._app._get_layout_auto_hide(lid)
        except Exception:
            cfg = {"on_layout_switch": False, "on_motion": False, "on_detections": False}

        try:
            show_cfg = self._app._get_layout_auto_show(lid)
        except Exception:
            show_cfg = {"on_motion": False, "on_detections": False}

        try:
            from PySide6.QtWidgets import QDialog, QVBoxLayout, QCheckBox
        except Exception:
            return

        dlg = QDialog(self)
        dlg.setWindowTitle("Auto visibility settings")
        v = QVBoxLayout(dlg)
        v.addWidget(QLabel(f"Layout: {lid}"))
        cb_switch = QCheckBox("Auto-hide when switching to a new layout (Keep previous)")
        cb_switch.setChecked(bool(cfg.get("on_layout_switch")))
        v.addWidget(cb_switch)

        v.addWidget(QLabel("Auto-hide:"))
        cb_motion_hide = QCheckBox("Hide on motion/shape triggers (zones/lines/tags)")
        cb_motion_hide.setChecked(bool(cfg.get("on_motion")))
        v.addWidget(cb_motion_hide)
        cb_det_hide = QCheckBox("Hide on detections (backend/desktop)")
        cb_det_hide.setChecked(bool(cfg.get("on_detections")))
        v.addWidget(cb_det_hide)

        v.addWidget(QLabel("Auto-show:"))
        cb_motion_show = QCheckBox("Show on motion/shape triggers")
        cb_motion_show.setChecked(bool(show_cfg.get("on_motion")))
        v.addWidget(cb_motion_show)
        cb_det_show = QCheckBox("Show on detections")
        cb_det_show.setChecked(bool(show_cfg.get("on_detections")))
        v.addWidget(cb_det_show)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        v.addWidget(buttons)
        buttons.accepted.connect(dlg.accept)
        buttons.rejected.connect(dlg.reject)
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return

        prefs = self._prefs()
        if not isinstance(prefs, dict):
            prefs = {}
        auto = prefs.get("layout_auto_hide")
        if not isinstance(auto, dict):
            auto = {}
        auto[lid] = {
            "on_layout_switch": bool(cb_switch.isChecked()),
            "on_motion": bool(cb_motion_hide.isChecked()),
            "on_detections": bool(cb_det_hide.isChecked()),
        }
        prefs["layout_auto_hide"] = auto

        show_map = prefs.get("layout_auto_show")
        if not isinstance(show_map, dict):
            show_map = {}
        show_map[lid] = {
            "on_motion": bool(cb_motion_show.isChecked()),
            "on_detections": bool(cb_det_show.isChecked()),
        }
        prefs["layout_auto_show"] = show_map

        self._save_prefs(prefs)
        self.refresh()

    def _edit_widget_auto_selected(self):
        """
        Minimal per-camera-widget auto show/hide editor (for camera widgets only).
        Applies rules keyed by "camera:<camera_id>".
        """
        ids = self._selected_layout_ids()
        if len(ids) != 1:
            return
        layout_id = ids[0]
        layout = self._app.layouts_store.get_layout(layout_id) if self._app else None
        if not layout:
            return

        # Collect camera ids in this layout
        cam_ids = []
        for w in layout.widgets or []:
            try:
                if getattr(w, "type", None) == "camera" and getattr(w, "camera_id", None):
                    cam_ids.append(str(w.camera_id))
            except Exception:
                continue
        cam_ids = [c for c in cam_ids if c]
        cam_ids = list(dict.fromkeys(cam_ids))  # stable de-dupe
        if not cam_ids:
            QMessageBox.information(self, "Widget auto show/hide", "This layout has no camera widgets.")
            return

        try:
            from PySide6.QtWidgets import (
                QDialog,
                QVBoxLayout,
                QHBoxLayout,
                QCheckBox,
                QScrollArea,
                QWidget,
                QGroupBox,
                QSpinBox,
                QFormLayout,
                QComboBox,
            )
        except Exception:
            return

        prefs = self._prefs()
        if not isinstance(prefs, dict):
            prefs = {}
        w_hide = prefs.get("widget_auto_hide")
        if not isinstance(w_hide, dict):
            w_hide = {}
        w_show = prefs.get("widget_auto_show")
        if not isinstance(w_show, dict):
            w_show = {}
        rules = prefs.get("visibility_rules")
        if not isinstance(rules, list):
            rules = []
        focus_steal_enabled = True if "focus_steal_enabled" not in prefs else bool(prefs.get("focus_steal_enabled"))

        # Pre-index existing timed rules by camera
        cam_rule_index: Dict[str, Dict] = {}
        for r in list(rules or []):
            try:
                if not isinstance(r, dict):
                    continue
                trig = r.get("trigger") if isinstance(r.get("trigger"), dict) else {}
                tgt = r.get("target") if isinstance(r.get("target"), dict) else {}
                act = r.get("action") if isinstance(r.get("action"), dict) else {}
                if str(trig.get("type") or "") != "shape_trigger":
                    continue
                if str(tgt.get("type") or "") != "widget":
                    continue
                act_type = str(act.get("type") or "")
                if act_type not in {"show_and_activate", "bring_to_front"}:
                    continue
                cam_id = str(trig.get("camera_id") or "").strip()
                if not cam_id:
                    continue
                shape_id = str(trig.get("shape_id") or "").strip()
                kind = str(trig.get("kind") or "").strip().lower()
                src = str(trig.get("source") or "").strip().lower()
                if not shape_id or kind not in {"zone", "line", "tag"} or src not in {"motion", "detection"}:
                    continue
                try:
                    duration = float(act.get("duration_sec", 0) or 0)
                except Exception:
                    duration = 0.0
                try:
                    cooldown = float(act.get("cooldown_sec", 0) or 0)
                except Exception:
                    cooldown = 0.0
                ent = cam_rule_index.get(cam_id) or {"motion": set(), "detection": set(), "duration": 0.0, "cooldown": 0.0, "action_type": "show_and_activate"}
                ent[src].add((shape_id, kind))
                ent["duration"] = max(float(ent.get("duration") or 0.0), float(duration or 0.0))
                ent["cooldown"] = max(float(ent.get("cooldown") or 0.0), float(cooldown or 0.0))
                if act_type == "bring_to_front":
                    ent["action_type"] = "bring_to_front"
                cam_rule_index[cam_id] = ent
            except Exception:
                continue

        dlg = QDialog(self)
        dlg.setWindowTitle("Widget auto show/hide (Camera widgets)")
        dlg.resize(640, 520)
        v = QVBoxLayout(dlg)
        v.addWidget(QLabel(f"Layout: {layout_id}"))
        v.addWidget(QLabel("Rules apply to camera widgets by camera id. Auto-show takes priority over auto-hide."))
        cb_focus = QCheckBox("Bring to front / activate window when timed auto-show triggers")
        cb_focus.setChecked(bool(focus_steal_enabled))
        v.addWidget(cb_focus)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        inner = QWidget()
        inner_v = QVBoxLayout(inner)
        scroll.setWidget(inner)
        v.addWidget(scroll, stretch=1)

        # Build per-camera rows
        rows: List[Dict] = []
        cams_map = getattr(getattr(self._app, "camera_manager", None), "cameras", {}) if self._app else {}

        for cam_id in cam_ids:
            key = f"camera:{cam_id}"
            name = cam_id
            try:
                cfg = (cams_map or {}).get(cam_id)
                if cfg and getattr(cfg, "name", None):
                    name = str(cfg.name)
            except Exception:
                pass

            box = QGroupBox(f"{name}  ({cam_id})")
            box_v = QVBoxLayout(box)

            cb_show_motion = QCheckBox("Auto-show on motion")
            cb_show_det = QCheckBox("Auto-show on detections")
            cb_hide_motion = QCheckBox("Auto-hide on motion")
            cb_hide_det = QCheckBox("Auto-hide on detections")

            cb_show_motion.setChecked(bool((w_show.get(key) or {}).get("on_motion")) if isinstance(w_show.get(key), dict) else False)
            cb_show_det.setChecked(bool((w_show.get(key) or {}).get("on_detections")) if isinstance(w_show.get(key), dict) else False)
            cb_hide_motion.setChecked(bool((w_hide.get(key) or {}).get("on_motion")) if isinstance(w_hide.get(key), dict) else False)
            cb_hide_det.setChecked(bool((w_hide.get(key) or {}).get("on_detections")) if isinstance(w_hide.get(key), dict) else False)

            box_v.addWidget(cb_show_motion)
            box_v.addWidget(cb_show_det)
            box_v.addWidget(cb_hide_motion)
            box_v.addWidget(cb_hide_det)

            # Timed auto-show section (shape-triggered)
            timed = QGroupBox("Auto-show (timed) on selected shapes")
            timed_v = QVBoxLayout(timed)
            timed_v.addWidget(QLabel("Select shapes (zones/lines/tags) and a duration. The widget will re-hide if it was hidden."))

            ent = cam_rule_index.get(cam_id) or {"motion": set(), "detection": set(), "duration": 0.0, "cooldown": 0.0, "action_type": "show_and_activate"}
            cb_timed_motion = QCheckBox("Trigger on motion")
            cb_timed_det = QCheckBox("Trigger on detections")
            cb_timed_motion.setChecked(bool(ent.get("motion")))
            cb_timed_det.setChecked(bool(ent.get("detection")))
            timed_v.addWidget(cb_timed_motion)
            timed_v.addWidget(cb_timed_det)

            form = QFormLayout()
            combo_action = QComboBox()
            combo_action.addItem("Show and Activate (Timed)", "show_and_activate")
            combo_action.addItem("Bring to Front (Permanent)", "bring_to_front")
            cur_action = str(ent.get("action_type") or "show_and_activate")
            idx = combo_action.findData(cur_action)
            if idx >= 0:
                combo_action.setCurrentIndex(idx)
            form.addRow("Action:", combo_action)

            sp_duration = QSpinBox()
            sp_duration.setRange(1, 3600)
            sp_duration.setSuffix(" s")
            sp_duration.setValue(int(max(1.0, float(ent.get("duration") or 10.0))))
            sp_cooldown = QSpinBox()
            sp_cooldown.setRange(0, 3600)
            sp_cooldown.setSuffix(" s")
            sp_cooldown.setValue(int(max(0.0, float(ent.get("cooldown") or 10.0))))
            form.addRow("Duration:", sp_duration)
            form.addRow("Cooldown:", sp_cooldown)
            timed_v.addLayout(form)

            def _on_action_changed(index, d=sp_duration, c=sp_cooldown, cb=combo_action):
                is_timed = cb.currentData() == "show_and_activate"
                d.setEnabled(is_timed)
                c.setEnabled(is_timed)
            combo_action.currentIndexChanged.connect(_on_action_changed)
            _on_action_changed(combo_action.currentIndex())

            shape_cbs: List[Tuple[str, str, QCheckBox]] = []
            shapes: list = []
            try:
                shapes = self._app._list_assigned_shapes_for_camera(str(cam_id)) if self._app else []
            except Exception:
                shapes = []

            if shapes:
                timed_v.addWidget(QLabel("Shapes:"))
                for sh in list(shapes or []):
                    try:
                        if not isinstance(sh, dict):
                            continue
                        sid = str(sh.get("id") or "").strip()
                        kind = str(sh.get("kind") or "").strip().lower()
                        if not sid or kind not in {"zone", "line", "tag"}:
                            continue
                        label = str(sh.get("label") or sid)
                        cb = QCheckBox(f"{kind}: {label}  ({sid})")
                        if (sid, kind) in set(ent.get("motion") or set()) or (sid, kind) in set(ent.get("detection") or set()):
                            cb.setChecked(True)
                        timed_v.addWidget(cb)
                        shape_cbs.append((sid, kind, cb))
                    except Exception:
                        continue
            else:
                timed_v.addWidget(QLabel("No assigned shapes found for this camera (assign a profile with overlays to enable shape selection)."))

            box_v.addWidget(timed)
            inner_v.addWidget(box)

            rows.append(
                {
                    "camera_id": str(cam_id),
                    "key": key,
                    "cb_show_motion": cb_show_motion,
                    "cb_show_det": cb_show_det,
                    "cb_hide_motion": cb_hide_motion,
                    "cb_hide_det": cb_hide_det,
                    "cb_timed_motion": cb_timed_motion,
                    "cb_timed_det": cb_timed_det,
                    "combo_action": combo_action,
                    "sp_duration": sp_duration,
                    "sp_cooldown": sp_cooldown,
                    "shape_cbs": shape_cbs,
                }
            )

            inner_v.addWidget(QLabel(""))

        inner_v.addStretch()

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        v.addWidget(buttons)
        buttons.accepted.connect(dlg.accept)
        buttons.rejected.connect(dlg.reject)

        if dlg.exec() != QDialog.DialogCode.Accepted:
            return

        # Persist standard show/hide + global focus steal flag
        prefs["focus_steal_enabled"] = bool(cb_focus.isChecked())

        cam_ids_set = set(str(c) for c in cam_ids if c)
        for row in rows:
            key = str(row.get("key") or "")
            w_show[key] = {
                "on_motion": bool(row["cb_show_motion"].isChecked()),
                "on_detections": bool(row["cb_show_det"].isChecked()),
            }
            w_hide[key] = {
                "on_motion": bool(row["cb_hide_motion"].isChecked()),
                "on_detections": bool(row["cb_hide_det"].isChecked()),
            }

        # Rebuild timed auto-show rules for cameras in this layout, keep others.
        def _is_timed_rule_for_cam(rule: dict) -> bool:
            try:
                if not isinstance(rule, dict):
                    return False
                trig = rule.get("trigger") if isinstance(rule.get("trigger"), dict) else {}
                tgt = rule.get("target") if isinstance(rule.get("target"), dict) else {}
                act = rule.get("action") if isinstance(rule.get("action"), dict) else {}
                if str(trig.get("type") or "") != "shape_trigger":
                    return False
                if str(tgt.get("type") or "") != "widget":
                    return False
                if str(act.get("type") or "") not in {"show_and_activate", "bring_to_front"}:
                    return False
                return str(trig.get("camera_id") or "") in cam_ids_set
            except Exception:
                return False

        kept_rules = []
        for r in list(rules or []):
            if not _is_timed_rule_for_cam(r):
                kept_rules.append(r)

        new_rules = list(kept_rules)
        for row in rows:
            cam_id = str(row.get("camera_id") or "").strip()
            if not cam_id:
                continue
            key = str(row.get("key") or "").strip()
            if not key:
                continue
            timed_sources = []
            if bool(row["cb_timed_motion"].isChecked()):
                timed_sources.append("motion")
            if bool(row["cb_timed_det"].isChecked()):
                timed_sources.append("detection")
            if not timed_sources:
                continue
            try:
                duration = int(row["sp_duration"].value())
            except Exception:
                duration = 10
            try:
                cooldown = int(row["sp_cooldown"].value())
            except Exception:
                cooldown = 10
            duration = max(1, min(3600, int(duration)))
            cooldown = max(0, min(3600, int(cooldown)))

            selected_shapes: List[Tuple[str, str]] = []
            for sid, kind, cb in list(row.get("shape_cbs") or []):
                try:
                    if cb.isChecked():
                        selected_shapes.append((str(sid), str(kind)))
                except Exception:
                    continue
            if not selected_shapes:
                continue

            rule_action_type = str(row.get("combo_action").currentData() or "show_and_activate")

            for sid, kind in selected_shapes:
                for src in timed_sources:
                    action_payload: dict = {"type": rule_action_type}
                    if rule_action_type == "show_and_activate":
                        action_payload["duration_sec"] = duration
                        action_payload["cooldown_sec"] = cooldown
                    new_rules.append(
                        {
                            "trigger": {
                                "type": "shape_trigger",
                                "camera_id": cam_id,
                                "shape_id": sid,
                                "kind": kind,
                                "source": src,
                            },
                            "target": {"type": "widget", "widget_key": key},
                            "action": action_payload,
                        }
                    )

        prefs["widget_auto_show"] = w_show
        prefs["widget_auto_hide"] = w_hide
        prefs["visibility_rules"] = new_rules
        self._save_prefs(prefs)
        self.refresh()

    def _set_startup_selected(self):
        ids = self._selected_layout_ids()
        if not ids:
            return
        try:
            self._app._set_startup_layout(ids[0])
        except Exception:
            pass
        self.refresh()

    def _clear_startup(self):
        try:
            self._app._set_startup_layout(None)
        except Exception:
            pass
        self.refresh()

    def _apply_bulk_policy(self):
        ids = self._selected_layout_ids()
        if not ids:
            return
        policy = str(self.bulk_policy.currentData() or "ask")
        prefs = self._prefs()
        if not isinstance(prefs, dict):
            prefs = {}
        ov = prefs.get("layout_switch_policy_overrides")
        if not isinstance(ov, dict):
            ov = {}
        for lid in ids:
            if policy == "ask":
                ov.pop(lid, None)
            else:
                ov[lid] = policy
        prefs["layout_switch_policy_overrides"] = ov
        self._save_prefs(prefs)
        self.refresh()

    def _clear_overrides_selected(self):
        ids = self._selected_layout_ids()
        if not ids:
            return
        prefs = self._prefs()
        if not isinstance(prefs, dict):
            prefs = {}
        ov = prefs.get("layout_switch_policy_overrides")
        if not isinstance(ov, dict):
            ov = {}
        for lid in ids:
            ov.pop(lid, None)
        prefs["layout_switch_policy_overrides"] = ov
        self._save_prefs(prefs)
        self.refresh()

    def _rename_selected(self):
        ids = self._selected_layout_ids()
        if len(ids) != 1:
            return
        old_id = ids[0]
        layout = self._app.layouts_store.get_layout(old_id) if self._app else None
        if not layout:
            return
        new_id, ok = QInputDialog.getText(self, "Rename layout", "New layout name:", text=str(layout.name or layout.id))
        if not (ok and new_id.strip()):
            return
        new_id = new_id.strip()
        if new_id == old_id:
            return

        # Stop any running instance of this layout to avoid dangling references.
        try:
            self._app._layout_stop(old_id)
        except Exception:
            pass

        # Update prefs references
        prefs = self._prefs()
        if not isinstance(prefs, dict):
            prefs = {}
        if prefs.get("startup_layout") == old_id:
            prefs["startup_layout"] = new_id
        ov = prefs.get("layout_switch_policy_overrides")
        if isinstance(ov, dict) and old_id in ov:
            ov[new_id] = ov.pop(old_id)
        prefs["layout_switch_policy_overrides"] = ov if isinstance(ov, dict) else {}
        self._save_prefs(prefs)

        # Rename in store by delete+upsert
        try:
            self._app.layouts_store.delete_layout(old_id)
        except Exception:
            pass
        try:
            layout.id = new_id
            layout.name = new_id
            self._app.layouts_store.upsert_layout(layout)
        except Exception as e:
            QMessageBox.warning(self, "Rename layout", f"Failed to rename: {e}")
            return

        # Update app runtime pointers if needed
        try:
            if getattr(self._app, "current_layout_name", None) == old_id:
                self._app.current_layout_name = new_id
        except Exception:
            pass
        try:
            sid = self._app._layout_sessions.pop(old_id, None)
            if sid:
                self._app._layout_sessions[new_id] = sid
        except Exception:
            pass
        try:
            if old_id in self._app._layout_paused:
                self._app._layout_paused.discard(old_id)
                self._app._layout_paused.add(new_id)
        except Exception:
            pass

        self.refresh()

    def _duplicate_selected(self):
        ids = self._selected_layout_ids()
        if len(ids) != 1:
            return
        src_id = ids[0]
        layout = self._app.layouts_store.get_layout(src_id) if self._app else None
        if not layout:
            return
        new_id, ok = QInputDialog.getText(self, "Duplicate layout", "New layout name:", text=f"{layout.name}-copy")
        if not (ok and new_id.strip()):
            return
        new_id = new_id.strip()
        if not new_id:
            return
        if self._app.layouts_store.get_layout(new_id):
            QMessageBox.warning(self, "Duplicate layout", "That layout name already exists.")
            return
        try:
            dup = replace(layout)
            dup.id = new_id
            dup.name = new_id
            self._app.layouts_store.upsert_layout(dup)
        except Exception as e:
            QMessageBox.warning(self, "Duplicate layout", f"Failed to duplicate: {e}")
            return
        self.refresh()

    def _delete_selected(self):
        ids = self._selected_layout_ids()
        if not ids:
            return
        try:
            box = QMessageBox(self)
            box.setWindowTitle("Delete layouts")
            box.setText(f"Delete {len(ids)} layout(s)?")
            box.setInformativeText("This cannot be undone.")
            box.setIcon(QMessageBox.Icon.Warning)

            delete_btn = box.addButton("Delete", QMessageBox.ButtonRole.DestructiveRole)
            box.addButton(QMessageBox.StandardButton.Cancel)
            box.setDefaultButton(QMessageBox.StandardButton.Cancel)

            box.exec()
            if box.clickedButton() != delete_btn:
                return
        except Exception:
            resp = QMessageBox.question(
                self,
                "Delete layouts",
                f"Delete {len(ids)} layout(s)? This cannot be undone.",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel,
                QMessageBox.StandardButton.Cancel,
            )
            if resp != QMessageBox.StandardButton.Yes:
                return

        prefs = self._prefs()
        if not isinstance(prefs, dict):
            prefs = {}
        ov = prefs.get("layout_switch_policy_overrides")
        if not isinstance(ov, dict):
            ov = {}
        for lid in ids:
            try:
                self._app._layout_stop(lid)
            except Exception:
                pass
            try:
                self._app.layouts_store.delete_layout(lid)
            except Exception:
                pass
            ov.pop(lid, None)
            if prefs.get("startup_layout") == lid:
                prefs.pop("startup_layout", None)

        prefs["layout_switch_policy_overrides"] = ov
        self._save_prefs(prefs)
        self.refresh()

    def _save_current(self):
        try:
            self._app.prompt_save_layout()
        except Exception:
            pass
        self.refresh()

    def _save_current_as(self):
        try:
            prev = getattr(self._app, "current_layout_name", None)
            self._app.current_layout_name = None
            self._app.prompt_save_layout()
            self._app.current_layout_name = prev
        except Exception:
            try:
                self._app.current_layout_name = prev
            except Exception:
                pass
        self.refresh()


class LayoutSchedulesDialog(QDialog):
    """
    Manage layout schedules (time-of-day + uptime/runtime).
    Persisted in desktop prefs under:
      - scheduler_enabled: bool
      - scheduler_snooze_until: ISO string | None
      - scheduler_manual_override_minutes: int
      - layout_schedules: list[dict]
    """

    def __init__(self, parent=None, app=None):
        super().__init__(parent)
        self._app = app
        self.setWindowTitle("Layout Schedules")
        self.setWindowFlags(Qt.WindowType.Tool | Qt.WindowType.WindowStaysOnTopHint)
        self.resize(980, 520)

        root = QVBoxLayout(self)

        # Global controls
        top = QHBoxLayout()
        self.enabled_cb = QCheckBox("Enable scheduler")
        top.addWidget(self.enabled_cb)
        top.addWidget(QLabel("Manual override (minutes):"))
        self.override_minutes = QSpinBox()
        self.override_minutes.setRange(0, 24 * 60)
        self.override_minutes.setValue(30)
        top.addWidget(self.override_minutes)
        top.addStretch()
        self.snooze_1h_btn = QPushButton("Snooze 1h")
        self.snooze_1h_btn.clicked.connect(lambda: self._snooze_minutes(60))
        top.addWidget(self.snooze_1h_btn)
        self.clear_snooze_btn = QPushButton("Clear snooze")
        self.clear_snooze_btn.clicked.connect(self._clear_snooze)
        top.addWidget(self.clear_snooze_btn)
        root.addLayout(top)

        self.snooze_lbl = QLabel("")
        self.snooze_lbl.setStyleSheet("color: #e5e7eb;")
        root.addWidget(self.snooze_lbl)

        # Table
        self.tbl = QTableWidget(0, 11)
        self.tbl.setHorizontalHeaderLabels(
            ["Enabled", "Name", "Layout", "Type", "When", "Action", "Policy", "Priority", "Cooldown", "Last run", "Id"]
        )
        self.tbl.horizontalHeader().setStretchLastSection(True)
        self.tbl.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.tbl.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.tbl.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.tbl.setColumnHidden(10, True)
        root.addWidget(self.tbl, stretch=1)

        # Actions
        row = QHBoxLayout()
        self.add_btn = QPushButton("Add…")
        self.add_btn.clicked.connect(self._add)
        row.addWidget(self.add_btn)
        self.edit_btn = QPushButton("Edit…")
        self.edit_btn.clicked.connect(self._edit)
        row.addWidget(self.edit_btn)
        self.del_btn = QPushButton("Delete…")
        self.del_btn.clicked.connect(self._delete)
        row.addWidget(self.del_btn)
        row.addStretch()
        self.run_now_btn = QPushButton("Run now")
        self.run_now_btn.clicked.connect(self._run_now)
        row.addWidget(self.run_now_btn)
        root.addLayout(row)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close | QDialogButtonBox.StandardButton.Save)
        buttons.accepted.connect(self._save_all)
        buttons.rejected.connect(self.reject)
        root.addWidget(buttons)

        self._load()

    def _prefs(self) -> Dict:
        try:
            return self._app._load_prefs() if self._app else {}
        except Exception:
            return {}

    def _save_prefs(self, prefs: Dict) -> None:
        try:
            if self._app:
                self._app._save_prefs(prefs)
        except Exception:
            pass

    def _layouts(self):
        try:
            return self._app._list_layouts_v2() if self._app else []
        except Exception:
            return []

    def _schedule_list(self) -> List[dict]:
        prefs = self._prefs()
        v = prefs.get("layout_schedules") if isinstance(prefs, dict) else None
        return list(v) if isinstance(v, list) else []

    def _set_schedule_list(self, schedules: List[dict]) -> None:
        prefs = self._prefs()
        if not isinstance(prefs, dict):
            prefs = {}
        prefs["layout_schedules"] = list(schedules or [])
        self._save_prefs(prefs)

    def _selected_schedule_id(self) -> Optional[str]:
        for idx in self.tbl.selectionModel().selectedRows():
            try:
                return str(self.tbl.item(idx.row(), 10).text())
            except Exception:
                return None
        return None

    def _load(self):
        prefs = self._prefs()
        if not isinstance(prefs, dict):
            prefs = {}
        self.enabled_cb.setChecked(True if ("scheduler_enabled" not in prefs) else bool(prefs.get("scheduler_enabled")))
        try:
            self.override_minutes.setValue(int(prefs.get("scheduler_manual_override_minutes", 30) or 0))
        except Exception:
            self.override_minutes.setValue(30)

        until = prefs.get("scheduler_snooze_until")
        if until:
            self.snooze_lbl.setText(f"Snoozed until: {until}")
        else:
            self.snooze_lbl.setText("Snooze: (none)")

        schedules = self._schedule_list()
        self.tbl.setRowCount(0)
        layouts = {str(l.id): str(l.name or l.id) for l in self._layouts()}
        for s in schedules:
            if not isinstance(s, dict):
                continue
            sid = str(s.get("id") or "")
            lid = str(s.get("layout_id") or "")
            stype = str(s.get("type") or "time_of_day")
            enabled = bool(s.get("enabled", True))
            name = str(s.get("name") or sid or "(schedule)")
            action = str(s.get("action") or "load")
            policy = str(s.get("switch_policy") or "stop")
            pr = str(s.get("priority", 100))
            cd = str(s.get("cooldown_sec", 300))
            last = str(s.get("last_run_at") or "")

            when = ""
            if stype in {"time", "time_of_day", "timeofday"}:
                days = s.get("days")
                if isinstance(days, list) and days:
                    when += "days=" + ",".join(str(int(x)) for x in days if str(x).strip())
                st = str(s.get("start_time") or "00:00")
                en = str(s.get("end_time") or "00:00")
                when = (when + " " if when else "") + f"{st}-{en}"
            elif stype in {"uptime", "runtime"}:
                when = f"after={int(float(s.get('after_sec') or 0))}s"

            r = self.tbl.rowCount()
            self.tbl.insertRow(r)
            self.tbl.setItem(r, 0, QTableWidgetItem("yes" if enabled else ""))
            self.tbl.setItem(r, 1, QTableWidgetItem(name))
            self.tbl.setItem(r, 2, QTableWidgetItem(layouts.get(lid, lid)))
            self.tbl.setItem(r, 3, QTableWidgetItem(stype))
            self.tbl.setItem(r, 4, QTableWidgetItem(when))
            self.tbl.setItem(r, 5, QTableWidgetItem(action))
            self.tbl.setItem(r, 6, QTableWidgetItem(policy))
            self.tbl.setItem(r, 7, QTableWidgetItem(pr))
            self.tbl.setItem(r, 8, QTableWidgetItem(cd))
            self.tbl.setItem(r, 9, QTableWidgetItem(last))
            self.tbl.setItem(r, 10, QTableWidgetItem(sid))

    def _save_all(self):
        prefs = self._prefs()
        if not isinstance(prefs, dict):
            prefs = {}
        prefs["scheduler_enabled"] = bool(self.enabled_cb.isChecked())
        prefs["scheduler_manual_override_minutes"] = int(self.override_minutes.value())
        self._save_prefs(prefs)
        self.accept()

    def _snooze_minutes(self, minutes: int):
        prefs = self._prefs()
        if not isinstance(prefs, dict):
            prefs = {}
        try:
            until = datetime.now() + timedelta(minutes=int(minutes))
        except Exception:
            until = datetime.now()
        prefs["scheduler_snooze_until"] = until.isoformat(timespec="seconds")
        self._save_prefs(prefs)
        self._load()

    def _clear_snooze(self):
        prefs = self._prefs()
        if not isinstance(prefs, dict):
            prefs = {}
        prefs["scheduler_snooze_until"] = None
        self._save_prefs(prefs)
        self._load()

    def _run_now(self):
        sid = self._selected_schedule_id()
        if not sid or not self._app:
            return
        schedules = self._schedule_list()
        chosen = next((s for s in schedules if isinstance(s, dict) and str(s.get("id") or "") == sid), None)
        if not chosen:
            return
        lid = str(chosen.get("layout_id") or "").strip()
        if not lid:
            return
        action = str(chosen.get("action") or "load").strip().lower()
        policy = str(chosen.get("switch_policy") or "stop").strip().lower()
        try:
            if action in {"run", "run_background", "background"}:
                self._app._layout_start_in_background(lid)
            else:
                self._app._switch_layout_with_decision(lid, policy, source="scheduler")
        except Exception:
            return

    def _add(self):
        cur = {
            "id": f"sched_{uuid.uuid4().hex[:8]}",
            "enabled": True,
            "name": "New schedule",
            "layout_id": "",
            "type": "time_of_day",
            "days": [0, 1, 2, 3, 4, 5, 6],
            "start_time": "08:00",
            "end_time": "17:00",
            "action": "load",
            "switch_policy": "stop",
            "priority": 100,
            "cooldown_sec": 300,
        }
        out = self._edit_schedule_dialog(cur)
        if not out:
            return
        schedules = self._schedule_list()
        schedules.append(out)
        self._set_schedule_list(schedules)
        self._load()

    def _edit(self):
        sid = self._selected_schedule_id()
        if not sid:
            return
        schedules = self._schedule_list()
        idx = next((i for i, s in enumerate(schedules) if isinstance(s, dict) and str(s.get("id") or "") == sid), None)
        if idx is None:
            return
        out = self._edit_schedule_dialog(dict(schedules[idx]))
        if not out:
            return
        schedules[idx] = out
        self._set_schedule_list(schedules)
        self._load()

    def _delete(self):
        sid = self._selected_schedule_id()
        if not sid:
            return
        resp = QMessageBox.question(self, "Delete schedule", "Delete selected schedule?", QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel, QMessageBox.StandardButton.Cancel)
        if resp != QMessageBox.StandardButton.Yes:
            return
        schedules = [s for s in self._schedule_list() if not (isinstance(s, dict) and str(s.get("id") or "") == sid)]
        self._set_schedule_list(schedules)
        self._load()

    def _edit_schedule_dialog(self, schedule: dict) -> Optional[dict]:
        """
        Add/Edit schedule dialog.
        """
        dlg = QDialog(self)
        dlg.setWindowTitle("Edit Schedule")
        dlg.setWindowFlags(Qt.WindowType.Tool | Qt.WindowType.WindowStaysOnTopHint)
        dlg.resize(520, 420)
        root = QVBoxLayout(dlg)
        form = QFormLayout()
        root.addLayout(form)

        cb_enabled = QCheckBox("Enabled")
        cb_enabled.setChecked(bool(schedule.get("enabled", True)))
        form.addRow("", cb_enabled)

        name = QLineEdit(str(schedule.get("name") or ""))
        form.addRow("Name", name)

        # Layout
        layout_combo = QComboBox()
        layouts = self._layouts()
        for l in layouts:
            layout_combo.addItem(str(l.name or l.id), str(l.id))
        want = str(schedule.get("layout_id") or "")
        idx = layout_combo.findData(want)
        if idx >= 0:
            layout_combo.setCurrentIndex(idx)
        form.addRow("Layout", layout_combo)

        # Type
        type_combo = QComboBox()
        type_combo.addItem("Time of day (recurring)", "time_of_day")
        type_combo.addItem("After app uptime", "uptime")
        idx = type_combo.findData(str(schedule.get("type") or "time_of_day"))
        if idx >= 0:
            type_combo.setCurrentIndex(idx)
        form.addRow("Type", type_combo)

        # Time-of-day fields
        tod_box = QWidget()
        tod_form = QFormLayout(tod_box)
        days_row = QHBoxLayout()
        day_cbs: List[QCheckBox] = []
        labels = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
        days_saved = set(int(x) for x in (schedule.get("days") or []) if str(x).strip()) if isinstance(schedule.get("days"), list) else set()
        for i, lab in enumerate(labels):
            cb = QCheckBox(lab)
            cb.setChecked(i in days_saved if days_saved else True)
            day_cbs.append(cb)
            days_row.addWidget(cb)
        tod_form.addRow("Days", days_row)

        start_te = QTimeEdit()
        start_te.setDisplayFormat("HH:mm")
        end_te = QTimeEdit()
        end_te.setDisplayFormat("HH:mm")
        try:
            from PySide6.QtCore import QTime
            st = str(schedule.get("start_time") or "08:00")
            en = str(schedule.get("end_time") or "17:00")
            st_h, st_m = [int(x) for x in st.split(":")]
            en_h, en_m = [int(x) for x in en.split(":")]
            start_te.setTime(QTime(st_h, st_m))
            end_te.setTime(QTime(en_h, en_m))
        except Exception:
            pass
        tod_form.addRow("Start", start_te)
        tod_form.addRow("End", end_te)
        form.addRow(tod_box)

        # Uptime fields
        up_box = QWidget()
        up_form = QFormLayout(up_box)
        after = QSpinBox()
        after.setRange(5, 24 * 3600)
        after.setSuffix(" s")
        try:
            after.setValue(int(float(schedule.get("after_sec") or 300)))
        except Exception:
            after.setValue(300)
        once = QCheckBox("Once per launch")
        once.setChecked(bool(schedule.get("once_per_launch", True)))
        up_form.addRow("After uptime", after)
        up_form.addRow("", once)
        form.addRow(up_box)

        # Action + policy
        action_combo = QComboBox()
        action_combo.addItem("Load (becomes current)", "load")
        action_combo.addItem("Run in background", "run_background")
        idx = action_combo.findData(str(schedule.get("action") or "load"))
        if idx >= 0:
            action_combo.setCurrentIndex(idx)
        form.addRow("Action", action_combo)

        policy_combo = QComboBox()
        policy_combo.addItem("Stop previous", "stop")
        policy_combo.addItem("Keep previous running", "keep")
        idx = policy_combo.findData(str(schedule.get("switch_policy") or "stop"))
        if idx >= 0:
            policy_combo.setCurrentIndex(idx)
        form.addRow("Switch policy", policy_combo)

        pr = QSpinBox()
        pr.setRange(0, 10000)
        pr.setValue(int(schedule.get("priority", 100) or 100))
        form.addRow("Priority", pr)

        cd = QSpinBox()
        cd.setRange(0, 24 * 3600)
        cd.setSuffix(" s")
        cd.setValue(int(schedule.get("cooldown_sec", 300) or 300))
        form.addRow("Cooldown", cd)

        # Toggle visibility based on type
        def _sync():
            stype = str(type_combo.currentData() or "time_of_day")
            tod_box.setVisible(stype == "time_of_day")
            up_box.setVisible(stype == "uptime")
        type_combo.currentIndexChanged.connect(_sync)
        _sync()

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        root.addWidget(buttons)
        buttons.accepted.connect(dlg.accept)
        buttons.rejected.connect(dlg.reject)
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return None

        out = dict(schedule)
        out["enabled"] = bool(cb_enabled.isChecked())
        out["name"] = str(name.text() or "").strip() or str(out.get("id") or "schedule")
        out["layout_id"] = str(layout_combo.currentData() or "").strip()
        out["type"] = str(type_combo.currentData() or "time_of_day")
        out["action"] = str(action_combo.currentData() or "load")
        out["switch_policy"] = str(policy_combo.currentData() or "stop")
        out["priority"] = int(pr.value())
        out["cooldown_sec"] = int(cd.value())

        if out["type"] == "time_of_day":
            out["days"] = [i for i, cb in enumerate(day_cbs) if cb.isChecked()]
            out["start_time"] = start_te.time().toString("HH:mm")
            out["end_time"] = end_te.time().toString("HH:mm")
            out.pop("after_sec", None)
            out.pop("once_per_launch", None)
        else:
            out["after_sec"] = int(after.value())
            out["once_per_launch"] = bool(once.isChecked())
            out.pop("days", None)
            out.pop("start_time", None)
            out.pop("end_time", None)

        # Basic validation
        if not out.get("layout_id"):
            QMessageBox.warning(self, "Schedule", "Choose a layout.")
            return None
        return out


