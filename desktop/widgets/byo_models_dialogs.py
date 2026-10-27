from __future__ import annotations

import shutil
import threading
from pathlib import Path
from typing import Optional

from PySide6.QtCore import Qt, QUrl
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import (
    QCheckBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
)

from core.model_library.byo_install import import_local_onnx, install_from_huggingface
from core.model_library.byo_models import BYOModelManifest, byo_models_dir, list_installed_manifests, load_manifest
from core.model_library.hf_metadata import try_get_hf_license
from core.model_library.license_acceptance import is_license_accepted, record_license_acceptance


def _open_folder(path: Path) -> None:
    try:
        # Create the directory if it doesn't exist so "Open folder" never becomes a no-op.
        path.mkdir(parents=True, exist_ok=True)
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(path.resolve())))
    except Exception:
        pass


class LicenseAcceptanceDialog(QDialog):
    def __init__(self, manifest: BYOModelManifest, parent=None):
        super().__init__(parent)
        self.manifest = manifest
        self.setWindowTitle("Accept Model License")
        self.setWindowFlags(Qt.WindowType.Tool | Qt.WindowType.WindowStaysOnTopHint)
        self.resize(520, 240)

        root = QVBoxLayout(self)
        title = QLabel(f"<b>{manifest.display_name}</b>")
        title.setTextFormat(Qt.TextFormat.RichText)
        root.addWidget(title)

        msg = QLabel(
            "Model licenses are between you and the model provider. "
            "You must review and accept the terms before enabling this model."
        )
        msg.setWordWrap(True)
        msg.setStyleSheet("color: #cbd5e1;")
        root.addWidget(msg)

        lic_lines = []
        if manifest.license and manifest.license.spdx:
            lic_lines.append(f"SPDX: {manifest.license.spdx}")
        if manifest.license and manifest.license.url:
            lic_lines.append(f"URL: {manifest.license.url}")
        if not lic_lines:
            lic_lines.append("License: (not provided). You must obtain the license terms from the provider.")
        lic = QLabel("\n".join(lic_lines))
        lic.setStyleSheet("color: #e5e7eb;")
        lic.setWordWrap(True)
        root.addWidget(lic)

        btn_row = QHBoxLayout()
        open_btn = QPushButton("Open License URL")
        open_btn.setEnabled(bool(manifest.license and manifest.license.url))
        open_btn.clicked.connect(lambda: QDesktopServices.openUrl(QUrl(str(manifest.license.url))) if manifest.license and manifest.license.url else None)
        btn_row.addWidget(open_btn)
        btn_row.addStretch()
        root.addLayout(btn_row)

        self.accept_chk = QCheckBox("I accept this model license")
        root.addWidget(self.accept_chk)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        root.addWidget(buttons)
        buttons.accepted.connect(self._on_accept)
        buttons.rejected.connect(self.reject)

    def _on_accept(self):
        if not bool(self.accept_chk.isChecked()):
            QMessageBox.warning(self, "License acceptance", "You must check the acceptance box to enable this model.")
            return
        record_license_acceptance(self.manifest)
        self.accept()


class InstallFromHuggingFaceDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Install Model from Hugging Face")
        self.setWindowFlags(Qt.WindowType.Tool | Qt.WindowType.WindowStaysOnTopHint)
        self.resize(560, 320)

        root = QVBoxLayout(self)
        form = QFormLayout()
        root.addLayout(form)

        self.repo_id = QLineEdit()
        self.repo_id.setPlaceholderText("e.g. org/model-repo")
        form.addRow("Repo ID", self.repo_id)

        self.revision = QLineEdit()
        self.revision.setPlaceholderText("optional (branch/tag/commit)")
        form.addRow("Revision", self.revision)

        self.model_file = QLineEdit()
        self.model_file.setPlaceholderText("e.g. model.onnx (leave blank to auto-detect)")
        form.addRow("Model file (.onnx)", self.model_file)

        self.labels_file = QLineEdit()
        self.labels_file.setPlaceholderText("optional (labels.json / labels.txt)")
        form.addRow("Labels file", self.labels_file)

        self.display_name = QLineEdit()
        self.display_name.setPlaceholderText("optional (friendly name)")
        form.addRow("Display name", self.display_name)

        lic_row = QHBoxLayout()
        self.license_spdx = QLineEdit()
        self.license_spdx.setPlaceholderText("optional SPDX (e.g. apache-2.0)")
        lic_row.addWidget(self.license_spdx, stretch=2)
        self.license_url = QLineEdit()
        self.license_url.setPlaceholderText("optional license URL")
        lic_row.addWidget(self.license_url, stretch=3)
        self.fetch_btn = QPushButton("Fetch license")
        lic_row.addWidget(self.fetch_btn)
        form.addRow("License", lic_row)

        note = QLabel("You’ll be required to accept the model license before enabling it.")
        note.setStyleSheet("color: #cbd5e1;")
        root.addWidget(note)

        self.status = QLabel("")
        self.status.setStyleSheet("color: #9aa4b2;")
        root.addWidget(self.status)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.button(QDialogButtonBox.StandardButton.Ok).setText("Install")
        root.addWidget(buttons)
        buttons.accepted.connect(self._install)
        buttons.rejected.connect(self.reject)

        self.fetch_btn.clicked.connect(self._fetch_license)

    def _fetch_license(self):
        repo = str(self.repo_id.text() or "").strip()
        rev = str(self.revision.text() or "").strip() or None
        info = try_get_hf_license(repo, revision=rev)
        if info.license_spdx and not self.license_spdx.text().strip():
            self.license_spdx.setText(info.license_spdx)
        if info.license_url and not self.license_url.text().strip():
            self.license_url.setText(info.license_url)
        self.status.setText("Fetched HF metadata (best effort).")

    def _install(self):
        repo = str(self.repo_id.text() or "").strip()
        rev = str(self.revision.text() or "").strip() or None
        model_file = str(self.model_file.text() or "").strip()
        labels_file = str(self.labels_file.text() or "").strip() or None
        display = str(self.display_name.text() or "").strip() or None
        lic_spdx = str(self.license_spdx.text() or "").strip() or None
        lic_url = str(self.license_url.text() or "").strip() or None

        self.status.setText("Downloading…")

        def _bg():
            try:
                res = install_from_huggingface(
                    repo_id=repo,
                    revision=rev,
                    model_filename=model_file,
                    labels_filename=labels_file,
                    display_name=display,
                    license_spdx=lic_spdx,
                    license_url=lic_url,
                )
                def _ui_ok():
                    QMessageBox.information(self, "Install complete", f"✓ Installed: {res.slug}\n{res.model_path}")
                    # Make the flow obvious: offer to enable immediately.
                    try:
                        from PySide6.QtWidgets import QApplication

                        app = QApplication.instance()
                        if app is not None and hasattr(app, "_set_detector_prefs"):
                            resp = QMessageBox.question(
                                self,
                                "Enable detector now?",
                                "Set this model as the active detector model now?\n\n"
                                "Note: you may be prompted to accept the model license before enabling.",
                                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                                QMessageBox.StandardButton.Yes,
                            )
                            if resp == QMessageBox.StandardButton.Yes:
                                try:
                                    if hasattr(app, "_ensure_model_license_accepted") and not app._ensure_model_license_accepted(res.slug):
                                        return
                                except Exception:
                                    return
                                try:
                                    app._set_detector_prefs(backend="auto", model=res.slug)
                                    if hasattr(app, "_apply_detector_prefs_to_open_cameras"):
                                        app._apply_detector_prefs_to_open_cameras()
                                except Exception:
                                    pass
                    except Exception:
                        pass
                    self.accept()
                self.status.setText("")
                self._call_ui(_ui_ok)
            except Exception as e:
                def _ui_err():
                    QMessageBox.warning(self, "Install failed", str(e))
                self._call_ui(_ui_err)

        threading.Thread(target=_bg, daemon=True).start()

    def _call_ui(self, fn):
        try:
            from PySide6.QtCore import QTimer
            QTimer.singleShot(0, fn)
        except Exception:
            fn()


class ImportLocalOnnxDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Import Local ONNX Model")
        self.setWindowFlags(Qt.WindowType.Tool | Qt.WindowType.WindowStaysOnTopHint)
        self.resize(560, 260)

        root = QVBoxLayout(self)
        form = QFormLayout()
        root.addLayout(form)

        path_row = QHBoxLayout()
        self.onnx_path = QLineEdit()
        self.onnx_path.setPlaceholderText("Select a .onnx file…")
        browse = QPushButton("Browse…")
        browse.clicked.connect(self._browse_onnx)
        path_row.addWidget(self.onnx_path, stretch=1)
        path_row.addWidget(browse)
        form.addRow("ONNX file", path_row)

        lbl_row = QHBoxLayout()
        self.labels_path = QLineEdit()
        self.labels_path.setPlaceholderText("Optional labels.json / labels.txt")
        browse_lbl = QPushButton("Browse…")
        browse_lbl.clicked.connect(self._browse_labels)
        lbl_row.addWidget(self.labels_path, stretch=1)
        lbl_row.addWidget(browse_lbl)
        form.addRow("Labels file", lbl_row)

        self.display_name = QLineEdit()
        self.display_name.setPlaceholderText("optional (friendly name)")
        form.addRow("Display name", self.display_name)

        self.license_url = QLineEdit()
        self.license_url.setPlaceholderText("optional license URL")
        form.addRow("License URL", self.license_url)

        note = QLabel("You’ll be required to accept the model license before enabling it.")
        note.setStyleSheet("color: #cbd5e1;")
        root.addWidget(note)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.button(QDialogButtonBox.StandardButton.Ok).setText("Import")
        root.addWidget(buttons)
        buttons.accepted.connect(self._import)
        buttons.rejected.connect(self.reject)

    def _browse_onnx(self):
        p, _ = QFileDialog.getOpenFileName(self, "Select ONNX model", "", "ONNX Models (*.onnx)")
        if p:
            self.onnx_path.setText(p)

    def _browse_labels(self):
        p, _ = QFileDialog.getOpenFileName(self, "Select labels file", "", "Labels (*.json *.txt)")
        if p:
            self.labels_path.setText(p)

    def _import(self):
        onnx = str(self.onnx_path.text() or "").strip()
        labels = str(self.labels_path.text() or "").strip() or None
        display = str(self.display_name.text() or "").strip() or None
        lic_url = str(self.license_url.text() or "").strip() or None
        try:
            res = import_local_onnx(
                onnx_path=onnx,
                labels_path=labels,
                display_name=display,
                license_url=lic_url,
            )
            QMessageBox.information(self, "Import complete", f"✓ Imported: {res.slug}\n{res.model_path}")
            # Make the flow obvious: offer to enable immediately.
            try:
                from PySide6.QtWidgets import QApplication

                app = QApplication.instance()
                if app is not None and hasattr(app, "_set_detector_prefs"):
                    resp = QMessageBox.question(
                        self,
                        "Enable detector now?",
                        "Set this model as the active detector model now?\n\n"
                        "Note: you may be prompted to accept the model license before enabling.",
                        QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                        QMessageBox.StandardButton.Yes,
                    )
                    if resp == QMessageBox.StandardButton.Yes:
                        try:
                            if hasattr(app, "_ensure_model_license_accepted") and not app._ensure_model_license_accepted(res.slug):
                                return
                        except Exception:
                            return
                        try:
                            app._set_detector_prefs(backend="auto", model=res.slug)
                            if hasattr(app, "_apply_detector_prefs_to_open_cameras"):
                                app._apply_detector_prefs_to_open_cameras()
                        except Exception:
                            pass
            except Exception:
                pass
            self.accept()
        except Exception as e:
            QMessageBox.warning(self, "Import failed", str(e))


class ManageInstalledModelsDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Manage Installed Models")
        self.setWindowFlags(Qt.WindowType.Tool | Qt.WindowType.WindowStaysOnTopHint)
        self.resize(640, 420)

        root = QVBoxLayout(self)

        top = QHBoxLayout()
        top.addWidget(QLabel("Installed BYO models (models/byo/)"))
        top.addStretch()
        open_root = QPushButton("Open models folder")
        open_root.clicked.connect(lambda: _open_folder(byo_models_dir()))
        top.addWidget(open_root)
        root.addLayout(top)

        self.list = QListWidget()
        root.addWidget(self.list, stretch=1)

        btns = QHBoxLayout()
        self.accept_btn = QPushButton("Accept license…")
        self.license_btn = QPushButton("Open license URL")
        self.folder_btn = QPushButton("Open folder")
        self.remove_btn = QPushButton("Remove")
        btns.addWidget(self.accept_btn)
        btns.addWidget(self.license_btn)
        btns.addStretch()
        btns.addWidget(self.folder_btn)
        btns.addWidget(self.remove_btn)
        root.addLayout(btns)

        self.accept_btn.clicked.connect(self._accept_selected)
        self.license_btn.clicked.connect(self._open_license)
        self.folder_btn.clicked.connect(self._open_selected_folder)
        self.remove_btn.clicked.connect(self._remove_selected)

        self.refresh()

    def refresh(self):
        self.list.clear()
        for m in list_installed_manifests():
            accepted = is_license_accepted(m)
            suffix = "✓ license accepted" if accepted else "license NOT accepted"
            it = QListWidgetItem(f"{m.display_name}  ({m.slug}) — {suffix}")
            it.setData(Qt.ItemDataRole.UserRole, m.slug)
            self.list.addItem(it)

    def _selected_slug(self) -> Optional[str]:
        it = self.list.currentItem()
        if not it:
            return None
        return str(it.data(Qt.ItemDataRole.UserRole) or "").strip() or None

    def _selected_manifest(self) -> Optional[BYOModelManifest]:
        slug = self._selected_slug()
        return load_manifest(slug) if slug else None

    def _open_selected_folder(self):
        m = self._selected_manifest()
        if not m:
            return
        _open_folder((byo_models_dir() / m.slug))

    def _open_license(self):
        m = self._selected_manifest()
        if not m or not m.license or not m.license.url:
            QMessageBox.information(self, "License", "No license URL is recorded for this model.")
            return
        QDesktopServices.openUrl(QUrl(str(m.license.url)))

    def _accept_selected(self):
        m = self._selected_manifest()
        if not m:
            return
        if is_license_accepted(m):
            QMessageBox.information(self, "License", "License already accepted for this model.")
            return
        dlg = LicenseAcceptanceDialog(m, self)
        if dlg.exec() == QDialog.DialogCode.Accepted:
            self.refresh()

    def _remove_selected(self):
        slug = self._selected_slug()
        if not slug:
            return
        resp = QMessageBox.question(
            self,
            "Remove model",
            f"Remove installed model '{slug}'?\n\nThis deletes: models/byo/{slug}/",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if resp != QMessageBox.StandardButton.Yes:
            return
        try:
            shutil.rmtree(str(byo_models_dir() / slug), ignore_errors=False)
        except Exception as e:
            QMessageBox.warning(self, "Remove failed", str(e))
            return
        self.refresh()

