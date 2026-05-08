from __future__ import annotations

import threading
from pathlib import Path

from PySide6.QtCore import Qt, QUrl
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import (
    QDialog,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
)

from core.model_library.depth_anything_v2 import (
    depth_anything_weights_path,
    ensure_depth_anything_v2_weights,
)
from core.model_library.catalog import ensure_model_installed, list_catalog_entries
from core.model_library.providers import get_provider_statuses
from core.model_library.vision_detection import ensure_mobilenetssd
from core.model_library.store import load_model_library_state
from core.model_library.vision_detection import (
    check_mobilenetssd,
)


class ModelLibraryDialog(QDialog):
    """
    Minimal “model library” UI.

    Today: supports installing DepthAnythingV2 weights (Small/Base/Large).
    Next: we’ll extend this to additional vision + LLM runners using the same `core/model_library` pattern.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Model Library")
        self.setWindowFlags(Qt.WindowType.Tool | Qt.WindowType.WindowStaysOnTopHint)
        self.resize(520, 360)

        root = QVBoxLayout(self)
        root.addWidget(QLabel("Installed / available models (Hugging Face)"))

        self.list = QListWidget()
        root.addWidget(self.list, stretch=1)

        btn_row = QHBoxLayout()
        self.refresh_btn = QPushButton("Refresh")
        self.refresh_btn.clicked.connect(self.refresh)
        self.install_btn = QPushButton("Install / Verify Selected")
        self.install_btn.clicked.connect(self._install_selected)
        self.info_btn = QPushButton("Info")
        self.info_btn.clicked.connect(self._show_selected_info)
        self.open_folder_btn = QPushButton("Open models folder")
        self.open_folder_btn.clicked.connect(self._open_models_folder)
        btn_row.addWidget(self.refresh_btn)
        btn_row.addWidget(self.install_btn)
        btn_row.addWidget(self.info_btn)
        btn_row.addStretch()
        btn_row.addWidget(self.open_folder_btn)
        root.addLayout(btn_row)

        self.status = QLabel("")
        self.status.setStyleSheet("color: #9aa4b2;")
        root.addWidget(self.status)

        self.list.itemDoubleClicked.connect(lambda _item: self._show_selected_info())

        self.refresh()

    def refresh(self):
        self.list.clear()

        state = load_model_library_state()
        installed = (state.get("installed") or {}) if isinstance(state, dict) else {}

        # --- Local vision models ---
        self.list.addItem(QListWidgetItem("— Vision (Local) —"))

        # MobileNetSSD
        mn = check_mobilenetssd()
        mn_meta = "tracked" if isinstance(installed, dict) and "mobilenetssd" in installed else "untracked"
        mn_item = QListWidgetItem(f"MobileNetSSD (Caffe) — {mn.status} ({mn_meta})")
        mn_item.setData(Qt.ItemDataRole.UserRole, {"kind": "mobilenetssd"})
        self.list.addItem(mn_item)

        # YOLOv8 variants
        byo_item = QListWidgetItem("YOLO (ONNXRuntime) — bring your own model (use tray: Models ▸ …)")
        byo_item.setData(Qt.ItemDataRole.UserRole, {"kind": "yolo_byo"})
        self.list.addItem(byo_item)

        # --- Hugging Face models ---
        self.list.addItem(QListWidgetItem("— Vision (Hugging Face) —"))

        def add_depth(size: str, label: str):
            p = depth_anything_weights_path(model_size=size)  # type: ignore[arg-type]
            exists = p.exists()
            key = f"depth_anything_v2_{size}"
            in_store = isinstance(installed, dict) and key in installed
            status = "installed" if exists else "missing"
            meta = "tracked" if in_store else "untracked"
            item = QListWidgetItem(f"DepthAnythingV2 {label}  —  {status} ({meta})")
            item.setData(Qt.ItemDataRole.UserRole, {"kind": "depth_anything_v2", "size": size, "path": str(p)})
            self.list.addItem(item)

        add_depth("vits", "vits (Small)")
        add_depth("vitb", "vitb (Base)")
        add_depth("vitl", "vitl (Large)")

        # --- Hugging Face (Image → Text) ---
        self.list.addItem(QListWidgetItem("— Vision (Hugging Face, Image → Text) —"))
        for entry in list_catalog_entries(modality="image_to_text"):
            missing = entry.missing_files()
            if entry.is_installed():
                status = "installed"
            elif entry.local_dir.exists() and missing:
                status = "partial"
            else:
                status = "missing"
            meta = "tracked" if isinstance(installed, dict) and entry.id in installed else "untracked"
            size_note = f"{entry.size_gb:.1f} GB" if entry.size_gb else "size unknown"
            gpu_note = "GPU" if entry.requires_gpu else "CPU/GPU"
            item = QListWidgetItem(
                f"{entry.display_name} — {status} ({meta})  [{size_note}, {gpu_note}]"
            )
            item.setData(
                Qt.ItemDataRole.UserRole,
                {
                    "kind": "catalog_model",
                    "id": entry.id,
                    "display_name": entry.display_name,
                    "repo_id": entry.repo_id,
                    "revision": entry.revision,
                    "local_dir": str(entry.local_dir),
                    "required_files": list(entry.required_files),
                    "missing_files": missing,
                    "size_gb": entry.size_gb,
                    "requires_gpu": entry.requires_gpu,
                    "min_vram_gb": entry.min_vram_gb,
                    "license_url": entry.license_url,
                },
            )
            self.list.addItem(item)

        # --- Multimodal API providers ---
        self.list.addItem(QListWidgetItem("— Multimodal APIs —"))
        for p in get_provider_statuses():
            status = "configured" if p.configured else "not configured"
            item = QListWidgetItem(f"{p.name} — {status}")
            item.setData(Qt.ItemDataRole.UserRole, {"kind": "provider", "id": p.id, "env_key": p.env_key, "source": p.source})
            self.list.addItem(item)

        self.status.setText(
            "Tip: YOLO uses ONNXRuntime BYO models via the tray menu. HF downloads use HF_TOKEN or data/llm_user_keys.json; APIs need keys in .env or data/llm_user_keys.json."
        )

    def _install_selected(self):
        item = self.list.currentItem()
        if not item:
            QMessageBox.information(self, "Model Library", "Select a model first.")
            return
        data = item.data(Qt.ItemDataRole.UserRole) or {}

        kind = data.get("kind")
        if kind == "depth_anything_v2":
            size = str(data.get("size") or "vits")
            self.install_btn.setEnabled(False)
            self.status.setText(f"Downloading/verifying DepthAnythingV2 {size}…")

            def worker():
                try:
                    p = ensure_depth_anything_v2_weights(model_size=size)  # type: ignore[arg-type]
                    QMessageBox.information(self, "Model Library", f"✓ Ready: {p}")
                except Exception as e:
                    QMessageBox.warning(self, "Model Library", f"Install failed:\n{e}")
                finally:
                    self.install_btn.setEnabled(True)
                    self.refresh()

            threading.Thread(target=worker, daemon=True).start()
            return

        if kind == "yolo_byo":
            QMessageBox.information(
                self,
                "YOLO (BYO)",
                "YOLO models are managed via the system tray:\n\n"
                "Models ▸ Install from Hugging Face…\n"
                "Models ▸ Import local ONNX…\n"
                "Models ▸ Manage installed models…",
            )
            return

        if kind == "mobilenetssd":
            st = check_mobilenetssd()
            if st.ok:
                QMessageBox.information(self, "Model Library", "✓ MobileNetSSD files are present in models/")
            else:
                self.install_btn.setEnabled(False)
                self.status.setText("Downloading/verifying MobileNetSSD…")

                def worker():
                    try:
                        ensure_mobilenetssd(force_download=False)
                        QMessageBox.information(
                            self,
                            "Model Library",
                            "✓ MobileNetSSD installed.\n\n"
                            "Tip: You can now enable Object Detection (MobileNet SSD) and it should work on CPU.",
                        )
                    except Exception as e:
                        QMessageBox.warning(self, "Model Library", f"MobileNetSSD install failed:\n{e}")
                    finally:
                        self.install_btn.setEnabled(True)
                        self.refresh()

                threading.Thread(target=worker, daemon=True).start()
            return

        if kind == "catalog_model":
            model_id = str(data.get("id") or "")
            if not model_id:
                QMessageBox.warning(self, "Model Library", "Missing model id.")
                return
            self.install_btn.setEnabled(False)
            self.status.setText(f"Downloading/verifying {model_id}…")

            def worker():
                try:
                    p = ensure_model_installed(model_id)
                    QMessageBox.information(self, "Model Library", f"✓ Ready: {p}")
                except Exception as e:
                    QMessageBox.warning(self, "Model Library", f"Install failed:\n{e}")
                finally:
                    self.install_btn.setEnabled(True)
                    self.refresh()

            threading.Thread(target=worker, daemon=True).start()
            return

        if kind == "provider":
            env_key = str(data.get("env_key") or "")
            if env_key:
                QMessageBox.information(
                    self,
                    "Model Library",
                    f"This provider is configured via {env_key} (or data/llm_user_keys.json).\n"
                    f"You can also set keys in the Desktop Terminal widget: Agent Settings.",
                )
            return

        QMessageBox.warning(self, "Model Library", "Unsupported selection.")
        return

        self.install_btn.setEnabled(False)
        self.install_btn.setEnabled(True)

    def _open_models_folder(self):
        # Cross-platform “open folder” best effort (writable per-user dir in frozen builds).
        from core.paths import get_models_dir

        folder = get_models_dir()
        try:
            folder.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass
        try:
            import subprocess
            import sys

            if sys.platform.startswith("linux"):
                subprocess.Popen(["xdg-open", str(folder)])
            elif sys.platform == "darwin":
                subprocess.Popen(["open", str(folder)])
            elif sys.platform.startswith("win"):
                subprocess.Popen(["explorer", str(folder)])
            else:
                raise RuntimeError("Unsupported platform")
        except Exception as e:
            QMessageBox.information(self, "Model Library", f"Models folder: {folder}\n\n(Unable to open automatically: {e})")

    def _show_selected_info(self):
        item = self.list.currentItem()
        if not item:
            QMessageBox.information(self, "Model Library", "Select a model first.")
            return
        data = item.data(Qt.ItemDataRole.UserRole) or {}
        kind = data.get("kind")

        title = "Model Information"
        text_lines = []
        license_url = None
        hf_url = None

        if kind == "catalog_model":
            title = str(data.get("display_name") or data.get("id") or "Model")
            model_id = str(data.get("id") or "")
            repo_id = str(data.get("repo_id") or "")
            revision = data.get("revision")
            local_dir = str(data.get("local_dir") or "")
            required_files = data.get("required_files") or []
            size_gb = data.get("size_gb")
            requires_gpu = bool(data.get("requires_gpu"))
            min_vram_gb = data.get("min_vram_gb")
            license_url = data.get("license_url")

            text_lines.append(f"Model ID: {model_id}")
            if repo_id:
                text_lines.append(f"HF Repo: {repo_id}")
                hf_url = f"https://huggingface.co/{repo_id}"
            if revision:
                text_lines.append(f"Revision: {revision}")
            if local_dir:
                text_lines.append(f"Local Path: {local_dir}")
            if size_gb:
                text_lines.append(f"Estimated Size: {size_gb:.1f} GB")
            text_lines.append("Runtime: GPU" if requires_gpu else "Runtime: CPU/GPU")
            if min_vram_gb:
                text_lines.append(f"Min VRAM: {min_vram_gb:.1f} GB")
            if required_files:
                text_lines.append("Required Files:")
                text_lines.extend([f"  - {name}" for name in required_files])
            missing_files = data.get("missing_files") or []
            if missing_files:
                text_lines.append("Missing Files:")
                text_lines.extend([f"  - {name}" for name in missing_files])
            if license_url:
                text_lines.append(f"License: {license_url}")
        elif kind == "depth_anything_v2":
            title = "DepthAnythingV2"
            text_lines.append(f"Variant: {data.get('size')}")
            if data.get("path"):
                text_lines.append(f"Local Path: {data.get('path')}")
            license_url = "https://github.com/DepthAnything/Depth-Anything-V2/blob/main/LICENSE"
            text_lines.append(f"License: {license_url}")
        elif kind == "yolov8":
            title = "YOLOv8"
            text_lines.append(f"Variant: {data.get('variant')}")
            text_lines.append("Local Path: models/<variant>.pt")
        elif kind == "mobilenetssd":
            title = "MobileNetSSD (Caffe)"
            text_lines.append("Local Path: models/mobilenet_iter_73000.caffemodel")
            text_lines.append("Requires: models/deploy1.prototxt (or MobileNetSSD_deploy.prototxt)")
        elif kind == "provider":
            title = str(data.get("id") or "Provider")
            env_key = str(data.get("env_key") or "")
            source = str(data.get("source") or "")
            if env_key:
                text_lines.append(f"Env Key: {env_key}")
            if source:
                text_lines.append(f"Configured via: {source}")
        else:
            QMessageBox.information(self, "Model Library", "No additional info available.")
            return

        msg = QMessageBox(self)
        msg.setWindowTitle(title)
        msg.setText("\n".join(text_lines))
        msg.setIcon(QMessageBox.Icon.Information)

        open_license_btn = None
        open_hf_btn = None
        if isinstance(license_url, str) and license_url.strip():
            open_license_btn = msg.addButton("Open License", QMessageBox.ButtonRole.ActionRole)
        if isinstance(hf_url, str) and hf_url.strip():
            open_hf_btn = msg.addButton("Open HF Page", QMessageBox.ButtonRole.ActionRole)
        msg.addButton(QMessageBox.StandardButton.Ok)

        msg.exec()
        clicked = msg.clickedButton()
        if open_license_btn and clicked is open_license_btn:
            QDesktopServices.openUrl(QUrl(str(license_url)))
        elif open_hf_btn and clicked is open_hf_btn:
            QDesktopServices.openUrl(QUrl(str(hf_url)))


