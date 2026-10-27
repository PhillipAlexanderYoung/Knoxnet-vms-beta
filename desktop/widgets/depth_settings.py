from __future__ import annotations

import threading
from typing import Optional

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QSlider,
    QSpinBox,
    QVBoxLayout,
)

from core.model_library.depth_anything_v2 import ensure_depth_anything_v2_weights
from desktop.utils.depth_worker import DepthOverlayConfig


class DepthOverlaySettingsDialog(QDialog):
    def __init__(self, config: DepthOverlayConfig, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Depth Overlay Settings (DepthAnythingV2)")
        self.setWindowFlags(Qt.WindowType.Tool | Qt.WindowType.WindowStaysOnTopHint)
        self._config = config
        self.result_config: Optional[DepthOverlayConfig] = None

        root = QVBoxLayout(self)
        form = QFormLayout()

        # FPS
        self.fps_spin = QSpinBox()
        self.fps_spin.setRange(1, 60)
        self.fps_spin.setValue(int(config.fps_limit))
        form.addRow("FPS limit", self.fps_spin)

        # Opacity
        self.opacity_slider = QSlider(Qt.Orientation.Horizontal)
        self.opacity_slider.setRange(0, 100)
        self.opacity_slider.setValue(int(max(0.0, min(1.0, float(config.opacity))) * 100))
        self.opacity_label = QLabel(f"{self.opacity_slider.value()}%")
        self.opacity_slider.valueChanged.connect(lambda v: self.opacity_label.setText(f"{v}%"))
        op_row = QHBoxLayout()
        op_row.addWidget(self.opacity_slider)
        op_row.addWidget(self.opacity_label)
        form.addRow("Visualization opacity", op_row)

        # Camera opacity (lets the camera be faded under/behind the visualization)
        cam_opacity = float(getattr(config, "camera_opacity", 1.0) or 1.0)
        cam_opacity = max(0.0, min(1.0, cam_opacity))
        self.camera_opacity_slider = QSlider(Qt.Orientation.Horizontal)
        self.camera_opacity_slider.setRange(0, 100)
        self.camera_opacity_slider.setValue(int(round(cam_opacity * 100)))
        self.camera_opacity_label = QLabel(f"{self.camera_opacity_slider.value()}%")
        self.camera_opacity_slider.valueChanged.connect(lambda v: self.camera_opacity_label.setText(f"{v}%"))
        cam_row = QHBoxLayout()
        cam_row.addWidget(self.camera_opacity_slider)
        cam_row.addWidget(self.camera_opacity_label)
        form.addRow("Camera opacity", cam_row)

        # Colormap
        self.cmap_combo = QComboBox()
        # Treat point cloud as another "colormap" option
        self.cmap_combo.addItem("pointcloud (1st person)", userData="pointcloud")
        for name in ["turbo", "jet", "viridis", "plasma", "inferno", "magma", "bone", "ocean"]:
            self.cmap_combo.addItem(name, userData=name)
        cur = self.cmap_combo.findText(config.colormap)
        if cur >= 0:
            self.cmap_combo.setCurrentIndex(cur)
        form.addRow("Visualization", self.cmap_combo)

        # Point cloud density slider (user-friendly). Higher = denser.
        # Internally we store `pointcloud_step` (lower = denser). Map: density 1..24 → step (25-density).
        current_step = int(getattr(config, "pointcloud_step", 3) or 3)
        current_step = max(1, min(24, current_step))
        current_density = max(1, min(24, 25 - current_step))

        self.pc_density_slider = QSlider(Qt.Orientation.Horizontal)
        self.pc_density_slider.setRange(1, 24)
        self.pc_density_slider.setValue(int(current_density))
        self.pc_density_label = QLabel("")

        def _pc_set_label(v: int):
            step = max(1, min(24, 25 - int(v)))
            self.pc_density_label.setText(f"{int(v)}  (step={step}px)")

        self.pc_density_slider.valueChanged.connect(_pc_set_label)
        _pc_set_label(self.pc_density_slider.value())

        dens_row = QHBoxLayout()
        dens_row.addWidget(self.pc_density_slider)
        dens_row.addWidget(self.pc_density_label)
        form.addRow("Point cloud density", dens_row)

        # Point cloud color mode (only used when viz == pointcloud)
        self.pc_color_combo = QComboBox()
        self.pc_color_combo.addItem("Camera colors", userData="camera")
        self.pc_color_combo.addItem("Depth colors (turbo)", userData="depth_turbo")
        self.pc_color_combo.addItem("Depth colors (viridis)", userData="depth_viridis")
        self.pc_color_combo.addItem("Depth colors (inferno)", userData="depth_inferno")
        self.pc_color_combo.addItem("White", userData="white")
        current_pc_color = str(getattr(config, "pointcloud_color", "camera") or "camera")
        for i in range(self.pc_color_combo.count()):
            if str(self.pc_color_combo.itemData(i)) == current_pc_color:
                self.pc_color_combo.setCurrentIndex(i)
                break
        form.addRow("Point cloud color", self.pc_color_combo)

        # Point cloud zoom (matches React)
        current_zoom = float(getattr(config, "pointcloud_zoom", 1.0) or 1.0)
        current_zoom = max(0.5, min(2.5, current_zoom))
        self.pc_zoom_slider = QSlider(Qt.Orientation.Horizontal)
        self.pc_zoom_slider.setRange(5, 25)  # 0.5..2.5 in 0.1 steps
        self.pc_zoom_slider.setValue(int(round(current_zoom * 10)))
        self.pc_zoom_label = QLabel(f"{current_zoom:.1f}x")
        self.pc_zoom_slider.valueChanged.connect(lambda v: self.pc_zoom_label.setText(f"{(v/10.0):.1f}x"))
        zoom_row = QHBoxLayout()
        zoom_row.addWidget(self.pc_zoom_slider)
        zoom_row.addWidget(self.pc_zoom_label)
        form.addRow("Point cloud zoom", zoom_row)

        # Overlay scale (scales the visualization relative to the camera view)
        current_overlay_scale = float(getattr(config, "overlay_scale", 1.0) or 1.0)
        current_overlay_scale = max(0.5, min(2.5, current_overlay_scale))
        self.overlay_scale_slider = QSlider(Qt.Orientation.Horizontal)
        self.overlay_scale_slider.setRange(5, 25)  # 0.5..2.5 in 0.1 steps
        self.overlay_scale_slider.setValue(int(round(current_overlay_scale * 10)))
        self.overlay_scale_label = QLabel(f"{current_overlay_scale:.1f}x")
        self.overlay_scale_slider.valueChanged.connect(lambda v: self.overlay_scale_label.setText(f"{(v/10.0):.1f}x"))
        os_row = QHBoxLayout()
        os_row.addWidget(self.overlay_scale_slider)
        os_row.addWidget(self.overlay_scale_label)
        form.addRow("Visualization scale", os_row)

        # Blackout base camera image (show visualization only)
        self.blackout_chk = QCheckBox("Blackout camera image (show visualization only)")
        self.blackout_chk.setChecked(bool(getattr(config, "blackout_base", False)))
        form.addRow(self.blackout_chk)

        # Model size
        self.size_combo = QComboBox()
        self.size_combo.addItem("vits (Small / fast)", userData="vits")
        self.size_combo.addItem("vitb (Base / balanced)", userData="vitb")
        self.size_combo.addItem("vitl (Large / accurate)", userData="vitl")
        # Set current
        for i in range(self.size_combo.count()):
            if self.size_combo.itemData(i) == config.model_size:
                self.size_combo.setCurrentIndex(i)
                break
        form.addRow("Model size", self.size_combo)

        # Device
        self.device_combo = QComboBox()
        self.device_combo.addItem("auto", userData="auto")
        self.device_combo.addItem("cuda", userData="cuda")
        self.device_combo.addItem("cpu", userData="cpu")
        for i in range(self.device_combo.count()):
            if self.device_combo.itemData(i) == config.device:
                self.device_combo.setCurrentIndex(i)
                break
        form.addRow("Device", self.device_combo)

        # FP16 / optimize
        self.fp16_chk = QCheckBox("Use FP16 (faster on CUDA)")
        self.fp16_chk.setChecked(bool(config.use_fp16))
        form.addRow(self.fp16_chk)

        self.opt_chk = QCheckBox("Enable optimizations (cuDNN benchmark / etc)")
        self.opt_chk.setChecked(bool(config.optimize))
        form.addRow(self.opt_chk)

        root.addLayout(form)

        # Download/verify button
        self.download_btn = QPushButton("Download / verify weights")
        self.download_btn.clicked.connect(self._download_weights)
        root.addWidget(self.download_btn)

        # Buttons
        row = QHBoxLayout()
        row.addStretch()
        ok_btn = QPushButton("Save")
        ok_btn.clicked.connect(self._accept)
        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)
        row.addWidget(ok_btn)
        row.addWidget(cancel_btn)
        root.addLayout(row)

    def _selected_model_size(self) -> str:
        return str(self.size_combo.currentData() or "vits")

    def _download_weights(self):
        size = self._selected_model_size()
        self.download_btn.setEnabled(False)
        self.download_btn.setText("Downloading…")

        def worker():
            try:
                ensure_depth_anything_v2_weights(model_size=size)  # type: ignore[arg-type]
                QMessageBox.information(self, "DepthAnythingV2", f"✓ Weights ready for {size}")
            except Exception as e:
                QMessageBox.warning(self, "DepthAnythingV2", f"Failed to download weights:\n{e}")
            finally:
                self.download_btn.setEnabled(True)
                self.download_btn.setText("Download / verify weights")

        threading.Thread(target=worker, daemon=True).start()

    def _accept(self):
        # Resolve cmap value from userData (handles pointcloud label)
        cmap = self.cmap_combo.currentData()
        cmap = str(cmap or self.cmap_combo.currentText() or "turbo").strip().split(" ", 1)[0]
        # Density slider → internal step
        density = int(getattr(self, "pc_density_slider", None).value()) if hasattr(self, "pc_density_slider") else 22
        pointcloud_step = max(1, min(24, 25 - int(density)))
        pc_color = str(getattr(self, "pc_color_combo", None).currentData()) if hasattr(self, "pc_color_combo") else "camera"
        pc_zoom = float(getattr(self, "pc_zoom_slider", None).value() / 10.0) if hasattr(self, "pc_zoom_slider") else 1.0
        overlay_scale = float(getattr(self, "overlay_scale_slider", None).value() / 10.0) if hasattr(self, "overlay_scale_slider") else 1.0
        blackout_base = bool(getattr(self, "blackout_chk", None).isChecked()) if hasattr(self, "blackout_chk") else False
        camera_opacity = float(getattr(self, "camera_opacity_slider", None).value() / 100.0) if hasattr(self, "camera_opacity_slider") else 1.0
        self.result_config = DepthOverlayConfig(
            enabled=self._config.enabled,
            fps_limit=int(self.fps_spin.value()),
            opacity=float(self.opacity_slider.value() / 100.0),
            colormap=cmap,  # type: ignore[arg-type]
            pointcloud_step=int(pointcloud_step),
            pointcloud_color=pc_color,  # type: ignore[arg-type]
            pointcloud_zoom=float(pc_zoom),
            overlay_scale=float(overlay_scale),
            blackout_base=bool(blackout_base),
            camera_opacity=float(camera_opacity),
            model_size=self._selected_model_size(),  # type: ignore[arg-type]
            device=str(self.device_combo.currentData() or "cuda"),
            use_fp16=bool(self.fp16_chk.isChecked()),
            optimize=bool(self.opt_chk.isChecked()),
            memory_fraction=self._config.memory_fraction,
        )
        self.accept()


