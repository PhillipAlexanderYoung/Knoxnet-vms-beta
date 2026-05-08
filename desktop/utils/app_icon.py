from __future__ import annotations

import sys
from pathlib import Path
from textwrap import dedent

from PySide6.QtGui import QGuiApplication, QIcon
from PySide6.QtWidgets import QApplication

APP_ID = "knoxnetvmsbeta"
APP_DISPLAY_NAME = "Knoxnet VMS Beta"


def _icon_candidates(prefer_tray: bool = False) -> list[Path]:
    source_assets = Path(__file__).resolve().parent.parent / "assets"
    source_candidates = [
        source_assets / "knoxnet_icon.png",
        source_assets / "knoxnet_appimage.png",
        source_assets / "knoxnet_tray.png",
    ]
    if prefer_tray:
        source_candidates = [
            source_assets / "knoxnet_tray.png",
            source_assets / "knoxnet_icon.png",
            source_assets / "knoxnet_appimage.png",
        ]

    candidates = list(source_candidates)
    if getattr(sys, "frozen", False):
        base = Path(sys._MEIPASS) if hasattr(sys, "_MEIPASS") else Path(sys.executable).resolve().parent
        frozen_assets = base / "desktop" / "assets"
        frozen_candidates = [
            frozen_assets / "knoxnet_icon.png",
            frozen_assets / "knoxnet_appimage.png",
            frozen_assets / "knoxnet_tray.png",
        ]
        if prefer_tray:
            frozen_candidates = [
                frozen_assets / "knoxnet_tray.png",
                frozen_assets / "knoxnet_icon.png",
                frozen_assets / "knoxnet_appimage.png",
            ]
        candidates = frozen_candidates + candidates
    return candidates


def _best_icon_path(prefer_tray: bool = False) -> Path | None:
    for path in _icon_candidates(prefer_tray=prefer_tray):
        if path.exists():
            return path
    return None


def ensure_linux_desktop_entry() -> Path | None:
    if not sys.platform.startswith("linux"):
        return None

    icon_path = _best_icon_path()
    if icon_path is None:
        return None

    desktop_dir = Path.home() / ".local" / "share" / "applications"
    desktop_path = desktop_dir / f"{APP_ID}.desktop"

    try:
        desktop_dir.mkdir(parents=True, exist_ok=True)
    except Exception:
        return None

    try:
        if getattr(sys, "frozen", False):
            exec_line = str(Path(sys.executable).resolve())
        else:
            repo_root = Path(__file__).resolve().parents[2]
            exec_line = f"{Path(sys.executable).resolve()} {repo_root / 'start_desktop.py'}"

        content = dedent(
            f"""\
            [Desktop Entry]
            Type=Application
            Version=1.0
            Name={APP_DISPLAY_NAME}
            Comment=Knoxnet VMS Beta Desktop
            Exec={exec_line}
            Icon={icon_path}
            Terminal=false
            Categories=Utility;AudioVideo;Video;
            StartupNotify=true
            StartupWMClass={APP_ID}
            """
        )
        if not desktop_path.exists() or desktop_path.read_text(encoding="utf-8") != content:
            desktop_path.write_text(content, encoding="utf-8")
        return desktop_path
    except Exception:
        return None


def load_knoxnet_icon(prefer_tray: bool = False) -> QIcon:
    for path in _icon_candidates(prefer_tray=prefer_tray):
        if not path.exists():
            continue
        icon = QIcon(str(path))
        if not icon.isNull():
            return icon

    app = QApplication.instance()
    if app is not None:
        return app.style().standardIcon(app.style().StandardPixmap.SP_ComputerIcon)
    return QIcon()


def apply_app_icon(app: QApplication | None = None) -> QIcon:
    icon = load_knoxnet_icon()
    target = app or QApplication.instance()
    ensure_linux_desktop_entry()
    if target is not None and not icon.isNull():
        try:
            target.setApplicationName(APP_ID)
        except Exception:
            pass
        try:
            target.setApplicationDisplayName(APP_DISPLAY_NAME)
        except Exception:
            pass
        try:
            target.setWindowIcon(icon)
        except Exception:
            pass
        try:
            QGuiApplication.setDesktopFileName(APP_ID)
        except Exception:
            pass
    return icon


def apply_window_icon(window) -> QIcon:
    icon = load_knoxnet_icon()
    if window is not None and not icon.isNull():
        try:
            window.setWindowIcon(icon)
        except Exception:
            pass
    return icon
