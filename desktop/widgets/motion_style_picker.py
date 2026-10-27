"""Tabbed icon-grid pickers for motion box styles and animations."""
from __future__ import annotations

from typing import Callable, Dict, List, Union

from PySide6.QtCore import Qt, Signal, QSize
from PySide6.QtGui import QIcon, QPixmap
from PySide6.QtWidgets import (
    QButtonGroup,
    QGridLayout,
    QScrollArea,
    QTabWidget,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from desktop.utils.motion_overlay_styles import (
    motion_animation_icon_pixmap,
    motion_animation_short_label,
    motion_style_icon_pixmap,
    motion_style_short_label,
)


class _IconGridPage(QWidget):
    """One category page: grid of icon buttons."""

    picked = Signal(str)

    def __init__(
        self,
        items: List[str],
        icon_fn: Callable[[str, int], Union[QPixmap, QIcon]],
        label_fn: Callable[[str], str],
        current: str,
        *,
        columns: int = 5,
        icon_size: int = 34,
        grid_max_height: int = 118,
        compact: bool = False,
        parent=None,
    ):
        super().__init__(parent)
        self._items = list(items)
        self._group = QButtonGroup(self)
        self._group.setExclusive(True)
        self._buttons: Dict[str, QToolButton] = {}

        if compact:
            icon_size = min(icon_size, 26)
            btn_w, btn_h = 36, 36
            tool_style = Qt.ToolButtonStyle.ToolButtonIconOnly
        else:
            btn_w, btn_h = 62, 54
            tool_style = Qt.ToolButtonStyle.ToolButtonTextUnderIcon

        outer = QVBoxLayout(self)
        outer.setContentsMargins(4, 4, 4, 4)
        outer.setSpacing(0)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setMaximumHeight(grid_max_height)

        inner = QWidget()
        grid = QGridLayout(inner)
        grid.setContentsMargins(2, 2, 2, 2)
        grid.setHorizontalSpacing(6 if compact else 4)
        grid.setVerticalSpacing(6 if compact else 4)

        for idx, name in enumerate(items):
            btn = QToolButton()
            btn.setCheckable(True)
            btn.setAutoRaise(True)
            btn.setToolButtonStyle(tool_style)
            pix = icon_fn(name, icon_size)
            btn.setIcon(pix if isinstance(pix, QIcon) else QIcon(pix))
            btn.setIconSize(QSize(icon_size, icon_size))
            if not compact:
                btn.setText(label_fn(name))
            btn.setToolTip(name)
            btn.setFixedSize(btn_w, btn_h)
            btn.setStyleSheet(
                "QToolButton { font-size: 9px; color: #cbd5e1; padding: 2px; border-radius: 4px; "
                "border: 1px solid transparent; }"
                "QToolButton:checked { background: #334155; border: 1px solid #f87171; }"
                "QToolButton:hover { background: #1e293b; border: 1px solid #475569; }"
            )
            btn.clicked.connect(lambda checked=False, n=name: self._on_click(n))
            self._group.addButton(btn)
            self._buttons[name] = btn
            grid.addWidget(btn, idx // columns, idx % columns)

        scroll.setWidget(inner)
        outer.addWidget(scroll)
        self.set_selected(current, emit=False)

    def _on_click(self, name: str):
        self.picked.emit(name)

    def set_selected(self, name: str, *, emit: bool = True) -> None:
        btn = self._buttons.get(name)
        if btn is not None:
            btn.blockSignals(True)
            btn.setChecked(True)
            btn.blockSignals(False)
            if emit:
                self.picked.emit(name)

    def clear_selection(self) -> None:
        self._group.setExclusive(False)
        for btn in self._buttons.values():
            btn.setChecked(False)
        self._group.setExclusive(True)


class CategorizedIconPicker(QWidget):
    """Compact tabbed picker — one tab per category, icon grid inside each."""

    valueChanged = Signal(str)

    def __init__(
        self,
        categories: Dict[str, List[str]],
        icon_fn: Callable[[str, int], Union[QPixmap, QIcon]],
        label_fn: Callable[[str], str],
        current: str,
        *,
        columns: int = 5,
        compact: bool = False,
        parent=None,
    ):
        super().__init__(parent)
        self._categories = categories
        self._pages: Dict[str, _IconGridPage] = {}
        self._value = current
        self._building = True

        grid_max_height = 84 if compact else 118
        icon_size = 26 if compact else 34

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        self._tabs = QTabWidget()
        self._tabs.setDocumentMode(True)
        tab_pad = "2px 6px" if compact else "3px 8px"
        tab_min = "28px" if compact else "36px"
        tab_fs = "9px" if compact else "10px"
        self._tabs.setStyleSheet(
            "QTabWidget::pane { border: 1px solid #334155; border-radius: 4px; top: -1px; }"
            f"QTabBar::tab {{ font-size: {tab_fs}; padding: {tab_pad}; min-width: {tab_min}; }}"
            "QTabBar::tab:selected { background: #334155; color: #f8fafc; }"
        )
        if compact:
            self._tabs.setMaximumHeight(grid_max_height + 28)

        tab_labels = {
            "Classic": "Classic",
            "Modern": "Modern",
            "Tactical": "Tac",
            "Minimal": "Min",
            "Cosmetic": "Glow",
            "Funny": "Fun",
            "Emoji": "😀",
            "Basic": "Basic",
        }

        for cat, items in categories.items():
            page = _IconGridPage(
                items,
                icon_fn,
                label_fn,
                current,
                columns=columns,
                icon_size=icon_size,
                grid_max_height=grid_max_height,
                compact=compact,
            )
            page.picked.connect(self._on_picked)
            self._pages[cat] = page
            self._tabs.addTab(page, tab_labels.get(cat, cat[:4]))

        layout.addWidget(self._tabs)
        self._building = False
        self.set_value(current, emit=False)
        self._select_tab_for_value(current)

    def _on_picked(self, name: str):
        if self._building:
            return
        self._value = name
        for cat, page in self._pages.items():
            if name in self._categories.get(cat, []):
                page.set_selected(name, emit=False)
            else:
                page.clear_selection()
        self.valueChanged.emit(name)

    def _select_tab_for_value(self, name: str) -> None:
        for i, (cat, items) in enumerate(self._categories.items()):
            if name in items:
                self._tabs.setCurrentIndex(i)
                for c, page in self._pages.items():
                    if c == cat:
                        page.set_selected(name, emit=False)
                    else:
                        page.clear_selection()
                return

    def set_value(self, name: str, *, emit: bool = True) -> None:
        self._value = name
        self._select_tab_for_value(name)
        if emit:
            self.valueChanged.emit(name)

    def value(self) -> str:
        return self._value


def motion_style_picker(current: str, parent=None, *, compact: bool = False) -> CategorizedIconPicker:
    from desktop.utils.motion_overlay_styles import MOTION_STYLE_CATEGORIES

    return CategorizedIconPicker(
        MOTION_STYLE_CATEGORIES,
        motion_style_icon_pixmap,
        motion_style_short_label,
        current,
        columns=7 if compact else 5,
        compact=compact,
        parent=parent,
    )


def motion_animation_picker(current: str, parent=None, *, compact: bool = False) -> CategorizedIconPicker:
    from desktop.utils.motion_overlay_styles import MOTION_ANIMATION_CATEGORIES

    return CategorizedIconPicker(
        MOTION_ANIMATION_CATEGORIES,
        motion_animation_icon_pixmap,
        motion_animation_short_label,
        current,
        columns=6 if compact else 5,
        compact=compact,
        parent=parent,
    )
