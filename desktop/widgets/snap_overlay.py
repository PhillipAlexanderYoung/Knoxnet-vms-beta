"""Transparent full-screen overlay that draws snap alignment guide lines."""

from PySide6.QtWidgets import QWidget
from PySide6.QtCore import Qt
from PySide6.QtGui import QPainter, QPen, QColor

# Subtle cyan guide line -- visible against the dark theme without being distracting.
_GUIDE_COLOR = QColor(100, 200, 255, 150)
_GUIDE_WIDTH = 1


class SnapGuideOverlay(QWidget):

    def __init__(self):
        super().__init__(None)
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
            | Qt.WindowType.BypassWindowManagerHint
            | Qt.WindowType.Tool
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating)
        self._h_lines: list[int] = []
        self._v_lines: list[int] = []
        self._screen_geo = None

    def set_guides(self, h_lines: list, v_lines: list, screen_geo):
        """Update visible guide lines.

        *h_lines*: x-coordinates where vertical guide lines should appear.
        *v_lines*: y-coordinates where horizontal guide lines should appear.
        """
        changed = self._h_lines != h_lines or self._v_lines != v_lines
        self._h_lines = list(h_lines)
        self._v_lines = list(v_lines)
        self._screen_geo = screen_geo

        if h_lines or v_lines:
            if screen_geo is not None:
                if not self.isVisible():
                    self.setGeometry(screen_geo)
                    self.show()
                    self.raise_()
                elif self.geometry() != screen_geo:
                    self.setGeometry(screen_geo)
            if changed:
                self.update()
        else:
            if self.isVisible():
                self.hide()

    def paintEvent(self, event):
        if not self._h_lines and not self._v_lines:
            return
        painter = QPainter(self)
        pen = QPen(_GUIDE_COLOR, _GUIDE_WIDTH, Qt.PenStyle.SolidLine)
        painter.setPen(pen)
        ox = self._screen_geo.x() if self._screen_geo else 0
        oy = self._screen_geo.y() if self._screen_geo else 0
        for x in self._h_lines:
            painter.drawLine(x - ox, 0, x - ox, self.height())
        for y in self._v_lines:
            painter.drawLine(0, y - oy, self.width(), y - oy)
        painter.end()
