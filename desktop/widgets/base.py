from PySide6.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton
from PySide6.QtCore import QEvent, Qt, QPoint, QSize, QPropertyAnimation, QEasingCurve
from PySide6.QtGui import QColor, QIcon
from desktop.utils.app_icon import apply_window_icon
from desktop.utils.qt_helpers import KnoxnetStyle

class BaseDesktopWidget(QMainWindow):
    """
    Base class for all desktop widgets.
    Provides:
    - Frameless window
    - Custom title bar (Drag, Close, Minimize, Pin)
    - Resizable edges
    """
    def __init__(self, title="Widget", width=400, height=300):
        super().__init__()
        apply_window_icon(self)
        # Ensure closed widgets are actually destroyed (prevents “zombie” timers/slots accumulating).
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, True)
        self.setWindowFlags(Qt.WindowType.FramelessWindowHint)
        
        # Dragging state
        self.old_pos = None
        self.is_pinned = False
        self._drag_handles = {}
        self._drag_candidate = None
        self._drag_start_window_pos = None
        self._drag_source = None
        self._drag_active = False
        self._snap_anim = None

        # Main Container (for rounded corners/borders if needed)
        self.central_widget = QWidget()
        self.central_widget.setMouseTracking(True)
        self.central_widget.setStyleSheet(f"""
            QWidget#Central {{
                background-color: {KnoxnetStyle.BG_DARK};
                border: 1px solid {KnoxnetStyle.BORDER};
                border-radius: 0px; 
            }}
        """)
        self.central_widget.setObjectName("Central")
        self.setCentralWidget(self.central_widget)
        
        # Resizing logic
        self.resize_margin = 8
        self.resize_edge = None
        self.keep_aspect_ratio = False
        self.aspect_ratio = 16/9
        self.setMouseTracking(True)

        self.resize(width, height)
        
        self.main_layout = QVBoxLayout(self.central_widget)
        self.main_layout.setContentsMargins(0, 0, 0, 0)
        self.main_layout.setSpacing(0)

        # 1. Custom Title Bar
        self.title_bar = QWidget()
        self.title_bar.setStyleSheet(f"background-color: {KnoxnetStyle.BG_LIGHT};")
        self.title_bar.setFixedHeight(30)
        self.title_layout = QHBoxLayout(self.title_bar)
        self.title_layout.setContentsMargins(10, 0, 10, 0)

        # Title
        self.title_label = QLabel(title)
        self.title_label.setStyleSheet("font-weight: bold;")
        
        # Buttons
        self.pin_btn = QPushButton("📌")
        self.pin_btn.setFixedSize(24, 24)
        self.pin_btn.setCheckable(True)
        self.pin_btn.clicked.connect(self.toggle_pin)
        self.pin_btn.setToolTip("Pin to top")

        self.min_btn = QPushButton("_")
        self.min_btn.setFixedSize(24, 24)
        self.min_btn.clicked.connect(self.showMinimized)

        self.close_btn = QPushButton("✕")
        self.close_btn.setFixedSize(24, 24)
        self.close_btn.setStyleSheet("QPushButton:hover { background-color: #ef4444; }")
        self.close_btn.clicked.connect(self.close)

        self.title_layout.addWidget(self.title_label)
        self.title_layout.addStretch()
        self.title_layout.addWidget(self.pin_btn)
        self.title_layout.addWidget(self.min_btn)
        self.title_layout.addWidget(self.close_btn)

        self.main_layout.addWidget(self.title_bar)
        self.title_bar.hide() # Hide custom title bar by default for clean look

        # 2. Content Area (Placeholder)
        self.content_area = QWidget()
        self.content_layout = QVBoxLayout(self.content_area)
        self.content_layout.setContentsMargins(0, 0, 0, 0)
        self.content_layout.setSpacing(0)
        self.main_layout.addWidget(self.content_area)

        self.register_drag_handle(self.title_bar)
        self.register_drag_handle(self.title_label)
        self.register_drag_handle(self.central_widget)
        self.register_drag_handle(self.content_area)

    def register_drag_handle(self, widget, should_start=None):
        """Allow a child widget to drag the frameless window."""
        if widget is None:
            return
        self._drag_handles[id(widget)] = (widget, should_start)
        widget.installEventFilter(self)

    def _reset_child_drag(self):
        self._drag_candidate = None
        self._drag_start_window_pos = None
        self._drag_source = None
        self._drag_active = False

    def _can_start_child_drag(self, watched, event):
        entry = self._drag_handles.get(id(watched))
        if entry is None or self.is_pinned or event.button() != Qt.MouseButton.LeftButton:
            return False
        _, should_start = entry
        if should_start is None:
            return True
        try:
            return bool(should_start(event))
        except Exception:
            return False

    def _start_system_move(self, event) -> bool:
        """Use the window manager's native drag when supported."""
        if self.is_pinned or event.button() != Qt.MouseButton.LeftButton:
            return False
        try:
            handle = self.windowHandle()
            if handle is None:
                win = self.window()
                handle = win.windowHandle() if win is not None else None
            if handle is not None and hasattr(handle, "startSystemMove") and handle.startSystemMove():
                self._reset_child_drag()
                event.accept()
                return True
        except Exception:
            pass
        return False

    def eventFilter(self, watched, event):
        if id(watched) in self._drag_handles:
            etype = event.type()
            if etype == QEvent.Type.MouseButtonPress and self._can_start_child_drag(watched, event):
                if self._start_system_move(event):
                    return True
                self._drag_candidate = event.globalPosition().toPoint()
                self._drag_start_window_pos = self.pos()
                self._drag_source = watched
                self._drag_active = False
            elif etype == QEvent.Type.MouseMove and watched is self._drag_source and self._drag_candidate and self._drag_start_window_pos:
                if not (event.buttons() & Qt.MouseButton.LeftButton):
                    self._reset_child_drag()
                else:
                    delta = event.globalPosition().toPoint() - self._drag_candidate
                    if self._drag_active or delta.manhattanLength() >= QApplication.startDragDistance():
                        self._drag_active = True
                        self.move(self._drag_start_window_pos + delta)
                        self._show_snap_preview()
                        event.accept()
                        return True
            elif etype == QEvent.Type.MouseButtonRelease and watched is self._drag_source:
                was_dragging = self._drag_active
                self._reset_child_drag()
                if was_dragging:
                    self._try_snap()
                    event.accept()
                    return True
            elif etype in (QEvent.Type.Hide, QEvent.Type.Close):
                self._drag_handles.pop(id(watched), None)
                if watched is self._drag_source:
                    self._reset_child_drag()
        return super().eventFilter(watched, event)
        
    def resizeEvent(self, event):
        super().resizeEvent(event)

    # --- Resizing Logic (Manual for Frameless) ---
    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            if self.resize_edge:
                self.is_resizing = True
                self.resize_start_pos = event.globalPosition().toPoint()
                self.resize_start_geo = self.geometry()
            else:
                if self._start_system_move(event):
                    return
                # Drag from anywhere since title bar is hidden
                self.old_pos = event.globalPosition().toPoint()

    def mouseMoveEvent(self, event):
        # Only update cursor/edge detection if NOT currently resizing
        if not (hasattr(self, 'is_resizing') and self.is_resizing) and not self.is_pinned:
            pos = event.position()
            rect = self.rect()
            margin = self.resize_margin
            
            on_left = pos.x() <= margin
            on_right = pos.x() >= rect.width() - margin
            on_top = pos.y() <= margin
            on_bottom = pos.y() >= rect.height() - margin
            
            if on_left and on_top:
                self.setCursor(Qt.CursorShape.SizeFDiagCursor)
                self.resize_edge = "top-left"
            elif on_right and on_bottom:
                self.setCursor(Qt.CursorShape.SizeFDiagCursor)
                self.resize_edge = "bottom-right"
            elif on_right and on_top:
                self.setCursor(Qt.CursorShape.SizeBDiagCursor)
                self.resize_edge = "top-right"
            elif on_left and on_bottom:
                self.setCursor(Qt.CursorShape.SizeBDiagCursor)
                self.resize_edge = "bottom-left"
            elif on_left:
                self.setCursor(Qt.CursorShape.SizeHorCursor)
                self.resize_edge = "left"
            elif on_right:
                self.setCursor(Qt.CursorShape.SizeHorCursor)
                self.resize_edge = "right"
            elif on_top:
                self.setCursor(Qt.CursorShape.SizeVerCursor)
                self.resize_edge = "top"
            elif on_bottom:
                self.setCursor(Qt.CursorShape.SizeVerCursor)
                self.resize_edge = "bottom"
            else:
                self.setCursor(Qt.CursorShape.ArrowCursor)
                self.resize_edge = None

        if hasattr(self, 'is_resizing') and self.is_resizing:
            delta = event.globalPosition().toPoint() - self.resize_start_pos
            geo = self.resize_start_geo
            
            new_x = geo.x()
            new_y = geo.y()
            new_w = geo.width()
            new_h = geo.height()

            # 1. Calculate new dimensions based on mouse movement and edge
            if "left" in self.resize_edge:
                new_w = max(200, geo.width() - delta.x())
            elif "right" in self.resize_edge:
                new_w = max(200, geo.width() + delta.x())
            
            if "top" in self.resize_edge:
                new_h = max(150, geo.height() - delta.y())
            elif "bottom" in self.resize_edge:
                new_h = max(150, geo.height() + delta.y())

            # 2. Apply Aspect Ratio Constraint
            if self.keep_aspect_ratio:
                if "left" in self.resize_edge or "right" in self.resize_edge:
                    # Width drives height
                    new_h = int(new_w / self.aspect_ratio)
                elif "top" in self.resize_edge or "bottom" in self.resize_edge:
                    # Pure vertical drag -> Height drives width
                    new_w = int(new_h * self.aspect_ratio)

            # 3. Apply Anchors to calculate Position
            if "left" in self.resize_edge:
                new_x = (geo.x() + geo.width()) - new_w
            
            if "top" in self.resize_edge:
                new_y = (geo.y() + geo.height()) - new_h

            self.setGeometry(new_x, new_y, new_w, new_h)

        elif self.old_pos:
            delta = event.globalPosition().toPoint() - self.old_pos
            self.move(self.pos() + delta)
            self.old_pos = event.globalPosition().toPoint()
            self._show_snap_preview()

    def mouseReleaseEvent(self, event):
        was_moving = self.old_pos is not None
        was_resizing = hasattr(self, 'is_resizing') and self.is_resizing
        self.old_pos = None
        self._reset_child_drag()
        if was_resizing:
            self.is_resizing = False
            self.resize_edge = None
            self.setCursor(Qt.CursorShape.ArrowCursor)
        if was_moving or was_resizing:
            self._try_snap()

    def _show_snap_preview(self):
        """Compute where the widget would snap and show guide lines."""
        try:
            app = QApplication.instance()
            if not app or not hasattr(app, 'compute_snap_info'):
                return
            info = app.compute_snap_info(self, self.geometry())
            app.show_snap_guides(info.get("h_guides", []), info.get("v_guides", []))
        except Exception:
            pass

    def _hide_snap_preview(self):
        try:
            app = QApplication.instance()
            if app and hasattr(app, 'hide_snap_guides'):
                app.hide_snap_guides()
        except Exception:
            pass

    def _try_snap(self):
        """Hide snap guides and animate the widget into its snapped position/size."""
        self._hide_snap_preview()
        try:
            app = QApplication.instance()
            if not app or not hasattr(app, 'compute_snap_info'):
                return
            info = app.compute_snap_info(self, self.geometry())
            snapped = info.get("geo", self.geometry())
            if snapped == self.geometry():
                return
            if self._snap_anim is not None:
                self._snap_anim.stop()
            self._snap_anim = QPropertyAnimation(self, b"geometry")
            self._snap_anim.setDuration(100)
            self._snap_anim.setStartValue(self.geometry())
            self._snap_anim.setEndValue(snapped)
            self._snap_anim.setEasingCurve(QEasingCurve.Type.OutCubic)
            self._snap_anim.start()
        except Exception:
            pass

    # --- Features ---
    def toggle_pin(self):
        self.is_pinned = not self.is_pinned
        if self.is_pinned:
            self.setWindowFlag(Qt.WindowType.WindowStaysOnTopHint, True)
            self.pin_btn.setStyleSheet(f"background-color: {KnoxnetStyle.ACCENT}; border: 1px solid {KnoxnetStyle.ACCENT};")
        else:
            self.setWindowFlag(Qt.WindowType.WindowStaysOnTopHint, False)
            self.pin_btn.setStyleSheet("")
        
        self.show() # Re-show needed after changing flags

    def set_content(self, widget):
        """Replace the content area with a custom widget."""
        # clear existing
        for i in reversed(range(self.content_layout.count())): 
            self.content_layout.itemAt(i).widget().setParent(None)
        self.content_layout.addWidget(widget)
