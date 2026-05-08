from PySide6.QtGui import QColor, QPalette
from PySide6.QtCore import Qt

class KnoxnetStyle:
    """Central styling configuration for the desktop app."""
    
    # Colors (Dark Theme)
    BG_DARK = "#1a1a1a"
    BG_LIGHT = "#2d2d2d"
    ACCENT = "#a855f7"  # Purple accent
    TEXT_MAIN = "#ffffff"
    TEXT_DIM = "#a0a0a0"
    BORDER = "#3f3f3f"
    
    @staticmethod
    def apply_dark_theme(app):
        """Apply a global dark theme to the QApplication."""
        palette = QPalette()
        palette.setColor(QPalette.ColorRole.Window, QColor(KnoxnetStyle.BG_DARK))
        palette.setColor(QPalette.ColorRole.WindowText, QColor(KnoxnetStyle.TEXT_MAIN))
        palette.setColor(QPalette.ColorRole.Base, QColor(KnoxnetStyle.BG_LIGHT))
        palette.setColor(QPalette.ColorRole.AlternateBase, QColor(KnoxnetStyle.BG_DARK))
        palette.setColor(QPalette.ColorRole.ToolTipBase, QColor(KnoxnetStyle.BG_LIGHT))
        palette.setColor(QPalette.ColorRole.ToolTipText, QColor(KnoxnetStyle.TEXT_MAIN))
        palette.setColor(QPalette.ColorRole.Text, QColor(KnoxnetStyle.TEXT_MAIN))
        palette.setColor(QPalette.ColorRole.Button, QColor(KnoxnetStyle.BG_LIGHT))
        palette.setColor(QPalette.ColorRole.ButtonText, QColor(KnoxnetStyle.TEXT_MAIN))
        palette.setColor(QPalette.ColorRole.BrightText, QColor(KnoxnetStyle.ACCENT))
        palette.setColor(QPalette.ColorRole.Link, QColor(KnoxnetStyle.ACCENT))
        palette.setColor(QPalette.ColorRole.Highlight, QColor(KnoxnetStyle.ACCENT))
        palette.setColor(QPalette.ColorRole.HighlightedText, QColor(KnoxnetStyle.BG_DARK))
        
        app.setPalette(palette)
        
        # Set global stylesheet
        app.setStyleSheet(f"""
            QMainWindow, QDialog {{
                background-color: {KnoxnetStyle.BG_DARK};
                border: 1px solid {KnoxnetStyle.BORDER};
            }}
            QLabel {{
                color: {KnoxnetStyle.TEXT_MAIN};
            }}
            QPushButton {{
                background-color: {KnoxnetStyle.BG_LIGHT};
                color: {KnoxnetStyle.TEXT_MAIN};
                border: 1px solid {KnoxnetStyle.BORDER};
                padding: 5px;
                border-radius: 4px;
            }}
            QPushButton:hover {{
                background-color: {KnoxnetStyle.BORDER};
            }}
        """)

    @staticmethod
    def context_menu() -> str:
        """Returns a unified stylesheet for context menus."""
        return f"""
            QMenu {{
                background-color: {KnoxnetStyle.BG_LIGHT};
                color: {KnoxnetStyle.TEXT_MAIN};
                border: 1px solid {KnoxnetStyle.BORDER};
                padding: 4px;
            }}
            QMenu::item {{
                padding: 6px 24px 6px 32px;
                border-radius: 2px;
            }}
            QMenu::item:selected {{
                background-color: {KnoxnetStyle.ACCENT};
                color: {KnoxnetStyle.BG_DARK};
            }}
            QMenu::separator {{
                height: 1px;
                background: {KnoxnetStyle.BORDER};
                margin: 4px 8px;
            }}
            QMenu::indicator {{
                width: 13px;
                height: 13px;
                margin-left: 10px;
            }}
        """