import sys
import os
from PyQt6.QtWidgets import QApplication, QMainWindow, QWidget, QHBoxLayout, QVBoxLayout, QLabel, QFrame
from PyQt6.QtCore import Qt
from ui.dashboard import DashboardWidget

def load_stylesheet():
    style_path = os.path.join(os.path.dirname(__file__), "ui", "styles.qss")
    if os.path.exists(style_path):
        with open(style_path, "r") as f:
            return f.read()
    return ""

class NetraMainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Netra Command Center")
        self.resize(1600, 900)
        
        # Central Dashboard
        self.dashboard = DashboardWidget()
        self.setCentralWidget(self.dashboard)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    
    # Load Theme
    app.setStyleSheet(load_stylesheet())
    
    window = NetraMainWindow()
    window.show()
    sys.exit(app.exec())
