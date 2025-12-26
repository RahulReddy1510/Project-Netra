from PyQt6.QtWidgets import QWidget, QHBoxLayout, QVBoxLayout, QFrame, QLabel, QListWidget, QListWidgetItem
from PyQt6.QtCore import Qt, QTimer, pyqtSignal, QThread
from PyQt6.QtGui import QImage, QPixmap
import cv2
import datetime
import numpy as np

# Import the Inference Loop
import sys
import os
# Add 'edge_deployment' to path so 'import polygon_zone' inside inference_loop works
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../edge_deployment")))
# Add project root as well for good measure
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from inference_loop import NetraInferenceLoop

class VideoThread(QThread):
    change_pixmap_signal = pyqtSignal(np.ndarray)
    alert_signal = pyqtSignal(str, str) # title, message

    def __init__(self):
        super().__init__()
        self._run_flag = True
        
        # Initialize Logic Engine
        # We try to use the exported model if available, else standard yolo
        model_path = 'yolov8m.pt' 
        # Check for trained weight
        trained_weight = r"..\vision_core\Netra_Vision_Core\v1_meta_enhanced\weights\best.pt"
        if os.path.exists(trained_weight):
             model_path = trained_weight

        try:
            self.netra_engine = NetraInferenceLoop(source=0, model_path=model_path)
        except Exception as e:
            print(f"Failed to init engine: {e}")
            self.netra_engine = None

    def run(self):
        if not self.netra_engine:
            return

        # Generator loop
        for frame, alert in self.netra_engine.process_stream():
            if not self._run_flag:
                break
                
            self.change_pixmap_signal.emit(frame)
            
            if alert:
                self.alert_signal.emit("INTRUSION DETECTED", alert)
            
            self.msleep(1) # Yield to QT

        self.netra_engine.release()

    def stop(self):
        self._run_flag = False
        self.wait()

class DashboardWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()
        self.start_video_feed()

    def initUI(self):
        main_layout = QHBoxLayout()
        self.setLayout(main_layout)
        main_layout.setSpacing(0)
        main_layout.setContentsMargins(0,0,0,0)

        # --- LEFT: Event Log (20%) ---
        self.left_panel = QFrame()
        self.left_panel.setObjectName("Panel")
        self.left_panel.setFixedWidth(300)
        left_layout = QVBoxLayout()
        self.left_panel.setLayout(left_layout)
        
        header_log = QLabel("EVENT LOG")
        header_log.setObjectName("Header")
        left_layout.addWidget(header_log)
        
        self.event_list = QListWidget()
        left_layout.addWidget(self.event_list)
        
        main_layout.addWidget(self.left_panel)

        # --- CENTER: Video Feed (60%) ---
        self.center_panel = QFrame()
        self.center_panel.setStyleSheet("background-color: #000; border-left: 1px solid #333; border-right: 1px solid #333;")
        center_layout = QVBoxLayout()
        self.center_panel.setLayout(center_layout)
        
        self.video_label = QLabel()
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        center_layout.addWidget(self.video_label)
        
        # Bottom heatmap/stats strip in center
        stat_strip = QHBoxLayout()
        self.lbl_zone_status = QLabel("ZONE STATUS: SECURE")
        self.lbl_zone_status.setStyleSheet("color: #32CD32; font-weight: bold; font-size: 16px;")
        stat_strip.addWidget(self.lbl_zone_status)
        center_layout.addLayout(stat_strip)
        
        main_layout.addWidget(self.center_panel, stretch=1)

        # --- RIGHT: System Health (20%) ---
        self.right_panel = QFrame()
        self.right_panel.setObjectName("Panel")
        self.right_panel.setFixedWidth(300)
        right_layout = QVBoxLayout()
        self.right_panel.setLayout(right_layout)
        
        header_sys = QLabel("SYSTEM HEALTH")
        header_sys.setObjectName("Header")
        right_layout.addWidget(header_sys)

        # Stats
        self.add_stat(right_layout, "GPU TEMP", "42°C")
        self.add_stat(right_layout, "FPS", "30.1")
        self.add_stat(right_layout, "LATENCY", "15ms")
        self.add_stat(right_layout, "ACTIVE ZONES", "3")
        
        right_layout.addStretch()
        
        main_layout.addWidget(self.right_panel)

    def add_stat(self, layout, title, value):
        container = QFrame()
        vbox = QVBoxLayout()
        container.setLayout(vbox)
        
        lbl_title = QLabel(title)
        lbl_title.setStyleSheet("color: #888; font-size: 12px;")
        lbl_val = QLabel(value)
        lbl_val.setObjectName("StatLabel")
        
        vbox.addWidget(lbl_title)
        vbox.addWidget(lbl_val)
        layout.addWidget(container)

    def start_video_feed(self):
        self.thread = VideoThread()
        self.thread.change_pixmap_signal.connect(self.update_image)
        self.thread.alert_signal.connect(self.add_alert)
        self.thread.start()

    def update_image(self, cv_img):
        """Updates the video_label with a new opencv image"""
        qt_img = self.convert_cv_qt(cv_img)
        self.video_label.setPixmap(qt_img)

    def convert_cv_qt(self, cv_img):
        """Convert from an opencv image to QPixmap"""
        rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        convert_to_Qt_format = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        p = convert_to_Qt_format.scaled(self.video_label.width(), self.video_label.height(), Qt.AspectRatioMode.KeepAspectRatio)
        return QPixmap.fromImage(p)

    def add_alert(self, title, msg):
        time_str = datetime.datetime.now().strftime("%H:%M:%S")
        item = QListWidgetItem(f"[{time_str}] {title}\n{msg}")
        self.event_list.insertItem(0, item)
        # Flash effect or sound could go here
        
    def closeEvent(self, event):
        self.thread.stop()
        event.accept()
