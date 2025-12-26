import cv2
import time
from ultralytics import YOLO
import numpy as np
from polygon_zone import PolygonZone

# Mock Alert System
def send_alert(alert_type, details):
    print(f"🚨 ALERT [{alert_type}]: {details}")
    # Integration with Webhook/Sonic Alarm goes here

class NetraInferenceLoop:
    def __init__(self, source=0, model_path='yolov8m.pt'):
        """
        Args:
            source: 0 for webcam, or RTSP string "rtsp://..."
            model_path: Path to .pt or .engine file
        """
        print(f"Initing Netra Inference on {source}...")
        self.cap = cv2.VideoCapture(source)
        self.model = YOLO(model_path)
        
        # Define a Danger Zone (Interactive in real UI, hardcoded here)
        # Top-left, Top-right, Bottom-right, Bottom-left
        self.danger_zone = PolygonZone([(200, 200), (500, 200), (500, 400), (100, 400)], "High Voltage Area")

    def process_stream(self):
        """
        Yields (processed_frame, alert) tuples for UI consumption.
        """
        frame_count = 0
        fps_start = time.time()

        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                break
            
            # 1. Inference with TRACKING (ByteTrack)
            # persist=True keeps IDs alive across frames
            # tracker="bytetrack.yaml" is the state-of-the-art tracker in YOLOv8
            results = self.model.track(frame, verbose=False, conf=0.25, persist=True, tracker="bytetrack.yaml")
            
            # 2. Process Detections
            detections = []
            for result in results:
                # boxes with Track ID: [x1, y1, x2, y2, id, conf, cls] (sometimes id is missing if no track)
                # We need to handle both cases
                boxes = result.boxes.data.tolist()
                
                for box in boxes:
                    # Check format length to handle cases where tracker hasn't assigned ID yet
                    if len(box) == 7:
                        x1, y1, x2, y2, track_id, conf, cls = box
                    else:
                        x1, y1, x2, y2, conf, cls = box
                        track_id = -1

                    # Normalize for polygon (it expects x1,y1,x2,y2,conf,cls)
                    detections.append([x1, y1, x2, y2, conf, cls]) # Polygon doesn't care about ID yet
                    
                    # Visuals
                    color = (0, 255, 0) # Green default
                    label_text = f"#{int(track_id)} {self.model.names[int(cls)]}"

                    
                    # Logic: Violation Color Coding
                    # Assuming Class 0=Helmet, 1=Vest, 3=No-Helmet, 4=No-Vest
                    if int(cls) in [3, 4]: 
                        color = (0, 0, 255) # Red for violation
                        label_text = f"VIOLATION: {label_text}"

                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                    cv2.putText(frame, label_text, (int(x1), int(y1)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            # 3. Logic: Intrusion Detection
            alert_msg = None
            intruders = self.danger_zone.trigger(detections)
            if intruders:
                alert_msg = f"{len(intruders)} object(s) in {self.danger_zone.name}"
                self.danger_zone.draw(frame, is_alert=True)
            else:
                self.danger_zone.draw(frame, is_alert=False)

            # FPS Calculation
            frame_count += 1
            if frame_count % 10 == 0:
                fps = frame_count / (time.time() - fps_start)
                cv2.putText(frame, f"FPS: {fps:.1f}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            
            yield frame, alert_msg
    
    def release(self):
        self.cap.release()

if __name__ == "__main__":
    # Use webcam 0 for demo
    loop = NetraInferenceLoop(source=0)
    loop.run()
