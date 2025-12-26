import cv2
import numpy as np

class PolygonZone:
    """
    Manages a single intrusion zone.
    """
    def __init__(self, polygon, name="Zone 1"):
        """
        Args:
            polygon (list of tuples): [(x1,y1), (x2,y2), ...] points defining the zone.
            name (str): Zone identifier.
        """
        self.polygon = np.array(polygon, np.int32)
        self.polygon = self.polygon.reshape((-1, 1, 2))
        self.name = name
        self.color = (0, 0, 255) # Red for intrusion

    def trigger(self, detections):
        """
        Checks if any detections are inside the polygon.
        Args:
            detections (list): List of [x1, y1, x2, y2, conf, cls_id]
        Returns:
            list: List of detections that are INSIDE the zone.
        """
        prohibited_objects = []
        for det in detections:
            x1, y1, x2, y2 = det[:4]
            # Calculate center point of the object base (better for 'standing in zone')
            center_x = int((x1 + x2) / 2)
            center_y = int(y2) # Feet level
            
            # Check point in polygon
            # measureDist=False returns +1 (inside), -1 (outside), 0 (edge)
            is_inside = cv2.pointPolygonTest(self.polygon, (center_x, center_y), False)
            
            if is_inside >= 0:
                prohibited_objects.append(det)
                
        return prohibited_objects

    def draw(self, frame, is_alert=False):
        """ Draws the zone on the frame. """
        color = self.color if not is_alert else (0, 255, 255) # Yellow on alert
        cv2.polylines(frame, [self.polygon], isClosed=True, color=color, thickness=2)
        
        # Draw label
        M = cv2.moments(self.polygon)
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            cv2.putText(frame, self.name, (cX - 20, cY), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

