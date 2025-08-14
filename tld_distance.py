from ultralytics import YOLO
import cv2
import numpy as np

# Load your YOLOv8 model once
model = YOLO('traffic_light_detection_model.pt')

def detect_traffic_lights(frame):
    """
    Detect traffic lights/signs in an RGB frame and return an annotated frame.
    """
    results = model(frame)
    annotated = frame.copy()
    
    for r in results:
        boxes = r.boxes.xyxy.cpu().numpy()  # [x1, y1, x2, y2]
        classes = r.boxes.cls.cpu().numpy().astype(int)
        
        for box, cls in zip(boxes, classes):
            x1, y1, x2, y2 = map(int, box)
            color = (0, 255, 0)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            class_name = model.names[cls]
            cv2.putText(annotated, class_name, (x1, y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2, cv2.LINE_AA)
    
    return annotated

def detect_objects_with_distance(frame, depth_frame):
    """
    Detect objects and annotate with distances using depth frame.
    """
    results = model(frame)
    annotated = frame.copy()
    
    for r in results:
        boxes = r.boxes.xyxy.cpu().numpy()
        classes = r.boxes.cls.cpu().numpy().astype(int)
        
        for box, cls in zip(boxes, classes):
            x1, y1, x2, y2 = map(int, box)
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)
            
            # Distance in meters from depth camera
            distance = depth_frame[cy, cx]
            
            color = (0, 255, 0)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            class_name = model.names[cls]
            label = f"{class_name}: {distance:.2f}m"
            cv2.putText(annotated, label, (x1, y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2, cv2.LINE_AA)
    
    return annotated

import time
import cv2
from pal.products.qcar import QCarRealSense
from traffic_detection_utils import detect_objects_with_distance

runTime = 30.0
max_distance = 50  # meters for depth scaling

with QCarRealSense(mode='RGB, Depth') as myCam:
    t0 = time.time()
    while time.time() - t0 < runTime:
        myCam.read_RGB()
        myCam.read_depth(dataMode='PX')
        
        # Annotate RGB frame with detections and distance info
        annotated = detect_objects_with_distance(myCam.imageBufferRGB, myCam.imageBufferDepthPX)
        
        # Show the annotated RGB frame
        cv2.imshow("Traffic & Sign Detection", annotated)
        
        # Show depth frame (normalized for display)
        cv2.imshow("Depth View", myCam.imageBufferDepthPX / max_distance)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cv2.destroyAllWindows()


#################### OR ####################

from ultralytics import YOLO
import cv2
import numpy as np

# Load your YOLOv8 model once
model = YOLO('traffic_light_detection_model.pt')

def detect_objects_with_distance(frame, depth_frame):
    """
    Detect objects in the RGB frame and annotate them with distance from depth frame.
    Only annotated objects have distance info. Depth window remains clean.
    """
    results = model(frame)
    annotated = frame.copy()
    
    for r in results:
        boxes = r.boxes.xyxy.cpu().numpy()
        classes = r.boxes.cls.cpu().numpy().astype(int)
        
        for box, cls in zip(boxes, classes):
            x1, y1, x2, y2 = map(int, box)
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)
            
            # Distance in meters at the center of the bounding box
            distance = depth_frame[cy, cx]
            
            color = (0, 255, 0)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            class_name = model.names[cls]
            label = f"{class_name}: {distance:.2f}m"
            cv2.putText(annotated, label, (x1, y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2, cv2.LINE_AA)
    
    return annotated

import time
import cv2
from pal.products.qcar import QCarRealSense
from traffic_detection_utils import detect_objects_with_distance

runTime = 30.0
max_distance = 50  # meters (for depth display)

with QCarRealSense(mode='RGB, Depth') as myCam:
    t0 = time.time()
    while time.time() - t0 < runTime:
        myCam.read_RGB()
        myCam.read_depth(dataMode='PX')
        
        # Annotate RGB frame with detection + distance
        annotated_rgb = detect_objects_with_distance(myCam.imageBufferRGB, myCam.imageBufferDepthPX)
        
        # Show annotated RGB frame
        cv2.imshow("Traffic & Sign Detection", annotated_rgb)
        
        # Show clean depth frame
        cv2.imshow("Depth View", myCam.imageBufferDepthPX / max_distance)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cv2.destroyAllWindows()
