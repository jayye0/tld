from ultralytics import YOLO
import cv2
import numpy as np

model = YOLO('yolov8n.pt')

video_path = 'images/traffic_vid2.mp4'
cap = cv2.VideoCapture(video_path)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)
    annotated = frame.copy()

    for r in results:
        boxes = r.boxes
        for box in boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            label = r.names[cls_id]

            if label != 'traffic light':
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            crop = frame[y1:y2, x1:x2]

            hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)

            color_ranges = {
                'red1': ((0, 100, 100), (10, 255, 255)),
                'red2': ((160, 100, 100), (179, 255, 255)),
                'yellow': ((20, 100, 100), (30, 255, 255)),
                'green': ((40, 50, 50), (90, 255, 255))
            }

            red_mask1 = cv2.inRange(hsv, np.array(color_ranges['red1'][0]), np.array(color_ranges['red1'][1]))
            red_mask2 = cv2.inRange(hsv, np.array(color_ranges['red2'][0]), np.array(color_ranges['red2'][1]))
            yellow_mask = cv2.inRange(hsv, np.array(color_ranges['yellow'][0]), np.array(color_ranges['yellow'][1]))
            green_mask = cv2.inRange(hsv, np.array(color_ranges['green'][0]), np.array(color_ranges['green'][1]))

            red_count = cv2.countNonZero(red_mask1) + cv2.countNonZero(red_mask2)
            yellow_count = cv2.countNonZero(yellow_mask)
            green_count = cv2.countNonZero(green_mask)

            max_count = max(red_count, yellow_count, green_count)

            if max_count == red_count:
                color = 'RED'
                color_rgb = (0, 0, 255)
            elif max_count == yellow_count:
                color = 'YELLOW'
                color_rgb = (0, 255, 255)
            elif max_count == green_count:
                color = 'GREEN'
                color_rgb = (0, 255, 0)
            else:
                color = 'UNKNOWN'
                color_rgb = (128, 128, 128)

            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)

            label_prefix = f"{label.upper()} {conf:.2f} - "
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.5
            thickness = 1
            label_y = max(y1 - 10, 30)

            cv2.putText(annotated, label_prefix, (x1, label_y), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
            (text_width, _), _ = cv2.getTextSize(label_prefix, font, font_scale, thickness)
            cv2.putText(annotated, color, (x1 + text_width, label_y), font, font_scale, color_rgb, thickness, cv2.LINE_AA)

    img_height, img_width = annotated.shape[:2]
    resized = cv2.resize(annotated, (640, int(img_height * 640 / img_width)))

    cv2.imshow("Traffic Light Detection (Video)", resized)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
