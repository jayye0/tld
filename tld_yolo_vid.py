from ultralytics import YOLO
import cv2

model = YOLO('traffic_light_detection_model.pt')

video_path = 'videos/traffic_vid1.mp4'
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)

    for r in results:
        annotated = r.plot()

    h, w = annotated.shape[:2]
    new_width = 480
    new_height = int(h * new_width / w)
    resized_annotated = cv2.resize(annotated, (new_width, new_height))

    cv2.imshow("Traffic Light Detection", resized_annotated)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
cap.release()
cv2.destroyAllWindows()