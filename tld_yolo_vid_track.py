from ultralytics import YOLO
import cv2

# Load the  model
model = YOLO('traffic_light_detection_model.pt')

# Load the video
video_path = 'videos/traffic_vid1.mp4'

results = model.track(source=video_path, persist=True, tracker="bytetrack.yaml", stream=True)

for r in results:
    annotated = r.plot()

    h, w = annotated.shape[:2]
    new_width = 480
    new_height = int(h * new_width / w)
    resized_annotated = cv2.resize(annotated, (new_width, new_height))

    cv2.imshow("Traffic Light Detection (Tracked)", resized_annotated)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
