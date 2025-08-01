from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt

model = YOLO('traffic_light_detection_model.pt')

image_path = 'images/traffic_light_img.jpg'
image = cv2.imread(image_path)

results = model(image)

for r in results:
    annotated = r.plot()

img_height, img_width = annotated.shape[:2]
resized_annotated = cv2.resize(annotated, (480, int(img_height * 480 / img_width)))

cv2.imshow("Traffic Light Detection", resized_annotated)
cv2.waitKey(0)
cv2.destroyAllWindows()