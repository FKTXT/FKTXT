from ultralytics import YOLO
import cv2
from PIL import Image

# Load the YOLO model
model = YOLO('yolov8n.pt')

img = cv2.imread("dog.jpeg")

# Perform prediction on the input image
results = model.predict(
   source='https://media.roboflow.com/notebooks/examples/dog.jpeg',
   conf=0.25
)

result = results[0]
classes = result.names

for box in result.boxes:
   x1, y1, x2, y2 = list(map(int, box.xyxy[0].tolist()))
   class_id = box.cls[0].item()
   conf = box.conf[0].item()

   obje = classes[class_id]
   
   cv2.rectangle(img, (x1, y1), (x2, y2), (255,0,0), 2)
   cv2.putText(img, f"/{obje}, {conf:.2f}", (x1, y1), cv2.FONT_HERSHEY_COMPLEX, .5, (0, 255, 255), 1)

cv2.imshow("img", img)
cv2.waitKey(0)
cv2.destroyAllWindows()