import cv2
import numpy as np
from HandTracker import HandDetector
import os
from ultralytics import YOLO
from datetime import datetime
# import injecter



# Load YOLO
# Load the YOLO model


model = YOLO('yolov8n.pt')

# Load image

def checkCoords(landmarks:list, ox, oy, w, h):
    count = 0

    for i, x, y in landmarks:
        if x > ox and x < ox + h:
            if y > oy:
                count += 1
        
        if count > 3:
            return True
    return False







handdetector = HandDetector()
cap = cv2.VideoCapture(0)
while cap.isOpened():
    success, img = cap.read()
    # img = cv2.imread("datasets\images\windshield1.jpg")

    if not (success or img):
        print("Failed to get image")
        break

    h, w, _ = img.shape

    handdetector.findHands(img, draw=False)
    landmarks = handdetector.find_Landmarks(img, draw=False)

    results = model.predict(
        source=img,
        conf=0.4
    )

    result = results[0]
    classes = result.names
    phones_detected = 0


    for box in result.boxes:
        x1, y1, x2, y2 = list(map(int, box.xyxy[0].tolist()))
        class_id = box.cls[0].item()

        if class_id != 67:
            continue

        phones_detected +=1
        conf = box.conf[0].item()

        obje = classes[class_id]
        
        cv2.rectangle(img, (x1, y1), (x2, y2), (0,255,0), 2)
        cv2.putText(img, f"/{obje}, {conf:.2f}", (x1, y1), cv2.FONT_HERSHEY_COMPLEX, .5, (255, 0, 255), 1)

        if landmarks:
            if checkCoords(landmarks, x1, y1, x2-x1, y2-y1):
                print("INJECT")
                currenttime = datetime.now().strftime("%m/%d/%Y, %H:%M:%S")
                cv2.putText(img, f"hands detected on phone", (20, 20), cv2.FONT_HERSHEY_SIMPLEX, .6, (0, 0, 255), 2)
                cv2.putText(img, f"{currenttime}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, .6, (0, 0, 255), 2)

                # Save the image to the directory with a specific file name (e.g., 'saved_image.jpg')
                output_directory = 'capturedimages/'
                output_file_path = output_directory + f'{currenttime}.jpg'
                # cv2.imwrite(output_file_path, img)

        else:
            print("no hands")

    # Display the output
    handdetector.findHands(img, draw=True)
    landmarks = handdetector.find_Landmarks(img, draw=True)
    cv2.putText(img, f"phones detected: {phones_detected}", (20, h - 20), cv2.FONT_HERSHEY_COMPLEX, .6, (20, 255, 0), 2)
    cv2.imshow("Phone Detection", img)
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break


cap.release()

# Closes all the frames
cv2.destroyAllWindows()