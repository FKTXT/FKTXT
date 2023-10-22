import libjevois as jevois
import re
import cv2
import time
import numpy as np
from HandTracker import HandDetector
from ultralytics import YOLO
from datetime import datetime

def checkCoords(landmarks:list, ox, oy, w, h):
    count = 0

    for i, x, y in landmarks:
        if x > ox and x < ox + h:
            if y > oy:
                count += 1
        
        if count > 3:
            return True
    return False



class Detectorv1:
    # ###################################################################################################
    ## Constructor
    def __init__(self):
        # Instantiate a JeVois Timer to measure our processing framerate:
        self.timer = jevois.Timer("processing timer", 100, jevois.LOG_INFO)
        self.CPULoad_pct = "0"
        self.CPUTemp_C = "0"
        self.pattern = re.compile('([0-9]*\.[0-9]+|[0-9]+) fps, ([0-9]*\.[0-9]+|[0-9]+)% CPU, ([0-9]*\.[0-9]+|[0-9]+)C,')
        self.frame = 0
        self.calibrating=True
        self.detected = False
        self.model = YOLO('yolov8n.pt')
        self.handdetector = HandDetector()


    # ###################################################################################################
    def processNoUSB(self, inframe):
        self.commonProcess(inframe=inframe)
    # ###################################################################################################
    ## Process function with USB output
    def process(self, inframe, outframe):
        _, outimg = self.commonProcess(inframe)
        outframe.sendCv(outimg)
   
        
    
    ## Process function with USB output
    def commonProcess(self, inframe):   
        img = inframe.getCvBGR()
        # Start measuring image processing time (NOTE: does not account for input conversion time):
            
        self.timer.start()

        h, w, _ = img.shape

        self.handdetector.findHands(img, draw=False)
        landmarks = self.handdetector.find_Landmarks(img, draw=False)

        results = self.model.predict(
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
        self.handdetector.findHands(img, draw=True)
        landmarks = self.handdetector.find_Landmarks(img, draw=True)
        cv2.putText(img, f"phones detected: {phones_detected}", (20, h - 20), cv2.FONT_HERSHEY_COMPLEX, .6, (20, 255, 0), 2)



        fps = self.timer.stop()
        outimg = img
        height = outimg.shape[0]
        width = outimg.shape[1]
        cv2.putText(outimg, fps, (3, height - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255))
        
        # Convert our output image to video output format and send to host over USB:
        results = self.pattern.match(self.timer.stop())
        if(results is not None):
            self.framerate_fps = results.group(1)
            self.CPULoad_pct = results.group(2)
            self.CPUTemp_C = results.group(3)

        
        serialstr = "{%s %d %s %s}"%(
            self.detected,
            self.frame,
            self.CPULoad_pct,
            self.CPUTemp_C
        )

        jevois.sendSerial(serialstr)

        self.frame += 1
        self.frame %= 999

        return self.detected, outimg