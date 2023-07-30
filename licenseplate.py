import cv2
import pytesseract
from ultralytics import YOLO
pytesseract.pytesseract.tesseract_cmd = "C:/Program Files/Tesseract-OCR/tesseract.exe"
model = YOLO('best.pt')


def detect_license_plate(image):

    results = model.predict(
        source=img,
        conf=0.25
    )

    result = results[0]
    classes = result.names
    # Initialize an empty list to store the license plate candidates
    license_plate_candidates = []

    for box in result.boxes:
        x1, y1, x2, y2 = list(map(int, box.xyxy[0].tolist()))
        class_id = box.cls[0].item()
        conf = box.conf[0].item()

        obje = classes[class_id]
        
        cv2.rectangle(img, (x1, y1), (x2, y2), (255,0,0), 2)
        cv2.putText(img, f"/{obje}, {conf:.2f}", (x1, y1), cv2.FONT_HERSHEY_COMPLEX, .5, (0, 255, 255), 1)
        roi = image[y1:y2, x1:x2]

        # Append the ROI to the list of candidates
        license_plate_candidates.append(roi)




    # Perform OCR on each candidate and extract the characters
    license_plate_characters = []
    for candidate in license_plate_candidates:
        # Convert the candidate to grayscale
        gray_candidate = cv2.cvtColor(candidate, cv2.COLOR_BGR2GRAY)

        # Apply thresholding to make the characters more prominent
        _, thresholded = cv2.threshold(gray_candidate, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Perform OCR using Tesseract
        text = pytesseract.image_to_string(thresholded, config='--psm 6')

        # Remove any non-alphanumeric characters from the OCR result
        cleaned_text = ''.join(char for char in text if char.isalnum())

        license_plate_characters.append(cleaned_text)

    return license_plate_characters





cap = cv2.VideoCapture(0)

while 1:
    success, img = cap.read()
    #img = cv2.imread("images\Car-Number-Plate.jpg")
    result = detect_license_plate(img)
    print("Detected license plate characters:", result)
    cv2.imshow("frame", img)
    cv2.waitKey(1)


