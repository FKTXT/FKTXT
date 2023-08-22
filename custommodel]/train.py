from ultralytics import YOLO
 
# Load the model.
model = YOLO('yolov8n.pt')
 
# Training.
model.train(
   data='datasets/images/data.yaml',
   imgsz=640,
   epochs=10,
   batch=8,
   name='hpd'
)