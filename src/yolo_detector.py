import cv2
from ultralytics import YOLO

class YoloDetector:
    def __init__(self, model_name='yolov8n.pt'):
        self.model = YOLO(model_name) #load the ai model into memory
        self.class_names = self.model.names

    def detect(self, frame, conf_threshold=0.5):
        #runs the model and returns detection data
        results = self.model(frame)
        detections = []
        for box in results[0].boxes:
            #extract the coords
            x1,y1,x2,y2 = map(int, box.xyxy[0])
            confidence = float(box.conf[0])
            class_id = int(box.cls[0])
            label = self.class_names[class_id]

            detections.append({
                "bbox": (x1,y1,x2,y2),
                "label": label,
                "class_id": class_id,
            })
        return detections