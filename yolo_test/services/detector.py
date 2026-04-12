from ultralytics import YOLO
from PIL import Image
import cv2
import base64
import io


class MoleDetector:
    def __init__(self, model_path: str = "best.pt"):
        self.model = YOLO(model_path)

    def predict(self, image_bytes: bytes, conf: float = 0.45, iou: float = 0.45, agnostic_nms: bool = True) -> dict:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        results = self.model.predict(
            source=image, 
            conf=conf, 
            iou=iou,
            agnostic_nms=agnostic_nms
        )[0]

        annotated = results.plot()
        _, buffer = cv2.imencode('.jpg', annotated)
        img_base64 = base64.b64encode(buffer).decode('utf-8')

        detections = []
        for box in results.boxes:
            detections.append({
                "bbox": box.xyxy[0].tolist(),
                "confidence": float(box.conf[0]),
                "class": int(box.cls[0]),
                "class_name": self.model.names[int(box.cls[0])]
            })

        return {
            "image": img_base64,
            "detections": detections
        }