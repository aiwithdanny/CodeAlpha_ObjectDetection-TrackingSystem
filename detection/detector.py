"""YOLO detector wrapper.

YOLO (You Only Look Once): single neural network jo ek frame ko
ek hi forward pass me dekh kar saare objects ke boxes + labels +
confidence scores de deta hai. Hum pre-trained weights (COCO, 80
classes) use karte hain — training ki zaroorat nahi.
"""

import logging

logger = logging.getLogger(__name__)


class ObjectDetector:
    """Ultralytics YOLO model ka clean wrapper."""

    def __init__(self, model_name: str = "yolov8n.pt", conf: float = 0.25):
        from ultralytics import YOLO  # lazy import: tests me heavy import se bacho

        logger.info("Loading YOLO model: %s", model_name)
        self.model = YOLO(model_name)  # pehli dafa weights download honge (~6MB nano)
        self.conf = conf
        self.class_names = self.model.names  # {0: 'person', 1: 'bicycle', ...}
        logger.info("Model ready: %d classes.", len(self.class_names))

    def detect(self, frame):
        """Ek frame -> list of detections (dicts)."""
        results = self.model.predict(frame, conf=self.conf, verbose=False)
        detections = []
        for box in results[0].boxes:
            detections.append(
                {
                    "bbox": box.xyxy[0].tolist(),  # [x1, y1, x2, y2]
                    "confidence": float(box.conf[0]),
                    "class_id": int(box.cls[0]),
                    "label": self.class_names[int(box.cls[0])],
                }
            )
        return detections
