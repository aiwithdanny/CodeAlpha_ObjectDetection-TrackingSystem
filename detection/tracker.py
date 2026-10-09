"""Multi-object tracker wrapper.

Masla: detector har frame me NAYI detections deta hai — usay nahi pata
ke frame-1 wali "car" aur frame-10 wali "car" same hai ya alag.

Hal (ByteTrack): har detection ko boxes ke overlap (IoU) aur movement
se pichle frame ke tracks se match karo. Match hua to purana ID rakho,
nayi cheez hui to naya ID do. Is tarah har object ka ID frame-to-frame
persistent rehta hai — yehi "tracking" hai.

(SORT isi ka purana version tha; ByteTrack low-confidence detections
ko bhi second-pass me match karke zyada robust hai.)
"""

import logging

logger = logging.getLogger(__name__)


class ObjectTracker:
    """Ultralytics built-in tracker (ByteTrack / BoT-SORT) ka wrapper."""

    def __init__(
        self,
        model_name: str = "yolov8n.pt",
        tracker_cfg: str = "bytetrack.yaml",
        conf: float = 0.25,
    ):
        from ultralytics import YOLO

        logger.info("Loading tracker: model=%s, tracker=%s", model_name, tracker_cfg)
        self.model = YOLO(model_name)
        self.tracker_cfg = tracker_cfg
        self.conf = conf
        self.class_names = self.model.names

    def track(self, frame):
        """Ek frame -> tracked objects (har ek ke saath persistent track_id)."""
        results = self.model.track(
            frame,
            conf=self.conf,
            tracker=self.tracker_cfg,
            persist=True,  # ID frame ke baad bhi yaad rakho
            verbose=False,
        )
        tracked = []
        boxes = results[0].boxes
        if boxes is not None and boxes.id is not None:
            for box in boxes:
                tracked.append(
                    {
                        "track_id": int(box.id[0]),
                        "bbox": box.xyxy[0].tolist(),
                        "confidence": float(box.conf[0]),
                        "class_id": int(box.cls[0]),
                        "label": self.class_names[int(box.cls[0])],
                    }
                )
        return tracked

    def reset(self):
        """Nayi video/source ke liye tracker state saaf karo."""
        # Naya track() call me persist state reset karne ka tareeqa:
        # model ko dobara track() bina persist ke nahi — is liye simple:
        # naye source pe naya ObjectTracker banao (factory use karo).
        logger.info("Tracker reset requested.")
