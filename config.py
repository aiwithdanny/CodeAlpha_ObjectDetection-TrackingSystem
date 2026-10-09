"""Application configuration (env variables se)."""

import os


class Config:
    SECRET_KEY = os.environ.get("SECRET_KEY", "dev-only-secret-key")
    DEBUG = os.environ.get("FLASK_DEBUG", "false").lower() == "true"

    # YOLO model: nano (fast) default; accuracy chahiye to yolov8s/m/l
    YOLO_MODEL = os.environ.get("YOLO_MODEL", "yolov8n.pt")

    # Tracker: "bytetrack.yaml" (modern, default) ya "botsort.yaml"
    TRACKER = os.environ.get("TRACKER", "bytetrack.yaml")

    # Confidence threshold: is se kam score wali detection ignore
    CONF_THRESHOLD = float(os.environ.get("CONF_THRESHOLD", "0.25"))

    # Upload limits (web app)
    MAX_UPLOAD_MB = int(os.environ.get("MAX_UPLOAD_MB", "50"))
    ALLOWED_EXTENSIONS = {"mp4", "avi", "mov", "mkv"}

    UPLOAD_FOLDER = os.environ.get("UPLOAD_FOLDER", "uploads")
    OUTPUT_FOLDER = os.environ.get("OUTPUT_FOLDER", "outputs")
