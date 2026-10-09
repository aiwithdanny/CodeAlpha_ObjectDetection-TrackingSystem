"""Video input/output helpers (OpenCV)."""

import cv2


class VideoSource:
    """Webcam (int) ya video file (str) dono ko ek interface me lapeto."""

    def __init__(self, source):
        self.cap = cv2.VideoCapture(source)
        if not self.cap.isOpened():
            raise ValueError(f"Video source nahi khul saka: {source}")
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.total_frames = total if total > 0 else None  # webcam pe unknown

    def read(self):
        """Agla frame do; khatam ho to (False, None)."""
        ok, frame = self.cap.read()
        return ok, frame

    def release(self):
        self.cap.release()


class VideoWriter:
    """Annotated frames ko output video file me likho."""

    def __init__(self, path: str, fps: float, width: int, height: int):
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self.out = cv2.VideoWriter(path, fourcc, fps, (width, height))
        if not self.out.isOpened():
            raise ValueError(f"Output video nahi ban saki: {path}")

    def write(self, frame):
        self.out.write(frame)

    def release(self):
        self.out.release()
