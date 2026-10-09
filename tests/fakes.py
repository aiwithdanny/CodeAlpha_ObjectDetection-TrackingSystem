"""Fake ultralytics module taake tests bina torch ke chalein."""

import sys
import types

import numpy as np


class FakeBox:
    def __init__(self, bbox, conf, cls_id, track_id=None):
        self.xyxy = np.array([bbox], dtype=float)
        self.conf = np.array([conf], dtype=float)
        self.cls = np.array([cls_id], dtype=float)
        self.id = np.array([track_id], dtype=float) if track_id is not None else None


class FakeBoxes:
    def __init__(self, boxes):
        self._boxes = boxes
        ids = [b.id[0] for b in boxes if b.id is not None]
        self.id = np.array(ids, dtype=float) if ids else None

    def __iter__(self):
        return iter(self._boxes)


class FakeResult:
    def __init__(self, boxes):
        self.boxes = FakeBoxes(boxes)


class FakeYOLO:
    """ultralytics.YOLO ka test double."""

    names = {0: "person", 1: "bicycle", 2: "car"}
    last_track_kwargs = None

    def __init__(self, model_name="yolov8n.pt"):
        self.model_name = model_name

    def predict(self, frame, conf=0.25, verbose=False):
        return [FakeResult([
            FakeBox([10, 20, 100, 200], 0.9, 0),
            FakeBox([150, 30, 300, 250], 0.75, 2),
        ])]

    def track(self, frame, conf=0.25, tracker="bytetrack.yaml",
              persist=True, verbose=False):
        FakeYOLO.last_track_kwargs = dict(tracker=tracker, persist=persist)
        return [FakeResult([
            FakeBox([10, 20, 100, 200], 0.9, 0, track_id=1),
            FakeBox([150, 30, 300, 250], 0.75, 2, track_id=2),
        ])]


def install_fake_ultralytics():
    """sys.modules me fake ultralytics daalo (asli import se pehle call karo)."""
    mod = types.ModuleType("ultralytics")
    mod.YOLO = FakeYOLO
    sys.modules["ultralytics"] = mod
    return FakeYOLO
