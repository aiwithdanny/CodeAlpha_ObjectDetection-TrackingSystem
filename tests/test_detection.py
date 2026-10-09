"""Detector + tracker unit tests (mocked YOLO)."""

import pytest

from tests.fakes import install_fake_ultralytics

install_fake_ultralytics()  # asli ultralytics import hone se PEHLE

from detection.detector import ObjectDetector
from detection.tracker import ObjectTracker


def test_detector_returns_parsed_detections():
    det = ObjectDetector("yolov8n.pt", conf=0.25)
    out = det.detect(object())  # fake ko frame ki parwah nahi
    assert len(out) == 2
    assert out[0]["label"] == "person"
    assert out[0]["bbox"] == [10.0, 20.0, 100.0, 200.0]
    assert out[0]["confidence"] == pytest.approx(0.9)
    assert out[1]["label"] == "car"


def test_tracker_returns_persistent_ids():
    tr = ObjectTracker("yolov8n.pt", "bytetrack.yaml", conf=0.25)
    out = tr.track(object())
    assert len(out) == 2
    assert out[0]["track_id"] == 1
    assert out[1]["track_id"] == 2
    assert out[0]["label"] == "person"


def test_tracker_uses_persist_and_config():
    from tests.fakes import FakeYOLO

    tr = ObjectTracker("yolov8n.pt", "botsort.yaml", conf=0.5)
    tr.track(object())
    assert FakeYOLO.last_track_kwargs["persist"] is True
    assert FakeYOLO.last_track_kwargs["tracker"] == "botsort.yaml"


def test_tracker_empty_frame():
    from tests.fakes import FakeResult, FakeYOLO, install_fake_ultralytics  # noqa

    class EmptyYOLO(FakeYOLO):
        def track(self, *a, **k):
            return [FakeResult([])]

    import detection.tracker as tmod

    orig = tmod.__dict__.get("YOLO", None)
    # seedha instance ka model replace karo
    tr = ObjectTracker.__new__(ObjectTracker)
    tr.model = EmptyYOLO()
    tr.tracker_cfg = "bytetrack.yaml"
    tr.conf = 0.25
    tr.class_names = EmptyYOLO.names
    assert tr.track(object()) == []
