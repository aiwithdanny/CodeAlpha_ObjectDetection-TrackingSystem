"""Video + annotation unit tests (sirf OpenCV, koi model nahi)."""

import os

import cv2
import numpy as np
import pytest

from utils.annotate import draw_stats, draw_tracks
from utils.video import VideoSource, VideoWriter


@pytest.fixture
def sample_video(tmp_path):
    """10 frames ki chhoti test video banao."""
    path = str(tmp_path / "in.mp4")
    w = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), 10, (160, 120))
    for i in range(10):
        frame = np.zeros((120, 160, 3), dtype=np.uint8)
        cv2.rectangle(frame, (10 + i * 5, 10), (60 + i * 5, 60), (255, 255, 255), -1)
        w.write(frame)
    w.release()
    return path


def test_video_source_reads_frames(sample_video):
    src = VideoSource(sample_video)
    assert src.width == 160 and src.height == 120
    n = 0
    while True:
        ok, frame = src.read()
        if not ok:
            break
        n += 1
        assert frame.shape == (120, 160, 3)
    src.release()
    assert n == 10


def test_video_source_bad_path():
    with pytest.raises(ValueError):
        VideoSource("/nahi/hai/video.mp4")


def test_video_writer_roundtrip(tmp_path, sample_video):
    out = str(tmp_path / "out.mp4")
    src = VideoSource(sample_video)
    writer = VideoWriter(out, 10, 160, 120)
    while True:
        ok, frame = src.read()
        if not ok:
            break
        writer.write(frame)
    src.release()
    writer.release()
    assert os.path.getsize(out) > 0


def test_draw_tracks_keeps_shape():
    frame = np.zeros((120, 160, 3), dtype=np.uint8)
    tracked = [{"track_id": 7, "bbox": [10, 10, 60, 60],
                "confidence": 0.88, "class_id": 0, "label": "person"}]
    out = draw_tracks(frame.copy(), tracked)
    assert out.shape == frame.shape
    # box draw hua -> kuch pixels change hone chahiye
    assert (out != frame).any()


def test_draw_tracks_empty():
    frame = np.zeros((120, 160, 3), dtype=np.uint8)
    out = draw_tracks(frame.copy(), [])
    assert (out == frame).all()


def test_draw_stats():
    frame = np.zeros((120, 160, 3), dtype=np.uint8)
    out = draw_stats(frame.copy(), 29.5, {"person": 2}, 42)
    assert out.shape == frame.shape
    assert (out != frame).any()
