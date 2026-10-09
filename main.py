"""CLI: real-time object detection + tracking.

Usage:
    python main.py --source 0                    # webcam (live)
    python main.py --source traffic.mp4           # video file
    python main.py --source traffic.mp4 --output out.mp4 --save
    python main.py --source 0 --model yolov8s.pt --tracker botsort.yaml

'q' dabao band karne ke liye (sirf display mode me).
"""

import argparse
import logging
import time
from collections import Counter

from config import Config
from detection.tracker import ObjectTracker
from utils.annotate import draw_stats, draw_tracks
from utils.video import VideoSource, VideoWriter

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


def parse_args():
    p = argparse.ArgumentParser(description="YOLO + ByteTrack real-time tracking")
    p.add_argument("--source", default="0",
                   help="'0' webcam ke liye, ya video file ka path")
    p.add_argument("--model", default=Config.YOLO_MODEL)
    p.add_argument("--tracker", default=Config.TRACKER,
                   choices=["bytetrack.yaml", "botsort.yaml"])
    p.add_argument("--conf", type=float, default=Config.CONF_THRESHOLD)
    p.add_argument("--output", default="output_tracked.mp4",
                   help="save mode me output file ka naam")
    p.add_argument("--save", action="store_true",
                   help="annotated video file me save karo")
    p.add_argument("--no-display", action="store_true",
                   help="window na kholo (sirf save/process)")
    p.add_argument("--classes", nargs="*", type=int, default=None,
                   help="sirf in class IDs ko track karo, e.g. --classes 0 (person)")
    return p.parse_args()


def main():
    args = parse_args()
    source = int(args.source) if args.source.isdigit() else args.source

    tracker = ObjectTracker(
        model_name=args.model, tracker_cfg=args.tracker, conf=args.conf
    )
    vsrc = VideoSource(source)
    logger.info("Source: %s (%dx%d @ %.1f fps)",
                args.source, vsrc.width, vsrc.height, vsrc.fps)

    writer = None
    if args.save:
        writer = VideoWriter(args.output, vsrc.fps, vsrc.width, vsrc.height)
        logger.info("Saving to %s", args.output)

    counts = Counter()
    seen_ids = set()
    frame_no = 0
    t0 = time.time()

    try:
        while True:
            ok, frame = vsrc.read()
            if not ok:
                break
            frame_no += 1

            tracked = tracker.track(frame)
            if args.classes is not None:
                tracked = [t for t in tracked if t["class_id"] in args.classes]

            for t in tracked:
                seen_ids.add(t["track_id"])
            counts = Counter(t["label"] for t in tracked)

            frame = draw_tracks(frame, tracked)
            fps = frame_no / max(time.time() - t0, 1e-6)
            frame = draw_stats(frame, fps, dict(counts), frame_no)

            if writer:
                writer.write(frame)
            if not args.no_display:
                import cv2
                cv2.imshow("Object Detection + Tracking (q = quit)", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    finally:
        vsrc.release()
        if writer:
            writer.release()
        if not args.no_display:
            import cv2
            cv2.destroyAllWindows()

    logger.info("Done. Frames: %d | Unique objects tracked: %d | %s",
                frame_no, len(seen_ids), dict(counts))


if __name__ == "__main__":
    main()
