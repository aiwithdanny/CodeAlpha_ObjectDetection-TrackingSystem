"""Flask web app: video upload karo -> YOLO+ByteTrack processing -> annotated video + stats.

Professional pattern: upload ke baad processing background thread me hoti hai,
browser /api/job/<id> ko poll karke progress dikhata hai (request timeout nahi hota).
"""

import logging
import os
import threading
import time
import uuid
from collections import Counter

from flask import Flask, jsonify, render_template, request, send_from_directory
from werkzeug.utils import secure_filename

from config import Config
from detection.tracker import ObjectTracker
from utils.annotate import draw_stats, draw_tracks
from utils.video import VideoSource, VideoWriter

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)

jobs = {}  # job_id -> {status, progress, ...} (demo scale ke liye memory me)


def create_app(config_class=Config):
    app = Flask(__name__)
    app.config.from_object(config_class)
    app.config["MAX_CONTENT_LENGTH"] = app.config["MAX_UPLOAD_MB"] * 1024 * 1024

    os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)
    os.makedirs(app.config["OUTPUT_FOLDER"], exist_ok=True)

    # Tracker ek dafa load (model weights memory me reuse)
    app.tracker = ObjectTracker(
        model_name=app.config["YOLO_MODEL"],
        tracker_cfg=app.config["TRACKER"],
        conf=app.config["CONF_THRESHOLD"],
    )

    def process_job(job_id, in_path, out_path):
        """Background worker: video process karo, progress update karo."""
        job = jobs[job_id]
        try:
            job["status"] = "processing"
            vsrc = VideoSource(in_path)
            writer = VideoWriter(out_path, vsrc.fps, vsrc.width, vsrc.height)
            total = vsrc.total_frames or 1

            counts = Counter()
            seen_ids = set()
            frame_no = 0
            t0 = time.time()
            # Har naye video pe tracker state fresh (purani IDs na aayein)
            tracker = ObjectTracker(
                model_name=app.config["YOLO_MODEL"],
                tracker_cfg=app.config["TRACKER"],
                conf=app.config["CONF_THRESHOLD"],
            )
            while True:
                ok, frame = vsrc.read()
                if not ok:
                    break
                frame_no += 1
                tracked = tracker.track(frame)
                for t in tracked:
                    seen_ids.add(t["track_id"])
                counts = Counter(t["label"] for t in tracked)
                frame = draw_tracks(frame, tracked)
                fps = frame_no / max(time.time() - t0, 1e-6)
                frame = draw_stats(frame, fps, dict(counts), frame_no)
                writer.write(frame)
                job["progress"] = round(100 * frame_no / total, 1)

            vsrc.release()
            writer.release()
            job.update(
                status="done",
                progress=100.0,
                frames=frame_no,
                unique_objects=len(seen_ids),
                counts=dict(counts),
                output_file=os.path.basename(out_path),
                elapsed=round(time.time() - t0, 1),
            )
            logger.info("Job %s done: %d frames, %d objects",
                        job_id, frame_no, len(seen_ids))
        except Exception as exc:  # noqa: BLE001
            logger.exception("Job %s failed", job_id)
            job.update(status="error", error=str(exc))

    @app.route("/")
    def index():
        return render_template("index.html",
                               max_mb=app.config["MAX_UPLOAD_MB"])

    @app.route("/api/upload", methods=["POST"])
    def upload():
        if "video" not in request.files:
            return jsonify({"error": "Koi file nahi bheji."}), 400
        f = request.files["video"]
        ext = f.filename.rsplit(".", 1)[-1].lower() if "." in f.filename else ""
        if ext not in app.config["ALLOWED_EXTENSIONS"]:
            return jsonify({"error": f"Format support nahi: {ext or 'unknown'} "
                                     f"(mp4/avi/mov/mkv)"}), 400

        job_id = uuid.uuid4().hex[:12]
        in_path = os.path.join(app.config["UPLOAD_FOLDER"],
                               f"{job_id}_{secure_filename(f.filename)}")
        out_path = os.path.join(app.config["OUTPUT_FOLDER"], f"{job_id}_tracked.mp4")
        f.save(in_path)

        jobs[job_id] = {"status": "queued", "progress": 0.0}
        threading.Thread(target=process_job,
                         args=(job_id, in_path, out_path),
                         daemon=True).start()
        return jsonify({"job_id": job_id})

    @app.route("/api/job/<job_id>")
    def job_status(job_id):
        job = jobs.get(job_id)
        if not job:
            return jsonify({"error": "Job nahi mila."}), 404
        return jsonify(job)

    @app.route("/result/<job_id>")
    def result(job_id):
        job = jobs.get(job_id)
        if not job or job.get("status") != "done":
            return "Result abhi ready nahi.", 404
        return render_template("result.html", job=job, job_id=job_id)

    @app.route("/outputs/<path:filename>")
    def outputs(filename):
        return send_from_directory(app.config["OUTPUT_FOLDER"], filename)

    @app.route("/health")
    def health():
        return {"status": "ok"}

    return app


app = create_app()

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5002))
    app.run(host="0.0.0.0", port=port, debug=app.config["DEBUG"])
