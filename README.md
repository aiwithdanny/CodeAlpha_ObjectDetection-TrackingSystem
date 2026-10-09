# 🎯 CodeAlpha Task 3: Object Detection and Tracking

Real-time object detection (**YOLOv8**) + multi-object tracking (**ByteTrack**) with bounding boxes and persistent track IDs — via CLI (webcam/video file) and a Flask web app (upload → annotated video + stats).

## ✨ Features

- **YOLOv8 detection** — pre-trained COCO weights (80 classes), configurable model/confidence
- **ByteTrack tracking** — har object ko persistent ID (ID:1, ID:2...) frame-to-frame; `botsort.yaml` bhi supported
- **CLI** (`main.py`) — webcam live ya video file; `--save` se annotated video export; `--classes 0` se sirf person track karo
- **Web app** (`app.py`) — video upload → background processing → progress bar → result page (video + stats: frames, unique objects, per-class counts)
- Annotated output: boxes + `label #ID (confidence)`, FPS/stats overlay
- Input validation, `/health` endpoint, env-based config
- **Unit tests** (pytest, YOLO mocked — bina GPU/torch ke chalte hain)

## 🧠 How It Works

1. **Detect** (`detection/detector.py`): YOLO har frame ko ek forward pass me dekh kar boxes + labels + confidence deta hai
2. **Track** (`detection/tracker.py`): ByteTrack detections ko pichle frames ke tracks se match karta hai (IoU + motion) — match hua to purana ID, nayi cheez to naya ID
3. **Annotate** (`utils/annotate.py`): boxes, labels, IDs aur stats frame pe draw karo
4. **Output**: live window (CLI) ya processed video file + stats (web)

## 🛠 Tech Stack

Python 3 · Ultralytics YOLOv8 · ByteTrack · OpenCV · Flask · pytest

## 🚀 Setup

```bash
pip install -r requirements.txt   # pehli dafa torch download hoga (~200MB)
```

### CLI — webcam (live)

```bash
python main.py --source 0
```

### CLI — video file

```bash
python main.py --source traffic.mp4 --save --output out.mp4
python main.py --source 0 --classes 0        # sirf persons
python main.py --source 0 --tracker botsort.yaml --model yolov8s.pt
```

`q` dabao band karne ke liye.

### Web app

```bash
python app.py
```

Open http://127.0.0.1:5002 → video upload karo → progress dekho → annotated video + stats download karo.

### Environment variables (optional)

| Variable | Default | Purpose |
|---|---|---|
| `YOLO_MODEL` | `yolov8n.pt` | `yolov8s.pt`/`m`/`l` = zyada accurate, slow |
| `TRACKER` | `bytetrack.yaml` | ya `botsort.yaml` |
| `CONF_THRESHOLD` | `0.25` | detection confidence cutoff |
| `MAX_UPLOAD_MB` | `50` | web upload limit |

## 🧪 Tests

```bash
pytest tests/ -v
```

## 📁 Project Structure

```
├── main.py                 # CLI: webcam / video file real-time tracking
├── app.py                  # Flask web app (upload -> background job -> result)
├── config.py
├── detection/
│   ├── detector.py         # YOLO wrapper
│   └── tracker.py          # ByteTrack wrapper (persistent IDs)
├── utils/
│   ├── video.py            # VideoSource / VideoWriter
│   └── annotate.py         # boxes, labels, IDs, stats overlay
├── templates/ static/
└── tests/                  # test_detection/video/api.py (+fakes.py)
```

## ⚠️ Notes

- Pehli run pe YOLO weights download honge (~6MB nano).
- Webcamiran sirf local machine pe kaam karega (server pe koi camera nahi hota) — server demo ke liye web upload use karo.
- Heavy video + nano model bhi CPU pe ~10-20 FPS deta hai; GPU ho to aur fast.

## 🔮 Future Improvements

- Live webcam streaming (WebRTC/MJPEG) · DeepSORT with appearance embeddings ·
  line-crossing counter · Docker image
