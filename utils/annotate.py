"""Frame annotation: boxes, labels, track IDs, stats overlay."""

import cv2

# Har class_id ke liye consistent color (hash se)
def _color_for(class_id: int):
    import hashlib

    h = int(hashlib.md5(str(class_id).encode()).hexdigest(), 16)
    return (h % 200 + 30, (h >> 8) % 200 + 30, (h >> 16) % 200 + 30)


def draw_tracks(frame, tracked):
    """Tracked objects: box + 'label #ID (conf)' likho."""
    for t in tracked:
        x1, y1, x2, y2 = map(int, t["bbox"])
        color = _color_for(t["class_id"])
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        text = f"{t['label']} #{t['track_id']} ({t['confidence']:.2f})"
        (w, h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        cv2.rectangle(frame, (x1, y1 - h - 8), (x1 + w, y1), color, -1)
        cv2.putText(
            frame, text, (x1, y1 - 6),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2,
        )
    return frame


def draw_stats(frame, fps: float, counts: dict, frame_no: int):
    """Upar-left corner me FPS + per-class counts."""
    lines = [f"FPS: {fps:.1f} | frame: {frame_no}"]
    for label, n in sorted(counts.items()):
        lines.append(f"{label}: {n}")
    y = 28
    for line in lines:
        cv2.putText(frame, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.65, (0, 0, 0), 4)          # black outline
        cv2.putText(frame, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.65, (255, 255, 255), 2)    # white text
        y += 26
    return frame
