"""
Step 1 preview: camera + YOLO-World detection.
Run from repo root: python backend/scripts/preview.py
"""
import time
import cv2
import numpy as np
from ultralytics import YOLO

CAMERA_INDEX = 0
MODEL_PATH = "models/yolov8l-worldv2.pt"
IMGSZ = 640

CLASSES = [
    "glasses", "wallet", "pill bottle", "keys", "phone", "remote control",
    "mug", "cup", "can", "jar", "bottle", "water bottle", "book", "notebook",
    "paper", "pen", "laptop", "keyboard", "monitor", "bag", "backpack",
    "plate", "bowl", "spoon", "person",
    # more specific vocabulary for better recall
    "chip can", "jar with lid", "travel mug", "tumbler", "spice jar",
    "computer monitor", "cable",
]

TRACKED = {"glasses", "wallet", "pill bottle"}

# Maps raw class name → clean display label
DISPLAY_NAME = {
    "chip can":         "can",
    "jar with lid":     "jar",
    "travel mug":       "travel mug",
    "tumbler":          "travel mug",
    "spice jar":        "jar",
    "computer monitor": "monitor",
    "monitor":          "monitor",
}

CONF_TRACKED    = 0.25
CONF_OTHER      = 0.35
CONF_LABEL      = 0.50   # non-tracked: show text only above this
IOU_SUPPRESS    = 0.6
DETECT_INTERVAL = 2      # run inference every Nth frame; reuse boxes in between

# Distinct colors for tracked classes (BGR)
TRACKED_COLORS = {
    "glasses":     (0,   255,   0),
    "wallet":      (255, 128,   0),
    "pill bottle": (0,   128, 255),
}
GRAY = (180, 180, 180)


def iou(a, b):
    """Intersection-over-union for two (x1,y1,x2,y2) boxes."""
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    if inter == 0:
        return 0.0
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return inter / (area_a + area_b - inter)


def suppress(detections):
    """Remove lower-confidence box when IoU with a higher-confidence box > IOU_SUPPRESS."""
    # detections: list of (conf, x1, y1, x2, y2, cls_name)
    detections = sorted(detections, key=lambda d: d[0], reverse=True)
    kept = []
    for cand in detections:
        box_c = cand[1:5]
        if all(iou(box_c, k[1:5]) <= IOU_SUPPRESS for k in kept):
            kept.append(cand)
    return kept


def main():
    model = YOLO(MODEL_PATH)
    model.set_classes(CLASSES)

    device = "mps"
    try:
        model.predict(source=np.zeros((640, 640, 3), dtype=np.uint8),
                      imgsz=IMGSZ, device=device, verbose=False)
    except Exception:
        device = "cpu"

    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print(f"ERROR: Could not open camera index {CAMERA_INDEX}.")
        print("Check System Settings > Privacy & Security > Camera, or try a different index.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    fps_smooth = 0.0
    t_prev = time.perf_counter()
    frame_idx = 0
    last_detections = []

    print(f"Camera {CAMERA_INDEX} opened. Device: {device}. Press q to quit.")

    while True:
        ok, frame = cap.read()
        if not ok:
            print("Frame read failed — camera disconnected?")
            break

        if frame_idx % DETECT_INTERVAL == 0:
            # Run with the lower threshold; filter non-tracked in Python
            results = model.predict(source=frame, conf=CONF_TRACKED,
                                    imgsz=IMGSZ, device=device, verbose=False)
            detections = []
            for r in results:
                for box in r.boxes:
                    cls_name = CLASSES[int(box.cls[0])]
                    conf = float(box.conf[0])
                    if cls_name not in TRACKED and conf < CONF_OTHER:
                        continue
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    detections.append((conf, x1, y1, x2, y2, cls_name))
            last_detections = suppress(detections)
        frame_idx += 1

        detections = last_detections

        for conf, x1, y1, x2, y2, cls_name in detections:
            is_tracked = cls_name in TRACKED
            display = DISPLAY_NAME.get(cls_name, cls_name)
            if is_tracked:
                color      = TRACKED_COLORS[cls_name]
                thickness  = 3
                font_scale = 0.65
                font_thick = 2
                label      = f"{display} {conf:.2f}"
            else:
                color      = GRAY
                thickness  = 1
                font_scale = 0.45
                font_thick = 1
                label      = f"{display} {conf:.2f}" if conf >= CONF_LABEL else None

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
            if label:
                cv2.putText(frame, label, (x1, max(y1 - 6, 12)),
                            cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, font_thick)

        t_now = time.perf_counter()
        inst_fps = 1.0 / max(t_now - t_prev, 1e-6)
        t_prev = t_now
        fps_smooth = 0.9 * fps_smooth + 0.1 * inst_fps
        cv2.putText(frame, f"FPS {fps_smooth:.1f}", (10, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        cv2.imshow("Recall preview", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
