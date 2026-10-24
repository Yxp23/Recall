"""
Step 1 preview: camera + YOLO-World detection.
Run from repo root: python backend/scripts/preview.py
"""
import time
import cv2
from ultralytics import YOLO

CAMERA_INDEX = 0
MODEL_PATH = "models/yolov8s-worldv2.pt"
CLASSES = ["glasses", "wallet", "pill bottle"]
CONF = 0.25
IMGSZ = 640
COLORS = [(0, 255, 0), (255, 128, 0), (0, 128, 255)]

def main():
    model = YOLO(MODEL_PATH)
    model.set_classes(CLASSES)

    # Try MPS first, fall back to CPU
    device = "mps"
    try:
        model.predict(source=None, imgsz=IMGSZ, device=device, verbose=False)
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

    print(f"Camera {CAMERA_INDEX} opened. Device: {device}. Press q to quit.")

    while True:
        ok, frame = cap.read()
        if not ok:
            print("Frame read failed — camera disconnected?")
            break

        results = model.predict(source=frame, conf=CONF, imgsz=IMGSZ, device=device, verbose=False)

        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                color = COLORS[cls_id % len(COLORS)]
                label = f"{CLASSES[cls_id]} {conf:.2f}"
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, label, (x1, y1 - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

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
