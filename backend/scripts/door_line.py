"""
Calibrate the door line. Run from the repo root:  python backend/scripts/door_line.py

Shows the Mac's webcam. Click two points to draw the door line, then click once on the OUTSIDE
side. The line and the shaded outside update live.
Keys: s = save to data/door.json   r = start over   q = quit
"""
import sys

import cv2

sys.path.insert(0, "backend")
from recall.cameras import door_camera_index
from recall.door_geometry import DOOR_JSON, draw_door_overlay, load_door_config, save_door_config

W, H = 640, 360     # what the door detector sees
SCALE = 2           # window is bigger so the clicks are easier
WIN = "Door line (click 2 points, then the outside)"

clicks: list[tuple[float, float]] = []   # fractions of the frame
saved_msg = ""


def on_mouse(event, x, y, flags, param):
    global saved_msg
    if event == cv2.EVENT_LBUTTONDOWN and len(clicks) < 3:
        clicks.append((x / (W * SCALE), y / (H * SCALE)))
        saved_msg = ""


def main() -> None:
    idx = door_camera_index()
    if idx is None:
        print("No built-in Mac camera (FaceTime/MacBook) found.")
        return
    cap = cv2.VideoCapture(idx)
    if not cap.isOpened():
        print(f"Could not open camera index {idx}.")
        return
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, H)
    existing = load_door_config()
    cv2.namedWindow(WIN)
    cv2.setMouseCallback(WIN, on_mouse)
    print(__doc__)

    global saved_msg
    while True:
        ok, raw = cap.read()
        if not ok:
            continue
        frame = cv2.resize(raw, (W * SCALE, H * SCALE))
        if len(clicks) == 3:
            cfg = {"line": [list(clicks[0]), list(clicks[1])], "out_side": list(clicks[2])}
            draw_door_overlay(frame, cfg)
            hint = "s = save   r = start over"
        else:
            if not clicks:
                draw_door_overlay(frame, existing)   # what's saved now
                hint = "click the first point of the door line (grey = current saved line)"
            else:
                for c in clicks:
                    cv2.circle(frame, (int(c[0] * W * SCALE), int(c[1] * H * SCALE)), 6, (255, 255, 255), -1)
                if len(clicks) == 2:
                    p1, p2 = clicks
                    cv2.line(frame, (int(p1[0] * W * SCALE), int(p1[1] * H * SCALE)),
                             (int(p2[0] * W * SCALE), int(p2[1] * H * SCALE)), (255, 255, 255), 2)
                hint = "click the second point" if len(clicks) == 1 else "now click once on the OUTSIDE side"
        cv2.putText(frame, saved_msg or hint, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.imshow(WIN, frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord("r"):
            clicks.clear()
            saved_msg = ""
        if key == ord("s"):
            if len(clicks) == 3:
                save_door_config({"line": [list(clicks[0]), list(clicks[1])], "out_side": list(clicks[2])})
                saved_msg = f"saved to {DOOR_JSON}"
                print(saved_msg)
            else:
                saved_msg = "need 2 line points and 1 outside point first"
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
