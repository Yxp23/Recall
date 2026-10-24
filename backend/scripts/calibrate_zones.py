"""
Zone calibration tool.
Run from repo root: python backend/scripts/calibrate_zones.py

Controls:
  Left-click   add point to current polygon
  u            undo last point
  Enter        finish current polygon → type name in terminal
  s            save all zones to data/zones.json
  q            quit without saving
"""
import json
import os
import sys
import cv2
import numpy as np

CAMERA_INDEX = 0
ZONES_PATH   = "data/zones.json"
WIDTH, HEIGHT = 1280, 720

# Palette for saved zones (BGR)
ZONE_COLORS = [
    (255, 100, 100), (100, 255, 100), (100, 100, 255),
    (255, 255, 100), (255, 100, 255), (100, 255, 255),
]


def load_zones() -> list[dict]:
    if os.path.exists(ZONES_PATH):
        with open(ZONES_PATH) as f:
            return json.load(f).get("zones", [])
    return []


def save_zones(zones: list[dict]) -> None:
    os.makedirs(os.path.dirname(ZONES_PATH), exist_ok=True)
    with open(ZONES_PATH, "w") as f:
        json.dump({"zones": zones}, f, indent=2)
    print(f"Saved {len(zones)} zone(s) to {ZONES_PATH}")


def draw_saved_zones(canvas: np.ndarray, zones: list[dict]) -> None:
    overlay = canvas.copy()
    for i, zone in enumerate(zones):
        pts = np.array(zone["points"], dtype=np.int32)
        color = ZONE_COLORS[i % len(ZONE_COLORS)]
        cv2.fillPoly(overlay, [pts], color)
        cv2.polylines(canvas, [pts], isClosed=True, color=color, thickness=2)
        cx = int(pts[:, 0].mean())
        cy = int(pts[:, 1].mean())
        cv2.putText(canvas, zone["name"], (cx, cy),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    cv2.addWeighted(overlay, 0.25, canvas, 0.75, 0, canvas)


def draw_current_polygon(canvas: np.ndarray, pts: list[tuple[int, int]],
                         cursor: tuple[int, int] | None) -> None:
    color = (0, 220, 255)
    for p in pts:
        cv2.circle(canvas, p, 5, color, -1)
    for a, b in zip(pts, pts[1:]):
        cv2.line(canvas, a, b, color, 2)
    if pts and cursor:
        cv2.line(canvas, pts[-1], cursor, color, 1)
    if len(pts) >= 3 and cursor:
        cv2.line(canvas, pts[0], cursor, (100, 100, 200), 1)


def mouse_cb(event, x, y, flags, param):
    state = param
    state["cursor"] = (x, y)
    if event == cv2.EVENT_LBUTTONDOWN:
        state["current"].append((x, y))


def main():
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print(f"ERROR: Could not open camera index {CAMERA_INDEX}.")
        return
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)

    zones = load_zones()
    print(f"Loaded {len(zones)} existing zone(s).")
    print("Left-click to add points. Enter=finish polygon, u=undo, s=save, q=quit.")

    state = {"current": [], "cursor": None}
    cv2.namedWindow("Calibrate zones")
    cv2.setMouseCallback("Calibrate zones", mouse_cb, state)

    while True:
        ok, frame = cap.read()
        if not ok:
            print("Frame read failed.")
            break

        draw_saved_zones(frame, zones)
        draw_current_polygon(frame, state["current"], state["cursor"])

        hint = f"Points: {len(state['current'])}  |  Enter=finish  u=undo  s=save  q=quit"
        cv2.putText(frame, hint, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (255, 255, 255), 2)

        cv2.imshow("Calibrate zones", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            print("Quit without saving.")
            break

        elif key == ord("u"):
            if state["current"]:
                state["current"].pop()
                print(f"Undone. {len(state['current'])} point(s) remain.")
            else:
                print("Nothing to undo.")

        elif key == 13:  # Enter
            pts = state["current"]
            if len(pts) < 3:
                print("Need at least 3 points to finish a zone.")
                continue
            cap.release()
            cv2.destroyAllWindows()
            name = input("Zone name: ").strip()
            if not name:
                name = f"zone{len(zones) + 1}"
            zones.append({"name": name, "points": list(pts)})
            print(f"Zone '{name}' added ({len(pts)} points). "
                  f"Press Enter for another, s to save, q to quit.")
            state["current"] = []
            # Reopen camera
            cap = cv2.VideoCapture(CAMERA_INDEX)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
            cv2.namedWindow("Calibrate zones")
            cv2.setMouseCallback("Calibrate zones", mouse_cb, state)

        elif key == ord("s"):
            if not zones:
                print("No zones to save.")
            else:
                save_zones(zones)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
