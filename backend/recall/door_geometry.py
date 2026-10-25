"""Door line geometry + drawing (light module: numpy / cv2 only, shared with scripts/door_line.py)."""
from __future__ import annotations

import json
import os
from typing import Optional

import cv2
import numpy as np

DOOR_JSON = "data/door.json"
# line + a point on the OUTSIDE, all as fractions of the frame. Default: vertical line, outside = right half.
DEFAULT_DOOR = {"line": [[0.5, 0.0], [0.5, 1.0]], "out_side": [0.75, 0.5]}

Point = tuple[float, float]


def load_door_config(path: str = DOOR_JSON) -> dict:
    try:
        with open(path) as f:
            d = json.load(f)
        (ax, ay), (bx, by) = d["line"]
        ox, oy = d["out_side"]
        return {"line": [[float(ax), float(ay)], [float(bx), float(by)]], "out_side": [float(ox), float(oy)]}
    except (OSError, ValueError, KeyError, TypeError):
        return json.loads(json.dumps(DEFAULT_DOOR))


def save_door_config(cfg: dict, path: str = DOOR_JSON) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(cfg, f, indent=2)


def to_px(cfg: dict, w: int, h: int) -> tuple[Point, Point, Point]:
    (ax, ay), (bx, by) = cfg["line"]
    ox, oy = cfg["out_side"]
    return (ax * w, ay * h), (bx * w, by * h), (ox * w, oy * h)


def cross(a: Point, b: Point, p: Point) -> float:
    return (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0])


def is_outside(a: Point, b: Point, out: Point, p: Point) -> bool:
    """True if p is on the same side of the line as the outside point."""
    return cross(a, b, p) * cross(a, b, out) > 0


def _clip_outside(poly: list[Point], a: Point, b: Point, out: Point) -> list[Point]:
    """Sutherland–Hodgman: keep the part of `poly` on the outside of the line."""
    sgn = 1.0 if cross(a, b, out) > 0 else -1.0
    d = lambda p: sgn * cross(a, b, p)  # noqa: E731
    res: list[Point] = []
    for i, p in enumerate(poly):
        q = poly[(i + 1) % len(poly)]
        dp, dq = d(p), d(q)
        if dp >= 0:
            res.append(p)
        if (dp >= 0) != (dq >= 0):
            t = dp / (dp - dq)
            res.append((p[0] + t * (q[0] - p[0]), p[1] + t * (q[1] - p[1])))
    return res


def draw_door_overlay(frame: np.ndarray, cfg: dict, tracks: Optional[list[tuple[int, tuple[int, int, int, int]]]] = None) -> None:
    """Outside shaded lightly, the door line, and person boxes with track ids."""
    h, w = frame.shape[:2]
    a, b, out = to_px(cfg, w, h)
    poly = _clip_outside([(0, 0), (w, 0), (w, h), (0, h)], a, b, out)
    if len(poly) >= 3:
        shade = frame.copy()
        cv2.fillPoly(shade, [np.array(poly, dtype=np.int32)], (60, 140, 255))
        cv2.addWeighted(shade, 0.18, frame, 0.82, 0, frame)
    cv2.line(frame, (int(a[0]), int(a[1])), (int(b[0]), int(b[1])), (255, 255, 255), 2)
    cv2.circle(frame, (int(out[0]), int(out[1])), 5, (60, 140, 255), -1)
    for tid, (x1, y1, x2, y2) in tracks or []:
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"#{tid}", (x1, max(y1 - 6, 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
