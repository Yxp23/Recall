"""
Wraps MediaPipe Tasks API (v1.x) for hand + face landmarks.
Returns pixel-space coordinates; caller owns the lifecycle (use as context manager).
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional
import mediapipe as mp
import numpy as np

# Face-mesh landmark indices for mouth center (standard 468-point topology)
_MOUTH_UPPER = 13
_MOUTH_LOWER = 14

# MediaPipe hand connections (21-point skeleton)
_HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),       # thumb
    (0, 5), (5, 6), (6, 7), (7, 8),       # index
    (5, 9), (9, 10), (10, 11), (11, 12),  # middle
    (9, 13), (13, 14), (14, 15), (15, 16), # ring
    (13, 17), (17, 18), (18, 19), (19, 20), # pinky
    (0, 17),                               # palm base
]


@dataclass
class HandResult:
    landmarks: list[tuple[int, int]]   # 21 (x, y) pixel points
    palm_center: tuple[int, int]        # mean of wrist + MCP joints


@dataclass
class OverlayResult:
    hands: list[HandResult] = field(default_factory=list)
    mouth_center: Optional[tuple[int, int]] = None


class HandFaceOverlay:
    """
    Runs MediaPipe hand + face landmarking on BGR frames.
    Use as a context manager:
        with HandFaceOverlay("models/hand_landmarker.task",
                             "models/face_landmarker.task") as overlay:
            result = overlay.process(frame)
    """

    def __init__(self, hand_model: str, face_model: str) -> None:
        Vision    = mp.tasks.vision
        BaseOpts  = mp.tasks.BaseOptions
        RunMode   = Vision.RunningMode

        self._hand_lm = Vision.HandLandmarker.create_from_options(
            Vision.HandLandmarkerOptions(
                base_options=BaseOpts(model_asset_path=hand_model),
                running_mode=RunMode.VIDEO,
                num_hands=2,
                min_hand_detection_confidence=0.5,
                min_hand_presence_confidence=0.5,
                min_tracking_confidence=0.5,
            )
        )
        self._face_lm = Vision.FaceLandmarker.create_from_options(
            Vision.FaceLandmarkerOptions(
                base_options=BaseOpts(model_asset_path=face_model),
                running_mode=RunMode.VIDEO,
                num_faces=1,
                min_face_detection_confidence=0.5,
                min_face_presence_confidence=0.5,
                min_tracking_confidence=0.5,
            )
        )
        self._ts_us = 0  # monotonic timestamp in microseconds

    def __enter__(self) -> "HandFaceOverlay":
        return self

    def __exit__(self, *_) -> None:
        self.close()

    def close(self) -> None:
        self._hand_lm.close()
        self._face_lm.close()

    def process(self, bgr: np.ndarray) -> OverlayResult:
        h, w = bgr.shape[:2]
        self._ts_us += 33_333  # ~30 fps synthetic clock; keeps VIDEO mode happy

        rgb = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=bgr[:, :, ::-1].copy(),
        )

        hand_res = self._hand_lm.detect_for_video(rgb, self._ts_us)
        face_res = self._face_lm.detect_for_video(rgb, self._ts_us)

        hands: list[HandResult] = []
        for lm_list in hand_res.hand_landmarks:
            pts = [(int(lm.x * w), int(lm.y * h)) for lm in lm_list]
            # Palm center: wrist (0) + five MCP joints (5,9,13,17)
            palm_pts = [pts[i] for i in (0, 5, 9, 13, 17)]
            cx = int(sum(p[0] for p in palm_pts) / len(palm_pts))
            cy = int(sum(p[1] for p in palm_pts) / len(palm_pts))
            hands.append(HandResult(landmarks=pts, palm_center=(cx, cy)))

        mouth: Optional[tuple[int, int]] = None
        if face_res.face_landmarks:
            lms = face_res.face_landmarks[0]
            ux, uy = int(lms[_MOUTH_UPPER].x * w), int(lms[_MOUTH_UPPER].y * h)
            lx, ly = int(lms[_MOUTH_LOWER].x * w), int(lms[_MOUTH_LOWER].y * h)
            mouth = ((ux + lx) // 2, (uy + ly) // 2)

        return OverlayResult(hands=hands, mouth_center=mouth)


# ── drawing helpers ────────────────────────────────────────────────────────────

def draw_overlay(frame: np.ndarray, result: OverlayResult,
                 tracked_boxes: list[tuple[int, int, int, int, str]]) -> None:
    """
    Draw hands/face overlay and annotate tracked boxes that are IN HAND.
    tracked_boxes: list of (x1, y1, x2, y2, cls_name)
    Mutates frame in place.
    """
    import cv2

    mouth = result.mouth_center

    for hand in result.hands:
        pts = hand.landmarks
        # skeleton
        for a, b in _HAND_CONNECTIONS:
            cv2.line(frame, pts[a], pts[b], (255, 255, 255), 1)
        # palm dot
        cv2.circle(frame, hand.palm_center, 6, (255, 255, 255), -1)

        if mouth:
            dist = int(((hand.palm_center[0] - mouth[0]) ** 2 +
                        (hand.palm_center[1] - mouth[1]) ** 2) ** 0.5)
            cv2.line(frame, hand.palm_center, mouth, (200, 200, 200), 1)
            mid = ((hand.palm_center[0] + mouth[0]) // 2,
                   (hand.palm_center[1] + mouth[1]) // 2)
            cv2.putText(frame, str(dist), mid,
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

    if mouth:
        cv2.circle(frame, mouth, 5, (0, 200, 255), -1)

    # mark tracked items whose box overlaps any hand bounding box
    held = _held_items(result.hands, tracked_boxes)
    for cls_name in held:
        # caller draws the boxes; we just return the set — but we can also
        # annotate directly since we have the frame
        pass

    return held


def _held_items(hands: list[HandResult],
                tracked_boxes: list[tuple[int, int, int, int, str]]) -> set[str]:
    """Return cls_names of tracked boxes that overlap any hand's bounding rect."""
    held = set()
    for hand in hands:
        xs = [p[0] for p in hand.landmarks]
        ys = [p[1] for p in hand.landmarks]
        hx1, hy1, hx2, hy2 = min(xs), min(ys), max(xs), max(ys)
        for x1, y1, x2, y2, cls_name in tracked_boxes:
            if (x1 < hx2 and x2 > hx1 and y1 < hy2 and y2 > hy1):
                held.add(cls_name)
    return held
