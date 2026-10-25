"""
Wraps MediaPipe Tasks API (v1.x) for hand + pose + face landmarks.
Returns pixel-space coordinates; caller owns the lifecycle (use as context manager).

Pose is the primary source for mouth position and hand-to-mouth distance.
Face landmarker is kept only for WORN-glasses detection (face_box).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import cv2
import mediapipe as mp
import numpy as np

# ── Pose landmark indices (BlazePose 33-point) ─────────────────────────────────
_P_NOSE       = 0
_P_MOUTH_L    = 9    # mouth left corner
_P_MOUTH_R    = 10   # mouth right corner
_P_SH_L       = 11   # left shoulder
_P_SH_R       = 12   # right shoulder
_P_ELBOW_L    = 13
_P_ELBOW_R    = 14
_P_WRIST_L    = 15
_P_WRIST_R    = 16
_P_PINKY_L    = 17
_P_PINKY_R    = 18
_P_INDEX_L    = 19   # left index fingertip
_P_INDEX_R    = 20   # right index fingertip
_P_THUMB_L    = 21
_P_THUMB_R    = 22
_P_HIP_L      = 23
_P_HIP_R      = 24
_P_KNEE_L     = 25
_P_KNEE_R     = 26
_P_ANKLE_L    = 27
_P_ANKLE_R    = 28

# Upper-body pose connections to draw
_POSE_CONNECTIONS = [
    (_P_MOUTH_L, _P_MOUTH_R),
    (_P_SH_L,    _P_SH_R),
    (_P_SH_L,    _P_ELBOW_L),  (_P_ELBOW_L, _P_WRIST_L),
    (_P_SH_R,    _P_ELBOW_R),  (_P_ELBOW_R, _P_WRIST_R),
    (_P_WRIST_L, _P_PINKY_L),  (_P_WRIST_L, _P_INDEX_L), (_P_WRIST_L, _P_THUMB_L),
    (_P_WRIST_R, _P_PINKY_R),  (_P_WRIST_R, _P_INDEX_R), (_P_WRIST_R, _P_THUMB_R),
    (_P_SH_L,    _P_HIP_L),    (_P_SH_R,    _P_HIP_R),   (_P_HIP_L,   _P_HIP_R),
    (_P_HIP_L,   _P_KNEE_L),   (_P_HIP_R,   _P_KNEE_R),
    (_P_KNEE_L,  _P_ANKLE_L),  (_P_KNEE_R,  _P_ANKLE_R),
]

# Key joints to highlight (dose-relevant)
_POSE_KEY_JOINTS = (_P_MOUTH_L, _P_MOUTH_R,
                    _P_SH_L,    _P_SH_R,
                    _P_WRIST_L, _P_WRIST_R,
                    _P_INDEX_L, _P_INDEX_R)

FACE_EVERY_N = 5  # run face landmarker on every Nth frame

# ── Face-mesh mouth indices (468-pt, for legacy mouth center fallback) ─────────
_FACE_MOUTH_UPPER = 13
_FACE_MOUTH_LOWER = 14

# ── Hand skeleton connections (21-point) ───────────────────────────────────────
_HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
]

_VIS_MOUTH = 0.5   # minimum pose visibility to count mouth as visible
_VIS_JOINT = 0.3   # minimum pose visibility to include a shoulder / ear / eye
_VIS_HAND  = 0.1   # …but hand points near the lips are partly hidden, which is exactly when pose
                   # visibility drops: keep them at a low bar or the nearest point jumps far away
_P_EAR_L, _P_EAR_R = 7, 8
_P_EYE_L, _P_EYE_R = 2, 5


@dataclass
class HandResult:
    landmarks:   list[tuple[int, int]]   # 21 (x, y) pixel points
    palm_center: tuple[int, int]          # mean of wrist + MCP joints


@dataclass
class OverlayResult:
    hands:             list[HandResult]             = field(default_factory=list)
    # pose-derived (primary for dose logic)
    mouth_center:      Optional[tuple[int, int]]    = None    # midpoint of pose lm 9+10
    mouth_visible:     bool                         = False   # lm 9+10 visibility > 0.5
    pose_wrists:       list[tuple[int, int]]        = field(default_factory=list)
    pose_index_tips:   list[tuple[int, int]]        = field(default_factory=list)
    shoulder_width:    Optional[float]              = None    # px distance lm 11–12
    head_width:        Optional[float]              = None    # px ear-to-ear (or eyes × 2.4): a scale that survives cropped shoulders
    # face-landmarker (for WORN-glasses) with pose head fallback
    face_box:          Optional[tuple[int, int, int, int]] = None
    # full 33-point pose in pixels (for skeleton drawing)
    pose_pts:          list[tuple[int, int]]        = field(default_factory=list)


class HandFaceOverlay:
    """
    Runs MediaPipe hand + pose + face landmarking on BGR frames.
    Use as a context manager:
        with HandFaceOverlay(hand_model, face_model, pose_model) as ov:
            result = ov.process(frame)
    """

    def __init__(self, hand_model: str, face_model: str, pose_model: str) -> None:
        Vision   = mp.tasks.vision
        BaseOpts = mp.tasks.BaseOptions
        RunMode  = Vision.RunningMode

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
        self._pose_lm = Vision.PoseLandmarker.create_from_options(
            Vision.PoseLandmarkerOptions(
                base_options=BaseOpts(model_asset_path=pose_model),
                running_mode=RunMode.VIDEO,
                num_poses=1,
                min_pose_detection_confidence=0.5,
                min_pose_presence_confidence=0.5,
                min_tracking_confidence=0.5,
            )
        )
        self._ts_us = 0  # monotonic synthetic clock shared by all three models
        self._frame_n = 0
        self._face_lm_box: Optional[tuple[int, int, int, int]] = None  # cached between runs

    def __enter__(self) -> "HandFaceOverlay":
        return self

    def __exit__(self, *_) -> None:
        self.close()

    def close(self) -> None:
        self._hand_lm.close()
        self._face_lm.close()
        self._pose_lm.close()

    def process(self, bgr: np.ndarray) -> OverlayResult:
        h, w = bgr.shape[:2]
        self._ts_us += 33_333  # ~30 fps synthetic clock

        rgb = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=bgr[:, :, ::-1].copy(),
        )

        hand_res = self._hand_lm.detect_for_video(rgb, self._ts_us)
        pose_res = self._pose_lm.detect_for_video(rgb, self._ts_us)

        # ── Hands ──────────────────────────────────────────────────────────────
        hands: list[HandResult] = []
        for lm_list in hand_res.hand_landmarks:
            pts = [(int(lm.x * w), int(lm.y * h)) for lm in lm_list]
            palm_pts = [pts[i] for i in (0, 5, 9, 13, 17)]
            cx = int(sum(p[0] for p in palm_pts) / len(palm_pts))
            cy = int(sum(p[1] for p in palm_pts) / len(palm_pts))
            hands.append(HandResult(landmarks=pts, palm_center=(cx, cy)))

        # ── Pose ───────────────────────────────────────────────────────────────
        pose_pts:        list[tuple[int, int]] = []
        mouth_center:    Optional[tuple[int, int]] = None
        mouth_visible:   bool = False
        pose_wrists:     list[tuple[int, int]] = []
        pose_index_tips: list[tuple[int, int]] = []
        shoulder_width:  Optional[float] = None
        head_width:      Optional[float] = None

        if pose_res.pose_landmarks:
            lms = pose_res.pose_landmarks[0]
            pose_pts = [(int(lm.x * w), int(lm.y * h)) for lm in lms]

            # mouth center (landmarks 9 + 10)
            ml, mr = lms[_P_MOUTH_L], lms[_P_MOUTH_R]
            mouth_visible = (ml.visibility > _VIS_MOUTH and mr.visibility > _VIS_MOUTH)
            mouth_center = (
                (int(ml.x * w) + int(mr.x * w)) // 2,
                (int(ml.y * h) + int(mr.y * h)) // 2,
            )

            # wrists
            for idx in (_P_WRIST_L, _P_WRIST_R):
                lm = lms[idx]
                if lm.visibility > _VIS_HAND:
                    pose_wrists.append(pose_pts[idx])

            # index fingertips
            for idx in (_P_INDEX_L, _P_INDEX_R, _P_PINKY_L, _P_PINKY_R, _P_THUMB_L, _P_THUMB_R):
                lm = lms[idx]
                if lm.visibility > _VIS_HAND:
                    pose_index_tips.append(pose_pts[idx])

            # head width: a second ruler, in the same pixels as the mouth and the hands
            head_width = None
            el, er = lms[_P_EAR_L], lms[_P_EAR_R]
            ye, yr = lms[_P_EYE_L], lms[_P_EYE_R]
            if el.visibility > _VIS_JOINT and er.visibility > _VIS_JOINT:
                head_width = ((pose_pts[_P_EAR_L][0] - pose_pts[_P_EAR_R][0]) ** 2
                              + (pose_pts[_P_EAR_L][1] - pose_pts[_P_EAR_R][1]) ** 2) ** 0.5
            elif ye.visibility > _VIS_JOINT and yr.visibility > _VIS_JOINT:
                head_width = 2.4 * ((pose_pts[_P_EYE_L][0] - pose_pts[_P_EYE_R][0]) ** 2
                                    + (pose_pts[_P_EYE_L][1] - pose_pts[_P_EYE_R][1]) ** 2) ** 0.5

            # shoulder width
            sl, sr = lms[_P_SH_L], lms[_P_SH_R]
            if sl.visibility > _VIS_JOINT and sr.visibility > _VIS_JOINT:
                dx = pose_pts[_P_SH_L][0] - pose_pts[_P_SH_R][0]
                dy = pose_pts[_P_SH_L][1] - pose_pts[_P_SH_R][1]
                shoulder_width = (dx * dx + dy * dy) ** 0.5

        # ── Face box: face landmarker first, pose head (lm 0-10) as fallback ──
        # Face landmarker is expensive and only feeds WORN-glasses detection
        # (pose handles the mouth), so run it every FACE_EVERY_N frames and
        # reuse the last box in between.
        if self._frame_n % FACE_EVERY_N == 0:
            face_res = self._face_lm.detect_for_video(rgb, self._ts_us)
            self._face_lm_box = None
            if face_res.face_landmarks:
                lms_f = face_res.face_landmarks[0]
                xs = [int(lm.x * w) for lm in lms_f]
                ys = [int(lm.y * h) for lm in lms_f]
                self._face_lm_box = (min(xs), min(ys), max(xs), max(ys))
        self._frame_n += 1

        face_box: Optional[tuple[int, int, int, int]] = self._face_lm_box
        if face_box is None and pose_pts:
            head_pts = [(pose_pts[i], lms[i]) for i in range(11) if i < len(lms)]
            visible  = [pt for pt, lm in head_pts if lm.visibility > _VIS_JOINT]
            if visible:
                xs = [p[0] for p in visible]
                ys = [p[1] for p in visible]
                face_box = (min(xs), min(ys), max(xs), max(ys))

        return OverlayResult(
            hands=hands,
            mouth_center=mouth_center,
            mouth_visible=mouth_visible,
            pose_wrists=pose_wrists,
            pose_index_tips=pose_index_tips,
            shoulder_width=shoulder_width,
            head_width=head_width,
            face_box=face_box,
            pose_pts=pose_pts,
        )


# ── drawing helpers ────────────────────────────────────────────────────────────

def draw_overlay(frame: np.ndarray, result: OverlayResult,
                 tracked_boxes: list[tuple[int, int, int, int, str]]) -> set[str]:
    """
    Draw pose skeleton, hand skeleton, mouth dot, and wrist-to-mouth lines.
    Returns set of cls_names whose boxes overlap a hand.
    """
    # ── Pose skeleton (draw first so hands render on top) ─────────────────────
    if result.pose_pts:
        pts = result.pose_pts
        for a, b in _POSE_CONNECTIONS:
            if a < len(pts) and b < len(pts):
                cv2.line(frame, pts[a], pts[b], (70, 70, 70), 1)
        for idx in _POSE_KEY_JOINTS:
            if idx < len(pts):
                cv2.circle(frame, pts[idx], 3, (110, 110, 210), -1)

    # ── Mouth dot (pose-derived) ───────────────────────────────────────────────
    if result.mouth_center:
        color = (0, 200, 255) if result.mouth_visible else (0, 80, 200)
        cv2.circle(frame, result.mouth_center, 5, color, -1)

    # ── Hand skeleton + wrist-to-mouth lines ───────────────────────────────────
    mouth = result.mouth_center
    for hand in result.hands:
        pts = hand.landmarks
        for a, b in _HAND_CONNECTIONS:
            cv2.line(frame, pts[a], pts[b], (255, 255, 255), 1)
        cv2.circle(frame, hand.palm_center, 6, (255, 255, 255), -1)

        if mouth:
            cv2.line(frame, hand.palm_center, mouth, (200, 200, 200), 1)

    # ── Pose wrist-to-mouth lines (when hands not detected by MediaPipe Hands) -
    if mouth:
        for wp in result.pose_wrists:
            cv2.line(frame, wp, mouth, (150, 150, 150), 1)

    return _held_items(result.hands, tracked_boxes)


def _held_items(hands: list[HandResult],
                tracked_boxes: list[tuple[int, int, int, int, str]]) -> set[str]:
    """Return cls_names of tracked boxes that overlap any hand's bounding rect."""
    held = set()
    for hand in hands:
        xs = [p[0] for p in hand.landmarks]
        ys = [p[1] for p in hand.landmarks]
        hx1, hy1, hx2, hy2 = min(xs), min(ys), max(xs), max(ys)
        for x1, y1, x2, y2, cls_name in tracked_boxes:
            if x1 < hx2 and x2 > hx1 and y1 < hy2 and y2 > hy1:
                held.add(cls_name)
    return held
