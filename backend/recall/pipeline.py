"""
Per-frame pipeline: camera + YOLO-World + hands/pose + auto zones + item memory
+ dose logic + clips + event log. No GUI — callers (scripts/preview.py, the
FastAPI server) display or stream the frames it returns.
"""
from __future__ import annotations

import json
import os
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

os.environ.setdefault("HF_HUB_OFFLINE", "1")

import cv2
import numpy as np
from ultralytics import YOLO

from recall.vision.hands   import HandFaceOverlay, draw_overlay
from recall.events.tracker import ItemTracker, Observation, State
from recall.events.store   import EventStore
from recall.events.answers import where_is, state_label
from recall.events.clips   import RollingBuffer, ClipSaver, CLIP_EVENTS
from recall.events.dose    import DoseMonitor, BANNER_TEXT, BANNER_SECOND_TEXT

# OpenCV's device order on this Mac (verified by grabbing frames, not the same as the
# system list): 0 = iPhone (Continuity Camera, 1920x1080), 1 = FaceTime HD Camera.
# The order can change if the iPhone isn't connected when the app starts, so check
# the "Camera N opened" line at startup. Override: RECALL_CAMERA=1 ./backend/scripts/serve.sh
CAMERA_INDEX    = int(os.environ.get("RECALL_CAMERA", "0"))
MODEL_PATH      = "models/yolov8l-worldv2.pt"
HAND_MODEL_PATH = "models/hand_landmarker.task"
FACE_MODEL_PATH = "models/face_landmarker.task"
POSE_MODEL_PATH = "models/pose_landmarker_full.task"
IMGSZ           = 640

def _load_item_descriptions(path: str = "data/items.json") -> tuple[list[str], dict[str, str]]:
    """Return (extra_phrases, phrase→item_name) from items.json, or empty if missing."""
    if not os.path.exists(path):
        return [], {}
    with open(path) as f:
        data = json.load(f)
    phrases, display = [], {}
    for item_name, phrase_list in data.items():
        if not isinstance(phrase_list, list):   # e.g. "mom_phone"
            continue
        for phrase in phrase_list:
            if not isinstance(phrase, str):     # e.g. "schedule" entries
                continue
            if phrase not in display:   # first occurrence wins
                phrases.append(phrase)
                display[phrase] = item_name
    return phrases, display


_item_phrases, _item_display = _load_item_descriptions()

# glasses synonyms — all map to "glasses"
GLASSES_CLASSES: set[str] = {"eyeglasses", "reading glasses", "spectacles"}

# base medication vocab — all map to "medication"
MEDICATION_CLASSES: set[str] = {
    "pill bottle", "medicine bottle", "supplement bottle", "vitamin bottle",
    "prescription bottle", "blister pack", "pill pack", "pills", "medicine box",
}
# merge in any extra phrases from items.json that belong to "medication"
MEDICATION_CLASSES |= {p for p, n in _item_display.items() if n == "medication"}

_CLASSES_BASE = [
    "glasses", "wallet", "keys", "phone", "remote control",
    "mug", "cup", "can", "jar", "book", "notebook",
    "paper", "pen", "laptop", "keyboard", "monitor", "bag", "backpack",
    "plate", "bowl", "spoon", "person",
    "chip can", "jar with lid", "travel mug", "tumbler",
    "computer monitor", "cable",
    # zone surface classes
    "basket", "tray", "table", "desk", "shelf",
    "wicker basket", "storage basket", "laundry basket",   # all become the "basket" zone
    # glasses synonyms
    *sorted(GLASSES_CLASSES),
    # medication
    *sorted(MEDICATION_CLASSES),
]
# append any items.json phrases not already present
_base_set = set(_CLASSES_BASE)
CLASSES = _CLASSES_BASE + [p for p in _item_phrases if p not in _base_set]

# wallet vocabulary — all map to "wallet"
WALLET_CLASSES: set[str] = {"wallet"} | {p for p, n in _item_display.items() if n == "wallet"}

TRACKED      = WALLET_CLASSES | {"glasses"} | GLASSES_CLASSES | MEDICATION_CLASSES
# each detection becomes its own zone keyed by class name
ZONE_ALIAS   = {"wicker basket": "basket", "storage basket": "basket", "laundry basket": "basket"}
ZONE_ITEM    = {"basket", "tray", "bowl"} | set(ZONE_ALIAS)
# largest detection of these collapses to one "counter" zone
ZONE_SURFACE = {"table", "desk"}
# "shelf" gets its own zone per detection
ZONE_SINGLE  = {"shelf"}

DISPLAY_NAME = {
    "chip can":         "can",
    "jar with lid":     "jar",
    "travel mug":       "travel mug",
    "tumbler":          "travel mug",
    "computer monitor": "monitor",
    "monitor":          "monitor",
    **{cls: "glasses"    for cls in GLASSES_CLASSES},
    **{cls: "medication" for cls in MEDICATION_CLASSES},
    **_item_display,   # items.json mappings (may override, always → item name)
}

CONF_GLASSES             = 0.18   # glasses + synonyms (new detection)
CONF_TRACKED             = 0.25   # wallet (new detection)
CONF_MEDICATION          = 0.25   # medication (new detection) — high: a false "pill bottle" is unsafe
CONF_OTHER               = 0.35
CONF_TRACKING_MEDICATION = 0.15   # medication once already tracked (hysteresis)
CONF_TRACKING_OTHER      = 0.12   # glasses/wallet once already tracked (hysteresis)
_ALL_GLASSES    = {"glasses"} | GLASSES_CLASSES
CONF_FLOOR      = min(CONF_TRACKING_MEDICATION, CONF_TRACKING_OTHER)   # YOLO must return everything we might keep

# Medication sanity checks (a wallet once triggered dose alerts)
MED_MIN_ASPECT   = 1.1    # height / width — bottles and boxes stand upright
MED_MAX_AREA     = 0.15   # fraction of the frame
MED_COMPETE_IOU  = 0.4    # a medication box this overlapped with a competitor is dropped
MED_COMPETITORS  = WALLET_CLASSES | {"book", "phone", "notebook"}
CONF_LABEL      = 0.50
IOU_SUPPRESS    = 0.6
DETECT_INTERVAL = 2
ZONE_INTERVAL    = DETECT_INTERVAL   # re-derive zones after every detection run (flicker-proof with the linger below)
CONF_ZONE_ITEM   = 0.15        # baskets / trays / bowls / shelves are often detected with low confidence
ZONE_EMA         = 0.3        # smoothing factor for box coordinates
ZONE_LINGER_S    = 15.0       # keep a zone this many seconds after it vanishes
MIN_ZONE_AREA_PX = int(0.015 * 1280 * 720)  # 1.5% of frame — filters phantom small zones

TRACKED_COLORS = {
    "glasses":    (0,   255,   0),
    "wallet":     (255, 128,   0),
    "medication": (0,   128, 255),
}
GRAY       = (180, 180, 180)
ZONE_COLOR = (180, 220, 255)   # single color for all auto-zones (dashed outlines)


# ── geometry helpers ───────────────────────────────────────────────────────────

def iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    if inter == 0:
        return 0.0
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return inter / (area_a + area_b - inter)


def suppress(detections):
    detections = sorted(detections, key=lambda d: d[0], reverse=True)
    kept = []
    for cand in detections:
        if all(iou(cand[1:5], k[1:5]) <= IOU_SUPPRESS for k in kept):
            kept.append(cand)
    return kept


def box_area(box):
    return max(0, box[2] - box[0]) * max(0, box[3] - box[1])


def medication_shape_ok(x1: int, y1: int, x2: int, y2: int, frame_area: int) -> bool:
    """Taller than wide (h/w ≥ 1.1) and under 15% of the frame; anything else isn't medication."""
    w, h = x2 - x1, y2 - y1
    if w <= 0 or h <= 0:
        return False
    return h / w >= MED_MIN_ASPECT and (w * h) / frame_area < MED_MAX_AREA


# ── auto-zone manager ──────────────────────────────────────────────────────────

class AutoZones:
    """
    Derives zones from YOLO detections.
    Internal state per zone key:
        box  : smoothed [x1, y1, x2, y2] (float)
        name : display name
        seen : time.perf_counter() of last detection
    """

    def __init__(self):
        # key → {"box": np.array, "name": str, "seen": float}
        self._zones: dict[str, dict] = {}

    def update(self, detections: list[tuple]) -> None:
        """
        detections: list of (conf, x1, y1, x2, y2, cls_name)
        """
        now = time.perf_counter()
        seen_keys: set[str] = set()

        # collect zone-class detections
        items:    list[tuple[str, str, tuple]] = []   # (key, name, box4)
        surfaces: list[tuple]                  = []   # (area, box4)

        for _, x1, y1, x2, y2, cls_name in detections:
            box = (x1, y1, x2, y2)
            cls_name = ZONE_ALIAS.get(cls_name, cls_name)
            if cls_name in ZONE_ITEM or cls_name in ZONE_SINGLE:
                # unique key per class+position so multiple baskets get separate zones
                key = f"{cls_name}_{x1}_{y1}"
                # deduplicate: if we already have a close key, reuse it
                key = self._nearest_key(cls_name, box) or key
                items.append((key, cls_name, box))
            elif cls_name in ZONE_SURFACE:
                surfaces.append((box_area(box), box))

        # largest surface → "counter"
        if surfaces:
            surfaces.sort(reverse=True)
            key, name, box = "counter", "counter", surfaces[0][1]
            items.append((key, name, box))

        # apply EMA or register new (skip zones smaller than MIN_ZONE_AREA_PX)
        for key, name, box in items:
            if box_area(box) < MIN_ZONE_AREA_PX:
                continue
            box_arr = np.array(box, dtype=float)
            if key in self._zones:
                prev = self._zones[key]["box"]
                self._zones[key]["box"]  = prev + ZONE_EMA * (box_arr - prev)
                self._zones[key]["seen"] = now
            else:
                self._zones[key] = {"box": box_arr, "name": name, "seen": now}
            seen_keys.add(key)

        # expire stale zones
        self._zones = {
            k: v for k, v in self._zones.items()
            if now - v["seen"] <= ZONE_LINGER_S
        }

    def _nearest_key(self, cls_name: str, box: tuple) -> str | None:
        """Return existing key for same class whose smoothed box overlaps this one."""
        for key, v in self._zones.items():
            if v["name"] == cls_name:
                sb = v["box"].astype(int)
                if iou(box, tuple(sb)) > 0.3:
                    return key
        return None

    def snapshot(self) -> list[dict]:
        """JSON-friendly list of active zones."""
        return [{"name": v["name"], "box": [int(c) for c in v["box"]]}
                for v in list(self._zones.values())]

    def zone_for_box(self, box: tuple) -> str | None:
        """Zone an item sits in. A specific zone (basket, tray…) wins over the big counter, and the
        item's center counts before its bottom edge, so things lying in a basket aren't 'on the counter'."""
        x1, y1, x2, y2 = box
        pts = [((x1 + x2) // 2, (y1 + y2) // 2), ((x1 + x2) // 2, y2)]
        for specific_only in (True, False):
            for cx, cy in pts:
                hits = []
                for v in list(self._zones.values()):
                    if specific_only and v["name"] == "counter":
                        continue
                    zx1, zy1, zx2, zy2 = v["box"].astype(int)
                    if zx1 <= cx <= zx2 and zy1 <= cy <= zy2:
                        hits.append((box_area((zx1, zy1, zx2, zy2)), v["name"]))
                if hits:
                    return min(hits)[1]
        return None

    def zone_for_center(self, cx: int, cy: int) -> str | None:
        """Return name of the smallest zone containing (cx, cy)."""
        candidates = []
        for v in self._zones.values():
            x1, y1, x2, y2 = v["box"].astype(int)
            if x1 <= cx <= x2 and y1 <= cy <= y2:
                candidates.append((box_area((x1, y1, x2, y2)), v["name"]))
        if not candidates:
            return None
        candidates.sort()
        return candidates[0][1]

    def draw(self, canvas: np.ndarray) -> None:
        """Draw dashed outlines + name labels for all active zones."""
        for v in self._zones.values():
            x1, y1, x2, y2 = v["box"].astype(int)
            _draw_dashed_rect(canvas, x1, y1, x2, y2, ZONE_COLOR)
            cv2.putText(canvas, v["name"], (x1 + 4, y1 + 16),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, ZONE_COLOR, 1)


def _draw_dashed_rect(img, x1, y1, x2, y2, color, dash=10, gap=6):
    """Draw a rectangle with dashed lines."""
    pts = [(x1, y1, x2, y1), (x2, y1, x2, y2),
           (x2, y2, x1, y2), (x1, y2, x1, y1)]
    for ax, ay, bx, by in pts:
        length = int(((bx - ax) ** 2 + (by - ay) ** 2) ** 0.5)
        if length == 0:
            continue
        dx, dy = (bx - ax) / length, (by - ay) / length
        pos = 0
        drawing = True
        while pos < length:
            seg = dash if drawing else gap
            end = min(pos + seg, length)
            if drawing:
                p1 = (int(ax + pos * dx), int(ay + pos * dy))
                p2 = (int(ax + end * dx), int(ay + end * dy))
                cv2.line(img, p1, p2, color, 1)
            pos = end
            drawing = not drawing


# ── item-memory helpers ────────────────────────────────────────────────────────

LOGICAL_ITEMS = ["glasses", "wallet", "medication"]


def _logical_item(cls_name: str) -> str:
    """Map raw YOLO class name to logical tracked item name."""
    return DISPLAY_NAME.get(cls_name, cls_name)


def _item_is_held(item: str, held: set[str]) -> bool:
    if item == "glasses":
        return bool(_ALL_GLASSES & held)
    if item == "medication":
        return bool(MEDICATION_CLASSES & held)
    if item == "wallet":
        return bool(WALLET_CLASSES & held)
    return item in held


def _boxes_overlap(a: tuple, b: tuple) -> bool:
    return a[0] < b[2] and a[2] > b[0] and a[1] < b[3] and a[3] > b[1]


def _hand_near_box(box: tuple, hands, radius: int = 80) -> bool:
    """True if any hand landmark bbox comes within `radius` px of item box."""
    x1, y1, x2, y2 = box
    ex1, ey1, ex2, ey2 = x1 - radius, y1 - radius, x2 + radius, y2 + radius
    for hand in hands:
        xs = [p[0] for p in hand.landmarks]
        ys = [p[1] for p in hand.landmarks]
        if ex1 < max(xs) and ex2 > min(xs) and ey1 < max(ys) and ey2 > min(ys):
            return True
    return False


def _build_observation(item: str, last_detections: list, held: set[str],
                        mp_result, auto_zones) -> Observation:
    now = time.perf_counter()
    best_conf, best_box, best_cls = 0.0, None, None
    for conf, x1, y1, x2, y2, cls_name in last_detections:
        if cls_name not in TRACKED:
            continue
        if _logical_item(cls_name) != item:
            continue
        if conf > best_conf:
            best_conf, best_box, best_cls = conf, (x1, y1, x2, y2), cls_name

    if best_box is None:
        return Observation(detected=False, box=None, zone=None,
                           held=False, worn=False, hand_near=False,
                           confidence=0.0, ts=now)

    zone = auto_zones.zone_for_box(best_box)

    is_held = _item_is_held(item, held)
    is_worn = False
    if item == "glasses":
        # primary: glasses box overlaps face landmark bounding box
        if mp_result.face_box is not None and _boxes_overlap(best_box, mp_result.face_box):
            is_worn = True
        # fallback: glasses center (not bottom) is inside top 35% of a "person" box
        if not is_worn:
            gcx = bcx
            gcy = (best_box[1] + best_box[3]) // 2   # true vertical center for face check
            for _, px1, py1, px2, py2, pcls in last_detections:
                if pcls != "person":
                    continue
                top35_y = py1 + int(0.35 * (py2 - py1))
                if px1 <= gcx <= px2 and py1 <= gcy <= top35_y:
                    is_worn = True
                    break

    hand_near = _hand_near_box(best_box, mp_result.hands) if best_box else False

    return Observation(detected=True, box=best_box, zone=zone,
                       held=is_held, worn=is_worn, hand_near=hand_near,
                       confidence=best_conf, ts=now)


def _draw_status_panel(frame: np.ndarray, trackers: dict[str, ItemTracker],
                        monitor=None) -> None:
    """Top-right panel showing each item's current state + dose count."""
    lines = [f"{item:<12}{state_label(t)}" for item, t in trackers.items()]
    if monitor is not None:
        lines.append(monitor.panel_line())
    pad, lh = 8, 22
    w_panel = 300
    h_panel = pad * 2 + lh * len(lines)
    x0 = frame.shape[1] - w_panel - 10
    y0 = 10

    overlay = frame.copy()
    cv2.rectangle(overlay, (x0, y0), (x0 + w_panel, y0 + h_panel), (30, 30, 30), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    for i, line in enumerate(lines):
        cv2.putText(frame, line, (x0 + pad, y0 + pad + lh * (i + 1) - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1)


def _draw_answer(frame: np.ndarray, text: str) -> None:
    """Bottom-center answer overlay, word-wrapped at ~50 chars."""
    words, lines, cur = text.split(), [], ""
    for w in words:
        if len(cur) + len(w) + 1 > 52:
            lines.append(cur.rstrip())
            cur = w + " "
        else:
            cur += w + " "
    if cur.strip():
        lines.append(cur.rstrip())

    lh      = 28
    h_total = lh * len(lines) + 16
    y0      = frame.shape[0] - h_total - 20
    x0      = 20
    w_box   = frame.shape[1] - 40

    overlay = frame.copy()
    cv2.rectangle(overlay, (x0 - 6, y0 - 6),
                  (x0 + w_box, y0 + h_total), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

    for i, line in enumerate(lines):
        cv2.putText(frame, line, (x0, y0 + lh * (i + 1)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 200), 2)


def _draw_dose_debug(frame: np.ndarray, dbg: dict) -> None:
    """Bottom-left overlay: dose state, face visible, palm/mouth ratio, last event."""
    state_str = dbg["state"].upper()
    if dbg["seconds_left"] > 0:
        state_str += f"  {dbg['seconds_left']:.1f}s left"

    ratio = dbg["closest_ratio"]
    ratio_str = (f"{ratio:.2f} (thresh {dbg['threshold']:.2f})"
                 if ratio is not None else "N/A")

    since = dbg["since_ref"]
    since_str = f"{since:.1f}s" if since is not None else "N/A"

    lines = [
        "[DOSE DEBUG]",
        f"State:      {state_str}",
        f"Mouth vis:  {'yes' if dbg['mouth_visible'] else 'no'}",
        f"Hand/mouth: {ratio_str}",
        f"Since last pickup/placed: {since_str}",
        f"Last:       {dbg['last_event'] or '-'}",
    ]

    lh = 20
    h  = lh * len(lines) + 12
    w  = 380
    x0 = 10
    y0 = frame.shape[0] - h - 10

    overlay = frame.copy()
    cv2.rectangle(overlay, (x0 - 4, y0 - 4), (x0 + w, y0 + h), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.72, frame, 0.28, 0, frame)

    for i, line in enumerate(lines):
        color = (110, 200, 255) if i == 0 else (200, 200, 200)
        cv2.putText(frame, line, (x0, y0 + lh * (i + 1) - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)


def _draw_banner(frame: np.ndarray, text: str) -> None:
    """Large center overlay for double-dose alert (orange text on dark bg)."""
    words, lines, cur = text.split(), [], ""
    for w in words:
        if len(cur) + len(w) + 1 > 42:
            lines.append(cur.rstrip())
            cur = w + " "
        else:
            cur += w + " "
    if cur.strip():
        lines.append(cur.rstrip())

    lh      = 38
    h_total = lh * len(lines) + 24
    w_total = frame.shape[1] - 80
    x0      = 40
    y0      = frame.shape[0] // 2 - h_total // 2

    overlay = frame.copy()
    cv2.rectangle(overlay, (x0 - 12, y0 - 12),
                  (x0 + w_total, y0 + h_total + 12), (10, 10, 60), -1)
    cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)

    for i, line in enumerate(lines):
        cv2.putText(frame, line, (x0, y0 + lh * (i + 1)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.85, (60, 130, 255), 2)


def _draw_reset_flash(frame: np.ndarray) -> None:
    """Center confirmation: 'Doses reset' + a drawn check (Hershey fonts have no ✓)."""
    text  = "Doses reset"
    font  = cv2.FONT_HERSHEY_SIMPLEX
    scale, thick = 1.4, 3
    (tw, th), _ = cv2.getTextSize(text, font, scale, thick)
    check_w = 50
    total_w = tw + 24 + check_w
    cx, cy  = frame.shape[1] // 2, frame.shape[0] // 2
    x0      = cx - total_w // 2

    overlay = frame.copy()
    cv2.rectangle(overlay, (x0 - 24, cy - th - 24), (x0 + total_w + 24, cy + 24),
                  (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.8, frame, 0.2, 0, frame)

    green = (90, 220, 120)
    cv2.putText(frame, text, (x0, cy), font, scale, green, thick)
    bx, by = x0 + tw + 24, cy - th // 2
    cv2.polylines(frame, [np.array([(bx, by), (bx + 14, by + 16), (bx + check_w, by - 20)])],
                  False, green, 6)




# ── events + pipeline ──────────────────────────────────────────────────────────

CLIPS_DIR = "data/clips"
ALERT_EVENTS = {"repeat_reach", "possible_second_dose"}

_EVENT_TEXT = {
    "picked_up":    "picked up",
    "placed":       "put down",
    "worn":         "is being worn",
    "taken_off":    "taken off",
    "went_into":    "went into",
    "went_worn":    "was last seen being worn",
    "went_in_hand": "was last seen in hand",
    "reappeared":   "reappeared",
}


def event_message(item: str, event: str, zone: Optional[str] = None,
                  detail: str = "") -> str:
    """Human-readable text for an event (caregiver-safe wording)."""
    if event == "dose_appears_taken":
        return "Dose appears taken"
    if event == "dose_unclear":
        return "Not sure."
    if event == "repeat_reach":
        return BANNER_TEXT
    if event == "possible_second_dose":
        return BANNER_SECOND_TEXT
    if event == "doses_reset":
        return "Doses reset"
    text = f"{item.capitalize()} {_EVENT_TEXT.get(event, event.replace('_', ' '))}"
    if zone:
        text += f" ({zone})"
    return text


def clip_url_for(path: Optional[str]) -> Optional[str]:
    return f"/clips/{os.path.basename(path)}" if path else None


@dataclass
class PipelineEvent:
    item:      str
    event:     str
    zone:      Optional[str]
    ts:        float
    clip_path: Optional[str] = None
    detail:    str = ""

    def to_dict(self) -> dict:
        return {
            "type":     "event",
            "item":     self.item,
            "event":    self.event,
            "zone":     self.zone,
            "ts":       self.ts,
            "clip_url": clip_url_for(self.clip_path),
            "message":  event_message(self.item, self.event, self.zone, self.detail),
        }


@dataclass
class FrameResult:
    annotated: np.ndarray
    raw:       np.ndarray
    events:    list[PipelineEvent] = field(default_factory=list)


class Pipeline:
    """Owns camera, models, trackers, dose monitor, clips and store."""

    def __init__(self, camera_index: int = CAMERA_INDEX, hud: bool = True) -> None:
        self.camera_index = camera_index
        # hud=True (the OpenCV preview): status panel, FPS, dose-debug box, answer/banner/flash
        # text. hud=False (the web UI): the annotated frame carries only detection boxes with
        # labels, zone outlines and the hand/pose skeleton; the UI draws everything else itself.
        self.hud = hud
        self.fps = 0.0
        self.last_alert: Optional[dict] = None

        self._lock = threading.Lock()          # guards the overlay messages + pending events
        self._pending: list[PipelineEvent] = []
        self._answer_msg: Optional[tuple[str, float]] = None   # (text, expire_perf_ts)
        self._banner_msg: Optional[tuple[str, float]] = None
        self._reset_flash_until = 0.0
        self._last_reset_t = 0.0

        self.frame_size = (1280, 720)   # (w, h) of the camera frame
        self._frame_idx = 0
        self._t_prev = time.perf_counter()
        self._last_detections: list[tuple] = []

    # ── lifecycle ──────────────────────────────────────────────────────────────

    def start(self) -> None:
        """Load models and open the camera. Raises RuntimeError if no camera."""
        self.model = YOLO(MODEL_PATH)
        self.model.set_classes(CLASSES)

        self.device = "mps"
        try:
            self.model.predict(source=np.zeros((640, 640, 3), dtype=np.uint8),
                               imgsz=IMGSZ, device=self.device, verbose=False)
        except Exception:
            self.device = "cpu"

        self.cap = cv2.VideoCapture(self.camera_index)
        if not self.cap.isOpened():
            raise RuntimeError(
                f"Could not open camera index {self.camera_index}. Check System Settings > "
                "Privacy & Security > Camera, or try a different index.")
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        self.auto_zones = AutoZones()
        self.trackers   = {item: ItemTracker(item) for item in LOGICAL_ITEMS}
        self.store      = EventStore()
        self.monitor    = DoseMonitor(self.store)
        self.buffer     = RollingBuffer(duration_s=10.0)
        self.clip_saver = ClipSaver(out_dir=CLIPS_DIR)
        self.overlay    = HandFaceOverlay(HAND_MODEL_PATH, FACE_MODEL_PATH, POSE_MODEL_PATH)

        # Continuity Camera can open fine yet stream pure black (phone locked, lens covered,
        # session not started). Say so loudly instead of silently running on black.
        w0 = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h0 = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        brightest = 0.0
        t0 = time.time()
        while time.time() - t0 < 4.0 and brightest < 2.0:
            ok, f = self.cap.read()
            if ok:
                brightest = max(brightest, float(f.mean()))
        print(f"Camera {self.camera_index} opened ({w0}x{h0}). Device: {self.device}.")
        if brightest < 2.0:
            print(f"WARNING: camera {self.camera_index} is streaming black frames. If this is the iPhone: "
                  "unlock it, keep it upright with the rear camera uncovered, and make sure no other app "
                  "is using it. (RECALL_CAMERA=1 uses the Mac's camera.)")
        print(f"Doses loaded from DB: {len(self.monitor._doses)}")

    def close(self) -> None:
        self.overlay.close()
        self.store.close()
        self.cap.release()

    # ── controls (safe to call from another thread) ────────────────────────────

    def ask(self, item: str) -> tuple[str, Optional[str]]:
        """Answer 'where is <item>?'; also shows it on the annotated frame for 4 s."""
        answer = where_is(self.trackers[item])
        with self._lock:
            self._answer_msg = (answer, time.perf_counter() + 4.0)
        return answer, self.store.last_clip_path(item)

    def reset(self) -> bool:
        """Reset today's doses (same as pressing r). Returns False if debounced."""
        now = time.perf_counter()
        print(f"[dose] {datetime.now().strftime('%H:%M:%S.%f')[:-3]} r pressed")
        if now - self._last_reset_t < 1.0:   # debounce vs. last *reset*
            return False
        self._last_reset_t = now
        with self._lock:
            self._reset_flash_until = now + 2.0
            self._pending.append(PipelineEvent("medication", "doses_reset", None, time.time()))
        self.monitor.reset_doses()
        return True

    def set_talk_to_mom(self, enabled: bool) -> None:
        self.monitor.audio_enabled = enabled

    # ── state snapshot (for API / WebSocket) ───────────────────────────────────

    def snapshot(self) -> dict:
        items = {}
        for name, t in self.trackers.items():
            items[name] = {
                "state": t.state.value,
                "label": state_label(t),
                "zone":  t.zone,
                "since": t.state_since_wall,
            }
        return {
            "type":         "state",
            "ts":           time.time(),
            "items":        items,
            "zones":        self.auto_zones.snapshot(),
            "frame":        list(self.frame_size),
            "doses_today":  self.monitor.doses_today,
            "dose_times":   self.monitor.dose_times,
            "last_dose_ts": self.monitor.last_dose_ts,
            "unclear":      self.monitor.unclear_count,
            "last_alert":   self.last_alert,
            "fps":          round(self.fps, 1),
            "talk_to_mom":  self.monitor.audio_enabled,
            "dose":         self.monitor.debug_state(time.perf_counter()),
        }

    # ── per-frame ──────────────────────────────────────────────────────────────

    def step(self) -> Optional[FrameResult]:
        """Process one frame. Returns None if the camera read failed."""
        ok, frame = self.cap.read()
        if not ok:
            return None

        auto_zones, trackers = self.auto_zones, self.trackers
        monitor, store, buffer, clip_saver = self.monitor, self.store, self.buffer, self.clip_saver
        out: list[PipelineEvent] = []

        # ── Buffer raw frame (before any overlays) ────────────────────────────
        raw_ts = time.perf_counter()
        frame_area = frame.shape[0] * frame.shape[1]
        self.frame_size = (frame.shape[1], frame.shape[0])
        raw    = frame.copy()
        small  = buffer.push(raw, raw_ts)

        # ── YOLO every Nth frame ──────────────────────────────────────────────
        if self._frame_idx % DETECT_INTERVAL == 0:
            all_zone_classes = ZONE_ITEM | ZONE_SURFACE | ZONE_SINGLE
            results = self.model.predict(source=frame, conf=CONF_FLOOR,
                                         imgsz=IMGSZ, device=self.device, verbose=False)
            found = []
            for r in results:
                for box in r.boxes:
                    cls_name = CLASSES[int(box.cls[0])]
                    conf = float(box.conf[0])
                    is_zone = cls_name in all_zone_classes
                    logical  = _logical_item(cls_name)
                    tracking = (logical in trackers and trackers[logical].is_tracking)
                    if tracking:
                        min_conf = (CONF_TRACKING_MEDICATION if cls_name in MEDICATION_CLASSES
                                    else CONF_TRACKING_OTHER if cls_name in TRACKED
                                    else CONF_OTHER)
                    else:
                        min_conf = (CONF_MEDICATION if cls_name in MEDICATION_CLASSES
                                    else CONF_GLASSES if cls_name in _ALL_GLASSES
                                    else CONF_TRACKED if cls_name in TRACKED
                                    else CONF_OTHER)
                    if is_zone:
                        zone_min = CONF_ZONE_ITEM if cls_name in (ZONE_ITEM | ZONE_SINGLE) else CONF_TRACKED
                        if conf < zone_min:
                            continue
                    elif conf < min_conf:
                        continue
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    if cls_name in MEDICATION_CLASSES and not medication_shape_ok(x1, y1, x2, y2, frame_area):
                        continue
                    found.append((conf, x1, y1, x2, y2, cls_name))
            # class competition: medication that overlaps a wallet / book / phone / notebook is that object
            rivals = [d[1:5] for d in found if d[5] in MED_COMPETITORS]
            if rivals:
                found = [d for d in found
                         if d[5] not in MEDICATION_CLASSES
                         or all(iou(d[1:5], rv) <= MED_COMPETE_IOU for rv in rivals)]
            self._last_detections = suppress(found)
        last_detections = self._last_detections

        # ── Update zones every ZONE_INTERVAL frames ───────────────────────────
        if self._frame_idx % ZONE_INTERVAL == 0:
            auto_zones.update(last_detections)

        self._frame_idx += 1

        # ── Draw zones ────────────────────────────────────────────────────────
        auto_zones.draw(frame)

        # ── MediaPipe every frame ─────────────────────────────────────────────
        mp_result = self.overlay.process(frame)

        tracked_boxes = [
            (x1, y1, x2, y2, cls_name)
            for _, x1, y1, x2, y2, cls_name in last_detections
            if cls_name in TRACKED
        ]
        held = draw_overlay(frame, mp_result, tracked_boxes)

        # ── Update item trackers + log events ─────────────────────────────────
        med_detected  = False
        med_hand_near = False
        med_box = None
        med_events: list[str] = []
        for item, tracker in trackers.items():
            obs = _build_observation(item, last_detections, held, mp_result, auto_zones)
            if item == "medication":
                med_detected  = obs.detected
                med_hand_near = obs.hand_near
                med_box       = obs.box
            events = tracker.update(obs)
            if item == "medication":
                med_events = [e.event for e in events]
            for ev in events:
                clip_path = None
                if ev.event in CLIP_EVENTS:
                    clip_path = clip_saver.trigger(
                        ev.item, ev.event,
                        buffer.snapshot(pre_s=6.0),
                        ev.ts_wall,
                    )
                store.log(ev.item, ev.event, ev.zone,
                          ev.detail, ev.confidence, ev.ts_wall,
                          clip_path=clip_path)
                print(f"[{ev.item}] {ev.event}"
                      + (f" → {ev.zone}" if ev.zone else "")
                      + (" [clip]" if clip_path else ""))
                out.append(PipelineEvent(ev.item, ev.event, ev.zone, ev.ts_wall,
                                         clip_path, ev.detail))

        # feed current small frame to any active clip jobs
        clip_saver.feed(small, raw_ts)

        # ── Dose monitor ──────────────────────────────────────────────────────
        hand_boxes = [
            (min(p[0] for p in h.landmarks), min(p[1] for p in h.landmarks),
             max(p[0] for p in h.landmarks), max(p[1] for p in h.landmarks))
            for h in mp_result.hands
        ]
        wrists_and_tips = mp_result.pose_wrists + mp_result.pose_index_tips
        if med_box and trackers["medication"].state == State.IN_HAND:
            # a held bottle at the mouth is itself a cue (hands are often hidden by the bottle)
            mx = (med_box[0] + med_box[2]) // 2
            wrists_and_tips += [(mx, med_box[1]), (mx, (med_box[1] + med_box[3]) // 2)]
        for hnd in mp_result.hands:   # palm center + thumb/index/middle tips
            wrists_and_tips.append(hnd.palm_center)
            wrists_and_tips += [hnd.landmarks[i] for i in (4, 8, 12)]

        dose_events = monitor.update(
            trackers["medication"].state,
            med_events,
            med_detected,
            med_hand_near,
            med_box,
            hand_boxes,
            wrists_and_tips,
            mp_result.mouth_center,
            mp_result.mouth_visible,
            mp_result.shoulder_width,
            raw_ts,
            time.time(),
        )
        for dev in dose_events:
            clip_path = None
            if dev.save_clip:
                clip_path = clip_saver.trigger(
                    "medication", dev.event,
                    buffer.snapshot(pre_s=6.0),
                    dev.ts_wall,
                )
            store.log("medication", dev.event, detail=dev.detail,
                      confidence=0.0, ts=dev.ts_wall, clip_path=clip_path)
            print(f"[dose] {dev.event}"
                  + (f" ({dev.detail})" if dev.detail else "")
                  + (" [clip]" if clip_path else ""))
            pev = PipelineEvent("medication", dev.event, None, dev.ts_wall,
                                clip_path, dev.detail)
            out.append(pev)
            if dev.event in ALERT_EVENTS:
                text = BANNER_TEXT if dev.event == "repeat_reach" else BANNER_SECOND_TEXT
                with self._lock:
                    self._banner_msg = (text, time.perf_counter() + 8.0)
                self.last_alert = {"event": dev.event, "message": text, "ts": dev.ts_wall}

        # ── Draw YOLO boxes ───────────────────────────────────────────────────
        all_zone_classes = ZONE_ITEM | ZONE_SURFACE | ZONE_SINGLE
        for conf, x1, y1, x2, y2, cls_name in last_detections:
            if cls_name in all_zone_classes:
                continue
            is_tracked = cls_name in TRACKED
            display    = DISPLAY_NAME.get(cls_name, cls_name)
            if is_tracked:
                zone_name = auto_zones.zone_for_box((x1, y1, x2, y2))
                color      = TRACKED_COLORS.get(display, TRACKED_COLORS.get(cls_name, GRAY))
                thickness  = 3
                font_scale = 0.65
                font_thick = 2
                suffix     = (" IN HAND" if cls_name in held
                              else (f" · {zone_name}" if zone_name else ""))
                label      = f"{display} {conf:.2f}{suffix}"
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

        # ── Status panel + answer overlay + reset flash + banner ──────────────
        if self.hud:
            _draw_status_panel(frame, trackers, monitor)

        now_pc = time.perf_counter()
        with self._lock:
            if self._answer_msg and now_pc < self._answer_msg[1]:
                if self.hud:
                    _draw_answer(frame, self._answer_msg[0])
            elif self._answer_msg:
                self._answer_msg = None

            if now_pc < self._reset_flash_until:
                if self.hud:
                    _draw_reset_flash(frame)

            if self._banner_msg and now_pc < self._banner_msg[1]:
                if self.hud:
                    _draw_banner(frame, self._banner_msg[0])
            elif self._banner_msg:
                self._banner_msg = None

            out = self._pending + out
            self._pending = []

        # ── FPS ───────────────────────────────────────────────────────────────
        t_now = time.perf_counter()
        self.fps = 0.9 * self.fps + 0.1 / max(t_now - self._t_prev, 1e-6)
        self._t_prev = t_now
        if self.hud:
            cv2.putText(frame, f"FPS {self.fps:.1f}", (10, 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            _draw_dose_debug(frame, monitor.debug_state(raw_ts))
        return FrameResult(annotated=frame, raw=raw, events=out)
