"""
Per-frame pipeline: camera + YOLO-World + hands/pose + auto zones + item memory
+ dose logic + clips + event log. No GUI — callers (scripts/preview.py, the
FastAPI server) display or stream the frames it returns.
"""
from __future__ import annotations

import json
import os
import subprocess
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

os.environ.setdefault("HF_HUB_OFFLINE", "1")

import cv2
import numpy as np
from ultralytics import YOLO

from recall.cameras import camera_names
from recall.vision.hands   import HandFaceOverlay, draw_overlay
from recall.events.tracker import ItemTracker, Observation, State
from recall.events.store   import EventStore
from recall.events.answers import where_is, state_label
from recall.events.clips   import RollingBuffer, ClipSaver, CLIP_EVENTS
from recall.events.dose    import DoseMonitor, BANNER_TEXT, BANNER_SECOND_TEXT

# Camera choice. OpenCV numbers cameras by position, and Continuity Camera (the iPhone) is only
# in the list while the phone is connected: with it index 0 is the iPhone, without it index 0
# silently becomes the Mac's FaceTime camera. So never trust an index: ask macOS which cameras
# exist, use index 0 only when the iPhone is really there, and otherwise WAIT instead of
# falling back to the Mac camera.
#   RECALL_CAMERA=N        force an OpenCV index (e.g. RECALL_CAMERA=0 with no iPhone = Mac camera)
#   RECALL_CAMERA_NAME=... match another camera name (default "iPhone")
CAMERA_NAME     = os.environ.get("RECALL_CAMERA_NAME", "iPhone")
CAMERA_RETRY_S  = 3.0     # how often to look for the camera again while it's missing
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
    # look-alikes of a pill bottle: detected so they can be told apart, never tracked or used as zones
    "air freshener", "spray bottle", "lotion bottle", "hand sanitizer", "deodorant", "perfume bottle",
    "soap dispenser", "candle", "salt shaker", "coffee mug", "glass of water", "water bottle",
    "snack bag", "speaker",
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

# Things that look like a medication bottle. Detected (at a low bar, so they win the argument), never
# tracked, never zones: a medication box overlapping one of these is that object, not a medication.
NEGATIVE_CLASSES: set[str] = {
    "air freshener", "spray bottle", "lotion bottle", "hand sanitizer", "deodorant", "perfume bottle",
    "soap dispenser", "candle", "salt shaker", "coffee mug", "mug", "cup", "glass of water", "water bottle",
    "snack bag", "speaker", "remote control", "phone",
}
# each detection becomes its own zone keyed by class name
# Only the word "basket" (and its synonyms) makes a basket zone. Bowls and trays are ordinary objects:
# they can no longer become zones, so there is no "went into the bowl".
ZONE_ALIAS   = {"wicker basket": "basket", "storage basket": "basket", "laundry basket": "basket"}
ZONE_ITEM    = {"basket"} | set(ZONE_ALIAS)
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
CONF_MEDICATION          = 0.30   # medication (new detection) — high: a false "pill bottle" is unsafe
CONF_NEGATIVE            = 0.20   # look-alikes: a low bar, so they can veto a medication box
CONF_OTHER               = 0.35
CONF_TRACKING_MEDICATION = 0.15   # medication once already tracked (hysteresis)
CONF_TRACKING_OTHER      = 0.12   # glasses/wallet once already tracked (hysteresis)
_ALL_GLASSES    = {"glasses"} | GLASSES_CLASSES
CONF_FLOOR      = min(CONF_TRACKING_MEDICATION, CONF_TRACKING_OTHER)   # YOLO must return everything we might keep

# Medication sanity checks (a wallet once triggered dose alerts)
MED_MIN_ASPECT   = 1.1    # height / width — bottles and boxes stand upright
MED_MAX_AREA     = 0.15   # fraction of the frame
MED_COMPETE_IOU  = 0.4    # a medication box this overlapped with a competitor is dropped
MED_COMPETITORS  = WALLET_CLASSES | {"book", "phone", "notebook"} | NEGATIVE_CLASSES
CONF_LABEL      = 0.50
IOU_SUPPRESS    = 0.6
DETECT_INTERVAL = 2
# Pill spot lock: once the medication has rested at one spot for 5 s, a medication detection somewhere
# else that is not being carried by a hand is a look-alike (a spray bottle, a mug…) and is ignored.
PILL_REST_S          = 5.0
PILL_RESTING_RADIUS_PX = 40
PILL_LOCK_RADIUS_PX  = 150
PILL_LOCK_TIMEOUT_S  = 60.0   # a spot that has not been seen for this long is forgotten

ZONE_INTERVAL    = DETECT_INTERVAL   # re-derive zones after every detection run (flicker-proof with the linger below)
CONF_ZONE_ITEM   = 0.15        # baskets / shelves are often detected with low confidence
ZONE_EMA         = 0.3        # smoothing factor for box coordinates
ZONE_LINGER_S    = 15.0       # keep a zone this many seconds after it vanishes
MIN_ZONE_AREA_PX = int(0.015 * 1280 * 720)  # 1.5% of frame — filters phantom small zones
# Zone stability (see AutoZones): one basket, one counter, one shelf; a zone appears after 3 s of stable detection
ZONE_STABLE_S    = 3.0
ZONE_GAP_S       = 1.0         # a detection missing for longer than this starts its 3 s clock over
ZONE_MATCH_IOU   = 0.3        # a new stable detection this close to the active zone just updates it
ZONE_ANCHOR_IOU  = 0.5        # a detection must stay this close to where it first appeared to stay "stable"
ZONE_MAX_CANDIDATES = 6
ZONE_MIN_DENSITY = 0.6        # share of detection runs a candidate must be found in over its 3 s
ZONE_MIN_FRAC    = 0.02        # basket/shelf smaller than 2% of the frame: ignored
ZONE_MAX_FRAC    = 0.35        # …or larger than 35%: it is the room, not a basket

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


def pick_camera_index() -> int:
    """OpenCV index of the wanted camera, or RuntimeError if it isn't connected right now."""
    forced = os.environ.get("RECALL_CAMERA")
    if forced is not None:
        return int(forced)
    names = camera_names()
    if any(CAMERA_NAME.lower() in n.lower() for n in names):
        return 0   # OpenCV lists Continuity (iPhone) cameras before the built-in one
    raise RuntimeError(
        f"No '{CAMERA_NAME}' camera right now (macOS sees: {', '.join(names) or 'none'}). Unlock the "
        "iPhone, keep it near the Mac, and check Continuity Camera is on. "
        "(RECALL_CAMERA=0 forces the Mac camera.)")


def box_area(box):
    return max(0, box[2] - box[0]) * max(0, box[3] - box[1])


def medication_shape_ok(x1: int, y1: int, x2: int, y2: int, frame_area: int) -> bool:
    """Taller than wide (h/w ≥ 1.1) and under 15% of the frame; anything else isn't medication."""
    w, h = x2 - x1, y2 - y1
    if w <= 0 or h <= 0:
        return False
    return h / w >= MED_MIN_ASPECT and (w * h) / frame_area < MED_MAX_AREA


# ── auto-zone manager ──────────────────────────────────────────────────────────

def _zone_category(cls_name: str) -> Optional[str]:
    """Which single zone a detection class can feed: basket, counter (table/desk), shelf."""
    n = ZONE_ALIAS.get(cls_name, cls_name)
    if n == "basket":
        return "basket"
    if n in ZONE_SURFACE:
        return "counter"
    if n in ZONE_SINGLE:
        return "shelf"
    return None


class AutoZones:
    """
    Derives zones from YOLO detections, conservatively: at most ONE zone per kind
    (one basket, one counter, one shelf), and a zone only appears after the same detection has been
    stable for ZONE_STABLE_S. Moving or setting up the camera makes YOLO see lots of "baskets" for a
    moment; those never get stable, so they never become zones.

      candidates[kind] : detections being watched, each {"first", "seen", "anchor", "box", "hits", "runs"}
      _zones[kind]     : the active zone {"box": smoothed [x1,y1,x2,y2], "name", "seen"}
    """

    def __init__(self):
        self._zones: dict[str, dict] = {}
        self._cands: dict[str, list[dict]] = {}

    def update(self, detections: list[tuple], frame_area: int = 1280 * 720) -> None:
        """detections: list of (conf, x1, y1, x2, y2, cls_name)"""
        now = time.perf_counter()
        boxes: dict[str, list[tuple]] = {"basket": [], "counter": [], "shelf": []}
        for _, x1, y1, x2, y2, cls_name in detections:
            kind = _zone_category(cls_name)
            if kind is None:
                continue
            box = (x1, y1, x2, y2)
            frac = box_area(box) / frame_area
            if kind in ("basket", "shelf") and not (ZONE_MIN_FRAC <= frac <= ZONE_MAX_FRAC):
                continue   # too small to be a basket, or so big it is the room
            if kind == "counter" and box_area(box) < MIN_ZONE_AREA_PX:
                continue
            boxes[kind].append(box)
        for kind, bs in boxes.items():
            self._update_kind(kind, bs, now)

        # an active zone fades out after it has not been re-confirmed for a while
        self._zones = {k: v for k, v in self._zones.items() if now - v["seen"] <= ZONE_LINGER_S}

    def _update_kind(self, kind: str, boxes: list[tuple], now: float) -> None:
        cands = self._cands.setdefault(kind, [])

        # 1. match this run's detections to the candidates already being watched
        free = list(boxes)
        for c in cands:
            best, best_iou = None, ZONE_ANCHOR_IOU
            for b in free:
                # compared with where it FIRST appeared, so a chain of overlapping detections cannot
                # "walk" a candidate across the room and make it look stable
                v = iou(b, c["anchor"])
                if v > best_iou:
                    best, best_iou = b, v
            if best is not None:
                free.remove(best)
                c["hits"] += 1
                c["seen"] = now
                c["box"] = c["box"] + ZONE_EMA * (np.array(best, dtype=float) - c["box"])
        for c in cands:
            c["runs"] += 1   # one more detection run this candidate has lived through
        # 2. new, unmatched detections start a stability clock (largest first; the list stays small)
        for b in sorted(free, key=box_area, reverse=True):
            if len(cands) < ZONE_MAX_CANDIDATES:
                cands.append({"first": now, "seen": now, "anchor": b, "box": np.array(b, dtype=float), "hits": 1, "runs": 1})
        # 3. a detection that vanishes for more than ZONE_GAP_S starts over
        cands[:] = [c for c in cands if now - c["seen"] <= ZONE_GAP_S]

        # 4. the largest candidate that has been stable long enough is THE zone for this kind
        # stable = there for 3 s AND detected on most runs (a spurious "basket" blinks in and out)
        stable = [c for c in cands
                  if c["seen"] - c["first"] >= ZONE_STABLE_S and c["hits"] >= ZONE_MIN_DENSITY * c["runs"]]
        if not stable:
            return
        top = max(stable, key=lambda c: box_area(tuple(c["box"])))
        active = self._zones.get(kind)
        if active is not None and iou(tuple(top["box"]), tuple(active["box"])) > ZONE_MATCH_IOU:
            active["box"] = active["box"] + ZONE_EMA * (top["box"] - active["box"])
            active["seen"] = now
        else:
            self._zones[kind] = {"box": top["box"].copy(), "name": kind, "seen": now}

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
            gcx = (best_box[0] + best_box[2]) // 2
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
    if event == "left_house":
        return "Left the house"
    if event == "came_back":
        return "Back home"
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
            "silent":   self.detail == "silent",   # door events logged while "Door alerts" is off
        }


@dataclass
class FrameResult:
    annotated: np.ndarray
    raw:       np.ndarray
    events:    list[PipelineEvent] = field(default_factory=list)


class Pipeline:
    """Owns camera, models, trackers, dose monitor, clips and store."""

    def __init__(self, camera_index: Optional[int] = None, hud: bool = True) -> None:
        self.camera_index = camera_index   # None = pick by name (see pick_camera_index)
        self.cap: Optional[cv2.VideoCapture] = None
        self.active_camera_index: Optional[int] = None   # OpenCV index actually open (door camera avoids it)
        self.camera_error: Optional[str] = None
        self._last_cam_try = 0.0
        self._read_fails = 0
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
        self._pill_spot: Optional[tuple[float, float]] = None   # where the medication has been resting
        self._pill_rest: Optional[dict] = None                  # candidate resting spot being timed
        self._pill_seen = 0.0
        self._pill_logged = 0.0
        self._frame_idx = 0
        self._t_prev = time.perf_counter()
        self._last_detections: list[tuple] = []

    # ── lifecycle ──────────────────────────────────────────────────────────────

    def start(self, require_camera: bool = True) -> None:
        """Load models and open the camera. With require_camera=False a missing camera isn't
        fatal: step() keeps looking for it and returns None until it appears."""
        self.model = YOLO(MODEL_PATH)
        self.model.set_classes(CLASSES)

        self.device = "mps"
        try:
            self.model.predict(source=np.zeros((640, 640, 3), dtype=np.uint8),
                               imgsz=IMGSZ, device=self.device, verbose=False)
        except Exception:
            self.device = "cpu"

        self.auto_zones = AutoZones()
        self.trackers   = {item: ItemTracker(item) for item in LOGICAL_ITEMS}
        self.store      = EventStore()
        self.monitor    = DoseMonitor(self.store)
        self.buffer     = RollingBuffer(duration_s=10.0)
        self.clip_saver = ClipSaver(out_dir=CLIPS_DIR)
        self.overlay    = HandFaceOverlay(HAND_MODEL_PATH, FACE_MODEL_PATH, POSE_MODEL_PATH)

        print(f"Doses loaded from DB: {len(self.monitor._doses)}")
        try:
            self._open_camera()
        except RuntimeError as e:
            if require_camera:
                raise
            self.camera_error = str(e)
            print(f"WARNING: {e}")

    def _open_camera(self) -> None:
        """Open the wanted camera (never silently the wrong one). Raises RuntimeError."""
        idx = self.camera_index if self.camera_index is not None else pick_camera_index()
        cap = cv2.VideoCapture(idx)
        if not cap.isOpened():
            cap.release()
            raise RuntimeError(f"Could not open camera index {idx}. Check System Settings > "
                               "Privacy & Security > Camera.")
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        # Continuity Camera can open fine yet stream pure black (phone locked, lens covered,
        # session not started). Say so loudly instead of silently running on black.
        w0 = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h0 = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        brightest = 0.0
        t0 = time.time()
        while time.time() - t0 < 4.0 and brightest < 2.0:
            ok, f = cap.read()
            if ok:
                brightest = max(brightest, float(f.mean()))
        self.cap = cap
        self.active_camera_index = idx
        self.camera_error = None
        self._read_fails = 0
        print(f"Camera {idx} opened ({w0}x{h0}). Device: {self.device}.")
        if brightest < 2.0:
            print(f"WARNING: camera {idx} is streaming black frames. If this is the iPhone: unlock it, "
                  "keep it upright with the rear camera uncovered, and make sure no other app is using it.")

    def close(self) -> None:
        self.overlay.close()
        self.store.close()
        if self.cap is not None:
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

    # ── pill spot lock ─────────────────────────────────────────────────────────

    def _pill_lock(self, dets: list[tuple], hands: list, now: float) -> list[tuple]:
        """Drop medication detections far from the spot the medication has been resting at, unless a
        hand is carrying them. The lock lifts as soon as the medication is handled."""
        def center(d):
            return ((d[1] + d[3]) / 2, (d[2] + d[4]) / 2)

        def far(a, b):
            return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5

        if self._pill_spot is not None and now - self._pill_seen > PILL_LOCK_TIMEOUT_S:
            print("[pill] spot forgotten (not seen for a while)")
            self._pill_spot = None
        if self._pill_rest is not None and now - self._pill_rest["last"] > 2.0:
            self._pill_rest = None

        keep: list[tuple] = []
        carried_any = False
        for d in dets:
            if d[5] not in MEDICATION_CLASSES:
                keep.append(d)
                continue
            if _hand_near_box(d[1:5], hands, radius=30):
                carried_any = True
                keep.append(d)
            elif self._pill_spot is not None and far(center(d), self._pill_spot) > PILL_LOCK_RADIUS_PX:
                if now - self._pill_logged > 5:   # don't spam
                    self._pill_logged = now
                    print(f"[pill] ignored a medication detection {far(center(d), self._pill_spot):.0f}px "
                          "from the resting spot (not in a hand): look-alike")
            else:
                keep.append(d)

        if carried_any:   # being handled: the lock is off until it rests again
            if self._pill_spot is not None:
                print("[pill] spot unlocked (medication is in a hand)")
            self._pill_spot = None
            self._pill_rest = None
            return keep

        survivors = [d for d in keep if d[5] in MEDICATION_CLASSES]
        if survivors:
            c = center(max(survivors, key=lambda d: d[0]))
            if self._pill_spot is None:
                r = self._pill_rest
                if r is not None and far(c, r["center"]) <= PILL_RESTING_RADIUS_PX:
                    r["last"] = now
                    if now - r["since"] >= PILL_REST_S:
                        self._pill_spot, self._pill_seen = r["center"], now
                        self._pill_rest = None
                        print(f"[pill] spot locked at ({c[0]:.0f}, {c[1]:.0f}) after resting {PILL_REST_S:.0f}s")
                else:
                    self._pill_rest = {"center": c, "since": now, "last": now}
            elif far(c, self._pill_spot) <= PILL_LOCK_RADIUS_PX:
                self._pill_seen = now
        return keep

    # ── per-frame ──────────────────────────────────────────────────────────────

    def step(self) -> Optional[FrameResult]:
        """Process one frame. Returns None if the camera read failed."""
        if self.cap is None:   # camera missing: keep looking, never fall back to another one
            if time.time() - self._last_cam_try >= CAMERA_RETRY_S:
                self._last_cam_try = time.time()
                try:
                    self._open_camera()
                except RuntimeError as e:
                    self.camera_error = str(e)
            return None
        ok, frame = self.cap.read()
        if not ok:
            self._read_fails += 1
            if self._read_fails >= 30:
                print("[camera] lost the camera; will reconnect when it's back")
                self.cap.release()
                self.cap = None
                self.camera_error = "Camera disconnected. Reconnecting…"
            return None
        self._read_fails = 0

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
                    if cls_name in NEGATIVE_CLASSES:
                        min_conf = CONF_NEGATIVE
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
            auto_zones.update(last_detections, frame_area)

        self._frame_idx += 1

        # ── Draw zones ────────────────────────────────────────────────────────
        auto_zones.draw(frame)

        # ── MediaPipe every frame ─────────────────────────────────────────────
        mp_result = self.overlay.process(frame)

        last_detections = self._pill_lock(last_detections, mp_result.hands, raw_ts)

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

        # One ruler for the ratio. The shoulder width can be wrong (shoulders cropped or only guessed by
        # the pose model); a human's shoulders are about 2.0–3.4 × their ear-to-ear width, so clamp it to
        # that, and fall back to the head when shoulders are not visible. All of it is in this frame's pixels.
        scale = mp_result.shoulder_width
        hw = mp_result.head_width
        if hw:
            scale = 2.7 * hw if scale is None else min(max(scale, 2.0 * hw), 3.4 * hw)
        monitor.frame_size = self.frame_size

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
            scale,
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

        # ── Draw boxes: ONLY glasses, wallet and medication get a box and a label ───
        # (zones are drawn with their names above; background objects are not drawn at all)
        for conf, x1, y1, x2, y2, cls_name in last_detections:
            if cls_name not in TRACKED:
                continue
            display   = DISPLAY_NAME.get(cls_name, cls_name)
            zone_name = auto_zones.zone_for_box((x1, y1, x2, y2))
            color     = TRACKED_COLORS.get(display, TRACKED_COLORS.get(cls_name, GRAY))
            suffix    = (" IN HAND" if cls_name in held
                         else (f" · {zone_name}" if zone_name else ""))
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
            cv2.putText(frame, f"{display} {conf:.2f}{suffix}", (x1, max(y1 - 6, 12)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)

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
