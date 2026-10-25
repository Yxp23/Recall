"""
Door camera ("left the house"). Runs in its own thread, completely separate from the kitchen
pipeline: its own camera (the Mac's built-in webcam), rolling buffer and clip saver, plus a tiny
person detector and a centroid tracker. Never touches kitchen state.
"""
from __future__ import annotations

import os
import threading
import time
import traceback
from dataclasses import dataclass
from typing import Callable, Optional

import cv2
import numpy as np

from recall.door_geometry import DOOR_JSON, draw_door_overlay, is_outside, load_door_config, to_px
from recall.events.clips import ClipSaver, RollingBuffer
from recall.events.store import EventStore
from recall.pipeline import PipelineEvent

DOOR_MODEL_PATH   = "models/yolov8n.pt"   # COCO; we only keep class 0 (person)
DOOR_W, DOOR_H    = 640, 360
DOOR_IMGSZ        = 320
DOOR_DEVICE       = "cpu"                 # NEVER mps: only the kitchen pipeline uses the GPU (see _run)
DETECT_EVERY      = 3                     # run the detector on every 3rd frame
DETECT_CONF       = 0.35
MAX_FPS           = 20.0                  # don't spin the CPU faster than this
TRACK_MAX_DIST_PX = 120                   # nearest-neighbour match radius
TRACK_DROP_S      = 1.5                   # drop a track after this long unseen
STAY_S            = 1.0                   # must stay on the new side this long
COOLDOWN_S        = 30.0                  # per direction
CONFIG_CHECK_S    = 3.0
CAMERA_RETRY_S    = 3.0


@dataclass
class Track:
    id: int
    centroid: tuple[float, float]
    box: tuple[int, int, int, int]
    last_seen: float
    stable_out: bool      # side we last confirmed (True = outside)
    cur_out: bool         # side right now
    since: float          # when cur side began


class CentroidTracker:
    """Nearest-neighbour matching of person centroids; tracks drop after TRACK_DROP_S unseen."""

    def __init__(self) -> None:
        self.tracks: dict[int, Track] = {}
        self._next = 1

    def update(self, boxes: list[tuple[int, int, int, int]], now: float,
               side: Callable[[tuple[float, float]], bool]) -> list[Track]:
        """Returns the tracks that were matched/created on this frame."""
        dets = [((b[0] + b[2]) / 2, (b[1] + b[3]) / 2, b) for b in boxes]
        pairs = sorted(
            ((((t.centroid[0] - cx) ** 2 + (t.centroid[1] - cy) ** 2) ** 0.5, tid, di)
             for tid, t in self.tracks.items() for di, (cx, cy, _) in enumerate(dets)),
        )
        used_t: set[int] = set()
        used_d: set[int] = set()
        seen: list[Track] = []
        for dist, tid, di in pairs:
            if dist > TRACK_MAX_DIST_PX or tid in used_t or di in used_d:
                continue
            used_t.add(tid)
            used_d.add(di)
            t = self.tracks[tid]
            cx, cy, box = dets[di]
            t.centroid, t.box, t.last_seen = (cx, cy), box, now
            out = side(t.centroid)
            if out != t.cur_out:
                t.cur_out, t.since = out, now
            seen.append(t)
        for di, (cx, cy, box) in enumerate(dets):
            if di in used_d:
                continue
            out = side((cx, cy))
            t = Track(self._next, (cx, cy), box, now, out, out, now)
            self.tracks[t.id] = t
            self._next += 1
            seen.append(t)
        self.tracks = {i: t for i, t in self.tracks.items() if now - t.last_seen <= TRACK_DROP_S}
        return seen


class DoorCamera:
    def __init__(self, camera_index: int, store: EventStore,
                 on_event: Callable[[PipelineEvent], None],
                 alerts_enabled: Callable[[], bool]) -> None:
        self.camera_index = camera_index
        self._store = store
        self._on_event = on_event
        self._alerts_enabled = alerts_enabled

        self.running = False                       # camera open and streaming
        self.error: Optional[str] = None
        self.fps = 0.0
        self.people = 0
        self.last_event: Optional[dict] = None
        self.annotated: Optional[np.ndarray] = None
        self.frame_id = 0

        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="door", daemon=True)
        self._tracker = CentroidTracker()
        self._last_fire = {"left_house": float("-inf"), "came_back": float("-inf")}
        self._cfg = load_door_config()
        self._cfg_mtime = self._mtime()
        self._cfg_checked = 0.0

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=5)

    def snapshot(self) -> dict:
        return {"enabled": self.running, "fps": round(self.fps, 1), "people": self.people,
                "last_event": self.last_event, "error": self.error}

    # ── internals ──────────────────────────────────────────────────────────────

    @staticmethod
    def _mtime() -> float:
        try:
            return os.path.getmtime(DOOR_JSON)
        except OSError:
            return 0.0

    def _reload_config(self, now: float) -> None:
        if now - self._cfg_checked < CONFIG_CHECK_S:
            return
        self._cfg_checked = now
        m = self._mtime()
        if m != self._cfg_mtime:
            self._cfg_mtime = m
            self._cfg = load_door_config()
            print(f"[door] reloaded {DOOR_JSON}")

    def _open(self) -> Optional[cv2.VideoCapture]:
        cap = cv2.VideoCapture(self.camera_index)
        if not cap.isOpened():
            cap.release()
            return None
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, DOOR_W)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, DOOR_H)
        return cap

    def _run(self) -> None:
        """Thread entry: whatever goes wrong here is logged and the kitchen keeps running."""
        try:
            self._run_inner()
        except Exception as e:
            self.error = f"door thread crashed: {e}"
            traceback.print_exc()
            print(f"WARNING: {self.error}; door features disabled, kitchen unaffected")
        finally:
            self.running = False

    def _run_inner(self) -> None:
        from ultralytics import YOLO   # same package the kitchen uses; imported late so it's the thread's cost

        # The detector runs on the CPU, always. Two threads driving MPS at once crashes torch
        # (NSInvalidArgumentException in the Metal compute pipeline), so the GPU belongs to the
        # kitchen pipeline alone. Nothing here may use device="mps" or .to("mps").
        device = DOOR_DEVICE
        try:
            model = YOLO(DOOR_MODEL_PATH)
            model.predict(np.zeros((DOOR_H, DOOR_W, 3), np.uint8), imgsz=DOOR_IMGSZ, device=device,
                          classes=[0], verbose=False)
        except Exception as e:   # missing model etc. must never take the kitchen down
            self.error = f"door model failed to load: {e}"
            print(f"WARNING: {self.error}; door features disabled")
            return

        buffer = RollingBuffer(duration_s=10.0)
        clips = ClipSaver(out_dir="data/clips")
        cap = self._open()
        if cap is None:
            self.error = f"could not open the door camera (index {self.camera_index})"
            print(f"WARNING: {self.error}; door features disabled")
            return
        self.running = True
        print(f"[door] camera {self.camera_index} opened ({DOOR_W}x{DOOR_H}); detector on {device}")

        frame_n = 0
        fails = 0
        errors = 0
        t_prev = time.perf_counter()
        last_boxes: list[tuple[int, int, int, int]] = []
        next_try = 0.0
        while not self._stop.is_set():
            t_loop = time.perf_counter()
            if cap is None:   # camera dropped: keep trying
                if t_loop >= next_try:
                    next_try = t_loop + CAMERA_RETRY_S
                    cap = self._open()
                    self.running = cap is not None
                if cap is None:
                    time.sleep(0.2)
                    continue
            ok, raw = cap.read()
            if not ok:
                fails += 1
                if fails >= 50:
                    print("[door] lost the camera; will reconnect")
                    cap.release()
                    cap, fails, self.running = None, 0, False
                time.sleep(0.05)
                continue
            fails = 0
            frame = raw if raw.shape[:2] == (DOOR_H, DOOR_W) else cv2.resize(raw, (DOOR_W, DOOR_H))
            try:
                ts = time.perf_counter()
                small = buffer.push(frame.copy(), ts)
                self._reload_config(ts)
                a, b, out = to_px(self._cfg, DOOR_W, DOOR_H)
                side = lambda p: is_outside(a, b, out, p)  # noqa: E731

                if frame_n % DETECT_EVERY == 0:
                    res = model.predict(frame, imgsz=DOOR_IMGSZ, device=device, classes=[0],
                                        conf=DETECT_CONF, verbose=False)
                    last_boxes = [tuple(map(int, bx.xyxy[0])) for r in res for bx in r.boxes]
                    seen = self._tracker.update(last_boxes, ts, side)
                    self.people = len(seen)
                    for t in seen:
                        self._check_crossing(t, ts, buffer, clips)
                frame_n += 1
                clips.feed(small, ts)

                vis = frame.copy()
                draw_door_overlay(vis, self._cfg, [(t.id, t.box) for t in self._tracker.tracks.values()
                                                  if ts - t.last_seen <= 0.5])
                self.annotated = vis
                self.frame_id += 1

            except Exception:
                errors += 1
                if errors <= 3 or errors % 100 == 0:
                    traceback.print_exc()
                    print(f"[door] frame error #{errors}; skipping this frame")
                if errors >= 300:
                    self.error = "door camera stopped after repeated errors"
                    print(f"WARNING: {self.error}; kitchen unaffected")
                    break
                time.sleep(0.1)
                continue
            now = time.perf_counter()
            self.fps = 0.9 * self.fps + 0.1 / max(now - t_prev, 1e-6)
            t_prev = now
            spare = 1.0 / MAX_FPS - (now - t_loop)
            if spare > 0:
                time.sleep(spare)

        if cap is not None:
            cap.release()
        self.running = False

    def _check_crossing(self, t: Track, ts: float, buffer: RollingBuffer, clips: ClipSaver) -> None:
        """Fire once a person has stayed on the other side of the line for STAY_S."""
        if t.cur_out == t.stable_out or ts - t.since < STAY_S:
            return
        t.stable_out = t.cur_out
        event = "left_house" if t.cur_out else "came_back"
        if ts - self._last_fire[event] < COOLDOWN_S:
            print(f"[door] {event} (track #{t.id}) suppressed: cooldown")
            return
        self._last_fire[event] = ts
        ts_wall = time.time()
        clip_path = clips.trigger("door", event, buffer.snapshot(pre_s=6.0), ts_wall)
        alert = self._alerts_enabled()
        detail = "alerted" if alert else "silent"      # silent = logged but no caregiver alert
        self._store.log("door", event, None, detail, 0.0, ts_wall, clip_path=clip_path)
        self.last_event = {"event": event, "ts": ts_wall}
        print(f"[door] {event} (track #{t.id}, {detail})" + (" [clip]" if clip_path else ""))
        self._on_event(PipelineEvent("door", event, None, ts_wall, clip_path, detail))
