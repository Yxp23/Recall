"""
Rolling frame buffer and background clip writer.
"""
from __future__ import annotations

import os
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

import cv2
import numpy as np

CLIP_EVENTS = {"went_into", "went_worn", "went_in_hand", "placed", "picked_up"}
_OUT_W, _OUT_H = 640, 360
_PRE_S  = 6.0
_POST_S = 2.0


# ── rolling buffer ─────────────────────────────────────────────────────────────

class RollingBuffer:
    """Keeps the last `duration_s` seconds of frames, downscaled to 640×360."""

    def __init__(self, duration_s: float = 10.0, fps_hint: int = 30) -> None:
        maxlen = int(duration_s * fps_hint)
        self._buf: deque[tuple[float, np.ndarray]] = deque(maxlen=maxlen)

    def push(self, frame: np.ndarray, ts: float) -> np.ndarray:
        """Downscale frame, store it, and return the small copy."""
        small = cv2.resize(frame, (_OUT_W, _OUT_H))
        self._buf.append((ts, small))
        return small

    def snapshot(self, pre_s: float = _PRE_S) -> list[tuple[float, np.ndarray]]:
        """Return a list of (ts, small_frame) for the last pre_s seconds."""
        cutoff = time.perf_counter() - pre_s
        return [(t, f) for t, f in self._buf if t >= cutoff]


# ── clip job ───────────────────────────────────────────────────────────────────

@dataclass
class _ClipJob:
    path:        str
    pre_frames:  list[tuple[float, np.ndarray]]
    deadline:    float                              # perf_counter; collect post until here
    post_frames: list[tuple[float, np.ndarray]] = field(default_factory=list)
    writing:     bool = False                       # True once writing thread starts


# ── clip saver ─────────────────────────────────────────────────────────────────

class ClipSaver:
    """
    trigger() at event time → plans a clip job (pre + post frames).
    feed()    each frame    → appends to active jobs; starts write thread at deadline.
    """

    def __init__(self, out_dir: str = "data/clips") -> None:
        self._out_dir = out_dir
        os.makedirs(out_dir, exist_ok=True)
        self._jobs: list[_ClipJob] = []
        self._lock = threading.Lock()

    def trigger(self, item: str, event: str,
                pre_snapshot: list[tuple[float, np.ndarray]],
                ts_wall: float) -> Optional[str]:
        """
        Schedule a clip. Returns the planned output path (written asynchronously).
        Returns None if the event is not clip-worthy.
        """
        if event not in CLIP_EVENTS:
            return None
        dt   = datetime.fromtimestamp(ts_wall).strftime("%Y%m%d_%H%M%S")
        path = os.path.join(self._out_dir, f"{dt}_{item}_{event}.mp4")
        job  = _ClipJob(
            path=path,
            pre_frames=pre_snapshot,
            deadline=time.perf_counter() + _POST_S,
        )
        with self._lock:
            self._jobs.append(job)
        return path

    def feed(self, small: np.ndarray, ts: float) -> None:
        """
        Feed a downscaled frame to active jobs.
        Starts the write thread for jobs whose post-capture window has closed.
        """
        with self._lock:
            remaining = []
            for job in self._jobs:
                if job.writing:
                    continue
                if ts <= job.deadline:
                    job.post_frames.append((ts, small.copy()))
                    remaining.append(job)
                else:
                    job.writing = True
                    frames = list(job.pre_frames) + list(job.post_frames)
                    threading.Thread(
                        target=_write_mp4,
                        args=(job.path, frames),
                        daemon=True,
                    ).start()
                    # job removed from active list — thread owns the data now
            self._jobs = remaining


# ── writer ─────────────────────────────────────────────────────────────────────

def _write_mp4(path: str, frames: list[tuple[float, np.ndarray]]) -> None:
    if not frames:
        return
    duration = frames[-1][0] - frames[0][0] if len(frames) > 1 else 1.0
    fps = max(5.0, min(30.0, len(frames) / max(duration, 0.1)))
    h, w = frames[0][1].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(path, fourcc, fps, (w, h))
    for _, f in frames:
        writer.write(f)
    writer.release()
    print(f"[clip] {os.path.basename(path)}  ({len(frames)} frames @ {fps:.0f} fps)")
