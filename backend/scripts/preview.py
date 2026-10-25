"""
Step 4 preview: thin OpenCV window around recall.pipeline.Pipeline.
Run from repo root: python backend/scripts/preview.py
Keys: g=glasses  w=wallet  m=medication  c=play clip  r=reset doses  q=quit
"""
import os
import sys
import time
from typing import Optional
import cv2
import numpy as np

sys.path.insert(0, "backend")
from recall.pipeline import Pipeline


def _load_clip(path: str) -> tuple[list[np.ndarray], float]:
    """Return (frames, seconds_per_frame) for a clip file, or ([], 1/15) on failure."""
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        return [], 1 / 15
    fps    = cap.get(cv2.CAP_PROP_FPS) or 15
    frames = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        frames.append(f)
    cap.release()
    return frames, 1.0 / max(fps, 1)


def main():
    pipe = Pipeline()
    try:
        pipe.start()
    except RuntimeError as e:
        print(f"ERROR: {e}")
        return

    last_asked_item: Optional[str] = None                 # for c key
    clip_frames:     list[np.ndarray] = []                # loaded clip for playback
    clip_frame_idx:  int              = 0
    clip_frame_spf:  float            = 1 / 15            # seconds per frame
    clip_next_ts:    float            = 0.0

    print("g=glasses  w=wallet  m=medication  c=play clip  r=reset doses  q=quit")

    while True:
        res = pipe.step()
        if res is None:
            print("Frame read failed — camera disconnected?")
            break

        # ── Clip playback (main thread — macOS requirement) ───────────────
        if clip_frames:
            now_pc = time.perf_counter()
            if now_pc >= clip_next_ts:
                cv2.imshow("Recall clip", clip_frames[clip_frame_idx])
                clip_frame_idx += 1
                clip_next_ts    = now_pc + clip_frame_spf
                if clip_frame_idx >= len(clip_frames):
                    cv2.destroyWindow("Recall clip")
                    clip_frames    = []
                    clip_frame_idx = 0

        cv2.imshow("Recall preview", res.annotated)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key in (ord("g"), ord("w"), ord("m")):
            last_asked_item = {"g": "glasses", "w": "wallet", "m": "medication"}[chr(key)]
            pipe.ask(last_asked_item)
        elif key == ord("r"):
            pipe.reset()
        elif key == ord("c"):
            if last_asked_item:
                path = pipe.store.last_clip_path(last_asked_item)
                if path and os.path.exists(path):
                    frames, spf = _load_clip(path)
                    if frames:
                        clip_frames    = frames
                        clip_frame_idx = 0
                        clip_frame_spf = spf
                        clip_next_ts   = time.perf_counter()
                    else:
                        print(f"[clip] could not load {path}")
                else:
                    print(f"[clip] no clip found for {last_asked_item}")
            else:
                print("[clip] press g / w / m first to select an item")

    pipe.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
