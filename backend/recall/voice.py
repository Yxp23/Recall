"""
Voice: push-to-talk questions answered from the event log. No LLM — keywords and templates.

  browser audio → ffmpeg decode → faster-whisper "small" (cpu, int8) → intent → template answer

The model loads in a background thread at startup, so a missing model or a slow load never
blocks the server; until it is ready (or if it failed) /api/voice says so and the tap buttons
keep working.
"""
from __future__ import annotations

import os
import re
import subprocess
import threading
import time
from datetime import datetime
from typing import Optional, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from recall.pipeline import Pipeline

os.environ.setdefault("HF_HUB_OFFLINE", "1")

# Measured on this Mac with the kitchen + door pipelines running (load average ~11 on 8 cores):
# "small" took 7-9 s per question, base.en ~2.5 s, tiny.en ~1.5 s, all transcribing the sample
# questions correctly. The questions are short and keyword-matched, so base.en is plenty.
# RECALL_WHISPER=tiny.en (faster) or small (more accurate) overrides it.
WHISPER_MODEL = os.environ.get("RECALL_WHISPER", "base.en")
INITIAL_PROMPT = "Where are my glasses? Where is my wallet? Did I take my pills? Where are my pills?"
CAREGIVER = "Sarah"

NOT_UNDERSTOOD = "Sorry, I didn't catch that. You can ask where your glasses, wallet or pills are."
NOT_READY = "Voice is still starting. Please try again in a moment, or tap a button."

_GLASSES = {"glasses", "glass", "spectacles", "specs", "eyeglasses"}
_WALLET = {"wallet", "purse", "billfold"}
_PILLS = {"pill", "pills", "medicine", "medicines", "medication", "medications", "tablet", "tablets", "meds", "prescription"}
_DOSE_PHRASES = ("did i take", "did i already take", "have i taken", "have i already taken",
                 "did i have my", "did i take my", "had my")


class VoiceEngine:
    def __init__(self) -> None:
        self.ready = False
        self.error: Optional[str] = None
        self._model = None
        self._thread = threading.Thread(target=self._load, name="whisper-load", daemon=True)

    def start(self) -> None:
        self._thread.start()

    def status(self) -> dict:
        return {"ready": self.ready, "error": self.error}

    def _load(self) -> None:
        t0 = time.time()
        try:
            from faster_whisper import WhisperModel

            # offline: the model is already in the Hugging Face cache
            self._model = WhisperModel(WHISPER_MODEL, device="cpu", compute_type="int8",
                                       cpu_threads=4, local_files_only=True)
            print(f"[voice] loading whisper '{WHISPER_MODEL}'…")
            self._model.transcribe(np.zeros(16000, dtype=np.float32), language="en")  # warm up
            self.ready = True
            print(f"[voice] whisper '{WHISPER_MODEL}' ready in {time.time() - t0:.1f}s")
        except Exception as e:
            self.error = f"voice unavailable: {e}"
            print(f"WARNING: {self.error} (tap buttons still work)")

    # ── audio → text ───────────────────────────────────────────────────────────

    @staticmethod
    def decode(data: bytes) -> np.ndarray:
        """Browser audio (webm/opus, mp4…) → 16 kHz mono float32, using imageio-ffmpeg."""
        import imageio_ffmpeg

        out = subprocess.run(
            [imageio_ffmpeg.get_ffmpeg_exe(), "-v", "error", "-i", "pipe:0",
             "-f", "f32le", "-ac", "1", "-ar", "16000", "pipe:1"],
            input=data, capture_output=True, timeout=15, check=True,
        ).stdout
        return np.frombuffer(out, dtype=np.float32)

    def transcribe(self, audio: np.ndarray) -> str:
        if self._model is None or audio.size < 1600:   # under 0.1 s
            return ""
        segments, _ = self._model.transcribe(
            audio, language="en", vad_filter=True, initial_prompt=INITIAL_PROMPT,
            beam_size=1, condition_on_previous_text=False,
            without_timestamps=True,   # we only need the words: ~10% faster
        )
        return " ".join(s.text.strip() for s in segments).strip()


# ── intent (keywords, no LLM) ────────────────────────────────────────────────

def parse_intent(text: str) -> tuple[Optional[str], Optional[str]]:
    """→ ("item", "glasses"|"wallet"|"medication") | ("dose", None) | (None, None)."""
    t = re.sub(r"[^a-z' ]", " ", text.lower())
    words = set(t.replace("'", "").split())
    has_pills = bool(words & _PILLS)
    has_glasses = bool(words & _GLASSES)
    has_wallet = bool(words & _WALLET)
    dose_phrase = any(p in t for p in _DOSE_PHRASES)
    if dose_phrase and (has_pills or not (has_glasses or has_wallet)):
        return "dose", None
    if has_glasses:
        return "item", "glasses"
    if has_wallet:
        return "item", "wallet"
    if has_pills:
        return "item", "medication"
    return None, None


# ── answers (templates over the live state + event log) ──────────────────────

def _clock(ts: float) -> str:
    return datetime.fromtimestamp(ts).strftime("%-I:%M %p")


def candidate_answers(pipe: "Pipeline") -> list[str]:
    """Every answer she could be about to hear, so its audio can be rendered while she is still asking."""
    out: list[str] = []
    for q in ("where are my glasses", "where are my pills", "did i take my pills", "where is my wallet"):
        a, _ = answer_question(pipe, q)
        if a not in out:
            out.append(a)
    return out


def answer_question(pipe: "Pipeline", text: str) -> tuple[str, Optional[str]]:
    """→ (answer, clip_path or None). Never tells her to take or skip a dose."""
    from recall.events.answers import where_is

    kind, item = parse_intent(text)
    if kind == "item" and item:
        return where_is(pipe.trackers[item]), pipe.store.last_clip_path(item)
    if kind == "dose":
        times = pipe.monitor.dose_times
        if times:
            row = pipe.store.last_event("medication", "dose_appears_taken")
            return (f"It looks like you took them at {_clock(times[-1])}. "
                    f"If you're not sure, check with {CAREGIVER}."), (row["clip_path"] if row else None)
        if pipe.monitor.unclear_count:
            return f"I'm not sure. Please check with {CAREGIVER}.", None
        return f"I haven't seen you take them today. Please check with {CAREGIVER}.", None
    return NOT_UNDERSTOOD, None
