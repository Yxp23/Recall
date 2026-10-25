"""Small macOS audio helpers (afplay / say), always off the caller's thread."""
from __future__ import annotations

import hashlib
import os
import subprocess
import tempfile
import threading
import time
from pathlib import Path
from datetime import datetime
from typing import Callable, Optional

CHIME_PATH = "/System/Library/Sounds/Glass.aiff"
VOICE_DELAY_S = 0.8   # the voice starts this long after the chime starts


def _stamp() -> str:
    return datetime.now().strftime("%H:%M:%S.%f")[:-3]


def play_voice_file(path: str, label: str) -> bool:
    """Chime, then `path` 0.8 s later. No-op (False) if the file doesn't exist."""
    if not os.path.exists(path):
        return False
    print(f"[audio] {_stamp()} {label}: starting audio")

    def _run() -> None:
        chime = subprocess.Popen(["afplay", CHIME_PATH])
        time.sleep(VOICE_DELAY_S)
        voice = subprocess.Popen(["afplay", path])
        chime.wait()
        voice.wait()

    threading.Thread(target=_run, daemon=True).start()
    return True


def say(text: str, on_state: Optional[Callable[[bool], None]] = None,
        voice: str = "Samantha", rate: int = 170) -> None:
    """Speak with macOS `say` (natural female voice), non-blocking. on_state(True/False) brackets it."""
    def _run() -> None:
        if on_state:
            on_state(True)
        try:
            r = subprocess.run(["say", "-v", voice, "-r", str(rate), text], check=False)
            if r.returncode != 0:   # that voice isn't installed: use the system voice
                subprocess.run(["say", "-r", str(rate), text], check=False)
        except Exception as e:
            print(f"[audio] say failed: {e}")
        finally:
            if on_state:
                on_state(False)

    threading.Thread(target=_run, daemon=True).start()


class TTS:
    """
    Speak answers fast. macOS `say` needs 2-4 s just to synthesize a sentence when the Mac is busy,
    so the likely answers are rendered to audio files WHILE she is still asking (prerender), and
    when the answer is known it is just played (speak). Anything not rendered yet is rendered on demand.
    """

    VOICE = "Samantha"
    RATE = 170
    KEEP = 40   # audio files to keep around

    def __init__(self) -> None:
        self.dir = Path(tempfile.gettempdir()) / "recall_tts"
        self.dir.mkdir(exist_ok=True)
        self.cancel = False                    # set to stop speculative renders (CPU is needed elsewhere)
        self._lock = threading.Lock()
        self._rendering: dict[str, threading.Event] = {}

    def _path(self, text: str) -> Path:
        return self.dir / f"{hashlib.sha1(f'{self.VOICE}|{self.RATE}|{text}'.encode()).hexdigest()[:20]}.aiff"

    def render(self, text: str) -> Optional[Path]:
        """Audio file for `text` (rendered now if needed). Safe to call from several threads."""
        p = self._path(text)
        with self._lock:
            if p.exists() and p.stat().st_size > 0:
                return p
            pending = self._rendering.get(text)
            mine = pending is None
            if mine:
                pending = self._rendering[text] = threading.Event()
        if not mine:
            pending.wait(timeout=20)           # someone is already rendering this exact text
            return p if p.exists() else None
        try:
            t0 = time.time()
            r = subprocess.run(["say", "-v", self.VOICE, "-r", str(self.RATE), "-o", str(p), text], check=False)
            if r.returncode != 0 or not p.exists():   # voice not installed: system voice
                subprocess.run(["say", "-r", str(self.RATE), "-o", str(p), text], check=False)
            print(f"[audio] {_stamp()} rendered {len(text)} chars in {time.time() - t0:.1f}s")
            self._trim()
        finally:
            with self._lock:
                self._rendering.pop(text, None)
            pending.set()
        return p if p.exists() else None

    def _trim(self) -> None:
        files = sorted(self.dir.glob("*.aiff"), key=lambda f: f.stat().st_mtime, reverse=True)
        for f in files[self.KEEP:]:
            f.unlink(missing_ok=True)

    def prerender(self, texts: list[str]) -> None:
        """Render these in the background (two at a time) until told to stop (self.cancel)."""
        self.cancel = False

        def _one(t: str) -> None:
            if not self.cancel:
                self.render(t)

        def _run() -> None:
            # two at a time: the answer she needs is one of these, so don't make her wait for all of them
            from concurrent.futures import ThreadPoolExecutor
            with ThreadPoolExecutor(max_workers=2) as pool:
                list(pool.map(_one, texts))

        threading.Thread(target=_run, name="tts-prerender", daemon=True).start()

    def speak(self, text: str, on_state: Optional[Callable[[bool], None]] = None) -> None:
        """Play `text` as soon as its audio exists. on_state(True/False) brackets the speaking."""
        def _run() -> None:
            if on_state:
                on_state(True)
            try:
                p = self.render(text)
                print(f"[audio] {_stamp()} speaking ({'cached' if p else 'direct'})")
                if p is not None:
                    subprocess.run(["afplay", str(p)], check=False)
                else:
                    say(text)
            except Exception as e:
                print(f"[audio] speak failed: {e}")
            finally:
                if on_state:
                    on_state(False)

        threading.Thread(target=_run, name="tts-speak", daemon=True).start()
