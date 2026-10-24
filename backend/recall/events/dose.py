"""
Medication dose detection: hand-to-mouth sequence monitor.
"""
from __future__ import annotations

import os
import threading
from dataclasses import dataclass
from datetime import datetime
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from recall.events.store import EventStore

from recall.events.tracker import State

DOSE_WINDOW        = 4 * 3600  # prior dose within this triggers repeat_reach
MOUTH_DIST_THRESH  = 0.35      # wrist/tip must be < this × shoulder_width from mouth
PICKUP_TIMEOUT_S   = 10.0      # seconds from pickup before giving up
MOUTH_DWELL_S      = 0.4       # seconds hand must stay near mouth
MIN_SINCE_REF_S    = 2.0       # a pickup is valid only this long after the last placed/ON event or pickup
DOSE_COOLDOWN_S    = 30.0      # no second dose_appears_taken this long after one (pickups still fire)
HAND_NEAR_PICKUP_S = 1.0       # hand must have been within 80 px this recently to treat
                                # a disappearance as a pickup

BANNER_TEXT = "You may have already taken this one. Please check with Sarah."
BANNER_SECOND_TEXT = "This may be a second dose. Sarah has been notified."
_VOICE_PATH = "data/voice/double_dose.m4a"
_CHIME_PATH = "/System/Library/Sounds/Glass.aiff"

_IDLE           = "idle"
_AWAITING_MOUTH = "awaiting_mouth"


@dataclass
class DoseEvent:
    event:     str    # dose_appears_taken | dose_unclear | repeat_reach | possible_second_dose
    ts_wall:   float
    detail:    str
    save_clip: bool


def play_repeat_alert() -> None:
    """Non-blocking: chime then voice (m4a if present, else TTS)."""
    def _run() -> None:
        import subprocess
        subprocess.run(["afplay", _CHIME_PATH], check=False)
        if os.path.exists(_VOICE_PATH):
            subprocess.run(["afplay", _VOICE_PATH], check=False)
        else:
            subprocess.run(["say", BANNER_TEXT], check=False)
    threading.Thread(target=_run, daemon=True).start()


class DoseMonitor:
    def __init__(self, store: "EventStore") -> None:
        self._store = store
        self._state: str = _IDLE
        self._pickup_ts: float = 0.0
        self._pickup_reason: str = ""
        self._mouth_near_since: Optional[float] = None
        self._mouth_visible_ok: bool = True  # mouth visible throughout AWAITING_MOUTH
        self._is_repeat: bool = False        # current pickup raised repeat_reach
        self._last_ref_ts: Optional[float] = None  # last placed/ON event or valid pickup
        self._cooldown_until: float = 0.0

        # per-frame medication proximity tracking for "vanished near hand" pickup
        self._prev_med_state: State = State.UNKNOWN
        self._last_med_detected_ts: float = 0.0   # perf_counter of last detected frame
        self._last_med_hand_near_ts: float = 0.0  # perf_counter of last hand-near-med frame

        # debug / display
        self._last_event: str = ""
        self._closest_ratio: Optional[float] = None

        self._doses: list[float] = []
        self._load_today_doses()

    # ── public ─────────────────────────────────────────────────────────────────

    def update(
        self,
        med_state: State,
        med_events: list[str],
        med_detected: bool,
        med_hand_near: bool,
        wrists_and_tips: list[tuple[int, int]],
        mouth_center: Optional[tuple[int, int]],
        mouth_visible: bool,
        shoulder_width: Optional[float],
        ts: float,
        ts_wall: float,
    ) -> list[DoseEvent]:
        events: list[DoseEvent] = []

        # ── track medication proximity ────────────────────────────────────────
        if med_detected:
            self._last_med_detected_ts = ts
            if med_hand_near:
                self._last_med_hand_near_ts = ts

        # ── tracker events are the source of truth for pickups ────────────────
        for name in med_events:
            if name in ("placed", "reappeared", "taken_off"):
                self._last_ref_ts = ts

        # ── update closest hand-to-mouth ratio for debug overlay ─────────────
        self._closest_ratio = self._compute_closest_ratio(
            wrists_and_tips, mouth_center, shoulder_width
        )

        # ── IDLE: look for pickup ─────────────────────────────────────────────
        if self._state == _IDLE:
            pickup_reason: Optional[str] = None

            if "picked_up" in med_events:
                pickup_reason = "picked_up"

            elif (self._prev_med_state == State.ON
                  and med_state == State.LAST_SEEN_INTO
                  and self._last_med_hand_near_ts > 0
                  and self._last_med_detected_ts - self._last_med_hand_near_ts < HAND_NEAR_PICKUP_S):
                pickup_reason = "vanished near hand"

            valid = (self._last_ref_ts is None
                     or ts - self._last_ref_ts >= MIN_SINCE_REF_S)
            if pickup_reason is not None and valid:
                self._last_ref_ts = ts
                self._is_repeat = False
                prev_state_name = _IDLE.upper()
                self._state = _AWAITING_MOUTH
                self._pickup_ts = ts
                self._pickup_reason = pickup_reason
                self._mouth_near_since = None
                self._mouth_visible_ok = True
                print(f"[dose] {prev_state_name} → AWAITING_MOUTH"
                      f" (pickup: {pickup_reason})")

                if self._prior_dose_in_window(ts_wall):
                    self._is_repeat = True
                    ev = DoseEvent(
                        event="repeat_reach",
                        ts_wall=ts_wall,
                        detail="prior dose within window",
                        save_clip=True,
                    )
                    events.append(ev)
                    self._last_event = ev.event
                    play_repeat_alert()

        # ── AWAITING_MOUTH: track face + proximity ────────────────────────────
        elif self._state == _AWAITING_MOUTH:
            if not mouth_visible:
                self._mouth_visible_ok = False

            # timeout
            if ts - self._pickup_ts > PICKUP_TIMEOUT_S:
                self._emit_unclear("timeout", save_clip=False,
                                   events=events, ts_wall=ts_wall)
                return events

            # medication reappeared on surface — explicitly put back down
            if med_state == State.ON:
                self._emit_unclear("put down before mouth", save_clip=False,
                                   events=events, ts_wall=ts_wall)
                return events

            # check hand-to-mouth proximity
            if self._closest_ratio is not None and self._closest_ratio < MOUTH_DIST_THRESH:
                if self._mouth_near_since is None:
                    self._mouth_near_since = ts
                elif ts - self._mouth_near_since >= MOUTH_DWELL_S:
                    self._mouth_near_since = None
                    if self._mouth_visible_ok and (self._is_repeat or ts < self._cooldown_until):
                        self._emit_second_dose(events=events, ts_wall=ts_wall)
                    elif self._mouth_visible_ok:
                        self._emit_taken(events=events, ts=ts, ts_wall=ts_wall)
                    else:
                        self._emit_unclear("mouth not visible throughout", save_clip=True,
                                           events=events, ts_wall=ts_wall)
            else:
                self._mouth_near_since = None

        self._prev_med_state = med_state
        return events

    def record_dose(self, ts_wall: float) -> None:
        self._doses.append(ts_wall)

    def reset_doses(self) -> None:
        self._doses = []
        print("[dose] doses reset")

    def panel_line(self) -> str:
        n = len(self._doses)
        if n == 0:
            return "Doses today: none yet"
        t_str = datetime.fromtimestamp(self._doses[-1]).strftime("%-I:%M %p")
        return f"Doses today: {n} · last {t_str}"

    def debug_state(self, ts: float) -> dict:
        seconds_left = 0.0
        if self._state == _AWAITING_MOUTH:
            seconds_left = max(0.0, PICKUP_TIMEOUT_S - (ts - self._pickup_ts))
        since_ref = None if self._last_ref_ts is None else ts - self._last_ref_ts
        return {
            "state":          self._state,
            "since_ref":      since_ref,
            "seconds_left":   seconds_left,
            "mouth_visible":  self._mouth_visible_ok,
            "closest_ratio":  self._closest_ratio,
            "threshold":      MOUTH_DIST_THRESH,
            "last_event":     self._last_event,
        }

    # ── private ─────────────────────────────────────────────────────────────────

    def _emit_second_dose(self, events: list[DoseEvent], ts_wall: float) -> None:
        # Caregiver alert only: deliberately NOT recorded as a dose.
        ev = DoseEvent(event="possible_second_dose", ts_wall=ts_wall,
                       detail="hand-to-mouth after repeat_reach", save_clip=True)
        events.append(ev)
        self._last_event = ev.event
        print("[dose] AWAITING_MOUTH → IDLE (possible_second_dose)")
        self._state = _IDLE
        self._is_repeat = False

    def _emit_taken(self, events: list[DoseEvent], ts: float, ts_wall: float) -> None:
        self.record_dose(ts_wall)
        self._cooldown_until = ts + DOSE_COOLDOWN_S
        ev = DoseEvent(event="dose_appears_taken", ts_wall=ts_wall, detail="", save_clip=True)
        events.append(ev)
        self._last_event = ev.event
        print(f"[dose] AWAITING_MOUTH → IDLE (dose_appears_taken)")
        self._state = _IDLE

    def _emit_unclear(self, detail: str, save_clip: bool,
                      events: list[DoseEvent], ts_wall: float) -> None:
        ev = DoseEvent(event="dose_unclear", ts_wall=ts_wall,
                       detail=detail, save_clip=save_clip)
        events.append(ev)
        self._last_event = f"dose_unclear ({detail})"
        print(f"[dose] AWAITING_MOUTH → IDLE (dose_unclear: {detail})")
        self._state = _IDLE
        self._is_repeat = False
        self._mouth_near_since = None

    def _load_today_doses(self) -> None:
        today_ts = datetime.today().replace(
            hour=0, minute=0, second=0, microsecond=0
        ).timestamp()
        rows = self._store._conn.execute(
            "SELECT ts FROM events"
            " WHERE item='medication' AND event='dose_appears_taken' AND ts >= ?"
            " ORDER BY ts",
            (today_ts,),
        ).fetchall()
        self._doses = [row[0] for row in rows]

    def _prior_dose_in_window(self, ts_wall: float) -> bool:
        return any(ts_wall - d < DOSE_WINDOW for d in self._doses)

    def _compute_closest_ratio(
        self,
        wrists_and_tips: list[tuple[int, int]],
        mouth_center: Optional[tuple[int, int]],
        shoulder_width: Optional[float],
    ) -> Optional[float]:
        if not wrists_and_tips or mouth_center is None or not shoulder_width:
            return None
        mx, my = mouth_center
        min_dist = min(
            ((px - mx) ** 2 + (py - my) ** 2) ** 0.5
            for px, py in wrists_and_tips
        )
        return min_dist / shoulder_width
