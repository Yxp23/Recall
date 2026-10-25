"""
Medication dose detection: hand-to-mouth sequence monitor.
"""
from __future__ import annotations

import os
import threading
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from recall.events.store import EventStore

from recall.events.tracker import State

DOSE_WINDOW        = 4 * 3600  # prior dose within this triggers repeat_reach
MOUTH_DIST_THRESH  = 0.70      # hand point must be < this × shoulder_width from mouth (seated pill motions reach 0.6-0.65)
PICKUP_TIMEOUT_S   = 12.0      # seconds from pickup before giving up
PUT_DOWN_CONFIRM_S = 1.5       # "put down" = tracker says ON with no hand near it, for this long (a bottle held still is also "ON")
MOUTH_DWELL_S      = 0.4       # seconds hand must stay near mouth
MOUTH_GAP_S        = 0.4       # a hand-tracking flicker this short doesn't restart the dwell
MOUTH_VISIBLE_WINDOW_S = 1.5   # mouth must have been visible this recently at the reach (a hand/bottle at the mouth hides it)
UNCLEAR_MIN_HELD_S = 3.0       # an unclear pickup held/out of view this long counts for repeat_reach
REST_WINDOW_S      = 3.0       # "resting": the medication was seen at one spot within this window…
REST_MIN_SEEN_S    = 1.5       # …for at least this long in total…
REST_MAX_GAP_S     = 0.7       # …bridging detection flicker up to this long
MIN_SINCE_REF_S    = 2.0       # a pickup is valid only this long after the last placed/ON event or pickup
DOSE_COOLDOWN_S    = 30.0      # no second dose_appears_taken this long after one (pickups still fire)
VOICE_DELAY_S      = 0.8       # voice starts this long after the chime starts
HAND_NEAR_PICKUP_S = 1.0       # a hand box must have overlapped the medication this recently
                                # before it vanished for the disappearance to count
PICKUP_MOVE_PX     = 60        # IN_HAND pickup: box center must move this far from its resting spot
MISSING_CONFIRM_S  = 1.5       # disappeared-with-hand pickup: still missing after this long
OCCLUDED_MAX_S     = 2.0       # reappearing at the resting spot within this = it was only occluded
REST_RADIUS_PX     = 40        # "the same spot" / "at its resting spot"

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


def _center(box: tuple[int, int, int, int]) -> tuple[float, float]:
    return ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)


def _dist(a: tuple[float, float], b: tuple[float, float]) -> float:
    return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5


def _overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    """Boxes actually intersect (a hand merely nearby doesn't count)."""
    return a[0] < b[2] and a[2] > b[0] and a[1] < b[3] and a[3] > b[1]


def _stamp() -> str:
    return datetime.now().strftime("%H:%M:%S.%f")[:-3]


def play_repeat_alert() -> None:
    """Non-blocking: chime then voice (m4a if present, else TTS).

    Runs entirely in its own thread so it never waits on clips or the DB.
    """
    print(f"[dose] {_stamp()} repeat_reach fired → starting audio")

    def _run() -> None:
        import subprocess
        print(f"[dose] {_stamp()} afplay chime start")
        chime = subprocess.Popen(["afplay", _CHIME_PATH])
        time.sleep(VOICE_DELAY_S)   # voice overlaps the tail of the chime
        if os.path.exists(_VOICE_PATH):
            print(f"[dose] {_stamp()} afplay voice start")
            voice = subprocess.Popen(["afplay", _VOICE_PATH])
        else:
            print(f"[dose] {_stamp()} say start")
            voice = subprocess.Popen(["say", BANNER_TEXT])
        chime.wait()
        voice.wait()
    threading.Thread(target=_run, daemon=True).start()


class DoseMonitor:
    def __init__(self, store: "EventStore") -> None:
        self._store = store
        self._state: str = _IDLE
        self._pickup_ts: float = 0.0
        self._pickup_reason: str = ""
        self._mouth_near_since: Optional[float] = None
        self._mouth_far_since: Optional[float] = None
        self._last_mouth_vis_ts: float = float("-inf")  # last frame the mouth was visible
        self._mouth_visible_now: bool = False

        # per-AWAITING-window stats for diagnostics
        self._w_frames = 0
        self._w_mouth_vis = 0
        self._w_hand_pts = 0
        self._best_ratio: Optional[float] = None
        self._is_repeat: bool = False        # current pickup raised repeat_reach
        self._last_ref_ts: Optional[float] = None  # last placed/ON event or valid pickup
        # recent detections: (ts, center, a hand box overlapped the bottle on that frame)
        self._pos_hist: deque[tuple[float, tuple[float, float], bool]] = deque()
        self._cooldown_until: float = 0.0

        # pickup confirmation: resting position, in-hand episode, disappeared-with-hand candidate
        self._prev_med_state: State = State.UNKNOWN
        self._last_med_detected_ts: float = float("-inf")  # last frame the medication was detected
        self._last_det_state: State = State.UNKNOWN        # tracker state on that frame
        self._hand_overlap_ts: float = float("-inf")       # last frame a hand box overlapped it
        self._rest_center: Optional[tuple[float, float]] = None
        self._hand_ep: Optional[dict] = None   # {"max": px moved so far, "dead": gate failed}
        self._miss_checked = False             # evaluated this disappearance already
        self._miss_since: Optional[float] = None  # ts it was last seen, while a pickup is pending
        self._put_down_since: Optional[float] = None   # when the bottle last looked set down (no hand near, ON)
        self._pickup_active = False            # a pickup was confirmed and the bottle hasn't been set down since
        self._confirm_ts = float("-inf")       # when that pickup was confirmed
        self._lock_logged = False              # already printed "waiting for set-down" for this disappearance

        # debug / display
        self._last_event: str = ""
        self._closest_ratio: Optional[float] = None

        self.audio_enabled = True   # False = alerts still emit events, but no sound on this Mac
        self.frame_size: tuple[int, int] = (0, 0)   # set by the pipeline, for the debug line
        self._dbg: tuple = (None, None, None, 0)    # mouth, nearest hand point, scale, points

        self._doses: list[float] = []
        self._unclears: list[float] = []   # unclear pickups held ≥ UNCLEAR_MIN_HELD_S (repeat window)
        self._unclear_count = 0            # all dose_unclear events since reset (panel)
        self._load_today_doses()
        times = ", ".join(datetime.fromtimestamp(t).strftime("%-I:%M %p") for t in self._doses)
        print(f"[dose] prior doses today: {len(self._doses)}" + (f" ({times})" if times else ""))

    # ── public ─────────────────────────────────────────────────────────────────

    def update(
        self,
        med_state: State,
        med_events: list[str],
        med_detected: bool,
        med_hand_near: bool,
        med_box: Optional[tuple[int, int, int, int]],
        hand_boxes: list[tuple[int, int, int, int]],
        wrists_and_tips: list[tuple[int, int]],
        mouth_center: Optional[tuple[int, int]],
        mouth_visible: bool,
        shoulder_width: Optional[float],
        ts: float,
        ts_wall: float,
    ) -> list[DoseEvent]:
        events: list[DoseEvent] = []

        prev_state = self._prev_med_state
        self._prev_med_state = med_state

        # ── medication position + hand contact ────────────────────────────────
        center = _center(med_box) if (med_detected and med_box) else None
        hand_on = bool(med_detected and med_box and any(_overlap(med_box, hb) for hb in hand_boxes))
        if med_detected:
            self._last_med_detected_ts = ts
            self._last_det_state = med_state
            if hand_on:
                self._hand_overlap_ts = ts
        if center:
            self._pos_hist.append((ts, center, hand_on))
        while self._pos_hist and ts - self._pos_hist[0][0] > 2 * REST_WINDOW_S:
            self._pos_hist.popleft()

        # ── tracker events are the source of truth for pickups ────────────────
        for name in med_events:
            if name in ("placed", "reappeared", "taken_off"):
                self._last_ref_ts = ts

        self._mouth_visible_now = mouth_visible
        if mouth_visible:
            self._last_mouth_vis_ts = ts

        # ── update closest hand-to-mouth ratio for debug overlay ─────────────
        self._closest_ratio = self._compute_closest_ratio(
            wrists_and_tips, mouth_center, shoulder_width
        )

        # ── IDLE: look for a CONFIRMED pickup ─────────────────────────────────
        # ── after a confirmed pickup, wait for the bottle to be set down ──────
        if self._pickup_active:
            placed = med_state == State.ON or any(
                n in ("placed", "reappeared", "taken_off") for n in med_events)
            # a bottle held still in a hand is not set down: only hand-free frames count as resting
            seen, _ = self._rest_check(ts, since=self._confirm_ts, hand_free=True)
            if placed:
                self._pickup_active = False
                print("[dose] set-down detected (tracker: placed/ON); ready for the next pickup")
            elif seen >= REST_MIN_SEEN_S:
                self._pickup_active = False
                print(f"[dose] set-down detected (resting {seen:.1f}s at one spot, no hand on it); ready for the next pickup")
            else:
                reason = (f"no set-down since the last pickup: tracker not ON, resting "
                          f"{seen:.1f}s of {REST_MIN_SEEN_S:.1f}s without a hand on it")
                if med_state == State.IN_HAND and prev_state != State.IN_HAND:
                    print(f"[dose] ignored: waiting for set-down ({reason}; picked up again)")
                if med_detected:
                    self._lock_logged = False
                elif not self._lock_logged and self._last_med_detected_ts - self._hand_overlap_ts <= HAND_NEAR_PICKUP_S:
                    self._lock_logged = True
                    print(f"[dose] ignored: waiting for set-down ({reason}; vanished with a hand)")

        if self._state == _IDLE and not self._pickup_active:
            # (a) detected IN_HAND and its center moved > PICKUP_MOVE_PX from the resting spot
            if med_state == State.IN_HAND and prev_state != State.IN_HAND:
                if self._miss_since is not None:
                    # this pickup is already pending (it vanished with a hand): resting was checked then
                    self._hand_ep = {"max": 0.0, "dead": False}
                else:
                    gate = self._pickup_gate(ts)
                    self._hand_ep = {"max": 0.0, "dead": gate is not None}
                    if gate:
                        print(f"[dose] ignored: in hand but {gate}")
            if self._hand_ep is not None:
                if med_state == State.IN_HAND:
                    if center and self._rest_center:
                        d = _dist(center, self._rest_center)
                        self._hand_ep["max"] = max(self._hand_ep["max"], d)
                        if not self._hand_ep["dead"] and d > PICKUP_MOVE_PX:
                            self._confirm_pickup(f"moved {d:.0f}px", ts, ts_wall, events)
                else:
                    ep, self._hand_ep = self._hand_ep, None
                    if not ep["dead"]:
                        print(f"[dose] ignored: in hand but moved only {ep['max']:.0f}px")

            # (b) vanished with a hand box on it and still missing after MISSING_CONFIRM_S
            if self._state == _IDLE:
                if not med_detected:
                    if not self._miss_checked and self._miss_since is None:
                        self._miss_checked = True
                        hand_on_it = self._last_med_detected_ts - self._hand_overlap_ts <= HAND_NEAR_PICKUP_S
                        if self._last_det_state != State.IN_HAND and hand_on_it:
                            gate = self._pickup_gate(ts)
                            if gate:
                                print(f"[dose] ignored: vanished with hand but {gate}")
                            else:
                                self._miss_since = self._last_med_detected_ts
                    if self._miss_since is not None and ts - self._miss_since >= MISSING_CONFIRM_S:
                        self._confirm_pickup("missing 1.5s with hand", ts, ts_wall, events)
                else:
                    self._miss_checked = False
                    if self._miss_since is not None and center and self._rest_center:
                        d = _dist(center, self._rest_center)
                        gone = ts - self._miss_since
                        if d > PICKUP_MOVE_PX:
                            # it came back far from where it rested: it was moved = a real pickup
                            self._confirm_pickup(f"reappeared {d:.0f}px away", ts, ts_wall, events)
                        elif d <= REST_RADIUS_PX:
                            self._miss_since = None
                            if gone <= OCCLUDED_MAX_S:
                                print("[dose] ignored: occluded, reappeared at rest spot")
                            else:
                                print(f"[dose] ignored: reappeared at rest spot after {gone:.1f}s")
                        else:
                            self._miss_since = ts   # 40-60 px: undecided, keep waiting

        # ── AWAITING_MOUTH: track face + proximity ────────────────────────────
        elif self._state == _AWAITING_MOUTH:
            m, nh, sc, npts = self._dbg
            print(f"[dose] frame={self.frame_size[0]}x{self.frame_size[1]} mouth={m} nearest_hand={nh} "
                  f"shoulder={'%.0f' % sc if sc else 'none'}px ratio="
                  f"{'%.2f' % self._closest_ratio if self._closest_ratio is not None else 'none'} "
                  f"(need < {MOUTH_DIST_THRESH:.2f}) points={npts} mouth_visible={mouth_visible}")
            self._hand_ep = None
            self._miss_since = None
            self._w_frames += 1
            self._w_mouth_vis += int(mouth_visible)
            self._w_hand_pts += int(bool(wrists_and_tips))
            if self._closest_ratio is not None and (
                    self._best_ratio is None or self._closest_ratio < self._best_ratio):
                self._best_ratio = self._closest_ratio

            # timeout
            if ts - self._pickup_ts > PICKUP_TIMEOUT_S:
                self._emit_unclear("timeout", save_clip=False,
                                   events=events, ts=ts, ts_wall=ts_wall)
                return events

            # put back down before reaching the mouth. Not just "tracker says ON": a bottle held still in a
            # hand also becomes ON, and that ended 6 of 7 recent attempts. It must be ON with no hand near it
            # for PUT_DOWN_CONFIRM_S.
            if med_state == State.ON and not med_hand_near:
                if self._put_down_since is None:
                    self._put_down_since = ts
                elif ts - self._put_down_since >= PUT_DOWN_CONFIRM_S:
                    self._emit_unclear("put down before mouth", save_clip=False,
                                       events=events, ts=ts, ts_wall=ts_wall)
                    return events
            else:
                self._put_down_since = None

            # check hand-to-mouth proximity
            if self._closest_ratio is not None and self._closest_ratio < MOUTH_DIST_THRESH:
                self._mouth_far_since = None
                if self._mouth_near_since is None:
                    self._mouth_near_since = ts
                elif ts - self._mouth_near_since >= MOUTH_DWELL_S:
                    self._mouth_near_since = None
                    mouth_ok = ts - self._last_mouth_vis_ts <= MOUTH_VISIBLE_WINDOW_S
                    if mouth_ok and (self._is_repeat or ts < self._cooldown_until):
                        self._emit_second_dose(events=events, ts_wall=ts_wall)
                    elif mouth_ok:
                        self._emit_taken(events=events, ts=ts, ts_wall=ts_wall)
                    else:
                        self._emit_unclear("mouth not visible at reach", save_clip=True,
                                           events=events, ts=ts, ts_wall=ts_wall)
            elif self._mouth_near_since is not None:
                # hand points flicker while moving near the face: tolerate short gaps
                if self._mouth_far_since is None:
                    self._mouth_far_since = ts
                elif ts - self._mouth_far_since > MOUTH_GAP_S:
                    self._mouth_near_since = None
                    self._mouth_far_since = None

        return events

    @property
    def doses_today(self) -> int:
        return len(self._doses)

    @property
    def dose_times(self) -> list[float]:
        return list(self._doses)

    @property
    def last_dose_ts(self) -> Optional[float]:
        return self._doses[-1] if self._doses else None

    @property
    def unclear_count(self) -> int:
        return self._unclear_count

    def record_dose(self, ts_wall: float) -> None:
        self._doses.append(ts_wall)

    def reset_doses(self) -> None:
        """Clear in-memory doses now; write the doses_reset marker off-thread."""
        self._doses = []
        self._unclears = []
        self._unclear_count = 0
        ts = time.time()

        def _write() -> None:
            self._store.log("medication", "doses_reset", detail="manual (r)", ts=ts)
            print(f"[dose] {_stamp()} reset complete (doses_reset logged)")
        threading.Thread(target=_write, daemon=True).start()

    def panel_line(self) -> str:
        n = len(self._doses)
        unclear = f" · Unclear: {self._unclear_count}"
        if n == 0:
            return "Doses today: none yet" + unclear
        t_str = datetime.fromtimestamp(self._doses[-1]).strftime("%-I:%M %p")
        return f"Doses today: {n} · last {t_str}" + unclear

    def debug_state(self, ts: float) -> dict:
        seconds_left = 0.0
        if self._state == _AWAITING_MOUTH:
            seconds_left = max(0.0, PICKUP_TIMEOUT_S - (ts - self._pickup_ts))
        since_ref = None if self._last_ref_ts is None else ts - self._last_ref_ts
        return {
            "state":          self._state,
            "since_ref":      since_ref,
            "seconds_left":   seconds_left,
            "mouth_visible":  self._mouth_visible_now,
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
        best = f"{self._best_ratio:.2f}" if self._best_ratio is not None else "n/a"
        print(f"[dose] AWAITING_MOUTH → IDLE (dose_appears_taken, best ratio {best})")
        self._state = _IDLE

    def _emit_unclear(self, detail: str, save_clip: bool,
                      events: list[DoseEvent], ts: float, ts_wall: float) -> None:
        n = max(self._w_frames, 1)
        best = f"{self._best_ratio:.2f}" if self._best_ratio is not None else "n/a"
        print(f"[dose] unclear: best ratio {best} (threshold {MOUTH_DIST_THRESH:.2f}), "
              f"mouth visible {100 * self._w_mouth_vis // n}% of frames, "
              f"hand points seen {100 * self._w_hand_pts // n}% of frames")
        self._unclear_count += 1
        if ts - self._pickup_ts >= UNCLEAR_MIN_HELD_S:
            self._unclears.append(ts_wall)
            detail += " [held>=3s]"
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
        # only count doses after the most recent doses_reset
        row = self._store._conn.execute(
            "SELECT MAX(ts) FROM events"
            " WHERE item='medication' AND event='doses_reset' AND ts >= ?",
            (today_ts,),
        ).fetchone()
        since = max(today_ts, row[0]) if row and row[0] is not None else today_ts
        rows = self._store._conn.execute(
            "SELECT ts FROM events"
            " WHERE item='medication' AND event='dose_appears_taken' AND ts > ?"
            " ORDER BY ts",
            (since,),
        ).fetchall()
        self._doses = [row[0] for row in rows]

        urows = self._store._conn.execute(
            "SELECT ts, detail FROM events"
            " WHERE item='medication' AND event='dose_unclear' AND ts > ?"
            " ORDER BY ts",
            (since,),
        ).fetchall()
        self._unclear_count = len(urows)
        self._unclears = [r[0] for r in urows if "[held>=3s]" in (r[1] or "")]

    def _rest_check(self, ts: float, since: Optional[float] = None,
                    hand_free: bool = False) -> tuple[float, Optional[tuple[float, float]]]:
        """Longest time the medication was seen at ONE spot (within REST_RADIUS_PX) in the last
        REST_WINDOW_S, bridging detection gaps up to REST_MAX_GAP_S. Returns (seconds, spot)."""
        pts = [(t, c) for t, c, h in self._pos_hist
               if ts - t <= REST_WINDOW_S and (since is None or t >= since) and not (hand_free and h)]
        best_seen, best_spot = 0.0, None
        for _, anchor in pts:
            inside = [(t, c) for t, c in pts if _dist(c, anchor) <= REST_RADIUS_PX]
            seen = sum(t1 - t0 for (t0, _), (t1, _) in zip(inside, inside[1:])
                       if t1 - t0 <= REST_MAX_GAP_S)
            if best_spot is None or seen > best_seen:
                best_seen = seen
                best_spot = (sum(c[0] for _, c in inside) / len(inside),
                             sum(c[1] for _, c in inside) / len(inside))
        return best_seen, best_spot

    def _pickup_gate(self, ts: float) -> Optional[str]:
        """Why a pickup can't count right now, or None if it can. Always prints the resting check."""
        seen, spot = self._rest_check(ts)
        ok = spot is not None and seen >= REST_MIN_SEEN_S
        print(f"[dose] resting check: seen {seen:.1f}s of last {REST_WINDOW_S:.0f}s at rest spot "
              f"({'ok' if ok else f'need {REST_MIN_SEEN_S:.1f}s'})")
        if self._last_ref_ts is not None and ts - self._last_ref_ts < MIN_SINCE_REF_S:
            return f"only {ts - self._last_ref_ts:.1f}s since the last pickup/placed"
        if not ok:
            return f"medication was not resting (seen {seen:.1f}s of last {REST_WINDOW_S:.0f}s at one spot)"
        self._rest_center = spot   # used by the moved / occluded checks
        return None

    def _confirm_pickup(self, reason: str, ts: float, ts_wall: float,
                        events: list[DoseEvent]) -> None:
        """A pickup is confirmed: start waiting for the mouth, and raise repeat_reach if due."""
        self._hand_ep = None
        self._miss_since = None
        self._pickup_active = True   # until the bottle is set down (placed/ON, or resting 1.5s), pickups are ignored, loudly
        self._confirm_ts = ts
        self._lock_logged = False
        self._last_ref_ts = ts
        self._is_repeat = False
        self._state = _AWAITING_MOUTH
        self._pickup_ts = ts
        self._pickup_reason = reason
        self._mouth_near_since = None
        self._put_down_since = None
        self._w_frames = self._w_mouth_vis = self._w_hand_pts = 0
        self._best_ratio = None
        print(f"[dose] pickup confirmed ({reason})")
        print(f"[dose] IDLE → AWAITING_MOUTH")

        if self._prior_activity_in_window(ts_wall):
            self._is_repeat = True
            if self.audio_enabled:
                play_repeat_alert()   # first thing: audio must not wait on anything
            ev = DoseEvent(event="repeat_reach", ts_wall=ts_wall,
                           detail="prior dose within window", save_clip=True)
            events.append(ev)
            self._last_event = ev.event

    def _prior_activity_in_window(self, ts_wall: float) -> bool:
        """A dose, or an unclear pickup held ≥ UNCLEAR_MIN_HELD_S, inside DOSE_WINDOW."""
        return any(ts_wall - t < DOSE_WINDOW for t in self._doses + self._unclears)

    def _compute_closest_ratio(
        self,
        wrists_and_tips: list[tuple[int, int]],
        mouth_center: Optional[tuple[int, int]],
        shoulder_width: Optional[float],
    ) -> Optional[float]:
        self._dbg = (mouth_center, None, shoulder_width, len(wrists_and_tips))
        if not wrists_and_tips or mouth_center is None or not shoulder_width:
            return None
        mx, my = mouth_center
        nearest = min(wrists_and_tips, key=lambda p: (p[0] - mx) ** 2 + (p[1] - my) ** 2)
        min_dist = ((nearest[0] - mx) ** 2 + (nearest[1] - my) ** 2) ** 0.5
        self._dbg = (mouth_center, nearest, shoulder_width, len(wrists_and_tips))
        return min_dist / shoulder_width
