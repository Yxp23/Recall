"""
Per-item state machine with debounce and disappearance handling.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class State(Enum):
    UNKNOWN          = "unknown"
    ON               = "on"               # visible in a zone, center stable ≥1 s
    IN_HAND          = "in_hand"
    WORN             = "worn"             # glasses only
    LAST_SEEN_INTO   = "last_seen_into"   # disappeared while in a zone
    LAST_SEEN_WORN   = "last_seen_worn"   # disappeared while being worn
    LAST_SEEN_IN_HAND = "last_seen_in_hand"  # disappeared while held, no zone


@dataclass
class Observation:
    detected:   bool
    box:        Optional[tuple[int, int, int, int]]  # x1 y1 x2 y2
    zone:       Optional[str]
    held:       bool
    worn:       bool
    hand_near:  bool   # any hand landmark bbox within 80 px of item box
    confidence: float
    ts:         float = field(default_factory=time.perf_counter)


@dataclass
class StateEvent:
    item:       str
    event:      str   # picked_up | placed | worn | taken_off | went_into |
                      # went_worn | went_in_hand | reappeared
    zone:       Optional[str]
    detail:     str
    confidence: float
    ts_wall:    float  # time.time() for DB storage


class ItemTracker:
    DEBOUNCE_S          = 1.0  # candidate must be stable this long before committing
    REAPPEAR_DEBOUNCE_S = 0.5  # shorter debounce for reappear transitions
    STABLE_PX           = 15  # max drift (px) over STABLE_S to count as placed
    STABLE_S            = 1.0
    WORN_STABLE_S       = 1.5  # must be stationary in zone this long to leave WORN
    DISAPPEAR_S         = 3.0  # missing this long → gone
    HAND_NEAR_S         = 2.0  # hand must have been within 80 px this recently to trust disappearance

    _GONE_STATES = (State.LAST_SEEN_INTO, State.LAST_SEEN_WORN,
                    State.LAST_SEEN_IN_HAND, State.UNKNOWN)

    def __init__(self, item: str) -> None:
        self.item        = item
        self.state       = State.UNKNOWN
        self.zone: Optional[str] = None
        # wall-clock time of last state change, for human-readable answers
        self.state_since_wall: float = time.time()

        # debounce
        self._cand:       Optional[State] = None
        self._cand_zone:  Optional[str]   = None
        self._cand_since: float           = 0.0

        # center history for stability check: [(perf_ts, cx, cy), ...]
        self._centers: list[tuple[float, int, int]] = []

        # last-seen tracking
        self._last_ts:   float         = 0.0   # perf_counter of last detected frame
        self._last_zone: Optional[str] = None
        self._last_conf: float         = 0.0
        self._gone_emitted: bool       = False  # True once a gone-event fired

        # hand-proximity history (for "objects don't move by themselves" rule)
        self._last_hand_near_ts: float = 0.0   # perf_counter of last frame with hand within 80 px

        # deduplication: skip event if (event, zone) matches last emitted
        self._last_emitted: Optional[tuple[str, Optional[str]]] = None

    @property
    def is_tracking(self) -> bool:
        """True when the item has been seen (not UNKNOWN); used for hysteresis thresholds."""
        return self.state != State.UNKNOWN

    # ── public ────────────────────────────────────────────────────────────────

    def update(self, obs: Observation) -> list[StateEvent]:
        if obs.detected:
            return self._detected(obs)
        return self._missing(obs.ts)

    # ── detected ──────────────────────────────────────────────────────────────

    def _detected(self, obs: Observation) -> list[StateEvent]:
        now = obs.ts
        self._last_ts      = now
        self._gone_emitted = False
        if obs.confidence:
            self._last_conf = obs.confidence
        if obs.zone:
            self._last_zone = obs.zone
        if obs.hand_near:
            self._last_hand_near_ts = now

        if obs.box:
            # use bottom-center: x mid, y2 — tracks where the item rests on a surface
            bx = (obs.box[0] + obs.box[2]) // 2
            by = obs.box[3]
            self._centers.append((now, bx, by))
        # keep 2 s of history
        self._centers = [(t, x, y) for t, x, y in self._centers if now - t <= 2.0]

        # derive candidate
        if obs.worn:
            cand, cand_zone = State.WORN, None
        elif self.state == State.WORN:
            # sticky WORN: only leave when stationary in a zone for WORN_STABLE_S
            if obs.zone and self._is_stable(now, self.WORN_STABLE_S):
                cand, cand_zone = State.ON, obs.zone
            else:
                cand, cand_zone = State.WORN, None  # still moving / no zone — stay WORN
        elif obs.held:
            cand, cand_zone = State.IN_HAND, obs.zone
        elif obs.zone and self._is_stable(now):
            cand, cand_zone = State.ON, obs.zone
        elif self.state in self._GONE_STATES:
            # newly visible again — move to ON optimistically (short debounce)
            cand, cand_zone = State.ON, obs.zone
        else:
            # hold current state while moving / no zone
            cand, cand_zone = self.state, self.zone

        return self._debounce(cand, cand_zone, obs.confidence, now)

    # ── missing ───────────────────────────────────────────────────────────────

    def _missing(self, now: float) -> list[StateEvent]:
        if self._gone_emitted or self._last_ts == 0.0:
            return []
        if now - self._last_ts < self.DISAPPEAR_S:
            return []

        # "Objects don't move by themselves": if was ON and no hand was near
        # within HAND_NEAR_S seconds before it disappeared, assume occlusion — keep ON.
        if self.state == State.ON:
            hand_was_near = (self._last_hand_near_ts > 0 and
                             self._last_ts - self._last_hand_near_ts < self.HAND_NEAR_S)
            if not hand_was_near:
                return []   # stay ON silently; re-check every frame

        self._gone_emitted = True

        if self.state == State.WORN:
            new_state, new_zone, event_name = State.LAST_SEEN_WORN, None, "went_worn"
        elif self._last_zone:
            new_state, new_zone, event_name = (State.LAST_SEEN_INTO,
                                               self._last_zone, "went_into")
        elif self.state == State.IN_HAND:
            new_state, new_zone, event_name = (State.LAST_SEEN_IN_HAND,
                                               None, "went_in_hand")
        else:
            new_state, new_zone, event_name = State.UNKNOWN, None, None

        events = []
        if new_state != self.state or new_zone != self.zone:
            self._commit(new_state, new_zone)
            self._cand = None
            if event_name:
                ev = self._make_event(event_name, new_zone, self._last_conf)
                if ev:
                    events.append(ev)
        return events

    # ── helpers ───────────────────────────────────────────────────────────────

    def _debounce(self, cand: State, cand_zone: Optional[str],
                  conf: float, now: float) -> list[StateEvent]:
        if cand != self._cand or cand_zone != self._cand_zone:
            self._cand       = cand
            self._cand_zone  = cand_zone
            self._cand_since = now
            return []

        # use shorter debounce for reappear transitions
        is_reappear = (cand == State.ON and self.state in self._GONE_STATES)
        required = self.REAPPEAR_DEBOUNCE_S if is_reappear else self.DEBOUNCE_S
        if now - self._cand_since < required:
            return []
        if cand == self.state and cand_zone == self.zone:
            return []

        event_name = self._transition_event(self.state, cand)
        self._commit(cand, cand_zone)
        self._cand = None
        if not event_name:
            return []
        ev = self._make_event(event_name, cand_zone, conf)
        return [ev] if ev else []

    def _commit(self, state: State, zone: Optional[str]) -> None:
        self.state            = state
        self.zone             = zone
        self.state_since_wall = time.time()

    def _make_event(self, event: str, zone: Optional[str],
                    conf: float) -> Optional[StateEvent]:
        key = (event, zone)
        if key == self._last_emitted:
            return None   # duplicate
        self._last_emitted = key
        return StateEvent(
            item=self.item, event=event, zone=zone,
            detail="", confidence=conf, ts_wall=time.time(),
        )

    def _is_stable(self, now: float, stable_s: Optional[float] = None) -> bool:
        s = stable_s if stable_s is not None else self.STABLE_S
        if not any(now - t >= s for t, _, _ in self._centers):
            return False
        recent = [(x, y) for t, x, y in self._centers if now - t <= s]
        if not recent:
            return False
        cx0, cy0 = self._centers[-1][1], self._centers[-1][2]
        return all(((x - cx0) ** 2 + (y - cy0) ** 2) ** 0.5 <= self.STABLE_PX
                   for x, y in recent)

    @staticmethod
    def _transition_event(old: State, new: State) -> Optional[str]:
        if new == State.IN_HAND:
            return "picked_up"
        if new == State.WORN:
            return "worn"
        if new == State.ON:
            if old == State.IN_HAND:
                return "placed"
            if old == State.WORN:
                return "taken_off"
            if old in ItemTracker._GONE_STATES:
                return "reappeared"
        if new == State.LAST_SEEN_INTO:
            return "went_into"
        if new == State.LAST_SEEN_WORN:
            return "went_worn"
        if new == State.LAST_SEEN_IN_HAND:
            return "went_in_hand"
        return None
