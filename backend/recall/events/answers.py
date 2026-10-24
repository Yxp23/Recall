"""
Template answers for where_is() queries.
"""
from __future__ import annotations

import datetime
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from recall.events.tracker import ItemTracker

_NAMES = {
    "glasses":    "your glasses",
    "wallet":     "your wallet",
    "medication": "your medication",
}

_STATE_LABELS = {
    "on":                "ON",
    "in_hand":           "IN HAND",
    "worn":              "WORN",
    "last_seen_into":    "LAST SEEN",
    "last_seen_worn":    "LAST SEEN WORN",
    "last_seen_in_hand": "LAST SEEN HAND",
    "unknown":           "UNKNOWN",
}


def where_is(tracker: "ItemTracker") -> str:
    from recall.events.tracker import State

    name  = _NAMES.get(tracker.item, tracker.item)
    state = tracker.state
    zone  = tracker.zone
    when  = _fmt_time(tracker.state_since_wall)

    if state == State.ON:
        loc = f"on the {zone}" if zone else "nearby"
        return f"{name.capitalize()} is {loc}. I last saw it placed there at {when}."

    if state == State.WORN:
        return "You're wearing them."

    if state == State.IN_HAND:
        return f"You're holding {name} right now."

    if state == State.LAST_SEEN_INTO:
        loc = f"into the {zone}" if zone else "out of view"
        return f"I last saw {name} going {loc} at {when}."

    if state == State.LAST_SEEN_WORN:
        return (f"You were wearing them at {when}, "
                "so check your face or your shirt pocket.")

    if state == State.LAST_SEEN_IN_HAND:
        return f"I last saw it in your hand at {when}."

    # UNKNOWN
    return f"I'm not sure. I haven't seen {name} yet today."


def state_label(tracker: "ItemTracker") -> str:
    """Short label for the status panel (e.g. 'ON · counter')."""
    label = _STATE_LABELS.get(tracker.state.value, tracker.state.value)
    if tracker.zone and tracker.state.value == "on":
        label += f" · {tracker.zone}"
    elif tracker.zone and tracker.state.value == "last_seen_into":
        label += f" · {tracker.zone}"
    return label


def _fmt_time(wall_ts: float) -> str:
    return datetime.datetime.fromtimestamp(wall_ts).strftime("%-I:%M %p")
