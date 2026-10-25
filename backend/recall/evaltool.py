"""
Minimal eval tool. Before each rehearsal trial the operator presses a key on the Hub; this module
notes the start time, then judges the trial from the event log (what Recall itself recorded),
appends it to data/eval_trials.jsonl and rewrites data/eval.json, which the /how page displays.
It only READS the event log: detection logic is untouched.

  dose   real dose                         pass: dose_appears_taken within 12 s
  fake   pickup, no mouth                  pass: NO dose_appears_taken
  second second pickup after a dose        pass: repeat_reach (alert time = pickup → chime)
  where  a "where is it?" question         pass: the operator marks the answer correct (Y)
  door   walk out the door                 pass: left_house within 12 s
"""
from __future__ import annotations

import json
import os
import threading
import time
from collections import Counter
from typing import Callable, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from recall.events.store import EventStore

WINDOW_S = 12.0
LABEL = {"dose": "Dose", "fake": "Fake pickup", "second": "Second dose", "where": "Where", "door": "Door"}
ORDER = ["dose", "fake", "second", "where", "door"]
TRIALS_PATH = "data/eval_trials.jsonl"
OUT_PATH = "data/eval.json"


def _clean(detail: str) -> str:
    return (detail or "").replace("[held>=3s]", "").strip()


def _failure_reason(kind: str, evs: list[dict]) -> str:
    names = [e["event"] for e in evs]
    unclear = next((_clean(e["detail"]) for e in evs if e["event"] == "dose_unclear"), None)
    if kind == "fake":
        return "counted as taken"
    if kind == "door":
        return "crossing seen as 'back home'" if "came_back" in names else "no crossing seen"
    if kind == "where":
        return "wrong answer"
    # dose / second
    if unclear:
        return unclear if unclear != "timeout" else "no hand-to-mouth seen (timeout)"
    if "possible_second_dose" in names:
        return "treated as a second dose"
    if kind == "second" and "dose_appears_taken" in names:
        return "counted as another dose"
    if kind == "dose" and "repeat_reach" in names:
        return "repeat alert instead (earlier dose or unclear within 4 h)"
    return "pickup not confirmed"


class EvalTool:
    def __init__(self, store: "EventStore", publish: Callable[[dict], None]) -> None:
        self._store = store
        self._publish = publish
        self._lock = threading.Lock()
        self._active: Optional[dict] = None
        self.trials: list[dict] = []
        try:
            with open(TRIALS_PATH) as f:
                self.trials = [json.loads(line) for line in f if line.strip()]
        except (OSError, ValueError):
            pass

    # ── operator actions ───────────────────────────────────────────────────────

    def start(self, kind: str) -> dict:
        if kind not in LABEL:
            raise ValueError(f"unknown trial type {kind!r}")
        with self._lock:
            prev = self._active
            self._active = None
        if prev is not None:
            if prev["timer"] is not None:
                prev["timer"].cancel()
            if prev["type"] != "where":   # moved on early: judge it on what happened so far
                self._finish(prev, time.time())
        t0 = time.time()
        trial = {"id": int(t0 * 1000), "type": kind, "t0": t0, "timer": None}
        if kind != "where":
            trial["timer"] = threading.Timer(WINDOW_S + 0.6, self._finish, args=(trial, None))
            trial["timer"].daemon = True
        with self._lock:
            self._active = trial
        if trial["timer"] is not None:
            trial["timer"].start()
        info = {"type": "eval", "kind": "started", "trial": kind, "label": LABEL[kind],
                "deadline": None if kind == "where" else t0 + WINDOW_S}
        self._publish(info)
        print(f"[eval] trial started: {LABEL[kind]}")
        return info

    def mark(self, correct: bool) -> bool:
        """Y / N for a where-question trial."""
        with self._lock:
            t = self._active
            if t is None or t["type"] != "where":
                return False
            self._active = None
        self._finish(t, time.time(), correct=correct)
        return True

    def reset(self) -> None:
        with self._lock:
            t, self._active = self._active, None
        if t is not None and t["timer"] is not None:
            t["timer"].cancel()
        self.trials = []
        open(TRIALS_PATH, "w").close()
        self._write({})
        self._publish({"type": "eval", "kind": "reset", "data": {}})
        print("[eval] cleared all trials")

    # ── judging ────────────────────────────────────────────────────────────────

    def _finish(self, trial: dict, t_end: Optional[float], correct: Optional[bool] = None) -> None:
        with self._lock:
            if self._active is trial:
                self._active = None
        t0 = trial["t0"]
        end = min(t_end or (t0 + WINDOW_S), t0 + WINDOW_S)
        rows = self._store.events_between(t0, end if trial["type"] != "where" else t0 + 120)
        evs = [{"event": r["event"], "item": r["item"], "detail": r["detail"] or "", "ts": r["ts"]} for r in rows]
        names = [e["event"] for e in evs]
        kind = trial["type"]
        alert_s: Optional[float] = None
        if kind == "dose":
            ok = "dose_appears_taken" in names
        elif kind == "fake":
            ok = "dose_appears_taken" not in names
        elif kind == "second":
            ok = "repeat_reach" in names
            if ok:
                chime = next(e["ts"] for e in evs if e["event"] == "repeat_reach")
                picks = [e["ts"] for e in evs if e["event"] == "picked_up" and e["item"] == "medication" and e["ts"] <= chime]
                alert_s = round(chime - (picks[-1] if picks else t0), 2)   # pickup → chime starts
        elif kind == "door":
            ok = "left_house" in names
        else:
            ok = bool(correct)
        reason = None if ok else _failure_reason(kind, evs)
        rec = {"id": trial["id"], "type": kind, "t0": t0, "pass": ok, "reason": reason, "alert_seconds": alert_s,
               "events": [{"event": e["event"], "detail": _clean(e["detail"]), "dt": round(e["ts"] - t0, 2)} for e in evs if e["event"] != "doses_reset"]}
        self.trials.append(rec)
        with open(TRIALS_PATH, "a") as f:
            f.write(json.dumps(rec) + "\n")
        data = self.summary()
        self._write(data)
        n = sum(1 for t in self.trials if t["type"] == kind)
        p = sum(1 for t in self.trials if t["type"] == kind and t["pass"])
        toast = f"{LABEL[kind]} {p}/{n} {'✓' if ok else '✗'}" + ("" if ok else f" ({reason})")
        print(f"[eval] {toast}" + (f", alert {alert_s}s" if alert_s is not None else ""))
        self._publish({"type": "eval", "kind": "finished", "toast": toast, "ok": ok, "data": data})

    def summary(self) -> dict:
        by: dict[str, list[dict]] = {k: [t for t in self.trials if t["type"] == k] for k in ORDER}
        out: dict = {"trials": len(self.trials)}
        breakdown: dict = {}
        for k, ts in by.items():
            if not ts:
                continue
            passes = sum(1 for t in ts if t["pass"])
            alerts = [t["alert_seconds"] for t in ts if t["alert_seconds"] is not None]
            breakdown[k] = {
                "label": LABEL[k], "trials": len(ts), "passes": passes, "rate": round(passes / len(ts), 3),
                "avg_alert_seconds": round(sum(alerts) / len(alerts), 2) if alerts else None,
                "reasons": dict(Counter(t["reason"] for t in ts if not t["pass"] and t["reason"])),
            }
        if "dose" in breakdown:
            out["doses_detected"] = breakdown["dose"]["rate"]
        if "fake" in breakdown:
            out["false_taken"] = breakdown["fake"]["trials"] - breakdown["fake"]["passes"]   # a count, not a rate
        if "second" in breakdown:
            out["repeat_alerts"] = breakdown["second"]["rate"]
            if breakdown["second"]["avg_alert_seconds"] is not None:
                out["avg_alert_seconds"] = breakdown["second"]["avg_alert_seconds"]
        if "where" in breakdown:
            out["where_correct"] = breakdown["where"]["rate"]
        if "door" in breakdown:
            out["door_alerts"] = breakdown["door"]["rate"]
        out["breakdown"] = breakdown
        return out

    def status(self) -> dict:
        t = self._active
        return {"active": t["type"] if t else None,
                "deadline": (t["t0"] + WINDOW_S) if t and t["type"] != "where" else None,
                "data": self.summary()}

    @staticmethod
    def _write(data: dict) -> None:
        os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
        with open(OUT_PATH, "w") as f:
            json.dump(data, f, indent=2)
