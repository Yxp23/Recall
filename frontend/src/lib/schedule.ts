import type { ScheduleSlot } from "./types";

/** observed · unsure (a dose_unclear happened near the slot) · none (past, nothing seen) · upcoming */
export type SlotStatus = "observed" | "unsure" | "none" | "upcoming";
export interface Slot {
  label: string;
  time: string; // "8:00 AM"
  status: SlotStatus;
  at?: number; // unix seconds of the observed dose
}

const minutesOf = (hhmm: string) => {
  const [h, m] = hhmm.split(":").map(Number);
  return h * 60 + (m || 0);
};
const todMinutes = (ts: number) => {
  const d = new Date(ts * 1000);
  return d.getHours() * 60 + d.getMinutes();
};
const pretty = (hhmm: string) => {
  const d = new Date();
  d.setHours(...(hhmm.split(":").map(Number) as [number, number]), 0, 0);
  return d.toLocaleTimeString("en-US", { hour: "numeric", minute: "2-digit", hour12: true });
};

/** How long after the scheduled time we still say "Upcoming" rather than "No observation". */
const GRACE_MIN = 90;
/** A dose further than this from every slot is an extra, never credited to a slot. */
export const SLOT_REACH_MIN = 120;

/** Distance (minutes, around the clock) between two times of day. */
const gap = (a: number, b: number) => {
  const d = Math.abs(a - b);
  return Math.min(d, 24 * 60 - d);
};

/** Index of the nearest slot within SLOT_REACH_MIN of this time, or -1. */
function slotFor(mins: number[], ts: number): number {
  const m = todMinutes(ts);
  let best = -1;
  let bd = Infinity;
  mins.forEach((sm, i) => {
    const d = gap(sm, m);
    if (d <= SLOT_REACH_MIN && d < bd) {
      bd = d;
      best = i;
    }
  });
  return best;
}

export interface SlotResult {
  slots: Slot[];
  /** observed doses more than 2 h from any slot (listed as "Extra · 12:36 AM", not credited) */
  extras: number[];
}

/**
 * Credit today's observed doses to the nearest slot within 2 h; anything further away is an extra.
 */
export function buildSlots(
  schedule: ScheduleSlot[],
  doseTimes: number[],
  unclearTimes: number[],
  now = new Date(),
): SlotResult {
  const mins = schedule.map((s) => minutesOf(s.time));
  const nowMin = now.getHours() * 60 + now.getMinutes();
  const credited = new Map<number, number[]>();
  const extras: number[] = [];
  for (const t of [...doseTimes].sort((a, b) => a - b)) {
    const i = slotFor(mins, t);
    if (i < 0) extras.push(t);
    else credited.set(i, [...(credited.get(i) ?? []), t]);
  }
  const unclearSlots = new Set(unclearTimes.map((t) => slotFor(mins, t)).filter((i) => i >= 0));

  const slots = schedule.map((s, i): Slot => {
    const base = { label: s.label, time: pretty(s.time) };
    const dose = credited.get(i)?.[0];
    if (dose !== undefined) return { ...base, status: "observed", at: dose };
    if (unclearSlots.has(i)) return { ...base, status: "unsure" };
    if (nowMin < mins[i] + GRACE_MIN) return { ...base, status: "upcoming" };
    return { ...base, status: "none" };
  });
  return { slots, extras };
}
