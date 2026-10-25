import type { ScheduleSlot } from "./types";

/** observed · unsure (a dose_unclear happened in the slot) · none (past, nothing seen) · upcoming */
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

/** How long after the scheduled time we still say "Upcoming" rather than "Not sure". */
const GRACE_MIN = 90;

/**
 * Assign today's observed doses to schedule slots. Each slot owns the window
 * from the midpoint with the previous slot to the midpoint with the next.
 */
export function buildSlots(
  schedule: ScheduleSlot[],
  doseTimes: number[],
  unclearTimes: number[],
  now = new Date(),
): Slot[] {
  const mins = schedule.map((s) => minutesOf(s.time));
  const nowMin = now.getHours() * 60 + now.getMinutes();
  return schedule.map((s, i) => {
    const start = i === 0 ? 0 : (mins[i - 1] + mins[i]) / 2;
    const end = i === mins.length - 1 ? 24 * 60 : (mins[i] + mins[i + 1]) / 2;
    const inWin = (t: number) => {
      const m = todMinutes(t);
      return m >= start && m < end;
    };
    const dose = doseTimes.filter(inWin).sort((a, b) => a - b)[0];
    const base = { label: s.label, time: pretty(s.time) };
    if (dose !== undefined) return { ...base, status: "observed", at: dose } as Slot;
    if (unclearTimes.some(inWin)) return { ...base, status: "unsure" } as Slot;
    if (nowMin < mins[i] + GRACE_MIN) return { ...base, status: "upcoming" } as Slot;
    return { ...base, status: "none" } as Slot;
  });
}
