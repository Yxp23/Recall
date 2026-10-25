import { ALERT_EVENTS } from "./config";
import type { EventMsg } from "./types";

/** Events worth showing in the caregiver feed (pickups / reappearances are noise). */
export const FEED_EVENTS = new Set([
  "placed",
  "went_into",
  "worn",
  "taken_off",
  "dose_appears_taken",
  "dose_unclear",
  "repeat_reach",
  "possible_second_dose",
]);

export const evKey = (e: EventMsg) => `${e.item}|${e.event}|${e.ts}`;

/** Only events after the latest doses_reset (Reset demo clears the caregiver view). */
export function sinceReset(events: EventMsg[]): EventMsg[] {
  let resetTs = 0;
  for (const e of events) if (e.event === "doses_reset" && e.ts > resetTs) resetTs = e.ts;
  return events.filter((e) => e.event !== "doses_reset" && e.ts > resetTs);
}

export interface AlertGroup {
  key: string;
  latest: EventMsg;
  members: EventMsg[];
  clipUrl: string | null;
}

const GROUP_WINDOW_S = 60;

/** Alerts (newest first) within 60 s of each other become one card. */
export function groupAlerts(events: EventMsg[]): AlertGroup[] {
  const alerts = events.filter((e) => ALERT_EVENTS.has(e.event)).sort((a, b) => b.ts - a.ts);
  const groups: AlertGroup[] = [];
  for (const a of alerts) {
    const g = groups[groups.length - 1];
    if (g && g.latest.ts - a.ts <= GROUP_WINDOW_S) {
      g.members.push(a);
      g.clipUrl ??= a.clip_url;
    } else {
      groups.push({ key: evKey(a), latest: a, members: [a], clipUrl: a.clip_url });
    }
  }
  return groups;
}

export interface FeedItem {
  key: string;
  e: EventMsg;
  count: number;
}

/** Consecutive identical rows collapse into one row with a count. */
export function collapseFeed(events: EventMsg[]): FeedItem[] {
  const out: FeedItem[] = [];
  for (const e of events) {
    const prev = out[out.length - 1];
    if (prev && prev.e.item === e.item && prev.e.event === e.event && prev.e.zone === e.zone) {
      prev.count += 1;
      prev.e = { ...prev.e, clip_url: prev.e.clip_url ?? e.clip_url };
    } else {
      out.push({ key: evKey(e), e, count: 1 });
    }
  }
  return out;
}
