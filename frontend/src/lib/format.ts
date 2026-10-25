import type { EventMsg, ItemSnap } from "./types";

export const ITEM_LABEL: Record<string, string> = {
  glasses: "Glasses",
  wallet: "Wallet",
  medication: "Pills",
};
export const itemLabel = (i: string) => ITEM_LABEL[i] ?? i[0].toUpperCase() + i.slice(1);
const PRONOUN: Record<string, string> = { glasses: "them", wallet: "it", medication: "it" };
export const pronoun = (i: string) => PRONOUN[i] ?? "it";

export const cap = (s: string) => s.charAt(0).toUpperCase() + s.slice(1);

const IN_ZONES = new Set(["basket", "bowl"]);
const prep = (zone: string) => `${IN_ZONES.has(zone) ? "in" : "on"} the ${zone}`;

/** 12-hour time everywhere: "8:33 PM". */
export function clock(ts: number): string {
  return new Date(ts * 1000).toLocaleTimeString("en-US", { hour: "numeric", minute: "2-digit", hour12: true });
}

/** "just now", "2 min ago", "3 h ago", else the clock time (for "Last activity …"). */
export function ago(ts: number, now = Date.now() / 1000): string {
  const s = Math.max(0, now - ts);
  if (s < 60) return "just now";
  if (s < 3600) return `${Math.floor(s / 60)} min ago`;
  if (s < 86400) return `${Math.floor(s / 3600)} h ago`;
  return clock(ts);
}

export function relative(ts: number, now = Date.now() / 1000): string {
  const s = Math.max(0, now - ts);
  if (s < 10) return "Just now";
  if (s < 60) return `${Math.floor(s)}s ago`;
  if (s < 3600) return `${Math.floor(s / 60)} min ago`;
  return clock(ts);
}

export interface Where {
  headline: string; // huge answer
  detail: string; // smaller line under it
  short: string; // one-line location for lists
}

/** Plain-English location. `who` = whose perspective: Mom ("your") or caregiver ("her"). */
export function where(item: string, s: ItemSnap | undefined, who: "mom" | "care"): Where {
  const mom = who === "mom";
  const them = pronoun(item);
  const none: Where = {
    headline: "Not sure.",
    detail: mom ? `I haven't seen ${them === "them" ? "them" : "it"} yet today.` : "Not seen today.",
    short: "Not seen today",
  };
  if (!s) return none;
  const at = clock(s.since);
  switch (s.state) {
    case "on":
      return {
        headline: s.zone ? `${cap(prep(s.zone))}.` : "Nearby.",
        detail: mom ? `You put ${them} there at ${at}.` : `Placed there at ${clock(s.since)}.`,
        short: s.zone ? cap(prep(s.zone)) : "Nearby",
      };
    case "worn":
      return {
        headline: mom ? "You're wearing them." : "Being worn.",
        detail: mom ? `Since ${at}.` : `Since ${clock(s.since)}.`,
        short: "Being worn",
      };
    case "in_hand":
      return {
        headline: mom ? "In your hand." : "In her hand.",
        detail: mom ? "You're holding it right now." : "Being held right now.",
        short: mom ? "In your hand" : "In her hand",
      };
    case "last_seen_into":
      return {
        headline: s.zone ? `${cap(prep(s.zone))}.` : "Out of view.",
        detail: mom ? `I last saw ${them} go there at ${at}.` : `Last seen going there at ${clock(s.since)}.`,
        short: s.zone ? `Last seen ${prep(s.zone)}` : "Out of view",
      };
    case "last_seen_worn":
      return {
        headline: mom ? "You were wearing them." : "Last seen being worn.",
        detail: mom ? "Check your face or your shirt pocket." : `At ${clock(s.since)}.`,
        short: "Last seen being worn",
      };
    case "last_seen_in_hand":
      return {
        headline: mom ? "Last in your hand." : "Last seen in her hand.",
        detail: mom ? `I saw ${them} in your hand at ${at}.` : `At ${clock(s.since)}.`,
        short: "Last seen in her hand",
      };
    default:
      return none;
  }
}

/** Is this item currently somewhere we can name? (for the ✓ on the hub) */
export const isKnown = (s?: ItemSnap) => !!s && s.state !== "unknown";

const TOAST_VERB: Record<string, (z: string | null) => string> = {
  picked_up: () => "In your hand",
  placed: (z) => (z ? cap(z) : "Put down"),
  went_into: (z) => (z ? cap(z) : "Out of view"),
  worn: () => "Being worn",
  taken_off: (z) => (z ? cap(z) : "Taken off"),
  went_worn: () => "Last seen worn",
  went_in_hand: () => "Last seen in hand",
  reappeared: (z) => (z ? cap(z) : "Back in view"),
};

/** "Glasses → Basket" or null if this event isn't a toast. */
export function toastText(e: EventMsg): string | null {
  const v = TOAST_VERB[e.event];
  return v ? `${itemLabel(e.item)} → ${v(e.zone)}` : null;
}

/** Hub, Judge view only: "Door: left_house". */
export const doorToast = (e: EventMsg) => `Door: ${e.event}`;

/** Plain-English sentence for the caregiver activity feed. */
export function sentence(e: EventMsg): string {
  const name = itemLabel(e.item);
  switch (e.event) {
    case "dose_appears_taken":
      return "Dose appears taken";
    case "dose_unclear":
      return "Reached for pills — not sure";
    case "repeat_reach":
      return "Reached for pills again";
    case "possible_second_dose":
      return "Possible second dose — please check";
    case "doses_reset":
      return "Demo reset";
    case "left_house":
      return "Left the house";
    case "came_back":
      return "Back home";
    case "picked_up":
      return `Picked up ${name.toLowerCase()}`;
    case "placed":
      return e.zone ? `Put ${name.toLowerCase()} ${prep(e.zone)}` : `Put down ${name.toLowerCase()}`;
    case "went_into":
      return e.zone ? `${name} went ${e.zone === "basket" || e.zone === "bowl" ? "into" : "onto"} the ${e.zone}` : `${name} went out of view`;
    case "worn":
      return `${name} put on`;
    case "taken_off":
      return `${name} taken off`;
    case "reappeared":
      return e.zone ? `${name} back ${prep(e.zone)}` : `${name} back in view`;
    case "went_worn":
      return `${name} last seen being worn`;
    case "went_in_hand":
      return `${name} last seen in hand`;
    default:
      return e.message;
  }
}
