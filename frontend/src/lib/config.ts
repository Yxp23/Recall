export const PERSON = "Mom";
export const CAREGIVER = "Sarah";

import type { EventMsg } from "./types";

export const ALERT_EVENTS = new Set(["repeat_reach", "possible_second_dose", "left_house"]);
/** An event that should raise a caregiver alert (door events logged while Door alerts is off don't). */
export const isAlert = (e: EventMsg) => ALERT_EVENTS.has(e.event) && !e.silent;
export const DOSE_EVENTS = new Set([
  "dose_appears_taken",
  "dose_unclear",
  "repeat_reach",
  "possible_second_dose",
  "doses_reset",
]);
