export interface ItemSnap {
  state: string;
  label: string;
  zone: string | null;
  since: number; // unix seconds
}

export interface ZoneSnap {
  name: string;
  box: [number, number, number, number]; // x1 y1 x2 y2 in camera pixels
}

export interface Snapshot {
  type: "state";
  ts: number;
  ready?: boolean;
  error?: string | null;
  items?: Record<string, ItemSnap>;
  zones?: ZoneSnap[];
  frame?: [number, number]; // [width, height] of the camera frame
  doses_today?: number;
  dose_times?: number[];
  last_dose_ts?: number | null;
  unclear?: number;
  last_alert?: { event: string; message: string; ts: number } | null;
  fps?: number;
  talk_to_mom: boolean;
  door_alerts?: boolean;
  /** push-to-talk: the speech model loads in the background after the server starts */
  voice?: { ready: boolean; error: string | null };
  /** the door camera (the Mac's webcam); enabled=false when there is no second camera */
  door?: {
    enabled: boolean;
    fps: number;
    people: number;
    last_event: { event: string; ts: number } | null;
    error: string | null;
  };
  /** dose state machine, for the judge view */
  dose?: {
    state: string;
    seconds_left: number;
    mouth_visible: boolean;
    closest_ratio: number | null;
    threshold: number;
    since_ref: number | null;
    last_event: string;
  };
}

export interface EventMsg {
  type: "event";
  item: string;
  event: string;
  zone: string | null;
  ts: number;
  clip_url: string | null;
  message: string;
  /** door events logged while "Door alerts" was off: shown as activity, never as an alert */
  silent?: boolean;
}

export interface ScheduleSlot {
  label: string;
  time: string; // "HH:MM" 24h
}

export interface Config {
  schedule: ScheduleSlot[];
  mom_phone: string;
  /** item name → the plain-English phrases the detector is told to look for (data/items.json) */
  items?: Record<string, string[]>;
}

/** One test type in the eval results (data/eval.json → breakdown). */
export interface EvalRow {
  label: string;
  trials: number;
  passes: number;
  rate: number; // 0..1
  avg_alert_seconds: number | null;
  reasons: Record<string, number>; // why the failures failed, from the event log
}
/** data/eval.json: fractions 0..1, false_taken as a count. */
export interface EvalData {
  trials?: number;
  doses_detected?: number;
  false_taken?: number;
  repeat_alerts?: number;
  where_correct?: number;
  door_alerts?: number;
  avg_alert_seconds?: number;
  breakdown?: Record<string, EvalRow>;
}
/** live messages while the eval tool runs */
export interface EvalMsg {
  type: "eval";
  kind: "started" | "finished" | "reset";
  trial?: string;
  label?: string;
  deadline?: number | null;
  toast?: string;
  ok?: boolean;
  data?: EvalData;
}

export const ITEMS = ["glasses", "wallet", "medication"] as const;
export type ItemName = (typeof ITEMS)[number];
