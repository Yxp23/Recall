import { useEffect, useRef, useState } from "react";
import type { Config, EventMsg, Snapshot } from "./types";

const key = (e: EventMsg) => `${e.item}|${e.event}|${e.ts}`;

/** Merge events (newest first), de-duplicating and keeping any clip_url we learn. */
function merge(incoming: EventMsg[], prev: EventMsg[]): EventMsg[] {
  const map = new Map<string, EventMsg>();
  for (const e of [...prev, ...incoming]) {
    const old = map.get(key(e));
    map.set(key(e), old ? { ...old, ...e, clip_url: e.clip_url ?? old.clip_url } : e);
  }
  return [...map.values()].sort((a, b) => b.ts - a.ts).slice(0, 300);
}

export function useRecall(onEvent?: (e: EventMsg) => void) {
  const [snap, setSnap] = useState<Snapshot | null>(null);
  const [events, setEvents] = useState<EventMsg[]>([]);
  const [connected, setConnected] = useState(false);
  const cb = useRef(onEvent);
  cb.current = onEvent;

  useEffect(() => {
    let ws: WebSocket | null = null;
    let timer = 0;
    let closed = false;

    // The server answers 503 until its pipeline has started (models load for a few seconds),
    // so keep trying to load history until it succeeds, including after the first "ready" snapshot.
    let historyLoaded = false;
    const loadHistory = () =>
      fetch("/api/events?limit=200")
        .then((r) => (r.ok ? r.json() : Promise.reject(new Error(String(r.status)))))
        .then((list: EventMsg[]) => {
          historyLoaded = true;
          setEvents((prev) => merge(list, prev));
        })
        .catch(() => {});

    const connect = () => {
      const proto = location.protocol === "https:" ? "wss" : "ws";
      ws = new WebSocket(`${proto}://${location.host}/ws`);
      ws.onopen = () => {
        setConnected(true);
        loadHistory();
      };
      ws.onmessage = (m) => {
        const d = JSON.parse(m.data);
        if (d.type === "state") {
          setSnap(d as Snapshot);
          if (d.ready && !historyLoaded) loadHistory();
        }
        else if (d.type === "event") {
          setEvents((prev) => merge([d as EventMsg], prev));
          cb.current?.(d as EventMsg);
        }
      };
      ws.onclose = () => {
        setConnected(false);
        if (!closed) timer = window.setTimeout(connect, 1500);
      };
    };
    connect();
    return () => {
      closed = true;
      window.clearTimeout(timer);
      ws?.close();
    };
  }, []);

  return { snap, events, connected };
}

const FALLBACK_CONFIG: Config = {
  schedule: [
    { label: "Morning", time: "08:00" },
    { label: "Afternoon", time: "14:00" },
    { label: "Evening", time: "20:00" },
  ],
  mom_phone: "",
};

export function useConfig(): Config {
  const [cfg, setCfg] = useState<Config>(FALLBACK_CONFIG);
  useEffect(() => {
    fetch("/api/config")
      .then((r) => (r.ok ? r.json() : null))
      .then((c) => c && setCfg(c))
      .catch(() => {});
  }, []);
  return cfg;
}

export async function post<T = unknown>(url: string, body?: unknown): Promise<T> {
  const r = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: body === undefined ? undefined : JSON.stringify(body),
  });
  if (!r.ok) throw new Error(`${url} → ${r.status}`);
  return r.json();
}
