import { useEffect, useMemo, useRef, useState } from "react";
import { AnimatePresence, motion } from "motion/react";
import { BellRing, Eye, Volume2 } from "lucide-react";
import { Glass } from "../ui/Glass";
import { ClipVideo } from "../ui/ClipVideo";
import { Icon } from "../ui/Icon";
import { dur, spring } from "../ui/motion";
import { speak } from "../lib/audio";
import { ALERT_EVENTS, CAREGIVER, DOSE_EVENTS } from "../lib/config";
import { doorToast, toastText, where } from "../lib/format";
import { sinceReset } from "../lib/events";
import { buildSlots } from "../lib/schedule";
import { post, useConfig, useRecall } from "../lib/useRecall";
import { EvalPill, type EvalActive } from "./EvalPill";
import { useVoice, type StopReason } from "../lib/useVoice";
import type { EvalData, EvalMsg, EventMsg, ItemName } from "../lib/types";
import { AskCapsule, type VoiceState } from "./AskCapsule";
import { JudgePanel, WherePanel } from "./Panels";
import { Zones } from "./Zones";
import { GAP, PANEL_W, STAGE_RADIUS, STAGE_RIGHT_JUDGE } from "./layout";

const QUESTION: Record<ItemName, string> = {
  glasses: "Where are my glasses?",
  wallet: "Where is my wallet?",
  medication: "Where are my pills?",
};

interface AnswerState {
  id: number;
  /** a tapped item, or a spoken question (answered by the server, which also speaks it) */
  item: ItemName | "voice";
  transcript?: string;
  headline: string;
  detail: string;
  clipUrl: string | null;
}

export default function Hub() {
  const cfg = useConfig();
  const [toasts, setToasts] = useState<{ id: number; text: string }[]>([]);
  const [alertId, setAlertId] = useState<number | null>(null);
  const [highlight, setHighlight] = useState<string | null>(null);
  const [judge, setJudge] = useState(false);
  const [answer, setAnswer] = useState<AnswerState | null>(null);
  const [speaking, setSpeaking] = useState(false);
  const [voiceState, setVoiceState] = useState<VoiceState>("idle");
  const [voiceTimes, setVoiceTimes] = useState<{ ms: number; server: number | null } | null>(null);
  const voice = useVoice();
  const recording = useRef(false);   // the mic really started (not blocked, not still starting)
  const hlTimer = useRef(0);

  // ── eval mode (E): rehearsal trials judged from the event log; invisible when off ──
  const [evalMode, setEvalMode] = useState(false);
  const [evalActive, setEvalActive] = useState<EvalActive | null>(null);
  const [evalData, setEvalData] = useState<EvalData | null>(null);
  const onEvalMsg = (m: EvalMsg) => {
    if (m.kind === "started") setEvalActive({ trial: m.trial ?? "", label: m.label ?? "", deadline: m.deadline ?? null });
    else if (m.kind === "finished") {
      setEvalActive(null);
      if (m.data) setEvalData(m.data);
      if (m.toast) pushToast(m.toast);
    } else {
      setEvalActive(null);
      setEvalData({});
      pushToast("Eval cleared");
    }
  };

  const { snap, events: allEvents, serverSpeaking } = useRecall((e: EventMsg) => {
    if (e.item === "door") {
      // Mom's screen never shows door alerts; Judge view only gets a small toast
      if (judge) {
        const id = Date.now() + Math.random();
        setToasts((t) => [...t.slice(-2), { id, text: doorToast(e) }]);
        window.setTimeout(() => setToasts((t) => t.filter((x) => x.id !== id)), 4000);
      }
      return;
    }
    if (ALERT_EVENTS.has(e.event)) {
      setAlertId(Date.now());
      return;
    }
    if (DOSE_EVENTS.has(e.event)) return;
    const text = toastText(e);
    if (!text) return;
    const id = Date.now() + Math.random();
    setToasts((t) => [...t.slice(-2), { id, text }]);
    window.setTimeout(() => setToasts((t) => t.filter((x) => x.id !== id)), 4000);
    if (e.zone) {
      setHighlight(e.zone);
      window.clearTimeout(hlTimer.current);
      hlTimer.current = window.setTimeout(() => setHighlight(null), 6000);
    }
  }, onEvalMsg);
  const events = useMemo(() => sinceReset(allEvents), [allEvents]);
  const snapRef = useRef(snap);
  snapRef.current = snap;

  // alert takeover stays 10 s
  useEffect(() => {
    if (alertId === null) return;
    const t = window.setTimeout(() => setAlertId(null), 10000);
    return () => window.clearTimeout(t);
  }, [alertId]);

  // J toggles Judge view
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key.toLowerCase() === "j" && !e.metaKey && !e.ctrlKey) setJudge((j) => !j);
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, []);

  // E toggles Eval mode. In it: 1 real dose · 2 fake · 3 second pickup · 4 where-question (then Y/N) · 5 door · R reset
  const evalOn = useRef(false);
  evalOn.current = evalMode;
  useEffect(() => {
    const TRIALS: Record<string, string> = { "1": "dose", "2": "fake", "3": "second", "4": "where", "5": "door" };
    const fail = () => pushToast("Eval isn't available yet");
    const onKey = (e: KeyboardEvent) => {
      if (e.metaKey || e.ctrlKey || e.altKey) return;
      const k = e.key.toLowerCase();
      if (k === "e") return void setEvalMode((m) => !m);
      if (!evalOn.current) return;
      if (TRIALS[k]) post("/api/eval/start", { type: TRIALS[k] }).catch(fail);
      else if (k === "y" || k === "n") post("/api/eval/mark", { correct: k === "y" }).catch(fail);
      else if (k === "r") post("/api/eval/reset").catch(fail);
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, []); // eslint-disable-line react-hooks/exhaustive-deps

  // the latest results (for the "Tested" card), and the tool's state when Eval mode opens
  useEffect(() => {
    fetch("/api/eval").then((r) => (r.ok ? r.json() : null)).then((d) => d && setEvalData(d)).catch(() => {});
  }, []);
  useEffect(() => {
    if (!evalMode) return;
    fetch("/api/eval/status")
      .then((r) => (r.ok ? r.json() : null))
      .then((d) => {
        if (!d) return;
        setEvalData(d.data);
        setEvalActive(d.active ? { trial: d.active, label: d.active, deadline: d.deadline ?? null } : null);
      })
      .catch(() => {});
  }, [evalMode]);

  // answer card: speak it, auto-close after 12 s
  useEffect(() => {
    if (!answer) return;
    let cancel = () => {};
    // spoken answers are read aloud by the server (macOS "say"); tapped ones by this browser
    if (answer.item !== "voice" && snapRef.current?.talk_to_mom !== false) {
      setSpeaking(true);
      cancel = speak(`${answer.headline} ${answer.detail}`, () => setSpeaking(false));
    }
    const t = window.setTimeout(() => setAnswer(null), 12000);
    return () => {
      cancel();
      setSpeaking(false);
      window.clearTimeout(t);
    };
  }, [answer?.id]); // eslint-disable-line react-hooks/exhaustive-deps

  const pushToast = (text: string) => {
    const id = Date.now() + Math.random();
    setToasts((t) => [...t.slice(-2), { id, text }]);
    window.setTimeout(() => setToasts((t) => t.filter((x) => x.id !== id)), 4000);
  };

  // ── tap to ask: one tap listens, it stops by itself after 1.2 s of silence ─
  const log = (msg: string) => {
    console.log(`[voice] ${msg}`);
    fetch("/api/voice/log", {   // mirror it in the server terminal
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ msg }),
      keepalive: true,
    }).catch(() => {});
  };
  const unavailable = () => pushToast("Voice isn't available — tap a button instead");

  const finish = async (reason: StopReason) => {
    if (!recording.current) return;
    recording.current = false;
    const t0 = performance.now(); // release → answer
    const rec = await voice.stop(reason);
    if (!rec) {
      setVoiceState("idle");
      return;
    }
    log(`stop (${reason}) after ${(rec.ms / 1000).toFixed(1)} s, peak ${rec.peak.toFixed(3)}, room ${rec.floor.toFixed(4)}`);
    if (!rec.speech || rec.ms < 500) {
      setVoiceState("idle");
      pushToast("I didn't hear anything. Tap to try again.");
      return;
    }
    setVoiceState("thinking");
    try {
      log(`sent ${(rec.blob.size / 1024).toFixed(0)} KB`);
      const r = await fetch("/api/voice", {
        method: "POST",
        headers: { "Content-Type": rec.blob.type || "application/octet-stream" },
        body: rec.blob,
      });
      if (!r.ok) throw new Error(String(r.status));
      const d = (await r.json()) as { transcript: string; answer: string; clip_url: string | null; timing?: { total_ms?: number } };
      const took = performance.now() - t0;
      log(`answer in ${(took / 1000).toFixed(1)} s`);
      setVoiceTimes({ ms: took, server: d.timing?.total_ms ?? null });
      setAnswer({ id: Date.now(), item: "voice", transcript: d.transcript, headline: d.answer, detail: "", clipUrl: d.clip_url });
    } catch {
      unavailable();
    } finally {
      setVoiceState("idle");
    }
  };

  const onTapAsk = async () => {
    if (voiceState === "thinking" || voiceState === "starting") return;
    if (voiceState === "listening") return void finish("tap"); // a second tap stops
    const v = snapRef.current?.voice;
    if (v && !v.ready) return unavailable(); // model still loading, or voice is off
    window.speechSynthesis?.cancel();
    setAnswer(null);
    setVoiceState("starting"); // "Getting ready…": don't say "Listening" until the mic is really recording
    const ok = await voice.start((reason) => void finish(reason));
    if (!ok) {
      setVoiceState("idle");
      unavailable(); // mic blocked / not a secure origin
      return;
    }
    recording.current = true;
    log("start");
    fetch("/api/voice/prepare", { method: "POST" }).catch(() => {}); // warm up the spoken answers while she talks
    setVoiceState("listening");
  };

  const ask = async (item: ItemName) => {
    let clipUrl: string | null = null;
    try {
      clipUrl = (await post<{ clip_url: string | null }>("/api/ask", { item })).clip_url;
    } catch {
      /* still answer from the live state */
    }
    const w = where(item, snapRef.current?.items?.[item], "mom");
    setAnswer({ id: Date.now(), item, headline: w.headline, detail: w.detail, clipUrl });
  };

  const slots = useMemo(() => {
    const start = new Date().setHours(0, 0, 0, 0) / 1000;
    const unclear = events.filter((e) => e.event === "dose_unclear" && e.ts >= start).map((e) => e.ts);
    return buildSlots(cfg.schedule, snap?.dose_times ?? [], unclear).slots;
  }, [cfg.schedule, snap?.dose_times, events]);

  const frame = snap?.frame ?? [1280, 720];
  const openHow = () => {
    window.location.href = "/how?back=/&theme=dark";   // Esc on that page comes back here
  };

  return (
    <div data-theme="dark" className="fixed inset-0 overflow-hidden bg-card text-label">
      {/* live camera: full-bleed normally; in Judge view it resizes into the left slot */}
      <motion.div
        initial={false}
        animate={
          judge
            ? { top: GAP, left: GAP, bottom: GAP, right: STAGE_RIGHT_JUDGE, borderRadius: STAGE_RADIUS }
            : { top: 0, left: 0, bottom: 0, right: 0, borderRadius: 0 }
        }
        transition={spring.gentle}
        className="absolute overflow-hidden bg-black"
      >
        <img
          src={judge ? "/stream/annotated.mjpg" : "/stream/raw.mjpg"}
          alt=""
          className={`h-full w-full ${judge ? "object-contain" : "hub-video object-cover"}`}
        />
        {!judge && <Zones zones={snap?.zones ?? []} frame={frame as [number, number]} highlight={highlight} />}
        {/* Judge view: door camera picture-in-picture (door line, shaded outside, person boxes) */}
        {judge && snap?.door?.enabled && (
          <motion.div
            initial={{ opacity: 0, scale: 0.92 }}
            animate={{ opacity: 1, scale: 1 }}
            transition={spring.bouncy}
            className="absolute bottom-4 left-4 z-10 w-[240px] overflow-hidden rounded-r14 bg-black shadow-[var(--glass-shadow)]"
          >
            <img src="/stream/door.mjpg" alt="Door camera" className="block aspect-video w-full object-cover" />
            <span className="t-caption absolute left-2 top-2 rounded-r10 bg-scrim px-2 py-1 text-label">Door</span>
          </motion.div>
        )}
      </motion.div>

      {/* toasts: centered over the camera area */}
      <motion.div
        initial={false}
        animate={{ left: judge ? GAP : 0, right: judge ? STAGE_RIGHT_JUDGE : 0 }}
        transition={spring.gentle}
        className="pointer-events-none fixed top-6 z-40 flex flex-col items-center gap-2"
      >
        <AnimatePresence>
          {toasts.map((t) => (
            <motion.div
              key={t.id}
              layout
              initial={{ y: -90, scaleY: 1.3, scaleX: 0.88, opacity: 0 }}
              animate={{ y: 0, scaleY: 1, scaleX: 1, opacity: 1 }}
              exit={{ y: -60, opacity: 0, transition: { duration: dur.fast } }}
              transition={spring.bouncy}
            >
              <Glass radius={28} lens className="t-title3 px-6 py-3">
                {t.text} <span className="text-label-2">· Just now</span>
              </Glass>
            </motion.div>
          ))}
        </AnimatePresence>
      </motion.div>

      {/* right side: "Where things are" normally; the fixed "How it works" column in Judge view */}
      <AnimatePresence>
        {!judge && (
          <motion.div key="where" className="absolute right-6 top-6 z-20" exit={{ opacity: 0, scale: 0.94, transition: { duration: dur.fast } }}>
            <WherePanel snap={snap} slots={slots} />
          </motion.div>
        )}
        {judge && (
          <motion.div
            key="judge"
            className="absolute z-20"
            style={{ top: GAP, bottom: GAP, right: GAP, width: PANEL_W }}
            initial={{ x: PANEL_W + GAP * 2 }}
            animate={{ x: 0 }}
            exit={{ x: PANEL_W + GAP * 2 }}
            transition={spring.gentle}
          >
            <JudgePanel snap={snap} events={events} onHow={openHow} voiceTimes={voiceTimes} evalData={evalData} />
          </motion.div>
        )}
      </AnimatePresence>

      {/* shown while the iPhone camera is missing (the server keeps looking; it never falls back to the Mac) */}
      <AnimatePresence>
        {snap?.error && snap.error !== "starting" && (
          <motion.div
            key="cam-error"
            initial={{ opacity: 0, scale: 0.92 }}
            animate={{ opacity: 1, scale: 1 }}
            exit={{ opacity: 0 }}
            transition={spring.bouncy}
            className="pointer-events-none fixed inset-0 z-20 grid place-items-center px-8"
          >
            <Glass radius={28} pad={24} className="max-w-[560px] text-center">
              <div className="t-title3">Waiting for the camera</div>
              <p className="t-subhead mt-2 text-label-2">{snap.error}</p>
            </Glass>
          </motion.div>
        )}
      </AnimatePresence>

      <AskCapsule
        onPick={ask}
        area={{ left: judge ? GAP : 0, right: judge ? STAGE_RIGHT_JUDGE : 0 }}
        voiceState={voiceState}
        wave={voice.wave}
        onTap={onTapAsk}
      />

      {/* answer card */}
      <AnimatePresence>
        {answer && (
          <div className="pointer-events-none fixed inset-0 z-30 grid place-items-center px-8 pb-24">
            <Glass
              key={answer.id}
              appear
              lens
              radius={28}
              pad={32}
              onClick={() => setAnswer(null)}
              className="pointer-events-auto grid w-full max-w-[1200px] cursor-pointer gap-8"
              style={{ gridTemplateColumns: answer.clipUrl ? "1.1fr 1fr" : "1fr" }}
            >
              <AnimatePresence>
                {(speaking || serverSpeaking) && (
                  <motion.div
                    initial={{ opacity: 0, scale: 0.8 }}
                    animate={{ opacity: 1, scale: 1 }}
                    exit={{ opacity: 0, scale: 0.8 }}
                    transition={spring.bouncy}
                    className="absolute right-4 top-4 z-10"
                  >
                    <Glass radius={28} className="t-subhead flex items-center gap-2 px-4 py-2 font-medium">
                      <Icon as={Volume2} size="small" /> Speaking
                    </Glass>
                  </motion.div>
                )}
              </AnimatePresence>
              <div className="flex flex-col justify-center">
                <div className="t-title3 text-label-2">
                  {answer.item === "voice" ? (answer.transcript ? `“${answer.transcript}”` : "I didn’t hear anything") : QUESTION[answer.item]}
                </div>
                <div className={`${answer.item === "voice" ? "t-display-2" : "t-display"} mt-3`}>{answer.headline}</div>
                {answer.detail && <div className="t-display-3 mt-5 text-label-2">{answer.detail}</div>}
              </div>
              {answer.clipUrl && (
                <ClipVideo
                  url={answer.clipUrl}
                  className="inner-r aspect-video w-full self-center"
                />
              )}
            </Glass>
          </div>
        )}
      </AnimatePresence>

      {/* alert takeover */}
      <AnimatePresence>
        {alertId !== null && (
          <motion.div
            key="alert"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            transition={{ duration: dur.base }}
            className="fixed inset-0 z-50 grid place-items-center bg-scrim px-8"
            onClick={() => setAlertId(null)}
          >
            <Glass appear lens radius={28} pad={32} className="max-w-[1000px] text-center">
              <span className="mx-auto mb-4 grid h-12 w-12 place-items-center text-orange">
                <BellRing size={32} strokeWidth={1.75} />
              </span>
              <div className="t-display-2">You may have already taken this one.</div>
              <div className="t-display-3 mt-4">Please check with {CAREGIVER}.</div>
              <div className="t-title3 mt-6 text-label-2">{CAREGIVER} has been notified.</div>
            </Glass>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Eval mode: nothing at all is drawn when it is off */}
      {evalMode && <EvalPill active={evalActive} data={evalData} />}

      {/* tiny judge-view button */}
      <button
        onClick={() => setJudge((j) => !j)}
        aria-label="Toggle judge view (J)"
        className="pressable fixed left-4 top-4 z-40 grid h-8 w-8 place-items-center rounded-full bg-fill text-label-2"
      >
        <Icon as={Eye} size="small" />
      </button>

    </div>
  );
}
