import { useEffect, useMemo, useRef, useState } from "react";
import { AnimatePresence, motion } from "motion/react";
import { BellRing, Eye, Volume2 } from "lucide-react";
import { Glass } from "../ui/Glass";
import { ClipVideo } from "../ui/ClipVideo";
import { Icon } from "../ui/Icon";
import { dur, spring } from "../ui/motion";
import { speak } from "../lib/audio";
import { ALERT_EVENTS, CAREGIVER, DOSE_EVENTS } from "../lib/config";
import { toastText, where } from "../lib/format";
import { sinceReset } from "../lib/events";
import { buildSlots } from "../lib/schedule";
import { post, useConfig, useRecall } from "../lib/useRecall";
import type { EventMsg, ItemName } from "../lib/types";
import { AskCapsule } from "./AskCapsule";
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
  item: ItemName;
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
  const [asking, setAsking] = useState(false);
  const [answer, setAnswer] = useState<AnswerState | null>(null);
  const [speaking, setSpeaking] = useState(false);
  const hlTimer = useRef(0);

  const { snap, events: allEvents } = useRecall((e: EventMsg) => {
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
  });
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

  // answer card: speak it, auto-close after 12 s
  useEffect(() => {
    if (!answer) return;
    let cancel = () => {};
    if (snapRef.current?.talk_to_mom !== false) {
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

  const ask = async (item: ItemName) => {
    setAsking(false);
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
    return buildSlots(cfg.schedule, snap?.dose_times ?? [], unclear);
  }, [cfg.schedule, snap?.dose_times, events]);

  const frame = snap?.frame ?? [1280, 720];

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
            <JudgePanel snap={snap} events={events} />
          </motion.div>
        )}
      </AnimatePresence>

      <AskCapsule
        open={asking}
        onToggle={() => setAsking((a) => !a)}
        onPick={ask}
        area={{ left: judge ? GAP : 0, right: judge ? STAGE_RIGHT_JUDGE : 0 }}
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
                {speaking && (
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
                <div className="t-title3 text-label-2">{QUESTION[answer.item]}</div>
                <div className="t-display mt-3">{answer.headline}</div>
                <div className="t-display-3 mt-5 text-label-2">{answer.detail}</div>
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

      {/* tiny judge-view button */}
      <button
        onClick={() => setJudge((j) => !j)}
        aria-label="Toggle judge view (J)"
        className="pressable fixed bottom-4 left-4 z-40 grid h-8 w-8 place-items-center rounded-full bg-fill text-label-2"
      >
        <Icon as={Eye} size="small" />
      </button>

    </div>
  );
}
