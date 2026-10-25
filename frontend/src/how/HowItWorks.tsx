import { useEffect, useState } from "react";
import { motion } from "motion/react";
import { BellRing, Hand, Pill, ScanFace, Video, Eye, X, Play } from "lucide-react";
import { Glass } from "../ui/Glass";
import { Icon } from "../ui/Icon";
import { spring } from "../ui/motion";
import { ITEM_LABEL, where } from "../lib/format";
import { ITEMS, type ItemName, type ItemSnap } from "../lib/types";
import { useConfig, useRecall } from "../lib/useRecall";
import { DISCLAIMER } from "../care/copy";
import {
  Arrow, Chip, ClipDiagram, CountTo, DoorDiagram, Node, NotSure, PhraseFlow, Reveal, Section,
  StateMachine, Timeline, ZonesDiagram, type Step,
} from "./diagrams";

// ── results (data/eval.json) ─────────────────────────────────────────────────

interface Result {
  value: string;
  label: string;
}
type EvalFile = Record<string, unknown>;

const METRICS: { key: string; label: string; kind: "percent" | "count" | "seconds" }[] = [
  { key: "doses_detected", label: "doses detected", kind: "percent" },
  { key: "false_taken", label: "false “taken”", kind: "count" },
  { key: "repeat_alerts", label: "repeat alerts raised", kind: "percent" },
  { key: "where_correct", label: "“where is it?” answers correct", kind: "percent" },
  { key: "door_alerts", label: "door alerts", kind: "percent" },
  { key: "avg_alert_seconds", label: "average time to alert", kind: "seconds" },
];

/** null = data/eval.json hasn't been made yet. Accepts {"results":[{value,label}…]} or the keys above. */
function parseResults(d: EvalFile | null): Result[] | null {
  if (!d || Object.keys(d).length === 0) return null;
  if (Array.isArray(d.results)) {
    return (d.results as { value?: unknown; label?: unknown }[]).map((r) => ({
      value: r.value === undefined ? "—" : String(r.value),
      label: String(r.label ?? ""),
    }));
  }
  const out: Result[] = [];
  for (const m of METRICS) {
    const v = d[m.key];
    if (v === undefined) continue;
    let value = typeof v === "string" ? v : "—";
    if (typeof v === "number") {
      value = m.kind === "percent" && v <= 1 ? `${Math.round(v * 100)}%` : m.kind === "seconds" ? `${v}s` : String(v);
    }
    out.push({ value, label: m.label });
  }
  return out.length ? out : null;
}

// ── live item state ──────────────────────────────────────────────────────────

function stateText(it: ItemName, s: ItemSnap | undefined): string {
  if (!s || s.state === "unknown") return "not seen yet";
  if (s.state === "worn") return "worn";
  if (s.state === "in_hand") return "in hand";
  const w = where(it, s, "care").short;
  return w.charAt(0).toLowerCase() + w.slice(1);
}

// ── content ──────────────────────────────────────────────────────────────────

const DOSE_STEPS: Step[] = [
  { icon: Pill, title: "Bottle resting", body: "Seen at one spot (within 40 px) long enough that it is clearly sitting there.", chip: "1.5 s of the last 3 s" },
  { icon: Hand, title: "Confirmed pickup", body: "It moved, or a hand took it and it stayed gone. A hand passing in front is ignored.", chip: "moved > 60 px" },
  { icon: ScanFace, title: "Hand to mouth", body: "MediaPipe Pose: a wrist or fingertip close to the mouth, scaled by shoulder width.", chip: "< 0.7 × shoulders · 0.4 s" },
  { icon: Video, title: "“Dose appears taken”", body: "Logged with a clip. Never “taken”: it is a movement cue, not proof.", chip: "clip saved" },
  { icon: BellRing, title: "Second pickup", body: "A chime and a familiar voice. The caregiver gets the alert with the clip and the earlier dose time.", chip: "about 1 s" },
];

const BASE_PHRASES = ["glasses", "wallet", "pill bottle"];

const PROBLEM: { to: number; prefix?: string; suffix?: string; decimals?: number; line: string }[] = [
  { to: 55, suffix: "M", line: "people with dementia worldwide" },
  { to: 1.3, prefix: "$", suffix: "T", decimals: 1, line: "a year, about half of it unpaid family care (WHO)" },
];

const BUILT_WITH = [
  "YOLO-World", "MediaPipe", "OpenCV", "PyTorch (Apple GPU)", "FastAPI", "SQLite", "WebSocket + MJPEG",
  "ffmpeg (H.264)", "React", "TypeScript", "Vite", "Tailwind", "motion",
];

export default function HowItWorks() {
  const params = new URLSearchParams(location.search);
  const back = params.get("back");
  const theme = params.get("theme"); // the hub is always dark
  const { snap } = useRecall();
  const cfg = useConfig();
  const [results, setResults] = useState<Result[] | null>(null);
  const [resultsLoaded, setResultsLoaded] = useState(false);

  useEffect(() => {
    fetch("/api/eval")
      .then((r) => (r.ok ? r.json() : null))
      .then((d) => setResults(parseResults(d)))
      .catch(() => {})
      .finally(() => setResultsLoaded(true));
  }, []);

  const close = () => {
    if (back) window.location.href = back;
    else if (window.history.length > 1) window.history.back();
    else window.location.href = "/care";
  };
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") close();
    };
    window.addEventListener("keydown", onKey);
    return () => {
      window.removeEventListener("keydown", onKey);
    };
  }, []); // eslint-disable-line react-hooks/exhaustive-deps

  const live = snap?.ready;
  const phrases = [...new Set([...BASE_PHRASES, ...Object.values(cfg.items ?? {}).flat()])];

  return (
    <div data-theme={theme === "dark" ? "dark" : theme === "light" ? "light" : undefined} className="min-h-screen bg-bg text-label">
      <div className="fixed right-4 top-4 z-40" style={{ paddingTop: "env(safe-area-inset-top, 0px)" }}>
        <button onClick={close} aria-label="Close (Esc)">
          <Glass radius={28} lens className="t-subhead pressable flex items-center gap-2 px-4 py-2 font-medium">
            <Icon as={X} size="small" /> Close
          </Glass>
        </button>
      </div>

      <main className="mx-auto w-full max-w-[1080px] px-6 pb-8">
        {/* 1 · hero */}
        <section className="flex min-h-[88vh] flex-col justify-center py-8">
          <motion.div
            initial={{ opacity: 0, y: 16 }}
            animate={{ opacity: 1, y: 0 }}
            transition={spring.gentle}
            className="t-subhead tabular mb-6 flex w-fit items-center gap-2 rounded-full bg-fill px-4 py-2"
          >
            <span className={`h-2 w-2 rounded-full ${live ? "bg-green" : "bg-label-3"}`} />
            {live ? `3 models running · ${snap?.fps ?? 0} FPS` : "Connecting…"}
          </motion.div>
          <motion.h1 initial={{ opacity: 0, y: 28 }} animate={{ opacity: 1, y: 0 }} transition={{ ...spring.gentle, delay: 0.08 }} className="t-hero max-w-[14ch]">
            Recall gives the home a memory.
          </motion.h1>
          <motion.p initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ ...spring.gentle, delay: 0.18 }} className="t-title3 mt-8 max-w-[36ch] text-label-2">
            55 million people worldwide live with dementia, heading to 139 million by 2050.
          </motion.p>
        </section>

        {/* 2 · live now */}
        <section className="py-12">
          <Reveal>
            <div className="grid gap-4 md:grid-cols-[1.6fr_1fr]">
              <div className="relative overflow-hidden rounded-r28 bg-black">
                {live ? (
                  <img src="/stream/annotated.mjpg" alt="Live camera with detections" className="block aspect-video w-full object-cover" />
                ) : (
                  <div className="t-callout grid aspect-video w-full place-items-center text-label-2">Waiting for the camera…</div>
                )}
                <span className="t-caption absolute left-3 top-3 flex items-center gap-2 rounded-full bg-scrim px-3 py-1 text-label">
                  <span className="h-2 w-2 animate-pulse rounded-full bg-green" /> Live
                </span>
              </div>
              <div className="flex flex-col justify-center gap-2 rounded-r28 bg-card p-6">
                {ITEMS.map((it) => (
                  <motion.div key={it + (snap?.items?.[it]?.state ?? "")} initial={{ opacity: 0.4, x: 6 }} animate={{ opacity: 1, x: 0 }} transition={spring.default} className="t-title3">
                    {ITEM_LABEL[it]} <span className="text-label-2">· {stateText(it, snap?.items?.[it])}</span>
                  </motion.div>
                ))}
                <p className="t-footnote mt-3 text-label-2">This is running right now, on this laptop.</p>
              </div>
            </div>
          </Reveal>
        </section>

        {/* 3 · the problem */}
        <Section kicker="The problem" title="Two questions every family asks.">
          <div className="grid gap-4 md:grid-cols-3">
            {PROBLEM.map((p, i) => (
              <Reveal key={p.line} delay={i * 0.08}>
                <div className="h-full rounded-r28 bg-card p-8">
                  <div className="t-bignum-md tabular">
                    <CountTo to={p.to} prefix={p.prefix} suffix={p.suffix} decimals={p.decimals} />
                  </div>
                  <p className="t-body mt-3 text-label-2">{p.line}</p>
                </div>
              </Reveal>
            ))}
            <Reveal delay={0.16}>
              <div className="h-full rounded-r28 bg-card p-8">
                <div className="t-bignum-md">2</div>
                <p className="t-body mt-3 text-label-2">questions: did she take it? Where did it go?</p>
              </div>
            </Reveal>
          </div>
          <Reveal delay={0.1}>
            <p className="t-title2 mt-12 max-w-[30ch]">Recall catches the moments that turn into emergencies: a second dose, an unnoticed exit.</p>
          </Reveal>
        </Section>

        {/* 4 · medication safety */}
        <Section kicker="Medication safety" title="Five checks before it says a word.">
          <Timeline steps={DOSE_STEPS} />
          <Reveal delay={0.1} className="mt-6">
            <NotSure>Can’t see the mouth? It logs “not sure”, and a second reach still alerts.</NotSure>
          </Reveal>
        </Section>

        {/* 5 · memory for things */}
        <Section kicker="Memory for things" title="Every item is a tiny state machine.">
          <Reveal>
            <StateMachine />
          </Reveal>
          <Reveal delay={0.1} className="mt-8">
            <div className="grid gap-3 sm:grid-cols-2">
              {[
                ["Objects don’t move by themselves", "A state only changes when a hand is near, or the item is seen elsewhere."],
                ["Debounce 1 s", "A new state has to hold for a second before it counts."],
                ["Hysteresis", "Easy to keep tracking, harder to start. Detection flicker doesn’t reset memory."],
                ["Contact point", "Its center, then its base, decides which zone it is in."],
              ].map(([t, d]) => (
                <div key={t} className="rounded-r20 bg-card p-4">
                  <Chip strong>{t}</Chip>
                  <p className="t-subhead mt-2 text-label-2">{d}</p>
                </div>
              ))}
            </div>
          </Reveal>
          <Reveal delay={0.15} className="mt-6">
            <div className="flex items-center gap-4 rounded-r28 bg-card p-6">
              <span className="grid h-12 w-12 shrink-0 place-items-center rounded-full bg-label text-on-label">
                <Icon as={Play} />
              </span>
              <p className="t-title3">“I last saw your glasses going into the basket at 2:14 PM.”</p>
            </div>
          </Reveal>
        </Section>

        {/* 6 · sees without training */}
        <Section kicker="Sees without training" title="Add an item by typing its name.">
          <Reveal>
            <PhraseFlow phrases={phrases} />
            <p className="t-subhead mt-3 max-w-[60ch] text-label-2">
              YOLO-World matches the camera image against plain-English phrases through a CLIP text encoder. No photos, no labeling, no retraining.
            </p>
          </Reveal>
          <Reveal delay={0.1} className="mt-8">
            <div className="grid items-center gap-6 md:grid-cols-[1.3fr_1fr]">
              <ZonesDiagram />
              <div>
                <h3 className="t-title3">Zones find themselves.</h3>
                <p className="t-body mt-2 text-label-2">A table becomes “counter”, a basket becomes “basket”. Re-derived after every detection, so they follow if the camera moves.</p>
              </div>
            </div>
          </Reveal>
        </Section>

        {/* 7 · leaving the house */}
        <Section kicker="Leaving the house" title="A line at the door.">
          <Reveal>
            <div className="grid items-center gap-6 md:grid-cols-[1.3fr_1fr]">
              <DoorDiagram />
              <div>
                <p className="t-body text-label-2">Cross the line from inside to outside and stay there for a second: “Left the house.” Same line back is “Back home”, 30 s apart at most.</p>
                <p className="t-subhead mt-4 text-label-2">The person detector runs on the CPU in its own thread. The kitchen never slows down.</p>
              </div>
            </div>
          </Reveal>
        </Section>

        {/* 8 · proof */}
        <Section kicker="Proof" title="A clip around every moment.">
          <Reveal>
            <ClipDiagram />
          </Reveal>
        </Section>

        {/* 9 · architecture */}
        <Section kicker="Architecture" title="All on-device. Nothing leaves the home.">
          <Reveal>
            <div className="rounded-r28 border-2 border-dashed border-separator p-4">
              <div className="t-footnote mb-3 font-semibold text-label-2">On this Mac</div>
              <div className="flex flex-col items-stretch gap-3 min-[761px]:flex-row min-[761px]:items-center">
                <div className="flex flex-col gap-3 min-[761px]:w-[150px]">
                  <Node title="iPhone camera" lines={["kitchen"]} />
                  <Node title="Mac webcam" lines={["door"]} />
                </div>
                <Arrow />
                <div className="flex-[1.6] rounded-r20 bg-card p-4">
                  <div className="t-headline mb-2 text-center">Python pipeline</div>
                  <div className="flex flex-wrap justify-center gap-2">
                    {["YOLO-World · Apple GPU", "MediaPipe Hands + Pose", "state machines", "SQLite event log", "clip writer"].map((c) => (
                      <Chip key={c}>{c}</Chip>
                    ))}
                  </div>
                </div>
                <Arrow />
                <Node title="FastAPI" lines={["MJPEG", "WebSocket"]} className="min-[761px]:max-w-[130px]" />
                <Arrow />
                <div className="flex flex-col gap-3 min-[761px]:w-[150px]">
                  <Node title="Hub screen" lines={["for Mom"]} />
                  <Node title="Caregiver phone" lines={["for family"]} />
                </div>
              </div>
            </div>
          </Reveal>
        </Section>

        {/* 10 · results */}
        <Section kicker="Results" title="Measured, not promised.">
          {resultsLoaded && results ? (
            <div className="grid gap-12 sm:grid-cols-2 md:grid-cols-3">
              {results.map((r, i) => (
                <Reveal key={r.label} delay={i * 0.08}>
                  <div className="t-bignum tabular">{r.value}</div>
                  <div className="t-body mt-3 text-label-2">{r.label}</div>
                </Reveal>
              ))}
            </div>
          ) : (
            <Reveal>
              <div className="rounded-r28 bg-card p-8">
                <div className="t-title1">{resultsLoaded ? "Measured live, Sunday." : "Loading…"}</div>
              </div>
            </Reveal>
          )}
        </Section>

        {/* 11 · built with */}
        <section className="py-16">
          <Reveal>
            <div className="t-subhead mb-4 font-semibold text-label-2">Built with</div>
            <div className="flex flex-wrap gap-2">
              {BUILT_WITH.map((b) => (
                <Chip key={b}>{b}</Chip>
              ))}
            </div>
          </Reveal>
        </section>

        <footer className="t-footnote flex items-center justify-center gap-2 border-t border-separator py-8 text-center text-label-2">
          <Icon as={Eye} size="small" /> {DISCLAIMER}
        </footer>
      </main>
    </div>
  );
}
