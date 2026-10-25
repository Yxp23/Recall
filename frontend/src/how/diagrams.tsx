import { useEffect, useRef, useState, type ReactNode } from "react";
import { animate, motion, useInView, useMotionValue, useTransform } from "motion/react";
import { Check, Play } from "lucide-react";
import { Icon } from "../ui/Icon";
import { spring } from "../ui/motion";

// ── shared bits ──────────────────────────────────────────────────────────────

/** Fades and rises into place as it scrolls into view. */
export function Reveal({ children, delay = 0, className = "" }: { children: ReactNode; delay?: number; className?: string }) {
  return (
    <motion.div
      className={className}
      initial={{ opacity: 0, y: 28 }}
      whileInView={{ opacity: 1, y: 0 }}
      viewport={{ once: true, margin: "-80px" }}
      transition={{ ...spring.gentle, delay }}
    >
      {children}
    </motion.div>
  );
}

export function Section({ kicker, title, children }: { kicker: string; title: ReactNode; children: ReactNode }) {
  return (
    <section className="py-16 md:py-24">
      <Reveal>
        <div className="t-subhead mb-3 font-semibold text-label-2">{kicker}</div>
        <h2 className="t-section max-w-[20ch]">{title}</h2>
      </Reveal>
      <div className="mt-10 md:mt-14">{children}</div>
    </section>
  );
}

export function Chip({ children, strong = false, wrap = false }: { children: ReactNode; strong?: boolean; wrap?: boolean }) {
  return (
    <span className={`t-footnote tabular inline-block ${wrap ? "" : "whitespace-nowrap"} rounded-full px-3 py-1 ${strong ? "bg-label text-on-label" : "bg-fill text-label"}`}>
      {children}
    </span>
  );
}

export function Node({ title, lines, className = "" }: { title: string; lines?: string[]; className?: string }) {
  return (
    <div className={`flex min-w-[140px] flex-1 flex-col items-center justify-center rounded-r20 bg-card p-4 text-center ${className}`}>
      <div className="t-headline">{title}</div>
      {lines?.map((l) => (
        <div key={l} className="t-footnote text-label-2">
          {l}
        </div>
      ))}
    </div>
  );
}

export function Arrow() {
  return (
    <div className="flow-arrow" aria-hidden>
      {[0, 0.7, 1.4].map((d) => (
        <span key={d} className="flow-dot" style={{ ["--d" as string]: `${d}s` }} />
      ))}
    </div>
  );
}

/** Counts up to a number once it scrolls into view. */
export function CountTo({ to, prefix = "", suffix = "", decimals = 0 }: { to: number; prefix?: string; suffix?: string; decimals?: number }) {
  const ref = useRef<HTMLSpanElement>(null);
  const inView = useInView(ref, { once: true, margin: "-60px" });
  const mv = useMotionValue(0);
  const text = useTransform(mv, (v) => `${prefix}${v.toFixed(decimals)}${suffix}`);
  useEffect(() => {
    if (!inView) return;
    const c = animate(mv, to, { duration: 1.4, ease: "easeOut" });
    return () => {
      c.stop();
    };
  }, [inView, to, mv]);
  return <motion.span ref={ref}>{text}</motion.span>;
}

// ── medication safety: a timeline that plays when it scrolls into view ───────

export interface Step {
  icon: Parameters<typeof Icon>[0]["as"];
  title: string;
  body: string;
  chip: string;
}

export function Timeline({ steps }: { steps: Step[] }) {
  const ref = useRef<HTMLDivElement>(null);
  const inView = useInView(ref, { once: true, margin: "-120px" });
  const STEP_S = 0.5;
  return (
    <div ref={ref}>
      {/* progress line: fills as the steps light up */}
      <div className="mb-4 hidden h-1 overflow-hidden rounded-full bg-fill md:block">
        <motion.div
          className="h-full origin-left rounded-full bg-label"
          initial={{ scaleX: 0 }}
          animate={inView ? { scaleX: 1 } : {}}
          transition={{ duration: STEP_S * steps.length, ease: "linear" }}
        />
      </div>
      <div className="grid gap-4 md:grid-cols-5">
        {steps.map((s, i) => (
          <motion.div
            key={s.title}
            initial={{ opacity: 0, y: 24, scale: 0.96 }}
            animate={inView ? { opacity: 1, y: 0, scale: 1 } : {}}
            transition={{ ...spring.bouncy, delay: i * STEP_S }}
            className="flex flex-col rounded-r28 bg-card p-5"
          >
            <div className="mb-4 flex items-center justify-between">
              <span className="t-footnote tabular grid h-7 w-7 place-items-center rounded-full bg-label font-semibold text-on-label">{i + 1}</span>
              <Icon as={s.icon} className="text-label-2" />
            </div>
            <h3 className="t-headline">{s.title}</h3>
            <p className="t-subhead mt-1 text-label-2">{s.body}</p>
            <div className="mt-auto pt-4">
              <Chip wrap>{s.chip}</Chip>
            </div>
          </motion.div>
        ))}
      </div>
    </div>
  );
}

// ── memory for things: the state machine, with the active state walking through it ──

const WALK: { node: string; event: string }[] = [
  { node: "On surface", event: "resting in a zone" },
  { node: "In hand", event: "picked_up" },
  { node: "Worn", event: "worn (glasses at the face)" },
  { node: "In hand", event: "taken_off" },
  { node: "Went into basket", event: "went_into · basket" },
  { node: "On surface", event: "reappeared" },
  { node: "In hand", event: "picked_up" },
  { node: "Last seen worn", event: "vanished while worn" },
];

export function StateMachine() {
  const [i, setI] = useState(0);
  useEffect(() => {
    const t = window.setInterval(() => setI((x) => (x + 1) % WALK.length), 1700);
    return () => {
      window.clearInterval(t);
    };
  }, []);
  const active = WALK[i].node;
  const cell = (name: string) => (
    <motion.div
      animate={{ scale: active === name ? 1.05 : 1 }}
      transition={spring.bouncy}
      className={`t-subhead rounded-r20 px-4 py-3 text-center font-semibold transition-colors duration-[300ms] ${
        active === name ? "bg-label text-on-label" : "bg-card text-label"
      }`}
    >
      {name}
    </motion.div>
  );
  return (
    <div>
      <div className="flex flex-col items-stretch gap-3 min-[761px]:flex-row min-[761px]:items-center">
        <div className="min-[761px]:w-[170px]">{cell("On surface")}</div>
        <Arrow />
        <div className="min-[761px]:w-[170px]">{cell("In hand")}</div>
        <Arrow />
        <div className="flex flex-col gap-3 min-[761px]:w-[200px]">
          {cell("Worn")}
          {cell("Went into basket")}
          {cell("Last seen worn")}
        </div>
      </div>
      <motion.div key={i} initial={{ opacity: 0, y: 6 }} animate={{ opacity: 1, y: 0 }} transition={spring.default} className="t-footnote tabular mt-4 text-label-2">
        event: <span className="font-semibold text-label">{WALK[i].event}</span>
      </motion.div>
    </div>
  );
}

// ── sees without training: phrases flowing into the detector ─────────────────

export function PhraseFlow({ phrases }: { phrases: string[] }) {
  const list = phrases.length ? phrases : ["pill bottle", "medicine bottle", "wallet", "glasses"];
  const doubled = [...list, ...list];
  return (
    <div className="relative flex items-center overflow-hidden rounded-r28 bg-card p-4">
      <div className="min-w-0 flex-1 overflow-hidden">
        <motion.div
          className="flex w-max gap-2"
          initial={{ x: "-50%" }}
          animate={{ x: "0%" }}
          transition={{ duration: Math.max(12, list.length * 2.4), ease: "linear", repeat: Infinity }}
        >
          {doubled.map((p, i) => (
            <Chip key={`${p}-${i}`}>{p}</Chip>
          ))}
        </motion.div>
      </div>
      <Arrow />
      <div className="ml-3 shrink-0 rounded-r20 bg-label px-4 py-3 text-center text-on-label">
        <div className="t-headline">YOLO-World</div>
        <div className="t-caption opacity-70">open vocabulary + CLIP text</div>
      </div>
    </div>
  );
}

/** table → counter, basket → basket; outlines follow the scene when the camera moves. */
export function ZonesDiagram() {
  const sway = { x: [0, 16, -12, 0] };
  const tr = { duration: 6, repeat: Infinity, ease: "easeInOut" as const };
  return (
    <svg viewBox="0 0 400 230" className="w-full rounded-r28 bg-card" role="img" aria-label="Zones are found automatically and follow the camera">
      <motion.g animate={sway} transition={tr}>
        <rect x="40" y="110" width="320" height="90" rx="10" className="fill-fill" />
        <rect x="90" y="70" width="90" height="62" rx="8" className="fill-fill" />
        <circle cx="240" cy="98" r="9" className="fill-label-2" />
      </motion.g>
      <motion.g animate={sway} transition={{ ...tr, delay: 0.3 }}>
        <motion.rect
          x="40" y="110" width="320" height="90" rx="10" fill="none" strokeWidth="2" strokeDasharray="8 6" className="stroke-label"
          animate={{ strokeDashoffset: [0, -28] }} transition={{ duration: 1.2, repeat: Infinity, ease: "linear" }}
        />
        <motion.rect
          x="90" y="70" width="90" height="62" rx="8" fill="none" strokeWidth="2" strokeDasharray="8 6" className="stroke-label"
          animate={{ strokeDashoffset: [0, -28] }} transition={{ duration: 1.2, repeat: Infinity, ease: "linear" }}
        />
        <text x="52" y="132" className="fill-label" fontSize="13" fontWeight="600">counter (table)</text>
        <text x="96" y="62" className="fill-label" fontSize="13" fontWeight="600">basket</text>
      </motion.g>
    </svg>
  );
}

// ── leaving the house: a person crosses the door line ────────────────────────

export function DoorDiagram() {
  const T = 8; // seconds per loop
  const times = [0, 0.12, 0.42, 0.56, 0.9, 1];
  const loop = { duration: T, repeat: Infinity, ease: "linear" as const, times };
  return (
    <div className="rounded-r28 bg-card p-4">
      <svg viewBox="0 0 400 220" className="w-full" role="img" aria-label="A person crossing the door line">
        <rect x="0" y="0" width="400" height="220" rx="14" className="fill-fill" />
        <rect x="200" y="0" width="200" height="220" className="fill-orange" opacity="0.12" />
        <line x1="200" y1="0" x2="200" y2="220" strokeWidth="2" className="stroke-label" />
        <text x="14" y="24" className="fill-label-2" fontSize="12">inside</text>
        <text x="386" y="24" textAnchor="end" className="fill-orange" fontSize="12" fontWeight="600">outside</text>
        <motion.g animate={{ x: [30, 30, 120, 200, 250, 250] }} transition={loop}>
          <rect x="0" y="70" width="60" height="120" rx="8" fill="none" strokeWidth="2.5" className="stroke-green" />
          <text x="6" y="64" className="fill-green" fontSize="12" fontWeight="600">#1</text>
          <circle cx="30" cy="100" r="9" className="fill-label-2" />
          <rect x="18" y="114" width="24" height="52" rx="8" className="fill-label-2" />
        </motion.g>
      </svg>
      <motion.div
        animate={{ opacity: [0, 0, 0, 1, 1, 0], y: [8, 8, 8, 0, 0, 8] }}
        transition={loop}
        className="t-headline mt-3 flex items-center gap-2 rounded-r14 bg-fill px-4 py-3"
      >
        <span className="h-2 w-2 rounded-full bg-orange" /> Left the house · 9:42 PM
      </motion.div>
    </div>
  );
}

// ── proof: a rolling buffer, and the clip cut around an event ────────────────

export function ClipDiagram() {
  const SLOTS = 20; // 10 s at 0.5 s per block
  return (
    <div className="rounded-r28 bg-card p-5">
      <div className="relative">
        <div className="flex gap-1">
          {Array.from({ length: SLOTS }, (_, i) => (
            <motion.span
              key={i}
              className="h-10 flex-1 rounded-[4px] bg-label-3"
              animate={{ opacity: [0.35, 1, 0.35] }}
              transition={{ duration: 2.4, repeat: Infinity, delay: i * 0.07, ease: "easeInOut" }}
            />
          ))}
        </div>
        {/* the clip window: 6 s before the event (block 12), 2 s after */}
        <motion.div
          className="absolute -bottom-2 -top-2 rounded-r10 border-2 border-label"
          style={{ left: `${(2 / SLOTS) * 100}%`, width: `${(16 / SLOTS) * 100}%` }}
          initial={{ opacity: 0, scaleX: 0.6 }}
          whileInView={{ opacity: 1, scaleX: 1 }}
          viewport={{ once: true }}
          transition={spring.gentle}
        />
        <div className="absolute -bottom-2 -top-2 w-0.5 bg-orange" style={{ left: `${(14 / SLOTS) * 100}%` }} />
      </div>
      <div className="t-footnote tabular mt-5 flex justify-between text-label-2">
        <span>10 s rolling buffer, always recording</span>
        <span className="font-semibold text-orange">event</span>
      </div>
      <div className="mt-4 flex flex-wrap items-center gap-2">
        <Chip>6 s before</Chip>
        <Chip>2 s after</Chip>
        <Chip strong>
          <span className="inline-flex items-center gap-1">
            <Icon as={Play} size="small" /> clip.mp4 · H.264
          </span>
        </Chip>
      </div>
    </div>
  );
}

// ── "not sure" side note ─────────────────────────────────────────────────────

export function NotSure({ children }: { children: ReactNode }) {
  return (
    <div className="flex gap-4 rounded-r28 bg-card p-6">
      <span className="grid h-8 w-8 shrink-0 place-items-center rounded-full bg-fill">
        <Icon as={Check} size="small" />
      </span>
      <p className="t-body">{children}</p>
    </div>
  );
}
