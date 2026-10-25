import { useCallback, useEffect, useState, type ReactNode } from "react";
import { motion } from "motion/react";
import { spring } from "./motion";

/** Plays on every load/refresh, except with reduced motion. Decided once, at load. */
const PLAY_INTRO = (() => {
  try {
    return !window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  } catch {
    return true;
  }
})();

// Timeline (seconds): draw 0–1.0, write 0.85–1.7, hold, exit at 2.4 over 0.9 s (done by ~3.3 s).
const HOLD_S = 2.4;  // glyph drawn + word written + a beat to take it in: start the exit
const EXIT_S = 0.9;
const SMOOTH = [0.32, 0.72, 0, 1] as const; // long ease-out: the exit glides instead of snapping
const WORD = "Recall";

const stroke = { strokeLinecap: "round" as const, fill: "none", strokeWidth: 3.5 };

function Glyph() {
  // a clock-like circle with one hand, drawn with a stroke animation
  return (
    <svg viewBox="0 0 56 56" className="h-14 w-14" aria-hidden>
      <motion.circle
        cx="28" cy="28" r="24" {...stroke} className="stroke-label"
        style={{ rotate: -90, transformOrigin: "28px 28px" }}
        initial={{ pathLength: 0 }} animate={{ pathLength: 1 }}
        transition={{ duration: 0.95, ease: [0.65, 0, 0.35, 1] }}
      />
      <motion.line
        x1="28" y1="28" x2="28" y2="14" {...stroke} className="stroke-label"
        initial={{ pathLength: 0 }} animate={{ pathLength: 1 }}
        transition={{ duration: 0.35, delay: 0.8, ease: "easeOut" }}
      />
      <motion.circle
        cx="28" cy="28" r="2.4" className="fill-label"
        initial={{ scale: 0 }} animate={{ scale: 1 }}
        transition={{ ...spring.bouncy, delay: 0.9 }}
        style={{ transformOrigin: "28px 28px" }}
      />
    </svg>
  );
}

function Overlay({ dark, exiting, fast, onTimedExit, onSkip, onDone }: {
  dark: boolean; exiting: boolean; fast: boolean; onTimedExit: () => void; onSkip: () => void; onDone: () => void;
}) {
  useEffect(() => {
    const t = window.setTimeout(onTimedExit, HOLD_S * 1000);
    window.addEventListener("keydown", onSkip); // any key skips
    return () => {
      window.clearTimeout(t);
      window.removeEventListener("keydown", onSkip);
    };
  }, [onTimedExit, onSkip]);

  return (
    <motion.div
      data-theme={dark ? "dark" : undefined}
      onPointerDown={onSkip}   // any tap skips
      initial={{ opacity: 1 }}
      animate={{ opacity: exiting ? 0 : 1 }}
      transition={{ duration: fast ? 0.25 : EXIT_S, ease: fast ? "easeOut" : SMOOTH }}
      onAnimationComplete={() => {
        if (exiting) onDone();
      }}
      className="fixed inset-0 z-[100] grid place-items-center bg-bg text-label"
      style={{ pointerEvents: exiting ? "none" : "auto" }}
    >
      <motion.div
        animate={exiting ? { scale: 0.9, opacity: 0 } : { scale: 1, opacity: 1 }}
        transition={{ duration: fast ? 0.25 : EXIT_S, ease: fast ? "easeOut" : SMOOTH }}
        className="flex items-center gap-4"
      >
        <Glyph />
        <span className="t-display-2 flex font-semibold" aria-label={WORD}>
          {WORD.split("").map((c, i) => (
            <motion.span
              key={i}
              aria-hidden
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ ...spring.default, delay: 0.95 + i * 0.1 }}
            >
              {c}
            </motion.span>
          ))}
        </span>
      </motion.div>
    </motion.div>
  );
}

/**
 * Wraps a page: the page mounts immediately underneath (so the WebSocket connects and alerts
 * land as usual) and only its opacity is held back, never its rendering. Position is offset with
 * `top`, not a transform, so the page's fixed bars keep working while it rises in.
 */
export function IntroShell({ children, dark = false }: { children: ReactNode; dark?: boolean }) {
  const [phase, setPhase] = useState<"play" | "exit" | "done">(PLAY_INTRO ? "play" : "done");
  const [fast, setFast] = useState(false);
  const timedExit = useCallback(() => setPhase((p) => (p === "play" ? "exit" : p)), []);
  const skip = useCallback(() => {
    setFast(true); // a skip fades out quicker than the timed exit
    setPhase((p) => (p === "play" ? "exit" : p));
  }, []);

  return (
    <>
      <motion.div
        style={{ position: "relative" }}
        initial={PLAY_INTRO ? { opacity: 0, top: 24 } : false}
        animate={phase === "play" ? { opacity: 0, top: 24 } : { opacity: 1, top: 0 }}
        transition={spring.gentle}
      >
        {children}
      </motion.div>
      {phase !== "done" && (
        <Overlay dark={dark} exiting={phase === "exit"} fast={fast} onTimedExit={timedExit} onSkip={skip} onDone={() => setPhase("done")} />
      )}
    </>
  );
}
