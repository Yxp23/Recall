import type { Transition } from "motion/react";

/** Spring tokens (JS mirror of theme.css; no other spring values appear in components). */
export const spring = {
  default: { type: "spring", stiffness: 380, damping: 30 } as Transition,
  bouncy: { type: "spring", stiffness: 300, damping: 20 } as Transition,
  gentle: { type: "spring", stiffness: 220, damping: 28 } as Transition,
};

/** Fade durations (seconds), matching --dur-fast / --dur-base. */
export const dur = { fast: 0.18, base: 0.3 };

/** Selection-bubble edges: the leading edge moves first, the trailing edge follows. */
export const bubbleLead: Transition = { type: "spring", stiffness: 500, damping: 30 };
export const bubbleTrail: Transition = { type: "spring", stiffness: 260, damping: 26 };

/** Screen / tab switch: crossfade + 8 px slide. */
export const screen = {
  initial: { opacity: 0, y: 8 },
  animate: { opacity: 1, y: 0, transition: spring.default },
  exit: { opacity: 0, y: -8, transition: { duration: dur.fast } },
};

/** Cards appear scale 0.92 → 1 with overshoot. */
export const cardIn = {
  initial: { opacity: 0, scale: 0.92 },
  animate: { opacity: 1, scale: 1, transition: spring.bouncy },
  exit: { opacity: 0, scale: 0.94, transition: { duration: dur.fast } },
};

/** Row stagger on first load only. */
export const STAGGER_S = 0.025;
