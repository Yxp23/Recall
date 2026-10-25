import { useCallback, useEffect, useRef, useState } from "react";
import { animate, motion, useMotionValue, useReducedMotion, useSpring, useTransform } from "motion/react";
import type { AnimationPlaybackControls } from "motion/react";
import { useDrag } from "@use-gesture/react";
import type { LucideIcon } from "lucide-react";
import { Icon } from "./Icon";
import { bubbleLead, bubbleTrail, spring } from "./motion";

export interface LiquidOption<T extends string> {
  value: T;
  label: string;
  icon?: LucideIcon;
}

interface Props<T extends string> {
  options: LiquidOption<T>[];
  value: T;
  onChange: (v: T) => void;
  variant: "tabs" | "segmented";
  /** tabs only: collapse labels (floating bar shrunk while scrolling down) */
  compact?: boolean;
  ariaLabel: string;
  className?: string;
}

const HOLD_MS = 120; // press-and-hold before the bubble lifts
const DRAG_PX = 4;
const LIFT = 1.15;
const MAX_STRETCH_X = 1.25;
const MIN_STRETCH_Y = 0.9;

/** Asymptotic rubber band: the further past the edge, the less it follows. */
const rubber = (d: number) => 24 * (1 - Math.exp(-d / 48));

/**
 * iOS 26-style liquid selection bubble, shared by the tab bar and every segmented control.
 *  tap   — leading edge moves first, trailing edge follows (it stretches, then settles)
 *  hold  — bubble lifts (scale 1.15, clearer lens-like glass, bigger shadow)
 *  drag  — follows the finger, stretches with velocity, rubber-bands past the ends,
 *          icons highlight live; release snaps to the nearest option and selects it
 */
export function LiquidSelector<T extends string>({
  options,
  value,
  onChange,
  variant,
  compact = false,
  ariaLabel,
  className = "",
}: Props<T>) {
  const reduce = !!useReducedMotion();
  const barRef = useRef<HTMLDivElement>(null);
  const itemRefs = useRef<(HTMLButtonElement | null)[]>([]);
  const rects = useRef<{ left: number; width: number }[]>([]);
  const idx = Math.max(0, options.findIndex((o) => o.value === value));

  const left = useMotionValue(0);
  const right = useMotionValue(0);
  const width = useTransform([left, right], (v: number[]) => Math.max(v[1] - v[0], 0));
  const lift = useMotionValue(1);
  const sxRaw = useMotionValue(1);
  const syRaw = useMotionValue(1);
  const sx = useSpring(sxRaw, { stiffness: 400, damping: 25 });
  const sy = useSpring(syRaw, { stiffness: 400, damping: 25 });
  const originX = useMotionValue(0.5);

  const [lifted, setLifted] = useState(false);
  const [hover, setHover] = useState<number | null>(null);

  const idxRef = useRef(idx);
  idxRef.current = idx;
  const ctl = useRef<AnimationPlaybackControls[]>([]);
  const moved = useRef(false);
  const startX = useRef(0);
  const holdTimer = useRef(0);
  const dragWidth = useRef(0);
  const snapped = useRef<number | null>(null);
  const measured = useRef(false);
  const suppressClick = useRef(false);

  const stopAnims = () => {
    ctl.current.forEach((c) => c.stop());
    ctl.current = [];
  };

  const travelTo = useCallback(
    (i: number, snap: boolean) => {
      const r = rects.current[i];
      if (!r) return;
      const tl = r.left;
      const tr = r.left + r.width;
      stopAnims();
      if (reduce) {
        ctl.current = [animate(left, tl, spring.default), animate(right, tr, spring.default)];
      } else if (snap) {
        ctl.current = [animate(left, tl, spring.bouncy), animate(right, tr, spring.bouncy)];
      } else {
        const rightward = tl >= left.get();
        ctl.current = [
          animate(right, tr, rightward ? bubbleLead : bubbleTrail),
          animate(left, tl, rightward ? bubbleTrail : bubbleLead),
        ];
      }
    },
    [left, right, reduce],
  );

  const measure = useCallback(() => {
    rects.current = itemRefs.current.map((el) => ({ left: el?.offsetLeft ?? 0, width: el?.offsetWidth ?? 0 }));
    const r = rects.current[idxRef.current];
    if (!r || moved.current) return;
    if (!measured.current) {
      measured.current = true;
      left.jump(r.left);
      right.jump(r.left + r.width);
    } else {
      // layout changed under us (scrollbar, label collapse): glide to the new spot rather than jump
      stopAnims();
      ctl.current = [animate(left, r.left, spring.default), animate(right, r.left + r.width, spring.default)];
    }
  }, [left, right]);

  useEffect(() => {
    measure();
    const ro = new ResizeObserver(measure);
    if (barRef.current) ro.observe(barRef.current);
    itemRefs.current.forEach((el) => el && ro.observe(el));
    return () => ro.disconnect();
  }, [measure, options.length, compact]);

  // selection changed (tap, keyboard, or externally): travel there
  useEffect(() => {
    if (snapped.current === idx) {
      snapped.current = null;
      return;
    }
    if (rects.current[idx]) travelTo(idx, false);
  }, [idx, travelTo]);

  useEffect(() => {
    animate(lift, lifted && !reduce ? LIFT : 1, spring.bouncy);
  }, [lifted, reduce, lift]);

  const nearest = (cx: number) => {
    let best = 0;
    let bd = Infinity;
    rects.current.forEach((r, i) => {
      const d = Math.abs(r.left + r.width / 2 - cx);
      if (d < bd) {
        bd = d;
        best = i;
      }
    });
    return best;
  };

  const bind = useDrag(
    ({ first, last, xy, velocity, direction, tap }) => {
      const bar = barRef.current;
      if (!bar || !rects.current.length) return;
      const lx = xy[0] - bar.getBoundingClientRect().left;

      if (first) {
        moved.current = false;
        startX.current = xy[0];
        window.clearTimeout(holdTimer.current);
        holdTimer.current = window.setTimeout(() => setLifted(true), HOLD_MS);
      }
      if (!moved.current && Math.abs(xy[0] - startX.current) > DRAG_PX && !tap) {
        moved.current = true;
        window.clearTimeout(holdTimer.current);
        setLifted(true);
        stopAnims();
        dragWidth.current = Math.max(right.get() - left.get(), 1);
      }

      const first_ = rects.current[0];
      const last_ = rects.current[rects.current.length - 1];
      const lo = first_.left + first_.width / 2;
      const hi = last_.left + last_.width / 2;
      let c = lx;
      if (c < lo) c = lo - rubber(lo - c);
      else if (c > hi) c = hi + rubber(c - hi);

      if (moved.current && !last) {
        const w = dragWidth.current;
        left.set(c - w / 2);
        right.set(c + w / 2);
        if (!reduce) {
          const v = velocity[0];
          sxRaw.set(Math.min(MAX_STRETCH_X, 1 + v * 0.35));
          syRaw.set(Math.max(MIN_STRETCH_Y, 1 - v * 0.1));
          originX.set(direction[0] >= 0 ? 0.2 : 0.8); // stretch toward the direction of travel
        }
        setHover(nearest(c));
      }

      if (last) {
        window.clearTimeout(holdTimer.current);
        setLifted(false);
        sxRaw.set(1);
        syRaw.set(1);
        setHover(null);
        if (moved.current) {
          const i = nearest(c);
          snapped.current = i;
          travelTo(i, true); // release: snap with the bouncy spring
          if (options[i].value !== value) onChange(options[i].value);
          else snapped.current = null;
          moved.current = false;
          suppressClick.current = true; // the click the browser may fire after a drag isn't a tap
          window.setTimeout(() => {
            suppressClick.current = false;
          }, 80);
        } else {
          // pressed and released without dragging: select the option under the pointer
          const i = rects.current.findIndex((r) => lx >= r.left && lx <= r.left + r.width);
          const target = i >= 0 ? i : nearest(lx);
          if (options[target].value !== value) onChange(options[target].value);
        }
      }
    },
    { filterTaps: true, axis: "x" },
  );

  const active = hover ?? idx;
  const isTabs = variant === "tabs";

  return (
    <div
      ref={barRef}
      {...bind()}
      role="tablist"
      aria-label={ariaLabel}
      data-compact={compact || undefined}
      style={{
        touchAction: "pan-y",
        borderRadius: isTabs ? "var(--r-28)" : "var(--r-10)",
        padding: "var(--s-1)",
      }}
      className={`relative flex select-none ${isTabs ? "" : "bg-segment"} ${className}`}
    >
      <motion.div
        aria-hidden
        data-lifted={lifted}
        className="bubble"
        style={{
          x: left,
          width,
          scale: lift,
          scaleX: sx,
          scaleY: sy,
          originX,
          top: "var(--s-1)",
          bottom: "var(--s-1)",
          borderRadius: isTabs ? "calc(var(--r-28) - var(--s-1))" : "calc(var(--r-10) - var(--s-1))",
        }}
      />
      {options.map((o, i) => (
        <button
          key={o.value}
          ref={(el) => {
            itemRefs.current[i] = el;
          }}
          role="tab"
          aria-selected={i === idx}
          tabIndex={i === idx ? 0 : -1}
          onClick={() => {
            // direct selection fallback (mouse click, keyboard, assistive tech); drags are handled above
            if (!suppressClick.current && o.value !== value) onChange(o.value);
          }}
          data-active={i === active || undefined}
          className={`relative z-[1] flex items-center justify-center text-label outline-none ${
            isTabs ? "min-w-[72px] flex-col gap-0 px-3 py-2" : "t-subhead flex-1 px-3 py-1 font-medium"
          }`}
        >
          {isTabs && o.icon && <Icon as={o.icon} size="tab" />}
          <span className={isTabs ? "tab-label t-caption font-medium" : ""}>{o.label}</span>
        </button>
      ))}
    </div>
  );
}
