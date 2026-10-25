import { createContext, useContext, useEffect, useRef } from "react";
import { motion, type HTMLMotionProps } from "motion/react";
import { subscribePointer } from "./pointer";
import { cardIn } from "./motion";

const Concentric = createContext<{ r: number; pad: number } | null>(null);

type Radius = 10 | 14 | 20 | 28;
type Pad = 0 | 4 | 8 | 12 | 16 | 20 | 24 | 32;

type GlassProps = HTMLMotionProps<"div"> & {
  /** Outer radius token. Defaults to parent glass radius − parent padding (concentric). */
  radius?: Radius;
  pad?: Pad;
  /** Scale 0.92 → 1 spring entrance. */
  appear?: boolean;
  /** Use the SVG lens refraction where supported. */
  lens?: boolean;
};

/** Liquid-glass surface. Colors come from the theme tokens, so it follows light / dark. */
export function Glass({ radius, pad = 0, appear = false, lens = false, style, children, className = "", ...rest }: GlassProps) {
  const parent = useContext(Concentric);
  const r = radius ?? (parent ? Math.max(parent.r - parent.pad, 10) : 28);
  const ref = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    return subscribePointer((x, y) => {
      const b = el.getBoundingClientRect();
      const a = (Math.atan2(y - (b.top + b.height / 2), x - (b.left + b.width / 2)) * 180) / Math.PI + 90;
      el.style.setProperty("--rim-angle", `${a}deg`);
    });
  }, []);

  return (
    <Concentric.Provider value={{ r, pad }}>
      <motion.div
        ref={ref}
        data-lens={lens || undefined}
        {...(appear ? cardIn : {})}
        {...rest}
        className={`glass ${className}`}
        style={{ ["--r" as string]: `${r}px`, ["--pad" as string]: `${pad}px`, ...style }}
      >
        {children}
      </motion.div>
    </Concentric.Provider>
  );
}
