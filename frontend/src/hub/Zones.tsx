import { useEffect, useRef, useState } from "react";
import { motion } from "motion/react";
import { Glass } from "../ui/Glass";
import { spring } from "../ui/motion";
import { cap } from "../lib/format";
import type { ZoneSnap } from "../lib/types";

/** Maps camera pixels onto an object-cover <img> filling this container. */
function useCover(frame: [number, number]) {
  const ref = useRef<HTMLDivElement>(null);
  const [m, setM] = useState({ s: 1, ox: 0, oy: 0 });
  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    const calc = () => {
      const { width: w, height: h } = el.getBoundingClientRect();
      const s = Math.max(w / frame[0], h / frame[1]);
      setM({ s, ox: (w - frame[0] * s) / 2, oy: (h - frame[1] * s) / 2 });
    };
    calc();
    const ro = new ResizeObserver(calc);
    ro.observe(el);
    return () => ro.disconnect();
  }, [frame]);
  return { ref, ...m };
}

interface Props {
  zones: ZoneSnap[];
  frame: [number, number];
  highlight: string | null; // zone of the most recently moved item
}

export function Zones({ zones, frame, highlight }: Props) {
  const { ref, s, ox, oy } = useCover(frame);
  return (
    <div ref={ref} className="pointer-events-none absolute inset-0 overflow-hidden">
      {zones.map((z, i) => {
        const [x1, y1, x2, y2] = z.box;
        return (
          <motion.div
            key={`${z.name}-${i}`}
            initial={false}
            animate={{ left: ox + x1 * s, top: oy + y1 * s, width: (x2 - x1) * s, height: (y2 - y1) * s }}
            transition={spring.gentle}
            data-hot={highlight === z.name}
            className="zone absolute rounded-r28"
          >
            {/* small glass label pinned to the zone's top-left corner */}
            <Glass radius={28} className="t-footnote absolute left-3 top-3 px-3 py-1 font-medium">
              {cap(z.name)}
            </Glass>
          </motion.div>
        );
      })}
    </div>
  );
}
