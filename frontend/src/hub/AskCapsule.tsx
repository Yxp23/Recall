import { useState } from "react";
import { AnimatePresence, animate, motion, useMotionValue } from "motion/react";
import { useDrag } from "@use-gesture/react";
import { Glasses, Mic, Pill, Wallet } from "lucide-react";
import { Glass } from "../ui/Glass";
import { Icon } from "../ui/Icon";
import { Press } from "../ui/Press";
import { spring } from "../ui/motion";
import type { ItemName } from "../lib/types";

const CHOICES = [
  { item: "glasses" as ItemName, label: "Glasses", icon: Glasses },
  { item: "wallet" as ItemName, label: "Wallet", icon: Wallet },
  { item: "medication" as ItemName, label: "Pills", icon: Pill },
];

const rubber = (v: number) => Math.max(-24, Math.min(24, v * 0.2));

interface Props {
  open: boolean;
  onToggle: () => void;
  onPick: (item: ItemName) => void;
  /** horizontal band the capsule centers in (the camera area in Judge view) */
  area: { left: number; right: number };
}

/** "Hold to ask" capsule. Press-and-hold stretches it (voice comes later); tap opens the choices. */
export function AskCapsule({ open, onToggle, onPick, area }: Props) {
  const x = useMotionValue(0);
  const y = useMotionValue(0);
  const [pressed, setPressed] = useState(false);

  const bind = useDrag(
    ({ down, movement: [mx, my], tap }) => {
      setPressed(down);
      if (tap) {
        onToggle();
        return;
      }
      if (down) {
        x.set(rubber(mx));
        y.set(rubber(my));
      } else {
        animate(x, 0, spring.bouncy);
        animate(y, 0, spring.bouncy);
      }
    },
    { filterTaps: true },
  );

  return (
    <motion.div
      initial={false}
      animate={area}
      transition={spring.gentle}
      className="pointer-events-none fixed bottom-8 z-30 flex flex-col items-center gap-3"
    >
      <AnimatePresence>
        {open && (
          <Glass key="choices" appear radius={28} pad={8} className="pointer-events-auto flex gap-2">
            {CHOICES.map(({ item, label, icon }) => (
              <Press key={item} onClick={() => onPick(item)} className="inner-r t-title3 flex items-center gap-3 bg-fill px-6 py-4">
                <Icon as={icon} />
                {label}
              </Press>
            ))}
          </Glass>
        )}
      </AnimatePresence>

      <div {...bind()} style={{ touchAction: "none" }} className="pointer-events-auto cursor-pointer select-none">
        <motion.div style={{ x, y }} animate={{ scaleX: pressed ? 1.06 : 1, scaleY: pressed ? 0.95 : 1 }} transition={spring.bouncy}>
          <Glass radius={28} pad={8} lens className="flex items-center gap-4 pr-6">
            <span className="relative grid h-14 w-14 place-items-center rounded-full bg-hub text-thumb">
              {pressed && (
                <motion.span
                  className="absolute inset-0 rounded-full bg-hub"
                  initial={{ scale: 1, opacity: 0.5 }}
                  animate={{ scale: 1.8, opacity: 0 }}
                  transition={{ duration: 1.1, repeat: Infinity }}
                />
              )}
              <Mic size={24} strokeWidth={1.75} className="relative" />
            </span>
            <span className="t-title3">Hold to ask</span>
          </Glass>
        </motion.div>
      </div>
    </motion.div>
  );
}
