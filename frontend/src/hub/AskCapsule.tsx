import { useEffect, useRef, type MutableRefObject } from "react";
import { motion } from "motion/react";
import { Glasses, Mic, Pill, Wallet } from "lucide-react";
import { Glass } from "../ui/Glass";
import { Icon } from "../ui/Icon";
import { Press } from "../ui/Press";
import { spring } from "../ui/motion";
import { WAVE_BARS } from "../lib/useVoice";
import type { ItemName } from "../lib/types";

const CHOICES = [
  { item: "glasses" as ItemName, label: "Glasses", icon: Glasses },
  { item: "wallet" as ItemName, label: "Wallet", icon: Wallet },
  { item: "medication" as ItemName, label: "Pills", icon: Pill },
];

export type VoiceState = "idle" | "starting" | "listening" | "thinking";

/** Live waveform: the last moments of loudness, drawn as rounded bars. */
function Waveform({ wave }: { wave: MutableRefObject<Float32Array> }) {
  const ref = useRef<HTMLCanvasElement>(null);
  useEffect(() => {
    const cv = ref.current;
    const g = cv?.getContext("2d");
    if (!cv || !g) return;
    const dpr = window.devicePixelRatio || 1;
    const w = 220;
    const h = 40;
    cv.width = w * dpr;
    cv.height = h * dpr;
    g.scale(dpr, dpr);
    let raf = 0;
    const draw = () => {
      g.clearRect(0, 0, w, h);
      g.fillStyle = getComputedStyle(cv).color;
      const gap = 3;
      const bw = (w - gap * (WAVE_BARS - 1)) / WAVE_BARS;
      for (let i = 0; i < WAVE_BARS; i++) {
        const bh = Math.max(4, wave.current[i] * h);
        g.beginPath();
        g.roundRect(i * (bw + gap), (h - bh) / 2, bw, bh, bw / 2);
        g.fill();
      }
      raf = requestAnimationFrame(draw);
    };
    draw();
    return () => {
      cancelAnimationFrame(raf);
    };
  }, [wave]);
  return <canvas ref={ref} className="text-label" style={{ width: 220, height: 40 }} aria-hidden />;
}

interface Props {
  onPick: (item: ItemName) => void;
  /** horizontal band the capsule centers in (the camera area in Judge view) */
  area: { left: number; right: number };
  voiceState: VoiceState;
  wave: MutableRefObject<Float32Array>;
  /** one tap starts listening; a second tap stops it */
  onTap: () => void;
}

/** "Tap to ask": one big tap target, with the three quick buttons always visible just above it. */
export function AskCapsule({ onPick, area, voiceState, wave, onTap }: Props) {
  const listening = voiceState === "listening";
  const thinking = voiceState === "thinking";
  const starting = voiceState === "starting";

  return (
    <motion.div
      initial={false}
      animate={area}
      transition={spring.gentle}
      className="pointer-events-none fixed bottom-8 z-30 flex flex-col items-center gap-3"
    >
      {/* always-visible quick buttons */}
      <div className="pointer-events-auto flex items-center gap-2">
        <span className="t-footnote text-label-2">or tap:</span>
        {CHOICES.map(({ item, label, icon }) => (
          <Press key={item} onClick={() => onPick(item)} aria-label={label}>
            <Glass radius={28} className="t-subhead flex items-center gap-2 px-4 py-2 font-medium">
              <Icon as={icon} size="small" /> {label}
            </Glass>
          </Press>
        ))}
      </div>

      {/* the capsule grows into a larger pill while it listens */}
      <Press onClick={onTap} aria-label={listening ? "Stop listening" : "Tap to ask"} className="pointer-events-auto" disabled={thinking || starting}>
        <motion.div layout transition={spring.bouncy}>
          <Glass radius={28} pad={8} lens className="flex items-center gap-4 pr-6" style={{ minWidth: listening ? 460 : 280 }}>
            <span className="relative grid h-14 w-14 shrink-0 place-items-center rounded-full bg-hub text-thumb">
              {listening && (
                <motion.span
                  className="absolute inset-0 rounded-full bg-hub"
                  initial={{ scale: 1, opacity: 0.5 }}
                  animate={{ scale: 1.7, opacity: 0 }}
                  transition={{ duration: 1.1, repeat: Infinity }}
                />
              )}
              <Mic size={24} strokeWidth={1.75} className="relative" />
            </span>
            {listening ? (
              <span className="flex flex-1 items-center justify-between gap-4">
                <span className="flex flex-col">
                  <span className="t-title3">Listening…</span>
                  <span className="t-footnote text-label-2">tap when you’re done</span>
                </span>
                <Waveform wave={wave} />
              </span>
            ) : starting ? (
              <span className="t-title3 text-label-2">Getting ready…</span>
            ) : thinking ? (
              <motion.span
                className="t-title3"
                animate={{ opacity: [0.4, 1, 0.4] }}
                transition={{ duration: 1.4, repeat: Infinity, ease: "easeInOut" }}
              >
                Thinking…
              </motion.span>
            ) : (
              <span className="t-title3">Tap to ask</span>
            )}
          </Glass>
        </motion.div>
      </Press>
    </motion.div>
  );
}
