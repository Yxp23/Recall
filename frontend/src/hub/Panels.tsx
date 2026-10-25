import { AnimatePresence, motion } from "motion/react";
import { ArrowUpRight, Check, Glasses, Pill, Wallet } from "lucide-react";
import { Glass } from "../ui/Glass";
import { Icon } from "../ui/Icon";
import { spring } from "../ui/motion";
import { ITEM_LABEL, clock, isKnown, sentence, where } from "../lib/format";
import type { Slot } from "../lib/schedule";
import type { EventMsg, Snapshot } from "../lib/types";
import { ITEMS } from "../lib/types";

const ICON = { glasses: Glasses, wallet: Wallet, medication: Pill } as const;

export function WherePanel({ snap, slots }: { snap: Snapshot | null; slots: Slot[] }) {
  return (
    <Glass appear lens radius={28} pad={20} className="w-[360px]">
      <h2 className="t-headline mb-3 text-label-2">Where things are</h2>
      <div className="flex flex-col gap-2">
        {ITEMS.map((it) => {
          const s = snap?.items?.[it];
          const known = isKnown(s);
          return (
            <div key={it} className="inner-r flex items-center gap-3 bg-fill p-3">
              <span className="grid h-8 w-8 shrink-0 place-items-center text-label-2">
                <Icon as={ICON[it]} />
              </span>
              <div className="min-w-0 flex-1">
                <div className="t-headline">{ITEM_LABEL[it]}</div>
                <div className="t-subhead truncate text-label-2">{where(it, s, "mom").short}</div>
              </div>
              <motion.span
                animate={{ scale: known ? 1 : 0.8, opacity: known ? 1 : 0.35 }}
                transition={spring.bouncy}
                className={`grid h-6 w-6 shrink-0 place-items-center rounded-full text-thumb ${known ? "bg-hub" : "bg-fill"}`}
              >
                <Check size={14} strokeWidth={2.5} />
              </motion.span>
            </div>
          );
        })}
      </div>

      <h2 className="t-headline mb-3 mt-6 text-label-2">Pills today</h2>
      <div className="flex gap-3">
        {slots.map((sl) => (
          <div key={sl.label} className="flex flex-1 flex-col items-center gap-1">
            <motion.span
              animate={{ scale: sl.status === "observed" ? 1 : 0.92 }}
              transition={spring.bouncy}
              className={`grid h-10 w-10 place-items-center rounded-full ${sl.status === "observed" ? "bg-hub text-thumb" : "border border-label-3"}`}
            >
              {sl.status === "observed" && <Check size={18} strokeWidth={2.5} />}
            </motion.span>
            <span className="t-footnote text-label-2">{sl.label}</span>
            <span className="t-caption tabular text-label-2">{sl.status === "observed" && sl.at ? clock(sl.at) : sl.time}</span>
          </div>
        ))}
      </div>
    </Glass>
  );
}

const MODELS = [
  ["YOLO-World", "open-vocabulary detection"],
  ["MediaPipe Hands", "hand landmarks"],
  ["MediaPipe Pose", "wrists, mouth, shoulders"],
];

export function JudgePanel({ snap, events, onHow }: { snap: Snapshot | null; events: EventMsg[]; onHow: () => void }) {
  const d = snap?.dose;
  return (
    <Glass lens radius={28} pad={20} className="flex h-full w-full flex-col">
      <div className="mb-3 flex items-center justify-between gap-2">
        <h2 className="t-headline text-label-2">How it works</h2>
        <button onClick={onHow} aria-label="Open the How it works page" className="shrink-0">
          <Glass radius={28} className="t-footnote pressable flex items-center gap-1 px-3 py-1 font-medium">
            Open <Icon as={ArrowUpRight} size="small" />
          </Glass>
        </button>
      </div>

      <div className="mb-4 flex flex-col gap-1">
        {MODELS.map(([n, desc]) => (
          <div key={n} className="t-subhead flex items-baseline gap-2">
            <span className="h-2 w-2 shrink-0 rounded-full bg-green" />
            <span className="font-semibold">{n}</span>
            <span className="text-label-2">{desc}</span>
          </div>
        ))}
        <div className="t-subhead flex items-baseline gap-2">
          <span className={`h-2 w-2 shrink-0 rounded-full ${snap?.door?.enabled ? "bg-green" : "bg-label-3"}`} />
          <span className="font-semibold">Door camera</span>
          <span className="tabular text-label-2">
            {snap?.door?.enabled
              ? `${snap.door.fps} FPS · ${snap.door.people} ${snap.door.people === 1 ? "person" : "people"}`
              : "off (no second camera)"}
          </span>
        </div>
        <div className="t-footnote tabular mt-1 text-label-2">{snap?.fps ?? 0} FPS kitchen · fully offline on this Mac</div>
      </div>

      <div className="mb-4 grid grid-cols-3 gap-2">
        {ITEMS.map((it) => (
          <div key={it} className="inner-r bg-fill p-2">
            <div className="t-caption text-label-2">{ITEM_LABEL[it]}</div>
            <div className="t-footnote font-semibold">{snap?.items?.[it]?.label ?? "—"}</div>
          </div>
        ))}
      </div>

      <div className="inner-r t-footnote tabular mb-4 flex flex-col gap-1 bg-fill p-3">
        <div className="flex justify-between">
          <span className="text-label-2">Dose state</span>
          <span className="font-semibold">
            {(d?.state ?? "idle").toUpperCase()}
            {d && d.seconds_left > 0 ? ` · ${d.seconds_left.toFixed(1)}s left` : ""}
          </span>
        </div>
        <div className="flex justify-between">
          <span className="text-label-2">Hand-to-mouth</span>
          <span className="font-semibold">
            {d?.closest_ratio != null ? d.closest_ratio.toFixed(2) : "N/A"} / {d ? d.threshold.toFixed(2) : "0.70"}
          </span>
        </div>
        <div className="flex justify-between">
          <span className="text-label-2">Mouth visible</span>
          <span className="font-semibold">{d?.mouth_visible ? "yes" : "no"}</span>
        </div>
        <div className="flex justify-between">
          <span className="text-label-2">Since last pickup / placed</span>
          <span className="font-semibold">{d?.since_ref != null ? `${d.since_ref.toFixed(1)}s` : "N/A"}</span>
        </div>
        <div className="flex justify-between">
          <span className="text-label-2">Doses · unclear</span>
          <span className="font-semibold">
            {snap?.doses_today ?? 0} · {snap?.unclear ?? 0}
          </span>
        </div>
      </div>

      <div className="t-caption min-h-0 flex-1 overflow-y-auto pr-1 font-mono">
        <AnimatePresence initial={false}>
          {events.slice(0, 40).map((e) => (
            <motion.div
              key={`${e.item}|${e.event}|${e.ts}`}
              layout="position"
              initial={{ opacity: 0, x: 24 }}
              animate={{ opacity: 1, x: 0 }}
              transition={spring.gentle}
              className="flex gap-2 border-b border-separator py-1"
            >
              <span className="tabular text-label-2">{clock(e.ts)}</span>
              <span>{sentence(e)}</span>
            </motion.div>
          ))}
        </AnimatePresence>
      </div>
    </Glass>
  );
}
