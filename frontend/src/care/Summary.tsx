import { AnimatePresence, motion } from "motion/react";
import { Phone } from "lucide-react";
import { Icon } from "../ui/Icon";
import { spring } from "../ui/motion";
import { ago, clock } from "../lib/format";

export type SummaryState =
  | { kind: "left"; at: number }
  | { kind: "alert"; key: string; detail: string; ts: number }
  | { kind: "ok"; last: number | null };

/** The first thing the caregiver reads: one sentence about how things are, one small line behind it. */
export function SummaryCard({ state, name, phone, now }: { state: SummaryState; name: string; phone: string; now: number }) {
  const key = state.kind === "alert" ? `alert-${state.key}` : state.kind;
  const orange = state.kind !== "ok";
  return (
    <section className="rounded-r20 bg-card p-4">
      <AnimatePresence mode="wait" initial={false}>
        <motion.div
          key={key}
          initial={{ opacity: 0, y: 8 }}
          animate={{ opacity: 1, y: 0, transition: spring.gentle }}
          exit={{ opacity: 0, y: -8, transition: { duration: 0.18 } }}
        >
          <h2 className="t-title1">
            {state.kind === "left"
              ? `${name} left the house at ${clock(state.at)}.`
              : state.kind === "alert"
                ? `${name} may need a check-in.`
                : `${name} is home. All good today.`}
          </h2>
          <div className="t-subhead mt-2 flex items-center gap-2 text-label-2">
            <span className={`h-2 w-2 shrink-0 rounded-full ${orange ? "bg-orange" : "bg-green"}`} />
            <span className="tabular">
              {state.kind === "alert"
                ? state.detail
                : state.kind === "left"
                  ? "Door camera"
                  : state.last
                    ? `Last activity ${ago(state.last, now)}`
                    : "No activity yet today"}
            </span>
          </div>
          {state.kind === "left" &&
            (phone ? (
              <a href={`tel:${phone}`} className="t-headline pressable mt-4 flex items-center justify-center gap-2 rounded-r14 bg-label py-3 text-on-label">
                <Icon as={Phone} /> Call {name}
              </a>
            ) : (
              <div className="t-headline mt-4 flex items-center justify-center gap-2 rounded-r14 bg-fill py-3 text-label-3">
                <Icon as={Phone} /> Call {name}
              </div>
            ))}
        </motion.div>
      </AnimatePresence>
    </section>
  );
}
