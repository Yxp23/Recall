import { useEffect, useState } from "react";
import { Glass } from "../ui/Glass";
import type { EvalData } from "../lib/types";

export interface EvalActive {
  trial: string;
  label: string;
  deadline: number | null; // null = waiting for Y / N (a "where" question)
}

const SHORT: Record<string, string> = { dose: "Dose", fake: "Fake", second: "2nd", where: "Where", door: "Door" };
const ORDER = ["dose", "fake", "second", "where", "door"];

/** Small "EVAL" pill (top-left, next to the Judge button): what to do now, or the running tally. */
export function EvalPill({ active, data }: { active: EvalActive | null; data: EvalData | null }) {
  const [now, setNow] = useState(() => Date.now() / 1000);
  useEffect(() => {
    if (!active?.deadline) return;
    const t = window.setInterval(() => setNow(Date.now() / 1000), 250);
    return () => {
      window.clearInterval(t);
    };
  }, [active?.deadline]);

  const rows = ORDER.filter((k) => data?.breakdown?.[k]);
  let body: string;
  if (active) {
    body =
      active.deadline === null
        ? `${active.label}: ask it, then press Y or N`
        : `${active.label}: go · ${Math.max(0, Math.ceil(active.deadline - now))} s`;
  } else if (rows.length) {
    body = rows.map((k) => `${SHORT[k]} ${data!.breakdown![k].passes}/${data!.breakdown![k].trials}`).join(" · ");
  } else {
    body = "1 dose · 2 fake · 3 second · 4 where · 5 door · R reset";
  }
  return (
    <div className="pointer-events-none fixed left-14 top-4 z-40">
      <Glass radius={28} className="t-footnote tabular flex items-center gap-2 px-3 py-1">
        <span className="rounded-full bg-orange px-2 font-semibold text-on-label">EVAL</span>
        <span className="text-label">{body}</span>
      </Glass>
    </div>
  );
}
