import { motion } from "motion/react";
import { BellRing, Glasses, Pill, Play, Wallet } from "lucide-react";
import { ClipVideo } from "../ui/ClipVideo";
import { CountUp } from "../ui/CountUp";
import { Icon } from "../ui/Icon";
import { Press } from "../ui/Press";
import { Group, Row, SectionHeader, Stagger, useFirstLoad } from "../ui/List";
import { spring } from "../ui/motion";
import { ChevronRight } from "lucide-react";
import { ITEM_LABEL, clock, sentence, where } from "../lib/format";
import type { AlertGroup, FeedItem } from "../lib/events";
import type { Slot } from "../lib/schedule";
import type { EventMsg, Snapshot } from "../lib/types";
import { ITEMS } from "../lib/types";
import { CAREGIVER_PERSON } from "./copy";

// ── doses ────────────────────────────────────────────────────────────────────

const RING_R = 42;
const RING_C = 2 * Math.PI * RING_R;

export function DoseSection({ slots, extras }: { slots: Slot[]; extras: number[] }) {
  const total = slots.length || 3;
  const n = slots.filter((s) => s.status === "observed").length;
  const first = useFirstLoad("slots");
  return (
    <section>
      <SectionHeader>Doses today</SectionHeader>
      <Group>
        <div className="flex items-center gap-4 p-4">
          <div className="relative h-24 w-24 shrink-0">
            <svg viewBox="0 0 100 100" className="h-full w-full -rotate-90" aria-hidden>
              <circle cx="50" cy="50" r={RING_R} fill="none" strokeWidth="9" className="stroke-fill" />
              <motion.circle
                cx="50"
                cy="50"
                r={RING_R}
                fill="none"
                strokeWidth="9"
                strokeLinecap="round"
                strokeDasharray={RING_C}
                className="stroke-label"
                initial={{ strokeDashoffset: RING_C }}
                animate={{ strokeDashoffset: RING_C * (1 - n / total) }}
                transition={spring.gentle}
              />
            </svg>
            <div className="absolute inset-0 grid place-items-center text-center">
              <div>
                <div className="t-title1 tabular">
                  <CountUp value={n} />
                </div>
                <div className="t-caption -mt-1 text-label-2">of {total}</div>
              </div>
            </div>
          </div>
          <div className="min-w-0">
            <div className="t-headline">
              {n} of {total} doses observed
            </div>
            <div className="t-footnote text-label-2">From what Recall has seen today.</div>
          </div>
        </div>
        {slots.map((s, i) => (
          <Stagger key={s.label} i={i} first={first}>
            <Row
              title={s.label}
              subtitle={<span className="tabular">{s.time}</span>}
              value={
                s.status === "observed" ? (
                  <span className="tabular text-label">Observed{s.at ? ` · ${clock(s.at)}` : ""}</span>
                ) : s.status === "unsure" ? (
                  <span className="text-orange">Not sure</span>
                ) : s.status === "none" ? (
                  "No observation"
                ) : (
                  "Upcoming"
                )
              }
            />
          </Stagger>
        ))}
        {extras.map((t) => (
          <Row
            key={t}
            title={<span className="tabular">Extra · {clock(t)}</span>}
            subtitle="More than 2 hours from a scheduled time"
            value={<span className="text-label">Observed</span>}
          />
        ))}
      </Group>
    </section>
  );
}

// ── alerts ───────────────────────────────────────────────────────────────────

export function alertCopy(e: EventMsg) {
  if (e.event === "left_house") return { title: "Left the house", sub: "Door camera" };
  return e.event === "repeat_reach"
    ? { title: "Reached for pills again", sub: "Recall reminded her" }
    : { title: "Possible second dose", sub: "Please check in" };
}

export function AlertRow({ g, handled, onOpen }: { g: AlertGroup; handled: boolean; onOpen: () => void }) {
  const { title, sub } = alertCopy(g.latest);
  return (
    <Press onClick={onOpen} className="row" style={{ opacity: handled ? 0.5 : 1 }}>
      <span className="grid shrink-0 place-items-center text-orange" style={{ width: "var(--icon-box)" }}>
        <Icon as={BellRing} />
      </span>
      <span className="min-w-0 flex-1">
        <span className="t-headline block">{title}</span>
        <span className="t-subhead block text-label-2">
          {sub} · <span className="tabular">{clock(g.latest.ts)}</span>
          {g.members.length > 1 ? ` · ${g.members.length} reminders` : ""}
        </span>
      </span>
      {g.clipUrl && (
        <div className="h-11 w-11 shrink-0 overflow-hidden rounded-r10">
          <ClipVideo url={g.clipUrl} thumb className="h-full w-full" />
        </div>
      )}
      <Icon as={ChevronRight} size="small" className="shrink-0 text-label-3" />
    </Press>
  );
}

export function AlertList({ groups, handled, onOpen }: { groups: AlertGroup[]; handled: (g: AlertGroup) => boolean; onOpen: (g: AlertGroup) => void }) {
  const first = useFirstLoad("alerts");
  return (
    <Group>
      {groups.map((g, i) => (
        <Stagger key={g.key} i={i} first={first}>
          <AlertRow g={g} handled={handled(g)} onOpen={() => onOpen(g)} />
        </Stagger>
      ))}
    </Group>
  );
}

// ── her things ───────────────────────────────────────────────────────────────

const ICON = { glasses: Glasses, wallet: Wallet, medication: Pill } as const;

export function ThingsSection({ snap }: { snap: Snapshot | null }) {
  const first = useFirstLoad("things");
  return (
    <section>
      <SectionHeader>{`${CAREGIVER_PERSON.possessive} things`}</SectionHeader>
      <Group inset="icon">
        {ITEMS.map((it, i) => (
          <Stagger key={it} i={i} first={first}>
            <Row icon={ICON[it]} title={ITEM_LABEL[it]} value={where(it, snap?.items?.[it], "care").short} />
          </Stagger>
        ))}
      </Group>
    </section>
  );
}

// ── activity ─────────────────────────────────────────────────────────────────

export function FeedList({ items, onPlay }: { items: FeedItem[]; onPlay: (e: EventMsg) => void }) {
  const first = useFirstLoad("feed");
  if (!items.length) return <p className="t-body py-8 text-center text-label-2">Nothing here yet.</p>;
  return (
    <Group>
      {items.map(({ key, e, count }, i) => (
        <Stagger key={key} i={i} first={first}>
          <Row
            title={
              <>
                {sentence(e)}
                {count > 1 && <span className="tabular text-label-2"> ×{count}</span>}
              </>
            }
            subtitle={<span className="tabular">{clock(e.ts)}</span>}
            onClick={e.clip_url ? () => onPlay(e) : undefined}
            trailing={e.clip_url ? <Icon as={Play} className="shrink-0 text-label" /> : undefined}
          />
        </Stagger>
      ))}
    </Group>
  );
}
