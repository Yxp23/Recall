import { useEffect, useMemo, useRef, useState } from "react";
import { AnimatePresence, motion, useMotionValueEvent, useScroll, useTransform } from "motion/react";
import * as Tabs from "@radix-ui/react-tabs";
import * as Dialog from "@radix-ui/react-dialog";
import { Activity, BellRing, House, Settings as Cog, X } from "lucide-react";
import { Glass } from "../ui/Glass";
import { Icon } from "../ui/Icon";
import { ClipVideo } from "../ui/ClipVideo";
import { LiquidSelector } from "../ui/Liquid";
import { Group, Row, SectionHeader } from "../ui/List";
import { Press } from "../ui/Press";
import { Toggle } from "../ui/Toggle";
import { ErrorBoundary } from "../ui/ErrorBoundary";
import { cardIn, dur, screen, spring } from "../ui/motion";
import { chime, unlockAudio } from "../lib/audio";
import { ALERT_EVENTS, isAlert } from "../lib/config";
import { collapseFeed, evKey, FEED_EVENTS, groupAlerts, sinceReset, type AlertGroup } from "../lib/events";
import { clock, sentence } from "../lib/format";
import { buildSlots } from "../lib/schedule";
import { applyAppearance, getAppearance, type Appearance } from "../lib/theme";
import { post, useConfig, useRecall } from "../lib/useRecall";
import type { EventMsg } from "../lib/types";
import { AlertSheet } from "./AlertSheet";
import { AlertList, DoseSection, FeedList, ThingsSection, alertCopy } from "./parts";
import { SummaryCard, type SummaryState } from "./Summary";
import { CAREGIVER_PERSON, DISCLAIMER } from "./copy";

type Tab = "home" | "alerts" | "activity" | "settings";
type Filter = "all" | "items" | "medication" | "alerts";

const TAB_OPTIONS = [
  { value: "home" as const, label: "Home", icon: House },
  { value: "alerts" as const, label: "Alerts", icon: BellRing },
  { value: "activity" as const, label: "Activity", icon: Activity },
  { value: "settings" as const, label: "Settings", icon: Cog },
];
const TITLES: Record<Tab, string> = {
  home: CAREGIVER_PERSON.name,
  alerts: "Alerts",
  activity: "Activity",
  settings: "Settings",
};

function loadHandled(): Set<string> {
  try {
    return new Set(JSON.parse(localStorage.getItem("recall.handled") ?? "[]"));
  } catch {
    return new Set();
  }
}

export default function Care() {
  const cfg = useConfig();
  const [enabled, setEnabled] = useState(false);
  const [tab, setTab] = useState<Tab>("home");
  const [banner, setBanner] = useState<EventMsg | null>(null);
  const [sheet, setSheet] = useState<AlertGroup | null>(null);
  const [player, setPlayer] = useState<EventMsg | null>(null);
  const [filter, setFilter] = useState<Filter>("all");
  const [handled, setHandled] = useState<Set<string>>(loadHandled);
  const [compact, setCompact] = useState(false);
  const [talk, setTalk] = useState(true);
  const [doorAlerts, setDoorAlerts] = useState(true);
  const [confirmReset, setConfirmReset] = useState(false);
  const [appearance, setAppearance] = useState<Appearance>(getAppearance);

  const { snap, events: allEvents } = useRecall((e) => {
    if (isAlert(e)) {
      chime();
      setBanner(e);
    }
  });
  // "Reset demo" clears today's alerts and activity: only events after the latest reset
  const events = useMemo(() => sinceReset(allEvents), [allEvents]);

  useEffect(() => {
    if (snap) setTalk(snap.talk_to_mom);
  }, [snap?.talk_to_mom]); // eslint-disable-line react-hooks/exhaustive-deps
  useEffect(() => {
    if (snap?.door_alerts !== undefined) setDoorAlerts(snap.door_alerts);
  }, [snap?.door_alerts]);

  useEffect(() => {
    if (!banner) return;
    const t = window.setTimeout(() => setBanner(null), 6000);
    return () => window.clearTimeout(t);
  }, [banner]);

  useEffect(() => {
    if (!confirmReset) return;
    const t = window.setTimeout(() => setConfirmReset(false), 2500);
    return () => window.clearTimeout(t);
  }, [confirmReset]);

  // large title collapses into the nav bar; floating tab bar shrinks on scroll down
  const { scrollY } = useScroll();
  const titleOpacity = useTransform(scrollY, [0, 48], [1, 0]);
  const titleY = useTransform(scrollY, [0, 48], [0, -8]);
  // "none" at rest: even filter: blur(0px) makes the browser rasterize the title on its own layer (soft text)
  const titleBlur = useTransform(scrollY, (y) => (y < 1 ? "none" : `blur(${Math.min(8, (y / 48) * 8).toFixed(1)}px)`));
  const barOpacity = useTransform(scrollY, [24, 64], [0, 1]);
  const lastY = useRef(0);
  useMotionValueEvent(scrollY, "change", (y) => {
    if (y > lastY.current + 4 && y > 80) setCompact(true);
    else if (y < lastY.current - 4) setCompact(false);
    lastY.current = y;
  });
  useEffect(() => {
    // block body on purpose: Chrome's scrollTo() returns a Promise, which React would treat as a cleanup function
    window.scrollTo({ top: 0 });
  }, [tab]);

  const groups = useMemo(() => groupAlerts(events), [events]);
  // the latest door event decides the status: "Home" or "Left at 9:42 PM"
  const away = useMemo(() => {
    const last = events.find((e) => e.item === "door"); // events are newest first
    return last && last.event === "left_house" ? last : null;
  }, [events]);
  const isHandled = (g: AlertGroup) => g.members.every((m) => handled.has(evKey(m)));
  const { slots, extras } = useMemo(() => {
    const start = new Date().setHours(0, 0, 0, 0) / 1000;
    const unclear = events.filter((e) => e.event === "dose_unclear" && e.ts >= start).map((e) => e.ts);
    return buildSlots(cfg.schedule, snap?.dose_times ?? [], unclear);
  }, [cfg.schedule, snap?.dose_times, events]);

  // "Last activity 2 min ago" needs a clock that ticks
  const [nowSec, setNowSec] = useState(() => Date.now() / 1000);
  useEffect(() => {
    const t = window.setInterval(() => setNowSec(Date.now() / 1000), 30000);
    return () => {
      window.clearInterval(t);
    };
  }, []);

  const unhandledGroups = groups.filter((g) => !isHandled(g));
  const summary: SummaryState = useMemo(() => {
    if (away) return { kind: "left", at: away.ts };
    const open = unhandledGroups[0];
    if (open) return { kind: "alert", key: open.key, detail: `${alertCopy(open.latest).title} · ${clock(open.latest.ts)}`, ts: open.latest.ts };
    return { kind: "ok", last: events[0]?.ts ?? null };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [away, groups, handled, events]);

  const earlierDose = (g: AlertGroup | null) =>
    g ? (events.find((e) => e.event === "dose_appears_taken" && e.ts < g.members[g.members.length - 1].ts)?.ts ?? null) : null;

  const markHandled = (g: AlertGroup) =>
    setHandled((h) => {
      const n = new Set(h);
      g.members.forEach((m) => n.add(evKey(m)));
      try {
        localStorage.setItem("recall.handled", JSON.stringify([...n]));
      } catch {
        /* storage unavailable */
      }
      return n;
    });

  const feed = useMemo(() => {
    const f = events.filter((e) => {
      if (!FEED_EVENTS.has(e.event)) return false;
      switch (filter) {
        case "items":
          return e.item !== "medication" && e.item !== "door";
        case "medication":
          return e.item === "medication" && !ALERT_EVENTS.has(e.event);
        case "alerts":
          return isAlert(e);
        default:
          return true;
      }
    });
    return collapseFeed(f).slice(0, 80);
  }, [events, filter]);

  const setTalkRemote = (v: boolean) => {
    setTalk(v);
    post("/api/settings", { talk_to_mom: v }).catch(() => setTalk(!v));
  };
  const setDoorAlertsRemote = (v: boolean) => {
    setDoorAlerts(v);
    post("/api/settings", { door_alerts: v }).catch(() => setDoorAlerts(!v));
  };
  const doReset = async () => {
    if (!confirmReset) return setConfirmReset(true);
    setConfirmReset(false);
    try {
      await post("/api/reset");
    } catch {
      /* server unreachable */
    }
  };

  if (!enabled) {
    return (
      <div className="fixed inset-0 grid place-items-center bg-bg px-4">
        <motion.div {...cardIn} className="w-full max-w-[420px] text-center">
          <div className="t-large-title mb-2">Recall</div>
          <p className="t-body mb-8 text-label-2">A little peace of mind. Turn on alerts so this device can chime when something needs a look.</p>
          <Press
            onClick={async () => {
              await unlockAudio();
              setEnabled(true);
            }}
            className="t-headline w-full rounded-r14 bg-label py-4 text-on-label"
          >
            Tap to enable alerts
          </Press>
        </motion.div>
      </div>
    );
  }

  const unhandled = groups.filter((g) => !isHandled(g));

  return (
    <div className="min-h-screen bg-bg text-label">
      {/* nav bar: blur material + compact title fade in as the large title collapses */}
      <div className="navbar">
        <motion.div className="navbar-material" style={{ opacity: barOpacity }} />
        <motion.div style={{ opacity: barOpacity }} className="t-headline relative grid h-11 place-items-center">
          {TITLES[tab]}
        </motion.div>
      </div>

      <Tabs.Root value={tab} onValueChange={(v) => setTab(v as Tab)}>
        <main
          className="mx-auto w-full max-w-[var(--column-max)] px-4 pb-8"
          style={{ paddingTop: "calc(env(safe-area-inset-top, 0px) + var(--row-min))", minHeight: "100vh" }}
        >
          <motion.header style={{ opacity: titleOpacity, y: titleY, filter: titleBlur }} className="pb-6 pt-2">
            <h1 className="t-large-title">{TITLES[tab]}</h1>
            {tab === "home" && (
              <>
                <div className="t-footnote mt-1 flex items-center gap-2 text-label-2">
                  <span className={`h-2 w-2 rounded-full ${away ? "bg-orange" : "bg-green"}`} />
                  {away ? `Left at ${clock(away.ts)}` : "Home"}
                </div>
                <p className="t-subhead text-label-2">A little peace of mind.</p>
              </>
            )}
          </motion.header>

          <AnimatePresence mode="wait" initial={false}>
            <motion.div key={tab} {...screen} className="flex flex-col gap-6">
              <Tabs.Content value="home" forceMount hidden={tab !== "home"} className="flex flex-col gap-6">
                <ErrorBoundary area="home tab" resetKey={tab}>
                {tab === "home" && (
                  <>
                    <SummaryCard state={summary} name={CAREGIVER_PERSON.name} phone={cfg.mom_phone} now={nowSec} />
                    <DoseSection slots={slots} extras={extras} />
                    {unhandled.length > 0 && (
                      <section>
                        <SectionHeader>Needs attention</SectionHeader>
                        <AlertList groups={unhandled.slice(0, 2)} handled={isHandled} onOpen={setSheet} />
                      </section>
                    )}
                    <ThingsSection snap={snap} />
                    <p className="t-footnote px-4 text-center text-label-2">{DISCLAIMER}</p>
                  </>
                )}
                </ErrorBoundary>
              </Tabs.Content>

              <Tabs.Content value="alerts" forceMount hidden={tab !== "alerts"} className="flex flex-col gap-6">
                <ErrorBoundary area="alerts tab" resetKey={tab}>
                {tab === "alerts" &&
                  (groups.length ? (
                    <AlertList groups={groups} handled={isHandled} onOpen={setSheet} />
                  ) : (
                    <p className="t-body py-8 text-center text-label-2">No alerts. All quiet.</p>
                  ))}
                </ErrorBoundary>
              </Tabs.Content>

              <Tabs.Content value="activity" forceMount hidden={tab !== "activity"} className="flex flex-col gap-4">
                <ErrorBoundary area="activity tab" resetKey={tab}>
                {tab === "activity" && (
                  <>
                    <LiquidSelector<Filter>
                      variant="segmented"
                      ariaLabel="Filter activity"
                      value={filter}
                      onChange={setFilter}
                      options={[
                        { value: "all", label: "All" },
                        { value: "items", label: "Items" },
                        { value: "medication", label: "Medication" },
                        { value: "alerts", label: "Alerts" },
                      ]}
                    />
                    <FeedList items={feed} onPlay={setPlayer} />
                    <p className="t-footnote px-4 text-center text-label-2">{DISCLAIMER}</p>
                  </>
                )}
                </ErrorBoundary>
              </Tabs.Content>

              <Tabs.Content value="settings" forceMount hidden={tab !== "settings"} className="flex flex-col gap-6">
                <ErrorBoundary area="settings tab" resetKey={tab}>
                {tab === "settings" && (
                  <>
                    <section>
                      <SectionHeader>Appearance</SectionHeader>
                      <Group>
                        <div className="p-2">
                          <LiquidSelector<Appearance>
                            variant="segmented"
                            ariaLabel="Appearance"
                            value={appearance}
                            onChange={(a) => {
                              setAppearance(a);
                              applyAppearance(a);
                            }}
                            options={[
                              { value: "system", label: "System" },
                              { value: "light", label: "Light" },
                              { value: "dark", label: "Dark" },
                            ]}
                          />
                        </div>
                      </Group>
                    </section>
                    <section>
                      <SectionHeader>Reminders</SectionHeader>
                      <Group>
                        <Row title={`Talk to ${CAREGIVER_PERSON.name}`} trailing={<Toggle checked={talk} onChange={setTalkRemote} label={`Talk to ${CAREGIVER_PERSON.name}`} />} />
                        <Row title="Door alerts" trailing={<Toggle checked={doorAlerts} onChange={setDoorAlertsRemote} label="Door alerts" />} />
                      </Group>
                      <p className="t-footnote px-4 pt-2 text-label-2">Speak reminders out loud at home. With door alerts off, comings and goings are still logged under Activity.</p>
                    </section>
                    <section>
                      <Group>
                        <Row title="How Recall works" chevron onClick={() => { window.location.href = "/how?back=/care"; }} />
                      </Group>
                    </section>
                    <section>
                      <Group>
                        <Row title={confirmReset ? "Tap again to reset" : "Reset demo"} destructive onClick={doReset} />
                      </Group>
                      <p className="t-footnote px-4 pt-2 text-label-2">Clears today&apos;s doses, alerts and activity.</p>
                    </section>
                  </>
                )}
                </ErrorBoundary>
              </Tabs.Content>
            </motion.div>
          </AnimatePresence>
          <div style={{ height: "calc(var(--s-8) * 3)" }} />
        </main>

        {/* floating glass tab bar */}
        <div className="pointer-events-none fixed inset-x-0 bottom-0 z-40 flex justify-center px-4" style={{ paddingBottom: "max(env(safe-area-inset-bottom, 0px), var(--s-4))" }}>
          <motion.div animate={{ scale: compact ? 0.88 : 1, y: compact ? 8 : 0 }} transition={spring.default} className="pointer-events-auto">
            <Glass radius={28} lens>
              <LiquidSelector<Tab> variant="tabs" ariaLabel="Sections" value={tab} onChange={setTab} compact={compact} options={TAB_OPTIONS} />
            </Glass>
          </motion.div>
        </div>
      </Tabs.Root>

      {/* new-alert banner: drops in with a bounce, swipe up to dismiss */}
      <AnimatePresence>
        {banner && (
          <motion.div
            key={evKey(banner)}
            initial={{ y: -140, scaleY: 1.12, scaleX: 0.97, opacity: 0 }}
            animate={{ y: 0, scaleY: 1, scaleX: 1, opacity: 1 }}
            exit={{ y: -140, opacity: 0, transition: { duration: dur.fast } }}
            transition={spring.bouncy}
            drag="y"
            dragConstraints={{ top: 0, bottom: 0 }}
            dragElastic={{ top: 0.5, bottom: 0.05 }}
            onDragEnd={(_, info) => {
              if (info.offset.y < -40 || info.velocity.y < -400) setBanner(null);
            }}
            onTap={() => {
              setBanner(null);
              setTab("alerts");
            }}
            className="fixed inset-x-0 top-0 z-[60] mx-auto w-full max-w-[var(--column-max)] cursor-pointer px-4"
            style={{ paddingTop: "max(env(safe-area-inset-top, 0px), var(--s-3))" }}
          >
            <Glass radius={20} pad={12} lens className="flex items-center gap-3">
              <span className="grid shrink-0 place-items-center text-orange" style={{ width: "var(--icon-box)" }}>
                <Icon as={BellRing} />
              </span>
              <div className="min-w-0">
                <div className="t-headline">{alertCopy(banner).title}</div>
                <div className="t-subhead text-label-2">
                  <span className="tabular">{clock(banner.ts)}</span> · Tap to take a look
                </div>
              </div>
            </Glass>
          </motion.div>
        )}
      </AnimatePresence>

      <AlertSheet
        group={sheet}
        earlierDose={earlierDose(sheet)}
        phone={cfg.mom_phone}
        handled={sheet ? isHandled(sheet) : false}
        onHandled={() => sheet && markHandled(sheet)}
        onClose={() => setSheet(null)}
      />

      {/* clip player from the activity feed */}
      <Dialog.Root open={!!player} onOpenChange={(o) => !o && setPlayer(null)}>
        <Dialog.Portal>
          <Dialog.Overlay className="fixed inset-0 z-[70] bg-scrim" />
          <Dialog.Content asChild>
            <motion.div {...cardIn} className="fixed left-1/2 top-1/2 z-[70] w-[min(92vw,560px)] -translate-x-1/2 -translate-y-1/2 rounded-r20 bg-card p-4 outline-none">
              <div className="mb-3 flex items-center justify-between gap-3">
                <Dialog.Title className="t-headline">{player ? `${sentence(player)} · ${clock(player.ts)}` : ""}</Dialog.Title>
                <Dialog.Close aria-label="Close" className="pressable grid h-8 w-8 shrink-0 place-items-center rounded-full bg-fill">
                  <Icon as={X} size="small" />
                </Dialog.Close>
              </div>
              <Dialog.Description className="sr-only">Short camera clip of this moment.</Dialog.Description>
              {player?.clip_url && <ClipVideo url={player.clip_url} controls className="aspect-video w-full rounded-r14" />}
            </motion.div>
          </Dialog.Content>
        </Dialog.Portal>
      </Dialog.Root>
    </div>
  );
}
