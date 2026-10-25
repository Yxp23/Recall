import { useRef } from "react";
import { Drawer } from "vaul";
import { Phone } from "lucide-react";
import { ClipVideo } from "../ui/ClipVideo";
import { Icon } from "../ui/Icon";
import { Press } from "../ui/Press";
import { Group, Row } from "../ui/List";
import { clock } from "../lib/format";
import type { AlertGroup } from "../lib/events";
import { alertCopy } from "./parts";

interface Props {
  group: AlertGroup | null;
  earlierDose: number | null;
  phone: string;
  handled: boolean;
  onHandled: () => void;
  onClose: () => void;
}

export function AlertSheet({ group, earlierDose, phone, handled, onHandled, onClose }: Props) {
  const last = useRef<AlertGroup | null>(null);
  if (group) last.current = group; // keep content while the sheet animates closed
  const g = group ?? last.current;
  const copy = g ? alertCopy(g.latest) : null;

  return (
    <Drawer.Root open={!!group} onOpenChange={(o) => !o && onClose()}>
      <Drawer.Portal>
        <Drawer.Overlay className="fixed inset-0 z-50 bg-scrim" />
        <Drawer.Content className="fixed inset-x-0 bottom-0 z-50 mx-auto max-w-[var(--column-max)] rounded-t-r28 bg-bg px-4 pb-8 pt-2 outline-none">
          <div className="mx-auto mb-4 h-1 w-9 rounded-full bg-label-3" />
          {g && copy && (
            <>
              <Drawer.Title className="t-title2">{copy.title}</Drawer.Title>
              <Drawer.Description className="t-subhead mb-4 text-label-2">
                <span className="tabular">{clock(g.latest.ts)}</span> · {copy.sub}
              </Drawer.Description>

              {g.clipUrl ? (
                <div className="mb-4 aspect-video w-full overflow-hidden rounded-r20">
                  <ClipVideo url={g.clipUrl} controls className="h-full w-full" />
                </div>
              ) : (
                <div className="t-callout mb-4 grid aspect-video w-full place-items-center rounded-r20 bg-fill text-label-2">
                  No clip for this one
                </div>
              )}

              {g.latest.item === "medication" && (
                <div className="mb-4">
                  <Group>
                    <Row
                      title="Earlier dose"
                      value={earlierDose ? <span className="tabular">{clock(earlierDose)} · appears taken</span> : "None observed"}
                    />
                  </Group>
                </div>
              )}

              <div className="flex gap-3">
                {phone ? (
                  <a
                    href={`tel:${phone}`}
                    className="t-headline pressable flex flex-1 items-center justify-center gap-2 rounded-r14 bg-label py-4 text-on-label"
                  >
                    <Icon as={Phone} /> Call Mom
                  </a>
                ) : (
                  <div className="t-headline flex flex-1 items-center justify-center gap-2 rounded-r14 bg-fill py-4 text-label-3">
                    <Icon as={Phone} /> Call Mom
                  </div>
                )}
                <Press
                  onClick={() => {
                    onHandled();
                    onClose();
                  }}
                  disabled={handled}
                  className="t-headline flex-1 rounded-r14 bg-fill py-4 disabled:opacity-50"
                >
                  {handled ? "Handled" : "Mark as handled"}
                </Press>
              </div>
            </>
          )}
        </Drawer.Content>
      </Drawer.Portal>
    </Drawer.Root>
  );
}
