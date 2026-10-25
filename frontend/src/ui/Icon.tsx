import type { LucideIcon } from "lucide-react";

/** One icon style everywhere: lucide, 1.75 stroke; 20 px in rows, 24 px in the tab bar. */
const SIZE = { row: 20, tab: 24, small: 16 } as const;
export const STROKE = 1.75;

export function Icon({ as: Cmp, size = "row", className }: { as: LucideIcon; size?: keyof typeof SIZE; className?: string }) {
  return <Cmp size={SIZE[size]} strokeWidth={STROKE} className={className} aria-hidden />;
}
