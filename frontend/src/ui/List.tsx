import { useEffect, useRef, useState, type ReactNode } from "react";
import { motion } from "motion/react";
import { ChevronRight, type LucideIcon } from "lucide-react";
import { Icon } from "./Icon";
import { Press } from "./Press";
import { STAGGER_S, spring } from "./motion";

/** Section header above a grouped card: footnote, secondary label, sentence case. */
export function SectionHeader({ children }: { children: ReactNode }) {
  return <h2 className="t-footnote px-4 pb-2 pt-0 text-label-2">{children}</h2>;
}

/** Inset grouped card. `inset="icon"` starts separators after the leading icon column. */
export function Group({ children, className = "", inset }: { children: ReactNode; className?: string; inset?: "icon" }) {
  return (
    <div className={`group ${className}`} data-inset={inset}>
      {children}
    </div>
  );
}

interface RowProps {
  icon?: LucideIcon;
  title: ReactNode;
  subtitle?: ReactNode;
  value?: ReactNode;
  chevron?: boolean;
  onClick?: () => void;
  leading?: ReactNode;
  trailing?: ReactNode;
  destructive?: boolean;
}

/** iOS inset grouped row: 44 pt minimum, value right-aligned in secondary label, chevron if tappable. */
export function Row({ icon, title, subtitle, value, chevron, onClick, leading, trailing, destructive }: RowProps) {
  const body = (
    <>
      {icon && (
        <span className="grid shrink-0 place-items-center text-label-2" style={{ width: "var(--icon-box)" }}>
          <Icon as={icon} />
        </span>
      )}
      {leading}
      <span className="min-w-0 flex-1">
        <span className={`t-body block ${destructive ? "text-red" : ""}`}>{title}</span>
        {subtitle && <span className="t-footnote block text-label-2">{subtitle}</span>}
      </span>
      {value !== undefined && <span className="t-body shrink-0 text-right text-label-2">{value}</span>}
      {trailing}
      {chevron && <Icon as={ChevronRight} size="small" className="shrink-0 text-label-3" />}
    </>
  );
  return onClick ? (
    <Press onClick={onClick} className="row">
      {body}
    </Press>
  ) : (
    <div className="row">{body}</div>
  );
}

// ── first-load stagger ────────────────────────────────────────────────────────

const seen = new Set<string>();

/** True only the first time a list with this id mounts (rows stagger in once). */
export function useFirstLoad(id: string): boolean {
  const [first] = useState(() => !seen.has(id));
  useEffect(() => {
    seen.add(id);
  }, [id]);
  return first;
}

export function Stagger({ i, first, children, className }: { i: number; first: boolean; children: ReactNode; className?: string }) {
  const done = useRef(false);
  return (
    <motion.div
      className={className}
      initial={first && !done.current ? { opacity: 0, y: 8 } : false}
      animate={{ opacity: 1, y: 0 }}
      transition={{ ...spring.default, delay: first ? Math.min(i, 12) * STAGGER_S : 0 }}
      onAnimationComplete={() => {
        done.current = true;
      }}
    >
      {children}
    </motion.div>
  );
}
