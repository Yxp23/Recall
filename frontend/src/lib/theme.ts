export type Appearance = "system" | "light" | "dark";
const KEY = "recall.appearance";

export function getAppearance(): Appearance {
  try {
    const v = localStorage.getItem(KEY);
    if (v === "light" || v === "dark") return v;
  } catch {
    /* storage unavailable */
  }
  return "system";
}

/** System follows prefers-color-scheme (no attribute); light/dark set data-theme on <html>. */
export function applyAppearance(a: Appearance): void {
  const el = document.documentElement;
  if (a === "system") el.removeAttribute("data-theme");
  else el.setAttribute("data-theme", a);
  try {
    if (a === "system") localStorage.removeItem(KEY);
    else localStorage.setItem(KEY, a);
  } catch {
    /* storage unavailable */
  }
}
