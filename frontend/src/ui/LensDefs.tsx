import { useEffect } from "react";

// Displacement map: R ramps across the x-edges, G across the y-edges, B constant.
// Mid-grey (128) = no shift, so only the rim of each glass element refracts.
const ramp = (id: string, ch: "r" | "g", vertical: boolean) => {
  const c = (v: number) => (ch === "r" ? `rgb(${v},0,0)` : `rgb(0,${v},0)`);
  return `<linearGradient id='${id}' x1='0' y1='0' x2='${vertical ? 0 : 1}' y2='${vertical ? 1 : 0}'>
    <stop offset='0' stop-color='${c(255)}'/><stop offset='.1' stop-color='${c(128)}'/>
    <stop offset='.9' stop-color='${c(128)}'/><stop offset='1' stop-color='${c(0)}'/></linearGradient>`;
};
const MAP = `<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 100 100' width='100' height='100' preserveAspectRatio='none'>
  <defs>${ramp("x", "r", false)}${ramp("y", "g", true)}</defs>
  <rect width='100' height='100' fill='rgb(0,0,128)'/>
  <rect width='100' height='100' fill='url(#x)' style='mix-blend-mode:screen'/>
  <rect width='100' height='100' fill='url(#y)' style='mix-blend-mode:screen'/></svg>`;
const MAP_URI = `data:image/svg+xml,${encodeURIComponent(MAP)}`;

/** SVG filters only render in backdrop-filter on Chromium; elsewhere we keep plain blur. */
function lensSupported(): boolean {
  const ua = navigator.userAgent;
  if (/iPhone|iPad|iPod|CriOS|FxiOS/.test(ua)) return false;
  if (!/Chrome\/|Chromium\/|Edg\//.test(ua)) return false;
  return CSS.supports("backdrop-filter", "blur(1px) url(#lens-filter)");
}

export function LensDefs() {
  useEffect(() => {
    if (lensSupported()) document.documentElement.classList.add("lens");
  }, []);
  return (
    <svg width="0" height="0" style={{ position: "absolute" }} aria-hidden>
      <defs>
        <filter id="lens-filter" x="0" y="0" width="100%" height="100%" colorInterpolationFilters="sRGB">
          <feImage href={MAP_URI} x="0" y="0" width="100%" height="100%" preserveAspectRatio="none" result="map" />
          <feDisplacementMap in="SourceGraphic" in2="map" scale="22" xChannelSelector="R" yChannelSelector="G" />
        </filter>
      </defs>
    </svg>
  );
}
