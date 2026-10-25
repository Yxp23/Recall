import { useEffect, useState } from "react";

interface Props {
  url: string;
  className?: string;
  /** thumbnail: first frame only; otherwise autoplay muted loop */
  thumb?: boolean;
  controls?: boolean;
}

/** Plays a clip; clips arrive a few seconds after their event, so retry while it 404s. */
export function ClipVideo({ url, className = "", thumb, controls }: Props) {
  const [attempt, setAttempt] = useState(0);
  const [ready, setReady] = useState(false);

  useEffect(() => {
    setAttempt(0);
    setReady(false);
  }, [url]);

  return (
    <div className={`relative overflow-hidden bg-fill ${className}`}>
      <video
        key={attempt}
        src={`${url}${url.includes("?") ? "&" : "?"}r=${attempt}${thumb ? "#t=0.2" : ""}`}
        muted
        playsInline
        loop={!thumb}
        autoPlay={!thumb}
        controls={controls}
        preload={thumb ? "metadata" : "auto"}
        onLoadedData={() => setReady(true)}
        onError={() => {
          if (attempt < 40) window.setTimeout(() => setAttempt((a) => a + 1), 2000);
        }}
        className="h-full w-full object-cover transition-opacity duration-[300ms]"
        style={{ opacity: ready ? 1 : 0 }}
      />
      {!ready && attempt > 0 && (
        <div className="t-footnote absolute inset-0 grid place-items-center text-label-2">Clip is on its way…</div>
      )}
    </div>
  );
}
