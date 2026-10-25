// Web Audio helpers. iOS only lets audio start after a user gesture, hence unlock().
let ctx: AudioContext | null = null;

export async function unlockAudio(): Promise<void> {
  try {
    const AC = window.AudioContext ?? (window as unknown as { webkitAudioContext: typeof AudioContext }).webkitAudioContext;
    ctx ??= new AC();
    await ctx.resume();
    const b = ctx.createBuffer(1, 1, 22050); // silent blip completes the unlock
    const s = ctx.createBufferSource();
    s.buffer = b;
    s.connect(ctx.destination);
    s.start(0);
  } catch {
    /* audio unavailable — alerts still show visually */
  }
}

/** Short two-note chime. */
export function chime(): void {
  if (!ctx || ctx.state !== "running") return;
  const t0 = ctx.currentTime;
  [880, 1318.5].forEach((f, i) => {
    const o = ctx!.createOscillator();
    const g = ctx!.createGain();
    o.type = "sine";
    o.frequency.value = f;
    const t = t0 + i * 0.14;
    g.gain.setValueAtTime(0.0001, t);
    g.gain.exponentialRampToValueAtTime(0.25, t + 0.02);
    g.gain.exponentialRampToValueAtTime(0.0001, t + 0.5);
    o.connect(g).connect(ctx!.destination);
    o.start(t);
    o.stop(t + 0.55);
  });
}

/** Speak with the system voice (works offline). Returns a cancel function. */
export function speak(text: string, onEnd: () => void): () => void {
  if (!("speechSynthesis" in window)) {
    onEnd();
    return () => {};
  }
  const u = new SpeechSynthesisUtterance(text);
  u.rate = 0.95;
  u.onend = u.onerror = () => onEnd();
  window.speechSynthesis.cancel();
  window.speechSynthesis.speak(u);
  return () => window.speechSynthesis.cancel();
}
