import { useCallback, useEffect, useRef } from "react";
import { useMotionValue } from "motion/react";

/** Best container the browser can record: webm/opus on Chrome, mp4 on Safari. */
function pickMime(): string | undefined {
  const MR = window.MediaRecorder;
  if (!MR) return undefined;
  return ["audio/webm;codecs=opus", "audio/webm", "audio/mp4"].find((m) => MR.isTypeSupported(m));
}

// Voice-activity detection: stop after SILENCE_S of quiet once she has spoken.
//  - loudness is a decaying peak envelope, so the gaps between words and syllables never count as silence
//  - the room's noise floor is a slowly-rising minimum (not a fixed first-0.4-s sample, which swallowed
//    the start of her sentence when she began talking straight away)
//  - it needs at least MIN_VOICED_S of real speech before silence may end the recording
const SILENCE_S = 2.0; // people pause mid-question ("where are my… glasses?"): wait longer, and a 2nd tap sends at once
const MAX_S = 8;
const NO_SPEECH_S = 5; // nobody said anything: give up rather than record the room
const MIN_VOICED_S = 0.3;
const ENV_DECAY_S = 0.3; // how long a loud moment keeps counting as "still talking"
const MIN_THRESHOLD = 0.004;
export const WAVE_BARS = 32;

export type StopReason = "silence" | "max" | "tap" | "no speech";

export interface Recording {
  blob: Blob;
  ms: number; // how long we recorded
  speech: boolean; // some speech was heard
  reason: StopReason;
  peak: number; // loudest moment (for tuning)
  floor: number; // estimated room noise
}

/**
 * Tap-to-talk recorder with automatic stop. start(onAuto) begins; the recorder calls onAuto(reason)
 * itself when it hears 1.2 s of silence after speech, hits 8 s, or hears nothing for 4 s; call
 * stop("tap") yourself for a second tap. `level` (0..1) and `wave` (recent loudness, for the
 * waveform) update live. The mic stream is kept briefly so the next question starts instantly.
 */
export function useVoice() {
  const level = useMotionValue(0);
  const wave = useRef<Float32Array>(new Float32Array(WAVE_BARS));
  const stream = useRef<MediaStream | null>(null);
  const rec = useRef<MediaRecorder | null>(null);
  const chunks = useRef<Blob[]>([]);
  const t0 = useRef(0);
  const raf = useRef(0);
  const ctx = useRef<AudioContext | null>(null);
  const meterOff = useRef<() => void>(() => {});
  const idle = useRef(0);
  const auto = useRef<((r: StopReason) => void) | null>(null);
  const heard = useRef(false);
  const stats = useRef({ peak: 0, floor: 0 });

  const releaseMic = useCallback(() => {
    stream.current?.getTracks().forEach((t) => t.stop());
    stream.current = null;
  }, []);

  const start = useCallback(
    async (onAuto: (reason: StopReason) => void): Promise<boolean> => {
      try {
        window.clearTimeout(idle.current);
        if (!navigator.mediaDevices?.getUserMedia) return false; // not a secure origin
        if (!stream.current || !stream.current.active) {
          stream.current = await navigator.mediaDevices.getUserMedia({
            audio: { echoCancellation: true, noiseSuppression: true, autoGainControl: true }, // AGC lifts a quiet voice
          });
        }
        const mime = pickMime();
        const r = new MediaRecorder(stream.current, mime ? { mimeType: mime } : undefined);
        chunks.current = [];
        r.ondataavailable = (e) => {
          if (e.data.size) chunks.current.push(e.data);
        };
        r.start();
        rec.current = r;
        t0.current = performance.now();
        heard.current = false;
        auto.current = onAuto;
        wave.current.fill(0);

        ctx.current ??= new AudioContext();
        void ctx.current.resume();
        const src = ctx.current.createMediaStreamSource(stream.current);
        const an = ctx.current.createAnalyser();
        an.fftSize = 1024;
        src.connect(an);
        const buf = new Float32Array(an.fftSize);
        let env = 0; // decaying peak of the loudness
        let floor = -1; // room noise: follows the quietest moments, creeps up slowly
        let voicedMs = 0;
        let lastVoice = performance.now();
        let prev = performance.now();
        stats.current = { peak: 0, floor: 0 };
        const tick = () => {
          an.getFloatTimeDomainData(buf);
          let sum = 0;
          for (const v of buf) sum += v * v;
          const rms = Math.sqrt(sum / buf.length);
          level.set(Math.min(1, rms * 7));
          wave.current.copyWithin(0, 1);
          wave.current[WAVE_BARS - 1] = Math.min(1, rms * 7);

          const now = performance.now();
          const dt = now - prev;
          prev = now;
          const elapsed = (now - t0.current) / 1000;
          env = Math.max(rms, env * Math.exp(-dt / (ENV_DECAY_S * 1000)));
          floor = floor < 0 ? rms : Math.min(rms, floor * (1 + dt / 20000)); // min-tracker, +5%/s drift
          const threshold = Math.max(MIN_THRESHOLD, floor * 3 + 0.002);
          stats.current = { peak: Math.max(stats.current.peak, rms), floor };

          if (env > threshold) {
            voicedMs += dt;
            lastVoice = now;
            if (voicedMs >= MIN_VOICED_S * 1000) heard.current = true;
          }
          const fire = (reason: StopReason) => {
            const cb = auto.current;
            auto.current = null; // once
            cb?.(reason);
          };
          if (heard.current && now - lastVoice > SILENCE_S * 1000) fire("silence");
          else if (elapsed > MAX_S) fire("max");
          else if (!heard.current && elapsed > NO_SPEECH_S) fire("no speech");
          raf.current = requestAnimationFrame(tick);
        };
        tick();
        meterOff.current = () => {
          cancelAnimationFrame(raf.current);
          src.disconnect();
          level.set(0);
          wave.current.fill(0);
        };
        return true;
      } catch {
        return false; // mic blocked or unavailable
      }
    },
    [level],
  );

  const stop = useCallback(
    (reason: StopReason) =>
      new Promise<Recording | null>((resolve) => {
        const r = rec.current;
        rec.current = null;
        auto.current = null;
        meterOff.current();
        if (!r || r.state === "inactive") return resolve(null);
        r.onstop = () => {
          resolve({
            blob: new Blob(chunks.current, { type: r.mimeType || "audio/webm" }),
            ms: performance.now() - t0.current,
            speech: heard.current,
            reason,
            peak: stats.current.peak,
            floor: stats.current.floor,
          });
          idle.current = window.setTimeout(releaseMic, 45000); // let go of the mic after a quiet spell
        };
        r.stop();
      }),
    [releaseMic],
  );

  useEffect(
    () => () => {
      cancelAnimationFrame(raf.current);
      window.clearTimeout(idle.current);
      releaseMic();
    },
    [releaseMic],
  );

  return { level, wave, start, stop };
}
