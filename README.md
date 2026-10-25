# Recall

**A home camera system for people with dementia and their family caregivers.**
It notices when a dose *appears* taken, speaks up before a second dose, remembers where things went (with video proof), and tells the family when someone leaves the house. It runs entirely on one Mac, offline. Nothing goes to the cloud.

Built solo in 24 hours for HackPSU.

> Medication observations are movement cues, not proof of ingestion. Recall says **"Dose appears taken"**, never "taken", and it says **"Not sure"** instead of guessing.

---

## Contents

1. [What it does](#what-it-does)
2. [How it works](#how-it-works)
3. [Safety rules](#safety-rules)
4. [Architecture](#architecture)
5. [Requirements](#requirements)
6. [Setup](#setup)
7. [Running it](#running-it)
8. [The three screens](#the-three-screens)
9. [Voice](#voice)
10. [Door camera](#door-camera)
11. [Eval mode](#eval-mode)
12. [Configuration](#configuration)
13. [HTTP / WebSocket API](#http--websocket-api)
14. [Project layout](#project-layout)
15. [Troubleshooting](#troubleshooting)
16. [Limitations](#limitations)
17. [Credits](#credits)

---

## What it does

**For Mom (the Hub, a kitchen display):**
- Live camera, dimmed and warm, with soft glass outlines for the surfaces it found ("Counter", "Basket").
- Plain-English toasts as things move: "Glasses → Basket · Just now".
- "Where things are" panel with a check and a location for glasses, wallet and pills, plus "Pills today".
- **Tap to ask**: say "Where are my glasses?" or "Did I take my pills?" and get a huge, simple answer, the video clip, and a spoken reply. Three always-visible tap chips (Glasses / Wallet / Pills) work as a backup.
- If she reaches for pills a second time: a calm full-screen card ("You may have already taken this one. Please check with Sarah."), a chime, and a familiar voice.

**For Sarah (the caregiver app, built for a phone or iPad):**
- A summary card: "Mom is home. All good today." / "Mom may need a check-in." / "Mom left the house at 9:42 PM." (with Call Mom).
- Doses today: a ring and a schedule (Morning / Afternoon / Evening) with "Observed", "Not sure", "No observation" or "Upcoming"; doses far from any slot are listed as "Extra".
- Alert cards with a clip, a bottom sheet with the earlier dose time, Call Mom, and Mark as handled.
- A slide-down banner with a chime the moment a new alert arrives.
- "Her things": where glasses, wallet and pills are, in plain English.
- Activity feed with filters (All / Items / Medication / Alerts), repeats collapsed, clips one tap away.
- Settings: Appearance (System / Light / Dark), Talk to Mom, Door alerts, Reset demo, and a link to How Recall works.

**For judges (Judge view and `/how`):**
- Press **J** on the Hub for a split screen: the annotated camera, a picture-in-picture of the door camera, and a "How it works" panel with the models running, FPS, each item's state, the dose state machine live (hand-to-mouth ratio, mouth visibility), voice timings, test results, and a scrolling event log.
- `/how`: an animated, product-page style explanation of every mechanism, with live data and measured results.

---

## How it works

### Sees without training
**YOLO-World** (open-vocabulary detection with a CLIP text encoder) is told *in plain English* what to look for ("pill bottle", "black leather wallet", "reading glasses"). Add an item by typing its name into `data/items.json`. No photos, labeling or retraining.

### Remembers where things go
Every tracked item (glasses, wallet, medication) is a small state machine:

```
ON surface ⇄ IN HAND ──► WORN            (glasses at the face)
                     ├─► WENT INTO <zone> (basket, counter, shelf)
                     └─► LAST SEEN (worn / in hand / into a zone)
```

- A new state must hold for **1 s** (debounce) before it counts.
- Easier to *keep* tracking than to *start* (hysteresis on detection confidence), so flicker does not reset memory.
- Objects do not move by themselves: a state changes only when a hand is near or the item is seen elsewhere.
- Zone contact point: the item's center, then its base.
- Every event is logged to SQLite and triggers a clip.

### Zones find themselves
At most **one basket, one counter (largest table/desk) and one shelf**. A zone appears only after the same detection has been stable for **3 s** and found on at least 60% of detection runs, and it must be 2%–35% of the frame (a basket-sized thing, not the whole room). Spurious "baskets" seen while the camera moves never become zones. Bowls and trays are deliberately not zone sources.

### Medication safety: five checks before it says a word

1. **Resting.** The bottle was seen at one spot (within 40 px) for 1.5 s of the last 3 s, bridging detection gaps up to 0.7 s.
2. **Confirmed pickup.** It moved more than 60 px while in hand; or it vanished under a hand box and stayed gone for 1.5 s; or it reappeared more than 60 px from where it rested. If it reappears within 40 px of its rest spot within 2 s, it was only occluded and nothing happens. While a pickup is active, new attempts are ignored (loudly) until the bottle is set down.
3. **Hand to mouth.** A wrist, fingertip or palm point (MediaPipe Pose and Hands, or a held bottle) comes within **0.7 × shoulder width** of the mouth and stays for 0.4 s. The scale is clamped to the person's head width, and falls back to it if the shoulders are not visible. The mouth must have been visible within the last 1.5 s.
4. **"Dose appears taken".** Logged with a clip. A 30 s cooldown follows.
5. **Second pickup within 4 h.** `repeat_reach`: chime, then `data/voice/double_dose.m4a` (or the macOS voice if the file does not exist) within about a second, a full-screen card on the Hub, and an alert for the caregiver. A hand-to-mouth after that is logged as `possible_second_dose` for the caregiver and is never counted as another dose.

If the mouth cannot be seen, the attempt becomes `dose_unclear` ("not sure"). An unclear pickup held for 3 s or more still counts toward the next alert.

**False-positive guards for look-alikes** (a wallet once triggered dose alerts):
- Medication boxes must be taller than wide (height/width ≥ 1.1) and under 15% of the frame.
- A medication box overlapping a wallet, book, phone, notebook, or any of a list of look-alikes (air freshener, spray bottle, lotion bottle, hand sanitizer, deodorant, perfume bottle, soap dispenser, candle, salt shaker, mugs and cups, water bottle, snack bag, speaker, remote…) with IoU > 0.4 is dropped.
- Start-tracking confidence 0.30 (keep-tracking 0.15).
- **Pill spot lock:** once the bottle has rested at a spot for 5 s, a medication detection more than 150 px away that is not being carried by a hand is ignored.

### Proof
A rolling **10 s buffer** is always recording. Each event cuts a clip: 6 s before, 2 s after, written as **H.264** (OpenCV `avc1`, or re-encoded with ffmpeg) so it plays in any browser.

### Leaving the house
A second camera (the Mac's built-in webcam) watches the door. A small person detector (`yolov8n`, CPU, separate thread) feeds a nearest-neighbor tracker. Cross the virtual line from inside to outside and stay there for 1 s: `left_house`. Back the other way: `came_back`. 30 s cooldown per direction. The kitchen pipeline never slows down.

### Voice
Tap to ask, speak, stop talking. The browser records with voice-activity detection (stops after 2 s of silence, 8 s max, or a second tap). The server transcribes on the CPU with **faster-whisper** (`base.en`, int8), matches keywords to an intent (no LLM), and answers from templates over the live state and the event log. See [Voice](#voice).

---

## Safety rules

- Wording: **"Dose appears taken"** and **"You may have already taken this one. Please check with Sarah."**
- It never tells her to take or skip a dose.
- When unsure, it says **"Not sure."**
- A false "taken" is worse than a missed one, so dose detection is conservative.
- Must work fully offline (`HF_HUB_OFFLINE=1`).

---

## Architecture

```
 iPhone camera (Continuity)         Mac webcam
   kitchen, 1280x720                door, 640x360
          │                              │
          ▼                              ▼
 ┌─────────────────────────────────┐  ┌──────────────────────┐
 │ Pipeline thread                 │  │ Door thread (CPU)    │
 │  YOLO-World  (Apple GPU / MPS)  │  │  yolov8n person      │
 │  MediaPipe Hands + Pose (+Face) │  │  centroid tracker    │
 │  item state machines, zones     │  │  door-line crossing  │
 │  dose monitor                   │  └──────────┬───────────┘
 │  rolling buffer + clip writer   │             │
 └──────────────┬──────────────────┘             │
                └───────────────┬────────────────┘
                                ▼
                  SQLite event log (data/recall.db)
                                │
                                ▼
               FastAPI  (MJPEG streams · WebSocket · REST · clips)
                  │              │                 │
                  ▼              ▼                 ▼
          Hub  (/)        Caregiver  (/care)    How it works (/how)
          Mom's screen    Sarah's phone/iPad    for judges
```

All of it runs on one Mac. Nothing leaves the home.

**Stack:** Python 3.12 · OpenCV · Ultralytics YOLO-World · MediaPipe · faster-whisper · FastAPI + uvicorn · SQLite · React 19 + Vite + TypeScript · Tailwind CSS v4 · motion · vaul · Radix UI · @use-gesture/react · lucide-react.

---

## Requirements

- **macOS on Apple Silicon.** The Hub's spoken answers use macOS `say` and `afplay`. The GPU path uses Apple's MPS (it falls back to CPU).
- **Python 3.12**
- **Node 20+** (to build the frontend)
- **An iPhone with Continuity Camera** for the kitchen camera (or the Mac webcam, see `RECALL_CAMERA`). A second camera (the Mac webcam) is used for the door; without it, door features switch off and everything else keeps working.
- A modern browser. **Chrome** or Safari. The microphone only works on `localhost` / `127.0.0.1` (browsers block it on other addresses).

---

## Setup

### 1. Python environment

The run script expects a virtualenv at `~/recall-env` (override with `RECALL_PYTHON`, see below).

```bash
python3.12 -m venv ~/recall-env
~/recall-env/bin/pip install -r requirements.txt
```

### 2. Models

`models/` is git-ignored. Download these into it:

| File | Used for | Where from |
|---|---|---|
| `yolov8l-worldv2.pt` | open-vocabulary detection (kitchen) | Ultralytics assets release |
| `yolov8n.pt` | person detection (door) | Ultralytics assets release |
| `hand_landmarker.task` | hand landmarks | MediaPipe model storage |
| `pose_landmarker_full.task` | pose: wrists, mouth, shoulders | MediaPipe model storage |
| `face_landmarker.task` | face box for worn glasses | MediaPipe model storage |

```bash
mkdir -p models && cd models
curl -LO https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8l-worldv2.pt
curl -LO https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8n.pt
curl -LO https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task
curl -LO https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task
curl -LO https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task
cd ..
```

The first YOLO-World run also fetches the CLIP text encoder (one time). Then download the speech model once while online. After this, everything runs offline:

```bash
HF_HUB_OFFLINE=0 ~/recall-env/bin/python -c "from faster_whisper import WhisperModel; WhisperModel('base.en')"
```

### 3. Your data (`data/` is git-ignored)

Create `data/items.json`: what to look for, the dose schedule, and the caregiver's number.

```json
{
  "medication": ["pill bottle", "medicine bottle", "prescription bottle", "fish oil bottle",
                 "supplement bottle", "vitamin bottle", "black supplement bottle with orange label",
                 "blister pack", "pill pack", "pills", "medicine box"],
  "wallet": ["black leather wallet", "folded wallet", "billfold"],
  "schedule": [
    {"label": "Morning",   "time": "08:00"},
    {"label": "Afternoon", "time": "14:00"},
    {"label": "Evening",   "time": "20:00"}
  ],
  "mom_phone": "+1XXXXXXXXXX"
}
```

Everything else under `data/` is created for you (event database, clips, eval results). Optional files:

| File | What it is |
|---|---|
| `data/door.json` | the door line (made by `door_line.py`) |
| `data/voice/double_dose.m4a` | the recorded voice played after the chime on a second dose (falls back to the macOS voice) |
| `data/voice/door.m4a` | played for Mom when she leaves the house (silent if missing) |
| `data/eval.json`, `data/eval_trials.jsonl` | results from Eval mode |

### 4. Build the frontend

```bash
cd frontend && npm install && npm run build && cd ..
```

FastAPI serves `frontend/dist` itself. Rebuild after any frontend change.

---

## Running it

```bash
backend/scripts/serve.sh
```

Then open:

| URL | What |
|---|---|
| `http://localhost:8000/` | **Hub** (Mom's kitchen display) |
| `http://<mac-ip>:8000/care` | **Caregiver app** (open on a phone or iPad on the same network) |
| `http://localhost:8000/how` | **How it works** |
| `http://localhost:8000/debug` | plain debug page: live stream and raw event feed |

The server prints what it is doing (`[dose] …`, `[pill] …`, `[door] …`, `[voice] …`, `[eval] …`), including a reason for every pickup decision and, while waiting for the mouth, one line per frame with the mouth position, the nearest hand point, the scale and the ratio.

On the iPad, tap **"Tap to enable alerts"** once so the browser allows the alert chime.

### The OpenCV preview (no web UI)

```bash
~/recall-env/bin/python backend/scripts/preview.py
```

Keys: `g` glasses · `w` wallet · `m` medication (ask where) · `c` play the last clip · `r` reset doses · `q` quit.

### Calibration scripts

```bash
~/recall-env/bin/python backend/scripts/door_line.py        # click 2 points for the door line, 1 point outside, s to save
~/recall-env/bin/python backend/scripts/calibrate_zones.py  # draw zone polygons by hand (older manual tool)
```

---

## The three screens

**Hub (`/`)**: dark, always. Keyboard: `J` toggles Judge view, `E` toggles Eval mode. The little eye button top-left also toggles Judge view. If the camera is missing, a "Waiting for the camera" card explains why and the server keeps looking.

**Caregiver (`/care`)**: light or dark (Settings → Appearance, or the system setting). Home, Alerts, Activity and Settings tabs, in a floating glass tab bar with a liquid selection bubble you can tap or drag. "Reset demo" clears today's doses, alerts and activity from the caregiver view.

**How it works (`/how`)**: Esc or Close returns to where you came from.

Design: Apple-style tokens (system colors in light and dark, iOS type scale, a 4-point spacing scale, concentric radii), a liquid-glass material with a pointer-following specular rim, spring motion everywhere, and full respect for `prefers-reduced-motion` and `prefers-reduced-transparency`. A short logo animation plays on each load of `/` and `/care`; tap or press a key to skip.

---

## Voice

- Tap **Tap to ask**, speak, and it stops by itself after you finish (2 s of silence, 8 s max), or tap again to send immediately.
- The audio is decoded with the ffmpeg bundled in `imageio-ffmpeg` and transcribed with faster-whisper `base.en` (int8, CPU) with a prompt biased to the questions it knows.
- **Intents** (keywords, no LLM): glasses / spectacles → glasses · wallet / purse → wallet · pills / medicine / tablets → medication · "did I take …" → dose question.
- **Answers** are templates:
  - taken today: "It looks like you took them at 7:47 PM. If you're not sure, check with Sarah."
  - none seen: "I haven't seen you take them today. Please check with Sarah."
  - unclear: "I'm not sure. Please check with Sarah."
  - items: where, and when it was last seen.
  - not understood: "Sorry, I didn't catch that. You can ask where your glasses, wallet or pills are."
- The reply is spoken with macOS `say` (Samantha, rate 170) when **Talk to Mom** is on. While you are asking, the likely answers are already rendered to audio, so the voice starts without delay.
- If the model is not ready or the mic is blocked, a toast says so and the three tap chips keep working. `RECALL_VOICE=0` turns voice off.

Whisper latency depends on how busy the Mac is (the camera pipeline competes for the CPU). The Judge view shows the measured round trip.

---

## Door camera

1. Run `door_line.py`, click two points for the line, then one point on the **outside**, and press `s`. The line is saved to `data/door.json` (fractions of the frame, reloaded automatically).
2. Judge view shows the door feed with the line, the shaded outside and the person boxes with track IDs.
3. **Door alerts** (Settings) is on by default. When off, comings and goings are still logged and visible in Activity, but no alert is raised.
4. The door camera is the Mac's built-in webcam, found by name. It never uses the iPhone. It runs on the CPU so it cannot disturb the GPU used by the kitchen model. `RECALL_DOOR=0` switches it off completely.

---

## Eval mode

A small tool to measure how well it really works during rehearsal. On the Hub, press **E** to show a small orange "EVAL" pill, then press a key right *before* each trial:

| Key | Trial | Passes if (in the next 12 s) |
|---|---|---|
| `1` | real dose | `dose_appears_taken` is logged |
| `2` | fake (pickup, no mouth) | no `dose_appears_taken` |
| `3` | second pickup after a dose | `repeat_reach` is logged (alert time = pickup → chime) |
| `4` | where-question | you press `Y` after the answer |
| `5` | door walk-out | `left_house` is logged |

`N` marks a where-answer wrong; `R` clears all trials. Each trial is judged only from the event log, appended to `data/eval_trials.jsonl`, and `data/eval.json` is rewritten. Failures get a reason pulled from the log ("mouth not visible at reach", "pickup not confirmed", …). The headline numbers appear on `/how` (with a per-test table) and as a "Tested" card in Judge view.

---

## Configuration

Environment variables (all optional):

| Variable | Default | Effect |
|---|---|---|
| `RECALL_CAMERA` | auto | force an OpenCV camera index (e.g. `RECALL_CAMERA=0` uses the Mac webcam as the kitchen camera) |
| `RECALL_CAMERA_NAME` | `iPhone` | match a different camera name for the kitchen |
| `RECALL_DOOR` | `1` | `0` disables the door camera |
| `RECALL_VOICE` | `1` | `0` disables push-to-talk (tap chips still work) |
| `RECALL_WHISPER` | `base.en` | speech model: `tiny.en` (faster), `small` (more accurate) |
| `RECALL_PYTHON` | `~/recall-env/bin/python` | interpreter used by `serve.sh` |
| `HF_HUB_OFFLINE` | `1` (set by `serve.sh`) | keeps model loading offline |

The kitchen camera is chosen by name: the app asks macOS which cameras exist, and only opens the iPhone if it is really connected, so it never silently falls back to the Mac webcam. While the iPhone is missing it keeps looking and the Hub shows why.

Key thresholds live as named constants at the top of `backend/recall/events/dose.py` and `backend/recall/pipeline.py` (hand-to-mouth ratio, dwell, resting and pickup distances, zone stability, size limits, confidence bars).

---

## HTTP / WebSocket API

| Endpoint | |
|---|---|
| `GET /stream/annotated.mjpg` | kitchen camera with boxes for glasses/wallet/medication, zone outlines and the hand/pose skeleton |
| `GET /stream/raw.mjpg` | kitchen camera, untouched |
| `GET /stream/door.mjpg` | door camera with the line and person boxes |
| `WS /ws` | full state on connect, a state snapshot every 1 s, and every event as JSON |
| `GET /api/state` | the current snapshot (item states, zones, doses, door, voice, dose-debug) |
| `GET /api/events?limit=50` | recent events from the log |
| `GET /api/config` | dose schedule, caregiver number, detector vocabulary |
| `GET /clips/{file}` | event clips (H.264 MP4) |
| `POST /api/ask {item}` | "where is it?" → `{answer, clip_url}` |
| `POST /api/voice` | raw audio → `{transcript, answer, clip_url, timing}` |
| `POST /api/reset` | reset today's doses |
| `POST /api/settings {talk_to_mom?, door_alerts?}` | settings |
| `POST /api/eval/start`, `/mark`, `/reset`; `GET /api/eval/status`, `GET /api/eval` | Eval mode |

Event JSON: `{type, item, event, zone, ts, clip_url, message, silent}`. Events include `picked_up`, `placed`, `went_into`, `worn`, `taken_off`, `reappeared`, `dose_appears_taken`, `dose_unclear`, `repeat_reach`, `possible_second_dose`, `doses_reset`, `left_house`, `came_back`.

---

## Project layout

```
backend/
  recall/
    pipeline.py        camera, YOLO-World, zones, trackers, dose wiring, annotated frame
    events/
      dose.py          the dose monitor (pickup → hand-to-mouth → taken / repeat / second)
      tracker.py       per-item state machine
      answers.py       "where is it?" templates
      clips.py         rolling buffer + H.264 clip writer
      store.py         SQLite event log
    vision/hands.py    MediaPipe Hands + Pose (+ Face) wrapper and overlay drawing
    door.py            door camera thread, person tracker, crossing logic
    door_geometry.py   door line math and drawing
    cameras.py         which cameras macOS has (door camera lookup)
    voice.py           whisper, intent keywords, answer templates
    audio.py           afplay / say, cached speech
    evaltool.py        Eval mode
    api/server.py      FastAPI app
  scripts/             serve.sh · preview.py · door_line.py · calibrate_zones.py
frontend/src/
  hub/                 Hub, Judge panels, Ask capsule, zones overlay, eval pill
  care/                caregiver app (summary, doses, alerts, activity, settings)
  how/                 the How it works page and its animated diagrams
  ui/                  Glass, liquid selector, list rows, intro, error boundary…
  lib/                 types, formatting, schedule logic, WebSocket and voice hooks
  theme.css            all design tokens (color, type, spacing, radii)
```

---

## Troubleshooting

- **The Mac webcam is used instead of the iPhone.** The server only uses the iPhone when macOS lists it as a camera. Unlock the iPhone, keep it near the Mac, check Settings → General → AirPlay & Continuity → Continuity Camera, or plug it in with a cable. Check `system_profiler SPCameraDataType`. `RECALL_CAMERA=0` forces the Mac webcam on purpose.
- **The iPhone camera is black.** Unlock it, keep it upright with the rear lens uncovered, and close any other app using it. The server prints a warning when a camera streams only black.
- **The microphone does not work.** Open the Hub at `http://localhost:8000/`, not a LAN address, and allow the microphone for the page.
- **Voice says it is not ready.** The model loads in the background after startup; check the Voice line in Judge view. Make sure the speech model was downloaded once (see Setup). `RECALL_WHISPER=tiny.en` is faster.
- **Dose detection misses.** Sit facing the camera with head and shoulders in view, lift the bottle clearly to your lips for about half a second, and keep your hand visible. Read the `[dose]` lines in the server terminal: each pickup decision and each frame while waiting for the mouth is explained. Eval mode gives you a pass rate and the reasons.
- **Too many zones / the wrong zone.** Zones need 3 s of stable detection; move the camera, wait a few seconds, and the zones settle. Add synonyms to the detector vocabulary if needed.
- **`/how` or `/care` shows "Frontend not built".** Run `npm run build` in `frontend/`.
- **The dev server for the frontend:** `cd frontend && npm run dev` proxies `/api`, `/stream`, `/clips` and `/ws` to `localhost:8000`.

---

## Limitations

- macOS only, and tuned for one person in front of one camera.
- Dose detection reads movement, not swallowing. It cannot tell a pill from a sip of water, and it is deliberately conservative. A missed dose becomes "not sure"; a false "taken" is the failure it works hardest to avoid.
- Pose tracking can lose the hand when it covers the lips, and a busy Mac makes both detection and speech slower.
- Look-alike filtering is heuristic; an unusual object that looks like a pill bottle can still be mistaken for one until it fails the shape, overlap or spot-lock checks.
- There is no login. Anyone on the same network can open the streams, so use it on a trusted network.
- There is no automated test suite yet; behavior was verified with simulated sequences, rehearsal logs and Eval mode.
- The Ultralytics models are AGPL-3.0, so check the licensing before shipping a product built on them.

---

## Credits

Built with [Ultralytics YOLO-World / YOLOv8](https://github.com/ultralytics/ultralytics), [MediaPipe](https://developers.google.com/mediapipe), [faster-whisper](https://github.com/SYSTRAN/faster-whisper), [OpenCV](https://opencv.org), [FastAPI](https://fastapi.tiangolo.com), [React](https://react.dev), [Vite](https://vitejs.dev), [Tailwind CSS](https://tailwindcss.com), [motion](https://motion.dev), [vaul](https://github.com/emilkowalski/vaul), [Radix UI](https://www.radix-ui.com), [@use-gesture](https://use-gesture.netlify.app) and [lucide](https://lucide.dev).

Made solo by Yash ([@Yxp23](https://github.com/Yxp23)) for HackPSU.
