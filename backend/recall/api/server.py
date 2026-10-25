"""
FastAPI server: runs the Pipeline in a background thread and exposes
MJPEG streams, a WebSocket event feed, and a small REST API.

Run from the repo root via backend/scripts/serve.sh.
"""
from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
import threading
import time
import traceback
from contextlib import asynccontextmanager
from typing import Optional

os.environ.setdefault("HF_HUB_OFFLINE", "1")

import cv2
import numpy as np
from fastapi import FastAPI, HTTPException, Query, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from recall.pipeline import (
    CLIPS_DIR, LOGICAL_ITEMS, Pipeline, clip_url_for, event_message,
)

ITEMS_JSON   = "data/items.json"
FRONTEND_DIST = Path(__file__).resolve().parents[3] / "frontend" / "dist"

STREAM_FPS   = 10
JPEG_QUALITY = 70
SNAPSHOT_S   = 1.0


# ── WebSocket fan-out ──────────────────────────────────────────────────────────

class Hub:
    def __init__(self) -> None:
        self.loop: Optional[asyncio.AbstractEventLoop] = None
        self._clients: set[asyncio.Queue] = set()

    def subscribe(self) -> asyncio.Queue:
        q: asyncio.Queue = asyncio.Queue(maxsize=200)
        self._clients.add(q)
        return q

    def unsubscribe(self, q: asyncio.Queue) -> None:
        self._clients.discard(q)

    def _publish(self, msg: dict) -> None:
        for q in list(self._clients):
            try:
                q.put_nowait(msg)
            except asyncio.QueueFull:
                pass   # slow client: drop rather than block the pipeline

    def publish_threadsafe(self, msg: dict) -> None:
        if self.loop is not None:
            self.loop.call_soon_threadsafe(self._publish, msg)


# ── pipeline thread ────────────────────────────────────────────────────────────

class Runner:
    def __init__(self, hub: Hub) -> None:
        self.hub = hub
        self.pipe: Optional[Pipeline] = None
        self.error: Optional[str] = "starting"
        self.talk_to_mom = True
        self.frame_id = 0
        self.frames: dict[str, Optional[np.ndarray]] = {"annotated": None, "raw": None}
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="pipeline", daemon=True)
        self._jpeg: dict[str, tuple[int, bytes]] = {}

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=10)

    def _run(self) -> None:
        try:
            pipe = Pipeline(hud=False)   # clean stream: the web UI draws its own status, FPS and dose panel
            pipe.start()
            pipe.set_talk_to_mom(self.talk_to_mom)
            self.pipe = pipe
            self.error = None
        except Exception as e:
            self.error = f"pipeline failed to start: {e}"
            traceback.print_exc()
            return
        try:
            while not self._stop.is_set():
                try:
                    res = pipe.step()
                except Exception:
                    traceback.print_exc()
                    time.sleep(0.1)
                    continue
                if res is None:
                    self.error = "camera read failed"
                    time.sleep(0.5)
                    continue
                self.error = None
                self.frames["annotated"] = res.annotated
                self.frames["raw"]       = res.raw
                self.frame_id += 1
                for ev in res.events:
                    self.hub.publish_threadsafe(ev.to_dict())
        finally:
            pipe.close()

    def snapshot(self) -> dict:
        if self.pipe is None:
            return {"type": "state", "ts": time.time(), "ready": False,
                    "error": self.error, "talk_to_mom": self.talk_to_mom}
        snap = self.pipe.snapshot()
        snap["ready"] = True
        snap["error"] = self.error
        return snap

    def jpeg(self, kind: str) -> Optional[tuple[int, bytes]]:
        """Latest frame of `kind` as JPEG, encoded once per frame across all clients."""
        fid, frame = self.frame_id, self.frames[kind]
        if frame is None:
            return None
        cached = self._jpeg.get(kind)
        if cached and cached[0] == fid:
            return cached
        ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
        if not ok:
            return None
        self._jpeg[kind] = (fid, buf.tobytes())
        return self._jpeg[kind]


hub    = Hub()
runner = Runner(hub)


@asynccontextmanager
async def lifespan(_: FastAPI):
    hub.loop = asyncio.get_running_loop()
    runner.start()
    yield
    await asyncio.get_running_loop().run_in_executor(None, runner.stop)


app = FastAPI(title="Recall", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"],
                   allow_headers=["*"])


def _pipe() -> Pipeline:
    if runner.pipe is None:
        raise HTTPException(503, runner.error or "pipeline not ready")
    return runner.pipe


# ── MJPEG ──────────────────────────────────────────────────────────────────────

async def _mjpeg(kind: str):
    loop = asyncio.get_running_loop()
    last_id = -1
    while True:
        t0 = loop.time()
        item = await loop.run_in_executor(None, runner.jpeg, kind)
        if item is not None and item[0] != last_id:
            last_id, data = item
            yield (b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: "
                   + str(len(data)).encode() + b"\r\n\r\n" + data + b"\r\n")
        await asyncio.sleep(max(0.0, 1.0 / STREAM_FPS - (loop.time() - t0)))


def _stream(kind: str) -> StreamingResponse:
    return StreamingResponse(_mjpeg(kind),
                             media_type="multipart/x-mixed-replace; boundary=frame",
                             headers={"Cache-Control": "no-store"})


@app.get("/stream/annotated.mjpg")
def stream_annotated() -> StreamingResponse:
    return _stream("annotated")


@app.get("/stream/raw.mjpg")
def stream_raw() -> StreamingResponse:
    return _stream("raw")


# ── WebSocket ──────────────────────────────────────────────────────────────────

@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket) -> None:
    await ws.accept()
    loop = asyncio.get_running_loop()
    q = hub.subscribe()
    try:
        await ws.send_json(runner.snapshot())   # full state on connect
        next_snap = loop.time() + SNAPSHOT_S
        while True:
            if loop.time() >= next_snap:
                await ws.send_json(runner.snapshot())
                next_snap = loop.time() + SNAPSHOT_S
            try:
                msg = await asyncio.wait_for(q.get(), max(0.0, next_snap - loop.time()))
            except asyncio.TimeoutError:
                continue
            await ws.send_json(msg)
    except (WebSocketDisconnect, RuntimeError):
        pass
    finally:
        hub.unsubscribe(q)


# ── REST ───────────────────────────────────────────────────────────────────────

def _existing_clip_url(path: Optional[str]) -> Optional[str]:
    return clip_url_for(path) if path and os.path.isfile(path) else None


@app.get("/api/state")
def api_state() -> dict:
    return runner.snapshot()


@app.get("/api/events")
def api_events(limit: int = Query(50, ge=1, le=500)) -> list[dict]:
    rows = _pipe().store.recent_events(limit)
    return [{
        "type":     "event",
        "item":     r["item"],
        "event":    r["event"],
        "zone":     r["zone"],
        "ts":       r["ts"],
        "clip_url": _existing_clip_url(r["clip_path"]),
        "message":  event_message(r["item"], r["event"], r["zone"], r["detail"] or ""),
    } for r in rows]


_DEFAULT_SCHEDULE = [
    {"label": "Morning", "time": "08:00"},
    {"label": "Afternoon", "time": "14:00"},
    {"label": "Evening", "time": "20:00"},
]


@app.get("/api/config")
def api_config() -> dict:
    """Dose schedule + caregiver contact, from data/items.json."""
    try:
        with open(ITEMS_JSON) as f:
            data = json.load(f)
    except (OSError, ValueError):
        data = {}
    return {
        "schedule":  data.get("schedule") or _DEFAULT_SCHEDULE,
        "mom_phone": data.get("mom_phone", ""),
    }


@app.get("/clips/{filename}")
def get_clip(filename: str) -> FileResponse:
    if (filename != os.path.basename(filename) or not filename.endswith(".mp4")
            or ".part" in filename):
        raise HTTPException(404)
    path = os.path.join(CLIPS_DIR, filename)
    if not os.path.isfile(path):
        raise HTTPException(404, "clip not ready or missing")
    return FileResponse(path, media_type="video/mp4")


class AskBody(BaseModel):
    item: str


@app.post("/api/ask")
def api_ask(body: AskBody) -> dict:
    if body.item not in LOGICAL_ITEMS:
        raise HTTPException(400, f"item must be one of {LOGICAL_ITEMS}")
    answer, clip_path = _pipe().ask(body.item)
    return {"answer": answer, "clip_url": _existing_clip_url(clip_path)}


@app.post("/api/reset")
def api_reset() -> dict:
    return {"reset": _pipe().reset()}   # False = ignored (within 1 s of the last reset)


class SettingsBody(BaseModel):
    talk_to_mom: bool


@app.post("/api/settings")
def api_settings(body: SettingsBody) -> dict:
    runner.talk_to_mom = body.talk_to_mom
    if runner.pipe is not None:
        runner.pipe.set_talk_to_mom(body.talk_to_mom)
    return {"talk_to_mom": runner.talk_to_mom}


# ── debug page (testing only) ──────────────────────────────────────────────────

_DEBUG_HTML = """<!doctype html>
<meta charset="utf-8"><title>Recall debug</title>
<style>
 body{font:14px system-ui;margin:16px;background:#111;color:#ddd}
 img{max-width:100%;border-radius:6px}
 #wrap{display:flex;gap:16px;flex-wrap:wrap}
 #left{flex:2 1 480px} #right{flex:1 1 320px}
 pre{background:#1b1b1b;padding:8px;border-radius:6px;overflow:auto;max-height:200px}
 #events{max-height:480px;overflow:auto;font:12px ui-monospace,monospace}
 #events div{padding:3px 0;border-bottom:1px solid #2a2a2a}
 .alert{color:#ffb060} #status{font-size:12px;color:#8c8}
</style>
<h3>Recall debug <span id="status">connecting…</span></h3>
<div id="wrap">
 <div id="left"><img src="/stream/annotated.mjpg"></div>
 <div id="right">
  <b>State</b><pre id="state"></pre>
  <b>Events</b><div id="events"></div>
 </div>
</div>
<script>
const ev = document.getElementById('events'), st = document.getElementById('state'),
      status = document.getElementById('status');
function connect(){
  const ws = new WebSocket((location.protocol==='https:'?'wss://':'ws://')+location.host+'/ws');
  ws.onopen  = () => status.textContent = 'live';
  ws.onclose = () => { status.textContent = 'disconnected – retrying'; setTimeout(connect, 1500); };
  ws.onmessage = m => {
    const d = JSON.parse(m.data);
    if (d.type === 'state') { st.textContent = JSON.stringify(d, null, 1); return; }
    const row = document.createElement('div');
    if (d.event==='repeat_reach' || d.event==='possible_second_dose') row.className='alert';
    const t = new Date(d.ts*1000).toLocaleTimeString();
    row.textContent = t+'  '+d.item+'.'+d.event+(d.zone?' @'+d.zone:'')+'  '+d.message+(d.clip_url?'  ['+d.clip_url+']':'');
    ev.prepend(row);
  };
}
connect();
</script>
"""


@app.get("/debug", response_class=HTMLResponse)
def debug_page() -> str:
    return _DEBUG_HTML


# ── frontend (built by `npm run build` in frontend/) ───────────────────────────
# Hub at /, caregiver at /care — same single-page app, routed client-side.

_NOT_BUILT = HTMLResponse(
    "<h3>Frontend not built</h3><p>Run <code>cd frontend &amp;&amp; npm install &amp;&amp; npm run build</code>, "
    "then reload. (API, streams and <a href='/debug'>/debug</a> work without it.)</p>",
    status_code=503,
)


def _index() -> Response:
    index = FRONTEND_DIST / "index.html"
    if not index.is_file():
        return _NOT_BUILT
    return FileResponse(index, headers={"Cache-Control": "no-cache"})


@app.get("/", include_in_schema=False)
def hub_page() -> Response:
    return _index()


@app.get("/care", include_in_schema=False)
@app.get("/care/", include_in_schema=False)
def care_page() -> Response:
    return _index()


if (FRONTEND_DIST / "assets").is_dir():
    app.mount("/assets", StaticFiles(directory=FRONTEND_DIST / "assets"), name="assets")


@app.get("/favicon.svg", include_in_schema=False)
def favicon() -> Response:
    f = FRONTEND_DIST / "favicon.svg"
    return FileResponse(f) if f.is_file() else Response(status_code=404)
