"""
FastAPI application for the Search & Rescue Intelligence Dashboard.

Each camera runs its own capture -> detect -> track -> annotate loop in
its own background thread (CameraWorker), completely independent of every
other camera -- including its own YOLO+ByteTrack instance, so track IDs,
motion history, and priority state never leak between feeds. A
CameraRegistry owns the set of currently-running workers and is how
cameras get added/removed at runtime. FastAPI's async side only ever
reads small per-worker shared state guarded by a lock, and a single
background broadcaster task fans out a combined snapshot to every
connected dashboard over one WebSocket.
"""
from __future__ import annotations

import asyncio
import re
import threading
import time
from contextlib import asynccontextmanager
from typing import Optional

import cv2
import numpy as np
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from . import analytics, config
from .camera import Camera, probe_cameras
from .database import Database, now_iso
from .detector import Detection, PersonDetector
from .tracker import TrackManager

PRIORITY_COLORS_BGR = {
    config.PRIORITY_CRITICAL: (59, 69, 228),
    config.PRIORITY_HIGH: (74, 153, 242),
    config.PRIORITY_MEDIUM: (76, 201, 242),
    config.PRIORITY_LOW: (96, 174, 39),
    config.PRIORITY_UNKNOWN: (153, 140, 140),
}
_TEXT_DARK = (17, 17, 17)
_TEXT_LIGHT = (245, 245, 245)


def _label_text_color(bgr: tuple[int, int, int]) -> tuple[int, int, int]:
    b, g, r = bgr
    luminance = 0.114 * b + 0.587 * g + 0.299 * r
    return _TEXT_DARK if luminance > 150 else _TEXT_LIGHT


# ---------------------------------------------------------------------------
# Shared state between a camera's capture thread and the asyncio/FastAPI side
# ---------------------------------------------------------------------------
class SharedState:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.jpeg_bytes: bytes | None = None
        self.fps: float = 0.0
        self.latency_ms: float = 0.0
        self.connected: bool = False
        self.resolution: tuple[int, int] = (0, 0)

    def update(
        self,
        jpeg_bytes: bytes | None = None,
        fps: float | None = None,
        latency_ms: float | None = None,
        connected: bool | None = None,
        resolution: tuple[int, int] | None = None,
    ) -> None:
        with self._lock:
            if jpeg_bytes is not None:
                self.jpeg_bytes = jpeg_bytes
            if fps is not None:
                self.fps = fps
            if latency_ms is not None:
                self.latency_ms = latency_ms
            if connected is not None:
                self.connected = connected
            if resolution is not None:
                self.resolution = resolution

    def get_frame(self) -> bytes | None:
        with self._lock:
            return self.jpeg_bytes

    def get_metrics(self) -> tuple[float, float, bool, tuple[int, int]]:
        with self._lock:
            return self.fps, self.latency_ms, self.connected, self.resolution


def _draw_annotations(frame: np.ndarray, detections: list[Detection], track_manager: TrackManager) -> np.ndarray:
    annotated = frame
    for det in detections:
        if det.track_id is None:
            continue
        state = track_manager.tracks.get(det.track_id)
        prio = state.priority if state else config.PRIORITY_UNKNOWN
        color = PRIORITY_COLORS_BGR.get(prio, PRIORITY_COLORS_BGR[config.PRIORITY_UNKNOWN])
        text_color = _label_text_color(color)

        x1, y1, x2, y2 = (int(v) for v in det.bbox)
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

        # One compact line: identity, confidence, priority. Deliberately
        # NOT a multi-line dossier baked into the pixels -- the full
        # breakdown (posture, movement, inactivity, score) belongs in the
        # dashboard's subject list where it's actually readable, not
        # crammed onto the video.
        label = f"#{det.track_id}  {det.confidence * 100:.0f}%  {prio}"
        (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)

        if y1 - th - baseline - 6 >= 0:
            label_top, label_bottom = y1 - th - baseline - 6, y1
        else:
            label_top, label_bottom = y2 + 2, y2 + 2 + th + baseline + 6

        cv2.rectangle(annotated, (x1, label_top), (x1 + tw + 10, label_bottom), color, -1)
        cv2.putText(
            annotated, label, (x1 + 5, label_bottom - 6),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 2, cv2.LINE_AA,
        )

    cv2.putText(
        annotated, f"People: {len(track_manager.tracks)}", (14, 30),
        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA,
    )
    return annotated


def _placeholder_jpeg(message: str) -> bytes:
    frame = np.full((config.FRAME_HEIGHT, config.FRAME_WIDTH, 3), 24, dtype=np.uint8)
    cv2.putText(
        frame, message, (24, config.FRAME_HEIGHT // 2),
        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (70, 70, 230), 2, cv2.LINE_AA,
    )
    ok, buf = cv2.imencode(".jpg", frame)
    return buf.tobytes() if ok else b""


def _summarize_priority(counts: dict) -> str:
    for label in (config.PRIORITY_CRITICAL, config.PRIORITY_HIGH, config.PRIORITY_MEDIUM, config.PRIORITY_LOW):
        if counts.get(label, 0) > 0:
            return f"{counts[label]} {label.title()}"
    if counts.get(config.PRIORITY_UNKNOWN, 0) > 0:
        return f"{counts[config.PRIORITY_UNKNOWN]} Unknown"
    return "—"  # em dash: no one currently in frame


# ---------------------------------------------------------------------------
# Per-camera background capture / inference worker
# ---------------------------------------------------------------------------
class CameraWorker:
    def __init__(self, db: Database, row: dict):
        self.db = db
        self.camera_id: str = row["camera_id"]
        self.name: str = row["name"]
        self.source_type: str = row["source_type"]
        self.source_ref: str = str(row["source_ref"])
        self.source: str = row.get("source", config.SOURCE_RGB)
        self.loop_video: bool = bool(int(row.get("loop_video", 1)))

        self.shared = SharedState()
        self.stop_event = threading.Event()
        self.thread: threading.Thread | None = None

        self.camera = Camera(source_type=self.source_type, source_ref=self.source_ref, loop_video=self.loop_video)
        self.detector = PersonDetector()
        self.track_manager = TrackManager(db, camera_id=self.camera_id, source=self.source)

        self._fps_ema = 0.0
        self._was_connected = self.camera.connected

    def start(self) -> None:
        self.thread = threading.Thread(target=self.run, name=f"cam-{self.camera_id}", daemon=True)
        self.thread.start()

    def stop(self, timeout: float = 5) -> None:
        self.stop_event.set()
        if self.thread is not None:
            self.thread.join(timeout=timeout)
        self.camera.release()

    def run(self) -> None:
        print(f"[{self.camera_id}] starting '{self.name}' source={self.source_type}:{self.source_ref} "
              f"device={self.detector.device}")
        while not self.stop_event.is_set():
            loop_start = time.perf_counter()
            ok, frame = self.camera.read()

            if self.camera.connected != self._was_connected:
                if not self.camera.connected:
                    # Feed just dropped: close out active tracks cleanly and
                    # roll to a new epoch, since ByteTrack's ids reset too.
                    self.track_manager.start_new_epoch()
                    self.detector.reset_tracker_state()
                self._was_connected = self.camera.connected

            if not ok or frame is None:
                self.shared.update(
                    jpeg_bytes=_placeholder_jpeg(f"{self.name}: no signal -- retrying..."),
                    connected=False,
                )
                time.sleep(0.2)
                continue

            try:
                detections, latency_ms = self.detector.track(frame)
            except Exception as exc:  # pragma: no cover - defensive
                print(f"[{self.camera_id}] inference error: {exc}")
                time.sleep(0.05)
                continue

            self.track_manager.update(detections)
            annotated = _draw_annotations(frame, detections, self.track_manager)

            ok_enc, buf = cv2.imencode(".jpg", annotated, [int(cv2.IMWRITE_JPEG_QUALITY), config.JPEG_QUALITY])

            loop_time = time.perf_counter() - loop_start
            inst_fps = (1.0 / loop_time) if loop_time > 0 else 0.0
            self._fps_ema = inst_fps if self._fps_ema == 0 else (0.9 * self._fps_ema + 0.1 * inst_fps)

            self.shared.update(
                jpeg_bytes=buf.tobytes() if ok_enc else None,
                fps=self._fps_ema,
                latency_ms=latency_ms,
                connected=True,
                resolution=self.camera.resolution,
            )

        self.camera.release()
        print(f"[{self.camera_id}] stopped")

    def describe(self) -> dict:
        fps, latency_ms, connected, resolution = self.shared.get_metrics()
        snap = self.track_manager.snapshot()
        return {
            "camera_id": self.camera_id,
            "name": self.name,
            "source_type": self.source_type,
            "source_ref": self.source_ref,
            "source": self.source,
            "status": "LIVE" if connected else "OFFLINE",
            "people_count": snap["current_count"],
            "priority_counts": snap["priority_counts"],
            "priority_summary": _summarize_priority(snap["priority_counts"]),
            "fps": round(fps, 1),
            "latency_ms": round(latency_ms, 1),
            "resolution": f"{resolution[0]}x{resolution[1]}" if resolution[0] else "--",
            "last_update": time.strftime("%H:%M:%S"),
            "epoch_id": snap["epoch_id"],
        }


# ---------------------------------------------------------------------------
# Registry: the set of currently-running cameras
# ---------------------------------------------------------------------------
class CameraRegistry:
    def __init__(self, db: Database):
        self.db = db
        self.workers: dict[str, CameraWorker] = {}
        self._lock = threading.Lock()

    def load_seed_rows(self) -> list[dict]:
        rows = self.db.get_cameras(enabled_only=True)
        if not rows:
            rows = []
            for c in config.DEFAULT_CAMERAS:
                row = {**c, "loop_video": 1, "enabled": 1, "created_at": now_iso()}
                self.db.upsert_camera(row)
                rows.append(row)
        return rows

    def start_all(self, rows: list[dict]) -> None:
        for row in rows:
            self._start_worker(row)

    def _start_worker(self, row: dict) -> CameraWorker:
        worker = CameraWorker(self.db, row)
        worker.start()
        with self._lock:
            self.workers[worker.camera_id] = worker
        return worker

    def _next_camera_id(self) -> str:
        # Numbers are never reused (even a removed camera's number stays
        # retired), so a stale bookmark to "CAM-02" never silently points
        # at an unrelated later camera.
        nums = []
        for c in self.db.get_cameras(enabled_only=False):
            m = re.match(r"^CAM-(\d+)$", c["camera_id"])
            if m:
                nums.append(int(m.group(1)))
        n = (max(nums) + 1) if nums else 1
        return f"CAM-{n:02d}"

    def add_camera(
        self, name: str, source_type: str, source_ref: str,
        source: str = config.SOURCE_RGB, loop_video: bool = True,
    ) -> CameraWorker:
        with self._lock:
            if len(self.workers) >= config.MAX_CAMERAS:
                raise ValueError(f"Maximum of {config.MAX_CAMERAS} cameras reached")
            camera_id = self._next_camera_id()
        row = {
            "camera_id": camera_id,
            "name": name or camera_id,
            "source_type": source_type,
            "source_ref": str(source_ref),
            "source": source,
            "loop_video": 1 if loop_video else 0,
            "enabled": 1,
            "created_at": now_iso(),
        }
        self.db.upsert_camera(row)
        return self._start_worker(row)

    def remove_camera(self, camera_id: str) -> bool:
        with self._lock:
            worker = self.workers.pop(camera_id, None)
        if worker is None:
            return False
        worker.stop()
        self.db.delete_camera(camera_id)
        return True

    def stop_all(self) -> None:
        with self._lock:
            workers = list(self.workers.values())
        for w in workers:
            w.stop()

    def list_workers(self) -> list[CameraWorker]:
        with self._lock:
            return sorted(self.workers.values(), key=lambda w: w.camera_id)

    def get(self, camera_id: str) -> Optional[CameraWorker]:
        with self._lock:
            return self.workers.get(camera_id)


# ---------------------------------------------------------------------------
# WebSocket fan-out
# ---------------------------------------------------------------------------
class ConnectionManager:
    def __init__(self) -> None:
        self.active: set[WebSocket] = set()

    async def connect(self, ws: WebSocket) -> None:
        await ws.accept()
        self.active.add(ws)

    def disconnect(self, ws: WebSocket) -> None:
        self.active.discard(ws)

    async def broadcast(self, message: dict) -> None:
        dead = []
        for ws in list(self.active):
            try:
                await ws.send_json(message)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self.disconnect(ws)


def _global_snapshot(registry: CameraRegistry) -> dict:
    workers = registry.list_workers()
    return {
        "type": "snapshot",
        "overview": analytics.build_overview(registry),
        "cameras": [w.describe() for w in workers],
        "tracks_by_camera": {w.camera_id: w.track_manager.snapshot()["active_tracks"] for w in workers},
    }


async def broadcaster_loop(app: FastAPI) -> None:
    manager: ConnectionManager = app.state.ws_manager
    registry: CameraRegistry = app.state.registry
    last_seq: dict[str, int] = {}
    while True:
        await asyncio.sleep(config.WS_BROADCAST_INTERVAL_SEC)
        if not manager.active:
            continue

        await manager.broadcast(_global_snapshot(registry))

        for w in registry.list_workers():
            events = w.track_manager.recent_events
            seq0 = last_seq.get(w.camera_id, 0)
            new_events = [e for e in events if e.get("seq", 0) > seq0]
            if new_events:
                last_seq[w.camera_id] = new_events[-1]["seq"]
            for ev in new_events:
                payload = {k: v for k, v in ev.items() if k != "seq"}
                await manager.broadcast({"type": "event", **payload})


# ---------------------------------------------------------------------------
# App / lifespan
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    db = Database()
    registry = CameraRegistry(db)
    rows = registry.load_seed_rows()
    registry.start_all(rows)

    app.state.db = db
    app.state.registry = registry
    app.state.ws_manager = ConnectionManager()
    app.state.broadcaster_task = asyncio.create_task(broadcaster_loop(app))

    try:
        yield
    finally:
        app.state.broadcaster_task.cancel()
        registry.stop_all()
        db.close()


app = FastAPI(title="Search & Rescue Intelligence Dashboard", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=str(config.FRONTEND_DIR)), name="static")


@app.get("/")
async def index() -> FileResponse:
    return FileResponse(str(config.FRONTEND_DIR / "index.html"))


@app.get("/video_feed/{camera_id}")
async def video_feed(camera_id: str, request: Request):
    registry: CameraRegistry = app.state.registry
    worker = registry.get(camera_id)
    if worker is None:
        return JSONResponse({"error": "camera not found"}, status_code=404)

    async def generator():
        boundary = b"--frame"
        target_interval = 1 / 20  # cap the *stream* at 20fps; inference can run faster or slower
        while True:
            if worker.stop_event.is_set() or await request.is_disconnected():
                break
            frame = worker.shared.get_frame()
            if frame:
                yield (
                    boundary + b"\r\nContent-Type: image/jpeg\r\nContent-Length: "
                    + str(len(frame)).encode() + b"\r\n\r\n" + frame + b"\r\n"
                )
            await asyncio.sleep(target_interval)

    return StreamingResponse(generator(), media_type="multipart/x-mixed-replace; boundary=frame")


@app.websocket("/ws")
async def ws_endpoint(websocket: WebSocket) -> None:
    manager: ConnectionManager = app.state.ws_manager
    registry: CameraRegistry = app.state.registry
    await manager.connect(websocket)
    try:
        await websocket.send_json(_global_snapshot(registry))
        while True:
            # Dashboard doesn't need to send anything; this just detects disconnects.
            await websocket.receive_text()
    except WebSocketDisconnect:
        pass
    finally:
        manager.disconnect(websocket)


# -- camera management ------------------------------------------------------
class AddCameraRequest(BaseModel):
    name: str = ""
    source_type: str  # "webcam" | "file"
    source_ref: str
    source: str = config.SOURCE_RGB
    loop_video: bool = True


@app.get("/api/cameras")
async def api_list_cameras() -> JSONResponse:
    registry: CameraRegistry = app.state.registry
    return JSONResponse([w.describe() for w in registry.list_workers()])


@app.post("/api/cameras")
async def api_add_camera(payload: AddCameraRequest) -> JSONResponse:
    registry: CameraRegistry = app.state.registry
    if payload.source_type not in (config.SOURCE_TYPE_WEBCAM, config.SOURCE_TYPE_FILE):
        return JSONResponse({"error": "source_type must be 'webcam' or 'file'"}, status_code=400)
    if payload.source not in (config.SOURCE_RGB, config.SOURCE_THERMAL, config.SOURCE_FUSED):
        return JSONResponse({"error": "source must be RGB, THERMAL, or FUSED"}, status_code=400)
    try:
        worker = await asyncio.to_thread(
            registry.add_camera, payload.name, payload.source_type, payload.source_ref,
            payload.source, payload.loop_video,
        )
    except ValueError as exc:
        return JSONResponse({"error": str(exc)}, status_code=400)
    return JSONResponse(worker.describe())


@app.delete("/api/cameras/{camera_id}")
async def api_remove_camera(camera_id: str) -> JSONResponse:
    registry: CameraRegistry = app.state.registry
    ok = await asyncio.to_thread(registry.remove_camera, camera_id)
    if not ok:
        return JSONResponse({"error": "camera not found"}, status_code=404)
    return JSONResponse({"status": "removed", "camera_id": camera_id})


@app.get("/api/cameras/probe")
async def api_probe_cameras() -> JSONResponse:
    """Webcam device indices available to add, for the 'Add Camera' form."""
    found = await asyncio.to_thread(probe_cameras)
    return JSONResponse({"indices": found})


# -- tracks / events ----------------------------------------------------
@app.get("/api/tracks")
async def api_tracks(camera_id: Optional[str] = None, limit: int = config.DEFAULT_HISTORY_LIMIT) -> JSONResponse:
    return JSONResponse(app.state.db.get_recent_tracks(camera_id, limit))


@app.get("/api/events")
async def api_events(camera_id: Optional[str] = None, limit: int = config.DEFAULT_EVENTS_LIMIT) -> JSONResponse:
    return JSONResponse(app.state.db.get_recent_events(camera_id, limit))


# -- analytics ------------------------------------------------------------
@app.get("/api/analytics/overview")
async def api_overview() -> JSONResponse:
    return JSONResponse(analytics.build_overview(app.state.registry))


@app.get("/api/analytics/priority")
async def api_priority_distribution() -> JSONResponse:
    return JSONResponse(analytics.build_priority_distribution(app.state.registry))


@app.get("/api/analytics/camera_activity")
async def api_camera_activity() -> JSONResponse:
    return JSONResponse(analytics.build_camera_activity(app.state.registry))


@app.get("/api/analytics/confidence")
async def api_confidence() -> JSONResponse:
    return JSONResponse(await asyncio.to_thread(analytics.build_confidence_histogram, app.state.db))


@app.get("/api/analytics/people_over_time")
async def api_people_over_time() -> JSONResponse:
    return JSONResponse(await asyncio.to_thread(analytics.build_people_over_time, app.state.db))
