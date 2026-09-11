# Search & Rescue Intelligence Dashboard

A working (not a demo) multi-camera pipeline for drone / fixed-camera search
and rescue: OpenCV capture → YOLO person detection → ByteTrack multi-object
tracking → rule-based **AI-Assisted Rescue Priority Estimation** → SQLite
persistence → a live multi-camera web dashboard.

```
Camera 1 (RGB) ──┐
Camera 2 (RGB) ──┼──► independent CameraWorker per camera:
Camera N (RGB) ──┘        OpenCV → YOLO (person only) → ByteTrack
                              │
                              ▼
                     Observable-state estimation
                     (posture, movement, inactivity)
                              │
                              ▼
                     Rescue-priority estimation (rule-based, tunable)
                              │
                    ┌─────────┴─────────┐
                    ▼                   ▼
              Dashboard            SQLite
       (WebSocket + MJPEG)   (data/rescue.db)
```

> **Important — what "rescue priority" is and is not.** The priority score
> is an **AI-Assisted Rescue Priority Estimation**: a transparent, rule-based
> operational triage hint computed from *observable signals only*
> (detection confidence, posture inferred from bounding-box shape, motion
> over the last few seconds, and how long a subject has been stationary).
> **It does not diagnose injury, unconsciousness, trauma, or death**, and it
> has no medical knowledge — it never sees anything but pixels. HIGH and
> CRITICAL classifications are always shown with an "operator verification
> required" label, both on the video overlay's color coding and next to
> every numeric score in the UI and API. This is a deliberate design
> choice, not a placeholder: see `backend/priority.py` for the full
> rationale and every scoring weight.

## What this build does

- Connects any number of cameras at once (webcams and/or video files for
  testing today; drone/RGB and, later, thermal cameras use the same
  interface) from a structured camera management table — add or remove a
  camera at runtime, no restart, no code changes.
- Each camera gets its own YOLO + ByteTrack instance, so track IDs and
  motion history never leak between unrelated feeds.
- Live annotated video per camera (bounding box + `#id  confidence%  PRIORITY`
  above/below each person), streamed over MJPEG.
- Per-person rescue priority: LOW / MEDIUM / HIGH / CRITICAL / UNKNOWN with
  a 0–100 score, built from posture, movement state, and inactivity
  duration — recomputed every frame, with a `PRIORITY_CHANGED` event fired
  whenever a subject's classification changes.
- SQLite tracking database that stores one row per track (upserted
  periodically + on exit) — never a row per video frame — plus an events
  log (`PERSON_ENTERED` / `PERSON_EXITED` / `PERSON_COUNT_CHANGED` /
  `PRIORITY_CHANGED`).
- Live "Rescue Events" feed, newest first, over the same WebSocket that
  drives the rest of the dashboard.
- Real-time KPIs (active cameras, people detected, high-priority, critical)
  plus four side-by-side analytics charts: people-over-time, priority
  distribution, camera activity, and detection-confidence distribution —
  hand-rolled dependency-free SVG, so the dashboard works with no internet
  access in the field.
- A tracking-records history table, filterable by camera.
- No facial recognition, no real-world identity — every subject is only
  ever an anonymous per-camera tracking ID (`#17`, `#21`, …) that resets
  whenever a camera reconnects.

## Project structure

```
human-tracking/
├── backend/
│   ├── main.py        FastAPI app: per-camera CameraWorker, CameraRegistry,
│   │                   MJPEG stream, WebSocket fan-out, REST API
│   ├── detector.py     YOLO + ByteTrack inference (Ultralytics model.track())
│   ├── tracker.py      Track lifecycle: enter/exit, observable state, events
│   ├── priority.py     Rescue-priority engine (posture/movement/scoring) --
│   │                    read this first to understand the scoring model
│   ├── camera.py       Webcam / video-file capture wrapper, auto-reconnect
│   ├── database.py     SQLite schema + non-blocking writer thread
│   ├── analytics.py    Assembles dashboard/chart payloads from live state + DB
│   └── config.py       Every tunable: camera limits, priority weights,
│                        thresholds, paths, network settings
├── frontend/
│   └── dashboard/
│       ├── index.html
│       ├── style.css
│       ├── charts.js    dependency-free SVG line/donut/bar charts
│       └── app.js        WebSocket + REST wiring, camera table, forms
├── models/               yolo11n.pt lives here (auto-downloaded if missing)
├── data/                 rescue.db (created on first run)
├── run.py                `python run.py` entry point
├── requirements.txt
└── .gitignore
```

## Install

Windows (PowerShell / cmd):

```bat
cd human-tracking
py -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

macOS / Linux:

```bash
cd human-tracking
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

The first run downloads `yolo11n.pt` (a few MB) into `models/` if it isn't
there already, then reuses it offline afterwards.

## Run

```bash
python run.py
```

Then open **http://localhost:8000** in a browser. On first launch (empty
database) the app seeds one default camera, `CAM-01`, pointed at webcam
index 0 — add more, or replace it, from the dashboard's "+ Add Camera"
form.

Stop with `Ctrl+C`. Shutdown is graceful and bounded (a few seconds even
with a browser tab left open) — see "Shutdown" below if you ever see it
hang.

## Using the dashboard

- **Cameras table** — every configured camera as a row: name/ID, source
  (RGB/THERMAL), live/offline status, current people count, dominant
  priority, FPS, resolution. Click a row to view that camera; click
  **Remove** to stop and disable it.
- **+ Add Camera** — choose Webcam (pick from a rescanned device list) or
  Video file (give a path on the server, e.g. `C:\clips\test.mp4` or
  `/home/you/clips/test.mp4`) and click Add. Video-file sources loop by
  default, which is the easiest way to exercise multi-camera tracking and
  priority escalation without owning several physical webcams — record a
  short clip of yourself standing, then lying down, and point a camera at
  it.
- **Selected Camera panel** — the live annotated feed for whichever camera
  is selected, plus a card per currently-visible subject showing
  confidence, posture, movement, inactivity, and priority score (the same
  detail that's too dense to draw onto the video itself).
- **Live Rescue Events** — a running log of entries/exits and priority
  changes across every camera, newest on top.
- **Analytics** — four charts fed from `/api/analytics/*`, polled every
  few seconds. "People Over Time" is a trailing 15-minute activity trend,
  not an exact concurrent headcount (documented in `analytics.py`).
- **Tracking Records** — recent track history from SQLite, filterable by
  camera.

## Architecture notes

**Multi-camera isolation.** Each camera runs in its own background thread
(`CameraWorker` in `main.py`) with its own `Camera`, `PersonDetector`
(YOLO+ByteTrack), and `TrackManager`. Nothing is shared across cameras —
deliberately, since sharing one ByteTrack instance across unrelated video
sources would let track IDs and motion history leak between them. The
tradeoff is one model instance per camera (more memory); given
`MAX_CAMERAS` defaults to 8, this is the safer choice for correctness.

**Non-blocking pipeline.** The capture/inference loop never touches the
database or the dashboard directly. Frame bytes and metrics are written
into a small lock-guarded `SharedState` object that the async/FastAPI side
reads from; database writes go through one queue-backed background writer
thread shared by every camera (WAL mode, upsert-not-insert-per-frame). A
single `broadcaster_loop` fans a combined snapshot out to every connected
dashboard over one WebSocket every `WS_BROADCAST_INTERVAL_SEC` (0.5s).

**Four-stage pipeline, each stage independently replaceable:**
1. **Human detection** (`detector.py`) — YOLO, person class only.
2. **Human tracking** (`tracker.py`) — ByteTrack via Ultralytics, track
   lifecycle bookkeeping.
3. **Observable-state estimation** (`tracker.py` + `priority.py`) —
   posture and movement inferred from bounding-box shape and motion; no
   pose-estimation model in this build (documented as a v1 heuristic —
   swap `estimate_posture`/`estimate_movement` for a trained model later
   without touching anything downstream).
4. **Rescue-priority estimation** (`priority.py`) — a weighted, fully
   transparent scoring formula over the outputs of stage 3 plus detection
   confidence. Every weight is a named constant in `config.py`, not a
   magic number, and the full breakdown (not just the final label) is
   available via `PriorityBreakdown.as_dict()`.

**Why rule-based, not a trained "injury model."** No dedicated
injury/distress dataset exists for this project, and fabricating one by
assigning arbitrary probabilities would be actively misleading in a
rescue context. The rule-based approach is honest about its own
uncertainty (it reports `UNKNOWN` below a confidence/sample-count floor
rather than guessing) and every score can be explained in plain language
to an operator. See `backend/priority.py`'s module docstring for the full
reasoning.

**Database.** `data/rescue.db`, SQLite, WAL mode. `tracks` stores one row
per track, periodically upserted (`DB_UPDATE_INTERVAL_SEC`, default 1s)
plus a final write on exit — never a row per frame. Uniqueness is on
`(camera_id, epoch_id, track_id)`, not just `(camera_id, track_id)`: every
camera reconnect or source switch resets ByteTrack's internal ID counter,
so a new physical track's ID could otherwise collide with — and silently
corrupt — an old exited track's row that happens to share the same
`track_id`. Rolling to a fresh `epoch_id` on every reconnect/switch closes
that hole. `events` logs `PERSON_ENTERED` / `PERSON_EXITED` /
`PERSON_COUNT_CHANGED` / `PRIORITY_CHANGED`.

**Multi-camera tracking scope (MVP).** Tracking is independent per camera
— there is no cross-camera re-identification (e.g. recognizing that
`CAM-01`'s Person #17 is the same physical person as `CAM-02`'s Person
#4). That's intentional for this build; nothing in the schema or
architecture blocks adding it later (each track row already carries its
own `camera_id`, so a future re-ID layer can join across them without a
schema change).

**RGB + thermal extensibility.** Every detection, track, and event is
tagged with a `source` (`RGB` / `THERMAL` / `FUSED`). This build is
RGB-only — `priority.py`'s `estimate_thermal_signal()` always returns
"N/A" and contributes 0 to the score — but the field exists everywhere
(schema, `TrackManager`, dashboard camera table) so a future
`ThermalDetector` (matching `detector.py`'s `Detection` output shape) and
fusion step can plug in without touching the tracking, priority, or
dashboard code. No thermal hardware or thermal model is implemented here.

**No facial recognition, no real-world identity.** Detections are
anonymous per-camera tracking IDs only. IDs are scoped to a camera's
current `epoch_id` and are never linked to a name or persisted identity.

## API surface

- `GET /` — dashboard
- `GET /video_feed/{camera_id}` — MJPEG annotated stream
- `WS /ws` — snapshot every 0.5s (`{overview, cameras, tracks_by_camera}`)
  plus one `{"type":"event", ...}` message per rescue event
- `GET /api/cameras` / `POST /api/cameras` / `DELETE /api/cameras/{id}`
- `GET /api/cameras/probe` — available webcam device indices
- `GET /api/tracks?camera_id=&limit=` — tracking history
- `GET /api/events?camera_id=&limit=` — event log
- `GET /api/analytics/overview|priority|camera_activity|confidence|people_over_time`

## Search & Rescue Map

Not built in this version. `source`/coordinate fields are left modular in
the schema (a track's camera association is already there) so GPS/map
integration can be added later without a redesign — deliberately out of
scope for now per the original spec.

## Tuning the priority model

Every weight and threshold lives in `config.py`'s "Rescue priority
estimation" section (`PRIORITY_WEIGHT_*`, `PRIORITY_THRESHOLD_*`,
`IMMOBILE_AFTER_SEC`, `PRIORITY_MIN_CONFIDENCE`, `PRIORITY_MIN_SAMPLES`).
Change a constant, restart, and the effect is immediate and explainable —
there's no model to retrain.

## Troubleshooting

- **A camera shows OFFLINE / "no signal — retrying..."** — the device
  index or file path is wrong, or another application has the webcam
  open. Use "Rescan webcams" in the Add Camera form to see what indices
  actually respond.
- **Shutdown hangs** — `Ctrl+C` should return within ~3 seconds
  (`run.py` sets `timeout_graceful_shutdown=3` as a backstop, and every
  streaming/WebSocket loop also exits as soon as its camera's
  `stop_event` is set). If it doesn't, close any open dashboard browser
  tabs first, then Ctrl+C again.
- **"Maximum of N cameras reached"** — raise `HT_MAX_CAMERAS` (env var) or
  `MAX_CAMERAS` in `config.py`.
- **Video file camera never loops / freezes at the end** — pass
  `loop_video: true` when adding it (the dashboard form always does this);
  a one-shot file source is treated as disconnected at EOF instead.
- **No GPU used** — set `HT_DEVICE=cuda:0` (or your platform's device
  string) before running; default `auto` falls back to CPU if no CUDA
  device is available via `torch.cuda.is_available()`.
