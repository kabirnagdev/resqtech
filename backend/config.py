"""
Central configuration for the Search & Rescue Intelligence Dashboard.

Keeping every tunable in one place makes it easy to retune the rescue
priority weights, add a thermal camera pipeline, or change camera limits
without hunting through business logic elsewhere.
"""
from __future__ import annotations

import os
from pathlib import Path


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"
DATA_DIR = BASE_DIR / "data"
FRONTEND_DIR = BASE_DIR / "frontend" / "dashboard"

DATA_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

DB_PATH = str(DATA_DIR / "rescue.db")

# ---------------------------------------------------------------------------
# Camera sources
# ---------------------------------------------------------------------------
SOURCE_TYPE_WEBCAM = "webcam"
SOURCE_TYPE_FILE = "file"

# Requested capture resolution for webcam sources. Video files use their
# native resolution.
FRAME_WIDTH = int(os.environ.get("HT_FRAME_WIDTH", "960"))
FRAME_HEIGHT = int(os.environ.get("HT_FRAME_HEIGHT", "540"))

CAMERA_RECONNECT_DELAY_SEC = 2.0
CAMERA_PROBE_MAX_INDEX = 6

# Seed camera(s) created automatically the very first time the app runs
# (i.e. when the `cameras` table is empty), so the dashboard isn't blank
# on first launch. Add/remove cameras afterwards from the dashboard.
DEFAULT_CAMERAS = [
    {
        "camera_id": "CAM-01",
        "name": "CAM-01",
        "source_type": SOURCE_TYPE_WEBCAM,
        "source_ref": "0",
        "source": "RGB",
    }
]
MAX_CAMERAS = int(os.environ.get("HT_MAX_CAMERAS", "8"))

# ---------------------------------------------------------------------------
# Detection source labels. Persisted in the DB so the same schema/pipeline
# can eventually serve RGB, THERMAL and FUSED (RGB+thermal fusion)
# detections side by side.
# ---------------------------------------------------------------------------
SOURCE_RGB = "RGB"
SOURCE_THERMAL = "THERMAL"
SOURCE_FUSED = "FUSED"

# ---------------------------------------------------------------------------
# Model / detection
# ---------------------------------------------------------------------------
MODEL_NAME = os.environ.get("HT_MODEL_NAME", "yolo11n.pt")
MODEL_PATH = str(MODELS_DIR / MODEL_NAME)

PERSON_CLASS_ID = 0  # COCO class id for "person"
CONFIDENCE_THRESHOLD = float(os.environ.get("HT_CONF_THRESHOLD", "0.4"))
IOU_THRESHOLD = float(os.environ.get("HT_IOU_THRESHOLD", "0.5"))

# "cpu", "cuda", "cuda:0", "mps" ... leave as "auto" to let detector.py pick.
DEVICE = os.environ.get("HT_DEVICE", "auto")

# ByteTrack configuration, applied through Ultralytics' built-in tracker
# integration (model.track(..., tracker="bytetrack.yaml")). Each camera
# gets its OWN detector+tracker instance (see main.py's CameraWorker) so
# ByteTrack's internal track buffer is never shared across cameras --
# sharing one model/tracker across sources would let track IDs and motion
# history leak between unrelated video feeds.
TRACKER_CONFIG = "bytetrack.yaml"

# ---------------------------------------------------------------------------
# Tracking / session bookkeeping
# ---------------------------------------------------------------------------
TRACK_EXIT_TIMEOUT_SEC = float(os.environ.get("HT_EXIT_TIMEOUT", "2.0"))
DB_UPDATE_INTERVAL_SEC = float(os.environ.get("HT_DB_UPDATE_INTERVAL", "1.0"))

# How much recent centroid history each track keeps, for the movement
# estimator (see priority.py). Longer window = smoother but slower to
# react to a person actually moving again.
MOVEMENT_WINDOW_SEC = 4.0

# ---------------------------------------------------------------------------
# Rescue priority estimation (see priority.py for the full explanation).
# This is a transparent, rule-based scoring system, NOT a medical or
# injury-detection model. Every weight below is intentionally tunable and
# documented so the formula can be inspected, adjusted, or eventually
# swapped for a trained model per-component without touching call sites.
# ---------------------------------------------------------------------------
PRIORITY_WEIGHT_BASELINE = 8        # any confirmed detection gets minimal awareness weight
PRIORITY_WEIGHT_POSTURE_LYING = 35
PRIORITY_WEIGHT_POSTURE_COMPACT = 15  # sitting / crouching
PRIORITY_WEIGHT_POSTURE_UPRIGHT = 0   # standing / walking
PRIORITY_WEIGHT_POSTURE_UNKNOWN = 5
PRIORITY_WEIGHT_MOVEMENT_IMMOBILE = 25
PRIORITY_WEIGHT_MOVEMENT_STATIONARY = 12
PRIORITY_WEIGHT_MOVEMENT_MOVING = 0
PRIORITY_INACTIVITY_MAX_SCORE = 20
PRIORITY_INACTIVITY_SECONDS_FOR_MAX = 120  # inactivity_sec / this * max_score, capped
PRIORITY_WEIGHT_THERMAL_ABNORMAL = 12  # reserved for future thermal fusion; 0 contribution today

# How long (seconds) of continuous "Stationary" movement before a track
# is upgraded to "Potentially Immobile" for scoring purposes.
IMMOBILE_AFTER_SEC = 20.0

# Below this confidence (or before a track has accumulated a few
# detections), the priority engine reports UNKNOWN rather than guessing.
PRIORITY_MIN_CONFIDENCE = 0.35
PRIORITY_MIN_SAMPLES = 3

# Score thresholds -> classification label.
PRIORITY_THRESHOLD_CRITICAL = 70
PRIORITY_THRESHOLD_HIGH = 45
PRIORITY_THRESHOLD_MEDIUM = 20
# anything below MEDIUM's threshold is LOW.

PRIORITY_LOW = "LOW"
PRIORITY_MEDIUM = "MEDIUM"
PRIORITY_HIGH = "HIGH"
PRIORITY_CRITICAL = "CRITICAL"
PRIORITY_UNKNOWN = "UNKNOWN"
PRIORITY_ORDER = [PRIORITY_UNKNOWN, PRIORITY_LOW, PRIORITY_MEDIUM, PRIORITY_HIGH, PRIORITY_CRITICAL]

# ---------------------------------------------------------------------------
# Dashboard / networking
# ---------------------------------------------------------------------------
HOST = os.environ.get("HT_HOST", "0.0.0.0")
PORT = int(os.environ.get("HT_PORT", "8000"))
WS_BROADCAST_INTERVAL_SEC = 0.5
JPEG_QUALITY = 80
DEFAULT_HISTORY_LIMIT = 50
DEFAULT_EVENTS_LIMIT = 80
PEOPLE_OVER_TIME_WINDOW_MIN = 15
PEOPLE_OVER_TIME_BUCKET_SEC = 30
