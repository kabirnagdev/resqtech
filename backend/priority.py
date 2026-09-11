"""
AI-Assisted Rescue Priority Estimation.

IMPORTANT -- what this module is and is not:

This is a transparent, RULE-BASED scoring system built entirely from
observable signals (bounding-box shape, motion over time, detection
confidence, and -- once a thermal camera is attached -- thermal
signatures). It does NOT diagnose injury, unconsciousness, trauma, or
death, and it never will on its own; it has no medical knowledge and no
access to anything but pixels. Every score it produces is an operational
triage HINT -- "this subject looks like it may deserve a closer look
sooner" -- meant to help a human operator allocate attention across many
camera feeds, not to replace their judgement. HIGH/CRITICAL output must
always be paired with "operator verification required" in the UI; never
strip that framing when displaying a score elsewhere.

Pipeline (kept as four explicit stages so any one of them can later be
swapped for a trained model without touching the others):

    1. Human detection          (detector.py)
    2. Human tracking           (tracker.py)
    3. Observable-state estimation   <- this module: posture, movement
    4. Rescue-priority estimation    <- this module: weighted score

Posture and movement are both derived from nothing more than the
bounding box's shape and how its center moves over a few seconds --
there is no pose-estimation model here. That's a deliberate, documented
simplification (v1 heuristic), not a hidden limitation: swap
`estimate_posture` for a real pose-estimation model later and nothing
downstream needs to change, since it still just returns one of the
`Posture` labels.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from . import config

# ---------------------------------------------------------------------------
# Observable-state labels
# ---------------------------------------------------------------------------
POSTURE_STANDING = "Standing"
POSTURE_WALKING = "Walking"
POSTURE_SITTING = "Sitting"
POSTURE_CROUCHING = "Crouching"
POSTURE_LYING = "Lying"
POSTURE_UNKNOWN = "Unknown"

MOVEMENT_MOVING = "Moving"
MOVEMENT_STATIONARY = "Stationary"
MOVEMENT_IMMOBILE = "Potentially Immobile"

THERMAL_NOT_AVAILABLE = "N/A"  # this build is RGB-only; see estimate_thermal_signal()


@dataclass
class CentroidSample:
    t: float
    x: float
    y: float
    box_diag: float  # bbox diagonal length at this sample, for scale-normalizing motion


@dataclass
class PriorityBreakdown:
    """Every component of the score, kept around so the UI/API can show
    *why* a subject was classified a given way -- never present a bare
    number without this."""
    baseline: float
    posture_points: float
    movement_points: float
    inactivity_points: float
    thermal_points: float
    total: float
    label: str
    confidence: float
    posture: str
    movement: str
    inactivity_sec: float
    thermal_signal: str
    is_estimate_uncertain: bool

    def as_dict(self) -> dict:
        return {
            "priority": self.label,
            "priority_score": round(self.total, 1),
            "breakdown": {
                "baseline": self.baseline,
                "posture": round(self.posture_points, 1),
                "movement": round(self.movement_points, 1),
                "inactivity": round(self.inactivity_points, 1),
                "thermal": round(self.thermal_points, 1),
            },
            "posture": self.posture,
            "movement": self.movement,
            "inactivity_sec": round(self.inactivity_sec, 1),
            "thermal_signal": self.thermal_signal,
            "confidence": round(self.confidence, 3),
            "uncertain": self.is_estimate_uncertain,
            "note": "AI-Assisted Rescue Priority Estimation -- an operational "
                    "triage hint from observable signals, not a medical diagnosis. "
                    "Operator verification required.",
        }


def estimate_posture(bbox: tuple[float, float, float, float], is_moving: bool) -> str:
    """Heuristic v1: infer posture purely from the detection box's aspect
    ratio (height / width). A standing person's box is tall and narrow; a
    person lying down produces a short, wide box; a seated or crouched
    person falls in between. This is intentionally simple and replaceable
    -- see the module docstring.
    """
    x1, y1, x2, y2 = bbox
    w = max(1.0, x2 - x1)
    h = max(1.0, y2 - y1)
    aspect = h / w

    if aspect < 1.0:
        return POSTURE_LYING
    if aspect < 1.6:
        return POSTURE_CROUCHING if is_moving else POSTURE_SITTING
    return POSTURE_WALKING if is_moving else POSTURE_STANDING


def estimate_movement(history: list[CentroidSample]) -> tuple[str, float]:
    """Looks at recent centroid positions (already scale-normalized by
    box size) to classify motion. Returns (movement_state,
    normalized_displacement_per_sec) -- the second value is mostly useful
    for debugging/tuning the thresholds below.
    """
    if len(history) < 2:
        return MOVEMENT_MOVING, 0.0  # not enough data yet; assume moving rather than alarm

    span = history[-1].t - history[0].t
    if span <= 0:
        return MOVEMENT_MOVING, 0.0

    total_disp = 0.0
    for a, b in zip(history, history[1:]):
        dx, dy = b.x - a.x, b.y - a.y
        dist = (dx * dx + dy * dy) ** 0.5
        scale = max(1.0, (a.box_diag + b.box_diag) / 2)
        total_disp += dist / scale  # normalize by subject size so near/far cameras compare fairly

    rate = total_disp / span  # normalized units per second

    if rate > 0.15:
        return MOVEMENT_MOVING, rate
    if rate > 0.03:
        return MOVEMENT_STATIONARY, rate
    return MOVEMENT_STATIONARY, rate  # classified Immobile upstream once inactivity_sec crosses the threshold


def estimate_thermal_signal(source: str) -> str:
    """Extension point for thermal fusion (see README). This build is
    RGB-only, so it always reports N/A -- once a ThermalDetector exists,
    this becomes a real read of heat-signature presence/abnormality and
    starts contributing to the score via PRIORITY_WEIGHT_THERMAL_ABNORMAL.
    """
    if source == config.SOURCE_THERMAL or source == config.SOURCE_FUSED:
        return "Not yet implemented"
    return THERMAL_NOT_AVAILABLE


def compute_priority(
    *,
    confidence: float,
    posture: str,
    movement: str,
    inactivity_sec: float,
    thermal_signal: str,
    sample_count: int,
) -> PriorityBreakdown:
    """The weighted, transparent scoring step. Every weight is a named
    constant in config.py -- nothing here is a magic number."""

    uncertain = confidence < config.PRIORITY_MIN_CONFIDENCE or sample_count < config.PRIORITY_MIN_SAMPLES

    baseline = config.PRIORITY_WEIGHT_BASELINE

    if posture == POSTURE_LYING:
        posture_points = config.PRIORITY_WEIGHT_POSTURE_LYING
    elif posture in (POSTURE_SITTING, POSTURE_CROUCHING):
        posture_points = config.PRIORITY_WEIGHT_POSTURE_COMPACT
    elif posture in (POSTURE_STANDING, POSTURE_WALKING):
        posture_points = config.PRIORITY_WEIGHT_POSTURE_UPRIGHT
    else:
        posture_points = config.PRIORITY_WEIGHT_POSTURE_UNKNOWN

    if movement == MOVEMENT_IMMOBILE:
        movement_points = config.PRIORITY_WEIGHT_MOVEMENT_IMMOBILE
    elif movement == MOVEMENT_STATIONARY:
        movement_points = config.PRIORITY_WEIGHT_MOVEMENT_STATIONARY
    else:
        movement_points = config.PRIORITY_WEIGHT_MOVEMENT_MOVING

    inactivity_points = min(
        config.PRIORITY_INACTIVITY_MAX_SCORE,
        (inactivity_sec / config.PRIORITY_INACTIVITY_SECONDS_FOR_MAX) * config.PRIORITY_INACTIVITY_MAX_SCORE,
    )

    thermal_points = config.PRIORITY_WEIGHT_THERMAL_ABNORMAL if thermal_signal == "Abnormal" else 0.0

    total = baseline + posture_points + movement_points + inactivity_points + thermal_points
    total = max(0.0, min(100.0, total))

    if uncertain:
        label = config.PRIORITY_UNKNOWN
    elif total >= config.PRIORITY_THRESHOLD_CRITICAL:
        label = config.PRIORITY_CRITICAL
    elif total >= config.PRIORITY_THRESHOLD_HIGH:
        label = config.PRIORITY_HIGH
    elif total >= config.PRIORITY_THRESHOLD_MEDIUM:
        label = config.PRIORITY_MEDIUM
    else:
        label = config.PRIORITY_LOW

    return PriorityBreakdown(
        baseline=baseline,
        posture_points=posture_points,
        movement_points=movement_points,
        inactivity_points=inactivity_points,
        thermal_points=thermal_points,
        total=total,
        label=label,
        confidence=confidence,
        posture=posture,
        movement=movement,
        inactivity_sec=inactivity_sec,
        thermal_signal=thermal_signal,
        is_estimate_uncertain=uncertain,
    )
