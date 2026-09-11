"""
Track lifecycle management + rescue-priority integration.

Turns a raw stream of per-frame `Detection` objects (track id + bbox +
confidence) into what the rescue dashboard actually needs: who is
currently in frame on this camera, when each anonymous ID first/last
appeared, how long they've stuck around, their observable posture and
movement state, their current AI-Assisted Rescue Priority Estimation, and
PERSON_ENTERED / PERSON_EXITED / PERSON_COUNT_CHANGED / PRIORITY_CHANGED
events.

This is intentionally decoupled from the detector: it only needs a list
of `Detection`s with track ids already assigned, so a future fusion step
(RGB + thermal -> merged detections) can feed this exact same class. Each
camera owns its own TrackManager instance (see main.py's CameraWorker) --
track IDs and priority state are never shared across cameras.
"""
from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, field
from typing import Optional

from . import config, priority as priority_engine
from .database import Database
from .detector import Detection

EVENT_PERSON_ENTERED = "PERSON_ENTERED"
EVENT_PERSON_EXITED = "PERSON_EXITED"
EVENT_PERSON_COUNT_CHANGED = "PERSON_COUNT_CHANGED"
EVENT_PRIORITY_CHANGED = "PRIORITY_CHANGED"
# PERSON_DETECTED is part of the architecture's event vocabulary but is
# intentionally NOT written to the events table per-frame -- that would
# violate the "don't insert a row per frame" requirement. Per-frame
# detection activity is already represented by the periodic track upserts.
EVENT_PERSON_DETECTED = "PERSON_DETECTED"


@dataclass
class TrackState:
    track_id: int
    first_seen: float
    last_seen: float
    max_confidence: float
    conf_sum: float
    detection_count: int
    bbox: tuple[float, float, float, float]
    last_db_write: float = 0.0

    centroid_history: list[priority_engine.CentroidSample] = field(default_factory=list)
    stationary_since: Optional[float] = None

    posture: str = priority_engine.POSTURE_UNKNOWN
    movement: str = priority_engine.MOVEMENT_MOVING
    inactivity_sec: float = 0.0
    thermal_signal: str = priority_engine.THERMAL_NOT_AVAILABLE

    priority: str = config.PRIORITY_UNKNOWN
    priority_score: float = 0.0
    max_priority: str = config.PRIORITY_UNKNOWN
    max_priority_score: float = 0.0

    @property
    def avg_confidence(self) -> float:
        return self.conf_sum / self.detection_count if self.detection_count else 0.0

    @property
    def duration(self) -> float:
        return self.last_seen - self.first_seen


def _priority_rank(label: str) -> int:
    try:
        return config.PRIORITY_ORDER.index(label)
    except ValueError:
        return 0


class TrackManager:
    def __init__(
        self,
        db: Database,
        camera_id: str,
        source: str = config.SOURCE_RGB,
    ):
        self.db = db
        self.camera_id = camera_id
        self.source = source
        self.epoch_id = uuid.uuid4().hex[:12]
        self.epoch_started = time.time()

        self.tracks: dict[int, TrackState] = {}
        self.total_unique_tracks = 0
        self.max_simultaneous = 0
        self.total_detections = 0
        self._last_count = 0

        # Small in-memory ring buffer so the dashboard's live event feed
        # doesn't have to round-trip through SQLite for every tick.
        self.recent_events: list[dict] = []
        self._event_seq = 0

    def update(self, detections: list[Detection]) -> list[dict]:
        """Feed one frame's detections in; returns any events raised."""
        now = time.time()
        new_events: list[dict] = []
        seen_ids: set[int] = set()

        for det in detections:
            if det.track_id is None:
                continue  # ByteTrack hasn't confirmed an id for this box yet
            seen_ids.add(det.track_id)
            self.total_detections += 1
            state = self.tracks.get(det.track_id)
            is_new = state is None
            if is_new:
                state = TrackState(
                    track_id=det.track_id,
                    first_seen=now,
                    last_seen=now,
                    max_confidence=det.confidence,
                    conf_sum=det.confidence,
                    detection_count=1,
                    bbox=det.bbox,
                )
                self.tracks[det.track_id] = state
                self.total_unique_tracks += 1
            else:
                state.last_seen = now
                state.max_confidence = max(state.max_confidence, det.confidence)
                state.conf_sum += det.confidence
                state.detection_count += 1
                state.bbox = det.bbox

            new_events += self._update_observable_state(state, det, now)

            if is_new:
                new_events.append(self._emit(EVENT_PERSON_ENTERED, state.track_id, now))
                self._write_track(state, status="ACTIVE")
            elif now - state.last_db_write >= config.DB_UPDATE_INTERVAL_SEC:
                self._write_track(state, status="ACTIVE")

        for track_id, state in list(self.tracks.items()):
            if track_id in seen_ids:
                continue
            if now - state.last_seen >= config.TRACK_EXIT_TIMEOUT_SEC:
                new_events.append(self._emit(EVENT_PERSON_EXITED, track_id, now))
                self._write_track(state, status="EXITED")
                del self.tracks[track_id]

        current_count = len(self.tracks)
        self.max_simultaneous = max(self.max_simultaneous, current_count)
        if current_count != self._last_count:
            new_events.append(
                self._emit(EVENT_PERSON_COUNT_CHANGED, None, now, details={"count": current_count})
            )
            self._last_count = current_count

        return new_events

    # -- observable-state + priority ------------------------------------
    def _update_observable_state(self, state: TrackState, det: Detection, now: float) -> list[dict]:
        """Recomputes posture/movement/inactivity/priority for one track
        from its latest detection. Returns any events raised (currently:
        zero or one PRIORITY_CHANGED) so callers that care about "every
        event this frame" -- not just the ones already logged to
        recent_events/the DB -- get a complete list back."""
        x1, y1, x2, y2 = det.bbox
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        diag = ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5
        state.centroid_history.append(priority_engine.CentroidSample(t=now, x=cx, y=cy, box_diag=diag))
        cutoff = now - config.MOVEMENT_WINDOW_SEC
        state.centroid_history = [s for s in state.centroid_history if s.t >= cutoff]

        movement_raw, _rate = priority_engine.estimate_movement(state.centroid_history)

        if movement_raw == priority_engine.MOVEMENT_MOVING:
            state.stationary_since = None
            movement_final = priority_engine.MOVEMENT_MOVING
            inactivity_sec = 0.0
        else:
            if state.stationary_since is None:
                state.stationary_since = now
            inactivity_sec = now - state.stationary_since
            movement_final = (
                priority_engine.MOVEMENT_IMMOBILE
                if inactivity_sec >= config.IMMOBILE_AFTER_SEC
                else priority_engine.MOVEMENT_STATIONARY
            )

        posture = priority_engine.estimate_posture(det.bbox, is_moving=(movement_final == priority_engine.MOVEMENT_MOVING))
        thermal_signal = priority_engine.estimate_thermal_signal(self.source)

        breakdown = priority_engine.compute_priority(
            confidence=det.confidence,
            posture=posture,
            movement=movement_final,
            inactivity_sec=inactivity_sec,
            thermal_signal=thermal_signal,
            sample_count=state.detection_count,
        )

        state.posture = posture
        state.movement = movement_final
        state.inactivity_sec = inactivity_sec
        state.thermal_signal = thermal_signal

        previous_priority = state.priority
        state.priority = breakdown.label
        state.priority_score = breakdown.total

        if _priority_rank(breakdown.label) > _priority_rank(state.max_priority):
            state.max_priority = breakdown.label
            state.max_priority_score = breakdown.total

        if breakdown.label != previous_priority:
            return [
                self._emit(
                    EVENT_PRIORITY_CHANGED,
                    state.track_id,
                    now,
                    details={"priority": breakdown.label, "score": round(breakdown.total, 1)},
                    priority=breakdown.label,
                )
            ]
        return []

    # -- persistence helpers ------------------------------------------
    def _write_track(self, state: TrackState, status: str) -> None:
        state.last_db_write = state.last_seen
        self.db.upsert_track(
            {
                "camera_id": self.camera_id,
                "epoch_id": self.epoch_id,
                "track_id": state.track_id,
                "source": self.source,
                "first_seen": _fmt(state.first_seen),
                "last_seen": _fmt(state.last_seen),
                "duration_sec": round(state.duration, 2),
                "max_confidence": round(state.max_confidence, 4),
                "avg_confidence": round(state.avg_confidence, 4),
                "detection_count": state.detection_count,
                "bbox_x1": state.bbox[0],
                "bbox_y1": state.bbox[1],
                "bbox_x2": state.bbox[2],
                "bbox_y2": state.bbox[3],
                "posture": state.posture,
                "movement_state": state.movement,
                "inactivity_sec": round(state.inactivity_sec, 1),
                "priority": state.priority,
                "priority_score": round(state.priority_score, 1),
                "max_priority": state.max_priority,
                "max_priority_score": round(state.max_priority_score, 1),
                "status": status,
            }
        )

    def _emit(
        self,
        event_type: str,
        track_id: Optional[int],
        now: float,
        details: Optional[dict] = None,
        priority: Optional[str] = None,
    ) -> dict:
        record = {
            "camera_id": self.camera_id,
            "epoch_id": self.epoch_id,
            "track_id": track_id,
            "event_type": event_type,
            "priority": priority,
            "source": self.source,
            "timestamp": _fmt(now),
            "details": json.dumps(details) if details else None,
        }
        self.db.log_event(record)

        self._event_seq += 1
        readable = dict(record)
        readable["details"] = details
        readable["seq"] = self._event_seq
        self.recent_events.append(readable)
        if len(self.recent_events) > 200:
            self.recent_events = self.recent_events[-200:]
        return readable

    # -- read-side -------------------------------------------------------
    def snapshot(self) -> dict:
        """Live state for the dashboard, no DB round-trip required."""
        active = [
            {
                "track_id": s.track_id,
                "confidence": round(s.max_confidence, 3),
                "avg_confidence": round(s.avg_confidence, 3),
                "first_seen": _fmt(s.first_seen),
                "last_seen": _fmt(s.last_seen),
                "duration_sec": round(s.duration, 1),
                "bbox": s.bbox,
                "posture": s.posture,
                "movement": s.movement,
                "inactivity_sec": round(s.inactivity_sec, 1),
                "priority": s.priority,
                "priority_score": round(s.priority_score, 1),
                "thermal_signal": s.thermal_signal,
            }
            for s in sorted(self.tracks.values(), key=lambda s: s.track_id)
        ]
        priority_counts = {label: 0 for label in config.PRIORITY_ORDER}
        for s in self.tracks.values():
            priority_counts[s.priority] = priority_counts.get(s.priority, 0) + 1

        return {
            "camera_id": self.camera_id,
            "epoch_id": self.epoch_id,
            "current_count": len(self.tracks),
            "max_simultaneous": self.max_simultaneous,
            "total_unique_tracks": self.total_unique_tracks,
            "total_detections": self.total_detections,
            "active_tracks": active,
            "priority_counts": priority_counts,
        }

    def reset_for_reconnect(self) -> None:
        """Close out any still-active tracks as EXITED rather than leaving
        them dangling. Called on a camera drop and as the first step of
        `start_new_epoch`."""
        now = time.time()
        for track_id, state in list(self.tracks.items()):
            self._emit(EVENT_PERSON_EXITED, track_id, now)
            self._write_track(state, status="EXITED")
        self.tracks.clear()
        self._last_count = 0

    def start_new_epoch(self, source: Optional[str] = None) -> None:
        """Close out active tracks and roll over to a brand new epoch_id.

        Track IDs are only guaranteed unique *within* an epoch --
        ByteTrack assigns them starting from 1 again whenever its internal
        state is cleared (`detector.reset_tracker_state()`), which happens
        on every camera reconnect and camera source switch. Reusing the
        old epoch_id after that would let a new physical track collide
        with an old, already-EXITED row that happens to share the same
        track_id (the `tracks` table's uniqueness is (camera_id, epoch_id,
        track_id)), silently corrupting that row's history.
        """
        self.reset_for_reconnect()
        self.epoch_id = uuid.uuid4().hex[:12]
        self.epoch_started = time.time()
        self.total_unique_tracks = 0
        self.max_simultaneous = 0
        self.total_detections = 0
        self._last_count = 0
        self.recent_events = []
        self._event_seq = 0
        if source is not None:
            self.source = source


def _fmt(ts: float) -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(ts))
