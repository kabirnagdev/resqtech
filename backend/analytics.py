"""
Assembles every payload the dashboard's KPI row, camera table, and
analytics panels consume, mixing cheap in-memory live state (each
CameraWorker's TrackManager.snapshot()) with DB-backed historical numbers.

No type import of `CameraRegistry` at runtime (main.py imports this
module, so importing main.py back here would be circular) -- functions
just duck-type against `.list_workers()`.
"""
from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any

from . import config
from .database import Database

if TYPE_CHECKING:
    from .main import CameraRegistry


def build_overview(registry: "CameraRegistry") -> dict[str, Any]:
    workers = registry.list_workers()
    people = 0
    high = 0
    critical = 0
    active_cameras = 0
    for w in workers:
        _, _, connected, _ = w.shared.get_metrics()
        if connected:
            active_cameras += 1
        snap = w.track_manager.snapshot()
        people += snap["current_count"]
        high += snap["priority_counts"].get(config.PRIORITY_HIGH, 0)
        critical += snap["priority_counts"].get(config.PRIORITY_CRITICAL, 0)
    return {
        "active_cameras": active_cameras,
        "total_cameras": len(workers),
        "people_detected": people,
        "high_priority": high,
        "critical": critical,
    }


def build_priority_distribution(registry: "CameraRegistry") -> dict[str, int]:
    counts = {label: 0 for label in config.PRIORITY_ORDER}
    for w in registry.list_workers():
        snap = w.track_manager.snapshot()
        for label, c in snap["priority_counts"].items():
            counts[label] = counts.get(label, 0) + c
    return counts


def build_camera_activity(registry: "CameraRegistry") -> list[dict[str, Any]]:
    out = []
    for w in registry.list_workers():
        snap = w.track_manager.snapshot()
        out.append({"camera_id": w.camera_id, "name": w.name, "people_count": snap["current_count"]})
    return out


def build_confidence_histogram(db: Database, limit: int = 500) -> dict[str, Any]:
    """Distribution of recent per-track max detection confidence, bucketed
    into ten 10-point-wide bins."""
    samples = db.get_confidence_samples(limit=limit)
    bins = [0] * 10
    for s in samples:
        idx = min(9, max(0, int(s * 10)))
        bins[idx] += 1
    labels = [f"{i * 10}-{i * 10 + 10}%" for i in range(10)]
    return {"labels": labels, "counts": bins, "sample_count": len(samples)}


def build_people_over_time(db: Database) -> dict[str, Any]:
    """Bucketed count of how many tracks overlapped each time window over
    the trailing PEOPLE_OVER_TIME_WINDOW_MIN minutes, across every camera.

    This is an approximation, not an exact average: a track counts toward
    every bucket its first_seen..last_seen span overlaps, so a person who
    stood still in frame for 3 buckets straight is counted in all 3. For
    an operational "how busy has it been" trend line that's the right
    behavior; it is not a precise concurrent-headcount integral.
    """
    now = time.time()
    window_sec = config.PEOPLE_OVER_TIME_WINDOW_MIN * 60
    bucket_sec = config.PEOPLE_OVER_TIME_BUCKET_SEC
    since = now - window_sec
    since_iso = time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(since))

    rows = db.get_tracks_in_window(since_iso)
    n_buckets = max(1, int(window_sec // bucket_sec))
    counts = [0] * n_buckets

    def _parse(ts: str) -> float:
        return time.mktime(time.strptime(ts, "%Y-%m-%dT%H:%M:%S"))

    for r in rows:
        try:
            fs = _parse(r["first_seen"])
            ls = now if r["status"] == "ACTIVE" else _parse(r["last_seen"])
        except (ValueError, TypeError):
            continue
        for i in range(n_buckets):
            b_start = since + i * bucket_sec
            b_end = b_start + bucket_sec
            if fs <= b_end and ls >= b_start:
                counts[i] += 1

    labels = [time.strftime("%H:%M", time.localtime(since + i * bucket_sec)) for i in range(n_buckets)]
    return {"labels": labels, "counts": counts}
