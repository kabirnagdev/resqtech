"""
SQLite persistence layer for the Search & Rescue Intelligence Dashboard.

Design goals (unchanged from the single-camera build, now multi-camera):
  * Never write a row per frame -- writers upsert a periodic *snapshot* of
    each active track plus a final row on exit.
  * Never block a camera's capture/inference loop on disk I/O -- all
    writes are handed to a single background thread through an in-memory
    queue, shared across every camera.
  * Reads (used by the REST API / dashboard) use short-lived connections
    from whichever thread calls them; WAL mode lets those run concurrently
    with the writer thread.
"""
from __future__ import annotations

import json
import queue
import sqlite3
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Optional

from . import config

_SCHEMA = """
CREATE TABLE IF NOT EXISTS cameras (
    camera_id    TEXT PRIMARY KEY,
    name         TEXT NOT NULL,
    source_type  TEXT NOT NULL,
    source_ref   TEXT NOT NULL,
    source       TEXT NOT NULL DEFAULT 'RGB',
    loop_video   INTEGER NOT NULL DEFAULT 1,
    enabled      INTEGER NOT NULL DEFAULT 1,
    created_at   TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS tracks (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    camera_id        TEXT    NOT NULL,
    epoch_id         TEXT    NOT NULL,
    track_id         INTEGER NOT NULL,
    source           TEXT    NOT NULL,
    first_seen       TEXT    NOT NULL,
    last_seen        TEXT    NOT NULL,
    duration_sec     REAL    NOT NULL DEFAULT 0,
    max_confidence   REAL    NOT NULL DEFAULT 0,
    avg_confidence   REAL    NOT NULL DEFAULT 0,
    detection_count  INTEGER NOT NULL DEFAULT 0,
    bbox_x1          REAL,
    bbox_y1          REAL,
    bbox_x2          REAL,
    bbox_y2          REAL,
    posture          TEXT,
    movement_state   TEXT,
    inactivity_sec   REAL    NOT NULL DEFAULT 0,
    priority         TEXT    NOT NULL DEFAULT 'UNKNOWN',
    priority_score   REAL    NOT NULL DEFAULT 0,
    max_priority     TEXT    NOT NULL DEFAULT 'UNKNOWN',
    max_priority_score REAL  NOT NULL DEFAULT 0,
    status           TEXT    NOT NULL DEFAULT 'ACTIVE',
    UNIQUE(camera_id, epoch_id, track_id)
);
CREATE INDEX IF NOT EXISTS idx_tracks_camera ON tracks(camera_id);
CREATE INDEX IF NOT EXISTS idx_tracks_status ON tracks(status);
CREATE INDEX IF NOT EXISTS idx_tracks_lastseen ON tracks(last_seen);

CREATE TABLE IF NOT EXISTS events (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    camera_id   TEXT    NOT NULL,
    epoch_id    TEXT    NOT NULL,
    track_id    INTEGER,
    event_type  TEXT    NOT NULL,
    priority    TEXT,
    source      TEXT    NOT NULL,
    timestamp   TEXT    NOT NULL,
    details     TEXT
);
CREATE INDEX IF NOT EXISTS idx_events_camera_ts ON events(camera_id, timestamp);
"""


@dataclass
class _Job:
    kind: str
    payload: dict = field(default_factory=dict)


class Database:
    """Single background writer thread shared by every camera worker, so
    N cameras never fight over SQLite's single-writer lock -- they just
    all enqueue onto the same queue."""

    def __init__(self, path: str = config.DB_PATH):
        self.path = path
        self._queue: "queue.Queue[Optional[_Job]]" = queue.Queue()
        self._init_schema()
        self._writer = threading.Thread(target=self._writer_loop, name="db-writer", daemon=True)
        self._writer.start()

    # -- setup -------------------------------------------------------
    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path, timeout=10)
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=NORMAL;")
        conn.execute("PRAGMA foreign_keys=ON;")
        conn.row_factory = sqlite3.Row
        return conn

    def _init_schema(self) -> None:
        conn = self._connect()
        try:
            conn.executescript(_SCHEMA)
            conn.commit()
        finally:
            conn.close()

    # -- producer-side API (non-blocking) -----------------------------
    def upsert_track(self, record: dict) -> None:
        self._queue.put(_Job("upsert_track", record))

    def log_event(self, record: dict) -> None:
        self._queue.put(_Job("log_event", record))

    def upsert_camera(self, record: dict) -> None:
        self._queue.put(_Job("upsert_camera", record))

    def close(self) -> None:
        self._queue.put(None)
        self._writer.join(timeout=5)

    # -- writer thread -------------------------------------------------
    def _writer_loop(self) -> None:
        conn = self._connect()
        try:
            while True:
                job = self._queue.get()
                if job is None:
                    break
                try:
                    self._handle_job(conn, job)
                except Exception as exc:  # pragma: no cover - defensive
                    print(f"[database] write failed ({job.kind}): {exc}")
            conn.commit()
        finally:
            conn.close()

    def _handle_job(self, conn: sqlite3.Connection, job: _Job) -> None:
        p = job.payload
        if job.kind == "upsert_track":
            conn.execute(
                """
                INSERT INTO tracks (
                    camera_id, epoch_id, track_id, source,
                    first_seen, last_seen, duration_sec,
                    max_confidence, avg_confidence, detection_count,
                    bbox_x1, bbox_y1, bbox_x2, bbox_y2,
                    posture, movement_state, inactivity_sec,
                    priority, priority_score, max_priority, max_priority_score,
                    status
                ) VALUES (
                    :camera_id, :epoch_id, :track_id, :source,
                    :first_seen, :last_seen, :duration_sec,
                    :max_confidence, :avg_confidence, :detection_count,
                    :bbox_x1, :bbox_y1, :bbox_x2, :bbox_y2,
                    :posture, :movement_state, :inactivity_sec,
                    :priority, :priority_score, :max_priority, :max_priority_score,
                    :status
                )
                ON CONFLICT(camera_id, epoch_id, track_id) DO UPDATE SET
                    last_seen=excluded.last_seen,
                    duration_sec=excluded.duration_sec,
                    max_confidence=excluded.max_confidence,
                    avg_confidence=excluded.avg_confidence,
                    detection_count=excluded.detection_count,
                    bbox_x1=excluded.bbox_x1,
                    bbox_y1=excluded.bbox_y1,
                    bbox_x2=excluded.bbox_x2,
                    bbox_y2=excluded.bbox_y2,
                    posture=excluded.posture,
                    movement_state=excluded.movement_state,
                    inactivity_sec=excluded.inactivity_sec,
                    priority=excluded.priority,
                    priority_score=excluded.priority_score,
                    max_priority=excluded.max_priority,
                    max_priority_score=excluded.max_priority_score,
                    status=excluded.status
                """,
                p,
            )
            conn.commit()
        elif job.kind == "log_event":
            conn.execute(
                """
                INSERT INTO events (camera_id, epoch_id, track_id, event_type, priority, source, timestamp, details)
                VALUES (:camera_id, :epoch_id, :track_id, :event_type, :priority, :source, :timestamp, :details)
                """,
                p,
            )
            conn.commit()
        elif job.kind == "upsert_camera":
            conn.execute(
                """
                INSERT INTO cameras (camera_id, name, source_type, source_ref, source, loop_video, enabled, created_at)
                VALUES (:camera_id, :name, :source_type, :source_ref, :source, :loop_video, :enabled, :created_at)
                ON CONFLICT(camera_id) DO UPDATE SET
                    name=excluded.name,
                    source_type=excluded.source_type,
                    source_ref=excluded.source_ref,
                    source=excluded.source,
                    loop_video=excluded.loop_video,
                    enabled=excluded.enabled
                """,
                p,
            )
            conn.commit()

    # -- read-side API (safe to call from any thread) -------------------
    def get_cameras(self, enabled_only: bool = False) -> list[dict]:
        conn = self._connect()
        try:
            sql = "SELECT * FROM cameras"
            if enabled_only:
                sql += " WHERE enabled = 1"
            sql += " ORDER BY camera_id"
            return [dict(r) for r in conn.execute(sql).fetchall()]
        finally:
            conn.close()

    def delete_camera(self, camera_id: str) -> None:
        conn = self._connect()
        try:
            conn.execute("UPDATE cameras SET enabled = 0 WHERE camera_id = ?", (camera_id,))
            conn.commit()
        finally:
            conn.close()

    def get_recent_tracks(self, camera_id: Optional[str], limit: int = config.DEFAULT_HISTORY_LIMIT) -> list[dict]:
        conn = self._connect()
        try:
            if camera_id:
                rows = conn.execute(
                    "SELECT * FROM tracks WHERE camera_id = ? ORDER BY last_seen DESC LIMIT ?",
                    (camera_id, limit),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM tracks ORDER BY last_seen DESC LIMIT ?", (limit,)
                ).fetchall()
            return [dict(r) for r in rows]
        finally:
            conn.close()

    def get_recent_events(self, camera_id: Optional[str], limit: int = config.DEFAULT_EVENTS_LIMIT) -> list[dict]:
        conn = self._connect()
        try:
            if camera_id:
                rows = conn.execute(
                    "SELECT * FROM events WHERE camera_id = ? ORDER BY id DESC LIMIT ?",
                    (camera_id, limit),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM events ORDER BY id DESC LIMIT ?", (limit,)
                ).fetchall()
            out = []
            for r in rows:
                d = dict(r)
                if d.get("details"):
                    try:
                        d["details"] = json.loads(d["details"])
                    except (TypeError, json.JSONDecodeError):
                        pass
                out.append(d)
            return out
        finally:
            conn.close()

    def get_confidence_samples(self, limit: int = 500) -> list[float]:
        """Recent per-track max_confidence values, for the detection
        confidence distribution chart."""
        conn = self._connect()
        try:
            rows = conn.execute(
                "SELECT max_confidence FROM tracks ORDER BY last_seen DESC LIMIT ?", (limit,)
            ).fetchall()
            return [r["max_confidence"] for r in rows]
        finally:
            conn.close()

    def get_tracks_in_window(self, since_iso: str) -> list[dict]:
        """Tracks that overlapped the given time window at all (used to
        bucket the "people over time" chart) -- anything last seen at or
        after `since_iso`."""
        conn = self._connect()
        try:
            rows = conn.execute(
                "SELECT camera_id, first_seen, last_seen, status FROM tracks WHERE last_seen >= ?",
                (since_iso,),
            ).fetchall()
            return [dict(r) for r in rows]
        finally:
            conn.close()


def now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S")
