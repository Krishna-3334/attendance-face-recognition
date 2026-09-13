"""SQLite persistence for enrolments, attendance and rejected spoof attempts.

Embeddings are stored as raw float32 bytes in a BLOB column and loaded into a
single in-memory matrix at startup, so matching is one vectorised dot product
against the whole gallery rather than a Python loop over users.
"""

from __future__ import annotations

import sqlite3
import threading
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Iterator

import numpy as np

from . import config

_LOCK = threading.Lock()

SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    name        TEXT    NOT NULL UNIQUE,
    created_at  TEXT    NOT NULL,
    n_samples   INTEGER NOT NULL DEFAULT 0,
    embedding   BLOB    NOT NULL
);

CREATE TABLE IF NOT EXISTS samples (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id     INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    created_at  TEXT    NOT NULL,
    embedding   BLOB    NOT NULL
);

CREATE TABLE IF NOT EXISTS attendance (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id     INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    name        TEXT    NOT NULL,
    timestamp   TEXT    NOT NULL,
    date        TEXT    NOT NULL,
    similarity  REAL    NOT NULL,
    liveness    REAL,
    status      TEXT    NOT NULL DEFAULT 'Present'
);

CREATE TABLE IF NOT EXISTS spoof_events (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp    TEXT    NOT NULL,
    live_score   REAL    NOT NULL,
    attack_type  TEXT    NOT NULL,
    matched_name TEXT
);

CREATE INDEX IF NOT EXISTS idx_attendance_date ON attendance(date);
CREATE INDEX IF NOT EXISTS idx_samples_user ON samples(user_id);
"""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@contextmanager
def connect() -> Iterator[sqlite3.Connection]:
    conn = sqlite3.connect(config.DB_PATH, timeout=10)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db() -> None:
    with connect() as conn:
        conn.executescript(SCHEMA)


# --- users -------------------------------------------------------------------


def add_user(name: str, embedding: np.ndarray, sample_embeddings: list[np.ndarray]) -> int:
    blob = embedding.astype(np.float32).tobytes()
    with _LOCK, connect() as conn:
        cur = conn.execute(
            "INSERT INTO users (name, created_at, n_samples, embedding) VALUES (?,?,?,?)",
            (name, _now(), len(sample_embeddings), blob),
        )
        user_id = int(cur.lastrowid)
        conn.executemany(
            "INSERT INTO samples (user_id, created_at, embedding) VALUES (?,?,?)",
            [(user_id, _now(), e.astype(np.float32).tobytes()) for e in sample_embeddings],
        )
    return user_id


def user_exists(name: str) -> bool:
    with connect() as conn:
        return conn.execute("SELECT 1 FROM users WHERE name = ?", (name,)).fetchone() is not None


def list_users() -> list[dict]:
    with connect() as conn:
        rows = conn.execute(
            "SELECT id, name, created_at, n_samples FROM users ORDER BY name COLLATE NOCASE"
        ).fetchall()
    return [dict(r) for r in rows]


def delete_user(user_id: int) -> bool:
    with _LOCK, connect() as conn:
        cur = conn.execute("DELETE FROM users WHERE id = ?", (user_id,))
        return cur.rowcount > 0


def load_gallery() -> tuple[list[int], list[str], np.ndarray]:
    """Return (ids, names, matrix[N,512]) of L2-normalised gallery embeddings."""
    with connect() as conn:
        rows = conn.execute("SELECT id, name, embedding FROM users ORDER BY id").fetchall()
    if not rows:
        return [], [], np.zeros((0, 512), dtype=np.float32)
    ids = [int(r["id"]) for r in rows]
    names = [str(r["name"]) for r in rows]
    mat = np.stack([np.frombuffer(r["embedding"], dtype=np.float32) for r in rows])
    return ids, names, mat


# --- attendance --------------------------------------------------------------


def seconds_since_last_checkin(user_id: int) -> float | None:
    with connect() as conn:
        row = conn.execute(
            "SELECT timestamp FROM attendance WHERE user_id = ? ORDER BY id DESC LIMIT 1",
            (user_id,),
        ).fetchone()
    if row is None:
        return None
    last = datetime.fromisoformat(row["timestamp"])
    return (datetime.now(timezone.utc) - last).total_seconds()


def log_attendance(user_id: int, name: str, similarity: float, liveness: float | None) -> dict:
    now = datetime.now(timezone.utc)
    record = {
        "user_id": user_id,
        "name": name,
        "timestamp": now.isoformat(timespec="seconds"),
        "date": now.date().isoformat(),
        "similarity": round(float(similarity), 4),
        "liveness": None if liveness is None else round(float(liveness), 4),
        "status": "Present",
    }
    with _LOCK, connect() as conn:
        cur = conn.execute(
            "INSERT INTO attendance (user_id, name, timestamp, date, similarity, liveness, status)"
            " VALUES (:user_id,:name,:timestamp,:date,:similarity,:liveness,:status)",
            record,
        )
        record["id"] = int(cur.lastrowid)
    return record


def list_attendance(limit: int = 200) -> list[dict]:
    with connect() as conn:
        rows = conn.execute(
            "SELECT id, user_id, name, timestamp, date, similarity, liveness, status"
            " FROM attendance ORDER BY id DESC LIMIT ?",
            (limit,),
        ).fetchall()
    return [dict(r) for r in rows]


def log_spoof(live_score: float, attack_type: str, matched_name: str | None) -> None:
    with _LOCK, connect() as conn:
        conn.execute(
            "INSERT INTO spoof_events (timestamp, live_score, attack_type, matched_name)"
            " VALUES (?,?,?,?)",
            (_now(), float(live_score), attack_type, matched_name),
        )


def list_spoof_events(limit: int = 100) -> list[dict]:
    with connect() as conn:
        rows = conn.execute(
            "SELECT id, timestamp, live_score, attack_type, matched_name"
            " FROM spoof_events ORDER BY id DESC LIMIT ?",
            (limit,),
        ).fetchall()
    return [dict(r) for r in rows]


def stats() -> dict:
    today = datetime.now(timezone.utc).date().isoformat()
    with connect() as conn:
        users = conn.execute("SELECT COUNT(*) c FROM users").fetchone()["c"]
        today_unique = conn.execute(
            "SELECT COUNT(DISTINCT user_id) c FROM attendance WHERE date = ?", (today,)
        ).fetchone()["c"]
        total_events = conn.execute("SELECT COUNT(*) c FROM attendance").fetchone()["c"]
        spoofs = conn.execute("SELECT COUNT(*) c FROM spoof_events").fetchone()["c"]
        avg_sim = conn.execute(
            "SELECT AVG(similarity) a FROM attendance WHERE date = ?", (today,)
        ).fetchone()["a"]
    return {
        "registered_users": int(users),
        "attendance_today": int(today_unique),
        "total_events": int(total_events),
        "spoof_attempts_blocked": int(spoofs),
        "avg_similarity_today": None if avg_sim is None else round(float(avg_sim), 4),
    }
