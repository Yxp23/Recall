"""
SQLite event log.
"""
from __future__ import annotations

import sqlite3
import time
from pathlib import Path
from typing import Optional


class EventStore:
    def __init__(self, path: str = "data/recall.db") -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS events (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                ts         REAL    NOT NULL,
                item       TEXT    NOT NULL,
                event      TEXT    NOT NULL,
                zone       TEXT,
                detail     TEXT,
                confidence REAL,
                clip_path  TEXT
            );
            CREATE INDEX IF NOT EXISTS idx_item_ts ON events (item, ts);
        """)
        # migrate existing DBs that predate the clip_path column
        try:
            self._conn.execute("ALTER TABLE events ADD COLUMN clip_path TEXT")
            self._conn.commit()
        except sqlite3.OperationalError:
            pass  # column already exists

    def log(self, item: str, event: str, zone: Optional[str] = None,
            detail: str = "", confidence: float = 0.0,
            ts: Optional[float] = None,
            clip_path: Optional[str] = None) -> None:
        self._conn.execute(
            "INSERT INTO events"
            "  (ts, item, event, zone, detail, confidence, clip_path)"
            "  VALUES (?, ?, ?, ?, ?, ?, ?)",
            (ts or time.time(), item, event, zone, detail, confidence, clip_path),
        )
        self._conn.commit()

    def last_event(self, item: str,
                   event: Optional[str] = None) -> Optional[sqlite3.Row]:
        if event:
            return self._conn.execute(
                "SELECT * FROM events WHERE item=? AND event=?"
                " ORDER BY ts DESC LIMIT 1",
                (item, event),
            ).fetchone()
        return self._conn.execute(
            "SELECT * FROM events WHERE item=? ORDER BY ts DESC LIMIT 1",
            (item,),
        ).fetchone()

    def last_clip_path(self, item: str) -> Optional[str]:
        """Return path of the most recent clip for item, or None."""
        row = self._conn.execute(
            "SELECT clip_path FROM events"
            " WHERE item=? AND clip_path IS NOT NULL"
            " ORDER BY ts DESC LIMIT 1",
            (item,),
        ).fetchone()
        return row["clip_path"] if row else None

    def close(self) -> None:
        self._conn.close()
