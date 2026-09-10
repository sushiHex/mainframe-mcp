"""Append-only event store: SQLite in WAL mode so the daemon and the
hooks can append concurrently on Windows without a lock file. One short-lived
connection per call keeps it thread-safe and crash-only."""

import contextlib
import json
import sqlite3
from datetime import datetime
from pathlib import Path

_SCHEMA = """
CREATE TABLE IF NOT EXISTS events (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  ts TEXT NOT NULL,
  kind TEXT NOT NULL,
  project TEXT,
  path TEXT,
  detail TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS events_kind_id ON events(kind, id);
"""


class EventStore:
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as con:
            con.executescript(_SCHEMA)

    @contextlib.contextmanager
    def _connect(self):
        con = sqlite3.connect(str(self.path), timeout=10, isolation_level=None)
        try:
            con.execute("PRAGMA journal_mode=WAL")
            con.execute("PRAGMA synchronous=NORMAL")
            yield con
        finally:
            con.close()

    def emit(self, kind: str, project=None, path=None, **detail) -> int:
        payload = json.dumps(detail, default=str, ensure_ascii=False)
        with self._connect() as con:
            cur = con.execute("INSERT INTO events(ts, kind, project, path, detail) VALUES (?,?,?,?,?)",
                              (datetime.now().isoformat(timespec="seconds"), kind, project, path, payload))
            return int(cur.lastrowid)

    def recent(self, n: int = 20, kind=None) -> list:
        q, args = "SELECT id, ts, kind, project, path, detail FROM events", ()
        if kind:
            q, args = q + " WHERE kind = ?", (kind,)
        q += " ORDER BY id DESC LIMIT ?"
        with self._connect() as con:
            rows = con.execute(q, args + (int(n),)).fetchall()
        return [{"id": r[0], "ts": r[1], "kind": r[2], "project": r[3], "path": r[4], "detail": json.loads(r[5])}
                for r in rows]

    def count(self, kind=None) -> int:
        with self._connect() as con:
            return int(con.execute("SELECT COUNT(*) FROM events WHERE ? IS NULL OR kind = ?", (kind, kind)).fetchone()[0])
