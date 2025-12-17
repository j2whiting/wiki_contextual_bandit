from __future__ import annotations

import datetime
import json
import sqlite3
from pathlib import Path
from typing import Any, Dict, List


class QueueStore:
    """A small persistent FIFO queue shared across processes (SQLite).

    Stores full card payloads (including embeddings) so an external worker can
    enqueue and the FastAPI app can dequeue.
    """

    def __init__(self, db_path: Path) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=30)
        conn.row_factory = sqlite3.Row
        # Better cross-process concurrency.
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=NORMAL;")
        return conn

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS recommendation_queue (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    pageid INTEGER NOT NULL UNIQUE,
                    payload TEXT NOT NULL,
                    enqueued_at TEXT NOT NULL
                );
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_recommendation_queue_id ON recommendation_queue(id);"
            )
            # Persistently track served pageids so we never resurface an article twice
            # (unless the user wipes the volume via `docker compose down -v`).
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS seen_articles (
                    pageid INTEGER PRIMARY KEY,
                    seen_at TEXT NOT NULL
                );
                """
            )
            conn.execute("CREATE INDEX IF NOT EXISTS idx_seen_articles_seen_at ON seen_articles(seen_at);")

    def size(self) -> int:
        with self._connect() as conn:
            row = conn.execute("SELECT COUNT(1) AS n FROM recommendation_queue;").fetchone()
            return int(row["n"]) if row else 0

    def enqueue(self, payloads: List[Dict[str, Any]]) -> int:
        if not payloads:
            return 0

        rows = []
        for p in payloads:
            pageid = p.get("pageid")
            if pageid is None:
                continue
            enqueued_at = p.get("enqueued_at") or ""
            rows.append((int(pageid), json.dumps(p), str(enqueued_at)))

        if not rows:
            return 0

        with self._connect() as conn:
            before = conn.total_changes
            # Only enqueue if this pageid has never been served before.
            conn.executemany(
                """
                INSERT OR IGNORE INTO recommendation_queue (pageid, payload, enqueued_at)
                SELECT ?, ?, ?
                WHERE NOT EXISTS (
                    SELECT 1 FROM seen_articles WHERE pageid = ?
                );
                """,
                [(pid, payload, enq, pid) for (pid, payload, enq) in rows],
            )
            return int(conn.total_changes - before)

    def dequeue(self, limit: int) -> List[Dict[str, Any]]:
        limit = int(limit)
        if limit <= 0:
            return []

        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE;")
            rows = conn.execute(
                """
                SELECT id, pageid, payload
                FROM recommendation_queue
                ORDER BY id
                LIMIT ?;
                """,
                (limit,),
            ).fetchall()

            if not rows:
                conn.execute("COMMIT;")
                return []

            now = datetime.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"
            pageids = [int(r["pageid"]) for r in rows]
            conn.executemany(
                "INSERT OR IGNORE INTO seen_articles (pageid, seen_at) VALUES (?, ?);",
                [(pid, now) for pid in pageids],
            )

            ids = [int(r["id"]) for r in rows]
            placeholders = ",".join(["?"] * len(ids))
            conn.execute(f"DELETE FROM recommendation_queue WHERE id IN ({placeholders});", ids)
            conn.execute("COMMIT;")

        payloads: List[Dict[str, Any]] = []
        for r in rows:
            try:
                payloads.append(json.loads(r["payload"]))
            except Exception:  # noqa: BLE001
                continue
        return payloads

    def list_all(self) -> List[Dict[str, Any]]:
        """Return all queued payloads in current dequeue order."""
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT payload
                FROM recommendation_queue
                ORDER BY id;
                """
            ).fetchall()

        payloads: List[Dict[str, Any]] = []
        for r in rows:
            try:
                payloads.append(json.loads(r["payload"]))
            except Exception:  # noqa: BLE001
                continue
        return payloads

    def replace_all(self, payloads: List[Dict[str, Any]]) -> int:
        """Atomically replace the queue contents with the provided payloads, in order.

        Uses a single IMMEDIATE transaction so writers (worker/API) serialize safely.
        """
        rows = []
        for p in payloads:
            pageid = p.get("pageid")
            if pageid is None:
                continue
            enqueued_at = p.get("enqueued_at") or p.get("rescored_at") or ""
            rows.append((int(pageid), json.dumps(p), str(enqueued_at)))

        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE;")
            conn.execute("DELETE FROM recommendation_queue;")
            if rows:
                conn.executemany(
                    """
                    INSERT OR IGNORE INTO recommendation_queue (pageid, payload, enqueued_at)
                    SELECT ?, ?, ?
                    WHERE NOT EXISTS (
                        SELECT 1 FROM seen_articles WHERE pageid = ?
                    );
                    """,
                    [(pid, payload, enq, pid) for (pid, payload, enq) in rows],
                )
            conn.execute("COMMIT;")

            row = conn.execute("SELECT COUNT(1) AS n FROM recommendation_queue;").fetchone()
            return int(row["n"]) if row else 0


