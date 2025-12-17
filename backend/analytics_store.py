from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional


class AnalyticsStore:
    """Persistent store for display + reward events (for later analysis/clustering)."""

    def __init__(self, db_path: Path) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=30)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=NORMAL;")
        return conn

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS display_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    created_at TEXT NOT NULL,
                    step INTEGER,
                    pageid INTEGER NOT NULL,
                    selection_type TEXT,
                    source_type TEXT,
                    score REAL,
                    predicted_mean_reward REAL,
                    title TEXT,
                    extract TEXT,
                    url TEXT,
                    thumbnail_url TEXT,
                    embedding TEXT,
                    snapshot_loaded TEXT
                );
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_display_events_created_at ON display_events(created_at);"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_display_events_pageid ON display_events(pageid);"
            )

            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS reward_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    created_at TEXT NOT NULL,
                    step INTEGER,
                    pageid INTEGER NOT NULL,
                    reward REAL NOT NULL,
                    selection_type TEXT,
                    event_type TEXT,
                    dwell_time_ms INTEGER,
                    source_type TEXT,
                    title TEXT,
                    extract TEXT,
                    url TEXT,
                    thumbnail_url TEXT,
                    embedding TEXT,
                    snapshot_loaded TEXT
                );
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_reward_events_created_at ON reward_events(created_at);"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_reward_events_pageid ON reward_events(pageid);"
            )

            # Lightweight migrations: add columns if we evolve the schema.
            self._ensure_column(conn, "display_events", "extract", "TEXT")
            self._ensure_column(conn, "reward_events", "extract", "TEXT")
            self._ensure_column(conn, "display_events", "batch_predicted_mean_reward_avg", "REAL")
            self._ensure_column(conn, "display_events", "delta_vs_batch_avg", "REAL")

    def _ensure_column(self, conn: sqlite3.Connection, table: str, col: str, col_type: str) -> None:
        try:
            cols = conn.execute(f"PRAGMA table_info({table});").fetchall()
            existing = {r["name"] for r in cols}
            if col not in existing:
                conn.execute(f"ALTER TABLE {table} ADD COLUMN {col} {col_type};")
        except Exception:  # noqa: BLE001
            return

    def log_display(self, row: Dict[str, Any]) -> None:
        self._insert("display_events", row)

    def log_reward(self, row: Dict[str, Any]) -> None:
        self._insert("reward_events", row)

    def _insert(self, table: str, row: Dict[str, Any]) -> None:
        if not row:
            return

        # Normalize embedding payload (store as JSON text).
        if "embedding" in row and row["embedding"] is not None:
            if not isinstance(row["embedding"], str):
                try:
                    row["embedding"] = json.dumps(row["embedding"])
                except Exception:  # noqa: BLE001
                    row["embedding"] = None

        cols = list(row.keys())
        placeholders = ",".join(["?"] * len(cols))
        col_sql = ",".join(cols)
        values = [row[c] for c in cols]

        with self._connect() as conn:
            conn.execute(
                f"INSERT INTO {table} ({col_sql}) VALUES ({placeholders});",
                values,
            )

    def count_rewards(self) -> int:
        with self._connect() as conn:
            row = conn.execute("SELECT COUNT(1) AS n FROM reward_events;").fetchone()
            return int(row["n"]) if row else 0

    def latest_reward(self) -> Optional[Dict[str, Any]]:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM reward_events ORDER BY id DESC LIMIT 1;"
            ).fetchone()
            return dict(row) if row else None

    def get_liked_embeddings(self, *, min_reward: float = 0.0, limit: int = 250) -> List[Dict[str, Any]]:
        """Return distinct (by pageid) liked items with embeddings from reward_events.

        We treat reward > min_reward as "liked". Results are returned newest-first.
        """
        limit = max(0, int(limit))
        if limit <= 0:
            return []

        min_reward = float(min_reward)

        # Pull more rows than needed so dedup-by-pageid still yields `limit` items.
        max_rows = max(50, limit * 8)

        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT pageid, title, reward, url, embedding
                FROM reward_events
                WHERE reward > ?
                  AND embedding IS NOT NULL
                ORDER BY id DESC
                LIMIT ?;
                """,
                (min_reward, max_rows),
            ).fetchall()

        out: List[Dict[str, Any]] = []
        seen: set[int] = set()
        for r in rows:
            try:
                pageid = int(r["pageid"])
            except Exception:  # noqa: BLE001
                continue
            if pageid in seen:
                continue
            seen.add(pageid)

            emb_txt = r["embedding"]
            if emb_txt is None:
                continue
            try:
                emb = json.loads(emb_txt) if isinstance(emb_txt, str) else emb_txt
            except Exception:  # noqa: BLE001
                continue
            if not isinstance(emb, list):
                continue

            out.append(
                {
                    "pageid": pageid,
                    "title": str(r["title"] or ""),
                    "url": str(r["url"] or ""),
                    "reward": float(r["reward"] or 0.0),
                    "embedding": emb,
                }
            )
            if len(out) >= limit:
                break

        return out

    def get_labeled_embeddings(
        self,
        *,
        like_threshold: float = 0.0,
        limit_per_class: int = 250,
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Return recent distinct (by pageid) samples split into liked vs disliked.

        - liked: reward > like_threshold
        - disliked: reward <= like_threshold
        """
        like_threshold = float(like_threshold)
        limit_per_class = max(0, int(limit_per_class))
        if limit_per_class <= 0:
            return {"liked": [], "disliked": []}

        max_rows = max(100, limit_per_class * 12)

        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT id, pageid, title, reward, url, embedding
                FROM reward_events
                WHERE embedding IS NOT NULL
                ORDER BY id DESC
                LIMIT ?;
                """,
                (max_rows,),
            ).fetchall()

        liked: List[Dict[str, Any]] = []
        disliked: List[Dict[str, Any]] = []
        seen: set[int] = set()

        for r in rows:
            if len(liked) >= limit_per_class and len(disliked) >= limit_per_class:
                break
            try:
                pageid = int(r["pageid"])
            except Exception:  # noqa: BLE001
                continue
            if pageid in seen:
                continue
            seen.add(pageid)

            emb_txt = r["embedding"]
            if emb_txt is None:
                continue
            try:
                emb = json.loads(emb_txt) if isinstance(emb_txt, str) else emb_txt
            except Exception:  # noqa: BLE001
                continue
            if not isinstance(emb, list):
                continue

            sample = {
                "pageid": pageid,
                "title": str(r["title"] or ""),
                "url": str(r["url"] or ""),
                "reward": float(r["reward"] or 0.0),
                "embedding": emb,
            }

            if float(sample["reward"]) > like_threshold:
                if len(liked) < limit_per_class:
                    liked.append(sample)
            else:
                if len(disliked) < limit_per_class:
                    disliked.append(sample)

        return {"liked": liked, "disliked": disliked}

    def get_display_delta_embeddings(self, *, limit: int = 400) -> List[Dict[str, Any]]:
        """Return recent distinct (by pageid) display samples with embeddings + delta_vs_batch_avg.

        This is used for the "delta map" visualization: points are colored by whether
        delta_vs_batch_avg > 0 (above batch average predicted reward) or <= 0.
        """
        limit = max(0, int(limit))
        if limit <= 0:
            return []

        # Pull more rows than needed so dedup-by-pageid still yields `limit` items.
        max_rows = max(200, limit * 12)

        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT id, pageid, title, url, embedding,
                       predicted_mean_reward, batch_predicted_mean_reward_avg, delta_vs_batch_avg
                FROM display_events
                WHERE embedding IS NOT NULL
                ORDER BY id DESC
                LIMIT ?;
                """,
                (max_rows,),
            ).fetchall()

        out: List[Dict[str, Any]] = []
        seen: set[int] = set()
        for r in rows:
            if len(out) >= limit:
                break

            try:
                pageid = int(r["pageid"])
            except Exception:  # noqa: BLE001
                continue
            if pageid in seen:
                continue
            seen.add(pageid)

            emb_txt = r["embedding"]
            if emb_txt is None:
                continue
            try:
                emb = json.loads(emb_txt) if isinstance(emb_txt, str) else emb_txt
            except Exception:  # noqa: BLE001
                continue
            if not isinstance(emb, list):
                continue

            delta = r["delta_vs_batch_avg"]
            if delta is None:
                continue
            try:
                delta_f = float(delta)
            except Exception:  # noqa: BLE001
                continue

            out.append(
                {
                    "pageid": pageid,
                    "title": str(r["title"] or ""),
                    "url": str(r["url"] or ""),
                    "predicted_mean_reward": float(r["predicted_mean_reward"] or 0.0),
                    "batch_predicted_mean_reward_avg": float(
                        r["batch_predicted_mean_reward_avg"] or 0.0
                    ),
                    "delta_vs_batch_avg": delta_f,
                    "embedding": emb,
                }
            )

        return out


