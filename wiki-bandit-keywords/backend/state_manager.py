from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import time
import threading
from typing import Dict, List, Literal, Optional

import math
import random
import re

import numpy as np
from pydantic import BaseModel

try:
    from .bandit import LinearThompsonSamplingBandit
    from .analytics_store import AnalyticsStore
    from .persistence import atomic_write_json, read_json, utc_now_compact, utc_now_iso
    from .queue_store import QueueStore
    from .wiki_client import WikipediaClient, WikiPageSummary
except ImportError:  # pragma: no cover
    from bandit import LinearThompsonSamplingBandit
    from analytics_store import AnalyticsStore
    from persistence import atomic_write_json, read_json, utc_now_compact, utc_now_iso
    from queue_store import QueueStore
    from wiki_client import WikipediaClient, WikiPageSummary

SelectionType = Literal["explore", "exploit"]
SourceType = Literal["random", "neighbor", "search"]


# lightweight stopword list for naive keyword extraction
STOPWORDS = {
    "the",
    "and",
    "for",
    "with",
    "that",
    "this",
    "from",
    "were",
    "have",
    "which",
    "about",
    "into",
    "after",
    "their",
    "there",
    "been",
    "also",
    "such",
    "than",
    "over",
    "under",
    "between",
    "within",
    "they",
    "them",
    "when",
    "will",
    "would",
    "could",
    "should",
    "might",
    "your",
    "using",
    "where",
    "while",
    "these",
    "those",
    "some",
    "many",
    "most",
    "more",
    "other",
    "only",
    "very",
    "like",
    "just",
    "its",
    "being",
    "used",
    "use",
    "can",
    "may",
    "each",
    "per",
    "one",
    "two",
    "three",
    "four",
    "five",
    "first",
    "second",
    "third",
    "best",
    "better",
    "worse",
    "greatest",
    "notable",
    "famous",
    "popular",
    "known",
    "based",
    "called",
    "named",
    "direct",
    "directed",
    "director",
    "directing",
    "written",
    "writer",
    "writing",
    "produced",
    "producer",
    "production",
    "released",
    "release",
}


@dataclass
class Article:
    pageid: int
    title: str
    extract: str
    url: str
    embedding: np.ndarray
    source_type: SourceType
    thumbnail_url: Optional[str] = None
    times_shown: int = 0
    total_reward: float = 0.0
    times_rewarded: int = 0


@dataclass
class DisplayEvent:
    pageid: int
    selection_type: SelectionType
    score: float


@dataclass
class RewardEvent:
    step: int
    pageid: int
    reward: float
    selection_type: Optional[SelectionType] = None


@dataclass
class KeywordStat:
    total_weight: float = 0.0
    count: int = 0
    last_seen_step: int = 0


class ArticleCard(BaseModel):
    pageid: int
    title: str
    extract: str
    url: str
    thumbnail_url: Optional[str] = None
    source_type: SourceType
    selection_type: SelectionType
    score: float
    predicted_mean_reward: float


class RewardIn(BaseModel):
    pageid: int
    reward: float
    selection_type: Optional[SelectionType] = None
    event_type: str
    dwell_time_ms: Optional[int] = None


class RewardPoint(BaseModel):
    step: int
    reward: float


class TopArticle(BaseModel):
    pageid: int
    title: str
    mean_reward: float
    total_reward: float
    times_rewarded: int


class CalibrationBucket(BaseModel):
    label: str
    predicted_low: float
    predicted_high: float
    actual_rate: float
    count: int


class StateResponse(BaseModel):
    interactions: int
    avg_reward: float
    reward_history: List[RewardPoint]
    exploration_count: int
    exploitation_count: int
    top_articles: List[TopArticle]
    # Convergence metrics
    hit_rate: float  # clicks / total interactions
    rolling_avg_reward: float  # last N rewards averaged
    rolling_hit_rates: List[RewardPoint]  # rolling hit rate over time
    calibration_buckets: List[CalibrationBucket]  # predicted reward bins with actual rates
    rewards_since_last_rerank: int  # for progress bar


class SnapshotMeta(BaseModel):
    name: str
    saved_at: Optional[str] = None
    interactions: Optional[int] = None
    delta_interactions: Optional[int] = None
    l2_delta_theta_mean: Optional[float] = None


class ArticleStore:
    """In-memory article pool + bandit + keyword inventory."""

    def __init__(
        self,
        embed_model,
        wiki_client: WikipediaClient,
        bandit: LinearThompsonSamplingBandit,
        embedding_dim: int = 64,
        min_pool_size: int = 40,
        exploration_fraction: float = 0.3,
        state_path: Optional[Path] = None,
        checkpoint_every_batches: int = 10,
        checkpoint_min_interval_s: float = 5.0,
        snapshot_dir: Optional[Path] = None,
        snapshot_prefix: str = "model",
        queue_db_path: Optional[Path] = None,
        analytics_db_path: Optional[Path] = None,
        max_keywords: int = 800,
        max_persisted_reward_history: int = 2000,
        allowed_source_types: Optional[List[SourceType]] = None,
    ) -> None:
        self.embed_model = embed_model
        self.wiki_client = wiki_client
        self.bandit = bandit
        self.embedding_dim = embedding_dim
        self.min_pool_size = min_pool_size
        self.exploration_fraction = exploration_fraction

        # Controls how the candidate pool refills. The worker can use this to run
        # "discovery-only" mode (i.e. random articles only).
        allowed: set[str]
        if allowed_source_types is None:
            allowed = {"random", "neighbor", "search"}
        else:
            allowed = set(allowed_source_types)
        allowed = {s for s in allowed if s in ("random", "neighbor", "search")}
        if not allowed:
            allowed = {"random"}
        self.allowed_source_types = allowed

        # Persistence + worker configuration
        self.state_path = state_path
        self.checkpoint_every_batches = int(checkpoint_every_batches)
        self.checkpoint_min_interval_s = float(checkpoint_min_interval_s)
        self.snapshot_dir = snapshot_dir
        self.snapshot_prefix = str(snapshot_prefix)
        self.queue_db_path = queue_db_path
        self.queue_store: Optional[QueueStore] = (
            QueueStore(Path(queue_db_path)) if queue_db_path is not None else None
        )
        self.analytics_db_path = analytics_db_path
        self.analytics_store: Optional[AnalyticsStore] = (
            AnalyticsStore(Path(analytics_db_path)) if analytics_db_path is not None else None
        )
        self.max_keywords = int(max_keywords)
        self.max_persisted_reward_history = int(max_persisted_reward_history)

        self._loaded_snapshot_name: Optional[str] = None

        # Thread-safety: protect all mutable state that can be touched by API + worker.
        self._lock = threading.RLock()
        self._refill_lock = threading.Lock()
        self.batches_generated: int = 0
        self.batches_served: int = 0
        self._dirty: bool = False
        self._last_checkpoint_ts: float = 0.0

        self.articles: Dict[int, Article] = {}
        self.candidate_ids: set[int] = set()

        self.display_events: List[DisplayEvent] = []
        self.reward_events: List[RewardEvent] = []

        self.exploration_count: int = 0
        self.exploitation_count: int = 0
        self.rewards_since_last_rerank: int = 0

        # keyword_stats[keyword] = KeywordStat
        self.keyword_stats: Dict[str, KeywordStat] = {}

    # ---------- Embedding + article helpers ----------

    def _embed_text(self, text: str) -> np.ndarray:
        vecs = self.embed_model.encode(
            [text],
            truncate_dim=self.embedding_dim,
        )
        return np.asarray(vecs[0], dtype=np.float64)

    def _add_article(self, summary: WikiPageSummary, source_type: SourceType) -> None:
        with self._lock:
            if summary.pageid in self.articles:
                self.candidate_ids.add(summary.pageid)
                return

        text = summary.extract.strip() or summary.title
        if not text:
            return

        embedding = self._embed_text(text)

        article = Article(
            pageid=summary.pageid,
            title=summary.title,
            extract=summary.extract.strip(),
            url=summary.url,
            embedding=embedding,
            source_type=source_type,
            thumbnail_url=getattr(summary, "thumbnail_url", None),
        )
        with self._lock:
            # Double-check because another thread may have inserted while we embedded.
            if summary.pageid in self.articles:
                self.candidate_ids.add(summary.pageid)
                return
            self.articles[summary.pageid] = article
            self.candidate_ids.add(summary.pageid)

    def _top_rewarded_articles(self, limit: int = 5) -> List[Article]:
        scored = [
            (a.total_reward / a.times_rewarded, a)
            for a in self.articles.values()
            if a.times_rewarded > 0
        ]
        scored.sort(key=lambda pair: pair[0], reverse=True)
        return [a for _, a in scored[:limit]]

    # ---------- Keyword extraction + maintenance ----------

    def _normalize_keyword(self, tok: str) -> str:
        tok = (tok or "").strip().lower()
        if not tok:
            return ""

        # Normalize simple plurals (keeps "class" -> "class", drops "cats" -> "cat").
        if tok.endswith("ies") and len(tok) > 6:
            tok = tok[:-3] + "y"
        elif tok.endswith("s") and len(tok) > 5 and not tok.endswith("ss"):
            tok = tok[:-1]

        return tok

    def _extract_keywords_from_article(self, article: Article) -> List[str]:
        text = f"{article.title}. {article.extract}"
        text = text.lower()
        # tokens that start with a letter, length >= 4 (avoids pure numbers)
        tokens = re.findall(r"[a-z][a-z0-9]{3,}", text)

        keywords: List[str] = []
        seen: set[str] = set()
        for tok in tokens:
            tok = self._normalize_keyword(tok)
            if not tok:
                continue
            if tok in STOPWORDS:
                continue
            if tok in seen:
                continue
            seen.add(tok)
            keywords.append(tok)
            if len(keywords) >= 20:
                break
        return keywords

    def _update_keywords(self, article: Article, reward: float) -> None:
        if reward <= 0:
            return

        keywords = self._extract_keywords_from_article(article)
        if not keywords:
            return

        step = self.bandit.interactions
        for kw in keywords:
            stat = self.keyword_stats.get(kw)
            if stat is None:
                stat = KeywordStat()
                self.keyword_stats[kw] = stat
            stat.total_weight += reward
            stat.count += 1
            stat.last_seen_step = step

    def _build_keyword_queries(
        self,
        max_queries: int = 8,
        min_keywords_per_query: int = 1,
        max_keywords_per_query: int = 3,
    ) -> List[str]:
        if not self.keyword_stats:
            return []

        step = self.bandit.interactions
        scored_keywords: List[tuple[str, float]] = []
        for kw, stat in self.keyword_stats.items():
            if kw in STOPWORDS:
                continue
            age = max(0, step - stat.last_seen_step)
            decay = math.exp(-age / 200.0)
            base = stat.total_weight / (1.0 + math.log1p(stat.count))
            score = base * decay
            if score > 0:
                scored_keywords.append((kw, score))

        if not scored_keywords:
            return []

        keywords = [kw for kw, _ in scored_keywords]
        weights = [score for _, score in scored_keywords]

        queries: List[str] = []
        for _ in range(max_queries):
            k = random.randint(min_keywords_per_query, max_keywords_per_query)
            chosen = random.choices(keywords, weights=weights, k=k)
            chosen = list(dict.fromkeys(chosen))
            if not chosen:
                continue
            q = " ".join(chosen)
            if q not in queries:
                queries.append(q)
        return queries

    def _prune_keywords(
        self,
        min_weight: float = 0.3,
        max_age: int = 1500,
        max_keywords: Optional[int] = None,
    ) -> None:
        if not self.keyword_stats:
            return

        step = self.bandit.interactions
        to_delete: List[str] = []
        for kw, stat in self.keyword_stats.items():
            if kw in STOPWORDS:
                to_delete.append(kw)
                continue
            age = max(0, step - stat.last_seen_step)
            if age > max_age and stat.total_weight < min_weight:
                to_delete.append(kw)
        for kw in to_delete:
            del self.keyword_stats[kw]

        max_keywords = int(max_keywords) if max_keywords is not None else int(self.max_keywords)
        if max_keywords > 0 and len(self.keyword_stats) > max_keywords:
            scored: List[tuple[str, float]] = []
            for kw, stat in self.keyword_stats.items():
                age = max(0, step - stat.last_seen_step)
                decay = math.exp(-age / 200.0)
                base = stat.total_weight / (1.0 + math.log1p(stat.count))
                score = base * decay
                scored.append((kw, score))
            scored.sort(key=lambda pair: pair[1], reverse=True)
            keep = set(kw for kw, _ in scored[:max_keywords])
            for kw in list(self.keyword_stats.keys()):
                if kw not in keep:
                    del self.keyword_stats[kw]

    # ---------- Candidate pool refill ----------

    def _refill_candidates(self) -> None:
        """Refill pool via neighbors, keyword search, and random exploration."""
        # Ensure only one thread performs network I/O + embedding at a time.
        with self._refill_lock:
            allowed = self.allowed_source_types
            with self._lock:
                seeds = self._top_rewarded_articles(limit=5) if "neighbor" in allowed else []
                keyword_queries = self._build_keyword_queries(max_queries=8) if "search" in allowed else []

            # 1) local exploitation via neighbors
            if "neighbor" in allowed:
                try:
                    for seed in seeds:
                        neighbors = self.wiki_client.get_linked_articles(
                            seed.pageid,
                            max_articles=6,
                        )
                        for n in neighbors:
                            self._add_article(n, source_type="neighbor")
                except Exception as exc:  # noqa: BLE001
                    print(f"[WARN] Error fetching neighbors: {exc}")

            # 2) global exploitation via keyword-driven search
            if "search" in allowed:
                try:
                    for q in keyword_queries:
                        results = self.wiki_client.search_articles(
                            query_text=q,
                            limit=6,
                        )
                        for r in results:
                            self._add_article(r, source_type="search")
                except Exception as exc:  # noqa: BLE001
                    print(f"[WARN] Error fetching search-based articles: {exc}")

            # 3) pure exploration
            if "random" in allowed:
                try:
                    random_articles = self.wiki_client.get_random_articles(limit=20)
                    for ra in random_articles:
                        self._add_article(ra, source_type="random")
                except Exception as exc:  # noqa: BLE001
                    print(f"[WARN] Error fetching random articles: {exc}")

            with self._lock:
                self._prune_keywords()

    def _ensure_candidates(self) -> None:
        with self._lock:
            need_refill = len(self.candidate_ids) < self.min_pool_size
        if need_refill:
            self._refill_candidates()

    # ---------- Persistence ----------

    def _snapshot_glob(self) -> str:
        return f"{self.snapshot_prefix}_*.json"

    def _latest_snapshot_path(self) -> Optional[Path]:
        if self.snapshot_dir is None:
            return None
        snap_dir = Path(self.snapshot_dir)
        if not snap_dir.exists():
            return None
        candidates = [p for p in snap_dir.glob(self._snapshot_glob()) if p.is_file()]
        if not candidates:
            return None
        # Filenames embed a lexicographically sortable UTC timestamp.
        return max(candidates, key=lambda p: p.name)

    def _snapshot_name(self) -> str:
        return f"{self.snapshot_prefix}_{utc_now_compact()}.json"

    def _compute_bandit_diff(self, prev_bandit: Dict, cur_bandit: Dict) -> Dict:
        """Compute per-dimension + summary diffs for visualization."""
        try:
            prev_A = np.asarray(prev_bandit.get("A_diag", []), dtype=np.float64).reshape(-1)
            prev_b = np.asarray(prev_bandit.get("b", []), dtype=np.float64).reshape(-1)
            cur_A = np.asarray(cur_bandit.get("A_diag", []), dtype=np.float64).reshape(-1)
            cur_b = np.asarray(cur_bandit.get("b", []), dtype=np.float64).reshape(-1)
            if prev_A.shape != cur_A.shape or prev_b.shape != cur_b.shape or cur_A.size == 0:
                return {}

            prev_theta = prev_b / prev_A
            cur_theta = cur_b / cur_A

            delta_theta = cur_theta - prev_theta
            delta_b = cur_b - prev_b
            delta_A = cur_A - prev_A

            return {
                "delta_interactions": int(cur_bandit.get("interactions", 0))
                - int(prev_bandit.get("interactions", 0)),
                "metrics": {
                    "l2_delta_theta_mean": float(np.linalg.norm(delta_theta)),
                    "max_abs_delta_theta_mean": float(np.max(np.abs(delta_theta))),
                    "l2_delta_b": float(np.linalg.norm(delta_b)),
                    "l2_delta_A_diag": float(np.linalg.norm(delta_A)),
                },
                # Small enough (dim=256) to store for quick UI plotting.
                "delta_theta_mean": delta_theta.astype(float).tolist(),
            }
        except Exception:  # noqa: BLE001
            return {}

    def _build_state_payload(self, *, include_diff: bool = False) -> Dict:
        with self._lock:
            bandit_state = self.bandit.to_state_dict(
                max_reward_history=self.max_persisted_reward_history
            )
            payload = {
                "version": 1,
                "saved_at": utc_now_iso(),
                "bandit": bandit_state,
                "keyword_stats": {
                    kw: {
                        "total_weight": float(stat.total_weight),
                        "count": int(stat.count),
                        "last_seen_step": int(stat.last_seen_step),
                    }
                    for kw, stat in self.keyword_stats.items()
                },
                "counters": {
                    "exploration_count": int(self.exploration_count),
                    "exploitation_count": int(self.exploitation_count),
                    "batches_generated": int(self.batches_generated),
                    "batches_served": int(self.batches_served),
                    "rewards_since_last_rerank": int(self.rewards_since_last_rerank),
                },
            }

        if include_diff and self.snapshot_dir is not None:
            prev_path = self._latest_snapshot_path()
            prev_payload = read_json(prev_path) if prev_path else None
            if isinstance(prev_payload, dict):
                diff = self._compute_bandit_diff(
                    prev_payload.get("bandit", {}),
                    bandit_state,
                )
                if diff:
                    payload["diff"] = {
                        "from_snapshot": prev_path.name if prev_path else None,
                        "from_saved_at": prev_payload.get("saved_at"),
                        **diff,
                    }

        return payload

    def _save_snapshot(self, payload: Dict) -> Optional[Path]:
        if self.snapshot_dir is None:
            return None
        snap_dir = Path(self.snapshot_dir)
        snap_path = snap_dir / self._snapshot_name()
        atomic_write_json(snap_path, payload)
        with self._lock:
            self._loaded_snapshot_name = snap_path.name
        return snap_path

    def list_snapshots(self, limit: int = 50) -> List[SnapshotMeta]:
        if self.snapshot_dir is None:
            return []
        snap_dir = Path(self.snapshot_dir)
        if not snap_dir.exists():
            return []
        paths = [p for p in snap_dir.glob(self._snapshot_glob()) if p.is_file()]
        paths.sort(key=lambda p: p.name, reverse=True)
        items: List[SnapshotMeta] = []
        for p in paths[: max(0, int(limit))]:
            payload = read_json(p) or {}
            bandit = payload.get("bandit", {}) if isinstance(payload, dict) else {}
            diff = payload.get("diff", {}) if isinstance(payload, dict) else {}
            metrics = diff.get("metrics", {}) if isinstance(diff, dict) else {}
            items.append(
                SnapshotMeta(
                    name=p.name,
                    saved_at=payload.get("saved_at") if isinstance(payload, dict) else None,
                    interactions=bandit.get("interactions") if isinstance(bandit, dict) else None,
                    delta_interactions=diff.get("delta_interactions")
                    if isinstance(diff, dict)
                    else None,
                    l2_delta_theta_mean=metrics.get("l2_delta_theta_mean")
                    if isinstance(metrics, dict)
                    else None,
                )
            )
        return items

    def get_snapshot_payload(self, name: str) -> Optional[Dict]:
        if self.snapshot_dir is None:
            return None
        name = (name or "").strip()
        if not name:
            return None
        # Basic safety: only allow basename, no path traversal.
        if "/" in name or "\\" in name or ".." in name:
            return None
        path = Path(self.snapshot_dir) / name
        payload = read_json(path)
        if isinstance(payload, dict):
            return payload
        return None

    def load_state(self) -> bool:
        # Prefer the newest dated snapshot for ranking, fall back to state_path.
        payload = None
        latest_snap = self._latest_snapshot_path()
        if latest_snap is not None:
            payload = read_json(latest_snap)
            if payload:
                with self._lock:
                    self._loaded_snapshot_name = latest_snap.name
        if payload is None:
            if self.state_path is None:
                return False
            payload = read_json(Path(self.state_path))
        if not payload:
            return False

        try:
            with self._lock:
                self.bandit.load_state_dict(payload.get("bandit", {}))

                kw_payload = payload.get("keyword_stats", {})
                if isinstance(kw_payload, dict):
                    self.keyword_stats = {}
                    for kw, stat in kw_payload.items():
                        if not isinstance(stat, dict):
                            continue
                        self.keyword_stats[str(kw)] = KeywordStat(
                            total_weight=float(stat.get("total_weight", 0.0)),
                            count=int(stat.get("count", 0)),
                            last_seen_step=int(stat.get("last_seen_step", 0)),
                        )

                counters = payload.get("counters", {})
                if isinstance(counters, dict):
                    self.exploration_count = int(
                        counters.get("exploration_count", self.exploration_count)
                    )
                    self.exploitation_count = int(
                        counters.get("exploitation_count", self.exploitation_count)
                    )
                    self.batches_generated = int(
                        counters.get("batches_generated", self.batches_generated)
                    )
                    self.batches_served = int(counters.get("batches_served", self.batches_served))
                    self.rewards_since_last_rerank = int(
                        counters.get("rewards_since_last_rerank", self.rewards_since_last_rerank)
                    )

                self._dirty = False
                self._last_checkpoint_ts = time.time()
                # Enforce current stopwords + hard cap even if loading an old snapshot.
                self._prune_keywords()
            return True
        except Exception as exc:  # noqa: BLE001
            print(f"[WARN] Failed to load persisted state: {exc}")
            # If the persisted model doesn't match our configured embedding dim,
            # the queue likely contains incompatible embeddings too. Clear it so
            # the worker can repopulate immediately.
            try:
                msg = str(exc)
                if "Bandit dim mismatch" in msg and self.queue_store is not None:
                    self.queue_store.replace_all([])
            except Exception:  # noqa: BLE001
                pass
            return False

    def save_state(self) -> bool:
        try:
            payload = self._build_state_payload(include_diff=True)

            # 1) Always write the latest pointer file (if configured)
            if self.state_path is not None:
                atomic_write_json(Path(self.state_path), payload)

            # 2) Also write a dated snapshot (if configured)
            if self.snapshot_dir is not None:
                self._save_snapshot(payload)

            with self._lock:
                self._dirty = False
                self._last_checkpoint_ts = time.time()
            return True
        except Exception as exc:  # noqa: BLE001
            print(f"[WARN] Failed to save state: {exc}")
            return False

    def _maybe_checkpoint(self, *, force: bool = False) -> None:
        if self.state_path is None and self.snapshot_dir is None:
            return

        now = time.time()
        with self._lock:
            dirty = self._dirty
            last_ts = self._last_checkpoint_ts
            batches_served = self.batches_served
            interactions = self.bandit.interactions
            every = self.checkpoint_every_batches

        if not force and not dirty:
            return
        if not force and self.checkpoint_min_interval_s > 0:
            if now - last_ts < self.checkpoint_min_interval_s:
                return
        if (
            not force
            and every > 0
            and (batches_served % every) != 0
            and (interactions % every) != 0
        ):
            return

        self.save_state()

    # ---------- Recommendation generation ----------

    def _generate_batch(self, batch_size: int) -> List[ArticleCard]:
        batch_size = int(batch_size)
        if batch_size <= 0:
            return []

        self._ensure_candidates()

        with self._lock:
            if not self.candidate_ids:
                return []

            candidate_ids = list(self.candidate_ids)
            theta_sample = self.bandit.sample_theta()

            scored: List[tuple[int, float]] = []
            for pid in candidate_ids:
                article = self.articles.get(pid)
                if article is None:
                    continue
                score = float(np.dot(theta_sample, article.embedding))
                scored.append((pid, score))

            scored.sort(key=lambda pair: pair[1], reverse=True)
            remaining = [pid for pid, _ in scored]

            selected: List[tuple[int, SelectionType]] = []
            for _ in range(batch_size):
                if not remaining:
                    break

                if random.random() < self.exploration_fraction:
                    idx = random.randrange(len(remaining))
                    pid = remaining.pop(idx)
                    selection_type: SelectionType = "explore"
                    self.exploration_count += 1
                else:
                    pid = remaining.pop(0)
                    selection_type = "exploit"
                    self.exploitation_count += 1

                selected.append((pid, selection_type))
                self.candidate_ids.discard(pid)

            cards: List[ArticleCard] = []
            for pid, selection_type in selected:
                article = self.articles.get(pid)
                if article is None:
                    continue
                article.times_shown += 1

                mean_reward_pred = self.bandit.predict_mean_reward(article.embedding)
                score = float(np.dot(theta_sample, article.embedding))

                self.display_events.append(
                    DisplayEvent(pageid=pid, selection_type=selection_type, score=score)
                )

                cards.append(
                    ArticleCard(
                        pageid=article.pageid,
                        title=article.title,
                        extract=article.extract,
                        url=article.url,
                        thumbnail_url=article.thumbnail_url,
                        source_type=article.source_type,
                        selection_type=selection_type,
                        score=score,
                        predicted_mean_reward=float(mean_reward_pred),
                    )
                )

            return cards

    def queue_size(self) -> int:
        if self.queue_store is None:
            return 0
        return self.queue_store.size()

    def generate_and_enqueue(self, batch_size: int) -> int:
        """Generate a recommendation batch and enqueue it to the persistent queue.

        Intended to be called by an external worker process.
        """
        if self.queue_store is None:
            return 0

        cards = self._generate_batch(batch_size)
        if not cards:
            return 0

        payloads: List[Dict] = []
        with self._lock:
            for c in cards:
                article = self.articles.get(c.pageid)
                if article is None:
                    continue
                payloads.append(
                    {
                        "pageid": c.pageid,
                        "title": c.title,
                        "extract": c.extract,
                        "url": c.url,
                        "thumbnail_url": c.thumbnail_url,
                        "source_type": c.source_type,
                        "selection_type": c.selection_type,
                        "score": float(c.score),
                        "predicted_mean_reward": float(c.predicted_mean_reward),
                        # Critical: lets the API process update the bandit on reward.
                        "embedding": article.embedding.astype(float).tolist(),
                        # Metadata
                        "enqueued_at": utc_now_iso(),
                        "snapshot_loaded": self._loaded_snapshot_name,
                    }
                )

        inserted = self.queue_store.enqueue(payloads)
        with self._lock:
            if inserted > 0:
                self.batches_generated += 1
        return inserted

    def rescore_and_rebatch_queue(self, batch_size: int = 5) -> int:
        """Rescore *all* queued content under the current bandit and rewrite queue order.

        This is used by the "regenerate" button to immediately reflect the latest model.
        """
        if self.queue_store is None:
            return 0

        batch_size = int(batch_size)
        if batch_size <= 0:
            batch_size = 5

        queued = self.queue_store.list_all()
        if not queued:
            return 0

        with self._lock:
            theta_mean = self.bandit.theta_mean().astype(np.float64)
            exploration_fraction = float(self.exploration_fraction)
            snapshot_loaded = self._loaded_snapshot_name

        # Pre-parse embeddings once.
        items: List[Dict] = []
        embeddings: List[np.ndarray] = []
        for p in queued:
            try:
                emb = np.asarray(p.get("embedding", []), dtype=np.float64).reshape(-1)
                if emb.shape[0] != self.embedding_dim:
                    continue
                items.append(p)
                embeddings.append(emb)
            except Exception:  # noqa: BLE001
                continue

        if not items:
            return 0

        remaining = list(range(len(items)))
        new_order: List[Dict] = []

        # Generate new batches from the queued set, similar to _generate_batch().
        while remaining:
            # Sample a new theta per batch (Thompson Sampling) to emulate real serving.
            with self._lock:
                theta_sample = self.bandit.sample_theta().astype(np.float64)

            # Score all remaining items under this theta sample.
            scored: List[tuple[int, float]] = []
            for idx in remaining:
                s = float(np.dot(theta_sample, embeddings[idx]))
                scored.append((idx, s))
            scored.sort(key=lambda pair: pair[1], reverse=True)
            ordered = [idx for idx, _ in scored]

            # Select one batch with explore/exploit mix.
            batch_selected: List[tuple[int, str, float]] = []
            pool = ordered[:]
            for _ in range(min(batch_size, len(pool))):
                if not pool:
                    break
                if random.random() < exploration_fraction:
                    pick_pos = random.randrange(len(pool))
                    pick_idx = pool.pop(pick_pos)
                    sel = "explore"
                else:
                    pick_idx = pool.pop(0)
                    sel = "exploit"
                # keep the sampled score for display
                pick_score = next((s for i, s in scored if i == pick_idx), 0.0)
                batch_selected.append((pick_idx, sel, float(pick_score)))

            selected_indices = {i for i, _, _ in batch_selected}
            remaining = [i for i in remaining if i not in selected_indices]

            # Update payloads with refreshed scores/labels.
            for i, sel, s in batch_selected:
                p = dict(items[i])
                p["selection_type"] = sel
                p["score"] = float(s)
                p["predicted_mean_reward"] = float(np.dot(theta_mean, embeddings[i]))
                p["rescored_at"] = utc_now_iso()
                p["snapshot_loaded"] = snapshot_loaded
                new_order.append(p)

        with self._lock:
            self.rewards_since_last_rerank = 0
        return self.queue_store.replace_all(new_order)

    def _dequeue_from_persistent_queue(self, batch_size: int) -> List[ArticleCard]:
        if self.queue_store is None:
            return []

        payloads = self.queue_store.dequeue(batch_size)
        if not payloads:
            return []

        # Batch context: compute average predicted reward for the *served batch*.
        # Used to log delta_vs_batch_avg for embedding visualizations.
        predicted_vals: List[float] = []
        for p in payloads:
            try:
                emb_raw = p.get("embedding", None)
                if emb_raw is None:
                    continue
                emb = np.asarray(emb_raw, dtype=np.float64).reshape(-1)
                if emb.shape[0] != self.embedding_dim:
                    continue

                pv = p.get("predicted_mean_reward", None)
                if pv is None:
                    continue
                pvf = float(pv)
                if not math.isfinite(pvf):
                    continue
                predicted_vals.append(pvf)
            except Exception:  # noqa: BLE001
                continue

        batch_avg_pred: Optional[float] = (
            float(np.mean(np.asarray(predicted_vals, dtype=np.float64))) if predicted_vals else None
        )

        cards: List[ArticleCard] = []
        bad_embedding_dim = 0
        missing_embedding = 0
        other_errors = 0
        for p in payloads:
            try:
                pageid = int(p["pageid"])
                emb_raw = p.get("embedding", None)
                if emb_raw is None:
                    missing_embedding += 1
                    continue

                embedding = np.asarray(emb_raw, dtype=np.float64).reshape(-1)
                if embedding.shape[0] != self.embedding_dim:
                    bad_embedding_dim += 1
                    continue

                served_at = utc_now_iso()
                step_now: int
                with self._lock:
                    step_now = int(self.bandit.interactions)

                # Hydrate article in-memory so /api/reward can update the bandit.
                with self._lock:
                    article = self.articles.get(pageid)
                    if article is None:
                        article = Article(
                            pageid=pageid,
                            title=str(p.get("title", "")),
                            extract=str(p.get("extract", "")),
                            url=str(p.get("url", "")),
                            embedding=embedding,
                            source_type=str(p.get("source_type", "random")),  # type: ignore[arg-type]
                            thumbnail_url=p.get("thumbnail_url"),
                        )
                        self.articles[pageid] = article

                    article.times_shown += 1

                if self.analytics_store is not None:
                    delta_vs_batch_avg = None
                    try:
                        if batch_avg_pred is not None:
                            pvf = float(p.get("predicted_mean_reward", 0.0))
                            if math.isfinite(pvf) and math.isfinite(float(batch_avg_pred)):
                                delta_vs_batch_avg = float(pvf - float(batch_avg_pred))
                    except Exception:  # noqa: BLE001
                        delta_vs_batch_avg = None

                    # Best-effort logging for later clustering/analysis.
                    self.analytics_store.log_display(
                        {
                            "created_at": served_at,
                            "step": step_now,
                            "pageid": pageid,
                            "selection_type": p.get("selection_type"),
                            "source_type": p.get("source_type"),
                            "score": p.get("score"),
                            "predicted_mean_reward": p.get("predicted_mean_reward"),
                            "batch_predicted_mean_reward_avg": batch_avg_pred,
                            "delta_vs_batch_avg": delta_vs_batch_avg,
                            "title": p.get("title"),
                            "extract": p.get("extract"),
                            "url": p.get("url"),
                            "thumbnail_url": p.get("thumbnail_url"),
                            "embedding": p.get("embedding"),
                            "snapshot_loaded": p.get("snapshot_loaded"),
                        }
                    )

                cards.append(
                    ArticleCard(
                        pageid=pageid,
                        title=str(p.get("title", "")),
                        extract=str(p.get("extract", "")),
                        url=str(p.get("url", "")),
                        thumbnail_url=p.get("thumbnail_url"),
                        source_type=p.get("source_type", "random"),
                        selection_type=p.get("selection_type", "explore"),
                        score=float(p.get("score", 0.0)),
                        predicted_mean_reward=float(p.get("predicted_mean_reward", 0.0)),
                    )
                )
            except Exception:  # noqa: BLE001
                other_errors += 1
                continue

        # This is the "200 OK but 0 items" symptom in the UI. Make it loud in logs.
        # Most common causes: EMBEDDING_DIM mismatch between API and worker, or stale queue rows.
        if len(cards) == 0:
            try:
                example_len = None
                for p in payloads:
                    emb = p.get("embedding", None)
                    if emb is not None:
                        example_len = len(emb) if hasattr(emb, "__len__") else None
                        break
                print(
                    "[WARN] Dequeued payloads but produced 0 cards. "
                    f"expected_embedding_dim={self.embedding_dim} "
                    f"bad_embedding_dim={bad_embedding_dim} missing_embedding={missing_embedding} "
                    f"other_errors={other_errors} example_embedding_len={example_len}"
                )
            except Exception:  # noqa: BLE001
                pass

        return cards

    # ---------- Public API ----------

    def get_recommendations(self, batch_size: int = 5) -> List[ArticleCard]:
        batch_size = int(batch_size)
        if batch_size <= 0:
            return []

        cards = self._dequeue_from_persistent_queue(batch_size)
        with self._lock:
            self.batches_served += 1
        self._maybe_checkpoint(force=False)
        return cards

    def log_reward(self, reward_in: RewardIn) -> None:
        pid = reward_in.pageid
        with self._lock:
            article = self.articles.get(pid)
            if article is None:
                return

            reward = float(reward_in.reward)
            self.bandit.update(article.embedding, reward)

            article.total_reward += reward
            article.times_rewarded += 1

            step = self.bandit.interactions
            self.reward_events.append(
                RewardEvent(
                    step=step,
                    pageid=pid,
                    reward=reward,
                    selection_type=reward_in.selection_type,
                )
            )

            self._update_keywords(article, reward)
            if len(self.keyword_stats) > self.max_keywords:
                self._prune_keywords()
            self._dirty = True
            self.rewards_since_last_rerank += 1

            if self.analytics_store is not None:
                self.analytics_store.log_reward(
                    {
                        "created_at": utc_now_iso(),
                        "step": int(step),
                        "pageid": int(pid),
                        "reward": float(reward),
                        "selection_type": reward_in.selection_type,
                        "event_type": reward_in.event_type,
                        "dwell_time_ms": reward_in.dwell_time_ms,
                        "source_type": article.source_type,
                        "title": article.title,
                        "extract": article.extract,
                        "url": article.url,
                        "thumbnail_url": article.thumbnail_url,
                        "embedding": article.embedding.astype(float).tolist(),
                        "snapshot_loaded": self._loaded_snapshot_name,
                    }
                )
        self._maybe_checkpoint(force=False)

    def get_state(self, max_points: int = 100) -> StateResponse:
        with self._lock:
            interactions = self.bandit.interactions
            avg_reward = self.bandit.avg_reward()

            # Prefer the bandit's internal reward history so charts survive restarts.
            full_history = list(self.bandit.reward_history)
            recent_rewards = full_history[-max_points:]
            start_step = max(1, interactions - len(recent_rewards) + 1)
            reward_history = [
                RewardPoint(step=start_step + idx, reward=float(r))
                for idx, r in enumerate(recent_rewards)
            ]

            # Hit rate (reward > 0 counts as a "hit")
            hits = sum(1 for r in full_history if r > 0)
            hit_rate = hits / len(full_history) if full_history else 0.0

            # Rolling average reward (last 20)
            window_size = 20
            last_n = full_history[-window_size:] if full_history else []
            rolling_avg_reward = sum(last_n) / len(last_n) if last_n else 0.0

            # Rolling hit rate over time (window=20, sampled)
            rolling_hit_rates: List[RewardPoint] = []
            sample_interval = max(1, len(full_history) // 80)  # ~80 points max
            for i in range(len(full_history)):
                if i % sample_interval != 0 and i != len(full_history) - 1:
                    continue
                window_start = max(0, i - window_size + 1)
                window_slice = full_history[window_start : i + 1]
                window_hits = sum(1 for r in window_slice if r > 0)
                rate = window_hits / len(window_slice) if window_slice else 0.0
                rolling_hit_rates.append(RewardPoint(step=i + 1, reward=rate))

            # Calibration buckets - for now empty, would need display event tracking
            calibration_buckets: List[CalibrationBucket] = []

            top_articles_raw = [
                (
                    article.total_reward / article.times_rewarded,
                    article,
                )
                for article in self.articles.values()
                if article.times_rewarded > 0
            ]
            top_articles_raw.sort(key=lambda pair: pair[0], reverse=True)
            top_articles = [
                TopArticle(
                    pageid=article.pageid,
                    title=article.title,
                    mean_reward=float(mean_reward),
                    total_reward=float(article.total_reward),
                    times_rewarded=article.times_rewarded,
                )
                for mean_reward, article in top_articles_raw[:10]
            ]

            exploration_count = self.exploration_count
            exploitation_count = self.exploitation_count
            rewards_since_rerank = self.rewards_since_last_rerank

        return StateResponse(
            interactions=interactions,
            avg_reward=float(avg_reward),
            reward_history=reward_history,
            exploration_count=exploration_count,
            exploitation_count=exploitation_count,
            top_articles=top_articles,
            hit_rate=float(hit_rate),
            rolling_avg_reward=float(rolling_avg_reward),
            rolling_hit_rates=rolling_hit_rates,
            calibration_buckets=calibration_buckets,
            rewards_since_last_rerank=rewards_since_rerank,
        )
