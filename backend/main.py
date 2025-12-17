from __future__ import annotations

import os
from pathlib import Path
import threading
from typing import Any
from html import escape

import numpy as np

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles

try:
    # Preferred when launching from repo root: `uvicorn backend.main:app`
    from .bandit import LinearThompsonSamplingBandit
    from .state_manager import ArticleStore, RewardIn, StateResponse
    from .wiki_client import WikipediaClient
except ImportError:  # pragma: no cover
    # Fallback for launching from within `backend/`: `uvicorn main:app`
    from bandit import LinearThompsonSamplingBandit
    from state_manager import ArticleStore, RewardIn, StateResponse
    from wiki_client import WikipediaClient

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
DATA_DIR = BASE_DIR / "data"

SNAPSHOT_DIR = Path(os.getenv("BANDIT_SNAPSHOT_DIR", str(DATA_DIR / "snapshots")))
STATE_PATH = Path(os.getenv("BANDIT_STATE_PATH", str(DATA_DIR / "bandit_state.json")))
QUEUE_DB_PATH = Path(os.getenv("BANDIT_QUEUE_DB_PATH", str(DATA_DIR / "queue.sqlite3")))
ANALYTICS_DB_PATH = Path(os.getenv("BANDIT_ANALYTICS_DB_PATH", str(DATA_DIR / "analytics.sqlite3")))
CHECKPOINT_EVERY_BATCHES = int(os.getenv("CHECKPOINT_EVERY_BATCHES", "10"))
CHECKPOINT_MIN_INTERVAL_S = float(os.getenv("CHECKPOINT_MIN_INTERVAL_S", "5"))
MAX_PERSISTED_REWARD_HISTORY = int(os.getenv("MAX_PERSISTED_REWARD_HISTORY", "2000"))

# Embedding dimension must match across API + worker + persisted state.
# (Worker reads this env var already; API should too.)
EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", "64"))

# Fraction of batch picks that are forced "random-from-ranked-list" exploration.
# Note: Thompson Sampling already explores natively; this is an extra diversity knob.
try:
    _EXPLORATION_FRACTION = float(os.getenv("EXPLORATION_FRACTION", "0.3"))
except Exception:  # noqa: BLE001
    _EXPLORATION_FRACTION = 0.3
if not np.isfinite(_EXPLORATION_FRACTION):
    _EXPLORATION_FRACTION = 0.3
EXPLORATION_FRACTION = float(np.clip(_EXPLORATION_FRACTION, 0.0, 1.0))

class LazySentenceTransformer:
    """Lazily load the SentenceTransformer model on first encode().

    This keeps the API process fast to start (and avoids blocking the web server
    while model weights download). The worker still loads the model eagerly.
    """

    def __init__(self, model_name: str) -> None:
        self.model_name = str(model_name)
        self._model: Any | None = None
        self._lock = threading.Lock()

    def _get_model(self) -> Any:
        if self._model is None:
            with self._lock:
                if self._model is None:
                    from sentence_transformers import SentenceTransformer

                    self._model = SentenceTransformer(self.model_name)
        return self._model

    def encode(self, *args: Any, **kwargs: Any) -> Any:
        return self._get_model().encode(*args, **kwargs)


# Lumees matryoshka embedding model (lazy-loaded in API server)
embedding_model = LazySentenceTransformer("lumees/lumees-matryoshka-embedding-v1")

wiki_client = WikipediaClient()
bandit = LinearThompsonSamplingBandit(dim=EMBEDDING_DIM, lambda_=1.0, nu=1.0)
store = ArticleStore(
    embed_model=embedding_model,
    wiki_client=wiki_client,
    bandit=bandit,
    embedding_dim=EMBEDDING_DIM,
    min_pool_size=40,
    exploration_fraction=EXPLORATION_FRACTION,
    state_path=STATE_PATH,
    checkpoint_every_batches=CHECKPOINT_EVERY_BATCHES,
    checkpoint_min_interval_s=CHECKPOINT_MIN_INTERVAL_S,
    snapshot_dir=SNAPSHOT_DIR,
    queue_db_path=QUEUE_DB_PATH,
    analytics_db_path=ANALYTICS_DB_PATH,
    max_persisted_reward_history=MAX_PERSISTED_REWARD_HISTORY,
)

app = FastAPI(title="Wikipedia Bandit Explorer (Keywords)")


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

@app.middleware("http")
async def no_cache_api_responses(request: Request, call_next):
    """Prevent browsers/proxies from caching API responses and frontend assets.

    This app serves a queue-draining endpoint; if anything caches it (or a stale
    frontend JS keeps replaying an old cached response), the UI can get stuck
    showing empty results even when the queue is full.
    """
    response = await call_next(request)
    path = request.url.path
    if (
        path.startswith("/api/")
        or path == "/recs"
        or path == "/r"
        or path.startswith("/static/")
        or path.startswith("/minimal")
        or path == "/"
    ):
        response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
        response.headers["Pragma"] = "no-cache"
        response.headers["Expires"] = "0"
    return response



@app.get("/", response_class=FileResponse)
def root() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")

@app.get("/minimal", response_class=HTMLResponse)
def minimal() -> HTMLResponse:
    """Minimal *no-JS* UI: poll queue size via meta-refresh + dequeue via POST form."""
    qs = store.queue_store.size() if store.queue_store is not None else 0
    html = f"""<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <title>Minimal Queue Test</title>
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <meta http-equiv="refresh" content="1" />
    <style>
      body {{
        font-family: system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial, sans-serif;
        margin: 0;
        padding: 24px;
        background: #0b1220;
        color: #e5e7eb;
      }}
      .card {{
        background: #111a2e;
        border: 1px solid #23304a;
        border-radius: 12px;
        padding: 16px;
        margin-top: 16px;
      }}
      button {{
        background: #2563eb;
        border: none;
        color: white;
        padding: 10px 14px;
        border-radius: 10px;
        cursor: pointer;
        font-weight: 600;
      }}
      code {{
        background: #0b1220;
        border: 1px solid #23304a;
        border-radius: 8px;
        padding: 2px 6px;
      }}
      .mono {{
        font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace;
      }}
      a {{ color: #93c5fd; }}
    </style>
  </head>
  <body>
    <h1>Minimal Queue Test (No JS)</h1>
    <p>
      Queue size auto-refreshes every second (GET <code>/api/queue_size</code> server-side).
      Button posts to <code>/minimal/fetch</code> to dequeue 5 and render titles.
    </p>
    <div class="card">
      <div><strong>Queue size:</strong> <span class="mono">{int(qs)}</span></div>
      <form method="post" action="/minimal/fetch" style="margin-top: 12px;">
        <button type="submit">Fetch top 5</button>
      </form>
      <p style="margin-top: 12px;">
        Prefer JS version? <a href="/minimal-js">/minimal-js</a>
      </p>
    </div>
  </body>
</html>"""
    return HTMLResponse(content=html)


@app.get("/minimal-js", response_class=FileResponse)
def minimal_js() -> FileResponse:
    """Original JS-based minimal view (kept for debugging)."""
    return FileResponse(STATIC_DIR / "minimal.html")


@app.post("/minimal/fetch", response_class=HTMLResponse)
def minimal_fetch() -> HTMLResponse:
    qs_before = store.queue_store.size() if store.queue_store is not None else 0
    cards = store.get_recommendations(batch_size=5)
    qs_after = store.queue_store.size() if store.queue_store is not None else 0

    lis = []
    for c in cards:
        title = escape(str(c.title))
        url = escape(str(c.url))
        lis.append(f'<li><a href="{url}" target="_blank" rel="noreferrer">{title}</a></li>')
    items_html = "\n".join(lis) if lis else "<li>(no items)</li>"

    html = f"""<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <title>Minimal Queue Test - Result</title>
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <style>
      body {{
        font-family: system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial, sans-serif;
        margin: 0;
        padding: 24px;
        background: #0b1220;
        color: #e5e7eb;
      }}
      .card {{
        background: #111a2e;
        border: 1px solid #23304a;
        border-radius: 12px;
        padding: 16px;
        margin-top: 16px;
      }}
      button {{
        background: #2563eb;
        border: none;
        color: white;
        padding: 10px 14px;
        border-radius: 10px;
        cursor: pointer;
        font-weight: 600;
      }}
      code {{
        background: #0b1220;
        border: 1px solid #23304a;
        border-radius: 8px;
        padding: 2px 6px;
      }}
      .mono {{
        font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace;
      }}
      a {{ color: #93c5fd; }}
    </style>
  </head>
  <body>
    <h1>Minimal Queue Test - Result</h1>
    <div class="card">
      <div><strong>Queue size before:</strong> <span class="mono">{int(qs_before)}</span></div>
      <div><strong>Dequeued cards:</strong> <span class="mono">{len(cards)}</span></div>
      <div><strong>Queue size after:</strong> <span class="mono">{int(qs_after)}</span></div>
      <ol style="margin-top: 12px;">
        {items_html}
      </ol>
      <form method="get" action="/minimal" style="margin-top: 12px;">
        <button type="submit">Back</button>
      </form>
    </div>
  </body>
</html>"""
    return HTMLResponse(content=html)

@app.get("/api/queue_size")
def get_queue_size():
    qs = store.queue_store.size() if store.queue_store is not None else 0
    return {"size": int(qs)}

def _kmeans_labels(x: np.ndarray, k: int, *, iters: int = 25, seed: int = 0) -> np.ndarray:
    """Tiny k-means for small in-memory analysis (no sklearn dependency)."""
    x = np.asarray(x, dtype=np.float64)
    n = int(x.shape[0])
    if n <= 0:
        return np.zeros((0,), dtype=np.int64)

    k = max(1, min(int(k), n))
    rng = np.random.default_rng(int(seed))
    init_idx = rng.choice(n, size=k, replace=False)
    centroids = x[init_idx].copy()

    labels = np.zeros((n,), dtype=np.int64)
    for i in range(int(iters)):
        # squared distances (n,k)
        d2 = ((x[:, None, :] - centroids[None, :, :]) ** 2).sum(axis=2)
        new_labels = d2.argmin(axis=1)
        if i > 0 and np.array_equal(new_labels, labels):
            break
        labels = new_labels

        for j in range(k):
            mask = labels == j
            if not np.any(mask):
                centroids[j] = x[rng.integers(n)]
            else:
                centroids[j] = x[mask].mean(axis=0)

    return labels

def _roc_auc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Compute ROC AUC from scores + binary labels (1=positive) without sklearn."""
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    labels = np.asarray(labels, dtype=np.int64).reshape(-1)
    if scores.size == 0 or labels.size != scores.size:
        return 0.0

    n_pos = int(np.sum(labels == 1))
    n_neg = int(np.sum(labels == 0))
    if n_pos == 0 or n_neg == 0:
        return 0.0

    order = np.argsort(scores, kind="mergesort")  # stable for tie handling
    sorted_scores = scores[order]
    sorted_labels = labels[order]

    # Assign average ranks for ties (1-indexed ranks)
    ranks = np.empty_like(sorted_scores, dtype=np.float64)
    i = 0
    r = 1
    n = int(sorted_scores.size)
    while i < n:
        j = i + 1
        while j < n and sorted_scores[j] == sorted_scores[i]:
            j += 1
        # tie group i..j-1 has ranks r..r+(j-i)-1
        avg_rank = (r + (r + (j - i) - 1)) / 2.0
        ranks[i:j] = avg_rank
        r += (j - i)
        i = j

    sum_ranks_pos = float(np.sum(ranks[sorted_labels == 1]))
    auc = (sum_ranks_pos - (n_pos * (n_pos + 1) / 2.0)) / (n_pos * n_neg)
    return float(max(0.0, min(1.0, auc)))


@app.get("/api/taste_map")
def get_taste_map(limit: int = 250, k: int = 6, min_reward: float = 0.0):
    """Cluster liked items and return a 2D PCA projection for visualization."""
    if store.analytics_store is None:
        return {"items": [], "n": 0, "k": 0, "detail": "analytics_store not configured"}

    limit = max(0, int(limit))
    limit = min(limit, 800)
    if limit <= 0:
        return {"items": [], "n": 0, "k": 0}

    liked = store.analytics_store.get_liked_embeddings(min_reward=float(min_reward), limit=limit)

    expected_dim = int(store.embedding_dim)
    meta = []
    vecs = []
    for r in liked:
        emb = r.get("embedding", None)
        if emb is None or not hasattr(emb, "__len__"):
            continue
        if len(emb) != expected_dim:
            continue
        try:
            v = np.asarray(emb, dtype=np.float64).reshape(-1)
        except Exception:  # noqa: BLE001
            continue
        if v.shape[0] != expected_dim:
            continue
        vecs.append(v)
        meta.append(
            {
                "pageid": int(r.get("pageid", 0)),
                "title": str(r.get("title", "")),
                "url": str(r.get("url", "")),
                "reward": float(r.get("reward", 0.0)),
            }
        )

    n = len(vecs)
    if n < 2:
        return {"items": [], "n": n, "k": 0, "embedding_dim": expected_dim}

    x = np.vstack(vecs).astype(np.float64)  # (n, d)

    # PCA to 2D for plotting.
    xc = x - x.mean(axis=0, keepdims=True)
    try:
        # SVD-based PCA
        _, _, vt = np.linalg.svd(xc, full_matrices=False)
        coords = xc @ vt.T[:, :2]
    except Exception:  # noqa: BLE001
        coords = xc[:, :2] if xc.shape[1] >= 2 else np.pad(xc, ((0, 0), (0, 2 - xc.shape[1])))

    coords = coords.astype(np.float64)
    # Normalize for nicer plots
    std = coords.std(axis=0)
    std[std == 0] = 1.0
    coords = coords / std

    # K-means on the original embedding space (the "cluster over embeddings" part).
    k = int(k)
    if k <= 0:
        k = max(2, min(8, int(np.sqrt(n))))
    k = max(1, min(k, n))
    labels = _kmeans_labels(x, k)

    items = []
    for i in range(n):
        items.append(
            {
                "pageid": int(meta[i]["pageid"]),
                "title": meta[i]["title"],
                "url": meta[i]["url"],
                "reward": float(meta[i]["reward"]),
                "x": float(coords[i, 0]),
                "y": float(coords[i, 1]),
                "cluster": int(labels[i]),
            }
        )

    return {
        "items": items,
        "n": int(n),
        "k": int(k),
        "embedding_dim": expected_dim,
    }


@app.get("/api/delta_map")
def get_delta_map(limit: int = 400):
    """Return a 2D PCA projection of recently served cards, labeled by Δ vs batch avg.

    Δ is computed as:
        delta_vs_batch_avg = predicted_mean_reward - batch_predicted_mean_reward_avg

    and is logged at serve-time in display_events.
    """
    if store.analytics_store is None:
        return {"items": [], "n": 0, "detail": "analytics_store not configured"}

    limit = max(0, int(limit))
    limit = min(limit, 1200)
    if limit <= 0:
        return {"items": [], "n": 0}

    samples = store.analytics_store.get_display_delta_embeddings(limit=limit)

    expected_dim = int(store.embedding_dim)
    meta = []
    vecs = []
    for r in samples:
        emb = r.get("embedding", None)
        if emb is None or not hasattr(emb, "__len__"):
            continue
        if len(emb) != expected_dim:
            continue
        try:
            v = np.asarray(emb, dtype=np.float64).reshape(-1)
        except Exception:  # noqa: BLE001
            continue
        if v.shape[0] != expected_dim:
            continue

        try:
            delta = float(r.get("delta_vs_batch_avg", 0.0))
        except Exception:  # noqa: BLE001
            continue

        meta.append(
            {
                "pageid": int(r.get("pageid", 0)),
                "title": str(r.get("title", "")),
                "url": str(r.get("url", "")),
                "predicted_mean_reward": float(r.get("predicted_mean_reward", 0.0)),
                "batch_predicted_mean_reward_avg": float(r.get("batch_predicted_mean_reward_avg", 0.0)),
                "delta_vs_batch_avg": float(delta),
            }
        )
        vecs.append(v)

    n = len(vecs)
    if n < 2:
        return {"items": [], "n": n, "embedding_dim": expected_dim}

    x = np.vstack(vecs).astype(np.float64)  # (n, d)

    # PCA to 2D for plotting.
    xc = x - x.mean(axis=0, keepdims=True)
    try:
        _, _, vt = np.linalg.svd(xc, full_matrices=False)
        coords = xc @ vt.T[:, :2]
    except Exception:  # noqa: BLE001
        coords = xc[:, :2] if xc.shape[1] >= 2 else np.pad(xc, ((0, 0), (0, 2 - xc.shape[1])))

    coords = coords.astype(np.float64)
    std = coords.std(axis=0)
    std[std == 0] = 1.0
    coords = coords / std

    items = []
    n_pos = 0
    n_nonpos = 0
    for i in range(n):
        delta = float(meta[i]["delta_vs_batch_avg"])
        group = "pos" if delta > 0 else "nonpos"
        if delta > 0:
            n_pos += 1
        else:
            n_nonpos += 1
        items.append(
            {
                **meta[i],
                "x": float(coords[i, 0]),
                "y": float(coords[i, 1]),
                "group": group,
            }
        )

    return {
        "items": items,
        "n": int(n),
        "n_pos": int(n_pos),
        "n_nonpos": int(n_nonpos),
        "embedding_dim": expected_dim,
    }


@app.get("/api/embedding_quality")
def get_embedding_quality(limit_per_class: int = 250, like_threshold: float = 0.0, include_scores: bool = True):
    """Compare predicted reward distributions for liked vs disliked items.

    Uses current bandit theta_mean as the scoring function: score = theta_mean · embedding.
    """
    if store.analytics_store is None:
        return {"liked": {}, "disliked": {}, "detail": "analytics_store not configured"}

    split = store.analytics_store.get_labeled_embeddings(
        like_threshold=float(like_threshold),
        limit_per_class=int(limit_per_class),
    )
    liked = split.get("liked", [])
    disliked = split.get("disliked", [])

    expected_dim = int(store.embedding_dim)
    theta = store.bandit.theta_mean().astype(np.float64).reshape(-1)

    def scores_for(samples):
        out = []
        meta = []
        for s in samples:
            emb = s.get("embedding", None)
            if emb is None or not hasattr(emb, "__len__") or len(emb) != expected_dim:
                continue
            try:
                v = np.asarray(emb, dtype=np.float64).reshape(-1)
            except Exception:  # noqa: BLE001
                continue
            if v.shape[0] != expected_dim or theta.shape[0] != expected_dim:
                continue
            out.append(float(np.dot(theta, v)))
            meta.append({"pageid": int(s.get("pageid", 0)), "title": str(s.get("title", ""))})
        return np.asarray(out, dtype=np.float64), meta

    s_like, meta_like = scores_for(liked)
    s_dis, meta_dis = scores_for(disliked)

    n_like = int(s_like.size)
    n_dis = int(s_dis.size)
    if n_like == 0 or n_dis == 0:
        return {
            "n_liked": n_like,
            "n_disliked": n_dis,
            "embedding_dim": expected_dim,
            "detail": "Need at least 1 liked and 1 disliked item with embeddings.",
        }

    mean_like = float(np.mean(s_like))
    mean_dis = float(np.mean(s_dis))
    gap = float(mean_like - mean_dis)

    std_like = float(np.std(s_like)) if n_like > 1 else 0.0
    std_dis = float(np.std(s_dis)) if n_dis > 1 else 0.0

    # Cohen's d
    if n_like > 1 and n_dis > 1:
        var_like = float(np.var(s_like, ddof=1))
        var_dis = float(np.var(s_dis, ddof=1))
        pooled = ((n_like - 1) * var_like + (n_dis - 1) * var_dis) / max(1, (n_like + n_dis - 2))
        d = gap / float(np.sqrt(pooled)) if pooled > 0 else 0.0
    else:
        d = 0.0

    # AUC using predicted scores
    scores = np.concatenate([s_like, s_dis], axis=0)
    labels = np.concatenate([np.ones(n_like, dtype=np.int64), np.zeros(n_dis, dtype=np.int64)], axis=0)
    auc = _roc_auc(scores, labels)

    payload = {
        "n_liked": n_like,
        "n_disliked": n_dis,
        "embedding_dim": expected_dim,
        "mean_pred_liked": mean_like,
        "mean_pred_disliked": mean_dis,
        "gap": gap,
        "std_pred_liked": std_like,
        "std_pred_disliked": std_dis,
        "auc": float(auc),
        "cohens_d": float(d),
        # include a few example titles for quick sanity checks
        "examples": {
            "liked": meta_like[:5],
            "disliked": meta_dis[:5],
        },
    }

    if include_scores:
        # Limit payload size; enough for a small histogram in the UI.
        payload["scores_liked"] = [float(x) for x in s_like[:300].tolist()]
        payload["scores_disliked"] = [float(x) for x in s_dis[:300].tolist()]

    return payload

@app.api_route("/api/articles", methods=["POST", "GET"])
def get_articles(batch_size: int = 5):
    cards = store.get_recommendations(batch_size=batch_size)
    return {"items": [c.model_dump() for c in cards]}


# Alias endpoint: some blockers/extensions target /api/* patterns.
@app.api_route("/recs", methods=["POST", "GET"])
def get_recs(batch_size: int = 5):
    cards = store.get_recommendations(batch_size=batch_size)
    return {"items": [c.model_dump() for c in cards]}


# Extra-short alias: some blockers match longer /api/* or "recs" paths.
@app.api_route("/r", methods=["POST", "GET"])
def get_r(batch_size: int = 5):
    cards = store.get_recommendations(batch_size=batch_size)
    return {"items": [c.model_dump() for c in cards]}


# Frontend uses GET + cache-buster + no-store here to work around some blockers
# that are more aggressive with POST requests.
@app.api_route("/api/regenerate", methods=["POST", "GET"])
def regenerate(batch_size: int = 5):
    # 1) force a model checkpoint/snapshot so the "latest" model is persisted
    store.save_state()
    # 2+3) rescore and rebatch everything already in the queue
    queue_size = store.rescore_and_rebatch_queue(batch_size=batch_size)
    # return a fresh batch to immediately refresh the feed
    cards = store.get_recommendations(batch_size=batch_size)
    return {
        "queue_size": queue_size,
        "items": [c.model_dump() for c in cards],
    }


@app.post("/api/reward")
def post_reward(reward_in: RewardIn):
    store.log_reward(reward_in)
    return {"status": "ok"}


@app.get("/api/state", response_model=StateResponse)
def get_state():
    return store.get_state()


@app.get("/api/snapshots")
def list_snapshots(limit: int = 50):
    items = store.list_snapshots(limit=limit)
    return {"items": [i.model_dump() for i in items]}


@app.get("/api/snapshots/{name}")
def get_snapshot(name: str):
    payload = store.get_snapshot_payload(name)
    if payload is None:
        raise HTTPException(status_code=404, detail="Snapshot not found")
    return payload


@app.get("/api/article/{pageid}")
def get_article_html(pageid: int):
    """Fetch full article HTML for in-app viewing."""
    result = wiki_client.get_page_html(pageid)
    if result is None:
        raise HTTPException(status_code=404, detail="Article not found")
    return result
