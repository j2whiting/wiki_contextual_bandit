# Wiki Contextual Multi-armed bandit.

End-to-end demo of a **Thompson Sampling bandit** for personalized content recommendations using Wikipedia:

- Pull random Wikipedia articles via the public API.
- Embed first paragraphs with `lumees/lumees-matryoshka-embedding-v1` (64-d truncated matryoshka).
- Run a diagonal linear Thompson Sampling bandit over embeddings.
- Maintain a reward-weighted keyword inventory from successful articles.
- Use that keyword inventory to drive Wikipedia `list=search` queries for "similar" content.
- **Visualize model convergence** with hit rate over time, reward signals, and parameter drift charts.

## Features

- **In-app article reader**: View full Wikipedia articles in a scrollable modal without leaving the app
- **Model convergence visualization**: Rolling hit rate chart shows how the model learns your preferences
- **Expected reward indicators**: Each card shows the model's confidence it will interest you
- **Learning progress bar**: Visual indicator of when to rerank with the updated model
- **Persistent state**: Bandit model, keywords, and analytics survive restarts

## Getting Started

### Option 1: Docker (Recommended)

```bash
# Start (persists data between runs)
docker-compose up --build    # or: docker compose up --build

# Optional: tune exploration (0..1). Higher = more random picks within each batch.
EXPLORATION_FRACTION=0.15 docker compose up --build

# Fresh start - wipe all data first
./fresh-start.sh

# Stop containers
docker-compose down          # or: docker compose down

# Stop AND permanently wipe all data
docker-compose down -v       # or: docker compose down -v
```

Then open http://localhost:8888 in your browser.

### Option 2: Local Python

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

pip install -r backend/requirements.txt

# Terminal 1: Run the queue prefetch worker continuously
python -m backend.worker --target-size 200 --batch-size 10

# Optional: discovery-only mode (random articles only; disables keyword search + neighbor expansion)
python -m backend.worker --target-size 200 --batch-size 10 --discovery-only

# Terminal 2: Start the API server
uvicorn backend.main:app --reload --port 8888
```

Then open http://localhost:8888 in your browser.

## How It Works

### Feed Tab
- Cards show Wikipedia articles with **expected reward** (model confidence)
- Click "View article" to read in-app and send a positive reward
- Click "Skip" to send a zero reward
- The **progress bar** fills as you interact—rerank when full for best results

### Model Insights Tab
- **Hit rate over time**: Rolling 20-sample hit rate. Should trend upward as the model learns.
- **Raw reward signal**: Individual rewards per interaction (click = 0.4–1.0, skip = 0).
- **Model stability**: Parameter drift (||Δθ||) between snapshots. Decreasing = converging.
- **Top articles**: Articles with highest mean reward.

### API Endpoints

| Endpoint | Description |
|----------|-------------|
| `GET /api/articles` | Fetch recommendation cards from queue |
| `GET /api/regenerate` | Rerank queue with updated model |
| `POST /api/reward` | Send reward signal for an article |
| `GET /api/state` | Get model stats and convergence metrics |
| `GET /api/article/{pageid}` | Fetch full article HTML for in-app viewing |
| `GET /api/snapshots` | List model snapshots |

### Discovery-only mode

If you want the system to only use **discovery** (random Wikipedia articles), run the worker with `--discovery-only`.
This disables:

- keyword-driven search (`source_type="search"`)
- neighbor expansion (`source_type="neighbor"`)

Note: if you already have queued items from a previous run, you may want a fresh start (`./fresh-start.sh`) so the queue contains only discovery items.

### Architecture

```
┌─────────────────┐     ┌─────────────────┐
│   Frontend      │────▶│   FastAPI       │
│   (static)      │◀────│   backend       │
└─────────────────┘     └────────┬────────┘
                                 │
        ┌────────────────────────┼────────────────────────┐
        ▼                        ▼                        ▼
┌───────────────┐      ┌─────────────────┐      ┌─────────────────┐
│ Queue Store   │      │ Bandit Model    │      │ Analytics Store │
│ (SQLite)      │      │ (JSON state)    │      │ (SQLite)        │
└───────────────┘      └─────────────────┘      └─────────────────┘
        ▲
        │
┌───────────────┐
│ Worker        │◀──── Wikipedia API
│ (prefetch)    │
└───────────────┘
```

## Data Persistence

| File/Directory | Purpose | Env Override |
|----------------|---------|--------------|
| `backend/data/bandit_state.json` | Latest bandit + keyword state | `BANDIT_STATE_PATH` |
| `backend/data/snapshots/` | Dated model snapshots with diffs | `BANDIT_SNAPSHOT_DIR` |
| `backend/data/queue.sqlite3` | Pre-fetched recommendation queue | `BANDIT_QUEUE_DB_PATH` |
| `backend/data/analytics.sqlite3` | Display + reward event log | `BANDIT_ANALYTICS_DB_PATH` |

### Changing embedding dimension (e.g. 256 → 64)

Embedding dimension must match across **API + worker + persisted state**. If you change `EMBEDDING_DIM`, the safest reset is:

```bash
./fresh-start.sh
```

This wipes the persisted queue/state so the worker can repopulate with the new dimension.

## Tuning

- **`EXPLORATION_FRACTION`**: float in \([0,1]\). Controls how often the batch generator forces a random pick from the ranked list.
  Thompson Sampling already explores natively; this knob adds extra diversity within each batch.

## Reward Signal

Rewards are computed based on user interaction:

| Action | Dwell Time | Reward |
|--------|------------|--------|
| Click | > 8 seconds | 1.0 |
| Click | 3–8 seconds | 0.7 |
| Click | < 3 seconds | 0.4 |
| Skip | — | 0.0 |

The bandit updates its posterior after each reward, learning which embedding dimensions correlate with your interests.
