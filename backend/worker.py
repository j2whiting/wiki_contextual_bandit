from __future__ import annotations

import argparse
import math
import os
import time
from pathlib import Path

from sentence_transformers import SentenceTransformer

try:
    from .bandit import LinearThompsonSamplingBandit
    from .state_manager import ArticleStore
    from .wiki_client import WikipediaClient
except ImportError:  # pragma: no cover
    from bandit import LinearThompsonSamplingBandit
    from state_manager import ArticleStore
    from wiki_client import WikipediaClient


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Background worker to prefill the recommendations queue.")
    p.add_argument("--target-size", type=int, default=int(os.getenv("PREFETCH_QUEUE_TARGET", "80")))
    p.add_argument("--batch-size", type=int, default=int(os.getenv("PREFETCH_BATCH_SIZE", "8")))
    p.add_argument("--sleep-s", type=float, default=float(os.getenv("PREFETCH_SLEEP_S", "0.5")))
    p.add_argument(
        "--warmup-only",
        action="store_true",
        help="Fill the queue up to target-size, then exit.",
    )
    p.add_argument(
        "--discovery-only",
        action="store_true",
        help='Only generate/enqueue source_type="random" (UI label: discovery).',
    )
    return p


def main() -> int:
    args = build_parser().parse_args()

    base_dir = Path(__file__).resolve().parent
    data_dir = base_dir / "data"

    snapshot_dir = Path(os.getenv("BANDIT_SNAPSHOT_DIR", str(data_dir / "snapshots")))
    state_path = Path(os.getenv("BANDIT_STATE_PATH", str(data_dir / "bandit_state.json")))
    queue_db_path = Path(os.getenv("BANDIT_QUEUE_DB_PATH", str(data_dir / "queue.sqlite3")))

    embedding_dim = int(os.getenv("EMBEDDING_DIM", "64"))
    checkpoint_every_batches = int(os.getenv("CHECKPOINT_EVERY_BATCHES", "10"))
    checkpoint_min_interval_s = float(os.getenv("CHECKPOINT_MIN_INTERVAL_S", "5"))
    max_persisted_reward_history = int(os.getenv("MAX_PERSISTED_REWARD_HISTORY", "2000"))
    try:
        exploration_fraction = float(os.getenv("EXPLORATION_FRACTION", "0.3"))
    except Exception:  # noqa: BLE001
        exploration_fraction = 0.3
    if not math.isfinite(exploration_fraction):
        exploration_fraction = 0.3
    exploration_fraction = max(0.0, min(1.0, float(exploration_fraction)))

    embed_model = SentenceTransformer("lumees/lumees-matryoshka-embedding-v1")
    wiki_client = WikipediaClient()
    bandit = LinearThompsonSamplingBandit(dim=embedding_dim, lambda_=1.0, nu=1.0)

    allowed_source_types = ["random"] if args.discovery_only else None

    store = ArticleStore(
        embed_model=embed_model,
        wiki_client=wiki_client,
        bandit=bandit,
        embedding_dim=embedding_dim,
        min_pool_size=40,
        exploration_fraction=exploration_fraction,
        state_path=state_path,
        snapshot_dir=snapshot_dir,
        queue_db_path=queue_db_path,
        checkpoint_every_batches=checkpoint_every_batches,
        checkpoint_min_interval_s=checkpoint_min_interval_s,
        max_persisted_reward_history=max_persisted_reward_history,
        allowed_source_types=allowed_source_types,
    )

    store.load_state()

    print(f"[worker] queue_db={queue_db_path}")
    print(f"[worker] target_size={args.target_size} batch_size={args.batch_size} warmup_only={args.warmup_only}")

    try:
        while True:
            # Pick up new snapshots written by the API process so ranking stays fresh.
            store.load_state()

            qsize = store.queue_size()
            if qsize < args.target_size:
                need = max(0, args.target_size - qsize)
                to_make = min(int(args.batch_size), need) if need else int(args.batch_size)
                inserted = store.generate_and_enqueue(to_make)
                qsize2 = store.queue_size()
                print(f"[worker] qsize={qsize} -> {qsize2} (+{inserted})")

                if args.warmup_only and qsize2 >= args.target_size:
                    print("[worker] warmup complete; exiting.")
                    return 0
            else:
                if args.warmup_only:
                    print("[worker] queue already warm; exiting.")
                    return 0

            time.sleep(float(args.sleep_s))
    except KeyboardInterrupt:
        print("[worker] stopped.")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())


