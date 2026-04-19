#!/usr/bin/env python3
"""
Pre-compute and cache reference embeddings for all metrics.
Run this once after reference images are placed on the server.

Usage:
    docker exec leaderboard python3 /app/precache_reference.py --competition_id 1
"""
import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, "/app")
import database as db
from metrics.registry import create_default_registry

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("precache")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--competition_id", type=int, required=True)
    args = parser.parse_args()

    db.init_db()
    ref_dataset = db.get_reference_dataset(args.competition_id)
    if not ref_dataset:
        logger.error(f"No reference dataset for competition {args.competition_id}")
        sys.exit(1)

    ref_dir = Path(ref_dataset["file_path"])
    if not ref_dir.exists():
        logger.error(f"Reference dir not found: {ref_dir}")
        sys.exit(1)

    cache_dir = ref_dir.parent / "cache"
    cache_dir.mkdir(exist_ok=True)
    logger.info(f"Reference: {ref_dir} | Cache: {cache_dir}")

    registry = create_default_registry()
    for metric in registry._backends.values():
        if not hasattr(metric, "cache_reference_features"):
            continue
        logger.info(f"Caching {metric.display_name}...")
        try:
            metric.cache_reference_features(ref_dir, cache_dir)
            logger.info(f"{metric.display_name} cache done.")
        except Exception as e:
            logger.error(f"{metric.display_name} cache FAILED: {e}")

    db.update_cached_features(ref_dataset["id"], str(cache_dir))
    logger.info(f"DB updated: cached_features = {cache_dir}")
    logger.info("All done!")


if __name__ == "__main__":
    main()

