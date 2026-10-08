"""Write the random filter-k control selection for each task dataset.

Usage: uv run --directory baseline python experiments/cupid/make_random_selection.py
"""

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))

from paw_baseline.cupid.select import random_keep  # noqa: E402

FILTER_RATIO = 0.5
SEED = 0
DATASETS = {
    "pour_screws_into_funnel": "dataset:1435",
    "place_loaded_boxes_in_container_bimanual": "dataset:1436",
    "clamp_electrical_component": "dataset:1437",
}

for task, handle in DATASETS.items():
    members = json.loads((HERE / f"members_{handle.split(':')[1]}.json").read_text())["members"]
    ids = [m["id"] for m in members]
    keep = random_keep(ids, FILTER_RATIO, SEED)
    ratings = {m["id"]: m["rating"] for m in members}
    record = {
        "task": task,
        "dataset": handle,
        "method": "random",
        "filter_ratio": FILTER_RATIO,
        "seed": SEED,
        "pool_size": len(ids),
        "keep_count": len(keep),
        "keep_rating_counts": {
            r: sum(ratings[i] == r for i in keep) for r in sorted(set(ratings.values()))
        },
        "pool_rating_counts": {
            r: sum(v == r for v in ratings.values()) for r in sorted(set(ratings.values()))
        },
        "train_episode_ids": keep,
    }
    (HERE / f"selection_random_{task}.json").write_text(json.dumps(record, indent=1) + "\n")
    print(task, record["pool_size"], record["keep_count"], record["keep_rating_counts"], record["pool_rating_counts"])
