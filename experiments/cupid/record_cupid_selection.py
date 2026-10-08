"""Copy a CUPID scoring result into a committed selection record.

Usage: uv run --directory baseline python experiments/cupid/record_cupid_selection.py \
    TASK DATASET_HANDLE BASE_MODEL BASE_EVAL SCORE_DIR
"""

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
task, dataset, base_model, base_eval, score_dir = sys.argv[1:6]
scores = json.loads((Path(score_dir) / "cupid_scores.json").read_text())
members = json.loads((HERE / f"members_{dataset.split(':')[1]}.json").read_text())["members"]
ratings = {m["id"]: m["rating"] for m in members}
keep = scores["train_episode_ids"]
if scores.get("scoring_version") != 2:
    raise SystemExit("refusing to record scores that did not use (X^T X)^-1 (scoring_version 2)")
record = {
    "task": task,
    "dataset": dataset,
    "method": "cupid",
    "scoring_version": scores["scoring_version"],
    "base_model": base_model,
    "base_eval": base_eval,
    "filter_ratio": scores["filter_ratio"],
    "pool_size": len(scores["scores"]),
    "keep_count": len(keep),
    "keep_rating_counts": {r: sum(ratings[i] == r for i in keep) for r in sorted(set(ratings.values()))},
    "rollouts": scores["rollouts"],
    "rollout_successes": scores["rollout_successes"],
    "rollout_pairs": scores["rollout_pairs"],
    "featurization": {k: scores["train"][k] for k in (
        "checkpoint_sha256", "stride", "num_draws", "proj_dim", "seed", "train_noise_seed",
        "num_pairs", "num_params", "jpeg_quality", "frozen_arm", "matmul_precision")},
    "scores": scores["scores"],
    "train_episode_ids": keep,
}
(HERE / f"selection_cupid_{task}.json").write_text(json.dumps(record, indent=1) + "\n")
print(task, record["keep_count"], record["keep_rating_counts"])
