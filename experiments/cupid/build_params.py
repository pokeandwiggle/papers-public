"""Build cloud training params for one filter-k arm from its committed selection.

Usage: uv run --directory baseline python experiments/cupid/build_params.py ARM TASK OUT_JSON
ARM is "cupid" or "random". The CUPID arm must come from scoring_version 2.
Both arms share every training setting except data.train_episode_ids.
"""

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SHARED = {"preset": "diffusion", "overrides": {"steps": 100000}}  # with --gpus 2

arm, task, out = sys.argv[1], sys.argv[2], Path(sys.argv[3])
if arm not in ("cupid", "random"):
    raise SystemExit("ARM must be cupid or random")
selection = json.loads((HERE / f"selection_{arm}_{task}.json").read_text())
if arm == "cupid" and selection.get("scoring_version") != 2:
    raise SystemExit("CUPID selection is not scoring_version 2")
ids = selection["train_episode_ids"]
members = json.loads((HERE / f"members_{selection['dataset'].split(':')[1]}.json").read_text())["members"]
if len(set(ids)) != len(ids) or not set(ids) <= {m["id"] for m in members}:
    raise SystemExit("selection IDs must be unique source IDs of the dataset")
params = {**SHARED, "data": {"train_episode_ids": ids}}
out.write_text(json.dumps(params, separators=(",", ":")) + "\n")
check = json.loads(out.read_text())
assert check["data"]["train_episode_ids"] == ids
print(arm, task, selection["dataset"], len(ids), "->", out)
