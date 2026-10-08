"""Compare CUPID and random filter-k selections and eval failure stages.

Usage: uv run --directory baseline python experiments/cupid/compare_selections.py
"""

import json
import subprocess
from collections import Counter
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
TASKS = {
    "pour_screws_into_funnel": "1435",
    "place_loaded_boxes_in_container_bimanual": "1436",
    "clamp_electrical_component": "1437",
}
evals = json.loads((HERE / "evals.json").read_text())
rng = np.random.default_rng(0)


def entropy(counts):
    p = np.asarray(list(counts.values()), dtype=float)
    p /= p.sum()
    return float(-(p * np.log(p)).sum())


def first_failed_stage(handle):
    out = subprocess.run(["paw-ais", "operation", "status", handle, "--detailed", "--limit", "100"],
                         capture_output=True, text=True, check=True).stdout
    stages = Counter()
    for entry in json.loads(out)["data"]["operations"][0]["episodes"]:
        if not entry.get("counted_in_selected_sr"):
            continue
        segments = [s for s in entry["episode"]["segments"] if s["kind"] == "subtask"]
        segments.sort(key=lambda s: s["start_time"])
        failed = next((s["value"] for s in segments if (s.get("props") or {}).get("rating") == "failed"), None)
        if entry["episode"]["rating"] in ("excellent", "ok"):
            stages["success"] += 1
        else:
            stages[f"failed:{failed}" if failed else f"failed:after_{segments[-1]['value'] if segments else 'none'}"] += 1
    return dict(stages)


for task, ds in TASKS.items():
    members = {m["id"]: m for m in json.loads((HERE / f"members_{ds}.json").read_text())["members"]}
    cupid = set(json.loads((HERE / f"selection_cupid_{task}.json").read_text())["train_episode_ids"])
    rand = set(json.loads((HERE / f"selection_random_{task}.json").read_text())["train_episode_ids"])
    k, n = len(cupid), len(members)
    chance = [len(set(rng.choice(list(members), k, replace=False)) & set(rng.choice(list(members), k, replace=False))) for _ in range(2000)]
    print(f"== {task}: keep {k}/{n}, overlap cupid∩random {len(cupid & rand)} (chance {np.mean(chance):.0f})")
    for name, sel in (("all", set(members)), ("random", rand), ("cupid", cupid)):
        placements = Counter(members[i]["placement_combination_id"] for i in sel)
        durations = [members[i]["duration_s"] for i in sel]
        print(f"  {name:6s} distinct placements {len(placements):3d}  placement entropy {entropy(placements):.3f}  "
              f"mean duration {np.mean(durations):.1f}s  ok {sum(members[i]['rating'] == 'ok' for i in sel)}")
    for arm in ("base", "random", "cupid"):
        print(f"  eval {arm:6s} {evals[task][arm]}: {first_failed_stage(evals[task][arm])}")
