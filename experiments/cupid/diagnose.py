"""Diagnostics for a CUPID scoring output directory.

Usage: uv run --directory baseline python experiments/cupid/diagnose.py OUT [MEMBERS_JSON]

Reports split-half reliability of the net scores (rollouts split in halves,
stratified by outcome), the separate success/failure components, the relation
to demonstration length, and, when member ratings are given, how the kept set
differs by rating.
"""

import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats

out = Path(sys.argv[1])
traj_file = np.load(out / "trajectory_influence.npz")
train = np.load(out / "train_features.npz")
scores_json = json.loads((out / "cupid_scores.json").read_text())
traj, success = traj_file["traj"], traj_file["success"].astype(bool)
ids = [int(x) for x in traj_file["source_ids"]]
signs = np.where(success, 1.0, -1.0)
net = signs @ traj
counts = train["counts"]
print(f"rollouts {len(success)} (success {success.sum()}), demos {len(ids)}")
print(f"spearman(net, pairs per demo) = {stats.spearmanr(net, counts).statistic:.3f}")
succ_part = traj[success].sum(0) if success.any() else np.zeros(len(ids))
fail_part = -traj[~success].sum(0) if (~success).any() else np.zeros(len(ids))
print(f"spearman(success part, failure part) = {stats.spearmanr(succ_part, fail_part).statistic:.3f}")
rng = np.random.default_rng(0)
rhos, overlaps = [], []
keep = len(ids) - int(len(ids) * scores_json["filter_ratio"])
for _ in range(200):
    half = np.zeros(len(success), dtype=bool)
    for group in (np.flatnonzero(success), np.flatnonzero(~success)):
        chosen = rng.permutation(group)[: len(group) // 2]
        half[chosen] = True
    a, b = signs[half] @ traj[half], signs[~half] @ traj[~half]
    rhos.append(stats.spearmanr(a, b).statistic)
    top_a, top_b = set(np.argsort(a)[-keep:]), set(np.argsort(b)[-keep:])
    overlaps.append(len(top_a & top_b) / keep)
print(f"split-half spearman: mean {np.mean(rhos):.3f} (5-95%: {np.percentile(rhos, 5):.3f}..{np.percentile(rhos, 95):.3f})")
print(f"split-half kept-set overlap: mean {np.mean(overlaps):.3f} (chance {keep / len(ids):.3f})")
if len(sys.argv) > 2:
    members = {m["id"]: m for m in json.loads(Path(sys.argv[2]).read_text())["members"]}
    kept = set(scores_json["train_episode_ids"])
    for rating in sorted({m["rating"] for m in members.values()}):
        group = [i for i in ids if members[i]["rating"] == rating]
        print(f"rating {rating}: n={len(group)} kept={sum(i in kept for i in group)} "
              f"mean score={np.mean([net[ids.index(i)] for i in group]):.4g}")
    ok = np.array([members[i]["rating"] == "ok" for i in ids])
    if ok.any() and (~ok).any():
        auc = stats.mannwhitneyu(net[~ok], net[ok]).statistic / (ok.sum() * (~ok).sum())
        print(f"AUC(score excellent > ok) = {auc:.3f}")
