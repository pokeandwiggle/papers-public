"""How much of the CUPID net score is the outcome-independent common component?

net = sum_success traj - sum_failure traj. The unsigned total sum_all traj measures
how strongly a demo influences every rollout regardless of outcome. Reports
Spearman(net, total) and the correlation of the per-rollout-normalized
(balanced) score with net, per task.

Usage: uv run --directory baseline python experiments/cupid/score_components.py DIR...
"""

import sys

import numpy as np
from scipy import stats

for root in sys.argv[1:]:
    data = np.load(f"{root}/trajectory_influence.npz")
    traj, success = data["traj"], data["success"].astype(bool)
    net = traj[success].sum(0) - traj[~success].sum(0)
    total = traj.sum(0)
    balanced = traj[success].mean(0) - traj[~success].mean(0) if success.any() and (~success).any() else net
    keep = len(net) - len(net) // 2
    top = lambda s: set(np.argsort(s)[-keep:])
    print(f"{root.rsplit('_', 1)[-1]}: successes {success.sum()}/{len(success)}; "
          f"spearman(net, total) {stats.spearmanr(net, total).statistic:+.3f}; "
          f"spearman(net, mean-balanced) {stats.spearmanr(net, balanced).statistic:+.3f}; "
          f"kept overlap net vs top-total {len(top(net) & top(total))}/{keep}")
