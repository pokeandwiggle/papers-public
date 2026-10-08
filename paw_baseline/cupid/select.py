"""Filter-k demonstration subsets: CUPID ranking and the random control.

Both arms remove ``int(n * filter_ratio)`` demonstrations from the same pool,
as the reference ``filter_training_episodes`` does, so they train on equally
many demonstrations.
"""

import numpy as np


def num_filtered(total: int, filter_ratio: float) -> int:
    """Demonstrations removed by filter-k at ``filter_ratio``."""
    if not 0.0 <= filter_ratio < 1.0:
        raise ValueError("filter_ratio must be in [0, 1)")
    return int(total * filter_ratio)


def random_keep(source_ids: list[int], filter_ratio: float, seed: int) -> list[int]:
    """Uniformly random retained subset (control arm), sorted by source ID."""
    ids = sorted(set(source_ids))
    if len(ids) != len(source_ids):
        raise ValueError("source IDs must be unique")
    keep = len(ids) - num_filtered(len(ids), filter_ratio)
    rng = np.random.default_rng(seed)
    chosen = rng.choice(np.asarray(ids, dtype=np.int64), size=keep, replace=False)
    return sorted(int(x) for x in chosen)


def cupid_keep(scores: dict[int, float], filter_ratio: float) -> list[int]:
    """Remove the lowest performance-influence demonstrations (paper Eq. 4).

    Filter-k keeps the top-(n-k) of Psi_pi-inf, i.e. drops the k demonstrations
    whose upweighting is predicted to lower expected return the most. Ties are
    broken by source ID for determinism.
    """
    ids = sorted(scores)
    if not all(np.isfinite(scores[i]) for i in ids):
        raise ValueError("scores must be finite")
    drop = num_filtered(len(ids), filter_ratio)
    ranked = sorted(ids, key=lambda i: (scores[i], i))
    return sorted(ranked[drop:])
