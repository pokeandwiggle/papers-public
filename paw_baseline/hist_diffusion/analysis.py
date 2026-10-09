"""Paired comparison of control and method evals across tasks (SR and AP).

Usage:
  python -m paw_baseline.hist_diffusion.analysis <study.json>

``study.json``: {"tasks": [{"task": slug, "task_show": path, "baseline": [paths],
"proposed": [paths]}]} where each path holds the JSON printed by
``paw-ais operation status eval:<id> --detailed`` (all pages) and ``task_show``
the JSON printed by ``paw-ais task show <slug>``.

Episodes pair by ``episode.placement_combination_id`` within a task; only
entries with ``counted_in_selected_sr`` count. Success is a rating of
excellent or ok. Episode progress counts non-failed subtask segments capped at
each subtask's required count over the total required count (ignored and
intervention episodes excluded; missing annotations count zero). Per-task mean
paired differences are averaged with equal task weights. Uncertainty: a
stratified paired bootstrap (placements resampled within tasks) gives 95%
percentile intervals, and a within-pair sign-flip randomization test gives
two-sided p-values; both respect the pairing and suit small samples.
"""

import argparse
import json
from pathlib import Path

import numpy as np

SUCCESS = {"excellent", "ok"}
BOOTSTRAP_DRAWS = 20000
PERMUTATION_DRAWS = 50000


def load_episodes(paths: list[str]) -> tuple[list[dict], dict]:
    """Episodes and report fields from one eval's detailed status pages."""
    episodes, report = [], {}
    for path in paths:
        operation = json.loads(Path(path).read_text())["data"]["operations"][0]
        episodes.extend(operation.get("episodes") or [])
        report = {
            key: operation.get(key)
            for key in (
                "handle",
                "state",
                "eval_success_rate",
                "eval_average_progress_pct",
                "selected_assignment_rating_split",
            )
        }
    return episodes, report


def required_subtasks(task_show_path: str) -> dict[str, int]:
    task = json.loads(Path(task_show_path).read_text())["data"]["tasks"][0]
    required: dict[str, int] = {}
    for subtask in task["subtasks"]:
        required[subtask["slug"]] = required.get(subtask["slug"], 0) + (
            subtask.get("count") or 1
        )
    return required


def episode_progress(episode: dict, required: dict[str, int]) -> float | None:
    if episode.get("ignored") or episode.get("type") == "Intervention":
        return None
    done = dict.fromkeys(required, 0)
    for segment in episode.get("segments") or []:
        if segment.get("kind") != "subtask" or segment.get("value") not in done:
            continue
        if (segment.get("props") or {}).get("rating") == "failed":
            continue
        done[segment["value"]] += 1
    return sum(min(done[s], n) for s, n in required.items()) / sum(required.values())


def counted(entries: list[dict]) -> tuple[dict, list]:
    """Counted episodes keyed by placement combination, plus duplicate keys."""
    by_key, duplicates = {}, []
    for entry in entries:
        if not entry.get("counted_in_selected_sr"):
            continue
        episode = entry["episode"]
        key = episode["placement_combination_id"]
        if key in by_key:
            duplicates.append(key)
        by_key[key] = episode
    return by_key, duplicates


def arm_summary(by_key: dict, required: dict[str, int]) -> dict:
    ratings = [e.get("rating") for e in by_key.values()]
    rated = [r for r in ratings if r is not None]
    progress = [episode_progress(e, required) for e in by_key.values()]
    progress = [p for p in progress if p is not None]
    return {
        "successes": sum(r in SUCCESS for r in rated),
        "rated": len(rated),
        "ap_pct": 100 * float(np.mean(progress)) if progress else None,
    }


def paired_differences(base: dict, prop: dict, required: dict[str, int]):
    keys = sorted(set(base) & set(prop))
    sr, ap = [], []
    for key in keys:
        b, p = base[key], prop[key]
        if b.get("rating") is not None and p.get("rating") is not None:
            sr.append(float(p["rating"] in SUCCESS) - float(b["rating"] in SUCCESS))
        bp, pp = episode_progress(b, required), episode_progress(p, required)
        if bp is not None and pp is not None:
            ap.append(pp - bp)
    return keys, np.asarray(sr), np.asarray(ap)


def equal_weight_mean(diffs: list[np.ndarray]) -> float:
    return float(np.mean([d.mean() for d in diffs]))


def bootstrap_ci(diffs: list[np.ndarray], seed: int = 0) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    stats = np.empty(BOOTSTRAP_DRAWS)
    for b in range(BOOTSTRAP_DRAWS):
        stats[b] = np.mean([d[rng.integers(0, len(d), len(d))].mean() for d in diffs])
    return float(np.quantile(stats, 0.025)), float(np.quantile(stats, 0.975))


def sign_flip_p(diffs: list[np.ndarray], seed: int = 1) -> float:
    rng = np.random.default_rng(seed)
    observed = abs(equal_weight_mean(diffs))
    hits = 0
    for _ in range(PERMUTATION_DRAWS):
        flipped = [d * rng.choice((-1.0, 1.0), len(d)) for d in diffs]
        hits += abs(equal_weight_mean(flipped)) >= observed - 1e-12
    return (hits + 1) / (PERMUTATION_DRAWS + 1)


def summarize(study: dict) -> dict:
    rows, sr_diffs, ap_diffs = [], [], []
    for task in study["tasks"]:
        required = required_subtasks(task["task_show"])
        base_entries, base_report = load_episodes(task["baseline"])
        prop_entries, prop_report = load_episodes(task["proposed"])
        base, base_dup = counted(base_entries)
        prop, prop_dup = counted(prop_entries)
        keys, sr, ap = paired_differences(base, prop, required)
        row = {
            "task": task["task"],
            "baseline": {**arm_summary(base, required), "report": base_report},
            "proposed": {**arm_summary(prop, required), "report": prop_report},
            "pairs": len(keys),
            "unpaired": {
                "baseline": sorted(set(base) - set(prop)),
                "proposed": sorted(set(prop) - set(base)),
            },
            "duplicates": {"baseline": base_dup, "proposed": prop_dup},
        }
        for name, diffs in (("sr", sr), ("ap", ap)):
            if len(diffs):
                row[f"{name}_diff_pp"] = 100 * float(diffs.mean())
                low, high = bootstrap_ci([diffs])
                row[f"{name}_ci95_pp"] = [100 * low, 100 * high]
                row[f"{name}_p"] = sign_flip_p([diffs])
        rows.append(row)
        if len(sr):
            sr_diffs.append(sr)
        if len(ap):
            ap_diffs.append(ap)
    overall = {}
    for name, diffs in (("sr", sr_diffs), ("ap", ap_diffs)):
        if diffs:
            low, high = bootstrap_ci(diffs)
            overall[name] = {
                "diff_pp": 100 * equal_weight_mean(diffs),
                "ci95_pp": [100 * low, 100 * high],
                "p_two_sided": sign_flip_p(diffs),
                "tasks": len(diffs),
            }
    return {"tasks": rows, "overall": overall}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("study", type=Path)
    args = parser.parse_args()
    print(json.dumps(summarize(json.loads(args.study.read_text())), indent=2))


if __name__ == "__main__":
    main()
