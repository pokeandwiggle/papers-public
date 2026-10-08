"""Paired SR/AP comparison of CUPID vs random filtering across tasks.

Usage:
    uv run --directory baseline python experiments/cupid/analyze.py experiments/cupid/evals.json

``evals.json`` maps each task slug to its eval handles, e.g.
``{"pour_screws_into_funnel": {"base": "eval:1", "random": "eval:2", "cupid": "eval:3"}}``.

Per task, episodes counted in the selected SR are paired by placement
combination. SR uses excellent/ok as success; AP counts non-failed subtask
segments capped at each subtask's required count (task show), over the total
required count. Overall effects average the per-task paired mean differences
with equal weight. The 95% CI uses per-task paired variances combined with a
Welch-Satterthwaite t; the p-value is a stratified sign-flip randomization test.
"""

import json
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats

SUCCESS = ("excellent", "ok")


def cli(*args) -> dict:
    out = subprocess.run(["paw-ais", *args], capture_output=True, text=True, check=True)
    return json.loads(out.stdout)


def required_counts(task: str) -> dict[str, int]:
    subtasks = cli("task", "show", task)["data"]["tasks"][0]["subtasks"]
    counts: dict[str, int] = defaultdict(int)
    for subtask in subtasks:
        counts[subtask["slug"]] += subtask["count"] if subtask["count"] is not None else 1
    return dict(counts)


def eval_episodes(handle: str) -> tuple[dict, list[dict]]:
    episodes, offset, report = [], 0, None
    while offset is not None:
        report = cli("operation", "status", handle, "--detailed", "--limit", "100", "--offset", str(offset))
        op = report["data"]["operations"][0]
        episodes.extend(op.get("episodes") or [])
        offset = op.get("next_offset")
    return op, episodes


def progress(episode: dict, required: dict[str, int]) -> float:
    done: dict[str, int] = defaultdict(int)
    for segment in episode.get("segments") or []:
        if segment.get("kind") != "subtask" or segment.get("value") not in required:
            continue
        if (segment.get("props") or {}).get("rating") != "failed":
            done[segment["value"]] += 1
    total = sum(required.values())
    return sum(min(done[slug], count) for slug, count in required.items()) / total


def arm_outcomes(handle: str, required: dict[str, int]) -> dict:
    op, episodes = eval_episodes(handle)
    per_placement: dict[int, list[tuple[float, float]]] = defaultdict(list)
    for entry in episodes:
        episode = entry["episode"]
        if not entry.get("counted_in_selected_sr"):
            continue
        success = 1.0 if episode.get("rating") in SUCCESS else 0.0
        per_placement[episode["placement_combination_id"]].append(
            (success, progress(episode, required))
        )
    split = op.get("selected_assignment_rating_split") or {}
    sr_values = [s for values in per_placement.values() for s, _ in values]
    ap_values = [p for values in per_placement.values() for _, p in values]
    return {
        "handle": handle,
        "link": (cli("operation", "status", handle).get("links") or {}).get("poker"),
        "state": op.get("state"),
        "successes": int(split.get("excellent_n", 0) + split.get("ok_n", 0)),
        "rated": int(split.get("rated_n", 0)),
        "poker_ap_pct": op.get("eval_average_progress_pct"),
        "our_ap_pct": 100 * float(np.mean(ap_values)) if ap_values else None,
        "our_sr": float(np.mean(sr_values)) if sr_values else None,
        "duplicates": sorted(k for k, v in per_placement.items() if len(v) > 1),
        "placements": {k: tuple(np.mean(v, axis=0)) for k, v in per_placement.items()},
    }


def paired(a: dict, b: dict, index: int) -> np.ndarray:
    keys = sorted(set(a["placements"]) & set(b["placements"]))
    return np.asarray([b["placements"][k][index] - a["placements"][k][index] for k in keys])


def combine(diffs: list[np.ndarray], rng: np.random.Generator, draws: int = 200_000) -> dict:
    """Equal-weight mean of per-task paired mean differences with CI and p-value."""
    k = len(diffs)
    means = np.asarray([d.mean() for d in diffs])
    estimate = means.mean()
    weights = np.asarray([d.var(ddof=1) / len(d) / k**2 for d in diffs])
    se = float(np.sqrt(weights.sum()))
    dfs = np.asarray([len(d) - 1 for d in diffs])
    df = weights.sum() ** 2 / np.sum(weights**2 / dfs) if se > 0 else np.inf
    half = stats.t.ppf(0.975, df) * se if se > 0 else 0.0
    flipped = np.zeros(draws)
    for d in diffs:
        signs = rng.choice((-1.0, 1.0), size=(draws, len(d)))
        flipped += (signs * d).mean(axis=1) / k
    p = float((np.sum(np.abs(flipped) >= abs(estimate) - 1e-12) + 1) / (draws + 1))
    # Sensitivity check: stratified paired bootstrap (resample placements per task).
    boot = np.zeros(20_000)
    for d in diffs:
        idx = rng.integers(0, len(d), size=(len(boot), len(d)))
        boot += d[idx].mean(axis=1) / k
    return {
        "estimate_pp": 100 * estimate,
        "ci95_pp": [100 * (estimate - half), 100 * (estimate + half)],
        "bootstrap_ci95_pp": [100 * float(np.percentile(boot, 2.5)), 100 * float(np.percentile(boot, 97.5))],
        "se_pp": 100 * se,
        "df": float(df),
        "p_sign_flip": p,
        "per_task_pp": [100 * m for m in means],
        "pairs": [len(d) for d in diffs],
    }


def main() -> None:
    config = json.loads(Path(sys.argv[1]).read_text())
    rng = np.random.default_rng(0)
    rows, sr_diffs, ap_diffs = {}, [], []
    for task, handles in config.items():
        required = required_counts(task)
        arms = {arm: arm_outcomes(handle, required) for arm, handle in handles.items()}
        rows[task] = {arm: {k: v for k, v in res.items() if k != "placements"} for arm, res in arms.items()}
        if "random" in arms and "cupid" in arms:
            sr = paired(arms["random"], arms["cupid"], 0)
            ap = paired(arms["random"], arms["cupid"], 1)
            missing = set(arms["random"]["placements"]) ^ set(arms["cupid"]["placements"])
            rows[task]["paired"] = {
                "n": len(sr),
                "unpaired_placements": sorted(missing),
                "sr_diff_pp": 100 * sr.mean(),
                "ap_diff_pp": 100 * ap.mean(),
                "discordant_cupid_only": int(np.sum(sr > 0)),
                "discordant_random_only": int(np.sum(sr < 0)),
                "mcnemar_exact_p": float(
                    stats.binomtest(int(np.sum(sr > 0)), int(np.sum(sr != 0)), 0.5).pvalue
                )
                if np.any(sr != 0)
                else 1.0,
            }
            sr_diffs.append(sr)
            ap_diffs.append(ap)
    result = {"tasks": rows}
    if sr_diffs:
        result["overall_sr"] = combine(sr_diffs, rng)
        result["overall_ap"] = combine(ap_diffs, rng)
    print(json.dumps(result, indent=1, default=float))


if __name__ == "__main__":
    main()
