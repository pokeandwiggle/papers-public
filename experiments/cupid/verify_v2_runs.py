"""Verify that CUPID-v2 training runs (and, once finished, their models) used v2 data.

Usage: uv run --directory baseline python experiments/cupid/verify_v2_runs.py [CKPT_ROOT]

Compares each v2 run against its v1 (invalid) and random counterparts:
submitted params, code revision, dataset and machine shape from the training
report; and, when restored checkpoints are given under CKPT_ROOT/<model-id>,
the training manifest (source episode IDs) and normalization stats.
"""

import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
RUNS = {
    # task: (v2 train, v1 train, random train)
    "pour_screws_into_funnel": ("train:1108", "train:1061", "train:924"),
    "place_loaded_boxes_in_container_bimanual": ("train:1109", "train:1060", "train:925"),
    "clamp_electrical_component": ("train:1110", "train:1059", "train:926"),
}


def report(handle: str) -> dict:
    out = subprocess.run(["paw-ais", "operation", "status", handle], capture_output=True, text=True, check=True)
    return json.loads(out.stdout)["data"]["operations"][0]


def checkpoint(ckpt_root: Path, model: str | None) -> Path | None:
    if not model:
        return None
    paths = sorted((ckpt_root / model.split(":")[1]).glob("*/checkpoints/*/pretrained_model"))
    return paths[-1] if paths else None


def manifest_ids(ckpt_root: Path, model: str | None) -> list[int] | None:
    path = checkpoint(ckpt_root, model)
    if path is None:
        return None
    return [row["source_episode_id"] for row in json.loads((path / ".paw/training_data.json").read_text())["train_episodes"]]


def flat(d: dict, prefix: str = "") -> dict:
    out = {}
    for key, value in d.items():
        if isinstance(value, dict):
            out.update(flat(value, f"{prefix}{key}."))
        else:
            out[f"{prefix}{key}"] = value
    return out


def digest(path: Path) -> str:
    import hashlib

    return hashlib.sha256(path.read_bytes()).hexdigest()


def action_stats(path: Path):
    from safetensors.numpy import load_file

    files = sorted(path.glob("policy_preprocessor_step_*_normalizer_processor.safetensors"))
    tensors = load_file(str(files[0]))
    return {k: v for k, v in tensors.items() if k.startswith("action.")}


ckpt_root = Path(sys.argv[1]) if len(sys.argv) > 1 else None
ok = True
for task, (v2, v1, rnd) in RUNS.items():
    sel_v2 = json.loads((HERE / f"selection_cupid_{task}.json").read_text())["train_episode_ids"]
    sel_v1 = json.loads((HERE / f"selection_cupid_{task}_v1_invalid.json").read_text())["train_episode_ids"]
    r2, r1, rr = report(v2), report(v1), report(rnd)
    ids2 = r2["params"]["data"]["train_episode_ids"]
    shared = lambda r: ({k: v for k, v in r["params"].items() if k != "data"}, r["dataset"], r.get("world_size"))
    checks = {
        "v2 params IDs == v2 selection": ids2 == sel_v2,
        "v2 params IDs != v1 selection": ids2 != sel_v1,
        "v1 params IDs == v1 selection": r1["params"]["data"]["train_episode_ids"] == sel_v1,
        "same preset/overrides/dataset/world size as v1 and random": shared(r2) == shared(r1) == shared(rr),
        "same kept count as random": len(ids2) == len(rr["params"]["data"]["train_episode_ids"]),
    }
    if ckpt_root is not None:
        m2 = manifest_ids(ckpt_root, r2.get("model"))
        m1 = manifest_ids(ckpt_root, r1.get("model"))
        if m2 is not None:
            checks["v2 checkpoint manifest == v2 selection"] = m2 == sel_v2
        if m1 is not None:
            checks["v1 checkpoint manifest == v1 selection"] = m1 == sel_v1
        c2, c1 = checkpoint(ckpt_root, r2.get("model")), checkpoint(ckpt_root, r1.get("model"))
        if c2 is not None and c1 is not None:
            import numpy as np

            t2, t1 = json.loads((c2 / ".paw/training_data.json").read_text()), json.loads((c1 / ".paw/training_data.json").read_text())
            checks["v2 normalization fit on v2 episodes only"] = (
                t2["normalization"]["episode_indices"] == [row["episode_index"] for row in t2["train_episodes"]]
            )
            checks["v2 normalization episodes != v1"] = t2["normalization"]["episode_indices"] != t1["normalization"]["episode_indices"]
            s2, s1 = action_stats(c2), action_stats(c1)
            checks["v2 action stats differ from v1 (refit on v2 subset)"] = any(
                not np.array_equal(s2[k], s1[k]) for k in s2
            )
            checks["v2 weights differ from v1"] = digest(c2 / "model.safetensors") != digest(c1 / "model.safetensors")
            # Resume bookkeeping (a preempted run restarts from its own checkpoint)
            # and the intended data selection may differ; nothing else may.
            resume_keys = {"pretrained_path", "checkpoint_path", "resume", "policy.pretrained_path"}
            data_keys = {"dataset.episodes", "dataset.repo_id", "dataset.root", "wandb.run_id"}
            cfg = lambda c, n: flat(json.loads((c / n).read_text()))
            policy_diff = {k for k in set(cfg(c2, "config.json")) | set(cfg(c1, "config.json"))
                           if cfg(c2, "config.json").get(k) != cfg(c1, "config.json").get(k)}
            train_diff = {k for k in set(cfg(c2, "train_config.json")) | set(cfg(c1, "train_config.json"))
                          if cfg(c2, "train_config.json").get(k) != cfg(c1, "train_config.json").get(k)}
            checks[f"v2 policy config == v1 except resume metadata {sorted(policy_diff)}"] = policy_diff <= resume_keys
            checks[f"train config differs only in data/resume keys {sorted(train_diff)}"] = train_diff <= resume_keys | data_keys
            checks["v2 run was not resumed"] = not json.loads((c2 / "train_config.json").read_text()).get("resume")
            checks["v2 JPEG recipe == v1"] = (c2 / ".paw/image_transforms.json").read_text() == (c1 / ".paw/image_transforms.json").read_text()
    overlap = len(set(sel_v2) & set(sel_v1))
    print(f"== {task}: v2 {v2} ({r2['state']}, code {r2.get('code_sha', '')[:7]}, model {r2.get('model')}) "
          f"vs v1 {v1} (code {r1.get('code_sha', '')[:7]}, model {r1.get('model')}); v1/v2 overlap {overlap}/{len(sel_v2)}")
    for name, passed in checks.items():
        ok &= bool(passed)
        print(f"   [{'ok' if passed else 'FAIL'}] {name}")
sys.exit(0 if ok else 1)
