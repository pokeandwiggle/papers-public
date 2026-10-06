"""Materialize PI05 columns and fit normalization over the actual chunk targets."""

import hashlib
import json
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from paw_baseline.pi05.geometry import (
    CLOSED_FRACTION_DIMS,
    LAYOUT,
    ROTATION_CONVENTION,
    Pi05Representation,
    relative_action_mask,
)
from paw_baseline.prepare import (
    PROVENANCE_PATH,
    PreparedDataset,
    _fixed_list,
    _squash_arm_stats,
    _vector_stats,
)
from paw_baseline.representation import MODEL_ROTATION_DIMS, decode_model


def chunk_action_values(
    action, episodes, selected_episodes, horizon, *, state=None, relative=False
):
    """Fit uniform LeRobot chunk targets with last-frame padding at episode ends.

    PI05 includes padded targets in its loss, so they enter the distribution.
    Only selected training episodes contribute to normalization. With
    ``relative``, targets are the openpi DeltaActions view used in training:
    every encoded chunk action minus the encoded anchor state on the masked dims.
    """
    if type(horizon) is not int or horizon < 1:
        raise ValueError("chunk horizon must be positive")
    if relative and state is None:
        raise ValueError("relative chunk targets require the anchor states")
    count = np.isin(episodes, list(selected_episodes)).sum()
    values = np.empty((count * horizon, 20), dtype=np.float32)
    cursor = 0
    representation = Pi05Representation()
    mask = np.asarray(relative_action_mask())
    for episode in sorted(selected_episodes):
        rows = np.flatnonzero(episodes == episode)
        for first in range(0, len(rows), 1024):
            anchors = np.arange(first, min(first + 1024, len(rows)))
            targets = np.minimum(anchors[:, None] + np.arange(horizon), len(rows) - 1)
            encoded = representation.encode(action[rows[targets]])
            if relative:
                anchor_state = representation.encode(state[rows[anchors]])
                encoded[..., mask] -= anchor_state[:, None, mask]
            flat = encoded.reshape(-1, 20)
            values[cursor : cursor + len(flat)] = flat
            cursor += len(flat)
    return values


def prepare_pi05(
    base: PreparedDataset, horizon: int, *, relative: bool = False
) -> PreparedDataset:
    """Materialize absolute PI05 poses and fit statistics on training chunks.

    Stored actions stay absolute. With ``relative``, the policy's processors
    subtract the anchor state before normalization, so action statistics are
    fitted on those state-relative chunk targets instead.
    """
    representation = Pi05Representation()
    provenance = json.loads((base.root / PROVENANCE_PATH).read_text())
    contract = {
        "representation": representation.to_dict(),
        "chunk_size": horizon,
        "version": 2,
    }
    if relative:
        contract["relative_actions"] = {
            "mask": list(relative_action_mask()),
            "reference": "anchor_observation_state",
        }
    digest = hashlib.sha256(json.dumps(contract, sort_keys=True).encode()).hexdigest()[
        :16
    ]
    root = base.root.with_name(base.root.name + "-pi05-" + digest)
    if (root / PROVENANCE_PATH).is_file():
        saved = json.loads((root / PROVENANCE_PATH).read_text())
        if saved.get("pi05") != contract:
            raise ValueError("PI05 cache provenance mismatch")
        return PreparedDataset(root, base.source_repo_id, base.source_revision)
    if root.exists():
        raise ValueError(f"incomplete PI05 cache {root}")
    with tempfile.TemporaryDirectory(prefix=".pi05-", dir=root.parent) as temporary:
        stage = Path(temporary) / "view"
        stage.mkdir()
        shutil.copytree(base.root / "meta", stage / "meta")
        shutil.copytree(base.root / ".paw", stage / ".paw")
        (stage / PROVENANCE_PATH).unlink()
        (stage / "videos").symlink_to(
            (base.root / "videos").resolve(), target_is_directory=True
        )
        states, actions, episode_indices = [], [], []
        for path in sorted((base.root / "data").glob("chunk-*/file-*.parquet")):
            table = pq.read_table(path)
            for key, collected in [("observation.state", states), ("action", actions)]:
                physical = decode_model(
                    np.asarray(table[key].combine_chunks().flatten()).reshape(-1, 20)
                )
                collected.append(physical)
                table = table.set_column(
                    table.column_names.index(key),
                    key,
                    _fixed_list(representation.encode(physical)),
                )
            episode_indices.append(np.asarray(table["episode_index"]))
            dest = stage / path.relative_to(base.root)
            dest.parent.mkdir(parents=True, exist_ok=True)
            pq.write_table(table, dest)
        state, action = np.concatenate(states), np.concatenate(actions)
        episodes = np.concatenate(episode_indices)
        fit = set(provenance["normalization_episode_indices"])
        values = chunk_action_values(
            action, episodes, fit, horizon, state=state, relative=relative
        )
        stats = json.loads((stage / "meta/stats.json").read_text())
        stats["observation.state"] = _vector_stats(
            representation.encode(state[np.isin(episodes, list(fit))])
        )
        stats["action"] = _vector_stats(values)
        # Preserve absolute rotations and fractions in their known domains. Fitting
        # a tiny rotation coordinate range would amplify sensor noise. Relative
        # rotation offsets are fitted like positions, as FACT's openpi stats are.
        for key in ("observation.state", "action"):
            fixed_domains = [(CLOSED_FRACTION_DIMS, 0.0, 1.0)]
            if key == "observation.state" or not relative:
                fixed_domains.append((MODEL_ROTATION_DIMS, -1.0, 1.0))
            for dims, low, high in fixed_domains:
                for dim in dims:
                    for name, value in {
                        "min": low,
                        "max": high,
                        "q01": low,
                        "q99": high,
                        "mean": (low + high) / 2,
                        "std": 1.0,
                    }.items():
                        stats[key][name][dim] = value
            if provenance["frozen_arm"] is not None:
                _squash_arm_stats(stats, key, provenance["frozen_arm"])
        (stage / "meta/stats.json").write_text(json.dumps(stats, indent=2) + "\n")
        info = json.loads((stage / "meta/info.json").read_text())
        names = [
            f"{arm}.{name}"
            for arm in ("left", "right")
            for name in (
                "x",
                "y",
                "z",
                "r00",
                "r10",
                "r20",
                "r01",
                "r11",
                "r21",
                "closed_fraction",
            )
        ]
        for key in ("observation.state", "action"):
            info["features"][key]["names"] = names
        (stage / "meta/info.json").write_text(json.dumps(info, indent=2) + "\n")
        provenance.update(
            layout=LAYOUT,
            rotation_convention=ROTATION_CONVENTION,
            pi05=contract,
            physical_action_bounds={
                "min": action[np.isin(episodes, list(fit))].min(axis=0).tolist(),
                "max": action[np.isin(episodes, list(fit))].max(axis=0).tolist(),
            },
        )
        (stage / PROVENANCE_PATH).write_text(json.dumps(provenance, indent=2) + "\n")
        stage.rename(root)
    return PreparedDataset(root, base.source_repo_id, base.source_revision)
