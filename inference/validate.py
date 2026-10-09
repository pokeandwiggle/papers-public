"""Check a running policy server against recorded observations over gRPC.

Validate action shape, timestamps, motion bounds, grippers, and agreement
with demonstration motion. These checks do not establish task success.
Use saved checkpoint probes or an explicit dataset. Write a JSON report
and exit nonzero if any required check fails; latency is advisory.
"""

import argparse
import dataclasses
import io
import json
import os
import sys
import time
from pathlib import Path

import cv2
import grpc
import numpy as np
import pyarrow.parquet as pq
from paw_baseline.layout import FPS, GRIPPER_MAX_WIDTH_M
from paw_baseline.prepare import VIDEO_BACKEND
from paw_baseline.relative import (
    ACTION_REPRESENTATION as DELTA_REPRESENTATION,
)
from paw_baseline.relative import integrate_deltas
from paw_baseline.representation import decode_model, physical_bounds
from paw_baseline.serving_metadata import is_delta_metadata, load_serving_metadata
from paw_baseline.variant import load_policy_config
from paw_policy_toolkit.policy_service import (
    ActionChunk,
    CartesianAction,
    CartesianCommand,
    CartesianState,
    GripperCommand,
    ClientMessage,
    ConfigureRequest,
    GripperState,
    Observation,
    PolicyServiceStub,
    Pose,
    Quaternion,
    RgbFrame,
    Vector3,
)

ANCHORS = (0.05, 0.50, 0.85)
JPEG_QUALITY = 85

MAX_FIRST_ACTION_CMD_ERR_M = 0.10
MAX_GRIPPER_REVERSAL_M = 0.02
MAX_STEP_POS_M = 0.05
MAX_STEP_ROT_DEG = 15.0
BBOX_MARGIN_M = 0.10
# float32 wire rounding lands an exactly-full-open gripper a hair past 0.085.
GRIPPER_TOL_M = 1e-6
UNIT_QUAT_TOL = 0.01
# Below these, a probe cannot decide the corresponding plausibility question.
MIN_DECIDABLE_PATH_M = 0.001
MIN_DECIDABLE_DISP_M = 0.005
PATH_RATIO_LIMIT = 10.0

# Train-order slices: [tcp_pose_left(7), grip_left, tcp_pose_right(7), grip_right].
_ARM_SLICES = (
    ("left", slice(0, 3), slice(3, 7)),
    ("right", slice(8, 11), slice(11, 15)),
)
_GRIP_DIMS = (7, 15)


def _driven_arm_slices(frozen_arm: str | None):
    return tuple(item for item in _ARM_SLICES if item[0] != frozen_arm)


def _frozen_dims(frozen_arm: str) -> tuple[int, ...]:
    return tuple(range(8)) if frozen_arm == "left" else tuple(range(8, 16))


@dataclasses.dataclass(frozen=True)
class Probe:
    episode_index: int
    anchor: float
    frame: int
    global_index: int
    task: str


def chunk_to_train_rows(chunk: ActionChunk) -> np.ndarray:
    """Parse a wire chunk into (T, 16) rows in the flat-16 training order.

    Built from the proto structure rather than by permuting a flat vector, so
    a server that forgets the train→wire permutation shows up as garbage in
    the quaternion and gripper checks instead of silently passing.
    """
    if not chunk.actions:
        raise ValueError("server sent a chunk with zero actions")
    rows = []
    for action in chunk.actions:
        if len(action.arms) != 2 or len(action.grippers) != 2:
            raise ValueError(
                f"action carries {len(action.arms)} arms / {len(action.grippers)} grippers, expected 2/2"
            )
        row = []
        for arm, gripper in zip(action.arms, action.grippers, strict=True):
            p, o = arm.target_pose.position, arm.target_pose.orientation
            row.extend((p.x, p.y, p.z, o.x, o.y, o.z, o.w, gripper.target_width))
        rows.append(row)
    return np.asarray(rows, dtype=np.float64)


def _quat_step_degrees(quats: np.ndarray) -> np.ndarray:
    """Rotation angle in degrees between consecutive unit quaternions (N, 4)."""
    dots = np.abs(np.sum(quats[:-1] * quats[1:], axis=1))
    return np.degrees(2.0 * np.arccos(np.clip(dots, -1.0, 1.0)))


def check_protocol(
    chunk: ActionChunk,
    rows: np.ndarray,
    horizon: int,
    capture_time_ns: int,
    *,
    frozen_arm: str | None = None,
    observed_state: np.ndarray | None = None,
) -> dict:
    quat_norms = [
        np.linalg.norm(rows[:, sl_quat], axis=1) for _, _, sl_quat in _ARM_SLICES
    ]
    max_norm_error = float(max(np.abs(norms - 1.0).max() for norms in quat_norms))
    frozen_arm_holds = True
    max_frozen_error = 0.0
    if frozen_arm is not None:
        if observed_state is None:
            raise ValueError("observed_state is required to validate a frozen arm")
        dims = list(_frozen_dims(frozen_arm))
        max_frozen_error = float(np.abs(rows[:, dims] - observed_state[dims]).max())
        frozen_arm_holds = max_frozen_error <= 1e-5
    return {
        "actions": len(chunk.actions),
        "horizon_matches": len(chunk.actions) == horizon,
        "unit_quaternions": max_norm_error <= UNIT_QUAT_TOL,
        "max_quat_norm_error": max_norm_error,
        "target_time_echoed": chunk.target_time_ns == capture_time_ns,
        "frozen_arm_holds_observation": frozen_arm_holds,
        "max_frozen_arm_error": max_frozen_error,
        "pass": bool(
            len(chunk.actions) == horizon
            and max_norm_error <= UNIT_QUAT_TOL
            and chunk.target_time_ns == capture_time_ns
            and frozen_arm_holds
        ),
    }


def check_safety(
    rows: np.ndarray,
    first_command: np.ndarray,
    action_bbox: tuple[np.ndarray, np.ndarray],
    *,
    frozen_arm: str | None = None,
    allow_gripper_reversals: bool = False,
) -> dict:
    """Envelope checks, all in commanded-action space.

    ``first_command`` is the demonstrator's commanded action at the probe
    frame: the first predicted action is compared like for like. The measured
    state is not a fair anchor — the platform's commands deliberately lead
    the measurement by up to ~18 cm during fast transit.
    ``allow_gripper_reversals`` is an explicit simulation-only waiver; its use
    is recorded and every other envelope check remains mandatory.
    """
    max_pos_step = 0.0
    max_rot_step = 0.0
    first_action_err = 0.0
    driven = _driven_arm_slices(frozen_arm)
    for _, sl_pos, sl_quat in driven:
        if len(rows) > 1:
            steps = np.linalg.norm(np.diff(rows[:, sl_pos], axis=0), axis=1)
            max_pos_step = max(max_pos_step, float(steps.max()))
            max_rot_step = max(
                max_rot_step, float(_quat_step_degrees(rows[:, sl_quat]).max())
            )
        first_action_err = max(
            first_action_err,
            float(np.linalg.norm(rows[0, sl_pos] - first_command[sl_pos])),
        )
    low, high = action_bbox
    pos_dims = tuple(
        dim for _, sl_pos, _ in driven for dim in range(sl_pos.start, sl_pos.stop)
    )
    pos = rows[:, pos_dims]
    in_bbox = bool(
        (pos >= low[list(pos_dims)] - BBOX_MARGIN_M).all()
        and (pos <= high[list(pos_dims)] + BBOX_MARGIN_M).all()
    )
    grips = rows[:, _GRIP_DIMS]
    grippers_ok = bool(
        (grips >= -GRIPPER_TOL_M).all()
        and (grips <= GRIPPER_MAX_WIDTH_M + GRIPPER_TOL_M).all()
    )
    grip_delta = np.diff(grips, axis=0)
    reversals = int(
        np.sum(
            (grip_delta[:-1] * grip_delta[1:] < 0)
            & (
                np.minimum(abs(grip_delta[:-1]), abs(grip_delta[1:]))
                > MAX_GRIPPER_REVERSAL_M
            )
        )
    )
    return {
        "gripper_reversals_over_20mm": reversals,
        "gripper_reversals_waived": bool(allow_gripper_reversals and reversals),
        "max_pos_step_m": max_pos_step,
        "max_rot_step_deg": max_rot_step,
        "first_action_cmd_err_m": first_action_err,
        "within_bbox": in_bbox,
        "grippers_in_range": grippers_ok,
        "pass": bool(
            max_pos_step <= MAX_STEP_POS_M
            and max_rot_step <= MAX_STEP_ROT_DEG
            and first_action_err <= MAX_FIRST_ACTION_CMD_ERR_M
            and in_bbox
            and grippers_ok
            and (reversals == 0 or allow_gripper_reversals)
        ),
    }


def check_plausibility(
    rows: np.ndarray, truth: np.ndarray, *, frozen_arm: str | None = None
) -> dict:
    """Per-probe plausibility measures against the demonstrator's actions.

    Returns per-question verdicts of True/False/None, where None means the
    ground truth is too static for this probe to decide the question.
    """
    horizon = min(len(rows), len(truth))
    rows, truth = rows[:horizon], truth[:horizon]

    driven = _driven_arm_slices(frozen_arm)

    def path_length(arr: np.ndarray) -> float:
        return sum(
            float(np.linalg.norm(np.diff(arr[:, sl_pos], axis=0), axis=1).sum())
            for _, sl_pos, _ in driven
        )

    truth_path = path_length(truth)
    predicted_path = path_length(rows)
    if truth_path < MIN_DECIDABLE_PATH_M:
        magnitude_ok = None
        ratio = None
    else:
        ratio = predicted_path / truth_path
        magnitude_ok = bool(1.0 / PATH_RATIO_LIMIT <= ratio <= PATH_RATIO_LIMIT)

    pos_dims = tuple(
        dim for _, sl_pos, _ in driven for dim in range(sl_pos.start, sl_pos.stop)
    )
    truth_disp = truth[-1, pos_dims] - truth[0, pos_dims]
    dominant = int(np.argmax(np.abs(truth_disp)))
    if abs(truth_disp[dominant]) < MIN_DECIDABLE_DISP_M:
        sign_ok = None
    else:
        predicted_disp = rows[-1, pos_dims] - rows[0, pos_dims]
        sign_ok = bool(
            np.sign(predicted_disp[dominant]) == np.sign(truth_disp[dominant])
        )

    return {
        "truth_path_m": truth_path,
        "predicted_path_m": predicted_path,
        "path_ratio": ratio,
        "magnitude_ok": magnitude_ok,
        "dominant_axis": dominant,
        "dominant_sign_ok": sign_ok,
    }


def aggregate_plausibility(measures: list[dict]) -> dict:
    """Majority vote over decidable probes, per question."""
    verdict = {}
    for question in ("magnitude_ok", "dominant_sign_ok"):
        decidable = [m[question] for m in measures if m[question] is not None]
        verdict[question] = {
            "decidable": len(decidable),
            "agreeing": sum(decidable),
            "pass": bool(not decidable or sum(decidable) * 2 > len(decidable)),
        }
    verdict["pass"] = bool(
        all(verdict[q]["pass"] for q in ("magnitude_ok", "dominant_sign_ok"))
    )
    return verdict


def _select_episodes[T](episodes: list[T], count: int) -> list[T]:
    if count >= len(episodes):
        return episodes
    picks = np.linspace(0, len(episodes) - 1, count).round().astype(int)
    return [episodes[i] for i in dict.fromkeys(picks.tolist())]


def _load_episode_meta(root: Path) -> list[dict]:
    files = sorted((root / "meta/episodes").glob("chunk-*/file-*.parquet"))
    columns = [
        "episode_index",
        "tasks",
        "length",
        "dataset_from_index",
        "data/chunk_index",
        "data/file_index",
    ]
    rows: list[dict] = []
    for path in files:
        rows.extend(pq.read_table(path, columns=columns).to_pylist())
    return sorted(rows, key=lambda row: row["episode_index"])


def _episode_actions(root: Path, episode: dict) -> np.ndarray:
    """Absolute train-order commands (T, 16) of one prepared episode."""
    path = (
        root
        / f"data/chunk-{episode['data/chunk_index']:03d}/file-{episode['data/file_index']:03d}.parquet"
    )
    table = pq.read_table(
        path, columns=["episode_index", "frame_index", "action", "observation.state"]
    )
    mask = np.asarray(table["episode_index"]) == episode["episode_index"]
    order = np.argsort(np.asarray(table["frame_index"])[mask])
    actions = np.asarray(table["action"].combine_chunks().flatten()).reshape(
        len(table), -1
    )[mask][order]
    if _is_delta_root(root):
        # Frame 0 is relative to the measured pose, later frames to the last command.
        states = np.asarray(
            table["observation.state"].combine_chunks().flatten()
        ).reshape(len(table), -1)[mask][order]
        return integrate_deltas(decode_model(states[0]), actions)
    return _prepared_decoder(root)(actions).astype(np.float64)


def _is_delta_root(root: Path) -> bool:
    from paw_baseline.prepare import PROVENANCE_PATH

    stamp = json.loads((root / PROVENANCE_PATH).read_text())
    return stamp.get("action_representation") == DELTA_REPRESENTATION


def _prepared_decoder(root: Path):
    from paw_baseline.prepare import PROVENANCE_PATH

    stamp = json.loads((root / PROVENANCE_PATH).read_text())
    if "pi05" in stamp:
        from paw_baseline.pi05.geometry import Pi05Representation

        return Pi05Representation.from_dict(
            stamp["pi05"]["representation"]
        ).decode
    return decode_model


def _action_bbox(root: Path) -> tuple[np.ndarray, np.ndarray]:
    from paw_baseline.prepare import PROVENANCE_PATH

    stamp = json.loads((root / PROVENANCE_PATH).read_text())
    if "pi05" in stamp or stamp.get("action_representation") == DELTA_REPRESENTATION:
        bounds = stamp["physical_action_bounds"]
        return np.asarray(bounds["min"]), np.asarray(bounds["max"])
    stats = json.loads((root / "meta/stats.json").read_text())["action"]
    return physical_bounds(stats["min"]), physical_bounds(stats["max"])


def _encode_jpeg(image_chw: np.ndarray) -> bytes:
    """(3, H, W) float32 [0, 1] RGB → JPEG bytes, like the follower sends."""
    hwc = (
        (np.clip(image_chw, 0.0, 1.0) * 255.0)
        .round()
        .astype(np.uint8)
        .transpose(1, 2, 0)
    )
    ok, buf = cv2.imencode(
        ".jpg",
        cv2.cvtColor(hwc, cv2.COLOR_RGB2BGR),
        [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY],
    )
    if not ok:
        raise RuntimeError("JPEG encoding failed")
    return buf.tobytes()


def _pose(vec7: np.ndarray) -> Pose:
    return Pose(
        position=Vector3(x=float(vec7[0]), y=float(vec7[1]), z=float(vec7[2])),
        orientation=Quaternion(
            x=float(vec7[3]), y=float(vec7[4]), z=float(vec7[5]), w=float(vec7[6])
        ),
    )


def _held_command(command: np.ndarray) -> CartesianAction:
    """One train-order flat-16 command as the robot reports scheduled motion."""
    return CartesianAction(
        arms=[
            CartesianCommand(target_pose=_pose(command[0:7])),
            CartesianCommand(target_pose=_pose(command[8:15])),
        ],
        grippers=[
            GripperCommand(target_width=float(command[7])),
            GripperCommand(target_width=float(command[15])),
        ],
    )


def _build_observation(
    state: np.ndarray,
    images: dict[str, np.ndarray],
    capture_time_ns: int,
    previous: np.ndarray | None = None,
) -> Observation:
    """A chunk request; ``previous`` is the held command a synchronous robot reports."""
    observation = Observation(
        capture_time_ns=capture_time_ns,
        chunk_requested=True,
        tcp_left=CartesianState(pose=_pose(state[0:7])),
        tcp_right=CartesianState(pose=_pose(state[8:15])),
        gripper_left=GripperState(width=float(state[7])),
        gripper_right=GripperState(width=float(state[15])),
    )
    if previous is not None and np.isfinite(previous).all():
        observation.replaceable_actions.append(_held_command(previous))
    for key, image in images.items():
        slot = key.removeprefix("observation.images.")
        getattr(observation, slot).CopyFrom(RgbFrame(data=_encode_jpeg(image)))
    return observation


def _request_chunk(
    stub: PolicyServiceStub, observation: Observation, timeout: float
) -> tuple[ActionChunk, float]:
    start = time.monotonic()
    responses = stub.Run(
        iter([ClientMessage(observation=observation)]), timeout=timeout
    )
    first_chunk: tuple[ActionChunk, float] | None = None
    for message in responses:
        if first_chunk is None and message.WhichOneof("payload") == "action_chunk":
            first_chunk = message.action_chunk, time.monotonic() - start
    # EOF confirms the SDK has released this Run's stream lock. Returning on
    # the first chunk races the next probe and hides errors during teardown.
    if first_chunk is None:
        raise RuntimeError("Run stream ended without an action chunk")
    return first_chunk


def _train_stamp(checkpoint: Path) -> dict | None:
    """Find the train stamp in the run output dir above the checkpoint
    (``<output>/lerobot/checkpoints/<step>/pretrained_model``)."""
    for parent in checkpoint.parents[:6]:
        stamp = parent / ".paw/baseline_train.json"
        if stamp.exists():
            return json.loads(stamp.read_text())
    return None


def _resolve_dataset_root(checkpoint: Path, dataset: str | None) -> Path:
    if dataset is not None:
        from paw_baseline.prepare import prepare_dataset

        return prepare_dataset(dataset).root
    train_config = json.loads((checkpoint / "train_config.json").read_text())
    root = Path(train_config["dataset"]["root"])
    if (root / "meta/info.json").exists():
        return root

    # A restored model carries the portable dataset identity in serving
    # metadata, while train_config.json may name a producer-local cache path.
    serving = load_serving_metadata(checkpoint)
    repo_id = serving.get("dataset_repo_id")
    if isinstance(repo_id, str) and repo_id:
        from paw_baseline.prepare import prepare_dataset

        return prepare_dataset(repo_id).root
    raise SystemExit(
        f"prepared view {root} (from train_config.json) is not on this machine;"
        " pass --dataset"
    )


def dataset_probes(
    root: Path, camera_keys: list[str], horizon: int, count: int, holdout: set[int]
) -> tuple[list[dict], tuple[np.ndarray, np.ndarray]]:
    """Extract the same real observation/action probes used by validation."""
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    episodes = _load_episode_meta(root)
    candidates = (
        [episode for episode in episodes if episode["episode_index"] in holdout]
        if holdout
        else episodes
    )
    if holdout - {episode["episode_index"] for episode in candidates}:
        raise ValueError("held-out episodes are absent from the dataset")
    candidates = [episode for episode in candidates if episode["length"] >= horizon]
    if not candidates:
        raise ValueError(
            "validation requires an episode at least as long as its horizon"
        )
    dataset = LeRobotDataset(repo_id=root.name, root=root, video_backend=VIDEO_BACKEND)
    samples = []
    for episode in _select_episodes(candidates, count):
        length = episode["length"]
        actions = _episode_actions(root, episode)
        for anchor in ANCHORS:
            frame = min(round(anchor * (length - 1)), length - horizon)
            probe = Probe(
                episode["episode_index"],
                anchor,
                frame,
                episode["dataset_from_index"] + frame,
                episode["tasks"][0],
            )
            item = dataset[probe.global_index]
            samples.append(
                {
                    "probe": probe,
                    "state": _prepared_decoder(root)(
                        item["observation.state"].numpy()
                    ).astype(np.float64),
                    "images": {key: item[key].numpy() for key in camera_keys},
                    "truth": actions[frame : frame + horizon],
                    # The command held when the probe's chunk was requested.
                    "previous": actions[frame - 1]
                    if frame > 0
                    else np.full(actions.shape[1], np.nan),
                }
            )
    if not samples:
        raise ValueError("validation requires real dataset probes")
    return samples, _action_bbox(root)


def make_validation_bundle(root: Path, config: dict, holdout: set[int]) -> bytes:
    """Freeze real validation probes so a checkpoint is portable across machines."""
    cameras = [
        key
        for key, feature in config["input_features"].items()
        if feature["type"] == "VISUAL"
    ]
    horizon = config["n_action_steps"]
    samples, bbox = dataset_probes(root, cameras, horizon, 3, holdout)
    metadata = {
        "schema_version": 1,
        "camera_keys": cameras,
        "horizon": horizon,
        "probe_pool": "holdout" if holdout else "all",
        "probes": [dataclasses.asdict(sample["probe"]) for sample in samples],
    }
    arrays = {
        "metadata": np.asarray(json.dumps(metadata)),
        "bbox_min": bbox[0],
        "bbox_max": bbox[1],
    }
    for i, sample in enumerate(samples):
        arrays[f"state_{i}"] = sample["state"]
        arrays[f"truth_{i}"] = sample["truth"]
        arrays[f"previous_{i}"] = sample.get("previous", np.full(16, np.nan))
        arrays.update(
            {f"image_{i}_{j}": sample["images"][key] for j, key in enumerate(cameras)}
        )
    output = io.BytesIO()
    np.savez_compressed(output, **arrays)
    return output.getvalue()


def load_validation_bundle(
    path: Path, camera_keys: list[str], horizon: int, count: int
) -> tuple[list[dict], tuple[np.ndarray, np.ndarray], str]:
    """Read portable probes, rejecting mismatched model contracts and empty bundles."""
    if not 1 <= count <= 3:
        raise ValueError(
            "portable probes support 1-3 episodes; pass --dataset for a larger validation"
        )
    with np.load(path, allow_pickle=False) as bundle:
        metadata = json.loads(str(bundle["metadata"]))
        if (
            metadata["schema_version"] != 1
            or metadata["horizon"] != horizon
            or metadata["camera_keys"] != camera_keys
        ):
            raise ValueError("portable validation probes do not match the checkpoint")
        probes = [Probe(**value) for value in metadata["probes"]]
        if not probes:
            raise ValueError("portable validation bundle contains no probes")
        episodes = list(dict.fromkeys(probe.episode_index for probe in probes))
        selected = set(_select_episodes(episodes, count))
        samples = [
            {
                "probe": probe,
                "state": bundle[f"state_{i}"],
                "truth": bundle[f"truth_{i}"],
                "previous": bundle[f"previous_{i}"]
                if f"previous_{i}" in bundle
                else None,
                "images": {
                    key: bundle[f"image_{i}_{j}"] for j, key in enumerate(camera_keys)
                },
            }
            for i, probe in enumerate(probes)
            if probe.episode_index in selected
        ]
        for sample in samples:
            if sample["truth"].shape != (horizon, 16) or sample["state"].shape != (16,):
                raise ValueError("invalid portable probe dimensions")
        return samples, (bundle["bbox_min"], bundle["bbox_max"]), metadata["probe_pool"]


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="python -m inference.validate", description=__doc__
    )
    parser.add_argument(
        "--server", required=True, help="policy server HOST:PORT (plaintext gRPC)"
    )
    parser.add_argument(
        "--checkpoint", type=Path, required=True, help="lerobot pretrained_model dir"
    )
    parser.add_argument(
        "--model-id", default=None, help="Configure model_id (default: checkpoint path)"
    )
    parser.add_argument(
        "--dataset",
        default=None,
        help="Dataset repo id or local path (default: training dataset)",
    )
    parser.add_argument(
        "--episodes", type=int, default=3, help="number of episodes to probe"
    )
    parser.add_argument(
        "--configure-timeout",
        type=float,
        default=900.0,
        help="Configure RPC timeout in seconds, including model loading, compilation and warmup",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=5.0,
        help="per-probe Run RPC timeout in seconds, including stream teardown",
    )
    parser.add_argument(
        "--output", type=Path, default=None, help="report path (default: operation dir)"
    )
    parser.set_defaults(allow_simulation_gripper_reversals=False)
    if os.environ.get("PAW_ENABLE_SIMULATION") == "1":
        parser.add_argument(
            "--allow-simulation-gripper-reversals",
            action="store_true",
            help="operator-approved simulation only: waive gripper reversals, retaining all other checks",
        )
    args = parser.parse_args()

    checkpoint = args.checkpoint.resolve()
    config = json.loads((checkpoint / "config.json").read_text())
    serving_metadata = load_serving_metadata(checkpoint)
    frozen_arm = serving_metadata["frozen_arm"]
    relative_actions = is_delta_metadata(serving_metadata)
    native_action_steps = config["n_action_steps"]
    horizon = load_policy_config(checkpoint).n_action_steps
    replan_budget_s = horizon / FPS / 2
    camera_keys = [
        key
        for key, feat in config["input_features"].items()
        if feat["type"] == "VISUAL"
    ]

    # Variants keep the native probe contract and anchors; compare only the
    # configured action prefix below. Existing portable bundles stay unchanged.
    portable = checkpoint / ".paw/validation_probes.npz"
    if args.dataset is None and portable.is_file():
        samples, bbox, probe_pool = load_validation_bundle(
            portable, camera_keys, native_action_steps, args.episodes
        )
        root = portable
    else:
        root = _resolve_dataset_root(checkpoint, args.dataset)
        stamp = _train_stamp(checkpoint) if args.dataset is None else None
        holdout = set((stamp or {}).get("holdout_episodes") or [])
        samples, bbox = dataset_probes(
            root, camera_keys, native_action_steps, args.episodes, holdout
        )
        probe_pool = "holdout" if holdout else "all"

    channel = grpc.insecure_channel(args.server)
    stub = PolicyServiceStub(channel)

    results = []
    warnings = (
        ["Simulation-only gripper reversal waiver enabled by the operator"]
        if args.allow_simulation_gripper_reversals
        else []
    )
    configured_task = None
    for sample in samples:
        probe = sample["probe"]
        if probe.task != configured_task:
            stub.Configure(
                ConfigureRequest(
                    model_id=args.model_id or str(checkpoint),
                    task_instruction=probe.task,
                ),
                timeout=args.configure_timeout,
                metadata=(("x-paw-model-id", args.model_id or str(checkpoint)),),
            )
            configured_task = probe.task
        state = sample["state"]
        images = sample["images"]
        capture_time_ns = time.monotonic_ns()
        observation = _build_observation(
            state,
            images,
            capture_time_ns,
            sample.get("previous") if relative_actions else None,
        )

        chunk, latency_s = _request_chunk(stub, observation, args.timeout)
        rows = chunk_to_train_rows(chunk)

        truth = sample["truth"][:horizon]

        if latency_s > replan_budget_s:
            warnings.append(
                f"episode {probe.episode_index} anchor {probe.anchor}: latency {latency_s:.2f}s "
                f"exceeds the {replan_budget_s:.2f}s replan budget"
            )
        results.append(
            {
                "episode_index": probe.episode_index,
                "anchor": probe.anchor,
                "frame": probe.frame,
                "task": probe.task,
                "latency_s": round(latency_s, 3),
                "protocol": check_protocol(
                    chunk,
                    rows,
                    horizon,
                    capture_time_ns,
                    frozen_arm=frozen_arm,
                    observed_state=state,
                ),
                "safety": check_safety(
                    rows,
                    truth[0],
                    bbox,
                    frozen_arm=frozen_arm,
                    allow_gripper_reversals=args.allow_simulation_gripper_reversals,
                ),
                "plausibility": check_plausibility(rows, truth, frozen_arm=frozen_arm),
            }
        )
    channel.close()

    plausibility = aggregate_plausibility([r["plausibility"] for r in results])
    protocol_pass = all(r["protocol"]["pass"] for r in results)
    safety_pass = all(r["safety"]["pass"] for r in results)
    hard_pass = bool(protocol_pass and safety_pass and plausibility["pass"])
    report = {
        "server": args.server,
        "model_id": args.model_id or str(checkpoint),
        "checkpoint": str(checkpoint),
        "dataset_root": str(root),
        "probe_pool": probe_pool,
        "horizon": horizon,
        "native_action_steps": native_action_steps,
        "frozen_arm": frozen_arm,
        "replan_budget_s": replan_budget_s,
        "probes": results,
        "aggregate": {
            "protocol_pass": protocol_pass,
            "safety_pass": safety_pass,
            "plausibility": plausibility,
            "hard_pass": hard_pass,
        },
        "warnings": warnings,
    }

    output = args.output
    if output is None:
        operation_dir = os.environ.get("PAW_OPERATION_DIR")
        output = (
            Path(operation_dir) if operation_dir else Path.cwd()
        ) / "validate_report.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")

    print(json.dumps(report, indent=2))
    print(f"validate report written: {output}", file=sys.stderr)
    sys.exit(0 if hard_pass else 1)


if __name__ == "__main__":
    main()
