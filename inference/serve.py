"""Serve a LeRobot checkpoint over the policy gRPC protocol.

Load saved processors and warm up during the first Configure after binding.
Later Configure calls reuse the model; Run requires successful Configure.
Serving compilation can be configured independently of training.
Use only on a trusted network; this server provides no authentication.
"""

import argparse
import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Event
from typing import Literal

import numpy as np
import torch
from paw_baseline.layout import (
    GRIPPER_MAX_WIDTH_M,
    PAW_DIM,
    STATE_KEY_DIMS,
    TRAIN_GRIP_DIMS,
    TRAIN_QUAT_SLICES,
    WIRE_FROM_TRAIN,
    pin_frozen_arm,
)
from paw_baseline.hist_diffusion.serving import (
    ReplanSchedule,
    StridedHistory,
    blend_seam,
)
from paw_baseline.representation import MODEL_DIM, decode_model, encode_model
from paw_baseline.serving_metadata import load_serving_metadata
from paw_baseline.variant import load_policy_config
from paw_policy_toolkit import policy_service as ps
from paw_policy_toolkit.errors import StreamNotActive
from paw_policy_toolkit.policy_service import ConfigureRequest, ObservationRequirements
from paw_policy_toolkit.server import PolicyServer, serve

logger = logging.getLogger("paw_baseline.serve")

DEFAULT_PORT = 50262
#: Minimum predicted quaternion norm accepted for direction-preserving normalization.
MIN_QUATERNION_NORM = 0.1
#: Diffusion families sharing the native U-Net loading and compilation path.
DIFFUSION_TYPES = ("diffusion", "hist_diffusion")
#: Log history-policy pacing statistics after this many plans.
PLAN_LOG_INTERVAL = 100


# Measured TCP pose key -> desired TCP pose key (present when the station supplies it).
_DESIRED_POSE_KEYS = {
    f"observation.state.tcp_pose_{side}": f"observation.state.desired_tcp_pose_{side}"
    for side in ("left", "right")
}


def _attn_desired_key() -> str:
    from paw_baseline.attn_diffusion.serving import DESIRED_STATE

    return DESIRED_STATE


def _fold_state(observation: dict, encoder=encode_model) -> np.ndarray:
    """Encode physical per-key sub-features into the flat-20 model order.

    The toolkit's own flat ``observation.state`` is in wire order (grippers
    last) — folding from the sub-features is what keeps the model input in
    the order it was trained on.
    """
    parts = [
        np.asarray(observation[key], dtype=np.float32).reshape(1, dim)
        for key, dim in STATE_KEY_DIMS
    ]
    return encoder(np.concatenate(parts, axis=1))


def _emit_hygiene(chunk: np.ndarray) -> np.ndarray:
    """Enforce action-space invariants on a train-order chunk before emitting.

    Policies regress quaternion components and gripper widths as free floats,
    so normalize their quaternion direction and clamp gripper widths. Refuse
    non-finite outputs and quaternion norms below MIN_QUATERNION_NORM, where
    normalization would excessively amplify errors in the predicted components.
    """
    if not np.isfinite(chunk).all():
        raise ValueError("policy emitted non-finite action values")
    chunk = chunk.copy()
    for sl in TRAIN_QUAT_SLICES:
        norms = np.linalg.norm(chunk[..., sl], axis=-1, keepdims=True)
        if float(norms.min()) < MIN_QUATERNION_NORM:
            raise ValueError(
                f"policy emitted a degenerate quaternion (norm {float(norms.min()):.3f})"
            )
        chunk[..., sl] /= norms
    grips = list(TRAIN_GRIP_DIMS)
    chunk[..., grips] = np.clip(chunk[..., grips], 0.0, GRIPPER_MAX_WIDTH_M)
    return chunk


class BaselinePolicyServer(PolicyServer):
    """Serve one checkpoint and Run stream at a time.

    Diffusion, PI0.5 and SmolVLA default to compiled inference regardless of checkpoint
    settings; compile_model can explicitly enable or disable it. Other families retain
    their checkpoint setting unless overridden. PI0.5 uses max-autotune for
    serving; Diffusion uses default, independently of their training mode.
    Compiled inference defaults to
    highest FP32 matmul precision; matmul_precision overrides it before warmup.
    stop_event requests process shutdown after an initialization failure.
    """

    # hist_diffusion only: strided context, server-paced replanning, async plans.
    _history: StridedHistory | None = None
    _schedule: ReplanSchedule | None = None
    _pending = None
    _emitter = None
    _last_chunk: tuple[int, np.ndarray] | None = None
    # attn_diffusion only: cached-embedding planner with seam interpolation
    # (imported lazily so other policy types start without loading timm).
    _attn = None

    def __init__(
        self,
        checkpoint: Path,
        device: str,
        model_id: str | None = None,
        *,
        compile_model: bool | None = None,
        compile_mode: str | None = None,
        matmul_precision: Literal["high", "highest"] | None = None,
    ) -> None:
        super().__init__()
        self._device = device
        self._model_id = model_id
        self._checkpoint = checkpoint
        self._compile_model = compile_model
        self._compile_mode = compile_mode
        self._matmul_precision = matmul_precision
        self._inference_executor = None
        self._loaded = False
        self.stop_event = Event()
        self._configured = False

    def _load(self) -> None:
        checkpoint = self._checkpoint
        device = self._device
        compile_model = self._compile_model
        compile_mode = self._compile_mode
        matmul_precision = self._matmul_precision
        config = json.loads((checkpoint / "config.json").read_text())
        serving_metadata = load_serving_metadata(checkpoint)
        self._frozen_arm = serving_metadata["frozen_arm"]
        self._representation = None
        if "pi05" in serving_metadata:
            from paw_baseline.pi05.geometry import Pi05Representation

            self._representation = Pi05Representation.from_dict(
                serving_metadata["pi05"]["representation"]
            )
        native_config = load_policy_config(checkpoint)
        if compile_model is None and config["type"] in {
            *DIFFUSION_TYPES,
            "paw_pi05",
            "pi05",
            "smolvla",
        }:
            compile_model = True
        if compile_model is not None:
            if not hasattr(native_config, "compile_model"):
                raise ValueError(
                    f"{config['type']} does not support native model compilation"
                )
            native_config.compile_model = compile_model
        compiled = getattr(native_config, "compile_model", False)
        if compiled:
            if compile_mode is not None:
                native_config.compile_mode = compile_mode
            elif config["type"] in {"paw_pi05", "pi05"}:
                native_config.compile_mode = "max-autotune"
            elif config["type"] in DIFFUSION_TYPES:
                native_config.compile_mode = "default"
            logger.info(
                "Enabling native torch.compile (%s)", native_config.compile_mode
            )
            if matmul_precision is None:
                matmul_precision = "highest"

        state_shape = config["input_features"]["observation.state"]["shape"]
        if state_shape != [MODEL_DIM]:
            raise RuntimeError(
                f"checkpoint expects observation.state {state_shape}, not [{MODEL_DIM}]"
            )
        action_shape = config["output_features"]["action"]["shape"]
        if action_shape != [MODEL_DIM]:
            raise RuntimeError(
                f"checkpoint emits action {action_shape}, not [{MODEL_DIM}]"
            )

        # (camera key, wire slot, height, width) per VISUAL input; lerobot
        # shapes are channel-first (C, H, W).
        self._cameras = tuple(
            (
                key,
                key.removeprefix("observation.images."),
                feature["shape"][1],
                feature["shape"][2],
            )
            for key, feature in config["input_features"].items()
            if feature["type"] == "VISUAL"
        )
        if not self._cameras:
            raise RuntimeError(
                "checkpoint has no VISUAL input features; nothing to require from the client"
            )

        from lerobot.policies import get_policy_class, make_pre_post_processors

        policy_class = get_policy_class(config["type"])
        if config["type"] == "attn_diffusion":
            # Weights come from the checkpoint: build the encoders without
            # downloading pretrained backbones (normalization is unchanged).
            native_config.pretrained_backbone = False
            self._policy = policy_class.from_pretrained(
                checkpoint, config=native_config, strict=True
            )
        elif config["type"] in DIFFUSION_TYPES:
            # Native compile wraps U-Net and changes checkpoint keys (_orig_mod).
            # Load strictly with the saved structure before changing serving mode.
            saved_compiled = config.get("compile_model", False)
            native_config.compile_model = saved_compiled
            self._policy = policy_class.from_pretrained(
                checkpoint, config=native_config, strict=True
            )
            unet = self._policy.diffusion.unet
            if compiled and not saved_compiled:
                self._policy.diffusion.unet = torch.compile(
                    unet, mode=native_config.compile_mode
                )
            elif saved_compiled and not compiled:
                self._policy.diffusion.unet = unet._orig_mod
            native_config.compile_model = compiled
        else:
            self._policy = policy_class.from_pretrained(
                checkpoint, config=native_config
            )
        self._policy.to(device)
        self._policy.eval()
        # Native PI05 initialization sets high; override before lazy compilation.
        if matmul_precision is not None:
            torch.set_float32_matmul_precision(matmul_precision)
        logger.info("FP32 matmul precision: %s", torch.get_float32_matmul_precision())
        self._n_action_steps = self._policy.config.n_action_steps
        n_obs_steps = config.get("n_obs_steps") or self._policy.config.n_obs_steps
        if config["type"] == "hist_diffusion":
            if self._representation is not None:
                raise RuntimeError("hist_diffusion serves the generic representation only")
            native = self._policy.config
            self._history = StridedHistory(native.n_obs_steps, native.obs_stride)
            self._schedule = ReplanSchedule(native.replan_steps)
            self._seam_steps = native.seam_blend_steps
            self._plan_stats: list[tuple[int, float, int]] = []
            logger.info(
                "history policy: n_obs_steps=%d obs_stride=%d emitted=%d replan_steps=%d seam_blend_steps=%d",
                native.n_obs_steps,
                native.obs_stride,
                native.n_action_steps,
                native.replan_steps,
                native.seam_blend_steps,
            )
        elif config["type"] == "attn_diffusion":
            if self._representation is not None:
                raise RuntimeError("attn_diffusion serves the generic representation only")
            from paw_baseline.attn_diffusion.serving import AttnPlanner

            native = self._policy.config
            # Measured-pose anchor (training semantics). The desired-pose anchor
            # was clearly worse in the offline replay check (tmp/anchor_offline_*).
            self._attn = AttnPlanner(self._policy, anchor="measured")
            self._schedule = ReplanSchedule(native.replan_steps)
            self._interpolation_ms = native.interpolation_ms
            self._plan_stats = []
            logger.info(
                "attn policy: n_obs_steps=%d obs_stride=%d emitted=%d replan_steps=%d interpolation_ms=%.0f dtype=%s anchor=%s",
                native.n_obs_steps,
                native.obs_stride,
                native.n_action_steps,
                native.replan_steps,
                native.interpolation_ms,
                native.dtype,
                self._attn.anchor,
            )
        elif n_obs_steps != 1:
            raise RuntimeError(
                f"baseline serving requires n_obs_steps=1, got {n_obs_steps}; "
                "it emits only on chunk requests and cannot preserve adjacent-frame context"
            )
        device_override = {"device": device}
        self._preprocess, self._postprocess = make_pre_post_processors(
            self._policy.config,
            pretrained_path=checkpoint,
            preprocessor_overrides={"device_processor": device_override},
            postprocessor_overrides={"device_processor": device_override},
        )
        self._task_instruction = ""
        # cuDNN handles and CUDA Graph capture belong to the inference thread.
        # gRPC creates a new observation-pump thread for each Run stream.
        # History plans always run there so observation intake never waits.
        self._inference_executor = (
            ThreadPoolExecutor(max_workers=1, thread_name_prefix="policy-inference")
            if compiled or self._history is not None or self._attn is not None
            else None
        )
        self._warmup()
        logger.info(
            "loaded %s checkpoint %s on %s (cameras: %s, horizon: %d, frozen_arm: %s)",
            config["type"],
            checkpoint,
            device,
            [slot for _, slot, _, _ in self._cameras],
            self._n_action_steps,
            self._frozen_arm,
        )

    def close(self) -> None:
        """Release the inference thread after serving has stopped."""
        if self._inference_executor is not None:
            self._inference_executor.shutdown()

    def _warmup(self) -> None:
        """Run one full inference so lazy kernels compile before Configure succeeds."""
        state = np.zeros((1, PAW_DIM), dtype=np.float32)
        state[0, 6] = state[0, 14] = 1.0  # unit quaternions (w last per pose)
        encoder = (
            encode_model
            if self._representation is None
            else self._representation.encode
        )
        batch: dict = {"observation.state": encoder(state), "task": "warmup"}
        for key, _, height, width in self._cameras:
            batch[key] = np.zeros((1, 3, height, width), dtype=np.float32)
        self._policy.reset()
        if self._attn is not None:
            frame = {"observation.state": batch["observation.state"][0]}
            frame.update({key: batch[key][0] for key, _, _, _ in self._cameras})
            self._attn.append(frame)
            snapshot = self._attn.snapshot(0)
            self._inference_executor.submit(self._attn.plan, snapshot).result()
            self._attn.reset()
            self._attn.anchors_used = dict.fromkeys(self._attn.anchors_used, 0)
        elif self._history is not None:
            frame = {"observation.state": batch["observation.state"][0]}
            frame.update({key: batch[key][0] for key, _, _, _ in self._cameras})
            self._history.append(frame)
            context = self._history.context()
            self._history.reset()
            self._inference_executor.submit(self._predict_history, context).result()
        else:
            self._predict(batch)
        self._policy.reset()

    def _predict(self, batch: dict) -> np.ndarray:
        if self._inference_executor is not None:
            return self._inference_executor.submit(self._predict_chunk, batch).result()
        return self._predict_chunk(batch)

    def _predict_chunk(self, batch: dict) -> np.ndarray:
        """pre → public action API → per-step post; returns a train-order chunk.

        ``select_action`` owns each policy's observation/action queues. Calling
        ``predict_action_chunk`` directly after reset bypasses queue population
        and crashes queue-backed policies such as MultiTask DiT.
        """
        tensors = {
            key: torch.from_numpy(value) if isinstance(value, np.ndarray) else value
            for key, value in batch.items()
        }
        with torch.inference_mode():
            processed = self._preprocess(tensors)
            chunk = torch.stack(
                [
                    self._policy.select_action(processed)
                    for _ in range(self._n_action_steps)
                ],
                dim=1,
            )
            # The saved postprocessor (unnormalizer) takes one (B, action_dim)
            # action at a time, mirroring lerobot's own policy server.
            steps = [self._postprocess(chunk[:, i, :]) for i in range(chunk.shape[1])]
        return torch.stack(steps, dim=1).detach().cpu().numpy()

    def _predict_history(self, context: dict) -> np.ndarray:
        """Strided context (oldest first) → train-order (1, T, MODEL_DIM) chunk.

        Runs on the inference thread. The saved preprocessor normalizes the
        (1, n_obs_steps, ...) context exactly as it normalizes single frames.
        """
        tensors: dict = {
            key: torch.from_numpy(value).unsqueeze(0) for key, value in context.items()
        }
        tensors["task"] = self._task_instruction
        with torch.inference_mode():
            processed = self._preprocess(tensors)
            chunk = self._policy.predict_history_chunk(
                processed["observation.state"],
                {key: processed[key] for key, _, _, _ in self._cameras},
            )
            steps = [self._postprocess(chunk[:, i, :]) for i in range(chunk.shape[1])]
        return torch.stack(steps, dim=1).detach().cpu().numpy()

    def _plan_and_emit(self, context: dict, grounding: dict, emitter) -> None:
        """Plan from one strided context and emit it grounded at its observation.

        The seam crossfades from the server's previous chunk, which the robot
        executes until this one lands (in-order stream, no client crossfade
        under the sync runtime). The robot's own replaceable plan can predate
        that chunk when round trips exceed the replanning interval.
        """
        started = time.monotonic()
        decoded = decode_model(self._predict_history(context))
        chunk = _emit_hygiene(decoded)
        wire = np.ascontiguousarray(chunk[:, :, WIRE_FROM_TRAIN])
        if self._frozen_arm is not None:
            wire = pin_frozen_arm(wire, grounding, self._frozen_arm)
        committed = grounding["committed_count"]
        previous = self._last_chunk
        if previous is not None and grounding["robot_has_plan"]:
            offset = grounding["index"] - previous[0] + committed
            wire = blend_seam(
                wire, committed, previous[1][:, offset:], self._seam_steps
            )
        emitter.emit_chunk(
            {"action": wire},
            grounding["capture_time_ns"],
            discard_actions=grounding["actions_discarded"],
        )
        self._last_chunk = (grounding["index"], wire)
        self._record_plan(grounding["index"], time.monotonic() - started, committed)

    def _plan_and_emit_attn(self, snapshot: dict, grounding: dict, emitter) -> None:
        """Plan absolute targets from accumulated deltas and enter the new chunk
        by interpolating from the previous one (paper A.3), then emit."""
        from paw_baseline.attn_diffusion.serving import interpolate_into_chunk

        started = time.monotonic()
        absolute = self._attn.plan(snapshot)
        chunk = _emit_hygiene(decode_model(absolute[None]))
        wire = np.ascontiguousarray(chunk[:, :, WIRE_FROM_TRAIN])
        if self._frozen_arm is not None:
            wire = pin_frozen_arm(wire, grounding, self._frozen_arm)
        committed = grounding["committed_count"]
        previous = None
        if self._last_chunk is not None and grounding["robot_has_plan"]:
            # The robot runs the previous chunk until this one lands; its action
            # just before the first replaceable slot is the interpolation start.
            previous_index, previous_wire = self._last_chunk
            slot = grounding["index"] - previous_index + committed - 1
            if 0 <= slot < previous_wire.shape[1]:
                previous = previous_wire[0, slot]
        wire = interpolate_into_chunk(wire, committed, previous, self._interpolation_ms)
        emitter.emit_chunk(
            {"action": wire},
            grounding["capture_time_ns"],
            discard_actions=grounding["actions_discarded"],
        )
        self._last_chunk = (grounding["index"], wire)
        self._record_plan(grounding["index"], time.monotonic() - started, committed)

    def _record_plan(self, index: int, seconds: float, committed: int) -> None:
        """Log one episode's pacing; warn when replanning falls behind schedule."""
        stats = self._plan_stats
        if stats and index - stats[-1][0] > self._schedule.replan_steps:
            logger.warning(
                "replan interval %d steps exceeds replan_steps=%d (inference %.0f ms)",
                index - stats[-1][0],
                self._schedule.replan_steps,
                1000 * seconds,
            )
        stats.append((index, seconds, committed))
        if len(stats) >= PLAN_LOG_INTERVAL:
            self._log_plan_stats()

    def _log_plan_stats(self) -> None:
        stats = self._plan_stats
        if len(stats) > 1:
            intervals = np.diff([i for i, _, _ in stats])
            latency = [s for _, s, _ in stats]
            logger.info(
                "plans=%d inference_ms mean=%.1f max=%.1f interval_steps mean=%.2f max=%d committed mean=%.1f",
                len(stats),
                1000 * float(np.mean(latency)),
                1000 * float(np.max(latency)),
                float(np.mean(intervals)),
                int(np.max(intervals)),
                float(np.mean([c for _, _, c in stats])),
            )
        if self._attn is not None:
            used = self._attn.anchors_used
            log = logger.warning if used["measured"] and self._attn.anchor == "desired" else logger.info
            log("attn anchors used (cumulative): %s", used)
        # Keep the latest plan so the next interval stays within the episode.
        del stats[:-1]

    def _wait_for_plan(self, *, raise_errors: bool) -> None:
        """Join the in-flight plan; surface its failure unless the stream ended."""
        pending, self._pending = self._pending, None
        if pending is None:
            return
        try:
            pending.result()
        except StreamNotActive:
            pass
        except Exception:
            if raise_errors:
                raise
            logger.exception("history plan failed after its stream ended")

    def _on_history_observation(self, observation: dict) -> None:
        """Buffer every grid frame; plan when the schedule is due and idle."""
        frame = {"observation.state": _fold_state(observation)[0]}
        for key, _, height, width in self._cameras:
            image = np.asarray(observation[key], dtype=np.float32)
            if image.shape != (1, 3, height, width):
                raise ValueError(
                    f"{key} arrived as {image.shape}, required (1, 3, {height}, {width})"
                )
            frame[key] = image[0]
        if self._attn is not None and all(key in observation for key in _DESIRED_POSE_KEYS.values()):
            desired = {**observation, **{k: observation[v] for k, v in _DESIRED_POSE_KEYS.items()}}
            frame[_attn_desired_key()] = _fold_state(desired)[0]
        planner = self._attn if self._attn is not None else self._history
        index = planner.append(frame)
        if self._pending is not None and self._pending.done():
            self._wait_for_plan(raise_errors=True)
        if self._pending is not None or not self._schedule.due(index):
            return
        self._schedule.planned(index)
        grounding = {
            "index": index,
            "capture_time_ns": observation["capture_time_ns"],
            "actions_discarded": observation["actions_discarded"],
            "committed_count": int(np.asarray(observation["committed_actions"]).shape[1]),
            "robot_has_plan": np.asarray(observation["replaceable_actions"]).shape[1] > 0,
        }
        if self._frozen_arm is not None:
            for side in ("left", "right"):
                for key in (
                    f"observation.state.tcp_pose_{side}",
                    f"observation.state.gripper_width_{side}",
                ):
                    grounding[key] = np.array(observation[key])
        if self._attn is not None:
            job = (self._plan_and_emit_attn, self._attn.snapshot(index))
        else:
            job = (self._plan_and_emit, self._history.context())
        self._pending = self._inference_executor.submit(*job, grounding, self._emitter)
        if index == 0:
            # Prime the stream before intake continues, as single-frame serving
            # does; a client may close its stream after the first observation.
            self._wait_for_plan(raise_errors=True)

    def configure(self, request: ConfigureRequest) -> ObservationRequirements:
        """Load once, then set the task for subsequent episodes."""
        self._configured = False
        if self._model_id is not None and request.model_id != self._model_id:
            raise ValueError(
                f"server is pinned to model_id {self._model_id!r}, not"
                f" {request.model_id!r}"
            )
        task = request.task_instruction.strip()
        if not task:
            # Rejecting beats substituting: a made-up prompt makes a
            # language-conditioned model look like it works while it ignores
            # the task. ACT ignores the string, but a uniform contract keeps
            # the client honest across every baseline model.
            raise ValueError("task_instruction is required")
        if self.stop_event.is_set():
            raise RuntimeError(
                "model initialization failed; restart the server to retry"
            )
        if not self._loaded:
            started = time.monotonic()
            logger.info(
                "initializing model_id=%r checkpoint %s on %s",
                self._model_id,
                self._checkpoint,
                self._device,
            )
            try:
                self._load()
            except BaseException:
                self.stop_event.set()
                logger.exception("model initialization failed")
                raise
            self._loaded = True
            logger.info(
                "model initialization completed in %.3fs", time.monotonic() - started
            )
        self._task_instruction = task
        logger.info("configured: model_id=%r task=%r", request.model_id, task)
        requirements = ObservationRequirements()
        for _, slot, height, width in self._cameras:
            getattr(requirements, slot).CopyFrom(
                ps.RgbRequirement(width=width, height=height, format=ps.RGB_FORMAT_JPEG)
            )
        self._configured = True
        return requirements

    def activate(self) -> None:
        """Reset episode queues only after successful Configure."""
        if not self._configured:
            raise RuntimeError("Run requires successful Configure")
        self._policy.reset()
        if self._history is not None or self._attn is not None:
            self._wait_for_plan(raise_errors=False)
            (self._attn if self._attn is not None else self._history).reset()
            self._schedule.reset()
            self._plan_stats.clear()
            self._last_chunk = None
            self._emitter = self.emitter()

    def deactivate(self) -> None:
        """Join any in-flight history plan so none outlives its episode."""
        if self._history is not None or self._attn is not None:
            self._wait_for_plan(raise_errors=False)
            self._log_plan_stats()
            self._plan_stats.clear()
            self._last_chunk = None
            self._emitter = None

    def on_observation(self, observation: dict) -> None:
        if self._history is not None or self._attn is not None:
            self._on_history_observation(observation)
            return
        # The baseline models are markovian per chunk: frames the client is
        # not requesting a chunk for carry no state we need to accumulate.
        # Each chunk captures one measured pose before inference starts.
        if not observation["chunk_requested"]:
            return
        encoder = (
            encode_model
            if self._representation is None
            else self._representation.encode
        )
        batch: dict = {
            "observation.state": _fold_state(observation, encoder),
            "task": self._task_instruction,
        }
        for key, _, height, width in self._cameras:
            image = np.asarray(observation[key], dtype=np.float32)
            if image.shape != (1, 3, height, width):
                raise ValueError(
                    f"{key} arrived as {image.shape}, required (1, 3, {height}, {width})"
                )
            batch[key] = image
        prediction = self._predict(batch)
        decoded = (
            decode_model(prediction)
            if self._representation is None
            else self._representation.decode(prediction)
        )
        chunk = _emit_hygiene(decoded)
        wire = np.ascontiguousarray(chunk[:, :, WIRE_FROM_TRAIN])
        if self._frozen_arm is not None:
            wire = pin_frozen_arm(wire, observation, self._frozen_arm)
        self.emit_chunk(
            {"action": wire},
            observation["capture_time_ns"],
            discard_actions=observation["actions_discarded"],
        )


def main() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s"
    )
    parser = argparse.ArgumentParser(
        prog="python -m inference.serve", description=__doc__
    )
    parser.add_argument(
        "--checkpoint", type=Path, required=True, help="lerobot pretrained_model dir"
    )
    parser.add_argument("--model-id", default=None, help="expected Configure model_id")
    parser.add_argument("--host", default="[::]")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument(
        "--compile-model",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="override native policy compilation; default: enabled for Diffusion, PI0.5 and SmolVLA, otherwise checkpoint setting",
    )
    parser.add_argument(
        "--compile-mode",
        choices=(
            "default",
            "reduce-overhead",
            "max-autotune",
            "max-autotune-no-cudagraphs",
        ),
        default=None,
        help="native compile mode; default: max-autotune for PI0.5, default for Diffusion, otherwise checkpoint setting",
    )
    parser.add_argument(
        "--matmul-precision",
        choices=("high", "highest"),
        default=None,
        help="FP32 matmul precision; default: highest when compiled, otherwise policy setting",
    )
    args = parser.parse_args()

    checkpoint = args.checkpoint.resolve()
    if not (checkpoint / "config.json").exists():
        raise SystemExit(
            f"{checkpoint} is not a lerobot pretrained_model dir (no config.json)"
        )

    policy = BaselinePolicyServer(
        checkpoint,
        args.device,
        args.model_id,
        compile_model=args.compile_model,
        compile_mode=args.compile_mode,
        matmul_precision=args.matmul_precision,
    )
    try:
        serve(
            policy,
            verifier=None,
            tls=False,
            host=args.host,
            port=args.port,
            stop_event=policy.stop_event,
        )
        if policy.stop_event.is_set():
            raise SystemExit(1)
    finally:
        policy.close()


if __name__ == "__main__":
    main()
