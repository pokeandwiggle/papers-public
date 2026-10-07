"""Serve a Patch Policy checkpoint with its visual observation history.

Every 30 Hz observation's camera frames enter a ring buffer, so the history
frames ``obs_stride`` steps apart that the policy was trained on are always
available. Planning is server-paced rather than prefetch-paced: the server
plans from the newest frame on the first observation of a Run stream and then
every ``replan_every`` observations, emitting the full predicted chunk grounded
at that frame's capture time. This approximates the paper's per-step
closed-loop replanning; the robot's chunk blending plays the role of the
paper's temporal ensembling. ``chunk_requested`` is advisory and not needed.
"""

import json
import logging
import time
from collections import deque
from pathlib import Path

import numpy as np
import torch
from paw_baseline.layout import WIRE_FROM_TRAIN, pin_frozen_arm
from paw_baseline.representation import MODEL_DIM, decode_model, encode_model
from paw_baseline.serving_metadata import load_serving_metadata
from paw_baseline.variant import load_policy_config
from paw_policy_toolkit import policy_service as ps
from paw_policy_toolkit.policy_service import ConfigureRequest, ObservationRequirements
from paw_policy_toolkit.server import PolicyServer

from train.patch_policy.modeling_patch_vqbet import PatchVQBeTPolicy, history_indices

from inference.serve import _emit_hygiene, _fold_state

logger = logging.getLogger("paw_baseline.patch_serve")

class PatchPolicyServer(PolicyServer):
    """One checkpoint, one Run stream at a time; inference runs on the pump thread."""

    def __init__(self, checkpoint: Path, device: str, model_id: str | None = None) -> None:
        super().__init__()
        self._checkpoint = checkpoint
        self._device = device
        self._model_id = model_id
        self._loaded = False
        self._configured = False
        from threading import Event

        self.stop_event = Event()

    def _load(self) -> None:
        checkpoint = self._checkpoint
        config = json.loads((checkpoint / "config.json").read_text())
        if config["type"] != "patch_vqbet":
            raise RuntimeError(f"expected a patch_vqbet checkpoint, got {config['type']!r}")
        self._frozen_arm = load_serving_metadata(checkpoint)["frozen_arm"]
        native_config = load_policy_config(checkpoint)
        native_config.device = self._device
        if config["input_features"]["observation.state"]["shape"] != [MODEL_DIM]:
            raise RuntimeError("checkpoint expects a different observation.state layout")
        if config["output_features"]["action"]["shape"] != [MODEL_DIM]:
            raise RuntimeError("checkpoint emits a different action layout")
        self._cameras = tuple(
            (key, key.removeprefix("observation.images."), ft["shape"][1], ft["shape"][2])
            for key, ft in config["input_features"].items()
            if ft["type"] == "VISUAL"
        )
        from lerobot.policies import make_pre_post_processors

        self._policy = PatchVQBeTPolicy.from_pretrained(checkpoint, config=native_config)
        self._policy.to(self._device).eval()
        cfg = self._policy.config
        self._offsets = cfg.history_offsets
        self._replan_every = cfg.replan_every
        self._n_action_steps = cfg.n_action_steps
        self._span = (cfg.n_obs_steps - 1) * cfg.obs_stride + 1
        override = {"device": self._device}
        self._preprocess, self._postprocess = make_pre_post_processors(
            cfg,
            pretrained_path=checkpoint,
            preprocessor_overrides={"device_processor": override},
            postprocessor_overrides={"device_processor": override},
        )
        self._task_instruction = ""
        self._reset_stream()
        self._warmup()
        logger.info(
            "loaded patch_vqbet (%s features) %s on %s: cameras %s, history offsets %s, "
            "chunk %d, replan every %d observations, frozen_arm %s",
            cfg.visual_feature,
            checkpoint,
            self._device,
            [slot for _, slot, _, _ in self._cameras],
            self._offsets,
            self._n_action_steps,
            self._replan_every,
            self._frozen_arm,
        )

    def _reset_stream(self) -> None:
        self._frames = {key: deque(maxlen=self._span) for key, _, _, _ in self._cameras}
        self._index = 0

    def _warmup(self) -> None:
        state = np.zeros((1, 16), dtype=np.float32)
        state[0, 6] = state[0, 14] = 1.0
        for key, _, height, width in self._cameras:
            self._frames[key].append(torch.zeros(3, height, width))
        self._predict(encode_model(state), "warmup")
        self._reset_stream()

    def _predict(self, state_model: np.ndarray, task: str) -> np.ndarray:
        """Normalized history -> (1, n_action_steps, 20) train-order model actions."""
        batch: dict = {"observation.state": torch.from_numpy(state_model), "task": task}
        for key, _, _, _ in self._cameras:
            frames = list(self._frames[key])
            picks = history_indices(len(frames), self._offsets)
            batch[key] = torch.stack([frames[i] for i in picks]).unsqueeze(0)
        with torch.inference_mode():
            processed = self._preprocess(batch)
            chunk = self._policy.predict_action_chunk(processed)
            steps = [self._postprocess(chunk[:, i]) for i in range(chunk.shape[1])]
        return torch.stack(steps, dim=1).float().cpu().numpy()

    def configure(self, request: ConfigureRequest) -> ObservationRequirements:
        self._configured = False
        if self._model_id is not None and request.model_id != self._model_id:
            raise ValueError(f"server is pinned to model_id {self._model_id!r}, not {request.model_id!r}")
        task = request.task_instruction.strip()
        if not task:
            raise ValueError("task_instruction is required")
        if self.stop_event.is_set():
            raise RuntimeError("model initialization failed; restart the server to retry")
        if not self._loaded:
            started = time.monotonic()
            try:
                self._load()
            except BaseException:
                self.stop_event.set()
                logger.exception("model initialization failed")
                raise
            self._loaded = True
            logger.info("model initialization completed in %.3fs", time.monotonic() - started)
        self._task_instruction = task
        requirements = ObservationRequirements()
        for _, slot, height, width in self._cameras:
            getattr(requirements, slot).CopyFrom(
                ps.RgbRequirement(width=width, height=height, format=ps.RGB_FORMAT_JPEG)
            )
        self._configured = True
        return requirements

    def activate(self) -> None:
        if not self._configured:
            raise RuntimeError("Run requires successful Configure")
        self._reset_stream()

    def close(self) -> None:
        pass

    def on_observation(self, observation: dict) -> None:
        for key, _, height, width in self._cameras:
            image = np.asarray(observation[key], dtype=np.float32)
            if image.shape != (1, 3, height, width):
                raise ValueError(f"{key} arrived as {image.shape}, required (1, 3, {height}, {width})")
            self._frames[key].append(torch.from_numpy(image[0]))
        index = self._index
        self._index += 1
        if index % self._replan_every:
            return
        prediction = self._predict(_fold_state(observation, encode_model), self._task_instruction)
        chunk = _emit_hygiene(decode_model(prediction))
        wire = np.ascontiguousarray(chunk[:, :, WIRE_FROM_TRAIN])
        if self._frozen_arm is not None:
            wire = pin_frozen_arm(wire, observation, self._frozen_arm)
        self.emit_chunk(
            {"action": wire},
            observation["capture_time_ns"],
            discard_actions=observation["actions_discarded"],
        )
