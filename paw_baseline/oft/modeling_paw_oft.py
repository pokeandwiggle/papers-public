"""OpenVLA-OFT policy with an L1 regression head (control) or ActionMap heatmap head.

Training follows OpenVLA-OFT ``vla-scripts/finetune.py``: LoRA on every linear layer
of the frozen BF16 OpenVLA-7B, parallel decoding of ``chunk * 14`` empty action
tokens with bidirectional attention, a proprio token, BF16 heads, and the head's
own loss on normalized per-step delta actions. Only the action head differs
between the two arms of the comparison.
"""

import json
import shutil
from collections import deque
from pathlib import Path

import torch
import torch.nn as nn
from huggingface_hub import hf_hub_download
from lerobot.configs import PreTrainedConfig
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.utils.constants import ACTION, OBS_STATE
from peft import LoraConfig, inject_adapter_in_model
from safetensors.torch import load_file
from tokenizers import Tokenizer

from paw_baseline.layout import CAMERA_KEYS
from paw_baseline.oft.actions import (
    ACTION_DIM,
    GRIPPER_DIMS,
    BoundsNormalizer,
    decode_deltas,
    encode_deltas,
)
from paw_baseline.oft.configuration_paw_oft import PawOFTConfig
from paw_baseline.oft.heads import HeatmapActionHead, L1RegressionActionHead
from paw_baseline.oft.images import preprocess_images
from paw_baseline.oft.prismatic import (
    BOS_ID,
    EMPTY_ID,
    OpenVLABackbone,
    ProprioProjector,
    load_openvla_backbone,
)

TOKENIZER_FILE = "openvla_tokenizer.json"
PROPRIO_DIM = 20


def openvla_prompt(task: str) -> str:
    """OpenVLA ``PurePromptBuilder`` query; OFT lowercases the instruction."""
    message = f"What action should the robot take to {task.lower()}?".strip()
    return f"In: {message}\nOut:"


def _device(name: str | None) -> torch.device:
    device = torch.device(name or "cpu")
    if device.type == "cuda" and device.index is None:
        device = torch.device("cuda", torch.cuda.current_device())
    return device


class PawOFTPolicy(PreTrainedPolicy):
    """OpenVLA-OFT over 20D absolute TCP observations, emitting 20D absolute targets."""

    config_class = PawOFTConfig
    name = "paw_oft"

    def __init__(
        self,
        config: PawOFTConfig,
        dataset_stats=None,
        dataset_meta=None,
        *,
        _checkpoint_dir: Path | None = None,
        **kwargs,
    ):
        super().__init__(config)
        config.validate_features()
        if config.norm_stats is None:
            if not config.stats_path:
                raise ValueError("OFT requires statistics from paw_baseline.oft.prepare")
            config.norm_stats = json.loads(Path(config.stats_path).read_text())
        stats = config.norm_stats
        if stats["chunk_size"] != config.chunk_size:
            raise ValueError("OFT statistics were fitted for a different chunk size")
        self._action_normalizer = BoundsNormalizer(
            stats["action_q01"], stats["action_q99"], raw_dims=GRIPPER_DIMS
        )
        self._state_normalizer = BoundsNormalizer(stats["state_q01"], stats["state_q99"])
        self.register_buffer(
            "gripper_open_width", torch.tensor(stats["gripper_open_width"]), persistent=False
        )
        self.register_buffer(
            "gripper_close_width", torch.tensor(stats["gripper_close_width"]), persistent=False
        )
        device = _device(config.device)
        tokenizer_path = (
            _checkpoint_dir / TOKENIZER_FILE
            if _checkpoint_dir is not None
            else hf_hub_download(config.base_repo, "tokenizer.json", revision=config.base_revision)
        )
        self._tokenizer_path = Path(tokenizer_path)
        self._tokenizer = Tokenizer.from_file(str(tokenizer_path))
        self._prompt_cache: dict[str, torch.Tensor] = {}
        num_tokens = config.chunk_size * ACTION_DIM

        if _checkpoint_dir is not None:
            with torch.device("meta"):
                self.vla = OpenVLABackbone()
                self._build_trainable(num_tokens)
        else:
            self.vla = load_openvla_backbone(config.base_repo, config.base_revision, device)
            with torch.device(device):
                self._build_trainable(num_tokens)
            for parameter in self.action_head.parameters():
                parameter.data = parameter.data.to(torch.bfloat16)
        self._inject_lora()
        if config.gradient_checkpointing:
            self.vla.enable_gradient_checkpointing()
        self.reset()

    def _build_trainable(self, num_tokens: int) -> None:
        config = self.config
        llm_dim = 4096
        self.proprio_projector = ProprioProjector(llm_dim, PROPRIO_DIM)
        if config.action_head == "l1":
            self.action_head = L1RegressionActionHead(
                llm_dim, llm_dim, ACTION_DIM, config.chunk_size
            )
        else:
            self.action_head = HeatmapActionHead(
                llm_dim,
                llm_dim,
                ACTION_DIM,
                config.chunk_size,
                trans_grid=tuple(config.heatmap_trans_grid),
                rot_grid=tuple(config.heatmap_rot_grid),
                trans_sigma=config.heatmap_trans_sigma,
                rot_sigma=config.heatmap_rot_sigma,
                grid_pad=config.heatmap_grid_pad,
                decode_mode=config.heatmap_decode_mode,
                decode_top_k=config.heatmap_decode_top_k,
                decode_temperature=config.heatmap_decode_temperature,
            )
        self._num_action_tokens = num_tokens

    def _inject_lora(self) -> None:
        """OFT ``LoraConfig(target_modules="all-linear", init_lora_weights="gaussian")``."""
        for parameter in self.vla.parameters():
            parameter.requires_grad_(False)
        targets = sorted(
            name for name, module in self.vla.named_modules() if isinstance(module, nn.Linear)
        )
        lora = LoraConfig(
            r=self.config.lora_rank,
            lora_alpha=self.config.lora_alpha,
            lora_dropout=self.config.lora_dropout,
            target_modules=targets,
            init_lora_weights="gaussian",
        )
        inject_adapter_in_model(lora, self.vla)
        for name, parameter in self.vla.named_parameters():
            is_lora = "lora_" in name
            parameter.requires_grad_(is_lora)
            if is_lora and parameter.device.type != "meta":
                # peft 0.11 (OFT) kept adapters in the base layer's BF16 dtype.
                parameter.data = parameter.data.to(torch.bfloat16)

    @classmethod
    def from_pretrained(
        cls, pretrained_name_or_path, *, config=None, revision=None, strict=True, **kwargs
    ):
        path = Path(pretrained_name_or_path)
        if not path.is_dir():
            raise ValueError("OFT checkpoints load from a local pretrained_model directory")
        if config is None:
            config = PreTrainedConfig.from_pretrained(path)
        if not isinstance(config, PawOFTConfig):
            raise TypeError("OFT loading requires PawOFTConfig")
        policy = cls(config, _checkpoint_dir=path)
        device = _device(config.device)
        state = load_file(path / "model.safetensors", device=str(device))
        policy.load_state_dict(state, strict=True, assign=True)
        policy.vla.reset_rotary(device)
        policy.to(device)
        policy.eval()
        return policy

    def _save_pretrained(self, save_directory: Path, state_dict=None) -> None:
        super()._save_pretrained(save_directory, state_dict=state_dict)
        shutil.copyfile(self._tokenizer_path, Path(save_directory) / TOKENIZER_FILE)

    def get_optim_params(self):
        return [p for p in self.parameters() if p.requires_grad]

    def reset(self):
        self._queue: deque = deque(maxlen=self.config.n_action_steps)

    def _prompt_ids(self, task: str, device) -> torch.Tensor:
        if task not in self._prompt_cache:
            ids = self._tokenizer.encode(openvla_prompt(task)).ids
            if not ids or ids[0] != BOS_ID:
                ids = [BOS_ID, *ids]
            if ids[-1] != EMPTY_ID:
                ids.append(EMPTY_ID)
            self._prompt_cache[task] = torch.tensor(ids, dtype=torch.long)
        return self._prompt_cache[task].to(device)

    def _action_hidden(self, batch: dict, *, train: bool) -> torch.Tensor:
        images = torch.stack([batch[key] for key in CAMERA_KEYS], dim=1)
        pixels = preprocess_images(images, train=train, crop_scale=self.config.crop_scale)
        state = batch[OBS_STATE].float()
        tasks = batch["task"]
        if isinstance(tasks, str):
            tasks = [tasks] * state.shape[0]
        patches = self.vla.patch_embeddings(pixels.to(torch.bfloat16))
        proprio = self.proprio_projector(self._state_normalizer.normalize(state))
        patches = torch.cat((patches, proprio[:, None].to(patches.dtype)), dim=1)
        prompts = [self._prompt_ids(task, patches.device) for task in tasks]
        return self.vla(patches, prompts, self._num_action_tokens)

    def forward(self, batch: dict, reduction: str = "mean"):
        if reduction != "mean":
            raise ValueError("OFT supports only mean-reduced losses")
        hidden = self._action_hidden(batch, train=self.training and self.config.image_aug)
        with torch.no_grad():
            deltas = encode_deltas(batch[OBS_STATE].double(), batch[ACTION].double())
            target = self._action_normalizer.normalize(deltas.float())
        predicted, loss, metrics = self.action_head.predict_action_with_loss(hidden, target)
        with torch.no_grad():
            decoded = self.action_head.decode(predicted, GRIPPER_DIMS)
            metrics["curr_action_l1"] = (decoded[:, 0] - target[:, 0]).abs().mean()
            metrics["next_actions_l1"] = (decoded[:, 1:] - target[:, 1:]).abs().mean()
        return loss, {key: float(value) for key, value in metrics.items()}

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict, **kwargs) -> torch.Tensor:
        """Absolute 20D targets ``(B, chunk, 20)`` integrated from the live TCP state."""
        device_type = batch[OBS_STATE].device.type
        with torch.autocast(device_type, dtype=torch.bfloat16, enabled=device_type == "cuda"):
            hidden = self._action_hidden(batch, train=False)
            predicted = self.action_head.predict_action(hidden)
        deltas = self._action_normalizer.unnormalize(
            self.action_head.decode(predicted, GRIPPER_DIMS).float()
        )
        absolute = decode_deltas(
            batch[OBS_STATE].double(),
            deltas.double(),
            self.gripper_open_width.double(),
            self.gripper_close_width.double(),
        )
        return absolute.float()

    @torch.no_grad()
    def select_action(self, batch: dict, **kwargs) -> torch.Tensor:
        if not self._queue:
            chunk = self.predict_action_chunk(batch)[:, : self.config.n_action_steps]
            self._queue.extend(chunk.transpose(0, 1))
        return self._queue.popleft()
