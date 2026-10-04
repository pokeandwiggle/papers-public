"""OpenVLA-OFT policy with an L1 regression head (control) or ActionMap heatmap head.

Sequence layout (openvla-oft ``modeling_prismatic.py``):
``<s> [3 x 256 image patches] [proprio] In: What action should the robot take to
{task}?\\nOut: ▁ [chunk * 14 zeroed action tokens] </s>``. The head reads the
last hidden states one position before each action token (the training label
shift that OFT also uses at inference), i.e. from the ``▁`` token through the
second-to-last action token.
"""

import math
import shutil
from collections import deque
from pathlib import Path

import torch
import torch.nn as nn
import torchvision.transforms.v2.functional as TF
from huggingface_hub import snapshot_download
from lerobot.configs import PreTrainedConfig
from lerobot.policies.pretrained import PreTrainedPolicy
from peft import LoraConfig, inject_adapter_in_model
from safetensors import safe_open

from paw_baseline.layout import CAMERA_KEYS
from paw_baseline.oft.actions import (
    ACTION_DIM,
    DeltaNormalizer,
    ProprioNormalizer,
    decode_chunk,
    encode_chunk,
)
from paw_baseline.oft.configuration_paw_oft import PawOFTConfig
from paw_baseline.oft.heads import HeatmapActionHead, L1RegressionActionHead
from paw_baseline.oft.prismatic import (
    ACTION_PLACEHOLDER_ID,
    DINO_MEAN,
    DINO_STD,
    EMPTY_TOKEN_ID,
    IMAGE_SIZE,
    LLM_DIM,
    PAD_TOKEN_ID,
    SIGLIP_MEAN,
    SIGLIP_STD,
    STOP_TOKEN_ID,
    PrismaticVLA,
    ProprioProjector,
    load_openvla_weights,
)

TOKENIZER_FILES = (
    "tokenizer.json",
    "tokenizer.model",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "added_tokens.json",
)
TOKENIZER_DIR = "tokenizer"


def resolve_vla_snapshot(config: PawOFTConfig, weights: bool) -> Path:
    patterns = list(TOKENIZER_FILES)
    if weights:
        patterns += ["model.safetensors.index.json", "model-*.safetensors"]
    return Path(
        snapshot_download(config.vla_repo, revision=config.vla_revision, allow_patterns=patterns)
    )


def load_tokenizer(directory: Path):
    from transformers import LlamaTokenizer

    tokenizer = LlamaTokenizer.from_pretrained(str(directory))
    probe = tokenizer("In: What action should the robot take to x?\nOut:").input_ids
    if probe[:3] != [1, 512, 29901] or probe[-2:] != [3744, 29901]:
        raise RuntimeError(f"unexpected OpenVLA tokenizer behavior: {probe}")
    return tokenizer


class PawOFTPolicy(PreTrainedPolicy):
    config_class = PawOFTConfig
    name = "paw_oft"

    def __init__(
        self,
        config: PawOFTConfig,
        *,
        _checkpoint_dir: Path | None = None,
        **kwargs,
    ):
        super().__init__(config)
        config.validate_features()
        if config.oft_stats is None:
            raise ValueError("OFT requires delta-action/proprio statistics (oft_stats)")
        self.config = config
        fresh = _checkpoint_dir is None
        snapshot = resolve_vla_snapshot(config, weights=fresh) if fresh else None
        self._tokenizer_dir = snapshot if fresh else _checkpoint_dir / TOKENIZER_DIR
        self.tokenizer = load_tokenizer(self._tokenizer_dir)

        # Build frozen weights in BF16 like OpenVLA's torch_dtype loading; Llama's
        # rotary inverse frequencies are created explicitly in FP32 and stay so.
        # Allocate on the target accelerator to avoid a 15 GB host copy per process.
        build_device = (
            torch.device(config.device)
            if str(config.device).startswith("cuda") and torch.cuda.is_available()
            else torch.device("cpu")
        )
        default_dtype = torch.get_default_dtype()
        torch.set_default_dtype(torch.bfloat16)
        try:
            with build_device:
                self.vla = PrismaticVLA()
        finally:
            torch.set_default_dtype(default_dtype)
        if fresh:
            load_openvla_weights(self.vla, snapshot)
        self.vla.requires_grad_(False)
        # OpenVLA-OFT: LoRA on every linear layer of the VLA, rank 32, alpha min(r, 16).
        targets = sorted(
            name for name, module in self.vla.named_modules() if isinstance(module, nn.Linear)
        )
        inject_adapter_in_model(
            LoraConfig(
                r=config.lora_rank,
                lora_alpha=min(config.lora_rank, 16),
                lora_dropout=config.lora_dropout,
                target_modules=targets,
                init_lora_weights="gaussian",
            ),
            self.vla,
        )
        for name, param in self.vla.named_parameters():
            param.requires_grad = "lora_" in name
            if param.requires_grad:
                param.data = param.data.float()
        self.vla.set_gradient_checkpointing(config.gradient_checkpointing)

        with build_device:
            self._build_heads(config)
        self.to(build_device)
        self.reset()

    def _build_heads(self, config: PawOFTConfig) -> None:
        self.proprio_projector = ProprioProjector(LLM_DIM, config.robot_state_feature.shape[0])
        if config.action_head == "l1":
            self.action_head = L1RegressionActionHead(
                LLM_DIM, LLM_DIM, ACTION_DIM, config.chunk_size
            )
        else:
            self.action_head = HeatmapActionHead(
                LLM_DIM,
                LLM_DIM,
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
        self.delta_normalizer = DeltaNormalizer(config.oft_stats["action"])
        self.proprio_normalizer = ProprioNormalizer(config.oft_stats["proprio"])
        self.register_buffer(
            "dino_mean", torch.tensor(DINO_MEAN).view(1, 3, 1, 1), persistent=False
        )
        self.register_buffer("dino_std", torch.tensor(DINO_STD).view(1, 3, 1, 1), persistent=False)
        self.register_buffer(
            "siglip_mean", torch.tensor(SIGLIP_MEAN).view(1, 3, 1, 1), persistent=False
        )
        self.register_buffer(
            "siglip_std", torch.tensor(SIGLIP_STD).view(1, 3, 1, 1), persistent=False
        )

    # ----------------------------------------------------------------- LeRobot API
    def get_optim_params(self):
        return [param for param in self.parameters() if param.requires_grad]

    def reset(self):
        self._action_queue = deque([], maxlen=self.config.n_action_steps)

    def forward(self, batch: dict) -> tuple[torch.Tensor, dict]:
        state = batch["observation.state"]
        deltas = encode_chunk(state, batch["action"])
        target = self.delta_normalizer.normalize(deltas)
        hidden = self._action_hidden_states(batch)
        with self._autocast(state.device):
            predicted, loss = self.action_head.predict_action_with_loss(hidden, target)
        with torch.no_grad():
            error = (predicted.float() - target).abs()
            metrics = {
                "curr_action_l1": error[:, 0].mean().item(),
                "next_actions_l1": error[:, 1:].mean().item(),
            }
        return loss, metrics

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict, **kwargs) -> torch.Tensor:
        state = batch["observation.state"]
        hidden = self._action_hidden_states(batch)
        with self._autocast(state.device):
            normalized = self.action_head.predict_action(hidden).float()
        deltas = self.delta_normalizer.unnormalize(normalized)
        return decode_chunk(state, deltas)

    @torch.no_grad()
    def select_action(self, batch: dict, **kwargs) -> torch.Tensor:
        if not self._action_queue:
            chunk = self.predict_action_chunk(batch)[:, : self.config.n_action_steps]
            self._action_queue.extend(chunk.transpose(0, 1))
        return self._action_queue.popleft()

    # ------------------------------------------------------------------- internals
    @staticmethod
    def _autocast(device: torch.device):
        return torch.autocast(device_type=device.type, dtype=torch.bfloat16)

    def _preprocess_images(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """OFT augmentation (train) or center crop (eval), then Prismatic normalization.

        Train: random square crop of 90% area, resize to 224, then brightness 0.2,
        contrast [0.8, 1.2], saturation [0.8, 1.2] and hue 0.05 in that order.
        Eval: the same-size center crop, resized to 224.
        """
        images = images.float()
        n, _, height, width = images.shape
        side_h = int(round(height * math.sqrt(self.config.crop_scale)))
        side_w = int(round(width * math.sqrt(self.config.crop_scale)))
        augment = self.training and self.config.image_aug
        outputs = []
        for i in range(n):
            if augment:
                top = int(torch.randint(0, height - side_h + 1, ()).item())
                left = int(torch.randint(0, width - side_w + 1, ()).item())
            else:
                top, left = (height - side_h) // 2, (width - side_w) // 2
            image = images[i, :, top : top + side_h, left : left + side_w]
            image = TF.resize(
                image, [IMAGE_SIZE, IMAGE_SIZE], interpolation=TF.InterpolationMode.BICUBIC,
                antialias=True,
            )
            if augment:
                image = image + (torch.rand((), device=image.device) * 0.4 - 0.2)
                image = TF.adjust_contrast(image.clamp(0, 1), float(torch.empty(()).uniform_(0.8, 1.2)))
                image = TF.adjust_saturation(image, float(torch.empty(()).uniform_(0.8, 1.2)))
                image = TF.adjust_hue(image, float(torch.empty(()).uniform_(-0.05, 0.05)))
            outputs.append(image.clamp(0, 1))
        pixels = torch.stack(outputs)
        dino = (pixels - self.dino_mean) / self.dino_std
        siglip = (pixels - self.siglip_mean) / self.siglip_std
        return dino.to(torch.bfloat16), siglip.to(torch.bfloat16)

    def _prompt_ids(self, tasks) -> list[list[int]]:
        if isinstance(tasks, str):
            tasks = [tasks]
        ids = []
        for task in tasks:
            prompt = f"In: What action should the robot take to {task.strip().lower()}?\nOut:"
            tokens = self.tokenizer(prompt, add_special_tokens=True).input_ids
            if tokens[-1] != EMPTY_TOKEN_ID:
                tokens = tokens + [EMPTY_TOKEN_ID]
            ids.append(tokens)
        return ids

    def _action_hidden_states(self, batch: dict) -> torch.Tensor:
        """Hidden states (B, chunk * 14, 4096) feeding the action head."""
        state = batch["observation.state"]
        device = state.device
        batch_size = state.shape[0]
        tasks = batch["task"]
        if isinstance(tasks, str):
            tasks = [tasks] * batch_size
        if len(tasks) != batch_size:
            raise ValueError("one task instruction is required per sample")
        num_actions = self.config.chunk_size * ACTION_DIM

        cameras = torch.stack([batch[key] for key in CAMERA_KEYS], dim=1)
        dino, siglip = self._preprocess_images(cameras.flatten(0, 1))
        with self._autocast(device):
            patches = self.vla.embed_images(dino, siglip)
            patches = patches.reshape(batch_size, len(CAMERA_KEYS) * patches.shape[1], -1)
            proprio = self.proprio_projector(self.proprio_normalizer(state))
            prefix = torch.cat([patches, proprio[:, None]], dim=1)

            prompts = self._prompt_ids(tasks)
            lengths = [len(p) for p in prompts]
            text_len = max(lengths) + num_actions + 1
            ids = torch.full((batch_size, text_len), PAD_TOKEN_ID, dtype=torch.long)
            text_mask = torch.zeros((batch_size, text_len), dtype=torch.bool)
            action_mask = torch.zeros((batch_size, text_len), dtype=torch.bool)
            for b, prompt in enumerate(prompts):
                n = len(prompt)
                ids[b, :n] = torch.tensor(prompt)
                ids[b, n : n + num_actions] = ACTION_PLACEHOLDER_ID
                ids[b, n + num_actions] = STOP_TOKEN_ID
                text_mask[b, : n + num_actions + 1] = True
                action_mask[b, n : n + num_actions] = True
            ids, text_mask, action_mask = ids.to(device), text_mask.to(device), action_mask.to(device)
            embeddings = self.vla.llm.get_input_embeddings()(ids)
            embeddings = embeddings * ~action_mask[..., None]
            sequence = torch.cat([embeddings[:, :1], prefix.to(embeddings.dtype), embeddings[:, 1:]], 1)
            key_mask = torch.cat(
                [text_mask[:, :1], text_mask.new_ones(batch_size, prefix.shape[1]), text_mask[:, 1:]], 1
            )
            hidden = self.vla.bidirectional_hidden_states(sequence, key_mask)
        # Text token j sits at sequence position prefix + j; read positions L-1 .. L+K-2.
        start = torch.tensor(lengths, device=device) - 1 + prefix.shape[1]
        index = start[:, None] + torch.arange(num_actions, device=device)
        return hidden.gather(1, index[..., None].expand(-1, -1, hidden.shape[-1]))

    # --------------------------------------------------------------- save / load
    def _save_pretrained(self, save_directory: Path, state_dict=None) -> None:
        super()._save_pretrained(save_directory, state_dict)
        target = Path(save_directory) / TOKENIZER_DIR
        target.mkdir(parents=True, exist_ok=True)
        for name in TOKENIZER_FILES:
            source = Path(self._tokenizer_dir) / name
            if source.exists():
                shutil.copyfile(source, target / name)

    @classmethod
    def from_pretrained(
        cls, pretrained_name_or_path, *, config=None, revision=None, strict=True, **kwargs
    ):
        checkpoint = Path(pretrained_name_or_path)
        if not checkpoint.is_dir():
            raise ValueError("OFT checkpoints load from a local pretrained_model directory")
        if config is None:
            config = PreTrainedConfig.from_pretrained(checkpoint)
        if not isinstance(config, PawOFTConfig):
            raise TypeError("OFT loading requires a PawOFTConfig")
        policy = cls(config, _checkpoint_dir=checkpoint)
        # Stream tensors into the existing parameters to bound host memory.
        target = policy.state_dict()
        seen = set()
        with torch.no_grad(), safe_open(str(checkpoint / "model.safetensors"), "pt") as handle:
            for key in handle.keys():
                if key not in target:
                    raise RuntimeError(f"OFT checkpoint has unexpected tensor {key}")
                tensor = handle.get_tensor(key)
                if tensor.shape != target[key].shape or tensor.dtype != target[key].dtype:
                    raise RuntimeError(f"OFT checkpoint tensor {key} has a different shape/dtype")
                target[key].copy_(tensor)
                seen.add(key)
        missing = sorted(set(target) - seen)
        if missing:
            raise RuntimeError(f"OFT checkpoint lacks {len(missing)} tensors, e.g. {missing[:5]}")
        policy.to(config.device)
        policy.eval()
        return policy
