"""Configure OpenVLA-OFT with its L1 regression head or the ActionMap heatmap head.

The two heads share every other setting: OpenVLA-7B with rank-32 LoRA on all
linear layers, parallel decoding of ``chunk_size * 14`` action tokens with
bidirectional attention, three cameras, a proprio token, OFT image
augmentation, and AdamW at a constant 5e-4 learning rate.
"""

import json
from dataclasses import dataclass, field
from pathlib import Path

from lerobot.configs import NormalizationMode, PreTrainedConfig
from lerobot.optim import AdamWConfig

from paw_baseline.layout import CAMERA_KEYS

VLA_REPO = "openvla/openvla-7b"
VLA_REVISION = "47a0ec7fc4ec123775a391911046cf33cf9ed83f"
ACTION_HEADS = ("l1", "heatmap")


@PreTrainedConfig.register_subclass("paw_oft")
@dataclass
class PawOFTConfig(PreTrainedConfig):
    action_head: str = "l1"
    n_obs_steps: int = 1
    chunk_size: int = 32
    n_action_steps: int = 32

    vla_repo: str = VLA_REPO
    vla_revision: str = VLA_REVISION
    lora_rank: int = 32
    lora_dropout: float = 0.0
    gradient_checkpointing: bool = True
    # Consumed by the training hook: micro-batches per optimizer step.
    grad_accum_steps: int = 1
    # Accelerate drives BF16 autocast from this field; frozen weights are stored in BF16.
    dtype: str = "bfloat16"

    image_aug: bool = True
    crop_scale: float = 0.9

    heatmap_trans_grid: list[int] = field(default_factory=lambda: [48, 48, 48])
    heatmap_rot_grid: list[int] = field(default_factory=lambda: [16, 16, 16])
    heatmap_trans_sigma: float = 0.1
    heatmap_rot_sigma: float = 0.1
    heatmap_grid_pad: float = 0.2
    heatmap_decode_mode: str = "topk"
    heatmap_decode_top_k: int = 10
    heatmap_decode_temperature: float = 1.0

    # Training input only: statistics written by paw_baseline.oft.prepare.
    stats_path: str | None = None
    # Delta-action and proprio quantile bounds, saved with the checkpoint.
    oft_stats: dict | None = None

    optimizer_lr: float = 5e-4
    optimizer_weight_decay: float = 0.01

    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "VISUAL": NormalizationMode.IDENTITY,
            "STATE": NormalizationMode.IDENTITY,
            "ACTION": NormalizationMode.IDENTITY,
        }
    )

    def __post_init__(self):
        super().__post_init__()
        if self.action_head not in ACTION_HEADS:
            raise ValueError(f"action_head must be one of {ACTION_HEADS}")
        if self.n_obs_steps != 1:
            raise ValueError("OpenVLA-OFT uses one observation")
        if not 1 <= self.n_action_steps <= self.chunk_size:
            raise ValueError("n_action_steps must be within the chunk")
        if self.grad_accum_steps < 1:
            raise ValueError("grad_accum_steps must be positive")
        if self.oft_stats is None and self.stats_path:
            self.oft_stats = json.loads(Path(self.stats_path).read_text())["stats"]

    @property
    def observation_delta_indices(self) -> None:
        return None

    @property
    def action_delta_indices(self) -> list[int]:
        return list(range(self.chunk_size))

    @property
    def reward_delta_indices(self) -> None:
        return None

    def get_optimizer_preset(self) -> AdamWConfig:
        # OpenVLA-OFT: torch AdamW defaults, constant LR, no gradient clipping.
        return AdamWConfig(
            lr=self.optimizer_lr,
            weight_decay=self.optimizer_weight_decay,
            grad_clip_norm=0.0,
        )

    def get_scheduler_preset(self) -> None:
        return None

    def validate_features(self) -> None:
        if tuple(self.image_features) != CAMERA_KEYS:
            raise ValueError("OFT requires cameras in static, left wrist, right wrist order")
        if self.robot_state_feature is None or self.robot_state_feature.shape != (20,):
            raise ValueError("OFT requires the 20D TCP state")
        if self.action_feature is None or self.action_feature.shape != (20,):
            raise ValueError("OFT requires 20D TCP actions")
