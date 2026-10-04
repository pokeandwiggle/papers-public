"""Configure OpenVLA-OFT with an L1 regression (control) or ActionMap heatmap head."""

from dataclasses import dataclass, field

from lerobot.configs import NormalizationMode, PreTrainedConfig
from lerobot.optim import AdamWConfig

from paw_baseline.layout import CAMERA_KEYS

BASE_REPO = "openvla/openvla-7b"
BASE_REVISION = "47a0ec7fc4ec123775a391911046cf33cf9ed83f"
ACTION_HEADS = ("l1", "heatmap")


@PreTrainedConfig.register_subclass("paw_oft")
@dataclass
class PawOFTConfig(PreTrainedConfig):
    """OpenVLA-OFT parallel decoding over per-step delta actions.

    Normalization lives in the policy: LeRobot processors pass state, actions and
    images through unchanged, and ``norm_stats`` (fitted on training episodes by
    ``paw_baseline.oft.prepare``) travels in ``config.json`` to serving.
    """

    n_obs_steps: int = 1
    chunk_size: int = 16
    n_action_steps: int = 16

    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "VISUAL": NormalizationMode.IDENTITY,
            "STATE": NormalizationMode.IDENTITY,
            "ACTION": NormalizationMode.IDENTITY,
        }
    )

    action_head: str = "l1"
    base_repo: str = BASE_REPO
    base_revision: str = BASE_REVISION

    # OpenVLA-OFT LoRA recipe: rank 32, alpha min(rank, 16), no dropout, all linear layers.
    lora_rank: int = 32
    lora_alpha: int = 16
    lora_dropout: float = 0.0

    # OFT image pipeline: 224px, random 90%-area crops and color jitter in training,
    # the matching 90%-area center crop at inference.
    image_aug: bool = True
    crop_scale: float = 0.9
    gradient_checkpointing: bool = True
    # Drives LeRobot's Accelerate autocast; OFT trains and infers under BF16 autocast.
    dtype: str = "bfloat16"

    # ActionMap real-robot recipe: 48^3 translation and 16^3 rotation grids, sigma 0.1,
    # grid over [q01 - 2 sigma, q99 + 2 sigma], top-10 soft-argmax at temperature 1.
    heatmap_trans_grid: list[int] = field(default_factory=lambda: [48, 48, 48])
    heatmap_rot_grid: list[int] = field(default_factory=lambda: [16, 16, 16])
    heatmap_trans_sigma: float = 0.1
    heatmap_rot_sigma: float = 0.1
    heatmap_grid_pad: float = 0.2
    heatmap_decode_mode: str = "topk"
    heatmap_decode_top_k: int = 10
    heatmap_decode_temperature: float = 1.0

    # OFT optimizer: AdamW (torch defaults), constant 5e-4, no warmup, no clipping.
    optimizer_lr: float = 5e-4
    optimizer_weight_decay: float = 0.01
    optimizer_grad_clip_norm: float = 0.0

    # Written by training from paw_baseline.oft.prepare; serving reads norm_stats.
    stats_path: str | None = None
    norm_stats: dict | None = None

    def __post_init__(self):
        super().__post_init__()
        if self.action_head not in ACTION_HEADS:
            raise ValueError(f"action_head must be one of {ACTION_HEADS}")
        if self.n_obs_steps != 1:
            raise ValueError("OFT uses one observation")
        if not 0 < self.n_action_steps <= self.chunk_size:
            raise ValueError("n_action_steps must be in [1, chunk_size]")
        if len(self.heatmap_trans_grid) != 3 or len(self.heatmap_rot_grid) != 3:
            raise ValueError("heatmap grids need three axes")

    def get_optimizer_preset(self) -> AdamWConfig:
        return AdamWConfig(
            lr=self.optimizer_lr,
            weight_decay=self.optimizer_weight_decay,
            grad_clip_norm=self.optimizer_grad_clip_norm,
        )

    def get_scheduler_preset(self) -> None:
        return None

    def validate_features(self) -> None:
        if tuple(self.image_features) != CAMERA_KEYS:
            raise ValueError("OFT expects cameras in static, left wrist, right wrist order")
        if self.robot_state_feature is None or self.robot_state_feature.shape != (20,):
            raise ValueError("OFT expects the 20D TCP model state")
        if self.action_feature is None or self.action_feature.shape != (20,):
            raise ValueError("OFT expects 20D absolute TCP model actions")

    @property
    def observation_delta_indices(self) -> None:
        return None

    @property
    def action_delta_indices(self) -> list:
        return list(range(self.chunk_size))

    @property
    def reward_delta_indices(self) -> None:
        return None
